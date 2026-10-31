//! Regression tests: degenerate float constraints, unconverged values,
//! silent free variables, Schur-size guard.

use ncalg::ncsos::Relaxation;
use ncalg::relations;
use ncalg::sdp::{Affine, SdpModel};
use ncalg::solver::{SolverOptions, Status};
use ncalg::{Complex, GroebnerBasis, GroebnerOptions, Polynomial};

type C = Complex<f64>;

/// A single qubit in the pure state fixed by `n·σ` with `n = (sin θ, 0, cos θ)`
/// and `θ` tiny: the vector constraints `L(a (1 − n·σ)) = 0` have coefficients
/// spanning eight orders of magnitude, and the state has `⟨Z⟩ = cos θ`.
fn nearly_degenerate_relaxation(theta: f64) -> (Relaxation<'static, C>, Polynomial<C>) {
    let (alg, rels) = relations::pauli::<C>(1);
    let gb: &'static GroebnerBasis<C> = Box::leak(Box::new(GroebnerBasis::compute(
        rels,
        &GroebnerOptions::default(),
    )));
    let red = gb.reducer();
    let x: Polynomial<C> = alg.parse("X0").unwrap();
    let z: Polynomial<C> = alg.parse("Z0").unwrap();
    let one: Polynomial<C> = alg.one();
    let g = &one - &(&x * C::new(theta.sin(), 0.0) + &z * C::new(theta.cos(), 0.0));
    let mut rel = Relaxation::level(red, 1).minimize(&z);
    for a in red.normal_words_up_to(1) {
        let am: Polynomial<C> = alg.monomial(a.clone());
        rel = rel.moment_equality(&(&am * &g), C::new(0.0, 0.0));
        rel = rel.moment_equality(&(&g * &am), C::new(0.0, 0.0));
    }
    (rel, z)
}

#[test]
fn degenerate_float_constraints_build_and_solve() {
    for theta in [4e-8, 1.5e-9, 1e-12, 0.3] {
        let (rel, z) = nearly_degenerate_relaxation(theta);
        let sdp = rel
            .try_build()
            .unwrap_or_else(|e| panic!("theta {}: {}", theta, e));
        // all constraints are determined by the moment matrix
        assert!(sdp.free_constraint_words().iter().all(|w| w.is_empty()));
        // the constraints pin the state: a single free variable at most survives
        assert!(
            sdp.num_vars() <= 1,
            "theta {}: {} free variables",
            theta,
            sdp.num_vars()
        );
        let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
        assert!(sol.is_converged(1e-5), "theta {}: {:?}", theta, sol.status);
        assert!(
            (value - theta.cos()).abs() < 1e-6,
            "theta {}: value {}",
            theta,
            value
        );
        let _ = z;
    }
}

#[test]
fn inconsistent_constraints_are_reported_not_panicked() {
    let (alg, rels) = relations::pauli::<C>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let z: Polynomial<C> = alg.parse("Z0").unwrap();
    let one: Polynomial<C> = alg.one();
    let rel = Relaxation::level(gb.reducer(), 1)
        .minimize(&z)
        .moment_equality(&one, C::new(2.0, 0.0));
    let err = rel.try_build().unwrap_err();
    assert!(err.contains("L(1) = 1"), "{}", err);
    assert!(err.contains("build_moment_form"), "{}", err);
}

#[test]
fn free_constraint_words_are_reported() {
    let (alg, rels) = relations::pauli::<C>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<C> = alg.parse("Z0*Z1").unwrap();
    // degree-2 constraint: determined by the level-1 moment matrix; the
    // degree-3 word X0 Y1 Z0 ... reduces to a degree-2 word times i, so use a
    // genuinely new word: with level 1 on two qubits every product has degree
    // <= 2, so a Pauli string of weight 2 is determined but nothing of weight
    // 3 exists; take three qubits instead
    let (alg3, rels3) = relations::pauli::<C>(3);
    let gb3 = GroebnerBasis::compute(rels3, &GroebnerOptions::default());
    let red3 = gb3.reducer();
    let h3: Polynomial<C> = alg3.parse("Z0*Z1").unwrap();
    let covered: Polynomial<C> = alg3.parse("X0*X1").unwrap();
    let uncovered: Polynomial<C> = alg3.parse("X0*X1*X2").unwrap();
    // only words on qubits 0 and 1 in the index set: X0 X1 X2 is never a product
    let words: Vec<_> = red3
        .normal_words_up_to(1)
        .into_iter()
        .filter(|w| w.as_slice().iter().all(|v| *v < 6))
        .collect();
    let rel = Relaxation::new(red3, words)
        .minimize(&h3)
        .moment_equality(&covered, C::new(0.5, 0.0))
        .moment_equality(&uncovered, C::new(0.25, 0.0));
    let coverage = rel.constraint_coverage();
    assert_eq!(coverage.len(), 2);
    assert!(coverage[0].is_empty());
    assert_eq!(coverage[1].len(), 1);
    assert_eq!(alg3.format_word(&coverage[1][0]), "X0*X1*X2");
    let sdp = rel.build();
    assert_eq!(sdp.free_constraint_words(), &coverage[..]);
    let mf = rel.build_moment_form();
    assert_eq!(mf.free_constraint_words, coverage);
    let _ = (red, h);
}

#[test]
fn unconverged_value_is_nan_and_size_guard_fires() {
    let (alg, rels) = relations::pauli::<C>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<C> = alg.parse("X0 + Z0").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    let (value, sol) = sdp
        .solve(&SolverOptions {
            max_iter: 1,
            ..SolverOptions::default()
        })
        .unwrap();
    assert_eq!(sol.status, Status::MaxIterations);
    assert!(!sol.is_converged(1e-5));
    assert!(value.is_nan());
    // the guard refuses oversized problems with an explanation
    let mut m = SdpModel::new();
    let x = m.free_vars(3, "x");
    m.psd_affine(
        1,
        true,
        vec![(0, 0, Affine::var(x[0]) + Affine::var(x[1]) + Affine::var(x[2]))],
        "sum >= 0",
    );
    m.minimize(&(Affine::var(x[0]) + Affine::var(x[1]) + Affine::var(x[2])));
    let err = m
        .solve(&SolverOptions {
            max_variables: 2,
            ..SolverOptions::default()
        })
        .unwrap_err();
    assert!(err.0.contains("max_variables"), "{}", err);
}

/// The Hermitian assembly produces NF(u_a* u_b) only for a <= b; the
/// mirrored word NF(u_b* u_a) is determined through L(w*) = conj L(w) and
/// must not be reported as a free word of a constraint.
#[test]
fn coverage_counts_adjoints_of_entry_words() {
    use ncalg::FreeAlgebra;
    let alg = FreeAlgebra::new_star([("x", true), ("y", true)]);
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let gb = GroebnerBasis::compute(vec![p("x*x - 1"), p("y*y - 1")], &GroebnerOptions::default());
    let w = |n: &str| ncalg::Word::letter(alg.var_index(n).unwrap());
    let words = vec![ncalg::Word::empty(), w("x"), w("y")];
    for order in [words.clone(), words.iter().rev().cloned().collect()] {
        let rel = Relaxation::new(gb.reducer(), order)
            .minimize(&p("x"))
            .moment_equality(&p("x*y"), C::new(0.3, 0.0))
            .moment_equality(&p("y*x"), C::new(0.3, 0.0))
            .moment_equality(&p("x*y*x"), C::new(0.0, 0.0));
        let cov = rel.constraint_coverage();
        assert!(cov[0].is_empty() && cov[1].is_empty(), "{:?}", cov);
        // x*y*x is genuinely not a moment of the 3x3 matrix
        assert_eq!(cov[2].len(), 1, "{:?}", cov);
    }
}
