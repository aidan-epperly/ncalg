//! `Relaxation::symmetrize_constraints`: moment constraints averaged over the
//! symmetry group inside ncalg.

use ncalg::ncsos::{Relaxation, SignedPermutation};
use ncalg::relations;
use ncalg::solver::SolverOptions;
use ncalg::{Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Var};

type C = Complex<f64>;
const N: usize = 3;

fn ring() -> (FreeAlgebra, GroebnerBasis<C>) {
    let (alg, rels) = relations::pauli::<C>(N);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    (alg, gb)
}

/// Site rotation q -> q+1 on the Pauli generators `3q + k`.
fn rotation() -> Vec<Var> {
    (0..3 * N)
        .map(|v| (3 * ((v / 3 + 1) % N) + v % 3) as Var)
        .collect()
}

/// Rotation by pi about Y on every site: X -> -X, Z -> -Z (preserves XY = iZ etc.).
fn y_flip() -> SignedPermutation {
    let sign = (0..3 * N).map(|v| if v % 3 == 1 { 1 } else { -1 }).collect();
    SignedPermutation::new((0..3 * N as Var).collect(), sign)
}

fn p(alg: &FreeAlgebra, s: &str) -> Polynomial<C> {
    alg.parse(s).unwrap()
}

/// Ring energy `Σ Z_q Z_{q+1}` with the site-resolved constraints
/// `L(X_q Z_{q+1}) = 0.1` for every q (an orbit under the rotation).
fn relaxation<'a>(gb: &'a GroebnerBasis<C>, alg: &FreeAlgebra, sites: &[usize]) -> Relaxation<'a, C> {
    let h = p(alg, "Z0*Z1 + Z1*Z2 + Z2*Z0");
    let mut rel = Relaxation::level(gb.reducer(), 1)
        .minimize(&h)
        .symmetry(vec![rotation()]);
    for &q in sites {
        rel = rel.moment_equality(&p(alg, &format!("X{}*Z{}", q, (q + 1) % N)), C::new(0.1, 0.0));
    }
    rel
}

fn solve(rel: &Relaxation<'_, C>) -> f64 {
    let sdp = rel.try_build().unwrap();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-6), "{:?}", sol.status);
    value
}

#[test]
fn averaged_constraints_match_hand_averaging() {
    let (alg, gb) = ring();
    let auto = relaxation(&gb, &alg, &[0, 1, 2]).symmetrize_constraints(true);
    let eff = auto.effective_constraints().unwrap();
    assert_eq!(eff.len(), 1, "{:?}", eff);
    let hand_avg = p(&alg, "X0*Z1 + X1*Z2 + X2*Z0").scale(&C::new(1.0 / 3.0, 0.0));
    let by_hand = relaxation(&gb, &alg, &[]).moment_equality(&hand_avg, C::new(0.1, 0.0));
    let (a, b) = (solve(&auto), solve(&by_hand));
    assert!((a - b).abs() < 1e-7, "{} vs {}", a, b);
    // without the constraint the bound is lower (it is not vacuous)
    let free = solve(&relaxation(&gb, &alg, &[]));
    assert!(free < a - 1e-4 || (free - a).abs() < 1e-7);
}

#[test]
fn scalar_multiples_are_deduplicated() {
    let (alg, gb) = ring();
    let rel = relaxation(&gb, &alg, &[0, 1, 2])
        .moment_equality(&p(&alg, "2*X0*Z1 + 2*X1*Z2 + 2*X2*Z0"), C::new(0.6, 0.0))
        .symmetrize_constraints(true);
    assert_eq!(rel.effective_constraints().unwrap().len(), 1);
}

#[test]
fn non_invariant_constraint_set_is_rejected() {
    let (alg, gb) = ring();
    let rel = relaxation(&gb, &alg, &[0, 1]).symmetrize_constraints(true);
    let err = rel.try_build().unwrap_err();
    assert!(err.contains("not invariant"), "{}", err);
    assert!(rel.effective_constraints().is_err());
}

#[test]
fn zero_average_with_nonzero_value_is_rejected() {
    let (alg, gb) = ring();
    let h = p(&alg, "Z0*Z1 + Z1*Z2 + Z2*Z0");
    // invariant as a set under the flip (Z0 -> -Z0), but Z0 averages to zero
    let rel = Relaxation::level(gb.reducer(), 1)
        .minimize(&h)
        .signed_symmetry(vec![y_flip()])
        .moment_equality(&p(&alg, "Z0"), C::new(0.2, 0.0))
        .moment_equality(&p(&alg, "-Z0"), C::new(0.2, 0.0))
        .symmetrize_constraints(true);
    let err = rel.try_build().unwrap_err();
    assert!(err.contains("averages to zero"), "{}", err);
}

#[test]
fn off_by_default_and_group_elements() {
    let (alg, gb) = ring();
    let rel = relaxation(&gb, &alg, &[0, 1, 2]);
    assert_eq!(rel.effective_constraints().unwrap().len(), 3);
    assert_eq!(rel.group_elements().len(), N);
    // without averaging each constraint must be invariant by itself
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rel.try_build()));
    assert!(r.is_err());
}

/// A signed permutation that does not commute with the adjoint
/// (a ↦ a, a* ↦ −a*) is rejected; accepted, it forced L(a*a) = 0 and the
/// bound of min L(a² + a*²) came out 0 instead of −2.
#[test]
#[should_panic(expected = "commute with the adjoint")]
fn non_star_symmetry_is_rejected() {
    use ncalg::ncsos::{Relaxation, SignedPermutation};
    use ncalg::{DegLex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial};
    let alg = FreeAlgebra::with_order_and_adjoint(["a", "b"], DegLex::new(), vec![1, 0]);
    let gb = GroebnerBasis::<f64>::compute_in(&alg, Vec::new(), &GroebnerOptions::default());
    let h: Polynomial<f64> = alg.parse("a*a + b*b").unwrap();
    let _ = Relaxation::level(gb.reducer(), 1)
        .minimize(&h)
        .signed_symmetry(vec![SignedPermutation::new(vec![0, 1], vec![1, -1])])
        .build();
}

/// Over a real field, a cyclic symmetry of order 3 has complex isotypic
/// bases (ω, ω̄); their Gram blocks must be Hermitian.  Forced real, the
/// symmetric Gram bound was −1843 against −17.14 for every other route.
#[test]
fn real_field_cyclic_symmetry_gram_blocks_are_hermitian() {
    use ncalg::BigRational;
    let alg = FreeAlgebra::new(["x0", "x1", "x2"]);
    let rels = relations::commutative::<BigRational>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<BigRational> = gb.reducer().reduce(
        &alg.parse(
            "x0^4 + x1^4 + x2^4 + 3*x0^2*x1 + 3*x1^2*x2 + 3*x2^2*x0 \
             - 3*x0*x1^2 - 3*x1*x2^2 - 3*x2*x0^2 + x0 + x1 + x2",
        )
        .unwrap(),
    );
    let o = SolverOptions {
        warnings: false,
        ..Default::default()
    };
    let rel = || Relaxation::level(gb.reducer(), 2).minimize(&h);
    let (plain, _) = rel().build().solve(&o).unwrap();
    let sym = rel()
        .symmetry(vec![vec![1, 2, 0]])
        .block_diagonalize(true)
        .sos_model();
    let sol = sym.model.solve(&o).unwrap();
    assert!(
        (sol.objective - plain).abs() < 1e-5,
        "Gram {} vs moment {}",
        sol.objective,
        plain
    );
    // and its certificate proves the bound
    let cert = sym.certificate(&sol.x, 1e-12);
    let gbf = gb.convert::<Complex<f64>>();
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!(
        (ver.bound - plain).abs() < 1e-5 && ver.max_residual < 1e-6,
        "{} {}",
        ver.bound,
        ver.max_residual
    );
}
