//! `ncalg::multi`: several moment functionals linked by linear equalities.

use ncalg::multi::MultiRelaxation;
use ncalg::ncsos::Relaxation;
use ncalg::solver::SolverOptions;
use ncalg::{Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Word};

type C = Complex<f64>;

/// One qubit as two anticommuting Hermitian unitaries `X`, `Z`.
fn qubit() -> (FreeAlgebra, GroebnerBasis<C>) {
    let alg = FreeAlgebra::new_star([("X", true), ("Z", true)]);
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let gb = GroebnerBasis::compute(
        vec![p("X*X - 1"), p("Z*Z - 1"), p("X*Z + Z*X")],
        &GroebnerOptions::default(),
    );
    (alg, gb)
}

/// max <X>_0 + <X>_1 subject to <Z>_0 + <Z>_1 = 0.6: z_0 = z_1 = 0.3 by
/// concavity, value 2 sqrt(1 - 0.09).
const LINKED: f64 = 1.907878402833891;

#[test]
fn linked_functionals_reach_the_analytic_optimum() {
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let mut mr = MultiRelaxation::new(gb.reducer());
    let f0 = mr.functional("f0", Relaxation::level(gb.reducer(), 1));
    let f1 = mr.functional("f1", Relaxation::level(gb.reducer(), 1));
    mr.link(&[(f0, &p("Z")), (f1, &p("Z"))], C::new(0.6, 0.0));
    mr.maximize(&[(f0, &p("X")), (f1, &p("X"))]);
    let sdp = mr.build().unwrap();
    // one block per functional
    assert_eq!(sdp.functionals.len(), 2);
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-6), "{:?}", sol.status);
    assert!((value - LINKED).abs() < 1e-6, "value {}", value);
    // the link holds and each functional is normalized
    let z = |k: usize| -> f64 {
        let m = sdp.moments(k, &sol.x);
        m.iter().find(|(w, _)| alg.format_word(w) == "Z").unwrap().1.re
    };
    assert!((z(f0) + z(f1) - 0.6).abs() < 1e-7);
    for k in [f0, f1] {
        let one = sdp
            .moments(k, &sol.x)
            .iter()
            .find(|(w, _)| w.is_empty())
            .unwrap()
            .1;
        assert!((one.re - 1.0).abs() < 1e-9);
    }
}

#[test]
fn agrees_with_the_central_idempotent_encoding() {
    // the same problem with central orthogonal idempotents P0, P1 in one functional
    let alg = FreeAlgebra::new_star([("P0", true), ("P1", true), ("X", true), ("Z", true)]);
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let rels = vec![
        p("P0*P0 - P0"),
        p("P1*P1 - P1"),
        p("P0*P1"),
        p("P1*P0"),
        p("P0*X - X*P0"),
        p("P0*Z - Z*P0"),
        p("P1*X - X*P1"),
        p("P1*Z - Z*P1"),
        p("X*X - 1"),
        p("Z*Z - 1"),
        p("X*Z + Z*X"),
    ];
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let w = |s: &[&str]| Word::from_vec(s.iter().map(|n| alg.var_index(n).unwrap()).collect());
    let words = vec![
        w(&["P0"]),
        w(&["P0", "X"]),
        w(&["P0", "Z"]),
        w(&["P1"]),
        w(&["P1", "X"]),
        w(&["P1", "Z"]),
    ];
    let rel = Relaxation::new(gb.reducer(), words)
        .maximize(&(&p("P0*X") + &p("P1*X")))
        .moment_equality(&p("P0"), C::new(1.0, 0.0))
        .moment_equality(&p("P1"), C::new(1.0, 0.0))
        .moment_equality(&(&p("P0*Z") + &p("P1*Z")), C::new(0.6, 0.0));
    let sdp = rel.try_build().unwrap();
    assert_eq!(sdp.blocks().len(), 2, "{:?}", sdp.blocks()); // the connected components split the functionals
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-6));
    assert!((value - LINKED).abs() < 1e-6, "value {}", value);
}

#[test]
fn link_on_a_word_outside_the_moments_is_an_error() {
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let mut mr = MultiRelaxation::new(gb.reducer());
    let f0 = mr.functional("f0", Relaxation::level(gb.reducer(), 1));
    // level 0: the index set is {1}, so X is not a moment of f1
    let f1 = mr.functional("f1", Relaxation::level(gb.reducer(), 0));
    mr.link(&[(f0, &p("Z")), (f1, &p("X"))], C::new(0.0, 0.0));
    mr.maximize(&[(f0, &p("X"))]);
    let err = mr.build().unwrap_err();
    assert!(err.contains("f1") && err.contains("no moment"), "{}", err);
}

#[test]
fn symmetry_inside_a_functional() {
    // a functional with the X <-> Z swap symmetry (preserves the relations)
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let mut mr = MultiRelaxation::new(gb.reducer());
    let sym = Relaxation::level(gb.reducer(), 1)
        .signed_symmetry(vec![ncalg::ncsos::SignedPermutation::new(vec![1, 0], vec![1, 1])]);
    let f0 = mr.functional("sym", sym);
    let f1 = mr.functional("free", Relaxation::level(gb.reducer(), 1));
    mr.link(&[(f0, &p("X")), (f1, &p("Z"))], C::new(0.0, 0.0));
    // invariant functional: <X> = <Z> = t with 2t^2 <= 1; free one: <Z> = t, maximize <X>_1 + <X + Z>_0
    mr.maximize(&[(f0, &(&p("X") + &p("Z"))), (f1, &p("X"))]);
    let sdp = mr.build().unwrap();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-6));
    // max over |t| <= 1/sqrt 2 of 2t + sqrt(1 - t^2): at t = 2/sqrt 5 > 1/sqrt 2, so t = 1/sqrt 2
    let t = 1.0 / 2f64.sqrt();
    assert!(
        (value - (2.0 * t + (1.0 - t * t).sqrt())).abs() < 1e-6,
        "value {}",
        value
    );
}

/// Certificate of a linked bound: one SOS per functional plus link
/// multipliers, recovered from the native solver's duals.
#[test]
fn certificate_of_linked_functionals() {
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    for maximize in [true, false] {
        let mut mr = MultiRelaxation::new(gb.reducer());
        let f0 = mr.functional("f0", Relaxation::level(gb.reducer(), 1));
        let f1 = mr.functional("f1", Relaxation::level(gb.reducer(), 1));
        mr.link(&[(f0, &p("Z")), (f1, &p("Z"))], C::new(0.6, 0.0));
        if maximize {
            mr.maximize(&[(f0, &p("X")), (f1, &p("X"))]);
        } else {
            mr.minimize(&[(f0, &p("X")), (f1, &p("X"))]);
        }
        let sdp = mr.build().unwrap();
        let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
        let expected = if maximize { LINKED } else { -LINKED };
        assert!((value - expected).abs() < 1e-6, "value {}", value);
        let cert = sdp.certificate(&sol.duals, gb.reducer(), 1e-9);
        assert!(
            (cert.bound - expected).abs() < 1e-6,
            "bound {} ({:?})",
            cert.bound,
            cert.link_multipliers
        );
        assert!(
            cert.max_residual < 1e-6 && cert.residual_l1 < 1e-5,
            "residual {} / {}",
            cert.max_residual,
            cert.residual_l1
        );
        assert_eq!(cert.link_multipliers.len(), 1);
        // the multiplier is the derivative of the optimum in the link value:
        // d/dv [2 sqrt(1 - v^2/4)] = -v / (2 sqrt(1 - v^2/4)) at v = 0.6
        let slope = -0.6 / (2.0 * (1.0f64 - 0.09).sqrt());
        let mu = if maximize { slope } else { -slope };
        assert!(
            (cert.link_multipliers[0].re - mu).abs() < 1e-5,
            "mu {:?}",
            cert.link_multipliers
        );
    }
}

/// With a symmetric functional and a functional with its own moment
/// constraint, the certificate still closes.
#[test]
fn certificate_with_symmetry_and_own_constraints() {
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let mut mr = MultiRelaxation::new(gb.reducer());
    let sym = Relaxation::level(gb.reducer(), 1)
        .signed_symmetry(vec![ncalg::ncsos::SignedPermutation::new(vec![1, 0], vec![1, 1])]);
    let f0 = mr.functional("sym", sym);
    let f1 = mr.functional(
        "fixed",
        Relaxation::level(gb.reducer(), 1).moment_equality(&p("Z"), C::new(0.5, 0.0)),
    );
    let f2 = mr.functional("free", Relaxation::level(gb.reducer(), 1));
    mr.link(&[(f0, &p("X")), (f2, &p("Z"))], C::new(0.0, 0.0));
    mr.maximize(&[(f0, &(&p("X") + &p("Z"))), (f1, &p("X")), (f2, &p("X"))]);
    let sdp = mr.build().unwrap();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    let t = 1.0 / 2f64.sqrt();
    let expected = 2.0 * t + (1.0 - t * t).sqrt() + (1.0f64 - 0.25).sqrt();
    assert!((value - expected).abs() < 1e-6, "value {}", value);
    let cert = sdp.certificate(&sol.duals, gb.reducer(), 1e-9);
    assert!((cert.bound - expected).abs() < 1e-6, "bound {}", cert.bound);
    assert!(cert.max_residual < 1e-6, "residual {}", cert.max_residual);
    assert_eq!(cert.constraint_multipliers[1].len(), 1);
}

/// A link with tiny coefficients is the same link (the absolute
/// 1e-15 prune emptied it and the link was skipped).
#[test]
fn tiny_link_is_the_same_link() {
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let s = 1e-16;
    let mut mr = MultiRelaxation::new(gb.reducer());
    let f0 = mr.functional("f0", Relaxation::level(gb.reducer(), 1));
    let f1 = mr.functional("f1", Relaxation::level(gb.reducer(), 1));
    let z = p("Z").map_coeffs(|c| c * s);
    mr.link(&[(f0, &z), (f1, &z)], C::new(0.6 * s, 0.0));
    mr.maximize(&[(f0, &p("X")), (f1, &p("X"))]);
    let (value, _) = mr.build().unwrap().solve(&SolverOptions::default()).unwrap();
    assert!((value - LINKED).abs() < 1e-6, "value {}", value);
}

/// Functionals must reduce with the MultiRelaxation's ideal.
#[test]
#[should_panic(expected = "different ideal")]
fn functional_with_another_ideal_is_rejected() {
    let (alg, gb) = qubit();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let commuting = GroebnerBasis::compute(
        vec![p("X*X - 1"), p("Z*Z - 1"), p("X*Z - Z*X")],
        &GroebnerOptions::default(),
    );
    let mut mr = MultiRelaxation::new(gb.reducer());
    mr.functional("other", Relaxation::level(commuting.reducer(), 1));
}
