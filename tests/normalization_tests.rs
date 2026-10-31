//! Relaxations whose index set does not produce the empty word.

use ncalg::ncsos::Relaxation;
use ncalg::solver::SolverOptions;
use ncalg::{Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Word};

type C = Complex<f64>;

/// Two central orthogonal idempotents `P0, P1` and a Hermitian unitary `X`,
/// with index words `P_a`, `P_a X` only: no product `u* v` of index words
/// has a constant term, so `L(1)` is not a moment of the matrix.
fn two_projector_relaxation<'a>(gb: &'a GroebnerBasis<C>, alg: &FreeAlgebra) -> Relaxation<'a, C> {
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let w = |s: &[&str]| Word::from_vec(s.iter().map(|n| alg.var_index(n).unwrap()).collect());
    let words = vec![w(&["P0"]), w(&["P0", "X"]), w(&["P1"]), w(&["P1", "X"])];
    Relaxation::new(gb.reducer(), words)
        .minimize(&(&p("P0*X") + &p("P1*X")))
        .moment_equality(&p("P0"), C::new(1.0, 0.0))
        .moment_equality(&p("P1"), C::new(1.0, 0.0))
}

fn two_projector_algebra() -> (FreeAlgebra, GroebnerBasis<C>) {
    let alg = FreeAlgebra::new_star([("P0", true), ("P1", true), ("X", true)]);
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let rels = vec![
        p("P0*P0 - P0"),
        p("P1*P1 - P1"),
        p("P0*P1"),
        p("P1*P0"),
        p("P0*X - X*P0"),
        p("P1*X - X*P1"),
        p("X*X - 1"),
    ];
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    (alg, gb)
}

#[test]
fn index_set_without_the_empty_word_builds_and_solves() {
    // L(1) is not a moment of the matrix, so the normalization L(1) = 1 only
    // fixes a coordinate the SDP never sees; each functional is normalized by
    // its own L(P_a) = 1
    let (alg, gb) = two_projector_algebra();
    let rel = two_projector_relaxation(&gb, &alg);
    let sdp = rel.try_build().unwrap();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-5), "{:?}", sol.status);
    // min L(P0 X) + L(P1 X) with X^2 = 1 and L(P_a) = 1: -1 per functional
    assert!((value + 2.0).abs() < 1e-6, "value {}", value);
}

#[test]
fn adding_the_empty_word_builds() {
    let (alg, gb) = two_projector_algebra();
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let w = |s: &[&str]| Word::from_vec(s.iter().map(|n| alg.var_index(n).unwrap()).collect());
    let words = vec![
        Word::empty(),
        w(&["P0"]),
        w(&["P0", "X"]),
        w(&["P1"]),
        w(&["P1", "X"]),
    ];
    let rel = Relaxation::new(gb.reducer(), words)
        .minimize(&(&p("P0*X") + &p("P1*X")))
        .moment_equality(&p("P0"), C::new(0.5, 0.0))
        .moment_equality(&p("P1"), C::new(0.5, 0.0));
    rel.try_build().unwrap();
}

// --- connected components ---------------------------------------------

#[test]
fn orthogonal_idempotents_split_into_blocks() {
    let (alg, gb) = two_projector_algebra();
    let sdp = two_projector_relaxation(&gb, &alg).try_build().unwrap();
    // one 2x2 block per functional instead of one 4x4 block
    let sizes: Vec<usize> = sdp.blocks().iter().map(|b| b.size).collect();
    assert_eq!(sizes, vec![2, 2], "{:?}", sdp.blocks());
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-5), "{:?}", sol.status);
    assert!((value + 2.0).abs() < 1e-6, "value {}", value);
}

#[test]
fn splitting_can_be_disabled() {
    let (alg, gb) = two_projector_algebra();
    let sdp = two_projector_relaxation(&gb, &alg)
        .split_components(false)
        .try_build()
        .unwrap();
    let sizes: Vec<usize> = sdp.blocks().iter().map(|b| b.size).collect();
    assert_eq!(sizes, vec![4]);
    let (value, _) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!((value + 2.0).abs() < 1e-6, "value {}", value);
}

#[test]
fn components_permuted_by_the_symmetry_are_handled_jointly() {
    // swapping P0 <-> P1 maps one component onto the other
    let (alg, gb) = two_projector_algebra();
    let swap: Vec<ncalg::Var> = vec![1, 0, 2];
    let rel = two_projector_relaxation(&gb, &alg)
        .symmetry(vec![swap])
        .symmetrize_constraints(true);
    let sdp = rel.try_build().unwrap();
    let total: usize = sdp.blocks().iter().map(|b| b.size * b.irrep_dimension).sum();
    assert_eq!(total, 4, "{:?}", sdp.blocks());
    assert!(sdp.blocks().iter().all(|b| b.size <= 2), "{:?}", sdp.blocks());
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_converged(1e-5), "{:?}", sol.status);
    assert!((value + 2.0).abs() < 1e-6, "value {}", value);
}

#[test]
fn certificate_lifts_through_component_blocks() {
    let (alg, gb) = two_projector_algebra();
    let sdp = two_projector_relaxation(&gb, &alg).try_build().unwrap();
    let (_, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    // one Gram matrix per source matrix, in source coordinates
    let grams = sdp.dual_grams(&sol.duals);
    assert_eq!(grams.len(), 1);
    assert_eq!(grams[0].len(), 4);
    // the cross-functional entries of the Gram matrix are zero
    for row in &grams[0][..2] {
        assert!(row[2..4].iter().all(|g| g.norm() < 1e-12));
    }
}
