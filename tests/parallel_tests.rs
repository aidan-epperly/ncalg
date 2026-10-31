//! Parallel (batched) Gröbner computation agrees with the sequential one.

use ncalg::relations;
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial};

type Q = BigRational;

fn both(rels: Vec<Polynomial<Q>>, opts: GroebnerOptions) -> (GroebnerBasis<Q>, GroebnerBasis<Q>) {
    let par = GroebnerBasis::compute(rels.clone(), &opts.clone().parallel(true));
    let seq = GroebnerBasis::compute(rels, &opts.parallel(false));
    (par, seq)
}

#[test]
fn complete_bases_are_identical() {
    let (_, sl2) = relations::sl2::<Q>();
    let (par, seq) = both(sl2, GroebnerOptions::default());
    assert!(par.is_complete() && seq.is_complete());
    assert_eq!(par.elements(), seq.elements());

    let alg = FreeAlgebra::new(["a", "b", "c", "d"]);
    let (par, seq) = both(relations::commutative::<Q>(&alg), GroebnerOptions::default());
    assert_eq!(par.elements(), seq.elements());

    let (_, rels) = relations::symmetric_group::<Q>(5);
    let (par, seq) = both(rels, GroebnerOptions::default());
    assert!(par.is_complete());
    assert_eq!(par.elements(), seq.elements());
    assert_eq!(par.elements().len(), seq.elements().len());

    // complex coefficients, Pauli algebra on 3 qubits
    let (_, rels) = relations::pauli::<Complex<Q>>(3);
    let par = GroebnerBasis::compute(rels.clone(), &GroebnerOptions::default());
    let seq = GroebnerBasis::compute(rels, &GroebnerOptions::default().parallel(false));
    assert_eq!(par.elements(), seq.elements());
}

#[test]
fn partial_bases_are_valid_in_both_modes() {
    // a degree-bounded computation: both results are truncated bases of the
    // same ideal (all generators reduce to zero, same normal word counts up
    // to the bound)
    let (alg, rels) = relations::symmetric_group::<Q>(6);
    let opts = GroebnerOptions::default().max_degree(4);
    let (par, seq) = both(rels.clone(), opts);
    assert!(!par.is_complete() && !seq.is_complete());
    for r in &rels {
        assert!(par.reduce(r).is_zero() && seq.reduce(r).is_zero());
    }
    for d in 0..=3 {
        assert_eq!(
            par.reducer().normal_words(d).len(),
            seq.reducer().normal_words(d).len(),
            "normal words of degree {} differ",
            d
        );
    }
    let _ = alg;
}

#[test]
fn pair_limit_is_respected_by_batches() {
    let (_, rels) = relations::symmetric_group::<Q>(5);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default().max_pairs(7));
    assert_eq!(gb.termination(), &ncalg::Termination::PairLimit);
    assert!(gb.stats().pairs_reduced <= 7 + 2 * rayon_threads());
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}
