//! Independent verification of truncated Gröbner bases by linear algebra.
//!
//! For a homogeneous ideal `I` and degree `d`, the degree-`d` part `I_d` is
//! spanned by the products `a · g · b` over generators `g` and words `a, b`
//! with `|a| + deg g + |b| = d`.  Its dimension can be computed by Gaussian
//! elimination without any Gröbner theory, and
//!
//! ```text
//!     dim K<X>_d − dim I_d  =  number of normal words of length d
//! ```
//!
//! holds iff the leading words of the basis are exactly the degree-`d`
//! leading words of `I`, i.e. iff the truncated basis is correct in degree d.

use ncalg::{
    Coeff, Field, Fp, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, QuotientAlgebra, Word,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

type F = Fp<1_000_003>;

fn random_homog(rng: &mut StdRng, alg: &FreeAlgebra, deg: usize, n_terms: usize) -> Polynomial<F> {
    let n = alg.num_vars() as u32;
    let terms = (0..n_terms).map(|_| {
        let w: Vec<u32> = (0..deg).map(|_| rng.gen_range(0..n)).collect();
        let c = rng.gen_range(-4i64..=4);
        (Word::from_vec(w), F::from_i64(c))
    });
    Polynomial::from_terms(alg.clone(), terms)
}

fn all_words(n: u32, len: usize) -> Vec<Vec<u32>> {
    let mut out = vec![vec![]];
    for _ in 0..len {
        out = out
            .into_iter()
            .flat_map(|w| {
                (0..n).map(move |v| {
                    let mut x = w.clone();
                    x.push(v);
                    x
                })
            })
            .collect();
    }
    out
}

/// Rank of the span of the given polynomials (all homogeneous of degree d)
/// by Gaussian elimination over the field.
fn rank<C: Field>(polys: &[Polynomial<C>], n_vars: u32, d: usize) -> usize {
    let words = all_words(n_vars, d);
    let index: HashMap<Vec<u32>, usize> = words.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
    let ncols = words.len();
    let mut rows: Vec<Vec<C>> = polys
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| {
            let mut row = vec![C::zero(); ncols];
            for (w, c) in p.iter() {
                row[index[w.as_slice()]] = c.clone();
            }
            row
        })
        .collect();
    let mut rank = 0;
    for col in 0..ncols {
        let pivot = (rank..rows.len()).find(|&r| !rows[r][col].is_zero());
        let Some(pr) = pivot else { continue };
        rows.swap(rank, pr);
        let inv = rows[rank][col].inverse();
        for c in &mut rows[rank] {
            *c = c.mul_ref(&inv);
        }
        let pivot_row = rows[rank].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != rank && !row[col].is_zero() {
                let factor = row[col].clone();
                for (x, p) in row.iter_mut().zip(&pivot_row) {
                    *x = x.sub_ref(&p.mul_ref(&factor));
                }
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rank
}

/// dim I_d computed from the generators alone.
fn ideal_dim(gens: &[Polynomial<F>], alg: &FreeAlgebra, d: usize) -> usize {
    let n = alg.num_vars() as u32;
    let mut spanning = Vec::new();
    for g in gens {
        let dg = g.degree().unwrap();
        if dg > d {
            continue;
        }
        for la in 0..=(d - dg) {
            let lb = d - dg - la;
            for a in all_words(n, la) {
                for b in all_words(n, lb) {
                    spanning.push(g.sandwich(&a, &b));
                }
            }
        }
    }
    rank(&spanning, n, d)
}

#[test]
fn truncated_bases_match_linear_algebra() {
    let mut rng = StdRng::seed_from_u64(11);
    for (n_vars, bound, criterion) in [
        (2u32, 7usize, true),
        (3, 5, true),
        (3, 6, true),
        (3, 6, false),
        (2, 8, false),
    ] {
        let alg = FreeAlgebra::with_n_vars(n_vars as usize);
        for trial in 0..6 {
            let n_gens = 1 + trial % 2;
            let deg = 2 + trial % 2;
            let gens: Vec<Polynomial<F>> = (0..n_gens)
                .map(|_| random_homog(&mut rng, &alg, deg, 3))
                .filter(|p| !p.is_zero())
                .collect();
            if gens.is_empty() {
                continue;
            }
            let gb = GroebnerBasis::compute(
                gens.clone(),
                &GroebnerOptions::new()
                    .max_degree(bound)
                    .chain_criterion(criterion),
            );
            let q = QuotientAlgebra::new(gb.clone());
            let series = q.hilbert_series(bound);
            for (d, &count) in series.iter().enumerate() {
                let total = (n_vars as usize).pow(d as u32);
                let dim_i = ideal_dim(&gens, &alg, d);
                assert_eq!(
                    count,
                    total - dim_i,
                    "n={} trial={} degree {}: gens {:?}, basis {:?}",
                    n_vars,
                    trial,
                    d,
                    gens,
                    gb.elements()
                );
            }
        }
    }
}

#[test]
fn inhomogeneous_complete_bases_decide_membership() {
    // For inhomogeneous generators, a complete basis must reduce every
    // two-sided multiple of a generator to zero.  Random small inhomogeneous
    // ideals in two variables usually have finite bases; those that hit the
    // pair limit are skipped.
    let mut rng = StdRng::seed_from_u64(5);
    let alg = FreeAlgebra::with_n_vars(2);
    let n = 2u32;
    let mut checked = 0;
    for _ in 0..12 {
        let g1 = random_homog(&mut rng, &alg, 2, 3) + random_homog(&mut rng, &alg, 1, 1);
        let g2 = random_homog(&mut rng, &alg, 2, 2) + alg.constant(F::from_i64(rng.gen_range(-2..=2)));
        let gens: Vec<Polynomial<F>> = vec![g1, g2].into_iter().filter(|p| !p.is_zero()).collect();
        if gens.is_empty() {
            continue;
        }
        let gb = GroebnerBasis::compute(gens.clone(), &GroebnerOptions::new().max_pairs(3000));
        if !gb.is_complete() {
            continue;
        }
        checked += 1;
        assert!(gb.verify(None).is_empty());
        for g in &gens {
            for la in 0..=2 {
                for lb in 0..=2 {
                    for a in all_words(n, la) {
                        for b in all_words(n, lb) {
                            let e = g.sandwich(&a, &b);
                            assert!(gb.contains(&e), "{} not reduced to zero by {}", e, gb);
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no complete bases found to check");
}
