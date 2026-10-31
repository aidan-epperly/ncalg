//! Numerical verification of a qubit swap algebra Gröbner basis.
//!
//! The swap operators `S_ij` on `(C^2)^{⊗n}` are permutation matrices.  A
//! polynomial in the `s_ij` is evaluated by composing the permutations of
//! each word and accumulating the coefficient into a dense `2^n × 2^n`
//! matrix.  For random polynomials `p` we check
//!
//! ```text
//!     eval(p) == eval( NF(p) mod GB )
//! ```
//!
//! and additionally that every basis element evaluates to the zero matrix.
//!
//! `cargo run --release --example swap_verify <basis_file> [n] [trials]`

use ncalg::{DegLex, FreeAlgebra, GroebnerBasis, HybridRational, Polynomial, Termination, Var, Word};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

type Q = HybridRational;

fn pairs(n: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            v.push((i, j));
        }
    }
    v
}

fn algebra(n: usize, reversed: bool) -> FreeAlgebra {
    let names: Vec<String> = pairs(n)
        .into_iter()
        .map(|(i, j)| format!("s{}{}", i, j))
        .collect();
    if reversed {
        let chain: Vec<Var> = (0..names.len() as Var).rev().collect();
        FreeAlgebra::with_order(names, DegLex::with_chain(&chain))
    } else {
        FreeAlgebra::new(names)
    }
}

/// Permutation of basis states induced by swapping qubits i and j (1-indexed).
fn swap_perm(n: usize, i: usize, j: usize) -> Vec<usize> {
    let dim = 1usize << n;
    (0..dim)
        .map(|b| {
            let bi = (b >> (i - 1)) & 1;
            let bj = (b >> (j - 1)) & 1;
            if bi == bj {
                b
            } else {
                b ^ (1 << (i - 1)) ^ (1 << (j - 1))
            }
        })
        .collect()
}

/// Dense matrix of `p` evaluated at the swap permutations (row-major, dim×dim).
fn evaluate(p: &Polynomial<f64>, perms: &[Vec<usize>], dim: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; dim * dim];
    for (word, &c) in p.iter() {
        // matrix of the word S_{v1} S_{v2} ... S_{vk} applied to |b>: apply S_{vk} first
        for b in 0..dim {
            let mut x = b;
            for &v in word.iter().rev() {
                x = perms[v as usize][x];
            }
            m[x * dim + b] += c;
        }
    }
    m
}

fn max_abs(m: &[f64]) -> f64 {
    m.iter().fold(0.0f64, |a, &x| a.max(x.abs()))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |acc, (x, y)| acc.max((x - y).abs()))
}

fn main() {
    let path = std::env::args().nth(1).expect(
        "usage: swap_verify <basis_file> [n] [trials] (basis files are written by the swap_algebra example)",
    );
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let trials: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    let text = std::fs::read_to_string(&path).expect("read basis file");
    let reversed = text
        .lines()
        .next()
        .map(|l| l.contains("reversed"))
        .unwrap_or(false);
    let alg = algebra(n, reversed);
    let t = Instant::now();
    let elems: Vec<Polynomial<Q>> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            alg.parse::<Q>(l)
                .unwrap_or_else(|e| panic!("cannot parse line {:?}: {}", l, e))
        })
        .collect();
    println!(
        "loaded {} basis elements from {} ({} ordering) in {:.2?}",
        elems.len(),
        path,
        if reversed {
            "reversed deglex"
        } else {
            "natural deglex"
        },
        t.elapsed()
    );
    let gb = GroebnerBasis::from_elements(&alg, elems, Termination::Complete);

    let dim = 1usize << n;
    let perms: Vec<Vec<usize>> = pairs(n).into_iter().map(|(i, j)| swap_perm(n, i, j)).collect();

    // 1. every basis element must vanish on the swap matrices
    let t = Instant::now();
    let mut worst = 0.0f64;
    for g in gb.elements() {
        let m = evaluate(&g.convert::<f64>(), &perms, dim);
        worst = worst.max(max_abs(&m));
    }
    println!(
        "all {} basis elements evaluated on the {}x{} swap matrices: max |entry| = {:.3e} ({:.2?})",
        gb.len(),
        dim,
        dim,
        worst,
        t.elapsed()
    );

    // 2. random polynomials: eval(p) == eval(NF(p))
    let mut rng = StdRng::seed_from_u64(2025);
    let nv = alg.num_vars() as u32;
    let mut worst_diff = 0.0f64;
    let mut total_reduction = std::time::Duration::ZERO;
    let mut nonzero_nf = 0;
    for trial in 0..trials {
        let n_terms = rng.gen_range(3..=12);
        let max_len = 3 + trial % 6; // words up to length 8
        let terms = (0..n_terms).map(|_| {
            let len = rng.gen_range(1..=max_len);
            let w: Vec<Var> = (0..len).map(|_| rng.gen_range(0..nv)).collect();
            let c: Q = ncalg::Coeff::from_i64(rng.gen_range(-5i64..=5));
            (Word::from_vec(w), c)
        });
        let p = Polynomial::from_terms(alg.clone(), terms);
        if p.is_zero() {
            continue;
        }
        let t = Instant::now();
        let nf = gb.reduce(&p);
        total_reduction += t.elapsed();
        if !nf.is_zero() {
            nonzero_nf += 1;
        }
        let ep = evaluate(&p.convert::<f64>(), &perms, dim);
        let enf = evaluate(&nf.convert::<f64>(), &perms, dim);
        let diff = max_abs_diff(&ep, &enf);
        worst_diff = worst_diff.max(diff);
        if trial < 5 || diff > 1e-8 {
            println!(
                "  trial {:>2}: {} terms (max word length {}) -> normal form {} terms (degree {:?}); max |eval(p) - eval(NF p)| = {:.3e}",
                trial,
                p.num_terms(),
                p.degree().unwrap(),
                nf.num_terms(),
                nf.degree(),
                diff
            );
        }
    }
    println!(
        "{} random polynomials: max |eval(p) - eval(NF p)| = {:.3e}; {} had a nonzero normal form; reduction time {:.2?} total",
        trials, worst_diff, nonzero_nf, total_reduction
    );

    // 3. a polynomial that vanishes on the swap matrices but is far from normal form:
    //    the S_3 antisymmetrizer on a far-apart triple times random words
    let a: Polynomial<Q> = alg.parse("1 - s18 - s17 - s78 + s18*s78 + s78*s18").unwrap();
    let left: Polynomial<Q> = alg.parse("s12*s34*s56").unwrap();
    let right: Polynomial<Q> = alg.parse("s23*s45*s67*s27").unwrap();
    let z = &left * &a * &right;
    let nf = gb.reduce(&z);
    println!(
        "antisymmetrizer sandwich: eval max |entry| = {:.3e}, normal form has {} terms (expected 0)",
        max_abs(&evaluate(&z.convert::<f64>(), &perms, dim)),
        nf.num_terms()
    );

    let ok = worst < 1e-9 && worst_diff < 1e-8 && nf.is_zero();
    println!("verification {}", if ok { "PASSED" } else { "FAILED" });
}
