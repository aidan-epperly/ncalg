//! Gröbner basis timings: `BigRational` versus `HybridRational` (and their
//! complex versions) on the built-in relation sets.
//!
//!     cargo run --release --example coeff_bench [n_max]
//!
//! `RAYON_NUM_THREADS=1` for single-threaded numbers.

use ncalg::relations;
use ncalg::{BigRational, Complex, Field, GroebnerBasis, GroebnerOptions, HybridRational, Polynomial};
use std::time::Instant;

fn time_gb<C: Field>(label: &str, rels: Vec<Polynomial<C>>) -> (usize, f64) {
    let t = Instant::now();
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let secs = t.elapsed().as_secs_f64();
    assert!(gb.is_complete(), "{}: not complete", label);
    (gb.len(), secs)
}

/// Defining relations of the swap algebra on `n` sites (generators `s_ij`,
/// `i < j`): involutions, commuting disjoint swaps, `s_ij s_jk = s_ik s_ij`;
/// with `qubit`, the vanishing antisymmetrizer of every triple (d = 2).
fn swap_relations<C: Field>(n: usize, qubit: bool) -> Vec<Polynomial<C>> {
    let mut names = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            names.push(format!("s{}{}", i, j));
        }
    }
    let alg = ncalg::FreeAlgebra::new(names);
    let gen = |i: usize, j: usize| -> Polynomial<C> {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        alg.gen(&format!("s{}{}", a, b))
    };
    let mut rels = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            let s = gen(i, j);
            rels.push(&s * &s - alg.one::<C>());
        }
    }
    for i in 1..=n {
        for j in (i + 1)..=n {
            for k in 1..=n {
                for l in (k + 1)..=n {
                    if (i, j) < (k, l) && i != k && i != l && j != k && j != l {
                        rels.push(Polynomial::commutator(&gen(i, j), &gen(k, l)));
                    }
                }
            }
        }
    }
    for i in 1..=n {
        for j in 1..=n {
            for k in 1..=n {
                if i == j || j == k || i == k {
                    continue;
                }
                rels.push(&gen(i, j) * &gen(j, k) - &gen(i, k) * &gen(i, j));
            }
        }
    }
    if qubit {
        for i in 1..=n {
            for j in (i + 1)..=n {
                for k in (j + 1)..=n {
                    let (sij, sik, sjk) = (gen(i, j), gen(i, k), gen(j, k));
                    rels.push(alg.one::<C>() - &sij - &sik - &sjk + &sij * &sjk + &sjk * &sij);
                }
            }
        }
    }
    rels
}

fn main() {
    let n_max: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    println!(
        "{:<28} {:>8} {:>12} {:>12} {:>8}",
        "relations", "elements", "BigRational", "Hybrid", "speedup"
    );
    let row = |label: &str, (n1, t1): (usize, f64), (n2, t2): (usize, f64)| {
        assert_eq!(n1, n2);
        println!(
            "{:<28} {:>8} {:>10.3} s {:>10.3} s {:>7.1}x",
            label,
            n1,
            t1,
            t2,
            t1 / t2
        );
    };
    for n in 5..=n_max {
        row(
            &format!("swap algebra, n = {}", n),
            time_gb("swap", swap_relations::<BigRational>(n, false)),
            time_gb("swap", swap_relations::<HybridRational>(n, false)),
        );
        row(
            &format!("qubit swap algebra, n = {}", n),
            time_gb("qubit swap", swap_relations::<BigRational>(n, true)),
            time_gb("qubit swap", swap_relations::<HybridRational>(n, true)),
        );
    }
    row(
        "sl2 (PBW)",
        time_gb("sl2", relations::sl2::<BigRational>().1),
        time_gb("sl2", relations::sl2::<HybridRational>().1),
    );
    for n in 4..=n_max {
        row(
            &format!("symmetric group S_{}", n),
            time_gb("S_n", relations::symmetric_group::<BigRational>(n).1),
            time_gb("S_n", relations::symmetric_group::<HybridRational>(n).1),
        );
    }
    for q in 2..=3 {
        row(
            &format!("Pauli algebra, {} qubits", q),
            time_gb("pauli", relations::pauli::<Complex<BigRational>>(q).1),
            time_gb("pauli", relations::pauli::<Complex<HybridRational>>(q).1),
        );
    }
    for n in [6, 8] {
        row(
            &format!("Majorana algebra, N = {}", n),
            time_gb("majorana", relations::majorana::<Complex<BigRational>>(n).1),
            time_gb("majorana", relations::majorana::<Complex<HybridRational>>(n).1),
        );
    }
}
