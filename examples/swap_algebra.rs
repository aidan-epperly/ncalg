//! Gröbner bases of the swap algebra: generators `s_ij` (`i < j`) satisfying
//! the relations of transpositions in the symmetric group `S_n`,
//!
//! ```text
//!   s_ij^2 = 1
//!   s_ij s_kl = s_kl s_ij               ({i,j} ∩ {k,l} = ∅)
//!   s_ij s_jk = s_ik s_ij = s_jk s_ik   (i, j, k distinct)
//! ```
//!
//! and of the *qubit* swap algebra, which adds the vanishing of the S_3
//! antisymmetrizer on every triple (true for swaps acting on (C^2)^{⊗n}):
//!
//! ```text
//!   1 - s_ij - s_ik - s_jk + s_ij s_jk + s_jk s_ij = 0
//! ```
//!
//! `cargo run --release --example swap_algebra [n_max] [time_limit_seconds]`

use ncalg::{FreeAlgebra, GroebnerBasis, GroebnerOptions, HybridRational, Polynomial, QuotientAlgebra};
use std::time::{Duration, Instant};

type Q = HybridRational;

/// Generator names `s12, s13, …` in lexicographic order of (i, j).
fn swap_algebra(n: usize) -> (FreeAlgebra, Vec<Vec<usize>>) {
    let mut pairs = Vec::new();
    let mut names = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            pairs.push(vec![i, j]);
            names.push(format!("s{}{}", i, j));
        }
    }
    (FreeAlgebra::new(names), pairs)
}

fn gen(alg: &FreeAlgebra, i: usize, j: usize) -> Polynomial<Q> {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    alg.gen(&format!("s{}{}", a, b))
}

fn relations(alg: &FreeAlgebra, n: usize, qubit: bool) -> Vec<Polynomial<Q>> {
    let mut rels = Vec::new();
    // involutions
    for i in 1..=n {
        for j in (i + 1)..=n {
            let s = gen(alg, i, j);
            rels.push(&s * &s - alg.one::<Q>());
        }
    }
    // disjoint swaps commute
    for i in 1..=n {
        for j in (i + 1)..=n {
            for k in 1..=n {
                for l in (k + 1)..=n {
                    if (i, j) < (k, l) && i != k && i != l && j != k && j != l {
                        rels.push(Polynomial::commutator(&gen(alg, i, j), &gen(alg, k, l)));
                    }
                }
            }
        }
    }
    // s_ij s_jk = s_ik s_ij for all ordered triples of distinct indices
    for i in 1..=n {
        for j in 1..=n {
            for k in 1..=n {
                if i == j || j == k || i == k {
                    continue;
                }
                let sij = gen(alg, i, j);
                let sjk = gen(alg, j, k);
                let sik = gen(alg, i, k);
                rels.push(&sij * &sjk - &sik * &sij);
            }
        }
    }
    if qubit {
        // antisymmetrizer of S_3 on each triple i < j < k
        for i in 1..=n {
            for j in (i + 1)..=n {
                for k in (j + 1)..=n {
                    let sij = gen(alg, i, j);
                    let sik = gen(alg, i, k);
                    let sjk = gen(alg, j, k);
                    let r = alg.one::<Q>() - &sij - &sik - &sjk + &sij * &sjk + &sjk * &sij;
                    rels.push(r);
                }
            }
        }
    }
    rels
}

fn catalan(n: usize) -> usize {
    // C_n = (2n)! / (n! (n+1)!)
    let mut c = 1usize;
    for k in 0..n {
        c = c * (2 * (2 * k + 1)) / (k + 2);
    }
    c
}

fn main() {
    let n_max: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let limit: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    // optional output directory: every basis is written to <dir>/<name>_n<n>.txt
    let out_dir: Option<std::path::PathBuf> = std::env::args().nth(3).map(Into::into);
    if let Some(d) = &out_dir {
        std::fs::create_dir_all(d).expect("create output directory");
    }
    let opts = GroebnerOptions::new().time_limit(Duration::from_secs(limit));

    for qubit in [false, true] {
        println!(
            "==================== {} ====================",
            if qubit {
                "qubit swap algebra (d = 2)"
            } else {
                "swap algebra (group algebra of S_n)"
            }
        );
        for n in 3..=n_max {
            let (alg, _pairs) = swap_algebra(n);
            let rels = relations(&alg, n, qubit);
            let t = Instant::now();
            let gb = GroebnerBasis::compute(rels.clone(), &opts);
            let elapsed = t.elapsed();
            let st = gb.stats();
            println!(
                "\nn = {}: {} generators, {} relations -> {} basis elements in {:.2?} ({})",
                n,
                alg.num_vars(),
                rels.len(),
                gb.len(),
                elapsed,
                gb.termination()
            );
            println!(
                "   pairs reduced {} (to zero {}), criterion skips {}, reduction steps {}",
                st.pairs_reduced, st.pairs_reduced_to_zero, st.pairs_skipped_criterion, st.reduction_steps
            );
            let mut by_degree = std::collections::BTreeMap::new();
            for g in gb.elements() {
                *by_degree.entry(g.degree().unwrap()).or_insert(0usize) += 1;
            }
            println!("   elements by degree: {:?}", by_degree);
            let q = QuotientAlgebra::new(gb.clone());
            let dim = q.dimension();
            let expected = if qubit {
                catalan(n)
            } else {
                (1..=n).product::<usize>()
            };
            println!(
                "   dimension {:?} (expected {}), Hilbert series {:?}, star-closed: {}",
                dim,
                expected,
                q.hilbert_series(6),
                gb.is_star_closed()
            );
            if gb.is_complete() {
                let failures = gb.verify(None);
                println!(
                    "   brute-force verification: {} failing obstructions",
                    failures.len()
                );
            }
            if n <= 4 || gb.len() <= 40 {
                for (i, g) in gb.elements().iter().enumerate() {
                    println!("   [{:>3}] {}", i, g);
                }
            }
            if let Some(d) = &out_dir {
                let name = if qubit { "qubit_swap" } else { "swap" };
                let path = d.join(format!("{}_n{}.txt", name, n));
                let mut text = format!(
                    "# {} algebra, n = {}: generators s_ij (i<j) ordered s12 < s13 < ... < s{}{}, deglex\n# {} elements, {}; dimension {:?}; Hilbert series {:?}\n# computed by ncalg in {:.2?}\n",
                    if qubit { "qubit swap" } else { "swap" },
                    n,
                    n - 1,
                    n,
                    gb.len(),
                    gb.termination(),
                    dim,
                    q.hilbert_series(8),
                    elapsed
                );
                for g in gb.elements() {
                    text.push_str(&g.to_string());
                    text.push('\n');
                }
                std::fs::write(&path, text).expect("write basis file");
                println!("   written to {}", path.display());
            }
            if !gb.is_complete() {
                println!("   stopping: time limit hit at n = {}", n);
                break;
            }
        }
    }
}
