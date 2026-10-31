//! Heavier Gröbner basis computations, for profiling.
//!
//! `cargo run --release --example stress [case]` where `case` is one of
//! `pauli`, `sym`, `random`, `sl3`, `all` (default `all`).

use ncalg::relations;
use ncalg::{
    BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, QuotientAlgebra, Word,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    eprintln!("{:<56} {:>9.3} s", label, t.elapsed().as_secs_f64());
    r
}

fn report<C: ncalg::Field>(gb: &GroebnerBasis<C>) {
    let st = gb.stats();
    eprintln!(
        "    {} elements | pairs reduced {} (zero {}) | criterion skips {} | redundant drops {} | reduction steps {} | {}",
        gb.len(),
        st.pairs_reduced,
        st.pairs_reduced_to_zero,
        st.pairs_skipped_criterion,
        st.pairs_dropped_redundant,
        st.reduction_steps,
        gb.termination()
    );
}

fn random_homog(rng: &mut StdRng, alg: &FreeAlgebra, deg: usize, n_terms: usize) -> Polynomial<BigRational> {
    let n = alg.num_vars() as u32;
    let terms = (0..n_terms).map(|_| {
        let w: Vec<u32> = (0..deg).map(|_| rng.gen_range(0..n)).collect();
        let c = rng.gen_range(-5i64..=5);
        (Word::from_vec(w), ncalg::Coeff::from_i64(c))
    });
    Polynomial::from_terms(alg.clone(), terms)
}

fn main() {
    let case = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let all = case == "all";

    if all || case == "pauli" {
        for n in [4, 5] {
            let (_, rels) = relations::pauli::<Complex<BigRational>>(n);
            let gb = time(
                &format!("Pauli {} qubits exact ({} relations)", n, rels.len()),
                || GroebnerBasis::compute(rels, &GroebnerOptions::default()),
            );
            report(&gb);
            let q = QuotientAlgebra::new(gb);
            let d = time("    dimension", || q.dimension());
            assert_eq!(d, Some(4usize.pow(n as u32)));
        }
    }

    if all || case == "sym" {
        for n in [7, 8] {
            let (_, rels) = relations::symmetric_group::<BigRational>(n);
            let gb = time(
                &format!("group algebra of S_{} ({} relations)", n, rels.len()),
                || GroebnerBasis::compute(rels, &GroebnerOptions::default()),
            );
            report(&gb);
            let q = QuotientAlgebra::new(gb);
            let d = time("    dimension", || q.dimension());
            assert_eq!(d, Some((1..=n).product::<usize>()));
        }
    }

    if all || case == "sl3" {
        // U(sl_3) via its 8-dimensional adjoint structure constants is long to
        // type; use gl_2 x gl_2-style test instead: two commuting copies of sl_2
        let alg = FreeAlgebra::new(["e1", "f1", "h1", "e2", "f2", "h2"]);
        let two: BigRational = ncalg::Coeff::from_i64(2);
        let rels = relations::lie_algebra::<BigRational, _>(&alg, |j, i| {
            let g = |k: u32| -> Polynomial<BigRational> { alg.var(k) };
            match (j, i) {
                (1, 0) => g(2).negated(),
                (2, 0) => g(0).scale(&two),
                (2, 1) => g(1).scale(&two).negated(),
                (4, 3) => g(5).negated(),
                (5, 3) => g(3).scale(&two),
                (5, 4) => g(4).scale(&two).negated(),
                _ => alg.zero(),
            }
        });
        let gb = time("U(sl_2 x sl_2), 15 relations (PBW)", || {
            GroebnerBasis::compute(rels, &GroebnerOptions::default())
        });
        report(&gb);
        assert_eq!(gb.len(), 15);
        // (e1 + f1 + h1 + e2 + f2 + h2)^6 reduced while expanding
        let s: Polynomial<BigRational> = alg.parse("e1 + f1 + h1 + e2 + f2 + h2").unwrap();
        let p = time("    (sum of generators)^6 lazily reduced", || {
            gb.reduce_pow(&s, 6)
        });
        eprintln!("    {} terms", p.num_terms());
        let p2 = time("    (sum of generators)^6 expand then reduce", || {
            gb.reduce(&s.pow(6))
        });
        assert_eq!(p, p2);
    }

    if all || case == "random" {
        let mut rng = StdRng::seed_from_u64(2024);
        let alg = FreeAlgebra::new(["x", "y", "z"]);
        let extra: Option<usize> = std::env::args().nth(2).and_then(|s| s.parse().ok());
        let mut cases = vec![(0, 9), (1, 10), (2, 11)];
        if let Some(b) = extra {
            cases[2].1 = b;
        }
        for (trial, bound) in cases {
            let gens: Vec<Polynomial<BigRational>> =
                (0..2).map(|_| random_homog(&mut rng, &alg, 3, 4)).collect();
            let gb = time(
                &format!("random 2 cubics in 3 vars, trial {}, bound {}", trial, bound),
                || GroebnerBasis::compute(gens.clone(), &GroebnerOptions::new().max_degree(bound)),
            );
            report(&gb);
            let max_terms = gb.elements().iter().map(|g| g.num_terms()).max().unwrap_or(0);
            eprintln!(
                "    largest element has {} terms, pending pairs {}",
                max_terms,
                gb.stats().pending_pairs
            );
            let q = QuotientAlgebra::new(gb);
            eprintln!(
                "    hilbert series up to {}: {:?}",
                bound,
                q.hilbert_series(bound)
            );
            // same computation with f64 coefficients, to separate arithmetic
            // cost from the combinatorial cost
            let gens_f: Vec<Polynomial<f64>> = gens.iter().map(|g| g.to_f64()).collect();
            let gbf = time(
                &format!("    same ideal with f64 coefficients, bound {}", bound),
                || GroebnerBasis::compute(gens_f, &GroebnerOptions::new().max_degree(bound)),
            );
            assert_eq!(gbf.len(), q.groebner_basis().len());
            assert_eq!(gbf.leading_words(), q.groebner_basis().leading_words());
        }
    }
}
