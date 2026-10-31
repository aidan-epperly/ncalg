//! Rough timing of Gröbner basis computations and lazy reductions.
//!
//! Run with `cargo run --release --example bench`.

use ncalg::relations;
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, QuotientAlgebra};
use std::time::Instant;

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    eprintln!("{:<48} {:>9.3} s", label, t.elapsed().as_secs_f64());
    r
}

fn main() {
    // Pauli algebras: finite, 4^n dimensional
    for n in 1..=3 {
        let (_, rels) = relations::pauli::<Complex<BigRational>>(n);
        let gb = time(
            &format!("Pauli {} qubits, exact ({} relations)", n, rels.len()),
            || GroebnerBasis::compute(rels, &GroebnerOptions::default()),
        );
        let st = gb.stats();
        eprintln!(
            "    {} elements, {} pairs reduced ({} to zero), {} skipped by criterion, {} reduction steps",
            gb.len(),
            st.pairs_reduced,
            st.pairs_reduced_to_zero,
            st.pairs_skipped_criterion,
            st.reduction_steps
        );
        let q = QuotientAlgebra::new(gb);
        assert_eq!(q.dimension(), Some(4usize.pow(n as u32)));
    }
    for n in 1..=3 {
        let (_, rels) = relations::pauli::<Complex<f64>>(n);
        let gb = time(&format!("Pauli {} qubits, f64", n), || {
            GroebnerBasis::compute(rels, &GroebnerOptions::default())
        });
        assert!(gb.is_complete());
    }

    // Majorana / Clifford algebras: 2^n dimensional
    for n in [4, 6, 8] {
        let (_, rels) = relations::majorana::<BigRational>(n);
        let gb = time(&format!("Majorana {} modes", n), || {
            GroebnerBasis::compute(rels, &GroebnerOptions::default())
        });
        let q = QuotientAlgebra::new(gb);
        assert_eq!(q.dimension(), Some(1 << n));
    }

    // symmetric groups: n! dimensional group algebras
    for n in [3, 4, 5, 6] {
        let (_, rels) = relations::symmetric_group::<BigRational>(n);
        let gb = time(&format!("group algebra of S_{}", n), || {
            GroebnerBasis::compute(rels, &GroebnerOptions::default())
        });
        let st = gb.stats();
        eprintln!(
            "    {} elements, {} pairs reduced, {} skipped by criterion, {} reduction steps",
            gb.len(),
            st.pairs_reduced,
            st.pairs_skipped_criterion,
            st.reduction_steps
        );
        let q = QuotientAlgebra::new(gb);
        let dim = time(&format!("    dimension of S_{} quotient", n), || q.dimension());
        assert_eq!(dim, Some((1..=n).product::<usize>()));
    }

    // Weyl algebra
    for n in [1, 2, 3] {
        let (_, rels) = relations::weyl::<BigRational>(n);
        let gb = time(&format!("Weyl algebra A_{}", n), || {
            GroebnerBasis::compute(rels, &GroebnerOptions::default())
        });
        assert!(gb.is_complete());
    }

    // infinite basis, truncated
    let alg = FreeAlgebra::new(["x", "y"]);
    let f: Polynomial<BigRational> = alg.parse("x*y*x - x*y").unwrap();
    for d in [20, 40, 80] {
        let gb = time(&format!("<xyx - xy> truncated at degree {}", d), || {
            GroebnerBasis::compute(vec![f.clone()], &GroebnerOptions::new().max_degree(d))
        });
        eprintln!("    {} elements, {}", gb.len(), gb.termination());
    }

    // a harder homogeneous example: two cubic relations, truncated
    let gens: Vec<Polynomial<BigRational>> = vec![
        alg.parse("x*y*x - y*x*y").unwrap(),
        alg.parse("x^2*y - y*x^2 + x*y^2").unwrap(),
    ];
    for d in [8, 10, 12] {
        let gb = time(&format!("two cubics truncated at degree {}", d), || {
            GroebnerBasis::compute(gens.clone(), &GroebnerOptions::new().max_degree(d))
        });
        let st = gb.stats();
        eprintln!(
            "    {} elements, {} pairs reduced, {} skipped, {} pending, {}",
            gb.len(),
            st.pairs_reduced,
            st.pairs_skipped_criterion,
            st.pending_pairs,
            gb.termination()
        );
    }

    // reduce while expanding vs expand then reduce in the 2-qubit Pauli algebra
    let (palg, rels) = relations::pauli::<Complex<f64>>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<f64>> = palg.parse("X0*X1 + Y0*Y1 + Z0*Z1 + 0.5*Z0 + 0.3*X1").unwrap();
    let n = 10;
    let lazy = time(&format!("H^{} lazily reduced", n), || gb.reduce_pow(&h, n));
    let eager = time(
        &format!(
            "H^{} expanded then reduced ({} terms before)",
            n,
            h.num_terms().pow(n)
        ),
        || gb.reduce(&h.pow(n)),
    );
    assert!(lazy.approx_eq(&eager, 1e-6));
    eprintln!("    normal form has {} terms", lazy.num_terms());
}
