//! Reduction while expanding: evaluating expressions modulo an ideal without
//! ever building the unreduced expansion.
//!
//! `cargo run --release --example reduce_while_expanding`

use ncalg::relations;
use ncalg::{
    Accumulator, BigRational, Expr, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, ReductionMode,
};
use std::time::Instant;

fn main() {
    // --- 1. the Weyl algebra: y x = x y + 1 -------------------------------
    let alg = FreeAlgebra::new(["x", "y"]);
    let rel: Polynomial<BigRational> = alg.parse("y*x - x*y - 1").unwrap();
    let gb = GroebnerBasis::compute(vec![rel], &GroebnerOptions::default());
    println!("{}", gb);

    let f: Polynomial<BigRational> = alg.parse("y^3 + x").unwrap();
    let g: Polynomial<BigRational> = alg.parse("x^3 - y").unwrap();
    // The product f*g is streamed term by term through the reducer; the
    // reduction steps push *their* terms into the same heap, so at no point
    // does an unreduced product exist in memory.
    let (nf, stats) = {
        let mut acc = Accumulator::new(&alg);
        acc.push_product(&f, &g);
        gb.reducer().run(acc, ReductionMode::Full)
    };
    println!("NF(f g) = {}", nf);
    println!(
        "  {} reduction steps, remainder {} terms, heap peak {}",
        stats.steps, stats.remainder_terms, stats.max_heap
    );
    assert_eq!(nf, gb.reduce(&(&f * &g)));

    // --- 2. expression trees -----------------------------------------------
    let x: Expr<BigRational> = alg.parse::<BigRational>("x").unwrap().into();
    let y: Expr<BigRational> = alg.parse::<BigRational>("y").unwrap().into();
    // (x + y)^8 * [x, y^2] - 3 (x y)^4
    let e = (x.clone() + y.clone()).pow(8) * Expr::commutator(x.clone(), y.clone().pow(2))
        - (x.clone() * y.clone()).pow(4).scale(ncalg::Coeff::from_i64(3));
    println!("\nexpression: {}", e);
    let t = Instant::now();
    let lazy = e.eval_mod(gb.reducer());
    let t_lazy = t.elapsed();
    let t = Instant::now();
    let free = e.eval_free();
    let eager = gb.reduce(&free);
    let t_eager = t.elapsed();
    assert_eq!(lazy, eager);
    println!(
        "normal form has {} terms (free expansion had {} terms)\n  reduce-while-expanding: {:?}\n  expand-then-reduce:     {:?}",
        lazy.num_terms(),
        free.num_terms(),
        t_lazy,
        t_eager
    );

    // --- 3. a finite-dimensional algebra: powers never grow -------------------
    let (palg, rels) = relations::pauli::<ncalg::Complex<f64>>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<ncalg::Complex<f64>> = palg
        .parse("X0*X1 + X1*X2 + Y0*Y1 + Y1*Y2 + Z0*Z1 + Z1*Z2 + 0.7*Z0 + 0.7*Z1 + 0.7*Z2")
        .unwrap();
    let n = 12;
    let t = Instant::now();
    let hn = gb.reduce_pow(&h, n);
    println!(
        "\n3-qubit Heisenberg H^{}: {} terms in normal form, computed lazily in {:?}",
        n,
        hn.num_terms(),
        t.elapsed()
    );
    println!(
        "(the free expansion would have up to {} terms)",
        h.num_terms().pow(n)
    );
}
