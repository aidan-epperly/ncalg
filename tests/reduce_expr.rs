//! Lazy reduction, expression evaluation and quotient algebra arithmetic.

use ncalg::relations;
use ncalg::{
    Accumulator, BigRational, Complex, Expr, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial,
    QuotientAlgebra, Reducer, ReductionMode, Word,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type Q = BigRational;

fn rat(alg: &FreeAlgebra, s: &str) -> Polynomial<Q> {
    alg.parse(s).unwrap()
}

fn random_poly(rng: &mut StdRng, alg: &FreeAlgebra, max_len: usize, n_terms: usize) -> Polynomial<Q> {
    let n = alg.num_vars() as u32;
    let terms = (0..n_terms).map(|_| {
        let len = rng.gen_range(0..=max_len);
        let w: Vec<u32> = (0..len).map(|_| rng.gen_range(0..n)).collect();
        let c = rng.gen_range(-3i64..=3);
        (Word::from_vec(w), ncalg::Coeff::from_i64(c))
    });
    Polynomial::from_terms(alg.clone(), terms)
}

fn weyl_gb() -> (FreeAlgebra, GroebnerBasis<Q>) {
    let alg = FreeAlgebra::new(["x", "y"]);
    let rel = rat(&alg, "y*x - x*y - 1");
    let gb = GroebnerBasis::compute(vec![rel], &GroebnerOptions::default());
    assert!(gb.is_complete());
    (alg, gb)
}

#[test]
fn reduce_product_agrees_with_expand_then_reduce() {
    let mut rng = StdRng::seed_from_u64(1);
    let (alg, gb) = weyl_gb();
    for _ in 0..25 {
        let f = random_poly(&mut rng, &alg, 4, 4);
        let g = random_poly(&mut rng, &alg, 4, 4);
        let lazy = gb.reduce_product(&f, &g);
        let eager = gb.reduce(&(&f * &g));
        assert_eq!(lazy, eager);
        // bilinearity through the reducer
        let h = random_poly(&mut rng, &alg, 3, 3);
        let lhs = gb.reduce_product(&f, &(&g + &h));
        let rhs = gb.reduce_product(&f, &g) + gb.reduce_product(&f, &h);
        assert_eq!(lhs, rhs);
    }
}

#[test]
fn reduce_pow_and_chain() {
    let mut rng = StdRng::seed_from_u64(2);
    let (alg, gb) = weyl_gb();
    for _ in 0..10 {
        let f = random_poly(&mut rng, &alg, 2, 3);
        for n in 0..5 {
            assert_eq!(gb.reduce_pow(&f, n), gb.reduce(&f.pow(n)));
        }
        let g = random_poly(&mut rng, &alg, 2, 3);
        let h = random_poly(&mut rng, &alg, 2, 3);
        let chain = gb.reducer().reduce_product_chain([&f, &g, &h]);
        assert_eq!(chain, gb.reduce(&(&f * &g * &h)));
    }
}

#[test]
fn reduction_modes() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let rel = rat(&alg, "y*x - x*y");
    let red = Reducer::new(&alg, vec![rel]);
    let p = rat(&alg, "y*x*x + y*x + x");
    let (top, _) = red.reduce_with_stats(&p, ReductionMode::Top);
    // leading term y x x -> x x y ; the term y x stays unreduced in Top mode
    assert_eq!(top, rat(&alg, "x*x*y + y*x + x"));
    let full = red.reduce(&p);
    assert_eq!(full, rat(&alg, "x*x*y + x*y + x"));
    assert!(red.is_reduced(&full));
    assert!(!red.is_reduced(&top));
    let tail = red.reduce_tail(&rat(&alg, "y*x + y*x*x"));
    // leading term y x x kept, tail y x reduced to x y
    assert_eq!(tail, rat(&alg, "y*x*x + x*y"));
}

#[test]
fn reduction_statistics_and_heap_behavior() {
    let (alg, gb) = weyl_gb();
    let f = rat(&alg, "y^3");
    let g = rat(&alg, "x^3");
    let mut acc = Accumulator::new(&alg);
    acc.push_product(&f, &g);
    assert_eq!(acc.live_streams(), 1);
    let (nf, stats) = gb.reducer().run(acc, ReductionMode::Full);
    assert!(stats.steps > 0);
    assert_eq!(stats.remainder_terms, nf.num_terms());
    assert_eq!(nf, gb.reduce(&(&f * &g)));
}

#[test]
fn accumulator_sums_and_cancellations() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let p = rat(&alg, "x*y + y*x - 2");
    let q = rat(&alg, "-x*y + 3");
    let mut acc = Accumulator::new(&alg);
    acc.push_poly(&p);
    acc.push_poly(&q);
    acc.push_scaled(ncalg::Coeff::from_i64(-1), &[], &p, &[]);
    let r = acc.finish();
    assert_eq!(r, q);
    // sandwiches
    let mut acc = Accumulator::new(&alg);
    acc.push_sandwich(&[1], &p, &[0]);
    assert_eq!(acc.finish(), p.sandwich(&[1], &[0]));
}

#[test]
fn expression_evaluation_matches_free_expansion() {
    let mut rng = StdRng::seed_from_u64(5);
    let (alg, gb) = weyl_gb();
    for _ in 0..15 {
        let a = random_poly(&mut rng, &alg, 2, 3);
        let b = random_poly(&mut rng, &alg, 2, 3);
        let c = random_poly(&mut rng, &alg, 2, 2);
        let ea: Expr<Q> = a.clone().into();
        let eb: Expr<Q> = b.clone().into();
        let ec: Expr<Q> = c.clone().into();
        let e = (ea.clone() + eb.clone()).pow(3) * ec.clone()
            - Expr::commutator(ea.clone(), eb.clone()) * Expr::anticommutator(eb.clone(), ec.clone())
            + Expr::scale(ea.clone() * eb.clone() * ec.clone(), ncalg::Coeff::from_i64(3))
            - ea.clone().adjoint();
        let free = e.eval_free();
        let modular = e.eval_mod(gb.reducer());
        assert_eq!(modular, gb.reduce(&free), "{}", e);
        assert!(gb.reducer().is_reduced(&modular));
    }
}

#[test]
fn expression_display_and_structure() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let x: Expr<Q> = rat(&alg, "x").into();
    let y: Expr<Q> = rat(&alg, "y").into();
    let e = Expr::commutator(x.clone(), y.clone()).pow(2) + x.clone() * y;
    assert_eq!(e.num_nodes(), 8);
    let s = e.to_string();
    assert!(s.contains("[(x), (y)]^2"));
    assert_eq!(e.eval_free(), rat(&alg, "[x,y]^2 + x*y"));
    assert!(e.algebra().unwrap().same_as(&alg));
    let prod = Expr::product(vec![]);
    let gb = GroebnerBasis::compute(vec![rat(&alg, "y*x - x*y")], &GroebnerOptions::default());
    assert_eq!(prod.eval_mod(gb.reducer()), alg.one());
}

#[test]
fn quotient_arithmetic_pauli() {
    let (_alg, rels) = relations::pauli::<Complex<Q>>(1);
    let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
    assert!(q.is_complete());
    let x = q.gen("X0");
    let y = q.gen("Y0");
    let z = q.gen("Z0");
    let i = q.parse("i").unwrap();
    assert_eq!(&x * &y, &z * &i);
    assert_eq!(&y * &x, -(&z * &i));
    assert_eq!(x.pow(2), q.one());
    assert_eq!(x.pow(7), x);
    assert!(x.commutator(&x).is_zero());
    let two: Complex<Q> = ncalg::Coeff::from_i64(2);
    assert_eq!(x.commutator(&y), &z * two * &i);
    assert!(x.anticommutator(&y).is_zero());
    // the Pauli algebra on one qubit has center K
    let three: Complex<Q> = ncalg::Coeff::from_i64(3);
    let h = &x + &y * three - &z;
    assert!(!h.commutes_with(&x));
    assert_eq!(
        h.adjoint(),
        h,
        "Pauli combination with real coefficients is Hermitian"
    );
    // (X + Y + Z)^2 = 3 (cross terms anticommute)
    let s = &x + &y + &z;
    assert_eq!(s.pow(2), q.constant(ncalg::Coeff::from_i64(3)));
    assert_eq!(s.pow(2).to_string(), "3");
    assert_eq!(q.normal_words(1).len(), 3);
    assert!(q.normal_words(2).is_empty());
}

#[test]
fn quotient_elements_hold_normal_forms() {
    let (alg, gb) = weyl_gb();
    let q = QuotientAlgebra::new(gb);
    let x = q.gen("x");
    let y = q.gen("y");
    let yx = &y * &x;
    assert_eq!(yx.representative(), &rat(&alg, "x*y + 1"));
    let mut acc = q.one();
    for _ in 0..3 {
        acc *= &y;
        acc *= &x;
    }
    assert_eq!(acc, (&y * &x).pow(3));
    assert!(q.reducer().is_reduced(acc.representative()));
    let e = q.element(&rat(&alg, "y*y*x*x"));
    assert_eq!(e.to_string(), "x^2*y^2 + 4*x*y + 2");
    assert_eq!(q.normal_words(2).len(), 3); // x^2, x y, y^2
    assert_eq!(q.hilbert_series(3), vec![1, 2, 3, 4]);
    assert_eq!(q.dimension(), None);
}

#[test]
fn quotient_with_partial_basis_still_reduces() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let q = QuotientAlgebra::from_relations(
        vec![rat(&alg, "x*y*x - x*y")],
        &GroebnerOptions::new().max_degree(5),
    );
    assert!(!q.is_complete());
    let e = q.element(&rat(&alg, "x*y*x*y*x"));
    assert_eq!(e.representative(), &rat(&alg, "x*y^2"));
    // normal words of small degree are correct even for the partial basis
    assert_eq!(q.hilbert_series(3), vec![1, 2, 4, 7]);
}

#[test]
fn moment_matrix_entries_and_coefficient_vectors() {
    let (alg, rels) = relations::majorana::<Q>(3);
    let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
    let words = q.normal_words_up_to(1); // 1, g0, g1, g2
    assert_eq!(words.len(), 4);
    let m = q.adjoint_products(&words);
    assert_eq!(m.len(), 4);
    // diagonal entries are 1 (g_i^2 = 1), off-diagonal are the products
    for (i, row) in m.iter().enumerate() {
        assert_eq!(row[i], alg.one());
    }
    assert_eq!(m[1][2], rat(&alg, "g0*g1"));
    assert_eq!(m[2][1], rat(&alg, "-g0*g1")); // g1 g0 = -g0 g1
                                              // entries agree with element arithmetic
    for (i, u) in words.iter().enumerate() {
        for (j, v) in words.iter().enumerate() {
            let eu = q.element(&alg.monomial(u.clone()));
            let ev = q.element(&alg.monomial(v.clone()));
            assert_eq!(m[i][j], *(eu.adjoint() * ev).representative());
        }
    }
    // coefficient vectors over the full normal-word basis
    let basis = q.normal_words_up_to(3);
    assert_eq!(basis.len(), 8);
    let p = rat(&alg, "g0*g1 - 2*g2 + 1/3");
    let v = p.coefficient_vector(&basis).unwrap();
    assert_eq!(v.len(), 8);
    assert_eq!(v[0], ncalg::Coeff::parse_literal("1/3").unwrap());
    assert_eq!(v[3], ncalg::Coeff::from_i64(-2)); // g2 is the 4th basis word
    assert!(p.coefficient_vector(&basis[..2]).is_none());
    assert_eq!(alg.zero::<Q>().coefficient_vector(&basis).unwrap().len(), 8);
}

#[test]
fn normal_words_are_sorted_and_normal() {
    let (_, rels) = relations::symmetric_group_3::<Q>();
    let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
    let alg = q.algebra().clone();
    for d in 0..5 {
        let words = q.normal_words(d);
        for w in &words {
            assert!(q.groebner_basis().is_normal_word(w));
            assert_eq!(w.len(), d);
        }
        for pair in words.windows(2) {
            assert_eq!(alg.cmp_words(&pair[0], &pair[1]), std::cmp::Ordering::Less);
        }
        assert_eq!(words.len(), q.hilbert_series(d)[d]);
    }
}
