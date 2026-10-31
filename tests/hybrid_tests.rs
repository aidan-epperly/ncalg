//! `HybridRational` gives the same Gröbner bases, relaxations and files as
//! `BigRational`.

use ncalg::ncsos::Relaxation;
use ncalg::relations;
use ncalg::solver::SolverOptions;
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, HybridRational, Polynomial};

type Q = BigRational;
type H = HybridRational;

#[test]
fn groebner_bases_agree_with_bigrational() {
    let (_, rels_q) = relations::symmetric_group::<Q>(5);
    let (_, rels_h) = relations::symmetric_group::<H>(5);
    let gb_q = GroebnerBasis::compute(rels_q, &GroebnerOptions::default());
    let gb_h = GroebnerBasis::compute(rels_h, &GroebnerOptions::default());
    assert!(gb_h.is_complete());
    let converted: Vec<Polynomial<H>> = gb_q.elements().iter().map(|g| g.convert()).collect();
    assert_eq!(gb_h.elements(), &converted[..]);
    assert!(gb_h
        .elements()
        .iter()
        .all(|g| g.iter().all(|(_, c)| c.is_small())));

    // sl2 (PBW basis with fractions), both directions of conversion
    let (_, rels) = relations::sl2::<H>();
    let gb_h = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let gb_q: GroebnerBasis<Q> = gb_h.convert();
    let (_, rels_q) = relations::sl2::<Q>();
    let direct = GroebnerBasis::compute(rels_q, &GroebnerOptions::default());
    assert_eq!(gb_q.elements(), direct.elements());

    // complex coefficients: Pauli algebra on two qubits
    let (_, rels) = relations::pauli::<Complex<H>>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let (_, rels_q) = relations::pauli::<Complex<Q>>(2);
    let gb_q = GroebnerBasis::compute(rels_q, &GroebnerOptions::default());
    let converted: Vec<Polynomial<Complex<Q>>> = gb.elements().iter().map(|g| g.convert()).collect();
    assert_eq!(converted, gb_q.elements());
}

#[test]
fn parsing_display_and_overflow_in_polynomials() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let p: Polynomial<H> = alg.parse("1/2*x*y - 3*y + 7/3").unwrap();
    assert_eq!(p.to_string(), "1/2*x*y - 3*y + 7/3");
    // huge coefficients are exact (promoted to BigRational internally)
    let big: Polynomial<H> = alg.parse("123456789012345*x").unwrap();
    let sq = &big * &big;
    let c = sq.iter().next().unwrap().1.clone();
    assert!(!c.is_small());
    assert_eq!(c.to_string(), "15241578753238669120562399025");
    let q: Polynomial<Q> = sq.convert();
    assert_eq!(q.to_string(), sq.to_string());
}

#[test]
fn relaxations_and_files_with_hybrid_coefficients() {
    let (alg, rels) = relations::pauli::<Complex<H>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<H>> = alg.parse("X0 + Z0").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sol.is_optimal());
    assert!((value + 2f64.sqrt()).abs() < 1e-7);
    // save / load keeps the coefficient type
    let dir = std::env::temp_dir().join("ncalg_hybrid_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pauli.ncalg");
    gb.save(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.contains("coefficients: Complex<HybridRational>"),
        "{}",
        text.lines().take(8).collect::<Vec<_>>().join("\n")
    );
    let back = GroebnerBasis::<Complex<H>>::load(&path).unwrap();
    assert_eq!(back.elements(), gb.elements());
    let as_q = GroebnerBasis::<Complex<Q>>::load_as(&path).unwrap();
    assert_eq!(as_q.len(), gb.len());
}

/// The Henrici-Knuth fast paths (no gcd of the result for coprime
/// denominators, cross-canceled products) agree with BigRational and keep
/// the canonical form, including near the i64 limits.
#[test]
fn fast_paths_match_bigrational() {
    use ncalg::{Coeff, Field, HybridRational};
    use num_bigint::BigInt;
    use num_rational::BigRational;
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let edge = [
        i64::MAX,
        i64::MIN + 1,
        i64::MIN,
        1 << 62,
        -(1 << 62),
        3,
        -1,
        1,
        0,
        6,
        10,
        15,
    ];
    let mut vals: Vec<(i64, i64)> = Vec::new();
    for &n in &edge {
        for &d in &[1i64, 2, 3, 6, i64::MAX, 1 << 62, 7919] {
            vals.push((n, d));
        }
    }
    for _ in 0..300 {
        let n = (next() as i64) >> (next() % 60);
        let d = ((next() >> (next() % 62 + 1)) as i64).max(1);
        vals.push((n, d));
    }
    let big = |n: i64, d: i64| BigRational::new(BigInt::from(n), BigInt::from(d));
    for &(a, b) in vals.iter().step_by(3) {
        for &(c, d) in vals.iter().step_by(5) {
            let (x, y) = (HybridRational::new(a, b), HybridRational::new(c, d));
            let (bx, by) = (big(a, b), big(c, d));
            for (got, want) in [
                (x.add_ref(&y), &bx + &by),
                (x.sub_ref(&y), &bx - &by),
                (x.mul_ref(&y), &bx * &by),
            ] {
                assert_eq!(got.to_big(), want, "{a}/{b} op {c}/{d}");
                assert_eq!(got, HybridRational::from_big(want.clone()), "canonical form");
            }
            if !y.is_zero() {
                assert_eq!(x.div_ref(&y).to_big(), &bx / &by);
            }
        }
    }
}
