#![allow(clippy::needless_range_loop)]
//! Tests for the SDP modeling layer and its Gram / Gram-weight / moment-constraint
//! front ends in `ncsos`.

use ncalg::ncsos::{Relaxation, SosModel};
use ncalg::relations;
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Word};

type Q = BigRational;

fn position_of(words: &[Word], alg: &FreeAlgebra, name: &str) -> usize {
    let target: Polynomial<Complex<Q>> = alg.parse(name).unwrap();
    let w = target.leading_word().unwrap().clone();
    words.iter().position(|u| *u == w).expect("word in index set")
}

#[test]
fn sos_model_certificate_for_pauli_x_plus_z() {
    let (alg, rels) = relations::pauli::<Complex<Q>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<Q>> = alg.parse("X0 + Z0").unwrap();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let sos: SosModel = rel.sos_model();
    assert_eq!(sos.grams.len(), 1);
    assert_eq!(sos.grams[0].size(), 4);
    assert!(!sos.grams[0].is_real());
    // 16 real entries of the Hermitian Gram matrix + the bound c
    assert_eq!(sos.model.num_vars(), 17);
    assert!(!sos.model.is_minimize()); // max c
                                       // X + Z + sqrt2 = (1/(2 sqrt2)) (sqrt2 + X + Z)^2 : G = v v^T / (2 sqrt2), v = (sqrt2, 1, 0, 1)
    let words = &sos.source_words[0];
    let s2 = 2f64.sqrt();
    let mut v = [0.0; 4];
    v[position_of(words, &alg, "1")] = s2;
    v[position_of(words, &alg, "X0")] = 1.0;
    v[position_of(words, &alg, "Z0")] = 1.0;
    let g = &sos.grams[0];
    let mut x = vec![0.0; sos.model.num_vars()];
    for a in 0..4 {
        for b in a..4 {
            x[g.re_var(a, b)] = v[a] * v[b] / (2.0 * s2);
        }
    }
    x[sos.bound.unwrap()] = -s2;
    let check = sos.model.check(&x, 1e-9);
    assert!(check.feasible(), "{:?}", check);
    assert!((sos.bound_value(&x).unwrap() + s2).abs() < 1e-12);
    assert!((sos.model.objective_value(&x) + s2).abs() < 1e-12);
    // a larger bound with the same Gram matrix violates the coefficient identity
    x[sos.bound.unwrap()] = -1.0;
    assert!(!sos.model.check(&x, 1e-9).feasible());
    x[sos.bound.unwrap()] = -s2;
    // the Gram matrix is recovered from the solution
    let gm = sos.gram_value(0, &x);
    let i1 = position_of(words, &alg, "1");
    let ix = position_of(words, &alg, "X0");
    assert!((gm[i1][ix].re - 0.5).abs() < 1e-12 && gm[i1][ix].im.abs() < 1e-12);
    // JSON export
    let dir = std::env::temp_dir().join("ncalg_sdp_test");
    std::fs::create_dir_all(&dir).unwrap();
    sos.model.write_json(dir.join("sos.json")).unwrap();
    let text = std::fs::read_to_string(dir.join("sos.json")).unwrap();
    assert!(text.contains("\"equalities\":[{"));
    assert!(text.contains("\"names\":["));
    // eliminated form: the coefficient identities fix some coordinates
    let e = sos.model.eliminate().unwrap();
    assert!(e.model.equalities().is_empty());
    assert!(e.model.num_vars() < sos.model.num_vars());
    e.model.write_sdpa(dir.join("sos.dat-s")).unwrap();
    let z = e.project(&x);
    let lifted = e.lift(&z);
    assert!(sos.model.check(&lifted, 1e-9).feasible());
    assert!((e.model.objective_value(&z) - sos.model.objective_value(&x)).abs() < 1e-12);
}

#[test]
fn sos_model_real_commutative_square() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Q> = alg.parse("x^2 - 2*x*y + y^2").unwrap();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let sos = rel.sos_model();
    let g = &sos.grams[0];
    assert!(g.is_real());
    assert_eq!(sos.model.num_vars(), 6 + 1);
    let words = &sos.source_words[0];
    let find = |name: &str| {
        let p: Polynomial<Q> = alg.parse(name).unwrap();
        let w = p.leading_word().unwrap().clone();
        words.iter().position(|u| *u == w).unwrap()
    };
    let (ix, iy) = (find("x"), find("y"));
    let mut x = vec![0.0; sos.model.num_vars()];
    x[g.re_var(ix, ix)] = 1.0;
    x[g.re_var(iy, iy)] = 1.0;
    x[g.re_var(ix, iy)] = -1.0;
    x[sos.bound.unwrap()] = 0.0;
    assert!(sos.model.check(&x, 1e-9).feasible());
    // (x - y)^2 - c with c > 0 is not a sum of squares
    x[sos.bound.unwrap()] = 0.1;
    assert!(!sos.model.check(&x, 1e-9).feasible());
}

#[test]
fn gram_weight_model_structure_and_feasible_point() {
    let (alg, rels) = relations::pauli::<Complex<Q>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<Q>> = alg.parse("X0 + Z0").unwrap();
    let s2 = 2f64.sqrt();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let sos = rel.gram_weight_model(s2);
    assert!(sos.bound.is_none());
    assert_eq!(sos.weights.len(), 4);
    assert_eq!(sos.shift, s2);
    assert_eq!(sos.model.lmis().len(), 2); // G >= 0 and diag(w) - G >= 0
    assert!(sos.model.is_minimize());
    let words = &sos.source_words[0];
    let mut v = [0.0; 4];
    v[position_of(words, &alg, "1")] = s2;
    v[position_of(words, &alg, "X0")] = 1.0;
    v[position_of(words, &alg, "Z0")] = 1.0;
    let g = &sos.grams[0];
    let mut x = vec![0.0; sos.model.num_vars()];
    for a in 0..4 {
        for b in a..4 {
            x[g.re_var(a, b)] = v[a] * v[b] / (2.0 * s2);
        }
    }
    // Gershgorin: w_a = sum_b |G_ab| makes diag(w) - G diagonally dominant
    let gm = g.value(&x);
    for a in 0..4 {
        x[sos.weights[a]] = gm[a].iter().map(|c| c.norm()).sum::<f64>();
    }
    let check = sos.model.check(&x, 1e-9);
    assert!(check.feasible(), "{:?}", check);
    let total: f64 = sos.weight_values(&x).iter().sum();
    assert!((sos.model.objective_value(&x) - total).abs() < 1e-12);
    // too small a weight breaks the second LMI but not the identities
    x[sos.weights[position_of(words, &alg, "1")]] = 0.0;
    let check = sos.model.check(&x, 1e-9);
    assert!(check.max_equality_residual < 1e-9 && check.min_eigenvalue < -1e-6);
}

#[test]
fn moment_constraints_in_both_forms() {
    let (alg, rels) = relations::pauli::<Complex<Q>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Complex<Q>> = alg.parse("Z0").unwrap();
    let x0: Polynomial<Complex<Q>> = alg.parse("X0").unwrap();
    let half = Complex::new(Q::new(1.into(), 2.into()), Q::from(ncalg::BigInt::from(0)));
    let rel = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&x0, half)
        .moment_inequality(
            &h,
            Complex::new(Q::from(ncalg::BigInt::from(0)), Q::from(ncalg::BigInt::from(0))),
        );
    // eliminated form: L(X) is fixed, L(Y) and L(Z) remain; one extra 1x1 (embedded 2x2) block
    let sdp = rel.build();
    assert_eq!(sdp.num_vars(), 2);
    assert_eq!(sdp.block_sizes(), vec![8, 2]);
    let one: Polynomial<Complex<Q>> = alg.one();
    for z in [vec![0.0, 0.0], vec![0.3, -0.7]] {
        let v = sdp.evaluate(&z, red, &x0).unwrap();
        assert!((v.re - 0.5).abs() < 1e-12 && v.im.abs() < 1e-12);
        assert!((sdp.evaluate(&z, red, &one).unwrap().re - 1.0).abs() < 1e-12);
        // the affine form of L(Z) agrees with evaluate
        let a = sdp.moment_affine(&red.reduce(&h)).unwrap();
        assert!((a.evaluate(&z) - sdp.evaluate(&z, red, &h).unwrap()).norm() < 1e-12);
    }
    // moment form: the constraint is an explicit equality; eliminating all
    // equalities leaves the same number of free variables
    let mf = rel.build_moment_form();
    let m = mf.to_model();
    let e = m.eliminate().unwrap();
    assert_eq!(e.model.num_vars(), sdp.num_vars());
    let lx = mf.moment_affine(&red.reduce(&x0)).unwrap();
    // every lifted point satisfies L(X) = 1/2
    let xx = e.lift(&vec![0.2; e.model.num_vars()]);
    assert!((lx.evaluate(&xx) - Complex::new(0.5, 0.0)).norm() < 1e-12);
    // the sparse-QR elimination (used by SdpModel::solve) agrees: same number
    // of free variables, and its lifted points satisfy the constraint too
    let eq = m.eliminate_qr(1e-9).unwrap();
    assert_eq!(eq.model.num_vars(), sdp.num_vars());
    let xq = eq.lift(&vec![0.2; eq.model.num_vars()]);
    assert!((lx.evaluate(&xq) - Complex::new(0.5, 0.0)).norm() < 1e-12);
    assert!(m.equality_residuals(&xq).iter().all(|r| r.abs() < 1e-12));
    // the eliminated LMI form converts to a model too
    let lm = sdp.to_model();
    assert_eq!(lm.num_vars(), 2);
    assert_eq!(lm.lmis().len(), 2);
    assert!(lm.equalities().is_empty());
    // inconsistent constraint: L(1) = 2
    let bad = Relaxation::level(red, 1).minimize(&h).moment_equality(
        &one,
        Complex::new(Q::from(ncalg::BigInt::from(2)), Q::from(ncalg::BigInt::from(0))),
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bad.build())).is_err());
}

#[test]
fn moment_constraint_with_symmetry() {
    // two commuting variables with the swap symmetry; constrain L(x + y) = 1
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Q> = alg.parse("x^2 + y^2").unwrap();
    let s: Polynomial<Q> = alg.parse("x + y").unwrap();
    let plain = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&s, Q::from(ncalg::BigInt::from(1)))
        .build();
    let sym = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&s, Q::from(ncalg::BigInt::from(1)))
        .symmetry(vec![vec![1, 0]])
        .build();
    assert!(sym.num_vars() < plain.num_vars());
    let x: Polynomial<Q> = alg.parse("x").unwrap();
    for z in [vec![0.0; sym.num_vars()], vec![0.4; sym.num_vars()]] {
        assert!((sym.evaluate(&z, red, &s).unwrap().re - 1.0).abs() < 1e-12);
        assert!((sym.evaluate(&z, red, &x).unwrap().re - 0.5).abs() < 1e-12);
    }
}

#[test]
fn eliminated_moment_form_has_the_same_dimension_as_build() {
    let (alg, rels) = relations::pauli::<Complex<Q>>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<Q>> = alg.parse("X0*X1 + Z0*Z1").unwrap();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let built = rel.build();
    let m = rel.build_moment_form().to_model();
    let e = m.eliminate().unwrap();
    assert_eq!(e.model.num_vars(), built.num_vars());
    assert_eq!(e.model.block_sizes(), built.block_sizes());
    // the sparse-QR elimination gives the same dimension and blocks
    let eq = m.eliminate_qr(1e-9).unwrap();
    assert_eq!(eq.model.num_vars(), built.num_vars());
    assert_eq!(eq.model.block_sizes(), built.block_sizes());
    let xq = eq.lift(&vec![0.3; eq.model.num_vars()]);
    assert!(m.equality_residuals(&xq).iter().all(|r| r.abs() < 1e-12));
}

/// Right-hand sides much larger than their rows.  The homogenised
/// elimination read the independent rows of `x_1 = x_2 = b` as dependent
/// from `b = 1e10` on (only `x_1 + x_2 = 2b` survived) and called them
/// inconsistent at `1e12`.
#[test]
fn elimination_with_large_right_hand_sides() {
    use ncalg::sdp::SdpModel;
    for e in [0, 6, 10, 12, 14] {
        let b = 10f64.powi(e);
        let mut m = SdpModel::new();
        let x1 = m.free_var("x1");
        let x2 = m.free_var("x2");
        m.add_equality(vec![(x1, 1.0)], b);
        m.add_equality(vec![(x2, 1.0)], b);
        let el = m
            .eliminate_qr(1e-9)
            .unwrap_or_else(|err| panic!("b = 1e{e}: {err}"));
        assert_eq!(el.model.num_vars(), 0, "b = 1e{e}: both equalities kept");
        let x = el.lift(&[]);
        assert!(x.iter().all(|v| (v - b).abs() <= 1e-12 * b), "b = 1e{e}: {x:?}");
    }
}

/// `eliminate` judged consistency against an absolute 1e-8, so an
/// exactly consistent dependent system with right-hand sides ~1e8 was
/// rejected as inconsistent.
#[test]
fn row_reduction_with_large_right_hand_sides() {
    use ncalg::sdp::SdpModel;
    for e in [0, 8, 10] {
        let b = 10f64.powi(e);
        let mut m = SdpModel::new();
        let x = m.free_vars(3, "x");
        m.add_equality(vec![(x[0], 0.3), (x[1], 0.7)], b);
        m.add_equality(vec![(x[1], 0.1), (x[2], 0.9)], 3.0 * b);
        // 7 × the first + 3 × the second
        m.add_equality(vec![(x[0], 2.1), (x[1], 5.2), (x[2], 2.7)], 16.0 * b);
        let el = m.eliminate().unwrap_or_else(|err| panic!("b = 1e{e}: {err}"));
        assert_eq!(el.model.num_vars(), 1);
    }
}
