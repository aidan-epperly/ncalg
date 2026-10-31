//! Additional coverage: other orderings, solver API, expression trees with
//! partial bases, conversions.

use ncalg::relations;
use ncalg::{
    BigRational, DegRevLex, Expr, FreeAlgebra, GroebnerBasis, GroebnerOptions, GroebnerSolver, Polynomial,
    QuotientAlgebra, Termination, WeightedDegLex, Word,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type Q = BigRational;

fn rat(alg: &FreeAlgebra, s: &str) -> Polynomial<Q> {
    alg.parse(s).unwrap()
}

fn random_homog(rng: &mut StdRng, alg: &FreeAlgebra, deg: usize, n_terms: usize) -> Polynomial<Q> {
    let n = alg.num_vars() as u32;
    let terms = (0..n_terms).map(|_| {
        let w: Vec<u32> = (0..deg).map(|_| rng.gen_range(0..n)).collect();
        let c = rng.gen_range(-3i64..=3);
        (Word::from_vec(w), ncalg::Coeff::from_i64(c))
    });
    Polynomial::from_terms(alg.clone(), terms)
}

#[test]
fn commutative_relations_under_other_orderings() {
    for (name, alg) in [
        (
            "degrevlex",
            FreeAlgebra::with_order(["x", "y", "z"], DegRevLex::new()),
        ),
        (
            "weighted",
            FreeAlgebra::with_order(["x", "y", "z"], WeightedDegLex::new(vec![1, 2, 3])),
        ),
        (
            "chain",
            FreeAlgebra::with_order(["x", "y", "z"], ncalg::DegLex::with_chain(&[2, 0, 1])),
        ),
    ] {
        let rels = relations::commutative::<Q>(&alg);
        let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
        assert!(gb.is_complete(), "{}", name);
        assert_eq!(gb.len(), 3, "{}", name);
        assert!(gb.verify(None).is_empty(), "{}", name);
        let q = QuotientAlgebra::new(gb);
        assert_eq!(q.hilbert_series(3), vec![1, 3, 6, 10], "{}", name);
    }
}

#[test]
fn chain_criterion_three_variables() {
    let mut rng = StdRng::seed_from_u64(99);
    let alg = FreeAlgebra::new(["x", "y", "z"]);
    for trial in 0..6 {
        let gens: Vec<Polynomial<Q>> = (0..2)
            .map(|_| random_homog(&mut rng, &alg, 2, 3))
            .filter(|p| !p.is_zero())
            .collect();
        if gens.is_empty() {
            continue;
        }
        let a = GroebnerBasis::compute(gens.clone(), &GroebnerOptions::new().max_degree(6));
        let b = GroebnerBasis::compute(
            gens.clone(),
            &GroebnerOptions::new().max_degree(6).chain_criterion(false),
        );
        assert_eq!(a.elements(), b.elements(), "trial {}", trial);
        assert!(a.verify(Some(6)).is_empty(), "trial {}", trial);
    }
}

#[test]
fn solver_add_generators_and_resume() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let mut solver = GroebnerSolver::new_in(&alg, vec![rat(&alg, "y*x - x*y")], GroebnerOptions::default());
    assert_eq!(*solver.status(), Termination::NotStarted);
    assert!(solver.run().is_complete());
    assert_eq!(solver.basis().len(), 1);
    solver.add_generators(vec![rat(&alg, "x^2 - y")]);
    assert_eq!(*solver.status(), Termination::NotStarted);
    assert!(solver.run().is_complete());
    let gb = solver.basis();
    // K[x, y]/(x^2 - y) ≅ K[x]: one normal word per degree... up to x
    assert!(gb.contains(&rat(&alg, "x^2*y - y^2")));
    let q = QuotientAlgebra::new(gb);
    assert_eq!(q.hilbert_series(4), vec![1, 2, 2, 2, 2]);
    assert!(solver.active_elements().len() >= 2);
    assert_eq!(solver.pending_pairs(), 0);
    assert!(solver.min_pending_degree().is_none());
}

#[test]
fn expression_tree_with_partial_basis_is_consistent() {
    // homogeneous ideal, truncated basis: lazy and eager evaluation agree for
    // expressions of degree within the bound
    let alg = FreeAlgebra::new(["x", "y"]);
    let gens = vec![rat(&alg, "x*y*x - y*x*y")];
    let gb = GroebnerBasis::compute(gens, &GroebnerOptions::new().max_degree(8));
    assert!(!gb.is_complete());
    let x: Expr<Q> = rat(&alg, "x").into();
    let y: Expr<Q> = rat(&alg, "y").into();
    let e = (x.clone() * y.clone() + y.clone() * x.clone()).pow(3)
        + Expr::commutator(x.clone(), y.clone()).pow(2);
    let lazy = e.eval_mod(gb.reducer());
    let eager = gb.reduce(&e.eval_free());
    assert_eq!(lazy, eager);
    assert!(lazy.degree().unwrap() <= 6);
}

#[test]
fn groebner_basis_conversion_and_from_elements() {
    let (alg, rels) = relations::weyl::<Q>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let gbf = gb.convert::<f64>();
    assert!(gbf.is_complete());
    let p: Polynomial<f64> = alg.parse("d1*x1").unwrap();
    assert_eq!(gbf.reduce(&p), alg.parse("x1*d1 + 1").unwrap());
    // wrapping known elements
    let manual = GroebnerBasis::from_elements(
        &alg,
        vec![rat(&alg, "2*d1*x1 - 2*x1*d1 - 2")],
        Termination::Complete,
    );
    assert_eq!(manual.elements(), gb.elements());
    assert!(manual.verify(None).is_empty());
    // a non-Gröbner set is detected by verify
    let bad = GroebnerBasis::from_elements(
        &alg,
        vec![rat(&alg, "d1*x1*x1 - 1"), rat(&alg, "x1*d1*x1 - 2")],
        Termination::Complete,
    );
    assert!(!bad.verify(None).is_empty());
}

#[test]
fn termination_display() {
    assert_eq!(Termination::Complete.to_string(), "complete");
    let t = Termination::DegreeBound {
        bound: 4,
        smallest_pending: 5,
    };
    assert!(t.to_string().contains("degree bound 4"));
    assert!(!t.is_complete());
}

#[test]
fn quotient_element_misc() {
    let (alg, rels) = relations::so3::<Q>();
    let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
    let x = q.gen("x");
    let y = q.gen("y");
    let z = q.gen("z");
    assert_eq!(x.commutator(&y), z);
    let casimir = &x * &x + &y * &y + &z * &z;
    assert!(casimir.commutes_with(&x) && casimir.commutes_with(&y) && casimir.commutes_with(&z));
    let mut acc = q.zero();
    acc += &x;
    acc -= &y;
    acc *= &z;
    assert_eq!(acc, (&x - &y) * &z);
    assert_eq!(acc.quotient(), &q);
    let p = acc.clone().into_polynomial();
    assert_eq!(q.element(&p), acc);
    assert_eq!(format!("{:?}", q.one()), "[1]");
    assert_eq!(q.parse("[x, y] - z").unwrap(), q.zero());
    let _ = alg;
}

/// Minimal 2x2 complex matrices for evaluation tests.
#[derive(Clone, Debug, PartialEq)]
struct M2([ncalg::Complex<f64>; 4]);

impl std::ops::Add for M2 {
    type Output = M2;
    fn add(self, o: M2) -> M2 {
        M2([
            self.0[0] + o.0[0],
            self.0[1] + o.0[1],
            self.0[2] + o.0[2],
            self.0[3] + o.0[3],
        ])
    }
}
impl std::ops::Mul for M2 {
    type Output = M2;
    fn mul(self, o: M2) -> M2 {
        let a = &self.0;
        let b = &o.0;
        M2([
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3],
        ])
    }
}

#[test]
fn evaluate_in_matrices() {
    use ncalg::Complex;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let zero = M2([c(0.0, 0.0); 4]);
    let id = M2([c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]);
    let sx = M2([c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]);
    let sy = M2([c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]);
    let sz = M2([c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)]);
    let (alg, rels) = relations::pauli::<Complex<f64>>(1);
    let mats = [sx, sy, sz];
    let embed = |k: &Complex<f64>| M2([*k, c(0.0, 0.0), c(0.0, 0.0), *k]);
    // every defining relation vanishes on the Pauli matrices
    for r in &rels {
        let m = r.evaluate(|v| mats[v as usize].clone(), embed, zero.clone(), id.clone());
        assert_eq!(m, zero, "{}", r);
    }
    // and X Y Z = i
    let xyz: Polynomial<Complex<f64>> = alg.parse("X0*Y0*Z0").unwrap();
    let m = xyz.evaluate(|v| mats[v as usize].clone(), embed, zero.clone(), id.clone());
    assert_eq!(m, embed(&c(0.0, 1.0)));
}

#[test]
fn prime_field_coefficients() {
    use ncalg::Fp;
    type F = Fp<32003>;
    // Fp<P> = Fq<P, 1>: residues from signed integers via from_i64, `new` takes the residue itself
    let a = F::from_i64(5);
    let b = F::from_i64(-7);
    assert_eq!(a.add_ref(&b), F::from_i64(-2));
    assert_eq!(a.mul_ref(&a.inverse()), F::one());
    assert_eq!(
        F::parse_literal("1/2").unwrap().mul_ref(&F::from_i64(2)),
        F::one()
    );
    assert_eq!(F::from_i64(32003), F::zero());
    assert_eq!(F::new(32002), F::from_i64(-1));
    assert_eq!(F::from_i64(-1).to_string(), "-1");
    assert_eq!(F::from_i64(-1).signed_value(), -1);

    // Gröbner basis over F_p has the same leading words as over Q here
    let alg = FreeAlgebra::new(["x", "y"]);
    let gens_q = vec![rat(&alg, "x*y*x - y*x*y + 3*x^3"), rat(&alg, "x^2*y - 2*y*x^2")];
    let gens_p: Vec<Polynomial<F>> = gens_q
        .iter()
        .map(|g| g.map_coeffs(|c| F::parse_literal(&c.to_string()).unwrap()))
        .collect();
    let gq = GroebnerBasis::compute(gens_q, &GroebnerOptions::new().max_degree(8));
    let gp = GroebnerBasis::compute(gens_p, &GroebnerOptions::new().max_degree(8));
    assert_eq!(gq.leading_words(), gp.leading_words());
    assert_eq!(gq.termination(), gp.termination());
    let qp = QuotientAlgebra::new(gp);
    let qq = QuotientAlgebra::new(gq);
    assert_eq!(qp.hilbert_series(8), qq.hilbert_series(8));
    use ncalg::{Coeff, Field};
}

#[test]
fn star_closure() {
    // Pauli and Majorana relations are closed under the adjoint; the Weyl
    // relation y x - x y - 1 is not (its adjoint is -(y x - x y + 1)).
    let (_, rels) = relations::pauli::<ncalg::Complex<Q>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    assert!(gb.is_star_closed());
    let (_, rels) = relations::majorana::<Q>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    assert!(gb.is_star_closed());
    let (_, rels) = relations::weyl::<Q>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    assert!(!gb.is_star_closed());
}

#[test]
fn core_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FreeAlgebra>();
    assert_send_sync::<Polynomial<Q>>();
    assert_send_sync::<Polynomial<ncalg::Complex<f64>>>();
    assert_send_sync::<GroebnerBasis<Q>>();
    assert_send_sync::<GroebnerSolver<f64>>();
    assert_send_sync::<QuotientAlgebra<Q>>();
    assert_send_sync::<Expr<Q>>();
    // a basis computed on one thread is usable from another
    let (_, rels) = relations::sl2::<Q>();
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let handle = std::thread::spawn(move || gb.len());
    assert_eq!(handle.join().unwrap(), 3);
}

#[test]
fn latex_output() {
    let alg = FreeAlgebra::new(["x_1", "y"]);
    let p: Polynomial<Q> = alg.parse("x_1^2*y - 1/2*y*x_1 + 3").unwrap();
    assert_eq!(p.to_latex(), "x_1^{2} y - \\frac{1}{2} y x_1 + 3");
    let c: Polynomial<ncalg::Complex<f64>> = alg.parse("(1+2i)*x_1 - i*y").unwrap();
    assert_eq!(c.to_latex(), "-i y + (1+2 i) x_1");
    assert_eq!(alg.zero::<Q>().to_latex(), "0");
}

#[test]
fn save_and_load_groebner_bases_and_polynomials() {
    use ncalg::io;
    let dir = std::env::temp_dir().join("ncalg_io_test");
    std::fs::create_dir_all(&dir).unwrap();
    // exact basis with a custom ordering round-trips exactly
    let alg = FreeAlgebra::with_order(["x", "y", "z"], WeightedDegLex::new(vec![1, 2, 3]));
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let path = dir.join("commutative.ncalg");
    gb.save(&path).unwrap();
    let loaded = GroebnerBasis::<Q>::load(&path).unwrap();
    assert!(loaded.algebra().same_as(&alg));
    assert_eq!(loaded.elements(), gb.elements());
    assert_eq!(loaded.termination(), gb.termination());
    // wrong coefficient type is refused, explicit conversion works
    assert!(GroebnerBasis::<f64>::load(&path).is_err());
    let as_float = GroebnerBasis::<f64>::load_as(&path).unwrap();
    assert_eq!(as_float.len(), gb.len());
    // partial basis keeps its status; involution and complex coefficients survive
    let alg2 = FreeAlgebra::new_star([("a", false), ("n", true)]);
    let rels2: Vec<Polynomial<ncalg::Complex<Q>>> = vec![
        alg2.parse("a*a† - a†*a - 1").unwrap(),
        alg2.parse("n - a†*a").unwrap(),
    ];
    let gb2 = GroebnerBasis::compute(rels2.clone(), &GroebnerOptions::new().max_degree(3));
    let path2 = dir.join("boson.ncalg");
    gb2.save(&path2).unwrap();
    let loaded2 = GroebnerBasis::<ncalg::Complex<Q>>::load(&path2).unwrap();
    assert_eq!(loaded2.algebra().adjoint_map(), alg2.adjoint_map());
    assert_eq!(loaded2.elements(), gb2.elements());
    assert_eq!(loaded2.termination(), gb2.termination());
    let p3 = dir.join("rels.ncalg");
    io::save_polynomials(&p3, &alg2, &rels2).unwrap();
    let (alg3, back) = io::load_polynomials::<ncalg::Complex<Q>>(&p3).unwrap();
    assert!(alg3.same_as(&alg2));
    assert_eq!(back, rels2);
    let text = std::fs::read_to_string(&p3).unwrap();
    assert!(text.starts_with("ncalg v1\nkind: polynomials\ngenerators: a, a†, n\n"));
    assert!(text.contains("adjoint: 1 0 2"));
}

#[test]
fn derivation_is_zero_on_constants_and_reduces_degree() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let p: Polynomial<i64> = alg.parse("x^3*y + 7").unwrap();
    let d = p.derivation(0);
    assert_eq!(d, alg.parse("x^2*y + x^2*y + x^2*y").unwrap());
    assert_eq!(d.to_string(), "3*x^2*y");
    assert!(alg.parse::<i64>("5").unwrap().derivation(1).is_zero());
}
