//! Dense exact linear algebra over a `Field`.

use ncalg::cyclo::Cyclo;
use ncalg::linalg::exact;
use ncalg::{BigRational, Coeff, Field, Fp};
use rand::{Rng, SeedableRng};

type F101 = Fp<101>;
type Q8 = Cyclo<8>;

fn random<F: Coeff>(
    rng: &mut rand::rngs::StdRng,
    rows: usize,
    cols: usize,
    entry: impl Fn(i64) -> F,
    zero_bias: f64,
) -> Vec<Vec<F>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    if rng.gen_bool(zero_bias) {
                        F::zero()
                    } else {
                        entry(rng.gen_range(-5..=5))
                    }
                })
                .collect()
        })
        .collect()
}

fn rank_deficient<F: Field>(
    rng: &mut rand::rngs::StdRng,
    n: usize,
    r: usize,
    entry: impl Fn(i64) -> F + Copy,
) -> Vec<Vec<F>> {
    let a = random(rng, n, r, entry, 0.2);
    let b = random(rng, r, n, entry, 0.2);
    exact::mul(&a, &b)
}

#[test]
fn prime_field_matches_fqmat() {
    let f = ncalg::ff::field(101, 1);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let to_u64 = |m: &[Vec<F101>]| -> Vec<Vec<u64>> {
        m.iter().map(|r| r.iter().map(|x| x.value()).collect()).collect()
    };
    for trial in 0..60 {
        let (rows, cols) = (rng.gen_range(1..7), rng.gen_range(1..7));
        let m = if trial % 3 == 0 && rows == cols {
            rank_deficient(&mut rng, rows, rows.saturating_sub(1).max(1), F101::from_i64)
        } else {
            random(&mut rng, rows, cols, F101::from_i64, 0.3)
        };
        let fq = ncalg::ff::FqMat::from_rows(f, &to_u64(&m));
        assert_eq!(exact::rank(&m), fq.rank());
        assert_eq!(to_u64(&exact::nullspace(&m)), fq.nullspace());
        let b: Vec<F101> = (0..rows).map(|_| F101::from_i64(rng.gen_range(0..101))).collect();
        let bu: Vec<u64> = b.iter().map(|x| x.value()).collect();
        assert_eq!(
            exact::solve(&m, &b).map(|x| x.iter().map(|v| v.value()).collect::<Vec<_>>()),
            fq.solve(&bu)
        );
        if rows == cols {
            assert_eq!(exact::det(&m).value(), fq.det());
            let inv = exact::inverse(&m);
            assert_eq!(
                inv.as_ref().map(|i| to_u64(i)),
                fq.inverse()
                    .map(|i| (0..rows).map(|r| i.row(r).to_vec()).collect())
            );
        }
    }
}

fn leibniz<F: Field>(m: &[Vec<F>]) -> F {
    fn perms(n: usize) -> Vec<Vec<usize>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for p in perms(n - 1) {
            for pos in 0..n {
                let mut q = p.clone();
                q.insert(pos, n - 1);
                out.push(q);
            }
        }
        out
    }
    let n = m.len();
    let mut total = F::zero();
    for p in perms(n) {
        let inversions = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .filter(|&(i, j)| p[i] > p[j])
            .count();
        let mut t = F::one();
        for (i, &pi) in p.iter().enumerate() {
            t = t.mul_ref(&m[i][pi]);
        }
        if inversions % 2 == 1 {
            t = t.negated();
        }
        total.add_assign_ref(&t);
    }
    total
}

#[test]
fn rationals_against_definitions() {
    let q = |x: i64| BigRational::from_integer(x.into());
    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    for n in 0..6 {
        for _ in 0..8 {
            let m = random(&mut rng, n, n, q, 0.25);
            let d = exact::det(&m);
            assert_eq!(d, leibniz(&m), "det n = {n}");
            match exact::inverse(&m) {
                Some(inv) => {
                    assert!(!d.is_zero());
                    let id = exact::mul(&inv, &m);
                    for (i, row) in id.iter().enumerate() {
                        for (j, x) in row.iter().enumerate() {
                            assert_eq!(*x, if i == j { q(1) } else { q(0) });
                        }
                    }
                }
                None => assert!(d.is_zero()),
            }
        }
        // a singular matrix: the null space has the complementary dimension
        if n >= 2 {
            let m = rank_deficient(&mut rng, n, n - 1, q);
            let ns = exact::nullspace(&m);
            assert_eq!(ns.len() + exact::rank(&m), n);
            for v in &ns {
                let col: Vec<Vec<BigRational>> = v.iter().map(|x| vec![x.clone()]).collect();
                assert!(exact::mul(&m, &col).iter().all(|r| r[0].is_zero()));
            }
            assert!(exact::det(&m).is_zero());
        }
    }
    // inconsistent system
    let m = vec![vec![q(1), q(1)], vec![q(2), q(2)]];
    assert!(exact::solve(&m, &[q(1), q(3)]).is_none());
    assert_eq!(exact::solve(&m, &[q(1), q(2)]), Some(vec![q(1), q(0)]));
}

#[test]
fn cyclotomic_determinants_multiply() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    // entries a + b ζ_8^k with small integers
    let entry = |rng: &mut rand::rngs::StdRng| -> Q8 {
        let a = Q8::from_i64(rng.gen_range(-2..=2));
        let b = Q8::from_i64(rng.gen_range(-2..=2)).mul_ref(&Q8::zeta(rng.gen_range(0..8)));
        a.add_ref(&b)
    };
    for n in 1..5 {
        for _ in 0..4 {
            let a: Vec<Vec<Q8>> = (0..n)
                .map(|_| (0..n).map(|_| entry(&mut rng)).collect())
                .collect();
            let b: Vec<Vec<Q8>> = (0..n)
                .map(|_| (0..n).map(|_| entry(&mut rng)).collect())
                .collect();
            let ab = exact::mul(&a, &b);
            assert_eq!(exact::det(&ab), exact::det(&a).mul_ref(&exact::det(&b)));
            assert_eq!(exact::det(&a), leibniz(&a));
            if let Some(inv) = exact::inverse(&a) {
                assert_eq!(exact::det(&inv).mul_ref(&exact::det(&a)), Q8::one());
            }
        }
    }
    // a unitary 2x2 block (Hadamard-like), |det| = 1 exactly
    let h = [vec![Q8::one(), Q8::one()], vec![Q8::one(), Q8::from_i64(-1)]];
    let r2 = Q8::sqrt2();
    let u: Vec<Vec<Q8>> = h
        .iter()
        .map(|r| r.iter().map(|x| x.div_ref(&r2)).collect())
        .collect();
    assert_eq!(exact::det(&u), Q8::from_i64(-1));
}

#[test]
#[should_panic(expected = "exact field")]
fn floats_are_refused() {
    let _ = exact::det(&[vec![1.0f64]]);
}

#[test]
fn inertia_is_congruence_invariant() {
    let q = |n: i64| BigRational::from_integer(n.into());
    // diag(2, -3, 0, 0, 5) and a zero-diagonal block [[0, 1], [1, 0]] (signature 0)
    let d = [2, -3, 0, 0, 5];
    let diag: Vec<Vec<BigRational>> = (0..5)
        .map(|i| (0..5).map(|j| q(if i == j { d[i] } else { 0 })).collect())
        .collect();
    assert_eq!(exact::inertia(&diag), (2, 1, 2));
    assert_eq!(exact::inertia(&[vec![q(0), q(1)], vec![q(1), q(0)]]), (1, 1, 0));
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    for _ in 0..20 {
        // P D Pᵀ for a random invertible P has the inertia of D
        let p = loop {
            let p = random(&mut rng, 5, 5, q, 0.3);
            if !exact::det(&p).is_zero() {
                break p;
            }
        };
        let pt: Vec<Vec<BigRational>> = (0..5)
            .map(|i| (0..5).map(|j| p[j][i].clone()).collect())
            .collect();
        let a = exact::mul(&exact::mul(&p, &diag), &pt);
        assert_eq!(exact::inertia(&a), (2, 1, 2));
    }
}
