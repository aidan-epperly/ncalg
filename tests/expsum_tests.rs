//! Exact exponential sums of phase polynomials over affine subspaces.

use ncalg::cyclo::expsum::{CliffordSum, PhasePolynomial};
use ncalg::cyclo::Cyclo;
use ncalg::{BigRational, Coeff};
use rand::{Rng, SeedableRng};

type Q16 = Cyclo<16>;
type Q8 = Cyclo<8>;

/// Σ_{t ∈ 𝔽₂^k} ζ^{f(a + Σ t_i b_i)} by enumeration, as Σ_e count(e) ζ^e.
fn brute<const N: u32>(f: &PhasePolynomial, a: &[u8], basis: &[Vec<u8>]) -> Cyclo<N> {
    let bits = f.modulus_bits();
    let m = 1u64 << bits;
    let mut counts = vec![0i64; m as usize];
    for t in 0u64..(1 << basis.len()) {
        let mut x = a.to_vec();
        for (i, b) in basis.iter().enumerate() {
            if t >> i & 1 == 1 {
                for (xj, bj) in x.iter_mut().zip(b) {
                    *xj ^= bj;
                }
            }
        }
        counts[f.eval(&x) as usize] += 1;
    }
    let mut s = Cyclo::<N>::zero();
    for (e, &c) in counts.iter().enumerate() {
        if c != 0 {
            let z = if bits == 0 {
                Cyclo::<N>::one()
            } else {
                Cyclo::<N>::zeta((N as i64 / m as i64) * e as i64)
            };
            s = s.add_ref(&z.mul_ref(&Cyclo::<N>::from_i64(c)));
        }
    }
    s
}

fn random_basis(rng: &mut rand::rngs::StdRng, n: usize, k: usize) -> Vec<Vec<u8>> {
    loop {
        let b: Vec<Vec<u8>> = (0..k)
            .map(|_| (0..n).map(|_| rng.gen_range(0..2)).collect())
            .collect();
        let mut m = ncalg::ff::BitMat::zeros(k, n);
        for (r, v) in b.iter().enumerate() {
            for (c, &x) in v.iter().enumerate() {
                m.set(r, c, x == 1);
            }
        }
        if m.rank() == k {
            return b;
        }
    }
}

/// A random polynomial with terms of degree ≤ `deg`; `clifford` keeps every
/// term at level ≤ 2.
fn random_poly(
    rng: &mut rand::rngs::StdRng,
    n: usize,
    bits: u32,
    terms: usize,
    deg: usize,
    clifford: bool,
) -> PhasePolynomial {
    let m = 1i64 << bits;
    let mut f = PhasePolynomial::new(n, bits);
    f.add_constant(rng.gen_range(0..m));
    for _ in 0..terms {
        let d = rng.gen_range(1..=deg);
        let vars: Vec<usize> = (0..d).map(|_| rng.gen_range(0..n)).collect();
        let mut c = rng.gen_range(0..m);
        if clifford {
            // level |S| + bits − v2(c) − 1 ≤ 2
            let need = (bits as i64 + 1 - vars.len() as i64).max(0) as u32;
            if vars.len() > 2 {
                continue;
            }
            c = (c << need) % m.max(1);
        }
        if rng.gen_bool(0.3) && n >= 2 {
            // a parity factor instead of a plain variable
            let forms: Vec<Vec<usize>> = vars.iter().map(|&v| vec![v, rng.gen_range(0..n)]).collect();
            let refs: Vec<&[usize]> = forms.iter().map(|v| v.as_slice()).collect();
            if clifford && forms.len() > 2 {
                continue;
            }
            f.add_parity_term(c, &refs);
        } else {
            f.add_monomial(c, &vars);
        }
    }
    if clifford {
        assert!(f.is_clifford());
    }
    f
}

#[test]
fn random_instances_against_brute_force() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for trial in 0..400 {
        let n = rng.gen_range(1..=9);
        let bits = rng.gen_range(0..=4);
        let clifford = trial % 3 == 0;
        let deg = if clifford { 2 } else { rng.gen_range(1..=3) };
        let nterms = rng.gen_range(0..12);
        let f = random_poly(&mut rng, n, bits, nterms, deg, clifford);
        let k = rng.gen_range(0..=n);
        let basis = random_basis(&mut rng, n, k);
        let a: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
        let want = brute::<16>(&f, &a, &basis);
        let got = f.exp_sum_over::<16>(&a, &basis).unwrap();
        assert_eq!(got, want, "trial {trial}: n = {n}, r = {bits}, k = {k}\n{f:?}");
        // restriction semantics
        let g = f.restrict(&a, &basis).unwrap();
        for _ in 0..4 {
            let t: Vec<u8> = (0..k).map(|_| rng.gen_range(0..2)).collect();
            let mut x = a.clone();
            for (i, b) in basis.iter().enumerate() {
                if t[i] == 1 {
                    for (xj, bj) in x.iter_mut().zip(b) {
                        *xj ^= bj;
                    }
                }
            }
            assert_eq!(g.eval(&t), f.eval(&x));
        }
        // the restriction of a Clifford polynomial stays Clifford, and the
        // closed form agrees: 0, or |S|² = 2^p with the stated phase
        if clifford {
            assert!(g.is_clifford());
            match g.clifford_sum().unwrap() {
                CliffordSum::Zero => assert!(want.is_zero()),
                CliffordSum::Value {
                    sqrt2_power,
                    phase,
                    phase_bits,
                } => {
                    let abs2 = want.mul_ref(&want.galois(-1)); // |S|²
                    let two = Q16::from_rational(BigRational::from_integer(
                        ncalg::BigInt::from(2u32).pow(sqrt2_power),
                    ));
                    assert_eq!(abs2, two, "|S|² = 2^p");
                    let r2 = Q16::sqrt2();
                    let mut mag = Q16::one();
                    for _ in 0..sqrt2_power {
                        mag = mag.mul_ref(&r2);
                    }
                    let z = Q16::zeta((16 >> phase_bits.min(4)) as i64 * phase as i64);
                    if phase_bits <= 4 {
                        assert_eq!(mag.mul_ref(&z), want);
                    }
                }
            }
        }
    }
}

#[test]
fn closed_forms() {
    // Σ_x i^{|x|} = (1+i)^n
    for n in [1, 5, 40] {
        let mut f = PhasePolynomial::new(n, 2);
        for j in 0..n {
            f.add_monomial(1, &[j]);
        }
        let mut want = Q8::one();
        let one_plus_i = Q8::one().add_ref(&Q8::zeta(2));
        for _ in 0..n {
            want = want.mul_ref(&one_plus_i);
        }
        assert_eq!(f.exp_sum::<8>(), want);
        assert_eq!(
            f.clifford_sum(),
            Some(CliffordSum::Value {
                sqrt2_power: n as u32,
                phase: n as u64 % 8,
                phase_bits: 3
            })
        );
    }
    // Σ (−1)^{Σ x_{2j} x_{2j+1}} = 2^{n/2} on n = 2m bits
    let m = 30;
    let mut f = PhasePolynomial::new(2 * m, 1);
    for j in 0..m {
        f.add_monomial(1, &[2 * j, 2 * j + 1]);
    }
    let two_m = ncalg::BigInt::from(2u32).pow(m as u32);
    assert_eq!(
        f.exp_sum::<4>(),
        Cyclo::<4>::from_rational(BigRational::from_integer(two_m))
    );
    // a contradictory constraint: Σ_{x_0} (−1)^{x_0} = 0
    let mut z = PhasePolynomial::new(3, 1);
    z.add_monomial(1, &[0]);
    assert_eq!(z.clifford_sum(), Some(CliffordSum::Zero));
    assert!(z.exp_sum::<2>().is_zero());
}

#[test]
fn product_type_is_linear_time() {
    // T on every qubit of |+>^n: (1 + ζ_8)^n, no couplings, width 1
    let n = 200;
    let mut f = PhasePolynomial::new(n, 3);
    for j in 0..n {
        f.add_monomial(1, &[j]);
    }
    assert!(!f.is_clifford());
    assert_eq!(f.level_bound(), 3);
    let plan = f.exp_sum_plan();
    assert_eq!((plan.core, plan.width), (n, 1));
    let base = Q8::one().add_ref(&Q8::zeta(1));
    let mut want = Q8::one();
    for _ in 0..n {
        want = want.mul_ref(&base);
    }
    assert_eq!(f.exp_sum::<8>(), want);
}

#[test]
fn cs_chain_against_transfer_matrix() {
    // Σ_x i^{Σ_j x_j x_{j+1}} ζ_8^{x_0}: a chain of CS gates with a T at one end
    let n = 60;
    let mut f = PhasePolynomial::new(n, 3);
    for j in 0..n - 1 {
        f.add_monomial(2, &[j, j + 1]);
    }
    f.add_monomial(1, &[0]);
    let plan = f.exp_sum_plan();
    assert!(plan.width <= 3, "{plan:?}");
    // transfer matrix: v_{j+1}(y) = Σ_x v_j(x) i^{x y}
    let i = Q8::zeta(2);
    let mut v = [Q8::one(), Q8::zeta(1)];
    for _ in 0..n - 1 {
        v = [v[0].add_ref(&v[1]), v[0].add_ref(&v[1].mul_ref(&i))];
    }
    assert_eq!(f.exp_sum::<8>(), v[0].add_ref(&v[1]));
}

/// A large instance: compare the sum over a + V with the sum of the sums over
/// the two cosets of a hyperplane of V, and with another basis of V.
fn check_large(f: &PhasePolynomial, rng: &mut rand::rngs::StdRng, k: usize) {
    let n = f.num_vars();
    let basis = random_basis(rng, n, k);
    let a: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
    let s = f.exp_sum_over::<8>(&a, &basis).unwrap();
    // cosets a + V' and a + b_0 + V'
    let rest = basis[1..].to_vec();
    let a1: Vec<u8> = a.iter().zip(&basis[0]).map(|(x, y)| x ^ y).collect();
    let split = f
        .exp_sum_over::<8>(&a, &rest)
        .unwrap()
        .add_ref(&f.exp_sum_over::<8>(&a1, &rest).unwrap());
    assert_eq!(s, split);
    // another basis: b_i + b_{i+1}, plus the last one
    let mut other: Vec<Vec<u8>> = (0..k - 1)
        .map(|i| basis[i].iter().zip(&basis[i + 1]).map(|(x, y)| x ^ y).collect())
        .collect();
    other.push(basis[k - 1].clone());
    assert_eq!(f.exp_sum_over::<8>(&a, &other).unwrap(), s);
}

#[test]
fn large_clifford_instance() {
    // S or S† on every variable, CZ on random pairs, S on random parities: a
    // nonzero sum (random S/Z coefficients usually vanish after a few steps)
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let n = 300;
    let mut f = PhasePolynomial::new(n, 3);
    for j in 0..n {
        f.add_monomial(if rng.gen_bool(0.5) { 2 } else { 6 }, &[j]);
    }
    for _ in 0..600 {
        f.add_monomial(4, &[rng.gen_range(0..n), rng.gen_range(0..n)]);
    }
    for _ in 0..100 {
        let form: Vec<usize> = (0..4).map(|_| rng.gen_range(0..n)).collect();
        f.add_parity_term(2, &[&form]);
    }
    assert!(f.is_clifford());
    let t = std::time::Instant::now();
    assert!(matches!(f.clifford_sum(), Some(CliffordSum::Value { .. })));
    assert!(!f.exp_sum::<8>().is_zero());
    check_large(&f, &mut rng, 250);
    let plan = f.exp_sum_plan();
    assert_eq!((plan.core, plan.width, plan.clifford_eliminated), (0, 0, n));
    assert!(t.elapsed().as_secs_f64() < 30.0, "{:?}", t.elapsed());
}

#[test]
fn cs_ring_and_grid() {
    // Σ_x i^{Σ_j x_j x_{j+1 mod n}} = Tr(T^n), T = [[1, 1], [1, i]]
    let n = 40;
    let mut f = PhasePolynomial::new(n, 3);
    for j in 0..n {
        f.add_monomial(2, &[j, (j + 1) % n]);
    }
    assert!(f.exp_sum_plan().width <= 3);
    let i = Q8::zeta(2);
    let mul = |x: &[[Q8; 2]; 2], y: &[[Q8; 2]; 2]| -> [[Q8; 2]; 2] {
        let e = |r: usize, c: usize| x[r][0].mul_ref(&y[0][c]).add_ref(&x[r][1].mul_ref(&y[1][c]));
        [[e(0, 0), e(0, 1)], [e(1, 0), e(1, 1)]]
    };
    let t = [[Q8::one(), Q8::one()], [Q8::one(), i]];
    let mut p = [[Q8::one(), Q8::zero()], [Q8::zero(), Q8::one()]];
    for _ in 0..n {
        p = mul(&p, &t);
    }
    assert_eq!(f.exp_sum::<8>(), p[0][0].add_ref(&p[1][1]));
    // CS + T on a 4 x 5 grid, against enumeration
    let (w, l) = (4usize, 5usize);
    let mut g = PhasePolynomial::new(w * l, 3);
    for r in 0..l {
        for c in 0..w {
            let v = r * w + c;
            if c + 1 < w {
                g.add_monomial(2, &[v, v + 1]);
            }
            if r + 1 < l {
                g.add_monomial(2, &[v, v + w]);
            }
            g.add_monomial(1, &[v]);
        }
    }
    let basis: Vec<Vec<u8>> = (0..w * l)
        .map(|j| (0..w * l).map(|k| (j == k) as u8).collect())
        .collect();
    let zero = vec![0u8; w * l];
    assert!(g.exp_sum_plan().width <= w + 2);
    assert_eq!(g.exp_sum::<8>(), brute::<8>(&g, &zero, &basis));
}

#[test]
fn low_rank_non_clifford_instance() {
    // a Clifford bulk on 120 variables (S or S† everywhere, random CZ) with three
    // T gates on parities; the first seed whose sum does not vanish and keeps a
    // core (Clifford constraints can fix a T parity, which then becomes a constant)
    let n = 120;
    let build = |seed: u64| -> (PhasePolynomial, rand::rngs::StdRng) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut f = PhasePolynomial::new(n, 3);
        for j in 0..n {
            f.add_monomial(if rng.gen_bool(0.5) { 2 } else { 6 }, &[j]);
        }
        for _ in 0..90 {
            f.add_monomial(4, &[rng.gen_range(0..n), rng.gen_range(0..n)]);
        }
        for _ in 0..3 {
            let form: Vec<usize> = (0..5).map(|_| rng.gen_range(0..n)).collect();
            f.add_parity_term(1, &[&form]);
        }
        (f, rng)
    };
    let (mut f, mut rng) = (0..100)
        .map(build)
        .find(|(f, _)| {
            let p = f.exp_sum_plan();
            !p.zero && p.core >= 2
        })
        .expect("a nonzero instance with a core");
    let plan = f.exp_sum_plan();
    assert!(plan.core >= 1 && plan.core <= 3 && plan.width <= 3, "{plan:?}");
    assert!(!f.exp_sum::<8>().is_zero());
    check_large(&f, &mut rng, 100);
    // and a CCZ on parities
    let forms: Vec<Vec<usize>> = (0..3)
        .map(|_| (0..4).map(|_| rng.gen_range(0..n)).collect())
        .collect();
    let refs: Vec<&[usize]> = forms.iter().map(|v| v.as_slice()).collect();
    f.add_parity_term(4, &refs);
    assert!(f.exp_sum_plan().core <= 6);
    check_large(&f, &mut rng, 90);
}

#[test]
fn levels_and_errors() {
    let mut f = PhasePolynomial::new(3, 3);
    assert_eq!(f.level_bound(), 0);
    f.add_monomial(4, &[0]); // Z
    assert_eq!(f.level_bound(), 1);
    f.add_monomial(2, &[1]); // S
    f.add_monomial(4, &[0, 1]); // CZ
    assert_eq!(f.level_bound(), 2);
    assert!(f.is_clifford());
    f.add_monomial(2, &[1, 2]); // CS
    assert_eq!(f.level_bound(), 3);
    assert!(f.clifford_sum().is_none());
    let mut g = PhasePolynomial::new(3, 3);
    g.add_monomial(4, &[0, 1, 2]); // CCZ
    assert_eq!(g.level_bound(), 3);
    // a dependent basis is refused by exp_sum_over
    assert!(f
        .exp_sum_over::<8>(&[0, 0, 0], &[vec![1, 0, 0], vec![1, 0, 0]])
        .is_err());
    assert!(f.restrict(&[0, 0], &[]).is_err());
}

#[test]
#[should_panic(expected = "multiple of 8")]
fn modulus_must_divide_n() {
    let mut f = PhasePolynomial::new(1, 3);
    f.add_monomial(1, &[0]);
    let _ = f.exp_sum::<4>();
}

#[test]
fn heavier_random_instances() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(77);
    for trial in 0..250 {
        let n = rng.gen_range(1..=11);
        let bits = rng.gen_range(1..=6);
        let deg = rng.gen_range(1..=4);
        let nterms = rng.gen_range(0..16);
        let f = random_poly(&mut rng, n, bits, nterms, deg, false);
        let k = rng.gen_range(0..=n);
        let basis = random_basis(&mut rng, n, k);
        let a: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
        assert_eq!(
            f.exp_sum_over::<64>(&a, &basis).unwrap(),
            brute::<64>(&f, &a, &basis),
            "trial {trial}: n = {n}, r = {bits}\n{f:?}"
        );
    }
    // r ≤ 1: the sum is an integer, so any N works
    for trial in 0..40 {
        let n = rng.gen_range(1..=8);
        let bits = rng.gen_range(0..=1);
        let f = random_poly(&mut rng, n, bits, 10, 3, false);
        let all: Vec<Vec<u8>> = (0..n).map(|j| (0..n).map(|k| (j == k) as u8).collect()).collect();
        let zero = vec![0u8; n];
        let want = brute::<4>(&f, &zero, &all);
        let got3 = f.exp_sum::<3>();
        assert_eq!(got3.coeffs()[0], want.coeffs()[0], "trial {trial}");
        assert!(want.coeffs()[1..]
            .iter()
            .all(|c| *c == BigRational::from_integer(0.into())));
    }
}
