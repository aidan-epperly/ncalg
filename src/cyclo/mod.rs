//! Exact cyclotomic numbers ℚ(ζ_N) as coefficients.
//!
//! [`Cyclo<N>`] stores `Σ_{j<φ(N)} c_j ζ^j` with `ζ = e^{2πi/N}` and
//! `c_j ∈ ℚ`, reduced modulo the cyclotomic polynomial `Φ_N`, so equality is
//! structural.  It implements [`Coeff`] and [`Field`] (inverse through the
//! Galois norm), and [`ComplexCoeff`] when `4 | N`.
//!
//! Real elements can be compared exactly ([`Cyclo::sign`], [`Cyclo::cmp_real`]):
//! a real `x = Σ c_j cos(2πj/N) = g(cos 2π/N)` with `g = Σ c_j T_j`
//! (Chebyshev).  `cos 2π/N` is isolated as a simple root of `U_{N-1}`
//! (whose roots are `cos kπ/N`) by rational bisection, until interval Horner
//! evaluation of `g` excludes zero.  All of it is rational arithmetic, so the
//! answer is certified.
//!
//! Literals: rationals, `i` (needs `4 | N`), and `#k` for `ζ^k`.  Elements
//! print as sums of `c*#k`, which parse back.
//!
//! [`expsum`]: exact exponential sums `Σ_{x ∈ a+V} ζ^{f(x)}` of phase
//! polynomials over affine subspaces of `𝔽₂^n` (Gauss sums), polynomial for the
//! Clifford class.
//!
//! ```
//! use ncalg::cyclo::Cyclo;
//! use ncalg::{Coeff, Field};
//! use num_rational::BigRational;
//! type Q8 = Cyclo<8>;
//! let r2 = Q8::sqrt2();
//! assert_eq!(r2.mul_ref(&r2), Q8::from_i64(2));
//! // cos²(π/8) = (2 + √2)/4 > 3/4, decided exactly
//! let c2 = Q8::from_i64(2).add_ref(&r2).mul_ref(&Q8::from_rational(BigRational::new(1.into(), 4.into())));
//! assert_eq!(c2.cmp_real(&Q8::from_rational(BigRational::new(3.into(), 4.into()))), std::cmp::Ordering::Greater);
//! ```

pub mod expsum;

use crate::coeff::{Coeff, ComplexCoeff, Field};
use num_bigint::BigInt;
use num_complex::Complex;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::{OnceLock, RwLock};

/// Per-`N` tables.
#[derive(Debug)]
pub struct CycloData {
    n: u32,
    phi: usize,
    /// `Φ_N`, ascending coefficients (monic, degree `phi`).
    cyclotomic: Vec<i64>,
    /// `red[k]` = `ζ^k` in the power basis, `k < N`.
    red: Vec<Vec<i64>>,
    /// units of ℤ/N other than 1
    galois: Vec<u32>,
}

fn poly_divexact(num: &[i64], den: &[i64]) -> Vec<i64> {
    // both ascending, den monic
    let mut r = num.to_vec();
    let dd = den.len() - 1;
    let mut q = vec![0i64; r.len() - dd];
    for k in (0..q.len()).rev() {
        let c = r[k + dd];
        q[k] = c;
        if c != 0 {
            for (i, &d) in den.iter().enumerate() {
                // checked: a wrapped product would give a wrong Φ_N silently
                r[k + i] = c
                    .checked_mul(d)
                    .and_then(|cd| r[k + i].checked_sub(cd))
                    .expect("cyclotomic polynomial coefficient overflows i64");
            }
        }
    }
    debug_assert!(r.iter().all(|&x| x == 0));
    q
}

fn cyclotomic_poly(n: u32, cache: &mut HashMap<u32, Vec<i64>>) -> Vec<i64> {
    if let Some(p) = cache.get(&n) {
        return p.clone();
    }
    // x^n - 1 = Π_{d | n} Φ_d
    let mut p = vec![0i64; n as usize + 1];
    p[0] = -1;
    p[n as usize] = 1;
    for d in 1..n {
        if n % d == 0 {
            let f = cyclotomic_poly(d, cache);
            p = poly_divexact(&p, &f);
        }
    }
    cache.insert(n, p.clone());
    p
}

impl CycloData {
    fn new(n: u32) -> CycloData {
        assert!(n >= 1, "N must be positive");
        let phi_poly = cyclotomic_poly(n, &mut HashMap::new());
        let phi = phi_poly.len() - 1;
        let mut red = Vec::with_capacity(n as usize);
        let mut cur = vec![0i64; phi];
        cur[0] = 1;
        for _ in 0..n {
            red.push(cur.clone());
            // multiply by x and reduce: x^phi = -Σ_{i<phi} Φ_i x^i
            let top = cur[phi - 1];
            for i in (1..phi).rev() {
                cur[i] = cur[i - 1];
            }
            cur[0] = 0;
            if top != 0 {
                for i in 0..phi {
                    cur[i] = cur[i]
                        .checked_sub(top.checked_mul(phi_poly[i]).expect("overflow"))
                        .expect("overflow");
                }
            }
        }
        let galois = (2..n.max(2)).filter(|&a| a.gcd(&n) == 1).collect();
        CycloData {
            n,
            phi,
            cyclotomic: phi_poly,
            red,
            galois,
        }
    }
    /// Euler's totient `φ(n)`: the degree of `ℚ(ζ_n)`.
    pub fn phi(&self) -> usize {
        self.phi
    }
    /// Coefficients of the cyclotomic polynomial `Φ_n`, low degree first.
    pub fn cyclotomic_polynomial(&self) -> &[i64] {
        &self.cyclotomic
    }
}

fn registry() -> &'static RwLock<HashMap<u32, &'static CycloData>> {
    static R: OnceLock<RwLock<HashMap<u32, &'static CycloData>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Process-wide tables for ℚ(ζ_N).
pub fn cyclo_data(n: u32) -> &'static CycloData {
    if let Some(d) = registry().read().expect("registry").get(&n) {
        return d;
    }
    let mut w = registry().write().expect("registry");
    if let Some(d) = w.get(&n) {
        return d;
    }
    let d: &'static CycloData = Box::leak(Box::new(CycloData::new(n)));
    w.insert(n, d);
    d
}

/// An element of ℚ(ζ_N), `ζ = e^{2πi/N}`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Cyclo<const N: u32> {
    c: Vec<BigRational>,
}

fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

impl<const N: u32> Cyclo<N> {
    #[inline]
    fn data() -> &'static CycloData {
        cyclo_data(N)
    }
    /// From power-basis coefficients (`len ≤ φ(N)`; longer inputs are reduced).
    pub fn from_coeffs(c: Vec<BigRational>) -> Self {
        let d = Self::data();
        let mut out = vec![BigRational::zero(); d.phi];
        for (k, x) in c.into_iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            for (o, &r) in out.iter_mut().zip(&d.red[k % d.n as usize]) {
                if r != 0 {
                    *o += &x * rat(r);
                }
            }
        }
        Cyclo { c: out }
    }
    /// Coefficients in the power basis `1, ζ, …, ζ^{φ(N)-1}`.
    pub fn coeffs(&self) -> &[BigRational] {
        &self.c
    }
    /// The rational number `q`.
    pub fn from_rational(q: BigRational) -> Self {
        let mut c = vec![BigRational::zero(); Self::data().phi];
        c[0] = q;
        Cyclo { c }
    }
    /// `ζ^k` (any integer `k`).
    pub fn zeta(k: i64) -> Self {
        let d = Self::data();
        let k = k.rem_euclid(d.n as i64) as usize;
        Cyclo {
            c: d.red[k].iter().map(|&x| rat(x)).collect(),
        }
    }
    /// `√2 = ζ_8 + ζ_8^{-1}`; needs `8 | N`.
    pub fn sqrt2() -> Self {
        assert!(N % 8 == 0, "√2 = ζ_8 + ζ_8^7 needs 8 | N");
        Self::zeta((N / 8) as i64).add_ref(&Self::zeta(-((N / 8) as i64)))
    }
    /// The Galois automorphism `ζ ↦ ζ^a` (`gcd(a, N) = 1`).
    pub fn galois(&self, a: i64) -> Self {
        let d = Self::data();
        assert_eq!(
            (a.rem_euclid(d.n as i64) as u32).gcd(&d.n),
            1,
            "a must be a unit mod N"
        );
        let mut out = vec![BigRational::zero(); d.phi];
        for (j, x) in self.c.iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            let k = (a as i128 * j as i128).rem_euclid(d.n as i128) as usize;
            for (o, &r) in out.iter_mut().zip(&d.red[k]) {
                if r != 0 {
                    *o += x * rat(r);
                }
            }
        }
        Cyclo { c: out }
    }
    /// The field norm to ℚ (product of all Galois conjugates).
    pub fn norm(&self) -> BigRational {
        let mut y = self.clone();
        for &a in &Self::data().galois {
            y = y.mul_ref(&self.galois(a as i64));
        }
        debug_assert!(y.c[1..].iter().all(|x| x.is_zero()));
        y.c[0].clone()
    }
    /// Is this rational?
    pub fn is_rational(&self) -> bool {
        self.c[1..].iter().all(|x| x.is_zero())
    }
    /// Is this a real number (fixed by complex conjugation)?
    pub fn is_real(&self) -> bool {
        self.conj() == *self
    }
    /// Real part `(x + x̄)/2`, exact.
    pub fn real_part(&self) -> Self {
        let half = BigRational::new(BigInt::one(), BigInt::from(2));
        let s = self.add_ref(&self.conj());
        Cyclo {
            c: s.c.into_iter().map(|x| x * &half).collect(),
        }
    }
    /// `|x|^2 = x x̄`, exact.
    pub fn abs_sq(&self) -> Self {
        self.mul_ref(&self.conj())
    }
    /// Floating-point value, with `ζ = e^{2πi/N}`.
    pub fn to_complex(&self) -> Complex<f64> {
        let n = N as f64;
        self.c
            .iter()
            .enumerate()
            .filter(|(_, x)| !x.is_zero())
            .map(|(j, x)| {
                let t = 2.0 * std::f64::consts::PI * j as f64 / n;
                Complex::new(t.cos(), t.sin()) * x.to_f64().unwrap_or(f64::NAN)
            })
            .sum()
    }

    /// Exact sign of a real element.  Panics if the element is not real.
    pub fn sign(&self) -> Ordering {
        assert!(self.is_real(), "sign of a non-real cyclotomic number");
        if self.is_zero() {
            return Ordering::Equal;
        }
        if self.is_rational() {
            return self.c[0].cmp(&BigRational::zero());
        }
        // x = Σ c_j cos(2πj/N) = g(y), y = cos(2π/N) ∈ (lo, hi)
        let g = chebyshev_combination(&self.c);
        let (mut lo, mut hi) = cos_isolating_interval(N);
        let u = chebyshev_u(N as usize - 1);
        let s_lo = poly_sign(&u, &lo);
        loop {
            let (a, b) = interval_horner(&g, &lo, &hi);
            if a.is_positive() {
                return Ordering::Greater;
            }
            if b.is_negative() {
                return Ordering::Less;
            }
            let mid = (&lo + &hi) / rat(2);
            let s = poly_sign(&u, &mid);
            if s == Ordering::Equal {
                lo = mid.clone();
                hi = mid;
            } else if s == s_lo {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    }
    /// Exact comparison of two real elements.
    pub fn cmp_real(&self, other: &Self) -> Ordering {
        self.sub_ref(other).sign()
    }
}

// ------------------------------------------------------------------ exact sign helpers

/// Ascending integer coefficients of `T_k`, `k ≤ m`.
fn chebyshev_t(m: usize) -> Vec<Vec<BigInt>> {
    let mut t: Vec<Vec<BigInt>> = vec![vec![BigInt::one()], vec![BigInt::zero(), BigInt::one()]];
    while t.len() <= m {
        let k = t.len();
        let mut next = vec![BigInt::zero(); k + 1];
        for (i, x) in t[k - 1].iter().enumerate() {
            next[i + 1] += x * 2;
        }
        for (i, x) in t[k - 2].iter().enumerate() {
            next[i] -= x;
        }
        t.push(next);
    }
    t.truncate(m + 1);
    t
}

/// Ascending integer coefficients of `U_m` (`U_m(cos θ) = sin((m+1)θ)/sin θ`).
fn chebyshev_u(m: usize) -> Vec<BigInt> {
    let mut a = vec![BigInt::one()];
    let mut b = vec![BigInt::zero(), BigInt::from(2)];
    if m == 0 {
        return a;
    }
    for _ in 1..m {
        let mut next = vec![BigInt::zero(); b.len() + 1];
        for (i, x) in b.iter().enumerate() {
            next[i + 1] += x * 2;
        }
        for (i, x) in a.iter().enumerate() {
            next[i] -= x;
        }
        a = b;
        b = next;
    }
    b
}

/// `Σ_j c_j T_j` as an ascending rational polynomial.
fn chebyshev_combination(c: &[BigRational]) -> Vec<BigRational> {
    let t = chebyshev_t(c.len().max(1) - 1);
    let mut g = vec![BigRational::zero(); c.len().max(1)];
    for (j, x) in c.iter().enumerate() {
        if x.is_zero() {
            continue;
        }
        for (i, tc) in t[j].iter().enumerate() {
            g[i] += x * BigRational::from_integer(tc.clone());
        }
    }
    g
}

fn poly_sign(p: &[BigInt], x: &BigRational) -> Ordering {
    let mut acc = BigRational::zero();
    for c in p.iter().rev() {
        acc = acc * x + BigRational::from_integer(c.clone());
    }
    acc.cmp(&BigRational::zero())
}

/// Rigorous enclosure of `p([lo, hi])` by interval Horner.
fn interval_horner(p: &[BigRational], lo: &BigRational, hi: &BigRational) -> (BigRational, BigRational) {
    let mut a = BigRational::zero();
    let mut b = BigRational::zero();
    for c in p.iter().rev() {
        let prods = [&a * lo, &a * hi, &b * lo, &b * hi];
        let mn = prods.iter().min().expect("nonempty").clone();
        let mx = prods.iter().max().expect("nonempty").clone();
        a = mn + c;
        b = mx + c;
    }
    (a, b)
}

/// Snap an `f64` to a nearby dyadic rational.
fn dyadic(x: f64) -> BigRational {
    let scale = 1i64 << 52;
    BigRational::new(
        BigInt::from((x * scale as f64).round() as i64),
        BigInt::from(scale),
    )
}

/// A rational interval containing `cos(2π/N)` and no other root of `U_{N-1}`,
/// certified: `N - 1` disjoint intervals around the floating-point roots
/// `cos(kπ/N)` each show a sign change of `U_{N-1}` (degree `N - 1`), so each
/// holds exactly one root.
fn cos_isolating_interval(n: u32) -> (BigRational, BigRational) {
    static CACHE: OnceLock<RwLock<HashMap<u32, (BigRational, BigRational)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(v) = cache.read().expect("cache").get(&n) {
        return v.clone();
    }
    assert!(n >= 3);
    let u = chebyshev_u(n as usize - 1);
    let pi = std::f64::consts::PI;
    let roots: Vec<f64> = (1..n).map(|k| (k as f64 * pi / n as f64).cos()).collect(); // decreasing
    let mut out = None;
    'attempt: for shrink in [4.0, 16.0, 64.0] {
        let mut ivs = Vec::new();
        for (k, &r) in roots.iter().enumerate() {
            let gap_up = if k == 0 { 1.0 - r } else { roots[k - 1] - r };
            let gap_dn = if k + 1 == roots.len() {
                r + 1.0
            } else {
                r - roots[k + 1]
            };
            let d = gap_up.min(gap_dn) / shrink;
            let (lo, hi) = (dyadic(r - d), dyadic(r + d));
            let (sl, sh) = (poly_sign(&u, &lo), poly_sign(&u, &hi));
            if sl == Ordering::Equal || sh == Ordering::Equal || sl == sh {
                continue 'attempt;
            }
            ivs.push((lo, hi));
        }
        // disjoint by construction (each half-width is below half the gap); keep k = 2
        out = Some(ivs[1].clone());
        break;
    }
    let v = out.expect("could not isolate cos(2π/N)");
    cache.write().expect("cache").insert(n, v.clone());
    v
}

// ------------------------------------------------------------------ Coeff

impl<const N: u32> fmt::Debug for Cyclo<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} in Q(zeta_{})", self, N)
    }
}

impl<const N: u32> fmt::Display for Cyclo<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let terms: Vec<(usize, &BigRational)> =
            self.c.iter().enumerate().filter(|(_, x)| !x.is_zero()).collect();
        if terms.is_empty() {
            return write!(f, "0");
        }
        for (i, (j, x)) in terms.iter().enumerate() {
            let neg = x.is_negative();
            let a = x.abs();
            if i == 0 {
                if neg {
                    write!(f, "-")?;
                }
            } else {
                write!(f, "{}", if neg { " - " } else { " + " })?;
            }
            match (*j, a.is_one()) {
                (0, _) => write!(f, "{}", a)?,
                (j, true) => write!(f, "#{}", j)?,
                (j, false) => write!(f, "{}*#{}", a, j)?,
            }
        }
        Ok(())
    }
}

impl<const N: u32> Coeff for Cyclo<N> {
    fn zero() -> Self {
        Cyclo {
            c: vec![BigRational::zero(); Self::data().phi],
        }
    }
    fn one() -> Self {
        Self::from_rational(BigRational::one())
    }
    fn is_zero(&self) -> bool {
        self.c.iter().all(|x| x.is_zero())
    }
    fn is_one(&self) -> bool {
        self.c[0].is_one() && self.is_rational()
    }
    fn add_ref(&self, o: &Self) -> Self {
        Cyclo {
            c: self.c.iter().zip(&o.c).map(|(a, b)| a + b).collect(),
        }
    }
    fn sub_ref(&self, o: &Self) -> Self {
        Cyclo {
            c: self.c.iter().zip(&o.c).map(|(a, b)| a - b).collect(),
        }
    }
    fn mul_ref(&self, o: &Self) -> Self {
        let d = Self::data();
        let phi = d.phi;
        let mut full = vec![BigRational::zero(); 2 * phi - 1];
        for (i, a) in self.c.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in o.c.iter().enumerate() {
                if !b.is_zero() {
                    full[i + j] += a * b;
                }
            }
        }
        let mut out: Vec<BigRational> = full[..phi].to_vec();
        for (k, x) in full.iter().enumerate().skip(phi) {
            if x.is_zero() {
                continue;
            }
            for (o, &r) in out.iter_mut().zip(&d.red[k % d.n as usize]) {
                if r != 0 {
                    *o += x * rat(r);
                }
            }
        }
        Cyclo { c: out }
    }
    fn negated(&self) -> Self {
        Cyclo {
            c: self.c.iter().map(|x| -x).collect(),
        }
    }
    fn from_i64(n: i64) -> Self {
        Self::from_rational(rat(n))
    }
    fn parse_literal(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix('#') {
            let k: i64 = rest.parse().ok()?;
            return Some(Self::zeta(k));
        }
        let z = <Complex<BigRational> as Coeff>::parse_literal(s)?;
        if z.im.is_zero() {
            return Some(Self::from_rational(z.re));
        }
        if N % 4 != 0 {
            return None;
        }
        Some(
            Self::from_rational(z.re)
                .add_ref(&Self::zeta((N / 4) as i64).mul_ref(&Self::from_rational(z.im))),
        )
    }
    fn conj(&self) -> Self {
        self.galois(-1)
    }
    fn is_negative_for_display(&self) -> bool {
        self.is_rational() && self.c[0].is_negative()
    }
    fn needs_parens_for_display(&self) -> bool {
        self.c.iter().filter(|x| !x.is_zero()).count() > 1
    }
    fn magnitude(&self) -> f64 {
        self.to_complex().norm()
    }
}

impl<const N: u32> Field for Cyclo<N> {
    fn inverse(&self) -> Self {
        assert!(!self.is_zero(), "inverse of zero");
        // x · Π_{σ ≠ 1} σ(x) = N(x) ∈ ℚ
        let mut y = Self::one();
        for &a in &Self::data().galois {
            y = y.mul_ref(&self.galois(a as i64));
        }
        let nm = self.mul_ref(&y).c[0].clone();
        Cyclo {
            c: y.c.into_iter().map(|x| x / &nm).collect(),
        }
    }
}

/// Needs `4 | N` (panics otherwise when `i` is requested).
impl<const N: u32> ComplexCoeff for Cyclo<N> {
    fn imaginary_unit() -> Self {
        assert!(N % 4 == 0, "Q(zeta_N) contains i only when 4 | N");
        Self::zeta((N / 4) as i64)
    }
    fn re_f64(&self) -> f64 {
        self.to_complex().re
    }
    fn im_f64(&self) -> f64 {
        self.to_complex().im
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn random<const N: u32>(rng: &mut rand::rngs::StdRng) -> Cyclo<N> {
        let phi = cyclo_data(N).phi();
        Cyclo::from_coeffs(
            (0..phi)
                .map(|_| BigRational::new(rng.gen_range(-5..6).into(), rng.gen_range(1..4).into()))
                .collect(),
        )
    }

    fn field_axioms<const N: u32>() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(N as u64);
        for _ in 0..20 {
            let (a, b) = (random::<N>(&mut rng), random::<N>(&mut rng));
            let (za, zb) = (a.to_complex(), b.to_complex());
            assert!((a.mul_ref(&b).to_complex() - za * zb).norm() < 1e-9);
            assert!((a.conj().to_complex() - za.conj()).norm() < 1e-9);
            if !a.is_zero() {
                assert!(a.mul_ref(&a.inverse()).is_one());
            }
            let ab = a.abs_sq();
            assert!(ab.is_real());
            assert!((ab.to_complex().re - za.norm_sqr()).abs() < 1e-9);
            assert_eq!(
                Cyclo::<N>::parse_literal(&format!("{}", Cyclo::<N>::zeta(1))).unwrap(),
                Cyclo::<N>::zeta(1)
            );
            // exact sign agrees with floating point when it is not close
            let r = a.real_part();
            let f = r.to_complex().re;
            if f.abs() > 1e-6 {
                assert_eq!(
                    r.sign(),
                    if f > 0.0 {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                );
            }
        }
        assert!(Cyclo::<N>::zeta(N as i64).is_one());
    }

    #[test]
    fn arithmetic_matches_floating_point() {
        field_axioms::<3>();
        field_axioms::<4>();
        field_axioms::<5>();
        field_axioms::<8>();
        field_axioms::<12>();
        field_axioms::<16>();
    }

    #[test]
    fn real_comparisons_are_certified() {
        type Q16 = Cyclo<16>;
        let q = |a: i64, b: i64| Q16::from_rational(BigRational::new(a.into(), b.into()));
        let w = Q16::zeta(2); // ζ_8
                              // cos(π/8) = (ζ_16 + ζ_16^{-1}) / 2 and cos²(π/8) = (2 + √2)/4
        let c = Q16::zeta(1).add_ref(&Q16::zeta(-1)).mul_ref(&q(1, 2));
        let r2 = w.add_ref(&w.conj());
        assert_eq!(c.mul_ref(&c), q(2, 1).add_ref(&r2).mul_ref(&q(1, 4)));
        // 2^{-1/2} < cos(π/8) < 1, and cos(π/8)^2 differs from 0.8535533905932737 by < 1e-12
        let inv_r2 = r2.inverse();
        assert_eq!(inv_r2.cmp_real(&c), Ordering::Less);
        assert_eq!(c.cmp_real(&Q16::one()), Ordering::Less);
        assert_eq!(
            c.mul_ref(&c)
                .sub_ref(&q(8535533905932737, 10_000_000_000_000_000))
                .abs_sq()
                .cmp_real(&q(1, 1_000_000_000_000_000_000)),
            Ordering::Less
        );
        // certified bounds 0.27 < 1 - cos^4(π/8) < 0.28
        let c4 = c.mul_ref(&c).mul_ref(&c.mul_ref(&c));
        assert_eq!(Q16::one().sub_ref(&c4).cmp_real(&q(27, 100)), Ordering::Greater);
        assert_eq!(Q16::one().sub_ref(&c4).cmp_real(&q(28, 100)), Ordering::Less);
        // a value that is exactly zero
        assert_eq!(r2.mul_ref(&r2).sub_ref(&q(2, 1)).sign(), Ordering::Equal);
    }

    #[test]
    fn polynomials_over_q_zeta8() {
        use crate::Polynomial;
        let alg = crate::FreeAlgebra::new(["x"]);
        let p: Polynomial<Cyclo<8>> = alg.parse("#1*x + i").unwrap();
        let q = p.mul_ref(&p);
        let s = format!("{}", q);
        let back: Polynomial<Cyclo<8>> = alg.parse(&s).unwrap();
        assert_eq!(back, q);
    }
}
