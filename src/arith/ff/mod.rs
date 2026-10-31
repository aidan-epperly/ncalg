//! Finite fields GF(q), q = p^k, with dense and bit-packed linear algebra.
//!
//! * [`Gf`]: a runtime field object for any prime power `q = p^k` (`p < 2^32`,
//!   `q < 2^63`).
//!   * Elements are `u64` encodings `Σ a_i p^i`, where `a_i` is the
//!     coefficient of `x^i` in `F_p[x]/(f)`. For `p = 2` an element is a bit
//!     pattern and addition is XOR.
//!   * The modulus `f` is the lexicographically smallest monic irreducible
//!     polynomial of degree `k`, found by Rabin's test, or one you supply with
//!     [`Gf::with_modulus`] (for example a Conway polynomial).
//!   * When `q ≤ 2^22`, multiplication and inversion use exp/log tables to a
//!     primitive element, and addition for odd `p` uses Zech logarithms, so
//!     every operation is O(1). Above that, arithmetic is polynomial,
//!     carry-less for `p = 2`.
//! * [`FqMat`]: dense matrices over a [`Gf`]: row reduction, rank, null space,
//!   solve, inverse, determinant, product.
//! * [`BitMat`]: bit-packed GF(2) matrices: the same operations on `u64` words,
//!   plus [`BitMat::dependency`], the zero-sum finder.
//! * [`Fq<P, K>`]: a [`crate::Coeff`] + [`crate::Field`] type for GF(p^k), so
//!   polynomials and Gröbner bases work over any finite field. Its tables come
//!   from a process-wide registry built lazily for each `(P, K)`.

mod matrix;
pub use matrix::{BitEchelon, BitMat, FqMat};

use crate::coeff::{Coeff, Field};
use std::collections::HashMap;
use std::fmt;
use std::sync::{OnceLock, RwLock};

/// Fields up to this order get exp/log (and Zech) tables.
pub const TABLE_LIMIT: u64 = 1 << 22;

/// A finite field GF(p^k) with elements encoded as `u64` in `0..q`.
#[derive(Clone, Debug)]
pub struct Gf {
    p: u64,
    k: u32,
    q: u64,
    /// `f = x^k + Σ_{i<k} m_i x^i`; `modulus[i] = m_i` (empty when k = 1)
    modulus: Vec<u64>,
    /// `pw[i] = p^i`, i ≤ k
    pw: Vec<u64>,
    tables: Option<Tables>,
}

/// Two fields are equal when they have the same characteristic, degree and modulus.
impl PartialEq for Gf {
    fn eq(&self, o: &Self) -> bool {
        self.p == o.p && self.k == o.k && self.modulus == o.modulus
    }
}
impl Eq for Gf {}

#[derive(Clone, Debug)]
struct Tables {
    generator: u64,
    /// `exp[i] = g^i`, length `2(q-1)`
    exp: Vec<u32>,
    /// `log[a]`, `a ≠ 0`
    log: Vec<u32>,
    /// `zech[n] = log(1 + g^n)`, `u32::MAX` when `1 + g^n = 0` (odd p, k > 1 only)
    zech: Vec<u32>,
}

use crate::arith::int::{self, is_prime, prime_factors};

// ----------------------------------------------------------- polynomials over F_p (dense, low degree first)

fn poly_trim(a: &mut Vec<u64>) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

fn poly_mulmod(a: &[u64], b: &[u64], f: &[u64], p: u64) -> Vec<u64> {
    // f monic, degree d = f.len() - 1
    let d = f.len() - 1;
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![0u64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            r[i + j] = int::add_mod(r[i + j], int::mul_mod(x, y, p), p);
        }
    }
    poly_rem(&mut r, f, p);
    let _ = d;
    r
}

fn poly_rem(r: &mut Vec<u64>, f: &[u64], p: u64) {
    let d = f.len() - 1;
    while r.len() > d {
        let c = *r.last().expect("nonempty");
        let top = r.len() - 1;
        if c != 0 {
            for i in 0..d {
                let idx = top - d + i;
                r[idx] = int::sub_mod(r[idx], int::mul_mod(c, f[i], p), p);
            }
        }
        r.pop();
    }
    poly_trim(r);
}

fn poly_sub(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut r: Vec<u64> = (0..n)
        .map(|i| (a.get(i).copied().unwrap_or(0) + p - b.get(i).copied().unwrap_or(0)) % p)
        .collect();
    poly_trim(&mut r);
    r
}

fn inv_mod_p(a: u64, p: u64) -> u64 {
    int::inv_mod(a, p).expect("nonzero element of a prime field")
}

fn poly_gcd(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    poly_trim(&mut a);
    poly_trim(&mut b);
    while !b.is_empty() {
        // a mod b with b made monic
        let lc = *b.last().expect("nonempty");
        let inv = inv_mod_p(lc, p);
        let bm: Vec<u64> = b.iter().map(|&c| int::mul_mod(c, inv, p)).collect();
        poly_rem(&mut a, &bm, p);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// `x^(p^e) mod f` by repeated p-th powering.
fn x_pow_p_pow(f: &[u64], p: u64, e: u32) -> Vec<u64> {
    let mut r = vec![0, 1];
    poly_rem(&mut r, f, p);
    for _ in 0..e {
        r = poly_powmod(&r, p, f, p);
    }
    r
}

fn poly_powmod(a: &[u64], mut e: u64, f: &[u64], p: u64) -> Vec<u64> {
    let mut base = a.to_vec();
    let mut acc = vec![1u64];
    while e > 0 {
        if e & 1 == 1 {
            acc = poly_mulmod(&acc, &base, f, p);
        }
        base = poly_mulmod(&base, &base, f, p);
        e >>= 1;
    }
    acc
}

/// Rabin's irreducibility test for monic `f` of degree k over F_p.
fn is_irreducible(f: &[u64], p: u64) -> bool {
    let k = (f.len() - 1) as u32;
    if k == 1 {
        return true;
    }
    let x = vec![0u64, 1];
    if poly_sub(&x_pow_p_pow(f, p, k), &x, p) != Vec::<u64>::new() {
        return false;
    }
    for r in prime_factors(k as u64) {
        let h = poly_sub(&x_pow_p_pow(f, p, k / r as u32), &x, p);
        let g = poly_gcd(f, &h, p);
        if g.len() != 1 {
            return false;
        }
    }
    true
}

impl Gf {
    /// GF(p^k) with the lexicographically smallest monic irreducible modulus
    /// (coefficients `m_0, …, m_{k-1}` read as base-p digits, smallest first).
    pub fn new(p: u64, k: u32) -> Result<Gf, String> {
        if !is_prime(p) || p >= 1 << 32 {
            return Err(format!("p = {p} must be a prime below 2^32"));
        }
        if k == 0 {
            return Err("k must be at least 1".into());
        }
        let q = p
            .checked_pow(k)
            .filter(|&q| q < 1 << 63)
            .ok_or("q = p^k must be below 2^63")?;
        let modulus = if k == 1 {
            Vec::new()
        } else {
            let mut found = None;
            // m_0 ≠ 0; enumerate the low k coefficients as an integer in base p
            for code in 0..q {
                let mut m: Vec<u64> = (0..k).map(|i| (code / p.pow(i)) % p).collect();
                if m[0] == 0 {
                    continue;
                }
                m.push(1);
                if is_irreducible(&m, p) {
                    m.pop();
                    found = Some(m);
                    break;
                }
            }
            found.ok_or("no irreducible polynomial found")?
        };
        Gf::build(p, k, q, modulus)
    }

    /// GF(p) (prime field).
    pub fn prime(p: u64) -> Result<Gf, String> {
        Gf::new(p, 1)
    }

    /// GF(p^k) with a given monic modulus `x^k + Σ m_i x^i` (`m.len() = k`), e.g. a Conway polynomial.
    pub fn with_modulus(p: u64, m: &[u64]) -> Result<Gf, String> {
        if !is_prime(p) || p >= 1 << 32 {
            return Err(format!("p = {p} must be a prime below 2^32"));
        }
        let k = m.len() as u32;
        let q = p.checked_pow(k).filter(|&q| q < 1 << 63).ok_or("q too large")?;
        let mut f: Vec<u64> = m.iter().map(|&c| c % p).collect();
        f.push(1);
        if k > 1 && !is_irreducible(&f, p) {
            return Err("modulus is not irreducible".into());
        }
        f.pop();
        Gf::build(p, k, q, if k == 1 { Vec::new() } else { f })
    }

    fn build(p: u64, k: u32, q: u64, modulus: Vec<u64>) -> Result<Gf, String> {
        let pw = (0..=k).map(|i| p.pow(i)).collect();
        let mut g = Gf {
            p,
            k,
            q,
            modulus,
            pw,
            tables: None,
        };
        if q <= TABLE_LIMIT && q > 2 {
            g.tables = Some(g.make_tables());
        }
        Ok(g)
    }

    fn make_tables(&self) -> Tables {
        let q1 = self.q - 1;
        let fac = prime_factors(q1);
        let generator = (2..self.q)
            .find(|&a| fac.iter().all(|&r| self.pow_slow(a, q1 / r) != 1))
            .expect("multiplicative group is cyclic");
        let mut exp = vec![0u32; 2 * q1 as usize];
        let mut log = vec![0u32; self.q as usize];
        let mut x = 1u64;
        for i in 0..q1 as usize {
            exp[i] = x as u32;
            exp[i + q1 as usize] = x as u32;
            log[x as usize] = i as u32;
            x = self.mul_slow(x, generator);
        }
        let zech = if self.p != 2 && self.k > 1 {
            (0..q1 as usize)
                .map(|n| {
                    let v = self.add_slow(1, exp[n] as u64);
                    if v == 0 {
                        u32::MAX
                    } else {
                        log[v as usize]
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        Tables {
            generator,
            exp,
            log,
            zech,
        }
    }

    /// The characteristic `p`.
    pub fn p(&self) -> u64 {
        self.p
    }
    /// The degree `k` over the prime field.
    pub fn k(&self) -> u32 {
        self.k
    }
    /// The order `q = p^k`.
    pub fn q(&self) -> u64 {
        self.q
    }
    /// Low coefficients `m_0 … m_{k-1}` of the monic modulus (empty for prime fields).
    pub fn modulus(&self) -> &[u64] {
        &self.modulus
    }
    /// The primitive element used for the tables (`None` for fields without tables).
    pub fn generator(&self) -> Option<u64> {
        self.tables.as_ref().map(|t| t.generator)
    }
    /// Are multiplication and addition done by log/Zech tables?
    pub fn has_tables(&self) -> bool {
        self.tables.is_some()
    }
    /// All elements, as their integer encodings `0..q`.
    pub fn elements(&self) -> std::ops::Range<u64> {
        0..self.q
    }
    /// The image of an integer in the prime field.
    pub fn from_int(&self, n: i64) -> u64 {
        int::residue(n, self.p)
    }

    /// Base-p digits (polynomial coefficients, low degree first).
    pub fn digits(&self, a: u64) -> Vec<u64> {
        (0..self.k as usize).map(|i| (a / self.pw[i]) % self.p).collect()
    }
    /// The element with base-`p` digits `d` (low degree first), the inverse of [`digits`](Self::digits).
    pub fn from_digits(&self, d: &[u64]) -> u64 {
        d.iter()
            .enumerate()
            .map(|(i, &c)| (c % self.p) * self.pw[i])
            .sum()
    }

    // ---------------------------------------------------- slow (polynomial) arithmetic

    fn add_slow(&self, a: u64, b: u64) -> u64 {
        if self.p == 2 {
            return a ^ b;
        }
        if self.k == 1 {
            return int::add_mod(a, b, self.p);
        }
        let mut r = 0u64;
        for i in 0..self.k as usize {
            let d = ((a / self.pw[i]) % self.p + (b / self.pw[i]) % self.p) % self.p;
            r += d * self.pw[i];
        }
        r
    }

    fn neg_slow(&self, a: u64) -> u64 {
        if self.p == 2 {
            return a;
        }
        if self.k == 1 {
            return int::neg_mod(a, self.p);
        }
        let mut r = 0u64;
        for i in 0..self.k as usize {
            let d = (a / self.pw[i]) % self.p;
            r += ((self.p - d) % self.p) * self.pw[i];
        }
        r
    }

    fn mul_slow(&self, a: u64, b: u64) -> u64 {
        if self.k == 1 {
            return int::mul_mod(a, b, self.p);
        }
        if self.p == 2 {
            // carry-less multiply, reduced bit by bit
            let k = self.k;
            let red: u64 = self.from_digits(&self.modulus); // low part of f
            let mut r = 0u64;
            for i in (0..k).rev() {
                // r = r * x
                let top = (r >> (k - 1)) & 1;
                r = (r << 1) & ((1u64 << k) - 1);
                if top == 1 {
                    r ^= red;
                }
                if (b >> i) & 1 == 1 {
                    r ^= a;
                }
            }
            return r;
        }
        let mut f = self.modulus.clone();
        f.push(1);
        let mut da = self.digits(a);
        let mut db = self.digits(b);
        poly_trim(&mut da);
        poly_trim(&mut db);
        let r = poly_mulmod(&da, &db, &f, self.p);
        self.from_digits(&r)
    }

    fn pow_slow(&self, a: u64, mut e: u64) -> u64 {
        let mut base = a;
        let mut acc = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul_slow(acc, base);
            }
            base = self.mul_slow(base, base);
            e >>= 1;
        }
        acc
    }

    // ---------------------------------------------------- public arithmetic

    #[inline]
    /// `a + b`.
    pub fn add(&self, a: u64, b: u64) -> u64 {
        if self.p == 2 {
            return a ^ b;
        }
        if self.k == 1 {
            return int::add_mod(a, b, self.p);
        }
        match &self.tables {
            Some(t) => {
                if a == 0 {
                    return b;
                }
                if b == 0 {
                    return a;
                }
                let q1 = (self.q - 1) as u32;
                let (la, lb) = (t.log[a as usize], t.log[b as usize]);
                let n = if lb >= la { lb - la } else { lb + q1 - la };
                let z = t.zech[n as usize];
                if z == u32::MAX {
                    0
                } else {
                    t.exp[(la + z) as usize] as u64
                }
            }
            None => self.add_slow(a, b),
        }
    }

    #[inline]
    /// `-a`.
    pub fn neg(&self, a: u64) -> u64 {
        if self.p == 2 || self.k == 1 || a == 0 {
            return self.neg_slow(a);
        }
        match &self.tables {
            // −1 = g^((q−1)/2) for odd q: one table step instead of k digit divisions
            Some(t) => t.exp[(t.log[a as usize] as u64 + (self.q - 1) / 2) as usize] as u64,
            None => self.neg_slow(a),
        }
    }

    #[inline]
    /// `a - b`.
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        self.add(a, self.neg(b))
    }

    #[inline]
    /// `a · b`.
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        if a == 0 || b == 0 {
            return 0;
        }
        if self.k == 1 {
            // direct modular arithmetic beats the (cache-missing) tables in
            // prime fields: 2.4–3 ns against 7.5–32 ns per product
            return int::mul_mod(a, b, self.p);
        }
        match &self.tables {
            Some(t) => t.exp[(t.log[a as usize] + t.log[b as usize]) as usize] as u64,
            None => self.mul_slow(a, b),
        }
    }

    /// Multiplicative inverse; panics on zero.
    #[inline]
    pub fn inv(&self, a: u64) -> u64 {
        assert!(a != 0, "division by zero in GF({})", self.q);
        if self.k == 1 {
            return inv_mod_p(a, self.p);
        }
        match &self.tables {
            Some(t) => t.exp[(self.q - 1 - t.log[a as usize] as u64) as usize] as u64,
            None => self.pow_slow(a, self.q - 2),
        }
    }

    #[inline]
    /// `a / b`. Panics if `b` is zero.
    pub fn div(&self, a: u64, b: u64) -> u64 {
        self.mul(a, self.inv(b))
    }

    /// `a^e` (`a^0 = 1`).
    pub fn pow(&self, a: u64, e: u64) -> u64 {
        if a == 0 {
            return if e == 0 { 1 } else { 0 };
        }
        if self.k == 1 {
            return int::pow_mod(a, e, self.p);
        }
        match &self.tables {
            Some(t) => {
                let q1 = self.q - 1;
                t.exp[((t.log[a as usize] as u128 * (e % q1) as u128) % q1 as u128) as usize] as u64
            }
            None => self.pow_slow(a, e),
        }
    }

    /// Frobenius `a ↦ a^p`.
    pub fn frobenius(&self, a: u64) -> u64 {
        self.pow(a, self.p)
    }

    /// Absolute trace `Σ_{i<k} a^{p^i}` (an element of the prime field, returned in `0..p`).
    pub fn trace(&self, a: u64) -> u64 {
        let mut s = 0u64;
        let mut x = a;
        for _ in 0..self.k {
            s = self.add(s, x);
            x = self.frobenius(x);
        }
        s
    }

    /// Absolute norm `a^{(q-1)/(p-1)}` (in the prime field).
    pub fn norm(&self, a: u64) -> u64 {
        self.pow(a, (self.q - 1) / (self.p - 1))
    }

    /// Multiplicative order of a nonzero element (tables fields only).
    pub fn order(&self, a: u64) -> Option<u64> {
        let t = self.tables.as_ref()?;
        if a == 0 {
            return None;
        }
        let q1 = self.q - 1;
        let l = t.log[a as usize] as u64;
        Some(int::cyclic_order(q1, l))
    }
}

// ------------------------------------------------------------------ Fq<P, K> coefficients

fn registry() -> &'static RwLock<HashMap<(u64, u32), &'static Gf>> {
    static R: OnceLock<RwLock<HashMap<(u64, u32), &'static Gf>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(HashMap::new()))
}

/// The process-wide field GF(p^k) used by [`Fq`] (built on first use, then cached).
pub fn field(p: u64, k: u32) -> &'static Gf {
    if let Some(f) = registry().read().expect("registry").get(&(p, k)) {
        return f;
    }
    let mut w = registry().write().expect("registry");
    if let Some(f) = w.get(&(p, k)) {
        return f;
    }
    let f: &'static Gf = Box::leak(Box::new(
        Gf::new(p, k).unwrap_or_else(|e| panic!("GF({p}^{k}): {e}")),
    ));
    w.insert((p, k), f);
    f
}

/// An element of GF(P^K) as a [`Coeff`] / [`Field`].
///
/// For `K > 1` the arithmetic is that of [`field`]`(P, K)` (tables or
/// polynomial arithmetic); elements print as `#e`, the integer encoding.  For
/// `K = 1` (the prime field, also available as [`crate::Fp`]) the arithmetic
/// is direct modular arithmetic from [`crate::arith::int`], with no field
/// lookup, and residues print as their symmetric representative in
/// `(−P/2, P/2]`.  `parse_literal` accepts `#e` and integers or fractions of
/// integers (embedded through the prime field).
///
/// ```
/// use ncalg::ff::Fq;
/// use ncalg::{Coeff, Field};
/// type F4 = Fq<2, 2>;
/// let a = F4::new(2); // x
/// let b = a.mul_ref(&a).add_ref(&a); // x^2 + x = 1 in GF(4)
/// assert!(b.is_one());
/// assert_eq!(a.inverse().mul_ref(&a), F4::one());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fq<const P: u64, const K: u32>(u64);

impl<const P: u64, const K: u32> Fq<P, K> {
    const VALID: () = assert!(
        P >= 2 && P < (1u64 << 63) && K >= 1,
        "Fq<P, K>: need 2 <= P < 2^63, K >= 1"
    );
    #[inline]
    fn f() -> &'static Gf {
        field(P, K)
    }
    /// From the integer encoding (`0..q`; the residue itself when `K = 1`).
    pub fn new(e: u64) -> Self {
        let () = Self::VALID;
        let q = if K == 1 { P } else { Self::f().q() };
        assert!(e < q, "encoding out of range");
        Fq(e)
    }
    /// The integer encoding (`0..q`; the canonical residue when `K = 1`).
    pub fn value(self) -> u64 {
        self.0
    }
    /// Symmetric representative in `(−P/2, P/2]` (prime fields, `K = 1`).
    pub fn signed_value(self) -> i64 {
        assert!(K == 1, "signed_value is defined for prime fields (K = 1)");
        if self.0 > P / 2 {
            -((P - self.0) as i64)
        } else {
            self.0 as i64
        }
    }
    /// `self^e` (`self^0 = 1`).
    pub fn pow(self, e: u64) -> Self {
        if K == 1 {
            Fq(int::pow_mod(self.0, e, P))
        } else {
            Fq(Self::f().pow(self.0, e))
        }
    }
}

impl<const P: u64, const K: u32> fmt::Debug for Fq<P, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if K == 1 {
            write!(f, "{} (mod {})", self, P)
        } else {
            write!(f, "{} in GF({}^{})", self, P, K)
        }
    }
}

impl<const P: u64, const K: u32> fmt::Display for Fq<P, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if K == 1 {
            write!(f, "{}", self.signed_value())
        } else {
            write!(f, "#{}", self.0)
        }
    }
}

impl<const P: u64, const K: u32> Coeff for Fq<P, K> {
    fn zero() -> Self {
        let () = Self::VALID;
        Fq(0)
    }
    fn one() -> Self {
        let () = Self::VALID;
        Fq(1)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    fn is_one(&self) -> bool {
        self.0 == 1
    }
    #[inline]
    fn add_ref(&self, o: &Self) -> Self {
        if K == 1 {
            Fq(int::add_mod(self.0, o.0, P))
        } else {
            Fq(Self::f().add(self.0, o.0))
        }
    }
    #[inline]
    fn sub_ref(&self, o: &Self) -> Self {
        if K == 1 {
            Fq(int::sub_mod(self.0, o.0, P))
        } else {
            Fq(Self::f().sub(self.0, o.0))
        }
    }
    #[inline]
    fn mul_ref(&self, o: &Self) -> Self {
        if K == 1 {
            Fq(int::mul_mod(self.0, o.0, P))
        } else {
            Fq(Self::f().mul(self.0, o.0))
        }
    }
    #[inline]
    fn negated(&self) -> Self {
        if K == 1 {
            Fq(int::neg_mod(self.0, P))
        } else {
            Fq(Self::f().neg(self.0))
        }
    }
    fn from_i64(n: i64) -> Self {
        let () = Self::VALID;
        // integers embed through the prime field: the encoding of a residue is itself
        Fq(int::residue(n, P))
    }
    fn parse_literal(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix('#') {
            let e: u64 = rest.parse().ok()?;
            let q = if K == 1 { P } else { Self::f().q() };
            return (e < q).then_some(Fq(e));
        }
        if let Some((a, b)) = s.split_once('/') {
            let num = Self::parse_literal(a)?;
            let den = Self::parse_literal(b)?;
            if den.is_zero() {
                return None;
            }
            return Some(num.div_ref(&den));
        }
        use num_traits::ToPrimitive;
        let n = crate::coeff::parse_bigint(s)?;
        let p = num_bigint::BigInt::from(P);
        let r = ((&n % &p) + &p) % &p;
        Some(Fq(r.to_u64().expect("reduced residue fits")))
    }
    fn is_negative_for_display(&self) -> bool {
        K == 1 && self.signed_value() < 0
    }
    fn magnitude(&self) -> f64 {
        if K == 1 {
            self.signed_value().unsigned_abs() as f64
        } else {
            self.0 as f64
        }
    }
}

impl<const P: u64, const K: u32> Field for Fq<P, K> {
    fn inverse(&self) -> Self {
        if K == 1 {
            Fq(int::inv_mod(self.0, P).expect("division by zero (or P not prime) in Fp"))
        } else {
            Fq(Self::f().inv(self.0))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axioms(f: &Gf, samples: &[u64]) {
        for &a in samples {
            assert_eq!(f.add(a, 0), a);
            assert_eq!(f.mul(a, 1), a);
            assert_eq!(f.add(a, f.neg(a)), 0);
            if a != 0 {
                assert_eq!(f.mul(a, f.inv(a)), 1, "inverse of {a} in GF({})", f.q());
            }
            for &b in samples {
                assert_eq!(f.add(a, b), f.add(b, a));
                assert_eq!(f.mul(a, b), f.mul(b, a));
                assert_eq!(f.add(a, b), f.add_slow(a, b), "table add");
                assert_eq!(f.mul(a, b), f.mul_slow(a, b), "table mul");
                for &c in samples.iter().take(12) {
                    assert_eq!(f.mul(a, f.add(b, c)), f.add(f.mul(a, b), f.mul(a, c)));
                    assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
                    assert_eq!(f.add(f.add(a, b), c), f.add(a, f.add(b, c)));
                }
            }
        }
    }

    #[test]
    fn small_fields_exhaustive() {
        for (p, k) in [
            (2, 1),
            (3, 1),
            (2, 2),
            (5, 1),
            (7, 1),
            (2, 3),
            (3, 2),
            (2, 4),
            (5, 2),
            (3, 3),
            (7, 2),
            (2, 6),
            (3, 4),
            (5, 3),
            (2, 8),
        ] {
            let f = Gf::new(p, k).unwrap();
            let all: Vec<u64> = f.elements().collect();
            axioms(&f, &all);
            if let Some(g) = f.generator() {
                assert_eq!(f.order(g), Some(f.q() - 1));
            }
            // Frobenius is an automorphism fixing exactly the prime field
            let fixed = all.iter().filter(|&&a| f.frobenius(a) == a).count() as u64;
            assert_eq!(fixed, p);
            for &a in &all {
                assert!(f.trace(a) < p && f.norm(a) < p);
            }
        }
    }

    #[test]
    fn large_fields_sampled() {
        let mut s = 99u64;
        let mut rnd = |q: u64| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) % q
        };
        for (p, k) in [
            (2, 16),
            (3, 10),
            (2, 31),
            (2, 61),
            (65521, 1),
            (4294967291, 1),
            (65537, 2),
        ] {
            let f = Gf::new(p, k).unwrap();
            let samples: Vec<u64> = (0..24).map(|_| rnd(f.q())).chain([0, 1]).collect();
            for &a in &samples {
                assert_eq!(f.add(a, f.neg(a)), 0);
                if a != 0 {
                    assert_eq!(f.mul(a, f.inv(a)), 1, "GF({p}^{k})");
                    assert_eq!(f.pow(a, f.q() - 1), 1);
                }
                for &b in &samples {
                    for &c in samples.iter().take(6) {
                        assert_eq!(f.mul(a, f.add(b, c)), f.add(f.mul(a, b), f.mul(a, c)));
                        assert_eq!(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)));
                    }
                }
            }
        }
    }

    #[test]
    fn rabin_and_given_modulus() {
        // x^2 + x + 1 irreducible over F_2; x^2 + 1 is not
        assert!(Gf::with_modulus(2, &[1, 1]).is_ok());
        assert!(Gf::with_modulus(2, &[1, 0]).is_err());
        // Conway polynomial for GF(3^2): x^2 + 2x + 2
        let f = Gf::with_modulus(3, &[2, 2]).unwrap();
        assert_eq!(f.order(3), Some(8)); // x is primitive for a Conway polynomial
                                         // smallest irreducible for GF(2^8) is x^8 + x^4 + x^3 + x + 1 (0x11B) under our enumeration
        let g = Gf::new(2, 8).unwrap();
        assert_eq!(g.from_digits(g.modulus()), 0x1B);
    }

    #[test]
    fn fq_coefficients() {
        type F9 = Fq<3, 2>;
        let a = F9::new(5);
        assert_eq!(a.mul_ref(&a.inverse()), F9::one());
        assert_eq!(F9::from_i64(-1), F9::new(2));
        assert_eq!(F9::parse_literal("#5"), Some(a));
        assert_eq!(a.to_string(), "#5");
        type F4 = Fq<2, 2>;
        let x = F4::new(2);
        assert!(x.mul_ref(&x).add_ref(&x).is_one());
        // polynomials print and parse back with #e literals
        let alg = crate::FreeAlgebra::new(["u", "v"]);
        let p: crate::Polynomial<F4> = alg.parse("#2*u*v + #3*v + 1").unwrap();
        let back: crate::Polynomial<F4> = alg.parse(&p.to_string()).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn bit_echelon_combinations() {
        // 70 rows of 130 bits from an LCG: dependencies and reductions must reproduce the vectors
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            x
        };
        let rows: Vec<Vec<u64>> = (0..70)
            .map(|i| {
                let mut r = vec![next(), next(), next() & 3];
                if i % 7 == 6 {
                    r = vec![0, 0, 0]; // some zero rows
                }
                r
            })
            .collect();
        let xor_of = |set: &[usize]| -> Vec<u64> {
            let mut v = vec![0u64; 3];
            for &k in set {
                for (a, b) in v.iter_mut().zip(&rows[k]) {
                    *a ^= *b;
                }
            }
            v
        };
        let mut ech = BitEchelon::new(130);
        for (i, r) in rows.iter().enumerate() {
            match ech.insert(r.clone()) {
                Ok(k) => assert_eq!(k, i),
                Err(c) => {
                    let set = BitEchelon::indices(&c);
                    assert!(set.contains(&i));
                    assert_eq!(xor_of(&set), vec![0, 0, 0], "dependency at row {i}");
                }
            }
        }
        assert_eq!(ech.inserted(), 70);
        assert!(ech.rank() <= 60);
        // a vector in the span reduces to zero and its combination reproduces it
        let target = xor_of(&[1, 4, 9, 33]);
        let mut v = target.clone();
        let comb = ech.reduce(&mut v);
        assert_eq!(v, vec![0, 0, 0]);
        assert_eq!(xor_of(&BitEchelon::indices(&comb)), target);
    }

    #[test]
    fn groebner_over_gf4() {
        use crate::{FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, QuotientAlgebra};
        type F4 = Fq<2, 2>;
        // commutative ideal in K<u, v> over GF(4): u^2 + #2*u + 1, v^2 - v, u v - v u;
        // the quotient has dimension 2 x 2 = 4.
        let alg = FreeAlgebra::new(["u", "v"]);
        let u: Polynomial<F4> = alg.gen("u");
        let v: Polynomial<F4> = alg.gen("v");
        let w = F4::new(2);
        let rels = vec![
            &u * &u + u.scale(&w) + alg.constant(F4::one()),
            &v * &v - v.clone(),
            &u * &v - &v * &u,
        ];
        let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
        assert!(gb.is_complete());
        assert_eq!(QuotientAlgebra::new(gb).dimension(), Some(4));
    }
}
