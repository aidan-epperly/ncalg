//! [`HybridRational`]: exact rational arithmetic that stays in machine words
//! while it can.
//!
//! Most Gröbner basis computations in this crate (Pauli, Majorana, swap,
//! Weyl, Lie algebras) only ever see small coefficients, for which
//! `BigRational` spends almost all of its time allocating.  `HybridRational`
//! keeps `numerator / denominator` as `i64`s (reduced, denominator positive)
//! and computes with `i128` intermediates; whenever a result does not fit it
//! is promoted to a boxed `BigRational`, and values that fit again are
//! demoted, so every value has a unique representation and equality is
//! structural.  Arithmetic is exact in all cases.

use crate::coeff::{Coeff, CoeffConvert, Field};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use std::fmt;

/// A rational number: `i64` numerator and (positive) denominator, promoted
/// to a `BigRational` on overflow.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum HybridRational {
    /// `numerator / denominator`, reduced, denominator `> 0`.
    Small(i64, i64),
    /// A value outside the `i64` range (numerator or denominator).
    Big(Box<BigRational>),
}

impl HybridRational {
    /// Build from an `i128` fraction (reduced and demoted or promoted as needed).
    fn make(num: i128, den: i128) -> Self {
        assert!(den != 0, "zero denominator");
        if num == 0 {
            return HybridRational::Small(0, 1);
        }
        let g = num_integer::Integer::gcd(&num, &den);
        let (mut n, mut d) = (num / g, den / g);
        if d < 0 {
            n = -n;
            d = -d;
        }
        Self::reduced(n, d)
    }

    /// From an already reduced `n / d` with `d > 0`: demoted to machine
    /// words when both fit.
    #[inline]
    fn reduced(n: i128, d: i128) -> Self {
        match (i64::try_from(n), i64::try_from(d)) {
            (Ok(n), Ok(d)) => HybridRational::Small(n, d),
            _ => HybridRational::Big(Box::new(BigRational::new(BigInt::from(n), BigInt::from(d)))),
        }
    }

    /// `a/b ± c/d` for reduced operands (`b, d > 0`) without a gcd of the
    /// result (Henrici; Knuth, TAOCP 2, 4.5.1): with `g = gcd(b, d)`,
    /// `t = a (d/g) ± c (b/g)` and `g' = gcd(t, g)`, the result
    /// `(t/g') / ((b/g)(d/g'))` is already reduced.  All gcds on `u64`.
    #[inline]
    fn add_small(a: i64, b: i64, c: i64, d: i64, negate_c: bool) -> Self {
        let c = c as i128 * if negate_c { -1 } else { 1 };
        if b == 1 && d == 1 {
            return Self::reduced(a as i128 + c, 1);
        }
        let g = (b as u64).gcd(&(d as u64)) as i128;
        let (b, d) = (b as i128, d as i128);
        let t = a as i128 * (d / g) + c * (b / g);
        if t == 0 {
            return HybridRational::Small(0, 1);
        }
        if g == 1 {
            return Self::reduced(t, b * d);
        }
        let g2 = (t.unsigned_abs() % g as u128) as u64;
        let g2 = (g2.gcd(&(g as u64))) as i128;
        Self::reduced(t / g2, (b / g) * (d / g2))
    }

    /// `(a/b)(c/d)` for reduced operands, cross-canceled first so that the
    /// product is reduced: `(a/g1)(c/g2) / ((b/g2)(d/g1))` with
    /// `g1 = gcd(a, d)`, `g2 = gcd(c, b)`.
    #[inline]
    fn mul_small(a: i64, b: i64, c: i64, d: i64) -> Self {
        if a == 0 || c == 0 {
            return HybridRational::Small(0, 1);
        }
        let g1 = a.unsigned_abs().gcd(&(d as u64)) as i128;
        let g2 = c.unsigned_abs().gcd(&(b as u64)) as i128;
        let (a, b, c, d) = (a as i128, b as i128, c as i128, d as i128);
        Self::reduced((a / g1) * (c / g2), (b / g2) * (d / g1))
    }

    /// Canonical form of a `BigRational`.
    pub fn from_big(b: BigRational) -> Self {
        match (b.numer().to_i64(), b.denom().to_i64()) {
            (Some(n), Some(d)) => HybridRational::Small(n, d), // BigRational is already reduced, d > 0
            _ => HybridRational::Big(Box::new(b)),
        }
    }

    /// `n / d` (any sign, `d ≠ 0`).
    pub fn new(n: i64, d: i64) -> Self {
        Self::make(n as i128, d as i128)
    }

    /// The value as a `BigRational`.
    pub fn to_big(&self) -> BigRational {
        match self {
            HybridRational::Small(n, d) => BigRational::new(BigInt::from(*n), BigInt::from(*d)),
            HybridRational::Big(b) => (**b).clone(),
        }
    }

    /// Is the value held in machine words?
    pub fn is_small(&self) -> bool {
        matches!(self, HybridRational::Small(..))
    }

    /// Floating-point approximation.
    pub fn to_f64(&self) -> f64 {
        match self {
            HybridRational::Small(n, d) => *n as f64 / *d as f64,
            HybridRational::Big(b) => b.to_f64().unwrap_or(f64::NAN),
        }
    }

    /// The value as a `BigRational`, borrowed when it already is one.
    fn as_big(&self) -> std::borrow::Cow<'_, BigRational> {
        match self {
            HybridRational::Small(n, d) => {
                std::borrow::Cow::Owned(BigRational::new_raw(BigInt::from(*n), BigInt::from(*d)))
            }
            HybridRational::Big(b) => std::borrow::Cow::Borrowed(b),
        }
    }

    fn with_big(&self, o: &Self, f: impl FnOnce(&BigRational, &BigRational) -> BigRational) -> Self {
        Self::from_big(f(&self.as_big(), &o.as_big()))
    }
}

impl fmt::Debug for HybridRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for HybridRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HybridRational::Small(n, 1) => write!(f, "{}", n),
            HybridRational::Small(n, d) => write!(f, "{}/{}", n, d),
            HybridRational::Big(b) => write!(f, "{}", b),
        }
    }
}

impl Coeff for HybridRational {
    fn zero() -> Self {
        HybridRational::Small(0, 1)
    }
    fn one() -> Self {
        HybridRational::Small(1, 1)
    }
    fn is_zero(&self) -> bool {
        matches!(self, HybridRational::Small(0, _))
    }
    fn is_one(&self) -> bool {
        matches!(self, HybridRational::Small(1, 1))
    }
    fn add_ref(&self, o: &Self) -> Self {
        match (self, o) {
            (HybridRational::Small(a, b), HybridRational::Small(c, d)) => {
                Self::add_small(*a, *b, *c, *d, false)
            }
            _ => self.with_big(o, |x, y| x + y),
        }
    }
    fn sub_ref(&self, o: &Self) -> Self {
        match (self, o) {
            (HybridRational::Small(a, b), HybridRational::Small(c, d)) => {
                Self::add_small(*a, *b, *c, *d, true)
            }
            _ => self.with_big(o, |x, y| x - y),
        }
    }
    fn mul_ref(&self, o: &Self) -> Self {
        match (self, o) {
            (HybridRational::Small(a, b), HybridRational::Small(c, d)) => Self::mul_small(*a, *b, *c, *d),
            _ => self.with_big(o, |x, y| x * y),
        }
    }
    fn negated(&self) -> Self {
        match self {
            HybridRational::Small(n, d) => Self::make(-(*n as i128), *d as i128),
            HybridRational::Big(b) => Self::from_big(-(**b).clone()),
        }
    }
    fn from_i64(n: i64) -> Self {
        HybridRational::Small(n, 1)
    }
    fn parse_literal(s: &str) -> Option<Self> {
        BigRational::parse_literal(s).map(Self::from_big)
    }
    fn is_negative_for_display(&self) -> bool {
        match self {
            HybridRational::Small(n, _) => *n < 0,
            HybridRational::Big(b) => b.is_negative(),
        }
    }
    fn magnitude(&self) -> f64 {
        self.to_f64().abs()
    }
}

impl Field for HybridRational {
    fn inverse(&self) -> Self {
        match self {
            HybridRational::Small(0, _) => panic!("division by zero"),
            HybridRational::Small(n, d) => Self::make(*d as i128, *n as i128),
            HybridRational::Big(b) => Self::from_big(b.recip()),
        }
    }
}

impl From<i64> for HybridRational {
    fn from(n: i64) -> Self {
        HybridRational::Small(n, 1)
    }
}

impl From<BigRational> for HybridRational {
    fn from(b: BigRational) -> Self {
        Self::from_big(b)
    }
}

impl From<HybridRational> for BigRational {
    fn from(h: HybridRational) -> Self {
        h.to_big()
    }
}

impl CoeffConvert<BigRational> for HybridRational {
    fn convert(&self) -> BigRational {
        self.to_big()
    }
}
impl CoeffConvert<HybridRational> for BigRational {
    fn convert(&self) -> HybridRational {
        HybridRational::from_big(self.clone())
    }
}
impl CoeffConvert<HybridRational> for i64 {
    fn convert(&self) -> HybridRational {
        HybridRational::Small(*self, 1)
    }
}
impl CoeffConvert<HybridRational> for BigInt {
    fn convert(&self) -> HybridRational {
        HybridRational::from_big(BigRational::from_integer(self.clone()))
    }
}
impl CoeffConvert<f64> for HybridRational {
    fn convert(&self) -> f64 {
        self.to_f64()
    }
}
impl CoeffConvert<num_complex::Complex<f64>> for HybridRational {
    fn convert(&self) -> num_complex::Complex<f64> {
        num_complex::Complex::new(self.to_f64(), 0.0)
    }
}
impl CoeffConvert<num_complex::Complex<HybridRational>> for HybridRational {
    fn convert(&self) -> num_complex::Complex<HybridRational> {
        num_complex::Complex::new(self.clone(), HybridRational::Small(0, 1))
    }
}
impl CoeffConvert<num_complex::Complex<HybridRational>> for i64 {
    fn convert(&self) -> num_complex::Complex<HybridRational> {
        num_complex::Complex::new(HybridRational::Small(*self, 1), HybridRational::Small(0, 1))
    }
}
impl CoeffConvert<num_complex::Complex<BigRational>> for HybridRational {
    fn convert(&self) -> num_complex::Complex<BigRational> {
        num_complex::Complex::new(self.to_big(), Zero::zero())
    }
}
impl CoeffConvert<num_complex::Complex<f64>> for num_complex::Complex<HybridRational> {
    fn convert(&self) -> num_complex::Complex<f64> {
        num_complex::Complex::new(self.re.to_f64(), self.im.to_f64())
    }
}
impl CoeffConvert<num_complex::Complex<BigRational>> for num_complex::Complex<HybridRational> {
    fn convert(&self) -> num_complex::Complex<BigRational> {
        num_complex::Complex::new(self.re.to_big(), self.im.to_big())
    }
}
impl CoeffConvert<num_complex::Complex<HybridRational>> for num_complex::Complex<BigRational> {
    fn convert(&self) -> num_complex::Complex<HybridRational> {
        num_complex::Complex::new(
            HybridRational::from_big(self.re.clone()),
            HybridRational::from_big(self.im.clone()),
        )
    }
}

// --- std operator traits and num_traits::Num, so that `Complex<HybridRational>`
// --- gets num_complex's `Display` and arithmetic -----------------------------

impl std::ops::Add for HybridRational {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        self.add_ref(&o)
    }
}
impl std::ops::Sub for HybridRational {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self.sub_ref(&o)
    }
}
impl std::ops::Mul for HybridRational {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        self.mul_ref(&o)
    }
}
impl std::ops::Div for HybridRational {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        self.mul_ref(&o.inverse())
    }
}
impl std::ops::Rem for HybridRational {
    type Output = Self;
    fn rem(self, o: Self) -> Self {
        Self::from_big(self.to_big() % o.to_big())
    }
}
impl std::ops::Neg for HybridRational {
    type Output = Self;
    fn neg(self) -> Self {
        self.negated()
    }
}
impl num_traits::Zero for HybridRational {
    fn zero() -> Self {
        HybridRational::Small(0, 1)
    }
    fn is_zero(&self) -> bool {
        Coeff::is_zero(self)
    }
}
impl num_traits::One for HybridRational {
    fn one() -> Self {
        HybridRational::Small(1, 1)
    }
}
impl num_traits::Num for HybridRational {
    type FromStrRadixErr = String;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, String> {
        if radix != 10 {
            return Err("only radix 10 is supported".to_string());
        }
        Coeff::parse_literal(s).ok_or_else(|| format!("not a rational literal: {}", s))
    }
}
impl PartialOrd for HybridRational {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HybridRational {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        match (self, o) {
            (HybridRational::Small(a, b), HybridRational::Small(c, d)) => {
                // b, d > 0: a/b < c/d  <=>  a d < c b
                (*a as i128 * *d as i128).cmp(&(*c as i128 * *b as i128))
            }
            _ => self.to_big().cmp(&o.to_big()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_promotion() {
        let a = HybridRational::new(2, 6);
        assert_eq!(a, HybridRational::Small(1, 3));
        assert_eq!(a.to_string(), "1/3");
        let b = HybridRational::new(-4, -8);
        assert_eq!(b, HybridRational::Small(1, 2));
        assert_eq!(a.add_ref(&b).to_string(), "5/6");
        assert_eq!(a.sub_ref(&b).to_string(), "-1/6");
        assert_eq!(a.mul_ref(&b).to_string(), "1/6");
        assert_eq!(a.inverse().to_string(), "3");
        assert_eq!(HybridRational::new(-3, 7).inverse(), HybridRational::Small(-7, 3));
        // overflow promotes, and the promoted value agrees with BigRational
        let big = HybridRational::Small(i64::MAX, 1);
        let sq = big.mul_ref(&big);
        assert!(!sq.is_small());
        assert_eq!(sq.to_big(), big.to_big() * big.to_big());
        // and dividing back demotes
        let back = sq.mul_ref(&big.inverse());
        assert!(back.is_small());
        assert_eq!(back, big);
        // zero and one
        assert!(Coeff::is_zero(&a.sub_ref(&a)));
        assert!(Coeff::is_one(&a.mul_ref(&a.inverse())));
        // literals
        assert_eq!(
            HybridRational::parse_literal("-7/21").unwrap(),
            HybridRational::Small(-1, 3)
        );
        assert_eq!(
            HybridRational::parse_literal("2.5").unwrap(),
            HybridRational::Small(5, 2)
        );
        assert_eq!(
            HybridRational::parse_literal("123456789012345678901234567890")
                .unwrap()
                .to_string(),
            "123456789012345678901234567890"
        );
    }
}
