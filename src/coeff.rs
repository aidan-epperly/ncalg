//! Coefficient rings and fields.
//!
//! Every coefficient type used by [`crate::Polynomial`] implements [`Coeff`].
//! Types that additionally implement [`Field`] can be used for Gröbner basis
//! computation and reduction (which require division by leading coefficients).
//!
//! Provided implementations:
//!
//! | type | ring | field | notes |
//! |---|---|---|---|
//! | `i64`, `i128` | yes | no | checked arithmetic, panics on overflow |
//! | `BigInt` | yes | no | arbitrary precision |
//! | `BigRational` | yes | yes | exact rationals; what integer polynomials are promoted to for Gröbner bases |
//! | `f64` | yes | yes | relative cancellation test ([`Coeff::canceled`]), optional floor [`set_float_tolerance`] |
//! | `Complex<f64>` | yes | yes | complex floats |
//! | `Complex<BigRational>` | yes | yes | exact Gaussian rationals |
//!
//! Method names deliberately avoid clashing with the `std::ops` traits
//! (`add_ref` rather than `add`) so that calling them on `f64` or `BigInt`
//! is never ambiguous.

use num_bigint::BigInt;
use num_complex::Complex;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::fmt::{self, Debug, Display};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// A commutative coefficient ring.
pub trait Coeff: Clone + PartialEq + Debug + Display + Send + Sync + 'static {
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Is this coefficient zero?  Exact for exact types; for floating point
    /// types exactly zero or at most the optional absolute floor
    /// ([`set_float_tolerance`], default `0`).
    fn is_zero(&self) -> bool;
    /// Floating-point (inexact) arithmetic?  Merges then track the size of
    /// the summands for [`canceled`](Self::canceled).
    const INEXACT: bool = false;
    /// Is `sum`, the result of adding terms whose magnitudes add up to
    /// `abs_sum`, zero up to the rounding of that addition?  Exact types:
    /// `sum.is_zero()`.  Floating point: `|sum| ≤ 64 ε abs_sum` (or below
    /// the absolute floor), the backward-stable, scale-invariant cancellation
    /// test: the computed sum is the exact sum of perturbations of the
    /// summands by relative `O(ε)`, one of which sums to zero.
    fn canceled(sum: &Self, abs_sum: f64) -> bool {
        let _ = abs_sum;
        sum.is_zero()
    }
    /// Is this coefficient one?
    fn is_one(&self) -> bool {
        self.sub_ref(&Self::one()).is_zero()
    }
    /// `self + other`.
    fn add_ref(&self, other: &Self) -> Self;
    /// `self - other`.
    fn sub_ref(&self, other: &Self) -> Self;
    /// `self · other`.
    fn mul_ref(&self, other: &Self) -> Self;
    /// `-self`.
    fn negated(&self) -> Self;
    /// `self += other`.
    fn add_assign_ref(&mut self, other: &Self) {
        *self = self.add_ref(other);
    }
    /// `self -= other`.
    fn sub_assign_ref(&mut self, other: &Self) {
        *self = self.sub_ref(other);
    }
    /// `self *= other`.
    fn mul_assign_ref(&mut self, other: &Self) {
        *self = self.mul_ref(other);
    }
    /// Embed an integer.
    fn from_i64(n: i64) -> Self;
    /// Parse a numeric literal such as `3`, `-2`, `1/2`, `2.5`, `i` (for complex
    /// types).  Returns `None` if the literal is not representable.
    fn parse_literal(s: &str) -> Option<Self>;
    /// Complex conjugate (identity for real types).  Used by
    /// [`crate::Polynomial::adjoint`].
    fn conj(&self) -> Self {
        self.clone()
    }
    /// Used only for pretty printing: should `- c` be printed instead of `+ (-c)`?
    fn is_negative_for_display(&self) -> bool {
        false
    }
    /// Used only for pretty printing: does this coefficient need parentheses
    /// when multiplied with a monomial (e.g. `(1+2i)*x`)?
    fn needs_parens_for_display(&self) -> bool {
        false
    }
    /// Approximate magnitude, used for diagnostics.
    fn magnitude(&self) -> f64;
    /// Pretty-print for use inside polynomials (complex numbers print as
    /// `a+bi`, `bi`, `i`, `-i`, …).  Defaults to `Display`.
    fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// A coefficient field: a [`Coeff`] with multiplicative inverses.
pub trait Field: Coeff {
    /// Multiplicative inverse. Panics on zero.
    fn inverse(&self) -> Self;
    /// `self / other`. Panics if `other` is zero.
    fn div_ref(&self, other: &Self) -> Self {
        self.mul_ref(&other.inverse())
    }
}

/// Coefficient types that contain an imaginary unit `i` with `i*i = -1`.
pub trait ComplexCoeff: Coeff {
    /// The imaginary unit.
    fn imaginary_unit() -> Self;
    /// Real part as an `f64` approximation.
    fn re_f64(&self) -> f64;
    /// Imaginary part as an `f64` approximation.
    fn im_f64(&self) -> f64;
}

// ---------------------------------------------------------------------------
// floating point tolerance
// ---------------------------------------------------------------------------

static FLOAT_TOL_BITS: AtomicU64 = AtomicU64::new(0); // 0.0 as f64 bits

thread_local! {
    /// Exact zero tests on this thread (see [`with_exact_zero`]).
    static EXACT_ZERO: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set an absolute floor below which `f64` / `Complex<f64>` coefficients are
/// treated as zero (default `0`: none).  Cancellation is detected
/// independently of it, relative to the size of the summands (see
/// [`Coeff::canceled`]); the floor is an opt-in for data with a known
/// absolute noise level, and is not scale-invariant.
pub fn set_float_tolerance(tol: f64) {
    FLOAT_TOL_BITS.store(tol.abs().to_bits(), AtomicOrdering::Relaxed);
}

/// The absolute floor in effect on this thread, see [`set_float_tolerance`]
/// (`0` inside [`with_exact_zero`]).
pub fn float_tolerance() -> f64 {
    if EXACT_ZERO.with(|e| e.get()) {
        0.0
    } else {
        f64::from_bits(FLOAT_TOL_BITS.load(AtomicOrdering::Relaxed))
    }
}

/// Run `f` with exact zero tests on the calling thread: no absolute floor
/// and no cancellation test, so only exact zeros vanish.  Thread-local, so
/// it does not affect other threads (unlike changing the global floor).
pub fn with_exact_zero<R>(f: impl FnOnce() -> R) -> R {
    let old = EXACT_ZERO.with(|e| e.replace(true));
    let out = f();
    EXACT_ZERO.with(|e| e.set(old));
    out
}

/// Whether [`with_exact_zero`] is in effect on this thread (work handed to
/// other threads must re-enter it there).
pub(crate) fn exact_zero_active() -> bool {
    EXACT_ZERO.with(|e| e.get())
}

/// The cancellation factor `64 ε` of [`Coeff::canceled`] on this thread.
#[inline]
pub(crate) fn cancellation_factor() -> f64 {
    if EXACT_ZERO.with(|e| e.get()) {
        0.0
    } else {
        64.0 * f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// literal parsing helpers
// ---------------------------------------------------------------------------

pub(crate) fn parse_bigint(s: &str) -> Option<BigInt> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    s.parse::<BigInt>().ok()
}

/// Parse `a`, `a/b`, `a.b`, `a.be±n` into a `BigRational`.
fn parse_bigrational(s: &str) -> Option<BigRational> {
    let s = s.trim();
    if let Some((num, den)) = s.split_once('/') {
        let n = parse_bigint(num)?;
        let d = parse_bigint(den)?;
        if Zero::is_zero(&d) {
            return None;
        }
        return Some(BigRational::new(n, d));
    }
    if let Some(n) = parse_bigint(s) {
        return Some(BigRational::from_integer(n));
    }
    let (mantissa, exp) = match s.find(['e', 'E']) {
        Some(pos) => (&s[..pos], s[pos + 1..].parse::<i64>().ok()?),
        None => (s, 0),
    };
    let (neg, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((a, b)) => (a, b),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let digits = format!("{}{}", int_part, frac_part);
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let digits = if digits.is_empty() {
        "0".to_string()
    } else {
        digits
    };
    let mut n: BigInt = digits.parse().ok()?;
    if neg {
        n = -n;
    }
    // a literal like 1e-99999999999 would build a power of ten with 10¹¹
    // digits (the parser hung); exact rationals that large are not
    // meaningful input
    const MAX_EXP: i64 = 100_000;
    if exp.unsigned_abs() > MAX_EXP as u64 {
        return None;
    }
    let scale = frac_part.len() as i64 - exp;
    let ten = BigInt::from(10);
    let r = if scale >= 0 {
        BigRational::new(n, num_traits::pow(ten, scale as usize))
    } else {
        BigRational::from_integer(n * num_traits::pow(ten, (-scale) as usize))
    };
    Some(r)
}

// ---------------------------------------------------------------------------
// machine integers (checked)
// ---------------------------------------------------------------------------

macro_rules! impl_coeff_int {
    ($t:ty) => {
        impl Coeff for $t {
            #[inline]
            fn zero() -> Self {
                0
            }
            #[inline]
            fn one() -> Self {
                1
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == 0
            }
            #[inline]
            fn is_one(&self) -> bool {
                *self == 1
            }
            #[inline]
            fn add_ref(&self, o: &Self) -> Self {
                self.checked_add(*o)
                    .expect("integer coefficient overflow in addition")
            }
            #[inline]
            fn sub_ref(&self, o: &Self) -> Self {
                self.checked_sub(*o)
                    .expect("integer coefficient overflow in subtraction")
            }
            #[inline]
            fn mul_ref(&self, o: &Self) -> Self {
                self.checked_mul(*o)
                    .expect("integer coefficient overflow in multiplication")
            }
            #[inline]
            fn negated(&self) -> Self {
                self.checked_neg()
                    .expect("integer coefficient overflow in negation")
            }
            #[inline]
            fn from_i64(n: i64) -> Self {
                n as $t
            }
            fn parse_literal(s: &str) -> Option<Self> {
                let s = s.trim();
                if let Ok(v) = s.parse::<$t>() {
                    return Some(v);
                }
                let r = parse_bigrational(s)?;
                if !r.is_integer() {
                    return None;
                }
                let n = r.to_integer();
                <$t>::try_from(n).ok()
            }
            #[inline]
            fn is_negative_for_display(&self) -> bool {
                *self < 0
            }
            #[inline]
            fn magnitude(&self) -> f64 {
                (*self as f64).abs()
            }
        }
    };
}
impl_coeff_int!(i64);
impl_coeff_int!(i128);

// ---------------------------------------------------------------------------
// BigInt
// ---------------------------------------------------------------------------

impl Coeff for BigInt {
    fn zero() -> Self {
        Zero::zero()
    }
    fn one() -> Self {
        One::one()
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
    fn is_one(&self) -> bool {
        One::is_one(self)
    }
    fn add_ref(&self, o: &Self) -> Self {
        self + o
    }
    fn sub_ref(&self, o: &Self) -> Self {
        self - o
    }
    fn mul_ref(&self, o: &Self) -> Self {
        self * o
    }
    fn negated(&self) -> Self {
        -self
    }
    fn add_assign_ref(&mut self, o: &Self) {
        *self += o;
    }
    fn sub_assign_ref(&mut self, o: &Self) {
        *self -= o;
    }
    fn mul_assign_ref(&mut self, o: &Self) {
        *self *= o;
    }
    fn from_i64(n: i64) -> Self {
        BigInt::from(n)
    }
    fn parse_literal(s: &str) -> Option<Self> {
        let r = parse_bigrational(s)?;
        if r.is_integer() {
            Some(r.to_integer())
        } else {
            None
        }
    }
    fn is_negative_for_display(&self) -> bool {
        self.is_negative()
    }
    fn magnitude(&self) -> f64 {
        self.to_f64().map(f64::abs).unwrap_or(f64::INFINITY)
    }
}

// ---------------------------------------------------------------------------
// BigRational
// ---------------------------------------------------------------------------

impl Coeff for BigRational {
    fn zero() -> Self {
        Zero::zero()
    }
    fn one() -> Self {
        One::one()
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
    fn is_one(&self) -> bool {
        One::is_one(self)
    }
    fn add_ref(&self, o: &Self) -> Self {
        self + o
    }
    fn sub_ref(&self, o: &Self) -> Self {
        self - o
    }
    fn mul_ref(&self, o: &Self) -> Self {
        self * o
    }
    fn negated(&self) -> Self {
        -self
    }
    fn add_assign_ref(&mut self, o: &Self) {
        *self += o;
    }
    fn sub_assign_ref(&mut self, o: &Self) {
        *self -= o;
    }
    fn mul_assign_ref(&mut self, o: &Self) {
        *self *= o;
    }
    fn from_i64(n: i64) -> Self {
        BigRational::from_integer(BigInt::from(n))
    }
    fn parse_literal(s: &str) -> Option<Self> {
        parse_bigrational(s)
    }
    fn is_negative_for_display(&self) -> bool {
        self.is_negative()
    }
    fn magnitude(&self) -> f64 {
        self.to_f64().map(f64::abs).unwrap_or(f64::INFINITY)
    }
}

impl Field for BigRational {
    fn inverse(&self) -> Self {
        assert!(!Zero::is_zero(self), "division by zero");
        self.recip()
    }
    fn div_ref(&self, o: &Self) -> Self {
        assert!(!Zero::is_zero(o), "division by zero");
        self / o
    }
}

// ---------------------------------------------------------------------------
// f64
// ---------------------------------------------------------------------------

impl Coeff for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0.0 || self.abs() <= float_tolerance()
    }
    const INEXACT: bool = true;
    #[inline]
    fn canceled(sum: &Self, abs_sum: f64) -> bool {
        Coeff::is_zero(sum) || sum.abs() <= cancellation_factor() * abs_sum
    }
    #[inline]
    fn add_ref(&self, o: &Self) -> Self {
        self + o
    }
    #[inline]
    fn sub_ref(&self, o: &Self) -> Self {
        self - o
    }
    #[inline]
    fn mul_ref(&self, o: &Self) -> Self {
        self * o
    }
    #[inline]
    fn negated(&self) -> Self {
        -self
    }
    #[inline]
    fn from_i64(n: i64) -> Self {
        n as f64
    }
    fn parse_literal(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some((a, b)) = s.split_once('/') {
            let a: f64 = a.trim().parse().ok()?;
            let b: f64 = b.trim().parse().ok()?;
            if b == 0.0 {
                return None;
            }
            // finite only, as below ("inf/1", "1e300/1e-300" were accepted)
            return Some(a / b).filter(|v| v.is_finite());
        }
        s.parse::<f64>().ok().filter(|v| v.is_finite())
    }
    #[inline]
    fn is_negative_for_display(&self) -> bool {
        *self < 0.0
    }
    #[inline]
    fn magnitude(&self) -> f64 {
        self.abs()
    }
}

impl Field for f64 {
    #[inline]
    fn inverse(&self) -> Self {
        assert!(*self != 0.0, "division by zero");
        1.0 / self
    }
    #[inline]
    fn div_ref(&self, o: &Self) -> Self {
        self / o
    }
}

// ---------------------------------------------------------------------------
// Complex<T>
// ---------------------------------------------------------------------------

/// Split a complex literal such as `2+3i`, `-i`, `1/2i`, `3.5` into real and
/// imaginary literal strings.  Returns (real, imag) where either may be empty.
fn split_complex_literal(s: &str) -> Option<(String, String)> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if s.is_empty() {
        return None;
    }
    if !s.ends_with('i') {
        return Some((s, String::new()));
    }
    let body = &s[..s.len() - 1];
    let bytes = body.as_bytes();
    let mut split = None;
    for idx in (1..bytes.len()).rev() {
        let c = bytes[idx];
        if (c == b'+' || c == b'-') && !matches!(bytes[idx - 1], b'e' | b'E') {
            split = Some(idx);
            break;
        }
    }
    match split {
        Some(idx) => {
            let re = body[..idx].to_string();
            let im = match &body[idx..] {
                "+" => "1".to_string(),
                "-" => "-1".to_string(),
                other => other.to_string(),
            };
            Some((re, im))
        }
        None => {
            let im = match body {
                "" | "+" => "1".to_string(),
                "-" => "-1".to_string(),
                other => other.to_string(),
            };
            Some((String::new(), im))
        }
    }
}

macro_rules! impl_coeff_complex {
    ($t:ty) => {
        impl Coeff for Complex<$t> {
            fn zero() -> Self {
                Complex::new(<$t as Coeff>::zero(), <$t as Coeff>::zero())
            }
            fn one() -> Self {
                Complex::new(<$t as Coeff>::one(), <$t as Coeff>::zero())
            }
            fn is_zero(&self) -> bool {
                Coeff::is_zero(&self.re) && Coeff::is_zero(&self.im)
            }
            const INEXACT: bool = <$t as Coeff>::INEXACT;
            fn canceled(sum: &Self, abs_sum: f64) -> bool {
                if <$t as Coeff>::INEXACT {
                    Coeff::is_zero(sum) || Coeff::magnitude(sum) <= cancellation_factor() * abs_sum
                } else {
                    Coeff::is_zero(sum)
                }
            }
            fn add_ref(&self, o: &Self) -> Self {
                Complex::new(self.re.add_ref(&o.re), self.im.add_ref(&o.im))
            }
            fn sub_ref(&self, o: &Self) -> Self {
                Complex::new(self.re.sub_ref(&o.re), self.im.sub_ref(&o.im))
            }
            fn mul_ref(&self, o: &Self) -> Self {
                let re = self.re.mul_ref(&o.re).sub_ref(&self.im.mul_ref(&o.im));
                let im = self.re.mul_ref(&o.im).add_ref(&self.im.mul_ref(&o.re));
                Complex::new(re, im)
            }
            fn negated(&self) -> Self {
                Complex::new(self.re.negated(), self.im.negated())
            }
            fn from_i64(n: i64) -> Self {
                Complex::new(<$t as Coeff>::from_i64(n), <$t as Coeff>::zero())
            }
            fn parse_literal(s: &str) -> Option<Self> {
                let (re, im) = split_complex_literal(s)?;
                let re = if re.is_empty() {
                    <$t as Coeff>::zero()
                } else {
                    <$t as Coeff>::parse_literal(&re)?
                };
                let im = if im.is_empty() {
                    <$t as Coeff>::zero()
                } else {
                    <$t as Coeff>::parse_literal(&im)?
                };
                Some(Complex::new(re, im))
            }
            fn conj(&self) -> Self {
                Complex::new(self.re.clone(), self.im.negated())
            }
            fn is_negative_for_display(&self) -> bool {
                // real negative, or purely imaginary with negative imaginary part
                (Coeff::is_zero(&self.im) && self.re.is_negative_for_display())
                    || (Coeff::is_zero(&self.re) && self.im.is_negative_for_display())
            }
            fn needs_parens_for_display(&self) -> bool {
                !Coeff::is_zero(&self.im) && !Coeff::is_zero(&self.re)
            }
            fn magnitude(&self) -> f64 {
                self.re.magnitude().hypot(self.im.magnitude())
            }
            fn fmt_display(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let re_zero = Coeff::is_zero(&self.re);
                let im_zero = Coeff::is_zero(&self.im);
                if im_zero {
                    return self.re.fmt_display(f);
                }
                if !re_zero {
                    self.re.fmt_display(f)?;
                    if !self.im.is_negative_for_display() {
                        write!(f, "+")?;
                    }
                }
                if Coeff::is_one(&self.im) {
                    write!(f, "i")
                } else if Coeff::is_one(&self.im.negated()) {
                    write!(f, "-i")
                } else {
                    self.im.fmt_display(f)?;
                    write!(f, "i")
                }
            }
        }

        impl Field for Complex<$t> {
            fn inverse(&self) -> Self {
                assert!(!Coeff::is_zero(self), "division by zero");
                // Smith's algorithm: re² + im² overflows to inf (result 0)
                // above |z| ~ 1e154 and underflows to 0 below ~1e-162 in
                // floating point; exact types get the same value
                let (a, b) = (&self.re, &self.im);
                if a.magnitude() >= b.magnitude() {
                    // 1/(a + ib) = (1 − i r) / (a + b r), r = b/a
                    let r = b.div_ref(a);
                    let d = a.add_ref(&b.mul_ref(&r)).inverse();
                    Complex::new(d.clone(), r.negated().mul_ref(&d))
                } else {
                    // = (r − i) / (a r + b), r = a/b
                    let r = a.div_ref(b);
                    let d = a.mul_ref(&r).add_ref(b).inverse();
                    Complex::new(r.mul_ref(&d), d.negated())
                }
            }
        }

        impl ComplexCoeff for Complex<$t> {
            fn imaginary_unit() -> Self {
                Complex::new(<$t as Coeff>::zero(), <$t as Coeff>::one())
            }
            fn re_f64(&self) -> f64 {
                let m = self.re.magnitude();
                if self.re.is_negative_for_display() {
                    -m
                } else {
                    m
                }
            }
            fn im_f64(&self) -> f64 {
                let m = self.im.magnitude();
                if self.im.is_negative_for_display() {
                    -m
                } else {
                    m
                }
            }
        }
    };
}
impl_coeff_complex!(f64);
impl_coeff_complex!(BigRational);
impl_coeff_complex!(crate::rational::HybridRational);

// ---------------------------------------------------------------------------
// prime fields
// ---------------------------------------------------------------------------

/// The prime field `Z/pZ` for a prime `P < 2^63`: the degree-one case of
/// [`crate::ff::Fq`], with direct modular arithmetic (no field tables).  Fast
/// exact arithmetic for exploratory Gröbner basis computations (leading words,
/// Hilbert series) when rational coefficient growth is too expensive; results
/// are those of the ideal reduced modulo `P`, which agree with the rational
/// ones for all but finitely many primes.  Residues print as their symmetric
/// representative in `(−P/2, P/2]`.
///
/// ```
/// use ncalg::{FreeAlgebra, Polynomial, Fp, GroebnerBasis, GroebnerOptions};
/// type F = Fp<32003>;
/// let alg = FreeAlgebra::new(["x", "y"]);
/// let p: Polynomial<F> = alg.parse("1/2*y*x - x*y").unwrap();
/// let gb = GroebnerBasis::compute(vec![p], &GroebnerOptions::default());
/// assert_eq!(gb.elements()[0].to_string(), "y*x - 2*x*y");
/// ```
pub type Fp<const P: u64> = crate::arith::ff::Fq<P, 1>;

/// Format a coefficient the way the polynomial printer wants it: complex
/// numbers with both parts nonzero are printed as `(a+bi)`.
pub(crate) fn fmt_coeff<C: Coeff>(c: &C, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if c.needs_parens_for_display() {
        write!(f, "(")?;
        c.fmt_display(f)?;
        write!(f, ")")
    } else {
        c.fmt_display(f)
    }
}

/// Conversions between coefficient types that are used by the convenience
/// methods on polynomials (`to_rational`, `to_f64`, ...).
pub trait CoeffConvert<D: Coeff>: Coeff {
    /// The value as a `D`.
    fn convert(&self) -> D;
}

impl CoeffConvert<BigRational> for i64 {
    fn convert(&self) -> BigRational {
        BigRational::from_integer(BigInt::from(*self))
    }
}
impl CoeffConvert<BigRational> for i128 {
    fn convert(&self) -> BigRational {
        BigRational::from_integer(BigInt::from(*self))
    }
}
impl CoeffConvert<BigRational> for BigInt {
    fn convert(&self) -> BigRational {
        BigRational::from_integer(self.clone())
    }
}
impl CoeffConvert<BigInt> for i64 {
    fn convert(&self) -> BigInt {
        BigInt::from(*self)
    }
}
impl CoeffConvert<BigInt> for i128 {
    fn convert(&self) -> BigInt {
        BigInt::from(*self)
    }
}
impl CoeffConvert<f64> for i64 {
    fn convert(&self) -> f64 {
        *self as f64
    }
}
impl CoeffConvert<f64> for i128 {
    fn convert(&self) -> f64 {
        *self as f64
    }
}
impl CoeffConvert<f64> for BigInt {
    fn convert(&self) -> f64 {
        self.to_f64().unwrap_or(f64::NAN)
    }
}
impl CoeffConvert<f64> for BigRational {
    fn convert(&self) -> f64 {
        self.to_f64().unwrap_or(f64::NAN)
    }
}
impl CoeffConvert<Complex<f64>> for f64 {
    fn convert(&self) -> Complex<f64> {
        Complex::new(*self, 0.0)
    }
}
impl CoeffConvert<Complex<f64>> for i64 {
    fn convert(&self) -> Complex<f64> {
        Complex::new(*self as f64, 0.0)
    }
}
impl CoeffConvert<Complex<f64>> for BigRational {
    fn convert(&self) -> Complex<f64> {
        Complex::new(self.to_f64().unwrap_or(f64::NAN), 0.0)
    }
}
impl CoeffConvert<Complex<f64>> for Complex<BigRational> {
    fn convert(&self) -> Complex<f64> {
        Complex::new(
            self.re.to_f64().unwrap_or(f64::NAN),
            self.im.to_f64().unwrap_or(f64::NAN),
        )
    }
}
impl CoeffConvert<Complex<BigRational>> for BigRational {
    fn convert(&self) -> Complex<BigRational> {
        Complex::new(self.clone(), Zero::zero())
    }
}
impl CoeffConvert<Complex<BigRational>> for i64 {
    fn convert(&self) -> Complex<BigRational> {
        Complex::new(BigRational::from_integer(BigInt::from(*self)), Zero::zero())
    }
}
impl CoeffConvert<Complex<BigRational>> for BigInt {
    fn convert(&self) -> Complex<BigRational> {
        Complex::new(BigRational::from_integer(self.clone()), Zero::zero())
    }
}

/// Integer-like coefficients, for content/primitive-part computations.
pub trait IntegerCoeff: Coeff {
    /// Greatest common divisor, non-negative.
    fn gcd_ref(&self, other: &Self) -> Self;
    /// Absolute value.
    fn abs_ref(&self) -> Self;
    /// `self / other` for an exact divisor `other`.
    fn exact_div(&self, other: &Self) -> Self;
}

impl IntegerCoeff for i64 {
    fn gcd_ref(&self, o: &Self) -> Self {
        self.gcd(o)
    }
    fn abs_ref(&self) -> Self {
        self.checked_abs().expect("overflow in abs")
    }
    fn exact_div(&self, o: &Self) -> Self {
        self / o
    }
}
impl IntegerCoeff for i128 {
    fn gcd_ref(&self, o: &Self) -> Self {
        self.gcd(o)
    }
    fn abs_ref(&self) -> Self {
        self.checked_abs().expect("overflow in abs")
    }
    fn exact_div(&self, o: &Self) -> Self {
        self / o
    }
}
impl IntegerCoeff for BigInt {
    fn gcd_ref(&self, o: &Self) -> Self {
        self.gcd(o)
    }
    fn abs_ref(&self) -> Self {
        self.abs()
    }
    fn exact_div(&self, o: &Self) -> Self {
        self / o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rationals() {
        let r = BigRational::parse_literal("3/4").unwrap();
        assert_eq!(r, BigRational::new(3.into(), 4.into()));
        let r = BigRational::parse_literal("2.5").unwrap();
        assert_eq!(r, BigRational::new(5.into(), 2.into()));
        let r = BigRational::parse_literal("-1.25e2").unwrap();
        assert_eq!(r, BigRational::from_integer((-125).into()));
        let r = BigRational::parse_literal("1e-2").unwrap();
        assert_eq!(r, BigRational::new(1.into(), 100.into()));
        assert!(BigRational::parse_literal("1/0").is_none());
        assert!(BigRational::parse_literal("x").is_none());
    }

    #[test]
    fn parse_complex() {
        let c = Complex::<f64>::parse_literal("2+3i").unwrap();
        assert_eq!(c, Complex::new(2.0, 3.0));
        let c = Complex::<f64>::parse_literal("-i").unwrap();
        assert_eq!(c, Complex::new(0.0, -1.0));
        let c = Complex::<f64>::parse_literal("i").unwrap();
        assert_eq!(c, Complex::new(0.0, 1.0));
        let c = Complex::<f64>::parse_literal("1.5e-1-2i").unwrap();
        assert_eq!(c, Complex::new(0.15, -2.0));
        let c = Complex::<BigRational>::parse_literal("1/2i").unwrap();
        assert_eq!(c.im, BigRational::new(1.into(), 2.into()));
        let c = Complex::<BigRational>::parse_literal("7").unwrap();
        assert_eq!(c.re, BigRational::from_integer(7.into()));
    }

    #[test]
    fn checked_ints() {
        assert_eq!(5i64.add_ref(&7), 12);
        assert_eq!(i64::parse_literal("6/3"), Some(2));
        assert_eq!(i64::parse_literal("1/2"), None);
    }

    #[test]
    #[should_panic]
    fn int_overflow_panics() {
        let _ = i64::MAX.add_ref(&1);
    }

    #[test]
    fn complex_inverse() {
        let z = Complex::<BigRational>::parse_literal("1+i").unwrap();
        let inv = z.inverse();
        assert!(Coeff::is_one(&z.mul_ref(&inv)));
        let z = Complex::<f64>::new(3.0, -4.0);
        let inv = z.inverse();
        assert!(Coeff::is_one(&z.mul_ref(&inv)));
    }
}
