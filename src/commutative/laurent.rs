//! Laurent polynomials `Σ c_k t^k` (`k ∈ ℤ`) in one variable over a
//! [`Coeff`] ring.
//!
//! Storage is dense from the lowest to the highest nonzero term, so equality
//! is structural.  Exponents can be displayed as fractions
//! ([`Laurent::display_as`]), for polynomials in a root of the variable such
//! as the Jones polynomial in `t^{1/2}`.
//!
//! ```
//! use ncalg::commutative::Laurent;
//! use ncalg::BigInt;
//!
//! let t = Laurent::<BigInt>::t_pow(1);
//! let p = &(&t - &Laurent::one()) * &(&t + &Laurent::t_pow(-1));
//! assert_eq!(p.to_string(), "-t^-1 + 1 - t + t^2");
//! assert_eq!(p.div_exact(&(&t - &Laurent::one())).unwrap(), &t + &Laurent::t_pow(-1));
//! ```

use crate::coeff::{Coeff, Field};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A Laurent polynomial `Σ_{k=low}^{low+len-1} c_k t^k`; the first and last
/// stored coefficients are nonzero (the zero polynomial stores nothing).
#[derive(Clone, PartialEq, Debug)]
pub struct Laurent<C: Coeff> {
    low: i64,
    coeffs: Vec<C>,
}

impl<C: Coeff> Laurent<C> {
    /// The zero polynomial.
    pub fn zero() -> Self {
        Laurent {
            low: 0,
            coeffs: Vec::new(),
        }
    }

    /// The constant `1`.
    pub fn one() -> Self {
        Self::constant(C::one())
    }

    /// The constant `c`.
    pub fn constant(c: C) -> Self {
        Self::monomial(0, c)
    }

    /// `c t^k`.
    pub fn monomial(k: i64, c: C) -> Self {
        Self::from_coeffs(k, vec![c])
    }

    /// `t^k`.
    pub fn t_pow(k: i64) -> Self {
        Self::monomial(k, C::one())
    }

    /// `Σ_j coeffs[j] t^{low + j}`.
    pub fn from_coeffs(low: i64, coeffs: Vec<C>) -> Self {
        let mut p = Laurent { low, coeffs };
        p.trim();
        p
    }

    /// `Σ c t^k` over the given terms (repeated exponents add).
    pub fn from_terms(terms: impl IntoIterator<Item = (i64, C)>) -> Self {
        let mut p = Self::zero();
        for (k, c) in terms {
            p = p.add_ref(&Self::monomial(k, c));
        }
        p
    }

    fn trim(&mut self) {
        let Some(first) = self.coeffs.iter().position(|c| !c.is_zero()) else {
            self.coeffs.clear();
            self.low = 0;
            return;
        };
        let last = self.coeffs.iter().rposition(|c| !c.is_zero()).unwrap();
        self.coeffs.truncate(last + 1);
        self.coeffs.drain(..first);
        self.low += first as i64;
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The lowest exponent with a nonzero coefficient (`None` for zero).
    pub fn low_degree(&self) -> Option<i64> {
        (!self.is_zero()).then_some(self.low)
    }

    /// The highest exponent with a nonzero coefficient (`None` for zero).
    pub fn high_degree(&self) -> Option<i64> {
        (!self.is_zero()).then(|| self.low + self.coeffs.len() as i64 - 1)
    }

    /// The coefficient of `t^k`.
    pub fn coeff(&self, k: i64) -> C {
        let j = k - self.low;
        if j < 0 || j >= self.coeffs.len() as i64 {
            C::zero()
        } else {
            self.coeffs[j as usize].clone()
        }
    }

    /// The nonzero terms `(k, c_k)` in increasing order of `k`.
    pub fn terms(&self) -> impl Iterator<Item = (i64, &C)> + '_ {
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(move |(j, c)| (self.low + j as i64, c))
    }

    /// `self + other`.
    pub fn add_ref(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let low = self.low.min(other.low);
        let high = self.high_degree().unwrap().max(other.high_degree().unwrap());
        let coeffs = (low..=high)
            .map(|k| self.coeff(k).add_ref(&other.coeff(k)))
            .collect();
        Self::from_coeffs(low, coeffs)
    }

    /// `self - other`.
    pub fn sub_ref(&self, other: &Self) -> Self {
        self.add_ref(&other.negated())
    }

    /// `-self`.
    pub fn negated(&self) -> Self {
        Laurent {
            low: self.low,
            coeffs: self.coeffs.iter().map(|c| c.negated()).collect(),
        }
    }

    /// `self · other`.
    pub fn mul_ref(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![C::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                if !b.is_zero() {
                    out[i + j].add_assign_ref(&a.mul_ref(b));
                }
            }
        }
        Self::from_coeffs(self.low + other.low, out)
    }

    /// `self^e`.
    pub fn pow(&self, e: u32) -> Self {
        let mut out = Self::one();
        for _ in 0..e {
            out = out.mul_ref(self);
        }
        out
    }

    /// `c · self`.
    pub fn scale(&self, c: &C) -> Self {
        Self::from_coeffs(self.low, self.coeffs.iter().map(|x| x.mul_ref(c)).collect())
    }

    /// `t^k · self`.
    pub fn shift(&self, k: i64) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        Laurent {
            low: self.low + k,
            coeffs: self.coeffs.clone(),
        }
    }

    /// `self(t^k)` for `k ≠ 0`, e.g. `k = -1` for `t ↦ t^{-1}`.
    pub fn substitute_power(&self, k: i64) -> Self {
        assert!(k != 0, "t ↦ t^0 is not a substitution");
        Self::from_terms(self.terms().map(|(e, c)| (e * k, c.clone())))
    }

    /// `self(−t)`.
    pub fn negate_variable(&self) -> Self {
        Self::from_terms(self.terms().map(|(e, c)| {
            (
                e,
                if e.rem_euclid(2) == 1 {
                    c.negated()
                } else {
                    c.clone()
                },
            )
        }))
    }

    /// Is every exponent a multiple of `m`?
    pub fn exponents_divisible_by(&self, m: i64) -> bool {
        self.terms().all(|(e, _)| e.rem_euclid(m) == 0)
    }

    /// `self(t^{1/m})`: divides every exponent by `m`.  Panics unless
    /// [`exponents_divisible_by`](Self::exponents_divisible_by)`(m)`.
    pub fn compress_exponents(&self, m: i64) -> Self {
        assert!(self.exponents_divisible_by(m), "exponents not divisible by {m}");
        Self::from_terms(self.terms().map(|(e, c)| (e / m, c.clone())))
    }

    /// `self(x)` in a field `F`, with `embed` mapping coefficients into it.
    /// Panics if `x = 0` and a negative power occurs.
    pub fn eval_with<F: Field>(&self, x: &F, embed: impl Fn(&C) -> F) -> F {
        let mut sum = F::zero();
        for (e, c) in self.terms() {
            let base = if e < 0 { x.inverse() } else { x.clone() };
            let mut p = F::one();
            for _ in 0..e.unsigned_abs() {
                p = p.mul_ref(&base);
            }
            sum.add_assign_ref(&embed(c).mul_ref(&p));
        }
        sum
    }

    /// The exact quotient `self / d` when the leading coefficient of `d` is
    /// `±1` and `d` divides `self`; `None` otherwise.
    pub fn div_exact(&self, d: &Self) -> Option<Self> {
        let lead = d.coeffs.last()?.clone();
        let unit = if lead.is_one() {
            C::one()
        } else if lead.negated().is_one() {
            C::one().negated()
        } else {
            return None;
        };
        let dh = d.high_degree().unwrap();
        let mut rem = self.clone();
        let mut quot = Self::zero();
        while let Some(rh) = rem.high_degree() {
            // the quotient's terms run from rh - dh down to (low of self) - d.low
            if rh - dh < rem.low - d.low {
                return None;
            }
            let q = Self::monomial(rh - dh, rem.coeff(rh).mul_ref(&unit));
            rem = rem.sub_ref(&q.mul_ref(d));
            quot = quot.add_ref(&q);
        }
        Some(quot)
    }

    /// Formats with variable name `var`, the exponent `k` printed as `k/den`
    /// (`den = 2` prints `t^{1/2}` powers as `t^(1/2)`, `t`, `t^(3/2)`, …).
    pub fn display_as(&self, var: &str, den: i64) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut s = String::new();
        for (i, (e, c)) in self.terms().enumerate() {
            let neg = c.is_negative_for_display();
            let mag = if neg { c.negated() } else { c.clone() };
            if i == 0 {
                if neg {
                    s.push('-');
                }
            } else {
                s.push_str(if neg { " - " } else { " + " });
            }
            let exp = fmt_exponent(e, den);
            match (mag.is_one(), exp) {
                (true, None) => s.push('1'),
                (false, None) => s.push_str(&mag.to_string()),
                (true, Some(x)) => s.push_str(&format!("{var}{x}")),
                (false, Some(x)) => s.push_str(&format!("{mag}*{var}{x}")),
            }
        }
        s
    }
}

/// `None` for exponent 0, else `""`, `^k`, `^(k/den)`.
fn fmt_exponent(e: i64, den: i64) -> Option<String> {
    if e == 0 {
        return None;
    }
    let g = num_integer::gcd(e, den);
    let (n, d) = (e / g, den / g);
    Some(match (n, d) {
        (1, 1) => String::new(),
        (n, 1) => format!("^{n}"),
        (n, d) => format!("^({n}/{d})"),
    })
}

impl<C: Coeff> fmt::Display for Laurent<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display_as("t", 1))
    }
}

macro_rules! binop {
    ($tr:ident, $f:ident, $m:ident) => {
        impl<C: Coeff> $tr<&Laurent<C>> for &Laurent<C> {
            type Output = Laurent<C>;
            fn $f(self, o: &Laurent<C>) -> Laurent<C> {
                self.$m(o)
            }
        }
        impl<C: Coeff> $tr for Laurent<C> {
            type Output = Laurent<C>;
            fn $f(self, o: Laurent<C>) -> Laurent<C> {
                self.$m(&o)
            }
        }
    };
}
binop!(Add, add, add_ref);
binop!(Sub, sub, sub_ref);
binop!(Mul, mul, mul_ref);

impl<C: Coeff> Neg for &Laurent<C> {
    type Output = Laurent<C>;
    fn neg(self) -> Laurent<C> {
        self.negated()
    }
}

impl<C: Coeff> Neg for Laurent<C> {
    type Output = Laurent<C>;
    fn neg(self) -> Laurent<C> {
        self.negated()
    }
}
