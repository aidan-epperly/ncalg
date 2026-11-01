//! Polynomials in commuting variables `x_0, x_1, …` over a [`Coeff`] ring.
//!
//! An [`MPoly`] maps exponent vectors to nonzero coefficients, so equality is
//! structural.  Variables are numbered; names are supplied when printing
//! ([`MPoly::display`]) and parsing ([`MPoly::parse`]).  With rational
//! coefficients ([`MPoly<BigRational>`]) there are helpers for integer-valued
//! polynomials in integer parameters: evaluation at integers and the parity
//! of the value.
//!
//! ```
//! use ncalg::commutative::MPoly;
//! use ncalg::BigRational;
//!
//! let names = ["n".to_string(), "m".to_string()];
//! let p = MPoly::<BigRational>::parse("n(n+1)/2 - 2m", &names).unwrap();
//! assert_eq!(p.display(&names), "1/2*n^2 + 1/2*n - 2*m");
//! assert_eq!(p.eval_i64(&[3, 1]), BigRational::from_integer(4.into()));
//! let q = &p * &MPoly::var(1);
//! assert_eq!(q.display(&names), "1/2*n^2*m + 1/2*n*m - 2*m^2");
//! ```

use crate::coeff::{Coeff, Field};
use crate::parse::ParseError;
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A polynomial in commuting variables; see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub struct MPoly<C: Coeff> {
    /// Exponent vector (no trailing zeros) → nonzero coefficient.
    terms: BTreeMap<Vec<u32>, C>,
}

fn trim(mut e: Vec<u32>) -> Vec<u32> {
    while e.last() == Some(&0) {
        e.pop();
    }
    e
}

impl<C: Coeff + Eq> Eq for MPoly<C> {}

impl<C: Coeff> Default for MPoly<C> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<C: Coeff> MPoly<C> {
    /// The zero polynomial.
    pub fn zero() -> Self {
        MPoly {
            terms: BTreeMap::new(),
        }
    }

    /// The constant `c`.
    pub fn constant(c: C) -> Self {
        Self::monomial(&[], c)
    }

    /// The integer constant `n`.
    pub fn from_i64(n: i64) -> Self {
        Self::constant(C::from_i64(n))
    }

    /// The variable `x_i`.
    pub fn var(i: usize) -> Self {
        let mut e = vec![0; i + 1];
        e[i] = 1;
        Self::monomial(&e, C::one())
    }

    /// `c · Π x_i^{exponents[i]}`.
    pub fn monomial(exponents: &[u32], c: C) -> Self {
        let mut p = Self::zero();
        p.add_term(exponents.to_vec(), c);
        p
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// The value, if the polynomial is constant.
    pub fn as_constant(&self) -> Option<C> {
        match self.terms.len() {
            0 => Some(C::zero()),
            1 => self.terms.get(&Vec::new()).cloned(),
            _ => None,
        }
    }

    /// The total degree (`None` for zero).
    pub fn degree(&self) -> Option<u32> {
        self.terms.keys().map(|e| e.iter().sum()).max()
    }

    /// The number of variables the polynomial involves, counting up to the
    /// highest index used.
    pub fn num_vars(&self) -> usize {
        self.terms.keys().map(Vec::len).max().unwrap_or(0)
    }

    /// The nonzero terms `(exponents, coefficient)`; exponent vectors omit
    /// trailing zeros.
    pub fn terms(&self) -> impl Iterator<Item = (&[u32], &C)> + '_ {
        self.terms.iter().map(|(e, c)| (e.as_slice(), c))
    }

    /// The value at `values` (one per variable; missing ones count as 0),
    /// computed in a ring `R` that the coefficients embed into.
    pub fn eval_with<R: Coeff>(&self, values: &[R], embed: impl Fn(&C) -> R) -> R {
        let mut s = R::zero();
        for (e, c) in &self.terms {
            let mut t = embed(c);
            for (i, &k) in e.iter().enumerate() {
                let v = values.get(i).cloned().unwrap_or_else(R::zero);
                for _ in 0..k {
                    t = t.mul_ref(&v);
                }
            }
            s = s.add_ref(&t);
        }
        s
    }

    /// `self^k`.
    pub fn pow(&self, k: u32) -> Self {
        let mut r = Self::from_i64(1);
        for _ in 0..k {
            r = &r * self;
        }
        r
    }

    /// The polynomial written with these variable names (`x_i` for missing
    /// ones), terms by decreasing total degree, e.g. `n^2 + 2*n*m - 1`.
    pub fn display(&self, names: &[String]) -> String {
        if self.is_zero() {
            return "0".into();
        }
        let deg = |e: &Vec<u32>| e.iter().sum::<u32>();
        let mut terms: Vec<_> = self.terms.iter().collect();
        terms.sort_by(|a, b| deg(b.0).cmp(&deg(a.0)).then(b.0.cmp(a.0)));
        let mut s = String::new();
        for (j, (e, c)) in terms.into_iter().enumerate() {
            let mono: Vec<String> = e
                .iter()
                .enumerate()
                .filter(|(_, &k)| k > 0)
                .map(|(i, &k)| {
                    let name = names.get(i).cloned().unwrap_or_else(|| format!("x{i}"));
                    if k == 1 {
                        name
                    } else {
                        format!("{name}^{k}")
                    }
                })
                .collect();
            let neg = c.is_negative_for_display();
            let mag = if neg { c.negated() } else { c.clone() };
            let coeff = if mag.needs_parens_for_display() {
                format!("({mag})")
            } else {
                mag.to_string()
            };
            let body = match (mono.is_empty(), mag.is_one()) {
                (true, _) => coeff,
                (false, true) => mono.join("*"),
                (false, false) => format!("{coeff}*{}", mono.join("*")),
            };
            s += &match (j, neg) {
                (0, false) => body,
                (0, true) => format!("-{body}"),
                (_, false) => format!(" + {body}"),
                (_, true) => format!(" - {body}"),
            };
        }
        s
    }

    fn add_term(&mut self, e: Vec<u32>, c: C) {
        if c.is_zero() {
            return;
        }
        let e = trim(e);
        let entry = self.terms.entry(e.clone()).or_insert_with(C::zero);
        entry.add_assign_ref(&c);
        if entry.is_zero() {
            self.terms.remove(&e);
        }
    }

    fn mul_ref(&self, o: &Self) -> Self {
        let mut p = Self::zero();
        for (e1, c1) in &self.terms {
            for (e2, c2) in &o.terms {
                let len = e1.len().max(e2.len());
                let e = (0..len)
                    .map(|i| [e1, e2].iter().map(|e| e.get(i).unwrap_or(&0)).sum())
                    .collect();
                p.add_term(e, c1.mul_ref(c2));
            }
        }
        p
    }
}

impl<C: Field> MPoly<C> {
    /// Parses an expression in the named variables: numbers, names, `+ - *`,
    /// `/` by a nonzero constant, `^` with a natural exponent, parentheses, and
    /// implicit products like `2n`, `n(n+1)` or `nm` (when `nm` is not itself
    /// a name, a run of one-letter names).
    pub fn parse(text: &str, names: &[String]) -> Result<Self, ParseError> {
        let mut p = Parser {
            s: text.char_indices().collect(),
            len: text.len(),
            i: 0,
            names,
        };
        p.skip();
        if p.i == p.s.len() {
            return Err(p.error("empty expression"));
        }
        let v = p.sum()?;
        if p.i < p.s.len() {
            return Err(p.error(&format!("unexpected '{}'", p.s[p.i].1)));
        }
        Ok(v)
    }
}

impl MPoly<BigRational> {
    /// The value at integer `values`.
    pub fn eval_i64(&self, values: &[i64]) -> BigRational {
        let v: Vec<BigRational> = values
            .iter()
            .map(|&x| BigRational::from_integer(x.into()))
            .collect();
        self.eval_with(&v, BigRational::clone)
    }

    /// The value at integer `values`, if it is an integer.
    pub fn eval_integer(&self, values: &[i64]) -> Option<BigInt> {
        let v = self.eval_i64(values);
        v.is_integer().then(|| v.to_integer())
    }

    /// The value, if the polynomial is a constant integer that fits an `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        self.as_constant()
            .filter(|c| c.is_integer())
            .and_then(|c| c.to_integer().to_i64())
    }

    /// The parity of the value (`Some(true)`: odd) for all integer values of
    /// the variables with the given parities (`Some(true)`: odd), if that
    /// is determined: every coefficient must be an integer, and every
    /// variable in a term with an odd coefficient must have a known parity.
    /// (An integer-valued polynomial like `n(n+1)/2` has no fixed parity.)
    pub fn parity(&self, parities: &[Option<bool>]) -> Option<bool> {
        let mut odd = false;
        for (e, c) in &self.terms {
            if !c.is_integer() {
                return None;
            }
            if c.to_integer().is_even() {
                continue;
            }
            let mut t = true;
            for (i, &k) in e.iter().enumerate() {
                if k > 0 {
                    t &= (*parities.get(i)?)?;
                }
            }
            odd ^= t;
        }
        Some(odd)
    }
}

impl<C: Coeff> fmt::Display for MPoly<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display(&[]))
    }
}

struct Parser<'a> {
    s: Vec<(usize, char)>,
    len: usize,
    i: usize,
    names: &'a [String],
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).map(|x| x.1)
    }

    fn skip(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.i += 1;
        }
    }

    fn bump(&mut self) {
        self.i += 1;
        self.skip();
    }

    fn error(&self, msg: &str) -> ParseError {
        ParseError {
            pos: self.s.get(self.i).map_or(self.len, |x| x.0),
            msg: msg.into(),
        }
    }

    fn sum<C: Field>(&mut self) -> Result<MPoly<C>, ParseError> {
        let mut v = self.product()?;
        while let Some(c @ ('+' | '-')) = self.peek() {
            self.bump();
            let t = self.product()?;
            v = if c == '+' { &v + &t } else { &v - &t };
        }
        Ok(v)
    }

    fn product<C: Field>(&mut self) -> Result<MPoly<C>, ParseError> {
        let mut v = self.power()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.bump();
                    v = &v * &self.power()?;
                }
                Some('/') => {
                    self.bump();
                    let at = self.error("");
                    let d: MPoly<C> = self.power()?;
                    let c = d.as_constant().filter(|c: &C| !c.is_zero()).ok_or(ParseError {
                        msg: "division only by a nonzero constant".into(),
                        ..at
                    })?;
                    v = &v * &MPoly::constant(c.inverse());
                }
                // implicit products: 2n, n(n+1), (n+1)(n+2)
                Some(c) if c == '(' || c.is_alphabetic() || c == '_' => v = &v * &self.power()?,
                _ => return Ok(v),
            }
        }
    }

    fn power<C: Field>(&mut self) -> Result<MPoly<C>, ParseError> {
        if self.peek() == Some('-') {
            self.bump();
            return Ok(-self.power()?);
        }
        let base = self.atom()?;
        if self.peek() == Some('^') {
            self.bump();
            let start = self.i;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
            let digits: String = self.s[start..self.i].iter().map(|x| x.1).collect();
            let k: u32 = digits
                .parse()
                .map_err(|_| self.error("^ needs a natural number"))?;
            self.skip();
            return Ok(base.pow(k));
        }
        Ok(base)
    }

    fn atom<C: Field>(&mut self) -> Result<MPoly<C>, ParseError> {
        match self.peek() {
            Some('(') => {
                self.bump();
                let v = self.sum()?;
                if self.peek() != Some(')') {
                    return Err(self.error("missing )"));
                }
                self.bump();
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.i;
                while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '.') {
                    self.i += 1;
                }
                let t: String = self.s[start..self.i].iter().map(|x| x.1).collect();
                let c = C::parse_literal(&t).ok_or_else(|| self.error(&format!("bad number {t}")))?;
                self.skip();
                Ok(MPoly::constant(c))
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                let start = self.i;
                while self.peek().is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    self.i += 1;
                }
                let name: String = self.s[start..self.i].iter().map(|x| x.1).collect();
                let at = ParseError {
                    pos: self.s[start].0,
                    msg: format!("unknown variable {name}"),
                };
                self.skip();
                if let Some(k) = self.names.iter().position(|x| *x == name) {
                    return Ok(MPoly::var(k));
                }
                // a run of one-letter names, like nm
                let mut v = MPoly::from_i64(1);
                for ch in name.chars() {
                    let k = self
                        .names
                        .iter()
                        .position(|x| *x == ch.to_string())
                        .ok_or(at.clone())?;
                    v = &v * &MPoly::var(k);
                }
                Ok(v)
            }
            Some(c) => Err(self.error(&format!("unexpected '{c}'"))),
            None => Err(self.error("the expression ends too early")),
        }
    }
}

impl<C: Coeff> Add for &MPoly<C> {
    type Output = MPoly<C>;
    fn add(self, o: &MPoly<C>) -> MPoly<C> {
        let mut p = self.clone();
        for (e, c) in &o.terms {
            p.add_term(e.clone(), c.clone());
        }
        p
    }
}

impl<C: Coeff> Sub for &MPoly<C> {
    type Output = MPoly<C>;
    fn sub(self, o: &MPoly<C>) -> MPoly<C> {
        self + &(-o)
    }
}

impl<C: Coeff> Neg for &MPoly<C> {
    type Output = MPoly<C>;
    fn neg(self) -> MPoly<C> {
        MPoly {
            terms: self.terms.iter().map(|(e, c)| (e.clone(), c.negated())).collect(),
        }
    }
}

impl<C: Coeff> Neg for MPoly<C> {
    type Output = MPoly<C>;
    fn neg(self) -> MPoly<C> {
        -&self
    }
}

impl<C: Coeff> Mul for &MPoly<C> {
    type Output = MPoly<C>;
    fn mul(self, o: &MPoly<C>) -> MPoly<C> {
        self.mul_ref(o)
    }
}

macro_rules! by_value {
    ($tr:ident, $f:ident) => {
        impl<C: Coeff> $tr for MPoly<C> {
            type Output = MPoly<C>;
            fn $f(self, o: MPoly<C>) -> MPoly<C> {
                (&self).$f(&o)
            }
        }
        impl<C: Coeff> $tr<&MPoly<C>> for MPoly<C> {
            type Output = MPoly<C>;
            fn $f(self, o: &MPoly<C>) -> MPoly<C> {
                (&self).$f(o)
            }
        }
        impl<C: Coeff> $tr<i64> for MPoly<C> {
            type Output = MPoly<C>;
            fn $f(self, o: i64) -> MPoly<C> {
                (&self).$f(&MPoly::from_i64(o))
            }
        }
        impl<C: Coeff> $tr<i64> for &MPoly<C> {
            type Output = MPoly<C>;
            fn $f(self, o: i64) -> MPoly<C> {
                self.$f(&MPoly::from_i64(o))
            }
        }
    };
}
by_value!(Add, add);
by_value!(Sub, sub);
by_value!(Mul, mul);

impl<C: Coeff> From<i64> for MPoly<C> {
    fn from(n: i64) -> Self {
        Self::from_i64(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn arithmetic_display_and_parity() {
        let names = vec!["n".to_string(), "m".to_string()];
        let (n, m) = (MPoly::<BigRational>::var(0), MPoly::var(1));
        let p = &(&n + 1) * &n - &m * 2;
        assert_eq!(p.display(&names), "n^2 + n - 2*m");
        assert_eq!(p.to_string(), "x0^2 + x0 - 2*x1");
        assert_eq!(p.eval_integer(&[3, 5]), Some(2.into()));
        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.num_vars(), 2);
        assert!((&p - &p).is_zero());
        assert_eq!((&(&n + 1) * &n).parity(&[None, None]), None);
        assert_eq!((&m * 2 + 1).parity(&[None, None]), Some(true));
        assert_eq!(m.parity(&[None, Some(true)]), Some(true));
        assert_eq!(MPoly::<BigRational>::from_i64(-3).as_i64(), Some(-3));
        assert_eq!((&n * &n).eval_with(&[q(4)], BigRational::clone), q(16));
    }

    #[test]
    fn parsing() {
        let names = vec!["n".to_string(), "m".to_string()];
        let parse = |t: &str| MPoly::<BigRational>::parse(t, &names).map(|p| p.display(&names));
        assert_eq!(parse("n^2 + n - 2*m").unwrap(), "n^2 + n - 2*m");
        assert_eq!(parse("2n(n+1)/2 - nm").unwrap(), "n^2 - n*m + n");
        assert_eq!(parse(" -(m - 1) ").unwrap(), "-m + 1");
        assert_eq!(parse("(n+1)^3 - n^3").unwrap(), "3*n^2 + 3*n + 1");
        assert_eq!(parse("1/2").unwrap(), "1/2");
        let e = MPoly::<BigRational>::parse("n + k", &names).unwrap_err();
        assert_eq!(e.pos, 4);
        assert!(e.msg.contains("unknown variable k"));
        assert!(MPoly::<BigRational>::parse("n/m", &names).is_err());
        assert!(MPoly::<BigRational>::parse("(n", &names).is_err());
        assert!(MPoly::<BigRational>::parse("", &names).is_err());
        // floating point coefficients
        let f = MPoly::<f64>::parse("0.5n + 2", &names).unwrap();
        assert_eq!(f.eval_with(&[3.0], |c| *c), 3.5);
    }
}
