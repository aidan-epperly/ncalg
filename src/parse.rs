//! A small recursive-descent parser for polynomial expressions.
//!
//! Grammar (whitespace insignificant):
//!
//! ```text
//! expr   := ['+'|'-'] term { ('+'|'-') term }
//! term   := factor { ['*'] factor }            -- juxtaposition multiplies
//! factor := atom [ '^' integer ]
//! atom   := number [ '/' number ]
//!         | identifier                          -- generator (may contain †), or literal such as `i`
//!         | factor ''                           -- postfix adjoint (reverse words, map generators through the involution, conjugate)
//!         | '(' expr ')'
//!         | '[' expr ',' expr ']'               -- commutator
//!         | '{' expr ',' expr '}'               -- anticommutator
//! ```
//!
//! Numbers are passed to [`Coeff::parse_literal`], so `1/2` works for
//! rational and floating point coefficients and fails for integers unless the
//! quotient is exact.

use crate::algebra::FreeAlgebra;
use crate::coeff::Coeff;
use crate::poly::Polynomial;
use std::fmt;

/// Parse failure with a byte offset into the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset of the error in the input.
    pub pos: usize,
    /// What went wrong.
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at offset {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(String),
    Ident(String),
    Plus,
    Prime,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    End,
}

fn tokenize(s: &str) -> Result<Vec<(Tok, usize)>, ParseError> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let start = i;
        let tok = match c {
            '+' => Tok::Plus,
            '\'' => Tok::Prime,
            '-' | '−' => Tok::Minus,
            '*' | '·' => Tok::Star,
            '/' => Tok::Slash,
            '^' => Tok::Caret,
            '(' => Tok::LParen,
            ')' => Tok::RParen,
            '[' => Tok::LBracket,
            ']' => Tok::RBracket,
            '{' => Tok::LBrace,
            '}' => Tok::RBrace,
            ',' => Tok::Comma,
            '#' => {
                // finite-field literal #e (integer encoding), see ff::Fq
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                let t = Tok::Num(s[i..j].to_string());
                i = j;
                out.push((t, start));
                continue;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut j = i;
                while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
                    j += 1;
                }
                // exponent
                if j < bytes.len()
                    && (bytes[j] == b'e' || bytes[j] == b'E')
                    && j + 1 < bytes.len()
                    && (bytes[j + 1].is_ascii_digit()
                        || ((bytes[j + 1] == b'+' || bytes[j + 1] == b'-')
                            && j + 2 < bytes.len()
                            && bytes[j + 2].is_ascii_digit()))
                {
                    j += 2;
                    while j < bytes.len() && bytes[j].is_ascii_digit() {
                        j += 1;
                    }
                }
                let t = Tok::Num(s[i..j].to_string());
                i = j;
                out.push((t, start));
                continue;
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut j = i;
                while j < s.len() {
                    let ch = s[j..].chars().next().unwrap();
                    if ch.is_alphanumeric() || ch == '_' || ch == '†' {
                        j += ch.len_utf8();
                    } else {
                        break;
                    }
                }
                let t = Tok::Ident(s[i..j].to_string());
                i = j;
                out.push((t, start));
                continue;
            }
            _ => {
                return Err(ParseError {
                    pos: i,
                    msg: format!("unexpected character {:?}", c),
                });
            }
        };
        // multi-byte punctuation (−, ·)
        i += c.len_utf8();
        out.push((tok, start));
    }
    out.push((Tok::End, s.len()));
    Ok(out)
}

struct Parser<'a, C: Coeff> {
    alg: &'a FreeAlgebra,
    toks: Vec<(Tok, usize)>,
    pos: usize,
    _c: std::marker::PhantomData<C>,
}

impl<'a, C: Coeff> Parser<'a, C> {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].0
    }
    fn offset(&self) -> usize {
        self.toks[self.pos].1
    }
    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos].0.clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }
    fn err<T>(&self, msg: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            pos: self.offset(),
            msg: msg.into(),
        })
    }
    fn expect(&mut self, t: Tok) -> Result<(), ParseError> {
        if *self.peek() == t {
            self.bump();
            Ok(())
        } else {
            self.err(format!("expected {:?}, found {:?}", t, self.peek()))
        }
    }

    fn expr(&mut self) -> Result<Polynomial<C>, ParseError> {
        let mut negate = false;
        match self.peek() {
            Tok::Plus => {
                self.bump();
            }
            Tok::Minus => {
                self.bump();
                negate = true;
            }
            _ => {}
        }
        // the summands are collected and merged once (a heap merge): adding
        // them one by one re-merged the whole partial sum each time,
        // quadratic in the number of terms (32 000 terms took 12.6 s)
        let first = self.term()?;
        let mut parts = vec![if negate { first.negated() } else { first }];
        loop {
            match self.peek() {
                Tok::Plus => {
                    self.bump();
                    parts.push(self.term()?);
                }
                Tok::Minus => {
                    self.bump();
                    parts.push(self.term()?.negated());
                }
                _ => {
                    if parts.len() == 1 {
                        return Ok(parts.pop().expect("one summand"));
                    }
                    let alg = parts[0].algebra().clone();
                    return Ok(Polynomial::sum(&alg, &parts));
                }
            }
        }
    }

    fn starts_factor(t: &Tok) -> bool {
        matches!(
            t,
            Tok::Num(_) | Tok::Ident(_) | Tok::LParen | Tok::LBracket | Tok::LBrace
        )
    }

    fn term(&mut self) -> Result<Polynomial<C>, ParseError> {
        let mut acc = self.factor()?;
        loop {
            match self.peek() {
                Tok::Star => {
                    self.bump();
                    let f = self.factor()?;
                    acc = acc.mul_ref(&f);
                }
                t if Self::starts_factor(t) => {
                    let f = self.factor()?;
                    acc = acc.mul_ref(&f);
                }
                _ => return Ok(acc),
            }
        }
    }

    fn factor(&mut self) -> Result<Polynomial<C>, ParseError> {
        let mut base = self.atom()?;
        loop {
            match self.peek() {
                Tok::Caret => {
                    self.bump();
                    let neg = if *self.peek() == Tok::Minus {
                        self.bump();
                        true
                    } else {
                        false
                    };
                    base = match self.bump() {
                        Tok::Num(n) if !neg => match n.parse::<u32>() {
                            // x^2^3: left to right would give x^6, the usual
                            // reading is x^8
                            Ok(_) if *self.peek() == Tok::Caret => {
                                return self
                                    .err("chained exponents are ambiguous: use parentheses".to_string())
                            }
                            Ok(e) => base.pow(e),
                            Err(_) => {
                                return self
                                    .err(format!("exponent must be a non-negative integer, found {}", n))
                            }
                        },
                        Tok::Num(n) => return self.err(format!("negative exponent -{} not allowed", n)),
                        other => return self.err(format!("expected integer exponent, found {:?}", other)),
                    };
                }
                // postfix adjoint: x' or (expr)'
                Tok::Prime => {
                    self.bump();
                    base = base.adjoint();
                }
                _ => return Ok(base),
            }
        }
    }

    /// A number in exponent notation (`2e1`) next to generators named like
    /// `e1` is ambiguous: `2e1*e2` was read as 20·e2, not 2·e1·e2.
    fn check_exponent_notation(&self, n: &str) -> Result<(), ParseError> {
        let Some(k) = n.find(['e', 'E']) else {
            return Ok(());
        };
        // only a digit exponent lexes like a generator name (e1, E2, ...)
        let tail = &n[k..];
        if !tail[1..].starts_with(|c: char| c.is_ascii_digit()) {
            return Ok(());
        }
        let clash = self
            .alg
            .names()
            .iter()
            .find(|name| name.len() > 1 && tail.starts_with(name.as_str()));
        match clash {
            Some(name) => self.err(format!(
                "{} is ambiguous next to the generator {}: write {}*{} or the number without exponent notation",
                n,
                name,
                &n[..k],
                &n[k..]
            )),
            None => Ok(()),
        }
    }

    fn literal(&self, text: &str) -> Result<Polynomial<C>, ParseError> {
        match C::parse_literal(text) {
            Some(c) => Ok(self.alg.constant(c)),
            None => self.err(format!(
                "cannot interpret {:?} as a coefficient of type {}",
                text,
                std::any::type_name::<C>()
            )),
        }
    }

    fn atom(&mut self) -> Result<Polynomial<C>, ParseError> {
        match self.bump() {
            Tok::Num(n) => {
                self.check_exponent_notation(&n)?;
                if *self.peek() == Tok::Slash {
                    self.bump();
                    match self.bump() {
                        Tok::Num(d) => {
                            self.check_exponent_notation(&d)?;
                            // `a/b^k` read as (a/b)^k: 2/3^2 gave 4/9
                            if *self.peek() == Tok::Caret {
                                return self.err(format!(
                                    "ambiguous {}/{}^…: write ({}/{})^k or {}/({}^k)",
                                    n, d, n, d, n, d
                                ));
                            }
                            self.literal(&format!("{}/{}", n, d))
                        }
                        other => self.err(format!("expected denominator, found {:?}", other)),
                    }
                } else {
                    self.literal(&n)
                }
            }
            Tok::Ident(name) => {
                if let Some(v) = self.alg.var_index(&name) {
                    Ok(self.alg.var(v))
                } else if let Some(c) = C::parse_literal(&name) {
                    Ok(self.alg.constant(c))
                } else {
                    self.err(format!("unknown generator {:?}", name))
                }
            }
            Tok::LParen => {
                let e = self.expr()?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            Tok::LBracket => {
                let a = self.expr()?;
                self.expect(Tok::Comma)?;
                let b = self.expr()?;
                self.expect(Tok::RBracket)?;
                Ok(Polynomial::commutator(&a, &b))
            }
            Tok::LBrace => {
                let a = self.expr()?;
                self.expect(Tok::Comma)?;
                let b = self.expr()?;
                self.expect(Tok::RBrace)?;
                Ok(Polynomial::anticommutator(&a, &b))
            }
            Tok::Minus => {
                // unary minus inside a product, e.g. 2*-x
                let f = self.factor()?;
                Ok(f.negated())
            }
            other => self.err(format!("unexpected token {:?}", other)),
        }
    }
}

/// Parse `s` as a polynomial over `alg`.
pub fn parse<C: Coeff>(alg: &FreeAlgebra, s: &str) -> Result<Polynomial<C>, ParseError> {
    let toks = tokenize(s)?;
    let mut p = Parser::<C> {
        alg,
        toks,
        pos: 0,
        _c: std::marker::PhantomData,
    };
    if *p.peek() == Tok::End {
        return Err(ParseError {
            pos: 0,
            msg: "empty input".into(),
        });
    }
    let e = p.expr()?;
    if *p.peek() != Tok::End {
        return p.err(format!("unexpected trailing token {:?}", p.peek()));
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use num_rational::BigRational;

    #[test]
    fn parse_and_print() {
        let alg = FreeAlgebra::new(["x", "y"]);
        let p: Polynomial<i64> = alg.parse("2*x*y - y*x + x^2 - 1").unwrap();
        assert_eq!(p.to_string(), "-y*x + 2*x*y + x^2 - 1");
        let q: Polynomial<i64> = alg.parse("x y x - (x - y)^2").unwrap();
        assert_eq!(q.to_string(), "x*y*x - y^2 + y*x + x*y - x^2");
        let c: Polynomial<i64> = alg.parse("[x, y]").unwrap();
        assert_eq!(c.to_string(), "-y*x + x*y");
        let a: Polynomial<i64> = alg.parse("{x, y}").unwrap();
        assert_eq!(a.to_string(), "y*x + x*y");
        let z: Polynomial<i64> = alg.parse("x - x").unwrap();
        assert!(z.is_zero());
        assert_eq!(z.to_string(), "0");
    }

    #[test]
    fn parse_rationals_and_complex() {
        let alg = FreeAlgebra::new(["x", "y"]);
        let p: Polynomial<BigRational> = alg.parse("1/2*x + 0.25").unwrap();
        assert_eq!(p.to_string(), "1/2*x + 1/4");
        let p: Polynomial<Complex<f64>> = alg.parse("i*x*y - 2*i + (1+2i)*y").unwrap();
        assert_eq!(p.coeff_of(&[0, 1]), Complex::new(0.0, 1.0));
        assert_eq!(p.coeff_of(&[1]), Complex::new(1.0, 2.0));
        assert_eq!(p.coeff_of(&[]), Complex::new(0.0, -2.0));
    }

    #[test]
    fn parse_errors() {
        let alg = FreeAlgebra::new(["x", "y"]);
        assert!(alg.parse::<i64>("x + z").is_err());
        assert!(alg.parse::<i64>("1/2*x").is_err());
        assert!(alg.parse::<i64>("x +").is_err());
        assert!(alg.parse::<i64>("(x").is_err());
        assert!(alg.parse::<i64>("x^-1").is_err());
        assert!(alg.parse::<i64>("").is_err());
    }
}
