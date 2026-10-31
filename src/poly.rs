//! Noncommutative polynomials: elements of a free associative algebra.
//!
//! A [`Polynomial`] is a finite sum of terms `c · w` with `c` a coefficient
//! and `w` a [`Word`].  Terms are kept sorted in strictly decreasing monomial
//! order (the ordering of the owning [`FreeAlgebra`]) with no zero
//! coefficients, so the leading term is always `terms()[0]` and structural
//! equality is polynomial equality.

use crate::algebra::FreeAlgebra;
use crate::coeff::{fmt_coeff, Coeff, CoeffConvert, Field, IntegerCoeff};
use crate::reduce::Accumulator;
use crate::word::{Var, Word};
use num_bigint::BigInt;
use num_complex::Complex;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::One;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// One term `coeff · word` of a polynomial.
#[derive(Clone, Debug, PartialEq)]
pub struct Term<C> {
    /// The monomial.
    pub word: Word,
    /// Its coefficient (nonzero in a normalized polynomial).
    pub coeff: C,
}

/// An element of a free associative algebra with coefficients in `C`.
#[derive(Clone)]
pub struct Polynomial<C: Coeff> {
    alg: FreeAlgebra,
    /// Sorted strictly decreasing in `alg`'s monomial order; no zero coefficients.
    terms: Vec<Term<C>>,
}

impl<C: Coeff> Polynomial<C> {
    // ----------------------------------------------------------------------
    // construction
    // ----------------------------------------------------------------------

    /// The zero polynomial of `alg`.
    pub fn zero(alg: FreeAlgebra) -> Self {
        Polynomial {
            alg,
            terms: Vec::new(),
        }
    }

    /// Build a polynomial from arbitrary (word, coefficient) pairs; terms are
    /// sorted, like terms combined and zero terms dropped.
    pub fn from_terms<I>(alg: FreeAlgebra, terms: I) -> Self
    where
        I: IntoIterator<Item = (Word, C)>,
    {
        let mut v: Vec<Term<C>> = terms
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(word, coeff)| Term { word, coeff })
            .collect();
        for t in &v {
            alg.check_word(&t.word);
        }
        Self::normalize(&alg, &mut v);
        Polynomial { alg, terms: v }
    }

    /// Sort descending and merge equal words in place.
    pub(crate) fn normalize(alg: &FreeAlgebra, v: &mut Vec<Term<C>>) {
        v.sort_by(|a, b| alg.cmp_words(&b.word, &a.word));
        let mut out: Vec<Term<C>> = Vec::with_capacity(v.len());
        // size of the summands of the current run, for the cancellation test
        // of inexact coefficients
        let mut run_abs = 0.0f64;
        let mut merged = false;
        for t in v.drain(..) {
            match out.last_mut() {
                Some(last) if last.word == t.word => {
                    if C::INEXACT {
                        run_abs += t.coeff.magnitude();
                    }
                    last.coeff.add_assign_ref(&t.coeff);
                    merged = true;
                }
                _ => {
                    if merged && out.last().is_some_and(|l| C::canceled(&l.coeff, run_abs)) {
                        out.pop();
                    }
                    merged = false;
                    if !t.coeff.is_zero() {
                        if C::INEXACT {
                            run_abs = t.coeff.magnitude();
                        }
                        out.push(t)
                    }
                }
            }
        }
        if merged && out.last().is_some_and(|l| C::canceled(&l.coeff, run_abs)) {
            out.pop();
        }
        *v = out;
    }

    /// Wrap terms that are already sorted, merged and zero-free.
    pub(crate) fn from_sorted_terms_unchecked(alg: FreeAlgebra, terms: Vec<Term<C>>) -> Self {
        debug_assert!(terms
            .windows(2)
            .all(|w| alg.cmp_words(&w[0].word, &w[1].word) == Ordering::Greater));
        debug_assert!(terms.iter().all(|t| !t.coeff.is_zero()));
        Polynomial { alg, terms }
    }

    /// Sum of an iterator of polynomials.
    pub fn sum<'a, I>(alg: &FreeAlgebra, polys: I) -> Self
    where
        I: IntoIterator<Item = &'a Polynomial<C>>,
    {
        let mut acc = Accumulator::new(alg);
        for p in polys {
            acc.push_poly(p);
        }
        acc.finish()
    }

    /// Product of an iterator of polynomials, left to right.
    pub fn product<'a, I>(alg: &FreeAlgebra, polys: I) -> Self
    where
        I: IntoIterator<Item = &'a Polynomial<C>>,
    {
        let mut acc = alg.one();
        for p in polys {
            acc = acc.mul_ref(p);
        }
        acc
    }

    // ----------------------------------------------------------------------
    // inspection
    // ----------------------------------------------------------------------

    /// The algebra this polynomial lives in.
    #[inline]
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// Terms in strictly decreasing monomial order.
    #[inline]
    pub fn terms(&self) -> &[Term<C>] {
        &self.terms
    }

    /// Consume into the term vector (sorted decreasing).
    pub fn into_terms(self) -> Vec<Term<C>> {
        self.terms
    }

    /// Iterate over `(word, coefficient)` in decreasing order.
    pub fn iter(&self) -> impl Iterator<Item = (&Word, &C)> {
        self.terms.iter().map(|t| (&t.word, &t.coeff))
    }

    /// Number of terms.
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    #[inline]
    /// No terms?
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Is this a constant (including zero)?
    pub fn is_constant(&self) -> bool {
        self.terms.iter().all(|t| t.word.is_empty())
    }

    /// Coefficient of the empty word.
    pub fn constant_term(&self) -> C {
        self.coeff_of(&[])
    }

    /// Total degree (length of the longest word); `None` for zero.
    pub fn degree(&self) -> Option<usize> {
        self.terms.iter().map(|t| t.word.len()).max()
    }

    /// Smallest word length occurring; `None` for zero.
    pub fn min_degree(&self) -> Option<usize> {
        self.terms.iter().map(|t| t.word.len()).min()
    }

    /// Leading term (largest monomial), if nonzero.
    #[inline]
    pub fn leading_term(&self) -> Option<&Term<C>> {
        self.terms.first()
    }

    /// Leading word, if nonzero.
    #[inline]
    pub fn leading_word(&self) -> Option<&Word> {
        self.terms.first().map(|t| &t.word)
    }

    /// Leading coefficient, if nonzero.
    #[inline]
    pub fn leading_coeff(&self) -> Option<&C> {
        self.terms.first().map(|t| &t.coeff)
    }

    /// Everything except the leading term.
    pub fn tail(&self) -> Polynomial<C> {
        Polynomial {
            alg: self.alg.clone(),
            terms: self.terms.iter().skip(1).cloned().collect(),
        }
    }

    /// Coefficient of a word (zero if absent).  Binary search.
    pub fn coeff_of(&self, w: &[Var]) -> C {
        match self.terms.binary_search_by(|t| self.alg.cmp_words(w, &t.word)) {
            Ok(i) => self.terms[i].coeff.clone(),
            Err(_) => C::zero(),
        }
    }

    /// Does `w` occur with nonzero coefficient?
    pub fn contains_word(&self, w: &[Var]) -> bool {
        self.terms
            .binary_search_by(|t| self.alg.cmp_words(w, &t.word))
            .is_ok()
    }

    /// Set of generators that actually occur.
    pub fn support_vars(&self) -> Vec<Var> {
        let mut seen = vec![false; self.alg.num_vars()];
        for t in &self.terms {
            for &v in t.word.as_slice() {
                seen[v as usize] = true;
            }
        }
        (0..seen.len() as Var).filter(|&v| seen[v as usize]).collect()
    }

    /// Is every term of the same degree?
    pub fn is_homogeneous(&self) -> bool {
        match self.terms.first() {
            None => true,
            Some(t) => {
                let d = t.word.len();
                self.terms.iter().all(|t| t.word.len() == d)
            }
        }
    }

    /// Largest coefficient magnitude (0 for zero polynomial).
    pub fn max_coeff_magnitude(&self) -> f64 {
        self.terms.iter().map(|t| t.coeff.magnitude()).fold(0.0, f64::max)
    }

    // ----------------------------------------------------------------------
    // arithmetic
    // ----------------------------------------------------------------------

    /// `self + scale · other` by merging the two sorted term lists.
    pub fn add_scaled(&self, other: &Polynomial<C>, scale: &C) -> Polynomial<C> {
        self.alg.check_same(&other.alg);
        if scale.is_zero() {
            return self.clone();
        }
        let scale_one = scale.is_one();
        let mut out: Vec<Term<C>> = Vec::with_capacity(self.terms.len() + other.terms.len());
        let (mut i, mut j) = (0, 0);
        let a = &self.terms;
        let b = &other.terms;
        while i < a.len() && j < b.len() {
            match self.alg.cmp_words(&a[i].word, &b[j].word) {
                Ordering::Greater => {
                    out.push(a[i].clone());
                    i += 1;
                }
                Ordering::Less => {
                    let c = if scale_one {
                        b[j].coeff.clone()
                    } else {
                        b[j].coeff.mul_ref(scale)
                    };
                    if !c.is_zero() {
                        out.push(Term {
                            word: b[j].word.clone(),
                            coeff: c,
                        });
                    }
                    j += 1;
                }
                Ordering::Equal => {
                    let bj = if scale_one {
                        b[j].coeff.clone()
                    } else {
                        b[j].coeff.mul_ref(scale)
                    };
                    let c = a[i].coeff.add_ref(&bj);
                    let gone = if C::INEXACT {
                        C::canceled(&c, a[i].coeff.magnitude() + bj.magnitude())
                    } else {
                        c.is_zero()
                    };
                    if !gone {
                        out.push(Term {
                            word: a[i].word.clone(),
                            coeff: c,
                        });
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&a[i..]);
        for t in &b[j..] {
            let c = if scale_one {
                t.coeff.clone()
            } else {
                t.coeff.mul_ref(scale)
            };
            if !c.is_zero() {
                out.push(Term {
                    word: t.word.clone(),
                    coeff: c,
                });
            }
        }
        Polynomial {
            alg: self.alg.clone(),
            terms: out,
        }
    }

    /// `self + other`.
    pub fn add_ref(&self, other: &Polynomial<C>) -> Polynomial<C> {
        self.add_scaled(other, &C::one())
    }

    /// `self - other`.
    pub fn sub_ref(&self, other: &Polynomial<C>) -> Polynomial<C> {
        self.add_scaled(other, &C::one().negated())
    }

    /// `-self`.
    pub fn negated(&self) -> Polynomial<C> {
        Polynomial {
            alg: self.alg.clone(),
            terms: self
                .terms
                .iter()
                .map(|t| Term {
                    word: t.word.clone(),
                    coeff: t.coeff.negated(),
                })
                .collect(),
        }
    }

    /// `c · self`.
    pub fn scale(&self, c: &C) -> Polynomial<C> {
        if c.is_zero() {
            return Polynomial::zero(self.alg.clone());
        }
        Polynomial {
            alg: self.alg.clone(),
            terms: self
                .terms
                .iter()
                .filter_map(|t| {
                    let coeff = t.coeff.mul_ref(c);
                    (!coeff.is_zero()).then(|| Term {
                        word: t.word.clone(),
                        coeff,
                    })
                })
                .collect(),
        }
    }

    /// Multiply every coefficient by `c` in place.
    pub fn scale_assign(&mut self, c: &C) {
        if c.is_zero() {
            self.terms.clear();
            return;
        }
        for t in &mut self.terms {
            t.coeff.mul_assign_ref(c);
        }
        self.terms.retain(|t| !t.coeff.is_zero());
    }

    /// `self · other` using a heap merge of the `|self|` sorted streams
    /// `t · other` (Monagan–Pearce), so the result is produced directly in
    /// sorted order without an intermediate unsorted expansion.
    pub fn mul_ref(&self, other: &Polynomial<C>) -> Polynomial<C> {
        self.alg.check_same(&other.alg);
        let mut acc = Accumulator::new(&self.alg);
        acc.push_product(self, other);
        acc.finish()
    }

    /// `w · self` for a word `w` (left multiplication preserves term order).
    pub fn mul_word_left(&self, w: &[Var]) -> Polynomial<C> {
        self.alg.check_word(w);
        Polynomial {
            alg: self.alg.clone(),
            terms: self
                .terms
                .iter()
                .map(|t| Term {
                    word: Word::concat3(w, &t.word, &[]),
                    coeff: t.coeff.clone(),
                })
                .collect(),
        }
    }

    /// `self · w` for a word `w`.
    pub fn mul_word_right(&self, w: &[Var]) -> Polynomial<C> {
        self.alg.check_word(w);
        Polynomial {
            alg: self.alg.clone(),
            terms: self
                .terms
                .iter()
                .map(|t| Term {
                    word: Word::concat3(&[], &t.word, w),
                    coeff: t.coeff.clone(),
                })
                .collect(),
        }
    }

    /// `l · self · r` for words `l`, `r`, optionally scaled by `c`.
    pub fn sandwich(&self, l: &[Var], r: &[Var]) -> Polynomial<C> {
        self.alg.check_word(l);
        self.alg.check_word(r);
        Polynomial {
            alg: self.alg.clone(),
            terms: self
                .terms
                .iter()
                .map(|t| Term {
                    word: Word::concat3(l, &t.word, r),
                    coeff: t.coeff.clone(),
                })
                .collect(),
        }
    }

    /// `self^n` by binary exponentiation.
    pub fn pow(&self, n: u32) -> Polynomial<C> {
        let mut result = self.alg.one::<C>();
        let mut base = self.clone();
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                result = result.mul_ref(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul_ref(&base);
            }
        }
        result
    }

    /// Commutator `[a, b] = a·b - b·a`, computed in a single heap merge.
    pub fn commutator(a: &Polynomial<C>, b: &Polynomial<C>) -> Polynomial<C> {
        a.alg.check_same(&b.alg);
        let mut acc = Accumulator::new(&a.alg);
        acc.push_product(a, b);
        acc.push_scaled_product(&C::one().negated(), b, a);
        acc.finish()
    }

    /// Anticommutator `{a, b} = a·b + b·a`.
    pub fn anticommutator(a: &Polynomial<C>, b: &Polynomial<C>) -> Polynomial<C> {
        a.alg.check_same(&b.alg);
        let mut acc = Accumulator::new(&a.alg);
        acc.push_product(a, b);
        acc.push_product(b, a);
        acc.finish()
    }

    // ----------------------------------------------------------------------
    // transformations
    // ----------------------------------------------------------------------

    /// Apply a function to every coefficient (terms mapping to zero are
    /// dropped).  Words are unchanged so the ordering is preserved.
    pub fn map_coeffs<D: Coeff, F: FnMut(&C) -> D>(&self, mut f: F) -> Polynomial<D> {
        Polynomial {
            alg: self.alg.clone(),
            terms: self
                .terms
                .iter()
                .filter_map(|t| {
                    let coeff = f(&t.coeff);
                    (!coeff.is_zero()).then(|| Term {
                        word: t.word.clone(),
                        coeff,
                    })
                })
                .collect(),
        }
    }

    /// Convert coefficients to another type (e.g. `i64 → BigRational`).
    pub fn convert<D: Coeff>(&self) -> Polynomial<D>
    where
        C: CoeffConvert<D>,
    {
        self.map_coeffs(|c| c.convert())
    }

    /// Apply a function to every word, producing a new polynomial.  Words may
    /// collide, so the result is re-normalized.
    pub fn map_words<F: FnMut(&Word) -> Word>(&self, mut f: F) -> Polynomial<C> {
        Polynomial::from_terms(
            self.alg.clone(),
            self.terms.iter().map(|t| (f(&t.word), t.coeff.clone())),
        )
    }

    /// Image under the anti-involution reversing every word (`x·y ↦ y·x`).
    pub fn reversed(&self) -> Polynomial<C> {
        self.map_words(|w| w.reversed())
    }

    /// Hermitian adjoint: reverse every word and conjugate every coefficient.
    pub fn adjoint(&self) -> Polynomial<C> {
        Polynomial::from_terms(
            self.alg.clone(),
            self.terms
                .iter()
                .map(|t| (self.alg.adjoint_word(&t.word), t.coeff.conj())),
        )
    }

    /// Is `self` equal to its [`adjoint`](Self::adjoint)?
    pub fn is_hermitian(&self) -> bool {
        *self == self.adjoint()
    }

    /// Terms of degree exactly `d`.
    pub fn homogeneous_part(&self, d: usize) -> Polynomial<C> {
        Polynomial {
            alg: self.alg.clone(),
            terms: self.terms.iter().filter(|t| t.word.len() == d).cloned().collect(),
        }
    }

    /// Decompose into `(degree, homogeneous component)` pairs, ascending.
    pub fn homogeneous_components(&self) -> Vec<(usize, Polynomial<C>)> {
        let mut degs: Vec<usize> = self.terms.iter().map(|t| t.word.len()).collect();
        degs.sort_unstable();
        degs.dedup();
        degs.into_iter().map(|d| (d, self.homogeneous_part(d))).collect()
    }

    /// Drop all terms of degree greater than `d`.
    pub fn truncate_degree(&self, d: usize) -> Polynomial<C> {
        Polynomial {
            alg: self.alg.clone(),
            terms: self.terms.iter().filter(|t| t.word.len() <= d).cloned().collect(),
        }
    }

    /// Substitute polynomials for generators: every occurrence of generator
    /// `v` in a word is replaced by `subst(v)` (if `Some`), and the resulting
    /// products are expanded.  Generators mapping to `None` are kept.
    pub fn substitute_with<F>(&self, subst: F) -> Polynomial<C>
    where
        F: FnMut(Var) -> Option<Polynomial<C>>,
    {
        let n = self.alg.num_vars();
        let images: Vec<Option<Polynomial<C>>> = (0..n as Var).map(subst).collect();
        for img in images.iter().flatten() {
            self.alg.check_same(&img.alg);
        }
        let mut acc = Accumulator::new(&self.alg);
        let mut pieces: Vec<Polynomial<C>> = Vec::new();
        for t in &self.terms {
            // product of segments: run through the word, multiplying out
            let mut cur = self.alg.constant(t.coeff.clone());
            let mut pending: Vec<Var> = Vec::new();
            for &v in t.word.as_slice() {
                match &images[v as usize] {
                    None => pending.push(v),
                    Some(img) => {
                        if !pending.is_empty() {
                            cur = cur.mul_word_right(&pending);
                            pending.clear();
                        }
                        cur = cur.mul_ref(img);
                    }
                }
            }
            if !pending.is_empty() {
                cur = cur.mul_word_right(&pending);
            }
            pieces.push(cur);
        }
        for p in &pieces {
            acc.push_poly(p);
        }
        acc.finish()
    }

    /// Substitute a single generator.
    pub fn substitute(&self, v: Var, image: &Polynomial<C>) -> Polynomial<C> {
        self.substitute_with(|u| (u == v).then(|| image.clone()))
    }

    /// The derivation `∂/∂x_v` of the free algebra: the unique derivation
    /// with `∂(x_v) = 1` and `∂(x_u) = 0` for `u ≠ v`.  On a word it deletes
    /// one occurrence of `x_v` at a time and sums the results.
    pub fn derivation(&self, v: Var) -> Polynomial<C> {
        let mut out: Vec<(Word, C)> = Vec::new();
        for t in &self.terms {
            for (i, &u) in t.word.as_slice().iter().enumerate() {
                if u == v {
                    let mut w = Vec::with_capacity(t.word.len() - 1);
                    w.extend_from_slice(&t.word[..i]);
                    w.extend_from_slice(&t.word[i + 1..]);
                    out.push((Word::from_vec(w), t.coeff.clone()));
                }
            }
        }
        Polynomial::from_terms(self.alg.clone(), out)
    }

    /// Approximate equality: all coefficient differences within `tol`.
    pub fn approx_eq(&self, other: &Polynomial<C>, tol: f64) -> bool {
        self.sub_ref(other).max_coeff_magnitude() <= tol
    }

    /// Dense coefficient vector with respect to an ordered list of words
    /// (e.g. the normal words of a quotient): entry `k` is the coefficient of
    /// `basis[k]`.  Returns `None` if some term's word is not in `basis`.
    pub fn coefficient_vector(&self, basis: &[Word]) -> Option<Vec<C>> {
        let mut out = vec![C::zero(); basis.len()];
        let mut covered = 0;
        for (k, w) in basis.iter().enumerate() {
            let c = self.coeff_of(w);
            if !c.is_zero() {
                covered += 1;
            }
            out[k] = c;
        }
        (covered == self.terms.len()).then_some(out)
    }

    /// Rename generators: every generator `v` becomes `f(v)` (a generator of
    /// `target`, which may be a different algebra).  Words may collide, so the
    /// result is re-normalized; the ordering is that of `target`.
    pub fn map_vars<F: FnMut(Var) -> Var>(&self, target: &FreeAlgebra, mut f: F) -> Polynomial<C> {
        Polynomial::from_terms(
            target.clone(),
            self.terms.iter().map(|t| {
                (
                    Word::from_vec(t.word.iter().map(|&v| f(v)).collect()),
                    t.coeff.clone(),
                )
            }),
        )
    }

    /// The words occurring, in decreasing order.
    pub fn words(&self) -> impl Iterator<Item = &Word> {
        self.terms.iter().map(|t| &t.word)
    }

    /// The coefficients, in decreasing order of their words.
    pub fn coeffs(&self) -> impl Iterator<Item = &C> {
        self.terms.iter().map(|t| &t.coeff)
    }

    /// If every word of `self` starts with `w`, return `q` with `self = w · q`.
    pub fn left_quotient_by_word(&self, w: &[Var]) -> Option<Polynomial<C>> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            if t.word.len() < w.len() || &t.word[..w.len()] != w {
                return None;
            }
            terms.push(Term {
                word: Word::from_slice(&t.word[w.len()..]),
                coeff: t.coeff.clone(),
            });
        }
        // left cancellation preserves the order of an admissible ordering
        Some(Polynomial::from_sorted_terms_unchecked(self.alg.clone(), terms))
    }

    /// If every word of `self` ends with `w`, return `q` with `self = q · w`.
    pub fn right_quotient_by_word(&self, w: &[Var]) -> Option<Polynomial<C>> {
        let mut terms = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            let n = t.word.len();
            if n < w.len() || &t.word[n - w.len()..] != w {
                return None;
            }
            terms.push(Term {
                word: Word::from_slice(&t.word[..n - w.len()]),
                coeff: t.coeff.clone(),
            });
        }
        Some(Polynomial::from_sorted_terms_unchecked(self.alg.clone(), terms))
    }

    /// LaTeX source for the polynomial, e.g. `x^{2} y - \frac{1}{2} y x + 1`.
    /// Generator names are emitted verbatim (name them `x_1` etc. for
    /// subscripts); rational coefficients become `\frac{a}{b}`, complex ones
    /// `(a + b i)`.
    pub fn to_latex(&self) -> String {
        fn latex_coeff(s: &str) -> String {
            // a/b -> \frac{a}{b}; complex with both parts -> (…)
            let s = s.trim();
            if let Some((a, b)) = s.split_once('/') {
                if !a.contains(['+', 'i']) && !b.contains(['+', 'i']) {
                    let (neg, a) = match a.strip_prefix('-') {
                        Some(r) => ("-", r),
                        None => ("", a),
                    };
                    return format!("{}\\frac{{{}}}{{{}}}", neg, a, b);
                }
            }
            // put a thin space between a numeral and the imaginary unit
            let spaced: String = {
                let mut out = String::with_capacity(s.len() + 2);
                let mut prev: Option<char> = None;
                for ch in s.chars() {
                    if ch == 'i' && prev.map(|p| p.is_ascii_digit() || p == '}').unwrap_or(false) {
                        out.push(' ');
                    }
                    out.push(ch);
                    prev = Some(ch);
                }
                out
            };
            let has_parts = s.contains('i') && (s[1..].contains('+') || s[1..].contains('-'));
            if has_parts {
                format!("({})", spaced)
            } else {
                spaced
            }
        }
        if self.terms.is_empty() {
            return "0".to_string();
        }
        let mut out = String::new();
        for (i, t) in self.terms.iter().enumerate() {
            let (neg, c) = if t.coeff.is_negative_for_display() {
                (true, t.coeff.negated())
            } else {
                (false, t.coeff.clone())
            };
            if i == 0 {
                if neg {
                    out.push('-');
                }
            } else {
                out.push_str(if neg { " - " } else { " + " });
            }
            struct Disp<'a, C: Coeff>(&'a C);
            impl<'a, C: Coeff> fmt::Display for Disp<'a, C> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.0.fmt_display(f)
                }
            }
            let cs = latex_coeff(&Disp(&c).to_string());
            let word = crate::word::run_lengths(&t.word)
                .into_iter()
                .map(|(v, n)| {
                    if n == 1 {
                        self.alg.name(v).to_string()
                    } else {
                        format!("{}^{{{}}}", self.alg.name(v), n)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            if t.word.is_empty() {
                out.push_str(&cs);
            } else if c.is_one() {
                out.push_str(&word);
            } else {
                out.push_str(&cs);
                out.push(' ');
                out.push_str(&word);
            }
        }
        out
    }

    /// Evaluate in an arbitrary (noncommutative) ring `R`, e.g. matrices:
    /// generator `v` is replaced by `image(v)`, a coefficient `c` by
    /// `embed(c)`, and a term `c · x_{i_1} ⋯ x_{i_k}` becomes
    /// `embed(c) · image(i_1) ⋯ image(i_k)` (the empty word becomes `one`).
    /// Terms are summed starting from `zero`.
    pub fn evaluate<R, F, S>(&self, mut image: F, mut embed: S, zero: R, one: R) -> R
    where
        R: Clone + std::ops::Add<Output = R> + std::ops::Mul<Output = R>,
        F: FnMut(Var) -> R,
        S: FnMut(&C) -> R,
    {
        let mut acc = zero;
        for t in &self.terms {
            let mut m = embed(&t.coeff);
            if t.word.is_empty() {
                m = m * one.clone();
            } else {
                for &v in t.word.as_slice() {
                    m = m * image(v);
                }
            }
            acc = acc + m;
        }
        acc
    }
}

impl<C: Field> Polynomial<C> {
    /// The Hermitian and anti-Hermitian parts `((p + p*)/2, (p − p*)/2)`, so
    /// that `p = H + A`, `H* = H`, `A* = −A` (as polynomials of the free
    /// algebra; reduce them for normal forms).  For a Hermitian functional
    /// `L`, `L(p) = v` is equivalent to `L(H) = Re v` and `L(−iA) = Im v`.
    pub fn hermitian_split(&self) -> (Polynomial<C>, Polynomial<C>) {
        let ps = self.adjoint();
        let half = C::from_i64(2).inverse();
        (self.add_ref(&ps).scale(&half), self.sub_ref(&ps).scale(&half))
    }

    /// Divide by the leading coefficient.  Zero stays zero.
    pub fn monic(&self) -> Polynomial<C> {
        match self.leading_coeff() {
            None => self.clone(),
            Some(lc) if lc.is_one() => self.clone(),
            Some(lc) => self.scale(&lc.inverse()),
        }
    }

    /// Divide by the leading coefficient in place.
    pub fn make_monic(&mut self) {
        if let Some(lc) = self.leading_coeff().cloned() {
            if !lc.is_one() {
                let inv = lc.inverse();
                for t in &mut self.terms {
                    t.coeff.mul_assign_ref(&inv);
                }
                if let Some(t) = self.terms.first_mut() {
                    t.coeff = C::one();
                }
            }
        }
    }

    /// `self / c`.
    pub fn div_scalar(&self, c: &C) -> Polynomial<C> {
        self.scale(&c.inverse())
    }
}

impl<C: IntegerCoeff> Polynomial<C> {
    /// GCD of all coefficients (zero for the zero polynomial).
    pub fn content(&self) -> C {
        let mut g = C::zero();
        for t in &self.terms {
            g = g.gcd_ref(&t.coeff);
        }
        g.abs_ref()
    }

    /// Divide out the content; the leading coefficient becomes positive.
    pub fn primitive_part(&self) -> Polynomial<C> {
        let g = self.content();
        if g.is_zero() {
            return self.clone();
        }
        let g = if self
            .leading_coeff()
            .map(|c| c.is_negative_for_display())
            .unwrap_or(false)
        {
            g.negated()
        } else {
            g
        };
        self.map_coeffs(|c| c.exact_div(&g))
    }
}

impl Polynomial<BigRational> {
    /// Clear denominators and divide out the content, producing a primitive
    /// integer polynomial that generates the same ideal over `Q`.
    pub fn to_integer_primitive(&self) -> Polynomial<BigInt> {
        if self.is_zero() {
            return Polynomial::zero(self.alg.clone());
        }
        let mut l: BigInt = One::one();
        for t in &self.terms {
            l = l.lcm(t.coeff.denom());
        }
        let ints: Polynomial<BigInt> = self.map_coeffs(|c| (c * &l).to_integer());
        ints.primitive_part()
    }

    /// Convert to floating point coefficients.
    pub fn to_f64(&self) -> Polynomial<f64> {
        self.convert()
    }
}

impl Polynomial<i64> {
    /// Promote to exact rational coefficients (needed for Gröbner bases).
    pub fn to_rational(&self) -> Polynomial<BigRational> {
        self.convert()
    }
    /// Convert to floating-point coefficients.
    pub fn to_f64(&self) -> Polynomial<f64> {
        self.convert()
    }
}

impl Polynomial<i128> {
    /// Promote to exact rational coefficients (needed for Gröbner bases).
    pub fn to_rational(&self) -> Polynomial<BigRational> {
        self.convert()
    }
    /// Convert to floating-point coefficients.
    pub fn to_f64(&self) -> Polynomial<f64> {
        self.convert()
    }
}

impl Polynomial<BigInt> {
    /// Promote to exact rational coefficients (needed for Gröbner bases).
    pub fn to_rational(&self) -> Polynomial<BigRational> {
        self.convert()
    }
    /// Convert to floating-point coefficients.
    pub fn to_f64(&self) -> Polynomial<f64> {
        self.convert()
    }
}

impl Polynomial<Complex<BigRational>> {
    /// Multiply by the lcm of all denominators so coefficients become Gaussian
    /// integers; the result generates the same ideal.
    pub fn clear_denominators(&self) -> Polynomial<Complex<BigRational>> {
        let mut l: BigInt = One::one();
        for t in &self.terms {
            l = l.lcm(t.coeff.re.denom()).lcm(t.coeff.im.denom());
        }
        let l = BigRational::from_integer(l);
        self.map_coeffs(|c| Complex::new(&c.re * &l, &c.im * &l))
    }

    /// Convert to floating-point complex coefficients.
    pub fn to_complex_f64(&self) -> Polynomial<Complex<f64>> {
        self.convert()
    }
}

// --------------------------------------------------------------------------
// equality / formatting
// --------------------------------------------------------------------------

impl<C: Coeff> PartialEq for Polynomial<C> {
    fn eq(&self, other: &Self) -> bool {
        self.alg.same_as(&other.alg) && self.terms == other.terms
    }
}

impl<C: Coeff + Eq> Eq for Polynomial<C> {}

impl<C: Coeff + std::hash::Hash> std::hash::Hash for Polynomial<C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for t in &self.terms {
            t.word.hash(state);
            t.coeff.hash(state);
        }
    }
}

impl<C: Coeff> fmt::Display for Polynomial<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (i, t) in self.terms.iter().enumerate() {
            let (neg, c) = if t.coeff.is_negative_for_display() {
                (true, t.coeff.negated())
            } else {
                (false, t.coeff.clone())
            };
            if i == 0 {
                if neg {
                    write!(f, "-")?;
                }
            } else if neg {
                write!(f, " - ")?;
            } else {
                write!(f, " + ")?;
            }
            if t.word.is_empty() {
                c.fmt_display(f)?;
            } else if c.is_one() {
                write!(f, "{}", self.alg.format_word(&t.word))?;
            } else {
                fmt_coeff(&c, f)?;
                write!(f, "*{}", self.alg.format_word(&t.word))?;
            }
        }
        Ok(())
    }
}

impl<C: Coeff> fmt::Debug for Polynomial<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

// --------------------------------------------------------------------------
// operator overloading
// --------------------------------------------------------------------------

macro_rules! impl_binop {
    ($Trait:ident, $method:ident, $inner:ident) => {
        impl<C: Coeff> $Trait<Polynomial<C>> for Polynomial<C> {
            type Output = Polynomial<C>;
            fn $method(self, o: Polynomial<C>) -> Polynomial<C> {
                (&self).$inner(&o)
            }
        }
        impl<'a, C: Coeff> $Trait<&'a Polynomial<C>> for Polynomial<C> {
            type Output = Polynomial<C>;
            fn $method(self, o: &'a Polynomial<C>) -> Polynomial<C> {
                (&self).$inner(o)
            }
        }
        impl<'a, C: Coeff> $Trait<Polynomial<C>> for &'a Polynomial<C> {
            type Output = Polynomial<C>;
            fn $method(self, o: Polynomial<C>) -> Polynomial<C> {
                self.$inner(&o)
            }
        }
        impl<'a, 'b, C: Coeff> $Trait<&'b Polynomial<C>> for &'a Polynomial<C> {
            type Output = Polynomial<C>;
            fn $method(self, o: &'b Polynomial<C>) -> Polynomial<C> {
                self.$inner(o)
            }
        }
    };
}
impl_binop!(Add, add, add_ref);
impl_binop!(Sub, sub, sub_ref);
impl_binop!(Mul, mul, mul_ref);

impl<C: Coeff> Neg for Polynomial<C> {
    type Output = Polynomial<C>;
    fn neg(self) -> Polynomial<C> {
        self.negated()
    }
}
impl<C: Coeff> Neg for &Polynomial<C> {
    type Output = Polynomial<C>;
    fn neg(self) -> Polynomial<C> {
        self.negated()
    }
}

impl<C: Coeff> Polynomial<C> {
    /// `self += scale · other` in place.  A short `other` (at most
    /// [`SHORT_SUMMAND`] terms) is inserted term by term at its binary-search
    /// position, which moves the following terms instead of cloning every
    /// word of `self` (building a 32 000-term polynomial with `+=` took
    /// 12.7 s).  Still quadratic in the worst case: collect the terms
    /// and use [`from_terms`](Self::from_terms) or [`sum`](Self::sum) for
    /// large sums.
    fn add_scaled_in_place(&mut self, other: &Polynomial<C>, scale: &C) {
        if other.terms.len() > SHORT_SUMMAND {
            *self = self.add_scaled(other, scale);
            return;
        }
        self.alg.check_same(&other.alg);
        if scale.is_zero() {
            return;
        }
        for t in &other.terms {
            let c = if scale.is_one() {
                t.coeff.clone()
            } else {
                t.coeff.mul_ref(scale)
            };
            if c.is_zero() {
                continue;
            }
            // the terms are sorted strictly decreasing
            match self
                .terms
                .binary_search_by(|x| self.alg.cmp_words(&t.word, &x.word))
            {
                Ok(i) => {
                    let sum = self.terms[i].coeff.add_ref(&c);
                    let gone = if C::INEXACT {
                        C::canceled(&sum, self.terms[i].coeff.magnitude() + c.magnitude())
                    } else {
                        sum.is_zero()
                    };
                    if gone {
                        self.terms.remove(i);
                    } else {
                        self.terms[i].coeff = sum;
                    }
                }
                Err(i) => self.terms.insert(
                    i,
                    Term {
                        word: t.word.clone(),
                        coeff: c,
                    },
                ),
            }
        }
    }
}

/// Rounding level of a floating-point reduction relative to the size of its
/// input (the default of [`crate::GroebnerOptions::float_residue`]).
pub(crate) const FLOAT_RESIDUE: f64 = 1e-10;

impl<C: Coeff> Polynomial<C> {
    /// Largest coefficient magnitude (0 for zero).
    pub(crate) fn max_magnitude(&self) -> f64 {
        self.terms
            .iter()
            .map(|t| t.coeff.magnitude())
            .fold(0.0_f64, f64::max)
    }

    /// Floating-point coefficients only: is this nonzero polynomial, the
    /// result of a reduction whose input had largest coefficient `scale`,
    /// rounding residue, every coefficient at most `rel · scale`?  Always
    /// `false` for exact coefficients and inside
    /// [`crate::coeff::with_exact_zero`].  See
    /// [`crate::GroebnerOptions::float_residue`].
    pub(crate) fn is_rounding_residue(&self, scale: f64, rel: f64) -> bool {
        C::INEXACT
            && rel > 0.0
            && !crate::coeff::exact_zero_active()
            && !self.is_zero()
            && self.max_magnitude() <= rel * scale
    }

    /// Zero, or (floating-point coefficients) rounding residue relative to
    /// `scale` at the default level [`FLOAT_RESIDUE`]: the test for "reduces
    /// to zero" of a membership or invariance check.
    pub(crate) fn is_zero_up_to_rounding(&self, scale: f64) -> bool {
        self.is_zero() || self.is_rounding_residue(scale, FLOAT_RESIDUE)
    }
}

/// Summands of `+=` / `-=` up to this many terms are inserted in place.
const SHORT_SUMMAND: usize = 8;

impl<C: Coeff> AddAssign<Polynomial<C>> for Polynomial<C> {
    fn add_assign(&mut self, o: Polynomial<C>) {
        self.add_scaled_in_place(&o, &C::one());
    }
}
impl<'a, C: Coeff> AddAssign<&'a Polynomial<C>> for Polynomial<C> {
    fn add_assign(&mut self, o: &'a Polynomial<C>) {
        self.add_scaled_in_place(o, &C::one());
    }
}
impl<C: Coeff> SubAssign<Polynomial<C>> for Polynomial<C> {
    fn sub_assign(&mut self, o: Polynomial<C>) {
        self.add_scaled_in_place(&o, &C::one().negated());
    }
}
impl<'a, C: Coeff> SubAssign<&'a Polynomial<C>> for Polynomial<C> {
    fn sub_assign(&mut self, o: &'a Polynomial<C>) {
        self.add_scaled_in_place(o, &C::one().negated());
    }
}
impl<C: Coeff> MulAssign<Polynomial<C>> for Polynomial<C> {
    fn mul_assign(&mut self, o: Polynomial<C>) {
        *self = self.mul_ref(&o);
    }
}
impl<'a, C: Coeff> MulAssign<&'a Polynomial<C>> for Polynomial<C> {
    fn mul_assign(&mut self, o: &'a Polynomial<C>) {
        *self = self.mul_ref(o);
    }
}

// polynomial (op) scalar
impl<C: Coeff> Mul<C> for Polynomial<C> {
    type Output = Polynomial<C>;
    fn mul(self, c: C) -> Polynomial<C> {
        self.scale(&c)
    }
}
impl<C: Coeff> Mul<C> for &Polynomial<C> {
    type Output = Polynomial<C>;
    fn mul(self, c: C) -> Polynomial<C> {
        self.scale(&c)
    }
}
impl<'a, C: Coeff> Mul<&'a C> for Polynomial<C> {
    type Output = Polynomial<C>;
    fn mul(self, c: &'a C) -> Polynomial<C> {
        self.scale(c)
    }
}
impl<'b, C: Coeff> Mul<&'b C> for &Polynomial<C> {
    type Output = Polynomial<C>;
    fn mul(self, c: &'b C) -> Polynomial<C> {
        self.scale(c)
    }
}
impl<C: Coeff> MulAssign<C> for Polynomial<C> {
    fn mul_assign(&mut self, c: C) {
        self.scale_assign(&c);
    }
}
impl<C: Coeff> Add<C> for Polynomial<C> {
    type Output = Polynomial<C>;
    fn add(self, c: C) -> Polynomial<C> {
        let k = self.alg.constant(c);
        self.add_ref(&k)
    }
}
impl<C: Coeff> Add<C> for &Polynomial<C> {
    type Output = Polynomial<C>;
    fn add(self, c: C) -> Polynomial<C> {
        let k = self.alg.constant(c);
        self.add_ref(&k)
    }
}
impl<C: Coeff> Sub<C> for Polynomial<C> {
    type Output = Polynomial<C>;
    fn sub(self, c: C) -> Polynomial<C> {
        let k = self.alg.constant(c);
        self.sub_ref(&k)
    }
}
impl<C: Coeff> Sub<C> for &Polynomial<C> {
    type Output = Polynomial<C>;
    fn sub(self, c: C) -> Polynomial<C> {
        let k = self.alg.constant(c);
        self.sub_ref(&k)
    }
}

// scalar (op) polynomial, for the concrete coefficient types (orphan rules
// forbid a blanket impl)
macro_rules! impl_scalar_lhs {
    ($($t:ty),*) => {$(
        impl Mul<Polynomial<$t>> for $t {
            type Output = Polynomial<$t>;
            fn mul(self, p: Polynomial<$t>) -> Polynomial<$t> { p.scale(&self) }
        }
        impl<'a> Mul<&'a Polynomial<$t>> for $t {
            type Output = Polynomial<$t>;
            fn mul(self, p: &'a Polynomial<$t>) -> Polynomial<$t> { p.scale(&self) }
        }
        impl Add<Polynomial<$t>> for $t {
            type Output = Polynomial<$t>;
            fn add(self, p: Polynomial<$t>) -> Polynomial<$t> { p + self }
        }
        impl<'a> Add<&'a Polynomial<$t>> for $t {
            type Output = Polynomial<$t>;
            fn add(self, p: &'a Polynomial<$t>) -> Polynomial<$t> { p + self }
        }
        impl Sub<Polynomial<$t>> for $t {
            type Output = Polynomial<$t>;
            fn sub(self, p: Polynomial<$t>) -> Polynomial<$t> { p.negated() + self }
        }
        impl<'a> Sub<&'a Polynomial<$t>> for $t {
            type Output = Polynomial<$t>;
            fn sub(self, p: &'a Polynomial<$t>) -> Polynomial<$t> { p.negated() + self }
        }
    )*};
}
impl_scalar_lhs!(
    i64,
    i128,
    f64,
    BigInt,
    BigRational,
    Complex<f64>,
    Complex<BigRational>
);

impl<const P: u64, const K: u32> Mul<Polynomial<crate::ff::Fq<P, K>>> for crate::ff::Fq<P, K> {
    type Output = Polynomial<crate::ff::Fq<P, K>>;
    fn mul(self, p: Polynomial<crate::ff::Fq<P, K>>) -> Self::Output {
        p.scale(&self)
    }
}
impl<'a, const P: u64, const K: u32> Mul<&'a Polynomial<crate::ff::Fq<P, K>>> for crate::ff::Fq<P, K> {
    type Output = Polynomial<crate::ff::Fq<P, K>>;
    fn mul(self, p: &'a Polynomial<crate::ff::Fq<P, K>>) -> Self::Output {
        p.scale(&self)
    }
}
