//! Lazy, heap-driven expansion and reduction.
//!
//! # The idea
//!
//! Every polynomial operation this library performs can be written as a sum
//! of *scaled sandwiches* `c · a · p · b` where `c` is a coefficient, `a`, `b`
//! are words and `p` is an already-sorted polynomial.  Because the monomial
//! ordering is admissible, the terms of `a · p · b` appear in the same order
//! as the terms of `p`, so each sandwich is a *sorted stream* of terms that
//! can be generated one term at a time.
//!
//! An [`Accumulator`] holds a max-heap of such streams keyed by their current
//! word.  Pulling from it yields the terms of the whole sum in strictly
//! decreasing order, combining equal words as they meet at the top of the
//! heap.  Multiplication `f · g` is the sum of the `|f|` streams `f_i · g`
//! (Monagan–Pearce heap multiplication); a commutator is `f·g` and `-g·f`
//! pushed into the same heap.
//!
//! A [`Reducer`] drives an accumulator while watching the terms that emerge:
//! when the current top term `c·w` has a divisor `w = a · lw(g) · b` for a
//! basis element `g`, the reducer pushes the stream `-c/lc(g) · a · tail(g) · b`
//! back into the heap (the leading terms cancel exactly, so they are never
//! materialized) and continues.  Terms with no divisor are appended to the
//! remainder, which therefore comes out already sorted.  This is how
//! "reduction while expanding" works: the product `f·g` modulo a Gröbner
//! basis is computed without ever forming the unreduced product, and the
//! same machinery evaluates arbitrary expression trees ([`crate::expr`]).

use crate::algebra::FreeAlgebra;
use crate::automaton::AhoCorasick;
use crate::coeff::{Coeff, Field};
use crate::poly::{Polynomial, Term};
use crate::word::{Var, Word};
use std::cmp::Ordering;

/// One sorted stream of terms `coeff · left · terms[idx..] · right`.
struct Stream<'a, C: Coeff> {
    coeff: C,
    left: Vec<Var>,
    terms: &'a [Term<C>],
    right: Vec<Var>,
    idx: usize,
    /// `left · terms[idx].word · right`, cached.
    cur: Word,
}

impl<'a, C: Coeff> Stream<'a, C> {
    fn new(coeff: C, left: &[Var], terms: &'a [Term<C>], right: &[Var]) -> Option<Self> {
        if terms.is_empty() || coeff.is_zero() {
            return None;
        }
        let cur = Word::concat3(left, &terms[0].word, right);
        Some(Stream {
            coeff,
            left: left.to_vec(),
            terms,
            right: right.to_vec(),
            idx: 0,
            cur,
        })
    }

    #[inline]
    fn current_coeff(&self) -> C {
        if self.coeff.is_one() {
            self.terms[self.idx].coeff.clone()
        } else {
            self.terms[self.idx].coeff.mul_ref(&self.coeff)
        }
    }

    /// Advance to the next term; returns false when exhausted.
    #[inline]
    fn advance(&mut self) -> bool {
        self.idx += 1;
        if self.idx >= self.terms.len() {
            return false;
        }
        self.cur = Word::concat3(&self.left, &self.terms[self.idx].word, &self.right);
        true
    }
}

/// Max-heap of streams ordered by current word under the algebra's ordering.
struct StreamHeap<'a, C: Coeff> {
    alg: FreeAlgebra,
    items: Vec<Stream<'a, C>>,
}

impl<'a, C: Coeff> StreamHeap<'a, C> {
    fn new(alg: FreeAlgebra) -> Self {
        StreamHeap {
            alg,
            items: Vec::new(),
        }
    }

    #[inline]
    fn greater(&self, i: usize, j: usize) -> bool {
        self.alg.cmp_words(&self.items[i].cur, &self.items[j].cur) == Ordering::Greater
    }

    fn push(&mut self, s: Stream<'a, C>) {
        self.items.push(s);
        let mut i = self.items.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.greater(i, parent) {
                self.items.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn peek_word(&self) -> Option<&Word> {
        self.items.first().map(|s| &s.cur)
    }

    fn pop(&mut self) -> Option<Stream<'a, C>> {
        if self.items.is_empty() {
            return None;
        }
        let last = self.items.len() - 1;
        self.items.swap(0, last);
        let top = self.items.pop();
        self.sift_down(0);
        top
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.items.len();
        loop {
            let l = 2 * i + 1;
            let r = l + 1;
            let mut largest = i;
            if l < n && self.greater(l, largest) {
                largest = l;
            }
            if r < n && self.greater(r, largest) {
                largest = r;
            }
            if largest == i {
                break;
            }
            self.items.swap(i, largest);
            i = largest;
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// A lazily evaluated sum of scaled sandwiches `c · a · p · b`, whose terms
/// can be pulled out in strictly decreasing monomial order.
///
/// See the [module documentation](self) for the design.
pub struct Accumulator<'a, C: Coeff> {
    heap: StreamHeap<'a, C>,
    /// Streams pushed so far (for statistics).
    pushed: usize,
}

impl<'a, C: Coeff> std::fmt::Debug for Accumulator<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Accumulator({} live streams)", self.heap.len())
    }
}

impl<'a, C: Coeff> Accumulator<'a, C> {
    /// An empty accumulator for the given algebra.
    pub fn new(alg: &FreeAlgebra) -> Self {
        Accumulator {
            heap: StreamHeap::new(alg.clone()),
            pushed: 0,
        }
    }

    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.heap.alg
    }

    /// Add `coeff · left · p · right` where `p` is given by its sorted terms.
    pub fn push_scaled_terms(&mut self, coeff: C, left: &[Var], terms: &'a [Term<C>], right: &[Var]) {
        if let Some(s) = Stream::new(coeff, left, terms, right) {
            self.heap.push(s);
            self.pushed += 1;
        }
    }

    /// Add `coeff · left · p · right`.
    pub fn push_scaled(&mut self, coeff: C, left: &[Var], p: &'a Polynomial<C>, right: &[Var]) {
        self.heap.alg.check_same(p.algebra());
        self.push_scaled_terms(coeff, left, p.terms(), right);
    }

    /// Add `p`.
    pub fn push_poly(&mut self, p: &'a Polynomial<C>) {
        self.push_scaled(C::one(), &[], p, &[]);
    }

    /// Add `c · p`.
    pub fn push_scaled_poly(&mut self, c: &C, p: &'a Polynomial<C>) {
        self.push_scaled(c.clone(), &[], p, &[]);
    }

    /// Add `left · p · right`.
    pub fn push_sandwich(&mut self, left: &[Var], p: &'a Polynomial<C>, right: &[Var]) {
        self.push_scaled(C::one(), left, p, right);
    }

    /// Add the product `f · g` as `|f|` streams `f_i · g`.
    pub fn push_product(&mut self, f: &'a Polynomial<C>, g: &'a Polynomial<C>) {
        self.push_scaled_product(&C::one(), f, g);
    }

    /// Add `c · f · g`.
    pub fn push_scaled_product(&mut self, c: &C, f: &'a Polynomial<C>, g: &'a Polynomial<C>) {
        self.heap.alg.check_same(f.algebra());
        self.heap.alg.check_same(g.algebra());
        if c.is_zero() {
            return;
        }
        for t in f.terms() {
            let coeff = if c.is_one() {
                t.coeff.clone()
            } else {
                t.coeff.mul_ref(c)
            };
            self.push_scaled_terms(coeff, &t.word, g.terms(), &[]);
        }
    }

    /// Add `c · left · f · g · right`.
    pub fn push_scaled_product_sandwich(
        &mut self,
        c: &C,
        left: &[Var],
        f: &'a Polynomial<C>,
        g: &'a Polynomial<C>,
        right: &[Var],
    ) {
        self.heap.alg.check_same(f.algebra());
        self.heap.alg.check_same(g.algebra());
        if c.is_zero() {
            return;
        }
        for t in f.terms() {
            let coeff = if c.is_one() {
                t.coeff.clone()
            } else {
                t.coeff.mul_ref(c)
            };
            let l = Word::concat3(left, &t.word, &[]);
            self.push_scaled_terms(coeff, &l, g.terms(), right);
        }
    }

    /// Number of streams currently alive.
    pub fn live_streams(&self) -> usize {
        self.heap.len()
    }

    /// Total number of streams pushed so far.
    pub fn streams_pushed(&self) -> usize {
        self.pushed
    }

    /// Word of the next term to be produced, if any.
    pub fn peek_word(&self) -> Option<&Word> {
        self.heap.peek_word()
    }

    /// Pull the next (largest) term, combining all streams that currently sit
    /// on the same word.  Terms whose combined coefficient is zero are skipped.
    pub fn next_term(&mut self) -> Option<Term<C>> {
        loop {
            let mut top = self.heap.pop()?;
            let word = std::mem::take(&mut top.cur);
            let mut coeff = top.current_coeff();
            // size of the summands, for the cancellation test of inexact
            // coefficients
            let mut abs = if C::INEXACT { coeff.magnitude() } else { 0.0 };
            let mut merged = false;
            // the stream's cur was taken; restore it by advancing (recomputes cur)
            if top.advance() {
                self.heap.push(top);
            }
            while let Some(w) = self.heap.peek_word() {
                if *w != word {
                    break;
                }
                let mut s = self.heap.pop().expect("peeked");
                let c = s.current_coeff();
                if C::INEXACT {
                    abs += c.magnitude();
                }
                coeff.add_assign_ref(&c);
                merged = true;
                if s.advance() {
                    self.heap.push(s);
                }
            }
            let gone = if merged {
                C::canceled(&coeff, abs)
            } else {
                coeff.is_zero()
            };
            if !gone {
                return Some(Term { word, coeff });
            }
        }
    }

    /// Drain everything into a polynomial.
    pub fn finish(mut self) -> Polynomial<C> {
        let mut out = Vec::new();
        while let Some(t) = self.next_term() {
            out.push(t);
        }
        Polynomial::from_sorted_terms_unchecked(self.heap.alg.clone(), out)
    }
}

/// How far to reduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionMode {
    /// Reduce every term: the result is in normal form (no term divisible by
    /// any leading word).
    Full,
    /// Reduce only until the leading term is irreducible; remaining terms are
    /// merged but not reduced.
    Top,
}

/// Statistics from a reduction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReductionStats {
    /// Number of reduction steps (divisor substitutions).
    pub steps: usize,
    /// Number of terms produced in the remainder.
    pub remainder_terms: usize,
    /// Maximum number of streams simultaneously alive in the heap.
    pub max_heap: usize,
}

/// Reduction modulo a fixed set of nonzero polynomials over a field, with
/// leading-word lookup by Aho–Corasick automaton.
///
/// The set need not be a Gröbner basis; in that case the result depends on
/// the choice of divisor (this implementation always uses the divisor whose
/// occurrence ends earliest in the word).  For a Gröbner basis the result is
/// the unique normal form.
#[derive(Clone, Debug)]
pub struct Reducer<C: Field> {
    alg: FreeAlgebra,
    basis: Vec<Polynomial<C>>,
    lc_inv: Vec<C>,
    automaton: AhoCorasick,
    /// Some basis element is a nonzero constant: everything reduces to zero.
    has_unit: bool,
    /// The basis is a complete Gröbner basis, so normal forms are unique.
    complete: bool,
    /// Incremental leading-word matching (`GroebnerSolver`); when set,
    /// `automaton` is not maintained and retired elements stay in `basis`.
    dynamic: Option<crate::automaton::DynamicMatcher>,
}

impl<C: Field> Reducer<C> {
    /// Build a reducer for the given polynomials (zero polynomials are
    /// ignored).  Panics if the polynomials belong to different algebras.
    pub fn new(alg: &FreeAlgebra, basis: impl IntoIterator<Item = Polynomial<C>>) -> Self {
        let basis: Vec<Polynomial<C>> = basis.into_iter().filter(|p| !p.is_zero()).collect();
        for p in &basis {
            alg.check_same(p.algebra());
        }
        let lc_inv = basis
            .iter()
            .map(|p| p.leading_coeff().expect("nonzero").inverse())
            .collect();
        let patterns: Vec<&[Var]> = basis
            .iter()
            .map(|p| p.leading_word().expect("nonzero").as_slice())
            .collect();
        let automaton = AhoCorasick::new(alg.num_vars(), &patterns);
        let has_unit = automaton.has_empty_pattern();
        Reducer {
            alg: alg.clone(),
            basis,
            lc_inv,
            automaton,
            has_unit,
            complete: false,
            dynamic: None,
        }
    }

    /// An empty reducer whose basis grows by [`push`](Self::push) and shrinks
    /// by [`retire`](Self::retire), with incremental leading-word matching
    /// (the Gröbner solver's working reducer).  Its
    /// [`automaton`](Self::automaton) is not available.
    pub fn new_dynamic(alg: &FreeAlgebra) -> Self {
        let mut r = Reducer::new(alg, Vec::new());
        r.dynamic = Some(crate::automaton::DynamicMatcher::new(alg.num_vars()));
        r
    }

    /// Add a polynomial to a dynamic reducer; returns its basis index.
    pub fn push(&mut self, p: Polynomial<C>) -> usize {
        let lw = p.leading_word().expect("nonzero").clone();
        // a second live element with the same leading word: the matcher keeps
        // one index per pattern, and retiring that one would leave the other
        // unmatchable.  The Gröbner solver never does this.
        assert!(
            !self
                .occurrences(lw.as_slice())
                .iter()
                .any(|&(_, s, l)| s == 0 && l == lw.len()),
            "push: a live basis element already has this leading word"
        );
        let dynamic = self.dynamic.as_mut().expect("push needs a dynamic reducer");
        let id = dynamic.insert(lw.as_slice());
        debug_assert_eq!(id, self.basis.len());
        self.lc_inv.push(p.leading_coeff().expect("nonzero").inverse());
        if lw.is_empty() {
            self.has_unit = true;
        }
        self.basis.push(p);
        id
    }

    /// Every occurrence `(basis index, start, length)` of a (live) leading
    /// word in `w`, in no particular order.
    pub fn occurrences(&self, w: &[Var]) -> Vec<(usize, usize, usize)> {
        match &self.dynamic {
            Some(d) => d
                .find_all(w)
                .into_iter()
                .map(|(id, s)| (id, s, d.pattern_len(id)))
                .collect(),
            None => self
                .automaton
                .find_all(w)
                .into_iter()
                .map(|(p, s)| (p, s, self.automaton.pattern_len(p)))
                .collect(),
        }
    }

    /// Stop reducing by basis element `idx` of a dynamic reducer.
    pub fn retire(&mut self, idx: usize) {
        self.dynamic
            .as_mut()
            .expect("retire needs a dynamic reducer")
            .retire(idx);
    }

    /// Mark the basis as a complete Gröbner basis (normal forms then are
    /// unique; [`GroebnerBasis::reducer`](crate::GroebnerBasis) does this).
    pub fn with_complete(mut self, complete: bool) -> Self {
        self.complete = complete;
        self
    }

    /// Is the basis known to be a complete Gröbner basis (normal forms unique)?
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// The reducing polynomials.
    pub fn basis(&self) -> &[Polynomial<C>] {
        &self.basis
    }

    /// The leading-word automaton (not for dynamic reducers).
    pub fn automaton(&self) -> &AhoCorasick {
        assert!(
            self.dynamic.is_none(),
            "a dynamic reducer has no single automaton"
        );
        &self.automaton
    }

    /// Does the basis contain a nonzero constant (so the ideal is everything)?
    pub fn is_unit_ideal(&self) -> bool {
        self.has_unit
    }

    /// Is `w` a normal word (no leading word occurs as a subword)?
    #[inline]
    pub fn is_normal_word(&self, w: &[Var]) -> bool {
        match &self.dynamic {
            Some(d) => d.find_first(w).is_none(),
            None => self.automaton.is_normal(w),
        }
    }

    /// Find a basis element whose leading word occurs in `w`:
    /// `(basis index, start position)`.
    #[inline]
    pub fn find_divisor(&self, w: &[Var]) -> Option<(usize, usize)> {
        match &self.dynamic {
            Some(d) => d.find_first(w),
            None => self.automaton.find_first(w),
        }
    }

    /// Is `p` in normal form?
    pub fn is_reduced(&self, p: &Polynomial<C>) -> bool {
        p.terms().iter().all(|t| self.is_normal_word(&t.word))
    }

    /// Normal words of length exactly `d` (words containing no leading word
    /// of the basis), in increasing monomial order.  Enumerated by a depth
    /// first search over the leading-word automaton.
    /// Not for dynamic reducers (they enumerated every word).
    pub fn normal_words(&self, d: usize) -> Vec<Word> {
        let mut out: Vec<Word> = self
            .automaton()
            .avoiding_words(self.alg.num_vars(), d)
            .into_iter()
            .map(Word::from_vec)
            .collect();
        out.sort_by(|a, b| self.alg.cmp_words(a, b));
        out
    }

    /// Normal words of all lengths `0..=d`.
    pub fn normal_words_up_to(&self, d: usize) -> Vec<Word> {
        (0..=d).flat_map(|k| self.normal_words(k)).collect()
    }

    /// Drive an accumulator to completion, reducing the terms that emerge.
    /// The accumulator may borrow from `self` (its basis) and from the inputs.
    pub fn run<'a>(
        &'a self,
        mut acc: Accumulator<'a, C>,
        mode: ReductionMode,
    ) -> (Polynomial<C>, ReductionStats) {
        let mut out: Vec<Term<C>> = Vec::new();
        let mut stats = ReductionStats::default();
        if self.has_unit {
            // everything is zero modulo a unit; drain for completeness
            while acc.next_term().is_some() {}
            return (Polynomial::zero(self.alg.clone()), stats);
        }
        while let Some(t) = acc.next_term() {
            stats.max_heap = stats.max_heap.max(acc.live_streams() + 1);
            if mode == ReductionMode::Top && !out.is_empty() {
                out.push(t);
                continue;
            }
            match self.find_divisor(&t.word) {
                Some((k, start)) => {
                    let g = &self.basis[k];
                    let lw_len = g.leading_word().expect("nonzero").len();
                    let a = &t.word[..start];
                    let b = &t.word[start + lw_len..];
                    // subtract (c / lc(g)) · a · g · b; the leading terms cancel exactly,
                    // so only the tail of g enters the heap.
                    let c = t.coeff.mul_ref(&self.lc_inv[k]).negated();
                    acc.push_scaled_terms(c, a, &g.terms()[1..], b);
                    stats.steps += 1;
                }
                None => out.push(t),
            }
        }
        stats.remainder_terms = out.len();
        (
            Polynomial::from_sorted_terms_unchecked(self.alg.clone(), out),
            stats,
        )
    }

    /// Fully reduce `p`.
    pub fn reduce(&self, p: &Polynomial<C>) -> Polynomial<C> {
        self.reduce_with_stats(p, ReductionMode::Full).0
    }

    /// Reduce `p` with the given mode and return statistics.
    pub fn reduce_with_stats(
        &self,
        p: &Polynomial<C>,
        mode: ReductionMode,
    ) -> (Polynomial<C>, ReductionStats) {
        let mut acc = Accumulator::new(&self.alg);
        acc.push_poly(p);
        self.run(acc, mode)
    }

    /// Top-reduce `p`: the result has an irreducible leading term.
    pub fn reduce_top(&self, p: &Polynomial<C>) -> Polynomial<C> {
        self.reduce_with_stats(p, ReductionMode::Top).0
    }

    /// Reduce only the tail of `p`, keeping its leading term untouched.
    /// Used for interreducing Gröbner bases.
    pub fn reduce_tail(&self, p: &Polynomial<C>) -> Polynomial<C> {
        match p.leading_term() {
            None => p.clone(),
            Some(lt) => {
                let mut acc = Accumulator::new(&self.alg);
                acc.push_scaled_terms(C::one(), &[], &p.terms()[1..], &[]);
                let (tail, _) = self.run(acc, ReductionMode::Full);
                let mut terms = Vec::with_capacity(tail.num_terms() + 1);
                terms.push(lt.clone());
                terms.extend(tail.into_terms());
                Polynomial::from_sorted_terms_unchecked(self.alg.clone(), terms)
            }
        }
    }

    /// The normal form of `f · g`, computed without forming the unreduced
    /// product: product terms are generated in decreasing order and reduced
    /// as they appear.
    pub fn reduce_product(&self, f: &Polynomial<C>, g: &Polynomial<C>) -> Polynomial<C> {
        let mut acc = Accumulator::new(&self.alg);
        acc.push_product(f, g);
        self.run(acc, ReductionMode::Full).0
    }

    /// Normal form of `Σ c_i · f_i`.
    pub fn reduce_linear_combination(&self, parts: &[(C, &Polynomial<C>)]) -> Polynomial<C> {
        let mut acc = Accumulator::new(&self.alg);
        for (c, p) in parts {
            acc.push_scaled_poly(c, p);
        }
        self.run(acc, ReductionMode::Full).0
    }

    /// Normal form of `left · f · right`.
    pub fn reduce_sandwich(&self, left: &[Var], f: &Polynomial<C>, right: &[Var]) -> Polynomial<C> {
        let mut acc = Accumulator::new(&self.alg);
        acc.push_sandwich(left, f, right);
        self.run(acc, ReductionMode::Full).0
    }

    /// Normal form of the commutator `[f, g]`, both products streamed into one heap.
    pub fn reduce_commutator(&self, f: &Polynomial<C>, g: &Polynomial<C>) -> Polynomial<C> {
        let mut acc = Accumulator::new(&self.alg);
        acc.push_product(f, g);
        acc.push_scaled_product(&C::one().negated(), g, f);
        self.run(acc, ReductionMode::Full).0
    }

    /// Normal form of the anticommutator `{f, g}`.
    pub fn reduce_anticommutator(&self, f: &Polynomial<C>, g: &Polynomial<C>) -> Polynomial<C> {
        let mut acc = Accumulator::new(&self.alg);
        acc.push_product(f, g);
        acc.push_product(g, f);
        self.run(acc, ReductionMode::Full).0
    }

    /// Normal form of `f^n`, reducing after every multiplication (binary
    /// exponentiation, so `O(log n)` reduced products).
    pub fn reduce_pow(&self, f: &Polynomial<C>, n: u32) -> Polynomial<C> {
        // NF(1): zero in the unit ideal (an unreduced 1 made f^0 nonzero
        // in the zero algebra)
        let mut result = self.reduce(&self.alg.one::<C>());
        let mut base = self.reduce(f);
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                result = self.reduce_product(&result, &base);
            }
            e >>= 1;
            if e > 0 {
                base = self.reduce_product(&base, &base);
            }
        }
        result
    }

    /// Normal form of the product `f_1 · f_2 · … · f_k`, reducing after each
    /// factor so that no unreduced intermediate is ever formed.
    pub fn reduce_product_chain<'b, I>(&self, factors: I) -> Polynomial<C>
    where
        I: IntoIterator<Item = &'b Polynomial<C>>,
        C: 'b,
    {
        let mut acc: Option<Polynomial<C>> = None;
        for f in factors {
            acc = Some(match acc {
                None => self.reduce(f),
                Some(a) => self.reduce_product(&a, f),
            });
        }
        acc.unwrap_or_else(|| self.reduce(&self.alg.one::<C>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_multiplication_matches_naive() {
        let alg = FreeAlgebra::new(["x", "y"]);
        let f: Polynomial<i64> = alg.parse("x*y + 2*y - 3 + x^2").unwrap();
        let g: Polynomial<i64> = alg.parse("y*x - x + 5").unwrap();
        let prod = f.mul_ref(&g);
        // naive
        let mut terms = Vec::new();
        for a in f.terms() {
            for b in g.terms() {
                terms.push((a.word.concat(&b.word), a.coeff * b.coeff));
            }
        }
        let naive = Polynomial::from_terms(alg.clone(), terms);
        assert_eq!(prod, naive);
    }

    #[test]
    fn reduce_simple() {
        let alg = FreeAlgebra::new(["x", "y"]);
        // modulo yx - xy (commutation), x^2 y x should become x^3 y
        let rel: Polynomial<f64> = alg.parse("y*x - x*y").unwrap();
        let red = Reducer::new(&alg, vec![rel]);
        let p: Polynomial<f64> = alg.parse("x^2*y*x + y*y*x").unwrap();
        let r = red.reduce(&p);
        let expected: Polynomial<f64> = alg.parse("x^3*y + x*y^2").unwrap();
        assert_eq!(r, expected);
    }

    #[test]
    fn reduce_product_equals_reduce_of_product() {
        let alg = FreeAlgebra::new(["x", "y"]);
        let rel: Polynomial<f64> = alg.parse("y*x - x*y - 1").unwrap(); // Weyl
        let red = Reducer::new(&alg, vec![rel]);
        let f: Polynomial<f64> = alg.parse("y^2 + x").unwrap();
        let g: Polynomial<f64> = alg.parse("x*y + y").unwrap();
        let a = red.reduce_product(&f, &g);
        let b = red.reduce(&f.mul_ref(&g));
        assert_eq!(a, b);
    }
}
