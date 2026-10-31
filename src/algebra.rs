//! The free associative algebra `K<x_1, …, x_n>`: generator names plus a
//! monomial ordering.
//!
//! [`FreeAlgebra`] is a cheap-to-clone handle (an `Arc` underneath).  Every
//! [`Polynomial`] carries one so that it knows how to order and print its
//! terms.  Two algebras are compatible when they have the same generator
//! names and the same ordering.

use crate::coeff::Coeff;
use crate::order::{DegLex, MonomialOrder};
use crate::parse::{self, ParseError};
use crate::poly::Polynomial;
use crate::word::{run_lengths, Var, Word};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

struct Inner {
    names: Vec<String>,
    index: HashMap<String, Var>,
    order: Box<dyn MonomialOrder>,
    /// The order is the plain deglex with `x_0 < x_1 < …`, which can be
    /// compared with a direct slice comparison (hot path of the reducer).
    plain_deglex: bool,
    /// Involution on generators: `adjoint[v]` is the generator `x_v*`
    /// (`v` itself for Hermitian generators).
    adjoint: Vec<Var>,
}

/// A free associative algebra over named generators with a fixed admissible
/// monomial ordering.  Coefficients are chosen per polynomial.
#[derive(Clone)]
pub struct FreeAlgebra {
    inner: Arc<Inner>,
}

impl FreeAlgebra {
    /// Free algebra on the given generators with the [`DegLex`] ordering
    /// (`x_0 < x_1 < …`, i.e. later generators are larger).
    pub fn new<S: Into<String>, I: IntoIterator<Item = S>>(names: I) -> Self {
        Self::with_order(names, DegLex::new())
    }

    /// Free algebra with an explicit monomial ordering; all generators are
    /// Hermitian (`x* = x`).
    pub fn with_order<S, I, O>(names: I, order: O) -> Self
    where
        S: Into<String>,
        I: IntoIterator<Item = S>,
        O: MonomialOrder + 'static,
    {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        let n = names.len();
        Self::with_order_and_adjoint(names, order, (0..n as Var).collect())
    }

    /// Free `*`-algebra with an explicit involution on the generators:
    /// `adjoint[v]` is the index of `x_v*`.  The map must be an involution
    /// (`adjoint[adjoint[v]] == v`).  [`Polynomial::adjoint`] reverses words,
    /// maps every letter through this involution and conjugates coefficients.
    pub fn with_order_and_adjoint<S, I, O>(names: I, order: O, adjoint: Vec<Var>) -> Self
    where
        S: Into<String>,
        I: IntoIterator<Item = S>,
        O: MonomialOrder + 'static,
    {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        Self::with_boxed_order(names, Box::new(order), adjoint)
    }

    /// Like [`with_order_and_adjoint`](Self::with_order_and_adjoint) with an
    /// ordering given as a trait object (used when loading saved files).
    pub fn with_boxed_order(names: Vec<String>, order: Box<dyn MonomialOrder>, adjoint: Vec<Var>) -> Self {
        let mut index = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            assert!(!n.is_empty(), "generator names must be non-empty");
            assert!(
                index.insert(n.clone(), i as Var).is_none(),
                "duplicate generator name {:?}",
                n
            );
        }
        assert_eq!(adjoint.len(), names.len(), "adjoint map has the wrong length");
        for (v, &a) in adjoint.iter().enumerate() {
            assert!((a as usize) < names.len(), "adjoint index out of range");
            assert_eq!(
                adjoint[a as usize] as usize, v,
                "adjoint map is not an involution"
            );
        }
        let plain_deglex = order.name() == "deglex";
        FreeAlgebra {
            inner: Arc::new(Inner {
                names,
                index,
                order,
                plain_deglex,
                adjoint,
            }),
        }
    }

    /// Free `*`-algebra from `(name, is_hermitian)` pairs with the [`DegLex`]
    /// ordering.  Every non-Hermitian generator `a` gets a partner generator
    /// `a†` placed directly after it, with `a* = a†`.
    ///
    /// ```
    /// use ncalg::{FreeAlgebra, Polynomial};
    /// let alg = FreeAlgebra::new_star([("a", false), ("n", true)]);
    /// assert_eq!(alg.names(), &["a", "a†", "n"]);
    /// let p: Polynomial<f64> = alg.parse("a*n + 2*a†").unwrap();
    /// assert_eq!(p.adjoint().to_string(), "n*a† + 2*a");
    /// assert_eq!(alg.parse::<f64>("(a*n)'").unwrap(), alg.parse("n*a†").unwrap());
    /// ```
    pub fn new_star<S: Into<String>, I: IntoIterator<Item = (S, bool)>>(gens: I) -> Self {
        let mut names: Vec<String> = Vec::new();
        let mut adjoint: Vec<Var> = Vec::new();
        for (name, hermitian) in gens {
            let name: String = name.into();
            if hermitian {
                adjoint.push(names.len() as Var);
                names.push(name);
            } else {
                let i = names.len() as Var;
                let dagger = format!("{}†", name);
                names.push(name);
                names.push(dagger);
                adjoint.push(i + 1);
                adjoint.push(i);
            }
        }
        Self::with_order_and_adjoint(names, DegLex::new(), adjoint)
    }

    /// The generator `x_v*`.
    #[inline]
    pub fn adjoint_var(&self, v: Var) -> Var {
        self.inner.adjoint[v as usize]
    }

    /// Is generator `v` Hermitian (`x_v* = x_v`)?
    #[inline]
    pub fn is_hermitian_var(&self, v: Var) -> bool {
        self.inner.adjoint[v as usize] == v
    }

    /// The involution on generators as a vector.
    pub fn adjoint_map(&self) -> &[Var] {
        &self.inner.adjoint
    }

    /// Adjoint of a word: reversed, with every letter mapped through the involution.
    pub fn adjoint_word(&self, w: &[Var]) -> Word {
        Word::from_vec(w.iter().rev().map(|&v| self.inner.adjoint[v as usize]).collect())
    }

    /// Generators named `x0, x1, …`.
    pub fn with_n_vars(n: usize) -> Self {
        Self::new((0..n).map(|i| format!("x{}", i)))
    }

    /// Number of generators.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.inner.names.len()
    }

    /// Generator names.
    pub fn names(&self) -> &[String] {
        &self.inner.names
    }

    /// Name of generator `v`.
    pub fn name(&self, v: Var) -> &str {
        &self.inner.names[v as usize]
    }

    /// Index of the generator called `name`.
    pub fn var_index(&self, name: &str) -> Option<Var> {
        self.inner.index.get(name).copied()
    }

    /// The monomial ordering.
    #[inline]
    pub fn order(&self) -> &dyn MonomialOrder {
        self.inner.order.as_ref()
    }

    /// Compare two words in this algebra's monomial ordering.
    #[inline]
    pub fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering {
        if self.inner.plain_deglex {
            // length first, then lexicographic by generator index: the slice
            // ordering on `[u32]` is exactly the lexicographic comparison
            a.len().cmp(&b.len()).then_with(|| a.cmp(b))
        } else {
            self.inner.order.cmp_words(a, b)
        }
    }

    /// Are two algebra handles compatible (same generators and ordering)?
    pub fn same_as(&self, other: &FreeAlgebra) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
            || (self.inner.names == other.inner.names
                && self.inner.order.name() == other.inner.order.name()
                && self.inner.adjoint == other.inner.adjoint)
    }

    /// Panics with a helpful message if the two algebras differ.
    #[inline]
    pub(crate) fn check_same(&self, other: &FreeAlgebra) {
        assert!(
            self.same_as(other),
            "polynomials belong to different free algebras ({} vs {})",
            self,
            other
        );
    }

    /// Check a word only uses generators of this algebra.
    pub fn check_word(&self, w: &[Var]) {
        if let Some(&m) = w.iter().max() {
            assert!(
                (m as usize) < self.num_vars(),
                "generator index {} out of range for algebra with {} generators",
                m,
                self.num_vars()
            );
        }
    }

    // --- element constructors ------------------------------------------------

    /// The zero polynomial.
    pub fn zero<C: Coeff>(&self) -> Polynomial<C> {
        Polynomial::zero(self.clone())
    }

    /// The constant polynomial `1`.
    pub fn one<C: Coeff>(&self) -> Polynomial<C> {
        self.constant(C::one())
    }

    /// A constant polynomial.
    pub fn constant<C: Coeff>(&self, c: C) -> Polynomial<C> {
        self.term(c, Word::empty())
    }

    /// The generator with index `v` as a polynomial.
    pub fn var<C: Coeff>(&self, v: Var) -> Polynomial<C> {
        assert!((v as usize) < self.num_vars(), "generator index out of range");
        self.term(C::one(), Word::letter(v))
    }

    /// The generator called `name` as a polynomial.  Panics if unknown.
    pub fn gen<C: Coeff>(&self, name: &str) -> Polynomial<C> {
        match self.var_index(name) {
            Some(v) => self.var(v),
            None => panic!("unknown generator {:?} in {}", name, self),
        }
    }

    /// All generators as polynomials, in index order.
    pub fn gens<C: Coeff>(&self) -> Vec<Polynomial<C>> {
        (0..self.num_vars() as Var).map(|v| self.var(v)).collect()
    }

    /// The monomial `w` with coefficient one.
    pub fn monomial<C: Coeff>(&self, w: Word) -> Polynomial<C> {
        self.term(C::one(), w)
    }

    /// A single term `c · w`.
    pub fn term<C: Coeff>(&self, c: C, w: Word) -> Polynomial<C> {
        self.check_word(&w);
        if c.is_zero() {
            return Polynomial::zero(self.clone());
        }
        Polynomial::from_sorted_terms_unchecked(self.clone(), vec![crate::poly::Term { word: w, coeff: c }])
    }

    /// Parse a polynomial from a string such as `"2*x*y - y*x + x^2 - 1/2"`.
    /// Multiplication may be written as `*` or by juxtaposition (`x y`),
    /// `[a, b]` is the commutator and `{a, b}` the anticommutator.
    pub fn parse<C: Coeff>(&self, s: &str) -> Result<Polynomial<C>, ParseError> {
        parse::parse(self, s)
    }

    /// All words of the given length, in increasing monomial order.
    pub fn words_of_length(&self, len: usize) -> Vec<Word> {
        let n = self.num_vars() as Var;
        let mut out = vec![Word::empty()];
        for _ in 0..len {
            out = out
                .into_iter()
                .flat_map(|w| {
                    (0..n).map(move |v| {
                        let mut x = w.clone();
                        x.push(v);
                        x
                    })
                })
                .collect();
        }
        out.sort_by(|a, b| self.cmp_words(a, b));
        out
    }

    /// Render a word with generator names, e.g. `x*y^2*x`; the empty word is `1`.
    pub fn format_word(&self, w: &[Var]) -> String {
        if w.is_empty() {
            return "1".to_string();
        }
        let mut s = String::new();
        for (i, (v, n)) in run_lengths(w).into_iter().enumerate() {
            if i > 0 {
                s.push('*');
            }
            s.push_str(self.name(v));
            if n > 1 {
                s.push('^');
                s.push_str(&n.to_string());
            }
        }
        s
    }
}

impl PartialEq for FreeAlgebra {
    fn eq(&self, other: &Self) -> bool {
        self.same_as(other)
    }
}

impl fmt::Debug for FreeAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FreeAlgebra<{}; {}>",
            self.inner.names.join(", "),
            self.inner.order.name()
        )
    }
}

impl fmt::Display for FreeAlgebra {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "K<{}> ({})",
            self.inner.names.join(", "),
            self.inner.order.name()
        )
    }
}
