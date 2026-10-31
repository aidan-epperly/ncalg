//! Arithmetic in the quotient algebra `K<X>/I`.
//!
//! A [`QuotientAlgebra`] wraps a [`GroebnerBasis`]; its elements
//! ([`QuotientElement`]) are stored in normal form and every operation
//! reduces while expanding, so an element never holds an unreduced
//! representative.  Also provides the normal words (a vector space basis of
//! the quotient) by degree and the Hilbert series.
//!
//! If the Gröbner basis is only partial, "normal form" means normal form
//! modulo the partial basis: still a valid representative of the coset, but
//! two equal cosets may have different representatives.  For homogeneous
//! relations under a degree ordering the representatives are canonical up to
//! the degree bound.

use crate::algebra::FreeAlgebra;
use crate::coeff::Field;
use crate::groebner::{GroebnerBasis, GroebnerOptions};
use crate::parse::ParseError;
use crate::poly::Polynomial;
use crate::reduce::Reducer;
use crate::word::{Var, Word};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::Arc;

/// The quotient of a free algebra by a two-sided ideal, represented by a
/// (possibly partial) Gröbner basis.  Cheap to clone.
#[derive(Clone)]
pub struct QuotientAlgebra<C: Field> {
    gb: Arc<GroebnerBasis<C>>,
}

impl<C: Field> fmt::Debug for QuotientAlgebra<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuotientAlgebra({} / {} relations, {})",
            self.gb.algebra(),
            self.gb.len(),
            self.gb.termination()
        )
    }
}

impl<C: Field> QuotientAlgebra<C> {
    /// Quotient modulo a computed Gröbner basis.
    pub fn new(gb: GroebnerBasis<C>) -> Self {
        QuotientAlgebra { gb: Arc::new(gb) }
    }

    /// Compute the Gröbner basis of the relations and form the quotient.
    pub fn from_relations(relations: Vec<Polynomial<C>>, opts: &GroebnerOptions) -> Self {
        Self::new(GroebnerBasis::compute(relations, opts))
    }

    /// The underlying Gröbner basis.
    pub fn groebner_basis(&self) -> &GroebnerBasis<C> {
        &self.gb
    }

    /// The free algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        self.gb.algebra()
    }

    /// The reducer.
    pub fn reducer(&self) -> &Reducer<C> {
        self.gb.reducer()
    }

    /// Is the Gröbner basis complete (so representatives are canonical)?
    pub fn is_complete(&self) -> bool {
        self.gb.is_complete()
    }

    /// Is the quotient the zero algebra?
    pub fn is_zero_algebra(&self) -> bool {
        self.gb.is_unit_ideal()
    }

    fn same_as(&self, other: &QuotientAlgebra<C>) -> bool {
        Arc::ptr_eq(&self.gb, &other.gb)
            || (self.gb.algebra().same_as(other.gb.algebra()) && self.gb.elements() == other.gb.elements())
    }

    // --- elements ---------------------------------------------------------

    /// The coset of `p`, stored in normal form.
    pub fn element(&self, p: &Polynomial<C>) -> QuotientElement<C> {
        self.algebra().check_same(p.algebra());
        QuotientElement {
            q: self.clone(),
            poly: self.gb.reduce(p),
        }
    }

    /// Wrap a polynomial already known to be in normal form.
    fn element_reduced(&self, poly: Polynomial<C>) -> QuotientElement<C> {
        debug_assert!(self.gb.reducer().is_reduced(&poly));
        QuotientElement {
            q: self.clone(),
            poly,
        }
    }

    /// The zero element.
    pub fn zero(&self) -> QuotientElement<C> {
        self.element_reduced(self.algebra().zero())
    }

    /// The unit element (its normal form, zero if the ideal is the whole algebra).
    pub fn one(&self) -> QuotientElement<C> {
        self.element(&self.algebra().one())
    }

    /// The constant `c`, in normal form.
    pub fn constant(&self, c: C) -> QuotientElement<C> {
        self.element(&self.algebra().constant(c))
    }

    /// Generator with index `v`.
    pub fn var(&self, v: Var) -> QuotientElement<C> {
        self.element(&self.algebra().var(v))
    }

    /// Generator by name.
    pub fn gen(&self, name: &str) -> QuotientElement<C> {
        self.element(&self.algebra().gen(name))
    }

    /// All generators.
    pub fn gens(&self) -> Vec<QuotientElement<C>> {
        (0..self.algebra().num_vars() as Var)
            .map(|v| self.var(v))
            .collect()
    }

    /// Parse and reduce.
    pub fn parse(&self, s: &str) -> Result<QuotientElement<C>, ParseError> {
        let p: Polynomial<C> = self.algebra().parse(s)?;
        Ok(self.element(&p))
    }

    /// Normal form of a polynomial.
    pub fn reduce(&self, p: &Polynomial<C>) -> Polynomial<C> {
        self.gb.reduce(p)
    }

    // --- normal words / Hilbert series ------------------------------------

    /// Normal words of length exactly `d` (words containing no leading word
    /// of the basis), in increasing monomial order.  These are a basis of the
    /// degree-`d` part of the quotient when the relations are homogeneous and
    /// the Gröbner basis is complete up to degree `d`.
    pub fn normal_words(&self, d: usize) -> Vec<Word> {
        self.gb.reducer().normal_words(d)
    }

    /// Normal words of all lengths `0..=d`.
    pub fn normal_words_up_to(&self, d: usize) -> Vec<Word> {
        (0..=d).flat_map(|k| self.normal_words(k)).collect()
    }

    /// Number of normal words of each length `0..=max_degree` (the Hilbert
    /// series coefficients for homogeneous relations).  Computed by dynamic
    /// programming over the automaton, without enumerating words.
    pub fn hilbert_series(&self, max_degree: usize) -> Vec<usize> {
        let auto = self.gb.reducer().automaton();
        let n_states = auto.num_states();
        let n = self.algebra().num_vars() as Var;
        if auto.has_empty_pattern() {
            return vec![0; max_degree + 1];
        }
        let mut counts = vec![0u128; n_states];
        counts[auto.root() as usize] = 1;
        let mut out = Vec::with_capacity(max_degree + 1);
        out.push(1);
        for _ in 0..max_degree {
            counts = auto.next_layer(n as usize, &counts);
            out.push(to_usize(counts.iter().sum::<u128>()));
        }
        out
    }

    /// Normal forms of all products `u† · v` for `u, v` in `words` (row `i`,
    /// column `j` is `words[i]† · words[j]`): the symbolic entries of an
    /// NPA / noncommutative-SOS moment matrix indexed by `words`.  Each entry
    /// is computed with the lazy reducer, so no unreduced product is formed.
    pub fn adjoint_products(&self, words: &[Word]) -> Vec<Vec<Polynomial<C>>> {
        let alg = self.algebra();
        let red = self.reducer();
        words
            .iter()
            .map(|u| {
                let u_adj = alg.adjoint_word(u);
                words
                    .iter()
                    .map(|v| {
                        let mut acc = crate::reduce::Accumulator::new(alg);
                        let one = alg.one::<C>();
                        acc.push_sandwich(&u_adj, &one, v);
                        red.run(acc, crate::reduce::ReductionMode::Full).0
                    })
                    .collect()
            })
            .collect()
    }

    /// Total number of normal words if finite (the quotient is finite
    /// dimensional as a vector space when the relations are a complete
    /// Gröbner basis), `None` if there are infinitely many.
    pub fn dimension(&self) -> Option<usize> {
        let auto = self.gb.reducer().automaton();
        if auto.has_empty_pattern() {
            return Some(0);
        }
        let n = self.algebra().num_vars() as Var;
        // cycle detection on the graph of non-terminal states reachable from the root
        let n_states = auto.num_states();
        let mut color = vec![0u8; n_states]; // 0 white, 1 gray, 2 black
        let mut stack: Vec<(u32, Var)> = vec![(auto.root(), 0)];
        color[auto.root() as usize] = 1;
        while let Some(&mut (s, ref mut next_v)) = stack.last_mut() {
            if *next_v >= n {
                color[s as usize] = 2;
                stack.pop();
                continue;
            }
            let v = *next_v;
            *next_v += 1;
            let t = auto.step(s, v);
            if auto.is_terminal(t) {
                continue;
            }
            match color[t as usize] {
                0 => {
                    color[t as usize] = 1;
                    stack.push((t, 0));
                }
                1 => return None, // back edge: infinitely many normal words
                _ => {}
            }
        }
        // acyclic: the longest normal word has length < n_states; count by
        // dynamic programming until no normal word of the next length exists
        let mut counts = vec![0u128; n_states];
        counts[auto.root() as usize] = 1;
        let mut total = 0usize;
        loop {
            let layer = to_usize(counts.iter().sum::<u128>());
            if layer == 0 {
                break;
            }
            total = total
                .checked_add(layer)
                .expect("quotient dimension overflows usize");
            counts = auto.next_layer(n as usize, &counts);
        }
        Some(total)
    }
}

impl<C: Field> PartialEq for QuotientAlgebra<C> {
    fn eq(&self, other: &Self) -> bool {
        self.same_as(other)
    }
}

/// An element of a quotient algebra, kept in normal form.
#[derive(Clone)]
pub struct QuotientElement<C: Field> {
    q: QuotientAlgebra<C>,
    poly: Polynomial<C>,
}

impl<C: Field> QuotientElement<C> {
    /// The normal-form representative.
    pub fn representative(&self) -> &Polynomial<C> {
        &self.poly
    }

    /// Consume into the representative.
    pub fn into_polynomial(self) -> Polynomial<C> {
        self.poly
    }

    /// The quotient algebra.
    pub fn quotient(&self) -> &QuotientAlgebra<C> {
        &self.q
    }

    /// Is this element zero in the quotient?
    pub fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }

    fn check(&self, other: &QuotientElement<C>) {
        assert!(
            self.q.same_as(&other.q),
            "elements of different quotient algebras"
        );
    }

    /// `self + other`.
    pub fn add_ref(&self, other: &QuotientElement<C>) -> QuotientElement<C> {
        self.check(other);
        // sum of normal forms is a normal form
        self.q.element_reduced(self.poly.add_ref(&other.poly))
    }

    /// `self - other`.
    pub fn sub_ref(&self, other: &QuotientElement<C>) -> QuotientElement<C> {
        self.check(other);
        self.q.element_reduced(self.poly.sub_ref(&other.poly))
    }

    /// `self · other`, reduced while expanding.
    pub fn mul_ref(&self, other: &QuotientElement<C>) -> QuotientElement<C> {
        self.check(other);
        self.q
            .element_reduced(self.q.gb.reduce_product(&self.poly, &other.poly))
    }

    /// `-self`.
    pub fn negated(&self) -> QuotientElement<C> {
        self.q.element_reduced(self.poly.negated())
    }

    /// `c · self`.
    pub fn scale(&self, c: &C) -> QuotientElement<C> {
        self.q.element_reduced(self.poly.scale(c))
    }

    /// `self^n`, reduced after every squaring.
    pub fn pow(&self, n: u32) -> QuotientElement<C> {
        self.q.element_reduced(self.q.gb.reduce_pow(&self.poly, n))
    }

    /// `[self, other]`.
    pub fn commutator(&self, other: &QuotientElement<C>) -> QuotientElement<C> {
        self.check(other);
        self.q
            .element_reduced(self.q.gb.reduce_commutator(&self.poly, &other.poly))
    }

    /// `{self, other}`.
    pub fn anticommutator(&self, other: &QuotientElement<C>) -> QuotientElement<C> {
        self.check(other);
        self.q
            .element_reduced(self.q.gb.reducer().reduce_anticommutator(&self.poly, &other.poly))
    }

    /// Hermitian adjoint (reduced afterwards; the ideal should be closed
    /// under the adjoint for this to be well defined on cosets).
    pub fn adjoint(&self) -> QuotientElement<C> {
        self.q.element(&self.poly.adjoint())
    }

    /// Does `other` commute with `self` in the quotient?
    pub fn commutes_with(&self, other: &QuotientElement<C>) -> bool {
        self.commutator(other).is_zero()
    }
}

impl<C: Field> PartialEq for QuotientElement<C> {
    fn eq(&self, other: &Self) -> bool {
        self.q.same_as(&other.q) && self.poly == other.poly
    }
}

impl<C: Field> fmt::Display for QuotientElement<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.poly)
    }
}

impl<C: Field> fmt::Debug for QuotientElement<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.poly)
    }
}

macro_rules! impl_q_binop {
    ($Trait:ident, $method:ident, $inner:ident) => {
        impl<C: Field> $Trait<QuotientElement<C>> for QuotientElement<C> {
            type Output = QuotientElement<C>;
            fn $method(self, o: QuotientElement<C>) -> QuotientElement<C> {
                (&self).$inner(&o)
            }
        }
        impl<'a, C: Field> $Trait<&'a QuotientElement<C>> for QuotientElement<C> {
            type Output = QuotientElement<C>;
            fn $method(self, o: &'a QuotientElement<C>) -> QuotientElement<C> {
                (&self).$inner(o)
            }
        }
        impl<'a, C: Field> $Trait<QuotientElement<C>> for &'a QuotientElement<C> {
            type Output = QuotientElement<C>;
            fn $method(self, o: QuotientElement<C>) -> QuotientElement<C> {
                self.$inner(&o)
            }
        }
        impl<'a, 'b, C: Field> $Trait<&'b QuotientElement<C>> for &'a QuotientElement<C> {
            type Output = QuotientElement<C>;
            fn $method(self, o: &'b QuotientElement<C>) -> QuotientElement<C> {
                self.$inner(o)
            }
        }
    };
}
impl_q_binop!(Add, add, add_ref);
impl_q_binop!(Sub, sub, sub_ref);
impl_q_binop!(Mul, mul, mul_ref);

impl<C: Field> Neg for QuotientElement<C> {
    type Output = QuotientElement<C>;
    fn neg(self) -> QuotientElement<C> {
        self.negated()
    }
}
impl<C: Field> Neg for &QuotientElement<C> {
    type Output = QuotientElement<C>;
    fn neg(self) -> QuotientElement<C> {
        self.negated()
    }
}
impl<C: Field> Mul<C> for QuotientElement<C> {
    type Output = QuotientElement<C>;
    fn mul(self, c: C) -> QuotientElement<C> {
        self.scale(&c)
    }
}
impl<C: Field> Mul<C> for &QuotientElement<C> {
    type Output = QuotientElement<C>;
    fn mul(self, c: C) -> QuotientElement<C> {
        self.scale(&c)
    }
}
impl<C: Field> AddAssign<&QuotientElement<C>> for QuotientElement<C> {
    fn add_assign(&mut self, o: &QuotientElement<C>) {
        *self = self.add_ref(o);
    }
}
impl<C: Field> AddAssign<QuotientElement<C>> for QuotientElement<C> {
    fn add_assign(&mut self, o: QuotientElement<C>) {
        *self = self.add_ref(&o);
    }
}
impl<C: Field> SubAssign<&QuotientElement<C>> for QuotientElement<C> {
    fn sub_assign(&mut self, o: &QuotientElement<C>) {
        *self = self.sub_ref(o);
    }
}
impl<C: Field> SubAssign<QuotientElement<C>> for QuotientElement<C> {
    fn sub_assign(&mut self, o: QuotientElement<C>) {
        *self = self.sub_ref(&o);
    }
}
impl<C: Field> MulAssign<&QuotientElement<C>> for QuotientElement<C> {
    fn mul_assign(&mut self, o: &QuotientElement<C>) {
        *self = self.mul_ref(o);
    }
}
impl<C: Field> MulAssign<QuotientElement<C>> for QuotientElement<C> {
    fn mul_assign(&mut self, o: QuotientElement<C>) {
        *self = self.mul_ref(&o);
    }
}

/// A normal-word count as `usize`, failing loudly instead of wrapping.
fn to_usize(c: u128) -> usize {
    usize::try_from(c).expect("normal-word count overflows usize")
}
