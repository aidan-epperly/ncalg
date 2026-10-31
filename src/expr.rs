//! Expression trees evaluated modulo an ideal, reducing while expanding.
//!
//! An [`Expr`] is a symbolic combination of polynomials: sums, products,
//! powers, commutators, anticommutators, scalar multiples and adjoints.
//! [`Expr::eval_mod`] evaluates it modulo a [`Reducer`] (typically a Gröbner
//! basis) without ever forming an unreduced expansion:
//!
//! * products are evaluated left to right and every partial product is
//!   reduced before being multiplied by the next factor, with the product
//!   terms themselves streamed through the reducer's heap;
//! * sums, commutators and anticommutators of products push their product
//!   streams into one shared heap, so the whole node is reduced in a single
//!   pass;
//! * powers use binary exponentiation with reduction after every squaring.
//!
//! For a complete Gröbner basis the result is the unique normal form of the
//! fully expanded expression, but the peak size of any intermediate is
//! bounded by the sizes of *reduced* polynomials rather than by the size of
//! the free expansion.
//!
//! ```
//! use ncalg::{FreeAlgebra, Polynomial, Expr, GroebnerBasis, GroebnerOptions, BigRational};
//! let alg = FreeAlgebra::new(["x", "y"]);
//! let rel: Polynomial<BigRational> = alg.parse("y*x - x*y - 1").unwrap(); // Weyl algebra
//! let gb = GroebnerBasis::compute(vec![rel], &GroebnerOptions::default());
//! let x: Expr<BigRational> = alg.parse::<BigRational>("x").unwrap().into();
//! let y: Expr<BigRational> = alg.parse::<BigRational>("y").unwrap().into();
//! // (x + y)^4 modulo the Weyl relation, reduced while expanding
//! let e = (x + y).pow(4);
//! let nf = e.eval_mod(gb.reducer());
//! assert_eq!(nf, gb.reduce(&alg.parse::<BigRational>("(x + y)^4").unwrap()));
//! ```

use crate::algebra::FreeAlgebra;
use crate::coeff::{Coeff, Field};
use crate::poly::Polynomial;
use crate::reduce::{Accumulator, Reducer, ReductionMode};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A symbolic expression over polynomials of one free algebra.
#[derive(Clone, Debug)]
pub enum Expr<C: Coeff> {
    /// A polynomial leaf.
    Poly(Polynomial<C>),
    /// `e_1 + e_2 + …`
    Sum(Vec<Expr<C>>),
    /// `e_1 · e_2 · …`
    Prod(Vec<Expr<C>>),
    /// `-e`
    Neg(Box<Expr<C>>),
    /// `c · e`
    Scale(C, Box<Expr<C>>),
    /// `e^n`
    Pow(Box<Expr<C>>, u32),
    /// `[a, b] = ab - ba`
    Commutator(Box<Expr<C>>, Box<Expr<C>>),
    /// `{a, b} = ab + ba`
    Anticommutator(Box<Expr<C>>, Box<Expr<C>>),
    /// Hermitian adjoint (reverse words, conjugate coefficients).
    Adjoint(Box<Expr<C>>),
}

/// Stream specifications collected from an expression before they are pushed
/// into one accumulator (two-phase so the arena can be borrowed immutably).
enum Spec<C> {
    Single(C, usize),
    Product(C, usize, usize),
}

impl<C: Coeff> Expr<C> {
    /// Leaf.
    pub fn poly(p: Polynomial<C>) -> Self {
        Expr::Poly(p)
    }

    /// Sum of expressions.
    pub fn sum(terms: Vec<Expr<C>>) -> Self {
        Expr::Sum(terms)
    }

    /// Product of expressions (left to right).
    pub fn product(factors: Vec<Expr<C>>) -> Self {
        Expr::Prod(factors)
    }

    /// `self^n`.
    pub fn pow(self, n: u32) -> Self {
        Expr::Pow(Box::new(self), n)
    }

    /// `c · self`.
    pub fn scale(self, c: C) -> Self {
        Expr::Scale(c, Box::new(self))
    }

    /// `[a, b]`.
    pub fn commutator(a: Expr<C>, b: Expr<C>) -> Self {
        Expr::Commutator(Box::new(a), Box::new(b))
    }

    /// `{a, b}`.
    pub fn anticommutator(a: Expr<C>, b: Expr<C>) -> Self {
        Expr::Anticommutator(Box::new(a), Box::new(b))
    }

    /// Hermitian adjoint.
    pub fn adjoint(self) -> Self {
        Expr::Adjoint(Box::new(self))
    }

    /// The algebra of the first polynomial leaf found, if any.
    pub fn algebra(&self) -> Option<&FreeAlgebra> {
        match self {
            Expr::Poly(p) => Some(p.algebra()),
            Expr::Sum(v) | Expr::Prod(v) => v.iter().find_map(|e| e.algebra()),
            Expr::Neg(e) | Expr::Scale(_, e) | Expr::Pow(e, _) | Expr::Adjoint(e) => e.algebra(),
            Expr::Commutator(a, b) | Expr::Anticommutator(a, b) => a.algebra().or_else(|| b.algebra()),
        }
    }

    /// Number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        1 + match self {
            Expr::Poly(_) => 0,
            Expr::Sum(v) | Expr::Prod(v) => v.iter().map(|e| e.num_nodes()).sum(),
            Expr::Neg(e) | Expr::Scale(_, e) | Expr::Pow(e, _) | Expr::Adjoint(e) => e.num_nodes(),
            Expr::Commutator(a, b) | Expr::Anticommutator(a, b) => a.num_nodes() + b.num_nodes(),
        }
    }

    /// Evaluate in the free algebra (no reduction).  Products and sums still
    /// go through the heap accumulator so terms come out sorted directly.
    pub fn eval_free(&self) -> Polynomial<C> {
        match self {
            Expr::Poly(p) => p.clone(),
            Expr::Sum(v) => {
                let alg = self
                    .algebra()
                    .expect("expression without polynomial leaves")
                    .clone();
                let vals: Vec<Polynomial<C>> = v.iter().map(|e| e.eval_free()).collect();
                Polynomial::sum(&alg, vals.iter())
            }
            Expr::Prod(v) => {
                let alg = self
                    .algebra()
                    .expect("expression without polynomial leaves")
                    .clone();
                let vals: Vec<Polynomial<C>> = v.iter().map(|e| e.eval_free()).collect();
                Polynomial::product(&alg, vals.iter())
            }
            Expr::Neg(e) => e.eval_free().negated(),
            Expr::Scale(c, e) => e.eval_free().scale(c),
            Expr::Pow(e, n) => e.eval_free().pow(*n),
            Expr::Commutator(a, b) => Polynomial::commutator(&a.eval_free(), &b.eval_free()),
            Expr::Anticommutator(a, b) => Polynomial::anticommutator(&a.eval_free(), &b.eval_free()),
            Expr::Adjoint(e) => e.eval_free().adjoint(),
        }
    }
}

impl<C: Field> Expr<C> {
    /// Evaluate modulo the reducer's basis, reducing while expanding.  See the
    /// [module documentation](self).
    pub fn eval_mod(&self, red: &Reducer<C>) -> Polynomial<C> {
        let mut arena: Vec<Polynomial<C>> = Vec::new();
        let mut specs: Vec<Spec<C>> = Vec::new();
        self.collect(red, C::one(), &mut arena, &mut specs);
        let mut acc = Accumulator::new(red.algebra());
        for s in &specs {
            match s {
                Spec::Single(c, i) => acc.push_scaled_poly(c, &arena[*i]),
                Spec::Product(c, i, j) => acc.push_scaled_product(c, &arena[*i], &arena[*j]),
            }
        }
        red.run(acc, ReductionMode::Full).0
    }

    /// Flatten this node into stream specifications with an overall scale.
    fn collect(&self, red: &Reducer<C>, scale: C, arena: &mut Vec<Polynomial<C>>, specs: &mut Vec<Spec<C>>) {
        if scale.is_zero() {
            return;
        }
        match self {
            Expr::Poly(p) => {
                arena.push(p.clone());
                specs.push(Spec::Single(scale, arena.len() - 1));
            }
            Expr::Sum(v) => {
                for e in v {
                    e.collect(red, scale.clone(), arena, specs);
                }
            }
            Expr::Neg(e) => e.collect(red, scale.negated(), arena, specs),
            Expr::Scale(c, e) => e.collect(red, scale.mul_ref(c), arena, specs),
            Expr::Prod(v) => match v.len() {
                0 => {
                    arena.push(red.algebra().one());
                    specs.push(Spec::Single(scale, arena.len() - 1));
                }
                1 => v[0].collect(red, scale, arena, specs),
                n => {
                    // reduced prefix product of the first n-1 factors, then
                    // stream prefix · last into the caller's heap
                    let mut prefix = v[0].eval_mod(red);
                    for e in &v[1..n - 1] {
                        let f = e.eval_mod(red);
                        prefix = red.reduce_product(&prefix, &f);
                    }
                    let last = v[n - 1].eval_mod(red);
                    arena.push(prefix);
                    let i = arena.len() - 1;
                    arena.push(last);
                    let j = arena.len() - 1;
                    specs.push(Spec::Product(scale, i, j));
                }
            },
            Expr::Pow(e, n) => {
                let base = e.eval_mod(red);
                arena.push(red.reduce_pow(&base, *n));
                specs.push(Spec::Single(scale, arena.len() - 1));
            }
            Expr::Commutator(a, b) => {
                arena.push(a.eval_mod(red));
                let i = arena.len() - 1;
                arena.push(b.eval_mod(red));
                let j = arena.len() - 1;
                specs.push(Spec::Product(scale.clone(), i, j));
                specs.push(Spec::Product(scale.negated(), j, i));
            }
            Expr::Anticommutator(a, b) => {
                arena.push(a.eval_mod(red));
                let i = arena.len() - 1;
                arena.push(b.eval_mod(red));
                let j = arena.len() - 1;
                specs.push(Spec::Product(scale.clone(), i, j));
                specs.push(Spec::Product(scale, j, i));
            }
            Expr::Adjoint(e) => {
                // The adjoint of a normal form is not the normal form of the
                // adjoint unless the ideal is closed under the adjoint, so push
                // the adjoint down to the leaves (it is an anti-automorphism)
                // and evaluate the rewritten tree lazily.
                e.push_adjoint().collect(red, scale, arena, specs)
            }
        }
    }
}

impl<C: Coeff> Expr<C> {
    /// Rewrite `self†` as an expression without [`Expr::Adjoint`] nodes, using
    /// `(ab)† = b† a†`, `(a+b)† = a† + b†`, `(c·a)† = c̄·a†`, `[a,b]† = [b†,a†]`.
    pub fn push_adjoint(&self) -> Expr<C> {
        match self {
            Expr::Poly(p) => Expr::Poly(p.adjoint()),
            Expr::Sum(v) => Expr::Sum(v.iter().map(|e| e.push_adjoint()).collect()),
            Expr::Prod(v) => Expr::Prod(v.iter().rev().map(|e| e.push_adjoint()).collect()),
            Expr::Neg(e) => Expr::Neg(Box::new(e.push_adjoint())),
            Expr::Scale(c, e) => Expr::Scale(c.conj(), Box::new(e.push_adjoint())),
            Expr::Pow(e, n) => Expr::Pow(Box::new(e.push_adjoint()), *n),
            Expr::Commutator(a, b) => {
                Expr::Commutator(Box::new(b.push_adjoint()), Box::new(a.push_adjoint()))
            }
            Expr::Anticommutator(a, b) => {
                Expr::Anticommutator(Box::new(b.push_adjoint()), Box::new(a.push_adjoint()))
            }
            Expr::Adjoint(e) => (**e).clone(),
        }
    }
}

impl<C: Coeff> From<Polynomial<C>> for Expr<C> {
    fn from(p: Polynomial<C>) -> Self {
        Expr::Poly(p)
    }
}

impl<C: Coeff> From<&Polynomial<C>> for Expr<C> {
    fn from(p: &Polynomial<C>) -> Self {
        Expr::Poly(p.clone())
    }
}

impl<C: Coeff> fmt::Display for Expr<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Poly(p) => write!(f, "({})", p),
            Expr::Sum(v) => {
                write!(f, "(")?;
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Expr::Prod(v) => {
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, "*")?;
                    }
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
            Expr::Neg(e) => write!(f, "-{}", e),
            Expr::Scale(c, e) => write!(f, "{}*{}", c, e),
            Expr::Pow(e, n) => write!(f, "{}^{}", e, n),
            Expr::Commutator(a, b) => write!(f, "[{}, {}]", a, b),
            Expr::Anticommutator(a, b) => write!(f, "{{{}, {}}}", a, b),
            Expr::Adjoint(e) => write!(f, "{}†", e),
        }
    }
}

// operator overloading: flatten nested sums / products
impl<C: Coeff> Add for Expr<C> {
    type Output = Expr<C>;
    fn add(self, o: Expr<C>) -> Expr<C> {
        match (self, o) {
            (Expr::Sum(mut a), Expr::Sum(b)) => {
                a.extend(b);
                Expr::Sum(a)
            }
            (Expr::Sum(mut a), b) => {
                a.push(b);
                Expr::Sum(a)
            }
            (a, b) => Expr::Sum(vec![a, b]),
        }
    }
}
impl<C: Coeff> Sub for Expr<C> {
    type Output = Expr<C>;
    fn sub(self, o: Expr<C>) -> Expr<C> {
        self + Expr::Neg(Box::new(o))
    }
}
impl<C: Coeff> Mul for Expr<C> {
    type Output = Expr<C>;
    fn mul(self, o: Expr<C>) -> Expr<C> {
        match (self, o) {
            (Expr::Prod(mut a), Expr::Prod(b)) => {
                a.extend(b);
                Expr::Prod(a)
            }
            (Expr::Prod(mut a), b) => {
                a.push(b);
                Expr::Prod(a)
            }
            (a, b) => Expr::Prod(vec![a, b]),
        }
    }
}
impl<C: Coeff> Neg for Expr<C> {
    type Output = Expr<C>;
    fn neg(self) -> Expr<C> {
        Expr::Neg(Box::new(self))
    }
}
impl<C: Coeff> Add<Polynomial<C>> for Expr<C> {
    type Output = Expr<C>;
    fn add(self, p: Polynomial<C>) -> Expr<C> {
        self + Expr::Poly(p)
    }
}
impl<C: Coeff> Sub<Polynomial<C>> for Expr<C> {
    type Output = Expr<C>;
    fn sub(self, p: Polynomial<C>) -> Expr<C> {
        self - Expr::Poly(p)
    }
}
impl<C: Coeff> Mul<Polynomial<C>> for Expr<C> {
    type Output = Expr<C>;
    fn mul(self, p: Polynomial<C>) -> Expr<C> {
        self * Expr::Poly(p)
    }
}
