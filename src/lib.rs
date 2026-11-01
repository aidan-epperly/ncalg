//! # ncalg — noncommutative polynomial algebra in safe Rust
//!
//! Elements of free associative algebras `K<x_1,…,x_n>` with integer,
//! rational, floating point or complex coefficients; noncommutative Gröbner
//! bases (Mora's algorithm) with degree/time/size bounds that return a
//! *partial* basis instead of running forever; and a lazy, heap-based
//! evaluator that reduces modulo a Gröbner basis *while* an expression is
//! being expanded, so unreduced intermediate products are never built.
//!
//! The crate forbids `unsafe` code.
//!
//! ## Quick tour
//!
//! ```
//! use ncalg::{FreeAlgebra, Polynomial, GroebnerBasis, GroebnerOptions};
//! use num_rational::BigRational;
//!
//! // the free algebra Q<x, y> with the deglex ordering x < y
//! let alg = FreeAlgebra::new(["x", "y"]);
//! let x: Polynomial<i64> = alg.gen("x");
//! let y: Polynomial<i64> = alg.gen("y");
//!
//! // ordinary arithmetic; multiplication is noncommutative
//! let p = &x * &y - &y * &x;
//! assert_eq!(p.to_string(), "-y*x + x*y");
//! assert_eq!(alg.parse::<i64>("[x, y]").unwrap(), p);
//!
//! // Gröbner basis of the Weyl algebra relation yx - xy = 1, over Q
//! let rel: Polynomial<BigRational> = alg.parse("y*x - x*y - 1").unwrap();
//! let gb = GroebnerBasis::compute(vec![rel], &GroebnerOptions::default());
//! assert!(gb.is_complete());
//!
//! // normal forms: reduce while expanding (y^2 x^2 never materializes unreduced)
//! let f: Polynomial<BigRational> = alg.parse("y^2").unwrap();
//! let g: Polynomial<BigRational> = alg.parse("x^2").unwrap();
//! let nf = gb.reduce_product(&f, &g);
//! assert_eq!(nf.to_string(), "x^2*y^2 + 4*x*y + 2");
//! ```
//!
//! ## Module overview
//!
//! * [`coeff`] — coefficient traits ([`Coeff`], [`Field`]) and implementations.
//! * [`word`], [`order`] — monomials and admissible monomial orderings.
//! * [`commutative`] — Laurent polynomials and polynomials in commuting
//!   variables ([`commutative::Laurent`], [`commutative::MPoly`]).
//! * [`algebra`], [`poly`] — [`FreeAlgebra`] and [`Polynomial`].
//! * [`reduce`] — the lazy [`Accumulator`] / [`Reducer`] machinery.
//! * [`groebner`] — [`GroebnerBasis`], [`GroebnerOptions`], the incremental
//!   [`GroebnerSolver`].
//! * [`quotient`] — arithmetic in `K<X>/I`, normal words, Hilbert series.
//! * [`expr`] — expression trees evaluated modulo an ideal.
//! * [`ncsos`] — NC-SOS / NPA moment relaxations (SDP export) over a quotient algebra.
//! * [`linalg`] — sparse storage and the one sparse echelon ([`linalg::sparse`]),
//!   rank-revealing sparse QR ([`sparse_qr`]), crate-internal dense helpers.
//! * [`sdp`] — general SDP modeling (matrix variables, LMIs, equalities,
//!   elimination, SDPA/JSON export) underneath the Gram-matrix forms, the
//!   native solver ([`solver`]) and its facial reduction.
//! * [`io`] — save and load polynomials and Gröbner bases.
//! * [`arith`] — exact arithmetic over finite rings and ℤ: shared integer
//!   helpers ([`arith::int`]), finite fields ([`ff`]), ℤ/N and ℤ linear algebra
//!   ([`zmod`]).
//! * [`symmetry`] — finite groups and their actions: signed permutations and
//!   monomial matrices, group actions modulo an ideal, isotypic
//!   decomposition, p-groups ([`pcgroup`]), finite abelian groups ([`abelian`]).
//! * [`pcgroup`] — finite p-groups by power-conjugate presentations: collection,
//!   subgroups, central series, quotients, Clifford-hierarchy families.
//! * [`relations`] — generators for common algebras (Pauli, Majorana, Weyl,
//!   Lie/PBW, commutative, q-commuting).

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]
#![allow(
    clippy::suspicious_arithmetic_impl,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::while_let_loop
)]

pub mod algebra;
pub mod arith;
pub mod automaton;
pub mod coeff;
pub mod commutative;
pub mod cyclo;
pub mod expr;
pub mod gns;
pub mod groebner;
pub mod io;
pub mod linalg;
pub mod multi;
pub mod ncsos;
pub mod order;
pub mod parse;
pub mod poly;
pub mod quotient;
pub mod rational;
pub mod reduce;
pub mod relations;
pub mod sdp;
pub mod symmetry;
pub mod word;

pub use algebra::FreeAlgebra;
pub use arith::{ff, zmod};
pub use coeff::{set_float_tolerance, Coeff, CoeffConvert, ComplexCoeff, Field, Fp, IntegerCoeff};
pub use expr::Expr;
pub use gns::GnsRepresentation;
pub use groebner::{GroebnerBasis, GroebnerOptions, GroebnerSolver, GroebnerStats, Termination};
pub(crate) use linalg::dense;
pub use linalg::sparse_qr;
pub use order::{DegLex, DegRevLex, Elimination, MonomialOrder, WeightedDegLex, Wreath};
pub use parse::ParseError;
pub use poly::{Polynomial, Term};
pub use quotient::{QuotientAlgebra, QuotientElement};
pub use rational::HybridRational;
pub use reduce::{Accumulator, Reducer, ReductionMode, ReductionStats};
pub use sdp::{solver, Affine, SdpModel};
pub use solver::{SdpSolution, SolveProfile, SolverOptions};
pub use symmetry::{abelian, pcgroup};
pub use word::{Var, Word};

// re-export the numeric types users need for coefficients
pub use num_bigint::BigInt;
pub use num_complex::Complex;
pub use num_rational::BigRational;
