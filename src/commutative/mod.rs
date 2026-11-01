//! Commutative polynomials: Laurent polynomials in one variable
//! ([`Laurent`]) and polynomials in several commuting variables ([`MPoly`]),
//! over any [`Coeff`](crate::Coeff) ring.
//!
//! The rest of the crate works in free (noncommutative) algebras, where a
//! commutative ring is a quotient ([`relations`](crate::relations) has the
//! commutation relations).  These types are direct representations for when
//! the variables commute anyway: knot polynomials, generating functions,
//! closed forms in integer parameters.

pub mod laurent;
pub mod mpoly;

pub use laurent::Laurent;
pub use mpoly::MPoly;
