//! Finite groups and their actions.
//!
//! * [`perm`] — signed permutations of algebra generators, monomial matrices
//!   with root-of-unity phases, and the one group-closure routine.
//! * [`action`] — signed permutations acting on polynomials modulo an ideal:
//!   images of words, the Reynolds average, invariance tests, word orbits.
//! * `isotypic` — numerical isotypic decomposition of a representation (the
//!   block-diagonalization behind symmetry-reduced relaxations; crate-internal).
//! * [`pcgroup`] — finite p-groups by power-conjugate presentations.
//! * [`abelian`] — finite abelian groups: subgroups, quotients, duality,
//!   Fourier transform.
//!
//! `pcgroup` and `abelian` are also available at the crate root
//! (`ncalg::pcgroup`, `ncalg::abelian`), and the signed-permutation API at
//! its old place in [`crate::ncsos`].

pub mod abelian;
pub mod action;
pub(crate) mod isotypic;
pub mod pcgroup;
pub mod perm;
