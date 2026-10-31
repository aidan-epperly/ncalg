//! Exact arithmetic over finite rings and the integers.
//!
//! * [`int`] — the shared integer and modular helpers (gcd, inverses,
//!   powers, primality).
//! * [`ff`] — finite fields GF(p^k): runtime fields, dense and bit-packed
//!   matrices, incremental GF(2) echelon bases, `Fq<P, K>` coefficients
//!   (`Fp<P> = Fq<P, 1>`).
//! * [`zmod`] — linear algebra over ℤ/N (Howell form) and over ℤ (Hermite,
//!   Smith).
//!
//! Each of these is also available at the crate root (`ncalg::ff`,
//! `ncalg::zmod`).  Finite abelian groups are in [`crate::symmetry::abelian`].

pub mod ff;
pub mod int;
pub mod zmod;
