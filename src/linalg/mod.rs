//! Linear algebra used across ncalg.
//!
//! * `dense` (crate-internal) — dense products that avoid faer's parallel
//!   GEMM and Hermitian eigendecompositions.
//! * [`sparse`] — sparse Hermitian/symmetric storage, the one sparse
//!   echelon (incremental Gaussian elimination with partial pivoting) and
//!   reduced row echelon form built on it.
//! * [`sparse_qr`] — rank-revealing sparse QR (also at `ncalg::sparse_qr`).
//! * [`exact`] — dense row reduction, rank, null space, solve, inverse and
//!   (Bareiss) determinant over any exact [`Field`](crate::Field), such as
//!   rationals or cyclotomic numbers.

pub(crate) mod dense;
pub mod exact;
pub mod sparse;
pub mod sparse_qr;
