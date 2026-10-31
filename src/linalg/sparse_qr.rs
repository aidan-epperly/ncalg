//! Rank-revealing sparse QR factorization for the elimination of linear
//! (moment) constraints: the numerical rank, a pivot order, the triangular
//! factor `R` and, from it, a sparse basis of the near-kernel of a sparse real
//! matrix.
//!
//! ## The problem
//!
//! The constraint rows `C ∈ R^{p×m}` of a moment relaxation carry
//! floating-point coefficients (products of trigonometric values and the
//! like).  Taken as exact rationals such a matrix is typically *exactly*
//! inconsistent with the normalization `z_1 = 1`, although it is consistent to
//! about `1e-10`: the direction the relaxation needs lies in the near-kernel
//! `K_δ = span{v_i : σ_i ≤ δ‖C‖}` and not in the exact kernel.  Gaussian
//! elimination ([`crate::linalg::sparse::rref_sparse`]) cannot find `K_δ` reliably,
//! because the size of its oblique remainder is unrelated to the distance of a
//! row from the span of the earlier rows when those are nearly dependent.  A
//! QR factorization only uses orthogonal transformations, is backward stable,
//! and its rank decisions refer to Euclidean quantities.
//!
//! faer's sparse QR (`faer::sparse::linalg::qr`) is not used: it is a
//! supernodal Householder QR with a fill-reducing column ordering fixed in
//! advance and no rank detection, so it neither reports the numerical rank
//! nor gives a basis of `K_δ`.
//!
//! ## Method: row-sequential sparse Givens QR with pivot deferral
//!
//! The implementation (method (A) below) is a George–Heath row-sequential
//! Givens QR with Heath's rank detection.  [`SparseQr::factor`]
//!
//! 1. scales every row to unit 2-norm (this does not change the kernel; the
//!    stored `R` is the factor of the scaled matrix `C̃`);
//! 2. fixes a *column preference order*: ascending column count, ties broken
//!    by **descending** column index (the densest columns, and among equally
//!    dense columns the low-index ones — in `ncalg` the low-degree moments and
//!    the normalization `L(1)` — are pivoted last);
//! 3. orders the rows by the position of their leftmost entry in that column
//!    order, then by number of entries, then by index (the classical
//!    George–Heath row order);
//! 4. processes the rows one by one.  The incoming row `x` is rotated against
//!    the rows of `R` whose pivot columns it touches, in increasing *pivot
//!    index* (not column index), so that `R` stays upper triangular in pivot
//!    order: rotating with row `k` can only bring in entries in columns that
//!    are not pivots yet or are pivots of later rows.  A Givens rotation of the
//!    pair (row `k` of `R`, `x`) zeroes the pivot entry of `x`; entries below
//!    `1e-15` in absolute value are dropped after each rotation (rows are
//!    unit-order);
//! 5. when `x` has no entries in pivot columns any more, its 2-norm is the
//!    *remainder* norm.  Below `rank_tol` the row is dependent and discarded
//!    (it is listed in [`SparseQr::dependent_rows`]).  Otherwise it becomes a
//!    new row of `R`, and its pivot is chosen by *threshold pivoting*: among
//!    the entries with `|x_j| ≥ 0.1 · max_i |x_i|` the one earliest in the
//!    column preference order.
//!
//! [`SparseQr::kernel_basis`] then performs one backward pass over `R` shared
//! by all free columns, exactly the way a reduced row echelon form is
//! produced: with `R = [R11 R12]` (pivot columns in pivot order, free
//! columns), the rows of `T = R11⁻¹ R12` are formed from the last to the first
//! as sparse combinations of the rows below, and the kernel vector of the
//! free column `f` is `e_f − Π T e_f`.  [`SparseQr::kernel_vector`] and
//! [`SparseQr::verify`] run the same pass restricted to one or a few free
//! columns, so all three give bitwise identical vectors.
//!
//! ### Why (A) and not (B), Householder QR with column pivoting
//!
//! * The constraint matrices are extremely sparse (on the `elim_large` fixture
//!   at most 5 entries per row) and very rank deficient (rank 18 425 of
//!   31 744 columns).  The row-sequential scheme follows the data flow of the
//!   elimination it replaces and keeps `R` as sparse as a static fill-reducing
//!   order allows; full column pivoting by remaining column norm, as in (B),
//!   destroys any fill-reducing order and needs the whole trailing matrix.
//! * The rank decisions of (A) cannot produce bad kernel vectors, only a too
//!   small or too large kernel (see *Guarantee*).
//!
//! ### Guarantee
//!
//! Givens rotations preserve the Gram matrix of the pair they act on, so at
//! the end (up to round-off of order `u‖C̃‖`)
//!
//! ```text
//!     C̃ᵀ C̃ = Rᵀ R + Σ_d x_dᵀ x_d,      x_d the discarded remainders.
//! ```
//!
//! Every kernel vector `k` satisfies `R k = 0` (exactly, for the stored `R`, up
//! to the `1e-14` drop), hence `‖C̃ k‖² = Σ_d (x_d · k)²` and
//!
//! ```text
//!     ‖C̃ k‖₂ ≤ η ‖k‖₂,     η = (Σ_d ‖x_d‖²)^{1/2} ≤ rank_tol · √(#dependent rows)
//! ```
//!
//! ([`SparseQr::discarded_norm`] returns `η`).  For the unscaled matrix
//! `C = D C̃` (`D` the row norms) this gives
//! `‖C k‖₂ / (‖C‖_F ‖k‖₂) ≤ η · max_i d_i / ‖C‖_F ≤ η`.  So the vectors returned are
//! always in the near-kernel at level `η`, whatever the rank decisions were.
//! What rank detection by remainder norms does *not* guarantee is that every
//! direction of `K_δ` is found: a row whose remainder is above `rank_tol`
//! although the row is within `δ` of the span of the earlier rows makes the
//! rank too high and the kernel too small (Heath's heuristic: the remainder is
//! the volume increment `γ (x − y R)` with `γ = (1 + ‖y‖²)^{-1/2}`, not the
//! distance of `x` from the row space; the two agree up to the condition of
//! `R11`).  [`SparseQr::verify`] is the routine to run after every
//! factorization; it measures the residuals against the *unscaled* input.
//!
//! ## Known limitations
//!
//! * **Ill-conditioned triangular factors (Kahan).**  An unpivoted QR sees only
//!   the volume increments.  On the `n = 30` Kahan matrix (`θ = 1.2`, already
//!   upper triangular) the smallest diagonal of `R` is `0.456` while
//!   `σ_min = 3.1e-5` (`2.06e-4` for the row-scaled matrix that is actually
//!   factored, whose next singular value is `0.547`): `min_diagonal / σ_min`
//!   is `2.2e3` (row-scaled; `1.5e4` against the unscaled `σ_min`).  At
//!   `rank_tol = 1e-6` the reported rank 30 is correct (σ_min is above the
//!   tolerance), but at any tolerance between `2.1e-4` and `0.45` (the test
//!   uses `1.06e-2`) the SVD rank is 29 and the factorization still reports
//!   30: the kernel is too small.  Method (B) or an incremental condition
//!   estimator would be needed to detect such dependences.  (The residual
//!   guarantee is unaffected: the vectors returned are still near-kernel
//!   vectors; one is missing.)
//! * **Which columns are free** depends on the pivot rule, not only on the
//!   rank: a column `c` can be free only if `e_c` is not in the row space of
//!   `R`, but a pivot rule could also pick `c` when it need not.  The column
//!   preference order and the threshold `0.1` keep the dense low-index columns
//!   free.  Measured on `elim_large` (rank 18 425 in every case): with the
//!   textbook rule (pivot = largest entry, threshold 1) column 1
//!   becomes a pivot and `R` has 13.0 million nonzeros (114 s); with threshold
//!   `0.5` column 1 is also pivoted (2.2 million, 13 s); with `0.1` (the value
//!   used, as in UMFPACK) and `0.01` it stays free (0.82 and 0.74 million,
//!   under 4 s).  The remainders that decide this are rows like
//!   `0.894 z_1 + 0.447 z_j`, where pivoting on column 1 is a pure artefact.
//! * **Monotonicity in `rank_tol`** is not a theorem for this method (discarding
//!   a row changes all later remainders).  It holds on the tests and on both
//!   fixtures over `rank_tol ∈ [1e-14, 1e-4]`.
//! * The diagonal of `R` starts as the chosen pivot entry, which is at least
//!   `0.1 · ‖x‖₂ / √nnz(x) ≥ 0.1 · rank_tol / √nnz(x)`, and only grows under
//!   later rotations.  It is therefore not guaranteed to be `≥ rank_tol` in
//!   general; on both fixtures the smallest diagonal is above `3e-5`.
//! * `dead_columns` has a weaker meaning than in a column-pivoted method:
//!   method (A) defers *rows*.  A dead entry is the column on which a discarded
//!   (nonzero) remainder would have pivoted, kept only if that column is free at
//!   the end (first occurrence, with that remainder's norm).
//!
//! ## Design choices
//!
//! All measured on `elim_large` at `rank_tol = 1e-9`; the rank was 18 425 in
//! every variant, only fill, time and the status of column 1 changed.
//!
//! * **Row order.**  Rows are ordered by their leftmost column in the column
//!   preference order first (George–Heath), then by number of entries.  The
//!   naive order (number of entries, then smallest column index) gave
//!   `r_nnz = 26.4e6` and 124 s; number of entries first and then leftmost
//!   column in the preference order gave `14.8e6` and 77 s; leftmost column
//!   first gives `1.17e6` and 5.7 s
//!   with ascending-index ties in the column order and `0.82e6` and 3.8 s with
//!   descending-index ties (the variant used).
//! * **Pivot rule.**  Threshold pivoting with a column preference instead of
//!   "largest entry, ties smallest index", see the limitation on free columns
//!   above (the textbook rule pivots column 1 and gives `13e6` nonzeros).
//! * **Column order.**  Ascending column count.  An
//!   approximate-minimum-degree order on the column intersection graph
//!   (quotient-graph AMD with rows as initial elements) was tried: `3.9e6` nonzeros, 22 s, column 1 pivoted; it was not kept.
//! * [`SparseQr::r_row`] returns the diagonal first and the other entries
//!   sorted by column: a row of `R` is triangular in pivot order, not in column
//!   order, so it cannot be sorted by column with the diagonal first.
//! * `dead_columns` has the method-(A) meaning described above.
//! * Additional diagnostics:
//!   [`SparseQr::dependent_rows`], [`SparseQr::remainder_norms`],
//!   [`SparseQr::discarded_norm`].
//!
//! ## Example
//!
//! ```
//! use ncalg::sparse_qr::{QrOptions, SparseQr, SparseRows};
//! // z0 − z1 = 0, z1 − z2 = 0, z0 − z2 = 0 (dependent): kernel spanned by (1, 1, 1)
//! let a = SparseRows {
//!     ncols: 3,
//!     rows: vec![vec![(0, 1.0), (1, -1.0)], vec![(1, 1.0), (2, -1.0)], vec![(0, 1.0), (2, -1.0)]],
//! };
//! let opts = QrOptions::default();
//! let qr = SparseQr::factor(&a, &opts).unwrap();
//! assert_eq!(qr.rank(), 2);
//! assert_eq!(qr.free_columns(), &[0]);
//! let k = qr.kernel_vector(0);
//! assert_eq!(k.len(), 3);
//! assert!(k.iter().all(|&(_, v)| (v - 1.0).abs() < 1e-14));
//! let check = qr.verify(&a, 0);
//! assert!(check.max_relative_residual < 1e-15);
//! ```
//!
//! ## Measured on the fixtures (`fixtures/constraint_elimination/`)
//!
//! The small and large constraint-elimination fixtures, `elim_small` and
//! `elim_large`, are redundant equality systems from moment relaxations.
//! Release build, one thread, `rank_tol = 1e-9`, desktop CPU
//! (`cargo test --release --test sparse_qr_tests -- --nocapture`):
//!
//! | fixture      | size            | rank   | `r_nnz` | factor | kernel dim | kernel nnz | kernel basis |
//! |--------------|-----------------|--------|---------|--------|------------|------------|--------------|
//! | `elim_small` | 4 966 × 12 637  | 4 261  | 40 722  | 0.02 s | 8 376      | 15 957     | < 0.01 s     |
//! | `elim_large` | 22 716 × 31 744 | 18 425 | 817 004 | 3.8 s  | 13 319     | 65 584     | 0.01 s       |
//!
//! Peak working set of the `elim_large` test process: 52 MB.  On both fixtures
//! column 1 (index 0, the normalization `L(1)`) is free and its kernel vector
//! has relative residual `‖C k‖ / (‖C‖_F ‖k‖)` of `2.0e-17` (`elim_small`) and
//! `2.8e-17` (`elim_large`); `verify(.., 200)` reports `2.0e-17` and `3.2e-17`, and
//! the diagonal of `R` lies in `[3.2e-5, 3.3]` and `[4.2e-5, 4.8]`.  The
//! largest discarded remainder is `1.2e-14` (`elim_small`, 705 dependent rows) and
//! `1.0e-10` (`elim_large`, 4 291 dependent rows, `η = 1.2e-10`): the latter is the
//! `1e-10` inconsistency of the data, correctly treated as noise.  The
//! smallest accepted remainder is `1.8e-5` (`elim_small`) and `2.7e-9` (`elim_large`), yet
//! the rank is the same (18 425, resp. 4 261, column 1 free) for every
//! `rank_tol` from `1e-13` to `1e-5`: when a row with a small remainder is
//! discarded a later row takes over its pivot.  At `1e-4` the ranks drop to
//! 18 410 and 4 255; at `1e-14` the factorization starts to accept round-off
//! (ranks 18 546 and 4 262, column 1 pivoted, i.e. the exact-arithmetic
//! behavior).  The exact rational ranks are 18 617 and 4 270.  In a debug
//! build `factor` takes about 29 s on `elim_large`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;

/// Entries of magnitude below this are dropped after every rotation.
const DROP_TOL: f64 = 1e-15;
/// Entries of kernel vectors (and of `R11⁻¹ R12`) below this are dropped.
const KERNEL_DROP_TOL: f64 = 1e-14;
/// Threshold pivoting: the pivot of a new row of `R` is the entry earliest in
/// the column preference order among those with `|x_j| ≥ PIVOT_THRESHOLD · max |x|`.
const PIVOT_THRESHOLD: f64 = 0.1;
/// Marker for "not a pivot column".
const NONE: usize = usize::MAX;

/// Sparse matrix in row form.  Rows are `(column, value)` lists, not
/// necessarily sorted, without duplicate columns.  `ncols` is the number of
/// columns (columns may be empty).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseRows {
    /// Number of columns.
    pub ncols: usize,
    /// The rows, as `(column, value)` lists.
    pub rows: Vec<Vec<(usize, f64)>>,
}

impl SparseRows {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.rows.iter().map(Vec::len).sum()
    }

    /// Read a Matrix Market "coordinate real general" file (1-based indices;
    /// `%` comment lines; "integer" fields are accepted too).  Entries keep the
    /// order of the file within each row; a repeated `(row, column)` pair is an
    /// error.
    pub fn read_matrix_market(path: impl AsRef<Path>) -> std::io::Result<SparseRows> {
        fn invalid(msg: String) -> std::io::Error {
            std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
        }
        fn field<T: std::str::FromStr>(it: Option<&str>, line: &str) -> std::io::Result<T>
        where
            T::Err: fmt::Display,
        {
            let s = it.ok_or_else(|| invalid(format!("missing field in line: {line}")))?;
            s.parse::<T>()
                .map_err(|e| invalid(format!("bad field {s:?} in line {line:?}: {e}")))
        }
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut lines = reader.lines();
        let banner = lines.next().ok_or_else(|| invalid("empty file".into()))??;
        let lower = banner.to_ascii_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        if words.len() < 5
            || words[0] != "%%matrixmarket"
            || words[1] != "matrix"
            || words[2] != "coordinate"
            || !(words[3] == "real" || words[3] == "integer")
            || words[4] != "general"
        {
            return Err(invalid(format!("unsupported Matrix Market banner: {banner}")));
        }
        let mut size: Option<(usize, usize, usize)> = None;
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut count = 0usize;
        for line in lines {
            let line = line?;
            let t = line.trim();
            if t.is_empty() || t.starts_with('%') {
                continue;
            }
            let mut it = t.split_whitespace();
            match size {
                None => {
                    let m: usize = field(it.next(), t)?;
                    let n: usize = field(it.next(), t)?;
                    let nnz: usize = field(it.next(), t)?;
                    size = Some((m, n, nnz));
                    rows = vec![Vec::new(); m];
                }
                Some((m, n, _)) => {
                    let i: usize = field(it.next(), t)?;
                    let j: usize = field(it.next(), t)?;
                    let v: f64 = field(it.next(), t)?;
                    if i == 0 || j == 0 || i > m || j > n {
                        return Err(invalid(format!("entry out of range ({m} × {n}): {t}")));
                    }
                    rows[i - 1].push((j - 1, v));
                    count += 1;
                }
            }
        }
        let (_, n, nnz) = size.ok_or_else(|| invalid("missing size line".into()))?;
        if count != nnz {
            return Err(invalid(format!(
                "size line announces {nnz} entries, file has {count}"
            )));
        }
        let a = SparseRows { ncols: n, rows };
        if let Some(msg) = duplicate_column(&a) {
            return Err(invalid(msg));
        }
        Ok(a)
    }

    /// Write the matrix as a Matrix Market "coordinate real general" file
    /// (1-based indices, values with 17 significant digits so that reading
    /// the file back gives the same `f64`s).
    pub fn write_matrix_market(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        writeln!(w, "%%MatrixMarket matrix coordinate real general")?;
        writeln!(w, "% written by ncalg::sparse_qr")?;
        writeln!(w, "{} {} {}", self.nrows(), self.ncols, self.nnz())?;
        for (i, row) in self.rows.iter().enumerate() {
            for &(j, v) in row {
                writeln!(w, "{} {} {:.17e}", i + 1, j + 1, v)?;
            }
        }
        w.flush()
    }
}

/// Options of [`SparseQr::factor`] and [`SparseQr::kernel_basis`].
#[derive(Clone, Debug)]
pub struct QrOptions {
    /// Rows are scaled to unit 2-norm before factoring (this does not change
    /// the kernel).  A pivot candidate (the remainder of an incoming row after
    /// all its pivot-column entries were rotated away) whose 2-norm is below
    /// `rank_tol` is dead: the row is discarded as dependent and the column
    /// it would have pivoted on stays free unless a later row claims it.
    /// Default 1e-9.
    pub rank_tol: f64,
    /// Give up (`Err`) when the number of stored nonzeros of `R` would exceed
    /// this.  Default 50_000_000.
    pub max_r_nnz: usize,
    /// Give up (`Err`) when the number of stored nonzeros of the kernel basis
    /// would exceed this (checked in [`SparseQr::kernel_basis`]).  Default
    /// 50_000_000.
    pub max_kernel_nnz: usize,
}

impl Default for QrOptions {
    fn default() -> Self {
        QrOptions {
            rank_tol: 1e-9,
            max_r_nnz: 50_000_000,
            max_kernel_nnz: 50_000_000,
        }
    }
}

/// Errors of the factorization.
#[derive(Clone, Debug, PartialEq)]
pub enum QrError {
    /// Dimension mismatch, duplicate column in a row, non-finite entry,
    /// invalid tolerance.
    InvalidInput(String),
    /// Fill limit exceeded; the message says which and gives the counts reached.
    TooMuchFill(String),
}

impl fmt::Display for QrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QrError::InvalidInput(m) => write!(f, "invalid input: {m}"),
            QrError::TooMuchFill(m) => write!(f, "too much fill: {m}"),
        }
    }
}

impl std::error::Error for QrError {}

/// Result of [`SparseQr::verify`].
#[derive(Clone, Debug)]
pub struct Verification {
    /// Largest `‖A k‖₂ / (‖A‖_F ‖k‖₂)` over the sampled kernel vectors `k`
    /// (`A` unscaled, as given to [`SparseQr::factor`]); 0 when there is none.
    pub max_relative_residual: f64,
    /// Smallest `|R_kk|` (0 when the rank is 0).
    pub min_diagonal: f64,
    /// Largest `|R_kk|` (0 when the rank is 0).
    pub max_diagonal: f64,
    /// Numerical rank.
    pub rank: usize,
    /// Number of free columns (`ncols − rank`).
    pub kernel_dimension: usize,
}

/// `R` factor of a row-scaled `C Π = Q R` with rank detection (see the
/// [module documentation](self)).  `Q` is never formed.
#[derive(Clone)]
pub struct SparseQr {
    ncols: usize,
    /// Rows of `R` in pivot order: diagonal first, then the other entries
    /// sorted by column.
    rows: Vec<Vec<(usize, f64)>>,
    pivots: Vec<usize>,
    free: Vec<usize>,
    diag: Vec<f64>,
    /// Norm of the remainder that created each row of `R`, in pivot order.
    remainder: Vec<f64>,
    dead: Vec<(usize, f64)>,
    dependent: Vec<(usize, f64)>,
    /// Column → pivot index, or `NONE`.
    pos_of: Vec<usize>,
    r_nnz: usize,
}

impl fmt::Debug for SparseQr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseQr")
            .field("ncols", &self.ncols)
            .field("rank", &self.rank())
            .field("r_nnz", &self.r_nnz)
            .field("dependent_rows", &self.dependent.len())
            .field("discarded_norm", &self.discarded_norm())
            .finish_non_exhaustive()
    }
}

/// Working row of the factorization: values scattered into a dense array of
/// length `ncols` (a vector, not a matrix) with an explicit pattern list.
struct WorkRow {
    val: Vec<f64>,
    present: Vec<bool>,
    pattern: Vec<usize>,
}

impl WorkRow {
    fn new(n: usize) -> Self {
        WorkRow {
            val: vec![0.0; n],
            present: vec![false; n],
            pattern: Vec::new(),
        }
    }

    fn insert(&mut self, j: usize, v: f64) {
        self.val[j] = v;
        if !self.present[j] {
            self.present[j] = true;
            self.pattern.push(j);
        }
    }

    /// Remove an entry; the pattern list is cleaned by [`WorkRow::compact`].
    fn remove(&mut self, j: usize) {
        self.val[j] = 0.0;
        self.present[j] = false;
    }

    fn compact(&mut self) {
        let present = &self.present;
        self.pattern.retain(|&j| present[j]);
    }

    fn clear(&mut self) {
        for &j in &self.pattern {
            self.val[j] = 0.0;
            self.present[j] = false;
        }
        self.pattern.clear();
    }

    /// Entry earliest in the column preference order among those with
    /// magnitude at least `PIVOT_THRESHOLD · maxabs`.  The pattern must be
    /// compact and nonempty.
    fn choose_pivot(&self, maxabs: f64, prio: &[usize]) -> usize {
        let thr = PIVOT_THRESHOLD * maxabs;
        let mut best = NONE;
        for &j in &self.pattern {
            if self.val[j].abs() >= thr && (best == NONE || prio[j] < prio[best]) {
                best = j;
            }
        }
        best
    }
}

impl SparseQr {
    /// Factor `C̃ Π = Q R` (`C̃` = `a` with rows scaled to unit 2-norm) with
    /// rank detection at `opts.rank_tol` (see the [module documentation](self)
    /// for the method).  Deterministic: no randomness, single-threaded.
    ///
    /// Errors: [`QrError::InvalidInput`] for a column index `≥ ncols`, a
    /// duplicate column in a row, a non-finite entry or a negative or
    /// non-finite `rank_tol`; [`QrError::TooMuchFill`] when `R` would exceed
    /// `opts.max_r_nnz` stored nonzeros.
    pub fn factor(a: &SparseRows, opts: &QrOptions) -> Result<SparseQr, QrError> {
        validate(a, opts)?;
        let n = a.ncols;

        // column preference: ascending count, ties by descending index
        let mut count = vec![0usize; n];
        for row in &a.rows {
            for &(j, v) in row {
                if v != 0.0 {
                    count[j] += 1;
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&j| (count[j], Reverse(j)));
        let mut prio = vec![0usize; n];
        for (p, &j) in order.iter().enumerate() {
            prio[j] = p;
        }

        // row order: leftmost entry in the column order, number of entries, index
        let mut row_order: Vec<(usize, usize, usize)> = a
            .rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let nz = r.iter().filter(|e| e.1 != 0.0).count();
                let left = r
                    .iter()
                    .filter(|e| e.1 != 0.0)
                    .map(|e| prio[e.0])
                    .min()
                    .unwrap_or(NONE);
                (left, nz, i)
            })
            .collect();
        row_order.sort_unstable();

        let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        let mut remainder: Vec<f64> = Vec::new();
        let mut pos_of = vec![NONE; n];
        let mut dead_candidates: Vec<(usize, f64)> = Vec::new();
        let mut dependent: Vec<(usize, f64)> = Vec::new();
        let mut x = WorkRow::new(n);
        let mut mark = vec![0u32; n];
        let mut stamp = 0u32;
        let mut heap: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        let mut r_nnz = 0usize;

        for (processed, &(_, nz, i)) in row_order.iter().enumerate() {
            if nz == 0 {
                dependent.push((i, 0.0));
                continue;
            }
            let row = &a.rows[i];
            let scale = row.iter().fold(0.0f64, |m, e| m.max(e.1.abs()));
            let norm = scale
                * row
                    .iter()
                    .map(|e| (e.1 / scale) * (e.1 / scale))
                    .sum::<f64>()
                    .sqrt();
            for &(j, v) in row {
                if v != 0.0 {
                    x.insert(j, v / norm);
                    if pos_of[j] != NONE {
                        heap.push(Reverse(pos_of[j]));
                    }
                }
            }
            // rotate away the pivot-column entries in increasing pivot index
            while let Some(Reverse(k)) = heap.pop() {
                let p = pivots[k];
                if !x.present[p] {
                    continue; // duplicate heap entry, or dropped
                }
                let xp = x.val[p];
                if xp.abs() < DROP_TOL {
                    x.remove(p);
                    continue;
                }
                let rk = std::mem::take(&mut rows[k]);
                let d = rk[0].1;
                let r = d.hypot(xp);
                let (c, s) = (d / r, xp / r);
                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    mark.iter_mut().for_each(|m| *m = 0);
                    stamp = 1;
                }
                // new R row = c·rk + s·x, new x = c·x − s·rk on the union pattern
                let mut new_rk: Vec<(usize, f64)> = Vec::with_capacity(rk.len() + x.pattern.len());
                for &(j, rv) in &rk {
                    mark[j] = stamp;
                    if j == p {
                        new_rk.push((p, r));
                        x.remove(p);
                        continue;
                    }
                    let (nr, nx) = if x.present[j] {
                        let xv = x.val[j];
                        (c * rv + s * xv, c * xv - s * rv)
                    } else {
                        x.insert(j, 0.0);
                        if pos_of[j] != NONE {
                            heap.push(Reverse(pos_of[j]));
                        }
                        (c * rv, -s * rv)
                    };
                    if nr.abs() >= DROP_TOL {
                        new_rk.push((j, nr));
                    }
                    if nx.abs() >= DROP_TOL {
                        x.val[j] = nx;
                    } else {
                        x.remove(j);
                    }
                }
                for idx in 0..x.pattern.len() {
                    let j = x.pattern[idx];
                    if !x.present[j] || mark[j] == stamp {
                        continue;
                    }
                    let xv = x.val[j];
                    let (nr, nx) = (s * xv, c * xv);
                    if nr.abs() >= DROP_TOL {
                        new_rk.push((j, nr));
                    }
                    if nx.abs() >= DROP_TOL {
                        x.val[j] = nx;
                    } else {
                        x.remove(j);
                    }
                }
                x.compact();
                r_nnz = r_nnz + new_rk.len() - rk.len();
                rows[k] = new_rk;
                if r_nnz > opts.max_r_nnz {
                    return Err(fill_error(
                        opts.max_r_nnz,
                        r_nnz,
                        processed,
                        a.nrows(),
                        pivots.len(),
                    ));
                }
            }
            x.compact();
            let mut sq = 0.0;
            let mut maxabs = 0.0f64;
            for &j in &x.pattern {
                sq += x.val[j] * x.val[j];
                maxabs = maxabs.max(x.val[j].abs());
            }
            let rnorm = sq.sqrt();
            if x.pattern.is_empty() || rnorm < opts.rank_tol {
                if !x.pattern.is_empty() {
                    dead_candidates.push((x.choose_pivot(maxabs, &prio), rnorm));
                }
                dependent.push((i, rnorm));
                x.clear();
                continue;
            }
            let pc = x.choose_pivot(maxabs, &prio);
            let mut new_row = Vec::with_capacity(x.pattern.len());
            new_row.push((pc, x.val[pc]));
            new_row.extend(x.pattern.iter().filter(|&&j| j != pc).map(|&j| (j, x.val[j])));
            r_nnz += new_row.len();
            pos_of[pc] = pivots.len();
            pivots.push(pc);
            remainder.push(rnorm);
            rows.push(new_row);
            x.clear();
            if r_nnz > opts.max_r_nnz {
                return Err(fill_error(
                    opts.max_r_nnz,
                    r_nnz,
                    processed + 1,
                    a.nrows(),
                    pivots.len(),
                ));
            }
        }

        for row in rows.iter_mut() {
            row[1..].sort_unstable_by_key(|e| e.0);
        }
        let diag: Vec<f64> = rows.iter().map(|r| r[0].1.abs()).collect();
        let free: Vec<usize> = (0..n).filter(|&j| pos_of[j] == NONE).collect();
        let mut seen = vec![false; n];
        let mut dead = Vec::new();
        for (c, norm) in dead_candidates {
            if pos_of[c] == NONE && !seen[c] {
                seen[c] = true;
                dead.push((c, norm));
            }
        }
        Ok(SparseQr {
            ncols: n,
            rows,
            pivots,
            free,
            diag,
            remainder,
            dead,
            dependent,
            pos_of,
            r_nnz,
        })
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Numerical rank `r`: number of live pivot columns.
    pub fn rank(&self) -> usize {
        self.pivots.len()
    }

    /// The `r` pivot columns in pivot order: row `k` of `R` has its leading
    /// (diagonal) entry in `pivot_columns()[k]`.
    pub fn pivot_columns(&self) -> &[usize] {
        &self.pivots
    }

    /// All other columns (dead and never pivoted), ascending.
    pub fn free_columns(&self) -> &[usize] {
        &self.free
    }

    /// `|R_kk|` in pivot order.  See the module documentation for the lower
    /// bound (`≥ 0.1 · rank_tol / √nnz` in general, in practice far larger).
    pub fn diagonal(&self) -> &[f64] {
        &self.diag
    }

    /// For method (A): the columns on which a discarded (dead, nonzero)
    /// remainder would have pivoted and that are free at the end, each with
    /// the norm of the first remainder that condemned it.  Diagnostic only;
    /// [`free_columns`](Self::free_columns) is authoritative.
    pub fn dead_columns(&self) -> &[(usize, f64)] {
        &self.dead
    }

    /// Rows discarded as dependent: `(row index in the input, remainder norm)`
    /// in processing order (empty rows have norm 0).
    pub fn dependent_rows(&self) -> &[(usize, f64)] {
        &self.dependent
    }

    /// The norm of the remainder that created each row of `R`, in pivot order
    /// (all `≥ rank_tol`).  The smallest one shows how close the rank
    /// decisions came to the tolerance.
    pub fn remainder_norms(&self) -> &[f64] {
        &self.remainder
    }

    /// `η = (Σ ‖x_d‖²)^{1/2}` over the discarded remainders: every kernel
    /// vector `k` satisfies `‖C̃ k‖₂ ≤ η ‖k‖₂` up to round-off (`C̃` the
    /// row-scaled matrix), see the module documentation.
    pub fn discarded_norm(&self) -> f64 {
        self.dependent.iter().map(|e| e.1 * e.1).sum::<f64>().sqrt()
    }

    /// Row `k` of `R` as `(column, value)`: the diagonal entry (in
    /// `pivot_columns()[k]`) first, then the other entries sorted by column.
    /// The other entries lie in free columns or in pivot columns of later rows.
    pub fn r_row(&self, k: usize) -> &[(usize, f64)] {
        &self.rows[k]
    }

    /// Number of stored nonzeros of `R`.
    pub fn r_nnz(&self) -> usize {
        self.r_nnz
    }

    /// Kernel vector for the free column `f`: `z_f = 1`,
    /// `z_{pivot k} = −(R11⁻¹ R12 e_f)_k`, other entries 0, in original column
    /// indices, sorted by column.  Entries of `R11⁻¹ R12 e_f` below `1e-14` in
    /// absolute value are set to zero as they are produced (and used as zero in
    /// the rest of the substitution), exactly as in
    /// [`kernel_basis`](Self::kernel_basis), whose vectors are bitwise equal.
    /// Costs one pass over `R`.
    ///
    /// Panics if `free_column` is not a free column.
    pub fn kernel_vector(&self, free_column: usize) -> Vec<(usize, f64)> {
        assert!(
            free_column < self.ncols && self.pos_of[free_column] == NONE,
            "kernel_vector: column {free_column} is not a free column"
        );
        self.kernel_vectors(&[free_column], usize::MAX)
            .expect("no limit")
            .pop()
            .expect("one vector")
    }

    /// All kernel vectors (one per free column, in the order of
    /// [`free_columns`](Self::free_columns)), computed with one backward pass
    /// over `R` shared by all columns.  Errors with [`QrError::TooMuchFill`]
    /// when the total number of stored nonzeros would exceed
    /// `opts.max_kernel_nnz`.
    pub fn kernel_basis(&self, opts: &QrOptions) -> Result<Vec<Vec<(usize, f64)>>, QrError> {
        self.kernel_vectors(&self.free, opts.max_kernel_nnz)
            .map_err(|reached| {
                QrError::TooMuchFill(format!(
                    "kernel basis would exceed {} stored nonzeros (reached {} with {} free columns, rank {})",
                    opts.max_kernel_nnz,
                    reached,
                    self.free.len(),
                    self.rank()
                ))
            })
    }

    /// Kernel vectors of the free columns `cols` by the shared backward pass;
    /// `Err(count)` when more than `limit` nonzeros would be stored.
    fn kernel_vectors(&self, cols: &[usize], limit: usize) -> Result<Vec<Vec<(usize, f64)>>, usize> {
        let mut total = cols.len();
        if total > limit {
            return Err(total);
        }
        let n = self.ncols;
        let mut wanted = vec![false; n];
        for &f in cols {
            wanted[f] = true;
        }
        // t[k] = row k of R11⁻¹ R12 restricted to `cols`, sorted by column
        let r = self.rank();
        let mut t: Vec<Vec<(usize, f64)>> = vec![Vec::new(); r];
        let mut acc = vec![0.0f64; n];
        let mut present = vec![false; n];
        let mut pattern: Vec<usize> = Vec::new();
        for k in (0..r).rev() {
            let row = &self.rows[k];
            for &(c, v) in &row[1..] {
                let pc = self.pos_of[c];
                if pc == NONE {
                    if wanted[c] {
                        if !present[c] {
                            present[c] = true;
                            pattern.push(c);
                        }
                        acc[c] += v;
                    }
                } else {
                    for &(f, tv) in &t[pc] {
                        if !present[f] {
                            present[f] = true;
                            pattern.push(f);
                        }
                        acc[f] -= v * tv;
                    }
                }
            }
            let d = row[0].1;
            pattern.sort_unstable();
            let mut out = Vec::with_capacity(pattern.len());
            for &f in &pattern {
                let value = acc[f] / d;
                if value.abs() >= KERNEL_DROP_TOL {
                    out.push((f, value));
                }
                acc[f] = 0.0;
                present[f] = false;
            }
            pattern.clear();
            total += out.len();
            if total > limit {
                return Err(total);
            }
            t[k] = out;
        }
        // transpose: z = e_f − Σ_k t[k][f] e_{pivot k}
        let mut index = vec![NONE; n];
        for (i, &f) in cols.iter().enumerate() {
            index[f] = i;
        }
        let mut out: Vec<Vec<(usize, f64)>> = cols.iter().map(|&f| vec![(f, 1.0)]).collect();
        for (k, tk) in t.iter().enumerate() {
            let p = self.pivots[k];
            for &(f, v) in tk {
                out[index[f]].push((p, -v));
            }
        }
        for v in out.iter_mut() {
            v.sort_unstable_by_key(|e| e.0);
        }
        Ok(out)
    }

    /// `‖A v‖₂ / (‖A‖_F ‖v‖₂)` for a sparse vector `v` (`A` as given to
    /// [`factor`](Self::factor), unscaled); 0 when `A` or `v` is zero.
    pub fn relative_residual(a: &SparseRows, v: &[(usize, f64)]) -> f64 {
        let mut dense = vec![0.0; a.ncols];
        let mut vn = 0.0;
        for &(j, x) in v {
            dense[j] = x;
            vn += x * x;
        }
        let mut an = 0.0;
        let mut res = 0.0;
        for row in &a.rows {
            let mut s = 0.0;
            for &(j, x) in row {
                an += x * x;
                s += x * dense[j];
            }
            res += s * s;
        }
        if an == 0.0 || vn == 0.0 {
            return 0.0;
        }
        res.sqrt() / (an.sqrt() * vn.sqrt())
    }

    /// The check to run after every factorization: the largest
    /// [`relative_residual`](Self::relative_residual) against the unscaled `a`
    /// over `sample` kernel vectors (all when `sample == 0`), taken
    /// deterministically as every `⌈dim / sample⌉`-th free column, plus the
    /// smallest and largest diagonal of `R`, the rank and the kernel
    /// dimension.  `a` must be the matrix that was factored.  The sampled
    /// vectors are computed by one shared backward pass, so the cost is about
    /// one pass over `R` plus the products with `a`.
    ///
    /// A caller should refuse the kernel when `max_relative_residual` is well
    /// above `rank_tol` (the integration uses `100 · rank_tol`).
    pub fn verify(&self, a: &SparseRows, sample: usize) -> Verification {
        let dim = self.free.len();
        let cols: Vec<usize> = if sample == 0 || sample >= dim {
            self.free.clone()
        } else {
            self.free.iter().copied().step_by(dim.div_ceil(sample)).collect()
        };
        let vectors = self.kernel_vectors(&cols, usize::MAX).expect("no limit");
        // column structure of `a` for the products
        let mut colptr = vec![0usize; a.ncols + 1];
        for row in &a.rows {
            for &(j, _) in row {
                colptr[j + 1] += 1;
            }
        }
        for j in 0..a.ncols {
            colptr[j + 1] += colptr[j];
        }
        let mut next = colptr.clone();
        let mut entries = vec![(0usize, 0.0f64); colptr[a.ncols]];
        for (i, row) in a.rows.iter().enumerate() {
            for &(j, x) in row {
                entries[next[j]] = (i, x);
                next[j] += 1;
            }
        }
        let a_fro = a.rows.iter().flatten().map(|e| e.1 * e.1).sum::<f64>().sqrt();
        let mut y = vec![0.0f64; a.nrows()];
        let mut touched_flag = vec![false; a.nrows()];
        let mut touched: Vec<usize> = Vec::new();
        let mut worst = 0.0f64;
        for v in &vectors {
            let mut vn = 0.0;
            for &(j, x) in v {
                vn += x * x;
                for &(i, aij) in &entries[colptr[j]..colptr[j + 1]] {
                    if !touched_flag[i] {
                        touched_flag[i] = true;
                        touched.push(i);
                    }
                    y[i] += aij * x;
                }
            }
            let mut rs = 0.0;
            for &i in &touched {
                rs += y[i] * y[i];
                y[i] = 0.0;
                touched_flag[i] = false;
            }
            touched.clear();
            if a_fro > 0.0 && vn > 0.0 {
                worst = worst.max(rs.sqrt() / (a_fro * vn.sqrt()));
            }
        }
        let (min_diagonal, max_diagonal) = if self.diag.is_empty() {
            (0.0, 0.0)
        } else {
            self.diag
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(lo, hi), &d| (lo.min(d), hi.max(d)))
        };
        Verification {
            max_relative_residual: worst,
            min_diagonal,
            max_diagonal,
            rank: self.rank(),
            kernel_dimension: dim,
        }
    }
}

fn fill_error(limit: usize, reached: usize, processed: usize, nrows: usize, rank: usize) -> QrError {
    QrError::TooMuchFill(format!(
        "R would exceed {limit} stored nonzeros (reached {reached} after {processed} of {nrows} rows, rank {rank})"
    ))
}

/// First duplicate column found in a row, as an error message.
fn duplicate_column(a: &SparseRows) -> Option<String> {
    let mut seen = vec![NONE; a.ncols];
    for (i, row) in a.rows.iter().enumerate() {
        for &(j, _) in row {
            if j < a.ncols {
                if seen[j] == i {
                    return Some(format!("row {i}: duplicate column {j}"));
                }
                seen[j] = i;
            }
        }
    }
    None
}

fn validate(a: &SparseRows, opts: &QrOptions) -> Result<(), QrError> {
    if !(opts.rank_tol.is_finite() && opts.rank_tol >= 0.0) {
        return Err(QrError::InvalidInput(format!(
            "rank_tol must be finite and ≥ 0, got {}",
            opts.rank_tol
        )));
    }
    for (i, row) in a.rows.iter().enumerate() {
        for &(j, v) in row {
            if j >= a.ncols {
                return Err(QrError::InvalidInput(format!(
                    "row {i}: column {j} out of range (ncols = {})",
                    a.ncols
                )));
            }
            if !v.is_finite() {
                return Err(QrError::InvalidInput(format!(
                    "row {i}, column {j}: non-finite entry {v}"
                )));
            }
        }
    }
    if let Some(msg) = duplicate_column(a) {
        return Err(QrError::InvalidInput(msg));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn dense_to_sparse(m: &Mat<f64>) -> SparseRows {
        let rows = (0..m.nrows())
            .map(|i| {
                (0..m.ncols())
                    .filter(|&j| m[(i, j)] != 0.0)
                    .map(|j| (j, m[(i, j)]))
                    .collect()
            })
            .collect();
        SparseRows {
            ncols: m.ncols(),
            rows,
        }
    }

    fn random_orthonormal(rng: &mut StdRng, n: usize, k: usize) -> Mat<f64> {
        let g = Mat::<f64>::from_fn(n, k, |_, _| rng.gen_range(-1.0..1.0));
        g.qr().compute_thin_Q()
    }

    /// `U diag(s) Vᵀ` with `r` singular values in `[1, 10]` and the others
    /// uniformly in `[0, noise]` (exactly zero when `noise == 0`).
    fn planted(rng: &mut StdRng, p: usize, m: usize, r: usize, noise: f64) -> Mat<f64> {
        let k = p.min(m);
        let u = random_orthonormal(rng, p, k);
        let v = random_orthonormal(rng, m, k);
        let s: Vec<f64> = (0..k)
            .map(|i| {
                if i < r {
                    rng.gen_range(1.0..10.0)
                } else {
                    noise * rng.gen_range(0.0..1.0)
                }
            })
            .collect();
        Mat::<f64>::from_fn(p, m, |i, j| (0..k).map(|l| u[(i, l)] * s[l] * v[(j, l)]).sum())
    }

    /// Orthonormal basis of the null space of the row-normalized `a`, from
    /// faer's SVD (singular values below `1e-9 · σ_max`).
    fn svd_null_space(a: &Mat<f64>) -> Mat<f64> {
        let scaled = Mat::<f64>::from_fn(a.nrows(), a.ncols(), |i, j| {
            let nrm = (0..a.ncols()).map(|l| a[(i, l)] * a[(i, l)]).sum::<f64>().sqrt();
            if nrm == 0.0 {
                0.0
            } else {
                a[(i, j)] / nrm
            }
        });
        let svd = scaled.svd().unwrap();
        let s = svd.S().column_vector();
        let smax = if s.nrows() > 0 { s[0] } else { 0.0 };
        let rank = (0..s.nrows()).filter(|&i| s[i] > 1e-9 * smax.max(1.0)).count();
        let v = svd.V();
        Mat::<f64>::from_fn(a.ncols(), a.ncols() - rank, |i, j| v[(i, rank + j)])
    }

    /// Sine of the largest principal angle between span(k) and span(n) (`n`
    /// orthonormal, same dimension).
    fn max_angle_sine(kernel: &[Vec<(usize, f64)>], m: usize, n: &Mat<f64>) -> f64 {
        assert_eq!(kernel.len(), n.ncols());
        if kernel.is_empty() {
            return 0.0;
        }
        let mut k = Mat::<f64>::zeros(m, kernel.len());
        for (c, v) in kernel.iter().enumerate() {
            for &(i, x) in v {
                k[(i, c)] = x;
            }
        }
        let q = k.qr().compute_thin_Q();
        let proj = n * (n.transpose() * &q);
        let diff = &q - &proj;
        diff.singular_values().unwrap().first().copied().unwrap_or(0.0)
    }

    fn check_invariants(qr: &SparseQr) {
        assert_eq!(qr.rank() + qr.free_columns().len(), qr.ncols());
        let mut all: Vec<usize> = qr
            .pivot_columns()
            .iter()
            .chain(qr.free_columns())
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..qr.ncols()).collect::<Vec<_>>());
        assert!(qr.free_columns().windows(2).all(|w| w[0] < w[1]));
        for k in 0..qr.rank() {
            let row = qr.r_row(k);
            assert_eq!(row[0].0, qr.pivot_columns()[k]);
            assert_eq!(row[0].1.abs(), qr.diagonal()[k]);
            assert!(row[1..].windows(2).all(|w| w[0].0 < w[1].0));
            for &(c, _) in &row[1..] {
                // only free columns or pivots of later rows
                let later = qr.pivot_columns()[k + 1..].contains(&c);
                assert!(later || qr.free_columns().binary_search(&c).is_ok());
            }
        }
        assert_eq!(
            qr.r_nnz(),
            (0..qr.rank()).map(|k| qr.r_row(k).len()).sum::<usize>()
        );
    }

    /// Factor, check rank, residuals, consistency of
    /// `kernel_vector` with `kernel_basis` and the angle to the SVD null space.
    fn cross_check(a: &Mat<f64>, r: usize, what: &str) {
        let sp = dense_to_sparse(a);
        let opts = QrOptions::default();
        let qr = SparseQr::factor(&sp, &opts).unwrap();
        check_invariants(&qr);
        assert_eq!(qr.rank(), r, "{what}: rank");
        let basis = qr.kernel_basis(&opts).unwrap();
        assert_eq!(basis.len(), a.ncols() - r);
        for (i, &f) in qr.free_columns().iter().enumerate() {
            let k = qr.kernel_vector(f);
            assert_eq!(k, basis[i], "{what}: kernel_vector vs kernel_basis");
            let res = SparseQr::relative_residual(&sp, &k);
            assert!(res <= 10.0 * opts.rank_tol, "{what}: residual {res:e}");
        }
        let ver = qr.verify(&sp, 0);
        assert!(ver.max_relative_residual <= 10.0 * opts.rank_tol);
        assert_eq!(ver.rank, r);
        let null = svd_null_space(a);
        assert_eq!(null.ncols(), basis.len(), "{what}: SVD null space dimension");
        let sine = max_angle_sine(&basis, a.ncols(), &null);
        assert!(sine < 1e-8, "{what}: principal angle sine {sine:e}");
    }

    #[test]
    fn dense_cross_check_planted_rank() {
        let mut rng = StdRng::seed_from_u64(20261003);
        for &(p, m, r) in &[
            (20, 35, 7),
            (35, 20, 7),
            (30, 30, 12),
            (30, 30, 30),
            (60, 25, 10),
            (25, 60, 25),
            (60, 60, 1),
            (60, 60, 45),
            (12, 40, 0),
        ] {
            let noises: &[f64] = if r == 0 { &[0.0] } else { &[0.0, 1e-13] }; // row scaling turns pure noise into signal
            for &noise in noises {
                let a = planted(&mut rng, p, m, r, noise);
                cross_check(&a, r, &format!("{p}x{m} rank {r} noise {noise:e}"));
            }
        }
    }

    #[test]
    fn dense_cross_check_special_rows_and_columns() {
        let mut rng = StdRng::seed_from_u64(7);
        // all-zero column, empty row, duplicate rows, rows scaled by 1e6 and 1e-6
        let base = planted(&mut rng, 30, 24, 9, 1e-13);
        let (p, m) = (base.nrows(), base.ncols());
        let zero_col = 5;
        let a = Mat::<f64>::from_fn(p + 4, m + 1, |i, j| {
            if j == zero_col || i == 3 {
                return 0.0; // zero column; row 3 empty
            }
            let jj = if j > zero_col { j - 1 } else { j };
            let ii = match i {
                i if i < p => i,
                i if i == p => 0, // duplicates of rows 0, 1 and 2
                i if i == p + 1 => 1,
                i if i == p + 2 => 2,
                _ => 7,
            };
            let scale = match i {
                10 => 1e6,
                11 => 1e-6,
                i if i == p + 3 => -1e6, // a scaled duplicate of row 7
                _ => 1.0,
            };
            scale * base[(ii, jj)]
        });
        let sp = dense_to_sparse(&a);
        assert!(sp.rows[3].is_empty());
        cross_check(&a, 9, "special rows and columns");
        let qr = SparseQr::factor(&sp, &QrOptions::default()).unwrap();
        assert!(qr.free_columns().contains(&zero_col));
        assert!(qr.dependent_rows().iter().any(|&(i, n)| i == 3 && n == 0.0));
    }

    #[test]
    fn near_dependence_and_monotonicity() {
        let mut rng = StdRng::seed_from_u64(11);
        let m = 12;
        let a: Vec<f64> = (0..m).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f64> = (0..m).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let c: Vec<f64> = (0..m).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let third: Vec<f64> = (0..m).map(|j| a[j] + b[j] + 1e-10 * c[j]).collect();
        let to_row = |v: &[f64]| v.iter().copied().enumerate().collect::<Vec<_>>();
        let sp = SparseRows {
            ncols: m,
            rows: vec![to_row(&a), to_row(&b), to_row(&third)],
        };
        let rank_at = |tol: f64| {
            SparseQr::factor(
                &sp,
                &QrOptions {
                    rank_tol: tol,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let qr = rank_at(1e-9);
        assert_eq!(qr.rank(), 2);
        assert_eq!(qr.dependent_rows().len(), 1);
        assert_eq!(qr.dependent_rows()[0].0, 2);
        assert!(qr.discarded_norm() < 1e-9 && qr.discarded_norm() > 1e-12);
        // the kernel at 1e-9 is a near-kernel: residual of order 1e-10 / ‖C‖_F
        let ver = qr.verify(&sp, 0);
        assert!(ver.max_relative_residual < 1e-9, "{ver:?}");
        assert_eq!(rank_at(1e-12).rank(), 3);

        // monotonicity on this matrix and on graded matrices
        let tols: Vec<f64> = (0..=32).map(|i| 10f64.powf(-16.0 + 0.5 * i as f64)).collect();
        let mut matrices = vec![sp];
        for seed in 0..4 {
            let mut rng = StdRng::seed_from_u64(100 + seed);
            let (p, n) = (24, 18);
            let u = random_orthonormal(&mut rng, p, n);
            let v = random_orthonormal(&mut rng, n, n);
            let s: Vec<f64> = (0..n).map(|i| 10f64.powf(-(i as f64) * 0.8)).collect();
            let g = Mat::<f64>::from_fn(p, n, |i, j| (0..n).map(|l| u[(i, l)] * s[l] * v[(j, l)]).sum());
            matrices.push(dense_to_sparse(&g));
        }
        for (idx, mat) in matrices.iter().enumerate() {
            let mut last = usize::MAX;
            for &tol in &tols {
                let qr = SparseQr::factor(
                    mat,
                    &QrOptions {
                        rank_tol: tol,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(
                    qr.rank() <= last,
                    "matrix {idx}: rank increased to {} at tol {tol:e}",
                    qr.rank()
                );
                let ver = qr.verify(mat, 0);
                assert!(
                    ver.max_relative_residual <= 10.0 * tol.max(1e-15),
                    "matrix {idx}, tol {tol:e}: {ver:?}"
                );
                last = qr.rank();
            }
        }
    }

    #[test]
    fn kahan_matrix_diagnostic() {
        // Kahan matrix: diag(1, s, …, s^{n−1}) · (I − c · strictly upper ones)
        let n = 30;
        let theta: f64 = 1.2;
        let (s, c) = (theta.sin(), theta.cos());
        let k = Mat::<f64>::from_fn(n, n, |i, j| {
            let si = s.powi(i as i32);
            if i == j {
                si
            } else if j > i {
                -c * si
            } else {
                0.0
            }
        });
        let sp = dense_to_sparse(&k);
        let sigma = k.singular_values().unwrap();
        let sigma_min = *sigma.last().unwrap();
        let scaled = Mat::<f64>::from_fn(n, n, |i, j| {
            k[(i, j)] / (0..n).map(|l| k[(i, l)] * k[(i, l)]).sum::<f64>().sqrt()
        });
        let sigma_scaled = scaled.singular_values().unwrap();
        let sigma_min_scaled = sigma_scaled[n - 1];
        // at 1e-6 (a typical tolerance) and at a tolerance between
        // σ_min of the row-scaled matrix and its second smallest singular value,
        // where the SVD rank is n − 1
        let mid = (sigma_min_scaled * sigma_scaled[n - 2]).sqrt();
        for tol in [1e-6, mid] {
            let qr = SparseQr::factor(
                &sp,
                &QrOptions {
                    rank_tol: tol,
                    ..Default::default()
                },
            )
            .unwrap();
            check_invariants(&qr);
            let ver = qr.verify(&sp, 0);
            let svd_rank = sigma_scaled.iter().filter(|&&s| s > tol).count();
            println!(
                "Kahan n = {n}, θ = {theta}, rank_tol {tol:.3e}: rank {} (SVD of the row-scaled matrix: {svd_rank}), \
                 min diagonal {:.3e}, σ_min {:.3e} (row-scaled {:.3e}, next {:.3e}), \
                 min_diagonal / σ_min(row-scaled) = {:.3e}, min_diagonal / σ_min(unscaled) = {:.3e}, \
                 max relative residual {:.3e}",
                qr.rank(),
                ver.min_diagonal,
                sigma_min,
                sigma_min_scaled,
                sigma_scaled[n - 2],
                ver.min_diagonal / sigma_min_scaled,
                ver.min_diagonal / sigma_min,
                ver.max_relative_residual
            );
            // the residual guarantee holds whatever the rank decision
            assert!(ver.max_relative_residual <= 10.0 * tol);
        }
    }

    #[test]
    fn invalid_input_and_fill_limits() {
        let bad_col = SparseRows {
            ncols: 2,
            rows: vec![vec![(2, 1.0)]],
        };
        assert!(matches!(
            SparseQr::factor(&bad_col, &QrOptions::default()),
            Err(QrError::InvalidInput(_))
        ));
        let dup = SparseRows {
            ncols: 2,
            rows: vec![vec![(1, 1.0), (1, 2.0)]],
        };
        assert!(matches!(
            SparseQr::factor(&dup, &QrOptions::default()),
            Err(QrError::InvalidInput(_))
        ));
        let nan = SparseRows {
            ncols: 2,
            rows: vec![vec![(0, f64::NAN)]],
        };
        assert!(matches!(
            SparseQr::factor(&nan, &QrOptions::default()),
            Err(QrError::InvalidInput(_))
        ));
        let ok = SparseRows {
            ncols: 3,
            rows: vec![vec![(0, 1.0), (1, 1.0), (2, 1.0)], vec![(1, 1.0), (2, 2.0)]],
        };
        let small = QrOptions {
            max_r_nnz: 4,
            max_kernel_nnz: 1,
            ..Default::default()
        };
        assert!(matches!(
            SparseQr::factor(&ok, &small),
            Err(QrError::TooMuchFill(_))
        ));
        let qr = SparseQr::factor(&ok, &QrOptions::default()).unwrap();
        assert!(matches!(qr.kernel_basis(&small), Err(QrError::TooMuchFill(_))));
        assert_eq!(qr.kernel_basis(&QrOptions::default()).unwrap().len(), 1);
    }

    #[test]
    fn degenerate_shapes() {
        let empty = SparseRows {
            ncols: 4,
            rows: vec![vec![], vec![]],
        };
        let qr = SparseQr::factor(&empty, &QrOptions::default()).unwrap();
        assert_eq!(qr.rank(), 0);
        assert_eq!(qr.free_columns(), &[0, 1, 2, 3]);
        assert_eq!(qr.kernel_vector(2), vec![(2, 1.0)]);
        let ver = qr.verify(&empty, 0);
        assert_eq!(
            (ver.rank, ver.kernel_dimension, ver.max_relative_residual),
            (0, 4, 0.0)
        );
        // rank_tol = 0: only exact zeros are dependent
        let a = SparseRows {
            ncols: 2,
            rows: vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 2.0), (1, 2.0)]],
        };
        let qr = SparseQr::factor(
            &a,
            &QrOptions {
                rank_tol: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(qr.rank(), 1);
        let none = SparseRows {
            ncols: 0,
            rows: vec![],
        };
        assert_eq!(SparseQr::factor(&none, &QrOptions::default()).unwrap().rank(), 0);
    }
}
