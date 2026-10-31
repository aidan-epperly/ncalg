//! Linear algebra over the integers and over ℤ/N.
//!
//! * [`ZnMat`]: dense matrices over ℤ/N for any modulus `2 ≤ N < 2^63`.
//!   ℤ/N is not a field when `N` is composite, so ranks and echelon forms are
//!   replaced by the *Howell form*: the canonical generating set of the row
//!   span, in which every pivot divides `N`.  Two matrices have the same row
//!   span iff their Howell forms are equal, and a vector lies in the span iff it
//!   reduces to zero against the form (Howell 1986, Storjohann–Mulders 1998).
//!   On top of it: span membership, null spaces (as generating sets), and
//!   solving `A x = b`.
//! * [`SparseHowell`]: the Howell form of sparse vectors, built
//!   incrementally; membership, normal forms of cosets, the span size, sparse null
//!   spaces and solving.  It is the one implementation: [`ZnMat::howell`] and
//!   the methods built on it call it.
//! * [`IntMat`]: dense matrices over ℤ with `BigInt` entries: Hermite normal
//!   form, Smith normal form with unimodular transforms, integer kernels and
//!   integer solving.
//! * [`sparse_invariant_factors`]: the invariant factors (no transforms) of a
//!   large sparse integer matrix, by sparse elimination on `±1` pivots before
//!   a dense Smith form of what is left.
//!
//! Conventions match [`crate::ff::FqMat`]: `nullspace` and `solve` are about
//! `A x`, the `left_` variants about `x A`.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

use crate::arith::int::xgcd;
use std::collections::{BTreeMap, BTreeSet};

/// A unit `u` of ℤ/N with `u a ≡ gcd(a, N) (mod N)`.
pub fn unit_normalizer(a: u64, n: u64) -> u64 {
    let a = a % n;
    if a == 0 {
        return 1;
    }
    let g = a.gcd(&n);
    let m = n / g;
    if m == 1 {
        return 1;
    }
    let (_, s, _) = xgcd(((a / g) % m) as i128, m as i128);
    let u0 = s.rem_euclid(m as i128) as u64;
    // lift u0 (a unit mod m) to a unit mod n
    let mut u = u0;
    while u.gcd(&n) != 1 {
        u += m;
    }
    u % n
}

/// Dense row-major matrix over ℤ/N.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZnMat {
    n: u64,
    rows: usize,
    cols: usize,
    data: Vec<u64>,
}

impl ZnMat {
    /// The `rows × cols` zero matrix over ℤ/n, `2 ≤ n < 2^63`.
    pub fn zeros(n: u64, rows: usize, cols: usize) -> Self {
        assert!((2..1u64 << 63).contains(&n), "modulus must be in 2..2^63");
        ZnMat {
            n,
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }
    /// The `size × size` identity matrix over ℤ/n.
    pub fn identity(n: u64, size: usize) -> Self {
        let mut m = Self::zeros(n, size, size);
        for i in 0..size {
            m.data[i * size + i] = 1 % n;
        }
        m
    }
    /// From rows of integers (reduced mod N).
    pub fn from_rows(n: u64, rows: &[Vec<i64>]) -> Self {
        let c = rows.first().map_or(0, |x| x.len());
        let mut m = Self::zeros(n, rows.len(), c);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), c, "ragged rows");
            for (j, &x) in row.iter().enumerate() {
                m.data[i * c + j] = (x as i128).rem_euclid(n as i128) as u64;
            }
        }
        m
    }
    /// The modulus `n`.
    pub fn modulus(&self) -> u64 {
        self.n
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    /// Entry `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> u64 {
        self.data[i * self.cols + j]
    }
    #[inline]
    /// Set entry `(i, j)` to `v mod n`.
    pub fn set(&mut self, i: usize, j: usize, v: u64) {
        self.data[i * self.cols + j] = v % self.n;
    }
    /// Row `i`.
    pub fn row(&self, i: usize) -> &[u64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    /// The rows as vectors.
    pub fn to_rows(&self) -> Vec<Vec<u64>> {
        (0..self.rows).map(|i| self.row(i).to_vec()).collect()
    }
    /// The transpose.
    pub fn transpose(&self) -> ZnMat {
        let mut t = Self::zeros(self.n, self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j * self.rows + i] = self.get(i, j);
            }
        }
        t
    }

    #[inline]
    fn mulm(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % self.n as u128) as u64
    }
    #[inline]
    fn addm(&self, a: u64, b: u64) -> u64 {
        let s = a as u128 + b as u128;
        (s % self.n as u128) as u64
    }

    /// The matrix product `self · b`.
    pub fn mul(&self, b: &ZnMat) -> ZnMat {
        assert_eq!(self.n, b.n, "moduli differ");
        assert_eq!(self.cols, b.rows);
        let mut c = Self::zeros(self.n, self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0 {
                    continue;
                }
                for j in 0..b.cols {
                    let v = self.addm(c.get(i, j), self.mulm(a, b.get(k, j)));
                    c.data[i * b.cols + j] = v;
                }
            }
        }
        c
    }
    /// The matrix-vector product `self · v`.
    pub fn mul_vec(&self, v: &[u64]) -> Vec<u64> {
        assert_eq!(v.len(), self.cols);
        (0..self.rows)
            .map(|i| {
                (0..self.cols).fold(0, |acc, j| {
                    self.addm(acc, self.mulm(self.get(i, j), v[j] % self.n))
                })
            })
            .collect()
    }
    /// `v A` for a row vector `v`.
    pub fn vec_mul(&self, v: &[u64]) -> Vec<u64> {
        self.transpose().mul_vec(v)
    }

    /// The rows as sparse vectors.
    fn sparse_rows(&self) -> Vec<SparseRow> {
        (0..self.rows)
            .map(|i| {
                self.row(i)
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x != 0)
                    .map(|(j, &x)| (j, x))
                    .collect()
            })
            .collect()
    }

    fn howell_engine(&self) -> SparseHowell {
        SparseHowell::from_rows(self.n, self.cols, &self.sparse_rows())
    }

    /// Howell form of the row span: rows with strictly increasing pivot
    /// columns, each pivot a divisor of `N`, entries above a pivot reduced to
    /// `0..pivot`, and the Howell property (every span vector whose first `j`
    /// entries vanish is a combination of the rows with pivot column `≥ j`).
    /// Canonical: equal row spans give equal forms.  Computed by
    /// [`SparseHowell`].
    pub fn howell(&self) -> ZnMat {
        let rows = self.howell_engine().rows();
        let mut m = Self::zeros(self.n, rows.len(), self.cols);
        for (i, row) in rows.iter().enumerate() {
            for &(j, x) in row {
                m.data[i * self.cols + j] = x;
            }
        }
        m
    }

    /// `log₂`-free size of the row span: `|span| = Π N / pivot`, as a list of the factors.
    pub fn span_factors(&self) -> Vec<u64> {
        self.howell_engine().span_factors()
    }

    /// Is the row vector `v` in the row span?
    pub fn span_contains(&self, v: &[u64]) -> bool {
        assert_eq!(v.len(), self.cols);
        self.howell_engine().contains(&dense_to_sparse(v))
    }

    /// Generators of `{x : x A = 0}` (a submodule of (ℤ/N)^rows), in Howell form.
    pub fn left_nullspace(&self) -> Vec<Vec<u64>> {
        SparseHowell::left_nullspace(self.n, self.cols, &self.sparse_rows())
            .into_iter()
            .map(|v| sparse_to_dense(&v, self.rows))
            .collect()
    }

    /// Generators of `{x : A x = 0}`.
    pub fn nullspace(&self) -> Vec<Vec<u64>> {
        self.transpose().left_nullspace()
    }

    /// Some `x` with `x A = b`, or `None` if `b` is not in the row span.
    pub fn left_solve(&self, b: &[u64]) -> Option<Vec<u64>> {
        assert_eq!(b.len(), self.cols);
        SparseHowell::left_solve(self.n, self.cols, &self.sparse_rows(), &dense_to_sparse(b))
            .map(|x| sparse_to_dense(&x, self.rows))
    }

    /// Some `x` with `A x = b`, or `None` if inconsistent.
    pub fn solve(&self, b: &[u64]) -> Option<Vec<u64>> {
        assert_eq!(b.len(), self.rows);
        self.transpose().left_solve(b)
    }
}

/// A sparse vector over ℤ/N: `(column, value)` pairs with increasing columns
/// and values in `1..N`.
pub type SparseRow = Vec<(usize, u64)>;

fn dense_to_sparse(v: &[u64]) -> SparseRow {
    v.iter()
        .enumerate()
        .filter(|(_, &x)| x != 0)
        .map(|(j, &x)| (j, x))
        .collect()
}

fn sparse_to_dense(v: &[(usize, u64)], len: usize) -> Vec<u64> {
    let mut out = vec![0; len];
    for &(j, x) in v {
        out[j] = x;
    }
    out
}

#[inline]
fn mul_n(a: u64, b: u64, n: u64) -> u64 {
    ((a as u128 * b as u128) % n as u128) as u64
}

/// `s x + t y` over ℤ/n, merged by column, zeros dropped.
fn axpy(n: u64, x: &[(usize, u64)], s: u64, y: &[(usize, u64)], t: u64) -> SparseRow {
    let mut out = Vec::with_capacity(x.len() + y.len());
    let (mut i, mut k) = (0, 0);
    while i < x.len() || k < y.len() {
        let (j, v) = match (x.get(i), y.get(k)) {
            (Some(&(a, u)), Some(&(b, w))) if a == b => {
                i += 1;
                k += 1;
                (
                    a,
                    ((mul_n(s, u, n) as u128 + mul_n(t, w, n) as u128) % n as u128) as u64,
                )
            }
            (Some(&(a, u)), Some(&(b, _))) if a < b => {
                i += 1;
                (a, mul_n(s, u, n))
            }
            (Some(&(a, u)), None) => {
                i += 1;
                (a, mul_n(s, u, n))
            }
            (_, Some(&(b, w))) => {
                k += 1;
                (b, mul_n(t, w, n))
            }
            (None, None) => unreachable!(),
        };
        if v != 0 {
            out.push((j, v));
        }
    }
    out
}

fn scale_row(n: u64, x: &[(usize, u64)], s: u64) -> SparseRow {
    x.iter()
        .filter_map(|&(j, v)| {
            let w = mul_n(s, v, n);
            (w != 0).then_some((j, w))
        })
        .collect()
}

/// The Howell form over ℤ/N of a set of sparse vectors, built incrementally:
/// the canonical generating set of their span, as for
/// [`ZnMat::howell`], without dense rows.
///
/// Rows are kept by pivot column.  [`insert`](Self::insert) reduces a vector
/// against them, leading column first: a leading entry `b` divisible by the
/// pivot `a` there is canceled; otherwise a Bezout step (a unimodular 2×2
/// combination) replaces the pivot row by one with pivot `gcd(a, b)` and
/// continues with the other combination, whose leading entry vanishes.  A new
/// pivot row is scaled by a unit to make its pivot a divisor of `N`.  Every new
/// or replaced pivot row `r` with pivot `h` also inserts its annihilator row
/// `(N/h)·r`, which is zero at the pivot: that gives the Howell property, so
/// the stored rows are a Howell basis after every insertion and greedy
/// reduction decides membership.  The canonical form (entries above each
/// pivot reduced to `0..pivot`) is assembled by [`rows`](Self::rows).  The
/// result does not depend on the order of insertion.
///
/// ```
/// use ncalg::zmod::SparseHowell;
/// // ℤ/12: the span of (2, 1) and (0, 3)
/// let mut h = SparseHowell::new(12, 2);
/// h.insert(&[(0, 2), (1, 1)]);
/// h.insert(&[(1, 3)]);
/// assert!(h.contains(&[(0, 4), (1, 5)])); // 2·(2, 1) + (0, 3)
/// assert!(!h.contains(&[(0, 1)]));
/// // pivots 2 and 3: |span| = (12/2)·(12/3)
/// assert_eq!(h.pivots(), vec![(0, 2), (1, 3)]);
/// assert_eq!(h.span_factors(), vec![6, 4]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseHowell {
    n: u64,
    cols: usize,
    rows: BTreeMap<usize, SparseRow>,
}

impl SparseHowell {
    /// The zero module in `(ℤ/n)^cols`, `2 ≤ n < 2^63`.
    pub fn new(n: u64, cols: usize) -> Self {
        assert!((2..1u64 << 63).contains(&n), "modulus must be in 2..2^63");
        SparseHowell {
            n,
            cols,
            rows: BTreeMap::new(),
        }
    }

    /// The span of `rows`.
    pub fn from_rows(n: u64, cols: usize, rows: &[SparseRow]) -> Self {
        let mut h = Self::new(n, cols);
        for r in rows {
            h.insert(r);
        }
        h
    }

    /// The modulus `N`.
    pub fn modulus(&self) -> u64 {
        self.n
    }
    /// The length of the vectors.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Number of rows of the Howell form.
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    /// The zero module?
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Sort by column, add repeated columns, reduce mod `N`, drop zeros.
    fn normalize(&self, v: &[(usize, u64)]) -> SparseRow {
        let mut v: SparseRow = v.iter().map(|&(j, x)| (j, x % self.n)).collect();
        v.sort_by_key(|&(j, _)| j);
        let mut out: SparseRow = Vec::with_capacity(v.len());
        for (j, x) in v {
            assert!(j < self.cols, "column {j} out of range");
            match out.last_mut() {
                Some((k, y)) if *k == j => *y = ((*y as u128 + x as u128) % self.n as u128) as u64,
                _ => out.push((j, x)),
            }
        }
        out.retain(|&(_, x)| x != 0);
        out
    }

    /// Add `v` to the generating set; `true` if the span grew (`v` was not in it).
    pub fn insert(&mut self, v: &[(usize, u64)]) -> bool {
        let n = self.n;
        let md = |x: i128| x.rem_euclid(n as i128) as u64;
        let mut grew = false;
        // (vector, is it the inserted one); the others are annihilator rows
        let mut queue: Vec<(SparseRow, bool)> = vec![(self.normalize(v), true)];
        while let Some((mut v, original)) = queue.pop() {
            while let Some(&(j, b)) = v.first() {
                match self.rows.get_mut(&j) {
                    None => {
                        let v = scale_row(n, &v, unit_normalizer(b, n));
                        let h = v[0].1;
                        debug_assert_eq!(n % h, 0);
                        let ann = scale_row(n, &v, n / h);
                        if !ann.is_empty() {
                            queue.push((ann, false));
                        }
                        self.rows.insert(j, v);
                        grew |= original;
                        break;
                    }
                    Some(r) => {
                        let a = r[0].1;
                        if b % a == 0 {
                            v = axpy(n, &v, 1, r, n - b / a);
                            continue;
                        }
                        // s a + t b = g = gcd(a, b), a divisor of a and so of N
                        let (g, s, t) = xgcd(a as i128, b as i128);
                        let new_r = axpy(n, r, md(s), &v, md(t));
                        let new_v = axpy(n, r, md(-(b as i128 / g)), &v, md(a as i128 / g));
                        debug_assert_eq!(new_r[0], (j, g as u64));
                        let ann = scale_row(n, &new_r, n / g as u64);
                        if !ann.is_empty() {
                            queue.push((ann, false));
                        }
                        *r = new_r;
                        v = new_v;
                        grew |= original;
                    }
                }
            }
        }
        grew
    }

    /// The normal form of `v` modulo the span: left to right, each entry at a
    /// pivot column is reduced to `0..pivot`.  Two vectors differ by a span
    /// element iff their normal forms are equal; `v` is in the span iff its
    /// normal form is zero.
    pub fn reduce(&self, v: &[(usize, u64)]) -> SparseRow {
        let n = self.n;
        let mut v = self.normalize(v);
        let mut k = 0;
        while k < v.len() {
            let (j, b) = v[k];
            if let Some(r) = self.rows.get(&j) {
                let q = b / r[0].1;
                if q != 0 {
                    v = axpy(n, &v, 1, r, n - q);
                }
            }
            k = v.partition_point(|&(c, _)| c <= j);
        }
        v
    }

    /// Is `v` in the span?
    pub fn contains(&self, v: &[(usize, u64)]) -> bool {
        let n = self.n;
        let mut v = self.normalize(v);
        while let Some(&(j, b)) = v.first() {
            match self.rows.get(&j) {
                Some(r) if b % r[0].1 == 0 => v = axpy(n, &v, 1, r, n - b / r[0].1),
                _ => return false,
            }
        }
        true
    }

    /// `(pivot column, pivot)` of each row, by column.
    pub fn pivots(&self) -> Vec<(usize, u64)> {
        self.rows.iter().map(|(&j, r)| (j, r[0].1)).collect()
    }

    /// `|span| = Π N / pivot`, as the list of factors (by pivot column).
    pub fn span_factors(&self) -> Vec<u64> {
        self.rows.values().map(|r| self.n / r[0].1).collect()
    }

    /// The canonical Howell form: the rows by pivot column, with the entries
    /// above every pivot reduced to `0..pivot` (bottom-up, so each row is
    /// reduced against rows that already are).
    pub fn rows(&self) -> Vec<SparseRow> {
        let n = self.n;
        let mut done: BTreeMap<usize, SparseRow> = BTreeMap::new();
        for (&j, r) in self.rows.iter().rev() {
            let mut v = r.clone();
            let mut k = 1;
            while k < v.len() {
                let (c, b) = v[k];
                if let Some(p) = done.get(&c) {
                    let q = b / p[0].1;
                    if q != 0 {
                        v = axpy(n, &v, 1, p, n - q);
                    }
                }
                k = v.partition_point(|&(cc, _)| cc <= c);
            }
            done.insert(j, v);
        }
        done.into_values().collect()
    }

    /// `[A | I]` for sparse rows `A` with `cols` columns.
    fn augmented(n: u64, cols: usize, rows: &[SparseRow]) -> SparseHowell {
        let m = rows.len();
        let mut h = SparseHowell::new(n, cols + m);
        for (i, r) in rows.iter().enumerate() {
            let mut v = r.clone();
            v.push((cols + i, 1));
            h.insert(&v);
        }
        h
    }

    /// Generators of `{x : x A = 0}` for the sparse rows `A` (vectors indexed
    /// by the rows of `A`), in Howell form: the rows of the Howell form of
    /// `[A | I]` whose pivot lies in the identity block.
    pub fn left_nullspace(n: u64, cols: usize, rows: &[SparseRow]) -> Vec<SparseRow> {
        Self::augmented(n, cols, rows)
            .rows()
            .into_iter()
            .filter(|r| r[0].0 >= cols)
            .map(|r| r.into_iter().map(|(j, x)| (j - cols, x)).collect())
            .collect()
    }

    /// Some `x` with `x A = b` for the sparse rows `A`, or `None` if `b` is not
    /// in their span: the normal form of `(b, 0)` against `[A | I]` is
    /// `(0, −x)`.
    pub fn left_solve(n: u64, cols: usize, rows: &[SparseRow], b: &[(usize, u64)]) -> Option<SparseRow> {
        let h = Self::augmented(n, cols, rows);
        let rem = h.reduce(b);
        if rem.first().is_some_and(|&(j, _)| j < cols) {
            return None;
        }
        Some(rem.into_iter().map(|(j, x)| (j - cols, (n - x) % n)).collect())
    }
}

// ---------------------------------------------------------------------------
// integer matrices
// ---------------------------------------------------------------------------

/// A unimodular `[[s, t], [u, v]]` sending `(a, b)` to `(g, 0)`, `g = ±gcd(a, b)`.
/// When `a | b` it is the plain elimination `(1, 0, -b/a, 1)`, so a pivot that
/// already divides its row or column is never mixed with other entries; without
/// this the Smith loop can cycle.
fn gcd_step(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt, BigInt) {
    if !a.is_zero() && (b % a).is_zero() {
        return (BigInt::one(), BigInt::zero(), -(b / a), BigInt::one());
    }
    let e = a.extended_gcd(b);
    (e.x, e.y, -(b / &e.gcd), a / &e.gcd)
}

/// Dense row-major matrix over ℤ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMat {
    rows: usize,
    cols: usize,
    data: Vec<BigInt>,
}

/// Smith normal form `U A V = D` with `U`, `V` unimodular and `D` diagonal,
/// `d_0 | d_1 | …`, all `d_i ≥ 0`.
#[derive(Clone, Debug)]
pub struct Smith {
    /// The unimodular row transformation `U`.
    pub u: IntMat,
    /// The unimodular column transformation `V`.
    pub v: IntMat,
    /// The diagonal of `D` (length `min(rows, cols)`).
    pub diag: Vec<BigInt>,
}

impl Smith {
    /// Number of nonzero invariant factors.
    pub fn rank(&self) -> usize {
        self.diag.iter().filter(|d| !d.is_zero()).count()
    }
}

impl IntMat {
    /// The `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        IntMat {
            rows,
            cols,
            data: vec![BigInt::zero(); rows * cols],
        }
    }
    /// The `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = BigInt::one();
        }
        m
    }
    /// The matrix with these rows. Panics on ragged rows.
    pub fn from_rows(rows: &[Vec<i64>]) -> Self {
        let c = rows.first().map_or(0, |x| x.len());
        let mut m = Self::zeros(rows.len(), c);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), c, "ragged rows");
            for (j, &x) in row.iter().enumerate() {
                m.data[i * c + j] = BigInt::from(x);
            }
        }
        m
    }
    /// The matrix with these rows. Panics on ragged rows.
    pub fn from_big_rows(rows: &[Vec<BigInt>]) -> Self {
        let c = rows.first().map_or(0, |x| x.len());
        let mut data = Vec::with_capacity(rows.len() * c);
        for row in rows {
            assert_eq!(row.len(), c, "ragged rows");
            data.extend(row.iter().cloned());
        }
        IntMat {
            rows: rows.len(),
            cols: c,
            data,
        }
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Entry `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> &BigInt {
        &self.data[i * self.cols + j]
    }
    /// Set entry `(i, j)` to `v`.
    pub fn set(&mut self, i: usize, j: usize, v: BigInt) {
        self.data[i * self.cols + j] = v;
    }
    /// Row `i`.
    pub fn row(&self, i: usize) -> &[BigInt] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
    /// The transpose.
    pub fn transpose(&self) -> IntMat {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j * self.rows + i] = self.get(i, j).clone();
            }
        }
        t
    }
    /// The matrix product `self · b`.
    pub fn mul(&self, b: &IntMat) -> IntMat {
        assert_eq!(self.cols, b.rows);
        let mut c = Self::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a.is_zero() {
                    continue;
                }
                for j in 0..b.cols {
                    let x = a * b.get(k, j);
                    c.data[i * b.cols + j] += x;
                }
            }
        }
        c
    }
    /// The matrix-vector product `self · v`.
    pub fn mul_vec(&self, v: &[BigInt]) -> Vec<BigInt> {
        assert_eq!(v.len(), self.cols);
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self.get(i, j) * &v[j]).sum())
            .collect()
    }

    // elementary operations (on self and an optional transform)
    fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            for j in 0..self.cols {
                self.data.swap(a * self.cols + j, b * self.cols + j);
            }
        }
    }
    fn swap_cols(&mut self, a: usize, b: usize) {
        if a != b {
            for i in 0..self.rows {
                self.data.swap(i * self.cols + a, i * self.cols + b);
            }
        }
    }
    /// rows (a, b) ← (s·a + t·b, u·a + v·b)
    fn comb_rows(&mut self, a: usize, b: usize, s: &BigInt, t: &BigInt, u: &BigInt, v: &BigInt) {
        for j in 0..self.cols {
            let x = self.get(a, j).clone();
            let y = self.get(b, j).clone();
            self.data[a * self.cols + j] = s * &x + t * &y;
            self.data[b * self.cols + j] = u * &x + v * &y;
        }
    }
    /// cols (a, b) ← (s·a + t·b, u·a + v·b)
    fn comb_cols(&mut self, a: usize, b: usize, s: &BigInt, t: &BigInt, u: &BigInt, v: &BigInt) {
        for i in 0..self.rows {
            let x = self.get(i, a).clone();
            let y = self.get(i, b).clone();
            self.data[i * self.cols + a] = s * &x + t * &y;
            self.data[i * self.cols + b] = u * &x + v * &y;
        }
    }
    fn negate_row(&mut self, a: usize) {
        for j in 0..self.cols {
            let x = -self.get(a, j);
            self.data[a * self.cols + j] = x;
        }
    }
    /// row a ← row a − q · row b
    fn sub_row(&mut self, a: usize, b: usize, q: &BigInt) {
        for j in 0..self.cols {
            let x = self.get(b, j) * q;
            self.data[a * self.cols + j] -= x;
        }
    }

    /// Row-style Hermite normal form `H = U A`: echelon, positive pivots,
    /// entries above each pivot in `0..pivot`, zero rows at the bottom.
    /// Returns `(H, U)` with `U` unimodular.
    pub fn hermite(&self) -> (IntMat, IntMat) {
        let mut h = self.clone();
        let mut u = IntMat::identity(self.rows);
        let mut r = 0;
        for j in 0..self.cols {
            if r == self.rows {
                break;
            }
            for i in (r + 1)..self.rows {
                if h.get(i, j).is_zero() {
                    continue;
                }
                let (s, t, uu, vv) = gcd_step(h.get(r, j), h.get(i, j));
                h.comb_rows(r, i, &s, &t, &uu, &vv);
                u.comb_rows(r, i, &s, &t, &uu, &vv);
            }
            if h.get(r, j).is_zero() {
                continue;
            }
            if h.get(r, j).is_negative() {
                h.negate_row(r);
                u.negate_row(r);
            }
            let p = h.get(r, j).clone();
            for i in 0..r {
                let q = h.get(i, j).div_floor(&p);
                if !q.is_zero() {
                    h.sub_row(i, r, &q);
                    u.sub_row(i, r, &q);
                }
            }
            r += 1;
        }
        (h, u)
    }

    /// Smith normal form with transforms.
    pub fn smith(&self) -> Smith {
        let (m, n) = (self.rows, self.cols);
        let mut d = self.clone();
        let mut u = IntMat::identity(m);
        let mut v = IntMat::identity(n);
        let k = m.min(n);
        for t in 0..k {
            // pivot: smallest nonzero |entry| in the trailing block
            loop {
                let mut best: Option<(usize, usize)> = None;
                for i in t..m {
                    for j in t..n {
                        let x = d.get(i, j);
                        if !x.is_zero() && best.is_none_or(|(bi, bj)| x.abs() < d.get(bi, bj).abs()) {
                            best = Some((i, j));
                        }
                    }
                }
                let Some((bi, bj)) = best else {
                    // trailing block is zero
                    return Smith {
                        u,
                        v,
                        diag: (0..k).map(|i| d.get(i, i).clone()).collect(),
                    }
                    .normalized();
                };
                d.swap_rows(t, bi);
                u.swap_rows(t, bi);
                d.swap_cols(t, bj);
                v.swap_cols(t, bj);
                // clear column t and row t with gcd steps
                for i in (t + 1)..m {
                    if d.get(i, t).is_zero() {
                        continue;
                    }
                    let (s, tt, uu, vv) = gcd_step(d.get(t, t), d.get(i, t));
                    d.comb_rows(t, i, &s, &tt, &uu, &vv);
                    u.comb_rows(t, i, &s, &tt, &uu, &vv);
                }
                for j in (t + 1)..n {
                    if d.get(t, j).is_zero() {
                        continue;
                    }
                    let (s, tt, uu, vv) = gcd_step(d.get(t, t), d.get(t, j));
                    d.comb_cols(t, j, &s, &tt, &uu, &vv);
                    v.comb_cols(t, j, &s, &tt, &uu, &vv);
                }
                if (t + 1..m).any(|i| !d.get(i, t).is_zero()) {
                    continue;
                }
                // divisibility: the pivot must divide the whole trailing block
                let p = d.get(t, t).clone();
                let bad = (t + 1..m).find(|&i| (t + 1..n).any(|j| !(d.get(i, j) % &p).is_zero()));
                if let Some(i) = bad {
                    // add row i to row t and redo
                    let one = BigInt::one();
                    let zero = BigInt::zero();
                    d.comb_rows(t, i, &one, &one, &zero, &one);
                    u.comb_rows(t, i, &one, &one, &zero, &one);
                    continue;
                }
                break;
            }
        }
        Smith {
            u,
            v,
            diag: (0..k).map(|i| d.get(i, i).clone()).collect(),
        }
        .normalized()
    }

    /// Basis of the integer kernel `{x ∈ ℤ^cols : A x = 0}` (columns of `V` past the rank).
    pub fn kernel(&self) -> Vec<Vec<BigInt>> {
        let s = self.smith();
        let r = s.rank();
        (r..self.cols)
            .map(|j| (0..self.cols).map(|i| s.v.get(i, j).clone()).collect())
            .collect()
    }

    /// Some integer `x` with `A x = b`, or `None`.
    pub fn solve(&self, b: &[BigInt]) -> Option<Vec<BigInt>> {
        assert_eq!(b.len(), self.rows);
        let s = self.smith();
        let c = s.u.mul_vec(b);
        let r = s.rank();
        let mut y = vec![BigInt::zero(); self.cols];
        for i in 0..self.rows {
            if i < r {
                let (q, rem) = c[i].div_rem(&s.diag[i]);
                if !rem.is_zero() {
                    return None;
                }
                y[i] = q;
            } else if !c[i].is_zero() {
                return None;
            }
        }
        Some(s.v.mul_vec(&y))
    }
}

impl Smith {
    /// Make the diagonal non-negative (by negating rows of `U`) and put zeros last.
    fn normalized(mut self) -> Smith {
        for i in 0..self.diag.len() {
            if self.diag[i].is_negative() {
                self.diag[i] = -&self.diag[i];
                self.u.negate_row(i);
            }
        }
        debug_assert!(self
            .diag
            .windows(2)
            .all(|w| w[1].is_zero() || (!w[0].is_zero() && (&w[1] % &w[0]).is_zero())));
        self
    }
}

/// The nonzero invariant factors `d_1 | d_2 | …` (all positive) of the sparse
/// integer matrix with these rows (`(column, value)` entries, repeated
/// columns add), without the transforms.  Pivots on entries `±1` are
/// eliminated first, sparsely and in Markowitz order (each pivot is an
/// invariant factor 1, and deleting its row and column keeps the Smith form);
/// the rows left without units go to [`IntMat::smith`].  The rank is the
/// length of the result.  Meant for boundary matrices (cellular, Khovanov),
/// which are large, very sparse and mostly `±1`.
pub fn sparse_invariant_factors(cols: usize, rows: &[Vec<(usize, i64)>]) -> Vec<BigInt> {
    let mut r: Vec<BTreeMap<usize, BigInt>> = rows
        .iter()
        .map(|row| {
            let mut m: BTreeMap<usize, BigInt> = BTreeMap::new();
            for &(j, x) in row {
                assert!(j < cols, "column {j} out of range");
                *m.entry(j).or_insert_with(BigInt::zero) += x;
            }
            m.retain(|_, x| !x.is_zero());
            m
        })
        .collect();
    let mut col_rows: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); cols];
    for (i, row) in r.iter().enumerate() {
        for &j in row.keys() {
            col_rows[j].insert(i);
        }
    }
    let mut alive: Vec<bool> = vec![true; r.len()];
    let mut units = 0usize;
    loop {
        // one pass over the live rows, shortest first
        let mut order: Vec<usize> = (0..r.len()).filter(|&i| alive[i] && !r[i].is_empty()).collect();
        order.sort_by_key(|&i| r[i].len());
        let mut pivoted = false;
        for i in order {
            if !alive[i] {
                continue;
            }
            // the unit entry of row i in the column with the fewest entries
            let Some(j) = r[i]
                .iter()
                .filter(|(_, x)| x.abs().is_one())
                .map(|(&j, _)| j)
                .min_by_key(|&j| col_rows[j].len())
            else {
                continue;
            };
            let prow = std::mem::take(&mut r[i]);
            alive[i] = false;
            for &k in prow.keys() {
                col_rows[k].remove(&i);
            }
            let u = prow[&j].clone(); // ±1 = its own inverse
            for i2 in std::mem::take(&mut col_rows[j]) {
                let f = &r[i2][&j] * &u;
                for (&k, x) in &prow {
                    let e = r[i2].entry(k).or_insert_with(BigInt::zero);
                    *e -= &f * x;
                    if e.is_zero() {
                        r[i2].remove(&k);
                        col_rows[k].remove(&i2);
                    } else {
                        col_rows[k].insert(i2);
                    }
                }
            }
            units += 1;
            pivoted = true;
        }
        if !pivoted {
            break;
        }
    }
    let mut out = vec![BigInt::one(); units];
    let rest: Vec<&BTreeMap<usize, BigInt>> = r.iter().filter(|row| !row.is_empty()).collect();
    if !rest.is_empty() {
        let used: Vec<usize> = (0..cols).filter(|&j| !col_rows[j].is_empty()).collect();
        let dense: Vec<Vec<BigInt>> = rest
            .iter()
            .map(|row| {
                used.iter()
                    .map(|j| row.get(j).cloned().unwrap_or_else(BigInt::zero))
                    .collect()
            })
            .collect();
        out.extend(
            IntMat::from_big_rows(&dense)
                .smith()
                .diag
                .into_iter()
                .filter(|x| !x.is_zero()),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn sparse_invariant_factors_match_smith() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..200 {
            let (m, n) = (rng.gen_range(1..9), rng.gen_range(1..9));
            let dense: Vec<Vec<i64>> = (0..m)
                .map(|_| {
                    (0..n)
                        .map(|_| {
                            if rng.gen_bool(0.6) {
                                0
                            } else {
                                rng.gen_range(-3..=3)
                            }
                        })
                        .collect()
                })
                .collect();
            let sparse: Vec<Vec<(usize, i64)>> = dense
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .filter(|(_, &x)| x != 0)
                        .map(|(j, &x)| (j, x))
                        .collect()
                })
                .collect();
            let want: Vec<BigInt> = IntMat::from_rows(&dense)
                .smith()
                .diag
                .into_iter()
                .filter(|x| !x.is_zero())
                .collect();
            assert_eq!(sparse_invariant_factors(n, &sparse), want, "{dense:?}");
        }
    }

    fn all_vectors(n: u64, len: usize) -> Vec<Vec<u64>> {
        let mut out = vec![vec![]];
        for _ in 0..len {
            out = out
                .into_iter()
                .flat_map(|v| (0..n).map(move |x| [v.clone(), vec![x]].concat()))
                .collect();
        }
        out
    }

    fn span_brute(a: &ZnMat) -> std::collections::BTreeSet<Vec<u64>> {
        let n = a.modulus();
        all_vectors(n, a.rows())
            .into_iter()
            .map(|x| a.vec_mul(&x))
            .collect()
    }

    #[test]
    fn unit_normalizer_works() {
        for n in 2..60u64 {
            for a in 0..n {
                let u = unit_normalizer(a, n);
                assert_eq!(u.gcd(&n), 1);
                assert_eq!((u * a) % n, a.gcd(&n) % n);
            }
        }
    }

    #[test]
    fn howell_is_canonical_and_spans() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for &n in &[4u64, 6, 8, 9, 12] {
            for _ in 0..40 {
                let (r, c) = (rng.gen_range(1..4), rng.gen_range(1..4));
                let rows: Vec<Vec<i64>> = (0..r)
                    .map(|_| (0..c).map(|_| rng.gen_range(0..n as i64)).collect())
                    .collect();
                let a = ZnMat::from_rows(n, &rows);
                let h = a.howell();
                let span = span_brute(&a);
                assert_eq!(span_brute(&h), span);
                assert_eq!(a.span_factors().iter().product::<u64>() as usize, span.len());
                // a different generating set of the same span
                let mut rows2 = rows.clone();
                let k = rng.gen_range(0..r);
                let coeffs: Vec<i64> = (0..r).map(|_| rng.gen_range(0..n as i64)).collect();
                let extra: Vec<i64> = (0..c)
                    .map(|j| (0..r).map(|i| coeffs[i] * rows[i][j]).sum::<i64>())
                    .collect();
                rows2.push(extra);
                rows2.swap(k, r);
                let u = 1 + 2 * rng.gen_range(0..n as i64); // scale one row by a unit when coprime
                if (u as u64 % n).gcd(&n) == 1 {
                    rows2[0] = rows2[0].iter().map(|x| x * u).collect();
                }
                assert_eq!(ZnMat::from_rows(n, &rows2).howell(), h);
                for v in all_vectors(n, c) {
                    assert_eq!(a.span_contains(&v), span.contains(&v));
                }
            }
        }
    }

    #[test]
    fn nullspace_and_solve() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        for &n in &[4u64, 6, 8, 12] {
            for _ in 0..30 {
                let (r, c) = (rng.gen_range(1..4), rng.gen_range(1..4));
                let rows: Vec<Vec<i64>> = (0..r)
                    .map(|_| (0..c).map(|_| rng.gen_range(0..n as i64)).collect())
                    .collect();
                let a = ZnMat::from_rows(n, &rows);
                let ker: std::collections::BTreeSet<Vec<u64>> = all_vectors(n, c)
                    .into_iter()
                    .filter(|x| a.mul_vec(x).iter().all(|&y| y == 0))
                    .collect();
                let gens = a.nullspace();
                let gens_i: Vec<Vec<i64>> = gens
                    .iter()
                    .map(|v| v.iter().map(|&x| x as i64).collect())
                    .collect();
                let got = if gens.is_empty() {
                    [vec![0; c]].into_iter().collect()
                } else {
                    span_brute(&ZnMat::from_rows(n, &gens_i))
                };
                assert_eq!(got, ker);
                for b in all_vectors(n, r) {
                    let consistent = all_vectors(n, c).into_iter().any(|x| a.mul_vec(&x) == b);
                    match a.solve(&b) {
                        Some(x) => assert_eq!(a.mul_vec(&x), b),
                        None => assert!(!consistent),
                    }
                }
            }
        }
    }

    #[test]
    fn smith_and_hermite() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..60 {
            let (r, c) = (rng.gen_range(1..5), rng.gen_range(1..5));
            let rows: Vec<Vec<i64>> = (0..r)
                .map(|_| (0..c).map(|_| rng.gen_range(-9..10)).collect())
                .collect();
            let a = IntMat::from_rows(&rows);
            let s = a.smith();
            let d = s.u.mul(&a).mul(&s.v);
            for i in 0..r {
                for j in 0..c {
                    let want = if i == j { s.diag[i].clone() } else { BigInt::zero() };
                    assert_eq!(d.get(i, j), &want);
                }
            }
            let (h, u) = a.hermite();
            assert_eq!(u.mul(&a), h);
            for x in a.kernel() {
                assert!(a.mul_vec(&x).iter().all(|y| y.is_zero()));
            }
            let x0: Vec<BigInt> = (0..c).map(|_| BigInt::from(rng.gen_range(-5..6))).collect();
            let b = a.mul_vec(&x0);
            let x = a.solve(&b).expect("consistent");
            assert_eq!(a.mul_vec(&x), b);
        }
        // Z/2 x Z/6 x Z from relations diag(4, 6)·... : invariants of [[2,4],[6,8]] are 2, 4
        let s = IntMat::from_rows(&[vec![2, 4], vec![6, 8]]).smith();
        assert_eq!(s.diag, vec![BigInt::from(2), BigInt::from(4)]);
    }
}
