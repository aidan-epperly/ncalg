//! Sparse matrices stored as entry lists, and sparse Gaussian elimination.
//!
//! [`SparseEchelon`] is the one implementation of incremental sparse
//! Gaussian elimination in ncalg.  It backs [`rref_sparse`] (and through it
//! `SdpModel::eliminate`), the kernel basis of the eliminated moment
//! relaxation, and the choice of independent multipliers in the Gram forms.
//! Rank-revealing elimination of large, redundant floating-point systems is
//! [`crate::sparse_qr`].

use faer::Mat;
use num_complex::Complex;
use std::collections::{BTreeMap, HashMap};

type C64 = Complex<f64>;

/// Sparse Hermitian upper triangle `(row, col, value)`, `row <= col`.
pub type SparseHerm = Vec<(usize, usize, C64)>;

/// Sparse real symmetric upper triangle `(row, col, value)`, `row <= col`.
pub type SparseSym = Vec<(usize, usize, f64)>;

/// Both triangles of the Hermitian matrix stored as the upper triangle `f`:
/// `(r, c, v)` and `(c, r, conj v)`, duplicates summed, zeros dropped, sorted.
pub fn full_entries(f: &SparseHerm) -> Vec<(usize, usize, C64)> {
    let mut map: BTreeMap<(usize, usize), C64> = BTreeMap::new();
    for (r, c, v) in f {
        *map.entry((*r, *c)).or_insert(C64::new(0.0, 0.0)) += *v;
        if r != c {
            *map.entry((*c, *r)).or_insert(C64::new(0.0, 0.0)) += v.conj();
        }
    }
    map.into_iter()
        .filter(|(_, v)| v.norm_sqr() != 0.0)
        .map(|((r, c), v)| (r, c, v))
        .collect()
}

/// The dense `n × n` Hermitian matrix stored as the upper triangle `f`.
pub fn to_dense(f: &SparseHerm, n: usize) -> Mat<C64> {
    let mut m = Mat::<C64>::zeros(n, n);
    for (r, c, v) in full_entries(f) {
        m[(r, c)] = v;
    }
    m
}

/// Squared Frobenius norm of the Hermitian matrix stored as the upper triangle `f`.
pub fn norm_sq(f: &SparseHerm) -> f64 {
    full_entries(f).iter().map(|(_, _, v)| v.norm_sqr()).sum()
}

/// What [`SparseEchelon::insert`] did with a row.
#[derive(Clone, Debug, PartialEq)]
pub enum Inserted {
    /// The row was independent; its pivot column.
    Pivot(usize),
    /// The row reduced to zero (or was zero); the right-hand side left over,
    /// which must vanish for a consistent system.  `zero_row` when the row
    /// was zero to begin with (its right-hand side is then unscaled).
    Dependent {
        /// The leftover right-hand side.
        rhs: f64,
        /// The row was zero before reduction.
        zero_row: bool,
    },
}

/// Incremental sparse row echelon form over ℝ with partial pivoting.
///
/// Every incoming row `Σ a_j x_j = rhs` is scaled to unit max-norm, reduced
/// against the pivot rows found so far, and its pivot is the entry of largest
/// magnitude (smallest column on ties).  Pivoting by position instead (the
/// largest index, as an earlier version of `rref_sparse` and of the
/// relaxation's kernel basis did) divides by whatever coefficient happens to
/// sit there; with coefficients spanning many orders of magnitude that
/// amplified round-off until a consistent system looked inconsistent.
/// Entries at most `tol` (relative to the scaled row) are dropped.
///
/// faer's sparse LU is not a substitute: it factors a whole matrix with a
/// precomputed ordering, while the callers here insert rows one at a time
/// and need to know at once whether each row is independent of the earlier
/// ones (and keep the first independent ones, in order).
#[derive(Clone, Debug)]
pub struct SparseEchelon {
    tol: f64,
    /// (pivot column, row without the pivot column divided by the pivot, rhs);
    /// row k contains no pivot columns of rows created before it but may
    /// contain pivot columns of later rows
    rows: Vec<(usize, BTreeMap<usize, f64>, f64)>,
    pivot_of: HashMap<usize, usize>,
}

impl SparseEchelon {
    /// An empty echelon form; entries at most `tol` (relative to the scaled row) are dropped.
    pub fn new(tol: f64) -> Self {
        SparseEchelon {
            tol,
            rows: Vec::new(),
            pivot_of: HashMap::new(),
        }
    }

    /// Number of independent rows inserted.
    pub fn rank(&self) -> usize {
        self.rows.len()
    }

    /// Insert the row `Σ_j row[j] x_j = rhs`.
    pub fn insert(&mut self, row: BTreeMap<usize, f64>, rhs: f64) -> Inserted {
        let tol = self.tol;
        let scale = row.values().fold(0.0_f64, |m, c| m.max(c.abs()));
        if scale == 0.0 {
            return Inserted::Dependent { rhs, zero_row: true };
        }
        let mut row: BTreeMap<usize, f64> = row
            .into_iter()
            .map(|(j, c)| (j, c / scale))
            .filter(|(_, c)| c.abs() > tol)
            .collect();
        let mut rhs = rhs / scale;
        // Eliminate the pivot columns present in increasing order of their
        // rows' creation: row k contains no pivot columns of earlier rows, so
        // subtracting it introduces only later ones, and each pivot row is
        // applied at most once (a min-heap of pending row indices).
        let mut pending: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = row
            .keys()
            .filter_map(|j| self.pivot_of.get(j).map(|&k| std::cmp::Reverse(k)))
            .collect();
        let mut last = None;
        while let Some(std::cmp::Reverse(k)) = pending.pop() {
            if last == Some(k) {
                continue; // pushed twice
            }
            last = Some(k);
            let (col, prow, prhs) = &self.rows[k];
            let Some(coef) = row.remove(col) else {
                continue; // canceled meanwhile
            };
            for (j, c) in prow {
                let v = row.entry(*j).or_insert(0.0);
                let fresh = *v == 0.0;
                *v -= c * coef;
                if v.abs() <= tol {
                    row.remove(j);
                } else if fresh {
                    if let Some(&kj) = self.pivot_of.get(j) {
                        pending.push(std::cmp::Reverse(kj));
                    }
                }
            }
            rhs -= prhs * coef;
        }
        let Some((&pcol, &pval)) = row
            .iter()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap().then(b.0.cmp(a.0)))
        else {
            return Inserted::Dependent { rhs, zero_row: false };
        };
        let mut normalized: BTreeMap<usize, f64> = row
            .iter()
            .filter(|(j, _)| **j != pcol)
            .map(|(j, c)| (*j, c / pval))
            .collect();
        normalized.retain(|_, c| c.abs() > tol);
        self.pivot_of.insert(pcol, self.rows.len());
        self.rows.push((pcol, normalized, rhs / pval));
        Inserted::Pivot(pcol)
    }

    /// The reduced row echelon form: `(pivot, free-column coefficients, rhs)`
    /// per independent row, sorted by pivot, so that `x_pivot = rhs − Σ_f c_f x_f`.
    pub fn into_rref(self) -> Vec<(usize, BTreeMap<usize, f64>, f64)> {
        let SparseEchelon {
            tol,
            mut rows,
            pivot_of,
        } = self;
        // backward pass in reverse creation order: row k may contain pivot
        // columns of rows created after it, and those are already fully reduced
        for k in (0..rows.len()).rev() {
            let (pcol, row, rhs) = rows[k].clone();
            let mut out: BTreeMap<usize, f64> = BTreeMap::new();
            let mut rhs = rhs;
            for (j, c) in row {
                match pivot_of.get(&j) {
                    Some(&idx) if idx > k => {
                        let (_, rrow, rrhs) = &rows[idx];
                        rhs -= c * rrhs;
                        for (f, cf) in rrow {
                            *out.entry(*f).or_insert(0.0) -= c * cf;
                        }
                    }
                    _ => {
                        *out.entry(j).or_insert(0.0) += c;
                    }
                }
            }
            out.retain(|_, c| c.abs() > tol);
            rows[k] = (pcol, out, rhs);
        }
        rows.sort_by_key(|(p, _, _)| *p);
        rows
    }
}

/// Sparse reduced row echelon form of `rows` (`Σ_j a_j x_j = rhs`).  Returns
/// `(pivot, free-column coefficients, rhs)` per independent row, so that
/// `x_pivot = rhs − Σ_f c_f x_f`.  Fails on an inconsistent system.
pub fn rref_sparse(
    rows: Vec<(BTreeMap<usize, f64>, f64)>,
    tol: f64,
) -> Result<Vec<(usize, BTreeMap<usize, f64>, f64)>, String> {
    // consistency is judged against the right-hand sides' size, never below
    // the absolute 1e-8 used for unit-sized data: with rhs ~1e9 the rounding
    // of an exactly consistent system exceeded it
    let scale = rows.iter().map(|(_, b)| b.abs()).fold(1.0_f64, f64::max);
    let mut ech = SparseEchelon::new(tol);
    for (row, rhs) in rows {
        match ech.insert(row, rhs) {
            Inserted::Pivot(_) => {}
            Inserted::Dependent { rhs, zero_row: true } if rhs.abs() > 1e-8 * scale => {
                return Err(format!("inconsistent linear equalities (0 = {})", rhs));
            }
            Inserted::Dependent { rhs, zero_row: false } if rhs.abs() > (tol * 1e2).max(1e-8) * scale => {
                return Err(format!(
                    "inconsistent linear equalities (0 = {} after reduction)",
                    rhs
                ));
            }
            Inserted::Dependent { .. } => {}
        }
    }
    Ok(ech.into_rref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echelon_ranks_and_full_entries() {
        let row = |v: &[(usize, f64)]| v.iter().copied().collect::<BTreeMap<usize, f64>>();
        let mut e = SparseEchelon::new(1e-12);
        assert!(matches!(
            e.insert(row(&[(0, 1.0), (1, 2.0)]), 0.0),
            Inserted::Pivot(1)
        ));
        assert!(matches!(
            e.insert(row(&[(1, 1.0), (2, 1.0)]), 0.0),
            Inserted::Pivot(_)
        ));
        // dependent: twice the first row
        assert!(matches!(
            e.insert(row(&[(0, 2.0), (1, 4.0)]), 0.0),
            Inserted::Dependent { .. }
        ));
        assert_eq!(e.rank(), 2);
        let f: SparseHerm = vec![
            (0, 1, C64::new(1.0, 2.0)),
            (1, 1, C64::new(3.0, 0.0)),
            (0, 1, C64::new(1.0, 0.0)),
        ];
        let full = full_entries(&f);
        assert_eq!(
            full,
            vec![
                (0, 1, C64::new(2.0, 2.0)),
                (1, 0, C64::new(2.0, -2.0)),
                (1, 1, C64::new(3.0, 0.0))
            ]
        );
        assert_eq!(norm_sq(&f), 8.0 + 8.0 + 9.0);
        assert_eq!(to_dense(&f, 2)[(1, 0)], C64::new(2.0, -2.0));
    }
}

/// Rows of a sparse matrix in compressed form: row `i` is
/// `data[start[i]..start[i + 1]]`.  Built in two passes over the entries
/// (count, then fill): two allocations in all instead of one per row, which
/// matters for millions of short rows.
#[derive(Clone, Debug, Default)]
pub(crate) struct Csr<T> {
    start: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy + Default> Csr<T> {
    /// `entries(emit)` calls `emit(row, value)` for every entry.  It runs
    /// twice and must emit the same entries both times; a row keeps the
    /// order of its entries.
    pub(crate) fn build(n_rows: usize, entries: impl Fn(&mut dyn FnMut(usize, T))) -> Self {
        let mut start = vec![0usize; n_rows + 1];
        entries(&mut |i, _| start[i + 1] += 1);
        for i in 0..n_rows {
            start[i + 1] += start[i];
        }
        let mut fill = start[..n_rows].to_vec();
        let mut data = vec![T::default(); start[n_rows]];
        entries(&mut |i, v| {
            data[fill[i]] = v;
            fill[i] += 1;
        });
        Csr { start, data }
    }

    /// Row `i`.
    pub(crate) fn row(&self, i: usize) -> &[T] {
        &self.data[self.start[i]..self.start[i + 1]]
    }
}
