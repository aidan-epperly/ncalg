//! Dense matrix products that do not use faer's parallel GEMM.
//!
//! The products here split the right factor into column panels, multiply each
//! panel sequentially and run the panels in parallel with rayon.  For the
//! block sizes the solver meets (n ≤ 250) this is faster than faer's own
//! parallel product, whose dispatch overhead dominates there; above n ≈ 300
//! faer's product is as fast or faster, but end to end the difference is a
//! few percent.  faer's factorizations (Cholesky, Hermitian
//! eigendecomposition) are used directly.
//!
//! Note: `private-gemm-x86` 0.1.20 (a dependency of faer's parallel GEMM, now
//! yanked) returned wrong parallel products for some sizes; crates that call
//! faer's parallel products should resolve `private-gemm-x86` ≥ 0.1.22.

use faer::traits::ComplexField;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

/// `a * b`, computed panel by panel with sequential GEMM kernels.
pub(crate) fn matmul<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let mut out = Mat::<T>::zeros(a.nrows(), b.ncols());
    matmul_into(out.as_mut(), a, b);
    out
}

/// `out = a * b`, panel by panel: the column panels of `out` are disjoint,
/// so each is written directly by one sequential kernel (no temporary per
/// panel).
pub(crate) fn matmul_into<T: ComplexField>(out: MatMut<'_, T>, a: MatRef<'_, T>, b: MatRef<'_, T>) {
    assert_eq!(a.ncols(), b.nrows(), "matmul: inner dimensions differ");
    assert!(
        out.nrows() == a.nrows() && out.ncols() == b.ncols(),
        "matmul: output shape"
    );
    let (m, n) = (a.nrows(), b.ncols());
    let threads = rayon::current_num_threads().max(1);
    let work = m.saturating_mul(a.ncols()).saturating_mul(n);
    if !cfg!(feature = "parallel") || threads == 1 || n < 2 || work < 1 << 18 {
        faer::linalg::matmul::matmul(out, Accum::Replace, a, b, T::one_impl(), Par::Seq);
    } else {
        // one panel per thread, at least 16 columns
        #[cfg(feature = "parallel")]
        {
            let panel = n.div_ceil(threads).max(16);
            out.par_col_chunks_mut(panel).enumerate().for_each(|(k, p)| {
                let j0 = k * panel;
                let w = p.ncols();
                faer::linalg::matmul::matmul(p, Accum::Replace, a, b.subcols(j0, w), T::one_impl(), Par::Seq);
            });
        }
    }
}

/// `a * b * c`.
pub(crate) fn matmul3<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>, c: MatRef<'_, T>) -> Mat<T> {
    let ab = matmul(a, b);
    matmul(ab.as_ref(), c)
}

/// `a^† b` (conjugate transpose of `a`).
pub(crate) fn adjoint_mul<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let at = a.adjoint().to_owned();
    matmul(at.as_ref(), b)
}

// ---------------------------------------------------------------- complex Hermitian helpers
type C64 = num_complex::Complex<f64>;

/// All entries real (zero imaginary parts)?
fn is_real(a: &Mat<C64>) -> bool {
    (0..a.ncols()).all(|j| (0..a.nrows()).all(|i| a[(i, j)].im == 0.0))
}

fn real_part(a: &Mat<C64>) -> Mat<f64> {
    Mat::<f64>::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)].re)
}

fn complexify(a: &Mat<f64>) -> Mat<C64> {
    Mat::<C64>::from_fn(a.nrows(), a.ncols(), |i, j| C64::new(a[(i, j)], 0.0))
}

/// `a b`.  Real inputs (as the representations of signed-permutation and
/// swap-algebra actions are) are multiplied in real arithmetic, a quarter of
/// the work.
pub(crate) fn mm(a: &Mat<C64>, b: &Mat<C64>) -> Mat<C64> {
    if is_real(a) && is_real(b) {
        return complexify(&matmul(real_part(a).as_ref(), real_part(b).as_ref()));
    }
    matmul(a.as_ref(), b.as_ref())
}

/// `a b c`, in real arithmetic for real inputs.
pub(crate) fn mm3(a: &Mat<C64>, b: &Mat<C64>, c: &Mat<C64>) -> Mat<C64> {
    if is_real(a) && is_real(b) && is_real(c) {
        let (a, b, c) = (real_part(a), real_part(b), real_part(c));
        return complexify(&matmul3(a.as_ref(), b.as_ref(), c.as_ref()));
    }
    matmul3(a.as_ref(), b.as_ref(), c.as_ref())
}

/// Hermitian eigendecomposition: eigenvalues ascending, eigenvectors as
/// columns; in real arithmetic for a real (symmetric) matrix.
pub(crate) fn hermitian_eigen(a: &Mat<C64>) -> (Vec<f64>, Mat<C64>) {
    hermitian_eigen_checked(a, is_real(a)).expect("Hermitian eigendecomposition failed")
}

/// Hermitian eigendecomposition `(eigenvalues ascending, eigenvectors)`, or
/// `None` if it fails.  With `real` the real part of the symmetrized matrix
/// is decomposed in real arithmetic, so that the eigenvectors stay real.
pub(crate) fn hermitian_eigen_checked(a: &Mat<C64>, real: bool) -> Option<(Vec<f64>, Mat<C64>)> {
    let n = a.nrows();
    if real {
        let r = Mat::<f64>::from_fn(n, n, |i, j| 0.5 * (a[(i, j)].re + a[(j, i)].re));
        let e = r.self_adjoint_eigen(faer::Side::Lower).ok()?;
        let s = e.S();
        let u = e.U();
        Some((
            (0..n).map(|k| s[k]).collect(),
            Mat::<C64>::from_fn(n, n, |i, j| C64::new(u[(i, j)], 0.0)),
        ))
    } else {
        let e = a.self_adjoint_eigen(faer::Side::Lower).ok()?;
        let s = e.S();
        Some(((0..n).map(|k| s[k].re).collect(), e.U().to_owned()))
    }
}

/// Eigenvalues of a Hermitian matrix (real symmetric when `real`, read from
/// the real parts), `None` if the eigensolver fails.
pub(crate) fn hermitian_eigenvalues_checked(a: &Mat<C64>, real: bool) -> Option<Vec<f64>> {
    let n = a.nrows();
    if real {
        let r = Mat::<f64>::from_fn(n, n, |i, j| 0.5 * (a[(i, j)].re + a[(j, i)].re));
        r.self_adjoint_eigenvalues(faer::Side::Lower).ok()
    } else {
        a.self_adjoint_eigenvalues(faer::Side::Lower).ok()
    }
}

/// `f(A)` for Hermitian `A` through its eigendecomposition (faer has no
/// matrix functions).
pub(crate) fn hermitian_fn(a: &Mat<C64>, f: impl Fn(f64) -> f64) -> Mat<C64> {
    let (vals, u) = hermitian_eigen(a);
    let n = a.nrows();
    let mut d = Mat::<C64>::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = C64::new(f(vals[i]), 0.0);
    }
    let ut = u.adjoint().to_owned();
    mm3(&u, &d, &ut)
}

/// Smallest eigenvalue of a Hermitian matrix given as dense rows.
pub(crate) fn min_eigenvalue(a: &[Vec<C64>]) -> f64 {
    let m = a.len();
    if m == 0 {
        return f64::INFINITY;
    }
    let mut mat = Mat::<C64>::zeros(m, m);
    for r in 0..m {
        for c in 0..m {
            mat[(r, c)] = a[r][c];
        }
    }
    let ev = mat
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("eigenvalue computation failed");
    ev.into_iter().fold(f64::INFINITY, f64::min)
}

/// Minimum-norm least-squares solution of `A x ≈ b` for a real matrix given
/// by sparse rows (`(column, value)` lists), with `n` columns.
///
/// Backward stable and O(n²) memory: a blocked tall-skinny QR of the
/// augmented matrix `[A | b]` (each block of rows is folded into the running
/// triangular factor `R̃` by a Householder QR of `[R̃; block]`), then a
/// truncated SVD of the `n × n` triangle `R`, with `x = V Σ⁺ Uᵀ z` where
/// `z = Qᵀ b` is the last column of `R̃`.  Singular values at most
/// `max(rows, n) ε σ_max` are dropped (LAPACK `gelsd`'s default), which
/// handles dependent columns.  Replaces the normal equations, which square
/// the condition number.
pub(crate) fn least_squares_rows(rows: &[Vec<(usize, f64)>], rhs: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(rows.len(), rhs.len());
    if n == 0 {
        return Vec::new();
    }
    let block = (4 * (n + 1)).max(256);
    let mut r_aug = Mat::<f64>::zeros(0, n + 1);
    for start in (0..rows.len()).step_by(block) {
        let end = (start + block).min(rows.len());
        let top = r_aug.nrows();
        let mut stacked = Mat::<f64>::zeros(top + end - start, n + 1);
        stacked
            .as_mut()
            .submatrix_mut(0, 0, top, n + 1)
            .copy_from(r_aug.as_ref());
        for (i, (row, b)) in rows[start..end].iter().zip(&rhs[start..end]).enumerate() {
            for &(c, v) in row {
                stacked[(top + i, c)] += v;
            }
            stacked[(top + i, n)] = *b;
        }
        let qr = stacked.qr();
        let r = qr.thin_R();
        r_aug = r.to_owned();
    }
    let k = r_aug.nrows().min(n);
    if k == 0 {
        return vec![0.0; n];
    }
    let r = r_aug.as_ref().submatrix(0, 0, k, n).to_owned();
    let z: Vec<f64> = (0..k).map(|i| r_aug[(i, n)]).collect();
    let Ok(svd) = r.thin_svd() else {
        return vec![0.0; n];
    };
    let s = svd.S().column_vector();
    let smax = (0..s.nrows()).map(|i| s[i]).fold(0.0_f64, f64::max);
    let cut = (rows.len().max(n) as f64) * f64::EPSILON * smax;
    let (u, v) = (svd.U(), svd.V());
    let mut x = vec![0.0; n];
    for j in 0..s.nrows() {
        if s[j] <= cut {
            continue;
        }
        let coef = (0..k).map(|i| u[(i, j)] * z[i]).sum::<f64>() / s[j];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi += v[(i, j)] * coef;
        }
    }
    x
}

/// A maximal linearly independent subset of the sparse columns `cols`
/// (indices, ascending), by column-pivoted Householder QR of the columns,
/// scaled to unit norm; columns whose pivot falls below `tol` (relative to
/// the first pivot, 1) are dependent.
///
/// When the union of the supports has many more rows than there are
/// columns, the QR runs on a sparse sign sketch `S B` with `2r + 16`
/// rows instead (`r` columns; 8 random ±1 entries per column of `S`): `S`
/// is injective on the column space with probability one and distorts it
/// by a modest factor, so the dependences are those of `B`, far inside the
/// gap between dependent columns (rounding level) and independent ones.
/// The sketch costs `8 nnz`; the pivoted QR, which is BLAS-2 bound, then
/// runs on `(2r + 16) × r` instead of `U × r`.  The sparse echelon
/// form fills its rows in to `U` entries each and costs about `r² U`
/// scattered map operations.
pub(crate) fn independent_columns(cols: &[Vec<(usize, f64)>], tol: f64) -> Vec<usize> {
    let r_cols = cols.len();
    let mut support: Vec<usize> = cols.iter().flat_map(|c| c.iter().map(|(i, _)| *i)).collect();
    support.sort_unstable();
    support.dedup();
    let row = |i: usize| support.binary_search(&i).expect("in the support");
    let sketch_rows = 2 * r_cols + 16;
    let a = if support.len() > 2 * sketch_rows {
        // a sparse sign sketch: support row u goes to SKETCH_NNZ sketch rows
        // with signs ±1/√SKETCH_NNZ (deterministic, splitmix64)
        const SKETCH_NNZ: usize = 8;
        let mix = |mut z: u64| {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let scale = 1.0 / (SKETCH_NNZ as f64).sqrt();
        let hits: Vec<[(usize, f64); SKETCH_NNZ]> = (0..support.len())
            .map(|u| {
                std::array::from_fn(|t| {
                    let z = mix((u as u64) << 4 | t as u64);
                    (
                        (z % sketch_rows as u64) as usize,
                        if z >> 63 == 1 { scale } else { -scale },
                    )
                })
            })
            .collect();
        let cols_sk: Vec<Vec<f64>> = cols
            .par_iter()
            .map(|c| {
                let norm = c.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
                let mut out = vec![0.0; sketch_rows];
                if norm > 0.0 {
                    for &(i, v) in c {
                        for &(k, sgn) in &hits[row(i)] {
                            out[k] += sgn * v / norm;
                        }
                    }
                }
                out
            })
            .collect();
        Mat::<f64>::from_fn(sketch_rows, r_cols, |k, j| cols_sk[j][k])
    } else {
        let mut a = Mat::<f64>::zeros(support.len(), r_cols);
        for (j, c) in cols.iter().enumerate() {
            let norm = c.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for &(i, v) in c {
                    a[(row(i), j)] += v / norm;
                }
            }
        }
        a
    };
    if a.nrows() == 0 {
        return Vec::new();
    }
    let qr = a.col_piv_qr();
    let r = qr.R();
    // relative to the largest pivot (1 for unit columns, about 1 sketched)
    let r00 = if r.nrows() > 0 && r.ncols() > 0 {
        r[(0, 0)].abs().max(1e-300)
    } else {
        1.0
    };
    let rank = (0..r.nrows().min(r.ncols()))
        .take_while(|&k| r[(k, k)].abs() > tol * r00)
        .count();
    let (fwd, _) = qr.P().arrays();
    let mut chosen: Vec<usize> = fwd[..rank].to_vec();
    chosen.sort_unstable();
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn naive<T: ComplexField + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>(
        a: &Mat<T>,
        b: &Mat<T>,
    ) -> Mat<T> {
        Mat::from_fn(a.nrows(), b.ncols(), |i, j| {
            (0..a.ncols()).fold(T::zero_impl(), |acc, k| acc + a[(i, k)] * b[(k, j)])
        })
    }

    // sizes that exercise uneven panels; integer entries make the products exact
    #[test]
    fn panel_products_are_exact() {
        for n in [361usize, 384] {
            let a = Mat::<Complex<f64>>::from_fn(n, n, |i, j| {
                Complex::new(
                    ((i * 7 + j * 3) % 13) as f64 - 6.0,
                    ((i + 5 * j) % 7) as f64 - 3.0,
                )
            });
            let b = Mat::<Complex<f64>>::from_fn(n, n, |i, j| {
                Complex::new(((i * 5 + j) % 11) as f64 - 5.0, ((2 * i + j) % 5) as f64 - 2.0)
            });
            assert!(
                matmul(a.as_ref(), b.as_ref()) == naive(&a, &b),
                "complex n = {}",
                n
            );
        }
        for n in [826usize, 900] {
            let a = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 3) % 13) as f64 - 6.0);
            let b = Mat::<f64>::from_fn(n, n, |i, j| ((i * 5 + j) % 11) as f64 - 5.0);
            assert!(matmul(a.as_ref(), b.as_ref()) == naive(&a, &b), "real n = {}", n);
        }
    }

    #[test]
    fn independent_columns_keeps_a_basis() {
        // e0, 2 e0, e1, e0 + e1, e5 (+ 1e-14 noise on a dependent one): rank 3
        let cols = vec![
            vec![(0, 1.0)],
            vec![(0, 2.0)],
            vec![(1, 1.0)],
            vec![(0, 1.0), (1, 1.0), (5, 1e-14)],
            vec![(5, 3.0)],
        ];
        let chosen = independent_columns(&cols, 1e-9);
        assert_eq!(chosen.len(), 3);
        assert!(chosen.contains(&4), "{:?}", chosen);
        // the chosen columns span e0 and e1 as well
        let has = |c: usize, i: usize| cols[c].iter().any(|(j, v)| *j == i && v.abs() > 0.5);
        assert!(chosen.iter().any(|&c| has(c, 0)) && chosen.iter().any(|&c| has(c, 1)));
        assert!(independent_columns(&[], 1e-9).is_empty());
    }
}
