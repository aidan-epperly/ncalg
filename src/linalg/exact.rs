//! Dense linear algebra over an exact [`Field`]: reduced row echelon
//! form, rank, null space, solve, inverse and determinant, for any
//! compile-time field type, and the inertia of a symmetric matrix over an
//! ordered field, e.g. `BigRational`, `HybridRational`,
//! [`Cyclo<N>`](crate::cyclo::Cyclo) or [`Fq<P, K>`](crate::ff::Fq).
//!
//! Matrices are row-major `[Vec<F>]` with rows of equal length.  Elimination
//! pivots on the first nonzero entry, which is exact arithmetic's choice and
//! the same convention as [`FqMat`](crate::ff::FqMat), so the two agree
//! entry for entry over a prime field.  The determinant uses Bareiss'
//! fraction-free elimination: every intermediate entry is a minor of the
//! input, which bounds the growth of rationals and cyclotomic coefficients.
//!
//! Floating-point types are refused: first-nonzero pivoting is not backward
//! stable, and dense floating-point linear algebra belongs to faer.
//! [`FqMat`](crate::ff::FqMat) stays a separate implementation because its
//! field is a runtime [`Gf`](crate::ff::Gf) with `u64` encodings and table
//! arithmetic, which a compile-time [`Field`] type cannot express.

use crate::coeff::Field;

fn check<F: Field>(m: &[Vec<F>]) -> usize {
    assert!(
        !F::INEXACT,
        "linalg::exact needs an exact field; use faer for floating point"
    );
    let cols = m.first().map_or(0, |r| r.len());
    assert!(m.iter().all(|r| r.len() == cols), "ragged rows");
    cols
}

/// Reduced row echelon form in place; returns the pivot columns.  Every pivot
/// is 1 and is the only nonzero entry of its column.
pub fn rref<F: Field>(m: &mut [Vec<F>]) -> Vec<usize> {
    let cols = check(m);
    let rows = m.len();
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..cols {
        if r == rows {
            break;
        }
        let Some(p) = (r..rows).find(|&i| !m[i][c].is_zero()) else {
            continue;
        };
        m.swap(r, p);
        let inv = m[r][c].inverse();
        for j in c..cols {
            m[r][j] = m[r][j].mul_ref(&inv);
        }
        for i in 0..rows {
            if i != r && !m[i][c].is_zero() {
                let fac = m[i][c].clone();
                for j in c..cols {
                    let v = m[r][j].mul_ref(&fac);
                    m[i][j].sub_assign_ref(&v);
                }
            }
        }
        pivots.push(c);
        r += 1;
    }
    pivots
}

/// Rank, by row reduction of a copy.
pub fn rank<F: Field>(m: &[Vec<F>]) -> usize {
    rref(&mut m.to_vec()).len()
}

/// Basis of `{x : A x = 0}`: one vector per non-pivot column `f`, with
/// `x_f = 1` and zeros at the other non-pivot columns.
pub fn nullspace<F: Field>(m: &[Vec<F>]) -> Vec<Vec<F>> {
    let cols = check(m);
    let mut r = m.to_vec();
    let piv = rref(&mut r);
    (0..cols)
        .filter(|c| !piv.contains(c))
        .map(|fc| {
            let mut v = vec![F::zero(); cols];
            v[fc] = F::one();
            for (row, &pc) in piv.iter().enumerate() {
                v[pc] = r[row][fc].negated();
            }
            v
        })
        .collect()
}

/// Some `x` with `A x = b` (zero at the non-pivot columns), or `None` if the
/// system is inconsistent.
pub fn solve<F: Field>(m: &[Vec<F>], b: &[F]) -> Option<Vec<F>> {
    let cols = check(m);
    assert_eq!(b.len(), m.len(), "right-hand side length");
    let mut aug: Vec<Vec<F>> = m
        .iter()
        .zip(b)
        .map(|(row, bi)| {
            let mut r = row.clone();
            r.push(bi.clone());
            r
        })
        .collect();
    let piv = rref(&mut aug);
    if piv.last() == Some(&cols) {
        return None;
    }
    let mut x = vec![F::zero(); cols];
    for (row, &pc) in piv.iter().enumerate() {
        x[pc] = aug[row][cols].clone();
    }
    Some(x)
}

/// Inverse of a square matrix, or `None` if it is singular.
pub fn inverse<F: Field>(m: &[Vec<F>]) -> Option<Vec<Vec<F>>> {
    let n = check(m);
    assert_eq!(m.len(), n, "square matrix");
    if n == 0 {
        return Some(Vec::new());
    }
    let mut aug: Vec<Vec<F>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { F::one() } else { F::zero() }));
            r
        })
        .collect();
    let piv = rref(&mut aug);
    if piv.len() < n || piv[n - 1] != n - 1 {
        return None;
    }
    Some(aug.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// Determinant of a square matrix, by Bareiss' fraction-free elimination
/// (`n³` field operations, each intermediate entry a minor of the input).
pub fn det<F: Field>(m: &[Vec<F>]) -> F {
    let n = check(m);
    assert_eq!(m.len(), n, "square matrix");
    if n == 0 {
        return F::one();
    }
    let mut a = m.to_vec();
    let mut negate = false;
    let mut prev = F::one();
    for k in 0..n - 1 {
        if a[k][k].is_zero() {
            let Some(p) = (k + 1..n).find(|&i| !a[i][k].is_zero()) else {
                return F::zero();
            };
            a.swap(k, p);
            negate = !negate;
        }
        for i in k + 1..n {
            for j in k + 1..n {
                // a_ij <- (a_ij a_kk - a_ik a_kj) / a_{k-1,k-1}, exact
                let v = a[i][j].mul_ref(&a[k][k]).sub_ref(&a[i][k].mul_ref(&a[k][j]));
                a[i][j] = v.div_ref(&prev);
            }
        }
        prev = a[k][k].clone();
    }
    let d = a[n - 1][n - 1].clone();
    if negate {
        d.negated()
    } else {
        d
    }
}

/// Inertia `(positive, negative, zero)` of a symmetric matrix over an ordered
/// field (Sylvester's law), by symmetric elimination `A ↦ E A Eᵀ`.  A zero
/// diagonal with a nonzero off-diagonal entry `a_ij` is first fixed by adding
/// row and column `j` to `i`, which makes `a_ii = 2 a_ij ≠ 0` (characteristic
/// zero).  The signature is `positive − negative`.
pub fn inertia<F: Field + PartialOrd>(m: &[Vec<F>]) -> (usize, usize, usize) {
    let n = check(m);
    assert_eq!(m.len(), n, "square matrix");
    assert!(
        (0..n).all(|i| (0..i).all(|j| m[i][j] == m[j][i])),
        "symmetric matrix"
    );
    let mut a = m.to_vec();
    let (mut pos, mut neg) = (0, 0);
    let mut live: Vec<usize> = (0..n).collect();
    while !live.is_empty() {
        let p = match live.iter().position(|&i| !a[i][i].is_zero()) {
            Some(p) => live[p],
            None => {
                let Some((i, j)) = live
                    .iter()
                    .flat_map(|&i| live.iter().map(move |&j| (i, j)))
                    .find(|&(i, j)| !a[i][j].is_zero())
                else {
                    break;
                };
                // row_i += row_j, then col_i += col_j
                for k in 0..n {
                    let v = a[j][k].clone();
                    a[i][k].add_assign_ref(&v);
                }
                for k in 0..n {
                    let v = a[k][j].clone();
                    a[k][i].add_assign_ref(&v);
                }
                i
            }
        };
        let d = a[p][p].clone();
        if d > F::zero() {
            pos += 1;
        } else {
            neg += 1;
        }
        live.retain(|&i| i != p);
        for &i in &live {
            if a[i][p].is_zero() {
                continue;
            }
            let f = a[i][p].div_ref(&d);
            for &k in &live {
                let v = f.mul_ref(&a[p][k]);
                a[i][k].sub_assign_ref(&v);
            }
            a[i][p] = F::zero();
        }
        for &i in &live {
            a[p][i] = F::zero();
        }
    }
    (pos, neg, n - pos - neg)
}

/// Matrix product `A B`.
pub fn mul<F: Field>(a: &[Vec<F>], b: &[Vec<F>]) -> Vec<Vec<F>> {
    let inner = check(a);
    let cols = check(b);
    assert_eq!(b.len(), inner, "inner dimensions");
    a.iter()
        .map(|row| {
            (0..cols)
                .map(|j| {
                    let mut s = F::zero();
                    for (k, x) in row.iter().enumerate() {
                        if !x.is_zero() && !b[k][j].is_zero() {
                            s.add_assign_ref(&x.mul_ref(&b[k][j]));
                        }
                    }
                    s
                })
                .collect()
        })
        .collect()
}
