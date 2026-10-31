//! The sparse Howell form over ℤ/N against the dense algorithm it
//! replaced, brute force, and on large sparse inputs.

use ncalg::zmod::{SparseHowell, SparseRow, ZnMat};
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

/// The dense Howell algorithm `ZnMat::howell` used before the sparse form (column by
/// column: Bezout elimination below the pivot, unit normalization, the
/// annihilator row, reduction above), kept as an independent reference.
fn dense_reference(n: u64, rows: &[Vec<u64>], cols: usize) -> Vec<Vec<u64>> {
    let mulm = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let addm = |a: u64, b: u64| ((a as u128 + b as u128) % n as u128) as u64;
    let comb = |x: &[u64], y: &[u64], s: u64, t: u64| -> Vec<u64> {
        x.iter()
            .zip(y)
            .map(|(&a, &b)| addm(mulm(s, a), mulm(t, b)))
            .collect()
    };
    let xgcd = |a: i128, b: i128| -> (i128, i128, i128) {
        let (mut r0, mut r1, mut s0, mut s1, mut t0, mut t1) = (a, b, 1i128, 0i128, 0i128, 1i128);
        while r1 != 0 {
            let q = r0.div_euclid(r1);
            (r0, r1) = (r1, r0 - q * r1);
            (s0, s1) = (s1, s0 - q * s1);
            (t0, t1) = (t1, t0 - q * t1);
        }
        if r0 < 0 {
            (-r0, -s0, -t0)
        } else {
            (r0, s0, t0)
        }
    };
    let mut rows: Vec<Vec<u64>> = rows
        .iter()
        .filter(|r| r.iter().any(|&x| x != 0))
        .cloned()
        .collect();
    let mut r = 0;
    for j in 0..cols {
        if r == rows.len() {
            break;
        }
        for i in (r + 1)..rows.len() {
            let b = rows[i][j];
            if b == 0 {
                continue;
            }
            let a = rows[r][j];
            let (g, s, t) = xgcd(a as i128, b as i128);
            let md = |x: i128| x.rem_euclid(n as i128) as u64;
            let (s, t, u, v) = (md(s), md(t), md(-(b as i128 / g)), md(a as i128 / g));
            let new_r = comb(&rows[r], &rows[i], s, t);
            let new_i = comb(&rows[r], &rows[i], u, v);
            rows[r] = new_r;
            rows[i] = new_i;
        }
        let a = rows[r][j];
        if a == 0 {
            continue;
        }
        let u = ncalg::zmod::unit_normalizer(a, n);
        rows[r] = rows[r].iter().map(|&x| mulm(u, x)).collect();
        let h = rows[r][j];
        let ann: Vec<u64> = rows[r].iter().map(|&x| mulm(n / h, x)).collect();
        if ann.iter().any(|&x| x != 0) {
            rows.push(ann);
        }
        for i in 0..r {
            let q = rows[i][j] / h;
            if q != 0 {
                rows[i] = comb(&rows[i], &rows[r], 1, n - q);
            }
        }
        r += 1;
    }
    rows.truncate(r);
    rows
}

fn to_sparse(v: &[u64]) -> SparseRow {
    v.iter()
        .enumerate()
        .filter(|(_, &x)| x != 0)
        .map(|(j, &x)| (j, x))
        .collect()
}

fn to_dense(v: &[(usize, u64)], len: usize) -> Vec<u64> {
    let mut out = vec![0; len];
    for &(j, x) in v {
        out[j] = x;
    }
    out
}

fn random_rows(rng: &mut rand::rngs::StdRng, n: u64, r: usize, c: usize, density: f64) -> Vec<Vec<u64>> {
    (0..r)
        .map(|_| {
            (0..c)
                .map(|_| {
                    if rng.gen_bool(density) {
                        rng.gen_range(0..n)
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect()
}

/// A random element of the span: Σ x_i row_i.
fn combination(rng: &mut rand::rngs::StdRng, n: u64, rows: &[Vec<u64>], c: usize) -> Vec<u64> {
    let mut v = vec![0u64; c];
    for row in rows {
        let x = rng.gen_range(0..n);
        for (vj, &rj) in v.iter_mut().zip(row) {
            *vj = ((*vj as u128 + x as u128 * rj as u128) % n as u128) as u64;
        }
    }
    v
}

const MODULI: [u64; 14] = [
    2,
    3,
    4,
    6,
    8,
    9,
    12,
    16,
    30,
    36,
    64,
    97,
    1 << 40,
    (1 << 61) - 1, // a Mersenne prime
];

#[test]
fn equals_the_dense_algorithm() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for &n in MODULI.iter().chain(&[3 * (1 << 40), 2 * 3 * 5 * 7 * 11 * 13 * 4]) {
        for _ in 0..60 {
            let (r, c) = (rng.gen_range(0..9), rng.gen_range(1..9));
            let density = [0.2, 0.5, 1.0][rng.gen_range(0..3)];
            let rows = random_rows(&mut rng, n, r, c, density);
            let want = dense_reference(n, &rows, c);
            let sparse: Vec<SparseRow> = rows.iter().map(|v| to_sparse(v)).collect();
            let h = SparseHowell::from_rows(n, c, &sparse);
            let got: Vec<Vec<u64>> = h.rows().iter().map(|v| to_dense(v, c)).collect();
            assert_eq!(got, want, "n = {n}");
            // ZnMat::howell is the same engine
            let rows_i: Vec<Vec<i64>> = rows
                .iter()
                .map(|v| v.iter().map(|&x| x as i64).collect())
                .collect();
            if c > 0 && n < (1 << 62) && r > 0 {
                assert_eq!(ZnMat::from_rows(n, &rows_i).howell().to_rows(), want);
            }
            // the canonical form does not depend on the generators: shuffled,
            // with redundant combinations, scaled by units
            let mut more: Vec<Vec<u64>> = rows.clone();
            for _ in 0..3 {
                more.push(combination(&mut rng, n, &rows, c));
            }
            for k in (1..more.len()).rev() {
                more.swap(k, rng.gen_range(0..=k));
            }
            if let Some(first) = more.first_mut() {
                let u = (n / 3 + 1..n)
                    .find(|u| num_integer::Integer::gcd(u, &n) == 1)
                    .unwrap_or(1);
                *first = first
                    .iter()
                    .map(|&x| ((x as u128 * u as u128) % n as u128) as u64)
                    .collect();
            }
            let mut h2 = SparseHowell::new(n, c);
            for v in &more {
                h2.insert(&to_sparse(v));
            }
            assert_eq!(h2.rows(), h.rows(), "n = {n}: order and redundancy");
            // every combination is in the span; inserting it does not grow it
            for _ in 0..3 {
                let v = combination(&mut rng, n, &rows, c);
                assert!(h.contains(&to_sparse(&v)));
                assert!(h.reduce(&to_sparse(&v)).is_empty());
                assert!(!h.clone().insert(&to_sparse(&v)));
            }
        }
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

#[test]
fn membership_size_and_normal_forms_by_brute_force() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    for &n in &[4u64, 6, 8, 9, 12] {
        for _ in 0..30 {
            let (r, c) = (rng.gen_range(1..4), rng.gen_range(1..4));
            let rows = random_rows(&mut rng, n, r, c, 0.7);
            let span: BTreeSet<Vec<u64>> = all_vectors(n, r)
                .into_iter()
                .map(|x| {
                    let mut v = vec![0u64; c];
                    for (xi, row) in x.iter().zip(&rows) {
                        for (vj, &rj) in v.iter_mut().zip(row) {
                            *vj = (*vj + xi * rj) % n;
                        }
                    }
                    v
                })
                .collect();
            let h = SparseHowell::from_rows(n, c, &rows.iter().map(|v| to_sparse(v)).collect::<Vec<_>>());
            assert_eq!(h.span_factors().iter().product::<u64>() as usize, span.len());
            let s0: Vec<u64> = span.iter().next().cloned().unwrap();
            for v in all_vectors(n, c) {
                assert_eq!(h.contains(&to_sparse(&v)), span.contains(&v));
                // normal forms are coset representatives
                let w: Vec<u64> = v
                    .iter()
                    .zip(span.iter().nth(span.len() / 2).unwrap_or(&s0))
                    .map(|(a, b)| (a + b) % n)
                    .collect();
                assert_eq!(h.reduce(&to_sparse(&v)), h.reduce(&to_sparse(&w)));
            }
        }
    }
}

#[test]
fn sparse_nullspace_and_solve() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    for &n in &[4u64, 12, 36, 1 << 20] {
        for _ in 0..40 {
            let (r, c) = (rng.gen_range(1..8), rng.gen_range(1..8));
            let rows = random_rows(&mut rng, n, r, c, 0.4);
            let sparse: Vec<SparseRow> = rows.iter().map(|v| to_sparse(v)).collect();
            let times = |x: &[u64]| -> Vec<u64> {
                let mut v = vec![0u64; c];
                for (xi, row) in x.iter().zip(&rows) {
                    for (vj, &rj) in v.iter_mut().zip(row) {
                        *vj = ((*vj as u128 + *xi as u128 * rj as u128) % n as u128) as u64;
                    }
                }
                v
            };
            // left kernel: generators annihilate A, and |ker| = N^r / |row span|
            let ker = SparseHowell::left_nullspace(n, c, &sparse);
            for k in &ker {
                assert!(times(&to_dense(k, r)).iter().all(|&x| x == 0));
            }
            let span = SparseHowell::from_rows(n, c, &sparse).span_factors();
            let kspan = SparseHowell::from_rows(n, r, &ker).span_factors();
            let log = |f: &[u64]| f.iter().map(|&x| (x as f64).log2()).sum::<f64>();
            assert!((log(&span) + log(&kspan) - r as f64 * (n as f64).log2()).abs() < 1e-6);
            // solve x A = b for b in the span, and refuse a vector outside it
            let x0: Vec<u64> = (0..r).map(|_| rng.gen_range(0..n)).collect();
            let b = times(&x0);
            let x = SparseHowell::left_solve(n, c, &sparse, &to_sparse(&b)).expect("b is in the span");
            assert_eq!(times(&to_dense(&x, r)), b);
            let h = SparseHowell::from_rows(n, c, &sparse);
            let out: Vec<u64> = (0..c).map(|_| rng.gen_range(0..n)).collect();
            assert_eq!(
                SparseHowell::left_solve(n, c, &sparse, &to_sparse(&out)).is_some(),
                h.contains(&to_sparse(&out))
            );
        }
    }
}

/// A banded matrix over ℤ/2^32 with 4 entries per row: the sparse form stays
/// sparse and is fast where the dense algorithm is `O(rows · cols · rank)`.
#[test]
fn large_banded_matrix() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let n = 1u64 << 32;
    let size = 3000;
    let rows: Vec<SparseRow> = (0..size)
        .map(|i| (i..(i + 4).min(size)).map(|j| (j, rng.gen_range(1..n))).collect())
        .collect();
    let t = std::time::Instant::now();
    let h = SparseHowell::from_rows(n, size, &rows);
    let canon = h.rows();
    let elapsed = t.elapsed().as_secs_f64();
    assert!(elapsed < 20.0, "{elapsed} s");
    // every row is in the span, and the form is canonical against the dense
    // algorithm on a smaller instance of the same shape
    for r in rows.iter().step_by(97) {
        assert!(h.contains(r));
    }
    let avg = canon.iter().map(|r| r.len()).sum::<usize>() as f64 / canon.len() as f64;
    assert!(avg < 64.0, "fill-in: {avg} entries per row");
    let small = 120;
    let dense_rows: Vec<Vec<u64>> = rows[..small]
        .iter()
        .map(|r| {
            to_dense(
                &r.iter().copied().filter(|&(j, _)| j < small).collect::<Vec<_>>(),
                small,
            )
        })
        .collect();
    let hs = SparseHowell::from_rows(
        n,
        small,
        &dense_rows.iter().map(|v| to_sparse(v)).collect::<Vec<_>>(),
    );
    assert_eq!(
        hs.rows().iter().map(|v| to_dense(v, small)).collect::<Vec<_>>(),
        dense_reference(n, &dense_rows, small)
    );
}
