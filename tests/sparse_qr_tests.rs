//! Fixture tests for `ncalg::sparse_qr` on two constraint matrices
//! (`elim_small`, `elim_large`) of redundant equality systems from moment
//! relaxations (`fixtures/constraint_elimination/`).
//!
//! The time budgets are asserted in release builds only; run
//! `cargo test --release --test sparse_qr_tests -- --nocapture` to see the
//! measured numbers.

use ncalg::sparse_qr::{QrOptions, SparseQr, SparseRows};
use std::time::Instant;

fn fixture(name: &str) -> SparseRows {
    let path = format!(
        "{}/fixtures/constraint_elimination/{name}_C.mtx",
        env!("CARGO_MANIFEST_DIR")
    );
    SparseRows::read_matrix_market(path).expect("fixture readable")
}

/// The unit functional of the fixtures: `*_u.txt` says `u = e_1`
/// (1-based), i.e. column 0.
fn unit_column(name: &str) -> usize {
    let path = format!(
        "{}/fixtures/constraint_elimination/{name}_u.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("unit functional readable");
    let entries: Vec<(usize, f64)> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('%'))
        .map(|l| {
            let mut it = l.split_whitespace();
            (
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
            )
        })
        .collect();
    assert_eq!(entries, vec![(1, 1.0)], "unit functional is e_1");
    entries[0].0 - 1
}

struct Expect {
    rows: usize,
    cols: usize,
    nnz: usize,
    rank_lo: usize,
    rank_hi: usize,
}

fn check_fixture(name: &str, expect: Expect) {
    let a = fixture(name);
    assert_eq!(
        (a.nrows(), a.ncols, a.nnz()),
        (expect.rows, expect.cols, expect.nnz)
    );
    let unit = unit_column(name);
    let opts = QrOptions {
        rank_tol: 1e-9,
        ..Default::default()
    };

    let t0 = Instant::now();
    let qr = SparseQr::factor(&a, &opts).expect("factor within the fill budget");
    let t_factor = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let basis = qr
        .kernel_basis(&opts)
        .expect("kernel basis within the fill budget");
    let t_kernel = t1.elapsed().as_secs_f64();
    let kernel_nnz: usize = basis.iter().map(Vec::len).sum();

    let t2 = Instant::now();
    let ver = qr.verify(&a, 200);
    let t_verify = t2.elapsed().as_secs_f64();

    let unit_free = qr.free_columns().binary_search(&unit).is_ok();
    let unit_residual = if unit_free {
        SparseQr::relative_residual(&a, &qr.kernel_vector(unit))
    } else {
        f64::NAN
    };
    let min_remainder = qr.remainder_norms().iter().copied().fold(f64::INFINITY, f64::min);
    let max_discarded = qr.dependent_rows().iter().map(|e| e.1).fold(0.0, f64::max);
    println!(
        "{name}: {} x {}, nnz {}; rank_tol 1e-9 ({} build)\n  \
         rank {} (expected {}..={}), kernel dimension {}, r_nnz {}, factor {:.2} s\n  \
         kernel basis: {} vectors, {} nonzeros, {:.2} s\n  \
         column {} (unit functional) free: {}, relative residual of its kernel vector {:.3e}\n  \
         verify(200): max relative residual {:.3e}, diagonal in [{:.3e}, {:.3e}], {:.2} s\n  \
         dependent rows {}, largest discarded remainder {:.3e}, discarded norm {:.3e}, \
         smallest accepted remainder {:.3e}",
        a.nrows(),
        a.ncols,
        a.nnz(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        qr.rank(),
        expect.rank_lo,
        expect.rank_hi,
        qr.free_columns().len(),
        qr.r_nnz(),
        t_factor,
        basis.len(),
        kernel_nnz,
        t_kernel,
        unit + 1,
        unit_free,
        unit_residual,
        ver.max_relative_residual,
        ver.min_diagonal,
        ver.max_diagonal,
        t_verify,
        qr.dependent_rows().len(),
        max_discarded,
        qr.discarded_norm(),
        min_remainder,
    );

    assert!(
        qr.rank() >= expect.rank_lo && qr.rank() <= expect.rank_hi,
        "rank {}",
        qr.rank()
    );
    assert_eq!(qr.rank() + qr.free_columns().len(), a.ncols);
    assert!(unit_free, "the unit column must be free at rank_tol 1e-9");
    assert!(
        unit_residual <= 1e-9,
        "unit kernel vector residual {unit_residual:e}"
    );
    assert!(ver.max_relative_residual <= 1e-8, "{ver:?}");
    assert_eq!(ver.rank, qr.rank());
    assert_eq!(basis.len(), qr.free_columns().len());
    assert!(qr.r_nnz() <= 20_000_000, "r_nnz {}", qr.r_nnz());
    assert!(kernel_nnz <= 50_000_000, "kernel nnz {kernel_nnz}");
    // the kernel vector of the unit column from the basis equals kernel_vector
    let pos = qr.free_columns().binary_search(&unit).unwrap();
    assert_eq!(basis[pos], qr.kernel_vector(unit));
    if !cfg!(debug_assertions) {
        assert!(t_factor < 60.0, "factor took {t_factor:.1} s");
        assert!(t_kernel < 60.0, "kernel basis took {t_kernel:.1} s");
    }
}

#[test]
fn sparse_qr_fixture_elim_small() {
    check_fixture(
        "elim_small",
        Expect {
            rows: 4966,
            cols: 12637,
            nnz: 13083,
            rank_lo: 4255,
            rank_hi: 4270,
        },
    );
}

#[test]
fn sparse_qr_fixture_elim_large() {
    check_fixture(
        "elim_large",
        Expect {
            rows: 22716,
            cols: 31744,
            nnz: 64096,
            rank_lo: 18000,
            rank_hi: 18617,
        },
    );
}

/// The stability of the rank decision on real data: the rank does not
/// increase with `rank_tol` and column 1 stays free on the whole plateau.
#[test]
fn sparse_qr_fixture_rank_plateau_and_monotonicity() {
    let names: &[&str] = if cfg!(debug_assertions) {
        &["elim_small"]
    } else {
        &["elim_small", "elim_large"]
    };
    for name in names {
        let a = fixture(name);
        let mut last = usize::MAX;
        let mut line = String::new();
        for tol in [
            1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
        ] {
            let qr = SparseQr::factor(
                &a,
                &QrOptions {
                    rank_tol: tol,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(qr.rank() <= last, "{name}: rank increased at {tol:e}");
            let free = qr.free_columns().binary_search(&0).is_ok();
            if (1e-13..=1e-5).contains(&tol) {
                assert!(free, "{name}: column 1 not free at {tol:e}");
            }
            line += &format!(
                " {tol:.0e}:{}{}",
                qr.rank(),
                if free { "" } else { "(col 1 pivot)" }
            );
            last = qr.rank();
        }
        println!("{name}: rank by rank_tol:{line}");
    }
}

#[test]
fn sparse_qr_deterministic() {
    let a = fixture("elim_small");
    let opts = QrOptions::default();
    let q1 = SparseQr::factor(&a, &opts).unwrap();
    let q2 = SparseQr::factor(&a, &opts).unwrap();
    assert_eq!(q1.pivot_columns(), q2.pivot_columns());
    assert_eq!(q1.free_columns(), q2.free_columns());
    for k in 0..q1.rank() {
        let (r1, r2) = (q1.r_row(k), q2.r_row(k));
        assert_eq!(r1.len(), r2.len());
        assert!(r1
            .iter()
            .zip(r2)
            .all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits()));
    }
}

#[test]
fn sparse_qr_matrix_market_round_trip() {
    let a = fixture("elim_small");
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("sparse_qr_round_trip.mtx");
    a.write_matrix_market(&path).unwrap();
    let b = SparseRows::read_matrix_market(&path).unwrap();
    assert_eq!(a, b);
    let _ = std::fs::remove_file(&path);
}
