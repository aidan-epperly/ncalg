//! `Certificate::verify` against `verify_exact` and the optional absolute
//! float floor.  Kept in its own test binary: the test sets the
//! process-wide floor.

use ncalg::ncsos::{Relaxation, Square};
use ncalg::relations;
use ncalg::solver::SolverOptions;
use ncalg::{BigRational, Complex, GroebnerBasis, GroebnerOptions, Polynomial};

type K = Complex<BigRational>;
type C64 = Complex<f64>;

#[test]
fn verify_exact_reports_residuals_below_the_float_tolerance() {
    let (alg, rels) = relations::pauli::<K>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<K> = alg.parse("X0 + Z0").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    let (_, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    // a certificate of h' = 0 whose only square has weight 1e-12:
    // 0 − 1e-12 (1 + X0)* (1 + X0) = −1e-12 (2 + 2 X0) modulo X0² = 1,
    // so the certified bound is −2e-12 and the residual 2e-12 on X0
    let mut cert = sdp.certificate(&sol.duals, 1e-12);
    let p: Polynomial<K> = alg.parse("1 + X0").unwrap();
    cert.squares = vec![Square {
        weight: 1e-12,
        source: 0,
        poly: p.convert(),
    }];
    let gbf = gb.convert::<C64>();
    let zero: Polynomial<C64> = alg.zero();
    // relative zero tests: the residual 2e-12 is genuine, not rounding
    assert_eq!(
        ncalg::coeff::float_tolerance(),
        0.0,
        "no absolute floor by default"
    );
    let ver = cert.verify(gbf.reducer(), &zero);
    assert!((ver.max_residual - 2e-12).abs() < 1e-24, "{}", ver.max_residual);
    assert!((ver.bound + 2e-12).abs() < 1e-24, "{}", ver.bound);
    assert_eq!(ver.float_tolerance, 0.0);

    // an opt-in absolute floor drops it; verify_exact ignores the floor
    // (thread-locally, the global floor is untouched)
    ncalg::set_float_tolerance(1e-10);
    let floored = cert.verify(gbf.reducer(), &zero);
    assert_eq!(floored.max_residual, 0.0, "below the floor: dropped");
    assert_eq!(floored.float_tolerance, 1e-10);
    let exact = cert.verify_exact(gbf.reducer(), &zero);
    assert!(
        (exact.max_residual - 2e-12).abs() < 1e-24,
        "{}",
        exact.max_residual
    );
    assert_eq!(exact.float_tolerance, 0.0);
    assert_eq!(
        ncalg::coeff::float_tolerance(),
        1e-10,
        "the global floor is not changed"
    );
    ncalg::set_float_tolerance(0.0);
}
