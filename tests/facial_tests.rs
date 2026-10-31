//! Problems without a strictly feasible point: the exact common-kernel
//! presolve, `Relaxation::eigenstate_reduction` and the numerical facial
//! reduction (`SolverOptions::facial_reduction`).

use ncalg::ncsos::Relaxation;
use ncalg::relations;
use ncalg::sdp::{Affine, SdpModel};
use ncalg::solver::{SolverOptions, Status};
use ncalg::{BigRational, Complex, GroebnerBasis, GroebnerOptions, Polynomial};

type Q = BigRational;
type K = Complex<Q>;

fn one() -> K {
    Complex::new(Q::from(ncalg::BigInt::from(1)), Q::from(ncalg::BigInt::from(0)))
}

/// `L(X0) = 1` makes qubit 0 the +1 eigenstate of X0, so `(1 − X0) w` is in
/// the kernel of every feasible moment matrix: no interior point.  With
/// qubit 0 in |+>, `<Z0 Z1> = 0` and the minimum of `Z0 Z1 + X1` is −1
/// (−√2 without the constraint).  The plain solver gets only about the
/// square root of its tolerance here (3.5e-6).
fn eigenstate_problem<'a>(red: &'a ncalg::Reducer<K>, alg: &ncalg::FreeAlgebra) -> Relaxation<'a, K> {
    let h: Polynomial<K> = alg.parse("Z0*Z1 + X1").unwrap();
    let x0: Polynomial<K> = alg.parse("X0").unwrap();
    Relaxation::level(red, 2).minimize(&h).moment_equality(&x0, one())
}

#[test]
fn eigenstate_reduction_makes_the_problem_well_posed() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let sdp = eigenstate_problem(gb.reducer(), &alg)
        .eigenstate_reduction(true)
        .build();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 1.0).abs() < 1e-7, "{}", value);
    assert!(sol.facial_reduction.is_none());
}

#[test]
fn facial_reduction_recovers_the_accuracy() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let sdp = eigenstate_problem(gb.reducer(), &alg).build();
    let opts = SolverOptions {
        facial_reduction: true,
        ..SolverOptions::default()
    };
    let (value, sol) = sdp.solve(&opts).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 1.0).abs() < 1e-7, "{}", value);
    let fr = sol.facial_reduction.expect("facial reduction applied");
    // the kernel (1 − X0) w of the 16 × 16 moment matrix has dimension 8
    assert_eq!(fr.removed.iter().sum::<usize>(), 8);
    assert!(fr.min_eigenvalue > -1e-9 && fr.kernel_residual < 1e-9, "{:?}", fr);
    // the lifted duals are the reduced problem's: not dual feasible here,
    // which is reported
    assert!(fr.dual_residual > 1e-3, "{:?}", fr);
    assert!(sol.warnings.iter().any(|w| w.contains("not dual feasible")));
    // the moment matrix of the returned point is (numerically) PSD
    for b in 0..sdp.blocks().len() {
        let m = sdp.block_matrix(&sol.x, b);
        let n = m.len();
        let fm = faer::Mat::<Complex<f64>>::from_fn(n, n, |i, j| m[i][j]);
        let ev = fm.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        assert!(ev[0] > -1e-9, "block {} eigenvalue {}", b, ev[0]);
    }
}

/// Both qubits in |+> (an orbit of eigenstate constraints under the swap):
/// `<X0 X1> = 1`, `<Z0 Z1> = 0`, so `Z0 Z1 − X0 X1` has minimum −1.
#[test]
fn eigenstate_reduction_with_symmetry() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg.parse("Z0*Z1 - X0*X1").unwrap();
    let x0: Polynomial<K> = alg.parse("X0").unwrap();
    let x1: Polynomial<K> = alg.parse("X1").unwrap();
    let sdp = Relaxation::level(red, 2)
        .minimize(&h)
        .moment_equality(&x0, one())
        .moment_equality(&x1, one())
        .symmetry(vec![vec![3, 4, 5, 0, 1, 2]])
        .eigenstate_reduction(true)
        .build();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 1.0).abs() < 1e-7, "{}", value);
}

/// A block with a zero row and column: the presolve removes the direction
/// exactly and reports the dual at full size.
#[test]
fn presolve_removes_an_exact_common_kernel() {
    // min x + y  s.t.  [[x, 1, 0], [1, y, 0], [0, 0, 0]] ⪰ 0   (optimum 2)
    let mut m = SdpModel::new();
    let x = m.free_var("x");
    let y = m.free_var("y");
    m.psd_affine(
        3,
        true,
        vec![
            (0, 0, Affine::var(x)),
            (0, 1, Affine::constant(1.0)),
            (1, 1, Affine::var(y)),
        ],
        "M",
    );
    m.minimize(&(Affine::var(x) + Affine::var(y)));
    let sol = m.solve(&SolverOptions::default()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((sol.objective - 2.0).abs() < 1e-7, "{}", sol.objective);
    assert_eq!(sol.duals[0].len(), 3);
    assert!(sol.duals[0][2].iter().all(|v| v.norm() < 1e-12));
}

/// Regularization gives a certified bound on the correct side, looser by
/// about 0.7 √ε here, and estimates that looseness.
#[test]
fn regularization_gives_a_certified_bound() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let opts = SolverOptions {
        regularization: 1e-8,
        ..SolverOptions::default()
    };
    // minimization: optimum −1, bound below it
    let (value, sol) = eigenstate_problem(red, &alg).build().solve(&opts).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!(value <= -1.0 + 1e-9 && value > -1.0 - 2e-4, "{}", value);
    let reg = sol.regularization.expect("regularized");
    let bias = reg.bias_estimate.expect("bias estimate");
    let err = -1.0 - value;
    assert!(
        bias > 0.5 * err && bias < 2.0 * err,
        "bias {} vs error {}",
        bias,
        err
    );
    assert!((reg.extrapolated_objective.unwrap() + 1.0).abs() < 0.2 * err);
    assert!(reg.regularized_objective < value, "{:?}", reg);
    // maximization of the negated objective: optimum 1, bound above it
    let h: Polynomial<K> = alg.parse("-Z0*Z1 - X1").unwrap();
    let x0: Polynomial<K> = alg.parse("X0").unwrap();
    let sdp = Relaxation::level(red, 2)
        .maximize(&h)
        .moment_equality(&x0, one())
        .build();
    let (value, sol) = sdp.solve(&opts).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((1.0 - 1e-9..1.0 + 2e-4).contains(&value), "{}", value);
}
