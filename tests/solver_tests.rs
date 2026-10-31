//! The native interior-point solver on the NC-SOS models whose values are
//! known (and were cross-checked with CVXPY/SCS in `examples/sdp_models.rs`).

use ncalg::ncsos::Relaxation;
use ncalg::relations;
use ncalg::sdp::{Affine, SdpModel};
use ncalg::solver::{SolveProfile, SolverOptions, Status};
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial};
use std::collections::BTreeMap;

type Q = BigRational;
type K = Complex<Q>;
type C64 = Complex<f64>;

/// `Certificate::sum` (computed through the Gram matrices) against the
/// square-by-square expansion `Σ_k λ_k p_k* g p_k`, up to rounding (both
/// sum products that largely cancel, so the reference magnitude is
/// `Σ_k λ_k ‖p_k‖₁² ‖g‖₁`) and up to the float tolerance (default 1e-10):
/// the expansion drops every square's terms below it one by one, where the
/// Gram matrix first adds them up (six squares of weight 1.8e-11 with
/// constant term 1 lose 1.1e-10 of the constant in the expansion).
fn assert_sum_matches_expansion(cert: &ncalg::ncsos::Certificate) {
    let alg = cert.algebra();
    let mut naive: Polynomial<C64> = alg.constant(C64::new(0.0, 0.0));
    for sq in &cert.squares {
        let star = sq.poly.adjoint();
        let term = match &cert.localizers[sq.source] {
            None => &star * &sq.poly,
            Some(g) => &(&star * g) * &sq.poly,
        };
        naive += term.map_coeffs(|c| c * sq.weight);
    }
    let one_norm = |p: &Polynomial<C64>| p.iter().map(|(_, c)| c.norm()).sum::<f64>();
    let mass: f64 = cert
        .squares
        .iter()
        .map(|sq| {
            let g = cert.localizers[sq.source].as_ref().map_or(1.0, one_norm);
            sq.weight * one_norm(&sq.poly).powi(2) * g
        })
        .sum();
    let diff = cert
        .sum()
        .sub_ref(&naive)
        .iter()
        .map(|(_, c)| c.norm())
        .fold(0.0, f64::max);
    let dropped = 1e-10 * cert.num_squares() as f64;
    assert!(
        diff <= 1e-13 * mass + dropped,
        "sum differs from the expansion by {} (magnitude {})",
        diff,
        mass
    );
}

fn opts() -> SolverOptions {
    SolverOptions {
        tol: 1e-9,
        ..SolverOptions::default()
    }
}

/// The profile of a solve: every phase and counter filled, the phases
/// inside the total, and no warning on a small problem.
#[test]
fn profile_accounts_for_the_solve() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<K> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 2).minimize(&h).build();
    let (value, sol) = sdp.solve(&opts()).unwrap();
    assert!((value + 3.0).abs() < 1e-7, "value {}", value);
    let p = &sol.profile;
    assert_eq!(p.solves, 1);
    assert_eq!(p.iterations, sol.iterations);
    assert!(p.iterations > 0 && p.operator_products >= p.iterations);
    assert_eq!(p.kernel_certified + p.kernel_searched, sdp.block_sizes().len());
    assert!(p.total > 0.0 && p.iterations_time() > 0.0 && p.iterations_time() <= p.total);
    assert!(sol.warnings.is_empty());
}

/// The one-off warning: more than a quarter of the solve and over a second.
#[test]
fn one_off_warning_threshold() {
    let p = |total, presolve, iterate| SolveProfile {
        total,
        presolve,
        iterate,
        ..Default::default()
    };
    // e.g. Quantum Max Cut on 9 qubits: presolve 57 s of 63 s
    let w = p(63.0, 57.0, 6.0).one_off_warning().expect("a warning");
    assert!(w.contains("presolve 57.0 s"), "{}", w);
    assert!(
        p(10.0, 2.0, 8.0).one_off_warning().is_none(),
        "a fifth of the solve"
    );
    assert!(p(1.2, 0.9, 0.3).one_off_warning().is_none(), "under a second");
}

#[test]
fn pauli_x_plus_z_all_forms() {
    let (alg, rels) = relations::pauli::<K>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<K> = alg.parse("X0 + Z0").unwrap();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let s2 = 2f64.sqrt();
    // eliminated moment form
    let sdp = rel.build();
    let (value, sol) = sdp.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + s2).abs() < 1e-7, "moment value {}", value);
    // the optimal state has <X> = <Z> = -1/sqrt2
    let x0: Polynomial<K> = alg.parse("X0").unwrap();
    let lx = sdp.evaluate(&sol.x, gb.reducer(), &x0).unwrap();
    assert!((lx.re + 1.0 / s2).abs() < 1e-6 && lx.im.abs() < 1e-7);
    // certificate from the solver's dual
    let cert = sdp.certificate(&sol.duals, 1e-7);
    assert_sum_matches_expansion(&cert);
    let gbf = gb.convert::<C64>();
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!((ver.bound + s2).abs() < 1e-6, "certified {}", ver.bound);
    assert!(ver.max_residual < 1e-6, "residual {}", ver.max_residual);
    // moment form with explicit equalities (eliminated internally)
    let mf = rel.build_moment_form().to_model();
    let sol = mf.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((sol.objective + s2).abs() < 1e-7);
    // Gram form
    let sos = rel.sos_model();
    let sol = sos.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((sol.objective + s2).abs() < 1e-7, "sos value {}", sol.objective);
    let cert = sos.certificate(&sol.x, 1e-7);
    assert_sum_matches_expansion(&cert);
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!((ver.bound + s2).abs() < 1e-6 && ver.max_residual < 1e-6);
    // Gram-weight model with the exact shift: weights sum to 2 (G = v v^T/(2 sqrt2) allows w = diag)
    let gw = rel.gram_weight_model(s2);
    let sol = gw.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    let w: f64 = gw.weight_values(&sol.x).iter().sum();
    assert!(w > 1.0 && w < 3.0, "weights {}", w);
    assert_sum_matches_expansion(&gw.certificate(&sol.x, 1e-7));
    assert!(gw.model.check(&sol.x, 1e-6).feasible());
}

#[test]
fn heisenberg_pair_with_symmetry_and_constraint() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let (value, sol) = Relaxation::level(red, 1)
        .minimize(&h)
        .build()
        .solve(&opts())
        .unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 3.0).abs() < 1e-7, "{}", value);
    // qubit swap symmetry, block diagonalized: same value, certificate lifts and symmetrizes
    let sym = Relaxation::level(red, 1)
        .minimize(&h)
        .symmetry(vec![vec![3, 4, 5, 0, 1, 2]])
        .build();
    assert_eq!(sym.blocks().len(), 2);
    let (value, sol) = sym.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 3.0).abs() < 1e-7, "{}", value);
    let cert = sym.certificate(&sol.duals, 1e-7);
    let gbf = gb.convert::<C64>();
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!((ver.bound + 3.0).abs() < 1e-6, "certified {}", ver.bound);
    assert!(ver.max_residual < 1e-6, "residual {}", ver.max_residual);
    // moment constraint <Z0 Z1> = -1/3: optimum -7/3
    let zz: Polynomial<K> = alg.parse("Z0*Z1").unwrap();
    let third = Complex::new(Q::new((-1).into(), 3.into()), Q::from(ncalg::BigInt::from(0)));
    let (value, sol) = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&zz, third)
        .build()
        .solve(&opts())
        .unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 7.0 / 3.0).abs() < 1e-7, "{}", value);
}

/// The Gram forms carry multipliers for moment constraints, so they solve
/// the constrained problem and certify it.  Two qubits, total
/// `Sz = 0` sector imposed as `L((Z0 + Z1) w) = 0` for every level-1 word
/// `w` (non-Hermitian products such as `(Z0 + Z1) X0`), and the Heisenberg
/// pair with `<Z0 Z1> = -1/3`.
#[test]
fn gram_forms_with_moment_constraints() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let gbf = gb.convert::<C64>();
    let zero = Complex::new(Q::from(ncalg::BigInt::from(0)), Q::from(ncalg::BigInt::from(0)));
    // sector: -Z0 Z1 is +1 on Sz = 0, its unconstrained minimum is -1
    let h: Polynomial<K> = alg.parse("-Z0*Z1 + 0.5*X0*X1 + 0.5*Y0*Y1").unwrap();
    let sz: Polynomial<K> = alg.parse("Z0 + Z1").unwrap();
    let mut rel = Relaxation::level(red, 1).minimize(&h);
    for w in ["1", "X0", "Y0", "Z0", "X1", "Y1", "Z1"] {
        let w: Polynomial<K> = alg.parse(w).unwrap();
        rel = rel.moment_equality(&(&sz * &w), zero.clone());
    }
    let (moment_value, sol) = rel.build().solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    // ground state of h on {|01>, |10>}: 1 - 1 = 0 (the triplet-zero/singlet split)
    assert!((moment_value - 0.0).abs() < 1e-6, "moment value {}", moment_value);
    let sos = rel.sos_model();
    assert!(!sos.multipliers.is_empty());
    let sol = sos.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!(
        (sol.objective - moment_value).abs() < 1e-6,
        "sos {} vs moment {}",
        sol.objective,
        moment_value
    );
    let cert = sos.certificate(&sol.x, 1e-9);
    assert!(!cert.multipliers.is_empty());
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!((ver.bound - moment_value).abs() < 1e-6, "certified {}", ver.bound);
    assert!(ver.max_residual < 1e-6, "residual {}", ver.max_residual);
    // Gram-weight model: feasible for shift above -value, certificate of h + shift on the sector
    let shift = 0.5 - moment_value;
    let cs = rel.gram_weight_model(shift);
    let sol = cs.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    let ver = cs.certificate(&sol.x, 1e-9).verify(gbf.reducer(), &h.convert());
    assert!(
        (ver.bound + shift).abs() < 1e-6 && ver.max_residual < 1e-6,
        "{} {}",
        ver.bound,
        ver.max_residual
    );

    // Heisenberg pair with <Z0 Z1> = -1/3: -7/3 (unconstrained -3)
    let h: Polynomial<K> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let zz: Polynomial<K> = alg.parse("Z0*Z1").unwrap();
    let third = Complex::new(Q::new((-1).into(), 3.into()), Q::from(ncalg::BigInt::from(0)));
    let rel = Relaxation::level(red, 1).minimize(&h).moment_equality(&zz, third);
    let sos = rel.sos_model();
    assert_eq!(
        sos.multipliers.len(),
        1,
        "a Hermitian constraint has one real multiplier"
    );
    let sol = sos.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!(
        (sol.objective + 7.0 / 3.0).abs() < 1e-6,
        "sos value {}",
        sol.objective
    );
    let ver = sos.certificate(&sol.x, 1e-9).verify(gbf.reducer(), &h.convert());
    assert!((ver.bound + 7.0 / 3.0).abs() < 1e-6 && ver.max_residual < 1e-6);
    // a repeated constraint: the dependent multiplier is dropped
    let twice = Complex::new(Q::new((-2).into(), 3.into()), Q::from(ncalg::BigInt::from(0)));
    let zz2 = &zz + &zz;
    let sos = rel.moment_equality(&zz2, twice).sos_model();
    assert_eq!(sos.multipliers.len(), 1);
    let sol = sos.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!(
        (sol.objective + 7.0 / 3.0).abs() < 1e-6,
        "dependent multipliers: {}",
        sol.objective
    );
}

/// Symmetry-reduced Gram forms agree with the full ones: value, diagonal weight
/// sum and a certificate that verifies.  Heisenberg triangle with the
/// rotation only (complex irreducibles, complex block bases), with the full
/// S3 (two-dimensional irreducibles), and with S3 but no block-diagonalization;
/// two qubits with the swap symmetry and a sector constraint.
#[test]
fn symmetry_reduced_gram_forms() {
    let (alg, rels) = relations::pauli::<K>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let gbf = gb.convert::<C64>();
    let h: Polynomial<K> = alg
        .parse("X0*X1 + Y0*Y1 + Z0*Z1 + X1*X2 + Y1*Y2 + Z1*Z2 + X0*X2 + Y0*Y2 + Z0*Z2 + 0.3*Z0 + 0.3*Z1 + 0.3*Z2")
        .unwrap();
    let hf: Polynomial<C64> = h.convert();
    let rotation: Vec<ncalg::Var> = vec![3, 4, 5, 6, 7, 8, 0, 1, 2];
    let reflection: Vec<ncalg::Var> = vec![0, 1, 2, 6, 7, 8, 3, 4, 5];
    let full_sos = Relaxation::level(red, 1).minimize(&h).sos_model();
    let sol = full_sos.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    let value = sol.objective;
    let shift = 0.7 - value;
    let full_cs = Relaxation::level(red, 1).minimize(&h).gram_weight_model(shift);
    let sol = full_cs.model.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    let weight = sol.objective;
    for (gens, blocks) in [
        (vec![rotation.clone()], true),
        (vec![rotation.clone(), reflection.clone()], true),
        (vec![rotation.clone(), reflection.clone()], false),
    ] {
        let rel = Relaxation::level(red, 1)
            .minimize(&h)
            .symmetry(gens.clone())
            .block_diagonalize(blocks);
        let sos = rel.sos_model();
        if blocks {
            assert!(sos.grams.len() > 1, "{} generators: not reduced", gens.len());
        }
        let sol = sos.model.solve(&opts()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        assert!(
            (sol.objective - value).abs() < 1e-6,
            "{} gens: sos {} vs {}",
            gens.len(),
            sol.objective,
            value
        );
        let ver = sos.certificate(&sol.x, 1e-10).verify(gbf.reducer(), &hf);
        assert!(
            (ver.bound - value).abs() < 1e-6 && ver.max_residual < 1e-6,
            "{} {}",
            ver.bound,
            ver.max_residual
        );
        let cs = rel.gram_weight_model(shift);
        let sol = cs.model.solve(&opts()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        assert!(
            (sol.objective - weight).abs() < 1e-6,
            "{} gens, blocks {}: gram-weight {} vs {}",
            gens.len(),
            blocks,
            sol.objective,
            weight
        );
        let w: f64 = cs.weight_values(&sol.x).iter().sum();
        assert!((w - weight).abs() < 1e-6);
        let ver = cs.certificate(&sol.x, 1e-10).verify(gbf.reducer(), &hf);
        assert!(
            (ver.bound + shift).abs() < 1e-6 && ver.max_residual < 1e-6,
            "{} {}",
            ver.bound,
            ver.max_residual
        );
    }

    // two qubits, swap symmetry, Sz = 0 sector
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let gbf = gb.convert::<C64>();
    let zero = Complex::new(Q::from(ncalg::BigInt::from(0)), Q::from(ncalg::BigInt::from(0)));
    let h: Polynomial<K> = alg
        .parse("-Z0*Z1 + 0.5*X0*X1 + 0.5*Y0*Y1 + 0.2*X0 + 0.2*X1")
        .unwrap();
    let hf: Polynomial<C64> = h.convert();
    let sz: Polynomial<K> = alg.parse("Z0 + Z1").unwrap();
    let swap: Vec<ncalg::Var> = vec![3, 4, 5, 0, 1, 2];
    let make = |sym: bool| {
        let mut rel = Relaxation::level(red, 1).minimize(&h);
        for w in ["1", "X0", "Y0", "Z0", "X1", "Y1", "Z1"] {
            let w: Polynomial<K> = alg.parse(w).unwrap();
            rel = rel.moment_equality(&(&sz * &w), zero.clone());
        }
        if sym {
            rel.symmetry(vec![swap.clone()]).symmetrize_constraints(true)
        } else {
            rel
        }
    };
    let (moment_value, _) = make(true).build().solve(&opts()).unwrap();
    let shift = 0.5 - moment_value;
    let mut weights = Vec::new();
    for sym in [false, true] {
        let rel = make(sym);
        let sos = rel.sos_model();
        let sol = sos.model.solve(&opts()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        assert!(
            (sol.objective - moment_value).abs() < 1e-6,
            "sector sym {}: {} vs {}",
            sym,
            sol.objective,
            moment_value
        );
        let ver = sos.certificate(&sol.x, 1e-10).verify(gbf.reducer(), &hf);
        assert!(
            (ver.bound - moment_value).abs() < 1e-6 && ver.max_residual < 1e-6,
            "{} {}",
            ver.bound,
            ver.max_residual
        );
        let cs = rel.gram_weight_model(shift);
        let sol = cs.model.solve(&opts()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        weights.push(sol.objective);
        let ver = cs.certificate(&sol.x, 1e-10).verify(gbf.reducer(), &hf);
        assert!(
            (ver.bound + shift).abs() < 1e-6 && ver.max_residual < 1e-6,
            "{} {}",
            ver.bound,
            ver.max_residual
        );
    }
    assert!(
        (weights[0] - weights[1]).abs() < 1e-6,
        "sector gram-weight {:?}",
        weights
    );
}

#[test]
fn commutative_polynomial_optimization() {
    // min x^2 + y^2 - x - y over R^2 is -1/2 (x = y = 1/2); level 1 is exact
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Q> = alg.parse("x^2 + y^2 - x - y").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    assert!(sdp.is_real());
    let (value, sol) = sdp.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 0.5).abs() < 1e-7, "{}", value);
    let x: Polynomial<Q> = alg.parse("x").unwrap();
    assert!((sdp.evaluate(&sol.x, gb.reducer(), &x).unwrap().re - 0.5).abs() < 1e-6);
    // a localizing constraint x <= 0 moves the optimum to y = 1/2, x = 0: -1/4
    let g: Polynomial<Q> = alg.parse("-x").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1)
        .minimize(&h)
        .localizing_level(&g, 0)
        .build();
    let (value, sol) = sdp.solve(&opts()).unwrap();
    assert_eq!(sol.status, Status::Optimal);
    assert!((value + 0.25).abs() < 1e-7, "{}", value);
    // its certificate has squares on the localizer
    let cert = sdp.certificate(&sol.duals, 1e-12);
    assert!(cert.squares.iter().any(|sq| cert.localizers[sq.source].is_some()));
    assert_sum_matches_expansion(&cert);
    let gbf = gb.convert::<C64>();
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!(
        (ver.bound + 0.25).abs() < 1e-6 && ver.max_residual < 1e-6,
        "{:?}",
        ver.bound
    );
}

#[test]
fn certificate_with_two_dimensional_irreps() {
    // Heisenberg triangle with the full S3 symmetry (rotation + reflection):
    // the index set carries two-dimensional irreps, so the block-diagonalized
    // dual must be lifted back correctly
    let (alg, rels) = relations::pauli::<K>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg
        .parse("X0*X1 + Y0*Y1 + Z0*Z1 + X1*X2 + Y1*Y2 + Z1*Z2 + X0*X2 + Y0*Y2 + Z0*Z2")
        .unwrap();
    let rotation: Vec<ncalg::Var> = vec![3, 4, 5, 6, 7, 8, 0, 1, 2];
    let reflection: Vec<ncalg::Var> = vec![0, 1, 2, 6, 7, 8, 3, 4, 5];
    let gbf = gb.convert::<C64>();
    let hf: Polynomial<C64> = h.convert();
    let mut values = Vec::new();
    for blocks in [false, true] {
        let sdp = Relaxation::level(red, 1)
            .minimize(&h)
            .symmetry(vec![rotation.clone(), reflection.clone()])
            .block_diagonalize(blocks)
            .build();
        let (value, sol) = sdp.solve(&opts()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        values.push(value);
        if blocks {
            assert!(sdp.blocks().iter().any(|b| b.irrep_dimension == 2));
        }
        let cert = sdp.certificate(&sol.duals, 1e-12);
        assert_sum_matches_expansion(&cert);
        let ver = cert.verify(gbf.reducer(), &hf);
        assert!(
            (ver.bound - value).abs() < 1e-6,
            "blocks {}: certified {} vs value {}",
            blocks,
            ver.bound,
            value
        );
        assert!(
            ver.max_residual < 1e-6,
            "blocks {}: residual {}",
            blocks,
            ver.max_residual
        );
    }
    assert!((values[0] - values[1]).abs() < 1e-6);
}

/// Constraints that are not congruence-invariant, here
/// `c I − M_b ⪰ 0` on every moment-matrix block, must mean the same on the
/// symmetry-reduced blocks as on the full matrix.  With the reduced blocks
/// expressed in a non-orthonormal basis the two values differed.
#[test]
fn operator_norm_constraint_on_reduced_blocks() {
    let (alg, rels) = relations::pauli::<K>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<K> = alg
        .parse("X0*X1 + Y0*Y1 + Z0*Z1 + X1*X2 + Y1*Y2 + Z1*Z2 + X0*X2 + Y0*Y2 + Z0*Z2")
        .unwrap();
    let rotation: Vec<ncalg::Var> = vec![3, 4, 5, 6, 7, 8, 0, 1, 2];
    let reflection: Vec<ncalg::Var> = vec![0, 1, 2, 6, 7, 8, 3, 4, 5];
    let cap = 1.2;
    let solve = |blocks: bool, capped: bool| -> f64 {
        let sdp = Relaxation::level(gb.reducer(), 1)
            .minimize(&h)
            .symmetry(vec![rotation.clone(), reflection.clone()])
            .block_diagonalize(blocks)
            .build();
        if blocks {
            assert!(sdp.blocks().iter().any(|b| b.irrep_dimension == 2));
            assert!(sdp.block_vectors().iter().any(|v| v.is_some()));
        }
        let mut model = sdp.to_model();
        if capped {
            for (b, info) in sdp.blocks().iter().enumerate() {
                if info.source != 0 {
                    continue;
                }
                let neg = |f: &ncalg::ncsos::SparseHerm| -> ncalg::ncsos::SparseHerm {
                    f.iter().map(|(i, j, v)| (*i, *j, -*v)).collect()
                };
                let mut f0: BTreeMap<(usize, usize), C64> = neg(&sdp.hermitian_constant_matrices()[b])
                    .into_iter()
                    .map(|(i, j, v)| ((i, j), v))
                    .collect();
                for i in 0..info.size {
                    *f0.entry((i, i)).or_insert(C64::new(0.0, 0.0)) += C64::new(cap, 0.0);
                }
                model.add_lmi(ncalg::sdp::Lmi {
                    size: info.size,
                    real: info.real,
                    f0: f0.into_iter().map(|((i, j), v)| (i, j, v)).collect(),
                    fj: sdp.hermitian_coefficient_matrices()[b]
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| !f.is_empty())
                        .map(|(j, f)| (j, neg(f)))
                        .collect(),
                    label: format!("cap {}", b),
                });
            }
        }
        let sol = model.solve(&opts()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        sol.objective
    };
    let free = solve(false, false);
    let full = solve(false, true);
    let reduced = solve(true, true);
    // the cap binds, and the reduced program agrees with the full one
    assert!(full > free + 1e-3, "cap does not bind: {} vs {}", full, free);
    assert!(
        (full - reduced).abs() < 1e-6,
        "full {} vs reduced {}",
        full,
        reduced
    );
}

/// The same model with every complex Hermitian block replaced by its real
/// embedding `[[Re, -Im], [Im, Re]]` (built by hand through `psd_affine`).
fn embedded_copy(model: &SdpModel) -> SdpModel {
    let mut e = SdpModel::new();
    for name in model.names() {
        e.free_var(name.clone());
    }
    for lmi in model.lmis() {
        let m = lmi.size;
        let mut entries: BTreeMap<(usize, usize), Affine> = BTreeMap::new();
        let mut add = |r: usize, c: usize, term: Affine| {
            *entries.entry((r, c)).or_insert_with(|| Affine::constant(0.0)) += &term;
        };
        let mut place = |a: usize, b: usize, v: C64, unit: &Affine| {
            if lmi.real {
                add(a, b, unit.clone() * C64::new(v.re, 0.0));
                return;
            }
            if v.re != 0.0 {
                add(a, b, unit.clone() * C64::new(v.re, 0.0));
                add(a + m, b + m, unit.clone() * C64::new(v.re, 0.0));
            }
            if v.im != 0.0 {
                add(a, b + m, unit.clone() * C64::new(-v.im, 0.0));
                if a != b {
                    add(b, a + m, unit.clone() * C64::new(v.im, 0.0));
                }
            }
        };
        let one = Affine::constant(1.0);
        for (a, b, v) in &lmi.f0 {
            place(*a, *b, *v, &one);
        }
        for (j, f) in &lmi.fj {
            let var = Affine::var(*j);
            for (a, b, v) in f {
                place(*a, *b, *v, &var);
            }
        }
        let size = if lmi.real { m } else { 2 * m };
        e.psd_affine(
            size,
            true,
            entries.into_iter().map(|((r, c), a)| (r, c, a)).collect(),
            lmi.label.clone(),
        );
    }
    for (row, rhs) in model.equalities() {
        let mut lhs = Affine::constant(0.0);
        for (j, c) in row {
            lhs += &(Affine::var(*j) * *c);
        }
        e.equal(&lhs, &Affine::constant(*rhs));
    }
    // the stored objective is already negated for maximization
    let mut obj = Affine::constant(model.objective_constant());
    for (j, c) in model.objective().iter().enumerate() {
        if *c != 0.0 {
            obj += &(Affine::var(j) * *c);
        }
    }
    if model.is_minimize() {
        e.minimize(&obj);
    } else {
        e.maximize(&(obj * -1.0));
    }
    e
}

/// Solve natively (complex blocks in complex arithmetic) and through the
/// hand-made real embedding; objective and variables must agree, and the
/// duals too when the optimal dual is unique (`check_duals`; the embedded
/// problem's optimal face is larger otherwise and its analytic center need
/// not un-embed to the native one).
fn compare_native_with_embedding(model: &SdpModel, expected: Option<f64>, label: &str, check_duals: bool) {
    let native = model.solve(&opts()).unwrap();
    let emb = embedded_copy(model).solve(&opts()).unwrap();
    assert_eq!(native.status, Status::Optimal, "{}", label);
    assert_eq!(emb.status, Status::Optimal, "{}", label);
    assert!(
        (native.objective - emb.objective).abs() < 1e-7,
        "{}: native {} vs embedded {}",
        label,
        native.objective,
        emb.objective
    );
    if let Some(v) = expected {
        assert!(
            (native.objective - v).abs() < 1e-7,
            "{}: {} vs {}",
            label,
            native.objective,
            v
        );
    }
    for (a, b) in native.x.iter().zip(&emb.x) {
        assert!(
            (a - b).abs() < 1e-5,
            "{}: variables differ: {} vs {}",
            label,
            a,
            b
        );
    }
    // duals: the native one is the m×m Hermitian matrix G, the embedded one
    // the 2m×2m real Y with G = (Y11 + Y22) + i (Y12 − Y21)
    for (lmi, (g, y)) in model.lmis().iter().zip(native.duals.iter().zip(&emb.duals)) {
        let m = lmi.size;
        assert_eq!(g.len(), m, "{}: native dual has the Hermitian size", label);
        if !check_duals {
            continue;
        }
        for a in 0..m {
            for b in 0..m {
                let unemb = if lmi.real {
                    y[a][b]
                } else {
                    C64::new(y[a][b].re + y[a + m][b + m].re, y[a][b + m].re - y[b][a + m].re)
                };
                assert!(
                    (g[a][b] - unemb).norm() < 1e-5,
                    "{}: dual entry ({}, {}) {} vs {}",
                    label,
                    a,
                    b,
                    g[a][b],
                    unemb
                );
                assert!(
                    (g[a][b] - g[b][a].conj()).norm() < 1e-12,
                    "{}: dual not Hermitian",
                    label
                );
            }
        }
    }
}

#[test]
fn native_complex_blocks_agree_with_the_real_embedding() {
    let s2 = 2f64.sqrt();
    // one qubit: X0 + Z0 in all forms
    let (alg, rels) = relations::pauli::<K>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<K> = alg.parse("X0 + Z0").unwrap();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    compare_native_with_embedding(&rel.build().to_model(), Some(-s2), "qubit moment", true);
    compare_native_with_embedding(&rel.sos_model().model, Some(-s2), "qubit sos", true);
    compare_native_with_embedding(&rel.gram_weight_model(s2).model, None, "qubit gram-weight", false);
    // two qubits: the Heisenberg pair, with the swap symmetry (two blocks) and
    // with a moment constraint (equalities); expected values (as in
    // examples/sdp_models.rs): moment -3, sos -3, moment_zz -7/3
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let rel = Relaxation::level(red, 1).minimize(&h);
    compare_native_with_embedding(&rel.build().to_model(), Some(-3.0), "pair moment", true);
    compare_native_with_embedding(&rel.sos_model().model, Some(-3.0), "pair sos", true);
    let sym = Relaxation::level(red, 1)
        .minimize(&h)
        .symmetry(vec![vec![3, 4, 5, 0, 1, 2]])
        .build();
    compare_native_with_embedding(&sym.to_model(), Some(-3.0), "pair moment, swap symmetry", true);
    let zz: Polynomial<K> = alg.parse("Z0*Z1").unwrap();
    let third = Complex::new(Q::new((-1).into(), 3.into()), Q::from(ncalg::BigInt::from(0)));
    let zz_model = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&zz, third)
        .build_moment_form()
        .to_model();
    assert!(!zz_model.equalities().is_empty());
    compare_native_with_embedding(
        &zz_model,
        Some(-7.0 / 3.0),
        "pair moment with <Z0 Z1> = -1/3",
        true,
    );
    // Gram-weight model: a complex Gram block together with real 1x1 weight blocks
    compare_native_with_embedding(
        &rel.gram_weight_model(3.0).model,
        None,
        "pair gram-weight (mixed blocks)",
        false,
    );
}

/// Standard-form models (the Gram form) solved through their dual LMI agree
/// with the direct solve: value, certificate, and a much smaller Schur
/// complement.  Heisenberg triangle level 1 with a field, the sector
/// pair, and a level-2 two-qubit Gram form.
#[test]
fn gram_forms_through_the_dual() {
    let (alg, rels) = relations::pauli::<K>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let gbf = gb.convert::<C64>();
    for (level, text) in [
        (
            1,
            "X0*X1 + Y0*Y1 + Z0*Z1 + X1*X2 + Y1*Y2 + Z1*Z2 + X0*X2 + Y0*Y2 + Z0*Z2 + 0.3*Z0 + 0.3*Z1",
        ),
        (2, "X0*X1 + Y0*Y1 + 0.5*Z0*Z1 + Z1*Z2 + 0.7*X2"),
    ] {
        let h: Polynomial<K> = alg.parse(text).unwrap();
        let sos = Relaxation::level(red, level).minimize(&h).sos_model();
        let solve = |d: Option<bool>| sos.model.solve(&SolverOptions { dualize: d, ..opts() }).unwrap();
        let (primal, dual, auto) = (solve(Some(false)), solve(Some(true)), solve(None));
        assert_eq!(primal.status, Status::Optimal);
        assert_eq!(dual.status, Status::Optimal);
        assert!(
            (primal.objective - dual.objective).abs() < 1e-6,
            "level {level}: {} vs {}",
            primal.objective,
            dual.objective
        );
        assert!((auto.objective - dual.objective).abs() < 1e-9);
        for sol in [&primal, &dual] {
            let ver = sos.certificate(&sol.x, 1e-10).verify(gbf.reducer(), &h.convert());
            assert!(
                (ver.bound - primal.objective).abs() < 1e-6 && ver.max_residual < 1e-6,
                "level {level}: {} {}",
                ver.bound,
                ver.max_residual
            );
        }
        // the dual has one variable per independent word equation
        let n = sos.model.num_vars();
        let r = sos.model.equalities().len();
        assert!(r < n, "level {level}: {r} equalities, {n} variables");
    }
}

/// The common-kernel presolve tested `‖F v‖` against the block's
/// largest matrix, so a constraint 10¹⁵ times smaller than another
/// matrix of its block was dropped as kernel: optimum 1 instead of 5.
#[test]
fn kernel_presolve_respects_small_matrices() {
    use ncalg::sdp::{Affine, Lmi, SdpModel};
    let c = |x: f64| Complex::new(x, 0.0);
    for big in [1.0, 1e8] {
        let mut m = SdpModel::new();
        let y1 = m.free_var("y1");
        let y2 = m.free_var("y2");
        let mut fj = std::collections::BTreeMap::new();
        fj.insert(y1, vec![(0, 0, c(big))]);
        fj.insert(y2, vec![(1, 1, c(1.0)), (2, 2, c(1e-7))]);
        m.add_lmi(Lmi {
            size: 3,
            real: true,
            f0: vec![(1, 1, c(-1.0)), (2, 2, c(-5e-7))],
            fj,
            label: "scales".into(),
        });
        let mut obj = Affine::zero();
        obj.terms.insert(y1, c(1.0));
        obj.terms.insert(y2, c(1.0));
        m.minimize(&obj);
        let sol = m.solve(&SolverOptions::default()).unwrap();
        assert!(
            (sol.objective - 5.0).abs() < 1e-6,
            "big {big:e}: objective {}",
            sol.objective
        );
    }
}
