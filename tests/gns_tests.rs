#![allow(clippy::needless_range_loop)]
//! GNS extraction from flat moment matrices of tight relaxations.

use ncalg::ncsos::Relaxation;
use ncalg::relations;
use ncalg::solver::SolverOptions;
use ncalg::{
    Complex, FreeAlgebra, GnsRepresentation, GroebnerBasis, GroebnerOptions, HybridRational, Polynomial, Word,
};

type H = HybridRational;
type K = Complex<H>;
type C64 = Complex<f64>;

fn opts() -> SolverOptions {
    SolverOptions {
        tol: 1e-10,
        ..SolverOptions::default()
    }
}

#[test]
fn single_qubit_representation() {
    let (alg, rels) = relations::pauli::<K>(1);
    let gb = GroebnerBasis::compute(rels.clone(), &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg.parse("X0 + Z0").unwrap();
    // the index set {1, X, Y, Z} is closed under left multiplication, so
    // already level 1 yields the two-dimensional representation
    for level in [1, 2] {
        let sdp = Relaxation::level(red, level).minimize(&h).build();
        let (value, sol) = sdp.solve(&opts()).unwrap();
        assert!(sol.is_optimal());
        let g = sdp.gns(&sol.x, red, 1e-6).unwrap();
        assert_eq!(
            g.dimension, 2,
            "eigenvalues {:?} / {:?}",
            g.kept_eigenvalues, g.dropped_eigenvalues
        );
        assert!(g.flatness_defect < 1e-4, "flatness {}", g.flatness_defect);
        assert!(
            g.relation_defect(&rels) < 1e-4,
            "relations {}",
            g.relation_defect(&rels)
        );
        assert!(g.adjoint_defect() < 1e-6);
        assert!(g.moment_defect(&sdp.moments(&sol.x)) < 1e-4);
        let e = g.expectation(&h);
        assert!(
            (e.re - value).abs() < 1e-5 && e.im.abs() < 1e-6,
            "{} vs {}",
            e,
            value
        );
        // the state is an eigenvector of π(h) with eigenvalue -sqrt2
        let hm = g.evaluate(&h);
        for i in 0..2 {
            let hv: C64 = (0..2).map(|j| hm[i][j] * g.state[j]).sum();
            assert!((hv - g.state[i] * value).norm() < 1e-4);
        }
        // the extracted X and Z anticommute
        let x = &g.generators[0];
        let z = &g.generators[2];
        let xz00 = x[0][0] * z[0][0] + x[0][1] * z[1][0];
        let zx00 = z[0][0] * x[0][0] + z[0][1] * x[1][0];
        assert!((xz00 + zx00).norm() < 1e-4);
    }
}

#[test]
fn heisenberg_pair_recovers_the_singlet_in_dimension_four() {
    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels.clone(), &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let sdp = Relaxation::level(red, 2).minimize(&h).build();
    let (value, sol) = sdp.solve(&opts()).unwrap();
    assert!((value + 3.0).abs() < 1e-7);
    let g = sdp.gns(&sol.x, red, 1e-6).unwrap();
    assert_eq!(g.dimension, 4, "{:?}", g.kept_eigenvalues);
    assert!(g.flatness_defect < 1e-3, "{}", g.flatness_defect);
    assert!(g.relation_defect(&rels) < 1e-4);
    assert!((g.expectation(&h).re + 3.0).abs() < 1e-5);
    // the singlet has <Z0 Z1> = -1
    let zz: Polynomial<K> = alg.parse("Z0*Z1").unwrap();
    assert!((g.expectation(&zz).re + 1.0).abs() < 1e-5);
}

/// Moments of a mixture of point evaluations in the commutative algebra.
fn point_moments(red: &ncalg::Reducer<H>, words: &[Word], points: &[(f64, f64)]) -> Vec<(Word, C64)> {
    let alg = red.algebra();
    let mut out = Vec::new();
    // all normal words up to the degree the construction needs (2 d + 2)
    let d = words.iter().map(|w| w.len()).max().unwrap_or(0);
    for w in red.normal_words_up_to(2 * d + 2) {
        let mut v = 0.0;
        for (x, y) in points {
            let mut m = 1.0;
            for &var in w.as_slice() {
                m *= if alg.name(var) == "x" { *x } else { *y };
            }
            v += m / points.len() as f64;
        }
        out.push((w, C64::new(v, 0.0)));
    }
    out
}

#[test]
fn commutative_points_from_exact_moments() {
    // Lasserre's extraction in the commutative case, from exactly flat moment
    // data: one point, then a two-point mixture (the SDP solver itself returns
    // the analytic center of the optimal face, which need not be flat when
    // the optimal moment matrix is not unique, so the moments are built here)
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<H>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let basis = red.normal_words_up_to(1); // {1, x, y}
    let moments = point_moments(red, &basis, &[(0.5, -2.0)]);
    let g = GnsRepresentation::extract(red, &moments, &basis, 1e-9).unwrap();
    assert_eq!(g.dimension, 1);
    assert!(g.flatness_defect < 1e-9);
    assert!(g.is_real(1e-12));
    assert!((g.generators[0][0][0].re - 0.5).abs() < 1e-9);
    assert!((g.generators[1][0][0].re + 2.0).abs() < 1e-9);

    let moments = point_moments(red, &basis, &[(1.0, 3.0), (-1.0, 0.0)]);
    let g = GnsRepresentation::extract(red, &moments, &basis, 1e-9).unwrap();
    assert_eq!(g.dimension, 2);
    assert!(g.flatness_defect < 1e-9, "{}", g.flatness_defect);
    // π(x) has eigenvalues ±1: trace 0, π(x)^2 = 1; π(x) and π(y) commute
    let x = &g.generators[0];
    assert!((x[0][0] + x[1][1]).norm() < 1e-9);
    let x2 = g.word_matrix(&[0, 0]);
    assert!((x2[0][0].re - 1.0).abs() < 1e-9 && x2[0][1].norm() < 1e-9);
    let xy = g.word_matrix(&[0, 1]);
    let yx = g.word_matrix(&[1, 0]);
    for i in 0..2 {
        for j in 0..2 {
            assert!((xy[i][j] - yx[i][j]).norm() < 1e-9);
        }
    }
    let py: Polynomial<H> = alg.parse("y").unwrap();
    assert!((g.expectation(&py).re - 1.5).abs() < 1e-9);
}

/// The rank decision is invariant under rescaling the variables (Jacobi
/// scaling of the Gram matrix).  Unscaled, the 3-point measure on
/// {1, x, x²} came out 2-dimensional for x at scales 10⁻³ and 10³.
#[test]
fn gns_rank_is_scale_invariant() {
    use ncalg::Reducer;
    let alg = FreeAlgebra::new(["x"]);
    let red: Reducer<f64> = Reducer::new(&alg, Vec::new());
    for s in [1e-3, 1.0, 1e3] {
        let pts: [f64; 3] = [s, 2.0 * s, 3.0 * s];
        let moments: Vec<(Word, Complex<f64>)> = (0..=6)
            .map(|k| {
                let m: f64 = pts.iter().map(|p| p.powi(k as i32)).sum();
                (Word::power(0, k), Complex::new(m, 0.0))
            })
            .collect();
        let basis: Vec<Word> = (0..=2).map(|k| Word::power(0, k)).collect();
        let g = GnsRepresentation::extract(&red, &moments, &basis, 1e-9).unwrap();
        assert_eq!(g.dimension, 3, "scale {s:e}");
        assert!(
            g.flatness_defect < 1e-12,
            "scale {s:e}: flatness {}",
            g.flatness_defect
        );
        for (w, v) in &moments {
            let d = (g.expectation(&alg.monomial::<f64>(w.clone())) - v).norm();
            assert!(d <= 1e-10 * v.norm(), "scale {s:e}, {w:?}: defect {d}");
        }
    }
}
