#![allow(clippy::needless_range_loop, clippy::type_complexity)]
//! SOS certificate extraction from Gram matrices and dual matrices.

use ncalg::ncsos::{read_dual_blocks, Relaxation};
use ncalg::relations;
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Word};

type Q = BigRational;
type C64 = Complex<f64>;

fn position_of<C: ncalg::Coeff + std::str::FromStr>(words: &[Word], alg: &FreeAlgebra, name: &str) -> usize {
    let target: Polynomial<C> = alg.parse(name).unwrap();
    let w = target.leading_word().unwrap().clone();
    words.iter().position(|u| *u == w).expect("word in index set")
}

/// Hand-built Gram solution of `X + Z − c = Σ G u* u` with `c = −√2`.
fn pauli_setup() -> (
    FreeAlgebra,
    GroebnerBasis<Complex<Q>>,
    Polynomial<Complex<Q>>,
    Vec<Vec<C64>>,
    Vec<Word>,
) {
    let (alg, rels) = relations::pauli::<Complex<Q>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<Q>> = alg.parse("X0 + Z0").unwrap();
    let words = gb.reducer().normal_words_up_to(1);
    let s2 = 2f64.sqrt();
    let mut v = [0.0; 4];
    v[position_of::<Complex<Q>>(&words, &alg, "1")] = s2;
    v[position_of::<Complex<Q>>(&words, &alg, "X0")] = 1.0;
    v[position_of::<Complex<Q>>(&words, &alg, "Z0")] = 1.0;
    let mut g = vec![vec![C64::new(0.0, 0.0); 4]; 4];
    for a in 0..4 {
        for b in 0..4 {
            g[a][b] = C64::new(v[a] * v[b] / (2.0 * s2), 0.0);
        }
    }
    (alg, gb, h, g, words)
}

#[test]
fn certificate_from_gram_model_solution() {
    let (_alg, gb, h, g, _words) = pauli_setup();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let sos = rel.sos_model();
    let gm = &sos.grams[0];
    let mut x = vec![0.0; sos.model.num_vars()];
    for a in 0..4 {
        for b in a..4 {
            x[gm.re_var(a, b)] = g[a][b].re;
        }
    }
    x[sos.bound.unwrap()] = -2f64.sqrt();
    let cert = sos.certificate(&x, 1e-9);
    assert_eq!(cert.num_squares(), 1); // rank-one Gram matrix
    assert!((cert.squares[0].weight - 2f64.sqrt()).abs() < 1e-9); // |v|^2 / (2 sqrt2) = sqrt2
    let gbf = gb.convert::<C64>();
    let hf: Polynomial<C64> = h.convert();
    let ver = cert.verify(gbf.reducer(), &hf);
    assert!((ver.bound + 2f64.sqrt()).abs() < 1e-9, "bound {}", ver.bound);
    assert!(ver.max_residual < 1e-9, "residual {}", ver.residual);
    let text = cert.to_string();
    assert!(text.contains("X0") && text.contains("Z0"));
    // the unreduced sum is a Hermitian polynomial
    let s = cert.sum();
    assert!(s.adjoint().sub_ref(&s).iter().all(|(_, c)| c.norm() < 1e-9));
}

#[test]
fn certificate_from_embedded_dual_matrix() {
    let (_alg, gb, h, g, _words) = pauli_setup();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    assert_eq!(sdp.block_sizes(), vec![8]);
    // a real dual Y with tr(Y embed(F)) = Σ_ab G_ab F_ab is Y = embed(conj G) / 2
    let m = 4;
    let mut y = vec![vec![C64::new(0.0, 0.0); 2 * m]; 2 * m];
    for a in 0..m {
        for b in 0..m {
            let c = g[a][b].conj() * 0.5;
            y[a][b] = C64::new(c.re, 0.0);
            y[a + m][b + m] = C64::new(c.re, 0.0);
            y[a][b + m] = C64::new(-c.im, 0.0);
            y[a + m][b] = C64::new(c.im, 0.0);
        }
    }
    let grams = sdp.dual_grams(&[y.clone()]);
    for a in 0..m {
        for b in 0..m {
            assert!((grams[0][a][b] - g[a][b]).norm() < 1e-12);
        }
    }
    let cert = sdp.certificate(&[y], 1e-9);
    let gbf = gb.convert::<C64>();
    let ver = cert.verify(gbf.reducer(), &h.convert());
    assert!((ver.bound + 2f64.sqrt()).abs() < 1e-9);
    assert!(ver.max_residual < 1e-9);
    // the Hermitian matrix itself is accepted as well
    let cert2 = sdp.certificate(std::slice::from_ref(&g), 1e-9);
    let ver2 = cert2.verify(gbf.reducer(), &h.convert());
    assert!((ver2.bound - ver.bound).abs() < 1e-9);
}

#[test]
fn symmetrized_certificate_and_dual_file_round_trip() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Q> = alg.parse("x^2 + y^2 - x*y").unwrap();
    let rel = Relaxation::level(gb.reducer(), 1).minimize(&h);
    let sos = rel.sos_model();
    let words = &sos.source_words[0];
    let (ix, iy) = (
        position_of::<Q>(words, &alg, "x"),
        position_of::<Q>(words, &alg, "y"),
    );
    // x^2 + y^2 - xy = G with G_xx = G_yy = 1, G_xy = -1/2 (PSD, eigenvalues 1/2 and 3/2)
    let g = &sos.grams[0];
    let mut x = vec![0.0; sos.model.num_vars()];
    x[g.re_var(ix, ix)] = 1.0;
    x[g.re_var(iy, iy)] = 1.0;
    x[g.re_var(ix, iy)] = -0.5;
    let cert = sos.certificate(&x, 1e-12);
    assert_eq!(cert.num_squares(), 2);
    let gbf = gb.convert::<C64>();
    let hf: Polynomial<C64> = h.convert();
    let ver = cert.verify(gbf.reducer(), &hf);
    assert!(ver.bound.abs() < 1e-12 && ver.max_residual < 1e-12);
    // averaging over the swap group keeps the identity
    let group = ncalg::ncsos::signed_permutation_group(
        &[ncalg::ncsos::SignedPermutation::permutation(vec![1, 0])],
        2,
        10,
    );
    let sym = cert.symmetrize(&group);
    assert_eq!(sym.num_squares(), 4);
    let ver = sym.verify(gbf.reducer(), &hf);
    assert!(ver.bound.abs() < 1e-12 && ver.max_residual < 1e-12);
    assert_eq!(sym.pruned(0.5).num_squares(), 2); // the two weight-3/4 squares survive

    // dual block file format
    let dir = std::env::temp_dir().join("ncalg_cert_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("duals.txt");
    std::fs::write(&path, "block 0 2 real\n1 2\n2 3\nblock 1 1 complex\n0.5 -0.25\n").unwrap();
    let blocks = read_dual_blocks(&path).unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0][1][0], C64::new(2.0, 0.0));
    assert_eq!(blocks[1][0][0], C64::new(0.5, -0.25));
    std::fs::write(&path, "block 0 2 real\n1 2\n").unwrap();
    assert!(read_dual_blocks(&path).is_err());
}

/// Gram-form certificates of a symmetry-reduced relaxation are averaged at the
/// level of the Gram matrices: they verify and do not repeat every
/// square for every group element.
#[test]
fn symmetric_gram_certificate_is_averaged_per_gram_matrix() {
    use ncalg::ncsos::SignedPermutation;
    type C = Complex<f64>;
    let n = 4;
    let (alg, rels) = relations::pauli::<C>(n);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let terms: Vec<String> = (0..n)
        .flat_map(|i| ["X", "Y", "Z"].map(|p| format!("{p}{i}*{p}{}", (i + 1) % n)))
        .collect();
    let h: Polynomial<C> = red.reduce(&alg.parse(&terms.join(" + ")).unwrap());
    let shift =
        SignedPermutation::permutation((0..3 * n).map(|v| (3 * ((v / 3 + 1) % n) + v % 3) as _).collect());
    let rel = Relaxation::level(red, 2)
        .minimize(&h)
        .signed_symmetry(vec![shift]);
    let sos = rel.sos_model();
    let sol = sos.model.solve(&ncalg::SolverOptions::default()).unwrap();
    let bound = sol.x[sos.bound.unwrap()];
    assert!((bound + 8.0).abs() < 1e-6, "bound {bound}");
    let cert = sos.certificate(&sol.x, 1e-9);
    let ver = cert.verify(red, &h);
    assert!(
        (ver.bound - bound).abs() < 1e-6 && ver.max_residual < 1e-6,
        "{ver:?}"
    );
    // at most one square per index word (67 here), not |G| = 4 times as many
    assert!(
        cert.num_squares() <= sos.source_words[0].len(),
        "{} squares",
        cert.num_squares()
    );
}

/// The check of a certificate with a large sum (Heisenberg 5-ring,
/// level 2: about 10⁴ word products) reduces it in parallel chunks; the
/// bound matches the solve, and the exact-zero check (thread-local mode,
/// re-entered on the worker threads) agrees with it.
#[test]
fn large_certificate_check_in_parallel_chunks() {
    use ncalg::ncsos::Relaxation;
    use ncalg::{relations, Complex, GroebnerBasis, GroebnerOptions, Polynomial, SolverOptions};
    type C64 = Complex<f64>;
    let n = 5;
    let (alg, rels) = relations::pauli::<C64>(n);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let terms: Vec<String> = (0..n)
        .flat_map(|i| ["X", "Y", "Z"].map(|p| format!("{p}{i}*{p}{}", (i + 1) % n)))
        .collect();
    let h: Polynomial<C64> = gb.reducer().reduce(&alg.parse(&terms.join(" + ")).unwrap());
    let sdp = Relaxation::level(gb.reducer(), 2).minimize(&h).build();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    let cert = sdp.certificate(&sol.duals, 1e-12);
    assert!(cert.sum().iter().count() > 4096, "the parallel path is exercised");
    let ver = cert.verify(gb.reducer(), &h);
    let exact = cert.verify_exact(gb.reducer(), &h);
    assert!(
        (ver.bound - value).abs() < 1e-6,
        "bound {} value {}",
        ver.bound,
        value
    );
    assert!((exact.bound - ver.bound).abs() < 1e-9);
    assert!(ver.max_residual < 1e-6 && exact.max_residual < 1e-6);
    assert!(
        exact.max_residual >= ver.max_residual,
        "exact zeros only count rounding residue too"
    );
}

/// `verify` reduces the squares row by row with memoised left
/// multiplication by letters; with a localizer it agrees with reducing the
/// unreduced sum directly.
#[test]
fn row_wise_check_matches_direct_reduction() {
    use ncalg::SolverOptions;
    let (alg, rels) = relations::pauli::<C64>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<C64> = gb.reducer().reduce(&alg.parse("X0*X1 + Y0*Y1 + Z0").unwrap());
    let g: Polynomial<C64> = alg.parse("1 - Z1").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 2)
        .minimize(&h)
        .localizing_level(&g, 1)
        .build();
    let (value, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    let cert = sdp.certificate(&sol.duals, 1e-12);
    assert!(cert.localizers.iter().any(|l| l.is_some()), "a localizer square");
    let ver = cert.verify(gb.reducer(), &h);
    let mut s = cert.sum();
    for (m, r) in &cert.multipliers {
        s += r.map_coeffs(|c| c * *m);
    }
    let direct = gb.reducer().reduce(&h.sub_ref(&s));
    let diff = direct.sub_ref(&ver.residual);
    assert!(
        (ver.bound - value).abs() < 1e-6,
        "bound {} value {}",
        ver.bound,
        value
    );
    assert!(
        (diff.constant_term().re - ver.bound).abs() < 1e-9,
        "constant terms agree"
    );
    let off: f64 = diff
        .iter()
        .filter(|(w, _)| !w.is_empty())
        .map(|(_, c)| c.norm())
        .fold(0.0, f64::max);
    assert!(off < 1e-9, "residuals agree to rounding: {off}");
}

/// A dual given as its upper triangle only (as sparse solver output
/// is) gives the same certificate as the full matrix.
#[test]
fn triangle_only_duals_are_completed() {
    use ncalg::SolverOptions;
    let n = 3;
    let (alg, rels) = relations::pauli::<C64>(n);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<C64> = gb
        .reducer()
        .reduce(&alg.parse("X0*X1 + Y1*Y2 + Z0*Z2 + X2").unwrap());
    let sdp = Relaxation::level(gb.reducer(), 2).minimize(&h).build();
    let (_, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    let triu: Vec<Vec<Vec<C64>>> = sol
        .duals
        .iter()
        .map(|y| {
            (0..y.len())
                .map(|a| {
                    (0..y.len())
                        .map(|b| if b >= a { y[a][b] } else { C64::new(0.0, 0.0) })
                        .collect()
                })
                .collect()
        })
        .collect();
    let full = sdp.certificate(&sol.duals, 1e-12).verify(gb.reducer(), &h);
    let half = sdp.certificate(&triu, 1e-12).verify(gb.reducer(), &h);
    assert!(
        (full.bound - half.bound).abs() < 1e-9,
        "{} vs {}",
        full.bound,
        half.bound
    );
    assert!(half.max_residual < 1e-6, "residual {}", half.max_residual);
}
