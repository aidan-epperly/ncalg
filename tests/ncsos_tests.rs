#![allow(clippy::needless_range_loop)]
//! Structure tests for the NC-SOS moment relaxation builder.

use ncalg::ncsos::{read_solution, Relaxation};
use ncalg::relations;
use ncalg::{BigRational, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Word};

type Q = BigRational;

#[test]
fn pauli_level1_structure() {
    let (alg, rels) = relations::pauli::<Complex<Q>>(1);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h: Polynomial<Complex<Q>> = alg.parse("Z0").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    assert!(!sdp.is_real());
    assert_eq!(sdp.block_sizes(), vec![8]); // 4 words {1, X, Y, Z}, real embedding
    assert_eq!(sdp.moment_words().len(), 4);
    assert_eq!(sdp.num_vars(), 3); // L(X), L(Y), L(Z); L(1) fixed, all real
    assert_eq!(sdp.objective().iter().filter(|c| **c != 0.0).count(), 1);
    assert_eq!(sdp.objective_constant(), 0.0);
    // the state |0>: L(Z) = 1, L(X) = L(Y) = 0 must give the moment matrix of |0>
    // find which z is L(Z): evaluate with unit vectors
    let red = gb.reducer();
    let z_dir: Vec<f64> = (0..3)
        .map(|j| {
            let mut z = vec![0.0; 3];
            z[j] = 1.0;
            sdp.evaluate(&z, red, &h).unwrap().re
        })
        .collect();
    // exactly one free variable carries L(Z) (the kernel column is 2 e_Z, so the probe gives 2)
    assert_eq!(z_dir.iter().filter(|v| v.abs() > 1e-12).count(), 1);
    let j = z_dir.iter().position(|v| v.abs() > 1e-12).unwrap();
    let mut z = vec![0.0; 3];
    z[j] = 1.0 / z_dir[j];
    assert!((sdp.evaluate(&z, red, &h).unwrap().re - 1.0).abs() < 1e-12);
    let one: Polynomial<Complex<Q>> = alg.one();
    assert!((sdp.evaluate(&z, red, &one).unwrap().re - 1.0).abs() < 1e-12);
    // L(XY) = L(iZ) = i
    let xy: Polynomial<Complex<Q>> = alg.parse("X0*Y0").unwrap();
    let v = sdp.evaluate(&z, red, &xy).unwrap();
    assert!((v.re).abs() < 1e-12 && (v.im - 1.0).abs() < 1e-12);
    // moment matrix for |0>: M[u, v] = <0| u v |0> computed with 2x2 Pauli matrices
    let m = sdp.block_matrix(&z, 0);
    let words = &sdp.block_words()[0];
    type C64 = Complex<f64>;
    let c = |re: f64, im: f64| C64::new(re, im);
    let id = [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];
    let paulis = [
        [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]], // X0 = var 0
        [[c(0.0, 0.0), c(0.0, -1.0)], [c(0.0, 1.0), c(0.0, 0.0)]], // Y0 = var 1
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(-1.0, 0.0)]], // Z0 = var 2
    ];
    let mul = |a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]| -> [[C64; 2]; 2] {
        let mut r = [[c(0.0, 0.0); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
            }
        }
        r
    };
    let word_matrix = |w: &Word| -> [[C64; 2]; 2] {
        let mut r = id;
        for &v in w.iter() {
            r = mul(&r, &paulis[v as usize]);
        }
        r
    };
    for a in 0..4 {
        for b in 0..4 {
            // u* v with u* = reversed word (Paulis are Hermitian)
            let prod = mul(&word_matrix(&words[a].reversed()), &word_matrix(&words[b]));
            let expected = prod[0][0]; // <0| . |0>
            assert!(
                (m[a][b] - expected).norm() < 1e-12,
                "entry {} {}: {} vs {}",
                a,
                b,
                m[a][b],
                expected
            );
        }
    }
}

#[test]
fn real_algebra_moment_matrix_matches_direct_evaluation() {
    // commutative algebra K[x, y], level 2: the moment matrix entries must be
    // L(NF(u v)); check with the point evaluation functional L(p) = p(2, 3)
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Q> = alg.parse("x^2 + y^2 - x*y").unwrap();
    let sdp = Relaxation::level(red, 2).minimize(&h).build();
    assert!(sdp.is_real());
    assert_eq!(sdp.block_sizes(), vec![6]); // 1, x, y, x^2, xy, y^2
                                            // moments of the point (2,3): L(x^a y^b) = 2^a 3^b; solve for z by least squares
                                            // using the fact that y = B z + y0 is injective: build z from moment values via
                                            // the evaluate() linear structure (probe unit vectors).
    let words = sdp.moment_words().to_vec();
    let target: Vec<f64> = words
        .iter()
        .map(|w| w.iter().map(|&v| if v == 0 { 2.0 } else { 3.0 }).product::<f64>())
        .collect();
    // probe
    let n = sdp.num_vars();
    let base: Vec<f64> = sdp.moments(&vec![0.0; n]).iter().map(|(_, c)| c.re).collect();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for j in 0..n {
        let mut z = vec![0.0; n];
        z[j] = 1.0;
        let m: Vec<f64> = sdp
            .moments(&z)
            .iter()
            .map(|(_, c)| c.re - base[words.iter().position(|_| true).unwrap()] * 0.0)
            .collect();
        cols.push(m.iter().zip(&base).map(|(a, b)| a - b).collect());
    }
    // least squares via normal equations (small)
    let mut ata = vec![vec![0.0; n]; n];
    let mut atb = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            ata[i][j] = cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum();
        }
        atb[i] = cols[i]
            .iter()
            .zip(target.iter().zip(&base))
            .map(|(a, (t, b0))| a * (t - b0))
            .sum();
    }
    // gaussian elimination
    let mut z = vec![0.0; n];
    {
        let mut a = ata.clone();
        let mut b = atb.clone();
        for c in 0..n {
            let p = (c..n)
                .max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())
                .unwrap();
            a.swap(c, p);
            b.swap(c, p);
            for r in 0..n {
                if r != c && a[c][c].abs() > 1e-14 {
                    let f = a[r][c] / a[c][c];
                    for k in 0..n {
                        a[r][k] -= f * a[c][k];
                    }
                    b[r] -= f * b[c];
                }
            }
        }
        for c in 0..n {
            z[c] = b[c] / a[c][c];
        }
    }
    let got: Vec<f64> = sdp.moments(&z).iter().map(|(_, c)| c.re).collect();
    for (g, t) in got.iter().zip(&target) {
        assert!((g - t).abs() < 1e-8, "moment mismatch {} vs {}", g, t);
    }
    // the moment matrix is then v v^T with v = (1, 2, 3, 4, 6, 9)
    let m = sdp.block_matrix(&z, 0);
    let v = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0];
    let idx = sdp.block_words()[0].clone();
    assert_eq!(idx.len(), 6);
    for a in 0..6 {
        for b in 0..6 {
            let va = idx[a]
                .iter()
                .map(|&q| if q == 0 { 2.0 } else { 3.0 })
                .product::<f64>();
            let vb = idx[b]
                .iter()
                .map(|&q| if q == 0 { 2.0 } else { 3.0 })
                .product::<f64>();
            assert!(
                (m[a][b].re - va * vb).abs() < 1e-8,
                "{} {}: {} vs {}",
                a,
                b,
                m[a][b].re,
                va * vb
            );
        }
    }
    let _ = v;
    // objective value L(h) = 4 + 9 - 6 = 7
    let val = sdp.objective().iter().zip(&z).map(|(c, z)| c * z).sum::<f64>() + sdp.objective_constant();
    assert!((val - 7.0).abs() < 1e-8, "objective {}", val);
}

#[test]
fn localizing_matrix_and_sdpa_output() {
    let alg = FreeAlgebra::new(["x"]);
    // single generator, no relations: free algebra K[x]
    let gb = GroebnerBasis::<Q>::compute_in(&alg, vec![], &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Q> = alg.parse("x").unwrap();
    let g: Polynomial<Q> = alg.parse("1 - x^2").unwrap(); // -1 <= x <= 1
    let sdp = Relaxation::level(red, 1)
        .localizing_level(&g, 0)
        .minimize(&h)
        .build();
    assert_eq!(sdp.block_sizes(), vec![2, 1]);
    assert_eq!(sdp.block_words()[1], vec![Word::empty()]);
    let dir = std::env::temp_dir().join("ncalg_ncsos_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("problem.dat-s");
    sdp.write_sdpa(&path).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    assert!(lines.next().unwrap().starts_with('"'));
    assert_eq!(lines.next().unwrap().trim(), sdp.num_vars().to_string());
    assert_eq!(lines.next().unwrap().trim(), "2");
    assert_eq!(lines.next().unwrap().trim(), "2 1");
    // maximize variant negates the objective
    let sdp_max = Relaxation::level(red, 1).maximize(&h).build();
    assert_eq!(
        sdp_max.objective().iter().map(|c| -c).collect::<Vec<_>>(),
        sdp.objective()[..sdp_max.num_vars()].to_vec()
    );
    assert_eq!(sdp_max.relaxation_value(-1.0), 1.0);
    // solution reading helper
    let sol = dir.join("sol.txt");
    std::fs::write(&sol, "0.5\n-0.25\n").unwrap();
    assert_eq!(read_solution(&sol).unwrap(), vec![0.5, -0.25]);
}

#[test]
fn symmetry_reduction_commutative_swap() {
    // K[x, y] with the swap x <-> y; objective x^2 + y^2 (invariant)
    let alg = FreeAlgebra::new(["x", "y"]);
    let rels = relations::commutative::<Q>(&alg);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Q> = alg.parse("x^2 + y^2").unwrap();
    let plain = Relaxation::level(red, 2).minimize(&h).build();
    let sym = Relaxation::level(red, 2)
        .minimize(&h)
        .symmetry(vec![vec![1, 0]])
        .build();
    assert_eq!(sym.group_order(), 2);
    // moments of degree <= 4: 15 monomials, 15 variables (14 free) -> 9 orbits (8 free)
    assert_eq!(plain.num_vars(), 14);
    assert!(sym.num_vars() <= 8, "got {}", sym.num_vars());
    // the 6x6 moment matrix splits into the symmetric (4) and antisymmetric (2) parts
    let mut sizes: Vec<usize> = sym.blocks().iter().map(|b| b.size).collect();
    sizes.sort();
    assert_eq!(sizes, vec![2, 4]);
    let total: usize = sym.blocks().iter().map(|b| b.size * b.irrep_dimension).sum();
    assert_eq!(total, 6);
    // invariance violations are detected
    let h2: Polynomial<Q> = alg.parse("x").unwrap();
    let bad = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Relaxation::level(red, 1)
            .minimize(&h2)
            .symmetry(vec![vec![1, 0]])
            .build()
    }));
    assert!(bad.is_err());
}

#[test]
fn symmetry_reduction_pauli_qubit_swap() {
    let (alg, rels) = relations::pauli::<Complex<Q>>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Complex<Q>> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    // swap the two qubits: generator v (X0 Y0 Z0 X1 Y1 Z1) -> (v + 3) mod 6
    let swap: Vec<ncalg::Var> = (0..6).map(|v| (v + 3) % 6).collect();
    let plain = Relaxation::level(red, 1).minimize(&h).build();
    let sym = Relaxation::level(red, 1)
        .minimize(&h)
        .symmetry(vec![swap])
        .build();
    assert!(sym.num_vars() < plain.num_vars());
    let total: usize = sym.blocks().iter().map(|b| b.size * b.irrep_dimension).sum();
    assert_eq!(total, 7); // index words: 1 and the six Paulis
    assert_eq!(sym.blocks().len(), 2);
    // without block diagonalization the single block is kept
    let sym_nb = Relaxation::level(red, 1)
        .minimize(&h)
        .symmetry(vec![(0..6).map(|v| (v + 3) % 6).collect()])
        .block_diagonalize(false)
        .build();
    assert_eq!(sym_nb.blocks().len(), 1);
    assert_eq!(sym_nb.num_vars(), sym.num_vars());
    let dir = std::env::temp_dir().join("ncalg_ncsos_sym");
    std::fs::create_dir_all(&dir).unwrap();
    sym.write_sdpa(dir.join("sym.dat-s")).unwrap();
}

#[test]
fn signed_symmetry_majorana_parity() {
    use ncalg::ncsos::SignedPermutation;
    let (alg, rels) = relations::majorana::<Complex<Q>>(4);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    // even Hamiltonian: γ0 γ1 γ2 γ3 (Hermitian in the algebra)
    let h: Polynomial<Complex<Q>> = alg.parse("g0*g1*g2*g3").unwrap();
    let plain = Relaxation::level(red, 2).minimize(&h).build();
    let parity = SignedPermutation::signs(vec![-1; 4]);
    let sym = Relaxation::level(red, 2)
        .minimize(&h)
        .signed_symmetry(vec![parity.clone()])
        .build();
    assert_eq!(sym.group_order(), 2);
    // index words 1, γ_a, γ_a γ_b: 1 + 4 + 6 = 11 split into even (7) and odd (4)
    let mut sizes: Vec<usize> = sym.blocks().iter().map(|b| b.size).collect();
    sizes.sort();
    assert_eq!(sizes, vec![4, 7]);
    assert_eq!(plain.source_words()[0].len(), 11);
    // odd moments vanish, so there are fewer free variables
    assert!(sym.num_vars() < plain.num_vars());
    // group algebra: composition and word images
    let (s, w) = parity.apply_word(&Word::from_vec(vec![0, 1, 2]));
    assert_eq!(s, -1);
    assert_eq!(w, Word::from_vec(vec![0, 1, 2]));
    let sq = parity.compose(&parity);
    assert_eq!(sq, SignedPermutation::identity(4));
    // a sign pattern that is not a symmetry of h is rejected
    let bad = SignedPermutation::signs(vec![-1, 1, 1, 1]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Relaxation::level(red, 1)
            .minimize(&h)
            .signed_symmetry(vec![bad])
            .build()
    }));
    assert!(r.is_err());
}

#[test]
fn moment_form_matches_eliminated_form_structure() {
    use ncalg::ncsos::SignedPermutation;
    // Majorana N = 4 with parity: moment form keeps all coordinates and adds equalities
    let (alg, rels) = relations::majorana::<Complex<Q>>(4);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<Complex<Q>> = alg.parse("g0*g1*g2*g3").unwrap();
    let relax = Relaxation::level(red, 2)
        .minimize(&h)
        .signed_symmetry(vec![SignedPermutation::signs(vec![-1; 4])]);
    let mf = relax.build_moment_form();
    let sdp = relax.build();
    assert!(mf.complex);
    assert_eq!(mf.num_vars(), 2 * mf.moment_words.len());
    assert_eq!(mf.blocks.len(), 1);
    assert_eq!(mf.blocks[0].0, 11);
    // L(1) = 1 (real and imaginary rows) plus Hermiticity and parity rows
    assert!(mf.equalities.len() >= 2 + mf.moment_words.len());
    assert!(mf
        .equalities
        .iter()
        .any(|(row, rhs)| *rhs == 1.0 && row == &vec![(0usize, 1.0)]));
    // dimension count: variables minus independent equalities equals the eliminated form's variables + 1 fixed
    let dir = std::env::temp_dir().join("ncalg_moment_form");
    std::fs::create_dir_all(&dir).unwrap();
    mf.write_json(dir.join("mf.json")).unwrap();
    let text = std::fs::read_to_string(dir.join("mf.json")).unwrap();
    assert!(text.contains("\"equalities\""));
    assert!(sdp.num_vars() < mf.num_vars());
    // real algebra: no imaginary coordinates
    let alg2 = FreeAlgebra::new(["x", "y"]);
    let rels2 = relations::commutative::<Q>(&alg2);
    let gb2 = GroebnerBasis::compute(rels2, &GroebnerOptions::default());
    let h2: Polynomial<Q> = alg2.parse("x^2 + y^2").unwrap();
    let mf2 = Relaxation::level(gb2.reducer(), 1)
        .minimize(&h2)
        .build_moment_form();
    assert!(!mf2.complex);
    assert_eq!(mf2.num_vars(), mf2.moment_words.len());
    // commutative words are self-adjoint, so the only equality is L(1) = 1
    assert_eq!(mf2.equalities.len(), 1);
}

#[test]
fn star_algebra_with_non_hermitian_generators() {
    // one bosonic mode: a a† - a† a = 1, generators a (non-Hermitian) and n (Hermitian)
    let alg = FreeAlgebra::new_star([("a", false), ("n", true)]);
    assert_eq!(alg.names(), &["a", "a†", "n"]);
    assert_eq!(alg.adjoint_var(0), 1);
    assert!(alg.is_hermitian_var(2));
    let p: Polynomial<Complex<Q>> = alg.parse("i*a*n").unwrap();
    assert_eq!(p.adjoint(), alg.parse("-i*n*a†").unwrap());
    assert_eq!(
        alg.parse::<Complex<Q>>("(a*n)'").unwrap(),
        alg.parse("n*a†").unwrap()
    );
    // relations: commutation and n = a† a; both Hermitian modulo the ideal
    let rels: Vec<Polynomial<Complex<Q>>> = vec![
        alg.parse("a*a† - a†*a - 1").unwrap(),
        alg.parse("n - a†*a").unwrap(),
    ];
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    assert!(gb.is_complete());
    assert!(gb.is_star_closed());
    // the number operator is positive: level-1 relaxation gives L(a† a) >= 0
    let h: Polynomial<Complex<Q>> = alg.parse("a†*a").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    assert!(sdp.num_vars() > 0);
    // index words are 1, a, a†, n (n is eliminated by the basis? depends on ordering) — just check they are normal
    for w in &sdp.source_words()[0] {
        assert!(gb.is_normal_word(w));
    }
    // moment matrix entry (a, a) is L(a† a) and (a†, a†) is L(a a†) = L(a† a) + 1
    let words = &sdp.source_words()[0];
    let ia = words
        .iter()
        .position(|w| **w == *Word::from_vec(vec![0]).as_slice())
        .unwrap();
    let iad = words
        .iter()
        .position(|w| **w == *Word::from_vec(vec![1]).as_slice())
        .unwrap();
    let z = vec![0.3; sdp.num_vars()];
    let m = sdp.block_matrix(&z, 0);
    let red = gb.reducer();
    let l_ada = sdp.evaluate(&z, red, &h).unwrap();
    assert!((m[ia][ia] - l_ada).norm() < 1e-9);
    assert!((m[iad][iad] - (l_ada + Complex::new(1.0, 0.0))).norm() < 1e-9);
}

#[test]
fn partial_basis_still_builds() {
    let alg = FreeAlgebra::new(["x", "y"]);
    let f: Polynomial<Q> = alg.parse("x*y*x - x*y").unwrap();
    let gb = GroebnerBasis::compute(vec![f], &GroebnerOptions::new().max_degree(4));
    assert!(!gb.is_complete());
    let h: Polynomial<Q> = alg.parse("x + y").unwrap();
    let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
    assert_eq!(sdp.block_sizes(), vec![3]);
    assert!(sdp.num_vars() >= 2);
}

/// `constraint_coverage` keeps the prepared data for the next build: the
/// build afterwards is the same as a fresh one, and a builder method in
/// between discards the kept data.
#[test]
fn coverage_then_build_matches_a_fresh_build() {
    type C = Complex<f64>;
    let (alg, rels) = relations::pauli::<C>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<C> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1 + Z0").unwrap();
    let z0: Polynomial<C> = alg.parse("Z0").unwrap();
    let fresh = Relaxation::level(red, 2).minimize(&h).build();
    let rel = Relaxation::level(red, 2).minimize(&h);
    assert!(rel.constraint_coverage().is_empty());
    let after = rel.build();
    assert_eq!(after.num_vars(), fresh.num_vars());
    assert_eq!(after.block_sizes(), fresh.block_sizes());
    assert_eq!(after.objective(), fresh.objective());
    // a constraint added after the coverage check is not lost
    let rel = Relaxation::level(red, 2).minimize(&h);
    let _ = rel.constraint_coverage();
    let constrained = rel.moment_equality(&z0, C::new(0.5, 0.0)).build();
    let expected = Relaxation::level(red, 2)
        .minimize(&h)
        .moment_equality(&z0, C::new(0.5, 0.0))
        .build();
    assert_eq!(constrained.num_vars(), expected.num_vars());
    assert!(constrained.num_vars() < fresh.num_vars());
}

/// A localizer with tiny coefficients is the same constraint.  At
/// `1e-9 (Z0 + Z1 + 1)` every route returned the unconstrained −2.236:
/// the moment route because a block that small fell under the global
/// feasibility tolerance (now: per-block scaling in `SdpModel::solve`), the
/// Gram route because it needs a Gram matrix ~1e9 (now: localizers are
/// normalized by a power of two).
#[test]
fn tiny_localizer_is_the_same_constraint() {
    use ncalg::ncsos::SignedPermutation;
    use ncalg::SolverOptions;
    type C64 = Complex<f64>;
    let (alg, rels) = relations::pauli::<C64>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<C64> = red.reduce(&alg.parse("Z0 + Z1 + X0*X1").unwrap());
    // the qubit swap X0 Y0 Z0 X1 Y1 Z1 -> X1 Y1 Z1 X0 Y0 Z0
    let swap = SignedPermutation::new(vec![3, 4, 5, 0, 1, 2], vec![1; 6]);
    let g0: Polynomial<C64> = alg.parse("Z0 + Z1 + 1").unwrap();
    let o = SolverOptions {
        warnings: false,
        ..Default::default()
    };
    let exact = 0.5 * (3f64.sqrt() - 2.0); // -0.1339746
    for s in [1.0, 1e-9] {
        let g = g0.map_coeffs(|c| c * s);
        let mk = |sym: bool| {
            let r = Relaxation::level(red, 2).minimize(&h).localizing_level(&g, 1);
            if sym {
                r.signed_symmetry(vec![swap.clone()])
            } else {
                r
            }
        };
        for sym in [false, true] {
            let gram = mk(sym).sos_model().model.solve(&o).unwrap().objective;
            let (moment, _) = mk(sym).build().solve(&o).unwrap();
            assert!(
                (gram - exact).abs() < 1e-6,
                "s {s:e}, symmetric {sym}: Gram bound {gram}"
            );
            assert!(
                (moment - exact).abs() < 1e-6,
                "s {s:e}, symmetric {sym}: moment bound {moment}"
            );
        }
    }
}

/// Unparseable tokens are errors, Fortran exponents are numbers.
#[test]
fn read_solution_is_strict() {
    let dir = std::env::temp_dir().join("ncalg_read_solution");
    std::fs::create_dir_all(&dir).unwrap();
    let ok = dir.join("ok.txt");
    std::fs::write(&ok, "1.0D-03\n2.5e1\n-3\n").unwrap();
    assert_eq!(read_solution(&ok).unwrap(), vec![1e-3, 25.0, -3.0]);
    let bad = dir.join("bad.txt");
    std::fs::write(&bad, "x 1.0\n2.0\n").unwrap();
    assert!(read_solution(&bad).is_err());
}

/// A moment that occurs only in a constraint (no matrix, not the
/// objective) is undetermined: `moment_affine` and `evaluate` say so instead
/// of returning a constant.
#[test]
fn undetermined_moments_have_no_expression() {
    use ncalg::SolverOptions;
    type C64 = Complex<f64>;
    let (alg, rels) = relations::pauli::<C64>(3);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<C64> = alg.parse("Z0 + Z1 + Z2").unwrap();
    let p: Polynomial<C64> = red.reduce(&alg.parse("X0*X1*X2 - Y0*Y1*Y2").unwrap());
    let sdp = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&p, C64::new(0.0, 0.0))
        .build();
    let w: Polynomial<C64> = red.reduce(&alg.parse("X0*X1*X2").unwrap());
    assert!(sdp.moment_affine(&w).is_none(), "{:?}", sdp.moment_affine(&w));
    let (_, sol) = sdp.solve(&SolverOptions::default()).unwrap();
    assert!(sdp.evaluate(&sol.x, red, &w).is_none());
    // determined moments are unaffected
    let z0: Polynomial<C64> = alg.parse("Z0").unwrap();
    assert!(sdp.moment_affine(&z0).is_some());
    assert!(sdp.evaluate(&sol.x, red, &z0).is_some());
}
