//! Performance canary: one instance per structural regime, built, solved
//! and certified, with the solver's profile and counters, written as JSON
//! for `scripts/canary.py` (which compares them with this machine's
//! baseline).  About 15 s in all.
//!
//! `cargo run --release --example canary -- [out.json] [--repeat N]`
//!
//! `--rescale [seeds]` instead solves every instance's SDP with random block
//! scales 10^{±8} and variable scales 10^{±4} and compares the values with
//! the unscaled solve (a scale-invariance check, not part of the baseline).
//!
//! | instance | regime | what it exercises |
//! |---|---|---|
//! | `dense_overlap` | coefficient matrices sharing positions, as in the swap-algebra (QMC) relaxations | presolve kernel search, dense Schur formula |
//! | `kernel_overlap` | the same with a planted common kernel | kernel search and restriction |
//! | `heisenberg6` | Pauli, disjoint supports | build, sparse Schur formula, certificate |
//! | `syk16` | Majorana, complex | build, certificate |
//! | `i3322_l3` | projectors, maximization | build, certificate |
//! | `rosenbrock10` | commuting variables, degree 4 | build, certificate |
//! | `maxcut800` | one large block, rank-one constraints | large-block products, step lengths |
//! | `theta101` | many constraints on a small block | Schur factorization |
//! | `heisenberg8_sym` | signed symmetry, block diagonalized | symmetric build, block-level certificate |
//! | `qmc7_sym` | qubit swap algebra, dihedral symmetry (not monomial) | f64 Gröbner basis, non-monomial build, certificate check |
//! | `rescaled_blocks` | `heisenberg6`'s moment SDP with its blocks scaled by 10⁹ and its variables by 10⁻¹⁰ | scale invariance of the solve: its value must be `heisenberg6`'s |
//! | `tiny_localizer` | Pauli pair with the localizer `10⁻⁹ (Z0 + Z1 + 1)`, Gram (SOS) model | scale invariance of the relaxation: value −0.1339746 as for the unscaled localizer |
//!
//! Counters are deterministic for a given build; times vary with the
//! machine's load, so run it on an otherwise idle machine.

use ncalg::ncsos::{MomentSdp, Relaxation, SdpScalar, SignedPermutation};
use ncalg::sdp::{Affine, Lmi, SdpModel, SparseHerm};
use ncalg::{
    relations, Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, SdpSolution, SolverOptions,
    Var, Word,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::time::Instant;

type C64 = Complex<f64>;

/// One timed run: build, solve, certificate (`post`), and the solution.
struct Run {
    build: f64,
    solve: f64,
    post: f64,
    value: f64,
    certified: Option<f64>,
    num_vars: usize,
    blocks: Vec<usize>,
    sol: SdpSolution,
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn opts() -> SolverOptions {
    SolverOptions {
        warnings: false, // counted in the output instead
        ..Default::default()
    }
}

fn parse<C: ncalg::Coeff>(alg: &FreeAlgebra, s: &str) -> Polynomial<C> {
    alg.parse(s).unwrap_or_else(|e| panic!("parse {s}: {e:?}"))
}

/// A relaxation: Gröbner basis, objective, level, direction, symmetry.
struct Spec<C: SdpScalar> {
    gb: GroebnerBasis<C>,
    h: Polynomial<C>,
    level: usize,
    maximize: bool,
    symmetry: Option<Vec<SignedPermutation>>,
}

fn relaxation<C: SdpScalar>(s: &Spec<C>) -> MomentSdp {
    let rel = Relaxation::level(s.gb.reducer(), s.level);
    let rel = if s.maximize {
        rel.maximize(&s.h)
    } else {
        rel.minimize(&s.h)
    };
    let rel = match &s.symmetry {
        Some(g) => rel.signed_symmetry(g.clone()).block_diagonalize(true),
        None => rel,
    };
    rel.build()
}

/// Build (Gröbner basis and relaxation), solve, and verify the certificate
/// over `C64` (`to64` converts the basis and the objective).
fn pipeline<C: SdpScalar>(
    make: impl Fn() -> Spec<C>,
    to64: impl Fn(&Spec<C>) -> (GroebnerBasis<C64>, Polynomial<C64>),
) -> Run {
    let t = Instant::now();
    let spec = make();
    let sdp = relaxation(&spec);
    let build = secs(t);
    let t = Instant::now();
    let (value, sol) = sdp.solve(&opts()).expect("solve");
    let solve = secs(t);
    let t = Instant::now();
    let cert = sdp.certificate(&sol.duals, 1e-9);
    let (gb64, h64) = to64(&spec);
    let ver = cert.verify(gb64.reducer(), &h64);
    let post = secs(t);
    Run {
        build,
        solve,
        post,
        value,
        certified: Some(ver.bound),
        num_vars: sdp.num_vars(),
        blocks: sdp.block_sizes(),
        sol,
    }
}

fn complex(s: &Spec<C64>) -> (GroebnerBasis<C64>, Polynomial<C64>) {
    (s.gb.clone(), s.h.clone())
}

fn real(s: &Spec<f64>) -> (GroebnerBasis<C64>, Polynomial<C64>) {
    (s.gb.convert::<C64>(), s.h.convert::<C64>())
}

/// Solve a ready-made LMI model.
fn model_run(make: impl Fn() -> SdpModel) -> Run {
    let t = Instant::now();
    let model = make();
    let build = secs(t);
    let t = Instant::now();
    let sol = model.solve(&opts()).expect("solve");
    let solve = secs(t);
    Run {
        build,
        solve,
        post: 0.0,
        value: sol.objective,
        certified: None,
        num_vars: model.num_vars(),
        blocks: model.lmis().iter().map(|l| l.size).collect(),
        sol,
    }
}

/// `Σ_i σ_i · σ_{i+1}` on the n-ring, level 2; with `symmetric`, reduced by
/// translations, the reflection, the axis cycle `X → Y → Z` and the π
/// rotation about z (generators `X_i, Y_i, Z_i` at `3i, 3i + 1, 3i + 2`).
fn heisenberg(n: usize, symmetric: bool) -> Spec<C64> {
    let (alg, rels) = relations::pauli::<C64>(n);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let terms: Vec<String> = (0..n)
        .flat_map(|i| ["X", "Y", "Z"].map(|p| format!("{p}{i}*{p}{}", (i + 1) % n)))
        .collect();
    let h = gb.reducer().reduce(&parse(&alg, &terms.join(" + ")));
    let symmetry = symmetric.then(|| {
        let perm = |f: &dyn Fn(usize, usize) -> (usize, usize)| -> Vec<Var> {
            (0..3 * n)
                .map(|v| {
                    let (i, p) = f(v / 3, v % 3);
                    (3 * i + p) as Var
                })
                .collect()
        };
        let plus = vec![1i8; 3 * n];
        vec![
            SignedPermutation::new(perm(&|i, p| ((i + 1) % n, p)), plus.clone()),
            SignedPermutation::new(perm(&|i, p| (n - 1 - i, p)), plus.clone()),
            SignedPermutation::new(perm(&|i, p| (i, (p + 1) % 3)), plus),
            SignedPermutation::signs((0..3 * n).map(|v| if v % 3 == 2 { 1 } else { -1 }).collect()),
        ]
    });
    Spec {
        gb,
        h,
        level: 2,
        maximize: false,
        symmetry,
    }
}

/// SYK, `q = 4`, `J ~ N(0, 6 / N³)` (seeded), level 2.
fn syk(nmaj: usize, seed: u64) -> Spec<C64> {
    let (alg, rels) = relations::majorana::<C64>(nmaj);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let mut rng = StdRng::seed_from_u64(seed);
    let sd = (6.0 / (nmaj as f64).powi(3)).sqrt();
    let mut terms = Vec::new();
    for a in 0..nmaj {
        for b in a + 1..nmaj {
            for c in b + 1..nmaj {
                for d in c + 1..nmaj {
                    // Box–Muller
                    let (u1, u2): (f64, f64) = (rng.gen::<f64>().max(1e-300), rng.gen());
                    let j = sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    terms.push((
                        Word::from_vec(vec![a as Var, b as Var, c as Var, d as Var]),
                        C64::new(j, 0.0),
                    ));
                }
            }
        }
    }
    let h = gb.reducer().reduce(&Polynomial::from_terms(alg.clone(), terms));
    Spec {
        gb,
        h,
        level: 2,
        maximize: false,
        symmetry: None,
    }
}

/// I3322 in Collins–Gisin form (projectors onto the first outcome), maximized.
fn i3322(level: usize) -> Spec<f64> {
    let names = ["A1", "A2", "A3", "B1", "B2", "B3"];
    let alg = FreeAlgebra::new(names.to_vec());
    let mut rels: Vec<Polynomial<f64>> = names
        .iter()
        .map(|x| parse(&alg, &format!("{x}*{x} - {x}")))
        .collect();
    for a in &names[..3] {
        for b in &names[3..] {
            rels.push(parse(&alg, &format!("{b}*{a} - {a}*{b}")));
        }
    }
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let h = parse(
        &alg,
        "-A1 - 2*B1 - B2 + A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 - A2*B3 + A3*B1 - A3*B2",
    );
    Spec {
        gb,
        h,
        level,
        maximize: true,
        symmetry: None,
    }
}

/// Chained Rosenbrock in commuting variables, level 2 (minimum 0).
fn rosenbrock(n: usize) -> Spec<f64> {
    let names: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    let alg = FreeAlgebra::new(names.clone());
    let gb = GroebnerBasis::compute(relations::commutative::<f64>(&alg), &GroebnerOptions::default());
    let terms: Vec<String> = (0..n - 1)
        .map(|i| format!("100*({} - {}^2)^2 + (1 - {})^2", names[i + 1], names[i], names[i]))
        .collect();
    let h = gb.reducer().reduce(&parse(&alg, &terms.join(" + ")));
    Spec {
        gb,
        h,
        level: 2,
        maximize: false,
        symmetry: None,
    }
}

/// Quantum Max Cut on the n-cycle in the qubit swap algebra (generators
/// `s{i}_{j}`, the S_n relations and the antisymmetrizer on every triple),
/// level 2, with the dihedral symmetry: an action that is not monomial
/// (normal forms of permuted words are combinations), so the f64 Gröbner
/// basis, the non-monomial symmetric build and the dense certificate check
/// all run.
fn qmc_cycle(n: usize) -> Spec<f64> {
    let names: Vec<String> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| format!("s{i}_{j}")))
        .collect();
    let alg = FreeAlgebra::new(names);
    let g = |i: usize, j: usize| alg.gen::<f64>(&format!("s{}_{}", i.min(j), i.max(j)));
    let one: Polynomial<f64> = alg.constant(1.0);
    let mut rels = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            rels.push(&g(i, j) * &g(i, j) - &one);
            for k in 0..n {
                for l in k + 1..n {
                    if (i, j) < (k, l) && i != k && i != l && j != k && j != l {
                        rels.push(&g(i, j) * &g(k, l) - &g(k, l) * &g(i, j));
                    }
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if i != j && j != k && i != k {
                    rels.push(&g(i, j) * &g(j, k) - &g(i, k) * &g(i, j));
                }
            }
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let (a, b, c) = (g(i, j), g(i, k), g(j, k));
                rels.push(&one - &a - &b - &c + &a * &c + &c * &a);
            }
        }
    }
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let mut h: Polynomial<f64> = alg.constant(0.0);
    for i in 0..n {
        h += g(i, (i + 1) % n);
    }
    let h = gb.reducer().reduce(&h);
    // the dihedral group acting on the swaps through the qubits
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
    let index = |a: usize, b: usize| {
        pairs
            .iter()
            .position(|&p| p == (a.min(b), a.max(b)))
            .expect("pair") as Var
    };
    let act =
        |f: &dyn Fn(usize) -> usize| -> Vec<Var> { pairs.iter().map(|&(a, b)| index(f(a), f(b))).collect() };
    let plus = vec![1i8; pairs.len()];
    let symmetry = Some(vec![
        SignedPermutation::new(act(&|i| (i + 1) % n), plus.clone()),
        SignedPermutation::new(act(&|i| (n - i) % n), plus),
    ]);
    Spec {
        gb,
        h,
        level: 2,
        maximize: false,
        symmetry,
    }
}

fn c(x: f64) -> C64 {
    C64::new(x, 0.0)
}

/// `min c·y  s.t.  I + Σ_j y_j F_j ⪰ 0` with `F_j` of `per` random entries
/// (values ±1/4, ±1/2) on positions shared between the `F_j`, and
/// `c_j = tr F_j`, so that `y = 0` and `X = I` are strictly feasible.
fn dense_overlap(n: usize, m: usize, per: usize, seed: u64) -> SdpModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut model = SdpModel::new();
    let y = model.free_vars(m, "y");
    let mut fj: BTreeMap<_, SparseHerm> = BTreeMap::new();
    let mut obj = Affine::zero();
    for &v in &y {
        let mut e: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for _ in 0..per {
            let (a, b) = (rng.gen_range(0..n), rng.gen_range(0..n));
            let val = [0.25, -0.25, 0.5, -0.5][rng.gen_range(0..4)];
            *e.entry((a.min(b), a.max(b))).or_insert(0.0) += val;
        }
        let tr: f64 = e.iter().filter(|((a, b), _)| a == b).map(|(_, x)| x).sum();
        if tr != 0.0 {
            obj.terms.insert(v, c(tr));
        }
        fj.insert(
            v,
            e.into_iter()
                .filter(|(_, x)| *x != 0.0)
                .map(|((a, b), x)| (a, b, c(x)))
                .collect(),
        );
    }
    model.add_lmi(Lmi {
        size: n,
        real: true,
        f0: (0..n).map(|i| (i, i, c(1.0))).collect(),
        fj,
        label: "dense overlap".into(),
    });
    model.minimize(&obj);
    model
}

/// [`dense_overlap`] with a planted common kernel `e_0 − e_1`: generated on
/// `n − 1` coordinates, the first of which is then duplicated onto
/// coordinates 0 and 1 (rows, columns and the diagonal entry), so every
/// coefficient matrix annihilates `e_0 − e_1` and the presolve must remove
/// it (common-kernel search, then the restriction of every `F_j`).
fn kernel_overlap(n: usize, m: usize, per: usize, seed: u64) -> SdpModel {
    let base = dense_overlap(n - 1, m, per, seed);
    let lmi = &base.lmis()[0];
    // index i of the base model is i + 1 here, and its index 0 is also 0
    let spread = |f: &SparseHerm| -> SparseHerm {
        let mut out: BTreeMap<(usize, usize), C64> = BTreeMap::new();
        for &(a, b, x) in f {
            let (a, b) = (a + 1, b + 1);
            let aa: &[usize] = if a == 1 { &[0, 1] } else { &[a] };
            let bb: &[usize] = if b == 1 { &[0, 1] } else { &[b] };
            for &i in aa {
                for &j in bb {
                    if i <= j {
                        *out.entry((i, j)).or_insert(c(0.0)) += x;
                    }
                }
            }
        }
        out.into_iter().map(|((i, j), x)| (i, j, x)).collect()
    };
    let mut model = SdpModel::new();
    let y = model.free_vars(m, "y");
    let fj = lmi.fj.iter().map(|(j, f)| (y[*j], spread(f))).collect();
    model.add_lmi(Lmi {
        size: n,
        real: true,
        f0: spread(&lmi.f0),
        fj,
        label: "kernel overlap".into(),
    });
    let mut obj = Affine::zero();
    for (j, cj) in base.objective().iter().enumerate() {
        if *cj != 0.0 {
            obj.terms.insert(y[j], c(*cj));
        }
    }
    model.minimize(&obj);
    model
}

/// `heisenberg6`'s moment-form SDP with its blocks scaled by 10⁹ and the
/// variables by 10⁻¹⁰ (`y = 10⁻¹⁰ y'`: equality coefficients 10⁻¹⁰ against
/// right-hand sides 1): the same problem and optimum.
fn rescaled_blocks() -> SdpModel {
    let spec = heisenberg(6, false);
    let base = Relaxation::level(spec.gb.reducer(), 2)
        .minimize(&spec.h)
        .build_moment_form()
        .to_model();
    let (block, var) = (1e9, 1e-10);
    let mut model = SdpModel::new();
    let y = model.free_vars(base.num_vars(), "y");
    for l in base.lmis() {
        let sc =
            |f: &SparseHerm, s: f64| -> SparseHerm { f.iter().map(|&(a, b, v)| (a, b, v * s)).collect() };
        model.add_lmi(Lmi {
            size: l.size,
            real: l.real,
            f0: sc(&l.f0, block),
            fj: l.fj.iter().map(|(j, f)| (y[*j], sc(f, block * var))).collect(),
            label: l.label.clone(),
        });
    }
    for (row, rhs) in base.equalities() {
        model.add_equality(row.iter().map(|(j, c)| (y[*j], c * var)).collect(), *rhs);
    }
    let mut obj = Affine::zero();
    for (j, cj) in base.objective().iter().enumerate() {
        if *cj != 0.0 {
            obj.terms.insert(y[j], c(*cj * var));
        }
    }
    model.minimize(&obj);
    model
}

/// `min Z0 + Z1 + X0 X1` on two qubits subject to `10⁻⁹ (Z0 + Z1 + 1) ⪰ 0`
/// (level 2, localizer level 1), as the Gram (SOS) model.
fn tiny_localizer() -> SdpModel {
    let (alg, rels) = relations::pauli::<C64>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h = red.reduce(&parse(&alg, "Z0 + Z1 + X0*X1"));
    let g = parse::<C64>(&alg, "Z0 + Z1 + 1").map_coeffs(|v| v * 1e-9);
    let sos = Relaxation::level(red, 2)
        .minimize(&h)
        .localizing_level(&g, 1)
        .sos_model();
    sos.model
}

/// Goemans–Williamson on G(n, p): `min Σ y_i  s.t.  Diag(y) − L/4 ⪰ 0`.
fn maxcut(n: usize, p: f64, seed: u64) -> SdpModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut lap: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for a in 0..n {
        for b in a + 1..n {
            if rng.gen::<f64>() < p {
                *lap.entry((a, a)).or_insert(0.0) += 1.0;
                *lap.entry((b, b)).or_insert(0.0) += 1.0;
                *lap.entry((a, b)).or_insert(0.0) -= 1.0;
            }
        }
    }
    let mut model = SdpModel::new();
    let y = model.free_vars(n, "y");
    let fj = y
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, vec![(i, i, c(1.0))]))
        .collect();
    model.add_lmi(Lmi {
        size: n,
        real: true,
        f0: lap.into_iter().map(|((a, b), x)| (a, b, c(-x / 4.0))).collect(),
        fj,
        label: "maxcut".into(),
    });
    let mut obj = Affine::zero();
    for &v in &y {
        obj.terms.insert(v, c(1.0));
    }
    model.minimize(&obj);
    model
}

/// Lovász theta of the Paley graph on Z_q (q prime, q ≡ 1 mod 4):
/// `min t  s.t.  t I + Σ_e x_e E_e − J ⪰ 0` (`E_e` symmetric with 1/2 at the
/// edge), optimum √q.
fn theta_paley(q: usize) -> SdpModel {
    let mut square = vec![false; q];
    for x in 1..q {
        square[x * x % q] = true;
    }
    let mut model = SdpModel::new();
    let t = model.free_var("t");
    let mut fj: BTreeMap<_, SparseHerm> = BTreeMap::new();
    fj.insert(t, (0..q).map(|i| (i, i, c(1.0))).collect());
    for a in 0..q {
        for b in a + 1..q {
            if square[b - a] {
                let x = model.free_var(format!("x{a}_{b}"));
                fj.insert(x, vec![(a, b, c(0.5))]);
            }
        }
    }
    model.add_lmi(Lmi {
        size: q,
        real: true,
        f0: (0..q)
            .flat_map(|a| (a..q).map(move |b| (a, b, c(-1.0))))
            .collect(),
        fj,
        label: "theta".into(),
    });
    model.minimize(&Affine::var(t));
    model
}

/// The record of an instance: the fastest time of each phase over the
/// repeats, the solver's profile of the fastest solve, and its counters.
fn json(name: &str, runs: &[Run]) -> String {
    let min = |f: &dyn Fn(&Run) -> f64| runs.iter().map(f).fold(f64::INFINITY, f64::min);
    let best = runs
        .iter()
        .min_by(|a, b| a.solve.total_cmp(&b.solve))
        .expect("a run");
    let p = &best.sol.profile;
    let num = |x: f64| {
        if x.is_finite() {
            format!("{x:.6e}")
        } else {
            "null".into()
        }
    };
    let fields: Vec<(&str, String)> = vec![
        ("name", format!("{name:?}")),
        ("build", num(min(&|r| r.build))),
        ("solve", num(min(&|r| r.solve))),
        ("post", num(min(&|r| r.post))),
        ("value", num(best.value)),
        ("certified", best.certified.map_or("null".into(), num)),
        ("status", format!("\"{:?}\"", best.sol.status)),
        ("num_vars", best.num_vars.to_string()),
        ("blocks", format!("{:?}", best.blocks)),
        ("warnings", best.sol.warnings.len().to_string()),
        ("presolve", num(p.presolve)),
        ("elimination", num(p.elimination)),
        ("dualize", num(p.dualize)),
        ("setup", num(p.setup)),
        ("schur_assembly", num(p.schur_assembly)),
        ("schur_factor", num(p.schur_factor)),
        ("newton_solves", num(p.newton_solves)),
        ("iterate", num(p.iterate)),
        ("solves", p.solves.to_string()),
        ("iterations", p.iterations.to_string()),
        ("operator_products", p.operator_products.to_string()),
        ("factor_retries", p.factor_retries.to_string()),
        ("kernel_certified", p.kernel_certified.to_string()),
        ("kernel_searched", p.kernel_searched.to_string()),
        ("kernel_rows", p.kernel_rows.to_string()),
    ];
    let body: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
    format!("  {{{}}}", body.join(", "))
}

/// `model` with block `b` scaled by `c_b` and variable `j` by `t_j`
/// (`y_j = t_j y'_j`): the same problem, and the same optimum.
fn rescaled(model: &SdpModel, c: &[f64], t: &[f64]) -> SdpModel {
    let mut out = SdpModel::new();
    let y = out.free_vars(model.num_vars(), "y");
    let sc = |f: &SparseHerm, s: f64| -> SparseHerm { f.iter().map(|&(a, b, v)| (a, b, v * s)).collect() };
    for (l, &cb) in model.lmis().iter().zip(c) {
        out.add_lmi(Lmi {
            size: l.size,
            real: l.real,
            f0: sc(&l.f0, cb),
            fj: l.fj.iter().map(|(j, f)| (y[*j], sc(f, cb * t[*j]))).collect(),
            label: l.label.clone(),
        });
    }
    for (row, rhs) in model.equalities() {
        out.add_equality(row.iter().map(|(j, a)| (y[*j], a * t[*j])).collect(), *rhs);
    }
    // stored negated for maximization
    let sign = if model.is_minimize() { 1.0 } else { -1.0 };
    let mut obj = Affine::constant(sign * model.objective_constant());
    for (j, cj) in model.objective().iter().enumerate() {
        if *cj != 0.0 {
            obj.terms.insert(y[j], c64(sign * *cj * t[j]));
        }
    }
    if model.is_minimize() {
        out.minimize(&obj)
    } else {
        out.maximize(&obj)
    }
    out
}

fn c64(x: f64) -> C64 {
    C64::new(x, 0.0)
}

/// `--rescale`: every instance's SDP against random rescalings of it.
fn rescale_check(seeds: u64) {
    let models: Vec<(&str, SdpModel)> = vec![
        ("dense_overlap", dense_overlap(300, 750, 200, 1)),
        ("kernel_overlap", kernel_overlap(300, 750, 200, 2)),
        ("heisenberg6", relaxation(&heisenberg(6, false)).to_model()),
        ("syk16", relaxation(&syk(16, 1)).to_model()),
        ("i3322_l3", relaxation(&i3322(3)).to_model()),
        ("rosenbrock10", relaxation(&rosenbrock(10)).to_model()),
        ("maxcut800", maxcut(800, 0.06, 1)),
        ("theta101", theta_paley(101)),
        ("heisenberg8_sym", relaxation(&heisenberg(8, true)).to_model()),
        ("qmc7_sym", relaxation(&qmc_cycle(7)).to_model()),
        ("tiny_localizer", tiny_localizer()),
    ];
    // RESCALE_ONLY=<instance>, RESCALE_MODE=identity|blocks|vars (default both)
    let only = std::env::var("RESCALE_ONLY").ok();
    let mode = std::env::var("RESCALE_MODE").unwrap_or_default();
    let (bs, vs) = match mode.as_str() {
        "identity" => (0.0, 0.0),
        "blocks" => (8.0, 0.0),
        "vars" => (0.0, 4.0),
        _ => (8.0, 4.0),
    };
    for (name, model) in &models {
        if only.as_deref().is_some_and(|o| o != *name) {
            continue;
        }
        let base = model.solve(&opts()).expect("solve");
        let mut line = format!("{:<16} {:+.8} ({:>2} it)", name, base.objective, base.iterations);
        for seed in 0..seeds {
            let mut rng = StdRng::seed_from_u64(1000 + seed);
            let mut draw = |r: f64| {
                if r == 0.0 {
                    1.0
                } else {
                    10f64.powf(rng.gen_range(-r..r))
                }
            };
            let c: Vec<f64> = (0..model.lmis().len()).map(|_| draw(bs)).collect();
            let t: Vec<f64> = (0..model.num_vars()).map(|_| draw(vs)).collect();
            match rescaled(model, &c, &t).solve(&opts()) {
                Ok(sol) => {
                    let rel = (sol.objective - base.objective).abs() / base.objective.abs().max(1.0);
                    line += &format!("  | {:.1e} {:?} {} it", rel, sol.status, sol.iterations);
                }
                Err(e) => line += &format!("  | error {e:?}"),
            }
        }
        println!("{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(i) = args.iter().position(|a| a == "--rescale") {
        let seeds = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
        rescale_check(seeds);
        return;
    }
    let mut repeat = 2;
    let mut out = "canary-out.json".to_string();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--repeat" {
            repeat = args[i + 1].parse().expect("--repeat N");
            i += 1;
        } else {
            out = args[i].clone();
        }
        i += 1;
    }
    type Case = (&'static str, Box<dyn Fn() -> Run>);
    let cases: Vec<Case> = vec![
        (
            "dense_overlap",
            Box::new(|| model_run(|| dense_overlap(300, 750, 200, 1))),
        ),
        (
            "kernel_overlap",
            Box::new(|| model_run(|| kernel_overlap(300, 750, 200, 2))),
        ),
        (
            "heisenberg6",
            Box::new(|| pipeline(|| heisenberg(6, false), complex)),
        ),
        ("syk16", Box::new(|| pipeline(|| syk(16, 1), complex))),
        ("i3322_l3", Box::new(|| pipeline(|| i3322(3), real))),
        ("rosenbrock10", Box::new(|| pipeline(|| rosenbrock(10), real))),
        ("maxcut800", Box::new(|| model_run(|| maxcut(800, 0.06, 1)))),
        ("theta101", Box::new(|| model_run(|| theta_paley(101)))),
        (
            "heisenberg8_sym",
            Box::new(|| pipeline(|| heisenberg(8, true), complex)),
        ),
        ("qmc7_sym", Box::new(|| pipeline(|| qmc_cycle(7), real))),
        ("rescaled_blocks", Box::new(|| model_run(rescaled_blocks))),
        ("tiny_localizer", Box::new(|| model_run(tiny_localizer))),
    ];
    let mut records = Vec::new();
    for (name, run) in &cases {
        let runs: Vec<Run> = (0..repeat.max(1)).map(|_| run()).collect();
        let min = |f: &dyn Fn(&Run) -> f64| runs.iter().map(f).fold(f64::INFINITY, f64::min);
        eprintln!(
            "{:<16} build {:>7.3} s  solve {:>7.3} s  post {:>7.3} s  value {:+.8}  {:?}, {} iterations",
            name,
            min(&|r| r.build),
            min(&|r| r.solve),
            min(&|r| r.post),
            runs[0].value,
            runs[0].sol.status,
            runs[0].sol.iterations
        );
        records.push(json(name, &runs));
    }
    std::fs::write(&out, format!("[\n{}\n]\n", records.join(",\n"))).expect("write output");
    eprintln!("wrote {out}");
}
