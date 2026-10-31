//! A relaxation in a hand-built *-algebra: the depth-1 QAOA MaxCut energy on
//! the 3-vertex path, bounded by a moment relaxation of the Heisenberg-picture
//! circuit.  It shows the pieces such a relaxation needs and that the generic
//! `Relaxation::level` examples do not:
//!
//! 1. **A custom algebra.**  Hermitian generators `V{l}_{i}`, `W{l}_{i}` (X- and
//!    Z-like Paulis of site `i` at circuit level `l = 0, 1`) with Pauli
//!    relations per level, and, with `--free`, central Hermitian *scalar*
//!    generators `c, s, d, t` on two circles (`c² + s² = 1`, `d² + t² = 1`)
//!    standing for `cos γ, sin γ, cos 2β, sin 2β`.  The scalars are listed
//!    first so that they move to the left of every word; the Gröbner basis is
//!    then finite and instant.  It is computed with exact coefficients and
//!    converted to floats for the relaxation.
//! 2. **A structured index set.**  Pauli words of weight ≤ `w` at each level,
//!    plus level-1 generators times level-0 Pauli words of weight ≤ `wp`
//!    (instead of all words up to a degree).
//! 3. **Vector-form constraints.**  An identity `R ψ = 0` for the (unknown)
//!    state vector gives `L(a* R b) = 0` for index words `a` and right factors
//!    `b`: the initial state `(V0_i − 1) ψ = 0` and the circuit recursion
//!    `V1_i = V0_i Π_{j~i} (c + i s W0_i W0_j)`, `W1_i = W0_i (d − i t V1_i)`.
//! 4. **Coverage filtering.**  A constraint that mentions a moment no matrix
//!    entry determines only fixes that free variable: `constraint_coverage()`
//!    finds those, and they are dropped.
//! 5. **Symmetry with constraints in orbits.**  The path's reflection
//!    `i ↦ 2 − i` maps the constraint of site 0 to that of site 2;
//!    `symmetrize_constraints(true)` averages them inside ncalg.
//! 6. **Routing by size.**  The native solver up to a few thousand variables,
//!    otherwise the moment form is exported for SCS.
//!
//! With fixed angles the bound is compared with the exact energy from a
//! 3-qubit statevector simulation: the default word set (`--w 2 --wp 1`) is
//! exact (1 326 variables with the symmetry, about 2 s), `--w 1 --wp 0` gives
//! 1.866 against the exact 1.6495.  With `--free` the angles are variables
//! too, and the relaxation bounds `max_θ F(θ)`; the mode illustrates central
//! scalar generators.
//!
//! ```text
//! cargo run --release --example custom_algebra -- [--gamma G --beta B | --free]
//!     [--w W] [--wp WP] [--no-sym] [--max-native-vars 5000]
//! ```

use ncalg::ncsos::Relaxation;
use ncalg::solver::SolverOptions;
use ncalg::{Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, HybridRational, Polynomial, Var, Word};
use std::collections::{HashMap, HashSet};

type K = Complex<HybridRational>; // exact, for the Gröbner basis
type C = Complex<f64>; // the relaxation

const N: usize = 3;
const EDGES: [(usize, usize); 2] = [(0, 1), (1, 2)];

struct Gens {
    alg: FreeAlgebra,
    /// (c, s, d, t) with --free
    scalars: Option<[Var; 4]>,
    /// (level, site) -> (V, W)
    op: HashMap<(usize, usize), (Var, Var)>,
}

fn generators(free: bool) -> Gens {
    let mut names: Vec<String> = Vec::new();
    let scalars = free.then(|| {
        names.extend(["c", "s", "d", "t"].map(String::from));
        [0, 1, 2, 3]
    });
    let mut op = HashMap::new();
    for l in 0..=1 {
        for i in 0..N {
            let base = names.len() as Var;
            names.push(format!("V{}_{}", l, i));
            names.push(format!("W{}_{}", l, i));
            op.insert((l, i), (base, base + 1));
        }
    }
    Gens {
        alg: FreeAlgebra::new(names),
        scalars,
        op,
    }
}

fn relations(g: &Gens) -> Vec<Polynomial<K>> {
    let alg = &g.alg;
    let v = |a: Var| -> Polynomial<K> { alg.var(a) };
    let one: Polynomial<K> = alg.one();
    let mut rels = Vec::new();
    if let Some([c, s, d, t]) = g.scalars {
        // central: commute with every other generator
        for a in [c, s, d, t] {
            for b in 0..alg.num_vars() as Var {
                if b > a {
                    rels.push(&v(b) * &v(a) - &v(a) * &v(b));
                }
            }
        }
        rels.push(&v(c) * &v(c) + &v(s) * &v(s) - &one);
        rels.push(&v(d) * &v(d) + &v(t) * &v(t) - &one);
    }
    for l in 0..=1 {
        for i in 0..N {
            let (x, z) = g.op[&(l, i)];
            rels.push(&v(x) * &v(x) - &one);
            rels.push(&v(z) * &v(z) - &one);
            rels.push(&v(x) * &v(z) + &v(z) * &v(x));
            for j in (i + 1)..N {
                let (x2, z2) = g.op[&(l, j)];
                for a in [x, z] {
                    for b in [x2, z2] {
                        rels.push(&v(b) * &v(a) - &v(a) * &v(b));
                    }
                }
            }
        }
    }
    rels
}

/// Pauli words of weight ≤ `w` at one level (products over sites of V, W or
/// VW, in site order), including the empty word.
fn pauli_words(g: &Gens, level: usize, w: usize) -> Vec<Word> {
    fn rec(g: &Gens, level: usize, start: usize, left: usize, cur: &mut Vec<Var>, out: &mut Vec<Word>) {
        if left == 0 {
            return;
        }
        for i in start..N {
            let (x, z) = g.op[&(level, i)];
            for choice in [vec![x], vec![z], vec![x, z]] {
                let len = cur.len();
                cur.extend(choice);
                out.push(Word::from_vec(cur.clone()));
                rec(g, level, i + 1, left - 1, cur, out);
                cur.truncate(len);
            }
        }
    }
    let mut out = vec![Word::empty()];
    rec(g, level, 0, w.min(N), &mut Vec::new(), &mut out);
    out
}

fn index_set(g: &Gens, w: usize, wp: usize) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |word: Word, words: &mut Vec<Word>| {
        if seen.insert(word.clone()) {
            words.push(word);
        }
    };
    for l in 0..=1 {
        for word in pauli_words(g, l, w) {
            push(word, &mut words);
        }
    }
    for b in pauli_words(g, 0, wp).into_iter().filter(|b| !b.is_empty()) {
        for i in 0..N {
            let (x, z) = g.op[&(1, i)];
            for gen in [x, z] {
                push(Word::letter(gen).concat(&b), &mut words);
            }
        }
    }
    // with free angles: also the scalar multiples c·u, s·u, d·u, t·u
    if let Some(sc) = g.scalars {
        let base = words.clone();
        for a in sc {
            for u in &base {
                push(Word::letter(a).concat(u), &mut words);
            }
        }
    }
    words
}

/// The recursion polynomials `R` (= 0 as operators), two per site.
fn recursion(g: &Gens, angles: Option<(f64, f64)>) -> Vec<Polynomial<C>> {
    let alg = &g.alg;
    let v = |a: Var| -> Polynomial<C> { alg.var(a) };
    let iu = C::new(0.0, 1.0);
    let [c, s, d, t]: [Polynomial<C>; 4] = match (g.scalars, angles) {
        (Some(sc), _) => sc.map(v),
        (None, Some((gamma, beta))) => [gamma.cos(), gamma.sin(), (2.0 * beta).cos(), (2.0 * beta).sin()]
            .map(|x| alg.constant(C::new(x, 0.0))),
        (None, None) => unreachable!(),
    };
    let mut out = Vec::new();
    for i in 0..N {
        let (x0, z0) = g.op[&(0, i)];
        let (x1, z1) = g.op[&(1, i)];
        let mut prod = v(x0);
        for &(a, b) in &EDGES {
            let j = if a == i {
                b
            } else if b == i {
                a
            } else {
                continue;
            };
            let zj = v(g.op[&(0, j)].1);
            prod = &prod * &(&c + &(&(&v(z0) * &zj) * &s).scale(&iu));
        }
        out.push(&v(x1) - &prod);
        out.push(&v(z1) - &(&v(z0) * &(&d - &(&v(x1) * &t).scale(&iu))));
    }
    out
}

/// Exact depth-1 QAOA energy on the path: |ψ> = e^{-iβB} e^{-iγH_C} |+++>.
fn exact_energy(gamma: f64, beta: f64) -> f64 {
    let cut = |z: usize| -> f64 {
        EDGES
            .iter()
            .filter(|&&(a, b)| (z >> a) & 1 != (z >> b) & 1)
            .count() as f64
    };
    let dim = 1 << N;
    let mut psi: Vec<C> = (0..dim)
        .map(|z| C::new(0.0, -gamma * cut(z)).exp() / (dim as f64).sqrt())
        .collect();
    let (cb, sb) = (beta.cos(), beta.sin());
    for q in 0..N {
        for z in 0..dim {
            if (z >> q) & 1 == 0 {
                let (a, b) = (psi[z], psi[z | (1 << q)]);
                psi[z] = a * cb + b * C::new(0.0, -sb);
                psi[z | (1 << q)] = a * C::new(0.0, -sb) + b * cb;
            }
        }
    }
    (0..dim).map(|z| psi[z].norm_sqr() * cut(z)).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let value =
        |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| args[i + 1].clone()) };
    let free = flag("--free");
    // default angles: the optimum of F (γ = π/3, β = π/8)
    let gamma: f64 = value("--gamma").map_or(std::f64::consts::FRAC_PI_3, |x| x.parse().unwrap());
    let beta: f64 = value("--beta").map_or(std::f64::consts::FRAC_PI_8, |x| x.parse().unwrap());
    let w: usize = value("--w").map_or(if free { 1 } else { 2 }, |x| x.parse().unwrap());
    let wp: usize = value("--wp").map_or(if free { 0 } else { 1 }, |x| x.parse().unwrap());
    let max_native: usize = value("--max-native-vars").map_or(5000, |x| x.parse().unwrap());

    // 1. the algebra and its Gröbner basis (exact), then float coefficients
    let g = generators(free);
    let gb_exact = GroebnerBasis::compute(relations(&g), &GroebnerOptions::default());
    assert!(gb_exact.is_complete());
    let gb = gb_exact.convert::<C>();
    let red = gb.reducer();
    let alg = &g.alg;
    println!(
        "{} generators, Gröbner basis of {} elements",
        alg.num_vars(),
        gb.len()
    );

    // 2. index set and objective Σ_e (1 − W1_a W1_b)/2
    let words = index_set(&g, w, wp);
    let mut h: Polynomial<C> = alg.zero();
    for &(a, b) in &EDGES {
        let (wa, wb): (Polynomial<C>, Polynomial<C>) = (alg.var(g.op[&(1, a)].1), alg.var(g.op[&(1, b)].1));
        h = &h + &(alg.one::<C>() - &wa * &wb).scale(&C::new(0.5, 0.0));
    }

    // 3. vector-form constraints L(a* R b) = 0
    let mut candidates: Vec<Polynomial<C>> = Vec::new();
    for i in 0..N {
        let r: Polynomial<C> = alg.var::<C>(g.op[&(0, i)].0) - alg.one::<C>();
        for a in &words {
            candidates.push(r.mul_word_left(alg.adjoint_word(a).as_slice()));
        }
    }
    let rights = pauli_words(&g, 0, wp);
    for r in recursion(&g, (!free).then_some((gamma, beta))) {
        for b in &rights {
            for a in &words {
                candidates.push(r.sandwich(alg.adjoint_word(a).as_slice(), b.as_slice()));
            }
        }
    }
    // reduce and drop zeros and duplicates (up to a scalar multiple)
    let mut keys = HashSet::new();
    let constraints: Vec<Polynomial<C>> = candidates
        .iter()
        .map(|p| red.reduce(p))
        .filter(|p| !p.is_zero() && keys.insert(p.monic().to_string()))
        .collect();

    // 4. coverage: keep the constraints whose moments the matrix determines
    let mut probe = Relaxation::new(red, words.clone()).maximize(&h);
    for p in &constraints {
        probe = probe.moment_equality(p, C::new(0.0, 0.0));
    }
    let coverage = probe.constraint_coverage();
    let kept: Vec<&Polynomial<C>> = constraints
        .iter()
        .zip(&coverage)
        .filter(|(_, free_words)| free_words.is_empty())
        .map(|(p, _)| p)
        .collect();
    println!(
        "{} index words; {} distinct constraints, {} kept, {} dropped as not covered",
        words.len(),
        constraints.len(),
        kept.len(),
        constraints.len() - kept.len()
    );

    // 5. the relaxation, with the reflection i -> 2 - i at every level
    let mut rel = Relaxation::new(red, words.clone()).maximize(&h);
    if !flag("--no-sym") {
        let mut perm: Vec<Var> = (0..alg.num_vars() as Var).collect();
        for l in 0..=1 {
            for i in 0..N {
                let (x, z) = g.op[&(l, i)];
                let (x2, z2) = g.op[&(l, N - 1 - i)];
                perm[x as usize] = x2;
                perm[z as usize] = z2;
            }
        }
        rel = rel.symmetry(vec![perm]).symmetrize_constraints(true);
    }
    for p in kept {
        rel = rel.moment_equality(p, C::new(0.0, 0.0));
    }

    // 6. solve natively when small, otherwise export the moment form
    let sdp = rel.try_build().unwrap_or_else(|e| panic!("build failed: {}", e));
    println!(
        "SDP: {} variables, blocks {:?}",
        sdp.num_vars(),
        sdp.block_sizes()
    );
    let bound = if sdp.num_vars() <= max_native {
        let (value, sol) = sdp.solve(&SolverOptions::default()).expect("native solver");
        println!(
            "native solver: {:?} after {} iterations",
            sol.status, sol.iterations
        );
        value
    } else {
        let path = std::env::temp_dir().join("custom_algebra_moment.json");
        rel.build_moment_form()
            .write_json(&path)
            .expect("write moment form");
        println!(
            "too large for the native solver; solve with\n  python scripts/solve_moment_json.py {} 1e-6 SCS",
            path.display()
        );
        return;
    };
    if bound.is_nan() {
        println!("the native solver did not converge; re-solve with CSDP (write_sdpa) or SCS (moment form)");
        return;
    }
    if free {
        println!("bound on max_θ F(θ): {:.8}", bound);
    } else {
        let exact = exact_energy(gamma, beta);
        println!(
            "bound {:.8} at γ = {:.6}, β = {:.6}; exact F = {:.8}; gap {:+.2e}",
            bound,
            gamma,
            beta,
            exact,
            bound - exact
        );
    }
}
