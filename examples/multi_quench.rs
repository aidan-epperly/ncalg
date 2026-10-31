//! Several linked moment functionals (`ncalg::multi`): a one-qubit
//! quench in the differential-NPO hierarchy of Araújo, Garner and Navascués
//! (arXiv 2408.02572, Sec. 5.2).
//!
//! The state |+> evolves under `H = Z` for time `τ`; the exact expectation
//! afterwards is `<X>(τ) = cos 2τ`.  Unknowns, all on the algebra of a qubit
//! (`X`, `Z`) and a central `t`:
//!
//! * the final state `b`, a functional on qubit words;
//! * the time integral `ω(t^j o) = ∫_0^1 t^j b(s)(o) ds` along the quench
//!   (`b(s)` the state at time `s τ`), a functional on `t^r ×` qubit words
//!   with the localizing matrix of `t − t² ⪰ 0`.
//!
//! Integration by parts of `d/ds b(s)(o) = i τ b(s)([H, o])` links them:
//! `ω(j t^{j-1} o + i τ t^j [H, o]) = b(o) − δ_{j0} <+|o|+>` for every
//! observable `o` and `j ≥ 0`.  The program maximizes and minimizes `b(X)`,
//! solves natively (regularized: these relaxations have no strictly
//! feasible point), and certifies each bound with an SOS certificate.
//!
//! ```text
//! cargo run --release --example multi_quench
//! ```

use ncalg::multi::{FunctionalId, MultiRelaxation};
use ncalg::ncsos::Relaxation;
use ncalg::solver::SolverOptions;
use ncalg::{Complex, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial, Word};

type C = Complex<f64>;

fn main() {
    let tau: f64 = 0.4;
    let alg = FreeAlgebra::new_star([("t", true), ("X", true), ("Z", true)]);
    let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
    let gb = GroebnerBasis::compute(
        vec![
            p("t*X - X*t"),
            p("t*Z - Z*t"),
            p("X*X - 1"),
            p("Z*Z - 1"),
            p("X*Z + Z*X"),
        ],
        &GroebnerOptions::default(),
    );
    assert!(gb.is_complete());
    let red = gb.reducer();
    let qubit_words = ["", "X", "Z", "X*Z"];
    let word = |s: &str| -> Word {
        if s.is_empty() {
            Word::empty()
        } else {
            red.reduce(&p(s)).leading_word().unwrap().clone()
        }
    };
    let h = p("Z");
    let plus = |o: &str| -> f64 {
        // <+|o|+>: 1 for the identity and X, 0 for Z and XZ
        if o.is_empty() || o == "X" {
            1.0
        } else {
            0.0
        }
    };
    println!(
        "one-qubit quench, tau = {}: exact <X> = cos 2 tau = {:.10}",
        tau,
        (2.0 * tau).cos()
    );
    for kt in 1..=3 {
        for maximize in [true, false] {
            let mut mr = MultiRelaxation::new(red);
            let b_words: Vec<Word> = qubit_words.iter().map(|w| word(w)).collect();
            let b: FunctionalId = mr.functional("b", Relaxation::new(red, b_words));
            let tpow = |r: usize, w: &str| -> Word {
                let mut s = vec!["t"; r].join("*");
                if !w.is_empty() {
                    if !s.is_empty() {
                        s.push('*');
                    }
                    s.push_str(w);
                }
                word(&s)
            };
            let omega_words: Vec<Word> = (0..=kt)
                .flat_map(|r| qubit_words.iter().map(move |w| (r, *w)))
                .map(|(r, w)| tpow(r, w))
                .collect();
            let loc_words: Vec<Word> = (0..kt)
                .flat_map(|r| qubit_words.iter().map(move |w| (r, *w)))
                .map(|(r, w)| tpow(r, w))
                .collect();
            let omega = Relaxation::new(red, omega_words).localizing(&p("t - t*t"), loc_words);
            let w: FunctionalId = mr.functional("omega", omega);
            // links for every observable o and j = 0..2 kt
            for o in qubit_words {
                let op = if o.is_empty() { p("1") } else { p(o) };
                let comm = red.reduce(&(&(&h * &op) - &(&op * &h)));
                for j in 0..=2 * kt {
                    let tj = if j == 0 {
                        p("1")
                    } else {
                        p(&vec!["t"; j].join("*"))
                    };
                    let mut q = (&tj * &comm).scale(&C::new(0.0, tau));
                    if j >= 1 {
                        let tj1 = if j == 1 {
                            p("1")
                        } else {
                            p(&vec!["t"; j - 1].join("*"))
                        };
                        q += (&tj1 * &op).scale(&C::new(j as f64, 0.0));
                    }
                    let q = red.reduce(&q);
                    let minus_op = red.reduce(&op).scale(&C::new(-1.0, 0.0));
                    let value = if j == 0 {
                        C::new(-plus(o), 0.0)
                    } else {
                        C::new(0.0, 0.0)
                    };
                    let mut terms: Vec<(FunctionalId, &Polynomial<C>)> = vec![(b, &minus_op)];
                    if !q.is_zero() {
                        terms.push((w, &q));
                    }
                    mr.link(&terms, value);
                }
            }
            let x = p("X");
            if maximize {
                mr.maximize(&[(b, &x)]);
            } else {
                mr.minimize(&[(b, &x)]);
            }
            let sdp = mr.build().expect("build");
            let opts = SolverOptions {
                regularization: 1e-8,
                ..SolverOptions::default()
            };
            let (value, sol) = sdp.solve(&opts).expect("solve");
            let cert = sdp.certificate(&sol.duals, red, 1e-9);
            println!(
                "k_t = {}, {}: value {:.10} ({:?}, {} variables), certificate {:.10} (residual l1 {:.1e})",
                kt,
                if maximize { "max" } else { "min" },
                value,
                sol.status,
                sdp.num_vars(),
                cert.bound,
                cert.residual_l1
            );
        }
    }
}
