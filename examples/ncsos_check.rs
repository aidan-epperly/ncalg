//! Consistency check of the NC-SOS builder: assembled moment matrix vs direct normal forms and Hermitian symmetry
//! on the 5-qubit swap algebra, level 2.
use ncalg::ncsos::Relaxation;
use ncalg::{BigRational, FreeAlgebra, GroebnerBasis, GroebnerOptions, Polynomial};
use num_traits::ToPrimitive;
type Q = BigRational;

fn main() {
    let n = 5;
    let mut names = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            names.push(format!("s{}{}", i, j));
        }
    }
    let alg = FreeAlgebra::new(names);
    let g = |a: usize, b: usize| -> Polynomial<Q> { alg.gen(&format!("s{}{}", a.min(b), a.max(b))) };
    let mut rels = Vec::new();
    for i in 1..=n {
        for j in (i + 1)..=n {
            rels.push(g(i, j) * g(i, j) - alg.one::<Q>());
            for k in 1..=n {
                for l in (k + 1)..=n {
                    if (i, j) < (k, l) && i != k && i != l && j != k && j != l {
                        rels.push(Polynomial::commutator(&g(i, j), &g(k, l)));
                    }
                }
            }
        }
    }
    for i in 1..=n {
        for j in 1..=n {
            for k in 1..=n {
                if i != j && j != k && i != k {
                    rels.push(g(i, j) * g(j, k) - g(i, k) * g(i, j));
                }
            }
        }
    }
    for i in 1..=n {
        for j in (i + 1)..=n {
            for k in (j + 1)..=n {
                rels.push(
                    alg.one::<Q>() - g(i, j) - g(i, k) - g(j, k) + g(i, j) * g(j, k) + g(j, k) * g(i, j),
                );
            }
        }
    }
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let mut h = alg.zero::<Q>();
    for i in 1..=n {
        h += g(i, if i == n { 1 } else { i + 1 });
    }
    let sdp = Relaxation::level(red, 2).minimize(&h).build();
    let words = &sdp.block_words()[0];
    let nz = sdp.num_vars();
    let m0 = sdp.block_matrix(&vec![0.0; nz], 0);
    println!(
        "block {}x{}, {} vars, {} moment words",
        words.len(),
        words.len(),
        nz,
        sdp.moment_words().len()
    );
    let _ = m0;
    // random z: (1) assembled matrix vs direct evaluation of L_z(NF(u* v))
    let z: Vec<f64> = (0..nz)
        .map(|j| ((j * 7919 + 13) % 97) as f64 / 97.0 - 0.5)
        .collect();
    let mz = sdp.block_matrix(&z, 0);
    let mut bad = 0;
    for a in 0..words.len() {
        for b in a..words.len() {
            let p = alg.monomial(words[a].reversed().concat(&words[b]));
            let direct = sdp.evaluate(&z, red, &p).unwrap().re;
            if (mz[a][b].re - direct).abs() > 1e-9 {
                bad += 1;
                if bad <= 5 {
                    println!(
                        "assembly mismatch M[{},{}]: assembled {} vs direct {}",
                        alg.format_word(&words[a]),
                        alg.format_word(&words[b]),
                        mz[a][b].re,
                        direct
                    );
                }
            }
        }
    }
    println!("assembly mismatches: {}", bad);
    // (2) Hermitian property of the parametrized functional: L_z(w*) == L_z(w)
    let mut bad = 0;
    for (w, val) in sdp.moments(&z) {
        let star = alg.monomial(w.reversed());
        let v2 = sdp.evaluate(&z, red, &star).unwrap().re;
        if (v2 - val.re).abs() > 1e-9 {
            bad += 1;
            if bad <= 5 {
                println!(
                    "Hermitian mismatch for {}: L(w) = {}, L(w*) = {}",
                    alg.format_word(&w),
                    val.re,
                    v2
                );
            }
        }
    }
    println!("Hermitian mismatches: {}", bad);
    let _ = <Q as ncalg::Coeff>::from_i64(1).to_f64();
    // y0 check
    let mom = sdp.moments(&vec![0.0; nz]);
    let nonzero: Vec<String> = mom
        .iter()
        .filter(|(_, c)| c.re.abs() > 1e-12)
        .map(|(w, c)| format!("{}={}", alg.format_word(w), c.re))
        .collect();
    println!("moments at z=0 with nonzero value: {:?}", nonzero);
}
