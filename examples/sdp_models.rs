//! The SDP modeling layer end to end: moment form, Gram (SOS) form, the
//! Gram-weight problem and a moment constraint for the two-qubit Heisenberg
//! interaction `h = X0 X1 + Y0 Y1 + Z0 Z1` (ground energy −3).
//!
//! Writes JSON files to `<temp dir>/ncalg_sdp_models/` and, with `--solve`, runs
//! `scripts/solve_moment_json.py` on each and checks the solutions with
//! `SdpModel::check`.
//!
//!     cargo run --release --example sdp_models -- [--solve] [--python <exe>]

use ncalg::ncsos::{read_dual_blocks, read_solution, Relaxation};
use ncalg::{relations, BigRational, Complex, GroebnerBasis, GroebnerOptions, Polynomial, SdpModel};
use std::path::Path;
use std::process::Command;

type Q = BigRational;
type K = Complex<Q>;

fn solve(python: &str, json: &Path, out: &Path, duals: Option<&Path>) -> Option<(f64, Vec<f64>)> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/solve_moment_json.py");
    let mut cmd = Command::new(python);
    cmd.arg(&script)
        .arg(json)
        .arg("1e-9")
        .arg("SCS")
        .arg("--out")
        .arg(out);
    if let Some(d) = duals {
        cmd.arg("--duals").arg(d);
    }
    let output = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    print!("{}", text);
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    let value = text
        .lines()
        .find_map(|l| l.strip_prefix("objective value "))?
        .trim()
        .parse::<f64>()
        .ok()?;
    Some((value, read_solution(out).ok()?))
}

fn report(name: &str, model: &SdpModel, x: &[f64]) {
    let check = model.check(x, 1e-5);
    println!(
        "   {}: objective {:.8}, max equality residual {:.1e}, min eigenvalue {:.1e} -> {}",
        name,
        model.objective_value(x),
        check.max_equality_residual,
        check.min_eigenvalue,
        if check.feasible() {
            "feasible"
        } else {
            "NOT feasible"
        }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let do_solve = args.iter().any(|a| a == "--solve");
    let python = args
        .iter()
        .position(|a| a == "--python")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "python".to_string());
    // output files go to a scratch directory, not into the library
    let dir = std::env::temp_dir().join("ncalg_sdp_models");
    std::fs::create_dir_all(&dir).expect("create output directory");

    let (alg, rels) = relations::pauli::<K>(2);
    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    let red = gb.reducer();
    let h: Polynomial<K> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let rel = Relaxation::level(red, 1).minimize(&h);
    // float versions of the ideal and the objective, for verifying certificates
    let gbf = gb.convert::<Complex<f64>>();
    let hf: Polynomial<Complex<f64>> = h.convert();

    // 1. moment form (variables = moments, explicit equalities)
    let moment = rel.build_moment_form().to_model();
    // 2. Gram / SOS form: max c s.t. h - c = Σ G_ab u_a* u_b, G ⪰ 0
    let sos = rel.sos_model();
    // 3. Gram-weight problem with shift 3 (= -λ_min)
    let gw = rel.gram_weight_model(3.0);
    // 4. moment constraint: fix the ZZ correlator to -1/3 (not the singlet's -1)
    let zz: Polynomial<K> = alg.parse("Z0*Z1").unwrap();
    let third = Complex::new(Q::new((-1).into(), 3.into()), Q::from(ncalg::BigInt::from(0)));
    let constrained = Relaxation::level(red, 1)
        .minimize(&h)
        .moment_equality(&zz, third)
        .build_moment_form()
        .to_model();

    let models: Vec<(&str, SdpModel)> = vec![
        ("moment", moment),
        ("sos", sos.model.clone()),
        ("gram_weight", gw.model.clone()),
        ("moment_zz", constrained),
    ];
    for (name, model) in &models {
        let json = dir.join(format!("{}.json", name));
        model.write_json(&json).unwrap();
        println!(
            "{}: {} variables, blocks {:?}, {} equalities -> {}",
            name,
            model.num_vars(),
            model.block_sizes(),
            model.equalities().len(),
            json.display()
        );
        if let Ok(e) = model.eliminate() {
            let sdpa = dir.join(format!("{}.dat-s", name));
            e.model.write_sdpa(&sdpa).unwrap();
            println!(
                "   eliminated: {} variables -> {}",
                e.model.num_vars(),
                sdpa.display()
            );
        }
    }
    if !do_solve {
        println!("(run with --solve to solve the models with CVXPY/SCS)");
        return;
    }
    for (name, model) in &models {
        let json = dir.join(format!("{}.json", name));
        let out = dir.join(format!("{}.sol", name));
        let Some((value, x)) = solve(&python, &json, &out, None) else {
            println!("   {}: solver failed", name);
            continue;
        };
        report(name, model, &x);
        match *name {
            "sos" => {
                println!(
                    "   SOS bound c = {:.8} (exact ground energy -3)",
                    sos.bound_value(&x).unwrap()
                );
                let g = sos.gram_value(0, &x);
                let trace: f64 = (0..g.len()).map(|i| g[i][i].re).sum();
                println!("   Gram matrix {}x{}, trace {:.6}", g.len(), g.len(), trace);
                // explicit certificate h - c = Σ λ_k p_k* p_k, verified modulo the Pauli relations
                let cert = sos.certificate(&x, 1e-7);
                let ver = cert.verify(gbf.reducer(), &hf);
                println!(
                    "   certificate: {} squares, certified bound {:.8}, max residual {:.1e}",
                    cert.num_squares(),
                    ver.bound,
                    ver.max_residual
                );
                print!("{}", cert.pruned(1e-6));
            }
            "gram_weight" => {
                let w = gw.weight_values(&x);
                println!(
                    "   diagonal weights sum {:.6}: {:?}",
                    w.iter().sum::<f64>(),
                    w.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>()
                );
            }
            _ => println!("   value {:.8}", value),
        }
    }

    // 5. certificates from the dual of the eliminated moment SDP (write_json ->
    //    CVXPY --duals), without and with the qubit-swap symmetry (block diagonal)
    let swap: Vec<ncalg::Var> = vec![3, 4, 5, 0, 1, 2];
    let variants: Vec<(&str, ncalg::ncsos::MomentSdp)> = vec![
        ("moment_lmi", rel.build()),
        (
            "moment_lmi_sym",
            Relaxation::level(red, 1)
                .minimize(&h)
                .symmetry(vec![swap])
                .build(),
        ),
    ];
    for (name, sdp) in &variants {
        let json = dir.join(format!("{}.json", name));
        let out = dir.join(format!("{}.sol", name));
        let duals_path = dir.join(format!("{}.duals", name));
        sdp.write_json(&json).unwrap();
        println!(
            "{}: {} variables, blocks {:?} (group order {})",
            name,
            sdp.num_vars(),
            sdp.block_sizes(),
            sdp.group_order()
        );
        let Some((_value, z)) = solve(&python, &json, &out, Some(&duals_path)) else {
            println!("   {}: solver failed", name);
            continue;
        };
        println!(
            "   relaxation value {:.8}",
            sdp.relaxation_value(sdp.objective().iter().zip(&z).map(|(c, z)| c * z).sum())
        );
        let duals = read_dual_blocks(&duals_path).unwrap();
        let cert = sdp.certificate(&duals, 1e-7);
        let ver = cert.verify(gbf.reducer(), &hf);
        println!(
            "   certificate from the dual: {} squares, certified bound {:.8}, max residual {:.1e}",
            cert.num_squares(),
            ver.bound,
            ver.max_residual
        );
    }
}
