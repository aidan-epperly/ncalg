//! Solving standard-form models through their dual LMI.
//!
//! A model whose LMIs are all bare matrix variables (`SdpModel::psd_var`),
//! with every other variable free and appearing only in equalities and the
//! objective, is a standard-form SDP
//!
//! ```text
//!   min ⟨C, X⟩ + c_f·f   s.t.   ⟨A_i, X⟩ + (A_f f)_i = b_i,   X ⪰ 0.
//! ```
//!
//! The Gram forms (`Relaxation::sos_model`) are of this shape.  Solved as an
//! LMI in its own variables, its Schur complement has `N − r` rows (`N` matrix
//! entries, `r` independent equalities); its Lagrangian dual
//!
//! ```text
//!   max b·y   s.t.   C − Σ_i y_i A_i ⪰ 0,   A_fᵀ y = c_f
//! ```
//!
//! is an LMI in `r` variables, so when `r < N − r` the dual is solved
//! instead: the matrices `X` are its dual (Gram) matrices, the free variables
//! come from a backward-stable least-squares solve of the primal equalities,
//! and the model's own dual matrices are the slacks `C − Σ y_i A_i`.

use super::solver::{SdpSolution, SolverError, SolverOptions};
use super::{Affine, Lmi, SdpModel, SparseHerm, VarId};
use crate::sparse_qr::{QrOptions, SparseQr, SparseRows};
use num_complex::Complex;
use std::collections::BTreeMap;
use std::time::Instant;

type C64 = Complex<f64>;

/// Where a matrix-variable entry lives: block, row, column, imaginary part?
#[derive(Clone, Copy)]
struct Slot {
    block: usize,
    r: usize,
    c: usize,
    imag: bool,
}

/// `Some(result)` when the model is in standard form and the dual is
/// smaller (or `opts.dualize == Some(true)`), `None` to solve it as it is.
pub(crate) fn solve_dual(model: &SdpModel, opts: &SolverOptions) -> Option<Result<SdpSolution, SolverError>> {
    if opts.dualize == Some(false)
        || model.psd_blocks.is_empty()
        || model.psd_blocks.len() != model.lmis.len()
    {
        return None;
    }
    let t = Instant::now();
    let n = model.num_vars();
    // every LMI is one registered matrix variable, untouched
    let mut slot: Vec<Option<Slot>> = vec![None; n];
    for (b, &(lmi, first, size, real)) in model.psd_blocks.iter().enumerate() {
        if lmi != b
            || !model.lmis[lmi].f0.is_empty()
            || model.lmis[lmi].size != size
            || model.lmis[lmi].real != real
        {
            return None;
        }
        let mv = super::MatrixVar {
            size,
            real,
            first,
            name: String::new(),
        };
        for r in 0..size {
            for c in r..size {
                slot[mv.re_var(r, c)] = Some(Slot {
                    block: b,
                    r,
                    c,
                    imag: false,
                });
                if let Some(j) = mv.im_var(r, c) {
                    slot[j] = Some(Slot {
                        block: b,
                        r,
                        c,
                        imag: true,
                    });
                }
            }
        }
        if model.lmis[lmi]
            .fj
            .keys()
            .any(|j| slot[*j].is_none_or(|s| s.block != b))
        {
            return None;
        }
    }
    let free: Vec<VarId> = (0..n).filter(|j| slot[*j].is_none()).collect();
    let mut free_index = vec![usize::MAX; n];
    for (k, &j) in free.iter().enumerate() {
        free_index[j] = k;
    }
    // independent equality rows: drop the rows dependent on earlier ones as
    // rows of [A | b] (consistent redundancy); if A alone is still rank
    // deficient the system is inconsistent and the normal route reports it
    // (the right-hand side column scaled to the rows)
    let (rows, _) = super::homogenised_rows(&model.equalities, n);
    let qr_opts = QrOptions::default();
    let aug = SparseQr::factor(
        &SparseRows {
            ncols: n + 1,
            rows: rows.clone(),
        },
        &qr_opts,
    )
    .ok()?;
    let mut keep = vec![true; rows.len()];
    for &(i, _) in aug.dependent_rows() {
        keep[i] = false;
    }
    let kept: Vec<usize> = (0..rows.len()).filter(|&i| keep[i]).collect();
    let plain: Vec<Vec<(usize, f64)>> = kept
        .iter()
        .map(|&i| rows[i].iter().filter(|(j, _)| *j < n).cloned().collect())
        .collect();
    let a_only = SparseQr::factor(
        &SparseRows {
            ncols: n,
            rows: plain.clone(),
        },
        &qr_opts,
    )
    .ok()?;
    if !a_only.dependent_rows().is_empty() {
        return None;
    }
    let r = kept.len();
    if opts.dualize.is_none() && r >= n.saturating_sub(r) {
        return None; // the model's own formulation is not larger
    }
    let rhs: Vec<f64> = kept.iter().map(|&i| model.equalities[i].1).collect();

    // the Hermitian matrix of a linear functional on a block's entries:
    // Re X_aa ↔ E_aa, Re X_rc ↔ upper (r,c) += 1/2, Im X_rc ↔ upper += i/2
    let add = |m: &mut BTreeMap<(usize, usize), C64>, s: Slot, coef: f64| {
        let v = if s.r == s.c {
            C64::new(coef, 0.0)
        } else if s.imag {
            C64::new(0.0, 0.5 * coef)
        } else {
            C64::new(0.5 * coef, 0.0)
        };
        *m.entry((s.r, s.c)).or_insert(C64::new(0.0, 0.0)) += v;
    };
    let nb = model.psd_blocks.len();
    let mut c_mats: Vec<BTreeMap<(usize, usize), C64>> = vec![BTreeMap::new(); nb];
    for (j, &cj) in model.objective.iter().enumerate() {
        if let (Some(s), true) = (slot[j], cj != 0.0) {
            add(&mut c_mats[s.block], s, cj);
        }
    }
    let mut a_mats: Vec<BTreeMap<usize, BTreeMap<(usize, usize), C64>>> = vec![BTreeMap::new(); nb]; // [block][row]
    let mut free_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); free.len()]; // [free var] -> (row, A_if)
    for (k, row) in plain.iter().enumerate() {
        for &(j, c) in row {
            match slot[j] {
                Some(s) => add(a_mats[s.block].entry(k).or_default(), s, c),
                None => free_rows[free_index[j]].push((k, c)),
            }
        }
    }
    let to_herm = |m: &BTreeMap<(usize, usize), C64>, scale: f64| -> SparseHerm {
        m.iter()
            .filter(|(_, v)| v.norm() != 0.0)
            .map(|(&(r, c), v)| (r, c, v * scale))
            .collect()
    };
    // the dual: min −b·y  s.t.  C_b − Σ_i y_i A_ib ⪰ 0,  Σ_i A_if y_i = c_f
    let mut dual = SdpModel::new();
    let y = dual.free_vars(r, "y");
    for (b, &(_, _, size, real)) in model.psd_blocks.iter().enumerate() {
        let fj: BTreeMap<VarId, SparseHerm> =
            a_mats[b].iter().map(|(k, m)| (y[*k], to_herm(m, -1.0))).collect();
        dual.add_lmi(Lmi {
            size,
            real,
            f0: to_herm(&c_mats[b], 1.0),
            fj,
            label: format!("dual of {}", model.lmis[b].label),
        });
    }
    for (k, rows_f) in free_rows.iter().enumerate() {
        let row: Vec<(VarId, f64)> = rows_f.iter().map(|(i, c)| (y[*i], *c)).collect();
        dual.add_equality(row, model.objective[free[k]]);
    }
    let mut obj = Affine::zero();
    for (k, b) in rhs.iter().enumerate() {
        if *b != 0.0 {
            obj.terms.insert(y[k], C64::new(-b, 0.0));
        }
    }
    dual.minimize(&obj);
    let setup = t.elapsed().as_secs_f64();
    let mut sol = match dual.solve_inner(opts) {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
    };
    let t = Instant::now();
    // the matrix variables: X_b = conj(G_b) of the dual's duals
    let mut x = vec![0.0; n];
    for (j, s) in slot.iter().enumerate() {
        if let Some(s) = s {
            let g = sol.duals[s.block][s.r][s.c];
            x[j] = if s.imag { -g.im } else { g.re };
        }
    }
    // the free variables: least squares of the primal equalities
    if !free.is_empty() {
        let mut fr: Vec<Vec<(usize, f64)>> = vec![Vec::new(); r];
        let mut res = rhs.clone();
        for (k, row) in plain.iter().enumerate() {
            for &(j, c) in row {
                if slot[j].is_some() {
                    res[k] -= c * x[j];
                } else {
                    fr[k].push((free_index[j], c));
                }
            }
        }
        let f = crate::linalg::dense::least_squares_rows(&fr, &res, free.len());
        for (k, &j) in free.iter().enumerate() {
            x[j] = f[k];
        }
    }
    // the model's dual matrices: conj of the slacks C − Σ y A
    let duals: Vec<Vec<Vec<C64>>> = dual
        .lmis()
        .iter()
        .map(|l| {
            l.value(&sol.x)
                .into_iter()
                .map(|row| row.into_iter().map(|v| v.conj()).collect())
                .collect()
        })
        .collect();
    let objective = if sol.regularization.is_some() {
        let v = -sol.objective + model.objective_constant;
        if model.minimize {
            v
        } else {
            -v
        }
    } else {
        model.objective_value(&x)
    };
    sol.profile.dualize += setup + t.elapsed().as_secs_f64();
    Some(Ok(SdpSolution {
        x,
        objective,
        duals,
        ..sol
    }))
}
