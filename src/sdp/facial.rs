//! Numerical facial reduction for the native solver
//! ([`SolverOptions::facial_reduction`]).
//!
//! Some relaxations have no strictly feasible moment point: the linear
//! constraints together with `Z(y) ⪰ 0` force every feasible block to be
//! singular (e.g. when linear links between several moment functionals pin
//! boundary states to pure states, whose moment matrices have rank at most the Hilbert-space
//! dimension).  The SOS side then has no optimal solution: the Gram matrices
//! `X` grow without bound, the Schur complement becomes hopelessly
//! ill-conditioned, and the iteration stalls.  Even where it gets close, the
//! answer is only accurate to about the square root of its feasibility
//! error (Sturm's error bound for singularity degree 1, "Error bounds for
//! linear matrix inequalities", SIAM J. Optim. 2000).
//!
//! Facial reduction (Borwein–Wolkowicz; see Drusvyatskiy–Wolkowicz, "The
//! many faces of degeneracy in conic optimization", 2017): a certificate
//! `X̄ ⪰ 0` with `⟨F_j, X̄⟩ = 0` for all `j` and `⟨F_0, X̄⟩ = 0` gives
//! `⟨Z(y), X̄⟩ = 0` for every `y`, hence `Z(y) X̄ = 0` for every feasible
//! `y`.  With `U` spanning `range X̄` and `V` its orthogonal complement the
//! problem is equivalent to `U* Z(y) = 0` (linear) and `V* Z(y) V ⪰ 0`
//! (smaller).  This module
//!
//! 1. takes as candidates the Gram directions that diverge as `μ → 0` (for
//!    unbounded iterates every limit point of `X_k / ‖X_k‖` is such a
//!    certificate);
//! 2. computes an exact certificate near them by Gauss–Newton on the data
//!    alone (`X̄ = B B*`, positive semidefinite by construction), and keeps
//!    the directions it certifies to the accuracy the caller asks for;
//! 3. adds `U* Z(y) = 0`, eliminates it with the sparse QR at the usual
//!    tolerance, restricts every block to `V`, and re-solves;
//! 4. repeats (one round per unit of singularity degree, at most three) if
//!    the reduced problem is again degenerate.
//!
//! The solution is reported in the original variables, the duals as
//! `G = conj(V) G' Vᵀ` (zero on the removed directions).  A reduced result
//! is only returned if every step checks out (certificate, consistent
//! elimination, converged re-solve, and an objective that does not
//! contradict the unreduced iterate); otherwise the unreduced result is.

use crate::linalg::dense::{hermitian_eigen_checked, hermitian_eigenvalues_checked};
use crate::linalg::sparse::{full_entries, norm_sq, to_dense};
use crate::sdp::{Affine, Eliminated, Lmi, SdpModel, SparseHerm};
use crate::solver::{
    solve_lmi_traced, FacialReduction, SdpSolution, SolveProfile, SolverError, SolverOptions, Status, Trace,
};
use faer::Mat;
use num_complex::Complex;
use std::time::Instant;

type C64 = Complex<f64>;

/// Candidate detection: compare the Gram matrices at the returned iterate
/// with those at an earlier one where `μ` was at least `MU_RATIO` times
/// larger.  Along the central path a direction of the structural kernel
/// grows like `μ^(-2^(-d))` for singularity degree `d` (Sturm), i.e. by 10,
/// 3.2, 1.8 over a factor 100 in `μ` for `d` = 1, 2, 3, while the bounded
/// directions converge (observed: at most 1.4).  `GROWTH` = 2 catches
/// `d ≤ 2` with margin on both sides; slower directions surface in a later
/// round, after the first reduction has removed the faster ones.  False candidates
/// are harmless because only certified directions are removed.
const MU_RATIO: f64 = 100.0;
const GROWTH: f64 = 2.0;
/// Rounds (one per unit of singularity degree); a cap, not a tuning knob.
const MAX_ROUNDS: usize = 3;
/// Smallest QR rank tolerance for the certified equalities: the library
/// default ([`SdpModel::solve`] eliminates with the same tolerance).
const RANK_TOL: f64 = 1e-9;

/// One reduction: equalities eliminated, then blocks restricted.
struct Stage {
    elim: Eliminated,
    /// Variables of the restricted model: `used[w]` is the eliminated
    /// model's variable behind the restricted model's variable `w`.
    used: Vec<usize>,
    /// Per block of the stage's input model: the basis `V` of the kept
    /// subspace (`None`: block unchanged), and the block's index in the
    /// restricted model (`None`: block removed entirely).
    bases: Vec<Option<Mat<C64>>>,
    /// The removed directions `U` per block of the input model.
    removed: Vec<Option<Mat<C64>>>,
    index: Vec<Option<usize>>,
    sizes: Vec<usize>,
    model: SdpModel,
}

impl Stage {
    /// `max |U* Z(x)|` over the removed directions, relative to the largest
    /// entry of `Z(x)`, for a point `x` of the stage's input model.
    fn kernel_residual(&self, input: &SdpModel, x: &[f64]) -> f64 {
        let mut worst = 0.0_f64;
        for (lmi, u) in input.lmis().iter().zip(&self.removed) {
            let Some(u) = u else { continue };
            let n = lmi.size;
            let zv = lmi.value(x);
            let z = Mat::<C64>::from_fn(n, n, |i, j| zv[i][j]);
            let zmax = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .fold(0.0_f64, |a, (i, j)| a.max(z[(i, j)].norm()));
            let r = crate::dense::adjoint_mul(u.as_ref(), z.as_ref());
            let rmax = (0..r.nrows())
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .fold(0.0_f64, |a, (i, j)| a.max(r[(i, j)].norm()));
            worst = worst.max(rmax / zmax.max(1.0));
        }
        worst
    }

    fn lift_x(&self, w: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0; self.elim.model.num_vars()];
        for (k, &j) in self.used.iter().enumerate() {
            z[j] = w[k];
        }
        self.elim.lift(&z)
    }

    fn lift_duals(&self, duals: &[Vec<Vec<C64>>]) -> Vec<Vec<Vec<C64>>> {
        (0..self.bases.len())
            .map(|b| {
                let n = self.sizes[b];
                match (self.index[b], &self.bases[b]) {
                    (None, _) => vec![vec![C64::new(0.0, 0.0); n]; n],
                    (Some(i), None) => duals[i].clone(),
                    (Some(i), Some(v)) => {
                        // G = conj(V) G' Vᵀ
                        let k = v.ncols();
                        let g = Mat::<C64>::from_fn(k, k, |r, c| duals[i][r][c]);
                        let vc = Mat::<C64>::from_fn(n, k, |r, c| v[(r, c)].conj());
                        let t = crate::dense::matmul(vc.as_ref(), g.as_ref());
                        let full = crate::dense::matmul(t.as_ref(), v.as_ref().transpose());
                        (0..n).map(|r| (0..n).map(|c| full[(r, c)]).collect()).collect()
                    }
                }
            })
            .collect()
    }
}

/// `max_j |Σ_b ⟨G_b, F_j⟩ − c_j| / (1 + |c_j|)` (the duals' convention:
/// both triangles of each `F_j`).
fn dual_residual(model: &SdpModel, duals: &[Vec<Vec<C64>>]) -> f64 {
    let mut acc = vec![0.0_f64; model.num_vars()];
    for (l, g) in model.lmis().iter().zip(duals) {
        for (j, f) in &l.fj {
            for &(r, c, v) in f {
                let mut t = g[r][c] * v;
                if r != c {
                    t += g[c][r] * v.conj();
                }
                acc[*j] += t.re;
            }
        }
    }
    acc.iter()
        .zip(model.objective())
        .map(|(a, c)| (a - c).abs() / (1.0 + c.abs()))
        .fold(0.0_f64, f64::max)
}

/// Solve an equality-free model, with facial reduction if the plain solve
/// does not converge.
pub(crate) fn solve_with_facial_reduction(
    model: &SdpModel,
    opts: &SolverOptions,
) -> Result<SdpSolution, SolverError> {
    let (first, trace) = solve_lmi_traced(model, opts)?;
    if first.status == Status::Optimal {
        return Ok(first);
    }
    let say = |msg: String| {
        if opts.verbose {
            println!("facial reduction: {}", msg);
        }
    };
    let mut stages: Vec<Stage> = Vec::new();
    let mut removed = vec![0usize; model.lmis().len()];
    let mut current: (SdpSolution, Trace) = (first.clone(), trace);
    // every round's solves count, also when the unreduced result is kept
    let mut profile = first.profile.clone();
    let keep_first = |profile: SolveProfile| SdpSolution {
        profile,
        ..first.clone()
    };
    let mut iterations = first.iterations;
    for round in 1..=MAX_ROUNDS {
        let input = stages.last().map_or(model, |s| &s.model);
        let candidates = candidate_directions(input, &current.0, &current.1);
        let ncand: usize = candidates
            .iter()
            .map(|c| c.as_ref().map_or(0, |b| b.ncols()))
            .sum();
        if ncand == 0 {
            say(format!("round {}: no diverging Gram directions", round));
            break;
        }
        let blocks: Vec<bool> = candidates.iter().map(|c| c.is_some()).collect();
        let cert = match certify(input, &blocks, opts, &mut profile) {
            Ok(c) => c,
            Err(e) => {
                say(format!("round {}: {} candidate directions, {}", round, ncand, e));
                break;
            }
        };
        let dirs = cert.dirs;
        let count: usize = dirs
            .iter()
            .map(|d| d.as_ref().map_or(0, |(u, _)| u.ncols()))
            .sum();
        let stage = match reduce(input, &dirs, cert.rank_tol, true) {
            Ok(s) => s,
            Err(e) => {
                say(format!("round {}: {}", round, e));
                break;
            }
        };
        say(format!(
            "round {}: {} directions certified and removed ({} candidates), {} -> {} variables",
            round,
            count,
            ncand,
            input.num_vars(),
            stage.model.num_vars()
        ));
        let (sol, tr) = match solve_lmi_traced(&stage.model, opts) {
            Ok(r) => r,
            Err(e) => {
                say(format!("round {}: reduced solve failed: {}", round, e));
                break;
            }
        };
        profile.absorb(&sol.profile);
        iterations += sol.iterations;
        // removed directions per original block: `orig[i]` is the original
        // block behind the input model's block i
        let mut orig: Vec<usize> = (0..model.lmis().len()).collect();
        for s in &stages {
            let mut next = vec![usize::MAX; s.model.lmis().len()];
            for (b, idx) in s.index.iter().enumerate() {
                if let Some(i) = idx {
                    next[*i] = orig[b];
                }
            }
            orig = next;
        }
        for (b, d) in dirs.iter().enumerate() {
            if let Some((u, _)) = d {
                removed[orig[b]] += u.ncols();
            }
        }
        stages.push(stage);
        let done = sol.status == Status::Optimal;
        current = (sol, tr);
        if done {
            break;
        }
    }
    if stages.is_empty() {
        return Ok(keep_first(profile));
    }
    if !current.0.is_converged(opts.accept_tol) {
        say("the reduced problem did not converge either; keeping the unreduced result".to_string());
        return Ok(keep_first(profile));
    }
    // lift through the stages, last first
    let mut x = current.0.x.clone();
    let mut duals = current.0.duals.clone();
    let mut kernel_residual = 0.0_f64;
    for (i, s) in stages.iter().enumerate().rev() {
        x = s.lift_x(&x);
        duals = s.lift_duals(&duals);
        let input = if i == 0 { model } else { &stages[i - 1].model };
        kernel_residual = kernel_residual.max(s.kernel_residual(input, &x));
    }
    let objective = model.objective_value(&x);
    // consistency: the reduced problem is (up to the √η sliver) a
    // restriction of the original, so its optimum cannot be better than a
    // nearly feasible original point; being clearly worse than the
    // unreduced iterate means the reduction cut off feasible points
    let worse = if model.is_minimize() {
        objective - first.objective
    } else {
        first.objective - objective
    };
    let slack = opts.accept_tol * (1.0 + first.objective.abs())
        + first.relative_gap.abs() * (1.0 + first.objective.abs());
    if first.objective.is_finite() && worse > slack {
        say(format!(
            "reduced objective {:.10} is worse than the unreduced iterate's {:.10} by more than its accuracy; keeping the unreduced result",
            objective, first.objective
        ));
        return Ok(keep_first(profile));
    }
    let min_eigenvalue = (0..model.lmis().len())
        .map(|b| model.min_eigenvalue(b, &x))
        .fold(f64::INFINITY, f64::min);
    say(format!(
        "{} round(s), objective {:.10} (unreduced best iterate {:.10}), smallest eigenvalue of the original blocks {:.2e}, kernel residual {:.2e}",
        stages.len(),
        objective,
        first.objective,
        min_eigenvalue,
        kernel_residual
    ));
    let dual_residual = dual_residual(model, &duals);
    let mut warnings = current.0.warnings.clone();
    if dual_residual > 1e-6 {
        let w = format!(
            "facial reduction: the returned duals are not dual feasible for the original problem              (relative residual {:.1e}); a certificate built from them does not prove the bound",
            dual_residual
        );
        if opts.warnings {
            eprintln!("warning: {w}");
        }
        warnings.push(w);
    }
    Ok(SdpSolution {
        x,
        objective,
        duals,
        iterations,
        facial_reduction: Some(FacialReduction {
            rounds: stages.len(),
            removed,
            unreduced_objective: first.objective,
            min_eigenvalue,
            kernel_residual,
            dual_residual,
        }),
        profile,
        warnings,
        ..current.0
    })
}

/// Per block: the candidate directions (the Gram eigenvectors that grew by
/// `GROWTH` while `μ` fell by `MU_RATIO`) as the columns of the starting
/// certificate `B`, or `None`.
fn candidate_directions(model: &SdpModel, sol: &SdpSolution, trace: &Trace) -> Vec<Option<Mat<C64>>> {
    let none = || model.lmis().iter().map(|_| None).collect();
    if trace.mu.is_nan() || trace.mu <= 0.0 {
        return none();
    }
    let Some((_, old)) = trace
        .snapshots
        .iter()
        .rev()
        .find(|(mu, _)| *mu >= MU_RATIO * trace.mu)
    else {
        return none();
    };
    model
        .lmis()
        .iter()
        .enumerate()
        .map(|(b, lmi)| {
            let n = lmi.size;
            if n == 0 {
                return None;
            }
            // X = Gᵀ
            let x = Mat::<C64>::from_fn(n, n, |i, j| sol.duals[b][j][i]);
            let xo = Mat::<C64>::from_fn(n, n, |i, j| old[b][j][i]);
            let (lam, u) = hermitian_eigen_checked(&x, lmi.real)?;
            let grown: Vec<usize> = (0..n)
                .filter(|&k| {
                    let col = u.col(k);
                    let mut prev = C64::new(0.0, 0.0);
                    for i in 0..n {
                        for j in 0..n {
                            prev += col[i].conj() * xo[(i, j)] * col[j];
                        }
                    }
                    lam[k] > 0.0 && lam[k] >= GROWTH * prev.re.max(0.0)
                })
                .collect();
            if grown.is_empty() {
                return None;
            }
            // equal weights: the certificate should be of maximal rank (in the
            // relative interior of the certificates supported on the
            // candidates), and Gauss–Newton with minimum-norm steps stays
            // close to its starting point
            Some(Mat::<C64>::from_fn(n, grown.len(), |i, j| u[(i, grown[j])]))
        })
        .collect()
}

/// A certified facial reduction: `(U, V)` per block (removed directions and
/// an orthonormal basis of their complement), and the rank tolerance for
/// eliminating `U* Z(y) = 0`.
struct Certified {
    dirs: Vec<Option<(Mat<C64>, Mat<C64>)>>,
    rank_tol: f64,
}

/// Compute a maximal-rank facial-reduction certificate on the blocks where
/// the Gram matrix diverged, with the auxiliary problem of Cheung, Schurr
/// and Wolkowicz ("Preprocessing and regularization for degenerate
/// semidefinite programs", 2013), in its ℓ∞ form:
///
/// ```text
///   min δ   s.t.   −δ ≤ ⟨F_j, D⟩ / ‖F_j‖ ≤ δ   (every variable j, and F_0),
///                  Σ_b tr D_b = 1,   D_b ⪰ 0.
/// ```
///
/// It is strictly feasible (`D = I / n`, δ large), so the interior-point
/// solver handles it well; its optimum is `δ = 0` exactly when a
/// certificate exists, and the interior-point solution lies in the relative
/// interior of the optimal set, i.e. `D` has maximal rank: its range is the
/// whole kernel that a single certificate can expose.  Blocks without
/// diverging Gram directions are left out (`D_b = 0` there), which loses
/// nothing: a certificate vanishes on blocks with a strictly feasible part.
///
/// The range of `D` is read off its eigenvalues, which stay `O(λ_max)` on
/// the range and go to zero off it, like `μ` under strict complementarity
/// and like `√μ` without it (the usual case for these degenerate
/// problems); the cut is the geometric mean of `λ_max` and the slower rate
/// at the solver's tolerance, `tol^(1/4) λ_max`.  By Davis–Kahan the computed range is then accurate to
/// `θ = λ_out / λ_in` (largest eigenvalue below the cut over the smallest
/// above it), which bounds the error of the equalities `U* Z(y) = 0`; the
/// reduction is accepted only if `10 θ` stays below a tenth of `accept_tol`,
/// and `10 θ` is the rank tolerance for eliminating them.
fn certify(
    model: &SdpModel,
    blocks: &[bool],
    opts: &SolverOptions,
    profile: &mut SolveProfile,
) -> Result<Certified, String> {
    let m = model.num_vars();
    let mut aux = SdpModel::new();
    let delta = aux.free_var("delta");
    let mut dvars: Vec<Option<crate::sdp::MatrixVar>> = Vec::new();
    let mut trace = Affine::zero();
    for (b, lmi) in model.lmis().iter().enumerate() {
        if blocks[b] {
            let d = aux.psd_var(lmi.size, lmi.real, format!("D{}", b));
            trace += &d.trace();
            dvars.push(Some(d));
        } else {
            dvars.push(None);
        }
    }
    let nvars = aux.num_vars() - 1;
    if nvars == 0 {
        return Err("no candidate blocks".to_string());
    }
    if nvars > opts.max_variables {
        return Err(format!(
            "the certificate problem has {} variables (max_variables {})",
            nvars, opts.max_variables
        ));
    }
    // v_j = ⟨F_j, D⟩ / ‖F_j‖ (row m: F_0), ⟨F, D⟩ = Re tr(F D)
    let mut v: Vec<Affine> = vec![Affine::zero(); m + 1];
    let mut norms = vec![0.0_f64; m + 1];
    for (b, lmi) in model.lmis().iter().enumerate() {
        for (j, f) in &lmi.fj {
            norms[*j] += norm_sq(f);
        }
        norms[m] += norm_sq(&lmi.f0);
        let Some(d) = &dvars[b] else { continue };
        let rows = lmi
            .fj
            .iter()
            .map(|(j, f)| (*j, f))
            .chain(std::iter::once((m, &lmi.f0)));
        for (row, f) in rows {
            for (r, c, val) in f {
                // Re(conj(F_rc) D_rc) over both triangles
                if r == c {
                    v[row].add_scaled(&Affine::var(d.re_var(*r, *c)), val.re);
                } else {
                    v[row].add_scaled(&Affine::var(d.re_var(*r, *c)), 2.0 * val.re);
                    if let Some(im) = d.im_var(*r, *c) {
                        v[row].add_scaled(&Affine::var(im), 2.0 * val.im);
                    }
                }
            }
        }
    }
    for (row, e) in v.iter().enumerate() {
        if e.is_zero(0.0) {
            continue;
        }
        let e = e.scale(1.0 / norms[row].sqrt().max(1e-300));
        aux.nonneg(&(Affine::var(delta) - e.clone()), "delta - v");
        aux.nonneg(&(Affine::var(delta) + e), "delta + v");
    }
    aux.equal(&trace, &Affine::constant(1.0));
    aux.minimize(&Affine::var(delta));
    let aux_opts = SolverOptions {
        facial_reduction: false,
        verbose: false,
        ..opts.clone()
    };
    let sol = aux
        .solve_inner(&aux_opts)
        .map_err(|e| format!("certificate problem: {}", e))?;
    profile.absorb(&sol.profile);
    let dstar = sol.x[delta];
    if opts.verbose {
        println!(
            "facial reduction: certificate problem ({} variables): δ = {:.2e}, {:?} in {} iterations",
            nvars, dstar, sol.status, sol.iterations
        );
    }
    if !sol.is_converged(opts.accept_tol) {
        return Err(format!("certificate problem did not converge ({:?})", sol.status));
    }
    // δ is zero up to the solver's accuracy iff a certificate exists
    if dstar > 10.0 * opts.tol.max(sol.relative_gap) {
        return Err(format!("no certificate: δ = {:.2e}", dstar));
    }
    // eigen-decompositions of the D blocks and the global rank decision
    let mut eig: Vec<Option<(Vec<f64>, Mat<C64>)>> = Vec::new();
    let mut lmax = 0.0_f64;
    for (b, lmi) in model.lmis().iter().enumerate() {
        let Some(d) = &dvars[b] else {
            eig.push(None);
            continue;
        };
        let n = lmi.size;
        let dv = d.value(&sol.x);
        let dm = Mat::<C64>::from_fn(n, n, |i, j| dv[i][j]);
        let e = hermitian_eigen_checked(&dm, lmi.real).ok_or("eigendecomposition failed")?;
        lmax = e.0.iter().cloned().fold(lmax, f64::max);
        eig.push(Some(e));
    }
    let cut = opts.tol.powf(0.25) * lmax;
    let (mut l_in, mut l_out) = (f64::INFINITY, 0.0_f64);
    for e in eig.iter().flatten() {
        for &l in &e.0 {
            if l >= cut {
                l_in = l_in.min(l);
            } else {
                l_out = l_out.max(l);
            }
        }
    }
    if !l_in.is_finite() {
        return Err("certificate is zero".to_string());
    }
    let theta = l_out / l_in;
    let rank_tol = RANK_TOL.max(10.0 * theta);
    if opts.verbose {
        println!(
            "facial reduction: certificate eigenvalues ≥ {:.2e} kept, ≤ {:.2e} dropped: range accurate to θ = {:.1e}",
            l_in, l_out, theta
        );
    }
    if 10.0 * theta > 0.1 * opts.accept_tol {
        return Err(format!(
            "certificate range not resolved (θ = {:.1e} against accept_tol {:.0e})",
            theta, opts.accept_tol
        ));
    }
    let dirs = model
        .lmis()
        .iter()
        .zip(eig)
        .map(|(lmi, e)| {
            let (lam, u) = e?;
            let n = lmi.size;
            let keep: Vec<usize> = (0..n).filter(|&i| lam[i] >= cut).collect();
            if keep.is_empty() {
                return None;
            }
            let rest: Vec<usize> = (0..n).filter(|&i| lam[i] < cut).collect();
            let pick = |idx: &[usize]| Mat::<C64>::from_fn(n, idx.len(), |i, j| u[(i, idx[j])]);
            Some((pick(&keep), pick(&rest)))
        })
        .collect();
    Ok(Certified { dirs, rank_tol })
}

/// Add `U* Z(y) [U V] = 0` (upper triangle of the `U`–`U` part), eliminate,
/// and restrict the blocks to `V`.  `dirs` holds `(U, V)` per block.
/// Without `equalities` only the restriction is done (for directions that
/// every coefficient matrix already annihilates).
fn reduce(
    model: &SdpModel,
    dirs: &[Option<(Mat<C64>, Mat<C64>)>],
    rank_tol: f64,
    equalities: bool,
) -> Result<Stage, String> {
    let mut m1 = model.clone();
    for (lmi, d) in model.lmis().iter().zip(dirs).filter(|_| equalities) {
        let Some((u, v)) = d else { continue };
        let n = lmi.size;
        let k = u.ncols();
        let w = Mat::<C64>::from_fn(n, n, |i, j| if j < k { u[(i, j)] } else { v[(i, j - k)] });
        let proj = |f: &SparseHerm| -> Mat<C64> {
            let fd = to_dense(f, n);
            crate::dense::adjoint_mul(u.as_ref(), crate::dense::matmul(fd.as_ref(), w.as_ref()).as_ref())
        };
        let t0 = proj(&lmi.f0);
        let tj: Vec<(usize, Mat<C64>)> = lmi.fj.iter().map(|(j, f)| (*j, proj(f))).collect();
        let scale = tj
            .iter()
            .flat_map(|(_, t)| (0..k).flat_map(move |p| (0..n).map(move |i| t[(p, i)].norm())))
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        for p in 0..k {
            for i in p..n {
                let parts: &[bool] = if lmi.real || i == p {
                    &[false]
                } else {
                    &[false, true]
                };
                for &imag in parts {
                    let part = |c: C64| if imag { c.im } else { c.re };
                    let row: Vec<(usize, f64)> = tj
                        .iter()
                        .map(|(j, t)| (*j, part(t[(p, i)])))
                        .filter(|(_, c)| c.abs() > 1e-15 * scale)
                        .collect();
                    let rhs = -part(t0[(p, i)]);
                    // The entries carry absolute errors relative to `scale`
                    // (rounding, and the certificate's θ), so a row is
                    // judged against `scale`, not against its own norm: one
                    // within `rank_tol · scale` of the zero row is implied
                    // and dropped here.  `eliminate_qr` scales every row to
                    // unit norm before its rank decision, which would turn
                    // such a row of noise (structurally zero entries of
                    // `U* F_j W`, ~1e-15) into an independent equation and
                    // pin variables that should stay free.
                    let norm = (row.iter().map(|(_, c)| c * c).sum::<f64>() + rhs * rhs).sqrt();
                    if norm <= rank_tol * scale {
                        continue;
                    }
                    m1.add_equality(row, rhs);
                }
            }
        }
    }
    // the rows are implied by feasibility (certified) up to `rank_tol`;
    // eliminate_qr verifies that the rows it treats as dependent hold to
    // within 100 rank_tol, and reports an inconsistency as an error
    let elim = m1.eliminate_qr(rank_tol)?;
    // restrict the blocks of the eliminated model
    let em = &elim.model;
    let mut lmis: Vec<Lmi> = Vec::new();
    let mut index = Vec::new();
    let mut bases = Vec::new();
    let mut sizes = Vec::new();
    for (lmi, d) in em.lmis().iter().zip(dirs) {
        sizes.push(lmi.size);
        match d {
            None => {
                index.push(Some(lmis.len()));
                bases.push(None);
                lmis.push(lmi.clone());
            }
            Some((_, v)) if v.ncols() == 0 => {
                index.push(None);
                bases.push(None);
            }
            Some((_, v)) => {
                let n = lmi.size;
                let k = v.ncols();
                let sel = selection(v);
                let restrict = |f: &SparseHerm| -> SparseHerm {
                    if let Some(map) = &sel {
                        // a principal submatrix: the order of the kept
                        // coordinates is preserved, so upper entries stay upper
                        return f
                            .iter()
                            .filter_map(|&(r, c, x)| Some((map[r]?, map[c]?, x)))
                            .collect();
                    }
                    let fd = to_dense(f, n);
                    let t = crate::dense::adjoint_mul(
                        v.as_ref(),
                        crate::dense::matmul(fd.as_ref(), v.as_ref()).as_ref(),
                    );
                    let mut out = Vec::new();
                    for i in 0..k {
                        for j in i..k {
                            let val = if i == j {
                                C64::new(t[(i, i)].re, 0.0)
                            } else {
                                (t[(i, j)] + t[(j, i)].conj()) * 0.5
                            };
                            let val = if lmi.real { C64::new(val.re, 0.0) } else { val };
                            if val.norm() > 1e-14 {
                                out.push((i, j, val));
                            }
                        }
                    }
                    out
                };
                index.push(Some(lmis.len()));
                bases.push(Some(v.clone()));
                lmis.push(Lmi {
                    size: k,
                    real: lmi.real,
                    f0: restrict(&lmi.f0),
                    fj: lmi
                        .fj
                        .iter()
                        .map(|(j, f)| (*j, restrict(f)))
                        .filter(|(_, f)| !f.is_empty())
                        .collect(),
                    label: lmi.label.clone(),
                });
            }
        }
    }
    // variables that no longer occur in any block are fixed at 0; that is
    // only sound if they do not enter the objective
    let nz = em.num_vars();
    let mut occurs = vec![false; nz];
    for l in &lmis {
        for j in l.fj.keys() {
            occurs[*j] = true;
        }
    }
    let obj_scale = em
        .objective()
        .iter()
        .fold(0.0_f64, |a, c| a.max(c.abs()))
        .max(1e-300);
    for j in 0..nz {
        if !occurs[j] && em.objective()[j].abs() > 1e-9 * obj_scale {
            return Err(format!(
                "variable {} left every block but enters the objective",
                j
            ));
        }
    }
    let used: Vec<usize> = (0..nz).filter(|&j| occurs[j]).collect();
    let mut renum = vec![usize::MAX; nz];
    for (w, &j) in used.iter().enumerate() {
        renum[j] = w;
    }
    let mut out = SdpModel::new();
    for &j in &used {
        out.free_var(em.names()[j].clone());
    }
    for l in lmis {
        out.add_lmi(Lmi {
            fj: l.fj.into_iter().map(|(j, f)| (renum[j], f)).collect(),
            ..l
        });
    }
    let obj: Vec<f64> = used.iter().map(|&j| em.objective()[j]).collect();
    out.set_raw_objective(obj, em.objective_constant(), em.is_minimize());
    let removed = dirs.iter().map(|d| d.as_ref().map(|(u, _)| u.clone())).collect();
    Ok(Stage {
        elim,
        used,
        bases,
        removed,
        index,
        sizes,
        model: out,
    })
}

/// Directions annihilated by every coefficient matrix of a block:
/// `F_0 v = 0` and `F_j v = 0` for all `j`, so `Z(y) v = 0` for every `y`.
/// Removing them (`Z ↦ V* Z V` for the complement `V`) is an exact
/// congruence: nothing is lost and no tolerance judgement is involved
/// beyond rounding.  Candidates come from the kernel of the stacked `F`s by
/// the rank-revealing sparse QR (residual `≈ ε κ`); the eigenvectors of
/// `Σ F*F` (residual `≈ ε κ²`, so true kernel directions of ill-conditioned
/// blocks fail the check) are only a fallback if the QR gives up.
/// Each candidate is verified directly, `‖F v‖ ≤ 100 ε ‖F‖` for every `F`.
fn common_kernel(model: &SdpModel) -> (Vec<Option<(Mat<C64>, Mat<C64>)>>, KernelStats) {
    let mut stats = KernelStats::default();
    let dirs = model
        .lmis()
        .iter()
        .map(|lmi| {
            let n = lmi.size;
            if n == 0 {
                return None;
            }
            // full entry lists, duplicates merged: (row, col, value), both triangles
            let mats: Vec<Vec<(usize, usize, C64)>> = std::iter::once(&lmi.f0)
                .chain(lmi.fj.values())
                .map(full_entries)
                .collect();
            let fnorms: Vec<f64> = mats
                .iter()
                .map(|f| f.iter().map(|(_, _, v)| v.norm_sqr()).sum::<f64>().sqrt())
                .collect();
            let fmax = fnorms.iter().copied().fold(0.0_f64, f64::max);
            // a random combination of the matrices first: one n × n
            // eigenvalue computation, less than an interior-point iteration.
            // It shows most blocks kernel-free, and otherwise its
            // near-null eigenspace contains the common kernel, which is found
            // inside it; the QR of all stacked rows, whose cost grows with
            // the rows and their fill, is only the fallback
            stats.searched += 1;
            let (q, cand): (Mat<C64>, Vec<usize>) = match combination_kernel(&mats, n, lmi.real, fmax) {
                Combination::KernelFree => {
                    stats.searched -= 1;
                    stats.certified += 1;
                    return None;
                }
                Combination::Candidates(c, rows) => {
                    stats.rows += rows;
                    let k = c.ncols();
                    (c, (0..k).collect())
                }
                Combination::Failed => {
                    stats.rows += mats
                        .iter()
                        .map(|f| {
                            let mut r: Vec<usize> = f.iter().map(|(a, _, _)| *a).collect();
                            r.sort_unstable();
                            r.dedup();
                            r.len()
                        })
                        .sum::<usize>();
                    match kernel_by_qr(&mats, n, lmi.real) {
                        Some(found) => found,
                        None => kernel_by_eigen(&mats, n, lmi.real)?,
                    }
                }
            };
            let kernel: Vec<usize> = cand
                .into_iter()
                .filter(|&k| {
                    let col = q.col(k);
                    // relative to each matrix's own norm: against the
                    // block's largest, a matrix 10¹⁵ times smaller than it
                    // has every vector as "kernel", and its constraint was
                    // dropped (F = 1e8 E₁₁ beside 1e-7 E₃₃: optimum 1
                    // instead of 5)
                    mats.iter().zip(&fnorms).all(|(f, &norm)| {
                        let mut fv = vec![C64::new(0.0, 0.0); n];
                        for &(a, b, v) in f {
                            fv[a] += v * col[b];
                        }
                        let worst = fv.iter().map(|x| x.norm()).fold(0.0_f64, f64::max);
                        worst <= 100.0 * f64::EPSILON * norm
                    })
                })
                .collect();
            if kernel.is_empty() {
                return None;
            }
            let u = Mat::<C64>::from_fn(n, kernel.len(), |i, j| q[(i, kernel[j])]);
            let v = coordinate_complement(&u);
            Some((u, v))
        })
        .collect();
    (dirs, stats)
}

/// A complement of the kernel basis `U` (`n × k`, orthonormal columns) made
/// of unit vectors: the coordinates outside `k` pivot rows of `U`, chosen by
/// column-pivoted Gram–Schmidt on `Uᵀ` (twice per step), so that `[U V]` is
/// invertible and well conditioned.  Any complement works for the
/// congruence: with `Z U = 0` for every `Z` of the block,
/// `[U V]* Z [U V] = diag(0, V* Z V)`, and for unit vectors `V* Z V` is a
/// principal submatrix, which keeps the restricted coefficient matrices as
/// sparse as the original ones.  An orthonormal complement would make them
/// dense, and the Schur assembly after it dense too.
fn coordinate_complement(u: &Mat<C64>) -> Mat<C64> {
    let (n, k) = (u.nrows(), u.ncols());
    // columns of Uᵀ: one length-k vector per coordinate
    let mut cols: Vec<Vec<C64>> = (0..n).map(|i| (0..k).map(|j| u[(i, j)]).collect()).collect();
    let mut pivot = vec![false; n];
    for _ in 0..k {
        let best = (0..n)
            .filter(|&i| !pivot[i])
            .max_by(|&a, &b| {
                let na: f64 = cols[a].iter().map(|x| x.norm_sqr()).sum();
                let nb: f64 = cols[b].iter().map(|x| x.norm_sqr()).sum();
                na.total_cmp(&nb)
            })
            .expect("k ≤ n");
        pivot[best] = true;
        let norm = cols[best].iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let q: Vec<C64> = cols[best].iter().map(|x| x / norm).collect();
        // project the other columns off q, twice for orthogonality
        for col in cols
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| !pivot[*i])
            .map(|(_, c)| c)
        {
            for _ in 0..2 {
                let d: C64 = q.iter().zip(col.iter()).map(|(a, b)| a.conj() * b).sum();
                for (c, a) in col.iter_mut().zip(&q) {
                    *c -= d * a;
                }
            }
        }
    }
    let rest: Vec<usize> = (0..n).filter(|&i| !pivot[i]).collect();
    Mat::<C64>::from_fn(n, rest.len(), |i, j| C64::new(f64::from(i == rest[j]), 0.0))
}

/// `Some(new index of each row)` when the columns of `v` are distinct unit
/// vectors in increasing order (a coordinate selection), else `None`.
fn selection(v: &Mat<C64>) -> Option<Vec<Option<usize>>> {
    let mut map = vec![None; v.nrows()];
    let mut last = None;
    for j in 0..v.ncols() {
        let mut hit = None;
        for i in 0..v.nrows() {
            let x = v[(i, j)];
            if x == C64::new(1.0, 0.0) {
                if hit.is_some() {
                    return None;
                }
                hit = Some(i);
            } else if x != C64::new(0.0, 0.0) {
                return None;
            }
        }
        let i = hit?;
        if last.is_some_and(|l| l >= i) {
            return None;
        }
        last = Some(i);
        map[i] = Some(j);
    }
    Some(map)
}

/// What [`common_kernel`] did, for the solve's [`SolveProfile`].
#[derive(Default)]
struct KernelStats {
    /// Blocks shown kernel-free by the random-combination test.
    certified: usize,
    /// Blocks searched for a common kernel, and the stacked rows factored.
    searched: usize,
    rows: usize,
}

/// The outcome of [`combination_kernel`].
enum Combination {
    /// No common kernel (the test of [`certified_kernel_free`]).
    KernelFree,
    /// Orthonormal candidate directions (`n × w`) containing the common
    /// kernel, and the stacked rows factored to find them.
    Candidates(Mat<C64>, usize),
    /// The eigensolver failed: search by QR.
    Failed,
}

/// Is the block free of a common kernel, shown without a factorization?
/// A direction passes the check in [`common_kernel`] only if
/// `‖F v‖_∞ ≤ 100 ε f_max` for every coefficient matrix `F` of the block,
/// so for any combination `R = Σ_k r_k F_k`
///
/// ```text
///   ‖R v‖₂ ≤ Σ_k |r_k| ‖F_k v‖₂ ≤ √n · 100 ε f_max · Σ_k |r_k|.
/// ```
///
/// If every eigenvalue of `R` exceeds that bound in absolute value (with
/// room for the rounding of `R` and of its eigenvalues), no direction can
/// pass, and the QR search would find nothing.  For a moment relaxation the
/// span of the `F` contains a positive definite moment matrix, so a generic
/// combination is far from singular: on the QMC level-2 relaxations the
/// smallest eigenvalue clears the bound by a factor 10⁸.  The test costs one
/// `n × n` eigenvalue computation, where the QR of the stacked rows took 57
/// s on QMC 9 (10⁵–10⁶ rows).  The weights are pseudo-random but fixed, so
/// solves stay reproducible.
#[cfg(test)]
fn certified_kernel_free(mats: &[Vec<(usize, usize, C64)>], n: usize, real: bool, fmax: f64) -> bool {
    matches!(combination_kernel(mats, n, real, fmax), Combination::KernelFree)
}

/// The random-combination test of [`certified_kernel_free`], and when it
/// fails, candidates for the common kernel: every common-kernel vector `v`
/// has `‖R v‖` below the test's bound, so it lies (up to rounding) in the
/// span `W` of the eigenvectors of `R` with eigenvalues below it.  That span
/// is small (the kernel plus accidental near-null directions of the
/// combination), and the common kernel is the null space of the stacked
/// `F_k W`: a QR of `rows × dim W` and an SVD, against a QR of all stacked
/// rows over `n` columns (10 s for a planted two-dimensional kernel in a
/// block of 300 with 750 dense-overlap matrices).  The candidates are
/// the right singular vectors mapped back, each then checked against every
/// `F` by [`common_kernel`].
fn combination_kernel(mats: &[Vec<(usize, usize, C64)>], n: usize, real: bool, fmax: f64) -> Combination {
    // splitmix64, mapped to [-1, 1)
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut weight = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let mut r = Mat::<C64>::zeros(n, n);
    let mut terms = vec![0u32; n * n];
    let mut wsum = 0.0;
    for f in mats {
        let w = weight();
        wsum += w.abs();
        for &(a, b, v) in f {
            r[(a, b)] += v * w;
            terms[a * n + b] += 1;
        }
    }
    let Some(lam) = hermitian_eigenvalues_checked(&r, real) else {
        return Combination::Failed;
    };
    let (smin, smax) = lam.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), x| {
        (lo.min(x.abs()), hi.max(x.abs()))
    });
    let eps = f64::EPSILON;
    let bound = (n as f64).sqrt() * 100.0 * eps * fmax * wsum;
    // each entry of R is a sum of up to `kmax` rounded terms, and a
    // backward-stable eigensolver is accurate to a small multiple of n ε ‖R‖
    let kmax = terms.iter().copied().max().unwrap_or(0) as f64;
    let rounding = (kmax + n as f64) * eps * (wsum * fmax + smax);
    let threshold = 10.0 * (bound + rounding);
    if smin > threshold {
        return Combination::KernelFree;
    }
    // the near-null eigenspace W of R
    let Some((lam, u)) = hermitian_eigen_checked(&r, real) else {
        return Combination::Failed;
    };
    let near: Vec<usize> = (0..n).filter(|&k| lam[k].abs() <= threshold).collect();
    if near.is_empty() || near.len() * 2 > n {
        return Combination::Failed; // nothing near null, or no reduction: QR
    }
    let w = Mat::<C64>::from_fn(n, near.len(), |i, j| u[(i, near[j])]);
    let k = w.ncols();
    // the nonzero rows of every F W, stacked
    let mut stacked: Vec<Vec<C64>> = Vec::new();
    let mut fw = vec![C64::new(0.0, 0.0); n * k];
    let mut touched = vec![false; n];
    for f in mats {
        for &(a, b, v) in f {
            touched[a] = true;
            for j in 0..k {
                fw[a * k + j] += v * w[(b, j)];
            }
        }
        for a in 0..n {
            if touched[a] {
                stacked.push(fw[a * k..(a + 1) * k].to_vec());
                fw[a * k..(a + 1) * k].fill(C64::new(0.0, 0.0));
                touched[a] = false;
            }
        }
    }
    let rows = stacked.len();
    let a = Mat::<C64>::from_fn(rows.max(k), k, |i, j| {
        if i < rows {
            stacked[i][j]
        } else {
            C64::new(0.0, 0.0)
        }
    });
    // R factor (k × k), then its right singular vectors: the null space of
    // the stack, with backward-stable accuracy (no squaring as in AᴴA)
    let qr = a.qr();
    let rk = qr.R().to_owned();
    let Ok(svd) = rk.svd() else {
        return Combination::Failed;
    };
    let cand = crate::dense::matmul(w.as_ref(), svd.V());
    Combination::Candidates(cand, rows)
}

/// Candidate common kernel of the Hermitian matrices `mats` (full entry
/// lists) from the rank-revealing sparse QR of their stacked rows (real
/// embedding `[[Re, −Im], [Im, Re]]` for complex blocks), orthonormalized
/// with a column-pivoted QR: `(Q, 0..k)` with `Q` unitary.  `None` when the
/// sparse QR gives up (fill limits).
fn kernel_by_qr(mats: &[Vec<(usize, usize, C64)>], n: usize, real: bool) -> Option<(Mat<C64>, Vec<usize>)> {
    use crate::sparse_qr::{QrOptions, SparseQr, SparseRows};
    let width = if real { n } else { 2 * n };
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
    for f in mats {
        let mut by_row: Vec<Vec<(usize, C64)>> = vec![Vec::new(); n];
        for &(a, b, v) in f {
            by_row[a].push((b, v));
        }
        for r in by_row.into_iter().filter(|r| !r.is_empty()) {
            if real {
                let mut row: Vec<(usize, f64)> = r
                    .iter()
                    .filter(|(_, v)| v.re != 0.0)
                    .map(|(b, v)| (*b, v.re))
                    .collect();
                row.sort_by_key(|(c, _)| *c);
                rows.push(row);
            } else {
                // F v = 0, v = x + i y:  Re F x − Im F y = 0,  Im F x + Re F y = 0
                let mut re_row: Vec<(usize, f64)> = Vec::new();
                let mut im_row: Vec<(usize, f64)> = Vec::new();
                for &(b, v) in &r {
                    if v.re != 0.0 {
                        re_row.push((b, v.re));
                        im_row.push((n + b, v.re));
                    }
                    if v.im != 0.0 {
                        re_row.push((n + b, -v.im));
                        im_row.push((b, v.im));
                    }
                }
                re_row.sort_by_key(|(c, _)| *c);
                im_row.sort_by_key(|(c, _)| *c);
                rows.push(re_row);
                rows.push(im_row);
            }
        }
    }
    let opts = QrOptions {
        rank_tol: 1e-10,
        ..Default::default()
    };
    let a = SparseRows { ncols: width, rows };
    let qr = SparseQr::factor(&a, &opts).ok()?;
    let kernel = qr.kernel_basis(&opts).ok()?;
    if kernel.is_empty() {
        return Some((Mat::<C64>::identity(n, n), Vec::new()));
    }
    let to_c = |v: &[(usize, f64)]| -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); n];
        for &(i, x) in v {
            if i < n {
                out[i].re += x;
            } else {
                out[i - n].im += x;
            }
        }
        out
    };
    let vecs: Vec<Vec<C64>> = kernel.iter().map(|v| to_c(v)).collect();
    let k_mat = Mat::<C64>::from_fn(n, vecs.len(), |i, j| vecs[j][i]);
    // orthonormal basis of the span (complex kernels come in pairs v, iv)
    let cp = k_mat.col_piv_qr();
    let r = cp.R();
    let d = r.nrows().min(r.ncols());
    let r0 = if d > 0 { r[(0, 0)].norm() } else { 0.0 };
    let rank = (0..d).take_while(|&i| r[(i, i)].norm() > 1e-8 * r0).count();
    Some((cp.compute_Q(), (0..rank).collect()))
}

/// Fallback candidates: the eigenvectors of `S = Σ F*F` with eigenvalues at
/// most `1e-12 λ_max` (`(U, indices)`, `U` unitary).
fn kernel_by_eigen(
    mats: &[Vec<(usize, usize, C64)>],
    n: usize,
    real: bool,
) -> Option<(Mat<C64>, Vec<usize>)> {
    {
        {
            // S = Σ F*F = Σ F², accumulated row by row of F: (F²)[a, b] = Σ_c F[a, c] F[c, b]
            // and F[a, c] = conj(F[c, a]), so each row c of F contributes the
            // outer product of its entries; the cost is Σ_c nnz(row c)²
            let mut s = Mat::<C64>::zeros(n, n);
            let mut rows: Vec<Vec<(usize, C64)>> = vec![Vec::new(); n];
            for f in mats {
                for r in rows.iter_mut() {
                    r.clear();
                }
                for &(c, b, v) in f {
                    rows[c].push((b, v));
                }
                for r in &rows {
                    for &(a, va) in r {
                        for &(b, vb) in r {
                            s[(a, b)] += va.conj() * vb;
                        }
                    }
                }
            }
            let (lam, u) = hermitian_eigen_checked(&s, real)?;
            let lmax = lam.iter().cloned().fold(0.0_f64, f64::max);
            Some((u, (0..n).filter(|&k| lam[k] <= 1e-12 * lmax).collect()))
        }
    }
}

/// Solve an equality-free model: remove the exact common kernels of the
/// blocks first (lossless), then solve, with facial reduction if
/// `opts.facial_reduction`.
pub(crate) fn solve_presolved(model: &SdpModel, opts: &SolverOptions) -> Result<SdpSolution, SolverError> {
    let t = Instant::now();
    let (dirs, stats) = common_kernel(model);
    let record = |p: &mut SolveProfile, seconds: f64| {
        p.presolve += seconds;
        p.kernel_certified += stats.certified;
        p.kernel_searched += stats.searched;
        p.kernel_rows += stats.rows;
    };
    let count: usize = dirs
        .iter()
        .map(|d| d.as_ref().map_or(0, |(u, _)| u.ncols()))
        .sum();
    if count == 0 {
        let presolve = t.elapsed().as_secs_f64();
        let mut sol = solve_kernel_free(model, opts)?;
        record(&mut sol.profile, presolve);
        return Ok(sol);
    }
    let stage = reduce(model, &dirs, RANK_TOL, false).map_err(SolverError)?;
    let presolve = t.elapsed().as_secs_f64();
    if opts.verbose {
        println!(
            "presolve: removed {} directions in the common kernel of the blocks' coefficient matrices",
            count
        );
    }
    // the complement of a block's full common kernel has none left
    let mut sol = solve_kernel_free(&stage.model, opts)?;
    let t = Instant::now();
    let x = stage.lift_x(&sol.x);
    let duals = stage.lift_duals(&sol.duals);
    record(&mut sol.profile, presolve + t.elapsed().as_secs_f64());
    // a regularized solve reports a certified bound, not the value at x
    let objective = if sol.regularization.is_some() {
        sol.objective
    } else {
        model.objective_value(&x)
    };
    // facial-reduction counts refer to the presolved blocks: map them back
    let facial_reduction = sol.facial_reduction.clone().map(|mut f| {
        f.removed = (0..model.lmis().len())
            .map(|b| stage.index[b].map_or(0, |i| f.removed.get(i).copied().unwrap_or(0)))
            .collect();
        f
    });
    Ok(SdpSolution {
        x,
        objective,
        duals,
        facial_reduction,
        ..sol
    })
}

/// Solve a model whose blocks have no common kernel left.
fn solve_kernel_free(model: &SdpModel, opts: &SolverOptions) -> Result<SdpSolution, SolverError> {
    if opts.regularization > 0.0 {
        solve_regularized(model, opts)
    } else if opts.facial_reduction {
        solve_with_facial_reduction(model, opts)
    } else {
        crate::solver::solve_lmi(model, opts)
    }
}

/// Solve the regularized problem `Z(y) + ε s_b I ⪰ 0` and report the
/// original problem's dual bound at its Gram matrices.
///
/// The shift makes every feasible point of the original strictly feasible,
/// so the interior-point method is well posed.  The Gram matrices `X` of
/// the regularized problem satisfy the same equations `A(X) = c`, so they
/// are dual feasible for the original problem too, with value
/// `−⟨F_0, X⟩ = (regularized optimum) + Σ_b ε s_b tr X_b`: a valid bound,
/// tighter than the regularized optimum.  Its distance to the true optimum
/// is `c √ε` to leading order for singularity degree 1 (Sturm's error
/// bound), so a second solve at `100 ε` (well conditioned and cheap)
/// measures it: `bias ≈ |bound(100 ε) − bound(ε)| / 9`.
fn solve_regularized(model: &SdpModel, opts: &SolverOptions) -> Result<SdpSolution, SolverError> {
    let eps = opts.regularization;
    let (mut sol, bound, regularized_objective, trace_term) = shifted_solve(model, opts, eps)?;
    let check = shifted_solve(model, opts, 100.0 * eps).ok();
    if let Some(c) = &check {
        sol.profile.absorb(&c.0.profile);
    }
    let check = check.filter(|r| r.0.is_converged(opts.accept_tol) && r.1.is_finite());
    let bias_estimate = check.as_ref().map(|r| (r.1 - bound).abs() / 9.0);
    // the bound lies below the optimum for a minimization, above it for a
    // maximization
    let extrapolated_objective =
        bias_estimate.map(|b| if model.is_minimize() { bound + b } else { bound - b });
    if opts.verbose {
        println!(
            "regularization ε = {:.1e}: regularized optimum {:.10}, certified bound {:.10} ({:?}), estimated bias {:?}",
            eps, regularized_objective, bound, sol.status, bias_estimate
        );
    }
    Ok(SdpSolution {
        objective: bound,
        regularization: Some(crate::solver::Regularization {
            epsilon: eps,
            regularized_objective,
            trace_term,
            bias_estimate,
            extrapolated_objective,
        }),
        ..sol
    })
}

/// One regularized solve: `(solution, certified bound, regularized
/// optimum, trace term)`, values in the model's objective units.
fn shifted_solve(
    model: &SdpModel,
    opts: &SolverOptions,
    eps: f64,
) -> Result<(SdpSolution, f64, f64, f64), SolverError> {
    let mut shifted = model.clone();
    let mut lmis: Vec<Lmi> = Vec::new();
    let mut scales = Vec::new();
    for lmi in model.lmis() {
        let s = std::iter::once(&lmi.f0)
            .chain(lmi.fj.values())
            .flat_map(|f| f.iter().map(|(_, _, v)| v.norm()))
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let mut f0: std::collections::BTreeMap<(usize, usize), C64> =
            lmi.f0.iter().map(|(r, c, v)| ((*r, *c), *v)).collect();
        for i in 0..lmi.size {
            *f0.entry((i, i)).or_insert(C64::new(0.0, 0.0)) += C64::new(eps * s, 0.0);
        }
        lmis.push(Lmi {
            f0: f0.into_iter().map(|((r, c), v)| (r, c, v)).collect(),
            ..lmi.clone()
        });
        scales.push(s);
    }
    shifted.set_lmis(lmis);
    let inner = SolverOptions {
        regularization: 0.0,
        facial_reduction: false,
        ..opts.clone()
    };
    // the shift adds no common kernel: a direction with F_j v = 0 for all j
    // and (F_0 + ε s I) v = 0 would make the unshifted problem infeasible
    let sol = solve_kernel_free(&shifted, &inner)?;
    // ⟨F_0, X⟩ with Σ_ab G_ab F_ab = ⟨F, X⟩ (G = Xᵀ the reported dual)
    let mut f0x = 0.0;
    let mut trace_term = 0.0;
    for (b, lmi) in model.lmis().iter().enumerate() {
        let g = &sol.duals[b];
        f0x += lmi
            .f0
            .iter()
            .map(|(r, c, v)| {
                if r == c {
                    (g[*r][*c] * v).re
                } else {
                    (g[*r][*c] * v + g[*c][*r] * v.conj()).re
                }
            })
            .sum::<f64>();
        trace_term += eps * scales[b] * (0..lmi.size).map(|i| g[i][i].re).sum::<f64>();
    }
    let bound = model.value_from_optimum(-f0x);
    let regularized_objective = model.objective_value(&sol.x);
    Ok((sol, bound, regularized_objective, trace_term))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Rows of `U* Z(y) [U V] = 0` that are structurally zero but
    /// carry rounding noise must not become equations.  Block
    /// `Z(y) = diag(y1, 1, 1) + y2 (E23 + E32)` with face direction `U = e1`:
    /// the only real equation is `y1 = 0`, so `y2` stays free.  With `U`
    /// perturbed at the rounding level, the rows `u* Z(y) v_k` are ~1e-15;
    /// scaled to unit norm by the QR they used to pin `y2` as well.
    #[test]
    fn noise_rows_of_the_certified_equalities_are_dropped() {
        let c = |x: f64| C64::new(x, 0.0);
        let mut model = SdpModel::new();
        let y1 = model.free_var("y1");
        let y2 = model.free_var("y2");
        let mut fj = BTreeMap::new();
        fj.insert(y1, vec![(0, 0, c(1.0))]);
        fj.insert(y2, vec![(1, 2, c(1.0))]);
        model.add_lmi(Lmi {
            size: 3,
            real: true,
            f0: vec![(1, 1, c(1.0)), (2, 2, c(1.0))],
            fj,
            label: "test".into(),
        });
        model.minimize(&Affine::var(y2));
        for noise in [0.0, 3e-15] {
            let mut u = [1.0, noise, -0.7 * noise];
            let un = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            u.iter_mut().for_each(|x| *x /= un);
            // orthonormal complement by Gram–Schmidt on e2, e3
            let mut basis: Vec<[f64; 3]> = vec![u];
            for e in [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                let mut v = e;
                for b in &basis {
                    let d: f64 = (0..3).map(|i| b[i] * v[i]).sum();
                    (0..3).for_each(|i| v[i] -= d * b[i]);
                }
                let vn = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                basis.push(v.map(|x| x / vn));
            }
            let umat = Mat::<C64>::from_fn(3, 1, |i, _| c(basis[0][i]));
            let vmat = Mat::<C64>::from_fn(3, 2, |i, j| c(basis[j + 1][i]));
            let stage = reduce(&model, &[Some((umat, vmat))], RANK_TOL, true).expect("reduce");
            assert_eq!(stage.model.num_vars(), 1, "noise {}: y2 must stay free", noise);
        }
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;
    use std::collections::BTreeMap;

    /// The common kernel `e0 − e1` next to a direction of singular value
    /// ~1e-6: the eigenvectors of `Σ F*F` (gap ~1e-12) are too inaccurate to
    /// pass the direct check, the QR route finds it.
    #[test]
    fn ill_conditioned_block_keeps_its_common_kernel() {
        let c = |x: f64| C64::new(x, 0.0);
        let f0: SparseHerm = vec![(0, 0, c(1.0)), (0, 1, c(1.0)), (1, 1, c(1.0)), (2, 2, c(1e-6))];
        let f1: SparseHerm = vec![
            (0, 0, c(1.0)),
            (0, 1, c(1.0)),
            (1, 1, c(1.0)),
            (0, 2, c(3e-7)),
            (1, 2, c(3e-7)),
        ];
        let mut m = SdpModel::new();
        let z = m.free_var("z");
        let mut fj = BTreeMap::new();
        fj.insert(z, f1);
        m.add_lmi(Lmi {
            size: 3,
            real: true,
            f0,
            fj,
            label: "B".into(),
        });
        let (dirs, _) = common_kernel(&m);
        let (u, v) = dirs[0].as_ref().expect("a common kernel");
        assert_eq!((u.ncols(), v.ncols()), (1, 2));
        let s = 1.0 / 2f64.sqrt();
        assert!((u[(0, 0)].norm() - s).abs() < 1e-12 && (u[(1, 0)] + u[(0, 0)]).norm() < 1e-12);
        assert!(u[(2, 0)].norm() < 1e-12);
    }

    /// One real block of size `n` with `F_0 = P` and `F_k = P G_k P` for
    /// dense pseudo-random symmetric `G_k` (`k = 1..=count`), `P` the
    /// projector `I − u uᵀ` (or `I` without `u`).
    fn projected_block(n: usize, count: usize, u: Option<&[f64]>) -> SdpModel {
        let mut state = 7u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        };
        let p: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| f64::from(i == j) - u.map_or(0.0, |u| u[i] * u[j]))
                    .collect()
            })
            .collect();
        let project = |g: &[Vec<f64>]| -> SparseHerm {
            // P G P, upper triangle
            let pg: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| (0..n).map(|k| p[i][k] * g[k][j]).sum()).collect())
                .collect();
            let mut out = Vec::new();
            for i in 0..n {
                for j in i..n {
                    let v: f64 = (0..n).map(|k| pg[i][k] * p[k][j]).sum();
                    if v.abs() > 1e-15 {
                        out.push((i, j, C64::new(v, 0.0)));
                    }
                }
            }
            out
        };
        let identity: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| f64::from(i == j)).collect())
            .collect();
        let mut m = SdpModel::new();
        let mut fj = BTreeMap::new();
        for k in 0..count {
            let mut g = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in i..n {
                    let v = next();
                    g[i][j] = v;
                    g[j][i] = v;
                }
            }
            fj.insert(m.free_var(format!("y{k}")), project(&g));
        }
        m.add_lmi(Lmi {
            size: n,
            real: true,
            f0: project(&identity),
            fj,
            label: "B".into(),
        });
        m
    }

    /// A generic block is shown kernel-free by the random-combination
    /// test, without the QR; a planted common kernel defeats the test, and
    /// the QR still finds it.
    #[test]
    fn generic_block_skips_the_qr_and_a_planted_kernel_is_found() {
        let (dirs, stats) = common_kernel(&projected_block(12, 30, None));
        assert!(dirs[0].is_none());
        assert_eq!((stats.certified, stats.searched), (1, 0));

        let s = 1.0 / 2f64.sqrt();
        let mut u = vec![0.0; 12];
        u[0] = s;
        u[1] = -s;
        let (dirs, stats) = common_kernel(&projected_block(12, 30, Some(&u)));
        assert_eq!((stats.certified, stats.searched), (0, 1));
        let (k, rest) = dirs[0].as_ref().expect("the planted kernel");
        assert_eq!((k.ncols(), rest.ncols()), (1, 11));
        let overlap: C64 = (0..12).map(|i| k[(i, 0)] * u[i]).sum();
        assert!(
            (overlap.norm() - 1.0).abs() < 1e-10,
            "kernel direction {:?}",
            overlap
        );
    }

    /// The test is one-sided: a pencil without a common kernel whose
    /// combinations are all singular is left to the QR, which finds nothing.
    #[test]
    fn singular_pencil_without_common_kernel_is_not_certified() {
        let c1 = C64::new(1.0, 0.0);
        // F_1 = E_12 + E_21, F_2 = E_13 + E_31: every combination has rank 2
        let mats = vec![
            Vec::new(),
            vec![(0, 1, c1), (1, 0, c1)],
            vec![(0, 2, c1), (2, 0, c1)],
        ];
        assert!(!certified_kernel_free(&mats, 3, true, 2f64.sqrt()));
        let mut m = SdpModel::new();
        let mut fj = BTreeMap::new();
        fj.insert(m.free_var("a"), vec![(0, 1, c1)]);
        fj.insert(m.free_var("b"), vec![(0, 2, c1)]);
        m.add_lmi(Lmi {
            size: 3,
            real: true,
            f0: Vec::new(),
            fj,
            label: "pencil".into(),
        });
        let (dirs, stats) = common_kernel(&m);
        assert!(dirs[0].is_none());
        assert_eq!(stats.searched, 1);
    }
}
