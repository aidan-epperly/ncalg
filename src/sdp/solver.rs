//! A native primal–dual interior-point SDP solver.
//!
//! Solves an [`SdpModel`] in LMI form
//!
//! ```text
//!   (D)  min  c·y   s.t.   Z = F_0 + Σ_j y_j F_j ⪰ 0        (block diagonal)
//!   (P)  max −⟨F_0, X⟩   s.t.   ⟨F_j, X⟩ = c_j,  X ⪰ 0
//! ```
//!
//! with the HKM (Helmberg–Rendl–Vanderbei–Wolkowicz / Kojima–Shindoh–Hara /
//! Monteiro) search direction and Mehrotra's predictor–corrector, starting
//! from the infeasible point `X = Z = λI`, `y = 0`.  Equalities in the model
//! are eliminated first ([`SdpModel::eliminate`]).  Real symmetric blocks are
//! solved in real arithmetic and complex Hermitian blocks natively in complex
//! arithmetic (`⟨F, X⟩ = Re tr(F X)`), so a Hermitian block of size `m` costs
//! `m × m` complex operations instead of the `2m × 2m` real embedding; the
//! dual matrices returned for complex blocks are the `m × m` Hermitian
//! matrices themselves (see [`SdpSolution::duals`]).
//!
//! Dense linear algebra (faer) throughout: each iteration costs
//! `O(Σ_b n_b³ + nnz² + m³)` for block sizes `n_b`, `m` variables and `nnz`
//! total nonzeros of the coefficient matrices (the Schur complement is
//! assembled entry by entry from the sparse `F_j`).  This is adequate for
//! blocks up to a few hundred and a few thousand variables; larger problems
//! should go to an external solver through the export routines.
//!
//! ```
//! use ncalg::sdp::{Affine, SdpModel};
//! use ncalg::solver::SolverOptions;
//! // min x + y  s.t.  [[x, 1], [1, y]] ⪰ 0   (optimum 2 at x = y = 1)
//! let mut m = SdpModel::new();
//! let x = m.free_var("x");
//! let y = m.free_var("y");
//! m.psd_affine(2, true, vec![(0, 0, Affine::var(x)), (0, 1, Affine::constant(1.0)), (1, 1, Affine::var(y))], "M");
//! m.minimize(&(Affine::var(x) + Affine::var(y)));
//! let sol = m.solve(&SolverOptions::default()).unwrap();
//! assert!((sol.objective - 2.0).abs() < 1e-6);
//! assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 1.0).abs() < 1e-5);
//! ```

use crate::sdp::{unembed_dual, Lmi, SdpModel, SparseHerm};
use faer::linalg::matmul::dot::inner_prod;
use faer::linalg::solvers::{DenseSolveCore, Llt};
use faer::linalg::triangular_solve::{solve_lower_triangular_in_place, solve_upper_triangular_in_place};
use faer::traits::ComplexField;
use faer::{unzip, zip, Accum, ColRef, Conj, Mat, MatMut, MatRef, Par, Scale, Side};
use num_complex::Complex;
use rayon::prelude::*;
use std::cell::Cell;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use std::time::Instant;

type C64 = Complex<f64>;

/// Solver settings.
#[derive(Clone, Debug)]
pub struct SolverOptions {
    /// Maximum number of interior-point iterations.
    pub max_iter: usize,
    /// Tolerance on the relative duality gap and the scaled infeasibilities.
    pub tol: f64,
    /// Fraction of the maximal feasible step taken (`0 < τ < 1`).
    pub step_fraction: f64,
    /// Print one line per iteration.
    pub verbose: bool,
    /// Refuse problems with more free variables than this: the solver forms a
    /// dense Schur complement with that many rows, factored in place (8 bytes
    /// per entry: 21 000 variables take 3.5 GB, and each iteration costs a
    /// 21 000³ / 3 Cholesky; measured: a Lovász theta SDP with
    /// 20 205 variables peaked at 3.2 GB and solved in 2.2 min on 8 threads).
    /// Raise it per solve when the memory is there; larger problems should
    /// otherwise go to a first-order solver through `MomentForm::write_json` /
    /// `SdpModel::write_json`.
    pub max_variables: usize,
    /// Looser tolerance at which an unconverged final iterate still counts as
    /// a usable solution for `MomentSdp::solve` (which otherwise reports `NaN`).
    pub accept_tol: f64,
    /// Numerical facial reduction (opt-in, default `false`): if the solve
    /// stops without converging and some Gram (dual) directions grew without
    /// bound, the problem has no strictly feasible moment point (every
    /// feasible block is singular).  The solver then restricts those blocks
    /// to the face the iterates converge to, adds the equalities `Z(y) u = 0`
    /// for the diverging directions `u`, and re-solves the smaller,
    /// well-posed problem (at most three rounds).  See
    /// [`SdpSolution::facial_reduction`].
    pub facial_reduction: bool,
    /// Regularization shift `ε ≥ 0` (default `0`, off): solve with every
    /// block's constraint relaxed to `Z(y) + ε s_b I ⪰ 0` (`s_b` the block's
    /// largest coefficient), which is strictly feasible whenever the
    /// original problem is feasible, and report the original problem's dual
    /// bound at the regularized Gram matrices (see
    /// [`SdpSolution::regularization`]).  For problems without a strictly
    /// feasible point, where the plain solve stalls: the result
    /// is a certified bound on the correct side, looser by about `c √ε`
    /// (estimated by a second solve at `100 ε`).  `1e-8` is a good value;
    /// much smaller shifts make the regularized problem ill-conditioned
    /// again.
    pub regularization: f64,
    /// Solve standard-form models (all LMIs bare `psd_var` matrices, every
    /// other variable free, e.g. the Gram forms) through their dual LMI in
    /// one variable per independent equality: `None` (default) when
    /// that is the smaller problem, `Some(true)` whenever the model has that
    /// form, `Some(false)` never.
    pub dualize: Option<bool>,
    /// Print [`SdpSolution::warnings`] to stderr when they arise (default
    /// `true`).  They are recorded in the solution either way.
    pub warnings: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            max_iter: 200,
            tol: 1e-8,
            step_fraction: 0.95,
            verbose: false,
            max_variables: 21_000,
            accept_tol: 1e-5,
            facial_reduction: false,
            regularization: 0.0,
            dualize: None,
            warnings: true,
        }
    }
}

/// Where a solve spent its time and how much work it did; filled by every
/// [`SdpModel::solve`] (see [`SdpSolution::profile`]), printed with
/// [`SolverOptions::verbose`], and compared against a baseline by ncalg's
/// canary (`scripts/canary.py`).
///
/// Times are wall-clock seconds.  The iteration phases and the counters add
/// up over every interior-point solve the call runs (facial-reduction
/// rounds, the two solves of a regularized solve, the dual route's solve).
/// Everything else is one-off work (presolve, elimination of equalities,
/// the dual route's set-up), [`one_off`](SolveProfile::one_off) seconds in
/// all.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolveProfile {
    /// The whole `solve` call.
    pub total: f64,
    /// Presolve: the common-kernel search of every block and the reduction.
    pub presolve: f64,
    /// Elimination of the model's equalities, and lifting the solution back.
    pub elimination: f64,
    /// Setting up the solver's blocks from the model (per solve).
    pub setup: f64,
    /// The dual route: building the dual LMI and recovering the primal.
    pub dualize: f64,
    /// Iterations: assembling the Schur complement (refills after a failed
    /// factorization included).
    pub schur_assembly: f64,
    /// Iterations: factoring the Schur complement.
    pub schur_factor: f64,
    /// Iterations: the Newton solves (conjugate gradients on the exact
    /// Schur operator, preconditioned by the factor).
    pub newton_solves: f64,
    /// Iterations: everything else (residuals, `Z⁻¹`, right-hand sides,
    /// directions, step lengths, updates).
    pub iterate: f64,
    /// Interior-point solves run.
    pub solves: usize,
    /// Interior-point iterations, over all solves.
    pub iterations: usize,
    /// Schur operator products in the Newton solves.
    pub operator_products: usize,
    /// Schur factorizations retried with a diagonal shift.
    pub factor_retries: usize,
    /// Presolve: blocks shown free of a common kernel by the
    /// random-combination test, without a factorization.
    pub kernel_certified: usize,
    /// Presolve: blocks searched for a common kernel (they had one, or the
    /// test could not rule one out).
    pub kernel_searched: usize,
    /// Presolve: stacked coefficient rows those searches factored.
    pub kernel_rows: usize,
}

impl SolveProfile {
    /// Seconds in the interior-point iterations.
    pub fn iterations_time(&self) -> f64 {
        self.schur_assembly + self.schur_factor + self.newton_solves + self.iterate
    }

    /// Seconds outside the interior-point iterations: `total` minus the
    /// iteration phases (presolve, elimination, set-up and lifting, and
    /// anything not itemised).
    pub fn one_off(&self) -> f64 {
        (self.total - self.iterations_time()).max(0.0)
    }

    /// Add the phases and counters of a solve run inside this one (not its
    /// `total`: the caller's clock runs around it).
    pub(crate) fn absorb(&mut self, other: &SolveProfile) {
        self.schur_assembly += other.schur_assembly;
        self.schur_factor += other.schur_factor;
        self.newton_solves += other.newton_solves;
        self.iterate += other.iterate;
        self.solves += other.solves;
        self.iterations += other.iterations;
        self.operator_products += other.operator_products;
        self.factor_retries += other.factor_retries;
        self.kernel_certified += other.kernel_certified;
        self.kernel_searched += other.kernel_searched;
        self.kernel_rows += other.kernel_rows;
        self.presolve += other.presolve;
        self.elimination += other.elimination;
        self.dualize += other.dualize;
        self.setup += other.setup;
    }

    /// The warning for a solve dominated by one-off work: more than a
    /// quarter of the time and more than a second outside the iterations.
    /// Such steps are meant to be cheap next to the solve they prepare, so
    /// this is usually a performance bug (one was a presolve QR that took
    /// 57 s of a 63 s solve).
    pub fn one_off_warning(&self) -> Option<String> {
        let one_off = self.one_off();
        if one_off <= 1.0 || one_off <= 0.25 * self.total {
            return None;
        }
        let other = (one_off - self.presolve - self.elimination - self.dualize - self.setup).max(0.0);
        Some(format!(
            "ncalg: one-off steps took {:.1} s of this {:.1} s solve (presolve {:.1} s, elimination {:.1} s, \
             dual route {:.1} s, block set-up {:.1} s, other {:.1} s) against {:.1} s of interior-point \
             iterations; unless that is expected for this problem it is a performance bug, see \
             CONTRIBUTING.md (Performance)",
            one_off,
            self.total,
            self.presolve,
            self.elimination,
            self.dualize,
            self.setup,
            other,
            self.iterations_time()
        ))
    }

    /// One line for logs.
    pub fn summary(&self) -> String {
        format!(
            "profile: {:.3} s = one-off {:.3} s (presolve {:.3}, elimination {:.3}, dual route {:.3}, set-up {:.3}) + \
             {} iterations in {} solve(s) {:.3} s (Schur assembly {:.3}, factor {:.3}, Newton solves {:.3} \
             [{} operator products], other {:.3}); kernel: {} certified, {} searched ({} rows); {} factor retries",
            self.total,
            self.one_off(),
            self.presolve,
            self.elimination,
            self.dualize,
            self.setup,
            self.iterations,
            self.solves,
            self.iterations_time(),
            self.schur_assembly,
            self.schur_factor,
            self.newton_solves,
            self.operator_products,
            self.iterate,
            self.kernel_certified,
            self.kernel_searched,
            self.kernel_rows,
            self.factor_retries
        )
    }
}

/// Termination status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// All tolerances met.
    Optimal,
    /// The iteration limit was reached (the returned point is the last iterate).
    MaxIterations,
    /// A factorization failed; the problem may be infeasible, unbounded or
    /// badly scaled.
    NumericalError,
}

/// Solver error (structural problems found before iterating, or an
/// inconsistent equality system).
#[derive(Clone, Debug)]
pub struct SolverError(pub String);

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SolverError {}

/// Result of [`SdpModel::solve`].
#[derive(Clone, Debug)]
pub struct SdpSolution {
    /// The model variables.
    pub x: Vec<f64>,
    /// The objective value in the model's sign convention.
    pub objective: f64,
    /// Dual matrices `G_b` (Gram matrices of the SOS side), one per LMI
    /// block, of the block's Hermitian size: real symmetric for real blocks,
    /// Hermitian for complex blocks, normalized so that
    /// `Σ_ab G_b[a,b] F_j^b[a,b] = c_j` (the convention of
    /// `MomentSdp::dual_grams`, equal to `tr(X_b F_j^b)` for the primal
    /// matrix `X_b = conj(G_b)`).
    pub duals: Vec<Vec<Vec<C64>>>,
    /// How the iteration ended.
    pub status: Status,
    /// Number of interior-point iterations.
    pub iterations: usize,
    /// `‖c − A(X)‖ / (1 + ‖c‖)`.
    pub primal_infeasibility: f64,
    /// `max_b ‖F_0 + Σ y_j F_j − Z‖_F / (1 + ‖F_0‖_F)`.
    pub dual_infeasibility: f64,
    /// `|c·y + ⟨F_0, X⟩| / (1 + |c·y| + |⟨F_0, X⟩|)`.
    pub relative_gap: f64,
    /// Set when the solution comes from a facially reduced problem
    /// ([`SolverOptions::facial_reduction`]); the convergence measures above
    /// are then those of the reduced problem.
    pub facial_reduction: Option<FacialReduction>,
    /// Set when the solution comes from a regularized problem
    /// ([`SolverOptions::regularization`]); `objective` is then the
    /// certified bound, `x` the regularized moment point.
    pub regularization: Option<Regularization>,
    /// Where the solve spent its time, and how much work it did.
    pub profile: SolveProfile,
    /// Warnings about the solve (also printed to stderr unless
    /// [`SolverOptions::warnings`] is off); see
    /// [`SolveProfile::one_off_warning`].
    pub warnings: Vec<String>,
}

/// What [`SolverOptions::regularization`] did.
#[derive(Clone, Debug, PartialEq)]
pub struct Regularization {
    /// The shift `ε` (relative to each block's largest coefficient).
    pub epsilon: f64,
    /// Optimum of the regularized problem, in the model's objective units:
    /// a weaker bound on the same side as `objective`.
    pub regularized_objective: f64,
    /// `Σ_b ε s_b tr X_b`: the amount by which the certified bound improves
    /// on the regularized optimum.
    pub trace_term: f64,
    /// Estimated distance of the certified bound from the true optimum,
    /// `|bound(100 ε) − bound(ε)| / 9` from the leading `c √ε` behavior
    /// (a second solve at `100 ε`); `None` if that solve failed.
    pub bias_estimate: Option<f64>,
    /// The bound corrected by the bias estimate: the best estimate of the
    /// optimum, but **not** a certified bound.
    pub extrapolated_objective: Option<f64>,
}

/// What [`SolverOptions::facial_reduction`] did.
#[derive(Clone, Debug, PartialEq)]
pub struct FacialReduction {
    /// Reduction rounds (each followed by a re-solve).
    pub rounds: usize,
    /// Directions removed from each LMI block of the model, over all rounds.
    pub removed: Vec<usize>,
    /// Objective of the unreduced solve's best iterate, for comparison.
    pub unreduced_objective: f64,
    /// Smallest eigenvalue of the original LMI blocks at the returned point
    /// (zero up to rounding when the removed directions are the exact
    /// structural kernel; clearly negative means the reduction relaxed the
    /// problem).
    pub min_eigenvalue: f64,
    /// `max |u* Z(y)|` over the removed directions `u` at the returned
    /// point, relative to the largest entry of the block: zero if the
    /// reduction is exact.  The value of a problem without a strictly
    /// feasible point reacts to such a residual linearly (not through an
    /// eigenvalue, which only sees its square), so it is the error indicator
    /// to look at.
    pub kernel_residual: f64,
    /// `max_j |Σ_b ⟨G_b, F_j⟩ − c_j| / (1 + |c_j|)` for the returned duals
    /// on the original model.  The duals are the reduced problem's, lifted:
    /// they omit the multipliers of the removed directions (and a problem
    /// without a strictly feasible point need not have an attained dual at
    /// all), so this is generally not small, and a certificate built from
    /// them does not prove the bound.  `x` and `objective` are
    /// unaffected.
    pub dual_residual: f64,
}

impl SdpSolution {
    /// Dual matrices as Hermitian matrices of the blocks' Hermitian sizes.
    /// The native solver already returns them in this form; a dual of the
    /// real-embedded size `2n` (e.g. one read from an SDPA solution) is
    /// un-embedded.
    pub fn hermitian_duals(&self, model: &SdpModel) -> Vec<Vec<Vec<C64>>> {
        model
            .lmis()
            .iter()
            .zip(&self.duals)
            .map(|(lmi, y)| {
                if lmi.real || y.len() == lmi.size {
                    crate::sdp::complete_triangle(y)
                } else {
                    unembed_dual(y, lmi.size)
                }
            })
            .collect()
    }

    /// Did the solver converge?
    pub fn is_optimal(&self) -> bool {
        self.status == Status::Optimal
    }

    /// Is the final iterate usable at the given looser tolerance: either
    /// `Optimal`, or a `NumericalError` / `MaxIterations` stop whose relative
    /// gap and scaled infeasibilities are all below `tol`?  The objective of
    /// an iterate that fails this is not a bound on anything.
    pub fn is_converged(&self, tol: f64) -> bool {
        self.status == Status::Optimal
            || (self.relative_gap.is_finite()
                && self.relative_gap < tol
                && self.primal_infeasibility < tol
                && self.dual_infeasibility < tol)
    }
}

// ---------------------------------------------------------------------------
// Per-block scalar type
// ---------------------------------------------------------------------------

/// Scalar type of an LMI block: `f64` for real symmetric blocks, `C64` for
/// complex Hermitian blocks.  The block routines below are generic over it;
/// faer supplies the dense factorizations for both instantiations.
trait BlockScalar:
    ComplexField<Real = f64>
    + Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
{
    fn conjugate(self) -> Self;
    fn real(self) -> f64;
    fn abs_sq(self) -> f64;
    fn from_real(x: f64) -> Self;
    fn times(self, s: f64) -> Self;
    fn from_c64(c: C64) -> Self;
    fn to_c64(self) -> C64;
    /// This scalar type's buffers in a Schur workspace.
    fn scratch(ws: &mut SchurWorkspace) -> &mut Scratch<Self>;
}

impl BlockScalar for f64 {
    fn scratch(ws: &mut SchurWorkspace) -> &mut Scratch<Self> {
        &mut ws.real
    }

    #[inline(always)]
    fn conjugate(self) -> Self {
        self
    }
    #[inline(always)]
    fn real(self) -> f64 {
        self
    }
    #[inline(always)]
    fn abs_sq(self) -> f64 {
        self * self
    }
    #[inline(always)]
    fn from_real(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn times(self, s: f64) -> Self {
        self * s
    }
    #[inline(always)]
    fn from_c64(c: C64) -> Self {
        c.re
    }
    #[inline(always)]
    fn to_c64(self) -> C64 {
        C64::new(self, 0.0)
    }
}

impl BlockScalar for C64 {
    fn scratch(ws: &mut SchurWorkspace) -> &mut Scratch<Self> {
        &mut ws.complex
    }

    #[inline(always)]
    fn conjugate(self) -> Self {
        self.conj()
    }
    #[inline(always)]
    fn real(self) -> f64 {
        self.re
    }
    #[inline(always)]
    fn abs_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    #[inline(always)]
    fn from_real(x: f64) -> Self {
        C64::new(x, 0.0)
    }
    #[inline(always)]
    fn times(self, s: f64) -> Self {
        C64::new(self.re * s, self.im * s)
    }
    #[inline(always)]
    fn from_c64(c: C64) -> Self {
        c
    }
    #[inline(always)]
    fn to_c64(self) -> C64 {
        self
    }
}

/// Full Hermitian entry list `(a, b, v)` (both triangles) of a block
/// coefficient matrix.
type Entries<T> = Vec<(usize, usize, T)>;

/// The full Hermitian entry list of a stored upper triangle
/// ([`crate::linalg::sparse::full_entries`]); real blocks keep the real parts only.
fn full_herm<T: BlockScalar>(upper: &SparseHerm) -> Entries<T> {
    crate::linalg::sparse::full_entries(upper)
        .into_iter()
        .map(|(a, b, c)| (a, b, T::from_c64(c)))
        .filter(|(_, _, v)| v.abs_sq() != 0.0)
        .collect()
}

/// `⟨F, M⟩ = Re tr(F M) = Re Σ_{(a,b,v) ∈ F} conj(v) M[a,b]` for Hermitian `F`, `M`.
fn inner<T: BlockScalar>(entries: &Entries<T>, m: &Mat<T>) -> f64 {
    entries
        .iter()
        .map(|(a, b, v)| (v.conjugate() * m[(*a, *b)]).real())
        .sum()
}

fn add_scaled<T: BlockScalar>(target: &mut Mat<T>, entries: &Entries<T>, s: f64) {
    for (a, b, v) in entries {
        target[(*a, *b)] += v.times(s);
    }
}

/// `Re tr(A B) = Re Σ_ab A[a,b] conj(B[a,b])` for Hermitian `A`, `B`, column
/// by column with faer's (SIMD) inner product.
fn dot<T: BlockScalar>(a: &Mat<T>, b: &Mat<T>) -> f64 {
    (0..a.ncols())
        .map(|j| inner_prod(b.col(j).transpose(), Conj::Yes, a.col(j), Conj::No).real())
        .sum()
}

/// Replace `M` by its Hermitian part `(M + Mᴴ)/2`.
fn hermitize<T: BlockScalar>(m: &mut Mat<T>) {
    let n = m.nrows();
    for i in 0..n {
        m[(i, i)] = T::from_real(m[(i, i)].real());
        for j in i + 1..n {
            let v = (m[(i, j)] + m[(j, i)].conjugate()).times(0.5);
            m[(i, j)] = v;
            m[(j, i)] = v.conjugate();
        }
    }
}

/// `m += a d`, re-Hermitised, in place.
fn advance<T: BlockScalar>(m: &mut Mat<T>, d: &Mat<T>, a: f64) {
    zip!(m.as_mut(), d.as_ref()).for_each(|unzip!(m, d)| *m += d.times(a));
    hermitize(m);
}

fn identity<T: BlockScalar>(n: usize, lambda: f64) -> Mat<T> {
    let mut m = Mat::<T>::identity(n, n);
    m *= Scale(T::from_real(lambda));
    m
}

/// Blocks up to this size get their step length from a full eigenvalue
/// computation; larger ones from Lanczos.  Dense: two triangular solves with
/// `n` right-hand sides and a Hermitian eigenvalue computation, about
/// `4 n³` flops.  Lanczos: per step two triangular solves and a product with
/// one vector plus reorthogonalisation, `3 n² + 2 k n`; with the `k ≈ 20–40`
/// steps it typically needs, it is cheaper from `n ≈ 50` on.
const LANCZOS_MIN_SIZE: usize = 64;
/// Lanczos steps before falling back to the dense computation.
const LANCZOS_MAX_STEPS: usize = 60;
/// Relative accuracy of the Lanczos estimate of the smallest eigenvalue `λ`:
/// residual `≤ 1e-2 max(|λ|, 1)`, i.e. the step length `−1/λ` to 1% (and
/// steps beyond 1, which are capped, only to that absolute accuracy).
const LANCZOS_RTOL: f64 = 1e-2;

/// Largest `α` with `X + α ΔX ⪰ 0` (`+∞` if unbounded) for `X = L Lᴴ`: the
/// smallest eigenvalue `λ` of `W = L⁻¹ ΔX L⁻ᴴ` gives `α = −1/λ`.  With
/// `estimate`, large blocks use a Lanczos estimate `θ − r` (smallest Ritz
/// value minus its residual) instead of the exact `λ`.  The estimate is
/// not guaranteed to bound `λ` from below; the caller verifies the step it
/// takes by factoring the new iterate and falls back to `estimate = false`
/// if that fails.
fn step_length<T: BlockScalar>(l: MatRef<'_, T>, d: &Mat<T>, estimate: bool) -> Option<f64> {
    let n = l.nrows();
    if n == 0 {
        return Some(f64::INFINITY);
    }
    let lam = if estimate && n > LANCZOS_MIN_SIZE {
        // two independent start vectors: θ − r bounds some eigenvalue, and a
        // single start vector nearly orthogonal to the extreme eigenvector
        // would miss λ_min; both missing it is far less likely
        match (
            lanczos_min_eig(l, d, 0x9E37_79B9_7F4A_7C15),
            lanczos_min_eig(l, d, 0xD1B5_4A32_D192_ED03),
        ) {
            (Some((t1, r1)), Some((t2, r2))) => (t1 - r1).min(t2 - r2),
            _ => dense_min_eig(l, d)?,
        }
    } else {
        dense_min_eig(l, d)?
    };
    Some(if lam >= -1e-14 { f64::INFINITY } else { -1.0 / lam })
}

/// Smallest eigenvalue of `L⁻¹ D L⁻ᴴ`, formed explicitly.
fn dense_min_eig<T: BlockScalar>(l: MatRef<'_, T>, d: &Mat<T>) -> Option<f64> {
    let mut w = d.clone();
    solve_lower_triangular_in_place(l, w.as_mut(), Par::Seq);
    let mut wt = w.as_ref().adjoint().to_owned();
    solve_lower_triangular_in_place(l, wt.as_mut(), Par::Seq);
    hermitize(&mut wt);
    let ev = wt.self_adjoint_eigenvalues(Side::Lower).ok()?;
    Some(ev.into_iter().fold(f64::INFINITY, f64::min))
}

/// Lanczos (full reorthogonalisation, fixed start vector) for the smallest
/// eigenvalue of `W = L⁻¹ D L⁻ᴴ`.  Not faer's
/// `operator::partial_self_adjoint_eigen`: that targets the eigenvalues of
/// largest magnitude to full accuracy, while the step length needs the
/// smallest (algebraic) eigenvalue of an indefinite `W` only to 1% and stops
/// on the Ritz residual; applied as two triangular solves and one
/// product: `(θ, r)`, the smallest Ritz value and its residual norm, once
/// `r ≤ LANCZOS_RTOL max(|θ|, 1)`; `None` if that takes more than
/// `LANCZOS_MAX_STEPS` steps.
fn lanczos_min_eig<T: BlockScalar>(l: MatRef<'_, T>, d: &Mat<T>, seed: u64) -> Option<(f64, f64)> {
    let n = l.nrows();
    let apply = |v: &Mat<T>| -> Mat<T> {
        let mut w = v.clone();
        solve_upper_triangular_in_place(l.adjoint(), w.as_mut(), Par::Seq);
        let mut u = crate::dense::matmul(d.as_ref(), w.as_ref());
        solve_lower_triangular_in_place(l, u.as_mut(), Par::Seq);
        u
    };
    let dotc =
        |a: &Mat<T>, b: &Mat<T>| -> T { inner_prod(a.col(0).transpose(), Conj::Yes, b.col(0), Conj::No) };
    let norm = |a: &Mat<T>| -> f64 { a.norm_l2() };
    // pseudo-random start vector from `seed` (the solver stays deterministic);
    // entries of both signs, so different seeds give unrelated vectors
    let mut state: u64 = seed;
    let mut q = Mat::<T>::from_fn(n, 1, |_, _| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        T::from_real(state as f64 / u64::MAX as f64 - 0.5)
    });
    let q_norm = norm(&q);
    q = &q * (1.0 / q_norm);
    let mut basis = vec![q];
    let mut alpha: Vec<f64> = Vec::new();
    let mut beta: Vec<f64> = Vec::new();
    for k in 0..n.min(LANCZOS_MAX_STEPS) {
        let mut w = apply(&basis[k]);
        alpha.push(dotc(&basis[k], &w).real());
        // orthogonalise against the whole basis, twice ("twice is enough");
        // this also removes the α_k q_k and β_{k−1} q_{k−1} components
        for _ in 0..2 {
            for qj in &basis {
                let c = dotc(qj, &w);
                for i in 0..n {
                    let v = qj[(i, 0)];
                    w[(i, 0)] -= c * v;
                }
            }
        }
        let b = norm(&w);
        let kk = k + 1;
        let t = Mat::<f64>::from_fn(kk, kk, |i, j| {
            if i == j {
                alpha[i]
            } else if i + 1 == j {
                beta[i]
            } else if j + 1 == i {
                beta[j]
            } else {
                0.0
            }
        });
        let e = t.self_adjoint_eigen(Side::Lower).ok()?;
        let theta = e.S().column_vector()[0];
        let res = b * e.U()[(kk - 1, 0)].abs();
        let scale = alpha.iter().map(|a| a.abs()).fold(0.0_f64, f64::max);
        if res <= LANCZOS_RTOL * theta.abs().max(1.0) || b <= 1e-14 * scale.max(1e-300) {
            return Some((theta, res));
        }
        beta.push(b);
        basis.push(&w * (1.0 / b));
    }
    None
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

/// A candidate step `(α, β, factor of X + α ΔX, factor of Z + β ΔZ)`,
/// factored to verify it before it is taken (see [`BlockOps::try_step`]);
/// the iterates themselves are formed again when the step is taken.
type Candidate<T> = (f64, f64, Option<Llt<T>>, Option<Llt<T>>);

/// Per-block scratch matrices (`n × n`) for the products of one iteration
/// and of the Schur operator, reused instead of allocated per call.  Behind
/// a mutex because the operator runs on `&self`; never contended (one
/// thread per block).
struct BlockScratch<T: BlockScalar> {
    s: Mat<T>,
    t: Mat<T>,
}

/// One LMI block with its coefficient data and iterates, in the block's own
/// scalar type.
struct Block<T: BlockScalar> {
    n: usize,
    f0: Entries<T>,
    /// `(global variable, entries)` for the variables that occur in this
    /// block, sorted by decreasing number of entries.
    vars: Vec<(usize, Entries<T>)>,
    /// Distinct column indices of each variable's entries (same order as `vars`).
    cols: Vec<Vec<usize>>,
    /// `tail[p] = Σ_{q ≥ p} nnz(F_q)`: the entries the Schur column `p` pairs with.
    tail: Vec<usize>,
    /// Global variable indices, same order as `vars`.
    globals: Vec<usize>,
    /// Primal and dual iterates.
    x: Mat<T>,
    z: Mat<T>,
    /// Cholesky factors of the current `X` and `Z`, when computed.
    llt_x: Option<Llt<T>>,
    llt_z: Option<Llt<T>>,
    /// The step verified last by `try_step`.
    candidate: Option<Candidate<T>>,
    /// `Z⁻¹` of the current iterate.
    zinv: Mat<T>,
    /// Scratch for the products.
    scratch: std::sync::Mutex<BlockScratch<T>>,
    /// Dual residual `R_d = F_0 + Σ y_j F_j − Z`.
    rd: Mat<T>,
    /// `X R_d Z⁻¹` (shared by the predictor and the corrector).
    xrdz: Mat<T>,
    /// Complementarity right-hand side `Q` of the current Newton system.
    q: Mat<T>,
    /// Current search direction (the affine one between predictor and corrector).
    dx: Mat<T>,
    dz: Mat<T>,
}

impl<T: BlockScalar> Block<T> {
    fn new(lmi: &Lmi) -> Self {
        let n = lmi.size;
        let f0 = full_herm::<T>(&lmi.f0);
        let mut vars = Vec::new();
        for (j, f) in &lmi.fj {
            let e = full_herm::<T>(f);
            if !e.is_empty() {
                vars.push((*j, e));
            }
        }
        // densest variables first: the dense formula is chosen for them and
        // the pair loops below then run over the sparser remainder
        vars.sort_by_key(|v| std::cmp::Reverse(v.1.len()));
        let cols: Vec<Vec<usize>> = vars
            .iter()
            .map(|(_, e)| {
                let mut c: Vec<usize> = e.iter().map(|(_, d, _)| *d).collect();
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        let mut tail = vec![0usize; vars.len() + 1];
        for p in (0..vars.len()).rev() {
            tail[p] = tail[p + 1] + vars[p].1.len();
        }
        let zero = || Mat::<T>::zeros(n, n);
        let globals = vars.iter().map(|(j, _)| *j).collect();
        Block {
            n,
            f0,
            vars,
            cols,
            tail,
            globals,
            x: zero(),
            z: zero(),
            llt_x: None,
            llt_z: None,
            candidate: None,
            zinv: zero(),
            scratch: std::sync::Mutex::new(BlockScratch { s: zero(), t: zero() }),
            rd: zero(),
            xrdz: zero(),
            q: zero(),
            dx: zero(),
            dz: zero(),
        }
    }

    /// `out = Σ_j dy_j F_j`.
    fn combination_into(vars: &[(usize, Entries<T>)], dy: &[f64], out: &mut Mat<T>) {
        out.as_mut().fill(T::from_real(0.0));
        for (j, e) in vars {
            add_scaled(out, e, dy[*j]);
        }
    }

    /// `(global variable, ⟨F_j, M⟩)` for every variable of the block.
    fn inners(&self, m: &Mat<T>) -> Vec<(usize, f64)> {
        self.vars.iter().map(|(j, e)| (*j, inner(e, m))).collect()
    }
}

/// The per-block operations of one interior-point iteration, independent of
/// the block's scalar type.  The solver runs them for all blocks in
/// parallel; contributions to shared vectors are returned as
/// `(variable, value)` lists and added in block order, so the result does
/// not depend on the thread schedule.
trait BlockOps: Send + Sync {
    fn size(&self) -> usize;
    /// Global indices of the variables occurring in the block.
    fn variables(&self) -> Vec<usize>;
    /// Largest coefficient magnitude.
    fn max_entry(&self) -> f64;
    /// `‖F_0‖_F²`.
    fn f0_norm_sq(&self) -> f64;
    /// `X = Z = λI`.
    fn init(&mut self, lambda: f64);
    /// `⟨X, Z⟩`.
    fn gap(&self) -> f64;
    /// `⟨F_j, X⟩` per variable.
    fn primal_inners(&self) -> Vec<(usize, f64)>;
    /// `⟨F_0, X⟩`.
    fn f0_inner(&self) -> f64;
    /// Form `R_d = F_0 + Σ y_j F_j − Z`; returns `‖R_d‖_F`.
    fn dual_residual(&mut self, y: &[f64]) -> f64;
    /// Form `Z⁻¹`; `false` if `Z` is not positive definite.
    fn invert_z(&mut self) -> bool;
    /// Number of Schur complement columns the block contributes to.
    fn schur_columns(&self) -> usize;
    /// Global indices of the block's variables, in the order of its Schur
    /// columns.
    fn globals(&self) -> &[usize];
    /// Column `p` of the block's contribution to the Schur complement:
    /// `Re tr(F_i X F_j Z⁻¹)` for its variable `j = globals()[p]` and the
    /// variables `i` in `globals()[p..]`, written to `out` in that order.
    fn schur_column(&self, p: usize, ws: &mut SchurWorkspace, out: &mut [f64]);
    /// Predictor right-hand side `Q = −X − X R_d Z⁻¹`.
    fn predictor_rhs(&mut self);
    /// `⟨F_j, Q⟩` per variable.
    fn rhs_inners(&self) -> Vec<(usize, f64)>;
    /// `ΔZ = R_d + Σ dy_j F_j`, `ΔX = Q − X (Σ dy_j F_j) Z⁻¹` (Hermitian part).
    fn direction(&mut self, dy: &[f64]);
    /// Largest steps keeping `X + α ΔX` and `Z + β ΔZ` positive definite
    /// (Lanczos estimates for large blocks if `estimate`, see [`step_length`]).
    fn max_steps(&mut self, estimate: bool) -> Option<(f64, f64)>;
    /// `⟨X + α ΔX, Z + β ΔZ⟩`.
    fn gap_after(&self, alpha: f64, beta: f64) -> f64;
    /// Corrector right-hand side `Q = σμ Z⁻¹ − X − ΔX ΔZ Z⁻¹ − X R_d Z⁻¹`
    /// with the affine direction currently stored.
    fn corrector_rhs(&mut self, sigma_mu: f64);
    /// Form and factor `X + α ΔX`, `Z + β ΔZ` (re-Hermitised); `true` if both
    /// are positive definite.  The factors are kept for the next iteration.
    fn try_step(&mut self, alpha: f64, beta: f64) -> bool;
    /// `X += α ΔX`, `Z += β ΔZ`, re-Hermitised (the verified candidate if it
    /// was formed for these steps).
    fn step(&mut self, alpha: f64, beta: f64);
    /// The block's part of the Schur complement product:
    /// `⟨F_i, X S Z⁻¹⟩` per variable `i`, for `S = Σ_j v_j F_j`.
    fn schur_apply(&self, v: &[f64]) -> Vec<(usize, f64)>;
    /// The dual (Gram) matrix `G = Xᵀ = conj(X)` as dense rows.
    fn dual(&self) -> Vec<Vec<C64>>;
}

impl<T: BlockScalar> BlockOps for Block<T> {
    fn size(&self) -> usize {
        self.n
    }

    fn variables(&self) -> Vec<usize> {
        self.vars.iter().map(|(j, _)| *j).collect()
    }

    fn max_entry(&self) -> f64 {
        self.f0
            .iter()
            .chain(self.vars.iter().flat_map(|(_, e)| e.iter()))
            .map(|(_, _, v)| v.abs_sq())
            .fold(0.0_f64, f64::max)
            .sqrt()
    }

    fn f0_norm_sq(&self) -> f64 {
        self.f0.iter().map(|(_, _, v)| v.abs_sq()).sum()
    }

    fn init(&mut self, lambda: f64) {
        self.x = identity(self.n, lambda);
        self.z = identity(self.n, lambda);
        self.llt_x = None;
        self.llt_z = None;
        self.candidate = None;
    }

    fn gap(&self) -> f64 {
        dot(&self.x, &self.z)
    }

    fn primal_inners(&self) -> Vec<(usize, f64)> {
        self.inners(&self.x)
    }

    fn f0_inner(&self) -> f64 {
        inner(&self.f0, &self.x)
    }

    fn dual_residual(&mut self, y: &[f64]) -> f64 {
        let mut r = -&self.z;
        add_scaled(&mut r, &self.f0, 1.0);
        for (j, e) in &self.vars {
            add_scaled(&mut r, e, y[*j]);
        }
        let norm = r.norm_l2();
        self.rd = r;
        norm
    }

    fn invert_z(&mut self) -> bool {
        let llt = match self.llt_z.take() {
            Some(llt) => llt,
            None => match self.z.llt(Side::Lower) {
                Ok(llt) => llt,
                Err(_) => return false,
            },
        };
        self.zinv = llt.inverse();
        self.llt_z = Some(llt);
        true
    }

    fn schur_columns(&self) -> usize {
        self.vars.len()
    }

    fn globals(&self) -> &[usize] {
        &self.globals
    }

    fn schur_column(&self, p: usize, ws: &mut SchurWorkspace, out: &mut [f64]) {
        schur_column(self, p, ws, out)
    }

    fn predictor_rhs(&mut self) {
        let sc = self.scratch.get_mut().expect("scratch");
        crate::dense::matmul_into(sc.t.as_mut(), self.x.as_ref(), self.rd.as_ref());
        crate::dense::matmul_into(self.xrdz.as_mut(), sc.t.as_ref(), self.zinv.as_ref());
        zip!(self.q.as_mut(), self.x.as_ref(), self.xrdz.as_ref()).for_each(|unzip!(q, x, r)| *q = -*x - *r);
    }

    fn rhs_inners(&self) -> Vec<(usize, f64)> {
        self.inners(&self.q)
    }

    fn direction(&mut self, dy: &[f64]) {
        let sc = self.scratch.get_mut().expect("scratch");
        // S = Σ dy_j F_j; ΔZ = R_d + S; ΔX = Q − X S Z⁻¹ (Hermitian part)
        Self::combination_into(&self.vars, dy, &mut sc.s);
        zip!(self.dz.as_mut(), self.rd.as_ref(), sc.s.as_ref()).for_each(|unzip!(dz, r, s)| *dz = *r + *s);
        crate::dense::matmul_into(sc.t.as_mut(), self.x.as_ref(), sc.s.as_ref());
        crate::dense::matmul_into(self.dx.as_mut(), sc.t.as_ref(), self.zinv.as_ref());
        zip!(self.dx.as_mut(), self.q.as_ref()).for_each(|unzip!(dx, q)| *dx = *q - *dx);
        hermitize(&mut self.dx);
    }

    fn max_steps(&mut self, estimate: bool) -> Option<(f64, f64)> {
        if self.llt_x.is_none() {
            self.llt_x = Some(self.x.llt(Side::Lower).ok()?);
        }
        if self.llt_z.is_none() {
            self.llt_z = Some(self.z.llt(Side::Lower).ok()?);
        }
        let lx = self.llt_x.as_ref()?.L();
        let lz = self.llt_z.as_ref()?.L();
        Some((
            step_length(lx, &self.dx, estimate)?,
            step_length(lz, &self.dz, estimate)?,
        ))
    }

    fn gap_after(&self, alpha: f64, beta: f64) -> f64 {
        // Re tr((X + α ΔX)(Z + β ΔZ)) in one pass, without forming either
        let mut s = 0.0;
        for j in 0..self.n {
            let (x, dx) = (self.x.col_as_slice(j), self.dx.col_as_slice(j));
            let (z, dz) = (self.z.col_as_slice(j), self.dz.col_as_slice(j));
            for i in 0..self.n {
                s += ((x[i] + dx[i].times(alpha)) * (z[i] + dz[i].times(beta)).conjugate()).real();
            }
        }
        s
    }

    fn corrector_rhs(&mut self, sigma_mu: f64) {
        let sc = self.scratch.get_mut().expect("scratch");
        crate::dense::matmul_into(sc.t.as_mut(), self.dx.as_ref(), self.dz.as_ref());
        crate::dense::matmul_into(sc.s.as_mut(), sc.t.as_ref(), self.zinv.as_ref());
        zip!(
            self.q.as_mut(),
            self.zinv.as_ref(),
            self.x.as_ref(),
            self.xrdz.as_ref(),
            sc.s.as_ref()
        )
        .for_each(|unzip!(q, zi, x, r, d)| *q = zi.times(sigma_mu) - *x - *r - *d);
    }

    fn try_step(&mut self, alpha: f64, beta: f64) -> bool {
        // the new iterates are formed in the scratch matrices, factored, and
        // dropped: `step` forms them again (n² flops) rather than keeping two
        // more n × n matrices per block
        let sc = self.scratch.get_mut().expect("scratch");
        let next = |out: &mut Mat<T>, m: &Mat<T>, d: &Mat<T>, a: f64| -> Option<Llt<T>> {
            out.copy_from(m);
            advance(out, d, a);
            out.llt(Side::Lower).ok()
        };
        let lx = next(&mut sc.s, &self.x, &self.dx, alpha);
        let lz = next(&mut sc.t, &self.z, &self.dz, beta);
        let ok = lx.is_some() && lz.is_some();
        self.candidate = Some((alpha, beta, lx, lz));
        ok
    }

    fn step(&mut self, alpha: f64, beta: f64) {
        match self.candidate.take() {
            Some((a, b, lx, lz)) if a == alpha && b == beta => {
                // the same arithmetic as in try_step: the factors match exactly
                advance(&mut self.x, &self.dx, alpha);
                advance(&mut self.z, &self.dz, beta);
                self.llt_x = lx;
                self.llt_z = lz;
            }
            _ => {
                advance(&mut self.x, &self.dx, alpha);
                advance(&mut self.z, &self.dz, beta);
                self.llt_x = None;
                self.llt_z = None;
            }
        }
    }

    fn schur_apply(&self, v: &[f64]) -> Vec<(usize, f64)> {
        let mut guard = self.scratch.lock().expect("scratch");
        let sc = &mut *guard;
        Self::combination_into(&self.vars, v, &mut sc.s);
        // T = X S Z⁻¹, with S overwritten by the second product
        crate::dense::matmul_into(sc.t.as_mut(), self.x.as_ref(), sc.s.as_ref());
        crate::dense::matmul_into(sc.s.as_mut(), sc.t.as_ref(), self.zinv.as_ref());
        self.inners(&sc.s)
    }

    fn dual(&self) -> Vec<Vec<C64>> {
        // G[a,b] = X[b,a]: Σ_ab G[a,b] F[a,b] = tr(X F) = ⟨F, X⟩
        (0..self.n)
            .map(|i| (0..self.n).map(|j| self.x[(j, i)].to_c64()).collect())
            .collect()
    }
}

/// Scratch buffers of the Schur assembly, one set per rayon worker
/// (`for_each_init`), reused for every column the worker computes instead
/// of allocating `X F_j` and `T` afresh per variable.
#[derive(Default)]
struct SchurWorkspace {
    real: Scratch<f64>,
    complex: Scratch<C64>,
}

#[derive(Default)]
struct Scratch<T> {
    /// `U = X F_j` on the columns `D_j`, row-major `n × k`.
    u: Vec<T>,
    /// Rows `D_j` of `Z⁻¹`, column-major `k × n` (column `b` contiguous).
    z: Vec<T>,
    /// `T = U Z⁻¹[D_j, :]`, column-major `n × n` (dense formula only).
    t: Vec<T>,
}

/// Resize `v` to `len` (contents unspecified; callers overwrite or zero).
fn sized<T: BlockScalar>(v: &mut Vec<T>, len: usize) -> &mut [T] {
    if v.len() < len {
        v.resize(len, T::from_real(0.0));
    }
    &mut v[..len]
}

/// Column `p` of the block's contribution `M_ij += Re tr(F_i X F_j Z⁻¹)` to
/// the Schur complement, for `j = vars[p]` and `i` in `vars[p..]`, written
/// to `out` in that order.  The cheapest of three formulas is used (as in
/// Fujisawa–Kojima–Nakata, whose F1–F3 these are), all built on
/// `U = X F_j` restricted to the nonzero columns `D_j` of `F_j` (`k = |D_j|`):
///
/// - dense: `T = U Z⁻¹[D_j, :]` as one matrix product (`n² k` flops), then
///   `tr(F_i T)` from the entries of each `F_i`;
/// - entrywise: only the entries of `T` that some `F_i` needs, each a dot
///   product of a row of `U` with a column of `Z⁻¹[D_j, :]` (`k` flops per
///   entry of the `F_i`, both operands contiguous, nothing written);
/// - sparse: `F_i[a,b] X[b,c] F_j[c,d] Z⁻¹[d,a]` summed over the entries of
///   both (`nnz(F_j)` flops per entry of the `F_i`, scattered reads).
fn schur_column<T: BlockScalar>(blk: &Block<T>, p: usize, ws: &mut SchurWorkspace, out: &mut [f64]) {
    let n = blk.n;
    let zero = T::from_real(0.0);
    let ej = &blk.vars[p].1;
    let cols = &blk.cols[p];
    let k = cols.len();
    let tail = blk.tail[p];
    // flops, weighted by their measured speed relative to the scalar loops
    let setup = n * ej.len() + k * n;
    let cost_dense = (n * n * k) as f64 / GEMM_SPEEDUP + setup as f64 + tail as f64;
    let cost_entry = (k * tail) as f64 / DOT_SPEEDUP + setup as f64;
    let cost_sparse = (ej.len() * tail) as f64;
    let scratch = T::scratch(ws);
    if cost_sparse <= cost_dense.min(cost_entry) {
        // X[b, c] = conj(X[c, b]) from column b and Z⁻¹[d, a] from column a,
        // both contiguous in the inner loop (the column slices are looked up
        // once: per entry of F_i the lookup would cost as much as the short
        // inner loop over F_j)
        let xcols: Vec<&[T]> = (0..n).map(|j| blk.x.col_as_slice(j)).collect();
        let zcols: Vec<&[T]> = (0..n).map(|j| blk.zinv.col_as_slice(j)).collect();
        for ((_, ei), o) in blk.vars[p..].iter().zip(out.iter_mut()) {
            let mut s = zero;
            for (a, bb, vi) in ei {
                let (xb, za) = (xcols[*bb], zcols[*a]);
                for (cc, d, vj) in ej {
                    s += *vi * *vj * xb[*cc].conjugate() * za[*d];
                }
            }
            *o = s.real();
        }
        return;
    }
    // U[a, i] = Σ_{(c, D_j[i], v)} X[a, c] v, row-major
    let u = sized(&mut scratch.u, n * k);
    u.fill(zero);
    for (c, d, v) in ej {
        let i = cols.binary_search(d).expect("column of F_j");
        for a in 0..n {
            u[a * k + i] += blk.x[(a, *c)] * *v;
        }
    }
    // Zr[i, b] = Z⁻¹[D_j[i], b], column-major
    let zr = sized(&mut scratch.z, k * n);
    for b in 0..n {
        for (i, &d) in cols.iter().enumerate() {
            zr[b * k + i] = blk.zinv[(d, b)];
        }
    }
    let (u, zr) = (&scratch.u[..n * k], &scratch.z[..k * n]);
    if cost_entry <= cost_dense {
        // tr(F_i T) = Σ_{(a,b) ∈ F_i} conj(F_i[a,b]) T[a,b], T[a,b] = U[a,:] · Zr[:,b]
        for ((_, ei), o) in blk.vars[p..].iter().zip(out.iter_mut()) {
            let mut s = zero;
            for (a, bb, v) in ei {
                let (ur, zc) = (&u[a * k..(a + 1) * k], &zr[bb * k..(bb + 1) * k]);
                let mut t = zero;
                for (x, y) in ur.iter().zip(zc) {
                    t += *x * *y;
                }
                s += v.conjugate() * t;
            }
            *o = s.real();
        }
        return;
    }
    let t = sized(&mut scratch.t, n * n);
    // sequential kernel: the columns already run in parallel
    faer::linalg::matmul::matmul(
        MatMut::from_column_major_slice_mut(t, n, n),
        Accum::Replace,
        MatRef::from_row_major_slice(u, n, k),
        MatRef::from_column_major_slice(zr, k, n),
        T::one_impl(),
        Par::Seq,
    );
    let t = &scratch.t[..n * n];
    for ((_, ei), o) in blk.vars[p..].iter().zip(out.iter_mut()) {
        let mut s = zero;
        for (a, bb, v) in ei {
            s += v.conjugate() * t[bb * n + a];
        }
        *o = s.real();
    }
}

/// Speed per flop of the dense formula's matrix product relative to the
/// scattered scalar loops of the sparse formula.  Measured on a multi-functional
/// relaxation (6 blocks of 73–219, m = 7178) and QMC (one block of 295, m = 749):
/// assembly time is flat for weights between 4 and 256 and 1.5–2× slower
/// at 1.
const GEMM_SPEEDUP: f64 = 16.0;

/// The same for the entrywise formula's contiguous dot products.  Measured
/// on that multi-functional relaxation (m = 7178), SYK N = 20, QMC and a Lovász theta problem
/// (n = 200, m = 3929): weights 1–2 are neutral (within ±3% of never using
/// the formula); at 4 it is chosen over the matrix product for most dense
/// columns and SYK's assembly is 1.4× slower.  The blocks there are at most
/// a few hundred, where `T` stays in cache; the formula's saving (never
/// writing `T`) grows with `n`.
const DOT_SPEEDUP: f64 = 1.0;

/// Schur values held in memory at once during assembly (8 bytes each): the
/// columns of all blocks are computed in parallel in chunks of about this
/// size and added to `M` in a fixed order.
const SCHUR_CHUNK_ENTRIES: usize = 1 << 22;

/// Add `W_ij = Σ_blocks Re tr(F_i X F_j Z⁻¹)` to `schur`, each pair once,
/// at `(i, j)` for the block's pair order (so `M = W + Wᵀ − diag W`; see
/// [`factor_with_regularization`], which folds the two triangles).
fn assemble_schur(blocks: &[Box<dyn BlockOps>], schur: &mut Mat<f64>) {
    use rayon::prelude::*;
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new(); // (block, column, entries)
    for (b, blk) in blocks.iter().enumerate() {
        let nv = blk.schur_columns();
        for p in 0..nv {
            tasks.push((b, p, nv - p));
        }
    }
    let mut values: Vec<f64> = Vec::new();
    let mut start = 0;
    while start < tasks.len() {
        let mut end = start;
        let mut entries = 0;
        while end < tasks.len() && (end == start || entries + tasks[end].2 <= SCHUR_CHUNK_ENTRIES) {
            entries += tasks[end].2;
            end += 1;
        }
        let chunk = &tasks[start..end];
        values.resize(entries, 0.0);
        {
            // one disjoint slice of `values` per column
            let mut pieces: Vec<(&(usize, usize, usize), &mut [f64])> = Vec::with_capacity(chunk.len());
            let mut rest = &mut values[..];
            for task in chunk {
                let (piece, tail) = rest.split_at_mut(task.2);
                pieces.push((task, piece));
                rest = tail;
            }
            pieces
                .into_par_iter()
                .for_each_init(SchurWorkspace::default, |ws, (&(b, p, _), out)| {
                    blocks[b].schur_column(p, ws, out)
                });
        }
        // the columns of M are filled in parallel; within a column the
        // contributions keep their task order (deterministic)
        let mut by_column: Vec<Vec<(usize, usize)>> = vec![Vec::new(); schur.ncols()];
        let mut offset = 0;
        for (t, &(b, p, len)) in chunk.iter().enumerate() {
            by_column[blocks[b].globals()[p]].push((t, offset));
            offset += len;
        }
        let fill = |(mut col, list): (faer::ColMut<'_, f64>, &Vec<(usize, usize)>)| {
            for &(t, offset) in list {
                let (b, p, len) = chunk[t];
                let rows = &blocks[b].globals()[p..];
                for (&gi, &s) in rows.iter().zip(&values[offset..offset + len]) {
                    col[gi] += s;
                }
            }
        };
        #[cfg(feature = "parallel")]
        schur.par_col_iter_mut().zip(by_column.par_iter()).for_each(fill);
        #[cfg(not(feature = "parallel"))]
        schur.col_iter_mut().zip(by_column.iter()).for_each(fill);
        start = end;
    }
}

// ---------------------------------------------------------------------------
// The interior-point iteration
// ---------------------------------------------------------------------------

impl SdpModel {
    /// Solve the model with the native interior-point solver.  Equalities are
    /// eliminated internally with the rank-revealing sparse QR
    /// ([`SdpModel::eliminate_qr`], rank tolerance `1e-9`), which stays well
    /// conditioned on large redundant floating-point systems where row
    /// reduction does not; the solution is reported in the model's own
    /// variables.  Directions that every coefficient matrix of a block
    /// annihilates (`Z(y) v = 0` for all `y`) are removed first; that is an
    /// exact congruence and loses nothing.  With
    /// [`SolverOptions::facial_reduction`] a solve that stalls on a problem
    /// without a strictly feasible point is retried on its facial reduction.
    ///
    /// The solution's [`profile`](SdpSolution::profile) says where the time
    /// went; a solve dominated by one-off work (presolve, elimination) gets a
    /// warning ([`SolveProfile::one_off_warning`]), printed to stderr unless
    /// [`SolverOptions::warnings`] is off.
    pub fn solve(&self, opts: &SolverOptions) -> Result<SdpSolution, SolverError> {
        let t = Instant::now();
        let mut sol = match self.block_scales() {
            None => self.solve_inner(opts)?,
            Some(scales) => {
                // F_b ⪰ 0 ⇔ c_b F_b ⪰ 0; the duals of the scaled blocks are
                // c_b⁻¹ G_b (Σ_b ⟨G_b, F_j^b⟩ = c_j is unchanged)
                let mut scaled = self.clone();
                scaled.set_lmis(
                    self.lmis()
                        .iter()
                        .zip(&scales)
                        .map(|(l, &c)| {
                            let sc = |f: &SparseHerm| f.iter().map(|&(a, b, v)| (a, b, v * c)).collect();
                            Lmi {
                                size: l.size,
                                real: l.real,
                                f0: sc(&l.f0),
                                fj: l.fj.iter().map(|(j, f)| (*j, sc(f))).collect(),
                                label: l.label.clone(),
                            }
                        })
                        .collect(),
                );
                let mut sol = scaled.solve_inner(opts)?;
                for (g, &c) in sol.duals.iter_mut().zip(&scales) {
                    for row in g.iter_mut() {
                        for v in row.iter_mut() {
                            *v *= c;
                        }
                    }
                }
                sol
            }
        };
        sol.profile.total = t.elapsed().as_secs_f64();
        // facial reduction's extra solves are expected work, reported in the profile
        if sol.facial_reduction.is_none() {
            if let Some(w) = sol.profile.one_off_warning() {
                if opts.warnings {
                    eprintln!("warning: {w}");
                }
                sol.warnings.push(w);
            }
        }
        if opts.verbose {
            println!("{}", sol.profile.summary());
        }
        Ok(sol)
    }

    /// Per LMI, the factor `1 / max |entry|` for blocks whose largest
    /// coefficient is outside `[0.1, 10]`; `None` if no block needs one.
    /// The convergence measures are global, so a block with coefficients
    /// ~1e-9 could be violated by O(1) relative to its own size within the
    /// tolerance: a localizer `1e-9 (Z0 + Z1 + 1)` was silently ignored by
    /// every route, optimum −2.236 instead of −0.134.
    fn block_scales(&self) -> Option<Vec<f64>> {
        let scales: Vec<f64> = self
            .lmis()
            .iter()
            .map(|l| {
                let m = std::iter::once(&l.f0)
                    .chain(l.fj.values())
                    .flat_map(|f| f.iter().map(|(_, _, v)| v.norm()))
                    .fold(0.0_f64, f64::max);
                if m == 0.0 || (0.1..=10.0).contains(&m) {
                    1.0
                } else {
                    1.0 / m
                }
            })
            .collect();
        scales.iter().any(|&c| c != 1.0).then_some(scales)
    }

    /// [`solve`](Self::solve) without the overall timing and the warnings,
    /// for the solves run inside another one.
    pub(crate) fn solve_inner(&self, opts: &SolverOptions) -> Result<SdpSolution, SolverError> {
        if let Some(result) = crate::sdp::dualize::solve_dual(self, opts) {
            return result;
        }
        if !self.equalities().is_empty() {
            let t = Instant::now();
            let e = self.eliminate_qr(1e-9).map_err(SolverError)?;
            let eliminate = t.elapsed().as_secs_f64();
            let mut sol = e.model.solve_inner(opts)?;
            let t = Instant::now();
            let x = e.lift(&sol.x);
            // a regularized solve reports a certified bound, not the value at x
            let objective = if sol.regularization.is_some() {
                sol.objective
            } else {
                self.objective_value(&x)
            };
            sol.profile.elimination += eliminate + t.elapsed().as_secs_f64();
            return Ok(SdpSolution { x, objective, ..sol });
        }
        crate::sdp::facial::solve_presolved(self, opts)
    }
}

/// Dual (Gram) matrices at earlier iterates, for the facial reduction's
/// growth test: `(μ, duals)`, one snapshot each time μ fell by a factor 10.
pub(crate) struct Trace {
    /// `μ = ⟨X, Z⟩ / n` at the returned iterate.
    pub mu: f64,
    pub snapshots: Vec<(f64, Vec<Vec<Vec<C64>>>)>,
}

pub(crate) fn solve_lmi(model: &SdpModel, opts: &SolverOptions) -> Result<SdpSolution, SolverError> {
    solve_lmi_traced(model, opts).map(|r| r.0)
}

pub(crate) fn solve_lmi_traced(
    model: &SdpModel,
    opts: &SolverOptions,
) -> Result<(SdpSolution, Trace), SolverError> {
    let m = model.num_vars();
    if m > opts.max_variables {
        return Err(SolverError(format!(
            "{} free variables exceed SolverOptions::max_variables = {}: the dense Schur complement \
             would take {:.1} GB per iteration; export the problem (MomentForm::write_json / \
             SdpModel::write_json) to a first-order solver such as SCS, or raise the limit",
            m,
            opts.max_variables,
            (m as f64).powi(2) * 8.0 / 1e9
        )));
    }
    let t_setup = Instant::now();
    let mut prof = SolveProfile {
        solves: 1,
        ..Default::default()
    };
    let c: Vec<f64> = model.objective().to_vec();
    let mut blocks: Vec<Box<dyn BlockOps>> = model
        .lmis()
        .iter()
        .map(|lmi| -> Box<dyn BlockOps> {
            if lmi.real {
                Box::new(Block::<f64>::new(lmi))
            } else {
                Box::new(Block::<C64>::new(lmi))
            }
        })
        .collect();
    let mut occurs = vec![false; m];
    for blk in &blocks {
        for j in blk.variables() {
            occurs[j] = true;
        }
    }
    for (j, occ) in occurs.iter().enumerate() {
        if !occ {
            return Err(SolverError(format!(
                "variable {} ({}) occurs in no LMI block; fix it or remove it before solving",
                j,
                model.names()[j]
            )));
        }
    }
    let n_total: usize = blocks.iter().map(|b| b.size()).sum();
    if n_total == 0 {
        return Err(SolverError("the model has no constraints".to_string()));
    }
    let max_entry = blocks.iter().map(|b| b.max_entry()).fold(0.0_f64, f64::max);
    let c_norm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    let f0_norm = blocks.iter().map(|b| b.f0_norm_sq()).sum::<f64>().sqrt();
    let scale = max_entry
        .max(c.iter().fold(0.0_f64, |a, v| a.max(v.abs())))
        .max(1.0);
    let lambda = (n_total as f64).sqrt().max(1.0) * scale;

    for blk in blocks.iter_mut() {
        blk.init(lambda);
    }
    let mut y = vec![0.0; m];
    prof.setup = t_setup.elapsed().as_secs_f64();

    let mut status = Status::MaxIterations;
    let mut iterations = 0;
    let (mut pinf, mut dinf, mut relgap) = (f64::NAN, f64::NAN, f64::NAN);
    let tau = opts.step_fraction.clamp(0.5, 0.999);
    // best iterate so far, judged by the worst of the three convergence
    // measures: it is returned instead of the last iterate when the iteration
    // ends without meeting the tolerance, because on degenerate (exactly tight)
    // problems the iterates can be essentially optimal and then drift apart
    // again before the stop
    let mut best: Option<(f64, Vec<f64>, Vec<Vec<Vec<C64>>>, (f64, f64, f64))> = None;
    let mut best_mu = f64::NAN;
    let mut last_mu = f64::NAN;
    let mut snapshots: Vec<(f64, Vec<Vec<Vec<C64>>>)> = Vec::new();
    let mut schur_buffer: Option<Mat<f64>> = None;
    let t_loop = Instant::now();

    for iter in 0..=opts.max_iter {
        iterations = iter;
        // --- residuals and progress -----------------------------------------
        let mu = sum_blocks(&blocks, |b| b.gap()) / n_total as f64;
        let mut rp = c.clone();
        for part in gather_blocks(&blocks, |b| b.primal_inners()) {
            for (j, v) in part {
                rp[j] -= v;
            }
        }
        let rd_norm = blocks
            .par_iter_mut()
            .map(|b| b.dual_residual(&y))
            .collect::<Vec<f64>>()
            .into_iter()
            .fold(0.0_f64, f64::max);
        let pobj: f64 = -sum_blocks(&blocks, |b| b.f0_inner());
        let dobj: f64 = c.iter().zip(&y).map(|(c, y)| c * y).sum();
        relgap = (dobj - pobj).abs() / (1.0 + dobj.abs() + pobj.abs());
        let rp_norm = rp.iter().map(|v| v * v).sum::<f64>().sqrt();
        pinf = rp_norm / (1.0 + c_norm);
        dinf = rd_norm / (1.0 + f0_norm);
        if opts.verbose {
            println!(
                "iter {:3}  pobj {:+.10e}  dobj {:+.10e}  gap {:.2e}  pinf {:.2e}  dinf {:.2e}  mu {:.2e}",
                iter, pobj, dobj, relgap, pinf, dinf, mu
            );
        }
        let score = relgap.max(pinf).max(dinf);
        if score.is_finite() && best.as_ref().is_none_or(|b| score < b.0) {
            best = Some((score, y.clone(), duals_of(&blocks), (pinf, dinf, relgap)));
            best_mu = mu;
        }
        last_mu = mu;
        if opts.facial_reduction && mu.is_finite() && snapshots.last().is_none_or(|s| mu < 0.1 * s.0) {
            snapshots.push((mu, duals_of(&blocks)));
        }
        if relgap < opts.tol && pinf < opts.tol && dinf < opts.tol {
            status = Status::Optimal;
            break;
        }
        if iter == opts.max_iter {
            status = Status::MaxIterations;
            break;
        }
        if !mu.is_finite() || mu > 1e300 {
            status = Status::NumericalError;
            break;
        }

        // --- Z⁻¹ and the Schur complement ------------------------------------
        let inverted: Vec<bool> = blocks.par_iter_mut().map(|b| b.invert_z()).collect();
        if !inverted.into_iter().all(|ok| ok) {
            status = Status::NumericalError;
            break;
        }
        // one m × m buffer for the whole solve: assembled, factored in place,
        // used by the Schur solves and handed back for the next iteration
        let mut schur = match schur_buffer.take() {
            Some(mut b) => {
                b.as_mut().fill(0.0);
                b
            }
            None => Mat::<f64>::zeros(m, m),
        };
        let t = Instant::now();
        assemble_schur(&blocks, &mut schur);
        prof.schur_assembly += t.elapsed().as_secs_f64();
        let (refill_time, retries) = (Cell::new(0.0), Cell::new(0usize));
        let mut refill = |s: &mut Mat<f64>| {
            let t = Instant::now();
            s.as_mut().fill(0.0);
            assemble_schur(&blocks, s);
            refill_time.set(refill_time.get() + t.elapsed().as_secs_f64());
            retries.set(retries.get() + 1);
        };
        let t = Instant::now();
        let factored = factor_with_regularization(schur, &mut refill);
        prof.schur_factor += t.elapsed().as_secs_f64() - refill_time.get();
        prof.schur_assembly += refill_time.get();
        prof.factor_retries += retries.get();
        let chol = match factored {
            Ok(c) => c,
            Err(_) => {
                status = Status::NumericalError;
                break;
            }
        };
        // inexact Newton: residual target of the Schur solves (see solve_dy)
        let target = (PCG_ETA * rp_norm).max(0.1 * opts.tol * (1.0 + c_norm));

        // --- predictor (σ = 0): Q = −X − X R_d Z⁻¹ -------------------------------
        blocks.par_iter_mut().for_each(|b| b.predictor_rhs());
        let t = Instant::now();
        let (dy, products) = solve_dy(&blocks, &rp, &chol, target);
        prof.newton_solves += t.elapsed().as_secs_f64();
        prof.operator_products += products;
        blocks.par_iter_mut().for_each(|b| b.direction(&dy));
        let (ap, ad) = match max_steps(&mut blocks, true) {
            Some(s) => s,
            None => {
                status = Status::NumericalError;
                break;
            }
        };
        let (ap1, ad1) = (ap.min(1.0), ad.min(1.0));
        let mu_aff = sum_blocks(&blocks, |b| b.gap_after(ap1, ad1)) / n_total as f64;
        let sigma = ((mu_aff / mu).max(0.0)).powi(3).min(1.0);

        // --- corrector: Q = σμ Z⁻¹ − X − ΔX_a ΔZ_a Z⁻¹ − X R_d Z⁻¹ ----------------
        blocks.par_iter_mut().for_each(|b| b.corrector_rhs(sigma * mu));
        let t = Instant::now();
        let (dy, products) = solve_dy(&blocks, &rp, &chol, target);
        prof.newton_solves += t.elapsed().as_secs_f64();
        prof.operator_products += products;
        blocks.par_iter_mut().for_each(|b| b.direction(&dy));
        let fraction = |s: f64| if s.is_finite() { (tau * s).min(1.0) } else { 1.0 };
        let (mut ap, mut ad) = match max_steps(&mut blocks, true) {
            Some((ap, ad)) => (fraction(ap), fraction(ad)),
            None => {
                status = Status::NumericalError;
                break;
            }
        };
        // the estimated steps are verified by factoring the new iterates (the
        // factors are reused in the next iteration); if one is not positive
        // definite, the exact steps are taken
        let verified: Vec<bool> = blocks.par_iter_mut().map(|b| b.try_step(ap, ad)).collect();
        if !verified.into_iter().all(|ok| ok) {
            match max_steps(&mut blocks, false) {
                Some((a, d)) => {
                    (ap, ad) = (fraction(a), fraction(d));
                    blocks.par_iter_mut().for_each(|b| {
                        b.try_step(ap, ad);
                    });
                }
                None => {
                    status = Status::NumericalError;
                    break;
                }
            }
        }
        blocks.par_iter_mut().for_each(|b| b.step(ap, ad));
        for j in 0..m {
            y[j] += ad * dy[j];
        }
        schur_buffer = Some(chol.into_buffer());
    }

    let last_score = relgap.max(pinf).max(dinf);
    let (y, duals, pinf, dinf, relgap, mu) = match (status, best) {
        (Status::Optimal, _) | (_, None) => (y, duals_of(&blocks), pinf, dinf, relgap, last_mu),
        (_, Some((score, by, bd, (bp, bdi, bg)))) => {
            if score < last_score || !last_score.is_finite() {
                (by, bd, bp, bdi, bg, best_mu)
            } else {
                (y, duals_of(&blocks), pinf, dinf, relgap, last_mu)
            }
        }
    };
    let objective = model.objective_value(&y);
    prof.iterations = iterations;
    prof.iterate =
        (t_loop.elapsed().as_secs_f64() - prof.schur_assembly - prof.schur_factor - prof.newton_solves)
            .max(0.0);
    let solution = SdpSolution {
        x: y,
        objective,
        duals,
        status,
        iterations,
        primal_infeasibility: pinf,
        dual_infeasibility: dinf,
        relative_gap: relgap,
        facial_reduction: None,
        regularization: None,
        profile: prof,
        warnings: Vec::new(),
    };
    Ok((solution, Trace { mu, snapshots }))
}

/// `Σ_b f(block b)`, evaluated in parallel and added in block order.
fn sum_blocks(blocks: &[Box<dyn BlockOps>], f: impl Fn(&dyn BlockOps) -> f64 + Sync) -> f64 {
    let parts: Vec<f64> = blocks.par_iter().map(|b| f(b.as_ref())).collect();
    parts.into_iter().sum()
}

/// Per-block `(variable, value)` lists, evaluated in parallel, in block order.
fn gather_blocks(
    blocks: &[Box<dyn BlockOps>],
    f: impl Fn(&dyn BlockOps) -> Vec<(usize, f64)> + Sync,
) -> Vec<Vec<(usize, f64)>> {
    blocks.par_iter().map(|b| f(b.as_ref())).collect()
}

fn duals_of(blocks: &[Box<dyn BlockOps>]) -> Vec<Vec<Vec<C64>>> {
    blocks.par_iter().map(|b| b.dual()).collect()
}

/// Smallest primal and dual step lengths over all blocks.
fn max_steps(blocks: &mut [Box<dyn BlockOps>], estimate: bool) -> Option<(f64, f64)> {
    let steps: Vec<Option<(f64, f64)>> = blocks.par_iter_mut().map(|b| b.max_steps(estimate)).collect();
    let mut ap = f64::INFINITY;
    let mut ad = f64::INFINITY;
    for s in steps {
        let (a, d) = s?;
        ap = ap.min(a);
        ad = ad.min(d);
    }
    Some((ap, ad))
}

/// Forcing factor of the inexact Newton steps (see [`solve_dy`]).
const PCG_ETA: f64 = 0.1;

/// Solve the Schur system `M Δy = −r_p + Σ_b ⟨F_j^b, Q_b⟩` for the stored
/// right-hand sides `Q_b`: conjugate gradients on the exact operator
/// (applied block by block), preconditioned with the Cholesky factor of the
/// assembled `M` (near the optimum of a degenerate problem `M` is
/// so ill-conditioned that the factor alone is inaccurate).
///
/// Accuracy (inexact Newton): `ΔZ` and `ΔX` are computed exactly from `Δy`,
/// so the only equation an inexact `Δy` violates is `A(ΔX) = r_p`, and its
/// violation `ξ = A(ΔX) − r_p` is exactly the residual of the Schur system.
/// After a step `α` the primal residual is `(1 − α) r_p − α ξ`.  The solve
/// stops at `‖ξ‖ ≤ target = max(η ‖r_p‖, 0.1 tol (1 + ‖c‖))`: the primal
/// residual then still contracts by `1 − α (1 − η)` per iteration, and once
/// it is below the stopping tolerance the steps keep it there
/// (`η` = [`PCG_ETA`]; cf. the inexact interior-point methods of
/// Bellavia 1998 and Zhou–Toh 2004).  (Before, the solve ran to
/// `1e-13 ‖b‖`; the target saves 25–35% of the operator products on the
/// degenerate problems without a strictly feasible point.)  Returns `Δy` and the number of operator products.
fn solve_dy(blocks: &[Box<dyn BlockOps>], rp: &[f64], chol: &SchurFactor, target: f64) -> (Vec<f64>, usize) {
    let mut rhs: Vec<f64> = rp.iter().map(|v| -v).collect();
    for part in gather_blocks(blocks, |b| b.rhs_inners()) {
        for (j, v) in part {
            rhs[j] += v;
        }
    }
    let products = Cell::new(0usize);
    let apply = |v: &[f64]| -> Vec<f64> {
        products.set(products.get() + 1);
        let mut out = vec![0.0; v.len()];
        for part in gather_blocks(blocks, |b| b.schur_apply(v)) {
            for (j, x) in part {
                out[j] += x;
            }
        }
        out
    };
    let dy = pcg(&apply, chol, &rhs, 50, target);
    (dy, products.get())
}

fn norm2(v: &[f64]) -> f64 {
    ColRef::from_slice(v).norm_l2()
}

fn dotv(a: &[f64], b: &[f64]) -> f64 {
    inner_prod(
        ColRef::from_slice(a).transpose(),
        Conj::No,
        ColRef::from_slice(b),
        Conj::No,
    )
}

/// Preconditioned conjugate gradients for `M x = b` with `M` given by
/// `apply`, started from the preconditioner's solution.  Not faer's
/// `operator::conjugate_gradient`, which returns its last iterate and fails
/// on a non-positive-definite operator: near the optimum of degenerate
/// problems the Schur operator is numerically indefinite, and the iterate with the
/// smallest residual is the one wanted; stops once the
/// residual is at most `target` (or `1e-13 ‖b‖`, below which rounding
/// dominates) or after `max_iter` iterations, returning the iterate with
/// the smallest residual.  The residual is CG's recursively updated one;
/// near the optimum of degenerate problems it can be far below the true
/// residual `b − M x`, whose evaluation is itself at a rounding floor
/// there (`X S Z⁻¹` with `Z` nearly singular), so the true residual is no
/// better a guide.
fn pcg(
    apply: &dyn Fn(&[f64]) -> Vec<f64>,
    pre: &SchurFactor,
    b: &[f64],
    max_iter: usize,
    target: f64,
) -> Vec<f64> {
    let bn = norm2(b);
    let mut x = pre.solve(b);
    if bn == 0.0 {
        return x;
    }
    let stop = target.max(1e-13 * bn);
    let mx = apply(&x);
    let mut r: Vec<f64> = b.iter().zip(&mx).map(|(b, m)| b - m).collect();
    let mut best = (norm2(&r), x.clone());
    let mut z = pre.solve(&r);
    let mut p = z.clone();
    let mut rz = dotv(&r, &z);
    for _ in 0..max_iter {
        if best.0 <= stop {
            break;
        }
        let mp = apply(&p);
        let pmp = dotv(&p, &mp);
        if pmp.is_nan() || pmp <= 0.0 || !rz.is_finite() {
            break;
        }
        let a = rz / pmp;
        for i in 0..x.len() {
            x[i] += a * p[i];
            r[i] -= a * mp[i];
        }
        let rn = norm2(&r);
        if rn < best.0 {
            best = (rn, x.clone());
        }
        z = pre.solve(&r);
        let rz_new = dotv(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i];
        }
    }
    best.1
}

/// Cholesky factor of the diagonally scaled Schur complement `D M D`,
/// `D = diag(M)^{-1/2}`, factored in place in the iteration's Schur buffer
/// (lower triangle; the upper one is scratch).  The buffer is handed back
/// with [`SchurFactor::into_buffer`] and reused in the next iteration.
struct SchurFactor {
    buf: Mat<f64>,
    scale: Vec<f64>,
}

impl SchurFactor {
    /// `M⁻¹ r` for the (possibly regularized) factored `M`.
    fn solve(&self, r: &[f64]) -> Vec<f64> {
        let m = r.len();
        let mut x: Vec<f64> = r.iter().zip(&self.scale).map(|(r, s)| r * s).collect();
        // the triangular solves read only the lower triangle of `buf`
        let l = self.buf.as_ref();
        solve_lower_triangular_in_place(l, MatMut::from_column_major_slice_mut(&mut x[..], m, 1), Par::Seq);
        solve_upper_triangular_in_place(
            l.adjoint(),
            MatMut::from_column_major_slice_mut(&mut x[..], m, 1),
            Par::Seq,
        );
        for (x, s) in x.iter_mut().zip(&self.scale) {
            *x *= s;
        }
        x
    }

    fn into_buffer(self) -> Mat<f64> {
        self.buf
    }
}

/// Form the lower triangle of `D M D` from the assembled `W` (each pair added
/// once, at `(i, j)` or `(j, i)`, so `M = W + Wᵀ − diag W`): one pass over
/// square tiles, whose transposed reads stay in cache.
fn fold_and_scale(schur: &mut Mat<f64>, scale: &[f64]) {
    const TILE: usize = 64;
    let m = schur.nrows();
    for jb in (0..m).step_by(TILE) {
        let je = (jb + TILE).min(m);
        for ib in (jb..m).step_by(TILE) {
            let ie = (ib + TILE).min(m);
            for j in jb..je {
                for i in ib.max(j + 1)..ie {
                    let v = (schur[(i, j)] + schur[(j, i)]) * (scale[i] * scale[j]);
                    schur[(i, j)] = v;
                }
            }
        }
    }
    for i in 0..m {
        schur[(i, i)] *= scale[i] * scale[i];
    }
}

/// Factor the Schur complement in place.  `schur` holds `W` as assembled by
/// [`assemble_schur`]; the matrix is scaled to unit diagonal first (near the
/// optimum of a degenerate problem its diagonal spans many orders of
/// magnitude, 1e-2 to 1e17 on multi-functional problems without a strictly feasible
/// point, and a
/// regularization relative to the largest entry would swamp the small ones).
/// If the scaled matrix is numerically singular, a tiny multiple of the
/// identity is added and the factorization retried; the failed attempt has
/// overwritten the buffer (faer's in-place Cholesky uses both triangles), so
/// `refill` assembles `W` again.  That costs about one more assembly per
/// retry, which is rare, instead of a second `m × m` copy every iteration.
fn factor_with_regularization(
    mut schur: Mat<f64>,
    refill: &mut dyn FnMut(&mut Mat<f64>),
) -> Result<SchurFactor, Mat<f64>> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
    let m = schur.nrows();
    let scale: Vec<f64> = (0..m)
        .map(|i| {
            let d = schur[(i, i)];
            if d > 0.0 && d.is_finite() {
                1.0 / d.sqrt()
            } else {
                1.0
            }
        })
        .collect();
    fold_and_scale(&mut schur, &scale);
    let max_diag = (0..m)
        .map(|i| schur[(i, i)].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let par = faer::get_global_parallelism();
    let mut mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(m, par, Default::default()));
    let mut delta = 1e-14 * max_diag;
    let mut shift = 0.0;
    for attempt in 0..6 {
        if attempt > 0 {
            refill(&mut schur);
            fold_and_scale(&mut schur, &scale);
            shift += delta;
            delta *= 100.0;
            for i in 0..m {
                schur[(i, i)] += shift;
            }
        }
        let stack = MemStack::new(&mut mem);
        if cholesky_in_place(schur.as_mut(), Default::default(), par, stack, Default::default()).is_ok() {
            return Ok(SchurFactor { buf: schur, scale });
        }
    }
    Err(schur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdp::Affine;

    /// `W` with each off-diagonal pair of the SPD `M` stored once, at a
    /// pseudo-randomly chosen position (as `assemble_schur` leaves it).
    fn split_pairs(mm: &Mat<f64>) -> Mat<f64> {
        let m = mm.nrows();
        let mut w = Mat::<f64>::zeros(m, m);
        for j in 0..m {
            w[(j, j)] = mm[(j, j)];
            for i in j + 1..m {
                if (i * 7 + j * 3) % 5 < 2 {
                    w[(i, j)] = mm[(i, j)];
                } else {
                    w[(j, i)] = mm[(i, j)];
                }
            }
        }
        w
    }

    #[test]
    fn schur_factor_in_place() {
        // badly scaled SPD matrix of a size above faer's recursion threshold
        let m = 150;
        let mut state: u64 = 12345;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) - 0.5
        };
        let b = Mat::<f64>::from_fn(m, m, |_, _| rnd());
        let mut mm = &b * b.transpose();
        for i in 0..m {
            mm[(i, i)] += 1.0;
        }
        let d: Vec<f64> = (0..m).map(|i| 10f64.powi((i % 9) as i32 - 4)).collect();
        let mm = Mat::<f64>::from_fn(m, m, |i, j| d[i] * mm[(i, j)] * d[j]);
        let w = split_pairs(&mm);
        let f =
            factor_with_regularization(w.clone(), &mut |s| s.copy_from(&w)).unwrap_or_else(|_| panic!("SPD"));
        // the factor solves M x = r
        let r: Vec<f64> = (0..m).map(|i| (i as f64).sin()).collect();
        let x = f.solve(&r);
        let mx: Vec<f64> = (0..m).map(|i| (0..m).map(|j| mm[(i, j)] * x[j]).sum()).collect();
        let err = mx.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let mmax = (0..m).map(|i| mm[(i, i)]).fold(0.0, f64::max);
        let xmax = x.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            err <= 1e-13 * mmax * xmax,
            "residual {} (|M| {}, |x| {})",
            err,
            mmax,
            xmax
        );
    }

    #[test]
    fn schur_factor_retry_after_failure() {
        // positive semidefinite of rank 1: the plain factorization fails, the
        // shifted one (after reassembly) succeeds
        let m = 120;
        let v: Vec<f64> = (0..m).map(|i| 1.0 + (i as f64).cos()).collect();
        let mm = Mat::<f64>::from_fn(m, m, |i, j| v[i] * v[j]);
        let w = split_pairs(&mm);
        let mut refills = 0;
        let f = factor_with_regularization(w.clone(), &mut |s| {
            refills += 1;
            s.copy_from(&w)
        })
        .unwrap_or_else(|_| panic!("regularized"));
        assert!(refills > 0);
        // L Lᵀ = scaled M + δ I with a tiny δ
        let l = Mat::<f64>::from_fn(m, m, |i, j| if i >= j { f.buf[(i, j)] } else { 0.0 });
        let llt = &l * l.transpose();
        let mut shift: f64 = 0.0;
        for j in 0..m {
            for i in 0..m {
                let target = mm[(i, j)] * f.scale[i] * f.scale[j];
                if i == j {
                    shift = shift.max(llt[(i, j)] - target);
                } else {
                    assert!((llt[(i, j)] - target).abs() < 1e-9, "({}, {})", i, j);
                }
            }
        }
        assert!(shift > 0.0 && shift < 1e-6, "shift {}", shift);
    }

    #[test]
    fn two_by_two_lmi() {
        // min x + y s.t. [[x, 1], [1, y]] >= 0  ->  2 at x = y = 1
        let mut m = SdpModel::new();
        let x = m.free_var("x");
        let y = m.free_var("y");
        m.psd_affine(
            2,
            true,
            vec![
                (0, 0, Affine::var(x)),
                (0, 1, Affine::constant(1.0)),
                (1, 1, Affine::var(y)),
            ],
            "M",
        );
        m.minimize(&(Affine::var(x) + Affine::var(y)));
        let sol = m.solve(&SolverOptions::default()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        assert!((sol.objective - 2.0).abs() < 1e-6, "{}", sol.objective);
        assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 1.0).abs() < 1e-5);
        // the dual is X = [[1, -1], [-1, 1]]: X_11 = X_22 = 1 and max -2 X_12
        let d = &sol.duals[0];
        assert!((d[0][0].re - 1.0).abs() < 1e-5 && (d[0][1].re + 1.0).abs() < 1e-5);
    }

    #[test]
    fn complex_block_and_equality() {
        // max t s.t. [[1, t i], [-t i, 1]] >= 0  ->  t = 1 (eigenvalues 1 ± t)
        let mut m = SdpModel::new();
        let t = m.free_var("t");
        m.psd_affine(
            2,
            false,
            vec![
                (0, 0, Affine::constant(1.0)),
                (0, 1, Affine::var(t) * C64::new(0.0, 1.0)),
                (1, 1, Affine::constant(1.0)),
            ],
            "H",
        );
        m.maximize(&Affine::var(t));
        let sol = m.solve(&SolverOptions::default()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        assert!((sol.objective - 1.0).abs() < 1e-6, "{}", sol.objective);
        // the dual is the 2x2 Hermitian matrix itself: G = [[1, i], [-i, 1]] / 2
        // (Σ_ab G_ab F_ab = -1 for F = [[0, i], [-i, 0]], the maximized t)
        let d = &sol.duals[0];
        assert_eq!(d.len(), 2);
        assert!((d[0][0].re - 0.5).abs() < 1e-5, "{:?}", d);
        assert!((d[0][1] - C64::new(0.0, 0.5)).norm() < 1e-5, "{:?}", d);
        assert!((d[1][0] - d[0][1].conj()).norm() < 1e-12);
        let herm = sol.hermitian_duals(&m);
        assert_eq!(herm[0].len(), 2);
        assert_eq!(herm[0], *d);
        // with an equality: max t s.t. same block, t = s, s <= 1/2 via a 1x1 block
        let s = m.free_var("s");
        m.equal(&Affine::var(t), &Affine::var(s));
        m.nonneg(&(Affine::constant(0.5) - Affine::var(s)), "s <= 1/2");
        let sol = m.solve(&SolverOptions::default()).unwrap();
        assert_eq!(sol.status, Status::Optimal);
        assert!((sol.objective - 0.5).abs() < 1e-6, "{}", sol.objective);
        assert!((sol.x[0] - 0.5).abs() < 1e-5 && (sol.x[1] - 0.5).abs() < 1e-5);
    }
}
