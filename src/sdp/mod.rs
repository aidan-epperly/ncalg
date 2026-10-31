//! A small semidefinite-programming modeling layer.
//!
//! A [`SdpModel`] has real scalar variables `x`, linear matrix inequalities
//! `F_0 + Σ_j x_j F_j ⪰ 0` (Hermitian or real symmetric blocks), linear
//! equalities `Σ_j e_j x_j = r` and a linear objective.  Hermitian matrix
//! variables ([`MatrixVar`]) and positive semidefinite constraints on affine
//! matrix expressions are built from these primitives, so Gram-matrix (SOS)
//! formulations, joint problems such as the Gram-weight problem
//! `min Σ_u w_u  s.t.  h + c = Σ G_uv u* v,  G ⪰ 0,  diag(w) − G ⪰ 0`
//! and arbitrary side constraints on moments can be expressed next to the
//! moment relaxations of [`crate::ncsos`].
//!
//! Export: [`SdpModel::write_json`] (free variables, LMIs and equalities, read
//! by `scripts/solve_moment_json.py`), or [`SdpModel::eliminate`] followed by
//! [`SdpModel::write_sdpa`] (equalities solved out, SDPA sparse format for
//! CSDP / SDPA / Mosek).  Solutions are evaluated with
//! [`SdpModel::evaluate`], [`MatrixVar::value`] and [`SdpModel::check`].
//!
//! ```
//! use ncalg::sdp::{Affine, SdpModel};
//! // min x + y  s.t.  [[x, 1], [1, y]] ⪰ 0,  x = y   (optimum 2 at x = y = 1)
//! let mut m = SdpModel::new();
//! let x = m.free_var("x");
//! let y = m.free_var("y");
//! m.psd_affine(2, true, vec![(0, 0, Affine::var(x)), (0, 1, Affine::constant(1.0)), (1, 1, Affine::var(y))], "M");
//! m.equal(&Affine::var(x), &Affine::var(y));
//! m.minimize(&(Affine::var(x) + Affine::var(y)));
//! let e = m.eliminate().unwrap();           // one free variable left
//! assert_eq!(e.model.num_vars(), 1);
//! let x = e.lift(&[1.0]);                   // z = 1  ->  x = y = 1
//! assert!(m.check(&x, 1e-9).feasible());
//! assert!((m.objective_value(&x) - 2.0).abs() < 1e-12);
//! ```

mod dualize;
pub(crate) mod facial;
pub mod solver;

use num_complex::Complex;
use std::collections::BTreeMap;
use std::fmt::Write as _;

type C64 = Complex<f64>;

/// Variable index.
pub type VarId = usize;

use crate::linalg::dense::min_eigenvalue;
use crate::linalg::sparse::rref_sparse;
pub use crate::linalg::sparse::{SparseHerm, SparseSym};

/// Complex-valued affine function `constant + Σ_j coeff_j x_j` of the real
/// model variables.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Affine {
    /// The coefficients `coeff_j`, by variable.
    pub terms: BTreeMap<VarId, C64>,
    /// The constant term.
    pub constant: C64,
}

impl Affine {
    /// The constant `c`.
    pub fn constant(c: impl Into<C64>) -> Self {
        Affine {
            terms: BTreeMap::new(),
            constant: c.into(),
        }
    }

    /// The variable `x_j`.
    pub fn var(j: VarId) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(j, C64::new(1.0, 0.0));
        Affine {
            terms,
            constant: C64::new(0.0, 0.0),
        }
    }

    /// Zero.
    pub fn zero() -> Self {
        Affine::default()
    }

    /// Is this identically zero (up to `tol`)?
    pub fn is_zero(&self, tol: f64) -> bool {
        self.constant.norm() <= tol && self.terms.values().all(|c| c.norm() <= tol)
    }

    /// Does the expression contain no variables?
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    /// Complex conjugate.
    pub fn conj(&self) -> Self {
        Affine {
            terms: self.terms.iter().map(|(j, c)| (*j, c.conj())).collect(),
            constant: self.constant.conj(),
        }
    }

    /// Multiply by a complex scalar.
    pub fn scale(&self, s: impl Into<C64>) -> Self {
        let s = s.into();
        Affine {
            terms: self.terms.iter().map(|(j, c)| (*j, c * s)).collect(),
            constant: self.constant * s,
        }
    }

    /// Add `s * other` in place.
    pub fn add_scaled(&mut self, other: &Affine, s: impl Into<C64>) {
        let s = s.into();
        for (j, c) in &other.terms {
            *self.terms.entry(*j).or_insert(C64::new(0.0, 0.0)) += c * s;
        }
        self.constant += other.constant * s;
    }

    /// Drop coefficients of magnitude at most `tol`.
    pub fn prune(&mut self, tol: f64) {
        self.terms.retain(|_, c| c.norm() > tol);
    }

    /// Real part as a real row `(Σ_j a_j x_j, constant)`.
    pub fn real_part(&self) -> (Vec<(VarId, f64)>, f64) {
        (
            self.terms
                .iter()
                .filter(|(_, c)| c.re != 0.0)
                .map(|(j, c)| (*j, c.re))
                .collect(),
            self.constant.re,
        )
    }

    /// Imaginary part as a real row `(Σ_j b_j x_j, constant)`.
    pub fn imag_part(&self) -> (Vec<(VarId, f64)>, f64) {
        (
            self.terms
                .iter()
                .filter(|(_, c)| c.im != 0.0)
                .map(|(j, c)| (*j, c.im))
                .collect(),
            self.constant.im,
        )
    }

    /// Is the expression real-valued for all real `x` (up to `tol`)?
    pub fn is_real(&self, tol: f64) -> bool {
        self.constant.im.abs() <= tol && self.terms.values().all(|c| c.im.abs() <= tol)
    }

    /// Value at `x`.
    pub fn evaluate(&self, x: &[f64]) -> C64 {
        let mut v = self.constant;
        for (j, c) in &self.terms {
            v += c * x[*j];
        }
        v
    }
}

impl std::ops::Add for Affine {
    type Output = Affine;
    fn add(mut self, rhs: Affine) -> Affine {
        self.add_scaled(&rhs, 1.0);
        self
    }
}

impl std::ops::Add<&Affine> for Affine {
    type Output = Affine;
    fn add(mut self, rhs: &Affine) -> Affine {
        self.add_scaled(rhs, 1.0);
        self
    }
}

impl std::ops::Sub for Affine {
    type Output = Affine;
    fn sub(mut self, rhs: Affine) -> Affine {
        self.add_scaled(&rhs, -1.0);
        self
    }
}

impl std::ops::Sub<&Affine> for Affine {
    type Output = Affine;
    fn sub(mut self, rhs: &Affine) -> Affine {
        self.add_scaled(rhs, -1.0);
        self
    }
}

impl std::ops::Neg for Affine {
    type Output = Affine;
    fn neg(self) -> Affine {
        self.scale(-1.0)
    }
}

impl std::ops::Mul<f64> for Affine {
    type Output = Affine;
    fn mul(self, rhs: f64) -> Affine {
        self.scale(rhs)
    }
}

impl std::ops::Mul<C64> for Affine {
    type Output = Affine;
    fn mul(self, rhs: C64) -> Affine {
        self.scale(rhs)
    }
}

impl std::ops::Mul<Affine> for f64 {
    type Output = Affine;
    fn mul(self, rhs: Affine) -> Affine {
        rhs.scale(self)
    }
}

impl std::ops::AddAssign<&Affine> for Affine {
    fn add_assign(&mut self, rhs: &Affine) {
        self.add_scaled(rhs, 1.0);
    }
}

impl std::ops::SubAssign<&Affine> for Affine {
    fn sub_assign(&mut self, rhs: &Affine) {
        self.add_scaled(rhs, -1.0);
    }
}

/// A Hermitian (or real symmetric) matrix variable: its independent real
/// entries are consecutive model variables.
#[derive(Clone, Debug)]
pub struct MatrixVar {
    size: usize,
    real: bool,
    first: VarId,
    name: String,
}

impl MatrixVar {
    /// Matrix size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Real symmetric (no imaginary parts)?
    pub fn is_real(&self) -> bool {
        self.real
    }

    /// Name given at creation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of real variables.
    pub fn num_vars(&self) -> usize {
        let m = self.size;
        if self.real {
            m * (m + 1) / 2
        } else {
            m * m
        }
    }

    /// Variable holding `Re X[r, c]` (`r <= c`).
    pub fn re_var(&self, r: usize, c: usize) -> VarId {
        let (r, c) = if r <= c { (r, c) } else { (c, r) };
        assert!(c < self.size, "matrix index out of range");
        // upper triangle in row-major order
        self.first + r * self.size - r * (r + 1) / 2 + c
    }

    /// Variable holding `Im X[r, c]` (`r < c`), `None` for real matrices or
    /// diagonal entries.
    pub fn im_var(&self, r: usize, c: usize) -> Option<VarId> {
        if self.real || r == c {
            return None;
        }
        let (r, c) = if r <= c { (r, c) } else { (c, r) };
        assert!(c < self.size, "matrix index out of range");
        let m = self.size;
        // strict upper triangle in row-major order, after the real parts
        Some(self.first + m * (m + 1) / 2 + r * (m - 1) - (r * r - r) / 2 + (c - r - 1))
    }

    /// The entry `X[r, c]` as an affine expression (`X[c, r] = conj X[r, c]`).
    pub fn entry(&self, r: usize, c: usize) -> Affine {
        let mut e = Affine::var(self.re_var(r, c));
        if let Some(b) = self.im_var(r, c) {
            let sign = if r <= c { 1.0 } else { -1.0 };
            e.terms.insert(b, C64::new(0.0, sign));
        }
        e
    }

    /// Trace.
    pub fn trace(&self) -> Affine {
        let mut t = Affine::zero();
        for i in 0..self.size {
            t.terms.insert(self.re_var(i, i), C64::new(1.0, 0.0));
        }
        t
    }

    /// The full matrix at a solution `x`.
    pub fn value(&self, x: &[f64]) -> Vec<Vec<C64>> {
        let m = self.size;
        let mut out = vec![vec![C64::new(0.0, 0.0); m]; m];
        for r in 0..m {
            for c in r..m {
                let v = self.entry(r, c).evaluate(x);
                out[r][c] = v;
                out[c][r] = v.conj();
            }
        }
        out
    }

    /// Upper triangle of the affine expression `X` itself, for [`SdpModel::psd_affine`].
    pub fn entries(&self) -> Vec<(usize, usize, Affine)> {
        let mut v = Vec::new();
        for r in 0..self.size {
            for c in r..self.size {
                v.push((r, c, self.entry(r, c)));
            }
        }
        v
    }
}

/// A linear matrix inequality `F_0 + Σ_j x_j F_j ⪰ 0`.
#[derive(Clone, Debug)]
pub struct Lmi {
    /// Hermitian size.
    pub size: usize,
    /// Real symmetric (exported without the real embedding)?
    pub real: bool,
    /// Constant matrix.
    pub f0: SparseHerm,
    /// Coefficient matrices of the variables that occur.
    pub fj: BTreeMap<VarId, SparseHerm>,
    /// Label (for diagnostics and file comments).
    pub label: String,
}

impl Lmi {
    /// Size of the exported real block (`2 size` for complex blocks).
    pub fn embedded_size(&self) -> usize {
        if self.real {
            self.size
        } else {
            2 * self.size
        }
    }

    /// The matrix at `x`.
    pub fn value(&self, x: &[f64]) -> Vec<Vec<C64>> {
        let m = self.size;
        let mut out = vec![vec![C64::new(0.0, 0.0); m]; m];
        let mut add = |r: usize, c: usize, v: C64| {
            out[r][c] += v;
            if r != c {
                out[c][r] += v.conj();
            }
        };
        for (r, c, v) in &self.f0 {
            add(*r, *c, *v);
        }
        for (j, f) in &self.fj {
            let xj = x[*j];
            if xj == 0.0 {
                continue;
            }
            for (r, c, v) in f {
                add(*r, *c, v * xj);
            }
        }
        out
    }
}

/// Result of [`SdpModel::check`].
#[derive(Clone, Debug)]
pub struct Check {
    /// Largest absolute equality residual.
    pub max_equality_residual: f64,
    /// Smallest eigenvalue over all LMI blocks (`+inf` if there are none).
    pub min_eigenvalue: f64,
    /// Tolerance used.
    pub tol: f64,
}

impl Check {
    /// All equalities hold and all blocks are PSD within the tolerance.
    pub fn feasible(&self) -> bool {
        self.max_equality_residual <= self.tol && self.min_eigenvalue >= -self.tol
    }
}

/// A semidefinite program in free real variables.
#[derive(Clone, Debug, Default)]
pub struct SdpModel {
    names: Vec<String>,
    lmis: Vec<Lmi>,
    equalities: Vec<(Vec<(VarId, f64)>, f64)>,
    objective: Vec<f64>,
    objective_constant: f64,
    minimize: bool,
    /// Bare matrix variables `(lmi index, first variable, size, real)` from
    /// [`psd_var`](Self::psd_var), in LMI order, for solving standard-form
    /// models through their dual.  Cleared by operations that rewrite
    /// the LMIs.
    psd_blocks: Vec<(usize, usize, usize, bool)>,
}

impl SdpModel {
    /// Empty model (minimization by default).
    pub fn new() -> Self {
        SdpModel {
            minimize: true,
            ..Default::default()
        }
    }

    /// Number of real variables.
    pub fn num_vars(&self) -> usize {
        self.names.len()
    }

    /// Variable names (empty strings for anonymous variables).
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The LMI blocks.
    pub fn lmis(&self) -> &[Lmi] {
        &self.lmis
    }

    /// The linear equalities `(Σ_j e_j x_j, rhs)`.
    pub fn equalities(&self) -> &[(Vec<(VarId, f64)>, f64)] {
        &self.equalities
    }

    /// Objective vector (negated internally for `maximize`).
    pub fn objective(&self) -> &[f64] {
        &self.objective
    }

    /// Objective constant (as stored, already negated for `maximize`).
    pub fn objective_constant(&self) -> f64 {
        self.objective_constant
    }

    /// Is this a minimization?
    pub fn is_minimize(&self) -> bool {
        self.minimize
    }

    /// Exported block sizes (complex blocks doubled).
    pub fn block_sizes(&self) -> Vec<usize> {
        self.lmis.iter().map(|l| l.embedded_size()).collect()
    }

    /// Add a free real variable.
    pub fn free_var(&mut self, name: impl Into<String>) -> VarId {
        self.names.push(name.into());
        self.objective.push(0.0);
        self.names.len() - 1
    }

    /// Append the variables, LMIs and equalities of `other` (not its
    /// objective): variable `j` of `other` becomes `offset + j`, where
    /// `offset` (returned) is this model's previous number of variables.
    /// Names and LMI labels get `prefix`.
    pub fn append(&mut self, other: &SdpModel, prefix: &str) -> VarId {
        self.psd_blocks.clear();
        let offset = self.num_vars();
        for name in &other.names {
            self.free_var(format!("{}{}", prefix, name));
        }
        for lmi in &other.lmis {
            let mut l = lmi.clone();
            l.fj = lmi.fj.iter().map(|(j, f)| (j + offset, f.clone())).collect();
            l.label = format!("{}{}", prefix, lmi.label);
            self.lmis.push(l);
        }
        for (row, rhs) in &other.equalities {
            self.equalities
                .push((row.iter().map(|(j, c)| (j + offset, *c)).collect(), *rhs));
        }
        offset
    }

    /// Add `n` free variables named `prefix0, prefix1, …`.
    pub fn free_vars(&mut self, n: usize, prefix: &str) -> Vec<VarId> {
        (0..n)
            .map(|i| self.free_var(format!("{}{}", prefix, i)))
            .collect()
    }

    /// Add a variable constrained to be nonnegative (a 1×1 block).
    pub fn nonneg_var(&mut self, name: impl Into<String>) -> VarId {
        let name = name.into();
        let j = self.free_var(name.clone());
        self.nonneg(&Affine::var(j), name);
        j
    }

    /// Add a Hermitian (`real = false`) or real symmetric matrix variable
    /// without any constraint.
    pub fn hermitian_var(&mut self, size: usize, real: bool, name: impl Into<String>) -> MatrixVar {
        let name = name.into();
        let first = self.names.len();
        let mv = MatrixVar {
            size,
            real,
            first,
            name: name.clone(),
        };
        for r in 0..size {
            for c in r..size {
                self.free_var(format!("Re {}[{},{}]", name, r, c));
            }
        }
        if !real {
            for r in 0..size {
                for c in r + 1..size {
                    self.free_var(format!("Im {}[{},{}]", name, r, c));
                }
            }
        }
        debug_assert_eq!(self.names.len() - first, mv.num_vars());
        mv
    }

    /// Add a positive semidefinite matrix variable `X ⪰ 0`.
    pub fn psd_var(&mut self, size: usize, real: bool, name: impl Into<String>) -> MatrixVar {
        let name = name.into();
        let mv = self.hermitian_var(size, real, name.clone());
        self.psd_blocks.push((self.lmis.len(), mv.first, size, real));
        let mut fj: BTreeMap<VarId, SparseHerm> = BTreeMap::new();
        for r in 0..size {
            for c in r..size {
                fj.entry(mv.re_var(r, c))
                    .or_default()
                    .push((r, c, C64::new(1.0, 0.0)));
                if let Some(b) = mv.im_var(r, c) {
                    fj.entry(b).or_default().push((r, c, C64::new(0.0, 1.0)));
                }
            }
        }
        self.lmis.push(Lmi {
            size,
            real,
            f0: Vec::new(),
            fj,
            label: name,
        });
        mv
    }

    /// Constrain the Hermitian affine matrix with upper-triangle entries
    /// `(r, c, expr)` (`r <= c`; omitted entries are zero) to be PSD.
    /// Diagonal entries must be real-valued.  Returns the block index.
    pub fn psd_affine(
        &mut self,
        size: usize,
        real: bool,
        entries: Vec<(usize, usize, Affine)>,
        label: impl Into<String>,
    ) -> usize {
        let mut f0: BTreeMap<(usize, usize), C64> = BTreeMap::new();
        let mut fj: BTreeMap<VarId, BTreeMap<(usize, usize), C64>> = BTreeMap::new();
        for (r, c, e) in entries {
            assert!(
                r <= c && c < size,
                "psd_affine: entry ({}, {}) must satisfy r <= c < size",
                r,
                c
            );
            if r == c {
                assert!(
                    e.is_real(1e-12),
                    "psd_affine: diagonal entry ({0}, {0}) is not real-valued",
                    r
                );
            }
            if real {
                assert!(
                    e.is_real(1e-12),
                    "psd_affine: real block has a complex entry ({}, {})",
                    r,
                    c
                );
            }
            if e.constant.norm() > 0.0 {
                *f0.entry((r, c)).or_insert(C64::new(0.0, 0.0)) += e.constant;
            }
            for (j, coef) in &e.terms {
                assert!(*j < self.num_vars(), "psd_affine: unknown variable {}", j);
                if coef.norm() > 0.0 {
                    *fj.entry(*j)
                        .or_default()
                        .entry((r, c))
                        .or_insert(C64::new(0.0, 0.0)) += coef;
                }
            }
        }
        let to_sparse = |m: BTreeMap<(usize, usize), C64>| -> SparseHerm {
            m.into_iter()
                .filter(|(_, v)| v.norm() > 0.0)
                .map(|((r, c), v)| (r, c, v))
                .collect()
        };
        self.lmis.push(Lmi {
            size,
            real,
            f0: to_sparse(f0),
            fj: fj.into_iter().map(|(j, m)| (j, to_sparse(m))).collect(),
            label: label.into(),
        });
        self.lmis.len() - 1
    }

    /// Add a ready-made LMI block (its variable indices must exist).
    pub fn add_lmi(&mut self, lmi: Lmi) -> usize {
        for j in lmi.fj.keys() {
            assert!(*j < self.num_vars(), "add_lmi: unknown variable {}", j);
        }
        self.lmis.push(lmi);
        self.lmis.len() - 1
    }

    /// Replace the LMI blocks wholesale (same variables).
    pub(crate) fn set_lmis(&mut self, lmis: Vec<Lmi>) {
        self.lmis = lmis;
        self.psd_blocks.clear();
    }

    /// Replace the objective (`c`, constant, sense) wholesale.
    pub(crate) fn set_raw_objective(&mut self, c: Vec<f64>, constant: f64, minimize: bool) {
        assert_eq!(c.len(), self.num_vars());
        self.objective = c;
        self.objective_constant = constant;
        self.minimize = minimize;
    }

    /// Add a raw linear equality `Σ_j c_j x_j = rhs`.
    pub fn add_equality(&mut self, row: Vec<(VarId, f64)>, rhs: f64) {
        for (j, _) in &row {
            assert!(*j < self.num_vars(), "add_equality: unknown variable {}", j);
        }
        self.equalities.push((row, rhs));
    }

    /// Constrain a real-valued affine expression to be nonnegative (1×1 block).
    pub fn nonneg(&mut self, expr: &Affine, label: impl Into<String>) -> usize {
        assert!(expr.is_real(1e-12), "nonneg: expression is not real-valued");
        self.psd_affine(1, true, vec![(0, 0, expr.clone())], label)
    }

    /// Impose `expr = 0` (real and imaginary parts).  Panics on an
    /// inconsistent constant equation.
    pub fn equal_zero(&mut self, expr: &Affine) {
        for (row, c) in [expr.real_part(), expr.imag_part()] {
            if row.is_empty() {
                assert!(
                    c.abs() <= 1e-9,
                    "equal_zero: inconsistent constant equation {} = 0",
                    c
                );
                continue;
            }
            self.equalities.push((row, -c));
        }
    }

    /// Impose `lhs = rhs`.
    pub fn equal(&mut self, lhs: &Affine, rhs: &Affine) {
        self.equal_zero(&(lhs.clone() - rhs));
    }

    /// Set the objective to minimize a real-valued expression.
    pub fn minimize(&mut self, expr: &Affine) {
        self.set_objective(expr, true);
    }

    /// Set the objective to maximize a real-valued expression.
    pub fn maximize(&mut self, expr: &Affine) {
        self.set_objective(expr, false);
    }

    fn set_objective(&mut self, expr: &Affine, minimize: bool) {
        assert!(expr.is_real(1e-12), "objective is not real-valued");
        let sign = if minimize { 1.0 } else { -1.0 };
        self.objective = vec![0.0; self.num_vars()];
        for (j, c) in &expr.terms {
            self.objective[*j] = sign * c.re;
        }
        self.objective_constant = sign * expr.constant.re;
        self.minimize = minimize;
    }

    /// The objective (in the user's sign convention) at `x`.
    pub fn objective_value(&self, x: &[f64]) -> f64 {
        let v: f64 = self.objective.iter().zip(x).map(|(c, x)| c * x).sum::<f64>() + self.objective_constant;
        if self.minimize {
            v
        } else {
            -v
        }
    }

    /// Convert a solver's optimum of the exported `min c·x + c0` into the
    /// user's objective value.
    pub fn value_from_optimum(&self, optimum: f64) -> f64 {
        let v = optimum + self.objective_constant;
        if self.minimize {
            v
        } else {
            -v
        }
    }

    /// Value of an affine expression at `x`.
    pub fn evaluate(&self, expr: &Affine, x: &[f64]) -> C64 {
        expr.evaluate(x)
    }

    /// Equality residuals `Σ_j e_j x_j − rhs`.
    pub fn equality_residuals(&self, x: &[f64]) -> Vec<f64> {
        self.equalities
            .iter()
            .map(|(row, rhs)| row.iter().map(|(j, c)| c * x[*j]).sum::<f64>() - rhs)
            .collect()
    }

    /// Smallest eigenvalue of LMI block `b` at `x`.
    pub fn min_eigenvalue(&self, b: usize, x: &[f64]) -> f64 {
        let v = self.lmis[b].value(x);
        min_eigenvalue(&v)
    }

    /// Feasibility check at `x`.
    pub fn check(&self, x: &[f64], tol: f64) -> Check {
        assert_eq!(x.len(), self.num_vars(), "solution vector has the wrong length");
        let max_equality_residual = self
            .equality_residuals(x)
            .iter()
            .fold(0.0_f64, |m, r| m.max(r.abs()));
        let min_eigenvalue = (0..self.lmis.len())
            .map(|b| self.min_eigenvalue(b, x))
            .fold(f64::INFINITY, f64::min);
        Check {
            max_equality_residual,
            min_eigenvalue,
            tol,
        }
    }

    /// Solve the equalities out: returns an equivalent model without
    /// equalities in new variables `z`, with `x = offset + N z`
    /// ([`Eliminated::lift`]).  Fails if the equalities are inconsistent.
    pub fn eliminate(&self) -> Result<Eliminated, String> {
        let n = self.num_vars();
        let rows: Vec<(BTreeMap<VarId, f64>, f64)> = self
            .equalities
            .iter()
            .map(|(row, rhs)| {
                let mut m: BTreeMap<VarId, f64> = BTreeMap::new();
                for (j, c) in row {
                    *m.entry(*j).or_insert(0.0) += c;
                }
                (m, *rhs)
            })
            .collect();
        let reduced = rref_sparse(rows, 1e-10)?;
        let is_pivot: Vec<bool> = {
            let mut v = vec![false; n];
            for (p, _, _) in &reduced {
                v[*p] = true;
            }
            v
        };
        // free variables become z
        let mut z_index: Vec<Option<usize>> = vec![None; n];
        let mut n_z = 0;
        for j in 0..n {
            if !is_pivot[j] {
                z_index[j] = Some(n_z);
                n_z += 1;
            }
        }
        // x_j = offset_j + Σ_f rows_j[f] z_f
        let mut offset = vec![0.0; n];
        let mut lift_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for j in 0..n {
            if let Some(z) = z_index[j] {
                lift_rows[j].push((z, 1.0));
            }
        }
        for (p, free, rhs) in &reduced {
            offset[*p] = *rhs;
            lift_rows[*p] = free
                .iter()
                .map(|(f, c)| (z_index[*f].expect("free column"), -c))
                .collect();
        }
        let names: Vec<String> = (0..n)
            .filter(|&j| z_index[j].is_some())
            .map(|j| self.names[j].clone())
            .collect();
        Ok(self.substitute(offset, lift_rows, names))
    }

    /// Solve the equalities out like [`eliminate`](Self::eliminate), but with
    /// the rank-revealing sparse QR of [`crate::sparse_qr`] instead of row
    /// reduction: the new variables span the kernel of the (homogenised)
    /// equality matrix with a well-conditioned basis.  Use it for large,
    /// redundant floating-point systems (row reduction declares them
    /// inconsistent or returns a badly scaled basis).  Rank
    /// tolerance `rank_tol` (relative, e.g. `1e-9`).
    pub fn eliminate_qr(&self, rank_tol: f64) -> Result<Eliminated, String> {
        let n = self.num_vars();
        // Σ_j c_j x_j − (rhs/s) x_n = 0 with the extra coordinate x_n = s
        let (rows, s) = homogenised_rows(&self.equalities, n);
        let kernel: Vec<Vec<(usize, f64)>> = if rows.is_empty() {
            (0..=n).map(|j| vec![(j, 1.0)]).collect()
        } else {
            let matrix = crate::sparse_qr::SparseRows { ncols: n + 1, rows };
            let opts = crate::sparse_qr::QrOptions {
                rank_tol,
                ..Default::default()
            };
            let qr = crate::sparse_qr::SparseQr::factor(&matrix, &opts)
                .map_err(|e| format!("eliminate_qr: {}", e))?;
            let check = qr.verify(&matrix, 200);
            if check.max_relative_residual.is_nan() || check.max_relative_residual > 100.0 * rank_tol {
                return Err(format!(
                    "eliminate_qr: kernel violates the equalities (relative residual {:e} at rank tolerance {:e})",
                    check.max_relative_residual, rank_tol
                ));
            }
            qr.kernel_basis(&opts)
                .map_err(|e| format!("eliminate_qr: {}", e))?
        };
        // the kernel vector with the largest constant component gives the offset;
        // the others are made constant-free
        let constant = |k: &Vec<(usize, f64)>| k.iter().find(|(j, _)| *j == n).map_or(0.0, |(_, v)| *v);
        let (base_idx, base_c) = kernel
            .iter()
            .enumerate()
            .map(|(i, k)| (i, constant(k)))
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .filter(|(_, c)| c.abs() > 1e-12)
            .ok_or_else(|| "eliminate_qr: the equalities are inconsistent".to_string())?;
        let base: BTreeMap<usize, f64> = kernel[base_idx].iter().map(|(j, v)| (*j, v / base_c)).collect();
        let mut offset = vec![0.0; n];
        for (j, v) in &base {
            if *j < n {
                // base has x_n = 1, i.e. one unit of rhs/s
                offset[*j] = *v * s;
            }
        }
        let mut lift_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut n_z = 0;
        for (i, k) in kernel.iter().enumerate() {
            if i == base_idx {
                continue;
            }
            let ck = constant(k);
            let mut col: BTreeMap<usize, f64> = k.iter().cloned().collect();
            if ck != 0.0 {
                for (j, v) in &base {
                    *col.entry(*j).or_insert(0.0) -= ck * v;
                }
            }
            for (j, v) in col {
                if j < n && v.abs() > 1e-14 {
                    lift_rows[j].push((n_z, v));
                }
            }
            n_z += 1;
        }
        let names = (0..n_z).map(|z| format!("z{}", z)).collect();
        Ok(self.substitute(offset, lift_rows, names))
    }

    /// The model in new variables `z` (one per name) with `x = offset + N z`,
    /// `N` given by `lift_rows[j] = [(z, N_jz)]`.
    fn substitute(
        &self,
        offset: Vec<f64>,
        lift_rows: Vec<Vec<(usize, f64)>>,
        names: Vec<String>,
    ) -> Eliminated {
        let n_z = names.len();
        let mut model = SdpModel::new();
        for name in names {
            model.free_var(name);
        }
        for lmi in &self.lmis {
            let mut f0: BTreeMap<(usize, usize), C64> = BTreeMap::new();
            for (r, c, v) in &lmi.f0 {
                *f0.entry((*r, *c)).or_insert(C64::new(0.0, 0.0)) += v;
            }
            let mut fz: BTreeMap<usize, BTreeMap<(usize, usize), C64>> = BTreeMap::new();
            for (j, f) in &lmi.fj {
                let off = offset[*j];
                if off != 0.0 {
                    for (r, c, v) in f {
                        *f0.entry((*r, *c)).or_insert(C64::new(0.0, 0.0)) += v * off;
                    }
                }
                for (z, coef) in &lift_rows[*j] {
                    let target = fz.entry(*z).or_default();
                    for (r, c, v) in f {
                        *target.entry((*r, *c)).or_insert(C64::new(0.0, 0.0)) += v * coef;
                    }
                }
            }
            let prune = |m: BTreeMap<(usize, usize), C64>| -> SparseHerm {
                m.into_iter()
                    .filter(|(_, v)| v.norm() > 1e-14)
                    .map(|((r, c), v)| (r, c, v))
                    .collect()
            };
            model.lmis.push(Lmi {
                size: lmi.size,
                real: lmi.real,
                f0: prune(f0),
                fj: fz
                    .into_iter()
                    .map(|(z, m)| (z, prune(m)))
                    .filter(|(_, f)| !f.is_empty())
                    .collect(),
                label: lmi.label.clone(),
            });
        }
        let mut objective = vec![0.0; n_z];
        let mut objective_constant = self.objective_constant;
        for (j, c) in self.objective.iter().enumerate() {
            if *c == 0.0 {
                continue;
            }
            objective_constant += c * offset[j];
            for (z, coef) in &lift_rows[j] {
                objective[*z] += c * coef;
            }
        }
        model.objective = objective;
        model.objective_constant = objective_constant;
        model.minimize = self.minimize;
        Eliminated {
            model,
            offset,
            rows: lift_rows,
        }
    }

    /// Write the model in SDPA sparse format, `min c·x  s.t.  Σ x_i F_i − F_0 ⪰ 0`
    /// per block (SDPA's `F_0` is the negative of this model's constant
    /// matrix).  The model must have no equalities ([`eliminate`](Self::eliminate)
    /// first); the constant term of the objective goes into the comment line.
    pub fn write_sdpa(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        assert!(
            self.equalities.is_empty(),
            "write_sdpa: the model has equalities; call eliminate() first"
        );
        let mut s = String::new();
        let _ = writeln!(
            s,
            "\"ncalg SDP model: value = {:.17e} + optimum{}",
            self.objective_constant,
            if self.minimize {
                ""
            } else {
                " (negate for the maximum)"
            }
        );
        let _ = writeln!(s, "{}", self.num_vars());
        let _ = writeln!(s, "{}", self.lmis.len());
        let sizes: Vec<String> = self.block_sizes().iter().map(|m| m.to_string()).collect();
        let _ = writeln!(s, "{}", sizes.join(" "));
        let c: Vec<String> = self.objective.iter().map(|c| format!("{:.17e}", c)).collect();
        let _ = writeln!(s, "{}", c.join(" "));
        for (b, lmi) in self.lmis.iter().enumerate() {
            for (r, col, v) in embed(&lmi.f0, lmi.size, lmi.real) {
                let _ = writeln!(s, "0 {} {} {} {:.17e}", b + 1, r + 1, col + 1, -v);
            }
        }
        for (b, lmi) in self.lmis.iter().enumerate() {
            for (j, f) in &lmi.fj {
                for (r, col, v) in embed(f, lmi.size, lmi.real) {
                    let _ = writeln!(s, "{} {} {} {} {:.17e}", j + 1, b + 1, r + 1, col + 1, v);
                }
            }
        }
        std::fs::write(path, s)
    }

    /// Write the model as JSON in the layout of
    /// [`crate::ncsos::MomentSdp::write_json`]: `num_vars`, `minimize`,
    /// `objective_constant`, `objective`, `blocks` (`size`, `real`, `f0`,
    /// `fj` per variable, upper triangles as `[r, c, re, im]`), `equalities`
    /// (`coeffs`, `rhs`) and `names`.  Solved by `scripts/solve_moment_json.py`.
    pub fn write_json(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut s = String::new();
        let _ = write!(
            s,
            "{{\"num_vars\":{},\"minimize\":{},\"objective_constant\":{:e},\"objective\":[",
            self.num_vars(),
            self.minimize,
            self.objective_constant
        );
        for (j, c) in self.objective.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            let _ = write!(s, "{:e}", c);
        }
        s.push_str("],\"blocks\":[");
        let entry = |s: &mut String, (r, c, v): &(usize, usize, C64)| {
            let _ = write!(s, "[{},{},{:e},{:e}]", r, c, v.re, v.im);
        };
        for (b, lmi) in self.lmis.iter().enumerate() {
            if b > 0 {
                s.push(',');
            }
            let _ = write!(
                s,
                "{{\"label\":{},\"size\":{},\"real\":{},\"f0\":[",
                json_string(&lmi.label),
                lmi.size,
                lmi.real
            );
            for (i, e) in lmi.f0.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                entry(&mut s, e);
            }
            s.push_str("],\"fj\":[");
            for j in 0..self.num_vars() {
                if j > 0 {
                    s.push(',');
                }
                s.push('[');
                if let Some(f) = lmi.fj.get(&j) {
                    for (i, e) in f.iter().enumerate() {
                        if i > 0 {
                            s.push(',');
                        }
                        entry(&mut s, e);
                    }
                }
                s.push(']');
            }
            s.push_str("]}");
        }
        s.push_str("],\"equalities\":[");
        for (q, (row, rhs)) in self.equalities.iter().enumerate() {
            if q > 0 {
                s.push(',');
            }
            s.push_str("{\"coeffs\":[");
            for (i, (idx, c)) in row.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                let _ = write!(s, "[{},{:e}]", idx, c);
            }
            let _ = write!(s, "],\"rhs\":{:e}}}", rhs);
        }
        s.push_str("],\"names\":[");
        for (j, name) in self.names.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            s.push_str(&json_string(name));
        }
        s.push_str("]}\n");
        std::fs::write(path, s)
    }
}

/// An equality-free model together with the affine map back to the original
/// variables (see [`SdpModel::eliminate`]).
#[derive(Clone, Debug)]
pub struct Eliminated {
    /// The reduced model in variables `z`.
    pub model: SdpModel,
    offset: Vec<f64>,
    rows: Vec<Vec<(usize, f64)>>,
}

impl Eliminated {
    /// Original variables `x = offset + N z`.
    pub fn lift(&self, z: &[f64]) -> Vec<f64> {
        assert_eq!(
            z.len(),
            self.model.num_vars(),
            "solution vector has the wrong length"
        );
        self.offset
            .iter()
            .zip(&self.rows)
            .map(|(o, row)| o + row.iter().map(|(zj, c)| c * z[*zj]).sum::<f64>())
            .collect()
    }

    /// Project a point of the original space to `z` (its free coordinates).
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        let mut z = vec![0.0; self.model.num_vars()];
        for (j, row) in self.rows.iter().enumerate() {
            if row.len() == 1 && row[0].1 == 1.0 && self.offset[j] == 0.0 {
                z[row[0].0] = x[j];
            }
        }
        z
    }
}

/// Real embedding of a Hermitian upper triangle: `[[Re, −Im], [Im, Re]]`
/// (returned as an upper triangle of the `2m × 2m` real symmetric matrix),
/// or the real parts when `real`.
pub(crate) fn embed(h: &SparseHerm, m: usize, real: bool) -> SparseSym {
    let mut out = Vec::with_capacity(if real { h.len() } else { 4 * h.len() });
    for (a, b, c) in h {
        if real {
            if c.re != 0.0 {
                out.push((*a, *b, c.re));
            }
        } else {
            if c.re != 0.0 {
                out.push((*a, *b, c.re));
                out.push((a + m, b + m, c.re));
            }
            // a Hermitian diagonal is real: an imaginary part there is
            // rounding, and (a, a+m) would break the embedding's structure
            if c.im != 0.0 && a != b {
                // (a, b+m) = -S[a,b]; (b, a+m) = -S[b,a] = S[a,b]
                out.push((*a, b + m, -c.im));
                out.push((*b, a + m, c.im));
            }
        }
    }
    out
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `y` with one triangle filled from the other when the other is exactly
/// zero and this one is not: solvers that write a sparse triangle (SDPA's
/// yMat, hand-parsed output) would otherwise lose every off-diagonal pair's
/// second half, and the Hermitian average halved them silently.
pub(crate) fn complete_triangle(y: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let n = y.len();
    let nz = |c: &C64| c.re != 0.0 || c.im != 0.0;
    let upper = (0..n).any(|a| (a + 1..n).any(|b| nz(&y[a][b])));
    let lower = (0..n).any(|a| (0..a).any(|b| nz(&y[a][b])));
    let mut g = y.to_vec();
    if upper && !lower {
        for a in 0..n {
            for b in 0..a {
                g[a][b] = y[b][a].conj();
            }
        }
    } else if lower && !upper {
        for a in 0..n {
            for b in a + 1..n {
                g[a][b] = y[b][a].conj();
            }
        }
    }
    g
}

/// `G = (A + D) + i (B − Bᵀ)` from a real `2m × 2m` dual `[[A, B], [Bᵀ, D]]`
/// of a real-embedded Hermitian block, so that `tr(Y · embed(F)) = Σ_ab G_ab F_ab`.
/// A dual given as one triangle is completed first.
pub(crate) fn unembed_dual(y: &[Vec<C64>], m: usize) -> Vec<Vec<C64>> {
    let y = &complete_triangle(y);
    let mut g = vec![vec![C64::new(0.0, 0.0); m]; m];
    for a in 0..m {
        for b in 0..m {
            let re = y[a][b].re + y[a + m][b + m].re;
            let im = y[a][b + m].re - y[b][a + m].re;
            g[a][b] = C64::new(re, im);
        }
    }
    g
}

/// The equality rows `[A | −b/s]`, homogenised with an extra coordinate
/// `x_n = s`, and the scale `s = max|b| / max|A|` (`1` without a nonzero
/// right-hand side).  With `x_n = 1` instead, a right-hand side much larger
/// than its row makes every unit-normalized row nearly `e_n`: independent
/// rows leave remainders of order `|A|/|b|` and are read as dependent, so at
/// `b = 1e10` and `rank_tol = 1e-9` of the two equalities `x_1 = x_2 = b`
/// only `x_1 + x_2 = 2b` survived, silently.
pub(crate) fn homogenised_rows(
    equalities: &[(Vec<(VarId, f64)>, f64)],
    n: usize,
) -> (Vec<Vec<(usize, f64)>>, f64) {
    let a_max = equalities
        .iter()
        .flat_map(|(row, _)| row.iter().map(|(_, c)| c.abs()))
        .fold(0.0_f64, f64::max);
    let b_max = equalities.iter().map(|(_, b)| b.abs()).fold(0.0_f64, f64::max);
    let s = if a_max > 0.0 && b_max > 0.0 {
        b_max / a_max
    } else {
        1.0
    };
    let rows = equalities
        .iter()
        .map(|(row, rhs)| {
            let mut m: BTreeMap<usize, f64> = BTreeMap::new();
            for (j, c) in row {
                *m.entry(*j).or_insert(0.0) += c;
            }
            if *rhs != 0.0 {
                m.insert(n, -rhs / s);
            }
            m.into_iter().filter(|(_, c)| *c != 0.0).collect()
        })
        .collect();
    (rows, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_var_indexing_is_a_bijection() {
        let mut m = SdpModel::new();
        let g = m.hermitian_var(4, false, "G");
        let mut seen = std::collections::HashSet::new();
        for r in 0..4 {
            for c in r..4 {
                assert!(seen.insert(g.re_var(r, c)));
                if let Some(b) = g.im_var(r, c) {
                    assert!(seen.insert(b));
                }
            }
        }
        assert_eq!(seen.len(), 16);
        assert_eq!(m.num_vars(), 16);
        assert_eq!(g.entry(2, 1), g.entry(1, 2).conj());
        let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let v = g.value(&x);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(v[r][c], v[c][r].conj());
            }
        }
    }

    #[test]
    fn rref_and_elimination() {
        // x0 + x1 = 1, x1 - x2 = 0, and a dependent copy
        let rows = vec![
            (BTreeMap::from([(0, 1.0), (1, 1.0)]), 1.0),
            (BTreeMap::from([(1, 1.0), (2, -1.0)]), 0.0),
            (BTreeMap::from([(0, 2.0), (1, 2.0)]), 2.0),
        ];
        let r = rref_sparse(rows, 1e-12).unwrap();
        assert_eq!(r.len(), 2);
        let bad = vec![
            (BTreeMap::from([(0, 1.0)]), 1.0),
            (BTreeMap::from([(0, 1.0)]), 2.0),
        ];
        assert!(rref_sparse(bad, 1e-12).is_err());

        let mut m = SdpModel::new();
        let x = m.free_vars(3, "x");
        m.equal(&(Affine::var(x[0]) + Affine::var(x[1])), &Affine::constant(1.0));
        m.equal(&Affine::var(x[1]), &Affine::var(x[2]));
        m.nonneg(&Affine::var(x[0]), "x0 >= 0");
        m.minimize(&(Affine::var(x[2]) * 3.0 + Affine::constant(0.5)));
        let e = m.eliminate().unwrap();
        assert_eq!(e.model.num_vars(), 1);
        for z in [0.0, 0.3, -2.0] {
            let xx = e.lift(&[z]);
            assert!(m.equality_residuals(&xx).iter().all(|r| r.abs() < 1e-12));
            assert!((e.model.objective_value(&[z]) - m.objective_value(&xx)).abs() < 1e-12);
            assert!((e.model.min_eigenvalue(0, &[z]) - m.min_eigenvalue(0, &xx)).abs() < 1e-12);
        }
    }

    #[test]
    fn embedding_is_psd_preserving() {
        // [[2, i], [-i, 2]] has eigenvalues 1 and 3; its real embedding too
        let h: SparseHerm = vec![
            (0, 0, C64::new(2.0, 0.0)),
            (0, 1, C64::new(0.0, 1.0)),
            (1, 1, C64::new(2.0, 0.0)),
        ];
        let e = embed(&h, 2, false);
        let mut dense = vec![vec![C64::new(0.0, 0.0); 4]; 4];
        for (r, c, v) in e {
            dense[r][c] += C64::new(v, 0.0);
            if r != c {
                dense[c][r] += C64::new(v, 0.0);
            }
        }
        assert!((min_eigenvalue(&dense) - 1.0).abs() < 1e-12);
    }
}
