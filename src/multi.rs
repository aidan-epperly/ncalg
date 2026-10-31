//! Several moment functionals in one SDP.
//!
//! Relaxations with several boundary states and time-integrated
//! functionals, sparse / chordal NPA or steady-state bounds use **several pseudo-states with no cross moments**,
//! each with its own moment matrix and normalization, coupled only by linear
//! equalities between their moments.  A [`MultiRelaxation`] holds one
//! ordinary [`Relaxation`] per functional (its own index set, localizing
//! matrices, moment constraints, symmetry and `L_k(1) = 1`), assembles each
//! with [`Relaxation::try_build`] (exact elimination, symmetry blocks,
//! connected components) and stacks them in one [`SdpModel`], adding
//!
//! * **links** `Σ_k L_k(p_k) = v` (real and imaginary parts), and
//! * the objective `Σ_k L_k(h_k)` (real-valued).
//!
//! ```
//! use ncalg::multi::MultiRelaxation;
//! use ncalg::ncsos::Relaxation;
//! use ncalg::solver::SolverOptions;
//! use ncalg::{relations, Complex, GroebnerBasis, GroebnerOptions, Polynomial};
//! type C = Complex<f64>;
//! // two qubit states, linked by <Z>_a + <Z>_b = 0; maximize <X>_a + <Z>_b
//! let (alg, rels) = relations::pauli::<C>(1);
//! let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
//! let p = |s: &str| -> Polynomial<C> { alg.parse(s).unwrap() };
//! let mut mr = MultiRelaxation::new(gb.reducer());
//! let a = mr.functional("a", Relaxation::level(gb.reducer(), 1));
//! let b = mr.functional("b", Relaxation::level(gb.reducer(), 1));
//! mr.link(&[(a, &p("Z0")), (b, &p("Z0"))], C::new(0.0, 0.0));
//! mr.maximize(&[(a, &p("X0")), (b, &p("Z0"))]);
//! let sdp = mr.build().unwrap();
//! let (value, _) = sdp.solve(&SolverOptions::default()).unwrap();
//! // <Z>_b = t, <Z>_a = -t, <X>_a <= sqrt(1 - t^2): max of sqrt(1 - t^2) + t is sqrt 2
//! assert!((value - 2f64.sqrt()).abs() < 1e-6);
//! ```

use crate::ncsos::{Certificate, MomentSdp, Relaxation, SdpScalar};
use crate::reduce::Reducer;
use crate::sdp::{Affine, Eliminated, SdpModel};
use crate::solver::{SdpSolution, SolverOptions};
use crate::word::Word;
use crate::Polynomial;
use num_complex::Complex;

type C64 = Complex<f64>;

/// Index of a functional in a [`MultiRelaxation`].
pub type FunctionalId = usize;

/// Builder for a relaxation with several moment functionals.
pub struct MultiRelaxation<'a, C: SdpScalar> {
    red: &'a Reducer<C>,
    functionals: Vec<(String, Relaxation<'a, C>)>,
    /// `(Σ_k L_k(p_k), value)`
    links: Vec<(Vec<(FunctionalId, Polynomial<C>)>, C)>,
    objective: Vec<(FunctionalId, Polynomial<C>)>,
    minimize: bool,
}

impl<'a, C: SdpScalar> std::fmt::Debug for MultiRelaxation<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.functionals.iter().map(|(n, _)| n.as_str()).collect();
        write!(
            f,
            "MultiRelaxation(functionals {:?}, {} links)",
            names,
            self.links.len()
        )
    }
}

impl<'a, C: SdpScalar> MultiRelaxation<'a, C> {
    /// No functionals yet; all of them must use the reducer `red`.
    pub fn new(red: &'a Reducer<C>) -> Self {
        MultiRelaxation {
            red,
            functionals: Vec::new(),
            links: Vec::new(),
            objective: Vec::new(),
            minimize: true,
        }
    }

    /// Add a functional `L_k` described by an ordinary relaxation (index set,
    /// localizing matrices, its own moment constraints and symmetry; its
    /// objective, if any, is ignored).  Each functional is normalized,
    /// `L_k(1) = 1`.
    ///
    /// Panics if `rel` reduces with a different ideal: links, the objective
    /// and constraints are reduced with this relaxation's reducer and looked
    /// up among `rel`'s moments, so a different ideal with the same normal
    /// words (Pauli against commuting qubits: `Z0 X0 → ∓X0 Z0`) would impose
    /// silently wrong values.
    pub fn functional(&mut self, name: &str, rel: Relaxation<'a, C>) -> FunctionalId {
        let r = rel.reducer();
        let same = std::ptr::eq(r, self.red)
            || (r.algebra().same_as(self.red.algebra()) && r.basis().iter().eq(self.red.basis().iter()));
        assert!(
            same,
            "functional '{}': its relaxation reduces with a different ideal than the MultiRelaxation",
            name
        );
        self.functionals.push((name.to_string(), rel));
        self.functionals.len() - 1
    }

    /// Impose `Σ_k L_k(p_k) = value` across functionals.
    pub fn link(&mut self, terms: &[(FunctionalId, &Polynomial<C>)], value: C) -> &mut Self {
        self.links
            .push((terms.iter().map(|(k, p)| (*k, (*p).clone())).collect(), value));
        self
    }

    /// Minimize `Σ_k L_k(h_k)` (real-valued: the `h_k` Hermitian).
    pub fn minimize(&mut self, terms: &[(FunctionalId, &Polynomial<C>)]) -> &mut Self {
        self.objective = terms.iter().map(|(k, p)| (*k, (*p).clone())).collect();
        self.minimize = true;
        self
    }

    /// Maximize `Σ_k L_k(h_k)`.
    pub fn maximize(&mut self, terms: &[(FunctionalId, &Polynomial<C>)]) -> &mut Self {
        self.minimize(terms);
        self.minimize = false;
        self
    }

    /// `Σ_k L_k(p_k)` as an affine expression in the stacked variables.
    fn affine(
        &self,
        terms: &[(FunctionalId, Polynomial<C>)],
        sdps: &[MomentSdp],
        offsets: &[usize],
        what: &str,
    ) -> Result<Affine, String> {
        let mut e = Affine::zero();
        for (k, p) in terms {
            let k = *k;
            if k >= sdps.len() {
                return Err(format!("{}: unknown functional {}", what, k));
            }
            let nf = self.red.reduce(p);
            let a = sdps[k].moment_affine(&nf).ok_or_else(|| {
                let missing: Vec<String> = nf
                    .iter()
                    .filter(|(w, _)| {
                        sdps[k]
                            .moment_affine(&self.red.algebra().monomial::<C>((*w).clone()))
                            .is_none()
                    })
                    .map(|(w, _)| self.red.algebra().format_word(w))
                    .collect();
                format!(
                    "{}: functional '{}' has no moment for {:?} (not a product u* g v of its index words)",
                    what, self.functionals[k].0, missing
                )
            })?;
            e.constant += a.constant;
            for (j, c) in a.terms {
                *e.terms.entry(offsets[k] + j).or_insert(C64::new(0.0, 0.0)) += c;
            }
        }
        // relative to the expression's size: an absolute 1e-15 emptied rows
        // whose coefficients are all that small (physical units)
        let scale = e.terms.values().map(|c| c.norm()).fold(0.0_f64, f64::max);
        e.prune(1e-15 * scale);
        Ok(e)
    }

    /// Assemble every functional and stack them with the links and objective.
    pub fn build(&self) -> Result<MultiSdp, String> {
        if self.functionals.is_empty() {
            return Err("MultiRelaxation: no functionals".to_string());
        }
        let mut model = SdpModel::new();
        let mut sdps = Vec::new();
        let mut offsets = Vec::new();
        let mut constraints = Vec::new();
        for (name, rel) in &self.functionals {
            let sdp = rel
                .try_build()
                .map_err(|e| format!("functional '{}': {}", name, e))?;
            let own = rel
                .effective_constraints()
                .map_err(|e| format!("functional '{}': {}", name, e))?;
            constraints.push(
                own.iter()
                    .map(|(p, v)| (to_c64(&self.red.reduce(p)), scalar_c64(v)))
                    .collect(),
            );
            offsets.push(model.append(&sdp.to_model(), &format!("{}.", name)));
            sdps.push(sdp);
        }
        for (i, (terms, value)) in self.links.iter().enumerate() {
            let mut e = self.affine(terms, &sdps, &offsets, &format!("link {}", i))?;
            let (re, im) = value.to_complex_f64();
            e.constant -= C64::new(re, im);
            // consistency relative to the link's size
            let scale = terms
                .iter()
                .flat_map(|(_, p)| p.iter().map(|(_, c)| c.magnitude()))
                .fold(C64::new(re, im).norm(), f64::max);
            for (row, c) in [e.real_part(), e.imag_part()] {
                if row.is_empty() {
                    if c.abs() > 1e-9 * scale {
                        return Err(format!("link {}: inconsistent ({} = 0 after elimination)", i, c));
                    }
                    continue;
                }
                model.add_equality(row, -c);
            }
        }
        let obj = self.affine(&self.objective, &sdps, &offsets, "objective")?;
        if !obj.is_real(1e-9) {
            return Err("objective is not real-valued (use Hermitian h_k)".to_string());
        }
        if self.minimize {
            model.minimize(&obj);
        } else {
            model.maximize(&obj);
        }
        let names = self.functionals.iter().map(|(n, _)| n.clone()).collect();
        let reduce_terms = |terms: &[(FunctionalId, Polynomial<C>)]| -> Vec<(FunctionalId, Polynomial<C64>)> {
            terms
                .iter()
                .map(|(k, p)| (*k, to_c64(&self.red.reduce(p))))
                .collect()
        };
        Ok(MultiSdp {
            model,
            functionals: sdps,
            names,
            offsets,
            links: self
                .links
                .iter()
                .map(|(t, v)| (reduce_terms(t), scalar_c64(v)))
                .collect(),
            objective: reduce_terms(&self.objective),
            constraints,
            minimize: self.minimize,
        })
    }
}

/// The assembled multi-functional SDP.
#[derive(Clone, Debug)]
pub struct MultiSdp {
    /// The stacked SDP (export with `eliminate()` + `write_sdpa`, or `write_json`).
    pub model: SdpModel,
    /// Each functional's eliminated relaxation (its variables are a slice of
    /// the model's, see [`functional_vars`](Self::functional_vars)).
    pub functionals: Vec<MomentSdp>,
    /// Functional names, in order.
    pub names: Vec<String>,
    offsets: Vec<usize>,
    /// Links, objective terms and each functional's own moment constraints
    /// in normal form, for [`certificate`](Self::certificate).
    links: Vec<(Vec<(FunctionalId, Polynomial<C64>)>, C64)>,
    objective: Vec<(FunctionalId, Polynomial<C64>)>,
    constraints: Vec<Vec<(Polynomial<C64>, C64)>>,
    minimize: bool,
}

fn scalar_c64<C: SdpScalar>(c: &C) -> C64 {
    let (re, im) = c.to_complex_f64();
    C64::new(re, im)
}

fn to_c64<C: SdpScalar>(p: &Polynomial<C>) -> Polynomial<C64> {
    p.map_coeffs(scalar_c64)
}

impl MultiSdp {
    /// Number of real variables of the stacked model.
    pub fn num_vars(&self) -> usize {
        self.model.num_vars()
    }

    /// Exported block sizes (complex blocks doubled), functional by functional.
    pub fn block_sizes(&self) -> Vec<usize> {
        self.model.block_sizes()
    }

    /// The variables of functional `k` within a solution `x` of the model.
    pub fn functional_vars<'x>(&self, k: FunctionalId, x: &'x [f64]) -> &'x [f64] {
        let n = self.functionals[k].num_vars();
        &x[self.offsets[k]..self.offsets[k] + n]
    }

    /// Moments `L_k(w)` of functional `k` at a solution `x`.
    pub fn moments(&self, k: FunctionalId, x: &[f64]) -> Vec<(Word, C64)> {
        self.functionals[k].moments(self.functional_vars(k, x))
    }

    /// The links (and any other equalities) solved out with the rank-revealing
    /// sparse QR ([`SdpModel::eliminate_qr`], rank tolerance `1e-9`), which is
    /// much better conditioned on large redundant link systems than row
    /// reduction.  Export the result with `write_sdpa` (CSDP) or `write_json`.
    pub fn eliminate(&self) -> Result<Eliminated, String> {
        self.model.eliminate_qr(1e-9)
    }

    /// Solve with the native solver after [`eliminate`](Self::eliminate):
    /// `(value, solution in the stacked variables)`, the value `NaN` unless
    /// the solution is converged to `opts.accept_tol`.
    pub fn solve(&self, opts: &SolverOptions) -> Result<(f64, SdpSolution), String> {
        let e = self.eliminate()?;
        let sol = e.model.solve(opts).map_err(|err| err.0)?;
        let converged = sol.is_converged(opts.accept_tol);
        let x = e.lift(&sol.x);
        let objective = if sol.regularization.is_some() {
            sol.objective
        } else {
            self.model.objective_value(&x)
        };
        let value = if converged { objective } else { f64::NAN };
        Ok((value, SdpSolution { x, objective, ..sol }))
    }
}

/// SOS certificate of a [`MultiSdp`] bound (see [`MultiSdp::certificate`]).
///
/// With `s = +1` for a minimization and `−1` for a maximization, it asserts,
/// for every functional `k`, modulo the ideal,
///
/// ```text
///   h_k − λ_k − Σ_l r_lk − Σ_c r_kc = s σ_k + e_k,
///   r = (conj(μ) p + μ p*) / 2   for a link or constraint term p with multiplier μ,
/// ```
///
/// with `σ_k` the functional's sum of squares (localizing terms included)
/// and `e_k` the residual.  Applying `L_k` and summing gives
/// `Σ_k L_k(h_k) ≥ bound` (minimization; `≤` for a maximization) for every
/// feasible family of functionals, up to `Σ_k L_k(e_k)`, which is at most
/// `residual_l1` when all moments of the residual's words are bounded by 1
/// (Pauli strings, unitaries, projectors).
#[derive(Clone, Debug)]
pub struct MultiCertificate {
    /// The certificate of each functional (its squares).
    pub functionals: Vec<Certificate>,
    /// `λ_k`, one per functional.
    pub constants: Vec<f64>,
    /// `μ_l`, one per link.
    pub link_multipliers: Vec<C64>,
    /// Multipliers of each functional's own moment constraints.
    pub constraint_multipliers: Vec<Vec<C64>>,
    /// `Σ_k λ_k + Σ_l Re(conj(μ_l) v_l) + Σ_c Re(conj(μ_c) v_c)`.
    pub bound: f64,
    /// `e_k` per functional.
    pub residuals: Vec<Polynomial<C64>>,
    /// Largest residual coefficient.  Sums that cancel to rounding level
    /// relative to their summands count as zero, as do coefficients at
    /// or below an absolute floor if one is set
    /// ([`crate::set_float_tolerance`], default `0`); for exact zero tests
    /// call [`MultiSdp::certificate`] inside
    /// [`crate::coeff::with_exact_zero`].
    pub max_residual: f64,
    /// Sum of the residual coefficients' magnitudes (same caveat).
    pub residual_l1: f64,
}

impl MultiSdp {
    /// SOS certificate from the solver's dual matrices (one per exported
    /// block, functional by functional, as [`solve`](Self::solve) returns
    /// them in `SdpSolution::duals`, or as read from a CSDP solution).
    ///
    /// Each functional's squares come from [`MomentSdp::certificate`]
    /// (Gram eigenvalues below `tol` dropped; averaged over its symmetry
    /// group).  The constants `λ_k` and the multipliers of the links and of
    /// the functionals' own constraints, which the elimination removed from
    /// the solved problem, are recovered by linear least squares on the
    /// identity of [`MultiCertificate`]; `red` reduces modulo the ideal in
    /// `Complex<f64>` (e.g. `gb.convert::<Complex<f64>>().reducer()`).  With
    /// a symmetry the objective, link and constraint polynomials are averaged
    /// over the functional's group, which is valid because the relaxation's
    /// functionals are invariant.
    pub fn certificate(&self, duals: &[Vec<Vec<C64>>], red: &Reducer<C64>, tol: f64) -> MultiCertificate {
        let alg = red.algebra();
        let nk = self.functionals.len();
        let sign = if self.minimize { 1.0 } else { -1.0 };
        let mut certs = Vec::with_capacity(nk);
        let mut start = 0;
        for f in &self.functionals {
            let nb = f.blocks().len();
            certs.push(f.certificate(&duals[start..start + nb], tol));
            start += nb;
        }
        assert_eq!(start, duals.len(), "one dual matrix per exported block expected");
        let average = |k: usize, p: &Polynomial<C64>| -> Polynomial<C64> {
            let group = self.functionals[k].group();
            red.reduce(&crate::symmetry::action::average(group, p, alg))
        };
        let hermitian_parts = |k: usize, p: &Polynomial<C64>| -> (Polynomial<C64>, Polynomial<C64>) {
            // conj(μ) p + μ p* = 2a H + 2b (−iA) for μ = a + i b, p = H + A
            let (h, a) = p.hermitian_split();
            (average(k, &h), average(k, &a.scale(&C64::new(0.0, -1.0))))
        };
        // unknowns (columns): λ_k, then (a, b) per link, then per constraint;
        // each column lists its polynomial per functional and its weight in
        // the bound
        let mut cols: Vec<(Vec<(usize, Polynomial<C64>)>, f64)> = Vec::new();
        for k in 0..nk {
            cols.push((vec![(k, alg.constant(C64::new(1.0, 0.0)))], 1.0));
        }
        let link_col = cols.len();
        for (terms, v) in &self.links {
            let parts: Vec<(usize, (Polynomial<C64>, Polynomial<C64>))> =
                terms.iter().map(|(k, p)| (*k, hermitian_parts(*k, p))).collect();
            cols.push((parts.iter().map(|(k, (h1, _))| (*k, h1.clone())).collect(), v.re));
            cols.push((parts.iter().map(|(k, (_, h2))| (*k, h2.clone())).collect(), v.im));
        }
        let mut constraint_col = Vec::with_capacity(nk);
        for (k, list) in self.constraints.iter().enumerate() {
            constraint_col.push(cols.len());
            for (p, v) in list {
                let (h1, h2) = hermitian_parts(k, p);
                cols.push((vec![(k, h1)], v.re));
                cols.push((vec![(k, h2)], v.im));
            }
        }
        // targets t_k = h_k − s σ_k
        let mut targets: Vec<Polynomial<C64>> = (0..nk)
            .map(|k| certs[k].sum().scale(&C64::new(-sign, 0.0)))
            .collect();
        for (k, h) in &self.objective {
            targets[*k] = targets[*k].add_ref(h);
        }
        let targets: Vec<Polynomial<C64>> = (0..nk).map(|k| average(k, &targets[k])).collect();
        // rows: (functional, word, re/im) of A x ≈ t
        let mut row_index: std::collections::HashMap<(usize, Word), usize> = std::collections::HashMap::new();
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut rhs: Vec<f64> = Vec::new();
        let mut row_of =
            |k: usize, w: &Word, rows: &mut Vec<Vec<(usize, f64)>>, rhs: &mut Vec<f64>| -> usize {
                *row_index.entry((k, w.clone())).or_insert_with(|| {
                    rows.push(Vec::new());
                    rows.push(Vec::new());
                    rhs.push(0.0);
                    rhs.push(0.0);
                    rows.len() - 2
                })
            };
        for (c, (entries, _)) in cols.iter().enumerate() {
            for (k, p) in entries {
                for (w, v) in p.iter() {
                    let r = row_of(*k, w, &mut rows, &mut rhs);
                    rows[r].push((c, v.re));
                    rows[r + 1].push((c, v.im));
                }
            }
        }
        for (k, t) in targets.iter().enumerate() {
            for (w, v) in t.iter() {
                let r = row_of(k, w, &mut rows, &mut rhs);
                rhs[r] += v.re;
                rhs[r + 1] += v.im;
            }
        }
        let n = cols.len();
        // least squares by a backward-stable QR + truncated SVD (not the
        // normal equations; dependent link columns get the minimum-norm fit)
        let x = crate::linalg::dense::least_squares_rows(&rows, &rhs, n);
        // residuals and bound from the fitted multipliers
        let mut residuals = targets;
        for (c, (entries, _)) in cols.iter().enumerate() {
            if x[c] == 0.0 {
                continue;
            }
            for (k, p) in entries {
                residuals[*k] = residuals[*k].sub_ref(&p.scale(&C64::new(x[c], 0.0)));
            }
        }
        let residuals: Vec<Polynomial<C64>> = residuals.iter().map(|p| red.reduce(p)).collect();
        let bound: f64 = cols.iter().zip(&x).map(|((_, w), xc)| w * xc).sum();
        let max_residual = residuals
            .iter()
            .flat_map(|p| p.iter().map(|(_, c)| c.norm()))
            .fold(0.0_f64, f64::max);
        let residual_l1 = residuals
            .iter()
            .flat_map(|p| p.iter().map(|(_, c)| c.norm()))
            .sum();
        let pair = |c: usize| C64::new(x[c], x[c + 1]);
        MultiCertificate {
            functionals: certs,
            constants: x[..nk].to_vec(),
            link_multipliers: (0..self.links.len()).map(|l| pair(link_col + 2 * l)).collect(),
            constraint_multipliers: (0..nk)
                .map(|k| {
                    (0..self.constraints[k].len())
                        .map(|c| pair(constraint_col[k] + 2 * c))
                        .collect()
                })
                .collect(),
            bound,
            residuals,
            max_residual,
            residual_l1,
        }
    }
}
