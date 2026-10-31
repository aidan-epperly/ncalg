//! Noncommutative sum-of-squares (NC-SOS / NPA / Lasserre) relaxations over a
//! quotient algebra given by a (complete or partial) Gröbner basis, with
//! optional symmetry reduction.
//!
//! # Problem
//!
//! Minimize a Hermitian polynomial `h` over states of the algebra
//! `A = K<X>/I`, optionally subject to operator inequalities `g_k ⪰ 0`.  The
//! moment relaxation of level `W` (a finite set of words, typically all normal
//! words of degree ≤ d) is
//!
//! ```text
//!   min  L(h)
//!   s.t. L : A → C linear,  L(1) = 1,  L(p*) = conj L(p),
//!        M[u, v]   = L(u* v)      ⪰ 0   (u, v ∈ W)          moment matrix
//!        M_k[u, v] = L(u* g_k v)  ⪰ 0   (u, v ∈ W_k)        localizing matrices
//! ```
//!
//! Every product is reduced to its normal form with the [`Reducer`], so `L`
//! is represented by its values on the normal words that occur ("moment
//! variables").  The Hermitian-symmetry equalities are eliminated exactly
//! (sparse elimination), `L(1) = 1` is folded into the constant matrix, and
//! complex coefficients are handled by the real embedding
//! `[[Re M, −Im M], [Im M, Re M]]`.  The result is an LMI
//!
//! ```text
//!   min  c·z + c0   s.t.   F_0 + Σ_j z_j F_j ⪰ 0     (one block per matrix)
//! ```
//!
//! exported in SDPA sparse format ([`MomentSdp::write_sdpa`]), whose conic
//! dual is the SOS / Gram-matrix form.  [`MomentSdp::moments`] recovers the
//! functional `L` from a solver's solution `z`.
//!
//! # Symmetry reduction
//!
//! [`Relaxation::symmetry`] takes generators of a finite group `G` of
//! permutations of the algebra generators that preserve the ideal, `h` and
//! the `g_k` (all of which is verified).  Two reductions follow:
//!
//! * **orbit reduction of the moments** (exact): the optimal `L` can be taken
//!   `G`-invariant, so `L(w) = L(NF(σ(w)))` is imposed for all `σ ∈ G`, which
//!   divides the number of free variables by roughly `|G|`;
//! * **block diagonalization** (numerical, [`Relaxation::block_diagonalize`]):
//!   `G` acts on `V = span(W)` by `ρ(σ) u = NF(σ(u))`, and an invariant `L`
//!   makes the moment matrix a `G`-invariant Hermitian form.  In a basis
//!   adapted to the isotypic decomposition `V = ⊕_λ V_λ^{m_λ}` such forms are
//!   block diagonal with one `m_λ × m_λ` block per irreducible `λ` repeated
//!   `d_λ` times, so a single copy of each block is exported.  The
//!   decomposition is computed numerically from the commutant (a random
//!   invariant Hermitian form is diagonalized and its eigenspaces are matched
//!   with a second one), in the spirit of Murota–Kanno–Kojima–Kojima.
//!
//! # Partial Gröbner bases
//!
//! Any [`Reducer`] works.  With a partial basis the normal forms are still
//! valid representatives (the reducing set generates the same ideal), so the
//! SDP is still a valid relaxation of the same problem, merely weaker: some
//! linear relations between moments that a complete basis would impose are
//! missing.
//!
//! ```
//! use ncalg::{relations, GroebnerBasis, GroebnerOptions, Polynomial, Complex, BigRational};
//! use ncalg::ncsos::Relaxation;
//! let (alg, rels) = relations::pauli::<Complex<BigRational>>(1);
//! let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
//! let h: Polynomial<Complex<BigRational>> = alg.parse("X0 + Z0").unwrap();
//! let sdp = Relaxation::level(gb.reducer(), 1).minimize(&h).build();
//! // one real-embedded 8x8 block (4 normal words of degree <= 1), 1 fixed + 3 free moments
//! assert_eq!(sdp.block_sizes(), &[8]);
//! assert_eq!(sdp.num_vars(), 3);
//! ```

use crate::algebra::FreeAlgebra;
use crate::coeff::Field;
use crate::dense::hermitian_eigen;
use crate::poly::Polynomial;
use crate::reduce::Reducer;
use crate::sdp::{embed, unembed_dual, Affine, Lmi, MatrixVar, SdpModel, VarId};
use crate::symmetry::action;
use crate::symmetry::isotypic::{
    group_irreps, isotypic_decomposition, monomial_isotypic_decomposition, orthonormality_error, Rep,
};
pub use crate::symmetry::perm::{
    pair_permutation, permutation_group, signed_permutation_group, SignedPermutation,
};
use crate::word::{Var, Word};
use faer::Mat;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

type C64 = Complex<f64>;

/// Coefficient fields whose elements can be embedded into the complex floats
/// for SDP export.
pub trait SdpScalar: Field {
    /// Are all values real (so the moment problem is real-symmetric)?
    const REAL: bool;
    /// `(re, im)` as floats.
    fn to_complex_f64(&self) -> (f64, f64);
}

impl SdpScalar for f64 {
    const REAL: bool = true;
    fn to_complex_f64(&self) -> (f64, f64) {
        (*self, 0.0)
    }
}
impl SdpScalar for BigRational {
    const REAL: bool = true;
    fn to_complex_f64(&self) -> (f64, f64) {
        (self.to_f64().unwrap_or(f64::NAN), 0.0)
    }
}
impl SdpScalar for Complex<f64> {
    const REAL: bool = false;
    fn to_complex_f64(&self) -> (f64, f64) {
        (self.re, self.im)
    }
}
impl SdpScalar for Complex<BigRational> {
    const REAL: bool = false;
    fn to_complex_f64(&self) -> (f64, f64) {
        (
            self.re.to_f64().unwrap_or(f64::NAN),
            self.im.to_f64().unwrap_or(f64::NAN),
        )
    }
}

use crate::linalg::sparse::{Csr, Inserted, SparseEchelon};
pub use crate::linalg::sparse::{SparseHerm, SparseSym};

/// Description of one exported block.
#[derive(Clone, Debug)]
pub struct BlockInfo {
    /// Which moment/localizing matrix it comes from (0 = moment matrix).
    pub source: usize,
    /// Hermitian size of the block.
    pub size: usize,
    /// Exported as a real symmetric block (no real embedding needed)?
    pub real: bool,
    /// For symmetry-reduced blocks: dimension of the irreducible representation
    /// (the block occurs this many times in the full matrix); 1 otherwise.
    /// So `tr F = Σ_b irrep_dimension_b · tr F_b` over the blocks of one
    /// source matrix.
    pub irrep_dimension: usize,
}

/// A moment relaxation ready for an SDP solver.
#[derive(Clone, Debug)]
pub struct MomentSdp {
    alg: FreeAlgebra,
    /// Index set of each source matrix (moment matrix first, then localizing).
    source_words: Vec<Vec<Word>>,
    blocks: Vec<BlockInfo>,
    /// Hermitian coefficient matrices per exported block (before embedding).
    herm_f0: Vec<SparseHerm>,
    herm_fj: Vec<Vec<SparseHerm>>, // [block][var]
    /// Moment words (normal words on which `L` is represented).
    moment_words: Vec<Word>,
    moment_index: HashMap<Vec<Var>, usize>,
    complex_moments: bool,
    /// `y = B z + y0` where `y` are the real coordinates of the moments
    /// (`a_k` or `(a_k, b_k)` pairs) and `z` the free variables.
    b_rows: Vec<Vec<(usize, f64)>>,
    y0: Vec<f64>,
    /// Coordinates that depended on a dropped variable (one that occurs in
    /// no matrix and not in the objective): the relaxation leaves them free,
    /// so they have no value and no affine expression.
    undetermined: Vec<bool>,
    objective: Vec<f64>,
    objective_constant: f64,
    minimize: bool,
    group_order: usize,
    /// The symmetry group (empty if none), for symmetrizing certificates.
    group: Vec<SignedPermutation>,
    /// For block-diagonalized blocks: the isotypic basis vectors (in the
    /// coordinates of the source matrix) that the block is expressed in.
    block_vectors: Vec<Option<Vec<Vec<C64>>>>,
    /// Per source matrix: the action of every group element on its index
    /// words when it is monomial modulo the ideal (`NF(g(u_i)) = c_i
    /// u_{π(i)}`, `(π, c)` per element), for averaging Gram matrices.
    index_action: Vec<Option<Vec<(Vec<usize>, Vec<C64>)>>>,
    /// Localizing polynomials `g_s` (index 0 is the moment matrix, `None`).
    localizers: Vec<Option<Polynomial<C64>>>,
    /// Per moment constraint: words it introduced that nothing else determines.
    free_constraint_words: Vec<Vec<Word>>,
}

/// Builder for a moment relaxation.
pub struct Relaxation<'a, C: SdpScalar> {
    red: &'a Reducer<C>,
    words: Vec<Word>,
    localizing: Vec<(Polynomial<C>, Vec<Word>)>,
    objective: Option<Polynomial<C>>,
    minimize: bool,
    symmetry: Vec<SignedPermutation>,
    block_diag: bool,
    split_components: bool,
    /// Linear constraints `L(p) = value` on the moments.
    constraints: Vec<(Polynomial<C>, C)>,
    /// Relative threshold below which a reduced constraint row counts as dependent.
    elimination_tolerance: f64,
    /// Replace every moment constraint by its average over the symmetry group.
    symmetrize_constraints: bool,
    /// Add the kernel equalities implied by eigenstate-type constraints.
    eigenstate_kernels: bool,
    /// [`prepare`](Self::try_prepare) output kept by
    /// [`constraint_coverage`](Self::constraint_coverage) for the next build,
    /// which takes it; cleared by every builder method.
    prepared: std::sync::Mutex<Option<Prepared>>,
}

impl<'a, C: SdpScalar> std::fmt::Debug for Relaxation<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Relaxation({} index words, {} localizing matrices, {} symmetry generators)",
            self.words.len(),
            self.localizing.len(),
            self.symmetry.len()
        )
    }
}

impl<'a, C: SdpScalar> Relaxation<'a, C> {
    /// Relaxation with an explicit index set for the moment matrix.
    pub fn new(red: &'a Reducer<C>, words: Vec<Word>) -> Self {
        for w in &words {
            red.algebra().check_word(w);
        }
        Relaxation {
            red,
            words,
            localizing: Vec::new(),
            objective: None,
            minimize: true,
            symmetry: Vec::new(),
            block_diag: true,
            split_components: true,
            constraints: Vec::new(),
            elimination_tolerance: 1e-9,
            symmetrize_constraints: false,
            eigenstate_kernels: false,
            prepared: std::sync::Mutex::new(None),
        }
    }

    /// The reducer the relaxation was built with.
    pub(crate) fn reducer(&self) -> &'a Reducer<C> {
        self.red
    }

    /// Level-`d` relaxation: the index set is all normal words of degree ≤ d.
    pub fn level(red: &'a Reducer<C>, d: usize) -> Self {
        Self::new(red, red.normal_words_up_to(d))
    }

    /// Add the constraint `g ⪰ 0` through the localizing matrix indexed by `words`.
    /// `g` must be Hermitian.
    pub fn localizing(mut self, g: &Polynomial<C>, words: Vec<Word>) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.red.algebra().check_same(g.algebra());
        assert!(
            self.red.reduce(&g.adjoint().sub_ref(g)).is_zero(),
            "localizing polynomial must be Hermitian modulo the ideal"
        );
        self.localizing.push((power_of_two_normalized(g), words));
        self
    }

    /// Localizing matrix for `g ⪰ 0` indexed by the normal words of degree ≤ d.
    pub fn localizing_level(self, g: &Polynomial<C>, d: usize) -> Self {
        let words = self.red.normal_words_up_to(d);
        self.localizing(g, words)
    }

    /// Minimize `L(h)` (`h` Hermitian).
    pub fn minimize(mut self, h: &Polynomial<C>) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.red.algebra().check_same(h.algebra());
        self.objective = Some(h.clone());
        self.minimize = true;
        self
    }

    /// Maximize `L(h)` (`h` Hermitian).
    pub fn maximize(mut self, h: &Polynomial<C>) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.red.algebra().check_same(h.algebra());
        self.objective = Some(h.clone());
        self.minimize = false;
        self
    }

    /// Exploit a finite group of generator permutations (given by generators
    /// of the group, `perm[v] = image of generator v`) that preserve the
    /// ideal, the objective and the localizing polynomials.
    pub fn symmetry(mut self, generators: Vec<Vec<Var>>) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.symmetry = generators
            .into_iter()
            .map(SignedPermutation::permutation)
            .collect();
        self
    }

    /// Like [`symmetry`](Self::symmetry) with signed permutations
    /// (`x_v ↦ ±x_{π(v)}`), e.g. the Majorana parity `γ_a ↦ −γ_a`.
    pub fn signed_symmetry(mut self, generators: Vec<SignedPermutation>) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.symmetry = generators;
        self
    }

    /// Average the moment constraints over the symmetry group instead of
    /// requiring each one to be invariant (default false).
    ///
    /// Every constraint `L(p) = v` is replaced by `L(avg_G p) = v` with
    /// `avg_G p = (1/|G|) Σ_g g(p)`; averages that reduce to zero are dropped
    /// (an error if `v ≠ 0`), and averaged constraints that agree up to a
    /// scalar multiple (with the value scaled alike) are kept once.  For an
    /// invariant functional `L(g(p)) = L(p)`, so this is equivalent to the
    /// original constraints **provided the constraint set is invariant**: for
    /// every constraint `(p, v)` and every symmetry generator `σ`, `(σ(p), v)`
    /// is also a constraint, up to a scalar multiple of both sides.  That is
    /// checked, and building fails with an error naming a constraint whose
    /// image is missing.  [`constraint_coverage`](Self::constraint_coverage)
    /// and `free_constraint_words` then refer to the averaged constraints,
    /// listed by [`effective_constraints`](Self::effective_constraints).
    /// No effect without a symmetry.
    pub fn symmetrize_constraints(mut self, on: bool) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.symmetrize_constraints = on;
        self
    }

    /// Exploit eigenstate-type constraints exactly (opt-in, default off).  A
    /// constraint `L(p) = c` such that `q = p − c` is Hermitian, satisfies
    /// `q² = κ q` modulo the ideal and lies in the span of the index words
    /// (e.g. `L(X0) = 1` for a Pauli `X0`, or a projector at 0 or 1: the
    /// state is an extreme eigenvector of `p`) forces `L(q* q) = κ L(q) = 0`,
    /// so every feasible moment matrix `M` has `M q = 0`.  The relaxation
    /// then has no strictly feasible point, which interior-point solvers
    /// handle badly.  This adds the exact equalities
    /// `L(u* q) = 0` for every index word `u`; the native solver's presolve
    /// then removes the now-exact common kernel of the moment matrix
    /// ([`crate::sdp::SdpModel::solve`]), leaving a well-posed problem with
    /// the same optimum.  With a symmetry the added equalities are averaged
    /// over the group (they come in orbits), as with
    /// [`symmetrize_constraints`](Self::symmetrize_constraints).
    pub fn eigenstate_reduction(mut self, on: bool) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.eigenstate_kernels = on;
        self
    }

    /// All elements of the symmetry group generated by the declared symmetry
    /// generators (empty without a symmetry), identity first.
    pub fn group_elements(&self) -> Vec<SignedPermutation> {
        if self.symmetry.is_empty() {
            Vec::new()
        } else {
            signed_permutation_group(&self.symmetry, self.red.algebra().num_vars(), 1 << 16)
        }
    }

    /// The moment constraints the relaxation imposes: the ones added with
    /// [`moment_equality`](Self::moment_equality), or with
    /// [`symmetrize_constraints`](Self::symmetrize_constraints) and a
    /// symmetry their deduplicated group averages (in normal form).  Fails
    /// if averaging is requested and the constraint set is not invariant, or
    /// a constraint averages to zero with a nonzero value.
    pub fn effective_constraints(&self) -> Result<Vec<(Polynomial<C>, C)>, String> {
        let group = self.group_elements();
        self.constraints_for_group(&group)
    }

    /// Exact kernel vectors of the moment matrix implied by eigenstate-type
    /// constraints ([`eigenstate_reduction`](Self::eigenstate_reduction)):
    /// for a constraint `L(p) = c` with `q = p − c` Hermitian, `q² = κ q`
    /// modulo the ideal and `q` in the span of the index words, the
    /// equalities `L(u* q) = 0` for every index word `u`.
    fn eigenstate_kernel_constraints(&self) -> Vec<(Polynomial<C>, C)> {
        if !self.eigenstate_kernels {
            return Vec::new();
        }
        let red = self.red;
        let alg = red.algebra();
        let index: HashSet<&Word> = self.words.iter().collect();
        let mut out = Vec::new();
        for (p, c) in &self.constraints {
            let q = red.reduce(&p.sub_ref(&alg.constant(c.clone())));
            let Some((w0, c0)) = q.iter().next().map(|(w, c)| (w.clone(), c.clone())) else {
                continue;
            };
            if !red.reduce(&q.adjoint().sub_ref(&q)).is_zero() || !q.words().all(|w| index.contains(w)) {
                continue;
            }
            let q2 = red.reduce(&q.mul_ref(&q));
            let kappa = q2.coeff_of(w0.as_slice()).div_ref(&c0);
            if !red.reduce(&q2.sub_ref(&q.scale(&kappa))).is_zero() {
                continue;
            }
            // L(q* q) = κ L(q) = 0 and M ⪰ 0 give M q = 0
            for u in &self.words {
                let row = red.reduce(&alg.monomial::<C>(u.clone()).adjoint().mul_ref(&q));
                if !row.is_zero() {
                    out.push((row, C::zero()));
                }
            }
        }
        out
    }

    /// The user's constraints plus the eigenstate kernel equalities.
    fn all_constraints(&self) -> Vec<(Polynomial<C>, C)> {
        let mut all = self.constraints.clone();
        all.extend(self.eigenstate_kernel_constraints());
        all
    }

    fn constraints_for_group(&self, group: &[SignedPermutation]) -> Result<Vec<(Polynomial<C>, C)>, String> {
        let constraints = self.all_constraints();
        // the kernel equalities come in orbits, not one by one invariant
        let symmetrize = self.symmetrize_constraints
            || (self.eigenstate_kernels && constraints.len() > self.constraints.len());
        if !symmetrize || group.is_empty() {
            return Ok(constraints);
        }
        let red = self.red;
        let alg = red.algebra();
        // the constraint set must be invariant (up to scalar multiples)
        let mut originals = ConstraintSet::default();
        let normal: Vec<(Polynomial<C>, C)> = constraints
            .iter()
            .map(|(p, v)| (red.reduce(p), v.clone()))
            .collect();
        for (p, v) in &normal {
            originals.insert(p, v);
        }
        for sigma in &self.symmetry {
            for (p, v) in &normal {
                let image = action::image(sigma, p, red);
                if !originals.contains(&image, v) {
                    return Err(format!(
                        "symmetrize_constraints: the constraint set is not invariant under the symmetry generator {:?}: the image of L({}) = {} is not a constraint (averaging would change the problem; add the missing images or average by hand)",
                        sigma, p, v
                    ));
                }
            }
        }
        // average, drop zeros, deduplicate up to scalar multiples
        let averaged: Vec<(Polynomial<C>, C)> = {
            use rayon::prelude::*;
            normal
                .par_iter()
                .map(|(p, v)| (red.reduce(&action::average(group, p, alg)), v.clone()))
                .collect()
        };
        let mut kept = ConstraintSet::default();
        let mut out = Vec::new();
        for ((avg, v), (p, _)) in averaged.into_iter().zip(&normal) {
            if avg.is_zero() {
                if v.magnitude() > 1e-12 {
                    return Err(format!(
                        "symmetrize_constraints: L({}) = {} averages to zero over the symmetry group, so no invariant functional satisfies it",
                        p, v
                    ));
                }
                continue;
            }
            if kept.insert(&avg, &v) {
                out.push((avg, v));
            }
        }
        Ok(out)
    }

    /// Whether to split every matrix into the connected components of its
    /// sparsity pattern (default true): index words `u`, `v` are linked when
    /// the entry `L(u* g v)` is not identically zero after the exact
    /// elimination, and each component becomes its own block.  Exact (only
    /// entries that vanish for every feasible functional are dropped), so it
    /// changes the block structure and nothing else.  Catches splittings that
    /// are not a declared symmetry: orthogonal idempotents (several
    /// functionals in one), superselection rules, index sets whose cross
    /// products vanish.  With a symmetry, components that the group maps into
    /// each other are kept together and block-diagonalized jointly.
    /// Identically zero components (`0 ⪰ 0`) are dropped.
    pub fn split_components(mut self, on: bool) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.split_components = on;
        self
    }

    /// Whether to block-diagonalize the matrices with the symmetry
    /// (default true when a symmetry is given; orbit reduction of the
    /// variables is always applied).
    ///
    /// When the symmetry acts on the index words by signed permutations
    /// (a unitary monomial representation), each reduced block `F_b = V^† F V`
    /// uses an orthonormal basis `V`, so the full matrix is unitarily
    /// equivalent to `⊕_b I_{d_b} ⊗ F_b` (`d_b` =
    /// [`BlockInfo::irrep_dimension`]).  Constraints added on top of the
    /// reduced blocks (`I − F_b ⪰ 0`, `Y_b − F_b ⪰ 0` with
    /// `Σ_b d_b tr Y_b = 1`, norms, traces) then mean the same as on the
    /// full matrix.  When the action is not monomial (normal forms of the
    /// permuted words are not single words), the group acts by congruence:
    /// the blocks are only congruent to the restriction, and only
    /// congruence-invariant constraints such as `F ⪰ 0` carry over.  See
    /// [`MomentSdp::block_vectors`].
    ///
    /// Only the block-diagonalizing build checks that the index set is
    /// closed under the group (it panics otherwise).  Without it the
    /// relaxation is still restricted to invariant functionals, which is a
    /// valid bound for an invariant problem, but if the index set is not
    /// closed the bound can differ from the unsymmetrized relaxation at the
    /// same index set.
    pub fn block_diagonalize(mut self, on: bool) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.block_diag = on;
        self
    }

    /// Relative threshold for the exact elimination of the moment constraints
    /// in [`build`](Self::build): a constraint row whose entries all fall below
    /// `tol` times its original size after reduction against the previous
    /// rows is treated as dependent (default `1e-9`).  Too small a value lets
    /// round-off from thousands of redundant rows masquerade as independent
    /// constraints (the elimination then declares the system inconsistent);
    /// too large a value drops genuine constraints, which only weakens the
    /// relaxation.  Irrelevant for [`build_moment_form`](Self::build_moment_form).
    pub fn elimination_tolerance(mut self, tol: f64) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.elimination_tolerance = tol;
        self
    }

    /// The elimination tolerance in effect: the environment variable
    /// `NCALG_ELIMINATION_TOL`, when set, overrides the configured value (for
    /// tuning programs built on ncalg without recompiling them).
    fn effective_elimination_tolerance(&self) -> f64 {
        std::env::var("NCALG_ELIMINATION_TOL")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(self.elimination_tolerance)
    }

    /// Impose the linear constraint `L(p) = value` on the moments (e.g. a
    /// fixed expectation value, a symmetry sector or a normalization other
    /// than `L(1) = 1`).  With a symmetry group the constraint must be
    /// invariant.  In the eliminated form ([`build`](Self::build)) the
    /// constraint is solved out exactly; in the moment form it is an explicit
    /// equality.
    pub fn moment_equality(mut self, p: &Polynomial<C>, value: C) -> Self {
        *self.prepared.get_mut().expect("lock") = None;
        self.red.algebra().check_same(p.algebra());
        self.constraints.push((p.clone(), value));
        self
    }

    /// Impose `L(p) ≥ lower` for a Hermitian `p` (a `1 × 1` localizing block).
    pub fn moment_inequality(self, p: &Polynomial<C>, lower: C) -> Self {
        let g = p.sub_ref(&p.algebra().constant(lower));
        self.localizing(&g, vec![Word::empty()])
    }

    /// Assemble the SDP.
    /// Shared preparation: moment words, matrix entries in moment
    /// coordinates, the adjoint map and the group action.
    fn prepare(&self) -> Prepared {
        self.try_prepare().unwrap_or_else(|e| panic!("{}", e))
    }

    fn try_prepare(&self) -> Result<Prepared, String> {
        if let Some(pre) = self.prepared.lock().expect("lock").take() {
            return Ok(pre);
        }
        let mut timer = PhaseTimer::new();
        let red = self.red;
        let alg = red.algebra().clone();
        let real = C::REAL;

        // --- symmetry group and invariance checks -----------------------------
        let group = self.group_elements();
        let constraints = self.constraints_for_group(&group)?;
        timer.mark("group + constraints_for_group");
        if !group.is_empty() {
            use rayon::prelude::*;
            for sigma in &self.symmetry {
                // a *-automorphism: σ(x*) = σ(x)*, i.e. π∘adj = adj∘π and
                // equal signs on x and x*.  Otherwise L∘σ need not be a
                // state, and restricting to invariant functionals gave a
                // wrong bound (a ↦ a, a* ↦ −a*: 0 instead of −2)
                let adj = alg.adjoint_map();
                for v in 0..adj.len() {
                    let a = adj[v] as usize;
                    assert!(
                        sigma.perm[a] == adj[sigma.perm[v] as usize] && sigma.sign[a] == sigma.sign[v],
                        "symmetry generator {:?} does not commute with the adjoint (generator {}):                          it must map x* to σ(x)*",
                        sigma,
                        alg.format_word(&[v as Var])
                    );
                }
                // one reduction per basis element: in parallel (5.6 s on one
                // core for the 28 270 elements of the 11-qubit swap basis)
                if let Some(g) = red
                    .basis()
                    .par_iter()
                    .find_first(|g| !action::image(sigma, g, red).is_zero_up_to_rounding(g.max_magnitude()))
                {
                    panic!(
                        "symmetry generator {:?} does not preserve the ideal (image of {} is not reduced to zero)",
                        sigma, g
                    );
                }
                if let Some(h) = &self.objective {
                    assert!(
                        action::preserves(sigma, h, red),
                        "symmetry generator {:?} does not preserve the objective",
                        sigma
                    );
                }
                for (g, _) in &self.localizing {
                    assert!(
                        action::preserves(sigma, g, red),
                        "symmetry generator {:?} does not preserve a localizing polynomial",
                        sigma
                    );
                }
                for (p, _) in &constraints {
                    assert!(
                        action::preserves(sigma, p, red),
                        "symmetry generator {:?} does not preserve a moment constraint",
                        sigma
                    );
                }
            }
        }

        timer.mark("invariance checks");
        // --- moment variables (discovered lazily) ----------------------------
        let mut moment_words: Vec<Word> = vec![Word::empty()];
        let mut moment_index: HashMap<Vec<Var>, usize> = HashMap::new();
        moment_index.insert(Vec::new(), 0);
        let index_of = |w: &Word, words: &mut Vec<Word>, idx: &mut HashMap<Vec<Var>, usize>| -> usize {
            if let Some(&k) = idx.get(w.as_slice()) {
                return k;
            }
            let k = words.len();
            words.push(w.clone());
            idx.insert(w.as_slice().to_vec(), k);
            k
        };
        let coords_of = |p: &Polynomial<C>,
                         words: &mut Vec<Word>,
                         idx: &mut HashMap<Vec<Var>, usize>|
         -> Vec<(usize, C64)> {
            p.iter()
                .map(|(w, c)| {
                    let (re, im) = c.to_complex_f64();
                    (index_of(w, words, idx), C64::new(re, im))
                })
                .collect()
        };

        // entries per source matrix: (a, b, Σ_k γ_k y_k) for a <= b
        let mut source_entries: Vec<Vec<(usize, usize, Vec<(usize, C64)>)>> = Vec::new();
        let mut source_words: Vec<Vec<Word>> = Vec::new();
        let mut sources: Vec<(Option<&Polynomial<C>>, &Vec<Word>)> = vec![(None, &self.words)];
        for (g, w) in &self.localizing {
            sources.push((Some(g), w));
        }
        for (g, words) in sources {
            let m = words.len();
            // normal forms row by row, in parallel (the reducer is shared
            // read-only); the moment indices are assigned sequentially below
            // so that the numbering does not depend on the thread schedule
            // With unique normal forms (complete basis) of a *-closed ideal and a
            // Hermitian g, NF(u_b* g u_a) = NF(u_a* g u_b)*: the Hermitian average
            // below is the entry itself, and the second normal form is skipped
            let star_exact = red.is_complete()
                && {
                    use rayon::prelude::*;
                    // one reduction per basis element, in parallel
                    red.basis().par_iter().all(|p| red.reduce(&p.adjoint()).is_zero())
                }
                && g.is_none_or(|g| red.reduce(&g.adjoint().sub_ref(g)).is_zero());
            let row = |a: usize| -> Vec<(usize, Polynomial<C>, Option<Polynomial<C>>)> {
                let ua = alg.adjoint_word(&words[a]);
                (a..m)
                    .map(|b| {
                        // M[a,b] = L(u_a* g u_b), Hermitian-averaged with conj L(u_b* g u_a)
                        // so that partial bases still give a Hermitian matrix
                        let p_ab = match g {
                            None => red.reduce(&alg.monomial(ua.concat(&words[b]))),
                            Some(g) => red.reduce(&g.sandwich(&ua, &words[b])),
                        };
                        let star = if a != b && !star_exact {
                            let ub = alg.adjoint_word(&words[b]);
                            let p_ba = match g {
                                None => red.reduce(&alg.monomial(ub.concat(&words[a]))),
                                Some(g) => red.reduce(&g.sandwich(&ub, &words[a])),
                            };
                            Some(red.reduce(&p_ba.adjoint()))
                        } else {
                            None
                        };
                        (b, p_ab, star)
                    })
                    .collect()
            };
            let rows: Vec<Vec<(usize, Polynomial<C>, Option<Polynomial<C>>)>> = {
                use rayon::prelude::*;
                (0..m).into_par_iter().map(row).collect()
            };
            // New moment words are numbered in order of first occurrence
            // (row by row, terms in order), found in parallel as the least
            // (row, position) per word; then the entries are converted in
            // parallel.  The sequential pass over all terms took 2.7 s of
            // the QMC 11-cycle's build (2.9 · 10⁷ terms).
            let entries: Vec<(usize, usize, Vec<(usize, C64)>)> = {
                use rayon::prelude::*;
                let first: HashMap<&Word, (usize, usize)> = rows
                    .par_iter()
                    .enumerate()
                    .fold(HashMap::new, |mut acc, (a, row)| {
                        let mut pos = 0usize;
                        for (_, p_ab, star) in row {
                            for (w, _) in p_ab.iter().chain(star.iter().flat_map(|s| s.iter())) {
                                acc.entry(w).or_insert((a, pos));
                                pos += 1;
                            }
                        }
                        acc
                    })
                    .reduce(HashMap::new, |mut x, y| {
                        for (w, k) in y {
                            x.entry(w).and_modify(|e| *e = (*e).min(k)).or_insert(k);
                        }
                        x
                    });
                let mut new: Vec<((usize, usize), &Word)> = first
                    .into_iter()
                    .filter(|(w, _)| !moment_index.contains_key(w.as_slice()))
                    .map(|(w, k)| (k, w))
                    .collect();
                new.sort_unstable_by_key(|(k, _)| *k);
                for (_, w) in new {
                    index_of(w, &mut moment_words, &mut moment_index);
                }
                let idx = &moment_index;
                let coords = |p: &Polynomial<C>| -> Vec<(usize, C64)> {
                    p.iter()
                        .map(|(w, c)| {
                            let (re, im) = c.to_complex_f64();
                            (idx[w.as_slice()], C64::new(re, im))
                        })
                        .collect()
                };
                rows.into_par_iter()
                    .enumerate()
                    .flat_map_iter(|(a, row)| {
                        row.into_iter().map(move |(b, p_ab, star)| {
                            let Some(star) = star else {
                                // a normal form has distinct words, so distinct
                                // indices: sorted, as the merge below would give,
                                // without a map per entry
                                let mut coeffs = coords(&p_ab);
                                coeffs.sort_unstable_by_key(|(k, _)| *k);
                                coeffs.retain(|(_, c)| c.norm() > 0.0);
                                return (a, b, coeffs);
                            };
                            let mut acc: BTreeMap<usize, C64> = BTreeMap::new();
                            for (k, c) in coords(&p_ab).into_iter().chain(coords(&star)) {
                                *acc.entry(k).or_insert(C64::new(0.0, 0.0)) += c;
                            }
                            for v in acc.values_mut() {
                                *v *= 0.5;
                            }
                            (a, b, acc.into_iter().filter(|(_, c)| c.norm() > 0.0).collect())
                        })
                    })
                    .collect()
            };
            source_entries.push(entries);
            source_words.push(words.clone());
        }
        // words discovered so far are determined by the matrices; anything the
        // objective or a constraint adds beyond this point is a free variable
        let determined = moment_words.len();
        let mut obj_coords: Vec<(usize, C64)> = Vec::new();
        if let Some(h) = &self.objective {
            let nf = red.reduce(h);
            obj_coords = coords_of(&nf, &mut moment_words, &mut moment_index);
        }
        let constraint_rows: Vec<(Vec<(usize, C64)>, C64)> = constraints
            .iter()
            .map(|(p, value)| {
                let nf = red.reduce(p);
                let (re, im) = value.to_complex_f64();
                (
                    coords_of(&nf, &mut moment_words, &mut moment_index),
                    C64::new(re, im),
                )
            })
            .collect();

        timer.mark("moment entries");
        // --- closure of the moment set under adjoint and group action ----------
        // P e_k = coordinates of NF(w_k*) ; rho_g e_k = coordinates of NF(g(w_k))
        let mut p_cols: Vec<Vec<(usize, C64)>> = Vec::new();
        // The normal forms are computed in parallel, a chunk of words at a
        // time; their coordinates are assigned sequentially in word order, so
        // newly discovered words get the same indices as in a serial pass.
        // Chunks bound the memory held by the image polynomials.
        const CHUNK: usize = 1 << 15;
        // Orbits: for a group element h with NF(h(r)) = a_h w (a single
        // term) and any g, NF(g(w)) = NF((g h)(r)) / a_h, since g preserves the
        // ideal and acts linearly.  So once a word r has been processed, the
        // group images of every word w reached from it as such a single term
        // are read off r's images through the multiplication table: one
        // reduction (the adjoint) per word instead of 1 + |G|.  Words are
        // visited and indexed in the same order as before, so the result is
        // unchanged.  Only the processed words keep their images: the
        // images of the others are derived when needed, instead of storing
        // |G| columns per moment word.
        // without a table (a group too large for one), every word is
        // processed: 1 + |G| reductions each instead of reading members off
        let (mul, group_inverse): (Option<Vec<Vec<usize>>>, Option<Vec<usize>>) = if group.is_empty() {
            (None, None)
        } else {
            match crate::symmetry::perm::multiplication_table(&group) {
                Some((m, i)) => (Some(m), Some(i)),
                None => (None, None),
            }
        };
        // member[w] = (r, h, a_h); images[w] = the |G| image columns of a processed word
        let mut member: Vec<Option<(usize, usize, C64)>> = Vec::new();
        let mut images: Vec<Option<Vec<Vec<(usize, C64)>>>> = Vec::new();
        let mut k = 0;
        while k < moment_words.len() {
            let end = moment_words.len().min(k + CHUNK);
            if member.len() < moment_words.len() {
                member.resize(moment_words.len(), None);
            }
            // the adjoints in parallel; the group images only for the words
            // that are not yet members when their turn comes (decided word by
            // word, so a chunk does not reduce |G| images of every word).
            // They are reduced in parallel for a batch of the next non-member
            // words (about BATCH_IMAGES reductions); images of batch words that
            // have become members by their turn are dropped
            let stars: Vec<Polynomial<C>> = {
                use rayon::prelude::*;
                let words = &moment_words;
                (k..end)
                    .into_par_iter()
                    .map(|i| red.reduce(&alg.monomial(alg.adjoint_word(&words[i]))))
                    .collect()
            };
            const BATCH_IMAGES: usize = 4096;
            let batch_words = (BATCH_IMAGES / group.len().max(1)).max(1);
            let mut prefetched: HashMap<usize, Vec<Polynomial<C>>> = HashMap::new();
            for (offset, star) in stars.into_iter().enumerate() {
                let w = k + offset;
                p_cols.push(coords_of(&star, &mut moment_words, &mut moment_index));
                if member.len() < moment_words.len() {
                    member.resize(moment_words.len(), None);
                }
                images.push(None);
                if group.is_empty() || (mul.is_some() && member[w].is_some()) {
                    prefetched.remove(&w);
                    continue;
                }
                if !prefetched.contains_key(&w) {
                    use rayon::prelude::*;
                    let batch: Vec<usize> = (w..end)
                        .filter(|&i| mul.is_none() || member[i].is_none())
                        .take(batch_words)
                        .collect();
                    let words = &moment_words;
                    let pairs: Vec<(usize, &SignedPermutation)> = batch
                        .iter()
                        .flat_map(|&i| group.iter().map(move |g| (i, g)))
                        .collect();
                    let flat: Vec<Polynomial<C>> = pairs
                        .par_iter()
                        .map(|&(i, g)| action::word_image(g, &words[i], red))
                        .collect();
                    let mut flat = flat.into_iter();
                    for &i in &batch {
                        prefetched.insert(i, flat.by_ref().take(group.len()).collect());
                    }
                }
                let imgs = prefetched.remove(&w).expect("prefetched");
                let mut cols = Vec::with_capacity(group.len());
                for (gi, image) in imgs.iter().enumerate() {
                    let col = coords_of(image, &mut moment_words, &mut moment_index);
                    if member.len() < moment_words.len() {
                        member.resize(moment_words.len(), None);
                    }
                    if let [(x, a)] = col.as_slice() {
                        if member[*x].is_none() && mul.is_some() {
                            member[*x] = Some((w, gi, *a));
                        }
                    }
                    cols.push(col);
                }
                images[w] = Some(cols);
            }
            k = end;
        }
        let n_m = moment_words.len();
        // the image column of word w under group element gi, and its scale
        let image_of = |gi: usize, w: usize| -> (&[(usize, C64)], C64) {
            match (&images[w], &mul, member[w]) {
                (Some(cols), _, _) => (cols[gi].as_slice(), C64::new(1.0, 0.0)),
                (None, Some(mul), Some((r, h, a_h))) => (
                    images[r].as_ref().expect("processed")[mul[gi][h]].as_slice(),
                    C64::new(1.0, 0.0) / a_h,
                ),
                _ => unreachable!("every word is processed or a member"),
            }
        };
        // components of the words under single-term adjoint and group images
        // (both invertible, so the relation is symmetric).  The images of a
        // member are scaled images of its processed word, whose single-term
        // images include the member itself, so the processed words' images
        // generate the same components.
        let roots: Vec<usize> = {
            let mut uf = UnionFind::new(n_m);
            for m in 0..n_m {
                if let [(x, _)] = p_cols[m].as_slice() {
                    uf.union(m, *x);
                }
                for col in images[m].iter().flatten() {
                    if let [(x, _)] = col.as_slice() {
                        uf.union(m, *x);
                    }
                }
            }
            (0..n_m).map(|m| uf.find(m)).collect()
        };
        // a moment is determined by the matrices if it is an entry word or, as
        // L(w*) = conj L(w) and L(g(w)) = ±L(w) for an invariant functional,
        // a single-term adjoint or group image of a determined moment (the
        // Hermitian assembly only produces NF(u_a* u_b) for a <= b, so the
        // words NF(u_b* u_a) are reached through the adjoint): the components
        // that contain an entry word
        let determined_set: Vec<bool> = {
            let mut hit = vec![false; n_m];
            for w in 0..determined {
                hit[roots[w]] = true;
            }
            (0..n_m).map(|m| hit[roots[m]]).collect()
        };
        let free_constraint_words: Vec<Vec<Word>> = constraint_rows
            .iter()
            .map(|(coeffs, _)| {
                coeffs
                    .iter()
                    .filter(|(k, _)| !determined_set[*k])
                    .map(|(k, _)| moment_words[*k].clone())
                    .collect()
            })
            .collect();
        let dim = if real { n_m } else { 2 * n_m };
        // the averaged coordinates of a component span those of its
        // lowest-index word (the averaging absorbs the group and the
        // adjoint), so the kernel basis needs averages for that word only
        let orbit_rep: Vec<usize> = {
            let mut least = vec![usize::MAX; n_m];
            for m in 0..n_m {
                least[roots[m]] = least[roots[m]].min(m);
            }
            (0..n_m).map(|m| least[roots[m]]).collect()
        };
        // rows: P_{km} = coeff of w_k in NF(w_m*)  ->  p_rows[k] = [(m, P_{km})]
        let p_rows = Csr::build(n_m, |emit| {
            for (m, col) in p_cols.iter().enumerate() {
                for &(k, c) in col {
                    emit(k, (m, c));
                }
            }
        });
        // indices (into `group`) of the given generators, for equality constraints
        let generator_indices: Vec<usize> = self
            .symmetry
            .iter()
            .filter_map(|s| group.iter().position(|g| g == s))
            .collect();
        // rows of the group action, rho_rows[g][k] = [(m, ρ^g_{km})], kept only
        // where they are used: all words for the generators (invariance
        // equalities), the component representatives for every element
        // (orbit averages of the kernel basis)
        let mut rep_slot: Vec<usize> = vec![usize::MAX; n_m];
        let mut n_reps = 0;
        for m in 0..n_m {
            if orbit_rep[m] == m {
                rep_slot[m] = n_reps;
                n_reps += 1;
            }
        }
        let n_g = group.len();
        // With a monomial action (every image a single term) and a closed
        // group, the one entry of row k of ρ^g comes from the image of w_k
        // under g⁻¹: NF(g⁻¹(w_k)) = d w_m gives NF(g(w_m)) = w_k / d, so
        // ρ^g_{km} = 1/d.  Otherwise the rows are gathered from all
        // columns.
        let monomial = images
            .iter()
            .flatten()
            .all(|cols| cols.iter().all(|c| c.len() == 1));
        let inverse: Option<Vec<usize>> = if monomial { group_inverse.clone() } else { None };
        let row_entry = |inv: &[usize], gi: usize, k: usize| -> (usize, C64) {
            let (col, scale) = image_of(inv[gi], k);
            let (m, d) = col[0];
            (m, C64::new(1.0, 0.0) / (d * scale))
        };
        let gen_rho_rows: Vec<Csr<(usize, C64)>> = generator_indices
            .iter()
            .map(|&gi| {
                Csr::build(n_m, |emit| match &inverse {
                    Some(inv) => {
                        for k in 0..n_m {
                            emit(k, row_entry(inv, gi, k));
                        }
                    }
                    None => {
                        for m in 0..n_m {
                            let (col, scale) = image_of(gi, m);
                            for &(x, c) in col {
                                emit(x, (m, c * scale));
                            }
                        }
                    }
                })
            })
            .collect();
        // row rep_slot[k] * |G| + g
        let rep_rho_rows = Csr::build(n_reps * n_g, |emit| match &inverse {
            Some(inv) => {
                for k in (0..n_m).filter(|&k| rep_slot[k] != usize::MAX) {
                    for gi in 0..n_g {
                        emit(rep_slot[k] * n_g + gi, row_entry(inv, gi, k));
                    }
                }
            }
            None => {
                for gi in 0..n_g {
                    for m in 0..n_m {
                        let (col, scale) = image_of(gi, m);
                        for &(x, c) in col {
                            if rep_slot[x] != usize::MAX {
                                emit(rep_slot[x] * n_g + gi, (m, c * scale));
                            }
                        }
                    }
                }
            }
        });

        timer.mark("closure (adjoint, group images)");
        Ok(Prepared {
            real,
            group,
            generator_indices,
            moment_words,
            moment_index,
            source_words,
            source_entries,
            obj_coords,
            constraint_rows,
            free_constraint_words,
            p_rows,
            gen_rho_rows,
            rep_slot,
            rep_rho_rows,
            orbit_rep,
            dim,
        })
    }

    /// Assemble the SDP in *moment coordinates*: the variables are the real
    /// (and, for complex coefficients, imaginary) parts of `L(w)` on every
    /// moment word, and `L(1) = 1`, Hermiticity and symmetry invariance are
    /// explicit linear equalities instead of being eliminated.  No block
    /// diagonalization is applied (restrict the index set yourself to exploit
    /// a symmetry).  This mirrors the layout most hand-written moment SDPs
    /// use and is exported with [`MomentForm::write_json`].
    pub fn build_moment_form(&self) -> MomentForm {
        let pre = self.prepare();
        let real = pre.real;
        let dim = pre.dim;
        let coord = |k: usize, imag: bool| pre.coord(k, imag);
        // blocks: entry (a, b) = Σ_k γ_k y_k  ->  coefficient γ_k on a_k and iγ_k on b_k
        let mut blocks = Vec::new();
        for (s, entries) in pre.source_entries.iter().enumerate() {
            let size = pre.source_words[s].len();
            let mut fj: Vec<SparseHerm> = vec![Vec::new(); dim];
            for (a, b, coeffs) in entries {
                for (k, gamma) in coeffs {
                    fj[coord(*k, false)].push((*a, *b, *gamma));
                    if !real {
                        fj[coord(*k, true)].push((*a, *b, gamma * C64::new(0.0, 1.0)));
                    }
                }
            }
            blocks.push((size, fj));
        }
        // equalities
        let mut equalities: Vec<(Vec<(usize, f64)>, f64)> = Vec::new();
        let mut seen: HashSet<Vec<(usize, i64)>> = HashSet::new();
        let mut push_row = |row: BTreeMap<usize, f64>,
                            rhs: f64,
                            equalities: &mut Vec<(Vec<(usize, f64)>, f64)>| {
            // drop entries at rounding level relative to the row
            let rmax = row.values().fold(0.0_f64, |m, c| m.max(c.abs()));
            let row: Vec<(usize, f64)> = row.into_iter().filter(|(_, c)| c.abs() > 1e-12 * rmax).collect();
            if row.is_empty() {
                return;
            }
            // canonical key: scale so the first coefficient is +1, quantise
            let scale = 1.0 / row[0].1;
            let mut key: Vec<(usize, i64)> = row
                .iter()
                .map(|(i, c)| (*i, (c * scale * 1e9).round() as i64))
                .collect();
            key.push((usize::MAX, (rhs * scale * 1e9).round() as i64));
            if seen.insert(key) {
                equalities.push((row, rhs));
            }
        };
        let mut norm: BTreeMap<usize, f64> = BTreeMap::new();
        norm.insert(coord(0, false), 1.0);
        push_row(norm, 1.0, &mut equalities);
        if !real {
            let mut im0: BTreeMap<usize, f64> = BTreeMap::new();
            im0.insert(coord(0, true), 1.0);
            push_row(im0, 0.0, &mut equalities);
        }
        // user constraints on the moments
        for (row, rhs) in pre.constraint_real_rows() {
            if row.is_empty() {
                assert!(rhs.abs() < 1e-9, "inconsistent moment constraint (0 = {})", rhs);
                continue;
            }
            push_row(row, rhs, &mut equalities);
        }
        // Hermiticity: (S - I) y = 0, and invariance: (ρ_g^T - I) y = 0 for generators g
        let mut columns: Vec<BTreeMap<usize, f64>> = Vec::new();
        for i in 0..dim {
            let mut e: BTreeMap<usize, f64> = BTreeMap::new();
            e.insert(i, 1.0);
            let mut col = pre.apply_s(&e);
            *col.entry(i).or_insert(0.0) -= 1.0;
            columns.push(col);
        }
        for j in 0..pre.generator_indices.len() {
            for i in 0..dim {
                let mut e: BTreeMap<usize, f64> = BTreeMap::new();
                e.insert(i, 1.0);
                let mut col = pre.apply_rho_t(|k| pre.gen_rho_rows[j].row(k), &e);
                *col.entry(i).or_insert(0.0) -= 1.0;
                columns.push(col);
            }
        }
        // transpose the (constraint-row × variable) structure: each "column"
        // above is the image of e_i, so rows of the constraint matrix are
        // gathered per output coordinate
        let n_maps = columns.len() / dim;
        for map in 0..n_maps {
            let mut rows: BTreeMap<usize, BTreeMap<usize, f64>> = BTreeMap::new();
            for i in 0..dim {
                for (out, c) in &columns[map * dim + i] {
                    *rows.entry(*out).or_default().entry(i).or_insert(0.0) += c;
                }
            }
            for (_, row) in rows {
                push_row(row, 0.0, &mut equalities);
            }
        }
        // objective: Re Σ_k γ_k y_k
        let sign = if self.minimize { 1.0 } else { -1.0 };
        let mut objective = vec![0.0; dim];
        for (k, gamma) in &pre.obj_coords {
            objective[coord(*k, false)] += sign * gamma.re;
            if !real {
                objective[coord(*k, true)] += sign * (gamma * C64::new(0.0, 1.0)).re;
            }
        }
        MomentForm {
            moment_words: pre.moment_words.clone(),
            moment_index: pre.moment_index.clone(),
            free_constraint_words: pre.free_constraint_words.clone(),
            source_words: pre.source_words.clone(),
            complex: !real,
            blocks,
            equalities,
            objective,
            minimize: self.minimize,
        }
    }

    /// Assemble the SDP.  Panics where [`try_build`](Self::try_build) returns
    /// an error.
    pub fn build(&self) -> MomentSdp {
        self.try_build().unwrap_or_else(|e| panic!("{}", e))
    }

    /// Assemble the SDP, eliminating Hermiticity, symmetry and the user's
    /// moment constraints exactly.  Fails if the constraints are inconsistent
    /// with `L(1) = 1` (or so ill-conditioned that the elimination cannot
    /// tell); the moment form ([`build_moment_form`](Self::build_moment_form))
    /// keeps the equalities explicit and does not have this failure mode.
    pub fn try_build(&self) -> Result<MomentSdp, String> {
        let mut timer = PhaseTimer::new();
        let pre = self.try_prepare()?;
        timer.mark("prepare (total)");
        let red = self.red;
        let alg = red.algebra().clone();
        let real = pre.real;
        let group = &pre.group;
        let dim = pre.dim;
        let source_entries = &pre.source_entries;
        let source_words = pre.source_words.clone();
        let obj_coords = &pre.obj_coords;
        let moment_words = pre.moment_words.clone();
        let moment_index = pre.moment_index.clone();
        let coord = |k: usize, imag: bool| pre.coord(k, imag);
        let averaged = |idx: usize| pre.averaged(idx);

        // --- kernel basis: the independent orbit averages ---------------------------
        use rayon::prelude::*;
        // dense matrix entries the pivoted QR may use (1 GB)
        const DENSE_ENTRIES: usize = 1 << 27;
        let tol = 1e-10;
        let candidates: Vec<BTreeMap<usize, f64>> = (0..dim)
            .into_par_iter()
            .filter(|&idx| {
                // later words of a component only give dependent averages
                let word = if pre.real { idx } else { idx / 2 };
                pre.orbit_rep[word] == word
            })
            .map(averaged)
            .filter(|a| !a.is_empty())
            .collect();
        let entries: usize = candidates.iter().map(|c| c.len()).sum();
        let support = {
            let mut s: Vec<usize> = candidates.iter().flat_map(|c| c.keys().copied()).collect();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        let mut b_cols: Vec<Vec<(usize, f64)>> = Vec::new();
        if entries > support && support.saturating_mul(candidates.len()) <= DENSE_ENTRIES {
            // overlapping averages (a non-monomial action): the sparse
            // echelon form would fill its rows in to the whole support
            // (45 s of a 67 s build on the 11-cycle QMC relaxation), so
            // pick the independent subset by a dense pivoted QR instead
            let cols: Vec<Vec<(usize, f64)>> = candidates
                .iter()
                .map(|c| c.iter().map(|(i, v)| (*i, *v)).collect())
                .collect();
            for j in crate::linalg::dense::independent_columns(&cols, 1e-9) {
                b_cols.push(cols[j].clone());
            }
        } else {
            // disjoint supports (monomial actions) or too large for dense:
            // the sparse echelon form, in coordinate order
            let mut echelon = SparseEchelon::new(tol);
            for original in candidates {
                if let Inserted::Pivot(_) = echelon.insert(original.clone(), 0.0) {
                    b_cols.push(original.into_iter().collect());
                }
            }
        }
        timer.mark("kernel basis");
        // --- user constraints: intersect with their (homogenised) kernel -----------
        let user_rows = pre.constraint_real_rows();
        if !user_rows.is_empty() {
            // r·y = ρ  ->  r·y − ρ y_∅ = 0 (as y_∅ = 1), expressed in z: (R B) z = 0
            let n_rows = user_rows.len();
            // B by coordinate (CSR of Bᵀ): each row of R meets only the
            // columns sharing its coordinates, O(Σ_i nnz(B_i)) per row
            // instead of a scan of all of B.  Every product is summed
            // in increasing coordinate order, as before.
            let by_coord = Csr::build(dim, |emit| {
                for (j, col) in b_cols.iter().enumerate() {
                    for &(i, c) in col {
                        emit(i, (j, c));
                    }
                }
            });
            let c_rows: Vec<(BTreeMap<usize, f64>, f64)> = {
                use rayon::prelude::*;
                user_rows
                    .into_par_iter()
                    .map(|(row, rhs)| {
                        let mut r = row;
                        *r.entry(0).or_insert(0.0) -= rhs;
                        // (sum, sum of |terms|): an entry is dropped when it cancels to
                        // rounding level relative to its terms, not below an absolute 1e-12
                        let mut acc: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
                        for (i, ri) in &r {
                            for (j, c) in by_coord.row(*i) {
                                let e = acc.entry(*j).or_insert((0.0, 0.0));
                                e.0 += ri * c;
                                e.1 += (ri * c).abs();
                            }
                        }
                        let cz: BTreeMap<usize, f64> = acc
                            .into_iter()
                            .filter(|(_, (v, a))| v.abs() > 1e-12 * a)
                            .map(|(j, (v, _))| (j, v))
                            .collect();
                        (cz, 0.0)
                    })
                    .collect()
            };
            // NCALG_DUMP_CONSTRAINTS=<prefix>: write C (Matrix Market coordinate
            // format, 1-based) and the unit functional u_j = B[∅, j] as fixtures
            // for the elimination (see `crate::sparse_qr`)
            if let Ok(prefix) = std::env::var("NCALG_DUMP_CONSTRAINTS") {
                let nnz: usize = c_rows.iter().map(|(r, _)| r.len()).sum();
                let mut s = format!(
                    "%%MatrixMarket matrix coordinate real general\n% ncalg constraint matrix C = R B: rows = real constraint rows, columns = free coordinates z\n{} {} {}\n",
                    c_rows.len(),
                    b_cols.len(),
                    nnz
                );
                for (i, (row, _)) in c_rows.iter().enumerate() {
                    for (j, v) in row {
                        let _ = writeln!(s, "{} {} {:.17e}", i + 1, j + 1, v);
                    }
                }
                std::fs::write(format!("{}_C.mtx", prefix), s).expect("write constraint dump");
                let mut u = String::from(
                    "% unit functional u_j = coefficient of y_1 in kernel column j (1-based index, value)\n",
                );
                for (j, col) in b_cols.iter().enumerate() {
                    if let Some((_, v)) = col.iter().find(|(i, _)| *i == 0) {
                        let _ = writeln!(u, "{} {:.17e}", j + 1, v);
                    }
                }
                std::fs::write(format!("{}_u.txt", prefix), u).expect("write unit dump");
            }
            // near-kernel of C at the data's noise level by rank-revealing sparse QR
            // (crate::sparse_qr): the float constraint systems met in practice are
            // exactly inconsistent with L(1) = 1 but consistent to ~1e-10, and only
            // an orthogonal factorization decides rank reliably on them
            let n_z = b_cols.len();
            let matrix = crate::sparse_qr::SparseRows {
                ncols: n_z,
                rows: c_rows.into_iter().map(|(r, _)| r.into_iter().collect()).collect(),
            };
            let qr_opts = crate::sparse_qr::QrOptions {
                rank_tol: self.effective_elimination_tolerance(),
                ..Default::default()
            };
            let qr = crate::sparse_qr::SparseQr::factor(&matrix, &qr_opts)
                .map_err(|e| format!("moment constraints: {}", e))?;
            let check = qr.verify(&matrix, 200);
            if check.max_relative_residual.is_nan() || check.max_relative_residual > 100.0 * qr_opts.rank_tol
            {
                return Err(format!(
                    "moment constraints: the eliminated kernel violates the constraints (relative residual \
                     {:e} at rank tolerance {:e}; rank {} of {} rows, {} columns); use build_moment_form(), \
                     which keeps the equalities explicit",
                    check.max_relative_residual, qr_opts.rank_tol, check.rank, n_rows, n_z
                ));
            }
            let kernel = qr
                .kernel_basis(&qr_opts)
                .map_err(|e| format!("moment constraints: {}", e))?;
            // kernel vectors live in z-space: the new moment-space basis is y = Σ_j k_j b_j
            let mut new_cols: Vec<Vec<(usize, f64)>> = Vec::with_capacity(kernel.len());
            for k in &kernel {
                let mut col: BTreeMap<usize, f64> = BTreeMap::new();
                for (j, kj) in k {
                    for (i, v) in &b_cols[*j] {
                        *col.entry(*i).or_insert(0.0) += kj * v;
                    }
                }
                let cmax = col.values().fold(0.0_f64, |m, x| m.max(x.abs()));
                col.retain(|_, x| x.abs() > tol * cmax);
                new_cols.push(col.into_iter().collect());
            }
            if std::env::var("NCALG_DEBUG_ELIM").is_ok() {
                let one_component = |cols: &[Vec<(usize, f64)>]| -> f64 {
                    cols.iter()
                        .flat_map(|c| c.iter().filter(|(i, _)| *i == 0).map(|(_, v)| v.abs()))
                        .fold(0.0_f64, f64::max)
                };
                eprintln!(
                    "[ncalg elim] z before {} ; constraint rows {} ; rank {} ; z after {} ; max |y_1| before {:e} after {:e} ; R nnz {} ; dependent rows {} ; discarded norm {:e} ; max relative residual {:e}",
                    n_z,
                    n_rows,
                    qr.rank(),
                    new_cols.len(),
                    one_component(&b_cols),
                    one_component(&new_cols),
                    qr.r_nnz(),
                    qr.dependent_rows().len(),
                    qr.discarded_norm(),
                    check.max_relative_residual
                );
            }
            b_cols = new_cols;
        }
        // the constant coordinate a_∅ must appear in exactly one column
        let one_coord = 0usize;
        // pivot on the largest |b_j[∅]|: every other column is divided by it,
        // and after the QR kernel step the first nonzero one can be tiny
        let one_entry = |col: &Vec<(usize, f64)>| {
            col.iter()
                .find(|(i, _)| *i == one_coord)
                .map_or(0.0, |(_, c)| c.abs())
        };
        let col_max = |col: &Vec<(usize, f64)>| col.iter().fold(0.0_f64, |m, (_, c)| m.max(c.abs()));
        let one_col = (0..b_cols.len())
            .filter(|&j| one_entry(&b_cols[j]) > tol * col_max(&b_cols[j]))
            .max_by(|&a, &b| {
                one_entry(&b_cols[a])
                    .total_cmp(&one_entry(&b_cols[b]))
                    .then(b.cmp(&a))
            })
            .ok_or_else(|| {
                "the constant word vanishes on the feasible subspace: the moment constraints are \
                 inconsistent with L(1) = 1, or so ill-conditioned (coefficients spanning many orders \
                 of magnitude) that the exact elimination cannot decide; use build_moment_form(), \
                 which keeps the equalities explicit"
                    .to_string()
            })?;
        let one_val = b_cols[one_col].iter().find(|(i, _)| *i == one_coord).unwrap().1;
        let one_column = b_cols[one_col].clone();
        for (j, col) in b_cols.iter_mut().enumerate() {
            if j == one_col {
                continue;
            }
            if let Some(pos) = col.iter().position(|(i, _)| *i == one_coord) {
                let beta = col[pos].1;
                let mut map: BTreeMap<usize, f64> = col.iter().cloned().collect();
                for (i, c) in &one_column {
                    *map.entry(*i).or_insert(0.0) -= beta / one_val * c;
                }
                map.retain(|_, x| x.abs() > tol);
                *col = map.into_iter().collect();
            }
        }
        let z_one = 1.0 / one_val;
        let mut z_map: Vec<Option<usize>> = vec![None; b_cols.len()];
        let mut n_free = 0;
        for (j, slot) in z_map.iter_mut().enumerate() {
            if j != one_col {
                *slot = Some(n_free);
                n_free += 1;
            }
        }
        let mut b_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); dim];
        let mut y0 = vec![0.0; dim];
        for (j, col) in b_cols.iter().enumerate() {
            for (i, c) in col {
                match z_map[j] {
                    Some(zj) => b_rows[*i].push((zj, *c)),
                    None => y0[*i] += c * z_one,
                }
            }
        }

        timer.mark("user constraints");
        // --- Hermitian LMI in z per source matrix ----------------------------------
        // Σ_k γ_k y_k  ->  (constant, Σ_j z_j coeff_j) as complex numbers
        let expand = |coeffs: &[(usize, C64)]| -> (C64, BTreeMap<usize, C64>) {
            let mut c0 = C64::new(0.0, 0.0);
            let mut cz: BTreeMap<usize, C64> = BTreeMap::new();
            for (k, gamma) in coeffs {
                let ak = coord(*k, false);
                c0 += gamma * y0[ak];
                for (zj, c) in &b_rows[ak] {
                    *cz.entry(*zj).or_insert(C64::new(0.0, 0.0)) += gamma * c;
                }
                if !real {
                    let bk = coord(*k, true);
                    let i_gamma = gamma * C64::new(0.0, 1.0);
                    c0 += i_gamma * y0[bk];
                    for (zj, c) in &b_rows[bk] {
                        *cz.entry(*zj).or_insert(C64::new(0.0, 0.0)) += i_gamma * c;
                    }
                }
            }
            cz.retain(|_, x| x.norm() > 1e-15);
            (c0, cz)
        };
        let mut src_f0: Vec<SparseHerm> = Vec::new();
        let mut src_fj: Vec<Vec<SparseHerm>> = Vec::new();
        for entries in source_entries {
            let mut f0: SparseHerm = Vec::new();
            let mut fj: Vec<SparseHerm> = vec![Vec::new(); n_free];
            // expanded in parallel, scattered in entry order (the same
            // matrices as one sequential pass, which took 1.8 s of the QMC
            // 11-cycle's build for 7·10⁵ entries)
            let expanded: Vec<(C64, BTreeMap<usize, C64>)> = {
                use rayon::prelude::*;
                entries.par_iter().map(|(_, _, coeffs)| expand(coeffs)).collect()
            };
            for ((a, b, _), (c0, cz)) in entries.iter().zip(expanded) {
                if c0.norm() > 1e-15 {
                    f0.push((*a, *b, c0));
                }
                for (zj, c) in cz {
                    fj[zj].push((*a, *b, c));
                }
            }
            src_f0.push(f0);
            src_fj.push(fj);
        }
        let (obj_c0, obj_cz) = expand(obj_coords);
        let sign = if self.minimize { 1.0 } else { -1.0 };
        let mut objective = vec![0.0; n_free];
        for (zj, c) in obj_cz {
            objective[zj] = sign * c.re;
        }

        timer.mark("Hermitian LMI");
        // --- block diagonalization -------------------------------------------------
        let mut blocks: Vec<BlockInfo> = Vec::new();
        let mut block_vectors: Vec<Option<Vec<Vec<C64>>>> = Vec::new();
        let mut index_action: Vec<Option<Vec<(Vec<usize>, Vec<C64>)>>> = Vec::new();
        let mut herm_f0: Vec<SparseHerm> = Vec::new();
        let mut herm_fj: Vec<Vec<SparseHerm>> = Vec::new();
        // the group's multiplication table and irreducible representations,
        // once per build, for the exact orbit-wise block diagonalization
        let irreps: Option<Vec<crate::symmetry::isotypic::Irrep>> =
            if !group.is_empty() && self.block_diag && group.len() <= 1024 {
                crate::symmetry::perm::multiplication_table(group).and_then(|(mul, _)| group_irreps(&mul))
            } else {
                None
            };
        for (s, words) in source_words.iter().enumerate() {
            let m = words.len();
            let use_group = !group.is_empty() && self.block_diag && m > 1;
            // images of the index words under every group element, in local
            // coordinates (a signed permutation maps a normal word to a
            // combination of normal words)
            let images: Vec<Vec<Vec<(usize, C64)>>> = if use_group {
                let local: HashMap<Vec<Var>, usize> = words
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (w.as_slice().to_vec(), i))
                    .collect();
                use rayon::prelude::*;
                // |G| · m reductions, in parallel over the words
                group
                    .iter()
                    .map(|g| {
                        words
                            .par_iter()
                            .map(|w| {
                                let image = action::word_image(g, w, red);
                                image
                                    .iter()
                                    .map(|(w2, c)| {
                                        let a = *local.get(w2.as_slice()).unwrap_or_else(|| {
                                            panic!(
                                                "the index set is not closed under the symmetry: NF(σ({})) involves {}",
                                                alg.format_word(w),
                                                alg.format_word(w2)
                                            )
                                        });
                                        let (re, im) = c.to_complex_f64();
                                        (a, C64::new(re, im))
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            index_action.push(if images.is_empty() {
                None
            } else {
                images
                    .iter()
                    .map(|per_g| {
                        let mut perm = Vec::with_capacity(m);
                        let mut coef = Vec::with_capacity(m);
                        for img in per_g {
                            match img.as_slice() {
                                [(a, c)] => {
                                    perm.push(*a);
                                    coef.push(*c);
                                }
                                _ => return None,
                            }
                        }
                        Some((perm, coef))
                    })
                    .collect()
            });
            // connected components of the sparsity pattern (after elimination),
            // merged along the group action so that every part is invariant
            let components: Vec<Vec<usize>> = if self.split_components && m > 1 {
                let mut uf = UnionFind::new(m);
                for (a, b, _) in &src_f0[s] {
                    uf.union(*a, *b);
                }
                for f in &src_fj[s] {
                    for (a, b, _) in f {
                        uf.union(*a, *b);
                    }
                }
                for per_g in &images {
                    for (u, img) in per_g.iter().enumerate() {
                        for (a, _) in img {
                            uf.union(u, *a);
                        }
                    }
                }
                uf.components()
            } else {
                vec![(0..m).collect()]
            };
            let split = components.len() > 1;
            for comp in &components {
                let mc = comp.len();
                let mut pos = vec![usize::MAX; m];
                for (i, &a) in comp.iter().enumerate() {
                    pos[a] = i;
                }
                // coefficient matrices restricted to the component (local indices)
                let (f0, fj): (SparseHerm, Vec<SparseHerm>) = if split {
                    let restrict = |f: &SparseHerm| -> SparseHerm {
                        f.iter()
                            .filter(|(a, _, _)| pos[*a] != usize::MAX)
                            .map(|(a, b, v)| {
                                let (pa, pb) = (pos[*a], pos[*b]);
                                if pa <= pb {
                                    (pa, pb, *v)
                                } else {
                                    (pb, pa, v.conj())
                                }
                            })
                            .collect()
                    };
                    (restrict(&src_f0[s]), src_fj[s].iter().map(restrict).collect())
                } else {
                    (src_f0[s].clone(), src_fj[s].clone())
                };
                if split && f0.is_empty() && fj.iter().all(|f| f.is_empty()) {
                    continue; // identically zero: 0 ⪰ 0
                }
                // local vector -> source coordinates
                let lift = |v: &[C64]| -> Vec<C64> {
                    let mut out = vec![C64::new(0.0, 0.0); m];
                    for (i, &a) in comp.iter().enumerate() {
                        out[a] = v[i];
                    }
                    out
                };
                if !use_group || mc <= 1 {
                    blocks.push(BlockInfo {
                        source: s,
                        size: mc,
                        real,
                        irrep_dimension: 1,
                    });
                    block_vectors.push(if split {
                        Some(
                            (0..mc)
                                .map(|i| {
                                    let mut e = vec![C64::new(0.0, 0.0); mc];
                                    e[i] = C64::new(1.0, 0.0);
                                    lift(&e)
                                })
                                .collect(),
                        )
                    } else {
                        None
                    });
                    herm_f0.push(f0);
                    herm_fj.push(fj);
                    continue;
                }
                // the action on span(component): monomial forms when every image
                // is one word with a unit coefficient (signed permutations; no
                // dense m x m matrices), dense matrices otherwise
                let forms: Option<Vec<(Vec<usize>, Vec<C64>)>> = images
                    .iter()
                    .map(|per_g| {
                        let mut perm = Vec::with_capacity(mc);
                        let mut coef = Vec::with_capacity(mc);
                        let mut hit = vec![false; mc];
                        for &u in comp.iter() {
                            match per_g[u].as_slice() {
                                [(a, c)]
                                    if (c.norm() - 1.0).abs() <= 1e-12
                                        && pos[*a] != usize::MAX
                                        && !hit[pos[*a]] =>
                                {
                                    hit[pos[*a]] = true;
                                    perm.push(pos[*a]);
                                    coef.push(*c);
                                }
                                _ => return None,
                            }
                        }
                        Some((perm, coef))
                    })
                    .collect();
                let decomposition = match &forms {
                    // exact orbit-wise projection with the group's irreps,
                    // the general numerical method as a fallback
                    Some(f) => irreps
                        .as_ref()
                        .and_then(|ir| monomial_isotypic_decomposition(f, mc, ir))
                        .unwrap_or_else(|| isotypic_decomposition(Rep::Monomial(f), mc)),
                    None => {
                        // sparse columns: each word maps to a few words
                        let rhos: Vec<crate::symmetry::isotypic::SparseCols> = images
                            .iter()
                            .map(|per_g| {
                                comp.iter()
                                    .map(|&u| per_g[u].iter().map(|(a, c)| (pos[*a], *c)).collect())
                                    .collect()
                            })
                            .collect();
                        isotypic_decomposition(Rep::Sparse(&rhos), mc)
                    }
                };
                for part in decomposition {
                    // reduced coefficient matrices: F^λ[k][l] = v_k^† F v_l.  The
                    // basis vectors are sparse (orthonormalized columns of a
                    // projector, supported on single word orbits for signed
                    // permutations), so each entry F[a, b] only meets the
                    // vectors nonzero at a and at b
                    let mult = part.vectors.len();
                    let mut at: Vec<Vec<(usize, C64)>> = vec![Vec::new(); mc];
                    for (k, v) in part.vectors.iter().enumerate() {
                        for (a, c) in v.iter().enumerate() {
                            if c.re != 0.0 || c.im != 0.0 {
                                at[a].push((k, *c));
                            }
                        }
                    }
                    let reduce_sparse = |f: &SparseHerm| -> SparseHerm {
                        // (sum, sum of |terms|) per entry: dropped when it cancels to
                        // rounding level relative to its terms
                        let mut out: BTreeMap<(usize, usize), (C64, f64)> = BTreeMap::new();
                        for (a, b, val) in f {
                            for &(k, vka) in &at[*a] {
                                for &(l, vlb) in &at[*b] {
                                    if l >= k {
                                        let t = vka.conj() * val * vlb;
                                        let e = out.entry((k, l)).or_insert((C64::new(0.0, 0.0), 0.0));
                                        e.0 += t;
                                        e.1 += t.norm();
                                    }
                                }
                            }
                            if a != b {
                                for &(k, vkb) in &at[*b] {
                                    for &(l, vla) in &at[*a] {
                                        if l >= k {
                                            let t = vkb.conj() * val.conj() * vla;
                                            let e = out.entry((k, l)).or_insert((C64::new(0.0, 0.0), 0.0));
                                            e.0 += t;
                                            e.1 += t.norm();
                                        }
                                    }
                                }
                            }
                        }
                        debug_assert!(out.keys().all(|&(k, l)| k <= l && l < mult));
                        out.into_iter()
                            .filter(|(_, (v, a))| v.norm() > 1e-10 * a)
                            .map(|((k, l), (v, _))| (k, l, v))
                            .collect()
                    };
                    let bf0 = reduce_sparse(&f0);
                    let bfj: Vec<SparseHerm> = {
                        use rayon::prelude::*;
                        fj.par_iter().map(reduce_sparse).collect()
                    };
                    // real if every matrix is, each against its own size: a
                    // large constant F_0 hid imaginary coefficients up to
                    // 1e-9 of it in the F_j
                    let is_real = std::iter::once(&bf0).chain(bfj.iter()).all(|f| {
                        let scale = f.iter().fold(0.0_f64, |m, (_, _, c)| m.max(c.norm()));
                        f.iter().all(|(_, _, c)| c.im.abs() <= 1e-9 * scale)
                    });
                    blocks.push(BlockInfo {
                        source: s,
                        size: part.vectors.len(),
                        real: is_real,
                        irrep_dimension: part.irrep_dimension,
                    });
                    block_vectors.push(Some(if split {
                        part.vectors.iter().map(|v| lift(v)).collect()
                    } else {
                        part.vectors.clone()
                    }));
                    herm_f0.push(bf0);
                    herm_fj.push(bfj);
                }
            }
        }

        timer.mark("block diagonalization");
        // --- drop variables that occur nowhere -------------------------------------
        let mut used = vec![false; n_free];
        for fj in &herm_fj {
            for (j, f) in fj.iter().enumerate() {
                if !f.is_empty() {
                    used[j] = true;
                }
            }
        }
        for (j, c) in objective.iter().enumerate() {
            if *c != 0.0 {
                used[j] = true;
            }
        }
        let mut new_index: Vec<Option<usize>> = vec![None; n_free];
        let mut n_used = 0;
        for j in 0..n_free {
            if used[j] {
                new_index[j] = Some(n_used);
                n_used += 1;
            }
        }
        let mut undetermined = vec![false; b_rows.len()];
        if n_used < n_free {
            for (i, row) in b_rows.iter().enumerate() {
                undetermined[i] = row.iter().any(|(j, _)| new_index[*j].is_none());
            }
            for fj in &mut herm_fj {
                let mut compact: Vec<SparseHerm> = Vec::with_capacity(n_used);
                for (j, f) in fj.iter().enumerate() {
                    if new_index[j].is_some() {
                        compact.push(f.clone());
                    }
                }
                *fj = compact;
            }
            objective = objective
                .iter()
                .enumerate()
                .filter(|(j, _)| new_index[*j].is_some())
                .map(|(_, c)| *c)
                .collect();
            for row in &mut b_rows {
                *row = row
                    .iter()
                    .filter_map(|(j, c)| new_index[*j].map(|nj| (nj, *c)))
                    .collect();
            }
        }

        timer.mark("drop unused variables");
        Ok(MomentSdp {
            alg,
            source_words,
            blocks,
            herm_f0,
            herm_fj,
            moment_words,
            moment_index,
            complex_moments: !real,
            b_rows,
            y0,
            undetermined,
            objective,
            objective_constant: sign * obj_c0.re,
            minimize: self.minimize,
            group_order: group.len().max(1),
            group: group.clone(),
            block_vectors,
            index_action,
            localizers: self.localizers_f64(),
            free_constraint_words: pre.free_constraint_words.clone(),
        })
    }

    /// For every moment constraint, the words of its normal form that no
    /// entry of the moment or localizing matrices determines.  A constraint
    /// with such words only fixes those (otherwise free) variables and does
    /// not restrict the relaxation; drop or rewrite it.  Costs one assembly.
    /// With [`symmetrize_constraints`](Self::symmetrize_constraints) the list
    /// follows [`effective_constraints`](Self::effective_constraints).
    pub fn constraint_coverage(&self) -> Vec<Vec<Word>> {
        let pre = self.prepare();
        let free = pre.free_constraint_words.clone();
        // kept for the build that usually follows
        *self.prepared.lock().expect("lock") = Some(pre);
        free
    }

    /// Localizing polynomials as complex-float polynomials (`None` for the
    /// moment matrix itself).
    fn localizers_f64(&self) -> Vec<Option<Polynomial<C64>>> {
        let mut v = vec![None];
        for (g, _) in &self.localizing {
            v.push(Some(to_c64_poly(g)));
        }
        v
    }

    /// Hermitian polynomials `r_j` (normal forms) with `L(r_j) = 0`
    /// equivalent to the moment constraints ([`moment_equality`] and the
    /// eigenstate kernel equalities): for `q = p − v`, the Hermitian parts
    /// `H` and `−iA` of [`Polynomial::hermitian_split`] (the second only over a complex field; a
    /// real symmetric functional has `L(q) = L(q*)`), zero parts dropped.
    /// The Gram forms carry one real multiplier per `r_j`.
    ///
    /// [`moment_equality`]: Self::moment_equality
    fn multiplier_polynomials(&self) -> Vec<Polynomial<C64>> {
        self.hermitian_parts(self.all_constraints())
    }

    /// The Hermitian parts `r_j` of `p − v` for the given constraints (see
    /// [`multiplier_polynomials`](Self::multiplier_polynomials)).
    fn hermitian_parts(&self, constraints: Vec<(Polynomial<C>, C)>) -> Vec<Polynomial<C64>> {
        let red = self.red;
        let alg = red.algebra();
        let mut out = Vec::new();
        for (p, v) in constraints {
            let (h, a) = p.sub_ref(&alg.constant(v)).hermitian_split();
            let re = red.reduce(&h);
            if !re.is_zero() {
                out.push(to_c64_poly(&re));
            }
            if !C::REAL {
                let im = red.reduce(&a);
                if !im.is_zero() {
                    out.push(to_c64_poly(&im).map_coeffs(|c| c * C64::new(0.0, -1.0)));
                }
            }
        }
        out
    }

    /// Free multipliers `μ_j` for the [`multiplier_polynomials`], added to
    /// the Gram target as `Σ_j μ_j r_j`.
    ///
    /// [`multiplier_polynomials`]: Self::multiplier_polynomials
    fn add_multipliers(
        &self,
        model: &mut SdpModel,
        target: &mut BTreeMap<Vec<Var>, Affine>,
    ) -> (Vec<VarId>, Vec<Polynomial<C64>>) {
        let all = self.multiplier_polynomials();
        let cols: Vec<BTreeMap<Vec<Var>, C64>> = all
            .iter()
            .map(|r| r.iter().map(|(w, c)| (w.as_slice().to_vec(), *c)).collect())
            .collect();
        let keep = independent_columns(&cols);
        let polys: Vec<Polynomial<C64>> = all
            .into_iter()
            .zip(keep)
            .filter(|(_, k)| *k)
            .map(|(r, _)| r)
            .collect();
        let mut vars = Vec::with_capacity(polys.len());
        for (j, r) in polys.iter().enumerate() {
            let mu = model.free_var(format!("mu{}", j));
            for (w, c) in r.iter() {
                target
                    .entry(w.as_slice().to_vec())
                    .or_insert_with(Affine::zero)
                    .add_scaled(&Affine::var(mu), *c);
            }
            vars.push(mu);
        }
        (vars, polys)
    }

    /// Symmetry-reduced Gram form ([`sos_model`](Self::sos_model) for
    /// `shift = None`, [`gram_weight_model`](Self::gram_weight_model) otherwise), or
    /// `None` when there is no symmetry or it does not act monomially (then
    /// the callers build the full form).
    fn symmetric_gram_model(&self, shift: Option<f64>) -> Option<SosModel> {
        let group = self.group_elements();
        if group.is_empty() {
            return None;
        }
        let sdp = self.try_build().ok()?;
        // the reduction needs orthonormal block bases (unitary)
        for v in sdp.block_vectors().iter().flatten() {
            if orthonormality_error(v) > 1e-9 {
                return None;
            }
        }
        let red = self.red;
        let alg = red.algebra();
        // orbit data of a normal word: (representative, coefficient of the
        // representative in the group average of the word); None when the
        // action is not monomial
        let mut orbits = action::WordOrbits::new(&group, red);
        let mut class_of = |w: &Word| -> Option<(Vec<Var>, C64)> {
            let (rep, a) = orbits.class(w)?;
            let (re, im) = a.to_complex_f64();
            Some((rep, C64::new(re, im)))
        };
        let alive = |a: C64| a.norm() > 1e-12;

        let mut model = SdpModel::new();
        let source_words = self.source_words();
        let localizers: Vec<Option<&Polynomial<C>>> = std::iter::once(None)
            .chain(self.localizing.iter().map(|(g, _)| Some(g)))
            .collect();
        // class-coefficient vector of NF(u_a* g_s u_c), cached per source pair,
        // with the classes numbered (no word per term in the accumulation)
        let mut pair_cache: HashMap<(usize, usize, usize), Vec<(usize, C64)>> = HashMap::new();
        let mut class_id: HashMap<Vec<Var>, usize> = HashMap::new();
        let mut class_rep: Vec<Vec<Var>> = Vec::new();
        let mut expansion: BTreeMap<Vec<Var>, Affine> = BTreeMap::new();
        let mut grams = Vec::new();
        let mut layout_sources = Vec::new();
        let mut layout_dims = Vec::new();
        let mut layout_vectors = Vec::new();
        for (b, info) in sdp.blocks().iter().enumerate() {
            let s = info.source;
            let words = &source_words[s];
            let vectors = sdp.block_vectors()[b].clone();
            let d = info.irrep_dimension as f64;
            // sparse basis vectors (identity for a whole source matrix)
            let sparse: Vec<Vec<(usize, C64)>> = match &vectors {
                None => (0..words.len()).map(|a| vec![(a, C64::new(1.0, 0.0))]).collect(),
                Some(vs) => vs
                    .iter()
                    .map(|v| {
                        v.iter()
                            .enumerate()
                            .filter(|(_, c)| c.norm() != 0.0)
                            .map(|(a, c)| (a, *c))
                            .collect()
                    })
                    .collect(),
            };
            // complex basis vectors (the ω, ω̄ parts of a cyclic group of
            // order ≥ 3) need a Hermitian Gram block also over a real field;
            // forcing it real gives a wrong (far too low) bound
            let real = info.real && sparse.iter().flatten().all(|(_, c)| c.im == 0.0);
            let gram = model.psd_var(info.size, real, format!("G{}_{}", s, b));
            let mut at: BTreeMap<usize, Vec<(usize, C64)>> = BTreeMap::new();
            for (k, v) in sparse.iter().enumerate() {
                for &(a, c) in v {
                    at.entry(a).or_default().push((k, c));
                }
            }
            // coefficient of G_b[k, l] in each class, and the summands' size
            let mut acc: HashMap<(usize, usize, usize), (C64, f64)> = HashMap::new();
            for (&a, ka) in &at {
                for (&c, lc) in &at {
                    if let std::collections::hash_map::Entry::Vacant(slot) = pair_cache.entry((s, a, c)) {
                        let ua = alg.adjoint_word(&words[a]);
                        let nf: Polynomial<C> = match localizers[s] {
                            None => red.reduce(&alg.monomial(ua.concat(&words[c]))),
                            Some(g) => red.reduce(&g.sandwich(&ua, &words[c])),
                        };
                        let mut t = Vec::new();
                        for (w, coef) in nf.iter() {
                            let (rep, aw) = class_of(w)?;
                            if alive(aw) {
                                let id = *class_id.entry(rep).or_insert_with_key(|rep| {
                                    class_rep.push(rep.clone());
                                    class_rep.len() - 1
                                });
                                let (re, im) = coef.to_complex_f64();
                                t.push((id, C64::new(re, im) * aw));
                            }
                        }
                        slot.insert(t);
                    }
                    let t = &pair_cache[&(s, a, c)];
                    for &(k, vka) in ka {
                        for &(l, vlc) in lc {
                            let f = vka.conj() * vlc * d;
                            for &(id, tc) in t {
                                let e = acc.entry((id, k, l)).or_insert((C64::new(0.0, 0.0), 0.0));
                                e.0 += f * tc;
                                e.1 += (f * tc).norm();
                            }
                        }
                    }
                }
            }
            for ((id, k, l), (mut c, size)) in acc {
                // products of float basis entries leave round-off where the
                // exact real or imaginary part is zero (e.g. Im of a
                // coefficient of a Hermitian word); kept, it would become a
                // spurious near-zero equality row.  Relative to the summands:
                // an absolute 1e-12 also zeroed genuine coefficients of a
                // localizer with coefficients ~1e-8
                if c.re.abs() <= 1e-12 * size {
                    c.re = 0.0;
                }
                if c.im.abs() <= 1e-12 * size {
                    c.im = 0.0;
                }
                if c.norm() > 0.0 {
                    expansion
                        .entry(class_rep[id].clone())
                        .or_insert_with(Affine::zero)
                        .add_scaled(&gram.entry(k, l), c);
                }
            }
            grams.push(gram);
            layout_sources.push(s);
            layout_dims.push(info.irrep_dimension);
            layout_vectors.push(vectors);
        }
        // target and multipliers, mapped to classes
        let bound = if shift.is_none() {
            Some(model.free_var("c"))
        } else {
            None
        };
        let word_target = self.sos_target(bound, shift.unwrap_or(0.0));
        let mut target: BTreeMap<Vec<Var>, Affine> = BTreeMap::new();
        for (w, e) in &word_target {
            let (rep, aw) = class_of(&Word::from(w.clone()))?;
            if alive(aw) {
                target.entry(rep).or_insert_with(Affine::zero).add_scaled(e, aw);
            }
        }
        let constraints = self.constraints_for_group(&group).ok()?;
        let mut multipliers = Vec::new();
        let mut multiplier_polynomials = Vec::new();
        let mut candidates = Vec::new();
        for r in self.hermitian_parts(constraints) {
            let mut t: BTreeMap<Vec<Var>, C64> = BTreeMap::new();
            for (w, c) in r.iter() {
                let (rep, aw) = class_of(w)?;
                if alive(aw) {
                    *t.entry(rep).or_insert(C64::new(0.0, 0.0)) += *c * aw;
                }
            }
            // relative to the constraint's size
            let size = r.iter().map(|(_, c)| c.norm()).fold(0.0_f64, f64::max);
            t.retain(|_, c| c.norm() > 1e-12 * size);
            candidates.push((r, t)); // empty: averages to zero, dropped below
        }
        let cols: Vec<BTreeMap<Vec<Var>, C64>> = candidates.iter().map(|(_, t)| t.clone()).collect();
        let keep = independent_columns(&cols);
        for ((r, t), k) in candidates.into_iter().zip(keep) {
            if !k {
                continue;
            }
            let mu = model.free_var(format!("mu{}", multipliers.len()));
            for (rep, c) in t {
                target
                    .entry(rep)
                    .or_insert_with(Affine::zero)
                    .add_scaled(&Affine::var(mu), c);
            }
            multipliers.push(mu);
            multiplier_polynomials.push(r);
        }
        // the (k, l) and (l, k) terms of a Hermitian block meet on the same
        // real variables, so round-off can survive the per-entry cleaning
        // above; drop real and imaginary parts at round-off level relative
        // to the largest coefficient of each identity
        for e in expansion.values_mut() {
            let scale = e.terms.values().map(|c| c.norm()).fold(0.0, f64::max);
            for c in e.terms.values_mut() {
                if c.re.abs() <= 1e-12 * scale {
                    c.re = 0.0;
                }
                if c.im.abs() <= 1e-12 * scale {
                    c.im = 0.0;
                }
            }
            e.terms.retain(|_, c| c.norm() > 0.0);
        }
        Self::equate_words(&mut model, &expansion, &target);
        // objective (and the diagonal weights)
        let mut weights = Vec::new();
        match shift {
            None => {
                let c = Affine::var(bound.unwrap());
                if self.minimize {
                    model.maximize(&c);
                } else {
                    model.minimize(&c);
                }
            }
            Some(_) => {
                // one weight per orbit of index words of the moment matrix
                let words = &source_words[0];
                let mut orbit_var: BTreeMap<Vec<Var>, VarId> = BTreeMap::new();
                let mut covered: HashSet<Vec<Var>> = HashSet::new();
                let mut reps = Vec::with_capacity(words.len());
                for w in words {
                    let (rep, _) = class_of(w)?;
                    let n = orbit_var.len();
                    let var = *orbit_var
                        .entry(rep.clone())
                        .or_insert_with(|| model.free_var(format!("w{}", n)));
                    weights.push(var);
                    reps.push(rep);
                }
                for (b, s) in layout_sources.iter().enumerate() {
                    if *s != 0 {
                        continue;
                    }
                    let size = grams[b].size();
                    let sparse: Vec<Vec<(usize, C64)>> = match &layout_vectors[b] {
                        None => (0..words.len()).map(|a| vec![(a, C64::new(1.0, 0.0))]).collect(),
                        Some(vs) => vs
                            .iter()
                            .map(|v| {
                                v.iter()
                                    .enumerate()
                                    .filter(|(_, c)| c.norm() != 0.0)
                                    .map(|(a, c)| (a, *c))
                                    .collect()
                            })
                            .collect(),
                    };
                    for v in &sparse {
                        for &(a, _) in v {
                            covered.insert(reps[a].clone());
                        }
                    }
                    // conj(V^† diag(w) V)[k, l] = Σ_a v_k[a] w_a conj(v_l[a]).  Each
                    // orthonormal basis vector lies in one orbit of index words
                    // (Gram–Schmidt only mixes columns within an orbit), so with
                    // orbit-constant w this is real diagonal, w_{o(k)} δ_kl.
                    let mut entries = Vec::with_capacity(size * (size + 1) / 2);
                    for k in 0..size {
                        for l in k..size {
                            let mut e = -grams[b].entry(k, l);
                            let mut i = 0;
                            let (vk, vl) = (&sparse[k], &sparse[l]);
                            for &(a, ca) in vk {
                                while i < vl.len() && vl[i].0 < a {
                                    i += 1;
                                }
                                if i < vl.len() && vl[i].0 == a {
                                    e.add_scaled(&Affine::var(weights[a]), ca * vl[i].1.conj());
                                }
                            }
                            entries.push((k, l, e));
                        }
                    }
                    model.psd_affine(size, grams[b].is_real(), entries, format!("W - G{}", b));
                }
                // index words in no block (identically zero components): w = 0
                for (rep, var) in &orbit_var {
                    if !covered.contains(rep) {
                        model.equal_zero(&Affine::var(*var));
                    }
                }
                let mut total = Affine::zero();
                for w in &weights {
                    *total.terms.entry(*w).or_insert(C64::new(0.0, 0.0)) += C64::new(1.0, 0.0);
                }
                model.minimize(&total);
            }
        }
        Some(SosModel {
            model,
            grams,
            gram_sources: layout_sources,
            gram_irrep_dimensions: layout_dims,
            gram_vectors: layout_vectors,
            group,
            index_action: sdp.index_action.clone(),
            source_words,
            bound,
            weights,
            shift: shift.unwrap_or(0.0),
            multipliers,
            multiplier_polynomials,
            minimize: self.minimize,
            alg: alg.clone(),
            localizers: self.localizers_f64(),
        })
    }
}

/// Disjoint sets over `0..n` (path halving, union by size).
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }

    /// The sets, each sorted, ordered by their smallest element.
    fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut slot: Vec<usize> = vec![usize::MAX; n];
        let mut out: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            let r = self.find(i);
            if slot[r] == usize::MAX {
                slot[r] = out.len();
                out.push(Vec::new());
            }
            out[slot[r]].push(i);
        }
        out
    }
}

/// Moment constraints `(p, v)` in normal form, compared up to a common
/// scalar multiple of `p` and `v` with a relative float tolerance (`1e-9`).
#[derive(Default)]
struct ConstraintSet {
    /// support (words of `p`) -> normalized (coefficients, value) of the members
    by_support: HashMap<Vec<Word>, Vec<(Vec<C64>, C64)>>,
}

impl ConstraintSet {
    fn normalize<C: SdpScalar>(p: &Polynomial<C>, v: &C) -> (Vec<Word>, Vec<C64>, C64) {
        let to = |c: &C| {
            let (re, im) = c.to_complex_f64();
            C64::new(re, im)
        };
        let words: Vec<Word> = p.iter().map(|(w, _)| w.clone()).collect();
        let coeffs: Vec<C64> = p.iter().map(|(_, c)| to(c)).collect();
        // divide by the largest coefficient (the first one of maximal modulus)
        let pivot = coeffs.iter().copied().fold(C64::new(0.0, 0.0), |a, c| {
            if c.norm() > a.norm() * (1.0 + 1e-9) {
                c
            } else {
                a
            }
        });
        let inv = if pivot.norm() > 0.0 {
            1.0 / pivot
        } else {
            C64::new(1.0, 0.0)
        };
        (words, coeffs.iter().map(|c| c * inv).collect(), to(v) * inv)
    }

    fn matches(a: &(Vec<C64>, C64), b: &(Vec<C64>, C64)) -> bool {
        let close = |x: C64, y: C64| (x - y).norm() <= 1e-9 * (1.0 + x.norm().max(y.norm()));
        a.0.len() == b.0.len() && a.0.iter().zip(&b.0).all(|(x, y)| close(*x, *y)) && close(a.1, b.1)
    }

    fn contains<C: SdpScalar>(&self, p: &Polynomial<C>, v: &C) -> bool {
        let (words, coeffs, value) = Self::normalize(p, v);
        let key = (coeffs, value);
        self.by_support
            .get(&words)
            .is_some_and(|members| members.iter().any(|m| Self::matches(m, &key)))
    }

    /// Insert; false if an equivalent constraint is already present.
    fn insert<C: SdpScalar>(&mut self, p: &Polynomial<C>, v: &C) -> bool {
        let (words, coeffs, value) = Self::normalize(p, v);
        let key = (coeffs, value);
        let members = self.by_support.entry(words).or_default();
        if members.iter().any(|m| Self::matches(m, &key)) {
            return false;
        }
        members.push(key);
        true
    }
}

/// Wall-clock time of each build phase, printed to stderr when the
/// environment variable `NCALG_TIMING` is set.
struct PhaseTimer {
    on: bool,
    last: std::time::Instant,
}

impl PhaseTimer {
    fn new() -> Self {
        PhaseTimer {
            on: std::env::var_os("NCALG_TIMING").is_some(),
            last: std::time::Instant::now(),
        }
    }

    fn mark(&mut self, phase: &str) {
        if self.on {
            let now = std::time::Instant::now();
            eprintln!(
                "ncalg timing: {:<32} {:9.3} s",
                phase,
                (now - self.last).as_secs_f64()
            );
            self.last = now;
        }
    }
}

/// A maximal real-linearly independent subset of columns (each a sparse
/// complex vector, read as the real vector of its real and imaginary
/// parts): `keep[j]` for the kept ones.  The rank decisions are made by the
/// rank-revealing sparse Givens QR ([`crate::sparse_qr`], relative
/// tolerance `1e-9` on unit-norm rows), not by elimination: the data are
/// floating-point.  Free multipliers of dependent columns would span
/// directions that occur in no LMI.
fn independent_columns<K: Ord + Clone>(cols: &[BTreeMap<K, C64>]) -> Vec<bool> {
    let mut index: BTreeMap<K, usize> = BTreeMap::new();
    let rows: Vec<Vec<(usize, f64)>> = cols
        .iter()
        .map(|col| {
            let mut row = Vec::with_capacity(2 * col.len());
            for (k, x) in col {
                let n = index.len();
                let i = *index.entry(k.clone()).or_insert(n);
                if x.re != 0.0 {
                    row.push((2 * i, x.re));
                }
                if x.im != 0.0 {
                    row.push((2 * i + 1, x.im));
                }
            }
            row.sort_by_key(|(c, _)| *c);
            row
        })
        .collect();
    let a = crate::sparse_qr::SparseRows {
        ncols: 2 * index.len(),
        rows,
    };
    let opts = crate::sparse_qr::QrOptions {
        rank_tol: 1e-9,
        ..Default::default()
    };
    let mut keep = vec![true; cols.len()];
    match crate::sparse_qr::SparseQr::factor(&a, &opts) {
        Ok(qr) => {
            for &(r, _) in qr.dependent_rows() {
                keep[r] = false;
            }
        }
        Err(e) => panic!("independent multiplier columns: {}", e),
    }
    keep
}

impl MomentSdp {
    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// Number of free SDP variables `z`.
    pub fn num_vars(&self) -> usize {
        self.objective.len()
    }

    /// Exported block sizes (Hermitian size, doubled for blocks that need the real embedding).
    pub fn block_sizes(&self) -> Vec<usize> {
        self.blocks
            .iter()
            .map(|b| if b.real { b.size } else { 2 * b.size })
            .collect()
    }

    /// Description of the exported blocks.
    pub fn blocks(&self) -> &[BlockInfo] {
        &self.blocks
    }

    /// Per exported block: `None` for a block that is (a component of) its
    /// source matrix, or the basis vectors `v_k` (in the coordinates of
    /// [`source_words`](Self::source_words) of the block's source) in which
    /// a symmetry-reduced block is expressed, `F_b[k][l] = v_k^† F v_l`.
    /// Orthonormal for symmetries acting by signed permutations of words;
    /// see [`Relaxation::block_diagonalize`].
    pub fn block_vectors(&self) -> &[Option<Vec<Vec<C64>>>] {
        &self.block_vectors
    }

    /// Index words of each source matrix (moment matrix first, then localizing).
    pub fn source_words(&self) -> &[Vec<Word>] {
        &self.source_words
    }

    /// Index words of each source matrix (same as `source_words`).
    pub fn block_words(&self) -> &[Vec<Word>] {
        &self.source_words
    }

    /// Normal words on which the functional is represented.
    pub fn moment_words(&self) -> &[Word] {
        &self.moment_words
    }

    /// Is the problem real-symmetric (no real embedding anywhere)?
    pub fn is_real(&self) -> bool {
        self.blocks.iter().all(|b| b.real)
    }

    /// Order of the symmetry group used (1 if none).
    pub fn group_order(&self) -> usize {
        self.group_order
    }

    /// Objective vector `c` of `min c·z + c0` (already negated for `maximize`).
    pub fn objective(&self) -> &[f64] {
        &self.objective
    }

    /// Objective constant `c0`.
    pub fn objective_constant(&self) -> f64 {
        self.objective_constant
    }

    /// Hermitian constant matrices per exported block (upper triangle).
    pub fn hermitian_constant_matrices(&self) -> &[SparseHerm] {
        &self.herm_f0
    }

    /// Hermitian coefficient matrices per exported block and variable.
    pub fn hermitian_coefficient_matrices(&self) -> &[Vec<SparseHerm>] {
        &self.herm_fj
    }

    /// Real symmetric form of block `b`'s constant matrix (embedded if complex).
    pub fn constant_matrix(&self, b: usize) -> SparseSym {
        embed(&self.herm_f0[b], self.blocks[b].size, self.blocks[b].real)
    }

    /// Real symmetric form of block `b`'s coefficient matrix for variable `j`.
    pub fn coefficient_matrix(&self, b: usize, j: usize) -> SparseSym {
        embed(&self.herm_fj[b][j], self.blocks[b].size, self.blocks[b].real)
    }

    /// Total number of nonzero Hermitian LMI entries.
    pub fn num_entries(&self) -> usize {
        self.herm_f0.iter().map(|b| b.len()).sum::<usize>()
            + self.herm_fj.iter().flatten().map(|b| b.len()).sum::<usize>()
    }

    /// Convert a solver's optimum of `min c·z` into the bound on `L(h)`:
    /// the relaxation value is `objective_constant + optimum` (negated back
    /// for `maximize`).
    pub fn relaxation_value(&self, optimum: f64) -> f64 {
        let v = self.objective_constant + optimum;
        if self.minimize {
            v
        } else {
            -v
        }
    }

    /// Write the SDP in SDPA sparse format: `min c·x  s.t.  Σ x_i F_i − F_0 ⪰ 0`
    /// (SDPA's `F_0` is the negative of this struct's constant matrix).
    pub fn write_sdpa(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "\"ncalg NC-SOS moment relaxation in {}: value = {:.17e} + optimum{}",
            self.alg,
            self.objective_constant,
            if self.minimize {
                ""
            } else {
                " (negate for the maximum)"
            }
        );
        let _ = writeln!(s, "{}", self.num_vars());
        let _ = writeln!(s, "{}", self.blocks.len());
        let sizes: Vec<String> = self.block_sizes().iter().map(|m| m.to_string()).collect();
        let _ = writeln!(s, "{}", sizes.join(" "));
        let c: Vec<String> = self.objective.iter().map(|c| format!("{:.17e}", c)).collect();
        let _ = writeln!(s, "{}", c.join(" "));
        for b in 0..self.blocks.len() {
            for (r, col, v) in self.constant_matrix(b) {
                let _ = writeln!(s, "0 {} {} {} {:.17e}", b + 1, r + 1, col + 1, -v);
            }
        }
        for b in 0..self.blocks.len() {
            for j in 0..self.num_vars() {
                for (r, col, v) in self.coefficient_matrix(b, j) {
                    let _ = writeln!(s, "{} {} {} {} {:.17e}", j + 1, b + 1, r + 1, col + 1, v);
                }
            }
        }
        std::fs::write(path, s)
    }

    /// Write the Hermitian block data as JSON for solver front ends that take
    /// complex Hermitian LMIs directly (`scripts/solve_moment_json.py`):
    /// `{"num_vars", "objective", "objective_constant", "minimize",
    ///   "blocks": [{"size", "real", "f0": [[r, c, re, im], ...],
    ///               "fj": [[[r, c, re, im], ...] per variable]}]}`
    /// with `F_0 + Σ z_j F_j ⪰ 0` per block (upper triangles, 0-based).
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
        for (b, info) in self.blocks.iter().enumerate() {
            if b > 0 {
                s.push(',');
            }
            let _ = write!(s, "{{\"size\":{},\"real\":{},\"f0\":[", info.size, info.real);
            for (i, e) in self.herm_f0[b].iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                entry(&mut s, e);
            }
            s.push_str("],\"fj\":[");
            for (j, f) in self.herm_fj[b].iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                s.push('[');
                for (i, e) in f.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    entry(&mut s, e);
                }
                s.push(']');
            }
            s.push_str("]}");
        }
        s.push_str("]}\n");
        std::fs::write(path, s)
    }

    /// Write the index words of every source matrix and the block structure.
    pub fn write_words(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut s = String::new();
        for (b, words) in self.source_words.iter().enumerate() {
            let _ = writeln!(s, "# source matrix {} ({} words)", b, words.len());
            for w in words {
                let _ = writeln!(s, "{}", self.alg.format_word(w));
            }
        }
        let _ = writeln!(
            s,
            "# exported blocks (source, hermitian size, real, irrep dimension)"
        );
        for b in &self.blocks {
            let _ = writeln!(s, "{} {} {} {}", b.source, b.size, b.real, b.irrep_dimension);
        }
        std::fs::write(path, s)
    }

    /// Values `L(w)` of all moment words for a solution vector `z`.
    pub fn moments(&self, z: &[f64]) -> Vec<(Word, C64)> {
        assert_eq!(z.len(), self.num_vars(), "solution vector has the wrong length");
        self.moment_words
            .iter()
            .enumerate()
            .map(|(k, w)| (w.clone(), self.moment_value(z, k)))
            .collect()
    }

    /// `L(w_k)` for a solution vector `z`.
    fn moment_value(&self, z: &[f64], k: usize) -> C64 {
        let coord =
            |i: usize| -> f64 { self.y0[i] + self.b_rows[i].iter().map(|(j, c)| c * z[*j]).sum::<f64>() };
        if self.complex_moments {
            C64::new(coord(2 * k), coord(2 * k + 1))
        } else {
            C64::new(coord(k), 0.0)
        }
    }

    /// `L(p)` for a solution vector `z`; `p` is reduced with `red` first.
    /// Returns `None` if the normal form involves words outside the moment
    /// set or undetermined moments (see [`moment_affine`](Self::moment_affine)).
    ///
    /// Only the moments of the normal form's words are computed (all of
    /// them were, per call: 10³ correlators against 10⁶ moments took 10⁹
    /// operations).
    pub fn evaluate<C: SdpScalar>(&self, z: &[f64], red: &Reducer<C>, p: &Polynomial<C>) -> Option<C64> {
        assert_eq!(z.len(), self.num_vars(), "solution vector has the wrong length");
        let nf = red.reduce(p);
        let mut acc = C64::new(0.0, 0.0);
        for (w, c) in nf.iter() {
            let k = *self.moment_index.get(w.as_slice())?;
            let coords = if self.complex_moments {
                2 * k..2 * k + 2
            } else {
                k..k + 1
            };
            if coords.into_iter().any(|i| self.undetermined[i]) {
                return None;
            }
            let (re, im) = c.to_complex_f64();
            acc += C64::new(re, im) * self.moment_value(z, k);
        }
        Some(acc)
    }

    /// Dense Hermitian matrix of exported block `b` for a solution `z`.
    pub fn block_matrix(&self, z: &[f64], b: usize) -> Vec<Vec<C64>> {
        let m = self.blocks[b].size;
        let mut out = vec![vec![C64::new(0.0, 0.0); m]; m];
        let mut add = |r: usize, c: usize, v: C64| {
            out[r][c] += v;
            if r != c {
                out[c][r] += v.conj();
            }
        };
        for (r, c, v) in &self.herm_f0[b] {
            add(*r, *c, *v);
        }
        for (j, f) in self.herm_fj[b].iter().enumerate() {
            for (r, c, v) in f {
                add(*r, *c, v * z[j]);
            }
        }
        out
    }
}

/// Read a solution vector written one number per line (as the scripts in
/// `scripts/` do); Fortran exponents (`1.0D-03`) are accepted.  Any other
/// token is an error: skipping it shifted every later coordinate.
pub fn read_solution(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<f64>> {
    let text = std::fs::read_to_string(path)?;
    text.split_whitespace()
        .map(|t| {
            t.replace(['D', 'd'], "e").parse::<f64>().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("read_solution: {:?} is not a number", t),
                )
            })
        })
        .collect()
}

/// Output of [`Relaxation::prepare`]: everything both SDP forms need.
struct Prepared {
    real: bool,
    group: Vec<SignedPermutation>,
    generator_indices: Vec<usize>,
    moment_words: Vec<Word>,
    moment_index: HashMap<Vec<Var>, usize>,
    source_words: Vec<Vec<Word>>,
    /// Per source matrix: `(a, b, Σ_k γ_k y_k)` for `a <= b`.
    source_entries: Vec<Vec<(usize, usize, Vec<(usize, C64)>)>>,
    obj_coords: Vec<(usize, C64)>,
    /// User constraints `Σ_k γ_k L(w_k) = value`.
    constraint_rows: Vec<(Vec<(usize, C64)>, C64)>,
    /// Per constraint: the words not determined by the moment/localizing matrices.
    free_constraint_words: Vec<Vec<Word>>,
    /// Row `k`: `[(m, P_{km})]`, `P_{km}` = coefficient of `w_k` in `NF(w_m*)`.
    p_rows: Csr<(usize, C64)>,
    /// Same for the group action, `ρ^g_{km}` = coefficient of `w_k` in
    /// `NF(g(w_m))`: `gen_rho_rows[j][k] = [(m, ρ^g_{km})]` for the generator
    /// `g = group[generator_indices[j]]` and every word `k`.
    gen_rho_rows: Vec<Csr<(usize, C64)>>,
    /// Slot of a component representative (`orbit_rep[k] == k`) in
    /// `rep_rho_rows`, `usize::MAX` for the other words.
    rep_slot: Vec<usize>,
    /// Row `rep_slot[k] · |G| + g`: `[(m, ρ^g_{km})]` for every group element.
    rep_rho_rows: Csr<(usize, C64)>,
    /// Lowest-index word of each word's component under single-term adjoint
    /// and group images.
    orbit_rep: Vec<usize>,
    /// Number of real coordinates (`n_m` or `2 n_m`).
    dim: usize,
}

impl Prepared {
    /// Real coordinate index of `Re L(w_k)` (`imag = false`) or `Im L(w_k)`.
    fn coord(&self, k: usize, imag: bool) -> usize {
        if self.real {
            k
        } else if imag {
            2 * k + 1
        } else {
            2 * k
        }
    }

    /// User constraints as real rows `(Σ_i c_i y_i, rhs)`: for
    /// `Σ_k γ_k (a_k + i b_k) = ρ` the real part `Σ (Re γ_k a_k − Im γ_k b_k) = Re ρ`
    /// and the imaginary part `Σ (Im γ_k a_k + Re γ_k b_k) = Im ρ`.
    fn constraint_real_rows(&self) -> Vec<(BTreeMap<usize, f64>, f64)> {
        let mut out = Vec::new();
        for (coeffs, rhs) in &self.constraint_rows {
            let mut re: BTreeMap<usize, f64> = BTreeMap::new();
            let mut im: BTreeMap<usize, f64> = BTreeMap::new();
            for (k, gamma) in coeffs {
                *re.entry(self.coord(*k, false)).or_insert(0.0) += gamma.re;
                if self.real {
                    assert!(
                        gamma.im.abs() < 1e-12,
                        "complex moment constraint over a real coefficient field"
                    );
                } else {
                    *re.entry(self.coord(*k, true)).or_insert(0.0) -= gamma.im;
                    *im.entry(self.coord(*k, false)).or_insert(0.0) += gamma.im;
                    *im.entry(self.coord(*k, true)).or_insert(0.0) += gamma.re;
                }
            }
            re.retain(|_, c| c.abs() > 1e-14);
            im.retain(|_, c| c.abs() > 1e-14);
            out.push((re, rhs.re));
            if self.real {
                assert!(
                    rhs.im.abs() < 1e-12,
                    "moment constraint with a complex right-hand side over a real coefficient field"
                );
            } else {
                out.push((im, rhs.im));
            }
        }
        out
    }

    /// The Hermitian involution `(S y)_m = conj(Σ_k P_{km} y_k)` on real coordinates.
    fn apply_s(&self, v: &BTreeMap<usize, f64>) -> BTreeMap<usize, f64> {
        let mut out: BTreeMap<usize, f64> = BTreeMap::new();
        for (&idx, &val) in v {
            let (k, imag) = if self.real {
                (idx, false)
            } else {
                (idx / 2, idx % 2 == 1)
            };
            for (m, c) in self.p_rows.row(k) {
                if self.real {
                    *out.entry(*m).or_insert(0.0) += val * c.re;
                } else if !imag {
                    // y_k = val (real): conj(P_{km} val) = val (re - i im)
                    *out.entry(2 * m).or_insert(0.0) += val * c.re;
                    *out.entry(2 * m + 1).or_insert(0.0) -= val * c.im;
                } else {
                    // y_k = i val: conj(P_{km} i val) = val (-im - i re)
                    *out.entry(2 * m).or_insert(0.0) -= val * c.im;
                    *out.entry(2 * m + 1).or_insert(0.0) -= val * c.re;
                }
            }
        }
        out
    }

    /// Group action on functionals, `(σ L)_w = Σ_a ρ^σ_{aw} L_a`, i.e. `ρ^T y`,
    /// with `rows(k)` the row `k` of `ρ^σ`.
    fn apply_rho_t<'r>(
        &self,
        rows: impl Fn(usize) -> &'r [(usize, C64)],
        v: &BTreeMap<usize, f64>,
    ) -> BTreeMap<usize, f64> {
        let mut out: BTreeMap<usize, f64> = BTreeMap::new();
        for (&idx, &val) in v {
            let (k, imag) = if self.real {
                (idx, false)
            } else {
                (idx / 2, idx % 2 == 1)
            };
            for (w, c) in rows(k) {
                if self.real {
                    *out.entry(*w).or_insert(0.0) += val * c.re;
                } else if !imag {
                    *out.entry(2 * w).or_insert(0.0) += val * c.re;
                    *out.entry(2 * w + 1).or_insert(0.0) += val * c.im;
                } else {
                    *out.entry(2 * w).or_insert(0.0) -= val * c.im;
                    *out.entry(2 * w + 1).or_insert(0.0) += val * c.re;
                }
            }
        }
        out
    }

    /// Averaged column `A e_idx = ½(I + S)(1/|G|) Σ_g ρ_g^T e_idx` (a vector in the
    /// space of invariant Hermitian functionals), for a coordinate of a
    /// component representative.
    fn averaged(&self, idx: usize) -> BTreeMap<usize, f64> {
        let mut e: BTreeMap<usize, f64> = BTreeMap::new();
        e.insert(idx, 1.0);
        let mut v: BTreeMap<usize, f64> = BTreeMap::new();
        if self.group.is_empty() {
            v = e;
        } else {
            let k = if self.real { idx } else { idx / 2 };
            let slot = self.rep_slot[k];
            assert!(
                slot != usize::MAX,
                "orbit averages are kept for component representatives only"
            );
            for gi in 0..self.group.len() {
                let rows = self.rep_rho_rows.row(slot * self.group.len() + gi);
                for (i, c) in self.apply_rho_t(|_| rows, &e) {
                    *v.entry(i).or_insert(0.0) += c / self.group.len() as f64;
                }
            }
        }
        let sv = self.apply_s(&v);
        let mut out = v;
        for (i, c) in sv {
            *out.entry(i).or_insert(0.0) += c;
        }
        for c in out.values_mut() {
            *c *= 0.5;
        }
        out.retain(|_, c| c.abs() > 1e-13);
        out
    }
}

/// A moment relaxation in moment coordinates (see
/// [`Relaxation::build_moment_form`]): `Σ_i y_i F_i ⪰ 0` per block,
/// `E y = e`, `min c·y`.
#[derive(Clone, Debug)]
pub struct MomentForm {
    /// Moment words; variable `coord(k, imag)` is `Re`/`Im L(moment_words[k])`.
    pub moment_words: Vec<Word>,
    moment_index: HashMap<Vec<Var>, usize>,
    /// Per moment constraint: words it introduced that nothing else determines
    /// (see [`MomentSdp::free_constraint_words`]).
    pub free_constraint_words: Vec<Vec<Word>>,
    /// Index words of each block.
    pub source_words: Vec<Vec<Word>>,
    /// Complex coefficients (two real variables per moment word)?
    pub complex: bool,
    /// Per block: `(size, coefficient matrices per variable)`, Hermitian upper triangles.
    pub blocks: Vec<(usize, Vec<SparseHerm>)>,
    /// Linear equalities `(Σ_i c_i y_i, rhs)`.
    pub equalities: Vec<(Vec<(usize, f64)>, f64)>,
    /// Objective vector (already negated for `maximize`).
    pub objective: Vec<f64>,
    /// Minimization (`true`) or maximization of the objective.
    pub minimize: bool,
}

impl MomentForm {
    /// Number of real variables.
    pub fn num_vars(&self) -> usize {
        self.objective.len()
    }

    /// The relaxation value from a solver optimum of `min c·y`.
    pub fn relaxation_value(&self, optimum: f64) -> f64 {
        if self.minimize {
            optimum
        } else {
            -optimum
        }
    }

    /// JSON in the same layout as [`MomentSdp::write_json`] plus an
    /// `"equalities": [{"coeffs": [[i, c], ...], "rhs": r}, ...]` field; the
    /// constant matrices are empty.
    pub fn write_json(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let mut s = String::new();
        let _ = write!(
            s,
            "{{\"num_vars\":{},\"minimize\":{},\"objective_constant\":0,\"objective\":[",
            self.num_vars(),
            self.minimize
        );
        for (j, c) in self.objective.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            let _ = write!(s, "{:e}", c);
        }
        s.push_str("],\"blocks\":[");
        for (b, (size, fj)) in self.blocks.iter().enumerate() {
            if b > 0 {
                s.push(',');
            }
            let _ = write!(
                s,
                "{{\"size\":{},\"real\":{},\"f0\":[],\"fj\":[",
                size, !self.complex
            );
            for (j, f) in fj.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                s.push('[');
                for (i, (r, c, v)) in f.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    let _ = write!(s, "[{},{},{:e},{:e}]", r, c, v.re, v.im);
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
        s.push_str("]}\n");
        std::fs::write(path, s)
    }

    /// Real coordinate of `Re L(moment_words[k])` (`imag = false`) or `Im`.
    pub fn coord(&self, k: usize, imag: bool) -> usize {
        if !self.complex {
            k
        } else if imag {
            2 * k + 1
        } else {
            2 * k
        }
    }

    /// `L(p)` as an affine expression in the moment variables; `p` must
    /// already be in normal form.  `None` if a word of `p` is not a moment word.
    pub fn moment_affine<C: SdpScalar>(&self, nf: &Polynomial<C>) -> Option<Affine> {
        let mut e = Affine::zero();
        for (w, c) in nf.iter() {
            let k = *self.moment_index.get(w.as_slice())?;
            let (re, im) = c.to_complex_f64();
            let gamma = C64::new(re, im);
            *e.terms.entry(self.coord(k, false)).or_insert(C64::new(0.0, 0.0)) += gamma;
            if self.complex {
                *e.terms.entry(self.coord(k, true)).or_insert(C64::new(0.0, 0.0)) +=
                    gamma * C64::new(0.0, 1.0);
            }
        }
        e.prune(0.0);
        Some(e)
    }

    /// The same problem as a general [`SdpModel`] (variables = moment
    /// coordinates), to which further constraints and variables can be added.
    pub fn to_model(&self) -> SdpModel {
        let mut model = SdpModel::new();
        for (k, w) in self.moment_words.iter().enumerate() {
            let name = format!("{:?}", w.as_slice());
            model.free_var(format!("Re L{}", name));
            if self.complex {
                model.free_var(format!("Im L{}", name));
            }
            debug_assert_eq!(model.num_vars(), self.coord(k, self.complex) + 1);
        }
        for (s, (size, fj)) in self.blocks.iter().enumerate() {
            model.add_lmi(Lmi {
                size: *size,
                real: !self.complex,
                f0: Vec::new(),
                fj: fj
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| !f.is_empty())
                    .map(|(j, f)| (j, f.clone()))
                    .collect(),
                label: if s == 0 {
                    "moment matrix".to_string()
                } else {
                    format!("localizing matrix {}", s)
                },
            });
        }
        for (row, rhs) in &self.equalities {
            model.add_equality(row.clone(), *rhs);
        }
        let sign = if self.minimize { 1.0 } else { -1.0 };
        let mut obj = Affine::zero();
        for (j, c) in self.objective.iter().enumerate() {
            if *c != 0.0 {
                obj.terms.insert(j, C64::new(sign * c, 0.0));
            }
        }
        if self.minimize {
            model.minimize(&obj);
        } else {
            model.maximize(&obj);
        }
        model
    }
}

impl MomentSdp {
    /// `L(p)` as an affine expression in the free variables `z`; `p` must
    /// already be in normal form.  `None` if a word is not a moment word, or
    /// its moment is undetermined: it occurs in no matrix and not in the
    /// objective, so its free variable was dropped and it has no expression
    /// (it read as a constant, which made constraints added on it trivial or
    /// infeasible).
    pub fn moment_affine<C: SdpScalar>(&self, nf: &Polynomial<C>) -> Option<Affine> {
        let coord = |k: usize, imag: bool| -> usize {
            if !self.complex_moments {
                k
            } else if imag {
                2 * k + 1
            } else {
                2 * k
            }
        };
        let mut e = Affine::zero();
        for (w, c) in nf.iter() {
            let k = *self.moment_index.get(w.as_slice())?;
            let (re, im) = c.to_complex_f64();
            let gamma = C64::new(re, im);
            for (imag, factor) in [(false, gamma), (true, gamma * C64::new(0.0, 1.0))] {
                if imag && !self.complex_moments {
                    continue;
                }
                let i = coord(k, imag);
                if self.undetermined[i] {
                    return None;
                }
                e.constant += factor * self.y0[i];
                for (zj, c) in &self.b_rows[i] {
                    *e.terms.entry(*zj).or_insert(C64::new(0.0, 0.0)) += factor * c;
                }
            }
        }
        // relative to the polynomial's size, not an absolute 1e-15
        let scale = nf.iter().map(|(_, c)| c.magnitude()).fold(0.0_f64, f64::max);
        e.prune(1e-15 * scale);
        Some(e)
    }

    /// The eliminated LMI form as a general [`SdpModel`] in the variables `z`.
    pub fn to_model(&self) -> SdpModel {
        let mut model = SdpModel::new();
        model.free_vars(self.num_vars(), "z");
        for (b, info) in self.blocks.iter().enumerate() {
            model.add_lmi(Lmi {
                size: info.size,
                real: info.real,
                f0: self.herm_f0[b].clone(),
                fj: self.herm_fj[b]
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| !f.is_empty())
                    .map(|(j, f)| (j, f.clone()))
                    .collect(),
                label: format!("source {} block {}", info.source, b),
            });
        }
        let sign = if self.minimize { 1.0 } else { -1.0 };
        let mut obj = Affine::constant(sign * self.objective_constant);
        for (j, c) in self.objective.iter().enumerate() {
            if *c != 0.0 {
                obj.terms.insert(j, C64::new(sign * c, 0.0));
            }
        }
        if self.minimize {
            model.minimize(&obj);
        } else {
            model.maximize(&obj);
        }
        model
    }
}

/// A Gram-matrix (sum-of-squares) formulation of a relaxation on top of a
/// general [`SdpModel`]; see [`Relaxation::sos_model`] and
/// [`Relaxation::gram_weight_model`].
#[derive(Clone, Debug)]
pub struct SosModel {
    /// The SDP.
    pub model: SdpModel,
    /// Gram matrix variables.  Without a symmetry (or when the symmetry does
    /// not act monomially) one per source matrix (moment matrix first, then
    /// one per localizing polynomial `g_k`): `Σ_s Σ_{a,b} G_s[a,b] u_a* g_s u_b`.
    /// With a symmetry acting by signed permutations, one per
    /// symmetry-reduced block of [`MomentSdp::blocks`]; see
    /// [`gram_sources`](Self::gram_sources) and
    /// [`gram_value`](Self::gram_value).
    pub grams: Vec<MatrixVar>,
    /// Source matrix of each Gram variable.
    pub gram_sources: Vec<usize>,
    /// Irreducible dimension of each Gram variable (1 without reduction).
    pub gram_irrep_dimensions: Vec<usize>,
    /// Basis vectors (source coordinates) of each Gram variable, `None` for
    /// a whole source matrix.
    gram_vectors: Vec<Option<Vec<Vec<C64>>>>,
    /// The symmetry group the identity is averaged over (empty without
    /// reduction).
    group: Vec<SignedPermutation>,
    /// Per source matrix: the group's monomial action on its index words, if
    /// it is monomial (as [`MomentSdp`]'s), for Gram-level averaging.
    index_action: Vec<Option<Vec<(Vec<usize>, Vec<C64>)>>>,
    /// Index words of each Gram matrix.
    pub source_words: Vec<Vec<Word>>,
    /// The bound variable `c` of `h − c ∈ Σ` (`sos_model`), if any.
    pub bound: Option<VarId>,
    /// The diagonal weights `w` (`gram_weight_model`), empty otherwise.
    pub weights: Vec<VarId>,
    /// The fixed shift of `gram_weight_model`.
    pub shift: f64,
    /// Multipliers `μ_j` of the moment constraints, one per entry `r_j` of
    /// [`multiplier_polynomials`](Self::multiplier_polynomials) (a linearly
    /// independent subset of the Hermitian parts; with a symmetry, of their
    /// group averages); the identity is
    /// `target + Σ_j μ_j r_j = Σ_s Σ G_s[a,b] u_a* g_s u_b`.
    pub multipliers: Vec<VarId>,
    /// The Hermitian polynomials `r_j` (normal forms, `L(r_j) = 0` for every
    /// feasible functional) that the multipliers multiply.
    pub multiplier_polynomials: Vec<Polynomial<C64>>,
    minimize: bool,
    alg: FreeAlgebra,
    localizers: Vec<Option<Polynomial<C64>>>,
}

impl SosModel {
    /// Value of the bound `c` at a solution (the relaxation value).
    pub fn bound_value(&self, x: &[f64]) -> Option<f64> {
        self.bound.map(|c| x[c])
    }

    /// Gram matrix of source `s` at a solution, in the coordinates of
    /// [`source_words`](Self::source_words)`[s]`.  With a symmetry-reduced
    /// form this is the lift `Σ_b d_b conj(V_b) G_b V_bᵀ` of the reduced
    /// blocks (one copy per irreducible); the identity holds for its group
    /// average, which [`certificate`](Self::certificate) forms.
    pub fn gram_value(&self, s: usize, x: &[f64]) -> Vec<Vec<C64>> {
        let m = self.source_words[s].len();
        let zero = C64::new(0.0, 0.0);
        let mut g = vec![vec![zero; m]; m];
        for (b, gram) in self.grams.iter().enumerate() {
            if self.gram_sources[b] != s {
                continue;
            }
            let y = gram.value(x);
            match &self.gram_vectors[b] {
                None => {
                    for a in 0..m {
                        for c in 0..m {
                            g[a][c] += y[a][c];
                        }
                    }
                }
                Some(vecs) => {
                    let d = self.gram_irrep_dimensions[b] as f64;
                    let sparse: Vec<Vec<(usize, C64)>> = vecs
                        .iter()
                        .map(|v| {
                            v.iter()
                                .enumerate()
                                .filter(|(_, c)| c.norm() != 0.0)
                                .map(|(a, c)| (a, *c))
                                .collect()
                        })
                        .collect();
                    for (k, vk) in sparse.iter().enumerate() {
                        for (l, vl) in sparse.iter().enumerate() {
                            let ykl = y[k][l] * d;
                            if ykl.norm() == 0.0 {
                                continue;
                            }
                            for &(a, va) in vk {
                                let f = va.conj() * ykl;
                                for &(c, vc) in vl {
                                    g[a][c] += f * vc;
                                }
                            }
                        }
                    }
                }
            }
        }
        g
    }

    /// diagonal weights at a solution.
    pub fn weight_values(&self, x: &[f64]) -> Vec<f64> {
        self.weights.iter().map(|w| x[*w]).collect()
    }

    /// Was the underlying objective minimized (`h − c ∈ Σ`) or maximized
    /// (`c − h ∈ Σ`)?
    pub fn is_minimize(&self) -> bool {
        self.minimize
    }
}

impl<'a, C: SdpScalar> Relaxation<'a, C> {
    /// Gram matrices `G_s ⪰ 0` for every source matrix and the coefficient
    /// (an affine expression in their entries) of every normal word in
    /// `Σ_s Σ_{a,b} G_s[a,b] NF(u_a* g_s u_b)`.
    fn sos_expansion(&self, model: &mut SdpModel) -> (Vec<MatrixVar>, BTreeMap<Vec<Var>, Affine>) {
        let red = self.red;
        let alg = red.algebra();
        let real = C::REAL;
        let mut grams = Vec::new();
        let mut expansion: BTreeMap<Vec<Var>, Affine> = BTreeMap::new();
        let mut sources: Vec<(Option<&Polynomial<C>>, &Vec<Word>)> = vec![(None, &self.words)];
        for (g, w) in &self.localizing {
            sources.push((Some(g), w));
        }
        for (s, (g, words)) in sources.into_iter().enumerate() {
            let m = words.len();
            let gram = model.psd_var(
                m,
                real,
                if s == 0 {
                    "G".to_string()
                } else {
                    format!("G{}", s)
                },
            );
            let nf = |a: usize, b: usize| -> Polynomial<C> {
                let ua = alg.adjoint_word(&words[a]);
                match g {
                    None => red.reduce(&alg.monomial(ua.concat(&words[b]))),
                    Some(g) => red.reduce(&g.sandwich(&ua, &words[b])),
                }
            };
            let add = |p: &Polynomial<C>, coef: &Affine, expansion: &mut BTreeMap<Vec<Var>, Affine>| {
                for (w, c) in p.iter() {
                    let (re, im) = c.to_complex_f64();
                    expansion
                        .entry(w.as_slice().to_vec())
                        .or_insert_with(Affine::zero)
                        .add_scaled(coef, C64::new(re, im));
                }
            };
            for a in 0..m {
                add(&nf(a, a), &gram.entry(a, a), &mut expansion);
                for b in a + 1..m {
                    let e = gram.entry(a, b);
                    add(&nf(a, b), &e, &mut expansion);
                    add(&nf(b, a), &e.conj(), &mut expansion);
                }
            }
            grams.push(gram);
        }
        for e in expansion.values_mut() {
            e.prune(1e-15);
        }
        (grams, expansion)
    }

    /// Impose `expansion = target` coefficient-wise (normal words).
    fn equate_words(
        model: &mut SdpModel,
        expansion: &BTreeMap<Vec<Var>, Affine>,
        target: &BTreeMap<Vec<Var>, Affine>,
    ) {
        let words: std::collections::BTreeSet<&Vec<Var>> = expansion.keys().chain(target.keys()).collect();
        let zero = Affine::zero();
        for w in words {
            let lhs = expansion.get(w).unwrap_or(&zero);
            let rhs = target.get(w).unwrap_or(&zero);
            model.equal(lhs, rhs);
        }
    }

    /// Target coefficients `sign·(h + shift)` or with the bound variable.
    fn sos_target(&self, bound: Option<VarId>, shift: f64) -> BTreeMap<Vec<Var>, Affine> {
        let h = self
            .objective
            .as_ref()
            .expect("the Gram form needs an objective (minimize / maximize)");
        let nf = self.red.reduce(h);
        // minimize: certificate of h − c ⪰ 0 (or h + shift ⪰ 0)
        // maximize: certificate of c − h ⪰ 0 (or shift − h ⪰ 0)
        let sign = if self.minimize { 1.0 } else { -1.0 };
        let mut target: BTreeMap<Vec<Var>, Affine> = BTreeMap::new();
        for (w, c) in nf.iter() {
            let (re, im) = c.to_complex_f64();
            target.insert(
                w.as_slice().to_vec(),
                Affine::constant(C64::new(sign * re, sign * im)),
            );
        }
        let one = target.entry(Vec::new()).or_insert_with(Affine::zero);
        match bound {
            Some(c) => one.add_scaled(&Affine::var(c), -sign),
            None => one.constant += C64::new(shift, 0.0),
        }
        target
    }

    /// The Gram-matrix (SOS) form of the relaxation: for `minimize(h)`
    ///
    /// ```text
    ///   max c   s.t.   h − c = Σ_s Σ_{a,b} G_s[a,b] u_a* g_s u_b   (mod I),   G_s ⪰ 0
    /// ```
    ///
    /// (`g_0 = 1` for the moment matrix; `min c` with `c − h` for
    /// `maximize`).  Moment constraints `L(p_k) = v_k`
    /// ([`moment_equality`](Self::moment_equality)) enter as free
    /// multiplier terms `Σ_j μ_j r_j` on the left, one per Hermitian part
    /// `r_j` of `p_k − v_k` ([`SosModel::multiplier_polynomials`]): the exact
    /// dual of the constrained moment problem.  Its optimum equals the
    /// moment relaxation's value under strong duality, and the solution
    /// gives an explicit SOS certificate.
    ///
    /// With a [`symmetry`](Self::symmetry) acting on the words by signed
    /// permutations the form is symmetry-reduced: one Gram block per block
    /// of the eliminated moment SDP ([`build`](Self::build)), in its
    /// orthonormal basis, the identity matched on orbit representatives of
    /// its group average, and multipliers for the group-averaged
    /// constraints.  This is exact (an optimal Gram matrix can be taken
    /// invariant).  Otherwise the full Gram matrices are used and the
    /// symmetry is ignored.
    pub fn sos_model(&self) -> SosModel {
        if let Some(m) = self.symmetric_gram_model(None) {
            return m;
        }
        let mut model = SdpModel::new();
        let (grams, expansion) = self.sos_expansion(&mut model);
        let c = model.free_var("c");
        let mut target = self.sos_target(Some(c), 0.0);
        let (multipliers, multiplier_polynomials) = self.add_multipliers(&mut model, &mut target);
        Self::equate_words(&mut model, &expansion, &target);
        if self.minimize {
            model.maximize(&Affine::var(c));
        } else {
            model.minimize(&Affine::var(c));
        }
        let n = grams.len();
        SosModel {
            model,
            grams,
            gram_sources: (0..n).collect(),
            gram_irrep_dimensions: vec![1; n],
            gram_vectors: vec![None; n],
            group: Vec::new(),
            index_action: Vec::new(),
            source_words: self.source_words(),
            bound: Some(c),
            weights: Vec::new(),
            shift: 0.0,
            multipliers,
            multiplier_polynomials,
            minimize: self.minimize,
            alg: self.red.algebra().clone(),
            localizers: self.localizers_f64(),
        }
    }

    /// The Gram-weight problem for a fixed shift: for `minimize(h)`
    ///
    /// ```text
    ///   min Σ_u w_u   s.t.   h + shift = Σ_s Σ_{a,b} G_s[a,b] u_a* g_s u_b   (mod I),
    ///                        G_s ⪰ 0,   diag(w) − G_0 ⪰ 0
    /// ```
    ///
    /// (`shift − h` for `maximize`), with the multiplier terms `Σ_j μ_j r_j`
    /// of the moment constraints added on the left as in
    /// [`sos_model`](Self::sos_model).  With a symmetry it is reduced as in
    /// [`sos_model`](Self::sos_model), with weights constant on the orbits
    /// of the index words (exact: averaging a feasible `(G, w)` keeps it
    /// feasible with the same `Σ w`) and `conj(V_b^† diag(w) V_b) − G_b ⪰ 0`
    /// per block.  The problem is infeasible unless
    /// `shift ≥ −(relaxation value)`; the weights bound the diagonal of the
    /// Gram matrix of the moment-matrix words.
    pub fn gram_weight_model(&self, shift: f64) -> SosModel {
        if let Some(m) = self.symmetric_gram_model(Some(shift)) {
            return m;
        }
        let mut model = SdpModel::new();
        let (grams, expansion) = self.sos_expansion(&mut model);
        let mut target = self.sos_target(None, shift);
        let (multipliers, multiplier_polynomials) = self.add_multipliers(&mut model, &mut target);
        Self::equate_words(&mut model, &expansion, &target);
        let g = &grams[0];
        let m = g.size();
        let weights = model.free_vars(m, "w");
        let mut entries = Vec::with_capacity(m * (m + 1) / 2);
        for a in 0..m {
            entries.push((a, a, Affine::var(weights[a]) - g.entry(a, a)));
            for b in a + 1..m {
                entries.push((a, b, -g.entry(a, b)));
            }
        }
        model.psd_affine(m, g.is_real(), entries, "diag(w) - G");
        let mut total = Affine::zero();
        for w in &weights {
            total.terms.insert(*w, C64::new(1.0, 0.0));
        }
        model.minimize(&total);
        let n = grams.len();
        SosModel {
            model,
            grams,
            gram_sources: (0..n).collect(),
            gram_irrep_dimensions: vec![1; n],
            gram_vectors: vec![None; n],
            group: Vec::new(),
            index_action: Vec::new(),
            source_words: self.source_words(),
            bound: None,
            weights,
            shift,
            multipliers,
            multiplier_polynomials,
            minimize: self.minimize,
            alg: self.red.algebra().clone(),
            localizers: self.localizers_f64(),
        }
    }

    fn source_words(&self) -> Vec<Vec<Word>> {
        let mut v = vec![self.words.clone()];
        for (_, w) in &self.localizing {
            v.push(w.clone());
        }
        v
    }
}

/// A polynomial over any SDP scalar as a complex-float polynomial.
fn to_c64_poly<C: SdpScalar>(p: &Polynomial<C>) -> Polynomial<C64> {
    p.map_coeffs(|c| {
        let (re, im) = c.to_complex_f64();
        C64::new(re, im)
    })
}

/// One term `weight · p* g_source p` of a [`Certificate`].
#[derive(Clone, Debug)]
pub struct Square {
    /// Nonnegative weight `λ_k` (an eigenvalue of the Gram matrix).
    pub weight: f64,
    /// Which Gram matrix the square comes from (0 = moment matrix, `g = 1`;
    /// `s ≥ 1` = the localizing polynomial `g_s`).
    pub source: usize,
    /// The polynomial `p_k`, a combination of the index words.
    pub poly: Polynomial<C64>,
}

/// An explicit (floating-point) sum-of-squares certificate
///
/// ```text
///   h − c = Σ_k λ_k p_k* g_{s_k} p_k   (mod I)        (minimize)
///   c − h = Σ_k λ_k p_k* g_{s_k} p_k   (mod I)        (maximize)
/// ```
///
/// extracted from a Gram matrix solution ([`SosModel::certificate`]) or from
/// the dual matrices of the moment SDP ([`MomentSdp::certificate`]).  The
/// certified value `c` and the numerical defect are computed by
/// [`verify`](Certificate::verify) through the normal form of the identity.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// The squares.
    pub squares: Vec<Square>,
    /// `g_s` per source (`None` for the moment matrix).
    pub localizers: Vec<Option<Polynomial<C64>>>,
    /// Multiplier terms `(m_j, r_j)` of the moment constraints, with
    /// `L(r_j) = 0` for every feasible functional: the identity is
    /// `h − c = Σ_k λ_k p_k* g p_k + Σ_j m_j r_j` (minimize; `c − h` for
    /// maximize).  Empty without moment constraints.
    pub multipliers: Vec<(f64, Polynomial<C64>)>,
    /// Which identity is certified.
    pub minimize: bool,
    alg: FreeAlgebra,
}

/// `NF(p)` with the terms of `p` split into chunks reduced in parallel and
/// added (reduction is linear).  The sum of a certificate is merged to
/// distinct words first, so each word is still reduced once; reducing the
/// Gram rows separately instead repeated the shared words (7× the work).
/// On the QMC 11-cycle the check reduces 7.2 million distinct words: one
/// sequential call before, all cores now.
fn reduce_in_parallel(red: &Reducer<C64>, p: &Polynomial<C64>) -> Polynomial<C64> {
    use rayon::prelude::*;
    const CHUNK: usize = 1 << 12;
    let terms: Vec<(Word, C64)> = p.iter().map(|(w, c)| (w.clone(), *c)).collect();
    if terms.len() <= CHUNK {
        return red.reduce(p);
    }
    // the caller's exact-zero mode (verify_exact) is thread-local
    let exact = crate::coeff::exact_zero_active();
    let alg = p.algebra();
    let parts: Vec<Polynomial<C64>> = terms
        .par_chunks(CHUNK)
        .map(|chunk| {
            let nf = || red.reduce(&Polynomial::from_terms(alg.clone(), chunk.iter().cloned()));
            if exact {
                crate::coeff::with_exact_zero(nf)
            } else {
                nf()
            }
        })
        .collect();
    Polynomial::sum(alg, &parts)
}

/// Result of [`Certificate::verify`].
#[derive(Clone, Debug)]
pub struct Verification {
    /// The constant `c` of the identity (the certified bound on `L(h)`).
    pub bound: f64,
    /// `NF(h ∓ Σ) − c`: what is left after reduction, ideally zero.
    pub residual: Polynomial<C64>,
    /// Largest coefficient magnitude of the residual.  Sums that cancel to
    /// rounding level relative to their summands count as zero, and so
    /// do coefficients at or below an absolute floor if one is set
    /// ([`float_tolerance`](Self::float_tolerance));
    /// [`Certificate::verify_exact`] counts exact zeros only.
    pub max_residual: f64,
    /// The absolute float floor in effect during the verification
    /// ([`crate::set_float_tolerance`], default `0`; `0` for
    /// [`Certificate::verify_exact`]).
    pub float_tolerance: f64,
}

impl Certificate {
    /// Number of squares.
    pub fn num_squares(&self) -> usize {
        self.squares.len()
    }

    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// `Σ_k λ_k p_k* g_{s_k} p_k` as an unreduced polynomial.
    ///
    /// Computed through the Gram matrix of each source: with `p_k = Σ_w
    /// c_k[w] w` over the words `W` the squares of a source use,
    /// `Σ_k λ_k p_k* g p_k = Σ_{a,b ∈ W} G[a,b] a* g b` for
    /// `G = Σ_k λ_k conj(c_k) c_kᵀ`.  That takes `|W|²` word products
    /// instead of `Σ_k nnz(p_k)²` (a group-averaged certificate has `|G|`
    /// times as many squares over the same words), and one normalization
    /// of all terms at the end.
    pub fn sum(&self) -> Polynomial<C64> {
        use rayon::prelude::*;
        let zero = C64::new(0.0, 0.0);
        let mut terms: Vec<(Word, C64)> = Vec::new();
        for (g, words, gram) in self.grams() {
            let n = words.len();
            // row a: Σ_b G[a,b] (a* g) b
            let rows: Vec<Vec<(Word, C64)>> = (0..n)
                .into_par_iter()
                .map(|a| {
                    let star = self.alg.monomial::<C64>(words[a].clone()).adjoint();
                    let left = match g {
                        None => star,
                        Some(g) => &star * g,
                    };
                    let mut out = Vec::new();
                    for b in 0..n {
                        let gab = gram[a * n + b];
                        if gab == zero {
                            continue;
                        }
                        for (w, c) in left.mul_word_right(words[b].as_slice()).iter() {
                            out.push((w.clone(), *c * gab));
                        }
                    }
                    out
                })
                .collect();
            terms.extend(rows.into_iter().flatten());
        }
        Polynomial::from_terms(self.alg.clone(), terms)
    }

    /// `NF(Σ_k λ_k p_k* g_{s_k} p_k)`, row by row of the Gram matrices: for
    /// row `a`, `q = Σ_b G[a,b] NF(b)` and every term `c u` of `a* g`,
    /// `NF(u q) = L_{u_1}(⋯ L_{u_k}(q))` with `L_x(Σ_v c_v v) = Σ_v c_v
    /// NF(x v)` (`x (v − NF(v))` lies in the ideal).  `NF(x v)` is memoised
    /// over the pairs of a letter and a normal word that occur.  Reducing
    /// the `|W|²` products `a* g b` one by one instead took 12.5 s of the
    /// QMC 11-cycle's 22 s certificate check (7.2 · 10⁶ distinct words of
    /// degree 4).
    ///
    /// Coefficients are summed per word, with the cancellation test of
    /// [`Polynomial`] addition against the summands' total size;
    /// exact-zero mode ([`crate::coeff::with_exact_zero`]) is honored.
    fn reduced_sum(&self, red: &Reducer<C64>) -> Polynomial<C64> {
        use rayon::prelude::*;
        use std::sync::{Arc, Mutex, RwLock};
        // normal words are interned; terms are (id, coefficient)
        type Terms = Vec<(u32, C64)>;
        const SHARDS: usize = 64;
        let zero = C64::new(0.0, 0.0);
        let exact = crate::coeff::exact_zero_active();
        let in_mode = |f: &mut dyn FnMut()| {
            if exact {
                crate::coeff::with_exact_zero(f)
            } else {
                f()
            }
        };
        let ids: RwLock<HashMap<Word, u32>> = RwLock::new(HashMap::new());
        let words_of: RwLock<Vec<Word>> = RwLock::new(Vec::new());
        let intern = |w: &Word| -> u32 {
            if let Some(&i) = ids.read().expect("interner").get(w) {
                return i;
            }
            let mut map = ids.write().expect("interner");
            if let Some(&i) = map.get(w) {
                return i;
            }
            let mut list = words_of.write().expect("interner");
            let i = u32::try_from(list.len()).expect("fewer than 2³² normal words");
            list.push(w.clone());
            map.insert(w.clone(), i);
            i
        };
        let nf = |w: Word| -> Terms {
            red.reduce(&self.alg.monomial(w))
                .iter()
                .map(|(w, c)| (intern(w), *c))
                .collect()
        };
        // NF(x v) for a letter x and a normal word v, memoised
        let cache: Vec<Mutex<HashMap<(Var, u32), Arc<Terms>>>> =
            (0..SHARDS).map(|_| Mutex::new(HashMap::new())).collect();
        let times = |x: Var, v: u32| -> Arc<Terms> {
            let shard = &cache[(v as usize).wrapping_mul(31).wrapping_add(x as usize) % SHARDS];
            if let Some(t) = shard.lock().expect("cache").get(&(x, v)) {
                return t.clone();
            }
            let xv = {
                let list = words_of.read().expect("interner");
                let mut xv = Vec::with_capacity(list[v as usize].len() + 1);
                xv.push(x);
                xv.extend_from_slice(list[v as usize].as_slice());
                xv
            };
            let t = Arc::new(nf(Word::from_vec(xv)));
            shard.lock().expect("cache").insert((x, v), t.clone());
            t
        };
        // merge equal ids, dropping sums that cancel to rounding level
        // relative to their summands, as `Polynomial` addition does
        struct Scratch {
            dense: Vec<(C64, f64)>,
            seen: Vec<bool>,
            touched: Vec<u32>,
        }
        let merge = |sc: &mut Scratch, terms: &mut dyn Iterator<Item = (u32, C64)>| -> Terms {
            for (i, c) in terms {
                let i = i as usize;
                if i >= sc.dense.len() {
                    sc.dense.resize(i + 1 + i / 2, (zero, 0.0));
                    sc.seen.resize(i + 1 + i / 2, false);
                }
                if !sc.seen[i] {
                    sc.seen[i] = true;
                    sc.touched.push(i as u32);
                }
                sc.dense[i].0 += c;
                sc.dense[i].1 += c.norm();
            }
            let mut out = Vec::with_capacity(sc.touched.len());
            for &i in &sc.touched {
                let (c, m) = std::mem::replace(&mut sc.dense[i as usize], (zero, 0.0));
                sc.seen[i as usize] = false;
                if !<C64 as crate::coeff::Coeff>::canceled(&c, m) {
                    out.push((i, c));
                }
            }
            sc.touched.clear();
            out
        };
        // per word: the sum and the summands' total size (cancellation)
        let mut total: Vec<(C64, f64)> = Vec::new();
        let add_into = |acc: &mut Vec<(C64, f64)>, i: u32, c: C64| {
            let i = i as usize;
            if i >= acc.len() {
                acc.resize(i + 1 + i / 2, (zero, 0.0));
            }
            acc[i].0 += c;
            acc[i].1 += c.norm();
        };
        for (g, words, gram) in self.grams() {
            let n = words.len();
            let mut nf_words: Vec<Terms> = Vec::new();
            in_mode(&mut || nf_words = words.par_iter().map(|w| nf(w.clone())).collect());
            let part: Vec<(C64, f64)> = (0..n)
                .into_par_iter()
                .fold(
                    || {
                        (
                            Vec::new(),
                            Scratch {
                                dense: Vec::new(),
                                seen: Vec::new(),
                                touched: Vec::new(),
                            },
                        )
                    },
                    |(mut acc, mut sc), a| {
                        in_mode(&mut || {
                            let q = merge(
                                &mut sc,
                                &mut (0..n).filter(|&b| gram[a * n + b] != zero).flat_map(|b| {
                                    let gab = gram[a * n + b];
                                    nf_words[b].iter().map(move |&(w, c)| (w, c * gab))
                                }),
                            );
                            let star = self.alg.monomial::<C64>(words[a].clone()).adjoint();
                            let left = match g {
                                None => star,
                                Some(g) => &star * g,
                            };
                            for (u, cu) in left.iter() {
                                let mut r = q.clone();
                                for &x in u.as_slice().iter().rev() {
                                    r = merge(
                                        &mut sc,
                                        &mut r.iter().flat_map(|&(v, c)| {
                                            let t = times(x, v);
                                            (0..t.len()).map(move |k| (t[k].0, c * t[k].1))
                                        }),
                                    );
                                }
                                for (w, c) in r {
                                    add_into(&mut acc, w, c * *cu);
                                }
                            }
                        });
                        (acc, sc)
                    },
                )
                .map(|(acc, _)| acc)
                .reduce(Vec::new, |mut a, b| {
                    if a.len() < b.len() {
                        a.resize(b.len(), (zero, 0.0));
                    }
                    for (i, (c, m)) in b.into_iter().enumerate() {
                        a[i].0 += c;
                        a[i].1 += m;
                    }
                    a
                });
            if total.len() < part.len() {
                total.resize(part.len(), (zero, 0.0));
            }
            for (i, (c, m)) in part.into_iter().enumerate() {
                total[i].0 += c;
                total[i].1 += m;
            }
        }
        let list = words_of.into_inner().expect("interner");
        Polynomial::from_terms(
            self.alg.clone(),
            total
                .into_iter()
                .enumerate()
                .filter(|(_, (c, m))| *m > 0.0 && !<C64 as crate::coeff::Coeff>::canceled(c, *m))
                .map(|(i, (c, _))| (list[i].clone(), c)),
        )
    }

    /// Per source with squares: its localizer, the words `W` its squares
    /// use, and the Gram matrix `G = Σ_k λ_k conj(c_k) c_kᵀ` on them
    /// (row-major `|W| × |W|`).
    #[allow(clippy::type_complexity)]
    fn grams(&self) -> Vec<(Option<&Polynomial<C64>>, Vec<Word>, Vec<C64>)> {
        let zero = C64::new(0.0, 0.0);
        let mut out = Vec::new();
        for (s, g) in self.localizers.iter().enumerate() {
            let squares: Vec<&Square> = self.squares.iter().filter(|sq| sq.source == s).collect();
            if squares.is_empty() {
                continue;
            }
            let mut index: HashMap<Word, usize> = HashMap::new();
            let mut words: Vec<Word> = Vec::new();
            for sq in &squares {
                for (w, _) in sq.poly.iter() {
                    if !index.contains_key(w) {
                        index.insert(w.clone(), words.len());
                        words.push(w.clone());
                    }
                }
            }
            let n = words.len();
            // G[a,b] = Σ_k λ_k conj(c_k[a]) c_k[b]: entry by entry for sparse
            // squares, otherwise as the dense product (Λ C)ᵀ conj(C) in chunks
            // of squares (dense certificates of non-monomial symmetric
            // builds have m squares of m terms: Σ nnz² = 1e10 took 6 s on
            // one thread for the QMC 10-cycle)
            let work: usize = squares.iter().map(|sq| sq.poly.iter().count().pow(2)).sum();
            let mut gram = vec![zero; n * n];
            if work <= 1 << 24 {
                for sq in &squares {
                    let c: Vec<(usize, C64)> = sq.poly.iter().map(|(w, c)| (index[w], *c)).collect();
                    for &(a, ca) in &c {
                        let f = ca.conj() * sq.weight;
                        for &(b, cb) in &c {
                            gram[a * n + b] += f * cb;
                        }
                    }
                }
            } else {
                const CHUNK: usize = 2048;
                // real coefficients (real representations): a real product,
                // a quarter of the work
                let real = squares.iter().all(|sq| sq.poly.iter().all(|(_, c)| c.im == 0.0));
                let mut acc = faer::Mat::<C64>::zeros(n, n);
                if real {
                    let mut racc = faer::Mat::<f64>::zeros(n, n);
                    for part in squares.chunks(CHUNK) {
                        let mut c = faer::Mat::<f64>::zeros(part.len(), n);
                        let mut lc = faer::Mat::<f64>::zeros(part.len(), n);
                        for (k, sq) in part.iter().enumerate() {
                            for (w, x) in sq.poly.iter() {
                                let a = index[w];
                                c[(k, a)] = x.re;
                                lc[(k, a)] = x.re * sq.weight;
                            }
                        }
                        racc += crate::dense::matmul(c.transpose(), lc.as_ref());
                    }
                    acc = faer::Mat::<C64>::from_fn(n, n, |a, b| C64::new(racc[(a, b)], 0.0));
                }
                for part in squares.chunks(CHUNK).filter(|_| !real) {
                    // rows: conj(c_k) and λ_k c_k, so acc += conj(C)ᵀ (Λ C)
                    let mut cc = faer::Mat::<C64>::zeros(part.len(), n);
                    let mut lc = faer::Mat::<C64>::zeros(part.len(), n);
                    for (k, sq) in part.iter().enumerate() {
                        for (w, c) in sq.poly.iter() {
                            let a = index[w];
                            cc[(k, a)] = c.conj();
                            lc[(k, a)] = *c * sq.weight;
                        }
                    }
                    let t = crate::dense::matmul(cc.transpose(), lc.as_ref());
                    acc += &t;
                }
                for a in 0..n {
                    for b in 0..n {
                        gram[a * n + b] = acc[(a, b)];
                    }
                }
            }
            out.push((g.as_ref(), words, gram));
        }
        out
    }

    /// Reduce the identity (squares and multiplier terms): returns the
    /// constant `c` it certifies and the residual.  `red` must reduce modulo the same ideal (e.g.
    /// `gb.convert::<Complex<f64>>().reducer()`), `h` is the objective as a
    /// complex-float polynomial.
    ///
    /// Float sums that cancel to rounding level relative to their summands
    /// count as zero (`|Σ c_i| ≤ 64 ε Σ |c_i|`), as do coefficients at
    /// or below an absolute floor if one is set
    /// ([`crate::set_float_tolerance`], default `0`).  Genuinely small residual
    /// coefficients are reported.
    pub fn verify(&self, red: &Reducer<C64>, h: &Polynomial<C64>) -> Verification {
        // NF(h ∓ (Σ squares + Σ multipliers)), the squares reduced row by row
        let mut m = Polynomial::zero(self.alg.clone());
        for (mj, r) in &self.multipliers {
            m += r.map_coeffs(|c| c * *mj);
        }
        let s = self.reduced_sum(red);
        let nf = if self.minimize {
            reduce_in_parallel(red, &h.sub_ref(&m)).sub_ref(&s)
        } else {
            reduce_in_parallel(red, &(h.clone() + m)) + s
        };
        let bound = nf.constant_term().re;
        let residual = nf.sub_ref(&self.alg.constant(C64::new(bound, 0.0)));
        let max_residual = residual.iter().map(|(_, c)| c.norm()).fold(0.0, f64::max);
        Verification {
            bound,
            residual,
            max_residual,
            float_tolerance: crate::coeff::float_tolerance(),
        }
    }

    /// [`verify`](Self::verify) with exact zero tests: neither the absolute
    /// floor nor the cancellation test, so only exact zeros vanish and
    /// `max_residual` includes rounding-level residue too.  Thread-local
    /// ([`crate::coeff::with_exact_zero`]): other threads are unaffected.
    pub fn verify_exact(&self, red: &Reducer<C64>, h: &Polynomial<C64>) -> Verification {
        crate::coeff::with_exact_zero(|| self.verify(red, h))
    }

    /// Average the certificate over a group of signed permutations (each
    /// square `p` becomes `|G|` squares `σ(p)` with weight `λ/|G|`).  Needed
    /// when the Gram matrices come from a symmetry-reduced moment SDP, whose
    /// identity only holds modulo the orbit relations.
    pub fn symmetrize(&self, group: &[SignedPermutation]) -> Certificate {
        if group.is_empty() {
            return self.clone();
        }
        let scale = 1.0 / group.len() as f64;
        let mut squares = Vec::with_capacity(self.squares.len() * group.len());
        for sq in &self.squares {
            for g in group {
                squares.push(Square {
                    weight: sq.weight * scale,
                    source: sq.source,
                    poly: g.apply(&sq.poly, &self.alg),
                });
            }
        }
        let mut multipliers = Vec::with_capacity(self.multipliers.len() * group.len());
        for (m, r) in &self.multipliers {
            for g in group {
                multipliers.push((m * scale, g.apply(r, &self.alg)));
            }
        }
        Certificate {
            squares,
            localizers: self.localizers.clone(),
            multipliers,
            minimize: self.minimize,
            alg: self.alg.clone(),
        }
    }

    /// Drop squares with weight below `tol` times the largest weight.
    pub fn pruned(&self, tol: f64) -> Certificate {
        let max = self.squares.iter().map(|s| s.weight).fold(0.0, f64::max);
        Certificate {
            squares: self
                .squares
                .iter()
                .filter(|s| s.weight > tol * max)
                .cloned()
                .collect(),
            localizers: self.localizers.clone(),
            multipliers: self.multipliers.clone(),
            minimize: self.minimize,
            alg: self.alg.clone(),
        }
    }
}

impl std::fmt::Display for Certificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for sq in &self.squares {
            match &self.localizers[sq.source] {
                None => writeln!(f, "{:.6} * ({})* ({})", sq.weight, sq.poly, sq.poly)?,
                Some(g) => writeln!(f, "{:.6} * ({})* ({}) ({})", sq.weight, sq.poly, g, sq.poly)?,
            }
        }
        for (m, r) in &self.multipliers {
            writeln!(f, "{:.6} * [{}]   (L = 0)", m, r)?;
        }
        Ok(())
    }
}

/// Squares `λ_k p_k* p_k` of a Hermitian PSD matrix `G` indexed by `words`:
/// `G = Σ_k λ_k v_k v_k†`, `p_k = Σ_b conj(v_k[b]) u_b`.  Eigenvalues at most
/// `tol · max|λ|` are dropped.
/// `2^{-k} g` with `2^k` the power of two nearest to the largest coefficient
/// magnitude, unless that is within `[1/8, 8]` (`g` unchanged).  Exact in
/// every coefficient type, and `g ⪰ 0 ⇔ 2^{-k} g ⪰ 0`.  A localizer with
/// coefficients ~1e-9 needs a Gram matrix ~1e9 on the SOS side, which an
/// interior-point solver does not reach: the bound stayed at the
/// unconstrained −2.236 instead of −0.134.  The certificate's
/// localizers are the normalized polynomials.
fn power_of_two_normalized<C: Field>(g: &Polynomial<C>) -> Polynomial<C> {
    let m = g.iter().map(|(_, c)| c.magnitude()).fold(0.0_f64, f64::max);
    if m == 0.0 || !m.is_finite() || (0.125..=8.0).contains(&m) {
        return g.clone();
    }
    let k = m.log2().round() as i64;
    let mut p = C::one();
    let two = C::from_i64(2);
    for _ in 0..k.unsigned_abs() {
        p = p.mul_ref(&two);
    }
    let factor = if k > 0 { p.inverse() } else { p };
    g.scale(&factor)
}

/// Group average of a Gram matrix on whose index words the group acts
/// monomially (`NF(g(u_a)) = c_a u_{π(a)}`, one `(π, c)` per element):
/// `G_avg[π(a), π(b)] += conj(c_a) c_b G[a, b] / |G|`.  Congruent modulo the
/// ideal to averaging the squares of `G`, at `|G| n²` cost.
/// The images may lie outside the index set: `π` maps into `0..n_out`.
fn average_gram(g: &[Vec<C64>], action: &[(Vec<usize>, Vec<C64>)], n_out: usize) -> Vec<Vec<C64>> {
    let n = g.len();
    let inv = 1.0 / action.len() as f64;
    let mut avg = vec![vec![C64::new(0.0, 0.0); n_out]; n_out];
    for (perm, coef) in action {
        for a in 0..n {
            let ca = coef[a].conj() * inv;
            for b in 0..n {
                let v = g[a][b];
                if v.re != 0.0 || v.im != 0.0 {
                    avg[perm[a]][perm[b]] += ca * coef[b] * v;
                }
            }
        }
    }
    avg
}

/// The group's action on `words` in the free algebra: the image words (the
/// words first, then new images) and per element the index and sign of
/// each word's image.  `None` past [`FREE_IMAGE_WORDS`] image words, where
/// the dense averaged Gram matrix and its eigendecomposition would cost more
/// than averaging the squares.
#[allow(clippy::type_complexity)]
fn free_word_action(
    group: &[SignedPermutation],
    words: &[Word],
) -> Option<(Vec<Word>, Vec<(Vec<usize>, Vec<C64>)>)> {
    let mut images: Vec<Word> = words.to_vec();
    let mut index: HashMap<Word, usize> = words.iter().cloned().enumerate().map(|(i, w)| (w, i)).collect();
    let mut action = Vec::with_capacity(group.len());
    for g in group {
        let mut perm = Vec::with_capacity(words.len());
        let mut coef = Vec::with_capacity(words.len());
        for w in words {
            let (sign, img) = g.apply_word(w);
            let i = *index.entry(img).or_insert_with_key(|img| {
                images.push(img.clone());
                images.len() - 1
            });
            if images.len() > FREE_IMAGE_WORDS {
                return None;
            }
            perm.push(i);
            coef.push(C64::new(f64::from(sign), 0.0));
        }
        action.push((perm, coef));
    }
    Some((images, action))
}

/// Cap of [`free_word_action`]: a dense `4096²` Gram matrix is 270 MB.
const FREE_IMAGE_WORDS: usize = 4096;

fn gram_to_squares(
    alg: &FreeAlgebra,
    words: &[Word],
    g: &[Vec<C64>],
    source: usize,
    tol: f64,
) -> Vec<Square> {
    let m = words.len();
    assert_eq!(g.len(), m, "Gram matrix size does not match the index set");
    let mut mat = Mat::<C64>::zeros(m, m);
    for a in 0..m {
        for b in 0..m {
            // symmetrize to guard against solver round-off
            mat[(a, b)] = (g[a][b] + g[b][a].conj()) * 0.5;
        }
    }
    let (vals, vecs) = hermitian_eigen(&mat);
    let max = vals.iter().fold(0.0_f64, |x, v| x.max(v.abs()));
    let mut squares = Vec::new();
    for k in 0..m {
        let lam = vals[k];
        if lam <= tol * max || lam <= 0.0 {
            continue;
        }
        let poly = Polynomial::from_terms(
            alg.clone(),
            (0..m).filter_map(|b| {
                let v = vecs[(b, k)].conj();
                (v.norm() > 1e-14).then(|| (words[b].clone(), v))
            }),
        );
        squares.push(Square {
            weight: lam,
            source,
            poly,
        });
    }
    squares
}

impl SosModel {
    /// Extract the certificate from a solution `x` of the Gram model.
    /// Eigenvalues of the Gram matrices below `tol` (relative) are dropped.
    pub fn certificate(&self, x: &[f64], tol: f64) -> Certificate {
        // a symmetry-reduced identity holds for the group average: Gram-level
        // (as MomentSdp::certificate), multipliers by symmetrize
        let mut averaged = Vec::new();
        let mut squares = Vec::new();
        for s in 0..self.source_words.len() {
            if !self.gram_sources.contains(&s) {
                continue;
            }
            let gm = self.gram_value(s, x);
            match self.index_action.get(s).and_then(|a| a.as_ref()) {
                Some(action) if !self.group.is_empty() => {
                    let avg = average_gram(&gm, action, gm.len());
                    averaged.extend(gram_to_squares(&self.alg, &self.source_words[s], &avg, s, tol));
                }
                // over the free algebra's image words
                _ if !self.group.is_empty() => match free_word_action(&self.group, &self.source_words[s]) {
                    Some((images, action)) => {
                        let avg = average_gram(&gm, &action, images.len());
                        averaged.extend(gram_to_squares(&self.alg, &images, &avg, s, tol));
                    }
                    None => squares.extend(gram_to_squares(&self.alg, &self.source_words[s], &gm, s, tol)),
                },
                _ => squares.extend(gram_to_squares(&self.alg, &self.source_words[s], &gm, s, tol)),
            }
        }
        // target + Σ μ_j r_j = Σ G, i.e. target = Σ G + Σ (−μ_j) r_j
        let multipliers = self
            .multipliers
            .iter()
            .zip(&self.multiplier_polynomials)
            .map(|(mu, r)| (-x[*mu], r.clone()))
            .collect();
        let mut cert = Certificate {
            squares,
            localizers: self.localizers.clone(),
            multipliers,
            minimize: self.minimize,
            alg: self.alg.clone(),
        }
        .symmetrize(&self.group);
        averaged.append(&mut cert.squares);
        cert.squares = averaged;
        cert
    }
}

impl MomentSdp {
    /// The symmetry group used (empty if none).
    pub fn group(&self) -> &[SignedPermutation] {
        &self.group
    }

    /// Per moment constraint (in the order they were added): the words of its
    /// normal form that no entry of the moment or localizing matrices
    /// determines.  Moment words are discovered lazily, so such a word is a
    /// new free variable and the constraint restricts nothing but it.  Check
    /// `free_constraint_words().iter().filter(|w| !w.is_empty()).count()`
    /// after building, or [`Relaxation::constraint_coverage`] before.
    pub fn free_constraint_words(&self) -> &[Vec<Word>] {
        &self.free_constraint_words
    }

    /// Hermitian Gram matrices, one per source matrix, from the solver's
    /// dual matrices (one per exported block, in the order of
    /// [`blocks`](Self::blocks)).  Each dual may be given as the real
    /// embedded matrix (size `2m`, or `m` for real blocks) or directly as the
    /// `m × m` Hermitian matrix.  Block-diagonalized blocks are lifted back
    /// to the coordinates of their source matrix, once each:
    /// `G_s += conj(V) Y_b Vᵀ` with `V` the [`block_vectors`](Self::block_vectors).
    /// The lift pairs with every coefficient matrix of the source exactly as
    /// `Y_b` pairs with the reduced one, so it is a valid Gram matrix for the
    /// program as exported (each reduced block counted once, not
    /// `irrep_dimension` times).  It is not group-invariant when a block has
    /// `irrep_dimension > 1`; averaging it over the group gives an invariant
    /// Gram matrix with the same pairings.
    pub fn dual_grams(&self, duals: &[Vec<Vec<C64>>]) -> Vec<Vec<Vec<C64>>> {
        assert_eq!(
            duals.len(),
            self.blocks.len(),
            "one dual matrix per exported block expected"
        );
        let zero = C64::new(0.0, 0.0);
        let mut grams: Vec<Vec<Vec<C64>>> = self
            .source_words
            .iter()
            .map(|w| vec![vec![zero; w.len()]; w.len()])
            .collect();
        for (b, info) in self.blocks.iter().enumerate() {
            let y = &duals[b];
            let m = info.size;
            let herm: Vec<Vec<C64>> = if y.len() == m {
                crate::sdp::complete_triangle(y)
            } else if y.len() == 2 * m && !info.real {
                unembed_dual(y, m)
            } else {
                panic!(
                    "dual matrix {} has size {}, expected {} (Hermitian) or {} (embedded)",
                    b,
                    y.len(),
                    m,
                    2 * m
                );
            };
            let s = info.source;
            match &self.block_vectors[b] {
                None => {
                    for a in 0..m {
                        for c in 0..m {
                            grams[s][a][c] += herm[a][c];
                        }
                    }
                }
                Some(vecs) => {
                    // G += conj(V) Y Vᵀ with V = [v_k] (columns in source
                    // coordinates), over the supports of the v_k only (they lie
                    // in word orbits for signed permutations)
                    let support: Vec<Vec<(usize, C64)>> = vecs
                        .iter()
                        .map(|v| {
                            v.iter()
                                .enumerate()
                                .filter(|(_, c)| c.norm() != 0.0)
                                .map(|(a, c)| (a, *c))
                                .collect()
                        })
                        .collect();
                    for k in 0..m {
                        for l in 0..m {
                            let y_kl = herm[k][l];
                            if y_kl.norm() == 0.0 {
                                continue;
                            }
                            for &(a, vka) in &support[k] {
                                let f = vka.conj() * y_kl;
                                for &(c, vlc) in &support[l] {
                                    grams[s][a][c] += f * vlc;
                                }
                            }
                        }
                    }
                }
            }
        }
        // over a real coefficient field the identity must be real: the imaginary
        // (antisymmetric) part of a lifted Gram matrix pairs to zero with every
        // real F_j but is not zero modulo the ideal, so drop it (Re G ⪰ 0 still)
        if !self.complex_moments {
            for g in &mut grams {
                for row in g.iter_mut() {
                    for v in row.iter_mut() {
                        v.im = 0.0;
                    }
                }
            }
        }
        grams
    }

    /// Certificate from the dual matrices of the exported SDP (see
    /// [`dual_grams`](Self::dual_grams)); with a symmetry group the result is
    /// averaged over the group.  Eigenvalues below `tol` (relative) are dropped.
    ///
    /// The moment constraints are solved out in this form, so the
    /// certificate has no multiplier terms: with
    /// [`moment_equality`](Relaxation::moment_equality) constraints its
    /// identity holds only modulo the constraint polynomials and
    /// [`Certificate::verify`] reports that as residual.  Use
    /// [`Relaxation::sos_model`] for a certificate with multipliers.
    pub fn certificate(&self, duals: &[Vec<Vec<C64>>], tol: f64) -> Certificate {
        let grams = self.dual_grams(duals);
        // Sources on which the group acts monomially are averaged at the level
        // of the Gram matrix, G_avg[π(a), π(b)] += conj(c_a) c_b G[a, b] / |G|,
        // which is congruent modulo the ideal to the average of the squares
        // and costs |G| n² instead of |G| times the squares; the squares
        // of the other sources are averaged as before.
        let mut squares = Vec::new();
        let mut to_symmetrize = Vec::new();
        for (s, g) in grams.iter().enumerate() {
            match self.index_action.get(s).and_then(|a| a.as_ref()) {
                Some(action) if !self.group.is_empty() => {
                    let avg = average_gram(g, action, g.len());
                    squares.extend(gram_to_squares(&self.alg, &self.source_words[s], &avg, s, tol));
                }
                _ if !self.group.is_empty() => match free_word_action(&self.group, &self.source_words[s]) {
                    // a signed permutation maps words to signed words of the
                    // free algebra: the average of the squares is the squares
                    // of the averaged Gram matrix on the image words, exactly
                    // and without reduction (averaging |G| copies of every
                    // square instead took 3.3 s and 2 GB of terms for the
                    // QMC 11-cycle)
                    Some((images, action)) => {
                        let avg = average_gram(g, &action, images.len());
                        squares.extend(gram_to_squares(&self.alg, &images, &avg, s, tol));
                    }
                    None => {
                        to_symmetrize.extend(gram_to_squares(&self.alg, &self.source_words[s], g, s, tol))
                    }
                },
                _ => to_symmetrize.extend(gram_to_squares(&self.alg, &self.source_words[s], g, s, tol)),
            }
        }
        let rest = Certificate {
            squares: to_symmetrize,
            localizers: self.localizers.clone(),
            multipliers: Vec::new(),
            minimize: self.minimize,
            alg: self.alg.clone(),
        }
        .symmetrize(&self.group);
        squares.extend(rest.squares);
        Certificate {
            squares,
            localizers: self.localizers.clone(),
            multipliers: Vec::new(),
            minimize: self.minimize,
            alg: self.alg.clone(),
        }
    }
}

/// Read dual matrices written by the solver scripts (`--duals` /
/// extra path argument): blocks of the form
///
/// ```text
/// block <index> <size> real|complex
/// <size rows of size numbers, or size (re im) pairs>
/// ```
pub fn read_dual_blocks(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<Vec<Vec<C64>>>> {
    let text = std::fs::read_to_string(path)?;
    let mut blocks: Vec<Vec<Vec<C64>>> = Vec::new();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let bad = |msg: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string());
    while let Some(header) = lines.next() {
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() < 4 || parts[0] != "block" {
            return Err(bad(&format!("expected a block header, found: {}", header)));
        }
        let n: usize = parts[2].parse().map_err(|_| bad("bad block size"))?;
        let complex = parts[3] == "complex";
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let line = lines.next().ok_or_else(|| bad("truncated block"))?;
            let nums: Vec<f64> = line
                .split_whitespace()
                .map(|t| t.parse::<f64>().map_err(|_| bad("bad number")))
                .collect::<Result<_, _>>()?;
            let row: Vec<C64> = if complex {
                if nums.len() != 2 * n {
                    return Err(bad("wrong row length"));
                }
                nums.chunks(2).map(|p| C64::new(p[0], p[1])).collect()
            } else {
                if nums.len() != n {
                    return Err(bad("wrong row length"));
                }
                nums.iter().map(|v| C64::new(*v, 0.0)).collect()
            };
            rows.push(row);
        }
        blocks.push(rows);
    }
    Ok(blocks)
}

impl MomentSdp {
    /// Solve the relaxation with the native interior-point solver
    /// ([`crate::solver`]).  Returns the relaxation value (the bound on
    /// `L(h)`) and the solution, whose `duals` feed
    /// [`certificate`](Self::certificate) and whose `x` feeds
    /// [`moments`](Self::moments).  The value is `NaN` unless the solution
    /// is `Optimal` or converged to `1e-5` in gap and infeasibilities
    /// ([`SdpSolution::is_converged`](crate::solver::SdpSolution::is_converged)): the objective of an unconverged
    /// iterate is not a bound.
    pub fn solve(
        &self,
        opts: &crate::solver::SolverOptions,
    ) -> Result<(f64, crate::solver::SdpSolution), crate::solver::SolverError> {
        let sol = self.to_model().solve(opts)?;
        // the objective of an iterate that is not (nearly) feasible is not a
        // bound on anything: report NaN instead of a misleading number
        if !sol.is_converged(opts.accept_tol) {
            return Ok((f64::NAN, sol));
        }
        // a regularized solve reports a certified bound (in the model's
        // objective units, which are the relaxation's), not the value at z
        if sol.regularization.is_some() {
            return Ok((sol.objective, sol));
        }
        let optimum: f64 = self.objective.iter().zip(&sol.x).map(|(c, z)| c * z).sum();
        Ok((self.relaxation_value(optimum), sol))
    }
}

impl SdpScalar for crate::rational::HybridRational {
    const REAL: bool = true;
    fn to_complex_f64(&self) -> (f64, f64) {
        (self.to_f64(), 0.0)
    }
}
impl SdpScalar for Complex<crate::rational::HybridRational> {
    const REAL: bool = false;
    fn to_complex_f64(&self) -> (f64, f64) {
        (self.re.to_f64(), self.im.to_f64())
    }
}
