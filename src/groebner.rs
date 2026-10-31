//! Noncommutative Gröbner bases (Mora's algorithm / Buchberger procedure for
//! free associative algebras) with bounded, resumable computation.
//!
//! Coefficients: compute over an exact field (`BigRational`, `HybridRational`,
//! `Fp<P>`, cyclotomics).  Over `f64` / `Complex<f64>` the computation is not
//! backward stable, since the term order fixes every pivot regardless of
//! size, and no comparably stable algorithm is known for general ideals;
//! compute exactly and [`convert`](GroebnerBasis::convert) the basis instead.
//!
//! # Background
//!
//! For a two-sided ideal `I ⊂ K<X>` and an admissible monomial ordering, a
//! set `G ⊂ I` is a *Gröbner basis* if every leading word of an element of
//! `I` contains the leading word of some `g ∈ G` as a subword.  Gröbner bases
//! give unique normal forms and decide ideal membership; unlike the
//! commutative case they may be **infinite**, so the completion procedure
//! need not terminate.
//!
//! The procedure processes *obstructions* (noncommutative S-polynomials).
//! Two basis elements `g_i, g_j` with leading words `u, v` have an obstruction
//! for every proper overlap `u = a·c`, `v = c·b` (`c ≠ ε`, `a, b ≠ ε`):
//!
//! ```text
//!     S = g_i · b − a · g_j         (overlap word  w = a·c·b)
//! ```
//!
//! `G` is a Gröbner basis iff every such `S` reduces to zero modulo `G`
//! (Bergman's diamond lemma / Mora's theorem).  Inclusion obstructions
//! (`v` a subword of `u`) never arise here because the active basis is kept
//! *interreduced*: when a new element makes an old one's leading word
//! reducible, the old one is removed and its normal form re-inserted.
//!
//! # Bounded computation
//!
//! Obstructions are processed in order of increasing overlap-word length.
//! [`GroebnerOptions::max_degree`] stops the computation as soon as every
//! remaining obstruction has an overlap word longer than the bound.  The
//! result is then a **truncated Gröbner basis**: for a degree-compatible
//! ordering and homogeneous generators it gives correct normal forms for all
//! polynomials of degree ≤ bound; for inhomogeneous generators it is the
//! partial basis obtained from all obstructions up to that degree.  Limits on
//! the number of processed pairs, wall-clock time, basis size, and an
//! external cancellation flag are also available.  Partial results are
//! always returned, and a [`GroebnerSolver`] can be resumed with relaxed
//! limits.
//!
//! # Criteria
//!
//! A noncommutative chain criterion is used: an obstruction with overlap word
//! `w` is skipped when the leading word of some *other* active element occurs
//! strictly inside `w` (not at the two defining positions), because its
//! S-polynomial is then a combination of two S-polynomials with shorter
//! overlap words.  It can be disabled with
//! [`GroebnerOptions::chain_criterion`].

use crate::algebra::FreeAlgebra;
use crate::coeff::{Coeff, CoeffConvert, Field};
use crate::poly::Polynomial;
use crate::quotient::QuotientAlgebra;
use crate::reduce::{Accumulator, Reducer, ReductionMode};
use crate::word::{proper_overlaps, Var, Word};
use num_rational::BigRational;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Options controlling a Gröbner basis computation.
///
/// All limits are optional; with none set the computation runs until a
/// complete basis is found (which may be never for ideals with infinite
/// bases — set at least one bound in that case).
#[derive(Clone, Debug)]
pub struct GroebnerOptions {
    /// Ignore obstructions whose overlap word is longer than this.  The
    /// result is a truncated (degree-bounded) Gröbner basis.
    pub max_degree: Option<usize>,
    /// Stop after this many obstructions have been reduced (per `run`).
    pub max_pairs: Option<usize>,
    /// Stop when the active basis reaches this many elements.
    pub max_basis_size: Option<usize>,
    /// Wall-clock limit (per `run`).
    pub time_limit: Option<Duration>,
    /// Cooperative cancellation: set the flag to `true` from another thread
    /// to stop at the next opportunity.
    pub cancel: Option<Arc<AtomicBool>>,
    /// Use the chain criterion to skip redundant obstructions (default true).
    pub chain_criterion: bool,
    /// Tail-reduce the output so it is the *reduced* Gröbner basis
    /// (default true).
    pub interreduce: bool,
    /// Print progress to stderr.
    pub verbose: bool,
    /// Reduce the obstructions of each degree in parallel batches (rayon;
    /// default true).  The result is deterministic and, for complete
    /// computations, identical to the sequential one (the reduced Gröbner
    /// basis is unique); partial bases may differ in which representatives
    /// are found first.  Thread count: `RAYON_NUM_THREADS`.
    pub parallel: bool,
    /// Floating-point coefficients only (`f64`, `Complex<f64>`; default
    /// `1e-10`, `0` disables): a reduced obstruction whose coefficients are
    /// all at most this fraction of the largest coefficient of what was
    /// reduced is rounding residue, and counts as zero.
    ///
    /// The per-merge cancellation test is relative to the summands of
    /// each addition, which already carry the error of earlier steps; along
    /// long reduction chains residue passes it.  Made monic, it becomes a
    /// spurious basis element whose coefficients later grow by up to 10²⁰,
    /// and the basis can collapse to the unit ideal: the qubit swap algebras
    /// on 7 and 8 qubits did, where the exact bases have 796 and 2052
    /// elements.  Observed residues were 10⁻¹⁹ to 10⁻¹² of their input, and
    /// remainders of exact relations are of the input's order.  The test is
    /// relative, so unlike [`crate::set_float_tolerance`] it does not depend
    /// on the scale of the data.  Lower it for relations whose coefficients
    /// genuinely span more than ten orders of magnitude.
    pub float_residue: f64,
}

impl Default for GroebnerOptions {
    fn default() -> Self {
        GroebnerOptions {
            max_degree: None,
            max_pairs: None,
            max_basis_size: None,
            time_limit: None,
            cancel: None,
            chain_criterion: true,
            interreduce: true,
            verbose: false,
            parallel: true,
            float_residue: crate::poly::FLOAT_RESIDUE,
        }
    }
}

impl GroebnerOptions {
    /// The default options.
    pub fn new() -> Self {
        Self::default()
    }
    /// See [`GroebnerOptions::max_degree`].
    pub fn max_degree(mut self, d: usize) -> Self {
        self.max_degree = Some(d);
        self
    }
    /// See [`GroebnerOptions::max_pairs`].
    pub fn max_pairs(mut self, n: usize) -> Self {
        self.max_pairs = Some(n);
        self
    }
    /// See [`GroebnerOptions::max_basis_size`].
    pub fn max_basis_size(mut self, n: usize) -> Self {
        self.max_basis_size = Some(n);
        self
    }
    /// See [`GroebnerOptions::float_residue`].
    pub fn float_residue(mut self, r: f64) -> Self {
        self.float_residue = r;
        self
    }
    /// See [`GroebnerOptions::time_limit`].
    pub fn time_limit(mut self, d: Duration) -> Self {
        self.time_limit = Some(d);
        self
    }
    /// See [`GroebnerOptions::cancel`].
    pub fn cancel_token(mut self, flag: Arc<AtomicBool>) -> Self {
        self.cancel = Some(flag);
        self
    }
    /// See [`GroebnerOptions::chain_criterion`].
    pub fn chain_criterion(mut self, on: bool) -> Self {
        self.chain_criterion = on;
        self
    }
    /// See [`GroebnerOptions::interreduce`].
    pub fn interreduce(mut self, on: bool) -> Self {
        self.interreduce = on;
        self
    }
    /// See [`GroebnerOptions::verbose`].
    pub fn verbose(mut self, on: bool) -> Self {
        self.verbose = on;
        self
    }
    /// See [`GroebnerOptions::parallel`].
    pub fn parallel(mut self, on: bool) -> Self {
        self.parallel = on;
        self
    }
}

/// Why a computation stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Termination {
    /// Every obstruction reduced to zero: the basis is a complete Gröbner basis.
    Complete,
    /// All remaining obstructions exceed the degree bound.
    DegreeBound {
        /// The bound that was applied.
        bound: usize,
        /// Overlap-word length of the smallest pending obstruction.
        smallest_pending: usize,
    },
    /// The pair limit was reached.
    PairLimit,
    /// The basis-size limit was reached.
    BasisSizeLimit,
    /// The time limit was reached.
    TimeLimit,
    /// The cancellation flag was set.
    Canceled,
    /// No `run` has happened yet.
    NotStarted,
}

impl Termination {
    /// Is the basis a complete Gröbner basis?
    pub fn is_complete(&self) -> bool {
        matches!(self, Termination::Complete)
    }
}

impl fmt::Display for Termination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Termination::Complete => write!(f, "complete"),
            Termination::DegreeBound {
                bound,
                smallest_pending,
            } => write!(
                f,
                "partial: degree bound {} reached (smallest pending obstruction has degree {})",
                bound, smallest_pending
            ),
            Termination::PairLimit => write!(f, "partial: pair limit reached"),
            Termination::BasisSizeLimit => write!(f, "partial: basis size limit reached"),
            Termination::TimeLimit => write!(f, "partial: time limit reached"),
            Termination::Canceled => write!(f, "partial: canceled"),
            Termination::NotStarted => write!(f, "not started"),
        }
    }
}

/// Counters describing the work done.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroebnerStats {
    /// Obstructions whose S-polynomial was reduced.
    pub pairs_reduced: usize,
    /// … of which reduced to zero.
    pub pairs_reduced_to_zero: usize,
    /// Obstructions skipped by the chain criterion.
    pub pairs_skipped_criterion: usize,
    /// Obstructions dropped because one of their elements became redundant.
    pub pairs_dropped_redundant: usize,
    /// Elements inserted into the basis (including later-removed ones).
    pub elements_added: usize,
    /// Elements removed as redundant.
    pub elements_removed: usize,
    /// Total reduction steps performed.
    pub reduction_steps: usize,
    /// Total time spent in `run`.
    pub elapsed: Duration,
    /// Obstructions still pending.
    pub pending_pairs: usize,
    /// Largest overlap-word length processed so far.
    pub max_degree_processed: usize,
}

/// An obstruction between active elements `i` and `j`:
/// suffix of `lw_i` of length `overlap` equals prefix of `lw_j`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pair {
    /// Length of the overlap word (primary priority, smaller first).
    deg: usize,
    /// Insertion sequence number (secondary: FIFO).
    seq: u64,
    i: usize,
    j: usize,
    overlap: usize,
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deg.cmp(&other.deg).then_with(|| self.seq.cmp(&other.seq))
    }
}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Incremental Gröbner basis computation that can be paused and resumed.
///
/// ```
/// use ncalg::{FreeAlgebra, Polynomial, GroebnerSolver, GroebnerOptions, BigRational};
/// let alg = FreeAlgebra::new(["x", "y"]);
/// // <x y x - x y> has an infinite Gröbner basis {x y^n x - x y^n}
/// let f: Polynomial<BigRational> = alg.parse("x*y*x - x*y").unwrap();
/// let mut solver = GroebnerSolver::new(vec![f], GroebnerOptions::new().max_degree(5));
/// let status = solver.run();
/// assert!(!status.is_complete());
/// assert_eq!(solver.basis().len(), 2); // n = 1, 2 (overlap words of length <= 5)
/// // resume with a larger bound
/// solver.set_options(GroebnerOptions::new().max_degree(7));
/// solver.run();
/// assert_eq!(solver.basis().len(), 4); // n = 1..=4
/// ```
pub struct GroebnerSolver<C: Field> {
    alg: FreeAlgebra,
    opts: GroebnerOptions,
    /// Every element ever inserted (monic).  Index = element id.
    elems: Vec<Polynomial<C>>,
    active: Vec<bool>,
    n_active: usize,
    pairs: BinaryHeap<Reverse<Pair>>,
    seq: u64,
    /// Polynomials waiting to be reduced and inserted (initial generators and
    /// normal forms of removed elements).
    pending: VecDeque<Polynomial<C>>,
    /// The working reducer on the active elements, maintained incrementally
    /// (rebuilding it from clones after every insertion was most of the
    /// run time), and the reducer index of each element.
    reducer: Option<Reducer<C>>,
    red_id: Vec<Option<usize>>,
    stats: GroebnerStats,
    status: Termination,
    /// The ideal contains 1.
    unit: bool,
}

impl<C: Field> fmt::Debug for GroebnerSolver<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GroebnerSolver({} active elements, {} pending pairs, {})",
            self.n_active,
            self.pairs.len(),
            self.status
        )
    }
}

impl<C: Field> GroebnerSolver<C> {
    /// Create a solver for the ideal generated by `generators`.  Nothing is
    /// computed until [`run`](Self::run) is called.  Panics if the generators
    /// live in different algebras or if there are none.
    pub fn new(generators: Vec<Polynomial<C>>, opts: GroebnerOptions) -> Self {
        let alg = generators
            .first()
            .map(|p| p.algebra().clone())
            .expect("GroebnerSolver::new needs at least one generator (use new_in for an empty set)");
        Self::new_in(&alg, generators, opts)
    }

    /// Like [`new`](Self::new) but with an explicit algebra, allowing an
    /// empty generator list.
    pub fn new_in(alg: &FreeAlgebra, generators: Vec<Polynomial<C>>, opts: GroebnerOptions) -> Self {
        for g in &generators {
            alg.check_same(g.algebra());
        }
        GroebnerSolver {
            alg: alg.clone(),
            opts,
            elems: Vec::new(),
            active: Vec::new(),
            n_active: 0,
            pairs: BinaryHeap::new(),
            seq: 0,
            pending: generators.into_iter().filter(|g| !g.is_zero()).collect(),
            reducer: None,
            red_id: Vec::new(),
            stats: GroebnerStats::default(),
            status: Termination::NotStarted,
            unit: false,
        }
    }

    /// Replace the options (e.g. to raise a bound before resuming).
    pub fn set_options(&mut self, opts: GroebnerOptions) {
        self.opts = opts;
    }

    /// Current options.
    pub fn options(&self) -> &GroebnerOptions {
        &self.opts
    }

    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// Add more generators to the ideal; the computation becomes incomplete
    /// until `run` is called again.
    pub fn add_generators(&mut self, generators: impl IntoIterator<Item = Polynomial<C>>) {
        for g in generators {
            self.alg.check_same(g.algebra());
            if !g.is_zero() {
                self.pending.push_back(g);
            }
        }
        if !self.pending.is_empty() {
            self.status = Termination::NotStarted;
        }
    }

    /// Work statistics so far.
    pub fn stats(&self) -> &GroebnerStats {
        &self.stats
    }

    /// Status of the last `run`.
    pub fn status(&self) -> &Termination {
        &self.status
    }

    /// Has a complete Gröbner basis been found?
    pub fn is_complete(&self) -> bool {
        self.status.is_complete()
    }

    /// Number of obstructions still pending.
    pub fn pending_pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Overlap-word length of the smallest pending obstruction.
    pub fn min_pending_degree(&self) -> Option<usize> {
        self.pairs.peek().map(|p| p.0.deg)
    }

    /// The current active elements (monic, leading words pairwise
    /// subword-free, *not* tail-reduced).
    pub fn active_elements(&self) -> Vec<Polynomial<C>> {
        self.elems
            .iter()
            .zip(&self.active)
            .filter(|(_, &a)| a)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Snapshot of the current basis (interreduced if the options say so),
    /// together with the status of the last run.
    pub fn basis(&self) -> GroebnerBasis<C> {
        let mut elems = self.active_elements();
        if self.unit {
            elems = vec![self.alg.one()];
        } else if self.opts.interreduce && elems.len() > 1 {
            let red = Reducer::new(&self.alg, elems.clone());
            elems = elems.iter().map(|g| red.reduce_tail(g)).collect();
        }
        let alg = &self.alg;
        elems.sort_by(|a, b| {
            alg.cmp_words(
                a.leading_word().expect("nonzero"),
                b.leading_word().expect("nonzero"),
            )
        });
        let mut stats = self.stats.clone();
        stats.pending_pairs = self.pairs.len();
        GroebnerBasis::from_parts(&self.alg, elems, self.status.clone(), stats)
    }

    // ---------------------------------------------------------------------
    // the algorithm
    // ---------------------------------------------------------------------

    fn reducer(&mut self) -> &Reducer<C> {
        if self.reducer.is_none() {
            let mut red = Reducer::new_dynamic(&self.alg);
            self.red_id = vec![None; self.elems.len()];
            for (k, g) in self.elems.iter().enumerate() {
                if self.active[k] {
                    self.red_id[k] = Some(red.push(g.clone()));
                }
            }
            self.reducer = Some(red);
        }
        self.reducer.as_ref().expect("just built")
    }

    fn log(&self, msg: impl FnOnce() -> String) {
        if self.opts.verbose {
            eprintln!("[ncalg] {}", msg());
        }
    }

    /// Insert a nonzero polynomial that is already fully reduced modulo the
    /// active basis.  Removes elements made redundant and schedules their
    /// normal forms for re-insertion.  Generates the new obstructions.
    fn insert(&mut self, mut h: Polynomial<C>) {
        h.make_monic();
        let lw = h.leading_word().expect("nonzero").clone();
        if lw.is_empty() {
            // the ideal is the whole algebra
            self.unit = true;
            self.pairs.clear();
            self.pending.clear();
            self.active.fill(false);
            self.n_active = 0;
            self.elems.push(h);
            self.active.push(true);
            self.n_active = 1;
            self.reducer = None;
            self.stats.elements_added += 1;
            return;
        }

        // remove elements whose leading word contains lw
        let mut removed = Vec::new();
        for (k, g) in self.elems.iter().enumerate() {
            if self.active[k] && lw.is_subword_of(g.leading_word().expect("nonzero")) {
                removed.push(k);
            }
        }
        for &k in &removed {
            self.active[k] = false;
            self.n_active -= 1;
            self.stats.elements_removed += 1;
        }

        let id = self.elems.len();
        self.elems.push(h);
        self.active.push(true);
        self.n_active += 1;
        self.stats.elements_added += 1;
        // keep the working reducer in step instead of rebuilding it
        self.red_id.resize(self.elems.len(), None);
        if let Some(red) = self.reducer.as_mut() {
            for &k in &removed {
                if let Some(r) = self.red_id[k].take() {
                    red.retire(r);
                }
            }
            self.red_id[id] = Some(red.push(self.elems[id].clone()));
        }

        // obstructions with every active element, including itself
        let lw_new = self.elems[id].leading_word().expect("nonzero").clone();
        for k in 0..self.elems.len() {
            if !self.active[k] {
                continue;
            }
            let lw_k = self.elems[k].leading_word().expect("nonzero").clone();
            if k == id {
                for c in proper_overlaps(&lw_new, &lw_new) {
                    self.push_pair(id, id, c, lw_new.len() + lw_new.len() - c);
                }
            } else {
                for c in proper_overlaps(&lw_new, &lw_k) {
                    self.push_pair(id, k, c, lw_new.len() + lw_k.len() - c);
                }
                for c in proper_overlaps(&lw_k, &lw_new) {
                    self.push_pair(k, id, c, lw_k.len() + lw_new.len() - c);
                }
            }
        }

        // re-insert the removed elements (they are still in the ideal and may
        // contribute information not captured by the new element)
        for k in removed {
            let g = self.elems[k].clone();
            self.pending.push_back(g);
        }
        let lw_disp = self.alg.format_word(&lw_new);
        let n_active = self.n_active;
        let n_pairs = self.pairs.len();
        self.log(|| {
            format!(
                "added element with leading word {} ({} active, {} pending pairs)",
                lw_disp, n_active, n_pairs
            )
        });
    }

    fn push_pair(&mut self, i: usize, j: usize, overlap: usize, deg: usize) {
        self.seq += 1;
        self.pairs.push(Reverse(Pair {
            deg,
            seq: self.seq,
            i,
            j,
            overlap,
        }));
    }

    /// Chain criterion: does the leading word of some active element occur
    /// strictly inside the overlap word, away from the two defining positions?
    fn chain_criterion_applies(&mut self, pair: &Pair, w: &[Var]) -> bool {
        let len_i = self.elems[pair.i].leading_word().expect("nonzero").len();
        let len_j = self.elems[pair.j].leading_word().expect("nonzero").len();
        let total = w.len();
        self.reducer();
        let red = self.reducer.as_ref().expect("built");
        for (_, start, len) in red.occurrences(w) {
            let is_left = start == 0 && len == len_i;
            let is_right = start + len == total && len == len_j;
            if !is_left && !is_right {
                return true;
            }
        }
        false
    }

    /// Reduce the S-polynomials of a batch of pairs against the current
    /// basis (in parallel when enabled) and return their normal forms with
    /// the reduction statistics, in the order of `batch`.
    fn reduce_batch(&mut self, batch: &[Pair]) -> Vec<(Polynomial<C>, crate::reduce::ReductionStats)> {
        self.reducer();
        let red = self.reducer.as_ref().expect("built");
        let elems = &self.elems;
        let work = |pair: &Pair| -> (Polynomial<C>, crate::reduce::ReductionStats) {
            let gi = &elems[pair.i];
            let gj = &elems[pair.j];
            let lwi = gi.leading_word().expect("nonzero");
            let lwj = gj.leading_word().expect("nonzero");
            let r: Vec<Var> = lwj[pair.overlap..].to_vec();
            let l: Vec<Var> = lwi[..lwi.len() - pair.overlap].to_vec();
            let mut acc = Accumulator::new(red.algebra());
            // S = g_i · r − l · g_j; leading terms cancel exactly, push only tails
            acc.push_scaled_terms(C::one(), &[], &gi.terms()[1..], &r);
            acc.push_scaled_terms(C::one().negated(), &l, &gj.terms()[1..], &[]);
            red.run(acc, ReductionMode::Full)
        };
        if self.opts.parallel && batch.len() > 1 {
            use rayon::prelude::*;
            batch.par_iter().map(work).collect()
        } else {
            batch.iter().map(work).collect()
        }
    }

    /// Reduce a pending polynomial and insert it if nonzero.  Returns whether
    /// an element was inserted.
    fn absorb(&mut self, p: Polynomial<C>) -> bool {
        if self.unit {
            return false;
        }
        // the tail's size: a reducible leading term cancels exactly; an
        // irreducible one stays in nf, which is then not small
        let scale = if C::INEXACT { tail_magnitude(&p) } else { 0.0 };
        let nf = if self.n_active == 0 {
            p
        } else {
            let red = self.reducer();
            let (nf, st) = red.reduce_with_stats(&p, ReductionMode::Full);
            self.stats.reduction_steps += st.steps;
            nf
        };
        if nf.is_zero() || self.is_residue(&nf, scale) {
            false
        } else {
            self.insert(nf);
            true
        }
    }

    /// See [`GroebnerOptions::float_residue`]: `nf` is rounding residue of
    /// a reduction whose input had largest coefficient `scale`.
    fn is_residue(&self, nf: &Polynomial<C>, scale: f64) -> bool {
        nf.is_rounding_residue(scale, self.opts.float_residue)
    }

    fn check_limits(&self, start: Instant, pairs_this_run: usize) -> Option<Termination> {
        if let Some(flag) = &self.opts.cancel {
            if flag.load(AtomicOrdering::Relaxed) {
                return Some(Termination::Canceled);
            }
        }
        if let Some(t) = self.opts.time_limit {
            if start.elapsed() >= t {
                return Some(Termination::TimeLimit);
            }
        }
        if let Some(n) = self.opts.max_pairs {
            if pairs_this_run >= n {
                return Some(Termination::PairLimit);
            }
        }
        if let Some(n) = self.opts.max_basis_size {
            if self.n_active >= n {
                return Some(Termination::BasisSizeLimit);
            }
        }
        None
    }

    /// Run the completion procedure until it finishes or a limit is hit.
    /// Can be called again after relaxing the limits.
    pub fn run(&mut self) -> Termination {
        let start = Instant::now();
        let mut pairs_this_run = 0usize;
        let status = loop {
            // absorb pending generators / re-insertions first
            while let Some(p) = self.pending.pop_front() {
                self.absorb(p);
            }
            if self.unit {
                break Termination::Complete;
            }
            // nothing left: complete, even if a limit was reached exactly
            // now (it used to report PairLimit / BasisSizeLimit)
            if self.pairs.is_empty() {
                break Termination::Complete;
            }
            if let Some(t) = self.check_limits(start, pairs_this_run) {
                break t;
            }
            // collect a batch of obstructions of the smallest pending degree
            // (a single one when running sequentially), dropping redundant
            // and criterion-eliminated pairs on the way
            let batch_limit = if self.opts.parallel {
                (2 * rayon::current_num_threads()).max(2)
            } else {
                1
            };
            let mut batch: Vec<Pair> = Vec::new();
            let mut stop: Option<Termination> = None;
            loop {
                let pair = match self.pairs.peek() {
                    None => {
                        if batch.is_empty() {
                            stop = Some(Termination::Complete);
                        }
                        break;
                    }
                    Some(Reverse(p)) => p.clone(),
                };
                if let Some(bound) = self.opts.max_degree {
                    if pair.deg > bound {
                        if batch.is_empty() {
                            stop = Some(Termination::DegreeBound {
                                bound,
                                smallest_pending: pair.deg,
                            });
                        }
                        break;
                    }
                }
                if let Some(first) = batch.first() {
                    if pair.deg != first.deg || batch.len() >= batch_limit {
                        break;
                    }
                    if let Some(n) = self.opts.max_pairs {
                        if pairs_this_run + batch.len() >= n {
                            break;
                        }
                    }
                }
                self.pairs.pop();
                if !self.active[pair.i] || !self.active[pair.j] {
                    self.stats.pairs_dropped_redundant += 1;
                    continue;
                }
                // overlap word
                let w: Word = {
                    let lwi = self.elems[pair.i].leading_word().expect("nonzero");
                    let lwj = self.elems[pair.j].leading_word().expect("nonzero");
                    Word::concat3(lwi, &lwj[pair.overlap..], &[])
                };
                if self.opts.chain_criterion && self.chain_criterion_applies(&pair, &w) {
                    self.stats.pairs_skipped_criterion += 1;
                    continue;
                }
                batch.push(pair);
            }
            if let Some(t) = stop {
                break t;
            }
            if batch.is_empty() {
                continue;
            }
            // reduce the batch against the current basis, then absorb the
            // nonzero results one by one (each is re-reduced against the
            // elements inserted before it in this batch)
            let results = self.reduce_batch(&batch);
            pairs_this_run += batch.len();
            self.stats.pairs_reduced += batch.len();
            let max_deg = batch.iter().map(|p| p.deg).max().unwrap_or(0);
            self.stats.max_degree_processed = self.stats.max_degree_processed.max(max_deg);
            for ((nf, st), pair) in results.into_iter().zip(&batch) {
                self.stats.reduction_steps += st.steps;
                // rounding residue relative to what was reduced: the
                // S-polynomial holds the two elements' tails only, their
                // leading terms cancel exactly.  With the whole elements
                // (monic, so scale ≥ 1) a genuine remainder ε² of tails of
                // size ε = 1e-6 was dropped and y went missing from the
                // basis of {x² − εz, zx − xz − εy}
                let residue = C::INEXACT && {
                    let scale = tail_magnitude(&self.elems[pair.i]).max(tail_magnitude(&self.elems[pair.j]));
                    self.is_residue(&nf, scale)
                };
                if nf.is_zero() || residue || !self.absorb(nf) {
                    self.stats.pairs_reduced_to_zero += 1;
                }
            }
        };
        self.stats.elapsed += start.elapsed();
        self.stats.pending_pairs = self.pairs.len();
        self.status = status.clone();
        self.log(|| format!("run finished: {} ({} active elements)", status, self.n_active));
        status
    }
}

/// A (possibly partial) Gröbner basis together with a reducer for computing
/// normal forms.
///
/// Obtain one with [`GroebnerBasis::compute`] or from a [`GroebnerSolver`].
#[derive(Clone)]
pub struct GroebnerBasis<C: Field> {
    alg: FreeAlgebra,
    elements: Vec<Polynomial<C>>,
    termination: Termination,
    stats: GroebnerStats,
    reducer: Reducer<C>,
}

impl<C: Field> fmt::Debug for GroebnerBasis<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GroebnerBasis({} elements, {})",
            self.elements.len(),
            self.termination
        )
    }
}

impl<C: Field> fmt::Display for GroebnerBasis<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gröbner basis in {} ({}):", self.alg, self.termination)?;
        for (i, g) in self.elements.iter().enumerate() {
            writeln!(f, "  [{}] {}", i, g)?;
        }
        Ok(())
    }
}

impl<C: Field> GroebnerBasis<C> {
    /// Compute a Gröbner basis of the ideal generated by `generators` with the
    /// given options.  Returns a partial basis if a limit is hit; check
    /// [`is_complete`](Self::is_complete).
    pub fn compute(generators: Vec<Polynomial<C>>, opts: &GroebnerOptions) -> Self {
        let given_unit = has_unit(&generators);
        let mut solver = GroebnerSolver::new(generators, opts.clone());
        solver.run();
        let gb = solver.basis();
        warn_float_collapse(&gb, given_unit);
        gb
    }

    /// Like [`compute`](Self::compute) with an explicit algebra (allows an
    /// empty generator set, whose basis is empty).
    pub fn compute_in(alg: &FreeAlgebra, generators: Vec<Polynomial<C>>, opts: &GroebnerOptions) -> Self {
        let given_unit = has_unit(&generators);
        let mut solver = GroebnerSolver::new_in(alg, generators, opts.clone());
        solver.run();
        let gb = solver.basis();
        warn_float_collapse(&gb, given_unit);
        gb
    }

    /// Wrap polynomials that are already known to form a Gröbner basis (or a
    /// partial one).  The polynomials are made monic; no verification is
    /// performed (use [`verify`](Self::verify)).
    pub fn from_elements(alg: &FreeAlgebra, elements: Vec<Polynomial<C>>, termination: Termination) -> Self {
        let elements = elements
            .into_iter()
            .filter(|p| !p.is_zero())
            .map(|p| p.monic())
            .collect();
        Self::from_parts(alg, elements, termination, GroebnerStats::default())
    }

    fn from_parts(
        alg: &FreeAlgebra,
        elements: Vec<Polynomial<C>>,
        termination: Termination,
        stats: GroebnerStats,
    ) -> Self {
        let reducer = Reducer::new(alg, elements.clone()).with_complete(termination.is_complete());
        GroebnerBasis {
            alg: alg.clone(),
            elements,
            termination,
            stats,
            reducer,
        }
    }

    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// The basis elements (monic, sorted by leading word).
    pub fn elements(&self) -> &[Polynomial<C>] {
        &self.elements
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// No elements (the zero ideal)?
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Leading words of the basis elements.
    pub fn leading_words(&self) -> Vec<Word> {
        self.elements
            .iter()
            .map(|g| g.leading_word().expect("nonzero").clone())
            .collect()
    }

    /// Why the computation stopped.
    pub fn termination(&self) -> &Termination {
        &self.termination
    }

    /// Is this a complete Gröbner basis?
    pub fn is_complete(&self) -> bool {
        self.termination.is_complete()
    }

    /// Work statistics.
    pub fn stats(&self) -> &GroebnerStats {
        &self.stats
    }

    /// Does the ideal contain 1?
    pub fn is_unit_ideal(&self) -> bool {
        self.reducer.is_unit_ideal()
    }

    /// The reducer (for custom lazy evaluations with an [`Accumulator`]).
    pub fn reducer(&self) -> &Reducer<C> {
        &self.reducer
    }

    /// Normal form of `p`.  Unique if the basis is complete (or complete up
    /// to the degree of `p` for homogeneous ideals under degree orders).
    pub fn reduce(&self, p: &Polynomial<C>) -> Polynomial<C> {
        self.reducer.reduce(p)
    }

    /// Normal form of `f · g`, reducing while expanding.
    pub fn reduce_product(&self, f: &Polynomial<C>, g: &Polynomial<C>) -> Polynomial<C> {
        self.reducer.reduce_product(f, g)
    }

    /// Normal form of `f^n`.
    pub fn reduce_pow(&self, f: &Polynomial<C>, n: u32) -> Polynomial<C> {
        self.reducer.reduce_pow(f, n)
    }

    /// Normal form of `[f, g]`.
    pub fn reduce_commutator(&self, f: &Polynomial<C>, g: &Polynomial<C>) -> Polynomial<C> {
        self.reducer.reduce_commutator(f, g)
    }

    /// Ideal membership test `p ∈ I`.  Exact when the basis is complete;
    /// otherwise `true` is certain but `false` may be a false negative.
    pub fn contains(&self, p: &Polynomial<C>) -> bool {
        self.reduce(p).is_zero()
    }

    /// Is `w` a normal word (not divisible by any leading word)?
    pub fn is_normal_word(&self, w: &[Var]) -> bool {
        self.reducer.is_normal_word(w)
    }

    /// Is the ideal closed under the Hermitian adjoint (reverse words and
    /// conjugate coefficients)?  Checked by reducing the adjoint of every basis
    /// element; exact for a complete basis.  When this holds the adjoint is
    /// well defined on the quotient and `QuotientElement::adjoint` is an
    /// anti-involution.
    pub fn is_star_closed(&self) -> bool {
        self.elements.iter().all(|g| self.reduce(&g.adjoint()).is_zero())
    }

    /// Independently check the Gröbner property by brute force: every
    /// obstruction with overlap word of length ≤ `max_degree` (all of them if
    /// `None`) must reduce to zero, and every element must be reduced modulo
    /// the others.  Returns the list of failing `(i, j, overlap)` triples;
    /// empty means verified.
    pub fn verify(&self, max_degree: Option<usize>) -> Vec<(usize, usize, usize)> {
        let mut failures = Vec::new();
        if self.reducer.is_unit_ideal() {
            return failures;
        }
        let n = self.elements.len();
        for i in 0..n {
            for j in 0..n {
                let gi = &self.elements[i];
                let gj = &self.elements[j];
                let lwi = gi.leading_word().expect("nonzero");
                let lwj = gj.leading_word().expect("nonzero");
                if i != j && lwi.is_subword_of(lwj) {
                    failures.push((i, j, 0));
                    continue;
                }
                for c in proper_overlaps(lwi, lwj) {
                    let deg = lwi.len() + lwj.len() - c;
                    if let Some(b) = max_degree {
                        if deg > b {
                            continue;
                        }
                    }
                    let r = &lwj[c..];
                    let l = &lwi[..lwi.len() - c];
                    let mut acc = Accumulator::new(&self.alg);
                    acc.push_scaled_terms(C::one(), &[], &gi.terms()[1..], r);
                    acc.push_scaled_terms(C::one().negated(), l, &gj.terms()[1..], &[]);
                    let (nf, _) = self.reducer.run(acc, ReductionMode::Full);
                    if !nf.is_zero() {
                        failures.push((i, j, c));
                    }
                }
            }
        }
        failures
    }

    /// The quotient algebra `K<X>/I` with arithmetic modulo this basis.
    pub fn quotient(self) -> QuotientAlgebra<C> {
        QuotientAlgebra::new(self)
    }

    /// Convert coefficients (e.g. to floating point for numerical work).
    pub fn convert<D: Field>(&self) -> GroebnerBasis<D>
    where
        C: CoeffConvert<D>,
    {
        GroebnerBasis::from_parts(
            &self.alg,
            self.elements.iter().map(|g| g.convert()).collect(),
            self.termination.clone(),
            self.stats.clone(),
        )
    }
}

impl GroebnerBasis<BigRational> {
    /// Gröbner basis over `Q` of an ideal given by integer polynomials.
    pub fn compute_from_integers<I>(generators: Vec<Polynomial<I>>, opts: &GroebnerOptions) -> Self
    where
        I: Coeff + CoeffConvert<BigRational>,
    {
        let gens: Vec<Polynomial<BigRational>> = generators.iter().map(|g| g.convert()).collect();
        Self::compute(gens, opts)
    }

    /// The basis elements scaled to primitive integer polynomials.
    pub fn integer_elements(&self) -> Vec<Polynomial<num_bigint::BigInt>> {
        self.elements.iter().map(|g| g.to_integer_primitive()).collect()
    }
}

/// Convenience: compute a Gröbner basis with the given options.
pub fn groebner_basis<C: Field>(generators: Vec<Polynomial<C>>, opts: &GroebnerOptions) -> GroebnerBasis<C> {
    GroebnerBasis::compute(generators, opts)
}

/// Largest coefficient magnitude of `p` without its leading term (0 for a
/// monomial or zero).
fn tail_magnitude<C: Coeff>(p: &Polynomial<C>) -> f64 {
    p.iter()
        .skip(1)
        .map(|(_, c)| c.magnitude())
        .fold(0.0_f64, f64::max)
}

/// Does the generating set contain a nonzero constant?
fn has_unit<C: Coeff>(generators: &[Polynomial<C>]) -> bool {
    generators
        .iter()
        .any(|g| !g.is_zero() && g.iter().all(|(w, _)| w.is_empty()))
}

/// A floating-point basis that ends at the unit ideal, from generators that
/// contain no constant, is most often rounding residue: the cancellation
/// test is relative to the summands of each merge, while errors accumulate
/// over long reduction chains (the 7-qubit swap algebra in `f64` collapses to
/// `{1}`, where the exact basis has 796 elements).  Say so loudly
/// instead of returning a silently wrong ideal.
fn warn_float_collapse<C: Field>(gb: &GroebnerBasis<C>, given_unit: bool) {
    if C::INEXACT && !given_unit && gb.reducer().is_unit_ideal() {
        eprintln!(
            "warning: ncalg: this floating-point Gröbner basis is the unit ideal {{1}}, from generators \
             without a constant; that is usually accumulated rounding residue, not a property of the \
             ideal.  Compute with exact coefficients (BigRational, HybridRational) or set an \
             absolute floor with ncalg::set_float_tolerance(1e-12)."
        );
    }
}
