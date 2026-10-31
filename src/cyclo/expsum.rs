//! Exact exponential sums of phase polynomials over `𝔽₂`:
//! `S(f, a + V) = Σ_{x ∈ a+V} ζ^{f(x)}`, `ζ = e^{2πi/2^r}`, in [`Cyclo<N>`].
//!
//! # Representation
//!
//! A [`PhasePolynomial`] on `x ∈ 𝔽₂^n` is `f = c + Σ_T c_T Π_{λ ∈ T} [λ·x]`
//! with coefficients mod `2^r`, where `[λ·x] ∈ {0, 1}` is the parity of the
//! bits selected by `λ ∈ 𝔽₂^n`, read as an integer.  Monomials `x_S` are the
//! case of unit vectors.  The form is closed under affine substitution
//! `x = a + M t`, which only maps each `λ` to `λM` (and turns `[λ·a ⊕ μ·t]`
//! into `1 − [μ·t]` when `λ·a = 1`), so [`restrict`](PhasePolynomial::restrict)
//! is cheap.  In monomials it is not: `x_1 = t_1 ⊕ t_2 ⊕ t_3` is
//! `t_1 + t_2 + t_3 − 2(t_1t_2 + t_1t_3 + t_2t_3) + 4 t_1t_2t_3`, so a T phase
//! (`ζ_8^{x_1}`) restricted to a subspace is cubic in general.
//!
//! The level of a term `c Π_{k ≤ s} [λ_k·x]` is `s + r − v₂(c) − 1` (with
//! `λ_k` unit vectors this is the Clifford-hierarchy level of the diagonal gate
//! `ζ^{c x_S}`: 1 for Z, 2 for S and CZ, 3 for T, CS and CCZ).  Affine
//! substitution never raises it, since `[⊕_{i∈S} t_i]` expands into monomials
//! of degree `d` with coefficients divisible by `2^{d−1}`.
//!
//! # The Clifford class: polynomial time
//!
//! When every term has level at most 2 — linear coefficients in `2^{r−2}ℤ`
//! (S, Z), quadratic in `2^{r−1}ℤ` (CZ), nothing of higher degree, any
//! constant — `ζ^{f} = ζ^{c} i^{q(x)}` with `q` a `ℤ₄`-valued quadratic form
//! `Σ a_j x_j + 2 Σ_{j<k} B_{jk} x_j x_k`.  The class is closed under
//! restriction, and `Σ_x i^{q(x)}` is evaluated by eliminating one variable at
//! a time:
//!
//! * `a_j` odd: `Σ_{x_j} i^{a_j x_j + 2 x_j L} = (1+i)·i^{(4−a_j)[L] + …}`, a rank-one
//!   update of `q` on the neighbors `L` of `x_j`;
//! * `a_j` even: the sum is `2·[L = a_j/2]`, an affine constraint, solved for one
//!   neighbor and substituted (or `0`, or `2` when `L` is empty).
//!
//! Each step is `O(n²/64)` word operations, so the sum costs `O(n³/64)`, and it
//! is `0` or `ζ^{c} i^{e} (1+i)^{p}`: `|S|² ∈ {0} ∪ 2^ℤ`
//! ([`clifford_sum`](PhasePolynomial::clifford_sum)).
//!
//! # Everything else: exact, exponential only in a width
//!
//! A term of level ≥ 3 is not Clifford.  [`exp_sum`](PhasePolynomial::exp_sum)
//! is exact for any polynomial:
//!
//! 1. Change variables so that the parities in non-Clifford terms depend on
//!    only `d` = rank of those parities coordinates (the pivots of their row
//!    echelon form); every other direction appears in Clifford terms only and is
//!    eliminated as above.  Substitutions that land on core variables are
//!    applied to the non-Clifford terms as well.
//! 2. On the remaining core, run variable elimination (sum–product) with a
//!    greedy min-size order, over tables of exact cyclotomic integers.
//!
//! The cost is polynomial plus `O(m · 2^w)` for `w` the largest table arity
//! along the order ([`exp_sum_plan`](PhasePolynomial::exp_sum_plan)), and
//! `w ≤ d ≤` the number of distinct non-Clifford parities.  So it is
//! polynomial for the Clifford class (`w = 0`), for product-type instances —
//! no couplings between non-Clifford variables, e.g. T on every qubit of a
//! product state (`w = 1`), for a few non-Clifford gates (`2^d`), and for
//! non-Clifford structure of bounded treewidth (a chain of CS gates: `w ≤ 3`).
//!
//! # Why not more
//!
//! The sum is a weighted #CSP over `𝔽₂` with constraint functions `ζ^{term}`
//! and `χ_{a+V}`.  By the dichotomy of Cai, Lu and Xia (complex-weighted
//! Boolean #CSP), such a problem is in polynomial time when all functions are
//! affine (`λ χ_{Ax=b} i^{Q(x)}` with `Q` quadratic over `ℤ₄` with even cross
//! terms: exactly the Clifford class) or all are of product type (unary
//! functions, equalities, disequalities), and #P-hard otherwise.  So any phase
//! type beyond the Clifford class — T (`ζ_8^{x}`), CS (`i^{xy}`), CCZ
//! (`(−1)^{xyz}`) — together with CZ couplings or general subspaces gives a
//! #P-hard family: no algorithm is polynomial on a larger class defined by the
//! allowed terms unless FP = #P.  The algorithm above is polynomial on both
//! tractable sides and degrades gracefully in between.

use super::Cyclo;
use crate::ff::BitMat;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use std::collections::BTreeMap;

type Bits = Vec<u64>;

fn cmul<const N: u32>(a: &Cyclo<N>, b: &Cyclo<N>) -> Cyclo<N> {
    <Cyclo<N> as crate::coeff::Coeff>::mul_ref(a, b)
}
fn cadd<const N: u32>(a: &Cyclo<N>, b: &Cyclo<N>) -> Cyclo<N> {
    <Cyclo<N> as crate::coeff::Coeff>::add_ref(a, b)
}
fn czero<const N: u32>() -> Cyclo<N> {
    <Cyclo<N> as crate::coeff::Coeff>::zero()
}

fn nwords(n: usize) -> usize {
    n.div_ceil(64).max(1)
}
fn bit(b: &[u64], i: usize) -> bool {
    (b[i / 64] >> (i % 64)) & 1 == 1
}
fn toggle(b: &mut [u64], i: usize) {
    b[i / 64] ^= 1u64 << (i % 64);
}
fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}
fn zero_bits(b: &[u64]) -> bool {
    b.iter().all(|&w| w == 0)
}
fn dot(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum::<u32>() % 2 == 1
}
fn ones(b: &[u64]) -> Vec<usize> {
    let mut v = Vec::new();
    for (w, &x) in b.iter().enumerate() {
        let mut x = x;
        while x != 0 {
            v.push(w * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    v
}
fn unit(n: usize, i: usize) -> Bits {
    let mut b = vec![0; nwords(n)];
    toggle(&mut b, i);
    b
}

/// A phase polynomial `f: 𝔽₂^n → ℤ/2^r` in parity-product form (see the
/// [module documentation](self)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhasePolynomial {
    n: usize,
    bits: u32,
    constant: u64,
    /// `Π_{λ ∈ key} [λ·x]` with coefficient `value`: every `λ` nonzero, the
    /// key sorted and without repeats, every value nonzero mod `2^r`.
    terms: BTreeMap<Vec<Bits>, u64>,
}

/// The value of a Clifford-class exponential sum ([`PhasePolynomial::clifford_sum`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordSum {
    /// The sum vanishes.
    Zero,
    /// `√2^{sqrt2_power} · e^{2πi·phase/2^{phase_bits}}`.
    Value {
        /// `|S|² = 2^{sqrt2_power}`.
        sqrt2_power: u32,
        /// The argument, in units of `2π/2^{phase_bits}`.
        phase: u64,
        /// `max(r, 3)`.
        phase_bits: u32,
    },
}

/// The shape of an exponential sum's computation ([`PhasePolynomial::exp_sum_plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpSumPlan {
    /// Variables summed over.
    pub variables: usize,
    /// Variables removed by Clifford (Gauss-sum) elimination.
    pub clifford_eliminated: usize,
    /// Variables left for variable elimination (the non-Clifford core).
    pub core: usize,
    /// Largest table arity of the core's elimination order: the cost is
    /// `O(terms · 2^width)`.  0 when the core is empty.
    pub width: usize,
    /// The sum is zero already after Clifford elimination.
    pub zero: bool,
}

impl PhasePolynomial {
    /// The zero polynomial on `n` variables with values mod `2^bits` (`bits ≤ 32`).
    pub fn new(n: usize, bits: u32) -> Self {
        assert!(bits <= 32, "phases mod 2^bits with bits ≤ 32");
        PhasePolynomial {
            n,
            bits,
            constant: 0,
            terms: BTreeMap::new(),
        }
    }

    /// The degree-≤ 2 polynomial `c + Σ_j linear_j x_j + Σ (b, j, k) b x_j x_k`.
    pub fn quadratic(n: usize, bits: u32, c: u64, linear: &[u64], quad: &[(usize, usize, u64)]) -> Self {
        assert_eq!(linear.len(), n, "one linear coefficient per variable");
        let mut f = PhasePolynomial::new(n, bits);
        f.add_constant(c as i64);
        for (j, &a) in linear.iter().enumerate() {
            f.add_monomial(a as i64, &[j]);
        }
        for &(j, k, b) in quad {
            f.add_monomial(b as i64, &[j, k]);
        }
        f
    }

    fn modulus(&self) -> u64 {
        1u64 << self.bits
    }
    fn reduce(&self, c: i64) -> u64 {
        c.rem_euclid(self.modulus() as i64) as u64
    }

    /// Number of variables `n`.
    pub fn num_vars(&self) -> usize {
        self.n
    }
    /// `r`: values are mod `2^r`, `ζ = e^{2πi/2^r}`.
    pub fn modulus_bits(&self) -> u32 {
        self.bits
    }

    /// Add the constant `c`.
    pub fn add_constant(&mut self, c: i64) {
        self.constant = (self.constant + self.reduce(c)) % self.modulus();
    }

    /// Add `c · Π_{i ∈ vars} x_i` (repeated variables count once; an empty
    /// list adds a constant).
    pub fn add_monomial(&mut self, c: i64, vars: &[usize]) {
        let factors = vars
            .iter()
            .map(|&i| {
                assert!(i < self.n, "variable {i} out of range");
                unit(self.n, i)
            })
            .collect();
        self.add_term(self.reduce(c), factors);
    }

    /// Add `c · Π_k [⊕_{i ∈ forms[k]} x_i]`, a product of parities.
    pub fn add_parity_term(&mut self, c: i64, forms: &[&[usize]]) {
        let factors = forms
            .iter()
            .map(|form| {
                let mut b = vec![0; nwords(self.n)];
                for &i in *form {
                    assert!(i < self.n, "variable {i} out of range");
                    toggle(&mut b, i);
                }
                b
            })
            .collect();
        self.add_term(self.reduce(c), factors);
    }

    fn add_term(&mut self, c: u64, mut factors: Vec<Bits>) {
        let m = self.modulus();
        let c = c % m;
        if c == 0 || factors.iter().any(|f| zero_bits(f)) {
            return; // [0] = 0
        }
        factors.sort();
        factors.dedup(); // [λ]^2 = [λ]
        if factors.is_empty() {
            self.constant = (self.constant + c) % m;
            return;
        }
        match self.terms.entry(factors) {
            std::collections::btree_map::Entry::Occupied(mut e) => {
                let v = (*e.get() + c) % m;
                if v == 0 {
                    e.remove();
                } else {
                    *e.get_mut() = v;
                }
            }
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(c);
            }
        }
    }

    /// `f(x)` mod `2^r`.
    pub fn eval(&self, x: &[u8]) -> u64 {
        assert_eq!(x.len(), self.n);
        let mut xb = vec![0; nwords(self.n)];
        for (i, &v) in x.iter().enumerate() {
            if v & 1 == 1 {
                toggle(&mut xb, i);
            }
        }
        let m = self.modulus();
        let mut s = self.constant;
        for (factors, &c) in &self.terms {
            if factors.iter().all(|f| dot(f, &xb)) {
                s = (s + c) % m;
            }
        }
        s
    }

    /// The level `s + r − v₂(c) − 1` of a term (the Clifford-hierarchy level
    /// of the corresponding diagonal gate).
    fn term_level(&self, s: usize, c: u64) -> u32 {
        s as u32 + self.bits - c.trailing_zeros() - 1
    }

    /// The largest term level (0 without terms): 1 for Pauli Z phases, 2 for the
    /// Clifford class, 3 for T, CS and CCZ.  An upper bound for the level of
    /// the function, exact for monomials.
    pub fn level_bound(&self) -> u32 {
        self.terms
            .iter()
            .map(|(k, &c)| self.term_level(k.len(), c))
            .max()
            .unwrap_or(0)
    }

    /// Every term in the Clifford class (level ≤ 2), so that the exponential
    /// sum is polynomial ([`clifford_sum`](Self::clifford_sum)).
    pub fn is_clifford(&self) -> bool {
        self.level_bound() <= 2
    }

    /// `g(t) = f(a + Σ_i t_i basis_i)` on `t ∈ 𝔽₂^k`, `k = basis.len()`
    /// (vectors as 0/1 entries).  The basis need not be independent; then the
    /// sum of `g` counts each point of `a + V` `2^{k − dim V}` times.
    pub fn restrict(&self, a: &[u8], basis: &[Vec<u8>]) -> Result<PhasePolynomial, String> {
        let n = self.n;
        if a.len() != n || basis.iter().any(|b| b.len() != n) {
            return Err(format!("points of 𝔽₂^{n} expected"));
        }
        let to_bits = |v: &[u8]| -> Bits {
            let mut b = vec![0; nwords(n)];
            for (i, &x) in v.iter().enumerate() {
                if x & 1 == 1 {
                    toggle(&mut b, i);
                }
            }
            b
        };
        let ab = to_bits(a);
        let bb: Vec<Bits> = basis.iter().map(|v| to_bits(v)).collect();
        let k = basis.len();
        let mut out = PhasePolynomial::new(k, self.bits);
        out.constant = self.constant;
        for (factors, &c) in &self.terms {
            let mapped: Vec<(Bits, bool)> = factors
                .iter()
                .map(|lam| {
                    let mut mu = vec![0; nwords(k)];
                    for (i, b) in bb.iter().enumerate() {
                        if dot(lam, b) {
                            toggle(&mut mu, i);
                        }
                    }
                    (mu, dot(lam, &ab))
                })
                .collect();
            out.add_expanded(c, &mapped);
        }
        Ok(out)
    }

    /// Add `c Π_k (flip_k ? 1 − [μ_k] : [μ_k])`, expanded.
    fn add_expanded(&mut self, c: u64, factors: &[(Bits, bool)]) {
        let fixed: Vec<Bits> = factors
            .iter()
            .filter(|(_, f)| !f)
            .map(|(m, _)| m.clone())
            .collect();
        let flipped: Vec<&Bits> = factors.iter().filter(|(_, f)| *f).map(|(m, _)| m).collect();
        let m = self.modulus();
        for mask in 0u64..(1u64 << flipped.len()) {
            let mut fs = fixed.clone();
            fs.extend(
                flipped
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| mask >> k & 1 == 1)
                    .map(|(_, f)| (*f).clone()),
            );
            let coeff = if mask.count_ones() % 2 == 1 {
                (m - c % m) % m
            } else {
                c
            };
            self.add_term(coeff, fs);
        }
    }

    /// `Σ_{x ∈ 𝔽₂^n} ζ^{f(x)}` exactly, `ζ = e^{2πi/2^r}`.  Needs `2^r | N`
    /// (for `r ≤ 1` any `N`: the sum is an integer).  Polynomial for the
    /// Clifford class, otherwise exponential in [`ExpSumPlan::width`].
    pub fn exp_sum<const N: u32>(&self) -> Cyclo<N> {
        assert!(
            self.bits <= 1 || N % (1u32 << self.bits) == 0,
            "the sum lies in Q(ζ_{{2^{}}}): N must be a multiple of {}",
            self.bits,
            1u64 << self.bits
        );
        let mut red = Reduced::new(self);
        red.run_clifford();
        if red.zero {
            return czero();
        }
        let pre = red.prefactor::<N>(self);
        if red.nc.is_empty() && red.alive_count() == 0 {
            return pre;
        }
        assert!(self.bits <= 12, "non-Clifford sums are computed for r ≤ 12");
        let (value, twos) = red.dp();
        let v = cmul(
            &value.to_cyclo::<N>(),
            &Cyclo::from_rational(BigRational::from_integer(BigInt::one() << twos)),
        );
        cmul(&pre, &v)
    }

    /// `S(f, a + V) = Σ_{x ∈ a+V} ζ^{f(x)}` for `V` spanned by the
    /// (independent) rows of `basis`: [`restrict`](Self::restrict), then
    /// [`exp_sum`](Self::exp_sum).
    pub fn exp_sum_over<const N: u32>(&self, a: &[u8], basis: &[Vec<u8>]) -> Result<Cyclo<N>, String> {
        let mut m = BitMat::zeros(basis.len(), self.n);
        for (r, v) in basis.iter().enumerate() {
            if v.len() != self.n {
                return Err(format!("points of 𝔽₂^{} expected", self.n));
            }
            for (c, &x) in v.iter().enumerate() {
                m.set(r, c, x & 1 == 1);
            }
        }
        if m.rank() < basis.len() {
            return Err("the basis is not linearly independent".into());
        }
        Ok(self.restrict(a, basis)?.exp_sum::<N>())
    }

    /// The exponential sum of a Clifford-class polynomial in closed form
    /// (`0` or `√2^p · e^{2πi φ}`), in `O(n³/64)`; `None` if some term is not
    /// Clifford ([`is_clifford`](Self::is_clifford)).
    pub fn clifford_sum(&self) -> Option<CliffordSum> {
        if !self.is_clifford() {
            return None;
        }
        let mut red = Reduced::new(self);
        red.run_clifford();
        debug_assert!(red.zero || red.alive_count() == 0);
        if red.zero {
            return Some(CliffordSum::Zero);
        }
        // ζ_M^c i^e (1+i)^p 2^q, (1+i) = √2 ζ_8
        let phase_bits = self.bits.max(3);
        let l = 1u64 << phase_bits;
        let phase = (self.constant * (l >> self.bits) + red.p as u64 * (l / 8) + red.e4 as u64 * (l / 4)) % l;
        Some(CliffordSum::Value {
            sqrt2_power: red.p + 2 * red.q,
            phase,
            phase_bits,
        })
    }

    /// The cost structure of [`exp_sum`](Self::exp_sum) without computing it.
    pub fn exp_sum_plan(&self) -> ExpSumPlan {
        let mut red = Reduced::new(self);
        red.run_clifford();
        let core = if red.zero { 0 } else { red.alive_count() };
        let width = if red.zero || core == 0 {
            0
        } else {
            let scopes = red.factor_scopes();
            greedy_order(&red.alive_vars(), &scopes).1
        };
        ExpSumPlan {
            variables: self.n,
            clifford_eliminated: red.eliminated,
            core,
            width,
            zero: red.zero,
        }
    }
}

/// The Clifford part as a `ℤ₄` quadratic form `Σ a_j x_j + 2 Σ B_jk x_j x_k`
/// plus the non-Clifford terms, with the factor collected by eliminations.
struct Reduced {
    n: usize,
    bits: u32,
    a: Vec<u8>,
    b: Vec<Bits>,
    alive: Vec<bool>,
    /// non-Clifford terms, coefficients mod 2^bits
    nc: BTreeMap<Vec<Bits>, u64>,
    /// i^{e4} (1+i)^p 2^q, or zero
    e4: u8,
    p: u32,
    q: u32,
    zero: bool,
    eliminated: usize,
}

impl Reduced {
    fn new(f: &PhasePolynomial) -> Self {
        let n = f.n;
        let w = nwords(n);
        let mut red = Reduced {
            n,
            bits: f.bits,
            a: vec![0; n],
            b: vec![vec![0; w]; n],
            alive: vec![true; n],
            nc: BTreeMap::new(),
            e4: 0,
            p: 0,
            q: 0,
            zero: false,
            eliminated: 0,
        };
        let mut terms: Vec<(Vec<Bits>, u64)> = f.terms.iter().map(|(k, &c)| (k.clone(), c)).collect();
        // pivot the non-Clifford parities: change variables so that they only
        // depend on the pivot coordinates of their row echelon form
        let nc_forms: Vec<&Bits> = terms
            .iter()
            .filter(|(k, c)| f.term_level(k.len(), *c) > 2)
            .flat_map(|(k, _)| k.iter())
            .collect();
        if !nc_forms.is_empty() {
            let mut m = BitMat::zeros(nc_forms.len(), n);
            for (r, lam) in nc_forms.iter().enumerate() {
                for i in ones(lam) {
                    m.set(r, i, true);
                }
            }
            let pivots = m.rref();
            let rows: Vec<Bits> = (0..pivots.len()).map(|r| m.row_words(r).to_vec()).collect();
            // λ ↦ λ ⊕ Σ_{i : λ_{j_i} = 1} (r_i ⊕ e_{j_i}) (the parity in the new coordinates)
            let transform = |lam: &Bits| -> Bits {
                let mut out = lam.clone();
                for (i, &j) in pivots.iter().enumerate() {
                    if bit(lam, j) {
                        xor_into(&mut out, &rows[i]);
                        toggle(&mut out, j);
                    }
                }
                out
            };
            terms = terms
                .into_iter()
                .map(|(k, c)| (k.iter().map(&transform).collect(), c))
                .collect();
        }
        let mut canon = PhasePolynomial::new(n, f.bits);
        for (k, c) in terms {
            canon.add_term(c, k);
        }
        for (k, c) in canon.terms {
            red.add(k, c);
        }
        red
    }

    fn level(&self, s: usize, c: u64) -> u32 {
        s as u32 + self.bits - c.trailing_zeros() - 1
    }

    /// Route a term to the ℤ₄ form (Clifford) or to the non-Clifford list.
    fn add(&mut self, k: Vec<Bits>, c: u64) {
        if self.level(k.len(), c) > 2 {
            let m = 1u64 << self.bits;
            let e = self.nc.entry(k.clone()).or_insert(0);
            *e = (*e + c) % m;
            let v = *e;
            if v == 0 {
                self.nc.remove(&k);
            } else if self.level(k.len(), v) <= 2 {
                // merged into a Clifford term
                self.nc.remove(&k);
                self.add(k, v);
            }
            return;
        }
        match k.len() {
            1 => {
                let beta = ((c << 2) >> self.bits) as u8 & 3;
                self.add_parity(beta, &k[0]);
            }
            2 => {
                if ((c << 1) >> self.bits) & 1 == 1 {
                    self.add_bilinear(&k[0], &k[1]);
                }
            }
            _ => unreachable!("Clifford terms have degree ≤ 2"),
        }
    }

    /// `i^{β [L·x]}`: `[L·x] ≡ Σ_{l∈L} x_l + 2 e_2(x_L)` mod 4.
    fn add_parity(&mut self, beta: u8, l: &Bits) {
        let ls = ones(l);
        for &x in &ls {
            self.a[x] = (self.a[x] + beta) & 3;
        }
        if beta & 1 == 1 {
            for &x in &ls {
                xor_into(&mut self.b[x], l);
                toggle(&mut self.b[x], x);
            }
        }
    }

    /// `i^{2 [λ·x][μ·x]} = (−1)^{(λ·x)(μ·x)}`.
    fn add_bilinear(&mut self, lam: &Bits, mu: &Bits) {
        for x in ones(lam) {
            xor_into(&mut self.b[x], mu);
        }
        for x in ones(mu) {
            xor_into(&mut self.b[x], lam);
        }
        // diagonal: 2 λ_l μ_l x_l (the two toggles above cancel on it)
        for (w, (a, b)) in lam.iter().zip(mu).enumerate() {
            let mut both = a & b;
            while both != 0 {
                let x = w * 64 + both.trailing_zeros() as usize;
                self.a[x] = (self.a[x] + 2) & 3;
                both &= both - 1;
            }
        }
    }

    fn remove(&mut self, v: usize) {
        for u in ones(&self.b[v]) {
            toggle(&mut self.b[u], v);
        }
        self.b[v].iter_mut().for_each(|w| *w = 0);
        self.a[v] = 0;
        self.alive[v] = false;
        self.eliminated += 1;
    }

    fn protected(&self) -> Bits {
        let mut p = vec![0; nwords(self.n)];
        for k in self.nc.keys() {
            for lam in k {
                for (x, y) in p.iter_mut().zip(lam) {
                    *x |= *y;
                }
            }
        }
        p
    }

    fn alive_vars(&self) -> Vec<usize> {
        (0..self.n).filter(|&v| self.alive[v]).collect()
    }
    fn alive_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Sum out every variable outside the non-Clifford supports.
    fn run_clifford(&mut self) {
        loop {
            if self.zero {
                return;
            }
            let prot = self.protected();
            let Some(v) = (0..self.n).find(|&v| self.alive[v] && !bit(&prot, v)) else {
                return;
            };
            self.eliminate(v, &prot);
        }
    }

    fn eliminate(&mut self, v: usize, prot: &Bits) {
        let av = self.a[v] & 3;
        let mut l = self.b[v].clone();
        self.remove(v);
        if av & 1 == 1 {
            // 1 + i^{a} (−1)^{[L]} = (1+i) i^{3 [a = 3]} i^{(4−a)[L]}
            self.p += 1;
            if av == 3 {
                self.e4 = (self.e4 + 3) & 3;
            }
            self.add_parity(4 - av, &l);
            return;
        }
        // 1 + (−1)^{[L] + a/2} = 2 [ [L] = a/2 ]
        let c = av / 2;
        let ls = ones(&l);
        if ls.is_empty() {
            if c == 1 {
                self.zero = true;
            } else {
                self.q += 1;
            }
            return;
        }
        self.q += 1;
        let m = *ls.iter().find(|&&x| !bit(prot, x)).unwrap_or(&ls[0]);
        toggle(&mut l, m);
        self.substitute(m, c == 1, &l);
    }

    /// `x_m = c ⊕ [R·x]` in the form and the non-Clifford terms; removes `x_m`.
    fn substitute(&mut self, m: usize, c: bool, r: &Bits) {
        let am = self.a[m] & 3;
        let mrow = self.b[m].clone();
        self.remove(m);
        let rs = ones(r);
        // a_m x_m, with x_m ≡ c + (−1)^c ([R] mod 4)
        if c {
            self.e4 = (self.e4 + am) & 3;
        }
        let lin = if c { (4 - am) & 3 } else { am };
        for &x in &rs {
            self.a[x] = (self.a[x] + lin) & 3;
        }
        if am & 1 == 1 {
            for &x in &rs {
                xor_into(&mut self.b[x], r);
                toggle(&mut self.b[x], x);
            }
        }
        // 2 x_m M = 2c M + 2 [R] M (mod 4)
        let ms = ones(&mrow);
        if c {
            for &x in &ms {
                self.a[x] = (self.a[x] + 2) & 3;
            }
        }
        for &x in &rs {
            xor_into(&mut self.b[x], &mrow);
            if bit(&mrow, x) {
                self.a[x] = (self.a[x] + 2) & 3;
            }
        }
        for &x in &ms {
            xor_into(&mut self.b[x], r);
        }
        // the non-Clifford terms
        if self.nc.keys().any(|k| k.iter().any(|lam| bit(lam, m))) {
            let old = std::mem::take(&mut self.nc);
            let mut f = PhasePolynomial::new(self.n, self.bits);
            for (k, coeff) in old {
                let mapped: Vec<(Bits, bool)> = k
                    .into_iter()
                    .map(|lam| {
                        if bit(&lam, m) {
                            let mut l2 = lam;
                            toggle(&mut l2, m);
                            xor_into(&mut l2, r);
                            (l2, c)
                        } else {
                            (lam, false)
                        }
                    })
                    .collect();
                f.add_expanded(coeff, &mapped);
            }
            // constants produced by the expansion are global phases ζ^c
            if f.constant != 0 {
                self.nc_constant_pending(f.constant);
            }
            for (k, coeff) in f.terms {
                self.add(k, coeff);
            }
        }
    }

    /// A constant phase from expanding non-Clifford terms, kept as a term on
    /// no parities: folded into a degree-0 entry of `nc`.
    fn nc_constant_pending(&mut self, c: u64) {
        let m = 1u64 << self.bits;
        let e = self.nc.entry(Vec::new()).or_insert(0);
        *e = (*e + c) % m;
        if *e == 0 {
            self.nc.remove(&Vec::new());
        }
    }

    /// `ζ_M^{constant} i^{e4} (1+i)^p 2^q` in `ℚ(ζ_N)`.
    fn prefactor<const N: u32>(&self, f: &PhasePolynomial) -> Cyclo<N> {
        let rat = |x: BigInt| Cyclo::<N>::from_rational(BigRational::from_integer(x));
        let ipow = |e: u64| -> Cyclo<N> {
            match e & 3 {
                0 => rat(BigInt::one()),
                2 => rat(-BigInt::one()),
                e => {
                    assert!(N % 4 == 0, "a factor i needs 4 | N");
                    Cyclo::zeta((N / 4) as i64 * e as i64)
                }
            }
        };
        let mut v = rat(BigInt::one() << self.q);
        // (1+i)^p = (2i)^{p/2} (1+i)^{p mod 2}
        if self.p > 0 {
            v = cmul(
                &cmul(&v, &rat(BigInt::one() << (self.p / 2))),
                &ipow((self.p / 2) as u64),
            );
            if self.p % 2 == 1 {
                v = cmul(&v, &cadd(&ipow(0), &ipow(1)));
            }
        }
        v = cmul(&v, &ipow(self.e4 as u64));
        let mut c = f.constant;
        if let Some(&k) = self.nc.get(&Vec::new()) {
            c = (c + k) % f.modulus();
        }
        if c != 0 {
            // the constant phase ζ_{2^r}^c (r ≥ 1 here)
            if f.bits == 1 {
                v = cmul(&v, &rat(-BigInt::one()));
            } else {
                v = cmul(&v, &Cyclo::zeta((N >> f.bits) as i64 * c as i64));
            }
        }
        v
    }

    /// Scopes of the factors the core starts with.
    fn factor_scopes(&self) -> Vec<Vec<usize>> {
        self.factors().into_iter().map(|f| f.scope).collect()
    }

    /// Initial factors of the core: unary and pairwise Clifford phases and the
    /// non-Clifford terms, as phase tables (exponents of ζ_{2h}).
    fn factors(&self) -> Vec<Factor> {
        let h = half_order(self.bits);
        let scale = (2 * h as u64) >> self.bits; // ζ_{2^r} = ζ_{2h}^{scale}
        let vars = self.alive_vars();
        let mut by_scope: BTreeMap<Vec<usize>, Vec<u64>> = BTreeMap::new();
        let mut add = |scope: Vec<usize>, table: Vec<u64>| {
            let e = by_scope.entry(scope).or_insert_with(|| vec![0; table.len()]);
            for (x, y) in e.iter_mut().zip(table) {
                *x = (*x + y) % (2 * h as u64);
            }
        };
        for &v in &vars {
            if self.a[v] != 0 {
                add(vec![v], vec![0, self.a[v] as u64 * (h as u64 / 2)]);
            }
            for u in ones(&self.b[v]) {
                if u > v {
                    add(vec![v, u], vec![0, 0, 0, h as u64]);
                }
            }
        }
        for (k, &c) in &self.nc {
            if k.is_empty() {
                continue; // a constant, in the prefactor
            }
            let mut sup = vec![0; nwords(self.n)];
            for lam in k {
                for (x, y) in sup.iter_mut().zip(lam) {
                    *x |= *y;
                }
            }
            let scope = ones(&sup);
            let size = 1usize << scope.len();
            let table: Vec<u64> = (0..size)
                .map(|idx| {
                    let mut y = vec![0; nwords(self.n)];
                    for (t, &x) in scope.iter().enumerate() {
                        if idx >> t & 1 == 1 {
                            toggle(&mut y, x);
                        }
                    }
                    if k.iter().all(|lam| dot(lam, &y)) {
                        c * scale % (2 * h as u64)
                    } else {
                        0
                    }
                })
                .collect();
            add(scope, table);
        }
        by_scope
            .into_iter()
            .map(|(scope, t)| Factor {
                scope,
                table: Table::Phase(t),
            })
            .collect()
    }

    /// Sum over the core by variable elimination; returns the value and a
    /// power of 2 (variables that appear in no factor).
    fn dp(&self) -> (ZZeta, u32) {
        let h = half_order(self.bits);
        let mut factors = self.factors();
        let vars = self.alive_vars();
        let scopes: Vec<Vec<usize>> = factors.iter().map(|f| f.scope.clone()).collect();
        let (order, _) = greedy_order(&vars, &scopes);
        let mut twos = 0u32;
        for v in order {
            let (mine, rest): (Vec<Factor>, Vec<Factor>) =
                factors.into_iter().partition(|f| f.scope.contains(&v));
            factors = rest;
            if mine.is_empty() {
                twos += 1;
                continue;
            }
            let mut union: Vec<usize> = mine.iter().flat_map(|f| f.scope.iter().copied()).collect();
            union.sort_unstable();
            union.dedup();
            let pv = union.binary_search(&v).expect("v in its factors");
            let maps: Vec<Vec<usize>> = mine
                .iter()
                .map(|f| {
                    f.scope
                        .iter()
                        .map(|x| union.binary_search(x).expect("in union"))
                        .collect()
                })
                .collect();
            let mut out = vec![ZZeta::zero(h); 1usize << (union.len() - 1)];
            for idx in 0..(1usize << union.len()) {
                let mut e = 0u64;
                let mut val: Option<ZZeta> = None;
                for (f, map) in mine.iter().zip(&maps) {
                    let li = map
                        .iter()
                        .enumerate()
                        .fold(0usize, |acc, (t, &p)| acc | ((idx >> p & 1) << t));
                    match &f.table {
                        Table::Phase(x) => e += x[li],
                        Table::Value(x) => {
                            val = Some(match val {
                                None => x[li].clone(),
                                Some(z) => z.mul(&x[li]),
                            })
                        }
                    }
                }
                let term = match val {
                    None => ZZeta::zeta(h, e),
                    Some(z) => z.mul_zeta(e),
                };
                let low = idx & ((1usize << pv) - 1);
                let high = (idx >> (pv + 1)) << pv;
                out[low | high].add_assign(&term);
            }
            union.remove(pv);
            factors.push(Factor {
                scope: union,
                table: Table::Value(out),
            });
        }
        let mut total = ZZeta::one(h);
        for f in factors {
            debug_assert!(f.scope.is_empty());
            total = match f.table {
                Table::Phase(x) => total.mul_zeta(x[0]),
                Table::Value(x) => total.mul(&x[0]),
            };
        }
        (total, twos)
    }
}

/// `h` with `ζ_{2h}` the internal root of unity: `2h = max(2^r, 4)`.
fn half_order(bits: u32) -> usize {
    (1usize << bits).max(4) / 2
}

/// A table over the assignments of `scope` (bit `t` of the index is `scope[t]`).
struct Factor {
    scope: Vec<usize>,
    table: Table,
}

enum Table {
    /// `ζ_{2h}^{e}` per assignment.
    Phase(Vec<u64>),
    /// General values in `ℤ[ζ_{2h}]`.
    Value(Vec<ZZeta>),
}

/// Greedy variable elimination order (smallest resulting table first, then
/// the smallest variable), and the largest table arity along it.
fn greedy_order(vars: &[usize], scopes: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let mut scopes: Vec<Vec<usize>> = scopes.to_vec();
    let mut left: Vec<usize> = vars.to_vec();
    let mut order = Vec::with_capacity(left.len());
    let mut width = 0;
    while !left.is_empty() {
        let mut best: Option<(usize, usize)> = None; // (size, index into left)
        for (i, &v) in left.iter().enumerate() {
            let mut u: Vec<usize> = scopes
                .iter()
                .filter(|s| s.contains(&v))
                .flat_map(|s| s.iter().copied())
                .collect();
            u.sort_unstable();
            u.dedup();
            if best.is_none_or(|(b, _)| u.len() < b) {
                best = Some((u.len(), i));
            }
        }
        let (size, i) = best.expect("nonempty");
        let v = left.remove(i);
        width = width.max(size);
        let (mine, rest): (Vec<Vec<usize>>, Vec<Vec<usize>>) =
            scopes.into_iter().partition(|s| s.contains(&v));
        scopes = rest;
        if !mine.is_empty() {
            let mut u: Vec<usize> = mine.into_iter().flatten().filter(|&x| x != v).collect();
            u.sort_unstable();
            u.dedup();
            scopes.push(u);
        }
        order.push(v);
    }
    (order, width)
}

/// An element `Σ_{j<h} c_j ζ^j` of `ℤ[ζ_{2h}] = ℤ[ζ]/(ζ^h + 1)`, `2h` a power
/// of two: exact cyclotomic integers for the tables of [`PhasePolynomial::exp_sum`],
/// with negacyclic products.  ([`Cyclo<N>`] is the field arithmetic; this is
/// the integral ring at a run-time power-of-two order, which the elimination
/// tables need in their inner loop.)
#[derive(Clone, Debug, PartialEq, Eq)]
struct ZZeta {
    c: Vec<BigInt>,
}

impl ZZeta {
    fn zero(h: usize) -> Self {
        ZZeta {
            c: vec![BigInt::zero(); h],
        }
    }
    fn one(h: usize) -> Self {
        Self::zeta(h, 0)
    }
    fn zeta(h: usize, e: u64) -> Self {
        let mut z = Self::zero(h);
        let e = (e % (2 * h as u64)) as usize;
        if e < h {
            z.c[e] = BigInt::one();
        } else {
            z.c[e - h] = -BigInt::one();
        }
        z
    }
    fn mul_zeta(&self, e: u64) -> Self {
        let h = self.c.len();
        let e = (e % (2 * h as u64)) as usize;
        let mut out = Self::zero(h);
        for (j, x) in self.c.iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            let k = (j + e) % (2 * h);
            if k < h {
                out.c[k] += x;
            } else {
                out.c[k - h] -= x;
            }
        }
        out
    }
    fn mul(&self, o: &Self) -> Self {
        let h = self.c.len();
        let mut out = Self::zero(h);
        for (j, x) in self.c.iter().enumerate() {
            if x.is_zero() {
                continue;
            }
            for (k, y) in o.c.iter().enumerate() {
                if y.is_zero() {
                    continue;
                }
                let p = x * y;
                if j + k < h {
                    out.c[j + k] += p;
                } else {
                    out.c[j + k - h] -= p;
                }
            }
        }
        out
    }
    fn add_assign(&mut self, o: &Self) {
        for (x, y) in self.c.iter_mut().zip(&o.c) {
            *x += y;
        }
    }
    /// The same number in `ℚ(ζ_N)`; `2h | N`, or the number is an integer.
    fn to_cyclo<const N: u32>(&self) -> Cyclo<N> {
        let h = self.c.len() as u32;
        if N % (2 * h) == 0 {
            let step = (N / (2 * h)) as usize;
            let mut c = vec![BigRational::zero(); N as usize];
            for (j, x) in self.c.iter().enumerate() {
                c[j * step] = BigRational::from_integer(x.clone());
            }
            Cyclo::from_coeffs(c)
        } else {
            assert!(
                self.c[1..].iter().all(|x| x.is_zero()),
                "the value is not in Q(ζ_{N})"
            );
            Cyclo::from_rational(BigRational::from_integer(self.c[0].clone()))
        }
    }
}
