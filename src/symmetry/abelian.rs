//! Finite abelian groups `ℤ/d_0 × ⋯ × ℤ/d_{k-1}`: subgroups, quotients,
//! duality and the Fourier transform.
//!
//! Subgroups are stored through the embedding
//! `x ↦ (x_i · e/d_i)_i ∈ (ℤ/e)^k`, `e` the exponent, as the Howell form of
//! their image ([`crate::zmod::ZnMat::howell`]).  That form is canonical, so
//! subgroup equality is structural, and membership, order, intersection and
//! annihilators all reduce to linear algebra over ℤ/e.
//!
//! The dual group is identified with `G` through the pairing
//! `χ_y(x) = exp(2πi Σ_i x_i y_i / d_i)`.
//!
//! Zero-sum subsequences: [`AbelianGroup::davenport_bound`] and
//! [`AbelianGroup::zero_sum_subset`], a nonempty subset of a sequence with
//! sum zero, found from the elements alone by one of the [`ZeroSum`]
//! strategies.

use crate::zmod::{IntMat, ZnMat};
use num_bigint::BigInt;
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};

/// `ℤ/d_0 × ⋯ × ℤ/d_{k-1}` with every `d_i ≥ 1`; elements are `Vec<u64>` with `x_i < d_i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbelianGroup {
    d: Vec<u64>,
    e: u64,
}

/// A subgroup of an [`AbelianGroup`], in canonical form (compare with `==`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbSubgroup {
    /// Howell form of the embedded generators in `(ℤ/e)^k`; `None` when `e = 1`.
    h: Option<ZnMat>,
    k: usize,
}

impl AbelianGroup {
    /// `ℤ/d_0 × ⋯`. Panics if some `d_i = 0` or the exponent reaches `2^63`.
    pub fn new(d: Vec<u64>) -> Self {
        assert!(d.iter().all(|&x| x >= 1), "cyclic factors must have order ≥ 1");
        let e = d.iter().fold(1u64, |a, &b| {
            let l = a.lcm(&b);
            assert!(l < 1 << 63, "exponent too large");
            l
        });
        AbelianGroup { d, e }
    }
    /// Elementary abelian `(ℤ/p)^k`.
    pub fn elementary(p: u64, k: usize) -> Self {
        Self::new(vec![p; k])
    }

    /// `ℤ^k / ⟨rows of R⟩` in invariant-factor form (trivial factors dropped),
    /// together with the images of the `k` standard generators.  `None` if the
    /// quotient is infinite.
    pub fn from_relations(k: usize, relations: &IntMat) -> Option<(AbelianGroup, Vec<Vec<u64>>)> {
        assert_eq!(relations.cols(), k);
        if relations.rows() == 0 {
            return if k == 0 {
                Some((AbelianGroup::new(vec![]), vec![]))
            } else {
                None
            };
        }
        // rows are relations: L = ℤ^r R, and L V = row span of U R V = D.
        let s = relations.smith();
        if s.diag.len() < k || s.diag.iter().any(|x| x.is_zero()) {
            return None;
        }
        let keep: Vec<usize> = (0..k).filter(|&i| !s.diag[i].is_one()).collect();
        let d: Vec<u64> = keep
            .iter()
            .map(|&i| s.diag[i].to_u64().expect("invariant factor fits in u64"))
            .collect();
        let images = (0..k)
            .map(|g| {
                keep.iter()
                    .zip(&d)
                    .map(|(&i, &di)| {
                        s.v.get(g, i)
                            .mod_floor(&BigInt::from(di))
                            .to_u64()
                            .expect("reduced")
                    })
                    .collect()
            })
            .collect();
        Some((AbelianGroup::new(d), images))
    }

    /// The cyclic factor orders `d_i` of `G = ⊕ ℤ/d_i`.
    pub fn orders(&self) -> &[u64] {
        &self.d
    }
    /// Number of cyclic factors.
    pub fn rank(&self) -> usize {
        self.d.len()
    }
    /// The exponent `lcm(d_i)`.
    pub fn exponent(&self) -> u64 {
        self.e
    }
    /// The group order `Π d_i`.
    pub fn order(&self) -> BigInt {
        self.d.iter().map(|&x| BigInt::from(x)).product()
    }
    /// The invariant factors `a_0 | a_1 | ⋯` (all `> 1`) of the group.
    pub fn invariants(&self) -> Vec<u64> {
        let k = self.rank();
        let rel = IntMat::from_rows(
            &(0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| if i == j { self.d[i] as i64 } else { 0 })
                        .collect()
                })
                .collect::<Vec<_>>(),
        );
        Self::from_relations(k, &rel)
            .map(|(g, _)| g.d)
            .unwrap_or_default()
    }

    /// The identity element.
    pub fn zero(&self) -> Vec<u64> {
        vec![0; self.rank()]
    }
    /// The `i`-th standard generator.
    pub fn gen(&self, i: usize) -> Vec<u64> {
        let mut x = self.zero();
        x[i] = 1 % self.d[i];
        x
    }
    /// An integer vector reduced coordinatewise mod `d_i`.
    pub fn reduce(&self, x: &[i64]) -> Vec<u64> {
        x.iter()
            .zip(&self.d)
            .map(|(&a, &d)| (a as i128).rem_euclid(d as i128) as u64)
            .collect()
    }
    /// `x + y`.
    pub fn add(&self, x: &[u64], y: &[u64]) -> Vec<u64> {
        x.iter()
            .zip(y)
            .zip(&self.d)
            .map(|((&a, &b), &d)| ((a as u128 + b as u128) % d as u128) as u64)
            .collect()
    }
    /// `-x`.
    pub fn neg(&self, x: &[u64]) -> Vec<u64> {
        x.iter().zip(&self.d).map(|(&a, &d)| (d - a % d) % d).collect()
    }
    /// `x - y`.
    pub fn sub(&self, x: &[u64], y: &[u64]) -> Vec<u64> {
        self.add(x, &self.neg(y))
    }
    /// `c · x`.
    pub fn scale(&self, x: &[u64], c: i64) -> Vec<u64> {
        x.iter()
            .zip(&self.d)
            .map(|(&a, &d)| ((a as i128 * c as i128).rem_euclid(d as i128)) as u64)
            .collect()
    }
    /// Mixed-radix index, first coordinate most significant.
    pub fn index(&self, x: &[u64]) -> u64 {
        x.iter().zip(&self.d).fold(0u64, |acc, (&a, &d)| acc * d + a)
    }
    /// The element with mixed-radix index `idx`, the inverse of [`index`](Self::index).
    pub fn from_index(&self, mut idx: u64) -> Vec<u64> {
        let mut x = vec![0; self.rank()];
        for i in (0..self.rank()).rev() {
            x[i] = idx % self.d[i];
            idx /= self.d[i];
        }
        x
    }

    /// `χ_y(x) = exp(2πi · num / e)`; returns `num ∈ 0..e`.
    pub fn pairing(&self, x: &[u64], y: &[u64]) -> u64 {
        let e = self.e as u128;
        let mut s = 0u128;
        for i in 0..self.rank() {
            s = (s + (x[i] as u128 * y[i] as u128 % e) * (self.e / self.d[i]) as u128) % e;
        }
        s as u64
    }
    /// The character value `χ_y(x)` as a complex number.
    pub fn character(&self, x: &[u64], y: &[u64]) -> Complex<f64> {
        let t = 2.0 * std::f64::consts::PI * self.pairing(x, y) as f64 / self.e as f64;
        Complex::new(t.cos(), t.sin())
    }

    // ------------------------------------------------------------------ subgroups

    fn embed(&self, x: &[u64]) -> Vec<i64> {
        x.iter()
            .zip(&self.d)
            .map(|(&a, &d)| (a % d * (self.e / d)) as i64)
            .collect()
    }
    fn unembed(&self, v: &[u64]) -> Vec<u64> {
        v.iter()
            .zip(&self.d)
            .map(|(&a, &d)| a / (self.e / d) % d)
            .collect()
    }
    fn subgroup_of_embedded_rows(&self, rows: Vec<Vec<i64>>) -> AbSubgroup {
        let k = self.rank();
        if self.e == 1 || rows.is_empty() {
            return AbSubgroup {
                h: if self.e == 1 {
                    None
                } else {
                    Some(ZnMat::zeros(self.e, 0, k))
                },
                k,
            };
        }
        AbSubgroup {
            h: Some(ZnMat::from_rows(self.e, &rows).howell()),
            k,
        }
    }

    /// The subgroup generated by `gens`.
    pub fn subgroup(&self, gens: &[Vec<u64>]) -> AbSubgroup {
        self.subgroup_of_embedded_rows(gens.iter().map(|g| self.embed(g)).collect())
    }
    /// The trivial subgroup `{0}`.
    pub fn trivial_subgroup(&self) -> AbSubgroup {
        self.subgroup(&[])
    }
    /// The whole group.
    pub fn whole(&self) -> AbSubgroup {
        self.subgroup(&(0..self.rank()).map(|i| self.gen(i)).collect::<Vec<_>>())
    }
    /// Canonical generators of `H` (one per Howell row).
    pub fn gens(&self, h: &AbSubgroup) -> Vec<Vec<u64>> {
        match &h.h {
            Some(m) => (0..m.rows()).map(|i| self.unembed(m.row(i))).collect(),
            None => vec![],
        }
    }
    /// Is `x` in `h`?
    pub fn contains(&self, h: &AbSubgroup, x: &[u64]) -> bool {
        match &h.h {
            Some(m) => {
                let v: Vec<u64> = self.embed(x).into_iter().map(|a| a as u64).collect();
                m.rows() > 0 && m.span_contains(&v) || v.iter().all(|&a| a == 0)
            }
            None => true,
        }
    }
    /// `|H|`.
    pub fn subgroup_order(&self, h: &AbSubgroup) -> BigInt {
        match &h.h {
            Some(m) if m.rows() > 0 => m.span_factors().into_iter().map(BigInt::from).product(),
            _ => BigInt::one(),
        }
    }
    fn rows_i64(m: &Option<ZnMat>) -> Vec<Vec<i64>> {
        match m {
            Some(m) => (0..m.rows())
                .map(|i| m.row(i).iter().map(|&x| x as i64).collect())
                .collect(),
            None => vec![],
        }
    }
    /// `H + K`.
    pub fn join(&self, h: &AbSubgroup, k: &AbSubgroup) -> AbSubgroup {
        let mut rows = Self::rows_i64(&h.h);
        rows.extend(Self::rows_i64(&k.h));
        self.subgroup_of_embedded_rows(rows)
    }
    /// `H ∩ K`: the rows `(0, z)` of the Howell form of `[[H, H], [K, 0]]`.
    pub fn intersection(&self, h: &AbSubgroup, k: &AbSubgroup) -> AbSubgroup {
        let r = self.rank();
        let (hr, kr) = (Self::rows_i64(&h.h), Self::rows_i64(&k.h));
        if hr.is_empty() || kr.is_empty() {
            return self.trivial_subgroup();
        }
        let mut rows: Vec<Vec<i64>> = hr.iter().map(|x| [x.clone(), x.clone()].concat()).collect();
        rows.extend(kr.iter().map(|x| [x.clone(), vec![0; r]].concat()));
        let m = ZnMat::from_rows(self.e, &rows).howell();
        let out = (0..m.rows())
            .filter(|&i| m.row(i)[..r].iter().all(|&x| x == 0))
            .map(|i| m.row(i)[r..].iter().map(|&x| x as i64).collect())
            .collect();
        self.subgroup_of_embedded_rows(out)
    }
    /// Is `h` contained in `k`?
    pub fn is_subgroup(&self, h: &AbSubgroup, k: &AbSubgroup) -> bool {
        self.gens(h).iter().all(|g| self.contains(k, g))
    }
    /// `H^⊥ = {y : χ_y(x) = 1 for all x ∈ H}`, a subgroup of the dual (≅ G).
    pub fn annihilator(&self, h: &AbSubgroup) -> AbSubgroup {
        let rows = Self::rows_i64(&h.h);
        if rows.is_empty() {
            return self.whole();
        }
        // Σ X_i y_i ≡ 0 (mod e), with X the embedded generators
        let m = ZnMat::from_rows(self.e, &rows);
        let gens: Vec<Vec<u64>> = m
            .nullspace()
            .into_iter()
            .map(|y| y.iter().zip(&self.d).map(|(&a, &d)| a % d).collect())
            .collect();
        self.subgroup(&gens)
    }
    /// `G / H` in invariant-factor form, with the images of the generators of `G`.
    pub fn quotient(&self, h: &AbSubgroup) -> (AbelianGroup, Vec<Vec<u64>>) {
        let k = self.rank();
        let mut rows: Vec<Vec<i64>> = (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| if i == j { self.d[i] as i64 } else { 0 })
                    .collect()
            })
            .collect();
        rows.extend(
            self.gens(h)
                .into_iter()
                .map(|g| g.into_iter().map(|x| x as i64).collect()),
        );
        Self::from_relations(k, &IntMat::from_rows(&rows)).expect("finite")
    }

    // ------------------------------------------------------------------ Fourier transform

    /// In-place unitary Fourier transform of a function on `G`, stored by
    /// [`AbelianGroup::index`]: `f̂(y) = |G|^{-1/2} Σ_x conj(χ_y(x)) f(x)`
    /// (`inverse` uses `χ_y(x)`).  One pass per cyclic factor; a factor of
    /// order `d` costs `|G| · d` operations (`|G| log d` for `d` a power of two).
    pub fn fourier(&self, f: &mut [Complex<f64>], inverse: bool) {
        let size: usize = self.d.iter().map(|&x| x as usize).product();
        assert_eq!(f.len(), size, "length must be |G|");
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut stride = size;
        let mut buf = Vec::new();
        for &d in &self.d {
            let d = d as usize;
            stride /= d;
            if d == 1 {
                continue;
            }
            let roots: Vec<Complex<f64>> = (0..d)
                .map(|k| {
                    let t = sign * 2.0 * std::f64::consts::PI * k as f64 / d as f64;
                    Complex::new(t.cos(), t.sin())
                })
                .collect();
            for block in 0..size / (d * stride) {
                for off in 0..stride {
                    let base = block * d * stride + off;
                    buf.clear();
                    buf.extend((0..d).map(|j| f[base + j * stride]));
                    if d.is_power_of_two() {
                        fft_pow2(&mut buf, &roots);
                    } else {
                        let out: Vec<Complex<f64>> = (0..d)
                            .map(|y| (0..d).map(|x| roots[(x * y) % d] * buf[x]).sum())
                            .collect();
                        buf.copy_from_slice(&out);
                    }
                    for (j, v) in buf.iter().enumerate() {
                        f[base + j * stride] = *v;
                    }
                }
            }
        }
        let s = 1.0 / (size as f64).sqrt();
        for v in f.iter_mut() {
            *v *= s;
        }
    }
}

/// Iterative radix-2 DFT `out[y] = Σ_x roots[xy mod d] in[x]`.
fn fft_pow2(a: &mut [Complex<f64>], roots: &[Complex<f64>]) {
    let n = a.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let w = roots[k * step];
                let u = a[start + k];
                let v = a[start + k + len / 2] * w;
                a[start + k] = u + v;
                a[start + k + len / 2] = u - v;
            }
        }
        len <<= 1;
    }
}

impl AbSubgroup {
    /// Number of canonical generators.
    pub fn num_gens(&self) -> usize {
        self.h.as_ref().map_or(0, |m| m.rows())
    }
    /// Rank of the ambient group.
    pub fn ambient_rank(&self) -> usize {
        self.k
    }
}

/// The Davenport constant `D(G)`, the least `q` such that every sequence of
/// `q` elements of `G` has a nonempty zero-sum subsequence, or an upper bound
/// for it ([`AbelianGroup::davenport_bound`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Davenport {
    /// `D(G)` when [`exact`](Self::exact), otherwise an upper bound.
    pub value: u64,
    /// Whether [`value`](Self::value) is `D(G)` itself.
    pub exact: bool,
}

/// How [`AbelianGroup::zero_sum_subset`] looks for a zero-sum subset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZeroSum {
    /// `G` elementary abelian of exponent 2 (`G = 𝔽_2^d`): Gaussian elimination;
    /// a dependency over `𝔽_2` is a subset.  Complete (it finds a subset
    /// whenever one exists); `d + 1` elements always suffice.  Polynomial.
    Linear,
    /// `G` a 2-group `⊕ ℤ/2^{e_i}`: level by level, collect disjoint subsets
    /// with even sum (the [`Linear`](Self::Linear) step on the parities), halve
    /// their sums and recurse.  Succeeds on any `Π_ℓ (d_ℓ + 1)` elements, `d_ℓ`
    /// the number of factors with `e_i ≥ ℓ` ([`AbelianGroup::zero_sum_length`]);
    /// it may miss a subset in shorter sequences.  Polynomial.
    Leveled,
    /// Any `G`: subsets in order of size, then lexicographically; the first
    /// zero-sum one.  Complete and exponential, a reference for short sequences.
    Exhaustive,
    // A polynomial-time finder for F_p^d with odd p is not implemented; the
    // enum is non_exhaustive so it can be added.
}

impl AbelianGroup {
    /// `D(G)` exactly when `G` is a p-group or has rank at most 2, where it
    /// equals `D*(G) = 1 + Σ (a_i − 1)` over the invariant factors `a_i`
    /// (Olson 1969, and independently van Emde Boas and Kruyswijk).  Otherwise
    /// the upper bound `D(G) ≤ exp(G) (1 + ln(|G| / exp(G)))` (van Emde Boas and
    /// Kruyswijk), and never more than `|G|`; `D(G) ≥ D*(G)` always.
    pub fn davenport_bound(&self) -> Davenport {
        let inv = self.invariants();
        let d_star = 1 + inv.iter().map(|a| a - 1).sum::<u64>();
        let prime_power = |a: u64| -> Option<u64> {
            let p = (2..=a).find(|q| a % q == 0)?;
            let mut x = a;
            while x % p == 0 {
                x /= p;
            }
            (x == 1).then_some(p)
        };
        let primes: Vec<Option<u64>> = inv.iter().map(|&a| prime_power(a)).collect();
        let p_group = primes.iter().all(|q| q.is_some()) && primes.windows(2).all(|w| w[0] == w[1]);
        if inv.len() <= 2 || p_group {
            return Davenport {
                value: d_star,
                exact: true,
            };
        }
        let e = self.e as f64;
        let ln_order: f64 = self.d.iter().map(|&x| (x as f64).ln()).sum();
        let bound = e * (1.0 + ln_order - e.ln());
        // the floor of a real upper bound is one; guard the last ulp
        let mut value = (bound * (1.0 + 1e-12)).floor() as u64;
        if let Some(order) = self.order().to_u64() {
            value = value.min(order);
        }
        Davenport {
            value: value.max(d_star),
            exact: false,
        }
    }

    /// The sequence length that makes `strategy` succeed on every input, or
    /// `None` if the strategy does not apply to this group.  For
    /// [`ZeroSum::Exhaustive`] it is the [Davenport bound](Self::davenport_bound).
    pub fn zero_sum_length(&self, strategy: ZeroSum) -> Option<u64> {
        match strategy {
            ZeroSum::Linear => self
                .d
                .iter()
                .all(|&x| x <= 2)
                .then(|| 1 + self.d.iter().filter(|&&x| x == 2).count() as u64),
            ZeroSum::Leveled => {
                if !self.d.iter().all(|x| x.is_power_of_two()) {
                    return None;
                }
                let top = self.d.iter().map(|x| x.trailing_zeros()).max().unwrap_or(0);
                let mut q = 1u64;
                for level in 1..=top {
                    let dl = self.d.iter().filter(|x| x.trailing_zeros() >= level).count() as u64;
                    q = q.checked_mul(dl + 1)?;
                }
                Some(q)
            }
            ZeroSum::Exhaustive => Some(self.davenport_bound().value),
        }
    }

    /// A nonempty set of indices (increasing) into `seq` whose elements sum to
    /// zero, found by `strategy`; `Ok(None)` if the strategy found none (for
    /// [`Linear`](ZeroSum::Linear) and [`Exhaustive`](ZeroSum::Exhaustive):
    /// none exists).  The result depends only on `seq`.  An error if an
    /// element is malformed or the strategy does not apply to this group.
    pub fn zero_sum_subset(&self, seq: &[Vec<u64>], strategy: ZeroSum) -> Result<Option<Vec<usize>>, String> {
        for x in seq {
            if x.len() != self.rank() || x.iter().zip(&self.d).any(|(&a, &d)| a >= d) {
                return Err(format!("{x:?} is not an element of the group"));
            }
        }
        if self.zero_sum_length(strategy).is_none() {
            return Err(format!("{strategy:?} does not apply to {:?}", self.d));
        }
        Ok(match strategy {
            ZeroSum::Linear => self.zero_sum_linear(seq),
            ZeroSum::Leveled => self.zero_sum_leveled(seq),
            ZeroSum::Exhaustive => self.zero_sum_exhaustive(seq),
        })
    }

    fn zero_sum_linear(&self, seq: &[Vec<u64>]) -> Option<Vec<usize>> {
        let coords: Vec<usize> = (0..self.rank()).filter(|&i| self.d[i] == 2).collect();
        let mut m = crate::ff::BitMat::zeros(seq.len(), coords.len());
        for (r, x) in seq.iter().enumerate() {
            for (c, &i) in coords.iter().enumerate() {
                m.set(r, c, x[i] == 1);
            }
        }
        m.dependency()
    }

    fn zero_sum_leveled(&self, seq: &[Vec<u64>]) -> Option<Vec<usize>> {
        use crate::ff::BitEchelon;
        let k = self.rank();
        let mut exps: Vec<u32> = self.d.iter().map(|x| x.trailing_zeros()).collect();
        // (value at the current level, original indices): the indices' sum is
        // 2^level · value, with value in ⊕ ℤ/2^{e_i − level}
        let mut items: Vec<(Vec<u64>, Vec<usize>)> =
            seq.iter().cloned().zip((0..seq.len()).map(|i| vec![i])).collect();
        loop {
            if let Some((_, idx)) = items.iter().find(|(v, _)| v.iter().all(|&a| a == 0)) {
                let mut idx = idx.clone();
                idx.sort_unstable();
                return Some(idx);
            }
            let active: Vec<usize> = (0..k).filter(|&i| exps[i] >= 1).collect();
            let mut next = Vec::new();
            let mut pool = items;
            loop {
                // the first dependency of the parities, in pool order
                let mut ech = BitEchelon::new(active.len());
                let found = pool.iter().find_map(|(v, _)| {
                    let mut bits = vec![0u64; active.len().div_ceil(64).max(1)];
                    for (c, &i) in active.iter().enumerate() {
                        bits[c / 64] |= (v[i] & 1) << (c % 64);
                    }
                    ech.insert(bits).err()
                });
                let Some(comb) = found else {
                    break;
                };
                let subset = BitEchelon::indices(&comb);
                let mut sum = vec![0u64; k];
                let mut idx = Vec::new();
                for &j in &subset {
                    for &i in &active {
                        sum[i] = (sum[i] + pool[j].0[i]) % (1u64 << exps[i]);
                    }
                    idx.extend_from_slice(&pool[j].1);
                }
                debug_assert!(active.iter().all(|&i| sum[i] % 2 == 0));
                next.push((sum.into_iter().map(|a| a / 2).collect(), idx));
                pool = pool
                    .into_iter()
                    .enumerate()
                    .filter(|(j, _)| !subset.contains(j))
                    .map(|(_, x)| x)
                    .collect();
            }
            if next.is_empty() {
                return None;
            }
            items = next;
            for e in exps.iter_mut() {
                *e = e.saturating_sub(1);
            }
        }
    }

    fn zero_sum_exhaustive(&self, seq: &[Vec<u64>]) -> Option<Vec<usize>> {
        let q = seq.len();
        for size in 1..=q {
            // combinations of `size` indices in lexicographic order
            let mut c: Vec<usize> = (0..size).collect();
            loop {
                let mut sum = self.zero();
                for &i in &c {
                    sum = self.add(&sum, &seq[i]);
                }
                if sum.iter().all(|&a| a == 0) {
                    return Some(c);
                }
                let Some(pos) = (0..size).rev().find(|&t| c[t] < q - size + t) else {
                    break;
                };
                c[pos] += 1;
                for t in pos + 1..size {
                    c[t] = c[t - 1] + 1;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeSet;

    fn elements(g: &AbelianGroup) -> Vec<Vec<u64>> {
        let n = g.order().to_u64().unwrap();
        (0..n).map(|i| g.from_index(i)).collect()
    }
    fn closure(g: &AbelianGroup, gens: &[Vec<u64>]) -> BTreeSet<Vec<u64>> {
        let mut s: BTreeSet<Vec<u64>> = [g.zero()].into_iter().collect();
        loop {
            let new: Vec<Vec<u64>> = s
                .iter()
                .flat_map(|x| gens.iter().map(move |y| g.add(x, y)))
                .filter(|z| !s.contains(z))
                .collect();
            if new.is_empty() {
                return s;
            }
            s.extend(new);
        }
    }

    #[test]
    fn subgroups_match_brute_force() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for d in [vec![2, 4], vec![4, 4, 2], vec![3, 6], vec![8, 2], vec![2, 3, 4]] {
            let g = AbelianGroup::new(d);
            let els = elements(&g);
            for _ in 0..25 {
                let pick = |rng: &mut rand::rngs::StdRng| -> Vec<Vec<u64>> {
                    (0..rng.gen_range(0..3))
                        .map(|_| els[rng.gen_range(0..els.len())].clone())
                        .collect()
                };
                let (a, b) = (pick(&mut rng), pick(&mut rng));
                let (ha, hb) = (g.subgroup(&a), g.subgroup(&b));
                let (sa, sb) = (closure(&g, &a), closure(&g, &b));
                assert_eq!(g.subgroup_order(&ha), BigInt::from(sa.len()));
                for x in &els {
                    assert_eq!(g.contains(&ha, x), sa.contains(x));
                }
                assert_eq!(g.subgroup(&g.gens(&ha)), ha);
                let inter: BTreeSet<_> = sa.intersection(&sb).cloned().collect();
                let hi = g.intersection(&ha, &hb);
                assert_eq!(g.subgroup_order(&hi), BigInt::from(inter.len()));
                assert!(inter.iter().all(|x| g.contains(&hi, x)));
                let ann = g.annihilator(&ha);
                let perp: BTreeSet<_> = els
                    .iter()
                    .filter(|y| sa.iter().all(|x| g.pairing(x, y) == 0))
                    .cloned()
                    .collect();
                assert_eq!(g.subgroup_order(&ann), BigInt::from(perp.len()));
                assert!(perp.iter().all(|y| g.contains(&ann, y)));
                let (q, _) = g.quotient(&ha);
                assert_eq!(q.order() * BigInt::from(sa.len()), g.order());
            }
        }
    }

    #[test]
    fn invariants_and_relations() {
        assert_eq!(AbelianGroup::new(vec![2, 3, 4]).invariants(), vec![2, 12]);
        // Z^2 / <(2,4),(6,8)> = Z/2 x Z/4
        let (q, imgs) =
            AbelianGroup::from_relations(2, &IntMat::from_rows(&[vec![2, 4], vec![6, 8]])).unwrap();
        assert_eq!(q.orders(), &[2, 4]);
        assert_eq!(imgs.len(), 2);
        assert!(AbelianGroup::from_relations(2, &IntMat::from_rows(&[vec![1, 1]])).is_none());
    }

    #[test]
    fn fourier_matches_definition() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for d in [vec![2, 3], vec![4, 2], vec![8], vec![3, 4, 2]] {
            let g = AbelianGroup::new(d);
            let els = elements(&g);
            let f: Vec<Complex<f64>> = els.iter().map(|_| Complex::new(rng.gen(), rng.gen())).collect();
            let mut h = f.clone();
            g.fourier(&mut h, false);
            let n = els.len() as f64;
            for y in &els {
                let want: Complex<f64> = els
                    .iter()
                    .map(|x| g.character(x, y).conj() * f[g.index(x) as usize])
                    .sum::<Complex<f64>>()
                    / n.sqrt();
                assert!((h[g.index(y) as usize] - want).norm() < 1e-10);
            }
            g.fourier(&mut h, true);
            for (a, b) in h.iter().zip(&f) {
                assert!((a - b).norm() < 1e-10);
            }
        }
    }
}
