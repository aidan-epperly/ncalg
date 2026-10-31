//! Finite p-groups given by power-conjugate (pc) presentations.
//!
//! A pc presentation of a finite p-group `G` on generators `g_0, …, g_{n-1}`
//! (the *pcgs*, top of the series first) consists of
//!
//! * power relations `g_i^p = w_i`, with `w_i` a word in `g_{i+1}, …`;
//! * conjugate relations `g_i^{-1} g_j g_i = g_j c_{ij}` for `i < j`, with `c_{ij}`
//!   a word in `g_{j+1}, …`.
//!
//! Every element has a unique *collected* normal form
//! `g_0^{e_0} g_1^{e_1} ⋯ g_{n-1}^{e_{n-1}}` with `0 ≤ e_i < p`, stored as an
//! exponent vector ([`Exps`]).  Multiplication is *collection from the left*.  It
//! needs no enumeration of the group, so `|G| = p^n` can be far beyond memory.
//!
//! On top of the collector:
//!
//! * [`Subgroup`]: induced pc sequences in canonical (fully reduced echelon)
//!   form, membership by sifting, closure, normal closure, equality and order;
//! * [`PcPresentation::lower_exponent_p_central_series`] and a check that the
//!   pcgs refines it ([`PcPresentation::series_blocks`]); the lower and upper
//!   central series, the nilpotency class and the abelian invariants of their
//!   factors ([`PcPresentation::series_factor_invariants`]);
//! * quotients by any normal subgroup, with the projection
//!   ([`PcPresentation::quotient`], [`PcProjection`]), direct and central
//!   products ([`PcPresentation::direct_product`],
//!   [`PcPresentation::central_product`]);
//! * the group algebra as relations under the [`Wreath`] ordering, which are
//!   already a complete Gröbner basis ([`PcPresentation::group_algebra_relations`]);
//! * Clifford-hierarchy families with monomial representations ([`families`]).
//!
//! Index convention: [`PcPresentation::index`] maps an element to the integer
//! `Σ e_i p^{n-1-i}`, so the first generator is the most significant digit.
//! When the pcgs refines a central series top-first, the quotient by the
//! bottom of the series is "drop the low digits".

pub mod families;

use crate::algebra::FreeAlgebra;
use crate::coeff::Coeff;
use crate::order::Wreath;
use crate::poly::Polynomial;
use crate::word::Var;

/// Exponent vector of a collected word, entries in `0..p`.
pub type Exps = Vec<u8>;

/// A power-conjugate presentation of a finite p-group with all relative orders `p`.
#[derive(Clone, Debug)]
pub struct PcPresentation {
    p: u8,
    names: Vec<String>,
    /// `power[i]` = exponent vector of `g_i^p` (support in `i+1..`).
    power: Vec<Exps>,
    /// `conj[i][j]` (i < j) = exponent vector of `g_i^{-1} g_j g_i` (= `g_j c_ij`).
    conj: Vec<Vec<Exps>>,
    /// letter expansions of `power[i]` and `conj[i][j]`.
    power_words: Vec<Vec<usize>>,
    conj_words: Vec<Vec<Vec<usize>>>,
}

fn letters(e: &[u8]) -> Vec<usize> {
    let mut w = Vec::new();
    for (k, &x) in e.iter().enumerate() {
        for _ in 0..x {
            w.push(k);
        }
    }
    w
}

/// Builder for a [`PcPresentation`]: every relation defaults to the trivial one
/// (`g_i^p = 1`, generators commute).
#[derive(Clone, Debug)]
pub struct PcBuilder {
    p: u8,
    names: Vec<String>,
    power: Vec<Exps>,
    conj: Vec<Vec<Exps>>,
}

impl PcBuilder {
    /// A presentation of a `p`-group on the named generators, with all relations trivial.
    pub fn new<S: Into<String>, I: IntoIterator<Item = S>>(p: u8, names: I) -> Self {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        let n = names.len();
        let power = vec![vec![0u8; n]; n];
        let conj = (0..n)
            .map(|_i| {
                (0..n)
                    .map(|j| {
                        let mut e = vec![0u8; n];
                        e[j] = 1;
                        e
                    })
                    .collect()
            })
            .collect();
        PcBuilder {
            p,
            names,
            power,
            conj,
        }
    }

    /// `g_i^p = w`, with `w` given as `(generator, exponent)` pairs.
    pub fn power(mut self, i: usize, w: &[(usize, u8)]) -> Self {
        let n = self.names.len();
        let mut e = vec![0u8; n];
        for &(k, x) in w {
            e[k] = x;
        }
        self.power[i] = e;
        self
    }

    /// `g_i^{-1} g_j g_i = g_j · c`, with `c` given as `(generator, exponent)` pairs
    /// (all generators `> j`), for `i < j`.
    pub fn conj(mut self, i: usize, j: usize, c: &[(usize, u8)]) -> Self {
        let n = self.names.len();
        let mut e = vec![0u8; n];
        e[j] = 1;
        for &(k, x) in c {
            e[k] = x;
        }
        self.conj[i][j] = e;
        self
    }

    /// The presentation; an error if the relations are malformed.
    pub fn build(self) -> Result<PcPresentation, String> {
        PcPresentation::new(self.p, self.names, self.power, self.conj)
    }
}

impl PcPresentation {
    /// Validate the shape of the relations (supports, exponents) and build.
    /// Consistency is checked separately by [`PcPresentation::check_consistency`].
    pub fn new(p: u8, names: Vec<String>, power: Vec<Exps>, conj: Vec<Vec<Exps>>) -> Result<Self, String> {
        let n = names.len();
        if p < 2 || !(2..p).all(|d| p % d != 0) {
            return Err(format!("p = {p} is not prime"));
        }
        if power.len() != n || conj.len() != n {
            return Err("relation tables have the wrong size".into());
        }
        for i in 0..n {
            if power[i].len() != n || power[i].iter().any(|&x| x >= p) {
                return Err(format!("power[{i}] malformed"));
            }
            if power[i][..=i].iter().any(|&x| x != 0) {
                return Err(format!("power[{i}] must lie in generators > {i}"));
            }
            for j in (i + 1)..n {
                let c = &conj[i][j];
                if c.len() != n || c.iter().any(|&x| x >= p) {
                    return Err(format!("conj[{i}][{j}] malformed"));
                }
                if c[..j].iter().any(|&x| x != 0) || c[j] != 1 {
                    return Err(format!("conj[{i}][{j}] must be g_{j} times generators > {j}"));
                }
            }
        }
        let power_words = power.iter().map(|e| letters(e)).collect();
        let conj_words = conj
            .iter()
            .map(|row| row.iter().map(|e| letters(e)).collect())
            .collect();
        Ok(PcPresentation {
            p,
            names,
            power,
            conj,
            power_words,
            conj_words,
        })
    }

    /// The prime `p`.
    pub fn p(&self) -> u8 {
        self.p
    }
    /// Number of pc generators (`log_p |G|`).
    pub fn len(&self) -> usize {
        self.names.len()
    }
    /// No generators (the trivial group)?
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
    /// Generator names.
    pub fn names(&self) -> &[String] {
        &self.names
    }
    /// `log_p |G|`.
    pub fn log_order(&self) -> usize {
        self.names.len()
    }
    /// Exponents of `g_i^p`.
    pub fn power_relation(&self, i: usize) -> &Exps {
        &self.power[i]
    }
    /// Exponents of `g_i^{-1} g_j g_i` (for `i < j`).
    pub fn conj_relation(&self, i: usize, j: usize) -> &Exps {
        &self.conj[i][j]
    }

    /// The identity element.
    pub fn identity(&self) -> Exps {
        vec![0; self.len()]
    }
    /// The generator `g_i`.
    pub fn gen(&self, i: usize) -> Exps {
        let mut e = self.identity();
        e[i] = 1;
        e
    }
    /// Is `e` the identity?
    pub fn is_identity(&self, e: &[u8]) -> bool {
        e.iter().all(|&x| x == 0)
    }

    /// Multiply `e` (collected) on the right by the letters in `stack`
    /// (popped from the end, i.e. `stack` holds the pending word reversed).
    fn collect(&self, e: &mut Exps, stack: &mut Vec<usize>) {
        let n = self.len();
        let mut seq: Vec<usize> = Vec::new();
        while let Some(i) = stack.pop() {
            // e = x · g_i^{e_i} · t, with t the tail in generators > i
            seq.clear();
            let wrap = e[i] + 1 == self.p;
            if wrap {
                seq.extend_from_slice(&self.power_words[i]);
            }
            for j in (i + 1)..n {
                let ej = e[j];
                if ej > 0 {
                    e[j] = 0;
                    for _ in 0..ej {
                        seq.extend_from_slice(&self.conj_words[i][j]);
                    }
                }
            }
            e[i] = if wrap { 0 } else { e[i] + 1 };
            // e · g_i = x g_i^{e_i+1} t^{g_i}: pending = (power word if wrapped) ++ t^{g_i}
            stack.extend(seq.iter().rev());
        }
    }

    /// `a · b`.
    pub fn mul(&self, a: &[u8], b: &[u8]) -> Exps {
        let mut e = a.to_vec();
        let mut stack: Vec<usize> = letters(b).into_iter().rev().collect();
        self.collect(&mut e, &mut stack);
        e
    }

    /// `a · g_i`.
    pub fn mul_gen(&self, a: &[u8], i: usize) -> Exps {
        let mut e = a.to_vec();
        let mut stack = vec![i];
        self.collect(&mut e, &mut stack);
        e
    }

    /// Product of a word given as a list of generator indices.
    pub fn word(&self, w: &[usize]) -> Exps {
        let mut e = self.identity();
        let mut stack: Vec<usize> = w.iter().rev().copied().collect();
        self.collect(&mut e, &mut stack);
        e
    }

    /// The inverse of `a`.
    pub fn inv(&self, a: &[u8]) -> Exps {
        let mut e = a.to_vec();
        let mut x = self.identity();
        for i in 0..self.len() {
            if e[i] != 0 {
                let c = self.p - e[i];
                for _ in 0..c {
                    e = self.mul_gen(&e, i);
                    x = self.mul_gen(&x, i);
                }
            }
        }
        debug_assert!(self.is_identity(&e));
        x
    }

    /// `a^k`.
    pub fn pow(&self, a: &[u8], k: u64) -> Exps {
        // binary powering without the identity factor and without the last
        // squaring: pow(a, 1) collects nothing (it used to collect three
        // times, on every sift step for p = 2)
        let mut result: Option<Exps> = None;
        let mut base = a.to_vec();
        let mut k = k;
        while k > 0 {
            if k & 1 == 1 {
                result = Some(match result {
                    None => base.clone(),
                    Some(r) => self.mul(&r, &base),
                });
            }
            k >>= 1;
            if k > 0 {
                base = self.mul(&base, &base);
            }
        }
        result.unwrap_or_else(|| self.identity())
    }

    /// Order of `a` (a power of p).
    pub fn order(&self, a: &[u8]) -> u64 {
        let mut x = a.to_vec();
        let mut ord = 1u64;
        while !self.is_identity(&x) {
            x = self.pow(&x, self.p as u64);
            ord *= self.p as u64;
        }
        ord
    }

    /// `[a, b] = a^{-1} b^{-1} a b`.
    pub fn comm(&self, a: &[u8], b: &[u8]) -> Exps {
        let ab = self.mul(a, b);
        let ba = self.mul(b, a);
        self.mul(&self.inv(&ba), &ab)
    }

    /// `b^{-1} a b`.
    pub fn conjugate(&self, a: &[u8], b: &[u8]) -> Exps {
        self.mul(&self.mul(&self.inv(b), a), b)
    }

    /// `Σ e_i p^{n-1-i}` (first generator most significant).  Panics if it overflows `u64`.
    pub fn index(&self, e: &[u8]) -> u64 {
        let p = self.p as u64;
        e.iter().fold(0u64, |acc, &x| {
            acc.checked_mul(p)
                .and_then(|v| v.checked_add(x as u64))
                .expect("index overflows u64")
        })
    }

    /// The element with index `idx`, the inverse of [`index`](Self::index).
    pub fn from_index(&self, mut idx: u64) -> Exps {
        let n = self.len();
        let p = self.p as u64;
        let mut e = vec![0u8; n];
        for i in (0..n).rev() {
            e[i] = (idx % p) as u8;
            idx /= p;
        }
        e
    }

    /// Consistency test for pc presentations with all relative orders p
    /// (Vaughan-Lee; Sims §9.8).  Returns the first failing test.
    pub fn check_consistency(&self) -> Result<(), String> {
        let n = self.len();
        let p = self.p as u64;
        let g = |i: usize| self.gen(i);
        let gp = |i: usize, k: u64| {
            let mut e = self.identity();
            for _ in 0..k {
                e = self.mul_gen(&e, i);
            }
            e
        };
        for k in 0..n {
            for j in 0..k {
                for i in 0..j {
                    // g_k (g_j g_i) = (g_k g_j) g_i
                    let l = self.mul(&self.mul(&g(k), &g(j)), &g(i));
                    let r = self.mul(&g(k), &self.mul(&g(j), &g(i)));
                    if l != r {
                        return Err(format!("associativity fails for g{k} g{j} g{i}"));
                    }
                }
            }
        }
        for j in 0..n {
            for i in 0..j {
                // g_j^p g_i = g_j^{p-1} (g_j g_i)
                let l = self.mul(&self.power[j], &g(i));
                let r = self.mul(&gp(j, p - 1), &self.mul(&g(j), &g(i)));
                if l != r {
                    return Err(format!("g{j}^p g{i} inconsistent"));
                }
                // g_j g_i^p = (g_j g_i) g_i^{p-1}
                let l = self.mul(&g(j), &self.power[i]);
                let r = self.mul(&self.mul(&g(j), &g(i)), &gp(i, p - 1));
                if l != r {
                    return Err(format!("g{j} g{i}^p inconsistent"));
                }
            }
        }
        for i in 0..n {
            // g_i g_i^p = g_i^p g_i
            let l = self.mul(&g(i), &self.power[i]);
            let r = self.mul(&self.power[i], &g(i));
            if l != r {
                return Err(format!("g{i} does not commute with g{i}^p"));
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------ subgroups

    /// Subgroup generated by `gens`.
    pub fn subgroup(&self, gens: &[Exps]) -> Subgroup {
        let mut s = Subgroup {
            slots: vec![None; self.len()],
        };
        self.extend_subgroup(&mut s, gens.to_vec());
        s.canonicalize(self);
        s
    }

    fn depth(e: &[u8]) -> Option<usize> {
        e.iter().position(|&x| x != 0)
    }

    /// Residue of `g` after sifting through the (echelon) subgroup: identity iff `g ∈ S`.
    fn sift(&self, s: &Subgroup, g: &[u8]) -> Exps {
        let mut g = g.to_vec();
        while let Some(d) = Self::depth(&g) {
            match &s.slots[d] {
                Some(h) => {
                    let c = self.p - g[d];
                    g = self.mul(&g, &self.pow(h, c as u64));
                }
                None => return g,
            }
        }
        g
    }

    fn extend_subgroup(&self, s: &mut Subgroup, mut queue: Vec<Exps>) {
        while let Some(g) = queue.pop() {
            let r = self.sift(s, &g);
            let d = match Self::depth(&r) {
                Some(d) => d,
                None => continue,
            };
            let r = self.pow(
                &r,
                crate::arith::int::inv_mod(r[d] as u64, self.p as u64).expect("p prime"),
            );
            debug_assert_eq!(r[d], 1);
            queue.push(self.pow(&r, self.p as u64));
            for h in s.slots.iter().flatten() {
                queue.push(self.comm(&r, h));
            }
            s.slots[d] = Some(r);
        }
    }

    /// Normal closure in G of the subgroup generated by `gens`.
    pub fn normal_closure(&self, gens: &[Exps]) -> Subgroup {
        let mut s = Subgroup {
            slots: vec![None; self.len()],
        };
        self.extend_subgroup(&mut s, gens.to_vec());
        loop {
            let mut extra = Vec::new();
            for h in s.slots.iter().flatten() {
                for i in 0..self.len() {
                    let c = self.conjugate(h, &self.gen(i));
                    if !self.is_identity(&self.sift(&s, &c)) {
                        extra.push(c);
                    }
                }
            }
            if extra.is_empty() {
                break;
            }
            self.extend_subgroup(&mut s, extra);
        }
        s.canonicalize(self);
        s
    }

    /// Is `g` in `s`?
    pub fn contains(&self, s: &Subgroup, g: &[u8]) -> bool {
        self.is_identity(&self.sift(s, g))
    }

    /// Is `s` normal in G?
    pub fn is_normal(&self, s: &Subgroup) -> bool {
        s.gens()
            .all(|h| (0..self.len()).all(|i| self.contains(s, &self.conjugate(h, &self.gen(i)))))
    }

    /// The whole group as a subgroup.
    pub fn whole_group(&self) -> Subgroup {
        Subgroup {
            slots: (0..self.len()).map(|i| Some(self.gen(i))).collect(),
        }
    }

    /// The trivial subgroup.
    pub fn trivial_subgroup(&self) -> Subgroup {
        Subgroup {
            slots: vec![None; self.len()],
        }
    }

    // ------------------------------------------------------------------ centralizers, intersections, normalizers
    //
    // Every pcgs here refines a central series: the tails N_j = <g_j, g_{j+1}, ...>
    // are normal and N_j / N_{j+1} = Z/p is central in G / N_{j+1} (the
    // conjugate relations only add tails).  The three algorithms below walk
    // down that series one generator at a time.  Centralizer and intersection
    // are kernels of homomorphisms to elementary abelian groups at every step,
    // so they are polynomial; the normalizer needs an orbit computation.

    /// Subgroup of `c` cut out by the linear conditions `Σ a_k images[k] = 0`
    /// on exponent vectors `a` of the canonical generators `h_k` of `c`
    /// (well defined when `h ↦ image` is a homomorphism `c → (Z/p)^m`).
    fn kernel_of(&self, c: &Subgroup, images: &[Vec<u8>]) -> Subgroup {
        let gens: Vec<&Exps> = c.gens().collect();
        let r = gens.len();
        let m = images.first().map_or(0, |v| v.len());
        // null space of the m x r matrix whose column k is images[k]
        let basis: Vec<Vec<u64>> = if m == 0 {
            (0..r)
                .map(|k| (0..r).map(|i| (i == k) as u64).collect())
                .collect()
        } else {
            let f = crate::ff::field(self.p as u64, 1);
            let rows: Vec<Vec<u64>> = (0..m)
                .map(|j| (0..r).map(|k| images[k][j] as u64).collect())
                .collect();
            crate::ff::FqMat::from_rows(f, &rows).nullspace()
        };
        let mut out = Vec::new();
        for a in &basis {
            let mut g = self.identity();
            for (k, &x) in a.iter().enumerate() {
                if x != 0 {
                    g = self.mul(&g, &self.pow(gens[k], x));
                }
            }
            out.push(g);
        }
        self.subgroup(&out)
    }

    /// `C_G(X)`, the centralizer of a set of elements.
    pub fn centralizer(&self, xs: &[Exps]) -> Subgroup {
        self.centralizer_mod(xs, &self.trivial_subgroup())
    }

    /// `C_G(X mod M) = {g : [g, x] ∈ M for all x ∈ X}` for a normal subgroup
    /// `M`: the preimage of the centralizer of `XM/M` in `G/M`.  `M = 1` gives
    /// [`centralizer`](Self::centralizer); `X = G` and `M = Z_i` give the next
    /// term of the [upper central series](Self::upper_central_series).
    pub fn centralizer_mod(&self, xs: &[Exps], m: &Subgroup) -> Subgroup {
        let mut c = self.whole_group();
        for j in 0..self.len() {
            if xs.is_empty() || c.log_order() == 0 {
                break;
            }
            // C_j = {g : [g, x] ∈ M N_j ∀x}.  M N_j / M N_{j+1} is central in
            // G / M N_{j+1}, so g ↦ ([g, x] M N_{j+1})_x is a homomorphism on
            // C_j with kernel C_{j+1}.  It is trivial when M has an element of
            // depth j (then M N_j = M N_{j+1}); otherwise the coset is the
            // depth-j exponent left after peeling M's generators of smaller depth.
            if m.slots[j].is_some() {
                continue;
            }
            let images: Vec<Vec<u8>> = c
                .gens()
                .map(|g| {
                    xs.iter()
                        .map(|x| {
                            let k = self.comm(g, x);
                            let r = self.sift_above(m, &k, j).expect("[g, x] lies in M N_j");
                            debug_assert!(r[..j].iter().all(|&a| a == 0));
                            r[j]
                        })
                        .collect()
                })
                .collect();
            if images.iter().all(|v| v.iter().all(|&a| a == 0)) {
                continue;
            }
            c = self.kernel_of(&c, &images);
        }
        c
    }

    /// `C_G(S)` for a subgroup.
    pub fn centralizer_of(&self, s: &Subgroup) -> Subgroup {
        self.centralizer(&s.gens().cloned().collect::<Vec<_>>())
    }

    /// The center `Z(G)`.
    pub fn center(&self) -> Subgroup {
        self.centralizer(&(0..self.len()).map(|i| self.gen(i)).collect::<Vec<_>>())
    }

    /// Peel the generators of `b` at depths `< j` off `a` (multiplying on the right).
    fn sift_above(&self, b: &Subgroup, a: &[u8], j: usize) -> Option<Exps> {
        let mut g = a.to_vec();
        while let Some(d) = Self::depth(&g) {
            if d >= j {
                break;
            }
            let h = b.slots[d].as_ref()?;
            g = self.mul(&g, &self.pow(h, (self.p - g[d]) as u64));
        }
        Some(g)
    }

    /// `A ∩ B`.
    pub fn intersection(&self, a: &Subgroup, b: &Subgroup) -> Subgroup {
        // K_j = A ∩ B N_j.  If B has no generator of depth j, B N_{j+1} has
        // index p in B N_j, so it is normal there and a ↦ (coset) is a
        // homomorphism K_j → Z/p whose kernel is K_{j+1}.
        let mut k = a.clone();
        for j in 0..self.len() {
            if k.log_order() == 0 {
                break;
            }
            if b.slots[j].is_some() {
                continue;
            }
            let images: Vec<Vec<u8>> = k
                .gens()
                .map(|g| {
                    let r = self.sift_above(b, g, j).expect("element of A lies in B N_j");
                    debug_assert!(r[..j].iter().all(|&x| x == 0));
                    vec![r[j]]
                })
                .collect();
            if images.iter().all(|v| v[0] == 0) {
                continue;
            }
            k = self.kernel_of(&k, &images);
        }
        k
    }

    /// Is `a ⊆ b`?
    pub fn is_subgroup(&self, a: &Subgroup, b: &Subgroup) -> bool {
        a.gens().all(|g| self.contains(b, g))
    }

    /// `<A, B>`.
    pub fn join(&self, a: &Subgroup, b: &Subgroup) -> Subgroup {
        let gens: Vec<Exps> = a.gens().chain(b.gens()).cloned().collect();
        self.subgroup(&gens)
    }

    /// `S^g = g^{-1} S g`.
    pub fn conjugate_subgroup(&self, s: &Subgroup, g: &[u8]) -> Subgroup {
        // the pcgs refines a central series, so h^g = h [h, g] keeps the depth
        // and leading exponent of h: the conjugated induced pcgs is already one
        // of S^g and only needs canonicalising, not a new closure
        let mut t = Subgroup {
            slots: s
                .slots
                .iter()
                .map(|h| h.as_ref().map(|h| self.conjugate(h, g)))
                .collect(),
        };
        t.canonicalize(self);
        t
    }

    /// `N_G(S)`.  Walks down the series keeping `M_j = N_G(S N_j)`; each step
    /// is the stabilizer of `S N_{j+1}` in `M_j` under conjugation, found by
    /// orbit–stabilizer along the pcgs of `M_j`.  The orbit at step `j` has
    /// size `[M_j : M_{j+1}]`, so time and memory are polynomial in `n` times
    /// the largest such index (exponential only when the normalizer drops by
    /// many layers in one step).
    pub fn normalizer(&self, s: &Subgroup) -> Subgroup {
        let n = self.len();
        let mut m = self.whole_group();
        for j in 0..n {
            if s.slots[j].is_some() {
                continue;
            }
            let mut point_gens: Vec<Exps> = s.gens().cloned().collect();
            point_gens.extend((j + 1..n).map(|k| self.gen(k)));
            let omega = self.subgroup(&point_gens);
            m = self.stabilizer(&m, &omega);
        }
        m
    }

    /// Stabilizer in `u` of the subgroup `omega` under conjugation, by
    /// orbit–stabilizer along the pcgs of `u` from the bottom up.
    fn stabilizer(&self, u: &Subgroup, omega: &Subgroup) -> Subgroup {
        use std::collections::HashMap;
        let gens: Vec<Exps> = u.gens().cloned().collect();
        let mut orbit: Vec<Subgroup> = vec![omega.clone()];
        let mut trans: Vec<Exps> = vec![self.identity()];
        let mut index: HashMap<Subgroup, usize> = HashMap::new();
        index.insert(omega.clone(), 0);
        let mut stab: Vec<Exps> = Vec::new();
        for h in gens.iter().rev() {
            let img = self.conjugate_subgroup(omega, h);
            if let Some(&i) = index.get(&img) {
                // omega^{h t_i^{-1}} = omega
                stab.push(self.mul(h, &self.inv(&trans[i])));
                continue;
            }
            // the orbit of <h, U'> is the disjoint union of orbit^{h^e}, e < p
            let base = orbit.len();
            let mut hp = self.identity();
            for _ in 1..self.p {
                hp = self.mul(&hp, h);
                for i in 0..base {
                    let pt = self.conjugate_subgroup(&orbit[i], &hp);
                    let t = self.mul(&trans[i], &hp);
                    index.insert(pt.clone(), orbit.len());
                    orbit.push(pt);
                    trans.push(t);
                }
            }
        }
        self.subgroup(&stab)
    }

    // ------------------------------------------------------------------ series and quotients

    /// Lower exponent-p central series `P_1 = G ⊃ P_2 ⊃ … ⊃ P_{ℓ+1} = 1`,
    /// `P_{i+1} = [G, P_i] P_i^p`.  The returned list ends with the trivial group.
    pub fn lower_exponent_p_central_series(&self) -> Vec<Subgroup> {
        let mut series = vec![self.whole_group()];
        loop {
            let last = series.last().expect("nonempty");
            if last.log_order() == 0 {
                break;
            }
            let mut cand = Vec::new();
            for y in last.gens() {
                cand.push(self.pow(y, self.p as u64));
                for i in 0..self.len() {
                    cand.push(self.comm(&self.gen(i), y));
                }
            }
            let next = self.normal_closure(&cand);
            assert!(
                next.log_order() < last.log_order(),
                "series does not descend (not a p-group?)"
            );
            series.push(next);
        }
        series
    }

    /// Lower central series `G = γ_1 ⊃ γ_2 = [G, G] ⊃ … ⊃ γ_{c+1} = 1`,
    /// `γ_{i+1} = [G, γ_i]`: the normal closure of the commutators of the pc
    /// generators with the generators of `γ_i`.  The returned list ends with
    /// the trivial group (a finite p-group is nilpotent, so it gets there).
    pub fn lower_central_series(&self) -> Vec<Subgroup> {
        let mut series = vec![self.whole_group()];
        loop {
            let last = series.last().expect("nonempty");
            if last.log_order() == 0 {
                break;
            }
            let mut cand = Vec::new();
            for y in last.gens() {
                for i in 0..self.len() {
                    cand.push(self.comm(&self.gen(i), y));
                }
            }
            let next = self.normal_closure(&cand);
            assert!(
                next.log_order() < last.log_order(),
                "lower central series does not descend (not a p-group?)"
            );
            series.push(next);
        }
        series
    }

    /// Upper central series `1 = Z_0 ⊂ Z_1 = Z(G) ⊂ … ⊂ Z_c = G`,
    /// `Z_{i+1} = {g : [g, x] ∈ Z_i for all x}`, so `Z_{i+1}/Z_i = Z(G/Z_i)`
    /// ([`centralizer_mod`](Self::centralizer_mod) of the pc generators).
    /// Ascending: starts with the trivial group, ends with `G`.  It has as many
    /// terms as the [lower central series](Self::lower_central_series).
    pub fn upper_central_series(&self) -> Vec<Subgroup> {
        let gens: Vec<Exps> = (0..self.len()).map(|i| self.gen(i)).collect();
        let mut series = vec![self.trivial_subgroup()];
        loop {
            let last = series.last().expect("nonempty");
            if last.log_order() == self.len() {
                break;
            }
            let next = self.centralizer_mod(&gens, last);
            assert!(
                next.log_order() > last.log_order(),
                "upper central series stalls (not a p-group?)"
            );
            series.push(next);
        }
        series
    }

    /// Nilpotency class `c` (`γ_{c+1} = 1`, `Z_c = G`); 0 for the trivial group.
    pub fn nilpotency_class(&self) -> usize {
        self.lower_central_series().len() - 1
    }

    /// Abelian invariants of `A/B`, for `B ⊴ A ⊆ G` with `A/B` abelian: the
    /// orders `p^e > 1` of the cyclic factors, ascending (`A/B ≅ ⊕ ℤ/p^e`).
    /// The generators of `A` at depths without a generator of `B` generate
    /// `A/B` with relative orders `p`; the invariants are the Smith normal form
    /// of their power relations.  An error if `B ⊄ A`, `B` is not normal in
    /// `A`, or `A/B` is not abelian.
    pub fn abelian_invariants(&self, a: &Subgroup, b: &Subgroup) -> Result<Vec<u64>, String> {
        if !self.is_subgroup(b, a) {
            return Err("B is not contained in A".into());
        }
        let a_gens: Vec<&Exps> = a.gens().collect();
        for h in b.gens() {
            if a_gens.iter().any(|x| !self.contains(b, &self.conjugate(h, x))) {
                return Err("B is not normal in A".into());
            }
        }
        for (k, x) in a_gens.iter().enumerate() {
            if a_gens[k + 1..]
                .iter()
                .any(|y| !self.contains(b, &self.comm(x, y)))
            {
                return Err("A/B is not abelian".into());
            }
        }
        // generators of A/B: A's canonical generators at depths outside B
        let top: Vec<usize> = a.depths().into_iter().filter(|&d| b.slots[d].is_none()).collect();
        if top.is_empty() {
            return Ok(Vec::new());
        }
        let index = |d: usize| top.iter().position(|&t| t == d);
        // coordinates of yB in the generators of A/B (A/B abelian, so additive)
        let coords = |y: &[u8]| -> Vec<i64> {
            let mut out = vec![0i64; top.len()];
            let mut y = y.to_vec();
            while let Some(d) = Self::depth(&y) {
                match (&b.slots[d], index(d)) {
                    (Some(h), _) => y = self.mul(&y, &self.pow(h, (self.p - y[d]) as u64)),
                    (None, Some(k)) => {
                        let e = y[d];
                        out[k] = e as i64;
                        let g = a.slots[d].as_ref().expect("depth of A");
                        y = self.mul(&self.pow(g, (self.p - e) as u64), &y);
                    }
                    (None, None) => unreachable!("an element of A has depths of A only"),
                }
            }
            out
        };
        // relations p t_k - (t_k^p) on Z^top
        let rows: Vec<Vec<i64>> = top
            .iter()
            .enumerate()
            .map(|(k, &d)| {
                let g = a.slots[d].as_ref().expect("depth of A");
                let mut row: Vec<i64> = coords(&self.pow(g, self.p as u64))
                    .into_iter()
                    .map(|c| -c)
                    .collect();
                row[k] += self.p as i64;
                row
            })
            .collect();
        let smith = crate::zmod::IntMat::from_rows(&rows).smith();
        let mut inv: Vec<u64> = smith
            .diag
            .iter()
            .filter_map(|x| {
                let x: u64 =
                    num_traits::ToPrimitive::to_u64(&num_traits::Signed::abs(x)).expect("small invariant");
                (x > 1).then_some(x)
            })
            .collect();
        inv.sort_unstable();
        Ok(inv)
    }

    /// [`abelian_invariants`](Self::abelian_invariants) of the consecutive
    /// factors of a series given in either direction (descending like the
    /// [lower central series](Self::lower_central_series), ascending like the
    /// [upper central series](Self::upper_central_series)): entry `i` belongs to
    /// the factor between `series[i]` and `series[i + 1]`.
    pub fn series_factor_invariants(&self, series: &[Subgroup]) -> Result<Vec<Vec<u64>>, String> {
        series
            .windows(2)
            .map(|w| {
                if w[0].log_order() >= w[1].log_order() {
                    self.abelian_invariants(&w[0], &w[1])
                } else {
                    self.abelian_invariants(&w[1], &w[0])
                }
            })
            .collect()
    }

    /// If the pcgs refines the lower exponent-p central series, i.e. every term
    /// is `⟨g_k : k ≥ s_i⟩`, the layer dimensions `d_i`; otherwise `None`.
    pub fn series_blocks(&self) -> Option<Vec<usize>> {
        let n = self.len();
        let mut d = Vec::new();
        let mut prev = n;
        for s in self.lower_exponent_p_central_series() {
            let len = s.log_order();
            let start = n - len;
            for (k, slot) in s.slots.iter().enumerate() {
                let ok = match slot {
                    Some(h) => k >= start && *h == self.gen(k),
                    None => k < start,
                };
                if !ok {
                    return None;
                }
            }
            if prev != n || len != n {
                d.push(prev - len);
            }
            prev = len;
        }
        Some(d.into_iter().filter(|&x| x > 0).collect())
    }

    /// Quotient by the subgroup `N` generated by the pcgs generators in
    /// `positions`.  `N` must be central, and those generators must form an
    /// induced pcgs of `N` (their powers stay inside `N`).  The quotient's pcgs
    /// is the remaining generators, in order; since `N` is central, its
    /// relations are `G`'s with `N`'s exponents deleted.  The checked special
    /// case of [`quotient`](Self::quotient), which it calls.
    pub fn quotient_central(&self, positions: &[usize]) -> Result<PcPresentation, String> {
        let n = self.len();
        let inn: Vec<bool> = (0..n).map(|k| positions.contains(&k)).collect();
        for &k in positions {
            for i in 0..n {
                if self.conjugate(&self.gen(k), &self.gen(i)) != self.gen(k) {
                    return Err(format!("generator {} is not central", self.names[k]));
                }
            }
            if self.power[k].iter().enumerate().any(|(j, &x)| x != 0 && !inn[j]) {
                return Err(format!("power of {} leaves N", self.names[k]));
            }
        }
        let gens: Vec<Exps> = positions.iter().map(|&k| self.gen(k)).collect();
        self.quotient(&self.subgroup(&gens)).map(|(q, _)| q)
    }

    /// `G/N` for a normal subgroup `N`, and the projection.  The quotient's
    /// pcgs is `G`'s generators at the depths where `N` has none, in order:
    /// with `N_j` the tails of `G`'s series, `G N_j / G N_{j+1}` is `ℤ/p` exactly
    /// at those depths and central in `G / N N_{j+1}`, so the images form a pcgs
    /// refining a central series of `G/N`.  Its relations are the images of the
    /// kept generators' relations under the projection ([`PcProjection::apply`]),
    /// so `N` need not be spanned by generators or sit at the bottom of the
    /// series (for example a diagonal subgroup of a center, or `Z_i` for `i ≥ 2`).
    /// An error if `N` is not normal.
    pub fn quotient(&self, n: &Subgroup) -> Result<(PcPresentation, PcProjection), String> {
        if n.slots.len() != self.len() {
            return Err("subgroup of another group".into());
        }
        if !self.is_normal(n) {
            return Err("the subgroup is not normal".into());
        }
        let top: Vec<usize> = (0..self.len()).filter(|&d| n.slots[d].is_none()).collect();
        let proj = PcProjection {
            source: self.clone(),
            kernel: n.clone(),
            top,
        };
        let top = &proj.top;
        let names = top.iter().map(|&k| self.names[k].clone()).collect();
        let power = top
            .iter()
            .map(|&k| proj.apply(&self.pow(&self.gen(k), self.p as u64)))
            .collect();
        let conj = top
            .iter()
            .map(|&i| {
                top.iter()
                    .map(|&j| proj.apply(&self.conjugate(&self.gen(j), &self.gen(i))))
                    .collect()
            })
            .collect();
        let q = PcPresentation::new(self.p, names, power, conj)?;
        Ok((q, proj))
    }

    /// The direct product `G × H` (same `p`).  Its pcgs is `G`'s generators
    /// followed by `H`'s; the conjugate relations between the two are trivial,
    /// so the pcgs refines the central series `G_i × H`, then `1 × H_j`.  Names
    /// shared by the factors get the suffixes `_1` and `_2`.  An element of `G`
    /// is `(x, 0, …, 0)`, one of `H` is `(0, …, 0, y)`.
    pub fn direct_product(&self, other: &PcPresentation) -> Result<PcPresentation, String> {
        if self.p != other.p {
            return Err(format!("different primes {} and {}", self.p, other.p));
        }
        let (n, m) = (self.len(), other.len());
        let names: Vec<String> = self
            .names
            .iter()
            .map(|x| {
                if other.names.contains(x) {
                    format!("{x}_1")
                } else {
                    x.clone()
                }
            })
            .chain(other.names.iter().map(|x| {
                if self.names.contains(x) {
                    format!("{x}_2")
                } else {
                    x.clone()
                }
            }))
            .collect();
        let left = |e: &Exps| -> Exps { e.iter().copied().chain(std::iter::repeat_n(0, m)).collect() };
        let right = |e: &Exps| -> Exps { std::iter::repeat_n(0, n).chain(e.iter().copied()).collect() };
        let mut power: Vec<Exps> = self.power.iter().map(left).collect();
        power.extend(other.power.iter().map(right));
        let mut conj: Vec<Vec<Exps>> = Vec::with_capacity(n + m);
        for i in 0..n + m {
            let row = (0..n + m)
                .map(|j| match (i < n, j < n) {
                    (true, true) => left(&self.conj[i][j]),
                    (false, false) => right(&other.conj[i - n][j - n]),
                    _ => {
                        let mut e = vec![0u8; n + m];
                        e[j] = 1;
                        e
                    }
                })
                .collect();
            conj.push(row);
        }
        PcPresentation::new(self.p, names, power, conj)
    }

    /// The central product of `G` and `H` along `identification`: pairs
    /// `(z_k, w_k)` of central elements, `z_k ∈ Z(G)`, `w_k ∈ Z(H)`, such that
    /// `z_k ↦ w_k` defines an isomorphism `⟨z_k⟩ → ⟨w_k⟩`.  The result is
    /// `(G × H) / {(z, φ(z)^{-1})}`, with the projection from the
    /// [direct product](Self::direct_product).  An error if an element is not
    /// central or the map is not an isomorphism.
    pub fn central_product(
        &self,
        other: &PcPresentation,
        identification: &[(Exps, Exps)],
    ) -> Result<(PcPresentation, PcProjection), String> {
        let central =
            |g: &PcPresentation, x: &Exps| (0..g.len()).all(|i| g.comm(x, &g.gen(i)) == g.identity());
        for (z, w) in identification {
            if z.len() != self.len() || w.len() != other.len() {
                return Err("identification element of the wrong length".into());
            }
            if !central(self, z) || !central(other, w) {
                return Err("identified elements must be central".into());
            }
        }
        let d = self.direct_product(other)?;
        let diag: Vec<Exps> = identification
            .iter()
            .map(|(z, w)| z.iter().copied().chain(other.inv(w)).collect())
            .collect();
        let n = d.subgroup(&diag);
        let zs: Vec<Exps> = identification.iter().map(|(z, _)| z.clone()).collect();
        let ws: Vec<Exps> = identification.iter().map(|(_, w)| w.clone()).collect();
        let (a, b) = (self.subgroup(&zs).log_order(), other.subgroup(&ws).log_order());
        if a != b || n.log_order() != a {
            return Err("the identification is not an isomorphism of the generated subgroups".into());
        }
        d.quotient(&n)
    }

    /// The group algebra `K[G]` as relations in the free algebra on the pc
    /// generators, under [`Wreath::descending`].  For a consistent presentation
    /// these relations are a complete Gröbner basis and the normal words are
    /// the collected words (`p^n` of them).
    pub fn group_algebra_relations<C: Coeff>(&self) -> (FreeAlgebra, Vec<Polynomial<C>>) {
        let n = self.len();
        let alg = FreeAlgebra::with_order(self.names.clone(), Wreath::descending(n));
        let w = |e: &Exps| -> Polynomial<C> {
            let l: Vec<Var> = letters(e).into_iter().map(|k| k as Var).collect();
            crate::relations::word(&alg, &l)
        };
        let mut rels = Vec::new();
        for i in 0..n {
            let gi: Vec<Var> = vec![i as Var; self.p as usize];
            rels.push(crate::relations::word::<C>(&alg, &gi) - w(&self.power[i]));
            for j in (i + 1)..n {
                // g_j g_i = g_i (g_i^{-1} g_j g_i)
                let lhs = crate::relations::word::<C>(&alg, &[j as Var, i as Var]);
                let mut rhs_l = vec![i as Var];
                rhs_l.extend(letters(&self.conj[i][j]).into_iter().map(|k| k as Var));
                rels.push(lhs - crate::relations::word::<C>(&alg, &rhs_l));
            }
        }
        (alg, rels)
    }
}

/// The projection `G → G/N` of [`PcPresentation::quotient`].
#[derive(Clone, Debug)]
pub struct PcProjection {
    source: PcPresentation,
    kernel: Subgroup,
    /// positions in `G`'s pcgs of the quotient's generators
    top: Vec<usize>,
}

impl PcProjection {
    /// The image of `x ∈ G` in the quotient's exponents.  Walks down `G`'s
    /// series: at a depth of `N` the leading exponent is removed by a power of
    /// `N`'s generator on the right (the coset is unchanged); at any other
    /// depth it is the next quotient exponent, removed on the left by the
    /// generator's power.  So `x = (Π g_{kept}^{e}) · n` with `n ∈ N`.
    pub fn apply(&self, x: &[u8]) -> Exps {
        let g = &self.source;
        let mut out = vec![0u8; self.top.len()];
        let mut y = x.to_vec();
        while let Some(d) = PcPresentation::depth(&y) {
            match &self.kernel.slots[d] {
                Some(h) => y = g.mul(&y, &g.pow(h, (g.p - y[d]) as u64)),
                None => {
                    let k = self.top.binary_search(&d).expect("a kept depth");
                    let e = y[d];
                    out[k] = e;
                    // the inverse of g_d^e exactly (g_d^{p-e} differs from it
                    // by the power relation when g_d has order above p)
                    y = g.mul(&g.inv(&g.pow(&g.gen(d), e as u64)), &y);
                }
            }
        }
        out
    }

    /// A preimage of `y`: the product of the kept generators' powers.
    pub fn lift(&self, y: &[u8]) -> Exps {
        assert_eq!(y.len(), self.top.len());
        let word: Vec<usize> = self
            .top
            .iter()
            .zip(y)
            .flat_map(|(&k, &e)| std::iter::repeat_n(k, e as usize))
            .collect();
        self.source.word(&word)
    }

    /// The kernel `N`.
    pub fn kernel(&self) -> &Subgroup {
        &self.kernel
    }

    /// The group projected from.
    pub fn source(&self) -> &PcPresentation {
        &self.source
    }

    /// Positions in the source's pcgs of the quotient's generators, in order.
    pub fn kept(&self) -> &[usize] {
        &self.top
    }
}

/// A subgroup, stored as an induced pc sequence in canonical form: one element
/// per depth that occurs, with leading exponent 1 and zeros at the depths of
/// the other elements.  Two subgroups are equal iff their canonical sequences are.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Subgroup {
    slots: Vec<Option<Exps>>,
}

impl Subgroup {
    /// `log_p |S|`.
    pub fn log_order(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }
    /// The canonical generators, by increasing depth.
    pub fn gens(&self) -> impl Iterator<Item = &Exps> {
        self.slots.iter().flatten()
    }
    /// Depths of the canonical generators.
    pub fn depths(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_some())
            .map(|(k, _)| k)
            .collect()
    }

    fn canonicalize(&mut self, g: &PcPresentation) {
        let n = self.slots.len();
        for d in (0..n).rev() {
            let Some(mut h) = self.slots[d].clone() else {
                continue;
            };
            for d2 in (d + 1)..n {
                if h[d2] != 0 {
                    if let Some(k) = &self.slots[d2] {
                        let c = g.p - h[d2];
                        h = g.mul(&h, &g.pow(k, c as u64));
                    }
                }
            }
            self.slots[d] = Some(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// D_4 = <s, r, c | s^2 = 1, r^2 = c, c^2 = 1, s^-1 r s = r c>
    fn d4() -> PcPresentation {
        PcBuilder::new(2, ["s", "r", "c"])
            .power(1, &[(2, 1)])
            .conj(0, 1, &[(2, 1)])
            .build()
            .unwrap()
    }

    #[test]
    fn d4_basic() {
        let g = d4();
        g.check_consistency().unwrap();
        let (s, r) = (g.gen(0), g.gen(1));
        assert_eq!(g.order(&r), 4);
        assert_eq!(g.order(&s), 2);
        // s r s^-1 = r^-1
        assert_eq!(g.conjugate(&r, &s), g.inv(&r));
        // all 8 elements distinct, inverses correct
        for a in 0..8 {
            let ea = g.from_index(a);
            assert_eq!(g.index(&ea), a);
            assert!(g.is_identity(&g.mul(&ea, &g.inv(&ea))));
            for b in 0..8 {
                for c in 0..8 {
                    let (eb, ec) = (g.from_index(b), g.from_index(c));
                    assert_eq!(g.mul(&g.mul(&ea, &eb), &ec), g.mul(&ea, &g.mul(&eb, &ec)));
                }
            }
        }
        assert_eq!(g.series_blocks(), Some(vec![2, 1]));
        let z = g.subgroup(std::slice::from_ref(&r));
        assert_eq!(z.log_order(), 2);
        assert!(g.is_normal(&z));
        assert!(!g.is_normal(&g.subgroup(std::slice::from_ref(&s))));
        assert_eq!(g.normal_closure(std::slice::from_ref(&s)).log_order(), 2); // <s, c>
        let q = g.quotient_central(&[2]).unwrap();
        assert_eq!(q.series_blocks(), Some(vec![2]));
    }

    #[test]
    fn inconsistent_is_detected() {
        // r^2 = c but c not central in a way that contradicts: claim s^-1 c s = c s ... use s^-1 r s = r^-1 without c
        let bad = PcBuilder::new(2, ["s", "r", "c"])
            .power(1, &[(2, 1)])
            .conj(0, 1, &[])
            .conj(0, 2, &[])
            .conj(1, 2, &[])
            .power(0, &[(2, 1)])
            .power(2, &[])
            .build()
            .unwrap();
        // this one is consistent (Z4 x Z2-like); now break it
        bad.check_consistency().unwrap();
        let broken = PcBuilder::new(2, ["a", "b"])
            .power(0, &[(1, 1)])
            .conj(0, 1, &[])
            .build()
            .unwrap();
        broken.check_consistency().unwrap(); // Z4
        let broken2 = PcBuilder::new(3, ["a", "b", "c"])
            .power(0, &[(2, 1)])
            .conj(0, 1, &[(2, 1)])
            .conj(1, 2, &[]) // fine
            .build()
            .unwrap();
        broken2.check_consistency().unwrap(); // Heisenberg-like mod 3 with a^3 = c
        let wrong = PcBuilder::new(2, ["a", "b", "c"])
            .power(1, &[(2, 1)])
            .conj(0, 1, &[(2, 1)])
            .conj(0, 2, &[])
            .power(2, &[])
            .power(0, &[(1, 1)]) // a^2 = b, but then a commutes with b, contradicting a^-1 b a = b c
            .build()
            .unwrap();
        assert!(wrong.check_consistency().is_err());
    }

    #[test]
    fn heisenberg_mod_3() {
        let g = PcBuilder::new(3, ["x", "y", "z"])
            .conj(0, 1, &[(2, 1)])
            .build()
            .unwrap();
        g.check_consistency().unwrap();
        assert_eq!(g.series_blocks(), Some(vec![2, 1]));
        let (x, y) = (g.gen(0), g.gen(1));
        assert_eq!(g.comm(&y, &x), g.gen(2));
        assert_eq!(g.order(&x), 3);
    }

    fn elements(g: &PcPresentation) -> Vec<Exps> {
        (0..(g.p() as u64).pow(g.len() as u32))
            .map(|i| g.from_index(i))
            .collect()
    }

    fn brute_closure(g: &PcPresentation, gens: &[Exps]) -> std::collections::BTreeSet<Exps> {
        let mut s: std::collections::BTreeSet<Exps> = [g.identity()].into_iter().collect();
        loop {
            let new: Vec<Exps> = s
                .iter()
                .flat_map(|x| gens.iter().map(move |y| g.mul(x, y)))
                .filter(|z| !s.contains(z))
                .collect();
            if new.is_empty() {
                return s;
            }
            s.extend(new);
        }
    }

    fn check_algorithms(g: &PcPresentation, seed: u64, trials: usize) {
        use rand::{Rng, SeedableRng};
        use std::collections::BTreeSet;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let els = elements(g);
        let set = |s: &Subgroup| brute_closure(g, &s.gens().cloned().collect::<Vec<_>>());
        let z: BTreeSet<Exps> = els
            .iter()
            .filter(|x| els.iter().all(|y| g.mul(x, y) == g.mul(y, x)))
            .cloned()
            .collect();
        assert_eq!(set(&g.center()), z);
        for _ in 0..trials {
            let pick = |rng: &mut rand::rngs::StdRng| -> Vec<Exps> {
                (0..rng.gen_range(0..3))
                    .map(|_| els[rng.gen_range(0..els.len())].clone())
                    .collect()
            };
            let (a, b) = (g.subgroup(&pick(&mut rng)), g.subgroup(&pick(&mut rng)));
            let (sa, sb) = (set(&a), set(&b));
            // conjugation by canonicalising only agrees with the full closure
            let x = &els[rng.gen_range(0..els.len())];
            let conj: Vec<Exps> = a.gens().map(|h| g.conjugate(h, x)).collect();
            assert_eq!(g.conjugate_subgroup(&a, x), g.subgroup(&conj));
            for k in 0..5u64 {
                let mut want = g.identity();
                for _ in 0..k {
                    want = g.mul(&want, x);
                }
                assert_eq!(g.pow(x, k), want, "pow({x:?}, {k})");
            }
            let cen: BTreeSet<Exps> = els
                .iter()
                .filter(|x| sa.iter().all(|y| g.mul(x, y) == g.mul(y, x)))
                .cloned()
                .collect();
            assert_eq!(set(&g.centralizer_of(&a)), cen);
            let inter: BTreeSet<Exps> = sa.intersection(&sb).cloned().collect();
            assert_eq!(set(&g.intersection(&a, &b)), inter);
            let nor: BTreeSet<Exps> = els
                .iter()
                .filter(|x| sa.iter().all(|y| sa.contains(&g.conjugate(y, x))))
                .cloned()
                .collect();
            assert_eq!(set(&g.normalizer(&a)), nor);
        }
    }

    #[test]
    fn centralizer_intersection_normalizer_brute_force() {
        check_algorithms(&d4(), 1, 30);
        let h3 = PcBuilder::new(3, ["x", "y", "z"])
            .conj(0, 1, &[(2, 1)])
            .build()
            .unwrap();
        check_algorithms(&h3, 2, 30);
        let mut tested = 0;
        for (name, m) in [("PauliCZ", 2), ("XS", 2), ("XT", 2), ("D4N", 2), ("PauliCCZ", 3)] {
            let f = families::by_name(name, m).unwrap();
            if f.pres.len() <= 11 {
                check_algorithms(&f.pres, 3, 15);
                tested += 1;
            }
        }
        assert!(tested >= 3);
    }
}
