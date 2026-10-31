//! Admissible monomial orderings on words.
//!
//! An ordering is *admissible* if it is a well-founded total order on words
//! that is compatible with multiplication on both sides
//! (`u < v` implies `a·u·b < a·v·b`).  All orderings in this module are
//! admissible.  The degree-compatible ones ([`DegLex`], [`DegRevLex`],
//! [`WeightedDegLex`] with positive weights) additionally satisfy
//! `|u| < |v| ⇒ u < v`, which is what makes degree-truncated Gröbner bases
//! meaningful (see [`crate::groebner`]).
//!
//! Orderings are used through the object-safe trait [`MonomialOrder`] so a
//! [`crate::FreeAlgebra`] can hold any of them.

use crate::word::Var;
use std::cmp::Ordering;
use std::fmt::Debug;

/// An admissible monomial ordering on words.
pub trait MonomialOrder: Send + Sync + Debug {
    /// Compare two words.  `Greater` means `a` is the larger monomial.
    fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering;

    /// Human readable name, also used to test two algebras for compatibility.
    fn name(&self) -> String;

    /// Is the ordering compatible with total degree (`|u| < |v| ⇒ u < v`)?
    fn is_degree_compatible(&self) -> bool;

    /// Compare two generators as single-letter words.
    fn cmp_vars(&self, a: Var, b: Var) -> Ordering {
        self.cmp_words(&[a], &[b])
    }
}

/// Lexicographic comparison of two equal-length words under a given variable
/// ranking (`rank[v]` larger means `v` is the larger generator).
#[inline]
fn lex_by_rank(a: &[Var], b: &[Var], rank: &dyn Fn(Var) -> u64) -> Ordering {
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x != y {
            return rank(x).cmp(&rank(y));
        }
    }
    a.len().cmp(&b.len())
}

/// Degree-lexicographic ordering: first by length, then lexicographically
/// from the left with `x_0 < x_1 < …` (or a custom ranking of generators).
///
/// This is the standard ordering for noncommutative Gröbner bases.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DegLex {
    /// Optional ranking: `rank[v]` is the position of generator `v` in the
    /// chain `smallest … largest`.  `None` means `x_0 < x_1 < …`.
    rank: Option<Vec<u64>>,
}

impl DegLex {
    /// `x_0 < x_1 < … < x_{n-1}`.
    pub fn new() -> Self {
        DegLex { rank: None }
    }

    /// Generators listed from smallest to largest.  `order[i]` is the index of
    /// the generator in position `i` of the chain.  Every generator must
    /// appear exactly once.
    pub fn with_chain(order: &[Var]) -> Self {
        let n = order.len();
        let mut rank = vec![u64::MAX; n];
        for (pos, &v) in order.iter().enumerate() {
            assert!((v as usize) < n, "generator index out of range");
            assert!(rank[v as usize] == u64::MAX, "duplicate generator in chain");
            rank[v as usize] = pos as u64;
        }
        DegLex { rank: Some(rank) }
    }

    #[inline]
    fn rank(&self, v: Var) -> u64 {
        match &self.rank {
            None => v as u64,
            Some(r) => r.get(v as usize).copied().unwrap_or(v as u64),
        }
    }
}

impl MonomialOrder for DegLex {
    #[inline]
    fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering {
        match a.len().cmp(&b.len()) {
            Ordering::Equal => lex_by_rank(a, b, &|v| self.rank(v)),
            o => o,
        }
    }
    fn name(&self) -> String {
        match &self.rank {
            None => "deglex".to_string(),
            Some(r) => format!("deglex(rank={:?})", r),
        }
    }
    fn is_degree_compatible(&self) -> bool {
        true
    }
}

/// Degree-reverse-lexicographic ordering: first by length, then the words are
/// compared from the *right*, with the word having the *smaller* generator at
/// the first difference being *larger*.  This is the mirror image of
/// [`DegLex`] and is admissible.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DegRevLex {
    rank: Option<Vec<u64>>,
}

impl DegRevLex {
    /// Degree reverse lexicographic order with generators ranked by index.
    pub fn new() -> Self {
        DegRevLex { rank: None }
    }
    /// See [`DegLex::with_chain`].
    pub fn with_chain(order: &[Var]) -> Self {
        let d = DegLex::with_chain(order);
        DegRevLex { rank: d.rank }
    }
    #[inline]
    fn rank(&self, v: Var) -> u64 {
        match &self.rank {
            None => v as u64,
            Some(r) => r.get(v as usize).copied().unwrap_or(v as u64),
        }
    }
}

impl MonomialOrder for DegRevLex {
    #[inline]
    fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering {
        match a.len().cmp(&b.len()) {
            Ordering::Equal => {
                for (&x, &y) in a.iter().rev().zip(b.iter().rev()) {
                    if x != y {
                        // smaller generator at the first difference from the right wins
                        return self.rank(y).cmp(&self.rank(x));
                    }
                }
                Ordering::Equal
            }
            o => o,
        }
    }
    fn name(&self) -> String {
        match &self.rank {
            None => "degrevlex".to_string(),
            Some(r) => format!("degrevlex(rank={:?})", r),
        }
    }
    fn is_degree_compatible(&self) -> bool {
        true
    }
}

/// Weighted-degree lexicographic ordering: first by `Σ weight[v]`, then by
/// length, then lexicographically.  Weights must be positive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightedDegLex {
    weights: Vec<u64>,
}

impl WeightedDegLex {
    /// Panics if any weight is zero.
    pub fn new(weights: Vec<u64>) -> Self {
        assert!(weights.iter().all(|&w| w > 0), "weights must be positive");
        WeightedDegLex { weights }
    }
    #[inline]
    fn weight(&self, w: &[Var]) -> u64 {
        // checked: a wrapped sum would make the order non-admissible
        w.iter()
            .map(|&v| self.weights.get(v as usize).copied().unwrap_or(1))
            .try_fold(0u64, |a, b| a.checked_add(b))
            .expect("WeightedDegLex: word weight overflows u64")
    }
}

impl MonomialOrder for WeightedDegLex {
    fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering {
        self.weight(a)
            .cmp(&self.weight(b))
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| lex_by_rank(a, b, &|v| v as u64))
    }
    fn name(&self) -> String {
        format!("wdeglex({:?})", self.weights)
    }
    fn is_degree_compatible(&self) -> bool {
        // weighted degree refines by length but |u|<|v| does not imply w(u)<w(v)
        self.weights.iter().all(|&w| w == self.weights[0])
    }
}

/// Elimination (block) ordering: words are compared first by the number of
/// letters from the first `n_elim` generators, then by [`DegLex`].  Under this
/// ordering the Gröbner basis elements free of the first block generate the
/// elimination ideal.  Not degree compatible, but admissible and well founded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Elimination {
    n_elim: usize,
    inner: DegLex,
}

impl Elimination {
    /// Eliminate generators `0..n_elim`.
    pub fn new(n_elim: usize) -> Self {
        Elimination {
            n_elim,
            inner: DegLex::new(),
        }
    }
    #[inline]
    fn block_degree(&self, w: &[Var]) -> usize {
        w.iter().filter(|&&v| (v as usize) < self.n_elim).count()
    }
}

impl MonomialOrder for Elimination {
    fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering {
        self.block_degree(a)
            .cmp(&self.block_degree(b))
            .then_with(|| self.inner.cmp_words(a, b))
    }
    fn name(&self) -> String {
        format!("elim({})", self.n_elim)
    }
    fn is_degree_compatible(&self) -> bool {
        false
    }
}

/// Wreath-product ordering (Sims, *Computation with finitely presented
/// groups*, §2.1).
///
/// Let `a` be the largest generator occurring in either word, and write
/// `u = u_0 a u_1 a … a u_r` and `v = v_0 a v_1 … a v_s`, where the `u_i` and
/// `v_i` contain only smaller generators.  Then `u > v` if `r > s`.  If
/// `r = s`, the tuples `(u_0, …, u_r)` and `(v_0, …, v_s)` are compared
/// lexicographically from the left, each component recursively under the same
/// ordering restricted to the smaller generators.
///
/// The ordering is admissible and well founded but not degree compatible.  It
/// orients power-commutator relations `g_j g_i → g_i g_j w_ij` (with `g_i > g_j`
/// and `w_ij` in generators below `g_j`) and `g_i^p → w_i`, even though they
/// make words longer.  So the relations of a consistent pc presentation, with
/// its first generator as the largest, are a complete Gröbner basis whose
/// normal words are the collected words; see [`crate::pcgroup`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wreath {
    /// `rank[v]` is the position of generator `v` in the chain `smallest … largest`.
    /// `None` means `x_0 < x_1 < …`.
    rank: Option<Vec<u64>>,
}

impl Wreath {
    /// `x_0 < x_1 < … < x_{n-1}`.
    pub fn new() -> Self {
        Wreath { rank: None }
    }

    /// Generators listed from smallest to largest (see [`DegLex::with_chain`]).
    pub fn with_chain(order: &[Var]) -> Self {
        Wreath {
            rank: DegLex::with_chain(order).rank,
        }
    }

    /// `x_0 > x_1 > … > x_{n-1}`: the ordering for a pc presentation whose first
    /// generator is the top of the series.
    pub fn descending(n: usize) -> Self {
        let chain: Vec<Var> = (0..n as Var).rev().collect();
        Wreath::with_chain(&chain)
    }

    #[inline]
    fn rank(&self, v: Var) -> u64 {
        match &self.rank {
            Some(r) => r.get(v as usize).copied().unwrap_or(v as u64),
            None => v as u64,
        }
    }

    /// Compare `a` and `b`, using only letters of rank `< bound`.  Every letter
    /// of `a` and `b` has rank below `bound`.
    fn cmp_bounded(&self, a: &[Var], b: &[Var], bound: u64) -> Ordering {
        if a.is_empty() || b.is_empty() {
            return a.len().cmp(&b.len());
        }
        // largest letter present in either word
        let top = a
            .iter()
            .chain(b.iter())
            .map(|&v| self.rank(v))
            .max()
            .expect("nonempty");
        debug_assert!(top < bound);
        let ca = a.iter().filter(|&&v| self.rank(v) == top).count();
        let cb = b.iter().filter(|&&v| self.rank(v) == top).count();
        if ca != cb {
            return ca.cmp(&cb);
        }
        let mut sa = a.split(|&v| self.rank(v) == top);
        let mut sb = b.split(|&v| self.rank(v) == top);
        loop {
            match (sa.next(), sb.next()) {
                (Some(x), Some(y)) => {
                    let c = self.cmp_bounded(x, y, top);
                    if c != Ordering::Equal {
                        return c;
                    }
                }
                (None, None) => return Ordering::Equal,
                // equal counts give equally many segments
                _ => unreachable!(),
            }
        }
    }
}

impl MonomialOrder for Wreath {
    fn cmp_words(&self, a: &[Var], b: &[Var]) -> Ordering {
        self.cmp_bounded(a, b, u64::MAX)
    }
    fn name(&self) -> String {
        match &self.rank {
            None => "wreath".to_string(),
            Some(r) => format!("wreath(rank={:?})", r),
        }
    }
    fn is_degree_compatible(&self) -> bool {
        false
    }
}

/// Parse a list like `[1, 2, 3]` or `1, 2, 3`.
fn parse_u64_list(s: &str) -> Option<Vec<u64>> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner.split(',').map(|t| t.trim().parse::<u64>().ok()).collect()
}

/// Chain (smallest generator first) from a rank vector (`rank[v]` = position).
fn chain_from_rank(rank: &[u64]) -> Option<Vec<Var>> {
    let mut chain = vec![Var::MAX; rank.len()];
    for (v, &r) in rank.iter().enumerate() {
        let pos = usize::try_from(r).ok()?;
        if pos >= rank.len() || chain[pos] != Var::MAX {
            return None;
        }
        chain[pos] = v as Var;
    }
    Some(chain)
}

/// Reconstruct an ordering from the string produced by [`MonomialOrder::name`]
/// (`deglex`, `deglex(rank=[…])`, `degrevlex`, `degrevlex(rank=[…])`,
/// `wdeglex([…])`, `elim(k)`, `wreath`, `wreath(rank=[…])`).  Used when loading saved bases.
pub fn parse_order(name: &str) -> Option<Box<dyn MonomialOrder>> {
    let name = name.trim();
    if name == "deglex" {
        return Some(Box::new(DegLex::new()));
    }
    if name == "degrevlex" {
        return Some(Box::new(DegRevLex::new()));
    }
    if let Some(rest) = name
        .strip_prefix("deglex(rank=")
        .and_then(|r| r.strip_suffix(')'))
    {
        let chain = chain_from_rank(&parse_u64_list(rest)?)?;
        return Some(Box::new(DegLex::with_chain(&chain)));
    }
    if let Some(rest) = name
        .strip_prefix("degrevlex(rank=")
        .and_then(|r| r.strip_suffix(')'))
    {
        let chain = chain_from_rank(&parse_u64_list(rest)?)?;
        return Some(Box::new(DegRevLex::with_chain(&chain)));
    }
    if let Some(rest) = name.strip_prefix("wdeglex(").and_then(|r| r.strip_suffix(')')) {
        let weights = parse_u64_list(rest)?;
        if weights.contains(&0) {
            return None;
        }
        return Some(Box::new(WeightedDegLex::new(weights)));
    }
    if name == "wreath" {
        return Some(Box::new(Wreath::new()));
    }
    if let Some(rest) = name
        .strip_prefix("wreath(rank=")
        .and_then(|r| r.strip_suffix(')'))
    {
        let chain = chain_from_rank(&parse_u64_list(rest)?)?;
        return Some(Box::new(Wreath::with_chain(&chain)));
    }
    if let Some(rest) = name.strip_prefix("elim(").and_then(|r| r.strip_suffix(')')) {
        return Some(Box::new(Elimination::new(rest.trim().parse().ok()?)));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_names_round_trip() {
        let orders: Vec<Box<dyn MonomialOrder>> = vec![
            Box::new(DegLex::new()),
            Box::new(DegLex::with_chain(&[2, 0, 1])),
            Box::new(DegRevLex::new()),
            Box::new(DegRevLex::with_chain(&[1, 0])),
            Box::new(WeightedDegLex::new(vec![1, 3, 2])),
            Box::new(Elimination::new(2)),
            Box::new(Wreath::new()),
            Box::new(Wreath::descending(4)),
        ];
        for o in orders {
            let back = parse_order(&o.name()).expect("parse");
            assert_eq!(back.name(), o.name());
        }
        assert!(parse_order("nonsense").is_none());
    }

    fn check_admissible(order: &dyn MonomialOrder, words: &[Vec<Var>]) {
        // total + compatible with multiplication on both sides
        for u in words {
            for v in words {
                let c = order.cmp_words(u, v);
                assert_eq!(c.reverse(), order.cmp_words(v, u));
                if u == v {
                    assert_eq!(c, Ordering::Equal);
                } else {
                    assert_ne!(c, Ordering::Equal, "{:?} vs {:?}", u, v);
                }
                for a in words {
                    for b in words {
                        let mut au = a.clone();
                        au.extend(u);
                        au.extend(b);
                        let mut av = a.clone();
                        av.extend(v);
                        av.extend(b);
                        assert_eq!(order.cmp_words(&au, &av), c);
                    }
                }
            }
        }
    }

    fn all_words(n_vars: Var, max_len: usize) -> Vec<Vec<Var>> {
        let mut out = vec![vec![]];
        let mut layer = vec![vec![]];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for w in &layer {
                for v in 0..n_vars {
                    let mut x = w.clone();
                    x.push(v);
                    next.push(x);
                }
            }
            out.extend(next.iter().cloned());
            layer = next;
        }
        out
    }

    #[test]
    fn deglex_basic() {
        let o = DegLex::new();
        assert_eq!(o.cmp_words(&[0], &[1]), Ordering::Less);
        assert_eq!(o.cmp_words(&[1, 0], &[0, 1]), Ordering::Greater);
        assert_eq!(o.cmp_words(&[0, 0, 0], &[1, 1]), Ordering::Greater);
        let o = DegLex::with_chain(&[1, 0]); // x1 < x0
        assert_eq!(o.cmp_words(&[0], &[1]), Ordering::Greater);
    }

    #[test]
    fn orders_are_admissible() {
        let words = all_words(2, 3);
        check_admissible(&DegLex::new(), &words);
        check_admissible(&DegRevLex::new(), &words);
        check_admissible(&DegLex::with_chain(&[1, 0]), &words);
        check_admissible(&WeightedDegLex::new(vec![1, 3]), &words);
        check_admissible(&Elimination::new(1), &words);
        check_admissible(&Wreath::new(), &words);
        check_admissible(&Wreath::descending(2), &words);
        let words3 = all_words(3, 3);
        check_admissible(&Wreath::new(), &words3);
        check_admissible(&Wreath::descending(3), &words3);
    }

    #[test]
    fn wreath_orients_pc_relations() {
        // generators g0 > g1 > g2 (descending): g1 g0 -> g0 g1 g2 and g0^2 -> g1 g2 g2
        let o = Wreath::descending(3);
        assert_eq!(o.cmp_words(&[1, 0], &[0, 1, 2]), Ordering::Greater);
        assert_eq!(o.cmp_words(&[0, 0], &[1, 2, 2, 2, 1]), Ordering::Greater);
        assert_eq!(o.cmp_words(&[2, 1], &[1, 2, 2, 2]), Ordering::Greater);
    }

    #[test]
    fn degrevlex_differs_from_deglex() {
        let dl = DegLex::new();
        let drl = DegRevLex::new();
        // x0 x2 x1 vs x1 x0 x2 : deglex compares from the left (x0 < x1, so
        // the first word is smaller); degrevlex compares from the right, where
        // the word with the *smaller* generator at the first difference wins
        // (x1 < x2, so the first word is larger).
        assert_eq!(dl.cmp_words(&[0, 2, 1], &[1, 0, 2]), Ordering::Less);
        assert_eq!(drl.cmp_words(&[0, 2, 1], &[1, 0, 2]), Ordering::Greater);
    }
}
