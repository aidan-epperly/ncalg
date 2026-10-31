//! Words (noncommutative monomials) over a set of generators.
//!
//! A [`Word`] is a finite sequence of generator indices.  The empty word is
//! the multiplicative identity.  Words are compared for *equality* and hashed
//! structurally; the *monomial ordering* used by polynomials lives in
//! [`crate::order`] because it is a property of the algebra, not of the word.

use std::fmt;
use std::ops::Deref;

/// Index of a generator in a [`crate::FreeAlgebra`].
pub type Var = u32;

/// A word in the free monoid on the generators (a noncommutative monomial).
#[derive(Clone, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct Word(Vec<Var>);

impl Word {
    /// The empty word (multiplicative identity).
    #[inline]
    pub fn empty() -> Self {
        Word(Vec::new())
    }

    /// A single generator.
    #[inline]
    pub fn letter(v: Var) -> Self {
        Word(vec![v])
    }

    /// Build a word from generator indices.
    #[inline]
    pub fn from_vec(v: Vec<Var>) -> Self {
        Word(v)
    }

    /// Build a word from a slice of generator indices.
    #[inline]
    pub fn from_slice(s: &[Var]) -> Self {
        Word(s.to_vec())
    }

    /// `x^n` for a single generator.
    pub fn power(v: Var, n: usize) -> Self {
        Word(vec![v; n])
    }

    /// Generator indices.
    #[inline]
    pub fn as_slice(&self) -> &[Var] {
        &self.0
    }

    /// Consume into the underlying vector.
    #[inline]
    pub fn into_vec(self) -> Vec<Var> {
        self.0
    }

    /// Length (total degree).
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Is this the empty word?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Concatenation `self · other`.
    pub fn concat(&self, other: &Word) -> Word {
        let mut v = Vec::with_capacity(self.len() + other.len());
        v.extend_from_slice(&self.0);
        v.extend_from_slice(&other.0);
        Word(v)
    }

    /// Concatenation of three slices `a · b · c`.
    pub fn concat3(a: &[Var], b: &[Var], c: &[Var]) -> Word {
        let mut v = Vec::with_capacity(a.len() + b.len() + c.len());
        v.extend_from_slice(a);
        v.extend_from_slice(b);
        v.extend_from_slice(c);
        Word(v)
    }

    /// Append a generator in place.
    #[inline]
    pub fn push(&mut self, v: Var) {
        self.0.push(v);
    }

    /// `self^n`.
    pub fn pow(&self, n: usize) -> Word {
        let mut v = Vec::with_capacity(self.len() * n);
        for _ in 0..n {
            v.extend_from_slice(&self.0);
        }
        Word(v)
    }

    /// Number of occurrences of generator `v`.
    pub fn degree_in(&self, v: Var) -> usize {
        self.0.iter().filter(|&&x| x == v).count()
    }

    /// The reversed word (image under the canonical anti-involution).
    pub fn reversed(&self) -> Word {
        let mut v = self.0.clone();
        v.reverse();
        Word(v)
    }

    /// Sub-word `self[start..end]`.
    pub fn sub(&self, start: usize, end: usize) -> Word {
        Word(self.0[start..end].to_vec())
    }

    /// Does `pattern` occur as a contiguous subword?  Returns the first start
    /// position if so.
    pub fn find(&self, pattern: &[Var]) -> Option<usize> {
        find_subword(&self.0, pattern)
    }

    /// All start positions at which `pattern` occurs (possibly overlapping).
    pub fn find_all(&self, pattern: &[Var]) -> Vec<usize> {
        find_all_subwords(&self.0, pattern)
    }

    /// Is `self` a contiguous subword of `other`?
    pub fn is_subword_of(&self, other: &[Var]) -> bool {
        find_subword(other, &self.0).is_some()
    }

    /// Proper overlaps: lengths `c` with `0 < c < min(|self|,|other|)` such
    /// that the suffix of `self` of length `c` equals the prefix of `other`
    /// of length `c`.
    pub fn overlaps_with(&self, other: &[Var]) -> Vec<usize> {
        proper_overlaps(&self.0, other)
    }

    /// Maximum generator index occurring, if any.
    pub fn max_var(&self) -> Option<Var> {
        self.0.iter().copied().max()
    }
}

impl Deref for Word {
    type Target = [Var];
    fn deref(&self) -> &[Var] {
        &self.0
    }
}

impl From<Vec<Var>> for Word {
    fn from(v: Vec<Var>) -> Self {
        Word(v)
    }
}

impl From<&[Var]> for Word {
    fn from(v: &[Var]) -> Self {
        Word(v.to_vec())
    }
}

impl fmt::Debug for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "1");
        }
        let mut first = true;
        for (v, n) in run_lengths(&self.0) {
            if !first {
                write!(f, "*")?;
            }
            first = false;
            if n == 1 {
                write!(f, "x{}", v)?;
            } else {
                write!(f, "x{}^{}", v, n)?;
            }
        }
        Ok(())
    }
}

/// Run-length encode a word: `[(generator, multiplicity)]`.
pub fn run_lengths(w: &[Var]) -> Vec<(Var, usize)> {
    let mut out: Vec<(Var, usize)> = Vec::new();
    for &v in w {
        match out.last_mut() {
            Some((last, n)) if *last == v => *n += 1,
            _ => out.push((v, 1)),
        }
    }
    out
}

/// First start position of `pattern` in `text`, naive search (fast enough for
/// the short words that occur as monomials; bulk searches use the
/// Aho–Corasick automaton in [`crate::automaton`]).
pub fn find_subword(text: &[Var], pattern: &[Var]) -> Option<usize> {
    if pattern.is_empty() {
        return Some(0);
    }
    if pattern.len() > text.len() {
        return None;
    }
    text.windows(pattern.len()).position(|w| w == pattern)
}

/// All start positions of `pattern` in `text`.
pub fn find_all_subwords(text: &[Var], pattern: &[Var]) -> Vec<usize> {
    if pattern.is_empty() || pattern.len() > text.len() {
        return Vec::new();
    }
    text.windows(pattern.len())
        .enumerate()
        .filter(|(_, w)| *w == pattern)
        .map(|(i, _)| i)
        .collect()
}

/// Proper overlaps between `u` and `v`: all `c` with `0 < c < min(|u|,|v|)`
/// such that `u[|u|-c..] == v[..c]`.
pub fn proper_overlaps(u: &[Var], v: &[Var]) -> Vec<usize> {
    let max = u.len().min(v.len());
    let mut out = Vec::new();
    for c in 1..max {
        if u[u.len() - c..] == v[..c] {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlaps() {
        // xyx and xyx: suffix "x" == prefix "x"; suffix "yx" != prefix "xy"
        let u = Word::from_vec(vec![0, 1, 0]);
        assert_eq!(u.overlaps_with(&u), vec![1]);
        // aba, abab: suffix "a" == prefix "a"; suffix "ba" != "ab"
        let v = Word::from_vec(vec![0, 1, 0, 1]);
        assert_eq!(u.overlaps_with(&v), vec![1]);
        // abab, aba: c=1 "b" vs "a" no; c=2 "ab" == "ab" yes
        assert_eq!(v.overlaps_with(&u), vec![2]);
    }

    #[test]
    fn subwords() {
        let w = Word::from_vec(vec![0, 1, 1, 0, 1]);
        assert_eq!(w.find(&[1, 1]), Some(1));
        assert_eq!(w.find(&[1, 0, 1]), Some(2));
        assert_eq!(w.find(&[0, 0]), None);
        assert_eq!(w.find_all(&[1]), vec![1, 2, 4]);
        assert_eq!(w.find(&[]), Some(0));
    }

    #[test]
    fn debug_format() {
        let w = Word::from_vec(vec![0, 0, 1, 0]);
        assert_eq!(format!("{:?}", w), "x0^2*x1*x0");
        assert_eq!(format!("{:?}", Word::empty()), "1");
    }
}
