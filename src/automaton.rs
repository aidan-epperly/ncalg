//! Aho–Corasick automaton over words.
//!
//! Reduction modulo a set of polynomials repeatedly asks "does any leading
//! word occur as a subword of this word, and where?".  With a dictionary of
//! `k` leading words of total length `L`, the automaton answers that in
//! `O(|word|)` time after `O(L · n_vars)` construction, independent of `k`.
//! It also drives the enumeration of normal words (words avoiding every
//! pattern) used for Hilbert series.

use crate::word::Var;

const NONE: u32 = u32::MAX;

/// Multi-pattern subword matcher over words on `n_vars` generators.
#[derive(Clone, Debug)]
pub struct AhoCorasick {
    n_vars: usize,
    /// Dense transition table: `goto[node * n_vars + v]`.
    goto: Vec<u32>,
    /// Pattern ending exactly at this node, if any.
    pattern_at: Vec<u32>,
    /// Nearest proper suffix node that ends a pattern (dictionary link).
    out_link: Vec<u32>,
    /// Some pattern ends at this node or at a node on its dictionary chain:
    /// the index of the *shortest* such pattern (so `terminal[s] != NONE` iff
    /// the word read so far has a pattern as suffix).
    terminal: Vec<u32>,
    pattern_lens: Vec<usize>,
    has_empty_pattern: bool,
}

impl AhoCorasick {
    /// Build the automaton for the given patterns.  Patterns may repeat or be
    /// empty; an empty pattern matches everywhere (`has_empty_pattern`).
    pub fn new<P: AsRef<[Var]>>(n_vars: usize, patterns: &[P]) -> Self {
        let n_vars = n_vars.max(1);
        let mut goto: Vec<u32> = vec![NONE; n_vars];
        let mut pattern_at: Vec<u32> = vec![NONE];
        let mut depth: Vec<u32> = vec![0];
        let mut pattern_lens = Vec::with_capacity(patterns.len());
        let mut has_empty_pattern = false;

        // trie
        for (idx, p) in patterns.iter().enumerate() {
            let p = p.as_ref();
            pattern_lens.push(p.len());
            if p.is_empty() {
                has_empty_pattern = true;
                if pattern_at[0] == NONE {
                    pattern_at[0] = idx as u32;
                }
                continue;
            }
            let mut node = 0usize;
            for &v in p {
                let v = v as usize;
                assert!(v < n_vars, "generator index out of range in automaton pattern");
                let slot = node * n_vars + v;
                if goto[slot] == NONE {
                    let new = pattern_at.len() as u32;
                    goto[slot] = new;
                    goto.extend(std::iter::repeat_n(NONE, n_vars));
                    pattern_at.push(NONE);
                    depth.push(depth[node] + 1);
                }
                node = goto[slot] as usize;
            }
            // keep the first pattern index for duplicates
            if pattern_at[node] == NONE {
                pattern_at[node] = idx as u32;
            }
        }

        let n_nodes = pattern_at.len();
        let mut fail: Vec<u32> = vec![0; n_nodes];
        let mut out_link: Vec<u32> = vec![NONE; n_nodes];
        let mut terminal: Vec<u32> = vec![NONE; n_nodes];
        terminal[0] = pattern_at[0];

        // BFS to fill failure links and complete the transition table
        let mut queue = std::collections::VecDeque::new();
        for v in 0..n_vars {
            let t = goto[v];
            if t == NONE {
                goto[v] = 0;
            } else {
                fail[t as usize] = 0;
                queue.push_back(t as usize);
            }
        }
        while let Some(node) = queue.pop_front() {
            let f = fail[node] as usize;
            // dictionary link: nearest suffix node that is a pattern end
            out_link[node] = if pattern_at[f] != NONE {
                f as u32
            } else {
                out_link[f]
            };
            terminal[node] = if pattern_at[node] != NONE {
                // shortest pattern ending here: prefer the one via the chain
                // only if it is shorter — all patterns on the chain are
                // proper suffixes, hence shorter, so take the chain's if any.
                if terminal[f] != NONE {
                    terminal[f]
                } else {
                    pattern_at[node]
                }
            } else {
                terminal[f]
            };
            for v in 0..n_vars {
                let slot = node * n_vars + v;
                let t = goto[slot];
                if t == NONE {
                    goto[slot] = goto[f * n_vars + v];
                } else {
                    fail[t as usize] = goto[f * n_vars + v];
                    queue.push_back(t as usize);
                }
            }
        }

        AhoCorasick {
            n_vars,
            goto,
            pattern_at,
            out_link,
            terminal,
            pattern_lens,
            has_empty_pattern,
        }
    }

    /// Number of patterns.
    pub fn num_patterns(&self) -> usize {
        self.pattern_lens.len()
    }

    /// Does the empty word occur as a pattern (so everything matches)?
    pub fn has_empty_pattern(&self) -> bool {
        self.has_empty_pattern
    }

    /// Number of automaton states.
    pub fn num_states(&self) -> usize {
        self.pattern_at.len()
    }

    /// The start state.
    #[inline]
    pub fn root(&self) -> u32 {
        0
    }

    /// Transition on generator `v`.
    #[inline]
    pub fn step(&self, state: u32, v: Var) -> u32 {
        self.goto[state as usize * self.n_vars + v as usize]
    }

    /// Does the word read so far (ending in `state`) have a pattern as a suffix?
    #[inline]
    pub fn is_terminal(&self, state: u32) -> bool {
        self.terminal[state as usize] != NONE
    }

    /// The shortest pattern that is a suffix of the word read so far.
    #[inline]
    pub fn terminal_pattern(&self, state: u32) -> Option<usize> {
        let t = self.terminal[state as usize];
        (t != NONE).then_some(t as usize)
    }

    /// Length of pattern `idx`.
    #[inline]
    pub fn pattern_len(&self, idx: usize) -> usize {
        self.pattern_lens[idx]
    }

    /// Find the occurrence with the smallest end position of any pattern in
    /// `word`.  Returns `(pattern index, start position)`.
    pub fn find_first(&self, word: &[Var]) -> Option<(usize, usize)> {
        if self.has_empty_pattern {
            return Some((self.pattern_at[0] as usize, 0));
        }
        let mut s = 0u32;
        for (end, &v) in word.iter().enumerate() {
            s = self.step(s, v);
            let t = self.terminal[s as usize];
            if t != NONE {
                let len = self.pattern_lens[t as usize];
                return Some((t as usize, end + 1 - len));
            }
        }
        None
    }

    /// [`find_first`](Self::find_first) restricted to the patterns accepted
    /// by `alive`: the occurrence with the smallest end position, and at that
    /// position the shortest pattern (the same rule as `find_first`).
    pub fn find_first_where(&self, word: &[Var], alive: impl Fn(usize) -> bool) -> Option<(usize, usize)> {
        if self.has_empty_pattern && alive(self.pattern_at[0] as usize) {
            return Some((self.pattern_at[0] as usize, 0));
        }
        let mut s = 0u32;
        for (end, &v) in word.iter().enumerate() {
            s = self.step(s, v);
            if self.terminal[s as usize] == NONE {
                continue;
            }
            // patterns ending here: at s itself, then along the dictionary links
            let mut best: Option<usize> = None;
            let mut q = if self.pattern_at[s as usize] != NONE {
                s
            } else {
                self.out_link[s as usize]
            };
            while q != NONE {
                let p = self.pattern_at[q as usize];
                if p != NONE
                    && alive(p as usize)
                    && best.is_none_or(|b| self.pattern_lens[p as usize] <= self.pattern_lens[b])
                {
                    best = Some(p as usize);
                }
                q = self.out_link[q as usize];
            }
            if let Some(p) = best {
                return Some((p, end + 1 - self.pattern_lens[p]));
            }
        }
        None
    }

    /// Does any pattern occur in `word`?
    pub fn contains_match(&self, word: &[Var]) -> bool {
        self.find_first(word).is_some()
    }

    /// Is `word` free of all patterns?
    pub fn is_normal(&self, word: &[Var]) -> bool {
        self.find_first(word).is_none()
    }

    /// All occurrences `(pattern index, start position)` of all patterns in
    /// `word`, in order of end position.  Duplicate patterns are reported
    /// under their first index.
    pub fn find_all(&self, word: &[Var]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if self.has_empty_pattern {
            for pos in 0..=word.len() {
                out.push((self.pattern_at[0] as usize, pos));
            }
        }
        let mut s = 0u32;
        for (end, &v) in word.iter().enumerate() {
            s = self.step(s, v);
            let mut n = s;
            while n != NONE {
                let p = self.pattern_at[n as usize];
                if p != NONE && self.pattern_lens[p as usize] > 0 {
                    out.push((p as usize, end + 1 - self.pattern_lens[p as usize]));
                }
                n = self.out_link[n as usize];
            }
        }
        out
    }

    /// Words of length exactly `d` over `n_vars` generators that avoid every
    /// pattern (normal words), by depth-first search over non-terminal
    /// states, in lexicographic order of generator indices.  The one
    /// normal-word enumeration in ncalg.
    pub fn avoiding_words(&self, n_vars: usize, d: usize) -> Vec<Vec<Var>> {
        if self.has_empty_pattern() {
            return Vec::new();
        }
        let n = n_vars as Var;
        let mut out = Vec::new();
        let mut stack: Vec<(u32, Vec<Var>)> = vec![(self.root(), Vec::new())];
        while let Some((state, word)) = stack.pop() {
            if word.len() == d {
                out.push(word);
                continue;
            }
            // push in reverse so smaller generators are explored first
            for v in (0..n).rev() {
                let s = self.step(state, v);
                if !self.is_terminal(s) {
                    let mut w = word.clone();
                    w.push(v);
                    stack.push((s, w));
                }
            }
        }
        out
    }

    /// One layer of the count of pattern-avoiding words: from the number of
    /// avoiding words of length `k` ending in each state to those of length
    /// `k + 1`.  Shared by the Hilbert series and the dimension count;
    /// panics instead of wrapping when a count exceeds `u128`.
    pub fn next_layer(&self, n_vars: usize, counts: &[u128]) -> Vec<u128> {
        let mut next = vec![0u128; counts.len()];
        for (s, &c) in counts.iter().enumerate() {
            if c == 0 {
                continue;
            }
            for v in 0..n_vars as Var {
                let t = self.step(s as u32, v);
                if !self.is_terminal(t) {
                    next[t as usize] = next[t as usize]
                        .checked_add(c)
                        .expect("normal-word count overflows u128");
                }
            }
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_patterns() {
        let pats: Vec<Vec<Var>> = vec![vec![0, 1], vec![1, 1, 0], vec![2]];
        let ac = AhoCorasick::new(3, &pats);
        assert_eq!(ac.find_first(&[1, 1, 0, 1]), Some((1, 0)));
        assert_eq!(ac.find_first(&[1, 0, 1, 1]), Some((0, 1)));
        assert_eq!(ac.find_first(&[1, 0, 0, 1, 1]), Some((0, 2)));
        assert_eq!(ac.find_first(&[1, 0, 0]), None);
        assert!(ac.is_normal(
            &[0, 0, 1, 0]
                .iter()
                .copied()
                .filter(|&v| v == 0)
                .collect::<Vec<_>>()
        ));
        let all = ac.find_all(&[1, 1, 0, 1, 2]);
        assert_eq!(all, vec![(1, 0), (0, 2), (2, 4)]);
    }

    #[test]
    fn suffix_patterns_are_found_via_fail_links() {
        // pattern "b" is a suffix of "ab"; reading "ab" must still detect "b"
        let pats: Vec<Vec<Var>> = vec![vec![0, 1, 1], vec![1]];
        let ac = AhoCorasick::new(2, &pats);
        assert_eq!(ac.find_first(&[0, 1, 1]), Some((1, 1)));
        assert_eq!(ac.find_all(&[0, 1, 1]), vec![(1, 1), (0, 0), (1, 2)]);
    }

    #[test]
    fn empty_pattern() {
        let pats: Vec<Vec<Var>> = vec![vec![]];
        let ac = AhoCorasick::new(2, &pats);
        assert!(ac.has_empty_pattern());
        assert_eq!(ac.find_first(&[0, 1]), Some((0, 0)));
        assert_eq!(ac.find_first(&[]), Some((0, 0)));
    }

    #[test]
    fn no_patterns() {
        let pats: Vec<Vec<Var>> = vec![];
        let ac = AhoCorasick::new(2, &pats);
        assert_eq!(ac.find_first(&[0, 1, 0]), None);
    }
}

/// A set of patterns under insertion and retirement with the queries of
/// [`AhoCorasick::find_first`], by the Bentley–Saxe logarithmic method:
/// static automata on groups of `≤ 2^i` patterns, merged like binary
/// carries on insertion; retired patterns are tombstones skipped by the
/// queries, and a group is rebuilt once half of it is retired.  An
/// insertion costs `O(L log N · n_vars)` amortized instead of rebuilding one
/// automaton on everything, a query `O(|w| log N)`.  For a pattern set that is
/// pairwise subword-free (the leading words of an active Gröbner set) at most
/// one live pattern ends at the earliest match position, so the result is
/// exactly that of a single automaton on the live patterns.
#[derive(Clone, Debug)]
pub struct DynamicMatcher {
    n_vars: usize,
    words: Vec<Vec<Var>>,
    alive: Vec<bool>,
    level_of: Vec<usize>,
    levels: Vec<Option<MatcherLevel>>,
}

#[derive(Clone, Debug)]
struct MatcherLevel {
    ac: AhoCorasick,
    /// pattern index in `ac` -> id
    ids: Vec<usize>,
    dead: usize,
}

impl DynamicMatcher {
    /// An empty matcher over the alphabet `0..n_vars`.
    pub fn new(n_vars: usize) -> Self {
        DynamicMatcher {
            n_vars,
            words: Vec::new(),
            alive: Vec::new(),
            level_of: Vec::new(),
            levels: Vec::new(),
        }
    }

    fn build(&mut self, i: usize, ids: Vec<usize>) {
        if ids.is_empty() {
            self.levels[i] = None;
            return;
        }
        let pats: Vec<&[Var]> = ids.iter().map(|&id| self.words[id].as_slice()).collect();
        let ac = AhoCorasick::new(self.n_vars, &pats);
        for &id in &ids {
            self.level_of[id] = i;
        }
        self.levels[i] = Some(MatcherLevel { ac, ids, dead: 0 });
    }

    /// Add a pattern; returns its id (ids are consecutive from 0).
    pub fn insert(&mut self, word: &[Var]) -> usize {
        let id = self.words.len();
        self.words.push(word.to_vec());
        self.alive.push(true);
        self.level_of.push(0);
        let mut pool = vec![id];
        let mut i = 0;
        loop {
            if i == self.levels.len() {
                self.levels.push(None);
            }
            match self.levels[i].take() {
                None => {
                    if pool.len() <= 1 << i {
                        self.build(i, pool);
                        return id;
                    }
                    i += 1;
                }
                Some(level) => {
                    pool.extend(level.ids.into_iter().filter(|&x| self.alive[x]));
                    i += 1;
                }
            }
        }
    }

    /// Retire pattern `id` (it no longer matches).
    pub fn retire(&mut self, id: usize) {
        if !self.alive[id] {
            return;
        }
        self.alive[id] = false;
        let i = self.level_of[id];
        let rebuild = match &mut self.levels[i] {
            Some(level) => {
                level.dead += 1;
                2 * level.dead > level.ids.len()
            }
            None => false,
        };
        if rebuild {
            let ids: Vec<usize> = self.levels[i]
                .take()
                .map(|l| l.ids.into_iter().filter(|&x| self.alive[x]).collect())
                .unwrap_or_default();
            self.build(i, ids);
        }
    }

    /// Is pattern `id` still matched (not retired)?
    pub fn is_alive(&self, id: usize) -> bool {
        self.alive[id]
    }

    /// Every occurrence `(id, start)` of a live pattern in `word` (no
    /// particular order).
    pub fn find_all(&self, word: &[Var]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for level in self.levels.iter().flatten() {
            for (p, start) in level.ac.find_all(word) {
                let id = level.ids[p];
                if self.alive[id] {
                    out.push((id, start));
                }
            }
        }
        out
    }

    /// Length of pattern `id`.
    pub fn pattern_len(&self, id: usize) -> usize {
        self.words[id].len()
    }

    /// The earliest-ending occurrence of a live pattern in `word` (the
    /// shortest at that end): `(id, start)`.
    pub fn find_first(&self, word: &[Var]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, usize)> = None; // (end, len, id)
        for level in self.levels.iter().flatten() {
            if let Some((p, start)) = level.ac.find_first_where(word, |p| self.alive[level.ids[p]]) {
                let len = level.ac.pattern_len(p);
                let cand = (start + len, len, level.ids[p]);
                if best.is_none_or(|b| (cand.0, cand.1) < (b.0, b.1)) {
                    best = Some(cand);
                }
            }
        }
        best.map(|(end, len, id)| (id, end - len))
    }
}
