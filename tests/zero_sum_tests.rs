//! Zero-sum subsequence finders and the Davenport constant.

use ncalg::symmetry::abelian::{AbelianGroup, ZeroSum};
use rand::{Rng, SeedableRng};

fn random_seq(rng: &mut rand::rngs::StdRng, g: &AbelianGroup, q: usize) -> Vec<Vec<u64>> {
    (0..q)
        .map(|_| g.orders().iter().map(|&d| rng.gen_range(0..d)).collect())
        .collect()
}

fn sums_to_zero(g: &AbelianGroup, seq: &[Vec<u64>], idx: &[usize]) -> bool {
    let mut s = g.zero();
    for &i in idx {
        s = g.add(&s, &seq[i]);
    }
    s.iter().all(|&a| a == 0)
}

fn check(g: &AbelianGroup, seq: &[Vec<u64>], idx: &[usize]) {
    assert!(!idx.is_empty());
    assert!(idx.windows(2).all(|w| w[0] < w[1]), "increasing, no repeats");
    assert!(sums_to_zero(g, seq, idx), "{idx:?} of {seq:?}");
}

#[test]
fn linear_always_succeeds_on_d_plus_one() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    for d in [1, 2, 5, 17, 64, 70] {
        let g = AbelianGroup::elementary(2, d);
        assert_eq!(g.zero_sum_length(ZeroSum::Linear), Some(d as u64 + 1));
        for _ in 0..10 {
            let seq = random_seq(&mut rng, &g, d + 1);
            let idx = g
                .zero_sum_subset(&seq, ZeroSum::Linear)
                .unwrap()
                .expect("d + 1 vectors are dependent");
            check(&g, &seq, &idx);
        }
        // independent vectors: no zero-sum subset, reported as None
        let basis: Vec<Vec<u64>> = (0..d).map(|i| g.gen(i)).collect();
        assert_eq!(g.zero_sum_subset(&basis, ZeroSum::Linear).unwrap(), None);
    }
}

#[test]
fn leveled_succeeds_at_its_length() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    let groups = [
        vec![4],
        vec![4, 4],
        vec![4, 4, 4],
        vec![8],
        vec![8, 8],
        vec![2, 8],
        vec![2, 4, 8],
        vec![16],
        vec![1, 4],
    ];
    for d in groups {
        let g = AbelianGroup::new(d.clone());
        let q = g.zero_sum_length(ZeroSum::Leveled).unwrap() as usize;
        // Π (d_ℓ + 1)
        let top = d.iter().map(|x| x.trailing_zeros()).max().unwrap();
        let want: usize = (1..=top)
            .map(|l| 1 + d.iter().filter(|x| x.trailing_zeros() >= l).count())
            .product();
        assert_eq!(q, want, "{d:?}");
        for _ in 0..20 {
            let seq = random_seq(&mut rng, &g, q);
            let idx = g
                .zero_sum_subset(&seq, ZeroSum::Leveled)
                .unwrap()
                .unwrap_or_else(|| panic!("{d:?}: no subset in {seq:?}"));
            check(&g, &seq, &idx);
        }
        // shorter sequences: whatever is returned is a zero-sum subset
        for _ in 0..20 {
            let len = rng.gen_range(1..q);
            let seq = random_seq(&mut rng, &g, len);
            if let Some(idx) = g.zero_sum_subset(&seq, ZeroSum::Leveled).unwrap() {
                check(&g, &seq, &idx);
            }
        }
    }
}

/// All subsets, smallest first, then lexicographically.
fn brute_first(g: &AbelianGroup, seq: &[Vec<u64>]) -> Option<Vec<usize>> {
    let q = seq.len();
    let mut subsets: Vec<Vec<usize>> = (1u32..(1 << q))
        .map(|m| (0..q).filter(|&i| m >> i & 1 == 1).collect())
        .collect();
    subsets.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    subsets.into_iter().find(|s| sums_to_zero(g, seq, s))
}

#[test]
fn exhaustive_matches_brute_force() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    for d in [vec![6], vec![2, 4], vec![3, 3], vec![5], vec![2, 2, 3]] {
        let g = AbelianGroup::new(d);
        for _ in 0..40 {
            let q = rng.gen_range(1..9);
            let seq = random_seq(&mut rng, &g, q);
            let got = g.zero_sum_subset(&seq, ZeroSum::Exhaustive).unwrap();
            assert_eq!(got, brute_first(&g, &seq));
            if let Some(idx) = got {
                check(&g, &seq, &idx);
            }
        }
    }
}

/// `D(G)` by brute force: one more than the longest zero-sum-free sequence,
/// searched over multisets with their sets of subset sums.
fn brute_davenport(g: &AbelianGroup) -> u64 {
    let order: usize = g.orders().iter().product::<u64>() as usize;
    let els: Vec<Vec<u64>> = (0..order as u64).map(|i| g.from_index(i)).collect();
    let zero = g.index(&g.zero()) as usize;
    fn dfs(
        g: &AbelianGroup,
        els: &[Vec<u64>],
        zero: usize,
        start: usize,
        sums: &[bool],
        len: usize,
        best: &mut usize,
    ) {
        *best = (*best).max(len);
        for (j, x) in els.iter().enumerate().skip(start) {
            if g.index(x) as usize == zero {
                continue;
            }
            // new sums: old ∪ (old + x) ∪ {x}
            let mut next = sums.to_vec();
            next[g.index(x) as usize] = true;
            for (s, &on) in sums.iter().enumerate() {
                if on {
                    next[g.index(&g.add(&els[s], x)) as usize] = true;
                }
            }
            if !next[zero] {
                dfs(g, els, zero, j, &next, len + 1, best);
            }
        }
    }
    let mut best = 0;
    dfs(g, &els, zero, 0, &vec![false; order], 0, &mut best);
    best as u64 + 1
}

#[test]
fn davenport_constant() {
    for (d, exact) in [
        (vec![], true),
        (vec![2], true),
        (vec![4], true),
        (vec![2, 2], true),
        (vec![2, 4], true),
        (vec![2, 2, 2], true),
        (vec![3, 3], true),
        (vec![6], true),
        (vec![2, 6], true),
        (vec![2, 2, 4], true),
        (vec![3, 9], true),
    ] {
        let g = AbelianGroup::new(d.clone());
        let b = g.davenport_bound();
        assert_eq!(b.exact, exact, "{d:?}");
        assert_eq!(b.value, brute_davenport(&g), "{d:?}");
        assert_eq!(g.zero_sum_length(ZeroSum::Exhaustive), Some(b.value));
    }
    // rank 3 and not a p-group: an upper bound, at least D*(G) = 8, at most |G|
    let g = AbelianGroup::new(vec![2, 2, 6]);
    let b = g.davenport_bound();
    assert!(!b.exact);
    assert!(b.value >= 8 && b.value <= 24, "{b:?}");
}

#[test]
fn small_examples_and_errors() {
    // Z/4: D = 4, and 1, 1, 1 has no zero-sum subset
    let z4 = AbelianGroup::new(vec![4]);
    assert_eq!(z4.davenport_bound().value, 4);
    let three = vec![vec![1], vec![1], vec![1]];
    assert_eq!(z4.zero_sum_subset(&three, ZeroSum::Exhaustive).unwrap(), None);
    assert_eq!(z4.zero_sum_subset(&three, ZeroSum::Leveled).unwrap(), None);
    let four = vec![vec![1], vec![1], vec![1], vec![1]];
    assert_eq!(
        z4.zero_sum_subset(&four, ZeroSum::Leveled).unwrap(),
        Some(vec![0, 1, 2, 3])
    );
    assert_eq!(
        z4.zero_sum_subset(&four, ZeroSum::Exhaustive).unwrap(),
        Some(vec![0, 1, 2, 3])
    );
    // a zero element is a zero-sum subset on its own
    let with_zero = vec![vec![1], vec![0], vec![3]];
    assert_eq!(
        z4.zero_sum_subset(&with_zero, ZeroSum::Leveled).unwrap(),
        Some(vec![1])
    );
    // strategies that do not apply, and malformed elements
    assert!(z4.zero_sum_subset(&four, ZeroSum::Linear).is_err());
    assert!(AbelianGroup::new(vec![3])
        .zero_sum_subset(&[vec![1]], ZeroSum::Leveled)
        .is_err());
    assert!(z4.zero_sum_subset(&[vec![4]], ZeroSum::Exhaustive).is_err());
    assert_eq!(AbelianGroup::new(vec![3]).zero_sum_length(ZeroSum::Linear), None);
}
