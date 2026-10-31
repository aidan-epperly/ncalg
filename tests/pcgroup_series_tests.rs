//! Lower and upper central series, nilpotency class, factor invariants and
//! centralizers modulo a normal subgroup for pc groups.

use ncalg::symmetry::pcgroup::{families, Exps, PcBuilder, PcPresentation, Subgroup};
use std::collections::BTreeSet;

/// Dihedral group of order `2^{k+1}` (symmetries of a regular `2^k`-gon):
/// pcgs `s, r_1 = r, r_2 = r^2, …, r_k = r^{2^{k-1}}`.
fn dihedral(k: usize) -> PcPresentation {
    let names: Vec<String> = std::iter::once("s".to_string())
        .chain((1..=k).map(|i| format!("r{i}")))
        .collect();
    let mut b = PcBuilder::new(2, names);
    for i in 1..=k {
        if i < k {
            b = b.power(i, &[(i + 1, 1)]);
        }
        // s^-1 r_i s = r_i^-1 = r_i r_{i+1} ⋯ r_k
        let tail: Vec<(usize, u8)> = ((i + 1)..=k).map(|l| (l, 1)).collect();
        b = b.conj(0, i, &tail);
    }
    let g = b.build().unwrap();
    g.check_consistency().unwrap();
    g
}

/// Extraspecial group `2^{1+2m}` of real type: `[x_i, y_i] = z`, all squares trivial.
fn extraspecial(m: usize) -> PcPresentation {
    let names: Vec<String> = (0..m)
        .map(|i| format!("x{i}"))
        .chain((0..m).map(|i| format!("y{i}")))
        .chain(std::iter::once("z".to_string()))
        .collect();
    let mut b = PcBuilder::new(2, names);
    for i in 0..m {
        b = b.conj(i, m + i, &[(2 * m, 1)]);
    }
    let g = b.build().unwrap();
    g.check_consistency().unwrap();
    g
}

fn elements(g: &PcPresentation) -> Vec<Exps> {
    (0..(g.p() as u64).pow(g.len() as u32))
        .map(|i| g.from_index(i))
        .collect()
}

fn closure(g: &PcPresentation, gens: &[Exps]) -> BTreeSet<Exps> {
    let mut s: BTreeSet<Exps> = [g.identity()].into_iter().collect();
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

fn set(g: &PcPresentation, s: &Subgroup) -> BTreeSet<Exps> {
    closure(g, &s.gens().cloned().collect::<Vec<_>>())
}

/// The series by brute force over all elements.
fn brute_series(g: &PcPresentation) -> (Vec<BTreeSet<Exps>>, Vec<BTreeSet<Exps>>) {
    let els = elements(g);
    let all: BTreeSet<Exps> = els.iter().cloned().collect();
    let mut lower = vec![all.clone()];
    while lower.last().unwrap().len() > 1 {
        let last = lower.last().unwrap();
        let comms: Vec<Exps> = els
            .iter()
            .flat_map(|x| last.iter().map(move |y| g.comm(x, y)))
            .collect();
        lower.push(closure(g, &comms));
    }
    let mut upper = vec![[g.identity()].into_iter().collect::<BTreeSet<Exps>>()];
    while upper.last().unwrap().len() < all.len() {
        let last = upper.last().unwrap();
        let next: BTreeSet<Exps> = els
            .iter()
            .filter(|x| els.iter().all(|y| last.contains(&g.comm(x, y))))
            .cloned()
            .collect();
        upper.push(next);
    }
    (lower, upper)
}

fn check_against_brute_force(g: &PcPresentation) {
    let (lower, upper) = brute_series(g);
    let lc = g.lower_central_series();
    let uc = g.upper_central_series();
    assert_eq!(lc.len(), lower.len());
    assert_eq!(uc.len(), upper.len());
    for (s, b) in lc.iter().zip(&lower) {
        assert_eq!(&set(g, s), b);
    }
    for (s, b) in uc.iter().zip(&upper) {
        assert_eq!(&set(g, s), b);
    }
    assert_eq!(g.nilpotency_class(), lc.len() - 1);
    assert_eq!(uc.len(), lc.len(), "upper and lower series have the same length");
    // factor orders match the subgroup orders
    for f in [&lc, &uc] {
        let inv = g.series_factor_invariants(f).unwrap();
        for (w, i) in f.windows(2).zip(&inv) {
            let diff = w[0].log_order().abs_diff(w[1].log_order()) as u32;
            assert_eq!(i.iter().product::<u64>(), (g.p() as u64).pow(diff));
        }
    }
    // the centralizer modulo the normal subgroups of the series, by brute force
    let els = elements(g);
    for m in lc.iter().chain(&uc) {
        let ms = set(g, m);
        for xs in [vec![g.gen(0)], (0..g.len()).map(|i| g.gen(i)).collect::<Vec<_>>()] {
            let want: BTreeSet<Exps> = els
                .iter()
                .filter(|y| xs.iter().all(|x| ms.contains(&g.comm(y, x))))
                .cloned()
                .collect();
            assert_eq!(set(g, &g.centralizer_mod(&xs, m)), want);
        }
    }
}

#[test]
fn dihedral_series() {
    for k in 2..=6 {
        let g = dihedral(k);
        assert_eq!(g.nilpotency_class(), k, "D of order 2^{}", k + 1);
        let lc = g.lower_central_series();
        let mut want = vec![vec![2, 2]];
        want.extend(std::iter::repeat_n(vec![2], k - 1));
        assert_eq!(g.series_factor_invariants(&lc).unwrap(), want);
        // Z(G) = <r^{2^{k-1}}>, of order 2
        let uc = g.upper_central_series();
        assert_eq!(uc[1].log_order(), 1);
        assert_eq!(uc[1], g.center());
        // every factor has exponent 2, so the lower exponent-2 central series agrees
        assert_eq!(g.lower_exponent_p_central_series().len(), lc.len());
        if k <= 4 {
            check_against_brute_force(&g);
        }
    }
}

#[test]
fn extraspecial_class_two() {
    for m in 1..=4 {
        let g = extraspecial(m);
        assert_eq!(g.nilpotency_class(), 2);
        let lc = g.lower_central_series();
        assert_eq!(lc[1].log_order(), 1);
        assert_eq!(lc[1], g.center());
        assert_eq!(
            g.series_factor_invariants(&lc).unwrap(),
            vec![vec![2; 2 * m], vec![2]]
        );
        assert_eq!(g.lower_exponent_p_central_series().len(), lc.len());
        if m <= 2 {
            check_against_brute_force(&g);
        }
    }
}

#[test]
fn cyclic_and_abelian_groups() {
    // Z/8 = <a | a^2 = b, b^2 = c>: class 1, invariants [8]; the lower
    // exponent-2 central series is longer, since G/γ_2 has exponent 8
    let z8 = PcBuilder::new(2, ["a", "b", "c"])
        .power(0, &[(1, 1)])
        .power(1, &[(2, 1)])
        .build()
        .unwrap();
    assert_eq!(z8.nilpotency_class(), 1);
    let lc = z8.lower_central_series();
    assert_eq!(z8.series_factor_invariants(&lc).unwrap(), vec![vec![8]]);
    assert_eq!(z8.lower_exponent_p_central_series().len(), 4);
    assert_eq!(z8.upper_central_series().len(), 2);
    // Heisenberg group mod 3: class 2
    let h3 = PcBuilder::new(3, ["x", "y", "z"])
        .conj(0, 1, &[(2, 1)])
        .build()
        .unwrap();
    assert_eq!(h3.nilpotency_class(), 2);
    check_against_brute_force(&h3);
    // the trivial group
    let one = PcBuilder::new(2, Vec::<String>::new()).build().unwrap();
    assert_eq!(one.nilpotency_class(), 0);
    assert_eq!(one.upper_central_series().len(), 1);
    // an error for a non-abelian factor
    let d = dihedral(2);
    assert!(d
        .abelian_invariants(&d.whole_group(), &d.trivial_subgroup())
        .is_err());
}

#[test]
fn families_brute_force() {
    for (name, m) in [("PauliCZ", 2), ("XS", 2), ("XT", 1), ("D4N", 2)] {
        let f = families::by_name(name, m).unwrap();
        if f.pres.len() <= 10 {
            check_against_brute_force(&f.pres);
        }
    }
}

/// Nilpotency classes of a few small `x_diag` groups ⟨X_j, ζ_8^{c x_S}⟩.
#[test]
fn x_diag_classes() {
    let bit = |x: usize, j: usize| ((x >> j) & 1) as u8;
    let mono = |m: usize, c: u8, s: &[usize]| -> Vec<u8> {
        (0..1usize << m)
            .map(|x| {
                if s.iter().all(|&j| bit(x, j) == 1) {
                    c % 8
                } else {
                    0
                }
            })
            .collect()
    };
    let shifts = |m: usize| -> Vec<u64> { (0..m).map(|j| 1u64 << j).collect() };
    // (name, qubits, diagonal generators (c, S) for ζ_8^{c x_S})
    type Case = (&'static str, usize, Vec<(u8, Vec<usize>)>);
    let cases: Vec<Case> = vec![
        ("XS", 2, vec![(2, vec![0]), (2, vec![1])]),
        ("XS+CZ", 2, vec![(2, vec![0]), (2, vec![1]), (4, vec![0, 1])]),
        ("XS+CS", 2, vec![(2, vec![0]), (2, vec![1]), (2, vec![0, 1])]),
        ("XT", 2, vec![(1, vec![0]), (1, vec![1])]),
        (
            "XT+CCZ",
            3,
            vec![(1, vec![0]), (1, vec![1]), (1, vec![2]), (4, vec![0, 1, 2])],
        ),
        ("Pauli+CZ", 2, vec![(4, vec![0]), (4, vec![1]), (4, vec![0, 1])]),
        ("X+CCZ", 3, vec![(4, vec![0, 1, 2])]),
    ];
    let expect = [3, 3, 4, 4, 4, 3, 4];
    for ((name, m, gens), want) in cases.into_iter().zip(expect) {
        let diag: Vec<Vec<u8>> = gens.iter().map(|(c, s)| mono(m, *c, s)).collect();
        let f = families::x_diag(name, m, &shifts(m), &diag).unwrap();
        assert_eq!(f.pres.nilpotency_class(), want, "{name}");
        assert_eq!(f.pres.upper_central_series().len(), want + 1, "{name}");
    }
}
