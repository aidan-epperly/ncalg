//! Quotients by normal subgroups, direct and central products of pc groups.

use ncalg::symmetry::pcgroup::{families, Exps, PcBuilder, PcPresentation, Subgroup};
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

fn d4() -> PcPresentation {
    PcBuilder::new(2, ["s", "r", "c"])
        .power(1, &[(2, 1)])
        .conj(0, 1, &[(2, 1)])
        .build()
        .unwrap()
}

/// Dihedral of order 2^{k+1}.
fn dihedral(k: usize) -> PcPresentation {
    let names: Vec<String> = std::iter::once("s".to_string())
        .chain((1..=k).map(|i| format!("r{i}")))
        .collect();
    let mut b = PcBuilder::new(2, names);
    for i in 1..=k {
        if i < k {
            b = b.power(i, &[(i + 1, 1)]);
        }
        let tail: Vec<(usize, u8)> = ((i + 1)..=k).map(|l| (l, 1)).collect();
        b = b.conj(0, i, &tail);
    }
    b.build().unwrap()
}

fn elements(g: &PcPresentation) -> Vec<Exps> {
    (0..(g.p() as u64).pow(g.len() as u32))
        .map(|i| g.from_index(i))
        .collect()
}

fn set(g: &PcPresentation, s: &Subgroup) -> BTreeSet<Exps> {
    let gens: Vec<Exps> = s.gens().cloned().collect();
    let mut out: BTreeSet<Exps> = [g.identity()].into_iter().collect();
    loop {
        let new: Vec<Exps> = out
            .iter()
            .flat_map(|x| gens.iter().map(move |y| g.mul(x, y)))
            .filter(|z| !out.contains(z))
            .collect();
        if new.is_empty() {
            return out;
        }
        out.extend(new);
    }
}

/// `G/N` against the definitions: consistency, order, kernel, homomorphism,
/// surjectivity, and `lift` a section.
fn check_quotient(g: &PcPresentation, n: &Subgroup) {
    let (q, proj) = g.quotient(n).unwrap();
    q.check_consistency().unwrap();
    assert_eq!(q.log_order(), g.log_order() - n.log_order());
    let els = elements(g);
    let ns = set(g, n);
    let mut images = BTreeSet::new();
    for x in &els {
        let px = proj.apply(x);
        assert_eq!(px == q.identity(), ns.contains(x), "kernel");
        images.insert(px.clone());
        assert_eq!(proj.apply(&proj.lift(&px)), px, "lift is a section");
    }
    assert_eq!(
        images.len() as u64,
        (q.p() as u64).pow(q.len() as u32),
        "surjective"
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(els.len() as u64);
    for _ in 0..200 {
        let a = &els[rng.gen_range(0..els.len())];
        let b = &els[rng.gen_range(0..els.len())];
        assert_eq!(
            proj.apply(&g.mul(a, b)),
            q.mul(&proj.apply(a), &proj.apply(b)),
            "homomorphism"
        );
    }
}

#[test]
fn quotients_by_normal_subgroups() {
    // D_16 by <r^2> (normal, not central): the Klein four group; by <r>: C_2
    let d16 = dihedral(3);
    d16.check_consistency().unwrap();
    for gens in [
        vec![d16.gen(2)],
        vec![d16.gen(1)],
        vec![d16.gen(3)],
        vec![d16.gen(0)],
    ] {
        let n = d16.normal_closure(&gens);
        check_quotient(&d16, &n);
    }
    // the upper central series terms Z_i (normal, Z_2 not central)
    for z in d16.upper_central_series() {
        check_quotient(&d16, &z);
    }
    // random normal closures in small Clifford-hierarchy groups
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    for (name, m) in [("XS", 1), ("PauliCZ", 2), ("XT", 1), ("D4N", 1)] {
        let f = families::by_name(name, m).unwrap();
        let g = &f.pres;
        if g.len() > 9 {
            continue;
        }
        let els = elements(g);
        for _ in 0..6 {
            let x = els[rng.gen_range(0..els.len())].clone();
            check_quotient(g, &g.normal_closure(&[x]));
        }
    }
    // not normal: an error
    let d = d4();
    assert!(d.quotient(&d.subgroup(&[d.gen(0)])).is_err());
}

#[test]
fn quotient_by_a_diagonal_central_subgroup() {
    // D4 x D4 modulo the diagonal of the centers, which is not spanned by
    // pcgs generators (quotient_central cannot do it)
    let d = d4().direct_product(&d4()).unwrap();
    d.check_consistency().unwrap();
    assert_eq!(d.log_order(), 6);
    let c = |left: bool| -> Exps {
        let mut e = vec![0u8; 6];
        e[if left { 2 } else { 5 }] = 1;
        e
    };
    let diag = d.mul(&c(true), &c(false));
    let n = d.subgroup(&[diag]);
    assert!(n.depths() != vec![5] && n.log_order() == 1);
    check_quotient(&d, &n);
    let (q, _) = d.quotient(&n).unwrap();
    assert_eq!(q.log_order(), 5);
    // D4 ∘ D4 = extraspecial 2^{1+4} of + type: class 2, center of order 2
    assert_eq!(q.nilpotency_class(), 2);
    assert_eq!(q.center().log_order(), 1);
}

#[test]
fn quotient_central_is_the_general_quotient() {
    // the projective quotients of the families divide out scalars that do not
    // sit at the bottom of the pcgs; for central N the general quotient must
    // agree with deleting N's exponents from G's relations (the algorithm
    // quotient_central used before it called quotient)
    for (name, m) in [("XS", 2), ("XT", 2), ("PauliCZ", 2), ("PauliCCZ", 3)] {
        let f = families::by_name(name, m).unwrap();
        let g = &f.pres;
        let keep: Vec<usize> = (0..g.len()).filter(|k| !f.scalars.contains(k)).collect();
        let delete = |e: &Exps| -> Exps { keep.iter().map(|&k| e[k]).collect() };
        let scalars: Vec<Exps> = f.scalars.iter().map(|&k| g.gen(k)).collect();
        let (q, p) = g.quotient(&g.subgroup(&scalars)).unwrap();
        q.check_consistency().unwrap();
        assert_eq!(p.kept(), keep.as_slice());
        let (proj, _) = f.projective().unwrap();
        assert_eq!(q.names(), proj.names());
        for (a, &i) in keep.iter().enumerate() {
            assert_eq!(q.power_relation(a), &delete(g.power_relation(i)));
            assert_eq!(proj.power_relation(a), q.power_relation(a));
            for (b, &j) in keep.iter().enumerate().skip(a + 1) {
                assert_eq!(q.conj_relation(a, b), &delete(g.conj_relation(i, j)));
                assert_eq!(proj.conj_relation(a, b), q.conj_relation(a, b));
            }
        }
    }
}

#[test]
fn direct_products() {
    let h3 = PcBuilder::new(3, ["x", "y", "z"])
        .conj(0, 1, &[(2, 1)])
        .build()
        .unwrap();
    let d = h3.direct_product(&h3).unwrap();
    d.check_consistency().unwrap();
    assert_eq!(d.log_order(), 6);
    assert_eq!(d.center().log_order(), 2);
    assert_eq!(d.nilpotency_class(), 2);
    assert_eq!(d.names(), ["x_1", "y_1", "z_1", "x_2", "y_2", "z_2"]);
    // the factors commute and keep their relations
    let (x1, x2, y1) = (d.gen(0), d.gen(3), d.gen(1));
    assert_eq!(d.comm(&x1, &x2), d.identity());
    assert_eq!(d.comm(&y1, &x1), d.gen(2));
    let xt = families::by_name("XT", 1).unwrap().pres;
    let dd = xt.direct_product(&d4()).unwrap();
    dd.check_consistency().unwrap();
    assert_eq!(dd.log_order(), xt.log_order() + 3);
    assert_eq!(dd.nilpotency_class(), xt.nilpotency_class().max(2));
    assert!(h3.direct_product(&d4()).is_err());
}

/// Central products of m copies of C_{2^r} ≀ C_2 = ⟨X, diag(1, ω)⟩ over
/// μ_{2^r}, compared with the m-qubit group ⟨X_j, diag(1, ω)_j⟩ (order and
/// class checked for small r and m).
#[test]
fn central_products_of_wreath_groups() {
    for r in 1..=3u32 {
        let w = families::x_diag_bits("W", 1, &[1], &[vec![0, 1]], r).unwrap();
        let g = &w.pres;
        assert_eq!(g.log_order(), 2 * r as usize + 1);
        let zs: Vec<Exps> = w.scalars.iter().map(|&k| g.gen(k)).collect();
        assert_eq!(g.subgroup(&zs), g.center());
        let mut p = g.clone();
        let mut pz = zs.clone();
        for m in 2..=3usize {
            let ident: Vec<(Exps, Exps)> = pz.iter().cloned().zip(zs.iter().cloned()).collect();
            let (q, proj) = p.central_product(g, &ident).unwrap();
            q.check_consistency().unwrap();
            let want = r as usize + m * (r as usize + 1);
            assert_eq!(q.log_order(), want, "r = {r}, m = {m}");
            assert_eq!(q.nilpotency_class(), r as usize + 1);
            assert_eq!(q.center().log_order(), r as usize);
            // the same as the m-qubit group
            let shifts: Vec<u64> = (0..m).map(|j| 1u64 << j).collect();
            let diag: Vec<Vec<u8>> = (0..m)
                .map(|j| (0..1usize << m).map(|x| ((x >> j) & 1) as u8).collect())
                .collect();
            let direct = families::x_diag_bits("Wm", m, &shifts, &diag, r).unwrap().pres;
            assert_eq!(direct.log_order(), want);
            assert_eq!(direct.nilpotency_class(), q.nilpotency_class());
            assert_eq!(
                direct
                    .series_factor_invariants(&direct.lower_central_series())
                    .unwrap(),
                q.series_factor_invariants(&q.lower_central_series()).unwrap()
            );
            // the identified scalars of the product
            pz = pz
                .iter()
                .map(|z| {
                    proj.apply(
                        &z.iter()
                            .copied()
                            .chain(std::iter::repeat_n(0, g.len()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            p = q;
        }
    }
}

#[test]
fn central_product_errors() {
    let d = d4();
    let c = d.gen(2);
    // s is not central
    assert!(d.central_product(&d, &[(d.gen(0), c.clone())]).is_err());
    // identifying c with the identity is not an isomorphism
    assert!(d.central_product(&d, &[(c.clone(), d.identity())]).is_err());
    // D4 ∘ D4 over the centers: order 2^5
    let (q, _) = d.central_product(&d, &[(c.clone(), c)]).unwrap();
    assert_eq!(q.log_order(), 5);
}
