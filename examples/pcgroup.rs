//! Polycyclic presentations of Clifford-hierarchy groups.
//!
//! `cargo run --release --example pcgroup`
//!
//! * builds Pauli-CZ, Pauli-CCZ, XS, XT and D4^N as pc presentations (no enumeration),
//! * prints the lower exponent-2 central series blocks, linear and projective,
//! * times the collector,
//! * computes the group algebra's Groebner basis under the wreath ordering
//!   (complete at once; under deglex Pauli-CZ(5) did not finish in 2 min).

use ncalg::pcgroup::families;
use ncalg::{BigRational, GroebnerBasis, GroebnerOptions, QuotientAlgebra};
use std::time::Instant;

fn main() {
    for (fam, size) in [
        ("PauliCZ", 5),
        ("PauliCZ", 8),
        ("PauliCCZ", 6),
        ("XS", 6),
        ("XS", 12),
        ("XT", 6),
        ("XT", 12),
        ("D4N", 12),
    ] {
        let t = Instant::now();
        let f = families::by_name(fam, size).unwrap();
        let tb = t.elapsed();
        let (q, _) = f.projective().unwrap();
        let g = &f.pres;
        let n = g.len();
        // collector timing: random products
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 11
        };
        let elems: Vec<_> = (0..2000)
            .map(|_| (0..n).map(|_| (rnd() & 1) as u8).collect::<Vec<u8>>())
            .collect();
        let t = Instant::now();
        let mut acc = g.identity();
        for k in 0..20000 {
            acc = g.mul(&acc, &elems[k % elems.len()]);
        }
        let per = t.elapsed().as_secs_f64() / 20000.0;
        println!(
            "{fam}({size}): log2|G| = {n}, blocks {:?}, projective blocks {:?}, built in {:.2?}, {:.2} us per product",
            f.blocks,
            q.series_blocks().unwrap(),
            tb,
            per * 1e6
        );
    }
    for (fam, size) in [("PauliCZ", 4), ("PauliCZ", 5), ("XS", 5), ("XT", 3)] {
        let f = families::by_name(fam, size).unwrap();
        let t = Instant::now();
        let (_alg, rels) = f.pres.group_algebra_relations::<BigRational>();
        let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
        let q = QuotientAlgebra::new(gb);
        println!(
            "group algebra of {fam}({size}) under wreath: dim {:?} (|G| = 2^{}), {:.2?}",
            q.dimension(),
            f.pres.len(),
            t.elapsed()
        );
    }
}
