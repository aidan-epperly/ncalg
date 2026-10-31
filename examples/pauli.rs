//! The two-qubit Pauli algebra: Gröbner basis, normal words, quotient arithmetic.
//!
//! `cargo run --example pauli`

use ncalg::relations;
use ncalg::{BigRational, Complex, GroebnerBasis, GroebnerOptions, Polynomial, QuotientAlgebra};

fn main() {
    type C = Complex<BigRational>;
    let (alg, rels) = relations::pauli::<C>(2);
    println!("algebra: {}", alg);
    println!("{} defining relations, e.g. {}", rels.len(), rels[3]);

    let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
    println!("\n{}", gb);
    println!("stats: {:?}", gb.stats());

    let q = QuotientAlgebra::new(gb);
    println!("dimension of the quotient: {:?}", q.dimension());
    println!("normal words by degree: {:?}", q.hilbert_series(3));
    println!(
        "normal words of degree 2: {}",
        q.normal_words(2)
            .iter()
            .map(|w| alg.format_word(w))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // arithmetic in the quotient: every element is kept in normal form
    let x0 = q.gen("X0");
    let z0 = q.gen("Z0");
    let x1 = q.gen("X1");
    let z1 = q.gen("Z1");
    let bell = &x0 * &x1 + &z0 * &z1; // stabilizer-style operator
    println!("\nX0 X1 + Z0 Z1              = {}", bell);
    println!("(X0 X1 + Z0 Z1)^2          = {}", bell.pow(2));
    println!(
        "[X0 X1, Z0 Z1]             = {}",
        (&x0 * &x1).commutator(&(&z0 * &z1))
    );
    println!("[X0, Z0]                   = {}", x0.commutator(&z0));
    println!("{{X0, Z0}}                   = {}", x0.anticommutator(&z0));

    // a Heisenberg-type Hamiltonian and its powers, reduced while expanding
    let h: Polynomial<C> = alg.parse("X0*X1 + Y0*Y1 + Z0*Z1").unwrap();
    let h = q.element(&h);
    println!("\nH   = {}", h);
    println!("H^2 = {}", h.pow(2));
    println!("H^3 = {}", h.pow(3));
    // H has eigenvalues 1 (triplet) and -3 (singlet): (H - 1)(H + 3) = 0
    let one = q.one();
    let three = q.constant(ncalg::Coeff::from_i64(3));
    let min_poly = (&h - &one) * (&h + &three);
    println!("(H - 1)(H + 3) = {}", min_poly);
    assert!(min_poly.is_zero());
}
