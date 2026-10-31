//! Normal words as an index set for NPA / noncommutative SOS moment matrices.
//!
//! For a `*`-algebra given by relations, the normal words of length ≤ k form
//! a basis of the level-k "localizing" space; products `u† v` of basis words
//! reduce to normal form, and the coefficients of that normal form are the
//! entries of the moment matrix in terms of the unknown moments.  This example
//! prints the index set and a few reduced products for the Majorana algebra
//! and the 2-qubit Pauli algebra.
//!
//! `cargo run --example npa_words`

use ncalg::relations;
use ncalg::{BigRational, Complex, GroebnerOptions, QuotientAlgebra};

fn main() {
    // --- Majorana modes -------------------------------------------------------
    let (alg, rels) = relations::majorana::<BigRational>(4);
    let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
    assert!(q.groebner_basis().is_star_closed());
    let level = 2;
    let words = q.normal_words_up_to(level);
    println!(
        "Majorana algebra on 4 modes: {} normal words up to degree {}:",
        words.len(),
        level
    );
    for w in &words {
        print!("  {}", alg.format_word(w));
    }
    println!(
        "\n(Hilbert series {:?}, total dimension {:?})",
        q.hilbert_series(4),
        q.dimension()
    );

    // moment-matrix entries: u† v for basis words u, v
    println!("\nsome reduced products u† v:");
    let basis: Vec<_> = words
        .iter()
        .map(|w| q.element(&alg.monomial(w.clone())))
        .collect();
    for (i, u) in basis.iter().enumerate().take(4) {
        for (j, v) in basis.iter().enumerate().take(6) {
            if j < i {
                continue;
            }
            let entry = u.adjoint() * v;
            println!("  ({},{}): ({})† ({}) = {}", i, j, u, v, entry);
        }
    }

    // --- Pauli strings --------------------------------------------------------
    let (palg, rels) = relations::pauli::<Complex<BigRational>>(2);
    let q = QuotientAlgebra::from_relations(rels, &GroebnerOptions::default());
    assert!(q.groebner_basis().is_star_closed());
    let words = q.normal_words_up_to(2);
    println!(
        "\n2-qubit Pauli algebra: {} normal words (Pauli strings) up to degree 2:",
        words.len()
    );
    for w in &words {
        print!("  {}", palg.format_word(w));
    }
    println!();
    let a = q.parse("X0*Y1").unwrap();
    let b = q.parse("Z0*Z1").unwrap();
    println!("(X0 Y1)† (Z0 Z1) = {}", a.adjoint() * &b);
    println!("[X0 Y1, Z0 Z1]    = {}", a.commutator(&b));
    println!("{{X0 Y1, Z0 Z1}}    = {}", a.anticommutator(&b));
}
