//! Polynomial arithmetic, parsing, printing and transformations.

use ncalg::{BigInt, BigRational, Complex, FreeAlgebra, Polynomial, Word};

fn alg2() -> FreeAlgebra {
    FreeAlgebra::new(["x", "y"])
}

#[test]
fn parse_display_roundtrip() {
    let alg = alg2();
    let inputs = [
        "x",
        "y",
        "1",
        "-1",
        "x*y - y*x",
        "x^3 + 3*x^2*y - 2",
        "-x + y",
        "2*x*y*x",
        "0",
    ];
    for s in inputs {
        let p: Polynomial<i64> = alg.parse(s).unwrap();
        let printed = p.to_string();
        let q: Polynomial<i64> = alg.parse(&printed).unwrap();
        assert_eq!(p, q, "roundtrip of {:?} via {:?}", s, printed);
    }
}

#[test]
fn display_formats() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x*x*x - 2*y*y + x").unwrap();
    assert_eq!(p.to_string(), "x^3 - 2*y^2 + x");
    let p: Polynomial<BigRational> = alg.parse("1/2*x - 3/4").unwrap();
    assert_eq!(p.to_string(), "1/2*x - 3/4");
    let p: Polynomial<Complex<f64>> = alg.parse("(1+2i)*x + i*y - 3").unwrap();
    assert_eq!(p.to_string(), "i*y + (1+2i)*x - 3");
    let p: Polynomial<Complex<f64>> = alg.parse("x - i*y - 2*i").unwrap();
    assert_eq!(p.to_string(), "-i*y + x - 2i");
    let p: Polynomial<Complex<f64>> = alg.parse("x - 2*i*y").unwrap();
    assert_eq!(p.to_string(), "-2i*y + x");
    let p: Polynomial<f64> = alg.parse("2.5*x - 1.25").unwrap();
    assert_eq!(p.to_string(), "2.5*x - 1.25");
    assert_eq!(alg.zero::<i64>().to_string(), "0");
}

#[test]
fn ring_axioms_on_samples() {
    let alg = alg2();
    let a: Polynomial<i64> = alg.parse("x*y - 2*y + 1").unwrap();
    let b: Polynomial<i64> = alg.parse("y*x*x + x - 3").unwrap();
    let c: Polynomial<i64> = alg.parse("x^2 - y^2 + x*y").unwrap();
    // associativity
    assert_eq!((&a * &b) * &c, &a * (&b * &c));
    // distributivity
    assert_eq!(&a * (&b + &c), &a * &b + &a * &c);
    assert_eq!((&a + &b) * &c, &a * &c + &b * &c);
    // noncommutativity
    assert_ne!(&a * &b, &b * &a);
    // additive inverse
    assert!((&a - &a).is_zero());
    assert!((&a + (-&a)).is_zero());
    // unit
    let one = alg.one::<i64>();
    assert_eq!(&a * &one, a);
    assert_eq!(&one * &a, a);
}

#[test]
fn powers_and_binomial_counts() {
    let alg = alg2();
    let s: Polynomial<i64> = alg.parse("x + y").unwrap();
    // (x+y)^n has 2^n distinct noncommutative words
    for n in 0..6 {
        assert_eq!(s.pow(n).num_terms(), 1 << n);
    }
    assert_eq!(s.pow(0), alg.one());
    let p3 = &s * &s * &s;
    assert_eq!(s.pow(3), p3);
}

#[test]
fn commutators() {
    let alg = alg2();
    let x: Polynomial<i64> = alg.gen("x");
    let y: Polynomial<i64> = alg.gen("y");
    let c = Polynomial::commutator(&x, &y);
    assert_eq!(c, &x * &y - &y * &x);
    assert_eq!(c, alg.parse("[x, y]").unwrap());
    assert!(Polynomial::commutator(&x, &x).is_zero());
    let ac = Polynomial::anticommutator(&x, &y);
    assert_eq!(ac, alg.parse("{x, y}").unwrap());
    // Jacobi identity holds in any associative algebra
    let z: Polynomial<i64> = alg.parse("x*y + 3").unwrap();
    let j = Polynomial::commutator(&x, &Polynomial::commutator(&y, &z))
        + Polynomial::commutator(&y, &Polynomial::commutator(&z, &x))
        + Polynomial::commutator(&z, &Polynomial::commutator(&x, &y));
    assert!(j.is_zero());
}

#[test]
fn scalar_operations() {
    let alg = alg2();
    let x: Polynomial<i64> = alg.gen("x");
    assert_eq!((2 * &x).to_string(), "2*x");
    assert_eq!((&x * 3).to_string(), "3*x");
    assert_eq!((&x + 1).to_string(), "x + 1");
    assert_eq!((1 - &x).to_string(), "-x + 1");
    let zero_scalar = 0i64;
    assert!((&x * zero_scalar).is_zero());
    let xf: Polynomial<f64> = alg.gen("x");
    assert_eq!((0.5 * &xf).to_string(), "0.5*x");
    let mut m = x.clone();
    m *= 4;
    m += &x;
    assert_eq!(m.to_string(), "5*x");
}

#[test]
fn leading_terms_follow_deglex() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x*y + y*x + x^3 + y").unwrap();
    assert_eq!(*p.leading_word().unwrap(), Word::from_vec(vec![0, 0, 0]));
    assert_eq!(p.degree(), Some(3));
    assert_eq!(p.min_degree(), Some(1));
    assert_eq!(p.coeff_of(&[1, 0]), 1);
    assert_eq!(p.coeff_of(&[1, 1]), 0);
    assert!(p.contains_word(&[1]));
    assert_eq!(p.tail().num_terms(), 3);
    assert!(alg.zero::<i64>().degree().is_none());
    assert_eq!(p.support_vars(), vec![0, 1]);
}

#[test]
fn custom_variable_chain() {
    // make x the larger generator
    let alg = FreeAlgebra::with_order(["x", "y"], ncalg::DegLex::with_chain(&[1, 0]));
    let p: Polynomial<i64> = alg.parse("x*y - y*x").unwrap();
    assert_eq!(p.to_string(), "x*y - y*x");
}

#[test]
fn substitution_and_derivation() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x*y*x").unwrap();
    let img: Polynomial<i64> = alg.parse("x + 1").unwrap();
    let s = p.substitute(0, &img);
    assert_eq!(s, alg.parse("(x+1)*y*(x+1)").unwrap());
    // derivation: delete one x at a time
    let d = p.derivation(0);
    assert_eq!(d, alg.parse("y*x + x*y").unwrap());
    // Leibniz rule
    let a: Polynomial<i64> = alg.parse("x*y + y").unwrap();
    let b: Polynomial<i64> = alg.parse("y*x - x").unwrap();
    let lhs = (&a * &b).derivation(1);
    let rhs = &a.derivation(1) * &b + &a * &b.derivation(1);
    assert_eq!(lhs, rhs);
}

#[test]
fn reversal_and_adjoint() {
    let alg = alg2();
    let p: Polynomial<Complex<f64>> = alg.parse("i*x*y + 2*x").unwrap();
    let adj = p.adjoint();
    assert_eq!(adj, alg.parse("-i*y*x + 2*x").unwrap());
    assert_eq!(p.reversed(), alg.parse("i*y*x + 2*x").unwrap());
    assert!(!p.is_hermitian());
    let h: Polynomial<Complex<f64>> = alg.parse("x*y + y*x + i*(x*y - y*x)").unwrap();
    assert!(h.is_hermitian());
    // adjoint is an anti-homomorphism
    let q: Polynomial<Complex<f64>> = alg.parse("y - i").unwrap();
    assert_eq!((&p * &q).adjoint(), q.adjoint() * p.adjoint());
}

#[test]
fn homogeneous_components_and_truncation() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x^3 + x*y + y*x - 5").unwrap();
    assert!(!p.is_homogeneous());
    let comps = p.homogeneous_components();
    assert_eq!(comps.iter().map(|(d, _)| *d).collect::<Vec<_>>(), vec![0, 2, 3]);
    assert_eq!(comps[1].1, alg.parse("x*y + y*x").unwrap());
    assert_eq!(p.truncate_degree(2), alg.parse("x*y + y*x - 5").unwrap());
    assert!(comps[1].1.is_homogeneous());
}

#[test]
fn coefficient_conversions() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("2*x*y - 4").unwrap();
    let q = p.to_rational();
    assert_eq!(q.to_string(), "2*x*y - 4");
    let half: Polynomial<BigRational> = alg.parse("1/2*x + 1/3").unwrap();
    let z = half.to_integer_primitive();
    assert_eq!(z.to_string(), "3*x + 2");
    let big: Polynomial<BigInt> = alg.parse("6*x - 9*y").unwrap();
    assert_eq!(big.content(), BigInt::from(3));
    // leading term is -9*y (y > x), so the primitive part is normalized to a positive leading coefficient
    assert_eq!(big.primitive_part().to_string(), "3*y - 2*x");
    let f = p.to_f64();
    assert_eq!(f.coeff_of(&[0, 1]), 2.0);
    let neg: Polynomial<i64> = alg.parse("-6*x + 9*y").unwrap();
    assert_eq!(neg.primitive_part().to_string(), "3*y - 2*x");
}

#[test]
fn word_multiplications() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x + y").unwrap();
    assert_eq!(p.mul_word_left(&[1]), alg.parse("y*x + y*y").unwrap());
    assert_eq!(p.mul_word_right(&[0]), alg.parse("x*x + y*x").unwrap());
    assert_eq!(p.sandwich(&[1], &[0]), alg.parse("y*x*x + y*y*x").unwrap());
}

#[test]
fn from_terms_merges_and_drops_zeros() {
    let alg = alg2();
    let p = Polynomial::from_terms(
        alg.clone(),
        vec![
            (Word::from_vec(vec![0]), 2i64),
            (Word::from_vec(vec![0]), -2),
            (Word::from_vec(vec![1]), 3),
            (Word::empty(), 0),
        ],
    );
    assert_eq!(p.to_string(), "3*y");
}

/// Float zero tests are relative: a small term is a term, a sum that
/// cancels to rounding level relative to its summands is zero, at any scale.
#[test]
fn float_zero_tests_are_relative() {
    let alg = alg2();
    let p: Polynomial<f64> = alg.parse("x + 1e-13*y").unwrap();
    assert_eq!(p.num_terms(), 2, "a small coefficient is kept");
    for scale in ["1", "1e-12", "1e12"] {
        let a: Polynomial<f64> = alg.parse(&format!("{scale}*(0.1*x + 0.2*x)")).unwrap();
        let b: Polynomial<f64> = alg.parse(&format!("{scale}*0.3*x")).unwrap();
        assert!(
            (a.clone() - b.clone()).is_zero(),
            "scale {scale}: 0.1 + 0.2 - 0.3 cancels"
        );
        let c: Polynomial<f64> = alg
            .parse(&format!("{scale}*(0.1*x + 0.2*x - 0.3*x + 1e-9*y)"))
            .unwrap();
        assert_eq!(c.num_terms(), 1, "scale {scale}: the y term survives");
    }
}

#[test]
fn word_quotients_iterators_and_hashing() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x*y*x + 2*x*y - x").unwrap();
    let q = p.left_quotient_by_word(&[0]).unwrap();
    assert_eq!(q, alg.parse("y*x + 2*y - 1").unwrap());
    assert_eq!(q.mul_word_left(&[0]), p);
    assert!(p.left_quotient_by_word(&[1]).is_none());
    assert!(p.right_quotient_by_word(&[0]).is_none()); // 2*x*y does not end in x
    let r: Polynomial<i64> = alg.parse("x*y*x + y*x").unwrap();
    assert_eq!(
        r.right_quotient_by_word(&[1, 0]).unwrap(),
        alg.parse("x + 1").unwrap()
    );
    assert_eq!(p.words().count(), 3);
    assert_eq!(p.coeffs().copied().collect::<Vec<_>>(), vec![1, 2, -1]);
    let mut set = std::collections::HashSet::new();
    set.insert(p.clone());
    set.insert(alg.parse::<i64>("x*y*x + 2*x*y - x").unwrap());
    set.insert(q);
    assert_eq!(set.len(), 2);
    let words = alg.words_of_length(2);
    assert_eq!(words.len(), 4);
    assert_eq!(words[0], Word::from_vec(vec![0, 0]));
    assert_eq!(words[3], Word::from_vec(vec![1, 1]));
    assert_eq!(alg.words_of_length(0), vec![Word::empty()]);
}

#[test]
#[should_panic(expected = "different free algebras")]
fn mixing_algebras_panics() {
    let a = FreeAlgebra::new(["x", "y"]);
    let b = FreeAlgebra::new(["u", "v"]);
    let p: Polynomial<i64> = a.gen("x");
    let q: Polynomial<i64> = b.gen("u");
    let _ = p + q;
}

#[test]
fn checked_integer_overflow_panics_instead_of_wrapping() {
    let alg = alg2();
    let p: Polynomial<i64> = alg.parse("x").unwrap() * i64::MAX;
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| &p + &p));
    assert!(r.is_err());
}

/// `+=` / `-=` with short summands insert in place: any order of
/// additions, with cancellations, gives the same polynomial as `from_terms`,
/// and parsing a long sum gives the same as adding its terms.
#[test]
fn in_place_sums_match_from_terms() {
    use ncalg::{FreeAlgebra, Polynomial, Word};
    let alg = FreeAlgebra::new(vec!["x", "y", "z"]);
    let words: Vec<Word> = (0..60u32)
        .map(|i| Word::from_vec(vec![i % 3, (i / 3) % 3, (i / 9) % 3][..(i % 3 + 1) as usize].to_vec()))
        .collect();
    let mut state = 11u64;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as usize
    };
    let mut expected: Vec<(Word, i64)> = Vec::new();
    let mut acc: Polynomial<f64> = alg.constant(0.0);
    for _ in 0..400 {
        let w = words[next() % words.len()].clone();
        let c = (next() % 7) as i64 - 3;
        let term: Polynomial<f64> = Polynomial::from_terms(alg.clone(), [(w.clone(), c as f64)]);
        if next() % 2 == 0 {
            acc += &term;
            expected.push((w, c));
        } else {
            acc -= term;
            expected.push((w, -c));
        }
    }
    let reference: Polynomial<f64> =
        Polynomial::from_terms(alg.clone(), expected.iter().map(|(w, c)| (w.clone(), *c as f64)));
    assert_eq!(acc, reference);
    let text = expected
        .iter()
        .map(|(w, c)| format!("{} * {}", c, alg.format_word(w)))
        .collect::<Vec<_>>()
        .join(" + ");
    let parsed: Polynomial<f64> = alg.parse(&text).unwrap();
    assert_eq!(parsed, reference);
}

/// The complex inverse neither overflows nor underflows (Smith's
/// algorithm; re² + im² gave 1/1e200 = 0 and 1/1e-170 a panic).
#[test]
fn complex_inverse_at_extreme_magnitudes() {
    use ncalg::{Complex, Field};
    for (re, im) in [
        (1e200, 0.0),
        (3e200, -4e200),
        (1e-170, 2e-170),
        (0.0, 5e-200),
        (1.5, -2.0),
    ] {
        let z = Complex::new(re, im);
        let w = z.inverse();
        let one = z * w;
        assert!((one.re - 1.0).abs() < 1e-14 && one.im.abs() < 1e-14, "{z} -> {w}");
    }
}

/// Readings the parser used to resolve silently are errors.
#[test]
fn ambiguous_numeric_forms_are_errors() {
    use ncalg::{BigRational, FreeAlgebra, Polynomial};
    let alg = FreeAlgebra::new(["e1", "e2", "x"]);
    // 2e1*e2 was 20 e2
    assert!(alg.parse::<f64>("2e1*e2").is_err());
    assert!(alg.parse::<f64>("2*e1*e2").is_ok());
    // exponent notation stays available when nothing clashes
    let p: Polynomial<f64> = alg.parse("2e-1*x + 3e5").unwrap();
    assert_eq!(p.iter().count(), 2);
    let plain = FreeAlgebra::new(["x"]);
    assert!(plain.parse::<f64>("2e1*x").is_ok());
    // 2/3^2 was (2/3)^2 = 4/9
    assert!(plain.parse::<BigRational>("2/3^2").is_err());
    assert!(plain.parse::<BigRational>("(2/3)^2").is_ok());
    // x^2^3 was x^6
    assert!(plain.parse::<f64>("x^2^3").is_err());
    assert!(plain.parse::<f64>("(x^2)^3").is_ok());
}

/// A generator named like a coefficient literal cannot be serialized
/// faithfully (`i` over complex coefficients), so saving refuses.
#[test]
fn literal_named_generator_is_not_serialized() {
    use ncalg::io::polynomials_to_string;
    use ncalg::{BigRational, Complex, FreeAlgebra, Polynomial};
    let alg = FreeAlgebra::new(["i", "j"]);
    let p: Polynomial<Complex<BigRational>> = alg.parse("j").unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| polynomials_to_string(&alg, &[p])));
    assert!(r.is_err());
    // real coefficients: "i" is no literal, fine
    let q: Polynomial<BigRational> = alg.parse("i*j").unwrap();
    assert!(!polynomials_to_string(&alg, &[q]).is_empty());
}

/// Literals that are not finite or would need a huge power of ten are
/// rejected rather than accepted or hanging.
#[test]
fn literal_edge_cases() {
    use ncalg::{BigRational, Coeff};
    assert!(<f64 as Coeff>::parse_literal("inf/1").is_none());
    assert!(<f64 as Coeff>::parse_literal("1e300/1e-300").is_none());
    assert_eq!(<f64 as Coeff>::parse_literal("1/4"), Some(0.25));
    assert!(<BigRational as Coeff>::parse_literal("1e-99999999999").is_none());
    assert!(<BigRational as Coeff>::parse_literal("2.5e3").is_some());
}
