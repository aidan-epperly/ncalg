//! Signed permutations acting on polynomials modulo an ideal: images of polynomials and words, the Reynolds average, invariance
//! tests, and orbit classes of normal words.  These are the only
//! implementations; `ncsos` and `multi` call them.

use super::perm::SignedPermutation;
use crate::algebra::FreeAlgebra;
use crate::coeff::Field;
use crate::poly::Polynomial;
use crate::reduce::Reducer;
use crate::word::{Var, Word};
use std::collections::HashMap;

/// `NF(σ(p))`.
pub fn image<C: Field>(sigma: &SignedPermutation, p: &Polynomial<C>, red: &Reducer<C>) -> Polynomial<C> {
    red.reduce(&sigma.apply(p, red.algebra()))
}

/// `NF(σ(w))` for a word.
pub fn word_image<C: Field>(sigma: &SignedPermutation, w: &Word, red: &Reducer<C>) -> Polynomial<C> {
    red.reduce(&sigma.apply_monomial::<C>(w, red.algebra()))
}

/// The Reynolds average `(1/|G|) Σ_g g(p)` over the elements of a group (not
/// reduced); `p` itself for the trivial group.
pub fn average<C: Field>(group: &[SignedPermutation], p: &Polynomial<C>, alg: &FreeAlgebra) -> Polynomial<C> {
    if group.is_empty() {
        return p.clone();
    }
    // images merged in chunks: summed one by one, every element moved the
    // whole accumulator (S_7 on 2000 terms of degree 7: 126 s, now 4.6 s);
    // chunks bound the images held at once
    const CHUNK: usize = 256;
    let mut acc = alg.zero();
    for part in group.chunks(CHUNK) {
        let images: Vec<Polynomial<C>> = part.iter().map(|g| g.apply(p, alg)).collect();
        acc += Polynomial::sum(alg, &images);
    }
    acc.scale(&C::from_i64(group.len() as i64).inverse())
}

/// Does `σ` fix `p` modulo the ideal (`NF(σ(p) − p) = 0`)?
pub fn preserves<C: Field>(sigma: &SignedPermutation, p: &Polynomial<C>, red: &Reducer<C>) -> bool {
    // with floating-point coefficients, up to rounding relative to p
    red.reduce(&sigma.apply(p, red.algebra()).sub_ref(p))
        .is_zero_up_to_rounding(p.max_magnitude())
}

/// Does `σ` map the ideal into itself (every basis element to zero)?
pub fn preserves_ideal<C: Field>(sigma: &SignedPermutation, red: &Reducer<C>) -> bool {
    red.basis()
        .iter()
        .all(|g| image(sigma, g, red).is_zero_up_to_rounding(g.max_magnitude()))
}

/// Orbit classes of normal words under a group acting monomially modulo the
/// ideal (every `NF(g(w))` a single term), cached.
///
/// [`class`](Self::class) gives the representative (the smallest word of the
/// orbit) and the coefficient `a_w` of the representative in the group
/// average of `w`: `avg(w) = (a_w / a_rep) avg(rep)`, and `a_w = 0` exactly
/// when the average of `w` vanishes.  Two polynomials have the same group
/// average iff their coefficients summed per class (weighted by `a_w`) agree.
#[derive(Debug)]
pub struct WordOrbits<'a, C: Field> {
    group: &'a [SignedPermutation],
    red: &'a Reducer<C>,
    cache: HashMap<Vec<Var>, (Vec<Var>, C)>,
}

impl<'a, C: Field> WordOrbits<'a, C> {
    /// Orbit classes of normal words under `group`, acting modulo the ideal of `red`.
    pub fn new(group: &'a [SignedPermutation], red: &'a Reducer<C>) -> Self {
        WordOrbits {
            group,
            red,
            cache: HashMap::new(),
        }
    }

    /// `(representative, a_w)`, or `None` when some `NF(g(w))` has more than
    /// one term (the action is not monomial on this word).
    pub fn class(&mut self, w: &Word) -> Option<(Vec<Var>, C)> {
        if let Some(c) = self.cache.get(w.as_slice()) {
            return Some(c.clone());
        }
        let mut images: Vec<(Vec<Var>, C)> = Vec::with_capacity(self.group.len());
        for g in self.group {
            let img = word_image(g, w, self.red);
            if img.num_terms() != 1 {
                return None;
            }
            let (iw, ic) = img.iter().next().expect("one term");
            images.push((iw.as_slice().to_vec(), ic.clone()));
        }
        let rep = images
            .iter()
            .map(|(iw, _)| iw.clone())
            .min()
            .expect("nonempty group");
        let mut a = C::zero();
        for (iw, c) in &images {
            if *iw == rep {
                a = a.add_ref(c);
            }
        }
        let a = a.mul_ref(&C::from_i64(self.group.len() as i64).inverse());
        // the whole orbit at once: NF(g(w)) = c_g w' gives
        // NF(h(w')) = NF((h g)(w)) / c_g for every h, and h g runs over the
        // group, so a_{w'} = a_w / c_g with the same representative
        for (iw, c) in &images {
            if !self.cache.contains_key(iw) {
                self.cache.insert(iw.clone(), (rep.clone(), a.div_ref(c)));
            }
        }
        self.cache.insert(w.as_slice().to_vec(), (rep.clone(), a.clone()));
        Some((rep, a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relations;
    use crate::{GroebnerBasis, GroebnerOptions};
    use num_rational::BigRational;

    #[test]
    fn average_and_orbits_on_two_qubits() {
        let (alg, rels) = relations::pauli::<num_complex::Complex<BigRational>>(2);
        let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
        let red = gb.reducer();
        let swap = SignedPermutation::permutation(vec![3, 4, 5, 0, 1, 2]);
        let flip = SignedPermutation::signs(vec![1, -1, -1, 1, -1, -1]); // π-rotation about X on both sites
        let group = crate::symmetry::perm::signed_permutation_group(&[swap.clone(), flip.clone()], 6, 100);
        assert_eq!(group.len(), 4);
        assert!(preserves_ideal(&swap, red) && preserves_ideal(&flip, red));
        let p: Polynomial<_> = alg.parse("X0 + Y0*Y1").unwrap();
        let avg = red.reduce(&average(&group, &p, &alg));
        assert_eq!(avg, red.reduce(&alg.parse("1/2*X0 + 1/2*X1 + Y0*Y1").unwrap()));
        assert!(group.iter().all(|g| preserves(g, &avg, red)));
        let mut orbits = WordOrbits::new(&group, red);
        let x1 = alg.parse::<num_complex::Complex<BigRational>>("X1").unwrap();
        let (rep, a) = orbits.class(x1.iter().next().unwrap().0).unwrap();
        assert_eq!(rep, vec![0]); // X0
        assert_eq!(
            a,
            num_complex::Complex::new(
                BigRational::new(1.into(), 2.into()),
                BigRational::from_integer(0.into())
            )
        );
        let y0 = alg.parse::<num_complex::Complex<BigRational>>("Y0").unwrap();
        let (_, a) = orbits.class(y0.iter().next().unwrap().0).unwrap();
        assert!(a.re == BigRational::from_integer(0.into())); // Y0 averages to zero under the flip
                                                              // a non-automorphism is detected
        assert!(!preserves_ideal(
            &SignedPermutation::signs(vec![1, -1, 1, 1, 1, 1]),
            red
        ));
    }
}
