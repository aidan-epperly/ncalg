//! Ready-made relation sets for common algebras.
//!
//! Each function returns the free algebra (with a sensible generator order
//! for the default deglex ordering) together with generators of the defining
//! two-sided ideal.  Feed them to [`crate::GroebnerBasis::compute`] or
//! [`crate::QuotientAlgebra::from_relations`].
//!
//! | function | algebra | Gröbner basis |
//! |---|---|---|
//! | [`commutative`] | polynomial ring `K[x_1..x_n]` | the relations themselves |
//! | [`q_commutative`] | quantum plane / quantum affine space | the relations themselves |
//! | [`weyl`] | Weyl algebra `A_n` | the relations themselves |
//! | [`lie_algebra`], [`sl2`], [`so3`] | universal enveloping algebra `U(g)` | the relations (PBW theorem) iff the bracket satisfies Jacobi |
//! | [`pauli`] | `n`-qubit Pauli algebra ≅ `M_{2^n}(C)` | finite, 4^n normal words |
//! | [`majorana`], [`clifford`] | Clifford algebra | finite, 2^n normal words |
//! | [`exterior`] | exterior (Grassmann) algebra | finite, 2^n normal words |

use crate::algebra::FreeAlgebra;
use crate::coeff::{Coeff, ComplexCoeff};
use crate::poly::Polynomial;
use crate::word::{Var, Word};

/// `x_j x_i - x_i x_j` for `i < j`: the commutative polynomial ring.
pub fn commutative<C: Coeff>(alg: &FreeAlgebra) -> Vec<Polynomial<C>> {
    let n = alg.num_vars() as Var;
    let mut rels = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let xi: Polynomial<C> = alg.var(i);
            let xj: Polynomial<C> = alg.var(j);
            rels.push(Polynomial::commutator(&xj, &xi));
        }
    }
    rels
}

/// `x_j x_i - q x_i x_j` for `i < j`: the quantum affine space.
pub fn q_commutative<C: Coeff>(alg: &FreeAlgebra, q: &C) -> Vec<Polynomial<C>> {
    let n = alg.num_vars() as Var;
    let mut rels = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let xi: Polynomial<C> = alg.var(i);
            let xj: Polynomial<C> = alg.var(j);
            rels.push(&xj * &xi - (&xi * &xj).scale(q));
        }
    }
    rels
}

/// The `n`-th Weyl algebra on generators `x1..xn, d1..dn` with
/// `d_i x_i - x_i d_i = 1` and all other pairs commuting.  Normal words are
/// `x^a d^b`.
pub fn weyl<C: Coeff>(n: usize) -> (FreeAlgebra, Vec<Polynomial<C>>) {
    let names: Vec<String> = (1..=n)
        .map(|i| format!("x{}", i))
        .chain((1..=n).map(|i| format!("d{}", i)))
        .collect();
    let alg = FreeAlgebra::new(names);
    let mut rels = commutative::<C>(&alg);
    // fix the d_i x_i relations: commutative() produced d_i x_i - x_i d_i,
    // we want d_i x_i - x_i d_i - 1
    for i in 0..n {
        let x: Polynomial<C> = alg.var(i as Var);
        let d: Polynomial<C> = alg.var((n + i) as Var);
        let comm = Polynomial::commutator(&d, &x);
        if let Some(pos) = rels.iter().position(|r| *r == comm) {
            rels[pos] = comm - C::one();
        }
    }
    (alg, rels)
}

/// Universal enveloping algebra of a Lie algebra with basis `x_0..x_{n-1}`:
/// relations `x_j x_i - x_i x_j - [x_j, x_i]` for `i < j`, where
/// `bracket(j, i)` returns the (linear) polynomial `[x_j, x_i]`.
pub fn lie_algebra<C, F>(alg: &FreeAlgebra, mut bracket: F) -> Vec<Polynomial<C>>
where
    C: Coeff,
    F: FnMut(Var, Var) -> Polynomial<C>,
{
    let n = alg.num_vars() as Var;
    let mut rels = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let xi: Polynomial<C> = alg.var(i);
            let xj: Polynomial<C> = alg.var(j);
            let b = bracket(j, i);
            alg.check_same(b.algebra());
            rels.push(Polynomial::commutator(&xj, &xi) - b);
        }
    }
    rels
}

/// `U(sl_2)` on generators `e, f, h` with `[h, e] = 2e`, `[h, f] = -2f`,
/// `[e, f] = h`.  Generator order `e < f < h`, so normal words are `e^a f^b h^c`.
pub fn sl2<C: Coeff>() -> (FreeAlgebra, Vec<Polynomial<C>>) {
    let alg = FreeAlgebra::new(["e", "f", "h"]);
    let two = C::from_i64(2);
    let rels = lie_algebra(&alg, |j, i| {
        let e: Polynomial<C> = alg.var(0);
        let f: Polynomial<C> = alg.var(1);
        let h: Polynomial<C> = alg.var(2);
        match (j, i) {
            (1, 0) => h.negated(),             // [f, e] = -h
            (2, 0) => e.scale(&two),           // [h, e] = 2e
            (2, 1) => f.scale(&two).negated(), // [h, f] = -2f
            _ => unreachable!(),
        }
    });
    (alg, rels)
}

/// `U(so_3)` on `x, y, z` with `[x, y] = z`, `[y, z] = x`, `[z, x] = y`.
pub fn so3<C: Coeff>() -> (FreeAlgebra, Vec<Polynomial<C>>) {
    let alg = FreeAlgebra::new(["x", "y", "z"]);
    let rels = lie_algebra(&alg, |j, i| {
        let x: Polynomial<C> = alg.var(0);
        let y: Polynomial<C> = alg.var(1);
        let z: Polynomial<C> = alg.var(2);
        match (j, i) {
            (1, 0) => z.negated(), // [y, x] = -z
            (2, 0) => y,           // [z, x] = y
            (2, 1) => x.negated(), // [z, y] = -x
            _ => unreachable!(),
        }
    });
    (alg, rels)
}

/// Pauli algebra on `n` qubits with generators `X0, Y0, Z0, X1, …`:
/// `P^2 = 1`, `XY = iZ`, `YX = -iZ` (and cyclic), different sites commute.
/// Needs complex coefficients.  Normal words are products of at most one
/// Pauli per site in site order (4^n of them).
pub fn pauli<C: Coeff + ComplexCoeff>(n: usize) -> (FreeAlgebra, Vec<Polynomial<C>>) {
    let names: Vec<String> = (0..n)
        .flat_map(|q| ["X", "Y", "Z"].into_iter().map(move |p| format!("{}{}", p, q)))
        .collect();
    let alg = FreeAlgebra::new(names);
    let i_unit = C::imaginary_unit();
    let mut rels = Vec::new();
    for q in 0..n {
        let x: Polynomial<C> = alg.var((3 * q) as Var);
        let y: Polynomial<C> = alg.var((3 * q + 1) as Var);
        let z: Polynomial<C> = alg.var((3 * q + 2) as Var);
        for p in [&x, &y, &z] {
            rels.push(p * p - C::one());
        }
        // XY = iZ, YZ = iX, ZX = iY and the reversed products
        for (a, b, c) in [(&x, &y, &z), (&y, &z, &x), (&z, &x, &y)] {
            rels.push(a * b - c.scale(&i_unit));
            rels.push(b * a + c.scale(&i_unit));
        }
    }
    // different sites commute
    for q1 in 0..n {
        for q2 in (q1 + 1)..n {
            for a in 0..3 {
                for b in 0..3 {
                    let p1: Polynomial<C> = alg.var((3 * q1 + a) as Var);
                    let p2: Polynomial<C> = alg.var((3 * q2 + b) as Var);
                    rels.push(Polynomial::commutator(&p2, &p1));
                }
            }
        }
    }
    (alg, rels)
}

/// Clifford algebra with `γ_i^2 = sign_i` (each sign `±1`, or any scalar) and
/// `γ_i γ_j + γ_j γ_i = 0` for `i ≠ j`.  Generators `g0, g1, …`.
pub fn clifford<C: Coeff>(signs: &[i64]) -> (FreeAlgebra, Vec<Polynomial<C>>) {
    let n = signs.len();
    let alg = FreeAlgebra::new((0..n).map(|i| format!("g{}", i)));
    let mut rels = Vec::new();
    for i in 0..n {
        let g: Polynomial<C> = alg.var(i as Var);
        rels.push(&g * &g - C::from_i64(signs[i]));
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let gi: Polynomial<C> = alg.var(i as Var);
            let gj: Polynomial<C> = alg.var(j as Var);
            rels.push(Polynomial::anticommutator(&gj, &gi));
        }
    }
    (alg, rels)
}

/// Majorana fermion algebra: `n` Hermitian generators with `γ_i^2 = 1` and
/// `{γ_i, γ_j} = 0`.  Normal words are strictly increasing products (2^n).
pub fn majorana<C: Coeff>(n: usize) -> (FreeAlgebra, Vec<Polynomial<C>>) {
    clifford(&vec![1; n])
}

/// Exterior algebra on `n` generators `e0, e1, …`: `e_i^2 = 0`,
/// `e_i e_j + e_j e_i = 0`.
pub fn exterior<C: Coeff>(n: usize) -> (FreeAlgebra, Vec<Polynomial<C>>) {
    clifford(&vec![0; n])
}

/// Group algebra of the symmetric group `S_3` as a Coxeter group:
/// `s^2 = 1`, `t^2 = 1`, `s t s = t s t`.  Six normal words.
pub fn symmetric_group_3<C: Coeff>() -> (FreeAlgebra, Vec<Polynomial<C>>) {
    let alg = FreeAlgebra::new(["s", "t"]);
    let s: Polynomial<C> = alg.var(0);
    let t: Polynomial<C> = alg.var(1);
    let rels = vec![
        &s * &s - C::one(),
        &t * &t - C::one(),
        &s * &t * &s - &t * &s * &t,
    ];
    (alg, rels)
}

/// Group algebra of the symmetric group `S_n` with Coxeter generators
/// `s1, …, s_{n-1}`: `s_i^2 = 1`, `s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}`,
/// `s_i s_j = s_j s_i` for `|i - j| > 1`.  The quotient has dimension `n!`.
pub fn symmetric_group<C: Coeff>(n: usize) -> (FreeAlgebra, Vec<Polynomial<C>>) {
    assert!(n >= 2, "S_n needs n >= 2");
    let m = n - 1;
    let alg = FreeAlgebra::new((1..=m).map(|i| format!("s{}", i)));
    let s: Vec<Polynomial<C>> = alg.gens();
    let mut rels = Vec::new();
    for i in 0..m {
        rels.push(&s[i] * &s[i] - C::one());
    }
    for i in 0..m {
        for j in (i + 1)..m {
            if j == i + 1 {
                rels.push(&s[i] * &s[j] * &s[i] - &s[j] * &s[i] * &s[j]);
            } else {
                rels.push(Polynomial::commutator(&s[j], &s[i]));
            }
        }
    }
    (alg, rels)
}

/// The word `x_{i_1} x_{i_2} …` as a monomial, for building relations by hand.
pub fn word<C: Coeff>(alg: &FreeAlgebra, letters: &[Var]) -> Polynomial<C> {
    alg.monomial(Word::from_slice(letters))
}
