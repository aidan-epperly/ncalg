//! GNS extraction: a finite-dimensional representation and state from a
//! (numerically) flat moment matrix.
//!
//! Let `L` be the functional returned by a moment relaxation with index words
//! `W_d` (degree ≤ d) and let `B = W_{d−1}`.  The Gram matrix
//! `G[u, v] = L(u* v)` (`u, v ∈ B`) is PSD; write `G = U Λ U†` with the `r`
//! positive eigenvalues kept.  Then the vectors `|u⟩` live in `C^r` with
//! coordinates `ψ(u) = Λ^{1/2} U† e_u`, and if the moment matrix is *flat*
//! (`rank M_d = rank M_{d−1}`, so `|x u⟩ ∈ span{|v⟩ : v ∈ B}`) the operators
//!
//! ```text
//!   π(x) = Λ^{-1/2} U† G_x U Λ^{-1/2},      G_x[u', u] = L(u'* x u)
//! ```
//!
//! form a `*`-representation of the quotient algebra on `C^r` and the vector
//! `ψ(1)` reproduces every moment: `L(w) = ⟨ψ(1)| π(w) |ψ(1)⟩` (Pironio,
//! Navascués, Acín 2010; Curto–Fialkow / Henrion–Lasserre in the commutative
//! case, where `r` points are recovered).  For a tight relaxation this
//! recovers the optimal state or representation.  When the moment matrix is
//! not flat the construction still returns the best approximation, and the
//! defects ([`GnsRepresentation::flatness_defect`],
//! [`relation_defect`](GnsRepresentation::relation_defect),
//! [`moment_defect`](GnsRepresentation::moment_defect)) show how far it is
//! from an actual representation.
//!
//! Flatness is a property of the particular optimal functional: an
//! interior-point solver returns the analytic center of the optimal face,
//! which is flat when the optimal moment matrix is unique (always the case
//! for finite-dimensional quotient algebras whose index set spans them,
//! e.g. Pauli or Majorana algebras at level 2), but not necessarily when the
//! optimum leaves higher moments free.

use crate::algebra::FreeAlgebra;
use crate::dense::hermitian_eigen;
use crate::ncsos::{MomentSdp, SdpScalar};
use crate::poly::Polynomial;
use crate::reduce::Reducer;
use crate::word::{Var, Word};
use faer::Mat;
use num_complex::Complex;
use std::collections::HashMap;

type C64 = Complex<f64>;

/// Dense complex matrix as rows.
pub type DenseMatrix = Vec<Vec<C64>>;

/// A finite-dimensional `*`-representation with a cyclic state, extracted
/// from moments (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct GnsRepresentation {
    alg: FreeAlgebra,
    /// Dimension `r` of the representation space.
    pub dimension: usize,
    /// The words whose vectors span the space (`B = W_{d−1}`).
    pub basis_words: Vec<Word>,
    /// `π(x_v)` for every generator `v`, `r × r`.
    pub generators: Vec<DenseMatrix>,
    /// Coordinates of the cyclic vector `|1⟩`.
    pub state: Vec<C64>,
    /// Eigenvalues of the Gram matrix that were kept (descending), after
    /// the Jacobi scaling `D^{-1/2} G D^{-1/2}` (`D = diag G`).
    pub kept_eigenvalues: Vec<f64>,
    /// Eigenvalues that were dropped as zero.
    pub dropped_eigenvalues: Vec<f64>,
    /// Largest relative distance of a vector `|x u⟩` (`u ∈ B`) from the
    /// span of the basis vectors, and largest mismatch between `L(u* x* x u)`
    /// and `‖ψ(x u)‖²`: zero for a flat moment matrix.
    pub flatness_defect: f64,
}

impl GnsRepresentation {
    /// Extract a representation from moments.  `moments` gives `L` on the
    /// normal words that occur (as [`MomentSdp::moments`] returns them),
    /// `red` reduces products to those words, `basis` is `B` (it must
    /// contain the empty word), and eigenvalues of the Jacobi-scaled Gram
    /// matrix (unit diagonal) below `tol · λ_max` are treated as zero.
    pub fn extract<C: SdpScalar>(
        red: &Reducer<C>,
        moments: &[(Word, C64)],
        basis: &[Word],
        tol: f64,
    ) -> Result<Self, String> {
        let alg = red.algebra().clone();
        let table: HashMap<Vec<Var>, C64> =
            moments.iter().map(|(w, v)| (w.as_slice().to_vec(), *v)).collect();
        let value = |p: &Polynomial<C>| -> Result<C64, String> {
            let nf = red.reduce(p);
            let mut acc = C64::new(0.0, 0.0);
            for (w, c) in nf.iter() {
                let m = table.get(w.as_slice()).ok_or_else(|| {
                    format!(
                        "the moment of {} is not available (enlarge the index set)",
                        alg.format_word(w)
                    )
                })?;
                let (re, im) = c.to_complex_f64();
                acc += C64::new(re, im) * m;
            }
            Ok(acc)
        };
        let l_word = |w: Word| -> Result<C64, String> { value(&alg.monomial::<C>(w)) };
        let n = basis.len();
        if n == 0 {
            return Err("empty basis".to_string());
        }
        let one_index = basis
            .iter()
            .position(|w| w.is_empty())
            .ok_or_else(|| "the basis must contain the empty word".to_string())?;
        // Gram matrix G[u, v] = L(u* v)
        let rows: Vec<Vec<C64>> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|u| {
                    let ua = alg.adjoint_word(&basis[u]);
                    (0..n)
                        .map(|v| l_word(ua.concat(&basis[v])))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<_, _>>()?
        };
        let mut g = Mat::<C64>::from_fn(n, n, |u, v| rows[u][v]);
        // Hermitian part (guards against solver round-off), and the Jacobi
        // scaling D^{-1/2} G D^{-1/2}, D = diag G: the rank decision is then
        // invariant under rescaling the variables.  Unscaled, x ↦ 10³ x
        // spreads the entries over 10¹², and the 3-point measure on {1, x, x²}
        // came out 2-dimensional at scales 10⁻³ and 10³.  All Gram
        // columns below are scaled alike, G = D^{1/2} U Λ U† D^{1/2}, so
        // ψ(q) = Λ^{-1/2} U† D^{-1/2} g_q is the same construction.
        let dinv: Vec<f64> = (0..n)
            .map(|u| {
                let d = g[(u, u)].re;
                if d > 0.0 {
                    1.0 / d.sqrt()
                } else {
                    1.0
                }
            })
            .collect();
        for u in 0..n {
            for v in u..n {
                let h = (g[(u, v)] + g[(v, u)].conj()) * (0.5 * dinv[u] * dinv[v]);
                g[(u, v)] = h;
                g[(v, u)] = h.conj();
            }
        }
        let (vals, vecs) = hermitian_eigen(&g);
        let max = vals.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        if max == 0.0 {
            return Err("the Gram matrix is zero".to_string());
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| vals[*b].partial_cmp(&vals[*a]).unwrap());
        let kept: Vec<usize> = order.iter().cloned().filter(|&k| vals[k] > tol * max).collect();
        let dropped: Vec<f64> = order
            .iter()
            .filter(|k| vals[**k] <= tol * max)
            .map(|k| vals[*k])
            .collect();
        let r = kept.len();
        // U_r (n × r), Λ^{±1/2}
        let ur = |i: usize, k: usize| vecs[(i, kept[k])];
        let sqrt_l: Vec<f64> = kept.iter().map(|k| vals[*k].sqrt()).collect();
        // ψ(q) for a vector q given by its Gram column g_q[u'] = L(u'* q): Λ^{-1/2} U_r† g_q
        let psi_of = |gq: &[C64]| -> Vec<C64> {
            (0..r)
                .map(|k| {
                    let mut s = C64::new(0.0, 0.0);
                    for i in 0..n {
                        s += ur(i, k).conj() * gq[i];
                    }
                    s / sqrt_l[k]
                })
                .collect()
        };
        // state ψ(1) = Λ^{1/2} U_r† e_1 = Λ^{-1/2} U_r† g_1 with g_1 = G e_1 (same thing)
        // (g holds D^{-1/2} G D^{-1/2}; ψ needs D^{-1/2} G e_1)
        let g_one: Vec<C64> = (0..n).map(|i| g[(i, one_index)] / dinv[one_index]).collect();
        let state = psi_of(&g_one);
        // generators π(x) = Λ^{-1/2} U† G_x U Λ^{-1/2}
        // per generator in parallel: n² + n reductions each, (1 + #vars) n²
        // in all, which is seconds for Pauli chains of 64 sites at level 2
        use rayon::prelude::*;
        let per_var = |x: usize| -> Result<(DenseMatrix, f64), String> {
            let mut flatness_defect = 0.0_f64;
            let mut gx = Mat::<C64>::zeros(n, n);
            for u in 0..n {
                // column u: g_q for q = x u
                let q = Word::from_vec({
                    let mut v = vec![x as Var];
                    v.extend_from_slice(basis[u].as_slice());
                    v
                });
                let mut gq = vec![C64::new(0.0, 0.0); n];
                for up in 0..n {
                    let upa = alg.adjoint_word(&basis[up]);
                    gq[up] = l_word(upa.concat(&q))? * dinv[up];
                    gx[(up, u)] = gq[up] * dinv[u];
                }
                // flatness: g_q must lie in the range of G, and ‖ψ(q)‖² = L(q* q)
                let norm_gq = gq.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if norm_gq > 0.0 {
                    let mut proj = vec![C64::new(0.0, 0.0); n];
                    for k in 0..r {
                        let mut s = C64::new(0.0, 0.0);
                        for i in 0..n {
                            s += ur(i, k).conj() * gq[i];
                        }
                        for i in 0..n {
                            proj[i] += ur(i, k) * s;
                        }
                    }
                    let res = gq
                        .iter()
                        .zip(&proj)
                        .map(|(a, b)| (a - b).norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    flatness_defect = flatness_defect.max(res / norm_gq);
                }
                let psi_q = psi_of(&gq);
                let qa = alg.adjoint_word(&q);
                let lqq = l_word(qa.concat(&q))?;
                let nq: f64 = psi_q.iter().map(|c| c.norm_sqr()).sum();
                flatness_defect = flatness_defect.max((lqq.re - nq).abs() / (1.0 + lqq.re.abs()));
            }
            // π(x)[k, l] = Σ_{u', u} conj(U[u', k]) G_x[u', u] U[u, l] / (sqrt λ_k sqrt λ_l)
            let mut pi = vec![vec![C64::new(0.0, 0.0); r]; r];
            // T = G_x U_r  (n × r)
            let mut t = vec![vec![C64::new(0.0, 0.0); r]; n];
            for up in 0..n {
                for l in 0..r {
                    let mut s = C64::new(0.0, 0.0);
                    for u in 0..n {
                        s += gx[(up, u)] * ur(u, l);
                    }
                    t[up][l] = s;
                }
            }
            for k in 0..r {
                for l in 0..r {
                    let mut s = C64::new(0.0, 0.0);
                    for up in 0..n {
                        s += ur(up, k).conj() * t[up][l];
                    }
                    pi[k][l] = s / (sqrt_l[k] * sqrt_l[l]);
                }
            }
            Ok((pi, flatness_defect))
        };
        let per_x: Vec<(DenseMatrix, f64)> = (0..alg.num_vars())
            .into_par_iter()
            .map(per_var)
            .collect::<Result<_, _>>()?;
        let flatness_defect = per_x.iter().map(|(_, d)| *d).fold(0.0_f64, f64::max);
        let generators: Vec<DenseMatrix> = per_x.into_iter().map(|(pi, _)| pi).collect();
        Ok(GnsRepresentation {
            alg,
            dimension: r,
            basis_words: basis.to_vec(),
            generators,
            state,
            kept_eigenvalues: kept.iter().map(|k| vals[*k]).collect(),
            dropped_eigenvalues: dropped,
            flatness_defect,
        })
    }

    /// The algebra.
    pub fn algebra(&self) -> &FreeAlgebra {
        &self.alg
    }

    /// `π(w)` for a word.
    pub fn word_matrix(&self, w: &[Var]) -> DenseMatrix {
        let r = self.dimension;
        let mut m = Mat::<C64>::identity(r, r);
        for &x in w {
            let g = &self.generators[x as usize];
            m = crate::dense::matmul(m.as_ref(), Mat::from_fn(r, r, |i, j| g[i][j]).as_ref());
        }
        (0..r).map(|i| (0..r).map(|j| m[(i, j)]).collect()).collect()
    }

    /// `π(p)` for a polynomial.
    pub fn evaluate<C: SdpScalar>(&self, p: &Polynomial<C>) -> DenseMatrix {
        let r = self.dimension;
        let mut m = vec![vec![C64::new(0.0, 0.0); r]; r];
        for (w, c) in p.iter() {
            let (re, im) = c.to_complex_f64();
            let coef = C64::new(re, im);
            let wm = self.word_matrix(w.as_slice());
            for i in 0..r {
                for j in 0..r {
                    m[i][j] += coef * wm[i][j];
                }
            }
        }
        m
    }

    /// `π(w) v`, letter by letter from the right: `|w| r²` work instead of
    /// the `|w| r³` of [`word_matrix`](Self::word_matrix).
    fn apply_word(&self, w: &[Var], v: &[C64]) -> Vec<C64> {
        let mut v = v.to_vec();
        let mut next = vec![C64::new(0.0, 0.0); v.len()];
        for &x in w.iter().rev() {
            let g = &self.generators[x as usize];
            for (out, row) in next.iter_mut().zip(g) {
                *out = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            }
            std::mem::swap(&mut v, &mut next);
        }
        v
    }

    /// `⟨ψ(1)| π(w) |ψ(1)⟩`.
    fn word_expectation(&self, w: &[Var]) -> C64 {
        self.state
            .iter()
            .zip(self.apply_word(w, &self.state))
            .map(|(a, b)| a.conj() * b)
            .sum()
    }

    /// `⟨ψ(1)| π(p) |ψ(1)⟩`.
    pub fn expectation<C: SdpScalar>(&self, p: &Polynomial<C>) -> C64 {
        p.iter()
            .map(|(w, c)| {
                let (re, im) = c.to_complex_f64();
                C64::new(re, im) * self.word_expectation(w.as_slice())
            })
            .sum()
    }

    /// Largest Frobenius norm of `π(g)` over the given relations (zero for an
    /// exact representation of the quotient algebra).
    pub fn relation_defect<C: SdpScalar>(&self, relations: &[Polynomial<C>]) -> f64 {
        relations
            .iter()
            .map(|g| {
                let m = self.evaluate(g);
                Mat::from_fn(m.len(), m.len(), |i, j| m[i][j]).norm_l2()
            })
            .fold(0.0, f64::max)
    }

    /// Largest `|L(w) − ⟨ψ| π(w) |ψ⟩|` over the given moments.
    ///
    /// In parallel, by matrix-vector products: a dense `r × r` product per
    /// letter and moment made this `#moments · |w| · r³`.
    pub fn moment_defect(&self, moments: &[(Word, C64)]) -> f64 {
        use rayon::prelude::*;
        moments
            .par_iter()
            .map(|(w, v)| (self.word_expectation(w.as_slice()) - v).norm())
            .reduce(|| 0.0, f64::max)
    }

    /// Largest `‖π(x†) − π(x)†‖_F` over the generators (the representation
    /// must be a `*`-representation).
    pub fn adjoint_defect(&self) -> f64 {
        let mut d = 0.0_f64;
        for x in 0..self.generators.len() {
            let xa = self.alg.adjoint_var(x as Var) as usize;
            let a = &self.generators[x];
            let b = &self.generators[xa];
            let r = self.dimension;
            let mut s = 0.0;
            for i in 0..r {
                for j in 0..r {
                    s += (b[i][j] - a[j][i].conj()).norm_sqr();
                }
            }
            d = d.max(s.sqrt());
        }
        d
    }

    /// Are all generator matrices real (up to `tol`)?
    pub fn is_real(&self, tol: f64) -> bool {
        self.generators
            .iter()
            .all(|m| m.iter().all(|row| row.iter().all(|c| c.im.abs() <= tol)))
            && self.state.iter().all(|c| c.im.abs() <= tol)
    }
}

impl MomentSdp {
    /// GNS extraction from a solution `z` of this relaxation.  The basis `B`
    /// is the set of index words `u` whose left multiples `x u` reduce into
    /// the span of the index set for every generator `x` (for a level-`d`
    /// index set this is `W_{d−1}`, or all of `W` when the quotient algebra
    /// is finite dimensional and `W` already spans it), so that every moment
    /// `L(u'* x u)` the construction needs is an entry of the moment matrix.
    /// `red` must be the reducer the relaxation was built with.
    pub fn gns<C: SdpScalar>(
        &self,
        z: &[f64],
        red: &Reducer<C>,
        tol: f64,
    ) -> Result<GnsRepresentation, String> {
        let words = &self.source_words()[0];
        let index: std::collections::HashSet<Vec<Var>> =
            words.iter().map(|w| w.as_slice().to_vec()).collect();
        let alg = red.algebra();
        let basis: Vec<Word> = words
            .iter()
            .filter(|u| {
                (0..alg.num_vars()).all(|x| {
                    let mut v = vec![x as Var];
                    v.extend_from_slice(u.as_slice());
                    let nf = red.reduce(&alg.monomial::<C>(Word::from_vec(v)));
                    let closed = nf.iter().all(|(w, _)| index.contains(w.as_slice()));
                    closed
                })
            })
            .cloned()
            .collect();
        if basis.is_empty() {
            return Err("no index word has all its left multiples inside the index set".to_string());
        }
        let moments = self.moments(z);
        GnsRepresentation::extract(red, &moments, &basis, tol)
    }
}
