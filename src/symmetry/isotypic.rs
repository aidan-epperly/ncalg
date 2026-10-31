//! Isotypic decomposition of a finite group representation, numerically:
//! the block-diagonalization behind symmetry-reduced moment relaxations and
//! Gram forms.

use crate::dense::{adjoint_mul, hermitian_eigen, hermitian_fn, mm, mm3};
use faer::Mat;
use num_complex::Complex;
use std::collections::BTreeMap;

type C64 = Complex<f64>;

/// One isotypic component: `vectors` (in the original coordinates) are a basis
/// of the span `W` of the first basis vectors of the `m_λ` copies of the
/// irreducible, aligned by intertwiners.  An invariant Hermitian form `M`
/// restricted to the component is `[v_k^† M v_l]` repeated `irrep_dimension`
/// times: up to unitary equivalence when the representation is unitary (the
/// vectors are then orthonormal), up to congruence otherwise.
#[derive(Clone)]
pub(crate) struct IsotypicPart {
    pub(crate) vectors: Vec<Vec<C64>>,
    pub(crate) irrep_dimension: usize,
}

/// A basis of the column space of the rank-`rank` projector `q`, chosen
/// greedily from the sparsest columns, with a Gram–Schmidt independence test
/// (modified Gram–Schmidt run twice, so the orthonormal basis is orthogonal
/// to working precision whatever the conditioning of the columns).
///
/// With `orthonormal` the result is the orthonormalized basis (Gram–Schmidt
/// only mixes columns with overlapping supports, so it stays sparse);
/// otherwise the chosen columns themselves.
fn sparse_basis(q: &Mat<C64>, rank: usize, orthonormal: bool) -> Vec<Vec<C64>> {
    let m = q.nrows();
    // entries at rounding level relative to the projector
    let qmax = (0..m)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .fold(0.0_f64, |mx, (i, j)| mx.max(q[(i, j)].norm()));
    let tiny = 1e-10 * qmax;
    let mut cols: Vec<(usize, usize)> = (0..m)
        .map(|j| ((0..m).filter(|&i| q[(i, j)].norm() > tiny).count(), j))
        .filter(|(nnz, _)| *nnz > 0)
        .collect();
    cols.sort();
    let mut ortho: Vec<Vec<C64>> = Vec::new(); // orthonormal basis of the chosen span
    let mut chosen: Vec<Vec<C64>> = Vec::new();
    for (_, j) in cols {
        if chosen.len() == rank {
            break;
        }
        let col: Vec<C64> = (0..m)
            .map(|i| {
                if q[(i, j)].norm() > tiny {
                    q[(i, j)]
                } else {
                    C64::new(0.0, 0.0)
                }
            })
            .collect();
        let norm0: f64 = col.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        let mut r = col.clone();
        for _pass in 0..2 {
            for o in &ortho {
                let dot: C64 = o.iter().zip(&r).map(|(a, b)| a.conj() * b).sum();
                if dot.re == 0.0 && dot.im == 0.0 {
                    continue; // disjoint supports (columns from different word orbits)
                }
                for (x, y) in r.iter_mut().zip(o) {
                    *x -= dot * y;
                }
            }
        }
        let norm: f64 = r.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-7 * norm0.max(1e-300) {
            ortho.push(r.iter().map(|c| c / norm).collect());
            chosen.push(col);
        }
    }
    assert_eq!(
        chosen.len(),
        rank,
        "could not find a sparse basis of the isotypic subspace"
    );
    if orthonormal {
        ortho
    } else {
        chosen
    }
}

/// `max |V^† V − I|` over the entries.
pub(crate) fn orthonormality_error(vectors: &[Vec<C64>]) -> f64 {
    let mut err: f64 = 0.0;
    for (k, u) in vectors.iter().enumerate() {
        for (l, v) in vectors.iter().enumerate().skip(k) {
            let dot: C64 = u.iter().zip(v).map(|(a, b)| a.conj() * b).sum();
            let target = if k == l { 1.0 } else { 0.0 };
            err = err.max((dot - C64::new(target, 0.0)).norm());
        }
    }
    err
}

/// Sparse columns of a matrix: `cols[j] = [(i, ρ_ij)]`.
pub(crate) type SparseCols = Vec<Vec<(usize, C64)>>;

/// A representation of a finite group on `C^m`, one entry per group element:
/// dense matrices, sparse columns, or unitary monomial forms
/// `ρ(g) e_i = coef[i] e_{perm[i]}` (signed or phased permutations), which
/// need no `m × m` storage.
#[derive(Clone, Copy)]
pub(crate) enum Rep<'a> {
    #[cfg_attr(not(test), allow(dead_code))]
    Dense(&'a [Mat<C64>]),
    Sparse(&'a [SparseCols]),
    Monomial(&'a [(Vec<usize>, Vec<C64>)]),
}

impl Rep<'_> {
    fn len(&self) -> usize {
        match self {
            Rep::Dense(r) => r.len(),
            Rep::Sparse(r) => r.len(),
            Rep::Monomial(f) => f.len(),
        }
    }

    /// `tr ρ(g)` for every group element.
    fn traces(&self) -> Vec<C64> {
        match self {
            Rep::Dense(r) => r
                .iter()
                .map(|rho| (0..rho.nrows()).map(|i| rho[(i, i)]).sum())
                .collect(),
            Rep::Sparse(r) => r
                .iter()
                .map(|cols| {
                    cols.iter()
                        .enumerate()
                        .flat_map(|(j, col)| col.iter().filter(move |(i, _)| *i == j).map(|(_, c)| *c))
                        .sum()
                })
                .collect(),
            Rep::Monomial(f) => f
                .iter()
                .map(|(perm, coef)| (0..perm.len()).filter(|&i| perm[i] == i).map(|i| coef[i]).sum())
                .collect(),
        }
    }
}

/// Isotypic decomposition of the representation `rep` (all group elements)
/// on `C^m`, following the commutant-based numerical method: make the
/// representation unitary, diagonalize a generic invariant Hermitian form,
/// match eigenspaces with a second one, align copies.  For monomial actions
/// with a known group, [`monomial_isotypic_decomposition`] is exact and much
/// cheaper.
pub(crate) fn isotypic_decomposition(rep: Rep<'_>, m: usize) -> Vec<IsotypicPart> {
    // The decomposition is a deterministic function of the matrices, and the
    // same representation recurs whenever relaxations with the same index
    // words and group are built (the functionals of a MultiRelaxation, a
    // coverage pass before the real build), so results are memoised on the
    // exact bit pattern of the input.
    let key = representation_key(rep, m);
    if let Some(parts) = ISOTYPIC_CACHE
        .lock()
        .ok()
        .and_then(|c| c.iter().find(|(k, _)| *k == key).map(|(_, p)| p.clone()))
    {
        return parts;
    }
    let parts = isotypic_decomposition_uncached(rep, m);
    if let Ok(mut c) = ISOTYPIC_CACHE.lock() {
        if c.len() >= ISOTYPIC_CACHE_ENTRIES {
            c.remove(0);
        }
        c.push((key, parts.clone()));
    }
    parts
}

/// Recently computed isotypic decompositions (oldest first).
static ISOTYPIC_CACHE: std::sync::Mutex<Vec<(Vec<u64>, Vec<IsotypicPart>)>> =
    std::sync::Mutex::new(Vec::new());
const ISOTYPIC_CACHE_ENTRIES: usize = 8;

/// The nonzero entries of every matrix, bit for bit, as the cache key.
fn representation_key(rep: Rep<'_>, m: usize) -> Vec<u64> {
    let mut key = vec![m as u64, rep.len() as u64];
    match rep {
        Rep::Dense(rhos) => {
            key.push(0);
            for rho in rhos {
                for j in 0..m {
                    for i in 0..m {
                        let v = rho[(i, j)];
                        if v.re != 0.0 || v.im != 0.0 {
                            key.extend([(i * m + j) as u64, v.re.to_bits(), v.im.to_bits()]);
                        }
                    }
                }
                key.push(u64::MAX);
            }
        }
        Rep::Sparse(rhos) => {
            key.push(2);
            for cols in rhos {
                for (j, col) in cols.iter().enumerate() {
                    for (i, v) in col {
                        key.extend([(i * m + j) as u64, v.re.to_bits(), v.im.to_bits()]);
                    }
                }
                key.push(u64::MAX);
            }
        }
        Rep::Monomial(forms) => {
            key.push(1);
            for (perm, coef) in forms {
                for (p, c) in perm.iter().zip(coef) {
                    key.extend([*p as u64, c.re.to_bits(), c.im.to_bits()]);
                }
                key.push(u64::MAX);
            }
        }
    }
    key
}

/// The nonzero entries of a dense matrix, by column.
fn sparse_cols(rho: &Mat<C64>) -> SparseCols {
    (0..rho.ncols())
        .map(|j| {
            (0..rho.nrows())
                .filter(|&i| rho[(i, j)].re != 0.0 || rho[(i, j)].im != 0.0)
                .map(|i| (i, rho[(i, j)]))
                .collect()
        })
        .collect()
}

/// `ρ e_i = c_i e_{π(i)}` with `|c_i| = 1`: `(π, c)` if `rho` is such a
/// unitary monomial matrix (a signed or phased permutation).
fn monomial_form(rho: &SparseCols) -> Option<(Vec<usize>, Vec<C64>)> {
    let m = rho.len();
    let mut perm = Vec::with_capacity(m);
    let mut coef = Vec::with_capacity(m);
    let mut hit = vec![false; m];
    for col in rho {
        let (i, v) = match col.as_slice() {
            [(i, v)] => (*i, *v),
            _ => return None,
        };
        if hit[i] || (v.norm() - 1.0).abs() > 1e-12 {
            return None;
        }
        hit[i] = true;
        perm.push(i);
        coef.push(v);
    }
    Some((perm, coef))
}

fn isotypic_decomposition_uncached(rep: Rep<'_>, m: usize) -> Vec<IsotypicPart> {
    let order = rep.len() as f64;
    // a representation by unitary monomial matrices (every symmetry acting by
    // signed permutations of words) is already unitary: S = I below, and its
    // Reynolds operator is a permutation of entries
    // the matrices as sparse columns: an action on words maps each word to a
    // few others, so a group element has O(m) entries, not m²
    let converted: Vec<SparseCols>;
    let rhos: &[SparseCols] = match rep {
        Rep::Monomial(_) => &[],
        Rep::Sparse(r) => r,
        Rep::Dense(r) => {
            converted = r.iter().map(sparse_cols).collect();
            &converted
        }
    };
    let monomial: Option<Vec<(Vec<usize>, Vec<C64>)>> = match rep {
        Rep::Monomial(f) => Some(f.to_vec()),
        _ => rhos.iter().map(monomial_form).collect(),
    };
    // invariant inner product S = (1/|G|) Σ ρ^† ρ ; unitarise with R = S^{1/2}
    let (r, r_inv) = if monomial.is_some() {
        (None, None)
    } else {
        // S[i, j] = (1/|G|) Σ_g Σ_k conj(ρ_ki) ρ_kj, from the rows of each ρ
        let mut s = Mat::<C64>::zeros(m, m);
        for cols in rhos {
            let mut rows: Vec<Vec<(usize, C64)>> = vec![Vec::new(); m];
            for (j, col) in cols.iter().enumerate() {
                for &(k, c) in col {
                    rows[k].push((j, c));
                }
            }
            for row in &rows {
                for &(i, a) in row {
                    for &(j, b) in row {
                        s[(i, j)] += a.conj() * b;
                    }
                }
            }
        }
        s *= faer::Scale(C64::new(1.0 / order, 0.0));
        let r = hermitian_fn(&s, |x| x.max(0.0).sqrt());
        let r_inv = hermitian_fn(&s, |x| 1.0 / x.max(1e-300).sqrt());
        (Some(r), Some(r_inv))
    };
    // two random Hermitian matrices and their Reynolds projections onto the commutant
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64) - 0.5
    };
    let random_hermitian = |rnd: &mut dyn FnMut() -> f64| -> Mat<C64> {
        let mut h = Mat::<C64>::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let v = if i == j {
                    C64::new(rnd(), 0.0)
                } else {
                    C64::new(rnd(), rnd())
                };
                h[(i, j)] = v;
                h[(j, i)] = v.conj();
            }
        }
        h
    };
    let reynolds = |h: &Mat<C64>| -> Mat<C64> {
        let mut out = Mat::<C64>::zeros(m, m);
        if let Some(mono) = &monomial {
            // (ρ^† H ρ)[i, j] = conj(c_i) H[π(i), π(j)] c_j
            for (perm, coef) in mono {
                for j in 0..m {
                    let (pj, cj) = (perm[j], coef[j]);
                    for i in 0..m {
                        out[(i, j)] += coef[i].conj() * h[(perm[i], pj)] * cj;
                    }
                }
            }
        } else {
            // with u_g = R ρ_g R⁻¹ (R Hermitian):
            //   Σ_g u_g† H u_g = R⁻¹ (Σ_g ρ_g† (R H R) ρ_g) R⁻¹,
            // four dense products in all, the sum by sparse columns in
            // O(|G| m² nnz/m) (it was 2 |G| dense products: 6.9 s of a QMC
            // 11-cycle build with |G| = 22 and m = 1211)
            use rayon::prelude::*;
            let (r, r_inv) = (r.as_ref().expect("R"), r_inv.as_ref().expect("R⁻¹"));
            let k = mm3(r, h, r);
            let sum = rhos
                .par_iter()
                .fold(
                    || Mat::<C64>::zeros(m, m),
                    |mut acc, cols| {
                        // K ρ, column by column
                        let mut kr = Mat::<C64>::zeros(m, m);
                        for (j, col) in cols.iter().enumerate() {
                            for &(i, c) in col {
                                for a in 0..m {
                                    kr[(a, j)] += k[(a, i)] * c;
                                }
                            }
                        }
                        // ρ† (K ρ): row i of the product is Σ_k conj(ρ_ki) (Kρ)[k, :]
                        for (i, col) in cols.iter().enumerate() {
                            for &(kk, c) in col {
                                let cc = c.conj();
                                for b in 0..m {
                                    acc[(i, b)] += cc * kr[(kk, b)];
                                }
                            }
                        }
                        acc
                    },
                )
                .reduce(|| Mat::<C64>::zeros(m, m), |a, b| a + b);
            out = mm3(r_inv, &sum, r_inv);
        }
        out *= faer::Scale(C64::new(1.0 / order, 0.0));
        // symmetrize against round-off
        let adj = out.adjoint().to_owned();
        (&out + &adj) * faer::Scale(C64::new(0.5, 0.0))
    };
    let x = reynolds(&random_hermitian(&mut rnd));
    let y = reynolds(&random_hermitian(&mut rnd));
    let (vals, vecs) = hermitian_eigen(&x);
    // cluster eigenvalues
    let spread = (vals[m - 1] - vals[0]).abs().max(1e-12);
    let ctol = 1e-7 * spread;
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..m {
        match clusters.last_mut() {
            Some(c) if (vals[i] - vals[*c.last().unwrap()]).abs() <= ctol => c.push(i),
            _ => clusters.push(vec![i]),
        }
    }
    // all blocks E_i^† Y E_j at once: the clusters are consecutive columns of
    // the eigenvector matrix, so they are sub-blocks of V^† Y V
    let mut vecs = vecs;
    let mut vyv = adjoint_mul(vecs.as_ref(), mm(&y, &vecs).as_ref());
    // Schur's lemma: Y compressed onto one irreducible copy is a multiple of
    // the identity.  Eigenvalues of X from different copies are independent
    // (they do not repel), so with m in the thousands two of them can fall
    // within the clustering tolerance by accident (a 1- and a
    // 2-dimensional copy of a D5 representation of dimension 1504, 8.5e-8 of
    // the spread apart, read as a 3-dimensional irreducible).  A cluster on
    // which Y is not scalar is split along the eigenvectors of Y's
    // compression, with the same relative tolerance.
    let yscale = {
        let d: Vec<f64> = (0..m).map(|i| vyv[(i, i)].re).collect();
        let hi = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = d.iter().cloned().fold(f64::INFINITY, f64::min);
        (hi - lo).abs().max(1e-12)
    };
    let ytol = 1e-7 * yscale;
    let mut split = false;
    let mut refined: Vec<Vec<usize>> = Vec::with_capacity(clusters.len());
    for c in &clusters {
        let s = c.len();
        if s == 1 {
            refined.push(c.clone());
            continue;
        }
        let b = Mat::<C64>::from_fn(s, s, |a, bb| vyv[(c[a], c[bb])]);
        let (bvals, bvecs) = hermitian_eigen(&b);
        if bvals[s - 1] - bvals[0] <= ytol {
            refined.push(c.clone());
            continue;
        }
        split = true;
        // rotate the cluster's columns to the eigenvectors of the compression
        let ec = Mat::<C64>::from_fn(m, s, |i, j| vecs[(i, c[j])]);
        let rotated = mm(&ec, &bvecs);
        for (j, &col) in c.iter().enumerate() {
            for i in 0..m {
                vecs[(i, col)] = rotated[(i, j)];
            }
        }
        let mut sub: Vec<usize> = vec![c[0]];
        for j in 1..s {
            if bvals[j] - bvals[j - 1] <= ytol {
                sub.push(c[j]);
            } else {
                refined.push(std::mem::take(&mut sub));
                sub.push(c[j]);
            }
        }
        refined.push(sub);
    }
    if split {
        clusters = refined;
        vyv = adjoint_mul(vecs.as_ref(), mm(&y, &vecs).as_ref());
    }
    let bases: Vec<Mat<C64>> = clusters
        .iter()
        .map(|c| Mat::from_fn(m, c.len(), |i, j| vecs[(i, c[j])]))
        .collect();
    let block = |i: usize, j: usize| -> Mat<C64> {
        let (ci, cj) = (&clusters[i], &clusters[j]);
        Mat::from_fn(ci.len(), cj.len(), |a, b| vyv[(ci[a], cj[b])])
    };
    // group clusters into irreducible types via the intertwining test with y
    let nc = clusters.len();
    let mut parent: Vec<usize> = (0..nc).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        if p[i] != i {
            let r = find(p, p[i]);
            p[i] = r;
        }
        p[i]
    }
    for i in 0..nc {
        for j in (i + 1)..nc {
            if clusters[i].len() != clusters[j].len() {
                continue;
            }
            // relative to the scale of Y
            if block(i, j).norm_l2() > 1e-6 * yscale {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..nc {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(i);
    }
    let mut parts = Vec::new();
    for (_, members) in groups {
        let d = clusters[members[0]].len();
        assert!(
            members.iter().all(|&i| clusters[i].len() == d),
            "isotypic decomposition failed (inconsistent eigenvalue multiplicities); disable block_diagonalize"
        );
        let u1 = &bases[members[0]];
        let mut aligned: Vec<Mat<C64>> = Vec::new();
        for &k in &members {
            // intertwiner E_1 -> E_k from y, made unitary by polar decomposition
            let uk = if k == members[0] {
                u1.to_owned()
            } else {
                let w = block(k, members[0]); // E_k^† Y E_1, d x d
                let wtw = adjoint_mul(w.as_ref(), w.as_ref());
                let inv_sqrt = hermitian_fn(&wtw, |x| 1.0 / x.max(1e-300).sqrt());
                let w_unitary = mm(&w, &inv_sqrt);
                mm(&bases[k], &w_unitary)
            };
            // first aligned basis vector of this copy (unitary coordinates)
            aligned.push(Mat::from_fn(m, 1, |i, _| uk[(i, 0)]));
        }
        // The reduced block depends on the subspace W spanned by these
        // vectors and on the basis of W: an orthonormal basis gives a block
        // unitarily equivalent to the restriction of F, any
        // other basis only a congruent one.  Congruence preserves F ⪰ 0 but
        // not I − F ⪰ 0, traces or norms, which callers add on top of the
        // reduced blocks.  Take the sparsest basis available: columns of the
        // projector onto W in the original coordinates,
        // Q = R^{-1} (Σ_k c_k c_k^†) R.
        //
        // Monomial representations (symmetries acting by signed permutations
        // of words) are unitary, Q is an orthogonal projector, and its columns
        // are orthonormalized; columns from different word orbits have
        // disjoint supports, so the basis stays sparse.  For a non-unitary
        // representation the group acts on F by congruence, I − F is not
        // invariant to begin with, and Q is oblique; there the columns are
        // kept as they are (the block is congruent to the restriction).
        //
        // Q has rank `mult`: with C = [c_1 … c_mult] it is (R⁻¹ C)(C† R),
        // m² mult work (dense m × m products per copy and per part were
        // 3.2 s of a QMC 11-cycle build, m = 1211).
        let mult = aligned.len();
        let c = Mat::<C64>::from_fn(m, mult, |i, k| aligned[k][(i, 0)]);
        let q = match (&r_inv, &r) {
            (Some(r_inv), Some(r)) => mm(&mm(r_inv, &c), &adjoint_mul(c.as_ref(), r.as_ref())),
            _ => mm(&c, &c.adjoint().to_owned()),
        };
        let unitary_rep = r.is_none();
        let mut vectors = sparse_basis(&q, mult, unitary_rep);
        if unitary_rep && orthonormality_error(&vectors) > 1e-12 {
            // cannot happen with two Gram–Schmidt passes on numerically
            // independent columns; the aligned vectors are orthonormal (dense)
            vectors = aligned
                .iter()
                .map(|c| (0..m).map(|i| c[(i, 0)]).collect())
                .collect();
        }
        parts.push(IsotypicPart {
            vectors,
            irrep_dimension: d,
        });
    }

    // sanity: Σ_λ d_λ m_λ = m, and the commutant dimension Σ_λ m_λ² equals the
    // character count (1/|G|) Σ_g |tr ρ(g)|² (cheap and exact); a mismatch means
    // the numerics failed, so fall back to the undecomposed space (always valid)
    let total: usize = parts.iter().map(|p| p.irrep_dimension * p.vectors.len()).sum();
    let commutant_found: usize = parts.iter().map(|p| p.vectors.len() * p.vectors.len()).sum();
    let commutant_exact = commutant_dimension(rep);
    if total != m || commutant_found as f64 != commutant_exact {
        eprintln!(
            "ncalg: isotypic decomposition failed (parts cover {} of {} dimensions, commutant dimension {} instead of {}); this matrix is not block-diagonalized",
            total, m, commutant_found, commutant_exact
        );
        let identity: Vec<Vec<C64>> = (0..m)
            .map(|i| {
                let mut e = vec![C64::new(0.0, 0.0); m];
                e[i] = C64::new(1.0, 0.0);
                e
            })
            .collect();
        return vec![IsotypicPart {
            vectors: identity,
            irrep_dimension: 1,
        }];
    }
    parts
}

/// `dim Hom_G(V, V) = (1/|G|) Σ_g |tr ρ(g)|²` for a unitary representation,
/// rounded to the nearest integer (the matrices `rhos` are all group elements).
fn commutant_dimension(rep: Rep<'_>) -> f64 {
    let sum: f64 = rep.traces().iter().map(|t| t.norm_sqr()).sum();
    (sum / rep.len() as f64).round()
}

/// An irreducible unitary representation of a finite group, by its dimension
/// and its (1,1) matrix coefficients `D(g)_11` for every group element.
#[derive(Clone, Debug)]
pub(crate) struct Irrep {
    pub(crate) dim: usize,
    pub(crate) d11: Vec<C64>,
}

/// The irreducible unitary representations of the finite group with
/// multiplication table `mul[g][h]` (the index of `g·h`), from the isotypic
/// decomposition of its regular representation `e_h ↦ e_{g·h}`: for a unit
/// vector `w` in the first-copy span of a type, `⟨w, ρ(g) w⟩ = D(g)_11`
/// (the copies transform by the same unitary irrep and are orthogonal).
/// `None` unless `Σ_λ d_λ² = |G|` with every type of multiplicity `d_λ`.
pub(crate) fn group_irreps(mul: &[Vec<usize>]) -> Option<Vec<Irrep>> {
    let n = mul.len();
    let forms: Vec<(Vec<usize>, Vec<C64>)> = mul
        .iter()
        .map(|row| (row.clone(), vec![C64::new(1.0, 0.0); n]))
        .collect();
    let parts = isotypic_decomposition(Rep::Monomial(&forms), n);
    let mut irreps = Vec::with_capacity(parts.len());
    let mut total = 0;
    for p in &parts {
        if p.vectors.len() != p.irrep_dimension {
            return None;
        }
        total += p.irrep_dimension * p.irrep_dimension;
        let w = &p.vectors[0];
        // ⟨w, ρ(g) w⟩ = Σ_h conj(w_{g·h}) w_h
        let d11 = (0..n)
            .map(|g| (0..n).map(|h| w[mul[g][h]].conj() * w[h]).sum())
            .collect();
        irreps.push(Irrep {
            dim: p.irrep_dimension,
            d11,
        });
    }
    (total == n).then_some(irreps)
}

/// Exact isotypic decomposition of a unitary monomial representation
/// (`forms[g]`: `ρ(g) e_i = coef[i] e_{perm[i]}`, all group elements, in the
/// order of `irreps`' coefficients), orbit by orbit with Serre's projection
/// formula: for each irrep `λ` and orbit `O` of the basis vectors,
/// `p^λ_11 = (d_λ/|G|) Σ_g conj(D^λ(g)_11) ρ(g)` restricted to `O` is the
/// orthogonal projector onto the first-copy vectors of `λ` inside `O`.  Its
/// orthonormal eigenvectors for eigenvalue 1 are the part's vectors; all of
/// them transform by the same `D^λ`, so copies in different orbits are
/// aligned by construction.  Costs `O(#irreps (|G| m + Σ_O |O|³))` instead
/// of the dense `O(m³)` of the general method, and needs no tolerance on
/// eigenvalue clusters.  `None` if a projector is not one (eigenvalues
/// away from 0 and 1) or the parts fail the dimension and commutant checks;
/// the caller then falls back to [`isotypic_decomposition`].
pub(crate) fn monomial_isotypic_decomposition(
    forms: &[(Vec<usize>, Vec<C64>)],
    m: usize,
    irreps: &[Irrep],
) -> Option<Vec<IsotypicPart>> {
    let order = forms.len();
    if irreps.iter().any(|ir| ir.d11.len() != order) {
        return None;
    }
    // orbits of the basis vectors
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        let mut i = i;
        while p[i] != r {
            let next = p[i];
            p[i] = r;
            i = next;
        }
        r
    }
    for (perm, _) in forms {
        for (i, &j) in perm.iter().enumerate() {
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let mut orbits: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..m {
        let r = find(&mut parent, i);
        orbits.entry(r).or_default().push(i);
    }
    let orbits: Vec<Vec<usize>> = orbits.into_values().collect();
    let mut local = vec![0usize; m];
    for o in &orbits {
        for (k, &i) in o.iter().enumerate() {
            local[i] = k;
        }
    }
    let mut parts = Vec::new();
    for ir in irreps {
        let scale = ir.dim as f64 / order as f64;
        let mut vectors: Vec<Vec<C64>> = Vec::new();
        for o in &orbits {
            let s = o.len();
            let mut p = Mat::<C64>::zeros(s, s);
            for ((perm, coef), d) in forms.iter().zip(&ir.d11) {
                let w = d.conj() * scale;
                if w.norm() == 0.0 {
                    continue;
                }
                for &i in o {
                    p[(local[perm[i]], local[i])] += w * coef[i];
                }
            }
            let adj = p.adjoint().to_owned();
            let p = (&p + &adj) * faer::Scale(C64::new(0.5, 0.0));
            let (vals, vecs) = hermitian_eigen(&p);
            for (k, &v) in vals.iter().enumerate() {
                if (v - 1.0).abs() <= 1e-8 {
                    let mut full = vec![C64::new(0.0, 0.0); m];
                    for (a, &i) in o.iter().enumerate() {
                        full[i] = vecs[(a, k)];
                    }
                    vectors.push(full);
                } else if v.abs() > 1e-8 {
                    return None; // not a projector: the irreps or the action are off
                }
            }
        }
        if !vectors.is_empty() {
            parts.push(IsotypicPart {
                vectors,
                irrep_dimension: ir.dim,
            });
        }
    }
    let total: usize = parts.iter().map(|p| p.irrep_dimension * p.vectors.len()).sum();
    let commutant_found: usize = parts.iter().map(|p| p.vectors.len() * p.vectors.len()).sum();
    (total == m && commutant_found as f64 == commutant_dimension(Rep::Monomial(forms))).then_some(parts)
}

#[cfg(test)]
mod isotypic_tests {
    use super::*;

    /// The D5 action on the 1504 index words of a moment matrix of a
    /// multi-functional relaxation on the 5-cycle.  Without the Schur-lemma refinement two eigenvalues of
    /// the random commutant element (a 1- and a 2-dimensional copy) fell
    /// within the clustering tolerance, were read as a 3-dimensional
    /// irreducible, and the decomposition fell back to the undecomposed
    /// space.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "dense 1504 x 1504 eigenproblems; run with --release"
    )]
    fn accidental_eigenvalue_coincidence_is_split() {
        let text = include_str!("../../fixtures/isotypic/cyclic5_d5_1504.txt");
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let head: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let (m, ng) = (head[0], head[1]);
        let rhos: Vec<Mat<C64>> = lines
            .take(ng)
            .map(|l| {
                let mut rho = Mat::<C64>::zeros(m, m);
                for (j, t) in l.split_whitespace().enumerate() {
                    let s: i64 = t.parse().unwrap();
                    rho[(s.unsigned_abs() as usize - 1, j)] = C64::new(s.signum() as f64, 0.0);
                }
                rho
            })
            .collect();
        assert_eq!(rhos.len(), ng);
        let parts = isotypic_decomposition_uncached(Rep::Dense(&rhos), m);
        let total: usize = parts.iter().map(|p| p.irrep_dimension * p.vectors.len()).sum();
        let commutant: usize = parts.iter().map(|p| p.vectors.len() * p.vectors.len()).sum();
        assert_eq!(total, m);
        assert_eq!(commutant as f64, commutant_dimension(Rep::Dense(&rhos)));
        // D5 has irreducibles of dimension 1 and 2 only
        assert!(parts.len() > 1 && parts.iter().all(|p| p.irrep_dimension <= 2));
        // signed permutations: orthonormal bases (unitary reduction)
        for p in &parts {
            assert!(orthonormality_error(&p.vectors) < 1e-12);
        }
    }

    /// For a monomial representation the reduced blocks are
    /// unitarily equivalent to the restriction, so the spectrum of an
    /// invariant form is the union of the block spectra, each repeated
    /// `irrep_dimension` times.  The regular representation of S3 with signs
    /// has the 2-dimensional irreducible twice, so several projector columns
    /// share the single orbit and Gram–Schmidt has to mix them.
    #[test]
    fn reduced_blocks_are_unitarily_equivalent() {
        let perms: Vec<[usize; 3]> = vec![[0, 1, 2], [1, 2, 0], [2, 0, 1], [1, 0, 2], [0, 2, 1], [2, 1, 0]];
        let compose = |g: &[usize; 3], h: &[usize; 3]| -> [usize; 3] { [g[h[0]], g[h[1]], g[h[2]]] };
        let index = |g: &[usize; 3]| perms.iter().position(|p| p == g).unwrap();
        // ρ(g) e_h = sgn(g) e_{gh}: the regular representation tensored with the sign
        let sign = |g: &[usize; 3]| if index(g) < 3 { 1.0 } else { -1.0 };
        let m = perms.len();
        let rhos: Vec<Mat<C64>> = perms
            .iter()
            .map(|g| {
                let mut rho = Mat::<C64>::zeros(m, m);
                for h in &perms {
                    rho[(index(&compose(g, h)), index(h))] = C64::new(sign(g), 0.0);
                }
                rho
            })
            .collect();
        let parts = isotypic_decomposition_uncached(Rep::Dense(&rhos), m);
        assert!(parts
            .iter()
            .any(|p| p.irrep_dimension == 2 && p.vectors.len() == 2));
        // an invariant Hermitian form: the group average of a fixed one
        let mut h0 = Mat::<C64>::zeros(m, m);
        for i in 0..m {
            for j in i..m {
                let v = C64::new(
                    ((3 * i + 5 * j) % 7) as f64 - 3.0,
                    if i == j {
                        0.0
                    } else {
                        (i + 2 * j) as f64 % 3.0 - 1.0
                    },
                );
                h0[(i, j)] = v;
                h0[(j, i)] = v.conj();
            }
        }
        let mut h = Mat::<C64>::zeros(m, m);
        for rho in &rhos {
            h += mm3(&rho.adjoint().to_owned(), &h0, rho);
        }
        let (full, _) = hermitian_eigen(&h);
        let mut reduced: Vec<f64> = Vec::new();
        for p in &parts {
            assert!(orthonormality_error(&p.vectors) < 1e-12);
            let k = p.vectors.len();
            let v = Mat::from_fn(m, k, |i, j| p.vectors[j][i]);
            let (vals, _) = hermitian_eigen(&mm3(&v.adjoint().to_owned(), &h, &v));
            for _ in 0..p.irrep_dimension {
                reduced.extend(&vals);
            }
        }
        reduced.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(reduced.len(), m);
        for (a, b) in full.iter().zip(&reduced) {
            assert!(
                (a - b).abs() < 1e-10,
                "spectra differ: {:?} vs {:?}",
                full,
                reduced
            );
        }
    }
}
