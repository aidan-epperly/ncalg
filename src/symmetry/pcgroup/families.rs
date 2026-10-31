//! Clifford-hierarchy families as pc presentations, with their monomial representations.
//!
//! Every family here acts by monomial unitaries `R(g)|x⟩ = ω^{phase[x]} |perm[x]⟩`
//! with `ω = e^{2πi/2^r}` ([`Monomial`]); `r = 3` (`ω = e^{iπ/4}`, phases mod 8) for the
//! built-in families, and any `1 ≤ r ≤ 7` for [`x_diag_bits`] (`r = 4` reaches √T).  The pcgs refines the lower exponent-2 central
//! series top-first; this is checked at construction.
//!
//! * [`pauli_cz`]`(m)`  = ⟨X_j, Z_j, CZ_jk⟩, pcgs `X…, CZ… | Z… | −1`.
//! * [`pauli_ccz`]`(m)` = ⟨X_j, Z_j, CZ_jk, CCZ_jkl⟩, pcgs `X…, CCZ… | CZ… | Z… | −1`.
//! * [`xs`]`(m)`        = ⟨X_j, S_j⟩, pcgs `X…, S… | Z…, i | −1`.
//! * [`xt`]`(m)`        = ⟨X_j, T_j⟩, pcgs `X…, T… | S…, ω | Z…, i | −1`.
//! * [`d4n`]`(N)`       = D_4^N on (trivial ⊕ ρ_2)^{⊗N}, pcgs `s…, r… | c = r²…`.
//!
//! For the qubit families the collected word is `X^a · D`, with `D` diagonal.
//! So the presentation is computed from the matrices: products of generators are
//! decomposed back into exponents by [`DiagSolver`].  The scalar generators are
//! listed in [`PcFamily::scalars`]; [`PcFamily::projective`] divides them out.

use super::{Exps, PcPresentation};
use crate::ff::BitEchelon;
use crate::symmetry::perm::phase_mask;
pub use crate::symmetry::perm::Monomial;

/// Elements of `(ℤ/2^bits)^dim` in a 2-adic level basis: representatives
/// `r_k` of valuation `t_k` (no bits below `t_k`) whose level parts
/// `(r_k >> t_k) & 1` are linearly independent over GF(2) at each level.
/// Every element of the binary span `Σ e_k r_k`, `e_k ∈ {0, 1}`, has exactly
/// one such expression, found level by level.  The one implementation of this
/// decomposition, behind [`DiagSolver`] (fixed generators) and the
/// closures of [`PhaseModule`]; each level is a [`BitEchelon`].
#[derive(Clone, Debug)]
struct LevelBasis {
    dim: usize,
    bits: u32,
    reps: Vec<Vec<u8>>,
    /// per level t: the echelon of the level-t parts, and the representative of each insertion index
    levels: Vec<(BitEchelon, Vec<usize>)>,
}

enum PushError {
    Trivial,
    Dependent(usize),
}

impl LevelBasis {
    fn new(dim: usize, bits: u32) -> Self {
        LevelBasis {
            dim,
            bits,
            reps: Vec::new(),
            levels: (0..bits).map(|_| (BitEchelon::new(dim), Vec::new())).collect(),
        }
    }

    fn level_part(&self, f: &[u8], t: usize) -> Vec<u64> {
        bitset(f.iter().map(|&x| (x >> t) & 1 == 1), self.dim)
    }

    /// Add `f` as a representative at its valuation.
    fn push(&mut self, f: &[u8]) -> Result<usize, PushError> {
        let mask = phase_mask(self.bits);
        let f: Vec<u8> = f.iter().map(|&x| x & mask).collect();
        let t = f
            .iter()
            .filter(|&&x| x != 0)
            .map(|&x| x.trailing_zeros() as usize)
            .min()
            .ok_or(PushError::Trivial)?;
        let part = self.level_part(&f, t);
        let k = self.reps.len();
        let (ech, owners) = &mut self.levels[t];
        ech.insert(part).map_err(|_| PushError::Dependent(t))?;
        owners.push(k);
        self.reps.push(f);
        Ok(k)
    }

    /// Subtract representatives level by level: the coefficients `e`, the
    /// residual, and the first level at which the residual is not in the
    /// span (`None` when `f` lies in the binary span, residual zero).
    fn reduce(&self, f: &[u8]) -> (Vec<u8>, Vec<u8>, Option<usize>) {
        let mask = phase_mask(self.bits);
        let mut r: Vec<u8> = f.iter().map(|&x| x & mask).collect();
        let mut e = vec![0u8; self.reps.len()];
        for t in 0..self.bits as usize {
            let mut part = self.level_part(&r, t);
            let (ech, owners) = &self.levels[t];
            let comb = ech.reduce(&mut part);
            for idx in BitEchelon::indices(&comb) {
                let k = owners[idx];
                e[k] = 1;
                for (v, &x) in r.iter_mut().zip(&self.reps[k]) {
                    *v = v.wrapping_sub(x) & mask;
                }
            }
            if part.iter().any(|&w| w != 0) {
                return (e, r, Some(t));
            }
        }
        (e, r, None)
    }
}

fn bitset(v: impl Iterator<Item = bool>, dim: usize) -> Vec<u64> {
    let mut b = vec![0u64; dim.div_ceil(64)];
    for (x, on) in v.enumerate() {
        if on {
            b[x / 64] |= 1 << (x % 64);
        }
    }
    b
}

/// Solves `φ = Σ_k e_k f_k (mod 2^r)` for `e_k ∈ {0,1}`, given the phase functions `f_k`
/// of commuting diagonal pcgs generators.  Works 2-adically, one bit level at a
/// time; requires the mod-2 parts at each level to be linearly independent.
#[derive(Clone, Debug)]
pub struct DiagSolver {
    basis: LevelBasis,
}

impl DiagSolver {
    /// Phases mod 8.
    pub fn new(funcs: Vec<Vec<u8>>) -> Result<Self, String> {
        Self::new_bits(funcs, 3)
    }
    /// Phases mod `2^bits`.
    pub fn new_bits(funcs: Vec<Vec<u8>>, bits: u32) -> Result<Self, String> {
        let dim = funcs.first().map_or(1, |f| f.len());
        let mut basis = LevelBasis::new(dim, bits);
        for (k, f) in funcs.iter().enumerate() {
            basis.push(f).map_err(|e| match e {
                PushError::Trivial => format!("diagonal generator {k} is trivial"),
                PushError::Dependent(t) => format!("diagonal generator {k} is dependent at level {t}"),
            })?;
        }
        Ok(DiagSolver { basis })
    }

    /// Exponents `e` with `Σ e_k f_k = φ (mod 2^r)`, or `None` if `φ` is not in the span.
    pub fn solve(&self, phi: &[u8]) -> Option<Vec<u8>> {
        let (e, _, fail) = self.basis.reduce(phi);
        fail.is_none().then_some(e)
    }
}

#[derive(Clone, Debug)]
enum Rep {
    /// pcgs positions `0..x_basis.len()` are `X^{a}` for the shifts in `x_basis`; the rest are diagonal.
    XDiag {
        m: usize,
        bits: u32,
        x_basis: Vec<u64>,
        diag_funcs: Vec<Vec<u8>>,
        solver: DiagSolver,
    },
    /// `D_4^N`: pcgs `s_0..s_{N-1}, r_0..r_{N-1}, c_0..c_{N-1}`.
    D4Tensor { n: usize },
}

/// A family member: pc presentation, series blocks, scalar generators and representation.
#[derive(Clone, Debug)]
pub struct PcFamily {
    /// Family name, as accepted by [`by_name`].
    pub family: String,
    /// The size parameter: number of qubits, or `N` for `D_4^N`.
    pub size: usize,
    /// The pc presentation.
    pub pres: PcPresentation,
    /// layer dimensions `d_i` of the lower exponent-2 central series (pcgs refines it)
    pub blocks: Vec<usize>,
    /// pcgs positions of the scalar generators (they span the scalar subgroup)
    pub scalars: Vec<usize>,
    rep: Rep,
}

enum GenSpec {
    /// `X^a` for the shift `a` (bit j = qubit j)
    X(u64),
    Diag(Vec<u8>),
}

/// Coordinates of `a` in an independent family of shifts (bit k of the result = coefficient of basis[k]).
fn shift_coords(basis: &[u64], a: u64) -> Option<u64> {
    let mut ech = BitEchelon::new(64);
    for &b in basis {
        let _ = ech.insert(vec![b]); // dependent shifts get no coordinate
    }
    let mut v = [a];
    let comb = ech.reduce(&mut v);
    (v[0] == 0).then(|| comb.first().copied().unwrap_or(0))
}

fn qubit_phase(m: usize, f: impl Fn(usize) -> u8) -> Vec<u8> {
    (0..1usize << m).map(|x| f(x) & 7).collect()
}
fn bit(x: usize, j: usize) -> u8 {
    ((x >> j) & 1) as u8
}

fn combos(m: usize, r: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, m: usize, r: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == r {
            out.push(cur.clone());
            return;
        }
        for j in start..m {
            cur.push(j);
            rec(j + 1, m, r, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, m, r, &mut Vec::new(), &mut out);
    out
}

fn xdiag_family(
    family: &str,
    m: usize,
    gens: Vec<(String, GenSpec)>,
    scalar_names: &[&str],
    bits: u32,
) -> Result<PcFamily, String> {
    let dim = 1usize << m;
    let names: Vec<String> = gens.iter().map(|(n, _)| n.clone()).collect();
    let x_basis: Vec<u64> = gens
        .iter()
        .map_while(|(_, g)| match g {
            GenSpec::X(a) => Some(*a),
            _ => None,
        })
        .collect();
    if gens
        .iter()
        .skip(x_basis.len())
        .any(|(_, g)| matches!(g, GenSpec::X(_)))
    {
        return Err("X generators must come first".into());
    }
    if x_basis.iter().any(|&a| a >> m != 0) {
        return Err("shift out of range".into());
    }
    for k in 0..x_basis.len() {
        if shift_coords(&x_basis[..k], x_basis[k]).is_some() {
            return Err("X shifts must be linearly independent".into());
        }
    }
    let diag_funcs: Vec<Vec<u8>> = gens
        .iter()
        .filter_map(|(_, g)| match g {
            GenSpec::Diag(f) => Some(f.clone()),
            _ => None,
        })
        .collect();
    let solver = DiagSolver::new_bits(diag_funcs.clone(), bits)?;
    let rep = Rep::XDiag {
        m,
        bits,
        x_basis: x_basis.clone(),
        diag_funcs,
        solver,
    };
    let mono = |g: &GenSpec| -> Monomial {
        match g {
            GenSpec::X(a) => Monomial {
                perm: (0..dim as u32).map(|x| x ^ *a as u32).collect(),
                phase: vec![0; dim],
            },
            GenSpec::Diag(f) => Monomial {
                perm: (0..dim as u32).collect(),
                phase: f.clone(),
            },
        }
    };
    let monos: Vec<Monomial> = gens.iter().map(|(_, g)| mono(g)).collect();
    let n = gens.len();
    let decomp = |r: &Rep, mo: &Monomial| decompose_xdiag(r, mo, n);
    let mut power = Vec::new();
    let mut conj = vec![vec![vec![0u8; n]; n]; n];
    for i in 0..n {
        let sq = monos[i].mul_bits(&monos[i], bits);
        power.push(decomp(&rep, &sq).ok_or_else(|| format!("{}^2 not decomposable", names[i]))?);
        for j in (i + 1)..n {
            let c = monos[i]
                .inv_bits(bits)
                .mul_bits(&monos[j], bits)
                .mul_bits(&monos[i], bits);
            conj[i][j] =
                decomp(&rep, &c).ok_or_else(|| format!("conj {} {} not decomposable", names[i], names[j]))?;
        }
    }
    let pres = PcPresentation::new(2, names.clone(), power, conj)?;
    pres.check_consistency()?;
    let blocks = pres
        .series_blocks()
        .ok_or("pcgs does not refine the lower exponent-2 central series")?;
    let scalars = scalar_names
        .iter()
        .map(|s| names.iter().position(|n| n == s).ok_or(format!("no scalar {s}")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PcFamily {
        family: family.to_string(),
        size: m,
        pres,
        blocks,
        scalars,
        rep,
    })
}

fn decompose_xdiag(rep: &Rep, mo: &Monomial, n: usize) -> Option<Exps> {
    let Rep::XDiag { x_basis, solver, .. } = rep else {
        return None;
    };
    let a = mo.perm[0] as usize;
    if mo.perm.iter().enumerate().any(|(x, &y)| y as usize != x ^ a) {
        return None;
    }
    let c = shift_coords(x_basis, a as u64)?;
    let d = solver.solve(&mo.phase)?;
    let nx = x_basis.len();
    let mut e = vec![0u8; n];
    for k in 0..nx {
        e[k] = ((c >> k) & 1) as u8;
    }
    e[nx..].copy_from_slice(&d);
    Some(e)
}

/// Pauli-CZ(m) = ⟨X_j, Z_j, CZ_jk⟩ (order 2·4^m·2^{m(m-1)/2}).
pub fn pauli_cz(m: usize) -> Result<PcFamily, String> {
    let mut g: Vec<(String, GenSpec)> = (0..m).map(|j| (format!("X{j}"), GenSpec::X(1 << j))).collect();
    for c in combos(m, 2) {
        g.push((
            format!("CZ{}{}", c[0], c[1]),
            GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, c[0]) * bit(x, c[1]))),
        ));
    }
    for j in 0..m {
        g.push((format!("Z{j}"), GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, j)))));
    }
    g.push(("-1".into(), GenSpec::Diag(qubit_phase(m, |_| 4))));
    xdiag_family("PauliCZ", m, g, &["-1"], 3)
}

/// Pauli-CCZ(m) = ⟨X_j, Z_j, CZ_jk, CCZ_jkl⟩.
pub fn pauli_ccz(m: usize) -> Result<PcFamily, String> {
    let mut g: Vec<(String, GenSpec)> = (0..m).map(|j| (format!("X{j}"), GenSpec::X(1 << j))).collect();
    for c in combos(m, 3) {
        g.push((
            format!("CCZ{}{}{}", c[0], c[1], c[2]),
            GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, c[0]) * bit(x, c[1]) * bit(x, c[2]))),
        ));
    }
    for c in combos(m, 2) {
        g.push((
            format!("CZ{}{}", c[0], c[1]),
            GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, c[0]) * bit(x, c[1]))),
        ));
    }
    for j in 0..m {
        g.push((format!("Z{j}"), GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, j)))));
    }
    g.push(("-1".into(), GenSpec::Diag(qubit_phase(m, |_| 4))));
    xdiag_family("PauliCCZ", m, g, &["-1"], 3)
}

/// XS(m) = ⟨X_j, S_j⟩, S = diag(1, i) (order 2^{3m+2}).
pub fn xs(m: usize) -> Result<PcFamily, String> {
    let mut g: Vec<(String, GenSpec)> = (0..m).map(|j| (format!("X{j}"), GenSpec::X(1 << j))).collect();
    for j in 0..m {
        g.push((format!("S{j}"), GenSpec::Diag(qubit_phase(m, |x| 2 * bit(x, j)))));
    }
    for j in 0..m {
        g.push((format!("Z{j}"), GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, j)))));
    }
    g.push(("i".into(), GenSpec::Diag(qubit_phase(m, |_| 2))));
    g.push(("-1".into(), GenSpec::Diag(qubit_phase(m, |_| 4))));
    xdiag_family("XS", m, g, &["i", "-1"], 3)
}

/// XT(m) = ⟨X_j, T_j⟩, T = diag(1, ω) (order 2^{4m+3}).
pub fn xt(m: usize) -> Result<PcFamily, String> {
    let mut g: Vec<(String, GenSpec)> = (0..m).map(|j| (format!("X{j}"), GenSpec::X(1 << j))).collect();
    for j in 0..m {
        g.push((format!("T{j}"), GenSpec::Diag(qubit_phase(m, |x| bit(x, j)))));
    }
    for j in 0..m {
        g.push((format!("S{j}"), GenSpec::Diag(qubit_phase(m, |x| 2 * bit(x, j)))));
    }
    g.push(("w".into(), GenSpec::Diag(qubit_phase(m, |_| 1))));
    for j in 0..m {
        g.push((format!("Z{j}"), GenSpec::Diag(qubit_phase(m, |x| 4 * bit(x, j)))));
    }
    g.push(("i".into(), GenSpec::Diag(qubit_phase(m, |_| 2))));
    g.push(("-1".into(), GenSpec::Diag(qubit_phase(m, |_| 4))));
    xdiag_family("XT", m, g, &["w", "i", "-1"], 3)
}

/// D_4^N with `R = (trivial ⊕ ρ_2)^{⊗N}`, `ρ_2(r) = [[0,−1],[1,0]]`, `ρ_2(s) = diag(1,−1)`.
/// Basis index of the tensor factor k is digit k in base 3.
pub fn d4n(nf: usize) -> Result<PcFamily, String> {
    let mut names = Vec::new();
    for k in 0..nf {
        names.push(format!("s{k}"));
    }
    for k in 0..nf {
        names.push(format!("r{k}"));
    }
    for k in 0..nf {
        names.push(format!("c{k}"));
    }
    let mut b = super::PcBuilder::new(2, names);
    for k in 0..nf {
        b = b.power(nf + k, &[(2 * nf + k, 1)]); // r^2 = c
        b = b.conj(k, nf + k, &[(2 * nf + k, 1)]); // s^-1 r s = r^-1 = r c
    }
    let pres = b.build()?;
    pres.check_consistency()?;
    let blocks = pres.series_blocks().ok_or("pcgs does not refine the series")?;
    Ok(PcFamily {
        family: "D4N".into(),
        size: nf,
        pres,
        blocks,
        scalars: vec![],
        rep: Rep::D4Tensor { n: nf },
    })
}

/// Signed permutation of R_1 = trivial ⊕ ρ_2 for s^a r^t: basis j -> (perm, sign).
fn d4_r1(a: u8, t: u8) -> [(usize, bool); 3] {
    // r: e0 -> e0, e1 -> e2, e2 -> -e1 ;  s: e0 -> e0, e1 -> e1, e2 -> -e2
    let r = |(j, neg): (usize, bool)| -> (usize, bool) {
        match j {
            0 => (0, neg),
            1 => (2, neg),
            _ => (1, !neg),
        }
    };
    let s = |(j, neg): (usize, bool)| -> (usize, bool) {
        if j == 2 {
            (2, !neg)
        } else {
            (j, neg)
        }
    };
    let mut out = [(0, false); 3];
    for (j, o) in out.iter_mut().enumerate() {
        let mut v = (j, false);
        for _ in 0..(t % 4) {
            v = r(v);
        }
        if a == 1 {
            v = s(v);
        }
        *o = v;
    }
    out
}

/// A finite subgroup of the additive group of functions `F_2^m -> Z_{2^r}`, stored as a 2-adic level basis:
/// representatives of valuation `t` whose mod-2 parts `(f >> t) & 1` are independent at each level `t`.
/// Binary combinations of the representatives enumerate the subgroup exactly once, provided it is closed
/// under doubling (see [`PhaseModule::insert_closed`]).
#[derive(Clone, Debug)]
struct PhaseModule {
    basis: LevelBasis,
}

impl PhaseModule {
    fn new(dim: usize, bits: u32) -> Self {
        PhaseModule {
            basis: LevelBasis::new(dim, bits),
        }
    }

    fn reps(&self) -> &[Vec<u8>] {
        &self.basis.reps
    }

    fn contains(&self, f: &[u8]) -> bool {
        self.basis.reduce(f).2.is_none()
    }

    /// Insert one element without closing; returns the new representative (if any).
    fn insert(&mut self, f: &[u8]) -> Option<Vec<u8>> {
        let (_, r, fail) = self.basis.reduce(f);
        fail?;
        // the residual has valuation `fail` and a new level part there
        self.basis
            .push(&r)
            .ok()
            .expect("residual is independent at its level");
        Some(r)
    }

    /// Insert `gens` and close under doubling and under the translations `f -> f(. xor a)`, `a` in `shifts`.
    fn insert_closed(&mut self, gens: Vec<Vec<u8>>, shifts: &[u64]) {
        let mut queue = gens;
        while let Some(f) = queue.pop() {
            if let Some(r) = self.insert(&f) {
                let mask = phase_mask(self.basis.bits);
                queue.push(r.iter().map(|&x| x.wrapping_mul(2) & mask).collect());
                for &a in shifts {
                    queue.push((0..self.basis.dim).map(|x| r[x ^ a as usize]).collect());
                }
            }
        }
    }
}

/// The group generated by the shifts `X^a` (`a` in `x_basis`, linearly independent over F_2) and the
/// diagonal gates `D_f |x> = w^{f(x)} |x>` (`f` in `diag`, values mod 8, `w = e^{i pi/4}`) on `m` qubits.
/// [`x_diag_bits`] is the same with values mod `2^r`, `w = e^{2 pi i / 2^r}`.
///
/// The diagonal subgroup is closed automatically under conjugation by the shifts and under squaring.  The
/// lower exponent-2 central series is computed on the phase side,
/// `M_1` = diagonal subgroup, `M_{i+1}` = span{ f(. xor a) - f, 2f : f in M_i, a in shifts },
/// with the shifts in the top layer.  The pcgs is `shifts | reps of M_1/M_2 | reps of M_2/M_3 | ...`, with
/// the scalar subgroup (the constant functions in `M_1`) spanned by its own generators, so
/// [`PcFamily::projective`] divides it out.  The presentation is computed from the matrices and checked
/// (consistency, series refinement) exactly as for the built-in families.
///
/// Generator names: `X<j>` (or `X<hex>` for multi-qubit shifts), `D<k>` for diagonal generators,
/// `w^<c>` for scalars.
pub fn x_diag(family: &str, m: usize, x_basis: &[u64], diag: &[Vec<u8>]) -> Result<PcFamily, String> {
    x_diag_bits(family, m, x_basis, diag, 3)
}

/// [`x_diag`] with phases mod `2^bits` (`1 ≤ bits ≤ 7`), `w = e^{2 pi i / 2^bits}`.  For example `bits = 4`
/// and `f_j(x) = x_j` gives the group generated by `X_j` and `√T_j = diag(1, e^{i pi/8})`.
pub fn x_diag_bits(
    family: &str,
    m: usize,
    x_basis: &[u64],
    diag: &[Vec<u8>],
    bits: u32,
) -> Result<PcFamily, String> {
    let mask = phase_mask(bits);
    let dim = 1usize << m;
    if m > 20 {
        return Err("x_diag: at most 20 qubits".into());
    }
    if diag.iter().any(|f| f.len() != dim) {
        return Err("diagonal phase functions must have length 2^m".into());
    }
    let mut series: Vec<PhaseModule> = Vec::new();
    let mut m1 = PhaseModule::new(dim, bits);
    m1.insert_closed(diag.to_vec(), x_basis);
    series.push(m1);
    loop {
        let last = series.last().expect("nonempty");
        if last.reps().is_empty() {
            break;
        }
        let mut gens = Vec::new();
        for r in last.reps() {
            gens.push(r.iter().map(|&x| x.wrapping_mul(2) & mask).collect::<Vec<u8>>());
            for &a in x_basis {
                gens.push(
                    (0..dim)
                        .map(|x| r[x ^ a as usize].wrapping_sub(r[x]) & mask)
                        .collect::<Vec<u8>>(),
                );
            }
        }
        let mut next = PhaseModule::new(dim, bits);
        next.insert_closed(gens, x_basis);
        if next.reps().len() >= last.reps().len() {
            return Err("series does not descend".into());
        }
        series.push(next);
    }
    // scalars: the constant functions in M_1, generated by the smallest constant c (then 2c, 4c, ...)
    let scal: Vec<u8> = match (0..bits)
        .map(|k| 1u8 << k)
        .find(|&c| series[0].contains(&vec![c; dim]))
    {
        Some(c) => {
            let mut v = Vec::new();
            let mut x = c;
            while x != 0 {
                v.push(x);
                x = x.wrapping_mul(2) & mask;
            }
            v
        }
        None => Vec::new(),
    };
    let layer_of = |f: &[u8]| -> usize {
        (0..series.len())
            .rev()
            .find(|&i| series[i].contains(f))
            .expect("element of M_1")
    };
    // ordered basis, built from the deepest layer up: (function, layer, scalar constant)
    let mut basis = PhaseModule::new(dim, bits);
    let mut chosen: Vec<(Vec<u8>, usize, Option<u8>)> = Vec::new();
    for i in (0..series.len()).rev() {
        for &c in &scal {
            let f = vec![c; dim];
            if layer_of(&f) == i {
                if let Some(r) = basis.insert(&f) {
                    chosen.push((r, i, Some(c)));
                }
            }
        }
        for r in series[i].reps().to_vec() {
            if let Some(r2) = basis.insert(&r) {
                chosen.push((r2, i, None));
            }
        }
    }
    chosen.sort_by_key(|(_, i, sc)| (*i, sc.is_some()));
    let mut gens: Vec<(String, GenSpec)> = x_basis
        .iter()
        .map(|&a| {
            let name = if a.count_ones() == 1 {
                format!("X{}", a.trailing_zeros())
            } else {
                format!("X<{a:x}>")
            };
            (name, GenSpec::X(a))
        })
        .collect();
    let mut scalar_names = Vec::new();
    let mut k = 0;
    for (f, _, sc) in chosen {
        let name = match sc {
            Some(c) => {
                let nm = format!("w^{c}");
                scalar_names.push(nm.clone());
                nm
            }
            None => {
                k += 1;
                format!("D{}", k - 1)
            }
        };
        gens.push((name, GenSpec::Diag(f)));
    }
    let scal_refs: Vec<&str> = scalar_names.iter().map(String::as_str).collect();
    xdiag_family(family, m, gens, &scal_refs, bits)
}

impl PcFamily {
    /// Phases of [`PcFamily::monomial`] are mod `2^phase_bits()`, `ω = e^{2πi/2^phase_bits()}`.
    pub fn phase_bits(&self) -> u32 {
        match &self.rep {
            Rep::XDiag { bits, .. } => *bits,
            Rep::D4Tensor { .. } => 3,
        }
    }

    /// Dimension of the representation space.
    pub fn dim(&self) -> usize {
        match &self.rep {
            Rep::XDiag { m, .. } => 1 << m,
            Rep::D4Tensor { n } => 3usize.pow(*n as u32),
        }
    }

    /// `R(g)` for the collected element with exponents `e`.
    pub fn monomial(&self, e: &[u8]) -> Monomial {
        match &self.rep {
            Rep::XDiag {
                m,
                bits,
                x_basis,
                diag_funcs,
                ..
            } => {
                let mask = phase_mask(*bits);
                let dim = 1usize << m;
                let nx = x_basis.len();
                let a: usize = (0..nx)
                    .filter(|&k| e[k] != 0)
                    .fold(0, |acc, k| acc ^ x_basis[k] as usize);
                let mut phase = vec![0u8; dim];
                for (k, f) in diag_funcs.iter().enumerate() {
                    if e[nx + k] != 0 {
                        for x in 0..dim {
                            phase[x] = phase[x].wrapping_add(f[x]) & mask;
                        }
                    }
                }
                Monomial {
                    perm: (0..dim as u32).map(|x| x ^ a as u32).collect(),
                    phase,
                }
            }
            Rep::D4Tensor { n } => {
                let dim = 3usize.pow(*n as u32);
                // R(s^a r^b c^d) = R(s)^a R(r)^(b + 2d); R(g) |j> for the collected word acts as s^a r^t
                // with the word read as an operator product: s^a · r^t (r acts first)
                let tables: Vec<[(usize, bool); 3]> = (0..*n)
                    .map(|k| {
                        let a = e[k];
                        let t = e[n + k] + 2 * e[2 * n + k];
                        // (s^a r^t)|j> = s^a (r^t |j>)
                        let rt = d4_r1(0, t);
                        let mut out = [(0, false); 3];
                        for j in 0..3 {
                            let (j1, n1) = rt[j];
                            let (j2, n2) = d4_r1(a, 0)[j1];
                            out[j] = (j2, n1 ^ n2);
                        }
                        out
                    })
                    .collect();
                let mut perm = vec![0u32; dim];
                let mut phase = vec![0u8; dim];
                for x in 0..dim {
                    let (mut y, mut neg, mut p3, mut rem) = (0usize, false, 1usize, x);
                    for t in &tables {
                        let (j, ng) = t[rem % 3];
                        y += j * p3;
                        neg ^= ng;
                        rem /= 3;
                        p3 *= 3;
                    }
                    perm[x] = y as u32;
                    phase[x] = if neg { 4 } else { 0 };
                }
                Monomial { perm, phase }
            }
        }
    }

    /// Exponents of the element acting as `mo`, if it lies in the group (qubit families only).
    pub fn decompose(&self, mo: &Monomial) -> Option<Exps> {
        decompose_xdiag(&self.rep, mo, self.pres.len())
    }

    /// The projective group G/⟨scalars⟩, with the pcgs positions kept.
    pub fn projective(&self) -> Result<(PcPresentation, Vec<usize>), String> {
        let q = self.pres.quotient_central(&self.scalars)?;
        let keep = (0..self.pres.len())
            .filter(|k| !self.scalars.contains(k))
            .collect();
        Ok((q, keep))
    }
}

/// Family by name: `PauliCZ`, `PauliCCZ`, `XS`, `XT`, `D4N`.
pub fn by_name(family: &str, size: usize) -> Result<PcFamily, String> {
    match family {
        "PauliCZ" => pauli_cz(size),
        "PauliCCZ" => pauli_ccz(size),
        "XS" => xs(size),
        "XT" => xt(size),
        "D4N" => d4n(size),
        _ => Err(format!("unknown family {family}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 11
    }

    fn check_family(f: &PcFamily, log_order: usize, blocks: &[usize], proj_blocks: &[usize]) {
        let g = &f.pres;
        assert_eq!(g.log_order(), log_order, "{} {}", f.family, f.size);
        assert_eq!(f.blocks, blocks, "{} {}", f.family, f.size);
        let mut seed = 12345u64 + f.size as u64;
        for _ in 0..300 {
            let a = g.from_index(lcg(&mut seed) % (1u64 << log_order));
            let b = g.from_index(lcg(&mut seed) % (1u64 << log_order));
            let ab = g.mul(&a, &b);
            assert_eq!(
                f.monomial(&ab),
                f.monomial(&a).mul(&f.monomial(&b)),
                "{} hom",
                f.family
            );
            assert_eq!(f.monomial(&g.inv(&a)), f.monomial(&a).inv());
            if f.family != "D4N" {
                assert_eq!(f.decompose(&f.monomial(&a)).as_ref(), Some(&a));
            }
        }
        // faithful on generators: distinct monomials
        let (q, _) = f.projective().unwrap();
        q.check_consistency().unwrap();
        assert_eq!(q.series_blocks().unwrap(), proj_blocks, "{} projective", f.family);
    }

    #[test]
    fn families_small() {
        for m in 2..=4 {
            let c2 = m * (m - 1) / 2;
            let c3 = m * (m - 1) * (m - 2) / 6;
            check_family(
                &pauli_cz(m).unwrap(),
                2 * m + c2 + 1,
                &[m + c2, m, 1],
                &[m + c2, m],
            );
            check_family(&xs(m).unwrap(), 3 * m + 2, &[2 * m, m + 1, 1], &[2 * m, m]);
            check_family(
                &xt(m).unwrap(),
                4 * m + 3,
                &[2 * m, m + 1, m + 1, 1],
                &[2 * m, m, m],
            );
            if m >= 3 {
                check_family(
                    &pauli_ccz(m).unwrap(),
                    2 * m + c2 + c3 + 1,
                    &[m + c3, c2, m, 1],
                    &[m + c3, c2, m],
                );
            }
        }
        for nf in 1..=4 {
            check_family(&d4n(nf).unwrap(), 3 * nf, &[2 * nf, nf], &[2 * nf, nf]);
        }
    }

    #[test]
    fn x_diag_reproduces_builtins() {
        for m in 2..=4usize {
            let units: Vec<u64> = (0..m).map(|j| 1u64 << j).collect();
            let ph = |f: &dyn Fn(usize) -> u8| -> Vec<u8> { (0..1usize << m).map(|x| f(x) & 7).collect() };
            let s_gates: Vec<Vec<u8>> = (0..m).map(|j| ph(&|x| 2 * bit(x, j))).collect();
            let t_gates: Vec<Vec<u8>> = (0..m).map(|j| ph(&|x| bit(x, j))).collect();
            let cz_gates: Vec<Vec<u8>> = combos(m, 2)
                .into_iter()
                .map(|c| ph(&move |x| 4 * bit(x, c[0]) * bit(x, c[1])))
                .collect();
            for (custom, builtin) in [
                (x_diag("XS*", m, &units, &s_gates).unwrap(), xs(m).unwrap()),
                (x_diag("XT*", m, &units, &t_gates).unwrap(), xt(m).unwrap()),
                (x_diag("CZ*", m, &units, &cz_gates).unwrap(), pauli_cz(m).unwrap()),
            ] {
                assert_eq!(custom.pres.len(), builtin.pres.len(), "{}", custom.family);
                assert_eq!(custom.blocks, builtin.blocks, "{}", custom.family);
                assert_eq!(custom.scalars.len(), builtin.scalars.len(), "{}", custom.family);
                let (pc, _) = custom.projective().unwrap();
                let (pb, _) = builtin.projective().unwrap();
                assert_eq!(pc.series_blocks(), pb.series_blocks());
                for i in 0..custom.pres.len() {
                    assert!(builtin.decompose(&custom.monomial(&custom.pres.gen(i))).is_some());
                }
                for i in 0..builtin.pres.len() {
                    assert!(custom
                        .decompose(&builtin.monomial(&builtin.pres.gen(i)))
                        .is_some());
                }
            }
        }
        // a non-unit shift basis: X^{11}, X^{10} on 2 qubits with a CZ gives Pauli-CZ(2) again
        let f = x_diag("odd", 2, &[0b11, 0b10], &[vec![0, 0, 0, 4]]).unwrap();
        f.pres.check_consistency().unwrap();
        assert_eq!(f.pres.len(), pauli_cz(2).unwrap().pres.len());
    }

    #[test]
    fn x_sqrt_t_has_four_projective_layers() {
        for m in 1..=3usize {
            let units: Vec<u64> = (0..m).map(|j| 1u64 << j).collect();
            let rt: Vec<Vec<u8>> = (0..m)
                .map(|j| (0..1usize << m).map(|x| bit(x, j)).collect())
                .collect();
            let f = x_diag_bits("XsqrtT", m, &units, &rt, 4).unwrap();
            assert_eq!(f.phase_bits(), 4);
            // X, √T | T, ζ16 | S, ω | Z, i | -1, with the scalars ζ16 = e^{iπ/8} on top of XT's
            assert_eq!(f.pres.len(), 5 * m + 4);
            assert_eq!(f.blocks, vec![2 * m, m + 1, m + 1, m + 1, 1]);
            let (q, _) = f.projective().unwrap();
            assert_eq!(q.series_blocks().unwrap(), vec![2 * m, m, m, m]);
            let g = &f.pres;
            let mut seed = 7u64;
            for _ in 0..200 {
                let a = g.from_index(lcg(&mut seed) % (1u64 << g.len()));
                let b = g.from_index(lcg(&mut seed) % (1u64 << g.len()));
                let ab = g.mul(&a, &b);
                assert_eq!(f.monomial(&ab), f.monomial(&a).mul_bits(&f.monomial(&b), 4));
                assert_eq!(f.decompose(&f.monomial(&a)).as_ref(), Some(&a));
            }
        }
    }

    #[test]
    fn group_algebra_groebner_is_complete() {
        use crate::{BigRational, GroebnerBasis, GroebnerOptions, QuotientAlgebra};
        for f in [pauli_cz(3).unwrap(), xs(2).unwrap(), d4n(2).unwrap()] {
            let (_alg, rels) = f.pres.group_algebra_relations::<BigRational>();
            let gb = GroebnerBasis::compute(rels, &GroebnerOptions::default());
            assert!(gb.is_complete());
            let q = QuotientAlgebra::new(gb);
            assert_eq!(q.dimension(), Some(1usize << f.pres.len()), "{}", f.family);
        }
    }
}

#[cfg(test)]
mod damage_tests {
    use super::*;

    #[test]
    fn damaged_family_presentation_is_inconsistent() {
        // flip one bit in one conjugate relation of XS(2) and of Pauli-CZ(3)
        for f in [xs(2).unwrap(), pauli_cz(3).unwrap()] {
            let g = &f.pres;
            let n = g.len();
            let mut found = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut conj: Vec<Vec<Exps>> = (0..n)
                        .map(|a| (0..n).map(|b| g.conj_relation(a, b).clone()).collect())
                        .collect();
                    // only alter relations whose correction can be changed within the support rule
                    if j + 1 >= n {
                        continue;
                    }
                    conj[i][j][n - 1] ^= 1;
                    let power = (0..n).map(|a| g.power_relation(a).clone()).collect();
                    let bad = PcPresentation::new(2, g.names().to_vec(), power, conj).unwrap();
                    if bad.check_consistency().is_err() {
                        found += 1;
                    }
                }
            }
            assert!(found > 0, "{}: no damaged relation detected", f.family);
        }
    }
}
