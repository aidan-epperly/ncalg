//! Dense matrices over a [`Gf`] and bit-packed matrices over GF(2).

use super::Gf;

/// Dense row-major matrix over a finite field.
///
/// The same algorithms over compile-time [`Field`](crate::Field) types (rationals,
/// cyclotomics, [`Fq<P, K>`](super::Fq)) are in [`crate::linalg::exact`]; this one
/// is kept for fields chosen at run time, whose elements are `u64` encodings with
/// table arithmetic in a [`Gf`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FqMat<'f> {
    f: &'f Gf,
    rows: usize,
    cols: usize,
    data: Vec<u64>,
}

impl<'f> FqMat<'f> {
    /// The `rows × cols` zero matrix over `f`.
    pub fn zeros(f: &'f Gf, rows: usize, cols: usize) -> Self {
        FqMat {
            f,
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }
    /// The `n × n` identity matrix over `f`.
    pub fn identity(f: &'f Gf, n: usize) -> Self {
        let mut m = Self::zeros(f, n, n);
        for i in 0..n {
            m.data[i * n + i] = 1;
        }
        m
    }
    /// From rows of encodings.
    pub fn from_rows(f: &'f Gf, rows: &[Vec<u64>]) -> Self {
        let r = rows.len();
        let c = rows.first().map_or(0, |x| x.len());
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c, "ragged rows");
            data.extend(row.iter().map(|&x| {
                assert!(x < f.q(), "entry out of range");
                x
            }));
        }
        FqMat {
            f,
            rows: r,
            cols: c,
            data,
        }
    }
    /// The field of the entries.
    pub fn field(&self) -> &'f Gf {
        self.f
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    /// Entry `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> u64 {
        self.data[i * self.cols + j]
    }
    #[inline]
    /// Set entry `(i, j)` to `v`.
    pub fn set(&mut self, i: usize, j: usize, v: u64) {
        self.data[i * self.cols + j] = v;
    }
    /// Row `i`.
    pub fn row(&self, i: usize) -> &[u64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// The matrix product `self · b`.
    pub fn mul(&self, b: &FqMat<'f>) -> FqMat<'f> {
        assert_eq!(self.cols, b.rows);
        let f = self.f;
        let mut c = FqMat::zeros(f, self.rows, b.cols);
        for i in 0..self.rows {
            for t in 0..self.cols {
                let a = self.get(i, t);
                if a == 0 {
                    continue;
                }
                for j in 0..b.cols {
                    let v = f.add(c.get(i, j), f.mul(a, b.get(t, j)));
                    c.set(i, j, v);
                }
            }
        }
        c
    }

    /// The matrix-vector product `self · v`.
    pub fn mul_vec(&self, v: &[u64]) -> Vec<u64> {
        assert_eq!(v.len(), self.cols);
        (0..self.rows)
            .map(|i| {
                self.row(i)
                    .iter()
                    .zip(v)
                    .fold(0, |s, (&a, &x)| self.f.add(s, self.f.mul(a, x)))
            })
            .collect()
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            for j in 0..self.cols {
                self.data.swap(a * self.cols + j, b * self.cols + j);
            }
        }
    }

    /// Reduced row echelon form in place; returns the pivot columns.
    pub fn rref(&mut self) -> Vec<usize> {
        let f = self.f;
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in 0..self.cols {
            if r == self.rows {
                break;
            }
            let Some(p) = (r..self.rows).find(|&i| self.get(i, c) != 0) else {
                continue;
            };
            self.swap_rows(r, p);
            let inv = f.inv(self.get(r, c));
            for j in c..self.cols {
                let v = f.mul(self.get(r, j), inv);
                self.set(r, j, v);
            }
            for i in 0..self.rows {
                if i != r {
                    let fac = self.get(i, c);
                    if fac != 0 {
                        for j in c..self.cols {
                            let v = f.sub(self.get(i, j), f.mul(fac, self.get(r, j)));
                            self.set(i, j, v);
                        }
                    }
                }
            }
            pivots.push(c);
            r += 1;
        }
        pivots
    }

    /// Rank, by row reduction of a copy.
    pub fn rank(&self) -> usize {
        self.clone().rref().len()
    }

    /// Basis of `{x : A x = 0}` (as vectors of length `cols`).
    pub fn nullspace(&self) -> Vec<Vec<u64>> {
        let f = self.f;
        let mut m = self.clone();
        let piv = m.rref();
        let free: Vec<usize> = (0..self.cols).filter(|c| !piv.contains(c)).collect();
        free.iter()
            .map(|&fc| {
                let mut v = vec![0u64; self.cols];
                v[fc] = 1;
                for (r, &pc) in piv.iter().enumerate() {
                    v[pc] = f.neg(m.get(r, fc));
                }
                v
            })
            .collect()
    }

    /// Some `x` with `A x = b`, or `None` if inconsistent.
    pub fn solve(&self, b: &[u64]) -> Option<Vec<u64>> {
        assert_eq!(b.len(), self.rows);
        let mut aug = FqMat::zeros(self.f, self.rows, self.cols + 1);
        for i in 0..self.rows {
            for j in 0..self.cols {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, self.cols, b[i]);
        }
        let piv = aug.rref();
        if piv.last() == Some(&self.cols) {
            return None;
        }
        let mut x = vec![0u64; self.cols];
        for (r, &pc) in piv.iter().enumerate() {
            x[pc] = aug.get(r, self.cols);
        }
        Some(x)
    }

    /// Inverse of a square matrix, or `None` if singular.
    pub fn inverse(&self) -> Option<FqMat<'f>> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut aug = FqMat::zeros(self.f, n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1);
        }
        let piv = aug.rref();
        if piv.len() < n || piv[n - 1] != n - 1 {
            return None;
        }
        let mut inv = FqMat::zeros(self.f, n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug.get(i, n + j));
            }
        }
        Some(inv)
    }

    /// Determinant of a square matrix.
    pub fn det(&self) -> u64 {
        assert_eq!(self.rows, self.cols);
        let f = self.f;
        let n = self.rows;
        let mut m = self.clone();
        let mut d = 1u64;
        for c in 0..n {
            let Some(p) = (c..n).find(|&i| m.get(i, c) != 0) else {
                return 0;
            };
            if p != c {
                m.swap_rows(p, c);
                d = f.neg(d);
            }
            let piv = m.get(c, c);
            d = f.mul(d, piv);
            let inv = f.inv(piv);
            for i in (c + 1)..n {
                let fac = f.mul(m.get(i, c), inv);
                if fac != 0 {
                    for j in c..n {
                        let v = f.sub(m.get(i, j), f.mul(fac, m.get(c, j)));
                        m.set(i, j, v);
                    }
                }
            }
        }
        d
    }
}

/// Bit-packed matrix over GF(2): each row is `ceil(cols/64)` words, bit `j` of the
/// row is column `j`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMat {
    rows: usize,
    cols: usize,
    words: usize,
    data: Vec<u64>,
}

impl BitMat {
    /// The `rows × cols` zero matrix over GF(2).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let words = cols.div_ceil(64).max(1);
        BitMat {
            rows,
            cols,
            words,
            data: vec![0; rows * words],
        }
    }
    /// The `n × n` identity matrix over GF(2).
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }
    /// From rows given as `u64` bit masks (`cols ≤ 64`).
    pub fn from_u64_rows(rows: &[u64], cols: usize) -> Self {
        assert!(cols <= 64);
        let mut m = Self::zeros(rows.len(), cols);
        for (i, &r) in rows.iter().enumerate() {
            m.data[i * m.words] = if cols == 64 { r } else { r & ((1u64 << cols) - 1) };
        }
        m
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    /// Entry `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> bool {
        (self.data[i * self.words + j / 64] >> (j % 64)) & 1 == 1
    }
    #[inline]
    /// Set entry `(i, j)` to `v`.
    pub fn set(&mut self, i: usize, j: usize, v: bool) {
        let w = &mut self.data[i * self.words + j / 64];
        if v {
            *w |= 1 << (j % 64);
        } else {
            *w &= !(1 << (j % 64));
        }
    }
    /// Row `i` as packed 64-bit words, column `j` at bit `j % 64` of word `j / 64`.
    pub fn row_words(&self, i: usize) -> &[u64] {
        &self.data[i * self.words..(i + 1) * self.words]
    }
    #[inline]
    fn xor_row(&mut self, dst: usize, src: usize) {
        let w = self.words;
        let (d, s) = (dst * w, src * w);
        for k in 0..w {
            let v = self.data[s + k];
            self.data[d + k] ^= v;
        }
    }
    fn swap_rows(&mut self, a: usize, b: usize) {
        if a != b {
            for k in 0..self.words {
                self.data.swap(a * self.words + k, b * self.words + k);
            }
        }
    }

    /// The matrix product `self · b`.
    pub fn mul(&self, b: &BitMat) -> BitMat {
        assert_eq!(self.cols, b.rows);
        let mut c = BitMat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for t in 0..self.cols {
                if self.get(i, t) {
                    let (dst, src) = (i * c.words, t * b.words);
                    for k in 0..c.words {
                        c.data[dst + k] ^= b.data[src + k];
                    }
                }
            }
        }
        c
    }

    /// Reduced row echelon form in place; returns the pivot columns.
    pub fn rref(&mut self) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut r = 0;
        for c in 0..self.cols {
            if r == self.rows {
                break;
            }
            let Some(p) = (r..self.rows).find(|&i| self.get(i, c)) else {
                continue;
            };
            self.swap_rows(r, p);
            for i in 0..self.rows {
                if i != r && self.get(i, c) {
                    self.xor_row(i, r);
                }
            }
            pivots.push(c);
            r += 1;
        }
        pivots
    }

    /// Rank, by row reduction of a copy.
    pub fn rank(&self) -> usize {
        self.clone().rref().len()
    }

    /// Basis of `{x : A x = 0}`, each vector as a `BitMat` row (1 x cols).
    pub fn nullspace(&self) -> BitMat {
        let mut m = self.clone();
        let piv = m.rref();
        let free: Vec<usize> = (0..self.cols).filter(|c| !piv.contains(c)).collect();
        let mut out = BitMat::zeros(free.len(), self.cols);
        for (k, &fc) in free.iter().enumerate() {
            out.set(k, fc, true);
            for (r, &pc) in piv.iter().enumerate() {
                if m.get(r, fc) {
                    out.set(k, pc, true);
                }
            }
        }
        out
    }

    /// Some `x` (bits) with `A x = b`, or `None`.
    pub fn solve(&self, b: &[bool]) -> Option<Vec<bool>> {
        assert_eq!(b.len(), self.rows);
        let mut aug = BitMat::zeros(self.rows, self.cols + 1);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.get(i, j) {
                    aug.set(i, j, true);
                }
            }
            aug.set(i, self.cols, b[i]);
        }
        let piv = aug.rref();
        if piv.last() == Some(&self.cols) {
            return None;
        }
        let mut x = vec![false; self.cols];
        for (r, &pc) in piv.iter().enumerate() {
            x[pc] = aug.get(r, self.cols);
        }
        Some(x)
    }

    /// Zero-sum finder: a nonempty set of row indices whose rows sum to zero, or
    /// `None` if the rows are independent.  Rows are inserted in order into a
    /// [`BitEchelon`] and the first dependency found is returned, so the
    /// result depends only on the rows themselves.
    pub fn dependency(&self) -> Option<Vec<usize>> {
        let mut ech = BitEchelon::new(self.cols);
        for i in 0..self.rows {
            if let Err(comb) = ech.insert(self.row_words(i).to_vec()) {
                return Some(BitEchelon::indices(&comb));
            }
        }
        None
    }

    /// Convert to a dense [`FqMat`] over GF(2).
    pub fn to_fqmat<'f>(&self, f: &'f Gf) -> FqMat<'f> {
        assert_eq!(f.q(), 2);
        let mut m = FqMat::zeros(f, self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.get(i, j) {
                    m.set(i, j, 1);
                }
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 11
    }

    #[test]
    fn dense_over_several_fields() {
        let mut s = 5u64;
        for (p, k) in [(2, 1), (3, 1), (2, 3), (3, 2), (5, 2), (2, 8)] {
            let f = Gf::new(p, k).unwrap();
            for trial in 0..20 {
                let (r, c) = (2 + trial % 6, 2 + (trial * 7) % 7);
                let rows: Vec<Vec<u64>> = (0..r)
                    .map(|_| (0..c).map(|_| rnd(&mut s) % f.q()).collect())
                    .collect();
                let a = FqMat::from_rows(&f, &rows);
                let ns = a.nullspace();
                assert_eq!(ns.len() + a.rank(), c);
                for v in &ns {
                    assert!(a.mul_vec(v).iter().all(|&x| x == 0));
                }
                let x: Vec<u64> = (0..c).map(|_| rnd(&mut s) % f.q()).collect();
                let b = a.mul_vec(&x);
                let y = a.solve(&b).expect("consistent");
                assert_eq!(a.mul_vec(&y), b);
                if r == c {
                    match a.inverse() {
                        Some(inv) => {
                            assert_eq!(inv.mul(&a), FqMat::identity(&f, r));
                            assert_ne!(a.det(), 0);
                        }
                        None => assert_eq!(a.det(), 0),
                    }
                }
            }
        }
    }

    #[test]
    fn bitmat_agrees_with_dense_gf2() {
        let f = Gf::prime(2).unwrap();
        let mut s = 17u64;
        for trial in 0..200 {
            let (r, c) = (1 + trial % 90, 1 + (trial * 13) % 130);
            let mut m = BitMat::zeros(r, c);
            for i in 0..r {
                for j in 0..c {
                    if rnd(&mut s) % 3 == 0 {
                        m.set(i, j, true);
                    }
                }
            }
            let d = m.to_fqmat(&f);
            assert_eq!(m.rank(), d.rank());
            let ns = m.nullspace();
            assert_eq!(ns.rows(), d.nullspace().len());
            for k in 0..ns.rows() {
                let v: Vec<u64> = (0..c).map(|j| ns.get(k, j) as u64).collect();
                assert!(d.mul_vec(&v).iter().all(|&x| x == 0));
            }
            let x: Vec<u64> = (0..c).map(|_| rnd(&mut s) & 1).collect();
            let b: Vec<bool> = d.mul_vec(&x).iter().map(|&v| v == 1).collect();
            let y = m.solve(&b).expect("consistent");
            let yb: Vec<u64> = y.iter().map(|&v| v as u64).collect();
            assert_eq!(d.mul_vec(&yb), b.iter().map(|&v| v as u64).collect::<Vec<_>>());
            // dependency finder
            match m.dependency() {
                Some(set) => {
                    assert!(!set.is_empty());
                    let mut acc = vec![0u64; m.row_words(0).len()];
                    for &i in &set {
                        for (a, b) in acc.iter_mut().zip(m.row_words(i)) {
                            *a ^= *b;
                        }
                    }
                    assert!(acc.iter().all(|&w| w == 0));
                }
                None => assert_eq!(m.rank(), r),
            }
            if r > c {
                assert!(m.dependency().is_some());
            }
        }
    }
}

/// An incrementally built row-echelon basis over GF(2) that remembers which
/// inserted rows each basis row is the sum of: the one implementation of
/// Gaussian elimination with combination tracking in ncalg, used by
/// [`BitMat::dependency`] and by the decompositions in `pcgroup::families`.
///
/// Rows are bit vectors of `cols` bits packed in `u64` words (bit `j` of the
/// row is column `j`).  Every inserted row gets the next insertion index,
/// independent or not; combinations are bit sets over insertion indices.  The
/// basis is kept fully reduced (each pivot, the lowest set bit of a basis row,
/// occurs in no other basis row), so one pass reduces a vector.
#[derive(Clone, Debug, Default)]
pub struct BitEchelon {
    words: usize,
    inserted: usize,
    /// (pivot, row, combination)
    basis: Vec<(usize, Vec<u64>, Vec<u64>)>,
}

impl BitEchelon {
    /// An empty basis for rows of `cols` bits.
    pub fn new(cols: usize) -> Self {
        BitEchelon {
            words: cols.div_ceil(64).max(1),
            inserted: 0,
            basis: Vec::new(),
        }
    }

    /// Number of independent rows inserted.
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// Number of rows inserted (the next insertion index).
    pub fn inserted(&self) -> usize {
        self.inserted
    }

    fn xor_into(a: &mut Vec<u64>, b: &[u64]) {
        if a.len() < b.len() {
            a.resize(b.len(), 0);
        }
        for (x, y) in a.iter_mut().zip(b) {
            *x ^= *y;
        }
    }

    /// Reduce `v` against the basis in place; returns the combination (over
    /// insertion indices) of the rows subtracted.  `v` reduces to zero iff it
    /// lies in the span, and then it equals that combination.
    pub fn reduce(&self, v: &mut [u64]) -> Vec<u64> {
        assert_eq!(v.len(), self.words, "row length");
        let mut comb = Vec::new();
        for (piv, row, c) in &self.basis {
            if (v[piv / 64] >> (piv % 64)) & 1 == 1 {
                for (x, y) in v.iter_mut().zip(row) {
                    *x ^= *y;
                }
                Self::xor_into(&mut comb, c);
            }
        }
        comb
    }

    /// Insert a row.  `Ok(index)` if it is independent of the rows before it;
    /// otherwise `Err(combination)`, a set of insertion indices (including
    /// this row's) whose rows sum to zero, and the basis is unchanged.
    pub fn insert(&mut self, mut v: Vec<u64>) -> Result<usize, Vec<u64>> {
        let index = self.inserted;
        self.inserted += 1;
        let mut comb = self.reduce(&mut v);
        Self::xor_into(&mut comb, &Self::single(index));
        let Some((w, &x)) = v.iter().enumerate().find(|(_, &x)| x != 0) else {
            return Err(comb);
        };
        let piv = w * 64 + x.trailing_zeros() as usize;
        // keep the basis reduced at the new pivot
        for (_, row, c) in self.basis.iter_mut() {
            if (row[piv / 64] >> (piv % 64)) & 1 == 1 {
                for (a, b) in row.iter_mut().zip(&v) {
                    *a ^= *b;
                }
                Self::xor_into(c, &comb);
            }
        }
        self.basis.push((piv, v, comb));
        Ok(index)
    }

    /// The bit set `{i}`.
    pub fn single(i: usize) -> Vec<u64> {
        let mut c = vec![0u64; i / 64 + 1];
        c[i / 64] |= 1 << (i % 64);
        c
    }

    /// The indices in a bit set, increasing.
    pub fn indices(set: &[u64]) -> Vec<usize> {
        let mut out = Vec::new();
        for (w, &word) in set.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                out.push(w * 64 + x.trailing_zeros() as usize);
                x &= x - 1;
            }
        }
        out
    }
}
