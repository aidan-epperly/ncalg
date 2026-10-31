//! Integer and modular arithmetic shared by every module of ncalg:
//! gcd and extended gcd, modular products, powers and inverses, primality and
//! prime factors of machine integers.
//!
//! These are the only implementations in the crate; add new number-theoretic
//! helpers here rather than next to their first caller.  Plain gcd and lcm
//! come from `num_integer` (`a.gcd(&b)`) and are not wrapped.

use num_integer::Integer;

/// `(g, s, t)` with `s a + t b = g = gcd(a, b) ≥ 0`.
pub fn xgcd(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut r0, mut r1, mut s0, mut s1, mut t0, mut t1) = (a, b, 1i128, 0i128, 0i128, 1i128);
    while r1 != 0 {
        let q = r0.div_euclid(r1);
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if r0 < 0 {
        (-r0, -s0, -t0)
    } else {
        (r0, s0, t0)
    }
}

/// `a + b mod n` for `a, b < n`.
#[inline]
pub fn add_mod(a: u64, b: u64, n: u64) -> u64 {
    let s = a as u128 + b as u128;
    (if s >= n as u128 { s - n as u128 } else { s }) as u64
}

/// `a − b mod n` for `a, b < n`.
#[inline]
pub fn sub_mod(a: u64, b: u64, n: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        n - (b - a)
    }
}

/// `−a mod n` for `a < n`.
#[inline]
pub fn neg_mod(a: u64, n: u64) -> u64 {
    if a == 0 {
        0
    } else {
        n - a
    }
}

/// The residue of a signed integer modulo `n ≥ 1`.
#[inline]
pub fn residue(a: i64, n: u64) -> u64 {
    (a as i128).rem_euclid(n as i128) as u64
}

/// `a b mod n` for `a, b < n` without overflow (`n < 2^64`).  For `n < 2^32`
/// the product fits in a `u64`, and the 64-bit remainder is about twice as
/// fast as the 128-bit one (for a constant modulus, e.g. `Fp<P>`, the
/// branch folds away).
#[inline]
pub fn mul_mod(a: u64, b: u64, n: u64) -> u64 {
    if n <= 1 << 32 {
        a * b % n
    } else {
        (a as u128 * b as u128 % n as u128) as u64
    }
}

/// `a^e mod n` by repeated squaring (`n ≥ 1`).
pub fn pow_mod(a: u64, mut e: u64, n: u64) -> u64 {
    let mut base = a % n;
    let mut acc = 1 % n;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_mod(acc, base, n);
        }
        base = mul_mod(base, base, n);
        e >>= 1;
    }
    acc
}

/// The inverse of `a` modulo `n` (`n ≥ 1`), or `None` when `gcd(a, n) ≠ 1`.
pub fn inv_mod(a: u64, n: u64) -> Option<u64> {
    let (g, s, _) = xgcd((a % n) as i128, n as i128);
    (g == 1).then(|| s.rem_euclid(n as i128) as u64)
}

/// Is `n` prime?  Deterministic Miller–Rabin for all `u64`.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    // these bases are a deterministic witness set for n < 3.3e24
    'witness: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Distinct prime factors of `n ≥ 1`, increasing, by trial division (for the
/// small orders that occur in field and group constructions).
pub fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut f = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            f.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        f.push(n);
    }
    f
}

/// Multiplicative order `n / gcd(n, l)` of the `l`-th power of a generator of
/// a cyclic group of order `n`.
pub fn cyclic_order(n: u64, l: u64) -> u64 {
    n / n.gcd(&l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modular_helpers() {
        for n in 1..60u64 {
            for a in 0..n {
                match inv_mod(a, n) {
                    Some(b) => assert_eq!(mul_mod(a, b, n), 1 % n, "{a}^-1 mod {n}"),
                    None => assert_ne!(a.gcd(&n), 1),
                }
                assert_eq!(pow_mod(a, 3, n), a * a * a % n);
            }
        }
        let (g, s, t) = xgcd(-12, 18);
        assert_eq!((g, -12 * s + 18 * t), (6, 6));
    }

    #[test]
    fn primes() {
        let small: Vec<u64> = (0..100).filter(|&n| is_prime(n)).collect();
        assert_eq!(small.len(), 25);
        assert!(is_prime(1_000_003) && is_prime((1 << 61) - 1) && !is_prime(1_000_001));
        assert_eq!(prime_factors(360), vec![2, 3, 5]);
        assert_eq!(cyclic_order(12, 8), 3);
    }
}
