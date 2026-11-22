// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: 2026 Rémy O

use num_bigint::BigInt;
use num_traits::{sign::Signed, ToPrimitive};

pub(crate) fn smallprimes(b: u32) -> Vec<u32> {
    // sieve[i] says that 2i+1 is composite
    let mut sieve = vec![false; b as usize / 2];
    let approxpi = (3 * b) / (2 * b.ilog2());
    let mut primes = Vec::with_capacity(approxpi as usize);
    primes.push(2);
    for i in 1..sieve.len() {
        if !sieve[i] {
            let p = 2 * i + 1;
            primes.push(p as u32);
            if p as u64 * p as u64 > b as u64 {
                // Push all remaining primes
                for j in (i + 1)..sieve.len() {
                    if !sieve[j] {
                        primes.push((2 * j + 1) as u32);
                    }
                }
                break;
            }
            // Mark p², p(p + 2), etc.
            let mut k = (p * p) / 2;
            while k < sieve.len() {
                sieve[k] = true;
                k += p
            }
        }
    }
    primes
}

/// Return a list of split primes and canonical roots
/// such r has the same parity as d.
fn primebase(d: BigInt, b: u32) -> Vec<u32> {
    let ps = smallprimes(b);
    let mut res = vec![];
    res.push(2);
    //res.push((2, u32::from(d.bit(0))));
    for &p in &ps[1..] {
        let dmodp: i64 = (&d % p).try_into().unwrap();
        if legendre_symbol(dmodp, p as i64) == 1 {
            res.push(p);
        }
    }
    res
}

#[test]
fn test_smallprimes() {
    let ps = smallprimes(1_000_000);
    assert_eq!(ps.len(), 78498);
    assert_eq!(ps.last(), Some(&999983));

    let ps = smallprimes(10_000_000);
    assert_eq!(ps.len(), 664579);
    assert_eq!(ps.last(), Some(&9999991));
}

#[test]
fn test_primebase() {
    let d = BigInt::parse_bytes(b"-123456789_123456789_123456811", 10).unwrap();
    let prs = primebase(d, 1000_000);
    assert_eq!(prs.len(), 39388);
    assert!(matches!(prs.last(), Some(999961)));
}

// Integer division is fast on some platforms like AMD Zen 4.
const USE_EUCLID: bool = true;

/// Compute the Legendre symbol (x|p) where p is a positive odd integer.
pub(crate) fn legendre_symbol(x: i64, p: i64) -> i32 {
    // See https://en.wikipedia.org/wiki/Jacobi_symbol
    assert!(p > 0 && p & 1 == 1);
    let mut sq = 0; // Result is (-1)^sq
    let mut x = x;
    let mut n = p;
    if x < 0 {
        if n & 3 == 3 {
            sq += 1;
        }
        x = -x;
    }
    if x == 0 {
        return 0;
    }
    if p == 1 {
        return 1;
    }
    assert!(x >= 0);
    loop {
        let tz = x.trailing_zeros();
        // (2|n) = -1 iff n%8 is 3 or 5
        sq += tz & ((n & 7).count_ones() == 2) as u32;
        x >>= tz;
        // Invariant: x and n are odd
        if x == 1 {
            return if sq & 1 == 0 { 1 } else { -1 };
        }
        // Invariant: x > 1 and n > 1
        if USE_EUCLID {
            // Euclid algorithm
            if x & n & 3 == 3 {
                // Apply quadratic reciprocity
                sq += 1;
            }
            (x, n) = (n % x, x);
        } else {
            // Binary GCD style
            if x < n {
                // Apply quadratic reciprocity
                if x & n & 3 == 3 {
                    sq += 1;
                }
                (x, n) = (n - x, x);
            } else {
                (x, n) = (x - n, n);
            }
        }
        if x == 0 {
            return 0;
        }
    }
}

#[test]
fn test_legendre() {
    for p in [
        63977,
        2500213,
        2500363,
        300 * 1024 + 1,
        (7 << 20) + 1,
        (13 << 20) + 1,
    ] {
        for i in 0..1000 {
            let x: u64 = 65537 * i + 1337;
            let x2 = (x * x) % p;
            assert_eq!(legendre_symbol(x2 as i64, p as i64), 1);
        }
    }
    // Not squares
    for p in [19774193805679, 65161807337651, 12631785623819] {
        for i in 0..1000 {
            let x: u64 = 65537 * i + 1337;
            let x2 = p - (x * x) % p;
            assert_eq!(legendre_symbol(x2 as i64, p as i64), -1);
        }
    }
    // Multiples
    for p in [63977, 2500213, 2500363] {
        for i in 20..40 {
            for j in 70..90 {
                assert_eq!(legendre_symbol((i * p) as i64, (2 * j + 1) * p as i64), 0);
            }
        }
    }
}

/// Compute an approximation of the class number of Q[sqrt(d)]
/// (quadratic imaginary field) using partial Euler products.
///
/// The prime bound is a u32 and the resulting precision is
/// usually more than a f32 ulp.
pub fn h_approx(d: &BigInt, b: u32) -> f64 {
    assert!(d.sign() == num_bigint::Sign::Minus);
    // The formula is: h(D) = sqrt(|D|)/π * Residue(L)
    // where log Residue = sum_p -log(1 - (D|p)/p)
    let ps = smallprimes(b);
    let mut minuslogres: f32 = 0.0;
    // Iterate large primes first to avoid rounding issues
    for &p in ps[1..].iter().rev() {
        let mut pinv: f32 = 1. / (p as f32);
        let dmodp: i64 = (d % p).to_i64().unwrap();
        match legendre_symbol(dmodp, p as i64) {
            0 => continue,
            1 => pinv = -pinv,
            _ => {}
        }
        minuslogres += pinv.ln_1p();
    }
    // Process 2 separately.
    if d.bit(0) {
        let dlow: u32 = (d & BigInt::from(7)).to_u32().unwrap();
        if dlow == 1 {
            // log(1 - 1/2) = -log(2)
            minuslogres -= std::f32::consts::LN_2;
        } else {
            // log(1 + 1/2)
            minuslogres += (1.5_f32).ln();
        }
    }

    let res = (-minuslogres).exp() as f64;
    let dfloat: f64 = d.abs().to_f64().unwrap();
    dfloat.sqrt() / std::f64::consts::PI * res
}

#[test]
fn test_h_approx() {
    let d = BigInt::parse_bytes(
        b"-1139325066844575699589813265217200398493708241839938355464231",
        10,
    )
    .unwrap();
    let h = h_approx(&d, 1_000_000);
    let err = (h / 964415698883565364637432450736.).ln();
    eprintln!("h_approx = {h}");
    assert!(err.abs() < 0.001);

    let d = BigInt::parse_bytes(
        b"-12239807779826253214859975412431303497371919444169932188160735019",
        10,
    )
    .unwrap();
    let h = h_approx(&d, 1_000_000);
    let err = (h / 109997901313565058259819609742265.).ln();
    eprintln!("h_approx = {h}");
    assert!(err.abs() < 0.001);

    let d = BigInt::parse_bytes(
        b"-40000000000000000000000000000000000000000000000000000000000000004",
        10,
    )
    .unwrap();
    let h = h_approx(&d, 1_000_000);
    let err = (h / 178397819605839608466892693850112.).ln();
    eprintln!("h_approx = {h}");
    assert!(err.abs() < 0.001);

    const CLASSNUMBERS: &[(&'static str, i64)] = &[
        // Precomputed class numbers.
        ("-999999999999999999999999999995", 508747295920704),
        ("-999999999999999999999999999991", 640872173075648),
        ("-999999999999999999999999999987", 179640070526784),
        ("-999999999999999999999999999983", 1075313728713491),
        ("-999999999999999999999999999979", 183573155984512),
        ("-999999999999999999999999999971", 422744623863104),
        ("-999999999999999999999999999967", 494628187108094),
        ("-999999999999999999999999999959", 1175964567336172),
        ("-999999999999999999999999999955", 155020826134944),
        ("-999999999999999999999999999951", 890836155198816),
        ("-999999999999999999999999999943", 514034639170776),
        ("-999999999999999999999999999939", 324434791807488),
        ("-999999999999999999999999999935", 1680237466100604),
        ("-999999999999999999999999999931", 159863243256280),
        ("-999999999999999999999999999923", 217523256291120),
        ("-999999999999999999999999999919", 511489179990784),
        ("-999999999999999999999999999911", 1262988595262320),
        ("-999999999999999999999999999907", 157736548757760),
        ("-999999999999999999999999999903", 376230733301496),
        ("-999999999999999999999999999899", 630315186659040),
    ];
    for &(dstr, hexact) in CLASSNUMBERS {
        // Check that error is ~1/sqrt(B)
        let d = BigInt::parse_bytes(dstr.as_bytes(), 10).unwrap();
        let h = h_approx(&d, 1000);
        assert!(
            (h / hexact as f64).ln().abs() < 0.03,
            "approx={h} exact={hexact}"
        );
        let h = h_approx(&d, 10000);
        assert!((h / hexact as f64).ln().abs() < 0.01);
        let h = h_approx(&d, 100_000);
        assert!((h / hexact as f64).ln().abs() < 0.003);
        let h = h_approx(&d, 1000_000);
        assert!((h / hexact as f64).ln().abs() < 0.001);
    }
}
