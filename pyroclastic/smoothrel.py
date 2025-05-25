"""
Utility functions to decompose a prime ideal as a product
of small norm ideals.
"""

import argparse
import logging
import math
import random

import flint
import numpy as np

import pyroclastic_flint_extras as flint_extras
from . import algebra


def hafner_mccurley(D, p, B=None, maxiters=10_000):
    """
    Find a representation of [p] as a smooth ideal using
    Hafner-McCurley algorithm
    """
    base = [l for l, _ in algebra.primebase(D, 2 * maxiters.bit_length() * maxiters)]
    base = base[:maxiters]
    assert flint.fmpz(p).is_prime()
    assert pow(D, p // 2, p) == 1

    forms = [flint_extras.qfb.prime_form(D, l) for l in base]
    unit = forms[0] ** 0

    q0 = flint_extras.qfb.prime_form(D, p)
    # Reduce
    q0 = q0 * unit
    best = flint.fmpz(q0.q()[0]).factor()
    logging.info(f"Found factorization with bound {best[-1][0]}")
    print(best)
    for i, li in enumerate(base):
        q1 = q0 * forms[i]
        f1 = flint.fmpz(q1.q()[0]).factor()
        if f1[-1][0] < best[-1][0]:
            best = f1
            logging.info(f"Found factorization with bound {best[-1][0]}")
            print(best)


def cpu_sieve(D, p, M=None, B=None):
    "Sieve one polynomial to find smooth relations"
    assert flint.fmpz(p).is_prime()
    assert pow(D, p // 2, p) == 1

    PARAMS = [
        # bits, M, B
        (0, 20_000_000, 1_000_000),
        (150, 30_000_000, 3_000_000),
        (200, 40_000_000, 5_000_000),
        (300, 50_000_000, 10_000_000),
        (350, 100_000_000, 40_000_000),
        (400, 200_000_000, 100_000_000),
        (500, 300_000_000, 200_000_000),
    ]
    if M is None or B is None:
        for _bits, _M, _B in PARAMS:
            if _bits < D.bit_length():
                M, B = _M, _B
        logging.info(f"Using parameters {M=} {B=}")

    q0 = flint_extras.qfb.prime_form(D, p)
    q0 = q0 * q0**0
    qA, qB, qC = q0.q()
    logging.info(f"Reduced p to ideal of norm {qA}")

    Af = flint.fmpz(qA).factor()
    Qf = []
    q = q0**0
    for l, e in Af:
        bl = qB % l
        if l == 2 and bl & 3 != 3:
            e = -e
        if l != 2 and bl & 1 != D & 1:
            e = -e
        q = q * flint_extras.qfb.prime_form(D, l) ** e
        Qf.append((l, e))
    assert q.q() == q0.q(), (q0, q)
    # print(q)

    Ai = [l**e for l, e in Af]
    logging.info(f"Prime factors {Ai}")

    base = list(algebra.primebase(D, B))

    q1 = q
    extra = []
    target = math.isqrt(abs(D)) // M
    if qA < target // 2:
        # A is too small
        while 1000 * B * qA < target:
            l = random.choice(base[1000 : 10000])[0]
            q1 = q1 * flint_extras.qfb.prime_form(D, l)
            qA *= l
            extra.append((l, -1))
        if B * qA < 2 * target:
            t = math.isqrt(target // qA)
            it = (_l for _l, _ in base if _l > t)
            l1, l2 = next(it), next(it)
            q1 = q1 * flint_extras.qfb.prime_form(D, l1)
            q1 = q1 * flint_extras.qfb.prime_form(D, l2)
            extra.append((l1, -1))
            extra.append((l2, -1))
        else:
            l = next(_l for _l, _ in base if qA * _l > target)
            q1 = q1 * flint_extras.qfb.prime_form(D, l)
            extra.append((l, -1))
    else:
        # A is too large
        if any(l < 2 * M for l, _ in Af):
            # Remove some factors to get A closer to sqrt(D)/M
            samples = []
            for _ in range(2**16):
                x = 1
                extra = []
                for l, e in Qf:
                    if l < 2 * M and l < B:
                        bit = random.randrange(2)
                        if bit:
                            x *= l ** abs(e)
                            extra.append((l, e))
                samples.append((abs(x - M), x, extra))
            samples.sort()
            _, x, extra = samples[0]
            logging.info(f"Removing factor {x} {extra}")
            for l, e in extra:
                q1 = q1 * flint_extras.qfb.prime_form(D, l) ** -e
    qA, qB, qC = q1.q()
    logging.info(f"Sieving with extra factors {extra}")

    I = np.zeros(2 * M + 1, dtype=np.uint8)
    # don't sieve 4 tiny primes
    for l, r in base[4:]:
        logp = l.bit_length() - 1
        if qA % l == 0:
            if qB % l == 0:
                # no root
                continue
            r1 = (-qC * pow(qB, -1, l) + M) % l
            I[r1::l] += logp
        else:
            ainv = pow(2 * qA, -1, l)
            r1 = ((r - qB) * ainv + M) % l
            r2 = ((-r - qB) * ainv + M) % l
            I[r1::l] += logp
            if r1 != r2:
                I[r2::l] += logp

    logging.info(f"Maximum sieve result {np.max(I)}")
    threshold = np.max(I) * 2 // 3
    candidates = []
    for idx in (I > threshold).nonzero()[0]:
        candidates.append((-int(I[idx]), idx - M))
    candidates.sort()
    logging.info(f"{len(candidates)} best sieve results (score >= {threshold})")

    best = abs(p)
    for _, x in candidates:
        qx = qA * x**2 + qB * x + qC
        rel = []
        u = 2 * qA * x + qB
        q = q0**0
        for l, e in flint.fmpz(qx).factor_smooth(bits=30):
            if not flint.fmpz(l).is_prime():
                # partial factorization, cofactor was too large
                rel = None
                break
            ul = u % l
            if l == 2 and ul & 3 == 3:
                e = -e
            if l > 2 and ul & 1 == D & 1:
                e = -e
            q = q * flint_extras.qfb.prime_form(D, l) ** e
            rel.append((l, e))
        if rel is None:
            continue
        # Add extra factors
        for l, e in extra:
            q = q * flint_extras.qfb.prime_form(D, l) ** e
            rel.append((l, e))
        rel.sort()
        if q.q() != q0.q():
            logging.error(f"INTERNAL ERROR: invalid relation {rel}")
            continue
        if rel[-1][0] < best:
            best = rel[-1][0]
            rel_str = " ".join(f"{l}^{e}" for l, e in rel)
            logging.info(f"Best smoothness bound {best}")
            logging.info(f"Found relation {rel_str}")


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--algo", choices=("hmc", "cpu"), default="cpu")
    argp.add_argument("D", type=int)
    argp.add_argument("p", type=int)
    args = argp.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.algo == "hmc":
        hafner_mccurley(args.D, args.p)
    elif args.algo == "cpu":
        cpu_sieve(args.D, args.p)


if __name__ == "__main__":
    main()
