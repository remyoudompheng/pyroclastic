"""
Utility functions to decompose a prime ideal as a product
of small norm ideals.
"""

import argparse
import logging
import math
import pathlib
import random

import flint
import numpy as np

import pyroclastic_flint_extras as flint_extras
from . import algebra
from . import sieve


def buchmann_mccurley(D, p, B=None, maxiters=10_000):
    """
    Find a representation of [p] as a smooth ideal using
    Buchmann-McCurley algorithm
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


def factor_q(D: int, q) -> list[tuple[int, int]]:
    """
    Factor an ideal represented by a binary quadratic form
    """
    A, B, C = q.q()
    Af = flint.fmpz(A).factor()
    Qf = []
    qf = q**0
    for l, e in Af:
        bl = B % l
        if l == 2 and bl & 3 != 3:
            e = -e
        if l != 2 and bl & 1 != D & 1:
            e = -e
        qf = qf * flint_extras.qfb.prime_form(D, l) ** e
        Qf.append((int(l), int(e)))
    assert qf.q() == q.q()
    return Qf


def product_q(D: int, factors: list[tuple[int, int]]):
    q = flint_extras.qfb.prime_form(D, 1)
    for l, e in factors:
        q *= flint_extras.qfb.prime_form(D, l) ** e
    return q


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
    Qf = factor_q(D, q0)

    Ai = [l**e for l, e in Af]
    logging.info(f"Prime factors {Ai}")

    base = list(algebra.primebase(D, B))

    q1 = q0
    extra = []
    target = math.isqrt(abs(D)) // M
    if qA < target // 2:
        # A is too small
        while 1000 * B * qA < target:
            l = random.choice(base[1000:10000])[0]
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


def gpu_sieve(D, p, dlogs: dict[int, any] | None = None):
    """
    Find a relation by running SIQS with a forced factor.

    If the input prime is too large, a first reduction pass will reduce it
    to a smaller prime.
    """
    assert flint.fmpz(p).is_prime()
    assert pow(D, p // 2, p) == 1

    B1, B2k, OUTSTRIDE, EXTRA_THRESHOLD, AFACS, ITERS, POLYS_PER_WG = sieve.get_params(
        D, sieve.smoothness_bias(D)
    )
    B2 = B2k * B1
    M = ITERS * sieve.SEGMENT_SIZE // 2
    THRESHOLD = (
        D.bit_length() // 2 + M.bit_length() - 2 * B1.bit_length() - EXTRA_THRESHOLD
    )

    sieve_params = sieve.get_params(D)
    B1 = sieve_params[0]
    logging.info(f"Small prime bound {B1=}")

    def is_small(l):
        if dlogs:
            return l in dlogs
        else:
            return l < B1

    ls, roots = [], []
    for l, lr in algebra.primebase(D, B1):
        ls.append(l)
        roots.append(lr)

    q0 = flint_extras.qfb.prime_form(D, p) * flint_extras.qfb.prime_form(D, 1)
    best_large = None
    best_rel = None
    for l in [1] + ls:
        if l == 2:
            continue
        if l == 1:
            q = q0
        else:
            q = q0 * flint_extras.qfb.prime_form(D, l)
        qA = q.q()[0]
        Af = sorted(flint.fmpz(qA).factor())
        # We want the largest factor to be smaller than D^1/4
        # and the second largest factor to be among the sieving primes (smallest)
        if any(_l.bit_length() > D.bit_length() // 6 for _l, _ in Af):
            continue

        larges = [_l for _l, _ in Af if not is_small(_l)]
        is_better = (
            best_large is None
            or len(larges) < len(best_large)
            or (len(larges) == len(best_large) and tuple(reversed(larges)) < best_large)
        )
        if is_better:
            best_large = tuple(reversed(larges))
            best_rel = (q, l)
            logging.info(f"Good decomposition {[(l, -1)] + factor_q(D, q)}")

        if len(best_large) <= 2:
            break

    ql, l = best_rel
    qlfacs = [(l, -1)] + factor_q(D, ql)
    assert q0 == product_q(D, qlfacs)
    logging.info(f"Reduced to {qlfacs}")

    result = {}
    for l, e in qlfacs:
        if is_small(l):
            result.setdefault(l, 0)
            result[l] += e
            continue
        root = flint_extras.sqrtmod(D, l)
        # Normalize so that root has the same parity as N
        if root & 1 != D & 1:
            root = l - root

        # Sieve to decompose l into small factors.
        # Polynomials will use A = l * product(ai)
        logging.info(f"Computing reduction of l={l}")

        A0 = math.isqrt(abs(D)) // (2 * M)
        BLEN = max(2, (A0.bit_length() + 36) // 32)
        As = sieve.make_a(ls, A0 // l, AFACS - 1)
        WARGS = {
            "primes": ls + [l],
            "roots": roots + [root],
            "D": D,
            "B1": B1,
            "B2": B2,
            "AFACS": AFACS,
            "BLEN": BLEN,
            "POLYS_PER_WG": POLYS_PER_WG,
            "ITERS": ITERS,
            "THRESHOLD": THRESHOLD,
            "OUTSTRIDE": OUTSTRIDE,
        }
        siever = sieve.Siever(WARGS)
        lfactors = None
        for ak in As:
            dt, nreports, rels = siever.process([l] + list(ak))
            for r in rels:
                assert l in r and r.count(l) == 1
                if all(is_small(abs(_l)) for _l in r if _l != l):
                    rel = [-x for x in r if x != l]
                    logging.info(f"Found good relation {l} = {rel}")
                    lfactors = rel
                    break
            if lfactors:
                break

        ql = product_q(D, [(abs(x), 1 if x > 0 else -1) for x in lfactors])
        assert ql == flint_extras.qfb.prime_form(D, l)

        for x in lfactors:
            x, ex = abs(x), 1 if x > 0 else -1
            result.setdefault(x, 0)
            result[x] += e * ex

    qf = product_q(D, list(result.items()))
    assert q0 == qf

    result = {l: e for l, e in result.items() if e}
    relstr = " ".join(f"{x}^{e}" for x, e in sorted(result.items()))
    logging.info(f"Found final decomposition {relstr}")
    return result


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--algo", choices=("bmc", "cpu", "gpu"), default="gpu")
    argp.add_argument("--datadir")
    argp.add_argument("D", type=int)
    argp.add_argument("p", type=int)
    args = argp.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    dlogs = None
    if args.datadir:
        dlogf = pathlib.Path(args.datadir) / "group.structure.extra"
        if dlogf.is_file():
            dlogs = {}
            with open(dlogf) as f:
                for line in f:
                    l = line.split()[0]
                    if l.isdigit():
                        dlogs[int(l)] = line
            logging.info(f"Read {len(dlogs)} coordinates from {dlogf}")

    if args.algo == "bmc":
        buchmann_mccurley(args.D, args.p)
    elif args.algo == "cpu":
        cpu_sieve(args.D, args.p)
    elif args.algo == "gpu":
        gpu_sieve(args.D, args.p, dlogs=dlogs)


if __name__ == "__main__":
    main()
