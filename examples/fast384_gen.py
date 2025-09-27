"""
Generate examples for fast384
"""

import bisect
import math
import random
from multiprocessing import Pool, Value

import numpy as np
import flint
from pyroclastic import algebra


primes = {}
logprimes = {}

MAXSMALL = 284
BIAS0 = 1 + sum(math.log2(p) / float(p) for p in algebra.smallprimes(MAXSMALL) if p > 2)


def smoothness_bias(D: int, bound: int):
    """
    The average amount of extra bits in the smooth part
    """
    assert D % 4 == 1
    DD = flint.fmpz(D)
    if bound not in primes:
        # Skip all primes < MAXSMALL
        primes[bound] = [l for l in algebra.smallprimes(bound) if l > MAXSMALL]
        logprimes[bound] = [math.log2(p) / float(p) for p in primes[bound]]
    b = BIAS0
    if D % 8 != 1:
        b -= 1
    for p, lp in zip(primes[bound], logprimes[bound]):
        if p < 3:
            continue
        legendre = DD.jacobi(p)
        if legendre == 0:
            # not a prime
            return None
        if legendre == 1:
            b += lp
        elif legendre == -1:
            b -= lp
    return b


def classgroup_bias(D: int, bound: int):
    ps = algebra.smallprimes(bound)
    pf = np.array(ps, dtype=np.float32)
    pf = 1.0 / pf

    DD = flint.fmpz(D)
    for i, p in enumerate(ps):
        if p == 2:
            if D & 1 == 0:
                pf[i] = 0
            elif (D % 8) == 1:
                pf[i] = -pf[i]
            continue

        dp = int(DD.jacobi(p))
        pf[i] *= -dp

    plog = -np.log1p(pf)

    lsum = np.cumsum(plog)
    l = np.average(lsum[-len(ps) // 4 :])
    return math.exp(l) / math.pi


def product(l: list[int]):
    p = l[0]
    for x in l[1:]:
        p *= x
    return p


def crt_basis(mod):
    M = product(mod)
    basis = []
    for i, mi in enumerate(mod):
        basis.append(M // mi * pow(M // mi, -1, mi))
    return basis


# All primes below 345

# fmt:off
ps1 = [8, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
       61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
       131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
       211, 223, 227, 229, 233]
ps2 = [239, 241, 251, 257, 263]
ps3 = [269, 271, 277, 281, 283]
# fmt:on

# Prepare CRT basis
P1 = product(ps1)
P2 = product(ps2)
P3 = product(ps3)
P23 = P2 * P3
C2 = P3 * pow(P3, -1, P2)
C3 = P2 * pow(P2, -1, P3)

TARGET = int(2**384 / 3.60**2 / P1)
TARGET_MIN = int(TARGET * 0.99)
TARGET_MAX = int(TARGET * 1.01)

# Admissible values m modulo p2 and p3, such that (1 - P1 m) is a square.
crt2 = crt_basis(ps2)
res2 = []
for p in ps2:
    # x^2 = (1 - P1 m) iff m = (1 - x^2)/P1
    resp = [(1 - i * i) * pow(P1, -1, p) % p for i in range(1, p // 2 + 1)]
    res2.append(resp)

crt3 = crt_basis(ps3)
res3 = []
for p in ps3:
    resp = [(1 - i * i) * pow(P1, -1, p) % p for i in range(1, p // 2 + 1)]
    res3.append(resp)

SAMPLES = 1_000_000

# Generate many interesting residues modulo P2
set2 = []


def sample2(_):
    set2 = []
    for _ in range(SAMPLES):
        x2 = sum(random.choice(r2) * c2 for r2, c2 in zip(res2, crt2))
        x2 = (C2 * x2) % P23
        # assert all(flint.fmpz(1 - P1 * x2).jacobi(l) == 1 for l in ps2)
        set2.append((C2 * x2) % P23)
    return set2


with Pool(8) as pool:
    for s2 in pool.map(sample2, list(range(8))):
        set2 += s2
set2.sort()

print("Working on", len(set2), "samples")

best = Value("d", 7.5)


def sample3(_):
    # Now randomly sample squares modulo P3
    for _ in range(SAMPLES):
        x3 = sum(random.choice(r3) * c3 for r3, c3 in zip(res3, crt3))
        x3 = (C3 * x3) % P23
        # assert all(flint.fmpz(1 - P1 * x3).jacobi(l) == 1 for l in ps3)

        t1 = (TARGET_MIN - x3) % P23
        t2 = (TARGET_MAX - x3) % P23
        tidx1 = bisect.bisect(set2, t1)
        tidx2 = bisect.bisect(set2, t2, lo=tidx1, hi=min(len(set2), tidx1 + 100))
        cur_best = best.value
        for y2 in set2[tidx1:tidx2]:
            xy = x3 + y2
            if xy > P23:
                xy -= P23
            p = P1 * xy - 1
            # assert all(flint.fmpz(-p).jacobi(l) == 1 for l in ps1+ps2+ps3)
            score0 = smoothness_bias(-p, 500)
            if score0 is None:
                continue
            if score0 < cur_best - 0.6:
                continue

            score1 = smoothness_bias(-p, 10000)
            if score1 is None:
                continue
            # assert score1 < score0 + 0.6

            if score1 > cur_best - 0.1:
                if not flint.fmpz(p).is_probable_prime():
                    continue

                score = smoothness_bias(-p, 100000)
                # Score cannot improve much
                # assert score - score1 < 0.1
                if score > cur_best - 0.05:
                    with best.get_lock():
                        best.value = max(best.value, score)
                    scoreh = classgroup_bias(-p, 100000)
                    print(f"p=0x{p:x} alpha={score:.3f} h/sqrt(D)={scoreh:.3f}")


with Pool() as pool:
    for _ in pool.map(sample3, list(range(800))):
        pass
