"""
Validation tests for SIQS
"""

import logging
import math
import random

import numpy as np
import kp

import flint
from pyroclastic import algebra
from pyroclastic import gpu
from pyroclastic import sieve


def sieve_simple(A, B, C, M, primes, roots, threshold):
    I = np.zeros(2 * M + 1, dtype=np.uint8)
    for p, r in zip(primes, roots):
        if p < primes[4]:
            # don't sieve 4 tiny primes
            continue
        logp = p.bit_length() - 1
        if A % p == 0:
            r1 = (-C * pow(B, -1, p)) % p
            I[r1::p] += logp
        else:
            ainv = pow(2 * A, -1, p)
            r1 = ((r - B) * ainv + M) % p
            r2 = ((-r - B) * ainv + M) % p
            I[r1::p] += logp
            if r1 != r2:
                I[r2::p] += logp

    good = (I > threshold).nonzero()[0]
    return [idx - M for idx in good]


def to_uvec(x: int, length: int):
    assert x.bit_length() <= 32 * length
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(length)]


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    devname = gpu.device_name()
    logging.debug(f"Running on device {devname}")

    # Use a fixed sieve length 128k
    ITERS = 8
    POLYS_PER_WG = 2
    SEGMENT_SIZE = sieve.SEGMENT_SIZE
    SUBSEGMENT_SIZE = sieve.SUBSEGMENT_SIZE
    M = ITERS * SEGMENT_SIZE // 2

    # 240 bit
    N = 3124353551174324859909941623684171879248927282143169485502905245672938703
    B1 = 50_000
    B2 = 15 * B1
    OUTSTRIDE = 16
    EXTRA_THRESHOLD = 20

    THRESHOLD = (
        N.bit_length() // 2 + M.bit_length() - 2 * B1.bit_length() - EXTRA_THRESHOLD
    )

    D = -abs(N)
    del N

    primes = []
    roots = []
    for p, k in algebra.primebase(D, B1):
        assert (k * k - D) % p == 0
        primes.append(p)
        roots.append(k)
    roots_d = {p: r for p, r in zip(primes, roots)}

    A0 = math.isqrt(abs(D)) // (2 * M)
    AFACS = 10
    # Make output deterministic
    random.seed(42)
    As = sieve.make_a(primes, A0, AFACS)
    A, Bi = sieve.make_poly(D, As[0], roots_d)

    BLEN = (A.bit_length() + 36) // 32
    WORKCHUNK = 2 ** (AFACS - 1)
    logging.debug(f"{AFACS} factors per A, {BLEN} words per coefficient")

    # A bucket contains offsets for a subsegment
    BUCKET_SIZE = 1
    AVG_BUCKET_SIZE = 0
    HUGE_PRIME = len(primes)
    if primes[-1] > 2 * SEGMENT_SIZE:
        HUGE_PRIME = next(idx for idx, p in enumerate(primes) if p > 1.5 * SEGMENT_SIZE)
        if HUGE_PRIME % 512 != 84:
            HUGE_PRIME += (84 - HUGE_PRIME) % 512
            assert HUGE_PRIME % 512 == 84
        phuge = primes[HUGE_PRIME]
        assert phuge > 2**14
        BUCKET_SIZE = int(
            SUBSEGMENT_SIZE * 0.07 * math.log2(primes[-1] / (0.8 * phuge))
        )
        AVG_BUCKET_SIZE = SUBSEGMENT_SIZE * 0.055 * math.log2(primes[-1] / phuge)
        logging.debug(
            f"Huge prime index {HUGE_PRIME} ({phuge}) bucket size %d expect usage %d",
            BUCKET_SIZE,
            int(AVG_BUCKET_SIZE),
        )

    siever = sieve.Siever(
        {
            "primes": primes,
            "roots": roots,
            "D": D,
            "B1": B1,
            "B2": B2,
            "AFACS": AFACS,
            "BLEN": BLEN,
            "POLYS_PER_WG": POLYS_PER_WG,
            "HUGE_PRIME": HUGE_PRIME,
            "BUCKET_SIZE": BUCKET_SIZE,
            "AVG_BUCKET_SIZE": AVG_BUCKET_SIZE,
            "THRESHOLD": THRESHOLD,
            "OUTSTRIDE": OUTSTRIDE,
            "ITERS": ITERS,
            "DEBUG": 1,
        }
    )

    ak = As[0]
    dt = siever._run(ak, A, Bi)
    INTERVAL = ITERS * SEGMENT_SIZE
    BLOCKSIZE = 2 ** (AFACS - 1) * INTERVAL
    speed = BLOCKSIZE / dt
    print(
        f"Sieved {BLOCKSIZE / 1e6:.0f}M (interval {INTERVAL >> 10}, {2 ** (AFACS - 1)} polys) in {dt * 1000:.3f}ms, speed {speed / 1e9:.1f}G/s"
    )

    polys = sieve.expand_polys(D, A, Bi)
    assert len(polys) == 1 << (AFACS - 1)

    xr = siever.algo1.get_tensors()[2]
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xr])).eval()
    proots = (
        xr.data()
        .astype(np.int32)
        .reshape((1 + 16 + 16 + 2 ** (AFACS - 9), len(primes)))
    )
    for j, (A, B, C) in enumerate(polys):
        print("Check polynomial", A, B, C)
        j1, j2, j3 = j % 16, (j >> 4) % 16, j >> 8
        if j % 17 != 0:
            continue
        for i, p in enumerate(primes):
            if p == 2 or A % p == 0:
                continue
            rs = [int(r) for r in proots[:, i]]
            # Unit testing
            r1 = rs[0] - (rs[1 + j1] + rs[17 + j2] + rs[33 + j3])
            r2 = -rs[0] - (rs[1 + j1] + rs[17 + j2] + rs[33 + j3])
            assert roots_d[p] == (2 * A * rs[0]) % p
            if (A * r1**2 + B * r1 + C) % p != 0:
                print(f"poly[{j}] wrong root at prime {p}")
            if (A * r2**2 + B * r2 + C) % p != 0:
                print(f"poly[{j}] wrong root at prime {p}")

    _, _, xout, xfacs = siever.tensors
    xdebug = siever.algo2.get_tensors()[-1]
    vout = xout.data().astype(np.int32)
    vfacs = xfacs.data()
    vdebug = xdebug.data().view(np.uint8).reshape((WORKCHUNK, INTERVAL))
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xdebug])).eval()
    total_expected = 0
    total_observed = 0
    for _i in range(2 ** (AFACS - 1)):
        _A, _B, _C = polys[_i]
        print("Check polynomial", _A, _B, _C)
        expected = sieve_simple(_A, _B, _C, M, primes, roots, THRESHOLD)
        if expected:
            print("Expect", expected)
        sieve_result = vdebug[_i, :]
        seen = set()
        for _j in range(OUTSTRIDE):
            oidx = OUTSTRIDE * _i + _j
            if not vout[oidx] and not vfacs[32 * oidx]:
                break
            _facs = [int(_f) for _f in vfacs[32 * oidx : 32 * oidx + 32] if _f]
            x = vout[oidx]
            seen.add(x)
            v = _A * x * x + _B * x + _C
            rel = sieve.build_relation(v, x, _facs, B1=B1, B2=10**9)
            if x not in expected:
                print(
                    f"Unexpected poly[{_i}]({x})",
                    flint.fmpz(v).factor(),
                    "reported facs",
                    _facs,
                    "score",
                    sieve_result[x + M],
                )
            else:
                print(f"poly[{_i}]({x})", rel and rel[1:], "score", sieve_result[x + M])
            if sieve_result[x + M] >= THRESHOLD:
                total_observed += 1

        for e in expected:
            if e not in seen:
                v = _A * e * e + _B * e + _C
                print(
                    "Missing",
                    f"poly[{_i}]({e})",
                    flint.fmpz(v).factor(),
                    f"sieve result {sieve_result[e + M]} <= {THRESHOLD}",
                )
        total_expected += len(expected)

    print(
        f"Total reports: {total_observed} observed / {total_expected} expected / sieve region {BLOCKSIZE / 1e6:.0f}M"
    )
    assert 0.9 < total_observed / total_expected < 1.1


if __name__ == "__main__":
    main()
