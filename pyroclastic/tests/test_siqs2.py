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
    POLYS_PER_WG = 2
    INTERVAL_SIZE = 1024
    M = INTERVAL_SIZE // 2

    # 240 bit
    N = 3124353551174324859909941623684171879248927282143169485502905245672938703
    B1 = 300_000
    B2 = 15 * B1
    OUTSTRIDE = 64 * 1024
    EXTRA_THRESHOLD = 15

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
    AFACS = 19
    # Make output deterministic
    random.seed(42)
    As = sieve.make_a(primes, A0, AFACS)
    A, Bi = sieve.make_poly(D, As[0], roots_d)

    BLEN = (A.bit_length() + 36) // 32
    WORKCHUNK = 2 ** (AFACS - 1)
    logging.debug(f"{AFACS} factors per A, {BLEN} words per coefficient")

    siever = sieve.Siever2(
        {
            "primes": primes,
            "roots": roots,
            "D": D,
            "B1": B1,
            "B2": B2,
            "AFACS": AFACS,
            "BLEN": BLEN,
            "POLYS_PER_WG": POLYS_PER_WG,
            "INTERVAL_SIZE": INTERVAL_SIZE,
            "THRESHOLD": THRESHOLD,
            "OUTSTRIDE": OUTSTRIDE,
            "DEBUG": 1,
        }
    )
    HUGE_PRIME = siever.defines["HUGE_PRIME"]

    ak = As[0]
    dt = siever._run(ak, A, Bi)
    BLOCKSIZE = 2 ** (AFACS - 1) * INTERVAL_SIZE
    speed = BLOCKSIZE / dt
    print(
        f"Sieved {BLOCKSIZE / 1e6:.0f}M (interval {INTERVAL_SIZE >> 10}, {2 ** (AFACS - 1)} polys) in {dt * 1000:.3f}ms, speed {speed / 1e9:.1f}G/s"
    )

    polys = sieve.expand_polys(D, A, Bi)
    assert len(polys) == 1 << (AFACS - 1)

    QUICK = True

    print("=> Checking roots")
    xr = siever.shader2.get_tensors()[2]
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xr])).eval()
    proots = (
        xr.data()
        .astype(np.int32)
        .reshape((len(primes), 1 + 16 + 16 + 16 + 16 + 2 * 2 ** (AFACS // 2 - 8)))
    )
    AHALF = AFACS // 2
    assert AFACS == 2 * AHALF + 1

    mid = 1 + 16 + 16 + 2 ** (AHALF - 8)

    def poly_roots(poly_idx, prime_idx):
        j, jhi = poly_idx, poly_idx >> AHALF
        l1, l2, l3 = j % 16, (j >> 4) % 16, (j >> 8) % (2 ** (AHALF - 8))
        h1, h2, h3 = jhi % 16, (jhi >> 4) % 16, jhi >> 8
        rs = [int(r) for r in proots[prime_idx, :]]
        # Unit testing
        r1 = (
            rs[0]
            + rs[1 + l1]
            - rs[17 + l2]
            - rs[33 + l3]
            - rs[mid + h1]
            - rs[mid + 16 + h2]
            - rs[mid + 32 + h3]
            - M
        )
        r2 = r1 - 2 * rs[0]
        return r1, r2

    def check_roots(j, A, B, C):
        # print("Check polynomial", A, B, C)
        for i, p in enumerate(primes):
            if p == 2 or A % p == 0:
                continue
            r1, r2 = poly_roots(j, i)
            assert roots_d[p] == (2 * A * int(proots[i, 0])) % p
            if (A * r1**2 + B * r1 + C) % p != 0:
                print(f"poly[{j}] wrong root at prime {p}")
            if (A * r2**2 + B * r2 + C) % p != 0:
                print(f"poly[{j}] wrong root at prime {p}")

    for j, (A, B, C) in enumerate(polys):
        if j % 997 != 0:
            continue
        if QUICK and j % 3:
            continue
        check_roots(j, A, B, C)
    print(f"=> ok for {len(polys)} polynomials")

    print("=> Checking sort output")
    xsort = siever.shader3.get_tensors()[-2]
    xidx = siever.shader3.get_tensors()[-1]
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xsort, xidx])).eval()
    vsort = xsort.data().reshape((len(primes), 2**AHALF, 2))
    vidx = xidx.data().reshape((len(primes), 2**AHALF))
    # print(vsort)
    # print(vidx)

    # Check sorted roots
    mid = 1 + 16 + 16 + 2 ** (AHALF - 8)

    def check_sort(j, l):
        plen = l.bit_length()
        # Check values
        rs = [int(r) for r in proots[j, :]]
        _seen = np.zeros(2**AHALF, dtype=np.uint8)
        for idx in range(2**AHALF):
            v = vsort[j, idx, 1]
            poly = vsort[j, idx, 0]
            assert poly >> AHALF == 0
            l1, l2, l3 = poly % 16, (poly >> 4) % 16, poly >> 8
            assert v == (rs[mid + l1] + rs[mid + 16 + l2] + rs[mid + 32 + l3]) % l, (
                j,
                l,
                idx,
            )
            _seen[poly] = 1
        assert np.all(_seen == 1)
        for i in range(2**AHALF):
            start, end = vidx[j, i], vidx[j, i + 1] if i + 1 < 2**AHALF else 0
            if start != 0 and end == 0:
                end = 2**AHALF
            if start == end:
                continue
            assert np.all(i == vsort[j, start:end, 1] >> (plen - AHALF)), (
                i,
                vsort[j, start:end, 1],
                vsort[j, start:end, 1] >> (plen - AHALF),
            )

    for j, l in enumerate(primes):
        if j < HUGE_PRIME:
            continue
        if QUICK and j % 5:
            continue
        check_sort(j, l)
    print(f"=> ok for {len(primes)} primes")

    SHARDS = sieve.SHARDS
    XXL_PRIME = siever.defines["XXL_PRIME"]

    print("=> Checking prefill")
    pxxl = primes[XXL_PRIME]
    fillratio = math.log(math.log(primes[-1])) - math.log(math.log(pxxl))
    expected_per_shard = WORKCHUNK * 2 * M * fillratio / SHARDS
    print(f"Expected per shard {expected_per_shard:.0f} ratio {100 * fillratio:.1f}%")

    xfill = siever.shader3.get_tensors()[2]
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xfill])).eval()
    bfill = xfill.data()
    bfill = bfill.reshape((SHARDS, len(bfill) // SHARDS))
    print("range shard length", np.min(bfill[:, 0]), np.max(bfill[:, 0]))
    print("total", np.sum(bfill[:, 0]))
    for i in range(16):
        assert 0.9 < bfill[i, 0] / expected_per_shard < 1.1
        print(bfill[i, : bfill[i, 0]])

    for j, (A, B, C) in enumerate(polys):
        if j % 997:
            continue
        if QUICK and j % 17:
            continue
        shard_idx = j % SHARDS
        row = set(int(x) for x in bfill[shard_idx, : bfill[shard_idx, 0]])
        # print("Check polynomial", A, B, C)

        missing = 0
        expected = 0
        for ix, p in enumerate(primes[XXL_PRIME:]):
            i = ix + XXL_PRIME
            # Unit testing
            r1, r2 = poly_roots(j, i)
            r1 = (M + r1) % p
            r2 = (M + r2) % p

            plog = min(max(0, p.bit_length() - 15), 7)
            if r1 < 2 * M:
                expected += 1
                v1 = plog + 8 * (r1 + INTERVAL_SIZE * (j >> 8))
                if v1 not in row:
                    logging.error(f"Missing {p=} r={r1} value={v1}")
                    missing += 1
            if r2 < 2 * M:
                expected += 1
                v2 = plog + 8 * (r2 + INTERVAL_SIZE * (j >> 8))
                if v2 not in row:
                    logging.error(f"Missing {p=} r={r2} value={v2}")
                    missing += 1
        if missing:
            logging.error(
                f"Missed {missing} values / expected {expected} / seen {bfill[shard_idx, 0]}"
            )
        print(f"poly {j} ok")
    print("=> end")

    print("=> Checking huge buckets")
    BUCKETS = WORKCHUNK * INTERVAL_SIZE // sieve.BUCKET_INTERVAL
    BUCKET_SIZE = siever.defines["BUCKET_SIZE"]
    xhuge = siever.shader5.get_tensors()[1]
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xhuge])).eval()
    bhuge = xhuge.data().view(np.uint16)
    bhuge = bhuge.reshape((BUCKETS, BUCKET_SIZE))
    print("buckets", bhuge.shape)
    print(bhuge[:, 0])

    for j, (A, B, C) in enumerate(polys):
        bucks = bhuge[j % BUCKETS, :]
        if j % 997:
            continue
        if QUICK and j % 3:
            continue
        print("size", bucks[0], bucks[1 : bucks[0]])
        # print("Check polynomial", A, B, C)
        missing = 0
        expected = 0
        for ix, p in enumerate(primes[XXL_PRIME:]):
            i = ix + XXL_PRIME
            # Unit testing
            r1, r2 = poly_roots(j, i)
            r1 = (M + r1) % p
            r2 = (M + r2) % p

            plog = min(max(0, p.bit_length() - 15), 7)
            if r1 < 2 * M:
                expected += 1
                v1 = plog + 8 * (r1 + INTERVAL_SIZE * (j // BUCKETS))
                if v1 not in bucks[1 : bucks[0] + 1]:
                    # logging.error(f"Missing {p=} r={r1} bucket={b1} value={v1}")
                    missing += 1
            if r2 < 2 * M:
                expected += 1
                v2 = plog + 8 * (r2 + INTERVAL_SIZE * (j // BUCKETS))
                if v2 not in bucks[1 : bucks[0] + 1]:
                    # logging.error(f"Missing {p=} r={r2} bucket={b2} value={v2}")
                    missing += 1
        if missing:
            reported = np.sum(bucks[0])
            logging.error(
                f"Missed {missing} values / expected {expected} / reported {reported}"
            )
        print(f"poly {j} ok")
    print("=> end")

    xout = siever.shader6.get_tensors()[-2]
    xdebug = siever.shader6.get_tensors()[-1]
    vout = xout.data().astype(np.int32)
    vdebug = xdebug.data().view(np.uint8).reshape((WORKCHUNK, INTERVAL_SIZE))
    siever.mgr.sequence().record(kp.OpTensorSyncLocal([xdebug])).eval()
    total_expected = 0
    total_observed = 0
    print(vout)
    assert vout[0] < OUTSTRIDE
    for _i in range(2 ** (AFACS - 1)):
        if _i % 97:
            continue
        if QUICK and _i % 7:
            continue

        _A, _B, _C = polys[_i]
        print("Check polynomial", _A, _B, _C)
        expected = sieve_simple(_A, _B, _C, M, primes, roots, THRESHOLD)
        if expected:
            print("Expect", expected)
        sieve_result = vdebug[_i, :].astype(np.uint32)
        # print(sieve_result)
        # print(np.sum(sieve_result))
        seen = set()
        for oidx in range(vout[0]):
            x = vout[oidx]
            if x // INTERVAL_SIZE != _i:
                continue
            x = (x % INTERVAL_SIZE) - M
            seen.add(x)
            v = _A * x * x + _B * x + _C
            rel = sieve.build_relation(v, x, [], B1=B1, B2=10**9)
            if x not in expected:
                print(
                    f"Unexpected poly[{_i}]({x})",
                    flint.fmpz(v).factor(),
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
