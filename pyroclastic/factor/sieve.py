"""
Quadratic Sieve for factoring

Given an input number N (positive integer), construct
a positive discriminant D=kN with interesting smoothness bias.

Then run SIQS for this discriminant.
"""

import argparse
import json
import logging
import math
from multiprocessing import Pool, Semaphore
import os
import time

import numpy.typing as npt

from pyroclastic import algebra
from pyroclastic import relations
from pyroclastic import sieve


def multiplier(N: int) -> int:
    """
    >>> multiplier(1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139)
    139
    """
    if N % 4 == 1:
        best, bestscore = 1, sieve.smoothness_bias(N)
    else:
        best, bestscore = 4, sieve.smoothness_bias(4 * N)
    for k in range(1, 200):
        if k * N % 8 not in (0, 4, 1, 5):
            continue
        score = sieve.smoothness_bias(k * N) - math.log2(float(k)) / 2
        if score > bestscore:
            best, bestscore = k, score
    return best, bestscore


class Siever(sieve.Siever):
    def process_sieve_reports(self, ABi, vout, vfacs, N, B1, B2, OUTSTRIDE):
        reports = 0
        results = []
        A, ak, Bi = ABi
        for _i in range(1 << (len(Bi) - 1)):
            poly = None
            for _j in range(OUTSTRIDE):
                oidx = OUTSTRIDE * _i + _j
                if not vout[oidx] and not vfacs[32 * oidx]:
                    break
                if poly is None:
                    poly = sieve.expand_one_poly(N, A, Bi, _i)
                _A, _B, _C = poly
                reports += 1
                x = int(vout[oidx])
                _facs = [int(_f) for _f in vfacs[32 * oidx : 32 * oidx + 32] if _f]
                v = _A * x * x + _B * x + _C
                u = 2 * _A * x + _B
                assert u * u == 4 * A * v + N
                # In the class group: (A, B, C) * (v, u, A) == 1
                row = sieve.build_relation(v, x, _facs, B1=B1, B2=B2)
                if row is None or any(_r > B2 for _r in row):
                    # factors too large
                    continue

                mod = N >> 2 if N & 3 == 0 else N
                if u & 1:
                    u = (u + N) >> 1
                else:
                    u = u >> 1
                assert (u * u - A * v) % mod == 0

                row = [u] + list(ak) + row
                results.append(row)

        return reports, results


class Siever2(sieve.Siever2):
    def _process_sieve_reports(
        self, ABi: tuple, vout: npt.NDArray, N: int, B1: int, B2: int, OUTSTRIDE: int
    ):
        INTERVAL = self.wargs["INTERVAL_SIZE"]
        results = []
        A, ak, Bi = ABi
        nout = int(vout[0])
        if nout >= len(vout):
            logging.error(f"output buffer too small {nout}/{len(vout)}")
        futures = []
        for oidx in range(min(len(vout) - 1, nout)):
            v = int(vout[oidx + 1])
            poly_idx = v // INTERVAL
            x = v % INTERVAL - INTERVAL // 2
            # FIXME: RADV driver for AMD seems to create false positives?
            # if x == 0:
            #     continue
            if x == 0:
                continue
            _A, _B, _C = sieve.expand_one_poly(N, A, Bi, poly_idx)
            v = _A * x * x + _B * x + _C
            u = 2 * _A * x + _B
            assert u * u == 4 * A * v + N
            # In the class group: (A, B, C) * (v, u, A) == 1
            row_async = self.factorpool.submit(
                sieve.build_relation, v, x, [], B1=B1, B2=B2
            )
            futures.append((_B, u, v, row_async))
        del u, v, x, _A, _B, _C, poly_idx

        for B, u, v, row_async in futures:
            row = row_async.result()
            # print(poly_idx, x, v, row)
            if row is None or any(_r > B2 for _r in row):
                # factors too large
                continue
            mod = N >> 2 if N & 3 == 0 else N
            if u & 1:
                u = (u + N) >> 1
            else:
                u = u >> 1
            assert (u * u - A * v) % mod == 0
            row = [u] + list(ak) + row
            results.append(row)
        return nout, results


SIEVER = None


def worker_init(initargs):
    global SIEVER
    if initargs["USE_SIEVER2"]:
        SIEVER = Siever2(initargs)
    else:
        SIEVER = Siever(initargs)


def worker_task(ak):
    return SIEVER.process(ak)


PARAMS = (
    # bitsize, B1, B2/B1, EXTRA_THRESHOLD, AFACS, INTERVAL_SIZE
    # FIXME: reduce EXTRA_THRESHOLD for discrete GPU
    # Simple polynomial initialization
    (120, 6000, 10, 0, 5, 1 * 16384),
    (140, 10_000, 10, 0, 6, 1 * 16384),
    (160, 15_000, 10, 0, 8, 1 * 16384),
    (180, 25_000, 10, 0, 9, 1 * 16384),
    (200, 40_000, 15, 0, 10, 4 * 16384),
    # (220, 80_000, 20, 5, 12, 8 * 16384),
    # 1 large prime
    # (200, 40_000, 15, 0, 17, 512),
    (220, 80_000, 20, 5, 17, 512),
    (240, 200_000, 20, 5, 17, 1024),
    (260, 400_000, 20, 5, 17, 1024),
    (280, 1000_000, 20, 5, 17, 2048),
    (300, 1500_000, 25, 5, 17, 2048),
    (320, 2500_000, 30, 10, 19, 1024),
    # (340, 4000_000, 40, 10, 19, 1024),
    # (360, 6000_000, 40, 10, 21, 512),
    # 2 large primes
    # (320, 1000_000, 25, 25, 19, 1024),
    (340, 1500_000, 30, 25, 19, 1024),
    (360, 3000_000, 40, 25, 21, 512),
    (380, 6000_000, 45, 25, 21, 1024),
    (400, 10_000_000, 50, 30, 21, 1024),
    (420, 15_000_000, 60, 35, 21, 1024),
    (440, 20_000_000, 60, 40, 21, 1024),
    # 3 large primes
    # FIXME: should increase region size
    (460, 15_000_000, 60, 50, 21, 1024),
    (480, 25_000_000, 60, 55, 21, 1024),
    (500, 40_000_000, 60, 60, 21, 1024),
    (520, 60_000_000, 60, 65, 21, 1024),
)

# Parameters when GPU is much more powerful than CPU
# (cofactorization becomes a bottleneck)
PARAMS_LOWCPU = (
    # Old siever
    (120, 6000, 10, 0, 5, 1 * 16384),
    (140, 10_000, 10, 0, 6, 1 * 16384),
    (160, 15_000, 10, 0, 8, 1 * 16384),
    (180, 25_000, 10, 0, 9, 1 * 16384),
    (200, 40_000, 15, 0, 10, 4 * 16384),
    # 1 large prime
    (220, 60_000, 20, 0, 17, 1024),
    (240, 100_000, 20, 0, 17, 1024),
    (260, 200_000, 20, 0, 17, 1024),
    (280, 500_000, 20, 0, 17, 1024),
    (300, 800_000, 25, 0, 17, 2048),
    (320, 1500_000, 30, 0, 19, 1024),
    (340, 2500_000, 35, 0, 19, 1024),
    (360, 4000_000, 40, 0, 19, 2048),
    (380, 6000_000, 45, 0, 19, 2048),
    # 2 large primes
    (400, 5000_000, 50, 20, 21, 512),
    (420, 7000_000, 55, 25, 21, 512),
    (440, 10_000_000, 60, 30, 21, 512),
    (460, 15_000_000, 60, 35, 21, 512),
    (480, 20_000_000, 60, 40, 21, 512),
)


def get_params(N: int, bias: float | None = None) -> tuple:
    sz: float = N.bit_length()
    if bias:
        sz -= 2.5 * bias
    res = min(PARAMS, key=lambda t: abs(t[0] - sz))
    return res[1:]


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument("--dev", help="List of GPU devices (example: 0,1,2)")
    argp.add_argument(
        "-j",
        metavar="THREADS",
        type=int,
        help="Number of parallel GPU jobs",
    )
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR")
    args = argp.parse_args()

    main_impl(args)


def main_impl(args: argparse.Namespace):
    logging.getLogger().setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    N = args.N
    assert N > 1
    k, bias = multiplier(N)
    D = k * N
    logging.info(f"Multiplier is {k}")
    logging.info(f"Sieve smoothness bias is {bias:.2f} bits")

    B1, B2k, EXTRA_THRESHOLD, AFACS, INTERVAL_SIZE = get_params(N, bias)
    USE_SIEVER2 = AFACS >= 17
    if not USE_SIEVER2:
        OUTSTRIDE = 64
        POLYS_PER_WG = 1
        SEGMENT_SIZE = 16384  # constant
        SUBSEGMENT_SIZE = 8192  # constant
        ITERS = INTERVAL_SIZE // SEGMENT_SIZE
        assert INTERVAL_SIZE == ITERS * SEGMENT_SIZE
    else:
        ITERS = 1
        OUTSTRIDE = 65536
        SEGMENT_SIZE = None
        SUBSEGMENT_SIZE = None
        POLYS_PER_WG = None

    B2 = B1 * B2k
    M = INTERVAL_SIZE // 2

    THRESHOLD = (
        N.bit_length() // 2 + M.bit_length() - 2 * B1.bit_length() - EXTRA_THRESHOLD
    )

    primes = []
    roots = []
    for p, r in algebra.primebase(D, B1):
        assert (r * r - D) % p == 0
        primes.append(p)
        roots.append(r)
    logging.debug(
        f"Prime base size {len(primes)} {primes[:10]} ... {primes[-1]}, B2={B2 / 1e6:.1f}M"
    )
    logging.debug(f"Interval size {M // 512}k")

    # Target A such that AM²~=-2C where -4AC~=D
    A0 = int(math.sqrt(abs(D / 2)) / M)
    # Enforce at least 2 words, because A can grow for very small D.
    BLEN = max(2, (A0.bit_length() + 36) // 32)
    WORKCHUNK = 2 ** (AFACS - 1)
    logging.debug(f"{AFACS} factors per A, {BLEN} words per coefficient")

    results = []
    all_primes: set[int] = set()

    os.makedirs(args.OUTDIR, exist_ok=True)
    with open(os.path.join(args.OUTDIR, "args.json"), "w") as w:
        json.dump(
            {
                "n": N,
                "k": k,
                "d": D,
                "AFACS": AFACS,
                "B1": B1,
                "B2": B2,
            },
            w,
        )

    del N
    w = open(os.path.join(args.OUTDIR, "relations.sieve"), "w", buffering=1)

    WARGS = {
        "USE_SIEVER2": USE_SIEVER2,
        "primes": primes,
        "roots": roots,
        "D": D,
        "B1": B1,
        "B2": B2,
        "AFACS": AFACS,
        "BLEN": BLEN,
        "POLYS_PER_WG": POLYS_PER_WG,
        "ITERS": ITERS,
        "THRESHOLD": THRESHOLD,
        "OUTSTRIDE": OUTSTRIDE,
        "INTERVAL_SIZE": INTERVAL_SIZE,
        # For sieve 1
        "SEGMENT_SIZE": SEGMENT_SIZE,
        "SUBSEGMENT_SIZE": SUBSEGMENT_SIZE,
    }

    if args.dev:
        gpu_ids = [int(_id) for _id in args.dev.split(",")]
    else:
        gpu_ids = [0]
    sieve.GPU_LOCK.clear()
    for _id in gpu_ids:
        sieve.GPU_LOCK.append((_id, Semaphore(1)))
    if args.j:
        NWORKERS = args.j
    else:
        NWORKERS = len(gpu_ids) * (2 if USE_SIEVER2 else 4)

    sieve_pool = Pool(NWORKERS, initializer=worker_init, initargs=(WARGS,))
    sieved = 0
    start_time = time.monotonic()
    last_print = time.monotonic()
    last_sieved = 0
    next_check = 0
    nb = 0
    done = False

    TARGET_GAP = 64 + D.bit_length()

    Aseen: set[tuple[int]] = set()
    xseen: set[int] = set()
    xdups = 0
    while not done:
        As = sieve.make_a(primes, A0, AFACS, Aseen)
        Aseen.update(As)

        for dt, nreports, rows in sieve_pool.imap_unordered(worker_task, As):
            for row in rows:
                all_primes.update(abs(_p) for _p in row[1:])
                x0 = min(row[0], D - row[0])
                if x0 in xseen:
                    xdups += 1
                    continue
                xseen.add(x0)
                print(" ".join(str(_x) for _x in row), file=w)
                results.append(row[1:])

            _nr = len(results)
            _np = len(all_primes)
            nb += 1
            sieved += WORKCHUNK * INTERVAL_SIZE
            avg_speed = sieved / (time.monotonic() - start_time)
            rel_speed = _nr / (time.monotonic() - start_time)
            if nb % 10 == 0 and (nb < 1000 or time.monotonic() > last_print + 1.0):
                cur_speed = (sieved - last_sieved) / (time.monotonic() - last_print)
                last_sieved = sieved
                last_print = time.monotonic()
                print(
                    f"Chunk {nb} ({sieved / 1e6:.0f}M) done in {dt:.3f}s ({cur_speed / 1e9:.1f}G/s avg {avg_speed / 1e9:.1f}G/s) "
                    + f"{len(rows)}/{nreports} items (relations={_nr} [{rel_speed:.4g}/s] primes={_np} excess={_nr - _np})"
                )

            # Check if finished
            gap = _nr - _np
            if gap > TARGET_GAP:
                done = True
                break
            if next_check == 0 and gap > -_nr:
                _, curexc = relations.step_prune(results, B1, len(primes))
                if curexc is None:
                    curexc = gap
                curexc -= TARGET_GAP
                if curexc > 64:
                    done = True
                    break
                needed = max(500, min(20000, 2 * abs(curexc)))
                next_check = gap + needed
            if next_check and gap > next_check:
                pruned, curexc = relations.step_prune(results, B1, len(primes))
                if curexc is None:
                    curexc = gap
                curexc -= TARGET_GAP
                if curexc > 64:
                    done = True
                    break
                # Wait for more relations (clamp to 500..20000)
                needed = max(500, min(20000, 2 * abs(curexc)))
                next_check = gap + needed

    sieve_pool.close()
    avg_speed = sieved / (time.monotonic() - start_time)
    rel_speed = _nr / (time.monotonic() - start_time)
    logging.info(
        f"Sieved {nb} A ({sieved / 1e6:.0f}M) in {dt:.3f}s (avg {avg_speed / 1e9:.1f}G/s) (relations={_nr} [{rel_speed:.1f}/s] primes={_np} excess={_nr - _np})"
    )

    total_time = time.monotonic() - start_time
    if xdups:
        logging.warning(f"Ignored {xdups} duplicate relations")
    logging.info(
        f"Got {len(results)} relations with {len(all_primes)} primes in {total_time:.3f}s"
    )


if __name__ == "__main__":
    main()
