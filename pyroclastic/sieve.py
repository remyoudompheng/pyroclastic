"""
Self-initializing Quadratic Sieve for classgroup computation

We assume that D=-N or -4N where N is a squarefree positive integer

The polynomials are:
  P = A x² + B x + C where B^2 - 4 A C = D = -N
such that
  4A P = 4A² x² + 4AB x + B² - D
       = (2 A x + B)² - D
  P(M) is minimal

and A ~= sqrt(N) / 2M where M is the sieve half-length
    C ~= sqrt(N) * M / 2

Computation is:
    B = sqrt(D) mod 4A
    -C = (D - B^2) / 4A

Roots mod p are:
    (±sqrt(N) - B) / 2A
where A has size sqrt(2N) / 2M
      B has size 4A

Polynomial extrema are P(0)=sqrt(N)*M/2, P(±M)=sqrt(N)*M
"""

from typing import Tuple, Optional

import argparse
import importlib
import itertools
import json
import logging
import math
import os
import random
import subprocess
import struct
import time
import queue
from multiprocessing import Pool, Semaphore, current_process

import numpy as np
import kp
import tqdm
import flint

import pyroclastic_flint_extras as flint_extras
from . import algebra
from . import gpu
from . import relations

SEGMENT_SIZE = 16384
SUBSEGMENT_SIZE = 8192


def product(l: list[int]):
    p = l[0]
    for x in l[1:]:
        p *= x
    return p


def make_a(
    pbase, target: int, AFACS: int, seen: Optional[set[tuple]] = None
) -> list[tuple[int]]:
    """
    Returns products of elements of pbase approximating target
    """
    assert len(pbase) > 50
    t = None
    for k in [AFACS]:
        t = target ** (1.0 / k)
        if (
            t < pbase[min(100, len(pbase) // 3)]
            or t > pbase[-min(100, len(pbase) // 3)]
        ):
            continue
        break
    if t is None:
        raise ValueError("Cannot find proper A for target {target}")
    idx = min(i for i, p in enumerate(pbase) if p > t)
    base = pbase[max(2, idx - 20) : idx + 20]
    logging.debug(f"Using {k} factors for A in {base[0]}..{base[-1]}")
    candidates: list[tuple[int]] = []
    batch_size = max(32, len(pbase) // 10)
    while len(candidates) < 2 * batch_size:
        ps = random.sample(base, k - 1)
        tgt = target // product(ps)
        try:
            q = next(_p for _p in base if _p >= tgt)
        except StopIteration:
            continue
        if q in base and q not in ps:
            facs = tuple(sorted(ps + [q]))
            if seen and facs in seen:
                continue
            candidates.append(facs)

    res = sorted((abs(product(ps) - target), ps) for ps in sorted(set(candidates)))
    res = res[: len(res) // 2]
    quality = float(res[-1][0]) / float(target)
    logging.debug(f"Prepared {len(res)} values of A, quality {100*quality:.2f}%")
    return [_ps for _, _ps in res]


def make_poly(N: int, ak: list, roots: dict) -> Tuple[int, list]:
    """
    Return A, [Bi] defining 2^len(Bi) polynomials

    A X^2 + B X + C
    such that B = sum(±Bi) is a square root of N modulo A
              C = (B²-N)/4A
    """
    assert N & 3 in (0, 1)
    # If N is odd, we want B0 odd and Bi even for i > 0
    # If N is even, we want B0 even and Bi even for i > 0
    parity = N & 1

    A = product(ak)
    Bi = []
    for i, ai in enumerate(ak):
        c = A // ai * (pow(A // ai, -1, ai) * roots[ai] % ai)
        # We want B0 even and Bi odd for i > 0
        if i == 0 and c & 1 != parity:
            c += A
        if i > 0 and c & 1 == 1:
            c += A
        Bi.append(c)

    assert (sum(Bi) - 2 * sum(random.sample(Bi, 3))) ** 2 % (4 * A) == N % (4 * A)

    return A, Bi


def expand_polys(N: int, A: int, Bi: list[int]):
    polys = []
    for bis in itertools.product(*[(b, -b) for b in Bi[:0:-1]]):
        B = Bi[0] + sum(bis)
        C = (B**2 - N) // (4 * A)
        polys.append((A, B, C))
    assert polys[0][1] == sum(Bi)
    assert polys[1][1] == sum(Bi) - 2 * Bi[1]
    assert polys[-1][1] == Bi[0] - sum(Bi[1:])
    return polys


def build_relation(value, idx, facs, B1=None, B2=None, NLARGE=2):
    row = []
    v = value
    assert v > 0
    # We maybe don't sieve 2
    tz = (v ^ (v - 1)).bit_length() - 1
    v >>= tz
    row.extend(tz * [2])
    for f in facs:
        while v % f == 0:
            row.append(f)
            v //= f
    if v == 1:
        return row
    if v > B2**NLARGE:
        return None
    cofacs = flint.fmpz(v).factor_smooth(bits=20)
    for _p, _e in cofacs:
        if _p < B1:
            pass
            # logging.error(f"WARNING: uncaught small prime {_p}")
        row.extend(_e * [_p])
    return row


def process_sieve_reports(ABi, bout, bfacs, N, B1, B2, NLARGE, OUTSTRIDE):
    if isinstance(bout, bytes):
        vout = np.frombuffer(bout, dtype=np.int32)
        vfacs = np.frombuffer(bfacs, dtype=np.uint32)
    else:
        vout, vfacs = bout, bfacs
    WORKCHUNK = len(vout) // OUTSTRIDE
    reports = 0
    results = []
    A, ak, Bi = ABi
    for _i, (_A, _B, _C) in enumerate(expand_polys(N, A, Bi)):
        for _j in range(OUTSTRIDE):
            oidx = OUTSTRIDE * _i + _j
            if not vout[oidx] and not vfacs[32 * oidx]:
                break
            reports += 1
            x = int(vout[oidx])
            _facs = [int(_f) for _f in vfacs[32 * oidx : 32 * oidx + 32] if _f]
            v = _A * x * x + _B * x + _C
            u = 2 * _A * x + _B
            assert u * u == 4 * A * v + N
            # In the class group: (A, B, C) * (v, u, A) == 1
            row = build_relation(v, x, _facs, B1=B1, B2=B2, NLARGE=NLARGE)
            if row is None or any(_r > B2 for _r in row):
                # factors too large
                continue
            assert product(row) == v
            # Add correct signs to ideals
            for i, p in enumerate(row):
                up = u % p
                if p == 2:
                    # -[2] if the root is 3 mod 4
                    if u & 3 == 3:
                        row[i] = -p
                else:
                    # -[p] if the root is even
                    if up & 1 == 0:
                        # Even root
                        row[i] = -p
            # Add factors of A
            for ai in ak:
                bp = _B % ai
                if bp & 1 == 0:
                    row.append(-ai)
                else:
                    row.append(ai)
            results.append(row)
    return reports, results


def to_uvec(x: int, length: int):
    assert x.bit_length() <= 32 * length
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(length)]


PARAMS1 = (
    # Single large prime
    # bitsize, B1, B2/B1, OUTSTRIDE, EXTRA_THRESHOLD, AFACS, ITERS, POLYS_PER_WG
    # Larger intervals for small inputs to avoid fixed costs
    (100, 2_000, 5, 1, -10, 6, 16, 1),
    (120, 3_000, 5, 1, -10, 7, 8, 1),
    (140, 6_000, 5, 1, -10, 8, 4, 1),
    (160, 10_000, 15, 1, -10, 10, 2, 1),
    (180, 20_000, 15, 1, -10, 10, 3, 1),
    (200, 40_000, 20, 1, -10, 11, 4, 1),
    (220, 60_000, 20, 1, -15, 11, 6, 1),
    (240, 100_000, 25, 1, -15, 12, 8, 1),
    (260, 200_000, 25, 1, -15, 12, 10, 2),
    (280, 300_000, 25, 1, -15, 12, 12, 2),
)

PARAMS2 = (
    # bitsize, B1, B2/B1, OUTSTRIDE, EXTRA_THRESHOLD, AFACS, ITERS, POLYS_PER_WG
    (140, 3_000, 5, 1, 5, 9, 1, 1),
    (160, 5_000, 15, 2, 15, 10, 2, 1),
    (180, 10_000, 20, 2, 20, 11, 4, 1),
    (200, 20_000, 20, 3, 20, 11, 5, 1),
    (220, 30_000, 20, 2, 20, 12, 6, 2),
    (240, 50_000, 25, 2, 25, 12, 8, 2),
    (260, 90_000, 25, 2, 25, 12, 8, 2),
    (280, 150_000, 25, 1, 25, 12, 10, 2),
    (300, 250_000, 25, 1, 25, 12, 12, 2),
    (320, 400_000, 25, 1, 25, 12, 14, 2),
    (340, 600_000, 25, 1, 25, 12, 16, 2),
    (360, 1000_000, 25, 1, 25, 12, 18, 2),
    (380, 1500_000, 25, 1, 25, 12, 20, 2),
    (400, 2000_000, 25, 1, 25, 12, 24, 2),
)

# Triple large prime
PARAMS3 = (
    (360, 500_000, 50, 2, 55, 12, 18, 2),  # too slow
    (380, 750_000, 50, 1, 45, 12, 12, 2),
    (400, 1000_000, 50, 1, 50, 12, 24, 2),
    (500, 7000_000, 200, 1, 70, 12, 16, 2),
)


def get_params(N: int):
    sz = N.bit_length()
    res = None
    if sz < 200:
        PARAMS, nlarge = PARAMS1, 1
    elif sz < 450:
        PARAMS, nlarge = PARAMS2, 2
    else:
        PARAMS, nlarge = PARAMS3, 3
    for p in PARAMS:
        if p[0] <= sz:
            res = p
    return res[1:] + (nlarge,)


# At most 2 GPU jobs at a time
GPU_LOCK = [Semaphore(2)]


class Siever:
    def __init__(self, wargs):
        self.wargs = wargs
        primes = wargs["primes"]
        roots = wargs["roots"]
        AFACS = wargs["AFACS"]
        BLEN = wargs["BLEN"]
        POLYS_PER_WG = wargs["POLYS_PER_WG"]
        HUGE_PRIME = wargs["HUGE_PRIME"]
        BUCKET_SIZE = wargs["BUCKET_SIZE"]
        AVG_BUCKET_SIZE = wargs["AVG_BUCKET_SIZE"]
        ITERS = wargs["ITERS"]
        THRESHOLD = wargs["THRESHOLD"]
        OUTSTRIDE = wargs["OUTSTRIDE"]

        self.roots_d = {p: r for p, r in zip(primes, roots)}

        WORKCHUNK = 2 ** (AFACS - 1)

        devname = gpu.device_name()
        self.stampPeriod = gpu.stamp_period()

        proc = current_process()
        self.gpu_idx = proc._identity[-1] % len(GPU_LOCK)
        mgr = kp.Manager(self.gpu_idx)
        gpu_name = mgr.get_device_properties().get("device_name", "unknown")
        logging.info(f"Worker {proc.name} running on GPU {self.gpu_idx} ({gpu_name})")
        xp = mgr.tensor_t(np.array(primes, dtype=np.uint32))
        xn = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        nsumroots = (
            (1 + 16 + 16 + 2 ** (AFACS - 9))
            if AFACS > 9
            else (1 + 16 + 2 ** (AFACS - 5))
        )
        xr = mgr.tensor_t(np.zeros(len(roots) * nsumroots, dtype=np.uint32))
        xb = mgr.tensor_t(np.zeros(AFACS * BLEN, dtype=np.uint32))
        xargs = mgr.tensor_t(np.zeros(BLEN, dtype=np.uint32))
        # Huge offsets are u16
        xhuge = mgr.tensor_t(
            np.zeros(
                (WORKCHUNK // POLYS_PER_WG)
                * BUCKET_SIZE
                * SEGMENT_SIZE
                // SUBSEGMENT_SIZE
                * ITERS
                // 2,
                dtype=np.uint32,
            )
        )
        xout = mgr.tensor_t(np.zeros(OUTSTRIDE * WORKCHUNK, dtype=np.uint32))
        # Buffer for factors (32 elements per report)
        xfacs = mgr.tensor_t(np.zeros(xout.size() * 32, dtype=np.uint32))

        self.tensors = (xargs, xb, xout, xfacs)

        mem_main = 4 * (xp.size() + xr.size() + xn.size() + xout.size() + xfacs.size())
        mem_huge = 4 * xhuge.size()
        mem_huge_avg = int(mem_huge * AVG_BUCKET_SIZE / BUCKET_SIZE)
        mem = mem_main + mem_huge

        logging.debug(
            f"Memory usage {mem >> 10} kB (main {mem_main >> 10} kB, huge reserved {mem_huge >> 10} kB, huge used {mem_huge_avg >> 10} kB)"
        )

        # Send initial buffers (immutable)
        mgr.sequence().record(kp.OpTensorSyncDevice([xp, xn, xargs])).eval()

        PRESHADER = gpu.compile("siqs_prepare.comp", {"BLEN": BLEN, "AFACS": AFACS})

        SHADER = gpu.compile(
            "siqs.comp",
            {
                "AFACS": AFACS,
                "POLYS_PER_WG": POLYS_PER_WG,
                "SEGMENT_SIZE": SEGMENT_SIZE,
                "SUBSEGMENT_SIZE": SUBSEGMENT_SIZE,
                "HUGE_PRIME": HUGE_PRIME,
                "BUCKET_SIZE": BUCKET_SIZE,
                "ITERS": ITERS,
                "THRESHOLD": THRESHOLD,
                "OUTSTRIDE": OUTSTRIDE,
            },
        )

        self.mgr = mgr
        self.algo1 = mgr.algorithm(
            [xp, xn, xr, xb, xargs],
            PRESHADER,
            workgroup=(len(primes) // 512 + 1, 1, 1),
        )
        self.algo2 = mgr.algorithm(
            [xp, xr, xhuge, xout, xfacs],
            SHADER,
            workgroup=(WORKCHUNK // POLYS_PER_WG, 1, 1),
        )

    def process(self, ak):
        AFACS = self.wargs["AFACS"]
        BLEN = self.wargs["BLEN"]
        OUTSTRIDE = self.wargs["OUTSTRIDE"]
        D, B1, B2 = self.wargs["D"], self.wargs["B1"], self.wargs["B2"]
        NLARGE = self.wargs["NLARGE"]
        A, Bi = make_poly(D, ak, self.roots_d)

        xargs, xb, xout, xfacs = self.tensors
        xargs.data()[:] = to_uvec(A, BLEN)
        for i in range(AFACS):
            xb.data()[BLEN * i : BLEN * (i + 1)] = to_uvec(Bi[i], BLEN)
        xout.data().fill(0)
        xfacs.data().fill(0)

        with GPU_LOCK[self.gpu_idx]:
            seq = (
                self.mgr.sequence(total_timestamps=16)
                .record(kp.OpTensorSyncDevice([xb, xout, xfacs, xargs]))
                .record(kp.OpAlgoDispatch(self.algo1))
                .record(kp.OpAlgoDispatch(self.algo2))
                .record(kp.OpTensorSyncLocal([xout, xfacs]))
                .eval()
            )

        # Get GPU results (round to 0.5 tick if ticks are zero)
        stamps = seq.get_timestamps()
        dt = max(0.5, stamps[-1] - stamps[0]) * self.stampPeriod * 1e-9

        vout = xout.data()
        vfacs = xfacs.data()

        nreports, rows = process_sieve_reports(
            (A, ak, Bi), vout.astype(np.int32), vfacs, D, B1, B2, NLARGE, OUTSTRIDE
        )
        return dt, nreports, rows


SIEVER = None


def worker_init(initargs):
    global SIEVER
    SIEVER = Siever(initargs)


def worker_task(ak):
    return SIEVER.process(ak)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument(
        "--check", action="store_true", help="Verify relations (requires cypari2)"
    )
    argp.add_argument(
        "-j", metavar="THREADS", type=int, help="Number of CPU threads (and parallel GPU jobs)"
    )
    argp.add_argument(
        "--ngpu", metavar="GPUS", type=int, default=1, help="Number of GPUs (usually a divisor of THREADS)"
    )
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR")
    args = argp.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    while len(GPU_LOCK) < args.ngpu:
        GPU_LOCK.append(Semaphore(2))

    N = args.N
    B1, B2k, OUTSTRIDE, EXTRA_THRESHOLD, AFACS, ITERS, POLYS_PER_WG, NLARGE = (
        get_params(N)
    )
    B2 = B2k * B1
    M = ITERS * SEGMENT_SIZE // 2

    THRESHOLD = (
        N.bit_length() // 2 + M.bit_length() - 2 * B1.bit_length() - EXTRA_THRESHOLD
    )

    D = -abs(N)
    del N
    COFACTOR_BACKGROUND = True

    primes = []
    roots = []
    for p, k in algebra.primebase(D, B1):
        assert (k * k - D) % p == 0
        primes.append(p)
        roots.append(k)
    logging.debug(f"Prime base size {len(primes)} {primes[:10]} ... {primes[-1]}, B2={B2/1e6:.1f}M")
    logging.debug(f"Interval size {M//512}k")

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

    A0 = math.isqrt(abs(D)) // (2 * M)
    BLEN = (A0.bit_length() + 36) // 32
    WORKCHUNK = 2 ** (AFACS - 1)
    logging.debug(f"{AFACS} factors per A, {BLEN} words per coefficient")

    BLOCK_SIZE = SEGMENT_SIZE * ITERS

    results = []
    all_primes = set()

    os.makedirs(args.OUTDIR, exist_ok=True)
    with open(os.path.join(args.OUTDIR, "args.json"), "w") as w:
        json.dump(
            {
                "d": D,
            },
            w,
        )
    w = open(os.path.join(args.OUTDIR, "relations.sieve"), "w", buffering=1)

    WARGS = {
        "primes": primes,
        "roots": roots,
        "D": D,
        "B1": B1,
        "B2": B2,
        "NLARGE": NLARGE,
        "AFACS": AFACS,
        "BLEN": BLEN,
        "POLYS_PER_WG": POLYS_PER_WG,
        "HUGE_PRIME": HUGE_PRIME,
        "BUCKET_SIZE": BUCKET_SIZE,
        "AVG_BUCKET_SIZE": AVG_BUCKET_SIZE,
        "ITERS": ITERS,
        "THRESHOLD": THRESHOLD,
        "OUTSTRIDE": OUTSTRIDE,
    }

    if args.j:
        NWORKERS = args.j
    else:
        NWORKERS = 8 if D.bit_length() < 270 else 3
    sieve_pool = Pool(NWORKERS, initializer=worker_init, initargs=(WARGS,))
    sieved = 0
    start_time = time.monotonic()
    last_print = time.monotonic()
    next_check = 0
    nb = 0
    done = False

    TARGET_GAP = 4 * len(primes)
    Aseen = set()
    while not done:
        As = make_a(primes, A0, AFACS, Aseen)
        Aseen.update(As)

        for dt, nreports, rows in sieve_pool.imap_unordered(worker_task, As):
            for row in rows:
                all_primes.update(abs(_p) for _p in row)
                print(" ".join(str(_x) for _x in row), file=w)
                results.append(row)

            speed = WORKCHUNK * BLOCK_SIZE / dt
            _nr = len(results)
            _np = len(all_primes)
            nb += 1
            sieved += WORKCHUNK * BLOCK_SIZE
            avg_speed = sieved / (time.monotonic() - start_time)
            rel_speed = _nr / (time.monotonic() - start_time)
            if nb % 10 == 0 and (nb < 1000 or time.monotonic() > last_print + 1.0):
                last_print = time.monotonic()
                print(
                    f"Chunk {nb} ({sieved/1e6:.0f}M) done in {dt:.3f}s ({speed/1e9:.1f}G/s avg {avg_speed/1e9:.1f}G/s) {len(rows)}/{nreports} items (relations={_nr} [{rel_speed:.1f}/s] primes={_np} excess={_nr-_np})"
                )

            # Check if finished
            gap = _nr - _np
            if next_check == 0 and gap > -_nr:
                _, curexc = relations.prune2(results, B1, len(primes))
                if curexc is None:
                    curexc = gap
                curexc -= TARGET_GAP
                if curexc > 64:
                    done = True
                    break
                needed = max(500, min(20000, 2 * abs(curexc)))
                next_check = gap + needed
            if next_check and gap > next_check:
                pruned, curexc = relations.prune2(results, B1, len(primes))
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
    print(
        f"Sieved {nb} A ({sieved/1e6:.0f}M) in {dt:.3f}s (avg {avg_speed/1e9:.1f}G/s) (relations={_nr} [{rel_speed:.1f}/s] primes={_np} excess={_nr-_np})"
    )

    total_time = time.monotonic() - start_time
    logging.info(
        f"Got {len(results)} relations with {len(all_primes)} primes in {total_time:.3f}s"
    )

    pruned, _ = relations.prune2(results, B1, len(primes))
    with open("/tmp/relations.pruned", "w") as wp:
        for row in pruned:
            print(" ".join(str(_x) for _x in row), file=wp)

    if args.check:
        forms = {}

        def ideal(p):
            if p not in forms:
                forms[p] = flint_extras.qfb.prime_form(D, p)
            return forms[p]

        for _row in tqdm.tqdm(results):
            q = ideal(1)
            for p in _row:
                if p > 0:
                    q = q * ideal(p)
                else:
                    q = q * ideal(-p) ** -1
            # q must be the unit form
            assert q.q()[0] == 1, (_row, q)

    for _row in results[:16]:
        print(_row)
    print("...")
    for _row in results[-16:]:
        print(_row)


if __name__ == "__main__":
    main()
