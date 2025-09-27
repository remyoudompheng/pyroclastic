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

import argparse
import itertools
import json
import logging
import math
import os
import pathlib
import random
import time
from multiprocessing import Pool, Semaphore, current_process
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import kp
import flint

try:
    import pymqs
except ImportError:
    pymqs = None

import pyroclastic_flint_extras as flint_extras
from . import algebra
from . import gpu
from . import relations

BUCKET_INTERVAL = 8192
SHARDS = 256

DEBUG_TIMINGS = False


def smoothness_bias(D: int) -> float:
    """
    The average amount of extra bits in the smooth part
    """
    match D % 8:
        case 1:
            b = 1.0
        case 5:
            b = 0.0
        case 0:
            b = -0.5
        case 4:
            b = -0.5
        case _:
            raise ValueError("invalid quadratic discriminant")

    for p in algebra.smallprimes(1000):
        if p < 3:
            continue
        legendre = pow(D, p // 2, p)
        if legendre == p - 1:
            legendre = -1
        b += legendre * math.log2(p) / float(p)
    return b


def product(l: list[int] | tuple[int]):
    p = l[0]
    for x in l[1:]:
        p *= x
    return p


def make_a(
    pbase, target: int, AFACS: int, seen: set[tuple] | None = None
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
    logging.debug(f"Prepared {len(res)} values of A, quality {100 * quality:.2f}%")
    return [_ps for _, _ps in res]


def make_poly(N: int, ak: list, roots: dict) -> tuple[int, list]:
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


def expand_one_poly(N: int, A: int, Bi: list[int], idx: int):
    B = Bi[0] + sum(-bi if idx & (1 << i) else bi for i, bi in enumerate(Bi[1:]))
    C = (B**2 - N) // (4 * A)
    return A, B, C


def build_relation(
    value: int, idx: int, facs: list[int], B1=None, B2=None
) -> list[int] | None:
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
    if pymqs is not None:
        cofacs = [
            (_l, 1) for _l in pymqs.factor_smooth(v, B2.bit_length() if B2 else 20)
        ]
    else:
        cofacs = flint.fmpz(v).factor_smooth(bits=B2.bit_length() if B2 else 20)
    for _p, _e in cofacs:
        if not flint.fmpz(_p).is_probable_prime():
            return None
        # logging.error(f"WARNING: uncaught small prime {_p}")
        row.extend(_e * [_p])
    return row


def process_sieve_reports(ABi, bout, bfacs, N, B1, B2, OUTSTRIDE):
    if isinstance(bout, bytes):
        vout = np.frombuffer(bout, dtype=np.int32)
        vfacs = np.frombuffer(bfacs, dtype=np.uint32)
    else:
        vout, vfacs = bout, bfacs

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
                poly = expand_one_poly(N, A, Bi, _i)
            _A, _B, _C = poly
            reports += 1
            x = int(vout[oidx])
            _facs = [int(_f) for _f in vfacs[32 * oidx : 32 * oidx + 32] if _f]
            v = _A * x * x + _B * x + _C
            u = 2 * _A * x + _B
            assert u * u == 4 * A * v + N
            # In the class group: (A, B, C) * (v, u, A) == 1
            row = build_relation(v, x, _facs, B1=B1, B2=B2)
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
    (0, 2000, 5, 1, -20, 6, 8, 1),
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
    (380, 1500_000, 30, 1, 30, 12, 20, 2),
    (400, 2000_000, 40, 1, 30, 12, 24, 2),
    (420, 2500_000, 50, 1, 35, 12, 28, 2),
    (440, 3000_000, 55, 1, 35, 12, 30, 2),
    (460, 4000_000, 60, 1, 40, 12, 36, 2),
    (480, 6000_000, 80, 1, 45, 12, 48, 2),
    (500, 8000_000, 100, 1, 50, 12, 64, 2),
)

# Parameters for optimized sieve with tiny intervals
PARAMS_SIEVE2 = (
    # bitsize, B1, B2/B1, EXTRA_THRESHOLD, AFACS, INTERVAL_SIZE
    (340, 1000_000, 25, 25, 17, 4096),
    (360, 1200_000, 25, 25, 19, 1024),
    (380, 1500_000, 30, 30, 19, 1024),
    (400, 1800_000, 35, 30, 19, 1024),
    (420, 2000_000, 50, 30, 19, 1024),
    (440, 2500_000, 55, 35, 21, 512),
    (460, 3500_000, 60, 40, 21, 512),
    (480, 5000_000, 80, 45, 21, 512),
    (500, 6000_000, 100, 50, 21, 512),
)


def get_params(N: int, bias: float | None = None) -> tuple:
    sz: float = N.bit_length()
    if bias:
        sz -= 2.5 * bias
    res = None
    PARAMS: tuple
    if sz < 200:
        PARAMS = PARAMS1
    else:
        PARAMS = PARAMS2
    for p in PARAMS:
        if p[0] <= sz:
            res = p
    assert res is not None
    return res[1:]


def get_params2(N: int, bias: float | None = None) -> tuple:
    sz: float = N.bit_length()
    if bias:
        sz -= 2.5 * bias
    res = None
    for p in PARAMS_SIEVE2:
        if p[0] <= sz:
            res = p
    assert res is not None
    return res[1:]


# At most 2 GPU jobs at a time
GPU_LOCK = [Semaphore(1)]


class Siever:
    """
    Sieve kernel for traditional SIQS

    * a small kernel computes Bi mod l for each sieve prime l
    * each workgroup handles 1 or several polynomial
    * each polynomial is sieved over a large interval (multiple 16k segments)
      and all roots modulo l are reconstructed on-the-fly during sieve
    """

    def __init__(self, wargs):
        self.wargs = wargs
        primes = wargs["primes"]
        roots = wargs["roots"]
        AFACS = wargs["AFACS"]
        BLEN = wargs["BLEN"]
        POLYS_PER_WG = wargs["POLYS_PER_WG"]
        SEGMENT_SIZE = wargs["SEGMENT_SIZE"]
        SUBSEGMENT_SIZE = wargs["SUBSEGMENT_SIZE"]
        HUGE_PRIME = wargs.get("HUGE_PRIME")
        BUCKET_SIZE = wargs.get("BUCKET_SIZE")
        AVG_BUCKET_SIZE = wargs.get("AVG_BUCKET_SIZE")
        ITERS = wargs["ITERS"]
        THRESHOLD = wargs["THRESHOLD"]
        OUTSTRIDE = wargs["OUTSTRIDE"]
        DEBUG = wargs.get("DEBUG")

        # A bucket contains offsets for a subsegment
        if HUGE_PRIME is None:
            BUCKET_SIZE = 1
            AVG_BUCKET_SIZE = 0
            HUGE_PRIME = len(primes)
            if primes[-1] > 2 * SEGMENT_SIZE:
                HUGE_PRIME = next(
                    idx for idx, p in enumerate(primes) if p > 1.5 * SEGMENT_SIZE
                )
                if HUGE_PRIME % 512 != 84:
                    HUGE_PRIME += (84 - HUGE_PRIME) % 512
                    assert HUGE_PRIME % 512 == 84
                phuge = primes[HUGE_PRIME]
                assert phuge > 2**14
                BUCKET_SIZE = int(
                    SUBSEGMENT_SIZE * 0.07 * math.log2(primes[-1] / (0.8 * phuge))
                )
                AVG_BUCKET_SIZE = (
                    SUBSEGMENT_SIZE * 0.055 * math.log2(primes[-1] / phuge)
                )
                logging.debug(
                    f"Huge prime index {HUGE_PRIME} ({phuge}) bucket size %d expect usage %d",
                    BUCKET_SIZE,
                    int(AVG_BUCKET_SIZE),
                )

        self.roots_d = {p: r for p, r in zip(primes, roots)}

        WORKCHUNK = 2 ** (AFACS - 1)

        self.stampPeriod = gpu.stamp_period()

        proc = current_process()
        proc_id = proc._identity or (0,)
        self.gpu_idx = proc_id[-1] % len(GPU_LOCK)
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

        # Output buffer to receive full sieve results.
        xdebug = None
        if DEBUG:
            logging.debug("sieve debug output enabled")
            xdebug = mgr.tensor_t(
                np.zeros(WORKCHUNK * ITERS * SEGMENT_SIZE // 4, dtype=np.uint32)
            )

        mem_main = 4 * (xp.size() + xr.size() + xn.size() + xout.size() + xfacs.size())
        mem_huge = 4 * xhuge.size()
        mem_huge_avg = int(mem_huge * AVG_BUCKET_SIZE / BUCKET_SIZE)
        mem = mem_main + mem_huge

        logging.debug(
            f"Memory usage {mem >> 10} kB (main {mem_main >> 10} kB, huge reserved {mem_huge >> 10} kB, huge used {mem_huge_avg >> 10} kB)"
        )

        # Send initial buffers (immutable)
        mgr.sequence().record(kp.OpTensorSyncDevice([xp, xn, xargs])).eval()

        PRESHADER = gpu.compile(
            "siqs_prepare.comp",
            {
                "BLEN": BLEN,
                "AFACS": AFACS,
                "SEGMENT_SIZE": SEGMENT_SIZE,
                "ITERS": ITERS,
            },
        )

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
            }
            | ({"DEBUG": 1} if DEBUG else {}),
        )

        self.mgr = mgr
        self.algo1 = mgr.algorithm(
            [xp, xn, xr, xb, xargs],
            PRESHADER,
            workgroup=(len(primes) // 512 + 1, 1, 1),
        )
        self.algo2 = mgr.algorithm(
            [xp, xr, xhuge, xout, xfacs] + ([xdebug] if xdebug else []),
            SHADER,
            workgroup=(WORKCHUNK // POLYS_PER_WG, 1, 1),
        )

    def process(self, ak):
        BLEN = self.wargs["BLEN"]
        OUTSTRIDE = self.wargs["OUTSTRIDE"]
        D, B1, B2 = self.wargs["D"], self.wargs["B1"], self.wargs["B2"]
        A, Bi = make_poly(D, ak, self.roots_d)
        if A.bit_length() + 2 > BLEN * 32:
            logging.error(f"Skipping A={A} (too large)")
            return 1.0, 0, []

        dt = self._run(ak, A, Bi)
        _, _, xout, xfacs = self.tensors
        vout = xout.data()
        vfacs = xfacs.data()
        nreports, rows = process_sieve_reports(
            (A, ak, Bi), vout.astype(np.int32), vfacs, D, B1, B2, OUTSTRIDE
        )
        return dt, nreports, rows

    def _run(self, ak, A, Bi):
        AFACS = self.wargs["AFACS"]
        BLEN = self.wargs["BLEN"]

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
        return dt


class Siever2:
    """
    Sieve kernels for very small interval sizes:

    * each workgroup of the sieve kernel handle a region of size 8k
      covering multiple polynomials
    * first root pass uses the meet-in-the-middle strategy and
      outputs 32-bit values stored in 256 shards (3-bit log, 29-bit index and polynomial)
      (total sieve region per batch is at most 2^37)
    * sieve hits are stored as 16-bit integers (region index and 3-bit prime size)
    * the 3-bit prime size is clamp((plog - 8) / 2) representing values 8, 10, ..22
    """

    def __init__(self, wargs):
        self.wargs = wargs
        primes = wargs["primes"]
        roots = wargs["roots"]
        AFACS = wargs["AFACS"]
        BLEN = wargs["BLEN"]
        HUGE_PRIME = wargs.get("HUGE_PRIME")
        BUCKET_SIZE = wargs.get("BUCKET_SIZE")
        AVG_BUCKET_SIZE = wargs.get("AVG_BUCKET_SIZE")
        INTERVAL_SIZE = wargs.get("INTERVAL_SIZE")
        ITERS = wargs.get("ITERS", 1)
        assert ITERS == 1
        THRESHOLD = wargs["THRESHOLD"]
        OUTSTRIDE = wargs["OUTSTRIDE"]
        DEBUG = wargs.get("DEBUG")

        AHALF = AFACS // 2
        assert AFACS == 2 * AHALF + 1

        # A bucket contains offsets for a subsegment
        if HUGE_PRIME is None:
            BUCKET_SIZE = 1
            AVG_BUCKET_SIZE = 0
            HUGE_PRIME = len(primes)
            if primes[-1] > 2 * INTERVAL_SIZE:
                HUGE_PRIME = next(
                    idx
                    for idx, p in enumerate(primes)
                    if p > 1.5 * INTERVAL_SIZE and p.bit_length() >= AHALF
                )
                # Small primes (except 4 tiny, are a multiple of 32)
                if HUGE_PRIME % 32 != 4:
                    HUGE_PRIME += (4 - HUGE_PRIME) % 32
                    assert HUGE_PRIME % 32 == 4
                phuge = primes[HUGE_PRIME]
                BUCKET_SIZE = int(
                    BUCKET_INTERVAL
                    * (math.log(math.log(primes[-1]) / math.log(phuge)))
                    * 1.1
                )
                BUCKET_SIZE += 64 - BUCKET_SIZE % 64
                AVG_BUCKET_SIZE = BUCKET_INTERVAL * (
                    math.log(math.log(primes[-1]) / math.log(phuge))
                )
                logging.debug(
                    f"Huge prime index {HUGE_PRIME} ({phuge}) bucket size %d expect usage %d",
                    BUCKET_SIZE,
                    int(AVG_BUCKET_SIZE),
                )

        WORKCHUNK = 2 ** (AFACS - 1)

        XXL_PRIME = HUGE_PRIME
        xxl_length = 1
        if HUGE_PRIME is not None:
            XXL_PRIME, pxxl = HUGE_PRIME, primes[HUGE_PRIME]
            xxl_ratio = math.log(math.log(primes[-1]) / math.log(pxxl))
            xxl_length = int(xxl_ratio * INTERVAL_SIZE * WORKCHUNK * 1.1)
            xxl_length = (xxl_length // 2048) * 2048
            xxl_size = 4 * xxl_length
            logging.debug(
                f"XXL primes {len(primes) - XXL_PRIME} buffer size {xxl_size >> 20}MiB (hit rate {100 * xxl_ratio:.1f}%)"
            )

        self.roots_d = {p: r for p, r in zip(primes, roots)}

        self.stampPeriod = gpu.stamp_period()

        proc = current_process()
        proc_id = proc._identity or (0,)
        self.gpu_idx = proc_id[-1] % len(GPU_LOCK)
        mgr = kp.Manager(self.gpu_idx)
        gpu_name = mgr.get_device_properties().get("device_name", "unknown")
        logging.info(f"Worker {proc.name} running on GPU {self.gpu_idx} ({gpu_name})")
        xp = mgr.tensor_t(np.array(primes, dtype=np.uint32))
        xn = mgr.tensor_t(np.array(roots, dtype=np.uint32))
        assert AFACS >= 17 and AFACS & 1 == 1
        nsumroots = 1 + 16 + 16 + 16 + 16 + 2 * 2 ** (AFACS // 2 - 8)
        xr = mgr.tensor_t(np.zeros(len(roots) * nsumroots, dtype=np.uint32))
        xb = mgr.tensor_t(np.zeros(AFACS * BLEN, dtype=np.uint32))
        xargs = mgr.tensor_t(np.zeros(BLEN, dtype=np.uint32))
        # Huge offsets are u16
        BUCKETS = WORKCHUNK * INTERVAL_SIZE // BUCKET_INTERVAL
        xhuge = mgr.tensor_t(
            np.zeros(
                BUCKETS * BUCKET_SIZE // 2,
                dtype=np.uint32,
            )
        )
        xout = mgr.tensor_t(np.zeros(OUTSTRIDE, dtype=np.uint32))

        self.tensors = (xargs, xb, xout)

        # Output buffer to receive full sieve results.
        xdebug = None
        if DEBUG:
            logging.debug("sieve debug output enabled")
            xdebug = mgr.tensor_t(
                np.zeros(WORKCHUNK * INTERVAL_SIZE // 4, dtype=np.uint32)
            )

        # Send initial buffers (immutable)
        mgr.sequence().record(kp.OpTensorSyncDevice([xp, xn, xargs])).eval()

        defines = {
            "BLEN": BLEN,
            "AFACS": AFACS,
            "SEGMENT_SIZE": INTERVAL_SIZE,
            "SUBSEGMENT_SIZE": INTERVAL_SIZE,
            "HUGE_PRIME": HUGE_PRIME,
            "XXL_PRIME": XXL_PRIME,
            "BUCKET_SIZE": BUCKET_SIZE,
            "ITERS": ITERS,
            "THRESHOLD": THRESHOLD,
            "OUTSTRIDE": OUTSTRIDE,
            "BUCKET_INTERVAL": 8192,
        }
        if DEBUG:
            defines |= {"DEBUG": 1}
        self.defines = defines

        SHADER1 = gpu.compile("siqs2a_reset.comp", defines)
        SHADER2 = gpu.compile("siqs2b_roots.comp", defines)
        SHADER3 = gpu.compile("siqs2c_sort.comp", defines)
        # SHADER4 = gpu.compile("siqs2d_prefill.comp", defines)
        SHADER5 = gpu.compile("siqs2e_buckets.comp", defines)
        SHADER6 = gpu.compile("siqs2f_sieve.comp", defines)

        xfill = mgr.tensor_t(np.zeros(xxl_length, dtype=np.uint32))
        if DEBUG:
            print("debug output for sort shader")
            xsort = mgr.tensor_t(np.zeros(2 * len(primes) << AHALF, dtype=np.uint32))
            xsidx = mgr.tensor_t(np.zeros(len(primes) << AHALF, dtype=np.uint32))

        mem_main = 4 * (xp.size() + xr.size() + xn.size() + xout.size())
        mem_huge = 4 * (xfill.size() + xhuge.size())
        mem = mem_main + mem_huge

        logging.debug(
            f"Memory usage {mem >> 10} kB (main {mem_main >> 10} kB, huge {mem_huge >> 10} kB)"
        )

        self.mgr = mgr
        self.shader1 = mgr.algorithm(
            [xfill],
            SHADER1,
            workgroup=(1, 1, 1),
        )
        self.shader2 = mgr.algorithm(
            [xp, xn, xr, xb, xargs],
            SHADER2,
            workgroup=(len(primes) // 512 + 1, 1, 1),
        )
        self.shader3 = mgr.algorithm(
            [xp, xr, xfill] + ([xsort, xsidx] if DEBUG else []),
            SHADER3,
            workgroup=(SHARDS, 1, 1),
        )
        # self.shader4 = mgr.algorithm(
        #    [xp, xr, xfill, xsort, xsidx],
        #    SHADER4,
        #    workgroup=(SHARDS, 1, 1),
        # )
        self.shader5 = mgr.algorithm(
            [xfill, xhuge],
            SHADER5,
            workgroup=(SHARDS, 1, 1),
        )
        self.shader6 = mgr.algorithm(
            [xp, xr, xhuge, xout] + ([xdebug] if xdebug else []),
            SHADER6,
            workgroup=(BUCKETS, 1, 1),
        )
        self.factorpool = ThreadPoolExecutor()

    def process(self, ak):
        BLEN = self.wargs["BLEN"]
        OUTSTRIDE = self.wargs["OUTSTRIDE"]
        D, B1, B2 = self.wargs["D"], self.wargs["B1"], self.wargs["B2"]
        A, Bi = make_poly(D, ak, self.roots_d)
        if A.bit_length() + 2 > BLEN * 32:
            logging.error(f"Skipping A={A} (too large)")
            return 1.0, 0, []

        dt = self._run(ak, A, Bi)
        _, _, xout = self.tensors
        vout = xout.data()
        nreports, rows = self._process_sieve_reports(
            (A, ak, Bi), vout.astype(np.uint32), D, B1, B2, OUTSTRIDE
        )
        return dt, nreports, rows

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
            _A, _B, _C = expand_one_poly(N, A, Bi, poly_idx)
            v = _A * x * x + _B * x + _C
            u = 2 * _A * x + _B
            assert u * u == 4 * A * v + N
            # In the class group: (A, B, C) * (v, u, A) == 1
            row_async = self.factorpool.submit(build_relation, v, x, [], B1=B1, B2=B2)
            futures.append((_B, u, v, row_async))
        del u, v, x, _A, _B, _C, poly_idx
        for B, u, v, row_async in futures:
            row = row_async.result()
            # print(poly_idx, x, v, row)
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
                bp = B % ai
                if bp & 1 == 0:
                    row.append(-ai)
                else:
                    row.append(ai)
            results.append(row)
        return nout, results

    def _run(self, ak, A, Bi):
        AFACS = self.wargs["AFACS"]
        BLEN = self.wargs["BLEN"]

        xargs, xb, xout = self.tensors
        xargs.data()[:] = to_uvec(A, BLEN)
        for i in range(AFACS):
            xb.data()[BLEN * i : BLEN * (i + 1)] = to_uvec(Bi[i], BLEN)
        xout.data().fill(0)

        with GPU_LOCK[self.gpu_idx]:
            seq = (
                self.mgr.sequence(total_timestamps=16)
                .record(kp.OpTensorSyncDevice([xb, xout, xargs]))
                .record(kp.OpAlgoDispatch(self.shader1))
                .record(kp.OpAlgoDispatch(self.shader2))
                .record(kp.OpAlgoDispatch(self.shader3))
                # .record(kp.OpAlgoDispatch(self.shader4))
                .record(kp.OpAlgoDispatch(self.shader5))
                .record(kp.OpAlgoDispatch(self.shader6))
                .record(kp.OpTensorSyncLocal([xout]))
                .eval()
            )

        # Get GPU results (round to 0.5 tick if ticks are zero)
        stamps = seq.get_timestamps()
        if DEBUG_TIMINGS:
            times = []
            for t1, t2 in zip(stamps, stamps[1:]):
                times.append((t2 - t1) * self.stampPeriod * 1e-9)
            timestr = " ".join(f"{1000 * dt:.2f}ms" for dt in times)
            logging.debug(f"times {timestr}")
        dt = max(0.5, stamps[-1] - stamps[0]) * self.stampPeriod * 1e-9
        return dt


SIEVER = None


def worker_init(initargs):
    global SIEVER
    if initargs.get("USE_SIEVER2"):
        SIEVER = Siever2(initargs)
    else:
        SIEVER = Siever(initargs)


def worker_task(ak):
    return SIEVER.process(ak)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("-v", "--verbose", action="store_true")
    argp.add_argument("--check", action="store_true", help="Verify relations")
    argp.add_argument("--siever2", action="store_true", help="Use Siever2")
    argp.add_argument(
        "-j",
        metavar="THREADS",
        type=int,
        default=2,
        help="Number of CPU threads (and parallel GPU jobs)",
    )
    argp.add_argument(
        "--ngpu",
        metavar="GPUS",
        type=int,
        default=1,
        help="Number of GPUs (usually a divisor of THREADS)",
    )
    argp.add_argument("N", type=int)
    argp.add_argument("OUTDIR")
    args = argp.parse_args()

    main_impl(args)


def main_impl(args: argparse.Namespace):
    logging.getLogger().setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    while len(GPU_LOCK) < args.ngpu:
        GPU_LOCK.append(Semaphore(1))

    N = args.N
    D = -abs(N)
    bias = smoothness_bias(D)
    logging.info(f"Sieve smoothness bias is {bias:.2f} bits")
    USE_SIEVER2 = args.siever2
    if not USE_SIEVER2:
        B1, B2k, OUTSTRIDE, EXTRA_THRESHOLD, AFACS, ITERS, POLYS_PER_WG = get_params(
            N, bias
        )
        SEGMENT_SIZE = 16384  # constant
        SUBSEGMENT_SIZE = 8192  # constant
        INTERVAL_SIZE = ITERS * SEGMENT_SIZE
    else:
        B1, B2k, EXTRA_THRESHOLD, AFACS, INTERVAL_SIZE = get_params2(N, bias)
        ITERS = 1
        OUTSTRIDE = 256
        SEGMENT_SIZE = None
        SUBSEGMENT_SIZE = None
        POLYS_PER_WG = None

    B2 = B1 * B2k
    M = INTERVAL_SIZE // 2

    THRESHOLD = (
        N.bit_length() // 2 + M.bit_length() - 2 * B1.bit_length() - EXTRA_THRESHOLD
    )

    del N

    primes = []
    roots = []
    for p, k in algebra.primebase(D, B1):
        assert (k * k - D) % p == 0
        primes.append(p)
        roots.append(k)
    logging.debug(
        f"Prime base size {len(primes)} {primes[:10]} ... {primes[-1]}, B2={B2 / 1e6:.1f}M"
    )
    logging.debug(f"Interval size {M // 512}k")

    A0 = math.isqrt(abs(D)) // (2 * M)
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
                "d": D,
            },
            w,
        )
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

    if args.j:
        NWORKERS = args.j
    else:
        NWORKERS = 8 if D.bit_length() < 270 else 3
    sieve_pool = Pool(NWORKERS, initializer=worker_init, initargs=(WARGS,))
    sieved = 0
    start_time = time.monotonic()
    last_print = time.monotonic()
    last_sieved = 0
    next_check = 0
    nb = 0
    done = False

    if len(primes) < 256:
        TARGET_GAP = 64
    else:
        TARGET_GAP = 4 * len(primes)
    Aseen: set[tuple[int]] = set()
    while not done:
        As = make_a(primes, A0, AFACS, Aseen)
        Aseen.update(As)

        for dt, nreports, rows in sieve_pool.imap_unordered(worker_task, As):
            for row in rows:
                all_primes.update(abs(_p) for _p in row)
                print(" ".join(str(_x) for _x in row), file=w)
                results.append(row)

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
                    f"Chunk {nb} ({sieved / 1e6:.0f}M) done in {dt:.3f}s ({cur_speed / 1e9:.1f}G/s avg {avg_speed / 1e9:.1f}G/s) {len(rows)}/{nreports} items (relations={_nr} [{rel_speed:.3g}/s] primes={_np} excess={_nr - _np})"
                )

            # Check if finished
            gap = _nr - _np
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
    print(
        f"Sieved {nb} A ({sieved / 1e6:.0f}M) in {dt:.3f}s (avg {avg_speed / 1e9:.1f}G/s) (relations={_nr} [{rel_speed:.1f}/s] primes={_np} excess={_nr - _np})"
    )

    total_time = time.monotonic() - start_time
    logging.info(
        f"Got {len(results)} relations with {len(all_primes)} primes in {total_time:.3f}s"
    )

    if args.check:
        forms = {}

        def ideal(p):
            if p not in forms:
                forms[p] = flint_extras.qfb.prime_form(D, p)
            return forms[p]

        for _row in results:
            q = ideal(1)
            for p in _row:
                if p > 0:
                    q = q * ideal(p)
                else:
                    q = q * ideal(-p) ** -1
            # q must be the unit form
            assert q.q()[0] == 1, (_row, q)
        logging.info(f"Checked {len(results)} class group relations")

    if args.verbose:
        for _row in results[:16]:
            print(_row)
        print("...")
        for _row in results[-16:]:
            print(_row)


def check():
    argp = argparse.ArgumentParser()
    argp.add_argument("DATADIR")
    args = argp.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    datadir = pathlib.Path(args.DATADIR)
    with open(datadir / "args.json") as j:
        clsargs = json.load(j)
    D = clsargs["d"]
    logging.info(f"D = {D}")

    forms = {}

    def ideal(p):
        if p not in forms:
            forms[p] = flint_extras.qfb.prime_form(D, p)
        return forms[p]

    t0 = time.monotonic()
    count = 0
    with open(datadir / "relations.sieve") as f:
        for line in f:
            row = [int(l) for l in line.split()]
            q = ideal(1)
            for p in row:
                if p > 0:
                    q = q * ideal(p)
                else:
                    q = q * ideal(-p) ** -1
            # q must be the unit form
            assert q.q()[0] == 1, (row, q)
            count += 1
    dt = time.monotonic() - t0
    logging.info(f"Checked {count} relations in {dt:.3f}s")

    if (datadir / "relations.filtered").is_file():
        logging.info("Checking filtered relations")
        t0 = time.monotonic()
        count = 0
        with open(datadir / "relations.filtered") as f:
            for l in f:
                facs = l.split()
                q = ideal(1)
                for f in facs:
                    p, _, e = f.partition("^")
                    q = q * ideal(int(p)) ** int(e)
                # q must be the unit form
                assert q.q()[0] == 1, (row, q)
                count += 1
        dt = time.monotonic() - t0
        logging.info(f"Checked {count} relations in {dt:.3f}s")


if __name__ == "__main__":
    main()
