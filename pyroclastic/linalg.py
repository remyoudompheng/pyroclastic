"""
Linear algebra step

The input of this step is a sparse matrix M (rows > cols).

The main algorithm is ordinary Wiedemann with the following cases:

- multiple word sized moduli (32 or 64 bits)
- 1 big integer modulus (W words)

Typical matrix sizes:
D = 240 bits => dim 2000
D = 280 bits => dim 6000
D = 320 bits => dim 10000
D = 360 bits => dim 25000
D = 400 bits => dim 50000
"""

from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import logging
import math
import pathlib
import random
import time
from multiprocessing import Pool, Semaphore, current_process

import flint
import kp
import numpy as np

from . import algebra
from . import linalg_alt
from . import lingen
from . import gpu
from . import relations
import pyroclastic_flint_extras as flint_extras


DEBUG_NO_SORT_ROWS = False


def to_sparse_matrix(rels, square=True):
    """
    Converts a list of relations into a representation suitable
    for sparse matrix kernels.

    The matrix rows may correspond to an unspecified permutation
    of input relations.
    """
    stats = {}
    for r in rels:
        for p in r:
            if r[p]:
                stats[p] = stats.get(p, 0) + abs(r[p])
    # Find densest columns above 33% fill ratio
    dense_counts = []
    for p, count in stats.items():
        if count > len(rels) // 3:
            dense_counts.append((count, p))
    dense_counts.sort()
    dense_p = sorted([p for _, p in dense_counts[len(dense_counts) % 4 :]])
    assert len(dense_p) % 4 == 0

    dense_weight = sum(stats[p] for p in dense_p) / float(len(rels))
    logging.debug(f"Dense columns for {len(dense_p)} primes {dense_p}")
    logging.info(
        f"Dense block has {len(dense_p)} columns, average weight {dense_weight:.1f} per row"
    )
    sparse_weight = sum(
        sum(abs(e) for p, e in r.items() if p not in dense_p) for r in rels
    ) / float(len(rels))
    logging.info(f"Sparse block has avg weight {sparse_weight:.1f} per row")
    norm_plus = max(sum(abs(e) for p, e in r.items() if e > 0) for r in rels)
    norm_minus = max(sum(abs(e) for p, e in r.items() if e < 0) for r in rels)

    # To reduce divergence, we sort rows by the number of ±signs in the sparse part.
    dense_set = frozenset(dense_p)
    sign_rels = []
    for r in rels:
        nplus, nminus = 0, 0
        for _p, _e in r.items():
            if _p not in dense_p:
                if _e > 0:
                    nplus += 1
                else:
                    nminus += 1
        sign_rels.append((nplus, nminus, r))
    if not DEBUG_NO_SORT_ROWS:
        sign_rels.sort(key=lambda t: t[:2])
    # print([(x, y) for x, y, z in sign_rels])
    rels = [_r for _, _, _r in sign_rels]

    # Dense coefficients must fit in int8 type
    for r in rels:
        for p in dense_p:
            if p in r:
                assert abs(r[p]) < 127

    dense = np.zeros((len(rels), len(dense_p)), dtype=np.int8)
    for i, r in enumerate(rels):
        dense[i, :] = [r.get(p, 0) for p in dense_p]
    dense_norm = max(np.sum(np.abs(dense[i, :])) for i in range(len(rels)))
    logging.info(f"Dense block has max row norm {dense_norm}")

    primes = dense_p + sorted(p for p in stats if p not in dense_set)
    if square:
        dim = len(primes)
        assert dim == len(rels)
    else:
        dim = len(rels)
        assert dim >= len(primes)
    # col_density = np.array([stats[p] / len(rels) for p in primes])
    # with np.printoptions(precision=5):
    #    print("Column densities (dense)")
    #    print(col_density[: len(dense_p)])
    #    print("Column densities (sparse)")
    #    print(col_density[len(dense_p) : dim // 3], "...", col_density[2 * dim // 3 :])

    prime_idx = {p: idx for idx, p in enumerate(primes)}
    plus = []
    minus = []
    for r in rels:
        row_p, row_m = [], []
        for p, e in r.items():
            if p in dense_set:
                continue
            idx = prime_idx[p]
            if e > 0:
                row_p.extend(e * [idx])
            else:
                row_m.extend(-e * [idx])
        row_p.sort()
        row_m.sort()
        plus.append(row_p)
        minus.append(row_m)
    return primes, dense, plus, minus, max(norm_plus, norm_minus)


def check_wiedemann(sequence, poly, p):
    # Check polynomial
    if (p * p * len(poly)).bit_length() < 64:
        poly_v = np.array(poly, dtype=np.uint64)
        conv = np.convolve(
            np.array(sequence, dtype=np.uint64),
            poly_v[::-1],
            mode="valid",
        )
        return np.all(conv % p == 0)
    else:
        ok = True
        # Too slow to test all convolutions.
        for i in range(len(poly) - 1, len(sequence), len(sequence) // 7):
            conv_i = sum(
                poly[-j] * sequence[i + 1 - j] for j in range(1, 1 + len(poly))
            )
            ok = ok and conv_i % p == 0
        return ok


class CSRMatrix:
    """
    Matrix represented as
    - a block of dense columns (int8 coefficients)
    - an array of sparse rows (int16 columns indices with sign for ±1 coefficient)

    2 kernels are implemented:
    - small variant where entire input/output fit in registers + shmem
    - medium variant where input vector in loaded in chunks
    """

    def __init__(self, dense, plus, minus, basis, weight, gpu_idx=0):
        dim, dense_n = dense.shape
        assert dim < 32768
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        # Merge back +1/-1 coefficients
        sparses = []
        for rp, rm in zip(plus, minus):
            row = rp + [-p for p in rm]
            row.sort(key=abs)
            sparses.append(row)

        rowlens = [len(l) for l in sparses]
        aidx = np.cumsum(np.array([0] + rowlens, dtype=np.uint32), dtype=np.uint32)
        sparsesize = sum(rowlens)
        sparsesize += sparsesize & 1
        arows = np.zeros(sparsesize, dtype=np.int16)
        for i, l in enumerate(sparses):
            arows[aidx[i] : aidx[i + 1]] = l
        # Kompute wants uint32, cast arrays to make it happy
        xrows = mgr.tensor_t(arows.view(np.uint32))
        xidx = mgr.tensor_t(aidx)

        self.mgr = mgr
        self.tensors = [xd, xrows, xidx]
        self.flops = 2 * dim * dense_n + sparsesize
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight:.1f})"
        )

    def krylov(self, ls: list[int], blockm=1, lock=None, bench=False):
        "Perform Wiedemann algorithm for multiple small moduli"
        MODULI = len(ls)
        if any(l.bit_length() > 32 for l in ls):
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32
        mgr = self.mgr
        dim = self.dim
        # assert dim >= 256
        if dim < 5000:
            BATCHSIZE = 64
        elif dim < 10000:
            BATCHSIZE = 32
        else:
            BATCHSIZE = 16
        ITERS = dim + dim // blockm + 1
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE
        if bench:
            ITERS = 1024
        # Tensor holding M^k V
        xv = mgr.tensor_t(np.zeros(dim * MODULI, dtype=word_t).view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(MODULI, dtype=np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        SEL_BLOCK = 256
        sel = np.zeros(dim // SEL_BLOCK // 8 * 8 + 16, dtype=np.uint8)
        for i in range(len(sel)):
            if i * SEL_BLOCK < dim:
                sel[i] = random.randrange(min(dim - i * SEL_BLOCK, SEL_BLOCK))
        xsel = mgr.tensor_t(sel.view(np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(
            np.zeros(MODULI * blockm * ITERS, dtype=np.uint64).view(np.uint32)
        )
        xmod = mgr.tensor_t(np.array(ls, dtype=word_t).view(np.uint32))

        xd, xrows, xidx = self.tensors
        defines = self.defines | {"BATCHSIZE": BATCHSIZE, "BLOCKM": blockm}
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv_small.comp", defines)
        algo = mgr.algorithm(
            [xd, xrows, xidx, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(MODULI, 1, 1),
        )
        (
            mgr.sequence()
            .record(kp.OpTensorSyncDevice([xd, xrows, xidx, xsel, xout, xmod]))
            .eval()
        )

        v = xv.data().view(word_t).reshape((MODULI, dim))
        for i, l in enumerate(ls):
            v[i, :] = np.random.randint(0, l, dim, dtype=word_t)
        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (xd.size() + xrows.size() + xidx.size())
        vec_size = 4 * xv.size()
        logging.debug(
            f"Buffer sizes: matrix {mat_size >> 10}kB vectors {vec_size >> 10}kB"
        )

        t0 = time.monotonic()
        gpu_ticks = 0.0
        with lock or nullcontext():
            for i in range(0, ITERS, BATCHSIZE):
                # Matrix multiplication is very fast so we launch multiple
                # iterations per batch.
                seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
                seq.record(kp.OpAlgoDispatch(algo))
                seq.eval()

                stamps = seq.get_timestamps()
                gpu_ticks += stamps[-1] - stamps[0]

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().view(np.uint64).reshape((ITERS, MODULI, blockm))

        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / gpu_dt
        speed = ITERS * MODULI / gpu_dt
        dt = time.monotonic() - t0

        logging.info(
            f"Krylov completed in {dt:.3}s (GPU {gpu_dt:.3f}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )

        return {
            l: [vout[:, lidx, j] for j in range(blockm)]
            for lidx, l in enumerate(ls)
        }

    def matmul_small(self, l: int, v):
        "For testing"
        MODULI = 1
        if l.bit_length() > 32:
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32
        mgr = self.mgr
        dim = self.dim
        xd, xrows, xidx = self.tensors
        # Tensor holding M^k V
        xv = mgr.tensor_t(np.zeros(dim, dtype=word_t).view(np.uint32))
        xv.data().view(word_t)[:] = v
        xiter = mgr.tensor_t(np.zeros(MODULI, dtype=np.uint32))
        # Ignored tensors
        xsel = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xiter, xout = xsel, xsel
        xmod = mgr.tensor_t(np.array([l], dtype=np.uint32))

        defines = self.defines | {"BATCHSIZE": 1}
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv_small.comp", defines)
        algo = mgr.algorithm(
            [xd, xrows, xidx, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(1, 1, 1),
        )
        (
            mgr.sequence()
            .record(
                kp.OpTensorSyncDevice([xd, xrows, xidx, xv, xiter, xsel, xout, xmod])
            )
            .record(kp.OpAlgoDispatch(algo))
            .record(kp.OpTensorSyncLocal([xv]))
            .eval()
        )
        return np.copy(xv.data().view(word_t))

    def wiedemann_medium(self, ls: list[int], check=False, lock=None):
        MODULI = len(ls)
        if any(l.bit_length() > 32 for l in ls):
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32
        mgr = self.mgr
        dim = self.dim
        assert dim >= 256
        if dim < 30000:
            BATCHSIZE = 8
        else:
            BATCHSIZE = 4
        N_CHUNKS = dim // 16384 + 1
        CHUNK_N = dim // N_CHUNKS + 1
        assert CHUNK_N < 16000
        assert self.defines["DENSE_N"] < CHUNK_N and CHUNK_N >= 64
        ITERS = 2 * dim
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE
        # Tensor holding M^k V
        xv = mgr.tensor_t(np.zeros(dim * MODULI, dtype=word_t).view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(MODULI, dtype=np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        SEL_BLOCK = 1024
        sel = np.zeros(dim // SEL_BLOCK // 8 * 8 + 16, dtype=np.uint8)
        for i in range(dim // SEL_BLOCK):
            # always zero on last workgroup
            sel[i] = random.randrange(256)
        xsel = mgr.tensor_t(sel.view(np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(MODULI * ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array(ls, dtype=word_t).view(np.uint32))

        xd, xrows, xidx = self.tensors
        defines = self.defines | {"BATCHSIZE": BATCHSIZE, "CHUNK_N": CHUNK_N}
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv_csr_medium.comp", defines)
        algo = mgr.algorithm(
            [xd, xrows, xidx, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(MODULI, 1, 1),
        )
        (
            mgr.sequence()
            .record(kp.OpTensorSyncDevice([xd, xrows, xidx, xsel, xout, xmod]))
            .eval()
        )

        v = xv.data().view(word_t).reshape((MODULI, dim))
        for i, l in enumerate(ls):
            v[i, :] = np.random.randint(0, l, dim, dtype=word_t)
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (xd.size() + xrows.size() + xidx.size())
        vec_size = 4 * xv.size()
        logging.debug(
            f"Buffer sizes: matrix {mat_size >> 10}kB vectors {vec_size >> 10}kB"
        )

        t0 = time.monotonic()
        gpu_ticks = 0.0
        with lock or nullcontext():
            for i in range(0, ITERS, BATCHSIZE):
                # Matrix multiplication is very fast so we launch multiple
                # iterations per batch.
                seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
                for _ in range(BATCHSIZE):
                    seq.record(kp.OpAlgoDispatch(algo))
                seq.eval()

                stamps = seq.get_timestamps()
                gpu_ticks += stamps[-1] - stamps[0]

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().view(np.uint64).reshape((ITERS, MODULI))

        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / gpu_dt
        speed = ITERS * MODULI / gpu_dt

        dets = []
        for i, li in enumerate(ls):
            sequence = [int(x) % li for x in vout[:, i]]

            poly = flint_extras.berlekamp_massey(sequence, li)
            assert len(poly) <= dim + 1, len(poly)
            # polys.append(poly)
            assert len(poly) == dim + 1
            det = -poly[0] * pow(poly[dim], -1, li) % li
            dets.append(det)
            if check:
                assert check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1, len(poly)
                det = -poly[0] * pow(poly[dim], -1, li) % li
                if i < 5 or i > len(ls) - 5:
                    logging.info(
                        f"Check Wiedemann modulo {li} OK: det(M % {li}) = {det}"
                    )

        dt = time.monotonic() - t0
        logging.info(
            f"Wiedemann completed in {dt:.3}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return dets, ls

    def matmul_medium(self, l: int, v, CHUNK_N=None):
        "For testing"
        if l.bit_length() > 32:
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32
        mgr = self.mgr
        dim = self.dim
        if CHUNK_N is None:
            CHUNK_N = dim // 16384 + 1
        assert CHUNK_N < 16000
        assert self.defines["DENSE_N"] < CHUNK_N and CHUNK_N >= 64
        xd, xrows, xidx = self.tensors
        # Tensor holding M^k V
        xv = mgr.tensor_t(np.zeros(dim, dtype=word_t).view(np.uint32))
        xv.data().view(word_t)[:] = v
        # Ignored tensors
        xsel = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xiter, xout = xsel, xsel
        xmod = mgr.tensor_t(np.array([l], dtype=np.uint32))

        defines = self.defines | {"BATCHSIZE": 1, "CHUNK_N": CHUNK_N}
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv_csr_medium.comp", defines)
        algo = mgr.algorithm(
            [xd, xrows, xidx, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(1, 1, 1),
        )
        (
            mgr.sequence()
            .record(kp.OpTensorSyncDevice([xd, xrows, xidx, xv, xsel, xout, xmod]))
            .record(kp.OpAlgoDispatch(algo))
            .record(kp.OpTensorSyncLocal([xv]))
            .eval()
        )
        return np.copy(xv.data().view(word_t))


class BlockCOO:
    """
    Matrix represented as
    - a block of dense columns (int8 coefficients)
    - a list of blocks stored in COO format (row: 10b, column: 21b, sign: 1b)

    Each workgroup handles a block and at most 16 moduli (u32) or 8 moduli (u64)
    """

    def __init__(self, dense, plus, minus, basis, weight, gpu_idx=0):
        dim, dense_n = dense.shape
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        # Kompute wants uint32, cast arrays to make it happy
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        if dim < 10000:
            BM = 512
        else:
            BM = 1024
        self.BM = BM

        assert len(plus) == dim
        assert len(minus) == dim
        sparses = []
        block = []
        block_idx = []
        for i in range(dim):
            if i % BM == 0:
                block.sort(key=lambda x: x & 0x7FFFFFFF)
                sparses.extend(block)
                block_idx.append(len(sparses))
                block = []
            block.extend((i % BM) + BM * j for j in plus[i])
            block.extend((i % BM) + BM * j + 2**31 for j in minus[i])
        if dim % BM != 0:
            block.sort(key=lambda x: x & 0x7FFFFFFF)
            sparses.extend(block)
            block_idx.append(len(sparses))
        assert len(sparses) == sum(len(l) for l in plus + minus)
        assert len(block_idx) == 1 + (dim + BM - 1) // BM
        logging.debug(
            f"Block sizes {[j - i for i, j in zip(block_idx, block_idx[1:])]}"
        )
        # print("Deltas")
        # print("plus ", max(j - i for i, j in zip(aplus, aplus[16:])))
        # print("minus", max(j - i for i, j in zip(aminus, aminus[16:])))

        xsparse = mgr.tensor_t(np.array(sparses, dtype=np.uint32))
        xidx = mgr.tensor_t(np.array(block_idx, dtype=np.uint32))

        self.mgr = mgr
        self.tensors = [xd, xsparse, xidx]
        self.flops = 2 * dim * dense_n + len(sparses)
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def mulvec(self, vi, l: int):
        BM = self.BM
        MODULI = 1
        if l.bit_length() > 32:
            INT64 = True
            word_t = np.uint64
            assert self.BM * 8 <= 65536
        else:
            INT64 = False
            word_t = np.uint32
            assert self.BM * 4 <= 65536

        mgr = self.mgr
        dim = self.dim
        # assert dim >= 256

        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2 * MODULI, dtype=word_t).view(np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_STRIPES = (dim + BM - 1) // BM
        # Ignored tensors
        xsel = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xiter, xout = xsel, xsel
        xmod = mgr.tensor_t(np.array([l], dtype=word_t).view(np.uint32))

        xd, xsparse, xidx = self.tensors
        defines = self.defines | {"BM": BM, "MODULI": MODULI}
        if INT64:
            defines["INT64"] = 1
        kernel = gpu.compile("spmv_blockcoo3.comp", defines)
        algo = mgr.algorithm(
            [xd, xsparse, xidx, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(N_STRIPES, 1, 1),
        )
        v = xv.data().view(word_t).reshape((2, 1, dim))
        v[0, 0, :] = vi
        (
            mgr.sequence()
            .record(kp.OpTensorSyncDevice([xd, xsparse, xidx, xv, xsel, xout, xmod]))
            .record(kp.OpAlgoDispatch(algo))
            .record(kp.OpTensorSyncLocal([xv]))
            .eval()
        )

        v = xv.data().view(word_t).reshape((2, 1, dim))
        return np.copy(v[1, 0, :])

    def krylov(self, ls: list[int], blockm=1, lock=None, bench=False):
        """
        Perform Wiedemann algorithm for multiple small moduli

        Variant with 1 workgroup per modulus.
        """
        BM = self.BM
        MODULI = len(ls)
        if ls[0].bit_length() > 32:
            INT64 = True
            word_t = np.uint64
            MOD_WG = 1 if len(ls) <= 8 else len(ls) // 8
            assert self.BM * (MODULI // MOD_WG) * 8 <= 65536
        else:
            INT64 = False
            word_t = np.uint32
            MOD_WG = 1 if len(ls) <= 16 else len(ls) // 16
            assert self.BM * (MODULI // MOD_WG) * 4 <= 65536

        MODULI_STRIDE = MODULI // MOD_WG

        mgr = self.mgr
        dim = self.dim
        assert dim >= 256

        if dim < 10000:
            BATCHSIZE = 64
        elif dim < 30000:
            BATCHSIZE = 16
        else:
            BATCHSIZE = 8
        ITERS = dim + dim // blockm + 16
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE
        if bench:
            ITERS = 1024
        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2 * MODULI, dtype=word_t).view(np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_STRIPES = (dim + BM - 1) // BM
        N_WG = N_STRIPES
        sel = np.zeros(N_STRIPES // 8 * 8 + 16, dtype=np.uint8)
        for i in range(dim // BM):
            # always zero on last workgroup
            sel[i] = random.randrange(min(256, BM))
        xsel = mgr.tensor_t(sel.view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(N_WG * MOD_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(
            np.zeros(MODULI * blockm * ITERS, dtype=np.uint64).view(np.uint32)
        )
        xmod = mgr.tensor_t(np.array(ls, dtype=word_t).view(np.uint32))

        xd, xsparse, xidx = self.tensors
        defines = self.defines | {
            "BM": BM,
            "MODULI": MODULI // MOD_WG,
            "BLOCKM": blockm,
        }
        if INT64:
            defines["INT64"] = 1
        kernel = gpu.compile("spmv_blockcoo3.comp", defines)
        algo = mgr.algorithm(
            [xd, xsparse, xidx, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(N_WG, MOD_WG, 1),
        )
        (
            mgr.sequence()
            .record(kp.OpTensorSyncDevice([xd, xsparse, xidx, xsel, xout, xmod]))
            .eval()
        )

        v = xv.data().view(word_t).reshape((MOD_WG, 2, dim, MODULI // MOD_WG))
        for i, l in enumerate(ls):
            v[i // MODULI_STRIDE, 0, :, i % MODULI_STRIDE] = np.random.randint(
                0, l, dim, dtype=word_t
            )
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (xd.size() + xsparse.size() + xidx.size())
        vec_size = 4 * xv.size()
        logging.debug(
            f"Buffer sizes: matrix {mat_size >> 10}kB vectors {vec_size >> 10}kB"
        )

        t0 = time.monotonic()
        gpu_ticks = 0.0
        with lock or nullcontext():
            for i in range(0, ITERS, BATCHSIZE):
                # Matrix multiplication is very fast so we launch multiple
                # iterations per batch.
                seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
                for _ in range(BATCHSIZE):
                    seq.record(kp.OpAlgoDispatch(algo))
                seq.eval()

                stamps = seq.get_timestamps()
                gpu_ticks += stamps[-1] - stamps[0]

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().view(np.uint64).reshape((ITERS, MODULI, blockm))

        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / gpu_dt
        speed = ITERS * MODULI / gpu_dt

        dt = time.monotonic() - t0
        logging.info(
            f"Krylov completed in {dt:.3f}s (GPU {gpu_dt:.3f}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return {
            l: [vout[:, lidx, j] for j in range(blockm)] for lidx, l in enumerate(ls)
        }


def update_crt(bigres, bigmod, res, mod):
    def product(l: list[int]):
        p = l[0]
        for x in l[1:]:
            p *= x
        return p

    M = product(mod)
    crt = sum(ri * pow(M // mi, -1, mi) * (M // mi) for ri, mi in zip(res, mod))
    crt = crt % M

    MM = bigmod * M
    z = crt * pow(bigmod, -1, M) * bigmod + bigres * pow(M, -1, bigmod) * M
    assert z % bigmod == bigres % bigmod
    assert all(z % mi == ri for ri, mi in zip(res, mod))
    z = z % MM
    if z > MM // 2:
        z -= MM
    return z, MM


WORKER_M = None
BLOCKM = 2

GPU_LOCK = [Semaphore(1)]


def worker_init(*initargs):
    global WORKER_M
    dim = len(initargs[1])
    proc = current_process()
    gpu_idx = proc._identity[-1] % len(GPU_LOCK)
    if gpu.is_discrete_gpu():
        # Simple CSR kernel is always faster on dGPU
        WORKER_M = linalg_alt.SpMV(*initargs, gpu_idx=gpu_idx)
    elif dim < gpu.max_shmem() // 4:
        WORKER_M = CSRMatrix(*initargs, gpu_idx=gpu_idx)
    else:
        WORKER_M = BlockCOO(*initargs, gpu_idx=gpu_idx)
    WORKER_M.gpu_idx = gpu_idx


def worker_task(moduli):
    return WORKER_M.krylov(moduli, blockm=BLOCKM, lock=GPU_LOCK[WORKER_M.gpu_idx])


def detz(subrels, threads, cputhreads: int | None, logfile=None):
    t0 = time.monotonic()

    weight = sum(len(r) for r in subrels)
    basis, dense, plus, minus, norm = to_sparse_matrix(subrels)
    dim = len(basis)
    BATCH_SIZE = 64
    if dim >= 16384 and not gpu.is_discrete_gpu():
        BATCH_SIZE = 32
    if dim <= 2048:
        BATCH_SIZE = 16

    if gpu.has_fast_add64():
        # Use 64-bit moduli (2.3x larger) when possible
        logging.info(f"Using 64-bit arithmetic on {gpu.device_name()}")
        BATCH_SIZE //= 2
        max_mod = (2**63) // norm
    else:
        max_mod = (2**31) // norm

    # Determinant is never larger than O(dim) bits
    moduli = [
        x for x in range(max_mod - 100 * dim, max_mod) if flint.fmpz(x).is_prime()
    ]
    logging.debug(f"Prepared {len(moduli)} small prime moduli for determinant")

    margs = (dense, plus, minus, basis, weight)
    detmod, mod = 0, 1
    done = 0
    # Create new locks for each pool
    for i in range(len(GPU_LOCK)):
        GPU_LOCK[i] = Semaphore(1)
    logging.info(f"Expected Krylov sequence length {dim + dim // BLOCKM + 1}")
    with Pool(threads, initializer=worker_init, initargs=margs) as mat_pool:
        with ProcessPoolExecutor(max_workers=cputhreads) as lingen_pool:
            for krys in mat_pool.imap_unordered(
                worker_task, itertools.batched(moduli, BATCH_SIZE), chunksize=1
            ):
                jobs = []
                for l, seqs in krys.items():
                    ljob = lingen_pool.submit(lingen.generating_polynomial_multi,
                                       [list(map(int, s)) for s in seqs],
                                       dim, l)
                    jobs.append((l, ljob))

                dets, mods = [] ,[]
                for l, ljob in jobs:
                    pol = ljob.result()
                    if len(pol) != dim + 1:
                        logging.error(f"Failed determinant for modulus {l}")
                        continue
                    # Ignore (-1)^dim sign
                    dets.append(pol[0])
                    mods.append(l)

                if logfile:
                    for _det, _mod in zip(dets, mods):
                        print(f"mod {_mod} det {_det}", file=logfile)
                detmod, mod = update_crt(detmod, mod, dets, mods)
                dt = time.monotonic() - t0
                done += len(mods)
                logging.info(f"Computed determinants for {done} moduli in {dt:.3f}s")
                if detmod.bit_length() + 128 < mod.bit_length():
                    logging.info(f"Found determinant (size {detmod.bit_length()} bits)")
                    return detmod


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "-j", metavar="THREADS", default=2, type=int, help="Number of parallel GPU jobs"
    )
    p.add_argument(
        "--ngpu",
        metavar="GPUS",
        type=int,
        default=1,
        help="Number of GPUs (usually a divisor of THREADS)",
    )
    p.add_argument(
        "--ncpu",
        type=int,
        default=None,
        help="Number of CPU threads for (block) Wiedemann",
    )
    p.add_argument("--bench", action="store_true")
    p.add_argument("--checkbench", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("DATADIR")
    args = p.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main_impl(args)


def main_impl(args):
    if args.deterministic:
        random.seed(1)

    while len(GPU_LOCK) < args.ngpu:
        GPU_LOCK.append(Semaphore(2))

    rels = []
    datadir = pathlib.Path(args.DATADIR)

    with open(datadir / "args.json") as j:
        clsargs = json.load(j)
    D = clsargs["d"]
    logging.info(f"D = {D}")

    if (datadir / "relations.filtered").is_file():
        with open(datadir / "relations.filtered") as f:
            for l in f:
                facs = l.split()
                rels.append(
                    {int(p): int(e) for p, _, e in (f.partition("^") for f in facs)}
                )

        logging.info(f"Imported {len(rels)} relations")

    else:
        with open(datadir / "relations.sieve") as f:
            for l in f:
                rels.append([int(x) for x in l.split()])

        logging.info(f"Imported {len(rels)} relations (before prune/filter)")
        pruned, _ = relations.step_prune(rels, 0, 0, pathlib.Path(args.DATADIR))
        filtered = relations.step_filter(pruned, pathlib.Path(args.DATADIR))
        rels = filtered

    if args.bench:
        bench(rels, check=args.checkbench)
        return

    basis1 = sorted(set(p for r in rels for p, e in r.items() if e))
    dim = len(basis1)

    # Compute at precision 0.01-0.001
    h_app = algebra.h_approx(D, 100 * max(100, D.bit_length()) ** 2)
    logging.info(f"Using approximate class number {h_app}")

    bigdets = []
    d = 0
    mat_idx = 0
    while d == 0 or d > 10000 * h_app:
        mat_idx += 1
        w = open(datadir / f"determinant.{mat_idx}", "w", buffering=1)
        logging.info(f"Selecting new square submatrix (dim {dim})")
        while True:
            subrels_idx = random.sample(range(len(rels)), dim)
            subrels = [rels[idx] for idx in subrels_idx]
            if len(basis1) == len(set(p for r in subrels for p, e in r.items() if e)):
                removed = set(range(len(rels))) - set(subrels_idx)
                logging.info(f"Square submatrix has removed rows {sorted(removed)}")
                print(f"submatrix {sorted(removed)}", file=w)
                break

        detM = detz(subrels, threads=args.j, cputhreads=args.ncpu, logfile=w)
        if detM == 0:
            logging.error("Matrix has zero determinant, trying again")
            continue
        bigdets.append(detM)
        d = int(flint.fmpz(d).gcd(flint.fmpz(detM)))
        if len(bigdets) > 1:
            logging.info(f"Multiple of group order {d}")

    k1 = max(1, int(math.floor(d / h_app * 0.95)) - 10)
    k2 = int(math.ceil(d / h_app * 1.05)) + 10
    best = 999, None
    for k in range(k1, k2):
        if d % k == 0 and 0.8 < d / k / h_app < 1.2:
            h = d // k
            # Note that is_probable_class_number only checks
            # that h is a multiple of the exponent of the group.
            # If the group is not cyclic, multiple values can pass this check.
            ok = algebra.is_probable_class_number(D, h)
            if ok:
                logging.info(f"Found probable class number {h=}")
                if abs(h / h_app - 1.0) < best[0]:
                    best = abs(h / h_app - 1.0), h
            else:
                logging.debug(f"Rejected candidate h={h}")

    h = best[1]
    logging.info(f"Found class number {h=}")
    with open(datadir / "classnumber", "w") as f:
        print(h, file=f)


def bench(rels, check=False):
    from .linalg_alt import SpMV

    random.seed(42)

    basis1 = sorted(set(p for r in rels for p, e in r.items() if e))
    dim = len(basis1)
    print(f"Truncating to square matrix (dim {dim})")
    while True:
        subres = random.sample(rels, dim)
        if len(basis1) == len(set(p for r in subres for p, e in r.items() if e)):
            rels = subres
            break

    weight = sum(len(r) for r in rels)
    basis, dense, plus, minus, norm = to_sparse_matrix(rels)
    assert sorted(basis) == basis1

    max_mod32 = (2**31) // norm
    max_mod64 = (2**63) // norm
    modratio = math.log2(max_mod64) / math.log2(max_mod32)
    logging.info(
        f"Matrix has norm {norm} max modulus32 {max_mod32} max64 {max_mod64} ratio {modratio:.2f}"
    )

    # print("Dense block")
    # print(dense)
    # print("Rows +1")
    # print(plus[:10], "...")
    # print("Rows -1")
    # print(minus[:10], "...")

    CHECK = check
    BENCH = not CHECK

    moduli = [
        x for x in range(max_mod32 - 10000, max_mod32) if flint.fmpz(x).is_prime()
    ]
    moduli64 = [
        x for x in range(max_mod64 - 10000, max_mod64) if flint.fmpz(x).is_prime()
    ][:3] + moduli[3:]

    BLOCKM = 1
    def wiedemann_multi(M, *args, **kwargs):
        nonlocal CHECK
        kwargs = kwargs | {"blockm": BLOCKM}
        res = M.krylov(*args, **kwargs)
        if CHECK:
            ls = sorted(res)
            for lidx, l in enumerate(ls):
                seqs = [list(map(int, s)) for s in res[l]]
                if lidx < 5 or lidx > len(ls) - 5:
                    pol = lingen.generating_polynomial_multi(seqs, M.dim, l)
                    det = -pol[0] if M.dim & 1 == 1 else pol[0]
                    logging.info(f"Check Wiedemann modulo {l} OK: det(M % {l}) = {det % l}")

    Mat = SpMV(dense, plus, minus, basis, weight)
    logging.info("Running with 1 modulus (naive kernel)")
    Mat.wiedemann(moduli[0], check=CHECK, bench=BENCH)
    logging.info("Running with 16 moduli (naive kernel)")
    wiedemann_multi(Mat, moduli[:16], bench=BENCH)
    logging.info("Running with 32 moduli (naive kernel)")
    wiedemann_multi(Mat, moduli[:32], bench=BENCH)
    logging.info("Running with 64 moduli (naive kernel)")
    wiedemann_multi(Mat, moduli[:64], bench=BENCH)
    logging.info("Running with 128 moduli (naive kernel)")
    wiedemann_multi(Mat, moduli[:128], bench=BENCH)

    logging.info("Running with 32 moduli64 (naive kernel)")
    wiedemann_multi(Mat, moduli64[:32], bench=BENCH)
    logging.info("Running with 64 moduli64 (naive kernel)")
    wiedemann_multi(Mat, moduli64[:64], bench=BENCH)

    if dim < 32768:
        # indices must fit int16
        Mat1 = CSRMatrix(dense, plus, minus, basis, weight)
        if dim < gpu.max_shmem() // 4:
            logging.info("Running with 16 moduli (small)")
            wiedemann_multi(Mat1, moduli[:16], bench=BENCH)
            logging.info("Running with 64 moduli (small)")
            wiedemann_multi(Mat1, moduli[:64], bench=BENCH)
            logging.info("Running with 128 moduli (small)")
            wiedemann_multi(Mat1, moduli[:128], bench=BENCH)

        # FIXME: broken?
        # logging.info("Running with 16 moduli (medium)")
        # Mat1.wiedemann_medium(moduli[:16], check=False)
        # logging.info("Running with 64 moduli (medium)")
        # Mat1.wiedemann_medium(moduli[:64], check=False)

    if dim < gpu.max_shmem() // 8:
        logging.info("Running with 16 moduli64 (small)")
        wiedemann_multi(Mat1, moduli64[:16], bench=BENCH)
        logging.info("Running with 64 moduli64 (small)")
        wiedemann_multi(Mat1, moduli64[:64], bench=BENCH)
        logging.info("Running with 128 moduli64 (small)")
        wiedemann_multi(Mat1, moduli64[:128], bench=BENCH)

    Mat2 = BlockCOO(dense, plus, minus, basis, weight)
    logging.info("Running BlockCOO v3 with 1 moduli")
    wiedemann_multi(Mat2, moduli[:1], bench=BENCH)
    logging.info("Running BlockCOO v3 with 8 moduli")
    wiedemann_multi(Mat2, moduli[:8], bench=BENCH)
    logging.info("Running BlockCOO v3 with 16 moduli")
    wiedemann_multi(Mat2, moduli[:16], bench=BENCH)
    logging.info("Running BlockCOO v3 with 32 moduli")
    wiedemann_multi(Mat2, moduli[:32], bench=BENCH)

    logging.info("Running BlockCOO v3 with 1 moduli64")
    wiedemann_multi(Mat2, moduli64[:1], bench=BENCH)
    logging.info("Running BlockCOO v3 with 8 moduli64")
    wiedemann_multi(Mat2, moduli64[:8], bench=BENCH)
    logging.info("Running BlockCOO v3 with 16 moduli64")
    wiedemann_multi(Mat2, moduli64[:16], bench=BENCH)


if __name__ == "__main__":
    main()
