"""
Linear algebra step

The input of this step is a sparse matrix M (rows > cols).
Computation kernels will assume than nrows < 65536

The memory representation of M (or submatrices of M) will be:
- a dense block of 30-50 columns with high density
- a CSR for coefficients +1 (an index may appear multiple times for larger values)
- a CSR for coefficients -1

The number of columns is usually ~half the number of primes used for sieving.

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
import logging
import random
import time

import kp
import numpy as np

from . import algebra
from . import gpu
from . import linalg
import pyroclastic_flint_extras as flint_extras


class SpMV:
    """
    A CSR encoded matrix with:
    - a block of dense columns (int8 coefficients)
    - an array of sparse positive rows (int16 columns indices with +1 coefficient)
    - an array of sparse negative rows (int16 columns indices with -1 coefficient)

    To support larger matrices, an index 0xffff can be inserted in sparse rows
    to explain that following indices belong to another block of size 0xffff
    """

    def __init__(self, dense, plus, minus, basis, weight, gpu_idx=0):
        dim, dense_n = dense.shape
        assert dim < 65536 * dense_n
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(gpu_idx)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        # Encode rows
        def encode_row(row):
            "Encode row when dimension is large"
            nonlocal dense_n, dim
            if dim <= 0xFFFF:
                return row
            enc = []
            base = 0
            for x in row:
                while x >= base + 0xFFFF:
                    enc.append(0xFFFF)
                    base += 0xFFFF
                assert 0 <= x - base < 0xFFFF
                enc.append(x - base)
            return enc

        enc_plus = [encode_row(l) for l in plus]
        enc_minus = [encode_row(l) for l in minus]

        rowlen_plus = [len(l) for l in enc_plus]
        rowlen_minus = [len(l) for l in enc_minus]
        aidx_plus = np.cumsum(
            np.array([0] + rowlen_plus, dtype=np.uint32), dtype=np.uint32
        )
        aidx_minus = np.cumsum(
            np.array([0] + rowlen_minus, dtype=np.uint32), dtype=np.uint32
        )
        size_plus = int(aidx_plus[-1])
        size_minus = int(aidx_minus[-1])
        aplus = np.zeros(size_plus + (size_plus & 1), dtype=np.uint16)
        aminus = np.zeros(size_minus + (size_minus & 1), dtype=np.uint16)
        for i, l in enumerate(enc_plus):
            aplus[aidx_plus[i] : aidx_plus[i + 1]] = l
        for i, l in enumerate(enc_minus):
            aminus[aidx_minus[i] : aidx_minus[i + 1]] = l
        # Kompute wants uint32, cast arrays to make it happy
        xplus = mgr.tensor_t(aplus.view(np.uint32))
        xminus = mgr.tensor_t(aminus.view(np.uint32))
        xidxp = mgr.tensor_t(aidx_plus)
        xidxm = mgr.tensor_t(aidx_minus)

        self.mgr = mgr
        self.tensors = [xd, xplus, xminus, xidxp, xidxm]
        bitsize = 32 * sum(t.size() for t in self.tensors)
        logging.debug(f"Matrix format using {bitsize / weight:.1f} bits/coefficient")
        self.flops = 2 * dim * dense_n + size_plus + size_minus
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def wiedemann(self, l: int, check=False, bench=False):
        "Perform Wiedemann algorithm for a single small modulus"
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        if dim < 1000:
            BATCHSIZE = 64
        elif dim < 10000:
            BATCHSIZE = 128
        else:
            BATCHSIZE = 32

        if l.bit_length() >= 24:
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32

        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2, dtype=word_t).view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(dim // WGSIZE + 1, dtype=np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_WG = (dim + WGSIZE - 1) // WGSIZE
        sel = np.zeros(N_WG // 8 * 8 + 16, dtype=np.uint8)
        if N_WG == 1:
            sel[0] = random.randrange(dim)
        elif N_WG < 10:
            # always zero on last workgroup
            for i in range(N_WG - 1):
                sel[i] = random.randrange(WGSIZE)
        else:
            # very sparse weights to avoid overflow
            assert WGSIZE < 255
            sel.fill(255)
            for i in random.sample(list(range(N_WG - 1)), 8):
                sel[i] = random.randrange(WGSIZE)
        xsel = mgr.tensor_t(sel.view(np.uint32))
        # Output sequence out[k] = S M^k V
        ITERS = (2 * dim // BATCHSIZE + 2) * BATCHSIZE
        if bench:
            ITERS = 1024
        xout = mgr.tensor_t(np.zeros(ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array([l], dtype=word_t).view(np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        defines = self.defines | {}
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv.comp", defines)
        algo = mgr.algorithm(
            [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(N_WG, 1, 1),
        )
        (
            mgr.sequence()
            .record(
                kp.OpTensorSyncDevice(
                    [xd, xplus, xminus, xidxp, xidxm, xsel, xout, xmod]
                )
            )
            .eval()
        )

        v = np.random.randint(0, l, dim, dtype=word_t)
        xv.data().view(word_t)[:dim] = v
        seq0 = sum(v[i * WGSIZE + sel[i]] for i in range(N_WG) if sel[i] < WGSIZE)

        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
        vec_size = 4 * xv.size()
        logging.debug(
            f"Buffer sizes: matrix {mat_size >> 10}kB vectors {vec_size >> 10}kB"
        )

        t0 = time.monotonic()
        gpu_ticks = 0.0
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
        vout = xout.data().view(np.uint64)
        sequence = [seq0] + [int(x) % l for x in vout]

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS / gpu_dt
        speed = ITERS / gpu_dt

        poly = flint_extras.berlekamp_massey(sequence, l)
        assert len(poly) <= dim + 1, len(poly)
        if check and not bench:
            assert linalg.check_wiedemann(sequence, poly, l)
            assert len(poly) == dim + 1, len(poly)
            det = -poly[0] * pow(poly[dim], -1, l) % l
            logging.info(f"Check Wiedemann modulo {l} OK: det(M % {l}) = {det}")

        dt = time.monotonic() - t0
        logging.info(
            f"Wiedemann completed in {dt:.3f}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )

        return poly

    def wiedemann_multi(self, ls: list[int], check=False, lock=None, bench=False):
        "Perform Wiedemann algorithm for multiple small moduli"
        MODULI = len(ls)
        assert MODULI in (1, 2, 4, 8, 16, 32, 64, 128)
        BATCH_ROW = min(128, 512 // MODULI)

        if any(l.bit_length() > 32 for l in ls):
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32

        mgr = self.mgr
        dim = self.dim
        assert dim >= 256
        if dim < 10000:
            BATCHSIZE = 64
        else:
            BATCHSIZE = 16
        ITERS = 2 * dim
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE
        if bench:
            ITERS = 1024
        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2 * MODULI, dtype=word_t).view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(dim // BATCH_ROW + 1, dtype=np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_WG = (dim + BATCH_ROW - 1) // BATCH_ROW
        sel = np.zeros(N_WG // 8 * 8 + 16, dtype=np.uint8)
        if N_WG == 1:
            sel[0] = random.randrange(dim)
        elif N_WG < 10:
            # always zero on last workgroup
            for i in range(N_WG - 1):
                sel[i] = random.randrange(BATCH_ROW)
        else:
            # very sparse weights to avoid overflow
            assert BATCH_ROW < 255
            sel.fill(255)
            for i in random.sample(list(range(N_WG - 1)), 8):
                sel[i] = random.randrange(BATCH_ROW)
        xsel = mgr.tensor_t(sel.view(np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(MODULI * ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array(ls, dtype=word_t).view(np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        defines = self.defines | {"MODULI": MODULI, "BATCH_ROW": BATCH_ROW}
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv_multi.comp", defines)
        algo = mgr.algorithm(
            [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(1, N_WG, 1),
        )
        (
            mgr.sequence()
            .record(
                kp.OpTensorSyncDevice(
                    [xd, xplus, xminus, xidxp, xidxm, xsel, xout, xmod]
                )
            )
            .eval()
        )

        v = xv.data().view(word_t).reshape((2, dim, MODULI))
        for i, l in enumerate(ls):
            v[0, :, i] = np.random.randint(0, l, dim, dtype=word_t)
        # Random (sparse) set of weights
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
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

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / gpu_dt
        speed = ITERS * MODULI / gpu_dt

        dets = []
        for i, li in enumerate(ls):
            if bench:
                break
            sequence = [int(x) % li for x in vout[:, i]]

            poly = flint_extras.berlekamp_massey(sequence, li)
            assert len(poly) == dim + 1, f"l={li} deg={len(poly) - 1}"
            det = -poly[0] * pow(poly[dim], -1, li) % li
            dets.append(det)
            if check:
                assert linalg.check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1
                det = -poly[0] * pow(poly[dim], -1, li) % li
                if i < 5 or i > len(ls) - 5:
                    logging.info(
                        f"Check Wiedemann modulo {li} OK: det(M % {li}) = {det}"
                    )

        dt = time.monotonic() - t0
        logging.info(
            f"Wiedemann completed in {dt:.3f}s (GPU {gpu_dt:.3f}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return dets, ls

    def wiedemann_big(self, l: int, check=False):
        "Perform Wiedemann algorithm for a single big modulus"
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + WGSIZE - 1) // WGSIZE

        if dim < 1000:
            BATCHSIZE = 32
        else:
            BATCHSIZE = 64
        ITERS = (2 * dim // BATCHSIZE + 2) * BATCHSIZE

        # FIXME: use actual norm
        BLEN = (l.bit_length() + 8 + 31) // 32
        pwords = to_uvec(l, BLEN)
        assert pwords[-2] > 2**16

        defines = self.defines | {"BLEN": BLEN}
        kernel = gpu.compile("spmv_bigint.comp", defines)

        # Tensor holding M^k V and M^(k+1) V
        v = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            v[0, i, :] = to_uvec(random.randrange(l), BLEN)
        xv = mgr.tensor_t(v.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(ITERS * BLEN, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array(pwords, dtype=np.uint32))

        tensors = self.tensors + [xv, xiter, xmod, xout]
        algo = mgr.algorithm(tensors, kernel, workgroup=(N_WG, 1, 1))
        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()

        seq0 = from_uvec(v[0, 0, :])
        t0 = time.monotonic()
        gpu_ticks = 0.0
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
        vout = xout.data().reshape((ITERS, BLEN))
        sequence = [seq0] + [from_uvec(vout[i, :]) for i in range(ITERS)]

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS / gpu_dt
        speed = ITERS / gpu_dt

        poly = algebra.berlekamp_massey(sequence, l)
        assert len(poly) <= dim + 1, len(poly)
        if check:
            assert linalg.check_wiedemann(sequence, poly, l)
            assert len(poly) == dim + 1, len(poly)
            det = -poly[0] * pow(poly[dim], -1, l) % l
            logging.info(f"Check Wiedemann modulo {l} OK: det(M % {l}) = {det}")

        dt = time.monotonic() - t0
        logging.info(
            f"Wiedemann completed in {dt:.3f}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )

        return poly

    def polyeval(self, v, l: int, poly: list[int]) -> list[int]:
        """
        Compute poly(M)*v mod l
        """
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        N_WG = (dim + WGSIZE - 1) // WGSIZE
        BATCHSIZE = 16

        # FIXME: use actual norm
        BLEN = (l.bit_length() + 8 + 31) // 32
        maxout = l * l * len(poly)
        ALEN = (maxout.bit_length() + 4 + 31) // 32
        pwords = to_uvec(l, BLEN)
        assert pwords[-2] > 2**16

        defines = self.defines | {"ALEN": ALEN, "BLEN": BLEN, "POLYEVAL": 1}
        kernel = gpu.compile("spmv_bigint.comp", defines)

        # Tensor holding M^k V and M^(k+1) V
        av = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            av[0, i, :] = to_uvec(v[i], BLEN)
        xv = mgr.tensor_t(av.flatten())
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array(pwords, dtype=np.uint32))
        vpoly = np.zeros((len(poly), BLEN), dtype=np.uint32)
        for k, ak in enumerate(poly):
            vpoly[k, :] = to_uvec(ak, BLEN)
        xpoly = mgr.tensor_t(vpoly.flatten())
        # Output sequence out[k] = S M^k V, initialize with a0 * v
        vout = np.zeros((dim, ALEN), dtype=np.uint32)
        for i, vi in enumerate(v):
            vout[i, :] = to_uvec(poly[0] * vi, ALEN)
        xout = mgr.tensor_t(vout.flatten())

        tensors = self.tensors + [xv, xiter, xmod, xpoly, xout]
        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()
        algo = mgr.algorithm(tensors, kernel, workgroup=(N_WG, 1, 1))

        t0 = time.monotonic()
        gpu_ticks = 0.0
        count = 0
        for i in range(1, len(poly), BATCHSIZE):
            # Matrix multiplication is very fast so we launch multiple
            # iterations per batch.
            seq = mgr.sequence(total_timestamps=2 * BATCHSIZE)
            for _ in range(min(BATCHSIZE, len(poly) - i)):
                count += 1
                seq.record(kp.OpAlgoDispatch(algo))
            seq.eval()

            stamps = seq.get_timestamps()
            gpu_ticks += stamps[-1] - stamps[0]
        assert count == len(poly) - 1

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * len(poly) / gpu_dt
        speed = len(poly) / gpu_dt

        mgr.sequence().record(kp.OpTensorSyncLocal([xout])).eval()
        vout = xout.data().reshape((dim, ALEN))
        dt = time.monotonic() - t0
        logging.info(
            f"Polyeval completed in {dt:.3f}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return [from_uvec(vout[i, :]) % l for i in range(dim)]

    def mulvec(self, v, l: int):
        result = None

        def func(i, x):
            nonlocal result
            if i == 1:
                result = np.copy(x)

        self.mulvec_iter(v, l, 2, func)
        return result

    def mulvec_iter(self, v, l: int, maxpow: int, callback):
        """
        Iterate over powers M^k V for k in range(maxpow)
        and call callback(k, M^k V) on each vector. The callback receives
        a borrowed reference to the vector and should copy
        data if required.
        """
        mgr = self.mgr
        dim = self.dim
        WGSIZE = 128

        if l.bit_length() >= 24:
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32

        defines = self.defines
        if INT64:
            defines |= {"INT64": 1}
        kernel = gpu.compile("spmv.comp", defines)

        xv = mgr.tensor_t(np.zeros(2 * dim, dtype=word_t).view(np.uint32))
        xv.data().view(word_t)[:dim] = v

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        xiter = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xout = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array([l], dtype=word_t).view(np.uint32))

        tensors = [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xout, xout, xmod]
        algo = mgr.algorithm(
            tensors,
            kernel,
            workgroup=((dim + WGSIZE - 1) // WGSIZE, 1, 1),
        )
        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()
        callback(0, v)
        for i in range(1, maxpow):
            (
                mgr.sequence()
                .record(kp.OpAlgoDispatch(algo))
                .record(kp.OpTensorSyncLocal([xv]))
                .eval()
            )
            vv = xv.data().view(word_t)
            if i % 2 == 1:
                callback(i, vv[dim:])
            else:
                callback(i, vv[:dim])

    def mulvec_big(self, v, l: int) -> list[int]:
        result = None

        def func(i, x):
            nonlocal result
            if i == 1:
                result = x.copy()

        self.mulvec_big_iter(v, l, 2, func)
        return result

    def mulvec_big_iter(self, v: list[int], l: int, maxpow: int, callback):
        mgr = self.mgr
        dim = self.dim
        WGSIZE = 128
        N_WG = (dim + WGSIZE - 1) // WGSIZE
        # FIXME: use actual norm
        BLEN = (l.bit_length() + 8 + 31) // 32
        pwords = to_uvec(l, BLEN)
        assert pwords[-2] > 2**16

        defines = self.defines | {"BLEN": BLEN}
        kernel = gpu.compile("spmv_bigint.comp", defines)

        vin = np.zeros((2, dim, BLEN), dtype=np.uint32)
        for i in range(dim):
            vin[0, i, :] = to_uvec(v[i], BLEN)
        xv = mgr.tensor_t(vin.flatten())

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        xout = mgr.tensor_t(np.zeros(maxpow * BLEN, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array(pwords, dtype=np.uint32))

        tensors = [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xout, xmod]
        algo = mgr.algorithm(
            tensors,
            kernel,
            workgroup=((dim + WGSIZE - 1) // WGSIZE, 1, 1),
        )
        mgr.sequence().record(kp.OpTensorSyncDevice(tensors)).eval()

        callback(0, v)
        for i in range(0, maxpow, 2):
            (
                mgr.sequence()
                .record(kp.OpAlgoDispatch(algo))
                .record(kp.OpAlgoDispatch(algo))
                .record(kp.OpTensorSyncLocal([xv]))
                .eval()
            )
            vv = xv.data().reshape((2, dim, BLEN))
            if i + 1 < maxpow:
                callback(i + 1, [from_uvec(vv[1, j, :]) for j in range(dim)])
            if i + 2 < maxpow:
                callback(i + 2, [from_uvec(vv[0, j, :]) for j in range(dim)])


def to_uvec(x: int, length: int):
    assert x.bit_length() <= 32 * length
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(length)]


def from_uvec(words: list) -> int:
    return sum(int(x) << (32 * i) for i, x in enumerate(words))


class BlockCOO:
    def __init__(self, dense, plus, minus, basis, weight, BM=None):
        dim, dense_n = dense.shape
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(0)
        # Kompute wants uint32, cast arrays to make it happy
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        aplus, aminus = [], []
        idx_plus, idx_minus = [], []
        # Enough to fit 8x u64 or 16x u32 moduli
        if BM is None:
            if dim < 10000:
                BM = 1024
            else:
                BM = 1024
        self.BM = BM
        blk_plus, blk_minus = [], []
        assert len(plus) == dim
        assert len(minus) == dim
        for i in range(dim):
            if i % BM == 0:
                aplus.extend(sorted(blk_plus))
                aminus.extend(sorted(blk_minus))
                idx_plus.append(len(aplus))
                idx_minus.append(len(aminus))
                blk_plus, blk_minus = [], []
            blk_plus.extend((i % BM) + BM * j for j in plus[i])
            blk_minus.extend((i % BM) + BM * j for j in minus[i])
        if dim % BM != 1:
            aplus.extend(sorted(blk_plus))
            aminus.extend(sorted(blk_minus))
            idx_plus.append(len(aplus))
            idx_minus.append(len(aminus))
        assert len(aplus) == sum(len(l) for l in plus)
        assert len(aminus) == sum(len(l) for l in minus)
        assert len(idx_plus) == 1 + (dim + BM - 1) // BM
        assert len(idx_minus) == 1 + (dim + BM - 1) // BM
        print("Block sizes")
        print([j - i for i, j in zip(idx_plus, idx_plus[1:])])
        print([j - i for i, j in zip(idx_minus, idx_minus[1:])])
        print("Deltas")
        print("plus ", max(j - i for i, j in zip(aplus, aplus[16:])))
        print("minus", max(j - i for i, j in zip(aminus, aminus[16:])))

        xplus = mgr.tensor_t(np.array(aplus, dtype=np.uint32))
        xminus = mgr.tensor_t(np.array(aminus, dtype=np.uint32))
        xidxp = mgr.tensor_t(np.array(idx_plus, dtype=np.uint32))
        xidxm = mgr.tensor_t(np.array(idx_minus, dtype=np.uint32))

        self.mgr = mgr
        self.tensors = [xd, xplus, xminus, xidxp, xidxm]
        self.flops = 2 * dim * dense_n + len(idx_plus) + len(idx_minus)
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def wiedemann_multi(self, ls: list[int], check=False, bench=False):
        "Perform Wiedemann algorithm for multiple small moduli"
        BM = self.BM
        MODULI = len(ls)
        assert MODULI in (1, 2, 4, 8, 16)
        if ls[0].bit_length() > 32:
            INT64 = True
            word_t = np.uint64
            assert self.BM * len(ls) * 8 <= 65536
        else:
            INT64 = False
            word_t = np.uint32
            assert self.BM * len(ls) * 4 <= 65536

        mgr = self.mgr
        dim = self.dim
        assert dim >= 256

        if dim < 10000:
            BATCHSIZE = 64
        else:
            BATCHSIZE = 16
        ITERS = (2 * dim // BATCHSIZE + 2) * BATCHSIZE
        if bench:
            ITERS = 1024
        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2 * MODULI, dtype=word_t).view(np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_WG = (dim + BM - 1) // BM
        sel = np.zeros(N_WG // 8 * 8 + 16, dtype=np.uint8)
        for i in range(N_WG - 1):
            # always zero on last workgroup
            sel[i] = random.randrange(min(256, BM))
        xsel = mgr.tensor_t(sel.view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(MODULI * ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array(ls, dtype=word_t).view(np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        defines = self.defines | {"BM": BM, "MODULI": MODULI}
        if INT64:
            defines["INT64"] = 1
        kernel = gpu.compile("spmv_blockcoo.comp", defines)
        algo = mgr.algorithm(
            [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(N_WG, 1, 1),
        )
        (
            mgr.sequence()
            .record(
                kp.OpTensorSyncDevice(
                    [xd, xplus, xminus, xidxp, xidxm, xsel, xout, xmod]
                )
            )
            .eval()
        )

        v = xv.data().view(word_t).reshape((2, dim, MODULI))
        for i, l in enumerate(ls):
            v[0, :, i] = np.random.randint(0, l, dim, dtype=word_t)
        # Random (sparse) set of weights
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
        vec_size = 4 * xv.size()
        logging.debug(
            f"Buffer sizes: matrix {mat_size >> 10}kB vectors {vec_size >> 10}kB"
        )

        t0 = time.monotonic()
        gpu_ticks = 0.0
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

        polys = []
        for i, li in enumerate(ls):
            if bench:
                break
            sequence = [int(x) % li for x in vout[:, i]]
            poly = flint_extras.berlekamp_massey(sequence, li)
            polys.append(poly)
            if check:
                assert linalg.check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1
                det = -poly[0] * pow(poly[dim], -1, li) % li
                logging.info(f"Check Wiedemann modulo {li} OK: det(M % {li}) = {det}")

        dt = time.monotonic() - t0
        logging.info(
            f"Wiedemann completed in {dt:.3}s (GPU {gpu_dt:.3}s, {flops / 1e9:.2f} GFLOPS, {speed:.1f} SpMV/s)"
        )
        return polys

    def mulvec(self, v, p: int):
        defines = self.defines | {"BM": 32, "MODULI": 1}

        SHADER = gpu.compile("spmv_blockcoo.comp", defines)

        BM = self.BM
        mgr = self.mgr
        dim = self.dim

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        xv = mgr.tensor_t(np.zeros(2 * dim, dtype=np.uint32))
        xv.data()[:dim] = v
        xiter = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xout = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array([p], dtype=np.uint32))

        algo = mgr.algorithm(
            [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xout, xout, xmod],
            SHADER,
            workgroup=((dim + BM - 1) // BM, 1, 1),
        )
        (
            mgr.sequence()
            .record(
                kp.OpTensorSyncDevice(
                    [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xout, xmod]
                )
            )
            .record(kp.OpAlgoDispatch(algo))
            .record(kp.OpTensorSyncLocal([xv]))
            .eval()
        )

        return np.copy(xv.data()[dim:])


class BlockCOOv2:
    def __init__(self, dense, plus, minus, basis, weight):
        dim, dense_n = dense.shape
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(0)
        # Kompute wants uint32, cast arrays to make it happy
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        aplus, aminus = [], []
        idx_plus, idx_minus = [], []
        if dim < 10000:
            BM = 512
        else:
            BM = 1024
        self.BM = BM
        blk_plus, blk_minus = [], []
        assert len(plus) == dim
        assert len(minus) == dim
        for i in range(dim):
            if i % BM == 0:
                aplus.extend(sorted(blk_plus))
                aminus.extend(sorted(blk_minus))
                idx_plus.append(len(aplus))
                idx_minus.append(len(aminus))
                blk_plus, blk_minus = [], []
            blk_plus.extend((i % BM) + BM * j for j in plus[i])
            blk_minus.extend((i % BM) + BM * j for j in minus[i])
        if dim % BM != 0:
            aplus.extend(sorted(blk_plus))
            aminus.extend(sorted(blk_minus))
            idx_plus.append(len(aplus))
            idx_minus.append(len(aminus))
        assert len(aplus) == sum(len(l) for l in plus)
        assert len(aminus) == sum(len(l) for l in minus)
        assert len(idx_plus) == 1 + (dim + BM - 1) // BM
        assert len(idx_minus) == 1 + (dim + BM - 1) // BM
        print("Block sizes")
        print([j - i for i, j in zip(idx_plus, idx_plus[1:])])
        print([j - i for i, j in zip(idx_minus, idx_minus[1:])])
        print("Deltas")
        print("plus ", max(j - i for i, j in zip(aplus, aplus[16:])))
        print("minus", max(j - i for i, j in zip(aminus, aminus[16:])))

        xplus = mgr.tensor_t(np.array(aplus, dtype=np.uint32))
        xminus = mgr.tensor_t(np.array(aminus, dtype=np.uint32))
        xidxp = mgr.tensor_t(np.array(idx_plus, dtype=np.uint32))
        xidxm = mgr.tensor_t(np.array(idx_minus, dtype=np.uint32))

        self.mgr = mgr
        self.tensors = [xd, xplus, xminus, xidxp, xidxm]
        self.flops = 2 * dim * dense_n + len(idx_plus) + len(idx_minus)
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def wiedemann_multi(self, ls: list[int], check=False):
        """
        Perform Wiedemann algorithm for multiple small moduli

        Variant with 1 workgroup per modulus.
        """
        BM = self.BM
        MODULI = len(ls)
        if ls[0].bit_length() > 32:
            INT64 = True
            word_t = np.uint64
        else:
            INT64 = False
            word_t = np.uint32

        mgr = self.mgr
        dim = self.dim
        assert dim >= 256

        if dim < 10000:
            BATCHSIZE = 64
        else:
            BATCHSIZE = 16
        ITERS = (2 * dim // BATCHSIZE + 2) * BATCHSIZE

        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2 * MODULI, dtype=word_t).view(np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_STRIPES = (dim + BM - 1) // BM
        N_WG = MODULI
        sel = np.zeros(N_STRIPES // 8 * 8 + 16, dtype=np.uint8)
        for i in range(dim // BM):
            # always zero on last workgroup
            sel[i] = random.randrange(min(256, BM))
        xsel = mgr.tensor_t(sel.view(np.uint32))
        xiter = mgr.tensor_t(np.zeros(N_WG, dtype=np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(MODULI * ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array(ls, dtype=word_t).view(np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        defines = self.defines | {"BM": BM, "MODULI": MODULI, "ITERS": ITERS}
        if INT64:
            defines["INT64"] = 1
        kernel = gpu.compile("spmv_blockcoo2.comp", defines)
        algo = mgr.algorithm(
            [xd, xplus, xminus, xidxp, xidxm, xv, xiter, xsel, xout, xmod],
            kernel,
            workgroup=(N_WG, 1, 1),
        )
        (
            mgr.sequence()
            .record(
                kp.OpTensorSyncDevice(
                    [xd, xplus, xminus, xidxp, xidxm, xsel, xout, xmod]
                )
            )
            .eval()
        )

        v = xv.data().view(word_t).reshape((MODULI, 2, dim))
        for i, l in enumerate(ls):
            v[i, 0, :] = np.random.randint(0, l, dim, dtype=word_t)
        # Random (sparse) set of weights
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xiter, xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
        vec_size = 4 * xv.size()
        logging.debug(
            f"Buffer sizes: matrix {mat_size >> 10}kB vectors {vec_size >> 10}kB"
        )

        t0 = time.monotonic()
        gpu_ticks = 0.0
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
        vout = xout.data().view(np.uint64).reshape((MODULI, ITERS))

        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / gpu_dt
        speed = ITERS * MODULI / gpu_dt

        dets = []
        for i, li in enumerate(ls):
            sequence = [int(x) % li for x in vout[i, :]]
            poly = flint_extras.berlekamp_massey(sequence, li)
            assert len(poly) == dim + 1
            det = -poly[0] * pow(poly[dim], -1, li) % li
            dets.append(det)
            if check:
                assert linalg.check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1
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
