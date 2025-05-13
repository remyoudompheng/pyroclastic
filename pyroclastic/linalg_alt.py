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

import logging
import pathlib
import random
import time

import flint
import kp
import numpy as np

from . import gpu
from . import linalg
from . import relations
import pyroclastic_flint_extras as flint_extras


class SpMV:
    def __init__(self, dense, plus, minus, basis, weight):
        dim, dense_n = dense.shape
        assert (dim * dense_n) % 4 == 0
        self.defines = {"N": dim, "DENSE_N": dense_n}

        self.basis = basis
        self.dim = dim

        # Prepare tensors
        mgr = kp.Manager(0)
        xd = mgr.tensor_t(dense.flatten().view(np.uint32))

        rowlen_plus = [len(l) for l in plus]
        rowlen_minus = [len(l) for l in minus]
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
        for i, l in enumerate(plus):
            aplus[aidx_plus[i] : aidx_plus[i + 1]] = l
        for i, l in enumerate(minus):
            aminus[aidx_minus[i] : aidx_minus[i + 1]] = l
        # Kompute wants uint32, cast arrays to make it happy
        xplus = mgr.tensor_t(aplus.view(np.uint32))
        xminus = mgr.tensor_t(aminus.view(np.uint32))
        xidxp = mgr.tensor_t(aidx_plus)
        xidxm = mgr.tensor_t(aidx_minus)

        self.mgr = mgr
        self.tensors = [xd, xplus, xminus, xidxp, xidxm]
        self.flops = 2 * dim * dense_n + size_plus + size_minus
        self.weight = weight
        logging.debug(
            f"{self.flops} FLOPS per matrix multiplication (original weight {weight})"
        )

    def wiedemann(self, l: int, check=False):
        "Perform Wiedemann algorithm for a single small modulus"
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        assert dim >= 256
        if dim < 1000:
            BATCHSIZE = 64
        elif dim < 10000:
            BATCHSIZE = 128
        else:
            BATCHSIZE = 32
        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2, dtype=np.uint32))
        xiter = mgr.tensor_t(np.zeros(dim // WGSIZE + 1, dtype=np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_WG = (dim + WGSIZE - 1) // WGSIZE
        sel = np.zeros(N_WG // 8 * 8 + 16, dtype=np.uint8)
        for i in range(N_WG - 1):
            # always zero on last workgroup
            sel[i] = random.randrange(WGSIZE)
        xsel = mgr.tensor_t(sel.view(np.uint32))
        # Output sequence out[k] = S M^k V
        ITERS = (2 * dim // BATCHSIZE + 2) * BATCHSIZE
        xout = mgr.tensor_t(np.zeros(ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array([l], dtype=np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        kernel = gpu.compile("spmv.comp", self.defines)
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

        v = np.random.randint(0, l, dim, dtype=np.int32)
        xv.data()[:dim] = v
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
        vec_size = 4 * xv.size()
        logging.info(f"Buffer sizes: matrix {mat_size>>10}kB vectors {vec_size>>10}kB")

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
        sequence = [int(x) % l for x in vout]

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS / dt
        flops2 = self.weight * ITERS / dt
        logging.info(
            f"Wiedemann completed in {dt:.3}s (GPU {gpu_dt:.3}s, {flops/1e9:.2f} GFLOPS, {flops2/1e9:.2f} GOPS)"
        )

        poly = flint_extras.berlekamp_massey(sequence, l)
        if check:
            assert linalg.check_wiedemann(sequence, poly, l)
            assert len(poly) == dim + 1, len(poly)
            det = -poly[0] * pow(poly[dim], -1, l) % l
            logging.info(f"Check Wiedemann modulo {l} OK: det(M % {l}) = {det}")
        return poly

    def wiedemann_multi(self, ls: list[int], check=False):
        "Perform Wiedemann algorithm for multiple small moduli"
        BATCH_ROW = 16
        MODULI = len(ls)
        assert MODULI in (8, 16)

        mgr = self.mgr
        dim = self.dim
        assert dim >= 256
        if dim < 10000:
            BATCHSIZE = 64
        else:
            BATCHSIZE = 16
        ITERS = 2 * dim
        ITERS = (ITERS // BATCHSIZE + 2) * BATCHSIZE
        # Tensor holding M^k V and M^(k+1) V
        xv = mgr.tensor_t(np.zeros(dim * 2 * MODULI, dtype=np.uint32))
        xiter = mgr.tensor_t(np.zeros(dim // BATCH_ROW + 1, dtype=np.uint32))
        # Random weights S (idx such that S[idx]=1 in each workgroup)
        N_WG = (dim + BATCH_ROW - 1) // BATCH_ROW
        sel = np.zeros(N_WG // 8 * 8 + 16, dtype=np.uint8)
        for i in range(N_WG - 1):
            # always zero on last workgroup
            sel[i] = random.randrange(BATCH_ROW)
        xsel = mgr.tensor_t(sel.view(np.uint32))
        # Output sequence out[k] = S M^k V
        xout = mgr.tensor_t(np.zeros(MODULI * ITERS, dtype=np.uint64).view(np.uint32))
        xmod = mgr.tensor_t(np.array(ls, dtype=np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        kernel = gpu.compile(
            "spmv_multi.comp", self.defines | {"MODULI": MODULI, "BATCH_ROW": BATCH_ROW}
        )
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

        v = xv.data().reshape((2, dim, MODULI))
        for i, l in enumerate(ls):
            v[0, :, i] = np.random.randint(0, l, dim, dtype=np.int32)
        # Random (sparse) set of weights
        sequence = []
        mgr.sequence().record(kp.OpTensorSyncDevice([xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
        vec_size = 4 * xv.size()
        logging.info(f"Buffer sizes: matrix {mat_size>>10}kB vectors {vec_size>>10}kB")

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

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / dt
        flops2 = self.weight * ITERS * MODULI / dt
        logging.info(
            f"Wiedemann completed in {dt:.3}s (GPU {gpu_dt:.3}s, {flops/1e9:.2f} GFLOPS, {flops2/1e9:.2f} GOPS)"
        )

        polys = []
        for i, li in enumerate(ls):
            sequence = [int(x) % li for x in vout[:, i]]

            poly = flint_extras.berlekamp_massey(sequence, li)
            assert len(poly) <= dim + 1
            polys.append(poly)
            if check:
                assert linalg.check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1
                det = -poly[0] * pow(poly[dim], -1, li) % li
                logging.info(f"Check Wiedemann modulo {li} OK: det(M % {li}) = {det}")

        return polys


class BlockCOO:
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
        "Perform Wiedemann algorithm for multiple small moduli"
        BM = self.BM
        MODULI = len(ls)
        assert MODULI in (1, 2, 4, 8, 16)
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
        mgr.sequence().record(kp.OpTensorSyncDevice([xv])).eval()

        mat_size = 4 * (
            xd.size() + xplus.size() + xminus.size() + xidxp.size() + xidxm.size()
        )
        vec_size = 4 * xv.size()
        logging.info(f"Buffer sizes: matrix {mat_size>>10}kB vectors {vec_size>>10}kB")

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

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / dt
        flops2 = self.weight * ITERS * MODULI / dt
        logging.info(
            f"Wiedemann completed in {dt:.3}s (GPU {gpu_dt:.3}s, {flops/1e9:.2f} GFLOPS, {flops2/1e9:.2f} GOPS)"
        )

        polys = []
        for i, li in enumerate(ls):
            sequence = [int(x) % li for x in vout[:, i]]
            poly = flint_extras.berlekamp_massey(sequence, li)
            polys.append(poly)
            if check:
                assert linalg.check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1
                det = -poly[0] * pow(poly[dim], -1, li) % li
                logging.info(f"Check Wiedemann modulo {li} OK: det(M % {li}) = {det}")

        return polys


def main():
    import argparse
    import os

    p = argparse.ArgumentParser()
    p.add_argument("DATADIR")
    args = p.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    rels = []
    datadir = pathlib.Path(args.DATADIR)
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

        logging.info(f"Imported {len(rels)} relations")
        pruned, _ = relations.prune2(rels, 0, 0)

        # for r in pruned:
        #    print(r)

        filtered = relations.step_filter(pruned, pathlib.Path(args.DATADIR))
        rels = filtered

    basis1 = sorted(set(p for r in rels for p, e in r.items() if e))
    dim = len(basis1)
    print(f"Truncating to square matrix (dim {dim})")
    while True:
        subres = random.sample(rels, dim)
        if len(basis1) == len(set(p for r in subres for p, e in r.items() if e)):
            rels = subres
            break

    weight = sum(len(r) for r in rels)
    basis, dense, plus, minus = linalg.to_sparse_matrix(rels)
    assert sorted(basis) == basis1

    # print("Dense block")
    # print(dense)
    # print("Rows +1")
    # print(plus[:10], "...")
    # print("Rows -1")
    # print(minus[:10], "...")

    CHECK = True

    moduli = [x for x in range(1000_000, 1010000) if flint.fmpz(x).is_prime()]

    logging.info("Running with 1 modulus")
    Mat = SpMV(dense, plus, minus, basis, weight)
    Mat.wiedemann(65537, check=CHECK)

    logging.info("Running with 8 moduli")
    Mat.wiedemann_multi(moduli[:8], check=CHECK)

    logging.info("Running with 16 moduli")
    Mat.wiedemann_multi(moduli[:16], check=CHECK)

    Mat2 = BlockCOO(dense, plus, minus, basis, weight)
    logging.info("Running BlockCOO with 1 modulus")
    Mat2.wiedemann_multi([65537], check=CHECK)
    logging.info("Running BlockCOO with 4 moduli")
    Mat2.wiedemann_multi(moduli[:4], check=CHECK)
    logging.info("Running BlockCOO with 8 moduli")
    Mat2.wiedemann_multi(moduli[:8], check=CHECK)
    logging.info("Running BlockCOO with 16 moduli")
    Mat2.wiedemann_multi(moduli[:16], check=CHECK)

    moduli64 = [10000000000000061] + moduli
    logging.info("Running BlockCOO with 1 moduli (64-bit)")
    Mat2.wiedemann_multi(moduli64[:1], check=CHECK)
    logging.info("Running BlockCOO with 4 moduli (64-bit)")
    Mat2.wiedemann_multi(moduli64[:4], check=CHECK)
    logging.info("Running BlockCOO with 8 moduli (64-bit)")
    Mat2.wiedemann_multi(moduli64[:8], check=CHECK)


if __name__ == "__main__":
    main()
