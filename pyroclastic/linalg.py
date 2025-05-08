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
from . import relations
import pyroclastic_flint_extras as flint_extras


def to_sparse_matrix(rels):
    stats = {}
    for r in rels:
        for p in r:
            if r[p]:
                stats[p] = stats.get(p, 0) + 1
    dense_p = []
    for p, count in stats.items():
        if count > len(rels) // 3:
            dense_p.append(p)
    dense_p.sort()
    dense_p = dense_p[len(dense_p) % 4 :]
    assert len(dense_p) % 4 == 0

    dense_weight = sum(stats[p] for p in dense_p) / float(len(rels))
    logging.info(f"Dense columns for {len(dense_p)} primes {dense_p}")
    logging.info(f"Dense block has average weight {dense_weight:.1f} per row")
    sparse_weight = sum(
        sum(abs(e) for p, e in r.items() if p not in dense_p) for r in rels
    ) / float(len(rels))
    logging.info(f"Sparse block has avg weight {sparse_weight:.1f} per row")

    for r in rels:
        for p in dense_p:
            if p in r:
                assert abs(r[p]) < 127

    dense = np.zeros((len(rels), len(dense_p)), dtype=np.int8)
    for i, r in enumerate(rels):
        dense[i, :] = [r.get(p, 0) for p in dense_p]
    dense_norm = max(np.sum(np.abs(dense[i, :])) for i in range(len(rels)))
    logging.info(f"Dense block has max row norm {dense_norm}")

    dense_set = frozenset(dense_p)
    primes = sorted(dense_p) + sorted(p for p in stats if p not in dense_set)
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
        plus.append(row_p)
        minus.append(row_m)
    return primes, dense, plus, minus


class SpMV:
    def __init__(self, dense, plus, minus, basis):
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
        logging.debug(f"{self.flops} FLOPS per matrix multiplication")

    def wiedemann(self, l: int, check=False):
        "Perform Wiedemann algorithm for a single small modulus"
        WGSIZE = 128
        mgr = self.mgr
        dim = self.dim
        xin = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xout = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
        xmod = mgr.tensor_t(np.array([l], dtype=np.uint32))

        xd, xplus, xminus, xidxp, xidxm = self.tensors
        kernel = gpu.compile("spmv.comp", self.defines)
        algo = mgr.algorithm(
            [xd, xplus, xminus, xidxp, xidxm, xin, xout, xmod],
            kernel,
            workgroup=((dim + WGSIZE - 1) // WGSIZE, 1, 1),
        )
        (
            mgr.sequence()
            .record(kp.OpTensorSyncDevice([xd, xplus, xminus, xidxp, xidxm, xmod]))
            .eval()
        )

        v = np.random.randint(0, l, dim, dtype=np.int32)
        w = np.random.randint(0, l, dim, dtype=np.int32)
        xin.data()[:] = v
        sequence = []

        t0 = time.monotonic()
        for _ in range(2 * dim + 1):
            (
                mgr.sequence()
                .record(kp.OpTensorSyncDevice([xin]))
                .record(kp.OpAlgoDispatch(algo))
                .record(kp.OpTensorSyncLocal([xout]))
                .eval()
            )
            mv = xout.data()
            xin.data()[:] = mv
            s = int(np.dot(mv.astype(np.int64), w.astype(np.int64)))
            sequence.append(s % l)
        dt = time.monotonic() - t0
        flops = self.flops * (2 * dim + 1) / dt
        logging.info(f"Wiedemann completed in {dt:.3}s ({flops/1e9:.2} GFLOPS)")

        poly = flint_extras.berlekamp_massey(sequence, l)
        if check:
            # Check polynomial
            poly_v = np.array(poly, dtype=np.uint64)
            conv = np.convolve(
                np.array(sequence, dtype=np.uint64),
                poly_v[::-1],
                mode="valid",
            )
            assert np.all(conv % l == 0)
        return poly


def main():
    import argparse
    import os

    p = argparse.ArgumentParser()
    p.add_argument("DATADIR")
    args = p.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    rels = []
    with open(os.path.join(args.DATADIR, "relations.sieve")) as f:
        for l in f:
            rels.append([int(x) for x in l.split()])

    logging.info(f"Imported {len(rels)} relations")
    pruned, _ = relations.prune2(rels, 0, 0)

    # for r in pruned:
    #    print(r)

    filtered = relations.step_filter(pruned, pathlib.Path(args.DATADIR))
    print("Truncating to square matrix")
    rels = filtered
    basis1 = sorted(set(p for r in rels for p, e in r.items() if e))
    dim = len(basis1)
    while True:
        subres = random.sample(rels, dim)
        if len(basis1) == len(set(p for r in subres for p, e in r.items() if e)):
            rels = subres
            break

    basis, dense, plus, minus = to_sparse_matrix(rels)
    assert sorted(basis) == basis1
    print("Dense block")
    print(dense)
    print("Rows +1")
    print(plus[:10], "...")
    print("Rows -1")
    print(minus[:10], "...")

    Mat = SpMV(dense, plus, minus, basis)
    Mat.wiedemann(65537)


if __name__ == "__main__":
    main()
