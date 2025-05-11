"""
Linear algebra step

The input of this step is a sparse matrix M (rows > cols).

The memory representation of M (or submatrices of M) is:
- a dense block of 30-50 columns with high density
- a list of sparse stripes (block of BM rows) stored in COO
  format sorted by column (for linear I/O patterns on vector)
  for coefficients +1 (an index may appear multiple times for larger values)
- a similar list for coefficients -1

The block size is chosen so that the (small) output vector
of each stripe fits in cache/local memory for efficient random read-modify-write.

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
    primes = dense_p + sorted(p for p in stats if p not in dense_set)
    dim = len(primes)
    assert dim == len(rels)
    col_density = np.array([stats[p] / len(rels) for p in primes])
    with np.printoptions(precision=5):
        print("Column densities (dense)")
        print(col_density[: len(dense_p)])
        print("Column densities (sparse)")
        print(col_density[len(dense_p) : dim // 3], "...", col_density[2 * dim // 3 :])

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


def check_wiedemann(sequence, poly, p):
    # Check polynomial
    if p.bit_length() < 32:
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
        vout = xout.data().view(np.uint64).reshape((MODULI, ITERS))

        dt = time.monotonic() - t0
        gpu_dt = gpu_ticks * gpu.stamp_period() * 1e-9
        flops = self.flops * ITERS * MODULI / dt
        flops2 = self.weight * ITERS * MODULI / dt
        logging.info(
            f"Wiedemann completed in {dt:.3}s (GPU {gpu_dt:.3}s, {flops/1e9:.2f} GFLOPS, {flops2/1e9:.2f} GOPS)"
        )

        polys = []
        for i, li in enumerate(ls):
            sequence = [int(x) % li for x in vout[i, :]]
            poly = flint_extras.berlekamp_massey(sequence, li)
            polys.append(poly)
            if check:
                assert check_wiedemann(sequence, poly, li)
                assert len(poly) == dim + 1
                det = -poly[0] * pow(poly[dim], -1, li) % li
                if i < 5 or i > len(ls) - 5:
                    logging.info(f"Check Wiedemann modulo {li} OK: det(M % {li}) = {det}")

        return polys


def main():
    import argparse
    import os

    from .linalg_alt import SpMV

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
    basis, dense, plus, minus = to_sparse_matrix(rels)
    assert sorted(basis) == basis1

    # print("Dense block")
    # print(dense)
    # print("Rows +1")
    # print(plus[:10], "...")
    # print("Rows -1")
    # print(minus[:10], "...")

    CHECK = True

    moduli = [x for x in range(1000_000, 1010000) if flint.fmpz(x).is_prime()]

    logging.info("Running with 16 moduli (old kernel)")
    Mat = SpMV(dense, plus, minus, basis, weight)
    Mat.wiedemann_multi(moduli[:16], check=CHECK)

    moduli64 = [10000000000000061] + moduli

    Mat2 = BlockCOO(dense, plus, minus, basis, weight)
    logging.info("Running BlockCOO v2 with 8 moduli")
    Mat2.wiedemann_multi(moduli[:8], check=CHECK)
    logging.info("Running BlockCOO v2 with 32 moduli")
    Mat2.wiedemann_multi(moduli[:32], check=CHECK)
    logging.info("Running BlockCOO v2 with 128 moduli")
    Mat2.wiedemann_multi(moduli[:128], check=CHECK)
    logging.info("Running BlockCOO v2 with 256 moduli")
    Mat2.wiedemann_multi(moduli[:256], check=CHECK)

    logging.info("Running BlockCOO v2 with 8 moduli64")
    Mat2.wiedemann_multi(moduli64[:8], check=CHECK)
    logging.info("Running BlockCOO v2 with 32 moduli64")
    Mat2.wiedemann_multi(moduli64[:32], check=CHECK)
    logging.info("Running BlockCOO v2 with 128 moduli64")
    Mat2.wiedemann_multi(moduli64[:128], check=CHECK)
    logging.info("Running BlockCOO v2 with 256 moduli64")
    Mat2.wiedemann_multi(moduli64[:256], check=CHECK)

if __name__ == "__main__":
    main()
