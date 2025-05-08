import logging
import random
import math
import numpy as np

import kp
from tqdm import trange

from . import algebra
from . import gpu
from . import relations
from . import linalg

def random_relations(N: int, primes: list, size: int, dense: list):
    sum1p = sum(1.0 / p for p in primes)
    sumlp = sum(math.log2(p) / p for p in primes)
    scale = size / sumlp
    assert scale < 20
    print("probability scale", scale)
    probs = scale / np.array(primes, dtype=np.float32)

    rels = []
    # Random exponents with average scale/p and random sign
    for _ in trange(N):
        rel = []
        # small primes
        for i, p in enumerate(primes):
            if p >= 30:
                break
            if p == 2:
                e = np.random.geometric(1.0 - 1.0 / p, 1) - 1
            else:
                e = np.random.geometric(1.0 - 2.0 / p, 1) - 1
            sign = 2 * random.getrandbits(1) - 1
            for _ in range(e[0]):
                rel.append(sign * p)
        # large primes
        rands = np.random.random(len(primes)) < probs
        for idx in rands.nonzero()[0]:
            if primes[idx] >= 30:
                sign = 2 * random.getrandbits(1) - 1
                rel.append(sign * primes[idx])
        # add dense primes
        for d in random.sample(dense, 10):
            sign = 2 * random.getrandbits(1) - 1
            rel.append(sign * d)

        rels.append(rel)

    return rels


def ref_matmul(rels, ps, v):
    prime_idx = {p: idx for idx, p in enumerate(ps)}
    out = []
    for r in rels:
        x = 0
        for p, e in r.items():
            x += e * v[prime_idx[p]]
        out.append(x)
    return np.array(out, dtype=np.int32)



def gpu_matmul(dense, plus, minus, v):
    dim, dense_n = dense.shape
    assert (dim * dense_n) % 4 == 0
    SHADER = gpu.compile("spmv.comp", {"N": dim, "DENSE_N": dense_n})
    WGSIZE = 128

    mgr = kp.Manager(0)
    xd = mgr.tensor_t(dense.flatten().view(np.uint32))

    rowlen_plus = [len(l) for l in plus]
    rowlen_minus = [len(l) for l in minus]
    aidx_plus = np.cumsum(np.array([0] + rowlen_plus, dtype=np.uint32), dtype=np.uint32)
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

    xin = mgr.tensor_t(np.array(v, dtype=np.uint32))
    xout = mgr.tensor_t(np.zeros(dim, dtype=np.uint32))
    xmod = mgr.tensor_t(np.array([65537], dtype=np.uint32))

    algo = mgr.algorithm(
        [xd, xplus, xminus, xidxp, xidxm, xin, xout, xmod],
        SHADER,
        workgroup=((dim + WGSIZE - 1) // WGSIZE, 1, 1),
    )
    (
        mgr.sequence()
        .record(
            kp.OpTensorSyncDevice([xd, xplus, xminus, xidxp, xidxm, xin, xout, xmod])
        )
        .record(kp.OpAlgoDispatch(algo))
        .record(kp.OpTensorSyncLocal([xout]))
        .eval()
    )

    return np.copy(xout.data())


def main():
    logging.getLogger().setLevel(logging.DEBUG)

    N = -1139325066844575699589813265217200398493708241839938355464231
    ps = [p for p, _ in algebra.primebase(N, 300000)]
    print(len(ps), "primes")
    print("dense", ps[100], "..", ps[139])
    size = 160
    print("Target size", size)
    rels = random_relations(len(ps), ps, size, ps[100:140])
    print(rels[0])
    print(rels[1])
    avgsize = sum(sum(math.log2(abs(p)) for p in rel) for rel in rels) / len(rels)
    print(f"Generated {len(rels)} random relations, average size {avgsize:.1f}")

    pruned, _ = relations.prune2(rels, 0, 0)
    result = relations.step_filter(pruned, None)
    print(result[0])

    print("Truncating to square matrix")
    basis = sorted(set(p for r in result for p, e in r.items() if e))
    dim = len(basis)
    while True:
        subres = random.sample(result, dim)
        if len(basis) == len(set(p for r in subres for p, e in r.items() if e)):
            result = subres
            break

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(result, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus = linalg.to_sparse_matrix(result)
    assert sorted(basis2) == basis
    print("Dense block")
    print(dense)
    print("Rows +1")
    print(plus[:10], "...")
    print("Rows -1")
    print(minus[:10], "...")

    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)
    mv2 = gpu_matmul(dense, plus, minus, v2)
    print(mv2)

    assert np.all(mv1 == mv2)


if __name__ == "__main__":
    main()
