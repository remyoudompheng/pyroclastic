import random
import math
import numpy as np

from pyroclastic import algebra
from pyroclastic import relations
from pyroclastic import linalg
from pyroclastic import linalg_alt


def random_relations(N: int, primes: list, size: int, dense: list):
    sumlp = sum(math.log2(p) / p for p in primes)
    scale = size / sumlp
    assert scale < 20
    print("probability scale", scale)
    probs = scale / np.array(primes, dtype=np.float32)

    rels = []
    # Random exponents with average scale/p and random sign
    for _ in range(N):
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
    out = ref_matmul_big(rels, ps, v)
    return np.array(out, dtype=np.int32)


def ref_matmul_big(rels, ps, v) -> list:
    prime_idx = {p: idx for idx, p in enumerate(ps)}
    out = []
    for r in rels:
        x = 0
        for p, e in r.items():
            x += e * v[prime_idx[p]]
        out.append(x)
    return out


MATRIX = None


def init_random_matrix():
    global MATRIX
    if MATRIX is not None:
        return MATRIX

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

    pruned, _ = relations.step_prune(rels, 0, 0)
    result = relations.step_filter(pruned, None)
    print(result[0])

    print("Truncating to square matrix")
    basis = sorted(set(p for r in result for p, e in r.items() if e))
    dim = len(basis)
    while True:
        subres = random.sample(result, dim)
        if len(basis) == len(set(p for r in subres for p, e in r.items() if e)):
            MATRIX = (subres, basis)
            return MATRIX


MATRIX_LARGE = None


def random_matrix_large():
    global MATRIX_LARGE
    if MATRIX_LARGE is not None:
        return MATRIX_LARGE

    basis = [p for p, _ in algebra.primebase(-1, 2000_000)]
    print("dim", len(basis))
    rels = []
    for j in range(len(basis)):
        rel = {}
        rel[basis[j]] = 1
        bits = random.getrandbits(130)
        for i, l in enumerate(random.sample(basis[:30], 20)):
            rel[l] = -1 if (bits >> i) & 1 else 1
        for i, l in enumerate(random.sample(basis[50:], 30)):
            rel[l] = -1 if (bits >> (30 + i)) & 1 else 1
        rels.append(rel)

    MATRIX_LARGE = (rels, basis)
    return MATRIX_LARGE


def unsorted_sparse_matrix(rows):
    linalg.DEBUG_NO_SORT_ROWS = True
    resp = linalg.to_sparse_matrix(rows)
    linalg.DEBUG_NO_SORT_ROWS = False
    return resp


def test_spmv():
    rows, basis = init_random_matrix()
    dim = len(rows)

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(rows, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    assert sorted(basis2) == basis
    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)

    m2 = linalg_alt.SpMV(dense, plus, minus, basis2, weight)
    mv2 = m2.mulvec(v2, 65537)
    print(mv2)
    print((mv1 - mv2).nonzero())
    assert np.all(mv1 == mv2)


def test_spmv_big():
    rows, basis = init_random_matrix()
    dim = len(rows)

    p = 100000000000000000000000000000049  # 107 bits

    v = [random.randrange(p) for _ in range(dim)]
    mv1 = [x % p for x in ref_matmul_big(rows, basis, v)]
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    assert sorted(basis2) == basis
    v2 = [v[idx1[p]] for p in basis2]

    m2 = linalg_alt.SpMV(dense, plus, minus, basis2, weight)
    mv2 = m2.mulvec_big(v2, p)
    # Is it correct modulo p?
    assert [x2 % p == x1 for x1, x2 in zip(mv1, mv2)]
    # Is it already reduced modulo p?
    assert [x2 == x1 for x1, x2 in zip(mv1, mv2)]


def test_spmv_large():
    rows, basis = random_matrix_large()
    dim = len(rows)

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(rows, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)
    m2 = linalg_alt.SpMV(dense, plus, minus, basis, weight)
    mv2 = m2.mulvec(v2, 65537)
    print(mv2)
    if not np.all(mv1 == mv2):
        print("diff at", (mv1 - mv2).nonzero())
    assert np.all(mv1 == mv2)


def test_blockcoo():
    rows, basis = init_random_matrix()
    dim = len(rows)

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(rows, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    assert sorted(basis2) == basis
    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)

    m2 = linalg_alt.BlockCOO(dense, plus, minus, basis2, weight, BM=32)
    mv2 = m2.mulvec(v2, 65537)
    print(mv2)
    assert np.all(mv1 == mv2)


def test_csr_matrix():
    rows, basis = init_random_matrix()
    dim = len(rows)

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(rows, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)
    m4 = linalg.CSRMatrix(dense, plus, minus, basis2, weight)
    mv4 = m4.matmul_small(65537, v2)
    print(mv4)
    assert np.all(mv1 == mv4)

    mv5 = m4.matmul_medium(65537, v2, CHUNK_N=dim * 2 // 3)
    print(mv5)
    assert np.all(mv1 == mv5)


def test_blockcoo3():
    rows, basis = init_random_matrix()
    dim = len(rows)

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(rows, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)
    m6 = linalg.BlockCOO(dense, plus, minus, basis, weight)
    mv6 = m6.mulvec(v2, 65537)
    print(mv6)
    assert np.all(mv1 == mv6)


def test_blockcoo3_large():
    rows, basis = random_matrix_large()
    dim = len(rows)

    v = np.random.randint(0, 65537, dim, dtype=np.int32)
    mv1 = ref_matmul(rows, basis, v) % 65537
    assert mv1.shape == (dim,), mv1.shape
    print(mv1)

    idx1 = {p: i for i, p in enumerate(basis)}

    basis2, dense, plus, minus, weight = unsorted_sparse_matrix(rows)
    v2 = np.array([v[idx1[p]] for p in basis2], dtype=np.int32)
    m2 = linalg.BlockCOO(dense, plus, minus, basis, weight)
    mv2 = m2.mulvec(v2, 65537)
    print(mv2)
    assert np.all(mv1 == mv2)
