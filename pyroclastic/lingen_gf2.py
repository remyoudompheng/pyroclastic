"""
Computation of linear generator for 1 or several sequences over GF(2)

This is the analogue for lingen for arithmetic modulo 2.
Currently it only implements the quadratic (original) version,
which does not require polynomial multiplication.

In this implementation matrices of GF(2) polynomials are represented
by matrices of big integers, using Python ints.

The implementation follows the conventions of:

Emmanuel Thomé
Fast computation of linear generators for matrix sequences and application to the block Wiedemann algorithm.
ISSAC '01: Proceedings of the 2001 international symposium on Symbolic and algebraic computation, Jul 2001, London, Ontario, Canada. pp.323-331,
https://inria.hal.science/inria-00517999
"""

import numpy
import numpy.typing as npt
import random

import flint

DEBUG_LINGEN = False

DEBUG_CHECK_LINGEN = False

# Check invariants of the main algorithm.
DEBUG_CHECK_INVARIANTS = False


def lingen(sequences, N: int):
    """
    Compute a (simultaneous) linear generating polynomial for input sequences.

    The polynomial sum a[i] X^i is such that:
       sum a[i] seq[k+i] = 0 for all k

    N: the expected degree of polynomial (matrix dimension in Block Wiedemann)
    """
    m = len(sequences)
    if m == 1:
        seq = sequences[0]
        assert len(seq) > 2 * N + 1
        ctx = flint.fmpz_mod_poly_ctx(2)
        pol = ctx.minpoly(seq)
        return flint.nmod_poly([int(b) for b in pol], 2)

    assert all(len(s) > N + N // m + 16 for s in sequences)

    M = numpy.zeros((m, 1), dtype=object)
    for i, seq in enumerate(sequences):
        M[i, 0] = sum(1 << i for i, b in enumerate(seq) if b)

    for t0 in range(m, m + 4):
        for _ in range(10):
            F = numpy.zeros((1, 1 + m), dtype=object)
            seen = set([0])
            for i in range(m):
                r = random.getrandbits(t0)
                while r in seen:
                    r = random.getrandbits(t0)
                seen.add(r)
                F[0, i] = r
            F[0, m] = 1 << t0

            # Compute matrix product E = (M @ F) >> m
            E = numpy.zeros((m, 1 + m), dtype=object)
            for i in range(m):
                for j in range(1 + m):
                    E[i, j] = clmul(M[i, 0], F[0, j]) >> t0

            E0 = (E[:, :m] & 1).astype(numpy.uint8)
            js = set()
            E0det = 1
            for i in range(m):
                try:
                    j0 = next(j for j in range(m) if j not in js and E0[i, j])
                except StopIteration:
                    E0det = 0
                    break
                js.add(j0)
                for j in range(m):
                    if j not in js and E0[i, j]:
                        E0[:, j] ^= E0[:, j0]
            if E0det != 0:
                break
        if E0det != 0:
            break

    assert E0det != 0
    # print("Selected t0 =", t0)
    # Constraint (Thomé, section 2.3.2)
    # Eij[0] must be a matrix of rank m

    delta = [t0 for _ in range(m + 1)]
    EXTRA_ITERS = 16

    if DEBUG_LINGEN:
        for i, j in numpy.ndindex(*E.shape):
            print(f"before E[{i},{j}]", hex(E[i, j]))

    P = mslgdc_array(E, delta, N + N // m + EXTRA_ITERS)
    if DEBUG_LINGEN:
        for i, j in numpy.ndindex(*E.shape):
            print(f"after E[{i},{j}]", hex(E[i, j]))

        print("P")
        print(P)
        print(delta)

    # We just want FP[i, 0]
    fp0 = 0
    for j in range(1 + m):
        fp0 ^= clmul(P[j, 0], F[0, j])

    fp0bits = bin(fp0)[2:]
    return [int(b) for b in fp0bits]


def lingen_mat(mats, N: int):
    """
    Compute a linear generating (matrix) polynomial for a matrix of sequences.

    The polynomial sum a[i] X^i is such that:
       sum a[i] seq[k+i] = 0 for all k

    N: the expected degree of polynomial (matrix dimension in Block Wiedemann)
    """
    m = len(mats)
    n = len(mats[0])

    assert all(len(s) > N // m + N // n + 16 for row in mats for s in row)

    M = numpy.zeros((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            seq = mats[i][j]
            M[i, j] = sum(1 << i for i, b in enumerate(seq) if b)

    for t0 in range(m, m + 4):
        for _ in range(10):
            # To obtain a nonsingular matrix, choose a larger number of columns
            EXTRA = 16
            F = numpy.zeros((n, m + EXTRA), dtype=object)
            for i in range(n):
                for j in range(m + EXTRA):
                    r = random.getrandbits(t0)
                    F[i, j] = r

            # Compute matrix product E = (M @ F) >> m
            E = numpy.zeros((m, m + EXTRA), dtype=object)
            for i in range(m):
                for j in range(m + EXTRA):
                    eij = 0
                    for k in range(n):
                        eij ^= clmul(M[i, k], F[k, j]) >> t0
                    E[i, j] = eij

            # Now find a nonsingular submatrix
            E0 = (E & 1).astype(numpy.uint8)
            js = set()
            sing = False
            for i in range(m):
                try:
                    j0 = next(j for j in range(m + EXTRA) if j not in js and E0[i, j])
                except StopIteration:
                    sing = True
                    break
                js.add(j0)
                for j in range(m):
                    if j not in js and E0[i, j]:
                        E0[:, j] ^= E0[:, j0]

            # Extract submatrix and augment with X^t0 Identity
            Esub = numpy.zeros((m, n + m), dtype=object)
            Fsub = numpy.zeros((n, n + m), dtype=object)
            for i, j in enumerate(sorted(js)):
                Esub[:, i] = E[:, j]
                Fsub[:, i] = F[:, j]
            for i in range(n):
                Fsub[i, i + m] = 1 << t0
                Esub[:, i + m] = M[:, i]
            E, F = Esub, Fsub

            if not sing:
                break
        if not sing:
            break

    assert not sing
    # print("Selected t0 =", t0)
    # Constraint (Thomé, section 2.3.2)
    # Eij[0] must be a matrix of rank m

    delta = [t0 for _ in range(m + n)]
    EXTRA_ITERS = 16

    if DEBUG_LINGEN:
        for i, j in numpy.ndindex(*E.shape):
            print(f"before E[{i},{j}]", hex(E[i, j]))

    P = mslgdc_array(E, delta, N // m + N // n + EXTRA_ITERS)
    if DEBUG_LINGEN:
        for i, j in numpy.ndindex(*E.shape):
            print(f"after E[{i},{j}]", hex(E[i, j]))

        print("P")
        print(P)
        print(delta)

    FP = numpy.zeros((n, n + m), dtype=object)
    degree = 0
    for i in range(m):
        for j in range(n + m):
            fij = 0
            for k in range(n + m):
                fij ^= clmul(P[k, j], F[i, k])
            if j < n:
                degree = max(degree, fij.bit_length())
            FP[i, j] = fij

    if DEBUG_CHECK_LINGEN:
        b = N // m + N // n + 1
        # The nxn square matrix D (left half of FP) must be such
        # that MD is small.
        for i in range(m):
            for j in range(n):
                mij = 0
                for k in range(n):
                    mij ^= clmul(M[i, k], FP[k, j])
                mij_trunc = mij & ((1 << b) - 1)
                if max(i, j) <= 2:
                    print(f"D[{i},{j}] length {FP[i, j].bit_length()}")
                    print(f"MD[{i},{j}] length {mij_trunc.bit_length()}")
                assert mij_trunc.bit_length() < N // m + 64, mij_trunc.bit_length()

    poly = [numpy.zeros((m, n), dtype=numpy.uint8) for _ in range(degree)]
    for i in range(m):
        for j in range(n):
            fpij = FP[i, j]
            assert fpij.bit_length() <= degree
            for k in range(degree):
                poly[k][i, j] = (fpij >> k) & 1
    return poly


def clmul(x: int, y: int, karatsuba=10000, block=True) -> int:
    """
    Polynomial multiplication over GF(2). y is usually smaller than x

    >>> clmul(57, 73)
    4017
    >>> clmul(123456789, 1234)
    135009063098
    """
    nx, ny = x.bit_length(), y.bit_length()
    # Basic Karatsuba for large inputs
    if karatsuba and min(nx, ny) > karatsuba:
        M = (nx + ny) // 4
        mask = (1 << M) - 1
        xlo, ylo = x & mask, y & mask
        xhi, yhi = x >> M, y >> M
        A = clmul(xlo, ylo, karatsuba, block)
        B = clmul(xlo ^ xhi, ylo ^ yhi, karatsuba, block)
        C = clmul(xhi, yhi, karatsuba, block)
        return A ^ ((A ^ B ^ C) << M) ^ (C << (2 * M))
    if nx < ny:
        x, y = y, x
        nx, ny = ny, nx
    if block and ny > 64:
        # Block multiplication
        mk = [0, x]
        for i in (1, 2, 3):
            xi = x << i
            mk += [m ^ xi for m in mk]
        z = 0
        for k in range(0, ny, 4):
            z ^= mk[(y >> k) & 15] << k
    else:
        z = 0
        for k in range(ny):
            if y & 1:
                z ^= x << k
            y >>= 1
    return z


def mslgdc(E, delta, b):
    P = None
    for _ in range(b):
        P, delta = lingen_step(E, delta, P=P)
        # print(delta)
    return P


def lingen_step(
    E: npt.NDArray, delta: list[int], P=None
) -> tuple[npt.NDArray, list[int]]:
    # Advance by 1 degree using Gauss elimination (ALGO1 in [Thomé])
    # E is modified in place and divided by X
    # P is optionally modified in place
    m, mn = E.shape
    if P is None:
        P = numpy.array(
            [[(1 if i == j else 0) for j in range(mn)] for i in range(mn)],
            dtype=object,
        )
    # Sort columns
    for i in range(mn):
        argmin = min(range(i, mn), key=lambda idx: (delta[idx], idx))
        if i < argmin:
            delta[i], delta[argmin] = delta[argmin], delta[i]
            for j in range(mn):
                P[j, i], P[j, argmin] = P[j, argmin], P[j, i]
            for j in range(m):
                E[j, i], E[j, argmin] = E[j, argmin], E[j, i]
    # Eliminate without increasing degree (col[j] is cancelled using col[j0<j])
    nonzero = set()
    for i in range(m):
        # Find first nonzero element of row i
        try:
            j0 = next(j for j in range(mn) if (E[i, j] & 1) and j not in nonzero)
        except StopIteration:
            continue
        nonzero.add(j0)
        assert E[i, j0] & 1 == 1
        # Use it to eliminate next columns
        for j in range(j0 + 1, mn):
            if E[i, j] & 1:
                for h in range(m):
                    E[h, j] ^= E[h, j0]
                for h in range(mn):
                    P[h, j] ^= P[h, j0]
    if DEBUG_LINGEN:
        for i, j in numpy.ndindex(*E.shape):
            if E[i, j] & 1:
                assert j in nonzero
    # Now each row has 1 nonzero coefficient
    for j in nonzero:
        delta[j] += 1
        for i in range(mn):
            P[i, j] <<= 1
    # Combine multiplication by X and division by X
    for j in range(mn):
        if j not in nonzero:
            for i in range(m):
                assert E[i, j] & 1 == 0
                E[i, j] >>= 1
    # debug("P", P)
    return P, delta


def mslgdc_array(E, delta, b):
    B = b // 32 + 1
    m, mn = E.shape
    EE = numpy.zeros((m, mn, B), dtype=numpy.uint32)
    for i in range(m):
        for j in range(mn):
            eij = E[i, j].to_bytes(max(4 * B, E[i, j].bit_length() // 8 + 1), "little")
            EE[i, j] = [
                int.from_bytes(eij[4 * k : 4 * k + 4], "little") for k in range(B)
            ]

    P = None
    t0 = delta[0]
    for t in range(t0, max(t0 + 1, 32 * B)):
        P, delta = lingen_step_array(EE, delta, t, P=P)
        # print(delta)

    # Output by increasing value of delta
    delta_sort = sorted((_d, _i) for _i, _d in enumerate(delta))
    Pint = numpy.zeros((mn, mn), dtype=object)
    for i in range(mn):
        for j in range(mn):
            jsort = delta_sort[j][1]
            pij = b"".join([int(pijk).to_bytes(4, "little") for pijk in P[i, jsort, :]])
            Pint[i, j] = int.from_bytes(pij, "little")
    return Pint


def lingen_step_array(
    E: npt.NDArray, delta: list[int], t: int, P=None
) -> tuple[npt.NDArray, list[int]]:
    """
    Similar to lingen_step but operates on numpy.array:
    each polynomial is a row of uint32 integers.
    """
    # print("t =", t)
    m, mn, b = E.shape
    if P is None:
        P = numpy.zeros((mn, mn, b), dtype=numpy.uint32)
        for i in range(mn):
            P[i, i, 0] = 1

    dege = b - t // 32
    # Sort columns
    delta_sort = sorted((_d, _i) for _i, _d in enumerate(delta))
    idx_sort = [_i for _, _i in delta_sort]
    if DEBUG_CHECK_INVARIANTS:
        # Check invariants:
        # PM = G + X^t E
        # where degree(G[j]) < delta[j] and degree(E[j]) <= N-t+delta[j]
        #
        # Actually we only consider PM mod X^b, so we can truncate E to degree b-t
        for i in range(mn):
            le = (t - delta[i]) // 32
            if le > 0:
                for j in range(m):
                    assert numpy.all(E[j, i, -le:] == 0), (
                        delta[j],
                        t,
                        list(E[j, i, :]),
                        le,
                    )
            lp = delta[i] // 32 + 1
            for j in range(mn):
                assert numpy.all(P[j, i, lp:] == 0)
    # Eliminate without increasing degree (col[j] is cancelled using col[j0<j])
    nonzero = set()
    for i in range(m):
        # Find first nonzero element of row i
        try:
            j0 = next(
                j
                for j, jsort in enumerate(idx_sort)
                if (E[i, jsort, 0] & 1) and jsort not in nonzero
            )
        except StopIteration:
            continue
        jsort0 = idx_sort[j0]
        nonzero.add(jsort0)
        assert E[i, jsort0, 0] & 1 == 1
        # Use it to eliminate next columns
        # Note that delta[j0] <= delta[j]
        for j in idx_sort[j0 + 1 :]:
            lp = delta[j] // 32 + 1
            if E[i, j, 0] & 1:
                E[:, j, :dege] ^= E[:, jsort0, :dege]
                P[:, j, :lp] ^= P[:, jsort0, :lp]
    if DEBUG_LINGEN:
        for i, j in numpy.ndindex(*E.shape):
            if E[i, j] & 1:
                assert j in nonzero
    # Now each row has 1 nonzero coefficient
    for j in range(mn):
        if j in nonzero:
            # P[j] *= X
            delta[j] += 1
            lp = min(b, delta[j] // 32 + 1)
            P[:, j, 1:lp] = (P[:, j, 1:lp] << 1) | (P[:, j, 0 : lp - 1] >> 31)
            P[:, j, 0] <<= 1
        else:
            le = b - t // 32
            # E[j] /= X
            if dege == b:
                E[:, j, :-1] = (E[:, j, :-1] >> 1) | (E[:, j, 1:] << 31)
                E[:, j, -1] >>= 1
            else:
                E[:, j, :dege] = (E[:, j, :dege] >> 1) | (E[:, j, 1 : dege + 1] << 31)
    # debug("P", P)
    return P, delta


def benchmark_m1():
    import time
    import sys

    sys.set_int_max_str_digits(10000)

    for N in (1000, 2000, 4000, 20000):
        random.seed(42)
        taps = [random.getrandbits(1) for _ in range(min(200, N))]

        # A single sequence of size 2N+O(1) for reference
        seqlong = [random.getrandbits(1) for _ in range(N)]
        for _ in range(N + 64):
            t = sum(a * seqlong[-N + i] for i, a in enumerate(taps)) & 1
            seqlong.append(t)

        t0 = time.monotonic()
        refpoly = lingen([seqlong], N)
        dt = time.monotonic() - t0
        print(f"lingen for {N=} m=1 in {dt:.3f}s")
        assert refpoly.degree() <= N, len(refpoly)

        want = (N * [0]) + [1]
        for i in range(len(taps)):
            want[i] = taps[i]
        assert flint.nmod_poly(want, 2) % refpoly == 0

        for k in range(200, N - 200, 37):
            assert sum(refpoly[i] * seqlong[k + i] for i in range(N + 1)) == 0

        for m in (2, 3, 4, 16, 32, 64):
            # m sequences of size N+N/m+O(1) for testing
            seqs = []
            for _ in range(m):
                seq = [random.getrandbits(1) for _ in range(N)]
                for _ in range(N // m + 80):
                    t = sum(a * seq[-N + i] for i, a in enumerate(taps)) & 1
                    seq.append(t)
                seqs.append(seq)

            t0 = time.monotonic()
            minpoly = lingen(seqs, N)
            dt = time.monotonic() - t0
            print(f"lingen for {N=} {m=} n=1 in {dt:.3f}s")
            print(f"=> polynomial has degree {len(minpoly) - 1}")
            # print(minpoly)

            # pol2 = flint.nmod_poly(minpoly, 2)
            # assert pol2 % refpoly == 0


def benchmark_mn():
    import time

    for N in (1000, 2000, 4000, 20000):
        # random.seed(42)
        taps = [random.getrandbits(1) for _ in range(min(200, N))]

        # for m in (2, 3, 4, 16):
        for m in (4, 16, 32, 64):
            # mn sequences of size N/m+N/n+O(1) for testing
            # First we build m sequences
            seqs = []
            for _ in range(m):
                seq = [random.getrandbits(1) for _ in range(N)]
                for _ in range(N):
                    t = sum(a * seq[-N + i] for i, a in enumerate(taps)) & 1
                    seq.append(t)
                seqs.append(seq)
            # Then we compute m projections
            projs = []
            for _ in range(m):
                idx = random.sample(list(range(N // 2)), 20)
                projs.append(idx)

            mats = [[] for _ in range(m)]
            for i in range(m):
                seq = seqs[i]
                for j, idx in enumerate(projs):
                    seqij = [
                        sum(seq[k + j] for j in idx) & 1
                        for k in range(N // m + N // m + 64)
                    ]
                    mats[j].append(seqij)

            t0 = time.monotonic()
            _minpoly = lingen_mat(mats, N)
            dt = time.monotonic() - t0
            print(f"lingen for {N=} {m=} n={m} in {dt:.3f}s")


if __name__ == "__main__":
    benchmark_mn()
