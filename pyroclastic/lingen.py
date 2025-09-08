"""
Block Wiedemann algorithm for small moduli

The implementation follows the algorithm description in:

Emmanuel Thomé
Fast computation of linear generators for matrix sequences and application to the block Wiedemann algorithm.
ISSAC '01: Proceedings of the 2001 international symposium on Symbolic and algebraic computation, Jul 2001, London, Ontario, Canada. pp.323-331,
https://inria.hal.science/inria-00517999

Since the cost of block Wiedemann is significantly higher than
ordinary generating polynomials for single sequences
(by a constant factor), it may not be relevant to use it with large block sizes.

The recommended use for computation of large determinants is m=2,3,4 and n=1.
Using larger n has no benefit if the computation of determinant modulo
small primes is already vectorized.
"""

import random
import time

import numpy
import flint

from pyroclastic import algebra
from pyroclastic_flint_extras import (
    nmod_poly_clamp,
    nmod_poly_copy_trunc,
    nmod_poly_mul_low,
)


def random_poly(d, p):
    return flint.nmod_poly([random.randrange(p) for _ in range(d)], p)


def block_wiedemann_step(E, delta, p, P=None):
    # Advance by 1 degree using Gauss elimination (ALGO1 in [Thomé])
    m, mn = E.shape
    if P is None:
        P = numpy.array(
            [
                [flint.nmod_poly([1 if i == j else 0], p) for j in range(mn)]
                for i in range(mn)
            ],
            dtype=object,
        )
    # Sort columns
    delta = list(iter(delta))
    for i in range(mn):
        argmin = min(range(i, mn), key=lambda idx: (delta[idx], idx))
        if i < argmin:
            delta[i], delta[argmin] = delta[argmin], delta[i]
            # Fancy numpy syntax for swapping
            for j in range(mn):
                P[j, i], P[j, argmin] = P[j, argmin], P[j, i]
            for j in range(m):
                E[j, i], E[j, argmin] = E[j, argmin], E[j, i]
    # Eliminate without increasing degree (col[j] is cancelled using col[j0<j])
    nonzero = set()
    for i in range(m):
        # Find first nonzero element of row i
        try:
            j0 = next(j for j in range(mn) if E[i, j][0] and j not in nonzero)
        except StopIteration:
            continue
        nonzero.add(j0)
        # Use it to eliminate next columns
        for j in range(j0 + 1, mn):
            k = E[i, j][0] / E[i, j0][0]
            for h in range(m):
                E[h, j] -= k * E[h, j0]
            for h in range(mn):
                P[h, j] -= k * P[h, j0]
    if False:
        for i, j in numpy.ndindex(*E.shape):
            if E[i, j][0]:
                assert j in nonzero
    # Now each row has 1 nonzero coefficient
    # assert len(nonzero) <= m
    x = flint.nmod_poly([0, 1], p)
    for j in nonzero:
        delta[j] += 1
        for i in range(mn):
            P[i, j] = P[i, j] * x
    # Combine multiplication by X and division by X
    for j in range(mn):
        if j in nonzero:
            continue
        for i in range(m):
            assert E[i, j][0] == 0
            E[i, j] //= x
    return P, delta


MSLGDC_THRESHOLD = 16


def mslgdc(E, delta, b, modulus, depth=0):
    """
    Returns a matrix P operating on columns (right multiplication)
    such that E * P is zero up to degree b and E[x^b] has rank m
    and delta*P is a valid profile for new polynomial degrees.

    E: a numpy.array with coefficients flint.nmod_poly
    """
    m, mn = E.shape
    p = modulus

    if b <= MSLGDC_THRESHOLD:
        P = None
        for _ in range(b):
            P, delta = block_wiedemann_step(E, delta, p, P=P)
        return P

    assert b > 2

    # Handle the low degree side
    EL = numpy.zeros((m, mn), dtype=object)
    bhalf = b // 2
    for i in range(m):
        for j in range(mn):
            EL[i, j] = nmod_poly_copy_trunc(E[i, j], bhalf)
    PL = mslgdc(EL, delta, b // 2, p, depth=depth + 1)

    # Compute right part
    ER = numpy.zeros((m, mn), dtype=object)
    for i in range(m):
        for j in range(mn):
            for k in range(mn):
                ER[i, j] += nmod_poly_mul_low(E[i, k], PL[k, j], b)
            nmod_poly_clamp(ER[i, j], b // 2, b)
    delta_R = [max(delta[i] + PL[i, j].degree() for i in range(mn)) for j in range(mn)]
    PR = mslgdc(ER, delta_R, b - b // 2, p, depth=depth + 1)
    return PL @ PR


def generating_polynomial_multi(seqs, n, p):
    m = len(seqs)
    assert all(len(s) > n + n // m + 16 for s in seqs)

    if m == 1:
        return algebra.berlekamp_massey(seqs[0], p)

    M = numpy.zeros((m, 1), dtype=object)
    for i in range(m):
        M[i, 0] = flint.nmod_poly(seqs[i], p)

    F = numpy.zeros((1, 1 + m), dtype=object)
    for i in range(m):
        F[0, i] = random_poly(m, p)
    F[0, m] = flint.nmod_poly(m * [0] + [1], p)

    E = M @ F
    for i, j in numpy.ndindex(*E.shape):
        E[i, j] = flint.nmod_poly(list(E[i, j])[m:], p)

    delta = tuple(m for _ in range(m + 1))
    P = mslgdc(E, delta, n + n // m + 1, p)

    FP = F @ P
    minpoly = FP[0, 0]
    assert 0 < minpoly.degree() <= n
    # Adjust result to match FLINT minpoly convention
    # (reversed and monic) => minpoly(1/X)*X^N
    shift = n - minpoly.degree()
    xs = flint.nmod_poly(shift * [0] + [1], p)
    minpoly = minpoly.reverse() * xs
    minpoly /= minpoly[minpoly.degree()]
    return [int(coef) for coef in minpoly]


def main():
    p = 10000000000000061
    random.seed(42)
    for N in (10000, 100000, 1000000):
        taps = [random.randrange(p) for _ in range(min(200, N))]

        # A single sequence of size 2N+O(1) for reference
        seqlong = [random.randrange(p) for _ in range(N)]
        for _ in range(N + 64):
            t = -sum(a * seqlong[-N + i] for i, a in enumerate(taps)) % p
            seqlong.append(t)

        t0 = time.monotonic()
        refpoly = generating_polynomial_multi([seqlong], N, p)
        assert len(refpoly) == N + 1
        dt = time.monotonic() - t0

        print(
            f"generating_polynomial_multi for {N=} m=1 p={p.bit_length()}b in {dt:.3f}s"
        )
        refpoly = list(refpoly)
        for i in range(N):
            assert int(refpoly[i]) == (taps[i] if i < len(taps) else 0)
        assert refpoly[N] == 1

        for k in range(200, 8000, 37):
            assert sum(refpoly[i] * seqlong[k + i] for i in range(N + 1)) % p == 0

        for m in (2, 3, 4):
            # m sequences of size N+N/m+O(1) for testing
            seqs = []
            for _ in range(m):
                seq = [random.randrange(p) for _ in range(N)]
                for _ in range(N // m + 64):
                    t = -sum(a * seq[-N + i] for i, a in enumerate(taps)) % p
                    seq.append(t)
                seqs.append(seq)

            t0 = time.monotonic()
            minpoly = generating_polynomial_multi(seqs, N, p)
            dt = time.monotonic() - t0
            print(
                f"generating_polynomial_multi for {N=} {m=} p={p.bit_length()}b in {dt:.3f}s"
            )
            for i in range(N + 1):
                assert refpoly[i] == minpoly[i]


if __name__ == "__main__":
    main()
