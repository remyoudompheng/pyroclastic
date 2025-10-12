import math

import flint
import numpy as np
import pyroclastic_flint_extras as flint_extras


def smallprimes(B):
    l = np.ones(B, dtype=np.uint8)
    l[0:2] = 0
    for i in range(math.isqrt(B) + 1):
        if l[i] == 0:
            continue
        l[i * i :: i] = 0
    return [int(_i) for _i in l.nonzero()[0]]


def primebase(N: int, B):
    primes = smallprimes(B)
    for p in primes:
        try:
            # flint.nmod.sqrt is only available in python-flint 0.7.0
            r = flint_extras.sqrtmod(N, p)
            # Normalize so that r has the same parity as N
            if r & 1 != N & 1:
                r = p - r
            yield p, r
        except Exception:
            pass


def product(l: list[int]) -> int:
    if len(l) == 0:
        return 1
    p = l[0]
    for x in l[1:]:
        p *= x
    return p


def crt_basis(moduli: list[int]):
    m = product(moduli)
    basis = []
    for i, mi in enumerate(moduli):
        c = (m // mi) * pow(m // mi, -1, mi)
        basis.append(c)
    return basis


def is_probable_class_number(D: int, h: int):
    # Check primes up to approx. the Grenié-Molteni experimental bound
    for l, _ in primebase(D, max(1000, D.bit_length() ** 2 // 2)):
        if l == 2:
            # FIXME: prime_form is incorrect for l=2
            continue
        ql = flint_extras.qfb.prime_form(D, l)
        if (ql**h).q()[0] != 1:
            return False

    return True


def berlekamp_massey(seq: list[int], l: int):
    ctx = flint.fmpz_mod_poly_ctx(l)
    poly = ctx.minpoly(seq)
    return [int(coef) for coef in poly]


def h_approx(D, B=1_000_000):
    """
    Compute lower/upper bounds for

    h(D) = sqrt(|D|)/pi * prod(1/(1 - (D|p)/p) for prime p)

    >>> h = h_approx(-1139325066844575699589813265217200398493708241839938355464231)
    >>> h * 0.999 < 964415698883565364637432450736 < h * 1.001
    True
    >>> h = h_approx(-12239807779826253214859975412431303497371919444169932188160735019)
    >>> h * 0.999 < 109997901313565058259819609742265 < h * 1.001
    True
    >>> h = h_approx(-40000000000000000000000000000000000000000000000000000000000000004)
    >>> h * 0.999 < 178397819605839608466892693850112 < h * 1.001
    True
    """
    ps = smallprimes(B)
    pf = np.array(ps, dtype=np.float32)
    pf = 1.0 / pf

    for i, p in enumerate(ps):
        if p == 2:
            if D & 1 == 0:
                pf[i] = 0
            elif (D % 8) == 1:
                pf[i] = -pf[i]
            continue

        dp = pow(D, p // 2, p)
        if dp == p - 1:
            dp = -1
        pf[i] *= -dp

    plog = -np.log1p(pf)

    lsum = np.cumsum(plog)
    l = np.average(lsum[-len(ps) // 4 :])
    h = math.sqrt(float(abs(D))) / math.pi
    return h * math.exp(l)
