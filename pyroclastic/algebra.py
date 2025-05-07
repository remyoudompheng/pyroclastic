import math

import flint
import numpy as np


def smallprimes(B):
    l = np.ones(B, dtype=np.uint8)
    l[0:2] = 0
    for i in range(math.isqrt(B) + 1):
        if l[i] == 0:
            continue
        l[i * i :: i] = 0
    return [int(_i) for _i in l.nonzero()[0]]


def primebase(N, B):
    primes = smallprimes(B)
    for p in primes:
        np = flint.nmod(N, p)
        try:
            r = int(np.sqrt())
            # Normalize so that r is odd
            if r & 1 == 0:
                r = p - r
            yield p, r
        except Exception:
            pass


def h_approx(D):
    """
    Compute lower/upper bounds for

    h(D) = sqrt(|D|)/pi * prod(1/(1 - (D|p)/p) for prime p)

    >>> h1, h2 = h_approx(-1139325066844575699589813265217200398493708241839938355464231)
    >>> h1 < 964415698883565364637432450736 < h2 * 1.001
    True
    >>> h1, h2 = h_approx(-12239807779826253214859975412431303497371919444169932188160735019)
    >>> h1 < 109997901313565058259819609742265 < h2
    True
    >>> h1, h2 = h_approx(-40000000000000000000000000000000000000000000000000000000000000004)
    >>> h1 < 178397819605839608466892693850112 < h2
    True
    """
    ps = smallprimes(1_000_000)
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
    l1 = np.min(lsum[len(ps) // 4 :])
    l2 = np.max(lsum[len(ps) // 4 :])
    h = math.sqrt(float(abs(D))) / math.pi
    return h * math.exp(l1), h * math.exp(l2)
