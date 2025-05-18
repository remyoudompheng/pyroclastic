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


def _test_binaryqf():
    """
    >>> q = flint_extras.qfb.prime_form(-103, 19)
    >>> q
    qfb(19, 7, 2)
    >>> q**-1
    qfb(19, -7, 2)
    >>> q**5
    qfb(1, 1, 26)
    >>> q * q**-1
    qfb(1, 1, 26)
    >>> q**0
    qfb(1, 1, 26)
    >>> q ** 10000000000001
    qfb(2, 1, 13)
    """
    pass
