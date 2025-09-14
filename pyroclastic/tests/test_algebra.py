import pickle

from pyroclastic import algebra
import pyroclastic_flint_extras as flint_extras


def test_binaryqf():
    q = flint_extras.qfb.prime_form(-103, 19)
    assert str(q) == "qfb(19, 7, 2)"
    assert str(q**-1) == "qfb(19, -7, 2)"
    assert str(q**5) == "qfb(1, 1, 26)"
    assert str(q * q**-1) == "qfb(1, 1, 26)"
    assert str(q**0) == "qfb(1, 1, 26)"
    assert str(q**10000000000001) == "qfb(2, 1, 13)"

    assert pickle.loads(pickle.dumps(q)) == q
    assert pickle.loads(pickle.dumps(q)).q() == q.q()


def test_berlekamp_massey_big():
    p = 10000000000000000000000000000000000000000000000000000000000000057
    seq = [12, 34, 56]
    for _ in range(10000):
        seq.append((seq[-1] + seq[-2] + seq[-3]) % p)

    poly = algebra.berlekamp_massey(seq, p)
    print(poly)
    assert poly == [p - 1, p - 1, p - 1, 1]
