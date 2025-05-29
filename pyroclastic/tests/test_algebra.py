import pyroclastic_flint_extras as flint_extras


def test_binaryqf():
    q = flint_extras.qfb.prime_form(-103, 19)
    assert str(q) == "qfb(19, 7, 2)"
    assert str(q**-1) == "qfb(19, -7, 2)"
    assert str(q**5) == "qfb(1, 1, 26)"
    assert str(q * q**-1) == "qfb(1, 1, 26)"
    assert str(q**0) == "qfb(1, 1, 26)"
    assert str(q**10000000000001) == "qfb(2, 1, 13)"
