import time

from pyroclastic import algebra
from pyroclastic.groupstruct import coord_hashtable
import pyroclastic_flint_extras as flint_extras


def check_dlogs(D, h, pk, ls, dlogs):
    for idx, pi in enumerate(pk):
        assert h % pi == 0
        g = next(_l for _l, _log in zip(ls, dlogs) if _log[idx] == 1 and _l != 2)
        cofactor = h // pi
        qg = flint_extras.qfb.prime_form(D, g) ** cofactor
        print(f"Generator of {pk}-component is {g}^{cofactor}")
        for l, dlog in zip(ls, dlogs):
            if l == 2:
                continue
            ql = flint_extras.qfb.prime_form(D, l) ** cofactor
            assert ql == qg ** dlog[idx], (ql, qg ** dlog[idx])


def test_dlog_hashtable():
    # Has cyclic factor Z/9
    D = -1387261450734019405515128121096146831845394927368234408722739
    h = 555518778862928228849838520233
    ls = [l for l, _ in algebra.primebase(D, 10000) if l != 2]
    t = time.monotonic()
    pk, dlogs = coord_hashtable(D, h, 3, 9, ls)
    dt = time.monotonic() - t
    print(f"Done {len(ls)} dlogs modulo 9 in {dt:.1f}s")

    assert pk == (9,)
    check_dlogs(D, h, pk, ls, dlogs)

    D = -926521594308185728247217331630315837344894790034275814586094016139277963552259
    h = 968406164605187692670594046363317127529
    ls = [l for l, _ in algebra.primebase(D, 10000) if l != 2]
    t = time.monotonic()
    pk, dlogs = coord_hashtable(D, h, 354251, 354251, ls)
    dt = time.monotonic() - t
    print(f"Done {len(ls)} dlogs modulo 354251 in {dt:.1f}s")

    assert pk == (354251,)
    check_dlogs(D, h, pk, ls, dlogs)

    D = -1380275789948747218615148789813397908331744431258453154803852163978010338546207690048972857749259
    h = 895778618231983069534949226241440759251213580187
    ls = [l for l, _ in algebra.primebase(D, 2000) if l != 2]
    t = time.monotonic()
    pk, dlogs = coord_hashtable(D, h, 19, 19**2, ls)
    dt = time.monotonic() - t
    print(f"Done {len(ls)} dlogs modulo 19^2 in {dt:.1f}s")

    assert pk == (19**2,)
    check_dlogs(D, h, pk, ls, dlogs)

    t = time.monotonic()
    pk, dlogs = coord_hashtable(D, h, 1661251, 1661251, ls)
    dt = time.monotonic() - t
    print(f"Done {len(ls)} dlogs modulo 1661251 in {dt:.1f}s")

    assert pk == (1661251,)
    check_dlogs(D, h, pk, ls, dlogs)
