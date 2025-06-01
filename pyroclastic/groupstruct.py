"""
Computation of class group structure

This assumes knowledge of the class number (which is the group order).
Since abelian groups are direct sums of groups of order p^k (p-Sylow),
the computation is done separately for each prime factor of the group
order.
"""

import argparse
import json
import logging
import math
import pathlib
import time

import flint
from . import algebra
import pyroclastic_flint_extras as flint_extras


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("DATADIR")
    args = argp.parse_args()
    main_impl(args)


def main_impl(args):
    logging.getLogger().setLevel(logging.INFO)
    datadir = pathlib.Path(args.DATADIR)
    with open(datadir / "args.json") as j:
        clsargs = json.load(j)
    D = clsargs["d"]
    logging.info(f"D = {D}")

    with open(datadir / "classnumber") as f:
        h = int(f.read())
    logging.info(f"{h =}")

    primes = set()
    with open(datadir / "relations.filtered") as f:
        for line in f:
            primes.update(abs(int(l.partition("^")[0])) for l in line.split())
    primes = sorted(primes)
    logging.info(f"Basis has {len(primes)} primes")

    factors = flint.fmpz(h).factor()
    gfactors = []
    dlogs = []
    partial = False
    for p, k in sorted(factors):
        p, k = int(p), int(k)
        logging.info(f"Class number factor {p}^{k}")
        if p**k * len(primes) > 1e15:
            logging.warning(f"Skip {p}^{k}")
            partial = True
            continue
        t0 = time.monotonic()
        # This is fast enough if p^k = O(log(D)^2 len(primes))
        pk, logs = coord_hashtable(D, h, p, p**k, primes)
        dt = time.monotonic() - t0
        assert len(logs) == len(primes)
        logging.info(f"Coordinates computed in {dt:.3f}s")
        gfactors += list(pk)
        dlogs.append(logs)

    print("Invariants")
    print(" ".join(str(f) for f in gfactors))
    for lidx, l in enumerate(primes):
        vlog = []
        for d in dlogs:
            vlog += list(d[lidx])
        print(l, vlog)
    if partial:
        logging.warning("Class group structure is incomplete")


def coord_hashtable(D, h, p, pk, primes):
    """
    Computes coordinates for a set of primes in the subgroup
    of order p^k of Cl(sqrt(D)), using a BSGS or lookup table search

    Returns:
    - a group structure (tuple of p^k1..p^kn such that G_p = Z/p^k1 x ... Z/p^kn)
    - a list of coordinates coords = [(a1, ..., an)]

    such that primes[i] -> coords[i] defines an isomorphism between
    the p-component of the class group with Z/p^k1 x ... Z/p^kn

    The complexity is O(max(P, sqrt(pk P))) where P=len(primes)
    """
    assert h % pk == 0
    assert math.gcd(h // pk, pk) == 1

    # Multiply each prime form by (h//pk) to map to the order p^k subgroup.
    qs = [flint_extras.qfb.prime_form(D, l) ** (h // pk) for l in primes]

    # Compute order of each element
    max_order = 1
    max_l = None
    for l, q in zip(primes, qs):
        if D % 8 == 5:
            assert l != 2
        assert (q**pk).q()[0] == 1
        order = 1
        qp = q
        while qp.q()[0] != 1:
            qp = qp**p
            order *= p
        if order > max_order:
            max_order, max_l = order, l
        if order == pk:
            break
    if max_order < pk:
        # Compute group invariants
        raise NotImplementedError("group is not cyclic")

    # print(max_order, max_l)
    if max_order == pk:
        # Group is cyclic (this is the most common case)
        # We need to compute len(primes) discrete logs in a group of order p^k,
        # this needs p^k/2K + P * K/2 group operations
        # the optimal hash table size is p^k/K = sqrt(p^k len(primes))
        K = max(1, int(round(math.sqrt(pk / len(primes)))))
        if K > 1:
            logging.info(f"Using BSGS with size {K}×{pk // K + 1}")

        g = flint_extras.qfb.prime_form(D, max_l) ** (h // pk)
        gK = g**K
        Gp = [g**0, gK]
        gi = gK
        for _ in range(max_order // K // 2 + 1):
            gi = gi * gK
            Gp.append(gi)
        assert K * (2 * len(Gp) - 1) >= pk
        logging.info(f"G[{pk}] is cyclic")
        dlog = {}
        for i, q in enumerate(Gp):
            qa, qb, qc = q.q()
            dlog[(int(qa), int(qb), int(qc))] = K * i
            dlog[(int(qa), -int(qb), int(qc))] = pk - K * i
        # print(dlog)

        dlogs = []
        for l, q in zip(primes, qs):
            qj = q
            dl = None
            for j in range(K):
                key = tuple(int(_x) for _x in qj.q())
                if key in dlog:
                    # [l] + [g]^j = [g]^aK => [l] = [g]^(aK-j)
                    dl = (dlog[key] - j) % pk
                    break
                qj = qj * g
            assert dl is not None
            dlogs.append((dl,))

        return (pk,), dlogs


if __name__ == "__main__":
    main()
