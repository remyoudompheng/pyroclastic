"""
Procedures to repair matrices with redundant relations
"""

import argparse
import json
import logging
import pathlib
import random

import flint
from . import linalg
from . import linalg_alt
from . import relations

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("DATADIR")
    args = p.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main_impl(args)


def main_impl(args):
    rels = []
    datadir = pathlib.Path(args.DATADIR)

    with open(datadir / "args.json") as j:
        clsargs = json.load(j)
    D = clsargs["d"]
    logging.info(f"D = {D}")

    if (datadir / "relations.filtered").is_file():
        with open(datadir / "relations.filtered") as f:
            for l in f:
                facs = l.split()
                rels.append(
                    {int(p): int(e) for p, _, e in (f.partition("^") for f in facs)}
                )

        logging.info(f"Imported {len(rels)} relations")

    else:
        with open(datadir / "relations.sieve") as f:
            for l in f:
                rels.append([int(x) for x in l.split()])

        logging.info(f"Imported {len(rels)} relations (before prune/filter)")
        pruned, _ = relations.step_prune(rels, 0, 0, pathlib.Path(args.DATADIR))
        filtered = relations.step_filter(pruned, pathlib.Path(args.DATADIR))
        rels = filtered

    # Use a random linear projection as a "hashing" function
    primes = set()
    for r in rels:
        primes.update(r)
    basis = sorted(primes)
    proj = [random.getrandbits(100) for b in basis]
    proj_d = {p:x for p, x in zip(basis, proj)}

    def fproj(r: dict) -> int:
        return sum(e * proj_d[l] for l, e in r.items())

    relp = [fproj(r) for r in rels]
    relp_d = {}
    for ridx, x in enumerate(relp):
        relp_d.setdefault(x, []).append(ridx)

    # Find easy relations x == k*y with small k
    seen = set()
    for ridx, rproj in enumerate(relp):
        for k in range(-5, 5):
            for j in relp_d.get(k * rproj, []):
                if j == ridx:
                    continue
                key = tuple(sorted([ridx, j]))
                if key in seen:
                    continue
                seen.add(key)
                logging.info("DUPLICATE")
                logging.info(f"rels[{ridx}] = {showrel(rels[ridx])}")
                logging.info(f"rels[{j}] = {showrel(rels[j])}")
                rels[j] = None
                logging.info(f"Removed rels[{j}]")

    # Find relations ax+by+cz == 0
    if len(rels) < 1000:
        # If we have few relations we search exhaustively for x == ay+bz
        sums = {}
        for ry, pry in enumerate(relp):
            for rz, prz in enumerate(relp):
                if rz >= ry:
                    break
                if rels[ry] is None:
                    continue
                if rels[rz] is None:
                    continue
                for a in (-1, 1):
                    for b in (-1, 1):
                        if 0 in (a, b):
                            continue
                        pab = a*pry + b*prz
                        if pab in sums:
                            print(f"{a}*[{ry}] + {b}*[{rz}] = {sums[pab]} !!!")
                            logging.info("DUPLICATE")
                            rels[rz] = None
                            logging.info(f"Removed rels[{rz}]")
                        sums[pab] = (a, b, ry, rz)
                        if a and b and (pab in relp_d or (-pab) in relp_d):
                            if pab in relp_d:
                                rx = relp_d[pab][0]
                            else:
                                rx = relp_d[-pab][0]
                            if rels[rx] is None:
                                continue
                            print("x", showrel(rels[rx]))
                            print("y", showrel(rels[ry]))
                            print("z", showrel(rels[rz]))
                            print(f"{a} * y@rels[{ry}] + {b} * z@rels[{rz}] =? ±x@rels[{rx}]")
                            logging.info("DUPLICATE")
                            rels[rx] = None
                            logging.info(f"Removed rels[{rx}]")
    else:
        # We assume that some large prime appears in x,y but not z
        by_prime = {}
        for ridx, r in enumerate(rels):
            if r is None:
                continue
            for p in r:
                if p > 1000:
                    by_prime.setdefault(p, []).append(ridx)
        for p, ridxs in by_prime.items():
            if len(ridxs) > 20:
                continue
            for idx, ri in enumerate(ridxs):
                if rels[ri] is None:
                    continue
                for rj in ridxs[:idx]:
                    if rels[rj] is None:
                        continue
                    ei = rels[ri][p]
                    ej = rels[rj][p]
                    xij = ej * relp[ri] - ei * relp[rj]
                    for k in range(-10, 10):
                        if k and xij % k == 0 and xij // k in relp_d:
                            rk = relp_d[xij // k]
                            if rels[rk] is None:
                                continue
                            print("x", showrel(rels[ri]))
                            print("y", showrel(rels[rj]))
                            print("z", showrel(rels[rk]))
                            logging.info("{ej} * [{ri}] + {-ei} * [{rj}] ?= {k} * [{rk}]")
                            rels[rk] = None
                            logging.info(f"Removed rels[{rk}]")

    with open(datadir / "relations.repaired", "w") as w:
        for r in rels:
            if r is None:
                continue
            line = " ".join(f"{p}^{e}" for p, e in sorted(r.items()))
            print(line, file=w)

    if not args.validate:
        return

    mod0 = random.getrandbits(48)
    moduli = [x for x in range(mod0, mod0 + 1000) if flint.fmpz(x).is_prime()][:2]
    logging.info(f"Check modulo {moduli}")

    rels = [r for r in rels if r is not None]
    basis1 = sorted(set(p for r in rels for p, e in r.items() if e))
    dim = len(basis1)
    while True:
        logging.info(f"Selecting new square submatrix (dim {dim})")
        subrels = random.sample(rels, dim)
        if len(basis1) == len(set(p for r in subrels for p, e in r.items() if e)):
            break

    basis, dense, plus, minus, norm = linalg.to_sparse_matrix(subrels)
    weight = sum(len(r) for r in subrels)
    M = linalg_alt.SpMV(dense, plus, minus, basis, weight)
    print(M.wiedemann_multi(moduli, check=True))

def showrel(rel: dict) -> str:
    return " ".join(f"{p}^{e}" for p, e in sorted(rel.items()))

if __name__ == "__main__":
    main()
