"""
Relation filtering for factoring (modulo 2)
Here relations are represented as numpy arrays,
and operations modulo 2 are using NumPy set operation methods.
"""

import logging
import pathlib
import time

from networkx import Graph, connected_components
import numpy
import numpy.typing as npt
from pyroclastic import algebra

RelFac = tuple[int, list[int]]

DEBUG_CHECK_RELFAC = True


def prune_gf2(
    n: int, rawrels: list[RelFac], datadir: pathlib.Path | None = None
) -> list[tuple[int, npt.NDArray]]:
    """
    Variant of prune for GF(2) linear algebra.

    A relation is a tuple (x, ys) such that x²=product(ys)
    """
    rels: list[tuple[int, npt.NDArray] | None] = []
    duplicates = 0
    seen = set()
    for x, rel in rawrels:
        if x in seen:
            duplicates += 1
            rels.append(None)
            continue
        seen.add(x)
        rel2 = []
        div = 1
        for l in sorted(rel):
            if rel2 and l == rel2[-1]:
                div *= l
                rel2.pop()
            else:
                rel2.append(l)
        x = x * pow(div, -1, n) % n
        if DEBUG_CHECK_RELFAC:
            assert x * x % n == algebra.product(rel2) % n
        rels.append((x, numpy.array(rel2, dtype=numpy.int32)))
    if duplicates:
        logging.warn(f"Found {duplicates} duplicate relations before pruning")

    # Prune relations in place: a removed relation is replaced by None.
    # We are only interested in coefficients ±1, exponent sign is ignored
    stats: dict[int, list[int] | None] = {}
    for ridx, r in enumerate(rels):
        if r is None:
            continue
        rx, rfacs = r
        for p in rfacs:
            stats_p = stats.setdefault(p, [])
            if stats_p is None:
                continue
            stats_p.append(ridx)
            if len(stats_p) > 20:
                stats[p] = None

    excess = len(rels) - len(stats)
    logging.info(f"[prune] {len(stats)} primes appear in relations")

    def prune(ridx):
        r = rels[ridx]
        if r is None:
            return
        for p in r[1]:
            sp = stats.get(p)
            if sp is not None:
                sp.remove(ridx)
                if len(sp) == 0:
                    del stats[p]
        rels[ridx] = None

    def score(clique):
        s = 0
        for ridx in clique:
            # Score is weight of relation
            # + bonus point is some primes have low weight.
            r = rels[ridx][1]
            s += len(r)
            for p in r:
                if (sp := stats.get(p)) is not None and len(sp) < 5:
                    s += 1
        return s

    while excess < 0:
        m1 = [p for p, rs in stats.items() if rs is not None and len(rs) == 1]
        singles = 0
        for p in m1:
            if stats_p := stats.get(p):
                prune(stats_p[0])
                singles += 1
        if singles:
            logging.info(f"[prune] pruned {singles} singletons")
            nr = sum(1 for r in rels if r is not None)
            excess = nr - len(stats)
            logging.info(f"[prune] {len(stats)} primes appear in relations")
        else:
            break

    removed = 0
    max_removed = (excess - 200) // 2
    while removed < max_removed:
        m1 = [p for p, rs in stats.items() if rs is not None and len(rs) == 1]
        singles = 0
        for p in m1:
            if stats_p := stats.get(p):
                prune(stats_p[0])
                singles += 1
        if singles:
            logging.info(f"[prune] pruned {singles} singletons")

        m2 = [(p, rs) for p, rs in stats.items() if rs is not None and len(rs) == 2]
        g = Graph()
        for p, rs in m2:
            g.add_edge(*rs)
        # They are not cliques at all but the term is used in literature.
        cliques = list(connected_components(g))
        cliques.sort(key=score)
        to_remove = max(100, max_removed // 4)
        to_remove = min(max_removed - removed, to_remove)
        if to_remove > 0:
            cliques_removed = cliques[-to_remove:]
        else:
            cliques_removed = []
        size = sum(len(c) for c in cliques_removed)
        if size:
            logging.info(
                f"[prune] pruning {len(cliques_removed)} cliques of {size} relations"
            )
        for c in cliques_removed:
            for ridx in c:
                prune(ridx)
        removed += len(cliques_removed)
        if not singles and not size:
            break

    assert len(rels) == len(rawrels)

    cols = set()
    pruned: list[tuple[int, npt.NDArray]] = [r for r in rels if r is not None]
    for rx, rps in pruned:
        cols.update([_p for _p in rps])
    logging.info(
        f"[prune] After pruning: {len(pruned)} relations with {len(cols)} primes"
    )

    if datadir is not None:
        with open(datadir / "relations.pruned", "w") as wp:
            for rx, row in pruned:
                line = " ".join(f"{l}" for l in sorted(row))
                print(rx, line, file=wp)

    return pruned


def filter_gf2(N: int, rels, datadir: pathlib.Path | None):
    t0 = time.time()
    D = 2**250
    dense_limit = 100

    stats = {}
    for ridx, (rx, rfacs) in enumerate(rels):
        for p in rfacs:
            stats.setdefault(p, set()).add(ridx)

    def addstat(ridx, r: RelFac):
        for p in r[1]:
            stats.setdefault(p, set()).add(ridx)

    def delstat(ridx, r: RelFac):
        for p in r[1]:
            stats[p].remove(ridx)
            if not stats[p]:
                stats.pop(p)

    def pivot(piv: RelFac, r: RelFac, p):
        y = numpy.setxor1d(piv[1], r[1])
        z = numpy.intersect1d(piv[1], r[1])
        x = piv[0] * r[0] * pow(algebra.product([int(_z) for _z in z]), -1, N) % N
        if DEBUG_CHECK_RELFAC:
            assert x * x % N == algebra.product(list(map(int, y))) % N
        return x, y

    def weight(row: RelFac):
        return len(row[1])

    excess = len(rels) - len(stats)
    logging.info(f"{len(stats)} primes appear in relations")
    logging.info(f"{excess} relations can be removed")

    # prime p = product(l^e)

    Ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    Ds += [25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    t = time.time()
    removed = 0
    for d in Ds:
        remaining = [_r for _r in rels if _r is not None]
        avgw = sum(weight(r) for r in remaining) / len(remaining)
        nc, nr = len(stats), len(remaining)
        assert nr > nc
        logging.info(
            f"Starting {d}-merge: {nc} columns {nr} rows excess={nr - nc} weight={avgw:.3f} elapsed={time.time() - t:.1f}s"
        )

        if d > nc // 3:
            # Matrix is too small
            break

        # Modulo p^k we have probabikity 1/p of missing a generator
        # for each excess relation
        MIN_EXCESS = 64 + D.bit_length()
        while True:
            # d-merges
            md = [k for k in stats if len(stats[k]) <= d]
            if not md:
                break
            logging.debug(f"{len(md)} {d}-merges candidates {min(md)}..{max(md)}")
            merged = 0
            for p in md:
                rs = stats.get(p)
                if not rs or len(rs) > d:
                    # prime already eliminated or weight has grown
                    continue
                # Pivot has fewest coefficients and pivot value is ±1
                assert all(p in rels[ridx][1] for ridx in stats[p])
                rs = sorted(rs, key=lambda ridx: weight(rels[ridx]))
                pividx = rs[0]
                piv = rels[pividx]
                for ridx in rs[1:]:
                    rp = pivot(piv, rels[ridx], p)
                    delstat(ridx, rels[ridx])
                    addstat(ridx, rp)
                    # If relation becomes empty, remove it
                    rels[ridx] = rp if len(rp) > 0 else None
                # Remove and save pivot
                delstat(pividx, piv)
                rels[pividx] = None
                removed += 1
                assert p not in stats
                merged += 1

            if not merged:
                break
            logging.debug(f"{merged} pivots done")

        remaining = [_r for _r in rels if _r is not None]
        nr, nc = len(remaining), len(stats)
        avgw = sum(len(r) for _x, r in remaining) / nr

        def score_sparse(rel, stats):
            t = max(2 * d, nr // 10)
            return sum(1 for l in rel[1] if l in stats and len(stats[l]) < t)

        stop = avgw > dense_limit
        # Remove most annoying relations
        excess = nr - nc
        if stop:
            break
        if excess > MIN_EXCESS:
            to_remove = (excess - MIN_EXCESS) // (len(Ds) // 2)
            if d < 10:
                # Still actively merging
                to_remove = 0
            if to_remove:
                scores = []
                for ridx, r in enumerate(rels):
                    if r is None:
                        continue
                    scores.append((score_sparse(r, stats), ridx))
                scores.sort()
                worst = scores[-to_remove:]
                logging.debug(
                    f"Worst rows ({len(worst)}) have score {worst[0][0]:.3f}..{worst[-1][0]:.3f}"
                )
                for _, ridx in worst:
                    # Not a pivot, no need to save.
                    delstat(ridx, rels[ridx])
                    rels[ridx] = None

    # For the last step, we just want to minimize sparse weight
    # We ignore dense columns when scoring
    nr = len([_r for _r in rels if _r is not None])
    dense = set([p for p, _rels in stats.items() if len(_rels) > nr // 3])
    logging.debug(f"Ignoring {len(dense)} dense columns to eliminate worst rows")

    def score_final(r):
        return sum(1 for p in r[1] if p not in dense)

    # Deduplicate before final step
    dedup = set()
    duplicates = 0
    for ridx, r in enumerate(rels):
        if r is None:
            continue
        if r[0] in dedup:
            duplicates += 1
            rels[ridx] = None
        dedup.add(r[0])
    if duplicates:
        logging.warn(f"Found {duplicates} duplicate relations after filtering")

    excess -= duplicates
    if excess > MIN_EXCESS:
        # scores = [(len(r), ridx) for ridx, r in enumerate(rels) if r is not None]
        scores = [
            (score_final(r), ridx) for ridx, r in enumerate(rels) if r is not None
        ]
        scores.sort()
        to_remove = excess - MIN_EXCESS
        worst = scores[-to_remove:]
        logging.info(
            f"Worst rows ({len(worst)}) have score {worst[0][0]:.3f}..{worst[-1][0]:.3f}"
        )
        for _, ridx in worst:
            # Not a pivot, no need to save.
            delstat(ridx, rels[ridx])
            rels[ridx] = None

    rels = [_r for _r in rels if _r is not None]
    nr, nc = len(rels), len(stats)
    avgw = sum(len(r) for rx, r in rels) / len(rels)
    dt = time.time() - t0
    logging.info(
        f"Final: {nc} columns {nr} rows excess={nr - nc} weight={avgw:.3f} elapsed={dt:.1f}s"
    )

    if datadir is not None:
        with open(datadir / "relations.filtered", "w") as w:
            for rx, r in rels:
                if DEBUG_CHECK_RELFAC:
                    assert rx * rx % N == algebra.product(list(map(int, r))) % N
                line = " ".join(str(l) for l in sorted(r))
                w.write(f"{rx} {line}")
                w.write("\n")
            logging.info(f"{len(rels)} relations written to {w.name}")

    return rels
