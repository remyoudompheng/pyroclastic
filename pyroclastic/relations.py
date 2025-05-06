import logging
from networkx import Graph, connected_components


def prune(rels: list, B1=None):
    rels = rels.copy()
    SMALLP = B1 or (len(rels) // 10)
    pruned = []
    stats = {}
    smalls = set()
    for ridx, r in enumerate(rels):
        for p in r[1:]:
            if p < SMALLP:
                smalls.add(p)
                continue
            stats.setdefault(p, []).append(ridx)

    excess = len(rels) - len(stats) - len(smalls)
    logging.info(
        f"[prune] {len(stats)+len(smalls)} primes appear in {len(rels)} relations (excess={excess})"
    )

    def prune(ridx):
        r = rels[ridx]
        for p in r[1:]:
            if p < SMALLP:
                continue
            sp = stats[p]
            sp.remove(ridx)
        rels[ridx] = None

    while True:
        singles = 0
        for p, rs in stats.items():
            if len(rs) == 1:
                prune(rs[0])
                singles += 1
        if singles:
            logging.info(f"[prune] pruned {singles} singletons")
        else:
            break

    rels = [_r for _r in rels if _r is not None]
    for p in list(stats):
        if len(stats[p]) == 0:
            stats.pop(p)
    logging.info(f"[prune] {len(stats)+len(smalls)} primes in {len(rels)} relations")
    if len(rels) == 0:
        return rels, None
    return rels, len(rels) - len(stats) - len(smalls)


def prune2(rawrels: list, B1: int, pbase: int):
    """
    Variant with both singleton and clique removal.
    """
    rels = []
    for rel in rawrels:
        exps = {}
        for p in rel:
            p = abs(p)
            if p < B1:
                continue
            exps[p] = exps.get(p, 0) + 1
        rels.append(exps)

    # Prune relations in place: a removed relation is replaced by None.
    # We are only interested in coefficients ±1, exponent sign is ignored
    stats = {}
    for ridx, r in enumerate(rels):
        for p, v in r.items():
            if v > 1:
                stats[p] = None
                continue
            stats.setdefault(p, [])
            if stats[p] is None:
                continue
            stats[p].append(ridx)
            if len(stats[p]) > 20:
                stats[p] = None

    excess = len(rels) - len(stats) - pbase
    logging.info(f"[prune2] {len(stats) + pbase} primes appear in relations")

    removed = 0
    max_removed = (excess - 200) // 2

    def prune(ridx):
        r = rels[ridx]
        for p, v in r.items():
            if v == 1:
                sp = stats[p]
                if sp is not None:
                    sp.remove(ridx)
        rels[ridx] = None

    def score(clique):
        s = 0
        for ridx in clique:
            # Score is weight of relation
            # + bonus point is some primes have low weight.
            r = rels[ridx]
            s += len(r)
            for p in r:
                if (sp := stats[p]) is not None and len(sp) < 5:
                    s += 1
        return s

    while removed < max_removed:
        m1 = [p for p, rs in stats.items() if rs is not None and len(rs) == 1]
        singles = 0
        for p in m1:
            if stats[p]:
                prune(stats[p][0])
                singles += 1
        if singles:
            logging.info(f"[prune2] pruned {singles} singletons")

        m2 = [p for p, rs in stats.items() if rs is not None and len(rs) == 2]
        g = Graph()
        for p in m2:
            g.add_edge(*stats[p])
        # They are not cliques at all but the term is used in literature.
        cliques = list(connected_components(g))
        cliques.sort(key=score)
        to_remove = max(100, max_removed // 4)
        to_remove = min(max_removed - removed, to_remove)
        assert to_remove > 0
        cliques_removed = cliques[-to_remove:]
        size = sum(len(c) for c in cliques_removed)
        if size:
            logging.info(
                f"[prune2] pruning {len(cliques_removed)} cliques of {size} relations"
            )
        for c in cliques_removed:
            for ridx in c:
                prune(ridx)
        removed += len(cliques_removed)
        if not singles and not size:
            break

    assert len(rels) == len(rawrels)
    pruned = [r for r, _r in zip(rawrels, rels) if _r is not None]

    cols = set()
    rels = [r for r in rels if r is not None]
    for r in rels:
        cols.update(r)
    logging.info(
        f"[prune2] After pruning: {len(rels)} relations with {len(cols)+pbase} primes"
    )

    return pruned, len(rels) - len(cols) - pbase
