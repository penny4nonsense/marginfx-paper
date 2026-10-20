#!/usr/bin/env python3
"""Check the geometric step of the jump-set bound used in
notes-xgboost-gradient-consistency.md, section 6.1.1, in any dimension.

Over random depth-D axis-aligned tree partitions of the unit cube in R^d,
verifies:

  (a) sum over internal nodes of face area  <=  (2^D - 1) * A
  (b) every adjacent leaf pair's shared boundary lies inside the split face of
      their lowest common ancestor                  [the claim in 6.1.1 step 2]
  (c) total internal interface area <= sum of internal-node face areas
                                                    [6.1.1 step 3, subadditivity]
  (d) |Dg|(Omega) <= 2 ||g||_inf (2^D - 1) A        [section 4, Step 1]

In d dimensions a split face is a (d-1)-rectangle and H^{d-1} is its area, so
everything below is computed exactly in rational-free floating point; the only
tolerances are for coordinate coincidence.

Usage:  python check-tree-faces.py [d_max] [D_max] [trials]
Default sweeps d = 2..4 and D = 2..5.
"""
import random
import sys
from itertools import combinations

EPS = 1e-9
SPLIT_MARGIN = 0.05       # keep thresholds strictly interior to the cell
LEAF_PROB = 0.25          # chance of stopping early, to vary tree shape


class Node:
    __slots__ = ("cell", "dim", "thr", "lo", "hi", "val", "parent")

    def __init__(self, cell, parent=None):
        self.cell = cell            # tuple of (lo, hi) per dimension
        self.dim = self.thr = None
        self.lo = self.hi = None
        self.val = None
        self.parent = parent


def build(cell, depth, rng, parent=None):
    nd = Node(cell, parent)
    if depth == 0 or rng.random() < LEAF_PROB:
        nd.val = rng.uniform(-1.0, 1.0)
        return nd
    d = len(cell)
    j = rng.randrange(d)
    lo, hi = cell[j]
    t = rng.uniform(lo + SPLIT_MARGIN * (hi - lo), hi - SPLIT_MARGIN * (hi - lo))
    nd.dim, nd.thr = j, t
    left = list(cell); left[j] = (lo, t)
    right = list(cell); right[j] = (t, hi)
    nd.lo = build(tuple(left), depth - 1, rng, nd)
    nd.hi = build(tuple(right), depth - 1, rng, nd)
    return nd


def walk(nd, internals, leaves):
    if nd.lo is None:
        leaves.append(nd)
    else:
        internals.append(nd)
        walk(nd.lo, internals, leaves)
        walk(nd.hi, internals, leaves)


def face_area(nd):
    """H^{d-1} of the split face at an internal node."""
    a = 1.0
    for i, (lo, hi) in enumerate(nd.cell):
        if i != nd.dim:
            a *= (hi - lo)
    return a


def max_face_area(d):
    """A: the largest possible face area in the unit cube."""
    return 1.0


def ancestors(nd):
    out = []
    while nd is not None:
        out.append(nd)
        nd = nd.parent
    return out


def lca(a, b):
    seen = {id(n) for n in ancestors(a)}
    for n in ancestors(b):
        if id(n) in seen:
            return n
    return None


def contact(a, b):
    """If leaves a, b share positive (d-1)-area, return (dim, coord, area,
    overlap) where overlap is the per-dimension intersection interval."""
    ca, cb = a.cell, b.cell
    d = len(ca)
    touch_dim = None
    coord = None
    for j in range(d):
        if abs(ca[j][1] - cb[j][0]) < EPS:
            if touch_dim is not None:
                return None            # contact in two dims => lower-dim set
            touch_dim, coord = j, ca[j][1]
        elif abs(cb[j][1] - ca[j][0]) < EPS:
            if touch_dim is not None:
                return None
            touch_dim, coord = j, cb[j][1]
    if touch_dim is None:
        return None
    area = 1.0
    overlap = []
    for i in range(d):
        if i == touch_dim:
            overlap.append(None)
            continue
        lo = max(ca[i][0], cb[i][0])
        hi = min(ca[i][1], cb[i][1])
        if hi - lo <= EPS:
            return None                # touching on an edge, not a face
        overlap.append((lo, hi))
        area *= (hi - lo)
    return touch_dim, coord, area, overlap


def run(d, D, trials, rng):
    A = max_face_area(d)
    fails = {"a": 0, "b": 0, "c": 0, "d": 0}
    worst_c = worst_d = 0.0
    pairs_checked = 0

    unit = tuple((0.0, 1.0) for _ in range(d))
    for _ in range(trials):
        root = build(unit, D, rng)
        internals, leaves = [], []
        walk(root, internals, leaves)

        face_sum = sum(face_area(n) for n in internals)
        if len(internals) > 2 ** D - 1 or face_sum > (2 ** D - 1) * A + EPS:
            fails["a"] += 1

        interface_area = 0.0
        tv = 0.0
        sup = max(abs(l.val) for l in leaves)
        for a, b in combinations(leaves, 2):
            c = contact(a, b)
            if c is None:
                continue
            j, coord, area, overlap = c
            pairs_checked += 1
            interface_area += area
            tv += abs(a.val - b.val) * area

            anc = lca(a, b)
            ok = anc is not None and anc.dim == j and abs(anc.thr - coord) < EPS
            if ok:
                for i in range(d):
                    if i == j or overlap[i] is None:
                        continue
                    lo, hi = overlap[i]
                    alo, ahi = anc.cell[i]
                    if lo < alo - EPS or hi > ahi + EPS:
                        ok = False
                        break
            if not ok:
                fails["b"] += 1

        if interface_area > face_sum + EPS:
            fails["c"] += 1
        if face_sum > 0:
            worst_c = max(worst_c, interface_area / face_sum)

        bd = 2 * sup * (2 ** D - 1) * A
        if tv > bd + EPS:
            fails["d"] += 1
        if bd > 0:
            worst_d = max(worst_d, tv / bd)

    return fails, worst_c, worst_d, pairs_checked


def eval_tree(nd, pt):
    while nd.lo is not None:
        nd = nd.lo if pt[nd.dim] <= nd.thr else nd.hi
    return nd.val


def sigma(t):
    import math
    if t >= 0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


def run_ensemble(d, D, M, trials, rng, nu=1.0, grid=8):
    """The actual section 6.1.1 claim: for F = nu * sum of M trees,

         |D sigma(F)|(Omega) <= M (2^D - 1) A       and <= (1/4)|DF|(Omega),

    computed exactly on the common refinement. Thresholds are drawn from a
    coarse discrete set so that faces from different trees COINCIDE often --
    this is the overlap case the bound is claimed to be indifferent to.
    """
    A = max_face_area(d)
    cap_fail = lip_fail = 0
    worst_cap = worst_lip = 0.0
    coincidences = 0

    cuts_allowed = [k / grid for k in range(1, grid)]
    unit = tuple((0.0, 1.0) for _ in range(d))

    for _ in range(trials):
        trees = []
        for _m in range(M):
            trees.append(build_discrete(unit, D, rng, cuts_allowed))

        # common refinement: union of all split coordinates per dimension
        cuts = [set([0.0, 1.0]) for _ in range(d)]
        per_dim_counts = [dict() for _ in range(d)]
        for t in trees:
            ints, _lv = [], []
            walk(t, ints, _lv)
            for nd in ints:
                cuts[nd.dim].add(nd.thr)
                per_dim_counts[nd.dim][nd.thr] = per_dim_counts[nd.dim].get(nd.thr, 0) + 1
        for j in range(d):
            coincidences += sum(1 for v in per_dim_counts[j].values() if v > 1)

        axes = [sorted(c) for c in cuts]
        sizes = [len(a) - 1 for a in axes]

        # evaluate F on every refinement cell
        import itertools
        vals = {}
        for idx in itertools.product(*[range(s) for s in sizes]):
            centre = tuple((axes[j][idx[j]] + axes[j][idx[j] + 1]) / 2.0 for j in range(d))
            vals[idx] = nu * sum(eval_tree(t, centre) for t in trees)

        tv_margin = 0.0
        tv_prob = 0.0
        for idx in vals:
            for j in range(d):
                nxt = list(idx); nxt[j] += 1
                nxt = tuple(nxt)
                if nxt not in vals:
                    continue
                area = 1.0
                for i in range(d):
                    if i != j:
                        area *= (axes[i][idx[i] + 1] - axes[i][idx[i]])
                tv_margin += abs(vals[nxt] - vals[idx]) * area
                tv_prob += abs(sigma(vals[nxt]) - sigma(vals[idx])) * area

        cap = M * (2 ** D - 1) * A
        if tv_prob > cap + EPS:
            cap_fail += 1
        worst_cap = max(worst_cap, tv_prob / cap)

        if tv_prob > 0.25 * tv_margin + EPS:
            lip_fail += 1
        if tv_margin > 0:
            worst_lip = max(worst_lip, tv_prob / (0.25 * tv_margin))

    return cap_fail, lip_fail, worst_cap, worst_lip, coincidences


def build_discrete(cell, depth, rng, allowed, parent=None):
    """Like build(), but thresholds come from a coarse shared grid, so splits
    in different trees land on the same hyperplane often."""
    nd = Node(cell, parent)
    if depth == 0 or rng.random() < LEAF_PROB:
        nd.val = rng.uniform(-3.0, 3.0)
        return nd
    d = len(cell)
    order = list(range(d))
    rng.shuffle(order)
    for j in order:
        lo, hi = cell[j]
        cand = [t for t in allowed if lo + EPS < t < hi - EPS]
        if cand:
            t = rng.choice(cand)
            nd.dim, nd.thr = j, t
            left = list(cell); left[j] = (lo, t)
            right = list(cell); right[j] = (t, hi)
            nd.lo = build_discrete(tuple(left), depth - 1, rng, allowed, nd)
            nd.hi = build_discrete(tuple(right), depth - 1, rng, allowed, nd)
            return nd
    nd.val = rng.uniform(-3.0, 3.0)
    return nd


def main():
    d_max = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    D_max = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    rng = random.Random(20261019)

    print("%-4s %-4s %-9s %-7s %-7s %-7s %-7s %-10s %s"
          % ("d", "D", "pairs", "(a)", "(b)", "(c)", "(d)", "worst (c)", "worst (d)"))
    total_fail = 0
    for d in range(2, d_max + 1):
        for D in range(2, D_max + 1):
            fails, wc, wd, pairs = run(d, D, trials, rng)
            total_fail += sum(fails.values())
            mark = lambda k: "OK" if fails[k] == 0 else "FAIL%d" % fails[k]
            print("%-4d %-4d %-9d %-7s %-7s %-7s %-7s %-10.4f %.4f"
                  % (d, D, pairs, mark("a"), mark("b"), mark("c"), mark("d"), wc, wd))
    print()
    print("trials per cell: %d" % trials)
    print("single-tree failures: %d" % total_fail)

    print()
    print("=== ensemble bound (section 6.1.1), common refinement computed exactly ===")
    print("%-4s %-4s %-4s %-9s %-10s %-12s %-12s %s"
          % ("d", "D", "M", "coincid.", "cap bound", "1/4-Lipschitz", "worst cap", "worst Lip"))
    ens_fail = 0
    for d in (2, 3):
        for D in (2, 3):
            for M in (2, 5, 10):
                t = max(60, trials // 20)
                cf, lf, wc, wl, co = run_ensemble(d, D, M, t, rng)
                ens_fail += cf + lf
                print("%-4d %-4d %-4d %-9d %-10s %-12s %-12.4f %.4f"
                      % (d, D, M, co,
                         "OK" if cf == 0 else "FAIL%d" % cf,
                         "OK" if lf == 0 else "FAIL%d" % lf, wc, wl))
    print()
    print("ensemble failures: %d" % ens_fail)
    print("TOTAL FAILURES: %d" % (total_fail + ens_fail))
    return 0 if (total_fail + ens_fail) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
