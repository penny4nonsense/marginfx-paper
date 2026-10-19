#!/usr/bin/env python3
"""Check the geometric step of the jump-set bound, in d=2 where H^1 is length
and everything can be computed exactly.

Verifies, over random depth-D axis-aligned tree partitions of [0,1]^2:

  (a) sum over internal nodes of face length  <=  2^D - 1
  (b) every adjacent leaf pair's shared boundary lies inside the split face of
      their lowest common ancestor                       [the claim in 6.1.1(2)]
  (c) the jump set's total length <= sum of internal-node face lengths
  (d) |Dg|(Omega) <= 2 ||g||_inf * (2^D - 1) * A         [the bound in 4, Step 1]
"""
import random

random.seed(20261019)
D = 4
TRIALS = 4000
EPS = 1e-12


class Node:
    __slots__ = ("cell", "dim", "thr", "lo", "hi", "val", "parent")

    def __init__(self, cell, parent=None):
        self.cell = cell          # (x0, x1, y0, y1)
        self.dim = None
        self.thr = None
        self.lo = self.hi = None
        self.val = None
        self.parent = parent


def build(cell, depth, parent=None):
    nd = Node(cell, parent)
    x0, x1, y0, y1 = cell
    if depth == 0 or random.random() < 0.25:
        nd.val = random.uniform(-1, 1)
        return nd
    dim = random.randint(0, 1)
    if dim == 0:
        t = random.uniform(x0 + 0.05 * (x1 - x0), x1 - 0.05 * (x1 - x0))
        nd.dim, nd.thr = 0, t
        nd.lo = build((x0, t, y0, y1), depth - 1, nd)
        nd.hi = build((t, x1, y0, y1), depth - 1, nd)
    else:
        t = random.uniform(y0 + 0.05 * (y1 - y0), y1 - 0.05 * (y1 - y0))
        nd.dim, nd.thr = 1, t
        nd.lo = build((x0, x1, y0, t), depth - 1, nd)
        nd.hi = build((x0, x1, t, y1), depth - 1, nd)
    return nd


def walk(nd, internals, leaves):
    if nd.lo is None:
        leaves.append(nd)
    else:
        internals.append(nd)
        walk(nd.lo, internals, leaves)
        walk(nd.hi, internals, leaves)


def face_len(nd):
    x0, x1, y0, y1 = nd.cell
    return (y1 - y0) if nd.dim == 0 else (x1 - x0)


def ancestors(nd):
    out = []
    while nd is not None:
        out.append(nd)
        nd = nd.parent
    return out


def lca(a, b):
    sa = set(id(n) for n in ancestors(a))
    for n in ancestors(b):
        if id(n) in sa:
            return n
    return None


def shared(a, b):
    """Shared boundary of two axis-aligned rects: (orientation, coord, lo, hi)."""
    ax0, ax1, ay0, ay1 = a.cell
    bx0, bx1, by0, by1 = b.cell
    # vertical contact
    for ca, cb in ((ax1, bx0), (bx1, ax0)):
        if abs(ca - cb) < EPS:
            lo, hi = max(ay0, by0), min(ay1, by1)
            if hi - lo > EPS:
                return ("v", ca, lo, hi)
    # horizontal contact
    for ca, cb in ((ay1, by0), (by1, ay0)):
        if abs(ca - cb) < EPS:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            if hi - lo > EPS:
                return ("h", ca, lo, hi)
    return None


fail_a = fail_b = fail_c = fail_d = 0
max_ratio_c = 0.0
max_ratio_d = 0.0

for _ in range(TRIALS):
    root = build((0.0, 1.0, 0.0, 1.0), D)
    internals, leaves = [], []
    walk(root, internals, leaves)

    face_sum = sum(face_len(n) for n in internals)
    if len(internals) > 2 ** D - 1 or face_sum > (2 ** D - 1) + EPS:
        fail_a += 1

    jump_len = 0.0
    tv = 0.0
    sup = max(abs(l.val) for l in leaves)
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            s = shared(leaves[i], leaves[j])
            if s is None:
                continue
            seg = s[3] - s[2]
            jump_len += seg
            tv += abs(leaves[i].val - leaves[j].val) * seg

            # (b) does this segment lie in the LCA's split face?
            anc = lca(leaves[i], leaves[j])
            if anc is None or anc.dim is None:
                fail_b += 1
                continue
            ox0, ox1, oy0, oy1 = anc.cell
            if anc.dim == 0:
                ok = (s[0] == "v" and abs(s[1] - anc.thr) < 1e-9
                      and s[2] >= oy0 - 1e-9 and s[3] <= oy1 + 1e-9)
            else:
                ok = (s[0] == "h" and abs(s[1] - anc.thr) < 1e-9
                      and s[2] >= ox0 - 1e-9 and s[3] <= ox1 + 1e-9)
            if not ok:
                fail_b += 1

    if jump_len > face_sum + 1e-9:
        fail_c += 1
    if face_sum > 0:
        max_ratio_c = max(max_ratio_c, jump_len / face_sum)

    bound_d = 2 * sup * (2 ** D - 1) * 1.0
    if tv > bound_d + 1e-9:
        fail_d += 1
    if bound_d > 0:
        max_ratio_d = max(max_ratio_d, tv / bound_d)

print("trials: %d, depth D = %d, 2^D - 1 = %d" % (TRIALS, D, 2 ** D - 1))
print()
print("(a) internal-node face sum <= 2^D - 1        : %s" % ("OK" if not fail_a else "%d FAIL" % fail_a))
print("(b) shared boundary inside LCA's split face  : %s" % ("OK" if not fail_b else "%d FAIL" % fail_b))
print("(c) jump-set length <= face sum              : %s  (worst ratio %.4f)"
      % ("OK" if not fail_c else "%d FAIL" % fail_c, max_ratio_c))
print("(d) |Dg| <= 2||g||_inf (2^D-1) A             : %s  (worst ratio %.4f)"
      % ("OK" if not fail_d else "%d FAIL" % fail_d, max_ratio_d))
