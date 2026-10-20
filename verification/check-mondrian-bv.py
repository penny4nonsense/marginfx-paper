#!/usr/bin/env python3
"""Test the central claim of notes-mondrian-bv-bound.md:

    sup_n E[ |D T|(Omega) ] < infinity   for a SINGLE Mondrian tree
    at the minimax budget lambda_n ~ n^{1/(d+2)}.

If true, the existing Step 2 of the proof -- second moment, cross terms,
independence, all three of which the referee found wrong -- is unnecessary,
because Step 1 already bounds the forest by the average of its trees. If false,
the proposed route fails and Theorem S3 needs withdrawing.

Also decomposes the tree's variation into the signal and noise parts the note
predicts, by fitting the same partition twice: once to f(X) and once to the
noise Y - f(X). And checks the forest, to confirm averaging does not make
things worse.

Mondrian process on [0,1]^d (Roy & Teh): in a cell with side lengths s, draw
E ~ Exp(sum s); if the budget is spent, stop; else split dimension j with
probability proportional to s_j at a uniform point, and recurse.
"""
import math
import sys
import numpy as np

RNG = np.random.default_rng(20261019)


class Node:
    __slots__ = ("lo", "hi", "dim", "thr", "left", "right", "val")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.dim = self.thr = None
        self.left = self.right = None
        self.val = 0.0


def mondrian(lo, hi, budget, rng):
    nd = Node(lo, hi)
    side = hi - lo
    lin = float(side.sum())
    if lin <= 0:
        return nd
    E = rng.exponential(1.0 / lin)
    if E >= budget:
        return nd
    j = int(rng.choice(len(side), p=side / lin))
    cut = float(rng.uniform(lo[j], hi[j]))
    nd.dim, nd.thr = j, cut
    lo2, hi2 = lo.copy(), hi.copy()
    hi2[j] = cut
    nd.left = mondrian(lo, hi2, budget - E, rng)
    lo2 = lo.copy(); lo2[j] = cut
    nd.right = mondrian(lo2, hi, budget - E, rng)
    return nd


def leaves(nd, out):
    if nd.left is None:
        out.append(nd)
    else:
        leaves(nd.left, out); leaves(nd.right, out)


def assign(nd, X, idx, buckets):
    if nd.left is None:
        buckets.append((nd, idx))
        return
    m = X[idx, nd.dim] <= nd.thr
    assign(nd.left, X, idx[m], buckets)
    assign(nd.right, X, idx[~m], buckets)


def fit(nd, X, y):
    buckets = []
    assign(nd, X, np.arange(len(X)), buckets)
    for leaf, idx in buckets:
        leaf.val = float(y[idx].mean()) if len(idx) else 0.0
    return [len(idx) for _, idx in buckets]


def predict(nd, P):
    out = np.empty(len(P))
    stack = [(nd, np.arange(len(P)))]
    while stack:
        node, idx = stack.pop()
        if node.left is None:
            out[idx] = node.val
            continue
        m = P[idx, node.dim] <= node.thr
        stack.append((node.left, idx[m]))
        stack.append((node.right, idx[~m]))
    return out


def tv_dim1(predict_fn, d, rng, ksec=200, grid=4000):
    """|D_1 g|(Omega) over [0,1]^d by 1-d sections; exact for piecewise const."""
    xs = np.linspace(0.0, 1.0, grid)
    rest = rng.random((ksec, d - 1))
    tot = 0.0
    for k in range(ksec):
        P = np.empty((grid, d))
        P[:, 0] = xs
        P[:, 1:] = rest[k]
        v = predict_fn(P)
        tot += float(np.abs(np.diff(v)).sum())
    return tot / ksec          # volume of the x_{-1} box is 1


def target(X):
    return 2.0 * X[:, 0] + 3.0 * X[:, 1]


def experiment(n, d=4, B=100, reps=6):
    lam = n ** (1.0 / (d + 2))
    lo0, hi0 = np.zeros(d), np.ones(d)
    tv_tree, tv_sig, tv_noi, tv_forest, nleaf = [], [], [], [], []
    for r in range(reps):
        rng = np.random.default_rng(1000 * r + n)
        X = rng.random((n, d))
        f = target(X)
        y = f + rng.standard_normal(n)

        # one tree, fitted three ways on the SAME partition
        t = mondrian(lo0, hi0, lam, rng)
        counts = fit(t, X, y)
        nleaf.append(len(counts))
        tv_tree.append(tv_dim1(lambda P: predict(t, P), d, rng))
        fit(t, X, f)
        tv_sig.append(tv_dim1(lambda P: predict(t, P), d, rng))
        fit(t, X, y - f)
        tv_noi.append(tv_dim1(lambda P: predict(t, P), d, rng))

        # forest
        trees = [mondrian(lo0, hi0, lam, rng) for _ in range(B)]
        for tt in trees:
            fit(tt, X, y)
        tv_forest.append(tv_dim1(
            lambda P: np.mean([predict(tt, P) for tt in trees], axis=0), d, rng))
    m = lambda a: float(np.mean(a))
    return lam, m(nleaf), m(tv_tree), m(tv_sig), m(tv_noi), m(tv_forest)


D = 4
print("Mondrian forests, d=%d, budget lambda_n = n^(1/%d), target 2x1+3x2" % (D, D + 2))
print("true |D_1 f|(Omega) = 2.000   (partial_1 f = 2 on the unit cube)")
print()
print("%-8s %-8s %-8s %-11s %-11s %-11s %s"
      % ("n", "lambda", "leaves", "TV tree", "  signal", "  noise", "TV forest(B=100)"))
rows = []
for n in (500, 2000, 8000, 32000):
    lam, nl, tv, ts, tn, tf = experiment(n, D)
    rows.append((n, tv, ts, tn, tf))
    print("%-8d %-8.2f %-8.0f %-11.4f %-11.4f %-11.4f %.4f"
          % (n, lam, nl, tv, ts, tn, tf))

print()
g = lambda a, b, n0, n1: math.log(b / a) / math.log(n1 / n0)
n0, tv0, ts0, tn0, tf0 = rows[0]
n1, tv1, ts1, tn1, tf1 = rows[-1]
print("growth over n = %d -> %d:" % (n0, n1))
print("   single tree TV : n^%+.2f      %s" % (g(tv0, tv1, n0, n1),
      "BOUNDED -> the per-tree route works" if abs(g(tv0, tv1, n0, n1)) < 0.08
      else "GROWS -> the per-tree route fails"))
print("   signal part    : n^%+.2f" % g(ts0, ts1, n0, n1))
print("   noise part     : n^%+.2f" % g(tn0, tn1, n0, n1))
print("   forest TV      : n^%+.2f" % g(tf0, tf1, n0, n1))
