#!/usr/bin/env python3
"""Does early stopping rescue the lemma for targets that are HARD for trees?

The linear DGP is easy: boosting is within 0.4% of it by ~130 rounds, so M*
barely grows with n and the sum at M* is flat almost by default. That tells us
little about the general case. Here the same measurement is repeated on
progressively harder targets, including two from the paper and two that are
genuinely awkward for axis-aligned trees.

If the sum at M* stays flat as the target hardens, early stopping is a real
fix. If it grows with n once the target needs many rounds, it is not -- the
bound would then be uniform only where boosting converges quickly anyway.
"""
import json
import math
import numpy as np
import xgboost as xgb

NU = 0.05
MMAX = 6000
NTEST = 20000


def targets(X):
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return {
        "linear  2x1+3x2": 2 * x1 + 3 * x2,
        "interact +2x1x2": 2 * x1 + 3 * x2 + 2 * x1 * x2,
        "nonlin  2x1^2+3x2": 2 * x1 ** 2 + 3 * x2,
        "oscil   3sin(4x1)+3x2": 3 * np.sin(4 * x1) + 3 * x2,
        "radial  5exp(-|x|^2)": 5 * np.exp(-(x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2) / 2),
    }


def run(key, n, seed=0):
    rng = np.random.default_rng(20261019 + seed)
    X = rng.standard_normal((n, 4))
    y = targets(X)[key] + rng.standard_normal(n)
    Xt = rng.standard_normal((NTEST, 4))
    ft = targets(Xt)[key]
    dtr, dte = xgb.DMatrix(X, label=y), xgb.DMatrix(Xt)

    bst = xgb.train({"max_depth": 4, "eta": NU, "objective": "reg:squarederror",
                     "min_child_weight": 1, "lambda": 1.0, "seed": 0},
                    dtr, num_boost_round=MMAX)
    cfg = json.loads(bst.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    base = float(np.asarray(json.loads(bs) if bs.startswith("[") else float(bs)).ravel()[0])

    h_sup = np.empty(MMAX)
    Fte = np.full(NTEST, base)
    risk = np.empty(MMAX)
    for m in range(MMAX):
        tr = bst.predict(dtr, iteration_range=(m, m + 1), output_margin=True) - base
        h_sup[m] = np.max(np.abs(tr)) / NU
        Fte += bst.predict(dte, iteration_range=(m, m + 1), output_margin=True) - base
        risk[m] = float(np.mean((Fte - ft) ** 2))

    partial = np.cumsum(NU * h_sup)
    mstar = int(np.argmin(risk)) + 1
    return mstar, float(risk[mstar - 1]), float(partial[mstar - 1]), float(np.var(ft))


NS = (500, 2000, 8000)
print("%-24s %-6s %-8s %-10s %-12s %s"
      % ("target", "n", "M*", "risk/var", "sum at M*", ""))
print("-" * 76)
for key in targets(np.zeros((1, 4))):
    ms, ss = [], []
    for n in NS:
        mstar, r, s, v = run(key, n)
        ms.append(mstar); ss.append(s)
        print("%-24s %-6d %-8d %-10.4f %-12.3f" % (key if n == NS[0] else "", n, mstar, r / v, s))
    gm = math.log(ms[-1] / ms[0]) / math.log(NS[-1] / NS[0])
    gs = math.log(ss[-1] / ss[0]) / math.log(NS[-1] / NS[0])
    print("%-24s %-6s M* ~ n^%.2f   sum at M* ~ n^%.2f" % ("", "", gm, gs))
    print()
