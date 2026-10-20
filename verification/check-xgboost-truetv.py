#!/usr/bin/env python3
"""Measure |D_1 f_hat|(Omega) directly, instead of the subadditive upper bound.

Everything so far has measured sum_m nu*||h_m||_inf, which bounds the total
variation of the ensemble by summing each tree's variation separately. That
discards cancellation: jumps from different trees can partly undo one another,
so the true TV of the sum can be far smaller than the sum of TVs, and can stay
bounded while the bound diverges.

The true quantity is computed by one-dimensional sections, which is exact for
this purpose:

    |D_1 g|(Omega) = integral over x_{-1} of TV_1( g(x_{-1}, .) ) dx_{-1}

estimated by Monte Carlo over x_{-1} and a fine grid in x_1. For piecewise
constant g a fine enough grid captures every jump.

Reported alongside the bound, at the risk-minimising stopping round, so the
gap between them is visible.
"""
import json
import math
import numpy as np
import xgboost as xgb

NU, MMAX, NTEST = 0.05, 6000, 20000
KSEC, GRID, LO, HI = 400, 6000, -3.0, 3.0      # sections, grid points, x1 range


def targets(X):
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return {
        "linear 2x1+3x2": 2 * x1 + 3 * x2,
        "radial 5exp(-|x|^2/2)": 5 * np.exp(-(x1**2 + x2**2 + x3**2 + x4**2) / 2),
    }


def true_tv_dim1(bst, rng, nbatch=KSEC):
    """Monte Carlo estimate of |D_1 f_hat| over the box, via 1-d sections."""
    xs = np.linspace(LO, HI, GRID)
    rest = rng.standard_normal((nbatch, 3))
    rest = np.clip(rest, LO, HI)
    tot = 0.0
    for k in range(nbatch):
        P = np.empty((GRID, 4))
        P[:, 0] = xs
        P[:, 1:] = rest[k]
        v = bst.predict(xgb.DMatrix(P), output_margin=True)
        tot += float(np.abs(np.diff(v)).sum())          # 1-d TV along x1
    return tot / nbatch * (HI - LO) ** 3                 # x volume of x_{-1} box


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
        h_sup[m] = np.max(np.abs(
            bst.predict(dtr, iteration_range=(m, m + 1), output_margin=True) - base)) / NU
        Fte += bst.predict(dte, iteration_range=(m, m + 1), output_margin=True) - base
        risk[m] = float(np.mean((Fte - ft) ** 2))
    partial = np.cumsum(NU * h_sup)
    mstar = int(np.argmin(risk)) + 1

    # refit to exactly M* so predict() uses the stopped ensemble
    bs2 = xgb.train({"max_depth": 4, "eta": NU, "objective": "reg:squarederror",
                     "min_child_weight": 1, "lambda": 1.0, "seed": 0},
                    dtr, num_boost_round=mstar)
    tv = true_tv_dim1(bs2, np.random.default_rng(99))
    return mstar, partial[mstar - 1], tv


# true TV of the target itself, for reference
def target_tv(key):
    rng = np.random.default_rng(99)
    xs = np.linspace(LO, HI, GRID)
    rest = np.clip(rng.standard_normal((KSEC, 3)), LO, HI)
    tot = 0.0
    for k in range(KSEC):
        P = np.empty((GRID, 4)); P[:, 0] = xs; P[:, 1:] = rest[k]
        tot += float(np.abs(np.diff(targets(P)[key])).sum())
    return tot / KSEC * (HI - LO) ** 3


NS = (500, 2000, 8000)
print("%-24s %-7s %-7s %-14s %-14s %s"
      % ("target", "n", "M*", "bound sum", "TRUE |D_1 f|", "bound/true"))
print("-" * 82)
for key in targets(np.zeros((1, 4))):
    tvs, bnds = [], []
    for n in NS:
        mstar, bnd, tv = run(key, n)
        tvs.append(tv); bnds.append(bnd)
        print("%-24s %-7d %-7d %-14.3f %-14.3f %.1fx"
              % (key if n == NS[0] else "", n, mstar, bnd, tv, bnd / tv))
    g_b = math.log(bnds[-1] / bnds[0]) / math.log(NS[-1] / NS[0])
    g_t = math.log(tvs[-1] / tvs[0]) / math.log(NS[-1] / NS[0])
    print("%-24s %-7s target |D_1 f| = %.3f   bound ~ n^%.2f   TRUE ~ n^%.2f"
          % ("", "", target_tv(key), g_b, g_t))
    print()
