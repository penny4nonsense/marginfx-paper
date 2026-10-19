#!/usr/bin/env python3
"""Can XGBoost settings make sum_m nu*||h_m||_inf bounded in M?

Section 4.4 found it growing like M^0.3 at the paper's settings. Three knobs
could plausibly attack the mechanism (late rounds spending their small energy
budget on spiky trees -- big on one small leaf):

  reg_lambda        shrinks leaf values hardest when the leaf is small:
                    |w_l| = |G_l|/(n_l+lambda) <= sqrt(n)||u||/(2 sqrt(lambda)),
                    maximised at n_l = lambda, and INDEPENDENT of leaf size
  min_child_weight  forbids the small leaves outright
  max_depth         fewer leaves, less room to be spiky

Reports the growth exponent of the partial sum in M. Exponent ~0 would mean
the lemma's conclusion is recoverable by configuration; positive for every
setting means it is structural.
"""
import json
import numpy as np
import xgboost as xgb

NU = 0.05
MMAX = 4000
GRID = [100, 200, 500, 1000, 2000, 4000]
N = 2000


def fit_and_measure(n, depth, mcw, lam, seed=0):
    rng = np.random.default_rng(20261019 + seed)
    X = rng.standard_normal((n, 4))
    y = 2 * X[:, 0] + 3 * X[:, 1] + rng.standard_normal(n)
    dm = xgb.DMatrix(X, label=y)
    bst = xgb.train({"max_depth": depth, "eta": NU,
                     "objective": "reg:squarederror",
                     "min_child_weight": mcw, "lambda": lam, "seed": 0},
                    dm, num_boost_round=MMAX)
    cfg = json.loads(bst.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    base = float(np.asarray(json.loads(bs) if bs.startswith("[") else float(bs)).ravel()[0])

    # per-tree contributions: O(M n)
    h_sup = np.empty(MMAX)
    for m in range(MMAX):
        t = bst.predict(dm, iteration_range=(m, m + 1), output_margin=True) - base
        h_sup[m] = np.max(np.abs(t)) / NU
    partial = np.cumsum(NU * h_sup)

    xs = np.log(np.array(GRID, dtype=float))
    ys = np.log(np.array([partial[M - 1] for M in GRID]))
    slope = float(np.polyfit(xs, ys, 1)[0])
    return partial, slope


def show(tag, n, depth, mcw, lam):
    partial, slope = fit_and_measure(n, depth, mcw, lam)
    print("%-34s %-7d %-9d %-9s %-9.3f %-9.3f %-9.3f M^%.2f"
          % (tag, depth, mcw, ("%g" % lam),
             partial[199], partial[999], partial[3999], slope))
    return slope


print("n = %d, eta = %g, rounds = %d, DGP = linear regression from the paper"
      % (N, NU, MMAX))
print()
print("%-34s %-7s %-9s %-9s %-9s %-9s %-9s %s"
      % ("setting", "depth", "min_chld", "lambda", "M=200", "M=1000", "M=4000", "growth"))
print("-" * 104)

results = {}
results["paper defaults"] = show("paper defaults", N, 4, 1, 1.0)
print()
for lam in (N // 100, N // 10, N, 10 * N):
    results["lambda=%g" % lam] = show("reg_lambda = %g" % lam, N, 4, 1, float(lam))
print()
for mcw in (N // 32, N // 16):
    results["mcw=%d" % mcw] = show("min_child_weight = %d" % mcw, N, 4, mcw, 1.0)
print()
for depth in (1, 2):
    results["depth=%d" % depth] = show("max_depth = %d" % depth, N, depth, 1, 1.0)
print()
results["combined"] = show("depth 2 + mcw n/16 + lam n", N, 2, N // 16, float(N))
results["combined2"] = show("depth 1 + mcw n/16 + lam 10n", N, 1, N // 16, float(10 * N))

print()
print("=" * 104)
print("growth exponents, sorted (0.00 would mean bounded):")
for k, v in sorted(results.items(), key=lambda kv: kv[1]):
    print("   %-34s M^%.3f" % (k, v))
