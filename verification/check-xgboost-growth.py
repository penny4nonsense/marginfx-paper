#!/usr/bin/env python3
"""Does sum_{m<M} nu*||h_m||_inf -- the quantity the whole of section 4 turns
on -- stay bounded as M grows, or does it grow?

Section 4 claims uniform boundedness in M via the edge condition. The edge
condition does not hold uniformly (gamma decays with n), so the question is
whether the CONCLUSION is nonetheless true. Two candidate behaviours:

    bounded      -> section 4's conclusion is right, its route is wrong
    ~ sqrt(M)    -> only the energy/Cauchy-Schwarz bound is available, and
                    uniformity in M is false; the fixed-M framing is the
                    honest one

Also reports the edge ratio's decay in M, and the alternative route
    ||h_m||_inf <= kappa^{-1/2} ||h_m||_{2,n},   sum nu ||h_m||_{2,n}
                                      <= sqrt(nu M/(2-nu)) ||Y||_{2,n}
which needs no edge condition at all.
"""
import json
import numpy as np
import xgboost as xgb

RNG = np.random.default_rng(20261019)
NU, DEPTH = 0.05, 4
MMAX = 4000


def run(n, mcw, tag):
    X = RNG.standard_normal((n, 4))
    y = 2 * X[:, 0] + 3 * X[:, 1] + RNG.standard_normal(n)
    dm = xgb.DMatrix(X, label=y)
    bst = xgb.train({"max_depth": DEPTH, "eta": NU,
                     "objective": "reg:squarederror",
                     "min_child_weight": mcw, "lambda": 1.0, "seed": 0},
                    dm, num_boost_round=MMAX)
    cfg = json.loads(bst.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    base = float(np.asarray(json.loads(bs) if bs.startswith("[") else float(bs)).ravel()[0])

    F = np.stack([np.full(n, base)] +
                 [bst.predict(dm, iteration_range=(0, m), output_margin=True)
                  for m in range(1, MMAX + 1)])
    U = y[None, :] - F
    Hs = (F[1:] - F[:-1]) / NU
    rms = lambda a: np.sqrt(np.mean(a ** 2, axis=-1))
    h_sup = np.max(np.abs(Hs), axis=1)
    h_rms = rms(Hs)
    u_rms = rms(U[:MMAX])
    partial = np.cumsum(NU * h_sup)
    yr = float(rms(y[None, :])[0])

    print("  %s  n=%d  min_child_weight=%d   ||Y||_{2,n}=%.3f" % (tag, n, mcw, yr))
    print("     %-8s %-14s %-14s %-12s %-12s" %
          ("M", "sum nu|h|inf", "/sqrt(M)", "edge ratio", "energy bnd"))
    for M in (50, 100, 200, 500, 1000, 2000, 4000):
        energy = np.sqrt(NU * M / (2 - NU)) * yr          # sum nu||h||_{2,n} bound
        print("     %-8d %-14.4f %-14.4f %-12.5f %-12.2f"
              % (M, partial[M - 1], partial[M - 1] / np.sqrt(M),
                 h_rms[M - 1] / max(u_rms[M - 1], 1e-300), energy))
    # is the energy inequality itself respected?
    lhs = float(np.sum(NU * h_rms))
    rhs = float(np.sqrt(NU * MMAX / (2 - NU)) * yr)
    print("     energy route at M=%d: sum nu||h||_2n = %.3f  <=  %.3f  %s"
          % (MMAX, lhs, rhs, "ok" if lhs <= rhs else "EXCEEDED"))
    tel = float(np.sum(NU * (2 - NU) * h_rms ** 2))
    print("     telescoping check: sum nu(2-nu)||h||^2 = %.4f  <=  ||Y-base||^2 = %.4f  %s"
          % (tel, float(rms(U[0:1])[0]) ** 2,
             "ok" if tel <= float(rms(U[0:1])[0]) ** 2 + 1e-9 else "EXCEEDED"))
    print()


for n in (1000, 5000):
    run(n, max(1, n // 32), "mcw=n/32")
run(1000, 1, "default ")
