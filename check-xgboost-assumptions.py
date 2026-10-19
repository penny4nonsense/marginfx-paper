#!/usr/bin/env python3
"""Check assumptions (A2) and (A3) of the draft XGBoost BV lemma against what
XGBoost actually does, on the paper's own linear regression DGP and settings.

(A2) every leaf holds at least kappa*n observations, kappa > 0
(A3) edge condition ||h_m||_{2,n} >= gamma ||u_m||_{2,n} for all m <= M

Also tests Step 2 of the lemma directly:
     ||h_m||_inf <= kappa^{-1/2} ||u_m||_{2,n}

and reports whether sum_m nu*||h_m||_inf -- the quantity the whole bound turns
on -- is in fact bounded, and whether it grows with n.

Paper settings: max_depth 4, learning_rate 0.05, 200 rounds, reg_lambda
default 1, min_child_weight default 1.
"""
import numpy as np
import xgboost as xgb

RNG = np.random.default_rng(20261019)
NU = 0.05
DEPTH = 4
ROUNDS = 200


def make_data(n, d=4):
    X = RNG.standard_normal((n, d))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + RNG.standard_normal(n)
    return X, y


def fit(X, y, min_child_weight=1):
    dm = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": DEPTH,
        "eta": NU,
        "objective": "reg:squarederror",
        "min_child_weight": min_child_weight,
        "lambda": 1.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": 0,
    }
    return xgb.train(params, dm, num_boost_round=ROUNDS), dm


def analyse(n, min_child_weight=1, label=""):
    X, y = make_data(n)
    bst, dm = fit(X, y, min_child_weight)

    # staged margins F_m, m = 0..ROUNDS.
    # NOTE: iteration_range=(0,0) does NOT mean "no trees" -- it returns the
    # full model -- so F_0 must be built from base_score by hand. Getting this
    # wrong injects a spurious enormous h_0 and makes Step 2 appear to fail.
    import json
    cfg = json.loads(bst.save_config())
    base = float(np.asarray(
        json.loads(cfg["learner"]["learner_model_param"]["base_score"])
        if cfg["learner"]["learner_model_param"]["base_score"].startswith("[")
        else cfg["learner"]["learner_model_param"]["base_score"]).ravel()[0])
    F = np.stack(
        [np.full(len(y), base)] +
        [bst.predict(dm, iteration_range=(0, m), output_margin=True)
         for m in range(1, ROUNDS + 1)])
    U = y[None, :] - F                      # residuals u_m
    Hs = (F[1:] - F[:-1]) / NU              # nu*h_m = F_{m+1} - F_m

    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    u_rms = np.array([rms(U[m]) for m in range(ROUNDS)])
    h_rms = np.array([rms(Hs[m]) for m in range(ROUNDS)])
    h_sup = np.array([float(np.max(np.abs(Hs[m]))) for m in range(ROUNDS)])

    # ---- (A2): leaf sizes.  For squared error, Cover == count in leaf.
    df = bst.trees_to_dataframe()
    leaves = df[df.Feature == "Leaf"]
    cover = leaves.Cover.to_numpy(dtype=float)
    per_tree_min = leaves.groupby("Tree").Cover.min().to_numpy(dtype=float)
    kappa_hat = float(per_tree_min.min()) / n       # worst leaf over all rounds

    # ---- (A3): edge ratios
    ratio = h_rms / np.maximum(u_rms, 1e-300)
    gamma_hat = float(ratio.min())

    # ---- Step 2 as stated, using the realised kappa
    if kappa_hat > 0:
        step2_lhs = h_sup
        step2_rhs = u_rms / np.sqrt(kappa_hat)
        step2_ok = bool(np.all(step2_lhs <= step2_rhs + 1e-9))
        step2_worst = float(np.max(step2_lhs / np.maximum(step2_rhs, 1e-300)))
    else:
        step2_ok, step2_worst = False, float("inf")

    total = float(np.sum(NU * h_sup))               # the quantity that matters
    bound = 2.0 / max(gamma_hat, 1e-12) ** 2 * rms(y)

    print("  %-26s n=%-6d" % (label, n))
    print("     (A2)  min leaf count over all %d trees : %d   (kappa_hat = %.2e)"
          % (ROUNDS, int(per_tree_min.min()), kappa_hat))
    print("           leaves with <10 obs              : %d of %d  (%.1f%%)"
          % (int((cover < 10).sum()), len(cover), 100.0 * (cover < 10).mean()))
    print("           median leaf count                : %.0f  (= %.3f n)"
          % (float(np.median(cover)), float(np.median(cover)) / n))
    print("     (A3)  min edge ratio over rounds       : %.4f" % gamma_hat)
    print("           edge ratio at rounds 1/50/100/200: %.4f %.4f %.4f %.4f"
          % (ratio[0], ratio[49], ratio[99], ratio[-1]))
    print("     Step2 ||h||_inf <= kappa^-1/2 ||u||    : %s  (worst ratio %.3f)"
          % ("HOLDS" if step2_ok else "FAILS", step2_worst))
    print("     sum nu*||h_m||_inf                     : %.4f"
          % total)
    print("     claimed bound 2 gamma^-2 ||Y||         : %.4f  %s"
          % (bound, "ok" if total <= bound else "EXCEEDED"))
    print()
    return dict(n=n, kappa=kappa_hat, gamma=gamma_hat, total=total,
                step2=step2_ok, ratio=ratio)


print("=" * 74)
print("XGBoost as the paper configures it (min_child_weight = 1, the default)")
print("=" * 74)
res_default = [analyse(n, 1, "default") for n in (250, 1000, 5000)]

print("=" * 74)
print("With a minimum leaf fraction imposed: min_child_weight = n/32")
print("=" * 74)
res_mcw = [analyse(n, max(1, n // 32), "min_child_weight=n/32") for n in (250, 1000, 5000)]

print("=" * 74)
print("SUMMARY")
print("=" * 74)
print("%-24s %-8s %-12s %-12s %-14s %s"
      % ("config", "n", "kappa_hat", "gamma_hat", "sum nu||h||inf", "Step 2"))
for tag, rs in (("default mcw=1", res_default), ("mcw = n/32", res_mcw)):
    for r in rs:
        print("%-24s %-8d %-12.2e %-12.4f %-14.4f %s"
              % (tag, r["n"], r["kappa"], r["gamma"], r["total"],
                 "holds" if r["step2"] else "FAILS"))
