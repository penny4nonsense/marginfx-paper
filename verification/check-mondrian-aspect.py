#!/usr/bin/env python3
"""Check the aspect-ratio identity the signal bound needs.

Mourtada, Gaiffas & Scornet: for MP(lambda) on R^d the cell containing a fixed
point has extents E_j^- , E_j^+ that are 2d iid Exp(lambda), so each side
s_j = E_j^- + E_j^+ ~ Gamma(2, lambda), independent across j. Then

    E[s_m / s_1] = E[s_m] E[1/s_1] = (2/lambda) * lambda = 2       (m != 1)

since E[1/X] = lambda for X ~ Gamma(2, lambda). The signal bound needs
E[ diam(cell)/s_1(cell) ] <= sum_m E[s_m/s_1] = 2d - 1.

Checked two ways: the unbounded law directly, and empirically on the unit
cube, where truncation at the boundary shrinks sides and could inflate the
ratio.
"""
import numpy as np

rng = np.random.default_rng(11)


def unbounded(d, lam, reps=200000):
    E = rng.exponential(1.0 / lam, size=(reps, d, 2))
    s = E.sum(axis=2)
    return float(np.mean(s[:, 1] / s[:, 0])), float(np.mean(s.sum(1) / s[:, 0]))


def mondrian_cell(lo, hi, budget, x, rng):
    """Sides of the MP(budget) cell containing x, inside the box [lo,hi]."""
    lo, hi = lo.copy(), hi.copy()
    while True:
        side = hi - lo
        lin = float(side.sum())
        E = rng.exponential(1.0 / lin)
        if E >= budget:
            return hi - lo
        j = int(rng.choice(len(side), p=side / lin))
        cut = float(rng.uniform(lo[j], hi[j]))
        if x[j] <= cut:
            hi[j] = cut
        else:
            lo[j] = cut
        budget -= E


print("E[s_m/s_1] should be 2;  E[sum_m s_m/s_1] should be 2d-1")
print()
print("%-4s %-8s %-22s %-22s %s" % ("d", "lambda", "E[s2/s1] unbounded", "E[s2/s1] unit cube", "E[sum/s1] cube vs 2d-1"))
for d in (2, 3, 4):
    for lam in (4.0, 8.0, 16.0):
        u2, usum = unbounded(d, lam, 100000)
        reps = 4000
        acc2 = np.empty(reps); accs = np.empty(reps)
        lo, hi = np.zeros(d), np.ones(d)
        for r in range(reps):
            x = rng.random(d)
            s = mondrian_cell(lo, hi, lam, x, rng)
            acc2[r] = s[1] / s[0]
            accs[r] = s.sum() / s[0]
        print("%-4d %-8.0f %-22.3f %-22.3f %.3f  vs %d"
              % (d, lam, u2, float(acc2.mean()), float(accs.mean()), 2 * d - 1))
