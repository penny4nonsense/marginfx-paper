#!/usr/bin/env python3
"""Check the closed form for the expected total interface area of a Mondrian
partition, which is the first gap listed in notes-mondrian-bv-bound.md section 4.

Claim (derived from the Mondrian recursion):

    E[ sum over internal nodes of H^{d-1}(F_k) ]  =  d * lambda * vol(C)

For the unit cube this is d*lambda, independent of the shape of the cuts.
Also checks E[#leaves] against the known (1+lambda)^d for calibration.
"""
import numpy as np

RNG = np.random.default_rng(4)


def mondrian_area(lo, hi, budget, rng):
    """Return (total interface area, number of leaves) for MP(budget, [lo,hi])."""
    side = hi - lo
    lin = float(side.sum())
    if lin <= 0:
        return 0.0, 1
    E = rng.exponential(1.0 / lin)
    if E >= budget:
        return 0.0, 1
    j = int(rng.choice(len(side), p=side / lin))
    cut = float(rng.uniform(lo[j], hi[j]))
    face = float(np.prod(np.delete(side, j)))      # H^{d-1} of the split face
    hi2 = hi.copy(); hi2[j] = cut
    lo2 = lo.copy(); lo2[j] = cut
    aL, nL = mondrian_area(lo, hi2, budget - E, rng)
    aR, nR = mondrian_area(lo2, hi, budget - E, rng)
    return face + aL + aR, nL + nR


print("E[total interface area] of MP(lambda) on the unit cube")
print("claim: d * lambda      (and E[#leaves] = (1+lambda)^d for calibration)")
print()
print("%-4s %-8s %-12s %-12s %-10s %-12s %s"
      % ("d", "lambda", "measured", "d*lambda", "ratio", "leaves", "(1+lam)^d"))
for d in (2, 3, 4):
    for lam in (1.0, 2.0, 4.0, 6.0):
        reps = 3000 if d <= 3 else 1500
        tot = np.empty(reps); nl = np.empty(reps)
        lo, hi = np.zeros(d), np.ones(d)
        for r in range(reps):
            tot[r], nl[r] = mondrian_area(lo, hi, lam, RNG)
        m = tot.mean(); se = tot.std(ddof=1) / np.sqrt(reps)
        print("%-4d %-8.1f %-12s %-12.3f %-10.3f %-12.1f %.1f"
              % (d, lam, "%.3f+-%.3f" % (m, se), d * lam, m / (d * lam),
                 nl.mean(), (1 + lam) ** d))
