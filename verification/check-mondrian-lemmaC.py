#!/usr/bin/env python3
"""Check Lemma C of draft-mondrian-per-tree-bound.md.

The lemma reduces to: for T_1..T_d iid Gamma(2,1), W = prod T_j, and
mu = M*W with M >= 1,

    E[ min(sqrt2 * mu^{-1/2}, mu) / T_1 ]  <=  C_d * M^{-1/2}

with C_d = sqrt(2*pi)*(sqrt(pi)/2)^{d-1} + pi^{(2+d)/6}, coming from a bulk
term (mu >= 1, Gamma moments) and a tail term (mu < 1, Holder + Chernoff).

Also checks the two moment facts the proof uses:
    E[T^{-1/2}] = Gamma(3/2) = sqrt(pi)/2
    E[T^{-3/2}] = Gamma(1/2) = sqrt(pi)        <- finite, which is the crux
and the Chernoff bound P(W < 1/M) <= M^{-3/2} * pi^{d/2}.
"""
import math
import numpy as np

rng = np.random.default_rng(23)
REPS = 400000


def Cd(d):
    return math.sqrt(2 * math.pi) * (math.sqrt(math.pi) / 2) ** (d - 1) \
        + math.pi ** ((2 + d) / 6)


print("moment facts for T ~ Gamma(2,1)")
T = rng.gamma(2.0, 1.0, REPS)
print("  E[T^-1/2] : %.4f   vs Gamma(3/2) = %.4f" % (np.mean(T ** -0.5), math.gamma(1.5)))
print("  E[T^-3/2] : %.4f   vs Gamma(1/2) = %.4f   (finite: the crux)"
      % (np.mean(T ** -1.5), math.gamma(0.5)))
print()

print("Lemma C bound vs realised value")
print("%-4s %-10s %-14s %-14s %-10s %s"
      % ("d", "M", "E[psi/T1]", "C_d * M^-1/2", "ratio", "P(W<1/M) vs bound"))
for d in (2, 3, 4):
    for M in (10.0, 100.0, 1000.0, 10000.0):
        Tm = rng.gamma(2.0, 1.0, (REPS, d))
        W = Tm.prod(axis=1)
        mu = M * W
        psi = np.minimum(math.sqrt(2) * mu ** -0.5, mu)
        val = float(np.mean(psi / Tm[:, 0]))
        bnd = Cd(d) * M ** -0.5
        pw = float(np.mean(W < 1.0 / M))
        pwb = M ** -1.5 * math.pi ** (d / 2)
        print("%-4d %-10.0f %-14.5f %-14.5f %-10.3f %.2e vs %.2e %s"
              % (d, M, val, bnd, val / bnd, pw, pwb, "ok" if pw <= pwb else "EXCEEDS"))
    print()
