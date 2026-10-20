#!/usr/bin/env python3
"""Numerically check the two algebraic claims in Step 3 and Step 4 of the
draft XGBoost BV lemma:

  (i)  rho = sqrt(1 - nu(2-nu) gamma^2) < 1,  and  1 - rho >= nu gamma^2 / 2
  (ii) sum_m nu ||h_m|| <= 2 gamma^-2 ||Y||, i.e. the learning rate cancels

For (ii) we simulate the recursion in its tightest form, where the edge
condition binds exactly (||h_m|| = gamma ||u_m||) and the residual decreases by
exactly nu(2-nu)||h_m||^2, and compare the realised sum against the claimed
bound.
"""
import math

print("%-7s %-7s %-10s %-12s %-12s %-12s %s"
      % ("nu", "gamma", "rho", "1-rho", "nu*g^2/2", "nu/(1-rho)", "2/g^2"))
bad = 0
for nu in (1.0, 0.5, 0.2, 0.05, 0.01, 0.001):
    for gamma in (1.0, 0.7, 0.3, 0.1, 0.03):
        disc = 1 - nu * (2 - nu) * gamma ** 2
        if disc < 0:
            print("  nu=%.3g gamma=%.3g : discriminant negative (rho=0)" % (nu, gamma))
            rho = 0.0
        else:
            rho = math.sqrt(disc)
        lhs = 1 - rho
        claim_lo = nu * gamma ** 2 / 2
        ratio = nu / lhs if lhs > 0 else float("inf")
        bound = 2 / gamma ** 2
        ok1 = rho < 1
        ok2 = lhs >= claim_lo - 1e-15
        ok3 = ratio <= bound + 1e-9
        if not (ok1 and ok2 and ok3):
            bad += 1
            flag = "  <-- FAILS"
        else:
            flag = ""
        print("%-7.3g %-7.3g %-10.6f %-12.6g %-12.6g %-12.6g %-10.6g%s"
              % (nu, gamma, rho, lhs, claim_lo, ratio, bound, flag))

print("\nalgebraic claims: %s" % ("all hold" if bad == 0 else "%d FAILURES" % bad))

print("\n--- simulated recursion, edge binding exactly, ||Y|| = 1 ---")
print("%-8s %-8s %-10s %-14s %-12s %s"
      % ("nu", "gamma", "rounds", "sum nu*||h||", "bound 2/g^2", "ok"))
worst = 0.0
for nu in (1.0, 0.5, 0.2, 0.05, 0.01):
    for gamma in (1.0, 0.7, 0.3, 0.1):
        u = 1.0          # ||u_0|| = ||Y||
        total = 0.0
        m = 0
        while u > 1e-14 and m < 10_000_000:
            h = gamma * u            # edge binds exactly
            total += nu * h
            u2 = u * u - nu * (2 - nu) * h * h
            if u2 <= 0:
                break
            u = math.sqrt(u2)
            m += 1
        bound = 2 / gamma ** 2
        ok = total <= bound + 1e-9
        worst = max(worst, total / bound)
        print("%-8.3g %-8.3g %-10d %-14.6f %-12.6g %s"
              % (nu, gamma, m, total, bound, "yes" if ok else "NO <--"))

print("\nworst realised / bound ratio: %.4f  (must be <= 1)" % worst)
