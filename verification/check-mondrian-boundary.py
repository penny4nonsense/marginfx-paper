#!/usr/bin/env python3
"""Check the boundary fix for Lemma B.

On [0,1]^d the Mondrian cell containing a uniform X is the unbounded cell
intersected with the box, so

    s_1 = min(E^-, X_1) + min(E^+, 1 - X_1),   E^- , E^+ iid Exp(lambda).

Truncation shrinks s_1 and so can inflate E[1/s_1] above the unbounded value
of lambda. The claim to check is

    E[1/s_1]  <=  lambda + 1 + exp(-lambda),

derived by writing 1/s = int_0^inf exp(-rs) dr, using
E[exp(-r min(E,u))] = [lambda + r exp(-(lambda+r)u)] / (lambda+r), and
integrating over X_1 ~ U(0,1) first.

Checked two ways: the exact one-dimensional integral left after the X-integral
is done analytically, and Monte Carlo. Note E[1/s_1^2] is infinite (near the
boundary s ~ U + Exp, and int dU/U diverges), so the Monte Carlo mean is noisy
by construction -- the quadrature is the reliable check.
"""
import math
import numpy as np

rng = np.random.default_rng(31)


def exact_E_inv_s(lam, R=4000.0, nr=2_000_000):
    """E[1/s] after integrating X_1 out analytically:

       int_0^inf [ lam^2 + 2 lam r (1-e^{-(lam+r)})/(lam+r) + r^2 e^{-(lam+r)} ]
                 / (lam+r)^2  dr
    """
    r = np.linspace(0.0, R, nr)
    lr = lam + r
    num = lam ** 2 + 2 * lam * r * (1 - np.exp(-lr)) / lr + r ** 2 * np.exp(-lr)
    return float(np.trapz(num / lr ** 2, r))


def mc_E_inv_s(lam, reps=4_000_000):
    U = rng.random(reps)
    Em = rng.exponential(1.0 / lam, reps)
    Ep = rng.exponential(1.0 / lam, reps)
    s = np.minimum(Em, U) + np.minimum(Ep, 1.0 - U)
    return float(np.mean(1.0 / s))


print("E[1/s_1] on the unit cube, against the claimed bound lambda + 1 + e^-lambda")
print("(unbounded value for comparison is exactly lambda)")
print()
print("%-8s %-12s %-12s %-14s %-10s %s"
      % ("lambda", "quadrature", "Monte Carlo", "bound", "ratio", "unbounded"))
for lam in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    q = exact_E_inv_s(lam)
    m = mc_E_inv_s(lam, 2_000_000)
    b = lam + 1 + math.exp(-lam)
    print("%-8.0f %-12.4f %-12.4f %-14.4f %-10.4f %.1f"
          % (lam, q, m, b, q / b, lam))

print()
print("implied Lemma B constant on the cube:  E[sum_m s_m/s_1] <= 1 + (d-1)(2 + 4/lambda)")
print("%-6s %-10s %-16s %s" % ("d", "lambda", "bound", "2d-1 (unbounded)"))
for d in (2, 3, 4):
    for lam in (4.0, 16.0, 64.0):
        print("%-6d %-10.0f %-16.3f %d" % (d, lam, 1 + (d - 1) * (2 + 4 / lam), 2 * d - 1))
