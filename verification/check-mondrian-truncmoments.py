#!/usr/bin/env python3
"""Check the truncated inverse-moment bound and the noise bound built on it.

Lemma (inverse moments of a truncated side).  For 0 < p < 2,

    E[s^{-p}]  <=  lambda^p Gamma(2-p) (1 + p/lambda) + exp(-lambda)
               =:  c_p(lambda) * lambda^p,
    c_p(lambda) = Gamma(2-p)(1 + p/lambda) + lambda^{-p} exp(-lambda),

where s = min(E^-, U) + min(E^+, 1-U), E^+- iid Exp(lambda), U ~ U(0,1), the
law of a side of the MP(lambda, [0,1]^d) cell containing a uniform point.
At p = 1 this is lambda + 1 + exp(-lambda), the bound used in Lemma S7.

Two checks:

 (1) eq:invmom itself, by quadrature on the exact Laplace-transform integral
     left after U is integrated out analytically.  The substitution r = u^2 is
     essential: for p < 1 the integrand is singular like r^{p-1} at the origin
     and a uniform grid silently overstates it.

 (2) the noise bound E[psi / s_1] <= C_d(lambda) lambda^{1+d/2} n^{-1/2} with
     psi = E[N^{-1/2} 1{N >= 1} | cell] and
     C_d(lambda) = sqrt(2) c_{3/2} c_{1/2}^{d-1} + c_{3/2}^{(2+d)/3},
     by Monte Carlo.  Note E[s^{-3}] = infinity, so this estimator has infinite
     variance and the ratio wobbles; it is a sanity check, not a proof.
"""
import math
import numpy as np

rng = np.random.default_rng(90125)


def c_p(p, lam):
    return math.gamma(2 - p) * (1 + p / lam) + lam ** (-p) * math.exp(-lam)


def quad_E_inv_sp(p, lam, U=400.0, nu=4_000_000):
    """E[s^{-p}] = Gamma(p)^{-1} int_0^inf r^{p-1} E[e^{-rs}] dr, substituting
    r = u^2 so the r^{p-1} singularity at the origin is integrable on a
    uniform grid in u."""
    u = np.linspace(0.0, U, nu)
    r = u ** 2
    lr = lam + r
    with np.errstate(divide="ignore", invalid="ignore"):
        lap = (lam ** 2
               + 2 * lam * r * (1 - np.exp(-lr)) / lr
               + r ** 2 * np.exp(-lr)) / lr ** 2
    integrand = r ** (p - 1) * lap * 2 * u          # dr = 2u du
    integrand[0] = 0.0                              # r^{p-1} * 2u -> 0 for p > 1/2
    return float(np.trapz(integrand, u)) / math.gamma(p)


def draw_sides(lam, d, reps):
    """Sides of the MP(lam, [0,1]^d) cell containing a uniform point."""
    X = rng.random((reps, d))
    Em = rng.exponential(1.0 / lam, (reps, d))
    Ep = rng.exponential(1.0 / lam, (reps, d))
    return np.minimum(Em, X) + np.minimum(Ep, 1.0 - X)


print("(1) eq:invmom, by quadrature")
print("%-6s %-8s %-14s %-14s %-8s" % ("p", "lambda", "quadrature", "bound", "ratio"))
worst = 0.0
for p in (0.5, 1.0, 1.5):
    for lam in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        q = quad_E_inv_sp(p, lam)
        b = c_p(p, lam) * lam ** p
        worst = max(worst, q / b)
        print("%-6.1f %-8.0f %-14.5f %-14.5f %-8.5f" % (p, lam, q, b, q / b))
print("worst ratio %.5f %s" % (worst, "OK" if worst <= 1.0 else "VIOLATED"))

print()
print("(2) limiting constants, against the untruncated values in the old proof")
for d in (2, 3, 4):
    old = math.sqrt(2 * math.pi) * (math.sqrt(math.pi) / 2) ** (d - 1) \
        + math.pi ** ((2 + d) / 6)
    for lam in (4.0, 16.0, 64.0, 1e6):
        C = (math.sqrt(2) * c_p(1.5, lam) * c_p(0.5, lam) ** (d - 1)
             + c_p(1.5, lam) ** ((2 + d) / 3))
        print("d=%d lambda=%-9.0f C_d(lambda)=%-10.4f  old C_d=%.4f" % (d, lam, C, old))

print()
print("(3) E[psi/s_1] against C_d(lambda) lambda^{1+d/2} n^{-1/2}, Monte Carlo")
print("%-5s %-8s %-9s %-14s %-14s %-8s" % ("d", "lambda", "n", "MC", "bound", "ratio"))
REPS = 4_000_000
for d in (2, 3):
    for lam in (4.0, 16.0):
        for n in (1000, 20000):
            s = draw_sides(lam, d, REPS)
            V = s.prod(axis=1)
            N = rng.binomial(n, np.clip(V, 0.0, 1.0))
            pos = N >= 1
            val = np.zeros(REPS)
            val[pos] = N[pos] ** -0.5 / s[pos, 0]
            mc = float(val.mean())
            C = (math.sqrt(2) * c_p(1.5, lam) * c_p(0.5, lam) ** (d - 1)
                 + c_p(1.5, lam) ** ((2 + d) / 3))
            b = C * lam ** (1 + d / 2) * n ** -0.5
            print("%-5d %-8.0f %-9d %-14.5f %-14.5f %-8.4f"
                  % (d, lam, n, mc, b, mc / b))
