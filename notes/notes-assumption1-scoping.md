# Scoping: what breaks under unbounded covariate support, and what it costs

Simulation 3 uses independent standard normal covariates. Assumption 1 requires
`0 < p_min <= p(x) <= p_max < infinity` on Omega, and Assumption 2(c) requires
`||alpha_hat||_inf <= A_bar` almost surely. Neither holds for a Gaussian: the
density is not bounded below, and `alpha_h(u) = e^{-h^2/2} sinh(h u_j)/h` is
unbounded. The flagship demonstration sits outside the hypotheses of the theorem
it demonstrates.

This note records where the density bounds are actually used, so the repair can
be scoped before it is attempted.

## The headline: the product-rate condition is untouched

Step 2 of Theorem 3.1 — the bias term, and the only place the product rate is
used — reads in full:

    |sqrt(n_k) A_k| <= sqrt(n_k) ||alpha_hat - alpha_h||_{L2} ||f_hat - f||_{L2}
                     = o_p(1)

It invokes Lemma 3.3 (exact product bias) and Cauchy-Schwarz. **No operator
bound, no sup-norm bound, no density ratio.** The bound is exact, with no
linearization remainder to control.

So Assumption 2(b) stays in `L2` exactly as written, and **all of Section 3.5
stands unchanged**: the `n^{-1/4}` benchmark, the `s > d/2` threshold, and the
Schmidt-Hieber / Mourtada / Buhlmann rate citations are all statements about
`L2` risk and are unaffected by anything below.

This is the whole reason the repair is cheap. The density bounds are nowhere
near the rate theory.

## Where the density bounds are actually used

`p_min` enters in five places, and in four of them it supports a *finiteness or
consistency* claim rather than a rate.

| # | Location | Used for | Needs |
|---|---|---|---|
| 1 | Lemma 3.1, line 670 | `\|\|alpha_h\|\|_inf <= p_max/(h p_min)` | sup-norm bound on the representer |
| 2 | Lemma 3.1 proof, ~2918-2931 | divides by `p(u) >= p_min`; compares `int\|g\|du` to `E\|g(X)\|` | existence and the norm bound |
| 3 | Thm 3.1 Step 3, ~3079 | `\|\|w D_h(f_hat - f)\|\|_{L2} <= kappa \|\|f_hat - f\|\|_{L2}`, and `\|\|alpha_hat (f_hat - f)\|\|_{L2} <= A_bar \|\|f_hat - f\|\|_{L2}` | **consistency only**, `o_p(1)` |
| 4 | Thm 3.1 Step 4, ~3096 | `Var(psi) < infinity` for the CLT | finiteness only |
| 5 | Thm 3.2 Step 1, ~3141 | `V_h(g) < infinity` | finiteness only |

Plus Proposition 3.2 (~3180), whose dominated-convergence step uses
`(A_bar + ||alpha_h||_inf)^2` as the dominating constant.

## The operator bound is the real problem, not 2(c)

Item 3 is worse than it looks. The constant is
`kappa = h^{-1} (p_max/p_min)^{1/2}`, and for a Gaussian the translation
operator is *genuinely* unbounded on `L2(P)` — not merely unbounded by this
proof. Take `g_a(x) = e^{a x}`:

    ||g_a||^2       = E[e^{2aX}]           = e^{2a^2}
    ||g_a(.+h)||^2  = e^{2ah} E[e^{2aX}]   = e^{2ah} e^{2a^2}

so the operator norm is `sup_a e^{ah} = infinity`. Weakening 2(c) alone would
leave Step 3's first term with no bound at all.

## The repair

Both failing terms are controlled by Holder instead of by a sup-norm, at the
cost of a slightly higher integrability requirement on the nuisance errors.
Writing `Delta = f_hat - f`:

    E[Delta(X + h e_j)^2] = E[Delta(X)^2 e^{h X_j - h^2/2}]
                          <= ||Delta||^2_{L^{2r}(P)} ||e^{h X_j - h^2/2}||_{L^{r'}}

and the Gaussian exponential has every moment. Likewise
`E[alpha_hat^2 Delta^2] <= ||alpha_hat||^2_{L^{2s}} ||Delta||^2_{L^{2s'}}`.

So the minimal restatement is:

* **2(a)**: `L2` consistency of `f_hat` strengthened to `L^{2+delta}`
  consistency, some `delta > 0`. Consistency, not a rate.
* **2(c)**: `||alpha_hat||_inf <= A_bar` replaced by
  `||alpha_hat||_{L^q} <= A_q` for some `q > 2`.
* **2(b)**: unchanged, in `L2`.
* **Steps 4 and 3.2/Step 1**: assume `Var(psi) < infinity` directly as a
  primitive, rather than deriving it from `kappa` and `||alpha_h||_inf`. It
  holds in the Gaussian design: `E[sinh^2(hU)] = (e^{2h^2} - 1)/2 < infinity`.
* **Proposition 3.2**: replace the dominated-convergence step with the same
  Holder bound; needs `f - g` in `L^{2r'}`, a mild condition on the fixed
  approximation error.

## What the strengthening costs

Nothing, in the settings where the rate results live. Schmidt-Hieber's networks
and Mourtada's Mondrian forests are both stated on compact support with a
bounded regression function and a truncated estimator, so `||Delta||_inf` is
bounded and interpolation gives

    ||Delta||_{L^q} <= ||Delta||_inf^{1 - 2/q} ||Delta||_{L2}^{2/q} -> 0

with no additional assumption. Section 3.5 needs one added sentence saying so.

On unbounded support it is a genuine extra condition, but a mild one, and it
holds in the Gaussian design for the learners used there.

## Recommendation

Do the moment version *and* add a bounded-support arm to Simulation 3 — uniform
covariates on a box, where `p_min`, `p_max` and `||alpha_h||_inf` are all finite
and the trimming weight actually activates. The moment version keeps the
Gaussian illustration, which is the cleanest one; the bounded arm demonstrates
the theorem under its literal hypotheses, which is what removes the referee's
opening. One extra DGP through machinery that already exists.

## Separately: data-dependent h

Definition 3.x fixes `h`, and the paper argues explicitly that the step is a
feature of the estimand chosen by the analyst. Section 2 then sets
`h_j = max(1e-4, 0.05 sigma_hat_j)`, which is random, and nothing bridges the
two. Since `sigma_hat_j` is root-n consistent and `theta_h` is smooth in `h`,
this is likely a short lemma or an explicit conditioning statement — but it is
unstated in a paper whose architecture depends on `h` being fixed.
