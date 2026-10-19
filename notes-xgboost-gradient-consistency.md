# Gradient consistency for XGBoost: why the current proof misses it, and a route

Status: **proposed route, not a proof.** Nothing here is established. §2 is a
finding about the existing theorem and I am confident in it; §3 onward is a
plan, with the open gaps listed in §6.

Written 2026-10-19, following the referee note on Theorem S2 (§S24).

Companion to `notes-mondrian-bv-bound.md`. The two are independent: this one
looks more tractable.

---

## 1. What has to be shown

To put XGBoost under Lemma S4 (BV derivative consistency) and conclude
`Df̂ₙ ⇀* Df`, two conditions are needed:

1. `sup_n E[ |Df̂ₙ|(Ω) ] < ∞`   — uniform BV boundedness
2. `‖f̂ₙ − f‖_{L¹} →_p 0`       — L¹ consistency

with `f̂ₙ = ν Σ_{m ≤ M} h_m`, each `h_m` a regression tree of depth ≤ D.

## 2. Why the current theorem cannot be stretched to cover it

Theorem S2 assumes `F̂_m = (I − (I−S)^{m+1})Y` for a **fixed, symmetric, linear**
`S` with `0 < λ_k ≤ 1` for all k. That representation holds only if the same
operator is applied every round.

**Case (ii) of the theorem — tree stumps — appears to have no examples at all.**
The eigenvalue condition forces `S` to be full rank. Any tree is a projection
onto the piecewise-constant functions on its partition, so its spectrum is
`{0,1}` with `n − (#leaves)` zeros, giving `ρ = max_k(1−λ_k) = 1` and a
divergent geometric series. The supplement says as much in its own closing
remark ("as occurs for linear projection operators ... the geometric series does
not converge"). So:

- a **fixed** stump is a projection and is excluded by the eigenvalue condition;
- an **adaptive** stump — which is what Case (ii) actually describes, its
  threshold `t_m` being indexed by iteration — is not a fixed linear `S`, so
  neither the representation nor the geometric decay applies.

Shrinkage does not rescue this (`νS` has spectrum `{0, ν}`), nor does an L2 leaf
penalty. The zeros are unavoidable for any finite-rank base learner. The same
objection applies to the "shallow neural net base learners" offered as an
example under Case (i); fixed-smoothing-parameter **splines** are the surviving
example, which is Bühlmann–Yu's original setting.

Conclusion: this is not a theorem to broaden. It needs a different engine.

## 3. The structural observation that makes a different engine possible

**The BV bound does not care where the splits are.** For a tree `h` whose
internal nodes each contribute one split face,

    |Dh|(Ω) = Σ_{adjacent leaves l,l'} |c_l − c_l'| · H^{d-1}(∂R_l ∩ ∂R_l')
            ≤ (#internal nodes) · 2‖h‖_∞ · diam(Ω)^{d-1},

since the total interface area is at most the sum of the internal nodes' split
faces, and each of those is at most `diam(Ω)^{d-1}` wherever it falls. Hence

    |Df̂ₙ|(Ω) ≤ ν · (2^D − 1) · 2 · diam(Ω)^{d-1} · Σ_{m ≤ M} ‖h_m‖_∞.

Split *locations* — the data-dependent, greedy, non-linear part, the part that
destroys the spectral argument — enter only through face areas that are bounded
regardless. Greediness is not an obstacle to condition 1.

It is an obstacle to condition 2, and that half is already in the literature:
Zhang & Yu (2005, *Ann. Statist.*), boosting with early stopping, covers tree
base learners under step-size restrictions.

**So the whole problem reduces to bounding `ν Σ_{m ≤ M} ‖h_m‖_∞`.**

## 4. The proposed engine: an edge condition in place of a spectral one

The existing proof needs `ρ < 1` and obtains it from the eigenvalues of a fixed
operator. Obtain it instead from a **weak-learning / edge condition**: each
round captures at least a fixed fraction `γ` of the remaining residual, so that

    ‖u_{m+1}‖² ≤ (1 − 2νγ² + O(ν²)) ‖u_m‖² ,

giving geometric decay and a summable `Σ_m ‖h_m‖`. This is the standard
assumption of the boosting literature, it is a statement about the base learner
class relative to `f` rather than about a fixed operator, and it **survives
adaptivity** — indeed greedy selection is what makes an edge condition
plausible.

Same engine as the current proof, different and compatible hypothesis.

## 5. XGBoost's own regularisers are the hypotheses one would want

Unusually, the practical defaults all push the right way. With the paper's
settings (`max_depth` 4, `learning_rate` 0.05, 100–200 rounds):

| setting | effect on the bound |
|---|---|
| `max_depth = 4` | ≤16 leaves, ≤15 split faces per round; bounds the face count with no asymptotics |
| `learning_rate ν = 0.05` | multiplies the total variation bound directly |
| `reg_lambda` (L2 on leaf weights, default 1) | leaf weight is `−G_l/(H_l+λ)`; for squared error this is the leaf-mean residual shrunk by `n_l/(n_l+λ)`, so it shrinks the jumps |
| `min_child_weight` | keeps leaves from being nearly empty — the exact obstacle that blocks the Mondrian argument |

A theorem stated under these conditions would be describing the estimator people
actually run, not an idealisation of it.

## 6. The gaps

1. **The edge condition itself.** Needs stating, and defending as reasonable for
   depth-D trees against an `f ∈ BV`. This is the substantive assumption.
2. **L² → L∞ — the real technical gap.** Bühlmann–Yu-style decay is in `L²`, but
   total variation needs the *jump* magnitudes, which are leaf-level residual
   averages. An `L²` bound controls those only if leaves are not tiny.
   `min_child_weight` supplies that, but the step has to be made explicitly
   rather than waved through. **This is where I would expect the work to be.**
3. **L¹ consistency under early stopping.** Cite rather than prove (Zhang & Yu
   2005; Biau & Cadre for related tree results).
4. **Subsampling**, if enabled, as extra randomness to condition on.
5. **Second-order (Newton) boosting.** XGBoost minimises a quadratic
   approximation with Hessian weights. For squared error the Hessian is 1 and
   this is ordinary L₂Boost; for logistic loss it is not, and the residual
   recursion changes. The classification experiments use the latter.

## 7. The alternative worth weighing

With `M` **fixed** at 100–200 rounds, which is what is actually run, condition 1
is trivial — a finite sum of bounded-depth trees has bounded total variation,
no edge condition needed — but condition 2 fails: a fixed-round ensemble does
not converge to `f`. That puts XGBoost-as-run in the pseudo-true regime, which
**Proposition 2 (double robustness) already covers**.

So there is a defensible position in which the correct statement about
XGBoost-as-run is a double-robustness statement rather than a gradient
consistency one, and the gradient result is about the `M → ∞` idealisation. That
is worth deciding deliberately rather than by default.

## 8. Effort, and what it does not block

Weeks to a couple of months for the BV lemma, granting the edge condition —
a paper section or a short standalone note. More tractable than the Mondrian
repair, which has no comparable workaround for its nearly-empty-cells obstacle.

Nothing in the main inference theory waits on this. Theorem 1 consumes an `L²`
rate and nothing else, and XGBoost has such rates. The pairing identity
(Theorem 3) already characterises what its window effect measures, since a
finite sum of piecewise-constant trees is `BV`. What is unproved is only that
XGBoost's **uncorrected plug-in** is consistent for the gradient — the quantity
the paper argues throughout should not be relied on.

## 9. If it is not proved before submission

Narrow the three places that currently imply coverage:

- the checklist row `Gradient boosting | established | BV, proved here`;
- the abstract's "boosted trees";
- Case (ii) of Theorem S2, which §2 above suggests is empty as stated.

and add a sentence to §3.5 recording that XGBoost is covered for inference and
for the pairing identity but not for gradient consistency. A referee will ask
the moment they see XGBoost in the empirical tables.
