# Gradient consistency for XGBoost: why the current proof misses it, and a route

Status, by section:

- **§2 is a finding about Theorem S2.** I am confident in it.
- **§4 is a draft lemma whose conclusion is false as an `M → ∞` statement.**
  Step 2 is verified and (A2) is repairable by configuration, but (A3) does not
  hold, and the quantity the lemma bounds — while convergent in `M` for each
  fixed `n` — has a limit growing like `n^{0.8}`, so it is not uniform in `n`.
  At a fixed round count it is flat in `n`, but a fixed round count gives no
  consistency; §4.5 shows that stopping at the risk-minimising round rescues
  the bound only for targets the tree class captures with a bounded boosting
  path. See §4.4 and §4.5.
- **§6.1–6.2 are verified** — geometry in `d ≤ 4` and the ensemble bound with
  face overlap deliberately induced. §6.5's two obstacles stand.
- **§8 is now the recommended position**, for regression as well as
  classification, which is not where this started.

Companion to `notes-mondrian-bv-bound.md`, and independent of it.

---

## 1. What has to be shown

To put XGBoost under Lemma S4 (BV derivative consistency) and conclude
`Df̂ₙ ⇀* Df`, two conditions are needed:

1. `sup_n E[ |Df̂ₙ|(Ω) ] < ∞`   — uniform BV boundedness
2. `‖f̂ₙ − f‖_{L¹} →_p 0`       — L¹ consistency

with `f̂_M = ν Σ_{m ≤ M} h_m`, each `h_m` a regression tree of depth ≤ D.

Condition 2 is available in the literature (Zhang & Yu 2005, *Ann. Statist.*,
boosting with early stopping, tree base learners under step-size restrictions).
**Condition 1 is the work**, and §4 is a candidate proof of it.

## 2. Why the current theorem cannot be stretched to cover it

Theorem S2 assumes `F̂_m = (I − (I−S)^{m+1})Y` for a **fixed, symmetric, linear**
`S` with `0 < λ_k ≤ 1` for all k. That representation holds only if the same
operator is applied every round.

**Case (ii) — tree stumps — appears to have no examples at all.** The eigenvalue
condition forces `S` to be full rank, but any tree is a projection onto the
piecewise-constant functions on its partition, so its spectrum is `{0,1}` with
`n − (#leaves)` zeros, giving `ρ = max_k(1−λ_k) = 1` and a divergent geometric
series. The supplement says as much in its own closing remark. So a **fixed**
stump is excluded by the hypothesis, while an **adaptive** stump — which is what
Case (ii) describes, threshold `t_m` indexed by iteration — is not a fixed
linear `S` at all. Shrinkage gives spectrum `{0, ν}`, still with zeros; an L2
leaf penalty does not help either. The zeros are unavoidable for any finite-rank
base learner. The same objection applies to the "shallow neural net base
learners" offered under Case (i); fixed-smoothing-parameter **splines** are the
surviving example, which is Bühlmann–Yu's original setting.

This is not a theorem to broaden. It needs a different engine.

## 3. The structural observation

**The BV bound does not care where the splits fall.** For a tree `h` whose
internal nodes each contribute one split face `F_k`,

    |Dh|(Ω) = Σ_k ∫_{F_k} |Δ_k(u)| dH^{d-1}(u) ≤ 2‖h‖_∞ Σ_k H^{d-1}(F_k),

and `H^{d-1}(F_k) ≤ diam(Ω)^{d-1}` wherever the split lands. Split locations —
the greedy, data-dependent, non-linear part, the part that destroys the spectral
argument — enter only through face areas that are bounded regardless.

Greediness obstructs condition 2, not condition 1. And condition 2 is citable.
So everything reduces to controlling `ν Σ_m ‖h_m‖_∞`.

## 4. Draft lemma: uniform BV boundedness

**Assumptions.** Squared-error loss; leaf weights `w_l = −G_l/(H_l + λ)` with
`λ ≥ 0`; and

- **(A1)** depth `≤ D`, so at most `2^D` leaves and `2^D − 1` internal nodes;
- **(A2)** every leaf holds at least `κn` observations, `κ ∈ (0, 2^{-D}]`;
- **(A3)** *edge condition*: `‖h_m‖_{2,n} ≥ γ‖u_m‖_{2,n}` for all `m ≤ M`, some
  `γ ∈ (0,1]`, where `‖·‖_{2,n}` is the empirical RMS norm and `u_m` the
  residual vector at round `m`;
- **(A4)** `ν ∈ (0,1]`;
- **(A5)** Assumption 1 of the paper (`E[Y²] < ∞`).

**Claim.** For every `M`,

    E|Df̂_M|(Ω)  ≤  4 (2^D − 1) diam(Ω)^{d-1} κ^{-1/2} γ^{-2} (E[Y²])^{1/2}  <  ∞,

uniformly in `n` and in `M`.

**Proof.**

*Step 1 (geometry).* By §3, `|Dh_m|(Ω) ≤ 2‖h_m‖_∞ (2^D − 1) A` with
`A = diam(Ω)^{d-1}`, using (A1).

*Step 2 (leaf values, empirical L² to sup norm).* For squared-error loss
`g_i = −u_i` and `h_i = 1`, so the leaf weight is
`w_l = (Σ_{i∈l} u_i)/(n_l + λ) = θ_l ū_l` with `θ_l = n_l/(n_l+λ) ∈ (0,1]`.
By Cauchy–Schwarz and then (A2),

    |ū_l| ≤ n_l^{-1} Σ_{i∈l}|u_i| ≤ n_l^{-1/2} (Σ_{i∈l} u_i²)^{1/2}
         ≤ (n/n_l)^{1/2} ‖u_m‖_{2,n} ≤ κ^{-1/2} ‖u_m‖_{2,n},

hence `‖h_m‖_∞ ≤ κ^{-1/2} ‖u_m‖_{2,n}`.

**This is the step that closes the L² → L∞ gap**, and it works because leaf
values are *averages*, not pointwise values: a minimum leaf fraction converts a
bound on the empirical L² norm of the residuals into a sup-norm bound on the
fitted function. It is also why (A2) is load-bearing rather than cosmetic.

*Step 3 (residual recursion).* `u_{m+1} = u_m − ν h_m`. Since
`⟨u_m, h_m⟩_n = n^{-1} Σ_l θ_l n_l ū_l²` and
`‖h_m‖²_{2,n} = n^{-1} Σ_l θ_l² n_l ū_l²`, and `θ_l ≥ θ_l²`, we get
`⟨u_m, h_m⟩_n ≥ ‖h_m‖²_{2,n}` — the L2 leaf penalty helps here rather than
hurting. Therefore

    ‖u_{m+1}‖²  =  ‖u_m‖² − 2ν⟨u_m,h_m⟩ + ν²‖h_m‖²
                ≤  ‖u_m‖² − ν(2−ν)‖h_m‖²
                ≤  ‖u_m‖² (1 − ν(2−ν)γ²)  =:  ρ² ‖u_m‖²

by (A3), with `ρ < 1` for `ν ∈ (0,1]`, `γ ∈ (0,1]`. Hence
`‖u_m‖_{2,n} ≤ ρ^m ‖Y‖_{2,n}`.

*Step 4 (the sum, and why ν cancels).* Since `‖h_m‖ ≤ ‖u_m‖`,

    Σ_{m<M} ν ‖h_m‖_{2,n} ≤ ν ‖Y‖_{2,n} Σ_m ρ^m ≤ ν ‖Y‖_{2,n} / (1−ρ),

and `1 − ρ = 1 − (1 − ν(2−ν)γ²)^{1/2} ≥ ν(2−ν)γ²/2 ≥ νγ²/2` for `ν ≤ 1`, so
`ν/(1−ρ) ≤ 2γ^{-2}` and

    Σ_{m<M} ν ‖h_m‖_{2,n}  ≤  2 γ^{-2} ‖Y‖_{2,n}.

**The learning rate cancels.** The bound does not degrade as `ν → 0`, which is
the right behaviour: a smaller learning rate buys more rounds, and the two
effects offset exactly.

*Step 5 (assemble).* Combining Steps 1, 2 and 4,

    |Df̂_M|(Ω) ≤ Σ_m ν|Dh_m|(Ω) ≤ 2(2^D−1) A κ^{-1/2} Σ_m ν‖u_m‖_{2,n}
              ≤ 4(2^D−1) A κ^{-1/2} γ^{-2} ‖Y‖_{2,n}.

Take expectations and use `E‖Y‖_{2,n} ≤ (E‖Y‖²_{2,n})^{1/2} = (E[Y²])^{1/2}`,
finite by (A5). ∎

**Numerical check (2026-10-19).** Steps 3 and 4 were checked numerically over
`ν ∈ [0.001, 1]` and `γ ∈ [0.03, 1]`: `ρ < 1`, `1 − ρ ≥ νγ²/2` and
`ν/(1−ρ) ≤ 2γ^{-2}` hold throughout, and simulating the recursion with the edge
binding exactly gives a realised `Σ ν‖h_m‖` at worst 60% of the claimed bound.

The same simulation shows **the constant `2γ^{-2}` is loose by a factor of about
`2/γ`**: the realised sum converges to `γ^{-1}‖Y‖` as `ν → 0`. The looseness
enters at `‖h_m‖ ≤ ‖u_m‖` in Step 4, which throws away the edge. Writing
`‖h_m‖ = c‖u_m‖` for `c ∈ [γ,1]` gives a sum of about `c^{-1}‖Y‖`, maximised at
`c = γ`; so the sharp constant looks like `γ^{-1}`, attained in the weak-learner
limit, and the proof above is a safe but crude route to it. Sharpening it is
cosmetic — uniform boundedness, which is all condition 1 needs, follows either
way — but it would improve the explicit constant if the lemma is ever stated.

**Two features that looked attractive — the first of which turned out to be the
problem.**

1. *The bound is uniform in M*, so condition 1 would place no constraint on the
   stopping rule and the two halves of §1 would decouple. **This is the claim
   §4.4 refutes.** It holds only because (A3) is assumed with a `γ` that does
   not vary with `m`; once the realised edge ratio is allowed to decay, as it
   does, the uniformity goes with it.
2. *The constant is explicit and interpretable*: depth enters as `2^D`, the
   minimum leaf fraction as `κ^{-1/2}`, the edge as `γ^{-2}`. That remains true
   and is what makes the failure diagnosable — the `γ^{-2}` is where the
   three-orders-of-magnitude slack in §4.4 comes from.

### 4.4 Empirical check — the hypotheses and the conclusion

Run 2026-10-19 with `check-xgboost-assumptions.py`, `check-xgboost-growth.py`
and `check-xgboost-settings.py`, on the paper's own linear regression DGP
(`y = 2x₁ + 3x₂ + ε`, `x ∼ N(0,I₄)`) and settings (`max_depth 4`, `eta 0.05`,
`reg_lambda 1`).

**Step 2 holds.** `‖h_m‖_∞ ≤ κ^{-1/2}‖u_m‖_{2,n}` is satisfied at every round in
every configuration, worst ratio 0.35. Reconstructing leaf values as shrunk
means `G_l/(n_l+λ)` matches XGBoost to `1e-6`, so the model of the algorithm
behind the step is right.

**(A2) fails as configured, but is repairable.** At the default
`min_child_weight = 1` the smallest leaf holds exactly one observation at every
sample size, so `κ̂ = 1/n`: `4.0e-3`, `1.0e-3`, `2.0e-4` at
`n = 250, 1000, 5000`. Setting `min_child_weight = n/32` gives
`κ̂ ≈ 0.03`, stable in `n`. A configuration change, not an obstacle.

**(A3) fails and is not repairable.** The realised edge ratio decays in both
arguments — roughly `n^{-0.4}` across sample sizes, and from `0.91` to `0.066`
within a 200-round run. No uniform `γ` exists, and the `2γ^{-2}‖Y‖` bound it
yields is three orders of magnitude loose: `694`, `2187`, `7833` against a
realised quantity near `10`.

**The conclusion fails too — but in `n`, not in `M`.** This needed two passes to
get right. Fitting the growth of `Σ ν‖h_m‖_∞` over `M ≤ 4000` gives `M^{0.3}`
to `M^{0.5}` and suggests divergence in `M`. **That is a transient.** Carried
to `M = 16000` at the paper's settings the local slope collapses —
`0.52, 0.46, 0.22, 0.05` — and the sum saturates. For each `n` it converges.

What it converges *to* is the problem:

| `n` | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|
| `Σ ν‖h_m‖_∞` at `M = 200` | 15.1 | 19.0 | 20.4 | 19.4 |
| saturated limit (`M → ∞`) | 34.6 | 60.9 | 106.3 | >195.8 |
| local slope at `M = 16000` | 0.000 | 0.000 | 0.048 | 0.242 |

The saturated limit grows like `n^{0.8}`, so
`sup_n E[|Df̂_n|(Ω)] = ∞` and the lemma's conclusion is false — but as a
statement about `n`, which is the uniformity Lemma S4 actually demands.

The mechanism is ordinary overfitting seen in the BV norm. Unstopped boosting
at these settings eventually interpolates the training data; an interpolant of
`n` noisy points needs about `n` jumps, so its total variation grows with `n`.
The energy budget does not prevent this: `Σ ν(2−ν)‖h_m‖²_{2,n}` is essentially
saturated (13.98 against 14.40), so late rounds carry almost no energy, but
they spend it on increasingly spiky trees — large on one small leaf, negligible
elsewhere. That costs nothing in RMS and a great deal in sup norm, which is the
norm the BV bound needs.

**The row that matters is the first one.** At `M = 200` the sum is `15.1, 19.0,
20.4, 19.4` — **flat in `n`**. Early stopping is what delivers the uniformity,
and it delivers it without any change of settings. So the obstruction is not
that XGBoost is unsuited to the BV route; it is that the `M → ∞` idealisation
is the wrong object, and §8's early-stopped regime is the right one.

**Settings cannot substitute for stopping.** Sweeping `reg_lambda ∈ {1, n/100,
n/10, n, 10n}`, `min_child_weight ∈ {1, n/32, n/16}` and `max_depth ∈ {1,2,4}`,
no configuration removes the `n`-growth of the unstopped limit, and the two
knobs pull against each other: light regularisation gives fast residual decay
but spiky late trees, heavy regularisation gives small trees but residuals that
barely decay, so the sum accrues over more rounds. `reg_lambda = n/10` was
best over `M ≤ 4000` yet, unlike the defaults, had **not** saturated by
`M = 16000` (slope 0.36) — heavy shrinkage prevents the interpolation that
makes the default sum converge at all.

Settings do help the **constant** in the early-stopped regime, which is where
they are worth using: at `M = 200`, `reg_lambda = n/10` cuts the sum from
`20.4` to `7.9`, and `depth 1 + min_child_weight = n/16 + reg_lambda = 10n`
reaches `0.93`.

### 4.5 Does early stopping rescue it? Only for easy targets

§4.4 ends on an encouraging note — at a fixed round count the sum is flat in
`n`. But a fixed round count gives no consistency: as `n → ∞` with `M` fixed,
`f̂` converges to the `M`-round *population* boosting limit, not to `f`. The
theorem needs `M_n → ∞`, so the question is whether `M_n` can grow fast enough
for consistency while `Σ_{m<M_n} ν‖h_m‖_∞` stays bounded.

Measured by stopping at `M*`, the round minimising risk against the noiseless
target on 20,000 held-out points (`check-xgboost-earlystop.py`):

| target | `M*` growth | sum at `M*` growth | risk/var at `n=8000` |
|---|---|---|---|
| `2x₁ + 3x₂` (paper, linear) | `n^{0.03}` | `n^{0.11}` | 0.004 |
| `2x₁ + 3x₂ + 2x₁x₂` (paper, interaction) | `n^{0.25}` | `n^{0.17}` | 0.006 |
| `2x₁² + 3x₂` (paper, nonlinear) | `n^{0.04}` | `n^{0.18}` | 0.011 |
| `3sin(4x₁) + 3x₂` | `n^{-0.05}` | `n^{0.04}` | 0.009 |
| `5exp(−‖x‖²/2)` (radial, 4-d) | `n^{0.36}` | `n^{0.32}` | 0.076 |

**The sum at `M*` grows whenever `M*` grows, and at about the same rate.** The
radial target is the clear failure: a smooth rotationally symmetric function is
badly suited to axis-aligned splits, boosting has not converged to it even at
`n = 8000` (risk still 7.6% of signal variance), `M*` keeps climbing at
`n^{0.36}` and the sum climbs with it at `n^{0.32}`. Nothing suggests either
levels off.

So **early stopping does not supply the uniformity on its own.** It supplies it
exactly when `M*` is bounded — when the tree class captures `f` with a bounded
boosting path. The first four rows are targets of that kind: the oscillation is
fine, despite looking hard, because a one-dimensional wiggle is cheap for
axis-aligned trees. The radial target is not, and it is not exotic.

**What this leaves is a theorem with a hypothesis on `f`, not on the
algorithm.** Something of the form: *if `f` lies in the variation space of the
depth-`D` tree dictionary — equivalently, if the population boosting path
reaches `f` with bounded total variation — then early-stopped XGBoost is
BV-bounded and gradient-consistent.* That is a real and statable result, and it
is the condition §4 should have carried from the start. It is also much
narrower than "gradient consistency for XGBoost", and verifying it for a given
application is not obviously easier than the problem it replaces.

**For the paper's own simulations this is benign.** All three of its DGPs sit
in the slow-growth group (`n^{0.11}` to `n^{0.18}` over a 16-fold range of `n`),
so nothing in Section 5 of the paper is in jeopardy. The difficulty is with the
generality a theorem would have to claim, not with the experiments reported.

## 5. XGBoost's regularisers in the light of §4

| setting | role in the lemma |
|---|---|
| `max_depth = 4` | (A1): ≤15 internal nodes per round |
| `learning_rate = 0.05` | (A4); cancels out of the final bound |
| `reg_lambda` (L2 on leaf weights) | gives `θ_l ≤ 1` in Step 3, strengthening `⟨u,h⟩ ≥ ‖h‖²` |
| `min_child_weight` | **(A2) — load-bearing.** See the gap below |

## 6. The logistic (Newton) case

Worked 2026-10-19. The conclusion is that §4 **does not translate routinely**,
but that the obstruction is narrower than it first looks, and that for the
configuration actually run the classification case is in fact the *easy* one.

### 6.1 The estimand is the probability, not the margin

The paper extracts classification predictions as `predict_proba(X)[:,1]`, so

    f̂ₙ = σ ∘ F_M ,      F_M = ν Σ_{m≤M} h_m   (the margin),

and the trees are fit on the margin scale. Two consequences, both helpful:

- `σ` is `1/4`-Lipschitz (`σ' = σ(1−σ) ≤ 1/4`), so
  `|D f̂ₙ|(Ω) ≤ ¼ |DF_M|(Ω)`. Bounding the margin's total variation suffices,
  with a factor of four gained.
- Independently, `0 ≤ σ ≤ 1`, so **every jump in the fitted probability is at
  most 1, however wild the margin is**, giving

      |D f̂ₙ|(Ω) ≤ M (2^D − 1) diam(Ω)^{d-1}

  unconditionally — no edge condition, no leaf condition, no assumption on the
  Hessians. The argument is in §6.1.1; what controls it is the **area** of the
  jump set, not a count of interfaces.

#### 6.1.1 The jump-set bound, carefully

For piecewise constant `g`, `Dg` is concentrated on the jump set `J(g)` and

    |Dg|(Ω) = ∫_{J(g)} |g⁺ − g⁻| dH^{d-1},

so what has to be bounded is `H^{d-1}(J)`, a `(d−1)`-dimensional measure, not a
number of faces. Three steps:

1. `F_M = ν Σ_m h_m` can only jump where some `h_m` jumps, so
   `J(F_M) ⊆ ⋃_m J(h_m)`.
2. Within one tree, two adjacent leaves meet only inside the split face of
   their lowest common ancestor — the first split that separated them — so
   `J(h_m) ⊆ ⋃_k F_k^m` over that tree's internal nodes. A binary tree with
   `L ≤ 2^D` leaves has exactly `L − 1 ≤ 2^D − 1` internal nodes, each
   contributing one face, and each face is a slice of a cell, so
   `H^{d-1}(F_k) ≤ diam(Ω)^{d-1}`.
3. `H^{d-1}` is subadditive, so
   `H^{d-1}(J(F_M)) ≤ Σ_m Σ_k H^{d-1}(F_k^m) ≤ M(2^D − 1) diam(Ω)^{d-1}`.

Then `|σ⁺ − σ⁻| ≤ 1` pointwise on `J` gives the display above.

**Overlap between rounds cannot break this, in either of the two ways it might
have.** Step 3 uses subadditivity, so faces that coincide or partially overlap
make the true area *smaller* than the bound — the slack is in the safe
direction. And where faces from different rounds do coincide, the margin jumps
add, possibly to something large; but the final step caps the *probability*
jump at 1 regardless of the margin, so the addition is absorbed. The cap is
exactly what makes the bound indifferent to overlap.

**Numerical check (2026-10-19).** `check-tree-faces.py` runs two checks, both
computing `H^{d-1}` exactly on axis-aligned partitions of the unit cube.

*Single tree*, `d = 2,3,4` and `D = 2…5`, 1200 trials per cell, ≈190,000
adjacent leaf pairs: all four claims pass with zero failures. In particular
**step 2 — the lowest-common-ancestor containment — holds in every trial,
including in `d = 3` and `d = 4`**, which was the step I was least sure of and
where faces are rectangles rather than segments. Step 3's inequality is
attained exactly (worst ratio 1.0000 at every `(d,D)`), so it is the right
bound rather than a safe one; the looseness sits in the `2‖h‖_∞` factor of §4
Step 1, whose realised ratio falls from 0.88 at `D=2` to 0.16 at `D=5`.

*Ensemble*, which is the claim as actually stated: `F = ν Σ_{m≤M} h_m` for
`M = 2, 5, 10`, with `|D σ(F)|(Ω)` computed exactly on the common refinement.
Thresholds are drawn from a coarse shared grid **so that faces from different
trees coincide on purpose** — 10 to 683 coincidences per configuration, so the
overlap case is exercised rather than avoided by continuous randomness. Both
the cap bound and the `¼`-Lipschitz bound hold in every trial.

Two things the ensemble run shows. The cap `M(2^D−1)A` is very loose and gets
looser as `M` grows (realised ratio 0.32 down to 0.06), because it grows
linearly in `M` while the total variation of a *probability* saturates. The
`¼` in the Lipschitz bound, by contrast, is nearly attained at `M = 2`
(ratio 0.99), as it should be: with small margins `σ` is locally `¼`-Lipschitz,
so that constant cannot be improved.

The bound is of course very loose overall — `200 · 15 = 3000` against a truth
whose `|Df|(Ω) = ∫|∇f|` is typically `O(1)`. That does not matter here:
condition 1 asks only for a bound uniform in `n`, and this one does not involve
`n` at all.

### 6.2 For fixed M, condition 1 is therefore trivial

At the 100–200 rounds actually run, with `max_depth = 4`, the display above
gives `|D f̂ₙ|(Ω) ≤ 200 · 15 · diam(Ω)^{d-1}`, uniformly in `n`, with no
assumptions beyond depth and round count. **The classification case needs none
of §4.** What it needs is condition 2, which fixed `M` does not supply — see §8.

### 6.3 Where §4 breaks if M → ∞

Step 2 is the casualty. For logistic loss `g_i = p_i − y_i` and
`h_i = p_i(1−p_i)`, so the leaf weight is

    w_l = −G_l/(H_l + λ) = Σ_{i∈l}(y_i − p_i) / (Σ_{i∈l} p_i(1−p_i) + λ),

and the working response `z_i = −g_i/h_i` equals `1/p_i` when `y_i = 1` and
`−1/(1−p_i)` when `y_i = 0`. **Both are unbounded as predictions sharpen.** A
confidently wrong observation has vanishing Hessian and order-one gradient, so
its Newton step diverges. This is the well-known instability of second-order
boosting under logistic loss, and `λ > 0` alone does not fix it: it gives only
`|w_l| ≤ n_l/λ`, which grows with the sample.

### 6.4 What replaces it

The structure of §4 does carry over, in the Hessian-weighted geometry. Write
`‖a‖²_{H,n} = n^{-1} Σ_i h_i a_i²`. The leaf weight is a shrunk
`h`-weighted mean of the working responses, so by Cauchy–Schwarz in that inner
product, and writing `H_l ≥ ηn` for a minimum *average* leaf Hessian,

    ‖h_m‖_∞ ≤ η^{-1/2} ‖z_m‖_{H,n} ,

which is exactly Step 2 with `κ → η` and `‖·‖_{2,n} → ‖·‖_{H,n}`. And `η` is
precisely what `min_child_weight` controls: **XGBoost defines it as a minimum
sum of instance Hessians, not a minimum count.** For squared error `h_i = 1` and
it reduces to the leaf-fraction assumption (A2); for logistic it is the
assumption that predictions in a leaf are not uniformly saturated. So the two
cases share one parameter, read in the loss's own units.

### 6.5 The two genuine obstacles

1. **The decay argument has to be rebuilt.** Step 3 used
   `u_{m+1} = u_m − νh_m` with an exact quadratic loss. Logistic loss is not
   quadratic, so what decays is the loss itself, via the gain
   `Σ_l G_l²/(H_l + λ)`, and the quadratic model's error has to be controlled.
   The standard tool is self-concordance of the logistic loss; that is real
   work, not a translation.
2. **`η` is in tension with what the algorithm does.** Logistic boosting
   *drives* `p` towards 0 and 1 — on separable data that is its whole
   behaviour — so the average leaf Hessian degrades as `M` grows and a fixed
   `η` cannot hold uniformly. This is the same shape as the (A2)-versus-edge
   interaction in item 2 of §7 but sharper, because here the algorithm actively
   destroys the assumption rather than merely exhausting it.

Obstacle 2 is the reason I would not expect a clean `M → ∞` theorem for the
logistic case, and the reason §6.2 and §8 matter more here than for regression.

## 7. What is still missing

1. **(A2)** — settled empirically in §4.4. It fails at XGBoost's defaults
   (`κ̂ = 1/n`) and is restored by `min_child_weight = n/32`, giving a stable
   `κ̂ ≈ 0.03`. A configuration change, not a mathematical obstacle.
2. **(A3)** — refuted empirically in §4.4. The edge ratio decays like `n^{-0.4}`
   and falls by an order of magnitude within a run, so no uniform `γ` exists,
   and the `γ^{-2}` bound it yields is three orders of magnitude loose. This
   hypothesis should be abandoned rather than repaired.
3. **The conclusion of §4** — refuted in §4.4 as an `M → ∞` statement (the
   unstopped limit grows like `n^{0.8}`, and no configuration removes it), and
   only partly recovered by early stopping in §4.5. What survives is a theorem
   conditional on `f` lying in the variation space of the tree dictionary.
   Anyone revisiting this should start there, and should decide whether that
   hypothesis is checkable in practice — if it is not, the result buys little.
4. **Newton boosting / logistic loss.** Worked out in §6. Step 2 survives in
   Hessian-weighted form; Step 3 does not, and the minimum-Hessian parameter is
   in tension with what logistic boosting does. For the fixed round counts
   actually run, §6.2 makes condition 1 trivial and the issue disappears.
5. **Subsampling**, if enabled, as additional randomness to condition on.
6. **The consistency half**, if it is ever needed, has to be imported and
   matched to the same assumption set — in particular, Zhang & Yu's step-size conditions must be
   compatible with (A3) and (A4).

## 8. The alternative worth weighing

With `M` **fixed** at the 100–200 rounds actually run, condition 1 is trivial —
a finite sum of bounded-depth trees has bounded total variation, with no edge
condition and no (A2) — but condition 2 fails: a fixed-round ensemble does not
converge to `f`. That places XGBoost-as-run in the pseudo-true regime, which
**Proposition 2 (double robustness) already covers**.

So there is a defensible position in which the correct statement about
XGBoost-as-run is a double-robustness one, and gradient consistency belongs to
the `M → ∞` idealisation. Worth deciding deliberately rather than by default.

**This is now the recommended position, and for regression too.** §4.4 shows
the unstopped limit of `Σ ν‖h_m‖_∞` growing like `n^{0.8}` — the BV bound is
not uniform in `n` precisely when boosting is allowed to run to interpolation,
which is what `L¹` consistency would require. At a fixed round count the same
quantity is flat in `n`. And §6.5 item 2 gives an independent obstruction for
logistic loss. Both routes to an `M → ∞` theorem are blocked, by different
mechanisms, while the early-stopped regime is empirically well behaved in both
cases.

§4.5 narrows this further: stopping at the risk-minimising round keeps the sum
flat only for targets the tree class captures cheaply, and fails on a plain
4-dimensional radial function. So even the early-stopped regression statement
needs a hypothesis on `f`.

For classification the fixed-`M` bound is additionally *unconditional* —
`|D f̂ₙ|(Ω) ≤ M(2^D−1)A` follows from `0 ≤ σ ≤ 1` alone, needing no edge
condition, no leaf condition and nothing from §4 — and that bound is verified
(§6.1.1). For regression the fixed-`M` bound is equally easy: a finite sum of
bounded-depth trees has bounded total variation.

What the position costs is that `f̂ₙ` is then not consistent for `f`, so the
object being estimated is a pseudo-true limit. The paper already has the
machinery for exactly that case.

## 9. Effort, and the revised recommendation

The honest summary after the checks: **do not chase the `M → ∞` theorem.**

§4's route is refuted, not merely unproved — (A3) is false, the conclusion is
false in `n` once boosting runs to interpolation (§4.4), and early stopping
recovers it only under a hypothesis on `f` that is no easier to check than the
original problem (§4.5). §6.5's obstruction for logistic loss is independent
and also structural. The remaining work to rescue a limit theorem would be to
find some other functional of the ensemble that is both uniformly bounded and
strong enough to drive BV convergence, and nothing in the checks suggests one
exists; the energy budget is saturated while the sup norms keep growing.

Against that, the fixed-`M` position (§8) costs nothing to state, is already
verified for classification and trivial for regression, and lands in a regime
the paper already covers. That is a far better use of the remaining time
before submission than a limit theorem that currently has two independent
obstructions.

Nothing in the main inference theory waits on any of this: Theorem 1 consumes
an `L²` rate and nothing else, and the pairing identity (Theorem 3) already
characterises what XGBoost's window effect measures, a finite sum of
piecewise-constant trees being BV. What is unproved — and now looks unlikely
to be provable in the form originally wanted — is only that XGBoost's
**uncorrected plug-in** is gradient-consistent, which is the quantity the paper
argues throughout should not be relied on.

## 10. If it is not proved before submission

Narrow the three places that currently imply coverage:

- the checklist row `Gradient boosting | established | BV, proved here`;
- the abstract's "boosted trees";
- Case (ii) of Theorem S2, which §2 suggests is empty as stated;

and add a sentence to §3.5 recording that XGBoost is covered for inference and
for the pairing identity but not for gradient consistency.
