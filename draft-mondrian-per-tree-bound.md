# The per-tree Mondrian BV bound, written out

Draft of a replacement for Substeps 2a–2c of §S25. Written 2026-10-19.

**Status.** Every step is now proved: Step 0, Lemmas A, B, B′ and C, and the
assembly. Each is independently confirmed numerically. The per-tree bound is
complete, and with Step 1 of §S25 it gives the forest bound with **no condition
on `B`**, replacing an argument that required `B ≫ n^{2(d-α)/(d+2)}`.

The remaining caveat is not mathematical but procedural: **none of this has
been read by anyone but me**, it was written in a day, and I have been wrong
twice in this project in ways that only measurement caught. The places I would
check hardest are Lemma C's Step C3, where everything turns on
`E[T^{-3/2}] = Γ(1/2)` being finite, and Lemma B′'s use of projectivity to get
the truncated cell law.

Companion to `notes-mondrian-bv-bound.md`, which gives the background and the
scaling heuristic this makes precise.


## Corrections after the second referee pass (2026-10-19)

The supplement is now ahead of this note. Three defects were found and fixed
there; this note has not been rewritten, so read §S25 of `jrssb-supp.tex` as
authoritative.

1. **Step 3's signal bound was stated wrongly.** It bounded the jump by
   `L·diam(C_k)`, the ancestor cell, and then used the *leaf* diameter in the
   section argument. Those are not interchangeable. The correct pointwise bound
   is `|f̄_l − f̄_m| ≤ L(diam(R_l) + diam(R_m))`, from the fact that every
   sample point in a leaf lies within that leaf's diameter of a boundary point
   of it — which is tighter than what was written and is what the section
   representation actually consumes. The conclusion is unchanged.

2. **Empty leaves were not handled.** With `ȳ_l = 0` on an empty leaf, `f̄_l`
   bears no relation to `f` near the face, so the Lipschitz bound does not
   apply there and a jump can be of size `‖f‖_∞`. A separate term `S_0` now
   collects those faces and is bounded by `2‖f‖_∞ √π(√π/2)^{d-1} λ^{1+d/2}
   n^{-1/2}` — the same order as the noise term — using
   `P(N_l = 0) ≤ e^{-n·vol(l)}` and `e^{-x} ≤ x^{-1/2}`.

   This mattered more than expected. Measured at the minimax budget,
   **24–44% of Mondrian leaves are empty** (41.6%, 44.4%, 32.1%, 24.5% at
   `n = 500, 2000, 8000, 32000`), so these faces are common, not a negligible
   tail. What saves the term is not their scarcity but their *geometry*: the
   quantity `Σ_{empty} vol(l)/s_1(l)` that the bound controls comes out at
   `0.29, 0.52, 0.26, 0.26`, flat in `n` and comfortably inside the constant
   bound of `1.23`.

3. **The boosting theorem needed uniformity in `n`.** `S_n` acts on `ℝ^n`, so a
   spectral gap for each `n` does not bound `sup_n (1-ρ_n)^{-1}`, which is what
   the geometric series produces, nor is the `L² → W^{2,p}` operator norm
   automatically uniform. Both are now explicit hypotheses, with a remark
   noting that they fail for a smoothing spline at fixed smoothing parameter
   (where `ρ_n → 1`) and hold for `S_n = εI + (1-ε)S_n^0`.

---

## Setup

`Ω = [0,1]^d`. A Mondrian partition `P ~ MP(λ, Ω)` is drawn independently of
the data: in a cell `C` with side lengths `s = (s_1,…,s_d)` and linear
dimension `a = |s|_1 = Σ_j s_j`, a clock `E ~ Exp(a)` is drawn; if `E` exceeds
the remaining budget the cell is a leaf, otherwise dimension `j` is chosen with
probability `s_j/a`, the cut position is uniform on the `j`-th side, and both
children recurse with budget reduced by `E`.

The tree is `T = Σ_l ȳ_l 1_{R_l}` over leaves `{R_l}`, with
`ȳ_l = N_l^{-1} Σ_{i∈R_l} Y_i` when `N_l ≥ 1` and `ȳ_l = 0` otherwise. Write
`Y_i = f(X_i) + ξ_i` with `E[ξ_i | X_i] = 0`, `Var(ξ_i | X_i) ≤ σ²`, and
`X_i` uniform on `Ω`. Assume `f` is Lipschitz with constant `L`.

Internal nodes are indexed by `k`; node `k` splits cell `C_k` along a face
`F_k` of area `H^{d-1}(F_k)`.

**Goal.** `sup_n E[|DT|(Ω)] < ∞` at the minimax budget `λ_n ≍ n^{1/(d+2)}`.

## Step 0: the jump-set representation — **proved**

> For `T` piecewise constant on the leaves of a hierarchical axis-aligned
> partition whose cuts are strictly interior to their cells,
>
>     |DT|(Ω) = Σ_k ∫_{F_k} |Δ_k(u)| dH^{d-1}(u),
>
> where `F_k = C̄_k ∩ \{x_{j_k} = t_k\}` is the split face at internal node `k`
> and `Δ_k(u)` is the difference of the two leaf values meeting at `u`.

Lemma~S3 (`lem:bv_existence`) already supplies the analytic content: for a
piecewise constant `T` with leaves `{R_l}` and values `{c_l}`,

    DT = Σ_{l ∼ m} (c_m - c_l) ν_{lm} H^{d-1}⌊(∂R_l ∩ ∂R_m),

the sum running over adjacent leaf pairs. What remains is to re-index that sum
by internal nodes rather than leaf pairs, which needs two facts.

**(i) Each leaf-pair interface lies in the face of their lowest common
ancestor.** Let `l, m` be adjacent leaves and `k` their lowest common ancestor.
Then `l` and `m` lie in opposite children of `k`, so, writing `C_k^-` and
`C_k^+` for those children, `R_l ⊆ C_k^-` and `R_m ⊆ C_k^+` after relabelling.
Taking closures, every `x ∈ ̄R_l` has `x_{j_k} ≤ t_k` and every `x ∈ ̄R_m` has
`x_{j_k} ≥ t_k`. Any `x ∈ ∂R_l ∩ ∂R_m` lies in both closures, so `x_{j_k} = t_k`;
and `x ∈ ̄R_l ⊆ ̄C_k`. Hence `∂R_l ∩ ∂R_m ⊆ F_k`. ∎

**(ii) Distinct faces meet in an `H^{d-1}`-null set.** Take internal nodes
`k ≠ k′`. If neither is an ancestor of the other they lie in opposite subtrees
of their own lowest common ancestor, so `C_k ∩ C_{k′} = ∅` and the faces are
disjoint. Otherwise say `k` is a strict ancestor of `k′`, so `C_{k′}` lies in
one child of `k`. If `j_{k′} ≠ j_k` the two hyperplanes are transverse and meet
in dimension `d-2`, which is `H^{d-1}`-null. If `j_{k′} = j_k` they are parallel,
and distinct: `t_{k′}` is strictly interior to the extent of `C_{k′}` in that
coordinate, while `C_{k′}` lies entirely on one side of `t_k`, so
`t_{k′} ≠ t_k`. ∎

Combining, the map sending an adjacent leaf pair to its lowest common ancestor
sorts the interfaces into the faces, with no interface counted twice by (ii),
and conversely every point in the relative interior of `F_k` lies between the
leaf of `C_k^-` and the leaf of `C_k^+` containing it, whose lowest common
ancestor is `k`. The correspondence is therefore a bijection up to
`H^{d-1}`-null sets, and regrouping Lemma~S3's sum gives the display. ∎

Three remarks. The argument is **purely deterministic** — no probability, no
Mondrian structure, any dimension — so it also covers the boosted-tree ensembles
of the companion note, where the same representation was used. The only
hypothesis is that cuts fall strictly inside their cells, which holds almost
surely for the Mondrian since cut positions are uniform on an open interval,
and which is needed in (ii): a cut on a cell boundary would produce an empty
child and a face coinciding with an ancestor's. And this closes the one item
that was previously only verified numerically; `check-tree-faces.py`, which
confirmed (i) over random partitions in `d ≤ 4`, is now corroboration rather
than evidence.

## Lemma A (expected interface area) — **proved**

> For `P ~ MP(λ, C)` on a box `C`,
> `E[ Σ_k H^{d-1}(F_k) ] = d · λ · vol(C)`.
> In particular the total interface area of `MP(λ, [0,1]^d)` has mean `dλ`.

*Proof.* Write `A(C,λ)` for the left side, `s` for the sides of `C`,
`a = |s|_1`, `V = vol(C)`. Conditioning on the first cut,

    A(C,λ) = ∫_0^λ a e^{-at} Σ_j (s_j/a) E_U[ V/s_j + A(C_L,λ−t) + A(C_R,λ−t) ] dt,

since a cut of `C` in dimension `j` has face area `Π_{m≠j} s_m = V/s_j`
whatever the cut position. Try `A(C,λ) = dλV`. The children satisfy
`vol(C_L) + vol(C_R) = V` for every `j` and every cut position, so the
induction term is `d(λ−t)V` regardless of `U` and `j`, and the first term is
`Σ_j (s_j/a)(V/s_j) = dV/a`. Hence

    A(C,λ) = dV ∫_0^λ e^{-at} dt + dV a ∫_0^λ e^{-at}(λ−t) dt
           = dV (1−e^{-aλ})/a + dV a [ λ/a − (1−e^{-aλ})/a² ]
           = dV λ. ∎

The two `(1−e^{-aλ})/a` terms cancel exactly, which is why the answer is free
of the cell's shape. Confirmed numerically by `check-mondrian-interface.py`:
measured/predicted ratios `0.990`–`1.013` for `d = 2,3,4` and `λ = 1,2,4`,
with leaf counts matching the known `(1+λ)^d`.

## Step 1: signal/noise split

With `f̄_l = N_l^{-1} Σ_{i∈R_l} f(X_i)` and `ε_l = N_l^{-1} Σ_{i∈R_l} ξ_i`
(both `0` when `N_l = 0`), `ȳ_l = f̄_l + ε_l`, so `|Δ_k| ≤ |Δ_k^{sig}| + |Δ_k^{noi}|`
and

    |DT|(Ω) ≤ S + N,    S = Σ_k ∫_{F_k}|Δ^{sig}|,   N = Σ_k ∫_{F_k}|Δ^{noi}|.

## Step 2: reduce both terms to sections

Fix the coordinate `1` (the argument is identical for each). Along the line
`ℓ(x_{-1}) = {(t, x_{-1}) : t ∈ [0,1]}` the tree is a step function whose jumps
occur at the `1`-perpendicular cuts, so

    |D_1 T|(Ω) = ∫_{[0,1]^{d-1}} Σ_{jumps on ℓ} |Δ| dx_{-1}.

Each jump is between two leaves crossed by `ℓ`, and each leaf `l` is crossed on
a set of lines of measure `vol(l)/s_1(l)`. Hence for either part,

    Σ_k ∫_{F_k}|Δ| ≤ 2 Σ_l (sup over the jump at l) · vol(l)/s_1(l).      (★)

## Lemma B (aspect ratio) — **proved**, from a published cell law

> Let `l(X)` be the Mondrian cell containing a uniform point `X`. Then for
> `m ≠ 1`, `E[ s_m(l(X)) / s_1(l(X)) ] = 2`, and consequently
> `E[ diam(l(X)) / s_1(l(X)) ] ≤ 2d − 1`.

*Proof.* Mourtada, Gaïffas & Scornet establish that for `MP(λ)` on `R^d` the
cell containing a fixed point is `Π_j [−E_j^-, E_j^+]` with the `2d` variables
iid `Exp(λ)`. So `s_j = E_j^- + E_j^+ ~ Gamma(2, λ)`, independent across `j`.
For `X ~ Gamma(2,λ)`, `E[1/X] = ∫_0^∞ λ²x e^{-λx} x^{-1} dx = λ`, so by
independence `E[s_m/s_1] = E[s_m] E[1/s_1] = (2/λ)·λ = 2`. Finally
`diam ≤ |s|_1 = Σ_m s_m`, and the `m = 1` term contributes `1`. ∎

**Numerical check** (`check-mondrian-aspect.py`). Sampling the unbounded law
directly gives `E[s_2/s_1] = 2.00` to two places across `d = 2,3,4` and
`λ = 4,8,16` (100,000 draws each), confirming the identity.

### Lemma B′ (the boundary case on `[0,1]^d`) — **proved**

> On the unit cube, `E[ diam(l(X)) / s_1(l(X)) ] ≤ 2d - 1 + 4(d-1)/λ`.

The concern is real: truncation at the boundary shrinks `s_1` as well as `s_m`,
and shrinking the denominator inflates the ratio. It costs a constant and
nothing more.

*B′1: the truncated cell law.* The Mondrian is projective, so `MP(λ,[0,1]^d)`
is `MP(λ,ℝ^d)` restricted to the cube, and restricting a partition intersects
every cell with the sub-box. Hence

    s_j = min(E_j^-, X_j) + min(E_j^+, 1 - X_j),    E_j^± iid Exp(λ).

Since `X` is uniform on the cube its coordinates are independent, and the
`E_j^±` are independent across `j`, so **`s_m` and `s_1` remain independent**
and `E[s_m/s_1] = E[s_m]·E[1/s_1]`. Truncation only shrinks, so
`E[s_m] ≤ 2/λ` as before. Everything reduces to `E[1/s_1]`.

*B′2: the inverse moment.* Write `1/s = ∫_0^∞ e^{-rs} dr`. For `E ~ Exp(λ)`
and a cap `u`,

    E[ e^{-r·min(E,u)} ] = [ λ + r e^{-(λ+r)u} ] / (λ+r).

Given `X_1`, the two sides of the cell are independent, so `E[e^{-rs} | X_1]`
is the product of that expression at `u = X_1` and at `u = 1-X_1`. Integrating
over `X_1 ~ U(0,1)` first turns the cross terms into
`λ r(1-e^{-(λ+r)})/(λ+r)`, leaving

    E[1/s_1] = ∫_0^∞ [ λ² + 2λ r(1-e^{-(λ+r)})/(λ+r) + r² e^{-(λ+r)} ] / (λ+r)² dr.

The three pieces are elementary: `λ²∫(λ+r)^{-2} = λ`;
`2λ∫ r(λ+r)^{-3} = 1` exactly; and the last is at most
`e^{-λ}∫_0^∞ e^{-r} dr = e^{-λ}` since `r²/(λ+r)² ≤ 1`. So

    E[1/s_1] ≤ λ + 1 + e^{-λ}.

*B′3: assembly.* `E[s_m/s_1] ≤ (2/λ)(λ + 1 + e^{-λ}) = 2 + 2(1+e^{-λ})/λ
≤ 2 + 4/λ` for `λ ≥ 1`, and summing over `m` with the `m=1` term equal to `1`
gives the claim. ∎

**The boundary costs an additive constant, not a rate.** The unbounded value is
`E[1/s_1] = λ` exactly; the cube value exceeds it by at most `1 + e^{-λ}`. That
is the `O(1/λ)`-thick boundary layer contributing `O(1)` to an integral of size
`λ`, which is what one would hope. Since `λ_n = n^{1/(d+2)} → ∞`, the constant
converges to the unbounded `2d-1`, and for any `λ ≥ 1` it is at most `6d-5`.

**Numerical check** (`check-mondrian-boundary.py`). The `E[1/s_1]` bound holds
at every `λ` tested and is tight: quadrature against the bound gives ratios
`0.84, 0.96, 0.995, 0.998, 0.996, 0.992` for `λ = 1,2,4,8,16,32`, with Monte
Carlo agreeing with the quadrature throughout. The resulting Lemma B′ constants
also bound the direct cube measurements reported above — for instance `7.75`
against a measured `7.25` at `d=4, λ=16`.

*(An aside on those direct measurements: `s_m/s_1` has infinite variance, since
`E[1/s_1²] = ∞` — near the boundary `s_1 ≈ X_1 + Exp(λ)` and `∫ dX_1/X_1`
diverges. The mean exists and is all the argument uses, but no sample mean of
it stabilises quickly, which is why the cube figures wander. The quadrature
above avoids the issue entirely, which is why it is the reliable check.)*

## Step 3: the signal term — **proved**

For two leaves meeting at `u`, both lie in the cell `C_k` being split, so
`|Δ^{sig}| ≤ L · diam(C_k) ≤ L · (diam(l) + diam(l'))`. Applying (★),

    E[S_1] ≤ 2L · E[ Σ_l diam(l) · vol(l)/s_1(l) ]
           = 2L · E[ diam(l(X)) / s_1(l(X)) ]          (★★)
           ≤ 2L (2d − 1),

where (★★) is the observation that `Σ_l g(l) vol(l)` is exactly
`E_X[g(l(X))]` for a uniform `X`. **No rate in `n` or `λ` enters.** This is
what the note asserted without proof: the signal part is `O(1)` because the
discrete variation of a Lipschitz function over a refining partition is
controlled by the partition's aspect ratio, not by how fine it is.

## Lemma C (the noise term) — **proved**

> `E[ Σ_l E[|ε_l| \mid P] · vol(l)/s_1(l) ] ≤ 2σ C_d · λ^{1+d/2} n^{-1/2}`,
> with `C_d = √(2π)(√π/2)^{d-1} + π^{(2+d)/6}`.
> At `λ_n ≍ n^{1/(d+2)}` the right side is a constant.

*Step C1: a binomial bound.* Given `P`, `N_l ~ Bin(n, vol(l))` and
`E[|ε_l| \mid N_l] ≤ σ N_l^{-1/2}` for `N_l ≥ 1`, so with `μ = n·vol(l)` we need
`ψ(n,p) := E[N^{-1/2} 1\{N ≥ 1\}]`. Two bounds:

- `ψ ≤ P(N ≥ 1) ≤ μ`, since `N^{-1/2} ≤ 1` on `\{N ≥ 1\}`;
- `N^{-1/2} ≤ √2 (N+1)^{-1/2}` on `\{N ≥ 1\}`, and by Jensen
  `E[(N+1)^{-1/2}] ≤ (E[(N+1)^{-1}])^{1/2}`, while the classical identity
  `E[(N+1)^{-1}] = (1-(1-p)^{n+1})/((n+1)p) ≤ 1/μ` gives `ψ ≤ √2\,μ^{-1/2}`.

So `ψ ≤ min(√2\,μ^{-1/2}, μ)`.

*Step C2: change of variables.* By the identity `Σ_l g(l)vol(l) = E_X[g(l(X))]`
and the cell law of Lemma B, write `S_j = T_j/λ` with `T_j` iid `Gamma(2,1)`,
`W = Π_j T_j`, and `M := nλ^{-d}`. Then `vol(l) = λ^{-d}W`, `μ = MW`,
`1/s_1 = λ/T_1`, and the quantity to bound is

    λ · E[ ψ_μ / T_1 ],     ψ_μ = min(√2 μ^{-1/2}, μ).

Since `λ M^{-1/2} = λ^{1+d/2} n^{-1/2}`, the claim is exactly
`E[ψ_μ/T_1] ≤ C_d M^{-1/2}`.

*Step C3: the bulk, `μ ≥ 1`.* Use `ψ ≤ √2μ^{-1/2}` and independence:

    E[ψ/T_1 ; μ≥1] ≤ √2 M^{-1/2} E[T_1^{-3/2}] ∏_{j≥2} E[T_j^{-1/2}]
                      = √2 M^{-1/2} Γ(1/2) Γ(3/2)^{d-1}
                      = √(2π) (√π/2)^{d-1} M^{-1/2}.

**The crux is that `E[T^{-3/2}] = Γ(1/2) = √π` is finite.** A `Gamma(2,1)`
density vanishes linearly at the origin, so `t^{-3/2}·t = t^{-1/2}` is
integrable — the `-3/2` power is exactly what the `1/s_1` from the section
representation and the `μ^{-1/2}` from the binomial together demand, and it is
the last moment that exists. Had the Mondrian cell law been `Exp` rather than
`Gamma(2,·)` per side, this step would fail.

*Step C4: the tail, `μ < 1`.* Use `ψ ≤ μ = MW`, and on the event `W < 1/M`
bound `MW ≤ 1`:

    E[ψ/T_1 ; μ<1] ≤ E[T_1^{-1} 1\{W < 1/M\}]
                     ≤ (E[T_1^{-3/2}])^{2/3} · P(W < 1/M)^{1/3}   (Hölder, 3/2 and 3)
                     = π^{1/3} P(W < 1/M)^{1/3}.

A Chernoff bound on the Mellin transform gives, for any `s ∈ (0,2)`,
`P(W<w) ≤ w^s E[W^{-s}] = w^s Γ(2-s)^d`; at `s = 3/2`,
`P(W < 1/M) ≤ M^{-3/2} π^{d/2}`. Hence the tail is at most
`π^{1/3}·π^{d/6} M^{-1/2} = π^{(2+d)/6} M^{-1/2}`.

Adding C3 and C4 gives `E[ψ_μ/T_1] ≤ C_d M^{-1/2}`. ∎

**Numerical check** (`check-mondrian-lemmaC.py`). The moment facts come out at
`E[T^{-1/2}] = 0.8867` against `Γ(3/2) = 0.8862` and
`E[T^{-3/2}] = 1.7785` against `Γ(1/2) = 1.7725`. The bound holds at every
`(d, M)` tested, `d ∈ \{2,3,4\}` and `M` from `10` to `10⁴`, with realised /
bound between `0.24` and `0.50`. The ratio **stabilises** as `M` grows
(`d=2`: `0.351, 0.448, 0.496, 0.502`), so the `M^{-1/2}` rate is sharp and only
the constant is loose. The Chernoff bound holds in every cell, by one to two
orders of magnitude.

**Why the small-cell worry dissolved.** The concern was cells with
`n·vol(l) = O(1)`, where `ψ` behaves like `μ` rather than `μ^{-1/2}`. Step C4
handles them without needing them to be rare: the `ψ ≤ μ` branch is *stronger*
than `μ^{-1/2}` there, and all that is required is that the event `W < 1/M`
carry little enough mass, which the Mellin bound supplies with room to spare.
No concentration machinery and no appeal to Mourtada et al.'s risk analysis
turned out to be necessary.

## Step 5: assembly

With Lemma C,

    E|D_1 T|(Ω) ≤ E[S_1] + E[N_1]
                ≤ 2L(2d−1 + 4(d−1)/λ) + 2σ C_d λ^{1+d/2} n^{-1/2},

uniformly bounded at `λ_n ≍ n^{1/(d+2)}`. Summing the `d` coordinates gives
`sup_n E[|DT|(Ω)] < ∞`, and Step 1 of §S25 — `|Df̂_n|(Ω) ≤ B^{-1} Σ_b |DT_b|(Ω)`
by convexity of total variation — carries it to the forest with **no condition
on `B`**.

## What this replaces

Substeps 2a–2c of §S25 in their entirety: the second moment, the independence
factorisation of the cross terms, and the per-term bound that discarded their
sum. All three of the referee's objections are to machinery that this argument
does not use. The condition `B ≫ n^{2(d−α)/(d+2)}` disappears with them.

## Honest summary

| piece | status |
|---|---|
| Step 0, jump-set representation | **proved**; deterministic, any `d`; numerics now corroboration |
| Lemma A, interface area `= dλ·vol` | **proved**; confirmed to ~1% |
| Lemma B, aspect ratio `≤ 2d−1` | **proved** from a published cell law |
| Lemma B′, the same on `[0,1]^d` | **proved**; boundary costs `+4(d−1)/λ`, an additive constant, not a rate |
| Step 3, signal `= O(1)` | **proved**, given Lemma B |
| Lemma C, noise `≲ λ^{1+d/2}n^{-1/2}` | **proved**; rate confirmed sharp, constant loose by 2–4× |
| Step 5, assembly and the forest | immediate from Step 1 of §S25 |

Nothing is outstanding. Set against the boosting analogue, where four
independent routes closed and measurement said the claim itself was false, the
forest claim was true and the proof turned out to be shorter than the one it
replaces.
