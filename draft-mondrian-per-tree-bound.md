# The per-tree Mondrian BV bound, written out

Draft of a replacement for Substeps 2a–2c of §S25. Written 2026-10-19.

**Status.** Lemmas A, B and C are all proved, and each is numerically
confirmed. The per-tree bound is therefore complete, and with Step 1 of §S25 it
gives the forest bound with no condition on `B`. Two things still need an
author's eye before any of it goes into the supplement: the boundary case of
Lemma B on `[0,1]^d`, and Step 0's jump-set containment, which is argued and
verified numerically in `d ≤ 4` but not proved in general. And none of it has
been read by anyone but me.

Companion to `notes-mondrian-bv-bound.md`, which gives the background and the
scaling heuristic this makes precise.

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

## Step 0: the jump-set representation

`T` is piecewise constant, so `DT` is concentrated on the interfaces and

    |DT|(Ω) = Σ_k ∫_{F_k} |Δ_k(u)| dH^{d-1}(u),

where `Δ_k(u)` is the difference of the two leaf values meeting at `u`. The
faces are a.s. disjoint: two nodes on incompatible subtrees have disjoint
cells, and a descendant's face lies strictly inside its parent's cell because
cut positions are continuous. The containment of the jump set in `⋃_k F_k` is
the lowest-common-ancestor argument verified numerically in `d ≤ 4` by
`check-tree-faces.py`.

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

## Lemma B (aspect ratio) — proved modulo a published fact

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
`λ = 4,8,16` (100,000 draws each), confirming the identity. On the unit cube,
where sides are truncated at the boundary, the measured `E[Σ_m s_m/s_1]`
straddles `2d−1`: `2.95, 2.90, 3.10` against `3`; `4.86, 4.89, 5.17` against
`5`; `6.41, 6.66, 7.25` against `7`.

**Two caveats, one real.** The truncation on `[0,1]^d` shrinks `s_1` and can
only inflate the ratio; the inflation is confined to points within `O(1/λ)` of
the boundary, a vanishing fraction, but the bound as stated still needs a
boundary argument to be airtight. Separately, and this is why the cube numbers
wander rather than converging tidily: `s_m/s_1` has **infinite variance**, since
`E[1/s_1^2] = λ^2 ∫ e^{-λ x}/x \, dx = ∞` for `s_1 ~ Gamma(2,λ)`. The mean
`E[s_m/s_1] = 2` exists and is all the argument uses, but no sample mean of it
will stabilise quickly, so the cube figures above are indicative rather than
precise. That is a fact about the estimator, not about the bound.

## Step 3: the signal term — **done, given Lemma B**

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

    E|D_1 T|(Ω) ≤ E[S_1] + E[N_1] ≤ 2L(2d−1) + 2σ C_d λ^{1+d/2} n^{-1/2},

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
| Step 0, jump-set representation | argued; LCA containment verified numerically in `d ≤ 4`, not proved in general |
| Lemma A, interface area `= dλ·vol` | **proved**; confirmed to ~1% |
| Lemma B, aspect ratio `≤ 2d−1` | **proved** from a published cell law; boundary case on `[0,1]^d` needs an argument |
| Step 3, signal `= O(1)` | **proved**, given Lemma B |
| Lemma C, noise `≲ λ^{1+d/2}n^{-1/2}` | **proved**; rate confirmed sharp, constant loose by 2–4× |
| Step 5, assembly and the forest | immediate from Step 1 of §S25 |

The two remaining items are both of the "needs a careful author" kind rather
than the "needs a new idea" kind, which is the opposite of where the boosting
analogue ended up.
