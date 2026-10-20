# Repairing the Mondrian forest BV bound (Theorem S3 / §S25)

Status: **proposed route, not yet a proof.** The scaling calculation in §3 below
is heuristic — it uses typical values where the real argument needs expectations
over the Mondrian process. §4 lists exactly what is missing. Nothing here should
go into the supplement until §4 is discharged.

Written 2026-10-19, in response to the referee note on the cross-term argument.

---

## 1. What has to be shown

`\fn = (1/B) Σ_{b≤B} T_b`, each `T_b` piecewise constant on an independent
Mondrian partition `MP(λ_n, [0,1]^d)` with budget `λ_n ≍ n^{1/(d+2)}`. The
well-behaved-in-BV condition needs

    sup_n E[ |D\fn|(Ω) ] < ∞.

## 2. Why the current argument fails, and why it is also unnecessary

**The three defects** (all correctly identified by the referee):

1. *Substep 2b factors an expectation over trees that are not independent.* The
   partitions are independent of each other and of the data, but `T_b` depends on
   the partition **and** on the shared responses, so `∫φ·d(DT_b)` and
   `∫φ·d(DT_b')` are dependent. This is exactly the between-tree correlation that
   stops forest variance from vanishing as `B → ∞`.
2. *The cross terms are discarded by a per-term bound.* Each expected jump is
   `O(D_λ^α)`, but the number of interfaces grows like `λ^d`. A bound on each
   term says nothing about the sum, and on the accounting in §3 the sum of the
   discarded terms does not obviously vanish.
3. *Substep 2a treats `|·|(Ω)` as a linear functional.* It pulls "a suitable test
   function φ with ‖φ‖∞ ≤ 1" out of `|μ|(Ω) = sup_{‖φ‖≤1} ∫φ dμ` and then squares
   and takes expectations as though φ were fixed. The maximising φ is
   realisation-dependent.

**The deeper problem is that none of this machinery is needed.** Step 1 of the
existing proof already gives, by convexity of total variation,

    |D\fn|(Ω) ≤ (1/B) Σ_b |DT_b|(Ω),

so it suffices to bound **a single tree**: `sup_n E[|DT|(Ω)] < ∞`. The proof
dismisses that route in one sentence — *"the triangle inequality applied to the
forest average is too crude because individual tree BV norms grow with n"* — and
the diagonal bound it computes does grow, reaching `O(n^{2(d-α)/(d+2)})`. But
that bound drops the interface-area factor: it bounds

    |DT|(Ω) = Σ_{j∼k} |ȳ_k − ȳ_j| · H^{d-1}(∂R_j ∩ ∂R_k)

by `K · sup|jump|`, discarding every `H^{d-1}`. Those areas are `≈ λ^{-(d-1)}`,
which is most of the mass.

**Corroborating symptom.** The existing route ends up requiring
`B ≫ n^{2(d-α)/(d+2)}` — the forest must grow polynomially in `n`, at a rate
depending on the dimension and on the smoothness of `f`. No implementation does
that; forests are grown to a fixed size. (The paper does not simulate Mondrian
forests — they are theory-only here, the simulations use Breiman forests — so
this is not a conflict with the reported experiments, but it does mean the
theorem as proved covers no forest anyone would actually fit.) A per-tree bound
needs no condition on `B` at all, and covers `B = 1`.

## 3. The proposed per-tree bound

Fix one tree. With `N_j` the count in leaf `R_j`, write
`ȳ_j = f̄_j + ε_j`, where `f̄_j = N_j^{-1} Σ_{i∈R_j} f(X_i)` and
`ε_j = N_j^{-1} Σ_{i∈R_j} (Y_i − f(X_i))`. Then

    |DT|(Ω) ≤ Σ_{j∼k} |f̄_k − f̄_j| H^{d-1}(·)   +   Σ_{j∼k} |ε_k − ε_j| H^{d-1}(·)
                └──────── signal ────────┘         └───────── noise ─────────┘

**Signal.** `Σ_{j∼k} |f̄_k − f̄_j| H^{d-1}(∂R_j ∩ ∂R_k)` is a discrete total
variation of `f` along coordinate `j` over the partition. For `f ∈ W^{1,1}` it is
bounded by (and converges to) `|D_j f|(Ω) ≤ |Df|(Ω) < ∞`, uniformly in the
partition. **O(1), no rate needed.**

**Noise.** Using typical values — this is the heuristic step:

| quantity | scale |
|---|---|
| number of interfaces (= number of splits `K`) | `λ^d` |
| area of one interface | `λ^{-(d-1)}` |
| points per leaf | `n/λ^d` |
| `E|ε_k − ε_j|` ≲ `sd` | `(λ^d/n)^{1/2}` |

    noise ≈ λ^d · λ^{-(d-1)} · (λ^d/n)^{1/2} = λ^{1+d/2} · n^{-1/2}.

At `λ_n ≍ n^{1/(d+2)}` the exponent of `n` is

    (1 + d/2)/(d+2) − 1/2 = [(2+d)/2]/(d+2) − 1/2 = 1/2 − 1/2 = 0,

so the noise term is **O(1)** as well.

**The budget is exactly the balancing rate.** `λ_n ≍ n^{1/(d+2)}` is the
minimax-optimal Mondrian budget, and it is precisely the rate at which signal and
noise contributions to the total variation are both `O(1)`. That is the same
bias–variance balance that produces the minimax `L²` rate, read in the BV norm —
which is why the result should be true, and true per tree.

## 3a. Empirical check of the per-tree claim — it holds

Run 2026-10-19, `check-mondrian-bv.py`. The Mondrian process is implemented
directly (Roy & Teh: in a cell with side lengths `s`, draw `E ~ Exp(Σs)`; if the
budget is spent, stop; else split dimension `j` with probability `∝ s_j` at a
uniform point). `d = 4`, budget `λ_n = n^{1/6}`, target `2x₁+3x₂` on `[0,1]⁴`,
so `|D₁f|(Ω) = 2` exactly. `|D₁·|(Ω)` is measured by the same one-dimensional
section method used in the XGBoost notes. The signal and noise parts are
separated by fitting the **same partition** three times — to `y`, to `f(X)`, and
to the noise `y − f(X)`.

| `n` | leaves | TV, one tree | signal | noise | TV, forest `B=50` |
|---|---|---|---|---|---|
| 500 | 180 | 3.68 | 2.86 | 1.45 | 3.54 |
| 2,000 | 553 | 4.48 | 3.71 | 1.72 | 3.56 |
| 8,000 | 768 | 3.50 | 2.91 | 1.31 | 3.70 |
| 32,000 | 1,666 | 3.57 | 2.90 | 1.64 | 3.58 |

Growth exponents over that 64-fold range of `n`, from two independent runs:

| quantity | run A | run B |
|---|---|---|
| single tree TV | `n^{-0.02}` | `n^{-0.01}` |
| signal part | `n^{+0.00}` | `n^{+0.00}` |
| noise part | `n^{+0.01}` | `n^{+0.03}` |
| forest TV (`B=50`) | — | `n^{+0.00}` |

**The claim of §3 holds, and both of its predicted components hold
separately.** A single Mondrian tree's expected total variation is flat in `n`,
and so is the forest's, which is what Step 1's convexity bound requires and all
it requires.

The informative column is `leaves`: the partition refines by a factor of ten
across this range (`n^{+0.54}`, against the predicted `λ^d = n^{2/3}`), so the
number of interfaces grows by an order of magnitude while the total variation
does not move. That is precisely the balance §3 argues for — more faces, each
carrying a smaller jump over a smaller area — and it is the content of the
claim, not an artefact of a short range.

**Consequence for the proof.** Substeps 2a–2c of §S25 are not merely wrong but
unnecessary. Step 1 already gives
`|Df̂_n|(Ω) ≤ B^{-1} Σ_b |DT_b|(Ω)` by convexity, and the per-tree quantity is
bounded. The second moment, the independence factorisation and the cross-term
argument can all be deleted, along with the condition `B ≫ n^{2(d−α)/(d+2)}`
they forced — the measured forest is flat at `B = 50` fixed.

**Caveats, stated plainly.** This is `d = 4`, one target, and `λ` ranges only
over `2.8` to `5.6`, so it is nowhere near asymptotic. The signal part sits at
about `2.9` against the target's `|D₁f| = 2.0`: a finite-`λ` bias, since leaf
means average over the other coordinates and make one-dimensional sections
non-monotone. It is flat, which is what boundedness needs; convergence down to
`2.0` is the *strict* convergence upgrade, which Part II of
Lemma~S4 already flags as requiring a further assumption. And an empirical
bound is not a proof — the gaps in §4 stand. What has changed is that the
effort is now known to be aimed at a true statement, which was not the case for
the boosting analogue.

## 4. What is missing before this is a proof

1. **Expected interface area.** The table uses typical values. The real argument
   needs `E[ Σ_splits Π_{m≠i} s_m ]` over the Mondrian process. The number of
   interfaces equals the number of splits exactly (each split creates one), and
   the Mondrian cuts a cell at rate equal to its linear dimension `Σ_m s_m`, so
   this expectation is plausibly available in closed form from the recursive
   construction. **Needs computing, not assuming.**
2. **Nearly-empty cells — the real obstacle.** `N_j` can be small or zero, making
   `ε_j` huge or undefined, and `E|ε_k − ε_j| ≲ (λ^d/n)^{1/2}` fails in that
   regime. Options: a minimum-leaf-size convention (what implementations do
   anyway), truncation, or showing the small-cell contribution is negligible in
   expectation. Mourtada, Gaïffas & Scornet (2017; 2020) handle the analogous
   difficulty for the `L²` risk — that analysis is the place to start.
3. **Dependence between `N_j` and the responses.** The leaf counts are random and
   not independent of the data; the decomposition above treats them loosely.
4. **The signal step at the stated regularity.** "Discrete TV over a partition
   ≤ `|D_j f|(Ω)`" needs a proof for axis-aligned hyperrectangle partitions at
   `f ∈ W^{1,1}`. The one-dimensional section argument already used in the proof
   of the pairing identity (§S22) is the natural tool — the machinery is in-house.

## 5. Fallbacks, in order of preference

1. **Prove the per-tree bound.** Shortens §S25 and strengthens the theorem
   (no condition on `B`).
2. **State it under a minimum-leaf-size condition.** Discharges obstacle 2 by
   assumption; honest, matches implementations, and still covers `B = 1`.
3. **Downgrade Theorem S3 to a conjecture** — the status Breiman forests already
   carry, so the paper has the precedent and the language. The inference theory
   is untouched either way: Theorem 1 consumes an `L²` rate and nothing else.

## 6. What does not need repairing

- The Sobolev twin (Lemma S2 / §S19) is sound; its hypothesis is already an
  almost sure bound, so its Step 1 merely invokes it.
- The BV consistency proof (§S21) had a genuine but separate defect — promoting
  boundedness in expectation to almost sure boundedness — **repaired 2026-10-19**
  via tightness plus a positive-probability event of bounded realisations.
