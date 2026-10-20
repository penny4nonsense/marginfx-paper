# marginfx-paper

Paper and reproduction code for *Model-Agnostic Interpretation of Machine
Learning via Average Marginal Effects*, prepared for JRSS-B.

The estimator itself lives in the separate [`marginfx`](https://github.com/penny4nonsense/marginfx)
package. A copy is vendored here under `marginfx/` because every script in
this repository does `sys.path.insert(0, PAPER_DIR)` and therefore imports the
vendored copy, never an installed one. **After any change to the package,
copy `marginfx/marginfx/*.py` and `engines/*.py` across before re-running
anything** — skipping that step silently produces numbers from the old code.

## The paper

| File | What it is |
|---|---|
| `jrssb-main.tex` | main text |
| `jrssb-supp.tex` | supplement (extensions, two of the three applications, full simulation results, all proofs) |
| `references.bib` | shared bibliography |
| `build-jrssb.ps1` | compiles both, in the required order |

Build with `.\build-jrssb.ps1`. Order matters: the two documents cross-
reference each other through `xr-hyper`, so each needs the other's `.aux` and
the script runs main → supp → main → supp → main → supp, with `bibtex` after
the first pass. It reports the page count and any unresolved reference.

Cross-references use the `M-` and `S-` prefix convention: inside the
supplement, `\ref{M-thm:main}` points at the main text; inside the main text,
`\ref{S-sec:extras}` points at the supplement.

LaTeX build artifacts are not tracked. The two compiled PDFs are, since they
are the submission deliverable.

## Reproducing the results

Needs Python 3.11 with numpy, pandas, scipy, scikit-learn, pyarrow, xgboost,
shap and tensorflow. No system interpreter here carries that stack; build a
venv outside this folder (Dropbox locks `site-packages` mid-write, which
leaves a half-deleted venv behind if you build one inside).

### Empirical applications

Order matters, and two steps fail silently if skipped.

1. Sync the vendored package, as above.
2. **Delete `empirical/*/results/*_ames_partial.parquet`.** The AME loop
   resumes from the partial file and skips every specification/model pair
   already in it, so without this a re-run does nothing at all. Keep the
   `_shap` and `_pdp` parquets — those are skipped when present, which is
   what you want.
3. Run the three analyses. They are independent and can go in parallel:
   ```
   python empirical/adult/analyze_adult.py
   python empirical/credit_default/analyze_credit_default.py
   python empirical/ames_housing/analyze_ames_housing.py
   ```
   Roughly 42, 15 and 6 minutes. Adult is dominated by its neural network, at
   about 10 minutes per specification against 18 seconds for logistic, forest
   and XGBoost combined. These scripts buffer stdout when redirected, so a
   piped log looks frozen for the whole run; check progress by reading the
   partial parquet instead.

### Simulations

```
python simulations/sim1_ame_recovery/run_regression.py
python simulations/sim1_ame_recovery/run_classification.py
python simulations/sim2_se_calibration/run_calibration.py
python simulations/sim2_se_calibration/run_calibration_regression.py
python simulations/sim3_debiased/run_debiased.py
```

Simulation 1 measures the bias and RMSE of the uncorrected plug-in window
average; Simulation 2 the coverage of the refitting bootstrap applied to that
plug-in; Simulation 3 both quantities for the debiased cross-fitted estimator,
holding designs, learners and windows fixed so the two estimators differ only
in the orthogonal correction. The `.ps1` scripts in `simulations/` run them as
a queue.

### Tables

```
python generate_tables.py     # results -> tables/*.tex and tables/*.parquet
python sync-tables.py         # tables/*.tex -> the two .tex files
```

Tables are pasted inline in the paper rather than `\input`, so `sync-tables.py`
replaces just the `\begin{tabular}`-through-`\end{tablenotes}` span of each,
matched by `\label`, preserving the wrappers and indentation around it.
`--check` dry-runs. It targets both documents, since the credit-default model
comparison sits in the main text and the remaining tables in the supplement.

Keras training is not reproducible even with `tf.random.set_seed`, so the
neural network column drifts between runs. Logistic, forest, XGBoost and all
SHAP values reproduce bit for bit, and PDP to 1e-11.

## Layout

| Path | Contents |
|---|---|
| `data/` | download scripts and processed datasets |
| `empirical/` | the three applications: cleaning, analysis, result parquets |
| `simulations/` | the three Monte Carlo studies and their runners |
| `tables/` | generated table bodies, `.tex` and `.parquet` |
| `marginfx/` | vendored copy of the estimator package |
| `verification/` | numerical checks behind the supplement's proofs |
| `tools/` | repository and manuscript checks |
| `notes/` | working notes on the proofs |
| `archive/` | superseded targets: the AoS pair, the ICDM and IEEE versions, the poster, the slides, and the `imsart` class they need |
| `empirical-icdm/`, `tables-icdm/` | results for the archived ICDM version |

## verification/

The supplement's two hard results rest on numerical checks kept here rather
than discarded, because both are claims a reader may reasonably doubt.

- `check-mondrian-*.py` — the per-tree BV bound for Mondrian forests: the
  interface identity, the aspect-ratio and boundary cases, and
  `check-mondrian-truncmoments.py`, which verifies the truncated
  inverse-moment bound by quadrature. The substitution `r = u²` in that one is
  load-bearing: a uniform grid over the singular integrand silently
  overstates the integral for p < 1.
- `check-xgboost-*.py` — why gradient consistency for greedily grown boosting
  is **not** achievable. Four routes are closed here: the edge condition,
  hyperparameter settings, early stopping, and subadditivity. The obstruction
  is in n, not in the number of rounds; an apparent `M^0.3` growth is a
  transient that saturates by M = 16000.
- `check-tree-faces.py` — the node-face representation of a tree's jump set.

## tools/

- `check-consistency.py` — JRSS-B submission requirements and internal
  consistency: abstract length, keyword count and ordering, UK spelling, every
  cross-document and internal reference, and the hypothesis/conclusion
  exponent match on each theorem. Note that Oxford spelling retains `-ize`, so
  `realise` is wrong but `analyse` and `centre` are correct and are
  deliberately not flagged.
- `check-pagebudget.py` — pages per section. Anchors on the printed heading
  number, because matching a heading like "Method" as plain text also hits the
  word in the introduction.
- `check-movability.py` — which labels a block owns and who references them,
  before moving it between documents.
- `check-deps.bat` — LaTeX package availability.

## Known gaps

- `tables/sim3_bias_classification_linear_uniform.tex` is generated but
  embedded in neither document, while the supplement's bounded-support
  discussion cites statistics covering its classification cells. Either the
  table belongs in the supplement or that text should not claim them.
- The data availability statement promises a DOI for this repository. One
  needs minting (Zenodo) before acceptance.
