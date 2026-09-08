"""
run_calibration_icdm.py
-----------------------
Simulation 2 for the ICDM camera-ready: bootstrap standard error calibration
on the linear classification design.

This is a separate runner from run_calibration.py because the two papers
bootstrap different things, and the difference is not cosmetic.

    conference paper   the adaptive step size is part of the estimator, so it
                       is recomputed on every resample along with the fit
    journal paper      h is fixed by the definition of the window estimand and
                       is not estimated, so it is held at its observed-sample
                       value across resamples

Both evaluate the replicate at the resample, which is the textbook bootstrap
and what the accepted paper's own code did.

The published table's forest, XGBoost and network panels came from a warm-start
refit path that was broken three ways -- forests were never refit at all,
XGBoost continued to 200 boosting rounds against the base model's 100, and the
network trained for 10 epochs against the base model's 50. Every replicate here
is built by calibration_models.build_fitted_model, the same function that
builds the point estimate, so a replicate and the base fit differ only in their
data.

Counts follow the paper: 500 Monte Carlo iterations and 200 resamples per
iteration, for every learner.

Resumable: finished cells are skipped and partial ones resume from their last
checkpoint. Writes to results_icdm/ so the journal paper's results are
untouched.

    python run_calibration_icdm.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM_DIR)
sys.path.insert(0, os.path.dirname(SIM_DIR))

from dgp import compute_ground_truth_ames, FEATURE_NAMES
from config import (
    BASE_DIR,
    GROUND_TRUTH_DIR,
    N_GROUND_TRUTH,
    GROUND_TRUTH_SEED,
    N_JOBS,
)
from parallel_bootstrap import run_iterations

DGP = 'linear'
OUTCOME = 'classification'

# As stated in the paper's simulation section.
N_ITER = 500
N_BOOTSTRAP = 200
SAMPLE_SIZES = [250, 1000, 5000]
MODELS = ['logistic', 'rf', 'xgboost', 'tensorflow']

RESULTS_DIR = os.path.join(BASE_DIR, 'sim2_se_calibration', 'results_icdm')

# Batch of iterations dispatched at once. Keras gets a smaller batch for the
# usual reason: its workers accumulate memory that is only released when the
# process is replaced, and the pool is torn down between batches.
BATCH = {'tensorflow': 12}
BATCH_DEFAULT = N_JOBS


def ground_truth() -> dict:
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    path = os.path.join(GROUND_TRUTH_DIR,
                        f'{OUTCOME}_{DGP}_ground_truth.parquet')
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return dict(zip(df['feature'], df['true_ame']))
    print(f'  Computing ground truth (n={N_GROUND_TRUTH:,})...', flush=True)
    true_ames = compute_ground_truth_ames(
        dgp_name=DGP, outcome_type=OUTCOME,
        n=N_GROUND_TRUTH, seed=GROUND_TRUTH_SEED,
    )
    pd.DataFrame([{'feature': k, 'true_ame': v}
                  for k, v in true_ames.items()]).to_parquet(path, index=False)
    return true_ames


def run_cell(n: int, model_name: str, true_ames: dict) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = f'calibration_icdm_{DGP}_n{n}_{model_name}.parquet'
    path = os.path.join(RESULTS_DIR, name)
    partial = path.replace('.parquet', '_partial.parquet')

    if os.path.exists(path):
        print(f'  Skipping (already exists): {name}', flush=True)
        return pd.read_parquet(path)

    done, frames = set(), []
    if os.path.exists(partial):
        prev = pd.read_parquet(partial)
        done = set(prev['iteration'].unique())
        frames.append(prev)
        print(f'  Resuming from {len(done)} completed iterations...', flush=True)

    remaining = [i for i in range(N_ITER) if i not in done]
    print(f'  Running: n={n}, model={model_name}, '
          f'{len(remaining)} iterations remaining...', flush=True)

    batch_size = BATCH.get(model_name, BATCH_DEFAULT)
    t0 = time.time()

    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        rows = run_iterations(
            outcome_type=OUTCOME,
            model_name=model_name,
            n=n,
            dgp_name=DGP,
            iterations=batch,
            true_ames=true_ames,
            feature_names=FEATURE_NAMES,
            n_bootstrap=N_BOOTSTRAP,
            n_jobs=N_JOBS,
            trim=False,
            # The conference paper's semantics: bootstrap the whole procedure,
            # adaptive step size included.
            eval_at='resample',
            refit_h=True,
        )
        frames.append(pd.DataFrame(rows))
        merged = pd.concat(frames, ignore_index=True)
        merged.to_parquet(partial, index=False)
        print(f"    {merged['iteration'].nunique()}/{N_ITER} iterations done "
              f'({time.time() - t0:.0f}s)', flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(path, index=False)
    if os.path.exists(partial):
        os.remove(partial)

    cov = df.groupby('feature')['covered'].mean().round(3)
    print(f'    Saved {len(df)} rows -> {name}', flush=True)
    print(f'    Coverage (target 0.95):\n{cov.to_string()}\n', flush=True)
    return df


def main():
    print('=' * 60)
    print('Simulation 2 (ICDM camera-ready): bootstrap SE calibration')
    print(f'  DGP={DGP} ({OUTCOME}), n={SAMPLE_SIZES}, models={MODELS}')
    print(f'  {N_ITER} iterations x {N_BOOTSTRAP} resamples per cell')
    print(f'  eval_at=resample, refit_h=True, trim=False')
    print(f'  -> {RESULTS_DIR}')
    print('=' * 60, flush=True)

    true_ames = ground_truth()
    print(f'  True AMEs: '
          f'{ {k: round(v, 4) for k, v in true_ames.items()} }\n', flush=True)

    # Learner-major, so the cheap panels all land before the expensive one
    # starts. Each Keras cell costs roughly a day at these counts, against
    # minutes for the others, and a partial table with three complete panels
    # is far more useful than three partial ones.
    cells = [(n, m) for m in MODELS for n in SAMPLE_SIZES]
    t0 = time.time()
    for k, (n, m) in enumerate(cells, start=1):
        print(f'[{k}/{len(cells)}]', flush=True)
        run_cell(n, m, true_ames)

    print(f'All ICDM calibration cells complete in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
