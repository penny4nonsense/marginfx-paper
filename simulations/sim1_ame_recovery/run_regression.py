"""
run_regression.py
-----------------
Simulation 1: AME Recovery — Regression DGPs.

For each DGP and sample size, runs N_ITER_REGRESSION Monte Carlo iterations.
Each iteration:
    1. Generate data from the DGP
    2. Fit each model
    3. Compute AMEs (no bootstrap — point estimates only)
    4. Compare to ground truth AME

Results saved as parquet files to SIM1_RESULTS_DIR.

Usage:
    python run_regression.py

To change settings (dev vs production, sample sizes, models):
    Edit ../config.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATIONS_DIR = os.path.join(SIM_DIR, '..')
PAPER_DIR = os.path.join(SIM_DIR, '..', '..')
sys.path.insert(0, PAPER_DIR)
sys.path.insert(0, SIMULATIONS_DIR)

# config.py and dgp.py both live in simulations/
from dgp import (
    generate_regression,
    compute_ground_truth_ames,
    FEATURE_NAMES,
)
from config import (
    MODE,
    N_ITER_REGRESSION,
    SAMPLE_SIZES,
    MODELS,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    TF_HIDDEN_UNITS,
    TF_EPOCHS,
    TF_BATCH_SIZE,
    TF_LEARNING_RATE,
    N_GROUND_TRUTH,
    GROUND_TRUTH_SEED,
    BASE_SEED,
    N_JOBS,
    SIM1_RESULTS_DIR,
    GROUND_TRUTH_DIR,
    print_config,
)

import marginfx as mfx


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(model_name: str, outcome_type: str, seed: int):
    """
    Build and return an unfitted model.

    Parameters
    ----------
    model_name : str
        One of 'linear', 'rf', 'xgboost', 'tensorflow'.
    outcome_type : str
        One of 'regression', 'classification'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Unfitted model object.
    """
    if model_name == 'linear':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,  # joblib handles outer parallelism
        )

    elif model_name == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=seed,
            verbosity=0,
        )

    elif model_name == 'tensorflow':
        import tensorflow as tf
        tf.random.set_seed(seed)

        layers = []
        for units in TF_HIDDEN_UNITS:
            layers.append(tf.keras.layers.Dense(units, activation='relu'))
        layers.append(tf.keras.layers.Dense(1))  # regression: no activation

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE),
            loss='mse',
        )
        return model

    else:
        raise ValueError(f"Unknown model: '{model_name}'")


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

def run_one_iteration(
    iteration: int,
    dgp_name: str,
    n: int,
    model_name: str,
    true_ames: dict,
) -> list:
    """
    Run a single Monte Carlo iteration for one DGP / sample size / model.

    Parameters
    ----------
    iteration : int
        Iteration index (used for seeding).
    dgp_name : str
        DGP name.
    n : int
        Sample size.
    model_name : str
        Model name.
    true_ames : dict
        Ground truth AMEs for this DGP.

    Returns
    -------
    list of dict
        One row per feature with columns:
        iteration, dgp, n, model, feature, ame_estimate, true_ame,
        bias, abs_bias, sq_error, elapsed.
    """
    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)

    # Generate data
    X, y = generate_regression(n, dgp_name, rng)

    # Build and fit model
    t0 = time.time()
    model = build_model(model_name, 'regression', seed)

    if model_name == 'tensorflow':
        model.fit(
            X, y,
            epochs=TF_EPOCHS,
            batch_size=TF_BATCH_SIZE,
            verbose=0,
        )
    else:
        model.fit(X, y)

    # Compute AMEs — no bootstrap, point estimates only
    result = mfx.fit(
        model, X, y,
        feature_names=FEATURE_NAMES,
        n_bootstrap=0,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Collect results — one row per feature
    rows = []
    for feature in FEATURE_NAMES:
        est = result.estimates[feature]
        true_ame = true_ames[feature]
        rows.append({
            'iteration':    iteration,
            'dgp':          dgp_name,
            'n':            n,
            'model':        model_name,
            'feature':      feature,
            'ame_estimate': est,
            'true_ame':     true_ame,
            'bias':         est - true_ame,
            'abs_bias':     abs(est - true_ame),
            'sq_error':     (est - true_ame) ** 2,
            'elapsed':      elapsed,
        })

    return rows


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    dgp_name: str,
    n: int,
    model_name: str,
    true_ames: dict,
) -> pd.DataFrame:
    """
    Run all iterations for one DGP / sample size / model combination.

    Results saved incrementally to parquet after each combination completes.
    If the output file already exists, skips and loads from disk instead.
    """
    os.makedirs(SIM1_RESULTS_DIR, exist_ok=True)
    filename = f"regression_{dgp_name}_n{n}_{model_name}.parquet"
    filepath = os.path.join(SIM1_RESULTS_DIR, filename)

    # Skip if already computed
    if os.path.exists(filepath):
        print(f"  Skipping (already exists): {filename}")
        return pd.read_parquet(filepath)

    print(f"  Running: dgp={dgp_name}, n={n}, model={model_name}...")

    # Run iterations in parallel
    all_rows = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_iteration)(
            iteration=i,
            dgp_name=dgp_name,
            n=n,
            model_name=model_name,
            true_ames=true_ames,
        )
        for i in range(N_ITER_REGRESSION)
    )

    # Flatten list of lists and save
    rows = [row for iteration_rows in all_rows for row in iteration_rows]
    df = pd.DataFrame(rows)
    df.to_parquet(filepath, index=False)

    # Print bias summary
    summary = df.groupby('feature')['bias'].agg(['mean', 'std']).round(4)
    print(f"    Saved {len(df)} rows -> {filename}")
    print(f"    Bias summary:\n{summary.to_string()}\n")

    return df


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def load_or_compute_ground_truth(dgp_name: str) -> dict:
    """
    Load ground truth AMEs from disk or compute and save them.

    Ground truth is computed once at N_GROUND_TRUTH observations
    and reused across all sample sizes and models.
    """
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    gt_file = os.path.join(
        GROUND_TRUTH_DIR, f"regression_{dgp_name}_ground_truth.parquet"
    )

    if os.path.exists(gt_file):
        print(f"  Loading ground truth: {gt_file}")
        df = pd.read_parquet(gt_file)
        return dict(zip(df['feature'], df['true_ame']))

    print(f"  Computing ground truth (n={N_GROUND_TRUTH:,})...")
    t0 = time.time()
    true_ames = compute_ground_truth_ames(
        dgp_name=dgp_name,
        outcome_type='regression',
        n=N_GROUND_TRUTH,
        seed=GROUND_TRUTH_SEED,
    )
    elapsed = time.time() - t0
    print(f"  Ground truth computed in {elapsed:.1f}s: {true_ames}")

    # Save to disk
    df = pd.DataFrame([
        {'feature': k, 'true_ame': v}
        for k, v in true_ames.items()
    ])
    df.to_parquet(gt_file, index=False)

    return true_ames


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print_config()
    print()

    dgp_names = ['linear', 'nonlinear', 'interaction']

    # Regression models — logistic is classification only
    # Replace 'logistic' with 'linear' (linear regression)
    regression_models = []
    for m in MODELS:
        if m == 'logistic':
            continue
        regression_models.append(m)
    if 'linear' not in regression_models:
        regression_models = ['linear'] + regression_models

    total = len(dgp_names) * len(SAMPLE_SIZES) * len(regression_models)
    done = 0
    t_start = time.time()

    for dgp_name in dgp_names:
        true_ames = load_or_compute_ground_truth(dgp_name)

        for n in SAMPLE_SIZES:
            for model_name in regression_models:
                done += 1
                print(f"[{done}/{total}]", end=' ')
                run_simulation(dgp_name, n, model_name, true_ames)

    elapsed = time.time() - t_start
    print(f"\nAll regression simulations complete in {elapsed:.1f}s")
    print(f"Results saved to: {SIM1_RESULTS_DIR}")