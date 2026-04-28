"""
run_classification.py
---------------------
Simulation 1: AME Recovery — Classification DGPs.

For each DGP and sample size, runs N_ITER_CLASSIFICATION Monte Carlo iterations.
Each iteration:
    1. Generate binary classification data from the DGP
    2. Fit each model
    3. Compute AMEs (no bootstrap — point estimates only)
    4. Compare to ground truth AME

Ground truth AMEs for classification are computed via Monte Carlo on
N_GROUND_TRUTH observations using the true sigmoid probability function.
AMEs are in probability units: a one-unit increase in x1 increases
P(y=1) by AME(x1) percentage points on average.

Results saved as parquet files to SIM1_RESULTS_DIR.

Usage:
    python run_classification.py

To change settings (dev vs production, sample sizes, models):
    Edit ../config.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings('ignore')

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
    generate_classification,
    compute_ground_truth_ames,
    FEATURE_NAMES,
)
from config import (
    MODE,
    N_ITER_CLASSIFICATION,
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

def build_model(model_name: str, seed: int):
    """
    Build and return an unfitted classification model.

    Parameters
    ----------
    model_name : str
        One of 'logistic', 'rf', 'xgboost', 'tensorflow'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Unfitted model object.
    """
    if model_name == 'logistic':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed)

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,  # joblib handles outer parallelism
        )

    elif model_name == 'xgboost':
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=seed,
            verbosity=0,
            eval_metric='logloss',
        )

    elif model_name == 'tensorflow':
        import tensorflow as tf
        tf.random.set_seed(seed)

        layers = []
        for units in TF_HIDDEN_UNITS:
            layers.append(tf.keras.layers.Dense(units, activation='relu'))
        layers.append(
            tf.keras.layers.Dense(1, activation='sigmoid')
        )

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE),
            loss='binary_crossentropy',
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
        Ground truth AMEs for this DGP (in probability units).

    Returns
    -------
    list of dict
        One row per feature with columns:
        iteration, dgp, n, model, feature, ame_estimate, true_ame,
        bias, abs_bias, sq_error, elapsed.
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)

    # Generate classification data
    X, y = generate_classification(n, dgp_name, rng)

    # Build and fit model
    t0 = time.time()
    model = build_model(model_name, seed)

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

    print('.', end='', flush=True)
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

    Results saved to parquet after each combination completes.
    If the output file already exists, skips and loads from disk.
    """
    os.makedirs(SIM1_RESULTS_DIR, exist_ok=True)
    filename = f"classification_{dgp_name}_n{n}_{model_name}.parquet"
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
        for i in range(N_ITER_CLASSIFICATION)
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

    Classification ground truth AMEs are in probability units, computed
    via Monte Carlo on N_GROUND_TRUTH observations using the true
    sigmoid probability function — no model involved.
    """
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    gt_file = os.path.join(
        GROUND_TRUTH_DIR, f"classification_{dgp_name}_ground_truth.parquet"
    )

    if os.path.exists(gt_file):
        print(f"  Loading ground truth: {gt_file}")
        df = pd.read_parquet(gt_file)
        return dict(zip(df['feature'], df['true_ame']))

    print(f"  Computing ground truth (n={N_GROUND_TRUTH:,})...")
    t0 = time.time()
    true_ames = compute_ground_truth_ames(
        dgp_name=dgp_name,
        outcome_type='classification',
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

    # Classification models — linear regression is regression only
    # Replace 'linear' with 'logistic' if not already present
    classification_models = []
    for m in MODELS:
        if m == 'linear':
            continue
        classification_models.append(m)
    if 'logistic' not in classification_models:
        classification_models = ['logistic'] + classification_models

    total = len(dgp_names) * len(SAMPLE_SIZES) * len(classification_models)
    done = 0
    t_start = time.time()

    for dgp_name in dgp_names:
        true_ames = load_or_compute_ground_truth(dgp_name)

        for n in SAMPLE_SIZES:
            for model_name in classification_models:
                done += 1
                print(f"[{done}/{total}]", end=' ')
                run_simulation(dgp_name, n, model_name, true_ames)

    elapsed = time.time() - t_start
    print(f"\nAll classification simulations complete in {elapsed:.1f}s")
    print(f"Results saved to: {SIM1_RESULTS_DIR}")
