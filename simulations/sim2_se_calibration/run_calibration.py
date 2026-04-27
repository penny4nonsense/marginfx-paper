"""
run_calibration.py
------------------
Simulation 2: Bootstrap SE Calibration.

For each model and sample size, runs N_ITER_CALIBRATION Monte Carlo iterations.
Each iteration:
    1. Generate classification data from the linear DGP
    2. Fit the model
    3. Run full bootstrap (N_BOOTSTRAP_CALIBRATION replicates)
    4. Check whether the 95% CI contains the true AME
    5. Record coverage, CI width, and computation time

Coverage should be close to 95% if bootstrap SEs are well-calibrated.
CI width tells you how precise the estimates are at each sample size.
Computation time documents the practical cost of the bootstrap.

Uses the linear classification DGP only — the cleanest case for
validating SE calibration. Coverage results generalize to other DGPs
since the bootstrap machinery is model-agnostic.

Results saved as parquet files to SIM2_RESULTS_DIR.

Usage:
    python run_calibration.py

To change settings (dev vs production, models, bootstrap replicates):
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
    generate_classification,
    compute_ground_truth_ames,
    FEATURE_NAMES,
)
from config import (
    MODE,
    N_ITER_CALIBRATION,
    N_BOOTSTRAP_CALIBRATION,
    CALIBRATION_MODELS,
    CALIBRATION_SAMPLE_SIZES,
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
    SIM2_RESULTS_DIR,
    GROUND_TRUTH_DIR,
    print_config,
)

import marginfx as mfx


# ---------------------------------------------------------------------------
# DGP for calibration
# ---------------------------------------------------------------------------

# Use linear classification DGP only — cleanest case for SE validation
CALIBRATION_DGP = 'linear'


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(model_name: str, seed: int):
    """
    Build and return an unfitted classification model.

    Parameters
    ----------
    model_name : str
        One of 'logistic', 'xgboost', 'tensorflow'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Unfitted model object.
    """
    if model_name == 'logistic':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed)

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
    n: int,
    model_name: str,
    true_ames: dict,
) -> list:
    """
    Run a single calibration iteration for one sample size / model.

    Fits the model, runs the full bootstrap, and checks whether each
    feature's 95% CI contains the true AME.

    Parameters
    ----------
    iteration : int
        Iteration index (used for seeding).
    n : int
        Sample size.
    model_name : str
        Model name.
    true_ames : dict
        Ground truth AMEs for the linear classification DGP.

    Returns
    -------
    list of dict
        One row per feature with columns:
        iteration, dgp, n, model, feature, ame_estimate, true_ame,
        se, conf_low, conf_high, covered, ci_width, elapsed.
    """
    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)

    # Generate classification data
    X, y = generate_classification(n, CALIBRATION_DGP, rng)

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

    # Run full bootstrap
    result = mfx.fit(
        model, X, y,
        feature_names=FEATURE_NAMES,
        n_bootstrap=N_BOOTSTRAP_CALIBRATION,
        seed=seed,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Collect results — one row per feature
    rows = []
    tidy = result.tidy().set_index('term')

    for feature in FEATURE_NAMES:
        est      = result.estimates[feature]
        true_ame = true_ames[feature]
        se       = result.std_errors[feature]
        conf_low = result.conf_int[feature][0]
        conf_high = result.conf_int[feature][1]
        covered  = int(conf_low <= true_ame <= conf_high)
        ci_width = conf_high - conf_low

        rows.append({
            'iteration':  iteration,
            'dgp':        CALIBRATION_DGP,
            'n':          n,
            'model':      model_name,
            'feature':    feature,
            'ame_estimate': est,
            'true_ame':   true_ame,
            'bias':       est - true_ame,
            'se':         se,
            'conf_low':   conf_low,
            'conf_high':  conf_high,
            'covered':    covered,
            'ci_width':   ci_width,
            'elapsed':    elapsed,
        })

    return rows


# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_calibration(
    n: int,
    model_name: str,
    true_ames: dict,
) -> pd.DataFrame:
    """
    Run all iterations for one sample size / model combination.

    Results saved to parquet after each combination completes.
    If the output file already exists, skips and loads from disk.
    """
    os.makedirs(SIM2_RESULTS_DIR, exist_ok=True)
    filename = f"calibration_{CALIBRATION_DGP}_n{n}_{model_name}.parquet"
    filepath = os.path.join(SIM2_RESULTS_DIR, filename)

    # Skip if already computed
    if os.path.exists(filepath):
        print(f"  Skipping (already exists): {filename}")
        return pd.read_parquet(filepath)

    print(f"  Running: n={n}, model={model_name}, "
          f"bootstrap={N_BOOTSTRAP_CALIBRATION}...")

    # Run iterations in parallel
    all_rows = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_iteration)(
            iteration=i,
            n=n,
            model_name=model_name,
            true_ames=true_ames,
        )
        for i in range(N_ITER_CALIBRATION)
    )

    # Flatten list of lists and save
    rows = [row for iteration_rows in all_rows for row in iteration_rows]
    df = pd.DataFrame(rows)
    df.to_parquet(filepath, index=False)

    # Print coverage summary
    coverage = df.groupby('feature')['covered'].mean().round(3)
    ci_width = df.groupby('feature')['ci_width'].mean().round(4)
    elapsed_mean = df.groupby('feature')['elapsed'].mean().round(2)

    print(f"    Saved {len(df)} rows -> {filename}")
    print(f"    Coverage (target 0.95):\n{coverage.to_string()}")
    print(f"    Mean CI width:\n{ci_width.to_string()}")
    print(f"    Mean elapsed (s):\n{elapsed_mean.to_string()}\n")

    return df


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def load_or_compute_ground_truth() -> dict:
    """
    Load or compute ground truth AMEs for the linear classification DGP.

    Reuses ground truth from Simulation 1 if already computed.
    """
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    gt_file = os.path.join(
        GROUND_TRUTH_DIR,
        f"classification_{CALIBRATION_DGP}_ground_truth.parquet"
    )

    if os.path.exists(gt_file):
        print(f"  Loading ground truth: {gt_file}")
        df = pd.read_parquet(gt_file)
        return dict(zip(df['feature'], df['true_ame']))

    print(f"  Computing ground truth (n={N_GROUND_TRUTH:,})...")
    t0 = time.time()
    true_ames = compute_ground_truth_ames(
        dgp_name=CALIBRATION_DGP,
        outcome_type='classification',
        n=N_GROUND_TRUTH,
        seed=GROUND_TRUTH_SEED,
    )
    elapsed = time.time() - t0
    print(f"  Ground truth computed in {elapsed:.1f}s: {true_ames}")

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
    print(f"\nCalibration settings:")
    print(f"  DGP:               {CALIBRATION_DGP} (classification)")
    print(f"  Models:            {CALIBRATION_MODELS}")
    print(f"  Sample sizes:      {CALIBRATION_SAMPLE_SIZES}")
    print(f"  Iterations:        {N_ITER_CALIBRATION}")
    print(f"  Bootstrap per iter:{N_BOOTSTRAP_CALIBRATION}")
    total = (
        len(CALIBRATION_SAMPLE_SIZES) * N_ITER_CALIBRATION *
        N_BOOTSTRAP_CALIBRATION * len(CALIBRATION_MODELS)
    )
    print(f"  Total bootstrap fits: {total:,}")
    print()

    # Load or compute ground truth
    true_ames = load_or_compute_ground_truth()
    print(f"  True AMEs: {true_ames}\n")

    total_combos = len(CALIBRATION_SAMPLE_SIZES) * len(CALIBRATION_MODELS)
    done = 0
    t_start = time.time()

    for n in CALIBRATION_SAMPLE_SIZES:
        for model_name in CALIBRATION_MODELS:
            done += 1
            print(f"[{done}/{total_combos}]", end=' ')
            run_calibration(n, model_name, true_ames)

    elapsed = time.time() - t_start
    print(f"\nAll calibration simulations complete in {elapsed:.1f}s")
    print(f"Results saved to: {SIM2_RESULTS_DIR}")