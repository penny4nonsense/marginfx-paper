import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import numpy as np
import pandas as pd
import gc
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

from dgp import (
    generate_classification,
    compute_ground_truth_ames,
    FEATURE_NAMES,
)
from config import (
    MODE,
    N_ITER_CALIBRATION,
    N_BOOTSTRAP_CALIBRATION,
    calibration_counts,
    calibration_jobs,
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
from parallel_bootstrap import run_iterations

CALIBRATION_DGP = 'linear'

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(model_name: str, seed: int):
    if model_name == 'logistic':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed)

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,
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
        # Set seeds inside builder for worker isolation
        tf.random.set_seed(seed)

        layers = []
        for units in TF_HIDDEN_UNITS:
            layers.append(tf.keras.layers.Dense(units, activation='relu'))
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

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

def run_one_iteration(iteration: int, n: int, model_name: str, true_ames: dict) -> list:
    # Aggressive memory cleanup for the start of the worker task
    if model_name == 'tensorflow':
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)

    # Generate data
    X, y = generate_classification(n, CALIBRATION_DGP, rng)

    # Build and fit
    t0 = time.time()
    model = build_model(model_name, seed)

    if model_name == 'tensorflow':
        model.fit(X, y, epochs=TF_EPOCHS, batch_size=TF_BATCH_SIZE, verbose=0)
    else:
        model.fit(X, y)

    # Run bootstrap
    # Refitting-bootstrap dispersion of the plug-in, which is the procedure
    # whose calibration Simulation 2 is measuring. Trimming is off because
    # the features are unbounded standard normals.
    result = mfx.bootstrap_diagnostic(
        model, X, y,
        feature_names=FEATURE_NAMES,
        n_bootstrap=calibration_counts(model_name)[1],
        trim=False,
        seed=seed,
        verbose=False,
        # Match the schedule the model being diagnosed was trained with.
        # Without these the replicates fall back to the package defaults of
        # 10 epochs at batch 32, so the bootstrap would describe the spread
        # of a differently-trained network than the point estimate it is
        # centred on. Ignored for the learners that are not Keras.
        n_epochs=TF_EPOCHS,
        batch_size=TF_BATCH_SIZE,
    )
    elapsed = time.time() - t0

    # Collect results
    rows = []
    for feature in FEATURE_NAMES:
        est      = result.estimates[feature]
        true_val = true_ames[feature]
        se       = result.std_errors[feature]
        conf_low, conf_high = result.conf_int[feature]

        rows.append({
            'iteration':  iteration,
            'dgp':        CALIBRATION_DGP,
            'n':          n,
            'model':      model_name,
            'feature':    feature,
            'ame_estimate': est,
            'true_ame':   true_val,
            'bias':       est - true_val,
            'se':         se,
            'conf_low':   conf_low,
            'conf_high':  conf_high,
            'covered':    int(conf_low <= true_val <= conf_high),
            'ci_width':   conf_high - conf_low,
            'elapsed':    elapsed,
        })

    # Final cleanup for this iteration
    if model_name == 'tensorflow':
        import tensorflow as tf
        tf.keras.backend.clear_session()

    del model
    gc.collect()

    print('.', end='', flush=True)
    return rows

# ---------------------------------------------------------------------------
# Main calibration runner
# ---------------------------------------------------------------------------

def run_calibration(n: int, model_name: str, true_ames: dict) -> pd.DataFrame:
    os.makedirs(SIM2_RESULTS_DIR, exist_ok=True)
    filename = f"calibration_{CALIBRATION_DGP}_n{n}_{model_name}.parquet"
    filepath = os.path.join(SIM2_RESULTS_DIR, filename)
    partial_path = filepath.replace('.parquet', '_partial.parquet')

    if os.path.exists(filepath):
        print(f"  Skipping (already exists): {filename}")
        return pd.read_parquet(filepath)

    completed_iterations = set()
    all_dfs = []
    if os.path.exists(partial_path):
        existing_df = pd.read_parquet(partial_path)
        completed_iterations = set(existing_df['iteration'].unique())
        all_dfs.append(existing_df)
        print(f"  Resuming from {len(completed_iterations)} completed iterations...")

    n_iter, n_boot = calibration_counts(model_name)
    remaining = [i for i in range(n_iter) if i not in completed_iterations]
    print(f"  Running: n={n}, model={model_name}, {len(remaining)} iterations remaining...")

    # maxtasksperchild=1 replaces each worker after one task so that whatever
    # it accumulated is handed back. For most learners a task is one whole
    # Monte Carlo iteration, which is the natural unit and holds nothing
    # troublesome.
    #
    # Keras is the exception. One iteration builds n_bootstrap networks in a
    # single process and does not release that memory until the process ends,
    # so with a task that large the recycling never gets a chance to help.
    # For TensorFlow the work is split the other way, across slices of the
    # resamples, which bounds a worker by the slice size instead of by
    # n_bootstrap. Same estimator either way -- see parallel_bootstrap.
    n_jobs = calibration_jobs(model_name)
    chunked = model_name == 'tensorflow'
    batch_size = n_jobs if not chunked else max(n_jobs, 12)

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        if chunked:
            rows = run_iterations(
                outcome_type='classification',
                model_name=model_name,
                n=n,
                dgp_name=CALIBRATION_DGP,
                iterations=batch,
                true_ames=true_ames,
                feature_names=FEATURE_NAMES,
                n_bootstrap=n_boot,
                n_jobs=n_jobs,
                trim=False,
                # Evaluate each replicate at the observed sample, not at the
                # resample. This is what marginfx's bootstrap does, and what
                # Section 2.3 of the paper analyses: it omits the sampling
                # variability of the average over covariates, which is zero
                # only when the fitted derivative is constant in x. Simulation 2
                # exists partly to measure that omission, so the choice is the
                # object of study rather than an implementation detail.
                eval_at='original',
            )
            batch_rows = [rows]
        else:
            batch_rows = Parallel(n_jobs=n_jobs, maxtasksperchild=1)(
                delayed(run_one_iteration)(
                    iteration=i,
                    n=n,
                    model_name=model_name,
                    true_ames=true_ames,
                )
                for i in batch
            )

        rows = [row for iter_rows in batch_rows for row in iter_rows]
        all_dfs.append(pd.DataFrame(rows))

        # Save partial
        partial_df = pd.concat(all_dfs, ignore_index=True)
        partial_df.to_parquet(partial_path, index=False)
        print(f"    {len(partial_df['iteration'].unique())}/{n_iter} iterations done")

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(filepath, index=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    coverage = df.groupby('feature')['covered'].mean().round(3)
    print(f"    Saved {len(df)} rows -> {filename}")
    print(f"    Coverage (target 0.95):\n{coverage.to_string()}\n")
    return df

# ---------------------------------------------------------------------------
# Ground truth and Main Entry
# ---------------------------------------------------------------------------

def load_or_compute_ground_truth() -> dict:
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    gt_file = os.path.join(GROUND_TRUTH_DIR, f"classification_{CALIBRATION_DGP}_ground_truth.parquet")

    if os.path.exists(gt_file):
        df = pd.read_parquet(gt_file)
        return dict(zip(df['feature'], df['true_ame']))

    print(f"  Computing ground truth (n={N_GROUND_TRUTH:,})...")
    true_ames = compute_ground_truth_ames(
        dgp_name=CALIBRATION_DGP,
        outcome_type='classification',
        n=N_GROUND_TRUTH,
        seed=GROUND_TRUTH_SEED,
    )
    pd.DataFrame([{'feature': k, 'true_ame': v} for k, v in true_ames.items()]).to_parquet(gt_file, index=False)
    return true_ames

if __name__ == '__main__':
    print_config()
    true_ames = load_or_compute_ground_truth()

    total_combos = len(CALIBRATION_SAMPLE_SIZES) * len(CALIBRATION_MODELS)
    done = 0
    t_start = time.time()

    for n in CALIBRATION_SAMPLE_SIZES:
        for model_name in CALIBRATION_MODELS:
            done += 1
            print(f"[{done}/{total_combos}]", end=' ')
            run_calibration(n, model_name, true_ames)

    print(f"\nAll calibration simulations complete in {time.time() - t_start:.1f}s")