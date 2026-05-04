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
    generate_regression,
    compute_ground_truth_ames,
    FEATURE_NAMES,
)
from config import (
    MODE,
    N_ITER_CALIBRATION,
    N_BOOTSTRAP_CALIBRATION,
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

CALIBRATION_DGP = 'linear'
CALIBRATION_MODELS_REGRESSION = ['linear', 'rf', 'xgboost', 'tensorflow']


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model(model_name: str, seed: int, X: np.ndarray, y: np.ndarray):
    """
    Build and fit a regression model. For TensorFlow, applies y standardization
    and X normalization, returning an UnscaledModel wrapper so predictions
    are in original units.
    """
    if model_name == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        return model

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X, y)
        return model

    elif model_name == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=seed,
            verbosity=0,
        )
        model.fit(X, y)
        return model

    elif model_name == 'tensorflow':
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.disable_interactive_logging()

        # Scale y
        y_mean = float(np.mean(y))
        y_scale = float(np.std(y))
        y_fit = (y - y_mean) / y_scale

        # Normalization layer for X
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(X.astype(np.float32))

        inner_model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=None),
        ])
        inner_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(TF_LEARNING_RATE)),
            loss='mse',
        )
        inner_model.fit(
            X.astype(np.float32),
            y_fit.astype(np.float32),
            epochs=TF_EPOCHS,
            batch_size=TF_BATCH_SIZE,
            verbose=0,
            validation_split=0.1,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=20, restore_best_weights=True, monitor='val_loss'
            )],
        )

        # Capture for closure
        y_mean_val = y_mean
        y_scale_val = y_scale
        captured_inner = inner_model

        class UnscaledModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self._inner = captured_inner
                self._mean = y_mean_val
                self._scale = y_scale_val
                self.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=float(TF_LEARNING_RATE)
                    ),
                    loss='mse',
                )

            def call(self, X, training=False):
                return self._inner(
                    tf.cast(X, dtype=tf.float32),
                    training=training,
                ) * self._scale + self._mean

            def get_weights(self):
                return self._inner.get_weights()

            def get_config(self):
                return self._inner.get_config()

        return UnscaledModel()

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

    if model_name == 'tensorflow':
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.keras.utils.disable_interactive_logging()
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)

    X, y = generate_regression(n, CALIBRATION_DGP, rng)

    t0 = time.time()
    model = build_model(model_name, seed, X, y)

    result = mfx.fit(
        model, X, y,
        feature_names=FEATURE_NAMES,
        n_bootstrap=N_BOOTSTRAP_CALIBRATION,
        seed=seed,
        verbose=False,
    )
    elapsed = time.time() - t0

    rows = []
    for feature in FEATURE_NAMES:
        est      = result.estimates[feature]
        true_val = true_ames[feature]
        se       = result.std_errors[feature]
        conf_low, conf_high = result.conf_int[feature]

        rows.append({
            'iteration':    iteration,
            'dgp':          CALIBRATION_DGP,
            'n':            n,
            'model':        model_name,
            'feature':      feature,
            'ame_estimate': est,
            'true_ame':     true_val,
            'bias':         est - true_val,
            'se':           se,
            'conf_low':     conf_low,
            'conf_high':    conf_high,
            'covered':      int(conf_low <= true_val <= conf_high),
            'ci_width':     conf_high - conf_low,
            'elapsed':      elapsed,
        })

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
    filename = f"calibration_regression_{CALIBRATION_DGP}_n{n}_{model_name}.parquet"
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

    remaining = [i for i in range(N_ITER_CALIBRATION) if i not in completed_iterations]
    print(f"  Running: n={n}, model={model_name}, {len(remaining)} iterations remaining...")

    batch_size = N_JOBS
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        n_jobs_actual = N_JOBS if model_name != 'tensorflow' else N_JOBS
        batch_rows = Parallel(
            n_jobs=n_jobs_actual,
            maxtasksperchild=1 if model_name == 'tensorflow' else None,
        )(
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

        partial_df = pd.concat(all_dfs, ignore_index=True)
        partial_df.to_parquet(partial_path, index=False)
        print(f"\n    {len(partial_df['iteration'].unique())}/{N_ITER_CALIBRATION} iterations done")

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
    gt_file = os.path.join(
        GROUND_TRUTH_DIR,
        f"regression_{CALIBRATION_DGP}_ground_truth.parquet"
    )

    if os.path.exists(gt_file):
        df = pd.read_parquet(gt_file)
        return dict(zip(df['feature'], df['true_ame']))

    print(f"  Computing ground truth (n={N_GROUND_TRUTH:,})...")
    true_ames = compute_ground_truth_ames(
        dgp_name=CALIBRATION_DGP,
        outcome_type='regression',
        n=N_GROUND_TRUTH,
        seed=GROUND_TRUTH_SEED,
    )
    pd.DataFrame(
        [{'feature': k, 'true_ame': v} for k, v in true_ames.items()]
    ).to_parquet(gt_file, index=False)
    return true_ames


if __name__ == '__main__':
    print_config()
    print(f"\nRegression calibration models: {CALIBRATION_MODELS_REGRESSION}")
    true_ames = load_or_compute_ground_truth()

    total_combos = len(CALIBRATION_SAMPLE_SIZES) * len(CALIBRATION_MODELS_REGRESSION)
    done = 0
    t_start = time.time()

    for n in CALIBRATION_SAMPLE_SIZES:
        for model_name in CALIBRATION_MODELS_REGRESSION:
            done += 1
            print(f"[{done}/{total_combos}]", end=' ')
            run_calibration(n, model_name, true_ames)

    print(f"\nAll regression calibration simulations complete in {time.time() - t_start:.1f}s")
