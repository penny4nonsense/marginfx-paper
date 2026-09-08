"""
run_debiased.py
---------------
Simulation 3: the debiased cross-fitted estimator with a closed-form
Riesz representer.

Simulations 1 and 2 measure the uncorrected plug-in: its regularization bias,
and the failure of the refitting bootstrap to see that bias. Simulation 3 runs
the corrected estimator over the same grid of DGPs, model classes and sample
sizes, and reports bias, RMSE, and the coverage of the influence-function
confidence intervals.

The design is chosen so that no part of the representer has to be estimated.
The features are independent standard normals, so the support is unbounded,
Omega_{j,h} is all of R^d, the trimming weight is identically one, and

    alpha_h(u) = ( p(u - h e_j) - p(u + h e_j) ) / ( 2h p(u) )
               = exp(-h^2 / 2) * sinh(h u_j) / h

exactly. Passing this closed form isolates the behaviour of the orthogonal
score from any error in estimating the representer, which is what makes the
comparison against Simulations 1 and 2 clean. marginfx.gaussian_window_riesz
returns it, and trim=False matches the weight the formula assumes.

Each iteration produces a point estimate and a standard error in one call, so
this single simulation covers the ground of both earlier ones.

Two arms are available. The 'closed' arm passes the exact representer above and
so isolates the orthogonal score from any error in estimating it: the bias of
the orthogonal moment is a product of the two nuisance errors, and with the
representer known exactly that product is identically zero however slowly the
regression converges. The 'sieve' arm estimates the representer by Riesz
regression instead, which is what a user faces on real data, and so measures
whether the product term is small enough in practice. Comparing the arms
separates 'the score is right' from 'the representer is estimable'.

Results saved as parquet files to SIM3_RESULTS_DIR, one file per
DGP / sample size / model, resumable at the level of the whole combination.

Usage:
    python run_debiased.py                  # both outcome types
    python run_debiased.py --outcome regression
    python run_debiased.py --outcome classification

To change settings (dev vs production, sample sizes, models):
    Edit ../config.py
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import gc
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

from dgp import (
    UNIFORM_SUPPORT,
    support_for,
    generate_regression,
    generate_classification,
    compute_ground_truth_ames,
    FEATURE_NAMES,
)
from config import (
    MODE,
    N_ITER_REGRESSION,
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
    SIM3_RESULTS_DIR,
    print_config,
)

import marginfx as mfx

# ---------------------------------------------------------------------------
# Simulation 3 settings
# ---------------------------------------------------------------------------

# Cross-fitting folds. Matches the default the paper describes in Section 2.4.
N_FOLDS = 5

# Nominal level for the influence-function intervals.
ALPHA = 0.05

DGP_NAMES = ['linear', 'nonlinear', 'interaction']

# Iterations per checkpoint. Each batch is one parallel call, after which the
# partial results are written and the worker pool is rebuilt. Smaller batches
# cap memory growth and shorten what an interrupted run has to redo, at the
# cost of respawning workers -- which for TensorFlow means paying the import
# again -- so a few iterations per worker per batch is the useful range.
BATCH_SIZE = N_JOBS * 4


def _recycle_workers() -> None:
    """
    Shut down the joblib worker pool so its memory is returned to the OS.

    joblib keeps a reusable executor alive between Parallel calls. That is
    normally what you want, but a worker that has built a few hundred Keras
    models never gives that memory back, and clear_session() inside the
    worker does not recover it. Terminating the pool does.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        # Not fatal: without recycling the run still completes, it just
        # holds more memory.
        pass
    gc.collect()


# Arm -> (results directory name, representer specification passed to mfx.fit)
ARMS = {
    'closed': 'results',
    'sieve': 'results_sieve',
    'uniform': 'results_uniform',
}

# The bounded-support arm runs one design rather than three: the point is to
# exhibit the theorems under the density-bounded assumption, not to re-sweep
# the DGP space.
ARM_DGPS = {
    'uniform': ['linear_uniform'],
}


def results_dir_for(arm: str) -> str:
    """Results directory for one arm, kept separate so both can coexist."""
    return os.path.join(os.path.dirname(SIM3_RESULTS_DIR), ARMS[arm])


def dgps_for(arm: str) -> list:
    """DGP names swept by one arm."""
    return ARM_DGPS.get(arm, DGP_NAMES)


def riesz_for(arm: str):
    """
    Representer specification for one arm.

    'closed' supplies the exact Gaussian representer, 'uniform' the exact
    bounded-support one, and 'sieve' returns None, which makes mfx.fit estimate
    the representer by Riesz regression on each fold.
    """
    if arm == 'closed':
        return gaussian_riesz_factory
    if arm == 'uniform':
        return uniform_riesz_factory
    return None


def gaussian_riesz_factory(feature_idx, h, is_categorical):
    """
    Supply the closed-form representer to the estimator.

    mfx.fit calls a representer factory as riesz(feature_idx, h,
    is_categorical). No feature in this design is categorical, so the flag is
    ignored.
    """
    return mfx.gaussian_window_riesz(feature_idx, h)


def uniform_riesz_factory(feature_idx, h, is_categorical):
    """
    Supply the bounded-support representer to the estimator.

    On the uniform design the density is constant, so it cancels and only the
    trimming weight survives: alpha_h is the step function taking -1/(2h) and
    +1/(2h) on the two boundary shells and zero in between. The endpoints
    passed are the true support, matching the bounds handed to mfx.fit, so that
    the weight in the score and the weight assumed by the representer agree.
    """
    lo, hi = UNIFORM_SUPPORT
    return mfx.uniform_window_riesz(feature_idx, h, lo, hi)


# ---------------------------------------------------------------------------
# Learner builders
# ---------------------------------------------------------------------------

def build_learner(model_name: str, outcome_type: str, seed: int):
    """
    Build an UNFITTED learner for the cross-fitted estimator.

    mfx.fit refits the learner once per fold, so it takes a specification
    rather than a trained model. Keras cannot be cloned back to an untrained
    state, so the network is supplied as a zero-argument factory.

    Hyperparameters match those in Simulations 1 and 2, so that any
    difference in the results is attributable to the estimator rather than
    to the learner.

    Parameters
    ----------
    model_name : str
        One of 'linear', 'logistic', 'rf', 'xgboost', 'tensorflow'.
    outcome_type : str
        One of 'regression', 'classification'.
    seed : int
        Random seed.

    Returns
    -------
    An unfitted scikit-learn compatible estimator, or a callable returning a
    freshly compiled Keras model.
    """
    classification = outcome_type == 'classification'

    if model_name in ('linear', 'logistic'):
        if classification:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=seed)
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    elif model_name == 'rf':
        if classification:
            from sklearn.ensemble import RandomForestClassifier
            cls = RandomForestClassifier
        else:
            from sklearn.ensemble import RandomForestRegressor
            cls = RandomForestRegressor
        return cls(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,  # joblib handles outer parallelism
        )

    elif model_name == 'xgboost':
        import xgboost as xgb
        kwargs = dict(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
        if classification:
            return xgb.XGBClassifier(eval_metric='logloss', **kwargs)
        return xgb.XGBRegressor(**kwargs)

    elif model_name == 'tensorflow':
        def make_model():
            import tensorflow as tf
            tf.random.set_seed(seed)

            layers = [
                tf.keras.layers.Dense(units, activation='relu')
                for units in TF_HIDDEN_UNITS
            ]
            layers.append(
                tf.keras.layers.Dense(
                    1, activation='sigmoid' if classification else None
                )
            )
            model = tf.keras.Sequential(layers)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=TF_LEARNING_RATE
                ),
                loss='binary_crossentropy' if classification else 'mse',
            )
            return model

        return make_model

    else:
        raise ValueError(f"Unknown model: '{model_name}'")


# ---------------------------------------------------------------------------
# Single iteration
# ---------------------------------------------------------------------------

def run_one_iteration(
    iteration: int,
    dgp_name: str,
    outcome_type: str,
    n: int,
    model_name: str,
    true_ames: dict,
    arm: str = 'closed',
) -> list:
    """
    Run one Monte Carlo iteration for a DGP / sample size / model.

    arm selects the representer: 'closed' for the exact Gaussian form,
    'sieve' to estimate it by Riesz regression.

    Returns
    -------
    list of dict
        One row per feature, carrying both the point-estimate columns that
        Simulation 1 reports and the interval columns that Simulation 2
        reports.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if model_name == 'tensorflow':
        import tensorflow as tf
        tf.keras.backend.clear_session()
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)

    if outcome_type == 'regression':
        X, y = generate_regression(n, dgp_name, rng)
    else:
        X, y = generate_classification(n, dgp_name, rng)

    # On the Gaussian designs the support is unbounded, Omega_{j,h} is all of
    # R^d and the trimming weight is identically one. On the bounded design it
    # is active and is part of the estimand, so the true support is passed
    # explicitly rather than letting the sample extremes stand in -- the sample
    # minimum of a uniform sits strictly inside its support and moves with n,
    # which would make theta_h a different target at every sample size.
    bounds = support_for(dgp_name)
    t0 = time.time()
    result = mfx.fit(
        build_learner(model_name, outcome_type, seed),
        X, y,
        feature_names=FEATURE_NAMES,
        trim=bounds is not None,
        bounds=bounds,
        riesz=riesz_for(arm),            # closed form, or estimated by sieve
        n_folds=N_FOLDS,
        alpha=ALPHA,
        n_epochs=TF_EPOCHS,
        batch_size=TF_BATCH_SIZE,
        seed=seed,
        verbose=False,
    )
    elapsed = time.time() - t0

    rows = []
    for feature in FEATURE_NAMES:
        est = result.estimates[feature]
        true_ame = true_ames[feature]
        se = result.std_errors[feature]
        conf_low, conf_high = result.conf_int[feature]

        rows.append({
            'iteration':    iteration,
            'dgp':          dgp_name,
            'outcome_type': outcome_type,
            'n':            n,
            'model':        model_name,
            'arm':          arm,
            'feature':      feature,
            'ame_estimate': est,
            'true_ame':     true_ame,
            'bias':         est - true_ame,
            'abs_bias':     abs(est - true_ame),
            'sq_error':     (est - true_ame) ** 2,
            'se':           se,
            'conf_low':     conf_low,
            'conf_high':    conf_high,
            'covered':      int(conf_low <= true_ame <= conf_high),
            'ci_width':     conf_high - conf_low,
            'elapsed':      elapsed,
        })

    if model_name == 'tensorflow':
        import tensorflow as tf
        tf.keras.backend.clear_session()
    gc.collect()

    print('.', end='', flush=True)
    return rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_combination(
    dgp_name: str,
    outcome_type: str,
    n: int,
    model_name: str,
    true_ames: dict,
    n_iter: int,
    arm: str = 'closed',
) -> pd.DataFrame:
    """
    Run every iteration for one DGP / sample size / model combination.

    Completed combinations are skipped; partial progress within a
    combination is checkpointed after each batch so a long run survives an
    interruption.
    """
    results_dir = results_dir_for(arm)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{outcome_type}_{dgp_name}_n{n}_{model_name}.parquet"
    filepath = os.path.join(results_dir, filename)
    partial_path = filepath.replace('.parquet', '_partial.parquet')

    if os.path.exists(filepath):
        print(f"  Skipping (already exists): {filename}")
        return pd.read_parquet(filepath)

    completed = set()
    frames = []
    if os.path.exists(partial_path):
        existing = pd.read_parquet(partial_path)
        completed = set(existing['iteration'].unique())
        frames.append(existing)
        print(f"  Resuming from {len(completed)} completed iterations...")

    remaining = [i for i in range(n_iter) if i not in completed]
    if remaining:
        print(f"  {filename}: {len(remaining)} iterations", flush=True)
        t0 = time.time()

        # Iterations are run in batches rather than in one long parallel call,
        # for two reasons. The partial file is rewritten after each batch, so
        # an interrupted run resumes at batch granularity instead of losing
        # the whole combination. And the worker pool is torn down between
        # batches: loky reuses workers for the lifetime of a call, and a
        # TensorFlow worker's memory grows with every model it builds, which
        # left long calls consuming the machine. Rebuilding the pool is the
        # only reliable way to give that memory back.
        for start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[start:start + BATCH_SIZE]
            results = Parallel(n_jobs=N_JOBS, verbose=0)(
                delayed(run_one_iteration)(
                    i, dgp_name, outcome_type, n, model_name, true_ames, arm
                )
                for i in batch
            )
            frames.append(pd.DataFrame([r for rows in results for r in rows]))

            pd.concat(frames, ignore_index=True).to_parquet(
                partial_path, index=False
            )
            _recycle_workers()
            done = start + len(batch)
            print(f"    {done}/{len(remaining)} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

        print(f"  done in {time.time() - t0:.1f}s")

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(filepath, index=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    return df


def ground_truth_for(dgp_name: str, outcome_type: str) -> dict:
    """
    Ground truth window AMEs, computed once per DGP at N_GROUND_TRUTH.

    Uses the same step size rule and the same trim=False setting as the
    estimator, so the target being measured is the one being estimated.
    """
    return compute_ground_truth_ames(
        dgp_name=dgp_name,
        outcome_type=outcome_type,
        n=N_GROUND_TRUTH,
        seed=GROUND_TRUTH_SEED,
    )


def main(outcomes: list, arm: str = 'closed') -> None:
    print("=" * 60)
    print("Simulation 3 — debiased estimator, closed-form representer")
    print("=" * 60)
    print_config()
    print(f"  Folds:               {N_FOLDS}")
    representer = {
        'closed': 'closed form (gaussian_window_riesz)',
        'uniform': 'closed form (uniform_window_riesz)',
    }.get(arm, 'estimated by Riesz regression (sieve)')
    trimming = ('on, at the true support (bounded design)' if arm == 'uniform'
                else 'off (unbounded support)')
    print(f"  Arm:                 {arm}")
    print(f"  DGPs:                {dgps_for(arm)}")
    print(f"  Representer:         {representer}")
    print(f"  Trimming:            {trimming}")
    print(f"  Results:             {results_dir_for(arm)}")
    print()

    t_start = time.time()
    for outcome_type in outcomes:
        n_iter = (N_ITER_REGRESSION if outcome_type == 'regression'
                  else N_ITER_CLASSIFICATION)
        models = [
            'logistic' if (m == 'linear' and outcome_type == 'classification')
            else 'linear' if (m == 'logistic' and outcome_type == 'regression')
            else m
            for m in MODELS
        ]

        for dgp_name in dgps_for(arm):
            print(f"\n--- {outcome_type} / {dgp_name} ---")
            true_ames = ground_truth_for(dgp_name, outcome_type)
            print("  true AMEs:",
                  {k: round(v, 4) for k, v in true_ames.items()})

            for n in SAMPLE_SIZES:
                for model_name in models:
                    run_combination(
                        dgp_name, outcome_type, n, model_name,
                        true_ames, n_iter, arm,
                    )

    print(f"\nDone in {time.time() - t_start:.1f}s")
    print(f"Results in: {results_dir_for(arm)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--outcome',
        choices=['regression', 'classification', 'both'],
        default='both',
        help='Which outcome type(s) to run. Default both.',
    )
    parser.add_argument(
        '--riesz',
        choices=['closed', 'sieve', 'uniform'],
        default='closed',
        help=(
            'Representer: the exact Gaussian closed form, or estimated by '
            'Riesz regression. Results go to separate directories. '
            'Default closed.'
        ),
    )
    args = parser.parse_args()

    outcomes = (['regression', 'classification'] if args.outcome == 'both'
                else [args.outcome])
    main(outcomes, args.riesz)
