"""
config.py
---------
Central configuration for marginfx simulation studies.

To switch between development and production:
    Set MODE = 'dev' for fast local iteration
    Set MODE = 'production' for full AWS run

All simulation scripts import from this file. Change settings here
and everything else updates automatically.
"""

# ---------------------------------------------------------------------------
# Mode — switch between dev and production
# ---------------------------------------------------------------------------

MODE = 'production'  # 'dev' or 'production'

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

if MODE == 'dev':
    N_ITER_REGRESSION = 1000
    N_ITER_CLASSIFICATION = 500
    SAMPLE_SIZES = [250, 500, 1000, 2500, 5000]
    MODELS = ['logistic', 'rf', 'xgboost', 'tensorflow']
    # Calibration-specific
    N_ITER_CALIBRATION = 100
    N_BOOTSTRAP_CALIBRATION = 100
    CALIBRATION_MODELS = ['logistic', 'xgboost']
    CALIBRATION_SAMPLE_SIZES = [250, 1000]

elif MODE == 'production':
    N_ITER_REGRESSION = 1000
    N_ITER_CLASSIFICATION = 500
    SAMPLE_SIZES = [250, 500, 1000, 2500, 5000]
    MODELS = ['logistic', 'rf', 'xgboost', 'tensorflow']
    # Calibration-specific. These remain the fallback for any learner not
    # named in the per-model tables below.
    N_ITER_CALIBRATION = 1000
    N_BOOTSTRAP_CALIBRATION = 1000
    CALIBRATION_MODELS = ['logistic', 'rf', 'xgboost', 'tensorflow']
    CALIBRATION_SAMPLE_SIZES = [250, 1000, 5000]

# ---------------------------------------------------------------------------
# Simulation 2 replication counts, per learner
# ---------------------------------------------------------------------------
#
# Two different counts, often confused. N_BOOTSTRAP is the number of resamples
# drawn inside one dataset; N_ITER is the number of independent datasets. The
# first governs how well each interval is pinned down, the second how well the
# coverage rate itself is measured.
#
# bootstrap_diagnostic returns PERCENTILE intervals, whose endpoints are the
# 2.5th and 97.5th quantiles of the resample distribution. At 200 resamples
# each endpoint rests on a handful of order statistics; the usual guidance is
# at least 1000 for percentile intervals, against roughly 200 for a standard
# error alone. Hence 1000 here.
#
# TensorFlow is the exception, and it is a cost exception rather than a
# statistical one. Every one of its resamples is a fresh 50-epoch training
# run, which makes it about 93% of this simulation's total cost. Raising it to
# the same counts as the others turns a six-day job into a five-week one, and
# the calibration result the paper actually rests on is the random forest's.
# It runs at the lower setting and the table notes record that.

CALIBRATION_N_BOOTSTRAP = {
    'logistic':   1000,
    'linear':     1000,
    'rf':         1000,
    'xgboost':    1000,
    'tensorflow':  200,
}

CALIBRATION_N_ITER = {
    'logistic':   1000,
    'linear':     1000,
    'rf':         1000,
    'xgboost':    1000,
    'tensorflow':  500,
}


def calibration_counts(model_name):
    """
    Return (n_iter, n_bootstrap) for one learner in Simulation 2.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    tuple of int
    """
    return (
        CALIBRATION_N_ITER.get(model_name, N_ITER_CALIBRATION),
        CALIBRATION_N_BOOTSTRAP.get(model_name, N_BOOTSTRAP_CALIBRATION),
    )


# Per-learner worker count for Simulation 2. Empty: every learner now uses
# the full N_JOBS.
#
# TensorFlow needed a reduced count while its whole Monte Carlo iteration was
# a single task, because one task built n_bootstrap networks in one process
# and Keras does not release that memory until the process ends -- twelve
# workers reached 81 GB with 20 GB free and falling. That is fixed properly in
# parallel_bootstrap, which splits TensorFlow's work across slices of the
# resamples so a worker's footprint is set by the slice size rather than by
# n_bootstrap. Throttling workers is no longer the lever.
#
# Anything added here falls back to N_JOBS, which is defined further down and
# so is resolved when the function is called rather than when this table is
# built.
CALIBRATION_N_JOBS = {}


def calibration_jobs(model_name):
    """
    Return the joblib worker count to use for one learner in Simulation 2.

    Parameters
    ----------
    model_name : str

    Returns
    -------
    int
    """
    return CALIBRATION_N_JOBS.get(model_name, N_JOBS)

# ---------------------------------------------------------------------------
# DGP settings
# ---------------------------------------------------------------------------

# Number of features — x1 and x2 have true effects, x3 and x4 are noise
N_FEATURES = 4

# True coefficients for x1 and x2 — x3 and x4 are always zero
TRUE_BETA_1 = 2.0
TRUE_BETA_2 = 3.0

# DGPs to run
# Each entry: (dgp_name, outcome_type)
DGPS = [
    ('linear',      'regression'),
    ('nonlinear',   'regression'),
    ('interaction', 'regression'),
    ('linear',      'classification'),
    ('nonlinear',   'classification'),
    ('interaction', 'classification'),
]

# ---------------------------------------------------------------------------
# Ground truth settings
# ---------------------------------------------------------------------------

# Number of observations for Monte Carlo ground truth AME computation
# At 1M observations, Monte Carlo error is negligible
N_GROUND_TRUTH = 1_000_000

# Random seed for ground truth — fixed so it's always the same
GROUND_TRUTH_SEED = 0

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

# Random forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 6

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05

# TensorFlow — small network for speed
TF_HIDDEN_UNITS = [32, 16]
TF_EPOCHS = 50
TF_BATCH_SIZE = 64
TF_LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Parallelization
# ---------------------------------------------------------------------------

# Number of parallel jobs for joblib
# -1 uses all available cores
# Set to 1 to disable parallelization (useful for debugging)
N_JOBS = 24

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # simulations/

SIM1_RESULTS_DIR = os.path.join(
    BASE_DIR, 'sim1_ame_recovery', 'results'
)
SIM2_RESULTS_DIR = os.path.join(
    BASE_DIR, 'sim2_se_calibration', 'results'
)
SIM3_RESULTS_DIR = os.path.join(
    BASE_DIR, 'sim3_debiased', 'results'
)
GROUND_TRUTH_DIR = os.path.join(
    BASE_DIR, 'sim1_ame_recovery', 'ground_truth'
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# Base random seed — each simulation iteration uses base_seed + iteration
BASE_SEED = 42

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_config():
    """Print current configuration for logging purposes."""
    print("=" * 55)
    print("marginfx simulation configuration")
    print("=" * 55)
    print(f"  Mode:                {MODE}")
    print(f"  DGPs:                {len(DGPS)}")
    print(f"  Sample sizes:        {SAMPLE_SIZES}")
    print(f"  Models:              {MODELS}")
    print(f"  Iters (regression):  {N_ITER_REGRESSION}")
    print(f"  Iters (classif.):    {N_ITER_CLASSIFICATION}")
    print(f"  Ground truth n:      {N_GROUND_TRUTH:,}")
    print(f"  Parallel jobs:       {N_JOBS}")
    total_reg = len([d for d in DGPS if d[1] == 'regression'])
    total_cls = len([d for d in DGPS if d[1] == 'classification'])
    total_fits = (
        total_reg * len(SAMPLE_SIZES) * N_ITER_REGRESSION * len(MODELS) +
        total_cls * len(SAMPLE_SIZES) * N_ITER_CLASSIFICATION * len(MODELS)
    )
    print(f"  Total model fits:    {total_fits:,}")
    print("=" * 55)


if __name__ == '__main__':
    print_config()
    print("BASE_DIR:", BASE_DIR)
    print("SIM1_RESULTS_DIR:", SIM1_RESULTS_DIR)
    print("SIM2_RESULTS_DIR:", SIM2_RESULTS_DIR)
    print("GROUND_TRUTH_DIR:", GROUND_TRUTH_DIR)
