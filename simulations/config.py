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
    # Calibration-specific
    N_ITER_CALIBRATION = 500
    N_BOOTSTRAP_CALIBRATION = 200
    CALIBRATION_MODELS = ['logistic', 'xgboost', 'tensorflow']
    CALIBRATION_SAMPLE_SIZES = [250, 1000, 5000]

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
N_JOBS = 4

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
