"""
analyze_credit_default.py
--------------------------
Empirical analysis of the UCI Credit Card Default dataset using marginfx.

Computes:
    1. AMEs with bootstrap SEs for all specifications and all models
    2. SHAP values for the full specification (ABC) only
    3. PDP slopes for the full specification (ABC) only

Results saved to empirical/credit_default/results/.

Usage:
    python analyze_credit_default.py

Settings:
    n_bootstrap: number of bootstrap replicates for AME SEs
    seed: random seed for reproducibility
    models: list of model families to fit
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMPIRICAL_DIR = os.path.join(SCRIPT_DIR, '..')
PAPER_DIR = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, EMPIRICAL_DIR)
sys.path.insert(0, PAPER_DIR)

from analyze import (
    fit_model,
    compute_ames_all_specs,
    compute_shap_full,
    compute_pdp_full,
    save_results,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 200
SEED = 42
OUTCOME_TYPE = 'classification'

MODELS = ['logistic', 'rf', 'xgboost', 'tensorflow']

PROCESSED_PATH = os.path.join(
    PAPER_DIR, 'data', 'processed', 'credit_default',
    'credit_default_processed.csv'
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(
    PAPER_DIR, 'data', 'processed', 'credit_default'
))
from feature_groups import (
    GROUP_A, GROUP_B, GROUP_C,
    OUTCOME, CATEGORICAL_FEATURES, SPECIFICATIONS
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("marginfx Empirical Analysis — UCI Credit Card Default")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Models:      {MODELS}")
    print(f"  Bootstrap:   {N_BOOTSTRAP} replicates")
    print(f"  Seed:        {SEED}")
    print(f"\nSpecifications:")
    for spec, features in SPECIFICATIONS.items():
        print(f"  {spec}: {features}")
    print()

    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"  {len(df):,} observations, {len(df.columns)} columns")
    print(f"  Outcome prevalence: {df[OUTCOME].mean():.1%} defaulted\n")

    # Full specification features
    full_features = GROUP_A + GROUP_B + GROUP_C

    t_start = time.time()

    # --- AMEs across all specifications ---
    print("Computing AMEs across all specifications...")
    ame_results = compute_ames_all_specs(
        model_names=MODELS,
        df=df,
        specifications=SPECIFICATIONS,
        outcome=OUTCOME,
        categorical_features=CATEGORICAL_FEATURES,
        outcome_type=OUTCOME_TYPE,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
        output_dir=OUTPUT_DIR,
        dataset_name='credit_default',
    )

    # --- SHAP for full specification ---
    print("\nComputing SHAP values (full specification ABC)...")
    X_full = df[full_features].values.astype(float)
    y = df[OUTCOME].values.astype(float)

    shap_results = compute_shap_full(
        model_names=MODELS,
        X=X_full,
        y=y,
        feature_names=full_features,
        outcome_type=OUTCOME_TYPE,
        seed=SEED,
    )

    # --- PDP for full specification ---
    print("\nComputing PDP slopes (full specification ABC)...")
    pdp_results = compute_pdp_full(
        model_names=MODELS,
        X=X_full,
        y=y,
        feature_names=full_features,
        categorical_features=CATEGORICAL_FEATURES,
        outcome_type=OUTCOME_TYPE,
        seed=SEED,
    )

    # --- Save results ---
    print("\nSaving results...")
    save_results(
        ame_results=ame_results,
        shap_results=shap_results,
        pdp_results=pdp_results,
        output_dir=OUTPUT_DIR,
        dataset_name='credit_default',
    )

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")

    # --- Quick summary ---
    if not ame_results.empty:
        print("\nAME Results Preview (full spec, logistic):")
        preview = ame_results[
            (ame_results['model'] == 'logistic') &
            (ame_results['spec'] == 'ABC')
        ][['term', 'estimate', 'std_error', 'p_value']].round(4)
        print(preview.to_string(index=False))
