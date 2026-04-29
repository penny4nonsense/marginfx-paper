"""
analyze.py
----------
Shared analysis engine for marginfx empirical examples.

Provides functions for:
    - Fitting models (logistic/linear, random forest, XGBoost, TensorFlow)
    - Computing AMEs with bootstrap SEs across all specifications
    - Computing SHAP values for the full specification
    - Computing PDP slopes for the full specification
    - Saving results to parquet

Used by dataset-specific scripts:
    analyze_adult.py
    analyze_credit_default.py
    analyze_ames_housing.py

Each dataset script calls these functions with its own data,
feature groups, and specifications.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Optional

import marginfx as mfx

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def fit_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    outcome_type: str,
    seed: int = 42,
):
    """
    Build and fit a model.

    Parameters
    ----------
    model_name : str
        One of 'logistic', 'linear', 'rf', 'xgboost', 'tensorflow'.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    outcome_type : str
        One of 'classification', 'regression'.
    seed : int
        Random seed.

    Returns
    -------
    Fitted model object.
    """
    if model_name in ('logistic', 'linear'):
        if outcome_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=5000, random_state=seed)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        model.fit(X, y)
        return model

    elif model_name == 'rf':
        if outcome_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=8,
                random_state=seed, n_jobs=-1,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=200, max_depth=8,
                random_state=seed, n_jobs=-1,
            )
        model.fit(X, y)
        return model

    elif model_name == 'xgboost':
        import xgboost as xgb
        if outcome_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=seed,
                verbosity=0, eval_metric='logloss',
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=seed,
                verbosity=0,
            )
        model.fit(X, y)
        return model

    elif model_name == 'tensorflow':
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.random.set_seed(seed)

        if outcome_type == 'classification':
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            output_activation = None
            loss = 'mse'

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu',
                                  input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=output_activation),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(1e-3)),
            loss=loss,
        )
        model.fit(X, y, epochs=50, batch_size=64, verbose=0)
        return model

    else:
        raise ValueError(f"Unknown model: '{model_name}'")


# ---------------------------------------------------------------------------
# AME computation across specifications
# ---------------------------------------------------------------------------

def compute_ames_all_specs(
    model_names: List[str],
    df: pd.DataFrame,
    specifications: dict,
    outcome: str,
    categorical_features: List[str],
    outcome_type: str,
    n_bootstrap: int = 200,
    seed: int = 42,
    output_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute AMEs with bootstrap SEs for all models and specifications.

    For each specification and model:
        1. Select features for this specification
        2. Fit the model
        3. Compute AMEs with bootstrap SEs
        4. Collect results into tidy DataFrame
        5. Save partial results incrementally (if output_dir provided)

    Parameters
    ----------
    model_names : list of str
        Models to fit.
    df : pd.DataFrame
        Processed dataset with all features and outcome.
    specifications : dict
        Mapping of spec name -> list of feature names.
        e.g. {'A': ['age', 'female'], 'AB': ['age', 'female', 'hours']}
    outcome : str
        Name of outcome column.
    categorical_features : list of str
        Names of categorical/binary features.
    outcome_type : str
        One of 'classification', 'regression'.
    n_bootstrap : int
        Number of bootstrap replicates. Default 200.
    seed : int
        Random seed.
    output_dir : str, optional
        Directory to save partial results after each model completes.
        If None, no incremental saving is performed.
    dataset_name : str, optional
        Dataset name used for partial results filename.
        Required if output_dir is provided.

    Returns
    -------
    pd.DataFrame
        Tidy results with columns:
        model, spec, feature, estimate, std_error,
        statistic, p_value, conf_low, conf_high.
    """
    y = df[outcome].values.astype(float)
    all_rows = []

    # Check for existing partial results — skip completed combinations
    partial_path = None
    completed = set()
    if output_dir and dataset_name:
        os.makedirs(output_dir, exist_ok=True)
        partial_path = os.path.join(
            output_dir, f'{dataset_name}_ames_partial.parquet'
        )
        if os.path.exists(partial_path):
            existing = pd.read_parquet(partial_path)
            all_rows.append(existing)
            completed = set(
                zip(existing['spec'], existing['model'])
            )
            print(f"  Resuming from partial results "
                  f"({len(completed)} combinations already done)")

    for spec_name, features in specifications.items():
        print(f"\n  Specification {spec_name}: {features}")

        X = df[features].values.astype(float)

        # Categorical features for this specification
        cat_feats = [f for f in categorical_features if f in features]

        for model_name in model_names:

            # Skip if already completed in a previous run
            if (spec_name, model_name) in completed:
                print(f"    Skipping {model_name} (already done)")
                continue

            print(f"    Fitting {model_name}...", end='', flush=True)
            t0 = time.time()

            try:
                model = fit_model(model_name, X, y, outcome_type, seed)
                result = mfx.fit(
                    model, X, y,
                    feature_names=features,
                    categorical_features=cat_feats,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                    verbose=False,
                )
                elapsed = time.time() - t0
                print(f" done ({elapsed:.1f}s)")

                tidy = result.tidy()
                tidy['model'] = model_name
                tidy['spec'] = spec_name
                all_rows.append(tidy)

                # Save partial results after every completed combination
                if partial_path:
                    partial = pd.concat(all_rows, ignore_index=True)
                    partial.to_parquet(partial_path, index=False)
                    print(f"      (saved partial: {len(partial)} rows)")

            except Exception as e:
                print(f" ERROR: {e}")
                continue

    if not all_rows:
        return pd.DataFrame()

    results = pd.concat(all_rows, ignore_index=True)

    # Reorder columns
    cols = ['model', 'spec', 'term', 'estimate', 'std_error',
            'statistic', 'p_value', 'conf_low', 'conf_high']
    cols = [c for c in cols if c in results.columns]
    results = results[cols]

    return results


# ---------------------------------------------------------------------------
# SHAP computation — full spec only
# ---------------------------------------------------------------------------

def compute_shap_full(
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    outcome_type: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute SHAP values for all models on the full specification.

    Parameters
    ----------
    model_names : list of str
        Models to compute SHAP for.
    X : np.ndarray
        Feature matrix for full specification.
    y : np.ndarray
        Target vector.
    feature_names : list of str
        Feature names.
    outcome_type : str
        One of 'classification', 'regression'.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Tidy results with columns: model, feature, shap_estimate, shap_abs.
    """
    # Add empirical directory to path for shap_utils
    empirical_dir = os.path.dirname(os.path.abspath(__file__))
    if empirical_dir not in sys.path:
        sys.path.insert(0, empirical_dir)

    from shap_utils import compute_shap_ames

    all_rows = []

    for model_name in model_names:
        print(f"    SHAP {model_name}...", end='', flush=True)
        t0 = time.time()

        try:
            model = fit_model(model_name, X, y, outcome_type, seed)
            shap_results = compute_shap_ames(model, X, feature_names, outcome_type=outcome_type)
            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s)")

            for feature in feature_names:
                all_rows.append({
                    'model':         model_name,
                    'feature':       feature,
                    'shap_estimate': shap_results[feature],
                    'shap_abs':      shap_results[f"{feature}_abs"],
                })

        except Exception as e:
            print(f" ERROR: {e}")
            continue

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# PDP computation — full spec only
# ---------------------------------------------------------------------------

def compute_pdp_full(
    model_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    categorical_features: List[str],
    outcome_type: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute PDP slopes for all models on the full specification.

    Parameters
    ----------
    model_names : list of str
        Models to compute PDP for.
    X : np.ndarray
        Feature matrix for full specification.
    y : np.ndarray
        Target vector.
    feature_names : list of str
        Feature names.
    categorical_features : list of str
        Names of categorical/binary features.
    outcome_type : str
        One of 'classification', 'regression'.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Tidy results with columns: model, feature, pdp_estimate.
    """
    empirical_dir = os.path.dirname(os.path.abspath(__file__))
    if empirical_dir not in sys.path:
        sys.path.insert(0, empirical_dir)

    from pdp_utils import compute_pdp_slopes

    all_rows = []

    for model_name in model_names:
        print(f"    PDP {model_name}...", end='', flush=True)
        t0 = time.time()

        try:
            model = fit_model(model_name, X, y, outcome_type, seed)
            pdp_results = compute_pdp_slopes(
                model, X, feature_names, categorical_features
            )
            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s)")

            for feature in feature_names:
                all_rows.append({
                    'model':        model_name,
                    'feature':      feature,
                    'pdp_estimate': pdp_results[feature],
                })

        except Exception as e:
            print(f" ERROR: {e}")
            continue

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    ame_results: pd.DataFrame,
    shap_results: pd.DataFrame,
    pdp_results: pd.DataFrame,
    output_dir: str,
    dataset_name: str,
) -> None:
    """
    Save AME, SHAP, and PDP results to parquet files.

    Parameters
    ----------
    ame_results : pd.DataFrame
        AME results from compute_ames_all_specs().
    shap_results : pd.DataFrame
        SHAP results from compute_shap_full().
    pdp_results : pd.DataFrame
        PDP results from compute_pdp_full().
    output_dir : str
        Directory to save results.
    dataset_name : str
        Dataset name for file naming.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not ame_results.empty:
        path = os.path.join(output_dir, f'{dataset_name}_ames.parquet')
        ame_results.to_parquet(path, index=False)
        print(f"  Saved AME results: {path}")

    if not shap_results.empty:
        path = os.path.join(output_dir, f'{dataset_name}_shap.parquet')
        shap_results.to_parquet(path, index=False)
        print(f"  Saved SHAP results: {path}")

    if not pdp_results.empty:
        path = os.path.join(output_dir, f'{dataset_name}_pdp.parquet')
        pdp_results.to_parquet(path, index=False)
        print(f"  Saved PDP results: {path}")
