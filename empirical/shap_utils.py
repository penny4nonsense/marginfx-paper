"""
shap_utils.py
-------------
SHAP computation utilities for marginfx empirical analysis.

Computes model-specific SHAP values using the appropriate explainer
for each model family:

    Logistic / Linear Regression -> LinearExplainer  (fast, exact)
    Random Forest                -> TreeExplainer    (fast, exact)
    XGBoost                      -> TreeExplainer    (fast, exact)
    TensorFlow / Keras           -> DeepExplainer    (reasonable speed)

Global feature importance is summarized as the mean signed SHAP value
across all observations for comparison with AMEs.

    global_shap_j = mean(SHAP_ij) for i = 1..n

Note on interpretation:
    SHAP values measure feature attribution — how much did feature x_j
    contribute to each prediction relative to a baseline. AMEs measure
    marginal effects — what happens to the average prediction if x_j
    increases by one unit. These are different quantities. On linear DGPs
    they are proportional. On nonlinear and interaction DGPs they diverge.
    This divergence is a key finding of the empirical comparison.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def _detect_model_type(model) -> str:
    """
    Detect model family for SHAP explainer selection.

    Returns
    -------
    str
        One of 'linear', 'tree', 'tensorflow', 'pytorch', 'unknown'.
    """
    class_name = type(model).__name__

    # TensorFlow / Keras
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return 'tensorflow'
    except Exception:
        pass

    # PyTorch
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
    except Exception:
        pass

    # XGBoost
    try:
        import xgboost as xgb
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            return 'tree'
    except Exception:
        pass

    # LightGBM
    try:
        import lightgbm as lgb
        if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
            return 'tree'
    except Exception:
        pass

    # Sklearn tree-based models
    tree_models = {
        'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'ExtraTreesClassifier', 'ExtraTreesRegressor',
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
    }
    if class_name in tree_models:
        return 'tree'

    # Sklearn linear models
    linear_models = {
        'LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso',
        'ElasticNet', 'BayesianRidge', 'SGDClassifier', 'SGDRegressor',
    }
    if class_name in linear_models:
        return 'linear'

    return 'unknown'


# ---------------------------------------------------------------------------
# SHAP explainers
# ---------------------------------------------------------------------------

def _shap_linear(model, X: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for a linear model using LinearExplainer.

    Parameters
    ----------
    model : sklearn linear model
        Fitted linear or logistic regression model.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).

    Returns
    -------
    np.ndarray
        SHAP values, shape (n_obs, n_features).
    """
    import shap

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)

    # For binary classification, LinearExplainer returns a list
    # Take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return np.array(shap_values)


def _shap_tree(
    model,
    X: np.ndarray,
    outcome_type: str = 'classification',
) -> np.ndarray:
    """
    Compute SHAP values for a tree-based model using TreeExplainer.

    For classification: uses interventional perturbation with probability
    output to get SHAP values in probability units comparable to AMEs.

    For regression: uses default TreeExplainer without probability output
    since model_output='probability' is not valid for regression models.

    Parameters
    ----------
    model : tree-based model
        Fitted RandomForest, XGBoost, GradientBoosting, etc.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    outcome_type : str
        One of 'classification' or 'regression'. Default 'classification'.

    Returns
    -------
    np.ndarray
        SHAP values, shape (n_obs, n_features).
    """
    import shap

    if outcome_type == 'regression':
        # For regression, use default TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X, check_additivity=False)

    else:
        # For classification, use interventional perturbation
        # with probability output for scale comparable to AMEs
        background = shap.sample(X, 100)
        explainer = shap.TreeExplainer(
            model,
            data=background,
            model_output='probability',
            feature_perturbation='interventional',
        )
        shap_values = explainer.shap_values(X, check_additivity=False)

        # Handle different output formats for binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    return np.array(shap_values)


def _shap_tensorflow(model, X: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for a TensorFlow/Keras model using GradientExplainer.

    Uses a random subsample of X as the background dataset for efficiency.
    Background size is capped at 100 observations.

    Parameters
    ----------
    model : tf.keras.Model
        Fitted Keras model.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).

    Returns
    -------
    np.ndarray
        SHAP values, shape (n_obs, n_features).
    """
    import shap

    n_background = min(100, X.shape[0])
    rng = np.random.default_rng(0)
    background_idx = rng.choice(X.shape[0], size=n_background, replace=False)
    background = X[background_idx].astype(np.float32)
    X_input = X.astype(np.float32)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.array(shap_values)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_shap_ames(
    model,
    X: np.ndarray,
    feature_names: Optional[list] = None,
    outcome_type: str = 'classification',
) -> dict:
    """
    Compute global SHAP feature importance as mean signed SHAP values.

    Uses the appropriate explainer for each model family:
        Linear models   -> LinearExplainer
        Tree models     -> TreeExplainer
        TensorFlow      -> DeepExplainer

    For classification tree models, uses interventional perturbation
    with probability output for scale comparable to AMEs.
    For regression tree models, uses default TreeExplainer output.

    Parameters
    ----------
    model : fitted model
        Any supported model type.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    feature_names : list, optional
        Feature names. Defaults to ['x0', 'x1', ...].
    outcome_type : str
        One of 'classification' or 'regression'. Default 'classification'.
        Controls TreeExplainer behavior for tree-based models.

    Returns
    -------
    dict
        Feature name -> mean signed SHAP value (for AME comparison).
        Also returns mean absolute SHAP under key '{feature}_abs'.
    """
    X = np.array(X, dtype=float)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    model_type = _detect_model_type(model)

    if model_type == 'linear':
        shap_values = _shap_linear(model, X)
    elif model_type == 'tree':
        shap_values = _shap_tree(model, X, outcome_type=outcome_type)
    elif model_type == 'tensorflow':
        shap_values = _shap_tensorflow(model, X)
    else:
        raise ValueError(
            f"Unsupported model type for SHAP: {type(model).__name__}. "
            f"Supported: linear sklearn models, tree models, "
            f"XGBoost, TensorFlow/Keras."
        )

    # Compute mean signed SHAP (for AME comparison) and mean absolute SHAP
    results = {}
    for idx, name in enumerate(feature_names):
        results[name] = float(np.mean(shap_values[:, idx]))
        results[f"{name}_abs"] = float(np.mean(np.abs(shap_values[:, idx])))

    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

    print("Testing shap_utils...")

    from dgp import generate_classification, generate_regression, FEATURE_NAMES
    import numpy as np

    rng = np.random.default_rng(42)

    # Test with logistic regression (classification)
    print("\n  Testing LinearExplainer (logistic regression)...")
    from sklearn.linear_model import LogisticRegression
    X, y = generate_classification(500, 'linear', rng)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    shap_ames = compute_shap_ames(model, X, FEATURE_NAMES, outcome_type='classification')
    print(f"  Signed SHAP: { {k: round(v, 4) for k, v in shap_ames.items() if '_abs' not in k} }")

    # Test with random forest (classification)
    print("\n  Testing TreeExplainer (random forest, classification)...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    shap_ames = compute_shap_ames(model, X, FEATURE_NAMES, outcome_type='classification')
    print(f"  Signed SHAP: { {k: round(v, 4) for k, v in shap_ames.items() if '_abs' not in k} }")

    # Test with XGBoost (classification)
    print("\n  Testing TreeExplainer (XGBoost, classification)...")
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=50, verbosity=0,
                               eval_metric='logloss').fit(X, y)
    shap_ames = compute_shap_ames(model, X, FEATURE_NAMES, outcome_type='classification')
    print(f"  Signed SHAP: { {k: round(v, 4) for k, v in shap_ames.items() if '_abs' not in k} }")

    # Test with linear regression (regression)
    print("\n  Testing LinearExplainer (linear regression)...")
    from sklearn.linear_model import LinearRegression
    X_reg, y_reg = generate_regression(500, 'linear', rng)
    model = LinearRegression().fit(X_reg, y_reg)
    shap_ames = compute_shap_ames(model, X_reg, FEATURE_NAMES, outcome_type='regression')
    print(f"  Signed SHAP: { {k: round(v, 4) for k, v in shap_ames.items() if '_abs' not in k} }")

    # Test with random forest (regression)
    print("\n  Testing TreeExplainer (random forest, regression)...")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_reg, y_reg)
    shap_ames = compute_shap_ames(model, X_reg, FEATURE_NAMES, outcome_type='regression')
    print(f"  Signed SHAP: { {k: round(v, 4) for k, v in shap_ames.items() if '_abs' not in k} }")

    print("\nAll shap_utils tests passed.")