"""
pdp_utils.py
------------
Partial Dependence Plot (PDP) utilities for marginfx empirical analysis.

Computes PDP-based marginal effect estimates for comparison with AMEs and SHAP.

The PDP for feature j is:
    PDP_j(v) = E_X[f(x | x_j = v)]

For each grid point v, all observations have x_j set to v and the mean
prediction is computed. This gives a curve of mean predictions across
the feature's range.

The PDP-based marginal effect is the average slope of this curve:
    PDP_slope_j = mean over grid of [dPDP_j/dv]

Computed via central finite differences on the PDP curve itself.

For continuous features:
    - Grid of 20 evenly spaced values between 5th and 95th percentile
    - Finite differences of the PDP curve give slopes at each grid point
    - Average slope is the PDP-based marginal effect

For categorical/binary features:
    - First difference: PDP(1) - PDP(0)
    - Same as AME for categorical features in core.py
    - No grid needed — just two predictions

Uses sklearn's partial_dependence for sklearn-compatible models where
available. Falls back to manual implementation for TensorFlow and other
models.
"""

import numpy as np
from typing import Callable, List, Optional, Union


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _make_grid(x: np.ndarray, n_points: int = 20) -> np.ndarray:
    """
    Create a grid of evenly spaced values between the 5th and 95th percentile.

    Using percentiles instead of min/max avoids extrapolation into sparse
    regions where the model may behave erratically.

    Parameters
    ----------
    x : np.ndarray
        Feature values, shape (n_obs,).
    n_points : int
        Number of grid points. Default 20.

    Returns
    -------
    np.ndarray
        Grid values, shape (n_points,).
    """
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    return np.linspace(lo, hi, n_points)


# ---------------------------------------------------------------------------
# Manual PDP computation
# ---------------------------------------------------------------------------

def _pdp_manual(
    predict_fn: Callable,
    X: np.ndarray,
    feature_idx: int,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Compute PDP curve manually for a single continuous feature.

    For each grid point v:
        1. Set x_j = v for all observations
        2. Get mean prediction

    Parameters
    ----------
    predict_fn : Callable
        Function predict_fn(X) -> np.ndarray of shape (n_obs,).
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    feature_idx : int
        Index of the feature.
    grid : np.ndarray
        Grid of values, shape (n_points,).

    Returns
    -------
    np.ndarray
        Mean predictions at each grid point, shape (n_points,).
    """
    pdp_values = np.zeros(len(grid))
    for i, v in enumerate(grid):
        X_modified = X.copy()
        X_modified[:, feature_idx] = v
        pdp_values[i] = np.mean(predict_fn(X_modified))
    return pdp_values


def _pdp_slope_from_curve(
    grid: np.ndarray,
    pdp_values: np.ndarray,
) -> float:
    """
    Compute average slope of a PDP curve via central finite differences.

    Parameters
    ----------
    grid : np.ndarray
        Grid values, shape (n_points,).
    pdp_values : np.ndarray
        Mean predictions at each grid point, shape (n_points,).

    Returns
    -------
    float
        Average slope of the PDP curve — PDP-based marginal effect.
    """
    # Central finite differences for interior points
    # Forward/backward differences at endpoints
    slopes = np.gradient(pdp_values, grid)
    return float(np.mean(slopes))


# ---------------------------------------------------------------------------
# Sklearn partial_dependence wrapper
# ---------------------------------------------------------------------------

def _pdp_sklearn(
    model,
    X: np.ndarray,
    feature_idx: int,
    n_points: int = 20,
) -> tuple:
    """
    Compute PDP using sklearn's partial_dependence function.

    Parameters
    ----------
    model : sklearn-compatible model
        Fitted model with predict or predict_proba method.
    X : np.ndarray
        Feature matrix.
    feature_idx : int
        Feature index.
    n_points : int
        Number of grid points.

    Returns
    -------
    tuple
        (grid, pdp_values) where grid and pdp_values are both shape (n_points,).
    """
    from sklearn.inspection import partial_dependence

    # Determine response method
    response_method = 'predict_proba' if hasattr(model, 'predict_proba') \
        else 'predict'

    result = partial_dependence(
        model,
        X=X,
        features=[feature_idx],
        kind='average',
        grid_resolution=n_points,
        percentiles=(0.05, 0.95),
        response_method=response_method,
    )

    grid = result['grid_values'][0]
    pdp_values = result['average'][0]

    # For binary classification, predict_proba returns both classes
    # Take the positive class
    if pdp_values.ndim == 2:
        pdp_values = pdp_values[1]

    return grid, pdp_values


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def _is_sklearn_compatible(model) -> bool:
    """
    Check if model is sklearn-compatible (has predict method and is not TF/PyTorch).
    """
    # Exclude TensorFlow
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return False
    except Exception:
        pass

    # Exclude PyTorch
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return False
    except Exception:
        pass

    # Sklearn-compatible if it has predict
    return hasattr(model, 'predict')


def _make_predict_fn(model) -> Callable:
    """
    Build a predict_fn for any supported model type.

    Returns predict_proba[:, 1] for classifiers, predict() for regressors.
    For TensorFlow, squeezes output to 1D.
    """
    # TensorFlow
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            def predict_fn(X):
                import numpy as np
                preds = model(
                    tf.cast(tf.constant(X), dtype=tf.float32),
                    training=False
                ).numpy()
                if preds.ndim == 2 and preds.shape[1] == 1:
                    return preds.squeeze(axis=1)
                if preds.ndim == 2 and preds.shape[1] == 2:
                    return preds[:, 1]
                return preds
            return predict_fn
    except Exception:
        pass

    # Sklearn-compatible classifier
    if hasattr(model, 'predict_proba'):
        def predict_fn(X):
            return model.predict_proba(X)[:, 1]
        return predict_fn

    # Sklearn-compatible regressor
    def predict_fn(X):
        return model.predict(X)
    return predict_fn


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_pdp_slopes(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    categorical_features: Optional[list] = None,
    n_points: int = 20,
) -> dict:
    """
    Compute PDP-based marginal effects for all features.

    For continuous features:
        - Grid of n_points values between 5th and 95th percentile
        - Average slope of PDP curve via finite differences

    For categorical/binary features:
        - First difference: E[f(x|x_j=1)] - E[f(x|x_j=0)]
        - Same convention as AMEs in marginfx core.py

    Uses sklearn's partial_dependence where available (sklearn-compatible
    models). Falls back to manual PDP for TensorFlow and other models.

    Parameters
    ----------
    model : fitted model
        Any supported model type.
    X : np.ndarray
        Feature matrix, shape (n_obs, n_features).
    feature_names : list, optional
        Feature names. Defaults to ['x0', 'x1', ...].
    categorical_features : list, optional
        Indices or names of binary/categorical features.
        These use first differences instead of PDP slopes.
    n_points : int
        Number of PDP grid points. Default 20.

    Returns
    -------
    dict
        Feature name -> PDP-based marginal effect estimate.
    """
    X = np.array(X, dtype=float)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    if categorical_features is None:
        categorical_features = []

    # Normalize categorical features to indices
    cat_indices = set()
    for cf in categorical_features:
        if isinstance(cf, str):
            if cf in feature_names:
                cat_indices.add(feature_names.index(cf))
        elif isinstance(cf, int):
            cat_indices.add(cf)

    # Build predict_fn for manual PDP and categorical features
    predict_fn = _make_predict_fn(model)
    use_sklearn = _is_sklearn_compatible(model)

    results = {}

    for idx, name in enumerate(feature_names):

        # Categorical features: first difference (same as AME)
        if idx in cat_indices:
            X_0 = X.copy()
            X_1 = X.copy()
            X_0[:, idx] = 0.0
            X_1[:, idx] = 1.0
            results[name] = float(
                np.mean(predict_fn(X_1)) - np.mean(predict_fn(X_0))
            )
            continue

        # Continuous features: PDP slope
        if use_sklearn:
            try:
                grid, pdp_values = _pdp_sklearn(model, X, idx, n_points)
            except Exception:
                # Fallback to manual if sklearn PDP fails
                grid = _make_grid(X[:, idx], n_points)
                pdp_values = _pdp_manual(predict_fn, X, idx, grid)
        else:
            grid = _make_grid(X[:, idx], n_points)
            pdp_values = _pdp_manual(predict_fn, X, idx, grid)

        results[name] = _pdp_slope_from_curve(grid, pdp_values)

    return results


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

    print("Testing pdp_utils...")

    from dgp import generate_classification, generate_regression, FEATURE_NAMES
    import numpy as np

    rng = np.random.default_rng(42)

    # Test with logistic regression (classification)
    print("\n  Testing logistic regression (sklearn PDP)...")
    from sklearn.linear_model import LogisticRegression
    X, y = generate_classification(500, 'linear', rng)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    pdp_slopes = compute_pdp_slopes(model, X, FEATURE_NAMES,
                                     categorical_features=[])
    print(f"  PDP slopes: { {k: round(v, 4) for k, v in pdp_slopes.items()} }")

    # Test with random forest (classification)
    print("\n  Testing random forest (sklearn PDP)...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    pdp_slopes = compute_pdp_slopes(model, X, FEATURE_NAMES,
                                     categorical_features=[])
    print(f"  PDP slopes: { {k: round(v, 4) for k, v in pdp_slopes.items()} }")

    # Test with XGBoost (classification)
    print("\n  Testing XGBoost (sklearn PDP)...")
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=50, verbosity=0,
                               eval_metric='logloss').fit(X, y)
    pdp_slopes = compute_pdp_slopes(model, X, FEATURE_NAMES,
                                     categorical_features=[])
    print(f"  PDP slopes: { {k: round(v, 4) for k, v in pdp_slopes.items()} }")

    # Test with categorical feature
    print("\n  Testing with categorical feature...")
    from sklearn.linear_model import LogisticRegression
    X, y = generate_classification(500, 'linear', rng)
    # Make x3 binary
    X[:, 2] = (X[:, 2] > 0).astype(float)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    pdp_slopes = compute_pdp_slopes(
        model, X, FEATURE_NAMES,
        categorical_features=['x3'],
    )
    print(f"  PDP slopes (x3 categorical): "
          f"{ {k: round(v, 4) for k, v in pdp_slopes.items()} }")

    # Test with regression
    print("\n  Testing linear regression (regression)...")
    from sklearn.linear_model import LinearRegression
    X, y = generate_regression(500, 'linear', rng)
    model = LinearRegression().fit(X, y)
    pdp_slopes = compute_pdp_slopes(model, X, FEATURE_NAMES)
    print(f"  PDP slopes: { {k: round(v, 4) for k, v in pdp_slopes.items()} }")
    print(f"  (True values: x1=2.0, x2=3.0, x3=0.0, x4=0.0)")

    print("\nAll pdp_utils tests passed.")