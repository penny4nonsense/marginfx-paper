"""
dgp.py
------
Data generating processes (DGPs) for marginfx simulation studies.

Three DGP structures, two outcome types each:

    Regression:
        linear:      y = 2*x1 + 3*x2 + eps
        nonlinear:   y = 2*x1^2 + 3*x2 + eps
        interaction: y = 2*x1 + 3*x2 + 2*x1*x2 + eps

    Classification (same linear predictors, wrapped in sigmoid):
        linear:      P(y=1) = sigmoid(2*x1 + 3*x2)
        nonlinear:   P(y=1) = sigmoid(2*x1^2 + 3*x2)
        interaction: P(y=1) = sigmoid(2*x1 + 3*x2 + 2*x1*x2)

Features:
    x1, x2 ~ N(0, 1) — have true effects
    x3, x4 ~ N(0, 1) — noise features, true AME = 0

True AMEs:
    Regression:
        linear:      AME(x1) = 2.0, AME(x2) = 3.0
        nonlinear:   AME(x1) = E[2*2*x1] = 0 (symmetric), AME(x2) = 3.0
        interaction: AME(x1) = 2 + E[2*x2] = 2.0, AME(x2) = 3 + E[2*x1] = 3.0

    Classification:
        No closed form — computed via Monte Carlo on N_GROUND_TRUTH observations.

Ground truth:
    All true AMEs are computed via Monte Carlo using N_GROUND_TRUTH = 1,000,000
    observations. This is computed once and saved to GROUND_TRUTH_DIR.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def generate_features(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate feature matrix X with 4 standard normal features.

    x1, x2 have true effects. x3, x4 are noise.

    Parameters
    ----------
    n : int
        Number of observations.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n, 4).
    """
    return rng.standard_normal((n, 4))


# ---------------------------------------------------------------------------
# Linear predictors
# ---------------------------------------------------------------------------

def _linear_predictor(X: np.ndarray, beta1: float, beta2: float) -> np.ndarray:
    """f(X) = beta1*x1 + beta2*x2"""
    return beta1 * X[:, 0] + beta2 * X[:, 1]


def _nonlinear_predictor(X: np.ndarray, beta1: float, beta2: float) -> np.ndarray:
    """f(X) = beta1*x1^2 + beta2*x2"""
    return beta1 * X[:, 0] ** 2 + beta2 * X[:, 1]


def _interaction_predictor(X: np.ndarray, beta1: float, beta2: float) -> np.ndarray:
    """f(X) = beta1*x1 + beta2*x2 + beta1*x1*x2"""
    return beta1 * X[:, 0] + beta2 * X[:, 1] + beta1 * X[:, 0] * X[:, 1]


def _get_predictor(dgp_name: str):
    """Return the linear predictor function for a given DGP name."""
    predictors = {
        'linear':      _linear_predictor,
        'nonlinear':   _nonlinear_predictor,
        'interaction': _interaction_predictor,
    }
    if dgp_name not in predictors:
        raise ValueError(
            f"Unknown DGP: '{dgp_name}'. "
            f"Choose from: {list(predictors.keys())}"
        )
    return predictors[dgp_name]


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


# ---------------------------------------------------------------------------
# DGP samplers
# ---------------------------------------------------------------------------

def generate_regression(
    n: int,
    dgp_name: str,
    rng: np.random.Generator,
    beta1: float = 2.0,
    beta2: float = 3.0,
    noise_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate regression data from a given DGP.

    Parameters
    ----------
    n : int
        Number of observations.
    dgp_name : str
        One of 'linear', 'nonlinear', 'interaction'.
    rng : np.random.Generator
        Random number generator.
    beta1 : float
        Coefficient for x1. Default 2.0.
    beta2 : float
        Coefficient for x2. Default 3.0.
    noise_std : float
        Standard deviation of Gaussian noise. Default 1.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X, y) where X is (n, 4) and y is (n,).
    """
    X = generate_features(n, rng)
    predictor = _get_predictor(dgp_name)
    mu = predictor(X, beta1, beta2)
    y = mu + rng.normal(0, noise_std, size=n)
    return X, y


def generate_classification(
    n: int,
    dgp_name: str,
    rng: np.random.Generator,
    beta1: float = 2.0,
    beta2: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary classification data from a given DGP.

    P(y=1) = sigmoid(linear_predictor(X))

    Parameters
    ----------
    n : int
        Number of observations.
    dgp_name : str
        One of 'linear', 'nonlinear', 'interaction'.
    rng : np.random.Generator
        Random number generator.
    beta1 : float
        Coefficient for x1. Default 2.0.
    beta2 : float
        Coefficient for x2. Default 3.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X, y) where X is (n, 4) and y is (n,) binary.
    """
    X = generate_features(n, rng)
    predictor = _get_predictor(dgp_name)
    prob = _sigmoid(predictor(X, beta1, beta2))
    y = rng.binomial(1, prob).astype(float)
    return X, y


def generate_data(
    n: int,
    dgp_name: str,
    outcome_type: str,
    rng: np.random.Generator,
    beta1: float = 2.0,
    beta2: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data from a given DGP and outcome type.

    Unified entry point — dispatches to generate_regression or
    generate_classification based on outcome_type.

    Parameters
    ----------
    n : int
        Number of observations.
    dgp_name : str
        One of 'linear', 'nonlinear', 'interaction'.
    outcome_type : str
        One of 'regression', 'classification'.
    rng : np.random.Generator
        Random number generator.
    beta1 : float
        Coefficient for x1. Default 2.0.
    beta2 : float
        Coefficient for x2. Default 3.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X, y)
    """
    if outcome_type == 'regression':
        return generate_regression(n, dgp_name, rng, beta1, beta2)
    elif outcome_type == 'classification':
        return generate_classification(n, dgp_name, rng, beta1, beta2)
    else:
        raise ValueError(
            f"Unknown outcome_type: '{outcome_type}'. "
            f"Choose from: 'regression', 'classification'."
        )


# ---------------------------------------------------------------------------
# Ground truth AME computation
# ---------------------------------------------------------------------------

def compute_ground_truth_ames(
    dgp_name: str,
    outcome_type: str,
    n: int = 1_000_000,
    seed: int = 0,
    beta1: float = 2.0,
    beta2: float = 3.0,
    h: str = 'adaptive',
) -> dict:
    """
    Compute ground truth AMEs via Monte Carlo on a very large dataset.

    For regression DGPs, applies the true prediction function directly.
    For classification DGPs, uses the true sigmoid probability function.

    At n=1,000,000, Monte Carlo error is negligible and the result can
    be treated as the true AME.

    Parameters
    ----------
    dgp_name : str
        One of 'linear', 'nonlinear', 'interaction'.
    outcome_type : str
        One of 'regression', 'classification'.
    n : int
        Number of observations for Monte Carlo. Default 1,000,000.
    seed : int
        Random seed. Default 0.
    beta1 : float
        Coefficient for x1. Default 2.0.
    beta2 : float
        Coefficient for x2. Default 3.0.
    h : str or float
        Step size for finite differences. Default 'adaptive'.

    Returns
    -------
    dict
        Feature name -> true AME.
    """
    from marginfx.core import all_ames

    rng = np.random.default_rng(seed)
    X = generate_features(n, rng)
    predictor = _get_predictor(dgp_name)

    # True prediction function — no model, just the DGP
    if outcome_type == 'regression':
        def true_predict_fn(X_input):
            return predictor(X_input, beta1, beta2)
    else:
        def true_predict_fn(X_input):
            return _sigmoid(predictor(X_input, beta1, beta2))

    feature_names = ['x1', 'x2', 'x3', 'x4']

    true_ames = all_ames(
        X=X,
        predict_fn=true_predict_fn,
        feature_names=feature_names,
        h=h,
    )

    return true_ames


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

FEATURE_NAMES = ['x1', 'x2', 'x3', 'x4']


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '', '..'))

    print("Testing DGP generation...")
    rng = np.random.default_rng(42)

    for dgp in ['linear', 'nonlinear', 'interaction']:
        for outcome in ['regression', 'classification']:
            X, y = generate_data(500, dgp, outcome, rng)
            assert X.shape == (500, 4)
            assert y.shape == (500,)
            if outcome == 'classification':
                assert set(np.unique(y)).issubset({0.0, 1.0})
            print(f"  {dgp:12} {outcome:14} — X{X.shape} y{y.shape} OK")

    print("\nComputing ground truth AMEs (n=10,000 for quick test)...")
    for dgp in ['linear', 'nonlinear', 'interaction']:
        for outcome in ['regression', 'classification']:
            true_ames = compute_ground_truth_ames(
                dgp, outcome, n=10_000, seed=0
            )
            print(f"  {dgp:12} {outcome:14} — {true_ames}")

    print("\nAll DGP tests passed.")
