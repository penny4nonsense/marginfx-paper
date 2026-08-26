"""
analyze.py
----------
Shared analysis engine for marginfx empirical examples.

Provides functions for:
    - Fitting models (logistic/linear, random forest, XGBoost, TensorFlow)
    - Computing debiased cross-fitted AMEs across all specifications
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

def make_learner(
    model_name: str,
    outcome_type: str,
    seed: int = 42,
):
    """
    Build an UNFITTED learner for the debiased cross-fitted estimator.

    mfx.fit refits the learner once per fold, so it takes a specification
    rather than a trained model.

    Parameters
    ----------
    model_name : str
        One of 'logistic', 'linear', 'rf', 'xgboost', 'tensorflow'.
    outcome_type : str
        One of 'classification', 'regression'.
    seed : int
        Random seed.

    Returns
    -------
    An unfitted scikit-learn compatible estimator, or a KerasLearner.
    """
    if model_name in ('logistic', 'linear'):
        if outcome_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=5000, random_state=seed)
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    elif model_name == 'rf':
        if outcome_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=200, max_depth=8,
                random_state=seed, n_jobs=-1,
            )
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, max_depth=8,
            random_state=seed, n_jobs=-1,
        )

    elif model_name == 'xgboost':
        import xgboost as xgb
        if outcome_type == 'classification':
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=seed,
                verbosity=0, eval_metric='logloss',
            )
        return xgb.XGBRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=seed,
            verbosity=0,
        )

    elif model_name == 'tensorflow':
        return KerasLearner(outcome_type, seed=seed)

    else:
        raise ValueError(f"Unknown model: '{model_name}'")


# ---------------------------------------------------------------------------
# Keras network: architecture and training recipe
#
# Shared by fit_model, which trains once on the full sample for the fit
# statistics, SHAP and PDP, and by KerasLearner, which is refit from scratch
# on each cross-fitting fold.
# ---------------------------------------------------------------------------

def _keras_scaling(y: np.ndarray, outcome_type: str):
    """
    Return (mean, scale) for the target.

    Regression standardizes the outcome so gradient updates are scale
    invariant; classification trains on the raw 0/1 labels.
    """
    if outcome_type == 'regression':
        scale = float(np.std(y))
        return float(np.mean(y)), (scale if scale > 0 else 1.0)
    return 0.0, 1.0


def _build_keras(X: np.ndarray, outcome_type: str, seed: int):
    """Build and compile the feedforward network for this outcome type."""
    import tensorflow as tf
    tf.random.set_seed(seed)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X.astype(np.float32))

    if outcome_type == 'regression':
        layers = [
            normalizer,
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=None),
        ]
        lr, loss = 3e-4, 'mse'
    else:
        layers = [
            normalizer,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
        lr, loss = float(1e-3), 'binary_crossentropy'

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
    )
    return model


def _fit_keras(model, X: np.ndarray, y_fit: np.ndarray, outcome_type: str):
    """Train with early stopping on a 10% validation split."""
    import tensorflow as tf
    patience = 50 if outcome_type == 'regression' else 20
    model.fit(
        X.astype(np.float32), y_fit.astype(np.float32),
        epochs=500, batch_size=32, verbose=0,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=patience, restore_best_weights=True,
            monitor='val_loss',
        )],
    )
    return model


class KerasLearner:
    """
    The feedforward network as an unfitted fit/predict learner.

    Cross-fitting trains the learner from scratch on each fold, so marginfx
    needs a specification it can instantiate rather than a trained Keras
    model. This wrapper carries the architecture, the early stopping
    schedule and the target standardization, and predicts on the natural
    outcome scale.

    Parameters
    ----------
    outcome_type : str
        One of 'classification', 'regression'.
    seed : int
        Random seed.
    """

    def __init__(self, outcome_type: str, seed: int = 42):
        self.outcome_type = outcome_type
        self.seed = seed
        self._model = None
        self._y_mean = 0.0
        self._y_scale = 1.0

    def fit(self, X, y):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._y_mean, self._y_scale = _keras_scaling(y, self.outcome_type)
        y_fit = (y - self._y_mean) / self._y_scale

        self._model = _build_keras(X, self.outcome_type, self.seed)
        _fit_keras(self._model, X, y_fit, self.outcome_type)
        return self

    def predict(self, X):
        pred = self._model.predict(
            np.asarray(X, dtype=float).astype(np.float32), verbose=0
        )
        return np.asarray(pred).squeeze() * self._y_scale + self._y_mean


def fit_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    outcome_type: str,
    seed: int = 42,
):
    """
    Build and fit a model on the full sample.

    Used for the fit statistics, SHAP and PDP, all of which need a single
    trained model. The AME estimator does not use this: it refits the
    specification from make_learner once per cross-fitting fold.

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
    if model_name == 'tensorflow':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        y_mean, y_scale = _keras_scaling(y, outcome_type)
        y_fit = (y - y_mean) / y_scale

        model = _build_keras(X, outcome_type, seed)
        _fit_keras(model, X, y_fit, outcome_type)

        if outcome_type == 'regression':
            y_mean_val = y_mean
            y_scale_val = y_scale
            inner_model = model

            class UnscaledModel(tf.keras.Model):
                def __init__(self):
                    super().__init__()
                    self._inner = inner_model
                    self._mean = y_mean_val
                    self._scale = y_scale_val
                    self.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=float(1e-3)),
                        loss='mse'
                    )

                def call(self, X, training=False):
                    return self._inner(
                        tf.cast(X, dtype=tf.float32),
                        training=training
                    ) * self._scale + self._mean

                def get_weights(self):
                    return self._inner.get_weights()

                def get_config(self):
                    return self._inner.get_config()

            unscaled = UnscaledModel()
            test_pred = unscaled(X[:5].astype(np.float32), training=False).numpy()
            print(f"Sample predictions: {test_pred.squeeze()}")
            print(f"Sample actuals: {y[:5]}")
            return unscaled

        return model

    model = make_learner(model_name, outcome_type, seed)
    model.fit(X, y)
    return model

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
    n_folds: int = 5,
    seed: int = 42,
    output_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute debiased cross-fitted AMEs for all models and specifications.

    For each specification and model:
        1. Select features for this specification
        2. Fit the model on the full sample for the fit statistics
        3. Compute fit statistics (McFadden R2 + Accuracy or R2 + RMSE)
        4. Compute AMEs and orthogonal-score standard errors, refitting the
           learner once per fold
        5. Collect results into tidy DataFrame
        6. Save partial results incrementally (if output_dir provided)

    Parameters
    ----------
    model_names : list of str
        Models to fit.
    df : pd.DataFrame
        Processed dataset with all features and outcome.
    specifications : dict
        Mapping of spec name -> list of feature names.
    outcome : str
        Name of outcome column.
    categorical_features : list of str
        Names of categorical/binary features.
    outcome_type : str
        One of 'classification', 'regression'.
    n_folds : int
        Cross-fitting folds K. Default 5.
    seed : int
        Random seed.
    output_dir : str, optional
        Directory to save partial results after each model completes.
    dataset_name : str, optional
        Dataset name used for partial results filename.

    Returns
    -------
    pd.DataFrame
        Tidy results with columns:
        model, spec, term, estimate, h, std_error,
        statistic, p_value, conf_low, conf_high, trimmed,
        fit_stat1, fit_stat1_name, fit_stat2, fit_stat2_name, n_obs.

        For classification: fit_stat1=mcfadden_r2, fit_stat2=accuracy
        For regression:     fit_stat1=r2,           fit_stat2=rmse
    """
    from sklearn.metrics import (
        log_loss, accuracy_score, r2_score, mean_squared_error
    )

    y = df[outcome].values.astype(float)
    n_obs = len(y)
    all_rows = []

    # Check for existing partial results
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
            completed = set(zip(existing['spec'], existing['model']))
            print(f"  Resuming from partial results "
                  f"({len(completed)} combinations already done)")

    for spec_name, features in specifications.items():
        print(f"\n  Specification {spec_name}: {features}")

        X = df[features].values.astype(float)
        cat_feats = [f for f in categorical_features if f in features]

        for model_name in model_names:

            if (spec_name, model_name) in completed:
                print(f"    Skipping {model_name} (already done)")
                continue

            print(f"    Fitting {model_name}...", end='', flush=True)
            t0 = time.time()

            try:
                model = fit_model(model_name, X, y, outcome_type, seed)

                # --- Fit statistics ---
                if outcome_type == 'classification':
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X)[:, 1]
                    else:
                        import tensorflow as tf
                        y_pred_proba = model(
                            tf.cast(tf.constant(X), dtype=tf.float32),
                            training=False
                        ).numpy().squeeze()

                    y_pred_class = (y_pred_proba > 0.5).astype(int)
                    null_ll = log_loss(y, np.full_like(y_pred_proba, y.mean()))
                    model_ll = log_loss(y, np.clip(y_pred_proba, 1e-10, 1 - 1e-10))
                    fit_stat1 = float(1 - model_ll / null_ll)  # McFadden R2
                    fit_stat2 = float(accuracy_score(y, y_pred_class))  # Accuracy
                    fit_stat1_name = 'mcfadden_r2'
                    fit_stat2_name = 'accuracy'

                else:
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X)
                    else:
                        import tensorflow as tf
                        y_pred = model(
                            tf.cast(tf.constant(X), dtype=tf.float32),
                            training=False
                        ).numpy().squeeze()

                    fit_stat1 = float(r2_score(y, y_pred))  # R2
                    fit_stat2 = float(np.sqrt(mean_squared_error(y, y_pred)))  # RMSE
                    fit_stat1_name = 'r2'
                    fit_stat2_name = 'rmse'

                # --- AMEs ---
                # The debiased estimator refits the learner on each fold, so
                # it takes the unfitted specification, not the model above.
                result = mfx.fit(
                    make_learner(model_name, outcome_type, seed),
                    X, y,
                    feature_names=features,
                    categorical_features=cat_feats,
                    n_folds=n_folds,
                    trim=True,
                    seed=seed,
                    verbose=False,
                )
                elapsed = time.time() - t0
                print(f" done ({elapsed:.1f}s) | "
                      f"{fit_stat1_name}={fit_stat1:.3f} "
                      f"{fit_stat2_name}={fit_stat2:.3f}")

                tidy = result.tidy()
                tidy['model'] = model_name
                tidy['spec'] = spec_name
                tidy['fit_stat1'] = fit_stat1
                tidy['fit_stat1_name'] = fit_stat1_name
                tidy['fit_stat2'] = fit_stat2
                tidy['fit_stat2_name'] = fit_stat2_name
                tidy['n_obs'] = n_obs
                all_rows.append(tidy)

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

    cols = ['model', 'spec', 'term', 'estimate', 'h', 'std_error',
            'statistic', 'p_value', 'conf_low', 'conf_high', 'trimmed',
            'fit_stat1', 'fit_stat1_name', 'fit_stat2', 'fit_stat2_name',
            'n_obs']
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
