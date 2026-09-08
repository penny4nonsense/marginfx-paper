"""
calibration_models.py
---------------------
Model builders for Simulation 2, in one place.

These were previously defined separately inside run_calibration.py and
run_calibration_regression.py. They are lifted out here because the parallel
bootstrap needs worker processes to construct models themselves: shipping a
fitted model across a process boundary is unreliable for Keras and needless
for everything else, whereas a model name and a seed always travel.

The hyperparameters are unchanged from the two scripts. Any edit here changes
both outcome types, which is the point.
"""

import numpy as np

from config import (
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    TF_HIDDEN_UNITS,
    TF_EPOCHS,
    TF_BATCH_SIZE,
    TF_LEARNING_RATE,
)


def _build_classification(model_name: str, seed: int, X: np.ndarray, y: np.ndarray):
    """Build and fit a classification learner."""
    if model_name == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=seed)

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,
        )

    elif model_name == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=seed,
            verbosity=0,
            eval_metric='logloss',
            n_jobs=1,
        )

    elif model_name == 'tensorflow':
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.disable_interactive_logging()

        layers = [
            tf.keras.layers.Dense(units, activation='relu')
            for units in TF_HIDDEN_UNITS
        ]
        layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TF_LEARNING_RATE),
            loss='binary_crossentropy',
        )
        model.fit(X, y, epochs=TF_EPOCHS, batch_size=TF_BATCH_SIZE, verbose=0)
        return model

    else:
        raise ValueError(f"Unknown model: '{model_name}'")

    model.fit(X, y)
    return model


def _build_regression(model_name: str, seed: int, X: np.ndarray, y: np.ndarray):
    """
    Build and fit a regression learner.

    The network trains on a standardized target and is returned wrapped so
    that it predicts in the original units. The wrapper is applied here, on
    every fit including bootstrap resamples, rather than being reconstructed
    from a Keras config later: UnscaledModel.get_config() returns the inner
    network's config, so anything that rebuilds from it silently drops the
    rescaling and predicts on the standardized scale.
    """
    if model_name == 'linear':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=seed,
            n_jobs=1,
        )

    elif model_name == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )

    elif model_name == 'tensorflow':
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.disable_interactive_logging()

        y_mean = float(np.mean(y))
        y_scale = float(np.std(y))
        if y_scale <= 0:
            y_scale = 1.0
        y_fit = (y - y_mean) / y_scale

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(X.astype(np.float32))

        inner = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation=None),
        ])
        inner.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=float(TF_LEARNING_RATE)
            ),
            loss='mse',
        )
        inner.fit(
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

        captured_inner, mean_val, scale_val = inner, y_mean, y_scale

        class UnscaledModel(tf.keras.Model):
            """Predicts in the original units of y."""

            def __init__(self):
                super().__init__()
                self._inner = captured_inner
                self._mean = mean_val
                self._scale = scale_val
                self.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=float(TF_LEARNING_RATE)
                    ),
                    loss='mse',
                )

            def call(self, X_in, training=False):
                return self._inner(
                    tf.cast(X_in, dtype=tf.float32), training=training
                ) * self._scale + self._mean

            def get_weights(self):
                return self._inner.get_weights()

            def get_config(self):
                return self._inner.get_config()

        return UnscaledModel()

    else:
        raise ValueError(f"Unknown model: '{model_name}'")

    model.fit(X, y)
    return model


def build_fitted_model(outcome_type: str, model_name: str, seed: int,
                       X: np.ndarray, y: np.ndarray):
    """
    Build and fit one learner on the data given.

    Parameters
    ----------
    outcome_type : str
        'classification' or 'regression'.
    model_name : str
        'logistic'/'linear', 'rf', 'xgboost', 'tensorflow'.
    seed : int
        Random seed.
    X, y : np.ndarray
        Training data. For a bootstrap replicate, pass the resample.

    Returns
    -------
    A fitted model whose predictions are on the natural outcome scale.
    """
    if outcome_type == 'classification':
        return _build_classification(model_name, seed, X, y)
    return _build_regression(model_name, seed, X, y)
