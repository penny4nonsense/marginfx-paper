"""
icdm_bootstrap.py
-----------------
The refitting bootstrap of the ICDM paper's Algorithm 1, done correctly.

Why this exists. The published tables were produced by a bootstrap that
warm-started each replicate from the full-sample fit. That path was broken in
four separate ways, and only the logistic column escaped:

  * random forests were never actually refit, so the replicate spread was
    essentially zero and every forest coefficient acquired three stars;
  * XGBoost replicates continued training from the fitted model, ending at
    200 boosting rounds against the base model's 100;
  * neural network replicates trained for 10 epochs against a base model that
    trains up to 500 with early stopping;
  * for regression, the replicate was rebuilt from the Keras config, and
    UnscaledModel.get_config() returns the INNER network's config, so the
    target-rescaling wrapper was silently dropped and replicates predicted on
    the standardized scale.

Every one of these is a discrepancy between how the base model is built and how
a replicate is built. So this module does not reimplement model construction at
all: it calls the pipeline's own fit_model() for each replicate, which makes the
two identical by construction. Architecture, epoch budget, early stopping,
learning rate and target standardization all come along for free, and any future
change to fit_model automatically applies to the replicates too.

What is NOT changed is the estimator, which is reproduced exactly as the
accepted paper's code computed it (marginfx at 8086697, the revision that
produced the published tables):

  * each replicate is evaluated at the resample, not at the observed sample;
  * the adaptive step size is recomputed on the resample, so the whole
    procedure is bootstrapped rather than only the fit;
  * no trimming weight is applied, because all_ames had none at that revision.

The first of these is worth a note, since the paper's Section II-C says the
opposite -- that the replicate is evaluated at D rather than D^(b). The code
has always evaluated at D^(b), and the published numbers come from the code.
Evaluating at D would drop the sampling variability of the average of the
derivative over the covariates, which is zero only when the fitted derivative
is constant in X; measured on this design it costs the logistic coefficient on
the largest signal roughly half its standard error. So the code is right and
the sentence is wrong, and the sentence is what should change.
"""

import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marginfx.core import plugin_ames
from marginfx.learner import make_predict_fn


# Workers per learner, and replicates built per worker task. Keras is given
# fewer of both: a process that has built many networks does not hand the
# memory back, and the worker is only replaced between tasks.
N_JOBS = {'tensorflow': 10}
N_JOBS_DEFAULT = 24
CHUNK = {'tensorflow': 8}
CHUNK_DEFAULT = 25


def _quiet_tf(model_name):
    if model_name != 'tensorflow':
        return
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.disable_interactive_logging()
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # Already initialized in this process; the limits are then whatever
        # the first call set, which is what we want anyway.
        pass


def _limit_threads():
    """
    Confine a worker to one thread per library.

    The bootstrap already parallelizes across replicates, so any threading
    inside a single fit is oversubscription: without this, twenty-four workers
    each ask for every core.
    """
    for var in ('MFX_INNER_JOBS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'


def _replicate_chunk(model_name, X, y, outcome_type, feature_names,
                     cat_feats, seeds):
    """Refit the learner on one slice of resamples and return their AMEs."""
    # Before importing analyze: fit_model reads MFX_INNER_JOBS at import time.
    _limit_threads()
    from analyze import fit_model

    _quiet_tf(model_name)
    n = X.shape[0]
    out = []
    for s in seeds:
        rs = np.random.default_rng(s)
        idx = rs.integers(0, n, size=n)
        X_boot, y_boot = X[idx], y[idx]
        model = fit_model(model_name, X_boot, y_boot, outcome_type,
                          int(s % (2 ** 31 - 1)))
        # Evaluated at the resample, with h recomputed there: the whole
        # procedure is bootstrapped, not just the fit.
        out.append(plugin_ames(
            X=X_boot, predict_fn=make_predict_fn(model),
            feature_names=feature_names, categorical_features=cat_feats,
            h='adaptive', trim=False,
        ))
        del model
        _quiet_tf(model_name)
    gc.collect()
    return out


def bootstrap_cell(base_model, model_name, X, y, outcome_type, feature_names,
                   cat_feats, n_bootstrap=200, seed=42, verbose=True):
    """
    Point estimates and refitting-bootstrap standard errors for one cell.

    Parameters
    ----------
    base_model : fitted model
        The full-sample fit, already built by fit_model.
    model_name : str
        Passed to fit_model when building each replicate.
    X, y : np.ndarray
    outcome_type : {'classification', 'regression'}
    feature_names : list of str
    cat_feats : list of str
    n_bootstrap : int
    seed : int
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Columns: term, estimate, std_error, statistic, p_value,
        conf_low, conf_high -- the schema of the published results.
    """
    from joblib import Parallel, delayed
    from scipy import stats

    point = plugin_ames(
        X=X, predict_fn=make_predict_fn(base_model),
        feature_names=feature_names, categorical_features=cat_feats,
        h='adaptive', trim=False,
    )
    if n_bootstrap <= 0:
        raise ValueError('n_bootstrap must be positive')

    ss = np.random.SeedSequence(seed)
    seeds = [int(c.generate_state(1)[0]) for c in ss.spawn(n_bootstrap)]

    size = CHUNK.get(model_name, CHUNK_DEFAULT)
    n_jobs = N_JOBS.get(model_name, N_JOBS_DEFAULT)
    chunks = [seeds[i:i + size] for i in range(0, len(seeds), size)]

    if verbose:
        print(f"      bootstrap: {n_bootstrap} replicates, "
              f"{len(chunks)} chunks, {n_jobs} workers", flush=True)

    # Set here, after the full-sample fit above has had the whole machine, and
    # before any worker is spawned: children inherit the environment, and the
    # BLAS thread count is fixed when the library first loads in the child.
    _limit_threads()

    # maxtasksperchild=1 and batch_size=1 for the same reason as in the
    # simulations: a Keras worker's footprint must be bounded by the chunk
    # size rather than by the replicate count.
    results = Parallel(n_jobs=n_jobs, maxtasksperchild=1, batch_size=1)(
        delayed(_replicate_chunk)(
            model_name, X, y, outcome_type, feature_names, cat_feats, chunk,
        )
        for chunk in chunks
    )
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass

    replicates = [r for chunk in results for r in chunk]

    rows = []
    for j, name in enumerate(feature_names):
        dist = np.asarray([r[name] for r in replicates], dtype=float)
        est = float(point[name])
        se = float(np.std(dist, ddof=1))
        z = est / se if se > 0 else np.nan
        rows.append({
            'term':       name,
            'estimate':   est,
            'std_error':  se,
            'statistic':  z,
            'p_value':    float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan,
            'conf_low':   float(np.percentile(dist, 2.5)),
            'conf_high':  float(np.percentile(dist, 97.5)),
        })
    return pd.DataFrame(rows)
