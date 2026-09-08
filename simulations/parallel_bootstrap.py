"""
parallel_bootstrap.py
---------------------
A refitting bootstrap that parallelizes across resamples rather than across
Monte Carlo iterations.

Why this exists. Simulation 2's natural unit of parallel work is one Monte
Carlo iteration, and for most learners that is the right unit. For Keras it is
not. One iteration draws n_bootstrap resamples and builds a network on each,
and a process that has built a few hundred Keras models does not give that
memory back -- clear_session() does not recover it, and the worker is only
replaced between tasks, so nothing is reclaimed until the whole iteration
finishes. With twenty-four workers each part-way through their own iteration,
the machine ran out of memory.

Splitting the other way fixes it. Each task here is one slice of one
iteration's resamples, so a worker builds a few dozen models rather than a few
hundred, and its footprint is bounded by the slice size instead of by
n_bootstrap. The estimator is unchanged: same resampling scheme, same plug-in,
same percentile intervals.

Where the replicate is evaluated is set by `eval_at`, and it matters. A version
of this file evaluated every replicate at the original sample while resampling
only the fit. That drops the sampling variability of the average of the
derivative over the covariates, which is zero only when the fitted derivative
is constant in X, and it cost the logistic coefficient on the largest signal
about half its standard error on the Gaussian design. The default is now
'resample', the textbook bootstrap.

Two details differ from the serial path, both deliberate:

* Each replicate draws from its own stream, spawned from the iteration's seed
  by SeedSequence, instead of consuming one generator in sequence. Independent
  streams are what the bootstrap assumes; the serial order was never part of
  the estimator.
* Replicates are refit through calibration_models.build_fitted_model rather
  than through the engine's refit path. That is what keeps the regression
  network's rescaling wrapper attached, which reconstructing from a Keras
  config does not.
"""

import numpy as np

from marginfx.core import plugin_ames, resolve_h, support_bounds
from marginfx.learner import make_predict_fn

from calibration_models import build_fitted_model
from config import BASE_SEED
from dgp import generate_classification, generate_regression


def _make_data(outcome_type: str, n: int, dgp_name: str, iteration: int):
    """
    Regenerate one iteration's dataset from its seed.

    Cheaper and safer than shipping arrays to every worker, and identical by
    construction to what the serial path generates.
    """
    seed = BASE_SEED + iteration
    rng = np.random.default_rng(seed)
    if outcome_type == 'classification':
        X, y = generate_classification(n, dgp_name, rng)
    else:
        X, y = generate_regression(n, dgp_name, rng)
    return X, y, seed


def _quiet_tensorflow(model_name: str) -> None:
    if model_name != 'tensorflow':
        return
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.disable_interactive_logging()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def replicate_chunk(outcome_type, model_name, n, dgp_name, iteration,
                    feature_names, replicate_seeds, want_point, trim=False,
                    eval_at='resample', refit_h=False):
    """
    Run one slice of an iteration's bootstrap resamples.

    Parameters
    ----------
    outcome_type, model_name, n, dgp_name, iteration
        Identify the cell and the Monte Carlo draw.
    feature_names : list of str
    replicate_seeds : list of int
        One seed per resample in this slice. May be empty.
    want_point : bool
        Whether this slice should also fit the model on the original sample
        and return the point estimates. Exactly one slice per iteration sets
        this.
    trim : bool
        Trimming weight. False for the Gaussian designs.
    eval_at : {'resample', 'original'}
        Where a replicate's plug-in average is evaluated. 'resample' is the
        textbook bootstrap and the default: the estimator is an average over
        the empirical distribution, so the bootstrap must resample that average
        as well as the fit. 'original' holds the evaluation sample fixed, which
        drops the sampling variability of the average of the derivative over X.
        That component is exactly zero when the fitted derivative is constant
        in X -- any linear-regression fit -- and grows with how much it varies,
        so 'original' understates the standard error for every nonlinear or
        classification fit, in proportion to the heterogeneity of the
        derivative. It is retained only to reproduce results computed that way.
    refit_h : bool
        Whether the adaptive step size is recomputed on each resample. False
        for the journal paper, where h is fixed by the definition of the window
        estimand and is not estimated. True for the conference paper, where the
        adaptive step size is part of the estimator and so is bootstrapped
        along with everything else.

    Returns
    -------
    dict with 'point' (dict or None) and 'replicates' (list of dict).
    """
    import gc

    _quiet_tensorflow(model_name)
    X, y, seed = _make_data(outcome_type, n, dgp_name, iteration)

    h_values = resolve_h(X, 'adaptive')
    bounds = support_bounds(X)
    n_obs = X.shape[0]

    point = None
    if want_point:
        base = build_fitted_model(outcome_type, model_name, seed, X, y)
        point = plugin_ames(
            X=X, predict_fn=make_predict_fn(base),
            feature_names=feature_names, h=h_values,
            trim=trim, bounds=bounds,
        )
        del base
        _quiet_tensorflow(model_name)

    replicates = []
    for rep_seed in replicate_seeds:
        rs = np.random.default_rng(rep_seed)
        idx = rs.integers(0, n_obs, size=n_obs)
        X_boot, y_boot = X[idx], y[idx]
        refit = build_fitted_model(
            outcome_type, model_name, seed, X_boot, y_boot
        )
        X_eval = X_boot if eval_at == 'resample' else X
        replicates.append(plugin_ames(
            X=X_eval, predict_fn=make_predict_fn(refit),
            feature_names=feature_names,
            h='adaptive' if refit_h else h_values,
            trim=trim,
            bounds=support_bounds(X_eval) if trim else bounds,
        ))
        del refit
        _quiet_tensorflow(model_name)

    gc.collect()
    return {'point': point, 'replicates': replicates}


def plan_iteration(iteration, n_bootstrap, n_chunks):
    """
    Split one iteration's resamples into slices, with their seeds.

    Seeds are spawned from the iteration's own seed, so a given iteration
    draws the same resamples however the work is divided.

    Returns
    -------
    list of (replicate_seeds, want_point)
    """
    ss = np.random.SeedSequence(BASE_SEED + iteration)
    seeds = [int(c.generate_state(1)[0]) for c in ss.spawn(n_bootstrap)]

    n_chunks = max(1, min(n_chunks, n_bootstrap))
    edges = np.linspace(0, n_bootstrap, n_chunks + 1).astype(int)
    return [
        (seeds[edges[k]:edges[k + 1]], k == 0)
        for k in range(n_chunks)
    ]


def assemble(point, replicates, feature_names, alpha=0.05):
    """
    Turn one iteration's point estimate and replicates into the same summary
    the serial path produces: percentile intervals and the replicate standard
    deviation.

    Returns
    -------
    dict of feature -> dict(estimate, se, conf_low, conf_high)
    """
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    out = {}
    for name in feature_names:
        dist = np.asarray([r[name] for r in replicates], dtype=float)
        out[name] = {
            'estimate': float(point[name]),
            'se': float(np.std(dist, ddof=1)),
            'conf_low': float(np.percentile(dist, lower_pct)),
            'conf_high': float(np.percentile(dist, upper_pct)),
        }
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Models built per task. This is the memory knob: a worker's footprint is set
# by how many networks it builds before the process is replaced, not by
# n_bootstrap. Fifty keeps a Keras worker near 1.5-2 GB, so a full complement
# of workers fits comfortably, while still amortizing the TensorFlow import
# that every new process pays.
CHUNK_MODELS = 50


def run_iterations(outcome_type, model_name, n, dgp_name, iterations,
                   true_ames, feature_names, n_bootstrap, n_jobs,
                   trim=False, alpha=0.05, chunk_models=CHUNK_MODELS,
                   progress=None, eval_at='resample', refit_h=False):
    """
    Run a set of Monte Carlo iterations, parallelizing across resamples.

    Work is dispatched as (iteration, slice-of-resamples) pairs so that many
    iterations are in flight at once and every worker stays busy even though
    a single iteration is split several ways.

    Parameters
    ----------
    iterations : list of int
        Which Monte Carlo iterations to run.
    true_ames : dict
        Ground truth, for the coverage columns.
    n_jobs : int
        Worker processes.
    progress : callable, optional
        Called with the number of iterations completed so far.

    Returns
    -------
    list of dict
        One row per iteration per feature, matching the serial path's schema.
    """
    from joblib import Parallel, delayed

    n_chunks = max(1, int(np.ceil(n_bootstrap / float(chunk_models))))

    jobs, owners = [], []
    for it in iterations:
        for seeds, want_point in plan_iteration(it, n_bootstrap, n_chunks):
            owners.append(it)
            jobs.append(delayed(replicate_chunk)(
                outcome_type, model_name, n, dgp_name, it,
                feature_names, seeds, want_point, trim, eval_at, refit_h,
            ))

    # maxtasksperchild=1: one chunk per process, so the memory a worker
    # accumulates building its models is handed back as soon as the chunk ends.
    #
    # batch_size=1 stops joblib from bundling several chunks into a single
    # dispatch. Its automatic batching is tuned for tasks that are short
    # relative to their dispatch cost, which these are not, and a bundle of
    # chunks would put the whole bundle's worth of models into one process --
    # exactly the concentration this split exists to avoid.
    results = Parallel(
        n_jobs=n_jobs, maxtasksperchild=1, batch_size=1,
    )(jobs)

    # Tear the pool down rather than leaving it for the next call. This
    # function is invoked once per batch of iterations, and a pool that
    # survives between batches was observed carrying memory forward: one
    # batch in isolation peaked at 42 GB, but consecutive batches reached
    # 97 GB with 9 GB free.
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass

    gathered = {it: {'point': None, 'replicates': []} for it in iterations}
    for it, res in zip(owners, results):
        if res['point'] is not None:
            gathered[it]['point'] = res['point']
        gathered[it]['replicates'].extend(res['replicates'])

    rows = []
    for done, it in enumerate(iterations, start=1):
        g = gathered[it]
        if g['point'] is None or not g['replicates']:
            continue
        summary = assemble(g['point'], g['replicates'], feature_names, alpha)
        for feature in feature_names:
            s = summary[feature]
            true_val = true_ames[feature]
            rows.append({
                'iteration':    it,
                'dgp':          dgp_name,
                'n':            n,
                'model':        model_name,
                'feature':      feature,
                'ame_estimate': s['estimate'],
                'true_ame':     true_val,
                'bias':         s['estimate'] - true_val,
                'se':           s['se'],
                'conf_low':     s['conf_low'],
                'conf_high':    s['conf_high'],
                'covered':      int(s['conf_low'] <= true_val <= s['conf_high']),
                'ci_width':     s['conf_high'] - s['conf_low'],
                'elapsed':      float('nan'),
            })
        if progress is not None:
            progress(done)

    return rows
