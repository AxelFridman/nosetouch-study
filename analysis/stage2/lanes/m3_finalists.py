"""Entry points of lane m3's finalists for the synthesis step (results/finalists_m3.json):

    import importlib
    mod = importlib.import_module(entry['module'])            # 'stage2.lanes.m3_finalists'
    X = getattr(mod, entry['build_x'])(d)                      # per-trial input of a Stage2Data (no labels)
    fit_predict = getattr(mod, entry['factory'])(**entry['params'])
    cv.run_experiment(name, fit_predict, X, d.trials, ...)

Every factory is deterministic for a given fold (seeded from ``fold.seed``; deep networks always train and predict
with one intra-op thread and deterministic algorithms, ``m3_deep.torch_runtime``), so a refit gives the same
predictions bit for bit on any machine with the same library versions.

Fit cache. By default (``use_cache=True``) fitted folds are read from and written to ``analysis/data/m3_fit_cache``
(git-ignored; ``$M3_FIT_CACHE`` or ``cache_dir=...`` choose another folder), keyed by the code, the configuration,
the runtime (library versions, torch threads) and every training input: a synthesis run on the lane's own training
rows reuses the lane's fits, and any other training set is fitted afresh. ``use_cache=False`` never touches the disk
(the setting also holds for the inner fits of the nested and grip-subset wrappers). Example for a synthesis that
must not write into the lane's folder::

    fit_predict = getattr(mod, entry['factory'])(**{**entry['params'], 'cache_dir': '<synthesis cache folder>'})
"""
from __future__ import annotations

from typing import Any

from stage2.lanes.m3_body import x_approach_body, x_fixed_body  # noqa: F401  (build_x entry points)
from stage2.lanes.m3_common import x_approach, x_fixed  # noqa: F401  (build_x entry points)
from stage2.lanes.m3_tf import x_cwt, x_stft  # noqa: F401  (build_x entry points)


def deep(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """Deep sequence model (CV ensemble of ``n_inner`` members, early-stopped on training persons)."""
    from stage2.lanes import m3_deep
    return m3_deep.factory(calib=calib, use_cache=use_cache, cache_dir=cache_dir, **params)


def deep_body(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """Deep sequence model on the gravity + heading-following body frame (grip-invariant; input ``x_approach_body``
    or ``x_fixed_body``)."""
    from stage2.lanes import m3_body
    return m3_body.factory(calib=calib, use_cache=use_cache, cache_dir=cache_dir, **params)


def minirocket(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """MiniRocket + L2 logistic regression (C nested on fold.inner_splits())."""
    from stage2.lanes import m3_minirocket
    return m3_minirocket.factory(calib=calib, use_cache=use_cache, cache_dir=cache_dir, **params)


def tflog(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """L2 logistic regression on pooled signed STFT / CWT planes (C nested on fold.inner_splits())."""
    from stage2.lanes import m3_tf
    return m3_tf.factory(calib=calib, use_cache=use_cache, cache_dir=cache_dir, **params)


FACTORY_OF_FAMILY = {'deep': 'deep', 'mr': 'minirocket', 'tflog': 'tflog', 'body': 'deep_body'}
BUILD_X_OF_INPUT = {'approach': 'x_approach', 'fixed': 'x_fixed', 'stft': 'x_stft', 'cwt': 'x_cwt',
                    'approach_body': 'x_approach_body', 'fixed_body': 'x_fixed_body'}
