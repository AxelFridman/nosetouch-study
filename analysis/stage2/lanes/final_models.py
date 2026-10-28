"""Lane final: the models that are new in this lane (task 2b) and the reference as a candidate.

* :func:`make_reference` - the README worked example (``stage2.scripts.reference.yaw_logreg``) on the shared input.
* :func:`make_lr_body_tc` - lane m1's small antisymmetric logistic model given the body-frame channels as time
  courses: the means of yaw rate, cumulative yaw, roll rate and lateral acceleration over 8 equal bins of the
  time-normalised approach (32 odd features), each scaled by its RMS over the training rows (never centred), no
  intercept, L2 with C chosen on ``fold.inner_splits()`` by the pooled inner log loss. Antisymmetric by construction:
  P(mirror image) = 1 - P exactly. The production candidate that is cheapest to port (a dot product).
* :func:`make_m2_lr` - lane m2's pre-specified pipeline (z-score, mirror augmentation, L2 logistic regression, C on
  3 inner person folds) on lane m2's feature families computed on a channel table with the body frame
  (``final_features``).

Every fit is deterministic (lbfgs), uses the training rows only and never sees a test row's label.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

#: L2 grid of the body-frame logistic: lane m1's half-decade grid (ties go to the more regularised value)
CS_BODY: tuple[float, ...] = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
#: the odd (hand-carrying), grip-invariant channels of the body-frame logistic, in ``m3_body.EXT_CHANNELS`` names
TC_CHANNELS: tuple[str, ...] = ('yaw_rate', 'yaw_cum', 'roll_rate', 'acc_lat')
TC_BINS = 8
EPS = 1e-15


def make_reference() -> Any:
    from stage2.scripts.reference import yaw_logreg
    from stage2.standardize import CHANNELS
    return yaw_logreg(tuple(CHANNELS))


def bin_edges(T: int, n_bins: int = TC_BINS) -> np.ndarray:
    return np.linspace(0, T, n_bins + 1).round().astype(int)


def tc_features(x_ext: np.ndarray, channels: Sequence[str] = TC_CHANNELS, n_bins: int = TC_BINS) -> np.ndarray:
    """(N, 21, T) approach with the body channels (``m3_body.x_approach_body(d)['x']``) -> (N, len(channels) *
    n_bins): the mean of each channel over equal time bins (channel-major order). Per trial; every column is odd."""
    from stage2.lanes.m3_body import EXT_CHANNELS
    X = np.asarray(x_ext, dtype=np.float64)
    e = bin_edges(X.shape[-1], n_bins)
    cols = []
    for c in channels:
        xi = X[:, EXT_CHANNELS.index(c), :]
        cols.extend(xi[:, e[b]:e[b + 1]].mean(-1) for b in range(n_bins))
    return np.column_stack(cols)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_antisym_logistic(F: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    """(scale, weights) of the antisymmetric logistic model: z = ((F / scale) @ weights), no intercept."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    s = np.sqrt(np.mean(F ** 2, axis=0))
    s = np.where(s > 0, s, 1.0)
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=10000, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        m.fit(F / s, y)
    return s, m.coef_[0].astype(np.float64)


def make_lr_body_tc(Cs: Sequence[float] = CS_BODY, channels: Sequence[str] = TC_CHANNELS, n_bins: int = TC_BINS,
                    store: dict | None = None) -> Any:
    """fit_predict of the body-frame time-course logistic (input ``x_approach_body``: {'x': (N, 21, 64)})."""
    Cs = tuple(float(c) for c in Cs)
    channels = tuple(channels)

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Ftr = tc_features(fold.X_train['x'], channels, n_bins)
        Fte = tc_features(fold.X_test['x'], channels, n_bins)
        y = np.asarray(fold.y_train).astype(int)
        scores = {}
        for C in Cs:
            z = np.full(y.size, np.nan)
            for tr, va in fold.inner_splits():
                s, w = fit_antisym_logistic(Ftr[tr], y[tr], C)
                z[va] = (Ftr[va] / s) @ w
            scores[C] = _log_loss(y, 1.0 / (1.0 + np.exp(-z)))
        best = min(Cs, key=lambda c: (round(scores[c], 12), Cs.index(c)))
        s, w = fit_antisym_logistic(Ftr, y, best)
        z_te = (Fte / s) @ w
        if store is not None:
            store[fold.fold] = {'C': best, 'scale': s.tolist(), 'weights': w.tolist()}
        return 1.0 / (1.0 + np.exp(-z_te)), {'C': best, 'inner_logloss': round(scores[best], 6)}

    return fit_predict


def make_m2_lr(channels: str = 'body12', n_splits: int = 3) -> Any:
    """Lane m2's default logistic pipeline on the features of channel table ``channels`` (``final_features``)."""
    from stage2.lanes import m2_models as M
    from stage2.lanes.final_features import mirror_map
    mi, ms = mirror_map(channels)
    return M.make_fit_predict('lr_l2', mirror='aug', scaler='z', winsor=None, calibrator='none', n_trials=None,
                              mirror_index=mi, mirror_sign=ms, n_splits=n_splits)


def sigmoid(z: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p: Any, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


__all__ = ['make_reference', 'make_lr_body_tc', 'make_m2_lr', 'tc_features', 'fit_antisym_logistic', 'sigmoid',
           'logit', 'TC_CHANNELS', 'TC_BINS', 'CS_BODY', 'math']
