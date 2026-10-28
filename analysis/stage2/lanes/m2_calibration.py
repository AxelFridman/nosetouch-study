"""Lane m2: post-hoc calibrators of P(right), fitted on cross-fitted predictions of the TRAINING persons only, and
the temperature of multi-attempt aggregates.

Each calibrator is fitted on (p_oof, y) - the inner cross-fitted P(right) of the training rows of an outer fold
(``fold.inner_splits()``: GroupKFold over the training persons) - and applied to the held-out person's P(right).

* ``platt``           logistic regression on logit(p): sigmoid(a * logit(p) + b)
* ``temperature``     sigmoid(logit(p) / T): one parameter, no bias, so p(mirror(x)) = 1 - p(x) survives
* ``isotonic``        monotone step function (clipped to [EPS_ISO, 1 - EPS_ISO] so the log loss stays finite)
* ``beta``            Kull et al. (2017): sigmoid(a ln p - b ln(1 - p) + c) with a, b >= 0
* ``platt_platform``  one Platt map per platform (the platform is known to production from the browser), falling
                      back to the global map for a platform with too few training rows or one class: the direct
                      test of whether one calibration map serves both platforms

``fit_calibrator(name, p, y, strata)`` returns a callable ``apply(p, strata=None)`` plus its parameters (stored
per fold); ``strata`` (the platform of each row) is used only by ``platt_platform``.

**Multi-attempt temperature.** k attempts of one person and hand are aggregated as sigmoid(sum logit / tau_k):
tau_k = 1 treats the attempts as independent evidence, tau_k = k is the mean logit. ``attempt_temperatures`` fits
tau_k on random k-subsets of the (person, hand) cells of cross-fitted predictions; inside a fit_predict these are
the training persons' inner cross-fitted predictions, so the tau_k applied to the held-out person is nested.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

CALIBRATORS: tuple[str, ...] = ('none', 'platt', 'temperature', 'isotonic', 'beta', 'platt_platform')
STRATIFIED: frozenset[str] = frozenset({'platt_platform'})
CLIP = 1e-6
EPS_ISO = 1e-3
#: a platform-specific map needs at least this many training rows of the platform, with both hands
MIN_STRATUM_ROWS = 40
#: attempts per aggregate (k) and random k-subsets per (person, hand) cell for the multi-attempt temperature
ATTEMPT_KS: tuple[int, ...] = (1, 2, 3, 4, 5)
ATTEMPT_DRAWS = 200


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _lr_fit(Z: np.ndarray, y: np.ndarray, fit_intercept: bool = True, C: float = 1e6) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, fit_intercept=fit_intercept, max_iter=1000).fit(Z, y)
    return m.coef_.ravel().copy(), float(m.intercept_[0]) if fit_intercept else 0.0


def _platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    w, b = _lr_fit(logit(p)[:, None], y)
    return float(w[0]), float(b)


def fit_calibrator(name: str, p: np.ndarray, y: np.ndarray, strata: Sequence[Any] | None = None
                   ) -> tuple[Callable[..., np.ndarray], dict[str, Any]]:
    """Fit calibrator ``name`` on cross-fitted (p, y) of the training rows; return (apply(q, strata=None), params).
    ``strata`` (one label per row, e.g. the platform) is needed by the stratified calibrators only."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if name == 'none':
        return (lambda q, strata=None: np.asarray(q, dtype=np.float64)), {}
    if not np.isfinite(p).all():
        raise ValueError(f'calibrator {name!r}: the cross-fitted predictions are not finite (a single pre-specified '
                         'configuration has none)')
    if name == 'platt':
        a, b = _platt(p, y)
        return (lambda q, strata=None: sigmoid(a * logit(q) + b)), {'a': a, 'b': b}
    if name == 'platt_platform':
        if strata is None:
            raise ValueError('platt_platform needs the platform of every training row')
        s = np.asarray(strata).astype(str)
        a0, b0 = _platt(p, y)
        maps: dict[str, tuple[float, float]] = {}
        for v in sorted(set(s.tolist())):
            m = s == v
            if m.sum() >= MIN_STRATUM_ROWS and 0 < y[m].sum() < m.sum():
                maps[v] = _platt(p[m], y[m])

        def apply(q: np.ndarray, strata: Sequence[Any] | None = None) -> np.ndarray:
            q = np.asarray(q, dtype=np.float64)
            st = np.full(q.size, '', dtype=object) if strata is None else np.asarray(strata).astype(str)
            out = sigmoid(a0 * logit(q) + b0)
            for v, (a, b) in maps.items():
                m = st == v
                out[m] = sigmoid(a * logit(q[m]) + b)
            return out
        params: dict[str, Any] = {'a': a0, 'b': b0}
        for v, (a, b) in maps.items():
            params[f'a_{v}'], params[f'b_{v}'] = a, b
        return apply, params
    if name == 'temperature':
        z = logit(p)
        r = minimize_scalar(lambda lt: _logloss(y, sigmoid(z / np.exp(lt))), bounds=(-3.0, 3.0), method='bounded',
                            options={'xatol': 1e-6})
        T = float(np.exp(r.x))
        return (lambda q, strata=None: sigmoid(logit(q) / T)), {'T': T}
    if name == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(y_min=EPS_ISO, y_max=1 - EPS_ISO, out_of_bounds='clip').fit(p, y)
        return (lambda q, strata=None: np.clip(iso.predict(np.asarray(q, dtype=np.float64)), EPS_ISO, 1 - EPS_ISO)), \
            {'n_steps': int(len(iso.X_thresholds_))}
    if name == 'beta':
        pc = np.clip(p, CLIP, 1 - CLIP)
        Z = np.column_stack([np.log(pc), -np.log1p(-pc)])
        w, c = _lr_fit(Z, y)
        a, b = float(w[0]), float(w[1])
        if a < 0 or b < 0:            # Kull et al.: refit with the offending input dropped
            keep = 0 if a >= b else 1
            w1, c = _lr_fit(Z[:, [keep]], y)
            a, b = (float(w1[0]), 0.0) if keep == 0 else (0.0, float(w1[0]))
            a, b = max(a, 0.0), max(b, 0.0)

        def apply_beta(q: np.ndarray, strata: Sequence[Any] | None = None) -> np.ndarray:
            qc = np.clip(np.asarray(q, dtype=np.float64), CLIP, 1 - CLIP)
            return sigmoid(a * np.log(qc) - b * np.log1p(-qc) + c)
        return apply_beta, {'a': a, 'b': b, 'c': c}
    raise ValueError(f'unknown calibrator {name!r} ({CALIBRATORS})')


# ───────────────────────────── multi-attempt aggregates ─────────────────────────────


def attempt_subsets(n: int, k: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """(m, k) index array: every k-subset of range(n) when there are at most ``n_draws`` of them, else ``n_draws``
    random k-subsets without replacement (vectorised: the k smallest of n uniform keys per draw)."""
    if math.comb(n, k) <= n_draws:
        import itertools
        return np.array(list(itertools.combinations(range(n), k)), dtype=int).reshape(-1, k)
    return np.argsort(rng.random((n_draws, n)), axis=1)[:, :k]


def attempt_draws(p: np.ndarray, y: np.ndarray, persons: np.ndarray, k: int, *, n_draws: int = ATTEMPT_DRAWS,
                  seed: int = 0, eligible: set[str] | None = None) -> dict[str, np.ndarray]:
    """Random k-subsets of every (person, hand) cell with >= k rows (only ``eligible`` persons when given):
    per draw the person, the hand, the mean P(right) and the sum of the logits of its k attempts."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    persons = np.asarray(persons).astype(str)
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {'person': [], 'y': [], 'mean_p': [], 'sum_logit': []}
    for u in sorted(set(persons.tolist())):
        if eligible is not None and u not in eligible:
            continue
        for hand in (0, 1):
            idx = np.flatnonzero((persons == u) & (y == hand))
            if idx.size < k:
                continue
            sub = idx[attempt_subsets(idx.size, k, n_draws, rng)]
            out['person'].append(np.full(sub.shape[0], u, dtype=object))
            out['y'].append(np.full(sub.shape[0], hand, dtype=int))
            out['mean_p'].append(p[sub].mean(1))
            out['sum_logit'].append(logit(p[sub]).sum(1))
    if not out['y']:
        return {k_: np.zeros(0) for k_ in out}
    return {k_: np.concatenate(v) for k_, v in out.items()}


def fit_tau(z: np.ndarray, y: np.ndarray) -> float:
    """tau minimising the log loss of sigmoid(z / tau) (z: sums of logits)."""
    yy = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if z.size == 0 or yy.min() == yy.max():
        return math.nan

    def nll(lt: float) -> float:
        return _logloss(yy, sigmoid(z / np.exp(lt)))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 4), method='bounded', options={'xatol': 1e-6}).x))


def attempt_temperatures(p: np.ndarray, y: np.ndarray, persons: np.ndarray, *, ks: Sequence[int] = ATTEMPT_KS,
                         n_draws: int = ATTEMPT_DRAWS, seed: int = 0) -> dict[str, float]:
    """{str(k): tau_k} fitted on k-subsets of cross-fitted predictions (NaN where they are not finite)."""
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0 or not np.isfinite(p).all():
        return {}
    out = {}
    for k in ks:
        dr = attempt_draws(p, y, persons, k, n_draws=n_draws, seed=seed + k)
        out[str(k)] = fit_tau(dr['sum_logit'], dr['y'])
    return out
