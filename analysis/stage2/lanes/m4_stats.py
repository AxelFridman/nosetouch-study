"""Lane m4 statistics on saved predictions (no model is fitted here, except the two-parameter recalibration
logistic of the calibration slope).

* **Half credit.** A cue-free pipeline abstains with P(right) = 0.5. Until 2026-10-28 ``cv.metrics`` predicted
  'right' when P >= 0.5, so an abstention counted as a right-hand answer: harmless on the balanced primary set, but
  on the all-right natural-use set every abstention became a correct answer. Every accuracy of this lane therefore
  scores P = 0.5 as half a correct answer (:func:`half_credit`; the harness has used the same rule since then), and
  reports coverage, the accuracy on the answered recordings and the forced-choice accuracy (the best candidate
  always scored) beside it.
* **Paired comparisons** (:func:`paired`): the same recordings under two pipelines; per-person mean differences of
  correctness (half credit), log loss and Brier score; person-bootstrap 95 % CIs; exact sign-flip p over persons;
  'identical' when no prediction differs by more than ``IDENTICAL_ATOL``.
* **Multiplicity** (:func:`holm`): Holm-adjusted p-values within a named family of comparisons.
* **Calibration** (cross-lane rule: no percentile bootstrap CI of the ECE, which can exclude its own estimate):
  the ECE with its perfect-calibration null band (:func:`ece_null`: persons resampled, outcomes redrawn as
  Bernoulli(P), so the band is what a perfectly calibrated P of the same sharpness and clustering would show),
  and the calibration slope and intercept of the logistic recalibration y ~ logit(P) with person-bootstrap CIs
  (:func:`calibration_slope`; slope 1 and intercept 0 = calibrated; slope < 1 = over-confident).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv

from .m4_config import MIN_PERSONS_CI, N_BOOT

TIE_ATOL = 1e-12          # |P - 0.5| below this is an abstention / a tie
IDENTICAL_ATOL = 1e-9     # paired predictions closer than this everywhere: identical (solver precision)
_EPS = float(np.finfo(np.float64).eps)


def correct(y: Any, p: Any) -> np.ndarray:
    """Per-row correctness with half credit for a tie (P = 0.5)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    tie = np.abs(p - 0.5) <= TIE_ATOL
    return np.where(tie, 0.5, ((p > 0.5).astype(int) == y).astype(float))


def half_credit(y: Any, p: Any) -> float:
    c = correct(y, p)
    return float(c.mean()) if c.size else math.nan


def person_ci(pred: pd.DataFrame, fn: Callable[[np.ndarray, np.ndarray], float], *, n_boot: int = N_BOOT,
              seed: int = C.SEED, min_persons: int = MIN_PERSONS_CI) -> tuple[float, float]:
    """Person-bootstrap 95 % CI of fn(y, p) over ``pred`` (person_id, y, p_right); (nan, nan) when it holds fewer
    than ``min_persons`` persons (cross-lane rule)."""
    if pred['person_id'].nunique() < min_persons or not len(pred):
        return (math.nan, math.nan)
    ci = cv.person_bootstrap(pred[['person_id', 'y', 'p_right']], fn, n_boot=n_boot, seed=seed).get('stat')
    return (math.nan, math.nan) if ci is None else (float(ci[0]), float(ci[1]))


def scores(pred: pd.DataFrame, *, p_forced: pd.Series | None = None, answered: pd.Series | None = None,
           n_boot: int = N_BOOT) -> dict[str, Any]:
    """Half-credit accuracy (with CI), coverage, accuracy on the answered rows and forced-choice accuracy (with
    CIs) of one set of predictions (person_id, y, p_right); ``answered`` defaults to P != 0.5 and ``p_forced``
    (the best candidate's P even when the pipeline abstains) to p_right."""
    y = pred['y'].to_numpy(int)
    p = pred['p_right'].to_numpy(float)
    ans = (np.abs(p - 0.5) > TIE_ATOL) if answered is None else np.asarray(answered, bool)
    pf = p if p_forced is None else np.asarray(p_forced, float)
    out: dict[str, Any] = {'n': int(len(pred)), 'n_persons': int(pred['person_id'].nunique()),
                           'n_right': int(y.sum()), 'accuracy_half': half_credit(y, p),
                           'accuracy_ge05': float(np.mean((p >= 0.5).astype(int) == y)) if y.size else math.nan,
                           'coverage': float(ans.mean()) if y.size else math.nan, 'n_answered': int(ans.sum()),
                           # answered by the detector but with P exactly 0.5 (a calibrator's flat centre): no hand shown
                           'n_answered_tie': int((ans & (np.abs(p - 0.5) <= TIE_ATOL)).sum()),
                           'accuracy_answered': half_credit(y[ans], p[ans]) if ans.any() else math.nan,
                           'accuracy_forced': half_credit(y, pf)}
    out['accuracy_half_ci'] = list(person_ci(pred, half_credit, n_boot=n_boot))
    sub = pred.loc[ans]
    out['accuracy_answered_ci'] = list(person_ci(sub, half_credit, n_boot=n_boot)) if len(sub) else [math.nan] * 2
    out['accuracy_forced_ci'] = list(person_ci(pred.assign(p_right=pf), half_credit, n_boot=n_boot))
    out['coverage_ci'] = list(person_ci(pred.assign(p_right=ans.astype(float), y=1),
                                        lambda yy, pp: float(np.mean(pp)), n_boot=n_boot))
    return out


# ───────────────────────────── paired comparisons and multiplicity ─────────────────────────────


def sign_flip_p(d: np.ndarray, *, exact_max: int = 20, n_perm: int = 100_000, seed: int = C.SEED) -> float:
    """Two-sided sign-flip p of mean(d) = 0 over persons (exact up to exact_max persons; as cv.paired_compare)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return math.nan
    obs = abs(d.mean())
    if obs <= IDENTICAL_ATOL:
        return 1.0
    if n <= exact_max:
        codes = np.arange(2 ** n, dtype=np.int64)
        total = np.zeros(codes.size)
        for i in range(n):
            total += (1.0 - 2.0 * ((codes >> i) & 1)) * d[i]
        stats = np.abs(total / n)
    else:
        rng = np.random.default_rng(seed)
        stats = np.abs(rng.choice([-1.0, 1.0], size=(n_perm, n)) @ d / n)
    return float(np.mean(stats >= obs - 1e-12))


def paired(a: pd.DataFrame, b: pd.DataFrame, *, key: str = 'trial_key', n_boot: int = N_BOOT,
           seed: int = C.SEED) -> dict[str, Any]:
    """A vs B on the same rows (both: key, person_id, y, p_right): A - B of half-credit correctness (pooled and
    per-person mean), per-person mean log loss and Brier, person-bootstrap CIs, exact sign-flip p over persons."""
    j = a[[key, 'person_id', 'y', 'p_right']].merge(b[[key, 'p_right']], on=key, suffixes=('_a', '_b'))
    y = j['y'].to_numpy(int)
    pa, pb = j['p_right_a'].to_numpy(float), j['p_right_b'].to_numpy(float)
    ca, cb = correct(y, pa), correct(y, pb)

    def ll(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, _EPS, 1 - _EPS)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    df = pd.DataFrame({'person_id': j['person_id'].astype(str), 'd_acc': ca - cb, 'd_ll': ll(pa) - ll(pb),
                       'd_brier': (pa - y) ** 2 - (pb - y) ** 2})
    pp = df.groupby('person_id').mean()
    persons = df['person_id'].to_numpy()
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, 4))
    D = df[['d_acc', 'd_ll', 'd_brier']].to_numpy(float)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([rows[u] for u in pick])
        boot[i, :3] = D[sel].mean(axis=0)
        boot[i, 3] = pp.loc[pick, 'd_acc'].mean()
    q = lambda k: [float(np.quantile(boot[:, k], 0.025)), float(np.quantile(boot[:, k], 0.975))]  # noqa: E731
    identical = bool(np.max(np.abs(pa - pb)) <= IDENTICAL_ATOL) if len(j) else False
    return {'n_trials': int(len(j)), 'n_persons': int(uniq.size), 'identical': identical,
            'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
            'd_acc': float(D[:, 0].mean()), 'd_acc_ci': q(0), 'd_acc_person': float(pp['d_acc'].mean()),
            'd_acc_person_ci': q(3), 'p_acc': sign_flip_p(pp['d_acc'].to_numpy(), seed=seed),
            'd_ll': float(D[:, 1].mean()), 'd_ll_ci': q(1), 'p_ll': sign_flip_p(pp['d_ll'].to_numpy(), seed=seed),
            'd_brier': float(D[:, 2].mean()), 'd_brier_ci': q(2),
            'n_discordant': int(np.sum(np.abs(ca - cb) > 0)), 'n_a_better': int(np.sum(ca > cb)),
            'n_b_better': int(np.sum(cb > ca))}


def paired_shares(persons: Any, a: Any, b: Any, *, n_boot: int = N_BOOT, seed: int = C.SEED) -> dict[str, Any]:
    """A - B of two per-row 0/1 indicators on the same rows (e.g. 'answered wrongly' with and without a check):
    pooled shares and their difference, person-bootstrap 95 % CI of the difference, exact sign-flip p over the
    persons' mean differences."""
    persons = np.asarray(persons).astype(str)
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = xa - xb
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = d[np.concatenate([rows[u] for u in rng.choice(uniq, size=uniq.size, replace=True)])].mean()
    pp = np.array([d[rows[u]].mean() for u in uniq])
    return {'n': int(d.size), 'n_persons': int(uniq.size), 'a': float(xa.mean()), 'b': float(xb.mean()),
            'd': float(d.mean()), 'd_ci': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            'p': sign_flip_p(pp, seed=seed), 'identical': bool(np.max(np.abs(d)) <= IDENTICAL_ATOL) if d.size else True}


def holm(p: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full(p.size, np.nan)
    if not ok.size:
        return out.tolist()
    order = ok[np.argsort(p[ok], kind='stable')]
    m = order.size
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out.tolist()


def verdict(d: dict[str, Any], a: str, b: str, *, p_key: str = 'p_acc_holm', ci_key: str = 'd_acc_ci',
            alpha: float = 0.05, metric: str = 'accuracy') -> str:
    """Generated wording of one paired comparison: identical / A better / B better / no difference shown."""
    if d.get('identical'):
        return f'{a} and {b}: identical predictions'
    p, ci = d.get(p_key), d.get(ci_key)
    if p is None or ci is None or not all(math.isfinite(x) for x in (p, *ci)):
        return 'not computed'
    if p < alpha and (ci[0] > 0 or ci[1] < 0):
        better = a if ci[0] > 0 else b
        return f'{better} has the higher {metric} (Holm p = {p:.3f})'
    return f'no {metric} difference shown (Holm p = {p:.2f})'


# ───────────────────────────── calibration ─────────────────────────────


def ece_fast(y: np.ndarray, p: np.ndarray, n_bins: int = C.ECE_BINS) -> float:
    """``cv.ece`` (equal-mass bins: quantile edges, ``searchsorted(edges[1:-1], p, 'right')``, empty bins dropped)
    without building a DataFrame: the same number (a test checks it), fast enough for bootstrap loops."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if not p.size:
        return math.nan
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    k = np.searchsorted(edges[1:-1], p, side='right')
    n = np.bincount(k, minlength=n_bins).astype(float)
    sy = np.bincount(k, weights=y, minlength=n_bins)
    sp = np.bincount(k, weights=p, minlength=n_bins)
    m = n > 0
    return float(np.sum(n[m] / n.sum() * np.abs(sy[m] / n[m] - sp[m] / n[m])))


def ece_null(pred: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = C.SEED) -> dict[str, float]:
    """The observed ECE (10 equal-mass bins) and its perfect-calibration null distribution: persons resampled with
    replacement, then y* ~ Bernoulli(P) for every resampled row; returns ece, the null 2.5 / 50 / 97.5 %
    quantiles and the one-sided p = P(null >= observed)."""
    y = pred['y'].to_numpy(int)
    p = pred['p_right'].to_numpy(float)
    obs = cv.ece(y, p)
    persons = pred['person_id'].astype(str).to_numpy()
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    for i in range(n_boot):
        sel = np.concatenate([rows[u] for u in rng.choice(uniq, size=uniq.size, replace=True)])
        ps = p[sel]
        null[i] = ece_fast((rng.random(ps.size) < ps).astype(int), ps)
    null = null[np.isfinite(null)]
    return {'ece': float(obs), 'null_lo': float(np.quantile(null, 0.025)), 'null_med': float(np.median(null)),
            'null_hi': float(np.quantile(null, 0.975)), 'p_null': float(np.mean(null >= obs - 1e-12)),
            'inside_null': bool(obs <= np.quantile(null, 0.975))}


def _recal_fit(z: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Unpenalised logistic regression y ~ a + b z by Newton-Raphson: (intercept a, slope b)."""
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ w, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def calibration_slope(pred: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = C.SEED) -> dict[str, Any]:
    """Calibration intercept and slope of y ~ logit(P) with person-bootstrap 95 % CIs (module docstring)."""
    y = pred['y'].to_numpy(float)
    p = np.clip(pred['p_right'].to_numpy(float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    a, b = _recal_fit(z, y)
    persons = pred['person_id'].astype(str).to_numpy()
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    bo = []
    for _ in range(n_boot):
        sel = np.concatenate([rows[u] for u in rng.choice(uniq, size=uniq.size, replace=True)])
        if len(set(y[sel].tolist())) < 2 or np.ptp(z[sel]) <= 0:
            continue
        bo.append(_recal_fit(z[sel], y[sel]))
    bo = np.asarray(bo, dtype=float).reshape(-1, 2)
    ci = (lambda k: [float(np.quantile(bo[:, k], 0.025)), float(np.quantile(bo[:, k], 0.975))]) if len(bo) else \
        (lambda k: [math.nan, math.nan])  # noqa: E731
    out = {'intercept': a, 'slope': b, 'intercept_ci': ci(0), 'slope_ci': ci(1)}
    out['slope_ok'] = bool(out['slope_ci'][0] <= 1.0 <= out['slope_ci'][1])
    out['intercept_ok'] = bool(out['intercept_ci'][0] <= 0.0 <= out['intercept_ci'][1])
    return out


def calibration_summary(pred: pd.DataFrame, *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Metrics, ECE with its null band, calibration slope / intercept and whether the predictions are consistent
    with calibration (ECE inside the null band and slope CI containing 1)."""
    m = cv.metrics(pred['y'], pred['p_right'])
    out = {'n': int(len(pred)), 'n_persons': int(pred['person_id'].nunique()), 'accuracy_half':
           half_credit(pred['y'], pred['p_right']), **{k: m[k] for k in ('roc_auc', 'log_loss', 'brier')}}
    out.update({f'ece_{k}' if k != 'ece' else 'ece': v for k, v in ece_null(pred, n_boot=n_boot).items()})
    out.update(calibration_slope(pred, n_boot=n_boot))
    out['calibrated'] = bool(out['ece_inside_null'] and out['slope_ok'])
    return out


# ───────────────────────────── natural use: is the rejection rate unusual? ─────────────────────────────


def poisson_binomial_sf(probs: Sequence[float], k: int) -> float:
    """P(X >= k) for X a sum of independent Bernoulli(probs) (exact, dynamic programming)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for q in probs:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return float(dist[k:].sum())
