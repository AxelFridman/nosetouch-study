"""Lane m2: the statistics of the derived analyses (no model is fitted here).

* ``holm``                  Holm step-down adjustment of a family of p-values (family-wise error rate).
* ``sign_flip_p``           exact two-sided sign-flip test of a mean per-person difference (as ``cv._sign_flip_p``,
                            computed meet-in-the-middle so 20+ persons stay exact and fast).
* ``paired_losses``         a paired comparison of two per-trial losses (log loss, Brier, ...) on the same trials:
                            mean per-person difference, person-bootstrap 95 % CI and sign-flip p.
* ``ece_null``              the ECE a PERFECTLY calibrated model with the same predictions would show (consistency
                            resampling: y* ~ Bernoulli(p)); ECE is positive even then, so an observed ECE is judged
                            against this band, not against 0 (and never with a percentile bootstrap CI, which can
                            exclude its own estimate because resampling inflates the ECE).
* ``recalibration``         calibration slope b and intercept a of the logistic recalibration y ~ sigmoid(a + b
                            logit p), and the calibration-in-the-large c (intercept with the slope fixed at 1):
                            perfect calibration is b = 1, c = 0; b > 1 means under-confident predictions.
* ``recalibration_ci``      their person-cluster bootstrap 95 % CIs.
* ``natural_use``           accuracy on a one-class free-choice set without counting an abstention (P = 0.5 exactly,
                            e.g. a constant predictor) as a right-hand answer: coverage, answered accuracy,
                            forced-choice accuracy (P >= 0.5 is right: the harness's rule until 2026-10-28) and
                            half-credit accuracy (the harness's rule since then).
* ``equivalent``            practical equivalence of two pipelines: no trial changes class and no P(right) moves by
                            ``EQUIV_MAX_DP`` (solver tolerance), so a difference between them is not a result.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from stage2 import config as C

#: two pipelines whose P(right) differ by less than this on every trial (and change no class) are identical
EQUIV_MAX_DP = 1e-3
#: significance level of the multiplicity-adjusted tests
ALPHA = 0.05
#: below this many persons a stratum gets no bootstrap CI (a 2-person bootstrap has 3 distinct resamples): the
#: per-person values are shown instead
MIN_PERSONS_CI = 5
#: a prediction this close to 0.5 is an abstention (no hand is claimed)
ABSTAIN_TOL = 1e-9
LOGIT_CLIP = 1e-6


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values (NaN entries stay NaN and do not count towards the family size)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind='mergesort')]
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out[order] = adj
    return out


def format_p(p: Any, floor: float = 1e-3) -> str:
    """'0.042', '<0.001' or 'n/a'."""
    try:
        f = float(p)
    except (TypeError, ValueError):
        return 'n/a'
    if not math.isfinite(f):
        return 'n/a'
    return f'<{floor:g}' if f < floor else f'{f:.3f}'


def sign_flip_p(d: np.ndarray, *, exact_max: int = 20, n_perm: int = 100_000, seed: int = C.SEED) -> float:
    """Two-sided sign-flip permutation p-value of mean(d) = 0 (exact enumeration of the 2^n sign patterns, the
    definition of ``cv._sign_flip_p``), computed meet-in-the-middle: all sums of one half of the persons plus all
    sums of the other. Above ``exact_max`` persons (the core's own limit): the core's Monte Carlo version, so both
    always agree."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return math.nan
    if n > exact_max:
        from stage2 import cv
        return cv._sign_flip_p(d, seed=seed, n_perm=n_perm)
    obs = abs(d.mean())

    def all_sums(v: np.ndarray) -> np.ndarray:
        s = np.zeros(1)
        for x in v:
            s = np.concatenate([s + x, s - x])
        return s
    h = n // 2
    tot = all_sums(d[:h])[:, None] + all_sums(d[h:])[None, :]
    return float(np.mean(np.abs(tot / n) >= obs - 1e-12))


def paired_losses(loss_a: np.ndarray, loss_b: np.ndarray, persons: np.ndarray, *, n_boot: int = C.N_BOOT,
                  seed: int = C.SEED) -> dict[str, float]:
    """Paired comparison of two per-trial losses on the same trials: the mean over persons of the per-person mean
    difference (a - b), its person-bootstrap 95 % CI and the exact sign-flip p over persons."""
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    persons = np.asarray(persons).astype(str)
    uniq, inv = np.unique(persons, return_inverse=True)
    n_p = np.bincount(inv, minlength=uniq.size).astype(float)
    d = (np.bincount(inv, a, uniq.size) - np.bincount(inv, b, uniq.size)) / n_p
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, uniq.size, (n_boot, uniq.size))
    boots = d[idx].mean(1)
    return {'diff': float(d.mean()), 'diff_lo': float(np.quantile(boots, 0.025)),
            'diff_hi': float(np.quantile(boots, 0.975)), 'p': sign_flip_p(d, seed=seed), 'n_persons': int(uniq.size),
            'persons_lower': int((d < 0).sum()), 'persons_higher': int((d > 0).sum())}


def log_loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-trial log loss with the clipping of ``cv.metrics`` (sklearn's float64 epsilon)."""
    eps = float(np.finfo(np.float64).eps)
    pc = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> np.ndarray:
    """ECE of every row of y (R, N) against one p (N,) (equal-mass bins of p, the definition of cv.ece)."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    k = np.searchsorted(edges[1:-1], p, side='right')
    y = np.atleast_2d(y)
    tot = np.zeros(y.shape[0])
    for b in np.unique(k):
        m = k == b
        tot += np.abs(y[:, m].sum(1) - p[m].sum())
    return tot / p.size


def ece_null(y: np.ndarray, p: np.ndarray, *, n_sim: int = C.N_BOOT, n_bins: int = C.ECE_BINS,
             seed: int = C.SEED) -> dict[str, float]:
    """Observed ECE and its distribution under perfect calibration (y* ~ Bernoulli(p), trials independent): mean,
    2.5th and 97.5th percentiles and the share of simulations at least as large as the observed ECE (consistency
    p). Person clustering is ignored, so the band is a lower bound on the noise."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    obs = float(_ece(y, p, n_bins)[0])
    rng = np.random.default_rng(seed)
    sims = _ece((rng.random((n_sim, p.size)) < p).astype(float), p, n_bins)
    return {'ece': obs, 'ece_null_mean': float(sims.mean()), 'ece_null_lo': float(np.quantile(sims, 0.025)),
            'ece_null_hi': float(np.quantile(sims, 0.975)), 'ece_null_p': float(np.mean(sims >= obs - 1e-12))}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def _newton(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, slope: bool = True,
            n_iter: int = 60) -> tuple[float, float]:
    """Weighted ML logistic regression of y on [1, z] (slope=True) or with offset z (slope fixed at 1):
    (intercept, slope). ``w``: integer weights (bootstrap multiplicities)."""
    w = np.ones_like(z) if w is None else w
    a, b = 0.0, 1.0
    for _ in range(n_iter):
        eta = np.clip(a + b * z, -35.0, 35.0)          # sigmoid is saturated beyond; avoids exp overflow
        q = 1.0 / (1.0 + np.exp(-eta))
        r = w * (y - q)
        v = w * q * (1 - q) + 1e-12
        if slope:
            g = np.array([r.sum(), (r * z).sum()])
            H = np.array([[v.sum(), (v * z).sum()], [(v * z).sum(), (v * z * z).sum()]])
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            a, b = a + step[0], b + step[1]
            if np.max(np.abs(step)) < 1e-10:
                break
        else:
            step = r.sum() / v.sum()
            a += step
            if abs(step) < 1e-10:
                break
    return float(a), float(b)


def recalibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration slope and intercept (y ~ a + b logit p) and calibration-in-the-large (offset logit p)."""
    y = np.asarray(y, dtype=float)
    z = _logit(p)
    a, b = _newton(z, y)
    c, _ = _newton(z, y, slope=False)
    return {'cal_slope': b, 'cal_intercept': a, 'cal_in_the_large': c}


def recalibration_ci(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, n_boot: int = 1000,
                     seed: int = C.SEED) -> dict[str, tuple[float, float]]:
    """Person-cluster bootstrap 95 % CIs (percentile) of the slope and the calibration-in-the-large."""
    y = np.asarray(y, dtype=float)
    z = _logit(p)
    persons = np.asarray(persons).astype(str)
    uniq, inv = np.unique(persons, return_inverse=True)
    rng = np.random.default_rng(seed)
    slopes, citl = [], []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)
        w = cnt[inv].astype(float)
        if y[w > 0].min() == y[w > 0].max():
            continue
        _, b = _newton(z, y, w)
        c, _ = _newton(z, y, w, slope=False)
        if math.isfinite(b) and math.isfinite(c):
            slopes.append(b)
            citl.append(c)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (math.nan, math.nan)  # noqa: E731
    return {'cal_slope': q(slopes), 'cal_in_the_large': q(citl)}


def natural_use(p: np.ndarray, y: np.ndarray, persons: np.ndarray, *, n_boot: int = 1000,
                seed: int = C.SEED) -> dict[str, float]:
    """Free-choice accuracy without rewarding abstentions: coverage (share with |P - 0.5| > ABSTAIN_TOL), answered
    accuracy (on the answered trials), forced-choice accuracy (P >= 0.5 = right: the harness's rule until 2026-10-28,
    which counts an abstention as a right-hand answer) and half-credit accuracy (an abstention scores 1/2: the
    harness's rule since then), with a person-bootstrap
    95 % CI of the half-credit accuracy (only with >= MIN_PERSONS_CI persons)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    if p.size == 0:
        return {'n': 0}
    answered = np.abs(p - 0.5) > ABSTAIN_TOL
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    half = np.where(answered, correct, 0.5)
    out = {'n': int(p.size), 'n_right': int(y.sum()), 'n_left': int((1 - y).sum()), 'n_abstain': int((~answered).sum()),
           'coverage': float(answered.mean()), 'forced_choice_accuracy': float(correct.mean()),
           'answered_accuracy': float(correct[answered].mean()) if answered.any() else math.nan,
           'half_credit_accuracy': float(half.mean()), 'n_correct_answered': int(correct[answered].sum())}
    persons = np.asarray(persons).astype(str)
    uniq, inv = np.unique(persons, return_inverse=True)
    out['n_persons'] = int(uniq.size)
    if uniq.size >= MIN_PERSONS_CI:
        s = np.bincount(inv, half, uniq.size)
        c = np.bincount(inv, minlength=uniq.size).astype(float)
        idx = np.random.default_rng(seed).integers(0, uniq.size, (n_boot, uniq.size))
        boots = s[idx].sum(1) / c[idx].sum(1)
        out['half_credit_lo'], out['half_credit_hi'] = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return out


def equivalent(p_a: np.ndarray, p_b: np.ndarray, max_dp: float = EQUIV_MAX_DP) -> dict[str, Any]:
    """Largest |P(right) difference|, classes changed, and whether the two are practically identical."""
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    dp = float(np.max(np.abs(p_a - p_b))) if p_a.size else math.nan
    changed = int(np.sum((p_a >= 0.5) != (p_b >= 0.5)))
    return {'max_abs_dp': dp, 'n_class_changes': changed, 'identical': bool(changed == 0 and dp < max_dp)}
