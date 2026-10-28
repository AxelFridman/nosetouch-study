"""Lane m3: the statistics of the derived analyses (no model is fitted here; pure functions of saved predictions).

* ``holm``              Holm step-down adjustment of a family of p-values (family-wise error rate). Every significance
                        claim of lane m3 names its family (``m3_derive.FAMILIES``) and uses the adjusted p.
* ``verdict``           one rule for every paired comparison: 'better' / 'worse' only when the Holm-adjusted sign-flip p
                        is below ``ALPHA`` AND the paired person-bootstrap CI of the same estimand (the mean per-person
                        difference) excludes 0; 'identical' when the two pipelines give the same P(right) to solver
                        precision (``identical``); otherwise 'no detectable difference'.
* ``ece_null``          the ECE a PERFECTLY calibrated model with the same predictions shows (consistency resampling,
                        y* ~ Bernoulli(p)): ECE is positive even then, so an observed ECE is judged against this band and
                        never given a percentile bootstrap CI (that one is biased upward and can exclude its own estimate).
* ``recalibration``     calibration slope b and intercept a of y ~ sigmoid(a + b logit p) (perfect: b = 1, a = 0; b < 1
                        over-confident, b > 1 under-confident), with person-cluster bootstrap CIs (``recalibration_ci``).
* ``binom_ci``          exact (Clopper-Pearson) 95 % CI of a proportion: the natural-use set has one trial per person, so
                        trials are independent across persons and the binomial interval is the honest one.
* ``natural_use``       coverage, answered accuracy, forced-choice accuracy and half-credit accuracy of the natural-use
                        set: a P(right) of exactly 0.5 is an abstention, never a right-hand answer.
* ``MIN_PERSONS_CI``    strata with fewer persons get no bootstrap CI.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from stage2 import config as C

#: significance level of the multiplicity-adjusted tests
ALPHA = 0.05
#: below this many persons a stratum gets no bootstrap CI (a resample of 2-4 persons has too few distinct values)
MIN_PERSONS_CI = 5
#: two pipelines whose P(right) differ by at most this on every shared trial are 'identical' (solver precision)
IDENTICAL_MAX_DP = 1e-9
#: docs/09 M4 acceptance (Rec.): development out-of-fold ECE at most this, and no reliability bin off by more than its CI
ECE_TARGET = 0.05
LOGIT_CLIP = 1e-6


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values of one family (NaN entries stay NaN and do not count towards the family size)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind='mergesort')]
    out[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return out


def identical(p_a: np.ndarray, p_b: np.ndarray, tol: float = IDENTICAL_MAX_DP) -> bool:
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    return bool(p_a.size and np.max(np.abs(p_a - p_b)) <= tol)


def verdict(diff: float, lo: float, hi: float, p_adj: float, *, same: bool = False, alpha: float = ALPHA,
            better: str = 'better', worse: str = 'worse', none: str = 'no detectable difference') -> str:
    """'identical', ``better``, ``worse`` or ``none`` (see the module docstring). ``diff`` is the estimate whose CI is
    [lo, hi]; a positive difference is 'better' for accuracy (pass lo/hi of minus the log-loss difference for log
    loss)."""
    if same:
        return 'identical'
    if not (math.isfinite(p_adj) and p_adj < alpha):
        return none
    if lo > 0 and diff > 0:
        return better
    if hi < 0 and diff < 0:
        return worse
    return none


def format_p(p: Any) -> str:
    """'p < 0.001' or 'p = 0.012' ('n/a' when missing)."""
    try:
        f = float(p)
    except (TypeError, ValueError):
        return 'n/a'
    if not math.isfinite(f):
        return 'n/a'
    return 'p < 0.001' if f < 0.001 else f'p = {f:.3f}'


def sign_flip_p(d: np.ndarray, *, exact_max: int = 20, n_perm: int = 200_000, seed: int = C.SEED) -> float:
    """Two-sided sign-flip p-value of mean(d) = 0 over persons (exact for at most ``exact_max`` persons; the same test
    as ``cv.paired_compare``)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return math.nan
    obs = abs(d.mean())
    if n <= exact_max:
        codes = np.arange(2 ** n, dtype=np.int64)
        tot = np.zeros(codes.size)
        for i in range(n):
            tot += (1.0 - 2.0 * ((codes >> i) & 1)) * d[i]
        stats = np.abs(tot / n)
    else:
        signs = np.random.default_rng([seed, 19]).choice([-1.0, 1.0], size=(n_perm, n))
        stats = np.abs(signs @ d / n)
    return float(np.mean(stats >= obs - 1e-12))


def person_mean_diff(d: np.ndarray, *, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict[str, float]:
    """Mean of per-person differences ``d`` with its person-bootstrap 95 % CI and the exact sign-flip p."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {'diff': math.nan, 'lo': math.nan, 'hi': math.nan, 'p': math.nan, 'n_persons': 0}
    W = np.random.default_rng([seed, 23]).multinomial(d.size, np.full(d.size, 1.0 / d.size), size=n_boot)
    b = (W @ d) / d.size
    return {'diff': float(d.mean()), 'lo': float(np.quantile(b, 0.025)), 'hi': float(np.quantile(b, 0.975)),
            'p': sign_flip_p(d, seed=seed), 'n_persons': int(d.size)}


def per_person_brier(persons: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Mean Brier score of every person."""
    s = (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2
    persons = np.asarray(persons).astype(str)
    return {u: float(s[persons == u].mean()) for u in np.unique(persons)}


# ───────────────────────────── calibration ─────────────────────────────


def _ece_rows(Y: np.ndarray, p: np.ndarray, n_bins: int) -> np.ndarray:
    """ECE (cv.ece: equal-mass bins of p, ties share a bin) of every row of Y (R, N) against one p (N,)."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    k = np.searchsorted(edges[1:-1], p, side='right')
    Y = np.atleast_2d(Y)
    tot = np.zeros(Y.shape[0])
    for b in np.unique(k):
        m = k == b
        tot += np.abs(Y[:, m].sum(axis=1) - p[m].sum())
    return tot / p.size


def ece_null(y: np.ndarray, p: np.ndarray, *, n_sim: int = C.N_BOOT, n_bins: int = C.ECE_BINS,
             seed: int = C.SEED) -> dict[str, float]:
    """Observed ECE and its distribution if the model were perfectly calibrated (y* ~ Bernoulli(p), trials
    independent): mean, 95th percentile and the consistency p (share of simulations at least as large as the observed
    ECE). Person clustering is ignored, so the band is a lower bound on the noise and the test is anti-conservative
    (a 'beyond the band' reading is the weaker evidence of miscalibration, an 'inside' reading the stronger)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    obs = float(_ece_rows(y, p, n_bins)[0])
    rng = np.random.default_rng([seed, 13])
    sims = _ece_rows((rng.random((n_sim, p.size)) < p).astype(float), p, n_bins)
    return {'ece': obs, 'ece_null_mean': float(sims.mean()), 'ece_null_q95': float(np.quantile(sims, 0.95)),
            'ece_null_p': float(np.mean(sims >= obs - 1e-12))}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def _loglik(a: float, b: float, z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    s = a + b * z
    return float(np.sum(w * (y * s - np.logaddexp(0.0, s))))


def _newton(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, n_iter: int = 100) -> tuple[float, float]:
    """Weighted maximum-likelihood logistic regression of y on [1, z]: (intercept, slope). Damped Newton: it starts at
    the intercept-only model (slope 0) and halves every step that would lower the (concave) log-likelihood, so it
    converges where plain Newton from slope 1 overshoots (large logits of an over-confident model)."""
    w = np.ones_like(z, dtype=float) if w is None else np.asarray(w, dtype=float)
    ybar = float(np.clip(np.sum(w * y) / max(np.sum(w), 1e-12), 1e-6, 1 - 1e-6))
    a, b = math.log(ybar / (1 - ybar)), 0.0
    cur = _loglik(a, b, z, y, w)
    for _ in range(n_iter):
        q = 1.0 / (1.0 + np.exp(-np.clip(a + b * z, -40, 40)))
        r = w * (y - q)
        v = w * q * (1 - q) + 1e-12
        g = np.array([r.sum(), (r * z).sum()])
        H = np.array([[v.sum(), (v * z).sum()], [(v * z).sum(), (v * z * z).sum()]])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _h in range(40):                                  # step halving: never lower the log-likelihood
            new = _loglik(a + t * step[0], b + t * step[1], z, y, w)
            if new >= cur - 1e-12:
                break
            t *= 0.5
        a, b, cur = a + t * step[0], b + t * step[1], max(cur, new)
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return float(a), float(b)


def recalibration(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Calibration slope and intercept of y ~ sigmoid(a + b logit p)."""
    a, b = _newton(_logit(p), np.asarray(y, dtype=float))
    return {'cal_slope': b, 'cal_intercept': a}


def recalibration_ci(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, n_boot: int = C.N_BOOT,
                     seed: int = C.SEED) -> dict[str, tuple[float, float]]:
    """Person-cluster bootstrap 95 % CIs (percentile) of the calibration slope and intercept."""
    y = np.asarray(y, dtype=float)
    z = _logit(p)
    uniq, inv = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    rng = np.random.default_rng([seed, 17])
    sl, ic = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)[inv].astype(float)
        yy = y[w > 0]
        if yy.min() == yy.max():
            continue
        a, b = _newton(z, y, w)
        sl.append(b)
        ic.append(a)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (math.nan, math.nan)  # noqa: E731
    return {'cal_slope': q(sl), 'cal_intercept': q(ic)}


# ───────────────────────────── natural use ─────────────────────────────


def binom_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) CI of k successes out of n."""
    from scipy.stats import beta
    if n <= 0:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def natural_use(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Natural-use summary (one free-choice trial per person). A P(right) of exactly 0.5 is an abstention:
    coverage = share answered; answered accuracy (exact CI) over the answered trials; forced-choice accuracy counts an
    abstention as wrong; half credit counts it as 1/2. Every natural-use trial so far was made with the right hand, so
    these are recalls of the right hand."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = int(y.size)
    if n == 0:
        return {'n': 0}
    ans = p != 0.5
    correct = ((p > 0.5) & (y == 1)) | ((p < 0.5) & (y == 0))
    k_ans, n_ans = int(correct[ans].sum()), int(ans.sum())
    lo, hi = binom_ci(int(correct.sum()), n)
    alo, ahi = binom_ci(k_ans, n_ans) if n_ans else (math.nan, math.nan)
    return {'n': n, 'n_right': int(y.sum()), 'n_answered': n_ans, 'coverage': n_ans / n,
            'answered_accuracy': (k_ans / n_ans) if n_ans else math.nan, 'answered_lo': alo, 'answered_hi': ahi,
            'forced_accuracy': float(correct.mean()), 'forced_lo': lo, 'forced_hi': hi,
            'half_credit_accuracy': float((correct.sum() + 0.5 * (~ans).sum()) / n), 'n_correct': int(correct.sum())}
