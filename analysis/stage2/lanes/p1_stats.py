"""Lane p1: the statistics of the derive step, in plain numpy (nothing here fits a panel model).

* :func:`paired_compare` - ``cv.paired_compare`` vectorised: the same definitions and the same resampling sequence
  (persons drawn with replacement), so the numbers equal the harness's within float rounding (test), in about a
  hundredth of the time (the harness indexes a DataFrame 2000 times per comparison).
* :func:`paired_metric_diff` - paired person bootstrap of the difference of POOLED metrics of two prediction sets
  on the same trials (log loss, Brier, AUC, calibration slope): the before / after evidence of a calibrator.
* :func:`calibration_fit` - calibration slope ``b`` of logit P(y = 1) = a + b logit(p) (1 = calibrated, < 1 =
  over-confident) and calibration-in-the-large ``a0`` of logit P(y = 1) = a0 + logit(p) (0 = calibrated), with
  person-bootstrap CIs (:func:`calibration_block`).
* :func:`ece_null` - the ECE expected under PERFECT calibration: y* ~ Bernoulli(p) on the same predictions and the
  same equal-mass bins (median, 95th percentile and the p-value of the observed ECE). ECE is biased upwards and
  its percentile bootstrap CI can exclude the estimate itself, so the lane reports this null band instead of an
  ECE CI. It treats trials as independent (an approximation: trials of one person are correlated).
* :func:`adjust` - Benjamini-Hochberg and Holm adjustments of a family of p-values (statsmodels).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stage2 import config as C

EPS = float(np.finfo(np.float64).eps)        # the harness's log-loss clipping (= cv._EPS)
LOGIT_CLIP = 1e-6                            # p clipped to [1e-6, 1 - 1e-6] before the logit of a calibration fit
N_NULL = 4000                                # simulations of the perfect-calibration ECE null


# ───────────────────────────── helpers ─────────────────────────────


def logit(p: Any, clip: float = LOGIT_CLIP) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), clip, 1 - clip)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -35, 35)))


def correct(y: Any, p: Any) -> np.ndarray:
    """Per-trial accuracy credit, the harness's rule (``cv.credit``; test: equal): 1 when P(right) is on the side of
    the true hand, 0 on the other side, 1/2 for P(right) = 0.5 up to rounding (|P - 0.5| <= 1e-12, ``cv.TIE_ATOL``:
    a tie claims no hand)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    c = ((p >= 0.5).astype(int) == y).astype(float)
    c[np.abs(p - 0.5) <= 1e-12] = 0.5
    return c


def log_loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def person_index(persons: Any) -> tuple[np.ndarray, np.ndarray]:
    """(sorted unique persons, index of each row's person)."""
    uniq, inv = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    return uniq, inv


def boot_picks(n_persons: int, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> np.ndarray:
    """(n_boot, n_persons) person indices drawn with replacement, in the harness's sequence
    (``rng.choice(uniq, size=uniq.size, replace=True)`` once per resample)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(n_persons, size=n_persons, replace=True) for _ in range(n_boot)])


def q95(v: Sequence[float]) -> tuple[float, float]:
    v = np.asarray([x for x in v if x is not None and math.isfinite(x)], dtype=float)
    if not v.size:
        return (math.nan, math.nan)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


# ───────────────────────────── paired comparison (the harness's, vectorised) ─────────────────────────────


def sign_flip_p(d: Any, *, exact_max: int = 20, seed: int = C.SEED) -> float:
    """``cv._sign_flip_p``: two-sided sign-flip p of mean(d) = 0, exact over all 2^n sign patterns for n <=
    exact_max persons, computed meet-in-the-middle (the sums of the two halves' patterns added as an outer sum)
    instead of n passes over 2^n codes; larger n falls back to the harness (Monte Carlo). Test: equal p-values."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return math.nan
    if n > exact_max:
        from stage2.cv import _sign_flip_p
        return _sign_flip_p(d, exact_max=exact_max, seed=seed)

    def half_sums(v: np.ndarray) -> np.ndarray:
        codes = np.arange(2 ** v.size, dtype=np.int64)
        tot = np.zeros(codes.size)
        for i in range(v.size):
            tot += (1.0 - 2.0 * ((codes >> i) & 1)) * v[i]
        return tot

    h = n // 2
    total = half_sums(d[:h])[None, :] + half_sums(d[h:])[:, None]
    obs = abs(d.mean())
    return float(np.mean(np.abs(total / n) >= obs - 1e-12))


def paired_compare(pa: pd.DataFrame, pb: pd.DataFrame, *, n_boot: int = C.N_BOOT, seed: int = C.SEED
                   ) -> dict[str, Any]:
    """``cv.paired_compare(a, b)`` on two prediction tables (trial_id, person_id, y, p_right): A - B on the shared
    trials; the same keys, definitions and resampling sequence (tests/test_p1.py checks the equality)."""
    _sign_flip_p = sign_flip_p
    j = pa[['trial_id', 'person_id', 'y', 'p_right']].merge(pb[['trial_id', 'p_right']], on='trial_id',
                                                             suffixes=('_a', '_b'))
    y = j['y'].to_numpy().astype(int)
    pa_, pb_ = j['p_right_a'].to_numpy(dtype=float), j['p_right_b'].to_numpy(dtype=float)
    ca, cb = correct(y, pa_), correct(y, pb_)
    la, lb = log_loss_terms(y, pa_), log_loss_terms(y, pb_)
    uniq, inv = person_index(j['person_id'])
    n = np.bincount(inv).astype(float)
    s_ca, s_cb = np.bincount(inv, ca), np.bincount(inv, cb)
    d_acc = s_ca / n - s_cb / n
    d_ll = np.bincount(inv, la) / n - np.bincount(inv, lb) / n
    picks = boot_picks(uniq.size, n_boot, seed)
    cnt = np.stack([np.bincount(pk, minlength=uniq.size) for pk in picks]).astype(float)
    tot = cnt @ n
    boot_pool = (cnt @ s_ca) / tot - (cnt @ s_cb) / tot
    boot_person = d_acc[picks].mean(axis=1)
    boot_ll = d_ll[picks].mean(axis=1)
    return {'n_trials': int(len(j)), 'n_persons': int(uniq.size), 'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
            'pooled_acc_diff': float(ca.mean() - cb.mean()), 'pooled_acc_diff_ci95': q95(boot_pool),
            'mean_person_acc_diff': float(d_acc.mean()), 'mean_person_acc_diff_ci95': q95(boot_person),
            'p_sign_flip_acc': _sign_flip_p(d_acc, seed=seed),
            'mean_person_logloss_diff': float(d_ll.mean()), 'mean_person_logloss_diff_ci95': q95(boot_ll),
            'p_sign_flip_logloss': _sign_flip_p(d_ll, seed=seed),
            'persons_a_better': int((d_acc > 0).sum()), 'persons_b_better': int((d_acc < 0).sum()),
            'max_abs_dp': float(np.max(np.abs(pa_ - pb_))) if len(j) else math.nan,
            'n_hand_changed': int(np.sum((pa_ >= 0.5) != (pb_ >= 0.5)))}


# ───────────────────────────── calibration statistics ─────────────────────────────


def calibration_fit(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, *, max_iter: int = 100
                    ) -> tuple[float, float, float]:
    """(slope b, intercept a, calibration-in-the-large a0): weighted maximum likelihood of logit P(y=1) = a + b z and
    of logit P(y=1) = a0 + z, z = logit(p), by DAMPED Newton-Raphson (:func:`calibration_fit_many`). NaN when a fit
    does not converge (e.g. separable data). Weights: the multiplicity of each row in a person-bootstrap resample."""
    z = np.asarray(p, dtype=float)
    w = np.ones(z.size) if w is None else np.asarray(w, dtype=float)
    b, a, a0 = calibration_fit_many(y, p, w[None, :], max_iter=max_iter)
    return float(b[0]), float(a[0]), float(a0[0])


def _loglik(W: np.ndarray, y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Weighted Bernoulli log-likelihood of every row of W (m, n) at linear predictors eta (m, n)."""
    return (W * (y[None, :] * eta - np.logaddexp(0.0, eta))).sum(axis=1)


def _accept(cur: np.ndarray, val: np.ndarray) -> np.ndarray:
    """A step is accepted when it does not lower the log-likelihood (beyond float rounding)."""
    return np.isfinite(val) & (val >= cur - 1e-12 * np.abs(cur) - 1e-12)


def calibration_fit_many(y: np.ndarray, p: np.ndarray, W: np.ndarray, *, max_iter: int = 100, chunk: int = 250,
                         max_halvings: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`calibration_fit` for every row of the weight matrix ``W`` (B, n) at once: Newton-Raphson from (a, b) =
    (0, 1) (and a0 = 0) with step halving until the log-likelihood does not decrease, so the iteration converges
    on the extreme logits of summed attempts too (plain Newton diverges there); converged when the accepted step
    is below 1e-10, failed (NaN) when no step is accepted, a coefficient exceeds 1e3 (separation) or max_iter is
    reached. Returns (slopes, intercepts, citl)."""
    y = np.asarray(y, dtype=float)
    z = logit(p)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    B = W.shape[0]
    slope, icpt, citl = np.full(B, np.nan), np.full(B, np.nan), np.full(B, np.nan)
    z2 = z * z
    for s0 in range(0, B, chunk):
        Wc = W[s0:s0 + chunk]
        m = Wc.shape[0]
        # two-parameter fit: slope and intercept
        a, b = np.zeros(m), np.ones(m)
        cur = _loglik(Wc, y, a[:, None] + b[:, None] * z[None, :])
        ok, active = np.zeros(m, dtype=bool), np.ones(m, dtype=bool)
        for _ in range(max_iter):
            idx = np.flatnonzero(active)
            if not idx.size:
                break
            Wi = Wc[idx]
            mu = sigmoid(a[idx, None] + b[idx, None] * z[None, :])
            r = Wi * (y[None, :] - mu)
            g0, g1 = r.sum(axis=1), r @ z
            v = Wi * (mu * (1.0 - mu))
            h00, h01, h11 = v.sum(axis=1), v @ z, v @ z2
            det = h00 * h11 - h01 * h01
            sing = ~(np.abs(det) > 0) | ~np.isfinite(det)
            with np.errstate(divide='ignore', invalid='ignore'):
                st0 = np.where(sing, 0.0, (h11 * g0 - h01 * g1) / det)
                st1 = np.where(sing, 0.0, (h00 * g1 - h01 * g0) / det)
            t = np.ones(idx.size)
            acc = sing.copy()                     # singular rows: nothing to try (they fail below)
            val = np.full(idx.size, -np.inf)
            for _h in range(max_halvings):
                j = np.flatnonzero(~acc)
                if not j.size:
                    break
                eta = (a[idx[j]] + t[j] * st0[j])[:, None] + (b[idx[j]] + t[j] * st1[j])[:, None] * z[None, :]
                vj = _loglik(Wi[j], y, eta)
                good = _accept(cur[idx[j]], vj)
                acc[j[good]] = True
                val[j[good]] = vj[good]
                t[j[~good]] *= 0.5
            fail = sing | ~np.isfinite(val)
            step0, step1 = t * st0, t * st1
            upd = ~fail
            a[idx[upd]] += step0[upd]
            b[idx[upd]] += step1[upd]
            cur[idx[upd]] = val[upd]
            div = fail | ~np.isfinite(a[idx]) | ~np.isfinite(b[idx]) | (np.maximum(np.abs(a[idx]), np.abs(b[idx])) > 1e3)
            conv = ~div & (np.maximum(np.abs(step0), np.abs(step1)) < 1e-10)
            ok[idx[conv]] = True
            active[idx[conv | div]] = False
        slope[s0:s0 + m] = np.where(ok, b, np.nan)
        icpt[s0:s0 + m] = np.where(ok, a, np.nan)
        # calibration-in-the-large: logit P = a0 + z (offset z)
        a0 = np.zeros(m)
        cur0 = _loglik(Wc, y, a0[:, None] + z[None, :])
        ok0, act0 = np.zeros(m, dtype=bool), np.ones(m, dtype=bool)
        for _ in range(max_iter):
            idx = np.flatnonzero(act0)
            if not idx.size:
                break
            Wi = Wc[idx]
            mu = sigmoid(a0[idx, None] + z[None, :])
            g = (Wi * (y[None, :] - mu)).sum(axis=1)
            h = (Wi * (mu * (1.0 - mu))).sum(axis=1)
            bad = ~(h > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                st = np.where(bad, 0.0, g / h)
            t = np.ones(idx.size)
            acc = bad.copy()
            val = np.full(idx.size, -np.inf)
            for _h in range(max_halvings):
                j = np.flatnonzero(~acc)
                if not j.size:
                    break
                vj = _loglik(Wi[j], y, (a0[idx[j]] + t[j] * st[j])[:, None] + z[None, :])
                good = _accept(cur0[idx[j]], vj)
                acc[j[good]] = True
                val[j[good]] = vj[good]
                t[j[~good]] *= 0.5
            fail = bad | ~np.isfinite(val)
            step = t * st
            a0[idx[~fail]] += step[~fail]
            cur0[idx[~fail]] = val[~fail]
            conv = ~fail & (np.abs(step) < 1e-12)
            ok0[idx[conv]] = True
            act0[idx[conv | fail]] = False
        citl[s0:s0 + m] = np.where(ok0, a0, np.nan)
    return slope, icpt, citl


def weighted_auc_many(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    """ROC AUC of p for every weight row of ``W`` (B, n): equal to the AUC of the rows repeated by their weights
    (ties count one half), i.e. of a person-bootstrap resample. Test: equal to the AUC of the expanded rows."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    order = np.argsort(p, kind='mergesort')
    ps, ys, Ws = p[order], y[order], W[:, order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    negG = np.add.reduceat(Ws * (ys == 0), starts, axis=1)
    posG = np.add.reduceat(Ws * (ys == 1), starts, axis=1)
    below = np.cumsum(negG, axis=1) - negG
    num = (posG * (below + 0.5 * negG)).sum(axis=1)
    den = posG.sum(axis=1) * negG.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def person_weights(inv: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """(B, n) multiplicity of every row in every person-bootstrap resample (``picks``: (B, persons))."""
    n_persons = int(inv.max()) + 1 if inv.size else 0
    cnt = np.stack([np.bincount(pk, minlength=n_persons) for pk in picks]).astype(float)
    return cnt[:, inv]


def calibration_block(pred: pd.DataFrame, *, n_boot: int = C.N_BOOT, seed: int = C.SEED,
                      min_persons: int | None = None) -> dict[str, Any]:
    """Calibration slope, intercept and calibration-in-the-large with person-bootstrap 95 % CIs (no CI below
    ``min_persons`` persons)."""
    from stage2.lanes.p1_labels import MIN_PERSONS_CI
    min_persons = MIN_PERSONS_CI if min_persons is None else min_persons
    y = pred['y'].to_numpy().astype(int)
    p = pred['p_right'].to_numpy(dtype=float)
    uniq, inv = person_index(pred['person_id'])
    b, a, a0 = calibration_fit(y, p)
    out = {'slope': b, 'intercept': a, 'citl': a0, 'slope_lo': None, 'slope_hi': None, 'citl_lo': None, 'citl_hi': None,
           'p_slope_below_1': None}
    if uniq.size < min_persons:
        return out
    bs, _, cs = calibration_fit_many(y, p, person_weights(inv, boot_picks(uniq.size, n_boot, seed)))
    out['slope_lo'], out['slope_hi'] = q95(bs.tolist())
    out['citl_lo'], out['citl_hi'] = q95(cs.tolist())
    fb = bs[np.isfinite(bs)]
    out['p_slope_below_1'] = float(np.mean(fb < 1.0)) if fb.size else None
    return out


def ece_bins(p: np.ndarray, n_bins: int = C.ECE_BINS) -> np.ndarray:
    """Equal-mass bin of every prediction, with the harness's edges (``cv.reliability``)."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    return np.searchsorted(edges[1:-1], p, side='right')


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = C.ECE_BINS) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    k = ece_bins(p, n_bins)
    return float(np.abs(np.bincount(k, y - p, n_bins)).sum() / y.size)


def ece_null(y: np.ndarray, p: np.ndarray, *, n_sim: int = N_NULL, seed: int = C.SEED, n_bins: int = C.ECE_BINS
             ) -> dict[str, float]:
    """The observed ECE and its distribution under perfect calibration (y* ~ Bernoulli(p), the same p and bins):
    median, 95th percentile, and p = P(ECE* >= ECE) (with the +1 correction)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    k = ece_bins(p, n_bins)
    B = np.zeros((p.size, n_bins))
    B[np.arange(p.size), k] = 1.0
    obs = float(np.abs((y - p) @ B).sum() / p.size)
    rng = np.random.default_rng([seed, 53])
    sims = []
    for start in range(0, n_sim, 500):
        m = min(500, n_sim - start)
        Ys = (rng.random((m, p.size)) < p).astype(float)
        sims.append(np.abs((Ys - p) @ B).sum(axis=1) / p.size)
    s = np.concatenate(sims)
    return {'ece': obs, 'null_median': float(np.median(s)), 'null_q95': float(np.quantile(s, 0.95)),
            'p_value': float((1 + np.sum(s >= obs - 1e-15)) / (1 + s.size)), 'n_sim': int(n_sim)}


# ───────────────────────────── paired differences of pooled metrics ─────────────────────────────


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = y.size - n1
    if not n1 or not n0:
        return math.nan
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


POOLED = ('log_loss', 'brier', 'roc_auc', 'slope', 'accuracy')


def pooled_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, rows: np.ndarray | None = None
                   ) -> dict[str, float]:
    """Pooled metrics of one resample: weights w (multiplicities) for the additive ones and the slope, the
    concatenated rows for the AUC."""
    w = np.ones(y.size) if w is None else w
    tot = w.sum()
    out = {'log_loss': float(np.sum(w * log_loss_terms(y, p)) / tot), 'brier': float(np.sum(w * (p - y) ** 2) / tot),
           'accuracy': float(np.sum(w * correct(y, p)) / tot)}
    out['roc_auc'] = _auc(y, p) if rows is None else _auc(y[rows], p[rows])
    out['slope'] = calibration_fit(y, p, w)[0]
    return out


def paired_metric_diff(pa: pd.DataFrame, pb: pd.DataFrame, *, n_boot: int = C.N_BOOT, seed: int = C.SEED,
                       metrics: Sequence[str] = POOLED) -> dict[str, Any]:
    """A - B of pooled metrics on the shared trials, with paired person-bootstrap 95 % CIs (both arms on the same
    resampled persons) and the share of resamples below 0."""
    j = pa[['trial_id', 'person_id', 'y', 'p_right']].merge(pb[['trial_id', 'p_right']], on='trial_id',
                                                             suffixes=('_a', '_b'))
    y = j['y'].to_numpy().astype(int)
    a, b = j['p_right_a'].to_numpy(dtype=float), j['p_right_b'].to_numpy(dtype=float)
    uniq, inv = person_index(j['person_id'])
    ma, mb = pooled_metrics(y, a), pooled_metrics(y, b)
    out: dict[str, Any] = {'n': int(len(j)), 'n_persons': int(uniq.size)}
    W = person_weights(inv, boot_picks(uniq.size, n_boot, seed))
    xa, xb = pooled_metrics_many(y, a, W, metrics), pooled_metrics_many(y, b, W, metrics)
    for m in metrics:
        d = ma[m] - mb[m]
        diff = xa[m] - xb[m]
        lo, hi = q95(diff.tolist())
        v = diff[np.isfinite(diff)]
        out[m] = {'a': ma[m], 'b': mb[m], 'diff': d, 'lo': lo, 'hi': hi,
                  'share_below_0': float(np.mean(v < 0)) if v.size else None}
    return out


def pooled_metrics_many(y: np.ndarray, p: np.ndarray, W: np.ndarray, metrics: Sequence[str] = POOLED
                        ) -> dict[str, np.ndarray]:
    """:func:`pooled_metrics` for every weight row of W (B, n), vectorised (test: equal to the loop)."""
    tot = W.sum(axis=1)
    out: dict[str, np.ndarray] = {}
    if 'log_loss' in metrics:
        out['log_loss'] = (W @ log_loss_terms(y, p)) / tot
    if 'brier' in metrics:
        out['brier'] = (W @ ((p - y) ** 2)) / tot
    if 'accuracy' in metrics:
        out['accuracy'] = (W @ correct(y, p)) / tot
    if 'roc_auc' in metrics:
        out['roc_auc'] = weighted_auc_many(y, p, W)
    if 'slope' in metrics:
        out['slope'] = calibration_fit_many(y, p, W)[0]
    return out


def person_boot_ci(pred: pd.DataFrame, fn: Callable[[np.ndarray, np.ndarray], float], *, n_boot: int = C.N_BOOT,
                   seed: int = C.SEED, min_persons: int | None = None) -> tuple[float | None, float | None]:
    """Person-bootstrap 95 % CI of fn(y, p) (``cv.person_bootstrap`` semantics), None below ``min_persons``."""
    from stage2.lanes.p1_labels import MIN_PERSONS_CI
    min_persons = MIN_PERSONS_CI if min_persons is None else min_persons
    if pred['person_id'].nunique() < min_persons:
        return None, None
    y = pred['y'].to_numpy().astype(int)
    p = pred['p_right'].to_numpy(dtype=float)
    uniq, inv = person_index(pred['person_id'])
    rows_of = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    vals = []
    for pk in boot_picks(uniq.size, n_boot, seed):
        sel = np.concatenate([rows_of[i] for i in pk])
        vals.append(fn(y[sel], p[sel]))
    lo, hi = q95(vals)
    return (lo if math.isfinite(lo) else None), (hi if math.isfinite(hi) else None)


# ───────────────────────────── multiplicity ─────────────────────────────


def adjust(pvals: Sequence[float]) -> tuple[list[float], list[float]]:
    """(Benjamini-Hochberg q-values, Holm-adjusted p-values) of one family (NaN stays NaN)."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, dtype=float)
    q, h = np.full(p.size, np.nan), np.full(p.size, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
        h[ok] = multipletests(p[ok], method='holm')[1]
    return q.tolist(), h.tolist()


# ───────────────────────────── natural use ─────────────────────────────


def natural_use(ex: pd.DataFrame | None, *, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict[str, Any]:
    """Natural-use scoring that never counts an abstention as an answer: a P(right) of exactly 0.5 (a tie, e.g.
    an isotonic step) is reported as not answered. coverage = answered share; answered accuracy; half credit (a
    tie scores 1/2: the harness's rule since 2026-10-28); ``harness_accuracy``: the forced-choice rule the harness
    used before (P >= 0.5 is 'right', which credits a tie on an all-right set)."""
    if ex is None or not len(ex):
        return {'n': 0}
    y = ex['y'].to_numpy().astype(int)
    p = ex['p_right'].to_numpy(dtype=float)
    tie = p == 0.5
    ans = ~tie
    correct = ((p > 0.5).astype(int) == y) & ans
    half = lambda yy, pp: float(np.mean(np.where(pp == 0.5, 0.5, ((pp > 0.5).astype(int) == yy))))  # noqa: E731
    lo, hi = person_boot_ci(ex[['person_id', 'y', 'p_right']], half, n_boot=n_boot, seed=seed)
    return {'n': int(y.size), 'n_right': int(y.sum()), 'n_ties': int(tie.sum()), 'coverage': float(ans.mean()),
            'answered_accuracy': float(correct.sum() / ans.sum()) if ans.any() else None,
            'half_credit': half(y, p), 'half_credit_lo': lo, 'half_credit_hi': hi,
            'harness_accuracy': float(np.mean((p >= 0.5).astype(int) == y))}


__all__ = ['paired_compare', 'paired_metric_diff', 'calibration_fit', 'calibration_fit_many', 'weighted_auc_many',
           'pooled_metrics_many', 'person_weights', 'calibration_block', 'ece', 'ece_null',
           'adjust', 'natural_use', 'person_boot_ci', 'logit', 'sigmoid', 'q95', 'boot_picks']
