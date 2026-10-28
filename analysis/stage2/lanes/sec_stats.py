"""Lane sec: statistics on SAVED multi-class predictions (nothing here fits a model).

* :func:`paired_balanced`: model A vs model B on the same trials by the balanced accuracy. The pooled difference
  bacc_A - bacc_B is a sum of per-person contributions (each trial weighs 1 / (n_classes * n_of_its_class)), so it
  gets the design of ``cv.paired_compare``: an exact sign-flip test on the per-person contributions
  (``cv._sign_flip_p``) and a person-cluster bootstrap CI (``cv.person_bootstrap``, the balanced accuracies recomputed
  on every resample, the same draws as every other CI of the lane).
* :func:`paired_swap`: the same paired design for a metric that is NOT a sum over persons (the macro-F1): the
  person-swap permutation test (under H0 the two models are exchangeable within each person, so each person's pair
  of predictions may be swapped; exact enumeration of the 2^n swap patterns for up to ``exact_max`` persons) and the
  person-bootstrap CI of the difference. For an additive metric (the balanced accuracy) it equals the sign-flip test.
* :func:`identical`: the reporting rule for differences below solver precision (no trial changes class and the
  log-loss difference is below :data:`LL_PRECISION`): such a pair is 'identical', never 'significant'.
* :func:`reference_agreement`: a grip model scored against another reference than the instruction (the grip the app
  itself detected at the start), and on the trials whose instruction the app confirmed.

Kept apart from ``sec_models`` on purpose: the label-permutation baseline records the hash of ``sec_models.py``
(an edit there recomputes it), and nothing here changes that baseline.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import sec_models as M

#: |mean per-person log-loss difference| below this, with no trial changing class, is 'identical' (solver precision)
LL_PRECISION = 1e-3
#: persons up to which the swap test enumerates every pattern (2^20 patterns take about a second)
SWAP_EXACT_MAX = 20
SWAP_N_MC = 20_000


def balanced_weights_present(y: np.ndarray) -> np.ndarray:
    """Per-trial weight 1 / (n_present_classes * n_class): sum(w * correct) is the balanced accuracy."""
    y = np.asarray(y, dtype=int)
    cls, cnt = np.unique(y, return_counts=True)
    w = {c: 1.0 / (cls.size * n) for c, n in zip(cls, cnt)}
    return np.array([w[v] for v in y])


def paired_balanced(person: Sequence[Any], y: Sequence[int], yhat_a: Sequence[int], yhat_b: Sequence[int], k: int, *,
                    n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict[str, Any]:
    """Balanced accuracy of A minus that of B on the same trials: the pooled difference, its person-bootstrap 95 % CI
    and the exact sign-flip p of the per-person contributions (persons whose contribution is positive / negative)."""
    person = np.asarray(person).astype(str)
    y = np.asarray(y, dtype=int)
    a = np.asarray(yhat_a, dtype=int)
    b = np.asarray(yhat_b, dtype=int)
    w = balanced_weights_present(y)
    contrib = pd.Series(w * ((a == y).astype(float) - (b == y).astype(float))).groupby(person).sum()
    bacc_a, bacc_b = M.balanced_accuracy(y, a, k), M.balanced_accuracy(y, b, k)
    pred = pd.DataFrame({'person_id': person, 'y': y, 'p_right': (a * k + b).astype(float)})

    def diff(yy: np.ndarray, code: np.ndarray) -> float:
        c = code.astype(int)
        return M.balanced_accuracy(yy, c // k, k) - M.balanced_accuracy(yy, c % k, k)

    lo, hi = cv.person_bootstrap(pred, diff, n_boot=n_boot, seed=seed)['stat']
    v = contrib.to_numpy(dtype=float)
    return {'bacc_a': bacc_a, 'bacc_b': bacc_b, 'd_bacc': float(bacc_a - bacc_b), 'd_bacc_ci': [lo, hi],
            'd_bacc_p': float(cv._sign_flip_p(v)), 'persons_a_better_bacc': int((v > 0).sum()),
            'persons_b_better_bacc': int((v < 0).sum())}


# ───────────────────────────── the person-swap test (any confusion-matrix metric) ─────────────────────────────


def macro_f1_cm(cm: np.ndarray) -> np.ndarray:
    """Macro-F1 of confusion matrices ``cm`` (..., k, k; rows = true class): the mean F1 over the classes present in
    the truth or the predictions (= sklearn ``f1_score(average='macro')``; ``sec_models.macro_f1``)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows, cols = cm.sum(axis=-1), cm.sum(axis=-2)
    den = rows + cols
    f1 = np.where(den > 0, 2.0 * tp / np.where(den > 0, den, 1.0), 0.0)
    present = den > 0
    return (f1 * present).sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)


def balanced_accuracy_cm(cm: np.ndarray) -> np.ndarray:
    """Balanced accuracy of confusion matrices (..., k, k): the mean recall over the classes present in the truth."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    rows = cm.sum(axis=-1)
    rec = np.where(rows > 0, tp / np.where(rows > 0, rows, 1.0), 0.0)
    present = rows > 0
    return (rec * present).sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)


CM_METRICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {'macro_f1': macro_f1_cm,
                                                             'balanced_accuracy': balanced_accuracy_cm}


def _person_cms(person: np.ndarray, y: np.ndarray, yhat: np.ndarray, k: int,
                persons: np.ndarray) -> np.ndarray:
    idx = {p: i for i, p in enumerate(persons.tolist())}
    pi = np.array([idx[p] for p in person.tolist()])
    out = np.zeros((persons.size, k * k))
    np.add.at(out, (pi, y * k + yhat), 1.0)
    return out


def swap_p(person: Sequence[Any], y: Sequence[int], yhat_a: Sequence[int], yhat_b: Sequence[int], k: int,
           metric: str = 'macro_f1', *, exact_max: int = SWAP_EXACT_MAX, n_mc: int = SWAP_N_MC,
           seed: int = C.SEED) -> tuple[float, float]:
    """(observed metric_A - metric_B, two-sided person-swap permutation p). Every pattern of swapping the two models'
    predictions of whole persons is evaluated from the per-person confusion matrices (exact for up to ``exact_max``
    persons, else ``n_mc`` seeded random patterns); p = share of patterns with |difference| >= the observed one."""
    person = np.asarray(person).astype(str)
    y, a, b = (np.asarray(v, dtype=int) for v in (y, yhat_a, yhat_b))
    f = CM_METRICS[metric]
    persons = np.unique(person)
    ca, cb = _person_cms(person, y, a, k, persons), _person_cms(person, y, b, k, persons)
    dd = cb - ca
    ta, tb = ca.sum(axis=0), cb.sum(axis=0)
    obs = float(f(ta.reshape(k, k)) - f(tb.reshape(k, k)))
    n = persons.size
    hits, total = 0, 0
    if n <= exact_max:
        bits = np.arange(n)
        step = 1 << 16
        for start in range(0, 1 << n, step):
            codes = np.arange(start, min(start + step, 1 << n))
            S = ((codes[:, None] >> bits) & 1).astype(float)
            sa = (ta + S @ dd).reshape(-1, k, k)
            sb = (tb - S @ dd).reshape(-1, k, k)
            d = f(sa) - f(sb)
            hits += int(np.sum(np.abs(d) >= abs(obs) - 1e-12))
            total += codes.size
    else:
        rng = np.random.default_rng(seed)
        S = rng.integers(0, 2, size=(n_mc, n)).astype(float)
        sa = (ta + S @ dd).reshape(-1, k, k)
        sb = (tb - S @ dd).reshape(-1, k, k)
        d = f(sa) - f(sb)
        hits, total = int(np.sum(np.abs(d) >= abs(obs) - 1e-12)), n_mc
    return obs, hits / total


def paired_swap(person: Sequence[Any], y: Sequence[int], yhat_a: Sequence[int], yhat_b: Sequence[int], k: int,
                metric: str = 'macro_f1', *, key: str = 'd_f1', n_boot: int = C.N_BOOT,
                seed: int = C.SEED) -> dict[str, Any]:
    """metric_A - metric_B on the same trials (default the macro-F1): the observed difference, its person-bootstrap
    95 % CI (``cv.person_bootstrap``, the same draws as every other CI of the lane) and the exact person-swap p."""
    person = np.asarray(person).astype(str)
    y, a, b = (np.asarray(v, dtype=int) for v in (y, yhat_a, yhat_b))
    f = CM_METRICS[metric]
    obs, p = swap_p(person, y, a, b, k, metric)
    pred = pd.DataFrame({'person_id': person, 'y': y, 'p_right': (a * k + b).astype(float)})

    def diff(yy: np.ndarray, code: np.ndarray) -> float:
        c = code.astype(int)
        cm_a = M.confusion(yy, c // k, k)
        cm_b = M.confusion(yy, c % k, k)
        return float(f(cm_a) - f(cm_b))

    lo, hi = cv.person_bootstrap(pred, diff, n_boot=n_boot, seed=seed)['stat']
    return {key: obs, f'{key}_ci': [lo, hi], f'{key}_p': p,
            f'{metric}_a': float(f(M.confusion(y, a, k))), f'{metric}_b': float(f(M.confusion(y, b, k)))}


def identical(n_changed: int, d_ll: float, tol: float = LL_PRECISION) -> bool:
    """True when no trial changes its predicted class and |log-loss difference| < tol: report 'identical'."""
    return int(n_changed) == 0 and math.isfinite(float(d_ll)) and abs(float(d_ll)) < tol


def reference_agreement(pred: pd.DataFrame, reference: pd.Series, classes: Sequence[str],
                        instructed_ok: pd.Series | None = None) -> dict[str, Any]:
    """A grip model's predictions (``pred``: trial_key, y, y_pred) against ``reference`` (trial_key -> class name or
    another value for 'unknown'): the trials with a known reference, how many the model agrees with, the balanced
    accuracy against it; and, when ``instructed_ok`` (trial_key -> bool) is given, the balanced accuracy against the
    instruction on the trials it marks."""
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    ref = pred['trial_key'].map(reference).map(idx)
    known = ref.notna().to_numpy()
    yr = ref[known].astype(int).to_numpy()
    yp = pred.loc[known, 'y_pred'].astype(int).to_numpy()
    out: dict[str, Any] = {'n_trials': int(len(pred)), 'n_known': int(known.sum()), 'n_unknown': int((~known).sum()),
                           'n_agree': int((yr == yp).sum()), 'share_agree': float(np.mean(yr == yp)) if yr.size else math.nan,
                           'bacc_vs_reference': M.balanced_accuracy(yr, yp, k) if yr.size else math.nan}
    if instructed_ok is not None:
        ok = pred['trial_key'].map(instructed_ok).fillna(False).astype(bool).to_numpy()
        y = pred['y'].astype(int).to_numpy()
        yh = pred['y_pred'].astype(int).to_numpy()
        out.update({'n_confirmed': int(ok.sum()), 'n_errors_confirmed': int(np.sum(y[ok] != yh[ok])),
                    'bacc_confirmed': M.balanced_accuracy(y[ok], yh[ok], k) if ok.any() else math.nan,
                    'bacc_all': M.balanced_accuracy(y, yh, k)})
    return out
