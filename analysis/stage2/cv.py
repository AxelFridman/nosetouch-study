"""Leak-proof leave-one-person-out (LOSO) evaluation, metrics and comparisons for every Stage 2 lane.

    from stage2.cv import run_experiment
    def fit_predict(fold):                    # sees ONLY the training persons' labels
        model = make_model().fit(fold.X_train, fold.y_train)
        return model.predict_proba(fold.X_test)[:, 1]      # P(right) of the held-out person's trials
    res = run_experiment('logreg_yaw', fit_predict, X, d.trials)   # X rows aligned with d.trials

Guarantees (each one unit-tested in tests/test_cv.py):

* **Persons, never codes or sessions.** The rows are matched by ``trial_id`` against the curated trial table
  of the cache (:func:`stage2.data.canonical_trial_map`): ``meta['person_id']`` must equal the curated
  person of every row, and ``groups=`` may merge persons (leave-group-out, with a warning) but never split
  one. Grouping by the app's participant code or by session therefore raises :class:`LeakageError`. One
  outer fold per person (as many folds as persons in the rows), checked for person overlap.
* **Only the primary set is evaluated** unless the caller opts in (``sets=``); natural-use trials are never
  rows (they go through ``extra=``, scored out of fold and reported apart).
* **No label within reach of fit_predict.** It receives copies of the training rows (X, y, groups, meta)
  and of the test rows WITHOUT labels, conditions, block / trial order or lookup keys (no ``trial_id``:
  test rows carry an opaque ``row`` index, in a random order). X itself may not carry a label: a DataFrame
  column or dict key with a label / condition / key / demographic name, or any binary column that equals
  the label, raises. While fit_predict runs, the cache loaders refuse to run (``stage2.guard``).
* **Test-set fitting is detected.** Every fold (``leak_check='all'``, the default; 'sample' checks
  ``LEAK_SAMPLE_FOLDS`` seeded folds; 'off') is re-run on a random half of its test rows in a new order;
  predictions that change raise :class:`LeakageError` (a scaler fitted on X_test, per-person test-time
  normalisation, a row-position feature). A non-deterministic fit_predict gets a warning instead (seed it
  with ``fold.seed``). The checked folds are recorded.
* **docs/09 D2 cap.** At most ``cap_per_person`` (40) trials of one person enter a training fold: a seeded,
  hand-balanced choice that is the same for that person in every fold; the held-out person's trials are never
  capped. The persons affected and their dropped rows are recorded (``info.capped_training_rows_per_fold``).
* **Nested tuning.** ``fold.inner_splits()`` = GroupKFold over the TRAINING persons only. A lane that picks
  among configurations on the outer folds instead must say how many it tried (``n_configs_tried``, stored in
  the registry), because the best of many outer-fold scores is optimistic.

Outputs: ``analysis/reports/stage2/results/<name>.json`` (per-trial predictions: trial_id, trial_key,
person_id, y, p_right, fold, model, params; metrics with person-cluster bootstrap 95 % CIs; per-person
accuracy; reliability curve; k-attempt curve; the metrics on the trials that pass the predictor gates G1 and
G2 of docs/09 D5; provenance: git commit, code fingerprint, cache version) and one summary row in
``results/registry.csv``.

Metrics: accuracy, balanced accuracy, ROC AUC, log loss, Brier score, ECE (10 equal-mass bins on P(right))
and reliability data; per-person accuracy; CIs by resampling PERSONS (B = 2000); :func:`paired_compare`
of two experiments (per-person differences, paired person bootstrap, exact sign-flip permutation test);
:func:`k_attempt_curve`: accuracy when the P(right) of k random trials of one person and one hand are
averaged (k = 1..5, on the SAME persons for every k: those with at least 5 trials per hand).

**Ties count half** (:func:`credit`). A prediction is a right-hand answer when P(right) > 0.5 and a left-hand
answer when P(right) < 0.5; P(right) = 0.5 exactly (up to floating-point rounding: within ``TIE_ATOL`` = 1e-12)
claims no hand (an abstention, or a constant predictor) and earns 1/2 in every accuracy here: accuracy, balanced accuracy, gated and natural-use accuracy, per-person accuracy,
the k-attempt curve (a mean of exactly 0.5) and :func:`paired_compare`. (Before 2026-10-28 a tie counted as a
right-hand answer, which scored every abstention as correct on the all-right natural-use set;
``stage2.scripts.rescore`` re-scored the saved results from their saved predictions.) When some P(right) is a
tie, :func:`metrics` also reports ``coverage`` (the share of trials answered) and ``accuracy_answered`` (the
accuracy on them).

**ECE caveats.** The percentile person-bootstrap CI of the ECE (``ci['ece']``) is biased upwards (a resample that
repeats persons adds binning noise to every bin) and can exclude its own estimate: judge an ECE against its
distribution under perfect calibration, or compare ECEs only through paired differences, never through that CI.
The pooled equal-mass ECE is also not comparable for a predictor whose P is constant within a fold (a rule whose P
is its training hit rate): its bins then mostly sort the folds.

docs/09 D3 (a locked 25 % test set) is waived: the researcher's instruction is to evaluate every model by
leave-one-person-out on all persons.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import itertools
import json
import math
import re
import warnings
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold

from . import config as C
from .guard import LeakageError, fit_predict_scope

__all__ = ['LeakageError', 'Fold', 'run_experiment', 'paired_compare', 'load_result', 'metrics', 'k_attempt_curve',
           'credit', 'is_tie', 'result_spec', 'spec_matches', 'spec_fingerprint']

#: the accuracy rule every result records (``info['scoring']``): P(right) = 0.5 exactly earns half credit
SCORING = 'tie_half_credit'
#: a tie is P(right) within this of 0.5 (exactly 0.5 up to floating-point rounding: a calibrator can return
#: 0.5 +- 1 ulp for a tie; lanes m1 and m4 use the same tolerance)
TIE_ATOL = 1e-12

#: The only columns of the test rows a fit_predict may see: an opaque row index (``row``, assigned by the
#: harness), the person, platform / device (known in production from the user agent) and sensor-derived
#: numbers. No hand, no condition (instructed grip, speed, variant), no block / trial order (the plan
#: alternates hands), no lookup key (trial_id, session_id), no demographics.
TEST_META_COLUMNS: tuple[str, ...] = (
    'row', 'person_id', 'platform', 'engine', 'device_model', 'app_version',
    'duration_ms', 'peak_gyro_dps', 'peak_acc_ms2', 'closure_deg', 'approach_valid_frac', 'motion_hz',
)
#: Labels and what reveals them through the plan (block / trial order, the first hand).
LABEL_COLUMNS: frozenset[str] = frozenset({
    'hand', 'y_right', 'y', 'y_true', 'label', 'labels', 'target', 'hand_source', 'first_hand', 'hand_is_dominant',
    'hand_is_phone_hand', 'block_index', 'trial_index', 'trial_in_block', 'block_kind', 'rep', 'analysis_set',
    'excl_rule', 'excl_detail'})
#: Instructed conditions (not known in production).
CONDITION_COLUMNS: frozenset[str] = frozenset({
    'orientation', 'instructed_rotation', 'speed', 'variant', 'start_position', 'posture', 'tier', 'block_tier',
    'block_optional'})
#: Keys that look a trial up in the cache or the raw records.
KEY_COLUMNS: frozenset[str] = frozenset({
    'trial_id', 'trial_key', 'session_id', 'session_label', 'participant_code', 'participant_id', 'created_at',
    't0_epoch_ms', 'person_key'})
#: Person-level data that production never has.
DEMOGRAPHIC_COLUMNS: frozenset[str] = frozenset({
    'age_years', 'sex', 'handedness_self', 'phone_hand', 'height_cm', 'dominant_hand', 'nickname'})
FORBIDDEN_TEST_COLUMNS = LABEL_COLUMNS | CONDITION_COLUMNS | KEY_COLUMNS | DEMOGRAPHIC_COLUMNS
FORBIDDEN_X_COLUMNS = FORBIDDEN_TEST_COLUMNS | {'person_id', 'groups', 'group', 'fold', 'row'}
_LABEL_TOKENS = frozenset({'hand', 'hands', 'label', 'labels', 'target', 'targets', 'handedness'})


def forbidden_name(name: Any) -> bool:
    """True for a column / key name that must never be a feature: a label, condition, lookup key or
    demographic, or a name with a label-like token (``hand``, ``label``, ``target``...)."""
    s = str(name).strip().lower()
    if s in FORBIDDEN_X_COLUMNS:
        return True
    return bool(set(re.split(r'[^a-z0-9]+', s)) & _LABEL_TOKENS)


# ───────────────────────────── splits ─────────────────────────────


@dataclass
class Split:
    fold: int
    person: str
    train: np.ndarray
    test: np.ndarray


def loso_splits(groups: Sequence[Any]) -> list[Split]:
    """One split per group (sorted ids): test = that group's rows, train = everybody else's."""
    g = np.asarray(groups).astype(str)
    out = []
    for k, p in enumerate(sorted(set(g.tolist()))):
        test = np.flatnonzero(g == p)
        train = np.flatnonzero(g != p)
        out.append(Split(k, p, train, test))
    assert_disjoint(out, g)
    return out


def assert_disjoint(splits: Sequence[Split], groups: Sequence[Any], persons: Sequence[Any] | None = None) -> None:
    """No group (and, when given, no person) in the training AND the test rows of a split."""
    g = np.asarray(groups).astype(str)
    pr = None if persons is None else np.asarray(persons).astype(str)
    for s in splits:
        both = set(g[s.train].tolist()) & set(g[s.test].tolist())
        if both:
            raise LeakageError(f'fold {s.fold}: groups in train AND test: {sorted(both)}')
        if pr is not None:
            both = set(pr[s.train].tolist()) & set(pr[s.test].tolist())
            if both:
                raise LeakageError(f'fold {s.fold}: persons in train AND test: {sorted(both)}')
        if np.intersect1d(s.train, s.test).size:
            raise LeakageError(f'fold {s.fold}: rows in train AND test')


def inner_splits(groups_train: Sequence[Any], n_splits: int = C.INNER_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold over the training persons (indices into the TRAINING rows) for nested tuning."""
    g = np.asarray(groups_train).astype(str)
    k = min(n_splits, len(set(g.tolist())))
    if k < 2:
        raise ValueError('nested tuning needs at least 2 training persons')
    out = list(GroupKFold(n_splits=k).split(np.zeros(g.size), groups=g))
    for tr, va in out:
        if set(g[tr].tolist()) & set(g[va].tolist()):
            raise LeakageError('inner split: a person in both parts')
    return out


def cap_rows(groups: Sequence[Any], y: Sequence[int], cap: int | None, seed: int = C.SEED) -> np.ndarray:
    """Indices of the rows kept when every group may contribute at most ``cap`` rows (docs/09 D2).

    The choice is seeded per group (the same rows of a person in every fold) and hand-balanced: each hand
    keeps up to cap // 2 rows and a hand with fewer rows leaves its share to the other."""
    g = np.asarray(groups).astype(str)
    yy = np.asarray(y).astype(int)
    if cap is None:
        return np.arange(g.size)
    keep: list[np.ndarray] = []
    for p in sorted(set(g.tolist())):
        idx = np.flatnonzero(g == p)
        if idx.size <= cap:
            keep.append(idx)
            continue
        rng = np.random.default_rng([seed, zlib.crc32(p.encode())])
        by = {c: rng.permutation(idx[yy[idx] == c]) for c in (0, 1)}
        q = {c: min(by[c].size, cap // 2) for c in (0, 1)}
        spare = cap - q[0] - q[1]
        for c in (0, 1):  # the hand with more rows takes what the other cannot use
            extra = min(spare, by[c].size - q[c])
            q[c] += extra
            spare -= extra
        keep.append(np.concatenate([by[0][:q[0]], by[1][:q[1]]]))
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


# ───────────────────────────── the fold handed to fit_predict ─────────────────────────────


def _take(X: Any, idx: np.ndarray) -> Any:
    """Copy of rows ``idx`` of an array / DataFrame / dict of arrays / list."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].reset_index(drop=True).copy()
    if isinstance(X, pd.Series):
        return X.iloc[idx].reset_index(drop=True).copy()
    if isinstance(X, np.ndarray):
        return X[idx].copy()
    if isinstance(X, dict):
        return {k: _take(v, idx) for k, v in X.items()}
    if isinstance(X, (list, tuple)):
        return [X[i] for i in idx]
    raise TypeError(f'unsupported X type {type(X).__name__}')


def _concat(a: Any, b: Any) -> Any:
    """Rows of b appended to rows of a (same container type)."""
    if isinstance(a, pd.DataFrame):
        return pd.concat([a, b], ignore_index=True)
    if isinstance(a, np.ndarray):
        return np.concatenate([a, b], axis=0)
    if isinstance(a, dict):
        return {k: _concat(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return a + list(b)
    raise TypeError(f'unsupported X type {type(a).__name__}')


def _n_rows(X: Any) -> int:
    if isinstance(X, dict):
        lens = {len(v) for v in X.values()}
        if len(lens) != 1:
            raise ValueError('every array of an X dict needs the same number of rows')
        return lens.pop()
    return len(X)


@dataclass
class Fold:
    """What fit_predict sees for one outer fold. ``*_test`` holds no label: return P(right) per test row,
    in the order of the test rows (which is random; ``meta_test['row']`` is an opaque index)."""

    fold: int
    test_person: str
    X_train: Any
    y_train: np.ndarray
    groups_train: np.ndarray
    meta_train: pd.DataFrame
    X_test: Any
    meta_test: pd.DataFrame
    seed: int

    def inner_splits(self, n_splits: int = C.INNER_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
        """GroupKFold over the training persons (indices into X_train)."""
        return inner_splits(self.groups_train, n_splits)

    @property
    def n_test(self) -> int:
        return _n_rows(self.X_test)


FitPredict = Callable[[Fold], Any]


def _as_proba(out: Any, n: int, fold: int) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {}
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        out, info = out
    p = np.asarray(out, dtype=float).reshape(-1)
    if p.size != n:
        raise ValueError(f'fold {fold}: fit_predict returned {p.size} probabilities for {n} test rows')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError(f'fold {fold}: P(right) must be finite and in [0, 1]')
    return p, info


def _test_meta(meta: pd.DataFrame, idx: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    bad = FORBIDDEN_TEST_COLUMNS & set(columns)
    if bad:
        raise LeakageError(f'label-revealing, condition or lookup-key columns requested on the test side: {sorted(bad)}')
    keep = [c for c in columns if c in meta.columns and c != 'row']
    return meta.iloc[idx][keep].reset_index(drop=True).copy()


def _check_x(X: Any, y: np.ndarray, what: str = 'X') -> None:
    """Refuse an X that carries a label: forbidden column / key names, or a binary column equal to y or 1 - y."""
    if isinstance(X, pd.DataFrame):
        bad = [c for c in X.columns if forbidden_name(c)]
        if bad:
            raise LeakageError(f'{what} has label, condition, key or demographic columns: {bad}')
        num = X.select_dtypes(include=['number', 'bool', 'boolean'])
        if num.shape[1]:
            hits = _label_columns_in(num.to_numpy(dtype=float, na_value=np.nan), np.asarray(y).astype(int))
            if hits:
                raise LeakageError(f'{what}: columns {[num.columns[j] for j in hits[:5]]} are binary and equal the '
                                   'label: the label is a feature')
        return
    elif isinstance(X, dict):
        bad = [k for k in X if forbidden_name(k)]
        if bad:
            raise LeakageError(f'{what} has label, condition, key or demographic keys: {bad}')
        arrays = {f'key {k}': v for k, v in X.items() if isinstance(v, np.ndarray)}
        for k, v in X.items():
            if isinstance(v, pd.DataFrame):
                _check_x(v, y, f'{what}[{k!r}]')
    elif isinstance(X, np.ndarray):
        arrays = {'array': X}
    else:
        arrays = {}
    yy = np.asarray(y).astype(int)
    for name, A in arrays.items():
        hits = _label_columns_in(np.asarray(A), yy)
        if hits:
            raise LeakageError(f'{what} {name}: {len(hits)} binary column(s) equal the label (flattened column '
                               f'index {hits[:5]}): the label is a feature')


def _label_columns_in(A: np.ndarray, y: np.ndarray, chunk: int = 4096, min_per_class: int = 5) -> list[int]:
    """Flattened column indices of A (N, ...) that take exactly two values and reproduce y or 1 - y.

    Skipped when a class has fewer than ``min_per_class`` rows (any two-valued column of 2 rows would match)."""
    if A.ndim == 0 or A.shape[0] != y.size or A.dtype.kind not in 'biuf':
        return []
    if min(int((y == 1).sum()), int((y == 0).sum())) < min_per_class:
        return []
    A2 = A.reshape(A.shape[0], -1)
    pos = y == 1
    hits: list[int] = []
    for j0 in range(0, A2.shape[1], chunk):
        B = A2[:, j0:j0 + chunk].astype(np.float64, copy=False)
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mn, mx = np.nanmin(B, axis=0), np.nanmax(B, axis=0)
        two = (mx > mn) & np.all((B == mn) | (B == mx), axis=0)
        for j in np.flatnonzero(two):
            hi = B[:, j] == mx[j]
            if np.array_equal(hi, pos) or np.array_equal(hi, ~pos):
                hits.append(j0 + int(j))
    return hits


def _check_transductive(fit_predict: FitPredict, fold: Fold, p_full: np.ndarray, atol: float,
                        rng: np.random.Generator) -> str:
    """Re-run fit_predict on a random half of the test rows in a new order; the shared predictions must not move."""
    n = p_full.size
    if n < 2:
        return 'skipped (one test row)'
    sub = rng.permutation(n)[: max(1, n // 2)]
    m2 = fold.meta_test.iloc[sub].reset_index(drop=True).copy()
    if 'row' in m2.columns:
        m2['row'] = np.arange(sub.size)
    f2 = dataclasses.replace(fold, X_test=_take(fold.X_test, sub), meta_test=m2)
    with fit_predict_scope():
        p_sub, _ = _as_proba(fit_predict(f2), sub.size, fold.fold)
    if np.allclose(p_sub, p_full[sub], rtol=0.0, atol=atol):
        return 'passed'
    with fit_predict_scope():
        p_again, _ = _as_proba(fit_predict(fold), n, fold.fold)
    if not np.allclose(p_again, p_full, rtol=0.0, atol=atol):
        warnings.warn('fit_predict is not deterministic for a fixed fold (seed it with fold.seed): the '
                      'test-set dependence check could not be run', stacklevel=3)
        return 'nondeterministic'
    raise LeakageError(
        f'fold {fold.fold} ({fold.test_person}): predictions for the same test trials changed when only a random '
        f'half of the test rows was passed in a new order (max |dp| = {np.max(np.abs(p_sub - p_full[sub])):.3g}): '
        'fit_predict uses statistics of the test set or the position of a row (e.g. a scaler fitted on X_test, '
        'per-person test-time normalisation)')


# ───────────────────────────── metrics ─────────────────────────────

_EPS = float(np.finfo(np.float64).eps)  # the clipping of sklearn.metrics.log_loss for float64 input


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted 1/2 (= sklearn.metrics.roc_auc_score); NaN with one class."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(p)  # average ranks for ties
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def is_tie(p: Any, threshold: float = 0.5) -> np.ndarray:
    """P(right) equal to the threshold up to floating-point rounding (|P - threshold| <= ``TIE_ATOL``): no hand is
    claimed."""
    return np.abs(np.asarray(p, dtype=float) - threshold) <= TIE_ATOL


def credit(y: Any, p: Any, threshold: float = 0.5) -> np.ndarray:
    """Per-trial accuracy credit: 1 when P(right) is on the side of the true hand (> threshold for a right hand,
    < threshold for a left one), 0 on the other side, 1/2 for a tie (P(right) = threshold up to rounding,
    :func:`is_tie`: no hand is claimed). Its mean is the accuracy. Without ties it is exactly the former rule
    ``(p >= 0.5) == y``."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    c = ((p >= threshold).astype(int) == y).astype(float)
    c[is_tie(p, threshold)] = 0.5
    return c


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = C.ECE_BINS, strategy: str = 'quantile') -> float:
    """Expected calibration error of P(right): sum_b n_b/N |mean(y_b) - mean(p_b)|.
    strategy 'quantile' (equal-mass bins, docs/09 D17) or 'uniform' (equal-width bins on [0, 1]).

    Not comparable across predictors whose P is constant within a fold (the bins then sort the folds), and its
    percentile person-bootstrap CI is biased upwards (see the module docstring).

    Computed with numpy on the bins of :func:`reliability` (no DataFrame: it runs once per bootstrap resample),
    with the same element operations and summation, so it equals the reliability-table formula exactly."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = _reliability_bins(p, n_bins, strategy)
    if not bins:
        return math.nan
    n = np.array([b.size for b in bins], dtype=np.int64)
    dev = np.abs(np.array([float(y[b].mean()) for b in bins]) - np.array([float(p[b].mean()) for b in bins]))
    return float(np.sum(n / n.sum() * dev))


def _reliability_bins(p: np.ndarray, n_bins: int, strategy: str) -> list[np.ndarray]:
    """Row indices of the non-empty bins of P(right), in bin order (see :func:`reliability`)."""
    if p.size == 0:
        return []
    if strategy == 'quantile':
        # equal-mass edges; tied probabilities always share a bin (so bins can be unequal with many ties)
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        k = np.searchsorted(edges[1:-1], p, side='right')
    elif strategy == 'uniform':
        k = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    bins = [np.flatnonzero(k == i) for i in range(n_bins)]
    return [b for b in bins if b.size]


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = C.ECE_BINS, strategy: str = 'quantile') -> pd.DataFrame:
    """Reliability-curve data: per bin p_lo, p_hi, p_mean, y_mean (observed share of right hands), n."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        return pd.DataFrame(columns=['bin', 'p_lo', 'p_hi', 'p_mean', 'y_mean', 'n'])
    bins = _reliability_bins(p, n_bins, strategy)
    rows = [{'bin': i, 'p_lo': float(p[b].min()), 'p_hi': float(p[b].max()), 'p_mean': float(p[b].mean()),
             'y_mean': float(y[b].mean()), 'n': int(b.size)} for i, b in enumerate(bins)]
    return pd.DataFrame(rows)


def metrics(y: Any, p: Any, threshold: float = 0.5) -> dict[str, float]:
    """accuracy, balanced_accuracy, roc_auc, log_loss, brier, ece (+ n, n_right). P(right) > 0.5 answers right,
    < 0.5 left, and a tie (0.5 up to rounding: no hand claimed) earns half credit (:func:`credit`). When some
    P(right) is a tie (a model that can abstain), also ``coverage`` (share of trials answered) and
    ``accuracy_answered`` (accuracy on the answered trials; NaN when none is answered)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out: dict[str, float] = {'n': float(y.size), 'n_right': float(y.sum())}
    if y.size == 0:
        return {**out, **{k: math.nan for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier', 'ece')}}
    c = credit(y, p, threshold)
    out['accuracy'] = float(np.mean(c))
    rec = [float(np.mean(c[y == k])) for k in (0, 1) if np.any(y == k)]
    out['balanced_accuracy'] = float(np.mean(rec))
    out['roc_auc'] = roc_auc(y, p)
    pc = np.clip(p, _EPS, 1 - _EPS)
    out['log_loss'] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    out['brier'] = float(np.mean((p - y) ** 2))
    out['ece'] = ece(y, p)
    tie = is_tie(p, threshold)
    if tie.any():
        out['coverage'] = float(np.mean(~tie))
        out['accuracy_answered'] = float(np.mean(c[~tie])) if (~tie).any() else math.nan
    return out


METRIC_NAMES = ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier', 'ece')


def per_person(pred: pd.DataFrame) -> pd.DataFrame:
    """Per person: n, accuracy, accuracy per hand, mean log loss (pred: person_id, y, p_right); a tie earns half
    credit (:func:`credit`)."""
    rows = []
    for pid, g in pred.groupby('person_id', sort=True):
        y = g['y'].to_numpy().astype(int)
        p = g['p_right'].to_numpy(dtype=float)
        ok = credit(y, p)
        pc = np.clip(p, _EPS, 1 - _EPS)
        rows.append({'person_id': pid, 'n': len(g), 'accuracy': float(ok.mean()),
                     'acc_left': float(ok[y == 0].mean()) if np.any(y == 0) else math.nan,
                     'acc_right': float(ok[y == 1].mean()) if np.any(y == 1) else math.nan,
                     'log_loss': float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))})
    return pd.DataFrame(rows)


def person_bootstrap(pred: pd.DataFrame, fn: Callable[[np.ndarray, np.ndarray], float] | None = None, *,
                     n_boot: int = C.N_BOOT, seed: int = C.SEED, ci: float = 0.95,
                     names: Sequence[str] = METRIC_NAMES) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling PERSONS with replacement (all trials of a drawn person come along).

    fn(y, p) -> float for one custom statistic (returned under 'stat'); default: every metric in names. The ECE's
    percentile CI is biased upwards and can exclude the estimate (module docstring): do not cite it."""
    persons = pred['person_id'].astype(str).to_numpy()
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    y = pred['y'].to_numpy().astype(int)
    p = pred['p_right'].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    stats: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([rows[u] for u in pick])
        vals = {'stat': fn(y[sel], p[sel])} if fn is not None else {k: v for k, v in metrics(y[sel], p[sel]).items()
                                                                     if k in names}
        for k, v in vals.items():
            if v is not None and math.isfinite(v):
                stats.setdefault(k, []).append(float(v))
    a = (1 - ci) / 2
    return {k: (float(np.quantile(v, a)), float(np.quantile(v, 1 - a))) for k, v in stats.items()}


# ───────────────────────────── k-attempt aggregation ─────────────────────────────


def k_attempt_curve(pred: pd.DataFrame, ks: Sequence[int] = C.K_ATTEMPTS, *, n_draws: int = 200,
                    seed: int = C.SEED, n_boot: int = 1000, common_persons: bool = True) -> pd.DataFrame:
    """Accuracy when the P(right) of k random trials of the same person and hand are averaged.

    For every (person, hand) cell with >= k trials, ``n_draws`` random k-subsets (without replacement) are
    averaged; a draw is correct when mean P(right) > 0.5 for a right hand (< 0.5 for a left one), and a mean of
    exactly 0.5 earns half credit (:func:`credit`). The
    cell accuracy is the share of correct draws; ``accuracy`` is the mean over persons of the mean over
    that person's hands, with a person-bootstrap CI. k = 1 is therefore the per-person balanced accuracy,
    not the pooled trial accuracy.

    ``common_persons`` (default): every k uses the SAME persons, those with at least max(ks) trials of
    each hand, so the curve shows the gain from averaging and not a change of persons (``n_persons`` and
    ``persons_left_out`` say who). Otherwise each k uses every cell with >= k trials."""
    rng = np.random.default_rng(seed)
    kmax = max(ks)
    counts = pred.groupby(['person_id', 'y']).size().unstack(fill_value=0)
    for c in (0, 1):
        if c not in counts.columns:
            counts[c] = 0
    eligible = set(counts.index[(counts[0] >= kmax) & (counts[1] >= kmax)].astype(str)) if common_persons else None
    left_out = sorted(set(counts.index.astype(str)) - eligible) if eligible is not None else []
    out = []
    for k in ks:
        cell: dict[tuple[str, int], float] = {}
        for (pid, yy), g in pred.groupby(['person_id', 'y'], sort=True):
            if eligible is not None and str(pid) not in eligible:
                continue
            p = g['p_right'].to_numpy(dtype=float)
            if p.size < k:
                continue
            if math.comb(p.size, k) <= n_draws:
                subsets = np.array(list(itertools.combinations(range(p.size), k)))
            else:
                subsets = np.array([rng.choice(p.size, size=k, replace=False) for _ in range(n_draws)])
            m = p[subsets].mean(axis=1)
            correct = ((m >= 0.5) if int(yy) == 1 else (m < 0.5)).astype(float)
            correct[is_tie(m)] = 0.5                   # a tie claims no hand: half credit
            cell[(str(pid), int(yy))] = float(correct.mean())
        base = {'k': k, 'common_persons': bool(common_persons), 'persons_left_out': ';'.join(left_out)}
        if not cell:
            out.append({**base, 'accuracy': math.nan, 'lo': math.nan, 'hi': math.nan, 'n_persons': 0, 'n_cells': 0})
            continue
        per_person_acc = pd.Series(cell).groupby(level=0).mean()
        vals = per_person_acc.to_numpy()
        b = np.random.default_rng(seed + k)
        boots = [float(np.mean(vals[b.integers(0, vals.size, vals.size)])) for _ in range(n_boot)]
        out.append({**base, 'accuracy': float(vals.mean()), 'lo': float(np.quantile(boots, 0.025)),
                    'hi': float(np.quantile(boots, 0.975)), 'n_persons': int(vals.size), 'n_cells': len(cell)})
    return pd.DataFrame(out)


# ───────────────────────────── results ─────────────────────────────


def _jsonable(v: Any) -> Any:
    """A JSON-safe copy (the form a result is saved in): NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.floating,)):
        return _jsonable(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _json_norm(v: Any) -> Any:
    """The saved form of a value (JSON round trip: tuples become lists, numpy scalars Python numbers)."""
    return json.loads(json.dumps(_jsonable(v), sort_keys=True, default=str))


#: the declared specification of a result: what the lane asked the harness for (not what came out of it)
SPEC_KEYS = ('model', 'params', 'notes', 'n_configs_tried', 'sets', 'cap_per_person', 'seed', 'n_boot', 'leak_check',
             'test_meta_columns')


def _declared_spec(model: Any, params: Any, info: dict[str, Any]) -> dict[str, Any]:
    spec = {'model': model, 'params': params or {}, 'notes': info.get('notes', ''),
            'n_configs_tried': info.get('n_configs_tried'), 'sets': info.get('sets'),
            'cap_per_person': info.get('cap_per_person'), 'seed': info.get('seed'), 'n_boot': info.get('n_boot'),
            'leak_check': info.get('leak_check_mode'), 'test_meta_columns': info.get('test_meta_columns')}
    for k, v in (info.get('spec') or {}).items():    # the lane's own extras (e.g. a code version); never a declared key
        spec.setdefault(k, v)
    return _json_norm(spec)


def result_spec(result: 'ExperimentResult | dict[str, Any]') -> dict[str, Any]:
    """The declared specification of a result (an ExperimentResult or its saved JSON): ``SPEC_KEYS`` read from the
    fields every result stores, plus the lane's ``spec=`` extras of :func:`run_experiment`. It is a pure function of
    the saved file, so every existing result has one without being rewritten."""
    if isinstance(result, ExperimentResult):
        return _declared_spec(result.model, result.params, result.info)
    return _declared_spec(result.get('model'), result.get('params'), result.get('info') or {})


def spec_matches(saved: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """True when every key of ``wanted`` (a lane's current declaration, any subset of the keys) equals the saved
    specification's value (compared in their saved JSON form)."""
    w = _json_norm(dict(wanted))
    return all(k in saved and saved[k] == v for k, v in w.items())


def spec_fingerprint(spec: dict[str, Any]) -> str:
    """sha256 of a specification in its saved JSON form (``info['spec_sha256']`` of a result)."""
    import hashlib
    return hashlib.sha256(json.dumps(_json_norm(spec), sort_keys=True).encode()).hexdigest()


@dataclass
class ExperimentResult:
    name: str
    model: str
    params: dict[str, Any]
    predictions: pd.DataFrame          # trial_id, trial_key, person_id, y, p_right, fold, model, params
    metrics: dict[str, float]
    ci: dict[str, tuple[float, float]]
    per_person: pd.DataFrame
    reliability: pd.DataFrame
    k_curve: pd.DataFrame
    leak_check: str
    info: dict[str, Any] = field(default_factory=dict)
    extra_predictions: pd.DataFrame | None = None   # e.g. the natural-use trials, scored out of fold
    extra_metrics: dict[str, float] | None = None
    gated_metrics: dict[str, float] | None = None   # on the trials that pass the predictor gates G1 and G2

    def summary_row(self) -> dict[str, Any]:
        pp = self.per_person['accuracy'].to_numpy() if len(self.per_person) else np.array([math.nan])
        row = {'name': self.name, 'model': self.model, 'created_at': self.info.get('created_at'),
               'n_trials': int(self.metrics.get('n', 0)), 'n_persons': int(self.predictions['person_id'].nunique()),
               'scheme': self.info.get('scheme'), 'n_folds': self.info.get('n_folds')}
        for k in METRIC_NAMES:
            row[k] = self.metrics.get(k)
            lo, hi = self.ci.get(k, (math.nan, math.nan))
            row[f'{k}_lo'], row[f'{k}_hi'] = lo, hi
        row.update({'person_acc_median': float(np.nanmedian(pp)), 'person_acc_p10': float(np.nanpercentile(pp, 10)),
                    'n_persons_below_80': int(np.sum(pp < 0.8))})
        for _, r in self.k_curve.iterrows():
            row[f'acc_k{int(r["k"])}'] = r['accuracy']
        if len(self.k_curve) and 'n_persons' in self.k_curve.columns:
            row['k_n_persons'] = ';'.join(str(int(v)) for v in self.k_curve['n_persons'])
        if self.gated_metrics:
            row['gated_n'] = int(self.gated_metrics.get('n', 0))
            row['gated_accuracy'] = self.gated_metrics.get('accuracy')
        if self.extra_metrics:
            row['extra_n'] = int(self.extra_metrics.get('n', 0))
            row['extra_accuracy'] = self.extra_metrics.get('accuracy')
        for pre, m in (('', self.metrics), ('extra_', self.extra_metrics or {})):   # only a model that abstains
            if 'coverage' in m:
                row[f'{pre}coverage'] = m.get('coverage')
                row[f'{pre}accuracy_answered'] = m.get('accuracy_answered')
        code = self.info.get('code') or {}
        row.update({'leak_check': self.leak_check, 'n_configs_tried': self.info.get('n_configs_tried'),
                    'cap_per_person': self.info.get('cap_per_person'), 'sets': ';'.join(self.info.get('sets') or []),
                    'git_commit': self.info.get('git_commit'), 'git_dirty': self.info.get('git_dirty'),
                    'stage2_sha256': (code.get('stage2_sha256') or '')[:12],
                    'cache_version': self.info.get('cache_version'), 'notes': self.info.get('notes', '')})
        return row

    def to_json(self) -> dict[str, Any]:
        clean = _jsonable
        return clean({
            'name': self.name, 'model': self.model, 'params': self.params, 'leak_check': self.leak_check,
            'info': self.info, 'metrics': self.metrics, 'ci95': {k: list(v) for k, v in self.ci.items()},
            'gated_metrics': self.gated_metrics,
            'per_person': self.per_person.to_dict(orient='records'),
            'reliability': self.reliability.to_dict(orient='records'),
            'k_attempts': self.k_curve.to_dict(orient='records'),
            'predictions': self.predictions.to_dict(orient='records'),
            'extra_metrics': self.extra_metrics,
            'extra_predictions': (None if self.extra_predictions is None
                                  else self.extra_predictions.to_dict(orient='records')),
        })

    def save(self, out_dir: str | Path = C.RES_DIR, registry: str | Path | None = C.REGISTRY_CSV) -> Path:
        """results/<name>.json (replaced atomically: a parallel reader never sees half a result) and its registry
        row."""
        from .locking import atomic_path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f'{self.name}.json'
        with atomic_path(path) as tmp:
            tmp.write_text(json.dumps(self.to_json(), indent=1, allow_nan=False), encoding='utf-8')
        if registry is not None:
            update_registry(self.summary_row(), registry)
        return path

    @classmethod
    def load(cls, path: str | Path) -> 'ExperimentResult':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        nan = lambda v: math.nan if v is None else v  # noqa: E731
        ex = d.get('extra_predictions')
        return cls(name=d['name'], model=d['model'], params=d.get('params') or {},
                   predictions=pd.DataFrame(d['predictions']),
                   metrics={k: nan(v) for k, v in d['metrics'].items()},
                   ci={k: tuple(nan(x) for x in v) for k, v in d['ci95'].items()},
                   per_person=pd.DataFrame(d['per_person']), reliability=pd.DataFrame(d['reliability']),
                   k_curve=pd.DataFrame(d['k_attempts']), leak_check=d.get('leak_check', ''), info=d.get('info') or {},
                   extra_predictions=None if ex is None else pd.DataFrame(ex), extra_metrics=d.get('extra_metrics'),
                   gated_metrics=d.get('gated_metrics'))


def update_registry(row: dict[str, Any], registry: str | Path = C.REGISTRY_CSV) -> pd.DataFrame:
    """Insert or replace (by name) one summary row of the results registry CSV.

    Process-safe: the read-modify-write holds the registry's cross-process lock (``<registry>.lock``) and the
    file is replaced atomically (``stage2.locking``), so parallel lanes or workers never lose a row and a reader
    never sees a half-written file."""
    from .locking import atomic_path, file_lock
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(registry):
        reg = pd.read_csv(registry) if registry.exists() else pd.DataFrame()
        if len(reg) and 'name' in reg.columns:
            reg = reg[reg['name'] != row['name']]
        new = pd.DataFrame([row])
        reg = new if not len(reg) else pd.concat([reg, new], ignore_index=True)
        with atomic_path(registry) as tmp:
            reg.to_csv(tmp, index=False)
    return reg


def evaluate_predictions(pred: pd.DataFrame, *, n_boot: int = C.N_BOOT, seed: int = C.SEED
                         ) -> tuple[dict[str, float], dict[str, tuple[float, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(metrics, CIs, per-person table, reliability, k-attempt curve) of a prediction table."""
    m = metrics(pred['y'], pred['p_right'])
    pp = per_person(pred)
    m['n_persons'] = float(pred['person_id'].nunique())
    m['person_acc_median'] = float(pp['accuracy'].median())
    m['person_acc_p10'] = float(np.percentile(pp['accuracy'], 10))
    m['n_persons_below_80'] = float((pp['accuracy'] < 0.8).sum())
    ci = person_bootstrap(pred, n_boot=n_boot, seed=seed) if n_boot else {}
    return m, ci, pp, reliability(pred['y'], pred['p_right']), k_attempt_curve(pred, seed=seed)


# ───────────────────────────── run_experiment ─────────────────────────────


def _canonical(meta: pd.DataFrame) -> pd.DataFrame | None:
    """Curated person_id / analysis_set / trial_key of the rows (by trial_id), or None for rows that are not
    from the Stage 2 cache (synthetic data). Mixing cached and unknown trial ids is refused."""
    from .data import canonical_trial_map  # local import: data pulls in motionlab
    cmap = canonical_trial_map()
    if cmap is None or 'trial_id' not in meta.columns:
        return None
    tid = meta['trial_id'].astype(str)
    known = tid.isin(cmap.index)
    if not known.any():
        return None
    if not known.all():
        raise ValueError(f'{int((~known).sum())} rows have trial ids that are not in the Stage 2 cache, '
                         f'{int(known.sum())} are: rows must all come from the cache')
    return cmap.loc[tid.to_numpy()].reset_index()


def _validate_rows(meta: pd.DataFrame, canon: pd.DataFrame | None, sets: Sequence[str], what: str) -> np.ndarray:
    """The curated person of every row; raises when meta disagrees or a row's analysis set is not allowed."""
    if meta['trial_id'].duplicated().any():
        raise ValueError(f'{what}: duplicated trial_id')
    if canon is not None:
        persons = canon['person_id'].astype(str).to_numpy()
        if 'person_id' in meta.columns:
            mine = meta['person_id'].astype(str).to_numpy()
            bad = mine != persons
            if bad.any():
                raise LeakageError(
                    f'{what}: meta person_id differs from the curated person_id in {int(bad.sum())} rows '
                    f'(e.g. {mine[bad][0]} instead of {persons[bad][0]}): group by person_id only, never by '
                    'participant_code or session')
        aset = canon['analysis_set'].astype(str).to_numpy()
    else:
        if 'person_id' not in meta.columns:
            raise ValueError(f'{what}: meta needs person_id')
        persons = meta['person_id'].astype(str).to_numpy()
        aset = (meta['analysis_set'].astype(str).to_numpy() if 'analysis_set' in meta.columns
                else np.full(len(meta), 'primary'))
    if np.any(pd.isna(persons)) or np.any(persons == 'nan') or np.any(persons == 'None'):
        raise ValueError(f'{what}: every row needs a person_id')
    bad = ~np.isin(aset, list(sets))
    if bad.any():
        vc = pd.Series(aset[bad]).value_counts().to_dict()
        raise ValueError(f'{what}: rows of analysis sets {vc} are not allowed here (allowed: {list(sets)})')
    return persons


def run_experiment(
    name: str,
    fit_predict: FitPredict,
    X: Any,
    meta: pd.DataFrame,
    *,
    y: Sequence[int] | None = None,
    groups: Sequence[Any] | None = None,
    sets: Sequence[str] = C.TRAIN_SETS,
    model: str | None = None,
    params: dict[str, Any] | None = None,
    test_meta_columns: Sequence[str] = TEST_META_COLUMNS,
    leak_check: str = C.LEAK_CHECK,
    leak_atol: float = 1e-9,
    cap_per_person: int | None = C.CAP_PER_PERSON,
    n_configs_tried: int = 1,
    seed: int = C.SEED,
    n_boot: int = C.N_BOOT,
    save: bool = True,
    out_dir: str | Path = C.RES_DIR,
    registry: str | Path | None = C.REGISTRY_CSV,
    notes: str = '',
    verbose: bool = False,
    extra: tuple[Any, pd.DataFrame] | None = None,
    extra_sets: Sequence[str] = ('natural_use', 'natural_use_repeat'),
    spec: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Leave-one-person-out: one outer fold per person, ``fit_predict(fold) -> P(right)`` (or
    ``(P(right), info_dict)``, info = chosen hyperparameters etc., stored per trial).

    X: rows aligned with ``meta`` (an array (N, ...), a DataFrame, or a dict of such arrays); it must not
    carry a label (see the module docstring).
    meta: the trial table of the rows (``Stage2Data.trials``) with trial_id, person_id and y_right; the
    training side gets all of it, the test side only ``test_meta_columns`` plus an opaque ``row`` index.
    groups: optional coarser grouping (leave-group-out); it may merge persons but never split one.
    sets: analysis sets allowed as rows (default the primary set; 'excluded' for a sensitivity analysis on
    purpose). Natural-use trials are never rows: pass them as ``extra``.
    extra: (X_extra, meta_extra), e.g. the natural-use set: each person's extra rows join that person's TEST
    rows (never trained on), are scored by the same fitted model and are reported separately
    (``extra_predictions``, ``extra_metrics``), never in the main metrics.
    leak_check: 'all' (default), 'sample' (``C.LEAK_SAMPLE_FOLDS`` seeded folds) or 'off'.
    cap_per_person: docs/09 D2 cap on the training rows of one person per fold (None: no cap).
    n_configs_tried: how many configurations the lane compared on these outer folds before choosing this one
    (1 = pre-specified; tuning inside fit_predict on fold.inner_splits() does not count).
    spec: optional JSON-able extras of the experiment's declared specification (for example a code version or the
    feature set), stored in ``info['spec']``; ``stage2.pipeline.should_run(..., spec=...)`` can then recompute a
    result whose specification changed. ``info['spec_sha256']`` fingerprints the whole declared specification
    (:func:`result_spec`), ``info['scoring']`` the accuracy rule (ties earn half credit)."""
    if leak_check not in ('all', 'sample', 'off'):
        raise ValueError("leak_check must be 'all', 'sample' or 'off'")
    bad_sets = {'natural_use', 'natural_use_repeat'} & set(sets)
    if bad_sets:
        raise ValueError(f'{sorted(bad_sets)} trials are never trained on: pass them with extra=')
    if int(n_configs_tried) < 1:
        raise ValueError('n_configs_tried must be >= 1')
    n = _n_rows(X)
    if len(meta) != n:
        raise ValueError(f'X has {n} rows, meta {len(meta)}')
    meta = meta.reset_index(drop=True)
    if 'trial_id' not in meta.columns:
        raise ValueError('meta needs trial_id')
    canon = _canonical(meta)
    persons = _validate_rows(meta, canon, sets, 'rows')
    yv = np.asarray(meta['y_right'] if y is None else y).astype(int)
    if yv.size != n:
        raise ValueError('y and X differ in length')
    _check_x(X, yv)
    if groups is None:
        g = persons
        scheme = 'loso'
    else:
        g = np.asarray(groups).astype(str)
        if g.size != n:
            raise ValueError('groups and X differ in length')
        per = pd.DataFrame({'p': persons, 'g': g}).groupby('p')['g'].nunique()
        split = sorted(per.index[per > 1])
        if split:
            raise LeakageError(f'groups split persons over several folds: {split} (group by person_id only, never by '
                               'participant_code or session)')
        merged = pd.DataFrame({'p': persons, 'g': g}).groupby('g')['p'].nunique()
        scheme = 'loso' if (merged == 1).all() else 'leave-group-out'
        if scheme != 'loso':
            warnings.warn(f'groups merge persons ({int((merged > 1).sum())} groups hold several persons): the '
                          'scheme is leave-group-out, not leave-one-person-out', stacklevel=2)
    Xe, me, ge, ye, pe, extra_persons = None, None, None, None, None, None
    if extra is not None:
        Xe, me = extra
        me = me.reset_index(drop=True)
        if _n_rows(Xe) != len(me):
            raise ValueError('extra: X and meta differ in length')
        if 'trial_id' not in me.columns:
            raise ValueError('extra: meta needs trial_id')
        canon_e = _canonical(me)
        extra_persons = _validate_rows(me, canon_e, extra_sets if canon_e is not None or 'analysis_set' in me.columns
                                       else ('primary',), 'extra rows')
        shared = set(me['trial_id'].astype(str)) & set(meta['trial_id'].astype(str))
        if shared:
            raise LeakageError(f'{len(shared)} trials are both rows and extra rows')
        ye = me['y_right'].astype(int).to_numpy()
        _check_x(Xe, ye, 'extra X')
        if groups is not None:  # an extra row follows its person's group
            gmap = dict(zip(persons.tolist(), g.tolist()))
            ge = np.array([gmap.get(p, p) for p in extra_persons], dtype=object).astype(str)
        else:
            ge = extra_persons
        pe = np.full(len(me), np.nan)
        orphans = sorted(set(ge.tolist()) - set(g.tolist()))
        if orphans:
            warnings.warn(f'extra rows of persons without a fold are not scored: {orphans}', stacklevel=2)
    splits = loso_splits(g)
    assert_disjoint(splits, g, persons)
    rng_check = np.random.default_rng([seed, 1])
    if leak_check == 'all':
        to_check = {s.fold for s in splits}
    elif leak_check == 'sample':
        to_check = set(rng_check.choice(len(splits), size=min(C.LEAK_SAMPLE_FOLDS, len(splits)), replace=False).tolist())
    else:
        to_check = set()
    p = np.full(n, np.nan)
    fold_of = np.full(n, -1)
    infos: dict[int, dict[str, Any]] = {}
    checks: dict[int, str] = {}
    capped: dict[str, int] = {}
    for s in splits:
        tr = s.train[cap_rows(persons[s.train], yv[s.train], cap_per_person, seed)]
        for pid, k in pd.Series(persons[s.train]).value_counts().items():
            kept = int(np.sum(persons[tr] == pid))
            if kept < k:
                capped[str(pid)] = int(k - kept)
        ie = np.flatnonzero(ge == s.person) if ge is not None else np.zeros(0, dtype=int)
        X_test = _take(X, s.test)
        m_test = _test_meta(meta, s.test, test_meta_columns)
        if ie.size:
            X_test = _concat(X_test, _take(Xe, ie))
            m_test = pd.concat([m_test, _test_meta(me, ie, test_meta_columns)], ignore_index=True)
        n_t = s.test.size + ie.size
        perm = np.random.default_rng([seed, 2, s.fold]).permutation(n_t)  # random order: position says nothing
        X_test = _take(X_test, perm)
        m_test = m_test.iloc[perm].reset_index(drop=True)
        if 'row' in test_meta_columns:
            m_test.insert(0, 'row', np.arange(n_t))
        fold = Fold(s.fold, s.person, _take(X, tr), yv[tr].copy(), g[tr].copy(),
                    meta.iloc[tr].reset_index(drop=True).copy(), X_test, m_test, seed + s.fold)
        if set(persons[s.test].tolist()) & set(persons[tr].tolist()):
            raise LeakageError(f'fold {s.fold}: a test person in the training rows')
        with fit_predict_scope():
            ps, info = _as_proba(fit_predict(fold), n_t, s.fold)
        if s.fold in to_check:
            checks[s.fold] = _check_transductive(fit_predict, fold, ps, leak_atol, rng_check)
        pf = np.empty(n_t)
        pf[perm] = ps  # back to the original order: X_test[i] was row perm[i]
        p[s.test] = pf[:s.test.size]
        if ie.size:
            pe[ie] = pf[s.test.size:]
        fold_of[s.test] = s.fold
        infos[s.fold] = info
        if verbose:
            print(f'  fold {s.fold:2d} {s.person}: n={s.test.size:3d} '
                  f'acc={np.mean(credit(yv[s.test], pf[:s.test.size])):.3f}', flush=True)
    if not checks:
        leak = 'off'
    else:
        vals = list(checks.values())
        what = (f'{len(checks)}/{len(splits)} folds' if len(checks) < len(splits) else f'all {len(splits)} folds')
        if all(c.startswith(('passed', 'skipped')) for c in vals):
            leak = f'passed ({what})'
        elif any(c == 'nondeterministic' for c in vals):
            leak = f'nondeterministic ({sum(c == "nondeterministic" for c in vals)} of {what})'
        else:
            leak = ';'.join(vals)
    tkey = canon['trial_key'].to_numpy() if canon is not None else meta['trial_id'].astype(str).to_numpy()
    pred = pd.DataFrame({'trial_id': meta['trial_id'].to_numpy(), 'trial_key': tkey, 'person_id': persons, 'y': yv,
                         'p_right': p, 'fold': fold_of, 'model': model or name,
                         'params': [json.dumps(infos.get(int(f), {}), default=str, sort_keys=True) for f in fold_of]})
    m, ci, pp, rel, kc = evaluate_predictions(pred, n_boot=n_boot, seed=seed)
    gated = None
    if 'gate_pass' in meta.columns:
        gp = meta['gate_pass'].astype('boolean').fillna(False).astype(bool).to_numpy()
        gated = metrics(yv[gp], p[gp])
        gated['n_persons'] = float(len(set(persons[gp].tolist())))
        gated['n_failing_gates'] = float((~gp).sum())
    from .data import git_commit  # local import: data pulls in motionlab
    info = {'created_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'), 'scheme': scheme,
            'n_folds': len(splits), 'seed': seed, 'n_boot': n_boot, 'notes': notes, 'sets': list(sets),
            'n_configs_tried': int(n_configs_tried), 'cap_per_person': cap_per_person,
            'capped_training_rows_per_fold': capped, 'leak_check_mode': leak_check,
            'leak_checked_folds': sorted(int(k) for k in checks), 'leak_check_per_fold': {str(k): v for k, v in sorted(checks.items())},
            'test_meta_columns': list(test_meta_columns), 'locked_test': 'waived (researcher: leave-one-person-out only)',
            'cache_version': C.CACHE_VERSION, 'cache_code': _cache_code(), **git_commit(), 'code': C.code_fingerprint(),
            'scoring': SCORING}
    if spec is not None:
        if not isinstance(spec, dict):
            raise TypeError('spec must be a dict of JSON-able values')
        info['spec'] = _json_norm(spec)
    info['spec_sha256'] = spec_fingerprint(_declared_spec(model or name, params or {}, info))
    ex_pred, ex_m = None, None
    if extra is not None:
        ok = np.isfinite(pe)
        ex_pred = pd.DataFrame({'trial_id': me['trial_id'].to_numpy(), 'person_id': extra_persons, 'y': ye,
                                'p_right': pe, 'model': model or name})[ok].reset_index(drop=True)
        ex_m = metrics(ex_pred['y'], ex_pred['p_right']) if len(ex_pred) else {'n': 0.0}
        ex_m['n_persons'] = float(ex_pred['person_id'].nunique()) if len(ex_pred) else 0.0
    res = ExperimentResult(name, model or name, params or {}, pred, m, ci, pp, rel, kc, leak, info, ex_pred, ex_m, gated)
    if save:
        res.save(out_dir, registry)
    return res


def _cache_code() -> dict[str, Any] | None:
    """The code fingerprint recorded in the cache's meta.json (what built the arrays a lane used)."""
    p = C.CACHE_DIR / 'meta.json'
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return {'code': meta.get('code'), 'snapshot_sha256': (meta.get('snapshot_fingerprint') or {}).get('sha256'),
            'cache_created_at': meta.get('created_at'),
            'data_id': (meta.get('data_identity') or {}).get('data_id')}   # what stage2.pipeline checks results against


# ───────────────────────────── comparisons ─────────────────────────────


def _sign_flip_p(d: np.ndarray, *, exact_max: int = 20, n_perm: int = 100_000, seed: int = C.SEED) -> float:
    """Two-sided sign-flip permutation p-value of mean(d) = 0 (exact enumeration for <= exact_max persons).

    The 2^n sums are built by doubling (after step i: every sum of +-d[0..i]); each one adds +-d[0], +-d[1], ... in
    that order from 0, exactly as an enumeration of the sign codes does, so the floating-point sums and the p-value
    are those of the enumeration (tests/test_cv.py), at about 2^(n+1) additions instead of n 2^n."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return math.nan
    obs = abs(d.mean())
    if n <= exact_max:
        total = np.zeros(1)
        for x in d:
            total = np.concatenate([total + x, total - x])
        stats = np.abs(total / n)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        stats = np.abs(signs @ d / n)
    return float(np.mean(stats >= obs - 1e-12))


def paired_compare(a: ExperimentResult | pd.DataFrame, b: ExperimentResult | pd.DataFrame, *,
                   n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict[str, Any]:
    """A vs B on the same trials: per-person differences, paired person bootstrap and sign-flip test.

    Returns mean per-person accuracy and log-loss differences (A - B), their 95 % CIs from resampling
    persons (both arms on the same resampled persons), the pooled accuracy difference with its CI, the
    exact sign-flip permutation p-values and the per-person table. Accuracy uses :func:`credit` (a tie earns
    half credit).

    The bootstrap is vectorised on the random stream of the former per-resample loop: ``Generator.choice(persons,
    n, replace=True)`` draws ``Generator.integers(0, n, n)``, one call per resample, so every statistic equals the
    former loop's (tests/test_cv.py compares them on saved results) at a fraction of the cost."""
    pa = a.predictions if isinstance(a, ExperimentResult) else a
    pb = b.predictions if isinstance(b, ExperimentResult) else b
    j = pa[['trial_id', 'person_id', 'y', 'p_right']].merge(pb[['trial_id', 'p_right']], on='trial_id',
                                                             suffixes=('_a', '_b'))
    if len(j) != len(pa) or len(j) != len(pb):
        warnings.warn(f'paired_compare: {len(j)} shared trials (A {len(pa)}, B {len(pb)})', stacklevel=2)
    y = j['y'].to_numpy().astype(int)
    ca = credit(y, j['p_right_a'].to_numpy())
    cb = credit(y, j['p_right_b'].to_numpy())

    def ll(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, _EPS, 1 - _EPS)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    la, lb = ll(j['p_right_a'].to_numpy()), ll(j['p_right_b'].to_numpy())
    df = pd.DataFrame({'person_id': j['person_id'].astype(str), 'acc_a': ca, 'acc_b': cb, 'll_a': la, 'll_b': lb})
    pp = df.groupby('person_id').mean()
    pp['d_acc'] = pp['acc_a'] - pp['acc_b']
    pp['d_ll'] = pp['ll_a'] - pp['ll_b']
    persons = df['person_id'].to_numpy()
    uniq, inv = np.unique(persons, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = (np.stack([rng.integers(0, uniq.size, uniq.size) for _ in range(n_boot)]) if n_boot
           else np.zeros((0, uniq.size), dtype=np.int64))             # (n_boot, persons): the drawn persons
    # pooled: the trials of the drawn persons; the credits (0, 1/2, 1) sum exactly, so S / N is the former mean
    n_tr = np.bincount(inv, minlength=uniq.size).astype(float)[idx].sum(axis=1)
    boot_pool = (np.bincount(inv, weights=ca, minlength=uniq.size)[idx].sum(axis=1) / n_tr
                 - np.bincount(inv, weights=cb, minlength=uniq.size)[idx].sum(axis=1) / n_tr)
    # per person: the mean of the drawn persons' differences (the same pairwise sum as Series.mean of the draw)
    boot_person = pp.loc[uniq, 'd_acc'].to_numpy(dtype=float)[idx].mean(axis=1)
    boot_ll = pp.loc[uniq, 'd_ll'].to_numpy(dtype=float)[idx].mean(axis=1)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
    return {
        'n_trials': int(len(j)), 'n_persons': int(uniq.size),
        'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
        'pooled_acc_diff': float(ca.mean() - cb.mean()), 'pooled_acc_diff_ci95': q(boot_pool),
        'mean_person_acc_diff': float(pp['d_acc'].mean()), 'mean_person_acc_diff_ci95': q(boot_person),
        'p_sign_flip_acc': _sign_flip_p(pp['d_acc'].to_numpy(), seed=seed),
        'mean_person_logloss_diff': float(pp['d_ll'].mean()), 'mean_person_logloss_diff_ci95': q(boot_ll),
        'p_sign_flip_logloss': _sign_flip_p(pp['d_ll'].to_numpy(), seed=seed),
        'persons_a_better': int((pp['d_acc'] > 0).sum()), 'persons_b_better': int((pp['d_acc'] < 0).sum()),
        'per_person': pp.reset_index(),
    }


def load_result(name: str, out_dir: str | Path = C.RES_DIR) -> ExperimentResult:
    return ExperimentResult.load(Path(out_dir) / f'{name}.json')
