"""Evaluation that never leaks a participant: subject-wise CV, cross-orientation and cross-platform
tests, metrics, subgroup error tables with participant-bootstrap confidence intervals, and the
mirror-consistency check (docs/05 section 6).

Schemes (``make_splits``):
  'stratified_group_kfold'  StratifiedGroupKFold over participant_id (5 folds, shuffled with ``seed``),
                 stratified on a participant stratum (default ehi_class x platform) - the development CV
  'group_kfold'  GroupKFold over participant_id (default 5 folds, fewer if there are fewer participants)
  'loso'         leave-one-subject-out
  'orientation'  leave-one-ORIENTATION-out, crossed with subject folds: test = (orientation o, subjects of
                 fold k), train = (other orientations, other subjects). Mirror pairs (top_left_* with
                 top_right_*) are held out together, because mirroring one creates the other. Measures
                 generalisation to an unseen way of holding the phone AND an unseen person.
  'platform'     leave-one-PLATFORM-out: test = one platform, train = the other platforms minus every
                 participant who also appears in the test platform.
Every split is checked with :func:`assert_subject_disjoint` before any model is fitted.

Labels: y = 1 for the right hand, 0 for the left; p = predicted p(right).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold

from .conventions import mirror_pair
from .models import p_right

SUBGROUP_COLUMNS: tuple[str, ...] = (
    'sex', 'handedness_self', 'ehi_class', 'age_band', 'orientation', 'speed', 'start_position', 'variant',
    'platform', 'hand', 'hand_is_dominant',
)
SCHEMES: tuple[str, ...] = ('stratified_group_kfold', 'group_kfold', 'loso', 'orientation', 'platform')


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    label: str


class LeakageError(AssertionError):
    """A participant appears in both the training and the test part of a split."""


def assert_subject_disjoint(splits: Sequence[Split], groups: Any) -> None:
    g = np.asarray(groups)
    for s in splits:
        both = set(g[s.train]) & set(g[s.test])
        if both:
            raise LeakageError(f'split {s.label}: participants in train AND test: {sorted(both)[:5]}')


def _subject_folds(g: np.ndarray, n_splits: int, strata: Any, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subject folds: StratifiedGroupKFold on ``strata`` when given (and usable), else GroupKFold."""
    idx = np.arange(g.size)
    n_groups = len(set(g.tolist()))
    k = min(n_splits, n_groups)
    if strata is not None:
        s = np.asarray(pd.Series(strata).astype(str))
        if len(set(s.tolist())) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)  # small strata: sklearn warns, still subject-disjoint
                cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                folds = list(cv.split(idx, s, groups=g))
            if all(te.size for _, te in folds):
                return folds
    return list(GroupKFold(n_splits=k).split(idx, groups=g))


def make_splits(
    scheme: str,
    groups: Any,
    *,
    orientations: Any = None,
    platforms: Any = None,
    strata: Any = None,
    n_splits: int = 5,
    seed: int = 0,
    pair_mirrors: bool = True,
) -> list[Split]:
    """Subject-disjoint (train, test) index splits; see the module docstring for the schemes.

    groups: participant id per trial. strata: participant stratum per trial (e.g. 'right|ios') for the
    stratified schemes; orientations / platforms: per trial, for the 'orientation' / 'platform' schemes.
    """
    g = np.asarray(groups)
    idx = np.arange(g.size)
    n_groups = len(set(g.tolist()))
    if n_groups < 2:
        raise ValueError(f'subject-wise evaluation needs >= 2 participants, got {n_groups}')
    out: list[Split] = []
    if scheme == 'stratified_group_kfold':
        for i, (tr, te) in enumerate(_subject_folds(g, n_splits, strata, seed)):
            out.append(Split(tr, te, f'fold{i}'))
    elif scheme == 'group_kfold':
        for i, (tr, te) in enumerate(_subject_folds(g, n_splits, None, seed)):
            out.append(Split(tr, te, f'fold{i}'))
    elif scheme == 'loso':
        for tr, te in LeaveOneGroupOut().split(idx, groups=g):
            out.append(Split(tr, te, f'loso:{g[te[0]]}'))
    elif scheme == 'orientation':
        if orientations is None:
            raise ValueError("scheme 'orientation' needs orientations")
        o = np.asarray([mirror_pair(x) if (pair_mirrors and isinstance(x, str) and x) else x for x in orientations],
                       dtype=object)
        folds = _subject_folds(g, n_splits, strata, seed)
        for oi in sorted({str(x) for x in o.tolist()}):
            for i, (_, te_g) in enumerate(folds):
                in_test_g = np.isin(g, np.unique(g[te_g]))
                is_o = np.asarray([str(x) == oi for x in o])
                te = idx[is_o & in_test_g]
                tr = idx[~is_o & ~in_test_g]
                if te.size and tr.size:
                    out.append(Split(tr, te, f'{oi}|fold{i}'))
    elif scheme == 'platform':
        if platforms is None:
            raise ValueError("scheme 'platform' needs platforms")
        pl = np.asarray([str(x) for x in platforms], dtype=object)
        for p in sorted(set(pl.tolist())):
            te = idx[pl == p]
            test_groups = np.unique(g[te])
            tr = idx[(pl != p) & ~np.isin(g, test_groups)]
            if te.size and tr.size:
                out.append(Split(tr, te, f'platform:{p}'))
    else:
        raise ValueError(f'unknown scheme {scheme!r}; use one of {SCHEMES}')
    assert_subject_disjoint(out, g)
    return out


# ───────────────────────────── metrics ─────────────────────────────


def binary_metrics(y_true: Any, p: Any, threshold: float = 0.5) -> dict[str, float]:
    """accuracy, balanced accuracy, ROC AUC, log loss, Brier (NaN when undefined, e.g. one class)."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out: dict[str, float] = {'n': float(y.size), 'n_right': float(y.sum())}
    if y.size == 0:
        return {**out, 'accuracy': math.nan, 'balanced_accuracy': math.nan, 'roc_auc': math.nan,
                'log_loss': math.nan, 'brier': math.nan}
    pred = (p >= threshold).astype(int)
    out['accuracy'] = float((pred == y).mean())
    both = 0 < y.sum() < y.size
    out['balanced_accuracy'] = float(balanced_accuracy_score(y, pred)) if both else math.nan
    out['roc_auc'] = float(roc_auc_score(y, p)) if both else math.nan
    out['log_loss'] = float(log_loss(y, p, labels=[0, 1]))
    out['brier'] = float(brier_score_loss(y, p, pos_label=1))
    return out


def participant_bootstrap(
    df: pd.DataFrame, fn: Callable[[pd.DataFrame], float], *, group_col: str = 'participant_id',
    n_boot: int = 1000, seed: int = 0, ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI of ``fn`` over bootstrap resamples of PARTICIPANTS (clusters), not of trials."""
    groups = df[group_col].to_numpy()
    uniq = np.unique(groups)
    if uniq.size < 2:
        return math.nan, math.nan
    rows = {u: np.flatnonzero(groups == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([rows[u] for u in pick])
        v = fn(df.iloc[sel])
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return math.nan, math.nan
    a = (1 - ci) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


# ───────────────────────────── cross-validation ─────────────────────────────


@dataclass
class CVResult:
    name: str
    scheme: str
    oof: pd.DataFrame  # index trial_id: fold, participant_id, y_true, p_right, pred[, p_right_mirror]
    folds: pd.DataFrame
    metrics: dict[str, float]
    metrics_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    mirror: dict[str, float] | None = None


def _fit(factory: Callable[[], Any], X: pd.DataFrame, y: np.ndarray) -> Any:
    model = factory()
    model = clone(model) if hasattr(model, 'get_params') else model
    return model.fit(X, y)


def cross_validate(
    factory: Callable[[], Any],
    X: pd.DataFrame,
    y: Any,
    groups: Any,
    *,
    scheme: str = 'stratified_group_kfold',
    n_splits: int = 5,
    orientations: Any = None,
    platforms: Any = None,
    strata: Any = None,
    X_mirror: pd.DataFrame | None = None,
    augment_mirror: bool = False,
    name: str = 'model',
    n_boot: int = 0,
    seed: int = 0,
    splits: Sequence[Split] | None = None,
) -> CVResult:
    """Out-of-fold predictions for every test trial of a subject-disjoint scheme.

    X: features (index = trial_id); y: 1 = right; groups: participant ids (aligned with X).
    X_mirror: features of the mirrored trials (same index) -> out-of-fold p(mirror(x)) for the mirror-
    consistency check. augment_mirror: also train on the mirrored TRAINING trials with flipped labels
    (the copies stay in their participant's training fold; test trials are always real).
    In the 'orientation' scheme a trial is tested once per subject fold of its orientation (once in total).
    ``splits`` overrides the scheme (they are still checked for subject leakage).
    """
    yv = np.asarray(y, dtype=int)
    g = np.asarray(groups)
    if X_mirror is not None:
        X_mirror = X_mirror.reindex(X.index)
    if splits is None:
        splits = make_splits(scheme, g, orientations=orientations, platforms=platforms, strata=strata,
                             n_splits=n_splits, seed=seed)
    else:
        assert_subject_disjoint(splits, g)
    p = np.full(yv.size, np.nan)
    pm = np.full(yv.size, np.nan)
    fold = np.empty(yv.size, dtype=object)
    frows = []
    for s in splits:
        ytr = yv[s.train]
        if np.unique(ytr).size < 2:
            continue
        Xtr = X.iloc[s.train]
        if augment_mirror and X_mirror is not None:
            Xtr = pd.concat([Xtr, X_mirror.iloc[s.train]])
            ytr = np.concatenate([ytr, 1 - ytr])
        model = _fit(factory, Xtr, ytr)
        p[s.test] = p_right(model, X.iloc[s.test])
        if X_mirror is not None:
            pm[s.test] = p_right(model, X_mirror.iloc[s.test])
        fold[s.test] = s.label
        m = binary_metrics(yv[s.test], p[s.test])
        frows.append({'fold': s.label, 'n_train': int(s.train.size), 'n_test': int(s.test.size),
                      'n_test_participants': int(len(set(g[s.test].tolist()))), **m})
    done = np.isfinite(p)
    oof = pd.DataFrame({'fold': fold, 'participant_id': g, 'y_true': yv, 'p_right': p}, index=X.index)
    oof['pred'] = np.where(np.isfinite(p), (p >= 0.5).astype(int), -1)
    if X_mirror is not None:
        oof['p_right_mirror'] = pm
    oof = oof[done]
    metrics = binary_metrics(oof['y_true'], oof['p_right'])
    metrics['n_participants'] = float(oof['participant_id'].nunique())
    metrics.update(participant_accuracy_summary(oof))
    ci = {}
    if n_boot and len(oof):
        ci['accuracy'] = participant_bootstrap(
            oof, lambda d: float(((d['p_right'] >= 0.5).astype(int) == d['y_true']).mean()), n_boot=n_boot, seed=seed)

        def auc(d: pd.DataFrame) -> float:
            yy = d['y_true'].to_numpy()
            return float(roc_auc_score(yy, d['p_right'])) if 0 < yy.sum() < yy.size else math.nan

        ci['roc_auc'] = participant_bootstrap(oof, auc, n_boot=n_boot, seed=seed + 1)
    mir = mirror_consistency(oof['p_right'], oof['p_right_mirror']) if X_mirror is not None and len(oof) else None
    return CVResult(name, scheme, oof, pd.DataFrame(frows), metrics, ci, mir)


def participant_accuracy(oof: pd.DataFrame) -> pd.DataFrame:
    """Per-participant accuracy of out-of-fold predictions (columns n, accuracy)."""
    correct = ((oof['p_right'] >= 0.5).astype(int) == oof['y_true']).astype(float)
    return correct.groupby(oof['participant_id']).agg(['size', 'mean']).rename(columns={'size': 'n', 'mean': 'accuracy'})


def participant_accuracy_summary(oof: pd.DataFrame) -> dict[str, float]:
    """Median and 10th percentile of per-participant accuracy, and how many participants are below 80 %."""
    if not len(oof):
        return {'participant_acc_median': math.nan, 'participant_acc_p10': math.nan, 'n_participants_below_80': math.nan}
    acc = participant_accuracy(oof)['accuracy'].to_numpy()
    return {'participant_acc_median': float(np.median(acc)), 'participant_acc_p10': float(np.percentile(acc, 10)),
            'n_participants_below_80': float((acc < 0.8).sum())}


def mirror_consistency(p: Any, p_mirror: Any) -> dict[str, float]:
    """How well p(mirror(x)) = 1 - p(x) holds (it must, if the model learned chirality, not confounds).

    mean_abs_dev = mean |p_mirror - (1 - p)|; label_flip_rate = fraction of trials whose predicted label
    flips under mirroring (1.0 ideal); corr = correlation of p_mirror with 1 - p.
    """
    a = np.asarray(p, dtype=float)
    b = np.asarray(p_mirror, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {'n': 0.0, 'mean_abs_dev': math.nan, 'max_abs_dev': math.nan, 'label_flip_rate': math.nan, 'corr': math.nan}
    dev = np.abs(b - (1 - a))
    corr = float(np.corrcoef(b, 1 - a)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else math.nan
    return {
        'n': float(a.size), 'mean_abs_dev': float(dev.mean()), 'max_abs_dev': float(dev.max()),
        'label_flip_rate': float(((a >= 0.5) != (b >= 0.5)).mean()), 'corr': corr,
    }


# ───────────────────────────── subgroups ─────────────────────────────


def subgroup_table(
    oof: pd.DataFrame,
    meta: pd.DataFrame,
    by: Sequence[str] = SUBGROUP_COLUMNS,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Error rate per level of each subgroup variable, with participant-bootstrap CIs.

    oof: CVResult.oof (index trial_id, y_true, p_right, participant_id). meta: trials table (has trial_id
    and the subgroup columns). For each level: n_trials, n_participants, accuracy, error and its CI from
    resampling the participants of that level with replacement (their trials move together).
    """
    m = meta.set_index('trial_id') if 'trial_id' in meta.columns else meta
    df = oof.join(m.drop(columns=[c for c in ('participant_id', 'y_true') if c in m.columns]), how='left')
    df['error'] = ((df['p_right'] >= 0.5).astype(int) != df['y_true']).astype(float)
    df['p_correct'] = np.where(df['y_true'] == 1, df['p_right'], 1 - df['p_right'])
    rng = np.random.default_rng(seed)
    rows = []
    a = (1 - ci) / 2
    for var in by:
        if var not in df.columns:
            continue
        col = df[var].astype(object).where(df[var].notna(), 'unknown')
        for level in sorted(col.unique(), key=lambda x: str(x)):
            sub = df[col == level]
            per = sub.groupby('participant_id')['error'].agg(['sum', 'count'])
            E, N = per['sum'].to_numpy(), per['count'].to_numpy()
            err = float(E.sum() / N.sum()) if N.sum() else math.nan
            lo = hi = math.nan
            if n_boot and per.shape[0] >= 2:
                pick = rng.integers(0, per.shape[0], size=(n_boot, per.shape[0]))
                boot = E[pick].sum(axis=1) / N[pick].sum(axis=1)
                lo, hi = float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
            rows.append({'variable': var, 'level': str(level), 'n_trials': int(N.sum()), 'n_participants': int(per.shape[0]),
                         'accuracy': 1 - err if math.isfinite(err) else math.nan, 'error': err,
                         'error_lo': lo, 'error_hi': hi, 'mean_p_correct': float(sub['p_correct'].mean())})
    return pd.DataFrame(rows, columns=['variable', 'level', 'n_trials', 'n_participants', 'accuracy', 'error',
                                       'error_lo', 'error_hi', 'mean_p_correct'])


# ───────────────────────────── analysis set ─────────────────────────────


def _b(df: pd.DataFrame, col: str, default: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return df[col].astype('boolean').fillna(default).astype(bool)


#: Trial-level exclusion rules of the primary analysis (docs/02 section 12, docs/05 section 1).
PRIMARY_RULES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    'T1 status complete': lambda d: d['status'] == 'complete',
    'T2 accepted': lambda d: _b(d, 'accepted', False),
    'T3 no participant flag': lambda d: d['participant_flag'].isna() if 'participant_flag' in d.columns else pd.Series(True, index=d.index),
    'T4 page not hidden': lambda d: ~_b(d, 'hidden_during', False),
    'T5 motion_hz >= 50': lambda d: pd.to_numeric(d['motion_hz'], errors='coerce').fillna(0) >= 50,
    'T6 no gaps > 50 ms': lambda d: pd.to_numeric(d['gaps_over_50ms'], errors='coerce').fillna(0) == 0,
    'T7 rotation rate available': lambda d: _b(d, 'rotation_available', False),
}


def analysis_set(
    trials: pd.DataFrame,
    *,
    rules: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
    simulated: str = 'include',
    require_calibration: bool = False,
    sessions: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """(mask over ``trials`` rows, table of how many attempts each rule removes) for the analysis set.

    simulated: 'exclude' applies S1 (real-data analyses), 'include' keeps simulated trials (pipeline
    tests), 'only' keeps only them. require_calibration: also drop sessions without a calibration.
    The table counts, per rule, the attempts failing it and the attempts still in after it (sequential).
    """
    rules = dict(PRIMARY_RULES if rules is None else rules)
    if simulated == 'exclude':
        rules = {'S1 not simulated': lambda d: ~_b(d, 'simulated', False), **rules}
    elif simulated == 'only':
        rules = {'S1 simulated only': lambda d: _b(d, 'simulated', False), **rules}
    if require_calibration and sessions is not None and 'calibration_present' in sessions.columns:
        cal = dict(zip(sessions['session_id'], sessions['calibration_present'].astype('boolean').fillna(False).astype(bool)))
        rules['S3 calibration present'] = lambda d: d['session_id'].map(cal).fillna(False).astype(bool)
    keep = pd.Series(True, index=trials.index)
    rows = []
    for name, rule in rules.items():
        ok = rule(trials).reindex(trials.index).fillna(False).astype(bool)
        rows.append({'rule': name, 'failing': int((~ok).sum()), 'remaining': int((keep & ok).sum())})
        keep &= ok
    return keep, pd.DataFrame(rows, columns=['rule', 'failing', 'remaining'])


def shuffle_labels_within(y: Any, groups: Any, seed: int = 0) -> np.ndarray:
    """Labels permuted WITHIN each participant (leakage control: a subject-wise CV must give ~50 %)."""
    yv = np.asarray(y).copy()
    g = np.asarray(groups)
    rng = np.random.default_rng(seed)
    for u in np.unique(g):
        idx = np.flatnonzero(g == u)
        yv[idx] = yv[rng.permutation(idx)]
    return yv


# ───────────────────────────── convenience ─────────────────────────────


def modeling_table(features: pd.DataFrame, trials: pd.DataFrame, *, accepted_only: bool = True,
                   exclude_flags: Sequence[str] = ('wrong_hand', 'wrong_orientation'),
                   require_gravity_sign: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(X, meta) aligned by trial_id: usable features (qc_ok) of labelled, accepted trials.

    Trials the participant flagged as performed with the wrong hand / orientation are excluded by default
    (their label is wrong by the participant's own report). ``require_gravity_sign`` drops sessions whose
    gravity sign is only the +1 default (nothing supports it, so every chiral feature is a coin flip).
    """
    meta = trials.set_index('trial_id')
    keep = meta.index.intersection(features.index)
    meta = meta.loc[keep]
    m = meta['y_right'].notna()
    if accepted_only:
        m &= meta['accepted'].fillna(False).astype(bool)
    if exclude_flags and 'participant_flag' in meta.columns:
        m &= ~meta['participant_flag'].isin(list(exclude_flags))
    if 'qc_ok' in features.columns:
        m &= features.loc[keep, 'qc_ok'].fillna(0).astype(int) == 1
    if require_gravity_sign and 'qc_gravity_sign_source' in features.columns:
        m &= features.loc[keep, 'qc_gravity_sign_source'] != 'default'
    ids = meta.index[m.fillna(False).astype(bool)]
    return features.loc[ids], meta.loc[ids].reset_index()


def participant_strata(meta: pd.DataFrame, columns: Sequence[str] = ('ehi_class', 'platform')) -> np.ndarray:
    """Per-trial participant stratum string (e.g. 'right|ios'), the stratification variable of the CV."""
    parts = [meta[c].astype(object).where(meta[c].notna(), 'unknown').astype(str) for c in columns if c in meta.columns]
    if not parts:
        return np.asarray(['all'] * len(meta), dtype=object)
    s = parts[0]
    for p in parts[1:]:
        s = s + '|' + p
    return s.to_numpy(dtype=object)


def evaluate_models(
    factories: dict[str, Callable[[], Any]],
    X: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    scheme: str = 'stratified_group_kfold',
    n_splits: int = 5,
    X_mirror: pd.DataFrame | None = None,
    augment_mirror: bool = False,
    n_boot: int = 0,
    seed: int = 0,
    strata_columns: Sequence[str] = ('ehi_class', 'platform'),
) -> dict[str, CVResult]:
    """Run :func:`cross_validate` for several model factories on the SAME splits.

    meta: rows aligned with X (``modeling_table``), with y_right, participant_id and, for the schemes that
    need them, orientation / platform and the strata columns.
    """
    y = meta['y_right'].astype(int).to_numpy()
    g = meta['participant_id'].to_numpy()
    o = meta['orientation'].to_numpy() if 'orientation' in meta.columns else None
    pl = meta['platform'].astype(object).fillna('unknown').to_numpy() if 'platform' in meta.columns else None
    strata = participant_strata(meta, strata_columns)
    splits = make_splits(scheme, g, orientations=o, platforms=pl, strata=strata, n_splits=n_splits, seed=seed)
    out = {}
    for name, fac in factories.items():
        out[name] = cross_validate(fac, X, y, g, scheme=scheme, splits=splits, X_mirror=X_mirror,
                                   augment_mirror=augment_mirror, name=name, n_boot=n_boot, seed=seed)
    return out


def summary_table(results: dict[str, CVResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {'model': name, 'scheme': r.scheme, **{k: r.metrics.get(k) for k in (
            'n', 'n_participants', 'accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier',
            'participant_acc_median', 'participant_acc_p10', 'n_participants_below_80')}}
        for k, (lo, hi) in r.metrics_ci.items():
            row[f'{k}_lo'], row[f'{k}_hi'] = lo, hi
        if r.mirror:
            row['mirror_mean_abs_dev'] = r.mirror['mean_abs_dev']
            row['mirror_label_flip_rate'] = r.mirror['label_flip_rate']
        rows.append(row)
    return pd.DataFrame(rows)
