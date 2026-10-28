"""Lane sec: models, the multi-class leave-one-person-out loop, metrics and person-level statistics.

Binary trial targets (the dominant hand) go through ``cv.run_experiment``. Multi-class targets (instructed speed,
instructed grip) cannot (the harness scores one P(right) per trial), so :func:`loso_multiclass` runs the SAME
leave-one-person-out splits (``cv.loso_splits``, checked with ``cv.assert_disjoint``), the same docs/09 D2 cap
(``cv.cap_rows``, hand-balanced, the same rows as every lane's training folds) and the same nested tuning splitter
(``cv.inner_splits`` over the training persons). The fit function never receives a test row (fit and predict are
separate calls), the cache is locked while it runs (``guard.fit_predict_scope``), and every fold is re-predicted on
a random half of its test rows in a new order (predictions must not move: the harness's transductive check).
Confidence intervals resample PERSONS with ``cv.person_bootstrap`` (2000 resamples, the same draws for every
statistic of one experiment).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.guard import LeakageError, fit_predict_scope

#: nested grids (tuned inside every outer fold on cv.inner_splits of the training persons)
LR_C_GRID: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
GBM_GRID: tuple[dict[str, Any], ...] = (
    {'num_leaves': 7, 'learning_rate': 0.05}, {'num_leaves': 7, 'learning_rate': 0.1},
    {'num_leaves': 15, 'learning_rate': 0.05}, {'num_leaves': 15, 'learning_rate': 0.1},
)
#: LightGBM settings fixed for every configuration: one thread (several one-thread workers beat one multi-thread
#: process on the shared machine, README section 11), 63 bins and half the features per tree (a fit takes about 1 s)
GBM_FIXED: dict[str, Any] = {'n_estimators': 100, 'max_bin': 63, 'min_child_samples': 10, 'reg_lambda': 1.0,
                             'colsample_bytree': 0.5, 'class_weight': 'balanced', 'n_jobs': 1, 'verbose': -1,
                             'deterministic': True, 'force_row_wise': True}
EPS = 1e-15


# ───────────────────────────── estimators ─────────────────────────────


def make_lr(c: float):
    """Standardise (fitted on the rows it is fitted on) + L2 logistic regression (multinomial for > 2 classes),
    class-balanced weights, Newton-Cholesky solver (exact, and about twice as fast as lbfgs here)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=5000,
                                                          solver='newton-cholesky'))


def make_gbm(params: dict[str, Any], seed: int):
    """LightGBM gradient boosting, class-balanced weights, deterministic (seeded feature subsampling)."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(**GBM_FIXED, **params, random_state=int(seed))


def proba_full(model: Any, X: np.ndarray, k: int) -> np.ndarray:
    """(n, k) class probabilities in class-index order 0..k-1 (a class the model never saw gets 0)."""
    P = np.zeros((X.shape[0], k))
    P[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X)
    return P


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-row weights n / (n_classes * n_class): every present class weighs the same."""
    y = np.asarray(y, dtype=int)
    cls, cnt = np.unique(y, return_counts=True)
    w = {c: y.size / (cls.size * n) for c, n in zip(cls, cnt)}
    return np.array([w[v] for v in y])


def weighted_log_loss(y: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.clip(P[np.arange(y.size), np.asarray(y, dtype=int)], EPS, 1.0)
    w = np.ones(y.size) if w is None else np.asarray(w, dtype=float)
    return float(-np.sum(w * np.log(p)) / np.sum(w))


def nested_fit(make: Callable[[Any], Any], grid: Sequence[Any], X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               k: int) -> tuple[Any, dict[str, Any]]:
    """Choose a configuration by the class-balanced log loss of the inner cross-fitted predictions (GroupKFold over
    the TRAINING persons, ``cv.inner_splits``), refit it on all the training rows. Returns (model, info)."""
    inner = cv.inner_splits(groups)
    w = balanced_weights(y)
    scores = []
    for cfg in grid:
        P = np.zeros((y.size, k))
        for tr, va in inner:
            P[va] = proba_full(make(cfg).fit(X[tr], y[tr]), X[va], k)
        scores.append(weighted_log_loss(y, P, w))
    best = int(np.argmin(scores))
    model = make(grid[best]).fit(X, y)
    return model, {'config': grid[best], 'inner_logloss': [round(float(s), 6) for s in scores],
                   'n_inner_folds': len(inner)}


def lr_fitter(k: int, grid: Sequence[float] = LR_C_GRID) -> Callable[..., tuple[Any, dict[str, Any]]]:
    def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
        return nested_fit(make_lr, grid, X, y, groups, k)
    return fit


def gbm_fitter(k: int, grid: Sequence[dict[str, Any]] = GBM_GRID) -> Callable[..., tuple[Any, dict[str, Any]]]:
    def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
        return nested_fit(lambda cfg: make_gbm(cfg, seed), grid, X, y, groups, k)
    return fit


def lr_coefficients(model: Any, names: Sequence[str]) -> dict[str, list[float]]:
    """Per class: the coefficients of the standardised features (log-odds per 1 SD of the training rows)."""
    lr = model[-1]
    return {str(int(c)): [float(v) for v in lr.coef_[i]] for i, c in enumerate(lr.classes_)} | {
        'features': list(names)}


# ───────────────────────────── multi-class LOSO ─────────────────────────────


def loso_multiclass(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, k: int, fit: Callable[..., Any], *,
                    feature_names: Sequence[str] = (), extra: tuple[np.ndarray, pd.DataFrame] | None = None,
                    seed: int = C.SEED, cap_per_person: int | None = C.CAP_PER_PERSON,
                    keep_coef: bool = False) -> dict[str, Any]:
    """Leave-one-person-out for a k-class target ``y`` (0..k-1) on feature rows ``X`` aligned with ``meta`` (the trial
    table: person_id, y_right for the cap). ``fit(X_train, y_train, groups_train, seed) -> (model, info)`` sees the
    training persons only. ``extra``: (X_extra, meta_extra) rows scored by their person's fold model, never trained
    on. Returns per-trial probabilities, folds, per-fold info and the leak checks."""
    groups = meta['person_id'].astype(str).to_numpy()
    yv = np.asarray(y, dtype=int)
    hand = meta['y_right'].astype(int).to_numpy()
    if X.shape[0] != yv.size or yv.size != len(meta):
        raise ValueError('X, y and meta differ in length')
    splits = cv.loso_splits(groups)
    cv.assert_disjoint(splits, groups, groups)
    P = np.full((yv.size, k), np.nan)
    fold_of = np.full(yv.size, -1)
    Pe, ge = None, None
    if extra is not None:
        Xe, me = extra
        ge = me['person_id'].astype(str).to_numpy()
        Pe = np.full((Xe.shape[0], k), np.nan)
    infos: dict[int, dict[str, Any]] = {}
    checks: dict[int, str] = {}
    for s in splits:
        tr = s.train[cv.cap_rows(groups[s.train], hand[s.train], cap_per_person, seed)]
        if set(groups[s.test].tolist()) & set(groups[tr].tolist()):
            raise LeakageError(f'fold {s.fold}: the test person is in the training rows')
        with fit_predict_scope():
            model, info = fit(X[tr].copy(), yv[tr].copy(), groups[tr].copy(), seed + s.fold)
            pt = proba_full(model, X[s.test], k)
            sub = np.random.default_rng([seed, 3, s.fold]).permutation(s.test.size)[:max(1, s.test.size // 2)]
            ps = proba_full(model, X[s.test][sub], k)
            if not np.allclose(ps, pt[sub], atol=1e-9, rtol=0):
                raise LeakageError(f'fold {s.fold}: predictions depend on the other test rows')
            checks[s.fold] = 'passed'
            if extra is not None:
                ie = np.flatnonzero(ge == s.person)
                if ie.size:
                    Pe[ie] = proba_full(model, extra[0][ie], k)
            if keep_coef:
                info['coef'] = lr_coefficients(model, feature_names)
        P[s.test] = pt
        fold_of[s.test] = s.fold
        infos[s.fold] = {**info, 'person': s.person, 'n_train': int(tr.size),
                         'n_capped': int(s.train.size - tr.size)}
    out = {'proba': P, 'fold': fold_of, 'fold_info': infos, 'leak_check': checks, 'n_folds': len(splits)}
    if extra is not None:
        out['extra_proba'] = Pe
    return out


# ───────────────────────────── metrics ─────────────────────────────


def confusion(y: np.ndarray, yhat: np.ndarray, k: int) -> np.ndarray:
    """(k, k) counts: rows = true class, columns = predicted class."""
    y = np.asarray(y, dtype=int)
    yhat = np.asarray(yhat, dtype=int)
    return np.bincount(y * k + yhat, minlength=k * k).reshape(k, k)


def balanced_accuracy(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    """Mean recall over the classes present in y (= sklearn.metrics.balanced_accuracy_score)."""
    cm = confusion(y, yhat, k)
    n = cm.sum(axis=1)
    ok = n > 0
    return float(np.mean(np.diag(cm)[ok] / n[ok])) if ok.any() else math.nan


def macro_f1(y: np.ndarray, yhat: np.ndarray, k: int) -> float:
    """Mean F1 over the classes present in y or in the predictions (= sklearn f1_score(average='macro'))."""
    cm = confusion(y, yhat, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return float(np.mean(f1[present])) if present.any() else math.nan


def recall(y: np.ndarray, yhat: np.ndarray, c: int) -> float:
    y = np.asarray(y, dtype=int)
    m = y == c
    return float(np.mean(np.asarray(yhat, dtype=int)[m] == c)) if m.any() else math.nan


def precision(y: np.ndarray, yhat: np.ndarray, c: int) -> float:
    yhat = np.asarray(yhat, dtype=int)
    m = yhat == c
    return float(np.mean(np.asarray(y, dtype=int)[m] == c)) if m.any() else math.nan


def multiclass_metrics(y: np.ndarray, P: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    """Accuracy, balanced accuracy, macro-F1, multinomial log loss, per-class recall / precision / F1."""
    k = len(classes)
    y = np.asarray(y, dtype=int)
    yhat = np.argmax(P, axis=1)
    out = {'n': float(y.size), 'accuracy': float(np.mean(yhat == y)), 'balanced_accuracy': balanced_accuracy(y, yhat, k),
           'macro_f1': macro_f1(y, yhat, k), 'log_loss': weighted_log_loss(y, P)}
    cm = confusion(y, yhat, k)
    for c, name in enumerate(classes):
        out[f'recall_{name}'] = recall(y, yhat, c)
        out[f'precision_{name}'] = precision(y, yhat, c)
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c]
        out[f'f1_{name}'] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else math.nan
        out[f'n_{name}'] = float(cm[c].sum())
    return out


def bootstrap_multiclass(person: np.ndarray, y: np.ndarray, P: np.ndarray, classes: Sequence[str], *,
                         n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict[str, tuple[float, float]]:
    """Person-cluster percentile 95 % CIs (``cv.person_bootstrap``: persons resampled with replacement, all their
    trials come along; the same seed gives every statistic the same resamples). The helper's (y, p_right) columns
    carry the class index and the predicted class (or the per-trial log loss for the log-loss CI)."""
    k = len(classes)
    yhat = np.argmax(P, axis=1)
    pred = pd.DataFrame({'person_id': person, 'y': np.asarray(y, dtype=int), 'p_right': yhat.astype(float)})
    stats: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        'accuracy': lambda a, b: float(np.mean(b.astype(int) == a)),
        'balanced_accuracy': lambda a, b: balanced_accuracy(a, b.astype(int), k),
        'macro_f1': lambda a, b: macro_f1(a, b.astype(int), k),
    }
    for c, name in enumerate(classes):
        stats[f'recall_{name}'] = (lambda cc: (lambda a, b: recall(a, b.astype(int), cc)))(c)
    out = {}
    for key, fn in stats.items():
        out[key] = cv.person_bootstrap(pred, fn, n_boot=n_boot, seed=seed)['stat']
    ll = -np.log(np.clip(P[np.arange(y.size), np.asarray(y, dtype=int)], EPS, 1.0))
    pred_ll = pd.DataFrame({'person_id': person, 'y': np.zeros(y.size, dtype=int), 'p_right': ll})
    out['log_loss'] = cv.person_bootstrap(pred_ll, lambda a, b: float(np.mean(b)), n_boot=n_boot, seed=seed)['stat']
    return out


def per_person_multiclass(person: np.ndarray, y: np.ndarray, P: np.ndarray, k: int) -> pd.DataFrame:
    yhat = np.argmax(P, axis=1)
    ll = -np.log(np.clip(P[np.arange(y.size), np.asarray(y, dtype=int)], EPS, 1.0))
    df = pd.DataFrame({'person_id': person, 'correct': (yhat == y).astype(float), 'log_loss': ll})
    g = df.groupby('person_id', sort=True)
    out = g.agg(n=('correct', 'size'), accuracy=('correct', 'mean'), log_loss=('log_loss', 'mean')).reset_index()
    ba = [balanced_accuracy(y[person == p], yhat[person == p], k) for p in out['person_id']]
    out['balanced_accuracy'] = ba
    return out


# ───────────────────────────── person-level statistics ─────────────────────────────


def boot_mean_ci(d: np.ndarray, *, n_boot: int = C.N_BOOT, seed: int = C.SEED,
                 stat: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[float, float]:
    """Percentile 95 % CI of a person-level statistic (default the mean) of per-person values ``d``, resampling the
    persons (the random stream of ``cv.paired_compare``: one ``integers(0, n, n)`` draw per resample)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.integers(0, d.size, d.size) for _ in range(n_boot)])
    vals = d[idx].mean(axis=1) if stat is None else stat(d[idx])
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p of mean(d) = 0 over persons (``cv._sign_flip_p``, the harness's test)."""
    return float(cv._sign_flip_p(np.asarray(d, dtype=float)))


def paired_multiclass(person: np.ndarray, y: np.ndarray, Pa: np.ndarray, Pb: np.ndarray, *, n_boot: int = C.N_BOOT,
                      seed: int = C.SEED) -> dict[str, Any]:
    """Model A vs model B on the same trials: per-person mean differences (A - B) of accuracy and of the multinomial
    log loss, their person-bootstrap 95 % CIs and exact sign-flip p-values (the design of ``cv.paired_compare``)."""
    y = np.asarray(y, dtype=int)
    ca = (np.argmax(Pa, axis=1) == y).astype(float)
    cb = (np.argmax(Pb, axis=1) == y).astype(float)
    la = -np.log(np.clip(Pa[np.arange(y.size), y], EPS, 1.0))
    lb = -np.log(np.clip(Pb[np.arange(y.size), y], EPS, 1.0))
    df = pd.DataFrame({'person_id': person, 'd_acc': ca - cb, 'd_ll': la - lb})
    pp = df.groupby('person_id', sort=True).mean()
    out: dict[str, Any] = {'n_persons': int(len(pp)), 'n_trials': int(y.size),
                           'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
                           'll_a': float(la.mean()), 'll_b': float(lb.mean()),
                           'n_changed': int(np.sum(np.argmax(Pa, axis=1) != np.argmax(Pb, axis=1)))}
    for key in ('d_acc', 'd_ll'):
        v = pp[key].to_numpy()
        out[key] = float(v.mean())
        out[f'{key}_ci'] = boot_mean_ci(v, n_boot=n_boot, seed=seed)
        out[f'{key}_p'] = sign_flip_p(v)
    out['persons_a_better_acc'] = int((pp['d_acc'] > 0).sum())
    out['persons_b_better_acc'] = int((pp['d_acc'] < 0).sum())
    return out


def holm(p: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.size, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not ok.size:
        return out.tolist()
    order = ok[np.argsort(p[ok], kind='mergesort')]
    m = ok.size
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out.tolist()


def within_person_auc(person: np.ndarray, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Per person: AUC of p for y (both classes needed), n per class."""
    rows = []
    for pid in sorted(set(np.asarray(person).tolist())):
        m = np.asarray(person) == pid
        yy = np.asarray(y)[m].astype(int)
        rows.append({'person_id': pid, 'n': int(m.sum()), 'n_pos': int(yy.sum()), 'n_neg': int((1 - yy).sum()),
                     'auc': cv.roc_auc(yy, np.asarray(p)[m])})
    return pd.DataFrame(rows)


def permute_within(person: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A label permutation inside each person (every person keeps its own class counts)."""
    out = np.asarray(y).copy()
    person = np.asarray(person)
    for pid in np.unique(person):
        idx = np.flatnonzero(person == pid)
        out[idx] = out[rng.permutation(idx)]
    return out


def loso_binary_scores(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, fit: Callable[..., Any], *,
                       seed: int = C.SEED, cap_per_person: int | None = C.CAP_PER_PERSON) -> np.ndarray:
    """Out-of-fold P(class 1) of a binary ``fit`` on the run_experiment design (the same LOSO splits, cap rows chosen
    with the binary label as the harness does, fold seeds seed + fold), without leak checks or bootstrap: the inner loop
    of the label-permutation baseline."""
    groups = meta['person_id'].astype(str).to_numpy()
    yv = np.asarray(y, dtype=int)
    p = np.full(yv.size, np.nan)
    for s in cv.loso_splits(groups):
        tr = s.train[cv.cap_rows(groups[s.train], yv[s.train], cap_per_person, seed)]
        with fit_predict_scope():
            model, _ = fit(X[tr], yv[tr], groups[tr], seed + s.fold)
            p[s.test] = proba_full(model, X[s.test], 2)[:, 1]
    return p
