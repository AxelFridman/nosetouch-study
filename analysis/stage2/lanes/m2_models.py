"""Lane m2: classical models, tuned INSIDE fit_predict on ``fold.inner_splits()`` (nested), with mirror handling,
in-fold probability calibration and a per-fold fit cache.

``make_fit_predict(model, ...)`` returns a ``fit_predict(fold) -> (P(right), info)`` for ``cv.run_experiment``:

1. **Training rows only.** ``fold.X_train`` (features of ``m2_features``), optionally restricted to a random subset
   of the training PERSONS (learning curve; seeded by ``fold.seed``).
2. **Nested tuning.** Hyperparameters are chosen on GroupKFold over the training persons (``cv.inner_splits``):
   an exhaustive grid for one-parameter models (C of the linear models), Optuna TPE (seeded with ``fold.seed``)
   for the others. Objective: the mean inner-validation log loss. Inner validation rows are REAL trials only:
   mirror augmentation happens inside every fit (docs/09 D13).
3. **Preprocessing fitted on the training rows**: z-score (or robust) scaling for linear / SVM / k-NN models,
   optional winsorising of the scaled features at +-w SD; trees take the raw features.
4. **Mirror handling** (``mirror``): ``none``; ``aug`` (the mirrored training rows with flipped labels join the
   training set); ``anti`` (p(x) = 0.5 (f(x) + 1 - f(m(x))), exactly antisymmetric); ``both``.
5. **Probabilities.** SVMs and k-NN give scores, mapped to P(right) by a Platt sigmoid. Inside the tuning the map
   is CROSS-FITTED: the map applied to an inner validation fold is fitted on the other inner folds' scores, so the
   inner log loss (the Optuna objective and the nested-selection criterion) and the stored cross-fitted P(right)
   are out of sample. The final model's map is fitted on every inner cross-fitted score (person-grouped, never
   trial-level CV).
6. **Calibration** (``calibrator``): a post-hoc map (``m2_calibration``) fitted on the inner cross-fitted P(right)
   of the chosen configuration and applied to the held-out person (``platt_platform``: one map per platform).
   **Multi-attempt temperatures** tau_k (``m2_calibration.attempt_temperatures``) are fitted on the same inner
   cross-fitted P(right) of the training persons and stored per fold (``info['attempt_tau']``): nested.
7. **Cache.** The tuned model of a fold is cached by (configuration, hash of the training rows): the leak check of
   the harness (which re-runs fit_predict on half of the test rows) and the calibration variants re-use it. The
   fit never receives the test rows, so the cache cannot hide a test-set dependence; the prediction step runs
   anew on every call and remains checked.

Every estimator is seeded (``fold.seed``) and single-process or ``n_jobs <= N_JOBS``: the same fold gives the same
predictions (``tests/test_m2_models.py``).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stage2 import config as C
from stage2 import cv
from stage2.lanes.m2_calibration import fit_calibrator, logit, sigmoid

#: this lane's thread budget (task rule 7): at most 2, fewer when the process was started with fewer
#: (``--threads 1``: two single-thread workers beat one 2-thread process for LightGBM under CPU contention)
N_JOBS = max(1, min(2, int(os.environ.get('OMP_NUM_THREADS', '2') or 2)))
MIRRORS = ('none', 'aug', 'anti', 'both')
#: models whose solver can stop at its iteration cap (convergence is recorded for these only)
ITERATIVE_LINEAR = frozenset({'lr_l2', 'lr_l1', 'lr_en', 'svm_lin'})
LR_GRID = tuple(float(c) for c in np.logspace(-4, 0, 9))        # C of L2 logistic regression (half-decade steps)
L1_GRID = tuple(float(c) for c in np.logspace(-3, 1, 9))
SVM_LIN_GRID = tuple(float(c) for c in np.logspace(-5, -1, 9))


# ───────────────────────────── model definitions ─────────────────────────────


@dataclass(frozen=True)
class ModelDef:
    name: str
    kind: str                                   # 'proba' or 'score' (Platt-mapped)
    scale: bool                                 # z-score / robust scaling before the estimator
    build: Callable[[dict[str, Any], int, int], Any]
    grid: tuple[dict[str, Any], ...] = ()       # exhaustive grid (one-parameter models)
    space: Callable[[Any], dict[str, Any]] | None = None   # Optuna search space
    default_trials: int = 0
    label: str = ''


def _lr(params: dict[str, Any], seed: int, p: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    l1 = float(params.get('l1_ratio', 0.0))
    solver = 'lbfgs' if l1 == 0.0 else ('liblinear' if l1 == 1.0 else 'saga')
    kw: dict[str, Any] = dict(C=float(params['C']), l1_ratio=l1, solver=solver, random_state=seed)
    kw['max_iter'] = int(params['max_iter']) if params.get('max_iter') else (3000 if solver != 'saga' else 400)
    if solver == 'saga':
        kw['tol'] = 1e-3
    return LogisticRegression(**kw)


def _svm_lin(params: dict[str, Any], seed: int, p: int) -> Any:
    from sklearn.svm import LinearSVC
    return LinearSVC(C=float(params['C']), dual='auto', max_iter=20000, random_state=seed)


def _svm_rbf(params: dict[str, Any], seed: int, p: int) -> Any:
    from sklearn.svm import SVC
    return SVC(C=float(params['C']), gamma=float(params['gamma_mult']) / max(p, 1), kernel='rbf', random_state=seed)


def _knn(params: dict[str, Any], seed: int, p: int) -> Any:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    knn = KNeighborsClassifier(n_neighbors=int(params['n_neighbors']), weights=params['weights'], p=int(params['p']),
                               n_jobs=1)
    k = int(params.get('pca', 0))
    return make_pipeline(PCA(n_components=k, random_state=seed), knn) if k else knn


def _rf(params: dict[str, Any], seed: int, p: int, extra: bool = False) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    cls = ExtraTreesClassifier if extra else RandomForestClassifier
    depth = int(params['max_depth'])
    mf = params['max_features']
    return cls(n_estimators=int(params.get('n_estimators', 200)), max_features=mf if isinstance(mf, str) else float(mf),
               min_samples_leaf=int(params['min_samples_leaf']), max_depth=None if depth <= 0 else depth,
               n_jobs=N_JOBS, random_state=seed)


def _hgb(params: dict[str, Any], seed: int, p: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=float(params['learning_rate']), max_iter=int(params['max_iter']),
        max_leaf_nodes=int(params['max_leaf_nodes']), min_samples_leaf=int(params['min_samples_leaf']),
        l2_regularization=float(params['l2_regularization']), max_features=float(params['max_features']),
        early_stopping=False, random_state=seed)


def _lgbm(params: dict[str, Any], seed: int, p: int) -> Any:
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=int(params['n_estimators']), learning_rate=float(params['learning_rate']),
        num_leaves=int(params['num_leaves']), min_child_samples=int(params['min_child_samples']),
        colsample_bytree=float(params['colsample_bytree']), subsample=float(params['subsample']), subsample_freq=1,
        reg_lambda=float(params['reg_lambda']), n_jobs=N_JOBS, random_state=seed, deterministic=True,
        force_col_wise=True, verbose=-1)


def _space_en(t: Any) -> dict[str, Any]:
    return {'C': t.suggest_float('C', 1e-3, 10.0, log=True), 'l1_ratio': t.suggest_float('l1_ratio', 0.05, 0.95)}


def _space_svm_rbf(t: Any) -> dict[str, Any]:
    return {'C': t.suggest_float('C', 0.1, 100.0, log=True), 'gamma_mult': t.suggest_float('gamma_mult', 1e-3, 10.0, log=True)}


def _space_knn(t: Any) -> dict[str, Any]:
    return {'n_neighbors': t.suggest_int('n_neighbors', 3, 101, log=True),
            'weights': t.suggest_categorical('weights', ['uniform', 'distance']),
            'p': t.suggest_categorical('p', [1, 2]), 'pca': t.suggest_categorical('pca', [0, 8, 16, 32, 64])}


def _space_forest(t: Any) -> dict[str, Any]:
    return {'max_features': t.suggest_float('max_features', 0.01, 0.5, log=True),
            'min_samples_leaf': t.suggest_int('min_samples_leaf', 1, 30, log=True),
            'max_depth': t.suggest_categorical('max_depth', [0, 4, 8, 16])}


def _space_hgb(t: Any) -> dict[str, Any]:
    return {'learning_rate': t.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_iter': t.suggest_int('max_iter', 50, 200, log=True),
            'max_leaf_nodes': t.suggest_int('max_leaf_nodes', 4, 16, log=True),
            'min_samples_leaf': t.suggest_int('min_samples_leaf', 5, 60, log=True),
            'l2_regularization': t.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
            'max_features': t.suggest_float('max_features', 0.05, 0.5, log=True)}


def _space_lgbm(t: Any) -> dict[str, Any]:
    return {'n_estimators': t.suggest_int('n_estimators', 50, 200, log=True),
            'learning_rate': t.suggest_float('learning_rate', 0.02, 0.3, log=True),
            'num_leaves': t.suggest_int('num_leaves', 4, 16, log=True),
            'min_child_samples': t.suggest_int('min_child_samples', 5, 60, log=True),
            'colsample_bytree': t.suggest_float('colsample_bytree', 0.05, 0.5, log=True),
            'subsample': t.suggest_float('subsample', 0.5, 1.0),
            'reg_lambda': t.suggest_float('reg_lambda', 1e-3, 10.0, log=True)}


#: random forest / extra trees: ONE fixed configuration (the usual defaults: sqrt(p) features per split,
#: 300 trees, leaves of >= 3 rows); tuning them costs > 30 min per experiment within this lane's CPU budget
FOREST_DEFAULT: dict[str, Any] = {'n_estimators': 300, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'max_depth': 0}

MODELS: dict[str, ModelDef] = {m.name: m for m in (
    ModelDef('lr_l2', 'proba', True, _lr, grid=tuple({'C': c} for c in LR_GRID), label='logistic regression (L2)'),
    ModelDef('lr_l1', 'proba', True, lambda pr, s, p: _lr({**pr, 'l1_ratio': 1.0}, s, p),
             grid=tuple({'C': c} for c in L1_GRID), label='logistic regression (L1)'),
    ModelDef('lr_en', 'proba', True, _lr, space=_space_en, default_trials=12, label='logistic regression (elastic net)'),
    ModelDef('svm_lin', 'score', True, _svm_lin, grid=tuple({'C': c} for c in SVM_LIN_GRID), label='linear SVM + Platt'),
    ModelDef('svm_rbf', 'score', True, _svm_rbf, space=_space_svm_rbf, default_trials=25, label='RBF SVM + Platt'),
    ModelDef('knn', 'score', True, _knn, space=_space_knn, default_trials=25, label='k-NN + Platt'),
    ModelDef('rf', 'proba', False, _rf, grid=(FOREST_DEFAULT,), label='random forest (fixed hyperparameters)'),
    ModelDef('et', 'proba', False, lambda pr, s, p: _rf(pr, s, p, extra=True), grid=(FOREST_DEFAULT,),
             label='extra trees (fixed hyperparameters)'),
    ModelDef('hgb', 'proba', False, _hgb, space=_space_hgb, default_trials=6, label='HistGradientBoosting'),
    ModelDef('lgbm', 'proba', False, _lgbm, space=_space_lgbm, default_trials=12, label='LightGBM'),
)}


# ───────────────────────────── one fitted base model ─────────────────────────────


@dataclass(frozen=True)
class Prep:
    scaler: str = 'z'                 # 'z' | 'robust' | 'none' (only for models with scale=True)
    winsor: float | None = None       # clip the scaled features at +- winsor


@dataclass
class FittedBase:
    mdef: ModelDef
    scaler: Any
    winsor: float | None
    est: Any
    converged: bool | None = None     # iterative solvers: stopped before max_iter (None: not iterative)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X if self.scaler is None else self.scaler.transform(X)
        if self.winsor is not None:
            Z = np.clip(Z, -self.winsor, self.winsor)
        return Z

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X)
        if self.mdef.kind == 'proba':
            return self.est.predict_proba(Z)[:, 1]
        if self.mdef.name == 'knn':
            return self.est.predict_proba(Z)[:, 1] - 0.5
        return self.est.decision_function(Z)


def fit_base(mdef: ModelDef, params: dict[str, Any], X: np.ndarray, y: np.ndarray, seed: int, prep: Prep) -> FittedBase:
    scaler = None
    if mdef.scale and prep.scaler != 'none':
        from sklearn.preprocessing import RobustScaler, StandardScaler
        scaler = (StandardScaler() if prep.scaler == 'z' else RobustScaler()).fit(X)
    fb = FittedBase(mdef, scaler, prep.winsor if (mdef.scale and prep.scaler != 'none') else None, None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # convergence is recorded below (FittedBase.converged), not printed
        fb.est = mdef.build(params, seed, X.shape[1]).fit(fb.transform(X), y)
    # convergence is a property of the iterative linear solvers only (a boosting model always runs its max_iter)
    fb.converged = _converged(fb.est) if mdef.name in ITERATIVE_LINEAR else None
    return fb


def _converged(est: Any) -> bool | None:
    """An iterative solver stopped before its iteration cap (sklearn ``n_iter_`` < ``max_iter``); None otherwise."""
    n_iter, max_iter = getattr(est, 'n_iter_', None), getattr(est, 'max_iter', None)
    if n_iter is None or max_iter is None or isinstance(max_iter, bool):
        return None
    try:
        if int(max_iter) <= 0:                       # no cap (SVC: max_iter = -1)
            return None
        return bool(int(np.max(n_iter)) < int(max_iter))
    except (TypeError, ValueError):
        return None


def augment(X: np.ndarray, y: np.ndarray, mi: np.ndarray, ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training rows plus their mirror images with the other hand (docs/09 D13)."""
    return np.vstack([X, X[:, mi] * ms]), np.concatenate([y, 1 - y])


# ───────────────────────────── tuned model of one fold (cached) ─────────────────────────────


@dataclass
class Tuned:
    """What one fold's tuning leaves: the final model on every training row, its configuration, and the inner
    cross-fitted P(right) of the training rows with that configuration (for calibrators and stacking)."""
    mdef: ModelDef
    mirror: str
    final: FittedBase
    params: dict[str, Any]
    platt: tuple[float, float] | None
    oof_p: np.ndarray
    y: np.ndarray
    inner_logloss: float
    n_configs: int
    trace: list[float] = field(default_factory=list)
    n_inner_fits: int = 0             # every inner fit of the search (all configurations)
    n_inner_nonconverged: int = 0     # of which stopped at the iteration cap


_CACHE: OrderedDict[str, Tuned] = OrderedDict()
CACHE_SIZE = 200


def clear_cache() -> None:
    _CACHE.clear()


def evict(model: str) -> int:
    """Drop the cached fits of one model (after its last experiment), keeping memory bounded."""
    keys = [k for k, t in _CACHE.items() if t.mdef.name == model]
    for k in keys:
        del _CACHE[k]
    return len(keys)


def _key(*parts: Any, X: np.ndarray, y: np.ndarray, g: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(parts, default=str, sort_keys=True).encode())
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).astype(np.int8).tobytes())
    h.update('|'.join(map(str, g)).encode())
    return h.hexdigest()


def _platt_fit(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s, dtype=np.float64)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def _to_p(s: np.ndarray, platt: tuple[float, float] | None) -> np.ndarray:
    return s if platt is None else sigmoid(platt[0] * s + platt[1])


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _inner_eval(mdef: ModelDef, params: dict[str, Any], X: np.ndarray, y: np.ndarray, splits: list, seed: int,
                mirror: str, mi: np.ndarray, ms: np.ndarray, prep: Prep
                ) -> tuple[float, np.ndarray, tuple[float, float] | None, int, int]:
    """Mean inner-validation log loss of one configuration, its cross-fitted P(right), the Platt map of the final
    model (score models: fitted on every inner score), the number of inner fits and of non-converged ones.

    Score models: the Platt map applied to the rows of one inner validation fold is fitted on the OTHER inner
    folds' cross-fitted scores (cross-fitted calibration), so neither the loss nor the stored P(right) has seen the
    labels of the rows it scores."""
    n = len(y)
    s, sm = np.full(n, np.nan), np.full(n, np.nan)
    anti = mirror in ('anti', 'both')
    n_fits = n_nonconv = 0
    for tr, va in splits:
        Xt, yt = X[tr], y[tr]
        if mirror in ('aug', 'both'):
            Xt, yt = augment(Xt, yt, mi, ms)
        fb = fit_base(mdef, params, Xt, yt, seed, prep)
        n_fits += 1
        n_nonconv += int(fb.converged is False)
        s[va] = fb.score(X[va])
        if anti:
            sm[va] = fb.score(X[va][:, mi] * ms)
    if mdef.kind == 'score':
        p, pm = np.full(n, np.nan), np.full(n, np.nan)
        for _, va in splits:
            rest = np.ones(n, dtype=bool)
            rest[va] = False
            pl = _platt_fit(s[rest], y[rest])          # the other inner folds only
            p[va] = _to_p(s[va], pl)
            if anti:
                pm[va] = _to_p(sm[va], pl)
        platt = _platt_fit(s, y)                       # the final model's map: every inner score
    else:
        p, pm, platt = s, sm, None
    if anti:
        p = 0.5 * (p + 1.0 - pm)
    loss = float(np.mean([_logloss(y[va], p[va]) for _, va in splits]))
    return loss, p, platt, n_fits, n_nonconv


def tune(mdef: ModelDef, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, *, mirror: str = 'aug',
         mi: np.ndarray | None = None, ms: np.ndarray | None = None, prep: Prep = Prep(), n_trials: int | None = None,
         n_splits: int = C.INNER_SPLITS, max_iter: int | None = None) -> Tuned:
    """Nested tuning on GroupKFold over the training persons, then the final fit on every training row (cached).
    ``max_iter`` caps the iterations of the linear models' solvers (default: the solver's own cap)."""
    if mirror not in MIRRORS:
        raise ValueError(f'mirror {mirror!r}')
    if mirror != 'none' and (mi is None or ms is None):
        raise ValueError('mirror handling needs the feature mirror map')
    n_trials = int(n_trials or mdef.default_trials)
    key = _key(mdef.name, mirror, prep.scaler, prep.winsor, n_trials, n_splits, seed, max_iter, X=X, y=y, g=groups)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    over = {'max_iter': int(max_iter)} if max_iter else {}
    splits = cv.inner_splits(groups, n_splits)
    results: list[tuple[float, dict[str, Any], np.ndarray, Any]] = []
    fits = [0, 0]
    if len(mdef.grid) == 1 and mdef.kind == 'proba':
        # one pre-specified configuration: nothing to tune, so no inner fits (no cross-fitted predictions either:
        # a post-hoc calibrator cannot be fitted for such a model)
        results.append((math.nan, dict(mdef.grid[0]), np.full(len(y), np.nan), None))
    elif mdef.grid:
        for params in mdef.grid:
            loss, p, platt, nf, nn = _inner_eval(mdef, {**params, **over}, X, y, splits, seed, mirror, mi, ms, prep)
            fits[0] += nf
            fits[1] += nn
            results.append((loss, dict(params), p, platt))
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        store: dict[int, tuple[np.ndarray, Any]] = {}

        def objective(trial: Any) -> float:
            params = mdef.space(trial)
            loss, p, platt, nf, nn = _inner_eval(mdef, {**params, **over}, X, y, splits, seed, mirror, mi, ms, prep)
            fits[0] += nf
            fits[1] += nn
            store[trial.number] = (p, platt)
            return loss if math.isfinite(loss) else 1e9

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        for t in study.trials:
            if t.value is not None and t.number in store:
                results.append((float(t.value), dict(t.params), *store[t.number]))
    best = min(range(len(results)), key=lambda i: (results[i][0] if math.isfinite(results[i][0]) else math.inf, i))
    loss, params, oof_p, platt = results[best]
    Xf, yf = (augment(X, y, mi, ms) if mirror in ('aug', 'both') else (X, y))
    final = fit_base(mdef, {**params, **over}, Xf, yf, seed, prep)
    t = Tuned(mdef, mirror, final, params, platt, oof_p, y.copy(), loss, len(results), [r[0] for r in results],
              n_inner_fits=fits[0], n_inner_nonconverged=fits[1])
    _CACHE[key] = t
    while len(_CACHE) > CACHE_SIZE:
        _CACHE.popitem(last=False)
    return t


def predict(t: Tuned, X: np.ndarray, mi: np.ndarray | None, ms: np.ndarray | None) -> np.ndarray:
    """P(right) of rows X under a tuned model (row-wise: no statistic across the rows)."""
    p = _to_p(t.final.score(X), t.platt)
    if t.mirror in ('anti', 'both'):
        p = 0.5 * (p + 1.0 - _to_p(t.final.score(X[:, mi] * ms), t.platt))
    return np.clip(p, 0.0, 1.0)


# ───────────────────────────── contributions (importance on held-out rows) ─────────────────────────────


def contributions(t: Tuned, X: np.ndarray) -> np.ndarray | None:
    """Per-row, per-feature additive contributions to the model's logit on rows X (no label needed):
    linear models coef * scaled x; LightGBM ``pred_contrib`` (TreeSHAP). None for other models."""
    name = t.mdef.name
    if name.startswith('lr_'):
        Z = t.final.transform(X)
        return Z * t.final.est.coef_.ravel()[None, :]
    if name == 'lgbm':
        return np.asarray(t.final.est.predict(t.final.transform(X), pred_contrib=True))[:, :-1]
    return None


# ───────────────────────────── fit_predict factories ─────────────────────────────


def _subsample_persons(groups: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Row indices of n training persons chosen at random (seeded), or every row when n is None."""
    if n is None:
        return np.arange(len(groups))
    persons = np.array(sorted(set(groups.tolist())))
    if n >= persons.size:
        return np.arange(len(groups))
    pick = np.random.default_rng(seed).choice(persons, size=n, replace=False)
    return np.flatnonzero(np.isin(groups, pick))


def platforms(fold: Any, rows: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Platform of the training rows (optionally a subset) and of the test rows, for the platform-specific
    calibrator only (task rule 2: platform is a permitted input, known to production from the browser). Nothing
    else of ``meta_train`` / ``meta_test`` is read by this lane."""
    mt, mx = getattr(fold, 'meta_train', None), getattr(fold, 'meta_test', None)
    tr = None if mt is None or 'platform' not in mt else mt['platform'].astype(str).to_numpy()
    te = None if mx is None or 'platform' not in mx else mx['platform'].astype(str).to_numpy()
    if tr is not None and rows is not None:
        tr = tr[rows]
    return tr, te


def calibrate(calibrator: str, oof: np.ndarray, y: np.ndarray, p_test: np.ndarray, fold: Any,
              rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit ``calibrator`` on the training rows' cross-fitted P(right) and apply it to the test rows and (in sample,
    for the attempt temperatures only) to the cross-fitted P(right) itself: (p_test, oof, parameters)."""
    from stage2.lanes.m2_calibration import STRATIFIED
    s_tr, s_te = platforms(fold, rows) if calibrator in STRATIFIED else (None, None)
    cal, cal_params = fit_calibrator(calibrator, oof, y, s_tr)
    p = np.clip(cal(p_test, s_te), 0.0, 1.0)
    oof_c = np.clip(cal(oof, s_tr), 0.0, 1.0) if np.isfinite(oof).all() else oof
    return p, oof_c, cal_params


def attempt_tau(oof: np.ndarray, y: np.ndarray, g: np.ndarray, seed: int) -> dict[str, float]:
    """Nested multi-attempt temperatures tau_k from the training persons' cross-fitted P(right) (empty when a
    single pre-specified configuration left no cross-fitted predictions)."""
    from stage2.lanes.m2_calibration import attempt_temperatures
    return attempt_temperatures(oof, y, g, seed=seed)


def make_fit_predict(model: str = 'lr_l2', *, mirror: str = 'aug', scaler: str = 'z', winsor: float | None = None,
                     calibrator: str = 'none', n_trials: int | None = None, mirror_index: np.ndarray | None = None,
                     mirror_sign: np.ndarray | None = None, n_train_persons: int | None = None, subsample_rep: int = 0,
                     n_splits: int = C.INNER_SPLITS, store: dict[int, Any] | None = None,
                     max_iter: int | None = None) -> Callable[[Any], Any]:
    """fit_predict of one m2 pipeline (see the module docstring). ``store`` (a dict) receives, per fold, the
    held-out rows' mean |contribution| per feature (importance without labels). ``max_iter`` caps the solver
    iterations of the linear models (the unscaled-feature experiment); convergence is recorded in the info."""
    mdef = MODELS[model]
    prep = Prep(scaler, winsor)
    mi = None if mirror_index is None else np.asarray(mirror_index, dtype=int)
    ms = None if mirror_sign is None else np.asarray(mirror_sign, dtype=np.float64)

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        X = np.asarray(fold.X_train, dtype=np.float64)
        y = np.asarray(fold.y_train).astype(int)
        g = np.asarray(fold.groups_train).astype(str)
        rows = _subsample_persons(g, n_train_persons, fold.seed * 1000 + subsample_rep)
        X, y, g = X[rows], y[rows], g[rows]
        t = tune(mdef, X, y, g, fold.seed, mirror=mirror, mi=mi, ms=ms, prep=prep, n_trials=n_trials, n_splits=n_splits,
                 max_iter=max_iter)
        Xte = np.asarray(fold.X_test, dtype=np.float64)
        p, oof, cal_params = calibrate(calibrator, t.oof_p, t.y, predict(t, Xte, mi, ms), fold, rows)
        if store is not None and fold.fold not in store:      # first call of a fold = its full test set
            c = contributions(t, Xte)
            if c is not None:
                store[fold.fold] = {'abs_sum': np.abs(c).sum(0), 'n': int(c.shape[0])}
        info = {'model': model, 'mirror': mirror, 'scaler': scaler if mdef.scale else 'none', 'winsor': winsor,
                'calibrator': calibrator, 'calib_params': cal_params,
                'params': {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                           for k, v in t.params.items()},
                'inner_logloss': t.inner_logloss, 'n_configs': t.n_configs, 'n_train_rows': int(len(y)),
                'n_train_persons': int(len(set(g.tolist()))), 'attempt_tau': attempt_tau(oof, t.y, g, fold.seed),
                'n_inner_fits': t.n_inner_fits, 'n_inner_nonconverged': t.n_inner_nonconverged,
                'final_converged': t.final.converged}
        if max_iter:
            info['max_iter'] = int(max_iter)
        return p, info

    return fit_predict


def stack_fit(oofs: list[np.ndarray], y: np.ndarray, g: np.ndarray, n_splits: int) -> tuple[Any, np.ndarray]:
    """The stacker (logistic regression on the components' cross-fitted logits, no intercept: a mirror-symmetric
    stack of antisymmetric logits stays antisymmetric) and its own CROSS-FITTED P(right) on the training rows
    (the stacker of each inner validation fold fitted on the other folds), so the stack's training-side
    predictions used by a calibrator or the attempt temperatures are out of sample too."""
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([logit(o) for o in oofs])
    st = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000).fit(Z, y)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.inner_splits(g, n_splits):
        oof[va] = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
    return st, oof


def make_ensemble_fit_predict(components: tuple[str, ...] = ('lr_l2', 'svm_rbf', 'lgbm'), *, method: str = 'soft',
                              mirror: str = 'aug', mirror_index: np.ndarray | None = None,
                              mirror_sign: np.ndarray | None = None, n_trials: dict[str, int] | None = None,
                              calibrator: str = 'none', n_splits: int = C.INNER_SPLITS) -> Callable[[Any], Any]:
    """Soft voting (mean P(right)) or stacking (logistic regression on the components' inner cross-fitted logits)
    of components tuned in the fold exactly as ``make_fit_predict`` tunes them (so their fits are shared)."""
    mi = None if mirror_index is None else np.asarray(mirror_index, dtype=int)
    ms = None if mirror_sign is None else np.asarray(mirror_sign, dtype=np.float64)
    n_trials = dict(n_trials or {})

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        X = np.asarray(fold.X_train, dtype=np.float64)
        y = np.asarray(fold.y_train).astype(int)
        g = np.asarray(fold.groups_train).astype(str)
        Xte = np.asarray(fold.X_test, dtype=np.float64)
        tuned = [tune(MODELS[c], X, y, g, fold.seed, mirror=mirror, mi=mi, ms=ms, n_trials=n_trials.get(c),
                      n_splits=n_splits) for c in components]
        P = np.column_stack([predict(t, Xte, mi, ms) for t in tuned])
        info: dict[str, Any] = {'components': list(components), 'method': method,
                                'component_params': {c: t.params for c, t in zip(components, tuned)}}
        if method == 'soft':
            p = P.mean(1)
            oof = np.column_stack([t.oof_p for t in tuned]).mean(1)
        elif method == 'stack':
            st, oof = stack_fit([t.oof_p for t in tuned], y, g, n_splits)
            p = st.predict_proba(logit(P))[:, 1]
            info['stack_weights'] = st.coef_.ravel().tolist()
        else:
            raise ValueError(method)
        p, oof_c, cal_params = calibrate(calibrator, oof, y, p, fold)
        info.update({'calibrator': calibrator, 'calib_params': cal_params,
                     'attempt_tau': attempt_tau(oof_c, y, g, fold.seed)})
        return p, info

    return fit_predict
