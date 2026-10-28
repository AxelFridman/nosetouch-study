"""Lane m1 models: ``fit_predict`` factories of the interpretable models.

Every factory returns ``fit_predict(fold) -> (P(right), info)`` for :func:`stage2.cv.run_experiment`. ``X`` is a
feature DataFrame (``m1_features.build_x`` or ``build_tc``); a model reads its own columns. Rules of the lane:

* everything that spans trials (scalers, winsorising limits, thresholds, spline knots, calibrators, the
  aggregation exponent) is fitted on the TRAINING rows of the fold only;
* hyperparameters are chosen inside fit_predict on ``fold.inner_splits()`` (GroupKFold over the training persons),
  scored by the pooled inner out-of-fold log loss;
* calibration (``calibration=``) is fitted on the inner cross-fitted predictions of the chosen configuration, then
  applied to the held-out person (never fitted on in-sample predictions or on the test person);
* every fit is deterministic (lbfgs, seeded trees), so the harness's test-set-dependence check can run.

Mirror symmetry: odd (chiral) columns change sign under the left/right mirror, even columns do not
(``m1_features.MIRROR_SIGN``; every time-course column is odd). ``antisym=True`` fits a logistic regression with
no intercept on odd columns that are scaled but never centred, so p(mirror(x)) = 1 - p(x) holds exactly.
``augment=True`` adds the mirror image of every training row with the other label.

``store`` (optional dict) receives per-fold artefacts (coefficient profiles, tree structures, GAM shapes,
calibrator parameters) that the figures need; they depend on the training rows only.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier

from stage2.lanes import m1_features as _MF
from stage2.lanes.m1_features import MIRROR_SIGN, build_tc, build_x  # noqa: F401  (build_x / build_tc of finalists)

EPS = 1e-6
#: L2 grid (inverse strength), ascending, in half-decade steps (the 1-3 sequence): the decade grid of the first
#: run left the high-dimensional models at its lower edge (audit); ties go to the first, more regularised, value
CS_DEFAULT: tuple[float, ...] = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
GAM_KNOTS: tuple[int, ...] = (3, 5, 7, 9, 11)       # grids are wide enough for interior in-fold choices: the
#                                                       summary reports how often each grid's edge was chosen
GAM_CS: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
TREE_DEPTHS: tuple[int, ...] = (1, 2, 3)
TREE_LEAFS: tuple[int, ...] = (10, 20, 40, 80)
AGG_KS: tuple[int, ...] = (2, 3, 4, 5)


def build_x_an(d) -> pd.DataFrame:
    """The amplitude-normalised PHYS6 columns of a Stage2Data (input of ``m1_lr_phys6_ampnorm``)."""
    from stage2.lanes.m1_experiments import PHYS6
    return _MF.amp_normalised(build_x(d), PHYS6)


def build_x_treat(d) -> pd.DataFrame:
    """Every catalogue feature plus the amplitude-normalised PHYS6 columns (input of the nested treatment choice)."""
    from stage2.lanes.m1_experiments import PHYS6
    F = build_x(d)
    return pd.concat([F, _MF.amp_normalised(F, PHYS6)], axis=1)


def build_x_screen(d) -> pd.DataFrame:
    """``build_x_treat`` plus the inputs of the core D6 wrong-hand screen (input of the sensitivity runs)."""
    return pd.concat([build_x_treat(d), _MF.build_screen(d)], axis=1)


# ───────────────────────────── small helpers ─────────────────────────────


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def parity(cols: Sequence[str]) -> np.ndarray:
    """-1 for odd, +1 for even columns; time-course columns (``<channel>_tKK``) are odd."""
    out = []
    for c in cols:
        if c in MIRROR_SIGN.index:
            out.append(float(MIRROR_SIGN[c]))
        elif c.endswith('_an') and c[:-3] in MIRROR_SIGN.index:     # amplitude-normalised: odd / even keeps the parity
            out.append(float(MIRROR_SIGN[c[:-3]]))
        elif '_t' in c and c.rsplit('_t', 1)[1].isdigit():
            out.append(-1.0)
        else:
            raise KeyError(f'unknown column {c!r}: no mirror parity')
    return np.asarray(out)


def _cols(X: Any, cols: Sequence[str]) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X[list(cols)].to_numpy(dtype=float)
    raise TypeError('m1 models take a feature DataFrame')


def _lr(C: float, intercept: bool) -> LogisticRegression:
    return LogisticRegression(C=C, fit_intercept=intercept, max_iter=10000, tol=1e-6)


def _fit(model: Any, Z: np.ndarray, y: np.ndarray) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        return model.fit(Z, y)


def _augment(Z: np.ndarray, y: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([Z, Z * sign]), np.concatenate([y, 1 - y])


# ───────────────────────────── preprocessing fitted on the training rows ─────────────────────────────


class Prep:
    """Winsorising and scaling, fitted on training rows; odd columns are never centred (antisymmetry).

    winsor: 'none' | 'clip' (odd: +-q99 of |x|; even: q01..q99) | 'asinh' (odd: s asinh(x/s), s = median |x|;
    even: the same around the median). scale: 'rms' (odd: RMS, even: z-score) | 'zscore' (every column centred
    and scaled: unconstrained models) | 'robust' (odd: 1.4826 median |x|; even: median and IQR/1.349) |
    'none' | 'group_rms' (one RMS per column group: keeps a time course's coefficient profile in common units)."""

    def __init__(self, sign: np.ndarray, winsor: str = 'none', scale: str = 'rms',
                 groups: Sequence[str] | None = None) -> None:
        if winsor not in ('none', 'clip', 'asinh'):
            raise ValueError(winsor)
        if scale not in ('rms', 'zscore', 'robust', 'none', 'group_rms'):
            raise ValueError(scale)
        self.sign, self.winsor, self.scale = np.asarray(sign, dtype=float), winsor, scale
        self.groups = None if groups is None else np.asarray(groups)

    def fit(self, X: np.ndarray) -> 'Prep':
        odd = self.sign < 0
        X = np.asarray(X, dtype=float)
        self.lo = np.full(X.shape[1], -np.inf)
        self.hi = np.full(X.shape[1], np.inf)
        self.med = np.zeros(X.shape[1])
        self.s_w = np.ones(X.shape[1])
        if self.winsor == 'clip':
            a = np.quantile(np.abs(X), 0.99, axis=0)
            self.lo = np.where(odd, -a, np.quantile(X, 0.01, axis=0))
            self.hi = np.where(odd, a, np.quantile(X, 0.99, axis=0))
        elif self.winsor == 'asinh':
            self.med = np.where(odd, 0.0, np.median(X, axis=0))
            s = np.median(np.abs(X - self.med), axis=0)
            self.s_w = np.where(s > 0, s, 1.0)
        W = self._winsor(X)
        if self.scale == 'none':
            self.c, self.s = np.zeros(X.shape[1]), np.ones(X.shape[1])
        elif self.scale == 'zscore':
            self.c, self.s = W.mean(axis=0), W.std(axis=0)
        elif self.scale == 'rms':
            self.c = np.where(odd, 0.0, W.mean(axis=0))
            self.s = np.where(odd, np.sqrt(np.mean(W ** 2, axis=0)), W.std(axis=0))
        elif self.scale == 'robust':
            med = np.median(W, axis=0)
            iqr = (np.quantile(W, 0.75, axis=0) - np.quantile(W, 0.25, axis=0)) / 1.349
            self.c = np.where(odd, 0.0, med)
            self.s = np.where(odd, 1.4826 * np.median(np.abs(W), axis=0), iqr)
        else:  # group_rms
            if self.groups is None:
                raise ValueError('group_rms needs groups')
            self.c = np.where(odd, 0.0, W.mean(axis=0))
            self.s = np.ones(X.shape[1])
            for g in np.unique(self.groups):
                m = self.groups == g
                self.s[m] = np.sqrt(np.mean(W[:, m] ** 2))
        self.s = np.where(self.s > 0, self.s, 1.0)
        return self

    def _winsor(self, X: np.ndarray) -> np.ndarray:
        if self.winsor == 'clip':
            return np.clip(X, self.lo, self.hi)
        if self.winsor == 'asinh':
            return self.med + self.s_w * np.arcsinh((X - self.med) / self.s_w)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._winsor(np.asarray(X, dtype=float)) - self.c) / self.s


# ───────────────────────────── calibration (fitted on inner cross-fitted logits) ─────────────────────────────

CALIBRATIONS = ('none', 'platt', 'temperature', 'isotonic', 'beta')


def fit_calibrator(kind: str, z: np.ndarray, y: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """(apply(z) -> p, params) of a calibrator fitted on logits z with labels y (inner out-of-fold predictions).

    platt: p = sigmoid(a z + b); temperature: p = sigmoid(z / T) (keeps p(mirror) = 1 - p); isotonic: monotone
    step map of z, clipped to [1/(n+2), 1 - 1/(n+2)]; beta (Kull et al. 2017): p = sigmoid(a ln p0 - b ln(1 - p0) + c)
    with a, b >= 0 (a negative coefficient is dropped and the map refitted)."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y).astype(int)
    if kind == 'none':
        return sigmoid, {}
    if kind == 'temperature':
        def f(lt: float) -> float:
            return log_loss(y, sigmoid(z / math.exp(lt)))
        r = minimize_scalar(f, bounds=(-3.0, 3.0), method='bounded', options={'xatol': 1e-6})
        T = float(math.exp(r.x))
        return (lambda zz: sigmoid(np.asarray(zz) / T)), {'T': T}
    if kind == 'platt':
        m = _fit(LogisticRegression(C=1e6, max_iter=10000, tol=1e-10), z.reshape(-1, 1), y)
        a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
        return (lambda zz: sigmoid(a * np.asarray(zz) + b)), {'a': a, 'b': b}
    if kind == 'isotonic':
        n = z.size
        lo, hi = 1.0 / (n + 2), 1.0 - 1.0 / (n + 2)
        iso = IsotonicRegression(y_min=lo, y_max=hi, out_of_bounds='clip', increasing=True).fit(z, y)
        return (lambda zz: np.clip(iso.predict(np.asarray(zz, dtype=float)), lo, hi)), {'n_steps': int(len(iso.X_thresholds_))}
    if kind == 'beta':
        def feats(zz: np.ndarray) -> np.ndarray:
            zz = np.asarray(zz, dtype=float)
            return np.column_stack([-np.logaddexp(0, -zz), np.logaddexp(0, zz)])   # ln p, -ln(1 - p)
        use = [0, 1]
        for _ in range(2):
            m = _fit(LogisticRegression(C=1e6, max_iter=10000, tol=1e-10), feats(z)[:, use], y)
            if np.all(m.coef_[0] >= 0) or len(use) == 1:
                break
            use = [use[int(np.argmax(m.coef_[0]))]]
        coef = np.zeros(2)
        coef[use] = m.coef_[0]
        c = float(m.intercept_[0])
        return (lambda zz: sigmoid(feats(zz) @ coef + c)), {'a': float(coef[0]), 'b': float(coef[1]), 'c': c}
    raise ValueError(f'calibration {kind!r}: one of {CALIBRATIONS}')


def fit_agg_alpha(z: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int, ks: Sequence[int] = AGG_KS,
                  n_draws: int = 40, grid: np.ndarray | None = None) -> float:
    """Exponent alpha of the k-attempt aggregate z_k = k**(-alpha) * sum(z_i), fitted on inner out-of-fold logits of
    the TRAINING persons (random k-subsets of one person and one hand): alpha = 0 sums the logits (independent
    attempts), alpha = 1 averages them. Chosen by the log loss of the aggregates (grid 0, 0.05, ..., 1)."""
    grid = np.linspace(0.0, 1.0, 21) if grid is None else grid
    rng = np.random.default_rng(seed)
    S, K, Y = [], [], []
    df = pd.DataFrame({'g': np.asarray(groups).astype(str), 'y': np.asarray(y).astype(int), 'z': z})
    for (_, yy), g in df.groupby(['g', 'y'], sort=True):
        zz = g['z'].to_numpy()
        for k in ks:
            if zz.size < k:
                continue
            for _ in range(n_draws):
                S.append(zz[rng.choice(zz.size, size=k, replace=False)].sum())
                K.append(k)
                Y.append(yy)
    if not S:
        return 1.0
    S, K, Y = np.asarray(S), np.asarray(K, dtype=float), np.asarray(Y)
    losses = [log_loss(Y, sigmoid(S / K ** a)) for a in grid]
    return float(grid[int(np.argmin(losses))])


# ───────────────────────────── generic nested tuning ─────────────────────────────


def _inner_oof(fold: Any, Xtr: np.ndarray, ytr: np.ndarray, configs: Sequence[Any],
               fit_fn: Callable[[np.ndarray, np.ndarray, Any], Callable[[np.ndarray], np.ndarray]]
               ) -> tuple[dict[Any, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Inner out-of-fold logits of every config on fold.inner_splits() (GroupKFold over training persons)."""
    splits = fold.inner_splits()
    oof = {cfg: np.full(ytr.size, np.nan) for cfg in configs}
    for tr, va in splits:
        for cfg in configs:
            oof[cfg][va] = fit_fn(Xtr[tr], ytr[tr], cfg)(Xtr[va])
    return oof, splits


def _choose(oof: dict[Any, np.ndarray], y: np.ndarray) -> tuple[Any, dict[str, float]]:
    """The config with the lowest pooled inner log loss (ties: the first, i.e. the more regularised)."""
    scores = {cfg: log_loss(y, sigmoid(z)) for cfg, z in oof.items()}
    best = min(scores, key=lambda c: (round(scores[c], 12), list(scores).index(c)))
    return best, {str(k): float(v) for k, v in scores.items()}


def _finish(fold: Any, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, configs: Sequence[Any],
            fit_fn: Callable[[np.ndarray, np.ndarray, Any], Callable[[np.ndarray], np.ndarray]],
            calibration: str, fit_agg: bool) -> tuple[np.ndarray, dict[str, Any], Any, Callable[[np.ndarray], np.ndarray]]:
    """Nested choice of the config (when several), calibration on the inner cross-fitted logits, refit on every
    training row, P(right) of the test rows. Returns (p, info, chosen config, the refitted logit function)."""
    need_inner = len(configs) > 1 or calibration != 'none' or fit_agg
    info: dict[str, Any] = {}
    cfg = configs[0]
    z_oof = None
    if need_inner:
        oof, _ = _inner_oof(fold, Xtr, ytr, configs, fit_fn)
        cfg, _scores = _choose(oof, ytr)
        z_oof = oof[cfg]
        info['inner_logloss'] = round(log_loss(ytr, sigmoid(z_oof)), 6)
    zfun = fit_fn(Xtr, ytr, cfg)
    z_te = zfun(Xte)
    apply, cal = fit_calibrator(calibration, z_oof, ytr) if calibration != 'none' else (sigmoid, {})
    p = np.clip(apply(z_te), 0.0, 1.0)
    if cal:
        info['calibration'] = {k: round(v, 6) if isinstance(v, float) else v for k, v in cal.items()}
    if fit_agg and z_oof is not None:
        z_cal = logit(apply(z_oof))
        info['agg_alpha'] = fit_agg_alpha(z_cal, ytr, fold.groups_train, seed=fold.seed)
    return p, info, cfg, zfun


# ───────────────────────────── linear (logistic) models ─────────────────────────────


def _interaction_cols(Xtr: np.ndarray, X: np.ndarray, pairs: Sequence[tuple[int, int]], stats: list[tuple[float, float]] | None):
    """odd x standardised log(even) products (odd); stats (mean, sd) of log(even) from the training rows."""
    out, new = [], []
    for k, (io, ie) in enumerate(pairs):
        le_tr = np.log(np.maximum(Xtr[:, ie], 1e-9))
        mu, sd = (le_tr.mean(), le_tr.std() or 1.0) if stats is None else stats[k]
        new.append((mu, sd))
        out.append(X[:, io] * (np.log(np.maximum(X[:, ie], 1e-9)) - mu) / sd)
    return (np.column_stack(out) if out else np.zeros((X.shape[0], 0))), new


def _linear_parts(cols: Sequence[str], *, antisym: bool = True, augment: bool = False, winsor: str = 'none',
                  scale: str = 'rms', interactions: Sequence[tuple[str, str]] = (),
                  scale_groups: Sequence[str] | None = None) -> tuple[Callable, list[str]]:
    """(fit_fn(X, y, C) -> logit function, names of the fitted columns) of the logistic model of :func:`make_linear`
    on X = the columns ``cols`` (in that order)."""
    cols = list(cols)
    sign = parity(cols)
    ix_even = {e for _, e in interactions}
    bad = [c for c, s in zip(cols, sign) if s > 0 and c not in ix_even]
    if antisym and bad:
        raise ValueError(f'antisymmetric model with even columns outside interactions: {bad}')
    pairs = [(cols.index(o), cols.index(e)) for o, e in interactions]
    for o, e in interactions:
        if sign[cols.index(o)] > 0 or sign[cols.index(e)] < 0:
            raise ValueError(f'interaction ({o}, {e}) must be (odd, even)')
    keep = [i for i in range(len(cols)) if not (antisym and sign[i] > 0)]
    groups = None
    if scale == 'group_rms':
        groups = list(scale_groups) if scale_groups is not None else [c.rsplit('_t', 1)[0] for c in cols]
    names = [cols[i] for i in keep] + [f'{cols[o]}*log({cols[e]})' for o, e in pairs]

    def design(Xtr: np.ndarray, X: np.ndarray, stats=None):
        I, st = _interaction_cols(Xtr, X, pairs, stats)
        return np.column_stack([X[:, keep], I]), st

    def fit_fn(X: np.ndarray, y: np.ndarray, C: float):
        D, st = design(X, X)
        sg = np.concatenate([sign[keep], -np.ones(len(pairs))])
        if augment:
            D, y = _augment(D, y, sg)
        prep = Prep(sg, winsor, scale, None if groups is None else [groups[i] for i in keep] + ['ix'] * len(pairs)).fit(D)
        m = _fit(_lr(C, not antisym), prep.transform(D), y)

        def z(Xn: np.ndarray) -> np.ndarray:
            Dn, _ = design(X, Xn, st)
            return m.decision_function(prep.transform(Dn))
        z.model, z.prep = m, prep          # type: ignore[attr-defined]
        return z

    return fit_fn, names


def _linear_store(C: float, names: Sequence[str], m: LogisticRegression, prep: 'Prep', antisym: bool,
                  info: dict[str, Any]) -> dict[str, Any]:
    """Per-fold artefacts of a fitted logistic model: coefficients (per scaled unit and per raw unit) and every
    parameter of its preprocessing, so the logit contribution of each feature can be recomputed from saved files."""
    return {'C': C, 'names': list(names), 'coef_std': m.coef_[0].tolist(), 'scale': prep.s.tolist(),
            'centre': prep.c.tolist(), 'coef_raw': (m.coef_[0] / prep.s).tolist(),
            'intercept': float(m.intercept_[0]) if not antisym else 0.0, 'winsor': prep.winsor,
            'winsor_lo': [float(v) if np.isfinite(v) else None for v in prep.lo],
            'winsor_hi': [float(v) if np.isfinite(v) else None for v in prep.hi],
            'winsor_med': prep.med.tolist(), 'winsor_s': prep.s_w.tolist(),
            **({'calibration': info['calibration']} if 'calibration' in info else {})}


def stored_transform(entry: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """The scaled design of a stored linear model (:func:`_linear_store`) for raw columns X (its ``names`` order;
    no interaction terms): winsorising, centring and scaling exactly as fitted. coef_std @ row = the logit."""
    X = np.asarray(X, dtype=float)
    w = entry.get('winsor', 'none')
    if w == 'clip':
        lo = np.array([-np.inf if v is None else v for v in entry['winsor_lo']])
        hi = np.array([np.inf if v is None else v for v in entry['winsor_hi']])
        X = np.clip(X, lo, hi)
    elif w == 'asinh':
        med, s = np.asarray(entry['winsor_med']), np.asarray(entry['winsor_s'])
        X = med + s * np.arcsinh((X - med) / s)
    return (X - np.asarray(entry['centre'])) / np.asarray(entry['scale'])


def make_linear(cols: Sequence[str], *, antisym: bool = True, augment: bool = False, winsor: str = 'none',
                scale: str = 'rms', Cs: Sequence[float] = CS_DEFAULT, calibration: str = 'none',
                interactions: Sequence[tuple[str, str]] = (), fit_agg: bool = False,
                scale_groups: Sequence[str] | None = None, store: dict | None = None) -> Callable:
    """L2 logistic regression on ``cols``; C chosen on the inner splits (log loss).

    antisym: no intercept, odd columns only (scaled, never centred): p(mirror(x)) = 1 - p(x) exactly.
    augment: train on every row and its mirror image (other label); with odd columns only this is the
    antisymmetric model again (the intercept and every even weight vanish at the optimum).
    interactions: (odd, even) column pairs entering as odd * standardised log(even) (still odd)."""
    cols = list(cols)
    fit_fn, names = _linear_parts(cols, antisym=antisym, augment=augment, winsor=winsor, scale=scale,
                                  interactions=interactions, scale_groups=scale_groups)

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Xtr, Xte = _cols(fold.X_train, cols), _cols(fold.X_test, cols)
        p, info, C, zfun = _finish(fold, Xtr, fold.y_train, Xte, list(Cs), fit_fn, calibration, fit_agg)
        info['C'] = C
        m, prep = zfun.model, zfun.prep
        if not antisym:
            info['intercept'] = round(float(m.intercept_[0]), 6)
        if store is not None:
            store[fold.fold] = _linear_store(C, names, m, prep, antisym, info)
        return p, info

    return fit_predict


def make_linear_select(options: dict[str, dict[str, Any]], *, Cs: Sequence[float] = CS_DEFAULT,
                       calibration: str = 'none', fit_agg: bool = False, store: dict | None = None) -> Callable:
    """Logistic models of :func:`make_linear` whose OPTION (a feature set and / or a data treatment) is chosen INSIDE
    the fold, jointly with C: every (option, C) pair is scored by the pooled inner out-of-fold log loss on
    fold.inner_splits() (GroupKFold over the training persons) and the winner is refitted on every training row.
    ``options``: name -> make_linear keyword arguments (``cols`` and e.g. ``winsor``, ``scale``); ties go to the
    first option and the smaller C. The choice of the option is a hyperparameter, so the leave-one-person-out
    estimate of this model includes the selection (nested)."""
    names = list(options)
    if not names:
        raise ValueError('make_linear_select needs at least one option')
    parts = {k: _linear_parts(**{kk: vv for kk, vv in v.items()}) for k, v in options.items()}
    allcols = list(dict.fromkeys(c for v in options.values() for c in v['cols']))
    idx = {k: [allcols.index(c) for c in options[k]['cols']] for k in names}
    antisym = {k: bool(options[k].get('antisym', True)) for k in names}
    configs = [(k, float(C)) for k in names for C in Cs]

    def fit_fn(X: np.ndarray, y: np.ndarray, cfg: tuple[str, float]):
        k, C = cfg
        zf = parts[k][0](X[:, idx[k]], y, C)

        def z(Xn: np.ndarray) -> np.ndarray:
            return zf(Xn[:, idx[k]])
        z.model, z.prep = zf.model, zf.prep          # type: ignore[attr-defined]
        return z

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Xtr, Xte = _cols(fold.X_train, allcols), _cols(fold.X_test, allcols)
        p, info, cfg, zfun = _finish(fold, Xtr, fold.y_train, Xte, configs, fit_fn, calibration, fit_agg)
        k, C = cfg
        info.update({'option': k, 'C': C})
        if not antisym[k]:
            info['intercept'] = round(float(zfun.model.intercept_[0]), 6)
        if store is not None:
            store[fold.fold] = {'option': k, **_linear_store(C, parts[k][1], zfun.model, zfun.prep, antisym[k], info)}
        return p, info

    return fit_predict


def make_forward(cands: Sequence[str], *, max_k: int = 6, C_select: float = 1.0, Cs: Sequence[float] = CS_DEFAULT,
                 min_gain: float = 0.002, calibration: str = 'none', fit_agg: bool = False,
                 store: dict | None = None) -> Callable:
    """Antisymmetric logistic regression on 1..max_k odd features chosen INSIDE the fold by greedy forward selection
    (pooled inner out-of-fold log loss at C = C_select; stop when the gain is below ``min_gain``), then C tuned on the
    inner splits for the chosen set."""
    cands = list(cands)
    if np.any(parity(cands) > 0):
        raise ValueError('forward selection pool must hold odd columns only')

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Xall = _cols(fold.X_train, cands)
        y = fold.y_train
        splits = fold.inner_splits()

        def oof_ll(idx: list[int]) -> float:
            z = np.full(y.size, np.nan)
            for tr, va in splits:
                prep = Prep(-np.ones(len(idx)), 'none', 'rms').fit(Xall[tr][:, idx])
                m = _fit(_lr(C_select, False), prep.transform(Xall[tr][:, idx]), y[tr])
                z[va] = m.decision_function(prep.transform(Xall[va][:, idx]))
            return log_loss(y, sigmoid(z))

        chosen: list[int] = []
        best = log_loss(y, np.full(y.size, 0.5))
        path = []
        for _ in range(max_k):
            trial = {j: oof_ll(chosen + [j]) for j in range(len(cands)) if j not in chosen}
            j = min(trial, key=lambda k: (round(trial[k], 12), k))
            if best - trial[j] < min_gain:
                break
            chosen.append(j)
            best = trial[j]
            path.append(round(best, 6))
        if not chosen:   # nothing helps: fall back to the best single feature
            trial = {j: oof_ll([j]) for j in range(len(cands))}
            chosen = [min(trial, key=lambda k: (round(trial[k], 12), k))]
        sel = [cands[j] for j in chosen]
        fp = make_linear(sel, antisym=True, Cs=Cs, calibration=calibration, fit_agg=fit_agg, store=store)
        p, info = fp(fold)
        info['selected'] = sel
        info['selection_path_logloss'] = path
        return p, info

    return fit_predict


# ───────────────────────────── rules ─────────────────────────────


def _direction(x: np.ndarray, y: np.ndarray) -> float:
    """+1 when larger values mean right on the training rows (mean difference; ties +1)."""
    d = x[y == 1].mean() - x[y == 0].mean()
    return 1.0 if d >= 0 else -1.0


def make_sign_rule(col: str, *, direction: float | None = 1.0, calibrated: bool = False) -> Callable:
    """Right iff direction * x > 0. direction None: learned on the training persons (one bit). calibrated: P(right) is
    the training hit rate q of the rule (q when it says right, 1 - q otherwise: one parameter); else 1 or 0."""
    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        x = fold.X_train[col].to_numpy(dtype=float)
        y = fold.y_train
        s = _direction(x, y) if direction is None else float(direction)
        xt = fold.X_test[col].to_numpy(dtype=float)
        says = s * xt > 0
        info: dict[str, Any] = {'direction': s}
        if calibrated:
            q = float(np.mean((s * x > 0).astype(int) == y))
            info['q'] = round(q, 6)
            return np.where(says, q, 1 - q), info
        return says.astype(float), info
    return fit_predict


def make_direction(col: str) -> Callable:
    """The univariate model of the catalogue: P(right) = sigmoid(s * x / rms), with the direction s (sign of the
    right-minus-left mean difference) and the RMS of x fitted on the training persons and NO intercept. Its ranking is
    the feature itself with the training direction, so the pooled AUC is free of the leave-one-person-out intercept
    artefact (a per-fold intercept tracks the training class balance, which is anticorrelated with the held-out
    person's own balance and drags the pooled AUC of an uninformative feature below 0.5). For an odd feature the
    decision at 0.5 is the sign rule; for an even one the decision is meaningless (only the AUC is reported)."""
    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        x = fold.X_train[col].to_numpy(dtype=float)
        s = _direction(x, fold.y_train)
        rms = float(np.sqrt(np.mean(x ** 2))) or 1.0
        xt = fold.X_test[col].to_numpy(dtype=float)
        return sigmoid(s * xt / rms), {'direction': s}
    return fit_predict


def _best_threshold(v: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(threshold, training accuracy) of 'right iff v > threshold', maximising accuracy; ties: closest to 0."""
    o = np.argsort(v, kind='stable')
    vs, ys = v[o], y[o]
    n = v.size
    # predicting right for indices >= i: correct = lefts below i + rights from i
    left_below = np.concatenate([[0], np.cumsum(ys == 0)])
    right_from = np.concatenate([[0], np.cumsum(ys[::-1] == 1)])[::-1]
    acc = (left_below + right_from) / n
    cuts = np.concatenate([[vs[0] - 1.0], (vs[1:] + vs[:-1]) / 2.0, [vs[-1] + 1.0]])
    valid = np.concatenate([[True], vs[1:] > vs[:-1], [True]])
    acc = np.where(valid, acc, -1.0)
    best = acc.max()
    cand = np.flatnonzero(np.isclose(acc, best))
    i = cand[np.argmin(np.abs(cuts[cand]))]
    return float(cuts[i]), float(best)


def make_threshold_rule(col: str) -> Callable:
    """Right iff s * x > theta: direction s and threshold theta fitted on the training persons (max accuracy, ties
    closest to 0); P(right) on each side is the training share of right hands there (Laplace-smoothed)."""
    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        x = fold.X_train[col].to_numpy(dtype=float)
        y = fold.y_train
        s = _direction(x, y)
        th, acc = _best_threshold(s * x, y)
        hi = s * x > th
        p_hi = (y[hi].sum() + 1) / (hi.sum() + 2)
        p_lo = (y[~hi].sum() + 1) / ((~hi).sum() + 2)
        xt = fold.X_test[col].to_numpy(dtype=float)
        return np.where(s * xt > th, p_hi, p_lo), {'direction': s, 'threshold': round(th, 6),
                                                    'train_acc': round(acc, 6), 'p_hi': round(float(p_hi), 6),
                                                    'p_lo': round(float(p_lo), 6)}
    return fit_predict


def make_nested_rule(cands: Sequence[str]) -> Callable:
    """The best single-feature threshold rule, chosen INSIDE the fold: for every candidate, the inner-split accuracy
    of its threshold rule (direction and threshold fitted on the inner training persons); the winner is refitted on
    every training row."""
    cands = list(cands)

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        X = fold.X_train[cands].to_numpy(dtype=float)
        y = fold.y_train
        splits = fold.inner_splits()
        acc = np.zeros(len(cands))
        for j in range(len(cands)):
            ok = 0
            for tr, va in splits:
                s = _direction(X[tr, j], y[tr])
                th, _ = _best_threshold(s * X[tr, j], y[tr])
                ok += int(np.sum((s * X[va, j] > th).astype(int) == y[va]))
            acc[j] = ok / y.size
        j = int(np.argmax(acc))   # ties: the first candidate
        p, info = make_threshold_rule(cands[j])(fold)
        info.update({'feature': cands[j], 'inner_acc': round(float(acc[j]), 6)})
        return p, info

    return fit_predict


# ───────────────────────────── shallow trees ─────────────────────────────


def tree_structure(t: DecisionTreeClassifier, names: Sequence[str], leaf_p: np.ndarray) -> list[dict[str, Any]]:
    """Nodes of a fitted tree for drawing: id, depth, feature, threshold, left, right, n, p_right (smoothed)."""
    tr = t.tree_
    depth = np.zeros(tr.node_count, dtype=int)
    for i in range(tr.node_count):
        for c in (tr.children_left[i], tr.children_right[i]):
            if c >= 0:
                depth[c] = depth[i] + 1
    out = []
    for i in range(tr.node_count):
        leaf = tr.children_left[i] < 0
        out.append({'id': i, 'depth': int(depth[i]), 'leaf': bool(leaf),
                    'feature': None if leaf else names[int(tr.feature[i])],
                    'threshold': None if leaf else float(tr.threshold[i]),
                    'left': int(tr.children_left[i]), 'right': int(tr.children_right[i]),
                    'n': int(tr.n_node_samples[i]), 'p_right': float(leaf_p[i])})
    return out


def make_tree(cols: Sequence[str], *, depths: Sequence[int] = TREE_DEPTHS, leafs: Sequence[int] = TREE_LEAFS,
              augment: bool = True, symmetrise: bool = True, calibration: str = 'none',
              store: dict | None = None) -> Callable:
    """Decision tree (depth and minimum leaf size chosen on the inner splits by log loss), trained on the rows and
    their mirror images (augment); leaf probabilities Laplace-smoothed. symmetrise: P = (p(x) + 1 - p(mirror x)) / 2,
    so p(mirror(x)) = 1 - p(x) exactly."""
    cols = list(cols)
    sign = parity(cols)
    configs = list(itertools.product(depths, leafs))

    def fit_fn(X: np.ndarray, y: np.ndarray, cfg: tuple[int, int], seed: int = 0):
        Xa, ya = _augment(X, y, sign) if augment else (X, y)
        t = _fit(DecisionTreeClassifier(max_depth=cfg[0], min_samples_leaf=cfg[1], random_state=seed), Xa, ya)
        leaves = t.apply(Xa)
        n1 = np.bincount(leaves, weights=ya, minlength=t.tree_.node_count)
        n = np.bincount(leaves, minlength=t.tree_.node_count)
        leaf_p = (n1 + 1.0) / (n + 2.0)

        def z(Xn: np.ndarray) -> np.ndarray:
            p = leaf_p[t.apply(Xn)]
            if symmetrise:
                p = 0.5 * (p + 1.0 - leaf_p[t.apply(Xn * sign)])
            return logit(p)
        z.tree, z.leaf_p = t, leaf_p      # type: ignore[attr-defined]
        return z

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Xtr, Xte = _cols(fold.X_train, cols), _cols(fold.X_test, cols)
        fit_seeded = lambda X, y, cfg: fit_fn(X, y, cfg, fold.seed)  # noqa: E731
        p, info, cfg, zfun = _finish(fold, Xtr, fold.y_train, Xte, configs, fit_seeded, calibration, False)
        info.update({'depth': cfg[0], 'min_leaf': cfg[1]})
        if store is not None:
            store[fold.fold] = {'depth': cfg[0], 'min_leaf': cfg[1],
                                'nodes': tree_structure(zfun.tree, cols, _node_p(zfun.tree, zfun.leaf_p))}
        return p, info

    return fit_predict


def _node_p(t: DecisionTreeClassifier, leaf_p: np.ndarray) -> np.ndarray:
    """Smoothed P(right) of every node (internal nodes from the class counts the tree stores)."""
    v = t.tree_.value[:, 0, :]
    nn = t.tree_.n_node_samples.astype(float)
    frac1 = v[:, 1] / np.maximum(v.sum(axis=1), 1e-12)          # sklearn >= 1.4 stores fractions
    p = (frac1 * nn + 1.0) / (nn + 2.0)
    leaf = t.tree_.children_left < 0
    p[leaf] = leaf_p[leaf]
    return p


def fit_tree_full(X: pd.DataFrame, y: np.ndarray, cols: Sequence[str], depth: int, min_leaf: int, *, seed: int,
                  augment: bool = True) -> list[dict[str, Any]]:
    """The tree fitted on EVERY row with given hyperparameters (the deployable model, drawn in the report; not an
    evaluation)."""
    cols = list(cols)
    sign = parity(cols)
    Xa, ya = _augment(X[cols].to_numpy(dtype=float), np.asarray(y).astype(int), sign) if augment else \
        (X[cols].to_numpy(dtype=float), np.asarray(y).astype(int))
    t = _fit(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed), Xa, ya)
    leaves = t.apply(Xa)
    n1 = np.bincount(leaves, weights=ya, minlength=t.tree_.node_count)
    n = np.bincount(leaves, minlength=t.tree_.node_count)
    return tree_structure(t, cols, _node_p(t, (n1 + 1.0) / (n + 2.0)))


# ───────────────────────────── GAM (splines + L2 logistic) ─────────────────────────────


def make_gam(cols: Sequence[str], *, knots: Sequence[int] = GAM_KNOTS, Cs: Sequence[float] = GAM_CS,
             augment: bool = True, symmetrise: bool = True, calibration: str = 'none', fit_agg: bool = False,
             n_grid: int = 41, store: dict | None = None) -> Callable:
    """Additive logistic model: a cubic B-spline basis per feature (knots at training quantiles, linear
    extrapolation) and an L2 logistic regression; number of knots and C chosen jointly on the inner splits. With
    mirror augmentation the knots are symmetric and every odd feature's shape function is odd up to the solver's
    tolerance; ``symmetrise`` makes it exact: logit(x) = (z(x) - z(mirror x)) / 2, so p(mirror(x)) = 1 - p(x).
    ``store`` receives the shape functions f_j(x) of each fold on a grid (training 2-98 % range), centred at f_j(0)
    for odd features (their odd part when symmetrised)."""
    cols = list(cols)
    sign = parity(cols)
    configs = list(itertools.product(knots, Cs))

    def fit_fn(X: np.ndarray, y: np.ndarray, cfg: tuple[int, float]):
        Xa, ya = _augment(X, y, sign) if augment else (X, y)
        st = SplineTransformer(n_knots=cfg[0], degree=3, knots='quantile', extrapolation='linear').fit(Xa)
        B = st.transform(Xa)
        m = _fit(_lr(cfg[1], True), B, ya)

        def z(Xn: np.ndarray) -> np.ndarray:
            raw = m.decision_function(st.transform(Xn))
            return 0.5 * (raw - m.decision_function(st.transform(Xn * sign))) if symmetrise else raw
        z.st, z.m, z.Xa = st, m, Xa       # type: ignore[attr-defined]
        return z

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Xtr, Xte = _cols(fold.X_train, cols), _cols(fold.X_test, cols)
        p, info, cfg, zfun = _finish(fold, Xtr, fold.y_train, Xte, configs, fit_fn, calibration, fit_agg)
        info.update({'n_knots': cfg[0], 'C': cfg[1]})
        if store is not None:
            store[fold.fold] = {'n_knots': cfg[0], 'C': cfg[1],
                                'shapes': gam_shapes(zfun, cols, sign, n_grid, symmetrise=symmetrise)}
        return p, info

    return fit_predict


def gam_shapes(zfun: Any, cols: Sequence[str], sign: np.ndarray, n_grid: int = 41,
               symmetrise: bool = False) -> dict[str, dict[str, list[float]]]:
    """f_j on a grid for every feature of a fitted GAM (other features at 0 for odd / the median for even); with
    ``symmetrise`` the odd part (f_j(x) - f_j(-x)) / 2 of an odd feature (what the symmetrised model uses)."""
    st, m, Xa = zfun.st, zfun.m, zfun.Xa
    base = np.where(sign < 0, 0.0, np.median(Xa, axis=0))
    nb = st.transform(base.reshape(1, -1)).shape[1] // len(cols)
    out = {}

    def f_of(j: int, g: np.ndarray) -> np.ndarray:
        Z = np.repeat(base.reshape(1, -1), g.size, axis=0)
        Z[:, j] = g
        return st.transform(Z)[:, j * nb:(j + 1) * nb] @ m.coef_[0, j * nb:(j + 1) * nb]

    for j, c in enumerate(cols):
        lo, hi = np.quantile(Xa[:, j], [0.02, 0.98])
        g = np.linspace(lo, hi, n_grid)
        if symmetrise and sign[j] < 0:
            f = 0.5 * (f_of(j, g) - f_of(j, -g))
        else:
            f = f_of(j, g) - float(f_of(j, base[j:j + 1])[0])
        out[c] = {'x': g.tolist(), 'f': f.tolist()}
    return out


# ───────────────────────────── training-row filters (sensitivity analyses) ─────────────────────────────


def make_train_filter(factory: str, params: dict[str, Any], *, drop: str, blocks: Sequence[Sequence[Any]] = (),
                      screen_cols: Sequence[str] = ()) -> Callable:
    """The model ``factory(**params)`` of this module trained WITHOUT some training rows; the test rows (and so the
    evaluation) are unchanged, so the cleaning cannot inflate accuracy (docs/09 D6).

    drop='blocks': the (session_label, block_index) blocks listed (e.g. the swap-signature blocks of
    ``stage2.screen.block_consistency``, found per person from the first-half yaw sign);
    drop='d6': the docs/09 D6 screen fitted inside the fold on the TRAINING persons only
    (``stage2.screen.wrong_hand_flags`` on the columns ``screen_cols``: trials with p(label) < 0.05 and runs of at
    least 3 consecutive trials of a session that it confidently reverses)."""
    if drop not in ('blocks', 'd6'):
        raise ValueError("drop is 'blocks' or 'd6'")
    if factory == 'make_train_filter':
        raise ValueError('nested filters are not supported')
    base = globals()[factory](**params)
    keyset = {(str(s), int(b)) for s, b in blocks}

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if drop == 'blocks':
            bi = pd.to_numeric(fold.meta_train['block_index'], errors='coerce').fillna(-1).astype(int).to_numpy()
            sl = fold.meta_train['session_label'].astype(str).to_numpy()
            keep = ~np.array([(s, int(b)) in keyset for s, b in zip(sl, bi)], dtype=bool)
        else:
            from stage2 import screen
            F = fold.X_train[list(screen_cols)].to_numpy(dtype=float)
            fl = screen.wrong_hand_flags(F, fold.y_train, fold.groups_train, fold.meta_train)
            keep = ~fl['flag'].to_numpy(dtype=bool)
        ix = np.flatnonzero(keep)
        sub = dataclasses.replace(fold, X_train=fold.X_train.iloc[ix].reset_index(drop=True),
                                  y_train=np.asarray(fold.y_train)[ix], groups_train=np.asarray(fold.groups_train)[ix],
                                  meta_train=fold.meta_train.iloc[ix].reset_index(drop=True))
        p, info = base(sub)
        info = dict(info)
        info['n_train_dropped'] = int((~keep).sum())
        return p, info

    return fit_predict
