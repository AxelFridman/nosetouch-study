"""Lane p1: calibration of the panel models' P(right), remedies fitted strictly inside the training folds.

For an outer fold, :func:`inner_oof` cross-fits the panel model on the TRAINING persons only
(``fold.inner_splits()``: GroupKFold, every inner validation fold holds real trials of persons the inner model
never saw) and returns out-of-fold P(right) for every training trial. A calibrator is fitted on those and
applied to the held-out person's predictions of the model refitted on the whole training fold:

* ``platt``       P = sigmoid(a logit(p) + b)
* ``temperature`` P = sigmoid(logit(p) / T)   (no bias: keeps p(mirror) = 1 - p)
* ``isotonic``    monotone step function of p (clipped to [EPS, 1 - EPS])
* ``beta``        P = sigmoid(a ln p - b ln(1 - p) + c), a, b >= 0 (Kull et al. 2017)

``auto`` is the NESTED choice among ``none`` and the RANK-PRESERVING calibrators (temperature, Platt, beta: strictly
increasing maps of p, so the AUC of a fold is unchanged): within the outer fold, every candidate is cross-validated
on the training persons' out-of-fold predictions (GroupKFold over those persons: fitted on the predictions of some
persons, scored by log loss on the others), the lowest mean log loss wins (ties: the simpler candidate, in
:data:`AUTO_CANDIDATES` order), and the winner is refitted on all of them. No outer-fold result takes part in the
choice, so ``auto`` gives an honest calibrated headline; the fixed calibrators are compared on the outer folds only
as an exploratory study. Isotonic regression is NOT a candidate: its steps tie many predictions (an earlier
``auto`` that allowed it chose it in most LR folds and cost the LR about 2 AUC points, paired CI below 0), so it
stays only as a fixed, exploratory calibrator.

Multi-attempt aggregates: the fold also fits, on the same inner out-of-fold predictions, one temperature per k
(:data:`K_AGG`) for the rule "sum the logits of k attempts of one person and hand" (the Bayes rule for
independent attempts) - correlated attempts make that sum over-confident, and T_k > 1 measures by how much.
The T_k are returned in the fold's ``info`` and applied by the derive step to the held-out person's aggregates.

The inner cross-fitted predictions depend on the training rows only, so they are memoised per outer fold: the
harness's leak check and the other calibrators of the same model reuse them.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.lanes.p1_panel import Treatment, make_fit_predict

EPS = 1e-3
METHODS = ('platt', 'temperature', 'isotonic', 'beta')
AUTO = 'auto'                                          # the nested choice among AUTO_CANDIDATES
AUTO_CANDIDATES = ('none', 'temperature', 'platt', 'beta')   # rank-preserving only; simplest first (tie-break order)
RANK_PRESERVING = ('none', 'temperature', 'platt', 'beta')
AUTO_SPLITS = 5                                        # GroupKFold over the training persons' cross-fitted P
K_AGG = (2, 3, 4, 5)
N_DRAWS = 200
_MEMO: dict[str, np.ndarray] = {}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -35, 35)))


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """T > 0 minimising the log loss of sigmoid(z / T) (bounded scalar search on log T)."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: _ll(y, sigmoid(z / np.exp(lt))), bounds=(np.log(0.05), np.log(50.0)),
                        method='bounded', options={'xatol': 1e-6})
    return float(np.exp(r.x))


class Calibrator:
    """One of :data:`METHODS`, fitted on (p, y) of training trials; ``__call__(p)`` maps new probabilities."""

    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError(f'unknown calibration method {method!r}')
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> 'Calibrator':
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y).astype(int)
        z = logit(p)
        if self.method == 'platt':
            m = LogisticRegression(C=1e4, max_iter=2000).fit(z.reshape(-1, 1), y)
            self.params_ = {'a': float(m.coef_[0, 0]), 'b': float(m.intercept_[0])}
            if not self.params_['a'] > 0:             # never reverse or flatten the ranking: identity instead
                self.params_ = {'a': 1.0, 'b': 0.0, 'fallback': 1.0}
        elif self.method == 'temperature':
            self.params_ = {'T': fit_temperature(z, y)}
        elif self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.iso_ = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds='clip').fit(p, y)
            self.params_ = {'n_steps': int(len(self.iso_.X_thresholds_))}
        else:   # beta calibration: features ln p and -ln(1 - p), non-negative coefficients
            pc = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
            F = np.column_stack([np.log(pc), -np.log1p(-pc)])
            m = LogisticRegression(C=1e4, max_iter=2000).fit(F, y)
            a, b = m.coef_[0]
            if a < 0 or b < 0:        # refit without the feature whose coefficient went negative
                keep = 1 if a < 0 else 0
                m1 = LogisticRegression(C=1e4, max_iter=2000).fit(F[:, [keep]], y)
                coef = [0.0, 0.0]
                coef[keep] = max(float(m1.coef_[0, 0]), 0.0)
                a, b, c = coef[0], coef[1], float(m1.intercept_[0])
            else:
                c = float(m.intercept_[0])
            self.params_ = {'a': float(a), 'b': float(b), 'c': float(c)}
            if not float(a) + float(b) > 0:           # a constant map would tie every prediction: identity instead
                self.params_ = {'a': 1.0, 'b': 1.0, 'c': 0.0, 'fallback': 1.0}
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.method == 'platt':
            return sigmoid(self.params_['a'] * logit(p) + self.params_['b'])
        if self.method == 'temperature':
            return sigmoid(logit(p) / self.params_['T'])
        if self.method == 'isotonic':
            return np.clip(self.iso_.predict(p), EPS, 1 - EPS)
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        a, b, c = self.params_['a'], self.params_['b'], self.params_['c']
        return sigmoid(a * np.log(pc) - b * np.log1p(-pc) + c)


def _key(model: str, tr: Treatment, fold: Any) -> str:
    h = hashlib.sha256()
    h.update(f'{model}|{tr!r}|{fold.fold}|{fold.seed}'.encode())
    h.update(np.asarray(fold.y_train).tobytes())
    h.update('|'.join(map(str, fold.groups_train)).encode())
    X = fold.X_train if not isinstance(fold.X_train, dict) else fold.X_train[tr.test_key]
    h.update(np.ascontiguousarray(X).tobytes())
    return h.hexdigest()


def inner_oof(model: str, tr: Treatment, fold: Any) -> np.ndarray:
    """Out-of-fold P(right) of every training trial, cross-fitted over fold.inner_splits() (memoised)."""
    from stage2.cv import TEST_META_COLUMNS, _take
    key = _key(model, tr, fold)
    if key in _MEMO:
        return _MEMO[key]
    fp = make_fit_predict(model, tr)
    oof = np.full(len(fold.y_train), np.nan)
    for k, (itr, iva) in enumerate(fold.inner_splits()):
        mt = fold.meta_train.iloc[iva]
        keep = [c for c in TEST_META_COLUMNS if c in mt.columns and c != 'row']
        meta_te = mt[keep].reset_index(drop=True).copy()
        meta_te.insert(0, 'row', np.arange(len(iva)))
        sub = dataclasses.replace(
            fold, X_train=_take(fold.X_train, itr), y_train=fold.y_train[itr].copy(),
            groups_train=fold.groups_train[itr].copy(), meta_train=fold.meta_train.iloc[itr].reset_index(drop=True),
            X_test=_take(fold.X_train, iva), meta_test=meta_te, seed=int(fold.seed) * 10 + k + 1)
        p, _ = fp(sub)
        oof[iva] = p
    _MEMO[key] = oof
    return oof


def agg_samples(p: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int, rng: np.random.Generator,
                n_draws: int = N_DRAWS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sum of logits, y, person) of up to n_draws random k-subsets of every (person, hand) cell with >= k trials."""
    z = logit(p)
    S, Y, G = [], [], []
    df = pd.DataFrame({'g': groups, 'y': y})
    for (g, yy), idx in df.groupby(['g', 'y']).indices.items():
        if len(idx) < k:
            continue
        for _ in range(n_draws):
            pick = rng.choice(idx, size=k, replace=False)
            S.append(z[pick].sum())
            Y.append(int(yy))
            G.append(g)
    return np.asarray(S), np.asarray(Y), np.asarray(G)


def fit_k_temperatures(p: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, float]:
    """T_k of the summed-logit aggregate of k attempts, k in K_AGG, on training persons' out-of-fold P."""
    out = {}
    for k in K_AGG:
        s, yy, _ = agg_samples(p, y, groups, k, np.random.default_rng([seed, 29, k]))
        out[f'T_sum_k{k}'] = fit_temperature(s, yy) if len(set(yy.tolist())) == 2 else float('nan')
    return out


def select_calibrator(p: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      candidates: tuple[str, ...] = AUTO_CANDIDATES, n_splits: int = AUTO_SPLITS
                      ) -> tuple[str, dict[str, float]]:
    """The nested choice: each candidate is fitted on the cross-fitted P of some training persons and scored by
    log loss on the others (GroupKFold over the persons); returns (winner, mean log loss per candidate). Uses
    the training persons' predictions only."""
    from stage2.cv import inner_splits
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    splits = inner_splits(np.asarray(groups), n_splits)
    losses: dict[str, float] = {}
    for meth in candidates:
        q = np.array(p, copy=True)
        if meth != 'none':
            for itr, iva in splits:
                if len(set(y[itr].tolist())) < 2:
                    continue
                q[iva] = Calibrator(meth).fit(p[itr], y[itr])(p[iva])
        losses[meth] = _ll(y, q)
    best = min(candidates, key=lambda m: (round(losses[m], 9), candidates.index(m)))
    return best, losses


def make_calibrated(model: str, method: str, tr: Treatment = Treatment()):
    """fit_predict(fold) -> (calibrated P(right), info): calibrator and aggregate temperatures fitted on the
    inner cross-fitted predictions of the training persons; the base model refitted on the whole fold.
    ``method='auto'``: the calibrator itself is chosen inside the fold (:func:`select_calibrator`)."""
    if method != AUTO and method not in METHODS:
        raise ValueError(f'unknown calibration method {method!r}')
    base = make_fit_predict(model, tr)

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        oof = inner_oof(model, tr, fold)
        extra: dict[str, Any] = {}
        meth = method
        if method == AUTO:
            meth, losses = select_calibrator(oof, fold.y_train, fold.groups_train)
            extra = {'chosen': meth, **{f'cv_logloss_{k}': round(v, 6) for k, v in losses.items()}}
        p, info = base(fold)
        temps = fit_k_temperatures(oof, fold.y_train, fold.groups_train, int(fold.seed))
        if meth == 'none':
            q, params = np.asarray(p, dtype=np.float64), {}
        else:
            cal = Calibrator(meth).fit(oof, fold.y_train)
            q, params = cal(p), cal.params_
        return q, {**info, 'calibrator': method, **extra, **{k: round(v, 6) for k, v in params.items()},
                   'inner_oof_logloss': round(_ll(fold.y_train, oof), 6), **{k: round(v, 6) for k, v in temps.items()}}

    return fit_predict


__all__ = ['Calibrator', 'METHODS', 'AUTO', 'AUTO_CANDIDATES', 'K_AGG', 'make_calibrated', 'select_calibrator',
           'inner_oof', 'agg_samples', 'fit_temperature', 'logit', 'sigmoid', 'C']
