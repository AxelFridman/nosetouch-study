"""Baseline left/right classifiers (scikit-learn compatible).

Every model takes the FULL feature table (``features.extract_features``; DataFrame, one row per trial)
and selects its own columns, so new models plug in without touching the evaluation code:

    def make_my_model():
        return Pipeline([('select', ColumnSelector(groups=('chiral',))), ('clf', SomeClassifier())])
    MODEL_REGISTRY['my_model'] = make_my_model        # or: --models my_package.my_module:make_my_model

Labels: any binary labels; the evaluation uses y = 1 for the RIGHT hand, so ``predict_proba[:, 1]`` is
p(right).
"""
from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_groups


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Selects feature columns by name and/or feature group from a DataFrame (numpy input passes through).

    Columns absent at transform time are filled with NaN, so a model trained on one feature version keeps
    working (imputed) when a column disappears.
    """

    def __init__(self, columns: Sequence[str] | None = None, groups: Sequence[str] | None = None):
        self.columns = columns
        self.groups = groups

    def _wanted(self) -> list[str]:
        cols: list[str] = list(self.columns or [])
        if self.groups:
            g = feature_groups()
            for name in self.groups:
                cols += [c for c in g[name] if c not in cols]
        return cols

    def fit(self, X: Any, y: Any = None) -> 'ColumnSelector':
        if isinstance(X, pd.DataFrame):
            wanted = self._wanted() or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            present = [c for c in wanted if c in X.columns]
            if not present:
                raise KeyError(f'none of the selected feature columns are present: {wanted[:5]}...')
            self.columns_ = present
        else:
            self.columns_ = None
            self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.columns_ is None:
            return np.asarray(X, dtype=float)
        if not isinstance(X, pd.DataFrame):
            raise TypeError('ColumnSelector was fitted on a DataFrame; transform needs a DataFrame too')
        return X.reindex(columns=self.columns_).to_numpy(dtype=float, na_value=np.nan)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns_ or [], dtype=object)


class YawSignRule(ClassifierMixin, BaseEstimator):
    """One-feature rule: the sign of the net yaw about gravity tells the hand.

    fit() learns WHICH sign means the positive class (classes_[1], 'right' for y in {0,1} or
    {'left','right'}) from the class medians, and a logistic slope (no intercept: symmetric around
    ``threshold``) by maximum likelihood, so predict_proba is calibrated-ish. NaN -> p = 0.5.
    """

    def __init__(self, feature: str = 'chi_yaw_net_deg', threshold: float = 0.0):
        self.feature = feature
        self.threshold = threshold

    def _x(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.feature not in X.columns:
                raise KeyError(f'YawSignRule needs feature column {self.feature!r}')
            return X[self.feature].to_numpy(dtype=float, na_value=np.nan)
        X = np.asarray(X, dtype=float)
        return X[:, 0] if X.ndim == 2 else X

    def fit(self, X: Any, y: Any) -> 'YawSignRule':
        x = self._x(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(f'YawSignRule needs exactly 2 classes, got {self.classes_}')
        pos = y == self.classes_[1]
        ok = np.isfinite(x)
        z = x - self.threshold
        if (ok & pos).any() and (ok & ~pos).any():
            diff = np.median(z[ok & pos]) - np.median(z[ok & ~pos])
            self.sign_ = 1.0 if diff >= 0 else -1.0
        else:
            self.sign_ = 1.0
        s = self.sign_ * z[ok]
        t = pos[ok].astype(float)
        scale = float(np.median(np.abs(s))) if s.size else 1.0
        scale = scale if scale > 0 else 1.0

        def nll(log_k: float) -> float:
            k = math.exp(log_k) / scale
            m = k * s
            return float(np.sum(np.logaddexp(0.0, -m) * t + np.logaddexp(0.0, m) * (1 - t)))

        if s.size:
            r = minimize_scalar(nll, bounds=(-6.0, 6.0), method='bounded')
            self.slope_ = math.exp(float(r.x)) / scale
        else:
            self.slope_ = 1.0
        self.n_features_in_ = 1 if not isinstance(X, pd.DataFrame) else X.shape[1]
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        z = self.sign_ * (self._x(X) - self.threshold)
        return np.where(np.isfinite(z), z, 0.0)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = expit(self.slope_ * self.decision_function(X))  # overflow-safe logistic
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


def make_yaw_rule() -> YawSignRule:
    return YawSignRule()


def make_logreg(C: float = 1.0) -> Pipeline:
    """Standardised logistic regression on the grip-invariant groups (invariant + chiral)."""
    return Pipeline([
        ('select', ColumnSelector(groups=('invariant', 'chiral'))),
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=C, max_iter=5000)),
    ])


def make_logreg_chiral(C: float = 1.0) -> Pipeline:
    """Logistic regression on the chiral group only (mirror-antisymmetric by construction, no intercept)."""
    return Pipeline([
        ('select', ColumnSelector(groups=('chiral',))),
        ('impute', SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)),
        ('scale', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(C=C, max_iter=5000, fit_intercept=False)),
    ])


def make_hgb(random_state: int = 0) -> Pipeline:
    """Histogram gradient boosting on every feature group (NaN handled natively)."""
    return Pipeline([
        ('select', ColumnSelector(groups=('basic', 'device', 'invariant', 'chiral'))),
        ('clf', HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1.0,
            early_stopping=False, random_state=random_state)),
    ])


def make_chance() -> Pipeline:
    """Rung 0 of the ladder: the training prior (accuracy ~50 %, AUC 0.5; a leakage check)."""
    return Pipeline([('select', ColumnSelector(columns=('duration_ms',))), ('clf', DummyClassifier(strategy='prior'))])


MODEL_REGISTRY: dict[str, Callable[[], Any]] = {
    'chance': make_chance,
    'yaw_rule': make_yaw_rule,
    'logreg': make_logreg,
    'logreg_chiral': make_logreg_chiral,
    'hgb': make_hgb,
}
DEFAULT_MODELS = ('yaw_rule', 'logreg', 'hgb')


def get_model_factory(spec: str) -> Callable[[], Any]:
    """'name' from MODEL_REGISTRY, or 'package.module:function' returning a fresh estimator."""
    if spec in MODEL_REGISTRY:
        return MODEL_REGISTRY[spec]
    if ':' in spec:
        mod, fn = spec.split(':', 1)
        f = getattr(importlib.import_module(mod), fn)
        if not callable(f):
            raise TypeError(f'{spec} is not callable')
        return f
    raise KeyError(f'unknown model {spec!r}; registered: {sorted(MODEL_REGISTRY)} (or use module:function)')


def p_right(model: Any, X: Any) -> np.ndarray:
    """p(class 1) from a fitted classifier (the evaluation uses y = 1 for the right hand)."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, 'classes_', [0, 1]))
    j = classes.index(1) if 1 in classes else (classes.index('right') if 'right' in classes else 1)
    return np.asarray(proba)[:, j]
