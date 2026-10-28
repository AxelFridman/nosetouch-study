"""Lane m4, part 2: the segment classifier (approach / return / other), the pluggable hand classifiers and
the probability calibrators. Nothing here reads the cache: every function gets arrays and labels from its
caller (inside ``fit_predict`` only the training rows' labels are visible).

Hand classifiers (``HAND_MODELS``; plug a new one in with :func:`register_hand_model`) take (N, 17, 64)
time-normalised windows and return a LOGIT of P(right):
* ``yaw8`` - the reference of the registry (README section 9): logistic regression on the cumulative yaw
  about gravity at 8 instants, mirror-augmented (pre-specified, C = 1).
* ``chiral16`` - L2 logistic regression WITHOUT intercept on the binned yaw rate (16 bins) and the cumulative
  yaw (16 instants), mirror-augmented, so it is exactly antisymmetric (p(mirror x) = 1 - p(x)); C is chosen
  inside the fold on ``inner_splits`` (nested).
Any object with ``fit(X, y, groups) -> self`` and ``logit(X) -> (N,)`` can be registered.

Calibrators (``CALIBRATORS``) map a logit to P(right); they are fitted on CROSS-FITTED logits of the training
persons (``fold.inner_splits``) and applied to the held-out person: ``none``, ``temperature`` (one scale,
keeps the antisymmetry), ``platt`` (logistic on the logit), ``isotonic``, ``beta`` (Kull et al. 2017: logistic
on ln p and -ln(1 - p)), ``isotonic_sym`` (isotonic fitted on (z, y) and its mirror (-z, 1 - y), so that
P(-z) = 1 - P(z)) and ``nested``: the calibrator chosen INSIDE the fold among the antisymmetric ones
(``NESTED_CANDIDATES``: none, temperature, isotonic_sym) by the log loss of a person-grouped cross-validation of
the cross-fitted logits. The hand models are antisymmetric by construction (mirror augmentation), and only an
antisymmetric calibrator keeps P(mirror image) = 1 - P, so Platt, beta and plain isotonic are compared but never
chosen.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from stage2.standardize import CHANNELS, mirror_channels

EPS = 1e-6


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -40, 40)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ───────────────────────────── hand classifiers ─────────────────────────────

K_YAW = CHANNELS.index('yaw_cum')
K_RATE = CHANNELS.index('yaw_rate')


class Yaw8:
    """Reference: logistic regression on yaw_cum at 8 instants, mirror-augmented (C = 1, with intercept)."""
    name = 'yaw8'

    def __init__(self, C: float = 1.0):
        self.C = C
        self.m: LogisticRegression | None = None

    @staticmethod
    def feat(X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(X)[:, K_YAW, ::8].astype(float))

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Any = None, seed: int = 0) -> 'Yaw8':
        Xa = np.concatenate([X, mirror_channels(X)])
        ya = np.concatenate([y, 1 - y])
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(self.feat(Xa), ya)
        return self

    def logit(self, X: np.ndarray) -> np.ndarray:
        assert self.m is not None
        return self.m.decision_function(self.feat(X))


class Chiral16:
    """Antisymmetric L2 logistic regression (no intercept) on 16 yaw-rate bins and yaw_cum at 16 instants,
    mirror-augmented; features scaled by the training RMS; C from inner_splits (nested)."""
    name = 'chiral16'
    C_GRID = (0.01, 0.1, 1.0)

    def __init__(self, C: float | None = None):
        self.C = C
        self.m: LogisticRegression | None = None
        self.scale: np.ndarray | None = None

    @staticmethod
    def feat(X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=float))
        rate = X[:, K_RATE, :].reshape(len(X), 16, 4).mean(axis=2)
        yaw = X[:, K_YAW, 3::4]
        return np.concatenate([rate, yaw], axis=1)

    def _fit(self, F: np.ndarray, y: np.ndarray, C: float) -> tuple[LogisticRegression, np.ndarray]:
        Fa = np.concatenate([F, -F])            # mirror: every chiral feature flips, the label flips
        ya = np.concatenate([y, 1 - y])
        sc = np.sqrt(np.mean(Fa ** 2, axis=0)) + 1e-9
        m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(Fa / sc, ya)
        return m, sc

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Any = None, seed: int = 0) -> 'Chiral16':
        F = self.feat(X)
        C = self.C
        if C is None:
            if groups is None or len(set(np.asarray(groups).tolist())) < 3:
                C = 0.1
            else:
                from stage2.cv import inner_splits
                best = None
                for c in self.C_GRID:
                    ll = []
                    for tr, va in inner_splits(groups):
                        m, sc = self._fit(F[tr], y[tr], c)
                        p = np.clip(m.predict_proba(F[va] / sc)[:, 1], EPS, 1 - EPS)
                        ll.append(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
                    score = float(np.mean(ll))
                    if best is None or score < best[0] - 1e-12:
                        best = (score, c)
                C = best[1]
        self.C_chosen = C
        self.m, self.scale = self._fit(F, y, C)
        return self

    def logit(self, X: np.ndarray) -> np.ndarray:
        assert self.m is not None and self.scale is not None
        return self.m.decision_function(self.feat(X) / self.scale)


HAND_MODELS: dict[str, Callable[[], Any]] = {'yaw8': Yaw8, 'chiral16': Chiral16}


def register_hand_model(name: str, factory: Callable[[], Any]) -> None:
    """Plug in another hand classifier: ``factory()`` returns an object with fit(X, y, groups, seed) and
    logit(X) on (N, 17, 64) windows (the synthesis step plugs its final model in here)."""
    HAND_MODELS[name] = factory


def make_hand_model(name: str) -> Any:
    return HAND_MODELS[name]()


# ───────────────────────────── the segment classifier (hand-agnostic) ─────────────────────────────


class SegmentClassifier:
    """Multinomial L2 logistic regression on hand-agnostic segment features. Production (pre-specified):
    robust scaling (median / IQR of the TRAINING segments), values clipped to +-5 IQR (winsorised), NaN -> the
    median, C = 1. Ablations: ``scaling`` 'z' (mean / SD, clipped to +-5 SD) or 'none' (raw values, NaN -> the
    median); ``C`` 'nested' (chosen among ``C_GRID`` by the multinomial log loss of a person-grouped inner
    cross-validation of the training segments; needs ``groups``). Classes: approach / return / other."""
    C_GRID = (0.1, 1.0, 10.0)

    def __init__(self, features: Sequence[str], C: float | str = 1.0, scaling: str = 'robust'):
        self.features = list(features)
        self.C = C
        self.scaling = scaling

    def _prep(self, F: np.ndarray) -> np.ndarray:
        Z = (F - self.med) / self.iqr
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return Z if self.scaling == 'none' else np.clip(Z, -5.0, 5.0)

    def _scale(self, F: np.ndarray) -> None:
        if self.scaling == 'robust':
            self.med = np.nanmedian(F, axis=0)
            q75, q25 = np.nanpercentile(F, 75, axis=0), np.nanpercentile(F, 25, axis=0)
            self.iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
            self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        elif self.scaling == 'z':
            self.med = np.nanmean(F, axis=0)
            sd = np.nanstd(F, axis=0)
            self.iqr = np.where(sd > 1e-9, sd, 1.0)
            self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        elif self.scaling == 'none':               # raw values; NaN -> the training median (then 0 offset)
            med = np.nanmedian(F, axis=0)
            self.med = np.zeros(F.shape[1])
            self.iqr = np.ones(F.shape[1])
            self._fill = np.where(np.isfinite(med), med, 0.0)
        else:
            raise ValueError(self.scaling)

    def _fill_nan(self, F: np.ndarray) -> np.ndarray:
        if self.scaling == 'none':
            return np.where(np.isfinite(F), F, self._fill)
        return F

    def fit(self, F: np.ndarray, labels: Sequence[str], seed: int = 0, groups: Any = None) -> 'SegmentClassifier':
        F = np.asarray(F, dtype=float)
        labels = np.asarray(labels)
        C = self.C
        if C == 'nested':
            C = self._nested_C(F, labels, groups)
        self.C_chosen = float(C)
        self._scale(F)
        with warnings.catch_warnings():
            if self.scaling == 'none':                # the unscaled ablation converges slowly (expected; it is the point)
                warnings.simplefilter('ignore', ConvergenceWarning)
            self.m = LogisticRegression(C=float(C), max_iter=3000).fit(self._prep(self._fill_nan(F)), labels)
        return self

    def _nested_C(self, F: np.ndarray, labels: np.ndarray, groups: Any) -> float:
        from stage2.cv import inner_splits
        if groups is None or len(set(np.asarray(groups).tolist())) < 3:
            return 1.0
        best = None
        for c in self.C_GRID:
            ll, n = 0.0, 0
            for tr, va in inner_splits(np.asarray(groups)):
                if len(set(labels[tr].tolist())) < len(set(labels.tolist())):
                    continue
                m = SegmentClassifier(self.features, C=c, scaling=self.scaling)
                m._scale(F[tr])
                with warnings.catch_warnings():
                    if self.scaling == 'none':
                        warnings.simplefilter('ignore', ConvergenceWarning)
                    m.m = LogisticRegression(C=c, max_iter=3000).fit(m._prep(m._fill_nan(F[tr])), labels[tr])
                P = m.m.predict_proba(m._prep(m._fill_nan(F[va])))
                col = {k: i for i, k in enumerate(m.m.classes_)}
                pk = np.array([P[i, col[l]] if l in col else EPS for i, l in enumerate(labels[va])])
                ll += float(-np.sum(np.log(np.clip(pk, EPS, 1.0))))
                n += va.size
            score = ll / max(n, 1)
            if best is None or score < best[0] - 1e-12:
                best = (score, c)
        return float(best[1])

    def proba(self, F: np.ndarray) -> dict[str, np.ndarray]:
        P = self.m.predict_proba(self._prep(self._fill_nan(np.asarray(F, dtype=float))))
        out = {c: P[:, i] for i, c in enumerate(self.m.classes_)}
        n = len(F)
        for c in ('approach', 'return', 'other'):
            out.setdefault(c, np.zeros(n))
        return out

    def coef_table(self) -> list[dict[str, Any]]:
        rows = []
        for i, c in enumerate(self.m.classes_):
            for f, w in zip(self.features, self.m.coef_[i]):
                rows.append({'class': str(c), 'feature': f, 'coef': float(w)})
        return rows


# ───────────────────────────── calibrators ─────────────────────────────


class Calibrator:
    method = 'none'
    antisymmetric = True                      # P(-z) = 1 - P(z) for every z

    def fit(self, z: np.ndarray, y: np.ndarray, groups: Any = None) -> 'Calibrator':
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    def params(self) -> dict[str, float]:
        return {}


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class Temperature(Calibrator):
    method = 'temperature'

    def fit(self, z, y, groups=None):
        z, y = np.asarray(z, float), np.asarray(y, float)
        r = minimize_scalar(lambda lt: _nll(sigmoid(z / math.exp(lt)), y), bounds=(-3.0, 3.0), method='bounded')
        self.T = float(math.exp(r.x))
        return self

    def __call__(self, z):
        return sigmoid(np.asarray(z, float) / self.T)

    def params(self):
        return {'T': self.T}


class Platt(Calibrator):
    method = 'platt'
    antisymmetric = False

    def fit(self, z, y, groups=None):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(z, float)[:, None], np.asarray(y, int))
        self.a, self.b = float(m.coef_[0, 0]), float(m.intercept_[0])
        return self

    def __call__(self, z):
        return sigmoid(self.a * np.asarray(z, float) + self.b)

    def params(self):
        return {'a': self.a, 'b': self.b}


class Isotonic(Calibrator):
    method = 'isotonic'
    antisymmetric = False

    def fit(self, z, y, groups=None):
        self.m = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(np.asarray(z, float),
                                                                                       np.asarray(y, float))
        return self

    def __call__(self, z):
        return np.clip(self.m.predict(np.asarray(z, float)), 0.01, 0.99)


class IsotonicSym(Isotonic):
    """Isotonic regression fitted on the cross-fitted logits AND their mirror images (-z, 1 - y): the PAV
    solution of a mirror-symmetric data set is itself antisymmetric, f(-z) = 1 - f(z) (checked by a test)."""
    method = 'isotonic_sym'
    antisymmetric = True

    def fit(self, z, y, groups=None):
        z, y = np.asarray(z, float), np.asarray(y, float)
        return super().fit(np.concatenate([z, -z]), np.concatenate([y, 1.0 - y]))

    def __call__(self, z):
        z = np.asarray(z, float)
        # evaluate on |z| and reflect, so the antisymmetry is exact whatever the interpolation does at the knots
        p = np.clip(self.m.predict(np.abs(z)), 0.01, 0.99)
        return np.where(z >= 0, p, 1.0 - p)


class Beta(Calibrator):
    method = 'beta'
    antisymmetric = False

    def fit(self, z, y, groups=None):
        p = sigmoid(z)
        Fz = np.column_stack([np.log(np.clip(p, EPS, 1)), -np.log(np.clip(1 - p, EPS, 1))])
        m = LogisticRegression(C=1e6, max_iter=1000).fit(Fz, np.asarray(y, int))
        self.coef, self.b = m.coef_[0].astype(float), float(m.intercept_[0])
        return self

    def __call__(self, z):
        p = sigmoid(z)
        Fz = np.column_stack([np.log(np.clip(p, EPS, 1)), -np.log(np.clip(1 - p, EPS, 1))])
        return sigmoid(Fz @ self.coef + self.b)

    def params(self):
        return {'a': float(self.coef[0]), 'b': float(self.coef[1]), 'c': self.b}


#: the antisymmetric calibrators the nested choice compares, simplest first (ties go to the simpler one)
NESTED_CANDIDATES = ('none', 'temperature', 'isotonic_sym')


class Nested(Calibrator):
    """The calibrator chosen INSIDE the fold (module docstring): each candidate of ``NESTED_CANDIDATES`` is fitted
    on the cross-fitted logits of all but one inner group of persons (``cv.inner_splits``) and scored by the log
    loss on the persons left out; the lowest pooled log loss wins (ties within 1e-9: the simpler), then it is
    refitted on every cross-fitted logit. Needs ``groups`` (the person of each logit)."""
    method = 'nested'

    def fit(self, z, y, groups=None):
        from stage2.cv import inner_splits
        z, y = np.asarray(z, float), np.asarray(y, float)
        scores: dict[str, float] = {}
        if groups is not None and len(set(np.asarray(groups).tolist())) >= 3:
            splits = inner_splits(np.asarray(groups).astype(str))
            for name in NESTED_CANDIDATES:
                tot = 0.0
                for tr, va in splits:
                    c = CALIBRATORS[name]().fit(z[tr], y[tr])
                    p = np.clip(c(z[va]), EPS, 1 - EPS)
                    tot += float(-np.sum(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)))
                scores[name] = tot / z.size
            best = min(NESTED_CANDIDATES, key=lambda k: (round(scores[k], 9), NESTED_CANDIDATES.index(k)))
        else:
            best = NESTED_CANDIDATES[0]
        self.chosen = best
        self.cv_log_loss = scores
        self.inner = CALIBRATORS[best]().fit(z, y)
        return self

    def __call__(self, z):
        return self.inner(z)

    def params(self):
        return {'chosen': self.chosen, **{f'cv_log_loss_{k}': v for k, v in self.cv_log_loss.items()},
                **self.inner.params()}


CALIBRATORS: dict[str, Callable[[], Calibrator]] = {'none': Calibrator, 'temperature': Temperature, 'platt': Platt,
                                                    'isotonic': Isotonic, 'beta': Beta, 'isotonic_sym': IsotonicSym,
                                                    'nested': Nested}


def make_calibrator(name: str) -> Calibrator:
    return CALIBRATORS[name]()


class AggregateScale:
    """Calibration of a k-attempt aggregate: P = sigmoid(s_k * mean of the k calibrated logits), s_k fitted per
    k on cross-fitted training attempts: :meth:`fit` on random k-subsets of the recordings of one person and hand
    (each recording's chosen segment), :meth:`fit_sessions` on the attempts a cue-free session actually yields
    (``m4_aggregate``). s_k = 1 is plain averaging, s_k = k is summing (independent attempts)."""

    def __init__(self, ks: Sequence[int] = (1, 2, 3, 4, 5), n_draws: int = 40):
        self.ks = tuple(ks)
        self.n_draws = n_draws
        self.s: dict[int, float] = {}

    def fit(self, z: np.ndarray, y: np.ndarray, persons: np.ndarray, seed: int = 0) -> 'AggregateScale':
        rng = np.random.default_rng(seed)
        z, y, persons = np.asarray(z, float), np.asarray(y, int), np.asarray(persons).astype(str)
        for k in self.ks:
            m_all, y_all = [], []
            for p in np.unique(persons):
                for h in (0, 1):
                    idx = np.flatnonzero((persons == p) & (y == h))
                    if idx.size < k:
                        continue
                    for _ in range(self.n_draws):
                        pick = rng.choice(idx, size=k, replace=False)
                        m_all.append(float(np.mean(z[pick])))
                        y_all.append(h)
            if len(set(y_all)) < 2:
                self.s[k] = 1.0
                continue
            m_all, y_all = np.asarray(m_all), np.asarray(y_all, float)
            r = minimize_scalar(lambda ls: _nll(sigmoid(math.exp(ls) * m_all), y_all), bounds=(-2.0, 2.5),
                                method='bounded')
            self.s[k] = float(math.exp(r.x))
        return self

    def fit_sessions(self, seqs: Sequence[Sequence[float]], y: Sequence[int], *, form: str = 'power',
                     min_sessions: int = 10) -> 'AggregateScale':
        """s_k fitted on SESSIONS instead of random recordings, on the mean calibrated logit of the first k
        accepted attempts (decision order) of each session with at least k attempts, against the session's hand
        - exactly the quantity the stopping rule thresholds after attempt k.

        form 'power' (default): s_k = s_1 * k^b with b in [0, 1] (b = 0: the attempts count as one, b = 1: as
        independent), two parameters fitted by maximum likelihood on every (session, k) prefix at once, so s_k
        grows with k and stays stable where few sessions reach k attempts. form 'per_k': one scale per k, as
        :meth:`fit` (a k with fewer than ``min_sessions`` sessions or one hand keeps the previous scale)."""
        y = np.asarray(y, dtype=int)
        self.s = {}
        if form == 'per_k':
            prev = 1.0
            for k in self.ks:
                sel = [i for i, z in enumerate(seqs) if len(z) >= k]
                yk = y[sel] if sel else np.zeros(0, dtype=int)
                if len(sel) < min_sessions or len(set(yk.tolist())) < 2:
                    self.s[k] = prev
                    continue
                m = np.array([float(np.mean(np.asarray(seqs[i], dtype=float)[:k])) for i in sel])
                r = minimize_scalar(lambda ls: _nll(sigmoid(math.exp(ls) * m), yk), bounds=(-2.0, 2.5),
                                    method='bounded')
                self.s[k] = prev = float(math.exp(r.x))
            return self
        kk, mm, yy = [], [], []
        for z, h in zip(seqs, y):
            z = np.asarray(z, dtype=float)
            for k in self.ks:
                if z.size >= k:
                    kk.append(k)
                    mm.append(float(z[:k].mean()))
                    yy.append(int(h))
        if len(set(yy)) < 2:
            self.s = {k: 1.0 for k in self.ks}
            return self
        kk, mm, ya = np.asarray(kk, float), np.asarray(mm, float), np.asarray(yy, float)
        from scipy.optimize import minimize
        r = minimize(lambda q: _nll(sigmoid(math.exp(q[0]) * kk ** q[1] * mm), ya), x0=np.array([0.0, 0.5]),
                     method='L-BFGS-B', bounds=[(-2.0, 2.5), (0.0, 1.0)])
        self.s1, self.b = float(math.exp(r.x[0])), float(r.x[1])
        self.s = {k: self.s1 * float(k) ** self.b for k in self.ks}
        return self

    def scale(self, k: int) -> float:
        if not self.s:
            return 1.0
        if k in self.s:
            return self.s[k]
        return self.s[max(self.s)]

    def __call__(self, z_list: Sequence[float]) -> float:
        k = len(z_list)
        if k == 0:
            return 0.5
        return float(sigmoid(self.scale(k) * float(np.mean(z_list))))
