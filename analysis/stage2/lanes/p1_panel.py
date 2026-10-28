"""Lane p1: the FIXED panel of reference models that isolates preprocessing choices.

Three pre-specified models of different kinds (every choice below is fixed before any variant is compared):

* ``lr``  - L2 logistic regression (C = 1) on the standardised time-binned means (8 bins) of the chiral
  gravity-frame channels (``yaw_rate``, ``yaw_cum``), mirror-augmented, like the reference; the feature scaler
  is fitted on the (augmented) training rows of the fold. Pre-specified: nothing is tuned.
* ``gbm`` - LightGBM with fixed parameters (:data:`GBM_PARAMS`) on a moderate rich summary of all 17
  channels (8 bin means + mean, SD, min, max, time of min and max: 14 per channel), mirror-augmented.
* ``mr``  - MiniRocket (:mod:`p1_minirocket`, biases fitted on the training rows of the fold only) on the 8
  gravity-frame channels + ridge regression; ridge alpha chosen by inner GroupKFold and the ridge decision
  mapped to P(right) by Platt scaling fitted on the inner cross-fitted decisions (real trials only).

:func:`make_fit_predict` builds the ``fit_predict(fold)`` of one model under one set of DATA-TREATMENT options
(:class:`Treatment`): channel set, normalisation, winsorising, augmentation (mirror, screen-away flip, time
warp, segmentation jitter), training-row filters (platform, flags, blocks, exclusion thresholds) and, for a
dict ``X``, which array is trained on and which is predicted. Everything that spans trials (normaliser,
winsor limits, MiniRocket biases, scaler, C / alpha, Platt) is fitted on the training rows of the fold only;
augmented copies never enter an inner validation fold (docs/09 D13).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from stage2 import config as C
from stage2.standardize import CHANNELS, CHIRAL_CHANNELS, GRAVITY_CHANNELS, mirror_channels, screen_flip_channels

# ───────────────────────────── fixed panel settings ─────────────────────────────

N_BINS = 8                                    # time bins of the LR and GBM features
LR_C = 1.0                                    # pre-specified (features standardised on the training rows)
MR_FEATURES = 2_000                           # MiniRocket features requested (84 kernels x 23 = 1932)
MR_ALPHAS = tuple(float(a) for a in np.logspace(0, 5, 11))   # inner grid of the ridge
GBM_PARAMS = dict(objective='binary', n_estimators=100, learning_rate=0.1, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, max_bin=63,
                  deterministic=True, force_col_wise=True, verbose=-1)
GBM_THREADS = 1                               # one thread: no OpenMP spin-waiting on a shared machine
MR_INNER_SPLITS = 3                           # GroupKFold folds for the ridge alpha + Platt (bounds the eigh cost)
WARP_MAX = 0.15                               # time-warp amplitude: phi(s) = s + a sin(pi s) / pi, |a| <= 0.15

#: default channel set of each panel model (the 'current choice' of its representation)
DEFAULT_CHANNELS = {'lr': tuple(CHIRAL_CHANNELS), 'gbm': tuple(CHANNELS), 'mr': tuple(GRAVITY_CHANNELS)}
from stage2.lanes.p1_labels import MODEL_LABELS, MODELS  # noqa: E402  (labels live in the light module)


@dataclass(frozen=True)
class Treatment:
    """Data-treatment options applied inside fit_predict (defaults = the pipeline's current choices)."""

    channels: tuple[str, ...] | None = None     # None: the model's default channel set
    norm: str = 'none'                          # 'none' | 'zscore' | 'robust' | 'trial_peak'
    winsor: float | None = None                 # clip each channel at the q / 1-q quantiles of the training rows
    mirror: bool = True                         # mirror augmentation (mirrored trial, other hand)
    screen_flip: bool = False                   # synthetic screen-away copies (label kept)
    time_warp: bool = False                     # one smoothly time-warped copy per training trial
    train_keys: tuple[str, ...] = ('base',)     # dict X: arrays trained on (stacked as extra copies)
    test_key: str = 'base'                      # dict X: the array predicted
    train_filter: tuple[tuple[str, Any], ...] = ()   # (kind, value) training-row filters, see _train_mask

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['train_filter'] = [list(x) for x in self.train_filter]
        return d


DEFAULT = Treatment()


# ───────────────────────────── array helpers ─────────────────────────────


def _arr(X: Any, key: str) -> np.ndarray:
    if isinstance(X, dict):
        return np.asarray(X[key], dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def _channel_idx(names: tuple[str, ...]) -> list[int]:
    return [CHANNELS.index(c) for c in names]


def time_warp(X: np.ndarray, rng: np.random.Generator, amp: float = WARP_MAX) -> np.ndarray:
    """One smooth monotone warp per trial: x'(s) = x(phi(s)), phi(s) = s + a sin(pi s) / pi, a ~ U(-amp, amp)
    (end points fixed, phi' = 1 + a cos(pi s) > 0). Mirror-covariant (a per-sample resampling)."""
    N, C, T = X.shape
    s = np.linspace(0.0, 1.0, T)
    out = np.empty_like(X)
    a = rng.uniform(-amp, amp, N)
    for i in range(N):
        phi = s + a[i] * np.sin(np.pi * s) / np.pi
        for c in range(C):
            out[i, c] = np.interp(phi, s, X[i, c])
    return out


def bin_means(X: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """(N, C * n_bins) float64: the mean of each channel over n_bins consecutive equal time bins. float64, so
    that a model's predictions for a row never depend on the other rows of the batch (float32 BLAS kernels
    round differently for different batch sizes, which the harness's leak check would flag)."""
    N, C, T = X.shape
    X = np.asarray(X, dtype=np.float64)
    edges = np.linspace(0, T, n_bins + 1).round().astype(int)
    return np.concatenate([X[:, :, a:b].mean(axis=2) for a, b in zip(edges[:-1], edges[1:])], axis=1)


def summary_features(X: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """(N, 14 C): per channel 8 bin means + mean, SD, min, max and the relative times of the min and max."""
    N, C, T = X.shape
    t = np.linspace(0.0, 1.0, T)
    X = np.asarray(X, dtype=np.float64)
    parts = [bin_means(X, n_bins), X.mean(axis=2), X.std(axis=2), X.min(axis=2), X.max(axis=2),
             t[X.argmin(axis=2)], t[X.argmax(axis=2)]]
    return np.concatenate(parts, axis=1).astype(np.float64)


# ───────────────────────────── normalisation and winsorising ─────────────────────────────


class Normaliser:
    """Per-channel scaling fitted on TRAINING arrays (N, C, T) over trials and time; 'trial_peak' divides each
    trial's channel by its own peak |x| (a per-trial transform, nothing fitted)."""

    def __init__(self, kind: str = 'none', winsor: float | None = None):
        if kind not in ('none', 'zscore', 'robust', 'trial_peak'):
            raise ValueError(f'unknown normalisation {kind!r}')
        self.kind, self.winsor = kind, winsor

    def fit(self, X: np.ndarray) -> 'Normaliser':
        C = X.shape[1]
        flat = np.transpose(X, (1, 0, 2)).reshape(C, -1)
        if self.winsor:
            self.lo_ = np.quantile(flat, self.winsor, axis=1)[None, :, None]
            self.hi_ = np.quantile(flat, 1 - self.winsor, axis=1)[None, :, None]
            flat = np.clip(flat, self.lo_[0], self.hi_[0])
        if self.kind == 'zscore':
            self.loc_ = flat.mean(axis=1)[None, :, None]
            sd = flat.std(axis=1)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)[None, :, None]
        elif self.kind == 'robust':
            self.loc_ = np.median(flat, axis=1)[None, :, None]
            q75, q25 = np.quantile(flat, [0.75, 0.25], axis=1)
            iqr = q75 - q25
            self.scale_ = np.where(iqr > 1e-12, iqr, 1.0)[None, :, None]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.winsor:
            X = np.clip(X, self.lo_, self.hi_)
        if self.kind in ('zscore', 'robust'):
            return ((X - self.loc_) / self.scale_).astype(np.float32)
        if self.kind == 'trial_peak':
            pk = np.abs(X).max(axis=2, keepdims=True)
            return (X / np.where(pk > 1e-9, pk, 1.0)).astype(np.float32)
        return X


# ───────────────────────────── training rows and augmentation ─────────────────────────────


def _test_platform(fold: Any) -> str:
    test_pf = set(fold.meta_test['platform'].astype(str))
    if len(test_pf) != 1:
        raise ValueError(f'test person on several platforms: {sorted(test_pf)}')
    return next(iter(test_pf))


def _flagged(m: Any, flags: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(m), dtype=bool)
    for f in flags:
        out |= m[f].astype('boolean').fillna(False).astype(bool).to_numpy()
    return out


def _train_mask(fold: Any, filters: tuple[tuple[str, Any], ...]) -> np.ndarray:
    """Training rows kept by the filters, applied in order (each (kind, value)):
    ('platform', 'cross' | 'same')   rows of the other / the same platform as the test person;
    ('drop_flags', (f1, f2, ...))    rows with none of these boolean flag columns set;
    ('drop_blocks', ((session_label, block_index), ...))  rows outside these blocks;
    ('min_duration_ms', v) / ('min_peak_gyro_dps', v)   rows at or above the threshold;
    ('sets', (set, ...))             rows of these analysis sets (e.g. ('primary',));
    ('cap', n)                       the docs/09 D2 cap (``cv.cap_rows``, same seed as the harness) on the rows
                                     kept so far: with ``cap_per_person=None`` in the harness and ('sets',
                                     ('primary',)) before it, the training set equals the base run's exactly;
    ('match_flags', (f1, ...))       SIZE-MATCHED CONTROL of ('drop_flags', ...): in every (person, hand) cell,
                                     as many rows as carry none of these flags, but chosen at random (seeded
                                     per person and hand), flagged or not;
    ('match_persons', 'same' | 'cross')  SIZE-MATCHED CONTROL of ('platform', ...): as many training persons as
                                     that filter keeps, chosen at random from both platforms (seeded per fold)."""
    m = fold.meta_train
    keep = np.ones(len(m), dtype=bool)
    for kind, val in filters:
        if kind == 'platform':
            pf = m['platform'].astype(str).to_numpy()
            keep &= (pf != _test_platform(fold)) if val == 'cross' else (pf == _test_platform(fold))
        elif kind == 'drop_flags':
            keep &= ~_flagged(m, tuple(val))
        elif kind == 'cap':
            from stage2.cv import cap_rows
            idx = np.flatnonzero(keep)
            sub = cap_rows(np.asarray(fold.groups_train)[idx], np.asarray(fold.y_train)[idx], int(val), C.SEED)
            keep = np.zeros(len(m), dtype=bool)
            keep[idx[sub]] = True
        elif kind == 'match_flags':
            import zlib
            clean = ~_flagged(m, tuple(val))
            g = np.asarray(fold.groups_train).astype(str)
            yy = np.asarray(fold.y_train).astype(int)
            new = np.zeros(len(m), dtype=bool)
            for p in sorted(set(g[keep].tolist())):
                for h in (0, 1):
                    cell = np.flatnonzero(keep & (g == p) & (yy == h))
                    n_keep = int(np.sum(clean[cell]))
                    if n_keep:
                        rng = np.random.default_rng([C.SEED, zlib.crc32(p.encode()), 43, h])
                        new[np.sort(rng.permutation(cell)[:n_keep])] = True
            keep = new
        elif kind == 'match_persons':
            pf = m['platform'].astype(str).to_numpy()
            g = np.asarray(fold.groups_train).astype(str)
            sel = (pf != _test_platform(fold)) if val == 'cross' else (pf == _test_platform(fold))
            n_persons = len(set(g[keep & sel].tolist()))
            pool = sorted(set(g[keep].tolist()))
            rng = np.random.default_rng([int(fold.seed), 47])
            chosen = set(rng.choice(pool, size=min(n_persons, len(pool)), replace=False).tolist())
            keep &= np.isin(g, list(chosen))
        elif kind == 'drop_blocks':
            import pandas as pd
            bi = pd.to_numeric(m['block_index'], errors='coerce').fillna(-1).astype(int)
            idx = pd.MultiIndex.from_arrays([m['session_label'].astype(str), bi])
            keep &= ~idx.isin([tuple(b) for b in val])
        elif kind == 'min_duration_ms':
            keep &= m['duration_ms'].astype(float).to_numpy() >= float(val)
        elif kind == 'min_peak_gyro_dps':
            keep &= m['peak_gyro_dps'].astype(float).to_numpy() >= float(val)
        elif kind == 'sets':
            keep &= m['analysis_set'].astype(str).isin(list(val)).to_numpy()
        else:
            raise ValueError(f'unknown training filter {kind!r}')
    return keep


def augment(Xs: list[np.ndarray], y: np.ndarray, tr: Treatment, rng: np.random.Generator
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the training arrays (one per train key) and their augmented copies.
    Returns (X (M, C, T) full channels, y (M,), is_real (M,): True for the first copy of each real trial)."""
    X0 = np.concatenate(Xs, axis=0)
    y0 = np.concatenate([y] * len(Xs))
    real = np.zeros(len(y0), dtype=bool)
    real[:len(y)] = True
    parts, labels, reals = [X0], [y0], [real]
    if tr.time_warp:
        parts.append(time_warp(X0, rng))
        labels.append(y0)
        reals.append(np.zeros(len(y0), dtype=bool))
    if tr.screen_flip:
        base = np.concatenate(parts)
        parts.append(screen_flip_channels(base))
        labels.append(np.concatenate(labels))
        reals.append(np.zeros(len(base), dtype=bool))
    X1, y1, r1 = np.concatenate(parts), np.concatenate(labels), np.concatenate(reals)
    if tr.mirror:
        X1 = np.concatenate([X1, mirror_channels(X1)])
        y1 = np.concatenate([y1, 1 - y1])
        r1 = np.concatenate([r1, np.zeros(len(r1), dtype=bool)])
    return X1.astype(np.float32), y1.astype(int), r1


# ───────────────────────────── models ─────────────────────────────


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)   # float64: sklearn keeps float32 inputs
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def platt_fit(d: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic regression of y on the score d (a, b): P = sigmoid(a d + b); almost unpenalised."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(d, float).reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


class _Prepared:
    """The fold's training arrays after filtering and augmentation, the test array, channel indices."""

    def __init__(self, fold: Any, tr: Treatment, channels: tuple[str, ...]):
        keep = _train_mask(fold, tr.train_filter)
        if keep.sum() < 10 or len(set(fold.y_train[keep].tolist())) < 2:
            raise ValueError(f'fold {fold.fold}: too few training rows after the filters ({int(keep.sum())})')
        self.keep = keep
        self.y = fold.y_train[keep]
        self.groups = fold.groups_train[keep]
        self.X_train = [_arr(fold.X_train, k)[keep] for k in tr.train_keys]
        self.X_test = _arr(fold.X_test, tr.test_key)
        self.ch = _channel_idx(channels)
        self.tr = tr
        self.seed = int(fold.seed)

    def inner(self, n_splits: int = C.INNER_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
        from stage2.cv import inner_splits
        return inner_splits(self.groups, n_splits)

    def augmented(self, rows: np.ndarray | None = None, salt: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augmented training arrays (selected channels) of the training rows ``rows`` (None: all)."""
        r = np.arange(len(self.y)) if rows is None else rows
        rng = np.random.default_rng([self.seed, 17, salt])
        Xa, ya, real = augment([x[r] for x in self.X_train], self.y[r], self.tr, rng)
        return Xa[:, self.ch], ya, real

    def real(self, rows: np.ndarray) -> np.ndarray:
        return self.X_train[0][rows][:, self.ch]

    def test(self) -> np.ndarray:
        return self.X_test[:, self.ch]


def fit_predict_lr(fold: Any, tr: Treatment, channels: tuple[str, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    P = _Prepared(fold, tr, channels)
    Xa, ya, _ = P.augmented()
    nm = Normaliser(tr.norm, tr.winsor).fit(Xa)
    Fa = bin_means(nm.transform(Xa))
    mu, sd = Fa.mean(axis=0), Fa.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    m = LogisticRegression(C=LR_C, max_iter=5000).fit((Fa - mu) / sd, ya)
    Ft = (bin_means(nm.transform(P.test())) - mu) / sd
    p = m.predict_proba(Ft)[:, 1].astype(np.float64)
    return p, {'n_train': int(P.keep.sum()), 'n_features': int(Fa.shape[1])}


def fit_predict_gbm(fold: Any, tr: Treatment, channels: tuple[str, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    import lightgbm as lgb
    P = _Prepared(fold, tr, channels)
    Xa, ya, _ = P.augmented()
    nm = Normaliser(tr.norm, tr.winsor).fit(Xa)
    Fa = summary_features(nm.transform(Xa))
    Ft = summary_features(nm.transform(P.test()))
    m = lgb.LGBMClassifier(**GBM_PARAMS, n_jobs=GBM_THREADS, random_state=P.seed)
    m.fit(Fa, ya)
    p = m.predict_proba(Ft)[:, 1].astype(np.float64)
    return p, {'n_train': int(P.keep.sum()), 'n_features': int(Fa.shape[1])}


def _centre_blocks(K: np.ndarray, tr: np.ndarray, other: np.ndarray, K_other: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Centred Gram blocks of a ridge with intercept from the uncentred Gram K of the augmented training rows:
    (Kc[tr, tr], Kc[other, tr]) with the centre = the mean feature vector of the rows ``tr``. ``K_other``:
    the uncentred (other x all training rows) block when ``other`` are not training rows (the test rows)."""
    Ktt = K[np.ix_(tr, tr)]
    Kot = K[np.ix_(other, tr)] if K_other is None else K_other[:, tr]
    col = Ktt.mean(axis=0)
    tot = col.mean()
    return Ktt - col[None, :] - col[:, None] + tot, Kot - Kot.mean(axis=1, keepdims=True) - col[None, :] + tot


def _ridge_decisions(Kc: np.ndarray, Koc: np.ndarray, t: np.ndarray, alphas: tuple[float, ...]) -> list[np.ndarray]:
    """Decision values of the ridge (dual form, intercept = mean target) on the 'other' rows for every alpha,
    from one eigendecomposition of the centred training Gram matrix."""
    lam, V = np.linalg.eigh(Kc)
    lam = np.clip(lam, 0, None)
    mt = t.mean()
    Vt = V.T @ (t - mt)
    KoV = Koc @ V
    return [KoV @ (Vt / (lam + a)) + mt for a in alphas]


def fit_predict_mr(fold: Any, tr: Treatment, channels: tuple[str, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    from stage2.lanes.p1_minirocket import MiniRocket
    P = _Prepared(fold, tr, channels)
    Xa, ya, real = P.augmented()
    nm = Normaliser(tr.norm, tr.winsor).fit(Xa)
    Xa_n = nm.transform(Xa)
    mr = MiniRocket(MR_FEATURES, seed=P.seed).fit(Xa_n)
    Fa = mr.transform(Xa_n)
    Ft = mr.transform(nm.transform(P.test()))
    sd = Fa.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    mu0 = Fa.mean(axis=0)
    Fa = ((Fa - mu0) / sd).astype(np.float64)
    Ft = ((Ft - mu0) / sd).astype(np.float64)
    K = Fa @ Fa.T
    t = 2.0 * ya - 1.0
    # which augmented row belongs to which real training trial (copies share the trial's person)
    n_real = len(P.y)
    owner = np.tile(np.arange(n_real), len(ya) // n_real)
    dec = {a: np.zeros(n_real) for a in MR_ALPHAS}
    for itr, iva in P.inner(MR_INNER_SPLITS):
        rows = np.flatnonzero(np.isin(owner, itr))
        va = np.flatnonzero(real & np.isin(owner, iva))
        Kc, Kvc = _centre_blocks(K, rows, va)
        for a, dv in zip(MR_ALPHAS, _ridge_decisions(Kc, Kvc, t[rows], MR_ALPHAS)):
            dec[a][owner[va]] = dv
    ll, platt = {}, {}
    for a in MR_ALPHAS:
        platt[a] = platt_fit(dec[a], P.y)
        ll[a] = _log_loss(P.y, _sigmoid(platt[a][0] * dec[a] + platt[a][1]))
    a_best = min(MR_ALPHAS, key=lambda a: (round(ll[a], 12), -a))
    allr = np.arange(len(ya))
    Kc, Ktc = _centre_blocks(K, allr, np.zeros(0, dtype=int), K_other=Ft @ Fa.T)
    (d,) = _ridge_decisions(Kc, Ktc, t, (a_best,))
    pa, pb = platt[a_best]
    p = _sigmoid(pa * d + pb)
    return p, {'alpha': a_best, 'platt_a': round(pa, 6), 'platt_b': round(pb, 6), 'inner_logloss': round(ll[a_best], 6),
               'n_train': int(P.keep.sum()), 'n_features': int(Fa.shape[1])}


_FIT = {'lr': fit_predict_lr, 'gbm': fit_predict_gbm, 'mr': fit_predict_mr}


def make_fit_predict(model: str, treatment: Treatment = DEFAULT, **kw: Any):
    """fit_predict(fold) -> (P(right), info) of panel model ``model`` under ``treatment`` (kw: Treatment fields)."""
    if model not in _FIT:
        raise ValueError(f'unknown panel model {model!r}: {MODELS}')
    tr = replace(treatment, **kw) if kw else treatment
    channels = tuple(tr.channels) if tr.channels else DEFAULT_CHANNELS[model]
    fn = _FIT[model]

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        return fn(fold, tr, channels)

    fit_predict.treatment = tr          # type: ignore[attr-defined]
    fit_predict.channels = channels     # type: ignore[attr-defined]
    return fit_predict


def factory(model: str = 'lr', calibration: str | None = None, **treatment: Any):
    """The finalist factory: ``factory(model='lr')`` -> fit_predict (Treatment keyword arguments allowed).
    ``calibration``: None (the panel model's own P(right)) or a :mod:`p1_calibration` method fitted inside the
    fold ('auto' = the nested choice among none / temperature / Platt / beta / isotonic)."""
    tf = treatment.pop('train_filter', ())
    tf = tuple((k, tuple(map(tuple, v)) if k == 'drop_blocks' else (tuple(v) if isinstance(v, list) else v))
               for k, v in tf)
    ch = treatment.pop('channels', None)
    tr = Treatment(channels=tuple(ch) if ch else None, train_filter=tf,
                   **{k: (tuple(v) if isinstance(v, list) else v) for k, v in treatment.items()})
    if calibration:
        from stage2.lanes.p1_calibration import make_calibrated
        return make_calibrated(model, calibration, tr)
    return make_fit_predict(model, tr)


def describe(model: str, tr: Treatment) -> dict[str, Any]:
    """A JSON-able description of one panel configuration (stored with the result)."""
    ch = tuple(tr.channels) if tr.channels else DEFAULT_CHANNELS[model]
    out = {'model': model, 'model_label': MODEL_LABELS[model], **tr.as_dict(), 'channels': list(ch)}
    if model == 'lr':
        out['C'] = LR_C
    elif model == 'gbm':
        out['gbm_params'] = dict(GBM_PARAMS)
    else:
        out['mr_features'] = MR_FEATURES
        out['mr_inner_splits'] = MR_INNER_SPLITS
        out['ridge_alphas'] = [round(a, 6) for a in MR_ALPHAS]
    return out


def is_finite_proba(p: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)))


def needs_harness_cap(tr: Treatment) -> bool:
    """False when the treatment applies the D2 cap itself (a ('cap', n) training filter after another filter):
    the harness must then be called with ``cap_per_person=None``."""
    return not any(k == 'cap' for k, _ in tr.train_filter)


__all__ = ['Treatment', 'DEFAULT', 'make_fit_predict', 'factory', 'describe', 'MODELS', 'MODEL_LABELS',
           'DEFAULT_CHANNELS', 'bin_means', 'summary_features', 'Normaliser', 'augment', 'time_warp', 'platt_fit',
           'needs_harness_cap']


def build_x(d: Any) -> np.ndarray:
    """The finalists' input: the time-normalised approach arrays (N, 17, 64) of a Stage2Data, per trial (the
    panel selects its channels inside fit_predict). No label is touched."""
    return np.asarray(d.approach, dtype=np.float32)
