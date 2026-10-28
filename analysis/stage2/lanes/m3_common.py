"""Lane m3 shared pieces: per-trial inputs, channel sets, mirror signs, normalisers fitted on training rows,
the fold-fit cache, calibrators fitted on cross-fitted training predictions, and the fit_predict wrapper.

Design (every m3 model family follows it):

* ``build_x`` functions take a ``Stage2Data`` and return a dict of per-trial arrays (no label, no statistic over
  trials), so they may run before the cross-validation.
* A model family implements ``fit(X_train, y_train, groups_train, inner_splits, seed) -> state`` and
  ``logits(state, X) -> z``. ``fit`` is never given the test rows (the leak check of ``cv.run_experiment`` then
  exercises the prediction stage). Every ``state`` carries ``oof_z``: a cross-fitted logit for every training
  row, made by a model that did not see that row's person (``fold.inner_splits()``); calibrators and the
  attempt-aggregation scale are fitted on them, never on the test person.
* ``make_fit_predict(family, calib=...)`` turns a family into the ``fit_predict`` of ``cv.run_experiment``.
  Fitted states are cached on disk, keyed by a hash of everything ``fit`` receives (code, configuration,
  training arrays, labels, groups, splits, seed) and of the runtime that shapes the result (library versions;
  for the deep families the torch version and the pinned training thread count, ``runtime_signature``): a
  calibration variant, the saliency step or a re-run reuses the trained models instead of retraining them, and
  a change of data, code or runtime makes a new key.
* The cache settings of a ``fit_predict`` (``use_cache``, ``cache_dir``) hold for every fit it triggers, the
  candidate fits of a wrapper family (nested choice, grip subset) included (``cache_settings``).
* ``calib='nested'`` chooses the calibrator INSIDE the outer fold: every candidate of ``NESTED_CAL_CANDIDATES``
  is cross-fitted over the training persons' cross-fitted logits (GroupKFold by person), and the one with the
  lowest log loss there is refitted on all of them and applied to the held-out person.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import pickle
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from stage2 import config as C
from stage2.standardize import CHANNEL_INFO, CHANNELS, CHIRAL_CHANNELS, GRAVITY_CHANNELS

# ───────────────────────────── channels and mirror signs ─────────────────────────────

CHANNEL_SETS: dict[str, tuple[str, ...]] = {'all': CHANNELS, 'grav': GRAVITY_CHANNELS, 'chiral': CHIRAL_CHANNELS}
CHANNEL_SET_LABELS = {'all': 'all 17 channels', 'grav': 'gravity frame (8)', 'chiral': 'chiral only (2)'}
EPS = 1e-6


def channel_names(channels: str | Sequence[str]) -> tuple[str, ...]:
    return CHANNEL_SETS[channels] if isinstance(channels, str) else tuple(channels)


def channel_idx(channels: str | Sequence[str]) -> np.ndarray:
    return np.array([CHANNELS.index(c) for c in channel_names(channels)], dtype=int)


def mirror_signs(channels: str | Sequence[str]) -> np.ndarray:
    """Sign of each selected channel under the left/right mirror (standardize.CHANNEL_INFO)."""
    return np.array([CHANNEL_INFO[c][2] for c in channel_names(channels)], dtype=np.float32)


# ───────────────────────────── per-trial inputs (build_x) ─────────────────────────────


def x_approach(d: Any) -> dict[str, np.ndarray]:
    """The time-normalised approach (onset to end, 64 samples), all 17 channels."""
    return {'x': np.ascontiguousarray(d.approach, dtype=np.float32)}


def x_fixed(d: Any) -> dict[str, np.ndarray]:
    """The fixed 3-s window (-0.5 s .. +2.5 s around the onset, 180 samples at 60 Hz, absolute time) and its
    mask (False from the end cue on and outside the recording; values there are frozen by the cache)."""
    return {'x': np.ascontiguousarray(d.fixed, dtype=np.float32),
            'mask': np.ascontiguousarray(d.fixed_mask, dtype=np.float32)}


# ───────────────────────────── normalisers (fitted on training rows only) ─────────────────────────────


class Normaliser:
    """Per-channel scaling fitted on the TRAINING rows and their mirror images, so that it commutes with the
    mirror (the centre of every chiral channel is exactly 0). Modes:

    * ``none``   - raw units;
    * ``zscore`` - (x - mean) / sd per channel over trials and time (default);
    * ``robust`` - (x - median) / IQR per channel, then winsorised at +-``clip``;
    * ``trial``  - per-trial amplitude normalisation: every channel of a trial divided by its own RMS over
      time (a function of one trial; nothing is fitted), so speed and amplitude are removed.
    """

    def __init__(self, mode: str = 'zscore', clip: float = 5.0):
        if mode not in ('none', 'zscore', 'robust', 'trial'):
            raise ValueError(mode)
        self.mode, self.clip = mode, float(clip)
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray, signs: np.ndarray) -> 'Normaliser':
        if self.mode in ('none', 'trial'):
            return self
        shape = (1, -1) + (1,) * (X.ndim - 2)
        both = np.concatenate([X, X * signs.reshape(shape)], axis=0)        # training rows + their mirrors
        flat = np.moveaxis(both, 1, 0).reshape(both.shape[1], -1).astype(np.float64)
        if self.mode == 'zscore':
            c, s = flat.mean(axis=1), flat.std(axis=1)
        else:
            q25, c, q75 = np.percentile(flat, [25, 50, 75], axis=1)
            s = q75 - q25
        c[signs < 0] = 0.0                                                   # exact by symmetry
        self.center, self.scale = c.astype(np.float32), np.maximum(s, EPS).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mode == 'none':
            return X.astype(np.float32, copy=True)
        if self.mode == 'trial':
            rms = np.sqrt(np.mean(X.astype(np.float64) ** 2, axis=-1, keepdims=True))
            return (X / np.maximum(rms, 1e-3)).astype(np.float32)
        shape = (1, -1) + (1,) * (X.ndim - 2)
        Z = (X - self.center.reshape(shape)) / self.scale.reshape(shape)
        if self.mode == 'robust':
            Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)


# ───────────────────────────── hashing and the fold-fit cache ─────────────────────────────

FIT_CACHE_DIR = Path(os.environ.get('M3_FIT_CACHE', str(C.DATA_DIR / 'm3_fit_cache')))   # git-ignored (analysis/data/*)


def _hash_update(h: Any, obj: Any) -> None:
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj):
            h.update(str(k).encode())
            _hash_update(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b'[')
        for v in obj:
            _hash_update(h, v)
        h.update(b']')
    else:
        h.update(json.dumps(obj, sort_keys=True, default=str).encode())


def digest(*objs: Any) -> str:
    h = hashlib.sha256()
    for o in objs:
        _hash_update(h, o)
    return h.hexdigest()


def module_code_hash(*modules: Any) -> str:
    """Hash of the source of the modules that define a family (a code change makes a new cache key)."""
    h = hashlib.sha256()
    for m in modules:
        h.update(inspect.getsource(m).encode())
    return h.hexdigest()[:16]


def runtime_signature(family: Any) -> dict[str, Any]:
    """What besides code, configuration and inputs shapes a fit: the numerical libraries' versions and the
    family's own runtime (``family.runtime()``: for the deep families the torch version and the pinned training
    thread count). Part of the cache key and recorded in every result's per-fold info."""
    import scipy
    import sklearn
    sig: dict[str, Any] = {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    rt = getattr(family, 'runtime', None)
    if callable(rt):
        sig.update(rt())
    return sig


_CACHE: dict[str, Any] = {'use_cache': True, 'cache_dir': None}


@contextmanager
def cache_settings(use_cache: bool = True, cache_dir: str | Path | None = None) -> Iterator[None]:
    """Cache settings for every ``cached_fit`` inside the block, the inner fits of wrapper families included."""
    old = dict(_CACHE)
    _CACHE.update(use_cache=bool(use_cache), cache_dir=cache_dir)
    try:
        yield
    finally:
        _CACHE.clear()
        _CACHE.update(old)


def fit_cache_dir() -> Path:
    """The fit-cache folder in force: ``cache_settings(cache_dir=...)``, else ``FIT_CACHE_DIR`` ($M3_FIT_CACHE)."""
    return Path(_CACHE['cache_dir']) if _CACHE['cache_dir'] else FIT_CACHE_DIR


def cached_fit(family: Any, X: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
               splits: list[tuple[np.ndarray, np.ndarray]], seed: int, *, use_cache: bool | None = None,
               cache_only: bool = False) -> dict[str, Any]:
    """``family.fit`` memoised on disk. The key covers the family's code, configuration and runtime
    (``runtime_signature``) and every input of ``fit``; ``fit`` never receives test rows. ``use_cache=None``
    follows ``cache_settings``; ``cache_only`` raises instead of fitting (saliency step)."""
    use_cache = _CACHE['use_cache'] if use_cache is None else bool(use_cache)
    key = digest(family.code_hash(), family.config(), runtime_signature(family), X, y, [str(g) for g in groups],
                 [(a, b) for a, b in splits], int(seed))
    path = fit_cache_dir() / family.cache_name() / f'{key[:40]}.pkl'
    if use_cache and path.exists():
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            state['_cache'] = 'hit'
            return state
        except (OSError, EOFError, pickle.UnpicklingError):
            pass
    if cache_only:
        raise FileNotFoundError(f'{family.cache_name()}: no cached fit for this fold ({path.name}); run compute first')
    t0 = time.process_time()
    w0 = time.time()
    state = family.fit(X, y, groups, splits, seed)
    state['fit_cpu_s'] = float(time.process_time() - t0)
    state['fit_wall_s'] = float(time.time() - w0)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    state['_cache'] = 'miss'
    return state


# ───────────────────────────── calibrators (fitted on cross-fitted training logits) ─────────────────────────────

CALIBRATORS = ('none', 'platt', 'temperature', 'isotonic', 'beta', 'nested')
CALIB_LABELS = {'none': 'uncalibrated', 'platt': 'Platt', 'temperature': 'temperature', 'isotonic': 'isotonic',
                'beta': 'beta', 'nested': 'chosen in fold'}
#: the calibrators the in-fold choice (``calib='nested'``) compares, in tie-break order
NESTED_CAL_CANDIDATES = ('none', 'platt', 'temperature', 'isotonic', 'beta')
P_CLIP = 1e-4                         # calibrated probabilities are kept in [1e-4, 1 - 1e-4]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """T > 0 minimising the log loss of sigmoid(z / T) (keeps p(mirror) = 1 - p)."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: _nll(sigmoid(z / math.exp(lt)), y), bounds=(-4, 4), method='bounded')
    return float(math.exp(r.x))


def _logreg_1d(F: np.ndarray, y: np.ndarray, intercept: bool = True) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e4, fit_intercept=intercept, max_iter=1000).fit(F, y)
    return m.coef_.ravel().astype(float), float(m.intercept_[0]) if intercept else 0.0


def fit_calibrator(method: str, z: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> dict[str, Any]:
    """Parameters of a calibrator of logits z (cross-fitted training predictions) to labels y. ``'nested'`` needs
    the persons of the rows (``groups``): it cross-fits every candidate over them and keeps the best."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=int)
    if method == 'nested':
        return fit_nested_calibrator(z, y, groups)
    if method == 'none':
        return {'method': 'none'}
    if method == 'temperature':
        return {'method': method, 'T': fit_temperature(z, y)}
    if method == 'platt':
        w, b = _logreg_1d(z[:, None], y)
        return {'method': method, 'a': float(w[0]), 'b': b}
    if method == 'beta':                       # Kull et al. 2017: logit p = a ln p0 - b ln(1 - p0) + c, a, b >= 0
        p0 = np.clip(sigmoid(z), 1e-6, 1 - 1e-6)
        F = np.column_stack([np.log(p0), -np.log(1 - p0)])
        w, c = _logreg_1d(F, y)
        if w[0] < 0 or w[1] < 0:               # the betacal fallback: drop the negative term
            j = 1 if w[0] < 0 else 0
            wj, c = _logreg_1d(F[:, [j]], y)
            w = np.zeros(2)
            w[j] = max(wj[0], 0.0)
        return {'method': method, 'a': float(w[0]), 'b': float(w[1]), 'c': float(c)}
    if method == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        p0 = sigmoid(z)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(p0, y)
        return {'method': method, 'x': iso.X_thresholds_.astype(float).tolist(),
                'y': iso.y_thresholds_.astype(float).tolist()}
    raise ValueError(method)


def fit_nested_calibrator(z: np.ndarray, y: np.ndarray, groups: np.ndarray | None) -> dict[str, Any]:
    """The calibrator chosen inside the outer fold: each of ``NESTED_CAL_CANDIDATES`` is fitted on the logits of
    some training persons and scored (log loss) on the others (GroupKFold by person, ``cv.inner_splits``); the
    lowest wins (ties: candidate order) and is refitted on every training row. The held-out person is never seen."""
    from stage2 import cv
    if groups is None:
        raise ValueError("calib='nested' needs the persons of the training rows")
    g = np.asarray(groups).astype(str)
    p_cv = {m: np.full(z.size, np.nan) for m in NESTED_CAL_CANDIDATES}
    for tr, va in cv.inner_splits(g):
        for m in NESTED_CAL_CANDIDATES:
            p_cv[m][va] = apply_calibrator(fit_calibrator(m, z[tr], y[tr]), z[va])
    ll = {m: _nll(p, y) for m, p in p_cv.items()}
    best = min(NESTED_CAL_CANDIDATES, key=lambda m: (round(ll[m], 12), NESTED_CAL_CANDIDATES.index(m)))
    return {**fit_calibrator(best, z, y), 'nested_choice': best,
            'nested_inner_log_loss': {m: round(v, 6) for m, v in ll.items()}}


def apply_calibrator(par: dict[str, Any], z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = par['method']
    if m == 'none':
        p = sigmoid(z)
    elif m == 'temperature':
        p = sigmoid(z / par['T'])
    elif m == 'platt':
        p = sigmoid(par['a'] * z + par['b'])
    elif m == 'beta':
        p0 = np.clip(sigmoid(z), 1e-6, 1 - 1e-6)
        p = sigmoid(par['a'] * np.log(p0) - par['b'] * np.log(1 - p0) + par['c'])
    elif m == 'isotonic':
        p = np.interp(sigmoid(z), np.asarray(par['x']), np.asarray(par['y']))
    else:
        raise ValueError(m)
    return np.clip(p, P_CLIP, 1 - P_CLIP) if m != 'none' else p


# ───────────────────────────── attempt aggregation scale (fitted on training persons) ─────────────────────────────

K_MAX = max(C.K_ATTEMPTS)


def k_subsets(n: int, k: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """n_draws random k-subsets (rows) of range(n) (without replacement within a subset)."""
    return np.array([rng.choice(n, size=k, replace=False) for _ in range(n_draws)], dtype=int)


def fit_sum_scale(z: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int, seed: int, n_draws: int = 50) -> float:
    """c_k such that sigmoid(c_k * sum of k logits) is calibrated, from k-subsets of the SAME training person and
    hand (cross-fitted logits). c_1 = 1 / T; c_k < 1 when attempts are correlated (sum over-confident)."""
    rng = np.random.default_rng([seed, 7, k])
    zs, ys = [], []
    for g in np.unique(groups):
        for c in (0, 1):
            idx = np.flatnonzero((groups == g) & (y == c))
            if idx.size < k:
                continue
            sub = k_subsets(idx.size, k, n_draws, rng)
            zs.append(z[idx][sub].sum(axis=1))
            ys.append(np.full(len(sub), c))
    if not zs:
        return 1.0 / k
    S, Y = np.concatenate(zs), np.concatenate(ys)
    return fit_temperature(S, Y) ** -1


# ───────────────────────────── the fit_predict wrapper ─────────────────────────────


def _rows_dict(X: Any) -> dict[str, np.ndarray]:
    if isinstance(X, dict):
        return {k: np.asarray(v) for k, v in X.items()}
    return {'x': np.asarray(X)}


def make_fit_predict(family: Any, *, calib: str = 'none', use_cache: bool = True, cache_dir: str | Path | None = None,
                     agg_scales: bool = True) -> Callable[[Any], tuple[np.ndarray, dict[str, Any]]]:
    """fit_predict for ``cv.run_experiment``: fit (or load) the family on the training rows (with
    ``fold.inner_splits()`` for its inner models), predict the test rows, then calibrate with ``calib`` fitted
    on the cross-fitted training logits. info: training cost, inner choices, calibrator, aggregation scales and the
    runtime signature. ``use_cache=False`` never reads or writes the fit cache; ``cache_dir`` chooses its folder
    (default ``FIT_CACHE_DIR``); both hold for the inner fits of wrapper families too."""
    if calib not in CALIBRATORS:
        raise ValueError(calib)

    def fit_predict(fold: Any) -> tuple[np.ndarray, dict[str, Any]]:
        Xtr = _rows_dict(fold.X_train)
        ytr = np.asarray(fold.y_train).astype(int)
        gtr = np.asarray(fold.groups_train).astype(str)
        n_inner = getattr(family, 'n_inner', None)
        splits = fold.inner_splits(n_inner) if n_inner else fold.inner_splits()
        with cache_settings(use_cache, cache_dir):
            state = cached_fit(family, Xtr, ytr, gtr, splits, fold.seed)
        z = family.logits(state, _rows_dict(fold.X_test))
        par = fit_calibrator(calib, state['oof_z'], ytr, gtr)
        p = apply_calibrator(par, z)
        info: dict[str, Any] = {'fit_cpu_s': round(state.get('fit_cpu_s', float('nan')), 2),
                                'fit_wall_s': round(state.get('fit_wall_s', float('nan')), 2),
                                'calib': {k: (round(v, 6) if isinstance(v, float) else v) for k, v in par.items()
                                          if k not in ('x', 'y')},
                                'oof_log_loss': round(_nll(sigmoid(state['oof_z']), ytr), 5),
                                'runtime': runtime_signature(family)}
        info.update(state.get('info', {}))
        if agg_scales:
            info['sum_scale'] = {str(k): round(fit_sum_scale(state['oof_z'], ytr, gtr, k, fold.seed), 5)
                                 for k in range(1, K_MAX + 1)}
        return p, info

    fit_predict.family = family          # type: ignore[attr-defined]
    fit_predict.calib = calib            # type: ignore[attr-defined]
    return fit_predict
