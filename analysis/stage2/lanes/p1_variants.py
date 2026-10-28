"""Lane p1: re-standardised VARIANTS of the canonical arrays (the data-treatment options under test).

Each variant re-runs the core standardisation (``stage2.standardize``) from the raw records of the snapshot
(``motionlab.io.load_dataset``) with ONE parameter changed, and writes its own cache under
``analysis/data/stage2_variants/<variant>/`` (the core cache is never touched). Rows are matched by
``trial_id`` to the core trial table, so ``cv.run_experiment`` checks them against the curated persons.

Stream options (:class:`StreamSpec`, steps 2-4 of ``standardize``): low-pass cut-off, order, zero-phase or
causal; the longest gap bridged and the interpolation (linear, cubic spline, sample-and-hold); a lower
sampling rate (every k-th raw sample kept); Chrome-like 0.1 quantisation of iOS values; Gaussian jitter of
the raw sample times. Window options (:class:`WindowSpec`, steps 5-6): the gravity estimate (gyro-propagated
gravity_v1, a 1-Hz low-pass of the accelerometer with gravity, or the browser's own ``accg - acc``); the
window (approach, its first half, approach + hold, a 0.5-s pre-onset context, the fixed 3-s window); random
or systematic errors of the segment bounds.

With every option at its default the builder reproduces the core cache exactly (tests/test_p1_variants.py
and the ``fidelity`` record of every build). A variant is rebuilt when the core cache's data identity, this
module's code or the variant's specification changes (``meta.json`` of the variant).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from stage2 import config as C
from stage2 import standardize as S

VARIANT_DIR = C.DATA_DIR / 'stage2_variants'
#: the analysis sets / exclusion rules a variant cache holds (primary, natural use, and the E4b/E4c/E4d
#: exclusions that the sensitivity analyses put back)
EXTRA_RULES = ('E4b', 'E4c', 'E4d')


# ───────────────────────────── specifications ─────────────────────────────


@dataclass(frozen=True)
class StreamSpec:
    lowpass_hz: float | None = C.LOWPASS_HZ   # None: no low-pass
    order: int = C.LOWPASS_ORDER
    causal: bool = False                      # sosfilt (causal) instead of sosfiltfilt (zero phase)
    max_gap_ms: float = C.MAX_GAP_MS
    interp: str = 'linear'                    # 'linear' | 'cubic' | 'hold'
    decimate: int = 1                         # keep every k-th raw sample (60 Hz -> 60 / k Hz)
    quantize_ios: bool = C.QUANTIZE_IOS
    t_jitter_ms: float = 0.0                  # SD of Gaussian noise added to the raw sample times


@dataclass(frozen=True)
class WindowSpec:
    gravity: str = 'gyro'                     # 'gyro' (gravity_v1) | 'lowpass' (1 Hz low-pass of accg) | 'browser'
    window: str = 'approach'                  # 'approach' | 'first_half' | 'approach_hold' | 'pre_context' | 'fixed'
    bound_jitter_ms: float = 0.0              # onset and end each moved by U(-J, +J) (seeded per trial)
    onset_shift_ms: float = 0.0               # onset moved by a constant (a late detector: > 0)


@dataclass(frozen=True)
class Variant:
    name: str
    stream: StreamSpec = field(default_factory=StreamSpec)
    window: WindowSpec = field(default_factory=WindowSpec)
    description: str = ''

    def spec(self) -> dict[str, Any]:
        return {'name': self.name, 'stream': asdict(self.stream), 'window': asdict(self.window)}


PRE_CONTEXT_MS = 500.0          # the pre-onset context window starts this long before the onset
LOWPASS_GRAVITY_HZ = 1.0        # cut-off of the 'lowpass' gravity estimate (motionlab.estimate_gravity default)
MIN_JITTERED_MS = 150.0         # a jittered segment keeps at least this duration


def _v(name: str, desc: str, **kw: Any) -> Variant:
    sk = {k: v for k, v in kw.items() if k in StreamSpec.__dataclass_fields__}
    wk = {k: v for k, v in kw.items() if k in WindowSpec.__dataclass_fields__}
    if set(kw) - set(sk) - set(wk):
        raise ValueError(f'unknown variant options {sorted(set(kw) - set(sk) - set(wk))}')
    return Variant(name, StreamSpec(**sk), WindowSpec(**wk), desc)


#: every variant the experiments use (the default one reproduces the core cache)
VARIANTS: dict[str, Variant] = {v.name: v for v in [
    _v('base', 'the core pipeline (reproduces the core cache)'),
    _v('lp_none', 'no low-pass', lowpass_hz=None),
    _v('lp5', 'low-pass 5 Hz', lowpass_hz=5.0),
    _v('lp10', 'low-pass 10 Hz', lowpass_hz=10.0),
    _v('lp20', 'low-pass 20 Hz', lowpass_hz=20.0),
    _v('lp_order2', 'low-pass 15 Hz, order 2', order=2),
    _v('lp_order8', 'low-pass 15 Hz, order 8', order=8),
    _v('lp_causal', 'causal low-pass 15 Hz, order 4 (sosfilt)', causal=True),
    _v('gap50', 'gaps bridged up to 50 ms', max_gap_ms=50.0),
    _v('gap250', 'gaps bridged up to 250 ms', max_gap_ms=250.0),
    _v('interp_cubic', 'cubic-spline resampling and gap filling', interp='cubic'),
    _v('interp_hold', 'sample-and-hold resampling and gap filling', interp='hold'),
    _v('fs30', '30 Hz (every 2nd raw sample; low-pass 12 Hz)', decimate=2),
    _v('fs20', '20 Hz (every 3rd raw sample; low-pass 8 Hz)', decimate=3),
    _v('quant_ios', 'iOS values rounded to Chrome\'s 0.1 grid', quantize_ios=True),
    _v('tjit2', 'raw sample times jittered, SD 2 ms', t_jitter_ms=2.0),
    _v('tjit5', 'raw sample times jittered, SD 5 ms', t_jitter_ms=5.0),
    _v('tjit10', 'raw sample times jittered, SD 10 ms', t_jitter_ms=10.0),
    _v('grav_lowpass', f'gravity = {LOWPASS_GRAVITY_HZ:g}-Hz low-pass of the accelerometer with gravity', gravity='lowpass'),
    _v('grav_browser', 'gravity = the browser\'s accg - acc', gravity='browser'),
    _v('win_first_half', 'first half of the approach (onset to mid-point)', window='first_half'),
    _v('win_hold', 'approach + hold (onset to the end cue)', window='approach_hold'),
    _v('win_pre', f'{PRE_CONTEXT_MS / 1000:g}-s pre-onset context + approach', window='pre_context'),
    _v('win_fixed', 'fixed 3-s window at 60 Hz (absolute time)', window='fixed'),
    _v('bjit50', 'segment bounds jittered by U(-50, +50) ms', bound_jitter_ms=50.0),
    _v('bjit100', 'segment bounds jittered by U(-100, +100) ms', bound_jitter_ms=100.0),
    _v('bjit200', 'segment bounds jittered by U(-200, +200) ms', bound_jitter_ms=200.0),
    _v('onset_late100', 'onset 100 ms late (a late detector)', onset_shift_ms=100.0),
]}


def fs_of(spec: StreamSpec) -> float:
    return C.FS_HZ / spec.decimate


def lowpass_of(spec: StreamSpec) -> float | None:
    """The cut-off used: the requested one, capped at 0.4 x the sampling rate for a decimated stream."""
    if spec.lowpass_hz is None:
        return None
    return min(spec.lowpass_hz, 0.4 * fs_of(spec)) if spec.decimate > 1 else spec.lowpass_hz


# ───────────────────────────── stream (steps 1-4) ─────────────────────────────


def _resample(t_ms: np.ndarray, values: np.ndarray, hz: float, max_gap_ms: float, interp: str
              ) -> tuple[np.ndarray, np.ndarray]:
    """motionlab.resample_uniform for 'linear'; the same grid and gap mask with a cubic spline or a
    sample-and-hold for the values otherwise."""
    from motionlab.preprocess import _prepare_times, resample_uniform, uniform_grid
    if interp == 'linear':
        return resample_uniform(t_ms, values, hz, max_gap_ms)
    t, X, _ = _prepare_times(t_ms, values)
    tn = uniform_grid(float(t[0]), float(t[-1]), hz)
    out = np.full((tn.size, X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j])
        tv, xv = t[ok], X[ok, j]
        if tv.size < 2:
            continue
        r = np.searchsorted(tv, tn, side='left')
        rc = np.clip(r, 0, tv.size - 1)
        exact = (r < tv.size) & (tv[rc] == tn)
        lc = np.clip(r - 1, 0, tv.size - 1)
        inside = (r < tv.size) & (r >= 1)
        gap = np.where(inside, tv[rc] - tv[lc], np.inf)
        valid = exact | (inside & (gap <= max_gap_ms))
        if interp == 'cubic':
            vals = CubicSpline(tv, xv)(tn)
        elif interp == 'hold':
            k = np.clip(np.searchsorted(tv, tn, side='right') - 1, 0, tv.size - 1)
            vals = xv[k]
        else:
            raise ValueError(f'unknown interpolation {interp!r}')
        out[valid, j] = vals[valid]
    return tn, out


def _fill(t: np.ndarray, x: np.ndarray, interp: str) -> np.ndarray:
    """Fill the NaN rows (long gaps) like motionlab.fill_gaps_linear, with the variant's interpolation."""
    from motionlab.preprocess import fill_gaps_linear
    if interp == 'linear':
        return fill_gaps_linear(t, x)
    X = np.array(x, dtype=float, copy=True)
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j])
        if ok.all() or not ok.any():
            continue
        if interp == 'cubic' and ok.sum() >= 4:
            X[~ok, j] = CubicSpline(t[ok], X[ok, j], extrapolate=False)(t[~ok])
            bad = ~np.isfinite(X[:, j])
            if bad.any():   # edges: hold the nearest value (as fill_gaps_linear does)
                X[bad, j] = np.interp(t[bad], t[~bad], X[~bad, j])
        else:
            okt = t[ok]
            k = np.clip(np.searchsorted(okt, t[~ok], side='right') - 1, 0, okt.size - 1)
            X[~ok, j] = X[ok, j][k]
    return X


def _lowpass(U: np.ndarray, fs: float, cutoff: float | None, order: int, causal: bool) -> np.ndarray:
    """motionlab.lowpass (zero phase, run by run over finite samples) or its causal counterpart (sosfilt with
    the steady-state initial condition of each run's first sample)."""
    from motionlab.preprocess import finite_runs, lowpass
    if not cutoff:
        return U
    if not causal:
        return lowpass(U, fs, cutoff, order=order)
    X = np.array(U, dtype=float, copy=True)
    if cutoff >= fs / 2:
        return X
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    zi0 = signal.sosfilt_zi(sos)
    for j in range(X.shape[1]):
        col = X[:, j]
        for a, b in finite_runs(np.isfinite(col)):
            if b - a < 3:
                continue
            X[a:b, j], _ = signal.sosfilt(sos, col[a:b], zi=zi0 * col[a])
    return X


def make_stream(record: dict[str, Any], norm: Any, spec: StreamSpec, *, seed: int = C.SEED) -> S.Stream:
    """``standardize.stream_from_record`` with the variant's options (identical with the defaults)."""
    from motionlab.conventions import Normalization, normalize_motion
    from motionlab.io import streams_from_record
    from motionlab.preprocess import quantize_like_chrome, repeated_sample_mask
    nm = norm or Normalization()
    s = streams_from_record(record)
    t_raw, acc_raw, accg_raw, rr_raw = s.t, s.acc, s.accg, s.rr
    if spec.decimate > 1:
        sl = slice(0, None, int(spec.decimate))
        t_raw, acc_raw, accg_raw, rr_raw = t_raw[sl], acc_raw[sl], accg_raw[sl], rr_raw[sl]
    quantized = False
    if spec.quantize_ios and nm.engine == 'webkit':
        acc_raw = quantize_like_chrome(acc_raw, 'acc')
        accg_raw = quantize_like_chrome(accg_raw, 'acc')
        rr_raw = quantize_like_chrome(rr_raw, 'rr')
        quantized = True
    sig = normalize_motion(t_raw, acc_raw, accg_raw, rr_raw, nm)
    acc, accg, gyro = sig.acc, sig.accg, sig.gyro
    bg = ((record.get('segmentation') or {}).get('baseline') or {}).get('gravity')
    base_g = None
    if bg is not None and len(bg) == 3 and all(v is not None for v in bg):
        base_g = np.asarray(bg, dtype=float) * nm.sign
    keep = np.ones(t_raw.size, dtype=bool)
    n_rep = 0
    if t_raw.size > 1:
        rep = repeated_sample_mask(acc_raw, accg_raw, rr_raw, engine=nm.engine)
        n_rep = int(rep.sum())
        keep = ~rep
    if keep.sum() < 5:
        raise ValueError(f"trial {record.get('trial_id')}: fewer than 5 motion samples")
    t_use = np.asarray(t_raw, dtype=float)
    if spec.t_jitter_ms > 0:
        rng = np.random.default_rng([seed, zlib.crc32(str(record.get('trial_id')).encode()), 3])
        t_use = t_use + rng.normal(0.0, spec.t_jitter_ms, t_use.size)
    fs = fs_of(spec)
    max_gap = max(spec.max_gap_ms, 2.5 * 1000.0 / fs) if spec.decimate > 1 else spec.max_gap_ms
    tu, U = _resample(t_use[keep], np.column_stack([accg, acc, gyro])[keep], fs, max_gap, spec.interp)
    valid = np.isfinite(U[:, 0:3]).all(axis=1) & np.isfinite(U[:, 6:9]).all(axis=1)
    U = _lowpass(U, fs, lowpass_of(spec), spec.order, spec.causal)
    return S.Stream(trial_id=record.get('trial_id'), t=tu, f=U[:, 0:3], acc=U[:, 3:6], gyro=U[:, 6:9], valid=valid,
                    fs_hz=fs, gravity_magnitude=nm.gravity_magnitude, baseline_gravity=base_g, engine=nm.engine,
                    quantized=quantized, repeats_dropped=n_rep, t_raw=np.asarray(t_raw, dtype=float))


# ───────────────────────────── canonical channels (step 5) ─────────────────────────────


def make_canonical(stream: S.Stream, onset_ms: float, end_ms: float, gravity: str = 'gyro',
                   interp: str = 'linear') -> S.Canonical:
    """``standardize.canonical`` (exactly, for gravity 'gyro' and linear filling) or the same channels from
    another gravity estimate: 'lowpass' (a 1-Hz zero-phase low-pass of the specific force) or 'browser'
    (accg - acc, the platform's own sensor fusion; low-pass where acc is missing)."""
    if gravity == 'gyro' and interp == 'linear':
        return S.canonical(stream, onset_ms, end_ms)
    from motionlab.preprocess import angle_deg, estimate_gravity, gyro_gravity, unit_rows
    t = stream.t
    i_on = max(int(np.searchsorted(t, onset_ms, side='left')), 0)
    i_end = min(int(np.searchsorted(t, end_ms, side='right')) - 1, t.size - 1)
    if not (math.isfinite(onset_ms) and math.isfinite(end_ms)) or end_ms <= onset_ms or i_end - i_on < 2:
        raise ValueError(f'invalid bounds {onset_ms}..{end_ms}')
    f = _fill(t, stream.f, interp)
    w = _fill(t, stream.gyro, interp)
    closure, pre_n, post_n, method = math.nan, 0, 0, gravity
    if gravity == 'gyro':
        gg = gyro_gravity(t, f, w, i_on, i_end, still_dps=C.STILL_GYRO_DPS, rest_ms=C.REST_MS,
                          magnitude=stream.gravity_magnitude, g_pre_fallback=stream.baseline_gravity)
        g, closure, pre_n, post_n, method = gg.g, gg.closure_deg, gg.pre_n, gg.post_n, gg.method
    elif gravity == 'lowpass':
        g, _ = estimate_gravity(np.full_like(f, np.nan), f, stream.fs_hz, cutoff_hz=LOWPASS_GRAVITY_HZ, method='lowpass')
    elif gravity == 'browser':
        a_br = _fill(t, stream.acc, interp) if np.isfinite(stream.acc).any() else stream.acc
        g, _ = estimate_gravity(a_br, f, stream.fs_hz, cutoff_hz=LOWPASS_GRAVITY_HZ, method='difference')
    else:
        raise ValueError(f'unknown gravity estimate {gravity!r}')
    u = unit_rows(g)
    a = f - g
    wv = np.sum(w * u, axis=1)
    av = np.sum(a * u, axis=1)
    ah = np.linalg.norm(a - av[:, None] * u, axis=1)
    wh = np.linalg.norm(w - wv[:, None] * u, axis=1)
    yaw = cumulative_trapezoid(wv, t / 1000.0, initial=0.0)
    yaw = yaw - yaw[i_on]
    tilt = angle_deg(u, np.broadcast_to(u[i_on], u.shape))
    ch = {'acc_x': a[:, 0], 'acc_y': a[:, 1], 'acc_z': a[:, 2], 'gyro_x': w[:, 0], 'gyro_y': w[:, 1],
          'gyro_z': w[:, 2], 'up_x': u[:, 0], 'up_y': u[:, 1], 'up_z': u[:, 2],
          'gyro_mag': np.linalg.norm(w, axis=1), 'acc_mag': np.linalg.norm(a, axis=1), 'acc_vert': av,
          'acc_horiz': ah, 'yaw_rate': wv, 'gyro_horiz': wh, 'yaw_cum': yaw, 'tilt_from_start': tilt}
    X = np.stack([ch[c] for c in S.CHANNELS]).astype(np.float64)
    return S.Canonical(t, X, g, stream.valid, float(onset_ms), float(end_ms), i_on, i_end, float(closure),
                       int(pre_n), int(post_n), method)


# ───────────────────────────── one trial, one variant ─────────────────────────────


def bounds(trial: dict[str, Any], win: WindowSpec, seed: int = C.SEED) -> tuple[float, float]:
    """The (possibly perturbed) approach bounds of one trial: oracle onset / end, jittered or shifted."""
    on, en = float(trial['onset_ms']), float(trial['end_ms'])
    if win.bound_jitter_ms > 0:
        rng = np.random.default_rng([seed, zlib.crc32(str(trial['trial_id']).encode()), int(win.bound_jitter_ms)])
        d_on, d_en = rng.uniform(-win.bound_jitter_ms, win.bound_jitter_ms, 2)
        on, en = on + d_on, en + d_en
    on += win.onset_shift_ms
    if en - on < MIN_JITTERED_MS:
        en = on + MIN_JITTERED_MS
    return on, en


def trial_array(record: dict[str, Any], norm: Any, trial: dict[str, Any], v: Variant) -> np.ndarray:
    """(C, T) float32 array of one trial under variant ``v`` (T = 64, or 180 for the fixed window)."""
    st = make_stream(record, norm, v.stream)
    on, en = bounds(trial, v.window)
    can = make_canonical(st, on, en, v.window.gravity, v.stream.interp)
    w = v.window.window
    if w == 'approach':
        return S.time_normalised(can, on, en)
    if w == 'first_half':
        return S.time_normalised(can, on, 0.5 * (on + en))
    if w == 'approach_hold':
        cue = trial.get('end_detected_ms')
        cue = float(cue) if cue is not None and np.isfinite(float(cue)) else en + 250.0
        return S.time_normalised(can, on, max(cue, en))
    if w == 'pre_context':
        return S.time_normalised(can, on - PRE_CONTEXT_MS, en)
    if w == 'fixed':
        cue = trial.get('end_detected_ms')
        cue = float(cue) if cue is not None and np.isfinite(float(cue)) else None
        return S.fixed_window(can, on, cut_ms=cue)[0]
    raise ValueError(f'unknown window {w!r}')


# ───────────────────────────── the variant cache ─────────────────────────────


def builder_sha256() -> str:
    """The code that shapes a variant's arrays: this module, the core standardisation and motionlab."""
    h = hashlib.sha256()
    for p in (Path(__file__), C.ANALYSIS_DIR / 'stage2' / 'standardize.py'):
        h.update(p.read_bytes())
    h.update(C.code_fingerprint()['motionlab_sha256'].encode())
    return h.hexdigest()


def core_data_id() -> str | None:
    from stage2.pipeline import cache_data_id
    return cache_data_id()


def variant_trials() -> Any:
    """The trial rows every variant cache holds: the primary and natural-use sets and the instructed attempts
    excluded by rules E4b, E4c and E4d (for the sensitivity analyses), in the core cache's order."""
    from stage2.data import load_cache
    t = load_cache(sets=None).trials
    keep = t['analysis_set'].isin(['primary', 'natural_use']) | (
        t['excl_rule'].isin(EXTRA_RULES) & (t['hand_source'].astype(str) != 'reported'))
    return t[keep].reset_index(drop=True)


def _meta_path(name: str) -> Path:
    return VARIANT_DIR / name / 'meta.json'


def expected_identity(v: Variant) -> dict[str, Any]:
    return {'variant': v.spec(), 'core_data_id': core_data_id(), 'builder_sha256': builder_sha256()}


def is_current(name: str) -> bool:
    p = _meta_path(name)
    if not p.exists() or not (VARIANT_DIR / name / 'arrays.npz').exists():
        return False
    try:
        meta = json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    return meta.get('identity') == expected_identity(VARIANTS[name])


def array_sha(name: str) -> str | None:
    p = _meta_path(name)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8')).get('arrays_sha256')


_DS: dict[str, Any] = {}


def _dataset() -> Any:
    if 'ds' not in _DS:
        from motionlab.io import load_dataset
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _DS['ds'] = load_dataset(C.SNAPSHOT_DIR, simulated='include')
    return _DS['ds']


def build(name: str, *, force: bool = False, progress: bool = True) -> Path:
    """Build (or keep, when current) the cache of variant ``name``."""
    v = VARIANTS[name]
    out = VARIANT_DIR / name
    if not force and is_current(name):
        return out
    t0 = time.time()
    trials = variant_trials()
    ds = _dataset()
    arrs, failed = [], []
    for _, r in trials.iterrows():
        tid = r['trial_id']
        rec = ds.record(tid, compact=True)
        nm = ds.trial_normalization(tid)
        try:
            a = trial_array(rec, nm, r.to_dict(), v)
            if not np.all(np.isfinite(a)):
                raise ValueError('non-finite values')
        except ValueError as e:
            failed.append({'trial_key': r['trial_key'], 'error': str(e)[:200]})
            a = None
        arrs.append(a)
    shape = next(a.shape for a in arrs if a is not None)
    X = np.stack([a if a is not None else np.full(shape, np.nan, np.float32) for a in arrs]).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'arrays.npz', trial_id=trials['trial_id'].to_numpy().astype(str), X=X)
    sha = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    meta = {'identity': expected_identity(v), 'description': v.description, 'n_trials': int(len(trials)),
            'shape': list(X.shape), 'failed': failed, 'arrays_sha256': sha,
            'build_seconds': round(time.time() - t0, 1)}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    if progress:
        print(f'  variant {name}: {X.shape} in {time.time() - t0:.0f} s ({len(failed)} failed)', flush=True)
    return out


def load(name: str, trial_ids: Any) -> np.ndarray:
    """The variant's arrays for these trial ids, in their order (raises if one is missing or failed)."""
    with np.load(VARIANT_DIR / name / 'arrays.npz') as z:
        ids = z['trial_id'].astype(str)
        X = z['X']
    pos = {t: i for i, t in enumerate(ids)}
    want = [str(t) for t in trial_ids]
    missing = [t for t in want if t not in pos]
    if missing:
        raise KeyError(f'variant {name}: {len(missing)} trials missing (rebuild it)')
    out = X[[pos[t] for t in want]]
    if not np.all(np.isfinite(out)):
        raise ValueError(f'variant {name}: non-finite arrays for some requested trials')
    return out


def fidelity(name: str = 'base') -> dict[str, Any]:
    """Max |difference| between a variant built with the default options and the core cache (should be 0)."""
    from stage2.data import load_cache
    d = load_cache(sets=None)
    t = variant_trials()
    sel = d.trials['trial_id'].isin(t['trial_id'])
    Xc = (d.fixed if VARIANTS[name].window.window == 'fixed' else d.approach)[sel.to_numpy()]
    Xv = load(name, d.trials.loc[sel, 'trial_id'])
    return {'variant': name, 'n_trials': int(sel.sum()), 'max_abs_diff': float(np.max(np.abs(Xv - Xc)))}


def describe_variants() -> list[dict[str, Any]]:
    return [{'variant': v.name, 'description': v.description, **{f's_{k}': x for k, x in asdict(v.stream).items()},
             **{f'w_{k}': x for k, x in asdict(v.window).items()}} for v in VARIANTS.values()]


__all__ = ['VARIANTS', 'Variant', 'StreamSpec', 'WindowSpec', 'build', 'load', 'is_current', 'array_sha',
           'fidelity', 'make_stream', 'make_canonical', 'trial_array', 'bounds', 'variant_trials', 'replace']
