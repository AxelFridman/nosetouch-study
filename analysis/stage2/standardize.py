"""The canonical per-trial representation of Stage 2 (one code path for oracle and detected windows).

Pipeline (every decision is justified with a number or figure in ``stage2/README.md``):

1. **Units and frame.** ``motionlab`` norm_v2 per session: the accelerometer sign of the platform (WebKit
   reports acc / accg negated; the session sign comes from the guided calibration or the engine prior
   confirmed by the trial votes) and the rotationRate mapping (W3C: alpha, beta, gamma = rates about
   device x, y, z; verified per session by the gyro-gravity closure). Result: specific force f (= accg,
   W3C sign: up at rest) and the browser's linear acceleration in m/s^2, angular velocity w in deg/s,
   all in the W3C DEVICE frame (x right edge, y top edge, z out of the screen).
2. **Precision.** Native precision on both platforms by default (Chrome rounds to 0.1 m/s^2 / 0.1 deg/s,
   WebKit does not). ``quantize_ios=True`` applies Chrome's rounding to iOS values (ablation A3).
3. **Repeats and resampling.** Isolated duplicated snapshots are dropped, then every channel is linearly
   interpolated onto a uniform 60 Hz grid (the native browser rate). Gaps up to ``MAX_GAP_MS`` = 100 ms
   are bridged; grid points inside longer gaps are NaN (``Stream.valid`` False).
4. **Low-pass.** Zero-phase Butterworth, order 4, 15 Hz, run by run over the finite samples (no NaN
   smearing). Zero phase is legitimate in production too: the predictor processes a recorded buffer.
5. **Gravity and the heading-free frame** (:func:`canonical`, given movement bounds): gravity_v1 - the
   rest gravity before the onset and after the end, propagated with the gyroscope and blended in time
   (``motionlab.preprocess.gyro_gravity``); a = f - g, u = g / |g| (unit up). No gyro bias is subtracted
   (norm_v2; production has no rest calibration either). Heading-free channels: vertical and horizontal
   parts of a and w, the yaw rate w.u (rotation about gravity, a pseudoscalar), its integral, the tilt
   from the start pose and the magnitudes.
6. **Windows** cut from those channels: the time-normalised window (``N_NORM`` = 64 samples between two
   bounds), the fixed window around the onset (-0.5 s .. +2.5 s at 60 Hz; samples after the end cue are
   frozen and masked so the return of the arm never enters), and the return window (time-normalised,
   labelled separately).

The same :func:`canonical` runs on oracle bounds (training) and on bounds found by a cue-free detector
(production): only the bounds differ.

Channel arrays are (C, T) float32 in :data:`CHANNELS` order. :data:`CHANNEL_INFO` gives each channel's
unit, frame and its sign under the left/right mirror (polar a -> S a, pseudovector w -> -S w with
S = diag(-1, 1, 1); pseudoscalars flip) and under the screen flip (F = diag(-1, 1, -1) for every vector),
so augmentation can act on cached arrays directly (:func:`mirror_channels`, :func:`screen_flip_channels`).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid

from motionlab.conventions import (
    Normalization, flip_device_vectors, mirror_device_vectors, normalize_motion,
)
from motionlab.io import streams_from_record
from motionlab.preprocess import (
    angle_deg, fill_gaps_linear, gyro_gravity, lowpass, quantize_like_chrome, repeated_sample_mask,
    resample_uniform, unit_rows,
)

from . import config as C

# ───────────────────────────── channels ─────────────────────────────

#: Device-frame channels (grip-dependent) then gravity-frame channels (heading-free, grip-invariant).
CHANNELS: tuple[str, ...] = (
    'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'up_x', 'up_y', 'up_z',
    'gyro_mag', 'acc_mag', 'acc_vert', 'acc_horiz', 'yaw_rate', 'gyro_horiz', 'yaw_cum', 'tilt_from_start',
)
DEVICE_CHANNELS: tuple[str, ...] = CHANNELS[:9]
GRAVITY_CHANNELS: tuple[str, ...] = CHANNELS[9:]

#: name -> (frame, unit, sign under the left/right mirror, sign under the screen flip, description)
CHANNEL_INFO: dict[str, tuple[str, str, int, int, str]] = {
    'acc_x': ('device', 'm/s^2', -1, -1, 'linear acceleration a = f - g along device x (right edge)'),
    'acc_y': ('device', 'm/s^2', +1, +1, 'linear acceleration along device y (top edge)'),
    'acc_z': ('device', 'm/s^2', +1, -1, 'linear acceleration along device z (out of the screen)'),
    'gyro_x': ('device', 'deg/s', +1, -1, 'angular velocity about device x'),
    'gyro_y': ('device', 'deg/s', -1, +1, 'angular velocity about device y'),
    'gyro_z': ('device', 'deg/s', -1, -1, 'angular velocity about device z'),
    'up_x': ('device', '1', -1, -1, 'unit up vector u = g/|g| (gyro-propagated gravity), device x'),
    'up_y': ('device', '1', +1, +1, 'unit up vector, device y'),
    'up_z': ('device', '1', +1, -1, 'unit up vector, device z'),
    'gyro_mag': ('gravity', 'deg/s', +1, +1, '|w|'),
    'acc_mag': ('gravity', 'm/s^2', +1, +1, '|a|'),
    'acc_vert': ('gravity', 'm/s^2', +1, +1, 'a.u: vertical linear acceleration (up positive)'),
    'acc_horiz': ('gravity', 'm/s^2', +1, +1, '|a - (a.u)u|: horizontal linear acceleration'),
    'yaw_rate': ('gravity', 'deg/s', -1, +1, 'w.u: yaw rate about gravity (pseudoscalar; + = counter-clockwise seen from above)'),
    'gyro_horiz': ('gravity', 'deg/s', +1, +1, '|w - (w.u)u|: tilting rate'),
    'yaw_cum': ('gravity', 'deg', -1, +1, 'integral of w.u from the window onset (pseudoscalar)'),
    'tilt_from_start': ('gravity', 'deg', +1, +1, 'angle between u(t) and u at the window onset'),
}
MIRROR_SIGN = np.array([CHANNEL_INFO[c][2] for c in CHANNELS], dtype=np.float32)
SCREEN_FLIP_SIGN = np.array([CHANNEL_INFO[c][3] for c in CHANNELS], dtype=np.float32)
#: Channels that change sign under the mirror and are grip-invariant: the chiral evidence.
CHIRAL_CHANNELS: tuple[str, ...] = tuple(c for c in GRAVITY_CHANNELS if CHANNEL_INFO[c][2] < 0)

#: Phase codes of a sample (fixed windows and full streams), from the trial's own events.
PHASES: dict[int, str] = {0: 'pre', 1: 'approach', 2: 'hold', 3: 'post_cue', 4: 'return', 5: 'after_return'}


def channel_index(names: str | Sequence[str]) -> list[int]:
    """Row indices of channels in the (C, T) arrays."""
    if isinstance(names, str):
        names = [names]
    return [CHANNELS.index(n) for n in names]


def mirror_channels(X: np.ndarray, axis: int = -2) -> np.ndarray:
    """Left/right mirror image of canonical arrays: multiply each channel by its mirror sign.

    Exact for every channel (all are computed with rotation- and mirror-covariant operations); the
    label must be flipped by the caller. ``axis`` is the channel axis ((N, C, T) -> -2)."""
    shape = [1] * X.ndim
    shape[axis] = len(CHANNELS)
    return X * MIRROR_SIGN.reshape(shape)


def screen_flip_channels(X: np.ndarray, axis: int = -2) -> np.ndarray:
    """The same movement with the phone turned 180 deg about its y axis (screen away); label unchanged.
    A synthetic grip for training augmentation only (docs/05 section 3.6)."""
    shape = [1] * X.ndim
    shape[axis] = len(CHANNELS)
    return X * SCREEN_FLIP_SIGN.reshape(shape)


# ───────────────────────────── stream ─────────────────────────────


@dataclass
class Stream:
    """A whole recording on the uniform grid, normalised and low-passed (no window, no bounds).

    t: (n,) ms on the trial clock (0 = recording start); f: specific force (accg, W3C sign, up at rest);
    acc: the browser's own linear acceleration (QC and cue-free detection only; NaN when missing);
    gyro: angular velocity about device x, y, z (deg/s); valid: True where f and gyro were measured
    (False inside gaps longer than MAX_GAP_MS and outside the recording)."""

    trial_id: str | None
    t: np.ndarray
    f: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray
    valid: np.ndarray
    fs_hz: float
    gravity_magnitude: float | None = None
    baseline_gravity: np.ndarray | None = None  # online baseline gravity (normalised), gravity_v1 fallback
    engine: str | None = None
    quantized: bool = False
    mirrored: bool = False
    screen_flipped: bool = False
    repeats_dropped: int = 0
    t_raw: np.ndarray = field(default_factory=lambda: np.zeros(0))  # raw sample times (ms), for gap QC

    @property
    def n(self) -> int:
        return int(self.t.size)


def stream_from_record(
    record: Mapping[str, Any],
    norm: Normalization | None,
    *,
    quantize_ios: bool = C.QUANTIZE_IOS,
    mirror: bool = False,
    screen_flip: bool = False,
    fs_hz: float = C.FS_HZ,
    lowpass_hz: float | None = C.LOWPASS_HZ,
    max_gap_ms: float = C.MAX_GAP_MS,
) -> Stream:
    """Steps 1-4 of the module docstring for one TrialRecord (``norm`` = ``Dataset.trial_normalization``)."""
    nm = norm or Normalization()
    s = streams_from_record(record)
    t_raw = s.t
    acc_raw, accg_raw, rr_raw = s.acc, s.accg, s.rr
    quantized = False
    if quantize_ios and nm.engine == 'webkit':
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
    if mirror:
        acc, gyro = mirror_device_vectors(acc, gyro)
        accg = mirror_device_vectors(accg, None)[0]
        base_g = None if base_g is None else mirror_device_vectors(base_g, None)[0]
    if screen_flip:
        acc, gyro = flip_device_vectors(acc, gyro)
        accg = flip_device_vectors(accg, None)[0]
        base_g = None if base_g is None else flip_device_vectors(base_g, None)[0]
    keep = np.ones(t_raw.size, dtype=bool)
    n_rep = 0
    if t_raw.size > 1:
        rep = repeated_sample_mask(acc_raw, accg_raw, rr_raw, engine=nm.engine)
        n_rep = int(rep.sum())
        keep = ~rep
    if keep.sum() < 5:
        raise ValueError(f"trial {record.get('trial_id')}: fewer than 5 motion samples")
    tu, U = resample_uniform(t_raw[keep], np.column_stack([accg, acc, gyro])[keep], fs_hz, max_gap_ms)
    valid = np.isfinite(U[:, 0:3]).all(axis=1) & np.isfinite(U[:, 6:9]).all(axis=1)
    if lowpass_hz:
        U = lowpass(U, fs_hz, lowpass_hz, order=C.LOWPASS_ORDER)
    return Stream(
        trial_id=record.get('trial_id'), t=tu, f=U[:, 0:3], acc=U[:, 3:6], gyro=U[:, 6:9], valid=valid,
        fs_hz=fs_hz, gravity_magnitude=nm.gravity_magnitude, baseline_gravity=base_g, engine=nm.engine,
        quantized=quantized, mirrored=mirror, screen_flipped=screen_flip, repeats_dropped=n_rep, t_raw=t_raw,
    )


# ───────────────────────────── canonical channels for given bounds ─────────────────────────────


@dataclass
class Canonical:
    """Canonical channels (C, n) on the stream grid for one pair of movement bounds.

    g: gravity estimate (n,3) (gyro-propagated from the rest windows around the bounds); closure_deg:
    angle between the two rest gravities in one frame (QA: > 3 deg suggests a bias, a wrong axis mapping
    or a rest that was not still); rest_pre_n / rest_post_n: samples in the rest windows (0 = fallback);
    i_on / i_end: grid indices of the bounds."""

    t: np.ndarray
    X: np.ndarray
    g: np.ndarray
    valid: np.ndarray
    onset_ms: float
    end_ms: float
    i_on: int
    i_end: int
    closure_deg: float
    rest_pre_n: int
    rest_post_n: int
    gravity_method: str

    def channel(self, name: str) -> np.ndarray:
        return self.X[CHANNELS.index(name)]


def canonical(
    stream: Stream,
    onset_ms: float,
    end_ms: float,
    *,
    still_dps: float = C.STILL_GYRO_DPS,
    rest_ms: float = C.REST_MS,
) -> Canonical:
    """Step 5: gravity_v1 from the rest windows around [onset_ms, end_ms], then every channel.

    ``yaw_cum`` and ``tilt_from_start`` are referenced to the onset (0 there). Raises ValueError when the
    bounds hold fewer than 3 grid samples."""
    t = stream.t
    if not (math.isfinite(onset_ms) and math.isfinite(end_ms)) or end_ms <= onset_ms:
        raise ValueError(f'invalid bounds {onset_ms}..{end_ms}')
    i_on = int(np.searchsorted(t, onset_ms, side='left'))
    i_end = int(np.searchsorted(t, end_ms, side='right')) - 1
    i_on, i_end = max(i_on, 0), min(i_end, t.size - 1)
    if i_end - i_on < 2:
        raise ValueError(f'bounds {onset_ms:.0f}..{end_ms:.0f} ms hold fewer than 3 grid samples')
    f = fill_gaps_linear(t, stream.f)
    w = fill_gaps_linear(t, stream.gyro)
    gg = gyro_gravity(t, f, w, i_on, i_end, still_dps=still_dps, rest_ms=rest_ms,
                      magnitude=stream.gravity_magnitude, g_pre_fallback=stream.baseline_gravity)
    g = gg.g
    u = unit_rows(g)
    a = f - g
    wv = np.sum(w * u, axis=1)
    av = np.sum(a * u, axis=1)
    ah = np.linalg.norm(a - av[:, None] * u, axis=1)
    wh = np.linalg.norm(w - wv[:, None] * u, axis=1)
    ts = t / 1000.0
    yaw = cumulative_trapezoid(wv, ts, initial=0.0)
    yaw = yaw - yaw[i_on]
    tilt = angle_deg(u, np.broadcast_to(u[i_on], u.shape))
    ch = {
        'acc_x': a[:, 0], 'acc_y': a[:, 1], 'acc_z': a[:, 2],
        'gyro_x': w[:, 0], 'gyro_y': w[:, 1], 'gyro_z': w[:, 2],
        'up_x': u[:, 0], 'up_y': u[:, 1], 'up_z': u[:, 2],
        'gyro_mag': np.linalg.norm(w, axis=1), 'acc_mag': np.linalg.norm(a, axis=1),
        'acc_vert': av, 'acc_horiz': ah, 'yaw_rate': wv, 'gyro_horiz': wh,
        'yaw_cum': yaw, 'tilt_from_start': tilt,
    }
    X = np.stack([ch[c] for c in CHANNELS]).astype(np.float64)
    return Canonical(t, X, g, stream.valid, float(onset_ms), float(end_ms), i_on, i_end, float(gg.closure_deg),
                     int(gg.pre_n), int(gg.post_n), gg.method)


# ───────────────────────────── windows ─────────────────────────────


def _interp_rows(t: np.ndarray, X: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(grid, t, X[k]) for k in range(X.shape[0])])


def time_normalised(can: Canonical, start_ms: float, end_ms: float, n: int = C.N_NORM) -> np.ndarray:
    """(C, n) float32: the channels at n equally spaced instants from start_ms to end_ms (inclusive);
    ``yaw_cum`` is re-zeroed at start_ms."""
    grid = np.linspace(float(start_ms), float(end_ms), n)
    Xn = _interp_rows(can.t, can.X, grid)
    k = CHANNELS.index('yaw_cum')
    Xn[k] -= Xn[k, 0]
    return Xn.astype(np.float32)


def window_valid(can: Canonical | Stream, start_ms: float, end_ms: float) -> float:
    """Fraction of grid samples in [start_ms, end_ms] that were measured (not inside a long gap)."""
    t = can.t
    sel = (t >= start_ms) & (t <= end_ms)
    return float(can.valid[sel].mean()) if sel.any() else 0.0


def fixed_grid(onset_ms: float, pre_s: float = C.FIXED_PRE_S, post_s: float = C.FIXED_POST_S,
               fs_hz: float = C.FS_HZ) -> np.ndarray:
    """Sample times (ms) of the fixed window: onset - pre_s .. onset + post_s (exclusive), 1/fs apart."""
    n = int(round((pre_s + post_s) * fs_hz))
    return float(onset_ms) - pre_s * 1000.0 + np.arange(n) * (1000.0 / fs_hz)


def fixed_window(
    can: Canonical,
    onset_ms: float,
    *,
    pre_s: float = C.FIXED_PRE_S,
    post_s: float = C.FIXED_POST_S,
    fs_hz: float = C.FS_HZ,
    cut_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X (C, T) float32, mask (T,) bool, grid (T,) ms) of the fixed window around ``onset_ms``.

    mask is False outside the recording, inside long gaps and from ``cut_ms`` on; there the values are
    frozen at the last sample before the cut (carry-forward), so samples after the end cue (the arm's
    return) never carry information. ``yaw_cum`` is 0 at the onset (the canonical reference)."""
    grid = fixed_grid(onset_ms, pre_s, post_s, fs_hz)
    X = _interp_rows(can.t, can.X, grid)
    inside = (grid >= can.t[0]) & (grid <= can.t[-1])
    idx = np.clip(np.searchsorted(can.t, grid), 0, can.t.size - 1)
    measured = can.valid[idx] & inside
    mask = measured.copy()
    if cut_ms is not None and math.isfinite(cut_ms):
        after = grid >= cut_ms
        if after.any():
            k0 = int(np.argmax(after))
            if k0 > 0:
                X[:, k0:] = X[:, k0 - 1:k0]
            mask &= ~after
    return X.astype(np.float32), mask, grid


def phase_labels(t: np.ndarray, onset_ms: float, end_ms: float, cue_ms: float | None = None,
                 return_onset_ms: float | None = None, return_end_ms: float | None = None) -> np.ndarray:
    """int8 phase code per time (``PHASES``): pre, approach [onset, end], hold (end, end cue),
    post_cue (from the end cue: the participant may return), return [return onset, return end), after."""
    ph = np.zeros(t.size, dtype=np.int8)
    ph[t >= onset_ms] = 1
    ph[t > end_ms] = 2

    def ok(v: Any) -> bool:
        return v is not None and isinstance(v, (int, float)) and math.isfinite(v)

    if ok(cue_ms):
        ph[t >= cue_ms] = 3
    if ok(return_onset_ms):
        ph[t >= return_onset_ms] = 4
        if ok(return_end_ms):
            ph[t >= return_end_ms] = 5
    return ph


# ───────────────────────────── raw-sample QC ─────────────────────────────


def gap_stats(t_raw: np.ndarray, start_ms: float, end_ms: float, threshold_ms: float | None = None) -> dict[str, float]:
    """Sampling gaps (intervals between consecutive raw samples) overlapping [start_ms, end_ms]: the longest,
    the number over 50 ms and over ``MAX_GAP_MS``, and (``n_gaps_over``) the number over ``threshold_ms``."""
    t = np.asarray(t_raw, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return {'gap_max_ms': math.nan, 'n_gaps_50': 0, 'n_gaps_100': 0, 'n_gaps_over': 0}
    dt = np.diff(t)
    over = (t[1:] > start_ms) & (t[:-1] < end_ms)
    d = dt[over]
    thr = C.MAX_GAP_MS if threshold_ms is None else float(threshold_ms)
    return {'gap_max_ms': float(d.max()) if d.size else math.nan,
            'n_gaps_50': int(np.sum(d > 50.0)), 'n_gaps_100': int(np.sum(d > C.MAX_GAP_MS)),
            'n_gaps_over': int(np.sum(d > thr))}


# ───────────────────────────── one trial, all windows ─────────────────────────────


@dataclass
class TrialArrays:
    """Every fixed-shape array of one trial plus its QC numbers (see :func:`standardize_trial`)."""

    approach: np.ndarray            # (C, N_NORM) time-normalised oracle approach
    fixed: np.ndarray               # (C, T_FIXED) fixed window around the onset
    fixed_mask: np.ndarray          # (T_FIXED,) bool
    fixed_phase: np.ndarray         # (T_FIXED,) int8
    ret: np.ndarray                 # (C, N_NORM) time-normalised return (NaN when unavailable)
    ret_ok: bool
    qc: dict[str, Any]
    stream: Stream
    stream_phase: np.ndarray        # (n,) int8 phase of every stream sample


T_FIXED = int(round((C.FIXED_PRE_S + C.FIXED_POST_S) * C.FS_HZ))


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def standardize_trial(
    record: Mapping[str, Any],
    norm: Normalization | None,
    onset_ms: float,
    end_ms: float,
    *,
    stream: Stream | None = None,
    **stream_kw: Any,
) -> TrialArrays:
    """All windows of one trial for the given approach bounds (oracle or detected).

    Events come from ``record['timing']``: the end cue (``end_detected_ms``) freezes the fixed window and
    the return window is [return_onset_ms, return_end_ms] when both were observed (schema 1.2)."""
    st = stream if stream is not None else stream_from_record(record, norm, **stream_kw)
    tim = record.get('timing') or {}
    cue = _num(tim.get('end_detected_ms'))
    r_on, r_end = _num(tim.get('return_onset_ms')), _num(tim.get('return_end_ms'))
    can = canonical(st, onset_ms, end_ms)
    approach = time_normalised(can, onset_ms, end_ms)
    fixed, fmask, fgrid = fixed_window(can, onset_ms, cut_ms=cue)
    fphase = phase_labels(fgrid, onset_ms, end_ms, cue, r_on, r_end)
    seg = (st.t >= onset_ms) & (st.t <= end_ms)
    gyro_mag = can.channel('gyro_mag')
    acc_mag = can.channel('acc_mag')
    pad = C.GAP_WINDOW_PAD_S * 1000.0
    gaps = gap_stats(st.t_raw, onset_ms - pad, end_ms + pad)
    g1pad = C.G1_PAD_S * 1000.0
    g1 = gap_stats(st.t_raw, onset_ms - g1pad, end_ms + g1pad, threshold_ms=C.G1_MAX_GAP_MS)
    # the grip before the movement: mean unit up vector (device frame, W3C sign) over the rest before the onset
    pre = (st.t >= onset_ms - C.PRE_GRAVITY_MS) & (st.t <= onset_ms)
    u_pre = (np.nanmean(can.X[[CHANNELS.index(c) for c in ('up_x', 'up_y', 'up_z')]][:, pre], axis=1)
             if pre.any() else np.full(3, np.nan))
    qc: dict[str, Any] = {
        'duration_ms': float(end_ms - onset_ms),
        'peak_gyro_dps': float(np.nanmax(gyro_mag[seg])) if seg.any() else math.nan,
        'peak_acc_ms2': float(np.nanmax(acc_mag[seg])) if seg.any() else math.nan,
        'closure_deg': can.closure_deg, 'rest_pre_n': can.rest_pre_n, 'rest_post_n': can.rest_post_n,
        'gravity_method': can.gravity_method,
        'approach_valid_frac': window_valid(can, onset_ms, end_ms),
        'win_gap_max_ms': gaps['gap_max_ms'], 'win_n_gaps_50': gaps['n_gaps_50'],
        'win_n_gaps_100': gaps['n_gaps_100'], 'g1_gap_max_ms': g1['gap_max_ms'], 'g1_n_gaps': g1['n_gaps_over'],
        'up_pre_x': float(u_pre[0]), 'up_pre_y': float(u_pre[1]), 'up_pre_z': float(u_pre[2]),
        'first_half_yaw_deg': float(np.interp(0.5 * (onset_ms + end_ms), can.t, can.channel('yaw_cum'))),
        'net_yaw_deg': float(np.interp(end_ms, can.t, can.channel('yaw_cum'))),
        'repeats_dropped': st.repeats_dropped, 'n_grid': st.n,
    }
    ret = np.full((len(CHANNELS), C.N_NORM), np.nan, dtype=np.float32)
    ret_ok = False
    reason = 'not_observed'
    if r_on is not None and r_end is not None:
        if r_end - r_on < 100.0:
            reason = 'too_short'
        else:
            rg = gap_stats(st.t_raw, r_on - 250.0, r_end + 250.0)
            if rg['n_gaps_100'] > 0:
                reason = 'gap'
            else:
                try:
                    can_r = canonical(st, r_on, r_end)
                    ret = time_normalised(can_r, r_on, r_end)
                    ret_ok, reason = True, 'ok'
                    qc['return_closure_deg'] = can_r.closure_deg
                    qc['return_yaw_first_half_deg'] = float(np.interp(0.5 * (r_on + r_end), can_r.t,
                                                                      can_r.channel('yaw_cum')))
                except ValueError:
                    reason = 'invalid_bounds'
    qc['return_status'] = reason
    sphase = phase_labels(st.t, onset_ms, end_ms, cue, r_on, r_end)
    return TrialArrays(approach, fixed, fmask, fphase, ret, ret_ok, qc, st, sphase)
