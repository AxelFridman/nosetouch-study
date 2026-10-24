"""Per-trial features over the detected movement segment, and sequence tensors for deep models.

Pipeline of :func:`prepare_trial` (docs/05 section 2): norm_v2 normalisation (session gravity sign,
rotationRate mapping and units, ``conventions.Normalization``) -> optional left/right mirror and screen
flip of the normalised vectors -> optional drop of duplicated samples -> uniform resampling -> zero-phase
low-pass -> segment (offline_v1, fallback to the app's online_v1) -> gyro bias (only a trusted calibration
bias, norm_v2) -> gravity_v1: the rest-window gravity propagated with the gyroscope forward from the onset
and backward from the end, blended in time (``preprocess.gyro_gravity``) -> linear acceleration
a = f - g and the unit 'up' vector u = g / |g|. The browser's own gravity (accg - acc) is only compared
with it (``qc_gravity_vs_browser_*``). :func:`rate_closures` runs the same gravity propagation under every
candidate rotationRate mapping (the session-level check of ``conventions.infer_rate_mapping``).

Feature groups (see :func:`feature_groups`):
  basic      timing and magnitude peaks: duration, reaction time, peak |w| and |a|, time-to-peak, total
             rotation, number of angular-speed peaks (movement units).
  device     device-frame per-axis statistics of the linear acceleration and angular velocity, the 'up'
             direction at onset/end and the net rotation vector. They depend on how the phone is held
             (orientation, grip): useful within an orientation, confounded across orientations.
  invariant  magnitudes and gravity-referenced components (vertical / horizontal acceleration, vertical /
             horizontal angular speed, tilt change, displacement): unchanged by ANY fixed rotation of the
             phone in the hand and by left/right mirroring (achiral).
  chiral     pseudoscalars: unchanged by any fixed grip rotation, sign-flipped by mirroring. By symmetry
             they are the features that can tell left from right regardless of the grip:
               chi_yaw_net_deg        Psi = integral of w·u dt: net rotation about the vertical
               chi_yaw_first/second_half_deg, chi_yaw_peak_dps (signed peak of w·u), chi_yaw_net_norm
               chi_yaw_rate_b1..b5    mean w·u in 5 time-normalised bins
               chi_omega_dot_ahat_dps |a|-weighted mean of w·a_hat; chi_helicity (|a| >= 0.5 m/s^2)
               chi_turning_sense      mean((v x a)·u), v from integrated world-frame acceleration with a
                                      zero-velocity detrend; chi_turning_sense_norm (/ mean|v||a|)
               chi_triple_acc         mean det[a(t), a(t + 100 ms), u]; chi_triple_acc_norm
               chi_rot_about_path_deg net rotation vector · unit displacement
             Velocity / displacement features need ZUPT integration and are NaN for movements longer than
             ``max_zupt_s`` (1.5 s; the error grows quickly with duration, docs/05 section 2.3).
Conventions: device-frame W3C signals (accg points UP at rest; w = [alpha, beta, gamma] = rates about
x, y, z after the session's mapping). World-frame quantities are expressed in the device frame at
movement onset (D0), transported with the integrated gyro, so every dot / cross / determinant is a
proper scalar or pseudoscalar. A wrong gravity sign negates u and a and therefore flips EVERY chiral
feature - see ``qc_gravity_sign_source`` ('default' = nothing supports the sign).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

from .conventions import (
    GYRO_BIAS_MAX_DPS, NORM_VERSION, RAD2DEG as _RAD2DEG, RATE_CANDIDATES, Normalization, RateMapping,
    calibration_gyro_bias, flip_device_vectors, mirror_device_vectors, normalize_motion, rate_candidate_key,
    rotation_rate_to_device,
)
from .io import streams_from_record
from .preprocess import (
    angle_deg, estimate_gravity, fill_gaps_linear, gyro_gravity, integrate_gyro, lowpass, matrix_to_rotvec,
    repeated_sample_mask, resample_uniform, trapezoid, unit_rows,
)
from .segment import segment_bounds_ms

RAD2DEG = 180.0 / math.pi
N_YAW_BINS = 5

BASIC = [
    'duration_ms', 'reaction_ms', 'peak_gyro_dps', 'peak_acc_ms2', 't_peak_gyro_ms', 't_peak_gyro_frac',
    't_peak_acc_ms', 't_peak_acc_frac', 'total_rotation_deg', 'n_gyro_peaks',
]
DEVICE = (
    [f'dev_{sig}_{ax}_{st}' for sig in ('acc', 'gyro') for ax in 'xyz' for st in ('mean', 'sd', 'min', 'max')]
    + [f'dev_up_start_{ax}' for ax in 'xyz'] + [f'dev_up_end_{ax}' for ax in 'xyz']
    + [f'dev_rot_{ax}_deg' for ax in 'xyz']
)
INVARIANT = [
    'inv_gyro_mag_mean', 'inv_gyro_mag_sd', 'inv_acc_mag_mean', 'inv_acc_mag_sd',
    'inv_acc_vert_mean', 'inv_acc_vert_max', 'inv_acc_vert_min', 'inv_acc_vert_sd',
    'inv_acc_horiz_mean', 'inv_acc_horiz_max', 'inv_gyro_vert_abs_mean', 'inv_gyro_horiz_mean',
    'inv_gyro_horiz_max', 'inv_tilt_change_deg', 'inv_net_rotation_deg', 'inv_yaw_abs_deg',
    'inv_disp_vert_m', 'inv_disp_horiz_m', 'inv_path_length_m', 'inv_peak_speed_ms',
]
CHIRAL = [
    'chi_yaw_net_deg', 'chi_yaw_first_half_deg', 'chi_yaw_second_half_deg', 'chi_yaw_peak_dps', 'chi_yaw_net_norm',
    *[f'chi_yaw_rate_b{k + 1}' for k in range(N_YAW_BINS)],
    'chi_omega_dot_ahat_dps', 'chi_helicity', 'chi_turning_sense', 'chi_turning_sense_norm', 'chi_triple_acc',
    'chi_triple_acc_norm', 'chi_rot_about_path_deg',
]
#: Features that need integrated velocity (ZUPT); NaN for movements longer than max_zupt_s.
ZUPT_FEATURES = ('inv_disp_vert_m', 'inv_disp_horiz_m', 'inv_path_length_m', 'inv_peak_speed_ms',
                 'chi_turning_sense', 'chi_turning_sense_norm', 'chi_rot_about_path_deg')
QC = [
    'qc_ok', 'qc_reason', 'qc_nan_frac', 'qc_gyro_missing_steps', 'qc_segment_source', 'qc_gravity_method',
    'qc_closure_deg', 'qc_gravity_vs_browser_deg', 'qc_gravity_vs_browser_max_deg', 'qc_rest_pre_n',
    'qc_rest_post_n', 'qc_gravity_sign', 'qc_gravity_sign_source', 'qc_rr_mapping', 'qc_rr_source',
    'qc_axis_verified', 'qc_gyro_bias_source', 'qc_repeats_dropped', 'qc_mirrored', 'qc_screen_flipped',
    'qc_norm_version',
]

SEQUENCE_CHANNELS: dict[str, list[str]] = {
    'invariant': ['gyro_mag', 'acc_mag', 'acc_vert', 'acc_horiz', 'gyro_horiz', 'tilt_from_start', 'yaw_rate', 'yaw_cum'],
    'device': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'up_x', 'up_y', 'up_z'],
}
CHIRAL_SEQUENCE_CHANNELS = ('yaw_rate', 'yaw_cum')


def feature_groups() -> dict[str, list[str]]:
    """{'basic', 'device', 'invariant', 'chiral'} -> feature names (copies)."""
    return {'basic': list(BASIC), 'device': list(DEVICE), 'invariant': list(INVARIANT), 'chiral': list(CHIRAL)}


def all_feature_names(groups: Iterable[str] = ('basic', 'device', 'invariant', 'chiral')) -> list[str]:
    g = feature_groups()
    out: list[str] = []
    for name in groups:
        out += g[name]
    return out


@dataclass
class PreparedTrial:
    """Uniform-grid, normalised device signals of one trial plus its movement segment.

    t: grid (ms since t0); f: specific force (accg, W3C sign) (n,3); acc: linear acceleration f - g (n,3)
    m/s^2; gyro: angular velocity (n,3) deg/s (bias-corrected); g: gravity estimate (n,3) (gyro-propagated
    when the segment is known, else the browser's); up: g / |g|; g_browser: accg - acc (QC only);
    i0..i1: segment on the grid (inclusive). gyro_bias_source: 'calibration', 'pre_go' or 'none'.
    """

    trial_id: str | None
    fs: float
    t: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray
    up: np.ndarray
    i0: int | None
    i1: int | None
    onset_ms: float | None
    end_ms: float | None
    go_ms: float | None
    segment_source: str
    gravity_method: str
    gravity_sign: int
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ok: bool = True
    reason: str = 'ok'
    f: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    g: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    g_browser: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    closure_deg: float = float('nan')
    browser_angle_mean_deg: float = float('nan')
    browser_angle_max_deg: float = float('nan')
    rest_pre_n: int = 0
    rest_post_n: int = 0
    gyro_missing_steps: int = 0
    repeats_dropped: int = 0
    mirrored: bool = False
    norm: Normalization = field(default_factory=Normalization)
    screen_flipped: bool = False
    gyro_bias_source: str = 'none'


def _as_norm(norm: Normalization | None, gravity_sign: int | None) -> Normalization:
    if norm is not None:
        return norm
    return Normalization.from_sign(1 if gravity_sign is None else int(gravity_sign))


def _still_threshold(record: Mapping[str, Any], default: float = 15.0) -> float:
    p = ((record.get('segmentation') or {}).get('params')) or {}
    v = p.get('still_gyro_dps')
    return float(v) if isinstance(v, (int, float)) and v > 0 else default


def prepare_trial(
    record: Mapping[str, Any],
    *,
    norm: Normalization | None = None,
    gravity_sign: int | None = None,
    segment: str = 'auto',
    bounds: tuple[float | None, float | None, str] | None = None,
    fs_hz: float = 100.0,
    smooth_hz: float = 15.0,
    browser_gravity_hz: float = 1.0,
    rest_ms: float = 300.0,
    mirror: bool = False,
    screen_flip: bool = False,
    remove_gyro_bias: str | bool = 'auto',
    drop_repeats: bool = True,
) -> PreparedTrial:
    """Normalise, resample, smooth, segment and estimate gravity for one trial.

    norm: the session's :class:`~motionlab.conventions.Normalization` (``Dataset.trial_normalization``);
    without it ``gravity_sign`` (default +1) and the W3C rate mapping are used.
    segment: 'auto' (offline_v1, fallback to the stored app segmentation), 'offline', 'stored', 'online';
    ``bounds`` = precomputed (onset_ms, end_ms, source) skips the segmentation.
    mirror: left/right mirror of the normalised signals (acc -> S acc, w -> -S w) - exact for any
    rotationRate mapping, unlike a raw-field mirror.
    screen_flip: the same movement with the phone turned 180 deg about its y axis (screen toward <-> away):
    acc, accg and w -> F·v with F = diag(-1, 1, -1) (``conventions.flip_device_vectors``). F is a proper
    rotation, so basic, invariant and chiral features do not change; only the device-frame ones do. It
    commutes with ``mirror``. A synthetic grip, never a substitute for real screen-away data.
    remove_gyro_bias: 'auto' (= True, 'calibration') = the calibration bias when the session has a trusted
    one (``conventions.calibration_gyro_bias``: still flat window, below 3 deg/s), else NOTHING (norm_v2:
    an in-hand pre-GO mean is tremor, not bias); 'pre_go' = the mean rate of the 400 ms before GO when it
    is below 3 deg/s (``GYRO_BIAS_MAX_DPS``; the norm_v1 fallback, for ablations); False / 'none' = nothing.
    drop_repeats: drop duplicated motion snapshots before resampling (``preprocess.repeated_sample_mask``).
    Every step is linear per component or rotation-covariant, so a fixed grip rotation of the input
    rotates every vector here and leaves the segment unchanged; mirroring flips every pseudoscalar.
    """
    nm = _as_norm(norm, gravity_sign)
    s = streams_from_record(record)
    sig = normalize_motion(s.t, s.acc, s.accg, s.rr, nm)
    tid = record.get('trial_id')
    go = (record.get('timing') or {}).get('go_ms')
    acc, accg, gyro, t = sig.acc, sig.accg, sig.gyro, sig.t

    def transform(a: np.ndarray | None, w: np.ndarray | None) -> tuple[Any, Any]:
        """mirror and / or screen flip of normalised device vectors (they commute)."""
        if mirror:
            a, w = mirror_device_vectors(a, w)
        if screen_flip:
            a, w = flip_device_vectors(a, w)
        return a, w

    acc, gyro = transform(acc, gyro)
    accg, _ = transform(accg, None)
    n_rep = 0
    if drop_repeats and t.size > 1:
        rep = repeated_sample_mask(s.acc, s.accg, s.rr, engine=nm.engine)
        n_rep = int(rep.sum())
        if n_rep:
            keep = ~rep
            t, acc, accg, gyro = t[keep], acc[keep], accg[keep], gyro[keep]

    def empty(reason: str) -> PreparedTrial:
        z = np.zeros((0, 3))
        return PreparedTrial(tid, fs_hz, np.zeros(0), z, z, z, None, None, None, None, go, 'none', 'none', nm.sign,
                             ok=False, reason=reason, repeats_dropped=n_rep, mirrored=mirror, norm=nm,
                             screen_flipped=screen_flip)

    if t.size < 5:
        return empty('too_few_samples')
    tu, U = resample_uniform(t, np.column_stack([acc, accg, gyro]), fs_hz, 100.0)
    if tu.size < 5:
        return empty('too_few_samples')
    U = lowpass(U, fs_hz, smooth_hz)
    acc_s, f, gyro = U[:, 0:3], U[:, 3:6], U[:, 6:9]
    g_browser, _ = estimate_gravity(acc_s, f, fs_hz, browser_gravity_hz)
    onset, end, src = bounds if bounds is not None else segment_bounds_ms(record, segment)

    # gyro bias (deg/s, device frame): subtract before integrating (1 deg/s integrates to 1 deg per second)
    bias = np.zeros(3)
    bias_src = 'none'
    mode = 'none' if remove_gyro_bias is False else ('auto' if remove_gyro_bias is True else str(remove_gyro_bias))
    if mode not in ('auto', 'calibration', 'pre_go', 'none'):
        raise ValueError(f"remove_gyro_bias must be 'auto', 'calibration', 'pre_go' or False, got {remove_gyro_bias!r}")
    if mode in ('auto', 'calibration') and nm.gyro_bias is not None:
        bias = transform(None, np.asarray(nm.gyro_bias, dtype=float))[1]
        bias_src = 'calibration'
    elif mode == 'pre_go' and go is not None:
        w = (tu >= float(go) - 400.0) & (tu <= float(go))
        if onset is not None:
            w &= tu < onset
        W = gyro[w]
        W = W[np.isfinite(W).all(axis=1)]
        if len(W) >= 5:
            b = W.mean(axis=0)  # mean, not median: the mean is rotation-covariant
            if np.linalg.norm(b) < GYRO_BIAS_MAX_DPS:
                bias, bias_src = b, 'pre_go'
    gyro = gyro - bias

    base = PreparedTrial(tid, fs_hz, tu, acc_s.copy(), gyro, unit_rows(g_browser), None, None, onset, end, go, src,
                         'browser', nm.sign, bias, f=f, g=g_browser, g_browser=g_browser, repeats_dropped=n_rep,
                         mirrored=mirror, norm=nm, screen_flipped=screen_flip, gyro_bias_source=bias_src)
    base.acc = f - g_browser
    if onset is None or end is None:
        base.ok, base.reason = False, f'no_segment ({src})'
        return base
    step = 1000.0 / fs_hz
    i0 = int(math.ceil((onset - tu[0]) / step - 1e-9))
    i1 = int(math.floor((end - tu[0]) / step + 1e-9))
    i0, i1 = max(i0, 0), min(i1, tu.size - 1)
    if i1 - i0 < 3:
        base.ok, base.reason = False, 'segment_too_short'
        return base
    base.i0, base.i1 = i0, i1

    # gravity_v1: gyro propagation from the rest windows (NaN gyro rows rotate by zero and are counted)
    seg = record.get('segmentation') or {}
    bg = (seg.get('baseline') or {}).get('gravity')
    fallback = None
    if bg is not None:
        fallback = transform(np.asarray([np.nan if v is None else v for v in bg], dtype=float) * nm.sign, None)[0]
    f_fill = fill_gaps_linear(tu, f)
    gg = gyro_gravity(tu, f_fill, gyro, i0, i1, still_dps=_still_threshold(record), rest_ms=rest_ms,
                      magnitude=nm.gravity_magnitude, g_pre_fallback=fallback)
    base.g = gg.g
    base.up = unit_rows(gg.g)
    base.acc = f - gg.g
    base.gravity_method = f'gyro_{gg.method}'
    base.closure_deg = gg.closure_deg
    base.rest_pre_n, base.rest_post_n = gg.pre_n, gg.post_n
    base.gyro_missing_steps = gg.gyro_missing_steps
    ang = angle_deg(gg.g[i0:i1 + 1], g_browser[i0:i1 + 1])
    ang = ang[np.isfinite(ang)]
    if ang.size:
        base.browser_angle_mean_deg, base.browser_angle_max_deg = float(ang.mean()), float(ang.max())
    return base


def rate_closures(
    record: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    engine: str | None = None,
    bounds: tuple[float | None, float | None, str] | None = None,
    fs_hz: float = 100.0,
    smooth_hz: float = 15.0,
    rest_ms: float = 300.0,
    min_rotation_deg: float = 30.0,
    candidates: Iterable[tuple[str, str]] = RATE_CANDIDATES,
) -> dict[str, float] | None:
    """Gyro-gravity closure (deg) of one trial under every candidate rotationRate mapping.

    The pipeline of :func:`prepare_trial` (repeats dropped, 100 Hz grid, zero-phase low-pass, segment
    'auto', gravity_v1 rest windows of ``rest_ms``) with the raw rotationRate read under each (verdict, units)
    of ``candidates``: closure = angle between the rest gravity before the movement, propagated through it
    with that gyro, and the rest gravity at the nose. The right mapping closes to the sensor noise; a wrong
    one keeps an error of the path rotation (pilot session medians: W3C 1.3 deg, legacy 4.0, inverted 5.8-7.0,
    rad/s ~90). The accelerometer sign does not matter (both rests flip together), and the rest samples
    are the same for every candidate (the still threshold scales with the units; a permutation or a sign
    keeps |w|). calibration: the session calibration; its gyro bias is subtracted (mapped per candidate)
    only when trusted (``conventions.calibration_gyro_bias``), never a hand-held pre-GO mean.
    Returns {'verdict/units': closure_deg} or None when the trial cannot tell the mappings apart: no
    segment, no rest window on either side, or less than ``min_rotation_deg`` of path rotation (read in
    the units that close best for this trial: units are decisive per trial, the axis mapping is not).
    """
    s = streams_from_record(record)
    t, accg, rr = s.t, s.accg, s.rr
    if t.size > 1:
        rep = repeated_sample_mask(s.acc, s.accg, s.rr, engine=engine)
        if rep.any():
            t, accg, rr = t[~rep], accg[~rep], rr[~rep]
    if t.size < 5:
        return None
    tu, U = resample_uniform(t, np.column_stack([accg, rr]), fs_hz, 100.0)
    if tu.size < 5:
        return None
    U = lowpass(U, fs_hz, smooth_hz)
    onset, end, _ = bounds if bounds is not None else segment_bounds_ms(record, 'auto')
    if onset is None or end is None:
        return None
    sel = (tu >= onset - 2.0 * rest_ms) & (tu <= end + 2.0 * rest_ms)
    tc = tu[sel]
    f = fill_gaps_linear(tc, U[sel, 0:3])
    raw = U[sel, 3:6]
    inside = np.flatnonzero((tc >= onset) & (tc <= end))
    if inside.size < 4:
        return None
    i0, i1 = int(inside[0]), int(inside[-1])
    mag = np.linalg.norm(raw[i0:i1 + 1], axis=1)
    path_raw = float(trapezoid(np.where(np.isfinite(mag), mag, 0.0), tc[i0:i1 + 1] / 1000.0))
    still = _still_threshold(record)
    out: dict[str, float] = {}
    for verdict, units in candidates:
        w = rotation_rate_to_device(raw, verdict, units)
        b = calibration_gyro_bias(calibration, RateMapping(verdict, units, True)) if calibration else None
        if b is not None:
            w = w - np.asarray(b, dtype=float)
        scale = _RAD2DEG if units == 'rad_per_s' else 1.0
        gg = gyro_gravity(tc, f, w, i0, i1, still_dps=still * scale, rest_ms=rest_ms)
        ok = gg.method == 'fwd_bwd' and gg.pre_n > 0 and math.isfinite(gg.closure_deg)
        out[rate_candidate_key(verdict, units)] = gg.closure_deg if ok else float('nan')
    finite = {k: v for k, v in out.items() if math.isfinite(v)}
    if not finite:
        return None
    best_units = min(finite, key=finite.get).split('/')[1]
    path = path_raw * (_RAD2DEG if best_units == 'rad_per_s' else 1.0)
    if not path >= min_rotation_deg:
        return None
    return out


def _nan_features(reason: str, pt: PreparedTrial | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {k: float('nan') for k in BASIC + DEVICE + INVARIANT + CHIRAL}
    out.update(_qc(pt, ok=0, reason=reason, nan_frac=float('nan')))
    return out


def _qc(pt: PreparedTrial | None, *, ok: int, reason: str, nan_frac: float) -> dict[str, Any]:
    nm = pt.norm if pt is not None else Normalization()
    return {
        'qc_ok': ok, 'qc_reason': reason, 'qc_nan_frac': nan_frac,
        'qc_gyro_missing_steps': pt.gyro_missing_steps if pt else 0,
        'qc_segment_source': pt.segment_source if pt else 'none',
        'qc_gravity_method': pt.gravity_method if pt else 'none',
        'qc_closure_deg': pt.closure_deg if pt else float('nan'),
        'qc_gravity_vs_browser_deg': pt.browser_angle_mean_deg if pt else float('nan'),
        'qc_gravity_vs_browser_max_deg': pt.browser_angle_max_deg if pt else float('nan'),
        'qc_rest_pre_n': pt.rest_pre_n if pt else 0, 'qc_rest_post_n': pt.rest_post_n if pt else 0,
        'qc_gravity_sign': nm.sign, 'qc_gravity_sign_source': nm.gravity.source,
        'qc_rr_mapping': nm.rates.verdict, 'qc_rr_source': nm.rates.source, 'qc_axis_verified': int(nm.rates.verified),
        'qc_gyro_bias_source': pt.gyro_bias_source if pt else 'none',
        'qc_repeats_dropped': pt.repeats_dropped if pt else 0, 'qc_mirrored': int(pt.mirrored) if pt else 0,
        'qc_screen_flipped': int(pt.screen_flipped) if pt else 0,
        'qc_norm_version': nm.version or NORM_VERSION,
    }


def _stats(x: np.ndarray) -> tuple[float, float, float, float]:
    if not np.isfinite(x).any():
        return (float('nan'),) * 4
    return float(np.nanmean(x)), float(np.nanstd(x)), float(np.nanmin(x)), float(np.nanmax(x))


def _world(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum('nij,nj->ni', R, v)


def _segment_arrays(pt: PreparedTrial, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sl = slice(i0, i1 + 1)
    t = pt.t[sl]
    raw = np.column_stack([pt.gyro[sl], pt.acc[sl], pt.up[sl]])
    nan_frac = float((~np.isfinite(raw).all(axis=1)).mean())
    filled = fill_gaps_linear(t, raw)
    w, a, u = filled[:, 0:3], filled[:, 3:6], unit_rows(filled[:, 6:9])
    return t, w, a, u, nan_frac


def trial_features(trial: Mapping[str, Any] | PreparedTrial, *, max_zupt_s: float = 1.5, **kw: Any) -> dict[str, Any]:
    """Feature dict (all groups + qc_* columns) of one trial (a TrialRecord or a :class:`PreparedTrial`).

    Keyword arguments are passed to :func:`prepare_trial` (norm, gravity_sign, segment, mirror, ...).
    Unusable trials return NaN features with qc_ok = 0 and the reason in qc_reason.
    """
    pt = trial if isinstance(trial, PreparedTrial) else prepare_trial(trial, **kw)
    if not pt.ok or pt.i0 is None or pt.i1 is None:
        return _nan_features(pt.reason, pt)
    t, w, a, u, nan_frac = _segment_arrays(pt, pt.i0, pt.i1)
    if not (np.isfinite(w).all() and np.isfinite(a).all() and np.isfinite(u).all()):
        return _nan_features('missing_signals_in_segment', pt)
    ts = (t - t[0]) / 1000.0
    T = float(ts[-1])
    f: dict[str, Any] = {}
    wmag = np.linalg.norm(w, axis=1)
    amag = np.linalg.norm(a, axis=1)

    # basic
    f['duration_ms'] = float(t[-1] - t[0])
    f['reaction_ms'] = float(pt.onset_ms - pt.go_ms) if (pt.go_ms is not None and pt.onset_ms is not None) else float('nan')
    kg, ka = int(np.argmax(wmag)), int(np.argmax(amag))
    f['peak_gyro_dps'] = float(wmag[kg])
    f['peak_acc_ms2'] = float(amag[ka])
    f['t_peak_gyro_ms'] = float(ts[kg] * 1000.0)
    f['t_peak_gyro_frac'] = float(ts[kg] / T) if T > 0 else float('nan')
    f['t_peak_acc_ms'] = float(ts[ka] * 1000.0)
    f['t_peak_acc_frac'] = float(ts[ka] / T) if T > 0 else float('nan')
    f['total_rotation_deg'] = float(trapezoid(wmag, ts))
    # movement units: |w| peaks with prominence >= 15 deg/s at least 150 ms apart (Thrane 2020 style)
    dist = max(1, int(round(0.15 * pt.fs)))
    f['n_gyro_peaks'] = float(len(find_peaks(wmag, prominence=15.0, distance=dist)[0]))

    # orientation over the segment (device(k) -> device(onset)) and world-frame vectors
    R, missing = integrate_gyro(t, w)
    aW, uW = _world(R, a), _world(R, u)
    v = cumulative_trapezoid(aW, ts, axis=0, initial=0.0)
    if T > 0:
        v = v - (ts / T)[:, None] * v[-1][None, :]  # zero-velocity detrend (still at onset and at the nose)
    d = cumulative_trapezoid(v, ts, axis=0, initial=0.0)
    rot = matrix_to_rotvec(R[-1]) * RAD2DEG  # in device(onset) coordinates

    # device
    for name, X in (('acc', a), ('gyro', w)):
        for j, ax in enumerate('xyz'):
            m, sd, lo, hi = _stats(X[:, j])
            f[f'dev_{name}_{ax}_mean'], f[f'dev_{name}_{ax}_sd'] = m, sd
            f[f'dev_{name}_{ax}_min'], f[f'dev_{name}_{ax}_max'] = lo, hi
    for j, ax in enumerate('xyz'):
        f[f'dev_up_start_{ax}'] = float(u[0, j])
        f[f'dev_up_end_{ax}'] = float(u[-1, j])
        f[f'dev_rot_{ax}_deg'] = float(rot[j])

    # invariant
    av = np.sum(a * u, axis=1)
    ah = np.linalg.norm(a - av[:, None] * u, axis=1)
    wv = np.sum(w * u, axis=1)
    wh = np.linalg.norm(w - wv[:, None] * u, axis=1)
    f['inv_gyro_mag_mean'], f['inv_gyro_mag_sd'] = float(wmag.mean()), float(wmag.std())
    f['inv_acc_mag_mean'], f['inv_acc_mag_sd'] = float(amag.mean()), float(amag.std())
    m, sd, lo, hi = _stats(av)
    f['inv_acc_vert_mean'], f['inv_acc_vert_sd'], f['inv_acc_vert_min'], f['inv_acc_vert_max'] = m, sd, lo, hi
    f['inv_acc_horiz_mean'], f['inv_acc_horiz_max'] = float(ah.mean()), float(ah.max())
    f['inv_gyro_vert_abs_mean'] = float(np.abs(wv).mean())
    f['inv_gyro_horiz_mean'], f['inv_gyro_horiz_max'] = float(wh.mean()), float(wh.max())
    f['inv_tilt_change_deg'] = float(angle_deg(u[0], u[-1]))
    f['inv_net_rotation_deg'] = float(np.linalg.norm(rot))
    yaw_cum = cumulative_trapezoid(wv, ts, initial=0.0)
    yaw = float(yaw_cum[-1])
    f['inv_yaw_abs_deg'] = abs(yaw)
    u_mean = uW.mean(axis=0)
    u_hat = u_mean / np.linalg.norm(u_mean)
    d_end = d[-1]
    dv = float(d_end @ u_hat)
    f['inv_disp_vert_m'] = dv
    f['inv_disp_horiz_m'] = float(np.linalg.norm(d_end - dv * u_hat))
    speed = np.linalg.norm(v, axis=1)
    f['inv_path_length_m'] = float(trapezoid(speed, ts))
    f['inv_peak_speed_ms'] = float(speed.max())

    # chiral
    f['chi_yaw_net_deg'] = yaw
    first = float(np.interp(T / 2.0, ts, yaw_cum)) if T > 0 else float('nan')
    f['chi_yaw_first_half_deg'] = first
    f['chi_yaw_second_half_deg'] = yaw - first
    f['chi_yaw_peak_dps'] = float(wv[int(np.argmax(np.abs(wv)))])
    f['chi_yaw_net_norm'] = yaw / f['total_rotation_deg'] if f['total_rotation_deg'] > 0 else float('nan')
    edges = np.linspace(0.0, T, N_YAW_BINS + 1)
    for k in range(N_YAW_BINS):
        sel = (ts >= edges[k]) & ((ts < edges[k + 1]) if k < N_YAW_BINS - 1 else (ts <= edges[k + 1]))
        f[f'chi_yaw_rate_b{k + 1}'] = float(wv[sel].mean()) if sel.any() else float('nan')
    sa = float(amag.sum())
    f['chi_omega_dot_ahat_dps'] = float(np.sum(np.sum(w * a, axis=1)) / sa) if sa > 0 else float('nan')
    big = amag >= 0.5
    den_h = float(np.sum(wmag[big] * amag[big]))
    f['chi_helicity'] = float(np.sum(np.sum(w[big] * a[big], axis=1)) / den_h) if den_h > 0 else float('nan')
    turn = np.sum(np.cross(v, aW) * uW, axis=1)
    f['chi_turning_sense'] = float(turn.mean())
    den = float(np.mean(speed * np.linalg.norm(aW, axis=1)))
    f['chi_turning_sense_norm'] = float(turn.mean() / den) if den > 0 else float('nan')
    lag = max(1, int(round(0.1 * pt.fs)))
    if aW.shape[0] > lag:
        a1, a2, uu = aW[:-lag], aW[lag:], uW[:-lag]
        tri = np.sum(np.cross(a1, a2) * uu, axis=1)
        f['chi_triple_acc'] = float(tri.mean())
        den = float(np.mean(np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1)))
        f['chi_triple_acc_norm'] = float(tri.mean() / den) if den > 0 else float('nan')
    else:
        f['chi_triple_acc'] = f['chi_triple_acc_norm'] = float('nan')
    nd = float(np.linalg.norm(d_end))
    f['chi_rot_about_path_deg'] = float(rot @ (d_end / nd)) if nd > 1e-9 else float('nan')
    if T > max_zupt_s:
        for k in ZUPT_FEATURES:
            f[k] = float('nan')

    f.update(_qc(pt, ok=1, reason='ok', nan_frac=nan_frac))
    f['qc_gyro_missing_steps'] = int(missing) + pt.gyro_missing_steps
    return f


def sequence_tensor(
    trial: Mapping[str, Any] | PreparedTrial,
    n: int = 64,
    *,
    channels: str = 'invariant',
    pad_frac: float = 0.0,
    return_names: bool = False,
    **kw: Any,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Time-normalised multichannel array (C, n) float32 over the movement segment (for deep models).

    channels: 'invariant' (grip-invariant; 'yaw_rate' and 'yaw_cum' are chiral, the rest achiral),
    'device' (raw device axes) or 'all'. pad_frac extends the window by that fraction of the duration on
    both sides (docs/05 suggests [onset - 0.5 s, end + 0.5 s]). Gaps are linearly filled; an unusable
    trial gives an all-NaN array.
    """
    names = SEQUENCE_CHANNELS['invariant'] + SEQUENCE_CHANNELS['device'] if channels == 'all' else list(SEQUENCE_CHANNELS[channels])
    pt = trial if isinstance(trial, PreparedTrial) else prepare_trial(trial, **kw)
    if not pt.ok or pt.i0 is None or pt.i1 is None:
        arr = np.full((len(names), n), np.nan, dtype=np.float32)
        return (arr, names) if return_names else arr
    pad = int(round(pad_frac * (pt.i1 - pt.i0)))
    j0, j1 = max(0, pt.i0 - pad), min(pt.t.size - 1, pt.i1 + pad)
    t, w, a, u, _ = _segment_arrays(pt, j0, j1)
    ts = (t - t[0]) / 1000.0
    wv = np.sum(w * u, axis=1)
    av = np.sum(a * u, axis=1)
    ch = {
        'gyro_mag': np.linalg.norm(w, axis=1), 'acc_mag': np.linalg.norm(a, axis=1), 'acc_vert': av,
        'acc_horiz': np.linalg.norm(a - av[:, None] * u, axis=1),
        'gyro_horiz': np.linalg.norm(w - wv[:, None] * u, axis=1),
        'tilt_from_start': angle_deg(u, np.broadcast_to(u[0], u.shape)),
        'yaw_rate': wv, 'yaw_cum': cumulative_trapezoid(wv, ts, initial=0.0),
        'acc_x': a[:, 0], 'acc_y': a[:, 1], 'acc_z': a[:, 2], 'gyro_x': w[:, 0], 'gyro_y': w[:, 1], 'gyro_z': w[:, 2],
        'up_x': u[:, 0], 'up_y': u[:, 1], 'up_z': u[:, 2],
    }
    grid = np.linspace(t[0], t[-1], n)
    arr = np.stack([np.interp(grid, t, np.nan_to_num(ch[k])) for k in names]).astype(np.float32)
    return (arr, names) if return_names else arr


def _items(data: Any, trial_ids: Iterable[str] | None, normalizations: Mapping[str, Normalization] | None,
           gravity_signs: Mapping[str, int] | None) -> Iterable[tuple[str, Mapping[str, Any], Normalization]]:
    if hasattr(data, 'record') and hasattr(data, 'trial_normalization'):
        ids = list(data.trial_ids if trial_ids is None else trial_ids)
        return ((tid, data.record(tid, compact=True), data.trial_normalization(tid)) for tid in ids)
    recs = list(data)
    norms = dict(normalizations or {})
    gs = gravity_signs or {}
    wanted = None if trial_ids is None else set(trial_ids)

    def norm_of(r: Mapping[str, Any]) -> Normalization:
        sid = r.get('session_id')
        if sid in norms:
            return norms[sid]
        return Normalization.from_sign(int(gs.get(sid, 1)))

    return ((r['trial_id'], r, norm_of(r)) for r in recs if wanted is None or r['trial_id'] in wanted)


def extract_features(
    data: Any,
    trial_ids: Iterable[str] | None = None,
    *,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    mirror: bool | str = False,
    normalizations: Mapping[str, Normalization] | None = None,
    gravity_signs: Mapping[str, int] | None = None,
    segment: str = 'auto',
    progress: bool = False,
    **kw: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Feature table (index trial_id) for a :class:`~motionlab.io.Dataset` or an iterable of records.

    mirror: False -> features of the trials; True -> features of their left/right mirror images (for
    mirror augmentation and the mirror-consistency check); 'both' -> (features, mirrored features), with
    the segmentation computed once per trial. ``transform`` is applied to each raw record first (e.g. a
    grip rotation or quantisation). For a Dataset the session's norm_v2 normalisation is used; for plain
    records pass ``normalizations`` {session_id: Normalization} or ``gravity_signs`` {session_id: sign}
    (default +1, W3C rates). Other keyword arguments go to :func:`prepare_trial`, e.g. ``screen_flip=True``
    for the features of the screen-flipped trials (screen-away augmentation, ``qc_screen_flipped`` = 1).
    """
    rows: dict[str, dict[str, Any]] = {}
    rows_m: dict[str, dict[str, Any]] = {}
    want_orig = mirror in (False, 'both')
    want_mirror = mirror in (True, 'both')
    for k, (tid, rec, nm) in enumerate(_items(data, trial_ids, normalizations, gravity_signs)):
        if transform is not None:
            rec = transform(rec)
        bounds = segment_bounds_ms(rec, segment)
        if want_orig:
            rows[tid] = trial_features(rec, norm=nm, bounds=bounds, **kw)
        if want_mirror:
            rows_m[tid] = trial_features(rec, norm=nm, bounds=bounds, mirror=True, **kw)
        if progress and (k + 1) % 100 == 0:
            print(f'  features: {k + 1} trials', flush=True)
    cols = BASIC + DEVICE + INVARIANT + CHIRAL + QC

    def frame(r: dict[str, dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(r, orient='index')
        df = df.reindex(columns=cols)
        df.index.name = 'trial_id'
        return df

    if mirror == 'both':
        return frame(rows), frame(rows_m)
    return frame(rows_m if want_mirror else rows)
