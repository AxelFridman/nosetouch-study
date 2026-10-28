"""Lane m4, part 1: cue-free candidate segments, their hand-agnostic features and their labels.

One code path serves training, the offline evaluation and the online detector (``m4_online``):

1. **Activity** s(t) = |w| (deg/s) of the standardised stream (gaps filled), low-passed at ``smooth_hz``,
   either *causally* (one-pass Butterworth: what a detector running on the incoming samples can compute) or
   with *zero phase* (cuefree v0, needs the whole recording). The causal variant lags by the filter's group
   delay; :func:`causal_delay_ms` gives it (a constant of the filter, never fitted) and the segment bounds
   are moved back by it.
2. **Hysteresis bouts** (:func:`find_segments`). A bout is a maximal stretch of activity between two
   stillnesses of at least ``end_sustain_ms`` (s < ``lo_dps``). It becomes a candidate segment when s stays
   above ``hi_dps`` for at least ``sustain_ms`` inside it. Onset = first non-still sample of the bout
   (back-dated, as online_v1 and v0 do), end = first sample of the stillness that ends it. Unlike v0 there
   is NO requirement of 1 s of stillness before a bout: the stillness before and after become features.
3. **Per segment** (:func:`segment_table`): ``standardize.canonical`` on the bounds (the same function that
   made the training windows), ~30 hand-agnostic features (every one is even under the left/right mirror,
   checked by a test) and the time-normalised (17, 64) window that a hand classifier reads.
4. **Labels** (training rows only; :func:`label_segments`): ``cuefree.classify`` against the app's own
   events, i.e. 'approach' (IoU >= 0.5 with the stored approach), 'partial' (overlaps it, IoU < 0.5),
   'return' (the movement after the end cue / within the recorded return) or 'other' (earlier or later
   movement).

Feature families (names in :data:`FEATURE_SETS`):
* ``physics``: duration, rotation and acceleration magnitudes, the start and end poses (up vector in the
  device frame; |up_x| because up_x changes sign under the mirror), where in the segment the rotation and
  the acceleration happen, and a ZUPT double integration of the linear acceleration (displacement along the
  vertical and along the start pose's device axes). A return is a time-reversed approach, so the pose change,
  the displacement and the timing asymmetries change sign between the two.
* ``context`` (set ``full``): the stillness before (capped at 3 s) and after (capped at the look-ahead the online
  detector waits for), whether the bout started with the recording, its index and the previous bout's duration.
  These depend on how people behave between movements and on the recording protocol.
* ``robust`` (the production set): physics + the stillness before capped at the INSTRUCTED stillness
  (``INSTRUCTED_STILLNESS_S``) + the stillness after; no recording artefacts (start of the recording, index).
* ``grip`` (an evaluated alternative): ``robust`` with the four quantities that change sign when the phone is held
  with the screen AWAY from the face (``SCREEN_ODD_FEATURES``: the screen-normal component of the up vector at both
  ends, its change and the displacement along the screen normal) multiplied by s = sign(up_z0 + up_z1), the side
  of the phone that faces upwards over the segment. Under the screen flip (180 deg about the device y axis,
  ``standardize.screen_flip_channels``) up_z and s both change sign, so every ``grip`` feature is exactly
  invariant, and also even under the left/right mirror (both checked by tests). The price: s is a per-segment
  guess, wrong whenever the screen tips past the vertical during the segment.

**Rest-pose grip check** (:func:`rest_up_z`, used by the inference page and the stress test, never by a model):
the screen-normal component of the up vector while the phone rests at arm's length before the first movement of
the stream. It is positive when the screen faces the user and tilts upwards, and changes sign with the screen-away
grip; the page asks to turn the screen towards the face when it is below ``m4_config.GRIP_GATE_UPZ``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumulative_trapezoid

from motionlab.preprocess import fill_gaps_linear, integrate_gyro, lowpass

from stage2 import config as C
from stage2.cuefree import Detection, classify
from stage2.standardize import CHANNELS, Stream, canonical, time_normalised

# ───────────────────────────── parameters ─────────────────────────────


@dataclass(frozen=True)
class SegParams:
    """Candidate segmentation (thresholds are the app's own: still_gyro_dps 20, onset floor 30 deg/s,
    end_sustain 250 ms; v0 uses the same)."""
    lo_dps: float = 20.0            # still below this (activity)
    hi_dps: float = 30.0            # a movement exceeds this ...
    sustain_ms: float = 100.0       # ... for at least this long
    end_sustain_ms: float = 250.0   # a stillness this long ends a bout
    smooth_hz: float = 5.0          # activity low-pass
    causal: bool = True             # one-pass filter (online) or zero-phase (offline, v0)
    min_duration_ms: float = 150.0  # shorter bouts are dropped (a tap, a tremor burst)
    max_duration_s: float = 12.0    # longer bouts are not a reach (kept but flagged)
    still_cap_s: float = 3.0        # stillness before a bout: features see at most this much
    lookahead_s: float = 0.5        # stillness after the end the detector waits for (decision latency)
    crop_s: float = 1.0             # canonical() runs on [onset - crop, end + crop] (enough for its rest windows)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT = SegParams()


def causal_delay_ms(p: SegParams, fs_hz: float = C.FS_HZ) -> float:
    """Group delay at 0.5 Hz of the causal activity filter (ms): subtracted from causal bounds. 0 for the
    zero-phase filter. A property of the filter, computed, not fitted."""
    if not p.causal or not p.smooth_hz:
        return 0.0
    sos = signal.butter(2, p.smooth_hz, btype='low', fs=fs_hz, output='sos')
    b, a = signal.sos2tf(sos)
    w, gd = signal.group_delay((b, a), w=[0.5], fs=fs_hz)
    return float(gd[0] / fs_hz * 1000.0)


# ───────────────────────────── activity and bouts ─────────────────────────────


def activity(t: np.ndarray, gyro: np.ndarray, p: SegParams = DEFAULT, fs_hz: float = C.FS_HZ) -> np.ndarray:
    """|w| (deg/s), gaps filled, low-passed at p.smooth_hz (causal: initial state at the first value)."""
    w = fill_gaps_linear(np.asarray(t, dtype=float), np.asarray(gyro, dtype=float))
    s = np.linalg.norm(w, axis=1)
    s = np.where(np.isfinite(s), s, 0.0)
    if not p.smooth_hz or s.size < 12:
        return s
    if p.causal:
        sos = signal.butter(2, p.smooth_hz, btype='low', fs=fs_hz, output='sos')
        zi = signal.sosfilt_zi(sos) * s[0]
        out, _ = signal.sosfilt(sos, s, zi=zi)
        return out
    return lowpass(s, fs_hz, p.smooth_hz, order=2)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return []
    dm = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(dm == 1).tolist(), np.flatnonzero(dm == -1).tolist()))


@dataclass
class Segment:
    onset_ms: float
    end_ms: float
    i_on: int
    i_end: int
    decided_ms: float        # when an online detector knows this segment (end stillness + look-ahead + lag)
    still_before_s: float    # the stillness that preceded the bout (activity signal), capped at still_cap_s
    still_after_s: float     # the stillness after the end, capped at lookahead_s
    started_active: bool     # the bout began with the recording (no stillness seen before it)
    ended: bool              # False when the recording stopped before the stillness that ends it
    peak_dps: float
    index: int = 0

    def detection(self) -> Detection:
        return Detection(self.onset_ms, self.end_ms, self.i_on, self.i_end, self.still_before_s, self.peak_dps,
                         self.ended)


def find_segments(t: np.ndarray, gyro: np.ndarray, p: SegParams = DEFAULT, fs_hz: float = C.FS_HZ,
                  s: np.ndarray | None = None) -> list[Segment]:
    """Every candidate segment of a stream in time order (module docstring, step 2). Causal bounds are moved
    back by the filter delay; ``decided_ms`` is when a causal detector would have confirmed the segment."""
    t = np.asarray(t, dtype=float)
    n = t.size
    if n < 3:
        return []
    dt = float(np.median(np.diff(t)))
    if s is None:
        s = activity(t, gyro, p, fs_hz)
    lag = causal_delay_ms(p, fs_hz)
    n_end = max(1, int(round(p.end_sustain_ms / dt)))
    n_move = max(1, int(round(p.sustain_ms / dt)))
    n_look = max(n_end, int(round(p.lookahead_s * 1000.0 / dt)))
    still = s < p.lo_dps
    # separators: still runs of at least n_end samples (a run touching the stream start also separates)
    seps = [(a, b) for a, b in _runs(still) if (b - a) >= n_end or a == 0]
    bounds: list[tuple[int, int, int | None]] = []   # (bout start, bout stop exclusive, index of the previous separator)
    prev_end = 0
    prev_sep = None
    for k, (a, b) in enumerate(seps):
        if a > prev_end:
            bounds.append((prev_end, a, prev_sep))
        prev_end = b
        prev_sep = k
    if prev_end < n:
        bounds.append((prev_end, n, prev_sep))
    out: list[Segment] = []
    for (a, b, ks) in bounds:
        above = s[a:b] > p.hi_dps
        if not any(r1 - r0 >= n_move for r0, r1 in _runs(above)):
            continue
        ended = b < n
        i_on, i_end = a, (b if ended else n - 1)
        on_ms = float(t[i_on]) - lag
        end_ms = float(t[i_end]) - lag
        if end_ms - on_ms < p.min_duration_ms:
            continue
        if ks is None:
            sb, started_active = 0.0, True
        else:
            sa, sb_ = seps[ks]
            sb, started_active = float((t[sb_ - 1] - t[sa] + dt) / 1000.0), False
        # stillness after the end, as far as the detector looks ahead
        if ended:
            j = i_end
            while j < n and still[j] and j - i_end < n_look:
                j += 1
            sa_s = float((j - i_end) * dt / 1000.0)
            dec_i = min(n - 1, i_end + n_look - 1)
        else:
            sa_s, dec_i = 0.0, n - 1
        out.append(Segment(on_ms, end_ms, int(i_on), int(i_end), float(t[dec_i]),
                           min(sb, p.still_cap_s), min(sa_s, p.lookahead_s), started_active, ended,
                           float(np.max(s[i_on:i_end + 1])), len(out)))
    return out


# ───────────────────────────── per-segment canonical window and features ─────────────────────────────

PHYSICS_FEATURES: tuple[str, ...] = (
    'dur_s', 'log_dur', 'peak_w', 'mean_w', 'rot_total', 'peak_a', 'mean_a', 'abs_net_yaw', 'tilt_end',
    'peak_tilt_rate', 'peak_abs_yaw_rate', 't_peak_w', 'w_first_frac', 't_peak_a', 'a_last_frac', 'a_first_frac',
    'up_y0', 'up_z0', 'abs_up_x0', 'up_y1', 'up_z1', 'abs_up_x1', 'd_up_y', 'd_up_z', 'd_abs_up_x',
    'disp_vert', 'disp_y0', 'disp_z0', 'abs_disp_x0', 'disp_norm', 'v_up_peak', 'v_down_peak', 'closure',
)
CONTEXT_FEATURES: tuple[str, ...] = ('still_before', 'still_after', 'started_active', 'seg_index', 'prev_dur')
#: The stillness the inference page asks for before each touch (s). The 'robust' feature set sees the stillness
#: before a bout only up to this value: the collection app enforced much longer rests (median about 2.4 s), and a
#: classifier that learns them rejects real approaches preceded by the instructed 1 s (stress test, M4_03).
INSTRUCTED_STILLNESS_S = 1.0
ROBUST_CONTEXT: tuple[str, ...] = ('still_before_inst', 'still_after')
#: physics features that change sign under the screen flip (every other feature is invariant: magnitudes,
#: gravity-frame quantities, up_y, |up_x|, displacement along device y, |displacement along device x|)
SCREEN_ODD_FEATURES: tuple[str, ...] = ('up_z0', 'up_z1', 'd_up_z', 'disp_z0')
GRIP_PHYSICS: tuple[str, ...] = tuple(f'g_{f}' if f in SCREEN_ODD_FEATURES else f for f in PHYSICS_FEATURES)
FEATURE_SETS: dict[str, tuple[str, ...]] = {'physics': PHYSICS_FEATURES,
                                            'full': PHYSICS_FEATURES + CONTEXT_FEATURES,
                                            'robust': PHYSICS_FEATURES + ROBUST_CONTEXT,
                                            'grip': GRIP_PHYSICS + ROBUST_CONTEXT}
ALL_FEATURES: tuple[str, ...] = tuple(dict.fromkeys(PHYSICS_FEATURES + CONTEXT_FEATURES + ROBUST_CONTEXT
                                                    + GRIP_PHYSICS))
CLASSES: tuple[str, ...] = ('approach', 'return', 'other')


def as_stream(st: Mapping[str, np.ndarray]) -> Stream:
    """A ``standardize.Stream`` from a stream dict (``Stage2Data.stream(i)``: t, f, acc, gyro, valid). No
    calibration magnitude and no online baseline gravity: production has neither."""
    t = np.asarray(st['t'], dtype=float)
    valid = np.asarray(st.get('valid', np.ones(t.size, bool)), dtype=bool)
    return Stream(trial_id=None, t=t, f=np.asarray(st['f'], dtype=float), acc=np.asarray(st['acc'], dtype=float),
                  gyro=np.asarray(st['gyro'], dtype=float), valid=valid, fs_hz=C.FS_HZ)


def crop_stream(st: Mapping[str, np.ndarray], start_ms: float, stop_ms: float) -> dict[str, np.ndarray]:
    t = np.asarray(st['t'])
    sel = (t >= start_ms) & (t <= stop_ms)
    return {k: np.asarray(v)[sel] for k, v in st.items() if k in ('t', 'f', 'acc', 'gyro', 'valid', 'phase')}


def window_for(st: Mapping[str, np.ndarray], onset_ms: float, end_ms: float, crop_s: float = DEFAULT.crop_s,
               post_s: float | None = None) -> tuple[np.ndarray, Any]:
    """((17, 64) time-normalised window, Canonical) for bounds on a stream, cropped to [onset - crop_s,
    end + post_s] first (post_s defaults to crop_s). With post_s = the detector's look-ahead the window uses
    no sample the online detector has not seen when it decides."""
    post_s = crop_s if post_s is None else post_s
    c = crop_stream(st, onset_ms - crop_s * 1000.0, end_ms + post_s * 1000.0)
    can = canonical(as_stream(c), onset_ms, end_ms)
    return time_normalised(can, onset_ms, end_ms), can


def _kinematics(can: Any) -> dict[str, float]:
    """ZUPT double integration of the linear acceleration over [onset, end], in the device frame of the
    onset (displacement along the vertical and the start pose's device axes; peak vertical velocities)."""
    i0, i1 = can.i_on, can.i_end
    t = can.t[i0:i1 + 1] / 1000.0
    X = can.X
    a = np.stack([X[CHANNELS.index(c)] for c in ('acc_x', 'acc_y', 'acc_z')], axis=1)[i0:i1 + 1]
    w = np.stack([X[CHANNELS.index(c)] for c in ('gyro_x', 'gyro_y', 'gyro_z')], axis=1)[i0:i1 + 1]
    u0 = np.array([X[CHANNELS.index(c)][i0] for c in ('up_x', 'up_y', 'up_z')])
    if t.size < 3 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(w))):
        return {k: math.nan for k in ('disp_vert', 'disp_y0', 'disp_z0', 'abs_disp_x0', 'disp_norm', 'v_up_peak',
                                      'v_down_peak')}
    R, _ = integrate_gyro(t * 1000.0, w)                    # device(k) -> device(onset)
    aw = np.einsum('nij,nj->ni', R, a)
    v = cumulative_trapezoid(aw, t, axis=0, initial=0.0)
    T = t[-1] - t[0]
    if T > 0:
        v = v - np.outer((t - t[0]) / T, v[-1])             # zero velocity at both rests (ZUPT)
    dp = np.trapezoid(v, t, axis=0) if hasattr(np, 'trapezoid') else np.trapz(v, t, axis=0)
    vv = v @ u0
    return {'disp_vert': float(dp @ u0), 'disp_y0': float(dp[1]), 'disp_z0': float(dp[2]),
            'abs_disp_x0': float(abs(dp[0])), 'disp_norm': float(np.linalg.norm(dp)),
            'v_up_peak': float(max(0.0, vv.max())), 'v_down_peak': float(max(0.0, -vv.min()))}


def segment_features(can: Any, seg: Segment, prev: Segment | None) -> dict[str, float]:
    """Hand-agnostic (mirror-even) features of one segment from its Canonical (module docstring)."""
    X, i0, i1 = can.X, can.i_on, can.i_end
    ch = lambda c: X[CHANNELS.index(c)][i0:i1 + 1]  # noqa: E731
    t = can.t[i0:i1 + 1]
    dur = max((seg.end_ms - seg.onset_ms) / 1000.0, 1e-3)
    wm, am = ch('gyro_mag'), ch('acc_mag')
    n = wm.size
    tt = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    tot_w = float(np.sum(wm)) or 1e-9
    half = tt <= 0.5
    q = max(1, int(round(0.2 * n)))
    amean = float(np.mean(am)) or 1e-9
    up0 = np.array([X[CHANNELS.index(c)][i0] for c in ('up_x', 'up_y', 'up_z')])
    up1 = np.array([X[CHANNELS.index(c)][i1] for c in ('up_x', 'up_y', 'up_z')])
    f = {
        'dur_s': dur, 'log_dur': math.log(dur), 'peak_w': float(wm.max()), 'mean_w': float(wm.mean()),
        'rot_total': float(np.sum(wm) / C.FS_HZ), 'peak_a': float(am.max()), 'mean_a': amean,
        'abs_net_yaw': float(abs(ch('yaw_cum')[-1] - ch('yaw_cum')[0])), 'tilt_end': float(ch('tilt_from_start')[-1]),
        'peak_tilt_rate': float(ch('gyro_horiz').max()), 'peak_abs_yaw_rate': float(np.abs(ch('yaw_rate')).max()),
        't_peak_w': float(tt[int(np.argmax(wm))]), 'w_first_frac': float(np.sum(wm[half]) / tot_w),
        't_peak_a': float(tt[int(np.argmax(am))]), 'a_last_frac': float(np.mean(am[-q:]) / amean),
        'a_first_frac': float(np.mean(am[:q]) / amean),
        'up_y0': float(up0[1]), 'up_z0': float(up0[2]), 'abs_up_x0': float(abs(up0[0])),
        'up_y1': float(up1[1]), 'up_z1': float(up1[2]), 'abs_up_x1': float(abs(up1[0])),
        'd_up_y': float(up1[1] - up0[1]), 'd_up_z': float(up1[2] - up0[2]),
        'd_abs_up_x': float(abs(up1[0]) - abs(up0[0])),
        'closure': float(can.closure_deg) if math.isfinite(can.closure_deg) else 0.0,
        'still_before': float(seg.still_before_s), 'still_after': float(seg.still_after_s),
        'still_before_inst': float(min(seg.still_before_s, INSTRUCTED_STILLNESS_S)),
        'started_active': float(seg.started_active), 'seg_index': float(min(seg.index, 5)),
        'prev_dur': float(0.0 if prev is None else min((prev.end_ms - prev.onset_ms) / 1000.0, 10.0)),
    }
    f.update(_kinematics(can))
    # grip-normalised copies of the screen-odd features (module docstring, set 'grip')
    tot = f['up_z0'] + f['up_z1']
    s = -1.0 if (math.isfinite(tot) and tot < 0) else 1.0
    for k in SCREEN_ODD_FEATURES:
        v = f[k]
        f[f'g_{k}'] = float(v * s) if math.isfinite(v) else math.nan
    return f


def segment_table(st: Mapping[str, np.ndarray], p: SegParams = DEFAULT, *, s: np.ndarray | None = None
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """(one row per candidate segment: bounds, timing and features; (n_seg, 17, 64) windows) of one stream.
    Label-free: a function of the stream alone."""
    segs = find_segments(st['t'], st['gyro'], p, s=s)
    rows, wins = [], []
    prev = None
    for sg in segs:
        try:
            win, can = window_for(st, sg.onset_ms, sg.end_ms, p.crop_s, p.lookahead_s)
            feats = segment_features(can, sg, prev)
            ok = True
        except ValueError:
            win, feats, ok = np.full((len(CHANNELS), C.N_NORM), np.nan, np.float32), {}, False
        rows.append({'seg': sg.index, 'onset_ms': sg.onset_ms, 'end_ms': sg.end_ms, 'decided_ms': sg.decided_ms,
                     'ended': sg.ended, 'peak_dps': sg.peak_dps, 'canonical_ok': ok, **feats})
        wins.append(win)
        prev = sg
    df = pd.DataFrame(rows)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    W = np.stack(wins).astype(np.float32) if wins else np.zeros((0, len(CHANNELS), C.N_NORM), np.float32)
    return df, W


# ───────────────────────────── labels from the app's events (training rows only) ─────────────────────────────

LABEL_OF_OUTCOME = {'approach': 'approach', 'approach_partial': 'partial', 'return': 'return',
                    'earlier_movement': 'other', 'later_movement': 'other', 'none': 'other'}


def classify_arrays(on: np.ndarray, end: np.ndarray, o_on: np.ndarray, o_end: np.ndarray, cue: np.ndarray,
                    r_on: np.ndarray, r_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``cuefree.classify`` (same rules; a test checks the equality): (outcome, IoU) per segment.
    Every argument is an array over segments (NaN = event not recorded)."""
    on, end, o_on, o_end = (np.asarray(a, dtype=float) for a in (on, end, o_on, o_end))
    cue, r_on, r_end = (np.asarray(a, dtype=float) for a in (cue, r_on, r_end))
    inter = np.maximum(0.0, np.minimum(end, o_end) - np.maximum(on, o_on))
    union = np.maximum(end, o_end) - np.minimum(on, o_on)
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    hits_return = np.isfinite(r_on) & (end > r_on) & (~np.isfinite(r_end) | (on < r_end))
    after_cue = np.isfinite(cue) & (on >= cue)
    ret = hits_return | (~np.isfinite(r_on) & after_cue)
    out = np.where(iou >= 0.5, 'approach', np.where(inter > 0, 'approach_partial',
                   np.where(end <= o_on, 'earlier_movement', np.where(ret, 'return', 'later_movement'))))
    return out.astype(object), iou


def label_segments(segs: pd.DataFrame, events: Mapping[str, Any]) -> pd.DataFrame:
    """outcome (cuefree.classify rules), label (approach / partial / return / other), IoU and bound errors of
    every segment against one recording's events (onset_ms, end_ms, end_detected_ms, return_onset_ms,
    return_end_ms)."""
    n = len(segs)
    if not n:
        return pd.DataFrame(columns=['outcome', 'label', 'iou', 'onset_error_ms', 'end_error_ms'])
    ev = {c: np.full(n, np.nan if _num(events.get(c)) is None else float(events.get(c))) for c in EVENT_COLUMNS}
    on, end = segs['onset_ms'].to_numpy(float), segs['end_ms'].to_numpy(float)
    oc, iou = classify_arrays(on, end, ev['onset_ms'], ev['end_ms'], ev['end_detected_ms'], ev['return_onset_ms'],
                              ev['return_end_ms'])
    return pd.DataFrame({'outcome': oc, 'label': [LABEL_OF_OUTCOME[o] for o in oc], 'iou': iou,
                         'onset_error_ms': on - ev['onset_ms'], 'end_error_ms': end - ev['end_ms']}, index=segs.index)


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


EVENT_COLUMNS = ('onset_ms', 'end_ms', 'end_detected_ms', 'return_onset_ms', 'return_end_ms')


def mirror_stream(st: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The left/right mirror image of a stream dict (f, acc -> S v; gyro -> -S w; S = diag(-1, 1, 1))."""
    out = {k: np.array(v, copy=True) for k, v in st.items()}
    for k in ('f', 'acc'):
        out[k] = out[k] * np.array([-1.0, 1.0, 1.0])
    out['gyro'] = out['gyro'] * np.array([1.0, -1.0, -1.0])
    return out


SCREEN_FLIP = np.array([-1.0, 1.0, -1.0])


def screen_flip_stream(st: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The same movement with the phone turned 180 deg about its y axis (screen away from the face): f, acc and
    the angular velocity -> F v with F = diag(-1, 1, -1), a proper rotation (the sign pattern of
    ``standardize.screen_flip_channels`` on the device vectors). The hand label is unchanged."""
    out = {k: np.array(v, copy=True) for k, v in st.items()}
    for k in ('f', 'acc', 'gyro'):
        out[k] = out[k] * SCREEN_FLIP
    return out


def rest_up_z(st: Mapping[str, np.ndarray], p: SegParams = DEFAULT, min_s: float = 0.25,
              fs_hz: float = C.FS_HZ) -> float:
    """The rest pose's screen-normal up component (module docstring): the median of f_z / |f| over the still
    samples (causal activity below ``p.lo_dps``) before the onset of the stream's first candidate bout; when fewer
    than ``min_s`` seconds of such samples exist, over the first ``min_s`` seconds of the stream. Causal (only
    samples before the first movement), label-free, even under the left/right mirror and odd under the screen flip
    (tests). NaN when the stream has no finite specific force."""
    t = np.asarray(st['t'], dtype=float)
    f = np.asarray(st['f'], dtype=float)
    if t.size == 0:
        return math.nan
    s = activity(t, st['gyro'], p, fs_hz)
    segs = find_segments(t, st['gyro'], p, fs_hz, s=s)
    first_on = segs[0].onset_ms if segs else math.inf
    nrm = np.linalg.norm(f, axis=1)
    ok = np.isfinite(nrm) & (nrm > 0)
    upz = np.where(ok, f[:, 2] / np.where(ok, nrm, 1.0), np.nan)
    sel = ok & (t < first_on) & (s < p.lo_dps)
    if sel.sum() < max(1, int(round(min_s * fs_hz))):
        sel = ok & (t < t[0] + min_s * 1000.0)
    return float(np.median(upz[sel])) if sel.any() else math.nan


def with_params(p: SegParams, **kw: Any) -> SegParams:
    return replace(p, **kw)


def segment_all(streams: Sequence[Mapping[str, np.ndarray]], p: SegParams = DEFAULT, *, progress: str = ''
                ) -> tuple[pd.DataFrame, np.ndarray]:
    """segment_table over many streams: one table (column ``stream`` = position in ``streams``) and the
    stacked windows (row-aligned with the table)."""
    import time
    t0 = time.time()
    tabs, wins = [], []
    for k, st in enumerate(streams):
        df, W = segment_table(st, p)
        df.insert(0, 'stream', k)
        tabs.append(df)
        wins.append(W)
        if progress and (k + 1) % 200 == 0:
            print(f'  {progress}: segmented {k + 1}/{len(streams)} streams ({time.time() - t0:.0f} s)', flush=True)
    tab = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    W = np.concatenate(wins) if wins else np.zeros((0, len(CHANNELS), C.N_NORM), np.float32)
    return tab, W
