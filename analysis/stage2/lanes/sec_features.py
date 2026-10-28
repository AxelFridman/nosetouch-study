"""Lane sec: per-trial features of the secondary tasks (instructed speed, instructed grip, dominant hand).

Every feature is a function of ONE trial (its cached arrays, its own 60 Hz stream and its own approach duration), so
the features are computed before the evaluation loops (README section 5.2: per-trial transforms may be computed
beforehand; anything that spans trials is fitted inside the folds). Three sets:

* **EVEN** (:func:`even_features`, :data:`EVEN_FEATURES`): mirror-INVARIANT kinematics. Each one is identical for a
  trial and its left/right mirror image (``standardize.mirror_channels`` on the arrays, w -> -S w on the stream;
  ``tests/test_sec.py``). They say how long, how fast, how far and how smoothly the phone moved, never which way it
  turned, so they cannot encode which hand moved: any dominant-hand signal they carry comes from HOW the dominant
  hand moves.
* **GRIP** (:func:`grip_features`, :data:`GRIP_FEATURES`): the device-frame orientation (unit up vector in the still
  hold before the onset, at the onset and at the end) and the mean device-frame angular velocity. The grip is a
  device-frame property (README section 3: the device channels depend on the grip, the gravity channels do not).
* **RICH** (:func:`rich_features`): summary statistics of every channel over the approach (mean, sd, min, max,
  first, last, 8 time bins) and of the still hold before the onset, plus EVEN: the input of the gradient-boosting
  models.

Sources: angular kinematics come from the trial's 60 Hz standardised stream between the oracle bounds (``onset_ms``
.. ``end_ms``: the whole approach at its true sampling rate, whatever its duration); the gravity-frame angles and
the linear kinematics come from the cached time-normalised approach (64 samples, dt = duration / 63); the hold
before the onset from the cached fixed window (its first 0.5 s, before the onset, never frozen).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from stage2.standardize import CHANNELS

FS_HZ = 60.0
FIXED_ONSET_INDEX = 30          # d.fixed starts 0.5 s before the oracle onset at 60 Hz
EDGE = 3                        # approach samples averaged for the orientation at the onset / at the end
N_BINS = 8                      # time bins of the RICH set
SPARC_FC_HZ = 10.0              # SPARC (Balasubramanian et al. 2015): cut-off frequency
SPARC_AMP_TH = 0.05             # ... amplitude threshold of the normalised spectrum
SPARC_PAD = 4                   # ... zero padding (powers of two above the signal length)
PEAK_PROMINENCE = 0.10          # sub-movements: local maxima of |w| with a prominence of at least 10 % of the peak

#: name -> (unit, source, definition); every one is mirror-invariant (parity 'even')
EVEN_FEATURES: dict[str, tuple[str, str, str]] = {
    'duration_log': ('log s', 'oracle bounds', 'natural log of the approach duration'),
    'ang_speed_peak': ('deg/s', '60 Hz stream', 'peak angular speed |w|'),
    'ang_speed_mean': ('deg/s', '60 Hz stream', 'mean angular speed |w|'),
    'ang_peak_time': ('fraction', '60 Hz stream', 'time of the |w| peak as a fraction of the approach'),
    'ang_path': ('deg', '60 Hz stream', 'angular path length: integral of |w| (total rotation)'),
    'ang_sparc': ('1', '60 Hz stream', 'SPARC smoothness of |w|(t) (spectral arc length; closer to 0 = smoother)'),
    'ang_ldlj': ('1', '60 Hz stream', 'log dimensionless jerk of |w|(t) (closer to 0 = smoother)'),
    'ang_n_peaks': ('count', '60 Hz stream', 'local maxima of |w| with prominence >= 10 % of its peak'),
    'tilt_net': ('deg', 'approach', 'angle between the up vector at the onset and at the end'),
    'yaw_net_abs': ('deg', 'approach', 'absolute net rotation about gravity |yaw_cum(end)| (the sign would carry the hand)'),
    'tilt_path': ('deg', 'approach', 'integral of the tilting rate |w - (w.u)u|'),
    'yaw_path': ('deg', 'approach', 'integral of the absolute yaw rate |w.u|'),
    'acc_peak': ('m/s^2', 'approach', 'peak linear acceleration |a|'),
    'acc_mean': ('m/s^2', 'approach', 'mean linear acceleration |a|'),
    'lin_path': ('m', 'approach', 'linear path length of the phone: integral of |v|, v by strapdown integration '
                 'with zero velocity at both bounds'),
    'lin_speed_peak': ('m/s', 'approach', 'peak linear speed |v| (same integration)'),
}
#: name -> (unit, source, definition); device-frame, NOT mirror-invariant (the grip is a device-frame property)
GRIP_FEATURES: dict[str, tuple[str, str, str]] = {
    **{f'up_before_{a}': ('1', 'fixed window', f'up vector, device {a}, mean over the 0.5 s hold before the onset')
       for a in 'xyz'},
    **{f'up_onset_{a}': ('1', 'approach', f'up vector, device {a}, mean of the first {EDGE} approach samples')
       for a in 'xyz'},
    **{f'up_end_{a}': ('1', 'approach', f'up vector, device {a}, mean of the last {EDGE} approach samples')
       for a in 'xyz'},
    **{f'gyro_mean_{a}': ('deg/s', 'approach', f'mean angular velocity about device {a} over the approach')
       for a in 'xyz'},
}
RICH_STATS: tuple[str, ...] = ('mean', 'sd', 'min', 'max', 'first', 'last', *(f'b{k + 1}' for k in range(N_BINS)),
                               'before')


def _ch(name: str) -> int:
    return CHANNELS.index(name)


def _trapz(x: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Trapezoid integral along the last axis with one step dt per row."""
    return 0.5 * (x[..., 1:] + x[..., :-1]).sum(axis=-1) * dt


# ───────────────────────────── angular kinematics (60 Hz stream) ─────────────────────────────


def angular_speed(gyro: np.ndarray) -> np.ndarray:
    """|w| (deg/s) of an (n, 3) angular-velocity array. Mirror-invariant exactly: the mirror maps w to (w_x, -w_y,
    -w_z), and a negated float squares to the same bits."""
    g = np.asarray(gyro, dtype=np.float64)
    return np.sqrt(np.sum(g * g, axis=1))


def approach_gyro(stream: dict[str, np.ndarray], onset_ms: float, end_ms: float) -> np.ndarray:
    """The (n, 3) angular velocity of the stream samples inside [onset_ms, end_ms] (interpolated over any NaN)."""
    t = stream['t']
    sel = (t >= onset_ms) & (t <= end_ms)
    g = np.asarray(stream['gyro'][sel], dtype=np.float64)
    tt = t[sel]
    for k in range(3):
        bad = ~np.isfinite(g[:, k])
        if bad.any() and (~bad).sum() >= 2:
            g[bad, k] = np.interp(tt[bad], tt[~bad], g[~bad, k])
    return g


def sparc(speed: np.ndarray, fs: float = FS_HZ, *, padlevel: int = SPARC_PAD, fc: float = SPARC_FC_HZ,
          amp_th: float = SPARC_AMP_TH) -> float:
    """Spectral arc length of a speed profile (Balasubramanian, Melendez-Calderon, Roby-Brami, Burdet 2015, J
    NeuroEng Rehabil 12:112; their reference implementation): the arc length of the normalised magnitude spectrum up
    to the adaptive cut-off. Negative; closer to 0 = smoother."""
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 3 or not np.any(v > 0):
        return float('nan')
    nfft = int(2 ** (np.ceil(np.log2(v.size)) + padlevel))
    f = np.arange(nfft) * (fs / nfft)
    mf = np.abs(np.fft.fft(v, nfft))
    mf = mf / mf.max()
    keep = f <= fc
    f_sel, m_sel = f[keep], mf[keep]
    inx = np.flatnonzero(m_sel >= amp_th)
    f_sel, m_sel = f_sel[inx[0]:inx[-1] + 1], m_sel[inx[0]:inx[-1] + 1]
    if f_sel.size < 2:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f_sel) / (f_sel[-1] - f_sel[0])) ** 2 + np.diff(m_sel) ** 2)))


def ldlj(speed: np.ndarray, fs: float = FS_HZ) -> float:
    """Log dimensionless jerk of a speed profile (Balasubramanian et al. 2015): -ln(T^3 / v_peak^2 * integral of
    (d^2 v / dt^2)^2 dt). Negative; closer to 0 = smoother."""
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 4 or not np.any(v > 0):
        return float('nan')
    dt = 1.0 / fs
    jerk = np.diff(v, 2) / dt ** 2
    scale = (v.size * dt) ** 3 / np.max(np.abs(v)) ** 2
    return float(-np.log(scale * np.sum(jerk ** 2) * dt))


def angular_features(w: np.ndarray, fs: float = FS_HZ) -> dict[str, float]:
    """The 60-Hz-stream features of EVEN from one angular-speed profile |w|(t) (deg/s)."""
    w = np.asarray(w, dtype=np.float64)
    n = w.size
    dt = 1.0 / fs
    k = int(np.argmax(w))
    pk, _ = find_peaks(w, prominence=PEAK_PROMINENCE * float(w.max()))
    return {
        'ang_speed_peak': float(w.max()),
        'ang_speed_mean': float(np.mean(w)),
        'ang_peak_time': float(k / max(n - 1, 1)),
        'ang_path': float(0.5 * np.sum(w[1:] + w[:-1]) * dt),
        'ang_sparc': sparc(w, fs),
        'ang_ldlj': ldlj(w, fs),
        'ang_n_peaks': float(max(len(pk), 1)),
    }


# ───────────────────────────── linear kinematics (time-normalised approach) ─────────────────────────────


def _rodrigues(theta: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices of (N, 3) rotation vectors (radians)."""
    ang = np.linalg.norm(theta, axis=1)
    small = ang < 1e-12
    safe = np.where(small, 1.0, ang)
    k = theta / safe[:, None]
    K = np.zeros((theta.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.where(small, 0.0, np.sin(ang))[:, None, None]
    c = np.where(small, 0.0, 1.0 - np.cos(ang))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def strapdown_speed(acc: np.ndarray, gyro: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """|v| (m/s) along the approach: the device-frame linear acceleration (N, 3, T; gravity removed) is rotated into
    the onset frame with the gyroscope (N, 3, T, deg/s; body-frame rates, trapezoid rotation per step), integrated, and
    the linear drift is removed so that the velocity is zero at both bounds (the phone is still at arm's length and at
    the nose). Mirror-invariant: the mirror conjugates the rotations by S = diag(-1, 1, 1) and maps the velocity to
    S v, whose norm is unchanged."""
    a = np.asarray(acc, dtype=np.float64)
    w = np.deg2rad(np.asarray(gyro, dtype=np.float64))
    dt = np.asarray(dt, dtype=np.float64)
    N, _, T = a.shape
    R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    aw = np.empty((N, T, 3))
    for k in range(T):
        aw[:, k] = np.einsum('nij,nj->ni', R, a[:, :, k])
        if k < T - 1:
            R = R @ _rodrigues(0.5 * (w[:, :, k] + w[:, :, k + 1]) * dt[:, None])
    v = np.zeros((N, T, 3))
    v[:, 1:] = np.cumsum(0.5 * (aw[:, 1:] + aw[:, :-1]) * dt[:, None, None], axis=1)
    v -= v[:, -1:, :] * np.linspace(0.0, 1.0, T)[None, :, None]
    return np.linalg.norm(v, axis=2)


# ───────────────────────────── feature sets ─────────────────────────────


def approach_even(approach: np.ndarray, duration_s: np.ndarray) -> pd.DataFrame:
    """The approach-array part of EVEN (gravity-frame angles, accelerations, strapdown path) plus duration_log."""
    X = np.asarray(approach, dtype=np.float64)
    dur = np.asarray(duration_s, dtype=np.float64)
    T = X.shape[2]
    dt = dur / (T - 1)
    sp = strapdown_speed(X[:, [_ch('acc_x'), _ch('acc_y'), _ch('acc_z')]],
                         X[:, [_ch('gyro_x'), _ch('gyro_y'), _ch('gyro_z')]], dt)
    am = X[:, _ch('acc_mag')]
    return pd.DataFrame({
        'duration_log': np.log(dur),
        'tilt_net': X[:, _ch('tilt_from_start'), -1],
        'yaw_net_abs': np.abs(X[:, _ch('yaw_cum'), -1]),
        'tilt_path': _trapz(X[:, _ch('gyro_horiz')], dt),
        'yaw_path': _trapz(np.abs(X[:, _ch('yaw_rate')]), dt),
        'acc_peak': am.max(axis=1),
        'acc_mean': _trapz(am, dt) / dur,
        'lin_path': _trapz(sp, dt),
        'lin_speed_peak': sp.max(axis=1),
    })


def even_features(approach: np.ndarray, duration_s: np.ndarray, gyro_list: Sequence[np.ndarray]) -> pd.DataFrame:
    """EVEN (:data:`EVEN_FEATURES` order) of N trials: ``approach`` (N, C, 64) cached arrays, ``duration_s`` (N,),
    ``gyro_list`` the (n_i, 3) 60 Hz angular velocity of each approach (:func:`approach_gyro`)."""
    A = approach_even(approach, duration_s)
    W = pd.DataFrame([angular_features(angular_speed(g)) for g in gyro_list])
    F = pd.concat([A, W], axis=1)
    return F[list(EVEN_FEATURES)].astype(float)


def grip_features(approach: np.ndarray, fixed: np.ndarray, fixed_mask: np.ndarray) -> pd.DataFrame:
    """GRIP (:data:`GRIP_FEATURES` order): up vector before the onset (masked mean of the fixed window's first 0.5 s;
    the approach start when no sample there is valid), at the onset and at the end; mean device-frame angular
    velocity over the approach."""
    X = np.asarray(approach, dtype=np.float64)
    Fx = np.asarray(fixed, dtype=np.float64)[:, :, :FIXED_ONSET_INDEX]
    m = np.asarray(fixed_mask, dtype=bool)[:, :FIXED_ONSET_INDEX].astype(float)
    out: dict[str, np.ndarray] = {}
    for a in 'xyz':
        c = _ch(f'up_{a}')
        nv = m.sum(axis=1)
        pre = np.where(nv > 0, (Fx[:, c] * m).sum(axis=1) / np.maximum(nv, 1), X[:, c, :EDGE].mean(axis=1))
        out[f'up_before_{a}'] = pre
    for a in 'xyz':
        out[f'up_onset_{a}'] = X[:, _ch(f'up_{a}'), :EDGE].mean(axis=1)
    for a in 'xyz':
        out[f'up_end_{a}'] = X[:, _ch(f'up_{a}'), -EDGE:].mean(axis=1)
    for a in 'xyz':
        out[f'gyro_mean_{a}'] = X[:, _ch(f'gyro_{a}')].mean(axis=1)
    return pd.DataFrame(out)[list(GRIP_FEATURES)].astype(float)


def rich_features(approach: np.ndarray, fixed: np.ndarray, fixed_mask: np.ndarray, even: pd.DataFrame) -> pd.DataFrame:
    """RICH: per channel, over the approach: mean, sd, min, max, first / last (mean of EDGE samples), N_BINS bin means;
    over the hold before the onset: the masked mean ('before'); then the EVEN features."""
    X = np.asarray(approach, dtype=np.float64)
    Fx = np.asarray(fixed, dtype=np.float64)[:, :, :FIXED_ONSET_INDEX]
    m = np.asarray(fixed_mask, dtype=bool)[:, :FIXED_ONSET_INDEX].astype(float)
    nv = np.maximum(m.sum(axis=1), 1)[:, None]
    T = X.shape[2]
    edges = np.linspace(0, T, N_BINS + 1).round().astype(int)
    cols: dict[str, np.ndarray] = {}
    before = (Fx * m[:, None, :]).sum(axis=2) / nv
    for j, c in enumerate(CHANNELS):
        x = X[:, j]
        cols[f'{c}_mean'] = x.mean(axis=1)
        cols[f'{c}_sd'] = x.std(axis=1)
        cols[f'{c}_min'] = x.min(axis=1)
        cols[f'{c}_max'] = x.max(axis=1)
        cols[f'{c}_first'] = x[:, :EDGE].mean(axis=1)
        cols[f'{c}_last'] = x[:, -EDGE:].mean(axis=1)
        for k in range(N_BINS):
            cols[f'{c}_b{k + 1}'] = x[:, edges[k]:edges[k + 1]].mean(axis=1)
        cols[f'{c}_before'] = before[:, j]
    R = pd.DataFrame(cols)
    return pd.concat([R, even.reset_index(drop=True)], axis=1).astype(float)


def gyro_profiles(d: Any) -> list[np.ndarray]:
    """The 60 Hz angular velocity of every approach of a Stage2Data (oracle bounds of its trial table)."""
    t = d.trials
    return [approach_gyro(d.stream(i), float(t['onset_ms'].iloc[i]), float(t['end_ms'].iloc[i])) for i in range(d.n)]


def build_all(d: Any) -> dict[str, pd.DataFrame]:
    """{'even', 'grip', 'rich'} feature tables of a Stage2Data, rows aligned with ``d.trials``."""
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    even = even_features(d.approach, dur, gyro_profiles(d))
    grip = grip_features(d.approach, d.fixed, d.fixed_mask)
    rich = rich_features(d.approach, d.fixed, d.fixed_mask, even)
    for name, F in (('even', even), ('grip', grip), ('rich', rich)):
        if not np.isfinite(F.to_numpy()).all():
            bad = F.columns[~np.isfinite(F.to_numpy()).all(axis=0)].tolist()
            raise ValueError(f'sec features {name}: non-finite values in {bad}')
    return {'even': even, 'grip': grip, 'rich': rich}


def catalogue() -> pd.DataFrame:
    """The interpretable features: set, name, unit, source, parity under the left/right mirror, definition."""
    rows = [{'set': 'even', 'feature': k, 'unit': u, 'source': s, 'parity': 'even (invariant)', 'definition': dfn}
            for k, (u, s, dfn) in EVEN_FEATURES.items()]
    rows += [{'set': 'grip', 'feature': k, 'unit': u, 'source': s, 'parity': 'device frame (not invariant)',
              'definition': dfn} for k, (u, s, dfn) in GRIP_FEATURES.items()]
    return pd.DataFrame(rows)
