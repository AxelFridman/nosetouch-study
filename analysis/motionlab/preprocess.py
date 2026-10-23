"""Signal preprocessing and 3D rotation math.

* :func:`resample_uniform` — linear interpolation onto a uniform grid (default 100 Hz). Output samples
  that fall in a gap longer than ``max_gap_ms`` (default 100 ms) between two valid input samples, or
  outside the input range, are NaN: gaps are never bridged silently.
* :func:`lowpass` — zero-phase Butterworth (``sosfiltfilt``), applied separately to every finite run of
  samples so that NaN gaps neither propagate nor get smeared.
* :func:`gyro_gravity` (``gravity_v1``) — gravity during the movement: the mean specific force of the
  rest windows before onset and after the end, propagated with the gyroscope (dg/dt = -w x g) forward
  from the onset and backward from the end, blended linearly in time (docs/05 section 2.3).
* :func:`estimate_gravity` — the platform's own gravity, ``accg - acc`` (or a low-pass of ``accg``):
  a quality-control comparison only, because accelerometer-corrected fusion mis-estimates the tilt
  while the hand accelerates.
* :func:`chrome_round` / :func:`quantize_like_chrome` — Chromium's 0.1 m/s^2 / 0.1 deg/s rounding.
* :func:`repeated_sample_mask` — duplicated devicemotion snapshots (iOS timer repeats).
* W3C DeviceOrientation Euler angles (intrinsic Z-X'-Y'': R = Rz(alpha)·Rx(beta)·Ry(gamma), R maps
  DEVICE-frame vectors to EARTH-frame vectors) <-> rotation matrices <-> quaternions [x, y, z, w].
* :func:`integrate_gyro` — device orientation over time relative to the first sample.

Everything here is linear in the vector components or covariant under rotations, which is what makes
the chiral/invariant features exactly invariant to how the phone is gripped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from .conventions import DEG2RAD, GRAVITY, RAD2DEG

# numpy >= 2.0 renamed trapz to trapezoid; requirements allow 1.26.
trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# ───────────────────────────── resampling / filtering ─────────────────────────────


def _prepare_times(t_ms: Any, values: Any) -> tuple[np.ndarray, np.ndarray, bool]:
    t = np.asarray(t_ms, dtype=float).ravel()
    X = np.asarray(values, dtype=float)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[:, None]
    if X.shape[0] != t.shape[0]:
        raise ValueError(f'times ({t.shape[0]}) and values ({X.shape[0]}) differ in length')
    ok = np.isfinite(t)
    t, X = t[ok], X[ok]
    order = np.argsort(t, kind='stable')
    t, X = t[order], X[order]
    if t.size > 1:
        keep = np.r_[True, np.diff(t) > 0]  # duplicate timestamps: keep the first reading
        t, X = t[keep], X[keep]
    return t, X, squeeze


def uniform_grid(t_start: float, t_end: float, hz: float) -> np.ndarray:
    """Grid t_start, t_start + 1000/hz, ... <= t_end (ms)."""
    step = 1000.0 / hz
    if not (math.isfinite(t_start) and math.isfinite(t_end)) or t_end < t_start:
        return np.zeros(0)
    n = int(math.floor((t_end - t_start) / step + 1e-9)) + 1
    return t_start + np.arange(n) * step


def resample_uniform(
    t_ms: Any,
    values: Any,
    hz: float = 100.0,
    max_gap_ms: float = 100.0,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling onto a uniform grid.

    Each column is interpolated from its own finite samples. A grid point is NaN when it lies outside
    the column's valid range or inside a gap (distance between the two bracketing valid samples) longer
    than ``max_gap_ms``. Returns ``(t_uniform_ms, values_uniform)`` with values shaped like the input.
    """
    t, X, squeeze = _prepare_times(t_ms, values)
    if t.size == 0:
        tn = np.zeros(0)
        out = np.zeros((0, X.shape[1]))
        return tn, (out[:, 0] if squeeze else out)
    t0 = float(t[0]) if t_start is None else float(t_start)
    t1 = float(t[-1]) if t_end is None else float(t_end)
    tn = uniform_grid(t0, t1, hz)
    out = np.full((tn.size, X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j])
        tv, xv = t[ok], X[ok, j]
        if tv.size == 0:
            continue
        r = np.searchsorted(tv, tn, side='left')
        rc = np.clip(r, 0, tv.size - 1)
        exact = (r < tv.size) & (tv[rc] == tn)
        lc = np.clip(r - 1, 0, tv.size - 1)
        inside = (r < tv.size) & (r >= 1)
        gap = np.where(inside, tv[rc] - tv[lc], np.inf)
        valid = exact | (inside & (gap <= max_gap_ms))
        if tv.size == 1:
            out[valid, j] = xv[0]
            continue
        vals = np.interp(tn, tv, xv)
        out[valid, j] = vals[valid]
    return tn, (out[:, 0] if squeeze else out)


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs of a boolean vector."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def lowpass(x: Any, fs_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0, run by run over finite samples.

    Runs shorter than 3 samples are copied unfiltered; short runs use a reduced pad length. A cutoff at
    or above Nyquist returns a copy.
    """
    X = np.array(x, dtype=float, copy=True)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[:, None]
    nyq = fs_hz / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq or X.shape[0] < 3:
        return X[:, 0] if squeeze else X
    sos = signal.butter(order, cutoff_hz, btype='low', fs=fs_hz, output='sos')
    default_pad = 3 * (2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    for j in range(X.shape[1]):
        col = X[:, j]
        for a, b in finite_runs(np.isfinite(col)):
            n = b - a
            if n < 3:
                continue
            padlen = min(default_pad, n - 1)
            X[a:b, j] = signal.sosfiltfilt(sos, col[a:b], padlen=padlen)
    return X[:, 0] if squeeze else X


def fill_gaps_linear(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN rows (edges hold the nearest valid value). All-NaN columns stay NaN."""
    X = np.array(x, dtype=float, copy=True)
    squeeze = X.ndim == 1
    if squeeze:
        X = X[:, None]
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j])
        if ok.all() or not ok.any():
            continue
        X[~ok, j] = np.interp(t[~ok], t[ok], X[ok, j])
    return X[:, 0] if squeeze else X


# ───────────────────────────── gravity ─────────────────────────────


def estimate_gravity(
    acc: Any,
    accg: Any,
    fs_hz: float,
    cutoff_hz: float = 1.0,
    method: str = 'auto',
) -> tuple[np.ndarray, str]:
    """Gravity reaction vector ("up", W3C sign) in the device frame, per sample.

    method: 'difference' = accg - acc (exact up to the platform's own sensor fusion; rows where acc is
    missing fall back to the low-pass), 'lowpass' = zero-phase low-pass of accg at ``cutoff_hz``, 'auto'
    = 'difference' when acc is present in at least half of the rows, else 'lowpass'.
    Returns ``(g, method_used)``. Inputs must already be sign-normalised.
    """
    A = np.asarray(acc, dtype=float)
    G = np.asarray(accg, dtype=float)
    acc_ok = np.all(np.isfinite(A), axis=1) if A.size else np.zeros(G.shape[0], dtype=bool)
    if method == 'auto':
        method = 'difference' if acc_ok.size and acc_ok.mean() >= 0.5 else 'lowpass'
    if method == 'lowpass':
        return lowpass(G, fs_hz, cutoff_hz), 'lowpass'
    if method != 'difference':
        raise ValueError(f'unknown gravity method {method!r}')
    g = G - A
    if not acc_ok.all():
        lp = lowpass(G, fs_hz, cutoff_hz)
        g[~acc_ok] = lp[~acc_ok]
    return g, 'difference'


def unit_rows(v: np.ndarray) -> np.ndarray:
    """Row-normalised copy (zero rows -> NaN)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, v / n, np.nan)


# ───────────────────────────── W3C Euler <-> matrix <-> quaternion ─────────────────────────────


def euler_to_matrix(alpha: Any, beta: Any, gamma: Any) -> np.ndarray:
    """W3C intrinsic Z-X'-Y'' angles in DEGREES -> rotation matrices (..., 3, 3), device -> earth."""
    a = np.asarray(alpha, dtype=float) * DEG2RAD
    b = np.asarray(beta, dtype=float) * DEG2RAD
    c = np.asarray(gamma, dtype=float) * DEG2RAD
    a, b, c = np.broadcast_arrays(a, b, c)
    ca, sa, cb, sb, cg, sg = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    R = np.empty(a.shape + (3, 3))
    R[..., 0, 0] = ca * cg - sa * sb * sg
    R[..., 0, 1] = -sa * cb
    R[..., 0, 2] = ca * sg + sa * sb * cg
    R[..., 1, 0] = sa * cg + ca * sb * sg
    R[..., 1, 1] = ca * cb
    R[..., 1, 2] = sa * sg - ca * sb * cg
    R[..., 2, 0] = -cb * sg
    R[..., 2, 1] = sb
    R[..., 2, 2] = cb * cg
    return R


def matrix_to_euler(R: Any, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation matrices (..., 3, 3) -> W3C angles in degrees with alpha in [0, 360), beta in [-180, 180),
    gamma in [-90, 90). Gimbal lock (cos beta = 0) returns gamma = 0."""
    R = np.asarray(R, dtype=float)
    m00, m01, m10, m11 = R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    sb = np.clip(m21, -1.0, 1.0)
    cb_abs = np.hypot(m20, m22)
    # cos(gamma) >= 0 in the W3C range, so sign(cos beta) = sign(m22); m22 == 0 means gamma = -90 and
    # then cos(beta) = m20.
    cb = np.where(np.abs(m22) > eps, np.sign(m22) * cb_abs, m20)
    lock = cb_abs <= eps
    cb_safe = np.where(lock, 1.0, cb)
    with np.errstate(invalid='ignore', divide='ignore'):
        beta = np.arctan2(sb, cb)
        gamma = np.arctan2(-m20 / cb_safe, m22 / cb_safe)
        alpha = np.arctan2(-m01 / cb_safe, m11 / cb_safe)
    # gimbal lock: m00 = cos(alpha ± gamma), m10 = sin(alpha ± gamma); choose gamma = 0
    alpha = np.where(lock, np.arctan2(m10, m00), alpha)
    beta = np.where(lock, np.where(sb > 0, np.pi / 2, -np.pi / 2), beta)
    gamma = np.where(lock, 0.0, gamma)
    a = alpha * RAD2DEG
    b = beta * RAD2DEG
    g = gamma * RAD2DEG
    # gamma can round to +90: same rotation as (alpha + 180, 180 - beta, gamma - 180)
    flip = g >= 90.0
    a = np.where(flip, a + 180.0, a)
    b = np.where(flip, 180.0 - b, b)
    g = np.where(flip, g - 180.0, g)
    a = np.mod(a, 360.0)
    a = np.where(a >= 360.0, a - 360.0, a)
    b = np.mod(b + 180.0, 360.0) - 180.0
    b = np.where(b >= 180.0, b - 360.0, b)
    return a, b, g


def matrix_to_quaternion(R: Any) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) as [x, y, z, w] with w >= 0."""
    R = np.asarray(R, dtype=float)
    shape = R.shape[:-2]
    M = R.reshape(-1, 3, 3)
    q = np.empty((M.shape[0], 4))
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    diag = np.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2]], axis=1)
    for i in range(M.shape[0]):
        m = M[i]
        if tr[i] > 0:
            s = math.sqrt(tr[i] + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            k = int(np.argmax(diag[i]))
            if k == 0:
                s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 0.0)) * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif k == 1:
                s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 0.0)) * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 0.0)) * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        q[i] = (x, y, z, w)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0
    return q.reshape(shape + (4,))


def quaternion_to_matrix(q: Any) -> np.ndarray:
    """Quaternions (..., 4) [x, y, z, w] (normalised internally) -> rotation matrices (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def euler_to_quaternion(alpha: Any, beta: Any, gamma: Any) -> np.ndarray:
    return matrix_to_quaternion(euler_to_matrix(alpha, beta, gamma))


def rotvec_to_matrix(v: Any) -> np.ndarray:
    """Rotation vector(s) (..., 3) in radians -> rotation matrices (Rodrigues)."""
    v = np.asarray(v, dtype=float)
    theta = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        k = np.where(theta[..., None] > 1e-15, v / theta[..., None], 0.0)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s = np.sin(theta)[..., None, None]
    c = np.cos(theta)[..., None, None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + s * K + (1 - c) * (K @ K)


def matrix_to_rotvec(R: Any) -> np.ndarray:
    """Rotation matrices -> rotation vectors (radians), via the quaternion (robust near pi)."""
    q = matrix_to_quaternion(R)
    v = q[..., :3]
    w = q[..., 3]
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, w)
    with np.errstate(invalid='ignore', divide='ignore'):
        axis = np.where(s[..., None] > 1e-15, v / s[..., None], 0.0)
    return axis * angle[..., None]


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed random rotation matrix."""
    q = rng.normal(size=4)
    return quaternion_to_matrix(q / np.linalg.norm(q))


def axis_rotation(axis: str, angle_deg: float) -> np.ndarray:
    """Elementary rotation about 'x', 'y' or 'z' (right-hand rule), angle in degrees."""
    a = angle_deg * DEG2RAD
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(axis)


# ───────────────────────────── gyro integration ─────────────────────────────


def integrate_gyro(t_ms: Any, gyro_dps: Any) -> tuple[np.ndarray, int]:
    """Device orientation relative to the first sample from the device-frame angular velocity.

    Returns ``(R, n_missing)`` where R[k] (3x3) maps device-frame vectors at sample k into the device
    frame of sample 0 (R[0] = I), integrated with the midpoint rate of each step
    (R_{k+1} = R_k · Exp(dt·(w_k + w_{k+1})/2)). Steps with a missing rate rotate by zero and are counted.
    Covariant: a fixed grip rotation Q of the rates gives Q·R·Qᵀ; the mirror -S·w gives S·R·S.
    """
    t = np.asarray(t_ms, dtype=float) / 1000.0
    W = np.asarray(gyro_dps, dtype=float) * DEG2RAD
    n = t.shape[0]
    qs = np.empty((n, 4))
    x = y = z = 0.0
    w = 1.0
    qs[0] = (x, y, z, w)
    missing = 0
    rows = W.tolist()
    for k in range(n - 1):
        dt = float(t[k + 1] - t[k])
        a, b = rows[k], rows[k + 1]
        if not (all(math.isfinite(v) for v in a) and all(math.isfinite(v) for v in b)) or not math.isfinite(dt):
            missing += 1
            qs[k + 1] = (x, y, z, w)
            continue
        ox, oy, oz = (a[0] + b[0]) * 0.5 * dt, (a[1] + b[1]) * 0.5 * dt, (a[2] + b[2]) * 0.5 * dt
        th = math.sqrt(ox * ox + oy * oy + oz * oz)
        if th < 1e-15:
            qs[k + 1] = (x, y, z, w)
            continue
        h = 0.5 * th
        s = math.sin(h) / th
        dx, dy, dz, dw = ox * s, oy * s, oz * s, math.cos(h)
        # q <- q ⊗ dq  (body-frame increment)
        x, y, z, w = (
            w * dx + x * dw + y * dz - z * dy,
            w * dy - x * dz + y * dw + z * dx,
            w * dz + x * dy - y * dx + z * dw,
            w * dw - x * dx - y * dy - z * dz,
        )
        nq = math.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / nq, y / nq, z / nq, w / nq
        qs[k + 1] = (x, y, z, w)
    return quaternion_to_matrix(qs), missing


# ───────────────────────────── gravity_v1: gyro propagation from the rest windows ─────────────────────────────


def angle_deg(u: Any, v: Any) -> np.ndarray:
    """Angle(s) between vectors (..., 3) in degrees (atan2 form, accurate near 0 and 180); NaN if missing."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    c = np.linalg.norm(np.cross(u, v), axis=-1)
    d = np.sum(u * v, axis=-1)
    return np.degrees(np.arctan2(c, d))


def rest_mean(
    t: np.ndarray, f: np.ndarray, gyro_mag: np.ndarray, t_from: float, t_to: float, still_dps: float,
) -> tuple[np.ndarray | None, int]:
    """Mean specific force over the samples in [t_from, t_to] with |w| < still_dps (None if none)."""
    sel = (t >= t_from) & (t <= t_to) & (gyro_mag < still_dps) & np.isfinite(f).all(axis=1)
    n = int(sel.sum())
    return (f[sel].mean(axis=0) if n else None), n


@dataclass
class GyroGravity:
    """Result of :func:`gyro_gravity`.

    g: (n,3) gravity reaction ("up", W3C sign) in the device frame at every grid sample, rescaled to
    ``magnitude``; g_fwd / g_bwd: the two propagated estimates (unscaled; g_bwd None without a post
    rest); closure_deg: angle between g_fwd at the end and the post-movement rest (QA: > 3 deg points to
    a gyro bias / scale problem or a rest window that was not still); method: 'fwd_bwd' or 'fwd_only'.
    """

    g: np.ndarray
    g_fwd: np.ndarray
    g_bwd: np.ndarray | None
    g_pre: np.ndarray
    g_post: np.ndarray | None
    pre_n: int
    post_n: int
    closure_deg: float
    magnitude: float
    method: str
    gyro_missing_steps: int


def gyro_gravity(
    t_ms: Any,
    f: Any,
    gyro_dps: Any,
    i_on: int,
    i_end: int,
    *,
    R: np.ndarray | None = None,
    still_dps: float = 15.0,
    rest_ms: float = 300.0,
    min_rest_n: int = 5,
    magnitude: float | None = None,
    g_pre_fallback: Any = None,
) -> GyroGravity:
    """Gravity in the device frame by gyro propagation from the rest windows (``gravity_v1``).

    1. Rest windows: 'pre' = samples of the ``rest_ms`` before the onset with |w| < ``still_dps`` (widened
       to 2x when too few; then ``g_pre_fallback``, e.g. the sign-normalised online baseline gravity;
       then f at the onset); 'post' = the same over the ``rest_ms`` after the end (the hold at the nose).
       Each window's specific force is averaged in ONE frame: every sample is first rotated into the
       device frame of sample 0 with the integrated gyro (R(k) f(k)), so tremor during the rest and the
       rotation between the rest and the onset sample do not bias the mean.
    2. Propagation: a world-fixed vector seen from the rotating device obeys dg/dt = -w x g, i.e.
       g(k) = R(k)^T g0 with R from :func:`integrate_gyro` (device(k) -> device(0)) and g0 the rest mean in
       that frame. Forward from the pre-movement rest, backward from the post-movement rest.
    3. Blend: g = (1 - lam) g_fwd + lam g_bwd with lam = (t - t_on) / (t_end - t_on) clipped to [0, 1]
       (before the onset only g_fwd, after the end only g_bwd), then rescaled to ``magnitude`` (calibration
       |g| when known, else |g_pre| when plausible, else 9.80665). closure_deg = angle between the two rest
       means in the common frame (= between g_fwd and g_bwd at any instant).
    Every step is rotation-covariant and mirror-covariant (g -> S g for f -> S f, w -> -S w).
    """
    t = np.asarray(t_ms, dtype=float)
    F = np.asarray(f, dtype=float)
    W = np.asarray(gyro_dps, dtype=float)
    n = t.size
    if not (0 <= i_on <= i_end < n):
        raise ValueError(f'invalid segment {i_on}..{i_end} for {n} samples')
    missing = 0
    if R is None:
        R, missing = integrate_gyro(t, W)
    Fw = np.einsum('nij,nj->ni', R, F)  # specific force in the device frame of sample 0
    wmag = np.linalg.norm(W, axis=1)
    wmag = np.where(np.isfinite(wmag), wmag, np.inf)
    t_on, t_end = float(t[i_on]), float(t[i_end])
    gw_pre, pre_n = rest_mean(t, Fw, wmag, t_on - rest_ms, t_on - 1e-9, still_dps)
    if pre_n < min_rest_n:
        gw_pre, pre_n = rest_mean(t, Fw, wmag, t_on - 2 * rest_ms, t_on - 1e-9, still_dps)
    if pre_n < min_rest_n:
        fb = None if g_pre_fallback is None else np.asarray(g_pre_fallback, dtype=float)
        if fb is not None and fb.shape == (3,) and np.all(np.isfinite(fb)):
            gw_pre, pre_n = R[i_on] @ fb, 0
        else:
            gw_pre, pre_n = Fw[i_on].copy(), 0
    gw_post, post_n = rest_mean(t, Fw, wmag, t_end, t_end + rest_ms, still_dps)
    if post_n < min_rest_n:
        gw_post, post_n = rest_mean(t, Fw, wmag, t_end, t_end + 2 * rest_ms, still_dps)
    if post_n < min_rest_n:
        gw_post, post_n = None, 0
    g_fwd = np.einsum('nji,j->ni', R, gw_pre)  # R[k]^T g0
    g_pre = R[i_on].T @ gw_pre
    g_bwd = g_post = None
    if gw_post is not None:
        g_bwd = np.einsum('nji,j->ni', R, gw_post)
        g_post = R[i_end].T @ gw_post
        span = t_end - t_on
        lam = np.clip((t - t_on) / span, 0.0, 1.0) if span > 0 else (t > t_on).astype(float)
        g = (1.0 - lam)[:, None] * g_fwd + lam[:, None] * g_bwd
        closure = float(angle_deg(gw_pre, gw_post))
        method = 'fwd_bwd'
    else:
        g = g_fwd.copy()
        closure = float('nan')
        method = 'fwd_only'
    if magnitude is None or not (8.0 <= float(magnitude) <= 11.5):
        m_pre = float(np.linalg.norm(gw_pre))
        magnitude = m_pre if (pre_n > 0 and 8.6 <= m_pre <= 11.0) else GRAVITY
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(norms > 0, g / norms, np.nan) * float(magnitude)
    return GyroGravity(g, g_fwd, g_bwd, g_pre, g_post, pre_n, post_n, closure, float(magnitude), method, missing)


# ───────────────────────────── platform harmonisation ─────────────────────────────

#: Chromium rounding multiples (services/device/generic_sensor/platform_sensor_util.h).
CHROME_ACC_MULTIPLE = 0.1  # m/s^2
CHROME_GYRO_MULTIPLE_RAD = 0.00174532925199432963  # 0.1 deg in rad (the gyro is rounded in rad/s)
CHROME_ANGLE_MULTIPLE = 0.1  # deg


def chrome_round(x: Any, multiple: float) -> np.ndarray:
    """Chromium's RoundToMultiple: nearest multiple, halves away from zero (NaN stays NaN)."""
    v = np.asarray(x, dtype=float)
    s = v / multiple
    with np.errstate(invalid='ignore'):
        return np.where(v < 0.0, -multiple * np.floor(-s + 0.5), multiple * np.floor(s + 0.5))


def quantize_like_chrome(x: Any, kind: str) -> np.ndarray:
    """Chrome-style value grid: kind 'acc' (0.1 m/s^2), 'rr' (0.1 deg/s, rounded in rad/s like Chromium)
    or 'angle' (0.1 deg). Symmetric in sign, so raw iOS values (inverted sign) can be quantised as stored."""
    if kind == 'acc':
        return chrome_round(x, CHROME_ACC_MULTIPLE)
    if kind == 'rr':
        return chrome_round(np.asarray(x, dtype=float) * DEG2RAD, CHROME_GYRO_MULTIPLE_RAD) * RAD2DEG
    if kind == 'angle':
        return chrome_round(x, CHROME_ANGLE_MULTIPLE)
    raise ValueError(f'unknown kind {kind!r}')


def repeated_sample_mask(acc: Any, accg: Any, rr: Any, *, engine: str | None = None,
                         chrome_min_dps: float = 20.0) -> np.ndarray:
    """True for samples that are an isolated exact repeat of the previous sample (a duplicated snapshot).

    WebKit polls Core Motion with a timer that is not phase-locked to the sensor and never rounds, so on
    'webkit' an exact 9-channel repeat is a duplicated tick. Chrome rounds to 0.1 units, so there a
    repeat is only suspicious while the phone turns faster than ``chrome_min_dps`` (docs/05 section 2.1).
    Only isolated repeats (two identical samples in a row) are marked: longer constant runs are a stalled
    or genuinely constant signal (e.g. noise-free test data) and are left to the resampling.
    """
    A = np.column_stack([np.asarray(acc, dtype=float), np.asarray(accg, dtype=float), np.asarray(rr, dtype=float)])
    n = A.shape[0]
    if n < 2:
        return np.zeros(n, dtype=bool)
    same = np.r_[False, np.all((A[1:] == A[:-1]) | (np.isnan(A[1:]) & np.isnan(A[:-1])), axis=1)]
    same &= np.r_[False, np.isfinite(A[1:]).any(axis=1)]
    isolated = same & ~np.r_[same[1:], False] & ~np.r_[False, same[:-1]]
    if engine == 'webkit':
        return isolated
    w = np.linalg.norm(np.asarray(rr, dtype=float), axis=1)
    return isolated & (w > chrome_min_dps)


# ───────────────────────────── sampling statistics ─────────────────────────────


def rate_stats(times_ms: Any) -> dict[str, float | int | None]:
    """Sampling statistics of increasing times (ms): n, hz = (n-1)/(t_last-t_first)*1000, dt median /
    p95 / max and the number of gaps over 50 ms (same definitions as recorder.rateStats)."""
    t = np.asarray(times_ms, dtype=float)
    t = t[np.isfinite(t)]
    n = int(t.size)
    if n < 2:
        return {'n': n, 'hz': None, 'dt_median': None, 'dt_p95': None, 'dt_max': None, 'gaps_over_50ms': 0}
    dt = np.diff(t)
    span = float(t[-1] - t[0])
    return {
        'n': n,
        'hz': (n - 1) / span * 1000.0 if span > 0 else None,
        'dt_median': float(np.median(dt)),
        'dt_p95': float(np.percentile(dt, 95)),
        'dt_max': float(np.max(dt)),
        'gaps_over_50ms': int(np.sum(dt > 50.0)),
    }
