"""Record-level augmentations (each returns a NEW, schema-valid TrialRecord; inputs are never modified).

mirror_trial
    Reflection of the body across its sagittal plane (left <-> right hand). With R the device->body
    rotation, the body reflection M = diag(-1,1,1) and the phone identified with its mirror image through
    the device-frame reflection S = diag(-1,1,1), the mirrored pose is R' = M·R·S and, FOR ANY GRIP:
        acceleration / accelerationIncludingGravity (polar vectors):  a' = S·a   -> x negated
        angular velocity (pseudo-vector):                             w' = -S·w  -> x kept, y and z negated
    i.e. (W3C mapping) rotationRate alpha (x) kept, beta (y) and gamma (z) negated - the same rule as
    app/src/core/mirror.ts. ``rr_verdict='legacy'`` handles sessions whose calibration found the legacy
    mapping (alpha = z): then beta (x) is kept and gamma, alpha are negated. World-referenced orientation:
    R'_ED = Mw·R_ED·S with Mw = diag(-1,1,1) in earth axes (a reflection across a vertical plane; the
    heading is arbitrary), giving Euler (alpha, beta, gamma) -> (-alpha, beta, -gamma) up to range
    normalisation; quaternions (x, y, z, w) -> (x, -y, -z, w); compass heading h -> 360 - h. The hand is
    flipped, the instructed orientation mapped through the mirror table (top_left_* <-> top_right_*),
    detected rotations mapped too (top_left <-> top_right), the baseline gravity vector mirrored, the
    screen angle a -> 360 - a with landscape-primary <-> landscape-secondary. Ids, timing, outcome and
    quality are kept. It is an exact involution (up to Euler range normalisation).
screen_away_trial
    Screen toward <-> away with the same hand: the phone turned 180 deg about its own y axis,
    F = diag(-1, 1, -1) (GRIP_MATRICES[o_away] = GRIP_MATRICES[o_toward]·F). F is a PROPER rotation, so every
    device vector transforms alike: acc, accg, rotation rate, magnetometer and baseline gravity
    (x, y, z) -> (-x, y, -z) (W3C: rr_alpha and rr_gamma negated; legacy: rr_beta and rr_alpha); world-
    referenced orientation R'_ED = R_ED·F (quaternion (x, y, z, w) -> (-z, w, x, -y) towards away, the
    inverse (z, -w, -x, y) back, so flipping twice is exact); label *_toward <-> *_away; detected rotations
    unchanged (rotation_from_gravity(F·g, s, 'away') = rotation_from_gravity(g, s, 'toward')); screen angle
    a -> 360 - a with landscape-primary <-> landscape-secondary (the gravity's x flips); touches cleared and
    timing.first_touch_ms null (the nose meets the back of the phone); compass headings null (iOS defines
    them by the screen's facing); trial id + '-away'. The basic, invariant and chiral features are exactly
    unchanged, only the device-frame ones change. It is NOT a real screen-away movement (the wrist, the
    fingers, the approach and the contact differ; the accelerometer sits off-centre, so its lever-arm
    terms are not flipped): use it to augment TRAINING data, never to report screen-away accuracy.
rotate_grip / random_grip_tilt
    The same movement with the phone held in a different fixed orientation in the hand: device-frame
    vectors v' = Q·v for a fixed rotation Q (acc, accg, rotation rate, baseline gravity, magnetometer),
    orientation R'_ED = R_ED·Qᵀ (also the absolute streams). Chiral and invariant features must not
    change. ``random_grip_tilt`` draws a small Q (at most 15-20 deg, docs/05 section 5) to mimic grip
    variation.
time_warp
    The same path executed ``factor`` times slower: t' = factor·t (all times), angular velocity / factor,
    gravity-free acceleration / factor^2 (the gravity part of accg is unchanged).
scale_magnitudes
    Linear acceleration and angular velocity scaled by one factor each (never per axis, never the
    gravity part of accg).
quantize_trial
    Chrome's value grid (0.1 m/s^2, 0.1 deg/s rounded in rad/s, 0.1 deg; Chromium platform_sensor_util),
    then the app's storage rounding: makes iOS-like (unrounded) data look Android-like, for the
    iOS-vs-Android ablation (docs/05 section 8.2, A3).
add_noise
    White Gaussian sensor noise (the same draw is added to acc and accg so that accg - acc stays gravity).

Every augmentation keeps the optional schema 1.2 streams (quaternion_abs, orientation_abs, magnetometer,
screen_events) consistent with the motion it produces, or drops what no longer describes it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .conventions import (
    DEG2RAD, ORIENTATIONS, PHONE_ROTATION_MIRROR, ROUNDING, SCREEN_FLIP, mirror_orientation, rr_fields_for_device_axes,
    screen_flip_orientation,
)
from .io import _arr, _json_list
from .preprocess import (
    euler_to_matrix, matrix_to_euler, matrix_to_quaternion, quantize_like_chrome, quaternion_to_matrix,
    random_rotation, rotvec_to_matrix,
)
from .schema import deep_copy

_LANDSCAPE = {'landscape-primary': 'landscape-secondary', 'landscape-secondary': 'landscape-primary'}


def _neg(col: Any) -> list[float | None]:
    a = _arr(col)
    return _json_list(-a + 0.0)  # + 0.0: no -0.0 in the output


def _mirror_heading(h: Any) -> Any:
    if h is None:
        return None
    r = (360.0 - float(h)) % 360.0
    return r + 0.0


def _mirror_euler_cols(alpha: Any, beta: Any, gamma: Any) -> tuple[list, list, list]:
    """R' = Mw·R·S (Mw = S = diag(-1,1,1)); rows with any missing angle become all-null."""
    A, B, G = _arr(alpha), _arr(beta), _arr(gamma)
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(G)
    a2 = np.full(A.shape, np.nan)
    b2 = np.full(A.shape, np.nan)
    g2 = np.full(A.shape, np.nan)
    if ok.any():
        R = euler_to_matrix(A[ok], B[ok], G[ok])
        M = np.diag([-1.0, 1.0, 1.0])
        a, b, g = matrix_to_euler(M @ R @ M)
        a2[ok], b2[ok], g2[ok] = a, b, g
    return _json_list(a2), _json_list(b2), _json_list(g2)


def _motion_copy(src: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (_json_list(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in src.items()}


def _stream_copy(s: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not s:
        return None
    return {k: (_json_list(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in s.items()}


#: Optional streams of schema 1.2 (absent in older records; an augmentation keeps them absent).
_EXTRA_STREAMS = ('quaternion_abs', 'orientation_abs', 'magnetometer', 'screen_events')


def _with_extras(st: dict[str, Any], src: Mapping[str, Any], **made: Any) -> dict[str, Any]:
    """``st`` plus the schema 1.2 streams of ``src``: those given in ``made`` (already transformed), the
    others copied unchanged. Streams absent from ``src`` stay absent."""
    for name in _EXTRA_STREAMS:
        if name not in src:
            continue
        if name in made:
            st[name] = made[name]
        elif name == 'screen_events':
            st[name] = deep_copy(src.get(name) or [])
        else:
            st[name] = _stream_copy(src.get(name))
    return st


def _euler_cols_times(o: Mapping[str, Any], Q: np.ndarray) -> tuple[list, list, list]:
    """Euler columns of R_ED·Qᵀ (a fixed rotation of the device frame); rows with a missing angle stay null."""
    A, B, G = _arr(o['alpha']), _arr(o['beta']), _arr(o['gamma'])
    ok = np.isfinite(A) & np.isfinite(B) & np.isfinite(G)
    a2, b2, g2 = (np.full(A.shape, np.nan) for _ in range(3))
    if ok.any():
        a, b, g = matrix_to_euler(euler_to_matrix(A[ok], B[ok], G[ok]) @ Q.T)
        a2[ok], b2[ok], g2[ok] = a, b, g
    return _json_list(a2), _json_list(b2), _json_list(g2)


def _quat_cols_times(q: Mapping[str, Any], Q: np.ndarray) -> dict[str, list]:
    """Quaternion columns of R_ED·Qᵀ (via matrices: w >= 0 form)."""
    Qs = np.column_stack([_arr(q[c]) for c in 'xyzw'])
    out = {c: _json_list(_arr(q[c])) for c in 'xyzw'}
    if len(Qs):
        qn = matrix_to_quaternion(quaternion_to_matrix(Qs) @ Q.T)
        out = {c: _json_list(qn[:, j]) for j, c in enumerate('xyzw')}
    return out


def _vec_cols_times(s: Mapping[str, Any], Q: np.ndarray, cols: tuple[str, str, str] = ('x', 'y', 'z')) -> list[list]:
    return _rotate_cols([s[c] for c in cols], Q)


def _mirror_screen_type(v: Any) -> Any:
    return _LANDSCAPE.get(v, v)


def _mirror_screen_events(events: Any) -> list[dict[str, Any]]:
    """Screen orientation events under a flip of the device x axis (mirror, screen flip): angle a ->
    360 - a, landscape-primary <-> landscape-secondary."""
    out = []
    for e in deep_copy(events or []):
        if isinstance(e, dict):
            e['type'] = _mirror_screen_type(e.get('type'))
            if e.get('angle') is not None:
                e['angle'] = _mirror_heading(e['angle'])
        out.append(e)
    return out


def mirror_condition(c: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(c)
    out['hand'] = 'left' if c['hand'] == 'right' else 'right'
    out['orientation'] = mirror_orientation(c['orientation'])
    return out


def mirror_trial(record: Mapping[str, Any], *, id_suffix: str | None = None, screen_w: float | None = None,
                 rr_verdict: str | None = 'w3c') -> dict[str, Any]:
    """Left <-> right mirror image of a trial (see the module docstring). Returns a deep copy.

    id_suffix: appended to trial_id (e.g. '-m') when mirrored records are mixed with originals.
    screen_w: CSS screen width; when given, touch x becomes screen_w - x (otherwise touches are kept).
    rr_verdict: the session's rotationRate mapping (``calibration.axis_check.verdict``); the sign part of
    '*_inverted' does not matter here.
    """
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    st: dict[str, Any] = {'touches': deep_copy(src.get('touches') or []), 'cues': deep_copy(src.get('cues') or [])}
    m = _motion_copy(src['motion'])
    m['acc_x'] = _neg(m['acc_x'])
    m['accg_x'] = _neg(m['accg_x'])
    _, fy, fz = rr_fields_for_device_axes(rr_verdict)  # device y and z rates flip, x is kept
    m[fy] = _neg(m[fy])
    m[fz] = _neg(m[fz])
    st['motion'] = m

    def mirror_orientation_stream(o: Any) -> dict[str, Any] | None:
        if not o:
            return None
        o2 = _stream_copy(o)
        o2['alpha'], o2['beta'], o2['gamma'] = _mirror_euler_cols(o['alpha'], o['beta'], o['gamma'])
        o2['compass'] = [_mirror_heading(v) for v in _json_list(o['compass'])]
        return o2

    def mirror_quaternion_stream(q: Any) -> dict[str, Any] | None:
        if not q:
            return None
        q2 = _stream_copy(q)
        q2['y'] = _neg(q['y'])
        q2['z'] = _neg(q['z'])
        return q2

    st['orientation'] = mirror_orientation_stream(src.get('orientation'))
    st['quaternion'] = mirror_quaternion_stream(src.get('quaternion'))
    mag = _stream_copy(src.get('magnetometer'))
    if mag:
        mag['x'] = _neg(src['magnetometer']['x'])  # the field in the device frame mirrors like acc: S·b
    _with_extras(st, src, quaternion_abs=mirror_quaternion_stream(src.get('quaternion_abs')),
                 orientation_abs=mirror_orientation_stream(src.get('orientation_abs')), magnetometer=mag,
                 screen_events=_mirror_screen_events(src.get('screen_events')))
    if screen_w is not None:
        for tch in st['touches']:
            if tch.get('x') is not None:
                tch['x'] = float(screen_w) - float(tch['x'])
            if tch.get('x_norm') is not None:  # schema 1.2 touch details
                tch['x_norm'] = 1.0 - float(tch['x_norm'])
            if tch.get('rotation_angle') is not None:
                tch['rotation_angle'] = None  # the ellipse angle of a mirrored contact is not defined by it
    out['streams'] = st
    out['condition'] = mirror_condition(record['condition'])
    det = out.get('detected') or {}
    for k in ('rotation_start', 'rotation_end'):
        if k in det:
            det[k] = PHONE_ROTATION_MIRROR.get(det[k], det[k])
    base = (out.get('segmentation') or {}).get('baseline')
    if base and base.get('gravity') is not None:
        g = base['gravity']
        base['gravity'] = [-g[0] + 0.0, g[1], g[2]]
    scr = out.get('screen') or {}
    for k in ('angle_start', 'angle_end'):
        if scr.get(k) is not None:
            scr[k] = _mirror_heading(scr[k])
    for k in ('type_start', 'type_end'):
        if scr.get(k) in _LANDSCAPE:
            scr[k] = _LANDSCAPE[scr[k]]
    if id_suffix:
        out['trial_id'] = f"{out['trial_id']}{id_suffix}"[:64]
    return out


def _flip_quaternion_cols(q: Mapping[str, Any], to_away: bool) -> dict[str, list]:
    """Quaternion columns of R_ED·F (F = 180 deg about device y, q_F = (0, 1, 0, 0) as (x, y, z, w)):
    q ⊗ q_F = (-z, w, x, -y) towards away, q ⊗ q_F⁻¹ = (z, -w, -x, y) back towards the screen, so a
    double flip returns the stored numbers exactly (q and -q are the same rotation either way)."""
    x, y, z, w = (_arr(q[c]) for c in 'xyzw')
    cols = (-z, w, x, -y) if to_away else (z, -w, -x, y)
    return {c: _json_list(v + 0.0) for c, v in zip('xyzw', cols)}  # + 0.0: no -0.0 in the output


def screen_away_trial(record: Mapping[str, Any], *, rr_verdict: str | None = 'w3c',
                      id_suffix: str | None = '-away') -> dict[str, Any]:
    """The trial with the phone turned 180 deg about its own y axis: screen toward <-> away, same hand.

    See the module docstring (F = diag(-1, 1, -1); acc, accg, rotation rate, magnetometer and baseline
    gravity (x, y, z) -> (-x, y, -z); quaternions (x, y, z, w) -> (-z, w, x, -y); *_toward <-> *_away;
    touches cleared; detected rotations unchanged). Works in both directions; flipping twice gives back
    the trial (Euler angles up to range normalisation, trial id with both suffixes).
    rr_verdict: the session's rotationRate mapping (which raw fields hold device x and z; the sign part of
    '*_inverted' and the iOS accelerometer sign commute with F and do not matter).
    id_suffix: appended to trial_id (max 64 characters), None to keep it.
    Raises ValueError for 'flat_up' (no counterpart: the screen would face the floor). A synthetic grip
    for TRAINING augmentation: never report screen-away accuracy from these records.
    """
    o_from = (record.get('condition') or {}).get('orientation')
    o_to = screen_flip_orientation(o_from)
    to_away = not str(o_from).endswith('_away')
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    m = _motion_copy(src['motion'])
    for c in ('acc_x', 'acc_z', 'accg_x', 'accg_z'):
        m[c] = _neg(m[c])
    fx, _, fz = rr_fields_for_device_axes(rr_verdict)  # W3C: rr_alpha (x) and rr_gamma (z)
    m[fx] = _neg(m[fx])
    m[fz] = _neg(m[fz])

    def flipped_orientation(o: Any) -> dict[str, Any] | None:
        if not o:
            return None
        o2 = _stream_copy(o)
        o2['alpha'], o2['beta'], o2['gamma'] = _euler_cols_times(o, SCREEN_FLIP)
        n = len(o2['t'])
        o2['compass'] = [None] * n  # iOS webkitCompassHeading depends on which way the screen faces
        if o2.get('compass_accuracy') is not None:
            o2['compass_accuracy'] = [None] * n
        return o2

    def flipped_quaternion(q: Any) -> dict[str, Any] | None:
        if not q:
            return None
        return {**_stream_copy(q), **_flip_quaternion_cols(q, to_away)}

    st: dict[str, Any] = {
        'motion': m, 'orientation': flipped_orientation(src.get('orientation')),
        'quaternion': flipped_quaternion(src.get('quaternion')),
        'touches': [],  # the nose meets the back of the phone: no screen contact
        'cues': deep_copy(src.get('cues') or []),
    }
    mag = _stream_copy(src.get('magnetometer'))
    if mag:
        mag['x'] = _neg(src['magnetometer']['x'])
        mag['z'] = _neg(src['magnetometer']['z'])
    _with_extras(st, src, quaternion_abs=flipped_quaternion(src.get('quaternion_abs')),
                 orientation_abs=flipped_orientation(src.get('orientation_abs')), magnetometer=mag,
                 screen_events=_mirror_screen_events(src.get('screen_events')))
    out['streams'] = st
    out['condition']['orientation'] = o_to
    if isinstance(out.get('timing'), dict):
        out['timing']['first_touch_ms'] = None
    base = (out.get('segmentation') or {}).get('baseline')
    if base and base.get('gravity') is not None:
        g = base['gravity']
        base['gravity'] = [(-g[0] + 0.0) if g[0] is not None else None, g[1],
                           (-g[2] + 0.0) if g[2] is not None else None]
    scr = out.get('screen') or {}
    for k in ('angle_start', 'angle_end'):
        if scr.get(k) is not None:
            scr[k] = _mirror_heading(scr[k])
    for k in ('type_start', 'type_end'):
        if scr.get(k) in _LANDSCAPE:
            scr[k] = _LANDSCAPE[scr[k]]
    if id_suffix:
        out['trial_id'] = f"{out['trial_id']}{id_suffix}"[:64]
    return out


def _rotate_cols(cols: list[Any], Q: np.ndarray) -> list[list[float | None]]:
    V = np.column_stack([_arr(c) for c in cols])
    W = V @ Q.T
    return [_json_list(W[:, j]) for j in range(3)]


def rotate_grip(record: Mapping[str, Any], Q: np.ndarray | None = None, *, rng: np.random.Generator | None = None,
                orientation: str | None = None, rr_verdict: str | None = 'w3c') -> dict[str, Any]:
    """The same trial recorded with the phone held in another fixed orientation (device vectors v' = Q·v).

    Q: 3x3 rotation (random uniform when None, from ``rng``). The instructed orientation is kept unless
    ``orientation`` is given; detected rotations are set to 'unknown' (they no longer describe the data).
    rr_verdict: the session's rotationRate mapping (which raw fields hold device x, y, z).
    """
    if Q is None:
        Q = random_rotation(rng or np.random.default_rng())
    Q = np.asarray(Q, dtype=float)
    if not (np.allclose(Q @ Q.T, np.eye(3), atol=1e-9) and np.linalg.det(Q) > 0):
        raise ValueError('Q must be a proper rotation matrix')
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    m = _motion_copy(src['motion'])
    m['acc_x'], m['acc_y'], m['acc_z'] = _rotate_cols([m['acc_x'], m['acc_y'], m['acc_z']], Q)
    m['accg_x'], m['accg_y'], m['accg_z'] = _rotate_cols([m['accg_x'], m['accg_y'], m['accg_z']], Q)
    fx, fy, fz = rr_fields_for_device_axes(rr_verdict)  # W3C: rr_alpha, rr_beta, rr_gamma = x, y, z
    m[fx], m[fy], m[fz] = _rotate_cols([m[fx], m[fy], m[fz]], Q)
    st: dict[str, Any] = {'motion': m, 'touches': deep_copy(src.get('touches') or []), 'cues': deep_copy(src.get('cues') or [])}

    def rotated_orientation(o: Any) -> dict[str, Any] | None:
        if not o:
            return None
        o2 = _stream_copy(o)
        o2['alpha'], o2['beta'], o2['gamma'] = _euler_cols_times(o, Q)
        return o2

    def rotated_quaternion(q: Any) -> dict[str, Any] | None:
        if not q:
            return None
        return {**_stream_copy(q), **_quat_cols_times(q, Q)}

    st['orientation'] = rotated_orientation(src.get('orientation'))
    st['quaternion'] = rotated_quaternion(src.get('quaternion'))
    mag = _stream_copy(src.get('magnetometer'))
    if mag:
        mag['x'], mag['y'], mag['z'] = _vec_cols_times(src['magnetometer'], Q)
    _with_extras(st, src, quaternion_abs=rotated_quaternion(src.get('quaternion_abs')),
                 orientation_abs=rotated_orientation(src.get('orientation_abs')), magnetometer=mag)
    out['streams'] = st
    base = (out.get('segmentation') or {}).get('baseline')
    if base and base.get('gravity') is not None:
        base['gravity'] = (Q @ np.asarray(base['gravity'], dtype=float)).tolist()
    det = out.get('detected') or {}
    det['rotation_start'] = 'unknown'
    det['rotation_end'] = 'unknown'
    det['orientation_match'] = None
    if orientation is not None:
        if orientation not in ORIENTATIONS:
            raise ValueError(f'unknown orientation {orientation!r}')
        out['condition']['orientation'] = orientation
    return out


def small_rotation(rng: np.random.Generator, max_deg: float = 15.0) -> np.ndarray:
    """Random rotation about a uniformly random axis by an angle uniform in [0, max_deg]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return rotvec_to_matrix(axis * (rng.uniform(0.0, max_deg) * DEG2RAD))


def random_grip_tilt(record: Mapping[str, Any], max_deg: float = 15.0, *, rng: np.random.Generator | None = None,
                     rr_verdict: str | None = 'w3c') -> dict[str, Any]:
    """:func:`rotate_grip` with a small random rotation (grip variation; the orientation label is kept)."""
    rec = rotate_grip(record, small_rotation(rng or np.random.default_rng(), max_deg), rr_verdict=rr_verdict)
    # a small tilt keeps the detected rotations meaningful: restore them
    det_src = record.get('detected') or {}
    rec['detected'] = deep_copy(det_src)
    return rec


def time_warp(record: Mapping[str, Any], factor: float) -> dict[str, Any]:
    """Uniform time scaling (factor > 1 = slower). Indices are unchanged (same samples, new times).

    The gravity-free part of accg is rescaled using ``acc``; samples without ``acc`` keep their accg
    unchanged (the split between gravity and motion is unknown there).
    """
    if not factor > 0:
        raise ValueError('factor must be > 0')
    f = float(factor)
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    m = _motion_copy(src['motion'])
    t = _arr(m['t'])
    m['t'] = _json_list(t * f)
    m['ts'] = _json_list(_arr(m['ts']) * f)
    A = np.column_stack([_arr(m[f'acc_{a}']) for a in 'xyz'])
    AG = np.column_stack([_arr(m[f'accg_{a}']) for a in 'xyz'])
    gravity_part = AG - np.where(np.isfinite(A), A, 0.0)
    A2 = A / (f * f)
    AG2 = gravity_part + np.where(np.isfinite(A), A2, 0.0)
    for j, a in enumerate('xyz'):
        m[f'acc_{a}'] = _json_list(A2[:, j])
        m[f'accg_{a}'] = _json_list(AG2[:, j])
    for c in ('rr_alpha', 'rr_beta', 'rr_gamma'):
        m[c] = _json_list(_arr(m[c]) / f)
    st: dict[str, Any] = {'motion': m}

    def warped(s: Any) -> dict[str, Any] | None:
        if not s:
            return None
        s2 = _stream_copy(s)
        s2['t'] = _json_list(_arr(s['t']) * f)
        if 'ts' in s2:
            s2['ts'] = _json_list(_arr(s['ts']) * f)
        return s2

    for name in ('orientation', 'quaternion'):
        st[name] = warped(src.get(name))
    st['touches'] = [dict(x, t=float(x['t']) * f) for x in deep_copy(src.get('touches') or [])]
    st['cues'] = [dict(x, t=float(x['t']) * f) for x in deep_copy(src.get('cues') or [])]
    _with_extras(st, src, quaternion_abs=warped(src.get('quaternion_abs')), orientation_abs=warped(src.get('orientation_abs')),
                 magnetometer=warped(src.get('magnetometer')),
                 screen_events=[dict(x, t=float(x['t']) * f) for x in deep_copy(src.get('screen_events') or [])])
    out['streams'] = st
    tm = out.get('timing') or {}
    for k in ('still_ms', 'go_ms', 'onset_ms', 'end_ms', 'end_detected_ms', 'stop_ms', 'first_touch_ms', 'foreperiod_ms',
              'return_onset_ms', 'return_end_ms'):
        if tm.get(k) is not None:
            tm[k] = float(tm[k]) * f
    sg = out.get('segmentation') or {}
    if sg.get('duration_ms') is not None:
        sg['duration_ms'] = float(sg['duration_ms']) * f
    if sg.get('peak_gyro_dps') is not None:
        sg['peak_gyro_dps'] = float(sg['peak_gyro_dps']) / f
    if sg.get('peak_acc_ms2') is not None:
        sg['peak_acc_ms2'] = float(sg['peak_acc_ms2']) / (f * f)
    q = out.get('quality') or {}
    for k in ('motion_dt_ms_median', 'motion_dt_ms_max'):
        if q.get(k) is not None:
            q[k] = float(q[k]) * f
    if q.get('motion_hz') is not None:
        q['motion_hz'] = float(q['motion_hz']) / f
    return out


def scale_magnitudes(record: Mapping[str, Any], acc_factor: float = 1.0, gyro_factor: float = 1.0) -> dict[str, Any]:
    """Scale the linear acceleration (acc and the non-gravity part of accg) and the rotation rate.

    One factor per vector (docs/05 section 5 suggests 0.9-1.1); samples without ``acc`` keep their accg.
    """
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    m = _motion_copy(src['motion'])
    A = np.column_stack([_arr(m[f'acc_{a}']) for a in 'xyz'])
    AG = np.column_stack([_arr(m[f'accg_{a}']) for a in 'xyz'])
    has = np.isfinite(A)
    AG2 = AG - np.where(has, A, 0.0) + np.where(has, A * float(acc_factor), 0.0)
    for j, a in enumerate('xyz'):
        m[f'acc_{a}'] = _json_list(A[:, j] * float(acc_factor))
        m[f'accg_{a}'] = _json_list(AG2[:, j])
    for c in ('rr_alpha', 'rr_beta', 'rr_gamma'):
        m[c] = _json_list(_arr(m[c]) * float(gyro_factor))
    out['streams'] = _with_extras({
        'motion': m, 'orientation': _stream_copy(src.get('orientation')), 'quaternion': _stream_copy(src.get('quaternion')),
        'touches': deep_copy(src.get('touches') or []), 'cues': deep_copy(src.get('cues') or []),
    }, src)
    return out


def _round_app(a: np.ndarray, decimals: int) -> np.ndarray:
    """The app's storage rounding (recorder.roundValue: halves away from zero, no -0)."""
    f = 10.0 ** decimals
    with np.errstate(invalid='ignore'):
        r = np.sign(a) * np.floor(np.abs(a) * f + 0.5) / f
    return r + 0.0


def quantize_trial(record: Mapping[str, Any], *, motion: bool = True, angles: bool = True) -> dict[str, Any]:
    """Chrome-style quantisation of a trial (for iOS-like records; Android records are already on it).

    acc / accg -> multiples of 0.1 m/s^2, rotationRate -> 0.1 deg/s (rounded in rad/s like Chromium),
    orientation angles -> 0.1 deg; then the app's storage rounding (ROUNDING). Rounding is symmetric in
    sign, so the raw (possibly sign-inverted) values are quantised as stored. The stored segmentation
    is kept: re-run online_v1 (``segment.rerun_online_v1``) if the indices matter.
    """
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    m = _motion_copy(src['motion'])
    if motion:
        for c in ('acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z'):
            m[c] = _json_list(_round_app(quantize_like_chrome(_arr(m[c]), 'acc'), ROUNDING['acc']))
        for c in ('rr_alpha', 'rr_beta', 'rr_gamma'):
            m[c] = _json_list(_round_app(quantize_like_chrome(_arr(m[c]), 'rr'), ROUNDING['rr']))
    o = _stream_copy(src.get('orientation'))
    if o and angles:
        for c in ('alpha', 'beta', 'gamma'):
            q = _round_app(quantize_like_chrome(_arr(o[c]), 'angle'), ROUNDING['angle'])
            if c == 'alpha':
                q = np.where(np.isfinite(q), np.mod(q, 360.0), q)
            o[c] = _json_list(q)
    out['streams'] = _with_extras({
        'motion': m, 'orientation': o, 'quaternion': _stream_copy(src.get('quaternion')),
        'touches': deep_copy(src.get('touches') or []), 'cues': deep_copy(src.get('cues') or []),
    }, src)
    return out


def add_noise(record: Mapping[str, Any], *, acc_sd: float = 0.05, gyro_sd: float = 0.5,
              rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Additive white Gaussian noise on acc/accg (shared draw, m/s^2) and the rotation rate (deg/s)."""
    rng = rng or np.random.default_rng()
    out = deep_copy({k: v for k, v in record.items() if k != 'streams'})
    src = record['streams']
    m = _motion_copy(src['motion'])
    n = len(m['t'])
    na = rng.normal(0.0, acc_sd, size=(n, 3))
    nw = rng.normal(0.0, gyro_sd, size=(n, 3))
    for j, a in enumerate('xyz'):
        m[f'acc_{a}'] = _json_list(_arr(m[f'acc_{a}']) + na[:, j])
        m[f'accg_{a}'] = _json_list(_arr(m[f'accg_{a}']) + na[:, j])
    for j, c in enumerate(('rr_alpha', 'rr_beta', 'rr_gamma')):
        m[c] = _json_list(_arr(m[c]) + nw[:, j])
    out['streams'] = _with_extras({
        'motion': m, 'orientation': _stream_copy(src.get('orientation')), 'quaternion': _stream_copy(src.get('quaternion')),
        'touches': deep_copy(src.get('touches') or []), 'cues': deep_copy(src.get('cues') or []),
    }, src)
    return out


def mirror_rotation_label(r: str) -> str:
    """Detected PhoneRotation under mirroring."""
    return PHONE_ROTATION_MIRROR.get(r, r)

