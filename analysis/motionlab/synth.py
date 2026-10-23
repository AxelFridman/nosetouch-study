"""Independent, minimal generator of schema-valid NoseTouch records (tests, examples, demos).

This is NOT the app's simulator (app/src/sim + ``npm run synth``): it is a separate implementation in
Python, so the analysis kit is testable without Node and the two generators can cross-check each other.
Every record has ``simulated: true``.

Kinematics (body frame B: x = participant's right, y = up, z = backward; origin at the nose tip)
  progress pi(t): minimum-jerk sigma(s) = 10s^3 - 15s^4 + 6s^5 from 0 to 1 over T seconds ('pause' =
  two half movements separated by a 0.8 s hold, 'jerky' = three thirds with short holds).
  position p(pi) = p0 + (p1 - p0)·pi + arc·4pi(1 - pi)·n,  n = normalize((h·0.6, 0.8, 0))
  rotation R_BD = R_motion(pi) · R_grip(o) · R_tilt · R_tremor(t)
           R_motion(pi) = Ry(h·(yaw·pi + yaw_tr·4pi(1-pi))) · Rx(pitch·pi) · Rz(h·roll·pi)
  with h = +1 for the right hand and -1 for the left. All random draws are hand-independent and the
  hand enters only through h, the mirrored grip (M·R_grip(o)·S = R_grip(mirror(o))) and S-conjugated
  tilt/tremor axes, so a LEFT trial is the exact mirror image of the RIGHT trial with the same seed
  (when the participant has no dominance asymmetry and noise = 0).
Signals (exact, analytic): f_B = p'' + (0, g, 0); accg = sign·R_BDᵀ f_B; acc = sign·R_BDᵀ p''
  (sign = platform.gravity_sign: -1 on 'ios', like WebKit, for BOTH acceleration fields);
  w_D = R_BDᵀ w_B with w_B the space angular velocity of R_BD; rotationRate = (alpha, beta, gamma) =
  (w_x, w_y, w_z) in deg/s (W3C CRD 2025; platform.rr_mapping 'legacy' writes (w_z, w_x, w_y) instead, to
  test the normalisation); orientation = W3C Euler angles of R_EB(heading)·R_BD; quaternion idem.
  platform.quantize applies Chrome's 0.1 m/s^2 / 0.1 deg/s grid before the app's storage rounding.
Trial flow: still baseline -> online_v1 (this package's port) detects stillness -> GO after a random
foreperiod -> movement after a reaction time -> the port detects onset/end -> recording stops
post_roll_ms after the end cue, exactly like the app's trial controller. Values are rounded like the app
(constants.ts ROUNDING) BEFORE segmentation, so re-running online_v1 on a stored record reproduces its
``segmentation`` block exactly. Sessions carry schema-1.1 fields: ``device.engine`` and a calibration
``axis_check`` whose verdict matches platform.rr_mapping.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import (
    DEFAULT_SEGMENTATION_PARAMS, FIXED_ORIENTATIONS, GRAVITY, GRIP_MATRICES, ORIENTATIONS, ROUNDING,
    device_to_rotation_rate, mirror_orientation, resolve_params, rotation_from_gravity, tilt_change_deg,
)
from .preprocess import matrix_to_euler, matrix_to_quaternion, quantize_like_chrome, rate_stats, rotvec_to_matrix
from .segment import OnlineSegmenterV1, iter_motion_samples, run_online_v1

SCHEMA_VERSION = '1.1.0'
APP_VERSION = 'motionlab-synth-0.1.0'
PROTOCOL_VERSION = '1.0.0'
ID_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
SPEED_FACTOR = {'natural': 1.0, 'fast': 0.55, 'slow': 2.6}
_EHI_SCORE = {'always_right': 100, 'usually_right': 50, 'both': 0, 'usually_left': -50, 'always_left': -100}
_HORIZON_MS = 30000.0


# ───────────────────────────── small utilities ─────────────────────────────


def iso_time(epoch_ms: float) -> str:
    """ISO 8601 UTC with milliseconds, like JS Date.toISOString()."""
    import datetime as _dt
    d = _dt.datetime.fromtimestamp(epoch_ms / 1000.0, tz=_dt.timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.') + f'{int(d.microsecond / 1000):03d}Z'


def make_id(rng: np.random.Generator) -> str:
    """UUID-v4-shaped id from the generator (deterministic for a seed)."""
    b = bytearray(rng.integers(0, 256, size=16, dtype=np.uint8).tobytes())
    b[6] = (b[6] & 0x0F) | 0x40
    b[8] = (b[8] & 0x3F) | 0x80
    h = b.hex()
    return f'{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'


def make_participant_id(rng: np.random.Generator) -> str:
    return 'P-' + ''.join(ID_ALPHABET[int(i)] for i in rng.integers(0, len(ID_ALPHABET), size=6))


def score_ehi(sf: dict[str, str]) -> tuple[float, str]:
    """Edinburgh short form (Veale 2014): LQ = mean item score; >= 61 right, <= -61 left, else mixed."""
    lq = sum(_EHI_SCORE[sf[k]] for k in ('writing', 'throwing', 'toothbrush', 'spoon')) / 4.0
    return lq, ('right' if lq >= 61 else 'left' if lq <= -61 else 'mixed')


def _r(x: float | None, d: int) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    v = round(float(x), d)
    return 0.0 if v == 0 else v  # no -0.0 in JSON


def _round_list(a: np.ndarray, d: int) -> list[float | None]:
    return [_r(v, d) for v in np.asarray(a, dtype=float).tolist()]


def _sigma(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2 = s * s
    return (10 * s2 * s - 15 * s2 * s2 + 6 * s2 * s2 * s,
            30 * s2 - 60 * s2 * s + 30 * s2 * s2,
            60 * s - 180 * s2 + 120 * s2 * s)


# ───────────────────────────── participants / platforms ─────────────────────────────


@dataclass
class SynthParticipant:
    participant_id: str
    sex: str
    age_years: int
    height_cm: float
    handedness: str  # right | left | ambidextrous
    dominant: str  # right | left
    arm_length_m: float = 0.62
    shoulder_half_width_m: float = 0.19
    duration_natural_s: float = 0.95
    yaw_deg: float = 32.0
    yaw_transient_deg: float = 5.0
    roll_deg: float = 15.0
    pitch_deg: float = -8.0
    arc_m: float = 0.04
    start_blend: float = 0.5
    reaction_ms: float = 260.0
    tremor_dps: float = 2.5
    nondominant_slowdown: float = 1.12
    nondominant_jitter: float = 1.3


@dataclass
class SynthPlatform:
    name: str = 'android'  # android | ios
    gravity_sign: int = 1  # how the platform reports acc AND accg (+1 W3C, -1 inverted like WebKit)
    hz: float = 60.0
    jitter_ms: float = 1.0
    drop_prob: float = 0.0
    acc_available: bool = True
    quaternion: bool = True
    acc_noise_ms2: float = 0.03
    gyro_noise_dps: float = 0.4
    gyro_bias_dps: tuple[float, float, float] = (0.15, -0.1, 0.05)  # device x, y, z
    orientation_noise_deg: float = 0.05
    heading_deg: float = 0.0
    engine: str = 'blink'  # device.engine (schema 1.1): webkit | blink | gecko | unknown
    rr_mapping: str = 'w3c'  # 'w3c' (alpha, beta, gamma) = (x, y, z); 'legacy' = (z, x, y)
    quantize: bool = False  # Chrome's 0.1 m/s^2 / 0.1 deg/s value grid


def random_participant(rng: np.random.Generator, *, left_fraction: float = 0.3, **overrides: Any) -> SynthParticipant:
    u = rng.random()
    sex = 'female' if u < 0.47 else 'male' if u < 0.94 else ('other' if u < 0.97 else 'prefer_not_to_say')
    v = rng.random()
    handedness = 'ambidextrous' if v < 0.05 else ('left' if v < 0.05 + left_fraction * 0.95 else 'right')
    dominant = handedness if handedness != 'ambidextrous' else ('right' if rng.random() < 0.6 else 'left')
    height = float(rng.normal(178 if sex == 'male' else 165, 7))
    p = SynthParticipant(
        participant_id=make_participant_id(rng), sex=sex, age_years=int(rng.integers(18, 71)),
        height_cm=round(height, 1), handedness=handedness, dominant=dominant,
        arm_length_m=0.36 * height / 100.0, shoulder_half_width_m=float(rng.uniform(0.17, 0.21)),
        duration_natural_s=float(rng.uniform(0.75, 1.2)), yaw_deg=float(rng.uniform(20, 45)),
        yaw_transient_deg=float(rng.uniform(0, 10)), roll_deg=float(rng.uniform(5, 25)),
        pitch_deg=float(rng.uniform(-20, 5)), arc_m=float(rng.uniform(0.02, 0.06)),
        start_blend=float(rng.uniform(0.2, 0.8)), reaction_ms=float(rng.uniform(200, 330)),
        tremor_dps=float(rng.uniform(1.0, 4.0)), nondominant_slowdown=float(rng.uniform(1.0, 1.25)),
        nondominant_jitter=float(rng.uniform(1.0, 1.5)),
    )
    return replace(p, **overrides) if overrides else p


def default_platform(name: str, rng: np.random.Generator | None = None, **overrides: Any) -> SynthPlatform:
    """'ios' (WebKit): acc and accg sign-inverted (gravity_sign -1), unrounded, no quaternion stream.
    'android' (Blink): W3C sign, quaternion stream; ``quantize=True`` adds Chrome's 0.1 value grid."""
    heading = float(rng.uniform(0, 360)) if rng is not None else 0.0
    if name == 'ios':
        p = SynthPlatform(name='ios', gravity_sign=-1, hz=60.0, quaternion=False, heading_deg=heading, engine='webkit')
    elif name == 'android':
        p = SynthPlatform(name='android', gravity_sign=1, hz=60.0, quaternion=True, heading_deg=heading, engine='blink')
    else:
        raise ValueError(name)
    return replace(p, **overrides) if overrides else p


# ───────────────────────────── kinematics ─────────────────────────────


@dataclass
class Movement:
    """Everything needed to evaluate one trial's phone motion (hand already applied)."""

    h: float
    grip: np.ndarray  # R_grip(o) · R_tilt
    p0: np.ndarray
    p1: np.ndarray
    arc: float
    n_hat: np.ndarray
    yaw: float
    yaw_tr: float
    pitch: float
    roll: float
    segments: list[tuple[float, float, float, float]]  # (t_a_s, t_b_s, pi_a, pi_b) relative to onset
    tremor_axis: np.ndarray  # unit axis, S-conjugated for the left hand
    tremor_amp_rad: float
    tremor_hz: float
    tremor_phase: float
    # 'too_slow' behaviour: an oscillation about a device-side axis after onset (|w| peaks ~45 deg/s),
    # so the movement is detected but never ends (like the app simulator's wobble)
    wobble_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    wobble_amp_rad: float = 0.0
    wobble_hz: float = 1.0

    @property
    def duration_s(self) -> float:
        return self.segments[-1][1] if self.segments else 0.0


def _progress(t_s: np.ndarray, segments: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = np.zeros_like(t_s)
    pid = np.zeros_like(t_s)
    pidd = np.zeros_like(t_s)
    for ta, tb, pa, pb in segments:
        D = tb - ta
        after = t_s >= tb
        pi[after] = pb
        pid[after] = 0.0
        pidd[after] = 0.0
        inside = (t_s >= ta) & (t_s < tb)
        if inside.any():
            s = (t_s[inside] - ta) / D
            sg, sd, sdd = _sigma(s)
            d = pb - pa
            pi[inside] = pa + d * sg
            pid[inside] = d * sd / D
            pidd[inside] = d * sdd / (D * D)
    return pi, pid, pidd


def _rot_y(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, s, 1.0, -s, c
    return R


def _rot_x(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 1, 1], R[..., 1, 2], R[..., 2, 1], R[..., 2, 2] = 1.0, c, -s, s, c
    return R


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def evaluate_movement(mv: Movement, t_rel_s: np.ndarray, t_abs_s: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Exact kinematics at times relative to movement onset (s; negative or -inf = still holding at p0).

    ``t_abs_s`` (absolute time, s) drives the tremor; defaults to ``t_rel_s``. Returns body-frame
    acceleration 'a_B' (n,3), rotation 'R_BD' (n,3,3), space angular velocity 'w_B' (n,3, rad/s) and the
    progress 'pi'.
    """
    t = np.asarray(t_rel_s, dtype=float)
    ta = t if t_abs_s is None else np.asarray(t_abs_s, dtype=float)
    pi, pid, pidd = _progress(t, mv.segments)
    bump, dbump, ddbump = 4 * pi * (1 - pi), 4 - 8 * pi, -8.0 * np.ones_like(pi)
    dp = (mv.p1 - mv.p0)[None, :] + (mv.arc * dbump)[:, None] * mv.n_hat[None, :]
    ddp = (mv.arc * ddbump)[:, None] * mv.n_hat[None, :]
    a_B = ddp * (pid * pid)[:, None] + dp * pidd[:, None]
    d2r = math.pi / 180.0
    theta = mv.h * (mv.yaw * pi + mv.yaw_tr * bump) * d2r
    theta_d = mv.h * (mv.yaw + mv.yaw_tr * dbump) * pid * d2r
    phi = mv.pitch * pi * d2r
    phi_d = mv.pitch * pid * d2r
    psi = mv.h * mv.roll * pi * d2r
    psi_d = mv.h * mv.roll * pid * d2r
    Ry, Rx, Rz = _rot_y(theta), _rot_x(phi), _rot_z(psi)
    RyRx = Ry @ Rx
    Rm = RyRx @ Rz
    ex, ey, ez = np.eye(3)
    w_B = (theta_d[:, None] * ey[None, :] + (Ry @ ex) * phi_d[:, None] + (RyRx @ ez) * psi_d[:, None])
    # physiological tremor: rotation about a fixed device-side axis
    tw = 2 * math.pi * mv.tremor_hz
    ang = mv.tremor_amp_rad * np.sin(tw * ta + mv.tremor_phase)
    ang_d = mv.tremor_amp_rad * tw * np.cos(tw * ta + mv.tremor_phase)
    Rt = rotvec_to_matrix(ang[:, None] * mv.tremor_axis[None, :])
    C = Rm @ mv.grip
    R_BD = C @ Rt
    w_B = w_B + (C @ mv.tremor_axis) * ang_d[:, None]
    if mv.wobble_amp_rad > 0:
        ww = 2 * math.pi * mv.wobble_hz
        on = np.where(np.isfinite(t) & (t >= 0), 1.0, 0.0)
        tt = np.where(on > 0, t, 0.0)
        a_w = on * mv.wobble_amp_rad * np.sin(ww * tt)
        a_wd = on * mv.wobble_amp_rad * ww * np.cos(ww * tt)
        Rw = rotvec_to_matrix(a_w[:, None] * mv.wobble_axis[None, :])
        w_B = w_B + np.einsum('nij,j->ni', R_BD, mv.wobble_axis) * a_wd[:, None]  # C·Rt·(axis·rate)
        R_BD = R_BD @ Rw
    return {'a_B': a_B, 'R_BD': R_BD, 'w_B': w_B, 'pi': pi}


def earth_from_body(heading_deg: float) -> np.ndarray:
    """R_EB for a participant facing ``heading_deg`` (clockwise from North): columns right, up, backward."""
    s, c = math.sin(math.radians(heading_deg)), math.cos(math.radians(heading_deg))
    return np.array([[c, 0.0, -s], [-s, 0.0, -c], [0.0, 1.0, 0.0]])


# ───────────────────────────── plan / conditions ─────────────────────────────

PRESETS: dict[str, dict[str, Any]] = {
    'tiny': {'orientations': ['upright_toward', 'top_left_toward'], 'speeds': ['natural', 'fast'],
             'start_positions': ['natural'], 'reps': 1, 'odd_variants': []},
    'small': {'orientations': ['upright_toward', 'upright_away', 'top_left_toward', 'top_right_toward'],
              'speeds': ['natural', 'fast', 'slow'], 'start_positions': ['natural'], 'reps': 1, 'odd_variants': []},
    'variants': {'orientations': ['upright_toward', 'top_left_toward'], 'speeds': ['natural'],
                 'start_positions': ['natural', 'shoulder'], 'reps': 1,
                 'odd_variants': ['curved', 'pause', 'twist', 'jerky', 'chin', 'lean_in']},
    'orientations': {'orientations': list(FIXED_ORIENTATIONS), 'speeds': ['natural'], 'start_positions': ['natural'],
                     'reps': 1, 'odd_variants': []},
}


def build_plan(spec: dict[str, Any], first_hand: str = 'right', posture: str = 'standing') -> list[dict[str, Any]]:
    """PlannedBlock list: main blocks (orientation x hand, ABBA hands), then one odd block per hand."""
    other = 'left' if first_hand == 'right' else 'right'
    blocks: list[dict[str, Any]] = []
    ti = 0
    order: list[tuple[str, str, str]] = []
    for k, o in enumerate(spec['orientations']):
        hands = (first_hand, other) if k % 2 == 0 else (other, first_hand)
        order += [('main', hands[0], o), ('main', hands[1], o)]
    if spec.get('odd_variants'):
        order += [('odd', first_hand, 'upright_toward'), ('odd', other, 'upright_toward')]
    for bi, (kind, hand, o) in enumerate(order):
        trials = []
        conds: list[dict[str, Any]] = []
        if kind == 'main':
            for sp in spec['speeds']:
                for st in spec['start_positions']:
                    for rep in range(int(spec.get('reps', 1))):
                        conds.append({'hand': hand, 'orientation': o, 'speed': sp, 'start_position': st,
                                      'variant': 'standard', 'posture': posture, 'rep': rep})
        else:
            for v in spec['odd_variants']:
                conds.append({'hand': hand, 'orientation': o, 'speed': 'natural', 'start_position': 'natural',
                              'variant': v, 'posture': posture, 'rep': 0})
        for j, c in enumerate(conds):
            trials.append({'trial_index': ti, 'block_index': bi, 'trial_in_block': j, 'condition': c})
            ti += 1
        blocks.append({'block_index': bi, 'kind': kind, 'hand': hand, 'orientation': o, 'trials': trials})
    return blocks


# ───────────────────────────── trial simulation ─────────────────────────────


@dataclass
class TrialTruth:
    trial_id: str
    hand: str
    orientation: str  # actual grip (differs from the condition for 'free')
    behavior: str
    go_ms: float | None
    onset_ms: float | None  # true movement onset (relative ms)
    end_ms: float | None  # phone reaches the nose
    duration_s: float
    yaw_net_deg: float  # signed (h * yaw)


def _movement(participant: SynthParticipant, condition: dict[str, Any], orientation: str, draws: dict[str, Any],
              noise: float) -> Movement:
    hand = condition['hand']
    h = 1.0 if hand == 'right' else -1.0
    variant = condition.get('variant', 'standard')
    speed = condition.get('speed', 'natural')
    dominant = hand == participant.dominant
    L = participant.arm_length_m * (0.8 if variant == 'bent_start' else 1.0)
    w = participant.shoulder_half_width_m
    shoulder = np.array([h * w, -0.24, 0.08 - L])
    center = np.array([h * 0.03, -0.12, -(L - 0.08)])
    sp = condition.get('start_position', 'natural')
    b = participant.start_blend if sp == 'natural' else (0.0 if sp == 'shoulder' else 1.0)
    p0 = (1 - b) * shoulder + b * center
    p1 = np.array([0.0, -0.03, -0.01]) + draws['p1_jitter'] * np.array([h, 1.0, 1.0])
    if variant == 'lean_in':
        p1 = p1 + np.array([0.0, 0.0, -0.08])
    elif variant == 'chin':
        p1 = p1 + np.array([0.0, -0.07, 0.0])
    elif variant == 'forehead':
        p1 = p1 + np.array([0.0, 0.06, 0.0])
    arc = participant.arc_m * draws['arc_f']
    yaw = participant.yaw_deg * draws['yaw_f']
    yaw_tr = participant.yaw_transient_deg * draws['yaw_tr_f']
    roll = participant.roll_deg * draws['roll_f']
    pitch = participant.pitch_deg + draws['pitch_d']
    if variant == 'curved':
        arc *= 3.0
    elif variant == 'twist':
        roll *= 3.0
        yaw_tr *= 2.0
    elif variant == 'elbow_high':
        pitch = -25.0
        arc *= 1.5
    T = participant.duration_natural_s * SPEED_FACTOR.get(speed, 1.0) * draws['T_f']
    if not dominant:
        T *= participant.nondominant_slowdown
    if variant == 'lean_in':
        T *= 0.9
    if variant == 'pause':
        segs = [(0.0, T / 2, 0.0, 0.5), (T / 2 + 0.8, T + 0.8, 0.5, 1.0)]
    elif variant == 'jerky':
        d = T / 3 * 1.1
        segs = [(0.0, d, 0.0, 1 / 3), (d + 0.12, 2 * d + 0.12, 1 / 3, 2 / 3), (2 * d + 0.24, 3 * d + 0.24, 2 / 3, 1.0)]
    else:
        segs = [(0.0, T, 0.0, 1.0)]
    S = np.diag([-1.0, 1.0, 1.0])
    tilt = draws['tilt']  # rotation vector (rad), right-hand version
    tilt_h = tilt if h > 0 else -(S @ tilt)
    grip = GRIP_MATRICES[orientation] @ rotvec_to_matrix(tilt_h)
    ax = draws['tremor_axis']
    ax_h = ax if h > 0 else -(S @ ax)
    trem_scale = noise * (1.0 if dominant else participant.nondominant_jitter)
    tw = 2 * math.pi * draws['tremor_hz']
    amp = participant.tremor_dps * trem_scale * math.pi / 180.0 / tw
    return Movement(
        h=h, grip=grip, p0=p0, p1=p1, arc=arc, n_hat=np.array([h * 0.6, 0.8, 0.0]), yaw=yaw, yaw_tr=yaw_tr,
        pitch=pitch, roll=roll, segments=segs, tremor_axis=ax_h, tremor_amp_rad=amp, tremor_hz=draws['tremor_hz'],
        tremor_phase=draws['tremor_phase'],
    )


def _draws(rng: np.random.Generator, n_max: int, platform: SynthPlatform) -> dict[str, Any]:
    """All random numbers of one trial, drawn in a fixed order that does not depend on the hand."""
    tilt = rng.normal(0.0, math.radians(4.0), size=3)
    if np.linalg.norm(tilt) > math.radians(10):
        tilt *= math.radians(10) / np.linalg.norm(tilt)
    ax = rng.normal(size=3)
    d = {
        'free_choice': int(rng.integers(0, len(FIXED_ORIENTATIONS))),
        'T_f': float(np.clip(1 + 0.08 * rng.normal(), 0.8, 1.25)),
        'rt_f': float(rng.uniform(0.75, 1.25)),
        'p1_jitter': rng.normal(0.0, 0.004, size=3),
        'arc_f': float(rng.uniform(0.7, 1.3)),
        'yaw_f': float(rng.uniform(0.8, 1.2)),
        'yaw_tr_f': float(rng.uniform(0.5, 1.5)),
        'roll_f': float(rng.uniform(0.7, 1.3)),
        'pitch_d': float(rng.normal(0.0, 3.0)),
        'tilt': tilt,
        'tremor_axis': ax / np.linalg.norm(ax),
        'tremor_hz': float(rng.uniform(8.0, 12.0)),
        'tremor_phase': float(rng.uniform(0, 2 * math.pi)),
        'fp_u': rng.random(size=16),
        'jitter': rng.uniform(-1.0, 1.0, size=n_max),
        'drop': rng.random(size=n_max),
        'acc_noise': rng.normal(size=(n_max, 3)),
        'gyro_noise': rng.normal(size=(n_max, 3)),
        'ori_noise': rng.normal(size=(n_max, 3)),
    }
    return d


def _sample_times(platform: SynthPlatform, draws: dict[str, Any], noise: float, t_first: float) -> np.ndarray:
    n = draws['jitter'].shape[0]
    step = 1000.0 / platform.hz
    jit = draws['jitter'] * min(platform.jitter_ms, 0.45 * step) * min(noise, 1.0)
    t = t_first + np.arange(n) * step + jit
    keep = draws['drop'] >= platform.drop_prob * (1.0 if noise > 0 else 0.0)
    keep[0] = True
    return t[keep], np.flatnonzero(keep)


def _signals(mv: Movement, platform: SynthPlatform, t_ms: np.ndarray, t_on_ms: float, rows: np.ndarray,
             draws: dict[str, Any], noise: float, R_EB: np.ndarray) -> dict[str, np.ndarray]:
    kin = evaluate_movement(mv, (t_ms - t_on_ms) / 1000.0, t_ms / 1000.0)
    R = kin['R_BD']
    Rt = np.swapaxes(R, -1, -2)
    a_B = kin['a_B']
    f_B = a_B + np.array([0.0, GRAVITY, 0.0])[None, :]
    acc = np.einsum('nij,nj->ni', Rt, a_B)
    accg = np.einsum('nij,nj->ni', Rt, f_B)
    w_D = np.einsum('nij,nj->ni', Rt, kin['w_B']) * (180.0 / math.pi)
    if noise > 0:
        an = draws['acc_noise'][rows] * platform.acc_noise_ms2 * noise
        acc = acc + an
        accg = accg + an
        w_D = w_D + draws['gyro_noise'][rows] * platform.gyro_noise_dps * noise + np.asarray(platform.gyro_bias_dps) * noise
    sign = float(platform.gravity_sign)
    R_ED = R_EB[None, :, :] @ R
    return {'acc': sign * acc, 'accg': sign * accg, 'w': w_D, 'R_ED': R_ED, 'pi': kin['pi']}


def simulate_trial(
    participant: SynthParticipant,
    platform: SynthPlatform,
    condition: dict[str, Any],
    *,
    seed: int,
    ids: dict[str, Any] | None = None,
    noise: float = 1.0,
    behavior: str = 'normal',
    params: dict[str, float] | None = None,
    t0_perf_ms: float = 15000.0,
    t0_epoch_ms: float = 1_790_000_000_000.0,
    gravity_sign_for_detection: int | None = None,
) -> tuple[dict[str, Any], TrialTruth]:
    """One schema-valid TrialRecord plus its ground truth.

    behavior: 'normal' | 'no_move' (ignores GO -> no_movement) | 'too_slow' (20 s movement -> too_long).
    noise: scales sensor noise, tremor, timing jitter and sample drops (0 = exact signals).
    """
    rng = np.random.default_rng(seed)
    hz = platform.hz
    n_max = int(_HORIZON_MS / (1000.0 / hz)) + 2
    draws = _draws(rng, n_max, platform)
    ids = dict(ids or {})
    hand = condition['hand']
    instructed = condition['orientation']
    if instructed == 'free':
        base = FIXED_ORIENTATIONS[draws['free_choice']]
        actual = base if hand == 'right' else mirror_orientation(base)
    else:
        actual = instructed
    p = dict(resolve_params(params or DEFAULT_SEGMENTATION_PARAMS, condition))
    mv = _movement(participant, condition, actual, draws, noise)
    if behavior == 'too_slow':
        mv.segments = [(0.0, 20.0, 0.0, 1.0)]
        mv.wobble_hz = 1.0
        mv.wobble_amp_rad = math.radians(45.0) / (2 * math.pi * mv.wobble_hz)  # peak rate 45 deg/s
        mv.wobble_axis = np.array([0.0, 0.0, mv.h])  # a pseudo-vector: -S·z = -z for the left hand
    t_all, rows = _sample_times(platform, draws, noise, t_first=float(1000.0 / hz * 0.37))
    R_EB = earth_from_body(platform.heading_deg)

    def build(t_on_ms: float) -> dict[str, Any]:
        sig = _signals(mv, platform, t_all, t_on_ms, rows, draws, noise, R_EB)
        acc, accg = sig['acc'], sig['accg']
        rr = device_to_rotation_rate(sig['w'], platform.rr_mapping)  # W3C: (alpha, beta, gamma) = (x, y, z)
        if platform.quantize:
            acc, accg = quantize_like_chrome(acc, 'acc'), quantize_like_chrome(accg, 'acc')
            rr = quantize_like_chrome(rr, 'rr')
        return {
            't': _round_list(t_all, ROUNDING['time']),
            'acc': [_round_list(acc[:, j], ROUNDING['acc']) for j in range(3)],
            'accg': [_round_list(accg[:, j], ROUNDING['acc']) for j in range(3)],
            'rr': [_round_list(rr[:, j], ROUNDING['rr']) for j in range(3)],
            'R_ED': sig['R_ED'], 'pi': sig['pi'],
        }

    def motion_cols(b: dict[str, Any], n: int | None = None) -> dict[str, Any]:
        sl = slice(None) if n is None else slice(0, n)
        acc = b['acc'] if platform.acc_available else [[None] * len(b['t'])] * 3
        return {
            't': b['t'][sl], 'acc_x': acc[0][sl], 'acc_y': acc[1][sl], 'acc_z': acc[2][sl],
            'accg_x': b['accg'][0][sl], 'accg_y': b['accg'][1][sl], 'accg_z': b['accg'][2][sl],
            'rr_alpha': b['rr'][0][sl], 'rr_beta': b['rr'][1][sl], 'rr_gamma': b['rr'][2][sl],
        }

    # pass 1: the still baseline only, to find when the app would give GO
    rest = build(math.inf)
    seg = OnlineSegmenterV1(p)
    cols = motion_cols(rest)
    go_t = None
    still_t = None
    fp = None
    n_still = 0
    for i, t, acc, accg, rr in iter_motion_samples(cols):
        for e in seg.push(t, acc, accg, rr, index=i):
            if e['type'] == 'still':
                still_t = e['t']
                u = float(draws['fp_u'][min(n_still, 15)])
                n_still += 1
                fp = p['foreperiod_min_ms'] + u * (p['foreperiod_max_ms'] - p['foreperiod_min_ms'])
            elif e['type'] in ('still_lost', 'false_start'):
                still_t, fp = None, None
        if still_t is not None and seg.state == 'still' and t >= still_t + fp:
            go_t = t
            break
    if go_t is None:
        raise RuntimeError('synthetic participant never held still long enough for GO (noise too high?)')

    if behavior == 'no_move':
        t_on = math.inf
    else:
        rt = participant.reaction_ms * draws['rt_f']
        t_on = go_t + rt
    full = build(t_on)
    run = run_online_v1(motion_cols(full), p, go_t, keep_trace=False)
    if not run.go_given:
        raise RuntimeError('online_v1 replay did not reach GO at the same sample (internal error)')
    t_list = full['t']
    n_all = len(t_list)
    status = 'complete'
    end_ev = run.event('end')
    fail_ev = run.event('no_movement') or run.event('too_long')
    if end_ev is not None:
        target = end_ev['detected_t'] + p['post_roll_ms']
        n = next((k + 1 for k in range(n_all) if t_list[k] >= target), n_all)
    elif fail_ev is not None:
        status = fail_ev['type']
        n = next((k + 1 for k in range(n_all) if t_list[k] >= fail_ev['t']), n_all)
    else:
        raise RuntimeError(f'trial did not finish within the {_HORIZON_MS} ms horizon')
    run = run_online_v1(motion_cols(full, n), p, go_t, keep_trace=False)  # what the stored record replays to
    t_arr = np.asarray(t_list[:n], dtype=float)
    stop_ms = float(t_arr[-1])

    # orientation stream: W3C Euler angles of R_EB·R_BD (+ small noise), sampled with the motion events
    R_ED = full['R_ED'][:n]
    if noise > 0:
        rv = draws['ori_noise'][rows[:n]] * math.radians(platform.orientation_noise_deg) * noise
        R_ED = R_ED @ rotvec_to_matrix(rv)
    al, be, ga = matrix_to_euler(R_ED)
    al_r = [(_r(v, ROUNDING['angle']) or 0.0) % 360.0 for v in al.tolist()]
    t_ori = _round_list(t_arr + 0.5, ROUNDING['time'])
    ori = {
        't': t_ori, 'ts': _round_list(t_arr + 0.5 - (t_arr[0] + 0.5), ROUNDING['time']),
        'alpha': al_r, 'beta': _round_list(be, ROUNDING['angle']), 'gamma': _round_list(ga, ROUNDING['angle']),
        'compass': [None] * n, 'absolute': False, 'ts_first_raw': _r(t0_perf_ms + 0.5 + t_arr[0], 2),
    }
    quat = None
    if platform.quaternion:
        q = matrix_to_quaternion(full['R_ED'][:n])
        quat = {
            't': _round_list(t_arr + 0.25, ROUNDING['time']),
            'x': _round_list(q[:, 0], ROUNDING['quat']), 'y': _round_list(q[:, 1], ROUNDING['quat']),
            'z': _round_list(q[:, 2], ROUNDING['quat']), 'w': _round_list(q[:, 3], ROUNDING['quat']),
            'sensor': 'RelativeOrientationSensor', 'frequency_hz': float(hz),
        }
    mc = motion_cols(full, n)
    motion = {
        **mc,
        'ts': _round_list(t_arr - t_arr[0], ROUNDING['time']),
        # iOS has reported event.interval in seconds, Chrome in ms: stored exactly as reported
        'interval_raw': round(1.0 / hz, 6) if platform.name == 'ios' else float(round(1000.0 / hz)),
        'ts_first_raw': _r(t0_perf_ms + t_arr[0], 2),
    }
    motion = {k: motion[k] for k in ('t', 'ts', 'acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z',
                                     'rr_alpha', 'rr_beta', 'rr_gamma', 'interval_raw', 'ts_first_raw')}

    # cues, touches, timing
    cues = [{'t': 0.0, 'kind': 'ready'}, {'t': _r(go_t, 2), 'kind': 'go'}]
    end_detected = end_ev['detected_t'] if end_ev else None
    if end_ev is not None:
        cues.append({'t': _r(end_detected, 2), 'kind': 'end'})
    else:
        cues.append({'t': _r(fail_ev['t'], 2), 'kind': 'error'})
    true_end = (t_on + mv.duration_s * 1000.0) if math.isfinite(t_on) else None
    touches = []
    first_touch = None
    if (status == 'complete' and ORIENTATIONS[actual]['screen'] == 'toward' and true_end is not None
            and true_end + 30.0 < stop_ms - 60.0):
        first_touch = _r(true_end + 30.0, 2)
        touches = [{'t': first_touch, 'type': 'start', 'n': 1, 'x': 187.5, 'y': 400.0},
                   {'t': _r(stop_ms - 50.0, 2), 'type': 'end', 'n': 0, 'x': 187.5, 'y': 400.0}]
    res = run.result
    on_i, en_i = res['onset_index'], res['end_index']
    timing = {
        't0_perf_ms': t0_perf_ms, 't0_epoch_ms': t0_epoch_ms,
        'still_ms': _r(run.still_t, 2), 'go_ms': _r(go_t, 2),
        'onset_ms': t_list[on_i] if on_i is not None else None,
        'end_ms': t_list[en_i] if en_i is not None else None,
        'end_detected_ms': _r(end_detected, 2) if end_detected is not None else None,
        'stop_ms': stop_ms, 'first_touch_ms': first_touch,
        'foreperiod_ms': _r(fp, 2) if fp is not None else None, 'false_starts': 0,
    }
    # detected rotation (orientationDetect port) on the raw accg, with the session's gravity sign
    gs = platform.gravity_sign if gravity_sign_for_detection is None else gravity_sign_for_detection
    screen_hint = ORIENTATIONS[instructed]['screen']
    base_g = (res['baseline'] or {}).get('gravity')
    G = np.column_stack([np.asarray([np.nan if v is None else v for v in mc[k]], dtype=float)
                         for k in ('accg_x', 'accg_y', 'accg_z')])
    tail = G[t_arr >= stop_ms - 200.0]
    end_g = tail.mean(axis=0).tolist() if len(tail) else None
    rot_start = rotation_from_gravity(base_g, gs, screen_hint)
    rot_end = rotation_from_gravity(end_g, gs, screen_hint)
    instructed_rot = ORIENTATIONS[instructed]['rotation']
    detected = {
        'rotation_start': rot_start, 'rotation_end': rot_end,
        'tilt_change_deg': _r(tilt_change_deg(base_g, end_g), 3),
        'orientation_match': (None if instructed_rot is None or rot_start == 'unknown' else rot_start == instructed_rot),
    }
    rs = rate_stats(t_arr)
    quality = {
        'n_motion': n, 'n_orientation': n, 'n_quaternion': n if quat else 0,
        'motion_hz': _r(rs['hz'], 3), 'motion_dt_ms_median': _r(rs['dt_median'], 3),
        'motion_dt_ms_max': _r(rs['dt_max'], 3), 'gaps_over_50ms': int(rs['gaps_over_50ms']),
        'acc_available': bool(platform.acc_available), 'rotation_available': True, 'hidden_during': False,
        'warnings': [] if detected['orientation_match'] in (True, None) else ['orientation_mismatch'],
    }

    def screen_of(g: Any) -> tuple[str, float]:
        if g is None:
            return 'portrait-primary', 0.0
        v = np.asarray(g, dtype=float) * gs
        if abs(v[0]) > abs(v[1]) and abs(v[0]) > abs(v[2]):
            return ('landscape-primary', 90.0) if v[0] > 0 else ('landscape-secondary', 270.0)
        return 'portrait-primary', 0.0

    s0, s1 = screen_of(base_g), screen_of(end_g)
    seg_out = dict(res)  # stored unrounded: this IS the port's output
    record = {
        'record_type': 'trial', 'schema_version': SCHEMA_VERSION, 'app_version': APP_VERSION,
        'trial_id': ids.get('trial_id') or make_id(np.random.default_rng(seed + 7_777_777)),
        'session_id': ids.get('session_id', 'synthetic-session'),
        'participant_id': ids.get('participant_id', participant.participant_id),
        'study_code': ids.get('study_code', 'synthetic-py'),
        'created_at': iso_time(t0_epoch_ms + stop_ms),
        'block_index': int(ids.get('block_index', 0)), 'trial_index': int(ids.get('trial_index', 0)),
        'trial_in_block': int(ids.get('trial_in_block', 0)), 'attempt': int(ids.get('attempt', 1)),
        'simulated': True, 'condition': dict(condition),
        'outcome': {'status': status, 'accepted': status == 'complete', 'participant_flag': None, 'note': None},
        'timing': timing, 'segmentation': seg_out, 'detected': detected, 'quality': quality,
        'screen': {'type_start': s0[0], 'angle_start': s0[1], 'type_end': s1[0], 'angle_end': s1[1]},
        'streams': {'motion': motion, 'orientation': ori, 'quaternion': quat, 'touches': touches, 'cues': cues},
    }
    truth = TrialTruth(
        trial_id=record['trial_id'], hand=hand, orientation=actual, behavior=behavior, go_ms=go_t,
        onset_ms=t_on if math.isfinite(t_on) else None, end_ms=true_end, duration_s=mv.duration_s,
        yaw_net_deg=mv.h * mv.yaw,
    )
    return record, truth


# ───────────────────────────── profiles / sessions / datasets ─────────────────────────────


def make_profile(p: SynthParticipant, created_epoch_ms: float) -> dict[str, Any]:
    d = p.dominant
    if p.handedness == 'ambidextrous':
        sf = {'writing': f'usually_{d}', 'throwing': 'both', 'toothbrush': 'both', 'spoon': f'usually_{d}'}
    else:
        o = 'left' if d == 'right' else 'right'
        sf = {'writing': f'always_{d}', 'throwing': f'always_{d}', 'toothbrush': f'usually_{d}', 'spoon': f'always_{d}'}
        if p.handedness == 'left':  # many left-handers are mixed for some items
            sf['throwing'] = f'usually_{o}' if p.height_cm > 175 else f'always_{d}'
    lq, cls = score_ehi(sf)
    return {
        'participant_id': p.participant_id, 'nickname': None, 'age_years': int(p.age_years), 'sex': p.sex,
        'handedness_self': p.handedness, 'ehi_sf': sf, 'ehi_lq': lq, 'ehi_class': cls,
        'phone_hand': p.dominant if p.handedness != 'ambidextrous' else 'both',
        'height_cm': float(p.height_cm), 'arm_issue': False, 'phone_model': None, 'phone_case': None,
        'created_at': iso_time(created_epoch_ms), 'updated_at': iso_time(created_epoch_ms),
    }


def _device(platform: SynthPlatform) -> dict[str, Any]:
    if platform.name == 'ios':
        ua = ('Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) '
              'Version/18.5 Mobile/15E148 Safari/604.1')
        return {'user_agent': ua, 'platform': 'ios', 'browser': 'safari', 'os_version': '18.5', 'model_hint': None,
                'standalone': True, 'screen_w': 390, 'screen_h': 844, 'dpr': 3, 'language': 'en-US',
                'hardware_concurrency': 6, 'device_memory_gb': None, 'time_origin_ms': 1_790_000_000_000.0,
                'engine': platform.engine, 'in_app_browser': None}
    ua = ('Mozilla/5.0 (Linux; Android 15; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 '
          'Mobile Safari/537.36')
    return {'user_agent': ua, 'platform': 'android', 'browser': 'chrome', 'os_version': '15', 'model_hint': 'Pixel 8',
            'standalone': False, 'screen_w': 412, 'screen_h': 915, 'dpr': 2.625, 'language': 'he-IL',
            'hardware_concurrency': 8, 'device_memory_gb': 8, 'time_origin_ms': 1_790_000_000_000.0,
            'engine': platform.engine, 'in_app_browser': None}


def _calibration(platform: SynthPlatform, epoch_ms: float, include_sign: bool, noise: float = 1.0,
                 axis_check: bool = True) -> dict[str, Any]:
    s = platform.gravity_sign
    b = device_to_rotation_rate(np.asarray(platform.gyro_bias_dps, dtype=float) * noise, platform.rr_mapping)
    bias = {'alpha': round(float(b[0]), 3) + 0.0, 'beta': round(float(b[1]), 3) + 0.0, 'gamma': round(float(b[2]), 3) + 0.0}

    def pose(v: tuple[float, float, float]) -> dict[str, Any]:
        return {'n': 120, 'duration_ms': 2000.0, 'accg_mean': [s * v[0] + 0.0, s * v[1] + 0.0, s * v[2] + 0.0],
                'accg_sd': [0.01, 0.01, 0.01], 'acc_mean': [0.0, 0.0, 0.0], 'rr_mean': dict(bias),
                'rr_sd': {'alpha': 0.3, 'beta': 0.3, 'gamma': 0.3}, 'still': True}

    cal = {
        'performed_at': iso_time(epoch_ms), 'flat': pose((0.0, 0.0, GRAVITY)), 'upright': pose((0.0, GRAVITY, 0.0)),
        'gravity_sign': s if include_sign else None, 'gravity_magnitude': GRAVITY,
        'upright_check': 'ok' if include_sign else 'unclear', 'gyro_bias_dps': dict(bias), 'notes': [],
    }
    if axis_check:
        # tilt of 45 deg about device y (right edge down), as the app's calibration gesture
        axis = [0.0, 1.0, 0.0]
        integral = device_to_rotation_rate(np.array([0.0, 45.0, 0.0]), platform.rr_mapping)
        cal['axis_check'] = {
            'n': 150, 'duration_ms': 2500.0, 'tilt_deg': 45.0, 'device_axis': axis,
            'rr_integral_deg': {'alpha': float(integral[0]) + 0.0, 'beta': float(integral[1]) + 0.0,
                                'gamma': float(integral[2]) + 0.0},
            'verdict': platform.rr_mapping, 'units': 'deg_per_s', 'similarity': 0.99, 'magnitude_ratio': 1.0,
        }
    return cal


def make_session(
    participant: SynthParticipant,
    platform: SynthPlatform,
    plan: list[dict[str, Any]],
    *,
    session_id: str,
    study_code: str,
    started_epoch_ms: float,
    n_attempts: int,
    n_accepted: int,
    ended_epoch_ms: float,
    calibration: str = 'ok',  # ok | no_sign | no_axis_check | missing
    seed: int = 0,
    spec: dict[str, Any] | None = None,
    noise: float = 1.0,
) -> dict[str, Any]:
    n_planned = sum(len(b['trials']) for b in plan)
    spec = spec or {}
    custom = {
        'orientations': list(spec.get('orientations') or ['upright_toward']),
        'speeds': list(spec.get('speeds') or ['natural']),
        'start_positions': list(spec.get('start_positions') or ['natural']),
        'reps': int(spec.get('reps', 1)), 'odd_variants': list(spec.get('odd_variants') or []), 'include_free': False,
    }
    cal = None if calibration == 'missing' else _calibration(
        platform, started_epoch_ms + 30_000, calibration in ('ok', 'no_axis_check'), noise,
        axis_check=calibration != 'no_axis_check')
    rs = {'duration_ms': 3000.0, 'motion_hz': platform.hz, 'orientation_hz': platform.hz,
          'quaternion_hz': platform.hz if platform.quaternion else None, 'motion_dt_ms_median': 1000.0 / platform.hz,
          'motion_dt_ms_p95': 1000.0 / platform.hz + 1.0, 'acc_available': platform.acc_available,
          'accg_available': True, 'rotation_available': True, 'interval_raw': None}
    return {
        'record_type': 'session', 'schema_version': SCHEMA_VERSION, 'app_version': APP_VERSION,
        'session_id': session_id, 'participant_id': participant.participant_id, 'study_code': study_code,
        'started_at': iso_time(started_epoch_ms), 'updated_at': iso_time(ended_epoch_ms),
        'ended_at': iso_time(ended_epoch_ms), 'status': 'completed',
        'participant': make_profile(participant, started_epoch_ms - 60_000),
        'device': _device(platform),
        'capabilities': {
            'secure_context': True, 'device_motion': True, 'device_orientation': True,
            'permission_api': platform.name == 'ios',
            'permission_state': 'granted' if platform.name == 'ios' else 'not_required',
            'generic_relative_orientation': platform.quaternion, 'wake_lock': True,
            'vibrate': platform.name == 'android', 'share_files': True, 'measured': rs,
        },
        'calibration': cal,
        'protocol': {
            'preset': 'custom', 'protocol_version': PROTOCOL_VERSION, 'seed': int(seed) % (2 ** 32),
            'posture': 'standing', 'first_hand': plan[0]['hand'] if plan else 'right',
            'n_planned_trials': n_planned, 'estimated_minutes': round(n_planned * 9 / 60 * 2) / 2 + 1.0,
            'custom': custom, 'plan': plan,
        },
        'settings': {'language': 'en', 'audio': True, 'speech': False,
                     'segmentation_params': dict(DEFAULT_SEGMENTATION_PARAMS)},
        'progress': {'next_trial_index': n_planned, 'n_attempts': n_attempts, 'n_accepted': n_accepted},
        'consent': {'version': '2026-10-v1', 'accepted_at': iso_time(started_epoch_ms - 30_000)},
        'timezone_offset_min': -180, 'simulated': True, 'notes': None,
    }


@dataclass
class SynthDataset:
    participants: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    truth: dict[str, TrialTruth] = field(default_factory=dict)
    sim_participants: list[SynthParticipant] = field(default_factory=list)

    def truth_frame(self) -> Any:
        import pandas as pd
        return pd.DataFrame([asdict(t) for t in self.truth.values()]).set_index('trial_id')


def make_dataset(
    n_participants: int = 6,
    *,
    seed: int = 0,
    preset: str | dict[str, Any] = 'small',
    left_fraction: float = 0.3,
    platforms: tuple[str, ...] = ('android', 'ios'),
    noise: float = 1.0,
    failure_rate: float = 0.0,
    calibration_missing_fraction: float = 0.0,
    study_code: str = 'synthetic-py',
    start_epoch_ms: float = 1_790_000_000_000.0,
    dominance_asymmetry: bool = True,
    participant_overrides: dict[str, Any] | None = None,
    platform_overrides: dict[str, dict[str, Any]] | None = None,
    calibration: str = 'ok',
) -> SynthDataset:
    """A complete multi-participant dataset (sessions + every attempt, failures redone once).

    platforms alternate per participant; ``platform_overrides`` {'android': {'quantize': True}, ...}
    changes SynthPlatform fields per platform. calibration: 'ok' | 'no_sign' | 'no_axis_check' |
    'missing' for every session (``calibration_missing_fraction`` drops it for a random share).
    """
    rng = np.random.default_rng(seed)
    spec = PRESETS[preset] if isinstance(preset, str) else preset
    ds = SynthDataset()
    epoch = start_epoch_ms
    for k in range(n_participants):
        prng = np.random.default_rng([seed, k, 1])
        over = dict(participant_overrides or {})
        if not dominance_asymmetry:
            over.update(nondominant_slowdown=1.0, nondominant_jitter=1.0)
        part = random_participant(prng, left_fraction=left_fraction, **over)
        pname = platforms[k % len(platforms)]
        plat = default_platform(pname, prng, **((platform_overrides or {}).get(pname) or {}))
        first_hand = 'right' if (k % 2 == 0) else 'left'
        plan = build_plan(spec, first_hand=first_hand)
        session_id = make_id(prng)
        cal_mode = 'missing' if prng.random() < calibration_missing_fraction else calibration
        t_epoch = epoch + 120_000
        perf = 20_000.0
        trials = []
        n_acc = 0
        for blk in plan:
            for pt in blk['trials']:
                attempt = 1
                while True:
                    tseed = int(rng.integers(0, 2 ** 31 - 1))
                    fail = attempt == 1 and failure_rate > 0 and rng.random() < failure_rate
                    behavior = ('no_move' if rng.random() < 0.5 else 'too_slow') if fail else 'normal'
                    ids = {'trial_id': make_id(prng), 'session_id': session_id, 'participant_id': part.participant_id,
                           'study_code': study_code, 'block_index': pt['block_index'],
                           'trial_index': pt['trial_index'], 'trial_in_block': pt['trial_in_block'], 'attempt': attempt}
                    rec, truth = simulate_trial(part, plat, pt['condition'], seed=tseed, ids=ids, noise=noise,
                                                behavior=behavior, t0_perf_ms=perf, t0_epoch_ms=t_epoch)
                    trials.append(rec)
                    ds.truth[rec['trial_id']] = truth
                    dur = rec['timing']['stop_ms']
                    perf += dur + 4000.0
                    t_epoch += dur + 4000.0
                    if rec['outcome']['accepted']:
                        n_acc += 1
                        break
                    attempt += 1
        sess = make_session(part, plat, plan, session_id=session_id, study_code=study_code,
                            started_epoch_ms=epoch + 60_000, n_attempts=len(trials), n_accepted=n_acc,
                            ended_epoch_ms=t_epoch, calibration=cal_mode, seed=seed * 1000 + k, spec=spec,
                            noise=noise)
        ds.participants.append(sess['participant'])
        ds.sessions.append(sess)
        ds.trials.extend(trials)
        ds.sim_participants.append(part)
        epoch = t_epoch + 3_600_000
    return ds


def trial_file_name(trial: dict[str, Any]) -> str:
    """Same naming as the backend / app synthesizer: trial_<idx:3>_a<attempt>_<trial_id>.json."""
    return f"trial_{int(trial['trial_index']):03d}_a{int(trial['attempt'])}_{trial['trial_id']}.json"


def write_folder_tree(ds: SynthDataset, root: str | Path) -> Path:
    """<root>/<study>/<participant>/<session>/session.json + trial_*.json (the backend layout)."""
    root = Path(root)
    by_session = {s['session_id']: s for s in ds.sessions}
    for s in ds.sessions:
        d = root / s['study_code'] / s['participant_id'] / s['session_id']
        d.mkdir(parents=True, exist_ok=True)
        (d / 'session.json').write_text(json.dumps(s), encoding='utf-8')
    for t in ds.trials:
        s = by_session[t['session_id']]
        d = root / s['study_code'] / s['participant_id'] / s['session_id']
        d.mkdir(parents=True, exist_ok=True)
        (d / trial_file_name(t)).write_text(json.dumps(t), encoding='utf-8')
    return root


def export_bundle(ds: SynthDataset, *, session_ids: list[str] | None = None, exported_epoch_ms: float | None = None) -> dict[str, Any]:
    """The app's 'nose-touch-export' bundle for some (default all) sessions."""
    keep = set(session_ids) if session_ids is not None else {s['session_id'] for s in ds.sessions}
    sessions = [s for s in ds.sessions if s['session_id'] in keep]
    pids = {s['participant_id'] for s in sessions}
    when = exported_epoch_ms if exported_epoch_ms is not None else 1_790_000_000_000.0 + 9e8
    return {
        'format': 'nose-touch-export', 'schema_version': SCHEMA_VERSION, 'app_version': APP_VERSION,
        'exported_at': iso_time(when),
        'participants': [p for p in ds.participants if p['participant_id'] in pids],
        'sessions': sessions, 'trials': [t for t in ds.trials if t['session_id'] in keep],
    }
