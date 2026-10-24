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
Protocol 2.0.0 (``preset='street'``, schema 1.2): the plan is ``build_street_plan``, an exact port of
protocol.ts (same seed -> same plan, the app's mulberry32 / mixSeed / Fisher-Yates); the profile is the short
one (name, age, sex, handedness, phone hand, height; no EHI); the calibration is the app's implicit_v1 (from
the pre-GO hold of the first still upright trial; no table, no tilt, no gyro bias); trials are auto-accepted
('auto_accept', or 'auto_reject:orientation' and re-run), the choice trial reports the participant's phone
hand, and the recording continues through the return of the arm (``v2=True`` in ``simulate_trial``: back to
the start pose 250-450 ms after the end cue in 0.8-1.2 s, until still again or return_cap_ms) with
``timing.return_onset_ms`` / ``return_end_ms``, touch contact details and, on Android, the absolute
orientation streams.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import (
    APP_PROTOCOL_VERSION, DEFAULT_SEGMENTATION_PARAMS, FIXED_ORIENTATIONS, GRAVITY, GRIP_MATRICES, ORIENTATIONS,
    ROUNDING, SCHEMA_VERSION, device_to_rotation_rate, engine_gravity_sign, mirror_orientation, resolve_params,
    rotation_from_gravity, tilt_change_deg,
)
from .preprocess import matrix_to_euler, matrix_to_quaternion, quantize_like_chrome, rate_stats, rotvec_to_matrix
from .segment import OnlineSegmenterV1, iter_motion_samples, run_online_v1

APP_VERSION = 'motionlab-synth-0.2.0'
#: protocol_version of 'street' sessions (= constants.ts PROTOCOL_VERSION).
PROTOCOL_VERSION = APP_PROTOCOL_VERSION
#: protocol_version of the sessions of the legacy presets below ('tiny', 'small', ...): they contain
#: screen-away grips, which only protocol 1.0.0 asked for.
LEGACY_PROTOCOL_VERSION = '1.0.0'
ID_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
#: Movement duration factor per speed. slow 1.7 (was 2.6): the pilot's slow approaches peaked at >= 50 deg/s
#: and took about twice the natural time; at 2.6 this model peaked at ~33 deg/s, near the protocol 2.0.0 slow
#: onset floor (25 deg/s), and online_v1 missed a quarter of them (60/80); 1.7 peaks at ~44 deg/s (median)
#: and is detected 80/80 (1.8: ~42 deg/s, 80/80).
SPEED_FACTOR = {'natural': 1.0, 'fast': 0.55, 'slow': 1.7}
_EHI_SCORE = {'always_right': 100, 'usually_right': 50, 'both': 0, 'usually_left': -50, 'always_left': -100}
_HORIZON_MS = 30000.0
#: protocol 2.0.0 return of the arm (spec v2 section 3): a return movement is |w| > 45 deg/s or
#: |a| > 1.5 m/s^2 (smoothed) for >= 100 ms; then still (still_gyro_dps / still_acc_ms2) for still_hold_ms.
RETURN_GYRO_DPS = 45.0
RETURN_ACC_MS2 = 1.5
RETURN_SUSTAIN_MS = 100.0
#: consent version of protocol 2.0.0 sessions (config.json consent_version).
CONSENT_VERSION_V2 = '2026-10-v2'


# ───────────────────────────── port of util.ts / protocol.ts: the 'street' plan ─────────────────────────────

_U32 = 0xFFFFFFFF
#: "PLAN" in ASCII: salt of the plan's random stream (protocol.ts PLAN_SALT).
PLAN_SALT = 0x504C414E


def _imul(a: int, b: int) -> int:
    """JavaScript Math.imul on uint32 bit patterns: the low 32 bits of the product."""
    return (a * b) & _U32


def mulberry32(seed: int) -> Any:
    """Port of util.makeRng (mulberry32): a uint32-seeded PRNG returning floats in [0, 1), bit-exact."""
    state = [int(seed) & _U32]

    def rng() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & _U32
        t = state[0]
        t = _imul(t ^ (t >> 15), t | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & _U32
        return ((t ^ (t >> 14)) & _U32) / 4294967296.0

    return rng


def mix_seed(seed: int, salt: int) -> int:
    """Port of util.mixSeed (murmur3 finalizer of seed ^ imul(salt, golden ratio)), bit-exact."""
    h = ((int(seed) & _U32) ^ _imul(int(salt) & _U32, 0x9E3779B9)) & _U32
    h = _imul(h ^ (h >> 16), 0x85EBCA6B)
    h = _imul(h ^ (h >> 13), 0xC2B2AE35)
    return (h ^ (h >> 16)) & _U32


def js_shuffle(items: Any, rng: Any) -> list[Any]:
    """Port of util.shuffle (Fisher-Yates from the end; a new list)."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = min(i, int(math.floor(rng() * (i + 1))))
        out[i], out[j] = out[j], out[i]
    return out


#: protocol.ts STREET_VARIETY_POOL / STREET_ALWAYS_CELL as (speed, start_position, variant).
STREET_VARIETY_POOL: tuple[tuple[str, str, str], ...] = (
    ('natural', 'shoulder', 'standard'), ('natural', 'center', 'standard'), ('natural', 'natural', 'curved'),
    ('natural', 'natural', 'pause'), ('natural', 'natural', 'twist'), ('natural', 'natural', 'elbow_high'),
    ('natural', 'natural', 'bent_start'), ('natural', 'natural', 'lean_in'), ('natural', 'natural', 'jerky'),
)
STREET_ALWAYS_CELL: tuple[str, str, str] = ('natural', 'natural', 'tilted_touch')
STREET_TIER3_DRAWN = 3
STREET_BONUS_DRAWN = 2


def resolve_first_hand(seed: int, first_hand: str | None = None) -> str:
    """protocol.resolveFirstHand: the override, else seed bit 0 (1 -> left)."""
    if first_hand in ('left', 'right'):
        return str(first_hand)
    return 'left' if int(seed) & 1 else 'right'


def street_sideways(seed: int) -> tuple[str, str]:
    """protocol.streetSideways: (S, S') = (top_left_toward, top_right_toward), swapped when seed bit 1 is set."""
    if ((int(seed) & _U32) >> 1) & 1:
        return 'top_right_toward', 'top_left_toward'
    return 'top_left_toward', 'top_right_toward'


def _materialize(specs: list[tuple[int | None, str, str, str, list[tuple[str, str, str]]]], posture: str,
                 rng: Any) -> list[dict[str, Any]]:
    """protocol.materialize: shuffle each block's cells in block order, number trials, count reps."""
    blocks: list[dict[str, Any]] = []
    ti = 0
    for bi, (tier, kind, hand, orientation, cells) in enumerate(specs):
        seen: dict[tuple[str, str, str], int] = {}
        trials = []
        for j, c in enumerate(js_shuffle(cells, rng)):
            rep = seen.get(c, 0)
            seen[c] = rep + 1
            cond = {'hand': hand, 'orientation': orientation, 'speed': c[0], 'start_position': c[1], 'variant': c[2],
                    'posture': posture, 'rep': rep, 'hand_source': 'reported' if kind == 'choice' else 'instructed'}
            trials.append({'trial_index': ti, 'block_index': bi, 'trial_in_block': j, 'condition': cond})
            ti += 1
        block: dict[str, Any] = {'block_index': bi, 'kind': kind}
        if tier is not None:
            block.update(tier=tier, optional=tier == 4)
        block.update(hand=hand, orientation=orientation, trials=trials)
        blocks.append(block)
    return blocks


def build_street_plan(seed: int, posture: str = 'standing', first_hand: str | None = None) -> list[dict[str, Any]]:
    """The 'street' plan of protocol 2.0.0, an exact port of protocol.buildPlan({preset: 'street'}): the same
    seed gives the same blocks, tiers, cells and trial order as the app (a test pins TS-generated plans).

    13 blocks / 39 trials (9 required blocks, 29 trials; 4 optional bonus blocks, 10 trials): tier 0 the
    choice trial (hand reported after it), tier 1 upright x both hands (natural, natural, fast, slow), tier 2
    both sideways grips x both hands (natural, natural, fast), tier 3 four variety cells per hand
    (tilted_touch + 3 drawn from STREET_VARIETY_POOL), tier 4 (optional) upright (natural, fast, natural)
    and the 2 next variety cells. A = first hand (seed bit 0), S = first sideways grip (seed bit 1).
    """
    rng = mulberry32(mix_seed(seed, PLAN_SALT))
    a = resolve_first_hand(seed, first_hand)
    b = 'left' if a == 'right' else 'right'
    s1, s2 = street_sideways(seed)
    drawn = js_shuffle(STREET_VARIETY_POOL, rng)
    variety = [STREET_ALWAYS_CELL, *drawn[:STREET_TIER3_DRAWN]]
    bonus = drawn[STREET_TIER3_DRAWN:STREET_TIER3_DRAWN + STREET_BONUS_DRAWN]
    n, f, s = ('natural', 'natural', 'standard'), ('fast', 'natural', 'standard'), ('slow', 'natural', 'standard')
    U = 'upright_toward'
    side = [n, n, f]
    specs = [
        (0, 'choice', a, 'free', [n]),
        (1, 'main', a, U, [n, n, f, s]), (1, 'main', b, U, [n, n, f, s]),
        (2, 'main', b, s1, side), (2, 'main', a, s1, side), (2, 'main', a, s2, side), (2, 'main', b, s2, side),
        (3, 'odd', b, U, variety), (3, 'odd', a, U, variety),
        (4, 'main', a, U, [n, f, n]), (4, 'main', b, U, [n, f, n]), (4, 'odd', b, U, bonus), (4, 'odd', a, U, bonus),
    ]
    return _materialize(specs, posture, rng)


#: protocol.estimateMinutes: seconds per trial by speed, +2 s to read a non-default start / variant, 6 s per
#: block change, 40 s of onboarding; over the non-optional, non-empty blocks; rounded to 0.5 min.
_TRIAL_SECONDS = {'fast': 5.5, 'natural': 6.5, 'slow': 10.0}


def estimate_minutes(plan: list[dict[str, Any]]) -> float:
    blocks = [b for b in plan if b.get('optional') is not True and b.get('trials')]
    sec = 40.0 + max(0, len(blocks) - 1) * 6.0
    for blk in blocks:
        for t in blk['trials']:
            c = t['condition']
            read = 2.0 if (c.get('start_position') != 'natural' or c.get('variant') != 'standard') else 0.0
            sec += _TRIAL_SECONDS.get(c.get('speed'), _TRIAL_SECONDS['natural']) + read
    return math.floor(sec / 60.0 * 2.0 + 0.5) / 2.0  # JS Math.round (halves up)


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
    elif variant == 'tilted_touch':
        pitch += 35.0  # the top edge meets the nose: the phone arrives tilted ~35 deg about its x axis
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
        # protocol 2.0.0 additions (drawn last, so the draws above stay those of earlier versions)
        'ret_delay_u': float(rng.random()),  # return of the arm: 250-450 ms after the end cue
        'ret_dur_u': float(rng.random()),  # lasting 0.8-1.2 s
        'touch_u': rng.random(size=4),  # nose contact: radius, force, position jitter
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
    free_orientation: str | None = None,
    grip_tilt: bool = True,
    v2: bool = False,
    screen_wh: tuple[float, float] | None = None,
) -> tuple[dict[str, Any], TrialTruth]:
    """One schema-valid TrialRecord plus its ground truth.

    behavior: 'normal' | 'no_move' (ignores GO -> no_movement) | 'too_slow' (20 s movement -> too_long).
    noise: scales sensor noise, tremor, timing jitter and sample drops (0 = exact signals).
    free_orientation: the grip actually used when the instructed orientation is 'free' (default: a random
    fixed orientation; the street choice trial uses 'upright_toward', the natural grip).
    grip_tilt: False holds the phone exactly in the grip (no random tilt of up to 10 deg), e.g. to compare
    a trial with its screen-flipped twin (a tilt does not commute with the flip).
    v2: a protocol 2.0.0 / schema 1.2 trial (see the module docstring): the recording continues through the
    return of the arm (timing.return_onset_ms / return_end_ms), touches carry contact details (radius,
    force, angle, normalised position on ``screen_wh``), Android adds the absolute orientation streams
    (quality.n_quaternion_abs / n_orientation_abs), and the outcome is auto-accepted ('auto_accept'; an
    orientation mismatch is 'auto_reject:orientation', not accepted).
    """
    rng = np.random.default_rng(seed)
    hz = platform.hz
    n_max = int(_HORIZON_MS / (1000.0 / hz)) + 2
    draws = _draws(rng, n_max, platform)
    if not grip_tilt:
        draws['tilt'] = np.zeros(3)
    ids = dict(ids or {})
    hand = condition['hand']
    instructed = condition['orientation']
    if instructed == 'free':
        if free_orientation is not None:
            actual = free_orientation
        else:
            base = FIXED_ORIENTATIONS[draws['free_choice']]
            actual = base if hand == 'right' else mirror_orientation(base)
    else:
        actual = instructed
    p = dict(resolve_params(params or DEFAULT_SEGMENTATION_PARAMS, condition))
    mv = _movement(participant, condition, actual, draws, noise)
    outward_s = mv.duration_s  # the phone reaches the nose (a v2 return segment is appended later)
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
    return_onset = return_end = None
    if end_ev is not None and v2 and math.isfinite(t_on):
        # protocol 2.0.0: the arm goes back out 250-450 ms after the end cue (0.8-1.2 s); the samples before
        # it are unchanged, so the online segmentation (ended at the end cue) is too
        delay_ms = 250.0 + 200.0 * draws['ret_delay_u']
        t_ret = (end_ev['detected_t'] + delay_ms - t_on) / 1000.0
        mv.segments = [*mv.segments, (t_ret, t_ret + 0.8 + 0.4 * draws['ret_dur_u'], 1.0, 0.0)]
        full = build(t_on)
        t_list = full['t']
        n, return_onset, return_end = _return_stop(full, end_ev['detected_t'], p, platform)
    elif end_ev is not None:
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
    true_end = (t_on + outward_s * 1000.0) if math.isfinite(t_on) else None
    touches = []
    first_touch = None
    lift = (return_onset if return_onset is not None else stop_ms - 50.0)  # the phone leaves the nose
    nose_on_screen = condition.get('variant') != 'tilted_touch'  # tilted_touch: the top edge meets the nose
    if (status == 'complete' and ORIENTATIONS[actual]['screen'] == 'toward' and true_end is not None
            and nose_on_screen and true_end + 30.0 < min(lift, stop_ms - 60.0)):
        first_touch = _r(true_end + 30.0, 2)
        touches = [{'t': first_touch, 'type': 'start', 'n': 1, 'x': 187.5, 'y': 400.0},
                   {'t': _r(min(lift, stop_ms - 50.0), 2), 'type': 'end', 'n': 0, 'x': 187.5, 'y': 400.0}]
        if v2:  # schema 1.2 contact details (the nose: an ellipse of ~1-2 cm, no force on Android)
            u = draws['touch_u']
            w_css, h_css = screen_wh or ((390.0, 844.0) if platform.name == 'ios' else (412.0, 915.0))
            x = round(float(w_css / 2 + (u[2] - 0.5) * 40.0), 2)
            y = round(float(0.45 * h_css + (u[3] - 0.5) * 60.0), 2)
            rad = round(8.0 + 12.0 * float(u[0]), 2)
            details = {'radius_x': rad, 'radius_y': round(rad * 1.2, 2),
                       'force': 0.0 if platform.name == 'android' else round(0.1 + 0.5 * float(u[1]), 3),
                       'rotation_angle': 0.0, 'x_norm': round(x / w_css, 4), 'y_norm': round(y / h_css, 4)}
            touches = [{**tch, 'x': x, 'y': y, **details} for tch in touches]
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
    if v2:  # schema 1.2: written by the v2 app for every trial (null = the return was not observed)
        timing['return_onset_ms'] = return_onset
        timing['return_end_ms'] = return_end
    # detected rotation (orientationDetect port) on the raw accg, with the session's gravity sign
    gs = platform.gravity_sign if gravity_sign_for_detection is None else gravity_sign_for_detection
    screen_hint = ORIENTATIONS[instructed]['screen']
    base_g = (res['baseline'] or {}).get('gravity')
    G = np.column_stack([np.asarray([np.nan if v is None else v for v in mc[k]], dtype=float)
                         for k in ('accg_x', 'accg_y', 'accg_z')])
    if v2 and timing['end_ms'] is not None and end_detected is not None:
        # the v2 app: the mean accg AT THE NOSE (end_ms ... end_detected_ms), the recording goes on through
        # the return of the arm (trialController.ts)
        tail = G[(t_arr >= timing['end_ms']) & (t_arr <= end_detected)]
    else:
        tail = G[t_arr >= stop_ms - 200.0]  # the last END_GRAVITY_WINDOW_MS before the stop
    tail = tail[np.isfinite(tail).all(axis=1)]
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

    stop_g = end_g
    if v2:  # screen.type_end is read when the recording stops (after the return of the arm)
        last = G[t_arr >= stop_ms - 200.0]
        last = last[np.isfinite(last).all(axis=1)]
        stop_g = last.mean(axis=0).tolist() if len(last) else None
    s0, s1 = screen_of(base_g), screen_of(stop_g)
    seg_out = dict(res)  # stored unrounded: this IS the port's output
    outcome = {'status': status, 'accepted': status == 'complete', 'participant_flag': None, 'note': None}
    streams: dict[str, Any] = {'motion': motion, 'orientation': ori, 'quaternion': quat, 'touches': touches, 'cues': cues}
    if v2:
        # hands-free flow (review_mode 'auto'): a complete trial whose grip matches is accepted at once; a
        # mismatching grip is saved as rejected and re-run
        if status == 'complete':
            ok = detected['orientation_match'] is not False
            outcome.update(accepted=ok, note='auto_accept' if ok else 'auto_reject:orientation')
        streams['screen_events'] = []
        abs_ok = platform.name == 'android' and quat is not None  # Chrome Android: AbsoluteOrientationSensor
        streams['quaternion_abs'] = ({**quat, 't': _round_list(t_arr + 0.3, ROUNDING['time']),
                                      'sensor': 'AbsoluteOrientationSensor'} if abs_ok else None)
        streams['orientation_abs'] = ({**ori, 't': _round_list(t_arr + 0.6, ROUNDING['time']), 'absolute': True}
                                      if abs_ok else None)
        streams['magnetometer'] = None  # behind a browser flag: practically never exposed
        quality['n_quaternion_abs'] = n if abs_ok else 0
        quality['n_orientation_abs'] = n if abs_ok else 0
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
        'outcome': outcome,
        'timing': timing, 'segmentation': seg_out, 'detected': detected, 'quality': quality,
        'screen': {'type_start': s0[0], 'angle_start': s0[1], 'type_end': s1[0], 'angle_end': s1[1]},
        'streams': streams,
    }
    truth = TrialTruth(
        trial_id=record['trial_id'], hand=hand, orientation=actual, behavior=behavior, go_ms=go_t,
        onset_ms=t_on if math.isfinite(t_on) else None, end_ms=true_end, duration_s=outward_s,
        yaw_net_deg=mv.h * mv.yaw,
    )
    return record, truth


def _return_stop(full: dict[str, Any], end_detected: float, p: dict[str, float],
                 platform: SynthPlatform) -> tuple[int, float | None, float | None]:
    """(number of samples recorded, return_onset_ms, return_end_ms) of a protocol 2.0.0 trial.

    After the end cue the app records at least post_roll_ms, then until a return movement (smoothed |w| >
    45 deg/s or |a| > 1.5 m/s^2 held 100 ms: its first sample = return_onset_ms) followed by stillness
    (|w| < still_gyro_dps and |a| < still_acc_ms2 held still_hold_ms: its first sample = return_end_ms), or
    return_cap_ms after the end cue (onset and / or end stay null). Smoothing: the online segmenter's EMA
    (smoothing_ms) of |rotationRate| and of |acc| (|accg| - g when the platform gives no acc).
    """
    t = np.asarray(full['t'], dtype=float)
    rr = np.column_stack([np.asarray([np.nan if v is None else v for v in c], dtype=float) for c in full['rr']])
    g_raw = np.nan_to_num(np.linalg.norm(rr, axis=1))
    if platform.acc_available:
        a = np.column_stack([np.asarray([np.nan if v is None else v for v in c], dtype=float) for c in full['acc']])
        a_raw = np.nan_to_num(np.linalg.norm(a, axis=1))
    else:
        ag = np.column_stack([np.asarray([np.nan if v is None else v for v in c], dtype=float) for c in full['accg']])
        a_raw = np.abs(np.nan_to_num(np.linalg.norm(ag, axis=1)) - GRAVITY)
    tau = float(p.get('smoothing_ms', 40.0))
    gs, as_ = np.empty_like(g_raw), np.empty_like(a_raw)
    gs[0], as_[0] = g_raw[0], a_raw[0]
    for k in range(1, t.size):
        c = 1.0 - math.exp(-max(0.0, t[k] - t[k - 1]) / tau) if tau > 0 else 1.0
        gs[k] = gs[k - 1] + c * (g_raw[k] - gs[k - 1])
        as_[k] = as_[k - 1] + c * (a_raw[k] - as_[k - 1])
    min_stop = end_detected + float(p.get('post_roll_ms', 500.0))
    cap = end_detected + float(p.get('return_cap_ms', 3000.0))
    onset = end = None
    run0: float | None = None
    for k in np.flatnonzero(t > end_detected):
        tk = float(t[k])
        if tk >= cap:
            break
        if onset is None:
            moving = gs[k] > RETURN_GYRO_DPS or as_[k] > RETURN_ACC_MS2
            run0 = (run0 if run0 is not None else tk) if moving else None
            if run0 is not None and tk - run0 >= RETURN_SUSTAIN_MS:
                onset, run0 = run0, None
            continue
        still = gs[k] < p['still_gyro_dps'] and as_[k] < p['still_acc_ms2']
        run0 = (run0 if run0 is not None else tk) if still else None
        if run0 is not None and tk - run0 >= p['still_hold_ms'] and tk >= min_stop:
            end = run0
            return int(k) + 1, onset, end
    stop = max(min_stop, cap)  # not observed within return_cap_ms: the recording stops at the cap
    n = next((k + 1 for k in range(t.size) if t[k] >= stop), t.size)
    return n, onset, None


# ───────────────────────────── profiles / sessions / datasets ─────────────────────────────


def make_profile(p: SynthParticipant, created_epoch_ms: float, *, short: bool = False) -> dict[str, Any]:
    """ParticipantProfile. ``short``: the protocol 2.0.0 profile (full name in ``nickname``, age, sex,
    dominant hand, usual phone hand, height): every removed field is written as null (EHI, phone model,
    phone case) or false (arm_issue: the consent covers it), never omitted, as the v2 app does."""
    if short:
        return {
            'participant_id': p.participant_id, 'nickname': f'Synthetic {p.participant_id}',
            'age_years': int(p.age_years), 'sex': p.sex, 'handedness_self': p.handedness,
            'ehi_sf': None, 'ehi_lq': None, 'ehi_class': None,
            'phone_hand': p.dominant if p.handedness != 'ambidextrous' else 'both',
            'height_cm': float(p.height_cm), 'arm_issue': False, 'phone_model': None, 'phone_case': None,
            'created_at': iso_time(created_epoch_ms), 'updated_at': iso_time(created_epoch_ms),
        }
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


#: Still window of a PoseStats (calibration.ts CALIBRATION_LIMITS) and the implicit sign threshold.
_POSE_STILL_ACCG_SD = 0.25
_POSE_STILL_RR_SD = 3.0
_POSE_STILL_MIN_N = 20
_IMPLICIT_MIN_Y = 7.0
UPRIGHT_SIGN_CONTRADICTS_ENGINE = 'upright_sign_contradicts_engine'


def pose_stats(motion: dict[str, Any], t_from: float, t_to: float) -> dict[str, Any]:
    """Port of calibration.poseStats over the motion samples with t_from <= t <= t_to: means and
    POPULATION SDs of accg / acc / rr; still = n >= 20, every accg SD < 0.25 m/s^2 and every rr SD < 3 deg/s."""
    t = np.asarray(motion['t'], dtype=float)
    sel = (t >= t_from) & (t <= t_to)

    def rows(keys: tuple[str, str, str]) -> np.ndarray:
        M = np.column_stack([np.asarray([np.nan if v is None else v for v in motion[k]], dtype=float)[sel] for k in keys])
        return M[np.isfinite(M).all(axis=1)]

    G = rows(('accg_x', 'accg_y', 'accg_z'))
    A = rows(('acc_x', 'acc_y', 'acc_z'))
    W = rows(('rr_alpha', 'rr_beta', 'rr_gamma'))
    ts = t[sel]

    def ms(M: np.ndarray) -> tuple[list[float], list[float]]:
        return ([float(x) + 0.0 for x in M.mean(axis=0)], [float(x) for x in M.std(axis=0)]) if len(M) else ([0.0] * 3, [0.0] * 3)

    gm, gsd = ms(G)
    am, _ = ms(A)
    wm, wsd = ms(W)
    still = len(G) >= _POSE_STILL_MIN_N and all(s < _POSE_STILL_ACCG_SD for s in gsd) and (
        len(W) == 0 or all(s < _POSE_STILL_RR_SD for s in wsd))
    euler = (lambda v: {'alpha': v[0], 'beta': v[1], 'gamma': v[2]})
    return {
        'n': int(len(G)), 'duration_ms': float(ts.max() - ts.min()) if ts.size > 1 else 0.0,
        'accg_mean': gm, 'accg_sd': gsd, 'acc_mean': am if len(A) else None,
        'rr_mean': euler(wm) if len(W) else None, 'rr_sd': euler(wsd) if len(W) else None, 'still': bool(still),
    }


def implicit_calibration(trials: list[dict[str, Any]], *, engine: str | None, performed_at: str) -> dict[str, Any]:
    """The app's implicit_v1 calibration (protocol 2.0.0 spec, section 5) from a session's trials.

    The first complete upright_toward trial (in trial order) whose pre-GO window (still_ms - still_hold_ms ...
    go_ms) is still gives ``upright`` = its PoseStats; |accg_y| > 7 sets ``gravity_sign`` = sign(y) and
    ``upright_check`` 'ok' when it matches the engine expectation (webkit -1, blink / gecko +1; any sign for
    an unknown engine), else 'inverted' + note 'upright_sign_contradicts_engine'; |y| <= 7 gives 'unclear';
    ``gravity_magnitude`` = |mean accg| when within 8.6-11.0. No flat window, no gyro bias (a hand-held mean
    is tremor), no axis_check (the analysis infers the mapping from the closure of the trials).
    """
    cal: dict[str, Any] = {
        'method': 'implicit_v1', 'performed_at': performed_at, 'flat': None, 'upright': None, 'gravity_sign': None,
        'gravity_magnitude': None, 'upright_check': None, 'gyro_bias_dps': None, 'notes': [], 'axis_check': None,
    }
    exp = engine_gravity_sign(engine)
    for t in sorted(trials, key=lambda r: (r['trial_index'], r['attempt'])):
        tm = t['timing']
        if (t['condition'].get('orientation') != 'upright_toward' or t['outcome']['status'] != 'complete'
                or tm.get('still_ms') is None or tm.get('go_ms') is None):
            continue
        hold = float(((t.get('segmentation') or {}).get('params') or {}).get('still_hold_ms',
                                                                           DEFAULT_SEGMENTATION_PARAMS['still_hold_ms']))
        ps = pose_stats(t['streams']['motion'], float(tm['still_ms']) - hold, float(tm['go_ms']))
        if not ps['still']:
            continue
        cal['upright'] = ps
        g = np.asarray(ps['accg_mean'], dtype=float)
        mag = float(np.linalg.norm(g))
        cal['gravity_magnitude'] = mag if 8.6 <= mag <= 11.0 else None
        y = float(g[1])
        if abs(y) > _IMPLICIT_MIN_Y:
            sign = 1 if y > 0 else -1
            cal['gravity_sign'] = sign
            if exp is None or exp == sign:
                cal['upright_check'] = 'ok'
            else:
                cal['upright_check'] = 'inverted'
                cal['notes'].append(UPRIGHT_SIGN_CONTRADICTS_ENGINE)
        else:
            cal['upright_check'] = 'unclear'
        break
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
    calibration: str = 'ok',  # ok | no_sign | no_axis_check | missing | implicit
    seed: int = 0,
    spec: dict[str, Any] | None = None,
    noise: float = 1.0,
    preset: str = 'custom',
    profile: str = 'full',
    trials: list[dict[str, Any]] | None = None,
    status: str = 'completed',
    next_trial_index: int | None = None,
) -> dict[str, Any]:
    """SessionRecord. calibration 'implicit' derives the app's implicit_v1 calibration from ``trials``
    (:func:`implicit_calibration`); preset 'street' writes a protocol 2.0.0 session (estimateMinutes of the
    plan, review_mode 'auto', consent 2026-10-v2, protocol.custom null like the app for every preset but
    'custom'); profile 'short' the protocol 2.0.0 profile."""
    n_planned = sum(len(b['trials']) for b in plan)
    street = preset == 'street'
    spec = spec or {}
    custom = None if street else {
        'orientations': list(spec.get('orientations') or ['upright_toward']),
        'speeds': list(spec.get('speeds') or ['natural']),
        'start_positions': list(spec.get('start_positions') or ['natural']),
        'reps': int(spec.get('reps', 1)), 'odd_variants': list(spec.get('odd_variants') or []), 'include_free': False,
    }
    if calibration == 'missing':
        cal = None
    elif calibration == 'implicit':
        cal = implicit_calibration(trials or [], engine=platform.engine, performed_at=iso_time(started_epoch_ms))
    else:
        cal = _calibration(platform, started_epoch_ms + 30_000, calibration in ('ok', 'no_axis_check'), noise,
                           axis_check=calibration != 'no_axis_check')
    rs = {'duration_ms': 3000.0, 'motion_hz': platform.hz, 'orientation_hz': platform.hz,
          'quaternion_hz': platform.hz if platform.quaternion else None, 'motion_dt_ms_median': 1000.0 / platform.hz,
          'motion_dt_ms_p95': 1000.0 / platform.hz + 1.0, 'acc_available': platform.acc_available,
          'accg_available': True, 'rotation_available': True, 'interval_raw': None}
    settings: dict[str, Any] = {'language': 'en', 'audio': True, 'speech': False,
                                'segmentation_params': dict(DEFAULT_SEGMENTATION_PARAMS)}
    if street:
        settings['review_mode'] = 'auto'
    return {
        'record_type': 'session', 'schema_version': SCHEMA_VERSION, 'app_version': APP_VERSION,
        'session_id': session_id, 'participant_id': participant.participant_id, 'study_code': study_code,
        'started_at': iso_time(started_epoch_ms), 'updated_at': iso_time(ended_epoch_ms),
        'ended_at': iso_time(ended_epoch_ms), 'status': status,
        'participant': make_profile(participant, started_epoch_ms - 60_000, short=profile == 'short'),
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
            'preset': preset, 'protocol_version': PROTOCOL_VERSION if street else LEGACY_PROTOCOL_VERSION,
            'seed': int(seed) % (2 ** 32), 'posture': plan[0]['trials'][0]['condition']['posture'] if plan else 'standing',
            'first_hand': plan[0]['hand'] if plan else 'right', 'n_planned_trials': n_planned,
            'estimated_minutes': estimate_minutes(plan) if street else round(n_planned * 9 / 60 * 2) / 2 + 1.0,
            'custom': custom, 'plan': plan,
        },
        'settings': settings,
        'progress': {'next_trial_index': n_planned if next_trial_index is None else int(next_trial_index),
                     'n_attempts': n_attempts, 'n_accepted': n_accepted},
        'consent': {'version': CONSENT_VERSION_V2 if street else '2026-10-v1',
                    'accepted_at': iso_time(started_epoch_ms - 30_000)},
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


#: outcome notes of the protocol 2.0.0 app (app/src/app/sessionFlow.ts).
AUTO_ACCEPT_NOTE = 'auto_accept'
AUTO_REJECT_ORIENTATION_NOTE = 'auto_reject:orientation'
REDO_REQUESTED_NOTE = 'redo_requested'
HAND_UNSURE_NOTE = 'hand_unsure'
#: seconds between two chained trials (automatic advance) and for a block intro (one tap), protocol 2.0.0.
_STREET_TRIAL_GAP_MS = 1500.0
_STREET_BLOCK_GAP_MS = 6000.0


def make_street_session(
    participant: SynthParticipant,
    platform: SynthPlatform,
    *,
    session_id: str,
    study_code: str,
    plan_seed: int,
    rng: np.random.Generator,
    id_rng: np.random.Generator,
    choice_rng: np.random.Generator,
    started_epoch_ms: float,
    noise: float = 1.0,
    failure_rate: float = 0.0,
    calibration: str = 'implicit',
    bonus_fraction: float = 0.5,
    ended_early_fraction: float = 0.0,
    redo_rate: float = 0.0,
    choice_unsure_fraction: float = 0.0,
    posture: str = 'standing',
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, TrialTruth], float]:
    """One protocol 2.0.0 'street' visit as the v2 app records it: (SessionRecord, trials, truth, end epoch).

    Plan = :func:`build_street_plan` (``plan_seed`` is stored as protocol.seed, so the stored seed rebuilds
    the plan). Every trial is a schema 1.2 trial (``simulate_trial(v2=True)``): auto-accepted ('auto_accept')
    when complete and the grip matches; a grip mismatch ('auto_reject:orientation') or a failure
    (``failure_rate``: no movement / too long) is saved not accepted and the same planned trial re-runs.
    The choice trial (tier 0): the participant uses the hand they usually hold the phone in, upright, and
    reports it: condition.hand = that hand, hand_source 'reported', note null (the app's final copy after
    the hand question); "not sure" (``choice_unsure_fraction``): not accepted, participant_flag
    'other_problem', note 'hand_unsure', the planned placeholder hand, never re-run. ``redo_rate``: an
    accepted trial is revoked from the pause sheet ('redo_requested', not accepted) and re-run.
    Bonus blocks (tier 4, optional) only with probability ``bonus_fraction``; with ``ended_early_fraction``
    the participant presses Finish after a random non-optional block: status 'ended_early'. The session
    carries the short profile, review_mode 'auto' and (default) the implicit_v1 calibration the app derives
    from the trials. Random draws: trial seeds / failures from ``rng`` (like :func:`make_dataset`), trial
    ids from ``id_rng``, the participant's choices (bonus, stop, unsure, redo) from ``choice_rng``.
    """
    plan = build_street_plan(plan_seed, posture)
    required = [b for b in plan if b.get('optional') is not True]
    do_bonus = bool(choice_rng.random() < bonus_fraction)
    stop_after = None
    if choice_rng.random() < ended_early_fraction and len(required) > 1:
        stop_after = int(choice_rng.integers(0, len(required) - 1))  # the last block done (never the last one)
    unsure = bool(choice_rng.random() < choice_unsure_fraction)
    blocks = required[:stop_after + 1] if stop_after is not None else (plan if do_bonus else required)
    reported = participant.dominant  # the hand they usually hold the phone in (make_profile short: phone_hand)
    t_epoch = started_epoch_ms + 60_000.0  # after the welcome / profile screens
    perf = 20_000.0
    trials: list[dict[str, Any]] = []
    truth: dict[str, TrialTruth] = {}
    n_acc = 0
    next_index = 0
    for bi, blk in enumerate(blocks):
        if bi:
            perf += _STREET_BLOCK_GAP_MS
            t_epoch += _STREET_BLOCK_GAP_MS
        choice = blk['kind'] == 'choice'
        for pt in blk['trials']:
            attempt = 1
            while True:
                tseed = int(rng.integers(0, 2 ** 31 - 1))
                fail = attempt == 1 and failure_rate > 0 and rng.random() < failure_rate
                behavior = ('no_move' if rng.random() < 0.5 else 'too_slow') if fail else 'normal'
                cond = dict(pt['condition'])
                if choice:
                    cond['hand'] = reported
                ids = {'trial_id': make_id(id_rng), 'session_id': session_id, 'participant_id': participant.participant_id,
                       'study_code': study_code, 'block_index': pt['block_index'], 'trial_index': pt['trial_index'],
                       'trial_in_block': pt['trial_in_block'], 'attempt': attempt}
                rec, tr = simulate_trial(participant, platform, cond, seed=tseed, ids=ids, noise=noise, behavior=behavior,
                                         t0_perf_ms=perf, t0_epoch_ms=t_epoch, v2=True,
                                         free_orientation='upright_toward' if choice else None)
                out = rec['outcome']
                done = False
                if choice and out['status'] == 'complete':
                    done = True  # the hand question follows; a choice trial is never re-run after it
                    if unsure:
                        rec['condition']['hand'] = pt['condition']['hand']  # the plan's placeholder stays
                        out.update(accepted=False, participant_flag='other_problem', note=HAND_UNSURE_NOTE)
                    else:
                        out.update(accepted=True, participant_flag=None, note=None)
                elif out['accepted'] and redo_rate > 0 and choice_rng.random() < redo_rate:
                    out.update(accepted=False, participant_flag=None, note=REDO_REQUESTED_NOTE)
                trials.append(rec)
                truth[rec['trial_id']] = tr
                dur = rec['timing']['stop_ms']
                perf += dur + _STREET_TRIAL_GAP_MS
                t_epoch += dur + _STREET_TRIAL_GAP_MS
                if out['accepted']:
                    n_acc += 1
                if out['accepted'] or done:
                    next_index = pt['trial_index'] + 1
                    break
                attempt += 1
    status = 'ended_early' if stop_after is not None else 'completed'
    sess = make_session(participant, platform, plan, session_id=session_id, study_code=study_code,
                        started_epoch_ms=started_epoch_ms, n_attempts=len(trials), n_accepted=n_acc,
                        ended_epoch_ms=t_epoch, calibration=calibration, seed=plan_seed, noise=noise, preset='street',
                        profile='short', trials=trials, status=status, next_trial_index=next_index)
    return sess, trials, truth, t_epoch


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
    calibration: str | None = None,
    bonus_fraction: float = 0.5,
    ended_early_fraction: float = 0.0,
    redo_rate: float = 0.0,
    choice_unsure_fraction: float = 0.0,
) -> SynthDataset:
    """A complete multi-participant dataset (sessions + every attempt, failures redone once).

    platforms alternate per participant; ``platform_overrides`` {'android': {'quantize': True}, ...}
    changes SynthPlatform fields per platform. calibration: 'ok' | 'no_sign' | 'no_axis_check' |
    'missing' | 'implicit' for every session (``calibration_missing_fraction`` drops it for a random
    share); None = 'implicit' for preset 'street', else 'ok'.
    preset 'street' (protocol 2.0.0, see :func:`make_street_session`): the street plan of the session seed,
    the short profile, schema 1.2 trials (v2) with auto-accept, the choice trial reported; the street-only
    options draw per participant: ``bonus_fraction`` do the 4 optional bonus blocks, ``ended_early_fraction``
    stop after a random non-optional block (status 'ended_early'), ``choice_unsure_fraction`` answer the
    hand question "not sure"; ``redo_rate`` per accepted trial: "Redo last movement" from the pause sheet.
    """
    rng = np.random.default_rng(seed)
    street = preset == 'street'
    spec = None if street else (PRESETS[preset] if isinstance(preset, str) else preset)
    if calibration is None:
        calibration = 'implicit' if street else 'ok'
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
        if street:
            sess, trials, truth, t_epoch = make_street_session(
                part, plat, session_id=make_id(prng), study_code=study_code, plan_seed=(seed * 1000 + k) % 2 ** 32,
                rng=rng, id_rng=prng, choice_rng=np.random.default_rng([seed, k, 2]), started_epoch_ms=epoch + 60_000,
                noise=noise, failure_rate=failure_rate,
                calibration='missing' if prng.random() < calibration_missing_fraction else calibration,
                bonus_fraction=bonus_fraction, ended_early_fraction=ended_early_fraction, redo_rate=redo_rate,
                choice_unsure_fraction=choice_unsure_fraction)
            ds.participants.append(sess['participant'])
            ds.sessions.append(sess)
            ds.trials.extend(trials)
            ds.truth.update(truth)
            ds.sim_participants.append(part)
            epoch = t_epoch + 3_600_000
            continue
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
