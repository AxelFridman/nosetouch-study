"""Frames, units, sign conventions and the versioned normalisation step (``norm_v2``).

This file mirrors the TypeScript contract (app/src/core/types.ts, constants.ts, protocol.ts,
orientationDetect.ts and the simulator docs in app/src/sim/model.ts). When one of those changes,
update the copy here; ``tests/test_conventions.py`` parses constants.ts and fails on drift.
The rules implemented here are specified in docs/03_SENSORS_AND_PLATFORMS.md section 5 and
docs/05_MODELING_PLAN.md section 2.1; ``norm_v2`` adds the implicit-calibration rules of protocol
2.0.0 (no table step, no tilt step): the sign consensus of the engine prior and the trial votes, the
rotationRate mapping inferred from the gyro-gravity closure of the trials, and no gyro-bias subtraction
without a trusted (flat, still) calibration bias.

Frames
------
Device frame D (W3C, fixed to the device, NOT to the current screen rotation):
    x -> right edge of the screen, y -> top edge, z -> out of the screen (towards its viewer).
Earth frame E (W3C DeviceOrientation): x = East, y = North, z = Up.
Body frame B (simulator + mirror derivation): x = participant's right, y = up, z = backward.

Units in records: time ms, acceleration m/s^2, rotation rate deg/s, angles deg. Stream times are
relative to ``timing.t0_perf_ms`` of the trial.

Rotation rate
-------------
``rotationRate.alpha`` is the rate about device X, ``beta`` about Y and ``gamma`` about Z (W3C CRD
12 Feb 2025; MDN's "alpha = Z" text is outdated), so the device-frame angular velocity is
``[alpha, beta, gamma]`` (deg/s, right-hand rule). ``deviceorientation`` angles use a DIFFERENT
convention (intrinsic Z-X'-Y'' Euler angles) and are metadata only. Sessions whose calibration tilt
check (``calibration.axis_check.verdict``) found another mapping are remapped (``RR_MAPPINGS``).
Sessions without a definite tilt check (implicit calibration, schema 1.0) get the mapping whose
gyro-propagated gravity closes best over the session's trials, when that is decisive
(:func:`infer_rate_mapping`); otherwise W3C deg/s is assumed and flagged ``axis_unverified``.

Gravity sign
------------
The W3C convention reports ``accelerationIncludingGravity`` as the reaction to gravity: a phone lying
face-up at rest reads z ~ +9.81 (the vector points UP). WebKit (every iOS browser) reports both
``acceleration`` and ``accelerationIncludingGravity`` negated on all axes; rotationRate signs agree.
Both acceleration fields are therefore multiplied by the session's resolved sign.

A wrong sign negates every polar vector (a, f, g) but not the angular velocity: that is a parity
transformation, it flips the sign of every chiral feature and silently turns left into right. The
sign is resolved explicitly (guided calibration -> consensus of the engine prior and the trial votes ->
trial vote -> engine expectation -> implicit hold -> +1, see :func:`resolve_gravity_sign`;
``order=('engine', 'vote')`` puts the engine before the vote). A calibrated sign that the trial votes
clearly contradict (a flat step done face-down) is overridden by the votes. Only an undisputed guided
calibration and an engine prior confirmed by the votes (source ``engine_vote``) carry no warning.

Mirror (left <-> right hand, any grip): acc (x, y, z) -> (-x, y, z); gyro (x, y, z) -> (x, -y, -z),
i.e. rotationRate {alpha kept, beta and gamma negated}; top_left_* <-> top_right_*.

Screen flip (screen toward <-> away, same hand): the phone turned 180 deg about its own y axis,
F = diag(-1, 1, -1), a proper rotation: acc, accg AND gyro (x, y, z) -> (-x, y, -z);
``GRIP_MATRICES[o_away] = GRIP_MATRICES[o_toward] @ F``; *_toward <-> *_away (:func:`flip_device_vectors`).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ───────────────────────────── units ─────────────────────────────

GRAVITY = 9.80665  # m/s^2, same constant as constants.ts
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
MS_PER_S = 1000.0
#: Version of the normalisation rules, stored with every feature row (``qc_norm_version``).
#: norm_v1 (motionlab 0.2): calibration -> vote -> engine sign; axis_check mapping or W3C assumed; gyro bias
#: from the calibration, else the 400 ms pre-GO mean. norm_v2 (motionlab 0.3, protocol 2.0.0): the engine
#: prior confirmed by the trial votes is a sign source of its own ('engine_vote', no warning; an implicit
#: calibration is not a measured sign); sessions without a definite axis_check get the closure-inferred
#: rotationRate mapping; the gyro bias is subtracted only when a still flat calibration measured it.
NORM_VERSION = 'norm_v2'

#: Copy of SCHEMA_VERSION in constants.ts: the record schema version the current app writes (and synth.py
#: writes). Records of every 1.x version load; 1.2.0 only added optional fields.
SCHEMA_VERSION = '1.2.0'
#: Copy of PROTOCOL_VERSION in constants.ts (2.0.0: the 'street' preset, no screen-away grips in any preset).
APP_PROTOCOL_VERSION = '2.0.0'


def ms_to_s(t_ms: Any) -> Any:
    """Milliseconds -> seconds (scalars or arrays)."""
    return np.asarray(t_ms, dtype=float) / MS_PER_S


def deg_to_rad(x: Any) -> Any:
    return np.asarray(x, dtype=float) * DEG2RAD


def rad_to_deg(x: Any) -> Any:
    return np.asarray(x, dtype=float) * RAD2DEG


def g_to_ms2(x: Any) -> Any:
    """Acceleration in units of g -> m/s^2."""
    return np.asarray(x, dtype=float) * GRAVITY


# ───────────────────────────── enumerations / tables (copy of constants.ts) ─────────────────────────────

HANDS: tuple[str, ...] = ('left', 'right')
SPEEDS: tuple[str, ...] = ('natural', 'fast', 'slow')
START_POSITIONS: tuple[str, ...] = ('natural', 'shoulder', 'center')
VARIANTS: tuple[str, ...] = (
    'standard', 'curved', 'pause', 'twist', 'elbow_high', 'bent_start', 'lean_in', 'chin', 'forehead', 'jerky',
    'tilted_touch',  # schema 1.2: the top edge of the phone touches the nose, tilted as it arrives
)
ODD_VARIANTS: tuple[str, ...] = VARIANTS[1:]
#: Copy of the ProtocolPreset enum ('street' = the protocol 2.0.0 participant visit, the default).
PROTOCOL_PRESETS: tuple[str, ...] = ('quick', 'standard', 'full', 'custom', 'street')
PHONE_ROTATIONS: tuple[str, ...] = ('upright', 'upside_down', 'top_left', 'top_right', 'flat')
ENGINES: tuple[str, ...] = ('webkit', 'blink', 'gecko', 'unknown')

#: OrientationId -> (rotation, screen facing, mirror orientation). Copy of ORIENTATIONS in constants.ts.
ORIENTATIONS: dict[str, dict[str, str | None]] = {
    'upright_toward': {'rotation': 'upright', 'screen': 'toward', 'mirror': 'upright_toward'},
    'upright_away': {'rotation': 'upright', 'screen': 'away', 'mirror': 'upright_away'},
    'top_left_toward': {'rotation': 'top_left', 'screen': 'toward', 'mirror': 'top_right_toward'},
    'top_right_toward': {'rotation': 'top_right', 'screen': 'toward', 'mirror': 'top_left_toward'},
    'upside_down_toward': {'rotation': 'upside_down', 'screen': 'toward', 'mirror': 'upside_down_toward'},
    'upside_down_away': {'rotation': 'upside_down', 'screen': 'away', 'mirror': 'upside_down_away'},
    'top_left_away': {'rotation': 'top_left', 'screen': 'away', 'mirror': 'top_right_away'},
    'top_right_away': {'rotation': 'top_right', 'screen': 'away', 'mirror': 'top_left_away'},
    'flat_up': {'rotation': 'flat', 'screen': 'up', 'mirror': 'flat_up'},
    'free': {'rotation': None, 'screen': None, 'mirror': 'free'},
}
ALL_ORIENTATIONS: tuple[str, ...] = tuple(ORIENTATIONS)
FIXED_ORIENTATIONS: tuple[str, ...] = tuple(o for o in ORIENTATIONS if o != 'free')
#: Copy of COLLECTED_ORIENTATIONS in constants.ts: what protocol 2.0.0 asks participants for (every id but
#: the screen-away grips; those come only from protocol 1.0.0 records or from synthesis).
COLLECTED_ORIENTATIONS: tuple[str, ...] = tuple(o for o, v in ORIENTATIONS.items() if v['screen'] != 'away')

#: Detected PhoneRotation under left/right mirroring (top_left <-> top_right, others unchanged).
PHONE_ROTATION_MIRROR: dict[str, str] = {
    'upright': 'upright', 'upside_down': 'upside_down', 'top_left': 'top_right', 'top_right': 'top_left',
    'flat': 'flat', 'unknown': 'unknown',
}


def mirror_orientation(o: str) -> str:
    """Orientation obtained by mirroring the body across its sagittal plane (left <-> right hand)."""
    return str(ORIENTATIONS[o]['mirror'])


def mirror_hand(h: str) -> str:
    return 'left' if h == 'right' else 'right'


def mirror_pair(o: str) -> str:
    """Name shared by an orientation and its mirror image (top_left_x / top_right_x -> top_side_x).

    Leave-one-orientation-out holds mirror pairs out together: mirroring one creates the other.
    """
    m = mirror_orientation(o)
    if m == o:
        return o
    return o.replace('top_left', 'top_side').replace('top_right', 'top_side')


#: Copy of DEFAULT_SEGMENTATION_PARAMS (constants.ts, protocol 2.0.0). Real trials store their own resolved
#: params in ``segmentation.params``; this copy is used by the synthetic generator and as a fallback.
#: Protocol 1.0.0 values (records up to app 0.1.0 store them): still 15 / 0.35 / 600, foreperiod
#: 500-1200, end 20 / 0.45 / 300, max_wait 6000, max_movement 8000, post_roll 1000, no return_cap_ms.
DEFAULT_SEGMENTATION_PARAMS: dict[str, float] = {
    'smoothing_ms': 40,
    'gravity_tau_ms': 600,
    'still_gyro_dps': 20,
    'still_acc_ms2': 0.5,
    'still_hold_ms': 400,
    'foreperiod_min_ms': 400,
    'foreperiod_max_ms': 900,
    'onset_k_sd': 6,
    'onset_gyro_min_dps': 30,
    'onset_acc_min_ms2': 0.6,
    'onset_sustain_ms': 60,
    'end_gyro_dps': 30,
    'end_acc_ms2': 0.6,
    'end_sustain_ms': 250,
    'min_movement_ms': 150,
    'max_wait_ms': 3000,
    'max_movement_ms': 6000,
    'post_roll_ms': 500,
    'return_cap_ms': 3000,
}
#: Optional SegmentationParams keys (schema 1.2): read by the app's trial controller, not by online_v1.
OPTIONAL_SEGMENTATION_PARAM_KEYS: tuple[str, ...] = ('return_cap_ms',)
#: The keys every stored SegmentationParams has (schema 1.0+) and the online_v1 segmenter reads.
SEGMENTATION_PARAM_KEYS: tuple[str, ...] = tuple(
    k for k in DEFAULT_SEGMENTATION_PARAMS if k not in OPTIONAL_SEGMENTATION_PARAM_KEYS
)

#: Serialisation rounding of the app (ROUNDING in constants.ts), decimals.
ROUNDING: dict[str, int] = {'time': 2, 'acc': 4, 'rr': 3, 'angle': 3, 'quat': 6}

# Copy of protocol.ts SPEED_OVERRIDES / VARIANT_OVERRIDES / TUNE_DIRECTION (protocol 2.0.0; a drift test
# parses protocol.ts). 1.0.0 had slow onset floors of 15 deg/s and 0.3 m/s^2, at or below its still
# thresholds; 2.0.0 keeps them above the still thresholds (20 / 0.5).
_TUNE_DIRECTION = {
    'onset_gyro_min_dps': 'min', 'onset_acc_min_ms2': 'min', 'onset_sustain_ms': 'min', 'end_gyro_dps': 'min',
    'end_acc_ms2': 'min', 'end_sustain_ms': 'max', 'max_movement_ms': 'max',
}
_SPEED_OVERRIDES: dict[str, dict[str, float]] = {
    'natural': {},
    'fast': {'onset_sustain_ms': 40},
    'slow': {'onset_gyro_min_dps': 25, 'onset_acc_min_ms2': 0.55, 'end_gyro_dps': 10, 'end_acc_ms2': 0.25,
             'end_sustain_ms': 450, 'max_movement_ms': 12000},
}
_VARIANT_OVERRIDES: dict[str, dict[str, float]] = {
    'pause': {'end_sustain_ms': 1800, 'max_movement_ms': 12000},
    'jerky': {'end_sustain_ms': 600},
}


def resolve_params(base: Mapping[str, float], condition: Mapping[str, Any]) -> dict[str, float]:
    """Port of protocol.resolveParams (per-speed / per-variant overrides of the segmentation params).

    Used only to synthesise realistic records: real trials already carry their resolved params.
    An override only moves a parameter in the direction it exists for (min for thresholds and the
    onset sustain, max for end_sustain_ms and max_movement_ms), speed first, then variant.
    """
    p = dict(base)
    for ov in (_SPEED_OVERRIDES.get(condition.get('speed'), {}), _VARIANT_OVERRIDES.get(condition.get('variant'), {})):
        for k, v in ov.items():
            p[k] = min(p[k], v) if _TUNE_DIRECTION[k] == 'min' else max(p[k], v)
    return p


# ───────────────────────────── grips (copy of app/src/sim/model.ts docs) ─────────────────────────────

def _cols(c0: tuple[float, float, float], c1: tuple[float, float, float], c2: tuple[float, float, float]) -> np.ndarray:
    return np.array([c0, c1, c2], dtype=float).T


#: R_BD(o): columns = device axes expressed in body coordinates, for the phone at rest at the start.
GRIP_MATRICES: dict[str, np.ndarray] = {
    'upright_toward': np.eye(3),
    'upright_away': _cols((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    'top_left_toward': _cols((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    'top_right_toward': _cols((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    'upside_down_toward': _cols((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    'upside_down_away': _cols((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    'top_left_away': _cols((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
    'top_right_away': _cols((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    'flat_up': _cols((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}

#: Reflection used for left/right mirroring, both in the body frame (M) and the device frame (S).
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def expected_up_direction(orientation: str) -> np.ndarray | None:
    """Unit 'up' direction in the device frame for an instructed orientation at rest (W3C sign).

    This is the direction a W3C ``accelerationIncludingGravity`` vector points to when the phone is held
    as instructed (second row of R_BD). None for 'free'.
    """
    R = GRIP_MATRICES.get(orientation)
    if R is None:
        return None
    return R[1].copy()


def mirror_device_vectors(acc: Any = None, gyro: Any = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Left/right mirror of device-frame vectors (n,3): polar acc -> S·acc, pseudovector gyro -> -S·gyro."""
    a = None if acc is None else np.asarray(acc, dtype=float) * np.array([-1.0, 1.0, 1.0])
    w = None if gyro is None else np.asarray(gyro, dtype=float) * np.array([1.0, -1.0, -1.0])
    return a, w


#: Screen flip: the phone turned 180 deg about its own y axis (top edge kept, screen toward <-> away).
#: GRIP_MATRICES[o_away] = GRIP_MATRICES[o_toward] @ SCREEN_FLIP for every rotation (a test checks it).
SCREEN_FLIP = np.diag([-1.0, 1.0, -1.0])

#: Orientation label under the screen flip, both directions. 'free' stays 'free' (any grip); 'flat_up'
#: has no counterpart in the enum (screen down), so it cannot be flipped.
SCREEN_FLIP_ORIENTATION: dict[str, str] = {
    **{o: o.replace('_toward', '_away') for o in ORIENTATIONS if o.endswith('_toward')},
    **{o: o.replace('_away', '_toward') for o in ORIENTATIONS if o.endswith('_away')},
    'free': 'free',
}


def screen_flip_orientation(o: str) -> str:
    """*_toward -> *_away and back ('free' -> 'free'); ValueError for 'flat_up' and unknown ids."""
    try:
        return SCREEN_FLIP_ORIENTATION[o]
    except KeyError:
        raise ValueError(f'orientation {o!r} has no screen-flipped counterpart') from None


def flip_device_vectors(acc: Any = None, gyro: Any = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Screen flip of device-frame vectors (n,3): F·v for BOTH acc and gyro (F = diag(-1, 1, -1)).

    F is a proper rotation, so polar vectors (acc, accg, gravity, magnetic field) and the angular velocity
    transform alike (unlike the mirror, where w' = -S·w). F is its own inverse and commutes with the
    mirror (both are diagonal).
    """
    f = np.diag(SCREEN_FLIP)
    a = None if acc is None else np.asarray(acc, dtype=float) * f
    w = None if gyro is None else np.asarray(gyro, dtype=float) * f
    return a, w


# ───────────────────────────── rotation rate mapping ─────────────────────────────

#: axis_check verdict -> (sign, rotationRate column [alpha, beta, gamma] giving device x, y, z).
RR_MAPPINGS: dict[str, tuple[int, tuple[int, int, int]]] = {
    'w3c': (1, (0, 1, 2)),
    'w3c_inverted': (-1, (0, 1, 2)),
    'legacy': (1, (1, 2, 0)),  # (beta, gamma, alpha) = (x, y, z): alpha about Z
    'legacy_inverted': (-1, (1, 2, 0)),
}
RR_FIELDS: tuple[str, str, str] = ('rr_alpha', 'rr_beta', 'rr_gamma')


def _rr_mapping(verdict: str | None) -> tuple[int, tuple[int, int, int]]:
    return RR_MAPPINGS.get(verdict or 'w3c', RR_MAPPINGS['w3c'])


def rotation_rate_to_device(rr: Any, verdict: str | None = None, units: str | None = None) -> np.ndarray:
    """rotationRate columns [alpha, beta, gamma] (..., 3) -> device angular velocity [x, y, z] in deg/s.

    verdict: ``calibration.axis_check.verdict``; None / 'unclear' -> W3C ([alpha, beta, gamma] = [x, y, z]).
    units: 'rad_per_s' converts to deg/s.
    """
    sign, cols = _rr_mapping(verdict)
    k = float(sign) * (RAD2DEG if units == 'rad_per_s' else 1.0)
    return np.asarray(rr, dtype=float)[..., list(cols)] * k


def device_to_rotation_rate(w: Any, verdict: str | None = None, units: str | None = None) -> np.ndarray:
    """Inverse of :func:`rotation_rate_to_device`: device [x, y, z] (deg/s) -> raw [alpha, beta, gamma]."""
    sign, cols = _rr_mapping(verdict)
    k = float(sign) * (RAD2DEG if units == 'rad_per_s' else 1.0)
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)
    for j, c in enumerate(cols):
        out[..., c] = w[..., j] / k
    return out


def rr_fields_for_device_axes(verdict: str | None = None) -> tuple[str, str, str]:
    """Names of the raw rotationRate fields that hold the device x, y and z rates (sign aside)."""
    _, cols = _rr_mapping(verdict)
    return tuple(RR_FIELDS[c] for c in cols)  # type: ignore[return-value]


@dataclass(frozen=True)
class RateMapping:
    """Effective rotationRate mapping of a session.

    verdict: mapping applied (a key of ``RR_MAPPINGS``); units: 'deg_per_s' or 'rad_per_s';
    verified: True when the calibration tilt check or a decisive closure check (:func:`infer_rate_mapping`)
    established it; otherwise W3C is assumed and the session is flagged ``axis_unverified`` (spec: docs/03
    section 5.1, S4 in docs/02 section 12).
    source: 'calibration' (axis_check), 'closure' (inferred from the trials) or 'assumed' (W3C deg/s).
    closure_deg / runner_up / runner_up_deg / n_trials / medians: the closure check's numbers (median
    closure of the best and of the second-best candidate, how many trials, every candidate's median as
    ('verdict/units', deg) pairs, best first); None / 0 / () when it did not run.
    """

    verdict: str = 'w3c'
    units: str = 'deg_per_s'
    verified: bool = False
    detail: str = ''
    source: str = 'assumed'
    closure_deg: float | None = None
    runner_up: str | None = None
    runner_up_deg: float | None = None
    n_trials: int = 0
    medians: tuple[tuple[str, float], ...] = ()

    @property
    def key(self) -> str:
        return rate_candidate_key(self.verdict, self.units)


def rate_mapping_from_calibration(calibration: Mapping[str, Any] | None) -> RateMapping:
    """RateMapping from ``session.calibration.axis_check`` (absent in schema 1.0 records, null in implicit
    calibrations).

    Units are only trusted together with a definite verdict: all current browsers report deg/s, and a
    'rad_per_s' reading next to an 'unclear' verdict is more likely a failed gesture than an old browser.
    """
    ac = (calibration or {}).get('axis_check') or None
    if not isinstance(ac, Mapping):
        why = ('implicit calibration' if calibration_method(calibration) == 'implicit_v1'
               else 'schema 1.0 or skipped')
        return RateMapping(detail=f'axis_unverified: no axis_check ({why}); W3C assumed')
    verdict = ac.get('verdict')
    units = ac.get('units')
    if verdict in RR_MAPPINGS:
        u = 'rad_per_s' if units == 'rad_per_s' else 'deg_per_s'
        detail = '' if (verdict == 'w3c' and u == 'deg_per_s') else f'axis_check verdict {verdict}, units {u}'
        return RateMapping(verdict, u, True, detail, source='calibration')
    return RateMapping(detail=f'axis_unverified: axis_check verdict {verdict!r} (units {units!r}); W3C deg/s assumed')


# ───────────────────────────── rotation rate mapping from the trials (closure) ─────────────────────────────

#: Every (verdict, units) a session's rotationRate could follow: the 4 axis mappings, in deg/s and rad/s.
RATE_CANDIDATES: tuple[tuple[str, str], ...] = tuple((v, u) for u in ('deg_per_s', 'rad_per_s') for v in RR_MAPPINGS)
#: Decision thresholds of :func:`infer_rate_mapping` (protocol 2.0.0 spec, section 10).
RATE_CLOSURE_MAX_DEG = 3.0
RATE_CLOSURE_MAX_RATIO = 0.5
RATE_CLOSURE_MIN_TRIALS = 8


def rate_candidate_key(verdict: str, units: str) -> str:
    """'w3c/deg_per_s' etc.: the key of a candidate mapping in closure dicts."""
    return f'{verdict}/{units}'


def infer_rate_mapping(
    closures: Iterable[Mapping[str, float] | None],
    *,
    min_trials: int = RATE_CLOSURE_MIN_TRIALS,
    max_best_deg: float = RATE_CLOSURE_MAX_DEG,
    max_ratio: float = RATE_CLOSURE_MAX_RATIO,
) -> RateMapping:
    """The rotationRate mapping of a session, from the gyro-gravity closure of its trials.

    ``closures``: one dict per trial, {'verdict/units': closure_deg} for every :data:`RATE_CANDIDATES`
    entry (``features.rate_closures``: the rest gravity before the movement, propagated through it with
    the gyro read under that mapping, against the rest gravity at the nose). None entries and trials
    with a missing candidate are skipped. Per candidate the median over the trials is taken; the best
    candidate is VERIFIED (source 'closure') when its median is below ``max_best_deg`` (3 deg), below
    ``max_ratio`` (0.5) times the runner-up's and at least ``min_trials`` (8) trials took part. One trial
    is not enough: on the pilot W3C won only 18 of 22 trials, by a median 1.3 deg, while the session
    medians were 1.3 deg (W3C) against 4.0 deg (legacy). Otherwise W3C deg/s is assumed (every current
    engine reports it, docs/03 section 2.2) and flagged ``axis_unverified``; the numbers stay in the result.
    The closure does not depend on the accelerometer sign (both rest windows flip together).
    """
    keys = [rate_candidate_key(v, u) for v, u in RATE_CANDIDATES]
    rows: list[list[float]] = []
    for c in closures:
        if not c:
            continue
        vals = [c.get(k) for k in keys]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vals):
            rows.append([float(x) for x in vals])  # type: ignore[arg-type]
    n = len(rows)
    if n == 0:
        return RateMapping(detail='axis_unverified: no trial with a rest before and after a movement to check '
                                  'the rotationRate mapping; W3C deg/s assumed')
    med = np.median(np.asarray(rows), axis=0)
    order = [int(i) for i in np.argsort(med, kind='stable')]
    b, r = order[0], order[1]
    best, run = float(med[b]), float(med[r])
    verdict, units = RATE_CANDIDATES[b]
    numbers = dict(closure_deg=best, runner_up=keys[r], runner_up_deg=run, n_trials=n,
                   medians=tuple((keys[i], float(med[i])) for i in order))
    reasons = []
    if n < min_trials:
        reasons.append(f'{n} < {min_trials} trials')
    if not best < max_best_deg:
        reasons.append(f'best {best:.1f} >= {max_best_deg:g} deg')
    if not best < max_ratio * run:
        reasons.append(f'best not below {max_ratio:g} x runner-up')
    if not reasons:
        detail = '' if (verdict, units) == ('w3c', 'deg_per_s') else f'closure check: {verdict}, units {units}'
        return RateMapping(verdict, units, True, detail, source='closure', **numbers)
    return RateMapping(
        detail=(f'axis_unverified: closure check inconclusive (best {keys[b]} {best:.1f} deg, runner-up {keys[r]} '
                f'{run:.1f} deg, {n} trials: {"; ".join(reasons)}); W3C deg/s assumed'),
        **numbers)


# ───────────────────────────── gravity sign ─────────────────────────────

#: Accelerometer sign expected from the browser engine (verified: WebKit inverts both fields).
ENGINE_GRAVITY_SIGN: dict[str, int] = {'webkit': -1, 'blink': 1, 'gecko': 1}

#: Fewest trial votes that may override a calibrated sign (with ``min_agreement`` of them against it).
MIN_OVERRIDE_VOTES = 5

#: CalibrationResult.method (schema 1.2). guided_v1 = the calibration screen (table, upright and tilt
#: steps; absent ``method`` means guided_v1); implicit_v1 = derived by the app from the pre-GO hold of the
#: first still upright trial (protocol 2.0.0: no table, no tilt step).
CALIBRATION_METHODS: tuple[str, ...] = ('guided_v1', 'implicit_v1')


def calibration_method(calibration: Mapping[str, Any] | None) -> str | None:
    """'guided_v1' or 'implicit_v1' of a session's calibration (absent ``method`` = guided_v1, schema 1.2);
    None when the session has no calibration object (or an unknown method)."""
    if not isinstance(calibration, Mapping) or not calibration:
        return None
    m = calibration.get('method')
    if m is None:
        return 'guided_v1'
    return str(m) if m in CALIBRATION_METHODS else None


def gravity_sign_from_calibration(calibration: Mapping[str, Any] | None) -> int | None:
    """+1 / -1 from ``session.calibration.gravity_sign``; None when missing or unknown."""
    if not calibration:
        return None
    s = calibration.get('gravity_sign')
    if s in (1, -1) and not isinstance(s, bool):
        return int(s)
    return None


def session_engine(device: Mapping[str, Any] | None) -> tuple[str | None, str]:
    """(engine, how) of a session's browser: ``device.engine`` when it names one (schema 1.1), else
    'webkit' for platform 'ios' (every iOS browser runs WebKit), else None."""
    d = device or {}
    e = d.get('engine')
    if e in ENGINE_GRAVITY_SIGN:
        return str(e), 'device.engine'
    if d.get('platform') == 'ios':
        return 'webkit', 'device.platform ios'
    return None, ''


def engine_gravity_sign(engine: str | None) -> int | None:
    """Expected accelerometer sign of an engine: webkit -1, blink / gecko +1, unknown None."""
    return ENGINE_GRAVITY_SIGN.get(engine) if engine else None


@dataclass(frozen=True)
class GravitySign:
    """Resolved gravity sign of a session.

    source: 'calibration' (measured by a guided calibration), 'engine_vote' (the engine expectation
    confirmed by a confident trial vote: the protocol 2.0.0 normal case), 'engine' (expectation from
    ``device.engine`` alone), 'inferred' (majority vote of trial baselines against the instructed
    orientation, also when that vote overrode a calibrated sign it clearly contradicts), 'implicit' (only
    the implicit calibration's single upright hold supports it) or 'default' (+1, nothing better
    available: exclude the session from accelerometer-based chiral features).
    ``warning`` is False only for an undisputed guided calibration and for an undisputed 'engine_vote'.
    ``votes_for`` / ``votes_against`` count the trial votes for / against the resolved ``sign``.
    """

    sign: int
    source: str
    warning: bool
    votes_for: int = 0
    votes_against: int = 0
    detail: str = ''
    engine: str | None = None

    @property
    def agreement(self) -> float:
        n = self.votes_for + self.votes_against
        return float('nan') if n == 0 else self.votes_for / n

    @property
    def usable_for_chirality(self) -> bool:
        """False when nothing supports the sign (source 'default'): chiral features are a coin flip."""
        return self.source != 'default'


def gravity_vote(raw_gravity: Any, orientation: str, min_cos: float = math.cos(math.radians(45))) -> int | None:
    """Sign implied by one trial: compares the RAW mean accg of the still baseline with the up direction
    of the instructed orientation. Returns +1 (W3C), -1 (inverted) or None (free / ambiguous / missing)."""
    e = expected_up_direction(orientation)
    if e is None or raw_gravity is None:
        return None
    g = np.asarray(raw_gravity, dtype=float)
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        return None
    n = float(np.linalg.norm(g))
    if n < 0.3 * GRAVITY:  # not a gravity reading (free fall, broken sensor)
        return None
    c = float(g @ e) / n
    if abs(c) < min_cos:
        return None
    return 1 if c > 0 else -1


def _stream_col(m: Mapping[str, Any], k: str) -> np.ndarray:
    v = m.get(k)
    if v is None:
        return np.zeros(0)
    if isinstance(v, np.ndarray):
        return v.astype(float, copy=False)
    return np.asarray([np.nan if x is None else x for x in v], dtype=float)


def trial_gravity_vote(trial: Mapping[str, Any]) -> int | None:
    """Vote of one TrialRecord: baseline gravity (segmentation.baseline.gravity) if present, else the mean
    accg of the first 300 ms of the motion stream (the participant is asked to hold still at the start)."""
    cond = trial.get('condition') or {}
    o = cond.get('orientation')
    if not o:
        return None
    seg = trial.get('segmentation') or {}
    base = seg.get('baseline') or {}
    g = base.get('gravity')
    if g is None:
        m = ((trial.get('streams') or {}).get('motion')) or {}
        t = _stream_col(m, 't')
        if t.size == 0:
            return None
        cols = [_stream_col(m, k) for k in ('accg_x', 'accg_y', 'accg_z')]
        if any(c.size != t.size for c in cols):
            return None
        sel = t <= t[0] + 300.0
        G = np.column_stack(cols)[sel]
        G = G[np.all(np.isfinite(G), axis=1)]
        if len(G) < 3:
            return None
        g = G.mean(axis=0)
    return gravity_vote(g, o)


def _confident_vote(votes: Sequence[int], upright_votes: Iterable[int | None] | None, min_votes: int,
                    min_agreement: float) -> tuple[int | None, str]:
    """(sign, how) of the trial votes: the upright-only vote when it is confident (sideways votes confound
    the sign with a left/right grip mix-up: a flipped sign turns top_left into top_right, while it turns
    upright into upside down, which nobody holds by mistake), else the vote of all trials; (None, '')."""
    def major(v: Sequence[int]) -> int | None:
        if len(v) < min_votes:
            return None
        pos = sum(1 for x in v if x > 0)
        m = 1 if pos >= len(v) - pos else -1
        return m if max(pos, len(v) - pos) / len(v) >= min_agreement else None

    u = [int(x) for x in (upright_votes or ()) if x in (1, -1)]
    mu = major(u)
    if mu is not None:
        agree = sum(1 for x in u if x == mu)
        return mu, f'upright trials {agree}/{len(u)}'
    ma = major(votes)
    if ma is not None:
        agree = sum(1 for x in votes if x == ma)
        return ma, f'all trials {agree}/{len(votes)}'
    return None, ''


def resolve_gravity_sign(
    calibration: Mapping[str, Any] | None,
    votes: Iterable[int | None] = (),
    *,
    engine: str | None = None,
    infer: bool = True,
    min_votes: int = 3,
    min_agreement: float = 0.8,
    order: Sequence[str] = ('vote', 'engine'),
    upright_votes: Iterable[int | None] | None = None,
) -> GravitySign:
    """Resolve the gravity sign of a session.

    1. A GUIDED calibration's ``gravity_sign`` (+1/-1), source 'calibration'. Flagged (warning) when
       ``upright_check`` is 'inverted' or when it differs from the engine expectation. When the trial
       votes clearly contradict it (>= ``MIN_OVERRIDE_VOTES`` = 5 votes, >= ``min_agreement`` of them
       against) and ``infer`` is on, the votes win: sign = their majority, source 'inferred', warning,
       ``detail`` 'calibration sign overridden by the trial votes: ...'. A flat step done face-down passes
       every calibration check with the wrong sign, so a confident calibration can still be wrong
       (docs/03 section 4(b)); without ``infer`` it is kept and flagged.
       An IMPLICIT calibration (``method: 'implicit_v1'``) is not a measurement of its own: its sign is the
       vote of one upright trial's pre-GO hold, which is among the trial votes already. It never takes
       this step; a contradiction with the resolved sign is flagged in ``detail``.
    2. Consensus: the engine expectation (webkit -1, blink / gecko +1) and a confident trial vote agree,
       source 'engine_vote', NO warning (unless the implicit sign contradicts them). This is the normal
       case of protocol 2.0.0 sessions. A confident vote is the vote of the upright trials when there are
       >= ``min_votes`` of them (``upright_votes``) with >= ``min_agreement`` agreement, else the vote of all
       trials with the same thresholds (sideways votes confound the sign with a left/right grip mix-up).
    3. then, in ``order`` (default: the trial vote first, then the engine expectation, as docs/03
       section 5.2 specifies: measured evidence beats a prior, e.g. if a WebKit release ever fixed the sign):
       'engine' - the engine expectation alone, source 'engine';
       'vote'   - when ``infer``, the confident trial vote, source 'inferred'.
       Both are flagged; a confident vote that contradicts the engine is named in ``detail``.
    4. an implicit calibration's sign alone (no engine prior, no confident vote), source 'implicit', flagged.
    5. else +1 (source 'default', warning) - exclude the session from accelerometer-based chirality.
    """
    v = [int(x) for x in votes if x in (1, -1)]
    pos = sum(1 for x in v if x > 0)
    neg = len(v) - pos
    exp = engine_gravity_sign(engine)
    method = calibration_method(calibration)
    measured = gravity_sign_from_calibration(calibration)
    implicit = measured if method == 'implicit_v1' else None
    cal = None if method == 'implicit_v1' else measured
    if cal is not None:
        agree = pos if cal > 0 else neg
        against = len(v) - agree
        contradicted = len(v) >= max(min_votes, MIN_OVERRIDE_VOTES) and against / len(v) >= min_agreement
        problems = []
        if contradicted:
            problems.append(f'calibration {cal:+d} contradicts the trial baselines ({against}/{len(v)} votes {-cal:+d})')
        if (calibration or {}).get('upright_check') == 'inverted':
            problems.append("upright_check 'inverted'")
        if exp is not None and exp != cal:
            problems.append(f'calibration differs from the {engine} engine expectation ({exp:+d})')
        if contradicted and infer:
            detail = 'calibration sign overridden by the trial votes: ' + '; '.join(problems)
            if exp is not None and exp != -cal:
                detail += f'; the votes differ from the {engine} engine expectation ({exp:+d})'
            return GravitySign(-cal, 'inferred', True, against, agree, detail, engine)
        return GravitySign(cal, 'calibration', bool(problems), agree, against, '; '.join(problems), engine)

    vote_sign, vote_how = _confident_vote(v, upright_votes, min_votes, min_agreement)

    def counts(sign: int) -> tuple[int, int]:
        agree = pos if sign > 0 else neg
        return agree, len(v) - agree

    def implicit_note(sign: int) -> str:
        if implicit is None or implicit == sign:
            return ''
        return f'; the implicit calibration sign ({implicit:+d}, first still upright hold) contradicts it'

    head = 'implicit calibration' if method == 'implicit_v1' else 'no calibrated sign'
    if exp is not None and vote_sign == exp:
        agree, against = counts(exp)
        extra = implicit_note(exp)
        return GravitySign(exp, 'engine_vote', bool(extra), agree, against,
                           f'{head}; {engine} engine expectation {exp:+d} confirmed by the trial votes ({vote_how})'
                           + extra, engine)
    for step in order:
        if step == 'engine' and exp is not None:
            agree, against = counts(exp)
            note = f'{head}; {engine} engine expectation {exp:+d}'
            if vote_sign is not None and vote_sign != exp:
                note += f'; CONTRADICTED by the trial votes ({vote_sign:+d}, {vote_how})'
            else:
                note += f'; no confident trial vote ({pos} x +1, {neg} x -1)'
            return GravitySign(exp, 'engine', True, agree, against, note + implicit_note(exp), engine)
        if step == 'vote' and infer and vote_sign is not None:
            agree, against = counts(vote_sign)
            note = f'{head}; inferred from trial baselines vs instructed orientation ({vote_how})'
            if exp is not None and exp != vote_sign:
                note += f'; differs from the {engine} engine expectation ({exp:+d})'
            return GravitySign(vote_sign, 'inferred', True, agree, against, note + implicit_note(vote_sign), engine)
    if implicit is not None:
        agree, against = counts(implicit)
        return GravitySign(implicit, 'implicit', True, agree, against,
                           'no known engine and no confident trial vote; sign of the implicit calibration '
                           '(one still upright hold)', engine)
    return GravitySign(1, 'default', True, pos, neg,
                       'no calibration, no known engine and no confident trial vote; assumed +1 (W3C)', engine)


def normalize_acceleration(a: Any, sign: int) -> np.ndarray:
    """Multiply acc / accg columns by the session gravity sign so that they follow the W3C convention."""
    if sign not in (1, -1):
        raise ValueError(f'gravity sign must be +1 or -1, got {sign!r}')
    return np.asarray(a, dtype=float) * float(sign)


# ───────────────────────────── norm_v2: one session's normalisation ─────────────────────────────

#: Largest gyroscope bias (deg/s, norm of the device-frame vector) accepted from a rest window - the
#: calibration's flat step or (only on request) the 400 ms before GO. A larger mean rate "at rest" is motion
#: (a hand on the phone, a phone still settling), not bias; subtracting it would bias every integrated angle.
GYRO_BIAS_MAX_DPS = 3.0


def calibration_gyro_bias(calibration: Mapping[str, Any] | None,
                          rates: RateMapping | None = None) -> tuple[float, float, float] | None:
    """``calibration.gyro_bias_dps`` mapped to device x, y, z (deg/s), or None when it is not trusted.

    Accepted only from a flat window the app judged still (``calibration.flat.still`` is true) and with a
    norm below :data:`GYRO_BIAS_MAX_DPS`, the limit of the pre-GO estimate. Records written before the app
    checked this store the mean rate of a MOVING flat window too. Implicit calibrations (protocol 2.0.0)
    have no flat window, so None. With None, ``features.prepare_trial`` subtracts nothing (norm_v2): a
    hand-held pre-GO mean is tremor, not bias (on the pilot, subtracting it worsened the median gravity
    closure from 1.6 to 2.7 deg); ``remove_gyro_bias='pre_go'`` still asks for it explicitly.
    """
    cal = calibration or {}
    flat = cal.get('flat')
    b = cal.get('gyro_bias_dps')
    if not (isinstance(flat, Mapping) and flat.get('still') is True and isinstance(b, Mapping)):
        return None
    raw = [b.get(k) for k in ('alpha', 'beta', 'gamma')]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in raw):
        return None
    r = rates or rate_mapping_from_calibration(cal)
    w = rotation_rate_to_device(raw, r.verdict, r.units)
    if not (np.all(np.isfinite(w)) and float(np.linalg.norm(w)) < GYRO_BIAS_MAX_DPS):
        return None
    return (float(w[0]), float(w[1]), float(w[2]))


@dataclass(frozen=True)
class Normalization:
    """Everything needed to turn a session's raw motion streams into W3C device-frame physics.

    gravity: resolved accelerometer sign; rates: rotationRate mapping applied; gravity_magnitude:
    calibration |accg| (m/s^2) or None; gyro_bias: calibration rest rate mapped to device x, y, z (deg/s)
    or None (not calibrated, or not trusted: :func:`calibration_gyro_bias`; nothing is subtracted then);
    engine: browser engine (or None); calibration_method: 'guided_v1' / 'implicit_v1' / None;
    rate_inference: the closure check of the session's trials (:func:`infer_rate_mapping`) when it ran,
    also when a definite axis_check outranked it (then it is a cross-check).
    """

    gravity: GravitySign = field(default_factory=lambda: GravitySign(1, 'default', True))
    rates: RateMapping = field(default_factory=RateMapping)
    gravity_magnitude: float | None = None
    gyro_bias: tuple[float, float, float] | None = None
    engine: str | None = None
    version: str = NORM_VERSION
    calibration_method: str | None = None
    rate_inference: RateMapping | None = None

    @property
    def sign(self) -> int:
        return self.gravity.sign

    @property
    def rate_check_disagrees(self) -> bool:
        """A definite axis_check and a decisive closure check name different mappings or units."""
        ri = self.rate_inference
        return bool(self.rates.source == 'calibration' and ri is not None and ri.verified
                    and (ri.verdict, ri.units) != (self.rates.verdict, self.rates.units))

    @property
    def flags(self) -> list[str]:
        out = []
        if self.gravity.warning:
            out.append(f'gravity_sign_{self.gravity.source}')
        if not self.rates.verified:
            out.append('axis_unverified')
        elif self.rates.verdict != 'w3c' or self.rates.units != 'deg_per_s':
            out.append(f'rr_remapped_{self.rates.verdict}_{self.rates.units}')
        if self.rate_check_disagrees:
            out.append('rr_closure_disagrees')
        return out

    @classmethod
    def from_sign(cls, sign: int, source: str = 'given') -> 'Normalization':
        """W3C rates, given gravity sign (for plain records without their session)."""
        return cls(gravity=GravitySign(int(sign), source, source != 'given'), rates=RateMapping(verified=True))


def session_normalization(
    session: Mapping[str, Any] | None,
    votes: Iterable[int | None] = (),
    *,
    infer: bool = True,
    order: Sequence[str] = ('vote', 'engine'),
    upright_votes: Iterable[int | None] | None = None,
    rate_inference: RateMapping | None = None,
) -> Normalization:
    """norm_v2 of one SessionRecord (``None`` = unknown session: default sign, W3C assumed).

    upright_votes: the votes of the trials instructed upright / upside down (see :func:`resolve_gravity_sign`).
    rate_inference: the closure check of the session (:func:`infer_rate_mapping`); it sets the mapping when
    the calibration has no definite axis_check and it is decisive, and is kept as a cross-check otherwise.
    The calibration gyro bias is mapped with the mapping finally applied.
    """
    s = session or {}
    cal = s.get('calibration') or None
    engine, how = session_engine(s.get('device'))
    gs = resolve_gravity_sign(cal, votes, engine=engine, infer=infer, order=order, upright_votes=upright_votes)
    if engine and how != 'device.engine':
        gs = GravitySign(gs.sign, gs.source, gs.warning, gs.votes_for, gs.votes_against,
                         (gs.detail + '; ' if gs.detail else '') + f'engine from {how}', engine)
    rates = rate_mapping_from_calibration(cal)
    if rate_inference is not None and not rates.verified:
        rates = rate_inference
    mag = (cal or {}).get('gravity_magnitude')
    mag = float(mag) if isinstance(mag, (int, float)) and 8.0 <= float(mag) <= 11.5 else None
    return Normalization(gs, rates, mag, calibration_gyro_bias(cal, rates), engine,
                         calibration_method=calibration_method(cal), rate_inference=rate_inference)


@dataclass
class DeviceSignals:
    """Motion stream in physical device-frame units with the W3C conventions applied.

    t: (n,) ms since t0 (handler time); acc / accg: (n,3) m/s^2 (sign-normalised; acc excludes gravity);
    gyro: (n,3) device angular velocity [x, y, z] in deg/s. Missing values are NaN.
    """

    t: np.ndarray
    acc: np.ndarray
    accg: np.ndarray
    gyro: np.ndarray
    gravity_sign: int = 1
    rr_verdict: str = 'w3c'
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    @property
    def acc_available(self) -> bool:
        return bool(np.isfinite(self.acc).all(axis=1).any())

    @property
    def gyro_available(self) -> bool:
        return bool(np.isfinite(self.gyro).all(axis=1).any())


def device_signals(
    t: Any, acc: Any, accg: Any, rr: Any, gravity_sign: int = 1, *,
    rr_verdict: str | None = None, rr_units: str | None = None,
) -> DeviceSignals:
    """Build :class:`DeviceSignals` from raw stream arrays (acc/accg raw sign, rr = [alpha, beta, gamma])."""
    verdict = rr_verdict if rr_verdict in RR_MAPPINGS else 'w3c'
    return DeviceSignals(
        t=np.asarray(t, dtype=float),
        acc=normalize_acceleration(acc, gravity_sign),
        accg=normalize_acceleration(accg, gravity_sign),
        gyro=rotation_rate_to_device(rr, verdict, rr_units),
        gravity_sign=int(gravity_sign),
        rr_verdict=verdict,
    )


def normalize_motion(t: Any, acc: Any, accg: Any, rr: Any, norm: Normalization | None) -> DeviceSignals:
    """Raw motion arrays -> :class:`DeviceSignals` under a session's :class:`Normalization` (norm_v2)."""
    n = norm or Normalization()
    sig = device_signals(t, acc, accg, rr, n.sign, rr_verdict=n.rates.verdict, rr_units=n.rates.units)
    sig.notes.extend(n.flags)
    return sig


# ───────────────────────────── phone rotation from gravity (port of orientationDetect.ts) ─────────────────────────────


def rotation_from_gravity(
    accg: Any,
    gravity_sign: int | None,
    screen_hint: str | None,
    max_angle_deg: float = 35.0,
) -> str:
    """PhoneRotation (or 'unknown') from a mean RAW accelerationIncludingGravity vector (rotationFromGravity).

    After sign normalisation the vector points UP: +y upright, -y upside_down, +z flat (screen up),
    +x top_left when the screen faces the participant (top_right when it faces away), -x the opposite.
    The dominant axis (largest |component|, ties x > y > z) must be strictly within ``max_angle_deg``.
    A face-down phone (-z) gives 'unknown'. screen_hint None / 'up' is treated as 'toward'.
    """
    if accg is None:
        return 'unknown'
    v = np.asarray(accg, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        return 'unknown'
    s = -1.0 if gravity_sign == -1 else 1.0
    x, y, z = s * float(v[0]), s * float(v[1]), s * float(v[2])
    n = math.sqrt(x * x + y * y + z * z)
    if not (n > 0) or not math.isfinite(n):
        return 'unknown'
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        axis, comp = 'x', x
    elif ay >= az:
        axis, comp = 'y', y
    else:
        axis, comp = 'z', z
    angle = math.degrees(math.acos(min(1.0, abs(comp) / n)))
    if not angle < max_angle_deg:
        return 'unknown'
    if axis == 'y':
        return 'upright' if comp > 0 else 'upside_down'
    if axis == 'z':
        return 'flat' if comp > 0 else 'unknown'
    away = screen_hint == 'away'
    if comp > 0:
        return 'top_right' if away else 'top_left'
    return 'top_left' if away else 'top_right'


def tilt_change_deg(a: Any, b: Any) -> float | None:
    """Angle (deg) between two gravity vectors (atan2 form, like tiltChangeDeg); None if missing/zero."""
    if a is None or b is None:
        return None
    u, v = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if u.shape != (3,) or v.shape != (3,) or not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return None
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if not (nu > 0 and nv > 0) or not (math.isfinite(nu) and math.isfinite(nv)):
        return None
    c = np.cross(u, v)
    return math.degrees(math.atan2(float(np.linalg.norm(c)), float(u @ v)))
