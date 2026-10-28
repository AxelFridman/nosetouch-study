"""Lane m1 feature catalogue: interpretable, per-trial, label-free features of one movement.

Every feature is a function of ONE trial (no statistic across trials), so it may be computed before the
leave-one-person-out loop (rule 3 of the lane). Each one is documented in :data:`CATALOGUE`: formula, physical
meaning, unit, mirror parity and grip dependence.

Frames (all heading-free: nothing depends on the compass direction the person faces)
------------------------------------------------------------------------------------
The cache gives, per sample, the unit up vector u (gyro-propagated gravity), the angular velocity w and the
linear acceleration a, all in the W3C device frame (x right edge, y top edge, z out of the screen). Lane m1 adds
a *heading-following body frame* built from u and the screen normal e_z:

    h = (e_z - (e_z.u) u) / |...|     the horizontal direction out of the screen (towards the face when the
                                      screen faces the person: every grip of the protocol)
    s = u x h                         the horizontal direction to the person's RIGHT

and projects w and a on (u, h, s):

    yaw_rate   = w.u   rotation about gravity          (cache channel; + = counter-clockwise seen from above)
    roll_rate  = w.h   rotation about the facing axis   (+ = counter-clockwise as seen by the person)
    pitch_rate = w.s   rotation about the lateral axis  (+ = top edge tips towards the person)
    acc_lat    = a.s   lateral acceleration             (+ = towards the person's right)
    acc_fwd    = a.h   acceleration towards the face
    acc_vert   = a.u   vertical acceleration            (cache channel)

Mirror parity (left/right mirror: a -> S a, w -> -S w, S = diag(-1, 1, 1); the label flips): u and h are polar
vectors and s = u x h is axial, so yaw_rate, roll_rate and acc_lat are ODD (change sign: they can carry the
hand) while pitch_rate, acc_fwd and acc_vert are EVEN (identical for a movement and its mirror image: they cannot
tell the hand on their own). The three grips of the protocol (upright, top-left, top-right; screen towards the
person) differ by a rotation about e_z, which leaves h, s and every projection unchanged: all body-frame channels
are grip-invariant. Under the screen-away flip (a 180-degree turn about y) h reverses, so roll_rate and acc_lat
change sign (yaw_rate does not): they assume the screen faces the person, as in every recorded trial.

Grip dependence of each feature (the ``grip`` text of :data:`CATALOGUE`) is generated from two declared facts:
every feature is unchanged by the three screen-toward grips (rotations about the screen normal), and its
``flips`` attribute says whether it changes sign under the screen-away flip
(``standardize.screen_flip_channels``). ``test_m1.test_grip_labels_match_the_screen_flip`` checks the attribute
on real trials.

Windows: the time-normalised approach (``d.approach``, 64 samples from onset to end; integrals use the trial's
own duration) and the fixed window (``d.fixed``, 60 Hz, -0.5 s .. +2.5 s around the onset; samples after the end
cue are masked and do not enter an integral).

De-drifted lateral velocity (``latvel_dd_q*``, family ``dedrift``): the linear acceleration is a = f - g with the
gyro-propagated gravity g, so a tilt error of g leaks into a.s as a slowly varying bias that the integral
accumulates (integration drift). The de-drifted variant subtracts the trial's own mean a.s over the still hold
[onset - 0.5 s, onset - 0.2 s] (the window of the negative control) before integrating: per trial, no fitting.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stage2.standardize import CHANNELS

FS_HZ = 60.0
FIXED_ONSET_INDEX = 30                     # d.fixed starts 0.5 s before the onset at 60 Hz
FIXED_TIMES_S = (0.25, 0.5, 0.75, 1.0)     # absolute times after the onset of the fixed-window features
QUARTILES = (0.25, 0.5, 0.75, 1.0)         # fractions of the approach of the cumulative features
LEAD_S = 0.2                               # lead-in window [onset - LEAD_S, onset]
PRE_S = (0.5, 0.2)                         # negative-control window [onset - 0.5 s, onset - 0.2 s] (still hold)
DEDRIFT_QUARTILES = (0.5, 1.0)             # fractions of the approach of the de-drifted lateral velocity


@dataclass(frozen=True)
class Feature:
    name: str
    family: str          # yaw | roll | lateral | dedrift | device | fixed | control | even
    parity: str          # 'odd' (changes sign under the mirror) or 'even'
    unit: str
    window: str          # approach (time-normalised) | fixed (absolute time) | trial | approach_dedrift
    frame: str           # the frame the feature is computed in (gravity, body (u, h, s), device, none)
    flips: bool          # True: changes sign under the screen-away flip (standardize.screen_flip_channels)
    formula: str
    meaning: str

    @property
    def grip(self) -> str:
        """How the feature depends on the grip (generated from ``frame`` and ``flips``)."""
        return (f'{self.frame}: unchanged by the three screen-toward grips (rotations about the screen normal); '
                + ('changes sign under the screen-away flip' if self.flips else 'unchanged by the screen-away flip'))


def _pct(q: float) -> str:
    return f'{int(round(q * 100))}'


def _fix(t: float) -> str:
    return f'{t:.2f}'.replace('.', 'p').rstrip('0').rstrip('p') if t != int(t) else f'{int(t)}p0'


GRAV, BODY, DEV, NONE = 'gravity frame', 'body frame (u, h, s)', 'device frame', 'frame-free'
#: the roll and lateral axes (h, s) reverse under the screen-away flip; yaw about gravity does not
_FLIPS = {'yaw': False, 'roll': True, 'lateral': True}


def _catalogue() -> list[Feature]:
    F: list[Feature] = []
    # A. yaw about gravity
    for q in QUARTILES:
        nm = {0.5: 'first-half yaw', 1.0: 'net yaw'}.get(q, f'yaw at {_pct(q)} %')
        F.append(Feature(f'yaw_q{_pct(q)}', 'yaw', 'odd', 'deg', 'approach', GRAV, False,
                         f'yaw_cum(tau = {q:g}) = integral of w.u from the onset to {_pct(q)} % of the approach',
                         f'{nm}: rotation of the phone about gravity so far (+ = counter-clockwise from above)'))
    F.append(Feature('yaw_second_half', 'yaw', 'odd', 'deg', 'approach', GRAV, False, 'yaw_q100 - yaw_q50',
                     'yaw rotation during the second half of the approach (the turn back)'))
    for nb in (4, 8):
        for b in range(nb):
            F.append(Feature(f'yaw_rate_b{nb}_{b + 1}', 'yaw', 'odd', 'deg/s', 'approach', GRAV, False,
                             f'mean of w.u over time bin {b + 1} of {nb} of the approach',
                             f'yaw rate in the {_ordinal(b + 1)} {"quarter" if nb == 4 else "eighth"} of the approach'))
    F.append(Feature('yaw_peak_signed', 'yaw', 'odd', 'deg/s', 'approach', GRAV, False,
                     'w.u at the instant of max |w.u|', 'signed peak yaw rate (the largest turn, with its direction)'))
    F.append(Feature('yaw_peak_time', 'yaw', 'even', 'fraction', 'approach', GRAV, False,
                     'argmax |w.u| / 63', 'when the largest yaw turn happens (fraction of the approach)'))
    F.append(Feature('yaw_extreme_balance', 'yaw', 'odd', 'deg/s', 'approach', GRAV, False,
                     'max(w.u) + min(w.u)', 'which turn direction reaches the larger peak rate'))
    F.append(Feature('yaw_order', 'yaw', 'odd', 'fraction', 'approach', GRAV, False,
                     '(argmin w.u - argmax w.u) / 63',
                     'which turn comes first: > 0 when the counter-clockwise peak precedes the clockwise one'))
    F.append(Feature('yaw_excursion', 'yaw', 'odd', 'deg', 'approach', GRAV, False, 'yaw_cum at argmax |yaw_cum|',
                     'largest yaw excursion from the start heading, with its direction'))
    # B. roll about the facing axis
    for q in QUARTILES:
        F.append(Feature(f'roll_q{_pct(q)}', 'roll', 'odd', 'deg', 'approach', BODY, True,
                         f'integral of w.h from the onset to {_pct(q)} % of the approach',
                         'rotation about the horizontal axis through the screen (+ = counter-clockwise as seen by the '
                         'person)'))
    for b in range(4):
        F.append(Feature(f'roll_rate_b4_{b + 1}', 'roll', 'odd', 'deg/s', 'approach', BODY, True,
                         f'mean of w.h over quarter {b + 1} of the approach', f'roll rate, {_ordinal(b + 1)} quarter'))
    # C. lateral motion
    for q in QUARTILES:
        F.append(Feature(f'latvel_q{_pct(q)}', 'lateral', 'odd', 'm/s', 'approach', BODY, True,
                         f'integral of a.s from the onset to {_pct(q)} % of the approach',
                         'lateral velocity change (+ = towards the person\'s right; raw integral)'))
    for b in range(4):
        F.append(Feature(f'acc_lat_b4_{b + 1}', 'lateral', 'odd', 'm/s^2', 'approach', BODY, True,
                         f'mean of a.s over quarter {b + 1} of the approach',
                         f'lateral acceleration, {_ordinal(b + 1)} quarter'))
    # C2. lateral motion with the hold bias removed (a data-treatment option, not in the candidate pools)
    for q in DEDRIFT_QUARTILES:
        F.append(Feature(f'latvel_dd_q{_pct(q)}', 'dedrift', 'odd', 'm/s', 'approach_dedrift', BODY, True,
                         f'integral of (a.s - b) from the onset to {_pct(q)} % of the approach, b = mean a.s over '
                         f'[onset - {PRE_S[0]:g} s, onset - {PRE_S[1]:g} s]',
                         'lateral velocity change with the still-hold bias removed (first-order de-drifting)'))
    # D. device-frame chiral combinations made grip-invariant
    for q in (0.5, 1.0):
        F.append(Feature(f'gyroz_q{_pct(q)}', 'device', 'odd', 'deg', 'approach', DEV, True,
                         f'integral of w_z (device) from the onset to {_pct(q)} % of the approach',
                         'rotation of the phone in its own screen plane'))
    for nm, k, what in (('inplane_dev_start', 0, 'onset'), ('inplane_dev_end', -1, 'end')):
        F.append(Feature(nm, 'device', 'odd', 'deg', 'approach', DEV, True,
                         f'((atan2(-u_x, u_y) + 45) mod 90) - 45 at the {what}',
                         f'how far the phone is rolled away from the nearest canonical grip at the {what}'))
    # E. fixed absolute-time window (independent of the end bound)
    for t in FIXED_TIMES_S:
        F.append(Feature(f'yaw_fix_{_fix(t)}', 'fixed', 'odd', 'deg', 'fixed', GRAV, False,
                         f'yaw_cum at onset + {t:g} s (fixed window, frozen after the end cue)',
                         f'yaw about gravity {t:g} s into the movement (absolute time)'))
    for t in (0.5, 1.0):
        F.append(Feature(f'roll_fix_{_fix(t)}', 'fixed', 'odd', 'deg', 'fixed', BODY, True,
                         f'integral of w.h from the onset to onset + {t:g} s', f'roll {t:g} s into the movement'))
        F.append(Feature(f'latvel_fix_{_fix(t)}', 'fixed', 'odd', 'm/s', 'fixed', BODY, True,
                         f'integral of a.s from the onset to onset + {t:g} s',
                         f'lateral velocity change {t:g} s into the movement'))
    # lead-in: the last LEAD_S before the oracle onset (the movement may start before the detected onset)
    for nm, ch, unit in (('yaw_lead', 'yaw', 'deg'), ('roll_lead', 'roll', 'deg'), ('latvel_lead', 'lateral', 'm/s')):
        F.append(Feature(nm, 'fixed', 'odd', unit, 'fixed', GRAV if ch == 'yaw' else BODY, _FLIPS[ch],
                         f'change of the {ch} integral over [onset - {LEAD_S:g} s, onset]',
                         'lead-in: what happens just before the detected onset'))
    # negative control: the still hold at arm's length, well before the onset (must carry no hand)
    for nm, ch, unit in (('yaw_pre', 'yaw', 'deg'), ('roll_pre', 'roll', 'deg'), ('latvel_pre', 'lateral', 'm/s')):
        F.append(Feature(nm, 'control', 'odd', unit, 'fixed', GRAV if ch == 'yaw' else BODY, _FLIPS[ch],
                         f'change of the {ch} integral over [onset - {PRE_S[0]:g} s, onset - {PRE_S[1]:g} s]',
                         'negative control: the still hold before the movement (expected AUC 0.5)'))
    # F. even descriptors (cannot tell the hand alone; may modulate)
    ev = [
        ('duration_s', 's', 'trial', NONE, False, 'end_ms - onset_ms (oracle segment)', 'movement duration'),
        ('peak_gyro', 'deg/s', 'approach', NONE, False, 'max |w| over the approach', 'peak angular speed'),
        ('peak_acc', 'm/s^2', 'approach', NONE, False, 'max |a| over the approach', 'peak linear acceleration'),
        ('tilt_end', 'deg', 'approach', GRAV, False, 'tilt_from_start at the end',
         'total tilt of the phone from its start pose'),
        ('tilt_rate_mean', 'deg/s', 'approach', GRAV, False, 'mean |w - (w.u)u|', 'mean tilting rate'),
        ('pitch_q100', 'deg', 'approach', BODY, True, 'integral of w.s over the approach',
         'net pitch (top edge towards the person)'),
        ('screen_up_start', '1', 'approach', DEV, True, 'u_z at the onset', 'how much the screen faces up at the start'),
        ('screen_up_end', '1', 'approach', DEV, True, 'u_z at the end', 'how much the screen faces up at the nose'),
        ('yaw_abs_total', 'deg', 'approach', GRAV, False, 'integral of |w.u|', 'total yaw activity, either direction'),
        ('roll_abs_total', 'deg', 'approach', BODY, False, 'integral of |w.h|', 'total roll activity, either direction'),
        ('acc_horiz_rms', 'm/s^2', 'approach', GRAV, False, 'RMS of |a - (a.u)u|', 'horizontal acceleration effort'),
        ('acc_vert_q50', 'm/s', 'approach', GRAV, False, 'integral of a.u to 50 %',
         'vertical velocity change in the first half'),
        ('fwdvel_q50', 'm/s', 'approach', BODY, True, 'integral of a.h to 50 %',
         'velocity change towards the face, first half'),
    ]
    for nm, unit, win, frame, flips, formula, meaning in ev:
        F.append(Feature(nm, 'even', 'even', unit, win, frame, flips, formula, meaning))
    return F


def _ordinal(k: int) -> str:
    return {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh',
            8: 'eighth'}[k]


CATALOGUE: list[Feature] = _catalogue()
NAMES: list[str] = [f.name for f in CATALOGUE]
BY_NAME: dict[str, Feature] = {f.name: f for f in CATALOGUE}
ODD: list[str] = [f.name for f in CATALOGUE if f.parity == 'odd']
EVEN: list[str] = [f.name for f in CATALOGUE if f.parity == 'even']
#: chiral features of the approach (no control, no absolute-time feature, no de-drifted variant): the candidate
#: pool of nested selection
CHIRAL_APPROACH: list[str] = [f.name for f in CATALOGUE if f.parity == 'odd' and f.window == 'approach']
#: +1 for even, -1 for odd features: the mirror image of a feature row is F * MIRROR_SIGN
MIRROR_SIGN = pd.Series({f.name: (-1.0 if f.parity == 'odd' else 1.0) for f in CATALOGUE})


def catalogue_table() -> pd.DataFrame:
    return pd.DataFrame([{**asdict(f), 'grip': f.grip} for f in CATALOGUE])


# ───────────────────────────── derived channels ─────────────────────────────


def body_channels(X: np.ndarray, channels: Sequence[str] = CHANNELS) -> dict[str, np.ndarray]:
    """roll_rate, pitch_rate, acc_lat, acc_fwd (N, T) from canonical arrays X (N, C, T): projections of w and a on
    the heading-following body frame (h, s) of the module docstring. Per sample, per trial (no fitting)."""
    ix = {c: i for i, c in enumerate(channels)}
    X = np.asarray(X, dtype=np.float64)
    u = X[:, [ix['up_x'], ix['up_y'], ix['up_z']], :]
    w = X[:, [ix['gyro_x'], ix['gyro_y'], ix['gyro_z']], :]
    a = X[:, [ix['acc_x'], ix['acc_y'], ix['acc_z']], :]
    h = -u[:, 2:3, :] * u
    h[:, 2, :] += 1.0
    nrm = np.linalg.norm(h, axis=1, keepdims=True)
    h = h / np.where(nrm > 1e-6, nrm, np.nan)          # undefined only with the screen exactly horizontal
    s = np.cross(u, h, axis=1)
    return {'roll_rate': np.sum(w * h, axis=1), 'pitch_rate': np.sum(w * s, axis=1),
            'acc_lat': np.sum(a * s, axis=1), 'acc_fwd': np.sum(a * h, axis=1)}


def _cumtrapz(x: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid along the last axis, x (N, T), dt (N,) or scalar; 0 at the first sample."""
    dt = np.asarray(dt, dtype=float).reshape(-1, 1) if np.ndim(dt) else float(dt)
    inc = 0.5 * (x[:, 1:] + x[:, :-1]) * dt
    return np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(inc, axis=1)], axis=1)


def _at(x: np.ndarray, q: float) -> np.ndarray:
    """Value of x (N, T) at the fraction q of the time-normalised window (linear interpolation)."""
    T = x.shape[1]
    pos = q * (T - 1)
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, T - 1)
    f = pos - i0
    return (1 - f) * x[:, i0] + f * x[:, i1]


def _bins(x: np.ndarray, nb: int) -> np.ndarray:
    """Means of x (N, T) over nb equal time bins (T divisible by nb)."""
    N, T = x.shape
    if T % nb:
        edges = np.linspace(0, T, nb + 1).round().astype(int)
        return np.stack([x[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return x.reshape(N, nb, T // nb).mean(axis=2)


def compute_features(approach: np.ndarray, fixed: np.ndarray, fixed_mask: np.ndarray, duration_s: np.ndarray,
                     channels: Sequence[str] = CHANNELS) -> pd.DataFrame:
    """The catalogue (columns in :data:`NAMES` order) for N trials. Pure per-trial computation, no labels."""
    ix = {c: i for i, c in enumerate(channels)}
    A = np.asarray(approach, dtype=np.float64)
    N, _, T = A.shape
    dur = np.asarray(duration_s, dtype=float).reshape(-1)
    if dur.size != N or not np.all(np.isfinite(dur)) or np.any(dur <= 0):
        raise ValueError('duration_s must be finite, positive and one per trial')
    dt = dur / (T - 1)
    yr = A[:, ix['yaw_rate']]
    yc = A[:, ix['yaw_cum']]
    bc = body_channels(A, channels)
    roll = _cumtrapz(bc['roll_rate'], dt)
    lat = _cumtrapz(bc['acc_lat'], dt)
    out: dict[str, np.ndarray] = {}
    for q in QUARTILES:
        out[f'yaw_q{_pct(q)}'] = _at(yc, q)
    out['yaw_second_half'] = out['yaw_q100'] - out['yaw_q50']
    for nb in (4, 8):
        b = _bins(yr, nb)
        for k in range(nb):
            out[f'yaw_rate_b{nb}_{k + 1}'] = b[:, k]
    iabs = np.argmax(np.abs(yr), axis=1)
    out['yaw_peak_signed'] = yr[np.arange(N), iabs]
    out['yaw_peak_time'] = iabs / (T - 1)
    out['yaw_extreme_balance'] = yr.max(axis=1) + yr.min(axis=1)
    out['yaw_order'] = (np.argmin(yr, axis=1) - np.argmax(yr, axis=1)) / (T - 1)
    iex = np.argmax(np.abs(yc), axis=1)
    out['yaw_excursion'] = yc[np.arange(N), iex]
    for q in QUARTILES:
        out[f'roll_q{_pct(q)}'] = _at(roll, q)
    rb = _bins(bc['roll_rate'], 4)
    for k in range(4):
        out[f'roll_rate_b4_{k + 1}'] = rb[:, k]
    for q in QUARTILES:
        out[f'latvel_q{_pct(q)}'] = _at(lat, q)
    lb = _bins(bc['acc_lat'], 4)
    for k in range(4):
        out[f'acc_lat_b4_{k + 1}'] = lb[:, k]
    gz = _cumtrapz(A[:, ix['gyro_z']], dt)
    for q in (0.5, 1.0):
        out[f'gyroz_q{_pct(q)}'] = _at(gz, q)
    ang = np.degrees(np.arctan2(-A[:, ix['up_x']], A[:, ix['up_y']]))
    dev = np.mod(ang + 45.0, 90.0) - 45.0
    out['inplane_dev_start'] = dev[:, 0]
    out['inplane_dev_end'] = dev[:, -1]
    # fixed window (60 Hz, onset at FIXED_ONSET_INDEX; masked samples do not enter an integral)
    Fx = np.asarray(fixed, dtype=np.float64)
    m = np.asarray(fixed_mask, dtype=bool)
    k0 = FIXED_ONSET_INDEX
    fyc = Fx[:, ix['yaw_cum']]
    fb = body_channels(Fx, channels)
    froll = _cumtrapz(np.where(m, fb['roll_rate'], 0.0), 1.0 / FS_HZ)
    flat = _cumtrapz(np.where(m, fb['acc_lat'], 0.0), 1.0 / FS_HZ)
    fyr = _cumtrapz(np.where(m, Fx[:, ix['yaw_rate']], 0.0), 1.0 / FS_HZ)
    for t in FIXED_TIMES_S:
        k = k0 + int(round(t * FS_HZ))
        out[f'yaw_fix_{_fix(t)}'] = fyc[:, k] - fyc[:, k0]
    for t in (0.5, 1.0):
        k = k0 + int(round(t * FS_HZ))
        out[f'roll_fix_{_fix(t)}'] = froll[:, k] - froll[:, k0]
        out[f'latvel_fix_{_fix(t)}'] = flat[:, k] - flat[:, k0]
    kl = k0 - int(round(LEAD_S * FS_HZ))
    ka, kb = k0 - int(round(PRE_S[0] * FS_HZ)), k0 - int(round(PRE_S[1] * FS_HZ))
    for nm, c in (('yaw', fyr), ('roll', froll), ('latvel', flat)):
        out[f'{nm}_lead'] = c[:, k0] - c[:, kl]
        out[f'{nm}_pre'] = c[:, kb] - c[:, ka]
    # de-drifted lateral velocity: the trial's own mean a.s over the still hold removed before integrating
    bias = hold_lateral_bias(fb['acc_lat'], m)
    for q in DEDRIFT_QUARTILES:
        out[f'latvel_dd_q{_pct(q)}'] = out[f'latvel_q{_pct(q)}'] - bias * q * dur
    # even descriptors
    out['duration_s'] = dur
    out['peak_gyro'] = A[:, ix['gyro_mag']].max(axis=1)
    out['peak_acc'] = A[:, ix['acc_mag']].max(axis=1)
    out['tilt_end'] = A[:, ix['tilt_from_start'], -1]
    out['tilt_rate_mean'] = A[:, ix['gyro_horiz']].mean(axis=1)
    out['pitch_q100'] = _cumtrapz(bc['pitch_rate'], dt)[:, -1]
    out['screen_up_start'] = A[:, ix['up_z'], 0]
    out['screen_up_end'] = A[:, ix['up_z'], -1]
    out['yaw_abs_total'] = _cumtrapz(np.abs(yr), dt)[:, -1]
    out['roll_abs_total'] = _cumtrapz(np.abs(bc['roll_rate']), dt)[:, -1]
    out['acc_horiz_rms'] = np.sqrt(np.mean(A[:, ix['acc_horiz']] ** 2, axis=1))
    out['acc_vert_q50'] = _at(_cumtrapz(A[:, ix['acc_vert']], dt), 0.5)
    out['fwdvel_q50'] = _at(_cumtrapz(bc['acc_fwd'], dt), 0.5)
    df = pd.DataFrame({n: out[n] for n in NAMES})
    bad = [c for c in df.columns if not np.all(np.isfinite(df[c].to_numpy()))]
    if bad:
        raise ValueError(f'non-finite feature values in {bad}')
    return df


def hold_lateral_bias(acc_lat_fixed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean lateral acceleration a.s (m/s^2) over the still hold [onset - PRE_S[0], onset - PRE_S[1]] of the fixed
    window, valid samples only (0 when none is valid): the per-trial bias the de-drifted features remove."""
    k0 = FIXED_ONSET_INDEX
    ka, kb = k0 - int(round(PRE_S[0] * FS_HZ)), k0 - int(round(PRE_S[1] * FS_HZ))
    x = np.asarray(acc_lat_fixed, dtype=float)[:, ka:kb + 1]
    m = np.asarray(mask, dtype=bool)[:, ka:kb + 1] & np.isfinite(x)
    n = m.sum(axis=1)
    return np.where(n > 0, np.where(m, x, 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)


def build_x(d) -> pd.DataFrame:
    """Every catalogue feature of the trials of a ``Stage2Data`` (primary, natural use, ...): (N, F) DataFrame aligned
    with ``d.trials``. Per trial only, no label: the ``build_x`` of the m1 models."""
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    return compute_features(d.approach, d.fixed, d.fixed_mask, dur, d.channels)


def mirror_features(F: pd.DataFrame) -> pd.DataFrame:
    """The feature rows of the mirror-image movements (odd columns change sign)."""
    return F * MIRROR_SIGN.reindex(F.columns).to_numpy()


# ───────────────────────────── time courses (temporal filter) ─────────────────────────────

#: chiral rate channels of the temporal filter and their number of equal time bins over the approach
TC_CHANNELS: tuple[str, ...] = ('yaw_rate', 'roll_rate', 'acc_lat')
TC_BINS = 16


def tc_columns(channels: Sequence[str] = TC_CHANNELS, nbins: int = TC_BINS) -> list[str]:
    return [f'{c}_t{k + 1:02d}' for c in channels for k in range(nbins)]


def tc_bin_centres(nbins: int = TC_BINS) -> np.ndarray:
    """Centre of each time bin as a fraction of the approach (0 = onset, 1 = end)."""
    return (np.arange(nbins) + 0.5) / nbins


def compute_timecourse(approach: np.ndarray, channels: Sequence[str] = CHANNELS,
                       which: Sequence[str] = TC_CHANNELS, nbins: int = TC_BINS) -> pd.DataFrame:
    """Mean of each chiral rate channel over ``nbins`` equal bins of the time-normalised approach: (N, len(which)
    * nbins) DataFrame, columns ``<channel>_tKK``. Every column is ODD under the mirror. Per trial, no labels."""
    A = np.asarray(approach, dtype=np.float64)
    ix = {c: i for i, c in enumerate(channels)}
    bc = body_channels(A, channels)
    cols = {}
    for c in which:
        x = A[:, ix[c]] if c in ix else bc[c]
        b = _bins(x, nbins)
        for k in range(nbins):
            cols[f'{c}_t{k + 1:02d}'] = b[:, k]
    df = pd.DataFrame(cols)
    if not np.all(np.isfinite(df.to_numpy())):
        raise ValueError('non-finite time-course values')
    return df


def build_tc(d) -> pd.DataFrame:
    """The binned chiral time courses of a ``Stage2Data`` (the ``build_x`` of the temporal-filter models)."""
    return compute_timecourse(d.approach, d.channels)


# ───────────────────────────── inputs of the core D6 wrong-hand screen ─────────────────────────────


def screen_columns() -> list[str]:
    """Column names of the core D6 screen's inputs (``stage2.screen.screen_features``: cumulative yaw and yaw rate
    at 8 instants of the approach), as used by the m1 sensitivity runs."""
    from stage2 import screen
    return [f'scr_{c}_s{k:02d}' for c in screen.SCREEN_CHANNELS for k in screen.SCREEN_STEPS]


def build_screen(d) -> pd.DataFrame:
    """The D6 screen's inputs of a ``Stage2Data`` as a DataFrame (per trial, no label)."""
    from stage2 import screen
    return pd.DataFrame(screen.screen_features(d.approach), columns=screen_columns())


# ───────────────────────────── per-trial amplitude normalisation (a data-treatment option) ─────────────────────────────

#: the even, per-trial amplitude each chiral family is divided by (unitless ratios; odd / even = odd)
AMP_DIVISOR = {'yaw': 'yaw_abs_total', 'roll': 'roll_abs_total', 'latvel': None}


def amp_normalised(F: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Per-trial amplitude-normalised copies of chiral columns (suffix ``_an``): yaw and roll integrals divided by
    the trial's total yaw / roll activity (integral of |rate|), lateral velocities by the trial's horizontal
    acceleration effort (RMS |a_h| x duration). The ratio keeps the sign (odd) and removes the movement's vigour:
    it tests whether the hand is in the SHAPE of the movement rather than its size. Per trial, no fitting."""
    out = {}
    for c in cols:
        fam = c.split('_')[0]
        if fam in ('yaw', 'roll'):
            div = F[AMP_DIVISOR[fam]].to_numpy(dtype=float)
        elif fam == 'latvel':
            div = F['acc_horiz_rms'].to_numpy(dtype=float) * F['duration_s'].to_numpy(dtype=float)
        else:
            raise ValueError(f'no amplitude for {c}')
        out[f'{c}_an'] = F[c].to_numpy(dtype=float) / np.maximum(div, 1e-9)
    return pd.DataFrame(out)
