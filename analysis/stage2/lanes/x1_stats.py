"""Lane x1: the characteristics, the matched comparisons and the single-case statistics (no model, no label use).

* :data:`CHARACTERISTICS` - every characteristic of the case study: label, unit, group, plain-language rationale,
  the matched comparison it is judged in (``stratum``), its declared scale (``log`` for strictly positive,
  right-skewed quantities: the robust z and the Crawford-Howell test use it), its *phenomenon* (characteristics that
  measure the same thing, e.g. the time and the share of the approach of one touch, count as one finding) and
  whether it was defined after PD01's recordings were inspected (``post_hoc``: reported apart, without a q-value).
* :data:`STRATA` - like-for-like comparisons: PD01's trials against the modelling persons' trials of the same hand,
  grip, speed and movement variant (``left_upright_natural`` / ``_slow`` / ``_fast`` / ``_all``), the free-choice
  trials against the natural-use set, every trial as a broad secondary reference, and the upright natural trials of
  either hand (a secondary, larger reference for the across-repeat measures).
* :data:`VARIANTS` - the sensitivity analyses. ``first_k``, ``same_protocol`` and ``no_clip`` recompute person values
  and can overturn a "beyond every reference person" verdict; ``same_posture`` does so only when a reference person
  has trials in both postures (otherwise it, like ``same_platform``, only drops persons and cannot overturn it).
* :func:`compare` - one exploratory person's value against the reference persons' person-level values: reference
  median [IQR] and range, mid-rank percentile, robust z on the declared scale (median / 1.4826 MAD) and its range
  when each reference person is left out, the two-sided rank p-value (``(1 + #{as or more extreme on that side}) /
  (n + 1)``, doubled; its floor is 2 / (n + 1)), the Crawford-Howell (1998) modified t-test on the declared and the
  other scale, and how many of the case's own trials lie beyond the reference persons' range.
* :func:`bh` - Benjamini-Hochberg q-values (the family: the pre-specified characteristics).
* :func:`beyond_all_counts` - the null reference of the "beyond every reference person" count: each reference
  person against the others, and the case against all of them, per characteristic and per phenomenon.
* :func:`multivariate` - the person-level vectors of the key characteristics on their declared scales,
  robust-standardised: a PCA map fitted on the reference persons (the exploratory person projected), and a
  Mahalanobis distance with a Ledoit-Wolf shrinkage covariance; each reference person's distance is computed leaving
  that person out (standardisation and covariance from the others), so its rank against the exploratory person's
  distance is a fair test; :func:`multivariate_loco` repeats it with each characteristic left out.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as SS

#: Fewer reference persons than this: no statistic is computed (only n).
MIN_REF = 5
#: |robust z| at or above this on the declared scale, with every reference person left out in turn, is "clear".
Z_CLEAR = 2.0

# ───────────────────────────── strata ─────────────────────────────

SETS_ALL = ('primary', 'natural_use', 'natural_use_repeat')


def _instructed(t: pd.DataFrame, hand: str | None, speed: str | None) -> pd.Series:
    m = ((t['analysis_set'] == 'primary') & (t['orientation'] == 'upright_toward') & (t['variant'] == 'standard')
         & (t['start_position'] == 'natural'))
    if hand is not None:
        m &= t['hand'] == hand
    if speed is not None:
        m &= t['speed'] == speed
    return m


@dataclass(frozen=True)
class Stratum:
    key: str
    label: str
    select: Callable[[pd.DataFrame], pd.Series]
    note: str = ''


STRATA: dict[str, Stratum] = {s.key: s for s in (
    Stratum('left_upright_natural', 'left hand, upright, natural speed',
            lambda t: _instructed(t, 'left', 'natural'),
            'instructed, standard movement, natural start position'),
    Stratum('left_upright_slow', 'left hand, upright, slow', lambda t: _instructed(t, 'left', 'slow')),
    Stratum('left_upright_fast', 'left hand, upright, fast', lambda t: _instructed(t, 'left', 'fast')),
    Stratum('left_upright_all', 'left hand, upright, all speeds', lambda t: _instructed(t, 'left', None)),
    Stratum('upright_natural_both', 'upright, natural speed, either hand', lambda t: _instructed(t, None, 'natural'),
            'a larger but hand-mixed reference for the across-repeat measures'),
    Stratum('free_choice', 'free choice (first free trial of a person)', lambda t: t['analysis_set'] == 'natural_use',
            'rule N1: the earliest free-choice trial that passed every rule'),
    Stratum('all_trials', 'every accepted trial', lambda t: t['analysis_set'].isin(SETS_ALL)),
)}

#: Matched strata in which every trial-level characteristic is also reported (the long table).
REPORT_STRATA = ('left_upright_natural', 'left_upright_slow', 'left_upright_fast', 'left_upright_all',
                 'free_choice', 'all_trials')

#: The sensitivity analyses (see the module docstring) and the ones that can overturn a "beyond all" verdict.
VARIANTS = ('main', 'first_k', 'same_protocol', 'no_clip', 'same_posture', 'same_platform')
VARIANT_LABELS = {'main': 'main', 'first_k': 'first k trials', 'same_protocol': 'same protocol',
                  'no_clip': 'without clip-flagged trials', 'same_posture': 'same posture',
                  'same_platform': 'same platform'}
VALUE_CHECKS = ('first_k', 'same_protocol', 'no_clip', 'same_posture')

# ───────────────────────────── characteristics ─────────────────────────────


@dataclass(frozen=True)
class Characteristic:
    key: str
    label: str
    unit: str
    group: str
    rationale: str
    stratum: str                  # the matched comparison it is judged in (the primary row of the main table)
    scale: str = 'log'            # 'log' or 'linear': the declared scale of the robust z and Crawford-Howell test
    fmt: str = '.2f'
    level: str = 'trial'          # 'trial': person value = mean of the person's trials; 'person': computed per person
    phenomenon: str = ''          # characteristics measuring the same thing share one (one finding, not several)
    key_set: bool = False         # in the compact main-text table and the strip figures
    multivariate: bool = False    # in the multivariate map (pre-specified)
    post_hoc: bool = False        # defined after PD01's recordings or results were inspected: no q-value
    report_z: bool = True         # False for shares of binary trial outcomes (a robust z means little there)


G_PREP, G_SPEED, G_TOUCH, G_AMP = 'preparation and hold', 'movement time and speed', 'first screen touch', 'amplitude'
G_SMOOTH, G_REACH, G_NOSE = 'smoothness', 'band power during the reach', 'band power at the nose'
G_VAR, G_MOD, G_SIG, G_TASK = 'repeats', 'speed scaling', 'hand signature', 'task behaviour'
GROUPS = (G_PREP, G_SPEED, G_TOUCH, G_AMP, G_SMOOTH, G_REACH, G_NOSE, G_VAR, G_MOD, G_SIG, G_TASK)
LUA, LUN = 'left_upright_all', 'left_upright_natural'

# phenomena: what is measured (declared by definition, not by the case's values)
P_READY, P_HOLD, P_DUR, P_SPEED = 'getting ready', 'hold steadiness', 'reach duration', 'reach angular speed'
P_ACC, P_TOUCH, P_ROT, P_NET = 'linear acceleration', 'first-touch timing', 'rotation amount', 'net orientation change'
P_SMOOTH, P_REACHBP, P_NOSEBP = 'smoothness', 'reach band power', 'nose band power'
P_REP, P_SCALE, P_EARLY, P_NETYAW = 'repeats', 'speed scaling', 'early hand-consistent yaw', 'net hand-consistent yaw'
P_TASK, P_RETURN = 'attempts', 'return of the arm'

_PRE_RAT = ('the 3-12 Hz band power relative to 0.5-3 and 12-15 Hz; it rises with more 3-12 Hz power or with less slow '
            'sway and less 12-15 Hz power, so its components are reported too')

CHARACTERISTICS: tuple[Characteristic, ...] = (
    # preparation (the hold before GO, initiation)
    Characteristic('hold_wait_s', 'Time to a valid hold', 's', G_PREP,
                   'Seconds from the start of the recording until the app\'s 400-ms hold-still requirement was met '
                   '(includes the instruction\'s reading time): difficulty holding the phone still prolongs it.',
                   LUA, 'log', '.1f', phenomenon=P_READY),
    Characteristic('false_starts', 'False starts', 'per trial', G_PREP,
                   'Movements detected during the hold before GO, each restarting the hold.', LUA, 'linear', '.2f',
                   phenomenon=P_READY, key_set=True),
    Characteristic('rt_s', 'Reaction time', 's', G_PREP,
                   'GO cue to movement onset; slowed initiation is part of bradykinesia.', LUA, 'linear', '.2f',
                   phenomenon=P_READY, key_set=True, multivariate=True),
    Characteristic('pre_rms_3_7', 'Hold 3-7 Hz band RMS', 'deg/s', G_PREP,
                   'Angular-velocity RMS in 3-7 Hz (the band of parkinsonian rest and re-emergent tremor) while the '
                   'outstretched arm holds the phone still (1 s before the onset; 1-Hz resolution).', LUA, 'log', '.2f',
                   phenomenon=P_HOLD, key_set=True, multivariate=True),
    Characteristic('pre_rms_7_12', 'Hold 7-12 Hz band RMS', 'deg/s', G_PREP,
                   'Angular-velocity RMS in 7-12 Hz (the band of physiological and action tremor) during the hold '
                   'before the movement.', LUA, 'log', '.2f', phenomenon=P_HOLD),
    Characteristic('pre_peak_hz', 'Hold spectral peak, 3-12 Hz', 'Hz', G_PREP,
                   'Frequency of the highest spectral peak in 3-12 Hz during the hold, at the window\'s 1-Hz '
                   'resolution (trials without a peak there are left out).', LUA, 'linear', '.0f', phenomenon=P_HOLD),
    Characteristic('pre_ratio_db', 'Hold power ratio, 3-12 Hz vs rest', 'dB', G_PREP,
                   f'During the hold, {_PRE_RAT}.', LUA, 'linear', '.1f', phenomenon=P_HOLD),
    Characteristic('pre_sway_gyro', 'Hold sway, rotation', 'deg/s RMS', G_PREP,
                   'Total rotational unsteadiness of the hold (RMS of the angular velocity about its mean).',
                   LUA, 'log', '.2f', phenomenon=P_HOLD),
    Characteristic('pre_sway_acc', 'Hold sway, acceleration', 'm/s$^2$ RMS', G_PREP,
                   'Total translational unsteadiness of the hold (RMS of the specific force about its mean).',
                   LUA, 'log', '.3f', phenomenon=P_HOLD),
    Characteristic('pre_rms_05_3', 'Hold 0.5-3 Hz band RMS (slow sway)', 'deg/s', G_PREP,
                   'Angular-velocity RMS in 0.5-3 Hz during the hold: slow postural drift (one component of the hold '
                   'power ratio; about two resolution cells of the 1-s window).', LUA, 'log', '.2f',
                   phenomenon=P_HOLD, post_hoc=True),
    Characteristic('pre_rms_3_12', 'Hold 3-12 Hz band RMS', 'deg/s', G_PREP,
                   'Angular-velocity RMS in 3-12 Hz during the hold (the numerator of the hold power ratio).', LUA,
                   'log', '.2f', phenomenon=P_HOLD, post_hoc=True),
    Characteristic('pre_rms_12_15', 'Hold 12-15 Hz band RMS', 'deg/s', G_PREP,
                   'Angular-velocity RMS in 12-15 Hz during the hold (the top of the analysable band, where the '
                   'sensor noise floor matters).', LUA, 'log', '.2f', phenomenon=P_HOLD, post_hoc=True),
    # movement time and speed (bradykinesia)
    Characteristic('duration_s', 'Approach duration', 's', G_SPEED,
                   'Movement time from the onset to the end at the nose; slower reaches take longer.', LUN, 'log',
                   '.2f', phenomenon=P_DUR, key_set=True, multivariate=True),
    Characteristic('active_s', 'Active movement time', 's', G_SPEED,
                   'Onset to the last moment the phone turns faster than 30 deg/s (the app\'s onset floor): the time '
                   'the phone really moves, without the wait for stillness that ends the segmented approach.', LUN,
                   'log', '.2f', phenomenon=P_DUR, post_hoc=True),
    Characteristic('settle_s', 'Settling before the end', 's', G_SPEED,
                   'From the last movement faster than 30 deg/s to the declared end: how long the end detector waited '
                   'for the phone to be still at the face.', LUN, 'log', '.2f', phenomenon=P_DUR, key_set=True,
                   post_hoc=True),
    Characteristic('peak_speed', 'Peak angular speed, as recorded', 'deg/s', G_SPEED,
                   'Fastest rotation of the phone in the reach, including any superimposed faster component.', LUN,
                   'log', '.0f', phenomenon=P_SPEED),
    Characteristic('peak_speed_mv', 'Peak angular speed, movement band', 'deg/s', G_SPEED,
                   'Fastest rotation of the voluntary reach (angular velocity low-passed at 3 Hz).', LUN, 'log',
                   '.0f', phenomenon=P_SPEED, key_set=True, multivariate=True),
    Characteristic('mean_speed_mv', 'Mean angular speed, movement band', 'deg/s', G_SPEED,
                   'Average rotation speed of the voluntary reach.', LUN, 'log', '.0f', phenomenon=P_SPEED),
    Characteristic('time_to_peak', 'Time to peak speed', 'share of reach', G_SPEED,
                   'Where in the reach the voluntary rotation is fastest (0 = onset, 1 = end).', LUN, 'linear',
                   '.2f', phenomenon=P_SPEED),
    Characteristic('peak_acc', 'Peak linear acceleration', 'm/s$^2$', G_SPEED,
                   'Largest linear acceleration of the phone in the reach (a lower bound where the accelerometer may '
                   'have saturated: the clip flag).', LUN, 'log', '.1f', phenomenon=P_ACC),
    # the first screen touch
    Characteristic('touch_s', 'Time to first screen touch', 's', G_TOUCH,
                   'Onset to the first screen touch (the app cannot tell the nose from a finger or the grip); persons '
                   'who never touched in the stratum are left out.', LUN, 'log', '.2f', phenomenon=P_TOUCH,
                   key_set=True, post_hoc=True),
    Characteristic('touch_share', 'First touch, share of the approach', 'share', G_TOUCH,
                   'When the first touch falls within the segmented approach (0 = onset, 1 = declared end).', LUN,
                   'linear', '.2f', phenomenon=P_TOUCH, post_hoc=True),
    Characteristic('rot_after_touch', 'Rotation after the first touch', 'share', G_TOUCH,
                   'Share of the approach\'s voluntary (3-Hz) rotation that comes after the first touch: small when '
                   'the touch marks the arrival, large when the movement goes on after it.', LUN, 'linear', '.2f',
                   phenomenon=P_TOUCH, post_hoc=True),
    Characteristic('touch_acc', 'Acceleration at first touch', 'm/s$^2$', G_TOUCH,
                   'Largest linear acceleration within 100 ms of the first touch (whole canonical recording; a lower '
                   'bound where the accelerometer may have saturated).', LUN, 'log', '.1f', phenomenon=P_ACC,
                   key_set=True, post_hoc=True),
    # amplitude (hypometria)
    Characteristic('rotation_total_mv', 'Rotation, movement band', 'deg', G_AMP,
                   'Angle the phone turns in the voluntary reach (integral of the 3-Hz angular speed); small '
                   'movements (hypometria) turn less.', LUN, 'log', '.0f', phenomenon=P_ROT, key_set=True,
                   multivariate=True),
    Characteristic('rotation_net', 'Net rotation', 'deg', G_AMP,
                   'Orientation change from the start pose to the pose at the nose (integrated gyroscope).', LUN,
                   'log', '.1f', phenomenon=P_NET),
    Characteristic('tilt_change', 'Tilt change', 'deg', G_AMP,
                   'Change of the phone\'s tilt relative to gravity from the onset to the nose.', LUN, 'log', '.1f',
                   phenomenon=P_NET),
    Characteristic('net_yaw_abs', 'Net yaw, magnitude', 'deg', G_AMP,
                   'Size of the turn about gravity from the onset to the nose.', LUN, 'log', '.1f', phenomenon=P_NET),
    # smoothness
    Characteristic('ldlj', 'Log dimensionless jerk', '', G_SMOOTH,
                   'Smoothness of the angular-speed profile (higher = smoother); jerky or oscillating reaches score '
                   'lower.', LUN, 'linear', '.1f', phenomenon=P_SMOOTH, key_set=True, multivariate=True),
    Characteristic('sparc', 'Spectral arc length (SPARC)', '', G_SMOOTH,
                   'Smoothness from the speed profile\'s spectrum (closer to 0 = smoother), robust to noise.', LUN,
                   'linear', '.2f', phenomenon=P_SMOOTH, key_set=True, multivariate=True),
    Characteristic('speed_peaks', 'Speed peaks, as recorded', 'count', G_SMOOTH,
                   'Peaks of the angular speed rising >= 10 % of its maximum; faster components and sub-movements '
                   'add peaks.', LUN, 'log', '.1f', phenomenon=P_SMOOTH),
    Characteristic('submovements', 'Sub-movements, movement band', 'count', G_SMOOTH,
                   'Peaks of the 3-Hz angular speed: separate voluntary sub-movements.', LUN, 'log', '.1f',
                   phenomenon=P_SMOOTH),
    Characteristic('hesitations', 'Hesitations', 'count', G_SMOOTH,
                   'Mid-reach pauses: >= 100 ms with the voluntary angular speed below the app\'s still threshold '
                   '(20 deg/s); freezing-like plateaus would appear here.', LUN, 'linear', '.2f',
                   phenomenon=P_SMOOTH),
    # band power during the reach
    Characteristic('app_rms_3_7', 'Reach 3-7 Hz band RMS', 'deg/s', G_REACH,
                   'Angular-velocity RMS in 3-7 Hz during the reach (the reach itself contributes at its low-frequency '
                   'end).', LUN, 'log', '.1f', phenomenon=P_REACHBP, key_set=True, multivariate=True),
    Characteristic('app_rms_7_12', 'Reach 7-12 Hz band RMS', 'deg/s', G_REACH,
                   'Angular-velocity RMS in 7-12 Hz during the reach: faster components on top of the voluntary '
                   'movement (called a tremor only where the spectrum shows a peak).', LUN, 'log', '.1f',
                   phenomenon=P_REACHBP, key_set=True, multivariate=True),
    Characteristic('app_peak_hz', 'Reach spectral peak, 3-12 Hz', 'Hz', G_REACH,
                   'Frequency of the highest spectral peak in 3-12 Hz during the reach.', LUN, 'linear', '.1f',
                   phenomenon=P_REACHBP),
    Characteristic('app_ratio_db', 'Reach power ratio, 3-12 Hz vs rest', 'dB', G_REACH,
                   f'During the reach, {_PRE_RAT}.', LUN, 'linear', '.1f', phenomenon=P_REACHBP),
    Characteristic('app_7_12_per_speed', 'Reach 7-12 Hz RMS per peak speed', 'ratio', G_REACH,
                   'The 7-12 Hz band RMS divided by the movement-band peak speed: the faster components relative to '
                   'the speed of the reach (a faster reach carries more of them).', LUN, 'log', '.2f',
                   phenomenon=P_REACHBP, post_hoc=True),
    # band power at the nose
    Characteristic('nose_rms_3_7', 'Nose 3-7 Hz band RMS', 'deg/s', G_NOSE,
                   'Angular-velocity RMS in 3-7 Hz while the phone rests at the nose (0.6 s after the end; '
                   '1.67-Hz resolution; only trials that became still).', LUA, 'log', '.2f', phenomenon=P_NOSEBP,
                   key_set=True),
    Characteristic('nose_rms_7_12', 'Nose 7-12 Hz band RMS', 'deg/s', G_NOSE,
                   'Angular-velocity RMS in 7-12 Hz while the phone rests at the nose (0.6 s after the end).',
                   LUA, 'log', '.2f', phenomenon=P_NOSEBP),
    Characteristic('nose_peak_hz', 'Nose spectral peak, 3-12 Hz', 'Hz', G_NOSE,
                   'Frequency of the highest spectral peak in 3-12 Hz at the nose, at the window\'s 1.67-Hz '
                   'resolution.', LUA, 'linear', '.0f', phenomenon=P_NOSEBP),
    Characteristic('nose_ratio_db', 'Nose power ratio, 3-12 Hz vs rest', 'dB', G_NOSE,
                   f'At the nose, {_PRE_RAT}.', LUA, 'linear', '.1f', phenomenon=P_NOSEBP),
    # repeats (person-level)
    Characteristic('cv_duration', 'Duration variability (CV)', '', G_VAR,
                   'Coefficient of variation of the approach duration over the first left-hand upright natural '
                   'reaches (as many as the exploratory person made).', LUN, 'log', '.2f', level='person',
                   phenomenon=P_REP, key_set=True),
    Characteristic('cv_peak_speed_mv', 'Peak-speed variability (CV)', '', G_VAR,
                   'Coefficient of variation of the movement-band peak angular speed over the first left-hand upright '
                   'natural reaches.', LUN, 'log', '.2f', level='person', phenomenon=P_REP),
    Characteristic('seq_peak_speed_mv', 'Peak-speed change per repeat', '%', G_VAR,
                   'Change of the movement-band peak speed from one repeat to the next (log-linear fit over the '
                   'first left-hand upright natural reaches): a progressive decline is the sequence effect.', LUN,
                   'linear', '+.0f', level='person', phenomenon=P_REP),
    Characteristic('seq_duration', 'Duration change per repeat', '%', G_VAR,
                   'Change of the approach duration from one repeat to the next (log-linear fit).', LUN, 'linear',
                   '+.0f', level='person', phenomenon=P_REP),
    # speed scaling (person-level)
    Characteristic('speed_ratio', 'Slow-to-fast duration ratio', '', G_MOD,
                   'Mean slow duration over mean fast duration (left hand, upright): how much the person changes '
                   'the reach when asked to go slow or fast.', LUA, 'log', '.2f', level='person', phenomenon=P_SCALE,
                   key_set=True),
    Characteristic('active_speed_ratio', 'Slow-to-fast active-time ratio', '', G_MOD,
                   'Mean slow over mean fast active movement time (left hand, upright): the speed scaling without '
                   'the wait for stillness that ends the segmented approach.', LUA, 'log', '.2f', level='person',
                   phenomenon=P_SCALE, post_hoc=True),
    # hand signature
    Characteristic('fhy_hand', 'First-half yaw, hand-consistent', 'deg', G_SIG,
                   'Turn about gravity in the first half of the reach, mirror-normalised so that positive is the '
                   'direction typical of the hand used: the hand evidence of the pilot.', LUA, 'linear', '.1f',
                   phenomenon=P_EARLY, key_set=True, multivariate=True),
    Characteristic('yaw_peak_hand', 'Early yaw peak, hand-consistent', 'deg', G_SIG,
                   'The largest mirror-normalised turn about gravity reached in the first half of the approach (the '
                   'first-half yaw reads only its value at the half, after which a reach can turn back).', LUA,
                   'linear', '.1f', phenomenon=P_EARLY, post_hoc=True),
    Characteristic('sign_typical', 'Typical first-half yaw sign', 'share', G_SIG,
                   'Share of trials whose first-half yaw turns in the typical direction for the hand.', LUA,
                   'linear', '.2f', phenomenon=P_EARLY, report_z=False),
    Characteristic('net_yaw_hand', 'Net yaw, hand-consistent', 'deg', G_SIG,
                   'Turn about gravity over the whole reach, mirror-normalised.', LUA, 'linear', '.1f',
                   phenomenon=P_NETYAW),
    Characteristic('net_sign_typical', 'Typical net-yaw sign', 'share', G_SIG,
                   'Share of trials whose net yaw turns in the typical direction for the hand (net yaw separates the '
                   'hands less well than the first-half yaw).', LUA, 'linear', '.2f', phenomenon=P_NETYAW,
                   post_hoc=True, report_z=False),
    # task behaviour
    Characteristic('attempts_per_accepted', 'Attempts per accepted trial', '', G_TASK,
                   'Attempts recorded per accepted trial over the person\'s included sessions (rejected attempts: no '
                   'movement in time, still moving at the time limit, ...).', 'all_trials', 'log', '.2f',
                   level='person', phenomenon=P_TASK),
    Characteristic('nose_no_return', 'No return within the recording', 'share', G_TASK,
                   'Share of trials in which no return of the arm (100 ms above 30 deg/s after the end) was recorded '
                   'before the recording stopped: the time at the nose is then censored, so the share, not a time, '
                   'is compared.', LUA, 'linear', '.2f', phenomenon=P_RETURN, report_z=False),
)
BY_KEY: dict[str, Characteristic] = {c.key: c for c in CHARACTERISTICS}
TRIAL_KEYS = tuple(c.key for c in CHARACTERISTICS if c.level == 'trial')
#: Person-level characteristics computed from the across-repeat series of a stratum.
REPEAT_KEYS = ('cv_duration', 'cv_peak_speed_mv', 'seq_peak_speed_mv', 'seq_duration')
#: The secondary (larger, hand-mixed) reference of the across-repeat measures.
REPEAT_SECONDARY = 'upright_natural_both'


def lc(label: str) -> str:
    """A label in running text: first letter lower-case unless the label starts with an acronym; units and
    acronyms inside (Hz, SPARC, CV) are kept."""
    return label[:1].lower() + label[1:] if label[1:2].islower() else label


# ───────────────────────────── person-level values ─────────────────────────────

ORDER_COLS = ('session_order', 'trial_index', 'attempt')


def visit_order(t: pd.DataFrame) -> pd.DataFrame:
    """Rows in visit order: the person's sessions in time, trial order, attempt."""
    return t.sort_values(['person_id', *ORDER_COLS], kind='stable')


def person_trial_values(t: pd.DataFrame, key: str, stratum: str, *, first_k: int | None = None) -> pd.DataFrame:
    """Person-level value of a trial-level characteristic: the mean of the person's trials of the stratum (NaN
    trials left out), or of their first ``first_k`` trials in visit order. Columns: person_id, value, n_trials."""
    sub = visit_order(t[STRATA[stratum].select(t)])
    rows = []
    for pid, g in sub.groupby('person_id', sort=True):
        g = g if first_k is None else g.head(first_k)
        v = pd.to_numeric(g[key], errors='coerce').dropna()
        rows.append({'person_id': pid, 'value': float(v.mean()) if len(v) else math.nan, 'n_trials': int(len(v))})
    return pd.DataFrame(rows, columns=['person_id', 'value', 'n_trials'])


def _log_slope_pct(y: np.ndarray) -> float:
    """Percent change per repeat of a positive series from a log-linear least-squares fit (NaN with < 2 values)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size < 2:
        return math.nan
    b = np.polyfit(np.arange(y.size, dtype=float), np.log(y), 1)[0]
    return 100.0 * (math.exp(b) - 1.0)


def _cv(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    return float(np.std(y, ddof=1) / np.mean(y)) if y.size >= 2 and np.mean(y) > 0 else math.nan


def person_level_values(t: pd.DataFrame, key: str, *, k_repeats: int, attempts: pd.DataFrame | None = None,
                        stratum: str | None = None) -> pd.DataFrame:
    """Person-level characteristics (``level='person'``). ``k_repeats``: the number of repeats used for the
    across-repeat measures (the exploratory person's count, so every person is judged on as many repeats);
    ``attempts``: every attempt of the included sessions (for attempts per accepted trial); ``stratum``: the
    stratum of the across-repeat series (default: the characteristic's own)."""
    c = BY_KEY[key]
    rows = []
    if key in REPEAT_KEYS:
        col = 'duration_s' if key.endswith('duration') else 'peak_speed_mv'
        fn = _cv if key.startswith('cv') else _log_slope_pct
        sub = visit_order(t[STRATA[stratum or c.stratum].select(t)])
        for pid, g in sub.groupby('person_id', sort=True):
            y = pd.to_numeric(g[col], errors='coerce').to_numpy()[:k_repeats]
            n = int(np.isfinite(y).sum())
            rows.append({'person_id': pid, 'value': fn(y) if n >= k_repeats else math.nan, 'n_trials': n})
    elif key in ('speed_ratio', 'active_speed_ratio'):
        col = 'duration_s' if key == 'speed_ratio' else 'active_s'
        for pid, g in t.groupby('person_id', sort=True):
            slow = pd.to_numeric(g.loc[STRATA['left_upright_slow'].select(g), col], errors='coerce').dropna()
            fast = pd.to_numeric(g.loc[STRATA['left_upright_fast'].select(g), col], errors='coerce').dropna()
            v = float(slow.mean() / fast.mean()) if len(slow) and len(fast) else math.nan
            rows.append({'person_id': pid, 'value': v, 'n_trials': int(len(slow) + len(fast))})
    elif key == 'attempts_per_accepted':
        if attempts is None:
            raise ValueError('attempts_per_accepted needs the attempt table')
        a = attempts[attempts['session_included'].astype(bool)]
        for pid, g in a.groupby('person_id', sort=True):
            acc = int(g['accepted'].astype('boolean').fillna(False).astype(bool).sum())
            rows.append({'person_id': pid, 'value': len(g) / acc if acc else math.nan, 'n_trials': int(len(g))})
    else:
        raise KeyError(key)
    return pd.DataFrame(rows, columns=['person_id', 'value', 'n_trials'])


# ───────────────────────────── single-case statistics ─────────────────────────────


def robust_scale(ref: np.ndarray) -> tuple[float, float]:
    """(median, scale): scale = 1.4826 MAD, or 1.2533 x the mean absolute deviation from the median when the MAD is
    0 (a characteristic where most persons share one value, e.g. counts), NaN when that is 0 too."""
    ref = np.asarray(ref, dtype=float)
    med = float(np.median(ref))
    mad = 1.4826 * float(np.median(np.abs(ref - med)))
    if mad <= 0:
        mad = 1.2533 * float(np.mean(np.abs(ref - med)))
    return med, (mad if mad > 0 else math.nan)


def robust_z(ref: np.ndarray, x: float) -> float:
    med, sc = robust_scale(ref)
    return (x - med) / sc if math.isfinite(sc) else math.nan


def rank_p(ref: np.ndarray, x: float) -> tuple[float, float, float]:
    """(two-sided, upper, lower) rank p-values of x against the n reference values: upper = (1 + #{ref >= x}) /
    (n + 1), lower = (1 + #{ref <= x}) / (n + 1), two-sided = min(1, 2 min(upper, lower)). With n reference persons
    the two-sided p can never fall below 2 / (n + 1)."""
    ref = np.asarray(ref, dtype=float)
    n = ref.size
    up = (1 + int(np.sum(ref >= x))) / (n + 1)
    lo = (1 + int(np.sum(ref <= x))) / (n + 1)
    return min(1.0, 2 * min(up, lo)), up, lo


def crawford_howell(ref: np.ndarray, x: float) -> tuple[float, float]:
    """(t, two-sided p) of the Crawford & Howell (1998) test of a single case against a small control sample:
    t = (x - mean) / (sd sqrt((n + 1) / n)), df = n - 1 (assumes normally distributed control values)."""
    ref = np.asarray(ref, dtype=float)
    n = ref.size
    sd = float(np.std(ref, ddof=1)) if n > 1 else math.nan
    if not (n > 2 and sd > 0 and math.isfinite(x)):
        return math.nan, math.nan
    tv = (x - float(np.mean(ref))) / (sd * math.sqrt((n + 1) / n))
    return float(tv), float(2 * SS.t.sf(abs(tv), df=n - 1))


STAT_KEYS = ('ref_median', 'ref_q1', 'ref_q3', 'ref_min', 'ref_max', 'percentile', 'robust_z', 'robust_z_linear',
             'z_loo_min', 'z_loo_max', 'p_rank', 'p_upper', 'p_lower', 'ch_t', 'ch_p', 'ch_p_alt', 'n_more_extreme',
             'p_floor', 'n_case_trials_beyond', 'side')


def compare(ref_values: Sequence[float], x: float, scale: str = 'linear',
            case_trials: Sequence[float] | None = None) -> dict[str, Any]:
    """Every statistic of one comparison (NaN reference values are dropped; fewer than ``MIN_REF`` reference persons:
    no statistic). ``scale`` is the declared scale of the robust z and the Crawford-Howell test (``log`` applies when
    every value is positive; ``z_scale`` / ``ch_scale`` say which was used); ``robust_z_linear`` and ``ch_p_alt`` are
    the other scale. ``z_loo_min`` / ``z_loo_max``: the robust z with each reference person left out in turn.
    ``case_trials``: the case's own trial values, of which ``n_case_trials_beyond`` lie beyond the reference
    persons' range on the case's side (``side``: 'above' or 'below' the reference median)."""
    ref = np.asarray([v for v in ref_values if v is not None and math.isfinite(v)], dtype=float)
    n = ref.size
    ct = np.asarray([v for v in (case_trials if case_trials is not None else []) if v is not None and math.isfinite(v)],
                    dtype=float)
    out: dict[str, Any] = {'n_ref': n, 'pd_value': x, 'n_case_trials_valid': int(ct.size)}
    if n < MIN_REF or x is None or not math.isfinite(x):
        return out | {k: math.nan for k in STAT_KEYS} | {'side': '', 'z_scale': '', 'ch_scale': ''}
    med = float(np.median(ref))
    below, equal = int(np.sum(ref < x)), int(np.sum(ref == x))
    p2, up, lo = rank_p(ref, x)
    use_log = scale == 'log' and bool(np.all(ref > 0)) and x > 0
    tr = (lambda a: np.log(a)) if use_log else (lambda a: np.asarray(a, dtype=float))
    rz, xz = tr(ref), float(tr(np.array([x]))[0])
    z = robust_z(rz, xz)
    loo = [robust_z(np.delete(rz, i), xz) for i in range(n)]
    loo = [v for v in loo if math.isfinite(v)]
    tv, pv = crawford_howell(rz, xz)
    can_log = bool(np.all(ref > 0)) and x > 0
    if use_log:
        _, p_alt = crawford_howell(ref, x)
    elif can_log:
        _, p_alt = crawford_howell(np.log(ref), math.log(x))
    else:
        p_alt = math.nan
    side = 'above' if x > med else ('below' if x < med else '')
    if side == 'above':
        beyond_trials = int(np.sum(ct > ref.max()))
    elif side == 'below':
        beyond_trials = int(np.sum(ct < ref.min()))
    else:
        beyond_trials = 0
    return out | {
        'ref_median': med, 'ref_q1': float(np.percentile(ref, 25)), 'ref_q3': float(np.percentile(ref, 75)),
        'ref_min': float(ref.min()), 'ref_max': float(ref.max()),
        'percentile': 100.0 * (below + 0.5 * equal) / n,
        'robust_z': z, 'robust_z_linear': robust_z(ref, x), 'z_scale': 'log' if use_log else 'linear',
        'z_loo_min': min(loo, key=abs) if loo else math.nan, 'z_loo_max': max(loo, key=abs) if loo else math.nan,
        'p_rank': p2, 'p_upper': up, 'p_lower': lo, 'p_floor': 2.0 / (n + 1),
        # persons at least as extreme as x on its side (0 = beyond every reference person)
        'n_more_extreme': int(np.sum(ref >= x)) if x >= med else int(np.sum(ref <= x)),
        'ch_t': tv, 'ch_p': pv, 'ch_scale': 'log' if use_log else 'linear', 'ch_p_alt': p_alt,
        'side': side, 'n_case_trials_beyond': beyond_trials,
    }


def bh(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p-values stay NaN and do not count towards m)."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return q
    order = ok[np.argsort(p[ok], kind='stable')]
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


# ───────────────────────────── the null reference of the "beyond all" count ─────────────────────────────


def _beyond(x: float, others: np.ndarray) -> bool:
    o = others[np.isfinite(others)]
    if not math.isfinite(x) or o.size < MIN_REF - 1:
        return False
    return bool(x > o.max() or x < o.min())


def beyond_all_counts(M: pd.DataFrame, case: str, phenomena: dict[str, str],
                      M_own: pd.DataFrame | None = None) -> dict[str, Any]:
    """How many characteristics (columns of ``M``; rows = persons, the case included) put each person beyond every
    other reference person, and how many phenomena that covers. Each reference person is compared with the other
    reference persons (n - 1); the case with all n, and also with every (n - 1)-subset. ``M_own`` (optional): the
    value a reference person is judged on (e.g. the mean of their first k trials, as many as the case made, so that
    their values are as noisy as the case's), compared with the others' ``M`` values; missing entries fall back to
    ``M``. Under exchangeability the case's expected count is sum_k 2 / (n_k + 1). The case's rank p is (1 + #{reference
    persons with a count >= the case's}) / (n + 1): conservative, since the case faces one more person than they do."""
    refs = [p for p in M.index if p != case]
    keys = list(M.columns)
    own = M if M_own is None else M_own.reindex(index=M.index, columns=keys).fillna(M)

    def count(j: str, others: list[str]) -> tuple[int, int, list[str]]:
        ks = [k for k in keys if _beyond(float(own.at[j, k]), M.loc[others, k].to_numpy(dtype=float))]
        return len(ks), len({phenomena[k] for k in ks}), ks

    per = []
    for j in refs:
        nc, nphen, ks = count(j, [p for p in refs if p != j])
        per.append({'person_id': j, 'group': 'reference', 'n_characteristics': nc, 'n_phenomena': nphen,
                    'keys': ks})
    nc, nphen, ks = count(case, refs)
    per.append({'person_id': case, 'group': 'exploratory', 'n_characteristics': nc, 'n_phenomena': nphen, 'keys': ks})
    sub = [count(case, [p for p in refs if p != j])[:2] for j in refs]
    n_k = M.loc[refs].notna().sum()
    has = M.loc[case].notna()
    expected = float(sum(2.0 / (n_k[k] + 1) for k in keys if has[k] and n_k[k] >= MIN_REF))
    rc = np.array([r['n_characteristics'] for r in per[:-1]])
    rp = np.array([r['n_phenomena'] for r in per[:-1]])
    return {
        'keys': keys, 'per_person': per, 'n_ref': len(refs),
        'case_n_characteristics': nc, 'case_n_phenomena': nphen, 'case_keys': ks,
        'case_subset_n_characteristics': {'mean': float(np.mean([s[0] for s in sub])), 'min': int(min(s[0] for s in sub)),
                                          'max': int(max(s[0] for s in sub))},
        'expected_case_n_characteristics': expected,
        'ref_n_characteristics': {'median': float(np.median(rc)), 'mean': float(np.mean(rc)), 'max': int(rc.max())},
        'ref_n_phenomena': {'median': float(np.median(rp)), 'mean': float(np.mean(rp)), 'max': int(rp.max())},
        'n_ref_ge_case_characteristics': int(np.sum(rc >= nc)), 'n_ref_ge_case_phenomena': int(np.sum(rp >= nphen)),
        'p_rank_characteristics': (1 + int(np.sum(rc >= nc))) / (len(refs) + 1),
        'p_rank_phenomena': (1 + int(np.sum(rp >= nphen))) / (len(refs) + 1),
        'top_reference': max(per[:-1], key=lambda r: (r['n_characteristics'], r['n_phenomena']))['person_id'],
    }


# ───────────────────────────── multivariate view ─────────────────────────────


def declared(M: pd.DataFrame, use: bool = True) -> pd.DataFrame:
    """Each column on its declared scale (log where declared and every value is positive); ``use=False``: as is."""
    out = M.astype(float).copy()
    for k in out.columns:
        v = out[k]
        if use and BY_KEY[k].scale == 'log' and bool((v.dropna() > 0).all()):
            out[k] = np.log(v)
    return out


def _robust_z(M: np.ndarray, centre_from: np.ndarray) -> np.ndarray:
    med = np.nanmedian(centre_from, axis=0)
    sc = np.array([robust_scale(centre_from[:, j][np.isfinite(centre_from[:, j])])[1] for j in range(M.shape[1])])
    return (M - med) / sc


def mahalanobis_lw(ref: np.ndarray, x: np.ndarray) -> float:
    """Mahalanobis distance of x from the reference rows with a Ledoit-Wolf shrinkage covariance (centre = the
    coordinate-wise median of the reference, which is robust-standardised already)."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(ref)
    d = x - np.median(ref, axis=0)
    return float(math.sqrt(max(0.0, d @ lw.precision_ @ d)))


def multivariate(ref: pd.DataFrame, x: pd.Series, declared_scales: bool = True) -> dict[str, Any]:
    """PCA map and Mahalanobis test of one exploratory person (``x``: key -> value) against the reference persons
    (``ref``: rows = persons, columns = keys), every column on its declared scale (``declared_scales=False``: all
    linear, the sensitivity analysis). Missing reference values are set to the reference median (robust z 0) and
    counted. Every reference person's own distance is computed with that person left out."""
    keys = list(ref.columns)
    both = declared(pd.concat([ref, x.to_frame().T]), declared_scales)
    R = both.iloc[:-1].to_numpy(dtype=float)
    xv = both.iloc[-1][keys].to_numpy(dtype=float)
    n_missing = int(np.isnan(R).sum())
    med = np.nanmedian(R, axis=0)
    R = np.where(np.isnan(R), med, R)
    Z = _robust_z(R, R)
    zx = _robust_z(xv[None, :], R)[0]
    # PCA on the reference persons (robust-z, centred at their median)
    mu = np.median(Z, axis=0)
    U, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    # deterministic sign: the largest loading of each component is positive
    sgn = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    Vt = Vt * sgn[:, None]
    var = S ** 2 / np.sum(S ** 2)
    scores_ref = (Z - mu) @ Vt.T
    score_x = (zx - mu) @ Vt.T
    d_x = mahalanobis_lw(Z, zx)
    d_ref = []
    for i in range(R.shape[0]):
        others = np.delete(R, i, axis=0)
        Zo = _robust_z(others, others)
        zi = _robust_z(R[i][None, :], others)[0]
        d_ref.append(mahalanobis_lw(Zo, zi))
    d_ref = np.asarray(d_ref)
    p_up = (1 + int(np.sum(d_ref >= d_x))) / (len(d_ref) + 1)
    return {
        'keys': keys, 'scales': [('log' if declared_scales and BY_KEY[k].scale == 'log'
                                  and bool((ref[k].dropna() > 0).all()) and float(x[k]) > 0 else 'linear')
                                 for k in keys],
        'persons': list(ref.index), 'n_missing_imputed': n_missing,
        'explained_variance': var.tolist(), 'loadings': Vt[:2].tolist(),
        'scores_ref': scores_ref[:, :2].tolist(), 'score_x': score_x[:2].tolist(),
        'robust_z_x': zx.tolist(), 'mahalanobis_x': d_x, 'mahalanobis_ref_loo': d_ref.tolist(),
        'p_mahalanobis': p_up, 'p_floor': 1.0 / (len(d_ref) + 1),
        'n_ref_beyond': int(np.sum(d_ref >= d_x)),
    }


def multivariate_loco(ref: pd.DataFrame, x: pd.Series, declared_scales: bool = True) -> list[dict[str, Any]]:
    """The Mahalanobis test with each characteristic left out in turn: does the result rest on one of them?"""
    rows = []
    for k in ref.columns:
        keep = [c for c in ref.columns if c != k]
        mv = multivariate(ref[keep], x[keep], declared_scales)
        rows.append({'left_out': k, 'mahalanobis_x': mv['mahalanobis_x'], 'n_ref_beyond': mv['n_ref_beyond'],
                     'p_mahalanobis': mv['p_mahalanobis']})
    return rows
