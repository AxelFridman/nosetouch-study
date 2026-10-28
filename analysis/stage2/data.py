"""Stage 2 data layer: the curated trial table, the exclusion cascade and the array cache.

    from stage2.data import load_cache
    d = load_cache()                    # the primary set (default), arrays aligned with d.trials
    d.approach.shape                    # (N, C, 64): time-normalised oracle approach, C = len(d.channels)
    d.y, d.groups                       # 1 = right hand; person_id P01..P19 (the ONLY grouping key)
    nat = load_cache(sets=('natural_use',))     # one free-choice trial per person (headline natural use)

Build (``python -m stage2.scripts.build_cache``, about 30 s): :func:`build_cache` loads the frozen
snapshot with ``motionlab.io.load_dataset`` (schema validation, deduplication, norm_v2 per session), joins
the researcher's curation (session -> person_id P01..P19, include / exclude), builds one row per trial
ATTEMPT, standardises every complete attempt of the included sessions (``stage2.standardize``), adds the
kept flags of docs/02 T8-T12 (``FLAG_COLUMNS``) and the predictor gates of docs/09 D5 (``GATE_COLUMNS``),
applies the exclusion cascade (``RULES``, then N1: one natural-use trial per person) and writes the cache
(git-ignored, under analysis/data/stage2_cache/) with the code fingerprint in meta.json.

Grouping: always by ``person_id``. ``participant_code`` (the app's P-XXXXXX) is kept only to trace a row
back to the raw records: one code holds two persons (P01 and P13 used the same phone) and one person
has two codes (P14), so grouping by it would leak a person into both sides of a split
(``cv.run_experiment`` refuses it: :func:`canonical_trial_map`).

Privacy: no names anywhere. The curation file's free-text reasons are replaced by fixed codes, its
``rules`` field is never read, the private name file is never opened (``config.assert_not_private``),
person-level demographics live only in the person table (never in the trial table, so they cannot slip
into a model), and ``user_agent`` is dropped. Report tables use pseudonymous labels (``session_label``
P14-s3, ``trial_key`` P14-s3-t07a1) and no quasi-identifier (``QUASI_IDENTIFIER_COLUMNS``,
:func:`report_sessions`); the cache keeps the backend keys for tracing.

Oracle approach window (``onset_ms`` .. ``end_ms`` of the trial table): the app's own online_v1
segmentation as stored with each trial (identical to ``timing.onset_ms`` / ``timing.end_ms``; exactly
reproducible with ``motionlab.segment.rerun_online_v1``): onset back-dated to the end of the stillness,
end back-dated to the start of the final stillness at the nose (the end cue follows 250 ms later). It
exists for every complete attempt; offline_v1 (``off_onset_ms`` / ``off_end_ms``, kept for ablations)
finds no end in 19 % of the complete street attempts because the return of the arm follows the end cue
after a 500 ms post-roll. README.md gives the agreement numbers.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import subprocess
import time
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motionlab.conventions import rotation_from_gravity
from motionlab.io import Dataset, load_dataset
from motionlab.qa import CLIP_ACC_MS2, CLIP_GYRO_DPS, stream_checks
from motionlab.report import write_csv
from motionlab.segment import offline_v1_from_record

from . import config as C
from .guard import assert_outside_fit_predict
from .standardize import CHANNEL_INFO, CHANNELS, T_FIXED, standardize_trial

# ───────────────────────────── curation ─────────────────────────────

#: Fixed, name-free wording of the researcher's exclusion reasons (the curation file's own text starts
#: with a first name, so it is never copied). Matched on the reason text, which is otherwise discarded.
EXCLUDE_CODES = {
    'superseded': 'researcher: an earlier, incomplete run of a person whose later run is kept',
    'failed': 'researcher: the recordings of this person did not work',
    'other': 'researcher decision',
}


def _reason_code(reason: str | None) -> str | None:
    if not reason:
        return None
    r = reason.lower()
    if 'later' in r or 'first run' in r:
        return 'superseded'
    if 'worked' in r or 'did not work' in r:
        return 'failed'
    return 'other'


def load_curation(path: str | Path = C.CURATION_FILE) -> pd.DataFrame:
    """The researcher's session curation: session_id, person_id, include, exclude_code, exclude_reason.

    Only ``sessions[*]`` is read; the free-text ``rules`` field and the private name file are never used.
    """
    p = C.assert_not_private(path)
    doc = json.loads(Path(p).read_text(encoding='utf-8'))
    rows = []
    for s in doc['sessions']:
        code = None if s.get('include') else _reason_code(s.get('exclude_reason'))
        rows.append({
            'session_id': s['session_id'], 'person_id': s.get('person_id'), 'include': bool(s.get('include')),
            'exclude_code': code, 'exclude_reason': EXCLUDE_CODES.get(code) if code else None,
        })
    cur = pd.DataFrame(rows)
    if cur['session_id'].duplicated().any():
        raise ValueError('curation: duplicated session_id')
    bad = cur['include'] & cur['person_id'].isna()
    if bad.any():
        raise ValueError('curation: an included session has no person_id')
    # Excluded sessions without a person_id: the researcher named one person per reason text, so sessions
    # sharing a reason text are one person (X1, X2, ...). Used only to count persons in the exclusion flow.
    unassigned = [s for s in doc['sessions'] if not s.get('person_id')]
    keys = {r: f'X{i + 1}' for i, r in enumerate(dict.fromkeys(s.get('exclude_reason') or s['session_id']
                                                               for s in unassigned))}
    ph = {s['session_id']: keys[s.get('exclude_reason') or s['session_id']] for s in unassigned}
    cur['person_key'] = cur['person_id'].fillna(cur['session_id'].map(ph))
    return cur


# ───────────────────────────── tables ─────────────────────────────

#: Session columns safe to share (no name, no user agent).
SESSION_COLUMNS = [
    'session_id', 'participant_id', 'started_at', 'ended_at', 'status', 'app_version', 'protocol_version',
    'schema_version', 'preset', 'review_mode', 'posture', 'first_hand', 'language', 'consent_version', 'simulated',
    'platform', 'engine', 'browser', 'os_version', 'device_model', 'in_app_browser', 'standalone', 'screen_w',
    'screen_h', 'dpr', 'measured_motion_hz', 'calibration_method', 'gravity_sign', 'gravity_sign_source',
    'gravity_sign_warning', 'gravity_votes_for', 'gravity_votes_against', 'rr_mapping', 'rr_units', 'rr_source',
    'rr_verified', 'rr_closure_deg', 'rr_closure_runner_up', 'rr_closure_runner_up_deg', 'rr_closure_trials',
    'n_planned_trials', 'n_planned_required', 'n_attempts', 'n_accepted', 'n_trials_loaded', 'n_accepted_loaded',
    # the participant snapshot of THIS session (demographics are per session: one P-code holds two persons)
    'age_years', 'sex', 'handedness_self', 'phone_hand', 'height_cm',
]
PERSON_FIELDS = ('age_years', 'sex', 'handedness_self', 'phone_hand', 'height_cm')

#: Trial columns kept from ``Dataset.trials`` (person-level demographics deliberately left out).
TRIAL_COLUMNS = [
    'trial_id', 'session_id', 'participant_id', 'block_index', 'block_kind', 'block_tier', 'block_optional',
    'trial_index', 'trial_in_block', 'attempt', 'n_attempts_trial', 'is_last_attempt', 'created_at', 'simulated',
    'schema_version', 'app_version', 'hand', 'hand_source', 'orientation', 'speed', 'start_position', 'variant',
    'posture', 'rep', 'instructed_rotation', 'status', 'accepted', 'participant_flag', 'note', 't0_epoch_ms',
    'still_ms', 'go_ms', 'onset_ms', 'end_ms', 'end_detected_ms', 'stop_ms', 'first_touch_ms', 'foreperiod_ms',
    'false_starts', 'reaction_ms', 'movement_ms', 'return_onset_ms', 'return_end_ms', 'return_observed',
    'return_delay_ms', 'return_duration_ms', 'seg_method', 'acc_source', 'onset_index', 'end_index',
    'peak_gyro_dps', 'peak_acc_ms2', 'rotation_start', 'rotation_end', 'tilt_change_deg', 'orientation_match',
    'n_motion', 'motion_hz', 'motion_dt_ms_median', 'motion_dt_ms_max', 'gaps_over_50ms', 'acc_available',
    'rotation_available', 'hidden_during', 'warnings', 'n_touches', 'n_touch_starts', 'platform', 'engine',
    'device_model', 'calibration_method', 'review_mode', 'gravity_sign', 'gravity_sign_source', 'rr_mapping',
    'rr_verified', 'dup_attempt_key', 'multi_accepted', 'y_right',
]
#: Columns renamed for clarity (the stored online_v1 peaks are of its own EMA-smoothed magnitudes).
TRIAL_RENAME = {'participant_id': 'participant_code', 'block_tier': 'tier', 'peak_gyro_dps': 'online_peak_gyro_dps',
                'peak_acc_ms2': 'online_peak_acc_ms2', 'reaction_ms': 'online_reaction_ms',
                'movement_ms': 'online_movement_ms'}


#: Session columns of report-facing tables: no app code (P-XXXXXX, the backend's deletion key), no session
#: id, no timestamp, no OS / browser version, no screen geometry (quasi-identifiers that would link P01..P19
#: to a person for anyone with backend access). The cache keeps them for tracing.
REPORT_SESSION_COLUMNS = [
    'person_id', 'session_label', 'session_order', 'include', 'exclude_code', 'status', 'app_version',
    'protocol_version', 'schema_version', 'preset', 'review_mode', 'posture', 'first_hand', 'simulated', 'platform',
    'engine', 'device_model', 'measured_motion_hz', 'calibration_method', 'gravity_sign', 'gravity_sign_source',
    'gravity_sign_warning', 'gravity_votes_for', 'gravity_votes_against', 'rr_mapping', 'rr_units', 'rr_source',
    'rr_verified', 'rr_closure_deg', 'rr_closure_runner_up', 'rr_closure_runner_up_deg', 'rr_closure_trials',
    'n_planned_trials', 'n_planned_required', 'n_attempts', 'n_accepted', 'n_trials_loaded', 'n_accepted_loaded',
]
#: Never in a report-facing table (checked by a test).
QUASI_IDENTIFIER_COLUMNS = ('participant_code', 'participant_codes', 'participant_id', 'session_id', 'session_ids',
                            'trial_id', 'started_at', 'ended_at', 'first_started_at', 'created_at', 't0_epoch_ms',
                            'os_version', 'browser', 'user_agent', 'screen_w', 'screen_h', 'dpr', 'nickname')


def session_table(ds: Dataset, cur: pd.DataFrame) -> pd.DataFrame:
    s = ds.sessions.copy()
    for c in SESSION_COLUMNS:
        if c not in s.columns:
            s[c] = None
    s = s[SESSION_COLUMNS].rename(columns={'participant_id': 'participant_code'})
    s = s.merge(cur, how='left', on='session_id')
    if s['include'].isna().any():
        missing = s.loc[s['include'].isna(), 'session_id'].tolist()
        raise ValueError(f'sessions missing from the curation file: {missing}')
    s['include'] = s['include'].astype(bool)
    # pseudonymous session label for report tables: <person>-s<k>, k = order of the person's sessions in time
    # (excluded sessions count too; a person without id is X1, X2, ...)
    s = s.sort_values(['person_key', 'started_at'], na_position='last').reset_index(drop=True)
    s['session_order'] = s.groupby('person_key').cumcount() + 1
    s['session_label'] = s['person_key'].astype(str) + '-s' + s['session_order'].astype(str)
    return s.sort_values(['person_id', 'started_at'], na_position='last').reset_index(drop=True)


def report_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """The session table without quasi-identifiers (for tables meant for the report)."""
    cols = [c for c in REPORT_SESSION_COLUMNS if c in sessions.columns]
    out = sessions[cols].copy()
    out['person_id'] = out['person_id'].fillna(sessions['person_key'])
    return out.sort_values(['person_id', 'session_order']).reset_index(drop=True)


def trial_table(ds: Dataset, sessions: pd.DataFrame) -> pd.DataFrame:
    t = ds.trials.copy()
    for c in TRIAL_COLUMNS:
        if c not in t.columns:
            t[c] = None
    t = t[TRIAL_COLUMNS].rename(columns=TRIAL_RENAME)
    s = sessions[['session_id', 'person_id', 'person_key', 'include', 'protocol_version', 'status', 'started_at',
                  'session_order', 'session_label']].rename(
        columns={'include': 'session_included', 'status': 'session_status', 'started_at': 'session_started_at'})
    t = t.merge(s, how='left', on='session_id')
    t['session_included'] = t['session_included'].fillna(False).astype(bool)
    # labels: 1 = right hand; reported hands are labels of the natural-use set only
    t['y_right'] = t['hand'].map({'right': 1, 'left': 0}).astype('Int64')
    t['hand_source'] = t['hand_source'].fillna('instructed')
    t['tier'] = pd.array(pd.to_numeric(t['tier'], errors='coerce'), dtype='Int64')
    t['protocol'] = np.where(t['app_version'].astype(str).str.startswith('0.1'), 'v1_quick', 'v2_street')
    # pseudonymous trial key for report tables (the trial_id is a backend key): <session label>-t<index>a<attempt>
    ti = pd.to_numeric(t['trial_index'], errors='coerce')
    at = pd.to_numeric(t['attempt'], errors='coerce')
    t['trial_key'] = (t['session_label'].astype(str) + '-t' + ti.map(lambda v: f'{int(v):02d}' if pd.notna(v) else 'NA')
                      + 'a' + at.map(lambda v: f'{int(v)}' if pd.notna(v) else 'NA'))
    if t['trial_key'].duplicated().any():
        raise ValueError('trial_key is not unique')
    # time order: person, the person's sessions in time, trial order, attempt
    return t.sort_values(['person_key', 'session_order', 'trial_index', 'attempt'],
                         na_position='last').reset_index(drop=True)


def person_table(sessions: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """One row per person (included sessions only): profile, devices, counts and tiers reached.

    Profile fields come from each session's own participant snapshot; when a person's sessions disagree
    the most frequent value is kept and ``profile_consistent`` is False."""
    s = sessions[sessions['include']].copy()
    rows = []
    for pid, g in s.groupby('person_id', sort=True):
        g = g.sort_values('started_at')
        prof: dict[str, Any] = {}
        consistent = True
        for f in PERSON_FIELDS:
            vals = g[f].dropna()
            if vals.nunique() > 1:
                consistent = False
            prof[f] = vals.mode().iloc[0] if len(vals) else None
        tr = trials[(trials['person_id'] == pid) & trials['session_included']]
        acc = tr[tr['accepted'].fillna(False).astype(bool)]
        rows.append({
            'person_id': pid, 'n_sessions': len(g), 'session_ids': ';'.join(g['session_id']),
            'session_labels': ';'.join(g['session_label']),
            'participant_codes': ';'.join(sorted(set(g['participant_code']))),
            'platform': ';'.join(sorted(set(g['platform']))), 'device_models': ';'.join(sorted(set(g['device_model']))),
            'app_versions': ';'.join(sorted(set(g['app_version']))),
            'protocols': ';'.join(sorted(set(g['protocol_version']))),
            **prof, 'profile_consistent': consistent,
            'session_statuses': ';'.join(g['status']), 'first_started_at': g['started_at'].iloc[0],
            'n_attempts': len(tr), 'n_accepted': len(acc),
            'max_tier_accepted': (int(acc['tier'].max()) if acc['tier'].notna().any() else None),
            'bonus_attempted': bool((tr['tier'] == 4).any()),
        })
    return pd.DataFrame(rows)


# ───────────────────────────── oracle bounds, QC and arrays ─────────────────────────────


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def oracle_bounds(record: dict[str, Any]) -> tuple[float | None, float | None]:
    """(onset_ms, end_ms) of the oracle approach: the stored online_v1 segment (see module docstring)."""
    tim = record.get('timing') or {}
    on, en = _f(tim.get('onset_ms')), _f(tim.get('end_ms'))
    if on is None or en is None or en <= on:
        return None, None
    return on, en


@dataclass
class _Built:
    qc: pd.DataFrame
    trial_ids: list[str]
    approach: np.ndarray
    fixed: np.ndarray
    fixed_mask: np.ndarray
    fixed_phase: np.ndarray
    ret: np.ndarray
    ret_ok: np.ndarray
    streams: dict[str, np.ndarray]


def standardize_all(ds: Dataset, trials: pd.DataFrame, *, progress: bool = True, **stream_kw: Any) -> _Built:
    """Oracle bounds, offline_v1 bounds, QC and every array for the complete attempts of included sessions."""
    todo = trials[trials['session_included'] & (trials['status'] == 'complete')]
    qc_rows: list[dict[str, Any]] = []
    ids: list[str] = []
    A, Fx, Fm, Fp, R, Rok = [], [], [], [], [], []
    st_t, st_f, st_a, st_w, st_v, st_p, offsets = [], [], [], [], [], [], [0]
    t0 = time.time()
    for k, tid in enumerate(todo['trial_id']):
        rec = ds.record(tid, compact=True)
        nm = ds.trial_normalization(tid)
        on, en = oracle_bounds(rec)
        off = offline_v1_from_record(rec)
        raw = stream_checks(rec, engine=nm.engine)  # T10 clipping candidates over the whole raw recording
        row: dict[str, Any] = {'trial_id': tid, 'oracle_ok': on is not None,
                               'off_ok': bool(off.ok), 'off_reason': off.reason,
                               'off_onset_ms': off.onset_ms, 'off_end_ms': off.end_ms if off.ok else None,
                               'raw_n_clip_acc': raw['n_clip_acc'], 'raw_n_clip_gyro': raw['n_clip_gyro']}
        if on is None:
            row['oracle_reason'] = 'no_stored_segment'
            qc_rows.append(row)
            continue
        try:
            ta = standardize_trial(rec, nm, on, en, **stream_kw)
        except ValueError as e:
            row.update({'oracle_ok': False, 'oracle_reason': f'standardize_failed: {e}'})
            qc_rows.append(row)
            continue
        row.update({'oracle_reason': 'ok', **ta.qc, 'ret_ok': ta.ret_ok})
        qc_rows.append(row)
        ids.append(tid)
        A.append(ta.approach), Fx.append(ta.fixed), Fm.append(ta.fixed_mask), Fp.append(ta.fixed_phase)
        R.append(ta.ret), Rok.append(ta.ret_ok)
        s = ta.stream
        st_t.append(s.t), st_f.append(s.f.astype(np.float32)), st_a.append(s.acc.astype(np.float32))
        st_w.append(s.gyro.astype(np.float32)), st_v.append(s.valid), st_p.append(ta.stream_phase)
        offsets.append(offsets[-1] + s.n)
        if progress and (k + 1) % 100 == 0:
            print(f'  standardised {k + 1}/{len(todo)} attempts ({time.time() - t0:.0f} s)', flush=True)
    qc = pd.DataFrame(qc_rows)
    z = (len(CHANNELS), C.N_NORM)
    return _Built(
        qc=qc, trial_ids=ids,
        approach=np.stack(A) if A else np.zeros((0, *z), np.float32),
        fixed=np.stack(Fx) if Fx else np.zeros((0, len(CHANNELS), T_FIXED), np.float32),
        fixed_mask=np.stack(Fm) if Fm else np.zeros((0, T_FIXED), bool),
        fixed_phase=np.stack(Fp) if Fp else np.zeros((0, T_FIXED), np.int8),
        ret=np.stack(R) if R else np.zeros((0, *z), np.float32),
        ret_ok=np.asarray(Rok, dtype=bool),
        streams={'offsets': np.asarray(offsets, dtype=np.int64),
                 't': np.concatenate(st_t) if st_t else np.zeros(0),
                 'f': np.concatenate(st_f) if st_f else np.zeros((0, 3), np.float32),
                 'acc': np.concatenate(st_a) if st_a else np.zeros((0, 3), np.float32),
                 'gyro': np.concatenate(st_w) if st_w else np.zeros((0, 3), np.float32),
                 'valid': np.concatenate(st_v) if st_v else np.zeros(0, bool),
                 'phase': np.concatenate(st_p) if st_p else np.zeros(0, np.int8)},
    )


# ───────────────────────────── flags (kept) and predictor gates (evaluated) ─────────────────────────────

#: docs/02 T8-T12 and the app's screen-rotation warning: flagged, never excluded (column -> meaning).
FLAG_COLUMNS: dict[str, str] = {
    'grip_mismatch': 'T8: the grip before the onset is not the instructed one, by the app (rotation_start, '
                     "'unknown' included) or by the standardised up vector of the 300 ms before the onset",
    'seg_disagree_100ms': f'T9: offline_v1 and the stored online_v1 onsets differ by more than {C.SEG_DISAGREE_MS:.0f} ms',
    'clip_candidate': f'T10: |accg| > {CLIP_ACC_MS2 / 9.80665:.1f} g or |w| > {CLIP_GYRO_DPS:.0f} deg/s somewhere in the '
                      'raw recording (motionlab.qa.stream_checks)',
    'repeated_samples': 'T11: isolated duplicated snapshots were dropped before resampling',
    'false_start': 'T12: the hold restarted before GO (timing.false_starts > 0)',
    'screen_rotated': 'the app warned screen_rotated_during_movement (a screen rotation between GO and the end)',
}
#: Predictor gates of docs/09 D5 as evaluation columns (True = the predictor would score the attempt).
GATE_COLUMNS: dict[str, str] = {
    'gate_g1': f'G1 recording: motion rate >= {C.MIN_MOTION_HZ:.0f} Hz, no sampling gap > {C.G1_MAX_GAP_MS:.0f} ms in '
               f'[onset - {C.G1_PAD_S} s, end + {C.G1_PAD_S} s], page not hidden, gyroscope present',
    'gate_g2': f'G2 segment: {C.G2_MIN_DURATION_MS / 1000:.2f} s <= duration <= {C.G2_MAX_DURATION_MS / 1000:.0f} s and '
               f'peak |w| >= {C.MIN_PEAK_GYRO_DPS:.0f} deg/s on the oracle segment (the offline_v1 clause of G2 is '
               'not applied: offline_v1 finds no end in 19 % of street attempts, docs/09 D10 is under revision)',
}


def warn_has(d: pd.DataFrame, code: str) -> pd.Series:
    """True where the app's ``warnings`` (a string or a list of strings) contain ``code``."""
    if 'warnings' not in d.columns:
        return pd.Series(False, index=d.index)
    w = d['warnings'].map(lambda v: ' '.join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray))
                          else ('' if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)))
    return w.str.contains(code, regex=False)


def grip_from_up(up: Any, orientation: Any = None) -> str:
    """The grip ('upright', 'top_left', ...) from a mean unit up vector in the W3C device frame (sign already
    normalised: +y up when upright); ``motionlab.conventions.rotation_from_gravity`` with sign +1."""
    u = np.asarray(up, dtype=float)
    if u.shape != (3,) or not np.all(np.isfinite(u)):
        return 'unknown'
    hint = 'away' if str(orientation).endswith('_away') else 'toward'
    return rotation_from_gravity(u, 1, hint)


def add_flags_and_gates(trials: pd.DataFrame) -> pd.DataFrame:
    """Boolean flag columns (``FLAG_COLUMNS``) and gate columns (``GATE_COLUMNS``) from the QC numbers."""
    d = trials.copy()
    ok = _b(d['oracle_ok']) if 'oracle_ok' in d.columns else pd.Series(False, index=d.index)
    up = d[['up_pre_x', 'up_pre_y', 'up_pre_z']].to_numpy(dtype=float) if 'up_pre_x' in d.columns else None
    d['grip_detected'] = [grip_from_up(up[i], d['orientation'].iat[i]) if (up is not None and ok.iat[i]) else None
                          for i in range(len(d))]
    instr = d['instructed_rotation']
    rs = d['rotation_start']
    d['grip_mismatch_app'] = (instr.notna() & rs.notna() & (rs.astype(str) != instr.astype(str))).astype(bool)
    gd = d['grip_detected']
    d['grip_mismatch_gravity'] = (instr.notna() & gd.notna() & (gd.astype(str) != instr.astype(str))).astype(bool)
    d['grip_mismatch'] = d['grip_mismatch_app'] | d['grip_mismatch_gravity']
    d['seg_onset_diff_ms'] = _num(d, 'off_onset_ms') - _num(d, 'onset_ms')
    d['seg_disagree_100ms'] = (d['seg_onset_diff_ms'].abs() > C.SEG_DISAGREE_MS).astype(bool)
    d['clip_candidate'] = ((_num(d, 'raw_n_clip_acc') > 0) | (_num(d, 'raw_n_clip_gyro') > 0)
                           | warn_has(d, 'possible_acc_clipping')).astype(bool)
    d['repeated_samples'] = (_num(d, 'repeats_dropped') > 0).astype(bool)
    d['false_start'] = (_num(d, 'false_starts') > 0).astype(bool)
    d['screen_rotated'] = warn_has(d, 'screen_rotated_during_movement').astype(bool)
    dur, peak = _num(d, 'duration_ms'), _num(d, 'peak_gyro_dps')
    d['gate_g1'] = (ok & (_num(d, 'g1_n_gaps') == 0) & (_num(d, 'motion_hz') >= C.MIN_MOTION_HZ)
                    & ~_b(d['hidden_during']) & _b(d['rotation_available'], True)).astype(bool)
    d['gate_g2'] = (ok & (dur >= C.G2_MIN_DURATION_MS) & (dur <= C.G2_MAX_DURATION_MS)
                    & (peak >= C.MIN_PEAK_GYRO_DPS)).astype(bool)
    d['gate_pass'] = d['gate_g1'] & d['gate_g2']
    return d


# ───────────────────────────── exclusion cascade ─────────────────────────────


@dataclass(frozen=True)
class Rule:
    code: str
    name: str
    description: str
    fails: Any  # callable(trials) -> bool Series (True = the attempt FAILS the rule)
    divert: str | None = None  # 'natural_use': failing attempts go to that set instead of being dropped


def _b(s: pd.Series, default: bool = False) -> pd.Series:
    return s.astype('boolean').fillna(default).astype(bool)


def _num(d: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(d[c], errors='coerce') if c in d.columns else pd.Series(np.nan, index=d.index)


def _not_accepted(d: pd.DataFrame) -> pd.Series:
    return ~((d['status'] == 'complete') & _b(d['accepted']) & d['participant_flag'].isna())


def e1_reason(d: pd.DataFrame) -> pd.Series:
    """Why an attempt is not accepted (first matching reason), for the E1 breakdown."""
    out = pd.Series(None, index=d.index, dtype=object)
    status = d['status'].astype(str)
    note = d['note'].fillna('').astype(str)
    flag = d['participant_flag']
    out[status == 'no_movement'] = 'failed: no movement within the time limit after GO'
    out[(status == 'too_long') & out.isna()] = 'failed: still moving at the time limit'
    out[(status == 'aborted') & out.isna()] = 'failed: aborted by the participant'
    out[(status == 'interrupted') & out.isna()] = 'failed: interrupted (page hidden or sensors stalled)'
    out[(note == 'redo_requested') & out.isna()] = 'revoked: "redo last movement" on the pause sheet'
    out[note.str.startswith('auto_reject') & out.isna()] = 'auto-rejected: grip detected as another grip'
    out[flag.notna() & out.isna()] = 'flagged by the participant at review'
    out[(~_b(d['accepted'])) & out.isna()] = 'not confirmed at tap review (protocol 1.0.0)'
    return out


RULES: tuple[Rule, ...] = (
    Rule('E0', 'session excluded by the researcher',
         'the curation file excludes the session (an earlier incomplete run superseded by a later one; two runs '
         'of one person that did not work)', lambda d: ~d['session_included']),
    Rule('E0s', 'simulated record (S1)', 'simulator data never enter real analyses',
         lambda d: _b(d['simulated'])),
    Rule('E1', 'attempt not accepted', 'failed (no movement, too long, aborted, interrupted), revoked, '
         'auto-rejected, flagged or not confirmed (docs/02 T1-T3)', _not_accepted),
    Rule('E2', 'screen-away grip (protocol 1.0.0)', 'protocol 2.0.0 never collects *_away grips; they are '
         'synthesised from screen-toward trials instead', lambda d: d['orientation'].astype(str).str.endswith('_away')),
    Rule('E3', 'free-choice trial (hand self-reported)', 'tier 0: the hand is reported after the movement, so '
         'the label is not instructed; these trials form the separate natural-use set',
         lambda d: d['hand_source'] == 'reported', divert='natural_use'),
    Rule('E4a', 'no approach window', 'no stored segment or the standardisation failed',
         lambda d: ~_b(d['oracle_ok'])),
    Rule('E4b', f'approach shorter than {C.MIN_DURATION_MS / 1000:.2f} s',
         'too short for a reach from arm\'s length to the nose: a fragment of the movement was segmented (the '
         'lower duration bound of docs/09 gate G2)',
         lambda d: _num(d, 'duration_ms') < C.MIN_DURATION_MS),
    Rule('E4c', f'phone did not turn (peak |w| < {C.MIN_PEAK_GYRO_DPS:.0f} deg/s)',
         'every reach to the nose rotates the forearm; below the online_v1 onset floor the segment is not the reach '
         '(the peak rule of docs/09 gate G2)',
         lambda d: _num(d, 'peak_gyro_dps') < C.MIN_PEAK_GYRO_DPS),
    Rule('E4d', f'sampling gap > {C.MAX_GAP_MS:.0f} ms near the movement',
         f'a gap longer than {C.MAX_GAP_MS:.0f} ms within [onset - {C.GAP_WINDOW_PAD_S} s, end + '
         f'{C.GAP_WINDOW_PAD_S} s] (shorter gaps are interpolated)', lambda d: _num(d, 'win_n_gaps_100') > 0),
    Rule('E4e', 'recording quality (T4, T5, T7, S3)',
         f'page hidden during the movement, motion rate < {C.MIN_MOTION_HZ:.0f} Hz, no gyroscope, or a gravity '
         'sign that nothing supports', lambda d: (
             _b(d['hidden_during']) | (_num(d, 'motion_hz') < C.MIN_MOTION_HZ) | ~_b(d['rotation_available'], True)
             | (d['gravity_sign_source'].fillna('default') == 'default'))),
)


def apply_exclusions(trials: pd.DataFrame, rules: Sequence[Rule] = RULES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(trials with ``analysis_set`` / ``excl_rule`` / ``excl_detail``, flow table, E1 breakdown).

    Rules apply in order to the attempts still in; an attempt is charged to the FIRST rule it fails.
    E3 diverts the free-choice trials to the natural-use set, which then goes through the E4 rules too.
    analysis_set: 'primary', 'natural_use' or 'excluded'."""
    d = trials.copy()
    d['analysis_set'] = 'primary'
    d['excl_rule'] = None
    d['excl_detail'] = None
    main = pd.Series(True, index=d.index)
    nat = pd.Series(False, index=d.index)
    flow = []

    pk = 'person_key' if 'person_key' in d.columns else 'person_id'

    def counts(m: pd.Series) -> tuple[int, int, int]:
        return int(m.sum()), int(d.loc[m, pk].nunique()), int(d.loc[m, 'session_id'].nunique())

    n0, p0, s0 = counts(main)
    flow.append({'step': 'start', 'rule': 'all attempts in the snapshot', 'description': '',
                 'removed': 0, 'remaining': n0, 'persons': p0, 'sessions': s0,
                 'natural_use_added': 0, 'natural_use_removed': 0, 'natural_use': 0, 'natural_use_persons': 0})
    e1 = None
    for r in rules:
        fails = r.fails(d).reindex(d.index).fillna(False).astype(bool)
        hit_main = main & fails
        hit_nat = nat & fails
        if r.code == 'E1':
            e1 = e1_reason(d[hit_main])
        added = 0
        if r.divert == 'natural_use':
            added = int(hit_main.sum())
            nat |= hit_main
            d.loc[hit_main, 'analysis_set'] = 'natural_use'
            d.loc[hit_main, 'excl_rule'] = r.code
            hit_nat = pd.Series(False, index=d.index)
        else:
            d.loc[hit_main | hit_nat, 'analysis_set'] = 'excluded'
            d.loc[hit_main | hit_nat, 'excl_rule'] = r.code
            nat &= ~hit_nat
        main &= ~hit_main
        n, p, s = counts(main)
        flow.append({'step': r.code, 'rule': r.name, 'description': r.description,
                     'removed': int(hit_main.sum()) - added, 'remaining': n, 'persons': p, 'sessions': s,
                     'natural_use_added': added, 'natural_use_removed': int(hit_nat.sum()),
                     'natural_use': int(nat.sum()), 'natural_use_persons': int(d.loc[nat, 'person_id'].nunique())})
    # N1: the natural-use headline keeps ONE free-choice trial per person, the earliest that passed every rule
    # (docs/09 D4: a person's first, unpractised movement); later ones (a repeat visit) are reported apart
    order = [c for c in ('session_started_at', 'session_order', 'trial_index', 'attempt') if c in d.columns]
    natd = d[nat].sort_values(order, kind='stable') if order else d[nat]
    rep = natd.index[natd['person_id'].duplicated(keep='first')]
    d.loc[rep, 'analysis_set'] = 'natural_use_repeat'
    d.loc[rep, 'excl_rule'] = 'N1'
    nat.loc[rep] = False
    n, p, s = counts(main)
    flow.append({'step': 'N1', 'rule': 'later free-choice trial of a person',
                 'description': 'one natural-use trial per person (the earliest that passed every rule); a later one '
                                '(a repeat visit, so a practised movement) moves to the natural_use_repeat set',
                 'removed': 0, 'remaining': n, 'persons': p, 'sessions': s, 'natural_use_added': 0,
                 'natural_use_removed': int(len(rep)), 'natural_use': int(nat.sum()),
                 'natural_use_persons': int(d.loc[nat, 'person_id'].nunique())})
    if e1 is not None:
        d.loc[e1.index, 'excl_detail'] = e1
    breakdown = (e1.value_counts().rename_axis('reason').reset_index(name='attempts') if e1 is not None
                 else pd.DataFrame(columns=['reason', 'attempts']))
    return d, pd.DataFrame(flow), breakdown


# ───────────────────────────── cache ─────────────────────────────


def snapshot_fingerprint(root: str | Path = C.SNAPSHOT_DIR) -> dict[str, Any]:
    """sha256 over (relative path, sha256 of the content) of every file of the snapshot, sorted."""
    root = Path(root)
    h = hashlib.sha256()
    n = 0
    size = 0
    for f in sorted(p for p in root.rglob('*') if p.is_file()):
        C.assert_not_private(f)
        b = f.read_bytes()
        h.update(str(f.relative_to(root)).replace('\\', '/').encode())
        h.update(hashlib.sha256(b).digest())
        n += 1
        size += len(b)
    return {'sha256': h.hexdigest(), 'n_files': n, 'bytes': size}


def file_sha256(path: str | Path) -> str:
    """sha256 of one file's bytes ('' when it does not exist)."""
    p = C.assert_not_private(path)
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else ''


def cache_params(**stream_kw: Any) -> dict[str, Any]:
    """Every parameter that shapes the cache (recorded in meta.json and part of :func:`data_identity`)."""
    return {'fs_hz': C.FS_HZ, 'lowpass_hz': C.LOWPASS_HZ, 'lowpass_order': C.LOWPASS_ORDER,
            'max_gap_ms': C.MAX_GAP_MS, 'still_gyro_dps': C.STILL_GYRO_DPS, 'rest_ms': C.REST_MS,
            'n_norm': C.N_NORM, 'fixed_pre_s': C.FIXED_PRE_S, 'fixed_post_s': C.FIXED_POST_S,
            'quantize_ios': bool(stream_kw.get('quantize_ios', C.QUANTIZE_IOS)),
            'min_duration_ms': C.MIN_DURATION_MS, 'min_peak_gyro_dps': C.MIN_PEAK_GYRO_DPS,
            'gap_window_pad_s': C.GAP_WINDOW_PAD_S, 'pre_gravity_ms': C.PRE_GRAVITY_MS,
            'g1_max_gap_ms': C.G1_MAX_GAP_MS, 'g1_pad_s': C.G1_PAD_S,
            'g2_duration_ms': [C.G2_MIN_DURATION_MS, C.G2_MAX_DURATION_MS],
            'seg_disagree_ms': C.SEG_DISAGREE_MS, 'clip_acc_ms2': CLIP_ACC_MS2, 'clip_gyro_dps': CLIP_GYRO_DPS,
            'norm_version': 'norm_v2', 'oracle': 'stored online_v1 (timing.onset_ms, timing.end_ms)'}


#: the modules whose code builds the cache arrays and tables (a change to any of them makes the cache stale)
BUILDER_MODULES = ('data.py', 'standardize.py')   # config's cache constants enter through cache_params()


def builder_sha256() -> str:
    """sha256 of the code that builds the cache: the stage2 builder modules and every motionlab source."""
    h = hashlib.sha256()
    for name in BUILDER_MODULES:
        h.update(name.encode())
        h.update(hashlib.sha256((C.ANALYSIS_DIR / 'stage2' / name).read_bytes()).digest())
    h.update(C.code_fingerprint()['motionlab_sha256'].encode())
    return h.hexdigest()


def data_identity(snapshot: str | Path = C.SNAPSHOT_DIR, curation: str | Path = C.CURATION_FILE, *,
                  snapshot_fp: dict[str, Any] | None = None, **stream_kw: Any) -> dict[str, Any]:
    """What a cache built now from these inputs would be: snapshot, curation, parameters, builder code.

    ``data_id`` (a sha256 over all of them) is recorded in meta.json and in every result
    (``info.cache_code.data_id``): ``stage2.pipeline`` rebuilds a cache whose identity no longer matches
    its inputs (a participant added to the snapshot or the curation, a changed threshold or builder) and
    recomputes every result made from another cache. Deterministic: no timestamp."""
    fp = snapshot_fp or snapshot_fingerprint(snapshot)
    params = cache_params(**stream_kw)
    out = {'cache_version': C.CACHE_VERSION, 'snapshot_sha256': fp['sha256'], 'curation_sha256': file_sha256(curation),
           'params_sha256': hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest(),
           'builder_sha256': builder_sha256()}
    out['data_id'] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out


def git_commit() -> dict[str, Any]:
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=C.REPO_DIR, capture_output=True, text=True,
                             timeout=20).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=C.REPO_DIR, capture_output=True, text=True,
                               timeout=20).stdout.strip() != ''
        return {'git_commit': sha or None, 'git_dirty': dirty}
    except (OSError, subprocess.SubprocessError):
        return {'git_commit': None, 'git_dirty': None}


def build_tables(ds: Dataset, cur: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = session_table(ds, cur)
    trials = trial_table(ds, sessions)
    persons = person_table(sessions, trials)
    return sessions, trials, persons


def build_cache(snapshot: str | Path = C.SNAPSHOT_DIR, curation: str | Path = C.CURATION_FILE,
                cache_dir: str | Path = C.CACHE_DIR, *, progress: bool = True, **stream_kw: Any) -> Path:
    """Load, curate, standardise, apply the exclusion cascade and write the cache. Returns its folder."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cur = load_curation(curation)
    if progress:
        print(f'loading {snapshot} ...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds = load_dataset(snapshot, simulated='include')
    sessions, trials, persons = build_tables(ds, cur)
    if progress:
        print(f'  {len(sessions)} sessions, {len(trials)} attempts ({time.time() - t0:.0f} s); standardising ...', flush=True)
    built = standardize_all(ds, trials, progress=progress, **stream_kw)
    trials = trials.merge(built.qc, how='left', on='trial_id')
    trials['oracle_ok'] = trials['oracle_ok'].astype('boolean').fillna(False).astype(bool)
    trials = add_flags_and_gates(trials)
    trials, flow, e1 = apply_exclusions(trials)
    # set counts per person
    for name, col in (('primary', 'n_primary'), ('natural_use', 'n_natural_use'),
                      ('natural_use_repeat', 'n_natural_use_repeat')):
        cnt = trials[trials['analysis_set'] == name].groupby('person_id').size()
        persons[col] = persons['person_id'].map(cnt).fillna(0).astype(int)
    prim = trials[trials['analysis_set'] == 'primary']
    for h in C.HANDS:
        cnt = prim[prim['hand'] == h].groupby('person_id').size()
        persons[f'n_primary_{h}'] = persons['person_id'].map(cnt).fillna(0).astype(int)
    row = {tid: i for i, tid in enumerate(built.trial_ids)}
    trials['row'] = trials['trial_id'].map(row).astype('Int64')
    np.savez_compressed(cache_dir / 'arrays.npz', trial_id=np.asarray(built.trial_ids), approach=built.approach,
                        fixed=built.fixed, fixed_mask=built.fixed_mask, fixed_phase=built.fixed_phase,
                        ret=built.ret, ret_ok=built.ret_ok, channels=np.asarray(CHANNELS))
    np.savez_compressed(cache_dir / 'streams.npz', trial_id=np.asarray(built.trial_ids), **built.streams)
    for name, df in (('trials', trials), ('persons', persons), ('sessions', sessions), ('exclusion_flow', flow),
                     ('exclusion_e1', e1)):
        df.to_pickle(cache_dir / f'{name}.pkl')
        write_csv(df, cache_dir / f'{name}.csv', index=False)
    snap_fp = snapshot_fingerprint(snapshot)
    ident = data_identity(snapshot, curation, snapshot_fp=snap_fp, **stream_kw)
    meta = {
        'cache_version': C.CACHE_VERSION, 'created_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'snapshot': str(Path(snapshot).name), 'snapshot_fingerprint': snap_fp, **git_commit(),
        # the code that built the cache (stage2 and motionlab are not committed yet, so the commit alone does not
        # identify it) and the frozen environment
        'code': C.code_fingerprint(),
        'params': cache_params(**stream_kw),
        'curation_sha256': file_sha256(curation),
        'data_identity': ident,
        'channels': list(CHANNELS), 'channel_info': {k: list(v) for k, v in CHANNEL_INFO.items()},
        'flags': FLAG_COLUMNS, 'gates': GATE_COLUMNS,
        'counts': {'attempts': int(len(trials)), 'cached': int(len(built.trial_ids)),
                   'primary': int((trials['analysis_set'] == 'primary').sum()),
                   'natural_use': int((trials['analysis_set'] == 'natural_use').sum()),
                   'natural_use_repeat': int((trials['analysis_set'] == 'natural_use_repeat').sum()),
                   'persons_primary': int(prim['person_id'].nunique()),
                   'primary_flags': {c: int(prim[c].sum()) for c in FLAG_COLUMNS},
                   'primary_gates_failed': {c: int((~prim[c]).sum()) for c in (*GATE_COLUMNS, 'gate_pass')}},
        'build_seconds': round(time.time() - t0, 1),
    }
    (cache_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    if progress:
        print(f'cache written to {cache_dir} in {time.time() - t0:.0f} s: {meta["counts"]}', flush=True)
    return cache_dir


@dataclass
class Stage2Data:
    """Cached arrays aligned row by row with ``trials`` (see :func:`load_cache`).

    approach (N, C, 64), fixed (N, C, 180), fixed_mask (N, 180), fixed_phase (N, 180), ret (N, C, 64),
    ret_ok (N,): float32 / bool / int8 arrays, channels in ``channels`` order (``standardize.CHANNELS``).
    """

    trials: pd.DataFrame
    persons: pd.DataFrame
    approach: np.ndarray
    fixed: np.ndarray
    fixed_mask: np.ndarray
    fixed_phase: np.ndarray
    ret: np.ndarray
    ret_ok: np.ndarray
    channels: tuple[str, ...]
    meta: dict[str, Any]
    cache_dir: Path
    _streams: dict[str, np.ndarray] | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def y(self) -> np.ndarray:
        """1 = right hand, 0 = left (instructed; reported for the natural-use set)."""
        return self.trials['y_right'].astype(int).to_numpy()

    @property
    def groups(self) -> np.ndarray:
        """person_id per row: the only grouping key for splits and bootstraps."""
        return self.trials['person_id'].astype(str).to_numpy()

    @property
    def trial_ids(self) -> np.ndarray:
        return self.trials['trial_id'].to_numpy()

    def channel_rows(self, names: str | Sequence[str]) -> list[int]:
        names = [names] if isinstance(names, str) else list(names)
        return [self.channels.index(c) for c in names]

    def subset(self, mask: Any) -> 'Stage2Data':
        m = np.asarray(mask)
        idx = np.flatnonzero(m) if m.dtype == bool else m.astype(int)
        return Stage2Data(self.trials.iloc[idx].reset_index(drop=True), self.persons, self.approach[idx],
                          self.fixed[idx], self.fixed_mask[idx], self.fixed_phase[idx], self.ret[idx],
                          self.ret_ok[idx], self.channels, self.meta, self.cache_dir, self._streams)

    def stream(self, i: int) -> dict[str, np.ndarray]:
        """The full standardised recording of row i (60 Hz, low-passed, W3C device frame, NOT windowed):
        t (ms), f (accg), acc (browser linear acc), gyro (deg/s), valid, phase (``standardize.PHASES``)."""
        if self._streams is None:
            with np.load(self.cache_dir / 'streams.npz') as z:
                self._streams = {k: z[k] for k in z.files}
        s = self._streams
        k = int(self.trials['row'].iloc[i])
        a, b = int(s['offsets'][k]), int(s['offsets'][k + 1])
        return {c: s[c][a:b] for c in ('t', 'f', 'acc', 'gyro', 'valid', 'phase')}


def load_cache(sets: Iterable[str] | None = ('primary',), cache_dir: str | Path = C.CACHE_DIR, *,
               build_if_missing: bool = False) -> Stage2Data:
    """Arrays and tables of the requested analysis sets ('primary', 'natural_use' (one free-choice trial per
    person), 'natural_use_repeat', 'excluded'; None = every attempt that has arrays). Rows are sorted by
    person, the person's sessions in time, trial order. Refused inside a fit_predict (``guard``)."""
    assert_outside_fit_predict('load_cache')
    cache_dir = Path(cache_dir)
    if not (cache_dir / 'arrays.npz').exists():
        if not build_if_missing:
            raise FileNotFoundError(f'no Stage 2 cache in {cache_dir}: run python -m stage2.scripts.build_cache')
        build_cache(cache_dir=cache_dir)
    trials = pd.read_pickle(cache_dir / 'trials.pkl')
    persons = pd.read_pickle(cache_dir / 'persons.pkl')
    meta = json.loads((cache_dir / 'meta.json').read_text(encoding='utf-8'))
    sel = trials['row'].notna()
    if sets is not None:
        sets = [sets] if isinstance(sets, str) else list(sets)
        sel &= trials['analysis_set'].isin(sets)
    t = trials[sel].reset_index(drop=True)
    rows = t['row'].astype(int).to_numpy()
    with np.load(cache_dir / 'arrays.npz') as z:
        ids = z['trial_id']
        if not np.array_equal(ids[rows], t['trial_id'].to_numpy().astype(ids.dtype)):
            raise RuntimeError('cache corrupt: trial table and arrays are misaligned')
        out = Stage2Data(t, persons, z['approach'][rows], z['fixed'][rows], z['fixed_mask'][rows],
                         z['fixed_phase'][rows], z['ret'][rows], z['ret_ok'][rows],
                         tuple(str(c) for c in z['channels']), meta, cache_dir)
    return out


def load_tables(cache_dir: str | Path = C.CACHE_DIR) -> dict[str, pd.DataFrame]:
    """Every cached table: trials (all attempts), persons, sessions, exclusion_flow, exclusion_e1."""
    assert_outside_fit_predict('load_tables')
    cache_dir = Path(cache_dir)
    return {n: pd.read_pickle(cache_dir / f'{n}.pkl')
            for n in ('trials', 'persons', 'sessions', 'exclusion_flow', 'exclusion_e1')}


_TRIAL_MAP: dict[tuple[str, float], pd.DataFrame] = {}


def canonical_trial_map(cache_dir: str | Path | None = None) -> pd.DataFrame | None:
    """trial_id -> (person_id, analysis_set, trial_key) of every cached attempt: the curated truth that
    ``cv.run_experiment`` checks a lane's rows against. None when there is no cache."""
    assert_outside_fit_predict('canonical_trial_map')
    p = Path(cache_dir if cache_dir is not None else C.CACHE_DIR) / 'trials.pkl'
    if not p.exists():
        return None
    key = (str(p.resolve()), p.stat().st_mtime)
    if key not in _TRIAL_MAP:
        t = pd.read_pickle(p)
        _TRIAL_MAP.clear()
        _TRIAL_MAP[key] = t[['trial_id', 'person_id', 'analysis_set', 'trial_key']].set_index('trial_id')
    return _TRIAL_MAP[key]
