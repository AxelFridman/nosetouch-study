"""Load NoseTouch records into tidy tables.

Accepted inputs (any mix, any number of paths):
  (a) the backend / receiver / synthesizer folder tree ``<root>/<study>/<participant>/<session>/session.json``
      + ``trial_*.json`` - in fact ANY directory: it is walked recursively and every ``*.json`` (also
      ``*.json.txt`` exports re-shared as text, ``*.json.gz``, ``*.jsonl`` / ``*.ndjson``, ``*.zip`` archives
      such as a Google Drive folder download) is classified by its content, not by its name;
  (b) export bundles (single JSON with ``format: 'nose-touch-export'`` and arrays participants / sessions /
      trials) shared by participants from the app;
  (c) upload envelopes (``{kind, study_key, sent_at, record}``) and JSON arrays of records.
Unrecognised JSON files (configs, manifests) are skipped and listed in ``Dataset.issues``.

Deduplication (docs/02 section 10): records with the same id are copies of one record; the newest
``updated_at`` (sessions, participant profiles) / ``created_at`` (trials) is kept, ties keep the first one
loaded (files are read in sorted order); every dropped copy is listed in ``Dataset.duplicates``. Two
DIFFERENT trial ids with the same (session_id, trial_index, attempt), and several accepted attempts of
one planned trial, are kept and flagged (columns ``dup_attempt_key`` / ``multi_accepted``).
Records failing schema validation are skipped (``on_invalid='skip'``, default), kept (``'keep'``) or raise
(``'raise'``); every problem is listed in ``Dataset.issues``.

Each session gets its norm_v1 normalisation (``conventions.session_normalization``: gravity sign from
calibration -> trial vote -> engine expectation -> +1, where trial votes that clearly contradict a
calibrated sign override it and ``gravity_sign_order=('engine', 'vote')`` puts the engine first;
rotationRate mapping from ``calibration.axis_check``); ``Dataset.signals`` applies it.

Stream columns are held as float64 numpy arrays (null -> NaN) to keep memory ~4x below Python lists;
``Dataset.record()`` rebuilds the exact JSON-compatible record (NaN -> null).
"""
from __future__ import annotations

import datetime as _dt
import gzip
import json
import math
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema as _schema
from .conventions import (
    ORIENTATIONS, DeviceSignals, GravitySign, Normalization, normalize_motion, rotation_rate_to_device,
    session_normalization, trial_gravity_vote,
)

PathLike = str | Path
_JSON_SUFFIXES = ('.json', '.json.txt', '.json.gz', '.jsonl', '.ndjson', '.zip')

MOTION_NUM_COLUMNS = ('t', 'ts', 'acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma')
ORIENTATION_NUM_COLUMNS = ('t', 'ts', 'alpha', 'beta', 'gamma', 'compass')
QUATERNION_NUM_COLUMNS = ('t', 'x', 'y', 'z', 'w')
_STREAM_COLUMNS = {'motion': MOTION_NUM_COLUMNS, 'orientation': ORIENTATION_NUM_COLUMNS, 'quaternion': QUATERNION_NUM_COLUMNS}
_SKIP_DIRS = {'.venv', 'venv', 'node_modules', '.git', '__pycache__', '.pytest_cache'}

# ───────────────────────────── streams as arrays ─────────────────────────────


def _arr(values: Any) -> np.ndarray:
    """List of numbers / None (or an array) -> float64 array with NaN for null."""
    if values is None:
        return np.zeros(0)
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    try:
        return np.asarray(values, dtype=float)  # fast path: no None inside
    except (TypeError, ValueError):
        return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def _json_list(a: Any) -> list[float | None]:
    """float array -> JSON list (NaN -> None). Lists pass through unchanged."""
    if isinstance(a, list):
        return list(a)
    arr = np.asarray(a, dtype=float)
    out = arr.tolist()
    if np.isnan(arr).any():
        out = [None if (v != v) else v for v in out]
    return out


@dataclass
class OrientationArrays:
    t: np.ndarray
    ts: np.ndarray
    angles: np.ndarray  # (m,3) W3C [alpha, beta, gamma] deg
    compass: np.ndarray
    absolute: bool | None
    ts_first_raw: float


@dataclass
class QuaternionArrays:
    t: np.ndarray
    q: np.ndarray  # (q,4) [x, y, z, w]
    sensor: str
    frequency_hz: float


@dataclass
class TrialStreams:
    """All streams of one trial as numpy arrays (null -> NaN). Raw device frame, RAW gravity sign.

    t: (n,) handler time, ms since t0. ts: event.timeStamp - first raw timeStamp (ms, NaN if missing).
    acc / accg: (n,3) m/s^2. rr: (n,3) rotationRate in schema order [alpha, beta, gamma] (deg/s).
    Use :meth:`gyro` for the device-frame vector [x, y, z] and ``Dataset.signals`` (or
    :func:`conventions.normalize_motion`) for the fully normalised physical signals.
    """

    t: np.ndarray
    ts: np.ndarray
    acc: np.ndarray
    accg: np.ndarray
    rr: np.ndarray
    interval_raw: float
    ts_first_raw: float
    orientation: OrientationArrays | None
    quaternion: QuaternionArrays | None
    touches: pd.DataFrame
    cues: pd.DataFrame

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    def gyro(self, verdict: str | None = None, units: str | None = None) -> np.ndarray:
        """Device-frame angular velocity [x, y, z] in deg/s: [alpha, beta, gamma] under the W3C mapping,
        or remapped per a calibration ``axis_check`` verdict / units."""
        return rotation_rate_to_device(self.rr, verdict, units)

    def motion_frame(self) -> pd.DataFrame:
        """The motion stream with its schema column names."""
        cols = {'t': self.t, 'ts': self.ts}
        for j, ax in enumerate('xyz'):
            cols[f'acc_{ax}'] = self.acc[:, j]
        for j, ax in enumerate('xyz'):
            cols[f'accg_{ax}'] = self.accg[:, j]
        for j, ax in enumerate(('alpha', 'beta', 'gamma')):
            cols[f'rr_{ax}'] = self.rr[:, j]
        return pd.DataFrame(cols)


def _num(x: Any) -> float:
    return float('nan') if x is None else float(x)


def streams_from_record(record: Mapping[str, Any]) -> TrialStreams:
    """Convert a TrialRecord's ``streams`` (lists or arrays) into :class:`TrialStreams`."""
    s = record['streams']
    m = s['motion']
    t = _arr(m['t'])
    n = t.shape[0]

    def col(st: Mapping[str, Any], k: str, length: int) -> np.ndarray:
        a = _arr(st.get(k))
        if a.shape[0] != length:
            raise ValueError(f"trial {record.get('trial_id')}: column {k} has {a.shape[0]} values, t has {length}")
        return a

    acc = np.column_stack([col(m, f'acc_{a}', n) for a in 'xyz']) if n else np.zeros((0, 3))
    accg = np.column_stack([col(m, f'accg_{a}', n) for a in 'xyz']) if n else np.zeros((0, 3))
    rr = np.column_stack([col(m, f'rr_{a}', n) for a in ('alpha', 'beta', 'gamma')]) if n else np.zeros((0, 3))
    o = s.get('orientation')
    ori = None
    if o:
        to = _arr(o['t'])
        k = to.shape[0]
        ang = np.column_stack([col(o, a, k) for a in ('alpha', 'beta', 'gamma')]) if k else np.zeros((0, 3))
        ori = OrientationArrays(to, col(o, 'ts', k), ang, col(o, 'compass', k), o.get('absolute'), _num(o.get('ts_first_raw')))
    q = s.get('quaternion')
    quat = None
    if q:
        tq = _arr(q['t'])
        k = tq.shape[0]
        Q = np.column_stack([col(q, a, k) for a in 'xyzw']) if k else np.zeros((0, 4))
        quat = QuaternionArrays(tq, Q, q.get('sensor', ''), float(q.get('frequency_hz') or float('nan')))
    touches = pd.DataFrame(list(s.get('touches') or []), columns=['t', 'type', 'n', 'x', 'y'])
    cues = pd.DataFrame(list(s.get('cues') or []), columns=['t', 'kind'])
    return TrialStreams(
        t=t, ts=col(m, 'ts', n), acc=acc, accg=accg, rr=rr, interval_raw=_num(m.get('interval_raw')),
        ts_first_raw=_num(m.get('ts_first_raw')), orientation=ori, quaternion=quat, touches=touches, cues=cues,
    )


def compact_streams(record: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy of a trial whose numeric stream columns are float arrays (NaN for null)."""
    out = dict(record)
    st = dict(record['streams'])
    for name, cols in _STREAM_COLUMNS.items():
        s = st.get(name)
        if not s:
            continue
        s2 = dict(s)
        for c in cols:
            if c in s2 and s2[c] is not None:
                s2[c] = _arr(s2[c])
        st[name] = s2
    out['streams'] = st
    return out


def jsonify_streams(record: Mapping[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`compact_streams`: a deep, JSON-compatible copy (NaN -> None)."""
    out = _schema.deep_copy({k: v for k, v in record.items() if k != 'streams'})
    st: dict[str, Any] = {}
    for name, s in record['streams'].items():
        if isinstance(s, Mapping):
            s2 = {}
            for k, v in s.items():
                s2[k] = _json_list(v) if isinstance(v, (np.ndarray, list)) else v
            st[name] = s2
        elif isinstance(s, list):
            st[name] = _schema.deep_copy(s)
        else:
            st[name] = s
    out['streams'] = st
    return out


# ───────────────────────────── reading files ─────────────────────────────


@dataclass
class LoadIssue:
    level: str  # error | warning | info
    source: str
    kind: str
    record_id: str | None
    message: str


def _parse_bytes(data: bytes, name: str) -> list[Any]:
    text = data.decode('utf-8-sig')
    if name.endswith(('.jsonl', '.ndjson')):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def _iter_file(path: Path) -> Iterator[tuple[str, Any]]:
    name = path.name.lower()
    if name.endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename):
                n = info.filename.lower()
                if info.is_dir() or not n.endswith(_JSON_SUFFIXES[:-1]):
                    continue
                data = zf.read(info)
                if n.endswith('.gz'):
                    data = gzip.decompress(data)
                    n = n[:-3]
                for doc in _parse_bytes(data, n):
                    yield f'{path}!{info.filename}', doc
        return
    if name.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            data = f.read()
        for doc in _parse_bytes(data, name[:-3]):
            yield str(path), doc
        return
    for doc in _parse_bytes(path.read_bytes(), name):
        yield str(path), doc


def _candidate_files(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    out = []
    for f in sorted(p.rglob('*')):
        if any(part in _SKIP_DIRS or part.startswith('.') for part in f.relative_to(p).parts[:-1]):
            continue
        n = f.name.lower()
        if f.is_file() and n.endswith(_JSON_SUFFIXES):
            out.append(f)
    return out


def iter_documents(paths: PathLike | Iterable[PathLike], issues: list[LoadIssue] | None = None) -> Iterator[tuple[str, Any]]:
    """(source, parsed JSON) for every JSON document under ``paths`` (sorted, recursive)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f'data path does not exist: {p}')
        for f in _candidate_files(p):
            try:
                yield from _iter_file(f)
            except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile, OSError, EOFError) as e:
                if issues is None:
                    raise
                issues.append(LoadIssue('error', str(f), 'file', None, f'unreadable: {type(e).__name__}: {e}'))


def extract_records(doc: Any, source: str, issues: list[LoadIssue] | None = None) -> Iterator[tuple[str, dict[str, Any], str]]:
    """(kind, record, source) for every participant / session / trial contained in one JSON document."""
    if isinstance(doc, list):
        for i, d in enumerate(doc):
            yield from extract_records(d, f'{source}[{i}]', issues)
        return
    what = _schema.detect_def(doc)
    if what == 'ExportBundle':
        for p in doc.get('participants') or []:
            yield 'participant', p, source
        for s in doc.get('sessions') or []:
            yield 'session', s, source
        for t in doc.get('trials') or []:
            yield 'trial', t, source
    elif what == 'UploadEnvelope':
        yield from extract_records(doc['record'], source, issues)
    elif what == 'SessionRecord':
        yield 'session', doc, source
    elif what == 'TrialRecord':
        yield 'trial', doc, source
    elif what == 'ParticipantProfile':
        yield 'participant', doc, source
    elif issues is not None:
        issues.append(LoadIssue('info', source, 'file', None, 'not a nose-touch record; skipped'))


def _time_key(s: Any) -> float:
    if not isinstance(s, str):
        return -math.inf
    try:
        d = _dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.timestamp()
    except ValueError:
        return -math.inf


_KIND_DEF = {'participant': 'ParticipantProfile', 'session': 'SessionRecord', 'trial': 'TrialRecord'}
_ID_FIELD = {'participant': 'participant_id', 'session': 'session_id', 'trial': 'trial_id'}
_TIME_FIELD = {'participant': ('updated_at', 'created_at'), 'session': ('updated_at', 'started_at'), 'trial': ('created_at',)}


def _record_time(kind: str, rec: Mapping[str, Any]) -> float:
    for f in _TIME_FIELD[kind]:
        k = _time_key(rec.get(f))
        if k > -math.inf:
            return k
    return -math.inf


#: outcome.note of the copy the app saves BEFORE the participant decides (app/src/app/sessionFlow.ts).
REVIEW_INTERRUPTED_NOTE = 'review_interrupted'


def _wins_tie(kind: str, new: Mapping[str, Any], old: Mapping[str, Any]) -> bool:
    """Same id and same time: does ``new`` replace ``old``? Only for trials: the app saves a complete
    attempt before its review (note 'review_interrupted') and again after the decision, with the same
    created_at, so the decided copy wins whatever order the files were loaded in."""
    if kind != 'trial':
        return False
    def pending(r: Mapping[str, Any]) -> bool:
        return (r.get('outcome') or {}).get('note') == REVIEW_INTERRUPTED_NOTE
    return pending(old) and not pending(new)


# ───────────────────────────── flattening ─────────────────────────────


def age_band(age: Any, edges: tuple[int, ...] = (30, 50)) -> str:
    """'<30', '30-49', '50+' (default edges) or 'unknown'."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return 'unknown'
    if not math.isfinite(a):
        return 'unknown'
    lo = None
    for e in edges:
        if a < e:
            return f'<{e}' if lo is None else f'{lo}-{e - 1}'
        lo = e
    return f'{lo}+'


def dominant_hand(handedness_self: Any, ehi_class: Any) -> str | None:
    if handedness_self in ('right', 'left'):
        return str(handedness_self)
    if ehi_class in ('right', 'left'):
        return str(ehi_class)
    return None


def participant_row(p: Mapping[str, Any]) -> dict[str, Any]:
    sf = p.get('ehi_sf') or {}
    return {
        'participant_id': p.get('participant_id'), 'age_years': p.get('age_years'), 'sex': p.get('sex'),
        'handedness_self': p.get('handedness_self'), 'ehi_lq': p.get('ehi_lq'), 'ehi_class': p.get('ehi_class'),
        'phone_hand': p.get('phone_hand'), 'height_cm': p.get('height_cm'), 'arm_issue': p.get('arm_issue'),
        'phone_model': p.get('phone_model'), 'phone_case': p.get('phone_case'),
        'ehi_writing': sf.get('writing'), 'ehi_throwing': sf.get('throwing'), 'ehi_toothbrush': sf.get('toothbrush'),
        'ehi_spoon': sf.get('spoon'), 'created_at': p.get('created_at'), 'updated_at': p.get('updated_at'),
        'dominant_hand': dominant_hand(p.get('handedness_self'), p.get('ehi_class')),
        'age_band': age_band(p.get('age_years')),
    }


def session_row(s: Mapping[str, Any], gs: GravitySign | Normalization | None = None) -> dict[str, Any]:
    """One row of ``Dataset.sessions``; ``gs`` = the session's Normalization (or just its GravitySign)."""
    p = s.get('participant') or {}
    d = s.get('device') or {}
    cap = s.get('capabilities') or {}
    meas = cap.get('measured') or {}
    cal = s.get('calibration') or {}
    bias = cal.get('gyro_bias_dps') or {}
    ac = cal.get('axis_check') or {}
    norm = gs if isinstance(gs, Normalization) else None
    if isinstance(gs, Normalization):
        gs = gs.gravity
    prot = s.get('protocol') or {}
    st = s.get('settings') or {}
    prog = s.get('progress') or {}
    row = {
        'session_id': s.get('session_id'), 'participant_id': s.get('participant_id'), 'study_code': s.get('study_code'),
        'started_at': s.get('started_at'), 'updated_at': s.get('updated_at'), 'ended_at': s.get('ended_at'),
        'status': s.get('status'), 'simulated': s.get('simulated'), 'app_version': s.get('app_version'),
        'schema_version': s.get('schema_version'), 'timezone_offset_min': s.get('timezone_offset_min'),
        'preset': prot.get('preset'), 'protocol_version': prot.get('protocol_version'), 'protocol_seed': prot.get('seed'),
        'posture': prot.get('posture'), 'first_hand': prot.get('first_hand'),
        'n_planned_trials': prot.get('n_planned_trials'), 'estimated_minutes': prot.get('estimated_minutes'),
        'language': st.get('language'), 'audio': st.get('audio'), 'speech': st.get('speech'),
        'next_trial_index': prog.get('next_trial_index'), 'n_attempts': prog.get('n_attempts'),
        'n_accepted': prog.get('n_accepted'), 'consent_version': (s.get('consent') or {}).get('version'),
    }
    pr = participant_row(p)
    for k in ('sex', 'age_years', 'age_band', 'handedness_self', 'ehi_lq', 'ehi_class', 'phone_hand', 'height_cm',
              'arm_issue', 'phone_model', 'phone_case', 'dominant_hand'):
        row[k] = pr[k]
    row.update({
        'platform': d.get('platform'), 'browser': d.get('browser'), 'os_version': d.get('os_version'),
        'model_hint': d.get('model_hint'), 'standalone': d.get('standalone'), 'screen_w': d.get('screen_w'),
        'screen_h': d.get('screen_h'), 'dpr': d.get('dpr'), 'device_language': d.get('language'),
        'hardware_concurrency': d.get('hardware_concurrency'), 'device_memory_gb': d.get('device_memory_gb'),
        'user_agent': d.get('user_agent'),
        'permission_state': cap.get('permission_state'),
        'generic_relative_orientation': cap.get('generic_relative_orientation'),
        'measured_motion_hz': meas.get('motion_hz'), 'measured_dt_ms_median': meas.get('motion_dt_ms_median'),
        'measured_acc_available': meas.get('acc_available'), 'measured_rotation_available': meas.get('rotation_available'),
        'engine': d.get('engine'), 'in_app_browser': d.get('in_app_browser'),
        'calibration_present': bool(s.get('calibration')),
        'calibration_gravity_sign': cal.get('gravity_sign'), 'gravity_magnitude': cal.get('gravity_magnitude'),
        'upright_check': cal.get('upright_check'),
        'axis_check_verdict': ac.get('verdict'), 'axis_check_units': ac.get('units'),
        'axis_check_similarity': ac.get('similarity'), 'axis_check_tilt_deg': ac.get('tilt_deg'),
        'gyro_bias_alpha': bias.get('alpha'), 'gyro_bias_beta': bias.get('beta'), 'gyro_bias_gamma': bias.get('gamma'),
    })
    if gs is not None:
        row.update({'gravity_sign': gs.sign, 'gravity_sign_source': gs.source, 'gravity_sign_warning': gs.warning,
                    'gravity_votes_for': gs.votes_for, 'gravity_votes_against': gs.votes_against,
                    'gravity_sign_detail': gs.detail, 'engine_effective': gs.engine})
    if norm is not None:
        row.update({'rr_mapping': norm.rates.verdict, 'rr_units': norm.rates.units,
                    'rr_verified': norm.rates.verified, 'norm_version': norm.version,
                    'norm_flags': ';'.join(norm.flags)})
    return row


def trial_row(t: Mapping[str, Any]) -> dict[str, Any]:
    c = t.get('condition') or {}
    o = t.get('outcome') or {}
    tm = t.get('timing') or {}
    sg = t.get('segmentation') or {}
    bl = sg.get('baseline') or {}
    grav = bl.get('gravity') or [None, None, None]
    det = t.get('detected') or {}
    q = t.get('quality') or {}
    scr = t.get('screen') or {}
    streams = t.get('streams') or {}
    orient = c.get('orientation')
    info = ORIENTATIONS.get(orient, {})
    warn = list(q.get('warnings') or [])

    def diff(a: Any, b: Any) -> float | None:
        return (float(a) - float(b)) if (a is not None and b is not None) else None

    return {
        'trial_id': t.get('trial_id'), 'session_id': t.get('session_id'), 'participant_id': t.get('participant_id'),
        'study_code': t.get('study_code'), 'block_index': t.get('block_index'), 'trial_index': t.get('trial_index'),
        'trial_in_block': t.get('trial_in_block'), 'attempt': t.get('attempt'), 'created_at': t.get('created_at'),
        'simulated': t.get('simulated'), 'schema_version': t.get('schema_version'), 'app_version': t.get('app_version'),
        'hand': c.get('hand'), 'orientation': orient, 'speed': c.get('speed'), 'start_position': c.get('start_position'),
        'variant': c.get('variant'), 'posture': c.get('posture'), 'rep': c.get('rep'),
        'instructed_rotation': info.get('rotation'), 'instructed_screen': info.get('screen'),
        'status': o.get('status'), 'accepted': o.get('accepted'), 'participant_flag': o.get('participant_flag'),
        'note': o.get('note'),
        't0_perf_ms': tm.get('t0_perf_ms'), 't0_epoch_ms': tm.get('t0_epoch_ms'), 'still_ms': tm.get('still_ms'),
        'go_ms': tm.get('go_ms'), 'onset_ms': tm.get('onset_ms'), 'end_ms': tm.get('end_ms'),
        'end_detected_ms': tm.get('end_detected_ms'), 'stop_ms': tm.get('stop_ms'),
        'first_touch_ms': tm.get('first_touch_ms'), 'foreperiod_ms': tm.get('foreperiod_ms'),
        'false_starts': tm.get('false_starts'),
        'reaction_ms': diff(tm.get('onset_ms'), tm.get('go_ms')),
        'movement_ms': diff(tm.get('end_ms'), tm.get('onset_ms')),
        'seg_method': sg.get('method'), 'acc_source': sg.get('acc_source'), 'onset_index': sg.get('onset_index'),
        'end_index': sg.get('end_index'), 'seg_duration_ms': sg.get('duration_ms'),
        'peak_gyro_dps': sg.get('peak_gyro_dps'), 'peak_acc_ms2': sg.get('peak_acc_ms2'),
        'baseline_n': bl.get('n'), 'baseline_gyro_mean': bl.get('gyro_mean'), 'baseline_gyro_sd': bl.get('gyro_sd'),
        'baseline_acc_mean': bl.get('acc_mean'), 'baseline_acc_sd': bl.get('acc_sd'),
        'baseline_gravity_x': grav[0], 'baseline_gravity_y': grav[1], 'baseline_gravity_z': grav[2],
        'rotation_start': det.get('rotation_start'), 'rotation_end': det.get('rotation_end'),
        'tilt_change_deg': det.get('tilt_change_deg'), 'orientation_match': det.get('orientation_match'),
        'n_motion': q.get('n_motion'), 'n_orientation': q.get('n_orientation'), 'n_quaternion': q.get('n_quaternion'),
        'motion_hz': q.get('motion_hz'), 'motion_dt_ms_median': q.get('motion_dt_ms_median'),
        'motion_dt_ms_max': q.get('motion_dt_ms_max'), 'gaps_over_50ms': q.get('gaps_over_50ms'),
        'acc_available': q.get('acc_available'), 'rotation_available': q.get('rotation_available'),
        'hidden_during': q.get('hidden_during'), 'warnings': ';'.join(warn), 'n_warnings': len(warn),
        'screen_type_start': scr.get('type_start'), 'screen_angle_start': scr.get('angle_start'),
        'screen_type_end': scr.get('type_end'), 'screen_angle_end': scr.get('angle_end'),
        'n_touches': len(streams.get('touches') or []),
        'has_orientation_stream': streams.get('orientation') is not None,
        'has_quaternion_stream': streams.get('quaternion') is not None,
        'y_right': 1 if c.get('hand') == 'right' else (0 if c.get('hand') == 'left' else None),
    }


_INT_COLS = ('block_index', 'trial_index', 'trial_in_block', 'attempt', 'rep', 'onset_index', 'end_index',
             'baseline_n', 'n_motion', 'n_orientation', 'n_quaternion', 'gaps_over_50ms', 'false_starts', 'y_right',
             'n_touches', 'n_warnings', 'age_years')
_BOOL_COLS = ('accepted', 'simulated', 'orientation_match', 'acc_available', 'rotation_available', 'hidden_during',
              'hand_is_dominant', 'is_last_attempt', 'gravity_sign_warning', 'calibration_present', 'rr_verified',
              'dup_attempt_key', 'multi_accepted')


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    for c in _INT_COLS:
        if c in df.columns:
            try:
                df[c] = pd.array(pd.to_numeric(df[c], errors='coerce'), dtype='Int64')
            except (TypeError, ValueError):
                pass  # non-integral values: keep as float
    for c in _BOOL_COLS:
        if c in df.columns:
            try:
                df[c] = df[c].astype('boolean')
            except (TypeError, ValueError):
                pass
    return df


# ───────────────────────────── the dataset ─────────────────────────────


@dataclass
class _Source:
    path: str
    kind: str


class Dataset:
    """Loaded records + tidy tables.

    Attributes
      participants  one row per participant (newest profile).
      sessions      one row per session: protocol, participant snapshot, device, calibration and the
                    resolved gravity sign (``gravity_sign``, ``gravity_sign_source``, ``gravity_sign_warning``).
      trials        one row per ATTEMPT (redo attempts included; filter ``accepted``): condition, outcome,
                    timing, segmentation, detected, quality, plus participant columns (sex, age_years,
                    age_band, handedness_self, ehi_lq, ehi_class, dominant_hand, hand_is_dominant), session
                    columns (platform, engine, gravity_sign, rr_mapping, ...), ``block_kind``,
                    ``is_last_attempt``, the duplicate flags and the label ``y_right`` (1 = right hand).
      issues        problems found while loading (invalid records, duplicates, orphans, skipped files).
      duplicates    one row per dropped copy of a record (kind, record_id, kept / dropped source).
    """

    def __init__(self) -> None:
        self.participants = pd.DataFrame()
        self.sessions = pd.DataFrame()
        self.trials = pd.DataFrame()
        self.issues = pd.DataFrame(columns=['level', 'source', 'kind', 'record_id', 'message'])
        self.duplicates = pd.DataFrame(columns=DUPLICATE_COLUMNS)
        self._trials: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._participants: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, _Source] = {}
        self._norm: dict[str, Normalization] = {}
        self._keep_streams = True
        self._cache: OrderedDict[str, list[tuple[str, dict[str, Any], str]]] = OrderedDict()

    # ── sizes ──
    def __len__(self) -> int:
        return len(self._trials)

    def __repr__(self) -> str:
        n_err = int((self.issues['level'] == 'error').sum()) if len(self.issues) else 0
        return (f'<Dataset participants={len(self.participants)} sessions={len(self.sessions)} '
                f'trials={len(self.trials)} issues={len(self.issues)} (errors={n_err})>')

    @property
    def trial_ids(self) -> list[str]:
        return list(self.trials['trial_id']) if len(self.trials) else []

    # ── records ──
    def _compact(self, trial_id: str) -> dict[str, Any]:
        rec = self._trials[trial_id]
        if 'streams' in rec:
            return rec
        return compact_streams(self._reload(trial_id))

    def record(self, trial_id: str, *, compact: bool = False) -> dict[str, Any]:
        """The trial record. Default: a deep, JSON-compatible copy (safe to modify / validate / write).
        ``compact=True``: the internal record with float-array stream columns (read-only, fast)."""
        if trial_id not in self._trials:
            raise KeyError(f'unknown trial_id {trial_id!r}')
        rec = self._compact(trial_id)
        return rec if compact else jsonify_streams(rec)

    def session_record(self, session_id: str) -> dict[str, Any]:
        return _schema.deep_copy(self._sessions[session_id])

    def participant_record(self, participant_id: str) -> dict[str, Any]:
        return _schema.deep_copy(self._participants[participant_id])

    def iter_records(self, trial_ids: Iterable[str] | None = None, *, compact: bool = True) -> Iterator[dict[str, Any]]:
        for tid in (self.trial_ids if trial_ids is None else trial_ids):
            yield self.record(tid, compact=compact)

    def streams(self, trial_id: str) -> TrialStreams:
        """All streams of a trial as numpy arrays (null -> NaN), raw device frame and raw gravity sign."""
        return streams_from_record(self._compact(trial_id))

    def normalization(self, session_id: str) -> Normalization:
        """norm_v1 of a session (unknown sessions: default sign +1 with a warning, W3C assumed)."""
        nm = self._norm.get(session_id)
        if nm is not None:
            return nm
        return Normalization(GravitySign(1, 'default', True, detail='unknown session'))

    def trial_normalization(self, trial_id: str) -> Normalization:
        return self.normalization(self._trials[trial_id]['session_id'])

    def gravity_sign(self, session_id: str) -> GravitySign:
        return self.normalization(session_id).gravity

    def trial_gravity_sign(self, trial_id: str) -> GravitySign:
        return self.gravity_sign(self._trials[trial_id]['session_id'])

    def signals(self, trial_id: str) -> DeviceSignals:
        """Motion stream in physical units under the session's norm_v1 (W3C sign, rates about x, y, z)."""
        s = self.streams(trial_id)
        nm = self.trial_normalization(trial_id)
        sig = normalize_motion(s.t, s.acc, s.accg, s.rr, nm)
        gs = nm.gravity
        if gs.warning:
            sig.notes.append(f'gravity sign {gs.sign:+d} from {gs.source}: {gs.detail}')
        if nm.rates.detail:
            sig.notes.append(nm.rates.detail)
        return sig

    def select(self, *, accepted: bool | None = True, final_attempt: bool | None = None, **equals: Any) -> pd.DataFrame:
        """Rows of ``trials`` filtered by acceptance, last attempt and column equality (scalar or list)."""
        df = self.trials
        if not len(df):
            return df
        m = pd.Series(True, index=df.index)
        if accepted is not None:
            m &= df['accepted'].fillna(False) == accepted
        if final_attempt is not None:
            m &= df['is_last_attempt'].fillna(False) == final_attempt
        for k, v in equals.items():
            m &= df[k].isin(v) if isinstance(v, (list, tuple, set)) else (df[k] == v)
        return df[m.fillna(False).astype(bool)]

    def subset(self, trial_ids: Iterable[str]) -> 'Dataset':
        """A new Dataset restricted to these trials (and their sessions / participants)."""
        keep = set(trial_ids)
        ds = Dataset()
        ds._keep_streams = self._keep_streams
        ds._trials = {k: v for k, v in self._trials.items() if k in keep}
        sess = {v['session_id'] for v in ds._trials.values()}
        pids = {v['participant_id'] for v in ds._trials.values()}
        ds._sessions = {k: v for k, v in self._sessions.items() if k in sess}
        ds._participants = {k: v for k, v in self._participants.items() if k in pids}
        ds._sources = {k: v for k, v in self._sources.items() if k in keep}
        ds._norm = {k: v for k, v in self._norm.items() if k in sess}
        ds.duplicates = self.duplicates.copy()
        ds.trials = self.trials[self.trials['trial_id'].isin(keep)].reset_index(drop=True)
        ds.sessions = self.sessions[self.sessions['session_id'].isin(sess)].reset_index(drop=True) if len(self.sessions) else self.sessions
        ds.participants = self.participants[self.participants['participant_id'].isin(pids)].reset_index(drop=True) if len(self.participants) else self.participants
        ds.issues = self.issues.copy()
        return ds

    def to_bundle(self, trial_ids: Iterable[str] | None = None) -> dict[str, Any]:
        """An ExportBundle (JSON-compatible dict) of these trials and their sessions / participants."""
        ids = self.trial_ids if trial_ids is None else list(trial_ids)
        trials = [self.record(t) for t in ids]
        sess = {t['session_id'] for t in trials}
        pids = {t['participant_id'] for t in trials}
        return {
            'format': 'nose-touch-export', 'schema_version': '1.1.0', 'app_version': 'motionlab',
            'exported_at': _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'participants': [self.participant_record(p) for p in sorted(pids) if p in self._participants],
            'sessions': [self.session_record(s) for s in sorted(sess) if s in self._sessions],
            'trials': trials,
        }

    # ── lazy reload ──
    def _reload(self, trial_id: str) -> dict[str, Any]:
        src = self._sources[trial_id]
        path = src.path.split('!')[0]
        if path not in self._cache:
            items = []
            for source, doc in iter_documents(path):
                items.extend(extract_records(doc, source))
            self._cache[path] = items
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        for kind, rec, _ in self._cache[path]:
            if kind == 'trial' and rec.get('trial_id') == trial_id:
                return rec
        raise KeyError(f'trial {trial_id} no longer found in {src.path}')


def load_dataset(
    paths: PathLike | Iterable[PathLike],
    validate: bool | str = True,
    *,
    on_invalid: str = 'skip',
    keep_streams: bool = True,
    infer_gravity_sign: bool = True,
    gravity_sign_order: tuple[str, ...] = ('vote', 'engine'),
    simulated: str = 'include',
    verbose: bool = False,
) -> Dataset:
    """Load every record under ``paths`` into a :class:`Dataset`.

    validate: True / 'fast' (structure + vectorised stream checks), 'full' (jsonschema on everything,
        slow), False (no validation).
    on_invalid: 'skip' (default; listed in ``issues``), 'keep', or 'raise'.
    keep_streams: False keeps only metadata in memory and re-reads a trial's file on demand.
    infer_gravity_sign: let the trial votes set the gravity sign: for sessions without a calibrated sign,
        and over a calibrated sign they clearly contradict (>= 5 votes, >= 80 % against). False keeps every
        calibrated sign (flagged when contradicted) and skips the vote step.
    gravity_sign_order: order of the fallbacks after the calibration: ('vote', 'engine') (default, the
        docs/03 section 5.2 order) or ('engine', 'vote') to put the engine expectation first; see
        ``conventions.resolve_gravity_sign``.
    simulated: 'include' (default), 'exclude' (real phones only) or 'only'.
    """
    if on_invalid not in ('skip', 'keep', 'raise'):
        raise ValueError("on_invalid must be 'skip', 'keep' or 'raise'")
    if simulated not in ('include', 'exclude', 'only'):
        raise ValueError("simulated must be 'include', 'exclude' or 'only'")
    mode = None if validate is False else ('full' if validate == 'full' else 'fast')
    issues: list[LoadIssue] = []
    best: dict[str, dict[str, tuple[float, int, dict[str, Any], str]]] = {'participant': {}, 'session': {}, 'trial': {}}
    order = 0
    n_dup = {'participant': 0, 'session': 0, 'trial': 0}
    dups: list[dict[str, Any]] = []
    for source, doc in iter_documents(paths, issues):
        for kind, rec, src in extract_records(doc, source, issues):
            rid = rec.get(_ID_FIELD[kind]) if isinstance(rec, Mapping) else None
            if mode is not None:
                errs = _schema.validate(rec, _KIND_DEF[kind], mode=mode, max_errors=10)
                if errs:
                    msg = f'schema: {errs[0]}' + (f' (+{len(errs) - 1} more)' if len(errs) > 1 else '')
                    if on_invalid == 'raise':
                        raise _schema.SchemaValidationError(_KIND_DEF[kind], [f'{src}: {e}' for e in errs])
                    issues.append(LoadIssue('error', src, kind, rid, msg))
                    if on_invalid == 'skip':
                        continue
            if not isinstance(rid, str):
                issues.append(LoadIssue('error', src, kind, None, f'{kind} without a usable id; skipped'))
                continue
            if kind == 'trial':
                try:
                    rec = compact_streams(rec)
                    _check_stream_lengths(rec)
                except (KeyError, TypeError, ValueError) as e:
                    issues.append(LoadIssue('error', src, kind, rid, f'unusable streams: {e}'))
                    continue
            key = _record_time(kind, rec)
            order += 1
            prev = best[kind].get(rid)
            if prev is not None:
                n_dup[kind] += 1
                if key < prev[0] or (key == prev[0] and not _wins_tie(kind, rec, prev[2])):
                    dups.append({'kind': kind, 'record_id': rid, 'kept_source': prev[3], 'dropped_source': src,
                                 'identical_time': key == prev[0]})
                    continue
                dups.append({'kind': kind, 'record_id': rid, 'kept_source': src, 'dropped_source': prev[3],
                             'identical_time': False})
            best[kind][rid] = (key, order, rec, src)
    for kind, n in n_dup.items():
        if n:
            issues.append(LoadIssue('info', '(all)', kind, None, f'{n} duplicate {kind} record(s) dropped (kept newest)'))

    ds = Dataset()
    ds._keep_streams = keep_streams
    sessions = {k: v[2] for k, v in best['session'].items()}
    trials = {k: v[2] for k, v in best['trial'].items()}
    # participant profiles: bundle profiles + session snapshots, newest wins
    profiles: dict[str, tuple[float, dict[str, Any]]] = {k: (v[0], v[2]) for k, v in best['participant'].items()}
    for s in sessions.values():
        p = s.get('participant')
        if isinstance(p, Mapping) and isinstance(p.get('participant_id'), str):
            k = _time_key(p.get('updated_at'))
            cur = profiles.get(p['participant_id'])
            if cur is None or k > cur[0]:
                profiles[p['participant_id']] = (k, p)

    if simulated != 'include':
        want = simulated == 'only'
        sessions = {k: v for k, v in sessions.items() if bool(v.get('simulated')) == want}
        trials = {k: v for k, v in trials.items() if bool(v.get('simulated')) == want}

    # norm_v1 per session: gravity sign (calibration -> trial vote -> engine -> +1 by default; votes that
    # clearly contradict a calibrated sign override it) and rate mapping
    votes: dict[str, list[int | None]] = {}
    for t in trials.values():
        votes.setdefault(t['session_id'], []).append(trial_gravity_vote(t))
    for sid in sorted(set(sessions) | set(votes)):
        ds._norm[sid] = session_normalization(sessions.get(sid), votes.get(sid, []), infer=infer_gravity_sign,
                                              order=gravity_sign_order)
    for sid, nm in ds._norm.items():
        gs = nm.gravity
        if gs.warning:
            issues.append(LoadIssue('warning', '(gravity)', 'session', sid,
                                    f'gravity sign {gs.sign:+d} from {gs.source}: {gs.detail}'))
        if not nm.rates.verified:
            issues.append(LoadIssue('info', '(axis)', 'session', sid, nm.rates.detail))
        elif nm.rates.detail:
            issues.append(LoadIssue('warning', '(axis)', 'session', sid, f'rotationRate remapped: {nm.rates.detail}'))

    # tables
    prow = [participant_row(p) for _, p in profiles.values()]
    ds.participants = pd.DataFrame(prow) if prow else pd.DataFrame(columns=list(participant_row({}).keys()))
    srows = [session_row(s, ds._norm.get(sid)) for sid, s in sessions.items()]
    ds.sessions = pd.DataFrame(srows) if srows else pd.DataFrame(columns=list(session_row({}, Normalization()).keys()))
    trows = [trial_row(t) for t in trials.values()]
    tdf = pd.DataFrame(trows) if trows else pd.DataFrame(columns=list(trial_row({}).keys()))
    tdf = _join_trials(tdf, ds.participants, ds.sessions, sessions, ds._norm, issues)
    ds.trials = _tidy(tdf)
    ds.duplicates = pd.DataFrame(dups, columns=DUPLICATE_COLUMNS)
    if len(ds.sessions):
        ds.sessions = ds.sessions.sort_values(['participant_id', 'started_at']).reset_index(drop=True)
        cnt = tdf.groupby('session_id').agg(n_trials_loaded=('trial_id', 'size'),
                                            n_accepted_loaded=('accepted', lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())))
        ds.sessions = ds.sessions.merge(cnt, how='left', left_on='session_id', right_index=True)
        ds.sessions = _tidy(ds.sessions)
    if len(ds.participants):
        ns = ds.sessions.groupby('participant_id').size().rename('n_sessions') if len(ds.sessions) else pd.Series(dtype=int, name='n_sessions')
        nt = tdf.groupby('participant_id').size().rename('n_trials') if len(tdf) else pd.Series(dtype=int, name='n_trials')
        ds.participants = (ds.participants.merge(ns, how='left', left_on='participant_id', right_index=True)
                           .merge(nt, how='left', left_on='participant_id', right_index=True))
        ds.participants[['n_sessions', 'n_trials']] = ds.participants[['n_sessions', 'n_trials']].fillna(0).astype(int)
        ds.participants = _tidy(ds.participants.sort_values('participant_id').reset_index(drop=True))

    ds._sessions = sessions
    ds._participants = {k: v[1] for k, v in profiles.items()}
    for tid, rec in trials.items():
        ds._sources[tid] = _Source(best['trial'][tid][3], 'trial')
        if keep_streams:
            ds._trials[tid] = rec
        else:
            ds._trials[tid] = {k: v for k, v in rec.items() if k != 'streams'}
    ds.issues = pd.DataFrame([vars(i) for i in issues], columns=['level', 'source', 'kind', 'record_id', 'message'])
    n_err = sum(1 for i in issues if i.level == 'error')
    if n_err:
        warnings.warn(f'{n_err} record/file error(s) while loading; see Dataset.issues', stacklevel=2)
    if verbose:
        print(ds)
    return ds


def _check_stream_lengths(rec: Mapping[str, Any]) -> None:
    st = rec['streams']
    for name, cols in _STREAM_COLUMNS.items():
        s = st.get(name)
        if not s:
            continue
        n = len(s['t'])
        for c in cols:
            if c in s and s[c] is not None and len(s[c]) != n:
                raise ValueError(f'streams.{name}.{c} has {len(s[c])} values, t has {n}')


DUPLICATE_COLUMNS = ['kind', 'record_id', 'kept_source', 'dropped_source', 'identical_time']


def _join_trials(tdf: pd.DataFrame, parts: pd.DataFrame, sess_df: pd.DataFrame, sessions: Mapping[str, Any],
                 norms: Mapping[str, Normalization], issues: list[LoadIssue]) -> pd.DataFrame:
    if not len(tdf):
        return tdf
    pcols = ['participant_id', 'sex', 'age_years', 'age_band', 'handedness_self', 'ehi_lq', 'ehi_class', 'phone_hand',
             'height_cm', 'dominant_hand']
    if len(parts):
        tdf = tdf.merge(parts[pcols], how='left', on='participant_id')
    else:
        for c in pcols[1:]:
            tdf[c] = None
    scols = ['session_id', 'platform', 'browser', 'engine', 'model_hint', 'phone_model', 'status']
    if len(sess_df):
        have = [c for c in scols if c in sess_df.columns]
        tdf = tdf.merge(sess_df[have].rename(columns={'status': 'session_status'}), how='left', on='session_id')
    for c in ('platform', 'browser', 'engine', 'model_hint', 'phone_model', 'session_status'):
        if c not in tdf.columns:
            tdf[c] = None
    orphan = ~tdf['session_id'].isin(list(sessions))
    for sid in tdf.loc[orphan, 'session_id'].unique():
        issues.append(LoadIssue('warning', '(join)', 'trial', None, f'trials of session {sid} loaded without its session record'))
    nm = [norms.get(s) or Normalization() for s in tdf['session_id']]
    tdf['gravity_sign'] = [n.gravity.sign for n in nm]
    tdf['gravity_sign_source'] = [n.gravity.source for n in nm]
    tdf['gravity_sign_warning'] = [n.gravity.warning for n in nm]
    tdf['rr_mapping'] = [n.rates.verdict for n in nm]
    tdf['rr_verified'] = [n.rates.verified for n in nm]
    tdf['device_model'] = [m if isinstance(m, str) and m else (p if isinstance(p, str) and p else None)
                           for m, p in zip(tdf['model_hint'], tdf['phone_model'])]
    # docs/02 section 10 rule 3: two trial ids with the same (session, trial_index, attempt) are both kept
    key = tdf.groupby(['session_id', 'trial_index', 'attempt'])['trial_id'].transform('size')
    tdf['dup_attempt_key'] = key > 1
    for (sid, ti, at), g in tdf[tdf['dup_attempt_key']].groupby(['session_id', 'trial_index', 'attempt']):
        issues.append(LoadIssue('warning', '(dedupe)', 'trial', None,
                                f'session {sid} trial_index {ti} attempt {at}: {len(g)} different trial ids (all kept)'))
    acc = tdf['accepted'].fillna(False).astype(bool)
    n_acc = acc.groupby([tdf['session_id'], tdf['trial_index']]).transform('sum')
    tdf['multi_accepted'] = n_acc > 1
    # block kind from the session plan
    kinds = {}
    for sid, s in sessions.items():
        for b in ((s.get('protocol') or {}).get('plan') or []):
            kinds[(sid, b.get('block_index'))] = b.get('kind')
    tdf['block_kind'] = [kinds.get((s, b)) for s, b in zip(tdf['session_id'], tdf['block_index'])]
    tdf['hand_is_dominant'] = [
        (None if (d is None or (isinstance(d, float) and math.isnan(d)) or h is None) else (h == d))
        for h, d in zip(tdf['hand'], tdf['dominant_hand'])
    ]
    last = tdf.groupby(['session_id', 'trial_index'])['attempt'].transform('max')
    tdf['n_attempts_trial'] = tdf.groupby(['session_id', 'trial_index'])['attempt'].transform('size')
    tdf['is_last_attempt'] = tdf['attempt'] == last
    tdf = tdf.sort_values(['participant_id', 'session_id', 'trial_index', 'attempt'], kind='stable').reset_index(drop=True)
    return tdf


def write_json(obj: Any, path: PathLike) -> Path:
    """Write a record / bundle as UTF-8 JSON (NaN-free: use Dataset.record / to_bundle outputs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, allow_nan=False), encoding='utf-8')
    return p

