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
loaded (files are read in sorted order), except that a trial copy saved LATER in the app's sequence wins
a tie whatever the file order (the app re-saves an attempt under the same id and created_at: a
protective copy 'review_interrupted' / 'hand_not_reported' < the decided copy < a revoked one,
'redo_requested'); every dropped copy is listed in ``Dataset.duplicates``. Two
DIFFERENT trial ids with the same (session_id, trial_index, attempt), and several accepted attempts of
one planned trial, are kept and flagged (columns ``dup_attempt_key`` / ``multi_accepted``).
Records failing schema validation are skipped (``on_invalid='skip'``, default), kept (``'keep'``) or raise
(``'raise'``); every problem is listed in ``Dataset.issues``. Every schema 1.x record loads: the fields
added in 1.1 and 1.2 are optional (a record WITHOUT them is valid and complete), so only a missing
REQUIRED key or a wrong value skips a record. The optional schema 1.2 streams (``quaternion_abs``,
``orientation_abs``, ``magnetometer``, ``screen_events``, ``orientation.compass_accuracy``) never cost a
trial: an inconsistent one (lengths differ from its ``t``) is ignored and listed in ``Dataset.issues``.

Each session gets its norm_v2 normalisation (``conventions.session_normalization``): the gravity sign
(guided calibration -> the engine prior confirmed by the trial votes, 'engine_vote' -> trial vote ->
engine expectation -> implicit hold -> +1; trial votes that clearly contradict a calibrated sign override
it, ``gravity_sign_order=('engine', 'vote')`` puts the engine first) and the rotationRate mapping
(``calibration.axis_check``; without a definite one, the closure check of up to ``rate_check_trials``
complete trials, ``conventions.infer_rate_mapping``); ``Dataset.signals`` applies it.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schema as _schema
from .conventions import (
    ORIENTATIONS, SCHEMA_VERSION, DeviceSignals, GravitySign, Normalization, RateMapping, calibration_method,
    infer_rate_mapping, normalize_motion, rate_mapping_from_calibration, rotation_rate_to_device, session_engine,
    session_normalization, trial_gravity_vote,
)

PathLike = str | Path
_JSON_SUFFIXES = ('.json', '.json.txt', '.json.gz', '.jsonl', '.ndjson', '.zip')

MOTION_NUM_COLUMNS = ('t', 'ts', 'acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma')
ORIENTATION_NUM_COLUMNS = ('t', 'ts', 'alpha', 'beta', 'gamma', 'compass')
QUATERNION_NUM_COLUMNS = ('t', 'x', 'y', 'z', 'w')
MAGNETOMETER_NUM_COLUMNS = ('t', 'x', 'y', 'z')
#: Numeric columns of the schema 1.0 streams: a column whose length differs from its ``t`` makes the trial
#: unusable (skipped, listed in ``Dataset.issues``).
_STREAM_COLUMNS = {'motion': MOTION_NUM_COLUMNS, 'orientation': ORIENTATION_NUM_COLUMNS, 'quaternion': QUATERNION_NUM_COLUMNS}
#: Optional numeric columns added by schema 1.2 (best-effort browser extras): an inconsistent one is ignored
#: (left as stored, listed in ``Dataset.issues`` as a warning) and never costs the trial.
_OPTIONAL_STREAM_COLUMNS = {
    'orientation': ('compass_accuracy',),
    'quaternion_abs': QUATERNION_NUM_COLUMNS,
    'orientation_abs': ORIENTATION_NUM_COLUMNS + ('compass_accuracy',),
    'magnetometer': MAGNETOMETER_NUM_COLUMNS,
}
#: TouchRecord fields (schema 1.2 added the six after 'y'; absent in older records).
TOUCH_COLUMNS = ('t', 'type', 'n', 'x', 'y', 'radius_x', 'radius_y', 'force', 'rotation_angle', 'x_norm', 'y_norm')
TOUCH_DETAIL_KEYS = ('radius_x', 'radius_y', 'force', 'rotation_angle', 'x_norm', 'y_norm')
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
    compass_accuracy: np.ndarray | None = None  # schema 1.2, iOS webkitCompassAccuracy (deg)


@dataclass
class QuaternionArrays:
    t: np.ndarray
    q: np.ndarray  # (q,4) [x, y, z, w]
    sensor: str
    frequency_hz: float


@dataclass
class MagnetometerArrays:
    t: np.ndarray
    xyz: np.ndarray  # (m,3) uT, device frame
    frequency_hz: float


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


@dataclass
class TrialStreams:
    """All streams of one trial as numpy arrays (null -> NaN). Raw device frame, RAW gravity sign.

    t: (n,) handler time, ms since t0. ts: event.timeStamp - first raw timeStamp (ms, NaN if missing).
    acc / accg: (n,3) m/s^2. rr: (n,3) rotationRate in schema order [alpha, beta, gamma] (deg/s).
    Use :meth:`gyro` for the device-frame vector [x, y, z] and ``Dataset.signals`` (or
    :func:`conventions.normalize_motion`) for the fully normalised physical signals.
    Schema 1.2 extras (None / empty when absent or inconsistent): quaternion_abs (AbsoluteOrientationSensor),
    orientation_abs ('deviceorientationabsolute'), magnetometer, screen_events (t, type, angle); touches
    carry radius_x / radius_y / force / rotation_angle / x_norm / y_norm (NaN in older records).
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
    quaternion_abs: QuaternionArrays | None = None
    orientation_abs: OrientationArrays | None = None
    magnetometer: MagnetometerArrays | None = None
    screen_events: pd.DataFrame = field(default_factory=lambda: _empty_frame(('t', 'type', 'angle')))

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

    def opt_col(st: Mapping[str, Any], k: str, length: int) -> np.ndarray | None:
        """An optional column: None when absent, null, non-numeric or of another length than t."""
        v = st.get(k)
        if v is None:
            return None
        try:
            a = _arr(v)
        except (TypeError, ValueError):
            return None
        return a if a.ndim == 1 and a.shape[0] == length else None

    def orientation(o: Any) -> OrientationArrays | None:
        to = _arr(o['t'])
        k = to.shape[0]
        ang = np.column_stack([col(o, a, k) for a in ('alpha', 'beta', 'gamma')]) if k else np.zeros((0, 3))
        return OrientationArrays(to, col(o, 'ts', k), ang, col(o, 'compass', k), o.get('absolute'),
                                 _num(o.get('ts_first_raw')), opt_col(o, 'compass_accuracy', k))

    def quaternion(q: Any) -> QuaternionArrays:
        tq = _arr(q['t'])
        k = tq.shape[0]
        Q = np.column_stack([col(q, a, k) for a in 'xyzw']) if k else np.zeros((0, 4))
        return QuaternionArrays(tq, Q, q.get('sensor', ''), float(q.get('frequency_hz') or float('nan')))

    def optional(build: Any, st: Any) -> Any:
        """A schema 1.2 stream: None when absent or inconsistent (it never makes the trial unusable)."""
        if not isinstance(st, Mapping) or st.get('t') is None:
            return None
        try:
            return build(st)
        except (KeyError, TypeError, ValueError):
            return None

    def magnetometer(mg: Any) -> MagnetometerArrays:
        tm = _arr(mg['t'])
        k = tm.shape[0]
        xyz = np.column_stack([col(mg, a, k) for a in 'xyz']) if k else np.zeros((0, 3))
        return MagnetometerArrays(tm, xyz, float(mg.get('frequency_hz') or float('nan')))

    o = s.get('orientation')
    ori = orientation(o) if o else None
    q = s.get('quaternion')
    quat = quaternion(q) if q else None
    touches = pd.DataFrame(list(s.get('touches') or []), columns=list(TOUCH_COLUMNS))
    cues = pd.DataFrame(list(s.get('cues') or []), columns=['t', 'kind'])
    sev = [e for e in (s.get('screen_events') or []) if isinstance(e, Mapping)]
    screen_events = pd.DataFrame(sev, columns=['t', 'type', 'angle'])
    return TrialStreams(
        t=t, ts=col(m, 'ts', n), acc=acc, accg=accg, rr=rr, interval_raw=_num(m.get('interval_raw')),
        ts_first_raw=_num(m.get('ts_first_raw')), orientation=ori, quaternion=quat, touches=touches, cues=cues,
        quaternion_abs=optional(quaternion, s.get('quaternion_abs')),
        orientation_abs=optional(orientation, s.get('orientation_abs')),
        magnetometer=optional(magnetometer, s.get('magnetometer')), screen_events=screen_events,
    )


def compact_streams(record: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy of a trial whose numeric stream columns are float arrays (NaN for null).

    Optional schema 1.2 columns are converted only when they are numeric and as long as their stream's
    ``t``; otherwise they stay as stored (``_check_stream_lengths`` reports them)."""
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
    for name, cols in _OPTIONAL_STREAM_COLUMNS.items():
        s = st.get(name)
        if not isinstance(s, Mapping) or s.get('t') is None:
            continue
        s2 = dict(s)
        n = len(s2['t'])
        for c in cols:
            v = s2.get(c)
            if v is None or isinstance(v, np.ndarray) or len(v) != n:
                continue
            try:
                s2[c] = _arr(v)
            except (TypeError, ValueError):
                pass
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
#: protocol 2.0.0: the choice trial saved before the hand question is answered (its hand is a placeholder).
HAND_NOT_REPORTED_NOTE = 'hand_not_reported'
#: protocol 2.0.0: an accepted trial revoked from the pause sheet ("Redo last movement" / "wrong hand").
REDO_REQUESTED_NOTE = 'redo_requested'
#: Rank of a trial copy in the app's save sequence (by outcome.note): protective copies saved before a
#: decision < the decided copy (every other note) < a revocation, which can only follow a decision.
_TRIAL_NOTE_RANK = {REVIEW_INTERRUPTED_NOTE: 0, HAND_NOT_REPORTED_NOTE: 0, REDO_REQUESTED_NOTE: 2}


def _trial_rank(r: Mapping[str, Any]) -> int:
    return _TRIAL_NOTE_RANK.get((r.get('outcome') or {}).get('note'), 1)


def _wins_tie(kind: str, new: Mapping[str, Any], old: Mapping[str, Any]) -> bool:
    """Same id and same time: does ``new`` replace ``old``? Only for trials: the app re-saves an attempt
    under the same trial_id and created_at (a protective copy before the review or the hand question,
    then the decided copy, possibly revoked later: 'redo_requested'), so the copy later in that sequence
    wins whatever order the files were loaded in (:data:`_TRIAL_NOTE_RANK`); equal ranks keep the first."""
    if kind != 'trial':
        return False
    return _trial_rank(new) > _trial_rank(old)


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


#: model_hint values that name no model: iOS only says 'iPhone' / 'iPad'; Chrome's reduced UA says 'K'.
_GENERIC_MODEL_HINTS = frozenset({'', 'k', 'iphone', 'ipad', 'ipod', 'ios', 'macintosh', 'mac', 'mobile', 'android'})


def _finite_num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def device_model(platform: Any, model_hint: Any, phone_model: Any = None, screen_w: Any = None,
                 screen_h: Any = None, dpr: Any = None) -> str | None:
    """Best available device model of a session (``Dataset.sessions`` / ``trials`` column ``device_model``).

    1. ``device.model_hint`` when it names a model (Android, from the browser);
    2. else a screen proxy '<family> <short>x<long>@<dpr>', e.g. 'iPhone 390x844@3': iOS reports only
       'iPhone', but the screen size in CSS px and the pixel ratio tell most models apart (portrait and
       landscape captures give the same proxy);
    3. else the participant's self-reported ``phone_model`` (profiles up to schema 1.1; v2 writes null);
    4. else the generic hint itself, or None.
    """
    hint = model_hint.strip() if isinstance(model_hint, str) else ''
    if hint and hint.lower() not in _GENERIC_MODEL_HINTS:
        return hint
    w, h, r = _finite_num(screen_w), _finite_num(screen_h), _finite_num(dpr)
    if w and h and r and w > 0 and h > 0 and r > 0:
        family = hint if hint and hint.lower() != 'k' else ('iPhone' if platform == 'ios' else str(platform or 'device'))
        return f'{family} {min(w, h):g}x{max(w, h):g}@{r:g}'
    if isinstance(phone_model, str) and phone_model.strip():
        return phone_model.strip()
    return hint or None


def _plan_summary(plan: Any) -> dict[str, Any]:
    blocks = [b for b in (plan or []) if isinstance(b, Mapping)]
    required = [b for b in blocks if b.get('optional') is not True]
    return {
        'n_blocks': len(blocks),
        'n_optional_blocks': len(blocks) - len(required),
        'n_planned_required': sum(len(b.get('trials') or []) for b in required),
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
        **_plan_summary(prot.get('plan')),
        'language': st.get('language'), 'audio': st.get('audio'), 'speech': st.get('speech'),
        # schema 1.2 (absent = 'tap': every trial was reviewed)
        'review_mode': st.get('review_mode') or ('tap' if s else None),
        'next_trial_index': prog.get('next_trial_index'), 'n_attempts': prog.get('n_attempts'),
        'n_accepted': prog.get('n_accepted'), 'consent_version': (s.get('consent') or {}).get('version'),
    }
    pr = participant_row(p)  # never the name (profile.nickname): it stays in the raw record only
    for k in ('sex', 'age_years', 'age_band', 'handedness_self', 'ehi_lq', 'ehi_class', 'phone_hand', 'height_cm',
              'arm_issue', 'phone_model', 'phone_case', 'dominant_hand'):
        row[k] = pr[k]
    row.update({
        'platform': d.get('platform'), 'browser': d.get('browser'), 'os_version': d.get('os_version'),
        'model_hint': d.get('model_hint'), 'standalone': d.get('standalone'), 'screen_w': d.get('screen_w'),
        'screen_h': d.get('screen_h'), 'dpr': d.get('dpr'), 'device_language': d.get('language'),
        'hardware_concurrency': d.get('hardware_concurrency'), 'device_memory_gb': d.get('device_memory_gb'),
        'user_agent': d.get('user_agent'),
        'device_model': device_model(d.get('platform'), d.get('model_hint'), p.get('phone_model'), d.get('screen_w'),
                                     d.get('screen_h'), d.get('dpr')),
        'permission_state': cap.get('permission_state'),
        'generic_relative_orientation': cap.get('generic_relative_orientation'),
        'measured_motion_hz': meas.get('motion_hz'), 'measured_dt_ms_median': meas.get('motion_dt_ms_median'),
        'measured_acc_available': meas.get('acc_available'), 'measured_rotation_available': meas.get('rotation_available'),
        'engine': d.get('engine'), 'in_app_browser': d.get('in_app_browser'),
        'calibration_present': bool(s.get('calibration')),
        # schema 1.2: guided_v1 (the calibration screen; also every calibration without `method`) or
        # implicit_v1 (from the first still upright trial); 'none' = no calibration object
        'calibration_method': calibration_method(s.get('calibration')) or ('none' if not s.get('calibration') else 'unknown'),
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
        ri = norm.rate_inference
        row.update({'rr_mapping': norm.rates.verdict, 'rr_units': norm.rates.units,
                    'rr_verified': norm.rates.verified, 'rr_source': norm.rates.source,
                    'rr_detail': norm.rates.detail,
                    # the closure check of the trials (conventions.infer_rate_mapping), when it ran
                    'rr_closure_best': (ri.medians[0][0] if ri is not None and ri.medians else None),
                    'rr_closure_deg': ri.closure_deg if ri is not None else None,
                    'rr_closure_runner_up': ri.runner_up if ri is not None else None,
                    'rr_closure_runner_up_deg': ri.runner_up_deg if ri is not None else None,
                    'rr_closure_trials': ri.n_trials if ri is not None else None,
                    'rr_closure_verified': ri.verified if ri is not None else None,
                    'rr_closure_disagrees': norm.rate_check_disagrees,
                    'gyro_bias_subtracted': norm.gyro_bias is not None,
                    'norm_version': norm.version, 'norm_flags': ';'.join(norm.flags)})
    return row


def _touch_summary(touches: Any, first_touch_ms: Any, go_ms: Any) -> dict[str, Any]:
    """Contact details (schema 1.2) of the first touch after GO (the nose-contact candidate: the touchstart
    at ``timing.first_touch_ms``, else the first touchstart at or after GO) and the largest radius / force
    of every touchstart. In landscape grips the first touch is often a gripping finger: the full list stays
    in the record (``Dataset.streams(...).touches``)."""
    starts = [x for x in (touches or []) if isinstance(x, Mapping) and x.get('type') == 'start']
    first = None
    if first_touch_ms is not None:
        first = next((x for x in starts if x.get('t') == first_touch_ms), None)
    if first is None and _finite_num(go_ms) is not None:
        first = next((x for x in starts if _finite_num(x.get('t')) is not None and x['t'] >= go_ms), None)
    radii = [r for x in starts for r in (_finite_num(x.get('radius_x')), _finite_num(x.get('radius_y'))) if r is not None]
    forces = [f for f in (_finite_num(x.get('force')) for x in starts) if f is not None]
    out: dict[str, Any] = {f'first_touch_{k}': (_finite_num(first.get(k)) if first else None) for k in TOUCH_DETAIL_KEYS}
    out.update({'n_touch_starts': len(starts), 'touch_radius_max': max(radii) if radii else None,
                'touch_force_max': max(forces) if forces else None})
    return out


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

    # schema 1.2: the return of the arm after the end (kept in the motion stream; the segment ends before it)
    has_return = 'return_onset_ms' in tm or 'return_end_ms' in tm
    r_on, r_end = tm.get('return_onset_ms'), tm.get('return_end_ms')
    v2 = {
        'hand_source': c.get('hand_source') or 'instructed',
        'return_onset_ms': r_on, 'return_end_ms': r_end,
        'return_observed': (r_on is not None) if has_return else None,
        'return_delay_ms': diff(r_on, tm.get('end_detected_ms')), 'return_duration_ms': diff(r_end, r_on),
        'post_end_ms': diff(tm.get('stop_ms'), tm.get('end_detected_ms')),
        'n_quaternion_abs': q.get('n_quaternion_abs'), 'n_orientation_abs': q.get('n_orientation_abs'),
        'has_quaternion_abs_stream': streams.get('quaternion_abs') is not None,
        'has_orientation_abs_stream': streams.get('orientation_abs') is not None,
        'has_magnetometer_stream': streams.get('magnetometer') is not None,
        'n_screen_events': len(streams.get('screen_events') or []),
        **_touch_summary(streams.get('touches'), tm.get('first_touch_ms'), tm.get('go_ms')),
    }

    row = {
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
    row.update(v2)
    return row


_INT_COLS = ('block_index', 'trial_index', 'trial_in_block', 'attempt', 'rep', 'onset_index', 'end_index',
             'baseline_n', 'n_motion', 'n_orientation', 'n_quaternion', 'gaps_over_50ms', 'false_starts', 'y_right',
             'n_touches', 'n_warnings', 'age_years', 'block_tier', 'n_quaternion_abs', 'n_orientation_abs',
             'n_screen_events', 'n_touch_starts', 'n_blocks', 'n_optional_blocks', 'n_planned_required',
             'rr_closure_trials')
_BOOL_COLS = ('accepted', 'simulated', 'orientation_match', 'acc_available', 'rotation_available', 'hidden_during',
              'hand_is_dominant', 'hand_is_phone_hand', 'is_last_attempt', 'gravity_sign_warning', 'calibration_present',
              'rr_verified', 'dup_attempt_key', 'multi_accepted', 'block_optional', 'return_observed',
              'has_quaternion_abs_stream', 'has_orientation_abs_stream', 'has_magnetometer_stream',
              'rr_closure_verified', 'rr_closure_disagrees', 'gyro_bias_subtracted')


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
      sessions      one row per session: protocol, participant snapshot (never the name), device
                    (``device_model``: the model hint, or the screen proxy on iOS), calibration
                    (``calibration_method``) and the resolved norm_v2: gravity sign (``gravity_sign``,
                    ``gravity_sign_source``, ``gravity_sign_warning``) and rotationRate mapping (``rr_mapping``,
                    ``rr_units``, ``rr_verified``, ``rr_source``, the closure check ``rr_closure_*``).
      trials        one row per ATTEMPT (redo attempts included; filter ``accepted``): condition (with
                    ``hand_source``: 'reported' for a street choice trial), outcome, timing (with the schema
                    1.2 return of the arm: ``return_onset_ms``, ``return_end_ms``, ``return_observed``,
                    ``return_delay_ms``, ``return_duration_ms``), segmentation, detected, quality, the
                    schema 1.2 extras (``has_quaternion_abs_stream``, ``has_orientation_abs_stream``,
                    ``n_quaternion_abs``, ``n_orientation_abs``, ``has_magnetometer_stream``,
                    ``n_screen_events``, ``first_touch_radius_x`` ... ``first_touch_y_norm``,
                    ``touch_radius_max``, ``touch_force_max``), participant columns (sex, age_years, age_band,
                    handedness_self, ehi_lq, ehi_class, phone_hand, dominant_hand, hand_is_dominant,
                    hand_is_phone_hand), session columns (platform, engine, device_model, calibration_method,
                    gravity_sign, rr_mapping, ...), ``block_kind``, ``block_tier``, ``block_optional``,
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
        """norm_v2 of a session (unknown sessions: default sign +1 with a warning, W3C assumed)."""
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
        """Motion stream in physical units under the session's norm_v2 (W3C sign, rates about x, y, z)."""
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
            'format': 'nose-touch-export', 'schema_version': SCHEMA_VERSION, 'app_version': 'motionlab',
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
    rate_mapping: str = 'auto',
    rate_check_trials: int = 16,
) -> Dataset:
    """Load every record under ``paths`` into a :class:`Dataset`.

    validate: True / 'fast' (structure + vectorised stream checks), 'full' (jsonschema on everything,
        slow), False (no validation).
    on_invalid: 'skip' (default; listed in ``issues``), 'keep', or 'raise'.
    keep_streams: False keeps only metadata in memory and re-reads a trial's file on demand.
    infer_gravity_sign: let the trial votes set the gravity sign: for sessions without a calibrated sign,
        and over a calibrated sign they clearly contradict (>= 5 votes, >= 80 % against). False keeps every
        calibrated sign (flagged when contradicted) and never lets the votes set a sign (they may still
        confirm the engine expectation: source 'engine_vote').
    gravity_sign_order: order of the fallbacks after the calibration and the engine + vote consensus:
        ('vote', 'engine') (default, the docs/03 section 5.2 order) or ('engine', 'vote') to put the engine
        expectation first; see ``conventions.resolve_gravity_sign``.
    simulated: 'include' (default), 'exclude' (real phones only) or 'only'.
    rate_mapping: where the rotationRate mapping comes from. 'auto' (default): the calibration's axis_check,
        and for sessions without a definite one (implicit calibration, schema 1.0, a failed tilt) the
        closure check of their trials (``conventions.infer_rate_mapping``, verified when decisive; W3C
        deg/s assumed and flagged otherwise). 'always': also run the closure check on calibrated sessions,
        as a cross-check (the axis_check still wins; a decisive disagreement is flagged
        ``rr_closure_disagrees``). 'calibration': never run it (norm_v1 behaviour).
    rate_check_trials: at most this many complete trials per session (evenly spread) for the closure check.
    """
    if on_invalid not in ('skip', 'keep', 'raise'):
        raise ValueError("on_invalid must be 'skip', 'keep' or 'raise'")
    if simulated not in ('include', 'exclude', 'only'):
        raise ValueError("simulated must be 'include', 'exclude' or 'only'")
    if rate_mapping not in ('auto', 'always', 'calibration'):
        raise ValueError("rate_mapping must be 'auto', 'always' or 'calibration'")
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
                    optional_problems = _check_stream_lengths(rec)
                except (KeyError, TypeError, ValueError) as e:
                    issues.append(LoadIssue('error', src, kind, rid, f'unusable streams: {e}'))
                    continue
                for msg in optional_problems:
                    issues.append(LoadIssue('warning', src, kind, rid, f'optional stream data ignored: {msg}'))
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

    # norm_v2 per session: gravity sign (guided calibration -> engine + vote consensus -> vote -> engine ->
    # implicit hold -> +1; votes that clearly contradict a calibrated sign override it) and rate mapping
    # (axis_check, else the closure check of the trials)
    votes: dict[str, list[int | None]] = {}
    upright: dict[str, list[int | None]] = {}
    by_session: dict[str, list[dict[str, Any]]] = {}
    for t in trials.values():
        sid = t['session_id']
        v = trial_gravity_vote(t)
        votes.setdefault(sid, []).append(v)
        if ORIENTATIONS.get((t.get('condition') or {}).get('orientation'), {}).get('rotation') in ('upright', 'upside_down'):
            upright.setdefault(sid, []).append(v)
        by_session.setdefault(sid, []).append(t)
    for sid in sorted(set(sessions) | set(votes)):
        sess = sessions.get(sid)
        cal = (sess or {}).get('calibration') or None
        inference = None
        if rate_mapping == 'always' or (rate_mapping == 'auto' and not rate_mapping_from_calibration(cal).verified):
            inference = _session_rate_check(by_session.get(sid, []), cal, session_engine((sess or {}).get('device'))[0],
                                            rate_check_trials)
        ds._norm[sid] = session_normalization(sess, votes.get(sid, []), infer=infer_gravity_sign,
                                              order=gravity_sign_order, upright_votes=upright.get(sid),
                                              rate_inference=inference)
    for sid, nm in ds._norm.items():
        gs = nm.gravity
        if gs.warning:
            issues.append(LoadIssue('warning', '(gravity)', 'session', sid,
                                    f'gravity sign {gs.sign:+d} from {gs.source}: {gs.detail}'))
        if not nm.rates.verified:
            issues.append(LoadIssue('info', '(axis)', 'session', sid, nm.rates.detail))
        elif nm.rates.source == 'closure':
            r = nm.rates
            issues.append(LoadIssue('warning' if r.detail else 'info', '(axis)', 'session', sid,
                                    f'rotationRate mapping {r.key} from the closure check: {r.closure_deg:.1f} deg '
                                    f'vs {r.runner_up} {r.runner_up_deg:.1f} deg over {r.n_trials} trials'
                                    + (f' (remapped: {r.detail})' if r.detail else '')))
        elif nm.rates.detail:
            issues.append(LoadIssue('warning', '(axis)', 'session', sid, f'rotationRate remapped: {nm.rates.detail}'))
        if nm.rate_check_disagrees:
            ri = nm.rate_inference
            issues.append(LoadIssue('warning', '(axis)', 'session', sid,
                                    f'the closure check ({ri.key}, {ri.closure_deg:.1f} deg) disagrees with the '
                                    f'calibration axis_check ({nm.rates.key}); the axis_check is used'))

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


def _session_rate_check(trials: list[dict[str, Any]], calibration: Mapping[str, Any] | None, engine: str | None,
                        max_trials: int) -> RateMapping:
    """``conventions.infer_rate_mapping`` over up to ``max_trials`` complete trials of one session (evenly
    spread over the session, in trial order), with the closures of ``features.rate_closures``."""
    from .features import rate_closures  # features imports io: import here

    done = sorted((t for t in trials if (t.get('outcome') or {}).get('status') == 'complete'),
                  key=lambda t: (t.get('trial_index') or 0, t.get('attempt') or 0))
    if max_trials > 0 and len(done) > max_trials:
        pick = sorted({int(round(x)) for x in np.linspace(0, len(done) - 1, max_trials)})
        done = [done[i] for i in pick]
    closures = []
    for t in done:
        try:
            closures.append(rate_closures(t, calibration=calibration, engine=engine))
        except (KeyError, TypeError, ValueError, IndexError, ArithmeticError, np.linalg.LinAlgError):
            closures.append(None)  # an odd trial cannot vote; the check never costs the session
    return infer_rate_mapping(closures)


def _check_stream_lengths(rec: Mapping[str, Any]) -> list[str]:
    """Raises ValueError when a schema 1.0 stream column is inconsistent (the trial is unusable); returns
    the problems of the optional schema 1.2 columns (ignored by every reader, listed as warnings)."""
    st = rec['streams']
    for name, cols in _STREAM_COLUMNS.items():
        s = st.get(name)
        if not s:
            continue
        n = len(s['t'])
        for c in cols:
            if c in s and s[c] is not None and len(s[c]) != n:
                raise ValueError(f'streams.{name}.{c} has {len(s[c])} values, t has {n}')
    problems = []
    for name, cols in _OPTIONAL_STREAM_COLUMNS.items():
        s = st.get(name)
        if not isinstance(s, Mapping):
            continue
        t = s.get('t')
        if t is None:
            problems.append(f'streams.{name} has no t')
            continue
        n = len(t)
        for c in cols:
            v = s.get(c)
            if v is None:
                continue
            if not isinstance(v, np.ndarray) or len(v) != n:
                problems.append(f'streams.{name}.{c} ignored ({len(v) if hasattr(v, "__len__") else "?"} values, '
                                f't has {n}, or not numeric)')
    return problems


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
    scols = ['session_id', 'platform', 'browser', 'engine', 'model_hint', 'phone_model', 'device_model', 'status',
             'calibration_method', 'review_mode']
    if len(sess_df):
        have = [c for c in scols if c in sess_df.columns]
        tdf = tdf.merge(sess_df[have].rename(columns={'status': 'session_status'}), how='left', on='session_id')
    for c in ('platform', 'browser', 'engine', 'model_hint', 'phone_model', 'device_model', 'session_status',
              'calibration_method', 'review_mode'):
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
    # orphan trials (no session record): the model hint / self-report, as before schema 1.2
    tdf['device_model'] = [d if isinstance(d, str) and d else (m if isinstance(m, str) and m else
                                                                (p if isinstance(p, str) and p else None))
                           for d, m, p in zip(tdf['device_model'], tdf['model_hint'], tdf['phone_model'])]
    # docs/02 section 10 rule 3: two trial ids with the same (session, trial_index, attempt) are both kept
    key = tdf.groupby(['session_id', 'trial_index', 'attempt'])['trial_id'].transform('size')
    tdf['dup_attempt_key'] = key > 1
    for (sid, ti, at), g in tdf[tdf['dup_attempt_key']].groupby(['session_id', 'trial_index', 'attempt']):
        issues.append(LoadIssue('warning', '(dedupe)', 'trial', None,
                                f'session {sid} trial_index {ti} attempt {at}: {len(g)} different trial ids (all kept)'))
    acc = tdf['accepted'].fillna(False).astype(bool)
    n_acc = acc.groupby([tdf['session_id'], tdf['trial_index']]).transform('sum')
    tdf['multi_accepted'] = n_acc > 1
    # block kind, tier and optional (bonus) flag from the session plan (tier / optional: street, schema 1.2)
    blocks: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    for sid, s in sessions.items():
        for b in ((s.get('protocol') or {}).get('plan') or []):
            if isinstance(b, Mapping):
                blocks[(sid, b.get('block_index'))] = b
    keys = list(zip(tdf['session_id'], tdf['block_index']))
    tdf['block_kind'] = [(blocks.get(k) or {}).get('kind') for k in keys]
    tdf['block_tier'] = [(blocks.get(k) or {}).get('tier') for k in keys]
    tdf['block_optional'] = [(None if k not in blocks else blocks[k].get('optional') is True) for k in keys]

    def same(h: Any, d: Any) -> bool | None:
        if d is None or (isinstance(d, float) and math.isnan(d)) or h is None or d not in ('left', 'right'):
            return None
        return h == d

    tdf['hand_is_dominant'] = [same(h, d) for h, d in zip(tdf['hand'], tdf['dominant_hand'])]
    # the hand the participant usually holds the phone in ('both' -> None)
    tdf['hand_is_phone_hand'] = [same(h, d) for h, d in zip(tdf['hand'], tdf['phone_hand'])]
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

