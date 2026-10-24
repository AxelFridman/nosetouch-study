"""The upload contract, ported from server/receiver.py (same rules, codes and messages).

Standard library only: import_export.py uses it for --dry-run without FastAPI installed.

One POST body is one UploadEnvelope {kind, study_key, sent_at, record} (schema/records.schema.json).
validate_envelope() accepts exactly what receiver.py and the Apps Script backend accept:
    kind 'session' | 'trial' matching record.record_type; the study key when one is configured;
    schema_version '1.x'; study_code, session_id and trial_id matching ^[A-Za-z0-9_-]{1,64}$ (and not
    'quarantine' or a Windows device name such as CON, because export.zip folders are unpacked on
    Windows); participant_id matching ^P-[A-HJ-NP-Z2-9]{6}$; trial_index >= 0 and attempt >= 1 as
    JavaScript safe integers.
One check is specific to this service (canonical_record): Postgres cannot store a NUL character or an
unpaired UTF-16 surrogate inside JSON text, so such a record is refused ('unsupported_text') instead of
failing on every retry. The app never produces either.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import math
import re
from typing import Any

RESERVED_STUDY_CODE = 'quarantine'  # older backends kept rejected payloads in <root>/quarantine/
IDENT_MAX_CHARS = 100
MAX_SAFE_INTEGER = 2**53 - 1  # JavaScript's limit: the app's numbers never exceed it
# outcome.note of the copies the app saves of one attempt (app/src/app/sessionFlow.ts), in save order:
REVIEW_INTERRUPTED_NOTE = 'review_interrupted'  # tap review: the copy saved before the review opens
HAND_NOT_REPORTED_NOTE = 'hand_not_reported'  # v2 choice trial: the copy saved before the hand question
REDO_REQUESTED_NOTE = 'redo_requested'  # v2 pause sheet: a decided attempt revoked afterwards (final)
#: placeholder copies, saved before the participant answered: any decided copy replaces them
PLACEHOLDER_NOTES = frozenset({REVIEW_INTERRUPTED_NOTE, HAND_NOT_REPORTED_NOTE})

# fullmatch (not match + '$'): in Python '$' also matches before a trailing newline
ID_RE = re.compile(r'[A-Za-z0-9_-]{1,64}')
PARTICIPANT_RE = re.compile(r'P-[A-HJ-NP-Z2-9]{6}')
ID_TEXT = '/^[A-Za-z0-9_-]{1,64}$/'  # messages quote the patterns exactly as Code.gs does
PARTICIPANT_TEXT = '/^P-[A-HJ-NP-Z2-9]{6}$/'
WINDOWS_RESERVED_RE = re.compile(r'(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])', re.IGNORECASE)
# A JSON \u0000 escape: preceded by an even number of backslashes (an odd number escapes the backslash).
_NUL_ESCAPE_RE = re.compile(r'(?<!\\)(?:\\\\)*\\u0000')


class Rejection(Exception):
    """An upload refused for good: answered {ok: false, retry: false} with `status`."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def iso(t: _dt.datetime | None) -> str | None:
    """UTC time in JavaScript's toISOString() format, e.g. 2026-10-23T09:15:02.123Z (naive = UTC)."""
    if t is None:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone(_dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def time_key(value: Any) -> float:
    """Epoch seconds of an ISO time string; anything unreadable sorts as the oldest (-inf)."""
    if not isinstance(value, str) or not value:
        return -math.inf
    try:
        t = _dt.datetime.fromisoformat(value.strip().replace('Z', '+00:00').replace('z', '+00:00'))
    except ValueError:
        return -math.inf
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    try:
        return t.timestamp()
    except (OverflowError, OSError, ValueError):
        return -math.inf


# ───────────────────────────── parsing ─────────────────────────────


def _reject_constant(name: str) -> Any:
    raise ValueError(f'{name} is not valid JSON')  # NaN / Infinity: JSON.parse refuses them too


def _finite_float(text: str) -> float:
    value = float(text)
    if value in (float('inf'), float('-inf')):
        raise ValueError(f'number out of range: {text}')
    return value


def parse_envelope(body: bytes) -> Any:
    """Strict JSON (UTF-8, no BOM, no NaN/Infinity, finite numbers), like JSON.parse in Apps Script."""
    try:
        text = body.decode('utf-8')
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Rejection('malformed_json', 'malformed JSON') from e


# ───────────────────────────── validation ─────────────────────────────


def is_id(v: Any) -> bool:
    return isinstance(v, str) and ID_RE.fullmatch(v) is not None


def is_participant_id(v: Any) -> bool:
    return isinstance(v, str) and PARTICIPANT_RE.fullmatch(v) is not None


def as_int(v: Any) -> int | None:
    """A whole number as JavaScript sees one (Number.isSafeInteger): 7 and 7.0 yes; True, 7.5, '7' no."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if abs(v) <= MAX_SAFE_INTEGER else None
    if isinstance(v, float) and v.is_integer() and abs(v) <= MAX_SAFE_INTEGER:
        return int(v)
    return None


def _same_key(sent: str, expected: str) -> bool:
    # surrogatepass: a lone surrogate in the sent key must be a mismatch, not a server error
    return hmac.compare_digest(sent.encode('utf-8', 'surrogatepass'), expected.encode('utf-8', 'surrogatepass'))


def validate_envelope(env: Any, study_key: str | None) -> None:
    """Raises Rejection unless `env` is an acceptable UploadEnvelope (receiver.py's rules and codes)."""
    if not isinstance(env, dict):
        raise Rejection('bad_envelope', 'the body must be a JSON object (an UploadEnvelope)')
    kind = env.get('kind')
    if kind not in ('session', 'trial'):
        raise Rejection('bad_kind', "kind must be 'session' or 'trial'")
    if study_key:
        sent = env.get('study_key')
        if not isinstance(sent, str) or not _same_key(sent, study_key):
            raise Rejection('bad_study_key', 'study key does not match')
    r = env.get('record')
    if not isinstance(r, dict):
        raise Rejection('bad_record', 'record must be an object')
    if r.get('record_type') != kind:
        raise Rejection('bad_record_type', 'record.record_type does not match kind')
    sv = r.get('schema_version')
    if not isinstance(sv, str) or not sv.startswith('1.'):
        raise Rejection('bad_schema_version', 'unsupported schema_version (this receiver accepts 1.x)')
    if not is_id(r.get('study_code')):
        raise Rejection('bad_study_code', f'study_code must match {ID_TEXT}')
    if r['study_code'].lower() == RESERVED_STUDY_CODE or WINDOWS_RESERVED_RE.fullmatch(r['study_code']):
        raise Rejection('bad_study_code', 'study_code is reserved')
    if not is_participant_id(r.get('participant_id')):
        raise Rejection('bad_participant_id', f'participant_id must match {PARTICIPANT_TEXT}')
    if not is_id(r.get('session_id')):
        raise Rejection('bad_session_id', f'session_id must match {ID_TEXT}')
    if WINDOWS_RESERVED_RE.fullmatch(r['session_id']):
        raise Rejection('bad_session_id', 'session_id is reserved')
    if kind == 'trial':
        if not is_id(r.get('trial_id')):
            raise Rejection('bad_trial_id', f'trial_id must match {ID_TEXT}')
        index = as_int(r.get('trial_index'))
        if index is None or index < 0:
            raise Rejection('bad_trial_index', 'trial_index must be an integer >= 0')
        attempt = as_int(r.get('attempt'))
        if attempt is None or attempt < 1:
            raise Rejection('bad_attempt', 'attempt must be an integer >= 1')


def canonical_record(record: dict[str, Any]) -> tuple[str, int]:
    """(sha256 hex of the canonical JSON, its size in bytes) of a validated record.

    Canonical = sorted keys, compact separators, UTF-8: a re-sent copy hashes the same whatever the key
    order. Raises Rejection('unsupported_text') for text Postgres cannot store (see the module doc)."""
    text = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    try:
        data = text.encode('utf-8')
    except UnicodeEncodeError as e:  # an unpaired surrogate, e.g. from a "\ud800" escape
        raise Rejection('unsupported_text', 'record contains text the database cannot store '
                        '(an unpaired surrogate)') from e
    if _NUL_ESCAPE_RE.search(text):
        raise Rejection('unsupported_text', 'record contains text the database cannot store (a NUL character)')
    return hashlib.sha256(data).hexdigest(), len(data)


# ───────────────────────────── layout ─────────────────────────────


def trial_file_name(trial_index: int, attempt: int, trial_id: str) -> str:
    return f'trial_{trial_index:03d}_a{attempt}_{trial_id}.json'


def record_path(env: dict[str, Any]) -> tuple[str, str]:
    """(folder, file name) of a validated envelope's record in the backend layout (docs/02 section 7)."""
    r = env['record']
    folder = '/'.join((r['study_code'], r['participant_id'], r['session_id']))
    if env['kind'] == 'session':
        return folder, 'session.json'
    index = as_int(r['trial_index'])
    attempt = as_int(r['attempt'])
    assert index is not None and attempt is not None  # validated
    return folder, trial_file_name(index, attempt, r['trial_id'])


# ───────────────────────────── log identifiers ─────────────────────────────


def ident(v: Any) -> str | None:
    """An identifier for the ingest log: a string or a number, cut to 100 characters; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        s = v
    elif isinstance(v, int) or (isinstance(v, float) and math.isfinite(v)):
        s = str(v)
    else:
        return None
    s = s.encode('utf-8', 'replace').decode('utf-8')  # lone surrogates cannot be stored
    s = s.replace('\x00', '\N{REPLACEMENT CHARACTER}')
    return s[:IDENT_MAX_CHARS] + '…' if len(s) > IDENT_MAX_CHARS else s


def envelope_identifiers(env: Any) -> dict[str, str | None]:
    """The identifiers of an envelope (best effort, None when the body was not parsed): never a value
    from the profile, the streams or a free-text field."""
    e = env if isinstance(env, dict) else {}
    r = e.get('record') if isinstance(e.get('record'), dict) else {}
    return {
        'kind': ident(e.get('kind')),
        'record_type': ident(r.get('record_type')),
        'study_code': ident(r.get('study_code')),
        'participant_id': ident(r.get('participant_id')),
        'session_id': ident(r.get('session_id')),
        'record_id': ident(r.get('trial_id')) or ident(r.get('session_id')),
        'schema_version': ident(r.get('schema_version')),
        'app_version': ident(r.get('app_version')),
    }
