"""Exports, streamed so that memory stays flat however large the study grows.

/api/export      the app's own ExportBundle ({format: 'nose-touch-export', schema_version, app_version,
                 exported_at, participants, sessions, trials}; app/src/core/storage.ts exportBundle)
/api/export.zip  the folder layout of server/receiver.py and the Apps Script backend:
                 <study_code>/<participant_id>/<session_id>/session.json
                 <study_code>/<participant_id>/<session_id>/trial_<index:03d>_a<attempt>_<trial_id>.json
Both read the stored JSON as text (no parse and re-dump per record) and accept the same filters:
participant_id, since (records whose stored version arrived at or after this time, plus the sessions
of those trials, so every export is self-contained) and include_simulated (default true).
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import io
import re
import zipfile
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Text, cast, or_, select, union
from sqlalchemy.engine import Connection

from . import SERVER_VERSION, SERVICE_NAME
from .db import Database, participants, sessions, trials, utc
from .validation import ID_RE, PARTICIPANT_RE, is_participant_id, iso, trial_file_name

CHUNK = 256 * 1024
DEFAULT_SCHEMA_VERSION = '1.1.0'
_SAFE_COMPONENT = re.compile(r'[^A-Za-z0-9_-]')


@dataclass(frozen=True)
class ExportFilter:
    participant_id: str | None = None
    since: _dt.datetime | None = None
    include_simulated: bool = True


def parse_filter(params: Mapping[str, str]) -> ExportFilter:
    """Query parameters -> ExportFilter; ValueError with a readable message on bad input."""
    pid = (params.get('participant_id') or '').strip() or None
    if pid is not None and not is_participant_id(pid):
        raise ValueError('participant_id must match /^P-[A-HJ-NP-Z2-9]{6}$/')
    since_raw = (params.get('since') or '').strip()
    since = None
    if since_raw:
        candidates = [since_raw.replace('Z', '+00:00').replace('z', '+00:00')]
        head, sep, tail = candidates[0].rpartition(' ')
        if sep and re.fullmatch(r'\d{2}:?\d{2}', tail):  # '+02:00' arrives as ' 02:00' when not URL-encoded
            candidates.append(f'{head}+{tail}')
        for candidate in candidates:
            try:
                since = utc(_dt.datetime.fromisoformat(candidate))
                break
            except (ValueError, OverflowError):  # OverflowError: 0001-01-01T00:00+05:00 has no UTC time
                continue
        if since is None:
            raise ValueError('since must be an ISO time such as 2026-11-20T00:00:00Z')
    sim = (params.get('include_simulated') or '').strip().lower()
    if sim in ('', '1', 'true', 'yes', 'on'):
        include = True
    elif sim in ('0', 'false', 'no', 'off'):
        include = False
    else:
        raise ValueError('include_simulated must be true or false')
    return ExportFilter(pid, since, include)


def file_stem(f: ExportFilter, now: _dt.datetime) -> str:
    """nosetouch_<participant or all>_<YYYYMMDD-HHMM> (UTC), like the app's export file names."""
    return f"nosetouch_{f.participant_id or 'all'}_{now.strftime('%Y%m%d-%H%M')}"


# ───────────────────────────── selection ─────────────────────────────


def _not_simulated(col: Any) -> Any:
    return or_(col.is_(None), col.is_(False))


def _trial_conditions(f: ExportFilter) -> list[Any]:
    conds: list[Any] = []
    if f.participant_id:
        conds.append(trials.c.participant_id == f.participant_id)
    if not f.include_simulated:
        conds.append(_not_simulated(trials.c.simulated))
    if f.since is not None:
        conds.append(trials.c.received_at >= f.since)
    return conds


def _session_conditions(f: ExportFilter) -> list[Any]:
    conds: list[Any] = []
    if f.participant_id:
        conds.append(sessions.c.participant_id == f.participant_id)
    if not f.include_simulated:
        conds.append(_not_simulated(sessions.c.simulated))
    if f.since is not None:
        owning = select(trials.c.session_id).where(*_trial_conditions(f))
        conds.append(or_(sessions.c.last_received_at >= f.since, sessions.c.session_id.in_(owning)))
    return conds


def _participant_ids(f: ExportFilter) -> Any:
    return union(
        select(sessions.c.participant_id).where(*_session_conditions(f)),
        select(trials.c.participant_id).where(*_trial_conditions(f)),
    ).subquery()


def _version_key(v: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(v).split('.'))
    except ValueError:
        return ()


@contextlib.contextmanager
def _snapshot(db: Database) -> Iterator[Connection]:
    """A connection that reads one consistent snapshot (Postgres) with server-side cursors."""
    if db.is_postgres:
        with db.engine.connect() as conn:
            yield conn.execution_options(isolation_level='REPEATABLE READ', stream_results=True, yield_per=20)
    else:
        with db.read() as conn:
            yield conn


def _participant_texts(conn: Connection, f: ExportFilter) -> Iterator[str]:
    ids = _participant_ids(f)
    q = (select(cast(participants.c.record, Text))
         .where(participants.c.record.is_not(None), participants.c.participant_id.in_(select(ids.c.participant_id)))
         .order_by(participants.c.participant_id))
    for (text,) in conn.execute(q):
        yield text


def _session_rows(conn: Connection, f: ExportFilter, with_text: bool = True) -> Iterator[Any]:
    cols = [sessions.c.session_id, sessions.c.participant_id, sessions.c.study_code, sessions.c.schema_version,
            sessions.c.last_received_at]
    if with_text:
        cols.append(cast(sessions.c.record, Text).label('text'))
    q = select(*cols).where(*_session_conditions(f)).order_by(sessions.c.started_at, sessions.c.session_id)
    yield from conn.execute(q)


def _trial_rows(conn: Connection, f: ExportFilter) -> Iterator[Any]:
    q = (
        select(trials.c.trial_id, trials.c.session_id, trials.c.participant_id, trials.c.study_code,
               trials.c.trial_index, trials.c.attempt, trials.c.schema_version, trials.c.received_at,
               cast(trials.c.record, Text).label('text'))
        .select_from(trials.outerjoin(sessions, sessions.c.session_id == trials.c.session_id))
        .where(*_trial_conditions(f))
        .order_by(sessions.c.started_at.is_(None), sessions.c.started_at, trials.c.session_id,
                  trials.c.trial_index, trials.c.attempt, trials.c.created_at, trials.c.trial_id)
    )
    yield from conn.execute(q)


def _schema_version(conn: Connection, f: ExportFilter) -> str:
    versions = set(conn.execute(select(sessions.c.schema_version).where(*_session_conditions(f)).distinct()).scalars())
    versions |= set(conn.execute(select(trials.c.schema_version).where(*_trial_conditions(f)).distinct()).scalars())
    versions = {v for v in versions if isinstance(v, str) and _version_key(v)}
    return max(versions, key=_version_key) if versions else DEFAULT_SCHEMA_VERSION


def _materialized(db: Database, gen: Iterator[bytes]) -> Iterator[bytes]:
    """The in-memory SQLite database has one shared connection: read everything under its lock at the
    first step (a generator: nothing runs before the response starts iterating it in a worker thread)."""
    if db.is_memory:
        chunks = list(gen)
        yield from chunks
    else:
        yield from gen


# ───────────────────────────── JSON bundle ─────────────────────────────


def iter_bundle(db: Database, f: ExportFilter, now: _dt.datetime) -> Iterator[bytes]:
    return _materialized(db, _bundle_chunks(db, f, now))


def _bundle_chunks(db: Database, f: ExportFilter, now: _dt.datetime) -> Iterator[bytes]:
    buf = io.StringIO()

    def flush() -> bytes:
        data = buf.getvalue().encode('utf-8')
        buf.seek(0)
        buf.truncate()
        return data

    with _snapshot(db) as conn:
        version = _schema_version(conn, f)
        buf.write('{"format":"nose-touch-export","schema_version":"%s","app_version":"%s %s","exported_at":"%s",'
                  % (version, SERVICE_NAME, SERVER_VERSION, iso(now)))
        for key, texts in (
            ('participants', _participant_texts(conn, f)),
            ('sessions', (r.text for r in _session_rows(conn, f))),
            ('trials', (r.text for r in _trial_rows(conn, f))),
        ):
            buf.write(f'"{key}":[' if key == 'participants' else f'],"{key}":[')
            for i, text in enumerate(texts):
                if i:
                    buf.write(',')
                buf.write(text)
                if buf.tell() >= CHUNK:
                    yield flush()
        buf.write(']}')
    yield flush()


def gzip_stream(chunks: Iterator[bytes], level: int = 6) -> Iterator[bytes]:
    """gzip (Content-Encoding: gzip) of a byte stream."""
    z = zlib.compressobj(level, zlib.DEFLATED, 31)
    for chunk in chunks:
        out = z.compress(chunk)
        if out:
            yield out
    yield z.flush()


# ───────────────────────────── folder layout as a zip ─────────────────────────────


class _Sink(io.RawIOBase):
    """Write-only, unseekable buffer: zipfile then writes data descriptors and never seeks back."""

    def __init__(self) -> None:
        super().__init__()
        self.buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        self.buf += b
        return len(b)

    def take(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


def _component(value: Any, pattern: re.Pattern[str]) -> str:
    s = str(value)
    return s if pattern.fullmatch(s) else (_SAFE_COMPONENT.sub('_', s)[:64] or '_')


def _zip_time(at: _dt.datetime | None) -> tuple[int, int, int, int, int, int]:
    t = utc(at) or _dt.datetime(1980, 1, 1, tzinfo=_dt.timezone.utc)
    if t.year < 1980:
        return (1980, 1, 1, 0, 0, 0)
    if t.year > 2107:
        return (2107, 12, 31, 23, 59, 58)
    return (t.year, t.month, t.day, t.hour, t.minute, t.second)


def _entry(name: str, at: _dt.datetime | None) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_zip_time(at))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def zip_path(study_code: Any, participant_id: Any, session_id: Any, name: str) -> str:
    return '/'.join((_component(study_code, ID_RE), _component(participant_id, PARTICIPANT_RE),
                     _component(session_id, ID_RE), name))


def iter_zip(db: Database, f: ExportFilter) -> Iterator[bytes]:
    return _materialized(db, _zip_chunks(db, f))


def _zip_chunks(db: Database, f: ExportFilter) -> Iterator[bytes]:
    sink = _Sink()
    with _snapshot(db) as conn, zipfile.ZipFile(sink, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for r in _session_rows(conn, f):
            zf.writestr(_entry(zip_path(r.study_code, r.participant_id, r.session_id, 'session.json'),
                               r.last_received_at), r.text.encode('utf-8'))
            if len(sink.buf) >= CHUNK:
                yield sink.take()
        for r in _trial_rows(conn, f):
            name = trial_file_name(int(r.trial_index), int(r.attempt), _component(r.trial_id, ID_RE))
            zf.writestr(_entry(zip_path(r.study_code, r.participant_id, r.session_id, name), r.received_at),
                        r.text.encode('utf-8'))
            if len(sink.buf) >= CHUNK:
                yield sink.take()
    yield sink.take()
