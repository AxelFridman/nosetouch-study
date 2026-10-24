"""Tables, engine and schema management (SQLAlchemy 2 Core; Postgres on Heroku, SQLite locally).

Records are stored exactly as received, as JSON (JSONB on Postgres), next to a few columns derived
from them for the admin page and the export filters. Phone-side times (started_at, updated_at,
created_at, sent_at) are kept as the ISO strings the phone sent; server-side times are timestamps.
The schema is created idempotently at startup; meta.schema_version allows later migrations.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import logging
import re
import threading
from collections.abc import Iterator
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import StaticPool

from .settings import Settings

log = logging.getLogger('nosetouch')

SCHEMA_VERSION = 1
_ADVISORY_LOCK_KEY = 815_407_221  # serialises schema creation when several dynos/workers start at once

metadata = MetaData()
JSONType = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), 'postgresql')
UTC = DateTime(timezone=True)

meta = Table(
    'meta', metadata,
    Column('key', String(64), primary_key=True),
    Column('value', Text, nullable=False),
)

participants = Table(
    'participants', metadata,
    Column('participant_id', String(16), primary_key=True),
    Column('study_code', Text),
    # newest ParticipantProfile seen in a session snapshot (redacted by the app); NULL until a session arrives
    Column('record', JSONType),
    Column('profile_updated_at', Text),  # the profile's own updated_at (phone clock)
    Column('profile_sha256', String(64)),
    Column('first_seen', UTC, nullable=False),
    Column('last_seen', UTC, nullable=False),  # last accepted upload of any record of this participant
)

sessions = Table(
    'sessions', metadata,
    Column('session_id', String(64), primary_key=True),
    Column('participant_id', String(16), nullable=False, index=True),
    Column('study_code', Text, nullable=False),
    Column('status', Text),
    Column('started_at', Text),
    Column('updated_at', Text),
    Column('ended_at', Text),
    Column('simulated', Boolean),
    Column('app_version', Text),
    Column('schema_version', Text),
    Column('platform', Text),
    Column('browser', Text),
    Column('preset', Text),
    Column('n_planned', BigInteger),
    Column('n_attempts', BigInteger),
    Column('n_accepted', BigInteger),
    Column('record', JSONType, nullable=False),
    Column('sent_at', Text),  # envelope sent_at of the stored version
    Column('body_sha256', String(64), nullable=False),
    Column('size_bytes', Integer, nullable=False),
    Column('first_received_at', UTC, nullable=False),
    Column('last_received_at', UTC, nullable=False, index=True),  # when the stored version arrived
)

trials = Table(
    'trials', metadata,
    Column('trial_id', String(64), primary_key=True),
    Column('session_id', String(64), nullable=False, index=True),
    Column('participant_id', String(16), nullable=False, index=True),
    Column('study_code', Text, nullable=False),
    Column('trial_index', BigInteger, nullable=False),
    Column('attempt', BigInteger, nullable=False),
    Column('hand', Text),
    Column('orientation', Text),
    Column('speed', Text),
    Column('variant', Text),
    Column('status', Text),
    Column('accepted', Boolean),
    Column('review_interrupted', Boolean, nullable=False, default=False),
    Column('simulated', Boolean),
    Column('created_at', Text),
    Column('app_version', Text),
    Column('schema_version', Text),
    Column('record', JSONType, nullable=False),
    Column('sent_at', Text),
    Column('body_sha256', String(64), nullable=False),
    Column('size_bytes', Integer, nullable=False),
    Column('first_received_at', UTC, nullable=False),
    Column('received_at', UTC, nullable=False, index=True),  # when the stored version arrived
)

# Metadata of every upload: never the body, never a profile value or a name.
ingest_log = Table(
    'ingest_log', metadata,
    Column('id', BigInteger().with_variant(Integer, 'sqlite'), primary_key=True, autoincrement=True),
    Column('at', UTC, nullable=False, index=True),
    Column('outcome', String(16), nullable=False),  # stored|replaced|unchanged|kept_newer|rejected
    Column('code', String(64)),  # rejection code, or why a re-send was kept/ignored
    Column('kind', Text),
    Column('record_type', Text),
    Column('study_code', Text),
    Column('participant_id', Text),
    Column('session_id', Text),
    Column('record_id', Text),
    Column('schema_version', Text),
    Column('app_version', Text),
    Column('bytes', BigInteger),
    Column('sha256', String(64)),
)
Index('ix_ingest_log_participant_id', ingest_log.c.participant_id)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def utc(t: _dt.datetime | None) -> _dt.datetime | None:
    """Aware UTC datetime (SQLite returns naive values that are UTC)."""
    if t is None:
        return None
    if t.tzinfo is None:
        return t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone(_dt.timezone.utc)


def make_engine(settings: Settings) -> Engine:
    url = settings.database_url
    # hide_parameters: an error message must never carry a record (logs, answers)
    common: dict[str, Any] = {'json_serializer': _dumps, 'hide_parameters': True}
    if url.startswith('sqlite'):
        memory = url.rstrip('/') in ('sqlite+pysqlite:', 'sqlite:') or ':memory:' in url
        if memory:
            # one connection shared by every thread (guarded by Database's lock)
            return create_engine(url, poolclass=StaticPool, connect_args={'check_same_thread': False}, **common)
        engine = create_engine(url, connect_args={'check_same_thread': False, 'timeout': 30}, **common)

        @event.listens_for(engine, 'connect')
        def _wal(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute('PRAGMA journal_mode=WAL')  # readers (exports) do not block the writer
            cur.close()

        return engine
    # Heroku Essential plans allow 20 connections: 3 + 2 per worker process leaves room for psql and
    # several WEB_CONCURRENCY workers
    return create_engine(
        url,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_timeout=15,
        connect_args={'connect_timeout': 10, 'application_name': 'nosetouch'},
        **common,
    )


class Database:
    """The engine plus the locking that SQLite needs (Postgres uses row locks instead)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = make_engine(settings)
        self.dialect = self.engine.dialect.name
        self.is_postgres = self.dialect == 'postgresql'
        self.is_sqlite = self.dialect == 'sqlite'
        self.is_memory = self.is_sqlite and isinstance(self.engine.pool, StaticPool)
        # SQLite has one writer at a time; the in-memory database has one shared connection
        self._write_lock = threading.RLock() if self.is_sqlite else None
        self._last_prune = 0.0
        self._prune_lock = threading.Lock()
        self._reject_day = ''
        self._reject_count = 0
        self._reject_lock = threading.Lock()
        self.ready = False

    # ── connections ──

    @contextlib.contextmanager
    def write(self) -> Iterator[Connection]:
        """A transaction for writes (committed on success)."""
        with self._locked():
            with self.engine.begin() as conn:
                yield conn

    @contextlib.contextmanager
    def read(self) -> Iterator[Connection]:
        """A connection for reads. The in-memory database serialises them with the writes."""
        lock = self._write_lock if self.is_memory else None
        with lock if lock is not None else contextlib.nullcontext():
            with self.engine.connect() as conn:
                yield conn

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        if self._write_lock is None:
            yield
            return
        with self._write_lock:
            yield

    def insert_ignore(self, conn: Connection, table: Table, values: dict[str, Any]) -> bool:
        """INSERT ... ON CONFLICT DO NOTHING RETURNING <key>; True when the row was inserted.
        (RETURNING, not rowcount: SQLAlchemy reports rowcount -1 for this INSERT on psycopg.)"""
        dialect = postgresql if self.is_postgres else sqlite
        key = list(table.primary_key.columns)[0]
        stmt = dialect.insert(table).values(**values).on_conflict_do_nothing().returning(key)
        return conn.execute(stmt).first() is not None

    # ── schema ──

    def init_schema(self) -> None:
        """Creates missing tables and indexes (idempotent) and records the schema version."""
        with self.write() as conn:
            if self.is_postgres:
                conn.execute(text('SELECT pg_advisory_xact_lock(:k)'), {'k': _ADVISORY_LOCK_KEY})
            metadata.create_all(conn, checkfirst=True)
            row = conn.execute(select(meta.c.value).where(meta.c.key == 'schema_version')).first()
            if row is None:
                conn.execute(insert(meta).values(key='schema_version', value=str(SCHEMA_VERSION)))
            elif int(row.value) > SCHEMA_VERSION:
                log.warning(_kv(event='schema_newer_than_code', db_version=row.value, code_version=SCHEMA_VERSION))
            # (future migrations: if int(row.value) < SCHEMA_VERSION, run them here, then update the row)
        self.ready = True

    def schema_version(self, conn: Connection) -> int | None:
        row = conn.execute(select(meta.c.value).where(meta.c.key == 'schema_version')).first()
        return int(row.value) if row is not None else None

    def drop_all(self) -> None:
        """Tests only: removes every table of this service."""
        with self.write() as conn:
            metadata.drop_all(conn, checkfirst=True)

    # ── ingest log ──

    def log_entry(self, conn: Connection, *, at: _dt.datetime, outcome: str, code: str | None,
                  ids: dict[str, str | None], size: int | None, sha256: str | None) -> None:
        conn.execute(insert(ingest_log).values(at=at, outcome=outcome, code=code, bytes=size, sha256=sha256, **ids))

    def log_rejection(self, *, at: _dt.datetime, code: str, ids: dict[str, str | None], size: int | None,
                      sha256: str | None) -> bool:
        """One ingest_log row for a refused upload, at most reject_log_daily_cap per UTC day (counted in
        this process, like receiver.py). Returns whether a row was written; database errors propagate."""
        day = at.strftime('%Y-%m-%d')
        with self._reject_lock:
            if day != self._reject_day:
                self._reject_day, self._reject_count = day, 0
            if self._reject_count >= self.settings.reject_log_daily_cap:
                return False
            self._reject_count += 1
        with self.write() as conn:
            self.log_entry(conn, at=at, outcome='rejected', code=code, ids=ids, size=size, sha256=sha256)
        return True

    def maybe_prune(self, now: _dt.datetime, every_s: float = 600.0) -> None:
        """Prunes ingest_log at most every `every_s` seconds (errors are logged, never raised)."""
        stamp = now.timestamp()
        if stamp - self._last_prune < every_s or not self._prune_lock.acquire(blocking=False):
            return
        try:
            self._last_prune = stamp
            self.prune_log(now)
        except Exception as e:  # noqa: BLE001 - pruning must never fail an upload
            log.warning(_kv(event='prune_failed', error=type(e).__name__))
        finally:
            self._prune_lock.release()

    def prune_log(self, now: _dt.datetime) -> int:
        """Keeps ingest_log within log_retention_days and log_max_rows. Returns the rows deleted."""
        cutoff = now - _dt.timedelta(days=self.settings.log_retention_days)
        with self.write() as conn:
            n = conn.execute(delete(ingest_log).where(ingest_log.c.at < cutoff)).rowcount or 0
            keep_from = (
                select(ingest_log.c.id).order_by(ingest_log.c.id.desc())
                .offset(self.settings.log_max_rows - 1).limit(1).scalar_subquery()
            )
            n += conn.execute(delete(ingest_log).where(ingest_log.c.id < keep_from)).rowcount or 0
        return n

    def counts(self, conn: Connection) -> dict[str, int]:
        return {
            'participants': conn.execute(select(func.count()).select_from(participants)).scalar_one(),
            'sessions': conn.execute(select(func.count()).select_from(sessions)).scalar_one(),
            'trials': conn.execute(select(func.count()).select_from(trials)).scalar_one(),
        }

    def size_bytes(self, conn: Connection) -> int | None:
        """Database size on disk (Postgres), or None."""
        if not self.is_postgres:
            return None
        try:
            return int(conn.execute(text('SELECT pg_database_size(current_database())')).scalar_one())
        except Exception:  # noqa: BLE001 - informative only
            return None


# Heroku's router id of the current request (X-Request-Id), added to every log line of that request
REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar('nosetouch_request_id', default=None)
_REQUEST_ID_RE = re.compile(r'[A-Za-z0-9+/=_-]{1,200}')


def set_request_id(value: str | None) -> contextvars.Token[str | None]:
    return REQUEST_ID.set(value if value and _REQUEST_ID_RE.fullmatch(value) else None)


def _kv(**fields: Any) -> str:
    """One structured log line (JSON). Callers pass identifiers and sizes only, never bodies or names."""
    rid = REQUEST_ID.get()
    if rid:
        fields.setdefault('request_id', rid)
    return json.dumps(fields, ensure_ascii=True, separators=(',', ':'), default=str)


kv = _kv
