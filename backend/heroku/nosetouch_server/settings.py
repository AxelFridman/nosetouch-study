"""Configuration from environment variables (README.md lists them)."""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.engine import make_url

log = logging.getLogger('nosetouch')

DEFAULT_ALLOWED_ORIGINS = (
    'https://axelfridman.github.io',
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:4173',
)
# Largest accepted request. The real trials measured so far are 37-165 KB (about 7 KB per second of
# recording); the wait for stillness before GO has no time limit, and a 413 is a permanent refusal on
# the phone, so the cap leaves room for minutes-long recordings (receiver.py and Apps Script: 2 MB).
DEFAULT_MAX_BYTES = 4_000_000
LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1', ''})
_HEROKU_PG_VAR = re.compile(r'HEROKU_POSTGRESQL_[A-Z0-9_]+_URL')
_TRUE = frozenset({'1', 'true', 'yes', 'on'})
_FALSE = frozenset({'0', 'false', 'no', 'off'})


@dataclass(frozen=True)
class Settings:
    database_url: str = 'sqlite+pysqlite://'
    study_key: str | None = None
    admin_token: str | None = None
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    max_bytes: int = DEFAULT_MAX_BYTES
    log_retention_days: int = 30
    log_max_rows: int = 20_000
    reject_log_daily_cap: int = 500
    force_https: bool = False
    admin_cookie_max_age: int = 7 * 24 * 3600
    pool_size: int = 3
    max_overflow: int = 2
    # True on Heroku (DYNO set) without a Postgres DATABASE_URL, e.g. deployed before the add-on was
    # attached. A dyno's memory and disk are wiped at every restart (at least daily), so the service then
    # stores nothing and answers every upload 503 retry: the phones keep the records and send them again
    # once the database is there, instead of marking them sent after a 200 and losing them.
    # ALLOW_EPHEMERAL_DB=1 accepts the throwaway database on purpose (a test app).
    database_missing: bool = False
    # free-form notes for the startup log line (never secrets)
    notes: tuple[str, ...] = field(default=(), compare=False)

    @property
    def any_origin(self) -> bool:
        return '*' in self.allowed_origins

    def origin_allowed(self, origin: str | None) -> bool:
        if origin is None:
            return False
        return self.any_origin or origin.strip().lower().rstrip('/') in self.allowed_origins

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        notes: list[str] = []
        force = _flag(env.get('FORCE_HTTPS'))
        database_url = normalize_database_url(env.get('DATABASE_URL'), env.get('SQLITE_PATH'))
        database_missing = (bool(env.get('DYNO')) and not database_url.startswith('postgresql')
                            and _flag(env.get('ALLOW_EPHEMERAL_DB')) is not True)
        if database_missing:
            spare = sorted(k for k in env if _HEROKU_PG_VAR.fullmatch(k))  # names only, never the values
            if spare:
                notes.append(f'{", ".join(spare)} is set but DATABASE_URL is not: make it the main database '
                             f'with heroku pg:promote {spare[0][:-4]}')
        return cls(
            database_url=database_url,
            study_key=(env.get('STUDY_KEY') or '').strip() or None,
            admin_token=(env.get('ADMIN_TOKEN') or '').strip() or None,
            allowed_origins=parse_origins(env.get('ALLOWED_ORIGINS')),
            max_bytes=_positive_int(env, 'MAX_BYTES', DEFAULT_MAX_BYTES, notes),
            log_retention_days=_positive_int(env, 'INGEST_LOG_DAYS', 30, notes),
            log_max_rows=_positive_int(env, 'INGEST_LOG_MAX_ROWS', 20_000, notes),
            reject_log_daily_cap=_positive_int(env, 'REJECT_LOG_DAILY_CAP', 500, notes),
            # Heroku sets DYNO on every dyno: plain-http requests are then redirected / refused
            force_https=force if force is not None else bool(env.get('DYNO')),
            admin_cookie_max_age=_positive_int(env, 'ADMIN_COOKIE_MAX_AGE', 7 * 24 * 3600, notes),
            pool_size=_positive_int(env, 'DB_POOL_SIZE', 3, notes),
            max_overflow=_positive_int(env, 'DB_MAX_OVERFLOW', 2, notes, allow_zero=True),
            database_missing=database_missing,
            notes=tuple(notes),
        )


def _flag(value: str | None) -> bool | None:
    v = (value or '').strip().lower()
    return True if v in _TRUE else False if v in _FALSE else None


def _positive_int(env: Mapping[str, str], name: str, default: int, notes: list[str], allow_zero: bool = False) -> int:
    raw = (env.get(name) or '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value > 0 or (allow_zero and value == 0):
        return value
    notes.append(f'{name}={raw!r} is not a valid number; using {default}')
    return default


def parse_origins(raw: str | None) -> tuple[str, ...]:
    """ALLOWED_ORIGINS: comma-separated origins (scheme://host[:port], no path), or '*' for any."""
    if raw is None or not raw.strip():
        return DEFAULT_ALLOWED_ORIGINS
    out: list[str] = []
    for part in raw.split(','):
        o = part.strip().lower().rstrip('/')
        if o and o not in out:
            out.append(o)
    return tuple(out) or DEFAULT_ALLOWED_ORIGINS


def normalize_database_url(raw: str | None, sqlite_path: str | None = None) -> str:
    """SQLAlchemy URL for DATABASE_URL.

    Heroku gives postgres://user:pw@host:5432/db: the scheme becomes postgresql+psycopg (psycopg 3)
    and sslmode=require is added for any non-local host that does not set sslmode itself. Without
    DATABASE_URL: SQLite in the file SQLITE_PATH, or in memory (data lost on restart; on a Heroku dyno
    the service then refuses to store anything, see Settings.database_missing)."""
    raw = (raw or '').strip()
    if not raw:
        path = (sqlite_path or '').strip()
        if not path:
            return 'sqlite+pysqlite://'
        return 'sqlite+pysqlite:///' + Path(path).expanduser().resolve().as_posix()
    scheme, sep, rest = raw.partition('://')
    if not sep:
        raise ValueError('DATABASE_URL must look like postgres://user:password@host:5432/dbname')
    base = scheme.split('+', 1)[0].lower()
    if base in ('postgres', 'postgresql'):
        url = make_url('postgresql+psycopg://' + rest)
        host = (url.host or '').lower()
        if host not in LOCAL_HOSTS and 'sslmode' not in url.query and 'host' not in url.query:
            url = url.update_query_dict({'sslmode': 'require'})
        return url.render_as_string(hide_password=False)
    return raw  # e.g. sqlite:///nosetouch.db, used as given


def safe_url(url: str) -> str:
    """The database URL without its password, for log lines."""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 - a log line must never fail
        return '(unparseable database URL)'
