"""The FastAPI application: create_app(settings) -> ASGI app (app.py builds it from the environment).

Public routes (CORS for ALLOWED_ORIGINS):
    POST /api/ingest (also /ingest)   one UploadEnvelope; the contract of server/receiver.py
    GET  /api/ingest (also /ingest)   the app's endpoint probe: 200 {ok: true, service, server_version, time,
                                      configured: true} when the database answers, else 503 {ok: false, error,
                                      retry: true, service, server_version, time, configured: false}
    GET  /api/health                  {ok, service, server_version, time, db, counts}
    GET  /                            a short text, no data
Admin routes (ADMIN_TOKEN; see admin.py): GET /admin, /admin/login, /admin/logout, GET /api/export,
GET /api/export.zip, DELETE /api/participants/{participant_id}.
Answers of the ingest route (the app's uploader reads ok / retry / error, uploader.ts):
    200 {ok: true, kind, id, stored_as, received_at, server_version} (+ header X-NoseTouch-Outcome:
        stored | replaced | unchanged | kept_newer)
    400 / 403 / 404 / 405 / 413 {ok: false, error, code, retry: false, received_at, server_version}:
        the phone keeps the record and stops sending it on its own ("Upload now" retries it). A body
        over MAX_BYTES is read to its end before its 413 (up to DRAIN_FACTOR x MAX_BYTES), or a browser
        would see a reset connection instead of the answer; a larger one is cut off with Connection: close.
    500 / 503 {ok: false, error, retry: true, received_at, server_version}: the phone retries later
        (also every upload on a dyno without DATABASE_URL: Settings.database_missing)
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import secrets
import sys
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qs

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import SERVER_VERSION, SERVICE_NAME
from .admin import COOKIE_NAME, DENIED, DISABLED, MISSING, OK, AdminAuth, csp, esc, render_admin, render_login
from .db import REQUEST_ID, Database, kv, set_request_id
from .exporter import ExportFilter, file_stem, gzip_stream, iter_bundle, iter_zip, parse_filter
from .settings import Settings, safe_url
from .store import Stored, delete_participant, overview, store_envelope
from .validation import (
    Rejection,
    canonical_record,
    envelope_identifiers,
    is_participant_id,
    iso,
    now_utc,
    parse_envelope,
    validate_envelope,
)

log = logging.getLogger('nosetouch')

INGEST_PATHS = ('/api/ingest', '/api/ingest/', '/ingest', '/ingest/')
CORS_PATHS = frozenset(INGEST_PATHS + ('/api/health', '/api/health/'))
CORS_METHODS = 'GET, POST'  # POST: the uploader (preflighted: Content-Type: application/json); GET: the probe
CORS_HEADERS = 'Content-Type'
CORS_MAX_AGE = '86400'
DRAIN_FACTOR = 8  # like receiver.py: a body over MAX_BYTES is read to its end (and discarded) up to 8 x MAX_BYTES
# How long startup waits for an unreachable database. uvicorn binds $PORT only after startup, and Heroku
# kills a web dyno that has not bound it within 60 s (R10), then restarts it with growing delays. One
# more attempt may start just before this and take up to the 10 s connect_timeout (db.make_engine).
STARTUP_DB_WAIT_S = 25.0
_DB_DOWN = (OperationalError, InterfaceError, PoolTimeoutError)


def _setup_logging() -> None:
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    access = logging.getLogger('uvicorn.access')
    if not any(isinstance(f, _RedactTokenFilter) for f in access.filters):
        access.addFilter(_RedactTokenFilter())


class _RedactTokenFilter(logging.Filter):
    """uvicorn's access log prints the full path: never a ?token=... value."""

    _re = re.compile(r'(token=)[^&\s"]*', re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and any(isinstance(a, str) and 'token=' in a.lower() for a in args):
            record.args = tuple(self._re.sub(r'\1[redacted]', a) if isinstance(a, str) else a for a in args)
        return True


# ───────────────────────────── middleware ─────────────────────────────


def _route_path(scope: Scope) -> str:
    return scope.get('path') or '/'


class CORSMiddleware:
    """CORS for the public routes only (the admin routes never answer cross-origin requests).

    Preflight (OPTIONS with Origin + Access-Control-Request-Method) from an allowed origin: 204 with
    Access-Control-Allow-Origin: <origin>, -Methods: GET, POST, -Headers: Content-Type, -Max-Age:
    86400 and no -Allow-Credentials (the uploader sends credentials: 'omit'). Other answers on these
    routes get Access-Control-Allow-Origin for allowed origins and Vary: Origin."""

    def __init__(self, app: ASGIApp, settings: Settings):
        self.app = app
        self.settings = settings

    def _allow_origin_value(self, origin: str | None) -> str | None:
        if origin is None or not self.settings.origin_allowed(origin):
            return None
        return '*' if self.settings.any_origin else origin

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or _route_path(scope) not in CORS_PATHS:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get('origin')
        allow = self._allow_origin_value(origin)
        if scope['method'] == 'OPTIONS':
            preflight = origin is not None and 'access-control-request-method' in headers
            if preflight and allow is None:
                body = b'{"ok":false,"error":"origin not allowed","code":"bad_origin","retry":false}'
                await _send_raw(send, 403, body, [(b'content-type', b'application/json'),
                                                  (b'vary', b'Origin, Access-Control-Request-Method, '
                                                            b'Access-Control-Request-Headers')])
                return
            out = [(b'vary', b'Origin, Access-Control-Request-Method, Access-Control-Request-Headers'),
                   (b'cache-control', b'no-store')]
            if allow is not None:
                out += [(b'access-control-allow-origin', allow.encode('latin-1')),
                        (b'access-control-allow-methods', CORS_METHODS.encode()),
                        (b'access-control-allow-headers', CORS_HEADERS.encode()),
                        (b'access-control-max-age', CORS_MAX_AGE.encode())]
            await _send_raw(send, 204, b'', out)
            return

        async def send_with_cors(message: Message) -> None:
            if message['type'] == 'http.response.start':
                h = MutableHeaders(scope=message)
                h.add_vary_header('Origin')
                if allow is not None:
                    h['Access-Control-Allow-Origin'] = allow
            await send(message)

        await self.app(scope, receive, send_with_cors)


class SecurityMiddleware:
    """HTTPS only on Heroku (FORCE_HTTPS, default when DYNO is set) plus a few safe default headers."""

    def __init__(self, app: ASGIApp, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        if self.settings.force_https and scope.get('scheme') == 'http':
            headers = Headers(scope=scope)
            host = headers.get('host', '')
            if scope['method'] in ('GET', 'HEAD') and host:
                qs = scope.get('query_string', b'').decode('latin-1')
                location = f'https://{host}{_route_path(scope)}' + (f'?{qs}' if qs else '')
                await _send_raw(send, 308, b'', [(b'location', location.encode('latin-1'))])
            else:
                await _send_raw(send, 403, b'{"ok":false,"error":"use https","code":"https_required","retry":false}',
                                [(b'content-type', b'application/json')])
            return
        https = scope.get('scheme') == 'https'

        async def send_with_headers(message: Message) -> None:
            if message['type'] == 'http.response.start':
                h = MutableHeaders(scope=message)
                h.setdefault('X-Content-Type-Options', 'nosniff')
                h.setdefault('Referrer-Policy', 'no-referrer')
                if https and self.settings.force_https:
                    h.setdefault('Strict-Transport-Security', 'max-age=31536000')
            await send(message)

        token = set_request_id(Headers(scope=scope).get('x-request-id'))
        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            REQUEST_ID.reset(token)


async def _send_raw(send: Send, status: int, body: bytes, headers: list[tuple[bytes, bytes]]) -> None:
    out = list(headers)
    if status != 204:
        out.append((b'content-length', str(len(body)).encode()))
    out.append((b'x-content-type-options', b'nosniff'))
    await send({'type': 'http.response.start', 'status': status, 'headers': out})
    await send({'type': 'http.response.body', 'body': body if status != 204 else b''})


# ───────────────────────────── answers ─────────────────────────────


def _json(status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(payload, status_code=status, headers={'Cache-Control': 'no-store', **(headers or {})})


def _refusal(status: int, code: str, message: str, received_at: str | None = None) -> dict[str, Any]:
    return {'ok': False, 'error': message, 'code': code, 'retry': False,
            'received_at': received_at or iso(now_utc()), 'server_version': SERVER_VERSION}


def _retry_later(message: str, received_at: str) -> dict[str, Any]:
    return {'ok': False, 'error': message, 'retry': True, 'received_at': received_at,
            'server_version': SERVER_VERSION}


DB_DOWN_MESSAGE = 'database unavailable, try again later'
NO_DATABASE_MESSAGE = ('no database: DATABASE_URL is not set on this server (attach Heroku Postgres); '
                       'nothing is stored until then, try again later')


class _TooLarge(Exception):
    """A body over the limit. `drained`: it was read to its end (so the answer can keep the connection),
    with its SHA-256 for the log; otherwise reading stopped early and the connection must be closed."""

    def __init__(self, seen: int, drained: bool = False, sha256: str | None = None):
        self.seen = seen
        self.drained = drained
        self.sha256 = sha256


async def _read_capped(request: Request, limit: int, drain_limit: int = 0) -> bytes:
    """The request body, read in chunks, at most `limit` bytes (with or without Content-Length, so a
    chunked body cannot bypass the cap); a larger body raises _TooLarge.

    Up to `drain_limit` bytes, a larger body is first read to its end and discarded (constant memory,
    only hashed). Closing a connection while request bytes are still arriving makes the OS reset it,
    and a browser, which sends the whole body before it reads the answer, then loses the 413: the app
    sees a network error and re-sends the record forever instead of setting it aside. Beyond
    `drain_limit` reading stops at once, and the caller closes the connection."""
    drain_limit = max(drain_limit, limit)
    data = bytearray()
    digest: Any = None  # once over the limit, the body is only hashed
    seen = 0
    async for chunk in request.stream():
        seen += len(chunk)
        if digest is None and seen <= limit:
            data += chunk
            continue
        if seen > drain_limit:
            raise _TooLarge(seen)
        if digest is None:
            digest = hashlib.sha256(data)
            data = bytearray()
        digest.update(chunk)
    if digest is not None:
        raise _TooLarge(seen, drained=True, sha256=digest.hexdigest())
    return bytes(data)


def _declared_length(request: Request) -> int | None:
    raw = request.headers.get('content-length')
    if raw is None:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n >= 0 else None


# ───────────────────────────── application ─────────────────────────────


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    _setup_logging()
    db = Database(settings)
    auth = AdminAuth(settings)
    # why the database cannot take data: the text of every 503 that the phones and the researcher see
    down_message = NO_DATABASE_MESSAGE if settings.database_missing else DB_DOWN_MESSAGE

    def init_db() -> bool:
        if db.ready:
            return True
        if settings.database_missing:
            return False  # never a throwaway database on a dyno (Settings.database_missing; logged at startup)
        try:
            db.init_schema()
            db.maybe_prune(now_utc(), every_s=0)
            return True
        except Exception as e:  # noqa: BLE001 - logged; uploads answer 503 until the database is back
            log.error(kv(event='db_init_failed', error=type(e).__name__, database=safe_url(settings.database_url)))
            return False

    def db_ping() -> bool:
        """Whether the database answers now (schema in place, SELECT 1). Never raises. Bounded by the
        engine's connect_timeout (10 s) and pool_timeout (15 s), below the app's 20 s probe timeout."""
        try:
            if not init_db():
                return False
            with db.read() as conn:
                conn.execute(text('SELECT 1'))
            return True
        except Exception:  # noqa: BLE001
            return False

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if settings.database_missing:
            log.error(kv(event='database_missing', error=NO_DATABASE_MESSAGE,
                         fix='heroku addons:create heroku-postgresql:essential-1 -a <app> (or set ALLOW_EPHEMERAL_DB=1 '
                             'for a throwaway test app); every upload answers 503 retry until then'))
        else:
            # a database that is still starting gets a few tries, within STARTUP_DB_WAIT_S; after that the
            # port is bound anyway: every request runs init_db() again and answers 503 retry until it works
            deadline = _monotonic() + STARTUP_DB_WAIT_S
            attempt = 0
            while not await run_in_threadpool(init_db):
                attempt += 1
                delay = min(2.0 * attempt, 8.0)
                if _monotonic() + delay >= deadline:
                    break
                await _sleep(delay)
        admin = ('disabled (ADMIN_TOKEN not set)' if not settings.admin_token
                 else 'enabled, BUT ADMIN_TOKEN is short: use 32+ random characters' if len(settings.admin_token) < 24
                 else 'enabled')
        log.info(kv(event='startup', service=SERVICE_NAME, version=SERVER_VERSION,
                    db='NONE (DATABASE_URL missing: nothing is stored)' if settings.database_missing else db.dialect,
                    database=safe_url(settings.database_url), db_ready=db.ready,
                    study_key='set' if settings.study_key else 'NOT SET (any upload accepted)', admin=admin,
                    origins=list(settings.allowed_origins), max_bytes=settings.max_bytes,
                    force_https=settings.force_https, notes=list(settings.notes)))
        try:
            yield
        finally:
            db.engine.dispose()

    app = FastAPI(title='NoseTouch upload service', version=SERVER_VERSION, docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    # added last = outermost: HTTPS check first, then CORS
    app.add_middleware(CORSMiddleware, settings=settings)
    app.add_middleware(SecurityMiddleware, settings=settings)

    # ── errors: every answer is JSON, in the receiver's shape ──

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        code = {404: 'not_found', 405: 'method_not_allowed'}.get(exc.status_code, 'bad_request')
        message = (f'not found: {request.url.path}' if exc.status_code == 404
                   else f'method {request.method} not allowed' if exc.status_code == 405
                   else str(exc.detail))
        return _json(exc.status_code, _refusal(exc.status_code, code, message))

    # ── public ──

    @app.get('/', include_in_schema=False)
    async def landing() -> Response:
        return PlainTextResponse('NoseTouch upload service. The study app posts its records to /api/ingest.\n',
                                 headers={'Cache-Control': 'no-store'})

    @app.get('/robots.txt', include_in_schema=False)
    async def robots() -> Response:
        return PlainTextResponse('User-agent: *\nDisallow: /\n')

    async def probe(request: Request) -> Response:
        """The app's endpoint probe (researcher settings, Test). ok: true only when an upload could be
        stored now: with the database unreachable it answers 503, which the app shows as an error."""
        base = {'service': SERVICE_NAME, 'server_version': SERVER_VERSION}
        if await run_in_threadpool(db_ping):
            return _json(200, {'ok': True, **base, 'time': iso(now_utc()), 'configured': True})
        return _json(503, {'ok': False, 'error': down_message, 'retry': True, **base,
                           'time': iso(now_utc()), 'configured': False}, {'Retry-After': '30'})

    async def ingest(request: Request) -> Response:
        started = time.perf_counter()
        received = now_utc()
        received_at = iso(received)
        origin = request.headers.get('origin')
        if origin is not None and not settings.origin_allowed(origin):
            # a browser page on another site (the app itself is preflighted and never gets here)
            return await _refuse(Rejection('bad_origin', 'origin not allowed', 403), None, None, None,
                                 received, started, close=True)
        declared = _declared_length(request)
        drain_limit = settings.max_bytes * DRAIN_FACTOR
        try:
            if declared is not None and declared > drain_limit:
                raise _TooLarge(declared)  # refused unread, and the connection closes (see _read_capped)
            body = await _read_capped(request, settings.max_bytes, drain_limit)
        except _TooLarge as big:
            size = declared if declared is not None else big.seen
            if declared is not None or big.drained:
                message = f'payload too large ({size} bytes > {settings.max_bytes})'
            else:  # chunked and cut off: only what was read is known
                message = f'payload too large (more than {settings.max_bytes} bytes)'
            return await _refuse(Rejection('too_large', message, 413), None, size, big.sha256, received, started,
                                 close=not big.drained)
        except ClientDisconnect:
            log.info(kv(event='ingest', outcome='client_disconnected'))
            return Response(status_code=400)
        seen: dict[str, Any] = {'env': None}

        def process() -> Rejection | Stored | None:
            """Parse, validate and store (in a worker thread: a 4 MB parse must not block the loop).
            Returns the Rejection, the Stored result, or None when the database is unavailable."""
            seen['sha'] = hashlib.sha256(body).hexdigest()
            try:
                if not body:
                    raise Rejection('empty_body', 'empty request body')
                seen['env'] = env = parse_envelope(body)
                validate_envelope(env, settings.study_key)
                digest, size = canonical_record(env['record'])
            except Rejection as rej:
                return rej
            if not init_db():
                return None
            stored = store_envelope(db, env, digest=digest, size=size, body_bytes=len(body),
                                    body_sha256=seen['sha'], received_at=received)
            db.maybe_prune(received)
            return stored

        try:
            result = await run_in_threadpool(process)
        except _DB_DOWN as e:
            log.error(kv(event='ingest', outcome='db_unavailable', error=type(e).__name__))
            return _unavailable(received_at, len(body), started)
        except Exception as e:  # noqa: BLE001 - every failure must still be a JSON answer
            ids = envelope_identifiers(seen['env'])
            log.exception(kv(event='ingest', outcome='server_error', error=type(e).__name__, kind=ids['kind'],
                             id=ids['record_id']))
            name = 'database error' if isinstance(e, DBAPIError) else type(e).__name__
            return _json(500, _retry_later(f'server error: {name}', received_at), {'Connection': 'close'})
        if result is None:
            return _unavailable(received_at, len(body), started)
        if isinstance(result, Rejection):
            return await _refuse(result, seen['env'], len(body), seen['sha'], received, started)
        env = seen['env']
        log.info(kv(event='ingest', outcome=result.outcome, reason=result.reason, kind=result.kind, id=result.record_id,
                    session_id=env['record']['session_id'], participant_id=env['record']['participant_id'],
                    bytes=len(body), ms=round((time.perf_counter() - started) * 1000, 1)))
        reply = {'ok': True, 'kind': result.kind, 'id': result.record_id, 'stored_as': result.stored_as,
                 'received_at': received_at, 'server_version': SERVER_VERSION}
        return _json(200, reply, {'X-NoseTouch-Outcome': result.outcome})

    async def _refuse(rej: Rejection, env: Any, size: int | None, body_sha: str | None, received: Any,
                      started: float, close: bool = False) -> Response:
        ids = envelope_identifiers(env)
        if not settings.database_missing:  # (no database: the log line below is all there is)
            try:
                await run_in_threadpool(db.log_rejection, at=received, code=rej.code, ids=ids, size=size,
                                        sha256=body_sha)
            except Exception as e:  # noqa: BLE001 - a log failure is still a plain rejection
                log.warning(kv(event='rejection_log_failed', code=rej.code, error=type(e).__name__))
        log.info(kv(event='ingest', outcome='rejected', code=rej.code, kind=ids['kind'], id=ids['record_id'],
                    participant_id=ids['participant_id'], bytes=size,
                    ms=round((time.perf_counter() - started) * 1000, 1)))
        headers = {'Connection': 'close'} if close else None
        return _json(rej.status, _refusal(rej.status, rej.code, rej.message, iso(received)), headers)

    def _unavailable(received_at: str, size: int, started: float) -> Response:
        log.info(kv(event='ingest', outcome='retry_later', bytes=size,
                    ms=round((time.perf_counter() - started) * 1000, 1)))
        return _json(503, _retry_later(down_message, received_at), {'Retry-After': '30'})

    for path in INGEST_PATHS:
        app.add_api_route(path, ingest, methods=['POST'], include_in_schema=False)
        app.add_api_route(path, probe, methods=['GET', 'HEAD'], include_in_schema=False)

    async def health(request: Request) -> Response:
        def check() -> tuple[int, dict[str, Any]]:
            base = {'service': SERVICE_NAME, 'server_version': SERVER_VERSION, 'time': iso(now_utc())}
            if settings.database_missing:
                return 503, {'ok': False, **base, 'db': {'ok': False, 'dialect': None, 'error': NO_DATABASE_MESSAGE},
                             'counts': None}
            try:
                if not init_db():
                    raise RuntimeError('schema not initialised')
                with db.read() as conn:
                    counts = db.counts(conn)
                    version = db.schema_version(conn)
                return 200, {'ok': True, **base, 'db': {'ok': True, 'dialect': db.dialect, 'schema_version': version},
                             'counts': counts}
            except Exception as e:  # noqa: BLE001
                return 503, {'ok': False, **base, 'db': {'ok': False, 'dialect': db.dialect, 'error': type(e).__name__},
                             'counts': None}

        status, payload = await run_in_threadpool(check)
        return _json(status, payload)

    app.add_api_route('/api/health', health, methods=['GET'], include_in_schema=False)
    app.add_api_route('/api/health/', health, methods=['GET'], include_in_schema=False)

    # ── admin ──

    def api_auth(request: Request) -> Response | None:
        state = auth.check(request.headers, request.cookies)
        if state == OK:
            return None
        if state == DISABLED:
            return _json(503, _refusal(503, 'admin_disabled', 'admin endpoints are off: set ADMIN_TOKEN'))
        if state == MISSING:
            return _json(401, _refusal(401, 'unauthorized', 'admin token required (Authorization: Bearer <token>)'),
                         {'WWW-Authenticate': 'Bearer realm="nosetouch-admin"'})
        log.warning(kv(event='admin_auth_failed', path=request.url.path))
        return _json(403, _refusal(403, 'forbidden', 'admin token rejected'))

    def html_page(content: str, status: int = 200, nonce: str = '') -> HTMLResponse:
        return HTMLResponse(content, status_code=status, headers={
            'Content-Security-Policy': csp(nonce), 'Cache-Control': 'no-store', 'X-Frame-Options': 'DENY'})

    def login_response(token: str | None, nonce: str) -> Response:
        if not auth.enabled:
            return html_page(render_login(nonce, disabled=True), 503, nonce)
        if token is None:
            return html_page(render_login(nonce), 200, nonce)
        if not auth.token_ok(token):
            log.warning(kv(event='admin_login_failed'))
            return html_page(render_login(nonce, 'That token is not right.'), 403, nonce)
        resp = RedirectResponse('/admin', status_code=303, headers={'Cache-Control': 'no-store'})
        resp.set_cookie(COOKIE_NAME, auth.make_cookie(), max_age=settings.admin_cookie_max_age, path='/',
                        secure=True, httponly=True, samesite='strict')
        log.info(kv(event='admin_login'))
        return resp

    @app.get('/admin/login', include_in_schema=False)
    async def login_get(request: Request) -> Response:
        return login_response(request.query_params.get('token'), secrets.token_urlsafe(16))

    @app.post('/admin/login', include_in_schema=False)
    async def login_post(request: Request) -> Response:
        try:
            raw = await _read_capped(request, 4096)
        except (_TooLarge, ClientDisconnect):
            return _json(413, _refusal(413, 'too_large', 'login form too large'))
        token = (parse_qs(raw.decode('utf-8', 'replace')).get('token') or [''])[0]
        return login_response(token, secrets.token_urlsafe(16))

    @app.get('/admin/logout', include_in_schema=False)
    async def logout() -> Response:
        resp = RedirectResponse('/admin/login', status_code=303, headers={'Cache-Control': 'no-store'})
        resp.delete_cookie(COOKIE_NAME, path='/', secure=True, httponly=True, samesite='strict')
        return resp

    @app.get('/admin', include_in_schema=False)
    @app.get('/admin/', include_in_schema=False)
    async def admin_page(request: Request) -> Response:
        nonce = secrets.token_urlsafe(16)
        state = auth.check(request.headers, request.cookies)
        if state == DISABLED:
            return html_page(render_login(nonce, disabled=True), 503, nonce)
        if state == MISSING:
            return html_page(render_login(nonce), 401, nonce)
        if state == DENIED:
            return html_page(render_login(nonce, 'That token is not right.'), 403, nonce)
        if settings.database_missing:
            return html_page(f'<!doctype html><title>NoseTouch admin</title><p>{esc(NO_DATABASE_MESSAGE)}. '
                             'Every upload answers 503 and stays on the phones until then.</p>', 503, nonce)
        try:
            data = await run_in_threadpool(overview, db, now_utc())
        except Exception as e:  # noqa: BLE001
            log.error(kv(event='admin_page_failed', error=type(e).__name__))
            return html_page(f'<!doctype html><title>NoseTouch admin</title><p>Database unavailable '
                             f'({type(e).__name__}). Reload in a minute.</p>', 503, nonce)
        return html_page(render_admin(data, nonce), 200, nonce)

    def export_filter(request: Request) -> ExportFilter | Response:
        try:
            return parse_filter(request.query_params)
        except ValueError as e:
            return _json(400, _refusal(400, 'bad_filter', str(e)))

    async def db_unavailable() -> Response | None:
        """503 before a streamed answer starts, when the database cannot be reached."""
        if await run_in_threadpool(db_ping):
            return None
        return _json(503, _retry_later(down_message, iso(now_utc())), {'Retry-After': '30'})

    @app.get('/api/export', include_in_schema=False)
    async def export_json(request: Request) -> Response:
        denied = api_auth(request)
        if denied is not None:
            return denied
        f = export_filter(request)
        if isinstance(f, Response):
            return f
        down = await db_unavailable()
        if down is not None:
            return down
        now = now_utc()
        log.info(kv(event='export', format='json', participant_id=f.participant_id, since=iso(f.since),
                    include_simulated=f.include_simulated))
        chunks = iter_bundle(db, f, now)
        headers = {'Cache-Control': 'no-store', 'Vary': 'Accept-Encoding',
                   'Content-Disposition': f'inline; filename="{file_stem(f, now)}.json"'}
        if 'gzip' in request.headers.get('accept-encoding', '').lower():
            chunks = gzip_stream(chunks)
            headers['Content-Encoding'] = 'gzip'
        return StreamingResponse(chunks, media_type='application/json', headers=headers)

    @app.get('/api/export.zip', include_in_schema=False)
    async def export_zip(request: Request) -> Response:
        denied = api_auth(request)
        if denied is not None:
            return denied
        f = export_filter(request)
        if isinstance(f, Response):
            return f
        down = await db_unavailable()
        if down is not None:
            return down
        now = now_utc()
        log.info(kv(event='export', format='zip', participant_id=f.participant_id, since=iso(f.since),
                    include_simulated=f.include_simulated))
        return StreamingResponse(iter_zip(db, f), media_type='application/zip', headers={
            'Cache-Control': 'no-store', 'Content-Disposition': f'attachment; filename="{file_stem(f, now)}.zip"'})

    @app.delete('/api/participants/{participant_id}', include_in_schema=False)
    async def delete_one(participant_id: str, request: Request) -> Response:
        denied = api_auth(request)
        if denied is not None:
            return denied
        if not is_participant_id(participant_id):
            return _json(400, _refusal(400, 'bad_participant_id', 'participant_id must match /^P-[A-HJ-NP-Z2-9]{6}$/'))
        try:
            deleted = await run_in_threadpool(delete_participant, db, participant_id)
        except Exception as e:  # noqa: BLE001 - nothing was deleted (one transaction)
            log.error(kv(event='participant_delete_failed', participant_id=participant_id, error=type(e).__name__))
            return _json(503, _retry_later('database unavailable, nothing was deleted: try again', iso(now_utc())))
        log.info(kv(event='participant_deleted', participant_id=participant_id, **deleted))
        return _json(200, {'ok': True, 'participant_id': participant_id, 'deleted': deleted})

    return app


async def _sleep(seconds: float) -> None:
    await anyio.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()
