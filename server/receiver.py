#!/usr/bin/env python3
"""NoseTouch local receiver: stores the collector app's uploads on this computer.

Same contract as the Google Apps Script backend (backend/google-apps-script/Code.gs): one
UploadEnvelope {kind, study_key, sent_at, record} per POST, the same validation, the same folder
layout and the same JSON answers, so the analysis kit reads both alike:
    <out>/<study_code>/<participant_id>/<session_id>/session.json
    <out>/<study_code>/<participant_id>/<session_id>/trial_<trial_index:03d>_a<attempt>_<trial_id>.json
    <out>/quarantine.log    one JSON line per rejected upload, the columns of Code.gs's 'quarantine'
                            sheet: received_at, reason, bytes, sha256 and the envelope's identifiers.
                            Never the body (it can hold a participant's profile); at most 500 lines
                            per UTC day while the receiver runs. '.log': the analysis kit skips it.
Each data file holds one record (not the envelope). Re-posting a record replaces its file atomically
(temporary file + os.replace), so the app's retries and its session re-sends are safe.

Routes (a trailing slash and a query string are ignored):
    POST /, /ingest, /api/ingest   one envelope, text/plain or application/json ('/api/ingest' in case
                                   a proxy keeps the prefix; the Vite dev proxy strips it)
    GET  the same paths, /health   health check (the app's settings screen probes its endpoint so)
    OPTIONS (any path)             CORS preflight: json_post mode sends Content-Type: application/json
Every answer is JSON and carries Access-Control-Allow-Origin: *. Status codes: 200 stored, 400
rejected (bad JSON or envelope, wrong study key), 413 too large, 404/405/411 wrong request, 500
server error (retry: true). The app takes a 4xx with {ok: false} as a refusal of that record (it
stops retrying it on its own until the participant taps "Upload now") and a 5xx as "retry later".

Usage:  py -3 server/receiver.py [--port 8787] [--host 127.0.0.1] [--out DIR] [--key STUDY_KEY]
                                 [--max-bytes 2000000] [--quiet]
Python standard library only (3.10+). Meant for development and supervised lab sessions: there is
no authentication beyond the optional study key, so do not leave it running unattended on a tunnel.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import hmac
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SERVER_VERSION = '1.0.0'
SERVICE_NAME = 'nosetouch-receiver'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8787
DEFAULT_OUT = Path(__file__).resolve().parent / 'received'
DEFAULT_MAX_BYTES = 2_000_000
RESERVED_STUDY_CODE = 'quarantine'  # older versions kept rejected payloads in <out>/quarantine/
QUARANTINE_LOG = 'quarantine.log'
QUARANTINE_DAILY_CAP = 500  # lines per UTC day, like the quarantine rows of the Apps Script backend
IDENT_MAX_CHARS = 100
DRAIN_FACTOR = 8  # an oversized body up to 8x the limit is read and discarded, a larger one closes the connection
MAX_SAFE_INTEGER = 2**53 - 1  # JavaScript's limit: the app's numbers never exceed it
INGEST_PATHS = frozenset({'/', '/ingest', '/api/ingest'})
# GET on the upload endpoint itself is the health check, as with Apps Script's /exec URL: the app's
# researcher settings probe the configured endpoint with a GET
HEALTH_PATHS = INGEST_PATHS | {'/health'}

# fullmatch (not match + '$'): in Python '$' also matches before a trailing newline
ID_RE = re.compile(r'[A-Za-z0-9_-]{1,64}')
PARTICIPANT_RE = re.compile(r'P-[A-HJ-NP-Z2-9]{6}')
ID_TEXT = '/^[A-Za-z0-9_-]{1,64}$/'  # messages quote the patterns exactly as Code.gs does
PARTICIPANT_TEXT = '/^P-[A-HJ-NP-Z2-9]{6}$/'
# Folder names Windows cannot create (CON, NUL, COM1, ...). The app never produces them (its ids
# are UUIDs), but a hand-made upload must not be able to break the receiver on the lab laptop.
WINDOWS_RESERVED_RE = re.compile(r'(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])', re.IGNORECASE)


class Rejection(Exception):
    """An upload that is refused for good (answered with retry: false, logged in quarantine.log)."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def now_iso() -> str:
    """UTC time in JavaScript's toISOString() format, e.g. 2026-10-23T09:15:02.123Z."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


# ───────────────────────────── validation ─────────────────────────────


def _reject_constant(name: str) -> Any:
    raise ValueError(f'{name} is not valid JSON')  # NaN / Infinity: JSON.parse refuses them too


def _finite_float(text: str) -> float:
    value = float(text)
    if value in (float('inf'), float('-inf')):
        raise ValueError(f'number out of range: {text}')
    return value


def parse_envelope(body: bytes) -> Any:
    """Strict JSON (UTF-8, no BOM, no NaN/Infinity), like JSON.parse in the Apps Script backend."""
    try:
        text = body.decode('utf-8')
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Rejection('malformed_json', 'malformed JSON') from e


def _is_id(v: Any) -> bool:
    return isinstance(v, str) and ID_RE.fullmatch(v) is not None


def _as_int(v: Any) -> int | None:
    """A whole number as JavaScript sees one (Number.isSafeInteger): 7 and 7.0 yes; True, 7.5, '7' no."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if abs(v) <= MAX_SAFE_INTEGER else None
    if isinstance(v, float) and v.is_integer() and abs(v) <= MAX_SAFE_INTEGER:
        return int(v)
    return None


def validate_envelope(env: Any, study_key: str | None) -> None:
    """Raises Rejection unless `env` is an acceptable UploadEnvelope (same rules and codes as Code.gs)."""
    if not isinstance(env, dict):
        raise Rejection('bad_envelope', 'the body must be a JSON object (an UploadEnvelope)')
    kind = env.get('kind')
    if kind not in ('session', 'trial'):
        raise Rejection('bad_kind', "kind must be 'session' or 'trial'")
    if study_key:
        sent = env.get('study_key')
        if not isinstance(sent, str) or not hmac.compare_digest(sent.encode('utf-8'), study_key.encode('utf-8')):
            raise Rejection('bad_study_key', 'study key does not match')
    r = env.get('record')
    if not isinstance(r, dict):
        raise Rejection('bad_record', 'record must be an object')
    if r.get('record_type') != kind:
        raise Rejection('bad_record_type', 'record.record_type does not match kind')
    sv = r.get('schema_version')
    if not isinstance(sv, str) or not sv.startswith('1.'):
        raise Rejection('bad_schema_version', 'unsupported schema_version (this receiver accepts 1.x)')
    if not _is_id(r.get('study_code')):
        raise Rejection('bad_study_code', f'study_code must match {ID_TEXT}')
    if r['study_code'].lower() == RESERVED_STUDY_CODE or WINDOWS_RESERVED_RE.fullmatch(r['study_code']):
        raise Rejection('bad_study_code', 'study_code is reserved')
    pid = r.get('participant_id')
    if not isinstance(pid, str) or PARTICIPANT_RE.fullmatch(pid) is None:
        raise Rejection('bad_participant_id', f'participant_id must match {PARTICIPANT_TEXT}')
    if not _is_id(r.get('session_id')):
        raise Rejection('bad_session_id', f'session_id must match {ID_TEXT}')
    if WINDOWS_RESERVED_RE.fullmatch(r['session_id']):
        raise Rejection('bad_session_id', 'session_id is reserved')
    if kind == 'trial':
        if not _is_id(r.get('trial_id')):
            raise Rejection('bad_trial_id', f'trial_id must match {ID_TEXT}')
        index = _as_int(r.get('trial_index'))
        if index is None or index < 0:
            raise Rejection('bad_trial_index', 'trial_index must be an integer >= 0')
        attempt = _as_int(r.get('attempt'))
        if attempt is None or attempt < 1:
            raise Rejection('bad_attempt', 'attempt must be an integer >= 1')


def record_path(env: dict[str, Any]) -> tuple[str, str]:
    """(folder, file name) of a validated envelope's record, relative to the output folder."""
    r = env['record']
    folder = '/'.join((r['study_code'], r['participant_id'], r['session_id']))
    if env['kind'] == 'session':
        return folder, 'session.json'
    index = _as_int(r['trial_index'])
    attempt = _as_int(r['attempt'])
    return folder, f"trial_{index:03d}_a{attempt}_{r['trial_id']}.json"


# ───────────────────────────── storage ─────────────────────────────


def _replace(src: str, dst: Path, attempts: int = 10) -> None:
    # Windows refuses to replace a file that another thread or a virus scanner holds open for a moment
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.02 * (i + 1))


def atomic_write(path: Path, data: bytes) -> None:
    """Writes `data` to `path` so that readers only ever see the old or the new complete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # '.tmp' suffix: the analysis loader skips it if a crash ever leaves one behind
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def store_record(out: Path, env: dict[str, Any], received_at: str) -> dict[str, Any]:
    folder, name = record_path(env)
    record = env['record']
    data = json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    atomic_write(out.joinpath(*folder.split('/'), name), data)
    return {
        'ok': True,
        'kind': env['kind'],
        'id': record['session_id'] if env['kind'] == 'session' else record['trial_id'],
        'stored_as': f'{folder}/{name}',
        'received_at': received_at,
        'server_version': SERVER_VERSION,
    }


def _ident(v: Any) -> str | None:
    """An identifier for the quarantine log: a string or a number, cut to 100 characters; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        s = v
    elif isinstance(v, int) or (isinstance(v, float) and math.isfinite(v)):
        s = str(v)
    else:
        return None
    return s[:IDENT_MAX_CHARS] + '…' if len(s) > IDENT_MAX_CHARS else s


def rejection_entry(received_at: str, code: str, length: int, sha256: str | None, env: Any) -> dict[str, Any]:
    """The quarantine.log line of a rejected upload: the columns of Code.gs's 'quarantine' sheet.

    Metadata and the identifiers of the envelope (None when the body was not parsed) only: never the
    record body or a profile value. `sha256` is None when the body was too large to be read."""
    e = env if isinstance(env, dict) else {}
    r = e.get('record') if isinstance(e.get('record'), dict) else {}
    return {
        'received_at': received_at,
        'reason': code,
        'bytes': length,
        'sha256': sha256,
        'kind': _ident(e.get('kind')),
        'record_type': _ident(r.get('record_type')),
        'study_code': _ident(r.get('study_code')),
        'participant_id': _ident(r.get('participant_id')),
        'session_id': _ident(r.get('session_id')),
        'record_id': _ident(r.get('trial_id')) or _ident(r.get('session_id')),
        'schema_version': _ident(r.get('schema_version')),
        'app_version': _ident(r.get('app_version')),
    }


# ───────────────────────────── HTTP ─────────────────────────────


class ReceiverServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR would let a second receiver bind the same port silently.
    allow_reuse_address = sys.platform != 'win32'

    def __init__(self, address: tuple[str, int], out: Path, study_key: str | None, max_bytes: int, quiet: bool):
        self.out = Path(out)
        self.study_key = study_key or None
        self.max_bytes = int(max_bytes)
        self.quiet = quiet
        self.quarantine_cap = QUARANTINE_DAILY_CAP
        self._quarantine_lock = threading.Lock()
        self._quarantine_day = ''
        self._quarantine_lines = 0
        super().__init__(address, ReceiverHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f'http://{host}:{port}'

    def log_rejection(self, entry: dict[str, Any]) -> bool:
        """Appends one line to <out>/quarantine.log, at most `quarantine_cap` per UTC day (the count
        restarts with the receiver). Returns whether the line was written; OSError propagates."""
        day = entry['received_at'][:10]  # received_at is a UTC ISO time
        with self._quarantine_lock:
            if day != self._quarantine_day:
                self._quarantine_day, self._quarantine_lines = day, 0
            if self._quarantine_lines >= self.quarantine_cap:
                return False
            self.out.mkdir(parents=True, exist_ok=True)
            # ensure_ascii: one physical line whatever the identifiers contain
            line = json.dumps(entry, separators=(',', ':')) + '\n'
            with open(self.out / QUARANTINE_LOG, 'a', encoding='utf-8', newline='\n') as f:
                f.write(line)
            self._quarantine_lines += 1
            return True


class ReceiverHandler(BaseHTTPRequestHandler):
    server: ReceiverServer
    server_version = f'NoseTouchReceiver/{SERVER_VERSION}'
    sys_version = ''
    protocol_version = 'HTTP/1.1'  # keep-alive; every answer carries Content-Length

    # ── plumbing ──

    def _route(self) -> str:
        path = urlsplit(self.path).path or '/'
        return path.rstrip('/') or '/'

    def _cors(self) -> None:
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '600')
        # Chrome's Private Network Access preflight (a public page calling a local address).
        # getattr: send_error() can run before the headers were parsed.
        headers = getattr(self, 'headers', None)
        if headers is not None and headers.get('Access-Control-Request-Private-Network', '').lower() == 'true':
            self.send_header('Access-Control-Allow-Private-Network', 'true')

    def _send(self, status: int, payload: dict[str, Any] | None, note: str = '') -> None:
        body = b'' if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(int(status))
        self._cors()
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if status != 204:  # a 204 must carry neither a body nor a Content-Length
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
        if self.close_connection:
            self.send_header('Connection', 'close')
        self.end_headers()
        if body and getattr(self, 'command', None) != 'HEAD':
            self.wfile.write(body)
        self._log(int(status), note)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send(status, {'ok': False, 'error': message, 'code': code, 'retry': False,
                            'received_at': now_iso(), 'server_version': SERVER_VERSION}, code)

    def _log(self, status: int, note: str) -> None:
        if self.server.quiet:
            return
        stamp = _dt.datetime.now().strftime('%H:%M:%S')
        request = f"{getattr(self, 'command', None) or '?'} {getattr(self, 'path', '') or '?'}"
        sys.stderr.write(f'{stamp} {self.client_address[0]} {request} -> {status} {note}\n')

    def log_request(self, code: Any = '-', size: Any = '-') -> None:
        pass  # _log() writes one line per answer instead

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (stdlib signature)
        if not self.server.quiet:
            sys.stderr.write(f'{self.address_string()} {format % args}\n')

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # the stdlib answers malformed requests (and unknown methods) with an HTML page; keep every
        # answer JSON, and close the connection: the request was not understood
        if int(code) == 501:
            self._drain()  # unknown method: only here were this request's headers fully parsed
        self.close_connection = True
        name = 'not_implemented' if int(code) == 501 else 'bad_request' if int(code) < 500 else 'server_error'
        try:
            self._error(int(code), name, message or 'bad request')
        except OSError:
            pass

    def _read_body(self, length: int) -> bytes:
        chunks, left = [], length
        while left > 0:
            chunk = self.rfile.read(min(left, 65536))
            if not chunk:
                raise ConnectionError('client closed the connection mid-body')
            chunks.append(chunk)
            left -= len(chunk)
        return b''.join(chunks)

    def _hash_body(self, length: int) -> str:
        """Reads and discards a body in chunks; returns its SHA-256 (hex). Nothing of it is kept."""
        digest, left = hashlib.sha256(), length
        while left > 0:
            chunk = self._read_body(min(left, 65536))
            digest.update(chunk)
            left -= len(chunk)
        return digest.hexdigest()

    def _drain(self) -> None:
        """Reads and discards a body this request will not use. Closing a socket that still holds
        unread data makes the OS reset the connection (always on Windows), and the client would then
        lose our answer. Bodies that are too large, or chunked, close the connection instead."""
        headers = getattr(self, 'headers', None)
        try:
            length = int(headers.get('Content-Length', '0') or 0) if headers is not None else -1
        except ValueError:
            length = -1
        if headers is not None and 'Transfer-Encoding' not in headers and 0 <= length <= self.server.max_bytes * DRAIN_FACTOR:
            try:
                self._read_body(length)
                return
            except OSError:  # ConnectionError included
                pass
        self.close_connection = True

    # ── methods ──

    def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib naming)
        self._send(204, None, 'preflight')

    def do_GET(self) -> None:  # noqa: N802
        if self._route() not in HEALTH_PATHS:
            self._error(404, 'not_found', f'not found: {self._route()}')
            return
        self._send(200, {'ok': True, 'service': SERVICE_NAME, 'server_version': SERVER_VERSION,
                         'time': now_iso(), 'configured': True}, 'health')

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _not_allowed(self) -> None:
        self._drain()
        self._error(405, 'method_not_allowed', f'method {self.command} not allowed (use POST)')

    do_PUT = do_DELETE = do_PATCH = _not_allowed

    def do_POST(self) -> None:  # noqa: N802
        received_at = now_iso()
        if self._route() not in INGEST_PATHS:
            self._drain()
            self._error(404, 'not_found', f'not found: {self._route()} (POST to /ingest)')
            return
        try:
            length = int(self.headers.get('Content-Length', ''))
            if length < 0:
                raise ValueError
        except ValueError:
            self.close_connection = True
            self._error(411, 'length_required', 'Content-Length required')
            return
        srv = self.server
        body: bytes | None = None
        env: Any = None
        sha256: str | None = None
        try:
            if length > srv.max_bytes:
                # size check BEFORE reading it: drained (and hashed for the log) or the connection closes
                if length <= srv.max_bytes * DRAIN_FACTOR:
                    sha256 = self._hash_body(length)
                else:
                    self.close_connection = True
                raise Rejection('too_large', f'payload too large ({length} bytes > {srv.max_bytes})', 413)
            body = self._read_body(length)
            if not body:
                raise Rejection('empty_body', 'empty request body')
            env = parse_envelope(body)
            validate_envelope(env, srv.study_key)
            reply = store_record(srv.out, env, received_at)
            self._send(200, reply, f"{reply['kind']} {reply['id']} -> {reply['stored_as']}")
        except Rejection as rej:
            reply = {'ok': False, 'error': rej.message, 'code': rej.code, 'retry': False,
                     'received_at': received_at, 'server_version': SERVER_VERSION}
            if sha256 is None and body is not None:
                sha256 = hashlib.sha256(body).hexdigest()
            try:
                srv.log_rejection(rejection_entry(received_at, rej.code, length, sha256, env))
            except OSError as e:
                sys.stderr.write(f'quarantine log failed ({rej.code}): {e}\n')
            self._send(rej.status, reply, f'rejected: {rej.code}')
        except ConnectionError:
            self.close_connection = True  # nobody is left to answer
        except Exception as e:  # noqa: BLE001 — every failure must still be a JSON answer
            self.close_connection = True
            reply = {'ok': False, 'error': f'server error: {type(e).__name__}: {e}'[:500], 'retry': True,
                     'received_at': received_at, 'server_version': SERVER_VERSION}
            self._send(500, reply, 'server error')


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, out: Path | str = DEFAULT_OUT,
                key: str | None = None, max_bytes: int = DEFAULT_MAX_BYTES, quiet: bool = False) -> ReceiverServer:
    """A ready (not yet serving) receiver; port 0 picks a free port (see .url)."""
    return ReceiverServer((host, port), Path(out), key, max_bytes, quiet)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='receiver.py',
        description='NoseTouch local receiver: stores uploads from the collector app in the backend folder layout.',
    )
    p.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'TCP port (default {DEFAULT_PORT})')
    p.add_argument('--host', default=DEFAULT_HOST,
                   help=f'interface to listen on (default {DEFAULT_HOST}; the Vite proxy and cloudflared connect locally)')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT, help=f'output folder (default {DEFAULT_OUT})')
    p.add_argument('--key', default=None, help='study key that every upload must carry (default: none required)')
    p.add_argument('--max-bytes', type=int, default=DEFAULT_MAX_BYTES,
                   help=f'largest accepted request in bytes (default {DEFAULT_MAX_BYTES})')
    p.add_argument('--quiet', action='store_true', help='no log line per request')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_bytes <= 0:
        build_parser().error('--max-bytes must be positive')
    try:
        srv = make_server(args.host, args.port, args.out, args.key, args.max_bytes, args.quiet)
    except OSError as e:
        sys.stderr.write(f'cannot listen on {args.host}:{args.port}: {e}\n'
                         '(is another receiver already running? try --port)\n')
        return 1
    print(f'NoseTouch receiver {SERVER_VERSION} listening on {srv.url}')
    print(f'  records go to   {Path(args.out).resolve()}')
    print(f"  study key       {'required' if args.key else 'not required (any upload is accepted)'}")
    print("  app settings    mode 'json_post', endpoint '/api/ingest' (through the Vite proxy)")
    print("                  or 'https://<tunnel>/ingest' (cloudflared tunnel --url " + srv.url + ')')
    print('Press Ctrl+C to stop.', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        srv.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
