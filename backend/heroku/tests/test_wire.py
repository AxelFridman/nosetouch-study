"""The ingest route over a real socket (uvicorn, as on Heroku), where TestClient cannot look.

A browser sends the whole request body before it reads the answer. A server that answers early and
then closes the connection while body bytes are still arriving makes the OS reset the connection: the
browser reports net::ERR_CONNECTION_RESET, the app files that as a network error, and the phone sends
the record again and again instead of seeing the 413 that would stop it. These tests send bodies the
way a browser does and check that the answer arrives and the connection stays usable.
"""
from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from records import envelope, make_trial

from nosetouch_server import main as main_module
from nosetouch_server.main import create_app
from nosetouch_server.settings import Settings

MAX = 100_000
DRAIN = MAX * main_module.DRAIN_FACTOR


@contextlib.contextmanager
def live_server(**over: Any) -> Iterator[int]:
    """uvicorn (h11, like the Procfile) on a free local port, in a thread; yields the port."""
    app = create_app(Settings(**over))
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning', access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError('uvicorn did not start')
        time.sleep(0.02)
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(15)


@pytest.fixture(scope='module')
def port() -> Iterator[int]:
    with live_server(max_bytes=MAX) as p:
        yield p


def _head(port: int, length: int | None, path: str = '/api/ingest') -> bytes:
    framing = 'Transfer-Encoding: chunked' if length is None else f'Content-Length: {length}'
    return (f'POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n'
            f'{framing}\r\n\r\n').encode('ascii')


def _chunked(body: bytes, size: int = 16_384) -> bytes:
    parts = [f'{len(body[i:i + size]):x}\r\n'.encode() + body[i:i + size] + b'\r\n' for i in range(0, len(body), size)]
    return b''.join(parts) + b'0\r\n\r\n'


def _send_slowly(sock: socket.socket, data: bytes, piece: int = 16_384) -> None:
    """Like a phone on a mobile network: the body follows the headers a little later, in pieces."""
    time.sleep(0.3)
    for i in range(0, len(data), piece):
        sock.sendall(data[i:i + piece])


def _read_answer(sock: socket.socket) -> tuple[int, dict[str, str], Any]:
    data = b''
    while b'\r\n\r\n' not in data:
        chunk = sock.recv(65_536)
        if not chunk:
            raise ConnectionError('connection closed before the answer')
        data += chunk
    head, _, rest = data.partition(b'\r\n\r\n')
    lines = head.decode('latin-1').split('\r\n')
    headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(':') for line in lines[1:])}
    length = int(headers.get('content-length', '0'))
    while len(rest) < length:
        chunk = sock.recv(65_536)
        if not chunk:
            raise ConnectionError('connection closed in the middle of the answer')
        rest += chunk
    return int(lines[0].split()[1]), headers, json.loads(rest[:length]) if length else None


def _oversized(size: int) -> bytes:
    body = json.dumps(envelope('trial', make_trial(notes='n' * size))).encode('utf-8')
    return body[:size]  # cut to the size: it is refused for its size before anyone parses it


def _valid_upload(port: int) -> bytes:
    body = json.dumps(envelope('trial', make_trial(trial_id='after-413'))).encode('utf-8')
    return _head(port, len(body)) + body


@pytest.mark.parametrize('size', [MAX + 1, 3 * MAX, DRAIN])
def test_a_too_large_body_gets_its_413_after_the_client_sent_it_all(port, size):
    body = _oversized(size)
    with socket.create_connection(('127.0.0.1', port), timeout=20) as sock:
        sock.sendall(_head(port, len(body)))
        _send_slowly(sock, body)  # raises ConnectionResetError/BrokenPipeError if the server gave up on us
        status, headers, reply = _read_answer(sock)
        assert status == 413
        assert (reply['ok'], reply['code'], reply['retry']) == (False, 'too_large', False)
        assert f'({size} bytes > {MAX})' in reply['error']
        assert headers.get('connection', '').lower() != 'close'
        # the body was consumed exactly: the same connection carries the next upload
        sock.sendall(_valid_upload(port))
        status, _, reply = _read_answer(sock)
        assert (status, reply['ok']) == (200, True)


def test_a_too_large_chunked_body_gets_its_413_after_the_client_sent_it_all(port):
    body = _oversized(3 * MAX)
    with socket.create_connection(('127.0.0.1', port), timeout=20) as sock:
        sock.sendall(_head(port, None))
        _send_slowly(sock, _chunked(body))
        status, headers, reply = _read_answer(sock)
        assert (status, reply['code'], reply['retry']) == (413, 'too_large', False)
        assert f'({len(body)} bytes > {MAX})' in reply['error']
        assert headers.get('connection', '').lower() != 'close'
        sock.sendall(_valid_upload(port))
        assert _read_answer(sock)[0] == 200


def test_a_body_beyond_the_drain_limit_is_refused_unread_and_the_connection_closes(port):
    declared = DRAIN + 1
    with socket.create_connection(('127.0.0.1', port), timeout=20) as sock:
        sock.sendall(_head(port, declared))  # the answer comes at once, before any body byte
        status, headers, reply = _read_answer(sock)
        assert (status, reply['code'], reply['retry']) == (413, 'too_large', False)
        assert f'({declared} bytes > {MAX})' in reply['error']
        assert headers['connection'].lower() == 'close'
        assert sock.recv(1) == b''  # closed by the server


def test_a_body_within_the_limit_keeps_the_connection(port):
    with socket.create_connection(('127.0.0.1', port), timeout=20) as sock:
        for _ in range(2):
            sock.sendall(_valid_upload(port))
            status, headers, reply = _read_answer(sock)
            assert (status, reply['ok']) == (200, True)
            assert headers.get('connection', '').lower() != 'close'
