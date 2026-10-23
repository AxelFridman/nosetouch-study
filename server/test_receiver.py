"""Tests for server/receiver.py: a real server on a free port in a background thread.

Run from the repository root:  python -m pytest server -q   (with the analysis venv active)
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import json
import socket
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receiver  # noqa: E402

STUDY = 'nose-touch-v1'
PID = 'P-ABCDEF'
SID = 'sess-0001'
T0 = '2026-10-23T10:00:00.000Z'


# ───────────────────────────── fixtures ─────────────────────────────


def make_session(**over: Any) -> dict[str, Any]:
    rec = {
        'record_type': 'session', 'schema_version': '1.1.0', 'app_version': '0.1.0',
        'session_id': SID, 'participant_id': PID, 'study_code': STUDY,
        'started_at': T0, 'updated_at': T0, 'ended_at': None, 'status': 'in_progress',
        'participant': {'participant_id': PID, 'nickname': None, 'age_years': 31},
        'progress': {'next_trial_index': 0, 'n_attempts': 0, 'n_accepted': 0},
        'timezone_offset_min': 180, 'simulated': False, 'notes': None,
    }
    rec.update(over)
    return rec


def make_trial(**over: Any) -> dict[str, Any]:
    rec = {
        'record_type': 'trial', 'schema_version': '1.1.0', 'app_version': '0.1.0',
        'trial_id': 'trial-0001', 'session_id': SID, 'participant_id': PID, 'study_code': STUDY,
        'created_at': T0, 'block_index': 0, 'trial_index': 0, 'trial_in_block': 0, 'attempt': 1,
        'simulated': False,
        'condition': {'hand': 'right', 'orientation': 'upright_toward', 'speed': 'natural'},
        'outcome': {'status': 'complete', 'accepted': True, 'participant_flag': None, 'note': None},
        'streams': {'motion': {'t': [0, 16.7, 33.4], 'acc_x': [0.01, -0.5, None], 'rr_alpha': [0.5, 120.25, 10]}},
    }
    rec.update(over)
    return rec


def envelope(kind: str, record: Any, key: str | None = None, **over: Any) -> dict[str, Any]:
    env = {'kind': kind, 'study_key': key, 'sent_at': T0, 'record': record}
    env.update(over)
    return env


class Client:
    def __init__(self, srv: receiver.ReceiverServer):
        self.srv = srv
        self.out = srv.out

    def request(self, method: str, path: str = '/', body: bytes | str | None = None,
                headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
        data = body.encode('utf-8') if isinstance(body, str) else body
        req = urllib.request.Request(self.srv.url + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()

    def post(self, payload: Any, path: str = '/', content_type: str = 'text/plain;charset=UTF-8') -> tuple[int, dict[str, Any]]:
        body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
        status, _, raw = self.request('POST', path, body, {'Content-Type': content_type})
        return status, json.loads(raw)

    def all_files(self) -> list[str]:
        if not self.out.exists():
            return []
        return sorted(p.relative_to(self.out).as_posix() for p in self.out.rglob('*') if p.is_file())

    def data_files(self) -> list[str]:
        return [f for f in self.all_files() if f != receiver.QUARANTINE_LOG]

    def quarantine_log(self) -> list[dict[str, Any]]:
        log = self.out / receiver.QUARANTINE_LOG
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding='utf-8').split('\n') if line]

    def stray_temp_files(self) -> list[Path]:
        return [p for p in self.out.rglob('*.tmp')] if self.out.exists() else []


@pytest.fixture
def start(tmp_path):
    started: list[tuple[receiver.ReceiverServer, threading.Thread]] = []

    def _start(**kw: Any) -> Client:
        srv = receiver.make_server(host='127.0.0.1', port=0, out=tmp_path / f'received{len(started)}', quiet=True, **kw)
        thread = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        started.append((srv, thread))
        return Client(srv)

    yield _start
    for srv, thread in started:
        srv.shutdown()
        srv.server_close()
        thread.join(5)


@pytest.fixture
def client(start) -> Client:
    return start()


# ───────────────────────────── health and CORS ─────────────────────────────


def test_health_check_is_json_with_cors(client: Client):
    status, headers, raw = client.request('GET', '/')
    assert status == 200
    assert headers['content-type'].startswith('application/json')
    assert headers['access-control-allow-origin'] == '*'
    body = json.loads(raw)
    assert body['ok'] is True
    assert body['service'] == 'nosetouch-receiver'
    assert body['server_version'] == receiver.SERVER_VERSION
    assert body['time'].endswith('Z') and len(body['time']) == 24
    # the app's settings screen probes the configured endpoint itself with a GET
    for path in ('/health?x=1', '/ingest', '/api/ingest/'):
        status, _, raw = client.request('GET', path)
        assert status == 200 and json.loads(raw)['service'] == 'nosetouch-receiver', path


def test_preflight_allows_json_posts_from_any_origin(client: Client):
    status, headers, raw = client.request('OPTIONS', '/ingest', headers={
        'Origin': 'https://example.github.io',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type',
    })
    assert status == 204 and raw == b''
    assert headers['access-control-allow-origin'] == '*'
    assert {m.strip() for m in headers['access-control-allow-methods'].split(',')} == {'POST', 'GET', 'OPTIONS'}
    assert 'content-type' in headers['access-control-allow-headers'].lower()
    assert 'access-control-allow-private-network' not in headers
    _, pna, _ = client.request('OPTIONS', '/ingest', headers={'Access-Control-Request-Private-Network': 'true'})
    assert pna['access-control-allow-private-network'] == 'true'


# ───────────────────────────── valid uploads ─────────────────────────────


def test_session_is_stored_as_the_record_not_the_envelope(client: Client):
    session = make_session(notes='Hebrew text: שלום')
    status, reply = client.post(envelope('session', session))
    assert status == 200
    assert set(reply) == {'ok', 'kind', 'id', 'stored_as', 'received_at', 'server_version'}
    assert reply['ok'] is True and reply['kind'] == 'session' and reply['id'] == SID
    assert reply['stored_as'] == f'{STUDY}/{PID}/{SID}/session.json'
    assert reply['server_version'] == receiver.SERVER_VERSION
    stored = json.loads((client.out / reply['stored_as']).read_text(encoding='utf-8'))
    assert stored == session


@pytest.mark.parametrize('path', ['/', '/ingest', '/ingest/', '/api/ingest', '/ingest?from=app'])
def test_ingest_paths_and_json_content_type(client: Client, path: str):
    status, reply = client.post(envelope('trial', make_trial()), path=path, content_type='application/json')
    assert status == 200, reply
    assert reply['stored_as'] == f'{STUDY}/{PID}/{SID}/trial_000_a1_trial-0001.json'


@pytest.mark.parametrize('index, attempt, name', [
    (0, 1, 'trial_000_a1_t-x.json'),
    (7, 2, 'trial_007_a2_t-x.json'),
    (123, 1, 'trial_123_a1_t-x.json'),
    (1234, 3, 'trial_1234_a3_t-x.json'),
    (7.0, 1.0, 'trial_007_a1_t-x.json'),  # JSON 7.0 is the number 7 for JavaScript too
])
def test_trial_file_names(client: Client, index: Any, attempt: Any, name: str):
    status, reply = client.post(envelope('trial', make_trial(trial_id='t-x', trial_index=index, attempt=attempt)))
    assert status == 200, reply
    assert reply['stored_as'] == f'{STUDY}/{PID}/{SID}/{name}'
    assert (client.out / reply['stored_as']).is_file()


def test_reposts_replace_the_file_atomically(client: Client):
    first = make_trial(outcome={'status': 'complete', 'accepted': True, 'participant_flag': None, 'note': 'first'})
    second = copy.deepcopy(first)
    second['outcome']['note'] = 'second'
    s1, r1 = client.post(envelope('trial', first))
    s2, r2 = client.post(envelope('trial', second))
    assert (s1, s2) == (200, 200) and r1['stored_as'] == r2['stored_as']
    assert client.data_files() == [r1['stored_as']]
    assert json.loads((client.out / r1['stored_as']).read_text(encoding='utf-8'))['outcome']['note'] == 'second'
    assert client.stray_temp_files() == []


def test_session_resend_and_redo_layout(client: Client):
    client.post(envelope('session', make_session(status='in_progress')))
    client.post(envelope('trial', make_trial(trial_id='tr-1', trial_index=4, attempt=1)))
    client.post(envelope('trial', make_trial(trial_id='tr-2', trial_index=4, attempt=2)))
    client.post(envelope('session', make_session(status='completed', ended_at=T0)))
    base = f'{STUDY}/{PID}/{SID}'
    assert client.data_files() == [f'{base}/session.json', f'{base}/trial_004_a1_tr-1.json', f'{base}/trial_004_a2_tr-2.json']
    assert json.loads((client.out / base / 'session.json').read_text(encoding='utf-8'))['status'] == 'completed'


def test_keep_alive_connection_serves_several_uploads(client: Client):
    conn = http.client.HTTPConnection('127.0.0.1', client.srv.server_address[1], timeout=10)
    try:
        for i in range(3):
            body = json.dumps(envelope('trial', make_trial(trial_id=f'k-{i}', trial_index=i)))
            conn.request('POST', '/ingest', body=body, headers={'Content-Type': 'application/json'})
            res = conn.getresponse()
            assert res.status == 200
            assert json.loads(res.read())['ok'] is True
    finally:
        conn.close()
    assert len(client.data_files()) == 3


def test_concurrent_uploads(client: Client):
    distinct = [envelope('trial', make_trial(trial_id=f'c-{i}', trial_index=i)) for i in range(24)]
    same = [envelope('trial', make_trial(trial_id='shared', trial_index=99, app_version=f'v{i}')) for i in range(12)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(client.post, distinct + same))
    assert all(status == 200 and reply['ok'] for status, reply in results)
    files = client.data_files()
    assert len(files) == 25
    shared = json.loads((client.out / STUDY / PID / SID / 'trial_099_a1_shared.json').read_text(encoding='utf-8'))
    assert shared['app_version'] in {f'v{i}' for i in range(12)}  # one complete version, never a mix
    assert client.stray_temp_files() == []


# ───────────────────────────── rejections and quarantine ─────────────────────────────

LONG = 'x' * 65


def _trial(**over: Any) -> str:
    return json.dumps(envelope('trial', make_trial(**over)))


INVALID = [
    ('malformed JSON', '{"kind": "trial", "record": {', 'malformed_json'),
    ('NaN token', _trial().replace('"trial_index": 0', '"trial_index": NaN'), 'malformed_json'),
    ('overflowing number', _trial().replace('"trial_index": 0', '"trial_index": 1e400'), 'malformed_json'),
    ('a JSON array', '[]', 'bad_envelope'),
    ('a JSON string', '"hello"', 'bad_envelope'),
    ('JSON null', 'null', 'bad_envelope'),
    ('unknown kind', json.dumps(envelope('participant', make_trial())), 'bad_kind'),
    ('no record', json.dumps({'kind': 'trial', 'study_key': None, 'sent_at': T0}), 'bad_record'),
    ('record is a list', json.dumps(envelope('trial', [])), 'bad_record'),
    ('kind/record_type mismatch', json.dumps(envelope('trial', make_session())), 'bad_record_type'),
    ('schema 2.0.0', _trial(schema_version='2.0.0'), 'bad_schema_version'),
    ('schema 10.1.0', _trial(schema_version='10.1.0'), 'bad_schema_version'),
    ('schema missing', _trial(schema_version=None), 'bad_schema_version'),
    ('study_code with a slash', _trial(study_code='a/b'), 'bad_study_code'),
    ('study_code ..', _trial(study_code='..'), 'bad_study_code'),
    ('study_code too long', _trial(study_code=LONG), 'bad_study_code'),
    ("study_code 'quarantine'", _trial(study_code='Quarantine'), 'bad_study_code'),
    ("study_code 'CON' (Windows device)", _trial(study_code='CON'), 'bad_study_code'),
    ('participant with 0', _trial(participant_id='P-ABC0EF'), 'bad_participant_id'),
    ('participant lower case', _trial(participant_id='p-abcdef'), 'bad_participant_id'),
    ('participant + newline', _trial(participant_id='P-ABCDEF\n'), 'bad_participant_id'),
    ('session_id + newline', _trial(session_id='sess\n'), 'bad_session_id'),
    ('session_id with a space', _trial(session_id='sess 1'), 'bad_session_id'),
    ("session_id 'nul' (Windows device)", _trial(session_id='nul'), 'bad_session_id'),
    ('trial_id with a dot', _trial(trial_id='../x'), 'bad_trial_id'),
    ('trial_index -1', _trial(trial_index=-1), 'bad_trial_index'),
    ('trial_index 1.5', _trial(trial_index=1.5), 'bad_trial_index'),
    ('trial_index true', _trial(trial_index=True), 'bad_trial_index'),
    ("trial_index '3'", _trial(trial_index='3'), 'bad_trial_index'),
    ('trial_index 2^53', _trial(trial_index=2**53), 'bad_trial_index'),
    ('attempt 0', _trial(attempt=0), 'bad_attempt'),
    ('attempt false', _trial(attempt=False), 'bad_attempt'),
]


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode('utf-8') if isinstance(data, str) else data).hexdigest()


@pytest.mark.parametrize('label, text, code', INVALID, ids=[c[0] for c in INVALID])
def test_invalid_envelopes_are_rejected_and_logged_without_the_body(client: Client, label: str, text: str, code: str):
    status, headers, raw = client.request('POST', '/ingest', text, {'Content-Type': 'application/json'})
    assert status == 400, raw
    assert headers['access-control-allow-origin'] == '*'
    reply = json.loads(raw)
    assert set(reply) == {'ok', 'error', 'code', 'retry', 'received_at', 'server_version'}
    assert reply['ok'] is False and reply['retry'] is False and reply['code'] == code
    assert isinstance(reply['error'], str) and len(reply['error']) > 5
    # no copy of the body anywhere: the log is the only file
    assert client.all_files() == [receiver.QUARANTINE_LOG]
    [line] = client.quarantine_log()
    assert line['received_at'] == reply['received_at']
    assert line['reason'] == code
    assert line['bytes'] == len(text.encode('utf-8'))
    assert line['sha256'] == sha256(text)


def test_the_log_holds_identifiers_only_like_the_apps_script_sheet(client: Client):
    profile = {'participant_id': PID, 'nickname': 'Dana Levi', 'age_years': 47, 'phone_model': 'Galaxy S24'}
    text = json.dumps(envelope('session', make_session(schema_version='2.0.0', participant=profile, notes='private remark')))
    client.post(text)
    [line] = client.quarantine_log()
    assert line == {
        'received_at': line['received_at'], 'reason': 'bad_schema_version', 'bytes': len(text), 'sha256': sha256(text),
        'kind': 'session', 'record_type': 'session', 'study_code': STUDY, 'participant_id': PID,
        'session_id': SID, 'record_id': SID, 'schema_version': '2.0.0', 'app_version': '0.1.0',
    }
    assert list(line) == ['received_at', 'reason', 'bytes', 'sha256', 'kind', 'record_type', 'study_code',
                          'participant_id', 'session_id', 'record_id', 'schema_version', 'app_version']
    log = (client.out / receiver.QUARANTINE_LOG).read_text(encoding='utf-8')
    for secret in ('Dana Levi', 'Galaxy S24', 'private remark', 'age_years', '{"kind"'):
        assert secret not in log, secret
    assert 47 not in line.values()


def test_log_identifiers_are_best_effort(client: Client):
    client.post(json.dumps(envelope(['trial'], make_trial(study_code='x' * 500, participant_id={'name': 'Dana'},
                                                          app_version=7, session_id='a\nb c'))))
    client.post('{"kind": "trial", "record": {"participant": {"nickname": "Dana"')  # not JSON at all
    first, second = client.quarantine_log()
    assert first['reason'] == 'bad_kind'
    assert first['kind'] is None and first['participant_id'] is None
    assert first['study_code'] == 'x' * 100 + '…'
    assert first['app_version'] == '7'
    assert first['session_id'] == 'a\nb c' and first['record_id'] == 'trial-0001'
    assert second['reason'] == 'malformed_json'
    assert {k: v for k, v in second.items() if v is not None} == {
        'received_at': second['received_at'], 'reason': 'malformed_json', 'bytes': second['bytes'], 'sha256': second['sha256']}
    # one physical line per rejection, whatever the identifiers contain
    assert (client.out / receiver.QUARANTINE_LOG).read_bytes().count(b'\n') == 2


def test_at_most_a_cap_of_lines_per_utc_day(client: Client, monkeypatch):
    client.srv.quarantine_cap = 3
    monkeypatch.setattr(receiver, 'now_iso', lambda: '2026-11-20T22:30:00.000Z')
    for i in range(5):
        status, reply = client.post(json.dumps(envelope('spam', make_trial(trial_id=f't-{i}'))))
        assert (status, reply['code'], reply['retry']) == (400, 'bad_kind', False)  # the answer never changes
    assert [line['record_id'] for line in client.quarantine_log()] == ['t-0', 't-1', 't-2']
    monkeypatch.setattr(receiver, 'now_iso', lambda: '2026-11-21T00:00:00.000Z')  # a new UTC day
    client.post(json.dumps(envelope('spam', make_trial(trial_id='t-5'))))
    assert [line['record_id'] for line in client.quarantine_log()] == ['t-0', 't-1', 't-2', 't-5']
    assert client.all_files() == [receiver.QUARANTINE_LOG]


def test_the_cap_holds_under_concurrent_rejections(client: Client):
    client.srv.quarantine_cap = 10
    bodies = [json.dumps(envelope('spam', make_trial(trial_id=f'c-{i}'))) for i in range(40)]
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(client.post, bodies))
    assert all(status == 400 and reply['code'] == 'bad_kind' for status, reply in results)
    lines = client.quarantine_log()
    assert len(lines) == 10 and len({line['record_id'] for line in lines}) == 10


def test_a_log_failure_is_still_a_plain_rejection(client: Client):
    (client.out / receiver.QUARANTINE_LOG).mkdir(parents=True)  # a folder where the log should be
    status, reply = client.post('{')
    assert (status, reply['code'], reply['retry']) == (400, 'malformed_json', False)


def test_invalid_utf8_and_bom_are_malformed(client: Client):
    for raw in (b'\xff\xfe{"kind":"trial"}', '﻿'.encode('utf-8') + _trial().encode('utf-8')):
        status, _, body = client.request('POST', '/', raw, {'Content-Type': 'text/plain'})
        assert status == 400
        assert json.loads(body)['code'] == 'malformed_json'


def test_messages_match_the_apps_script_backend(client: Client):
    _, reply = client.post(_trial(participant_id='bad'))
    assert reply['error'] == 'participant_id must match /^P-[A-HJ-NP-Z2-9]{6}$/'
    _, reply = client.post(json.dumps(envelope('nope', make_trial())))
    assert reply['error'] == "kind must be 'session' or 'trial'"


def test_study_key(start):
    keyed = start(key='k-123')
    assert keyed.post(envelope('trial', make_trial(), None))[1]['code'] == 'bad_study_key'
    assert keyed.post(envelope('trial', make_trial(), 'k-999'))[1]['code'] == 'bad_study_key'
    assert keyed.post(envelope('trial', make_trial(), 7))[1]['code'] == 'bad_study_key'
    status, reply = keyed.post(envelope('trial', make_trial(), 'k-123'))
    assert status == 200 and reply['ok'] is True
    open_ = start()
    assert open_.post(envelope('trial', make_trial(), None))[0] == 200
    assert open_.post(envelope('trial', make_trial(trial_id='t2'), 'anything'))[0] == 200


def test_size_limit_is_checked_before_parsing(start):
    small = start(max_bytes=5000)
    base = len(json.dumps(envelope('session', make_session(notes=''))))
    ok_body = json.dumps(envelope('session', make_session(notes='n' * (5000 - base))))
    assert len(ok_body) == 5000
    assert small.post(ok_body)[0] == 200
    big = json.dumps(envelope('session', make_session(notes='n' * (5001 - base))))
    status, reply = small.post(big)
    assert status == 413 and reply['code'] == 'too_large' and reply['retry'] is False
    assert '5001' in reply['error']
    # never parsed: size and hash only
    [line] = small.quarantine_log()
    assert (line['reason'], line['bytes'], line['sha256'], line['kind'], line['participant_id']) == ('too_large', 5001, sha256(big), None, None)
    status, reply = small.post('{' * 6000)  # garbage, but too large first
    assert (status, reply['code']) == (413, 'too_large')
    # only the accepted record and the log: nothing of the two rejected bodies
    assert small.all_files() == [f'{STUDY}/{PID}/{SID}/session.json', receiver.QUARANTINE_LOG]


def test_default_limit_and_an_oversized_body_is_hashed_not_kept(client: Client):
    text = json.dumps(envelope('session', make_session(notes='n' * 2_100_000)))
    status, reply = client.post(text)
    assert status == 413 and reply['code'] == 'too_large'
    assert client.all_files() == [receiver.QUARANTINE_LOG]
    [line] = client.quarantine_log()
    assert (line['bytes'], line['sha256']) == (len(text), sha256(text))
    assert (client.out / receiver.QUARANTINE_LOG).stat().st_size < 1000
    assert client.post(json.dumps(envelope('trial', make_trial())))[0] == 200  # same connection pool, still fine


def test_a_rejected_body_is_never_written_to_disk(client: Client):
    text = '{"kind": "trial", "padding": "' + 'p' * 300_000  # malformed (unterminated), 300 KB < 2 MB
    status, reply = client.post(text)
    assert (status, reply['code']) == (400, 'malformed_json')
    assert client.all_files() == [receiver.QUARANTINE_LOG]
    assert b'ppp' not in (client.out / receiver.QUARANTINE_LOG).read_bytes()


def test_huge_body_is_not_read_into_memory(start):
    small = start(max_bytes=1000)
    sock = socket.create_connection(('127.0.0.1', small.srv.server_address[1]), timeout=10)
    try:
        # announce 100 MB (far over 8 x the limit) and send none of it: the receiver answers at once
        # and closes the connection without reading anything
        sock.sendall(b'POST /ingest HTTP/1.1\r\nHost: x\r\nContent-Type: text/plain\r\n'
                     b'Content-Length: 100000000\r\n\r\n')
        data = b''
        while chunk := sock.recv(65536):
            data += chunk
        assert data.startswith(b'HTTP/1.1 413')
        assert b'Connection: close' in data
    finally:
        sock.close()
    [line] = small.quarantine_log()
    assert (line['reason'], line['bytes'], line['sha256']) == ('too_large', 100_000_000, None)  # never read, never hashed


def test_empty_body_is_rejected_and_logged_with_0_bytes(client: Client):
    status, _, raw = client.request('POST', '/', b'', {'Content-Type': 'text/plain'})
    assert status == 400 and json.loads(raw)['code'] == 'empty_body'
    [line] = client.quarantine_log()
    assert (line['reason'], line['bytes'], line['sha256']) == ('empty_body', 0, sha256(b''))


# ───────────────────────────── wrong requests and server errors ─────────────────────────────


def test_unknown_paths_and_methods_answer_json(client: Client):
    for method, path, status_expected, code in [
        ('POST', '/nope', 404, 'not_found'),
        ('GET', '/nope', 404, 'not_found'),
        ('PUT', '/', 405, 'method_not_allowed'),
        ('DELETE', '/ingest', 405, 'method_not_allowed'),
        ('FOO', '/', 501, 'not_implemented'),
    ]:
        status, headers, raw = client.request(method, path, b'{}' if method in ('POST', 'PUT') else None)
        assert status == status_expected, (method, path)
        assert headers['access-control-allow-origin'] == '*'
        assert json.loads(raw)['code'] == code


def test_a_wrong_path_drains_the_body_and_keeps_the_connection(client: Client):
    conn = http.client.HTTPConnection('127.0.0.1', client.srv.server_address[1], timeout=10)
    try:
        conn.request('POST', '/wrong', body=json.dumps(envelope('trial', make_trial())), headers={'Content-Type': 'application/json'})
        res = conn.getresponse()
        assert res.status == 404 and json.loads(res.read())['code'] == 'not_found'
        assert res.getheader('Connection') != 'close'
        conn.request('POST', '/ingest', body=json.dumps(envelope('trial', make_trial())), headers={'Content-Type': 'application/json'})
        res = conn.getresponse()
        assert res.status == 200 and json.loads(res.read())['ok'] is True
    finally:
        conn.close()


def test_missing_content_length_is_refused(client: Client):
    sock = socket.create_connection(('127.0.0.1', client.srv.server_address[1]), timeout=10)
    try:
        sock.sendall(b'POST /ingest HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n')
        data = b''
        while chunk := sock.recv(65536):  # headers and body arrive separately; the server then closes
            data += chunk
        assert data.startswith(b'HTTP/1.1 411')
        assert b'Connection: close' in data
        assert b'"length_required"' in data
    finally:
        sock.close()


def test_storage_failure_is_a_json_500_with_retry(client: Client):
    client.out.mkdir(parents=True, exist_ok=True)
    (client.out / STUDY).write_text('a file where the study folder should be', encoding='utf-8')
    status, headers, raw = client.request('POST', '/', json.dumps(envelope('trial', make_trial())))
    assert status == 500
    assert headers['access-control-allow-origin'] == '*'
    reply = json.loads(raw)
    assert reply['ok'] is False and reply['retry'] is True and reply['error'].startswith('server error')


# ───────────────────────────── CLI ─────────────────────────────


def test_cli_defaults():
    args = receiver.build_parser().parse_args([])
    assert args.port == 8787
    assert args.host == '127.0.0.1'
    assert args.key is None
    assert args.max_bytes == 2_000_000
    assert args.out == Path(receiver.__file__).resolve().parent / 'received'
    args = receiver.build_parser().parse_args(['--port', '9000', '--key', 'k', '--out', 'x', '--quiet'])
    assert (args.port, args.key, args.out, args.quiet) == (9000, 'k', Path('x'), True)


def test_cli_help_lists_the_options(capsys):
    with pytest.raises(SystemExit) as e:
        receiver.main(['--help'])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for option in ('--port', '--key', '--out', '--host', '--max-bytes'):
        assert option in out


def test_cli_reports_a_busy_port(capsys):
    blocker = socket.socket()
    blocker.bind(('127.0.0.1', 0))
    blocker.listen(1)
    try:
        port = blocker.getsockname()[1]
        assert receiver.main(['--port', str(port), '--quiet']) == 1
        assert 'cannot listen' in capsys.readouterr().err
    finally:
        blocker.close()
