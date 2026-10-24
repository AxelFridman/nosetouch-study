"""GET /api/export (the app's export format) and GET /api/export.zip (the backend folder layout)."""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_EXPORT, REPO
from records import PID, SID, STUDY, envelope, make_session, make_trial, post, profile
from sqlalchemy import update
from sqlalchemy.exc import OperationalError

import import_export
from nosetouch_server.db import sessions, trials

ANALYSIS_DIR = REPO / 'analysis'
ANALYSIS_PYTHON = ANALYSIS_DIR / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _study(client: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = make_session()
    trial_list = [
        make_trial(trial_id='t-0', trial_index=0),
        make_trial(trial_id='t-1a', trial_index=1, outcome={'status': 'no_movement', 'accepted': False,
                                                            'participant_flag': None, 'note': None}),
        make_trial(trial_id='t-1b', trial_index=1, attempt=2),
        make_trial(trial_id='t-2', trial_index=2, condition={'hand': 'left', 'orientation': 'upright_away',
                                                            'speed': 'fast', 'variant': 'standard'}),
    ]
    assert post(client, envelope('session', session)).status_code == 200
    for t in reversed(trial_list):  # upload order does not matter: the export sorts by plan order
        assert post(client, envelope('trial', t)).status_code == 200
    return session, trial_list


def _bundle(client: Any, headers: dict[str, str], **params: str) -> dict[str, Any]:
    r = client.get('/api/export', params=params, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _zip(client: Any, headers: dict[str, str], **params: str) -> zipfile.ZipFile:
    r = client.get('/api/export.zip', params=params, headers=headers)
    assert r.status_code == 200, r.text
    assert r.headers['content-type'] == 'application/zip'
    assert r.headers['content-disposition'].startswith('attachment; filename="nosetouch_')
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_export_round_trip_equals_the_posted_records(client, admin_headers):
    session, trial_list = _study(client)
    bundle = _bundle(client, admin_headers)
    assert list(bundle) == ['format', 'schema_version', 'app_version', 'exported_at', 'participants', 'sessions', 'trials']
    assert bundle['format'] == 'nose-touch-export' and bundle['schema_version'] == '1.1.0'
    assert bundle['exported_at'].endswith('Z')
    assert bundle['participants'] == [session['participant']]
    assert bundle['sessions'] == [session]
    assert bundle['trials'] == trial_list  # plan order: trial_index, then attempt


def test_export_is_gzipped_when_the_client_accepts_it(client, admin_headers):
    _study(client)
    r = client.get('/api/export', headers={**admin_headers, 'Accept-Encoding': 'gzip'})
    assert r.headers['content-encoding'] == 'gzip' and r.json()['format'] == 'nose-touch-export'
    plain = client.get('/api/export', headers={**admin_headers, 'Accept-Encoding': 'identity'})
    assert 'content-encoding' not in plain.headers
    assert plain.json() == {**r.json(), 'exported_at': plain.json()['exported_at']}


def test_export_filters(client, admin_headers):
    _study(client)
    other = make_session(session_id='sess-sim', participant_id='P-ZZZZZZ', simulated=True,
                         participant=profile(participant_id='P-ZZZZZZ'))
    post(client, envelope('session', other))
    post(client, envelope('trial', make_trial(trial_id='sim-0', session_id='sess-sim', participant_id='P-ZZZZZZ',
                                              simulated=True)))
    everything = _bundle(client, admin_headers)
    assert [s['session_id'] for s in everything['sessions']] == [SID, 'sess-sim']
    real = _bundle(client, admin_headers, include_simulated='false')
    assert [s['session_id'] for s in real['sessions']] == [SID]
    assert [p['participant_id'] for p in real['participants']] == [PID]
    assert 'sim-0' not in [t['trial_id'] for t in real['trials']]
    one = _bundle(client, admin_headers, participant_id='P-ZZZZZZ')
    assert [t['trial_id'] for t in one['trials']] == ['sim-0']
    assert [p['participant_id'] for p in one['participants']] == ['P-ZZZZZZ']


def test_export_since(client, admin_headers):
    _study(client)
    db = client.app.state.db
    old = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    with db.write() as conn:  # everything so far arrived long ago, except trial t-2
        conn.execute(update(sessions).values(last_received_at=old))
        conn.execute(update(trials).where(trials.c.trial_id != 't-2').values(received_at=old))
    recent = _bundle(client, admin_headers, since='2026-06-01T00:00:00Z')
    assert [t['trial_id'] for t in recent['trials']] == ['t-2']
    assert [s['session_id'] for s in recent['sessions']] == [SID]  # the session of a new trial comes along
    assert [p['participant_id'] for p in recent['participants']] == [PID]
    with zipfile.ZipFile(io.BytesIO(client.get('/api/export.zip', params={'since': '2026-06-01T00:00:00+00:00'},
                                               headers=admin_headers).content)) as zf:
        assert sorted(zf.namelist()) == [f'{STUDY}/{PID}/{SID}/session.json', f'{STUDY}/{PID}/{SID}/trial_002_a1_t-2.json']
    assert _bundle(client, admin_headers, since='2030-01-01')['trials'] == []


@pytest.mark.parametrize('params', [{'participant_id': 'nobody'}, {'since': 'yesterday'},
                                    {'include_simulated': 'maybe'}])
def test_bad_filters_are_400(client, admin_headers, params):
    for path in ('/api/export', '/api/export.zip'):
        r = client.get(path, params=params, headers=admin_headers)
        assert r.status_code == 400 and r.json()['code'] == 'bad_filter'


def test_export_zip_holds_the_backend_layout(client, admin_headers):
    session, trial_list = _study(client)
    with _zip(client, admin_headers) as zf:
        assert zf.testzip() is None
        base = f'{STUDY}/{PID}/{SID}'
        assert sorted(zf.namelist()) == sorted([
            f'{base}/session.json', f'{base}/trial_000_a1_t-0.json', f'{base}/trial_001_a1_t-1a.json',
            f'{base}/trial_001_a2_t-1b.json', f'{base}/trial_002_a1_t-2.json'])
        assert json.loads(zf.read(f'{base}/session.json')) == session
        by_id = {t['trial_id']: t for t in trial_list}
        for name in zf.namelist():
            if '/trial_' in name:
                rec = json.loads(zf.read(name))
                assert rec == by_id[rec['trial_id']]


def test_exports_answer_503_when_the_database_is_down(client, admin_headers, monkeypatch):
    def down(*_a: Any, **_k: Any) -> Any:
        raise OperationalError('SELECT 1', {}, ConnectionError('server closed the connection'))

    monkeypatch.setattr(client.app.state.db, 'read', down)
    for path in ('/api/export', '/api/export.zip'):
        r = client.get(path, headers=admin_headers)
        assert r.status_code == 503 and r.json()['retry'] is True, path


def test_empty_exports_are_valid(client, admin_headers):
    bundle = _bundle(client, admin_headers)
    assert (bundle['participants'], bundle['sessions'], bundle['trials']) == ([], [], [])
    with _zip(client, admin_headers) as zf:
        assert zf.namelist() == []


# ───────────────────────────── the analysis kit reads the exports unchanged ─────────────────────────────

_MOTIONLAB_LOAD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from motionlab.io import load_dataset
out = {}
for name in sys.argv[2:]:
    ds = load_dataset(name)
    issues = ds.issues
    out[name] = {
        'participants': sorted(ds.participants['participant_id']) if len(ds.participants) else [],
        'sessions': sorted(ds.sessions['session_id']) if len(ds.sessions) else [],
        'trials': sorted(ds.trial_ids),
        'errors': [str(m) for m in issues.loc[issues['level'] == 'error', 'message']] if len(issues) else [],
    }
print(json.dumps(out))
'''

_SYNTH = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
from motionlab import synth
ds = synth.make_dataset(2, seed=5, preset='tiny', failure_rate=0.1)
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    json.dump(synth.export_bundle(ds), f)
'''


def _analysis_python() -> list[str]:
    try:
        import jsonschema  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        return [sys.executable]
    except ImportError:
        pass
    if ANALYSIS_PYTHON.exists():
        return [str(ANALYSIS_PYTHON)]
    pytest.skip('the analysis kit needs numpy/pandas/jsonschema: create analysis/.venv '
                '(pip install -r analysis/requirements.txt) to run this check')


def _run(code: str, *args: str, cwd: Path) -> str:
    proc = subprocess.run([*_analysis_python(), '-c', code, str(ANALYSIS_DIR), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def _client_poster(client: Any) -> import_export.Poster:
    def post_body(body: bytes) -> tuple[int, dict[str, str], bytes]:
        r = client.post('/api/ingest', content=body, headers={'Content-Type': 'application/json'})
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content
    return post_body


@pytest.mark.parametrize('source', ['synthetic', 'real'])
def test_the_exports_load_with_motionlab(client, admin_headers, tmp_path, source):
    if source == 'synthetic':
        bundle_path = tmp_path / 'synthetic_bundle.json'
        _run(_SYNTH, str(bundle_path), cwd=tmp_path)
    else:
        if not REAL_EXPORT.exists():
            pytest.skip(f'{REAL_EXPORT.name} is not on this machine')
        bundle_path = REAL_EXPORT
    bundle = import_export.load_bundle(bundle_path)
    summary = import_export.run_import(bundle, _client_poster(client), study_key=None, sleep=lambda s: None)
    assert summary.ok, [r for r in summary.results if r.outcome != 'stored']
    zip_path = tmp_path / 'export.zip'
    zip_path.write_bytes(client.get('/api/export.zip', headers=admin_headers).content)
    json_path = tmp_path / 'export.json'
    json_path.write_bytes(client.get('/api/export', headers=admin_headers).content)
    loaded = json.loads(_run(_MOTIONLAB_LOAD, str(zip_path), str(json_path), cwd=tmp_path))
    expected = {
        'participants': sorted({s['participant_id'] for s in bundle['sessions']}),
        'sessions': sorted(s['session_id'] for s in bundle['sessions']),
        'trials': sorted(t['trial_id'] for t in bundle['trials']),
        'errors': [],
    }
    assert loaded[str(zip_path)] == expected
    assert loaded[str(json_path)] == expected
