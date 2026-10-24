"""scripts/pull_live.py: download and unpack the live export, with a mocked HTTP layer (no network)."""
from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest
from _mlhelpers import ANALYSIS_DIR

from motionlab import synth
from motionlab.io import load_dataset

_spec = importlib.util.spec_from_file_location('pull_live', ANALYSIS_DIR / 'scripts' / 'pull_live.py')
pull_live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pull_live)

BASE = 'https://nosetouch-test.herokuapp.com'
TOKEN = 'tok-123'


def layout_zip(sessions: list[dict[str, Any]], trials: list[dict[str, Any]], extra: dict[str, bytes] | None = None) -> bytes:
    """The service's export.zip: <study>/<participant>/<session>/session.json + trial_*.json."""
    buf = io.BytesIO()
    by_session = {s['session_id']: s for s in sessions}
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for s in sessions:
            zf.writestr(f"{s['study_code']}/{s['participant_id']}/{s['session_id']}/session.json", json.dumps(s))
        for t in trials:
            s = by_session[t['session_id']]
            zf.writestr(f"{s['study_code']}/{s['participant_id']}/{s['session_id']}/{synth.trial_file_name(t)}",
                        json.dumps(t))
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.status = 200

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeServer:
    """Stands in for urllib.request.urlopen: records the requests, answers with a zip or an HTTP error."""

    def __init__(self, data: bytes = b'', status: int = 200):
        self.data = data
        self.status = status
        self.requests: list[Any] = []

    def __call__(self, req: Any, timeout: float | None = None) -> FakeResponse:
        self.requests.append(req)
        if self.status != 200:
            raise urllib.error.HTTPError(req.full_url, self.status, 'error', {}, io.BytesIO(b'{"ok":false}'))
        return FakeResponse(self.data)


@pytest.fixture(scope='module')
def dataset() -> synth.SynthDataset:
    return synth.make_dataset(2, seed=3, preset='tiny')


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv('NOSETOUCH_ADMIN_TOKEN', raising=False)
    monkeypatch.delenv('NOSETOUCH_APP_URL', raising=False)
    return tmp_path / 'absent.env'


def test_pull_unpacks_the_layout_and_motionlab_loads_it(dataset, tmp_path, no_env, capsys):
    server = FakeServer(layout_zip(dataset.sessions, dataset.trials))
    out = tmp_path / 'live'
    code = pull_live.main(['--url', BASE + '/api/ingest', '--token', TOKEN, '--out', str(out),
                           '--env-file', str(no_env)], urlopen=server)
    assert code == 0
    [req] = server.requests
    assert req.full_url == BASE + '/api/export.zip'
    assert req.get_header('Authorization') == f'Bearer {TOKEN}'
    assert (out / pull_live.MARKER).exists()
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob('*.json'))
    assert len(files) == len(dataset.sessions) + len(dataset.trials)
    printed = capsys.readouterr().out
    assert f'2 participants, {len(dataset.sessions)} sessions, {len(dataset.trials)} trials' in printed
    assert TOKEN not in printed
    ds = load_dataset(out)
    assert sorted(ds.trial_ids) == sorted(t['trial_id'] for t in dataset.trials)
    assert len(ds.sessions) == len(dataset.sessions)
    assert not (ds.issues['level'] == 'error').any()


def test_a_new_pull_replaces_the_previous_one(dataset, tmp_path, no_env):
    out = tmp_path / 'live'
    first = FakeServer(layout_zip(dataset.sessions, dataset.trials))
    assert pull_live.main(['--url', BASE, '--token', TOKEN, '--out', str(out), '--env-file', str(no_env)], urlopen=first) == 0
    keep = dataset.sessions[0]
    trials = [t for t in dataset.trials if t['session_id'] == keep['session_id']]
    second = FakeServer(layout_zip([keep], trials))  # the other participant was deleted on the server
    assert pull_live.main(['--url', BASE, '--token', TOKEN, '--out', str(out), '--env-file', str(no_env)], urlopen=second) == 0
    assert sorted(p.name for p in (out / keep['study_code']).iterdir()) == [keep['participant_id']]
    assert not list(tmp_path.glob('.live.*'))  # no staging or old folders left behind


def test_filters_become_query_parameters(dataset, tmp_path, no_env):
    server = FakeServer(layout_zip(dataset.sessions[:1], []))
    pid = dataset.sessions[0]['participant_id']
    assert pull_live.main(['--url', BASE + '/', '--token', TOKEN, '--out', str(tmp_path / 'p'), '--env-file', str(no_env),
                           '--participant', pid, '--since', '2026-11-01T00:00:00Z', '--real-only'], urlopen=server) == 0
    assert server.requests[0].full_url == (f'{BASE}/api/export.zip?participant_id={pid}'
                                           '&since=2026-11-01T00%3A00%3A00Z&include_simulated=false')


def test_credentials_from_the_environment_and_the_env_file(dataset, tmp_path, monkeypatch, no_env):
    env_file = tmp_path / 'heroku.env'
    env_file.write_text(f'# NoseTouch\nAPP_URL={BASE}\nADMIN_TOKEN="from-file"\nSTUDY_KEY=k\n', encoding='utf-8')
    server = FakeServer(layout_zip(dataset.sessions[:1], []))
    assert pull_live.main(['--out', str(tmp_path / 'a'), '--env-file', str(env_file)], urlopen=server) == 0
    assert server.requests[-1].get_header('Authorization') == 'Bearer from-file'
    monkeypatch.setenv('NOSETOUCH_ADMIN_TOKEN', 'from-env')
    assert pull_live.main(['--out', str(tmp_path / 'b'), '--env-file', str(env_file)], urlopen=server) == 0
    assert server.requests[-1].get_header('Authorization') == 'Bearer from-env'
    assert pull_live.main(['--out', str(tmp_path / 'c'), '--env-file', str(env_file), '--token', 'from-arg'],
                          urlopen=server) == 0
    assert server.requests[-1].get_header('Authorization') == 'Bearer from-arg'


def test_missing_credentials_and_plain_http_are_usage_errors(tmp_path, no_env, capsys):
    assert pull_live.main(['--url', BASE, '--out', str(tmp_path / 'x'), '--env-file', str(no_env)]) == 2
    assert 'no admin token' in capsys.readouterr().err
    assert pull_live.main(['--url', 'http://example.org', '--token', TOKEN, '--out', str(tmp_path / 'x'),
                           '--env-file', str(no_env)]) == 2


@pytest.mark.parametrize('status, words', [(401, 'admin token'), (403, 'rejected'), (503, 'ADMIN_TOKEN unset')])
def test_http_errors_are_explained(tmp_path, no_env, capsys, status, words):
    out = tmp_path / 'live'
    code = pull_live.main(['--url', BASE, '--token', TOKEN, '--out', str(out), '--env-file', str(no_env)],
                          urlopen=FakeServer(status=status))
    assert code == 1
    err = capsys.readouterr().err
    assert f'HTTP {status}' in err and words in err and TOKEN not in err
    assert not out.exists()


@pytest.mark.parametrize('name', ['../evil.json', '/abs/evil.json', 'C:/evil.json', 'study/P-ABCDEF/s/session.json:ads',
                                  'study/P-ABCDEF/s/../../evil.json', 'study/P-ABCDEF/s/run.py'])
def test_unexpected_or_unsafe_names_are_refused(dataset, tmp_path, no_env, capsys, name):
    out = tmp_path / 'live'
    out.mkdir()
    (out / pull_live.MARKER).write_text('source=old\n', encoding='utf-8')
    (out / 'previous.json').write_text('{}', encoding='utf-8')
    server = FakeServer(layout_zip(dataset.sessions[:1], [], extra={name: b'{}'}))
    assert pull_live.main(['--url', BASE, '--token', TOKEN, '--out', str(out), '--env-file', str(no_env)], urlopen=server) == 1
    assert 'unexpected file' in capsys.readouterr().err
    assert (out / 'previous.json').exists()  # the previous pull is untouched
    assert not (tmp_path / 'evil.json').exists()


def test_a_folder_not_made_by_pull_live_is_not_replaced(dataset, tmp_path, no_env, capsys):
    out = tmp_path / 'mine'
    out.mkdir()
    (out / 'notes.txt').write_text('my notes', encoding='utf-8')
    server = FakeServer(layout_zip(dataset.sessions[:1], []))
    args = ['--url', BASE, '--token', TOKEN, '--out', str(out), '--env-file', str(no_env)]
    assert pull_live.main(args, urlopen=server) == 1
    assert 'was not created by pull_live.py' in capsys.readouterr().err
    assert (out / 'notes.txt').exists()
    assert pull_live.main([*args, '--force'], urlopen=server) == 0
    assert not (out / 'notes.txt').exists() and (out / pull_live.MARKER).exists()


def test_not_a_zip(tmp_path, no_env, capsys):
    server = FakeServer(b'<html>login</html>')
    assert pull_live.main(['--url', BASE, '--token', TOKEN, '--out', str(tmp_path / 'live'), '--env-file', str(no_env)],
                          urlopen=server) == 1
    assert 'not a zip' in capsys.readouterr().err


def test_keep_zip(dataset, tmp_path, no_env):
    keep = tmp_path / 'snapshots' / 'export.zip'
    server = FakeServer(layout_zip(dataset.sessions[:1], []))
    assert pull_live.main(['--url', BASE, '--token', TOKEN, '--out', str(tmp_path / 'live'), '--env-file', str(no_env),
                           '--keep-zip', str(keep)], urlopen=server) == 0
    assert zipfile.is_zipfile(keep)
