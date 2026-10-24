"""Environment variables and DATABASE_URL normalisation."""
from __future__ import annotations

import json
import logging

import pytest
from conftest import ADMIN_TOKEN
from fastapi.testclient import TestClient
from records import envelope, make_session, make_trial, post

from nosetouch_server import main as main_module
from nosetouch_server.main import create_app
from nosetouch_server.settings import DEFAULT_ALLOWED_ORIGINS, Settings, normalize_database_url, safe_url


@pytest.mark.parametrize('raw, expected', [
    ('postgres://u:p@ec2-1-2-3-4.compute-1.amazonaws.com:5432/d8abc',
     'postgresql+psycopg://u:p@ec2-1-2-3-4.compute-1.amazonaws.com:5432/d8abc?sslmode=require'),
    ('postgresql://u:p@db.example.org/x', 'postgresql+psycopg://u:p@db.example.org/x?sslmode=require'),
    ('postgres://u:p@db.example.org/x?sslmode=verify-full',
     'postgresql+psycopg://u:p@db.example.org/x?sslmode=verify-full'),
    ('postgresql+psycopg2://u:p@localhost:5432/x', 'postgresql+psycopg://u:p@localhost:5432/x'),
    ('postgres://u:p@127.0.0.1:55432/nosetouch_test', 'postgresql+psycopg://u:p@127.0.0.1:55432/nosetouch_test'),
    ('postgres://u:p%40ss@db.example.org/x', 'postgresql+psycopg://u:p%40ss@db.example.org/x?sslmode=require'),
    ('sqlite:///local.db', 'sqlite:///local.db'),
])
def test_database_url_normalisation(raw, expected):
    assert normalize_database_url(raw) == expected


def test_without_database_url_sqlite_is_used(tmp_path):
    assert normalize_database_url(None) == 'sqlite+pysqlite://'
    assert normalize_database_url('', str(tmp_path / 'n.db')) == 'sqlite+pysqlite:///' + (tmp_path / 'n.db').as_posix()
    with pytest.raises(ValueError):
        normalize_database_url('not a url')


def test_passwords_never_reach_log_lines():
    url = normalize_database_url('postgres://user:s3cret@db.example.org:5432/d')
    assert 's3cret' not in safe_url(url) and 'db.example.org' in safe_url(url)


def test_from_env():
    s = Settings.from_env({
        'DATABASE_URL': 'postgres://u:p@db.example.org/x', 'STUDY_KEY': ' k-1 ', 'ADMIN_TOKEN': 't',
        'ALLOWED_ORIGINS': 'https://A.example/ , http://localhost:5173,,https://a.example', 'MAX_BYTES': '123',
        'DYNO': 'web.1',
    })
    assert s.database_url.startswith('postgresql+psycopg://') and s.study_key == 'k-1' and s.admin_token == 't'
    assert s.allowed_origins == ('https://a.example', 'http://localhost:5173')
    assert s.origin_allowed('https://a.example') and not s.origin_allowed('https://b.example')
    assert s.max_bytes == 123 and s.force_https is True
    d = Settings.from_env({'MAX_BYTES': 'lots', 'STUDY_KEY': '  '})
    assert d.allowed_origins == DEFAULT_ALLOWED_ORIGINS and d.max_bytes == 4_000_000
    assert d.study_key is None and d.admin_token is None and d.force_https is False
    assert any('MAX_BYTES' in n for n in d.notes)
    assert Settings.from_env({'DYNO': 'web.1', 'FORCE_HTTPS': 'false'}).force_https is False


@pytest.mark.parametrize('env, missing', [
    ({'DYNO': 'web.1'}, True),  # deployed before the Postgres add-on was attached
    ({'DYNO': 'web.1', 'DATABASE_URL': '  '}, True),
    ({'DYNO': 'web.1', 'SQLITE_PATH': '/app/n.db'}, True),  # a dyno's disk is wiped at every restart
    ({'DYNO': 'web.1', 'DATABASE_URL': 'sqlite:///n.db'}, True),
    ({'DYNO': 'web.1', 'HEROKU_POSTGRESQL_RED_URL': 'postgres://u:p@db.example.org/x'}, True),  # not promoted
    ({'DYNO': 'web.1', 'DATABASE_URL': 'postgres://u:p@db.example.org/x'}, False),
    ({'DYNO': 'web.1', 'ALLOW_EPHEMERAL_DB': 'true'}, False),  # a throwaway test app, on purpose
    ({}, False),  # local runs: SQLite in memory is fine
    ({'SQLITE_PATH': 'n.db'}, False),
])
def test_a_dyno_needs_a_postgres_database(env, missing):
    assert Settings.from_env(env).database_missing is missing


def test_the_startup_notes_name_an_unpromoted_postgres_database():
    s = Settings.from_env({'DYNO': 'web.1', 'HEROKU_POSTGRESQL_RED_URL': 'postgres://u:s3cret@db.example.org/x'})
    note = ' '.join(s.notes)
    assert 'HEROKU_POSTGRESQL_RED_URL' in note and 'pg:promote' in note
    assert 's3cret' not in note and 'db.example.org' not in note  # names only, never values


def test_a_dyno_without_a_database_refuses_uploads_instead_of_losing_them(monkeypatch):
    """Before: in-memory SQLite answered 200 ok, the phones marked the records sent, and the next dyno
    restart (at least daily) deleted them. Now every upload answers 503 retry and the phones keep them."""
    sleeps: list[float] = []
    lines: list[str] = []

    async def no_sleep(s: float) -> None:
        sleeps.append(s)

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    monkeypatch.setattr(main_module, '_sleep', no_sleep)
    grab = Grab()
    logging.getLogger('nosetouch').addHandler(grab)
    try:
        settings = Settings.from_env({'DYNO': 'web.1', 'STUDY_KEY': 'k', 'ADMIN_TOKEN': ADMIN_TOKEN})
        with TestClient(create_app(settings), base_url='https://testserver') as c:
            assert sleeps == []  # no pointless retries before the port is bound
            for kind, record in (('session', make_session()), ('trial', make_trial())):
                r = post(c, envelope(kind, record, 'k'))
                assert (r.status_code, r.json()['ok'], r.json()['retry']) == (503, False, True), r.text
                assert 'DATABASE_URL' in r.json()['error'] and 'x-nosetouch-outcome' not in r.headers
            assert post(c, envelope('trial', make_trial(), 'wrong')).json()['code'] == 'bad_study_key'
            probe = c.get('/api/ingest')
            assert probe.status_code == 503 and 'DATABASE_URL' in probe.json()['error']
            health = c.get('/api/health')
            assert health.status_code == 503 and 'DATABASE_URL' in health.json()['db']['error']
            auth = {'Authorization': f'Bearer {ADMIN_TOKEN}'}
            page = c.get('/admin', headers=auth)
            assert page.status_code == 503 and 'DATABASE_URL' in page.text
            assert c.get('/api/export.zip', headers=auth).status_code == 503
            assert c.app.state.db.ready is False
    finally:
        logging.getLogger('nosetouch').removeHandler(grab)
    events = [json.loads(line) for line in lines if line.startswith('{')]
    assert any(e['event'] == 'database_missing' for e in events)
    assert not any(e['event'] in ('rejection_log_failed', 'db_init_failed') for e in events)


def test_allow_ephemeral_db_keeps_the_throwaway_database():
    settings = Settings.from_env({'DYNO': 'web.1', 'ALLOW_EPHEMERAL_DB': '1'})
    with TestClient(create_app(settings), base_url='https://testserver') as c:
        assert post(c, envelope('trial', make_trial())).status_code == 200


def test_https_is_enforced_on_heroku():
    with TestClient(create_app(Settings(force_https=True))) as c:
        r = c.get('/api/health?x=1', follow_redirects=False)
        assert r.status_code == 308 and r.headers['location'] == 'https://testserver/api/health?x=1'
        r = c.post('/api/ingest', content=b'{}', headers={'Content-Type': 'application/json'})
        assert r.status_code == 403 and r.json()['code'] == 'https_required'
    with TestClient(create_app(Settings(force_https=True)), base_url='https://testserver') as c:
        r = c.get('/api/health')
        assert r.status_code == 200 and r.headers['strict-transport-security'].startswith('max-age=')
