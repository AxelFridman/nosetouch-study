"""CORS: what a browser on the GitHub Pages origin needs for the app's json_post uploads, and no more."""
from __future__ import annotations

from conftest import PAGES_ORIGIN
from records import envelope, make_trial, post

PREFLIGHT = {'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'}


def test_preflight_from_the_pages_origin(client):
    r = client.options('/api/ingest', headers={'Origin': PAGES_ORIGIN, **PREFLIGHT})
    assert r.status_code == 204 and r.content == b''
    h = r.headers
    assert h['access-control-allow-origin'] == PAGES_ORIGIN
    assert {m.strip() for m in h['access-control-allow-methods'].split(',')} == {'GET', 'POST'}
    assert h['access-control-allow-headers'].lower() == 'content-type'
    assert h['access-control-max-age'] == '86400'
    assert 'access-control-allow-credentials' not in h
    assert 'origin' in h['vary'].lower()


def test_the_actual_post_and_probe_carry_allow_origin(client):
    r = post(client, envelope('trial', make_trial()), headers={'Origin': PAGES_ORIGIN})
    assert r.status_code == 200
    assert r.headers['access-control-allow-origin'] == PAGES_ORIGIN
    assert 'origin' in r.headers['vary'].lower()
    assert 'access-control-allow-credentials' not in r.headers
    probe = client.get('/api/ingest', headers={'Origin': PAGES_ORIGIN})
    assert probe.headers['access-control-allow-origin'] == PAGES_ORIGIN
    # a refusal must be readable too: the app shows the error and stops resending that record
    refused = post(client, '{', headers={'Origin': PAGES_ORIGIN})
    assert refused.status_code == 400 and refused.headers['access-control-allow-origin'] == PAGES_ORIGIN


def test_local_dev_origins_are_allowed_by_default(client):
    for origin in ('http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:4173'):
        r = client.options('/api/ingest', headers={'Origin': origin, **PREFLIGHT})
        assert r.status_code == 204 and r.headers['access-control-allow-origin'] == origin


def test_a_disallowed_origin_gets_no_cors_headers(client):
    evil = 'https://evil.example'
    r = client.options('/api/ingest', headers={'Origin': evil, **PREFLIGHT})
    assert r.status_code == 403
    assert 'access-control-allow-origin' not in r.headers
    r = post(client, envelope('trial', make_trial()), headers={'Origin': evil})
    assert r.status_code == 403 and 'access-control-allow-origin' not in r.headers
    # look-alikes are not the Pages origin
    for origin in ('https://axelfridman.github.io.evil.example', 'http://axelfridman.github.io', 'null'):
        r = client.options('/api/ingest', headers={'Origin': origin, **PREFLIGHT})
        assert r.status_code == 403, origin


def test_allowed_origins_from_settings(make_client):
    custom = make_client(allowed_origins=('https://lab.example.org',))
    assert custom.options('/api/ingest', headers={'Origin': 'https://lab.example.org', **PREFLIGHT}).status_code == 204
    assert custom.options('/api/ingest', headers={'Origin': PAGES_ORIGIN, **PREFLIGHT}).status_code == 403
    anyone = make_client(allowed_origins=('*',))
    r = anyone.options('/api/ingest', headers={'Origin': 'https://whatever.example', **PREFLIGHT})
    assert r.status_code == 204 and r.headers['access-control-allow-origin'] == '*'


def test_admin_routes_never_answer_cross_origin(client, admin_headers):
    for path in ('/api/export', '/api/export.zip', '/admin'):
        r = client.get(path, headers={'Origin': PAGES_ORIGIN, **admin_headers})
        assert r.status_code == 200
        assert 'access-control-allow-origin' not in r.headers, path
    r = client.options('/api/participants/P-ABCDEF', headers={'Origin': PAGES_ORIGIN,
                                                              'Access-Control-Request-Method': 'DELETE'})
    assert 'access-control-allow-origin' not in r.headers


def test_options_without_preflight_headers_is_harmless(client):
    r = client.options('/api/ingest')
    assert r.status_code == 204 and 'access-control-allow-origin' not in r.headers
