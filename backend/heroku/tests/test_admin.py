"""Admin authentication, the /admin page and deletion on request."""
from __future__ import annotations

import re

import pytest
from conftest import ADMIN_TOKEN
from records import PID, envelope, make_session, make_trial, post, profile
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from nosetouch_server import main as main_module
from nosetouch_server.admin import COOKIE_NAME, AdminAuth
from nosetouch_server.db import ingest_log, participants, sessions, trials
from nosetouch_server.settings import Settings

ADMIN_GETS = ['/admin', '/api/export', '/api/export.zip']


def _seed(client, pid: str = PID, sid: str = 'sess-0001', n_trials: int = 2, **profile_over) -> None:
    prof = profile(participant_id=pid, **profile_over)
    post(client, envelope('session', make_session(session_id=sid, participant_id=pid, participant=prof)))
    for i in range(n_trials):
        hand = 'left' if i % 2 else 'right'
        post(client, envelope('trial', make_trial(
            trial_id=f'{sid}-t{i}', session_id=sid, participant_id=pid, trial_index=i,
            condition={'hand': hand, 'orientation': 'upright_toward', 'speed': 'natural', 'variant': 'standard'})))


@pytest.mark.parametrize('path', ADMIN_GETS)
def test_admin_is_off_without_an_admin_token(make_client, path):
    client = make_client(admin_token=None)
    r = client.get(path, headers={'Authorization': 'Bearer anything'})
    assert r.status_code == 503
    assert client.delete(f'/api/participants/{PID}').status_code == 503


@pytest.mark.parametrize('path', ADMIN_GETS)
def test_401_without_credentials_and_403_with_a_wrong_token(client, path):
    r = client.get(path)
    assert r.status_code == 401
    if path != '/admin':
        assert r.headers['www-authenticate'].startswith('Bearer')
    r = client.get(path, headers={'Authorization': 'Bearer not-the-token'})
    assert r.status_code == 403


@pytest.mark.parametrize('path', ADMIN_GETS)
def test_200_with_the_bearer_token(client, admin_headers, path):
    _seed(client)
    r = client.get(path, headers=admin_headers)
    assert r.status_code == 200
    assert r.headers['cache-control'] == 'no-store'


def test_login_sets_a_signed_cookie_and_redirects(make_client):
    client = make_client(base_url='https://testserver')
    r = client.get('/admin/login', params={'token': ADMIN_TOKEN}, follow_redirects=False)
    assert r.status_code == 303 and r.headers['location'] == '/admin'
    cookie = r.headers['set-cookie']
    assert cookie.startswith(f'{COOKIE_NAME}=')
    for attribute in ('HttpOnly', 'Secure', 'SameSite=strict', 'Path=/'):
        assert attribute.lower() in cookie.lower(), attribute
    assert ADMIN_TOKEN not in cookie  # the cookie holds an expiry and its HMAC, not the token
    # the browser now opens the admin page and the exports without the token in any URL
    assert client.get('/admin').status_code == 200
    assert client.get('/api/export').status_code == 200
    # signing out clears it
    client.get('/admin/logout', follow_redirects=False)
    client.cookies.clear()
    assert client.get('/admin').status_code == 401


def test_login_form_post(make_client):
    client = make_client(base_url='https://testserver')
    r = client.post('/admin/login', data={'token': ADMIN_TOKEN}, follow_redirects=False)
    assert r.status_code == 303 and COOKIE_NAME in r.headers['set-cookie']
    assert client.get('/admin').status_code == 200


def test_a_wrong_login_sets_no_cookie(make_client):
    client = make_client(base_url='https://testserver')
    r = client.get('/admin/login', params={'token': 'wrong'}, follow_redirects=False)
    assert r.status_code == 403 and 'set-cookie' not in r.headers
    r = client.get('/admin/login')
    assert r.status_code == 200 and 'type="password"' in r.text  # the sign-in form


def test_expired_or_forged_cookies_are_refused(make_client):
    client = make_client(base_url='https://testserver')
    auth = AdminAuth(Settings(admin_token=ADMIN_TOKEN, admin_cookie_max_age=60))
    expired = auth.make_cookie(now_s=1_000_000)
    other = AdminAuth(Settings(admin_token='another-token')).make_cookie()
    for value in (expired, other, 'v1.99999999999.deadbeef', 'garbage', ADMIN_TOKEN):
        assert client.get('/admin', headers={'Cookie': f'{COOKIE_NAME}={value}'}).status_code == 401, value
    good = AdminAuth(Settings(admin_token=ADMIN_TOKEN)).make_cookie()
    assert client.get('/admin', headers={'Cookie': f'{COOKIE_NAME}={good}'}).status_code == 200
    assert auth.cookie_ok(auth.make_cookie())


BACKSLASH = chr(92)
CRAFTED_COOKIES = [
    # pure ASCII on the wire; the cookie parser unquotes the octal escape to 'v1.².abc', and '²'.isdigit()
    pytest.param(f'"v1.{BACKSLASH}262.abc"', id='superscript-digit'),
    pytest.param('v1.' + '9' * 5000 + '.x', id='5000-digits'),  # beyond int()'s 4300-digit limit
    pytest.param('v1.' + '9' * 19 + '.x', id='19-digits'),
    pytest.param('v1..x', id='empty-expiry'),
    pytest.param('v1.-5.x', id='negative-expiry'),
    pytest.param('v1. 99999999999.x', id='space'),
]


@pytest.mark.parametrize('cookie', CRAFTED_COOKIES)
def test_a_crafted_cookie_is_refused_and_never_crashes(make_client, cookie):
    client = make_client(base_url='https://testserver')
    headers = {'Cookie': f'{COOKIE_NAME}={cookie}'}
    for path in ADMIN_GETS:
        assert client.get(path, headers=headers).status_code == 401, path
    assert client.delete(f'/api/participants/{PID}', headers=headers).status_code == 401


def test_cookie_ok_never_raises():
    auth = AdminAuth(Settings(admin_token=ADMIN_TOKEN))
    for value in ('v1.².abc', 'v1.١٢.abc', 'v1.' + '9' * 5000 + '.x', 'v1.\ud800.x', '..', 'v1.1.2.3'):
        assert auth.cookie_ok(value) is False, value[:12]


def test_an_unconvertible_since_filter_is_a_400(client, admin_headers):
    for since in ('0001-01-01T00:00:00+05:00', '9999-12-31T23:59:59-05:00'):
        for path in ('/api/export', '/api/export.zip'):
            r = client.get(path, params={'since': since}, headers=admin_headers)
            assert (r.status_code, r.json()['code']) == (400, 'bad_filter'), (path, since)


def test_the_admin_page_shows_the_study(client, admin_headers):
    _seed(client, n_trials=3)
    _seed(client, pid='P-ZZZZZZ', sid='sess-0002', n_trials=1, age_years=58, sex='male')
    page = client.get('/admin', headers=admin_headers)
    assert page.status_code == 200
    text = page.text
    for expected in (PID, 'P-ZZZZZZ', '58', 'male', 'female', 'android', 'L 1 / R 2', 'L 0 / R 1',
                     '/api/export.zip?participant_id=P-ZZZZZZ', '/api/export.zip?include_simulated=false'):
        assert expected in text, expected
    csp = page.headers['content-security-policy']
    nonce = re.search(r"script-src 'nonce-([^']+)'", csp).group(1)
    assert f'<script nonce="{nonce}">' in text
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert page.headers['x-frame-options'] == 'DENY'
    assert 'http://' not in text.replace('http://localhost', '') and 'https://' not in text  # no external assets


def test_every_uploaded_string_is_escaped(client, admin_headers):
    xss = '<script>alert(1)</script>'
    attr = '"><img src=x onerror=alert(2)>'
    post(client, envelope('session', make_session(
        participant=profile(nickname=xss, sex=attr, handedness_self=xss, phone_hand=attr, ehi_class=xss),
        status=xss, device={'platform': attr, 'browser': xss}, protocol={'preset': xss}, app_version=attr)))
    post(client, envelope('trial', make_trial(condition={'hand': xss, 'orientation': attr, 'speed': xss,
                                                          'variant': attr},
                                              outcome={'status': xss, 'accepted': True, 'note': None})))
    post(client, envelope('trial', make_trial(trial_id='bad', schema_version='9.0.0', app_version=xss)))  # refused
    text = client.get('/admin', headers=admin_headers).text
    assert '<script>alert' not in text and '<img src=x' not in text
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in text
    assert '&quot;&gt;&lt;img src=x onerror=alert(2)&gt;' in text


def test_delete_a_participant(client, admin_headers):
    _seed(client, n_trials=3)
    _seed(client, pid='P-ZZZZZZ', sid='sess-0002', n_trials=2)
    # a trial filed under the participant's session with another code is deleted with the session
    post(client, envelope('trial', make_trial(trial_id='odd', session_id='sess-0001', participant_id='P-YYYYYY')))
    assert client.delete(f'/api/participants/{PID}').status_code == 401
    r = client.delete(f'/api/participants/{PID}', headers=admin_headers)
    assert r.status_code == 200
    assert r.json() == {'ok': True, 'participant_id': PID,
                        'deleted': {'participants': 1, 'sessions': 1, 'trials': 4, 'ingest_log': 4}}
    db = client.app.state.db
    with db.read() as conn:
        assert sorted(conn.execute(select(participants.c.participant_id)).scalars()) == ['P-YYYYYY', 'P-ZZZZZZ']
        assert conn.execute(select(sessions.c.session_id)).scalars().all() == ['sess-0002']
        assert sorted(conn.execute(select(trials.c.trial_id)).scalars().all()) == ['sess-0002-t0', 'sess-0002-t1']
        assert PID not in conn.execute(select(ingest_log.c.participant_id)).scalars().all()
    # deleting again finds nothing; a bad code is refused
    assert client.delete(f'/api/participants/{PID}', headers=admin_headers).json()['deleted']['sessions'] == 0
    assert client.delete('/api/participants/nobody', headers=admin_headers).status_code == 400


def test_a_failed_deletion_says_so(client, admin_headers, monkeypatch):
    _seed(client)

    def down(*_a, **_k):
        raise OperationalError('DELETE ...', {}, ConnectionError('server closed the connection'))

    monkeypatch.setattr(main_module, 'delete_participant', down)
    r = client.delete(f'/api/participants/{PID}', headers=admin_headers)
    assert r.status_code == 503 and r.json()['retry'] is True and 'nothing was deleted' in r.json()['error']
