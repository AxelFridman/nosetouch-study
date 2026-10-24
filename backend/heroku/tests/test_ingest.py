"""POST /api/ingest: the json_post contract of server/receiver.py, stored idempotently in the database."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from records import PID, SID, STUDY, T0, T1, T2, deep, envelope, make_session, make_trial, post, profile
from sqlalchemy import insert, select
from sqlalchemy.exc import OperationalError

from nosetouch_server import main as main_module
from nosetouch_server.db import Database, ingest_log, participants, sessions, trials
from nosetouch_server.validation import iso

OK_KEYS = {'ok', 'kind', 'id', 'stored_as', 'received_at', 'server_version'}
REFUSAL_KEYS = {'ok', 'error', 'code', 'retry', 'received_at', 'server_version'}


def rows(client: Any, table: Any) -> list[dict[str, Any]]:
    db = client.app.state.db
    with db.read() as conn:
        return [dict(r._mapping) for r in conn.execute(select(table))]


def data_counts(client: Any) -> tuple[int, int, int]:
    return len(rows(client, participants)), len(rows(client, sessions)), len(rows(client, trials))


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode('utf-8') if isinstance(data, str) else data).hexdigest()


# ───────────────────────────── happy paths ─────────────────────────────


def test_session_is_stored_and_answered_like_the_receiver(client):
    session = make_session(notes='Hebrew text: שלום')
    r = post(client, envelope('session', session))
    assert r.status_code == 200
    reply = r.json()
    assert set(reply) == OK_KEYS
    assert reply['ok'] is True and reply['kind'] == 'session' and reply['id'] == SID
    assert reply['stored_as'] == f'{STUDY}/{PID}/{SID}/session.json'
    assert reply['received_at'].endswith('Z') and len(reply['received_at']) == 24
    assert r.headers['x-nosetouch-outcome'] == 'stored'
    assert r.headers['cache-control'] == 'no-store'
    [row] = rows(client, sessions)
    assert row['record'] == session
    assert (row['participant_id'], row['study_code'], row['status'], row['platform'], row['browser'], row['preset'],
            row['n_planned'], row['simulated'], row['app_version'], row['schema_version'], row['updated_at']) == (
        PID, STUDY, 'in_progress', 'android', 'Chrome 155.0.0.0', 'quick', 24, False, '0.1.0', '1.1.0', T0)


def test_trial_is_stored_with_its_label_columns(client):
    trial = make_trial(trial_id='t-x', trial_index=7, attempt=2,
                       condition={'hand': 'left', 'orientation': 'top_left_toward', 'speed': 'slow', 'variant': 'curved'})
    r = post(client, envelope('trial', trial))
    assert r.status_code == 200, r.text
    assert r.json()['stored_as'] == f'{STUDY}/{PID}/{SID}/trial_007_a2_t-x.json'
    [row] = rows(client, trials)
    assert row['record'] == trial
    assert (row['session_id'], row['participant_id'], row['trial_index'], row['attempt'], row['hand'], row['orientation'],
            row['speed'], row['variant'], row['status'], row['accepted'], row['simulated'], row['created_at']) == (
        SID, PID, 7, 2, 'left', 'top_left_toward', 'slow', 'curved', 'complete', True, False, T0)


def test_the_participant_profile_comes_from_the_session_snapshot(client):
    post(client, envelope('session', make_session()))
    [p] = rows(client, participants)
    assert p['record'] == profile() and p['study_code'] == STUDY
    # a later session carries an edited profile: it replaces the stored one
    newer = profile(age_years=32, updated_at=T1)
    post(client, envelope('session', make_session(session_id='sess-0002', participant=newer)))
    # an older snapshot (an earlier session re-sent) does not undo the edit
    post(client, envelope('session', make_session(session_id='sess-0003', participant=profile(age_years=30))))
    [p] = rows(client, participants)
    assert p['record'] == newer
    assert p['last_seen'] >= p['first_seen']


def test_a_trial_without_its_session_still_registers_the_participant(client):
    post(client, envelope('trial', make_trial(participant_id='P-ZZZZZZ')))
    [p] = rows(client, participants)
    assert p['participant_id'] == 'P-ZZZZZZ' and p['record'] is None


def test_a_snapshot_of_another_participant_is_ignored(client):
    post(client, envelope('session', make_session(participant=profile(participant_id='P-ZZZZZZ'))))
    [p] = rows(client, participants)
    assert p['participant_id'] == PID and p['record'] is None


@pytest.mark.parametrize('path', ['/api/ingest', '/api/ingest/', '/ingest', '/ingest/'])
def test_ingest_paths(client, path):
    r = post(client, envelope('trial', make_trial()), path=path)
    assert r.status_code == 200, r.text
    assert r.json()['stored_as'] == f'{STUDY}/{PID}/{SID}/trial_000_a1_trial-0001.json'


def test_a_text_plain_body_is_accepted_like_the_receiver(client):
    r = post(client, envelope('session', make_session()), content_type='text/plain;charset=UTF-8')
    assert r.status_code == 200 and r.json()['ok'] is True


@pytest.mark.parametrize('index, attempt, name', [
    (0, 1, 'trial_000_a1_t-x.json'),
    (7, 2, 'trial_007_a2_t-x.json'),
    (123, 1, 'trial_123_a1_t-x.json'),
    (1234, 3, 'trial_1234_a3_t-x.json'),
    (7.0, 1.0, 'trial_007_a1_t-x.json'),  # JSON 7.0 is the number 7 for JavaScript too
    (2**53 - 1, 1, f'trial_{2**53 - 1}_a1_t-x.json'),  # the largest safe integer fits the BIGINT column
])
def test_trial_file_names(client, index, attempt, name):
    r = post(client, envelope('trial', make_trial(trial_id='t-x', trial_index=index, attempt=attempt)))
    assert r.status_code == 200, r.text
    assert r.json()['stored_as'] == f'{STUDY}/{PID}/{SID}/{name}'


# ───────────────────────────── retries and versions ─────────────────────────────


def test_retries_are_idempotent(client):
    session, trial = make_session(), make_trial()
    first = [post(client, envelope('session', session)), post(client, envelope('trial', trial))]
    again = [post(client, envelope('session', session)), post(client, envelope('trial', trial))]
    assert [r.status_code for r in first + again] == [200] * 4
    assert [r.headers['x-nosetouch-outcome'] for r in first + again] == ['stored', 'stored', 'unchanged', 'unchanged']
    assert [r.json()['stored_as'] for r in first] == [r.json()['stored_as'] for r in again]
    assert data_counts(client) == (1, 1, 1)
    # a re-send with another key order and another sent_at is the same record
    reordered = json.loads(json.dumps(trial, sort_keys=True))
    r = post(client, envelope('trial', reordered, sent_at=T2))
    assert r.headers['x-nosetouch-outcome'] == 'unchanged'
    assert [row['outcome'] for row in rows(client, ingest_log)] == ['stored', 'stored', 'unchanged', 'unchanged', 'unchanged']


def test_a_newer_session_wins_and_an_older_retry_is_ignored(client):
    v1 = make_session(updated_at=T0)
    v2 = make_session(updated_at=T2, status='completed', ended_at=T2,
                      progress={'next_trial_index': 24, 'n_attempts': 25, 'n_accepted': 24})
    assert post(client, envelope('session', v1)).headers['x-nosetouch-outcome'] == 'stored'
    assert post(client, envelope('session', v2)).headers['x-nosetouch-outcome'] == 'replaced'
    late = post(client, envelope('session', v1, sent_at=T2))  # an old copy arriving last (a slow retry)
    assert late.status_code == 200 and late.json()['ok'] is True  # the phone must stop sending it
    assert late.headers['x-nosetouch-outcome'] == 'kept_newer'
    [row] = rows(client, sessions)
    assert row['record'] == v2 and row['status'] == 'completed' and row['n_accepted'] == 24
    log = rows(client, ingest_log)
    assert [(e['outcome'], e['code']) for e in log] == [
        ('stored', None), ('replaced', 'newer_updated_at'), ('kept_newer', 'older_updated_at')]


def test_equal_updated_at_falls_back_to_sent_at(client):
    a = make_session(notes='a')
    b = make_session(notes='b')
    post(client, envelope('session', a, sent_at=T1))
    assert post(client, envelope('session', b, sent_at=T0)).headers['x-nosetouch-outcome'] == 'kept_newer'
    assert post(client, envelope('session', b, sent_at=T2)).headers['x-nosetouch-outcome'] == 'replaced'
    assert rows(client, sessions)[0]['record']['notes'] == 'b'


def test_trial_conflicts_keep_the_decided_and_newest_copy(client):
    pending = make_trial(outcome={'status': 'complete', 'accepted': False, 'participant_flag': None,
                                  'note': 'review_interrupted'})
    decided = make_trial()
    assert post(client, envelope('trial', pending, sent_at=T0)).headers['x-nosetouch-outcome'] == 'stored'
    assert post(client, envelope('trial', decided, sent_at=T1)).headers['x-nosetouch-outcome'] == 'replaced'
    # the protective copy never undoes the participant's decision, even when it arrives last
    r = post(client, envelope('trial', pending, sent_at=T2))
    assert (r.status_code, r.json()['ok'], r.headers['x-nosetouch-outcome']) == (200, True, 'kept_newer')
    [row] = rows(client, trials)
    assert row['record'] == decided and row['accepted'] is True and row['review_interrupted'] is False
    # two different decided copies: the later envelope wins, an earlier one is a stale retry
    flagged = deep(decided)
    flagged['outcome'].update(accepted=False, participant_flag='wrong_hand')
    assert post(client, envelope('trial', flagged, sent_at=T0)).headers['x-nosetouch-outcome'] == 'kept_newer'
    assert post(client, envelope('trial', flagged, sent_at=T2)).headers['x-nosetouch-outcome'] == 'replaced'
    [row] = rows(client, trials)
    assert row['record'] == flagged and row['accepted'] is False
    assert [(e['outcome'], e['code']) for e in rows(client, ingest_log)] == [
        ('stored', None), ('replaced', 'decided_copy'), ('kept_newer', 'review_interrupted_copy'),
        ('kept_newer', 'older_sent_at'), ('replaced', 'newer_sent_at')]


def test_v2_copies_follow_the_apps_save_order_whatever_the_envelope_times(client):
    """v2 (hands-free): the choice trial's copy before its hand question ('hand_not_reported') is a
    placeholder like 'review_interrupted'; a revocation from the pause sheet ('redo_requested') is final."""
    note = lambda n, accepted=False, flag=None: {'status': 'complete', 'accepted': accepted,  # noqa: E731
                                                 'participant_flag': flag, 'note': n}
    placeholder = make_trial(trial_id='choice-1', outcome=note('hand_not_reported'))
    reported = make_trial(trial_id='choice-1', outcome=note(None, accepted=True),
                          condition={'hand': 'left', 'orientation': 'free', 'speed': 'natural',
                                     'variant': 'standard', 'hand_source': 'reported'})
    revoked = deep(reported)
    revoked['outcome'] = note('redo_requested', flag='wrong_hand')
    outcomes = [
        post(client, envelope('trial', placeholder, sent_at=T0)),
        post(client, envelope('trial', reported, sent_at=T1)),  # the answer replaces the placeholder
        post(client, envelope('trial', placeholder, sent_at=T2)),  # a late retry of the placeholder
        post(client, envelope('trial', revoked, sent_at=T0)),  # revoked, even with an older envelope time
        post(client, envelope('trial', reported, sent_at=T2)),  # a stale decided copy cannot undo it
        post(client, envelope('trial', placeholder, sent_at=T2)),
    ]
    assert all(r.status_code == 200 and r.json()['ok'] is True for r in outcomes)  # the phone stops sending
    assert [r.headers['x-nosetouch-outcome'] for r in outcomes] == [
        'stored', 'replaced', 'kept_newer', 'replaced', 'kept_newer', 'kept_newer']
    [row] = rows(client, trials)
    assert row['record'] == revoked and row['accepted'] is False and row['review_interrupted'] is False
    assert [(e['outcome'], e['code']) for e in rows(client, ingest_log)] == [
        ('stored', None), ('replaced', 'decided_copy'), ('kept_newer', 'review_interrupted_copy'),
        ('replaced', 'revoked_copy'), ('kept_newer', 'kept_revoked'), ('kept_newer', 'kept_revoked')]


def test_a_stored_hand_not_reported_copy_is_marked_as_a_placeholder(client):
    placeholder = make_trial(outcome={'status': 'complete', 'accepted': False, 'participant_flag': None,
                                      'note': 'hand_not_reported'})
    post(client, envelope('trial', placeholder))
    [row] = rows(client, trials)
    assert row['review_interrupted'] is True and row['accepted'] is False


@pytest.mark.parametrize(('inc', 'old', 'expected'), [
    # (rank, sent_at) of the incoming and of the stored copy → (replace?, reason)
    ((1, T0), (0, T2), (True, 'decided_copy')),
    ((0, T2), (1, T0), (False, 'review_interrupted_copy')),
    ((2, T0), (1, T2), (True, 'revoked_copy')),
    ((2, T0), (0, T2), (True, 'revoked_copy')),
    ((1, T2), (2, T0), (False, 'kept_revoked')),
    ((0, T2), (2, T0), (False, 'kept_revoked')),
    ((1, T2), (1, T1), (True, 'newer_sent_at')),
    ((2, T1), (2, T1), (True, 'newer_sent_at')),
    ((0, T0), (0, T1), (False, 'older_sent_at')),
])
def test_trial_replaces_ranks_before_envelope_times(inc, old, expected):
    from nosetouch_server.store import trial_replaces
    assert trial_replaces(inc[0], inc[1], old[0], old[1]) == expected


def test_copy_rank_of_every_note_the_app_writes():
    from nosetouch_server.store import DECIDED, PLACEHOLDER, REVOKED, copy_rank
    rank = lambda n: copy_rank({'outcome': {'note': n}})  # noqa: E731
    assert [rank(n) for n in ('review_interrupted', 'hand_not_reported')] == [PLACEHOLDER] * 2
    assert rank('redo_requested') == REVOKED
    for decided in (None, 'auto_accept', 'auto_reject:orientation', 'hand_unsure', 'anything else'):
        assert rank(decided) == DECIDED
    assert copy_rank({}) == DECIDED and copy_rank({'outcome': None}) == DECIDED and copy_rank(None) == DECIDED


def test_concurrent_uploads(client):
    distinct = [envelope('trial', make_trial(trial_id=f'c-{i}', trial_index=i)) for i in range(16)]
    same = [envelope('trial', make_trial(trial_id='shared', trial_index=99, app_version=f'v{i}'), sent_at=T0)
            for i in range(8)]
    sessions_ = [envelope('session', make_session(updated_at=f'2026-10-23T10:00:{i:02d}.000Z')) for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda env: post(client, env), distinct + same + sessions_))
    assert all(r.status_code == 200 and r.json()['ok'] for r in results), [r.text for r in results if r.status_code != 200]
    assert data_counts(client) == (1, 1, 17)
    shared = [row for row in rows(client, trials) if row['trial_id'] == 'shared'][0]
    assert shared['record']['app_version'] in {f'v{i}' for i in range(8)}  # one complete version, never a mix
    assert rows(client, sessions)[0]['updated_at'] == '2026-10-23T10:00:07.000Z'  # the newest wins


# ───────────────────────────── rejections ─────────────────────────────

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
    ('schema 0.9.0', _trial(schema_version='0.9.0'), 'bad_schema_version'),
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
    ('a NUL character', json.dumps(envelope('session', make_session(notes='a\u0000b'))), 'unsupported_text'),
    ('an unpaired surrogate', json.dumps(envelope('session', make_session(notes='a\ud800b'))), 'unsupported_text'),
]


@pytest.mark.parametrize('label, text, code', INVALID, ids=[c[0] for c in INVALID])
def test_invalid_envelopes_are_refused_and_never_stored(client, label, text, code):
    r = post(client, text)
    assert r.status_code == 400, r.text
    reply = r.json()
    assert set(reply) == REFUSAL_KEYS
    assert reply['ok'] is False and reply['retry'] is False and reply['code'] == code
    assert isinstance(reply['error'], str) and len(reply['error']) > 5
    assert data_counts(client) == (0, 0, 0)
    [line] = rows(client, ingest_log)
    assert (line['outcome'], line['code'], line['bytes'], line['sha256']) == (
        'rejected', code, len(text.encode('utf-8')), sha256(text))


def test_messages_match_the_receiver(client):
    assert post(client, _trial(participant_id='bad')).json()['error'] == \
        'participant_id must match /^P-[A-HJ-NP-Z2-9]{6}$/'
    assert post(client, json.dumps(envelope('nope', make_trial()))).json()['error'] == "kind must be 'session' or 'trial'"
    assert post(client, _trial(schema_version='2.0.0')).json()['error'] == \
        'unsupported schema_version (this receiver accepts 1.x)'


def test_a_wrong_schema_major_is_refused_but_any_1x_minor_is_accepted(client):
    for version, status in (('2.0.0', 400), ('3.1.4', 400), ('1.1.0', 200), ('1.7.2', 200)):
        r = post(client, envelope('trial', make_trial(trial_id=f't-{version.replace(".", "_")}', schema_version=version)))
        assert r.status_code == status, (version, r.text)
        if status == 400:
            assert r.json()['code'] == 'bad_schema_version'


def test_malformed_bodies(client):
    for raw in (b'{', b'{"kind": "trial",}', b'\xff\xfe{"kind":"trial"}',
                '\ufeff'.encode('utf-8') + _trial().encode('utf-8')):
        r = post(client, raw)
        assert r.status_code == 400 and r.json()['code'] == 'malformed_json', raw[:20]
    assert data_counts(client) == (0, 0, 0)


def test_a_literal_backslash_u0000_text_is_not_a_nul(client):
    r = post(client, envelope('session', make_session(notes='C:\\u0000 is just text')))
    assert r.status_code == 200, r.text


def test_empty_body(client):
    r = post(client, b'')
    assert r.status_code == 400 and r.json()['code'] == 'empty_body'
    [line] = rows(client, ingest_log)
    assert (line['code'], line['bytes'], line['sha256']) == ('empty_body', 0, sha256(b''))


def test_study_key(make_client):
    keyed = make_client(study_key='k-123')
    for key in (None, 'k-999', 7, 'k-123 ', '\ud800'):
        r = post(keyed, envelope('trial', make_trial(), key))
        assert (r.status_code, r.json()['code'], r.json()['retry']) == (400, 'bad_study_key', False), key
    assert post(keyed, envelope('trial', make_trial(), 'k-123')).status_code == 200
    open_ = make_client()
    assert post(open_, envelope('trial', make_trial(), None)).status_code == 200
    assert post(open_, envelope('trial', make_trial(trial_id='t2'), 'anything')).status_code == 200


def test_size_limit_with_content_length(make_client):
    small = make_client(max_bytes=5000)
    base = len(json.dumps(envelope('session', make_session(notes=''))))
    ok_body = json.dumps(envelope('session', make_session(notes='n' * (5000 - base))))
    assert len(ok_body) == 5000
    assert post(small, ok_body).status_code == 200
    big_base = len(json.dumps(envelope('session', make_session(session_id='sess-big', notes=''))))
    big = json.dumps(envelope('session', make_session(session_id='sess-big', notes='n' * (5001 - big_base))))
    assert len(big) == 5001
    r = post(small, big)
    assert r.status_code == 413
    assert r.json()['code'] == 'too_large' and r.json()['retry'] is False and '5001' in r.json()['error']
    # read to its end before the answer, so the connection stays open (tests/test_wire.py: why)
    assert r.headers.get('connection') != 'close'
    assert [row['session_id'] for row in rows(small, sessions)] == [SID]  # the big one was not stored
    line = [e for e in rows(small, ingest_log) if e['outcome'] == 'rejected'][0]
    assert (line['code'], line['bytes'], line['sha256'], line['kind'], line['participant_id']) == (
        'too_large', 5001, sha256(big), None, None)  # hashed while it was read, never parsed


def test_a_body_beyond_the_drain_limit_is_refused_unread(make_client):
    small = make_client(max_bytes=5000)
    huge = b'x' * (5000 * main_module.DRAIN_FACTOR + 1)
    r = post(small, huge)
    assert (r.status_code, r.json()['code'], r.json()['retry']) == (413, 'too_large', False)
    assert f'({len(huge)} bytes > 5000)' in r.json()['error']
    assert r.headers['connection'] == 'close'
    [line] = rows(small, ingest_log)
    assert (line['code'], line['bytes'], line['sha256']) == ('too_large', len(huge), None)  # never read


def _chunked_post(client: Any, body: bytes) -> Any:
    def chunks():  # a generator body is sent chunked, without Content-Length
        for i in range(0, len(body), 1000):
            yield body[i:i + 1000]

    r = client.post('/api/ingest', content=chunks(), headers={'Content-Type': 'application/json'})
    assert r.request.headers.get('content-length') is None
    return r


def test_size_limit_without_content_length(make_client):
    small = make_client(max_bytes=5000)
    body = json.dumps(envelope('session', make_session(notes='n' * 6000))).encode('utf-8')
    r = _chunked_post(small, body)
    assert r.status_code == 413 and r.json()['code'] == 'too_large'
    assert f'({len(body)} bytes > 5000)' in r.json()['error']  # read to its end: the size is known
    assert r.headers.get('connection') != 'close'
    assert rows(small, sessions) == []
    [line] = rows(small, ingest_log)
    assert (line['bytes'], line['sha256']) == (len(body), sha256(body))


def test_a_chunked_body_beyond_the_drain_limit_is_cut_off(make_client):
    small = make_client(max_bytes=5000)
    r = _chunked_post(small, b'x' * (5000 * main_module.DRAIN_FACTOR + 1000))
    assert r.status_code == 413 and r.json()['code'] == 'too_large'
    assert 'more than 5000 bytes' in r.json()['error']  # reading stopped: the full size is unknown
    assert r.headers['connection'] == 'close'


def test_a_chunked_body_within_the_limit_is_accepted(client):
    body = json.dumps(envelope('trial', make_trial())).encode('utf-8')
    r = client.post('/api/ingest', content=iter([body[:50], body[50:]]), headers={'Content-Type': 'application/json'})
    assert r.request.headers.get('content-length') is None
    assert r.status_code == 200, r.text


def test_the_default_limit_accepts_a_long_recording(client):
    n = 30_000  # about 8 minutes at 60 Hz: far longer than any trial so far (165 KB, 23 s)
    trial = make_trial(streams={'motion': {k: [round(i * 0.0123, 4) for i in range(n)]
                                           for k in ('t', 'acc_x', 'acc_y', 'acc_z', 'rr_alpha')}})
    body = json.dumps(envelope('trial', trial))
    assert 1_000_000 < len(body) < 4_000_000
    assert post(client, body).status_code == 200


def test_the_log_holds_identifiers_only(make_client):
    client = make_client()
    captured: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    grab = Grab()
    logging.getLogger('nosetouch').addHandler(grab)
    try:
        secret_profile = profile(nickname='Dana Levi', phone_model='Galaxy S24')
        refused = envelope('session', make_session(schema_version='2.0.0', participant=secret_profile,
                                                   notes='private remark'))
        accepted = envelope('session', make_session(participant=secret_profile, notes='private remark'))
        assert post(client, refused).status_code == 400
        assert post(client, accepted).status_code == 200
    finally:
        logging.getLogger('nosetouch').removeHandler(grab)
    log_rows = rows(client, ingest_log)
    assert [e['outcome'] for e in log_rows] == ['rejected', 'stored']
    assert log_rows[0]['participant_id'] == PID and log_rows[0]['schema_version'] == '2.0.0'
    everything = json.dumps(log_rows, default=str) + '\n'.join(captured)
    for secret in ('Dana Levi', 'Galaxy S24', 'private remark', 'age_years', '"record"'):
        assert secret not in everything, secret
    assert any('"outcome":"stored"' in line for line in captured)


def test_log_lines_carry_the_routers_request_id(client):
    captured: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    grab = Grab()
    logging.getLogger('nosetouch').addHandler(grab)
    try:
        post(client, envelope('trial', make_trial()), headers={'X-Request-Id': 'f9ed4675-f3ea-4bbf-a6d3-b0e6e1b7fe36'})
        post(client, envelope('trial', make_trial()), headers={'X-Request-Id': 'bad id\n{"forged":1}'})
    finally:
        logging.getLogger('nosetouch').removeHandler(grab)
    lines = [json.loads(line) for line in captured if '"event":"ingest"' in line]
    assert lines[0]['request_id'] == 'f9ed4675-f3ea-4bbf-a6d3-b0e6e1b7fe36'
    assert 'request_id' not in lines[1] and 'forged' not in lines[1]


def test_rejection_log_rows_are_capped_per_day(make_client):
    client = make_client(reject_log_daily_cap=3)
    for i in range(5):
        r = post(client, json.dumps(envelope('spam', make_trial(trial_id=f't-{i}'))))
        assert (r.status_code, r.json()['code']) == (400, 'bad_kind')  # the answer never changes
    assert [e['record_id'] for e in rows(client, ingest_log)] == ['t-0', 't-1', 't-2']


def test_the_ingest_log_is_pruned_by_age_and_size(make_client):
    client = make_client(log_retention_days=30, log_max_rows=5)
    db = client.app.state.db
    now = dt.datetime(2026, 11, 20, 12, 0, tzinfo=dt.timezone.utc)
    with db.write() as conn:
        for i in range(10):
            at = now - dt.timedelta(days=40 if i < 3 else 1, minutes=10 - i)
            conn.execute(insert(ingest_log).values(at=at, outcome='stored', record_id=f'r{i}'))
    assert db.prune_log(now) == 5  # 3 too old, then the 2 oldest beyond 5 rows
    assert [e['record_id'] for e in rows(client, ingest_log)] == ['r5', 'r6', 'r7', 'r8', 'r9']


# ───────────────────────────── other answers ─────────────────────────────


def test_the_endpoint_probe_answers_like_the_receiver(client):
    for path in ('/api/ingest', '/ingest', '/api/ingest/'):
        r = client.get(path)
        assert r.status_code == 200
        body = r.json()
        assert body['ok'] is True and body['configured'] is True and body['service'] == 'nosetouch-heroku'
        assert body['time'].endswith('Z') and body['server_version']


@contextlib.contextmanager
def _database_down(*_a: Any, **_k: Any) -> Any:
    raise OperationalError('SELECT 1', {}, ConnectionError('connection refused'))
    yield  # a generator, like the real context managers


def test_the_endpoint_probe_reports_a_database_outage(client, monkeypatch):
    """The app's settings screen (Test) must not show 'Connection OK' while every upload gets 503."""
    db = client.app.state.db
    monkeypatch.setattr(db, 'read', _database_down)
    monkeypatch.setattr(db, 'write', _database_down)
    assert post(client, envelope('trial', make_trial())).status_code == 503
    for method in ('GET', 'HEAD'):
        r = client.request(method, '/api/ingest')
        assert r.status_code == 503, method
    body = client.get('/api/ingest').json()
    assert (body['ok'], body['retry'], body['configured'], body['service']) == (False, True, False, 'nosetouch-heroku')
    assert body['error'] == 'database unavailable, try again later' and body['time'].endswith('Z')
    monkeypatch.undo()  # the database is back
    assert client.get('/api/ingest').json()['ok'] is True


def test_the_endpoint_probe_reports_a_database_that_never_came_up(make_client, monkeypatch):
    clock = [0.0]

    async def no_sleep(s: float) -> None:
        clock[0] += s  # (startup waits for the database by the clock: a fake one, so the test does not wait)

    def cannot_connect(_self: Any) -> None:
        raise OperationalError('CREATE TABLE ...', {}, ConnectionError('connection refused'))

    monkeypatch.setattr(main_module, '_sleep', no_sleep)
    monkeypatch.setattr(main_module, '_monotonic', lambda: clock[0])
    monkeypatch.setattr(Database, 'init_schema', cannot_connect)
    client = make_client()
    r = client.get('/api/ingest')
    assert (r.status_code, r.json()['ok'], r.json()['configured']) == (503, False, False)


def test_startup_binds_the_port_in_time_when_the_database_hangs(make_client, monkeypatch):
    """uvicorn binds $PORT only after startup, and Heroku kills a web dyno that has not bound it within
    60 s (R10). Before: 5 attempts, each up to the 10 s connect_timeout of an unreachable host, plus
    28 s of pauses = 78 s, so a database outage during a dyno restart became a crash loop (measured:
    80 s on a real uvicorn) instead of 503 retry answers."""
    clock = [1000.0]  # fake time.monotonic(), advanced by the pauses and the connection attempts
    attempts: list[float] = []

    async def pause(seconds: float) -> None:
        clock[0] += seconds

    def times_out(_self: Any) -> None:
        attempts.append(clock[0])
        clock[0] += 10.0  # psycopg's connect_timeout (db.make_engine)
        raise OperationalError('SELECT pg_advisory_xact_lock(...)', {}, TimeoutError('connection timeout expired'))

    monkeypatch.setattr(main_module, '_sleep', pause)
    monkeypatch.setattr(main_module, '_monotonic', lambda: clock[0])
    monkeypatch.setattr(Database, 'init_schema', times_out)
    client = make_client()  # runs the startup
    startup = clock[0] - 1000.0
    assert startup <= 35.0, f'port bound after {startup} s (Heroku allows 60 s, and the dyno needs some itself)'
    assert len(attempts) >= 2  # it still waits a little for a database that is starting
    # the port is bound: requests answer 503 retry, and each one tries the database again
    assert client.get('/api/ingest').status_code == 503
    assert post(client, envelope('trial', make_trial())).json()['retry'] is True
    assert len(attempts) >= 4


def test_startup_retries_a_database_that_refuses_quickly(make_client, monkeypatch):
    """A refused connection fails at once: startup tries again with growing pauses and no pause after
    giving up, and a database that answers in time is used without any request having to wait."""
    clock = [0.0]
    pauses: list[float] = []
    real_init = Database.init_schema

    async def pause(seconds: float) -> None:
        pauses.append(seconds)
        clock[0] += seconds

    def refused_three_times(self: Any) -> None:
        if len(pauses) < 3:
            raise OperationalError('SELECT 1', {}, ConnectionError('connection refused'))
        real_init(self)

    monkeypatch.setattr(main_module, '_sleep', pause)
    monkeypatch.setattr(main_module, '_monotonic', lambda: clock[0])
    monkeypatch.setattr(Database, 'init_schema', refused_three_times)
    client = make_client()
    assert pauses == [2.0, 4.0, 6.0] and client.app.state.db.ready is True
    assert post(client, envelope('trial', make_trial())).status_code == 200


def test_unknown_paths_and_methods_answer_json(client):
    for method, path, status, code in [
        ('POST', '/nope', 404, 'not_found'),
        ('GET', '/nope', 404, 'not_found'),
        ('PUT', '/api/ingest', 405, 'method_not_allowed'),
        ('DELETE', '/api/ingest', 405, 'method_not_allowed'),
    ]:
        r = client.request(method, path, content=b'{}' if method in ('POST', 'PUT') else None)
        assert r.status_code == status, (method, path)
        assert set(r.json()) == REFUSAL_KEYS and r.json()['code'] == code


def test_a_database_outage_answers_503_retry(client, monkeypatch):
    def down(*_a: Any, **_k: Any) -> Any:
        raise OperationalError('INSERT ...', {}, ConnectionError('server closed the connection'))

    monkeypatch.setattr(main_module, 'store_envelope', down)
    r = post(client, envelope('trial', make_trial()))
    assert r.status_code == 503
    body = r.json()
    assert body['ok'] is False and body['retry'] is True and 'INSERT' not in r.text


def test_an_unexpected_error_answers_500_retry_without_details(client, monkeypatch):
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError('secret detail: Dana Levi')

    monkeypatch.setattr(main_module, 'store_envelope', boom)
    r = post(client, envelope('trial', make_trial()))
    assert r.status_code == 500
    assert r.json() == {'ok': False, 'error': 'server error: RuntimeError', 'retry': True,
                        'received_at': r.json()['received_at'], 'server_version': r.json()['server_version']}
    assert 'Dana' not in r.text


def test_a_post_from_another_site_is_refused(client):
    r = post(client, envelope('trial', make_trial()), headers={'Origin': 'https://evil.example'})
    assert r.status_code == 403 and r.json()['code'] == 'bad_origin'
    assert 'access-control-allow-origin' not in r.headers
    assert data_counts(client) == (0, 0, 0)


def test_health(client):
    post(client, envelope('session', make_session()))
    post(client, envelope('trial', make_trial()))
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] is True
    assert body['db']['ok'] is True and body['db']['schema_version'] == 1
    assert body['counts'] == {'participants': 1, 'sessions': 1, 'trials': 1}


def test_the_landing_page_holds_no_data(client):
    post(client, envelope('session', make_session()))
    r = client.get('/')
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/plain')
    assert PID not in r.text and SID not in r.text


def test_received_at_is_the_server_time(client):
    before = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
    r = post(client, envelope('trial', make_trial()))
    after = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=1)
    assert iso(before) <= r.json()['received_at'] <= iso(after)
