"""import_export.py: replaying app export files through the ingest contract."""
from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import REAL_EXPORT
from records import SID, make_session, make_trial, profile
from sqlalchemy import func, select

import import_export
from nosetouch_server.db import participants, sessions, trials


def client_poster(client: Any) -> import_export.Poster:
    def post_body(body: bytes) -> tuple[int, dict[str, str], bytes]:
        r = client.post('/api/ingest', content=body, headers={'Content-Type': 'application/json'})
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content
    return post_body


def counts(client: Any) -> tuple[int, int, int]:
    with client.app.state.db.read() as conn:
        return tuple(conn.execute(select(func.count()).select_from(t)).scalar_one()
                     for t in (participants, sessions, trials))


def small_bundle(**over: Any) -> dict[str, Any]:
    bundle = {
        'format': 'nose-touch-export', 'schema_version': '1.1.0', 'app_version': '0.1.0',
        'exported_at': '2026-10-23T11:00:00.000Z', 'participants': [profile()],
        'sessions': [make_session()],
        'trials': [make_trial(trial_id='t-2', trial_index=2), make_trial(trial_id='t-0', trial_index=0),
                   make_trial(trial_id='orphan', session_id='sess-gone')],
    }
    bundle.update(over)
    return bundle


@pytest.fixture
def real_bundle() -> dict[str, Any]:
    if not REAL_EXPORT.exists():
        pytest.skip(f'{REAL_EXPORT.name} is not on this machine')
    return import_export.load_bundle(REAL_EXPORT)


def test_the_real_export_replays_into_one_participant_one_session_23_trials(client, real_bundle, capsys):
    assert (len(real_bundle['participants']), len(real_bundle['sessions']), len(real_bundle['trials'])) == (1, 1, 23)
    code = import_export.main([str(REAL_EXPORT), '--quiet'], post=client_poster(client))
    assert code == 0
    assert counts(client) == (1, 1, 23)
    out = capsys.readouterr().out
    assert 'participants: 1 in the file, 1 carried by their session records' in out
    # the stored records are the file's records, byte for byte as JSON values
    db = client.app.state.db
    with db.read() as conn:
        stored_trials = {r.trial_id: r.record for r in conn.execute(select(trials.c.trial_id, trials.c.record))}
        [stored_session] = conn.execute(select(sessions.c.record)).scalars().all()
        [stored_profile] = conn.execute(select(participants.c.record)).scalars().all()
    assert stored_trials == {t['trial_id']: t for t in real_bundle['trials']}
    assert stored_session == real_bundle['sessions'][0]
    assert stored_profile == real_bundle['participants'][0]
    # a second run (e.g. after an interrupted first one) changes nothing
    summary = import_export.run_import(real_bundle, client_poster(client), study_key=None, sleep=lambda s: None)
    assert {r.outcome for r in summary.results} == {'unchanged'}
    assert counts(client) == (1, 1, 23)


def test_envelopes_are_the_apps(tmp_path):
    bundle = small_bundle()
    envs = import_export.build_envelopes(bundle, 'k-1')
    assert [(e.kind, e.record_id) for e in envs] == [
        ('session', SID), ('trial', 't-0'), ('trial', 't-2'), ('trial', 'orphan')]  # session first, plan order
    first = json.loads(envs[0].body)
    assert list(first) == ['kind', 'study_key', 'sent_at', 'record']
    assert first['study_key'] == 'k-1' and first['sent_at'] == bundle['exported_at']
    assert json.loads(import_export.build_envelopes(bundle, None)[0].body)['study_key'] is None


def test_summary_counts_and_outcomes(client):
    summary = import_export.run_import(small_bundle(), client_poster(client), study_key=None, sleep=lambda s: None)
    assert summary.ok
    assert (summary.count('session', 'stored'), summary.count('trial', 'stored')) == (1, 3)
    text = import_export.format_summary(summary, dry_run=False)
    assert 'sessions' in text and 'trials' in text and 'participants: 1 in the file, 1 carried' in text


def test_a_participant_without_a_session_is_reported(client):
    bundle = small_bundle(participants=[profile(), profile(participant_id='P-ZZZZZZ')])
    summary = import_export.run_import(bundle, client_poster(client), study_key=None, sleep=lambda s: None)
    assert 'NOT carried (no session in the file): P-ZZZZZZ' in import_export.format_summary(summary, False)


def test_the_study_key_is_sent(make_client):
    keyed = make_client(study_key='k-123')
    refused = import_export.run_import(small_bundle(), client_poster(keyed), study_key=None, sleep=lambda s: None)
    assert not refused.ok
    assert {r.outcome for r in refused.results} == {'rejected'}
    assert all(r.detail.startswith('HTTP 400 bad_study_key') for r in refused.results)
    assert all(r.attempts == 1 for r in refused.results)  # a refusal is never retried
    accepted = import_export.run_import(small_bundle(), client_poster(keyed), study_key='k-123', sleep=lambda s: None)
    assert accepted.ok and counts(keyed) == (1, 1, 3)


def test_transient_failures_are_retried_with_backoff():
    answers = [OSError('connection reset'), (503, {}, b'{"ok":false,"retry":true,"error":"busy"}'),
               (200, {'x-nosetouch-outcome': 'stored'}, b'{"ok":true}')]
    waits: list[float] = []

    def flaky(body: bytes) -> tuple[int, dict[str, str], bytes]:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    env = import_export.build_envelopes(small_bundle(trials=[]), None)[0]
    result = import_export.send(env, flaky, retries=4, backoff_s=2.0, sleep=waits.append)
    assert (result.outcome, result.attempts) == ('stored', 3)
    assert waits == [2.0, 4.0]


def test_retries_give_up_after_the_limit():
    env = import_export.build_envelopes(small_bundle(trials=[]), None)[0]
    result = import_export.send(env, lambda body: (502, {}, b'<html>bad gateway</html>'), retries=2,
                                sleep=lambda s: None)
    assert (result.outcome, result.attempts) == ('failed', 3)


def test_dry_run_sends_nothing(capsys, tmp_path):
    path = tmp_path / 'bundle.json'
    bad = make_trial(trial_id='bad', schema_version='2.0.0')
    path.write_text(json.dumps(small_bundle(trials=[make_trial(), bad])), encoding='utf-8')

    def never(body: bytes) -> Any:
        raise AssertionError('--dry-run must not send')

    assert import_export.main([str(path), '--dry-run', '--quiet'], post=never) == 1  # one invalid record
    out = capsys.readouterr().out
    assert 'checked' in out and 'bad_schema_version' in out


def test_files_and_urls_are_checked(tmp_path, capsys):
    not_export = tmp_path / 'other.json'
    not_export.write_text('{"hello": 1}', encoding='utf-8')
    assert import_export.main([str(not_export), '--url', 'https://x.example/api/ingest']) == 2
    good = tmp_path / 'bundle.json.txt'
    good.write_text(json.dumps(small_bundle()), encoding='utf-8')
    assert import_export.main([str(good), '--url', 'http://example.org/api/ingest']) == 2  # plain http refused
    assert 'plain http' in capsys.readouterr().err


def test_env_file_parsing(tmp_path):
    f = tmp_path / 'heroku.env'
    f.write_text('# NoseTouch\nAPP_URL=https://nosetouch-abc.herokuapp.com/\nexport STUDY_KEY="k 1"\n'
                 "ADMIN_TOKEN='t=2'\nbroken line\n", encoding='utf-8')
    env = import_export.read_env_file(f)
    assert env == {'APP_URL': 'https://nosetouch-abc.herokuapp.com/', 'STUDY_KEY': 'k 1', 'ADMIN_TOKEN': 't=2'}
    assert import_export._resolve_url(None, env) == 'https://nosetouch-abc.herokuapp.com/api/ingest'
    assert import_export.read_env_file(tmp_path / 'missing.env') == {}


def test_the_sent_at_of_a_backfill_never_beats_a_later_live_upload(client):
    """The phone uploaded the decided copy after the export was made: the backfill must not undo it."""
    pending = make_trial(outcome={'status': 'complete', 'accepted': False, 'participant_flag': 'wrong_hand',
                                  'note': None})
    live = make_trial()
    body = json.dumps({'kind': 'trial', 'study_key': None, 'sent_at': '2026-10-23T12:00:00.000Z', 'record': live})
    assert client.post('/api/ingest', content=body, headers={'Content-Type': 'application/json'}).status_code == 200
    summary = import_export.run_import(small_bundle(trials=[pending], exported_at='2026-10-23T11:00:00.000Z'),
                                       client_poster(client), study_key=None, sleep=lambda s: None)
    assert [r.outcome for r in summary.results] == ['stored', 'kept_newer']
    with client.app.state.db.read() as conn:
        assert conn.execute(select(trials.c.record)).scalar_one() == live
