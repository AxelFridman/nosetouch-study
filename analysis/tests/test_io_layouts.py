"""Loading: the backend folder tree, export bundles (also re-shared as .json.txt and inside zips),
upload envelopes, any mix of them, and the deduplication rules of docs/02 section 10."""
from __future__ import annotations

import copy
import json
import warnings
import zipfile

import numpy as np
import pytest

from motionlab import load_dataset, synth
from motionlab.schema import SchemaValidationError


def _counts(ds):
    return len(ds.participants), len(ds.sessions), len(ds.trials)


def test_folder_tree_layout(tiny_dataset, tiny_tree):
    # <root>/<study>/<participant>/<session>/{session.json, trial_*.json}
    files = sorted(p.relative_to(tiny_tree).as_posix() for p in tiny_tree.rglob('*.json'))
    assert any(f.endswith('/session.json') and f.count('/') == 3 for f in files)
    assert any('/trial_000_a1_' in f for f in files)
    ds = load_dataset(tiny_tree)
    assert _counts(ds) == (len(tiny_dataset.participants), len(tiny_dataset.sessions), len(tiny_dataset.trials))
    assert len(ds.issues[ds.issues['level'] == 'error']) == 0
    assert set(ds.trials['trial_id']) == {t['trial_id'] for t in tiny_dataset.trials}
    # joined participant / session / normalisation columns
    for col in ('hand', 'orientation', 'platform', 'engine', 'gravity_sign', 'gravity_sign_source', 'rr_mapping',
                'y_right', 'hand_is_dominant', 'is_last_attempt', 'ehi_class', 'block_kind'):
        assert col in ds.trials.columns, col
    assert set(ds.trials['gravity_sign_source']) == {'calibration'}
    ios = ds.trials['platform'] == 'ios'
    assert (ds.trials.loc[ios, 'gravity_sign'] == -1).all() and (ds.trials.loc[~ios, 'gravity_sign'] == 1).all()


def test_export_bundle_and_mix_dedupe(tiny_dataset, tiny_tree, tmp_path):
    bundle = synth.export_bundle(tiny_dataset)
    p = tmp_path / 'nosetouch_all_20261120-1530.json'
    p.write_text(json.dumps(bundle), encoding='utf-8')
    a = load_dataset(p)
    assert _counts(a) == (len(tiny_dataset.participants), len(tiny_dataset.sessions), len(tiny_dataset.trials))
    # the same records twice (tree + bundle): every copy is dropped, nothing doubles
    both = load_dataset([tiny_tree, p])
    assert _counts(both) == _counts(a)
    assert both.trials['trial_id'].is_unique
    n_dup = both.duplicates
    assert (n_dup['kind'] == 'trial').sum() == len(tiny_dataset.trials)
    assert (n_dup['kind'] == 'session').sum() == len(tiny_dataset.sessions)


def test_json_txt_zip_and_envelopes(tiny_dataset, tmp_path):
    sid = tiny_dataset.sessions[0]['session_id']
    bundle = synth.export_bundle(tiny_dataset, session_ids=[sid])
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'nosetouch_P_20261120-1530.json.txt').write_text(json.dumps(bundle), encoding='utf-8')
    ds = load_dataset(tmp_path / 'a')
    assert len(ds.trials) == len(bundle['trials'])
    zpath = tmp_path / 'drive.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        zf.writestr('study/export.json', json.dumps(bundle))
    assert len(load_dataset(zpath).trials) == len(bundle['trials'])
    env_dir = tmp_path / 'env'
    env_dir.mkdir()
    for k, t in enumerate(bundle['trials'][:3]):
        env = {'kind': 'trial', 'study_key': None, 'sent_at': '2026-11-20T10:00:00.000Z', 'record': t}
        (env_dir / f'env{k}.json').write_text(json.dumps(env), encoding='utf-8')
    (env_dir / 'session.json').write_text(json.dumps(
        {'kind': 'session', 'study_key': 'k', 'sent_at': '2026-11-20T10:00:00.000Z', 'record': bundle['sessions'][0]}), encoding='utf-8')
    (env_dir / 'config.json').write_text(json.dumps({'study_code': 'x'}), encoding='utf-8')  # not a record: skipped
    e = load_dataset(env_dir)
    assert len(e.trials) == 3 and len(e.sessions) == 1
    assert (e.issues['message'] == 'not a nose-touch record; skipped').any()


def test_newest_session_snapshot_wins(tiny_dataset, tmp_path):
    s = tiny_dataset.sessions[0]
    old = copy.deepcopy(s)
    new = copy.deepcopy(s)
    old['updated_at'] = '2026-11-01T10:00:00.000Z'
    old['progress']['n_accepted'] = 1
    new['updated_at'] = '2026-11-02T10:00:00.000Z'
    for name, rec in (('b_old.json', old), ('a_new.json', new)):
        (tmp_path / name).write_text(json.dumps(rec), encoding='utf-8')
    ds = load_dataset(tmp_path)
    assert len(ds.sessions) == 1
    assert ds.session_record(s['session_id'])['updated_at'] == new['updated_at']
    assert len(ds.duplicates) == 1


@pytest.mark.parametrize('pending_name, decided_name', [('a_pending.json', 'b_decided.json'), ('b_pending.json', 'a_decided.json')])
def test_decided_trial_copy_beats_the_pre_review_copy(tiny_dataset, tmp_path, pending_name, decided_name):
    # the app saves a complete attempt before its review and again after the decision, same created_at
    t = tiny_dataset.trials[0]
    s = next(x for x in tiny_dataset.sessions if x['session_id'] == t['session_id'])
    pending = copy.deepcopy(t)
    pending['outcome'] = {**pending['outcome'], 'accepted': False, 'note': 'review_interrupted'}
    decided = copy.deepcopy(t)
    decided['outcome'] = {**decided['outcome'], 'accepted': True, 'note': None}
    (tmp_path / 'session.json').write_text(json.dumps(s), encoding='utf-8')
    (tmp_path / pending_name).write_text(json.dumps(pending), encoding='utf-8')
    (tmp_path / decided_name).write_text(json.dumps(decided), encoding='utf-8')
    ds = load_dataset(tmp_path)
    rec = ds.record(t['trial_id'])
    assert rec['outcome']['accepted'] is True and rec['outcome']['note'] is None
    assert len(ds.duplicates) == 1


def test_same_attempt_key_with_two_trial_ids_is_flagged(tiny_dataset, tmp_path):
    t = tiny_dataset.trials[0]
    twin = copy.deepcopy(t)
    twin['trial_id'] = 'twin-' + t['trial_id'][:20]
    bundle = synth.export_bundle(tiny_dataset, session_ids=[t['session_id']])
    bundle['trials'].append(twin)
    (tmp_path / 'b.json').write_text(json.dumps(bundle), encoding='utf-8')
    ds = load_dataset(tmp_path)
    flagged = ds.trials[ds.trials['dup_attempt_key'].astype(bool)]
    assert set(flagged['trial_id']) == {t['trial_id'], twin['trial_id']}
    assert ds.trials['multi_accepted'].astype(bool).sum() == 2
    assert ds.issues['message'].str.contains('different trial ids').any()


def test_invalid_records_skip_keep_raise(tiny_dataset, tmp_path):
    bad = copy.deepcopy(tiny_dataset.trials[0])
    bad['condition']['hand'] = 'both'
    (tmp_path / 'bad.json').write_text(json.dumps(bad), encoding='utf-8')
    (tmp_path / 'broken.json').write_text('{not json', encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds = load_dataset(tmp_path)
        assert len(ds.trials) == 0
        assert (ds.issues['level'] == 'error').sum() == 2
        kept = load_dataset(tmp_path, on_invalid='keep')
        assert len(kept.trials) == 1
    with pytest.raises(SchemaValidationError):
        load_dataset(tmp_path / 'bad.json', on_invalid='raise')


def test_record_round_trip_and_lazy_streams(tiny_dataset, tiny_tree):
    ds = load_dataset(tiny_tree, keep_streams=False)
    t = tiny_dataset.trials[3]
    rec = ds.record(t['trial_id'])
    assert json.dumps(rec, sort_keys=True) == json.dumps(t, sort_keys=True)
    s = ds.streams(t['trial_id'])
    assert s.n == len(t['streams']['motion']['t'])
    np.testing.assert_array_equal(s.gyro(), s.rr)  # W3C: (alpha, beta, gamma) = (x, y, z)


def test_simulated_filter(tiny_tree):
    assert len(load_dataset(tiny_tree, simulated='exclude').trials) == 0
    assert len(load_dataset(tiny_tree, simulated='only').trials) > 0
