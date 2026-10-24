"""Schema 1.2.0 (protocol 2.0.0) records load next to the old ones; optional fields never cost a record.

Every field schema 1.2 added is optional: a record without it is valid and complete, so the loader must
keep it (only a missing REQUIRED key or a wrong value skips a record). The new fields are exposed in the
trial / session tables and kept in the raw record (Dataset.record)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from motionlab import evaluate as E
from motionlab import io as mio
from motionlab import load_dataset, schema
from motionlab.qa import qa_summary

#: Optional keys of schema 1.2 per object (removed by :func:`strip_v12`).
TRIAL_STREAMS_12 = ('quaternion_abs', 'orientation_abs', 'magnetometer', 'screen_events')
TOUCH_12 = ('radius_x', 'radius_y', 'force', 'rotation_angle', 'x_norm', 'y_norm')


def strip_v12(rec: dict) -> dict:
    """The record without ANY optional schema 1.1 / 1.2 field (still schema-valid)."""
    r = copy.deepcopy(rec)
    if r['record_type'] == 'trial':
        r['condition'].pop('hand_source', None)
        for k in ('return_onset_ms', 'return_end_ms'):
            r['timing'].pop(k, None)
        for k in ('n_quaternion_abs', 'n_orientation_abs'):
            r['quality'].pop(k, None)
        r['segmentation']['params'].pop('return_cap_ms', None)
        for k in TRIAL_STREAMS_12:
            r['streams'].pop(k, None)
        if r['streams'].get('orientation'):
            r['streams']['orientation'].pop('compass_accuracy', None)
        r['streams']['touches'] = [{k: v for k, v in t.items() if k not in TOUCH_12} for t in r['streams']['touches']]
    else:
        r['settings'].pop('review_mode', None)
        r['settings']['segmentation_params'].pop('return_cap_ms', None)
        if r.get('calibration'):
            r['calibration'].pop('method', None)
            r['calibration'].pop('axis_check', None)
        r['device'].pop('engine', None)
        r['device'].pop('in_app_browser', None)
        for b in r['protocol']['plan']:
            b.pop('tier', None)
            b.pop('optional', None)
            for t in b['trials']:
                t['condition'].pop('hand_source', None)
    return r


def write(root, sessions, trials):
    root.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(sessions):
        (root / f'session_{i}.json').write_text(json.dumps(s), encoding='utf-8')
    for i, t in enumerate(trials):
        (root / f'trial_{i:03d}.json').write_text(json.dumps(t), encoding='utf-8')
    return root


@pytest.fixture(scope='module')
def rich(street_dataset):
    """One android street session whose first upright trial also carries a magnetometer stream, a screen
    orientation change and iOS-style compass accuracies (every schema 1.2 field at once)."""
    sess = next(s for s in street_dataset.sessions if s['device']['platform'] == 'android')
    trials = [copy.deepcopy(t) for t in street_dataset.trials if t['session_id'] == sess['session_id']]
    t = next(x for x in trials if x['condition']['orientation'] == 'upright_toward' and x['outcome']['accepted'])
    n = len(t['streams']['orientation']['t'])
    t['streams']['orientation']['compass_accuracy'] = [10.0] * n
    t['streams']['magnetometer'] = {'t': [0.0, 100.0, 200.0], 'x': [12.5, 12.6, None], 'y': [-30.1, -30.0, -29.9],
                                    'z': [4.0, 4.1, 4.2], 'frequency_hz': 10.0}
    t['streams']['screen_events'] = [{'t': 150.0, 'type': 'portrait-primary', 'angle': 0.0}]
    for rec in (sess, *trials):
        assert schema.validate(rec, mode='fast') == [], rec.get('trial_id')
    return copy.deepcopy(sess), trials, t['trial_id']


def test_a_schema_1_2_trial_with_every_optional_field(rich, tmp_path):
    sess, trials, tid = rich
    ds = load_dataset(write(tmp_path, [sess], trials))
    assert len(ds.trials) == len(trials) and not (ds.issues['level'] == 'error').any()
    row = ds.trials.set_index('trial_id').loc[tid]
    src = next(t for t in trials if t['trial_id'] == tid)
    tm = src['timing']
    assert row['hand_source'] == 'instructed' and bool(row['return_observed'])
    assert row['return_onset_ms'] == tm['return_onset_ms'] and row['return_end_ms'] == tm['return_end_ms']
    assert row['return_delay_ms'] == pytest.approx(tm['return_onset_ms'] - tm['end_detected_ms'])
    assert row['return_duration_ms'] == pytest.approx(tm['return_end_ms'] - tm['return_onset_ms'])
    assert row['post_end_ms'] == pytest.approx(tm['stop_ms'] - tm['end_detected_ms'])
    first = src['streams']['touches'][0]
    for k in TOUCH_12:
        assert row[f'first_touch_{k}'] == pytest.approx(first[k]), k
    assert row['n_touch_starts'] == 1 and row['touch_radius_max'] == pytest.approx(max(first['radius_x'], first['radius_y']))
    assert bool(row['has_quaternion_abs_stream']) and bool(row['has_orientation_abs_stream'])
    assert bool(row['has_magnetometer_stream']) and row['n_screen_events'] == 1
    assert row['n_quaternion_abs'] == src['quality']['n_quaternion_abs'] > 0
    assert row['block_tier'] == 1 and not bool(row['block_optional']) and row['review_mode'] == 'auto'
    # the streams, as arrays
    st = ds.streams(tid)
    assert st.quaternion_abs.q.shape == (len(src['streams']['quaternion_abs']['t']), 4)
    assert st.quaternion_abs.sensor == 'AbsoluteOrientationSensor'
    assert st.orientation_abs is not None and st.orientation_abs.absolute is True
    np.testing.assert_array_equal(st.orientation.compass_accuracy, np.full(len(st.orientation.t), 10.0))
    assert st.magnetometer.xyz.shape == (3, 3) and np.isnan(st.magnetometer.xyz[2, 0])
    assert st.screen_events.to_dict('records') == [{'t': 150.0, 'type': 'portrait-primary', 'angle': 0.0}]
    assert list(st.touches.columns) == list(mio.TOUCH_COLUMNS) and st.touches['radius_x'].notna().all()
    # ... and kept verbatim in the raw record (the return of the arm stays in the motion stream)
    assert ds.record(tid) == src
    assert ds.streams(tid).t[-1] == tm['stop_ms'] > tm['return_onset_ms']


def test_records_without_the_optional_fields_are_never_dropped(rich, tmp_path):
    sess, trials, tid = rich
    bare_s, bare_t = strip_v12(sess), [strip_v12(t) for t in trials]
    for v, root in (('1.2.0', tmp_path / 'v12'), ('1.1.0', tmp_path / 'v11'), ('1.0.0', tmp_path / 'v10')):
        ss = {**bare_s, 'schema_version': v}
        tt = [{**t, 'schema_version': v} for t in bare_t]
        assert schema.validate(ss, mode='full') == []
        assert all(schema.validate(t, mode='fast') == [] for t in tt)
        ds = load_dataset(write(root, [ss], tt), rate_mapping='calibration')
        assert len(ds.trials) == len(trials) and len(ds.sessions) == 1, v
        assert not (ds.issues['level'] == 'error').any(), ds.issues
        row = ds.trials.set_index('trial_id').loc[tid]
        assert row['hand_source'] == 'instructed'  # absent = instructed
        assert pd.isna(row['return_onset_ms']) and pd.isna(row['return_observed']) and pd.isna(row['first_touch_force'])
        assert not bool(row['has_quaternion_abs_stream']) and row['n_screen_events'] == 0
        st = ds.streams(tid)
        assert st.quaternion_abs is None and st.magnetometer is None and st.orientation.compass_accuracy is None
        assert st.touches['radius_x'].isna().all()
        srow = ds.sessions.iloc[0]
        assert srow['review_mode'] == 'tap' and srow['calibration_method'] == 'guided_v1'  # absent method = guided
        assert pd.isna(row['block_tier']) and not bool(row['block_optional'])
    # the choice trial without hand_source: an instructed trial of a 'choice' block
    choice = ds.trials[ds.trials['block_kind'] == 'choice']
    assert len(choice) >= 1 and set(choice['hand_source']) == {'instructed'}


def test_inconsistent_optional_streams_are_ignored_not_fatal(rich, tmp_path):
    sess, trials, tid = rich
    bad = copy.deepcopy(trials)
    t = next(x for x in bad if x['trial_id'] == tid)
    t['streams']['quaternion_abs']['x'] = t['streams']['quaternion_abs']['x'][:-3]  # x shorter than t
    t['streams']['orientation']['compass_accuracy'] = [1.0, 2.0]
    assert schema.validate(t, mode='fast') == []  # the schema does not tie lengths together
    ds = load_dataset(write(tmp_path, [sess], bad))
    assert tid in set(ds.trials['trial_id'])
    warn = ds.issues[(ds.issues['record_id'] == tid) & (ds.issues['level'] == 'warning')]['message'].tolist()
    assert any('quaternion_abs.x ignored' in m for m in warn) and any('compass_accuracy ignored' in m for m in warn)
    st = ds.streams(tid)
    assert st.quaternion_abs is None and st.orientation.compass_accuracy is None and st.n > 0
    # a REQUIRED stream column out of step is still unusable: that trial is skipped (and listed)
    t['streams']['motion']['acc_x'] = t['streams']['motion']['acc_x'][:-1]
    with pytest.warns(UserWarning, match='record/file error'):
        ds = load_dataset(write(tmp_path / 'b', [sess], bad))
    assert tid not in set(ds.trials['trial_id'])
    assert ds.issues['message'].str.contains('unusable streams').any()


def test_a_missing_required_key_still_skips_the_record(rich, tmp_path):
    """Why the v2 app writes removed profile fields as null: omitting a required key invalidates the record."""
    sess, trials, _ = rich
    s = copy.deepcopy(sess)
    del s['participant']['ehi_sf']
    with pytest.warns(UserWarning, match='record/file error'):
        ds = load_dataset(write(tmp_path, [s], trials))
    assert len(ds.sessions) == 0 and ds.issues['message'].str.contains('ehi_sf').any()


@pytest.mark.parametrize('first_loaded', ['older', 'newer'])
def test_duplicate_trial_copies_keep_the_latest_state_of_the_app(rich, tmp_path, first_loaded):
    """The app re-saves an attempt under the same trial_id and created_at; the backend keeps the last one,
    but a phone export plus a database dump can hold two copies. The later state wins in either file order:
    hand_not_reported < reported choice trial; auto_accept < redo_requested; review_interrupted < decided."""
    sess, trials, tid = rich
    choice = next(t for t in trials if t['condition'].get('hand_source') == 'reported')
    plain = next(t for t in trials if t['trial_id'] == tid)
    pairs = []
    pending = copy.deepcopy(choice)
    pending['condition']['hand'] = 'left' if choice['condition']['hand'] == 'right' else 'right'  # the placeholder
    pending['outcome'].update(accepted=False, note=mio.HAND_NOT_REPORTED_NOTE)
    pairs.append((pending, choice))
    revoked = copy.deepcopy(plain)
    revoked['outcome'].update(accepted=False, note=mio.REDO_REQUESTED_NOTE, participant_flag='wrong_hand')
    pairs.append((plain, revoked))
    v1_pending = copy.deepcopy(trials[-1])
    v1_pending['outcome'].update(accepted=False, note=mio.REVIEW_INTERRUPTED_NOTE)
    pairs.append((v1_pending, trials[-1]))
    for k, (older, newer) in enumerate(pairs):
        a, b = (older, newer) if first_loaded == 'older' else (newer, older)
        d = tmp_path / f'pair{k}'
        write(d / 'a', [sess], [a])
        write(d / 'b', [], [b])
        ds = load_dataset(d)
        row = ds.trials.set_index('trial_id').loc[newer['trial_id']]
        assert (bool(row['accepted']), row['note'], row['hand']) == (
            newer['outcome']['accepted'], newer['outcome']['note'], newer['condition']['hand']), k
        assert len(ds.duplicates) == 1


def test_street_tables(street_dataset, street_ds):
    ds = street_ds
    tr, ss = ds.trials, ds.sessions
    assert len(tr) == len(street_dataset.trials) and len(ss) == 4
    assert set(ss['preset']) == {'street'} and set(ss['protocol_version']) == {'2.0.0'}
    assert sorted(ss['status']) == ['completed', 'completed', 'completed', 'ended_early']
    assert set(ss['review_mode']) == {'auto'} and set(ss['n_planned_trials']) == {39}
    assert set(ss['n_planned_required']) == {29} and set(ss['n_optional_blocks']) == {4}
    assert set(ss['estimated_minutes']) == {5.0}
    assert set(ss.loc[ss['platform'] == 'ios', 'device_model']) == {'iPhone 390x844@3'}
    for table in (ss, tr, ds.participants):  # names stay in the raw record
        assert 'nickname' not in table.columns and not table.isin(['Synthetic ' + p for p in ss['participant_id']]).any().any()
    assert ss['ehi_class'].isna().all() and ss['handedness_self'].notna().all()
    # choice trials: reported hand, tier 0
    ch = tr[tr['hand_source'] == 'reported']
    assert len(ch) == 4 and set(ch['block_kind']) == {'choice'} and set(ch['block_tier']) == {0}
    assert sorted(ch['note'].fillna('-')) == ['-', '-', 'hand_unsure', 'hand_unsure']
    assert set(tr.loc[tr['hand_source'] == 'instructed', 'block_kind']) == {'main', 'odd'}
    assert set(tr['block_tier']) == {0, 1, 2, 3, 4} and tr.loc[tr['block_tier'] == 4, 'block_optional'].astype(bool).all()
    done = tr[tr['status'] == 'complete']
    assert done['return_observed'].astype(bool).all() and (done['return_delay_ms'] > 0).all()
    assert set(tr.loc[tr['has_quaternion_abs_stream'].astype(bool), 'platform']) == {'android'}
    assert tr['hand_is_phone_hand'].notna().any() and set(tr['note'].dropna()) <= {
        'auto_accept', 'redo_requested', 'hand_unsure'}
    # the analysis set: no redo, no "not sure", no failure
    keep, _ = E.analysis_set(tr)
    assert set(tr.loc[keep, 'note'].fillna('auto_accept')) == {'auto_accept'} and keep.sum() == int(tr['accepted'].sum())


def test_qa_reports_the_protocol_2_fields(street_ds):
    rep = qa_summary(street_ds, check_online=True, offline=False, consistency_sample=4, gravity_sample=8)
    s = rep.summary
    assert s['calibration methods'] == 'implicit_v1 4'
    assert s['gravity sign sources'] == 'engine_vote 4' and s['sessions with uncertain gravity sign'] == 0
    assert s['rotationRate mapping sources'] == 'closure w3c/deg_per_s 4'
    assert s['sessions with a rotationRate closure check'] == 4 and s['closure check: decisive sessions'] == 4
    assert s['sessions with a trusted calibration gyro bias'] == 0 and s['sessions ended early'] == 1
    # 4 choice trials, 2 answered "not sure": only the 2 answered ones carry a reported hand
    assert s['choice trials'] == 4 and s['choice trials (hand reported)'] == 2 and s['choice trials accepted'] == 2
    assert s['schema 1.2 trial fields present'] is True
    assert s['online_v1 port == stored (all trials)'] == 1.0
    f = rep.tables['schema_1_2_fields']
    assert set(f.index) == {'android', 'ios'} and f.loc['android', 'quaternion_abs_rate'] == 1.0
    assert f.loc['ios', 'quaternion_abs_rate'] == 0.0 and (f['return_observed_rate'] > 0.9).all()
    assert len(rep.tables['normalisation']) == 4 and 'sessions_by_status' in rep.tables


def test_the_bundle_of_a_dataset_is_schema_1_2_and_reloads(street_ds, tmp_path):
    ds = street_ds
    b = ds.to_bundle()
    assert b['schema_version'] == '1.2.0' and schema.validate(b, 'ExportBundle', mode='fast') == []
    p = tmp_path / 'nosetouch_export.json'
    p.write_text(json.dumps(b), encoding='utf-8')
    again = load_dataset(p, rate_mapping='calibration')
    assert len(again.trials) == len(ds.trials) and set(again.sessions['calibration_method']) == {'implicit_v1'}


def test_device_model():
    assert mio.device_model('android', 'ALI-NX1') == 'ALI-NX1'
    assert mio.device_model('ios', 'iPhone', None, 390, 844, 3) == 'iPhone 390x844@3'
    assert mio.device_model('ios', None, None, 844, 390, 3) == 'iPhone 390x844@3'  # landscape capture
    assert mio.device_model('android', 'K', None, 412, 915, 2.625) == 'android 412x915@2.625'  # reduced UA
    assert mio.device_model('ios', 'iPhone', 'iPhone 13 mini') == 'iPhone 13 mini'  # v1 self-report
    assert mio.device_model('ios', 'iPhone') == 'iPhone' and mio.device_model(None, None) is None


def test_trials_written_by_the_app_load_with_the_return_of_the_arm(ts_reference, tmp_path):
    """The TypeScript reference records (fixtures/make_ts_reference.mts: the app's own TrialController,
    protocol 2.0.0) record the return of the arm after the end cue; they load (as orphans: no session)."""
    recs = ts_reference['records']
    ds = load_dataset(write(tmp_path, [], recs))
    assert len(ds.trials) == len(recs)
    tr = ds.trials
    done = tr['status'] == 'complete'
    assert done.any() and tr.loc[done, 'return_observed'].astype(bool).all()
    post_roll = recs[0]['segmentation']['params']['post_roll_ms']
    assert (tr.loc[done, 'post_end_ms'] >= post_roll).all() and (tr.loc[done, 'return_delay_ms'] > 0).all()
    assert tr.loc[~done, 'return_onset_ms'].isna().all()
