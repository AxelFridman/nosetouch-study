"""Synthetic records (both generators) validate against the shared contract schema/records.schema.json."""
from __future__ import annotations

import copy

import pytest

from motionlab import schema, synth


def test_schema_is_the_repository_file():
    assert schema.SCHEMA_PATH.name == 'records.schema.json'
    assert schema.SCHEMA_PATH.parent.name == 'schema'
    assert 'TrialRecord' in schema.load_schema()['$defs']


def test_every_synthetic_record_is_valid_full_mode(tiny_dataset):
    for s in tiny_dataset.sessions:
        assert schema.validate(s, 'SessionRecord', mode='full') == []
    for p in tiny_dataset.participants:
        assert schema.validate(p, 'ParticipantProfile', mode='full') == []
    for t in tiny_dataset.trials[:8]:  # full mode is slow on long streams
        assert schema.validate(t, 'TrialRecord', mode='full') == []
    for t in tiny_dataset.trials:
        assert schema.validate(t, 'TrialRecord', mode='fast') == []


def test_synthetic_sessions_use_schema_1_1_fields(tiny_dataset):
    s = tiny_dataset.sessions[0]
    assert s['schema_version'] == synth.SCHEMA_VERSION == '1.2.0'
    assert s['device']['engine'] in ('webkit', 'blink')
    assert s['calibration']['axis_check']['verdict'] == 'w3c'
    assert all(t['simulated'] for t in tiny_dataset.trials)


def test_street_records_are_valid_protocol_2_records(street_dataset):
    """synth's protocol 2.0.0 visits: schema 1.2, the short profile (removed fields written as null / false,
    never omitted), the implicit_v1 calibration, review_mode 'auto' and v2 trials."""
    for s in street_dataset.sessions:
        assert schema.validate(s, 'SessionRecord', mode='full') == []
        assert (s['schema_version'], s['protocol']['protocol_version'], s['consent']['version']) == (
            '1.2.0', '2.0.0', synth.CONSENT_VERSION_V2)
        assert s['settings']['review_mode'] == 'auto' and s['protocol']['preset'] == 'street'
        p = s['participant']
        assert {k: p[k] for k in ('ehi_sf', 'ehi_lq', 'ehi_class', 'phone_model', 'phone_case', 'arm_issue')} == {
            'ehi_sf': None, 'ehi_lq': None, 'ehi_class': None, 'phone_model': None, 'phone_case': None, 'arm_issue': False}
        assert p['nickname'] and p['phone_hand'] in ('left', 'right', 'both')
        cal = s['calibration']
        assert cal['method'] == 'implicit_v1' and cal['flat'] is None and cal['gyro_bias_dps'] is None
        assert cal['axis_check'] is None and cal['upright_check'] == 'ok' and cal['upright']['still']
        assert cal['gravity_sign'] == (-1 if s['device']['engine'] == 'webkit' else 1)
    for p in street_dataset.participants:
        assert schema.validate(p, 'ParticipantProfile', mode='full') == []
    for t in street_dataset.trials:
        assert schema.validate(t, 'TrialRecord', mode='fast') == [], t['trial_id']
    for t in street_dataset.trials[:3]:
        assert schema.validate(t, 'TrialRecord', mode='full') == []
    t = next(x for x in street_dataset.trials if x['outcome']['status'] == 'complete' and x['streams']['touches'])
    assert set(t['streams']['touches'][0]) == {'t', 'type', 'n', 'x', 'y', 'radius_x', 'radius_y', 'force',
                                               'rotation_angle', 'x_norm', 'y_norm'}
    assert t['timing']['return_onset_ms'] > t['timing']['end_detected_ms'] and 'return_cap_ms' in t['segmentation']['params']
    assert set(t['streams']) >= {'quaternion_abs', 'orientation_abs', 'magnetometer', 'screen_events'}


def test_export_bundle_is_valid(tiny_dataset):
    bundle = synth.export_bundle(tiny_dataset, session_ids=[tiny_dataset.sessions[0]['session_id']])
    assert schema.detect_def(bundle) == 'ExportBundle'
    assert schema.validate(bundle, mode='fast') == []
    assert len(bundle['trials']) > 0


def test_ts_reference_records_validate_in_python(ts_reference):
    """TrialRecords written by the app's TrialController pass the Python validator too."""
    for rec in ts_reference['records']:
        assert schema.validate(rec, 'TrialRecord', mode='fast') == [], rec['trial_id']


@pytest.mark.parametrize('mutate, needle', [
    (lambda r: r['streams']['motion']['acc_x'].__setitem__(5, 'oops'), 'acc_x'),
    (lambda r: r['streams']['motion']['t'].__setitem__(3, None), 'streams.motion.t'),
    (lambda r: r.pop('condition'), 'condition'),
    (lambda r: r['condition'].__setitem__('hand', 'both'), 'hand'),
    (lambda r: r['segmentation'].__setitem__('method', 'offline_v1'), 'method'),
])
def test_fast_mode_catches_what_full_mode_catches(tiny_dataset, mutate, needle):
    rec = copy.deepcopy(tiny_dataset.trials[0])
    mutate(rec)
    fast = schema.validate(rec, 'TrialRecord', mode='fast')
    full = schema.validate(rec, 'TrialRecord', mode='full')
    assert fast and full
    assert any(needle in e for e in fast), fast


def test_check_raises_with_all_errors(tiny_dataset):
    rec = copy.deepcopy(tiny_dataset.trials[0])
    rec['attempt'] = 0
    with pytest.raises(schema.SchemaValidationError) as ei:
        schema.check(rec)
    assert 'attempt' in str(ei.value)
