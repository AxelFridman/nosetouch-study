"""norm_v1: the W3C rotationRate mapping, gravity-sign resolution, the legacy verdict, and the copies of
the TypeScript constants (docs/03 section 5.6 lists the tests the normalisation must pass)."""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import re

import numpy as np
import pytest
from _mlhelpers import REPO, condition

from motionlab import conventions as C
from motionlab import evaluate as E
from motionlab import features, load_dataset, models, synth
from motionlab import io as motionlab_io
from motionlab import schema as S
from motionlab.qa import qa_summary

CONSTANTS_TS = (REPO / 'app' / 'src' / 'core' / 'constants.ts').read_text(encoding='utf-8')
PROTOCOL_TS = (REPO / 'app' / 'src' / 'core' / 'protocol.ts').read_text(encoding='utf-8')


def _ts_block(name: str, source: str = CONSTANTS_TS) -> str:
    """Body of the object literal declared as `const <name> ... = { ... };`, without line comments."""
    m = re.search(r'\bconst\s+' + name + r'\b[^=]*=\s*\{(.*?)\n\}(?: as const)?;', source, re.S)
    assert m, f'{name} not found'
    return re.sub(r'//[^\n]*', '', m.group(1))


def _ts_numbers(block: str) -> dict[str, float]:
    return {k: float(v) for k, v in re.findall(r'(\w+):\s*([-\d.]+)\s*[,}\n]', block)}


def _ts_nested_numbers(block: str) -> dict[str, dict[str, float]]:
    """{outer: {inner: number}} from a TS object literal of object literals (one nesting level)."""
    return {k: _ts_numbers(inner + '\n') for k, inner in re.findall(r'(\w+):\s*\{([^{}]*)\}', block)}


# ───────────────────────────── drift against the TypeScript contract ─────────────────────────────


def test_default_segmentation_params_match_constants_ts():
    ts = _ts_numbers(_ts_block('DEFAULT_SEGMENTATION_PARAMS'))
    assert ts == {k: float(v) for k, v in C.DEFAULT_SEGMENTATION_PARAMS.items()}
    assert ts['still_gyro_dps'] == 20 and ts['post_roll_ms'] == 500 and ts['return_cap_ms'] == 3000  # protocol 2.0.0


def test_segmentation_param_keys_follow_the_schema():
    """online_v1 needs exactly the required keys (present in every record since schema 1.0); the optional
    schema 1.2 keys (return_cap_ms) are for the trial controller, so old records still re-segment."""
    sp = S.load_schema()['$defs']['SegmentationParams']
    assert set(C.SEGMENTATION_PARAM_KEYS) == set(sp['required'])
    assert set(C.OPTIONAL_SEGMENTATION_PARAM_KEYS) == set(sp['properties']) - set(sp['required'])
    assert set(C.DEFAULT_SEGMENTATION_PARAMS) == set(sp['properties'])


def test_parameter_overrides_match_protocol_ts():
    assert _ts_nested_numbers(_ts_block('SPEED_OVERRIDES', PROTOCOL_TS)) == C._SPEED_OVERRIDES
    assert _ts_nested_numbers(_ts_block('VARIANT_OVERRIDES', PROTOCOL_TS)) == C._VARIANT_OVERRIDES
    tune = dict(re.findall(r"(\w+):\s*'(min|max)'", _ts_block('TUNE_DIRECTION', PROTOCOL_TS)))
    assert tune == C._TUNE_DIRECTION
    # protocol 2.0.0: slow onset floors above the still thresholds (no false start from a still sway)
    slow = C._SPEED_OVERRIDES['slow']
    assert slow['onset_gyro_min_dps'] > C.DEFAULT_SEGMENTATION_PARAMS['still_gyro_dps']
    assert slow['onset_acc_min_ms2'] > C.DEFAULT_SEGMENTATION_PARAMS['still_acc_ms2']


def test_enums_and_orientation_lists_match_the_contract():
    defs = S.load_schema()['$defs']
    assert C.VARIANTS == tuple(defs['Variant']['enum']) and 'tilted_touch' in C.ODD_VARIANTS
    assert C.PROTOCOL_PRESETS == tuple(defs['ProtocolPreset']['enum'])
    assert C.SPEEDS == tuple(defs['Speed']['enum']) and C.START_POSITIONS == tuple(defs['StartPosition']['enum'])
    assert C.ALL_ORIENTATIONS == tuple(defs['OrientationId']['enum'])
    assert C.COLLECTED_ORIENTATIONS == (
        'upright_toward', 'top_left_toward', 'top_right_toward', 'upside_down_toward', 'flat_up', 'free')
    assert re.search(r"COLLECTED_ORIENTATIONS[^=]*=\s*ALL_ORIENTATIONS\.filter\(\(o\) => ORIENTATIONS\[o\]\.screen !== 'away'\)",
                     CONSTANTS_TS)


def test_versions_match_constants_ts():
    assert re.search(r"SCHEMA_VERSION = '([\d.]+)'", CONSTANTS_TS).group(1) == C.SCHEMA_VERSION == '1.2.0'
    assert re.search(r"PROTOCOL_VERSION = '([\d.]+)'", CONSTANTS_TS).group(1) == C.APP_PROTOCOL_VERSION == '2.0.0'
    assert synth.SCHEMA_VERSION == C.SCHEMA_VERSION
    assert motionlab_io.SCHEMA_VERSION == C.SCHEMA_VERSION  # Dataset.to_bundle


def test_orientation_table_matches_constants_ts():
    rows = re.findall(r"(\w+): \{ rotation: (?:'(\w+)'|null), screen: (?:'(\w+)'|null), mirror: '(\w+)' \}",
                      _ts_block('ORIENTATIONS'))
    ts = {o: {'rotation': r or None, 'screen': s or None, 'mirror': m} for o, r, s, m in rows}
    assert ts == C.ORIENTATIONS


def test_rounding_gravity_and_schema_version_match():
    ts = {k: int(v) for k, v in re.findall(r'(\w+):\s*(\d+),', _ts_block('ROUNDING'))}
    assert ts == C.ROUNDING
    assert float(re.search(r'GRAVITY = ([\d.]+);', CONSTANTS_TS).group(1)) == C.GRAVITY
    assert re.search(r"SCHEMA_VERSION = '([\d.]+)'", CONSTANTS_TS).group(1) == synth.SCHEMA_VERSION


def test_grip_matrices_are_rotations_and_mirror_consistent():
    S = C.MIRROR_X
    for o, R in C.GRIP_MATRICES.items():
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        # M·R_grip(o)·S = R_grip(mirror(o)) (docs/05 section 3.2)
        assert np.allclose(S @ R @ S, C.GRIP_MATRICES[C.mirror_orientation(o)]), o


def test_resolve_params_composes_like_protocol_ts():
    p = C.resolve_params(C.DEFAULT_SEGMENTATION_PARAMS, {'speed': 'slow', 'variant': 'pause'})
    assert p['end_sustain_ms'] == 1800 and p['max_movement_ms'] == 12000 and p['onset_gyro_min_dps'] == 25
    assert p['onset_acc_min_ms2'] == 0.55 and p['return_cap_ms'] == 3000  # keys no override names are copied
    assert C.resolve_params(C.DEFAULT_SEGMENTATION_PARAMS, {'speed': 'natural', 'variant': 'tilted_touch'}) == \
        C.DEFAULT_SEGMENTATION_PARAMS
    p = C.resolve_params({**C.DEFAULT_SEGMENTATION_PARAMS, 'onset_sustain_ms': 30}, {'speed': 'fast', 'variant': 'jerky'})
    assert p['onset_sustain_ms'] == 30  # 'min' direction: an override never loosens a stricter base
    assert p['end_sustain_ms'] == 600


# ───────────────────────────── rotationRate mapping ─────────────────────────────


def test_w3c_mapping_is_alpha_beta_gamma_equals_xyz():
    rr = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    np.testing.assert_array_equal(C.rotation_rate_to_device(rr), rr)
    np.testing.assert_array_equal(C.rotation_rate_to_device(rr, 'w3c'), rr)
    np.testing.assert_array_equal(C.rotation_rate_to_device(rr, 'unclear'), rr)
    assert C.rr_fields_for_device_axes('w3c') == ('rr_alpha', 'rr_beta', 'rr_gamma')


def test_legacy_and_inverted_verdicts():
    rr = np.array([10.0, 20.0, 30.0])  # alpha, beta, gamma
    # legacy: rr_beta -> w_x, rr_gamma -> w_y, rr_alpha -> w_z (docs/03 section 5.6, test 4)
    np.testing.assert_array_equal(C.rotation_rate_to_device(rr, 'legacy'), [20.0, 30.0, 10.0])
    np.testing.assert_array_equal(C.rotation_rate_to_device(rr, 'w3c_inverted'), [-10.0, -20.0, -30.0])
    np.testing.assert_array_equal(C.rotation_rate_to_device(rr, 'legacy_inverted'), [-20.0, -30.0, -10.0])
    np.testing.assert_allclose(C.rotation_rate_to_device(rr, 'w3c', 'rad_per_s'), rr * 180 / math.pi)
    for v in C.RR_MAPPINGS:
        for u in ('deg_per_s', 'rad_per_s'):
            np.testing.assert_allclose(C.device_to_rotation_rate(C.rotation_rate_to_device(rr, v, u), v, u), rr)


def test_rate_mapping_from_calibration():
    m = C.rate_mapping_from_calibration(None)
    assert (m.verdict, m.units, m.verified) == ('w3c', 'deg_per_s', False)
    assert not C.rate_mapping_from_calibration({'axis_check': None}).verified
    m = C.rate_mapping_from_calibration({'axis_check': {'verdict': 'legacy', 'units': 'deg_per_s'}})
    assert (m.verdict, m.units, m.verified) == ('legacy', 'deg_per_s', True)
    m = C.rate_mapping_from_calibration({'axis_check': {'verdict': 'w3c', 'units': 'rad_per_s'}})
    assert (m.verdict, m.units) == ('w3c', 'rad_per_s')
    m = C.rate_mapping_from_calibration({'axis_check': {'verdict': 'unclear', 'units': 'rad_per_s'}})
    assert (m.verdict, m.units, m.verified) == ('w3c', 'deg_per_s', False) and 'axis_unverified' in m.detail


# ───────────────────────────── gravity sign ─────────────────────────────


def test_face_up_reads_plus_g_after_normalisation():
    # docs/03 section 5.6, test 1: a face-up still window gives f ~ (0, 0, +9.81) on both platforms
    for raw_z, sign in ((-9.81, -1), (9.81, 1)):
        f = C.normalize_acceleration([[0.0, 0.0, raw_z]], sign)
        np.testing.assert_allclose(f, [[0.0, 0.0, 9.81]])
    with pytest.raises(ValueError):
        C.normalize_acceleration([[0, 0, 1]], 0)


def test_gravity_sign_resolution_order():
    cal = {'gravity_sign': -1, 'upright_check': 'ok'}
    g = C.resolve_gravity_sign(cal, [-1] * 6, engine='webkit')
    assert (g.sign, g.source, g.warning) == (-1, 'calibration', False)
    # calibration contradicted by the engine or by the votes, or upright_check inverted -> flagged
    assert C.resolve_gravity_sign(cal, [], engine='blink').warning
    assert C.resolve_gravity_sign(cal, [1] * 6).warning
    assert C.resolve_gravity_sign({'gravity_sign': 1, 'upright_check': 'inverted'}).warning
    # no calibrated sign: engine expectation (flagged): webkit -1, blink / gecko +1
    for engine, sign in (('webkit', -1), ('blink', 1), ('gecko', 1)):
        g = C.resolve_gravity_sign({'gravity_sign': None}, [], engine=engine)
        assert (g.sign, g.source, g.warning) == (sign, 'engine', True)
    # default order (docs/03 section 5.2): a confident trial vote beats the engine prior, flagged
    g = C.resolve_gravity_sign(None, [1, 1, 1, 1], engine='webkit')
    assert (g.sign, g.source, g.warning) == (1, 'inferred', True)
    # engine-first order is still available
    g = C.resolve_gravity_sign(None, [1, 1, 1, 1], engine='webkit', order=('engine', 'vote'))
    assert g.source == 'engine' and 'CONTRADICTED' in g.detail
    # unknown engine: trial votes, then +1 default
    g = C.resolve_gravity_sign(None, [-1, -1, -1, 1, -1], engine=None)
    assert (g.sign, g.source) == (-1, 'inferred')
    g = C.resolve_gravity_sign(None, [1, -1, None], engine=None)
    assert (g.sign, g.source, g.warning, g.usable_for_chirality) == (1, 'default', True, False)


def test_trial_votes_override_a_calibrated_sign_they_clearly_contradict():
    # An Android phone calibrated screen-down: sign -1, upright_check inverted, every trial votes +1.
    face_down = {'gravity_sign': -1, 'upright_check': 'inverted'}
    g = C.resolve_gravity_sign(face_down, [1] * 12, engine='blink')
    assert (g.sign, g.source, g.warning) == (1, 'inferred', True)
    assert (g.votes_for, g.votes_against) == (12, 0)  # counted for / against the RESOLVED sign
    assert g.detail.startswith('calibration sign overridden by the trial votes: ')
    assert '12/12' in g.detail and "upright_check 'inverted'" in g.detail and 'blink engine expectation' in g.detail
    # thresholds: >= 5 votes and >= 80 % against
    g = C.resolve_gravity_sign({'gravity_sign': -1}, [1, 1, 1, 1, -1])
    assert (g.sign, g.source, g.votes_for, g.votes_against) == (1, 'inferred', 4, 1)
    assert C.resolve_gravity_sign({'gravity_sign': -1}, [1, 1, 1, 1]).source == 'calibration'  # 4 votes
    assert C.resolve_gravity_sign({'gravity_sign': -1}, [1, 1, 1, -1, 1, 1, -1, 1]).sign == -1  # 75 % against
    # agreeing votes never override; infer=False keeps the calibrated sign (flagged)
    assert C.resolve_gravity_sign(face_down, [-1] * 12, engine='blink').source == 'calibration'
    g = C.resolve_gravity_sign(face_down, [1] * 12, engine='blink', infer=False)
    assert (g.sign, g.source, g.warning) == (-1, 'calibration', True) and 'contradicts the trial baselines' in g.detail
    # votes against both the calibration and the engine: still the votes, and the engine is named
    g = C.resolve_gravity_sign({'gravity_sign': -1, 'upright_check': 'ok'}, [1] * 6, engine='webkit')
    assert (g.sign, g.source) == (1, 'inferred') and 'differ from the webkit engine expectation' in g.detail


def test_calibration_gyro_bias_needs_a_still_flat_window_and_a_small_norm():
    base = {'gravity_sign': 1, 'flat': {'still': True}, 'gyro_bias_dps': {'alpha': 0.4, 'beta': -0.3, 'gamma': 0.2},
            'axis_check': {'verdict': 'w3c', 'units': 'deg_per_s'}}

    def bias(**cal):
        return C.session_normalization({'calibration': {**base, **cal}}).gyro_bias

    assert bias() == pytest.approx((0.4, -0.3, 0.2))
    # a moving flat window (earlier app versions stored its mean rate anyway) or no flat stats: not a bias
    assert bias(flat={'still': False}) is None
    assert bias(flat=None) is None
    assert bias(gyro_bias_dps=None) is None
    assert bias(gyro_bias_dps={'alpha': 0.1, 'beta': None, 'gamma': 0.0}) is None
    # >= 3 deg/s (the pre-GO limit) is motion, not bias; the norm is taken after the rate mapping
    assert bias(gyro_bias_dps={'alpha': 2.5, 'beta': -2.0, 'gamma': 0.0}) is None  # |b| = 3.2 deg/s
    assert bias(gyro_bias_dps={'alpha': 2.0, 'beta': -2.0, 'gamma': 0.0}) == pytest.approx((2.0, -2.0, 0.0))
    legacy_rad = {'verdict': 'legacy', 'units': 'rad_per_s'}
    assert bias(gyro_bias_dps={'alpha': 0.01, 'beta': 0.0, 'gamma': 0.0}, axis_check=legacy_rad) == pytest.approx(
        (0.0, 0.0, 0.01 * 180 / math.pi))  # legacy: alpha is the z rate; 0.57 deg/s
    assert bias(gyro_bias_dps={'alpha': 0.06, 'beta': 0.0, 'gamma': 0.0}, axis_check=legacy_rad) is None  # 3.4 deg/s


def test_a_moving_calibration_bias_is_not_subtracted_from_the_trials(symmetric_participant):
    """A hand on the phone during the flat step: the stored 'bias' is motion; prepare_trial must not subtract
    it from every trial. norm_v2 subtracts nothing then (a hand-held pre-GO mean is tremor, not bias); the
    norm_v1 pre-GO estimate is still available on request."""
    p = synth.default_platform('android', None, gyro_bias_dps=(0.15, -0.1, 0.05))
    rec = synth.simulate_trial(symmetric_participant, p, condition('right', 'upright_toward'), seed=3, noise=1.0)[0]
    moving = {'gravity_sign': None, 'upright_check': 'unclear', 'notes': ['flat: moving'],
              'flat': {'still': False}, 'gyro_bias_dps': {'alpha': 5.0, 'beta': -3.0, 'gamma': 2.0},
              'axis_check': {'verdict': 'w3c', 'units': 'deg_per_s'}}
    nm = C.session_normalization({'calibration': moving, 'device': {'engine': 'blink'}})
    assert nm.gyro_bias is None
    pt = features.prepare_trial(rec, norm=nm)
    assert pt.ok and pt.gyro_bias_source == 'none'
    np.testing.assert_array_equal(pt.gyro_bias, np.zeros(3))
    pre = features.prepare_trial(rec, norm=nm, remove_gyro_bias='pre_go')
    assert pre.gyro_bias_source == 'pre_go'
    assert np.linalg.norm(pre.gyro_bias) < 1.0  # the pre-GO rest rate (the true bias is 0.19 deg/s)
    still = dict(moving, flat={'still': True}, gyro_bias_dps={'alpha': 0.15, 'beta': -0.1, 'gamma': 0.05})
    pt2 = features.prepare_trial(rec, norm=C.session_normalization({'calibration': still, 'device': {'engine': 'blink'}}))
    np.testing.assert_allclose(pt2.gyro_bias, [0.15, -0.1, 0.05])


def test_session_engine_and_trial_vote():
    assert C.session_engine({'engine': 'webkit'}) == ('webkit', 'device.engine')
    assert C.session_engine({'platform': 'ios'})[0] == 'webkit'  # schema 1.0: every iOS browser is WebKit
    assert C.session_engine({'platform': 'android', 'engine': 'unknown'}) == (None, '')
    up = C.expected_up_direction('top_left_toward')
    assert C.gravity_vote(9.8 * up, 'top_left_toward') == 1
    assert C.gravity_vote(-9.8 * up, 'top_left_toward') == -1
    assert C.gravity_vote(9.8 * up, 'free') is None
    assert C.gravity_vote([0.1, 0.1, 0.1], 'upright_toward') is None


def test_rotation_from_gravity_port():
    g = C.GRAVITY
    assert C.rotation_from_gravity([0, g, 0], 1, 'toward') == 'upright'
    assert C.rotation_from_gravity([0, g, 0], -1, 'toward') == 'upside_down'
    assert C.rotation_from_gravity([g, 0, 0], 1, 'toward') == 'top_left'
    assert C.rotation_from_gravity([g, 0, 0], 1, 'away') == 'top_right'
    assert C.rotation_from_gravity([0, 0, -g], -1, 'up') == 'flat'
    assert C.rotation_from_gravity([0, 0, -g], 1, None) == 'unknown'  # face down
    v = [math.sin(math.radians(35)) * g, math.cos(math.radians(35)) * g, 0.0]
    assert C.rotation_from_gravity(v, 1, 'toward') == 'unknown'  # the 35 deg limit is strict
    assert math.isclose(C.tilt_change_deg([0, 1, 0], [1, 0, 0]), 90.0)


def test_mirror_pairs():
    assert C.mirror_pair('top_left_toward') == C.mirror_pair('top_right_toward') == 'top_side_toward'
    assert C.mirror_pair('upright_away') == 'upright_away'


# ───────────────────────────── whole-pipeline consequences ─────────────────────────────


def _pair(participant, seed: int = 21, **plat):
    """The same movement recorded on an android-like and an ios-like phone (noise off)."""
    cond = condition('right', 'top_left_toward')
    out = {}
    for name in ('android', 'ios'):
        p = synth.default_platform(name, None, **plat)
        out[name] = synth.simulate_trial(participant, p, cond, seed=seed, noise=0.0)[0]
    return out


def test_ios_and_android_versions_normalise_to_the_same_signals(symmetric_participant):
    # docs/03 section 5.6, tests 2 and 3
    pair = _pair(symmetric_participant)
    a, i = pair['android'], pair['ios']
    np.testing.assert_allclose(np.asarray(i['streams']['motion']['accg_z'], float),
                               -np.asarray(a['streams']['motion']['accg_z'], float), atol=1e-12)
    fa = features.trial_features(a, norm=C.Normalization.from_sign(1))
    fi = features.trial_features(i, norm=C.Normalization.from_sign(-1))
    for k in features.CHIRAL + features.INVARIANT:
        if np.isfinite(fa[k]):
            assert math.isclose(fa[k], fi[k], rel_tol=1e-9, abs_tol=1e-9), k
    assert fa['chi_yaw_net_deg'] > 0
    # a wrong sign is a parity flip: every chiral feature changes sign
    wrong = features.trial_features(i, norm=C.Normalization.from_sign(1))
    for k in ('chi_yaw_net_deg', 'chi_helicity', 'chi_triple_acc'):
        assert math.isclose(wrong[k], -fa[k], rel_tol=1e-6, abs_tol=1e-9), k


def test_legacy_session_is_remapped_by_the_dataset(symmetric_participant, tmp_path):
    """A session whose calibration found the legacy mapping gives the same physics as a W3C session."""
    parts = {}
    for mapping in ('w3c', 'legacy'):
        p = synth.default_platform('android', None, rr_mapping=mapping)
        rec = synth.simulate_trial(symmetric_participant, p, condition('right', 'upright_toward'), seed=5, noise=0.0)[0]
        plan = [{'block_index': 0, 'kind': 'main', 'hand': 'right', 'orientation': 'upright_toward',
                 'trials': [{'trial_index': 0, 'block_index': 0, 'trial_in_block': 0, 'condition': rec['condition']}]}]
        sess = synth.make_session(symmetric_participant, p, plan, session_id=f'sess-{mapping}', study_code='t',
                                  started_epoch_ms=1.79e12, n_attempts=1, n_accepted=1, ended_epoch_ms=1.79e12 + 6e4,
                                  noise=0.0)
        rec = copy.deepcopy(rec)
        rec['session_id'] = sess['session_id']
        rec['participant_id'] = sess['participant_id']
        rec['trial_id'] = f'trial-{mapping}'
        parts[mapping] = (sess, rec)
    for mapping, (sess, rec) in parts.items():
        d = tmp_path / mapping
        d.mkdir()
        (d / 'session.json').write_text(json.dumps(sess), encoding='utf-8')
        (d / 'trial.json').write_text(json.dumps(rec), encoding='utf-8')
    ds = load_dataset(tmp_path)
    leg = ds.trial_normalization('trial-legacy')
    assert (leg.rates.verdict, leg.rates.verified) == ('legacy', True)
    raw_l = ds.streams('trial-legacy').rr
    raw_w = ds.streams('trial-w3c').rr
    np.testing.assert_allclose(raw_l, raw_w[:, [2, 0, 1]])  # legacy fields hold (z, x, y)
    np.testing.assert_allclose(ds.signals('trial-legacy').gyro, ds.signals('trial-w3c').gyro)
    fl = features.trial_features(ds.record('trial-legacy', compact=True), norm=leg)
    fw = features.trial_features(ds.record('trial-w3c', compact=True), norm=ds.trial_normalization('trial-w3c'))
    assert math.isclose(fl['chi_yaw_net_deg'], fw['chi_yaw_net_deg'], rel_tol=1e-9)


def _calibrated_face_down(session: dict) -> dict:
    """The session as the app records it when the flat step was done screen-down: |z| = |g| passes every
    check, so gravity_sign is confidently wrong, and the (correct) upright step then reads 'inverted'."""
    s = copy.deepcopy(session)
    cal = s['calibration']
    cal['gravity_sign'] = -cal['gravity_sign']
    cal['flat']['accg_mean'] = [-x + 0.0 for x in cal['flat']['accg_mean']]
    cal['upright_check'] = 'inverted'
    cal['notes'] = [*cal['notes'], 'flat_sign_contradicts_engine', 'upright: inverted']
    return s


def _yaw_rule_accuracy(ds, session_ids) -> float:
    """Accuracy of the yaw-sign rule on the trials of ``session_ids``, trained on the other sessions."""
    keep, _ = E.analysis_set(ds.trials)
    F = features.extract_features(ds, list(ds.trials.loc[keep, 'trial_id']))
    X, meta = E.modeling_table(F, ds.trials)
    test = meta['session_id'].isin(list(session_ids)).to_numpy()
    assert test.sum() >= 10 and (~test).sum() >= 10
    rule = models.YawSignRule().fit(X[~test], meta.loc[~test, 'y_right'].to_numpy())
    return float(np.mean(rule.predict(X[test]) == meta.loc[test, 'y_right'].to_numpy()))


def test_face_down_calibration_is_overridden_and_accuracy_restored(tiny_dataset, tmp_path):
    """docs/03 section 4(b): one Android-like and one iOS-like participant did the flat step screen-down.
    Keeping their calibrated sign mirrors every chiral feature (left looks like right); the trial votes
    override it, which restores the left/right accuracy of those sessions."""
    bad = {s['session_id']: s for s in tiny_dataset.sessions[:2]}
    assert {s['device']['engine'] for s in bad.values()} == {'blink', 'webkit'}
    sessions = [_calibrated_face_down(s) if s['session_id'] in bad else s for s in tiny_dataset.sessions]
    root = synth.write_folder_tree(dataclasses.replace(tiny_dataset, sessions=sessions), tmp_path / 'data')

    ds = load_dataset(root)
    ss = ds.sessions.set_index('session_id')
    for sid, orig in bad.items():
        true_sign = orig['calibration']['gravity_sign']
        row = ss.loc[sid]
        assert row['calibration_gravity_sign'] == -true_sign
        assert (row['gravity_sign'], row['gravity_sign_source'], bool(row['gravity_sign_warning'])) == (true_sign, 'inferred', True)
        assert row['gravity_sign_detail'].startswith('calibration sign overridden by the trial votes')
        assert row['gravity_votes_against'] == 0 and row['gravity_votes_for'] >= C.MIN_OVERRIDE_VOTES
    good = ss.drop(index=list(bad))
    assert set(good['gravity_sign_source']) == {'calibration'} and not good['gravity_sign_warning'].any()
    assert _yaw_rule_accuracy(ds, bad) > 0.95

    # the same data with the calibrated sign kept (infer_gravity_sign=False): mirrored, so almost always wrong
    kept = load_dataset(root, infer_gravity_sign=False)
    assert set(kept.sessions.set_index('session_id').loc[list(bad), 'gravity_sign_source']) == {'calibration'}
    assert _yaw_rule_accuracy(kept, bad) < 0.05

    rep = qa_summary(ds, check_online=False, offline=False, consistency_sample=4, gravity_sample=0)
    assert rep.summary['calibrated gravity signs overridden by the trial votes'] == 2
    assert any('trial votes clearly contradict' in w for w in rep.warnings)
