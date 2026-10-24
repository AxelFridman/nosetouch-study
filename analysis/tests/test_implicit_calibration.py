"""Protocol 2.0.0 normalisation without a calibration screen (norm_v2, spec v2 section 10).

* gravity sign: the engine prior confirmed by a confident trial vote is a source of its own ('engine_vote',
  no warning); conflicts stay flagged; an implicit calibration's sign is one upright hold, not a measurement.
* rotationRate mapping: inferred from the session-level gyro-vs-gravity closure of the trials (verified
  when the best candidate closes below 3 deg, below 0.5 x the runner-up, over >= 8 trials).
* gyro bias: nothing is subtracted without a trusted (still, flat) calibration bias.
* sessions carry ``calibration_method``; rule S3 means "gravity sign supported".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _mlhelpers import condition

from motionlab import conventions as C
from motionlab import evaluate as E
from motionlab import features, load_dataset, synth

#: An implicit calibration as createSession stores it, before the first still upright trial fills it.
EMPTY_IMPLICIT = {'method': 'implicit_v1', 'performed_at': '2026-10-24T10:00:00.000Z', 'flat': None, 'upright': None,
                  'gravity_sign': None, 'gravity_magnitude': None, 'upright_check': None, 'gyro_bias_dps': None,
                  'notes': [], 'axis_check': None}


def implicit(sign, check='ok', notes=()):
    return {**EMPTY_IMPLICIT, 'gravity_sign': sign, 'upright_check': check, 'notes': list(notes)}


KEYS = [C.rate_candidate_key(v, u) for v, u in C.RATE_CANDIDATES]


def closures(best='w3c/deg_per_s', best_deg=1.3, runner_up_deg=4.0, n=10, seed=0):
    """n per-trial closure dicts: ``best`` closes to ~best_deg, every other candidate to >= runner_up_deg."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        d = {k: runner_up_deg + float(rng.uniform(0, 30)) for k in KEYS}
        d[KEYS[(KEYS.index(best) + 1) % len(KEYS)]] = runner_up_deg  # the runner-up
        d[best] = best_deg
        rows.append(d)
    return rows


# ───────────────────────────── calibration method ─────────────────────────────


def test_calibration_method():
    assert C.calibration_method(None) is None and C.calibration_method({}) is None
    assert C.calibration_method({'gravity_sign': 1, 'upright_check': 'ok'}) == 'guided_v1'  # schema <= 1.1
    assert C.calibration_method({'method': 'guided_v1', 'gravity_sign': 1}) == 'guided_v1'
    assert C.calibration_method(EMPTY_IMPLICIT) == 'implicit_v1'
    assert C.calibration_method({'method': 'table_v9'}) is None
    assert C.CALIBRATION_METHODS == ('guided_v1', 'implicit_v1')


# ───────────────────────────── gravity sign: engine + votes ─────────────────────────────


def test_engine_prior_confirmed_by_the_votes_is_engine_vote_without_warning():
    for engine, s in (('blink', 1), ('gecko', 1), ('webkit', -1)):
        g = C.resolve_gravity_sign(implicit(s), [s] * 6, engine=engine)
        assert (g.sign, g.source, g.warning, g.votes_for, g.votes_against) == (s, 'engine_vote', False, 6, 0), engine
        assert g.usable_for_chirality and 'confirmed by the trial votes' in g.detail
    # before the first still upright trial (sign still null), and for sessions without a calibration object
    assert C.resolve_gravity_sign(EMPTY_IMPLICIT, [1] * 4, engine='blink').source == 'engine_vote'
    assert C.resolve_gravity_sign(None, [-1] * 3, engine='webkit').source == 'engine_vote'
    # >= 80 % agreement is confident: one stray vote does not break the consensus
    g = C.resolve_gravity_sign(EMPTY_IMPLICIT, [1, 1, 1, 1, -1], engine='blink')
    assert (g.source, g.warning, g.votes_for, g.votes_against) == ('engine_vote', False, 4, 1)
    # infer=False never lets the votes SET a sign, but they may confirm the engine prior
    assert C.resolve_gravity_sign(EMPTY_IMPLICIT, [1] * 6, engine='blink', infer=False).source == 'engine_vote'


def test_conflicts_stay_flagged():
    # the votes contradict the engine: the vote wins (docs/03 section 5.2 order), flagged, the engine named
    g = C.resolve_gravity_sign(EMPTY_IMPLICIT, [1] * 6, engine='webkit')
    assert (g.sign, g.source, g.warning) == (1, 'inferred', True)
    assert 'differs from the webkit engine expectation' in g.detail
    g = C.resolve_gravity_sign(EMPTY_IMPLICIT, [1] * 6, engine='webkit', order=('engine', 'vote'))
    assert (g.sign, g.source, g.warning) == (-1, 'engine', True) and 'CONTRADICTED' in g.detail
    # no confident vote (split or too few): the engine alone, flagged
    for votes in ([1, -1, 1, -1], [1, 1], []):
        g = C.resolve_gravity_sign(EMPTY_IMPLICIT, votes, engine='blink')
        assert (g.sign, g.source, g.warning) == (1, 'engine', True) and 'no confident trial vote' in g.detail
    # the implicit hold contradicts the consensus (e.g. the first upright hold was upside down): flagged
    bad = implicit(-1, 'inverted', [synth.UPRIGHT_SIGN_CONTRADICTS_ENGINE])
    g = C.resolve_gravity_sign(bad, [1] * 6, engine='blink')
    assert (g.sign, g.source, g.warning) == (1, 'engine_vote', True) and 'implicit calibration sign (-1' in g.detail
    # an implicit sign is never a 'calibration' source and cannot outvote the trials
    g = C.resolve_gravity_sign(implicit(-1), [1] * 6, engine=None)
    assert (g.sign, g.source) == (1, 'inferred') and 'contradicts it' in g.detail
    # nothing but the implicit hold (unknown engine, no confident vote): its sign, flagged
    g = C.resolve_gravity_sign(implicit(-1), [], engine=None)
    assert (g.sign, g.source, g.warning) == (-1, 'implicit', True)
    g = C.resolve_gravity_sign(EMPTY_IMPLICIT, [], engine=None)
    assert (g.sign, g.source, g.warning, g.usable_for_chirality) == (1, 'default', True, False)
    # a guided calibration keeps its precedence (and stays unflagged when undisputed)
    for cal in ({'gravity_sign': 1, 'upright_check': 'ok'}, {'method': 'guided_v1', 'gravity_sign': 1, 'upright_check': 'ok'}):
        g = C.resolve_gravity_sign(cal, [1] * 6, engine='blink')
        assert (g.sign, g.source, g.warning) == (1, 'calibration', False)


def test_upright_votes_decide_when_sideways_votes_are_split():
    """A sign flip turns top_left into top_right, so sideways votes confound the sign with left/right grip
    mix-ups; upright votes do not (nobody holds the phone upside down by mistake)."""
    all_votes = [1, 1, 1, 1, -1, -1, 1, -1]  # 4 upright (+1) and 4 split sideways trials
    g = C.resolve_gravity_sign(EMPTY_IMPLICIT, all_votes, engine='blink', upright_votes=[1, 1, 1, 1])
    assert (g.source, g.warning) == ('engine_vote', False) and 'upright trials 4/4' in g.detail
    assert C.resolve_gravity_sign(EMPTY_IMPLICIT, all_votes, engine='blink').source == 'engine'  # 62 %: not confident
    # too few upright votes: the vote of all trials decides
    g = C.resolve_gravity_sign(EMPTY_IMPLICIT, [1] * 8, engine='blink', upright_votes=[1, 1])
    assert g.source == 'engine_vote' and 'all trials 8/8' in g.detail


# ───────────────────────────── rotationRate mapping from the closure ─────────────────────────────


def test_infer_rate_mapping_thresholds():
    m = C.infer_rate_mapping(closures(n=8))
    assert (m.verdict, m.units, m.verified, m.source, m.detail) == ('w3c', 'deg_per_s', True, 'closure', '')
    assert (m.n_trials, m.closure_deg, m.runner_up_deg) == (8, 1.3, 4.0) and m.medians[0] == ('w3c/deg_per_s', 1.3)
    assert len(m.medians) == len(C.RATE_CANDIDATES) == 8
    # fewer than 8 trials: one trial cannot settle the mapping (pilot: W3C won 18 of 22 trials by 1.3 deg)
    m = C.infer_rate_mapping(closures(n=7))
    assert (m.verdict, m.units, m.verified, m.source) == ('w3c', 'deg_per_s', False, 'assumed')
    assert m.detail.startswith('axis_unverified') and '7 < 8 trials' in m.detail and m.closure_deg == 1.3
    # the best candidate must close below 3 deg ...
    m = C.infer_rate_mapping(closures(best_deg=3.0, runner_up_deg=9.0))
    assert not m.verified and 'best 3.0 >= 3 deg' in m.detail
    # ... and below half the runner-up
    m = C.infer_rate_mapping(closures(best_deg=1.3, runner_up_deg=2.5))
    assert not m.verified and 'runner-up' in m.detail
    # another mapping or unit that wins decisively is applied (and named)
    m = C.infer_rate_mapping(closures(best='legacy/deg_per_s'))
    assert (m.verdict, m.units, m.verified, m.key) == ('legacy', 'deg_per_s', True, 'legacy/deg_per_s')
    assert m.detail == 'closure check: legacy, units deg_per_s'
    m = C.infer_rate_mapping(closures(best='w3c/rad_per_s'))
    assert (m.verdict, m.units, m.verified) == ('w3c', 'rad_per_s', True)
    # trials that cannot tell (None) or incomplete dicts are skipped; nothing usable -> assumed W3C
    m = C.infer_rate_mapping([*closures(n=8), None, {}, {'w3c/deg_per_s': 0.5}, {k: float('nan') for k in KEYS}])
    assert m.verified and m.n_trials == 8
    m = C.infer_rate_mapping([None, None])
    assert (m.verified, m.n_trials, m.closure_deg) == (False, 0, None) and 'no trial' in m.detail
    # the session MEDIAN decides: a few trials won by another mapping do not flip it
    rows = closures(n=10) + closures(best='legacy/deg_per_s', n=3, seed=1)
    m = C.infer_rate_mapping(rows)
    assert m.verified and m.key == 'w3c/deg_per_s' and m.n_trials == 13


@pytest.mark.parametrize('mapping', ['w3c', 'legacy'])
def test_rate_closures_of_a_synthetic_trial(symmetric_participant, mapping):
    plat = synth.default_platform('android', None, rr_mapping=mapping)
    rec = synth.simulate_trial(symmetric_participant, plat, condition('right', 'top_left_toward'), seed=5)[0]
    c = features.rate_closures(rec, engine='blink')
    assert set(c) == set(KEYS)
    order = sorted(c, key=c.get)
    assert order[0] == f'{mapping}/deg_per_s' and c[order[0]] < 1.5 and c[order[1]] > 2 * c[order[0]]
    assert c[f'{mapping}/rad_per_s'] > 30  # read as rad/s: ~57x the rotation
    # the accelerometer sign does not matter (both rest windows flip together): the iOS twin closes alike
    ios = synth.default_platform('ios', None, rr_mapping=mapping)
    rec_i = synth.simulate_trial(symmetric_participant, ios, condition('right', 'top_left_toward'), seed=5)[0]
    ci = features.rate_closures(rec_i, engine='webkit')
    assert min(ci, key=ci.get) == f'{mapping}/deg_per_s'
    # no movement: nothing to tell the mappings apart
    still = synth.simulate_trial(symmetric_participant, plat, condition(), seed=3, behavior='no_move')[0]
    assert features.rate_closures(still, engine='blink') is None


def test_session_normalization_uses_the_closure_only_without_a_definite_axis_check():
    legacy = C.infer_rate_mapping(closures(best='legacy/deg_per_s'))
    nm = C.session_normalization({'calibration': implicit(1), 'device': {'engine': 'blink'}}, [1] * 10,
                                 rate_inference=legacy)
    assert (nm.rates.source, nm.rates.verdict, nm.rates.verified) == ('closure', 'legacy', True)
    assert nm.calibration_method == 'implicit_v1' and nm.gravity.source == 'engine_vote'
    assert 'axis_unverified' not in nm.flags and 'rr_remapped_legacy_deg_per_s' in nm.flags
    # a definite axis_check (guided calibration) wins; a decisive closure disagreeing with it is flagged
    guided = {'gravity_sign': 1, 'upright_check': 'ok', 'axis_check': {'verdict': 'w3c', 'units': 'deg_per_s'}}
    nm = C.session_normalization({'calibration': guided, 'device': {'engine': 'blink'}}, rate_inference=legacy)
    assert (nm.rates.source, nm.rates.verdict) == ('calibration', 'w3c')
    assert nm.rate_check_disagrees and 'rr_closure_disagrees' in nm.flags
    agree = C.infer_rate_mapping(closures())
    assert not C.session_normalization({'calibration': guided}, rate_inference=agree).rate_check_disagrees
    # an inconclusive closure check leaves W3C assumed and flagged
    nm = C.session_normalization({'calibration': implicit(1)}, rate_inference=C.infer_rate_mapping(closures(n=3)))
    assert (nm.rates.verified, nm.rates.source) == (False, 'assumed') and 'axis_unverified' in nm.flags


# ───────────────────────────── gyro bias ─────────────────────────────


def test_no_gyro_bias_is_subtracted_without_a_trusted_flat_calibration(symmetric_participant):
    """A hand-held pre-GO mean is tremor, not bias: on the pilot, subtracting it worsened the median gravity
    closure from 1.6 to 2.7 deg. Only a still flat (guided) calibration window is a bias measurement."""
    plat = synth.default_platform('android', None, gyro_bias_dps=(0.15, -0.1, 0.05))
    rec = synth.simulate_trial(symmetric_participant, plat, condition('right', 'upright_toward'), seed=3)[0]
    cal = synth.implicit_calibration([rec], engine='blink', performed_at='2026-10-24T10:00:00.000Z')
    assert cal['gyro_bias_dps'] is None and cal['flat'] is None and cal['axis_check'] is None
    nm = C.session_normalization({'calibration': cal, 'device': {'engine': 'blink'}})
    assert nm.gyro_bias is None
    pt = features.prepare_trial(rec, norm=nm)
    assert pt.ok and pt.gyro_bias_source == 'none'
    np.testing.assert_array_equal(pt.gyro_bias, np.zeros(3))
    assert features.trial_features(rec, norm=nm)['qc_gyro_bias_source'] == 'none'
    # the norm_v1 pre-GO fallback only on request (ablations)
    pt = features.prepare_trial(rec, norm=nm, remove_gyro_bias='pre_go')
    assert pt.gyro_bias_source == 'pre_go' and 0 < np.linalg.norm(pt.gyro_bias) < C.GYRO_BIAS_MAX_DPS
    # a still flat window of a guided calibration is subtracted
    guided = {'gravity_sign': 1, 'upright_check': 'ok', 'flat': {'still': True},
              'gyro_bias_dps': {'alpha': 0.15, 'beta': -0.1, 'gamma': 0.05},
              'axis_check': {'verdict': 'w3c', 'units': 'deg_per_s'}}
    pt = features.prepare_trial(rec, norm=C.session_normalization({'calibration': guided, 'device': {'engine': 'blink'}}))
    assert pt.gyro_bias_source == 'calibration'
    np.testing.assert_allclose(pt.gyro_bias, [0.15, -0.1, 0.05])
    with pytest.raises(ValueError):
        features.prepare_trial(rec, norm=nm, remove_gyro_bias='median')


# ───────────────────────────── the app's implicit_v1 (synth port) ─────────────────────────────


def test_synth_implicit_calibration_follows_the_app(symmetric_participant):
    def first_upright(platform, seed=3):
        return synth.simulate_trial(symmetric_participant, platform, condition('right', 'upright_toward'), seed=seed)[0]

    when = '2026-10-24T10:00:00.000Z'
    side = synth.simulate_trial(symmetric_participant, synth.default_platform('android', None),
                                condition('right', 'top_left_toward'), seed=4)[0]
    rec = first_upright(synth.default_platform('android', None))
    cal = synth.implicit_calibration([side, rec], engine='blink', performed_at=when)
    assert (cal['method'], cal['gravity_sign'], cal['upright_check'], cal['notes']) == ('implicit_v1', 1, 'ok', [])
    assert cal['upright']['still'] and cal['upright']['n'] >= 20 and 9.6 < cal['gravity_magnitude'] < 10.0
    assert cal['upright']['accg_mean'][1] > 7
    # WebKit reports both acceleration fields inverted: sign -1, ok against the webkit prior
    cal = synth.implicit_calibration([first_upright(synth.default_platform('ios', None))], engine='webkit',
                                     performed_at=when)
    assert (cal['gravity_sign'], cal['upright_check']) == (-1, 'ok')
    # a sign the engine contradicts: 'inverted' + note (the analysis distrusts it)
    cal = synth.implicit_calibration([first_upright(synth.default_platform('ios', None))], engine='blink',
                                     performed_at=when)
    assert (cal['gravity_sign'], cal['upright_check']) == (-1, 'inverted')
    assert cal['notes'] == [synth.UPRIGHT_SIGN_CONTRADICTS_ENGINE]
    # unknown engine: any sign is ok
    assert synth.implicit_calibration([rec], engine=None, performed_at=when)['upright_check'] == 'ok'
    # no complete upright trial: nothing filled
    failed = synth.simulate_trial(symmetric_participant, synth.default_platform('android', None), condition(), seed=3,
                                  behavior='no_move')[0]
    cal = synth.implicit_calibration([side, failed], engine='blink', performed_at=when)
    assert {k: cal[k] for k in ('upright', 'gravity_sign', 'upright_check', 'gravity_magnitude')} == dict.fromkeys(
        ('upright', 'gravity_sign', 'upright_check', 'gravity_magnitude'))


# ───────────────────────────── the dataset ─────────────────────────────


def test_street_sessions_normalise_without_a_calibration_screen(street_ds):
    street = street_ds
    ss = street.sessions
    assert len(ss) == 4 and set(ss['calibration_method']) == {'implicit_v1'}
    assert set(ss['gravity_sign_source']) == {'engine_vote'} and not ss['gravity_sign_warning'].astype(bool).any()
    assert set(zip(ss['platform'], ss['gravity_sign'])) == {('android', 1), ('ios', -1)}
    assert set(ss['rr_source']) == {'closure'} and ss['rr_verified'].astype(bool).all()
    assert set(ss['rr_mapping']) == {'w3c'} and set(ss['rr_units']) == {'deg_per_s'}
    assert (ss['rr_closure_trials'] >= C.RATE_CLOSURE_MIN_TRIALS).all() and (ss['rr_closure_deg'] < 1.0).all()
    assert (ss['rr_closure_runner_up_deg'] > 5 * ss['rr_closure_deg']).all()
    assert not ss['gyro_bias_subtracted'].astype(bool).any() and not ss['rr_closure_disagrees'].astype(bool).any()
    assert set(ss['norm_version']) == {C.NORM_VERSION} == {'norm_v2'}
    assert set(street.trials['calibration_method']) == {'implicit_v1'}
    msgs = ' '.join(street.issues['message'])
    assert 'from the closure check' in msgs and 'axis_unverified' not in msgs


def test_rate_mapping_modes(street_tree, tiny_tree):
    # 'calibration' never runs the closure check (norm_v1 behaviour): W3C assumed and flagged
    ds = load_dataset(street_tree, rate_mapping='calibration')
    assert set(ds.sessions['rr_source']) == {'assumed'} and not ds.sessions['rr_verified'].astype(bool).any()
    assert ds.sessions['rr_closure_deg'].isna().all()
    # guided sessions with a definite axis_check: no closure check by default, a cross-check with 'always'
    tiny = load_dataset(tiny_tree)
    assert set(tiny.sessions['rr_source']) == {'calibration'} and tiny.sessions['rr_closure_deg'].isna().all()
    assert set(tiny.sessions['calibration_method']) == {'guided_v1'}
    both = load_dataset(tiny_tree, rate_mapping='always')
    assert set(both.sessions['rr_source']) == {'calibration'} and both.sessions['rr_closure_deg'].notna().all()
    assert not both.sessions['rr_closure_disagrees'].astype(bool).any()
    with pytest.raises(ValueError):
        load_dataset(tiny_tree, rate_mapping='never')


def test_calibration_method_column_for_missing_calibrations(tmp_path):
    ds = synth.make_dataset(2, seed=4, preset='tiny', calibration='missing')
    synth.write_folder_tree(ds, tmp_path)
    d = load_dataset(tmp_path)
    assert set(d.sessions['calibration_method']) == {'none'} and not d.sessions['calibration_present'].any()


# ───────────────────────────── rule S3 ─────────────────────────────


def test_rule_s3_is_gravity_sign_supported():
    trials = pd.DataFrame({'trial_id': ['a', 'b', 'c', 'd'], 'session_id': ['s1', 's1', 's2', 's3'],
                           'gravity_sign_source': ['engine_vote', 'engine_vote', 'default', None]})
    keep, table = E.analysis_set(trials, rules={}, require_calibration=True)
    assert keep.tolist() == [True, True, False, False]
    assert table.to_dict('records') == [{'rule': 'S3 gravity sign supported', 'failing': 2, 'remaining': 2}]
    # from the sessions table when the trials lack the column
    sess = pd.DataFrame({'session_id': ['s1', 's2', 's3'], 'gravity_sign_source': ['implicit', 'default', 'calibration']})
    keep, _ = E.analysis_set(trials.drop(columns='gravity_sign_source'), rules={}, require_calibration=True, sessions=sess)
    assert keep.tolist() == [True, True, False, True]
    with pytest.raises(ValueError):
        E.analysis_set(trials.drop(columns='gravity_sign_source'), rules={}, require_calibration=True)
    # off by default
    assert E.analysis_set(trials, rules={})[0].all()


def test_rule_s3_keeps_every_implicit_session(street_ds):
    keep, _ = E.analysis_set(street_ds.trials, require_calibration=True)
    base, _ = E.analysis_set(street_ds.trials)
    assert keep.equals(base) and keep.sum() > 100
