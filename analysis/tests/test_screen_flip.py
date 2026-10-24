"""Screen-away synthesis: the phone turned 180 deg about its own y axis (screen toward <-> away, same hand).

F = diag(-1, 1, -1) is a PROPER rotation, so every device vector transforms alike: f' = F f and w' = F w
(the mirror, a reflection, has w' = -S w instead). GRIP_MATRICES[o_away] = GRIP_MATRICES[o_toward] F.
Under the W3C mapping acc_x, acc_z, accg_x, accg_z, rr_alpha and rr_gamma change sign; under the legacy
mapping the x and z rates live in rr_beta and rr_alpha. Protocol 2.0.0 collects no screen-away grips:
these copies only augment TRAINING data (never report screen-away accuracy from them)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from _mlhelpers import condition

from motionlab import augment, conventions as C, evaluate as E, features, schema, synth
from motionlab.preprocess import euler_to_matrix, quaternion_to_matrix, random_rotation, rotvec_to_matrix

F = C.SCREEN_FLIP
MOTION_VEC = ('acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma')
ROTATIONS = ('upright', 'top_left', 'top_right', 'upside_down')


def _col(rec, k, stream='motion'):
    return np.asarray([np.nan if v is None else v for v in rec['streams'][stream][k]], dtype=float)


def _vee(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def _quat_matrices(q):
    return quaternion_to_matrix(np.column_stack([np.asarray(q[c], dtype=float) for c in 'xyzw']))


def _trial(ds, orientation, platform=None):
    sids = {s['session_id'] for s in ds.sessions if platform is None or s['device']['platform'] == platform}
    return next(t for t in ds.trials if t['condition']['orientation'] == orientation and t['session_id'] in sids
                and t['outcome']['status'] == 'complete')


# ───────────────────────────── the rule ─────────────────────────────


def test_flip_rule_on_random_rigid_body_trajectories():
    """R' = R F (the device turned about its own y axis), same path p: f' = F f and w' = F w."""
    rng = np.random.default_rng(4)
    g_up = np.array([0.0, C.GRAVITY, 0.0])
    h = 1e-5
    assert np.allclose(F @ F.T, np.eye(3)) and np.isclose(np.linalg.det(F), 1.0)  # a rotation, its own inverse
    for _ in range(25):
        R0 = random_rotation(rng)
        w1, w2 = rng.normal(size=3) * 2, rng.normal(size=3) * 3
        c1, c2 = rng.normal(size=3), rng.normal(size=3)

        def R(t):
            return R0 @ rotvec_to_matrix(t * w1 + t * t * w2)

        def p(t):
            return c1 * t + c2 * t ** 3

        for t in rng.uniform(0, 1, size=3):
            pdd = (p(t + h) - 2 * p(t) + p(t - h)) / h ** 2
            Rd = (R(t + h) - R(t - h)) / (2 * h)
            f = R(t).T @ (pdd + g_up)
            w = _vee(R(t).T @ Rd)
            Rf = lambda s: R(s) @ F  # noqa: E731
            Rfd = (Rf(t + h) - Rf(t - h)) / (2 * h)
            ff = Rf(t).T @ (pdd + g_up)
            wf = _vee(Rf(t).T @ Rfd)
            np.testing.assert_allclose(ff, F @ f, atol=1e-4)
            np.testing.assert_allclose(wf, F @ w, atol=1e-4)  # NOT -F w: a proper rotation
            a2, w2_ = C.flip_device_vectors(f[None, :], w[None, :])
            np.testing.assert_allclose(a2[0], F @ f)
            np.testing.assert_allclose(w2_[0], F @ w)


def test_away_grip_is_toward_grip_times_flip():
    for rot in ROTATIONS:
        np.testing.assert_allclose(C.GRIP_MATRICES[f'{rot}_away'], C.GRIP_MATRICES[f'{rot}_toward'] @ F, err_msg=rot)
        assert C.screen_flip_orientation(f'{rot}_toward') == f'{rot}_away'
        assert C.screen_flip_orientation(f'{rot}_away') == f'{rot}_toward'
        # the up direction the trial votes compare with flips like any device vector
        np.testing.assert_allclose(C.expected_up_direction(f'{rot}_away'), F @ C.expected_up_direction(f'{rot}_toward'))
    assert C.screen_flip_orientation('free') == 'free'
    with pytest.raises(ValueError):
        C.screen_flip_orientation('flat_up')  # screen down is not in the enum


def test_flip_commutes_with_the_mirror_and_is_an_involution():
    rng = np.random.default_rng(0)
    a, w = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    m_then_f = C.flip_device_vectors(*C.mirror_device_vectors(a, w))
    f_then_m = C.mirror_device_vectors(*C.flip_device_vectors(a, w))
    np.testing.assert_array_equal(m_then_f[0], f_then_m[0])
    np.testing.assert_array_equal(m_then_f[1], f_then_m[1])
    twice = C.flip_device_vectors(*C.flip_device_vectors(a, w))
    np.testing.assert_array_equal(twice[0], a)
    np.testing.assert_array_equal(twice[1], w)
    for o in ('top_left_toward', 'top_right_away', 'upright_toward', 'upside_down_away'):
        assert C.mirror_orientation(C.screen_flip_orientation(o)) == C.screen_flip_orientation(C.mirror_orientation(o))


# ───────────────────────────── screen_away_trial ─────────────────────────────


def test_field_signs_under_w3c_and_legacy_mappings(tiny_dataset):
    rec = _trial(tiny_dataset, 'top_left_toward')
    fl = augment.screen_away_trial(rec)
    for k in ('acc_x', 'acc_z', 'accg_x', 'accg_z', 'rr_alpha', 'rr_gamma'):  # W3C: x and z
        np.testing.assert_array_equal(_col(fl, k), -_col(rec, k), err_msg=k)
    for k in ('acc_y', 'accg_y', 'rr_beta'):
        np.testing.assert_array_equal(_col(fl, k), _col(rec, k), err_msg=k)
    np.testing.assert_array_equal(_col(fl, 't'), _col(rec, 't'))
    # legacy (alpha, beta, gamma) = (z, x, y): x and z are rr_beta and rr_alpha
    lg = augment.screen_away_trial(rec, rr_verdict='legacy')
    np.testing.assert_array_equal(_col(lg, 'rr_beta'), -_col(rec, 'rr_beta'))
    np.testing.assert_array_equal(_col(lg, 'rr_alpha'), -_col(rec, 'rr_alpha'))
    np.testing.assert_array_equal(_col(lg, 'rr_gamma'), _col(rec, 'rr_gamma'))
    # labels, contact, ids; the input is untouched
    assert fl['condition']['orientation'] == 'top_left_away' and fl['condition']['hand'] == rec['condition']['hand']
    assert fl['streams']['touches'] == [] and fl['timing']['first_touch_ms'] is None
    assert fl['trial_id'] == rec['trial_id'] + '-away'
    assert augment.screen_away_trial(rec, id_suffix=None)['trial_id'] == rec['trial_id']
    assert rec['condition']['orientation'] == 'top_left_toward'
    g, gf = rec['segmentation']['baseline']['gravity'], fl['segmentation']['baseline']['gravity']
    np.testing.assert_allclose(gf, F @ np.asarray(g, dtype=float))
    # landscape grips: the screen x axis flips (primary <-> secondary, angle a -> 360 - a)
    assert {rec['screen']['type_start'], fl['screen']['type_start']} == {'landscape-primary', 'landscape-secondary'}
    assert (rec['screen']['angle_start'] + fl['screen']['angle_start']) % 360 == 0
    with pytest.raises(ValueError):
        augment.screen_away_trial({**rec, 'condition': {**rec['condition'], 'orientation': 'flat_up'}})


def test_world_orientation_streams_are_turned_about_device_y(tiny_dataset):
    rec = _trial(tiny_dataset, 'upright_toward', 'android')  # android trials carry the quaternion stream
    fl = augment.screen_away_trial(rec)
    R = _quat_matrices(rec['streams']['quaternion'])
    Rf = _quat_matrices(fl['streams']['quaternion'])
    np.testing.assert_allclose(Rf, R @ F, atol=1e-9)  # R'_ED = R_ED F: q (x, y, z, w) -> (-z, w, x, -y)
    q, qf = rec['streams']['quaternion'], fl['streams']['quaternion']
    np.testing.assert_array_equal(np.asarray(qf['x']), -np.asarray(q['z']) + 0.0)
    np.testing.assert_array_equal(np.asarray(qf['y']), np.asarray(q['w']))
    o, of = rec['streams']['orientation'], fl['streams']['orientation']
    E0 = euler_to_matrix(o['alpha'], o['beta'], o['gamma'])
    E1 = euler_to_matrix(of['alpha'], of['beta'], of['gamma'])
    np.testing.assert_allclose(E1, E0 @ F, atol=1e-9)
    assert all(v is None for v in of['compass'])  # iOS headings depend on which way the screen faces


def test_double_flip_is_the_identity(tiny_dataset):
    for rec in [t for t in tiny_dataset.trials if t['condition']['orientation'].endswith('_toward')][:6]:
        back = augment.screen_away_trial(augment.screen_away_trial(rec), id_suffix=None)
        assert json.dumps(back['streams']['motion']) == json.dumps(rec['streams']['motion'])
        assert back['condition'] == rec['condition'] and back['detected'] == rec['detected']
        assert back['segmentation']['baseline'] == rec['segmentation']['baseline'] and back['screen'] == rec['screen']
        assert back['trial_id'] == rec['trial_id'] + '-away'  # the first flip's suffix stays
        if rec['streams']['quaternion']:
            assert back['streams']['quaternion'] == rec['streams']['quaternion']  # exact: (z, -w, -x, y) undoes it
        o0, o2 = rec['streams']['orientation'], back['streams']['orientation']
        np.testing.assert_allclose(euler_to_matrix(o2['alpha'], o2['beta'], o2['gamma']),
                                   euler_to_matrix(o0['alpha'], o0['beta'], o0['gamma']), atol=1e-9)


def test_flip_commutes_with_the_record_mirror(tiny_dataset):
    rec = _trial(tiny_dataset, 'top_left_toward', 'android')
    a = augment.mirror_trial(augment.screen_away_trial(rec))
    b = augment.screen_away_trial(augment.mirror_trial(rec))
    assert a['condition'] == b['condition'] and a['condition']['orientation'] == 'top_right_away'
    for k in MOTION_VEC:
        np.testing.assert_array_equal(_col(a, k), _col(b, k), err_msg=k)
    np.testing.assert_allclose(_quat_matrices(a['streams']['quaternion']), _quat_matrices(b['streams']['quaternion']),
                               atol=1e-9)
    assert a['detected'] == b['detected']


@pytest.mark.parametrize('platform', ['android', 'ios'])
def test_gravity_votes_and_detected_rotation_are_preserved(tiny_dataset, platform):
    """Both gravity signs (Blink +1, WebKit -1): the vote of a flipped trial against its *_away label equals
    the vote of the original, and the detected rotation is unchanged (the sign commutes with F)."""
    for o in ('upright_toward', 'top_left_toward'):
        rec = _trial(tiny_dataset, o, platform)
        fl = augment.screen_away_trial(rec)
        v = C.trial_gravity_vote(rec)
        assert v in (1, -1) and C.trial_gravity_vote(fl) == v
        assert fl['detected'] == rec['detected']
        sign = 1 if platform == 'android' else -1
        g = np.asarray(rec['segmentation']['baseline']['gravity'], dtype=float)
        assert C.rotation_from_gravity(F @ g, sign, 'away') == C.rotation_from_gravity(g, sign, 'toward') == \
            C.ORIENTATIONS[o]['rotation']


def test_rotation_from_gravity_is_flip_invariant_for_every_rotation():
    for rot in ROTATIONS:
        up = C.expected_up_direction(f'{rot}_toward') * C.GRAVITY
        for sign in (1, -1):
            raw = sign * up
            assert C.rotation_from_gravity(raw, sign, 'toward') == rot
            assert C.rotation_from_gravity(F @ raw, sign, 'away') == rot


def test_flipped_records_are_schema_valid(tiny_dataset):
    for o in ('upright_toward', 'top_left_toward'):
        for plat in ('android', 'ios'):
            fl = augment.screen_away_trial(_trial(tiny_dataset, o, plat))
            assert schema.validate(fl, 'TrialRecord', mode='fast') == []
    assert schema.validate(augment.screen_away_trial(tiny_dataset.trials[0]), 'TrialRecord', mode='full') == []


def test_flip_keeps_the_schema_1_2_streams_consistent():
    """A street (v2) trial with the absolute orientation streams, touch details and screen events."""
    part = synth.random_participant(np.random.default_rng(2))
    plat = synth.default_platform('android', np.random.default_rng(2))
    rec, _ = synth.simulate_trial(part, plat, {**condition('left', 'top_right_toward'), 'hand_source': 'instructed'},
                                  seed=8, v2=True)
    rec['streams']['screen_events'] = [{'t': 100.0, 'type': 'landscape-primary', 'angle': 90.0}]
    rec['streams']['magnetometer'] = {'t': [1.0, 2.0], 'x': [10.0, 11.0], 'y': [20.0, None], 'z': [-30.0, -31.0],
                                      'frequency_hz': 10.0}
    assert rec['streams']['quaternion_abs'] is not None and rec['streams']['touches']
    fl = augment.screen_away_trial(rec)
    assert schema.validate(fl, 'TrialRecord', mode='full') == []
    np.testing.assert_allclose(_quat_matrices(fl['streams']['quaternion_abs']),
                               _quat_matrices(rec['streams']['quaternion_abs']) @ F, atol=1e-9)
    assert fl['streams']['magnetometer']['x'] == [-10.0, -11.0] and fl['streams']['magnetometer']['z'] == [30.0, 31.0]
    assert fl['streams']['magnetometer']['y'] == [20.0, None]
    assert fl['streams']['screen_events'] == [{'t': 100.0, 'type': 'landscape-secondary', 'angle': 270.0}]
    assert fl['streams']['touches'] == [] and fl['timing']['return_onset_ms'] == rec['timing']['return_onset_ms']
    assert fl['condition']['hand_source'] == 'instructed'


# ───────────────────────────── features ─────────────────────────────


@pytest.mark.parametrize('mapping', ['w3c', 'legacy'])
def test_invariant_chiral_and_basic_features_are_unchanged(symmetric_participant, mapping):
    plat = synth.default_platform('android', np.random.default_rng(1), rr_mapping=mapping)
    rec, _ = synth.simulate_trial(symmetric_participant, plat, condition('right', 'top_left_toward'), seed=17)
    nm = C.Normalization(rates=C.RateMapping(mapping, 'deg_per_s', True))
    fl = augment.screen_away_trial(rec, rr_verdict=mapping)
    a = features.trial_features(rec, norm=nm)
    b = features.trial_features(fl, norm=nm)
    c = features.trial_features(rec, norm=nm, screen_flip=True)
    g = features.feature_groups()
    assert a['qc_ok'] == b['qc_ok'] == c['qc_ok'] == 1
    for k in g['basic'] + g['invariant'] + g['chiral']:
        if np.isfinite(a[k]):
            assert math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-9), k
    # prepare_trial(screen_flip=True) is the same as featurising the flipped record
    for k in g['basic'] + g['invariant'] + g['chiral'] + g['device']:
        if np.isfinite(b[k]):
            assert math.isclose(b[k], c[k], rel_tol=1e-9, abs_tol=1e-9), k
    changed = [k for k in g['device'] if np.isfinite(a[k]) and not math.isclose(a[k], b[k], rel_tol=1e-6, abs_tol=1e-6)]
    assert changed  # only the device-frame features see the flip
    assert (a['qc_screen_flipped'], c['qc_screen_flipped']) == (0, 1)
    # the flip commutes with the feature-level mirror
    m1 = features.trial_features(rec, norm=nm, screen_flip=True, mirror=True)
    m2 = features.trial_features(augment.mirror_trial(fl, rr_verdict=mapping), norm=nm)
    for k in g['chiral'] + g['device']:
        if np.isfinite(m1[k]):
            assert math.isclose(m1[k], m2[k], rel_tol=1e-6, abs_tol=1e-6), k


@pytest.mark.parametrize('rotation', ROTATIONS)
def test_flipped_toward_trial_equals_the_synthesised_away_trial(symmetric_participant, rotation):
    """Noise-free synthetic data without grip tilt (a tilt does not commute with F): the flip of a *_toward
    trial reproduces the *_away trial simulated with the same seed."""
    plat = synth.default_platform('android', None)
    toward = synth.simulate_trial(symmetric_participant, plat, condition('right', f'{rotation}_toward'), seed=23,
                                  noise=0.0, grip_tilt=False)[0]
    away = synth.simulate_trial(symmetric_participant, plat, condition('right', f'{rotation}_away'), seed=23,
                                noise=0.0, grip_tilt=False)[0]
    fl = augment.screen_away_trial(toward)
    assert fl['condition'] == away['condition']
    np.testing.assert_array_equal(_col(fl, 't'), _col(away, 't'))
    for k in MOTION_VEC:
        np.testing.assert_allclose(_col(fl, k), _col(away, k), atol=2e-3, err_msg=k)
    assert fl['segmentation']['onset_index'] == away['segmentation']['onset_index']
    assert fl['segmentation']['end_index'] == away['segmentation']['end_index']
    assert fl['detected']['rotation_start'] == away['detected']['rotation_start'] == rotation
    assert fl['screen'] == away['screen'] and fl['streams']['touches'] == away['streams']['touches'] == []
    np.testing.assert_allclose(_quat_matrices(fl['streams']['quaternion']), _quat_matrices(away['streams']['quaternion']),
                               atol=1e-5)


# ───────────────────────────── training-fold augmentation ─────────────────────────────


def test_cross_validate_adds_flipped_copies_to_training_folds_only():
    rng = np.random.default_rng(0)
    n = 80
    y = np.tile([0, 1], n // 2)
    groups = np.repeat([f'P{i}' for i in range(8)], n // 8)
    idx = [f't{i}' for i in range(n)]
    X = pd.DataFrame({'x': y * 2.0 - 1 + rng.normal(0, 0.5, n), 'qc_ok': 1}, index=idx)
    Xf = X.copy()
    Xf.loc[idx[::4], 'qc_ok'] = 0  # trials without a usable flipped copy (e.g. not *_toward)
    from sklearn.linear_model import LogisticRegression

    def fac():
        from sklearn.pipeline import make_pipeline
        from sklearn.compose import ColumnTransformer
        return make_pipeline(ColumnTransformer([('x', 'passthrough', ['x'])]), LogisticRegression())

    base = E.cross_validate(fac, X, y, groups, scheme='group_kfold', n_splits=4)
    aug = E.cross_validate(fac, X, y, groups, scheme='group_kfold', n_splits=4, X_screen_flip=Xf, augment_screen_flip=True)
    usable = Xf['qc_ok'].to_numpy() == 1
    for row in aug.folds.to_dict('records'):
        assert row['n_train_fitted'] == row['n_train'] + row['n_train_screen_flipped']
        assert 0 < row['n_train_screen_flipped'] < row['n_train']
    total = sum(r['n_train_screen_flipped'] for r in aug.folds.to_dict('records'))
    assert total == 3 * int(usable.sum())  # each usable copy trains in the 3 folds its trial is not tested in
    assert list(aug.oof.index) == list(base.oof.index) == idx  # test trials are always the real ones
    assert (base.folds['n_train_screen_flipped'] == 0).all()
    with pytest.raises(ValueError):
        E.cross_validate(fac, X, y, groups, scheme='group_kfold', n_splits=4, augment_screen_flip=True)
