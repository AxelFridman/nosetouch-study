"""Left/right mirroring and the other record augmentations.

The mirror rule (any grip): acc (x, y, z) -> (-x, y, z); gyro (x, y, z) -> (x, -y, -z), i.e.
rotationRate {alpha kept, beta and gamma negated}; top_left_* <-> top_right_*."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from _mlhelpers import condition

from motionlab import augment, conventions as C, schema, synth
from motionlab.preprocess import quaternion_to_matrix, random_rotation, rotvec_to_matrix

MOTION_VEC = ('acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma')


def _col(rec, k, stream='motion'):
    return np.asarray([np.nan if v is None else v for v in rec['streams'][stream][k]], dtype=float)


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _vee(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def test_mirror_rule_on_random_rigid_body_trajectories():
    """f' = S f and w' = -S w for R' = M R S, p' = M p, for random grips and movements (docs/05 3.1)."""
    rng = np.random.default_rng(3)
    M = S = C.MIRROR_X
    g_up = np.array([0.0, C.GRAVITY, 0.0])
    h = 1e-5
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
            Rm = lambda s: M @ R(s) @ S  # noqa: E731
            Rmd = (Rm(t + h) - Rm(t - h)) / (2 * h)
            fm = Rm(t).T @ (M @ pdd + g_up)
            wm = _vee(Rm(t).T @ Rmd)
            np.testing.assert_allclose(fm, S @ f, atol=1e-4)
            np.testing.assert_allclose(wm, -S @ w, atol=1e-4)
            # the helper used by the feature pipeline implements exactly this
            a2, w2_ = C.mirror_device_vectors(f[None, :], w[None, :])
            np.testing.assert_allclose(a2[0], S @ f)
            np.testing.assert_allclose(w2_[0], -S @ w)


def test_mirror_trial_fields_and_validity(tiny_dataset):
    rec = tiny_dataset.trials[1]
    m = augment.mirror_trial(rec)
    assert schema.validate(m, 'TrialRecord') == []
    np.testing.assert_array_equal(_col(m, 'acc_x'), -_col(rec, 'acc_x'))
    np.testing.assert_array_equal(_col(m, 'accg_y'), _col(rec, 'accg_y'))
    np.testing.assert_array_equal(_col(m, 'rr_alpha'), _col(rec, 'rr_alpha'))  # x kept
    np.testing.assert_array_equal(_col(m, 'rr_beta'), -_col(rec, 'rr_beta'))  # y negated
    np.testing.assert_array_equal(_col(m, 'rr_gamma'), -_col(rec, 'rr_gamma'))  # z negated
    assert m['condition']['hand'] != rec['condition']['hand']
    assert m['condition']['orientation'] == C.mirror_orientation(rec['condition']['orientation'])
    # the legacy mapping keeps beta (its x) and negates gamma and alpha
    ml = augment.mirror_trial(rec, rr_verdict='legacy')
    np.testing.assert_array_equal(_col(ml, 'rr_beta'), _col(rec, 'rr_beta'))
    np.testing.assert_array_equal(_col(ml, 'rr_alpha'), -_col(rec, 'rr_alpha'))
    assert rec['streams']['motion']['acc_x'][0] == tiny_dataset.trials[1]['streams']['motion']['acc_x'][0]  # input untouched


def test_mirror_is_an_involution(tiny_dataset):
    for rec in tiny_dataset.trials[:6]:
        mm = augment.mirror_trial(augment.mirror_trial(rec))
        assert json.dumps(mm['streams']['motion']) == json.dumps(rec['streams']['motion'])
        assert mm['condition'] == rec['condition']
        assert mm['detected'] == rec['detected'] and mm['screen'] == rec['screen']
        assert mm['segmentation']['baseline'] == rec['segmentation']['baseline']
        q0, q2 = rec['streams']['quaternion'], mm['streams']['quaternion']
        if q0:
            for c in 'xyzw':
                np.testing.assert_allclose(q2[c], q0[c])
        # Euler angles: same rotation (angles may differ by range normalisation)
        o0, o2 = rec['streams']['orientation'], mm['streams']['orientation']
        from motionlab.preprocess import euler_to_matrix
        R0 = euler_to_matrix(o0['alpha'], o0['beta'], o0['gamma'])
        R2 = euler_to_matrix(o2['alpha'], o2['beta'], o2['gamma'])
        np.testing.assert_allclose(R2, R0, atol=1e-9)


@pytest.mark.parametrize('orientation, mapping', [('upright_toward', 'w3c'), ('top_left_toward', 'w3c'),
                                                  ('top_right_away', 'w3c'), ('upright_toward', 'legacy')])
def test_mirror_of_right_equals_left_on_noise_free_synth(symmetric_participant, orientation, mapping):
    plat = synth.default_platform('android', None, rr_mapping=mapping)
    right = synth.simulate_trial(symmetric_participant, plat, condition('right', orientation), seed=31, noise=0.0)[0]
    left_cond = augment.mirror_condition(condition('right', orientation))
    left = synth.simulate_trial(symmetric_participant, plat, left_cond, seed=31, noise=0.0)[0]
    mirrored = augment.mirror_trial(right, rr_verdict=mapping)
    assert mirrored['condition'] == left['condition']
    np.testing.assert_array_equal(_col(mirrored, 't'), _col(left, 't'))
    for k in MOTION_VEC:
        np.testing.assert_allclose(_col(mirrored, k), _col(left, k), atol=2e-3, err_msg=k)
    assert mirrored['segmentation']['onset_index'] == left['segmentation']['onset_index']
    assert mirrored['detected']['rotation_start'] == left['detected']['rotation_start']
    np.testing.assert_allclose(mirrored['segmentation']['baseline']['gravity'], left['segmentation']['baseline']['gravity'],
                               atol=1e-3)
    # world-referenced orientation: the mirrored attitude matches the left trial's up to the heading
    qm = mirrored['streams']['quaternion']
    ql = left['streams']['quaternion']
    Rm = quaternion_to_matrix(np.column_stack([qm[c] for c in 'xyzw']))
    Rl = quaternion_to_matrix(np.column_stack([ql[c] for c in 'xyzw']))
    np.testing.assert_allclose(Rm[:, 2, :], Rl[:, 2, :], atol=1e-4)  # gravity row (earth z in device axes)


def test_rotate_grip_and_tilt(tiny_dataset):
    rec = tiny_dataset.trials[2]
    Q = random_rotation(np.random.default_rng(5))
    r = augment.rotate_grip(rec, Q)
    assert schema.validate(r, 'TrialRecord') == []
    A = np.column_stack([_col(rec, f'accg_{a}') for a in 'xyz'])
    Ar = np.column_stack([_col(r, f'accg_{a}') for a in 'xyz'])
    np.testing.assert_allclose(Ar, A @ Q.T, atol=1e-9)
    W = np.column_stack([_col(rec, c) for c in C.RR_FIELDS])
    Wr = np.column_stack([_col(r, c) for c in C.RR_FIELDS])
    np.testing.assert_allclose(Wr, W @ Q.T, atol=1e-9)  # W3C: the rate vector rotates like any vector
    np.testing.assert_allclose(np.linalg.norm(Wr, axis=1), np.linalg.norm(W, axis=1), atol=1e-9)
    with pytest.raises(ValueError):
        augment.rotate_grip(rec, np.diag([-1.0, 1.0, 1.0]))  # a reflection is not a grip
    t = augment.random_grip_tilt(rec, 15.0, rng=np.random.default_rng(1))
    Ag = np.column_stack([_col(t, f'accg_{a}') for a in 'xyz'])
    ang = np.degrees(np.arccos(np.clip(np.sum(Ag * A, axis=1) / (np.linalg.norm(Ag, axis=1) * np.linalg.norm(A, axis=1)), -1, 1)))
    assert np.nanmax(ang) <= 15.0 + 1e-6
    assert t['detected'] == rec['detected']


def test_time_warp_and_scaling(tiny_dataset):
    rec = tiny_dataset.trials[0]
    w = augment.time_warp(rec, 1.25)
    assert schema.validate(w, 'TrialRecord') == []
    np.testing.assert_allclose(_col(w, 't'), 1.25 * _col(rec, 't'))
    np.testing.assert_allclose(_col(w, 'rr_beta'), _col(rec, 'rr_beta') / 1.25)
    np.testing.assert_allclose(_col(w, 'acc_y'), _col(rec, 'acc_y') / 1.25 ** 2)
    grav = _col(rec, 'accg_y') - _col(rec, 'acc_y')
    np.testing.assert_allclose(_col(w, 'accg_y') - _col(w, 'acc_y'), grav, atol=1e-9)  # gravity part kept
    assert w['timing']['go_ms'] == pytest.approx(1.25 * rec['timing']['go_ms'])
    s = augment.scale_magnitudes(rec, 1.1, 0.9)
    np.testing.assert_allclose(_col(s, 'acc_z'), 1.1 * _col(rec, 'acc_z'))
    np.testing.assert_allclose(_col(s, 'accg_z') - _col(s, 'acc_z'), _col(rec, 'accg_z') - _col(rec, 'acc_z'), atol=1e-9)
    np.testing.assert_allclose(_col(s, 'rr_gamma'), 0.9 * _col(rec, 'rr_gamma'))


def test_quantize_like_chrome(tiny_dataset):
    ios_sid = next(s['session_id'] for s in tiny_dataset.sessions if s['device']['platform'] == 'ios')
    ios = next(t for t in tiny_dataset.trials if t['session_id'] == ios_sid)
    q = augment.quantize_trial(ios)
    assert schema.validate(q, 'TrialRecord') == []
    for k in ('acc_x', 'accg_z'):
        v = _col(q, k)
        np.testing.assert_allclose(v * 10, np.round(v * 10), atol=1e-6)
        assert np.nanmax(np.abs(v - _col(ios, k))) <= 0.05 + 1e-9
    rr = _col(q, 'rr_beta')
    np.testing.assert_allclose(rr * 10, np.round(rr * 10), atol=1e-3)
    assert np.nanmax(np.abs(rr - _col(ios, 'rr_beta'))) <= 0.05 + 1e-6
    # symmetric in sign: quantising the inverted iOS values == inverting the quantised values
    neg = augment.quantize_trial({**ios, 'streams': {**ios['streams'], 'motion': {
        **ios['streams']['motion'], 'accg_z': [-v for v in ios['streams']['motion']['accg_z']]}}})
    np.testing.assert_array_equal(_col(neg, 'accg_z'), -_col(q, 'accg_z') + 0.0)
    a = _col(q, 'alpha', 'orientation')
    assert np.all((a >= 0) & (a < 360))


def test_add_noise_keeps_gravity(tiny_dataset):
    rec = tiny_dataset.trials[0]
    n = augment.add_noise(rec, rng=np.random.default_rng(0))
    np.testing.assert_allclose(_col(n, 'accg_x') - _col(n, 'acc_x'), _col(rec, 'accg_x') - _col(rec, 'acc_x'), atol=1e-9)
    assert not math.isclose(float(_col(n, 'rr_alpha')[10]), float(_col(rec, 'rr_alpha')[10]))
