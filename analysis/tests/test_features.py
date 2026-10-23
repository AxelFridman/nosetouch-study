"""Features: chiral features flip under mirroring, grip-invariant features do not change under a fixed
random grip rotation, and the gyro-propagated gravity (gravity_v1) behaves."""
from __future__ import annotations

import math

import numpy as np
import pytest
from _mlhelpers import condition

from motionlab import augment, features, load_dataset, synth
from motionlab.conventions import Normalization
from motionlab.preprocess import angle_deg, gyro_gravity, integrate_gyro, random_rotation, rotvec_to_matrix

W3C = Normalization.from_sign(1)


@pytest.fixture(scope='module')
def trials(symmetric_participant):
    out = []
    for k, (o, speed) in enumerate((('upright_toward', 'natural'), ('top_left_toward', 'fast'),
                                     ('upside_down_away', 'natural'), ('flat_up', 'slow'))):
        plat = synth.default_platform('android', np.random.default_rng(k))
        rec, truth = synth.simulate_trial(symmetric_participant, plat, condition('right' if k % 2 else 'left', o, speed),
                                          seed=100 + k)
        out.append((rec, truth))
    return out


def _finite(f, k):
    return isinstance(f[k], float) and math.isfinite(f[k])


def test_chiral_features_flip_under_mirroring(trials):
    for rec, _ in trials:
        f = features.trial_features(rec, norm=W3C)
        assert f['qc_ok'] == 1, f['qc_reason']
        for fm in (features.trial_features(augment.mirror_trial(rec), norm=W3C),  # record-level mirror
                   features.trial_features(rec, norm=W3C, mirror=True)):  # normalised-signal mirror
            for k in features.CHIRAL:
                if _finite(f, k):
                    assert math.isclose(fm[k], -f[k], rel_tol=1e-6, abs_tol=1e-9), k
            for k in features.INVARIANT + features.BASIC:
                if _finite(f, k):
                    assert math.isclose(fm[k], f[k], rel_tol=1e-6, abs_tol=1e-9), k


def test_features_invariant_to_a_fixed_random_grip_rotation(trials):
    rng = np.random.default_rng(42)
    for rec, _ in trials:
        Q = random_rotation(rng)
        f = features.trial_features(rec, norm=W3C)
        fr = features.trial_features(augment.rotate_grip(rec, Q), norm=W3C)
        for k in features.CHIRAL + features.INVARIANT + features.BASIC:
            if _finite(f, k):
                assert math.isclose(fr[k], f[k], rel_tol=1e-6, abs_tol=1e-8), k
        # device-frame features DO change (they encode the grip)
        assert not math.isclose(fr['dev_up_start_y'], f['dev_up_start_y'], abs_tol=1e-3)


def test_chiral_sign_tells_the_hand(trials):
    for rec, truth in trials:
        f = features.trial_features(rec, norm=W3C)
        assert np.sign(f['chi_yaw_net_deg']) == (1 if truth.hand == 'right' else -1)


def test_gyro_gravity_tracks_the_true_gravity_and_closes(trials):
    for rec, _ in trials:
        pt = features.prepare_trial(rec, norm=W3C)
        assert pt.ok and pt.gravity_method == 'gyro_fwd_bwd'
        assert pt.rest_pre_n >= 5 and pt.rest_post_n >= 5
        assert pt.closure_deg < 3.0
        assert pt.browser_angle_mean_deg < 2.0  # the synthetic 'acc' is exact, so accg - acc is true gravity
        assert np.allclose(np.linalg.norm(pt.g[pt.i0:pt.i1 + 1], axis=1), np.linalg.norm(pt.g[pt.i0]))
        np.testing.assert_allclose(pt.acc + pt.g, pt.f, atol=1e-9)  # a = f - g


def test_gyro_gravity_on_an_exact_rotation():
    """A world-fixed gravity seen from a device spinning at a known rate: propagation is exact."""
    t = np.arange(0.0, 2000.0, 10.0)
    w = np.zeros((t.size, 3))
    moving = (t >= 500) & (t < 1500)
    w[moving, 2] = 90.0  # 90 deg/s about device z for 1 s
    R, _ = integrate_gyro(t, w)
    g0 = np.array([0.0, 9.80665, 0.0])
    f = np.einsum('nji,j->ni', R, g0)  # the true specific force at rest: R(k)^T g0
    i0, i1 = int(np.flatnonzero(moving)[0]), int(np.flatnonzero(moving)[-1]) + 1
    gg = gyro_gravity(t, f, w, i0, i1)
    np.testing.assert_allclose(gg.g, f, atol=1e-9)
    assert gg.closure_deg < 1e-6
    # ~90 deg turn of gravity in the device frame (the midpoint rule smooths the rate steps)
    assert math.isclose(float(angle_deg(gg.g[0], gg.g[-1])), 90.0, abs_tol=1e-6)
    # a 5 deg/s gyro bias shows up as a closure error of ~5 deg
    gb = gyro_gravity(t, f, w + np.array([0.0, 5.0, 0.0]), i0, i1)
    assert 3.0 < gb.closure_deg < 7.0
    # rotation and mirror covariance
    Q = rotvec_to_matrix(np.array([0.3, -0.2, 0.5]))
    gq = gyro_gravity(t, f @ Q.T, w @ Q.T, i0, i1)
    np.testing.assert_allclose(gq.g, gg.g @ Q.T, atol=1e-9)
    S = np.array([-1.0, 1.0, 1.0])
    gm = gyro_gravity(t, f * S, w * -S, i0, i1)
    np.testing.assert_allclose(gm.g, gg.g * S, atol=1e-9)


def test_wrong_gravity_sign_flips_every_chiral_feature(trials):
    rec, _ = trials[0]
    f = features.trial_features(rec, norm=Normalization.from_sign(1))
    fw = features.trial_features(rec, norm=Normalization.from_sign(-1))
    for k in features.CHIRAL:
        if _finite(f, k):
            assert math.isclose(fw[k], -f[k], rel_tol=1e-6, abs_tol=1e-9), k


def test_zupt_features_need_short_movements(trials):
    slow = trials[3][0]
    f = features.trial_features(slow, norm=W3C)
    assert f['duration_ms'] > 1500
    for k in features.ZUPT_FEATURES:
        assert math.isnan(f[k]), k
    assert math.isfinite(f['chi_yaw_net_deg'])


def test_sequence_tensor(trials):
    rec, _ = trials[1]
    arr, names = features.sequence_tensor(rec, 64, channels='all', return_names=True, norm=W3C)
    assert arr.shape == (len(names), 64) and arr.dtype == np.float32 and np.isfinite(arr).all()
    mir = features.sequence_tensor(rec, 64, channels='invariant', norm=W3C, mirror=True)
    inv = features.sequence_tensor(rec, 64, channels='invariant', norm=W3C)
    names_inv = features.SEQUENCE_CHANNELS['invariant']
    for j, n in enumerate(names_inv):
        if n in features.CHIRAL_SEQUENCE_CHANNELS:
            np.testing.assert_allclose(mir[j], -inv[j], atol=1e-3)
        else:
            np.testing.assert_allclose(mir[j], inv[j], atol=1e-3)


def test_extract_features_from_a_dataset(tiny_dataset, tiny_tree):
    ds = load_dataset(tiny_tree)
    ids = list(ds.trials.loc[ds.trials['accepted'].astype(bool), 'trial_id'])
    F, Fm = features.extract_features(ds, ids, mirror='both')
    assert list(F.index) == ids and list(Fm.index) == ids
    assert (F['qc_ok'] == 1).all(), F['qc_reason'].value_counts()
    assert set(F['qc_gravity_sign_source']) == {'calibration'}
    np.testing.assert_allclose(Fm['chi_yaw_net_deg'], -F['chi_yaw_net_deg'], rtol=1e-6, atol=1e-9)
    hand = ds.trials.set_index('trial_id').loc[ids, 'hand']
    # iOS sessions (gravity sign -1) and Android ones agree on the chiral sign of each hand
    plat = ds.trials.set_index('trial_id').loc[ids, 'platform']
    for p in ('ios', 'android'):
        sel = plat == p
        right = F.loc[sel & (hand == 'right'), 'chi_yaw_net_deg']
        left = F.loc[sel & (hand == 'left'), 'chi_yaw_net_deg']
        assert (right > 0).mean() > 0.9 and (left < 0).mean() > 0.9, p
    groups = features.feature_groups()
    assert set(groups) == {'basic', 'device', 'invariant', 'chiral'}
    assert all(c in F.columns for g in groups.values() for c in g)
