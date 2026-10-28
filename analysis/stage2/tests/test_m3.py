"""Lane m3 (time-frequency and deep sequence models): fast checks of the physics, the symmetry guarantees, the
from-scratch MiniRocket, the calibrators (the in-fold choice included), the fold-fit cache and its runtime key,
thread-independent deep fits, leak-free and deterministic fit_predicts, a person's labels never reaching its own
predictions, the statistics, that every figure redraws from saved files without fitting anything, and that the
summary cites only numbers of the numbers file."""
from __future__ import annotations

import os

os.environ.setdefault('M3_THREADS', '1')

import json  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from stage2 import config as C  # noqa: E402
from stage2 import cv  # noqa: E402
from stage2.lanes import m3_common as M  # noqa: E402
from stage2.lanes import m3_minirocket as MR  # noqa: E402
from stage2.lanes import m3_tf as TF  # noqa: E402
from stage2.standardize import CHANNELS, MIRROR_SIGN  # noqa: E402


def _synthetic(n_persons: int = 6, per: int = 12, T: int = 32, seed: int = 0):
    """All 17 channels; the right hand turns counter-clockwise early (yaw_rate > 0 in the first half)."""
    rng = np.random.default_rng(seed)
    n = n_persons * per
    y = np.tile(np.r_[np.zeros(per // 2), np.ones(per // 2)], n_persons).astype(int)
    X = rng.normal(0, 1, (n, len(CHANNELS), T)).astype(np.float32)
    k = CHANNELS.index('yaw_rate')
    bump = np.sin(np.linspace(0, np.pi, T // 2))
    X[:, k, : T // 2] += ((2 * y - 1) * 3)[:, None] * bump[None, :]
    X[:, CHANNELS.index('yaw_cum')] = np.cumsum(X[:, k], axis=1) / 10
    groups = np.repeat([f'P{i + 1:02d}' for i in range(n_persons)], per)
    return X, y, groups


def _meta(y, groups):
    return pd.DataFrame({'trial_id': [f't{i}' for i in range(len(y))], 'person_id': groups, 'y_right': y,
                         'analysis_set': 'primary'})


# ───────────────────────────── physics and symmetry ─────────────────────────────


def test_tf_magnitude_is_mirror_invariant_and_signed_parts_flip():
    X, _, _ = _synthetic(T=64)
    k = [CHANNELS.index('yaw_rate')]
    for f in (TF.stft_planes, TF.cwt_planes):
        P, Pm = f(X[:5, k]), f(-X[:5, k])
        assert np.allclose(Pm, -P, atol=1e-5)
        assert np.allclose(np.hypot(P[:, :, 0], P[:, :, 1]), np.hypot(Pm[:, :, 0], Pm[:, :, 1]), atol=1e-5)
    F = TF.pooled(TF.stft_planes(X[:5]), 'stft', 'mag')
    assert np.allclose(F, TF.pooled(TF.stft_planes(X[:5] * MIRROR_SIGN[None, :, None]), 'stft', 'mag'), atol=1e-5)


def test_normaliser_commutes_with_the_mirror():
    X, _, _ = _synthetic()
    for ch in ('grav', 'all', 'chiral'):
        s = M.mirror_signs(ch)
        Xs = X[:, M.channel_idx(ch)]
        for mode in ('zscore', 'robust', 'trial', 'none'):
            nm = M.Normaliser(mode).fit(Xs, s)
            assert np.allclose(nm.transform(Xs * s[None, :, None]), nm.transform(Xs) * s[None, :, None], atol=1e-5)


def test_antisymmetric_deep_model_is_exactly_antisymmetric():
    from stage2.lanes.m3_deep import DeepFamily
    X, y, g = _synthetic()
    fam = DeepFamily(anti=True, n_inner=2, max_epochs=2, min_epochs=1, width=8, n_bins=4)
    splits = cv.inner_splits(g, 2)
    st = fam.fit({'x': X}, y, g, splits, seed=1)
    z = fam.logits(st, {'x': X})
    zm = fam.logits(st, {'x': X * MIRROR_SIGN[None, :, None]})
    assert np.allclose(zm, -z, atol=1e-5)
    assert np.isfinite(st['oof_z']).all() and st['oof_z'].shape == y.shape
    ig = fam.integrated_gradients(st, {'x': X[:4]}, steps=8)
    igm = fam.integrated_gradients(st, {'x': X[:4] * MIRROR_SIGN[None, :, None]}, steps=8)
    assert ig.shape == (4, 8, X.shape[-1]) and np.allclose(igm, -ig, atol=1e-4)     # IG(mirror x) = -IG(x)


def test_deep_shapes_for_every_architecture_and_input():
    import torch

    from stage2.lanes import m3_nets
    for arch in ('cnn', 'resnet', 'inception', 'tcn', 'gru', 'transformer'):
        for T in (64, 180):
            net = m3_nets.build_net(arch, 9, T, width=16, n_bins=8)
            net.eval()
            assert net(torch.zeros(3, 9, T)).shape == (3,)
    net = m3_nets.build_net('cnn2d', 16, 0, width=16, n_bins=8)
    net.eval()
    assert net(torch.zeros(2, 16, 10, 64)).shape == (2,)


def _loop_minirocket(mr: MR.MiniRocket, X: np.ndarray) -> np.ndarray:
    """A plain per-combination loop (explicit dilated 9-tap convolution) to check the vectorised transform."""
    N, Cc, T = X.shape
    out = []
    for di, d in enumerate(mr.dilations):
        pad = ((9 - 1) * int(d)) // 2
        for ki, kern in enumerate(MR.KERNELS):
            w = -np.ones(9)
            w[kern] = 2.0
            ch = np.flatnonzero(mr.masks[di, ki])
            c = np.zeros((N, T))
            for t in range(T):
                for j in range(9):
                    s = t + (j - 4) * int(d)
                    if 0 <= s < T:
                        c[:, t] += w[j] * X[:, ch, s].sum(axis=1)
            if (di + ki) % 2 == 1 and pad > 0 and T - 2 * pad > 0:
                c = c[:, pad:T - pad]
            out.append((c[:, :, None] > mr.biases[di][ki][None, None, :]).mean(axis=1))
    return np.concatenate(out, axis=1)


def test_minirocket_matches_a_plain_loop_and_its_biases_come_from_the_fit_rows():
    X, _, _ = _synthetic(n_persons=2, per=4, T=24)
    X = X[:, :5]
    mr = MR.MiniRocket(num_features=400, seed=3).fit(X)
    F = mr.transform(X)
    assert F.shape == (len(X), mr.n_features) and 0 <= F.min() and F.max() <= 1
    ref = _loop_minirocket(mr, X.astype(np.float64))
    assert np.mean(np.abs(F - ref) > 1e-6) < 1e-3          # float32 vs float64 ties at a bias are rare
    mr2 = MR.MiniRocket(num_features=400, seed=3).fit(X + 100)
    assert not np.allclose(np.concatenate(mr.biases), np.concatenate(mr2.biases))
    assert len(MR.KERNELS) == 84 and len({tuple(k) for k in MR.KERNELS}) == 84


def test_antisymmetric_minirocket_is_exactly_antisymmetric():
    X, y, g = _synthetic(T=24)
    fam = MR.MiniRocketFamily(mode='anti', num_features=300, Cs=(1e-2,))
    st = fam.fit({'x': X}, y, g, cv.inner_splits(g, 2), seed=0)
    z = fam.logits(st, {'x': X})
    zm = fam.logits(st, {'x': X * MIRROR_SIGN[None, :, None]})
    assert np.allclose(zm, -z, atol=1e-8)


# ───────────────────────────── the body frame (m3_body) ─────────────────────────────


def _device_motion(n: int = 5, T: int = 16, seed: int = 4) -> np.ndarray:
    """(n, 17, T) canonical arrays with a physically consistent device frame: unit up vectors that keep the screen
    off horizontal, random w and a; the gravity-frame slots are left at 0 (body_channels does not read them)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(CHANNELS), T))
    u = rng.normal(0, 1, (n, 3, T))
    u[:, 1] += 3.0                                                # mostly along the top edge (upright-like)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for k, name in enumerate(('up_x', 'up_y', 'up_z')):
        X[:, CHANNELS.index(name)] = u[:, k]
    for name in ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'):
        X[:, CHANNELS.index(name)] = rng.normal(0, 1, (n, T))
    return X


def test_body_channels_are_grip_invariant_and_have_the_documented_mirror_parity():
    from stage2.lanes import m3_body as B
    X = _device_motion()
    ref = B.body_channels(X)
    for theta in (np.pi / 2, -np.pi / 2, 0.65):                  # the two rotated grips and one in between
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # device coordinates after turning about e_z
        Xr = X.copy()
        for trip in (('acc_x', 'acc_y', 'acc_z'), ('gyro_x', 'gyro_y', 'gyro_z'), ('up_x', 'up_y', 'up_z')):
            idx = [CHANNELS.index(t) for t in trip]
            Xr[:, idx] = np.einsum('ij,njt->nit', R, X[:, idx])
        assert np.allclose(B.body_channels(Xr), ref, atol=1e-5)
    signs = np.array([B.BODY_SIGN[c] for c in B.BODY_CHANNELS], dtype=float)[None, :, None]
    assert np.allclose(B.body_channels(X * MIRROR_SIGN[None, :, None]), ref * signs, atol=1e-5)
    assert set(B.INVARIANT_ODD) == {'yaw_rate', 'yaw_cum', 'roll_rate', 'acc_lat'}


def test_body_family_selects_its_channels_and_commutes_with_the_mirror():
    from stage2.lanes import m3_body as B
    X, y, g = _synthetic(T=32)
    Xb = np.concatenate([X, B.body_channels(X)], axis=1).astype(np.float32)
    fam = B.BodyDeepFamily(channels='body', anti=True, n_inner=2, max_epochs=2, min_epochs=1, width=8, n_bins=4)
    assert fam.channel_names() == B.BODY_SET and fam._select({'x': Xb}).shape == (len(y), len(B.BODY_SET), 32)
    st = fam.fit({'x': Xb}, y, g, cv.inner_splits(g, 2), seed=2)
    ext = np.array([B.EXT_SIGN[c] for c in B.EXT_CHANNELS], dtype=np.float32)[None, :, None]
    assert np.allclose(fam.logits(st, {'x': Xb * ext}), -fam.logits(st, {'x': Xb}), atol=1e-5)
    with pytest.raises(ValueError):
        fam._select({'x': X})                                    # a 17-channel input is refused, not misread


def test_nested_choice_reuses_each_candidates_own_cached_fit(tmp_path, monkeypatch):
    from stage2.lanes import m3_body as B
    from stage2.lanes import m3_nested as N
    from stage2.lanes.m3_deep import DeepFamily
    monkeypatch.setattr(M, 'FIT_CACHE_DIR', tmp_path / 'cache')
    X, y, g = _synthetic(T=32)
    Xb = np.concatenate([X, B.body_channels(X)], axis=1).astype(np.float32)
    small = dict(n_inner=2, max_epochs=2, min_epochs=1, width=8, n_bins=4)
    cands = {'a': {'family': 'deep', 'x': 'approach', 'params': dict(small)},
             'b': {'family': 'body', 'x': 'approach_body', 'params': dict(small, channels='body')}}
    splits = cv.inner_splits(g, 2)
    own = {'a': M.cached_fit(DeepFamily(**small), {'x': X}, y, g, splits, 3),
           'b': M.cached_fit(B.BodyDeepFamily(**small, channels='body'), {'x': Xb}, y, g, splits, 3)}
    nest = N.NestedSelectMulti(cands)
    Xm = {'approach__x': X, 'approach_body__x': Xb}
    st = nest.fit(Xm, y, g, splits, 3)
    ll = {k: float(np.mean(np.logaddexp(0, -(2 * y - 1) * v['oof_z']))) for k, v in own.items()}
    assert st['choice'] == min(ll, key=lambda k: (ll[k], k))
    assert len(list((tmp_path / 'cache').rglob('*.pkl'))) == 2     # no candidate was refitted
    assert np.allclose(nest.logits(st, Xm), (DeepFamily(**small) if st['choice'] == 'a' else
                                             B.BodyDeepFamily(**small, channels='body')).logits(
        own[st['choice']], {'x': X if st['choice'] == 'a' else Xb}))


# ───────────────────────────── calibration ─────────────────────────────


def test_calibrators_fit_on_given_logits_and_temperature_keeps_antisymmetry():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < M.sigmoid(z_true)).astype(int)
    z = 3 * z_true                                              # over-confident logits
    ll = lambda p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
    base = ll(np.clip(M.sigmoid(z), 1e-12, 1 - 1e-12))
    for m in ('platt', 'temperature', 'isotonic', 'beta'):
        par = M.fit_calibrator(m, z, y)
        p = M.apply_calibrator(par, z)
        assert np.all((p > 0) & (p < 1)) and ll(p) < base
    par = M.fit_calibrator('temperature', z, y)
    assert 2.5 < par['T'] < 3.5
    assert np.allclose(M.apply_calibrator(par, -z), 1 - M.apply_calibrator(par, z))
    iso = M.fit_calibrator('isotonic', z, y)
    assert np.all(np.diff(M.apply_calibrator(iso, np.sort(z))) >= -1e-12)


def test_sum_scale_shrinks_correlated_attempts():
    rng = np.random.default_rng(1)
    groups = np.repeat([f'P{i}' for i in range(30)], 10)
    y = np.tile(np.r_[np.zeros(5), np.ones(5)], 30).astype(int)
    person = np.repeat(rng.normal(0, 2, 60), 5)                 # a shared per-person-hand offset: correlated attempts
    z = (2 * y - 1) * 1.0 + person + rng.normal(0, 0.3, y.size)
    c1 = M.fit_sum_scale(z, y, groups, 1, seed=0)
    c5 = M.fit_sum_scale(z, y, groups, 5, seed=0)
    assert c5 < c1                                              # the sum of 5 correlated logits needs shrinking


# ───────────────────────────── harness integration ─────────────────────────────


def test_fit_predict_is_leak_free_deterministic_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'FIT_CACHE_DIR', tmp_path / 'cache')
    X, y, g = _synthetic(T=64)
    P = TF.stft_planes(X)
    meta = _meta(y, g)
    fp = M.make_fit_predict(TF.TFLogisticFamily(Cs=(0.1, 1.0)), calib='temperature')
    kw = dict(save=False, n_boot=50, leak_check='all', registry=None, cap_per_person=None)
    r1 = cv.run_experiment('m3_test_tflog', fp, {'x': P}, meta, **kw)
    assert r1.leak_check.startswith('passed') and r1.metrics['accuracy'] > 0.8
    n_cached = len(list((tmp_path / 'cache').rglob('*.pkl')))
    assert n_cached == len(set(g))                               # one cached fit per outer fold
    r2 = cv.run_experiment('m3_test_tflog', fp, {'x': P}, meta, **kw)
    assert np.allclose(r1.predictions['p_right'], r2.predictions['p_right'])
    info = json.loads(r1.predictions['params'].iloc[0])
    assert info['calib']['method'] == 'temperature' and set(info['sum_scale']) == {str(k) for k in range(1, max(C.K_ATTEMPTS) + 1)}
    fp_nocache = M.make_fit_predict(TF.TFLogisticFamily(Cs=(0.1, 1.0)), calib='temperature', use_cache=False)
    r3 = cv.run_experiment('m3_test_tflog', fp_nocache, {'x': P}, meta, **kw)
    assert np.allclose(r1.predictions['p_right'], r3.predictions['p_right'])


def test_a_persons_own_labels_never_reach_its_predictions(tmp_path, monkeypatch):
    """Through the harness: flipping every label of one person leaves that person's held-out predictions unchanged
    (its labels reach only the other persons' folds), while the other persons' predictions do move (the test can fail)."""
    monkeypatch.setattr(M, 'FIT_CACHE_DIR', tmp_path / 'cache')
    X, y, g = _synthetic(T=64)
    P = {'x': TF.stft_planes(X)}
    kw = dict(save=False, n_boot=20, leak_check='off', registry=None, cap_per_person=None)
    fp = M.make_fit_predict(TF.TFLogisticFamily(Cs=(0.1, 1.0)), calib='nested', use_cache=False)
    r1 = cv.run_experiment('m3_test_labels', fp, P, _meta(y, g), **kw)
    y2 = y.copy()
    y2[g == 'P03'] = 1 - y2[g == 'P03']
    r2 = cv.run_experiment('m3_test_labels', fp, P, _meta(y2, g), **kw)
    a, b = r1.predictions.set_index('trial_id')['p_right'], r2.predictions.set_index('trial_id')['p_right']
    own = r1.predictions.set_index('trial_id')['person_id'] == 'P03'
    assert np.array_equal(a[own].to_numpy(), b[own].to_numpy())
    assert np.max(np.abs(a[~own].to_numpy() - b[~own].to_numpy())) > 1e-6


# ───────────────────────────── reproducibility and the fit cache ─────────────────────────────


def test_deep_fits_do_not_depend_on_the_process_thread_count():
    import torch

    from stage2.lanes.m3_deep import DeepFamily
    X, y, g = _synthetic(T=32)
    fam = DeepFamily(n_inner=2, max_epochs=3, min_epochs=1, width=8, n_bins=4)
    splits = cv.inner_splits(g, 2)
    n0 = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        s3 = fam.fit({'x': X}, y, g, splits, seed=5)
        z3 = fam.logits(s3, {'x': X})
        torch.set_num_threads(1)
        s1 = fam.fit({'x': X}, y, g, splits, seed=5)
        z1 = fam.logits(s1, {'x': X})
    finally:
        torch.set_num_threads(n0)
    assert np.array_equal(s3['oof_z'], s1['oof_z']) and np.array_equal(z3, z1)
    assert torch.get_num_threads() == n0 and not torch.are_deterministic_algorithms_enabled()   # settings restored


def test_the_fit_cache_key_holds_the_runtime_and_the_settings_reach_wrapper_fits(tmp_path, monkeypatch):
    from stage2.lanes import m3_deep
    from stage2.lanes import m3_nested as N
    monkeypatch.setattr(M, 'FIT_CACHE_DIR', tmp_path / 'cache')
    X, y, g = _synthetic(T=32)
    small = dict(n_inner=2, max_epochs=1, min_epochs=1, width=8, n_bins=4)
    fam = m3_deep.DeepFamily(**small)
    splits = cv.inner_splits(g, 2)
    assert {'torch', 'train_threads', 'numpy', 'sklearn'} <= set(M.runtime_signature(fam))
    M.cached_fit(fam, {'x': X}, y, g, splits, 1)
    monkeypatch.setattr(m3_deep, 'TRAIN_THREADS', 2)                       # another runtime: another key
    M.cached_fit(fam, {'x': X}, y, g, splits, 1)
    assert len(list((tmp_path / 'cache').rglob('*.pkl'))) == 2
    monkeypatch.setattr(m3_deep, 'TRAIN_THREADS', 1)
    nest = N.NestedSelectMulti({'a': {'family': 'deep', 'x': 'approach', 'params': dict(small, width=6)}})
    with M.cache_settings(use_cache=False):                                # no pickle for the candidate either
        nest.fit({'approach__x': X}, y, g, splits, 2)
    assert len(list((tmp_path / 'cache').rglob('*.pkl'))) == 2
    with M.cache_settings(cache_dir=tmp_path / 'other'):
        nest.fit({'approach__x': X}, y, g, splits, 2)
    assert len(list((tmp_path / 'other').rglob('*.pkl'))) == 1


def test_nested_calibrator_is_chosen_inside_the_fold():
    rng = np.random.default_rng(3)
    z_true = rng.normal(0, 2, 600)
    y = (rng.random(600) < M.sigmoid(z_true)).astype(int)
    groups = np.repeat([f'P{i:02d}' for i in range(12)], 50)
    par = M.fit_calibrator('nested', 3 * z_true, y, groups)                # over-confident logits
    assert par['nested_choice'] != 'none' and par['method'] == par['nested_choice']
    assert set(par['nested_inner_log_loss']) == set(M.NESTED_CAL_CANDIDATES)
    with pytest.raises(ValueError):
        M.fit_calibrator('nested', z_true, y)                              # needs the persons


# ───────────────────────────── statistics ─────────────────────────────


def test_stats_holm_verdict_natural_use_and_ece_null():
    from stage2.lanes import m3_stats as S
    assert np.allclose(S.holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan], equal_nan=True)
    assert S.verdict(0.05, 0.01, 0.09, 0.02) == 'better' and S.verdict(0.05, 0.01, 0.09, 0.2) == 'no detectable difference'
    assert S.verdict(-0.05, -0.09, -0.01, 0.01) == 'worse' and S.verdict(0, 0, 0, 1.0, same=True) == 'identical'
    nu = S.natural_use(np.ones(20, int), np.full(20, 0.5))                 # a constant 0.5 abstains, never 'right'
    assert nu['coverage'] == 0 and nu['forced_accuracy'] == 0 and nu['half_credit_accuracy'] == 0.5
    lo, hi = S.binom_ci(17, 20)
    assert abs(lo - 0.621) < 1e-3 and abs(hi - 0.968) < 1e-3
    ps, over = [], []
    for seed in range(1, 6):                     # calibrated predictions sit inside their null band (most of the time)
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.05, 0.95, 3000)
        y = (rng.random(3000) < p).astype(int)
        ps.append(S.ece_null(y, p, n_sim=300)['ece_null_p'])
        over.append(S.ece_null(y, M.sigmoid(3 * M.logit(p)), n_sim=300)['ece_null_p'])   # over-confident: outside
    assert np.median(ps) > 0.2 and max(over) < 0.01
    rc = S.recalibration(y, p)
    assert abs(rc['cal_slope'] - 1) < 0.2 and abs(rc['cal_intercept']) < 0.2
    assert S.recalibration(y, M.sigmoid(3 * M.logit(p)))['cal_slope'] < 0.5
    d = np.r_[np.full(10, 0.1), np.full(10, 0.05)]
    st = S.person_mean_diff(d, n_boot=200)
    assert st['lo'] > 0 and st['p'] < 0.01


def test_reliability_bins_ignore_degenerate_bands_and_flag_real_miscalibration():
    """A bin whose trials all have one hand has a zero-width person-bootstrap band; it counts as off only when the
    observed share is also outside the perfect-calibration consistency band. Over-confident predictions of the same
    labels are flagged."""
    from stage2.lanes import m3_derive as Dv
    levels = np.array([0.002, 0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95, 0.998])
    n = 60
    p = np.repeat(levels, n)
    y = np.concatenate([(np.arange(n) < round(q * n)).astype(int) for q in levels])    # observed share = rounded p
    persons = np.array([f'P{i % 20:02d}' for i in range(p.size)])
    rel = Dv.reliability_bands(y, p, persons)
    assert rel['degenerate_band'].any() and rel['off_boot'].any()        # the extreme bins: all one hand
    assert not rel['off'].any()                                           # ... yet consistent with their predictions
    assert 0 <= rel.attrs['bins_off_null_rate'] <= 1
    over = Dv.reliability_bands(y, M.sigmoid(3 * M.logit(p)), persons)     # the same labels, over-confident
    assert over['off'].any()


def test_aggregator_rule_prefers_the_designed_aggregator_when_it_is_calibrated():
    from stage2.lanes import m3_derive as Dv

    def blk(oc: dict) -> dict:
        return {'calibrated': {'rows': [{'k': 3, 'aggregator': a, 'overconfidence_lo': lo, 'overconfidence_hi': hi,
                                         'log_loss': ll} for a, (lo, hi, ll) in oc.items()]}}
    ok = {'mean_prob': (-0.2, -0.1, 0.30), 'mean_logit': (-0.1, -0.05, 0.28), 'sum_logit': (-0.01, 0.04, 0.237),
          'sum_logit_calibrated': (-0.01, 0.03, 0.239)}
    rec = Dv.recommend_aggregator({'deep': blk(ok), 'mr': blk(ok)})
    assert rec['aggregator'] == Dv.PREFERRED_AGG                         # preferred although its log loss is higher
    bad = dict(ok, sum_logit_calibrated=(0.01, 0.05, 0.239))               # over-confident: not kept
    assert Dv.recommend_aggregator({'deep': blk(ok), 'mr': blk(bad)})['aggregator'] == 'sum_logit'


def test_phase2_results_go_stale_when_a_source_is_saved_after_them(monkeypatch):
    from stage2.lanes import m3_derive as Dv
    t = {'base': '2026-01-02T00:00:00+00:00', 'cal': '2026-01-01T00:00:00+00:00'}
    monkeypatch.setattr(Dv, 'created_at', lambda n: t.get(n, ''))
    monkeypatch.setattr(Dv, 'exists', lambda n: n in t)
    assert Dv.saved_before_a_source('cal', ['base'])                       # the base was re-run after the variant
    t['cal'] = '2026-01-03T00:00:00+00:00'
    assert not Dv.saved_before_a_source('cal', ['base', 'missing'])


# ───────────────────────────── presentation from saved files ─────────────────────────────


DERIVED_FILES = ['m3_summary.json', 'm3_selection.json', 'm3_paired.json', 'm3_calibration.json', 'm3_tf_example.json',
                 'm3_aggregate.json', 'm3_breakdown.json', 'm3_saliency.json', 'm3_grip_transfer.json']


def test_every_figure_redraws_from_saved_files_without_fitting(tmp_path, monkeypatch):
    src = C.TAB_DIR / 'm3'
    if not all((src / f).exists() for f in DERIVED_FILES):
        pytest.skip('run python -m stage2.pipeline compute --lanes m3 first')
    shutil.copytree(src, tmp_path / 'tables' / 'm3')
    monkeypatch.setattr(C, 'TAB_DIR', tmp_path / 'tables')
    monkeypatch.setattr(C, 'FIG_DIR', tmp_path / 'figures')

    def boom(*a, **k):
        raise AssertionError('a figure tried to fit a model')

    monkeypatch.setattr(M, 'cached_fit', boom)
    from stage2.lanes import m3_figures
    done = m3_figures.draw()
    assert done == list(m3_figures.FIGURE_FUNCTIONS)
    for n in done:
        assert (tmp_path / 'figures' / 'm3' / f'{n}.png').exists()
        assert (tmp_path / 'tables' / 'm3' / f'{n}.csv').exists()
    caps = json.loads((tmp_path / 'figures' / 'm3' / 'captions.json').read_text(encoding='utf-8'))
    assert set(done) <= set(caps) and all(caps[n]['caption'] and caps[n]['decision'] for n in done)


def test_summary_prose_cites_only_numbers_of_the_numbers_file(tmp_path, monkeypatch):
    """write_summary formats every prose number through the numbers file (a missing key raises), so regenerating it
    from the saved files proves that each cited number exists in results/m3_numbers.json."""
    if not ((C.RES_DIR / 'm3_numbers.json').exists() and all((C.TAB_DIR / 'm3' / f).exists() for f in DERIVED_FILES)):
        pytest.skip('run python -m stage2.pipeline compute and numbers --lanes m3 first')
    monkeypatch.setattr(C, 'OUT_DIR', tmp_path)
    from stage2.lanes import m3_summary
    text = m3_summary.write_summary()
    assert (tmp_path / 'm3_SUMMARY.md').read_text(encoding='utf-8') == text and '## Conclusions' in text
    with pytest.raises(KeyError):
        m3_summary.Numbers(C.RES_DIR / 'm3_numbers.json')('no_such_key')


def test_saved_results_follow_the_lane_contract():
    reg = C.RES_DIR / 'registry_m3.csv'
    if not reg.exists():
        pytest.skip('run python -m stage2.pipeline compute --lanes m3 first')
    df = pd.read_csv(reg)
    assert df['name'].str.startswith('m3_').all()
    assert df['leak_check'].astype(str).str.startswith('passed').all()
    fin = C.RES_DIR / 'finalists_m3.json'
    if fin.exists():
        rows = json.loads(fin.read_text(encoding='utf-8'))
        assert 1 <= len(rows) <= 3
        import importlib
        for r in rows:
            mod = importlib.import_module(r['module'])
            assert callable(getattr(mod, r['factory'])) and callable(getattr(mod, r['build_x']))
            assert callable(getattr(mod, r['factory'])(**r['params']))
            res = cv.load_result(r['name'])
            base = cv.load_result(r['base_experiment'])
            # an outer-fold choice is recorded on the shipped result AND on its base, as in the finalists file
            assert int(res.info['n_configs_tried']) == int(base.info['n_configs_tried']) == int(r['n_configs_tried'])
