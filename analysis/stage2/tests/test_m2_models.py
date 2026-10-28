"""Lane m2 models: determinism, row-wise prediction, exact antisymmetry, calibrators, a leak-checked LOSO run,
the experiment registry and the finalists' entry points (fast: synthetic data, cheap models)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stage2 import cv
from stage2.lanes import m2_calibration as CAL
from stage2.lanes import m2_features as F
from stage2.lanes import m2_models as M


def _problem(n_persons: int = 6, per: int = 16, seed: int = 0):
    """Synthetic trials with a chiral (odd) signal: right hands turn one way, left hands the other."""
    from stage2.standardize import CHANNELS
    rng = np.random.default_rng(seed)
    n = n_persons * per
    y = np.tile(np.r_[np.zeros(per // 2), np.ones(per // 2)], n_persons).astype(int)
    A = rng.normal(size=(n, len(CHANNELS), 64)).astype(np.float32)
    k = CHANNELS.index('yaw_rate')
    A[:, k, 8:24] += ((2 * y - 1) * 1.5)[:, None]
    X_ = rng.normal(size=(n, len(CHANNELS), 180)).astype(np.float32)
    dur = rng.uniform(800, 3000, n)
    Fm, reg = F.features_from_arrays(A, X_, dur, F.DEFAULT_SPEC)
    groups = np.repeat([f'Q{i:02d}' for i in range(n_persons)], per)
    return Fm, y, groups, reg


def _fold(Fm, y, g, test_person='Q00', seed=7):
    tr, te = g != test_person, g == test_person
    return SimpleNamespace(fold=0, test_person=test_person, X_train=Fm[tr], y_train=y[tr], groups_train=g[tr],
                           X_test=Fm[te], seed=seed), te


def test_fit_predict_deterministic_and_rowwise():
    Fm, y, g, reg = _problem()
    mi, ms = F.mirror_map(reg)
    fold, te = _fold(Fm, y, g)
    M.clear_cache()
    fp = M.make_fit_predict('lr_l2', mirror='aug', mirror_index=mi, mirror_sign=ms, n_splits=3)
    p1, info = fp(fold)
    M.clear_cache()
    p2, _ = fp(fold)
    np.testing.assert_array_equal(p1, p2)                      # deterministic (refit from scratch)
    sub = np.array([5, 0, 9, 2])
    p3, _ = fp(SimpleNamespace(**{**vars(fold), 'X_test': fold.X_test[sub]}))
    np.testing.assert_allclose(p3, p1[sub], rtol=0, atol=1e-12)  # row-wise: other test rows do not matter
    assert info['params']['C'] in M.LR_GRID and info['n_train_persons'] == 5
    assert np.mean((p1 >= 0.5) == y[te]) > 0.8                  # the planted chiral signal is found


@pytest.mark.parametrize('model,mirror', [('lr_l2', 'anti'), ('lr_l2', 'both'), ('lr_l2', 'aug'), ('knn', 'both')])
def test_antisymmetry(model, mirror):
    """anti / both: p(m(x)) = 1 - p(x) exactly; aug with a linear model: antisymmetric up to the solver tolerance."""
    Fm, y, g, reg = _problem()
    mi, ms = F.mirror_map(reg)
    fold, _ = _fold(Fm, y, g)
    fp = M.make_fit_predict(model, mirror=mirror, mirror_index=mi, mirror_sign=ms, n_splits=3, n_trials=2)
    p, _ = fp(fold)
    pm, _ = fp(SimpleNamespace(**{**vars(fold), 'X_test': F.mirror_features(fold.X_test, mi, ms)}))
    tol = 1e-9 if mirror in ('anti', 'both') else 5e-3
    np.testing.assert_allclose(pm, 1 - p, rtol=0, atol=tol)


def test_calibrators():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, 400)
    y = (rng.uniform(size=400) < CAL.sigmoid(0.5 * z)).astype(int)      # over-confident scores
    p = CAL.sigmoid(z)
    strata = np.where(np.arange(z.size) % 2 == 0, 'android', 'ios')
    for name in CAL.CALIBRATORS:
        st = strata if name in CAL.STRATIFIED else None
        if st is not None:
            with pytest.raises(ValueError):
                CAL.fit_calibrator(name, p, y)                 # a stratified map needs the strata
        fn, params = CAL.fit_calibrator(name, p, y, st)
        q = fn(p, st)
        assert np.isfinite(q).all() and (q >= 0).all() and (q <= 1).all()
        if name == 'none':
            np.testing.assert_array_equal(q, p)
    fn, params = CAL.fit_calibrator('temperature', p, y)
    assert params['T'] > 1.3                                   # it softens over-confident scores
    np.testing.assert_allclose(fn(1 - p), 1 - fn(p), atol=1e-12)  # temperature keeps antisymmetry
    fn, params = CAL.fit_calibrator('beta', p, y)
    assert params['a'] >= 0 and params['b'] >= 0


def test_loso_run_passes_the_leak_check():
    """A full (synthetic) run_experiment: the fit cache must not hide anything from the harness's checks."""
    Fm, y, g, reg = _problem(n_persons=5, per=12, seed=3)
    mi, ms = F.mirror_map(reg)
    meta = pd.DataFrame({'trial_id': [f't{i}' for i in range(len(y))], 'person_id': g, 'y_right': y})
    M.clear_cache()
    fp = M.make_fit_predict('lr_l2', mirror='aug', mirror_index=mi, mirror_sign=ms, n_splits=2, calibrator='platt')
    r = cv.run_experiment('m2_test_synthetic', fp, Fm, meta, save=False, n_boot=50, cap_per_person=None)
    assert r.leak_check.startswith('passed')
    assert r.metrics['accuracy'] > 0.75


def test_experiment_registry():
    from stage2.lanes import m2_pipeline as P
    exps = P.experiments()
    assert all(n.startswith('m2_') for n in exps)
    assert len(set(exps)) == len(exps)
    for e in exps.values():
        assert not e.baseline or e.baseline in exps
        assert e.n_configs_tried >= 1
        assert e.mirror in M.MIRRORS
        assert e.model in M.MODELS or e.model in ('ens_soft', 'ens_stack', 'select')
    # n_configs_tried of the model comparison = every pipeline compared on the outer folds (default included), and
    # every one of them is declared (so a plain compute runs exactly these)
    k = sum(e.group in ('default', 'model') for e in exps.values())
    assert {e.n_configs_tried for e in exps.values() if e.group == 'model'} == {k}
    assert exps['m2_lr_full'].n_configs_tried == 1 and exps['m2_select'].n_configs_tried == 1
    for g in ('calibration', 'family_drop', 'scaling'):
        sizes = {e.n_configs_tried for e in exps.values() if e.group == g}
        assert len(sizes) == 1 and sizes.pop() > 1


def test_score_model_platt_map_is_cross_fitted(monkeypatch):
    """The Platt map that turns an SVM score into P(right) for one inner validation fold is fitted on the OTHER inner
    folds only; the final model's map on every inner score (so the inner log loss is out of sample)."""
    Fm, y, g, reg = _problem(n_persons=6, per=12, seed=5)
    mi, ms = F.mirror_map(reg)
    sizes = []
    real = M._platt_fit
    monkeypatch.setattr(M, '_platt_fit', lambda s, yy: (sizes.append(len(s)), real(s, yy))[1])
    splits = cv.inner_splits(g, 3)
    loss, p, platt, nf, _ = M._inner_eval(M.MODELS['svm_rbf'], {'C': 1.0, 'gamma_mult': 1.0}, Fm, y, splits, 3, 'aug',
                                          mi, ms, M.Prep())
    assert sizes == [len(y) - len(va) for _, va in splits] + [len(y)]
    assert nf == len(splits) and np.isfinite(p).all() and platt is not None and np.isfinite(loss)


def test_fold_info_carries_nested_attempt_temperatures():
    Fm, y, g, reg = _problem(n_persons=6, per=16, seed=2)
    mi, ms = F.mirror_map(reg)
    fold, _ = _fold(Fm, y, g)
    M.clear_cache()
    _, info = M.make_fit_predict('lr_l2', mirror='aug', mirror_index=mi, mirror_sign=ms, n_splits=3)(fold)
    tau = info['attempt_tau']
    assert set(tau) == {str(k) for k in CAL.ATTEMPT_KS}
    assert all(np.isfinite(v) and v > 0 for v in tau.values())
    assert info['n_inner_fits'] == len(M.LR_GRID) * 3 and info['final_converged'] is True


def test_platform_specific_platt_map():
    rng = np.random.default_rng(3)
    n = 400
    strata = np.where(np.arange(n) < n // 2, 'android', 'ios')
    z = rng.normal(0, 2, n)
    slope = np.where(strata == 'android', 1.0, 0.4)                  # iOS scores over-confident
    y = (rng.uniform(size=n) < CAL.sigmoid(slope * z)).astype(int)
    fn, params = CAL.fit_calibrator('platt_platform', CAL.sigmoid(z), y, strata)
    assert params['a_ios'] < params['a_android']
    q = fn(CAL.sigmoid(z), strata)
    np.testing.assert_allclose(q[strata == 'ios'], CAL.sigmoid(params['a_ios'] * z[strata == 'ios'] + params['b_ios']),
                               rtol=1e-6)
    np.testing.assert_allclose(fn(CAL.sigmoid(z)), CAL.sigmoid(params['a'] * z + params['b']), rtol=1e-6)


def test_stats_helpers():
    from stage2.lanes import m2_stats as S
    np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan])
    d = np.random.default_rng(4).normal(0.1, 1, 14)
    assert S.sign_flip_p(d) == cv._sign_flip_p(d)
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.98, 1500)
    y = (rng.uniform(size=p.size) < p).astype(float)
    e = S.ece_null(y, p, n_sim=300)
    assert e['ece_null_p'] > 0.01 and e['ece_null_lo'] <= e['ece_null_mean'] <= e['ece_null_hi']
    assert abs(S.recalibration(y, p)['cal_slope'] - 1) < 0.25
    z = np.log(p / (1 - p))
    y2 = (rng.uniform(size=p.size) < 1 / (1 + np.exp(-2 * z))).astype(float)   # predictions under-confident
    assert S.recalibration(y2, p)['cal_slope'] > 1.5
    nu = S.natural_use(np.array([0.9, 0.5, 0.2, 0.7]), np.ones(4), np.array(['a', 'b', 'c', 'd']))
    assert nu['n_abstain'] == 1 and nu['forced_choice_accuracy'] == 0.75 and nu['half_credit_accuracy'] == 0.625
    assert nu['answered_accuracy'] == 2 / 3
    eq = S.equivalent(np.array([0.2, 0.7]), np.array([0.2004, 0.6999]))
    assert eq['identical'] and eq['n_class_changes'] == 0


def test_finalist_entry_points():
    from stage2.lanes import m2_finalists as FIN
    assert callable(FIN.factory(model='lr_l2')) and callable(FIN.factory_select())


def test_fast_paired_compare_equals_the_core():
    """m2_derive.fast_paired_compare (vectorised, used for every derived comparison) reproduces cv.paired_compare."""
    from stage2 import config as C
    from stage2.lanes import m2_derive as D
    rng = np.random.default_rng(1)
    d = rng.normal(size=12)
    assert D.sign_flip_p(d) == cv._sign_flip_p(d)
    if not ((C.RES_DIR / 'm2_lr_full.json').exists() and (C.RES_DIR / 'm2_lr_mirror_none.json').exists()):
        pytest.skip('lane m2 results missing')
    a, b = cv.load_result('m2_lr_mirror_none'), cv.load_result('m2_lr_full')
    f = D.fast_paired_compare(a, b, n_boot=200)
    c = cv.paired_compare(a, b, n_boot=200)
    for k in ('pooled_acc_diff', 'mean_person_acc_diff', 'p_sign_flip_acc', 'mean_person_logloss_diff',
              'p_sign_flip_logloss', 'persons_a_better', 'persons_b_better'):
        assert f[k] == pytest.approx(c[k], abs=1e-12), k
    for k in ('pooled_acc_diff_ci95', 'mean_person_acc_diff_ci95', 'mean_person_logloss_diff_ci95'):
        assert f[k] == pytest.approx(c[k], abs=1e-12), k
