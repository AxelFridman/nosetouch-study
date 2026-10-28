"""Lane final: fast checks of the synthesis machinery (each well under a minute)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2 import cv


@pytest.fixture(scope='module')
def d():
    from stage2.data import load_cache
    if not (C.CACHE_DIR / 'arrays.npz').exists():
        pytest.skip('no cache')
    return load_cache()


def test_m2_feature_families_on_the_cache_channels_equal_lane_m2(d):
    from stage2.lanes import final_features as FF
    from stage2.lanes import m2_finalists
    sub = d.subset(np.arange(len(d.trials)) < 25)
    a = FF.build_m2_m2(sub)
    b = m2_finalists.build_x(sub)
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_body_frame_feature_tables_have_an_exact_mirror_map(d):
    from stage2.lanes import final_features as FF
    from stage2.lanes.m3_body import EXT_SIGN, EXT_CHANNELS
    sub = d.subset(np.arange(len(d.trials)) < 12)
    A = FF.ext_arrays(sub.approach)
    sign = np.array([EXT_SIGN[c] for c in EXT_CHANNELS], dtype=np.float32).reshape(1, -1, 1)
    dur = sub.trials['duration_ms'].to_numpy(float)
    for table in ('body12', 'all21'):
        F, reg = FF.features_ext(A, dur, table)
        Fm, _ = FF.features_ext(A * sign, dur, table)
        mi, ms = FF.mirror_map(table)
        assert np.allclose(F[:, mi] * ms, Fm, atol=1e-9)


def test_every_body_channel_changes_sign_under_the_screen_flip(d):
    from stage2.lanes.m3_body import body_channels
    from stage2.standardize import screen_flip_channels
    A = np.asarray(d.approach[:30], dtype=np.float32)
    assert np.allclose(body_channels(screen_flip_channels(A)), -body_channels(A), atol=1e-4)


def test_harness_folds_equal_the_folds_of_run_experiment(d):
    from stage2.lanes.final_nested import harness_folds
    seen = {}

    def fp(fold):
        seen[fold.fold] = (fold.meta_train['trial_id'].astype(str).tolist(), fold.X_test[:, 0].copy())
        return np.full(fold.n_test, 0.5)
    X = np.arange(len(d.trials), dtype=float).reshape(-1, 1)
    cv.run_experiment('final_test_folds', fp, X, d.trials, leak_check='off', n_boot=0, save=False)
    for f in harness_folds(X, d.trials):
        ids, xt = seen[f.fold]
        assert f.meta_train['trial_id'].astype(str).tolist() == ids
        assert np.array_equal(f.X_test[:, 0], xt)


def test_inner_folds_carry_no_label_and_cover_the_training_rows(d):
    from stage2.lanes.final_nested import harness_folds, inner_fold
    X = np.arange(len(d.trials), dtype=float).reshape(-1, 1)
    f = harness_folds(X, d.trials)[3]
    covered = []
    for k, (itr, iva) in enumerate(f.inner_splits()):
        g, order = inner_fold(f, itr, iva, k)
        assert set(g.meta_test.columns) <= set(cv.TEST_META_COLUMNS)
        assert not set(g.groups_train.tolist()) & set(f.groups_train[iva].tolist())
        assert np.array_equal(np.sort(order), np.sort(iva))
        assert np.array_equal(g.X_test[:, 0], f.X_train[order, 0])
        covered.extend(order.tolist())
    assert sorted(covered) == list(range(f.y_train.size))


def test_body_frame_logistic_is_exactly_antisymmetric(d):
    from stage2.lanes import final_models as FM
    from stage2.lanes.final_nested import harness_folds
    from stage2.lanes.m3_body import EXT_SIGN, EXT_CHANNELS, x_approach_body
    X = x_approach_body(d)
    f = harness_folds(X, d.trials)[0]
    p, _ = FM.make_lr_body_tc()(f)
    sign = np.array([EXT_SIGN[c] for c in EXT_CHANNELS], dtype=np.float32).reshape(1, -1, 1)
    import dataclasses
    fm = dataclasses.replace(f, X_test={'x': f.X_test['x'] * sign})
    pm, _ = FM.make_lr_body_tc()(fm)
    assert np.allclose(pm, 1 - p, atol=1e-12)


def test_rest_pose_check_and_unflip():
    from stage2.lanes.final_robust import detect_and_correct
    p_rec = np.array([0.9, 0.2, 0.7])
    p_flip = np.array([0.1, 0.8, 0.3])
    away = np.array([True, False, False])
    assert np.allclose(detect_and_correct(p_flip, p_rec, away), [0.9, 0.8, 0.3])
    assert np.allclose(detect_and_correct(p_flip, p_rec, away, mode='abstain'), [0.5, 0.8, 0.3])


def test_holm_and_poisson_binomial():
    from stage2.lanes.final_derive import holm, poisson_binomial
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    q = np.array([0.2, 0.5, 0.9])
    dist = poisson_binomial(q)
    assert np.isclose(dist.sum(), 1.0)
    assert np.isclose(dist @ np.arange(4), q.sum())


def test_calibrators_are_monotone_and_temperature_keeps_the_mirror():
    from stage2.lanes.final_calibration import fit_calibrator
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 400)
    y = (rng.random(400) < 1 / (1 + np.exp(-0.6 * z))).astype(int)
    grid = np.linspace(-6, 6, 50)
    for kind in ('platt', 'temperature', 'isotonic', 'beta'):
        f, par = fit_calibrator(kind, z, y)
        p = f(grid)
        assert np.all(np.diff(p) >= -1e-12)
        json.dumps(par)
    f, par = fit_calibrator('temperature', z, y)
    assert np.allclose(f(grid) + f(-grid), 1.0)
    assert par['T'] > 1.0                 # the labels were drawn at 0.6 z: the logits are over-confident


def test_final_outputs_follow_the_contract():
    """When the lane has run: every number in the summary is an entry of the numbers file, the figures are registered
    with captions, and the finalists file names this lane's results."""
    from stage2.pipeline import Lane
    L = Lane('final')
    if not L.numbers_path.exists() or not L.summary_path.exists():
        pytest.skip('lane final has not produced its numbers yet')
    from stage2 import report_export as RX
    nums = json.loads(L.numbers_path.read_text(encoding='utf-8'))
    allowed = set()
    for e in nums.values():
        s = RX.format_value(e['value'], e['format'])
        allowed.update(re.findall(r'\d+(?:\.\d+)?', s))
    # constants of the method that the text names (CI level, thresholds, window sizes, section numbers): not results
    structural = {'95', '0.5', '1', '2', '3', '4', '5', '0', '100', '8', '64', '17', '12', '21', '4b', '2b', '9', '6',
                  '7', '10', '11', '13', '14', '15', '16', '20', '2000', '0.05', '0.001', '0.2', '1.2', '0.69', '50',
                  '200', '40', '0.01', '0.02'}

    def missing_numbers(text: str) -> list[str]:
        text = re.sub(r'`[^`]*`', ' ', text)                      # code spans: file names, commands, keys
        text = re.sub(r'\b[A-Z][A-Za-z]*\d+[A-Za-z\d_]*\b', ' ', text)   # identifiers such as P01, FINAL_01, D2
        text = re.sub(r'\b[a-z]+\d+[a-z\d_]*\b', ' ', text)        # m1, p1, x1, k3 ...
        tokens = set(re.findall(r'(?<![\w.])\d+(?:\.\d+)?', text))
        return sorted(tokens - allowed - structural)
    missing = missing_numbers(L.summary_path.read_text(encoding='utf-8'))
    assert not missing, f'numbers in the summary that are not entries of the numbers file: {missing[:20]}'
    caps = json.loads((L.fig_dir / 'captions.json').read_text(encoding='utf-8'))
    assert caps and all(v['caption'] for v in caps.values())
    for name, v in caps.items():                                  # the figure captions cite numbers-file entries too
        miss = missing_numbers(v['caption'] + ' ' + v.get('decision', ''))
        assert not miss, f'{name}: caption numbers that are not entries of the numbers file: {miss[:10]}'
    fin = json.loads(L.finalists_path.read_text(encoding='utf-8'))
    assert len(fin) == 1 and fin[0]['name'].startswith('final_')


def test_json_null_round_trip():
    from stage2.lanes.final_io import clean, denull
    doc = clean({'a': float('nan'), 'b': [1.0, float('inf'), np.float32(2.5)], 'c': {'d': np.int64(3)}, 'e': 'x',
                 'f': np.bool_(True)})
    assert doc == {'a': None, 'b': [1.0, None, 2.5], 'c': {'d': 3}, 'e': 'x', 'f': True}
    json.dumps(doc, allow_nan=False)
    back = denull(doc)
    assert np.isnan(back['a']) and np.isnan(back['b'][1]) and back['b'][0] == 1.0 and back['e'] == 'x'


def test_component_occlusion_is_exact_for_the_linear_body_frame_model(d):
    """For the 32-feature logistic the three odd components add up to its whole logit (exact decomposition)."""
    from stage2.data import load_cache
    from stage2.lanes import final_candidates as FC
    from stage2.lanes import final_derive as FD
    nat = load_cache(('natural_use',))
    cands = FC.candidates()
    if 'lr_body_tc' not in cands:
        pytest.skip('no body-frame logistic candidate')
    out = FD.component_transfer('lr_body_tc', cands, d, nat)
    rows = {r['component']: r for r in out['rows']}
    for col in ('natural', 'primary_same_persons'):
        parts = sum(rows[c][col] for c in FD.COMPONENTS)
        assert abs(parts - rows['total'][col]) < 1e-9
