"""Lane m1: feature parity and grip labels, antisymmetry by construction, calibrators, rules, nested option choice,
training-row filters, multiplicity and calibration statistics, determinism, label-free inputs, the experiment
registry, finalists, figures redrawn from saved results, and the numbers behind the summary
(run: python -m pytest stage2/tests/test_m1.py)."""
from __future__ import annotations

import dataclasses
import importlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2 import cv
from stage2.lanes import m1_analysis as A
from stage2.lanes import m1_experiments as E
from stage2.lanes import m1_features as MF
from stage2.lanes import m1_models as M
from stage2.standardize import CHANNELS, mirror_channels, screen_flip_channels


# ───────────────────────────── a synthetic fold ─────────────────────────────


def _fold(n_persons: int = 8, per: int = 20, seed: int = 0, cols=('a_odd', 'b_odd')) -> cv.Fold:
    """A Fold with chiral columns named like catalogue features (odd parity) and a held-out person."""
    rng = np.random.default_rng(seed)
    names = ['yaw_q50', 'latvel_q75'][:len(cols)]
    rows, y, g, blk = [], [], [], []
    for p in range(n_persons + 1):
        for i in range(per):
            h = int(rng.integers(0, 2))
            s = 1 if h else -1
            rows.append([s * 5 + rng.normal(0, 6), s * 0.2 + rng.normal(0, 0.3)][:len(cols)])
            y.append(h)
            g.append(f'P{p:02d}')
            blk.append(i // 5)
    X = pd.DataFrame(rows, columns=names)
    y, g, blk = np.array(y), np.array(g), np.array(blk)
    te = g == f'P{n_persons:02d}'
    meta_tr = pd.DataFrame({'person_id': g[~te], 'session_label': [f'{p}-s1' for p in g[~te]], 'block_index': blk[~te]})
    return cv.Fold(0, f'P{n_persons:02d}', X[~te].reset_index(drop=True), y[~te], g[~te], meta_tr,
                   X[te].reset_index(drop=True), pd.DataFrame({'row': np.arange(te.sum())}), 123)


def _mirror_fold(f: cv.Fold) -> cv.Fold:
    return dataclasses.replace(f, X_test=MF.mirror_features(f.X_test))


# ───────────────────────────── features ─────────────────────────────


def test_catalogue_is_documented_and_label_free():
    names = [f.name for f in MF.CATALOGUE]
    assert len(names) == len(set(names))
    for f in MF.CATALOGUE:
        assert f.parity in ('odd', 'even') and f.unit and f.formula and f.meaning and f.grip and f.frame
        assert not cv.forbidden_name(f.name), f.name
    assert set(MF.ODD) | set(MF.EVEN) == set(names)
    assert all(MF.BY_NAME[c].parity == 'odd' for c in MF.CHIRAL_APPROACH)
    assert not any(MF.BY_NAME[c].family == 'dedrift' for c in MF.CHIRAL_APPROACH)   # a treatment, not a candidate


def test_features_have_exact_mirror_parity(primary):
    d = primary.subset(np.arange(0, primary.n, 9))
    F = MF.build_x(d)
    assert list(F.columns) == MF.NAMES and len(F) == d.n
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    Fm = MF.compute_features(mirror_channels(d.approach), mirror_channels(d.fixed), d.fixed_mask, dur, d.channels)
    np.testing.assert_allclose(Fm.to_numpy(), MF.mirror_features(F).to_numpy(), atol=1e-9)
    T = MF.build_tc(d)
    Tm = MF.compute_timecourse(mirror_channels(d.approach), d.channels)
    np.testing.assert_allclose(Tm.to_numpy(), -T.to_numpy(), atol=1e-9)    # every time-course column is odd


def test_grip_labels_match_the_screen_flip(primary):
    """The catalogue's grip text is generated from ``flips``: check it against the screen-away flip on real trials."""
    d = primary.subset(np.arange(0, primary.n, 11))
    F = MF.build_x(d)
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    Ff = MF.compute_features(screen_flip_channels(d.approach), screen_flip_channels(d.fixed), d.fixed_mask, dur,
                             d.channels)
    for f in MF.CATALOGUE:
        a, b = F[f.name].to_numpy(), Ff[f.name].to_numpy()
        np.testing.assert_allclose(b, -a if f.flips else a, atol=1e-9 * (1 + np.abs(a).max()), err_msg=f.name)
        assert ('changes sign' in f.grip) == f.flips


def test_body_frame_is_invariant_to_the_three_grips():
    """A rotation of the device about its screen normal (upright <-> sideways grips) leaves the body channels unchanged."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, len(CHANNELS), 64))
    u = rng.normal(size=(5, 3, 64))
    u[:, 2] *= 0.3
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    ix = {c: i for i, c in enumerate(CHANNELS)}
    X[:, [ix['up_x'], ix['up_y'], ix['up_z']]] = u
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # 90 degrees about z
    Xr = X.copy()
    for trip in (('acc_x', 'acc_y', 'acc_z'), ('gyro_x', 'gyro_y', 'gyro_z'), ('up_x', 'up_y', 'up_z')):
        k = [ix[c] for c in trip]
        Xr[:, k] = np.einsum('ij,njt->nit', R, X[:, k])
    a, b = MF.body_channels(X), MF.body_channels(Xr)
    for c in a:
        np.testing.assert_allclose(a[c], b[c], atol=1e-9)


def test_features_do_not_read_labels(primary):
    d = primary.subset(np.arange(0, primary.n, 13))
    F1 = M.build_x_screen(d)
    d.trials['hand'] = 'left'
    d.trials['y_right'] = 0
    pd.testing.assert_frame_equal(F1, M.build_x_screen(d))


# ───────────────────────────── models ─────────────────────────────


def test_antisymmetric_logistic_is_mirror_consistent():
    f = _fold()
    fp = M.make_linear(['yaw_q50', 'latvel_q75'], antisym=True)
    p, info = fp(f)
    pm, _ = fp(_mirror_fold(f))
    np.testing.assert_allclose(pm, 1 - p, atol=1e-12)
    assert 'C' in info and info['C'] in M.CS_DEFAULT


def test_mirror_augmentation_equals_the_antisymmetric_model():
    """On odd features, augmentation + intercept is the antisymmetric model: the intercept vanishes, the z-score of
    the augmented rows is the RMS scaling, and the doubled rows halve the L2 penalty (C -> 2C)."""
    f = _fold(seed=4)
    pa, _ = M.make_linear(['yaw_q50', 'latvel_q75'], antisym=True, Cs=(2.0,))(f)
    pb, info = M.make_linear(['yaw_q50', 'latvel_q75'], antisym=False, augment=True, scale='zscore', Cs=(1.0,))(f)
    assert abs(info['intercept']) < 1e-4
    np.testing.assert_allclose(pa, pb, atol=1e-4)


def test_symmetrised_models_keep_the_mirror_exactly():
    f = _fold(seed=2)
    for fp in (M.make_tree(['yaw_q50', 'latvel_q75'], depths=(2,), leafs=(10,)),
               M.make_gam(['yaw_q50', 'latvel_q75'], knots=(5,), Cs=(1.0,)),
               M.make_linear(['yaw_q50', 'latvel_q75'], calibration='temperature'),
               M.make_linear_select({'a': {'cols': ['yaw_q50']}, 'b': {'cols': ['yaw_q50', 'latvel_q75'],
                                                                       'winsor': 'asinh'}})):
        p, _ = fp(f)
        pm, _ = fp(_mirror_fold(f))
        np.testing.assert_allclose(pm, 1 - p, atol=1e-12)


@pytest.mark.parametrize('kind', M.CALIBRATIONS)
def test_calibrators_return_probabilities(kind):
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, 400)
    y = (rng.random(400) < M.sigmoid(0.5 * z)).astype(int)
    apply, params = M.fit_calibrator(kind, z, y)
    p = apply(np.linspace(-8, 8, 50))
    assert np.all((p >= 0) & (p <= 1)) and np.all(np.diff(p) >= -1e-12)        # monotone in the logit
    if kind == 'temperature':
        assert params['T'] > 1                                                 # over-confident logits get T > 1


def test_threshold_rule_and_sign_rule():
    v = np.array([-3.0, -2.0, -1.0, 0.5, 1.0, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    th, acc = M._best_threshold(v, y)
    assert acc == 1.0 and -1.0 < th < 0.5
    f = _fold(seed=3)
    p, info = M.make_sign_rule('yaw_q50', direction=1.0)(f)
    assert set(np.unique(p)) <= {0.0, 1.0} and info['direction'] == 1.0
    p, info = M.make_sign_rule('yaw_q50', direction=1.0, calibrated=True)(f)
    assert set(np.round(np.unique(p), 9)) <= {round(info['q'], 9), round(1 - info['q'], 9)}
    p2, info2 = M.make_sign_rule('yaw_q50', direction=None, calibrated=True)(f)     # direction learned in-fold
    assert info2['direction'] == 1.0
    np.testing.assert_array_equal(p, p2)


def test_linear_select_chooses_inside_the_fold():
    """The informative option wins against a pure-noise one, and the choice is reported."""
    f = _fold(seed=6)
    rng = np.random.default_rng(9)
    f = dataclasses.replace(f, X_train=f.X_train.assign(roll_q50=rng.normal(size=len(f.X_train))),
                            X_test=f.X_test.assign(roll_q50=rng.normal(size=len(f.X_test))))
    store: dict = {}
    p, info = M.make_linear_select({'noise': {'cols': ['roll_q50']}, 'signal': {'cols': ['yaw_q50', 'latvel_q75']}},
                                   store=store)(f)
    assert info['option'] == 'signal' and info['C'] in M.CS_DEFAULT
    e = store[0]
    Z = M.stored_transform(e, f.X_test[e['names']].to_numpy())
    np.testing.assert_allclose(M.sigmoid(Z @ np.asarray(e['coef_std'])), p, atol=1e-12)   # stored model = fitted one


def test_train_filter_drops_training_rows_only():
    f = _fold(seed=7)
    fp = M.make_train_filter('make_linear', {'cols': ['yaw_q50', 'latvel_q75']}, drop='blocks',
                             blocks=[['P00-s1', 0], ['P01-s1', 2]])
    p, info = fp(f)
    assert info['n_train_dropped'] == 10 and len(p) == len(f.X_test)
    base, _ = M.make_linear(['yaw_q50', 'latvel_q75'])(f)
    none, info0 = M.make_train_filter('make_linear', {'cols': ['yaw_q50', 'latvel_q75']}, drop='blocks', blocks=[])(f)
    assert info0['n_train_dropped'] == 0
    np.testing.assert_allclose(none, base, atol=1e-12)


def test_fit_predict_is_deterministic():
    f = _fold(seed=5)
    for fp in (M.make_linear(['yaw_q50', 'latvel_q75'], fit_agg=True), M.make_gam(['yaw_q50', 'latvel_q75'], knots=(3, 5)),
               M.make_nested_rule(['yaw_q50', 'latvel_q75']),
               M.make_linear_select({'a': {'cols': ['yaw_q50']}, 'b': {'cols': ['latvel_q75']}}, fit_agg=True)):
        a, ia = fp(f)
        b, ib = fp(f)
        np.testing.assert_array_equal(a, b)
        assert json.dumps(ia, sort_keys=True, default=str) == json.dumps(ib, sort_keys=True, default=str)


def test_aggregation_exponent_in_range():
    rng = np.random.default_rng(0)
    g = np.repeat([f'P{i}' for i in range(10)], 12)
    y = np.tile(np.repeat([0, 1], 6), 10)
    z = (2 * y - 1) * 1.5 + np.repeat(rng.normal(0, 1.5, 20), 6)            # person-level shared error: correlated
    a = M.fit_agg_alpha(z, y, g, seed=1)
    assert 0.0 <= a <= 1.0


def test_catalogue_rule_avoids_the_loso_intercept_artefact():
    """A pure-noise feature: a univariate logistic WITH an intercept ranks the held-out persons by their fold's
    intercept, which follows the TRAINING class balance and so anticorrelates with the held-out person's own balance:
    the pooled leave-one-person-out AUC falls below 0.5. The direction-only rule of the catalogue stays near 0.5."""
    rng = np.random.default_rng(7)
    rows = []
    for p in range(20):
        for h, n in ((0, int(rng.integers(4, 26))), (1, int(rng.integers(4, 26)))):
            rows += [{'trial_id': f'S{p:02d}-{h}-{k}', 'person_id': f'P{p:02d}', 'y_right': h} for k in range(n)]
    meta = pd.DataFrame(rows)
    X = pd.DataFrame({'yaw_q50': rng.normal(size=len(meta))})
    kw = dict(save=False, leak_check='off', n_boot=0, cap_per_person=None)
    a = cv.run_experiment('t_logit', M.make_linear(['yaw_q50'], antisym=False, scale='zscore', Cs=(1e3,)), X, meta, **kw)
    b = cv.run_experiment('t_dir', M.make_direction('yaw_q50'), X, meta, **kw)
    assert a.metrics['roc_auc'] < 0.45
    assert abs(b.metrics['roc_auc'] - 0.5) < 0.08


# ───────────────────────────── statistics ─────────────────────────────


def test_holm_and_bh():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    h = A.holm(p)
    np.testing.assert_allclose(h[[0, 2, 1, 4]], [0.04, 0.09, 0.09, 0.5])
    assert np.isnan(h[3])
    q = A.bh(p)
    np.testing.assert_allclose(q[[0, 1, 2, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert np.all(h[np.isfinite(h)] >= p[np.isfinite(p)])


def test_ece_null_and_calibration_slope():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.98, 1500)
    y = (rng.random(p.size) < p).astype(int)               # perfectly calibrated
    t = A.ece_test(y, p, n_sim=500)
    assert t['ece_p'] > 0.01 and t['ece'] <= t['ece_null_q95'] * 1.5
    a, b = A.cox(y, p)
    assert abs(b - 1) < 0.25 and abs(a) < 0.2
    z = M.logit(p)
    p_over = M.sigmoid(2.5 * z)                             # over-confident: slope < 1
    t2 = A.ece_test(y, p_over, n_sim=500)
    assert t2['ece_p'] < 0.01
    assert A.cox(y, p_over)[1] < 0.6


def test_calibration_verdicts_come_from_the_saved_statistics():
    """The text's verdict on calibration: the ECE null test, or more bins outside their 95 % band than the bands'
    nominal 5 % miss rate explains; the kind of error from the bins' directions and the slope CI."""
    from stage2.lanes import m1_texts as T
    q = dict(ece_p=0.3, bins_outside=2, n_bins=10, bins_under=2, bins_over=0, cox_slope_lo=0.8, cox_slope_hi=1.3)
    assert abs(T.bins_outside_p(pd.Series(q)) - 0.0861) < 1e-3          # two misses in ten bands: not evidence
    assert not T.miscalibrated(pd.Series(q)) and T.miscal_shape(pd.Series(q)) == 'none'
    s = pd.Series(dict(q, ece_p=0.001, bins_outside=5, bins_under=4, bins_over=1))
    assert T.miscalibrated(s) and T.miscal_shape(s) == 'S'                # both sides, slope compatible with 1
    assert T.miscal_shape(pd.Series(dict(s, cox_slope_lo=0.5, cox_slope_hi=0.9))) == 'over'
    assert T.miscal_shape(pd.Series(dict(s, cox_slope_lo=1.1, cox_slope_hi=1.6))) == 'under'
    assert T.miscal_shape(pd.Series(dict(s, bins_under=5, bins_over=0))) == 'under_bins'
    assert set(T.SHAPE_TEXT) == {'none', 'S', 'over', 'under', 'under_bins', 'over_bins', 'ece_only'}


def test_poisson_binomial_pmf():
    p = [0.2, 0.9, 0.5, 0.7]
    pmf = A.poisson_binomial_pmf(p)
    assert abs(pmf.sum() - 1) < 1e-12 and abs((np.arange(pmf.size) * pmf).sum() - sum(p)) < 1e-12


# ───────────────────────────── registry, finalists, figures, numbers ─────────────────────────────


def test_experiment_registry():
    names = [s.name for s in E.SPECS]
    assert len(names) == len(set(names)) and all(n.startswith('m1_') for n in names)
    for s in E.SPECS:
        assert callable(getattr(M, s.factory)) and callable(getattr(M, s.x))
        json.dumps(E.spec_params_json(s))
        assert s.design in E.DESIGN_LABELS
        if s.baseline:
            assert s.baseline in E.BY_NAME
        if s.design == 'family':
            assert s.n_configs == next(E.FAMILY_SIZE[f] for f, v in E.FAMILIES.items() if s.name in v) > 1
        else:
            assert s.n_configs == 1
    assert E.BY_NAME[E.BASELINE].design == 'post-pooled'                     # PHYS6 is not called pre-specified
    assert set(E.ANCHORS.values()) <= set(names)
    assert E.BY_NAME[E.ANCHORS['small']].design == 'nested'


def test_finalists_rebuild(primary):
    p = C.RES_DIR / 'finalists_m1.json'
    if not p.exists():
        pytest.skip('no finalists yet (python -m stage2.pipeline compute --lanes m1)')
    fins = json.loads(p.read_text(encoding='utf-8'))
    assert 1 <= len(fins) <= 3
    d = primary.subset(np.arange(0, primary.n, 17))
    for f in fins:
        mod = importlib.import_module(f['module'])
        X = getattr(mod, f['build_x'])(d)
        fp = getattr(mod, f['factory'])(**f['params'])
        assert callable(fp) and len(X) == d.n
        assert (C.RES_DIR / f'{f["name"]}.json').exists()
        assert (f['n_configs_tried'] == 1) == (E.BY_NAME[f['name']].design != 'family')     # true n_configs_tried


def test_figures_redraw_from_saved_results(tmp_path, monkeypatch):
    from stage2 import pipeline as P
    from stage2 import plotting
    from stage2.lanes import m1_figures
    if not (C.TAB_DIR / 'm1' / 'm1_natural.json').exists():
        pytest.skip('no saved m1 results (python -m stage2.pipeline compute --lanes m1)')
    saved = []

    def fake_save(fig, name, *, caption, decision='', fig_dir=None, png_dpi=200):
        import matplotlib.pyplot as plt
        assert caption and decision and name.startswith('M1_')
        fig.savefig(tmp_path / f'{name}.png', dpi=50)
        plt.close(fig)
        saved.append(name)
        return {}
    monkeypatch.setattr(plotting, 'save_fig', fake_save)
    monkeypatch.setattr(P.Lane, 'figure_data', lambda self, df, name: None)
    import stage2.lanes.m1_models as models_module
    monkeypatch.setattr(models_module, 'make_linear', None)          # a figure must never fit a model
    m1_figures.draw(['M1_05_model_ladder', 'M1_04_channel_sets', 'M1_15_finalist_calibration', 'M1_16_natural_transfer'])
    assert saved == ['M1_04_channel_sets', 'M1_05_model_ladder', 'M1_15_finalist_calibration', 'M1_16_natural_transfer']


def test_wilson_interval():
    from stage2.lanes.m1_tables import wilson
    lo, hi = wilson(0.85, 20)                      # 17 of 20: the textbook Wilson interval
    assert abs(lo - 0.6396) < 1e-3 and abs(hi - 0.9476) < 1e-3
    lo, hi = wilson(1.0, 20)
    assert abs(hi - 1.0) < 1e-9 and 0.8 < lo < 0.85
    assert all(np.isnan(v) for v in wilson(None, 20))


def test_book_registers_numbers_and_refuses_conflicts():
    from stage2.lanes.m1_texts import Book, render
    b = Book()
    assert b.t('x_acc', 0.8328, '.1%', 's', 'd') == '83.3 %'
    assert b.ci('x_acc', 0.79, 0.876, '.1%', 's', 'd') == '[79.0, 87.6]'
    assert set(b.entries) == {'x_acc', 'x_acc_lo', 'x_acc_hi'}
    assert render(0.034, '.2g') == '0.034' and render(4.0, '+.1f') == '+4.0'
    b.t('x_acc', 0.8328, '.1%', 's', 'd')                                   # same value again: fine
    with pytest.raises(ValueError):
        b.t('x_acc', 0.9, '.1%', 's', 'd')


def _numeric_tokens(text: str) -> list[str]:
    """Numeric tokens of the summary that should be results: code-defined names (experiment labels and notes,
    feature meanings and formulas, identifiers in backticks, person / figure / decision ids) are removed first."""
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)
    text = re.sub(r'`[^`]*`', ' ', text)                                    # identifiers, file names, commands
    names = [s.label for s in E.SPECS] + [s.notes for s in E.SPECS] + [f.meaning for f in MF.CATALOGUE]         + [f.formula for f in MF.CATALOGUE] + ['reference: yaw logreg (8 instants)']
    for nm in sorted(set(n for n in names if n), key=len, reverse=True):
        text = text.replace(nm, ' ')
    text = re.sub(r'docs/\d+', ' ', text)
    text = re.sub(r'\b(?:P|PD)\d\d\b|\bM1_\d\d\b|\bD\d+\b|docs/09( section)? \d+(\.\d+)?|\bk = \d\b|\bk\^\(-alpha\)', ' ', text)
    text = re.sub(r'\b[a-z]+[a-z0-9]*_[a-z0-9_]*\b', ' ', text)            # snake_case names outside backticks
    return [m.group(0).lstrip('+') for m in re.finditer(r'(?<![\w.])[-+]?\d+(?:\.\d+)?', text)]


def test_summary_numbers_are_in_the_numbers_file():
    """Every number of m1_SUMMARY.md renders from an entry of results/m1_numbers.json (the summary registers each
    number while it is rendered; this guards against typed numbers)."""
    from stage2.lanes.m1_texts import render
    sp, np_ = C.OUT_DIR / 'm1_SUMMARY.md', C.RES_DIR / 'm1_numbers.json'
    if not (sp.exists() and np_.exists()):
        pytest.skip('no summary yet (python -m stage2.pipeline numbers --lanes m1)')
    nums = json.loads(np_.read_text(encoding='utf-8'))
    ok = set()
    for e in nums.values():
        v, fmt = e['value'], e['format']
        if v is None or isinstance(v, str):
            continue
        for bare in (True, False):
            s = render(v, fmt, bare)
            # signed and unsigned forms: '6.4e-05' gives '6.4' and '-05', and the summary tokenizer sees '05'
            ok.update(x for t in re.findall(r'[-+]?\d+(?:\.\d+)?', s) for x in (t.lstrip('+'), t.lstrip('+-')))
    missing = [t for t in _numeric_tokens(sp.read_text(encoding='utf-8')) if t not in ok and t.lstrip('-') not in ok]
    assert not missing, f'numbers in the summary without a numbers entry: {sorted(set(missing))[:20]}'
