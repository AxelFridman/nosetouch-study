"""Fast tests of lane p1 (preprocessing panel): synthetic folds, no raw data, under a minute."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2.cv import Fold
from stage2.lanes import p1_calibration as K
from stage2.lanes import p1_panel as PN
from stage2.lanes.p1_minirocket import MiniRocket
from stage2.standardize import CHANNELS, mirror_channels

N_CH = len(CHANNELS)


def _synthetic(n_per_person: int = 12, persons: int = 6, T: int = 64, seed: int = 0):
    """Trials whose chiral channels carry the hand (a signed bump in yaw_rate / yaw_cum), even channels noise."""
    rng = np.random.default_rng(seed)
    n = n_per_person * persons
    y = np.tile([0, 1], n // 2)
    X = rng.normal(0, 1, (n, N_CH, T)).astype(np.float32)
    bump = np.exp(-0.5 * ((np.arange(T) - T / 4) / (T / 10)) ** 2)
    s = (2 * y - 1)[:, None] * bump[None, :] * 3.0
    X[:, CHANNELS.index('yaw_rate')] += s
    X[:, CHANNELS.index('yaw_cum')] += np.cumsum(s, axis=1) / 10
    groups = np.repeat([f'P{i + 1:02d}' for i in range(persons)], n_per_person)
    meta = pd.DataFrame({'person_id': groups, 'platform': np.where(np.arange(n) % 2 == 0, 'android', 'ios'),
                         'grip_mismatch': rng.random(n) < 0.1, 'false_start': False, 'seg_disagree_100ms': False,
                         'repeated_samples': False, 'clip_candidate': False, 'duration_ms': 1500.0,
                         'peak_gyro_dps': 100.0, 'analysis_set': 'primary', 'session_label': 's', 'block_index': 1})
    meta['platform'] = np.where(np.isin(groups, ['P01', 'P02', 'P03']), 'android', 'ios')
    return X, y, groups, meta


def _fold(test_person: str = 'P06', **kw):
    X, y, g, meta = _synthetic(**kw)
    tr, te = g != test_person, g == test_person
    mt = meta[te][['person_id', 'platform']].reset_index(drop=True)
    mt.insert(0, 'row', np.arange(int(te.sum())))
    return Fold(0, test_person, X[tr], y[tr], g[tr], meta[tr].reset_index(drop=True), X[te], mt, C.SEED), y[te]


def test_minirocket_deterministic_row_independent_and_label_free():
    X, _, _, _ = _synthetic()
    a = MiniRocket(500, seed=3).fit(X[:, :8])
    b = MiniRocket(500, seed=3).fit(X[:, :8])
    Fa, Fb = a.transform(X[:, :8]), b.transform(X[:, :8])
    assert Fa.shape == (len(X), a.n_features_) and np.array_equal(Fa, Fb)
    sub = np.arange(len(X))[::-3]
    assert np.array_equal(a.transform(X[sub, :8]), Fa[sub])          # a row never depends on the batch
    assert np.all((Fa >= 0) & (Fa <= 1))
    assert 'y' not in MiniRocket.fit.__code__.co_varnames              # fit takes no label


@pytest.mark.parametrize('model', ['lr', 'gbm'])
def test_panel_predictions_do_not_depend_on_the_test_batch(model):
    fold, _ = _fold()
    fp = PN.make_fit_predict(model)
    p, info = fp(fold)
    assert PN.is_finite_proba(p) and info['n_train'] == len(fold.y_train)
    sub = np.arange(fold.n_test)[::-2]
    f2 = Fold(0, fold.test_person, fold.X_train, fold.y_train, fold.groups_train, fold.meta_train,
              fold.X_test[sub], fold.meta_test.iloc[sub].reset_index(drop=True), fold.seed)
    p2, _ = fp(f2)
    assert np.allclose(p2, p[sub], atol=1e-12, rtol=0)


def test_lr_mirror_antisymmetric_on_chiral_channels():
    fold, _ = _fold()
    p, _ = PN.make_fit_predict('lr')(fold)
    fm = Fold(0, fold.test_person, fold.X_train, fold.y_train, fold.groups_train, fold.meta_train,
              mirror_channels(fold.X_test), fold.meta_test, fold.seed)
    pm, _ = PN.make_fit_predict('lr')(fm)
    assert np.allclose(pm, 1 - p, atol=1e-4)          # exact up to the solver tolerance on the intercept


def test_panel_learns_the_chiral_signal_and_mr_runs():
    fold, y_te = _fold()
    for m in ('lr', 'mr'):
        p, _ = PN.make_fit_predict(m)(fold)
        assert np.mean((p >= 0.5) == y_te) >= 0.8, m


def test_normaliser_fit_on_training_only_and_trial_peak_per_trial():
    X, _, _, _ = _synthetic()
    nz = PN.Normaliser('zscore').fit(X[:40])
    Z = nz.transform(X[:40])
    assert np.allclose(np.transpose(Z, (1, 0, 2)).reshape(N_CH, -1).mean(axis=1), 0, atol=1e-5)
    tp = PN.Normaliser('trial_peak').fit(X[:5])
    assert np.allclose(tp.transform(X[7:8]), tp.transform(X[[7, 3]])[0:1])
    assert np.allclose(np.abs(tp.transform(X)).max(axis=2), 1.0, atol=1e-6)
    w = PN.Normaliser('none', winsor=0.01).fit(X[:40])
    assert w.transform(X).max() <= w.hi_.max() + 1e-6


def test_training_filters():
    fold, _ = _fold()
    keep = PN._train_mask(fold, (('platform', 'cross'),))
    assert set(fold.meta_train.loc[keep, 'platform']) == {'android'}       # the test person P06 is on iOS
    keep = PN._train_mask(fold, (('platform', 'same'),))
    assert set(fold.meta_train.loc[keep, 'platform']) == {'ios'}
    keep = PN._train_mask(fold, (('drop_flags', ('grip_mismatch',)),))
    assert not fold.meta_train.loc[keep, 'grip_mismatch'].any()


def test_cap_after_filter_reproduces_the_harness_cap_on_the_kept_rows():
    from stage2.cv import cap_rows
    X, y, g, meta = _synthetic(n_per_person=12)
    meta = meta.copy()
    meta['analysis_set'] = np.where(np.arange(len(y)) % 5 == 0, 'excluded', 'primary')
    tr = g != 'P06'
    mt = meta[~tr][['person_id', 'platform']].reset_index(drop=True)
    mt.insert(0, 'row', np.arange(int((~tr).sum())))
    fold = Fold(0, 'P06', X[tr], y[tr], g[tr], meta[tr].reset_index(drop=True), X[~tr], mt, C.SEED)
    keep = PN._train_mask(fold, (('sets', ('primary',)), ('cap', 5)))
    prim = np.flatnonzero(fold.meta_train['analysis_set'] == 'primary')
    want = prim[cap_rows(fold.groups_train[prim], fold.y_train[prim], 5, C.SEED)]
    assert np.array_equal(np.flatnonzero(keep), np.sort(want))
    assert not PN.needs_harness_cap(PN.Treatment(train_filter=(('sets', ('primary',)), ('cap', 5))))
    assert PN.needs_harness_cap(PN.DEFAULT)


def test_size_matched_controls_keep_the_same_counts():
    fold, _ = _fold()
    m = fold.meta_train
    flagged = m['grip_mismatch'].to_numpy(bool)
    drop = PN._train_mask(fold, (('drop_flags', ('grip_mismatch',)),))
    match = PN._train_mask(fold, (('match_flags', ('grip_mismatch',)),))
    cells = pd.DataFrame({'g': fold.groups_train, 'y': fold.y_train})
    assert (cells[drop].value_counts().sort_index() == cells[match].value_counts().sort_index()).all()
    assert np.array_equal(match, PN._train_mask(fold, (('match_flags', ('grip_mismatch',)),)))   # seeded
    if flagged.any():
        assert match[flagged].any() or not (~drop).any()            # a random choice, not the clean rows
    for side in ('same', 'cross'):
        a = PN._train_mask(fold, (('platform', side),))
        b = PN._train_mask(fold, (('match_persons', side),))
        assert len(set(fold.groups_train[a])) == len(set(fold.groups_train[b]))


def test_nested_calibrator_candidates_preserve_the_ranking():
    rng = np.random.default_rng(3)
    p = K.sigmoid(rng.normal(0, 2, 500))
    y = (rng.random(500) < p).astype(int)
    assert 'isotonic' not in K.AUTO_CANDIDATES and set(K.AUTO_CANDIDATES) <= set(K.RANK_PRESERVING)
    for meth in (c for c in K.AUTO_CANDIDATES if c != 'none'):
        q = K.Calibrator(meth).fit(p, y)(p)
        o = np.argsort(p)
        assert np.all(np.diff(q[o]) > 0), meth                         # strictly increasing: no new ties
    flat = K.Calibrator('platt').fit(p, 1 - y)                          # an anti-correlated fit never reverses p
    assert flat.params_['a'] > 0


def test_augmentation_shapes_and_labels():
    X, y, _, _ = _synthetic()
    tr = PN.Treatment(mirror=True, screen_flip=True, time_warp=True)
    Xa, ya, real = PN.augment([X[:10]], y[:10], tr, np.random.default_rng(0))
    assert Xa.shape[0] == 10 * 2 * 2 * 2 and real.sum() == 10
    assert np.array_equal(ya[:40], np.tile(y[:10], 4)) and np.array_equal(ya[40:], 1 - ya[:40])


def test_calibrators_and_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, 4000)
    y = (rng.random(4000) < K.sigmoid(z / 2.5)).astype(int)        # true temperature 2.5 on the logit z
    T = K.fit_temperature(z, y)
    assert abs(T - 2.5) < 0.3
    p = K.sigmoid(z)
    for meth in K.METHODS:
        q = K.Calibrator(meth).fit(p, y)(p)
        assert np.all((q > 0) & (q < 1))
        assert np.all(np.diff(q[np.argsort(p)]) >= -1e-9), meth          # monotone in p
        assert K._ll(y, q) < K._ll(y, p)                                  # an over-confident p is improved


def test_nested_calibrator_choice_uses_training_predictions_only():
    rng = np.random.default_rng(2)
    n = 600
    groups = np.repeat([f'P{i:02d}' for i in range(12)], n // 12)
    z = rng.normal(0, 2, n)
    y = (rng.random(n) < K.sigmoid(z)).astype(int)
    best, losses = K.select_calibrator(K.sigmoid(z), y, groups)           # already calibrated
    assert set(losses) == set(K.AUTO_CANDIDATES)
    assert losses[best] <= losses['none'] + 1e-12
    assert losses['none'] - losses[best] < 0.01                            # nothing to gain
    best2, losses2 = K.select_calibrator(K.sigmoid(4 * z), y, groups)      # over-confident x4
    assert best2 != 'none' and losses2[best2] < losses2['none'] - 0.05


def test_auto_calibrated_fit_predict_on_a_fold():
    fold, y_te = _fold()
    p, info = K.make_calibrated('lr', 'auto')(fold)
    assert PN.is_finite_proba(p) and len(p) == fold.n_test
    assert info['chosen'] in K.AUTO_CANDIDATES and info['calibrator'] == 'auto'
    assert {f'cv_logloss_{c}' for c in K.AUTO_CANDIDATES} <= set(info)
    with pytest.raises(ValueError):
        K.make_calibrated('lr', 'unknown')


def test_panel_fit_predict_is_deterministic():
    fold, _ = _fold()
    for m in ('lr', 'gbm'):
        p1, _ = PN.make_fit_predict(m)(fold)
        p2, _ = PN.make_fit_predict(m)(fold)
        assert np.array_equal(p1, p2), m


def test_lane_csvs_always_have_a_header():
    from stage2.pipeline import Lane
    files = sorted(Lane('p1').tab_dir.glob('*.csv'))
    if not files:
        pytest.skip('tables not built yet')
    for f in files:
        assert len(pd.read_csv(f).columns) > 0, f.name


def test_empty_figure_and_table_frames_keep_their_columns():
    from stage2.lanes import p1_figures as F
    from stage2.lanes import p1_tables as T
    empty = F._concat([], F.DELTA_COLUMNS)
    assert list(empty.columns) == list(F.DELTA_COLUMNS) and not len(empty)
    assert list(F._delta_points(pd.DataFrame(columns=['model', 'option', 'd_acc', 'd_acc_lo', 'd_acc_hi', 'verdict']),
                                'lr', [('lp5', 5.0)]).columns) == list(F.DELTA_COLUMNS)
    assert T.AGG_TABLE_COLUMNS[0] == 'Model'


def test_the_tie_rule_equals_the_harness():
    """p1_stats.correct (no import of the harness: the presentation modules stay light) is cv.credit: a tie, 0.5 up
    to rounding, earns half credit (the harness's rule since 2026-10-28)."""
    from stage2 import cv
    from stage2.lanes import p1_stats as ST
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.random(400), 1)
    p[:5] = 0.5 + np.array([1e-16, -1e-16, 1e-13, 2e-12, -5e-12])          # three ties up to rounding, two not
    assert np.array_equal(ST.correct(y, p), cv.credit(y, p))
    assert (ST.correct(y, p)[:5] == 0.5).tolist() == [True, True, True, False, False]


def test_fast_metrics_and_bootstrap_equal_the_harness():
    from stage2 import cv
    from stage2.lanes import p1_derive as D
    rng = np.random.default_rng(5)
    n = 300
    pred = pd.DataFrame({'person_id': np.repeat([f'P{i:02d}' for i in range(10)], n // 10),
                         'y': rng.integers(0, 2, n), 'p_right': np.round(rng.random(n), 2)})   # ties on purpose
    a, b = D.fast_metrics(pred['y'], pred['p_right']), cv.metrics(pred['y'], pred['p_right'])
    for k in D.BLOCK_METRICS:
        assert a[k] == pytest.approx(b[k], abs=1e-12), k
    fa = D.fast_person_bootstrap(pred, n_boot=200)
    fb = cv.person_bootstrap(pred, n_boot=200, names=D.BLOCK_METRICS)
    for k in D.BLOCK_METRICS:
        assert fa[k] == pytest.approx(fb[k], abs=1e-12), k


def test_calibration_fit_is_damped_vectorised_and_matches_statsmodels():
    import statsmodels.api as sm
    from stage2.lanes import p1_stats as ST
    rng = np.random.default_rng(11)
    n = 900
    z = rng.normal(0, 6, n)                                   # extreme logits: plain Newton diverges from (0, 1)
    y = (rng.random(n) < ST.sigmoid(z / 2.5)).astype(int)
    p = ST.sigmoid(z)
    b, a, a0 = ST.calibration_fit(y, p)
    zz = ST.logit(p)
    ref = sm.GLM(y, sm.add_constant(zz), family=sm.families.Binomial()).fit().params
    ref0 = sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial(), offset=zz).fit().params
    assert b == pytest.approx(ref[1], abs=1e-7) and a == pytest.approx(ref[0], abs=1e-7)
    assert a0 == pytest.approx(ref0[0], abs=1e-7)
    groups = np.repeat(np.arange(15), n // 15)
    W = ST.person_weights(groups, ST.boot_picks(15, 20))
    many = ST.calibration_fit_many(y, p, W)
    for i in (0, 9, 19):
        one = ST.calibration_fit(y, p, W[i])
        assert np.allclose([many[0][i], many[1][i], many[2][i]], one, atol=1e-12, equal_nan=True)


def test_weighted_auc_and_pooled_metrics_equal_the_expanded_resample():
    from stage2.lanes import p1_stats as ST
    rng = np.random.default_rng(12)
    n, P = 240, 12
    y = rng.integers(0, 2, n)
    p = np.round(rng.random(n), 2)                            # ties on purpose
    groups = np.repeat(np.arange(P), n // P)
    picks = ST.boot_picks(P, 30)
    W = ST.person_weights(groups, picks)
    auc = ST.weighted_auc_many(y, p, W)
    many = ST.pooled_metrics_many(y, p, W)
    rows_of = [np.flatnonzero(groups == i) for i in range(P)]
    for i in (0, 13, 29):
        rows = np.concatenate([rows_of[j] for j in picks[i]])
        assert auc[i] == pytest.approx(ST._auc(y[rows], p[rows]), abs=1e-12)
        one = ST.pooled_metrics(y, p, W[i], rows)
        for k, v in one.items():
            assert many[k][i] == pytest.approx(v, abs=1e-10, nan_ok=True), k


def test_fast_paired_compare_and_sign_flip_equal_the_harness():
    import warnings
    from stage2 import cv
    from stage2.lanes import p1_stats as ST
    rng = np.random.default_rng(13)
    n = 160
    persons = np.repeat([f'P{i:02d}' for i in range(8)], n // 8)
    y = rng.integers(0, 2, n)
    pa = pd.DataFrame({'trial_id': [f't{i}' for i in range(n)], 'person_id': persons, 'y': y,
                       'p_right': np.clip(0.5 + (2 * y - 1) * rng.normal(0.15, 0.2, n), 0.01, 0.99)})
    pb = pa.assign(p_right=np.clip(pa['p_right'] + rng.normal(0, 0.1, n), 0.01, 0.99))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = cv.paired_compare(pa, pb, n_boot=200)
    fast = ST.paired_compare(pa, pb, n_boot=200)
    for k, v in ref.items():
        if k == 'per_person' or k not in fast:
            continue
        if isinstance(v, tuple):
            assert np.allclose(v, fast[k], atol=1e-12), k
        else:
            assert fast[k] == pytest.approx(v, abs=1e-12), k
    d = rng.normal(0.01, 0.05, 13)
    assert ST.sign_flip_p(d) == pytest.approx(cv._sign_flip_p(d), abs=1e-12)


def test_subset_status_calls_inconclusive_when_the_ci_covers_chance_and_the_primary():
    from stage2.lanes import p1_derive as D
    assert D.vs_primary(0.54, 0.14, 0.82, 0.81).startswith('inconclusive')
    assert D.vs_primary(0.60, 0.45, 0.70, 0.81) == 'below the primary accuracy'
    assert D.vs_primary(0.90, 0.83, 1.00, 0.81) == 'above the primary accuracy'
    assert D.vs_primary(0.85, 0.70, 0.95, 0.81) == 'compatible with the primary accuracy'
    assert D.vs_primary(0.85, None, None, 0.81) == 'too few persons for a CI'


def test_presentation_modules_import_no_model_code():
    import subprocess
    import sys
    code = ('import sys; import stage2.lanes.p1_figures, stage2.lanes.p1_tables, stage2.lanes.p1_summary, '
            'stage2.lanes.p1_numbers; heavy = [m for m in ("stage2.lanes.p1_panel", "stage2.lanes.p1_variants", '
            '"stage2.lanes.p1_experiments", "lightgbm", "scipy.signal", "sklearn") if m in sys.modules]; print(heavy)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=str(C.ANALYSIS_DIR),
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '[]', out.stdout


def _derived_ready() -> bool:
    from stage2.pipeline import Lane
    L = Lane('p1')
    return all((L.tab_dir / f'{n}.json').exists() for n in ('p1_runs', 'p1_comparisons', 'p1_subsets', 'p1_best',
                                                            'p1_calibration', 'p1_decisions', 'p1_filter_delay',
                                                            'p1_variants'))


def _summary_leftover_numbers(text: str) -> list[str]:
    """Numeric tokens of a summary built with ``mark=True`` once the numbers printed from the numbers file (between
    the sentinels), code spans, printed labels and definitions are removed: should be none."""
    import re
    from stage2.lanes import p1_labels as LB
    from stage2.lanes.p1_numbers import MARK_CLOSE, MARK_OPEN
    from stage2.pipeline import Lane
    text = re.sub(re.escape(MARK_OPEN) + '.*?' + re.escape(MARK_CLOSE), ' ', text, flags=re.S)
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)
    text = re.sub(r'`[^`]*`', ' ', text)                                  # code spans: file and result names
    # printed labels and definitions (option and family names, the defaults, figure captions written through the
    # numbers formatter by the figures step) are not results; remove them before looking for numbers
    L = Lane('p1')
    labels = [*LB.OPTION_LABELS.values(), *LB.FAMILY_LABELS.values(), *LB.FLAG_LABELS.values(),
              *LB.MODEL_LABELS.values(), *LB.CAL_LABELS.values(), *LB.RULE_LABELS.values()]
    dec = json.loads((L.tab_dir / 'p1_decisions.json').read_text(encoding='utf-8'))
    labels += [r['default'] for r in dec['rows']] + [dec['rule']]
    sub = json.loads((L.tab_dir / 'p1_subsets.json').read_text(encoding='utf-8'))['rows']
    labels += [r['label'] for r in sub] + [LB.subset_label(r['label']) for r in sub]
    cmp_ = json.loads((L.tab_dir / 'p1_comparisons.json').read_text(encoding='utf-8'))
    labels += [r['what'] for r in cmp_.get('remedies', []) + cmp_.get('controls', [])]
    caps = L.fig_dir / 'captions.json'
    if caps.exists():
        labels += [v.get(k, '') for v in json.loads(caps.read_text(encoding='utf-8')).values()
                   for k in ('caption', 'decision')]
    for lab in sorted({x for x in labels if x}, key=len, reverse=True):
        text = text.replace(lab, ' ')
    definitions = re.compile(r'(E4[a-d]|P1_\d\d|D\d+\b|T_\d|k = \d|k = k|95 %|95th|Stage \d|section \d+\.\d+|docs/\d+|'
                             r'exactly 0\.5|1-Hz|\d+ ms late|up to \d+ ms|= 1 - p|1 = calibrated|'
                             r'(?:below|above|exceeds|covers|about|of|to) 1\b|(?:excludes|includes|include|below|above|'
                             r'beyond) 0\b|^\d+\. |\| \w+ \| \d+ \|)', flags=re.M)   # list numbers; numeric levels
    text = definitions.sub(' ', text)
    return re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?', text)


def test_every_number_in_the_summary_comes_from_the_numbers_file():
    from stage2.lanes import p1_numbers, p1_summary
    if not _derived_ready():
        pytest.skip('derived files not built yet')
    entries = p1_numbers.build()
    bad = _summary_leftover_numbers(p1_summary.build(entries, mark=True))
    assert not bad, f'numbers printed without the numbers formatter: {sorted(set(bad))[:30]}'
    typed = _summary_leftover_numbers(p1_summary.build(entries, mark=True) + '\nA typed result: 77.7% of 13 persons.\n')
    assert typed == ['77.7%', '13']                                      # the check does catch a typed number


def test_finalists_are_standard_entries_with_the_calibrated_gbm():
    from stage2.pipeline import FINALIST_KEYS, Lane
    p = Lane('p1').finalists_path
    if not p.exists():
        pytest.skip('finalists not written yet')
    fin = json.loads(p.read_text(encoding='utf-8'))
    assert 1 <= len(fin) <= 3
    for f in fin:
        assert set(FINALIST_KEYS) <= set(f)
    by = {f['name']: f for f in fin}
    assert by.get('p1_calib_gbm_auto', {}).get('params') == {'model': 'gbm', 'calibration': 'auto'}
    assert not any(f['params'].get('calibration') and f['params'].get('model') == 'lr' for f in fin)


def test_aggregates_use_the_same_persons_at_every_k():
    from stage2.pipeline import Lane
    if not _derived_ready():
        pytest.skip('derived files not built yet')
    cal = json.loads((Lane('p1').tab_dir / 'p1_calibration.json').read_text(encoding='utf-8'))
    for m, a in cal['aggregates'].items():
        assert len({r['n_persons'] for r in a['rows']}) == 1, m
        assert {r['n_persons'] for r in a['rows']} == {a['n_persons']}


def test_variant_fidelity_recorded_zero():
    from stage2.pipeline import Lane
    p = Lane('p1').tab_dir / 'p1_variants.json'
    if not p.exists():
        pytest.skip('derived files not built yet')
    assert json.loads(p.read_text(encoding='utf-8'))['fidelity']['max_abs_diff'] == 0.0


def test_figures_redraw_from_saved_files():
    from stage2.lanes import p1_figures
    from stage2.pipeline import Lane
    L = Lane('p1')
    need = ('p1_runs', 'p1_comparisons', 'p1_subsets', 'p1_calibration', 'p1_filter_delay')
    if not all((L.tab_dir / f'{n}.json').exists() for n in need):
        pytest.skip('derived files not built yet')
    p1_figures.draw(['P1_02_cutoff', 'P1_10_filter_delay'])
    caps = json.loads((L.fig_dir / 'captions.json').read_text(encoding='utf-8'))
    assert {'P1_02_cutoff', 'P1_10_filter_delay'} <= set(caps)
    assert (L.tab_dir / 'P1_02_cutoff.csv').exists()
