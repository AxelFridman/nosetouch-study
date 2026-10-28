"""The LOSO harness cannot leak a person or a label, detects test-set fitting, and its metrics match sklearn."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from stage2 import cv


def toy(n_persons: int = 8, per: int = 12, seed: int = 0, signal: float = 1.5):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for p in range(n_persons):
        for k in range(per):
            y = k % 2
            X.append([signal * (2 * y - 1) + rng.normal() + 0.5 * p, rng.normal()])
            rows.append({'trial_id': f't{p}_{k}', 'person_id': f'P{p:02d}', 'y_right': y, 'hand': 'right' if y else 'left',
                         'session_id': f's{p}', 'platform': 'android' if p % 2 else 'ios', 'trial_index': k})
    return np.asarray(X), pd.DataFrame(rows)


def lr_fit_predict(fold):
    sc = StandardScaler().fit(fold.X_train)
    m = LogisticRegression().fit(sc.transform(fold.X_train), fold.y_train)
    return m.predict_proba(sc.transform(fold.X_test))[:, 1], {'C': 1.0}


def const(fold):
    return np.full(fold.n_test, 0.5)


QUICK = dict(save=False, n_boot=0, leak_check='off')


def test_loso_never_shares_a_person():
    _, meta = toy()
    splits = cv.loso_splits(meta['person_id'])
    assert len(splits) == meta['person_id'].nunique()
    for s in splits:
        assert set(meta['person_id'].iloc[s.train]).isdisjoint(set(meta['person_id'].iloc[s.test]))
        assert set(meta['person_id'].iloc[s.test]) == {s.person}
    assert sorted(np.concatenate([s.test for s in splits]).tolist()) == list(range(len(meta)))


def test_loso_on_the_real_groups_is_by_person(primary, expected):
    splits = cv.loso_splits(primary.groups)
    assert len(splits) == expected['primary']['persons']
    codes = primary.trials['participant_code'].to_numpy()
    for s in splits:
        assert set(primary.groups[s.train]).isdisjoint(primary.groups[s.test])
    # grouping by the app's code would put P01 and P13 (one phone, one code) in the same fold: by person they never are
    s13 = [s for s in splits if s.person == 'P13'][0]
    assert 'P01' in set(primary.groups[s13.train])
    assert set(codes[s13.test]) & set(codes[s13.train])  # same code on both sides, different persons


def test_assert_disjoint_raises():
    bad = [cv.Split(0, 'P00', np.array([0, 1]), np.array([1, 2]))]
    with pytest.raises(cv.LeakageError):
        cv.assert_disjoint(bad, ['P00', 'P00', 'P01'])


# ───────────────────────────── finding 1: person and label leaks ─────────────────────────────


def test_grouping_by_code_or_session_is_refused(primary):
    """The verifier's attacks (a): groups by participant_code (splits P14, merges P01 + P13), by session
    (splits P01, P09, P14), or a meta person_id overwritten with the code all raise instead of running."""
    X, meta = primary.approach, primary.trials
    with pytest.raises(cv.LeakageError, match='split persons'):
        cv.run_experiment('a', const, X, meta, groups=meta['participant_code'], **QUICK)
    with pytest.raises(cv.LeakageError, match='split persons'):
        cv.run_experiment('a', const, X, meta, groups=meta['session_id'], **QUICK)
    bad = meta.assign(person_id=meta['participant_code'])
    with pytest.raises(cv.LeakageError, match='curated person_id'):
        cv.run_experiment('a', const, X, bad, **QUICK)
    # a coarser grouping that never splits a person runs, as leave-group-out, with a warning
    plat = meta['platform'].astype(str) + '_' + (meta['person_id'].str[1:].astype(int) % 3).astype(str)
    with pytest.warns(UserWarning, match='leave-group-out'):
        res = cv.run_experiment('a', const, X, meta, groups=plat, **QUICK)
    assert res.info['scheme'] == 'leave-group-out'
    assert set(res.predictions['person_id']) == set(meta['person_id'])  # persons stay the reporting unit


def test_label_in_X_is_refused():
    """Attack (b): a label column, key or binary copy of the label inside X."""
    X, meta = toy()
    y = meta['y_right'].to_numpy()
    df = pd.DataFrame(X, columns=['f0', 'f1'])
    for bad in (df.assign(y_right=y), df.assign(hand=meta['hand']), df.assign(orientation='upright_toward'),
                df.assign(trial_index=meta['trial_index']), df.assign(target_proxy=0.3 * y), df.assign(age_years=25)):
        with pytest.raises(cv.LeakageError):
            cv.run_experiment('b', const, bad, meta, **QUICK)
    with pytest.raises(cv.LeakageError):                                  # an unnamed binary copy of y
        cv.run_experiment('b', const, df.assign(f2=np.where(y == 1, 2.5, -1.0)), meta, **QUICK)
    with pytest.raises(cv.LeakageError):                                  # 1 - y inside a numpy tensor
        cv.run_experiment('b', const, np.stack([X, np.column_stack([1 - y, y * 0])], axis=1), meta, **QUICK)
    with pytest.raises(cv.LeakageError):                                  # a dict key
        cv.run_experiment('b', const, {'x': X, 'label': y}, meta, **QUICK)
    res = cv.run_experiment('b', const, df.assign(platform_ios=(meta['platform'] == 'ios').astype(int)), meta, **QUICK)
    assert len(res.predictions) == len(meta)                              # a genuine binary feature is fine


def test_test_side_has_no_lookup_key_and_the_cache_is_locked(primary):
    """Attack (c): no trial_id on the test side (an opaque row index instead), and the cache loaders refuse to
    run inside fit_predict, so test labels cannot be looked up."""
    from stage2.data import canonical_trial_map, load_cache, load_tables
    seen = []

    def spy(fold):
        seen.append(list(fold.meta_test.columns))
        assert 'trial_id' not in fold.meta_test.columns and 'session_id' not in fold.meta_test.columns
        assert (fold.meta_test['row'].to_numpy() == np.arange(fold.n_test)).all()
        return np.full(fold.n_test, 0.5)

    cv.run_experiment('c', spy, primary.approach, primary.trials, **QUICK)
    assert seen and all(not (cv.FORBIDDEN_TEST_COLUMNS & set(c)) for c in seen)
    for loader in (lambda: load_cache(), lambda: load_tables(), lambda: canonical_trial_map()):
        def cheat(fold, loader=loader):
            loader()
            return np.full(fold.n_test, 0.5)
        with pytest.raises(cv.LeakageError, match='inside fit_predict'):
            cv.run_experiment('c', cheat, primary.approach, primary.trials, **QUICK)
    with pytest.raises(cv.LeakageError):
        cv.run_experiment('c', spy, primary.approach, primary.trials, test_meta_columns=('row', 'trial_id'), **QUICK)


def test_natural_use_and_excluded_rows_need_an_explicit_opt_in(primary, natural):
    """Attack (d): natural-use trials appended to the rows are refused (they go through extra=), and so are
    excluded trials unless the caller opts in."""
    X = np.concatenate([primary.approach, natural.approach])
    meta = pd.concat([primary.trials, natural.trials], ignore_index=True)
    with pytest.raises(ValueError, match='not allowed'):
        cv.run_experiment('d', const, X, meta, **QUICK)
    with pytest.raises(ValueError, match='never trained on'):
        cv.run_experiment('d', const, X, meta, sets=('primary', 'natural_use'), **QUICK)
    from stage2.data import load_cache
    allc = load_cache(sets=None)
    sub = allc.subset(allc.trials['analysis_set'].isin(['primary', 'excluded']).to_numpy())
    with pytest.raises(ValueError, match='not allowed'):
        cv.run_experiment('d', const, sub.approach, sub.trials, **QUICK)
    res = cv.run_experiment('d', const, sub.approach, sub.trials, sets=('primary', 'excluded'), **QUICK)
    assert res.info['sets'] == ['primary', 'excluded']
    some = (primary.approach[:5], primary.trials.iloc[:5])
    with pytest.raises(ValueError, match='not allowed'):                 # primary trials are no extra rows
        cv.run_experiment('d', const, primary.approach, primary.trials, extra=some, **QUICK)
    with pytest.raises(cv.LeakageError, match='both rows and extra'):  # and never rows AND extra rows
        cv.run_experiment('d', const, primary.approach, primary.trials, extra=some,
                          extra_sets=('natural_use', 'primary'), **QUICK)


def test_test_rows_are_shuffled_and_mapped_back():
    X, meta = toy()
    orders = []

    def by_row(fold):  # a per-row function of X: the harness must return it in the original row order
        orders.append(fold.X_test[:, 0].copy())
        return 1.0 / (1.0 + np.exp(-fold.X_test[:, 0]))

    res = cv.run_experiment('e', by_row, X, meta, **QUICK)
    assert np.allclose(res.predictions['p_right'], 1.0 / (1.0 + np.exp(-X[:, 0])))
    first = X[meta['person_id'] == 'P00', 0]
    assert not np.array_equal(orders[0], first) and np.allclose(np.sort(orders[0]), np.sort(first))


def test_fit_predict_sees_no_test_labels():
    X, meta = toy()
    seen = []

    def spy(fold):
        seen.append(set(fold.meta_test.columns))
        assert not (cv.LABEL_COLUMNS & set(fold.meta_test.columns))
        assert fold.test_person not in set(fold.groups_train)
        assert len(fold.y_train) == len(fold.X_train)
        return np.full(fold.n_test, 0.5)

    cv.run_experiment('spy', spy, X, meta, **QUICK)
    assert seen and all('hand' not in c and 'y_right' not in c and 'trial_index' not in c for c in seen)
    with pytest.raises(cv.LeakageError):
        cv.run_experiment('spy2', spy, X, meta, save=False, n_boot=0, test_meta_columns=('row', 'hand'))


# ───────────────────────────── finding 2: the leak check covers every fold ─────────────────────────────


def test_scaler_fitted_on_test_data_is_detected():
    X, meta = toy()

    def leaky(fold):
        sc = StandardScaler().fit(fold.X_test)  # statistics of the held-out person
        m = LogisticRegression().fit(StandardScaler().fit_transform(fold.X_train), fold.y_train)
        return m.predict_proba(sc.transform(fold.X_test))[:, 1]

    with pytest.raises(cv.LeakageError):
        cv.run_experiment('leaky', leaky, X, meta, save=False, n_boot=0)
    res = cv.run_experiment('clean', lr_fit_predict, X, meta, save=False, n_boot=0)
    assert res.leak_check.startswith('passed') and res.info['leak_checked_folds'] == list(range(8))


def test_a_leak_in_a_later_fold_is_caught_by_default():
    """The verifier's case: test-set standardisation in every fold except fold 0 passed 'first_fold'."""
    X, meta = toy()

    def leaky_late(fold):
        sc = StandardScaler().fit(fold.X_train if fold.fold == 0 else fold.X_test)
        m = LogisticRegression().fit(StandardScaler().fit_transform(fold.X_train), fold.y_train)
        return m.predict_proba(sc.transform(fold.X_test))[:, 1]

    with pytest.raises(cv.LeakageError, match='fold 1'):
        cv.run_experiment('late', leaky_late, X, meta, save=False, n_boot=0)
    res = cv.run_experiment('late', leaky_late, X, meta, save=False, n_boot=0, leak_check='off')
    assert res.leak_check == 'off' and res.info['leak_checked_folds'] == []
    with pytest.raises(ValueError):
        cv.run_experiment('late', leaky_late, X, meta, save=False, n_boot=0, leak_check='first_fold')


def test_sampled_leak_check_records_its_folds():
    X, meta = toy()
    res = cv.run_experiment('s', lr_fit_predict, X, meta, save=False, n_boot=0, leak_check='sample')
    assert len(res.info['leak_checked_folds']) == 3 and res.leak_check == 'passed (3/8 folds)'


def test_nondeterministic_fit_predict_warns_instead_of_failing():
    X, meta = toy()
    rng = np.random.default_rng(1)

    def noisy(fold):
        return np.clip(0.5 + 0.1 * rng.normal(size=fold.n_test), 0, 1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = cv.run_experiment('noisy', noisy, X, meta, save=False, n_boot=0)
    assert res.leak_check.startswith('nondeterministic')
    assert any('not deterministic' in str(x.message) for x in w)


def test_inner_splits_are_by_person():
    _, meta = toy()
    tr = cv.loso_splits(meta['person_id'])[0].train
    g = meta['person_id'].to_numpy()[tr]
    for a, b in cv.inner_splits(g, 4):
        assert set(g[a]).isdisjoint(g[b])


# ───────────────────────────── finding 5: D2 cap, configurations tried, gates ─────────────────────────────


def test_cap_rows_is_seeded_hand_balanced_and_per_person():
    g = np.array(['A'] * 45 + ['B'] * 30)
    y = np.array([0] * 23 + [1] * 22 + [0, 1] * 15)
    keep = cv.cap_rows(g, y, 40)
    assert (g[keep] == 'A').sum() == 40 and (g[keep] == 'B').sum() == 30
    assert (y[keep][g[keep] == 'A'] == 0).sum() == 20
    again = cv.cap_rows(g[:45], y[:45], 40)          # the same rows of A whoever else is in the fold
    assert np.array_equal(again, keep[g[keep] == 'A'])
    lop = cv.cap_rows(np.array(['C'] * 30), np.array([0] * 5 + [1] * 25), 20)
    assert lop.size == 20                              # a hand with few rows leaves its share to the other
    assert np.array_equal(cv.cap_rows(g, y, None), np.arange(g.size))


def test_run_experiment_caps_training_rows_and_records_configs(tmp_path):
    X, meta = toy(per=12)
    big = meta['person_id'] == 'P00'
    extra_rows = meta[big].assign(trial_id=lambda m: m['trial_id'] + 'b')
    meta2 = pd.concat([meta, extra_rows], ignore_index=True)            # P00 now has 24 trials
    X2 = np.concatenate([X, X[big.to_numpy()] + 0.01])
    seen = []

    def spy(fold):
        seen.append(pd.Series(fold.groups_train).value_counts().max())
        return lr_fit_predict(fold)

    res = cv.run_experiment('cap', spy, X2, meta2, cap_per_person=16, n_configs_tried=18, save=True, n_boot=0,
                            out_dir=tmp_path, registry=tmp_path / 'registry.csv', leak_check='off')
    assert max(seen) <= 16 and res.info['capped_training_rows_per_fold'] == {'P00': 8}
    reg = pd.read_csv(tmp_path / 'registry.csv')
    assert int(reg.loc[0, 'n_configs_tried']) == 18 and int(reg.loc[0, 'cap_per_person']) == 16
    with pytest.raises(ValueError):
        cv.run_experiment('cap', spy, X2, meta2, n_configs_tried=0, **QUICK)


def test_gated_metrics_use_the_trials_that_pass_the_gates():
    X, meta = toy()
    meta = meta.assign(gate_pass=np.arange(len(meta)) % 4 != 0)
    res = cv.run_experiment('g', lr_fit_predict, X, meta, **QUICK)
    gp = meta['gate_pass'].to_numpy()
    assert res.gated_metrics['n'] == gp.sum()
    assert res.gated_metrics['accuracy'] == pytest.approx(np.mean((res.predictions['p_right'][gp] >= 0.5)
                                                                  == meta['y_right'][gp]))


def test_e4_bounds_follow_gate_g2():
    from stage2 import config as C
    assert C.MIN_DURATION_MS == C.G2_MIN_DURATION_MS and C.MIN_PEAK_GYRO_DPS == 30.0


# ───────────────────────────── metrics ─────────────────────────────


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 300)
    p = np.clip(rng.beta(2, 2, 300) * 0.6 + 0.4 * y, 0, 1)
    p[:20] = np.round(p[:20], 1)  # ties
    p[20] = 0.0
    p[21] = 1.0
    m = cv.metrics(y, p)
    pred = (p >= 0.5).astype(int)
    tie = p == 0.5                                   # the rounded block holds exact ties: they earn half credit
    ok = np.where(tie, 0.5, (pred == y).astype(float))
    assert m['accuracy'] == pytest.approx(ok.mean(), abs=1e-12)
    assert m['balanced_accuracy'] == pytest.approx(np.mean([ok[y == c].mean() for c in (0, 1)]), abs=1e-12)
    if not tie.any():
        assert m['accuracy'] == pytest.approx(accuracy_score(y, pred), abs=1e-12)
        assert m['balanced_accuracy'] == pytest.approx(balanced_accuracy_score(y, pred), abs=1e-12)
    nt = ~tie                                        # off the ties, the rule is sklearn's
    assert accuracy_score(y[nt], pred[nt]) == pytest.approx(np.mean(cv.credit(y[nt], p[nt])), abs=1e-12)
    assert m['roc_auc'] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert m['log_loss'] == pytest.approx(log_loss(y, p, labels=[0, 1]), rel=1e-9)
    assert m['brier'] == pytest.approx(brier_score_loss(y, p), abs=1e-12)


def test_ece_and_reliability():
    y = np.array([0, 0, 1, 1] * 25)
    assert cv.ece(y, np.full(100, 0.5)) == pytest.approx(0.0)
    p = np.where(y == 1, 1.0, 0.0)
    assert cv.ece(y, p) == pytest.approx(0.0)
    assert cv.ece(y, 1 - p) == pytest.approx(1.0)
    r = cv.reliability(y, np.linspace(0, 1, 100))
    assert len(r) == 10 and int(r['n'].sum()) == 100
    # equal-mass by default; uniform on request
    assert cv.ece(y, np.full(100, 0.8), strategy='uniform') == pytest.approx(0.3)


def test_person_bootstrap_is_reproducible_and_brackets_the_estimate():
    X, meta = toy(signal=1.0)
    res = cv.run_experiment('boot', lr_fit_predict, X, meta, save=False, n_boot=300)
    lo, hi = res.ci['accuracy']
    assert lo <= res.metrics['accuracy'] <= hi
    again = cv.person_bootstrap(res.predictions, n_boot=300)
    assert again['accuracy'] == res.ci['accuracy']


def test_paired_compare_and_sign_flip():
    X, meta = toy(signal=3.0)
    a = cv.run_experiment('a', lr_fit_predict, X, meta, save=False, n_boot=0)
    same = cv.paired_compare(a, a, n_boot=200)
    assert same['pooled_acc_diff'] == 0 and same['p_sign_flip_acc'] == pytest.approx(1.0)
    chance = a.predictions.assign(p_right=0.5)  # always "right": 50 % per person
    cmp = cv.paired_compare(a, chance, n_boot=200)
    assert cmp['pooled_acc_diff'] > 0.2
    assert cmp['persons_a_better'] >= 7
    assert cmp['p_sign_flip_acc'] < 0.05
    lo, hi = cmp['mean_person_acc_diff_ci95']
    assert 0 < lo <= cmp['mean_person_acc_diff'] <= hi
    # exact enumeration: 8 equal positive differences give the most extreme of the 2^8 sign patterns
    assert cv._sign_flip_p(np.ones(8)) == pytest.approx(2 / 2 ** 8)


# ───────────────────────────── finding 7: the k-attempt curve keeps its persons ─────────────────────────────


def _k_pred(small_person_trials: int = 3) -> pd.DataFrame:
    rows = []
    for p in range(4):
        n = small_person_trials if p == 3 else 6
        for k in range(n):
            for y in (0, 1):
                # 4 of 6 trials right, 2 wrong but close to 0.5: averaging k >= 3 fixes most draws
                good = k < 4
                pr = (0.8 if good else 0.45) if y == 1 else (0.2 if good else 0.55)
                rows.append({'person_id': f'P{p}', 'y': y, 'p_right': pr})
    return pd.DataFrame(rows)


def test_k_attempt_curve():
    kc = cv.k_attempt_curve(_k_pred(6), ks=(1, 3, 5), n_boot=100)
    acc = dict(zip(kc['k'], kc['accuracy']))
    assert acc[1] == pytest.approx(4 / 6)
    assert acc[5] == pytest.approx(1.0)
    assert acc[3] > acc[1]


def test_k_attempt_curve_uses_the_same_persons_for_every_k():
    """The verifier's finding: P04 (4 trials per hand) dropped out at k = 5 only."""
    kc = cv.k_attempt_curve(_k_pred(3), ks=(1, 2, 3, 4, 5), n_boot=50)
    assert kc['n_persons'].tolist() == [3] * 5 and set(kc['persons_left_out']) == {'P3'}
    old = cv.k_attempt_curve(_k_pred(3), ks=(1, 2, 3, 4, 5), n_boot=50, common_persons=False)
    assert old['n_persons'].tolist() == [4, 4, 4, 3, 3]


def test_reference_k_curve_reports_its_persons(primary):
    from stage2 import config as C
    if not (C.RES_DIR / 'example_yaw_logreg.json').exists():
        pytest.skip('no reference result (python -m stage2.scripts.reference)')
    res = cv.load_result('example_yaw_logreg')
    assert res.k_curve['n_persons'].nunique() == 1 and res.k_curve['persons_left_out'].iloc[0] == 'P04'


# ───────────────────────────── natural use and results ─────────────────────────────


def test_extra_rows_are_scored_out_of_fold_and_kept_apart():
    X, meta = toy()
    Xe, me = toy(per=2, seed=9)
    me['trial_id'] = 'x_' + me['trial_id']
    res = cv.run_experiment('extra', lr_fit_predict, X, meta, save=False, n_boot=0, extra=(Xe, me))
    assert len(res.extra_predictions) == len(me)
    assert res.extra_metrics['n'] == len(me)
    assert not set(res.extra_predictions['trial_id']) & set(res.predictions['trial_id'])


def test_result_round_trip(tmp_path):
    X, meta = toy()
    res = cv.run_experiment('rt', lr_fit_predict, X, meta, save=True, n_boot=50, out_dir=tmp_path,
                            registry=tmp_path / 'registry.csv')
    back = cv.ExperimentResult.load(tmp_path / 'rt.json')
    assert back.metrics['accuracy'] == pytest.approx(res.metrics['accuracy'])
    assert list(back.predictions.columns)[:6] == ['trial_id', 'trial_key', 'person_id', 'y', 'p_right', 'fold']
    reg = pd.read_csv(tmp_path / 'registry.csv')
    assert reg['name'].tolist() == ['rt']
    cv.run_experiment('rt', lr_fit_predict, X, meta, save=True, n_boot=0, out_dir=tmp_path,
                      registry=tmp_path / 'registry.csv')
    assert pd.read_csv(tmp_path / 'registry.csv')['name'].tolist() == ['rt']  # replaced, not duplicated
    assert math.isfinite(back.summary_row()['accuracy'])
    assert back.info['code']['stage2_sha256'] and back.info['locked_test'].startswith('waived')


# ───────────────────────────── core maintenance 2026-10-28 ─────────────────────────────
# paired_compare vectorised, the sign-flip test by doubling, the numpy ECE: output-identical to the former code,
# kept below as private references; ties earn half credit; declared specifications of saved results.


def _old_sign_flip_p(d, *, exact_max=20, n_perm=100_000, seed=None):
    """The former enumeration of cv._sign_flip_p (reference)."""
    from stage2 import config as C
    seed = C.SEED if seed is None else seed
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return math.nan
    obs = abs(d.mean())
    if n <= exact_max:
        codes = np.arange(2 ** n, dtype=np.int64)
        total = np.zeros(codes.size)
        for i in range(n):
            sign = 1.0 - 2.0 * ((codes >> i) & 1)
            total += sign * d[i]
        stats = np.abs(total / n)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        stats = np.abs(signs @ d / n)
    return float(np.mean(stats >= obs - 1e-12))


def _old_paired_compare(a, b, *, n_boot=2000, seed=None):
    """The former per-resample loop of cv.paired_compare (reference; its forced-choice accuracy rule equals the
    half-credit rule on predictions without ties)."""
    from stage2 import config as C
    seed = C.SEED if seed is None else seed
    pa, pb = a.predictions, b.predictions
    j = pa[['trial_id', 'person_id', 'y', 'p_right']].merge(pb[['trial_id', 'p_right']], on='trial_id',
                                                             suffixes=('_a', '_b'))
    y = j['y'].to_numpy().astype(int)
    ca = ((j['p_right_a'].to_numpy() >= 0.5).astype(int) == y).astype(float)
    cb = ((j['p_right_b'].to_numpy() >= 0.5).astype(int) == y).astype(float)

    def ll(p):
        pc = np.clip(p, cv._EPS, 1 - cv._EPS)
        return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    la, lb = ll(j['p_right_a'].to_numpy()), ll(j['p_right_b'].to_numpy())
    df = pd.DataFrame({'person_id': j['person_id'].astype(str), 'acc_a': ca, 'acc_b': cb, 'll_a': la, 'll_b': lb})
    pp = df.groupby('person_id').mean()
    pp['d_acc'] = pp['acc_a'] - pp['acc_b']
    pp['d_ll'] = pp['ll_a'] - pp['ll_b']
    persons = df['person_id'].to_numpy()
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boot_pool, boot_person, boot_ll = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([rows[u] for u in pick])
        boot_pool.append(ca[sel].mean() - cb[sel].mean())
        boot_person.append(pp.loc[pick, 'd_acc'].mean())
        boot_ll.append(pp.loc[pick, 'd_ll'].mean())
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))  # noqa: E731
    return {
        'n_trials': int(len(j)), 'n_persons': int(uniq.size), 'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
        'pooled_acc_diff': float(ca.mean() - cb.mean()), 'pooled_acc_diff_ci95': q(boot_pool),
        'mean_person_acc_diff': float(pp['d_acc'].mean()), 'mean_person_acc_diff_ci95': q(boot_person),
        'p_sign_flip_acc': _old_sign_flip_p(pp['d_acc'].to_numpy(), seed=seed),
        'mean_person_logloss_diff': float(pp['d_ll'].mean()), 'mean_person_logloss_diff_ci95': q(boot_ll),
        'p_sign_flip_logloss': _old_sign_flip_p(pp['d_ll'].to_numpy(), seed=seed),
        'persons_a_better': int((pp['d_acc'] > 0).sum()), 'persons_b_better': int((pp['d_acc'] < 0).sum()),
        'per_person': pp.reset_index(),
    }


#: saved result pairs without ties (P(right) = 0.5): the former and the current rule agree on them
SAVED_PAIRS = (('m1_lr_phys6_nested', 'example_yaw_logreg'), ('m2_select', 'm2_lr_full'),
               ('m3_cal_deep_platt', 'm2_ens_stack'), ('p1_gbm_base', 'p1_lr_base'))


def test_vectorised_paired_compare_equals_the_former_loop_on_saved_results():
    from stage2 import config as C
    pairs = [(a, b) for a, b in SAVED_PAIRS
             if (C.RES_DIR / f'{a}.json').exists() and (C.RES_DIR / f'{b}.json').exists()]
    if len(pairs) < 3:
        pytest.skip('fewer than 3 saved result pairs')
    for an, bn in pairs:
        a, b = cv.load_result(an), cv.load_result(bn)
        assert not cv.is_tie(a.predictions['p_right']).any() and not cv.is_tie(b.predictions['p_right']).any()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old, new = _old_paired_compare(a, b), cv.paired_compare(a, b)
        assert set(old) == set(new)
        for k, v in old.items():
            if k == 'per_person':
                pd.testing.assert_frame_equal(new[k], v, check_exact=False, atol=1e-12, rtol=0)
            else:
                assert new[k] == pytest.approx(v, abs=1e-12), (an, bn, k)        # the p-values included
        assert new['p_sign_flip_acc'] == old['p_sign_flip_acc']
        assert new['p_sign_flip_logloss'] == old['p_sign_flip_logloss']


def test_sign_flip_by_doubling_equals_the_enumeration():
    rng = np.random.default_rng(3)
    for t in range(200):
        n = int(rng.integers(1, 15))
        d = rng.normal(rng.normal(0, 0.05), rng.choice([1e-3, 0.05, 1.0]), n)
        if t % 3 == 0:
            d = np.round(d, 2)                       # equal |differences|: many statistics at the threshold
        if t % 7 == 0:
            d[: n // 2] = 0.0
        assert cv._sign_flip_p(d) == _old_sign_flip_p(d)
    d = rng.normal(0.01, 0.05, 20)                   # 20 persons, the exact limit
    assert cv._sign_flip_p(d) == _old_sign_flip_p(d)
    assert math.isnan(cv._sign_flip_p(np.array([np.nan])))


def test_numpy_ece_equals_the_reliability_formula():
    rng = np.random.default_rng(4)
    for t in range(60):
        n = int(rng.integers(5, 400))
        y = rng.integers(0, 2, n)
        p = rng.beta(2, 2, n)
        if t % 2:
            p = np.round(p, 1)                       # ties share a bin
        for strategy in ('quantile', 'uniform'):
            r = cv.reliability(y, p, strategy=strategy)
            ref = float(np.sum(r['n'] / r['n'].sum() * np.abs(r['y_mean'] - r['p_mean'])))
            assert cv.ece(y, p, strategy=strategy) == ref
    assert math.isnan(cv.ece(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        cv.ece(np.array([1, 0]), np.array([0.2, 0.7]), strategy='nope')


def test_a_tie_earns_half_credit_everywhere():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.5, 0.5, 0.2, 0.5 + 1e-16, 0.6])     # two exact ties, one tie up to rounding
    assert cv.is_tie(p).tolist() == [False, True, True, False, True, False]
    assert cv.credit(y, p).tolist() == [1.0, 0.5, 0.5, 1.0, 0.5, 0.0]
    m = cv.metrics(y, p)
    assert m['accuracy'] == pytest.approx(3.5 / 6)
    assert m['balanced_accuracy'] == pytest.approx((2.0 / 3 + 1.5 / 3) / 2)   # right: 1, .5, .5; left: .5, 1, 0
    assert m['coverage'] == pytest.approx(3 / 6) and m['accuracy_answered'] == pytest.approx(2 / 3)
    # a model that never abstains has no coverage keys; without ties the former rule is unchanged
    m2 = cv.metrics(y, np.array([0.9, 0.7, 0.4, 0.2, 0.6, 0.6]))
    assert 'coverage' not in m2 and m2['accuracy'] == pytest.approx(5 / 6)
    # the all-right natural-use failure: a constant 0.5 is 50 %, not 100 %
    nat = cv.metrics(np.ones(20, int), np.full(20, 0.5))
    assert nat['accuracy'] == 0.5 and nat['coverage'] == 0.0 and math.isnan(nat['accuracy_answered'])
    pp = cv.per_person(pd.DataFrame({'person_id': ['A'] * 4, 'y': [1, 1, 0, 0], 'p_right': [0.5, 0.8, 0.5, 0.7]}))
    assert pp.loc[0, 'accuracy'] == pytest.approx(0.5) and pp.loc[0, 'acc_right'] == pytest.approx(0.75)
    assert pp.loc[0, 'acc_left'] == pytest.approx(0.25)


def test_k_attempt_curve_gives_half_credit_to_a_mean_of_one_half():
    rows = ([{'person_id': f'P{i}', 'y': 1, 'p_right': pr} for i in range(3) for pr in (0.25, 0.75) * 3]
            + [{'person_id': f'P{i}', 'y': 0, 'p_right': 0.2} for i in range(3) for _ in range(6)])
    kc = cv.k_attempt_curve(pd.DataFrame(rows), ks=(1, 2), n_boot=20)
    acc = dict(zip(kc['k'], kc['accuracy']))
    assert acc[1] == pytest.approx((0.5 + 1.0) / 2)  # right hand: half of the attempts wrong; left: all right
    # right hand, k = 2: 9 of the 15 pairs average to 0.5 exactly (half credit), 3 are both right, 3 both wrong;
    # the former rule counted the 9 ties as right: (12 / 15 + 1) / 2 = 0.9
    assert acc[2] == pytest.approx(((9 * 0.5 + 3) / 15 + 1.0) / 2)


def test_paired_compare_scores_ties_with_half_credit():
    X, meta = toy(signal=3.0)
    a = cv.run_experiment('a', lr_fit_predict, X, meta, save=False, n_boot=0)
    half = a.predictions.assign(p_right=0.5)
    c = cv.paired_compare(a, half, n_boot=50)
    assert c['acc_b'] == 0.5
    assert c['acc_a'] == pytest.approx(np.mean(cv.credit(a.predictions['y'], a.predictions['p_right'])))
    assert c['pooled_acc_diff'] == pytest.approx(c['acc_a'] - 0.5)


def test_result_spec_is_read_from_the_saved_fields_and_fingerprinted(tmp_path):
    import json
    X, meta = toy()
    res = cv.run_experiment('spec', lr_fit_predict, X, meta, save=True, n_boot=0, out_dir=tmp_path, registry=None,
                            notes='pre-specified', n_configs_tried=3, params={'C': 1.0}, spec={'code': 'v2'})
    saved = json.loads((tmp_path / 'spec.json').read_text(encoding='utf-8'))
    spec = cv.result_spec(saved)
    assert spec == cv.result_spec(res) and set(cv.SPEC_KEYS) | {'code'} == set(spec)
    assert saved['info']['spec_sha256'] == cv.spec_fingerprint(spec)
    assert saved['info']['scoring'] == cv.SCORING and saved['info']['spec'] == {'code': 'v2'}
    assert cv.spec_matches(spec, {'n_configs_tried': 3, 'notes': 'pre-specified', 'params': {'C': 1.0}})
    assert cv.spec_matches(spec, {'code': 'v2', 'sets': ('primary',)})      # tuples compare in their saved form
    assert not cv.spec_matches(spec, {'n_configs_tried': 1})
    assert not cv.spec_matches(spec, {'code': 'v3'}) and not cv.spec_matches(spec, {'features': 'x'})


def test_every_saved_result_has_a_declared_spec():
    """The fingerprint of an existing result comes from fields every result stores: nothing is rewritten."""
    import json
    from stage2 import config as C
    n = 0
    for f in sorted(C.RES_DIR.glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        if not (isinstance(d, dict) and 'predictions' in d and 'info' in d):
            continue
        spec = cv.result_spec(d)
        assert spec['n_configs_tried'] is not None and spec['seed'] is not None and spec['sets'], f.name
        assert cv.spec_matches(spec, spec) and len(cv.spec_fingerprint(spec)) == 64
        n += 1
    if not n:
        pytest.skip('no saved results')


def test_rescore_leaves_tie_free_results_alone_and_is_idempotent():
    """stage2.scripts.rescore: a result without ties reproduces from its saved predictions exactly; after the
    re-score every result follows the half-credit rule (nothing left to change)."""
    from stage2 import config as C
    from stage2.scripts import rescore
    names = [n for n in ('example_yaw_logreg', 'm2_select', 'm4_cal_nested', 'm2_lr_only_wavelet')
             if (C.RES_DIR / f'{n}.json').exists()]
    if len(names) < 2:
        pytest.skip('saved results missing')
    gates = rescore.gate_map()
    for n in names:
        assert rescore.rescore_file(C.RES_DIR / f'{n}.json', gates) is None, n
    if 'm2_lr_only_wavelet' in names:                # a constant 0.5: half credit, no coverage
        r = cv.load_result('m2_lr_only_wavelet')
        assert r.metrics['accuracy'] == 0.5 and r.metrics['coverage'] == 0.0 and r.extra_metrics['accuracy'] == 0.5
        assert r.info['rescored']['rule'] == cv.SCORING
