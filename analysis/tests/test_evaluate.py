"""Evaluation never leaks a participant; metrics, bootstrap CIs, mirror consistency and the models."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from motionlab import evaluate as E
from motionlab import features, load_dataset, models
from motionlab.conventions import mirror_pair


@pytest.fixture(scope='module')
def table():
    """60 trials per participant, 12 participants, 4 orientations, platform per participant."""
    rng = np.random.default_rng(0)
    rows = []
    orients = ['upright_toward', 'upright_away', 'top_left_toward', 'top_right_toward']
    for p in range(12):
        plat = 'ios' if p % 2 else 'android'
        ehi = 'left' if p % 3 == 0 else 'right'
        hand = 'left' if p % 3 == 0 else 'right'  # protocol 2.0.0 profiles: self-reported handedness, no EHI
        for k in range(60):
            rows.append({'trial_id': f't{p}-{k}', 'participant_id': f'P{p}', 'orientation': orients[k % 4],
                         'platform': plat, 'ehi_class': ehi, 'handedness_self': hand, 'y_right': k % 2,
                         'x': rng.normal()})
    df = pd.DataFrame(rows)
    df.loc[(df['participant_id'] == 'P5') & (df.index % 7 == 0), 'platform'] = 'android'  # one person, 2 phones
    return df


@pytest.mark.parametrize('scheme', E.SCHEMES)
def test_no_participant_is_ever_in_train_and_test(table, scheme):
    g = table['participant_id'].to_numpy()
    splits = E.make_splits(scheme, g, orientations=table['orientation'], platforms=table['platform'],
                           strata=E.participant_strata(table), n_splits=5, seed=3)
    assert splits
    tested = np.zeros(len(table), dtype=int)
    for s in splits:
        assert not set(g[s.train]) & set(g[s.test]), s.label
        assert len(np.intersect1d(s.train, s.test)) == 0
        tested[s.test] += 1
    assert tested.max() == 1  # every trial is tested at most once
    if scheme in ('stratified_group_kfold', 'group_kfold', 'loso'):
        assert (tested == 1).all()


def test_orientation_scheme_holds_mirror_pairs_out_together(table):
    g = table['participant_id'].to_numpy()
    o = table['orientation'].to_numpy()
    for s in E.make_splits('orientation', g, orientations=o):
        test_pairs = {mirror_pair(x) for x in o[s.test]}
        assert len(test_pairs) == 1
        assert not test_pairs & {mirror_pair(x) for x in o[s.train]}, s.label
    unpaired = E.make_splits('orientation', g, orientations=o, pair_mirrors=False)
    assert len({s.label.split('|')[0] for s in unpaired}) == 4


def test_platform_scheme_drops_people_seen_on_the_test_platform(table):
    g = table['participant_id'].to_numpy()
    splits = E.make_splits('platform', g, platforms=table['platform'])
    assert {s.label for s in splits} == {'platform:android', 'platform:ios'}
    ios_split = next(s for s in splits if s.label == 'platform:ios')
    assert 'P5' not in set(g[ios_split.train])  # P5 also has android trials, but is in the iOS test set


def test_stratified_folds_spread_the_strata(table):
    g = table['participant_id'].to_numpy()
    strata = E.participant_strata(table)
    splits = E.make_splits('stratified_group_kfold', g, strata=strata, n_splits=3, seed=1)
    for s in splits:
        assert len(set(strata[s.test])) >= 2  # each test fold mixes strata
    again = E.make_splits('stratified_group_kfold', g, strata=strata, n_splits=3, seed=1)
    assert all((a.test == b.test).all() for a, b in zip(splits, again))  # seeded, reproducible


def test_default_strata_are_handedness_and_platform(table):
    """EHI is not collected from protocol 2.0.0 on: the CV stratifies on handedness_self x platform; the
    protocol 1.0.0 stratum (ehi_class x platform) stays available."""
    assert E.DEFAULT_STRATA == ('handedness_self', 'platform')
    assert set(E.participant_strata(table)) == {'left|android', 'left|ios', 'right|android', 'right|ios'}
    meta = pd.DataFrame({'handedness_self': ['right', 'ambidextrous', None], 'platform': ['ios', 'android', 'ios'],
                         'ehi_class': [None, None, 'left']})
    assert list(E.participant_strata(meta)) == ['right|ios', 'ambidextrous|android', 'unknown|ios']
    assert list(E.participant_strata(meta, ('ehi_class', 'platform'))) == ['unknown|ios', 'unknown|android', 'left|ios']
    assert list(E.participant_strata(meta.drop(columns=['handedness_self', 'platform']), E.DEFAULT_STRATA)) == ['all'] * 3
    assert {'hand_is_phone_hand', 'hand_source', 'ehi_class'} <= set(E.SUBGROUP_COLUMNS)


def test_leakage_is_detected():
    g = np.array(['a', 'a', 'b', 'b'])
    with pytest.raises(E.LeakageError):
        E.assert_subject_disjoint([E.Split(np.array([0, 2]), np.array([1, 3]), 'bad')], g)
    with pytest.raises(ValueError):
        E.make_splits('loso', np.array(['a', 'a']))


def test_metrics_and_bootstrap():
    m = E.binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3])
    assert m['accuracy'] == 1.0 and m['roc_auc'] == 1.0 and m['log_loss'] < 0.3
    one = E.binary_metrics([1, 1], [0.9, 0.8])
    assert math.isnan(one['roc_auc']) and one['accuracy'] == 1.0
    df = pd.DataFrame({'participant_id': np.repeat(list('abcdef'), 10), 'v': np.r_[np.ones(30), np.zeros(30)]})
    lo, hi = E.participant_bootstrap(df, lambda d: float(d['v'].mean()), n_boot=300, seed=1)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    y = np.array([0, 1] * 10)
    gr = np.repeat(['p', 'q'], 10)
    ys = E.shuffle_labels_within(y, gr, seed=2)
    for u in ('p', 'q'):
        assert ys[gr == u].sum() == y[gr == u].sum()


def test_mirror_consistency_metric():
    p = np.array([0.9, 0.2, 0.7])
    ok = E.mirror_consistency(p, 1 - p)
    assert ok['label_flip_rate'] == 1.0 and ok['mean_abs_dev'] == pytest.approx(0.0)
    bad = E.mirror_consistency(p, p)
    assert bad['label_flip_rate'] == 0.0


def test_yaw_sign_rule_learns_the_sign():
    X = pd.DataFrame({'chi_yaw_net_deg': [-30.0, -20.0, 25.0, 40.0, np.nan]})
    y = np.array([1, 1, 0, 0, 1])
    m = models.YawSignRule().fit(X, y)
    assert m.sign_ == -1.0
    assert list(m.predict(X.iloc[:4])) == [1, 1, 0, 0]
    assert models.p_right(m, X)[4] == pytest.approx(0.5)  # NaN -> undecided
    with pytest.raises(KeyError):
        models.get_model_factory('nope')
    assert set(models.DEFAULT_MODELS) <= set(models.MODEL_REGISTRY)


def test_cross_validation_on_synthetic_features(tiny_tree):
    ds = load_dataset(tiny_tree)
    keep, excl = E.analysis_set(ds.trials)
    assert excl['rule'].iloc[0].startswith('T1') and excl['remaining'].iloc[-1] == int(keep.sum())
    ids = list(ds.trials.loc[keep, 'trial_id'])
    F, Fm = features.extract_features(ds, ids, mirror='both')
    X, meta = E.modeling_table(F, ds.trials)
    assert len(X) == len(ids)
    factories = {name: models.get_model_factory(name) for name in ('chance', 'yaw_rule', 'logreg', 'logreg_chiral', 'hgb')}
    res = E.evaluate_models(factories, X, meta, scheme='stratified_group_kfold', n_splits=3, X_mirror=Fm.reindex(X.index),
                            n_boot=100)
    for name in ('yaw_rule', 'logreg', 'logreg_chiral'):
        r = res[name]
        assert r.metrics['accuracy'] > 0.9, name
        assert r.mirror['label_flip_rate'] > 0.9, name
        lo, hi = r.metrics_ci['accuracy']
        assert lo <= r.metrics['accuracy'] <= hi
        assert set(r.oof.index) == set(X.index)
    assert res['chance'].metrics['roc_auc'] == pytest.approx(0.5)
    # the antisymmetric (chiral, no intercept) model is exactly mirror-consistent
    assert res['logreg_chiral'].mirror['mean_abs_dev'] < 1e-6
    sub = E.subgroup_table(res['logreg'].oof, meta, n_boot=50)
    assert {'variable', 'level', 'error', 'error_lo', 'error_hi', 'n_participants'} <= set(sub.columns)
    assert set(sub['variable']) >= {'hand', 'platform', 'orientation', 'speed'}
    st = E.summary_table(res)
    assert {'accuracy_lo', 'accuracy_hi', 'mirror_label_flip_rate', 'participant_acc_median'} <= set(st.columns)
    # augmentation adds mirrored TRAINING trials only: predictions stay one per real trial
    aug = E.cross_validate(factories['logreg'], X, meta['y_right'].to_numpy(), meta['participant_id'].to_numpy(),
                           scheme='group_kfold', n_splits=3, X_mirror=Fm.reindex(X.index), augment_mirror=True)
    assert len(aug.oof) == len(X) and aug.metrics['accuracy'] > 0.9
