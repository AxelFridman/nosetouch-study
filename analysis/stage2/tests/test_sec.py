"""Lane sec (secondary tasks): mirror invariance of the even features, the kinematic helpers, the multi-class loop and
its metrics, the person-level statistics, and the saved outputs (when computed). Fast: about 30 s."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2.guard import LeakageError
from stage2.lanes import sec_features as F
from stage2.lanes import sec_models as M
from stage2.standardize import CHANNELS, MIRROR_SIGN, mirror_channels

#: the gyro channels' mirror signs, as the cache defines them (checked against the physics below)
MIRROR_GYRO = np.asarray(MIRROR_SIGN, dtype=float)[[CHANNELS.index(f'gyro_{a}') for a in 'xyz']]


def test_the_gyro_mirror_signs_are_those_of_a_pseudovector():
    """The mirror S = diag(-1, 1, 1) maps an angular velocity (a pseudovector) to det(S) S w = (w_x, -w_y, -w_z)."""
    S = np.diag([-1.0, 1.0, 1.0])
    np.testing.assert_array_equal(MIRROR_GYRO, np.linalg.det(S) * np.diag(S))


def test_the_feature_constants_match_the_cache():
    from stage2.lanes import sec_compute as SC
    SC.check_constants()                           # raises when config.FS_HZ / FIXED_PRE_S and sec_features differ


def test_the_common_segmentation_parameters_ignore_the_condition():
    """One parameter set for every trial: the app's defaults with the longest movement cap of any condition; the
    premise of the check: the app's own per-speed parameters do differ (a stricter end for slow trials)."""
    from motionlab import conventions as CV
    from stage2.lanes import sec_reseg as RS
    p = RS.common_params()
    caps = [o['max_movement_ms'] for o in (*CV._SPEED_OVERRIDES.values(), *CV._VARIANT_OVERRIDES.values())
            if 'max_movement_ms' in o]
    assert p['max_movement_ms'] == max([CV.DEFAULT_SEGMENTATION_PARAMS['max_movement_ms'], *caps])
    assert {k: v for k, v in p.items() if k != 'max_movement_ms'} == {
        k: float(v) for k, v in CV.DEFAULT_SEGMENTATION_PARAMS.items() if k != 'max_movement_ms'}
    slow = CV.resolve_params(CV.DEFAULT_SEGMENTATION_PARAMS, {'speed': 'slow', 'variant': 'standard'})
    natural = CV.resolve_params(CV.DEFAULT_SEGMENTATION_PARAMS, {'speed': 'natural', 'variant': 'standard'})
    assert slow['end_gyro_dps'] < natural['end_gyro_dps'] and slow['end_sustain_ms'] > natural['end_sustain_ms']


def test_angular_features_on_given_bounds_equal_the_even_columns(sample):
    """The segmentation check computes its features with the functions of the headline set: on the oracle bounds
    they equal the EVEN columns."""
    from stage2.lanes import sec_reseg as RS
    d, gyro = sample
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    even = F.even_features(d.approach, dur, gyro)
    ang = RS.angular_table(d, d.trials['onset_ms'].to_numpy(float), d.trials['end_ms'].to_numpy(float))
    for c in RS.ANG_FEATURES:
        np.testing.assert_allclose(ang[c].to_numpy(), even[c].to_numpy(), rtol=1e-12, atol=1e-12, err_msg=c)


@pytest.fixture(scope='module')
def sample(primary):
    """Forty cached primary trials with their 60 Hz approach gyroscope."""
    d = primary.subset(np.arange(min(40, primary.n)))
    return d, F.gyro_profiles(d)


def test_even_features_are_identical_for_a_trial_and_its_mirror_image(sample):
    d, gyro = sample
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    a = F.even_features(d.approach, dur, gyro)
    b = F.even_features(mirror_channels(d.approach), dur, [g * MIRROR_GYRO for g in gyro])
    assert list(a.columns) == list(F.EVEN_FEATURES)
    for c in a.columns:                            # each feature, separately: bit for bit (IEEE negation is exact)
        np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy(), err_msg=c)


def test_grip_features_change_under_the_mirror(sample):
    d, _ = sample
    a = F.grip_features(d.approach, d.fixed, d.fixed_mask)
    b = F.grip_features(mirror_channels(d.approach), mirror_channels(d.fixed), d.fixed_mask)
    np.testing.assert_allclose(b['up_before_x'], -a['up_before_x'])      # the device x flips: the grip is chiral
    np.testing.assert_allclose(b['up_before_y'], a['up_before_y'])
    np.testing.assert_allclose(b['gyro_mean_y'], -a['gyro_mean_y'])


def test_even_feature_names_are_never_label_like():
    from stage2.cv import forbidden_name
    assert not [f for f in [*F.EVEN_FEATURES, *F.GRIP_FEATURES] if forbidden_name(f)]


def test_angular_speed_is_exactly_mirror_invariant():
    g = np.random.default_rng(0).normal(0, 50, (100, 3))
    np.testing.assert_array_equal(F.angular_speed(g), F.angular_speed(g * MIRROR_GYRO))


def test_strapdown_recovers_a_known_path():
    """A minimum-jerk reach of 0.4 m along a fixed direction, the phone turning about another axis at the same time:
    the integrated path length and the peak speed come back within 2 %."""
    T, dur = 64, 1.5
    t = np.linspace(0, 1, T)
    s = 0.4 * (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)                       # position (m)
    acc_w = 0.4 * (60 * t - 180 * t ** 2 + 120 * t ** 3) / dur ** 2          # world-frame acceleration along e
    vpk = 0.4 * 1.875 / dur
    e = np.array([0.6, 0.8, 0.0])
    rate = np.deg2rad(40.0)                                                  # rad/s about device z
    acc = np.zeros((1, 3, T))
    for k in range(T):                                                       # the device frame turns by rate * t
        th = rate * t[k] * dur
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        acc[0, :, k] = R.T @ (acc_w[k] * e)
    gyro = np.zeros((1, 3, T))
    gyro[0, 2] = 40.0
    sp = F.strapdown_speed(acc, gyro, np.array([dur / (T - 1)]))
    path = 0.5 * np.sum(sp[0, 1:] + sp[0, :-1]) * dur / (T - 1)
    assert abs(path - s[-1]) / s[-1] < 0.02
    assert abs(sp.max() - vpk) / vpk < 0.02


def test_smoothness_measures_rank_a_smooth_profile_above_a_jerky_one():
    t = np.linspace(0, 1.2, 73)
    smooth = 100 * np.sin(np.pi * t / 1.2) ** 2
    jerky = smooth * (1 + 0.3 * np.sin(2 * np.pi * 6 * t))
    assert F.sparc(smooth) > F.sparc(jerky)
    assert F.ldlj(smooth) > F.ldlj(jerky)
    assert F.angular_features(jerky)['ang_n_peaks'] > F.angular_features(smooth)['ang_n_peaks']


def test_multiclass_metrics_match_sklearn():
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    yhat = np.where(rng.random(300) < 0.6, y, rng.integers(0, 3, 300))
    np.testing.assert_array_equal(M.confusion(y, yhat, 3), confusion_matrix(y, yhat, labels=[0, 1, 2]))
    assert math.isclose(M.balanced_accuracy(y, yhat, 3), balanced_accuracy_score(y, yhat))
    assert math.isclose(M.macro_f1(y, yhat, 3), f1_score(y, yhat, average='macro'))
    y2 = np.array([0, 0, 1, 1])                                             # a class absent from both
    assert math.isclose(M.macro_f1(y2, np.array([0, 1, 1, 1]), 3), f1_score(y2, [0, 1, 1, 1], average='macro'))


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = [0.01, 0.04, 0.03, 0.2, 0.005]
    np.testing.assert_allclose(M.holm(p), multipletests(p, method='holm')[1])
    assert np.isnan(M.holm([0.01, float('nan')])[1])


def test_within_person_permutation_keeps_every_persons_counts():
    rng = np.random.default_rng(2)
    person = np.repeat(['P01', 'P02', 'P03'], [10, 7, 12])
    y = rng.integers(0, 2, person.size)
    yp = M.permute_within(person, y, np.random.default_rng(3))
    for p in np.unique(person):
        assert y[person == p].sum() == yp[person == p].sum()
    assert not np.array_equal(y, yp)


def _toy(n_persons: int = 6, n: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    person = np.repeat([f'P{i + 1:02d}' for i in range(n_persons)], n)
    y = rng.integers(0, 3, person.size)
    X = rng.normal(0, 1, (person.size, 4)) + y[:, None] * 0.8
    meta = pd.DataFrame({'person_id': person, 'y_right': rng.integers(0, 2, person.size)})
    return X, y, meta


def test_multiclass_loso_never_shows_the_test_person_to_the_fit():
    X, y, meta = _toy()
    seen: list[set[str]] = []

    def fit(Xtr, ytr, gtr, seed):
        seen.append(set(gtr.tolist()))
        return M.lr_fitter(3)(Xtr, ytr, gtr, seed)

    out = M.loso_multiclass(X, y, meta, 3, fit)
    persons = sorted(set(meta['person_id']))
    assert len(seen) == len(persons) and out['n_folds'] == len(persons)
    for s, p in zip(seen, persons):
        assert p not in s and len(s) == len(persons) - 1
    assert np.isfinite(out['proba']).all() and np.allclose(out['proba'].sum(axis=1), 1)
    assert set(out['leak_check'].values()) == {'passed'}
    assert M.balanced_accuracy(y, out['proba'].argmax(axis=1), 3) > 0.5    # the toy signal is learnt


def test_multiclass_loso_detects_a_batch_dependent_prediction():
    X, y, meta = _toy(n_persons=4, n=20)

    class BatchScaled:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, Xt):
            z = (Xt[:, 0] - Xt[:, 0].mean()) / (Xt[:, 0].std() + 1e-9)     # uses the other test rows
            P = np.stack([np.exp(-z), np.ones_like(z), np.exp(z)], axis=1)
            return P / P.sum(axis=1, keepdims=True)

    with pytest.raises(LeakageError):
        M.loso_multiclass(X, y, meta, 3, lambda *a: (BatchScaled(), {}))


def test_bootstrap_cis_cover_the_estimates_and_use_persons():
    X, y, meta = _toy(n_persons=8, n=15, seed=4)
    out = M.loso_multiclass(X, y, meta, 3, M.lr_fitter(3))
    person = meta['person_id'].to_numpy()
    ci = M.bootstrap_multiclass(person, y, out['proba'], ('a', 'b', 'c'), n_boot=200)
    m = M.multiclass_metrics(y, out['proba'], ('a', 'b', 'c'))
    for k in ('balanced_accuracy', 'macro_f1', 'accuracy'):
        assert ci[k][0] <= m[k] <= ci[k][1], k


def test_task_labels_and_dominance_labels(primary):
    from stage2.lanes import sec_compute as SC
    y = SC.task_labels(primary.trials, 'speed')
    assert set(np.unique(y)) <= {0, 1, 2}
    g = SC.task_labels(primary.trials, 'grip')
    assert (np.asarray(SC.GRIP_CLASSES)[g] == primary.trials['orientation'].str.replace('_toward', '')).all()
    with pytest.raises(ValueError):
        SC.task_labels(primary.trials.assign(speed='sprint'), 'speed')
    dom, rows = SC.dominant_labels(primary)
    hs = primary.persons.set_index('person_id')['handedness_self']
    t = primary.trials
    exp = (t['hand'] == t['person_id'].map(hs)).astype(int).to_numpy()
    assert rows.all() and np.array_equal(dom, exp)


def _outputs_exist() -> bool:
    return (C.RES_DIR / 'sec_numbers.json').exists() and (C.OUT_DIR / 'sec_SUMMARY.md').exists()


@pytest.mark.skipif(not _outputs_exist(), reason='lane sec outputs not computed yet')
def test_outputs_follow_the_contract():
    from stage2.lanes import sec_compute as SC
    from stage2.lanes import sec_figures as SF
    from stage2.pipeline import Lane
    L = Lane('sec')
    for name in SC.MULTICLASS:
        r = SC.load_multiclass(name)
        assert r['leak_check'].startswith('passed') and 'trial_predictions' in r and 'predictions' not in r
    assert json.loads(L.finalists_path.read_text(encoding='utf-8')) == []
    caps = json.loads((L.fig_dir / 'captions.json').read_text(encoding='utf-8'))
    assert sorted(caps) == sorted(SF.FIGURE_FUNCTIONS)
    for k in SF.FIGURE_FUNCTIONS:
        assert (L.fig_dir / f'{k}.pdf').exists() and (L.fig_dir / f'{k}.png').exists()
        assert (L.tab_dir / f'{k}.csv').exists()
    res = json.loads((C.RES_DIR / f'{SC.DOMINANCE}.json').read_text(encoding='utf-8'))
    assert res['leak_check'].startswith('passed')
    # the app's detector re-run with the stored parameters reproduces every stored bound (the port is exact), the
    # bounds table carries no app identifier, and every even feature equals that of the trial's mirror image
    rs = L.read_json('sec_reseg')
    assert rs['n_reproduced'] == rs['n_trials']
    bounds = pd.read_csv(L.tab_dir / f'{SC.RESEG_TABLE}.csv')
    assert 'trial_id' not in bounds.columns and int((bounds['set'] == 'primary').sum()) == rs['n_trials']
    # re-standardising on the ORACLE bounds reproduces the cached arrays exactly (the label-blind arrays come from the
    # same code path)
    chk = rs['restandardize_check']['primary']
    assert chk['n_checked'] == rs['n_trials'] and chk['approach_max_abs_diff'] == 0.0 and chk['pre_mask_mismatches'] == 0
    mc = L.read_json('sec_mirror_check')
    assert mc['n_identical_trials'] == mc['n_trials'] and mc['max_abs_diff_all'] == 0.0
    # the two segmentation experiments score the same trials
    a, b = (SC.load_multiclass(x) for x in SC.SEGMENTATION_PAIR)
    assert sorted(r['trial_key'] for r in a['trial_predictions']) == sorted(r['trial_key'] for r in b['trial_predictions'])


@pytest.mark.skipif(not _outputs_exist(), reason='lane sec outputs not computed yet')
def test_label_blind_results_go_stale_when_the_resegmentation_changes(monkeypatch):
    """Every experiment on the re-segmentation records the hashes of the saved bounds and arrays; a changed bounds
    table makes all of them non-current (so compute re-runs them)."""
    from stage2 import pipeline as PL
    from stage2.lanes import sec_compute as SC
    names = [m for m in SC.MULTICLASS if SC.uses_reseg(m)]
    want = {f'reports/stage2/{f}' for f in SC.RESEG_FILES}
    for m in names:
        saved = json.loads(SC.result_file(m).read_text(encoding='utf-8'))
        assert set(saved.get('provenance_inputs') or {}) == want, m
        assert PL.file_current(SC.result_file(m)), m
    real = PL._sha256_or_none
    monkeypatch.setattr(PL, '_sha256_or_none',
                        lambda p: 'changed' if str(p).replace('\\', '/').endswith(f'{SC.RESEG_TABLE}.csv') else real(p))
    assert not any(PL.file_current(SC.result_file(m)) for m in names)


@pytest.mark.skipif(not _outputs_exist(), reason='lane sec outputs not computed yet')
def test_every_number_of_the_summary_and_the_figure_texts_is_traced():
    """Token comparison: every numeric token of sec_SUMMARY.md, the captions and the drawn titles / annotations (not
    the tick labels: strings that are a bare number) is a formatted entry of results/sec_numbers.json."""
    from stage2.lanes import sec_report as SR
    from stage2.pipeline import Lane
    L = Lane('sec')
    n = SR.read_numbers()
    assert not SR.untraced_numbers(L.summary_path.read_text(encoding='utf-8'), n)
    caps = json.loads((L.fig_dir / 'captions.json').read_text(encoding='utf-8'))
    texts = json.loads((L.fig_dir / 'figure_texts.json').read_text(encoding='utf-8'))
    import re
    bare = re.compile(r'^[\s−+\-]*\d+(?:\.\d+)?\s*%?$')
    tick = re.compile(r'^\$\\mathdefault\{(.*)\}\$$')               # matplotlib's mathtext-wrapped tick labels
    for k, cap in caps.items():
        drawn = [s for s in texts[k] if not bare.match(tick.sub(r'\1', s))]
        bad = SR.untraced_numbers(' '.join([cap['caption'], cap['decision'], *drawn]), n)
        assert not bad, (k, bad)
