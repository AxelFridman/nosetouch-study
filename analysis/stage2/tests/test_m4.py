"""Fast tests of lane m4 (cue-free production pipeline): mirror and screen-flip invariance of the segment features,
the vectorised labelling, causality (one code path online and offline), determinism and label-free fitting, memo
isolation between prepared data sets, exact antisymmetry of the whole pipeline and of its calibrators, the rest-pose
grip check (causal, mirror-even, flip-odd; applied by predict_stream), half-credit scoring, the stopping rule, the
synthetic sessions, the finalist's masking of test windows, and figures redrawn from saved files with fitting
forbidden."""
from __future__ import annotations

import math
import shutil

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C

pytestmark = pytest.mark.skipif(not (C.CACHE_DIR / 'arrays.npz').exists(), reason='no Stage 2 cache')


@pytest.fixture(scope='module')
def data():
    from stage2.data import load_cache
    return load_cache()


@pytest.fixture(scope='module')
def small(data):
    """Four persons, one segmentation configuration: prepared data, training / test rows."""
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_pipeline as P
    persons = sorted(data.trials['person_id'].astype(str).unique())[:4]
    sub = data.subset(np.isin(data.trials['person_id'].astype(str).to_numpy(), persons))
    streams = [P.label_free_stream(sub.stream(i)) for i in range(sub.n)]
    grid = {'lo20_es250': E.SEG_GRID['lo20_es250']}
    prep = E.prepare(streams, sub.approach, sub.ret, sub.ret_ok, grid=grid, with_v0=False)
    g = sub.trials['person_id'].astype(str).to_numpy()
    tr, te = np.flatnonzero(g != persons[-1]), np.flatnonzero(g == persons[-1])
    return {'sub': sub, 'prep': prep, 'streams': streams, 'grid': grid, 'g': g, 'tr': tr, 'te': te}


def test_segment_features_are_mirror_even_and_grip_features_flip_invariant(data):
    from stage2.lanes import m4_detect as D
    from stage2.standardize import mirror_channels, screen_flip_channels
    for i in (0, 7, 33):
        st = data.stream(i)
        a, Wa = D.segment_table(st, D.DEFAULT)
        b, Wb = D.segment_table(D.mirror_stream(st), D.DEFAULT)
        c, Wc = D.segment_table(D.screen_flip_stream(st), D.DEFAULT)
        assert len(a) == len(b) == len(c) and len(a) > 0
        assert np.allclose(a[['onset_ms', 'end_ms']], b[['onset_ms', 'end_ms']])
        assert np.allclose(a[['onset_ms', 'end_ms']], c[['onset_ms', 'end_ms']])
        feats = list(D.ALL_FEATURES)
        assert np.allclose(a[feats].to_numpy(float), b[feats].to_numpy(float), atol=1e-6, equal_nan=True)
        # the production (grip) features are invariant to the screen flip; the raw screen-odd ones change sign
        grip = list(D.FEATURE_SETS['grip'])
        assert np.allclose(a[grip].to_numpy(float), c[grip].to_numpy(float), atol=1e-6, equal_nan=True)
        odd = list(D.SCREEN_ODD_FEATURES)
        assert np.allclose(a[odd].to_numpy(float), -c[odd].to_numpy(float), atol=1e-6, equal_nan=True)
        # the windows: mirrored (chiral channels flip) and screen-flipped (device x, z flip; gravity frame equal)
        assert np.allclose(mirror_channels(Wa), Wb, atol=1e-4, equal_nan=True)
        assert np.allclose(screen_flip_channels(Wa), Wc, atol=1e-4, equal_nan=True)


def test_vectorised_labels_equal_cuefree_classify():
    from stage2.cuefree import Detection, classify
    from stage2.lanes import m4_detect as D
    rng = np.random.default_rng(0)
    for _ in range(300):
        on = rng.uniform(0, 8000)
        end = on + rng.uniform(100, 3000)
        o_on = rng.uniform(1000, 5000)
        o_end = o_on + rng.uniform(300, 3000)
        cue = o_end + 250 if rng.random() < 0.9 else math.nan
        r_on = cue + rng.uniform(100, 1500) if rng.random() < 0.6 and math.isfinite(cue) else math.nan
        r_end = r_on + rng.uniform(300, 2000) if rng.random() < 0.8 and math.isfinite(r_on) else math.nan
        ref = classify(Detection(on, end, 0, 0, 0, 0, True), o_on, o_end, cue_ms=None if math.isnan(cue) else cue,
                       return_onset_ms=None if math.isnan(r_on) else r_on, return_end_ms=None if math.isnan(r_end) else r_end)
        oc, iou = D.classify_arrays([on], [end], [o_on], [o_end], [cue], [r_on], [r_end])
        assert oc[0] == ref['outcome'] and math.isclose(iou[0], ref['iou'], abs_tol=1e-12)


def test_windows_aligned_and_causal(data):
    """Every segment's features and window are the same when the stream stops at the segment's decision time
    (the online detector sees exactly what the offline evaluation uses)."""
    from stage2.lanes import m4_detect as D
    st = data.stream(5)
    full, W = D.segment_table(st, D.DEFAULT)
    assert W.shape == (len(full), 17, C.N_NORM)
    feats = list(D.ALL_FEATURES)
    checked = 0
    for j, r in full.iterrows():
        if not r['ended']:
            continue
        cut = {k: v[st['t'] <= r['decided_ms'] + 1e-6] for k, v in st.items()}
        part, Wp = D.segment_table(cut, D.DEFAULT)
        m = np.flatnonzero(np.isclose(part['onset_ms'], r['onset_ms']))
        assert m.size == 1
        k = int(m[0])
        assert math.isclose(part.loc[k, 'end_ms'], r['end_ms'])
        assert np.allclose(part.loc[k, feats].to_numpy(float), r[feats].to_numpy(float), atol=1e-6, equal_nan=True)
        assert np.allclose(Wp[k], W[j], atol=1e-5, equal_nan=True)
        checked += 1
    assert checked >= 1


def test_pipeline_fit_is_deterministic_and_label_free(small):
    from stage2.lanes import m4_e2e as E
    sub, prep, g, tr, te = small['sub'], small['prep'], small['g'], small['tr'], small['te']
    spec = E.PipelineSpec(seg='lo20_es250', features='robust', calibration='nested')
    m = sub.trials.iloc[tr].reset_index(drop=True)
    y = sub.y[tr]
    a = E.fit_pipeline(prep, spec, tr, y, g[tr], m, 7).choose(prep, te)
    E.clear_memos()
    b = E.fit_pipeline(prep, spec, tr, y, g[tr], m, 7).choose(prep, te)
    assert np.allclose(a['p_right'], b['p_right'])
    assert ((a['p_right'] >= 0) & (a['p_right'] <= 1)).all()
    # the held-out person's events never enter: scrambling them changes nothing
    m2 = sub.trials.copy()
    for c in ('onset_ms', 'end_ms', 'end_detected_ms', 'return_onset_ms', 'return_end_ms'):
        m2.loc[te, c] = np.nan
    E.clear_memos()
    c2 = E.fit_pipeline(prep, spec, tr, y, g[tr], m2.iloc[tr].reset_index(drop=True), 7).choose(prep, te)
    assert np.allclose(a['p_right'], c2['p_right'])


def test_memo_keys_on_the_prepared_data(small):
    """Two prepared data sets with the same spec, rows and seed must never share a fitted pipeline (the memo that
    once let the causal low-pass variant reuse the main streams' fit)."""
    from stage2.lanes import m4_e2e as E
    sub, prep, g, tr = small['sub'], small['prep'], small['g'], small['tr']
    other = [{**s, 'gyro': s['gyro'] * 1.3, 'f': s['f'] * np.array([1.0, 1.0, 1.0])} for s in small['streams']]
    prep2 = E.prepare(other, sub.approach, sub.ret, sub.ret_ok, grid=small['grid'], with_v0=False)
    assert prep.fingerprint() != prep2.fingerprint()
    spec = E.PipelineSpec(seg='lo20_es250', features='robust', inner_calibration=False)
    m = sub.trials.iloc[tr].reset_index(drop=True)
    a = E.get_pipeline(prep, spec, tr, sub.y[tr], g[tr], m, 3)
    b = E.get_pipeline(prep2, spec, tr, sub.y[tr], g[tr], m, 3)
    assert a is not b
    assert not np.allclose(a.segclf.m.coef_, b.segclf.m.coef_)
    assert E.get_pipeline(prep, spec, tr, sub.y[tr], g[tr], m, 3) is a     # same data: the memo is reused


def _scorer(small, features):
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    sub, prep, g, tr = small['sub'], small['prep'], small['g'], small['tr']
    spec = E.PipelineSpec(seg='lo20_es250', features=features, calibration='nested')
    fp = E.fit_pipeline(prep, spec, tr, sub.y[tr], g[tr], sub.trials.iloc[tr].reset_index(drop=True), 5)
    p = E.SEG_GRID['lo20_es250']

    def score(streams):
        tabs = [D.segment_table(s, p) for s in streams]
        tab = pd.concat([t.assign(stream=j) for j, (t, _) in enumerate(tabs)], ignore_index=True)
        W = np.concatenate([w for _, w in tabs])
        return fp.choose(prep, list(range(len(streams))), seg_override=(tab, W))
    return score


def test_production_pipeline_is_mirror_antisymmetric(small):
    """P(mirror image) = 1 - P(recording), exactly, for the production detector (robust features) with the nested
    (antisymmetric) calibrator; the grip gate's input (the start pose's up_z) is the same for both."""
    from stage2.lanes import m4_detect as D
    score = _scorer(small, 'robust')
    st = [small['streams'][int(i)] for i in small['te'][:12]]
    a, b = score(st), score([D.mirror_stream(s) for s in st])
    assert np.allclose(b['p_right'], 1.0 - a['p_right'], atol=1e-9)
    assert np.allclose(a['up_z0'], b['up_z0'], equal_nan=True)


def test_grip_alternative_is_screen_flip_invariant(small):
    """The grip-normalised alternative gives the same P for the screen-away twin; the production detector's chosen
    segment, when it is the same, has its start pose's up_z changed in sign (the first remedy's input)."""
    from stage2.lanes import m4_detect as D
    st = [small['streams'][int(i)] for i in small['te'][:12]]
    flipped = [D.screen_flip_stream(s) for s in st]
    grip = _scorer(small, 'grip')
    assert np.allclose(grip(flipped)['p_right'], grip(st)['p_right'], atol=1e-9)
    robust = _scorer(small, 'robust')
    a = robust(st)
    ok = a['status'] == 'ok'
    b = robust(flipped)
    same = ok & (b['onset_ms'].to_numpy() == a['onset_ms'].to_numpy())
    assert np.allclose(b.loc[same, 'up_z0'], -a.loc[same, 'up_z0'], atol=1e-9)


def test_rest_pose_is_causal_mirror_even_and_flip_odd(data):
    """The grip check's input reads only the samples before the first movement, does not see the hand (mirror) and
    changes sign with the screen-away grip."""
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_pipeline as P
    for i in (0, 7, 33):
        st = P.label_free_stream(data.stream(i))
        r = D.rest_up_z(st)
        assert math.isfinite(r)
        assert math.isclose(D.rest_up_z(D.mirror_stream(st)), r, abs_tol=1e-12)
        assert math.isclose(D.rest_up_z(D.screen_flip_stream(st)), -r, abs_tol=1e-12)
        first = D.find_segments(st['t'], st['gyro'])[0]
        cut = {k: v[st['t'] <= first.decided_ms] for k, v in st.items()}
        assert math.isclose(D.rest_up_z(cut), r, abs_tol=1e-12)


def test_predict_stream_applies_the_grip_check(small):
    """The runtime callable does not answer a stream whose rest pose fails the grip check (status 'grip', P = 0.5),
    and answers the same stream held with the screen towards the face as without the check."""
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_production as PR
    sub, prep, g, tr = small['sub'], small['prep'], small['g'], small['tr']
    spec = E.PipelineSpec(seg='lo20_es250', features='robust', calibration='nested')
    fp = E.fit_pipeline(prep, spec, tr, sub.y[tr], g[tr], sub.trials.iloc[tr].reset_index(drop=True), 5)
    model = PR.production_model(fp, agg=None, aggregation='mean')
    off = PR.production_model(fp, agg=None, aggregation='mean', grip_gate=None)
    stopped = 0
    for i in small['te'][:8]:
        st = small['streams'][int(i)]
        for s in (st, D.screen_flip_stream(st)):
            out = PR.predict_stream(s['t'], s['f'], s['acc'], s['gyro'], model)
            ref = PR.predict_stream(s['t'], s['f'], s['acc'], s['gyro'], off)
            assert 0.0 <= out['p_right'] <= 1.0
            if out['rest_up_z'] < model.grip_gate:
                assert out['status'] == 'grip' and not out['answered'] and out['p_right'] == 0.5 and out['grip_warning']
                stopped += 1
            else:
                assert out['p_right'] == ref['p_right'] and out['status'] == ref['status'] and not out['grip_warning']
    assert stopped >= 1


def test_calibrators_antisymmetric_where_declared():
    from stage2.lanes import m4_models as M
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 400)
    y = (rng.random(400) < M.sigmoid(0.6 * z)).astype(int)
    g = np.repeat([f'P{i:02d}' for i in range(8)], 50)
    cal = M.Temperature().fit(z, y)
    assert np.allclose(cal(-z), 1 - cal(z))
    assert 1.0 < cal.T < 3.5
    iso = M.IsotonicSym().fit(z, y)
    zz = np.linspace(-6, 6, 241)
    assert np.allclose(iso(-zz), 1 - iso(zz), atol=1e-12)
    assert np.all(np.diff(iso(zz)) >= -1e-12)
    nest = M.Nested().fit(z, y, groups=g)
    assert nest.chosen in M.NESTED_CANDIDATES and set(nest.cv_log_loss) == set(M.NESTED_CANDIDATES)
    assert np.allclose(nest(-zz), 1 - nest(zz), atol=1e-12)
    for name, cls in M.CALIBRATORS.items():
        if name in M.NESTED_CANDIDATES:
            assert cls.antisymmetric


def test_half_credit_scoring_and_holm():
    """An abstention (P = 0.5) is half a correct answer, never a right-hand answer on an all-right set."""
    from stage2.lanes import m4_stats as ST
    pred = pd.DataFrame({'person_id': [f'P{i:02d}' for i in range(6)], 'y': 1,
                         'p_right': [0.5, 0.5, 0.9, 0.2, 0.7, 0.5]})
    s = ST.scores(pred, n_boot=50)
    assert math.isclose(s['accuracy_half'], (0.5 + 0.5 + 1 + 0 + 1 + 0.5) / 6)
    assert math.isclose(s['accuracy_ge05'], 5 / 6)          # what the harness's P >= 0.5 rule would report
    assert math.isclose(s['coverage'], 0.5) and math.isclose(s['accuracy_answered'], 2 / 3)
    assert ST.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = pred.assign(trial_key=[f't{i}' for i in range(6)])
    d = ST.paired(a, a, n_boot=20)
    assert d['identical'] and d['d_acc'] == 0 and d['p_acc'] == 1.0


def test_fast_ece_equals_the_harness_ece():
    from stage2 import cv
    from stage2.lanes import m4_stats as ST
    rng = np.random.default_rng(4)
    for n in (30, 200, 601):
        p = np.round(rng.random(n), 2)                       # ties, as abstentions and isotonic plateaus make
        p[: n // 5] = 0.5
        y = (rng.random(n) < p).astype(int)
        assert math.isclose(ST.ece_fast(y, p), cv.ece(y, p), abs_tol=1e-12)


def test_stopping_rule():
    from stage2.lanes import m4_models as M
    from stage2.lanes import m4_online as O
    agg = M.AggregateScale()
    agg.s = {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0, 5: 2.0}
    att = pd.DataFrame({'decided_ms': [1000.0, 3000.0, 5000.0], 'z': [0.5, 1.2, 2.0]})
    r = O.stopping_rule(att, agg, p_stop=0.9, k_max=5)
    p2 = float(M.sigmoid(1.5 * np.mean([0.5, 1.2])))
    assert r['n_used'] == 3 and r['stopped'] == 'confident' and r['decided_ms'] == 5000.0 and p2 < 0.9
    r0 = O.stopping_rule(att.iloc[:0], agg)
    assert r0['stopped'] == 'no_attempt' and r0['p_right'] == 0.5 and not r0['answered']
    weak = pd.DataFrame({'decided_ms': [1.0, 2.0], 'z': [0.1, -0.05]})
    rw = O.stopping_rule(weak, agg, p_stop=0.9, k_max=2, p_answer=0.6)
    assert rw['stopped'] == 'k_max' and not rw['answered']


def test_sessions_one_person_one_hand(data):
    from stage2.lanes import m4_config as CFG
    from stage2.lanes import m4_sessions as S
    person = sorted(data.trials['person_id'].astype(str).unique())[1]
    sess = S.build_sessions(data.trials, data.stream, n_attempts=CFG.SESSION_RECORDINGS, keep_s=0.5,
                            persons=[person], stride=CFG.SESSION_STRIDE)
    assert sess and all(s.person_id == person for s in sess)
    for s in sess:
        rows = s.events['row'].to_numpy(int)
        assert set(data.trials['person_id'].iloc[rows]) == {person}
        assert set(data.trials['y_right'].iloc[rows].astype(int)) == {s.y}
        assert len(rows) == len(set(rows.tolist()))
        t = s.stream['t']
        assert np.allclose(np.diff(t), 1000.0 / C.FS_HZ, atol=1e-3)
        assert (s.events['onset_ms'] >= t[0]).all() and (s.events['end_ms'] <= t[-1]).all()


def test_session_scale_recovers_correlated_and_independent_attempts():
    """s_k = s_1 k^b fitted on session prefixes: b ~ 0 when a session's attempts are copies of one piece of
    evidence, b ~ 1 when they are independent log-likelihood ratios (s_k = k, summing)."""
    from stage2.lanes import m4_aggregate as A
    from stage2.lanes import m4_models as M
    rng = np.random.default_rng(3)
    n, k = 3000, 5
    z = rng.normal(0, 2, n)
    y = (rng.random(n) < M.sigmoid(z)).astype(int)
    corr = M.AggregateScale().fit_sessions([np.full(k, zi) for zi in z], y)
    assert corr.b < 0.15 and abs(corr.s1 - 1) < 0.15
    y2 = rng.integers(0, 2, n)
    seqs = [(2 * yi - 1) * 1.0 + rng.normal(0, math.sqrt(2.0), k) for yi in y2]   # LLR of one attempt = z
    ind = M.AggregateScale().fit_sessions(seqs, y2)
    assert ind.b > 0.85 and abs(ind.s1 - 1) < 0.15
    assert all(ind.s[j] <= ind.s[j + 1] for j in range(1, k))
    att = pd.DataFrame([{'person_id': f'P{i % 7:02d}', 'session': i, 'order': j, 'y': int(y2[i]), 'z': float(seqs[i][j])}
                        for i in range(200) for j in range(k)]).sample(frac=1.0, random_state=0)
    a2 = A.fit_session_scale(att)
    ref = M.AggregateScale().fit_sessions([seqs[i] for i in range(200)], y2[:200])
    assert np.allclose([a2.s[j] for j in a2.ks], [ref.s[j] for j in ref.ks])


def test_report_tables_never_overwrite_raw_files():
    """A report table must not take the name of a raw file the compute step reads back (it once did:
    m4_sessions_k)."""
    from stage2.lanes import m4_pipeline as P
    from stage2.lanes import m4_report as R
    assert not set(P.RAW_TABLES) & set(P.REPORT_TABLES)
    with pytest.raises(ValueError):
        R._export(pd.DataFrame({'a': [1]}), 'm4_sessions_krows')


def test_configuration_lives_in_one_place():
    from stage2.lanes import m4_aggregate as A
    from stage2.lanes import m4_config as CFG
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_pipeline as P
    from stage2.lanes import m4_production as PR
    assert A.SESSION_RECORDINGS == CFG.SESSION_RECORDINGS and P.SESSION_RECORDINGS == CFG.SESSION_RECORDINGS
    assert P.K_MAX == CFG.K_MAX and PR.ProductionModel.__dataclass_fields__['k_max'].default == CFG.K_MAX
    assert PR.ProductionModel.__dataclass_fields__['p_stop'].default == CFG.P_STOP
    assert E.PipelineSpec().pair_returns == CFG.PRODUCTION_PAIRING
    assert P.n_configs(P.PRODUCTION) == len(P.CUE_FREE_COMPARED) and P.PRODUCTION in P.CUE_FREE_COMPARED


def test_finalist_masks_the_test_windows(small, monkeypatch):
    """The finalist's fit_predict never lets the test items' event-cut windows reach the fit."""
    from types import SimpleNamespace
    from stage2.lanes import m4_finalist as F
    seen = {}

    def capture(items):
        seen['items'] = items
        raise RuntimeError('captured')

    monkeypatch.setattr(F, '_tables', capture)
    sub, tr, te = small['sub'], small['tr'], small['te']
    X = F.build_x(sub)
    fold = SimpleNamespace(X_train=[X[i] for i in tr], X_test=[X[i] for i in te], y_train=sub.y[tr],
                           groups_train=small['g'][tr], meta_train=sub.trials.iloc[tr], seed=1)
    with pytest.raises(RuntimeError, match='captured'):
        F.factory(spec={'features': 'robust', 'seg': 'nested'})(fold)
    items = seen['items']
    assert all(np.isfinite(it['oracle']).any() for it in items[:len(tr)])
    assert all(np.isnan(it['oracle']).all() and np.isnan(it['ret']).all() and not it['ret_ok'] for it in items[len(tr):])
    assert all('phase' not in it['stream'] for it in items)


def test_figures_redraw_from_saved_files_without_fitting(tmp_path, monkeypatch):
    """Figures need only the saved files: copy them into a temporary output folder, forbid every fit and the
    cache, and redraw."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from stage2 import cv, data as DATA
    from stage2.lanes import m4_figures as F
    from stage2.pipeline import Lane
    L = Lane('m4')
    need = [L.tab_dir / f'{n}.json' for n in ('m4_detection', 'm4_scores', 'm4_stress', 'm4_calibration', 'm4_paired')]
    if not all(p.exists() for p in need):
        pytest.skip('m4 compute has not run')
    out = tmp_path / 'out'
    (out / 'tables' / 'm4').mkdir(parents=True)
    (out / 'results').mkdir(parents=True)
    for p in need:
        shutil.copy(p, out / 'tables' / 'm4' / p.name)
    for n in F.FIGURES['M4_02_accuracy_ladder']['names']:
        if L.result_path(n).exists():
            shutil.copy(L.result_path(n), out / 'results' / f'{n}.json')
    for attr, sub in (('OUT_DIR', ''), ('RES_DIR', 'results'), ('TAB_DIR', 'tables'), ('FIG_DIR', 'figures')):
        monkeypatch.setattr(C, attr, out / sub if sub else out)

    def forbidden(*a, **k):
        raise AssertionError('a figure must not fit or load the cache')

    for obj, name in ((LogisticRegression, 'fit'), (IsotonicRegression, 'fit'), (cv, 'run_experiment'),
                      (DATA, 'load_cache')):
        monkeypatch.setattr(obj, name, forbidden)
    F.draw(['M4_01_detection_outcomes', 'M4_02_accuracy_ladder', 'M4_12_screen_flip'])
    for n in ('M4_01_detection_outcomes', 'M4_02_accuracy_ladder', 'M4_12_screen_flip'):
        assert (out / 'figures' / 'm4' / f'{n}.png').exists()
        assert (out / 'tables' / 'm4' / f'{n}.csv').exists()
