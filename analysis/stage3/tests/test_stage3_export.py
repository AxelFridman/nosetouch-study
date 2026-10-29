"""Stage 3: the exported models, ``reference_predict`` and the golden fixtures.

* ``reference_predict`` reproduces ``m4_production.predict_stream`` exactly on every fixture (it raises otherwise) and
  reproduces the stored reference outputs;
* the JSON-rebuilt models reproduce the in-memory fits bit for bit on every fixture window (needs the Stage 2 cache);
* the JSON files carry everything the documented formulas need (plain numpy from the JSON alone reproduces the
  reference within 1e-12), match the manifest, and the CNN file is lane final's export unchanged;
* the fixtures hold app-rounded raw samples starting at t = 0, and the live view is a prefix of the offline one.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from stage3 import export as EX
from stage3 import fixtures as FX
from stage3 import reference as REF
from stage3 import runtime as RT

SYN = sorted(FX.SYN_DIR.glob('*.json.gz'))
REAL = sorted(FX.REAL_DIR.glob('*.json.gz')) if FX.REAL_DIR.exists() else []
TOL = 1e-12


def _ids(paths: list[Path]) -> list[str]:
    return [p.name.split('.')[0] for p in paths]


def compare(a, b, path='', tol=TOL, out=None):
    """Recursive comparison: floats within tol (relative to max(1, |b|)), everything else equal. Returns the list
    of differences and records the largest float difference in out['max']."""
    out = out if out is not None else {'max': 0.0, 'diffs': []}
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            out['diffs'].append(f'{path}: keys {sorted(set(a) ^ set(b)) if isinstance(a, dict) else type(a)}')
            return out
        for k in b:
            compare(a[k], b[k], f'{path}.{k}', tol, out)
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            out['diffs'].append(f'{path}: length')
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]', tol, out)
    elif isinstance(b, float) and isinstance(a, (float, int)) and not isinstance(a, bool):
        d = abs(float(a) - b)
        out['max'] = max(out['max'], d)
        if d > tol * max(1.0, abs(b)):
            out['diffs'].append(f'{path}: {a!r} != {b!r}')
    elif a != b:
        out['diffs'].append(f'{path}: {a!r} != {b!r}')
    return out


# ───────────────────────────── reference_predict against predict_stream and the stored fixtures ─────────────────────


@pytest.mark.parametrize('path', SYN, ids=_ids(SYN))
def test_synthetic_fixture_reproduces_predict_stream_and_its_reference(path, json_models):
    doc = FX.read_case(path)
    ref = FX.clean(REF.reference_predict(doc['motion'], json_models, engine=doc['engine']))   # raises on a mismatch
    res = compare(ref, doc['reference'])
    assert not res['diffs'], res['diffs'][:5]


@pytest.mark.skipif(not REAL, reason='no local real fixtures (analysis/data/stage3_fixtures)')
@pytest.mark.parametrize('path', REAL, ids=_ids(REAL))
def test_real_fixture_reproduces_predict_stream_and_its_reference(path, json_models):
    doc = FX.read_case(path)
    ref = FX.clean(REF.reference_predict(doc['motion'], json_models, engine=doc['engine']))
    res = compare(ref, doc['reference'])
    assert not res['diffs'], res['diffs'][:5]


def test_reference_predict_raises_when_the_m4_view_differs(json_models):
    """The check is live: a decision that disagrees with predict_stream raises."""
    doc = FX.read_case(SYN[0])
    out = REF.reference_predict(doc['motion'], json_models, engine=doc['engine'], check=False)
    ps = out['m4_predict_stream']
    bad = {**out, 'decisions': {**out['decisions'], 'm4': {**out['decisions']['m4'], 'p_right': 0.123}}}
    from stage2.lanes import m4_production as PR
    from stage2 import standardize as S
    st = S.stream_from_record(REF.as_record(doc['motion']), REF.production_normalization(doc['engine']))
    raw = PR.predict_stream(st.t, st.f, st.acc, st.gyro, json_models.m4)
    assert raw['p_right'] == ps['p_right']
    with pytest.raises(AssertionError):
        REF._check_m4(bad, raw, json_models.m4.fitted.attempts(*_table(st, json_models)), json_models.m4, False)


def _table(st, models):
    from stage2.lanes import m4_detect as D
    s = {'t': st.t, 'f': st.f, 'acc': st.acc, 'gyro': st.gyro,
         'valid': np.isfinite(st.f).all(axis=1) & np.isfinite(st.gyro).all(axis=1)}
    return D.segment_table(s, models.m4.seg_params)


# ───────────────────────────── JSON models against the in-memory fits ─────────────────────────────


def test_json_models_reproduce_the_fitted_models_on_every_fixture(fitted_models, json_models):
    """Every model's z_raw, p and z on every fixture window, every segment probability and every decision: the
    models rebuilt from the JSON equal the in-memory fits exactly (and predict_stream gives the same dict)."""
    from stage2 import standardize as S
    from stage2.lanes import m4_production as PR
    n_win = 0
    worst = 0.0
    for path in SYN + REAL:
        doc = FX.read_case(path)
        a = FX.clean(REF.reference_predict(doc['motion'], fitted_models, engine=doc['engine']))
        b = FX.clean(REF.reference_predict(doc['motion'], json_models, engine=doc['engine']))
        a.pop('models_source')
        b.pop('models_source')
        res = compare(a, b, tol=0.0)
        assert not res['diffs'], (path.name, res['diffs'][:5])
        worst = max(worst, res['max'])
        n_win += sum(1 for s in b['segments'] if s['canonical_ok'])
        st = S.stream_from_record(REF.as_record(doc['motion']), REF.production_normalization(doc['engine']))
        pa = PR.predict_stream(st.t, st.f, st.acc, st.gyro, fitted_models.m4)
        pb = PR.predict_stream(st.t, st.f, st.acc, st.gyro, json_models.m4)
        assert json.dumps(pa, default=str) == json.dumps(pb, default=str)
    assert worst == 0.0 and n_win > 0


def test_phys6_and_yaw_wrappers_equal_the_fitted_closures():
    """The PHYS6 wrapper computes exactly what make_linear_select's own closure computes, and the yaw reference's
    float32 model (as evaluated) differs from the float64 evaluation the page uses by < 1e-6 in P."""
    from stage2 import config as C
    if not (C.CACHE_DIR / 'arrays.npz').exists():
        pytest.skip('no Stage 2 cache')
    from stage2.data import load_cache
    from stage2.lanes import m1_models as M1
    from stage3 import fit as F
    ph = F.fit_phys6()
    X = getattr(M1, 'build_x_treat')(load_cache())
    w = RT.phys6_fitted(ph)
    z_wrap = np.array([w.z_raw(X.iloc[[i]]) for i in range(0, len(X), 7)])
    Xa = X[ph['allcols']].to_numpy(float)
    z_clos = np.array([ph['zfun'](Xa[i:i + 1])[0] for i in range(0, len(X), 7)])      # row by row, as the page does
    assert np.array_equal(z_wrap, z_clos)
    z_batch = ph['zfun'](Xa)[::7]                                                      # BLAS blocking: ulp level
    assert np.max(np.abs(z_batch - z_clos)) < 1e-12
    y = F.fit_yaw_ref()
    d = load_cache()
    ref = RT.YawRef(y['model'], y['index'], y['step'])
    dp = [abs(o['p'] - o['p_f32']) for o in (ref.predict(d.approach[i]) for i in range(0, d.n, 5))]
    assert max(dp) < 1e-6
    p_fold = y['p_train_fold'][::5]
    assert np.allclose([ref.predict(d.approach[i])['p_f32'] for i in range(0, d.n, 5)], p_fold, atol=1e-6, rtol=0)


# ───────────────────────────── the JSON is complete: plain formulas reproduce the reference ─────────────────────────


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def _iso_sym(c, z):
    x, y = np.asarray(c['x_thresholds']), np.asarray(c['y_thresholds'])
    T = min(max(abs(z), c['x_min']), c['x_max'])
    if len(x) == 1:
        v = y[0]
    else:
        j = int(np.searchsorted(x, T, side='left'))
        j = min(max(j, 1), len(x) - 1)
        v = (y[j] - y[j - 1]) / (x[j] - x[j - 1]) * (T - x[j - 1]) + y[j - 1]
    p = min(max(v, c['p_lo']), c['p_hi'])
    return p if z >= 0 else 1.0 - p


def _calibrate_m4(c, z):
    if c['method'] == 'nested':
        return _calibrate_m4(c['inner'], z)
    if c['method'] == 'isotonic_sym':
        return _iso_sym(c, z)
    if c['method'] == 'temperature':
        return _sig(min(max(z / c['T'], -40.0), 40.0))
    return _sig(min(max(z, -40.0), 40.0))


def _logit(p, eps):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


@pytest.mark.parametrize('path', SYN + REAL, ids=_ids(SYN + REAL))
def test_plain_formulas_from_the_json_reproduce_the_reference(path, json_models):
    docs = json_models.docs
    m4, ph, yw = docs['m4'], docs['phys6'], docs['yaw']
    sc = m4['detector']['segmentClassifier']
    ref = FX.read_case(path)['reference']
    for s in ref['segments']:
        F = np.array([np.nan if s['features'][k] is None else s['features'][k] for k in sc['features']])
        Z = (F - np.asarray(sc['center'])) / np.asarray(sc['scale'])
        Z = np.clip(np.where(np.isfinite(Z), Z, 0.0), -sc['clip'], sc['clip'])
        L = np.asarray(sc['coef']) @ Z + np.asarray(sc['intercept'])
        P = np.exp(L - L.max())
        P /= P.sum()
        for c, v in zip(sc['classes'], P):
            assert abs(v - s['proba'][c]) <= 1e-12
        if not s['canonical_ok']:
            continue
        W = np.array(s['window'], dtype=float)
        o = s['models']['m4']
        x = np.nan_to_num(W[m4['hand']['index'], ::m4['hand']['step']])
        zr = float(np.dot(m4['hand']['coef'], x) + m4['hand']['intercept'])
        assert abs(zr - o['z_raw']) <= 1e-12 * max(1, abs(zr))
        p = _calibrate_m4(m4['calibrator'], o['z_raw'])
        assert abs(p - o['p']) <= 1e-12
        assert abs(_logit(o['p'], m4['logitEps']) - o['z']) <= 1e-9
        y = s['models']['yaw']
        if y['available']:
            x8 = W[yw['index'], ::yw['step']]
            zy = float(np.dot(yw['coef'], x8) + yw['intercept'])
            assert abs(zy - y['z_raw']) <= 1e-12 * max(1, abs(zy))
            assert abs(_sig(zy) - y['p']) <= 1e-12
        q = s['models']['phys6']
        if q['available']:
            f6 = np.asarray(q['features6'])
            Wv = np.asarray(ph['winsor_med']) + np.asarray(ph['winsor_s']) * np.arcsinh(
                (f6 - np.asarray(ph['winsor_med'])) / np.asarray(ph['winsor_s']))
            zp = float(np.dot(ph['coef'], (Wv - np.asarray(ph['centre'])) / np.asarray(ph['scale'])))
            assert ph['winsor'] == 'asinh' and abs(zp - q['z_raw']) <= 1e-12 * max(1, abs(zp))
    # the k-attempt aggregations and the stopping rule of every model
    agg = m4['aggregation']
    segs = {s['seg']: s for s in ref['segments']}
    for mid, dec in ref['decisions'].items():
        for row in dec['k_curve']:
            k = row['k']
            outs = [segs[r['seg']]['models'][mid] for r in dec['k_curve'][:k]]
            if mid == 'm4':
                s_k = agg['s'][str(min(k, max(int(i) for i in agg['s'])))]
                want = _sig(min(max(s_k * (sum(o['z'] for o in outs) / k), -40.0), 40.0))
            elif mid == 'cnn':
                tau = docs['cnn']['aggregation']['tau']
                t_k = 1.0 if k == 1 else tau.get(str(k), tau[str(max(int(i) for i in tau))])
                want = _sig(sum(_logit(o['p'], 1e-12) for o in outs) / t_k)
            elif mid == 'phys6':
                want = _sig(sum(o['z'] for o in outs) / k ** ph['aggregation']['alpha'])
            else:
                want = sum(o['p'] for o in outs) / k
            assert abs(want - row['p']) <= 1e-12, (mid, k, want, row['p'])


# ───────────────────────────── files, manifest, fixtures ─────────────────────────────


def test_manifest_matches_the_files_and_the_cnn_is_lane_finals_export(json_models):
    man = json.loads((EX.APP_MODELS_DIR / 'manifest.json').read_text(encoding='utf-8'))
    assert man['default'] == 'cnn' and [m['id'] for m in man['models']] == list(RT.MODEL_IDS)
    for m in man['models']:
        data = (EX.APP_MODELS_DIR / m['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == m['sha256'] and len(data) == m['bytes']
        acc = m['loso']['accuracy']
        lo, hi = m['loso']['ci95']
        assert 0.5 < acc < 1 and lo < acc < hi
    cnn = json.loads((EX.APP_MODELS_DIR / RT.FILES['cnn']).read_text(encoding='utf-8'))
    if EX.FINAL_MODEL.exists():
        assert cnn == json.loads(EX.FINAL_MODEL.read_text(encoding='utf-8'))
    text = (EX.APP_MODELS_DIR / RT.FILES['cnn']).read_text(encoding='utf-8')
    assert '\n' not in text.rstrip('\n') and text == EX.dumps(cnn, minify=True)
    for mid in ('m4', 'phys6', 'yaw'):
        doc = json_models.docs[mid]
        assert doc['identity']['data_id'] == man['identity']['data_id']
        assert doc['training']['n_persons'] == 20 and doc['training']['persons'][0] == 'P01'
    det = json_models.docs['m4']['detector']
    from scipy import signal
    sos = signal.butter(4, 15.0, btype='low', fs=60.0, output='sos')
    assert np.array_equal(np.asarray(det['lowpassSos']), sos) and det['lowpassPadlen'] == 15
    assert abs(det['causalDelayMs'] - 44.429095814613106) < 1e-12


@pytest.mark.parametrize('path', SYN + REAL, ids=_ids(SYN + REAL))
def test_fixture_motion_is_app_rounded_and_starts_at_zero(path):
    doc = FX.read_case(path)
    m = doc['motion']
    assert m['t'][0] == 0.0 and all(b > a for a, b in zip(m['t'], m['t'][1:]))
    assert FX.normalise_motion(m) == m                      # a LiveBuffer fed these samples stores these values
    if doc['kind'] == 'synthetic':
        assert doc['source'] is None and 'P0' not in json.dumps(doc['motion'])


@pytest.mark.parametrize('path', SYN + REAL, ids=_ids(SYN + REAL))
def test_live_view_is_a_prefix_of_the_offline_view(path):
    doc = FX.read_case(path)
    off, live = doc['reference']['attempts'], doc['live']['attempts']
    assert live == off[:len(live)]
    for mid, dec in doc['live']['decisions'].items():
        assert dec['stopped'] in ('watching', 'confident', 'k_max', 'grip')
        o = doc['reference']['decisions'][mid]
        if dec['stopped'] in ('confident', 'k_max'):
            assert dec['p_right'] == o['p_right'] and dec['n_used'] == o['n_used']


def test_synthetic_expectations():
    by = {p.name.split('.')[0]: FX.read_case(p) for p in SYN}
    away = by['syn03_android_screen_away']['reference']
    assert away['grip']['grip_warning'] and all(d['stopped'] == 'grip' for d in away['decisions'].values())
    a, b = by['syn01_android_right_3touch']['reference'], by['syn04_android_mirror_of_syn01']['reference']
    assert [(s['onset_ms'], s['end_ms']) for s in a['segments']] == [(s['onset_ms'], s['end_ms']) for s in b['segments']]
    for sa, sb in zip(a['segments'], b['segments']):
        for c in ('approach', 'return', 'other'):
            assert abs(sa['proba'][c] - sb['proba'][c]) < 1e-9          # the segment features are mirror-even
    assert a['attempts'] == b['attempts'] and a['attempts']
    for case in ('syn01_android_right_3touch', 'syn02_ios_left_gap_repeat', 'syn04_android_mirror_of_syn01',
                 'syn05_gecko_left_edges'):
        doc = by[case]
        hand = doc['truth']['hand']
        dec = doc['reference']['decisions']['m4']
        assert dec['answered'] and (dec['p_right'] > 0.5) == (hand == 'right'), case
    edges = by['syn05_gecko_left_edges']['reference']
    assert edges['segments'][0]['started_active'] and not edges['segments'][-1]['ended']
    gap = by['syn02_ios_left_gap_repeat']['reference']
    assert gap['repeats_dropped'] >= 1 and not all(gap['stream']['valid'])


def test_no_participant_name_in_any_stage3_output_or_source():
    """The name scan of stage2's privacy test, over the stage3 sources, the exported models and every fixture (the
    local real ones included). Only the count of hits is ever printed."""
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location(
        '_s2_privacy', Path(__file__).resolve().parents[2] / 'stage2' / 'tests' / 'test_cuefree_privacy.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    toks = mod._nickname_tokens()
    if not toks:
        pytest.skip('no snapshot with names to check against')
    texts = {str(f): f.read_text(encoding='utf-8') for f in Path(__file__).resolve().parents[1].rglob('*.py')}
    texts.update({str(f): f.read_text(encoding='utf-8') for f in EX.APP_MODELS_DIR.glob('*.json')})
    texts.update({str(f): f.read_text(encoding='utf-8') for f in FX.SYN_DIR.glob('*.json')})
    for f in SYN + REAL:
        texts[str(f)] = json.dumps(FX.read_case(f))
    pat = re.compile(r'\b(' + '|'.join(re.escape(t) for t in sorted(toks)) + r')\b', re.IGNORECASE)
    hits = [k for k, v in texts.items() if pat.search(v)]
    assert len(texts) > 10 and not hits, f'{len(hits)} file(s) contain a participant name token'
