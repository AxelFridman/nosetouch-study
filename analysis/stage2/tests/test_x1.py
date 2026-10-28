"""Lane x1 (the PD01 case study): same code path, no model, mirror invariance, statistics (declared-scale z,
robustness, the null of the "beyond all" count, multiplicity family), figures and summary from saved files, numbers
coverage, privacy. Run: cd analysis && .venv/Scripts/python.exe -m pytest stage2/tests/test_x1.py -q -p no:cacheprovider
(the tests marked ``exploratory`` read the exploratory files and run only with STAGE2_EXPLORATORY_TESTS=1 or
-m exploratory; see stage2/tests/conftest.py)"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2.lanes import x1_features as XF
from stage2.lanes import x1_stats as XS

CASE = 'PD01'


def _case_cache():
    from stage2.lanes import x1_data as XD
    d = XD.cache_dir(CASE)
    if not (d / 'meta.json').exists():
        pytest.skip('no exploratory cache (python -m stage2.pipeline compute --lanes x1)')
    return d


def _outputs() -> dict[str, str]:
    """Every x1 output as text (tables, figure registries, results, numbers, the summary, LaTeX tables)."""
    files = [*(C.TAB_DIR / 'x1').glob('*'), *(C.FIG_DIR / 'x1').glob('*.json'), *C.RES_DIR.glob('x1_*'),
             C.RES_DIR / 'finalists_x1.json', C.OUT_DIR / 'x1_SUMMARY.md']
    return {f.name: f.read_text(encoding='utf-8', errors='ignore') for f in files if f.is_file()}


# ───────────────────────────── same code path, ids apart ─────────────────────────────


@pytest.mark.exploratory           # reads the exploratory cache / raw files / curation (conftest.py)
def test_exploratory_cache_is_built_like_the_main_cache():
    d = _case_cache()
    if not (C.CACHE_DIR / 'meta.json').exists():
        pytest.skip('no main cache')
    case = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
    main = json.loads((C.CACHE_DIR / 'meta.json').read_text(encoding='utf-8'))
    assert case['params'] == main['params']
    assert case['channels'] == main['channels']
    for k in ('params_sha256', 'builder_sha256', 'cache_version'):
        assert case['data_identity'][k] == main['data_identity'][k], k


@pytest.mark.exploratory           # reads the exploratory cache / raw files / curation (conftest.py)
def test_exploratory_ids_never_reach_the_main_curation_or_cache():
    _case_cache()
    from stage2.lanes import x1_data as XD
    XD.assert_apart(CASE)                                     # raises when an id leaked
    t = pd.read_pickle(C.CACHE_DIR / 'trials.pkl')
    assert CASE not in set(t['person_id'].astype(str))


# ───────────────────────────── characteristics ─────────────────────────────


def _sine(f_hz: float, amp: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * 60)) / 60.0
    W = np.zeros((t.size, 3))
    W[:, 0] = amp * np.sin(2 * np.pi * f_hz * t)
    return W


def test_spectral_characteristics_find_a_known_oscillation():
    out = XF.spectral_characteristics(_sine(5.0, 10.0), 'x')
    assert abs(out['x_peak_hz'] - 5.0) < 0.3
    assert abs(out['x_rms_3_7'] - 10.0 / math.sqrt(2)) / (10.0 / math.sqrt(2)) < 0.1   # band RMS = amplitude / sqrt 2
    assert out['x_rms_7_12'] < 0.2 * out['x_rms_3_7']
    assert abs(out['x_rms_3_12'] ** 2 - (out['x_rms_3_7'] ** 2 + out['x_rms_7_12'] ** 2)) < 1e-9   # bands add up
    hi = XF.spectral_characteristics(_sine(9.0, 10.0), 'x')
    assert abs(hi['x_peak_hz'] - 9.0) < 0.3 and hi['x_rms_7_12'] > 3 * hi['x_rms_3_7']


def test_short_windows_leak_across_the_band_border():
    """The leakage the summary states: a tone just below 7 Hz in a 1-s window puts real power into 7-12 Hz, less in a
    long window."""
    short, long = XF.sine_leakage(6.5, 1.0), XF.sine_leakage(6.5, 4.0)
    assert short['rms_7_12'] > 0.3 * short['rms_3_7']
    assert long['rms_7_12'] < short['rms_7_12']


def test_smoothness_indices_fall_when_an_oscillation_is_added():
    t = np.arange(0, 1.5, 1 / 60)
    tau = t / t[-1]
    v = 100 * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4)          # minimum-jerk speed profile
    vo = v + 40 * np.sin(2 * np.pi * 5 * t)              # a 5-Hz oscillation, 21 % of the 187.5 deg/s peak
    assert XF.ldlj(vo) < XF.ldlj(v) and XF.sparc(vo) < XF.sparc(v)
    assert XF.n_peaks(v) == 1 and XF.n_peaks(vo) > 3
    assert XF.n_peaks(v + 15 * np.sin(2 * np.pi * 5 * t)) < 3    # a ripple under 10 % of the peak is not a peak


def _mirror(stream, row, approach, fixed):
    """The left/right mirror of one trial: polar vectors S x, pseudovectors -S w (S = diag(-1, 1, 1)), the chiral
    QC columns negated, the hand flipped (README section 3)."""
    s = {k: np.array(v, copy=True) for k, v in stream.items()}
    s['f'][:, 0] *= -1
    s['acc'][:, 0] *= -1
    s['gyro'][:, 1:] *= -1
    r = dict(row)
    r['hand'] = {'left': 'right', 'right': 'left'}[row['hand']]
    for k in ('first_half_yaw_deg', 'net_yaw_deg'):
        r[k] = -row[k]
    from stage2.standardize import mirror_channels
    return s, r, mirror_channels(approach), mirror_channels(fixed)


def test_characteristics_are_mirror_invariant(primary):
    """Mirroring a trial and flipping its hand changes no characteristic: left- and right-hand trials are
    measured alike, and the hand-consistent yaw keeps its meaning."""
    for i in (0, 7, 33):
        st, row = primary.stream(i), primary.trials.iloc[i].to_dict()
        a = XF.trial_characteristics(st, row, primary.approach[i], primary.channels, primary.fixed[i],
                                     primary.fixed_mask[i])
        s2, r2, ap2, fx2 = _mirror(st, row, primary.approach[i], primary.fixed[i])
        b = XF.trial_characteristics(s2, r2, ap2, primary.channels, fx2, primary.fixed_mask[i])
        for k in a:
            if k == 'g_mag_dev':
                continue                                     # a float-precision residual, not a characteristic
            if isinstance(a[k], float) and math.isnan(a[k]):
                assert math.isnan(b[k]), k
            else:
                assert b[k] == pytest.approx(a[k], rel=1e-5, abs=1e-5), k


def test_the_label_only_sets_the_sign_of_the_hand_consistent_yaw(primary):
    st, row = primary.stream(3), primary.trials.iloc[3].to_dict()
    a = XF.trial_characteristics(st, row, primary.approach[3], primary.channels, primary.fixed[3], primary.fixed_mask[3])
    other = dict(row, hand={'left': 'right', 'right': 'left'}[row['hand']])
    b = XF.trial_characteristics(st, other, primary.approach[3], primary.channels, primary.fixed[3], primary.fixed_mask[3])
    changed = {k for k in a if not (a[k] == b[k] or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k])))}
    assert changed <= {'fhy_hand', 'net_yaw_hand', 'sign_typical', 'net_sign_typical', 'yaw_peak_hand', 'yaw_peak_s'}
    again = XF.trial_characteristics(st, row, primary.approach[3], primary.channels, primary.fixed[3], primary.fixed_mask[3])
    assert json.dumps(a, sort_keys=True) == json.dumps(again, sort_keys=True)          # deterministic


def test_the_calibrated_gravity_is_recovered_from_the_cached_fixed_window(primary):
    """The canonical |a| re-derived from the cached stream matches the cache wherever the fixed window has it, so
    the acceleration at a late first touch is on the cache's own scale."""
    k = primary.channels.index('acc_mag')
    for i in (0, 11, 50):
        st, row = primary.stream(i), primary.trials.iloc[i].to_dict()
        can, mag, dev = XF.cached_canonical(st, row, primary.fixed[i], primary.fixed_mask[i], primary.channels)
        assert 8.0 <= mag <= 11.5 and dev < 1e-4
        on = float(row['onset_ms'])
        grid = on - C.FIXED_PRE_S * 1000.0 + np.arange(primary.fixed.shape[-1]) * (1000.0 / C.FS_HZ)
        msk = primary.fixed_mask[i].astype(bool)
        assert np.allclose(np.interp(grid[msk], can.t, can.channel('acc_mag')), primary.fixed[i, k][msk], atol=1e-4)


def test_every_characteristic_is_defined_and_has_a_rationale():
    keys = [c.key for c in XS.CHARACTERISTICS]
    assert len(keys) == len(set(keys))
    for c in XS.CHARACTERISTICS:
        assert c.stratum in XS.STRATA and c.group in XS.GROUPS and c.scale in ('log', 'linear')
        assert len(c.rationale) > 30 and c.phenomenon, c.key
    # the redundant pairs the audit named share a phenomenon (one finding, not two)
    for a, b in (('peak_acc', 'touch_acc'), ('touch_s', 'touch_share'), ('rotation_net', 'tilt_change')):
        assert XS.BY_KEY[a].phenomenon == XS.BY_KEY[b].phenomenon


# ───────────────────────────── statistics ─────────────────────────────


def test_rank_p_percentile_and_robust_z():
    ref = np.arange(1.0, 21.0)
    s = XS.compare(ref, 25.0)
    assert s['p_rank'] == pytest.approx(2 / 21) and s['p_floor'] == pytest.approx(2 / 21)
    assert s['percentile'] == 100 and s['n_more_extreme'] == 0 and s['side'] == 'above'
    assert s['robust_z'] == pytest.approx((25 - 10.5) / (1.4826 * 5.0))
    m = XS.compare(ref, 10.5)
    assert m['percentile'] == 50 and m['p_rank'] == 1.0
    low = XS.compare(ref, 0.0)
    assert low['p_rank'] == pytest.approx(2 / 21) and low['percentile'] == 0
    few = XS.compare(ref[:XS.MIN_REF - 1], 25.0)
    assert few['n_ref'] == XS.MIN_REF - 1 and math.isnan(few['percentile'])       # too few: no statistic


def test_robust_z_uses_the_declared_scale_and_counts_case_trials():
    rng = np.random.default_rng(3)
    ref = np.exp(rng.normal(1.0, 0.4, 20))                    # right-skewed reference
    x = float(np.max(ref) * 1.5)
    lin, log = XS.compare(ref, x, 'linear'), XS.compare(ref, x, 'log', [x * 0.9, x * 1.1, float(np.median(ref))])
    zl = (math.log(x) - np.median(np.log(ref))) / (1.4826 * np.median(np.abs(np.log(ref) - np.median(np.log(ref)))))
    assert log['z_scale'] == 'log' and log['robust_z'] == pytest.approx(zl)
    assert log['robust_z_linear'] == pytest.approx(lin['robust_z']) and log['robust_z'] < lin['robust_z']
    assert abs(log['z_loo_min']) <= abs(log['robust_z']) + 1e-12 <= abs(log['z_loo_max']) + 1e-12
    assert log['n_case_trials_beyond'] == 2 and log['n_case_trials_valid'] == 3
    neg = XS.compare(np.r_[ref, -1.0], x, 'log')               # a non-positive value: the log scale cannot apply
    assert neg['z_scale'] == 'linear'


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan, 0.5])
    q = XS.bh(p)
    ok = np.isfinite(p)
    assert np.allclose(q[ok], multipletests(p[ok], method='fdr_bh')[1])
    assert np.isnan(q[~ok]).all()


def test_crawford_howell_formula():
    rng = np.random.default_rng(0)
    ref = rng.normal(10, 2, 20)
    t, p = XS.crawford_howell(ref, 16.0)
    expect = (16.0 - ref.mean()) / (ref.std(ddof=1) * math.sqrt(21 / 20))
    from scipy import stats
    assert t == pytest.approx(expect) and p == pytest.approx(2 * stats.t.sf(abs(expect), 19))


def test_null_count_compares_each_reference_person_with_the_others():
    rng = np.random.default_rng(2)
    persons = [f'P{i:02d}' for i in range(1, 21)]
    M = pd.DataFrame(rng.normal(size=(20, 4)), columns=list('abcd'), index=persons)
    M.loc[CASE] = [9.0, -9.0, 0.0, 0.0]                       # beyond all in a and b
    phen = {'a': 'p1', 'b': 'p1', 'c': 'p2', 'd': 'p3'}
    r = XS.beyond_all_counts(M, CASE, phen)
    assert r['case_n_characteristics'] == 2 and r['case_n_phenomena'] == 1 and r['case_keys'] == ['a', 'b']
    per = pd.DataFrame(r['per_person']).set_index('person_id')
    for j in persons:                                          # a reference person's count against the other 19
        others = [p for p in persons if p != j]
        n = sum(bool(M.at[j, k] > M.loc[others, k].max() or M.at[j, k] < M.loc[others, k].min()) for k in 'abcd')
        assert per.at[j, 'n_characteristics'] == n
    assert r['p_rank_characteristics'] == pytest.approx((1 + r['n_ref_ge_case_characteristics']) / 21)
    assert r['expected_case_n_characteristics'] == pytest.approx(4 * 2 / 21)
    own = M.copy()
    own.loc['P01'] = [99.0, 0.0, 0.0, 0.0]                     # judged on another value: beyond all in a
    r2 = XS.beyond_all_counts(M, CASE, phen, own)
    assert pd.DataFrame(r2['per_person']).set_index('person_id').at['P01', 'keys'] == ['a']


def test_multivariate_reference_distances_leave_the_person_out():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(rng.normal(size=(20, 4)), columns=['rt_s', 'ldlj', 'sparc', 'fhy_hand'],
                       index=[f'P{i:02d}' for i in range(1, 21)])
    far = XS.multivariate(ref, pd.Series({'rt_s': 9.0, 'ldlj': -9.0, 'sparc': 9.0, 'fhy_hand': 9.0}))
    assert far['n_ref_beyond'] == 0 and far['p_mahalanobis'] == pytest.approx(1 / 21)
    mid = XS.multivariate(ref, ref.median())
    assert mid['p_mahalanobis'] > 0.5
    assert len(far['mahalanobis_ref_loo']) == 20 and len(far['scores_ref']) == 20
    loco = XS.multivariate_loco(ref, ref.median())
    assert [r['left_out'] for r in loco] == list(ref.columns)


# ───────────────────────────── saved outputs ─────────────────────────────


def _saved():
    if not (C.RES_DIR / 'x1_characteristics.json').exists():
        pytest.skip('x1 outputs not computed (python -m stage2.pipeline compute --lanes x1)')


def _main_rows() -> pd.DataFrame:
    return pd.DataFrame(json.loads((C.RES_DIR / 'x1_characteristics.json').read_text(encoding='utf-8'))['rows'])


def test_multiplicity_family_is_the_prespecified_characteristics():
    """q-values run over the pre-specified characteristics only; post hoc ones carry none."""
    _saved()
    m = _main_rows()
    post = m['post_hoc'].astype(bool)
    assert m.loc[post, 'q_rank'].isna().all() and m.loc[post, 'q_ch'].isna().all()
    pre = m[~post]
    assert np.allclose(pre['q_rank'], XS.bh(pre['p_rank']), equal_nan=True)
    assert np.allclose(pre['q_ch'], XS.bh(pre['ch_p']), equal_nan=True)


def test_robust_verdicts_follow_their_checks():
    """Robust implies beyond all; every robust verdict passed every check it ran, and a failed check is listed."""
    _saved()
    m = _main_rows()
    assert not (m['robust'] & ~m['beyond_all']).any()
    for _, r in m[m['beyond_all']].iterrows():
        failed = [c for c in str(r['checks_failed'] or '').split(';') if c]
        na = [c for c in str(r['checks_unavailable'] or '').split(';') if c]
        assert bool(r['robust']) == (not failed and not na), r['key']
        assert set(failed) | set(na) <= set(str(r['checks_run'] or '').split(';')), r['key']
        assert not set(failed) & set(na), r['key']


def test_figures_redraw_from_saved_files_only(monkeypatch, tmp_path):
    _saved()
    from stage2 import data
    from stage2.lanes import x1_data as XD
    from stage2.lanes import x1_figures as F
    from stage2.pipeline import Lane
    from stage2.plotting import save_fig

    def refuse(*a, **k):
        raise AssertionError('a figure function read a cache or raw data')
    for mod, name in ((data, 'load_cache'), (data, 'load_tables'), (XD, 'load'), (XD, 'load_tables'),
                      (XD, 'ensure_cache'), (XD, 'exploratory_doc')):
        monkeypatch.setattr(mod, name, refuse)
    saved = {}

    def fake_save(self, fig, name, *, caption, decision='', data=None):
        assert isinstance(data, pd.DataFrame) and len(data), name
        assert re.fullmatch(r'X1_\d\d_[a-z0-9_]+', name) and caption and decision, name
        saved[name] = caption
        return save_fig(fig, name, caption=caption, decision=decision, fig_dir=tmp_path)
    monkeypatch.setattr(Lane, 'save_fig', fake_save)
    F.draw()
    assert sorted(saved) == sorted(F.FIGURE_FUNCTIONS)
    assert sorted(p.stem for p in tmp_path.glob('*.png')) == sorted(saved)


def test_summary_does_not_need_the_figures(monkeypatch, tmp_path):
    """The summary lists every figure from x1_texts, so it is complete without figures/x1/captions.json."""
    _saved()
    from stage2.lanes import x1_report as XR
    from stage2.lanes import x1_texts as XT
    from stage2.pipeline import Lane
    monkeypatch.setattr(Lane, 'fig_dir', property(lambda self: tmp_path / 'no_figures'))
    D, N = XR.current_numbers()
    text = XR.summary(D, N)
    for name in XT.FIGURE_NAMES:
        assert f'`{name}`: ' + XT.texts(name)['decision'] in text, name


def _formatted_numbers(nums: dict) -> set[str]:
    from stage2 import report_export as RX
    have = set()
    for e in nums.values():
        v = e['value']
        if v is None:
            continue
        if isinstance(v, str):
            have |= {v, *re.findall(r'[-+]?\d+(?:\.\d+)?', v)}
            continue
        s = RX.format_value(v, e['format'])
        have |= {s, s.lstrip('+'), s.lstrip('-+'), s.rstrip('%'), s.lstrip('-+').rstrip('%')}
    return have


def _prose_numbers(text: str) -> list[str]:
    lines, inside = [], False
    for ln in text.splitlines():
        if ln.startswith('```'):
            inside = not inside
            continue
        if inside or ln.startswith('|'):
            continue
        lines.append(ln)
    prose = '\n'.join(lines)
    # names and labels carry digits that are not results: persons, bands, windows, file and figure names, versions
    prose = re.sub(r'`[^`]*`|\bP\d\d\b|PD\d\d(-s\d)?|X1_\d\d\w*|\d+(\.\d+)?-\d+(\.\d+)? (?:Hz|years|deg/s)'
                   r'|\d+(\.\d+)?-Hz|\d+\.\d+\.\d+|\bMDS-UPDRS\b|\(\d{4}\)|\d{4}\b|\bs\d\b', ' ', prose)
    return re.findall(r'(?<![A-Za-z_\d.])[-+]?\d+(?:\.\d+)?(?:e-?\d+)?', prose)


def test_every_number_in_the_summary_and_figure_texts_is_a_numbers_entry():
    _saved()
    from stage2.lanes import x1_texts as XT
    nums = json.loads((C.RES_DIR / 'x1_numbers.json').read_text(encoding='utf-8'))
    have = _formatted_numbers(nums)
    # numbers that are definitions, not results (thresholds, window lengths, band edges, q-level, filter order)
    allowed = {'0.05', '0.5', '2', '1', '0', '3', '7', '12', '15', '100', '200', '250', '450', '30', '20', '1.4826',
               '10', '5', '3.5', '400', '64', '60', '0.6', '1.0', '2/(n', '1998', '2015', '-2'}
    texts = [(C.OUT_DIR / 'x1_SUMMARY.md').read_text(encoding='utf-8')]
    texts += [' '.join(XT.texts(n).values()) for n in XT.FIGURE_NAMES]
    missing = sorted({tok for txt in texts for tok in _prose_numbers(txt)
                      if tok not in have and tok.lstrip('-+') not in have and tok not in allowed})
    assert not missing, missing


def test_no_model_metric_and_no_finalist_for_the_case():
    _saved()
    banned = {'accuracy', 'roc_auc', 'auc', 'log_loss', 'ece', 'brier', 'predictions', 'p_right', 'extra_predictions'}

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    for f in [*C.RES_DIR.glob('x1_*.json'), *(C.TAB_DIR / 'x1').glob('*.json')]:
        assert not banned & set(keys(json.loads(f.read_text(encoding='utf-8')))), f.name
    assert json.loads((C.RES_DIR / 'finalists_x1.json').read_text(encoding='utf-8')) == []
    assert not (C.RES_DIR / 'registry_x1.csv').exists()


@pytest.mark.exploratory           # reads the exploratory cache / raw files / curation (conftest.py)
def test_outputs_carry_no_name_device_height_or_quasi_identifier():
    _saved()
    _case_cache()
    from stage2.lanes import x1_data as XD
    toks, models = set(), set()
    for f in XD.raw_dir(CASE).rglob('session.json'):          # read in memory, never printed
        s = json.loads(f.read_text(encoding='utf-8'))
        nick = ((s.get('participant') or {}).get('nickname') or '').strip()
        toks |= {t for t in re.split(r'[\s\-]+', nick) if len(t) >= 3 and t.isalpha()}
    for s in XD.person_spec(CASE)['sessions']:
        if s.get('model_hint'):
            models.add(s['model_hint'])
    texts = _outputs()
    texts |= {p.name: p.read_text(encoding='utf-8') for p in C.LANES_DIR.glob('x1_*.py')}
    quasi = re.compile(r'\bP-[A-Z0-9]{6}\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                       r'|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}')
    name = re.compile(r'\b(' + '|'.join(map(re.escape, sorted(toks))) + r')\b', re.IGNORECASE) if toks else None
    pronoun = re.compile(r'\b(he|his|him|she|her|hers)\b', re.IGNORECASE)
    for fname, txt in texts.items():
        assert not (name and name.search(txt)), f'{fname} holds a name token'
        assert not any(m in txt for m in models), f'{fname} names the device model'
        if not fname.endswith('.py'):
            assert not quasi.search(txt), f'{fname} holds an app code, backend id or timestamp'
        if fname.endswith(('.md', '.json', '.tex')):
            assert not pronoun.search(txt), f'{fname} uses a gendered pronoun'
    task = json.loads((C.TAB_DIR / 'x1' / 'x1_task.json').read_text(encoding='utf-8'))
    assert 'height_cm' not in json.dumps(task) and 'age_years' not in json.dumps(task['case'])
    assert 'started_at' not in json.dumps(task) and 'ended_at' not in json.dumps(task)


def test_numbers_file_is_valid_and_cites_saved_files():
    _saved()
    from stage2 import report_export as RX
    nums = json.loads((C.RES_DIR / 'x1_numbers.json').read_text(encoding='utf-8'))
    RX.validate_numbers(nums)
    assert {'n_ref_persons', 'p_floor', 'n_beyond_prespecified', 'null_pre_lfl_p', 'mv_p', 'min_q_rank',
            'min_q_ch'} <= set(nums)
    for k, e in nums.items():
        assert (C.OUT_DIR / e['source']).exists(), (k, e['source'])
        if isinstance(e['value'], float) and 0 < e['value'] < 0.001 and e['format'].endswith('f'):
            assert False, f'{k}: a small value would print as 0 with {e["format"]}'
    assert nums['p_floor']['value'] == pytest.approx(2 / (nums['n_ref_persons']['value'] + 1))
