"""Cue-free baseline v0 on known streams, its evaluation contract and stress test, and the no-names /
no-quasi-identifier guarantees of every output."""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2 import cuefree


def bump(t, t0, dur, peak):
    x = np.clip((t - t0) / dur, 0, 1)
    return peak * np.sin(np.pi * x) ** 2


def test_detects_the_first_movement_after_stillness():
    t = np.arange(0, 9000, 1000 / 60)
    w = np.zeros((t.size, 3))
    w[:, 2] = bump(t, 2000, 1200, 150) + bump(t, 5000, 1000, -120)   # approach, then a return
    w += np.random.default_rng(0).normal(0, 1.0, w.shape)
    dets = cuefree.detect(t, w)
    assert len(dets) == 2
    d = dets[0]
    assert abs(d.onset_ms - 2000) < 200 and abs(d.end_ms - 3200) < 250 and d.still_before_s >= 1.0
    res = cuefree.classify(d, 2000, 3200, cue_ms=3450, return_onset_ms=5000, return_end_ms=6000)
    assert res['outcome'] == 'approach' and res['iou'] > 0.6
    assert cuefree.classify(dets[1], 2000, 3200, cue_ms=3450, return_onset_ms=5000,
                            return_end_ms=6000)['outcome'] == 'return'


def test_needs_stillness_before_and_a_sustained_movement():
    t = np.arange(0, 6000, 1000 / 60)
    w = np.zeros((t.size, 3))
    w[:, 0] = bump(t, 500, 1000, 150)            # starts after only 0.5 s of stillness: not a candidate
    w[:, 1] = bump(t, 3500, 60, 40)               # 60 ms blip: not sustained
    assert cuefree.detect(t, w) == []
    assert cuefree.classify(None, 1000, 2000)['outcome'] == 'none'
    assert cuefree.classify(cuefree.Detection(200, 600, 0, 0, 1.0, 50, True), 1000, 2000)['outcome'] == 'earlier_movement'


def _settle_then_reach():
    """An earlier movement (the arm settling), 2.5 s of stillness, the approach, then the return."""
    t = np.arange(0, 10000, 1000 / 60)
    w = np.zeros((t.size, 3))
    w[:, 0] = bump(t, 300, 900, 90)               # arm settling 0.3-1.2 s
    w[:, 2] = bump(t, 3700, 1200, 150) + bump(t, 6500, 1000, -120)
    return t, w + np.random.default_rng(1).normal(0, 1.0, w.shape)


def test_stillness_is_measured_in_the_detectors_own_signal():
    t, w = _settle_then_reach()
    s = cuefree.still_before_s(t, w, 3700.0)
    assert 2.0 < s < 2.8                           # ~2.5 s, nothing added for the smoothing
    assert cuefree.still_before_s(t, w, 9000.0) == 0.0


def test_stress_shortened_stillness_breaks_v0_below_its_rule():
    """Finding 8: the recorded stillness was protocol-enforced; cutting it below 1 s hides the approach."""
    t, w = _settle_then_reach()
    assert cuefree.classify(cuefree.first_movement(t, w), 3700, 4900)['outcome'] == 'approach'
    t2, w2, removed = cuefree.shorten_stillness(t, w, 3700.0, keep_s=0.4)
    assert 1.6 < removed / 1000 < 2.4 and np.allclose(np.diff(t2), 1000 / 60)
    det = cuefree.first_movement(t2, w2)
    assert det is None or cuefree.classify(det, 3700 - removed, 4900 - removed)['outcome'] != 'approach'
    t3, w3, removed3 = cuefree.shorten_stillness(t, w, 3700.0, keep_s=1.3)
    assert cuefree.classify(cuefree.first_movement(t3, w3), 3700 - removed3, 4900 - removed3)['outcome'] == 'approach'
    t4, _, removed4 = cuefree.shorten_stillness(t, w, 3700.0, keep_s=5.0)   # nothing to cut
    assert removed4 == 0.0 and t4.size == t.size


def test_evaluation_on_the_cache(primary):
    sub = primary.subset(np.arange(40))
    ev = cuefree.evaluate(sub)
    assert len(ev) == 40 and set(ev['outcome']) <= set(cuefree.OUTCOMES)
    s = cuefree.summarize(ev)
    assert abs(sum(s[f'share_{o}'] for o in cuefree.OUTCOMES) - 1.0) < 1e-9
    assert s['share_approach'] > 0.3
    st = cuefree.stress_evaluate(sub, keep_s=(0.5, 1.5))
    assert set(st['setting']) == {'recorded', '0.5', '1.5', 'random'} and len(st) == 4 * 40
    rec = st[st['setting'] == 'recorded'].reset_index(drop=True)
    assert (rec['outcome'] == ev['outcome']).all()   # 'recorded' is v0 on the unchanged stream
    assert (st.loc[st['setting'] == '0.5', 'outcome'] == 'approach').mean() < s['share_approach']


# ───────────────────────────── privacy (findings 9 and 11) ─────────────────────────────


def _nickname_tokens() -> set[str]:
    """First and last names as stored in the raw session records (read in memory, never printed)."""
    toks: set[str] = set()
    for f in C.SNAPSHOT_DIR.rglob('session.json'):
        s = json.loads(f.read_text(encoding='utf-8'))
        nick = ((s.get('participant') or {}).get('nickname') or '').strip()
        for tok in re.split(r'[\s\-]+', nick):
            if len(tok) >= 3 and tok.isalpha():
                toks.add(tok)
    return toks


def _texts_to_scan() -> dict[str, str]:
    """Every Stage 2 output and source as text: tables, captions, figure texts (recorded at save time and,
    when pdftotext is available, extracted from the PDFs), PNG bytes (metadata), results, the cache's tables
    (CSV, JSON and the pickles through to_csv), the string arrays of the npz files, the sources and READMEs."""
    out: dict[str, str] = {}
    for d, pats in ((C.TAB_DIR, ('**/*.csv', '**/*.json', '**/*.tex')), (C.FIG_DIR, ('**/*.json',)),
                    (C.RES_DIR, ('*.json', '*.csv', '*.tex')), (C.CACHE_DIR, ('*.csv', '*.json')),
                    (C.ANALYSIS_DIR / 'stage2', ('**/*.py', '**/*.md'))):
        for p in pats:
            for f in d.glob(p):
                if f.is_file():
                    out[str(f)] = f.read_text(encoding='utf-8', errors='ignore')
    for f in C.FIG_DIR.rglob('*.png'):
        out[str(f)] = f.read_bytes().decode('latin-1')
    exe = shutil.which('pdftotext')
    for f in C.FIG_DIR.rglob('*.pdf'):
        if exe:
            r = subprocess.run([exe, '-q', str(f), '-'], capture_output=True, timeout=60)
            out[f'{f} (pdftotext)'] = r.stdout.decode('utf-8', errors='ignore')
    for f in C.CACHE_DIR.glob('*.pkl'):
        out[str(f)] = pd.read_pickle(f).to_csv(index=False)
    for f in C.CACHE_DIR.glob('*.npz'):
        with np.load(f, allow_pickle=False) as z:
            out[str(f)] = '\n'.join('\n'.join(map(str, z[k].ravel())) for k in z.files if z[k].dtype.kind == 'U')
    return out


def test_no_participant_name_in_any_output_or_source():
    toks = _nickname_tokens()
    if not toks:
        pytest.skip('no snapshot with names to check against')
    texts = _texts_to_scan()
    assert len(texts) > 50, 'nothing to scan'
    assert any(k.endswith('figure_texts.json') for k in texts) and any(k.endswith('trials.pkl') for k in texts)
    pat = re.compile(r'\b(' + '|'.join(re.escape(t) for t in sorted(toks)) + r')\b', re.IGNORECASE)
    hits = [k.rsplit('\\', 1)[-1].rsplit('/', 1)[-1] for k, v in texts.items() if pat.search(v)]
    assert not hits, f'{len(hits)} file(s) contain a participant name token: {hits}'


def test_figure_texts_are_recorded_and_the_pdfs_are_scanned():
    from stage2.locking import file_lock
    for reg in sorted(C.FIG_DIR.rglob('figure_texts.json')):       # the core folder and every lane folder
        # plotting.save_fig moves a figure's images and registry entries in under this lock: a figure being saved
        # by another process right now is either complete in both or absent from both
        with file_lock(reg.parent / 'captions.json'):
            texts = json.loads(reg.read_text(encoding='utf-8'))
            pdfs = sorted(f.stem for f in reg.parent.glob('*.pdf'))
        assert sorted(texts) == pdfs and all(texts[k] for k in pdfs), reg
    texts = json.loads((C.FIG_DIR / 'figure_texts.json').read_text(encoding='utf-8'))
    assert any('P08' in s for s in texts['S2_13_first_half_yaw'])        # tick labels are captured
    if shutil.which('pdftotext'):
        r = subprocess.run([shutil.which('pdftotext'), '-q', str(C.FIG_DIR / 'S2_02_exclusion_flow.pdf'), '-'],
                           capture_output=True, timeout=60)
        assert 'Primary set' in r.stdout.decode('utf-8', errors='ignore')


QUASI = re.compile(r'\bP-[A-Z0-9]{6}\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                   r'|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}')


def test_report_tables_carry_no_quasi_identifiers(expected):
    """Finding 11: no app code, backend id, timestamp, OS / browser version or screen geometry in the tables
    meant for the report (the git-ignored cache keeps them for tracing)."""
    from stage2.data import QUASI_IDENTIFIER_COLUMNS
    files = list(C.TAB_DIR.rglob('*.csv'))                     # core tables and every lane folder
    assert len(files) > 20
    for f in files:
        df = pd.read_csv(f)
        assert not set(df.columns) & set(QUASI_IDENTIFIER_COLUMNS), f.name
        assert not QUASI.search(f.read_text(encoding='utf-8')), f.name
    for f in [*C.TAB_DIR.rglob('*.json'), *C.TAB_DIR.rglob('*.tex'), *C.FIG_DIR.rglob('captions.json'),
              *C.FIG_DIR.rglob('figure_texts.json'), *C.RES_DIR.glob('*_numbers.json'), *C.RES_DIR.glob('*.tex')]:
        assert not QUASI.search(f.read_text(encoding='utf-8')), f.name
    sess = pd.read_csv(C.TAB_DIR / 'sessions.csv')
    assert {'session_label', 'session_order', 'device_model'} <= set(sess.columns)
    assert len(sess) == expected['report_sessions']
    # the cache still traces every label back
    cache = pd.read_pickle(C.CACHE_DIR / 'sessions.pkl')
    assert {'participant_code', 'started_at', 'session_id'} <= set(cache.columns)
    assert set(sess['session_label']) == set(cache['session_label'])


def test_core_docstrings_type_no_result_numbers():
    """Result numbers live in the regenerated files (tables/describe_summary.json, results/*_numbers.json), never in
    a docstring, where they go stale (cuefree.py once said 86 % / 98 % and a median of 567 ms after the data had
    moved to 90 % / 94 % and 575 ms). data.py and standardize.py are left out: editing them changes the cache's
    data identity, so their docstrings are fixed at the next planned rebuild."""
    import ast
    core = ('config.py', 'cuefree.py', 'cv.py', 'guard.py', 'locking.py', 'pipeline.py', 'plotting.py', 'qc.py',
            'report_export.py', 'screen.py')
    pat = re.compile(r'(?<![\w.\'"`])\d+(?:\.\d+)? ?%|median (?:of )?\d+ ?ms')
    allowed = {'95 %', '25 %'}           # the CI level and docs/09's locked-test share: conventions, not results
    for name in core:
        tree = ast.parse((C.ANALYSIS_DIR / 'stage2' / name).read_text(encoding='utf-8'))
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))]
        for node in nodes:
            for m in pat.finditer(ast.get_docstring(node) or ''):
                assert m.group(0) in allowed, (name, getattr(node, 'name', 'module'), m.group(0))
