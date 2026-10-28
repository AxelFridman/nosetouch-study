"""Lane final: the exported final model and its numpy reference implementation."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from stage2 import config as C

EXPORT = C.OUT_DIR / 'export' / 'final_model.json'
CHECK = C.OUT_DIR / 'export' / 'final_model_check.json'


def test_reference_implementation_imports_numpy_only():
    src = Path(__file__).resolve().parents[1] / 'lanes' / 'final_infer.py'
    tree = ast.parse(src.read_text(encoding='utf-8'))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split('.')[0])
    assert mods <= {'numpy', 'json', 'pathlib', 'typing', '__future__'}, mods


def test_export_is_plain_json_with_every_part():
    if not EXPORT.exists():
        pytest.skip('no export yet (python -m stage2.pipeline compute --lanes final)')
    doc = json.loads(EXPORT.read_text(encoding='utf-8'))
    for k in ('format', 'identity', 'input', 'model', 'calibrator', 'aggregation', 'screen_check', 'prior_right',
              'decision', 'training'):
        assert k in doc, k
    assert doc['identity'].get('data_id')
    assert doc['prior_right'] == 0.5


def test_reference_reproduces_the_fitted_pipeline_on_every_trial():
    if not (EXPORT.exists() and CHECK.exists()):
        pytest.skip('no export yet')
    from stage2.data import load_cache
    from stage2.lanes import final_infer as FI
    doc = FI.load(EXPORT)
    chk = json.loads(CHECK.read_text(encoding='utf-8'))
    want = {(r['set'], r['trial_key']): r['p_fitted'] for r in chk['rows']}
    n = 0
    for which, sets in (('primary', ('primary',)), ('natural_use', ('natural_use',))):
        d = load_cache(sets)
        p = FI.predict_proba(doc, np.asarray(d.approach, dtype=np.float32))
        for tk, v in zip(d.trials['trial_key'].astype(str), p):
            assert abs(v - want[(which, tk)]) <= 1e-9, (which, tk, v, want[(which, tk)])
            n += 1
    assert n == len(want)


def test_aggregate_prior_and_screen_check():
    if not EXPORT.exists():
        pytest.skip('no export yet')
    from stage2.lanes import final_infer as FI
    doc = FI.load(EXPORT)
    assert abs(FI.aggregate(doc, np.array([0.7])) - 0.7) < 1e-12
    p3 = FI.aggregate(doc, np.array([0.7, 0.7, 0.7]))
    assert 0.7 < p3 < 1.0
    assert np.allclose(FI.apply_prior(np.array([0.5]), 0.5), 0.5)
    assert FI.apply_prior(np.array([0.5]), 0.8)[0] > 0.5
    A = np.ones((2, 17, 64), dtype=np.float32)
    assert np.array_equal(FI.unflip(doc, FI.unflip(doc, A)), A)
    assert list(FI.screen_away(doc, np.array([0.3, -0.3]))) == [False, True]
