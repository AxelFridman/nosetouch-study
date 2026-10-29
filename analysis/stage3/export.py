"""``export_models``: the production models as JSON files for the web page (bundled into the predict chunk).

Files written to ``app/src/predict/models/`` (no participant data: model parameters, the data identity and pooled
evaluation numbers only):

* ``m4_production.json`` - lane m4's production model fitted on all persons (``m4_production.fit_production``): the
  detector bundle (segmentation parameters, filter coefficients recomputed with scipy, the segment classifier, the
  attempt rule, the stopping rule, the grip check, constants), the yaw8 hand model, its nested calibrator and the
  session-fitted aggregation (docs/stage3/PORT_SPEC.md 4.2-4.3);
* ``final_cnn.json``     - ``reports/stage2/export/final_model.json`` (lane final's all-persons export), minified with
  sorted keys and otherwise unchanged (every number keeps its float64 value);
* ``phys6.json``         - lane m1's ``m1_lr_phys6_nested`` fitted on all persons (PORT_SPEC 4.4);
* ``yaw_ref.json``       - the registry reference ``example_yaw_logreg`` fitted on all persons (PORT_SPEC 4.5);
* ``manifest.json``      - model ids, UI labels, the sha256 and size of every file, the data identity, the library
  versions, and each model's leave-one-person-out accuracy with its 95 % CI from the Stage 2 numbers files.

Numbers are written with Python's float repr (round-trips float64); NaN never occurs (``allow_nan=False``).
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from stage2 import config as C

from . import runtime as RT

APP_MODELS_DIR = C.REPO_DIR / 'app' / 'src' / 'predict' / 'models'
FINAL_MODEL = C.OUT_DIR / 'export' / 'final_model.json'
FORMATS = {'m4': 'nosetouch-m4-production-1', 'phys6': 'nosetouch-phys6-1', 'yaw': 'nosetouch-yaw-ref-1'}
#: UI labels (APP_SPEC 5.2) and the Stage 2 result each model reproduces
MODELS = {
    'cnn': {'label': 'Final CNN', 'short': '1D CNN, 4 members', 'stage2Name': 'm3_cal_deep_platt', 'kind': 'cnn1d_ensemble',
            'isDefault': True},
    'm4': {'label': 'Detector model', 'short': 'lane m4, yaw logistic', 'stage2Name': 'm4_cal_nested', 'kind': 'logistic',
           'isDefault': False},
    'phys6': {'label': 'Physics logistic', 'short': 'PHYS6: yaw, roll, lateral', 'stage2Name': 'm1_lr_phys6_nested',
              'kind': 'logistic', 'isDefault': False},
    'yaw': {'label': 'Yaw reference', 'short': 'logistic on yaw, 8 instants', 'stage2Name': 'example_yaw_logreg',
            'kind': 'logistic', 'isDefault': False},
}
#: the leave-one-person-out accuracy each model reports (numbers file, key prefix): the final lane's master table puts
#: the CNN, PHYS6 and the reference on the same footing (oracle windows); lane m4's number is end to end, cue-free
LOSO = {'cnn': ('final_numbers.json', 'final_accuracy'), 'm4': ('m4_numbers.json', 'prod_acc'),
        'phys6': ('final_numbers.json', 'm1_phys6_accuracy'), 'yaw': ('final_numbers.json', 'ref_accuracy')}
LOSO_NOTE = {'cnn': 'one approach, the app\'s stored window (oracle bounds)',
             'm4': 'one recording end to end, cue-free (the detector picks the movement; an abstention counts half)',
             'phys6': 'one approach, the app\'s stored window (oracle bounds)',
             'yaw': 'one approach, the app\'s stored window (oracle bounds)'}


def dumps(doc: Any, *, minify: bool) -> str:
    if minify:
        return json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    return json.dumps(doc, sort_keys=True, indent=1, allow_nan=False) + '\n'


def _write(path: Path, text: str) -> dict[str, Any]:
    data = text.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)
    return {'file': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def identity() -> dict[str, Any]:
    """The data identity of the cache the models were fitted on and the code fingerprints. No git commit or timestamp:
    the same data and code give the same bytes (the code fingerprint covers stage3 and the Stage 2 modules it calls)."""
    from stage2.pipeline import cache_meta, provenance, source_fingerprint
    m = cache_meta() or {}
    code = dict(m.get('code') or {})
    code['stage3_fit_sha16'] = source_fingerprint('stage3.fit', 'stage3.runtime', 'stage3.export',
                                                  'stage2.lanes.m4_production', 'stage2.lanes.m4_detect',
                                                  'stage2.lanes.m4_e2e', 'stage2.lanes.m4_models',
                                                  'stage2.lanes.m1_models', 'stage2.lanes.m1_features',
                                                  'stage2.scripts.reference', 'stage2.standardize')
    return {**provenance(), 'code': code}


def versions() -> dict[str, str]:
    import pandas
    import scipy
    import sklearn
    return {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__, 'pandas': pandas.__version__}


def loso_numbers() -> dict[str, Any]:
    out = {}
    for mid, (fname, key) in LOSO.items():
        d = json.loads((C.RES_DIR / fname).read_text(encoding='utf-8'))
        v, lo, hi = d.get(key), d.get(f'{key}_lo'), d.get(f'{key}_hi')
        out[mid] = {'accuracy': None if v is None else v['value'], 'ci95': [None if lo is None else lo['value'],
                                                                             None if hi is None else hi['value']],
                    'what': LOSO_NOTE[mid], 'source': f'reports/stage2/results/{fname}#{key}',
                    'description': None if v is None else v.get('description')}
    return out


def _training(n_trials: int, persons: list[str]) -> dict[str, Any]:
    return {'n_trials': int(n_trials), 'n_persons': len(persons), 'persons': list(persons),
            'cap_per_person': C.CAP_PER_PERSON, 'seed': C.SEED, 'sets': ['primary']}


def model_docs(fitted: dict[str, Any], ident: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{model id: JSON document} (the CNN document is final_model.json as exported by lane final)."""
    m4 = RT.production_m4(fitted['m4'])          # the page's grip gate (runtime.PRODUCTION_GRIP_GATE_UPZ)
    from stage2.data import load_cache
    d = load_cache()
    from stage2 import cv
    g = d.trials['person_id'].astype(str).to_numpy()
    n_m4 = int(cv.cap_rows(g, d.y.astype(int), C.CAP_PER_PERSON, C.SEED).size)
    docs = {
        'm4': {'format': FORMATS['m4'], 'id': 'm4', **{k: MODELS['m4'][k] for k in ('label', 'short', 'stage2Name')},
               'identity': ident, 'training': _training(n_m4, sorted(m4.trained_on)), 'note': RT.NOTE_THRESHOLDS,
               **RT.m4_to_json(m4)},
        'phys6': {'format': FORMATS['phys6'], 'id': 'phys6',
                  **{k: MODELS['phys6'][k] for k in ('label', 'short', 'stage2Name')}, 'identity': ident,
                  'training': _training(fitted['phys6']['n_trials'], fitted['phys6']['persons']),
                  **RT.phys6_to_json(fitted['phys6'])},
        'yaw': {'format': FORMATS['yaw'], 'id': 'yaw', **{k: MODELS['yaw'][k] for k in ('label', 'short', 'stage2Name')},
                'identity': ident, 'training': _training(fitted['yaw']['n_trials'], fitted['yaw']['persons']),
                **RT.yaw_to_json(fitted['yaw'])},
    }
    docs['cnn'] = json.loads(FINAL_MODEL.read_text(encoding='utf-8'))
    return docs


def cnn_float32_check(cnn: dict[str, Any], windows: np.ndarray) -> dict[str, Any]:
    """Would float32 decimal strings for the CNN weights do? Every number of ``model`` written as the shortest
    float32 decimal and read back as float64 (what JSON.parse does without Math.fround): max |dP| on ``windows``."""
    from stage2.lanes import final_infer as FI

    def to32(v: Any) -> Any:
        if isinstance(v, list):
            return [to32(x) for x in v]
        if isinstance(v, dict):
            return {k: to32(x) for k, x in v.items()}
        if isinstance(v, float):
            return float(str(np.float32(v)))          # the shortest decimal that round-trips float32
        return v
    m32 = {**cnn, 'model': to32(cnn['model'])}
    exact32 = all(float(np.float32(x)) == x for x in _floats(cnn['model']))
    p64 = FI.predict_proba(cnn, windows)
    p32 = FI.predict_proba(m32, windows)
    size64 = len(dumps(cnn, minify=True).encode())
    size32 = len(dumps(m32, minify=True).encode())
    return {'max_abs_dP': float(np.max(np.abs(p64 - p32))), 'n_windows': int(len(windows)),
            'weights_exactly_float32': bool(exact32), 'bytes_float64': size64, 'bytes_float32_strings': size32}


def _floats(v: Any):
    if isinstance(v, list):
        for x in v:
            yield from _floats(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _floats(x)
    elif isinstance(v, float):
        yield v


def export_models(out_dir: str | Path = APP_MODELS_DIR, *, refit: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Fit (or load the cached fits), write every model file and the manifest; returns the manifest."""
    from . import fit as F
    t0 = time.time()
    fitted = F.fitted_models(refit=refit, verbose=verbose)
    t_fit = time.time() - t0
    out_dir = Path(out_dir)
    ident = identity()
    docs = model_docs(fitted, ident)
    files = {}
    for mid in RT.MODEL_IDS:
        files[mid] = _write(out_dir / RT.FILES[mid], dumps(docs[mid], minify=(mid == 'cnn')))
    loso = loso_numbers()
    manifest = {
        'format': 'nosetouch-models-manifest-1',
        'generated_by': 'analysis/stage3 export_models (python -m stage3 export)',
        'default': 'cnn', 'order': list(RT.MODEL_IDS), 'detector': 'm4',
        'note': RT.NOTE_THRESHOLDS,
        'identity': ident, 'versions': versions(),
        'models': [{'id': mid, **MODELS[mid], **files[mid], 'format': docs[mid].get('format'),
                    'calibration': _calibration(mid, docs[mid]), 'aggregation': _aggregation(mid, docs[mid]),
                    'trainedOn': {'persons': docs[mid]['training']['n_persons'],
                                  'trials': docs[mid]['training']['n_trials']},
                    'loso': loso[mid]} for mid in RT.MODEL_IDS],
        'fit_seconds': fitted.get('seconds'),
    }
    files['manifest'] = _write(out_dir / 'manifest.json', dumps(manifest, minify=False))
    if verbose:
        for k, v in files.items():
            print(f'stage3: wrote {v["file"]} ({v["bytes"] / 1024:.1f} kB)', flush=True)
        print(f'stage3: export done ({time.time() - t0:.0f} s, fitting or loading {t_fit:.0f} s)', flush=True)
    return manifest


def _calibration(mid: str, doc: dict[str, Any]) -> str:
    c = doc['calibrator']
    if c['method'] == 'nested':
        return f"nested:{c['chosen']}"
    return c['method']


def _aggregation(mid: str, doc: dict[str, Any]) -> dict[str, Any]:
    if mid == 'cnn':
        return {'kind': 'sum_over_tau', 'tau': doc['aggregation']['tau']}
    a = doc['aggregation']
    return {k: a[k] for k in a if k not in ('rule',)}
