"""Re-score saved ``run_experiment`` results from their saved predictions (no model is refitted).

    cd analysis
    .venv/Scripts/python.exe -m stage2.scripts.rescore --workers 6            # dry run: what would change, and how
    .venv/Scripts/python.exe -m stage2.scripts.rescore --workers 6 --write    # rewrite them and their registry rows

Why: the harness's accuracy rule changed on 2026-10-28. A P(right) of exactly 0.5 (a tie, up to floating-point
rounding: ``cv.TIE_ATOL``; no hand is claimed, as an abstention or a constant predictor gives) used to count as a
right-hand answer, so every abstention scored as correct on the all-right natural-use set; it now earns half credit
(``cv.credit``), in every accuracy of a result.

Every number of a saved result is a deterministic function of its saved predictions (plus, for the gated metrics,
the cache's ``gate_pass`` of each trial), so this script recomputes them exactly as ``cv.run_experiment`` does, with
the result's own ``n_boot`` and ``seed``. A result whose numbers come out unchanged is left untouched, byte for
byte. A changed result is rewritten with every other field kept (predictions, ``info.created_at``, provenance),
``info['scoring']`` set to the new rule and ``info['rescored']`` recording the rule, the date and the sections that
changed; its rows in the registries that list it are replaced. The list of changed results (old and new headline
numbers) goes to ``reports/stage2/maintenance/rescore_ties.csv`` and ``.json``. Running it again changes nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # run as a file
    import _bootstrap  # type: ignore # noqa: F401

import pandas as pd  # noqa: E402

from stage2 import config as C  # noqa: E402
from stage2 import cv  # noqa: E402

SECTIONS = ('metrics', 'ci95', 'gated_metrics', 'per_person', 'reliability', 'k_attempts', 'extra_metrics')
RULE = cv.SCORING
DATE = '2026-10-28'
MAINT_DIR = C.OUT_DIR / 'maintenance'
#: the headline numbers listed for every changed result (old -> new)
HEADLINE = ('accuracy', 'accuracy_lo', 'accuracy_hi', 'balanced_accuracy', 'coverage', 'accuracy_answered',
            'person_acc_median', 'n_persons_below_80', 'gated_accuracy', 'nat_accuracy', 'nat_coverage',
            'nat_accuracy_answered', 'acc_k1', 'acc_k2', 'acc_k3', 'acc_k4', 'acc_k5')


def result_files(res_dir: Path = C.RES_DIR) -> list[Path]:
    """Every saved run_experiment result (a JSON with predictions and metrics) in results/."""
    out = []
    for p in sorted(Path(res_dir).glob('*.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and 'predictions' in d and 'metrics' in d and 'info' in d:
            out.append(p)
    return out


def gate_map() -> dict[str, bool]:
    """trial_id -> gate_pass of the cache's trial table (what run_experiment saw in ``meta['gate_pass']``)."""
    from stage2.data import load_tables
    t = load_tables()['trials']
    gp = t['gate_pass'].astype('boolean').fillna(False).astype(bool)
    return dict(zip(t['trial_id'].astype(str), gp))


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(cv._jsonable(v), allow_nan=False))


def rescored_sections(d: dict[str, Any], gates: dict[str, bool] | None) -> dict[str, Any]:
    """The SECTIONS of a saved result recomputed from its saved predictions, in their saved JSON form."""
    info = d.get('info') or {}
    pred = pd.DataFrame(d['predictions'])
    n_boot, seed = int(info.get('n_boot', C.N_BOOT)), int(info.get('seed', C.SEED))
    m, ci, pp, rel, kc = cv.evaluate_predictions(pred, n_boot=n_boot, seed=seed)
    out: dict[str, Any] = {'metrics': m, 'ci95': {k: list(v) for k, v in ci.items()},
                           'per_person': pp.to_dict(orient='records'), 'reliability': rel.to_dict(orient='records'),
                           'k_attempts': kc.to_dict(orient='records'), 'gated_metrics': None, 'extra_metrics': None}
    if d.get('gated_metrics') is not None:
        if gates is None:
            raise ValueError('gated metrics need the gate map (stage2 cache)')
        tid = pred['trial_id'].astype(str)
        if not tid.isin(list(gates)).all():
            raise KeyError(f"{d.get('name')}: trials missing from the cache's trial table")
        gp = tid.map(gates).astype(bool).to_numpy()
        y = pred['y'].to_numpy().astype(int)
        p = pred['p_right'].to_numpy(dtype=float)
        g = cv.metrics(y[gp], p[gp])
        g['n_persons'] = float(len(set(pred['person_id'].astype(str).to_numpy()[gp].tolist())))
        g['n_failing_gates'] = float((~gp).sum())
        out['gated_metrics'] = g
    ex = d.get('extra_predictions')
    if ex is not None:
        ep = pd.DataFrame(ex)
        em = cv.metrics(ep['y'], ep['p_right']) if len(ep) else {'n': 0.0}
        em['n_persons'] = float(ep['person_id'].nunique()) if len(ep) else 0.0
        out['extra_metrics'] = em
    return {k: _norm(v) for k, v in out.items()}


def headline(d: dict[str, Any]) -> dict[str, Any]:
    """The HEADLINE numbers of a result in its saved JSON form."""
    m, ci, g, e = d.get('metrics') or {}, d.get('ci95') or {}, d.get('gated_metrics') or {}, d.get('extra_metrics') or {}
    acc_ci = ci.get('accuracy') or [None, None]
    h = {'accuracy': m.get('accuracy'), 'accuracy_lo': acc_ci[0], 'accuracy_hi': acc_ci[1],
         'balanced_accuracy': m.get('balanced_accuracy'), 'coverage': m.get('coverage'),
         'accuracy_answered': m.get('accuracy_answered'), 'person_acc_median': m.get('person_acc_median'),
         'n_persons_below_80': m.get('n_persons_below_80'), 'gated_accuracy': g.get('accuracy'),
         'nat_accuracy': e.get('accuracy'), 'nat_coverage': e.get('coverage'),
         'nat_accuracy_answered': e.get('accuracy_answered')}
    for r in d.get('k_attempts') or []:
        h[f"acc_k{int(r['k'])}"] = r.get('accuracy')
    return {k: h.get(k) for k in HEADLINE}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def rescore_file(path: Path, gates: dict[str, bool] | None, *, write: bool = False) -> dict[str, Any] | None:
    """None when the saved numbers already follow the rule; else the change record (and, with ``write``, the result
    rewritten and its registry rows replaced)."""
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    new = rescored_sections(d, gates)
    changed = [s for s in SECTIONS if not _same(_norm(d.get(s)), new[s])]
    if not changed:
        return None
    old_h = headline(d)
    d2 = dict(d)
    for s in SECTIONS:
        d2[s] = new[s]
    new_h = headline(d2)
    rec = {'name': d['name'], 'sections': changed,
           'n_ties': int(cv.is_tie([r['p_right'] for r in d['predictions']]).sum()),
           'n_ties_natural_use': int(cv.is_tie([r['p_right'] for r in (d.get('extra_predictions') or [])]).sum()),
           'old': old_h, 'new': new_h,
           'changed_numbers': [k for k in HEADLINE if not _same(old_h.get(k), new_h.get(k))]}
    if write:
        _write(path, d, new, changed)
    return rec


def _write(path: Path, d: dict[str, Any], new: dict[str, Any], changed: list[str]) -> None:
    res = cv.ExperimentResult.load(path)
    res.metrics = new['metrics']
    res.ci = {k: tuple(v) for k, v in new['ci95'].items()}
    res.per_person = pd.DataFrame(new['per_person'])
    res.reliability = pd.DataFrame(new['reliability'])
    res.k_curve = pd.DataFrame(new['k_attempts'])
    res.gated_metrics = new['gated_metrics']
    res.extra_metrics = new['extra_metrics']
    res.info = {**res.info, 'scoring': RULE,
                'rescored': {'rule': RULE, 'date': DATE, 'from': 'the saved predictions (no model refitted)',
                             'sections': changed, 'by': 'python -m stage2.scripts.rescore --write'}}
    registries = [p for p in sorted(C.RES_DIR.glob('registry*.csv'))
                  if p.name != 'registry_all.csv' and _lists(p, res.name)]
    res.save(path.parent, registry=None)
    for reg in registries:
        cv.update_registry(res.summary_row(), reg)


def _lists(registry: Path, name: str) -> bool:
    try:
        return name in set(pd.read_csv(registry, usecols=['name'])['name'].astype(str))
    except (OSError, ValueError, KeyError):
        return False


def _work(args: tuple[str, dict[str, bool], bool]) -> dict[str, Any] | None:
    path, gates, write = args
    return rescore_file(Path(path), gates, write=write)


def rescore_all(*, write: bool = False, names: list[str] | None = None, verbose: bool = True,
                workers: int = 1) -> list[dict[str, Any]]:
    """Re-score every saved result (or ``names``); ``workers`` > 1 runs one-thread worker processes (the bootstrap
    of each result takes seconds). Returns the change records, in file order."""
    gates = gate_map()
    files = [p for p in result_files() if names is None or p.stem in names]
    if workers > 1 and len(files) > 1:
        from concurrent.futures import ProcessPoolExecutor
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[v] = '1'                         # inherited by the workers before they load numpy
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            recs = ex.map(_work, [(str(p), gates, write) for p in files], chunksize=1)
            pairs = list(zip(files, recs))
    else:
        pairs = [(p, rescore_file(p, gates, write=write)) for p in files]
    out = []
    for p, rec in pairs:
        if rec:
            out.append(rec)
            if verbose:
                ch = ', '.join(f"{k} {_fmt(rec['old'][k])} -> {_fmt(rec['new'][k])}" for k in rec['changed_numbers'])
                print(f"  {p.stem}: {ch or 'per-person / bootstrap details only'}", flush=True)
    if verbose:
        print(f'{len(files)} results checked, {len(out)} changed' + ('' if write else ' (dry run: nothing written)'),
              flush=True)
    return out


def _fmt(v: Any) -> str:
    return 'none' if v is None else (f'{v:.4f}' if isinstance(v, float) else str(v))


def write_report(records: list[dict[str, Any]], folder: Path = MAINT_DIR) -> tuple[Path, Path]:
    """maintenance/rescore_ties.json (every record) and .csv (one row per changed result: old and new numbers)."""
    from stage2.report_export import write_csv_atomic, write_text_atomic
    rows = []
    for r in records:
        row = {'name': r['name'], 'n_ties': r['n_ties'], 'n_ties_natural_use': r['n_ties_natural_use'],
               'sections': ';'.join(r['sections']), 'changed_numbers': ';'.join(r['changed_numbers'])}
        for k in HEADLINE:
            row[f'old_{k}'], row[f'new_{k}'] = r['old'].get(k), r['new'].get(k)
        rows.append(row)
    cols = ['name', 'n_ties', 'n_ties_natural_use', 'sections', 'changed_numbers',
            *[f'{w}_{k}' for k in HEADLINE for w in ('old', 'new')]]
    doc = {'rule': RULE, 'date': DATE, 'what': 'saved results re-scored from their saved predictions (no refit); '
           'P(right) = 0.5 exactly now earns half credit in every accuracy', 'results': records}
    j = write_text_atomic(folder / 'rescore_ties.json', json.dumps(doc, indent=1, allow_nan=False) + '\n')
    c = write_csv_atomic(pd.DataFrame(rows, columns=cols), folder / 'rescore_ties.csv', index=False)
    return j, c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='rewrite the changed results and their registry rows')
    ap.add_argument('--names', nargs='+', default=None, help='only these results')
    ap.add_argument('--workers', type=int, default=1, help='one-thread worker processes (default 1)')
    a = ap.parse_args(argv)
    recs = rescore_all(write=a.write, names=a.names, workers=a.workers)
    if a.write and a.names is None and recs:          # a later run that changes nothing keeps the first record
        j, c = write_report(recs)
        print(f'changes listed in {j.relative_to(C.OUT_DIR).as_posix()} and {c.name}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
