"""Lane m3: time-frequency and deep sequence models (1D CNN, ResNet1D, InceptionTime-lite, TCN, GRU, Transformer,
2D CNN on spectrograms / scalograms, MiniRocket) against engineered features.

Steps (python -m stage2.pipeline <step> --lanes m3):
  compute  base experiments (stage2.lanes.m3_experiments.SPECS) whose results are missing or stale, then the
           data-driven second phase (each family's best model, its calibration remedies, the nested deep choice),
           then the derived files every figure / table / number reads (tables/m3/*.json|csv, saliency .npz);
  figures  stage2.lanes.m3_figures (saved files only);
  tables   CSV + booktabs .tex of the report tables, and the lane registry rebuilt from the saved results;
  numbers  results/m3_numbers.json, then reports/stage2/m3_SUMMARY.md generated from the saved files.

Threads: 3 (``M3_THREADS`` overrides). Every deep network trains with ONE intra-op thread whatever this setting
(``m3_deep.torch_runtime``: fits are then identical on any machine), so the lane's budget is used by processes: the
compute step runs the missing base experiments in ``M3_WORKERS`` worker processes of one thread each (default: the
thread budget; ``M3_WORKERS=1`` runs them in this process), balanced by their expected cost, then the second phase and
the derived files here. One stream by hand: ``python -m stage2.scripts.run_m3 --threads 1 --names m3_cnn m3_gru``."""
from __future__ import annotations

import os

from stage2.pipeline import Lane, limit_threads

THREADS = 3                                                     # the lane's thread budget
# M3_THREADS wins; otherwise a smaller cap set before this import (python -m stage2.pipeline --threads N) is kept
_n = int(os.environ.get('M3_THREADS') or min(THREADS, int(os.environ.get('OMP_NUM_THREADS') or THREADS)))
os.environ['M3_THREADS'] = str(_n)
limit_threads(_n)
#: worker processes (one thread each) for the base experiments of the compute step
WORKERS = max(1, int(os.environ.get('M3_WORKERS') or _n))

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stage2 import report_export as RX  # noqa: E402

L = Lane('m3')
ORDER = 130
DEPENDS = ()


# ───────────────────────────── compute: base experiments ─────────────────────────────

_DATA: dict[str, Any] = {}


def data() -> tuple[Any, Any]:
    if 'd' not in _DATA:
        from stage2.data import load_cache
        _DATA['d'] = load_cache()
        _DATA['nat'] = load_cache(('natural_use',))
    return _DATA['d'], _DATA['nat']


def inputs(kind: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-trial inputs of the primary and the natural-use rows (built once per process)."""
    from stage2.lanes import m3_experiments as E
    key = f'x_{kind}'
    if key not in _DATA:
        d, nat = data()
        _DATA[key] = (E.build_x(kind, d), E.build_x(kind, nat))
    return _DATA[key]


def run_spec(spec: Any, *, calib: str = 'none', name: str | None = None, n_configs_tried: int = 1,
             notes: str | None = None, verbose: bool = True) -> Any:
    from stage2.lanes import m3_experiments as E
    d, nat = data()
    X, Xn = inputs(spec.x)
    name = name or spec.name
    fp = E.fit_predict_of(spec, calib)
    t0 = time.time()
    params = {'family': spec.family, 'x': spec.x, 'calib': calib, 'base_experiment': spec.name, 'group': spec.group,
              'post_hoc': bool(spec.post_hoc),
              **{k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.params.items()}}
    r = L.run_experiment(name, fp, X, d.trials, extra=(Xn, nat.trials), model=spec.label, params=params,
                         n_configs_tried=n_configs_tried, notes=notes if notes is not None else spec.notes,
                         leak_check='all')
    if verbose:
        print(f'  {name}: accuracy {100 * r.metrics["accuracy"]:.1f} % AUC {r.metrics["roc_auc"]:.3f} '
              f'log loss {r.metrics["log_loss"]:.3f} ({r.leak_check}; {time.time() - t0:.0f} s)', flush=True)
    return r


#: CPU-minutes of one LOSO run when no saved result says better (deep fixed-window and recurrent nets cost more)
DEFAULT_COST = {'deep': 9.0, 'body': 9.0, 'gripsub': 8.0, 'gripsub_body': 8.0, 'mr': 4.0, 'tflog': 1.0}


def expected_cost(name: str) -> float:
    """Expected CPU-minutes of a base experiment: its saved training cost, else a family default (LPT balancing)."""
    from stage2.lanes import m3_experiments as E
    p = L.result_path(name)
    if p.exists():
        try:
            from stage2.lanes.m3_derive import cost_cpu_min
            c = cost_cpu_min(L.load_result(name))
            if c > 0:
                return c
        except Exception:                                      # an unreadable result: fall back to the default
            pass
    s = E.BY_NAME[name]
    return DEFAULT_COST.get(s.family, 5.0) * (2.5 if 'fixed' in s.x or s.params.get('arch') in ('gru', 'inception') else 1)


def streams(names: list[str], k: int) -> list[list[str]]:
    """``names`` in k streams of about equal expected cost (longest job first onto the lightest stream)."""
    load, out = [0.0] * k, [[] for _ in range(k)]
    for n in sorted(names, key=lambda n: (-expected_cost(n), n)):
        i = min(range(k), key=lambda j: (load[j], j))
        out[i].append(n)
        load[i] += expected_cost(n)
    return [s for s in out if s]


def compute_base(names: list[str] | None = None, force: bool = False, only: list[str] | None = None,
                 workers: int | None = None) -> list[str]:
    """Run the base experiments (all, or ``names``) that are missing or stale; with ``workers`` > 1 (default
    ``WORKERS``) in worker processes of one thread each (``stage2.scripts.run_m3``)."""
    from stage2.lanes import m3_experiments as E
    todo = [n for n in (names or E.BASE) if L.should_run(n, force, only)]
    workers = WORKERS if workers is None else max(1, int(workers))
    if workers > 1 and len(todo) > 1:
        return _run_streams(todo, force, workers)
    done = []
    for n in todo:
        if not force and not L.should_run(n, force, only):     # made meanwhile by a parallel process
            continue
        run_spec(E.BY_NAME[n])
        done.append(n)
    return done


def _run_streams(todo: list[str], force: bool, workers: int) -> list[str]:
    root = Path(__file__).resolve().parents[2]                 # analysis/
    env = {**os.environ, 'M3_WORKERS': '1', 'M3_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'}
    procs = []
    for s in streams(todo, workers):
        cmd = [sys.executable, '-m', 'stage2.scripts.run_m3', '--threads', '1', '--names', *s] + (['--force'] if force else [])
        print(f"m3: worker stream ({len(s)} experiments, about {sum(expected_cost(n) for n in s):.0f} CPU-min): "
              f"{' '.join(s)}", flush=True)
        procs.append((s, subprocess.Popen(cmd, cwd=root, env=env)))
    failed = [s for s, p in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f'm3: worker streams failed: {failed}')
    return todo


def compute(force: bool = False, only: list[str] | None = None) -> None:
    """Base experiments, phase 2, derived files. When an experiment ran, every derived file is refreshed (each one
    reads many results); otherwise ``only`` may name derived files (``--only m3_saliency --force``)."""
    ran = compute_base(None, force, only)
    from stage2.lanes import m3_derive
    ran += m3_derive.compute_phase2(force, only)
    m3_derive.derive(force=force or bool(ran), only=None if ran else only)


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m3_figures
    m3_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m3_derive
    m3_derive.tables(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    """results/m3_numbers.json, then reports/stage2/m3_SUMMARY.md (generated from the saved files only)."""
    from stage2.lanes import m3_derive, m3_summary
    m3_derive.numbers()
    m3_summary.write_summary()


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
