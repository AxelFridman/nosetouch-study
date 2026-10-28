"""Lane p1: preprocessing and data-treatment decisions, judged by a fixed panel of three models. Steps:

    cd analysis
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes p1            # missing / stale experiments, then derive
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes p1 --only "p1_gbm_*" --force
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes p1 --only p1_derive   # re-derive tables from saved results
    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes p1            # redraw every P1 figure (about a minute)
    .venv/Scripts/python.exe -m stage2.pipeline tables --lanes p1
    .venv/Scripts/python.exe -m stage2.pipeline numbers --lanes p1            # numbers, finalists, p1_SUMMARY.md

compute: (1) the experiment grid (:mod:`p1_experiments`) in two single-thread worker processes (the lane's
2-thread budget; LR + GBM, which builds the variant caches, and MiniRocket); (2) the calibration study
(:mod:`p1_calibration`; the nested-calibration headline runs with the leak check on every fold); (3) the registry
rebuilt from the saved results; (4) :func:`p1_derive.derive` (statistics in :mod:`p1_stats`), which writes every
derived file the presentation steps read. figures / tables / numbers (:mod:`p1_figures`, :mod:`p1_tables`,
:mod:`p1_numbers`, :mod:`p1_summary`, labels in :mod:`p1_labels`) only read those files and import no model code.
"""
from __future__ import annotations

import os

from stage2.pipeline import Lane, limit_threads, selected

limit_threads(min(2, int(os.environ.get('OMP_NUM_THREADS', '2') or 2)))   # the lane's budget (respects --threads 1)

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

L = Lane('p1')
ORDER = 150
DEPENDS = ()
DERIVED = ('p1_runs', 'p1_comparisons', 'p1_subsets', 'p1_best', 'p1_calibration', 'p1_decisions', 'p1_filter_delay',
           'p1_variants')


def _calib_names() -> list[tuple[str, str, str]]:
    from stage2.lanes.p1_calibration import AUTO, METHODS
    from stage2.lanes.p1_labels import CALIB_MODELS
    return [(f'p1_calib_{m}_{meth}', m, meth) for m in CALIB_MODELS for meth in (*METHODS, AUTO)]


def _calib_spec(model: str, method: str) -> dict:
    from stage2.lanes import p1_calibration as K
    from stage2.lanes.p1_panel import DEFAULT, describe
    spec = {'model': model, 'method': method, 'panel': describe(model, DEFAULT), 'k_agg': list(K.K_AGG),
            'n_draws': K.N_DRAWS, 'eps': K.EPS}
    if method == K.AUTO:
        spec |= {'candidates': list(K.AUTO_CANDIDATES), 'auto_splits': K.AUTO_SPLITS, 'leak_check': _leak_mode(method)}
    return spec


def _leak_mode(method: str) -> str:
    """The nested-calibration runs are headline rows: leak check on every fold; the fixed calibrators
    (exploratory) on the harness's sampled folds."""
    from stage2.lanes.p1_calibration import AUTO
    return 'all' if method == AUTO else 'sample'


def _calib_needs_run(name: str, model: str, method: str, force: bool, only: list[str] | None) -> bool:
    import hashlib
    from stage2.lanes.p1_experiments import stored_params
    if not selected(name, only):
        return False
    if force or L.should_run(name, False, None):
        return True
    sha = hashlib.sha256(json.dumps(_calib_spec(model, method), sort_keys=True, default=str).encode()).hexdigest()
    return (stored_params(name) or {}).get('p1_spec_sha256') != sha


def run_calibration(todo: list[tuple[str, str, str]]) -> None:
    import hashlib
    import warnings

    from stage2.data import load_cache
    from stage2.lanes.p1_calibration import make_calibrated
    d, nat = load_cache(), load_cache(('natural_use',))
    for name, model, method in todo:
        spec = _calib_spec(model, method)
        sha = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
        t0 = time.time()
        from stage2.lanes.p1_calibration import AUTO_CANDIDATES
        how = (f"the calibrator ({', '.join(AUTO_CANDIDATES)}; rank-preserving only) chosen inside each fold by "
               "grouped CV on the training persons' cross-fitted predictions, then" if method == 'auto' else method)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = L.run_experiment(name, make_calibrated(model, method), d.approach, d.trials,
                                 extra=(nat.approach, nat.trials), model=f'p1_{model}+{method}',
                                 params={'p1_spec_sha256': sha, 'spec': spec}, leak_check=_leak_mode(method),
                                 registry=None,
                                 notes=f'calibration: {how} fitted on inner cross-fitted predictions of the training '
                                       f'persons (fold.inner_splits()); base panel model {model}')
        print(f'  {name}: acc {100 * r.metrics["accuracy"]:.1f} % log loss {r.metrics["log_loss"]:.3f} '
              f'ECE {r.metrics["ece"]:.3f} ({r.leak_check}; {time.time() - t0:.0f} s)', flush=True)


def _workers(todo: list[str]) -> None:
    """The grid in two single-thread worker processes (logs in reports/stage2/p1_logs/)."""
    from stage2 import config as C
    from stage2.lanes.p1_experiments import worker_queues
    log_dir = C.OUT_DIR / 'p1_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[v] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    procs = []
    for i, (names, build) in enumerate(worker_queues(todo)):
        cmd = [sys.executable, '-m', 'stage2.lanes.p1_experiments', *names] + ([] if build else ['--no-build'])
        log = open(log_dir / f'worker_{i}.log', 'w', encoding='utf-8')
        procs.append((subprocess.Popen(cmd, cwd=str(C.ANALYSIS_DIR), env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f'p1: worker {i} runs {len(names)} experiment(s) (log: p1_logs/worker_{i}.log)', flush=True)
    failed = []
    for i, (p, log) in enumerate(procs):
        if p.wait() != 0:
            failed.append(i)
        log.close()
    if failed:
        raise RuntimeError(f'p1: worker(s) {failed} failed; see reports/stage2/p1_logs/')


def compute(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import p1_experiments as X
    from stage2.pipeline import file_current
    todo = [e.name for e in X.GRID if X.needs_run(e, force, only)]
    if todo:
        print(f'p1: {len(todo)} experiment(s) to run', flush=True)
        _workers(todo)
    calib = [t for t in _calib_names() if _calib_needs_run(*t, force, only)]
    if calib:
        run_calibration(calib)
    stale = any(not file_current(L.tab_dir / f'{n}.json') for n in DERIVED)
    if todo or calib or stale or force or selected('p1_derive', only):
        from stage2.lanes import p1_derive
        X.rebuild_registry()
        t0 = time.time()
        p1_derive.derive()
        print(f'p1: derived tables written ({time.time() - t0:.0f} s)', flush=True)
    else:
        print('p1: results and derived tables current; skipped (--force recomputes)', flush=True)


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import p1_figures
    p1_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import p1_tables
    p1_tables.write(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import p1_tables
    p1_tables.numbers()


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
