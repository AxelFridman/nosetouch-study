"""Lane m1: interpretable models, and why the hand is detectable. Steps: compute, figures, tables, numbers.

    cd analysis
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1            # missing / stale experiments (about an hour
                                                                              # on one thread; M1_THREADS=1 runs two
                                                                              # processes side by side)
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1 --only "m1_lr_*" --force
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1 --only m1_catalogue m1_derived --force
    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes m1            # 1-5 min, from saved files only (no model is fitted)
    .venv/Scripts/python.exe -m stage2.pipeline tables --lanes m1
    .venv/Scripts/python.exe -m stage2.pipeline numbers --lanes m1

compute: every experiment of ``m1_experiments.SPECS`` whose result is missing or stale (``--only`` selects names,
shell patterns allowed; ``--force`` reruns), the per-fold artefacts of the experiments that store them, the
feature catalogue (pseudo-name ``m1_catalogue``) and the derived tables (pseudo-name ``m1_derived``; rerun
whenever an experiment ran; it also writes results/finalists_m1.json). figures / tables / numbers read the saved
files only. ``M1_SKIP_DERIVE=1`` skips the derived tables (for partial parallel runs; run ``--only m1_derived``
afterwards).
"""
import os

from stage2.pipeline import Lane, limit_threads, selected

#: this lane's thread budget (2), set BEFORE numpy loads; M1_THREADS=1 runs two lane processes side by side within it
limit_threads(int(os.environ.get('M1_THREADS', 2)))

import time                           # noqa: E402

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

from stage2 import pipeline as P      # noqa: E402

L = Lane('m1')
ORDER = 110
DEPENDS = ()
CATALOGUE = 'm1_catalogue'
DERIVED = 'm1_derived'
#: the derived files the presentation steps read (all must be current, else compute re-derives)
DERIVED_FILES = ('m1_models', 'm1_strata', 'm1_calibration', 'm1_k_aggregation', 'm1_failures', 'm1_natural',
                 'm1_drift', 'm1_grid', 'm1_sensitivity', 'm1_tree_summary', 'm1_tf_yaw_profile', 'm1_tf_multi_profile',
                 'm1_gam_shapes', 'm1_phys6_coefficients', 'm1_small_coefficients', 'm1_physics_curves',
                 'm1_key_features')


def _inputs(d, nat, which):
    from stage2.lanes import m1_models as M
    return {b: (getattr(M, b)(d), getattr(M, b)(nat)) for b in dict.fromkeys(which)}


def compute(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m1_analysis as A
    from stage2.lanes import m1_experiments as E
    todo = [s for s in E.SPECS if L.should_run(s.name, force, only)
            or (s.store and selected(s.name, only) and not P.file_current(A.store_path(s.name)))]
    run_cat = selected(CATALOGUE, only) and (force or not P.file_current(L.tab_dir / f'{CATALOGUE}.json'))
    stale = [f for f in DERIVED_FILES if not P.file_current(L.tab_dir / f'{f}.json')]
    # derived tables: after any experiment ran, or when selected and forced / stale / the catalogue is redone
    run_der = bool(todo) or (selected(DERIVED, only) and (force or bool(stale) or run_cat))
    if not (todo or run_cat or run_der):
        print('m1: results current; skipped (--force recomputes)', flush=True)
        return
    from stage2 import cv
    from stage2.data import load_cache
    from stage2.lanes import m1_models as M
    d, nat = load_cache(), load_cache(('natural_use',))
    X = _inputs(d, nat, ['build_x'] + [s.x for s in todo])
    swap = A.swap_blocks(d) if any(s.dynamic for s in todo) else []
    for s in todo:
        t0 = time.time()
        s = E.with_dynamic(s, swap)
        store = {} if s.store else None
        params = dict(s.params)
        fp = getattr(M, s.factory)(**params, **({'store': store} if s.store else {}))
        Xp, Xn = X[s.x]
        r = L.run_experiment(s.name, fp, Xp, d.trials, extra=(Xn, nat.trials), model=s.factory,
                             params=E.spec_params_json(s), notes=s.notes, n_configs_tried=s.n_configs)
        if s.store:
            L.write_json(A.store_path(s.name).stem, {'name': s.name, 'folds': {str(k): v for k, v in sorted(store.items())}})
        print(f'  {s.name}: accuracy {100 * r.metrics["accuracy"]:.1f} % AUC {r.metrics["roc_auc"]:.3f} '
              f'log loss {r.metrics["log_loss"]:.3f} ({r.leak_check}; {time.time() - t0:.0f} s)', flush=True)
    if run_cat and (force or not P.file_current(L.tab_dir / f'{CATALOGUE}.json')):   # re-checked: may be done meanwhile
        t0 = time.time()
        A.evaluate_catalogue(d, nat, X['build_x'][0], X['build_x'][1])
        print(f'  catalogue: {time.time() - t0:.0f} s', flush=True)
    if run_der and not os.environ.get('M1_SKIP_DERIVE'):
        t0 = time.time()
        A.derive(d, nat, X['build_x'][0], X['build_x'][1])      # writes the finalists after the model table
        print(f'  derived tables: {time.time() - t0:.0f} s', flush=True)
    _ = (cv, np, pd)


# ───────────────────────────── finalists ─────────────────────────────

#: lane m1's finalists: the best rule, the best small logistic model and the best tree / GAM. The choice is made in
#: ``m1_select.choose_finalists`` from the saved model table by fixed criteria (see its docstring); ``derive``
#: writes them right after the model table.


# ───────────────────────────── presentation ─────────────────────────────


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m1_figures
    m1_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m1_tables
    m1_tables.write(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    """results/m1_numbers.json and reports/stage2/m1_SUMMARY.md, rendered together from the saved files: every
    number of the summary and of the figure texts is registered in the numbers file as it is rendered."""
    from stage2.lanes import m1_summary
    m1_summary.write()


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
