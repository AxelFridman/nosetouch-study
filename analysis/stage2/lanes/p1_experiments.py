"""Lane p1: the experiment grid (data-treatment option x panel model) and its runner.

Every experiment is one LOSO run (``cv.run_experiment`` through ``Lane.run_experiment``) of a FIXED panel model
(:mod:`p1_panel`) on one data treatment, and is compared PAIRED (same trials, same folds) with the same model
under the pipeline's current choice (``p1_<model>_base``). Families and options:

==============  ======================================================================================
family          options (default first)
==============  ======================================================================================
filter          zero-phase Butterworth 15 Hz order 4 | none, 5, 10, 20 Hz | order 2, 8 | causal
gaps            bridge <= 100 ms, linear | 50, 250 ms | cubic, sample-and-hold | E4d trials kept
sampling        60 Hz native precision | 30, 20 Hz | iOS quantised to 0.1 | sample-time jitter 2, 5, 10 ms
gravity         gyro-propagated gravity_v1 | 1-Hz low-pass of accg | browser accg - acc
frame           model default channel set | chiral / gravity / device / all channels
window          approach, time-normalised | first half | approach + hold | 0.5-s pre-onset context | fixed 3 s
segmentation    oracle bounds | bounds jittered +-50/100/200 ms at test time | onset 100 ms late | jitter-trained
normalisation   none | per-channel z-score | median / IQR | per-trial peak
outliers        none | winsorised 0.5/99.5 % | flagged trials out of training | swap blocks out of training |
                E4b/E4c loosened (excluded trials kept) | tightened (x2) in training
platform        pooled | train on the other platform only | on the same platform only
augmentation    mirror | no mirror | + screen-away flips | + time warp
==============  ======================================================================================

Names: ``p1_<model>_<option>``. Rows are the primary set unless an option puts excluded trials back
(``sets=('primary', 'excluded')`` on purpose, for the sensitivity analyses). Natural-use trials are scored
apart in every run (``extra=``).

How each run is READ is set in :mod:`p1_labels` (``role``), never here, so relabelling re-runs nothing: the
segmentation rows are robustness costs of the current model (not options), ``bjit100_trained`` is a remedy judged
against ``bjit100`` only, the ``*_scored`` runs only score excluded trials with exactly the base training set
(:data:`PRIMARY_ONLY`: primary rows, then the D2 cap), and the ``*_sizematch`` runs are size-matched controls of
``drop_all_flags`` and the two platform filters (as many training trials per person and hand, or as many training
persons, removed at random).
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.lanes import p1_variants as V
from stage2.lanes.p1_labels import (FAMILY_LABELS, FLAG_LABELS, FLAGS, OPTION_LABELS, TIGHT_DURATION_MS,  # noqa: F401
                                    TIGHT_PEAK_DPS, WINSOR_Q, subset_label)
from stage2.lanes.p1_panel import (CHANNELS, CHIRAL_CHANNELS, DEFAULT, GRAVITY_CHANNELS, MODELS, Treatment, describe,
                                   make_fit_predict, needs_harness_cap)
from stage2.pipeline import Lane

L = Lane('p1')
DEVICE_CHANNELS = tuple(CHANNELS[:9])
#: training filter of the runs that only SCORE excluded trials: primary rows only, then the D2 cap on them (the
#: harness runs with cap_per_person=None), so the model is exactly the base run's (identical primary predictions)
PRIMARY_ONLY = Treatment(train_filter=(('sets', ('primary',)), ('cap', C.CAP_PER_PERSON)))


@dataclass(frozen=True)
class Exp:
    """One experiment: a panel model on one data treatment."""
    name: str
    model: str
    family: str
    option: str                       # short label of the option
    data: str = 'core'                # 'core' (core cache approach), 'core_fixed', or a variant name
    jitter_test: str | None = None    # a variant whose arrays are predicted (X = {'base', 'jit'})
    treatment: Treatment = DEFAULT
    rows: str = 'primary'             # 'primary' | 'primary+E4d' | 'primary+E4bc'
    value: float | None = None        # numeric value of the option (line plots)
    leak_check: str = 'sample'
    notes: str = ''
    baseline: str = ''                # the experiment it is paired with ('' for a base run)

    def spec(self) -> dict[str, Any]:
        v = {'model': self.model, 'family': self.family, 'option': self.option, 'data': self.data,
             'jitter_test': self.jitter_test, 'rows': self.rows, 'treatment': self.treatment.as_dict(),
             'panel': describe(self.model, self.treatment)}
        for key in (self.data, self.jitter_test):
            if key in V.VARIANTS:
                v[f'variant_{key}'] = V.VARIANTS[key].spec()
        return v

    def spec_sha(self) -> str:
        return hashlib.sha256(json.dumps(self.spec(), sort_keys=True, default=str).encode()).hexdigest()

    def variants(self) -> list[str]:
        return [k for k in (self.data, self.jitter_test) if k and k in V.VARIANTS]


def _grid() -> list[Exp]:
    E: list[Exp] = []

    def add(model: str, family: str, option: str, **kw: Any) -> None:
        name = f'p1_{model}_{option}'
        base = '' if option == 'base' else f'p1_{model}_base'
        E.append(Exp(name, model, family, option, baseline=base, **kw))

    every, lg = MODELS, ('lr', 'gbm')
    for m in every:
        add(m, 'base', 'base', leak_check='all', notes='current pipeline choices (panel reference)')
    # 1 filtering
    for opt, val, models in (('lp_none', 0.0, every), ('lp5', 5.0, lg), ('lp10', 10.0, lg), ('lp20', 20.0, lg)):
        for m in models:
            add(m, 'filter', opt, data=opt, value=val)
    for m in lg:
        add(m, 'filter', 'lp_causal', data='lp_causal')
        for opt, order in (('lp_order2', 2.0), ('lp_order8', 8.0)):
            add(m, 'filter', opt, data=opt, value=order)
    # 2 gaps and imputation
    for opt, val in (('gap50', 50.0), ('gap250', 250.0)):
        for m in lg:
            add(m, 'gaps', opt, data=opt, value=val)
    for m in lg:
        add(m, 'gaps', 'interp_cubic', data='interp_cubic')
    for m in lg:
        add(m, 'gaps', 'interp_hold', data='interp_hold')
        add(m, 'gaps', 'e4d_scored', rows='primary+E4d', treatment=PRIMARY_ONLY,
            notes='E4d trials (long gaps, linearly imputed) scored by the base model (primary rows, D2 cap after the '
                  'filter: the base training set exactly)')
    for m in lg:
        add(m, 'gaps', 'e4d_trained', rows='primary+E4d',
            notes='E4d trials (long gaps, linearly imputed) kept in training and evaluation')
    for m in lg:
        for opt in ('interp_cubic', 'interp_hold'):
            add(m, 'gaps', f'e4d_scored_{opt.split("_")[1]}', data=opt, rows='primary+E4d', treatment=PRIMARY_ONLY,
                notes=f'E4d trials imputed by {opt.split("_")[1]} interpolation, scored by a primary-trained model '
                      '(D2 cap after the primary filter)')
    # 3 sampling rate and precision
    for m in lg:
        add(m, 'sampling', 'fs30', data='fs30', value=30.0)
    for m in every:
        add(m, 'sampling', 'fs20', data='fs20', value=20.0)
    for m in lg:
        add(m, 'sampling', 'quant_ios', data='quant_ios')
    for opt, val, models in (('tjit2', 2.0, lg), ('tjit5', 5.0, lg), ('tjit10', 10.0, lg)):
        for m in models:
            add(m, 'sampling', opt, data=opt, value=val)
    # 4 gravity and frames
    for m in lg:
        add(m, 'gravity', 'grav_lowpass', data='grav_lowpass')
    for m in every:
        add(m, 'gravity', 'grav_browser', data='grav_browser')
    frames = {'gravity': tuple(GRAVITY_CHANNELS), 'device': DEVICE_CHANNELS, 'all': tuple(CHANNELS)}
    for m, opts in (('lr', ('gravity', 'device', 'all')), ('gbm', ('gravity', 'device'))):
        for fname in opts:
            add(m, 'frame', f'frame_{fname}', treatment=Treatment(channels=frames[fname]))
    # 5 windows and segmentation errors
    for opt, models in (('win_first_half', lg), ('win_hold', lg), ('win_pre', lg), ('win_fixed', every)):
        for m in models:
            add(m, 'window', opt, data=opt)
    for opt, val in (('bjit50', 50.0), ('bjit100', 100.0), ('bjit200', 200.0), ('onset_late100', 100.0)):
        for m in lg:
            add(m, 'segmentation', opt, jitter_test=opt, value=val,
                treatment=Treatment(train_keys=('base',), test_key='jit'),
                notes='trained on oracle bounds, tested on perturbed bounds')
    for m in lg:
        add(m, 'segmentation', 'bjit100_trained', jitter_test='bjit100', value=100.0,
            treatment=Treatment(train_keys=('base', 'jit'), test_key='jit'),
            notes='trained on oracle and jittered bounds, tested on jittered bounds')
    # 6 normalisation (every option on every panel model)
    for opt, kind in (('norm_zscore', 'zscore'), ('norm_robust', 'robust'), ('norm_trial_peak', 'trial_peak')):
        for m in every:
            add(m, 'normalisation', opt, treatment=Treatment(norm=kind))
    # 7 outliers and corrections
    for m in lg:
        add(m, 'outliers', 'winsor', treatment=Treatment(winsor=WINSOR_Q))
    for f in FLAGS:
        for m in lg:
            add(m, 'outliers', f'drop_{f}', treatment=Treatment(train_filter=(('drop_flags', (f,)),)))
    for m in lg:
        add(m, 'outliers', 'drop_all_flags', treatment=Treatment(train_filter=(('drop_flags', FLAGS),)))
    for m in lg:
        add(m, 'outliers', 'drop_swap_blocks', treatment=Treatment(train_filter=(('drop_blocks', 'SWAP'),)),
            notes='blocks with the swap signature (data-driven, reference.swap_blocks) out of training only')
    for m in lg:
        add(m, 'outliers', 'excl_loose', rows='primary+E4bc',
            notes='E4b and E4c trials kept (thresholds loosened to 0) in training and evaluation')
    for m in lg:
        add(m, 'outliers', 'excl_tight', treatment=Treatment(train_filter=(('min_duration_ms', TIGHT_DURATION_MS),
                                                                         ('min_peak_gyro_dps', TIGHT_PEAK_DPS))),
            notes=f'training only on trials >= {TIGHT_DURATION_MS:.0f} ms and >= {TIGHT_PEAK_DPS:.0f} deg/s')
    for m in lg:
        add(m, 'outliers', 'excl_scored', rows='primary+E4bc', treatment=PRIMARY_ONLY,
            notes='E4b and E4c trials scored by the base model (primary rows, D2 cap after the filter)')
    for m in lg:     # size-matched control of drop_all_flags (same rows per person and hand, chosen at random)
        add(m, 'outliers', 'drop_all_flags_sizematch', treatment=Treatment(train_filter=(('match_flags', FLAGS),)),
            notes='size-matched control: as many random training trials per person and hand as no_flags keeps')
    # 8 platform
    for m in lg:
        add(m, 'platform', 'platform_cross', treatment=Treatment(train_filter=(('platform', 'cross'),)),
            notes='trained on the persons of the other platform only')
        add(m, 'platform', 'platform_same', treatment=Treatment(train_filter=(('platform', 'same'),)),
            notes='trained on the other persons of the same platform only')
    for m in lg:     # size-matched controls of the platform filters (as many random persons, both platforms)
        for side in ('cross', 'same'):
            add(m, 'platform', f'platform_{side}_sizematch',
                treatment=Treatment(train_filter=(('match_persons', side),)),
                notes=f'size-matched control: as many random training persons as platform_{side} keeps')
    # 9 augmentation
    for m in every:
        add(m, 'augmentation', 'no_mirror', treatment=Treatment(mirror=False))
    for m in lg:
        add(m, 'augmentation', 'time_warp', treatment=Treatment(time_warp=True))
    add('gbm', 'augmentation', 'screen_flip', treatment=Treatment(screen_flip=True))
    return E


GRID: list[Exp] = _grid()
BY_NAME: dict[str, Exp] = {e.name: e for e in GRID}
if len(BY_NAME) != len(GRID):
    raise RuntimeError('duplicate experiment names in the p1 grid')
#: the pipeline's current choice per family (from the config, so a changed default relabels itself)
DEFAULT_LABELS = {
    'filter': f'zero-phase Butterworth, {C.LOWPASS_HZ:g} Hz, order {C.LOWPASS_ORDER}',
    'gaps': f'gaps up to {C.MAX_GAP_MS:g} ms bridged linearly; E4d excluded',
    'sampling': f'{C.FS_HZ:g} Hz, native precision' + ('' if not C.QUANTIZE_IOS else ' (iOS rounded)'),
    'gravity': 'gyro-propagated gravity (gravity_v1)',
    'frame': f'model default (LR chiral, GBM all {len(CHANNELS)}, MR gravity frame)',
    'window': f'approach, time-normalised ({C.N_NORM} samples)',
    'segmentation': 'oracle (stored online_v1) bounds', 'normalisation': 'none (physical units)',
    'outliers': 'no winsorising; flagged trials and swap-signature blocks kept in training',
    'exclusions': (f'E4b (< {C.MIN_DURATION_MS / 1000:g} s), E4c (peak < {C.MIN_PEAK_GYRO_DPS:g} deg/s) and E4d '
                   f'(gap > {C.MAX_GAP_MS:g} ms near the movement) trials excluded'),
    'platform': 'pooled over platforms', 'augmentation': 'mirror augmentation',
}


# ───────────────────────────── data ─────────────────────────────


class Data:
    """Trial tables and arrays shared by the experiments of one compute run (loaded once)."""

    def __init__(self) -> None:
        from stage2.data import load_cache
        from stage2.scripts.reference import swap_blocks
        self.all = load_cache(sets=None)
        self.d = load_cache()
        self.nat = load_cache(('natural_use',))
        self.swap = swap_blocks(self.d)

    def rows(self, kind: str) -> pd.DataFrame:
        t = self.all.trials
        prim = t['analysis_set'] == 'primary'
        instructed = t['hand_source'].astype(str) != 'reported'
        if kind == 'primary':
            m = prim
        elif kind == 'primary+E4d':
            m = prim | ((t['excl_rule'] == 'E4d') & instructed)
        elif kind == 'primary+E4bc':
            m = prim | (t['excl_rule'].isin(['E4b', 'E4c']) & instructed)
        else:
            raise ValueError(kind)
        return t[m.to_numpy()]

    def arrays(self, source: str, trials: pd.DataFrame) -> np.ndarray:
        if source in ('core', 'core_fixed'):
            pos = {tid: i for i, tid in enumerate(self.all.trials['trial_id'])}
            idx = [pos[t] for t in trials['trial_id']]
            return (self.all.approach if source == 'core' else self.all.fixed)[idx]
        return V.load(source, trials['trial_id'])


def _resolve(tr: Treatment, data: Data) -> Treatment:
    """Replace the 'SWAP' placeholder by the data-driven swap-signature blocks."""
    from dataclasses import replace
    tf = tuple((k, tuple(data.swap)) if (k == 'drop_blocks' and v == 'SWAP') else (k, v) for k, v in tr.train_filter)
    return replace(tr, train_filter=tf)


def build_inputs(e: Exp, data: Data) -> tuple[Any, pd.DataFrame, Any, pd.DataFrame]:
    """(X, meta, X_nat, meta_nat) of experiment e."""
    meta = data.rows(e.rows).reset_index(drop=True)
    nat = data.nat.trials
    if e.jitter_test:
        X = {'base': data.arrays(e.data, meta), 'jit': data.arrays(e.jitter_test, meta)}
        Xn = {'base': data.arrays(e.data, nat), 'jit': data.arrays(e.jitter_test, nat)}
    else:
        X, Xn = data.arrays(e.data, meta), data.arrays(e.data, nat)
    return X, meta, Xn, nat


# ───────────────────────────── staleness ─────────────────────────────


def stored_params(name: str) -> dict[str, Any] | None:
    p = C.RES_DIR / f'{name}.json'
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8')).get('params') or {}
    except json.JSONDecodeError:
        return None


def current_params(e: Exp) -> dict[str, Any]:
    return {'p1_spec_sha256': e.spec_sha(), 'variant_sha256': {k: V.array_sha(k) for k in e.variants()}}


def needs_run(e: Exp, force: bool, only: list[str] | None) -> bool:
    """Selected and (forced, no result from the current core cache, or another spec / variant arrays)."""
    from stage2.pipeline import selected
    if not selected(e.name, only):
        return False
    if force or L.should_run(e.name, False, None):
        return True
    sp = stored_params(e.name) or {}
    if sp.get('p1_spec_sha256') != e.spec_sha():
        return True
    return any(not V.is_current(k) or sp.get('variant_sha256', {}).get(k) != V.array_sha(k) for k in e.variants())


# ───────────────────────────── running ─────────────────────────────


def run_one(e: Exp, data: Data) -> Any:
    for k in e.variants():
        V.build(k)
    X, meta, Xn, nat = build_inputs(e, data)
    tr = _resolve(e.treatment, data)
    sets = ('primary',) if e.rows == 'primary' else ('primary', 'excluded')
    t0 = time.time()
    cap_kw = {} if needs_harness_cap(tr) else {'cap_per_person': None}    # the treatment applies the D2 cap itself
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = L.run_experiment(e.name, make_fit_predict(e.model, tr), X, meta, extra=(Xn, nat), sets=sets,
                             model=f'p1_{e.model}', params={**current_params(e), 'spec': e.spec()},
                             notes=f'{e.family}: {e.option}. {e.notes}'.strip(), leak_check=e.leak_check,
                             n_configs_tried=1, registry=None, **cap_kw)
    print(f'  {e.name}: acc {100 * r.metrics["accuracy"]:.1f} % auc {r.metrics["roc_auc"]:.3f} '
          f'({r.leak_check}; {time.time() - t0:.0f} s)', flush=True)
    return r


def run_queue(names: list[str], *, build_variants: bool = True, wait_s: float = 30.0, max_wait_s: float = 3600.0) -> None:
    """Run these experiments in order. A worker that does not build variants defers an experiment whose
    variant is not built yet and comes back to it (another worker builds it)."""
    data = Data()
    pending = list(names)
    waited = 0.0
    while pending:
        progressed = False
        for name in list(pending):
            e = BY_NAME[name]
            if not build_variants and not all(V.is_current(k) for k in e.variants()):
                continue
            run_one(e, data)
            pending.remove(name)
            progressed = True
        if pending and not progressed:
            if waited >= max_wait_s:
                raise TimeoutError(f'variants for {pending[:3]} ... were never built')
            time.sleep(wait_s)
            waited += wait_s


def rebuild_registry() -> None:
    """results/registry_p1.csv from every saved p1 result (one writer, sorted: no race between workers)."""
    from stage2 import cv
    rows = []
    for e in GRID:
        p = C.RES_DIR / f'{e.name}.json'
        if p.exists():
            rows.append(cv.ExperimentResult.load(p).summary_row())
    for p in sorted(C.RES_DIR.glob('p1_calib_*.json')):
        rows.append(cv.ExperimentResult.load(p).summary_row())
    if rows:
        pd.DataFrame(rows).sort_values('name').to_csv(L.registry, index=False)


COST_S = {'lr': 15.0, 'gbm': 45.0, 'mr': 150.0}      # rough seconds per run on a shared machine (queue balance)


def worker_queues(todo: list[str]) -> list[tuple[list[str], bool]]:
    """Two queues for two single-thread workers with about the same estimated cost. Worker 0 builds the variant
    caches (it starts with the LR runs, which need every variant); worker 1 waits for them and starts with
    the experiments on the core cache."""
    order = sorted(todo, key=lambda n: (BY_NAME[n].model != 'lr', GRID.index(BY_NAME[n])))
    a, b, ca, cb = [], [], 0.0, 0.0
    for n in [x for x in order if BY_NAME[x].model == 'lr']:
        a.append(n)
        ca += COST_S['lr'] + 25.0 * bool(BY_NAME[n].variants())       # + variant builds
    rest = sorted([x for x in order if BY_NAME[x].model != 'lr'],
                  key=lambda n: (-COST_S[BY_NAME[n].model], bool(BY_NAME[n].variants()), GRID.index(BY_NAME[n])))
    for n in rest:
        if cb <= ca:
            b.append(n)
            cb += COST_S[BY_NAME[n].model]
        else:
            a.append(n)
            ca += COST_S[BY_NAME[n].model]
    b.sort(key=lambda n: (bool(BY_NAME[n].variants()), -COST_S[BY_NAME[n].model], GRID.index(BY_NAME[n])))
    return [(q, build) for q, build in ((a, True), (b, not a)) if q]


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description='lane p1 experiment worker (normally started by the p1 compute step)')
    ap.add_argument('names', nargs='+')
    ap.add_argument('--no-build', action='store_true', help='wait for variants instead of building them')
    a = ap.parse_args(argv)
    run_queue(a.names, build_variants=not a.no_build)


__all__ = ['GRID', 'BY_NAME', 'Exp', 'Data', 'run_one', 'run_queue', 'needs_run', 'rebuild_registry', 'build_inputs',
           'DEFAULT_LABELS', 'FLAGS', 'worker_queues']

if __name__ == '__main__':
    main()

