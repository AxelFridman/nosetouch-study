"""Lane final: every experiment of the lane, declared in one place and run through ``cv.run_experiment``.

Kinds (result names start with ``final_``):

* ``final_<key>``             a candidate new in this lane (task 2b models, the cross-lane ensemble);
* ``final_nested_select``     the nested cross-lane selection (``final_nested``);
* ``final_flip_<key>``        the candidate's predictions on its screen-flipped test trials;
* ``final_grip_<key>``        trained on upright-grip trials only (sensor-detected grip);
* ``final_plat_<key>``        trained on the other platform's persons only (lanes other than p1, which did it);
* ``final_noisy_<key>``       trained without the candidate wrong-hand and swap-signature blocks;
* ``final_flipaug_<key>``     remedy: screen-flip augmentation of the training rows (recorded test trials) and
  ``final_flipaugx_<key>`` (the same model on the screen-flipped test trials);
* ``final_platctl_<key>``     size-matched control of the platform transfer (random persons);
* ``final_cal_<key>_<method>`` the final model with a calibrator fitted on inner cross-fitted predictions;
* ``final_lc_<key>_n<NN>_r<R>`` learning curve: training persons subsampled inside fit_predict.

``python -m stage2.lanes.final_experiments --names final_flip_m2_lr,final_grip_m2_lr`` runs some of them in one
process (the compute step of ``final_pipeline`` runs every missing one; worker processes may share the work).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import final_candidates as FC
from stage2.lanes import final_nested as FN
from stage2.lanes import final_robust as FR

LANE = 'final'
#: learning curve: training-set sizes (persons) below "all", seeded repetitions per size
LC_SIZES: tuple[int, ...] = (3, 6, 9, 12, 15)
LC_REPS = 5
#: calibration remedies of the final model (fitted on inner cross-fitted predictions of the training persons)
CAL_METHODS: tuple[str, ...] = ('none', 'platt', 'temperature', 'isotonic', 'beta')
#: platform-transfer results lane p1 already has for a candidate's model (other-platform-only training)
P1_PLATFORM: dict[str, str] = {'p1_gbm': 'p1_gbm_platform_cross', 'p1_lr': 'p1_lr_platform_cross'}
#: test-time variants of the final model's input (lane p1's variant caches: the same canonical code path with one
#: parameter changed); the model is trained on the oracle-bound arrays and predicts the variant arrays
VARIANTS_TEST: dict[str, str] = {
    'bjit50': 'segment bounds jittered by U(-50, +50) ms', 'bjit100': 'segment bounds jittered by U(-100, +100) ms',
    'bjit200': 'segment bounds jittered by U(-200, +200) ms', 'onset_late100': 'onset 100 ms late',
    'lp_causal': 'causal 15-Hz low-pass instead of zero phase'}


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str                    # candidate | nested | flip | grip | plat | noisy | flipaug | variant | cal | lc
    cand: str | None = None
    arg: Any = None
    notes: str = ''
    n_configs_tried: int = 1
    leak_check: str = C.LEAK_CHECK


def base_result_name(c: FC.Candidate) -> str:
    """The saved LOSO result of a candidate (its lane's result, or this lane's for a new candidate)."""
    return c.source if c.source else f'final_{c.key}'


def flip_invariant(c: FC.Candidate) -> bool:
    """Inputs from the gravity frame only are unchanged by the screen flip (``standardize`` screen-flip signs)."""
    return set(c.frames) <= {'gravity'}


def plan(cands: dict[str, FC.Candidate], final_model: str | None = None) -> list[Spec]:
    """Every experiment of the lane, in run order. ``final_model``: the analyses of the chosen model."""
    S: list[Spec] = []
    for c in cands.values():
        if c.source is None:
            S.append(Spec(f'final_{c.key}', 'candidate', c.key, notes=f'{c.label}: {c.notes}'))
    S.append(Spec('final_nested_select', 'nested', notes='nested cross-lane selection: every candidate cross-validated '
                  'on the training persons (inner log loss), the lowest refitted on all of them'))
    for c in cands.values():
        if not flip_invariant(c):
            S.append(Spec(f'final_flip_{c.key}', 'flip', c.key, notes=f'{c.label}: screen-flipped test trials'))
        S.append(Spec(f'final_grip_{c.key}', 'grip', c.key, notes=f'{c.label}: trained on upright grips only '
                      '(sensor-detected), every trial predicted'))
        if c.key not in P1_PLATFORM:
            S.append(Spec(f'final_plat_{c.key}', 'plat', c.key, arg='other_platform',
                          notes=f'{c.label}: trained on the other platform only'))
        S.append(Spec(f'final_noisy_{c.key}', 'noisy', c.key, notes=f'{c.label}: trained without the candidate '
                      'wrong-hand and swap-signature blocks (labels unchanged, every trial evaluated)'))
        S.append(Spec(f'final_cal_{c.key}_auto', 'cal', c.key, arg='auto',
                      notes=f'{c.label}: its production form, the calibration remedy (none, temperature, Platt, beta) '
                      'chosen inside every outer fold by grouped cross-validation on the training persons\' inner '
                      'cross-fitted predictions, fitted on them and applied to the held-out person'))
    if final_model:
        c = cands[final_model]
        S.append(Spec(f'final_platctl_{c.key}', 'plat', c.key, arg='platform_control',
                      notes=f'{c.label}: size-matched control of the platform transfer (random persons)'))
        if not flip_invariant(c):
            S.append(Spec(f'final_flipaug_{c.key}', 'flipaug', c.key, arg='x',
                          notes=f'{c.label}: screen-flip augmentation of the training rows, recorded test trials'))
            S.append(Spec(f'final_flipaugx_{c.key}', 'flipaug', c.key, arg='xflip',
                          notes=f'{c.label}: screen-flip augmentation, screen-flipped test trials'))
        for v, desc in VARIANTS_TEST.items():
            S.append(Spec(f'final_var_{v}_{c.key}', 'variant', c.key, arg=v,
                          notes=f'{c.label}: trained on the oracle bounds, tested on lane p1 variant {v} ({desc})'))
        for m in CAL_METHODS:
            S.append(Spec(f'final_cal_{c.key}_{m}', 'cal', final_model, arg=m,
                          notes=f'{c.label}: calibrator {m} fitted on the training persons\' inner cross-fitted '
                          'predictions (5 inner person folds), applied to the held-out person'))
        for key in dict.fromkeys([final_model, 'ref']):
            for n in LC_SIZES:
                for r in range(LC_REPS):
                    S.append(Spec(f'final_lc_{key}_n{n:02d}_r{r}', 'lc', key, arg=(n, r), leak_check='sample',
                                  notes=f'learning curve: {n} training persons drawn at random (repetition {r})'))
    return S


# ───────────────────────────── fit_predict builders ─────────────────────────────


def candidate_fp(c: FC.Candidate, cands: dict[str, FC.Candidate]) -> Callable:
    return FN.ensemble_fit_predict(c.components, cands) if c.components else c.make()


def subsample_persons(fp: Callable, n: int, rep: int) -> Callable:
    """fit_predict trained on ``n`` training persons drawn at random (seeded by the fold and the repetition)."""
    def fit_predict(fold: cv.Fold) -> Any:
        persons = np.unique(fold.groups_train.astype(str))
        k = min(n, persons.size)
        pick = np.random.default_rng([fold.seed, 101, n, rep]).choice(persons, size=k, replace=False)
        out = fp(FR.restrict_training(fold, np.isin(fold.groups_train.astype(str), pick)))
        p, info = (out if isinstance(out, tuple) else (out, {}))
        return p, {**info, 'n_train_persons': int(k)}
    return fit_predict


def calibrated_fp(c: FC.Candidate, cands: dict[str, FC.Candidate], method: str) -> Callable:
    """The candidate with a calibrator fitted on its inner cross-fitted predictions of the training persons
    (``final_nested`` cache) and applied to the held-out person; 'none' returns the candidate's own P(right);
    'auto' chooses the remedy inside the fold (``final_calibration.choose_calibrator``: grouped cross-validation of
    the remedies on the same inner predictions; the held-out person never informs the choice)."""
    from stage2.lanes.final_calibration import choose_calibrator, fit_calibrator
    fp = candidate_fp(c, cands)
    fingerprints = {k: FN._cand_fingerprint(x) for k, x in cands.items() if not x.components}

    def fit_predict(fold: cv.Fold) -> Any:
        sub = fold if c.components else FN.subfold(fold, c.input)
        out = fp(sub)
        p = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
        if method == 'none':
            return p, {'calibrator': 'none'}
        p_in = FN.inner_predictions(c, fold, cands, compute=True, fingerprints=fingerprints)
        z_in = FN.logit(p_in)
        info: dict[str, Any] = {}
        m = method
        if method == 'auto':
            m, scores = choose_calibrator(z_in, fold.y_train, fold.groups_train)
            info = {'chosen': m, 'cv_log_loss': {k: round(v, 6) for k, v in scores.items()}}
        apply, params = fit_calibrator(m, z_in, fold.y_train)
        return np.clip(apply(FN.logit(p)), 0.0, 1.0), {'calibrator': method, **info, 'calib_params': params}

    return fit_predict


def final_model_constant() -> str:
    """``final_pipeline.FINAL_MODEL`` read from its source (importing the pipeline module would set this process's
    thread budget, a side effect a lane rebuilding the finalist must not suffer)."""
    import ast
    from pathlib import Path
    src = Path(__file__).with_name('final_pipeline.py').read_text(encoding='utf-8')
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'FINAL_MODEL' for t in node.targets):
            return str(ast.literal_eval(node.value))
    raise KeyError('FINAL_MODEL not found in final_pipeline.py')


def build_x_final(d: Any) -> Any:
    """``build_x`` of the finalists entry: the inputs of ``final_pipeline.FINAL_MODEL`` (and of its components) for the
    Stage2Data ``d``, as one flat dict (per trial, no label)."""
    cands = FC.candidates()
    c = cands[final_model_constant()]
    keys = FC.needed_inputs({c.key: c} | {k: cands[k] for k in c.components})
    return flatten_inputs({k: v for k, v in FC.build_inputs(d, keys).items()})


def finalist_factory(candidate: str, method: str = 'none') -> Callable[[cv.Fold], Any]:
    """``factory`` of the finalists entry: fit_predict of a candidate with its shipped calibration remedy, on the flat
    input dict of :func:`build_x_final` (inner predictions are read from, or written to, the lane's cache)."""
    cands = FC.candidates()
    return unflattening(calibrated_fp(cands[candidate], cands, method))


# ───────────────────────────── nested inputs as one flat dict ─────────────────────────────

#: separator of the flattened keys: ``cv.run_experiment`` takes an array, a DataFrame or a FLAT dict of them (its row
#: count of a dict is the length of each value, so a nested dict would count keys), so the shared input dict of the
#: nested selection, the ensemble and the screen-flip pairs ({'x': input, 'xflip': input}) is flattened before the
#: harness and rebuilt inside fit_predict (:func:`unflattening`); every array stays visible to the harness's label check
SEP = '::'


def flatten_inputs(X: Any) -> Any:
    """A nested dict of inputs as one level (keys joined with ``SEP``); anything else is returned unchanged."""
    if not isinstance(X, dict) or not any(isinstance(v, dict) for v in X.values()):
        return X
    out: dict[str, Any] = {}
    for k, v in X.items():
        if SEP in str(k):
            raise ValueError(f'input key {k!r} contains {SEP!r}')
        if isinstance(v, dict):
            for kk, vv in flatten_inputs(v).items():
                out[f'{k}{SEP}{kk}'] = vv
        else:
            out[k] = v
    return out


def unflatten_inputs(X: Any) -> Any:
    """The inverse of :func:`flatten_inputs` (a dict without ``SEP`` keys and any non-dict pass unchanged)."""
    if not isinstance(X, dict) or not any(SEP in str(k) for k in X):
        return X
    out: dict[str, Any] = {}
    for k, v in X.items():
        if SEP in str(k):
            head, rest = str(k).split(SEP, 1)
            out.setdefault(head, {})[rest] = v
        else:
            out[k] = v
    return {k: unflatten_inputs(v) if isinstance(v, dict) else v for k, v in out.items()}


def unflattening(fp: Callable[[cv.Fold], Any]) -> Callable[[cv.Fold], Any]:
    """fit_predict on the flat dict: rebuilds the nested inputs of both sides, then calls ``fp``."""
    import dataclasses

    def fit_predict(fold: cv.Fold) -> Any:
        return fp(dataclasses.replace(fold, X_train=unflatten_inputs(fold.X_train),
                                      X_test=unflatten_inputs(fold.X_test)))
    return fit_predict


def build_fp(spec: Spec, cands: dict[str, FC.Candidate], blocks: list[tuple[str, int]]) -> Callable:
    if spec.kind == 'nested':
        return FN.make_select_fit_predict(cands)
    c = cands[spec.cand]
    base = candidate_fp(c, cands)
    if spec.kind == 'candidate':
        return base
    if spec.kind in ('flip', 'variant'):
        return FR.flip_predict(base)            # trains on 'x', predicts the second input ('xflip' slot)
    if spec.kind == 'grip':
        return FR.train_filter(base, 'upright')
    if spec.kind == 'plat':
        return FR.train_filter(base, spec.arg)
    if spec.kind == 'noisy':
        return FR.train_filter(base, 'no_blocks', blocks=blocks)
    if spec.kind == 'flipaug':
        return FR.flip_augment(base, predict=spec.arg)
    if spec.kind == 'cal':
        return calibrated_fp(c, cands, spec.arg)
    if spec.kind == 'lc':
        n, r = spec.arg
        return subsample_persons(base, n, r)
    raise ValueError(spec.kind)


# ───────────────────────────── data ─────────────────────────────


def label_noise_blocks(d: Any) -> list[tuple[str, int]]:
    """The candidate wrong-hand blocks of lane m1 (``tables/m1/m1_candidate_blocks.csv``, data-driven there) and the
    swap-signature blocks of the core screen (``reference.swap_blocks``): (session_label, block_index)."""
    from stage2.scripts.reference import swap_blocks
    out = list(swap_blocks(d))
    p = C.TAB_DIR / 'm1' / 'm1_candidate_blocks.csv'
    if p.exists():
        t = pd.read_csv(p)
        if 'candidate' in t.columns:
            t = t[t['candidate'].astype(str).str.lower() == 'true']
        out += [(str(s), int(b)) for s, b in zip(t['session_label'], t['block_index'])]
    return sorted(dict.fromkeys(out))


@dataclass
class Data:
    d: Any
    nat: Any
    X: dict[str, Any] = field(default_factory=dict)
    Xn: dict[str, Any] = field(default_factory=dict)
    Xf: dict[str, Any] = field(default_factory=dict)
    Xnf: dict[str, Any] = field(default_factory=dict)

    def ensure(self, keys: list[str], flip: bool = False) -> None:
        miss = [k for k in keys if k not in self.X]
        if miss:
            self.X.update(FC.build_inputs(self.d, miss))
            self.Xn.update(FC.build_inputs(self.nat, miss))
        if flip:
            miss = [k for k in keys if k not in self.Xf]
            if miss:
                self.Xf.update(FC.build_inputs(FR.flipped(self.d), miss))
                self.Xnf.update(FC.build_inputs(FR.flipped(self.nat), miss))


def load_data() -> Data:
    from stage2.data import load_cache
    return Data(load_cache(), load_cache(('natural_use',)))


def inputs_for(spec: Spec, cands: dict[str, FC.Candidate], D: Data) -> tuple[Any, Any]:
    """(X, X_natural_use) of an experiment."""
    if spec.kind == 'nested':
        keys = FC.needed_inputs(cands)
    else:
        c = cands[spec.cand]
        keys = FC.needed_inputs({c.key: c} | {k: cands[k] for k in c.components})
    flip = spec.kind in ('flip', 'flipaug')
    D.ensure(keys, flip=flip)
    if spec.kind == 'variant':
        return variant_inputs(spec.arg, keys, D)
    whole = spec.kind in ('nested', 'cal') or (spec.cand is not None and bool(cands[spec.cand].components))
    if spec.kind == 'cal' and not cands[spec.cand].components:
        whole = True                                    # calibrated_fp reads the inner cache via the whole dict
    pick = (lambda src: {k: src[k] for k in keys}) if whole else (lambda src: src[keys[0]])
    if flip:
        return ({'x': pick(D.X), 'xflip': pick(D.Xf)}, {'x': pick(D.Xn), 'xflip': pick(D.Xnf)})
    return pick(D.X), pick(D.Xn)


def variant_inputs(variant: str, keys: list[str], D: Data) -> tuple[Any, Any]:
    """({'x': input, 'xflip': the input from p1's variant arrays}, the same for the natural-use rows; a natural-use
    trial missing from the variant cache keeps its oracle-bound input, recorded in the result's notes)."""
    if len(keys) != 1:
        raise ValueError('variant experiments take a single-input candidate')
    k = keys[0]
    xv = FC.build_inputs(FR.VariantView(D.d, variant), [k])[k]
    try:
        xnv = FC.build_inputs(FR.VariantView(D.nat, variant), [k])[k]
    except KeyError:
        xnv = D.Xn[k]
    return {'x': D.X[k], 'xflip': xv}, {'x': D.Xn[k], 'xflip': xnv}


def check_finalist(entry: dict[str, Any], D: Data | None = None) -> dict[str, Any]:
    """Rebuild the finalist exactly as its finalists entry documents it (``import_module(module)``,
    ``getattr(mod, build_x)(d)``, ``getattr(mod, factory)(**params)``), re-run it through ``run_experiment``
    (``save=False``, leak check off: the saved result passed it) and compare with its saved predictions."""
    import importlib
    D = D or load_data()
    mod = importlib.import_module(entry['module'])
    X, Xn = getattr(mod, entry['build_x'])(D.d), getattr(mod, entry['build_x'])(D.nat)
    t0 = time.time()
    r = cv.run_experiment('final_finalist_rebuild', getattr(mod, entry['factory'])(**entry['params']), X, D.d.trials,
                          extra=(Xn, D.nat.trials), leak_check='off', n_boot=0, save=False)
    saved = cv.load_result(entry['name'])
    a = r.predictions.set_index('trial_id')['p_right']
    b = saved.predictions.set_index('trial_id')['p_right'].reindex(a.index)
    an = r.extra_predictions.set_index('trial_id')['p_right']
    bn = saved.extra_predictions.set_index('trial_id')['p_right'].reindex(an.index)
    dp, dn = float(np.max(np.abs(a - b))), float(np.max(np.abs(an - bn)))
    return {'name': entry['name'], 'max_abs_dp_primary': dp, 'max_abs_dp_natural_use': dn,
            'reproduced': bool(dp <= 1e-9 and dn <= 1e-9), 'seconds': round(time.time() - t0, 1)}


def run_spec(spec: Spec, cands: dict[str, FC.Candidate], D: Data, blocks: list[tuple[str, int]], *,
             save: bool = True) -> cv.ExperimentResult:
    from stage2.pipeline import Lane
    fp = unflattening(build_fp(spec, cands, blocks))
    X, Xn = inputs_for(spec, cands, D)
    X, Xn = flatten_inputs(X), flatten_inputs(Xn)
    c = cands.get(spec.cand) if spec.cand else None
    params = {'kind': spec.kind, 'candidate': spec.cand, 'arg': spec.arg,
              'candidate_description': FC.describe(c) if c else None}
    t0 = time.time()
    res = Lane(LANE).run_experiment(spec.name, fp, X, D.d.trials, extra=(Xn, D.nat.trials), notes=spec.notes,
                                    n_configs_tried=spec.n_configs_tried, leak_check=spec.leak_check,
                                    model=spec.name, params=json.loads(json.dumps(params, default=str)), save=False,
                                    spec={'kind': spec.kind, 'candidate': spec.cand, 'arg': json.dumps(spec.arg)})
    res.info['wall_seconds'] = round(time.time() - t0, 1)
    if save:
        res.save(C.RES_DIR, Lane(LANE).registry)
    return res


def reproduce(c: FC.Candidate, cands: dict[str, FC.Candidate], D: Data) -> dict[str, Any]:
    """Rebuild a lane finalist as its finalists file documents it, re-run it (``save=False``, leak check off: the
    saved result already passed it) and compare with its saved predictions (primary and natural-use trials)."""
    keys = FC.needed_inputs({c.key: c} | {k: cands[k] for k in c.components})
    D.ensure(keys)
    X = {k: D.X[k] for k in keys} if c.components else D.X[keys[0]]
    Xn = {k: D.Xn[k] for k in keys} if c.components else D.Xn[keys[0]]
    t0 = time.time()
    r = cv.run_experiment(f'final_rep_{c.key}', candidate_fp(c, cands), X, D.d.trials, extra=(Xn, D.nat.trials),
                          leak_check='off', n_boot=0, save=False)
    secs = time.time() - t0
    saved = cv.load_result(c.source)
    a = r.predictions.set_index('trial_id')['p_right']
    b = saved.predictions.set_index('trial_id')['p_right'].reindex(a.index)
    dn = 0.0
    if r.extra_predictions is not None and saved.extra_predictions is not None:
        an = r.extra_predictions.set_index('trial_id')['p_right']
        bn = saved.extra_predictions.set_index('trial_id')['p_right'].reindex(an.index)
        dn = float(np.max(np.abs(an - bn)))
    return {'key': c.key, 'source': c.source, 'max_abs_dp_primary': float(np.max(np.abs(a - b))),
            'max_abs_dp_natural_use': dn, 'n_trials': int(len(a)), 'loso_seconds': round(secs, 1),
            'reproduced': bool(np.max(np.abs(a - b)) <= 1e-9 and dn <= 1e-9)}


def save_reproduction(entry: dict[str, Any]) -> None:
    """Merge one reproduction check into tables/final/final_reproduction.json (under a lock, written atomically)."""
    from stage2.locking import file_lock
    from stage2.pipeline import Lane
    L = Lane(LANE)
    p = L.tab_dir / 'final_reproduction.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(p):
        cur = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        rows = {r['key']: r for r in cur.get('rows', [])}
        rows[entry['key']] = entry
        L.write_json('final_reproduction', {'rows': [rows[k] for k in sorted(rows)]})


def main(argv: list[str] | None = None) -> int:
    """Worker entry point (the pipeline's compute step runs every missing experiment itself):

        python -m stage2.lanes.final_experiments --names "final_grip_*"            # plan order
        python -m stage2.lanes.final_experiments --jobs @jobs.txt --log run.log     # this order; rep:<key> reproduces

    Threads: one per worker process (``--threads``); several workers share the 3-thread budget of the lane."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', default='', help='comma-separated experiment names (shell patterns allowed)')
    ap.add_argument('--reproduce', default='', help='comma-separated candidate keys to reproduce')
    ap.add_argument('--jobs', default='', help='rep:<key> or experiment names, comma-separated or @file, run in order')
    ap.add_argument('--final-model', default=None, help='default: final_pipeline.FINAL_MODEL')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--log', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    from stage2.pipeline import Lane, limit_threads, selected
    limit_threads(a.threads)
    if a.final_model is None:
        a.final_model = final_model_constant()
    cands = FC.candidates()
    D = load_data()
    blocks = label_noise_blocks(D.d)

    def say(msg: str) -> None:
        msg = f'{time.strftime("%H:%M:%S")} {msg}'
        print(msg, flush=True)
        if a.log:
            with open(a.log, 'a', encoding='utf-8') as fh:
                fh.write(msg + '\n')

    def run_one(s: Spec) -> None:
        if not Lane(LANE).should_run(s.name, a.force, None):
            say(f'{s.name}: current, skipped')
            return
        t0 = time.time()
        r = run_spec(s, cands, D, blocks)
        say(f'{s.name}: acc {r.metrics["accuracy"]:.4f} ll {r.metrics["log_loss"]:.4f} {r.leak_check} '
            f'({time.time() - t0:.0f} s)')

    def rep_one(k: str) -> None:
        t0 = time.time()
        out = reproduce(cands[k], cands, D)
        save_reproduction(out)
        say(f'rep:{k}: max|dP| {out["max_abs_dp_primary"]:.2e} / natural {out["max_abs_dp_natural_use"]:.2e} '
            f'reproduced {out["reproduced"]} ({time.time() - t0:.0f} s)')

    if a.reproduce:
        for k in a.reproduce.split(','):
            rep_one(k)
    specs = {s.name: s for s in plan(cands, a.final_model)}
    names = [n for n in a.names.split(',') if n]
    if names:
        for s in specs.values():
            if selected(s.name, names):
                run_one(s)
    if a.jobs:
        from pathlib import Path
        jobs = Path(a.jobs[1:]).read_text(encoding='utf-8').split() if a.jobs.startswith('@') else a.jobs.split(',')
        import traceback
        failed = 0
        for j in (j.strip() for j in jobs):
            if not j:
                continue
            try:
                if j.startswith('rep:'):
                    rep_one(j[4:])
                elif j in specs:
                    run_one(specs[j])
                else:
                    say(f'{j}: not in the plan (final model {a.final_model}); skipped')
            except Exception as e:                       # one failing job must not stop the worker's queue
                failed += 1
                say(f'{j}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}')
        return 1 if failed else 0
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
