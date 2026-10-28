"""Lane sec: the compute step: every experiment and every derived file the figures, tables and numbers read.

Experiments (``results/<name>.json``):

* multi-class, instructed speed and instructed grip (:data:`MULTICLASS`): ``sec_models.loso_multiclass`` on the
  leave-one-person-out splits of ``cv.loso_splits`` with nested tuning on ``cv.inner_splits`` and person-cluster
  bootstrap CIs (``cv.person_bootstrap``); saved with ``Lane.write_json`` (provenance + the declared spec; the
  experiments on label-blind windows also record the hashes of the saved re-segmentation they read, so a changed
  re-segmentation makes them stale) under the key ``trial_predictions``, so the binary-result tools (registry,
  re-score) never mistake them for hand models;
* binary, the dominant hand (:data:`DOMINANCE`): ``cv.run_experiment`` with ``y`` = 1 when the trial was made with
  the person's dominant hand (self-reported handedness), inputs the mirror-invariant kinematics only; registry row in
  ``results/registry_sec.csv``.

Windows. The app resolved its segmentation parameters per instructed speed and variant, so the oracle approach
window (the app's stored segment, used by every lane) of a slow, pause or jerky trial is chosen by its label. The
speed models therefore run on LABEL-BLIND windows (``sec_reseg``: the app's detector re-run with one parameter set
for every trial, every trial re-standardised on those bounds through ``standardize.canonical``); the same models on
the oracle windows are kept as the label-informed (optimistic) reference. The headline speed model
(:data:`HEADLINE`) is the pre-specified kinematic logistic regression on the label-blind windows.

The conditions (speed, grip) are TARGETS here; they never enter a model as inputs, and they only stratify results
after prediction. Derived files (``tables/sec``): the per-trial feature tables, the re-segmentation and its arrays,
the design of the protocol (speed by grip and by position in the session), the mirror check of the even features on
every trial, the label-permutation baseline of the dominance model and its test, the per-person dominance AUCs, the
paired dominant vs non-dominant comparison, the speed manipulation check (both windows, with the position-adjusted
contrasts), the univariate speed effects, the speed models within grip strata, the paired model comparisons, the
grip errors and the handedness concordance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import sec_features as F
from stage2.lanes import sec_models as M
from stage2.lanes import sec_reseg as RS
from stage2.pipeline import Lane, file_current, selected

L = Lane('sec')
#: one-thread worker processes = the lane's thread budget (the parent waits while they run)
N_WORKERS = 2

SPEED_CLASSES: tuple[str, ...] = ('fast', 'natural', 'slow')
GRIP_CLASSES: tuple[str, ...] = ('upright', 'top_left', 'top_right')
SIDE_CLASSES: tuple[str, ...] = ('upright', 'sideways')
#: column, classes, suffix stripped from the stored value, and an optional grouping of the stripped values
TASKS: dict[str, dict[str, Any]] = {
    'speed': {'column': 'speed', 'classes': SPEED_CLASSES, 'suffix': '', 'group': None},
    'grip': {'column': 'orientation', 'classes': GRIP_CLASSES, 'suffix': '_toward', 'group': None},
    'side': {'column': 'orientation', 'classes': SIDE_CLASSES, 'suffix': '_toward',
             'group': {'upright': 'upright', 'top_left': 'sideways', 'top_right': 'sideways'}},
}
FEATURE_LABELS = {'duration': 'log duration only', 'even': 'mirror-invariant kinematics',
                  'up_before': 'up vector before the onset', 'grip': 'device-frame orientation',
                  'rich': 'summary statistics of every channel',
                  'ang': 'duration and angular kinematics',
                  'ang_common': 'duration and angular kinematics',
                  'duration_common': 'log duration only', 'even_common': 'mirror-invariant kinematics',
                  'rich_common': 'summary statistics of every channel',
                  'up_before_common': 'up vector before the onset (the grip)'}
#: feature sets computed on the label-blind windows of sec_reseg (all others: the oracle windows of the cache)
LABEL_BLIND_FEATURES = frozenset({'ang_common', 'duration_common', 'even_common', 'rich_common', 'up_before_common'})
MODEL_LABELS = {'lr': 'nested L2 multinomial logistic regression', 'gbm': 'nested LightGBM'}
#: the trial sets of the experiments ('+' joins several: their intersection)
SUBSET_LABELS = {'all': 'all primary', 'standard': 'standard-variant trials',
                 'reseg_ok': 'trials with a label-blind end'}
#: the multi-class experiments. role: baseline (one block), interpretable, rich, nuisance (what a grip proxy or the
#: grip itself gives), sensitivity (a subset of trials), segmentation (the stream features on both windows)
MULTICLASS: dict[str, dict[str, str]] = {
    # instructed speed on LABEL-BLIND windows: the sensor-only estimates
    'sec_speed_dur_common': {'task': 'speed', 'features': 'duration_common', 'model': 'lr', 'subset': 'reseg_ok',
                             'role': 'baseline'},
    'sec_speed_lr_common': {'task': 'speed', 'features': 'even_common', 'model': 'lr', 'subset': 'reseg_ok',
                            'role': 'interpretable'},
    'sec_speed_gbm_common': {'task': 'speed', 'features': 'rich_common', 'model': 'gbm', 'subset': 'reseg_ok',
                             'role': 'rich'},
    'sec_speed_grip': {'task': 'speed', 'features': 'up_before_common', 'model': 'lr', 'subset': 'reseg_ok',
                       'role': 'nuisance'},
    'sec_speed_lr_std': {'task': 'speed', 'features': 'even_common', 'model': 'lr', 'subset': 'standard+reseg_ok',
                         'role': 'sensitivity'},
    # instructed speed on the ORACLE windows (label-informed for slow, pause and jerky trials): the reference
    'sec_speed_dur': {'task': 'speed', 'features': 'duration', 'model': 'lr', 'subset': 'all', 'role': 'baseline'},
    'sec_speed_lr': {'task': 'speed', 'features': 'even', 'model': 'lr', 'subset': 'all', 'role': 'interpretable'},
    'sec_speed_gbm': {'task': 'speed', 'features': 'rich', 'model': 'gbm', 'subset': 'all', 'role': 'rich'},
    'sec_speed_ang': {'task': 'speed', 'features': 'ang', 'model': 'lr', 'subset': 'reseg_ok',
                      'role': 'segmentation'},
    'sec_speed_ang_common': {'task': 'speed', 'features': 'ang_common', 'model': 'lr', 'subset': 'reseg_ok',
                             'role': 'segmentation'},
    # instructed grip
    'sec_grip_up': {'task': 'grip', 'features': 'up_before', 'model': 'lr', 'subset': 'all', 'role': 'baseline'},
    'sec_grip_lr': {'task': 'grip', 'features': 'grip', 'model': 'lr', 'subset': 'all', 'role': 'interpretable'},
    'sec_grip_gbm': {'task': 'grip', 'features': 'rich', 'model': 'gbm', 'subset': 'all', 'role': 'rich'},
    # does the speed models' kinematics carry the grip? (the design confounds speed with the grip)
    'sec_side_even': {'task': 'side', 'features': 'even_common', 'model': 'lr', 'subset': 'reseg_ok',
                      'role': 'nuisance'},
}
#: the headline model of each task (pre-specified model class and features: the interpretable model the researcher
#: asked for; for speed on the label-blind windows, the only sensor-only windows)
HEADLINE = {'speed': 'sec_speed_lr_common', 'grip': 'sec_grip_lr'}
#: the label-informed (oracle-window) counterpart of each label-blind speed model
ORACLE_TWIN: dict[str, str] = {'sec_speed_lr_common': 'sec_speed_lr', 'sec_speed_gbm_common': 'sec_speed_gbm',
                               'sec_speed_dur_common': 'sec_speed_dur', 'sec_speed_ang_common': 'sec_speed_ang'}
#: the segmentation check of the headline: the same model on label-blind and on oracle windows
SEGMENTATION_PAIR = ('sec_speed_lr_common', 'sec_speed_lr')
#: paired model comparisons (A vs B on the trials both scored: a model of a subset of trials is compared on that
#: subset only); one family per metric over every pair, Holm-adjusted; balanced accuracy is the primary metric
MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ('sec_speed_lr_common', 'sec_speed_dur_common'),     # features beyond the duration (label-blind windows)
    ('sec_speed_gbm_common', 'sec_speed_lr_common'),     # model class (label-blind windows)
    ('sec_speed_lr_common', 'sec_speed_grip'),           # beyond the grip prior (the design confound)
    ('sec_speed_lr_std', 'sec_speed_lr_common'),         # training trials
    SEGMENTATION_PAIR,                                   # windows: label-blind - oracle, the headline model
    ('sec_speed_gbm_common', 'sec_speed_gbm'),           # windows, LightGBM
    ('sec_speed_ang_common', 'sec_speed_ang'),           # windows, the stream features only
    ('sec_speed_gbm', 'sec_speed_lr'),                   # model class on the oracle windows
    ('sec_grip_lr', 'sec_grip_up'), ('sec_grip_gbm', 'sec_grip_lr'))
#: the metric families of MODEL_PAIRS (key prefix -> what is compared); Holm within each
PAIR_FAMILIES: dict[str, str] = {'d_bacc': 'balanced accuracy (primary)', 'd_f1': 'macro-F1', 'd_acc': 'accuracy',
                                 'd_ll': 'log loss'}
GRIP_MODELS: tuple[str, ...] = tuple(n for n, s in MULTICLASS.items() if s['task'] == 'grip')
#: the speed models evaluated within grip strata after prediction, and the pairs compared there
STRATA_MODELS: tuple[str, ...] = ('sec_speed_lr_common', 'sec_speed_gbm_common', 'sec_speed_lr', 'sec_speed_gbm')
STRATA_PAIRS: tuple[tuple[str, str], ...] = (('sec_speed_gbm_common', 'sec_speed_lr_common'),
                                             ('sec_speed_gbm', 'sec_speed_lr'))
#: grip strata (instructed grips); a speed class enters a stratum only with trials of at least MIN_PERSONS persons
GRIP_STRATA: dict[str, tuple[str, ...]] = {'upright': ('upright',), 'sideways': ('top_left', 'top_right')}
MIN_PERSONS = 5
DOMINANCE = 'sec_dom_lr'
#: exploratory: the same question with LightGBM (a second model class tried after the pre-specified LR)
DOMINANCE_GBM = 'sec_dom_gbm'
N_PERM = 200                     # label permutations (within person) of the dominance baseline
N_PERM_CHUNKS = 4
#: the paired dominant vs non-dominant comparison: duration, peak angular speed, smoothness (SPARC, LDLJ), path
#: lengths; features compared as ratios (difference of logs) are in LOG_FEATURES (duration_log is already a log)
PAIRED_FEATURES: tuple[str, ...] = ('duration_log', 'ang_speed_peak', 'ang_sparc', 'ang_ldlj', 'ang_path', 'lin_path')
LOG_FEATURES = frozenset({'ang_speed_peak', 'ang_path', 'lin_path'})
#: the conditions a matched comparison holds fixed (instructed; used after prediction only, never as inputs)
MATCH_CELLS: tuple[str, ...] = ('speed', 'grip', 'variant')
#: the manipulation check: measure -> expected order of the per-person medians (smallest first)
MANIP_MEASURES: dict[str, tuple[str, ...]] = {'duration_log': ('fast', 'natural', 'slow'),
                                              'ang_speed_peak': ('slow', 'natural', 'fast')}
#: standard: standard-variant trials, oracle windows; all: every trial, oracle windows; common: standard-variant
#: trials on the label-blind windows of sec_reseg (the sensitivity of the check to the per-speed segmentation)
MANIP_SUBSETS: tuple[str, ...] = ('standard', 'all', 'common')
#: the position-adjusted fast vs natural contrasts (the protocol puts the fast trials earlier in the session)
ADJ_CONTRASTS: tuple[str, ...] = ('duration_log', 'ang_speed_peak', 'duration_overlap')


# ───────────────────────────── inputs ─────────────────────────────


def task_labels(trials: pd.DataFrame, task: str) -> np.ndarray:
    """Class index (0..k-1) of every trial for a multi-class task; raises on a value that is not a class."""
    t = TASKS[task]
    vals = trials[t['column']].astype(str)
    if t['suffix']:
        vals = vals.str.replace(t['suffix'] + '$', '', regex=True)
    if t['group']:
        vals = vals.map(t['group']).fillna(vals)
    m = {c: i for i, c in enumerate(t['classes'])}
    lab = vals.map(m)
    if lab.isna().any():
        raise ValueError(f"{task}: values {sorted(set(vals[lab.isna()]))} are not classes {t['classes']}")
    return lab.astype(int).to_numpy()


def feature_table(which: str, tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    up = ['up_before_x', 'up_before_y', 'up_before_z']
    table = {'duration': ('even', ['duration_log']), 'even': ('even', list(F.EVEN_FEATURES)),
             'ang': ('even', list(RS.ANG_FEATURES)), 'up_before': ('grip', up), 'grip': ('grip', list(F.GRIP_FEATURES)),
             'rich': ('rich', None), 'duration_common': ('even_common', ['duration_log']),
             'even_common': ('even_common', list(F.EVEN_FEATURES)), 'ang_common': ('even_common', list(RS.ANG_FEATURES)),
             'up_before_common': ('grip_common', up), 'rich_common': ('rich_common', None)}
    if which not in table:
        raise KeyError(which)
    src, cols = table[which]
    if src not in tabs:
        raise KeyError(f'feature set {which!r} needs {src!r}: run the re-segmentation (sec_reseg) first')
    return tabs[src] if cols is None else tabs[src][cols]


def subset_rows(trials: pd.DataFrame, subset: str, tabs: dict[str, pd.DataFrame] | None = None) -> np.ndarray:
    """The rows of a trial set; '+' joins several sets (their intersection)."""
    rows = np.ones(len(trials), dtype=bool)
    for part in subset.split('+'):
        if part == 'all':
            continue
        if part == 'standard':
            rows &= (trials['variant'].astype(str) == 'standard').to_numpy()
        elif part == 'reseg_ok':
            if tabs is None or 'reseg' not in tabs:
                raise KeyError('subset reseg_ok needs the saved re-segmentation (sec_reseg)')
            rows &= tabs['reseg']['common_ok'].to_numpy(dtype=bool)
        else:
            raise KeyError(subset)
    return rows


def subset_label(subset: str) -> str:
    return ' with '.join(SUBSET_LABELS[p] for p in subset.split('+') if p != 'all') or SUBSET_LABELS['all']


def label_blind(name: str) -> bool:
    return MULTICLASS[name]['features'] in LABEL_BLIND_FEATURES


def windows_label(name: str) -> str:
    """'label-blind' / 'oracle' (the speed models); grip models read the hold before the onset."""
    s = MULTICLASS[name]
    if label_blind(name):
        return 'label-blind'
    return 'oracle' if s['task'] == 'speed' else 'oracle (hold before the onset)'


def dominant_labels(d: Any) -> tuple[np.ndarray, np.ndarray]:
    """(y, rows): y = 1 when the trial was made with the person's dominant hand (self-reported handedness left or
    right; persons with another answer are left out: rows False)."""
    hs = d.persons.set_index('person_id')['handedness_self'].astype(str)
    h = d.trials['person_id'].map(hs).astype(str).to_numpy()
    rows = np.isin(h, ['left', 'right'])
    y = (d.trials['hand'].astype(str).to_numpy() == h).astype(int)
    return y, rows


RESEG_TABLE = 'sec_reseg_bounds'
RESEG_ARRAYS = 'sec_reseg_arrays'
#: the files every experiment on the re-segmentation reads (output-relative; their hashes make it stale on change)
RESEG_FILES: tuple[str, ...] = (f'tables/sec/{RESEG_TABLE}.csv', f'tables/sec/{RESEG_ARRAYS}.npz')


def uses_reseg(name: str) -> bool:
    s = MULTICLASS[name]
    return label_blind(name) or 'reseg_ok' in s['subset'].split('+')


def multiclass_spec(name: str, tabs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The declared specification of a multi-class experiment (a saved result is current only when it matches)."""
    s = MULTICLASS[name]
    cols = list(feature_table(s['features'], tabs).columns)
    grid = M.LR_C_GRID if s['model'] == 'lr' else M.GBM_GRID
    spec = {**s, 'classes': list(TASKS[s['task']]['classes']), 'n_features': len(cols),
            'features_sha': hashlib.sha256('|'.join(cols).encode()).hexdigest()[:16],
            'grid': json.loads(json.dumps(list(grid))), 'gbm_fixed': M.GBM_FIXED if s['model'] == 'gbm' else None,
            'cap_per_person': C.CAP_PER_PERSON, 'seed': C.SEED, 'n_boot': C.N_BOOT}
    if uses_reseg(name):                     # the windows depend on the common segmentation parameters
        spec['segmentation'] = RS.common_params()
    return spec


def dominance_spec() -> dict[str, Any]:
    return {'features': list(F.EVEN_FEATURES), 'grid': list(M.LR_C_GRID), 'solver': 'newton-cholesky'}


DOM_MODEL = 'nested L2 logistic regression on the mirror-invariant kinematics'
DOM_NOTES = ('target: the trial was made with the dominant hand (self-reported handedness), NOT the hand model; '
             'inputs: the mirror-invariant kinematic features only; C tuned on fold.inner_splits()')


def result_file(name: str):
    return L.res_dir / f'{L.check_name(name)}.json'


def multiclass_current(name: str, tabs: dict[str, pd.DataFrame]) -> bool:
    """Saved, made from the current cache, with the current declared spec and (experiments on the re-segmentation)
    from the current re-segmentation files (their recorded hashes, ``pipeline.file_current``)."""
    p = result_file(name)
    if not file_current(p):
        return False
    try:
        saved = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    if uses_reseg(name) and set(saved.get('provenance_inputs') or {}) != {f'reports/stage2/{f}' for f in RESEG_FILES}:
        return False
    try:
        return saved.get('spec') == json.loads(json.dumps(multiclass_spec(name, tabs)))
    except KeyError:
        return False


def load_multiclass(name: str) -> dict[str, Any]:
    return L.read_json(name, folder='results')


# ───────────────────────────── the label-blind re-segmentation ─────────────────────────────


#: the columns of the saved re-segmentation (no app identifier)
RESEG_COLUMNS = ('trial_key', 'set', 'person_id', 'speed', 'variant', 'onset_ms', 'end_ms', 'common_onset_ms',
                 'common_end_ms', 'common_status', 'strict_end_ms', 'strict_status', 'return_onset_ms', 'params_differ',
                 'reproduced', 'common_ok', 'same_bounds', 'strict_ok', 'duration_s', 'common_duration_s')


def reseg_inputs() -> list[Any]:
    """The code the re-segmentation depends on: this lane's module, the cache's standardisation and the app's
    detector port with its parameter tables."""
    import motionlab.conventions as mc
    import motionlab.segment as ms
    from stage2 import standardize as SD
    return [RS, SD, ms, mc]


def reseg_spec() -> dict[str, Any]:
    return {'common_params': RS.common_params(), 'default_params': RS.default_params(),
            'strict_params': RS.strict_params(), 'n_pre': RS.N_PRE, 'sets': ['primary', 'natural_use']}


def reseg_table(trials: pd.DataFrame) -> pd.DataFrame | None:
    """The saved re-segmentation (``tables/sec/sec_reseg_bounds.csv``) in the row order of ``trials``; None when it
    is missing or does not cover every trial (then it is recomputed)."""
    p = L.tab_dir / f'{RESEG_TABLE}.csv'
    if not p.exists():
        return None
    R = pd.read_csv(p, keep_default_na=False, na_values=[''])
    if any(c not in R.columns for c in RESEG_COLUMNS):      # written by an older version: recomputed
        return None
    R = R.set_index('trial_key').reindex(trials['trial_key'].astype(str))
    if R['common_ok'].isna().any():
        return None
    for c in ('reproduced', 'common_ok', 'same_bounds', 'strict_ok'):
        R[c] = R[c].astype(str).str.lower().eq('true')
    R['params_differ'] = R['params_differ'].fillna('').astype(str)
    return R.reset_index()


def reseg_arrays(trials: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """The label-blind arrays (approach, hold before the onset and its mask) in the row order of ``trials``."""
    p = L.tab_dir / f'{RESEG_ARRAYS}.npz'
    if not p.exists():
        return None
    z = RS.read_npz(p)
    pos = {k: i for i, k in enumerate(z['trial_key'].tolist())}
    keys = trials['trial_key'].astype(str).tolist()
    if any(k not in pos for k in keys):
        return None
    idx = np.array([pos[k] for k in keys], dtype=int)
    return {k: z[k][idx] for k in ('approach', 'pre', 'pre_mask')}


def common_tables(d: Any, R: pd.DataFrame, arr: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """The EVEN, GRIP and RICH features on the label-blind windows (NaN where the common segmentation found no end):
    the approach-array features from the arrays re-standardised on those bounds, the angular ones from the trial's
    60 Hz stream between them (the functions of ``sec_features``)."""
    ok = R['common_ok'].to_numpy(dtype=bool)
    on = R['common_onset_ms'].to_numpy(float)
    en = R['common_end_ms'].to_numpy(float)
    rich_cols = list(F.rich_features(arr['approach'][:1], arr['pre'][:1], arr['pre_mask'][:1],
                                     pd.DataFrame(np.zeros((1, len(F.EVEN_FEATURES))), columns=list(F.EVEN_FEATURES)))
                     .columns)
    even = pd.DataFrame(np.nan, index=range(d.n), columns=list(F.EVEN_FEATURES))
    grip = pd.DataFrame(np.nan, index=range(d.n), columns=list(F.GRIP_FEATURES))
    rich = pd.DataFrame(np.nan, index=range(d.n), columns=rich_cols)
    if ok.any():
        ii = np.flatnonzero(ok)
        gyro = [F.approach_gyro(d.stream(int(i)), on[i], en[i]) for i in ii]
        e = F.even_features(arr['approach'][ii], (en[ii] - on[ii]) / 1000.0, gyro)
        g = F.grip_features(arr['approach'][ii], arr['pre'][ii], arr['pre_mask'][ii])
        r = F.rich_features(arr['approach'][ii], arr['pre'][ii], arr['pre_mask'][ii], e)
        even.loc[ii] = e.to_numpy()
        grip.loc[ii] = g.to_numpy()
        rich.loc[ii] = r.to_numpy()
    return {'even_common': even.astype(float), 'grip_common': grip.astype(float), 'rich_common': rich.astype(float)}


def reseg_current(d: Any, nat: Any) -> bool:
    p = L.tab_dir / 'sec_reseg.json'
    if not (p.exists() and file_current(p)):
        return False
    if L.read_json('sec_reseg').get('spec') != json.loads(json.dumps(reseg_spec())):
        return False
    return all(reseg_table(x.trials) is not None and reseg_arrays(x.trials) is not None for x in (d, nat))


def run_reseg(d: Any, nat: Any) -> None:
    """Re-run the app's detector on every primary and free-choice trial (stored, common and strict parameters),
    re-standardise on the label-blind bounds, and save the bounds, the arrays and the summary."""
    from motionlab.io import load_dataset
    ds = load_dataset(C.SNAPSHOT_DIR, simulated='include')
    parts, arrays, checks = [], [], {}
    for name, x in (('primary', d), ('natural_use', nat)):
        R, A, v = RS.resegment(x, ds)
        R['set'] = name
        parts.append(R)
        arrays.append(A)
        checks[name] = v
    R = pd.concat(parts, ignore_index=True)
    A = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    L.export_table(R[list(RESEG_COLUMNS)], RESEG_TABLE, tex=False)
    RS.write_npz(L.tab_dir / f'{RESEG_ARRAYS}.npz', A)
    prim = R[R['set'] == 'primary']
    natr = R[R['set'] == 'natural_use']
    L.write_json('sec_reseg', {**RS.summary(prim), 'restandardize_check': checks,
                               'natural_use': {'n': int(len(natr)), 'n_common_ok': int(natr['common_ok'].sum()),
                                               'n_same_bounds': int(natr['same_bounds'].sum()),
                                               'n_reproduced': int(natr['reproduced'].sum())},
                               'spec': reseg_spec()}, inputs=reseg_inputs())


# ───────────────────────────── data ─────────────────────────────


def _data() -> tuple[Any, Any, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """The primary and natural-use sets and their feature tables: the oracle-window sets ('even', 'grip', 'rich') and,
    when the re-segmentation is saved, 'reseg' (its bounds) and the label-blind sets ('even_common', 'grip_common',
    'rich_common')."""
    from stage2.data import load_cache
    d, nat = load_cache(), load_cache(('natural_use',))
    out = []
    for x in (d, nat):
        tabs = F.build_all(x)
        R, arr = reseg_table(x.trials), reseg_arrays(x.trials)
        if R is not None and arr is not None:
            tabs['reseg'] = R
            tabs.update(common_tables(x, R, arr))
        out.append(tabs)
    return d, nat, out[0], out[1]


def check_constants() -> None:
    """sec_features keeps two cache parameters as literals (editing that module recomputes the permutation baseline);
    they must equal the cache's (config.FS_HZ, config.FIXED_PRE_S)."""
    expected = int(round(C.FIXED_PRE_S * C.FS_HZ))
    if float(F.FS_HZ) != float(C.FS_HZ) or int(F.FIXED_ONSET_INDEX) != expected:
        raise ValueError(f'sec_features: FS_HZ {F.FS_HZ} / FIXED_ONSET_INDEX {F.FIXED_ONSET_INDEX} differ from the '
                         f'cache (config.FS_HZ {C.FS_HZ}, onset index {expected}); update sec_features.py')


def mirror_check(d: Any, tabs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Every EVEN feature of every trial against the same feature of the trial's mirror image
    (``standardize.mirror_channels`` on the cached arrays; the gyro channels' mirror signs on the 60 Hz stream)."""
    from stage2.standardize import MIRROR_SIGN, mirror_channels
    sg = np.asarray(MIRROR_SIGN, dtype=float)[[F.CHANNELS.index(f'gyro_{a}') for a in 'xyz']]
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    mir = F.even_features(mirror_channels(d.approach), dur, [g * sg for g in F.gyro_profiles(d)])
    orig = tabs['even'][list(F.EVEN_FEATURES)]
    diff = (mir - orig).abs().max()
    same = (mir.to_numpy() == orig.to_numpy()).all(axis=1)
    return {'n_trials': int(d.n), 'n_features': len(F.EVEN_FEATURES), 'gyro_mirror_signs': sg.tolist(),
            'max_abs_diff': {f: float(diff[f]) for f in F.EVEN_FEATURES}, 'max_abs_diff_all': float(diff.max()),
            'n_identical_trials': int(same.sum())}


# ───────────────────────────── experiments (run in one-thread workers) ─────────────────────────────


def run_multiclass(name: str) -> str:
    """One multi-class experiment, saved to results/<name>.json."""
    d, nat, tabs, tabs_nat = _data()
    s = MULTICLASS[name]
    classes = TASKS[s['task']]['classes']
    k = len(classes)
    rows = subset_rows(d.trials, s['subset'], tabs)
    y = task_labels(d.trials, s['task'])[rows]
    Xdf = feature_table(s['features'], tabs)
    cols = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)[rows]
    if not np.isfinite(X).all():
        raise ValueError(f'{name}: non-finite features in the selected rows')
    meta = d.trials.loc[rows].reset_index(drop=True)
    fit = M.lr_fitter(k) if s['model'] == 'lr' else M.gbm_fitter(k)
    extra = None
    if s['task'] == 'speed' and 'standard' not in s['subset'].split('+'):
        try:                                       # the free-choice trials, each scored by its person's fold model
            Xn = feature_table(s['features'], tabs_nat).to_numpy(dtype=float)
            okn = np.isfinite(Xn).all(axis=1)
            extra = (Xn[okn], nat.trials.loc[okn].reset_index(drop=True))
        except KeyError:
            extra = None
    # coefficients are kept for the multinomial LRs (sec_models.lr_coefficients expects one row per class; a
    # two-class LR has a single row)
    out = M.loso_multiclass(X, y, meta, k, fit, feature_names=cols, extra=extra,
                            keep_coef=s['model'] == 'lr' and k > 2)
    P = out['proba']
    person = meta['person_id'].astype(str).to_numpy()
    met = M.multiclass_metrics(y, P, classes)
    met['n_persons'] = float(len(set(person.tolist())))
    ci = M.bootstrap_multiclass(person, y, P, classes)
    cm = M.confusion(y, np.argmax(P, axis=1), k)
    pp = M.per_person_multiclass(person, y, P, k)
    pred = pd.DataFrame({'trial_id': meta['trial_id'].astype(str), 'trial_key': meta['trial_key'].astype(str),
                         'person_id': person, 'hand': meta['hand'].astype(str), 'y': y,
                         'y_pred': np.argmax(P, axis=1), 'fold': out['fold']})
    for c, cname in enumerate(classes):
        pred[f'p_{cname}'] = P[:, c]
    doc: dict[str, Any] = {
        'name': name, 'task': s['task'], 'classes': list(classes), 'features': cols, 'spec': multiclass_spec(name, tabs),
        'label': f"{MODEL_LABELS[s['model']]}, {FEATURE_LABELS[s['features']]}, {windows_label(name)} windows",
        'windows': windows_label(name),
        'scheme': ('leave-one-person-out (cv.loso_splits), nested tuning on cv.inner_splits of the training persons '
                   '(class-balanced log loss), D2 cap (cv.cap_rows), person-cluster bootstrap (cv.person_bootstrap)'),
        'n_folds': out['n_folds'], 'n_configs_tried': 1, 'seed': C.SEED, 'n_boot': C.N_BOOT,
        'leak_check': f"passed (all {len(out['leak_check'])} folds)" if all(v == 'passed' for v in out['leak_check'].values())
        else json.dumps(out['leak_check']),
        'metrics': met, 'ci95': {kk: list(v) for kk, v in ci.items()}, 'confusion': cm.tolist(),
        'per_person': pp.to_dict(orient='records'),
        'fold_info': {str(f): v for f, v in out['fold_info'].items()},
        'trial_predictions': pred.to_dict(orient='records'),
    }
    if extra is not None:
        Pe = out['extra_proba']
        ok = np.isfinite(Pe).all(axis=1)
        mt = extra[1]
        ex = pd.DataFrame({'trial_key': mt['trial_key'].astype(str), 'person_id': mt['person_id'].astype(str),
                           'instructed': mt[TASKS[s['task']]['column']].astype(str),
                           'y_pred': np.where(ok, np.argmax(np.nan_to_num(Pe), axis=1), -1)})[ok]
        doc['extra'] = {'set': 'natural_use', 'n': int(ok.sum()),
                        'predicted_counts': {c: int(np.sum(ex['y_pred'] == i)) for i, c in enumerate(classes)},
                        'rows': ex.to_dict(orient='records')}
    L.write_json(name, doc, folder='results', inputs=list(RESEG_FILES) if uses_reseg(name) else None)
    return name


def run_dominance() -> str:
    """The dominant-hand model through cv.run_experiment (binary target, mirror-invariant inputs)."""
    d, _, tabs, _ = _data()
    y, rows = dominant_labels(d)
    X = feature_table('even', tabs).loc[rows].reset_index(drop=True)
    meta = d.trials.loc[rows].reset_index(drop=True)
    fit = M.lr_fitter(2)

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        model, info = fit(fold.X_train.to_numpy(dtype=float), fold.y_train, fold.groups_train, fold.seed)
        return M.proba_full(model, fold.X_test.to_numpy(dtype=float), 2)[:, 1], {'C': info['config']}

    L.run_experiment(DOMINANCE, fit_predict, X, meta, y=y[rows], model=DOM_MODEL,
                     params={'C_grid': list(M.LR_C_GRID)}, notes=DOM_NOTES, n_configs_tried=1, spec=dominance_spec())
    return DOMINANCE


DOM_GBM_MODEL = 'nested LightGBM on the mirror-invariant kinematics (exploratory)'
DOM_GBM_NOTES = ('exploratory second model class for the dominant-hand target (after the pre-specified LR); inputs: '
                 'the mirror-invariant kinematic features only; configuration tuned on fold.inner_splits()')


def run_dominance_gbm() -> str:
    """The dominant-hand question with nested LightGBM (exploratory, n_configs_tried=2). The tuned fit is cached by a
    hash of the TRAINING rows (README section 11), so the harness's leak check re-runs the prediction step on half
    of the test rows without a second tuning."""
    d, _, tabs, _ = _data()
    y, rows = dominant_labels(d)
    X = feature_table('even', tabs).loc[rows].reset_index(drop=True)
    meta = d.trials.loc[rows].reset_index(drop=True)
    fit = M.gbm_fitter(2)
    cache: dict[str, Any] = {}

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        Xtr = fold.X_train.to_numpy(dtype=float)
        key = hashlib.sha256(Xtr.tobytes() + np.asarray(fold.y_train).tobytes() + str(fold.seed).encode()).hexdigest()
        if key not in cache:
            cache[key] = fit(Xtr, fold.y_train, fold.groups_train, fold.seed)
        model, info = cache[key]
        return M.proba_full(model, fold.X_test.to_numpy(dtype=float), 2)[:, 1], {'config': info['config']}

    L.run_experiment(DOMINANCE_GBM, fit_predict, X, meta, y=y[rows], model=DOM_GBM_MODEL,
                     params={'grid': json.loads(json.dumps(list(M.GBM_GRID))), 'fixed': M.GBM_FIXED},
                     notes=DOM_GBM_NOTES, n_configs_tried=2, spec=dominance_spec())
    return DOMINANCE_GBM


def run_null_chunk(chunk: int) -> dict[str, Any]:
    """Permutations chunk*n .. of the label-permutation baseline: the dominance labels shuffled WITHIN each person,
    then the same nested LOSO pipeline (sec_models.loso_binary_scores); pooled and mean within-person AUC."""
    d, _, tabs, _ = _data()
    y, rows = dominant_labels(d)
    X = feature_table('even', tabs).loc[rows].to_numpy(dtype=float)
    meta = d.trials.loc[rows].reset_index(drop=True)
    y = y[rows]
    person = meta['person_id'].astype(str).to_numpy()
    per = int(math.ceil(N_PERM / N_PERM_CHUNKS))
    idx = range(chunk * per, min(N_PERM, (chunk + 1) * per))
    pooled, within = [], []
    for b in idx:
        rng = np.random.default_rng([C.SEED, 11, b])
        yp = M.permute_within(person, y, rng)
        p = M.loso_binary_scores(X, yp, meta, M.lr_fitter(2))
        pooled.append(cv.roc_auc(yp, p))
        within.append(float(np.nanmean(M.within_person_auc(person, yp, p)['auc'])))
    out = {'chunk': chunk, 'perm': list(idx), 'pooled_auc': pooled, 'within_auc': within}
    L.write_json(_null_part_name(chunk), {**out, 'spec': _null_part_spec(chunk)}, inputs=list(NULL_INPUTS))
    return out


def _job(kind: str, arg: Any) -> Any:
    if kind == 'mc':
        return run_multiclass(arg)
    if kind == 'dom':
        return run_dominance() if arg == DOMINANCE else run_dominance_gbm()
    if kind == 'null':
        return run_null_chunk(arg)
    raise KeyError(kind)


def _run_jobs(jobs: Sequence[tuple[str, Any]]) -> list[Any]:
    """Run jobs in N_WORKERS one-thread worker processes (the children inherit one-thread BLAS / OpenMP settings)."""
    if not jobs:
        return []
    keep = {v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    for v in keep:
        os.environ[v] = '1'
    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [(kind, arg, ex.submit(_job, kind, arg)) for kind, arg in jobs]
            out = []
            for kind, arg, f in futs:
                r = f.result()
                print(f'  sec: {kind} {arg} done', flush=True)
                out.append(r)
            return out
    finally:
        for v, val in keep.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


# ───────────────────────────── derived files ─────────────────────────────


def trial_table(d: Any, tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per trial: keys, conditions, hand, dominance, the relative position in the person's session sequence, and the
    EVEN and GRIP features of the oracle windows (the figures' input)."""
    y, rows = dominant_labels(d)
    hs = d.persons.set_index('person_id')['handedness_self'].astype(str)
    t = d.trials
    base = pd.DataFrame({'trial_key': t['trial_key'].astype(str), 'person_id': t['person_id'].astype(str),
                         'platform': t['platform'].astype(str), 'hand': t['hand'].astype(str),
                         'handedness': t['person_id'].map(hs).astype(str),
                         'dominant': np.where(rows, y, -1), 'speed': t['speed'].astype(str),
                         'grip': t['orientation'].astype(str).str.replace('_toward$', '', regex=True),
                         'variant': t['variant'].astype(str), 'position': relative_position(t),
                         'grip_app': t['rotation_start'].astype(str), 'grip_mismatch': t['grip_mismatch'].astype(bool),
                         'gravity_sign_source': t['gravity_sign_source'].astype(str)})
    return pd.concat([base, tabs['even'].reset_index(drop=True), tabs['grip'].reset_index(drop=True)], axis=1)


def relative_position(t: pd.DataFrame) -> np.ndarray:
    """Each trial's place in its person's sequence of primary trials (sessions, blocks and trials in order), scaled to
    0 (first) .. 1 (last)."""
    order = t.sort_values(['person_id', 'session_order', 'block_index', 'trial_index'], kind='mergesort')
    rank = order.groupby('person_id').cumcount()
    n = order.groupby('person_id')['person_id'].transform('size')
    pos = (rank / np.maximum(n - 1, 1)).reindex(t.index)
    return pos.to_numpy(dtype=float)


def common_trial_table(T: pd.DataFrame, tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The trial table with every EVEN feature measured on the label-blind windows."""
    Tc = T.copy()
    for c in F.EVEN_FEATURES:
        Tc[c] = tabs['even_common'][c].to_numpy()
    Tc['common_ok'] = tabs['reseg']['common_ok'].to_numpy(dtype=bool)
    return Tc


def _wilcoxon(v: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test over persons (exact when possible); (statistic, p)."""
    from scipy.stats import wilcoxon
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.all(v == 0):
        return (math.nan, math.nan)
    r = wilcoxon(v)
    return float(r.statistic), float(r.pvalue)


def _boot_median_ci(v: np.ndarray, seed: int = C.SEED) -> tuple[float, float]:
    return M.boot_mean_ci(v, seed=seed, stat=lambda a: np.median(a, axis=1))


def _contrast_row(diff: pd.Series, **meta: Any) -> dict[str, Any]:
    """A per-person log-ratio contrast (expected direction > 0): persons, persons as expected, the median ratio with
    its person-bootstrap 95 % CI, the Wilcoxon signed-rank test."""
    v = diff.to_numpy(dtype=float)
    stat, p = _wilcoxon(v)
    lo, hi = _boot_median_ci(v)
    return {**meta, 'n_persons': int(v.size), 'n_expected': int((v > 0).sum()),
            'median_log_ratio': float(np.median(v)) if v.size else math.nan,
            'median_ratio': float(np.exp(np.median(v))) if v.size else math.nan,
            'median_ratio_lo': float(np.exp(lo)) if math.isfinite(lo) else math.nan,
            'median_ratio_hi': float(np.exp(hi)) if math.isfinite(hi) else math.nan,
            'wilcoxon_w': stat, 'p': p, 'per_person': {str(k): float(x) for k, x in diff.items()}}


def _adjusted_fast_natural(S: pd.DataFrame) -> list[dict[str, Any]]:
    """Fast vs natural adjusted for the position in the session (the protocol runs fast trials earlier and natural
    reaches shorten with practice): per person an OLS of the log measure on [1, natural, position] over the fast and
    natural trials (the natural coefficient: natural - fast in log units, sign set so that > 0 is as instructed), and
    the duration contrast with only the natural trials made no later than the person's last fast trial."""
    out = []
    G = S[S['speed'].isin(['fast', 'natural'])]
    for meas in ('duration_log', 'ang_speed_peak'):
        vals = {}
        for pid, g in G.groupby('person_id', sort=True):
            if g['speed'].nunique() < 2 or len(g) < 4:
                continue
            Xd = np.column_stack([np.ones(len(g)), (g['speed'] == 'natural').astype(float), g['position']])
            if np.linalg.matrix_rank(Xd) < 3:
                continue
            yv = g[meas].to_numpy(dtype=float)
            yv = yv if meas == 'duration_log' else np.log(yv)
            b = np.linalg.lstsq(Xd, yv, rcond=None)[0][1]
            vals[str(pid)] = float(b if meas == 'duration_log' else -b)
        out.append(_contrast_row(pd.Series(vals, dtype=float), contrast=meas, measure=meas,
                                 low='fast' if meas == 'duration_log' else 'natural',
                                 high='natural' if meas == 'duration_log' else 'fast', adjustment='position (OLS)'))
    vals = {}
    for pid, g in G.groupby('person_id', sort=True):
        f = g[g['speed'] == 'fast']
        if f.empty:
            continue
        nat = g[(g['speed'] == 'natural') & (g['position'] <= f['position'].max())]
        if nat.empty:
            continue
        vals[str(pid)] = float(nat['duration_log'].median() - f['duration_log'].median())
    out.append(_contrast_row(pd.Series(vals, dtype=float), contrast='duration_overlap', measure='duration_log',
                             low='fast', high='natural', adjustment="natural trials up to the person's last fast trial"))
    for r, q in zip(out, M.holm([r['p'] for r in out])):
        r['p_holm'] = q
    return out


def speed_manipulation(T: pd.DataFrame, Tc: pd.DataFrame | None = None) -> dict[str, Any]:
    """Did the persons follow the speed instruction? Per person and instructed speed: the median of each measure;
    per adjacent contrast the per-person log ratio (expected direction > 0), the persons in the expected order, the
    median ratio with a person-bootstrap CI, a Wilcoxon signed-rank test over persons, Holm within each subset's
    family of contrasts; and the fast vs natural contrasts adjusted for the position in the session (their own Holm
    family). ``Tc``: the trial table measured on the label-blind windows (subset 'common')."""
    out: dict[str, Any] = {'measures': {k: list(v) for k, v in MANIP_MEASURES.items()}, 'subsets': {}}
    for sub in MANIP_SUBSETS:
        if sub == 'common':
            if Tc is None:
                continue
            S = Tc[(Tc['variant'] == 'standard') & Tc['common_ok']]
        else:
            S = T if sub == 'all' else T[T['variant'] == 'standard']
        med = S.groupby(['person_id', 'speed'])[list(MANIP_MEASURES)].median()
        cnt = S.groupby(['person_id', 'speed']).size().rename('n')
        per_rows = med.join(cnt).reset_index().to_dict(orient='records')
        contrasts = []
        full = {}
        for meas, order in MANIP_MEASURES.items():
            wide = med[meas].unstack('speed')
            for lo, hi in zip(order[:-1], order[1:]):
                both = wide[[lo, hi]].dropna()
                # log ratio in the expected direction (> 0 = as instructed); duration_log is already a log
                diff = (both[hi] - both[lo]) if meas == 'duration_log' else (np.log(both[hi]) - np.log(both[lo]))
                contrasts.append(_contrast_row(diff, measure=meas, low=lo, high=hi))
            allc = wide[list(order)].dropna()
            ok = np.ones(len(allc), dtype=bool)
            for lo, hi in zip(order[:-1], order[1:]):
                ok &= (allc[hi] > allc[lo]).to_numpy()
            full[meas] = {'n_persons': int(len(allc)), 'n_full_order': int(ok.sum()),
                          'persons_not_in_order': sorted(allc.index[~ok].astype(str).tolist())}
        for c, q in zip(contrasts, M.holm([c['p'] for c in contrasts])):
            c['p_holm'] = q
        mm = list(MANIP_MEASURES)
        w = {m: med[m].unstack('speed') for m in mm}
        idx = sorted(set.intersection(*[set(w[m][list(MANIP_MEASURES[m])].dropna().index) for m in mm]))
        okb = []
        for pid in idx:
            good = True
            for m in mm:
                o = MANIP_MEASURES[m]
                good &= all(w[m].loc[pid, b] > w[m].loc[pid, a] for a, b in zip(o[:-1], o[1:]))
            okb.append(good)
        out['subsets'][sub] = {'n_trials': int(len(S)), 'n_persons': int(S['person_id'].nunique()),
                               'per_person': per_rows, 'contrasts': contrasts, 'full_order': full,
                               'full_order_both': {'n_persons': len(idx), 'n_full_order': int(np.sum(okb))},
                               'adjusted': _adjusted_fast_natural(S)}
    return out


def speed_univariate(T: pd.DataFrame, Tc: pd.DataFrame | None = None) -> dict[str, Any]:
    """Which kinematic features carry the instructed speed, model-free: per person the difference of the medians
    (fast - natural, slow - natural; standard-variant trials) in units of the feature's SD over all trials; across
    persons its median, IQR, the persons with a positive difference and a Wilcoxon test, Holm over every
    feature x contrast; on the label-blind windows ('common', the headline's inputs) and on the oracle windows."""
    out: dict[str, Any] = {'subset': 'standard', 'windows': {}}
    for win, TT in (('common', Tc), ('oracle', T)):
        if TT is None:
            continue
        base = TT if win == 'oracle' else TT[TT['common_ok']]
        S = base[base['variant'] == 'standard']
        sd = base[list(F.EVEN_FEATURES)].std()
        rows = []
        for f in F.EVEN_FEATURES:
            wide = S.groupby(['person_id', 'speed'])[f].median().unstack('speed')
            for c in ('fast', 'slow'):
                both = wide[[c, 'natural']].dropna()
                v = ((both[c] - both['natural']) / sd[f]).to_numpy()
                stat, p = _wilcoxon(v)
                rows.append({'feature': f, 'contrast': f'{c} - natural', 'n_persons': int(v.size),
                             'median_std_diff': float(np.median(v)), 'q25': float(np.quantile(v, 0.25)),
                             'q75': float(np.quantile(v, 0.75)), 'n_positive': int((v > 0).sum()), 'p': p})
        for r, qq in zip(rows, M.holm([r['p'] for r in rows])):
            r['p_holm'] = qq
        out['windows'][win] = rows
    return out


def speed_followers(manip: dict[str, Any], name: str | None = None, subset: str = 'common') -> dict[str, Any]:
    """Descriptive (and partly circular: the duration is an input of the model): per person, the natural / fast
    ratio of the median durations (standard-variant trials of ``subset`` of the manipulation check) against the
    model's fast-vs-natural balanced accuracy on that person's standard-variant fast and natural trials, the class
    decided between fast and natural only (argmax of the two probabilities; chance 0.5); across persons a Spearman
    correlation and the mean of each group (shorter: ratio > 1)."""
    from scipy.stats import spearmanr
    name = name or HEADLINE['speed']
    r = load_multiclass(name)
    cls = r['classes']
    pr = pd.DataFrame(r['trial_predictions'])
    from stage2.data import load_cache
    var = load_cache().trials.set_index('trial_key')['variant'].astype(str)
    pr = pr[pr['trial_key'].map(var) == 'standard']
    fi, ni = cls.index('fast'), cls.index('natural')
    pr = pr[pr['y'].isin([fi, ni])].copy()
    pr['y2'] = np.where(pr[f'p_{cls[fi]}'] > pr[f'p_{cls[ni]}'], fi, ni)
    c = next(c for c in manip['subsets'][subset]['contrasts']
             if c['measure'] == 'duration_log' and c['low'] == 'fast' and c['high'] == 'natural')
    rows = []
    for pid, lr in c['per_person'].items():
        g = pr[pr['person_id'] == pid]
        rf = float(np.mean(g.loc[g['y'] == fi, 'y2'] == fi)) if (g['y'] == fi).any() else math.nan
        rn = float(np.mean(g.loc[g['y'] == ni, 'y2'] == ni)) if (g['y'] == ni).any() else math.nan
        rows.append({'person_id': pid, 'log_ratio': float(lr), 'ratio': float(np.exp(lr)), 'recall_fast': rf,
                     'recall_natural': rn, 'bacc_fast_natural': float(np.nanmean([rf, rn]))})
    D = pd.DataFrame(rows)
    rho, p = spearmanr(D['log_ratio'], D['bacc_fast_natural'])
    fol = D['log_ratio'] > 0
    return {'model': name, 'subset': subset, 'per_person': D.to_dict(orient='records'), 'spearman_rho': float(rho),
            'spearman_p': float(p), 'n_persons': int(len(D)), 'n_followed': int(fol.sum()),
            'n_not_followed': int((~fol).sum()),
            'bacc_followed': float(D.loc[fol, 'bacc_fast_natural'].mean()) if fol.any() else math.nan,
            'bacc_not_followed': float(D.loc[~fol, 'bacc_fast_natural'].mean()) if (~fol).any() else math.nan}


def speed_coefficients(name: str | None = None) -> dict[str, Any]:
    """The headline LR's log-odds contrasts per 1 SD of each feature (fast vs natural, slow vs natural), mean and
    range over the outer folds (from the coefficients saved per fold)."""
    name = name or HEADLINE['speed']
    r = load_multiclass(name)
    cls = r['classes']
    rows = []
    for f_i, f in enumerate(r['features']):
        for c in ('fast', 'slow'):
            vals = []
            for info in r['fold_info'].values():
                co = info['coef']
                vals.append(co[str(cls.index(c))][f_i] - co[str(cls.index('natural'))][f_i])
            rows.append({'feature': f, 'contrast': f'{c} vs natural', 'mean': float(np.mean(vals)),
                         'min': float(np.min(vals)), 'max': float(np.max(vals)), 'n_folds': len(vals)})
    return {'model': name, 'rows': rows}


def design(T: pd.DataFrame) -> dict[str, Any]:
    """The protocol's design, descriptive: instructed speed by grip (standard-variant and all trials), the per-person
    share of sideways grips per speed, the relative position of each speed in the person's sequence, and the practice
    trend of the natural reaches (within-person Spearman correlation of the log duration with the position)."""
    from scipy.stats import spearmanr
    out: dict[str, Any] = {}
    for sub, S in (('standard', T[T['variant'] == 'standard']), ('all', T)):
        ct = pd.crosstab(S['speed'], S['grip'])
        out[f'speed_by_grip_{sub}'] = {str(sp): {str(g): int(ct.loc[sp, g]) for g in ct.columns} for sp in ct.index}
    S = T[T['variant'] == 'standard'].assign(side=lambda x: (x['grip'] != 'upright').astype(float))
    share = S.groupby('speed')['side'].mean()
    out['share_sideways_standard'] = {str(k): float(v) for k, v in share.items()}
    pp = S.groupby(['person_id', 'speed'])['side'].mean().unstack('speed')
    fn = pp[['fast', 'natural']].dropna()
    out['persons_fast_more_sideways'] = {'n': int(len(fn)), 'n_more': int((fn['fast'] > fn['natural']).sum()),
                                         'n_less': int((fn['fast'] < fn['natural']).sum())}
    out['position_median'] = {str(k): float(v) for k, v in T.groupby('speed')['position'].median().items()}
    pm = T.groupby(['person_id', 'speed'])['position'].median().unstack('speed')
    for a in ('slow', 'fast'):
        both = pm[[a, 'natural']].dropna()
        out[f'persons_{a}_before_natural'] = {'n': int(len(both)), 'n_before': int((both[a] < both['natural']).sum())}
    N = T[(T['variant'] == 'standard') & (T['speed'] == 'natural')]
    rho = {}
    for pid, g in N.groupby('person_id', sort=True):
        if len(g) >= MIN_PERSONS:
            rho[str(pid)] = float(spearmanr(g['position'], g['duration_log']).correlation)
    v = np.array([x for x in rho.values() if math.isfinite(x)])
    stat, p = _wilcoxon(v)
    out['practice_natural'] = {'n_persons': int(v.size), 'median_rho': float(np.median(v)) if v.size else math.nan,
                               'n_negative': int((v < 0).sum()), 'wilcoxon_p': p, 'per_person': rho,
                               'min_trials': MIN_PERSONS}
    return out


def _stratum_metrics(pr: pd.DataFrame, k: int, classes: Sequence[str], included: Sequence[int]) -> dict[str, Any]:
    from stage2.lanes import sec_stats as SS
    sub = pr[pr['y'].isin(included)]
    y, yh = sub['y'].to_numpy(dtype=int), sub['y_pred'].to_numpy(dtype=int)
    cm = M.confusion(y, yh, k)
    return {'n': int(len(sub)), 'n_persons': int(sub['person_id'].nunique()),
            'balanced_accuracy': float(SS.balanced_accuracy_cm(cm)), 'macro_f1': float(SS.macro_f1_cm(cm)),
            'accuracy': float(np.mean(y == yh)) if y.size else math.nan,
            **{f'recall_{classes[c]}': M.recall(y, yh, c) for c in included},
            **{f'n_{classes[c]}': int(np.sum(y == c)) for c in included}}


def speed_strata(T: pd.DataFrame) -> dict[str, Any]:
    """The speed models within grip strata, after prediction (the design confounds speed with the grip: fast trials
    are more often sideways and slow trials almost never). Per model and stratum: the metrics over the speed classes
    with trials of at least MIN_PERSONS persons in the stratum, the predicted and the true class shares over every
    trial of the stratum; per STRATA_PAIRS and stratum: the paired balanced-accuracy and macro-F1 differences (person
    bootstrap, sign-flip / person-swap p), Holm within each metric over the strata comparisons."""
    from stage2.lanes import sec_stats as SS
    grip = T.set_index('trial_key')['grip']
    out: dict[str, Any] = {'strata': {g: list(v) for g, v in GRIP_STRATA.items()}, 'models': {}, 'pairs': [],
                           'min_persons': MIN_PERSONS}
    preds = {}
    for name in STRATA_MODELS:
        r = load_multiclass(name)
        cls, k = r['classes'], len(r['classes'])
        pr = pd.DataFrame(r['trial_predictions'])
        pr['stratum'] = pr['trial_key'].map(grip).map({g: s for s, gs in GRIP_STRATA.items() for g in gs})
        preds[name] = pr
        out['models'][name] = {}
        for s in GRIP_STRATA:
            ps = pr[pr['stratum'] == s]
            npers = ps.groupby('y')['person_id'].nunique()
            inc = [c for c in range(k) if npers.get(c, 0) >= MIN_PERSONS]
            out['models'][name][s] = {
                'included': [cls[c] for c in inc], 'excluded': [cls[c] for c in range(k) if c not in inc],
                **_stratum_metrics(ps, k, cls, inc),
                'pred_share': {cls[c]: float(np.mean(ps['y_pred'] == c)) for c in range(k)},
                'true_share': {cls[c]: float(np.mean(ps['y'] == c)) for c in range(k)}}
    for a, b in STRATA_PAIRS:
        for s in GRIP_STRATA:
            inc = [SPEED_CLASSES.index(c) for c in out['models'][a][s]['included']]
            pa, pb = preds[a], preds[b]
            j = pa[(pa['stratum'] == s) & pa['y'].isin(inc)].merge(pb[['trial_key', 'y_pred']], on='trial_key',
                                                                   suffixes=('_a', '_b'))
            person, y = j['person_id'].to_numpy(), j['y'].to_numpy(dtype=int)
            ya, yb = j['y_pred_a'].to_numpy(dtype=int), j['y_pred_b'].to_numpy(dtype=int)
            row = {'a': a, 'b': b, 'stratum': s, 'n_trials': int(len(j)), 'n_persons': int(len(set(person))),
                   **SS.paired_balanced(person, y, ya, yb, len(SPEED_CLASSES)),
                   **SS.paired_swap(person, y, ya, yb, len(SPEED_CLASSES), 'macro_f1', key='d_f1'),
                   'd_acc': float(np.mean(ya == y) - np.mean(yb == y))}
            out['pairs'].append(row)
    for key in ('d_bacc', 'd_f1'):
        for r, q in zip(out['pairs'], M.holm([r[f'{key}_p'] for r in out['pairs']])):
            r[f'{key}_p_holm'] = q
    return out


def dominance_paired(T: pd.DataFrame) -> dict[str, Any]:
    """Dominant vs non-dominant movements within each person: per person the difference of the medians (dominant -
    non-dominant; a difference of logs, i.e. a log ratio, for LOG_FEATURES and duration_log); across persons a
    Wilcoxon signed-rank test, Holm over PAIRED_FEATURES. Sensitivity: the condition-matched difference (the mean
    difference inside each cell of MATCH_CELLS that holds both hands, weighted by the harmonic mean of the cell's
    counts)."""
    S = T[T['dominant'] >= 0].copy()
    for f in LOG_FEATURES:
        S[f] = np.log(S[f])
    out: dict[str, Any] = {'features': list(PAIRED_FEATURES), 'log_features': sorted(LOG_FEATURES | {'duration_log'}),
                           'match_cells': list(MATCH_CELLS), 'variants': {}}
    hs = S.groupby('person_id')['handedness'].first()
    for variant in ('unmatched', 'matched'):
        rows, per = [], []
        for f in PAIRED_FEATURES:
            if variant == 'unmatched':
                m = S.groupby(['person_id', 'dominant'])[f].median().unstack('dominant')
                diff = (m[1] - m[0]).dropna()
            else:
                cell = S.groupby(['person_id', *MATCH_CELLS, 'dominant'])[f].agg(['mean', 'size']).unstack('dominant')
                cell = cell.dropna()
                dd = cell[('mean', 1)] - cell[('mean', 0)]
                wt = 2.0 / (1.0 / cell[('size', 1)] + 1.0 / cell[('size', 0)])
                diff = (dd * wt).groupby(level='person_id').sum() / wt.groupby(level='person_id').sum()
            v = diff.to_numpy()
            stat, p = _wilcoxon(v)
            lo, hi = _boot_median_ci(v, C.SEED)
            rows.append({'feature': f, 'n_persons': int(v.size), 'median_diff': float(np.median(v)),
                         'median_diff_lo': lo, 'median_diff_hi': hi, 'n_dominant_higher': int((v > 0).sum()),
                         'wilcoxon_w': stat, 'p': p, 'ratio': f in LOG_FEATURES or f == 'duration_log'})
            for pid, val in diff.items():
                per.append({'person_id': str(pid), 'feature': f, 'diff': float(val), 'handedness': str(hs.get(pid, ''))})
        for r, q in zip(rows, M.holm([r['p'] for r in rows])):
            r['p_holm'] = q
        out['variants'][variant] = {'rows': rows, 'per_person': per}
    return out


def dominance_persons() -> dict[str, Any]:
    """Per-person AUC of the dominance model (both classes in every person), handedness, and the mean within-person
    AUC with its person-bootstrap CI; the pooled AUC and its CI from the saved result; the nested C per trial and per
    outer fold."""
    from stage2.data import load_cache
    r = L.load_result(DOMINANCE)
    pr = r.predictions
    wp = M.within_person_auc(pr['person_id'].astype(str).to_numpy(), pr['y'].to_numpy(), pr['p_right'].to_numpy())
    hs = load_cache().persons.set_index('person_id')['handedness_self'].astype(str)
    wp['handedness'] = wp['person_id'].map(hs)
    acc = r.per_person.set_index('person_id')['accuracy']
    wp['accuracy'] = wp['person_id'].map(acc)
    v = wp['auc'].to_numpy(dtype=float)
    lo, hi = M.boot_mean_ci(v)
    cs = pd.Series([json.loads(x).get('C') for x in pr['params']])
    folds = pd.DataFrame({'fold': pr['fold'].to_numpy(), 'C': cs.to_numpy()}).drop_duplicates()
    return {'pooled_auc': r.metrics['roc_auc'], 'pooled_auc_ci': list(r.ci['roc_auc']),
            'accuracy': r.metrics['accuracy'], 'accuracy_ci': list(r.ci['accuracy']),
            'balanced_accuracy': r.metrics['balanced_accuracy'], 'log_loss': r.metrics['log_loss'],
            'p_min': float(pr['p_right'].min()), 'p_max': float(pr['p_right'].max()),
            'within_auc_mean': float(np.nanmean(v)), 'within_auc_ci': [lo, hi],
            'n_persons_auc_above_half': int(np.sum(v > 0.5)), 'n_persons': int(np.isfinite(v).sum()),
            'chosen_C': cs.value_counts().to_dict(), 'n_trials': int(len(pr)),
            'n_folds': int(folds['fold'].nunique()),
            'n_folds_at_grid_min': int(folds.loc[folds['C'] == min(M.LR_C_GRID), 'fold'].nunique()),
            'grid_min': float(min(M.LR_C_GRID)),
            'leak_check': r.leak_check, 'per_person': wp.to_dict(orient='records')}


def dominance_null(chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The label-permutation null distribution (the chunks in order); the test against the observed model is
    :func:`dominance_null_test`, recomputed from the saved files at every compute step."""
    pooled = np.concatenate([c['pooled_auc'] for c in sorted(chunks, key=lambda c: c['chunk'])])
    within = np.concatenate([c['within_auc'] for c in sorted(chunks, key=lambda c: c['chunk'])])
    return {'n_perm': int(pooled.size), 'scheme': 'labels permuted within each person, same nested LOSO pipeline',
            'pooled_auc_null': pooled.tolist(), 'within_auc_null': within.tolist()}


def dominance_null_test(null: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """One-sided p = (1 + #{null >= observed}) / (1 + B) for the pooled and the mean within-person AUC of the CURRENT
    dominance result against the saved null; the null's mean and 95th percentile."""
    out: dict[str, Any] = {'n_perm': int(null['n_perm']), 'scheme': null['scheme']}
    for key, obs in (('pooled', observed['pooled_auc']), ('within', observed['within_auc_mean'])):
        nl = np.asarray(null[f'{key}_auc_null'], dtype=float)
        out[f'{key}_observed'] = float(obs)
        out[f'{key}_null_mean'] = float(np.mean(nl))
        out[f'{key}_null_sd'] = float(np.std(nl, ddof=1))
        out[f'{key}_null_q95'] = float(np.quantile(nl, 0.95))
        out[f'{key}_p'] = float((1 + np.sum(nl >= obs - 1e-12)) / (1 + nl.size))
    return out


def model_pairs() -> dict[str, Any]:
    """Paired comparisons of MODEL_PAIRS on the trials both models scored: balanced accuracy (the headline metric,
    ``sec_stats.paired_balanced``), macro-F1 (``sec_stats.paired_swap``), accuracy and multinomial log loss
    (``sec_models.paired_multiclass``); Holm within each metric family over every pair (PAIR_FAMILIES); a pair below
    solver precision is flagged ``identical``. Also the exploratory dominance pair (LightGBM vs the pre-specified LR)
    through ``cv.paired_compare`` (binary results)."""
    from stage2.lanes import sec_stats as SS
    rows = []
    for a, b in MODEL_PAIRS:
        ra, rb = load_multiclass(a), load_multiclass(b)
        pa, pb = pd.DataFrame(ra['trial_predictions']), pd.DataFrame(rb['trial_predictions'])
        j = pa.merge(pb, on='trial_key', suffixes=('_a', '_b'))
        cls = ra['classes']
        Pa = j[[f'p_{c}_a' for c in cls]].to_numpy()
        Pb = j[[f'p_{c}_b' for c in cls]].to_numpy()
        person, y = j['person_id_a'].to_numpy(), j['y_a'].to_numpy()
        r = M.paired_multiclass(person, y, Pa, Pb)
        r.update(SS.paired_balanced(person, y, Pa.argmax(axis=1), Pb.argmax(axis=1), len(cls)))
        r.update(SS.paired_swap(person, y, Pa.argmax(axis=1), Pb.argmax(axis=1), len(cls), 'macro_f1', key='d_f1'))
        subs = sorted({MULTICLASS[a]['subset'], MULTICLASS[b]['subset']} - {'all'}) or ['all']
        rows.append({'a': a, 'b': b, 'task': ra['task'],
                     'scope': ' and '.join(subset_label(x) for x in subs),
                     **{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()},
                     'identical': SS.identical(r['n_changed'], r['d_ll'])})
    for key in PAIR_FAMILIES:
        for r, q in zip(rows, M.holm([r[f'{key}_p'] for r in rows])):
            r[f'{key}_p_holm'] = q
    dom = cv.paired_compare(L.load_result(DOMINANCE_GBM), L.load_result(DOMINANCE))
    dom_row = {'a': DOMINANCE_GBM, 'b': DOMINANCE, 'status': 'exploratory (two model classes tried)',
               **{k: (list(v) if isinstance(v, tuple) else v) for k, v in dom.items() if k != 'per_person'}}
    return {'pairs': rows, 'families': {k: f'{v}: {k} over every pair, Holm' for k, v in PAIR_FAMILIES.items()},
            'dominance': dom_row}


def grip_errors(T: pd.DataFrame) -> dict[str, Any]:
    """The misclassified trials of each grip model: instructed vs predicted grip, the hand, the app's own grip
    detection at the start and the grip-mismatch flag (both sensor-derived app outputs, used after prediction only),
    whether the trial's up vector before the onset lies closer to the median up vector of the PREDICTED grip than to
    that of the instructed one, and the median up vector per instructed grip (the physics); the consistency check
    (``sec_stats.reference_agreement``); the errors per person; and the sign check: where the gravity sign of each
    trial came from, and the recall of the upright trials of the persons with errors (a wrong sign would turn their
    upright trials too)."""
    from stage2.lanes import sec_stats as SS
    out: dict[str, Any] = {'models': {}}
    Tk = T.set_index('trial_key')
    up = ['up_before_x', 'up_before_y', 'up_before_z']
    centre = T.groupby('grip')[up].median()
    for name in GRIP_MODELS:
        r = load_multiclass(name)
        cls = r['classes']
        pr = pd.DataFrame(r['trial_predictions'])
        err = pr[pr['y'] != pr['y_pred']]
        rows = []
        for _, e in err.iterrows():
            t = Tk.loc[e['trial_key']]
            rows.append({'trial_key': e['trial_key'], 'person_id': e['person_id'], 'hand': e['hand'],
                         'instructed': cls[int(e['y'])], 'predicted': cls[int(e['y_pred'])],
                         'app_detected': str(t['grip_app']), 'grip_mismatch': bool(t['grip_mismatch']),
                         'up_before_x': float(t['up_before_x']), 'up_before_y': float(t['up_before_y']),
                         'up_before_z': float(t['up_before_z']),
                         'up_matches_predicted': bool(
                             np.linalg.norm(t[up].to_numpy(dtype=float) - centre.loc[cls[int(e['y_pred'])]].to_numpy())
                             < np.linalg.norm(t[up].to_numpy(dtype=float) - centre.loc[cls[int(e['y'])]].to_numpy()))})
        E = pd.DataFrame(rows, columns=['trial_key', 'person_id', 'hand', 'instructed', 'predicted', 'app_detected',
                                        'grip_mismatch', 'up_before_x', 'up_before_y', 'up_before_z',
                                        'up_matches_predicted'])
        err_persons = sorted(E['person_id'].astype(str).unique().tolist())
        upr = pr[(pr['person_id'].isin(err_persons)) & (pr['y'] == cls.index('upright'))]
        out['models'][name] = {
            'n_errors': int(len(E)), 'n_trials': int(len(pr)), 'n_persons_with_errors': int(E['person_id'].nunique()),
            'n_flagged_mismatch': int(E['grip_mismatch'].sum()),
            'n_app_agrees_with_prediction': int((E['app_detected'] == E['predicted']).sum()),
            'n_sideways_swaps': int(((E['instructed'] != 'upright') & (E['predicted'] != 'upright')).sum()),
            'n_up_matches_predicted': int(E['up_matches_predicted'].sum()),
            'error_persons_upright': {'n': int(len(upr)), 'n_correct': int((upr['y_pred'] == upr['y']).sum())},
            'errors': E.to_dict(orient='records')}
        # the consistency check: the same predictions against the grip the app itself detected at the start (where
        # it could tell), and against the instruction on the trials neither the app nor the gravity rule flagged
        out['models'][name]['reference'] = SS.reference_agreement(
            pr, T.set_index('trial_key')['grip_app'], cls, instructed_ok=~T.set_index('trial_key')['grip_mismatch'])
        per = []
        for pid, g in E.groupby('person_id', sort=True):
            n_of = T[T['person_id'] == pid].groupby('grip').size()
            for (i, p), c in g.groupby(['instructed', 'predicted']).size().items():
                per.append({'person_id': str(pid), 'instructed': i, 'predicted': p, 'n': int(c),
                            'n_instructed': int(n_of.get(i, 0))})
        out['models'][name]['per_person'] = per
    med = T.groupby('grip')[up].median()
    out['up_before_median'] = {g: {a: float(med.loc[g, f'up_before_{a}']) for a in 'xyz'} for g in med.index}
    out['flagged_trials'] = int(T['grip_mismatch'].sum())
    out['flagged_persons'] = int(T.loc[T['grip_mismatch'], 'person_id'].nunique())
    out['app_unknown_trials'] = int((~T['grip_app'].isin(GRIP_CLASSES)).sum())
    out['gravity_sign_source'] = {str(k): int(v) for k, v in T['gravity_sign_source'].value_counts().items()}
    return out


def concordance(d: Any, nat: Any, rep: Any) -> dict[str, Any]:
    """Descriptive: self-reported handedness x usual phone hand x the hand of the first free-choice trial (and of the
    repeat-visit free trials)."""
    p = d.persons[['person_id', 'handedness_self', 'phone_hand']].copy()
    nh = nat.trials.groupby('person_id')['hand'].first()
    p['natural_use_hand'] = p['person_id'].map(nh).fillna('none')
    tab = (p.groupby(['handedness_self', 'phone_hand', 'natural_use_hand']).size().rename('n_persons')
           .reset_index())
    reps = rep.trials[['person_id', 'hand']].merge(p[['person_id', 'handedness_self', 'phone_hand']], on='person_id')
    agree_h = int((p['natural_use_hand'] == p['handedness_self']).sum())
    clear = p['phone_hand'].isin(['left', 'right'])
    agree_p = int((p.loc[clear, 'natural_use_hand'] == p.loc[clear, 'phone_hand']).sum())
    return {'table': tab.to_dict(orient='records'), 'n_persons': int(len(p)),
            'n_natural_use': int((p['natural_use_hand'] != 'none').sum()),
            'natural_use_eq_handedness': agree_h, 'natural_use_eq_phone_hand': agree_p,
            'n_phone_hand_clear': int(clear.sum()),
            'n_phone_hand_right': int((p['phone_hand'] == 'right').sum()),
            'n_natural_use_right': int((p['natural_use_hand'] == 'right').sum()),
            'n_left_handed': int((p['handedness_self'] == 'left').sum()),
            'left_handed_persons': sorted(p.loc[p['handedness_self'] == 'left', 'person_id'].astype(str).tolist()),
            'left_handed_natural_use': {str(r.person_id): str(r.natural_use_hand)
                                        for r in p[p['handedness_self'] == 'left'].itertuples()},
            'left_handed_phone_hand': {str(r.person_id): str(r.phone_hand)
                                       for r in p[p['handedness_self'] == 'left'].itertuples()},
            'repeats': reps.rename(columns={'hand': 'free_choice_hand'}).to_dict(orient='records')}


def balance_checks(T: pd.DataFrame, d: Any) -> dict[str, Any]:
    """Is dominance confounded with the plan? Dominant / non-dominant trial counts per person, the hand of each
    session's first block, and the instructed conditions per dominance."""
    t = d.trials
    first = t.sort_values(['person_id', 'session_order', 'block_index', 'trial_index']).groupby(
        ['person_id', 'session_order'])['hand'].first()
    from scipy.stats import binomtest
    S = T[T['dominant'] >= 0]
    fb = first.value_counts()
    n_left, n_first = int(fb.get('left', 0)), int(fb.sum())
    return {'dominant_per_person': S.groupby('person_id')['dominant'].agg(['sum', 'size']).rename(
                columns={'sum': 'n_dominant', 'size': 'n'}).reset_index().to_dict(orient='records'),
            'first_block_hand': fb.to_dict(),
            'first_block_binom_p': float(binomtest(n_left, n_first, 0.5).pvalue) if n_first else math.nan,
            'by_condition': {c: pd.crosstab(S[c], S['dominant']).rename(columns={0: 'non_dominant', 1: 'dominant'})
                             .reset_index().to_dict(orient='records') for c in ('speed', 'grip', 'variant')}}


def reseg_shares(T: pd.DataFrame, tabs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Where a label-blind window ends inside the oracle window: per instructed speed, the median (and IQR) per-trial
    share of the oracle duration and of the oracle angular path (total rotation) that the label-blind window covers
    (both windows start within milliseconds of each other); and per speed the share of trials the strict (slow)
    criterion ends before the return (all and standard-variant trials)."""
    E, R = tabs['even_common'], tabs['reseg']
    ok = R['common_ok'].to_numpy(dtype=bool)
    dur = np.exp(E['duration_log']) / np.exp(tabs['even']['duration_log'])
    path = E['ang_path'] / tabs['even']['ang_path']
    out: dict[str, Any] = {}
    std = (T['variant'] == 'standard').to_numpy()
    sok = R['strict_ok'].to_numpy(dtype=bool)
    for sp in SPEED_CLASSES:
        m = ok & (T['speed'] == sp).to_numpy()
        ms = (T['speed'] == sp).to_numpy()
        if not ms.any():
            continue
        out[sp] = {'n': int(m.sum()),
                   'duration_share_median': float(np.median(dur[m])) if m.any() else math.nan,
                   'path_share_median': float(np.median(path[m])) if m.any() else math.nan,
                   'path_share_q25': float(np.quantile(path[m], 0.25)) if m.any() else math.nan,
                   'path_share_q75': float(np.quantile(path[m], 0.75)) if m.any() else math.nan,
                   'strict_ok_share': float(np.mean(sok[ms])),
                   'strict_ok_share_standard': float(np.mean(sok[ms & std])) if (ms & std).any() else math.nan,
                   'n_standard': int((ms & std).sum())}
    return {'by_speed': out}


# ───────────────────────────── the step ─────────────────────────────


def compute(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.data import load_cache
    check_constants()
    d0, nat0 = load_cache(), load_cache(('natural_use',))
    if (force and selected('sec_reseg', only)) or not reseg_current(d0, nat0):
        print('sec: re-segmenting and re-standardising every trial on label-blind bounds (reads the snapshot)',
              flush=True)
        run_reseg(d0, nat0)
    d, nat, tabs, _ = _data()
    rep = load_cache(('natural_use_repeat',))
    T = trial_table(d, tabs)
    Tc = common_trial_table(T, tabs)
    L.export_table(T, 'sec_trial_features', tex=False)
    L.export_table(Tc[['trial_key', 'common_ok', *F.EVEN_FEATURES]], 'sec_trial_features_common', tex=False)
    L.export_table(F.catalogue(), 'sec_feature_catalogue_rows', tex=False)
    jobs: list[tuple[str, Any]] = []
    for name in MULTICLASS:
        if selected(name, only) and (force or not multiclass_current(name, tabs)):
            jobs.append(('mc', name))
    spec = {'model': DOM_MODEL, 'params': {'C_grid': list(M.LR_C_GRID)}, 'notes': DOM_NOTES, 'n_configs_tried': 1,
            **dominance_spec()}
    if L.should_run(DOMINANCE, force, only, spec=spec):
        jobs.append(('dom', DOMINANCE))
    spec_gbm = {'model': DOM_GBM_MODEL, 'notes': DOM_GBM_NOTES, 'n_configs_tried': 2, **dominance_spec()}
    if L.should_run(DOMINANCE_GBM, force, only, spec=spec_gbm):
        jobs.append(('dom', DOMINANCE_GBM))
    null_path = L.tab_dir / 'sec_dom_null.json'
    null_needed = selected('sec_dom_null', only) and (force or not file_current(null_path) or not _null_spec_ok())
    if null_needed:                 # a chunk saved by an interrupted run with the same spec and code is kept
        jobs += [('null', c) for c in range(N_PERM_CHUNKS) if force or not _null_part_ok(c)]
    jobs.sort(key=lambda j: ('gbm' not in str(j[1]), j[0] == 'null'))   # the long gradient-boosting jobs first
    print(f'sec: {len(jobs)} job(s) in {N_WORKERS} one-thread workers', flush=True)
    _run_jobs(jobs)
    if null_needed:
        chunks = [L.read_json(_null_part_name(c)) for c in range(N_PERM_CHUNKS)]
        L.write_json('sec_dom_null', {**dominance_null(chunks), 'spec': _null_spec()}, inputs=list(NULL_INPUTS))
    # derived files (cheap, always rewritten from the saved results)
    L.write_json('sec_mirror_check', mirror_check(d, tabs))
    L.write_json('sec_reseg_shares', reseg_shares(T, tabs), inputs=list(RESEG_FILES))
    L.write_json('sec_design', design(T))
    manip = speed_manipulation(T, Tc)
    L.write_json('sec_speed_manip', manip, inputs=list(RESEG_FILES))
    L.write_json('sec_speed_followers', speed_followers(manip), inputs=[HEADLINE['speed']])
    L.write_json('sec_speed_univariate', speed_univariate(T, Tc), inputs=list(RESEG_FILES))
    L.write_json('sec_speed_coef', speed_coefficients(), inputs=[HEADLINE['speed']])
    L.write_json('sec_speed_strata', speed_strata(T), inputs=list(STRATA_MODELS))
    persons = dominance_persons()
    L.write_json('sec_dom_persons', persons, inputs=[DOMINANCE])
    L.write_json('sec_dom_null_test', dominance_null_test(L.read_json('sec_dom_null'), persons),
                 inputs=[DOMINANCE, 'tables/sec/sec_dom_null.json'])
    L.write_json('sec_dom_paired', dominance_paired(T))
    L.write_json('sec_pairs', model_pairs(),
                 inputs=sorted({m for pair in MODEL_PAIRS for m in pair} | {DOMINANCE, DOMINANCE_GBM}))
    L.write_json('sec_grip_errors', grip_errors(T), inputs=list(GRIP_MODELS))
    L.write_json('sec_concordance', concordance(d, nat, rep))
    L.write_json('sec_balance', balance_checks(T, d))


def _null_spec() -> dict[str, Any]:
    return {'n_perm': N_PERM, 'seed': C.SEED, 'features': list(F.EVEN_FEATURES), 'grid': list(M.LR_C_GRID)}


NULL_INPUTS = ('stage2/lanes/sec_models.py', 'stage2/lanes/sec_features.py')


def _null_part_name(chunk: int) -> str:
    return f'sec_dom_null_part{chunk}'


def _null_part_spec(chunk: int) -> dict[str, Any]:
    return {**_null_spec(), 'chunk': chunk, 'n_chunks': N_PERM_CHUNKS}


def _null_part_ok(chunk: int) -> bool:
    path = L.tab_dir / f'{_null_part_name(chunk)}.json'
    if not file_current(path):
        return False
    return L.read_json(_null_part_name(chunk)).get('spec') == json.loads(json.dumps(_null_part_spec(chunk)))


def _null_spec_ok() -> bool:
    try:
        return L.read_json('sec_dom_null').get('spec') == json.loads(json.dumps(_null_spec()))
    except FileNotFoundError:
        return False
