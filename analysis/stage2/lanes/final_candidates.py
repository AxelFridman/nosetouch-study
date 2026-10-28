"""Lane final: the candidate models of the cross-lane synthesis, in ONE registry.

Every trial-level finalist of lanes m1, m2, m3 and p1 (read from ``results/finalists_<lane>.json``, so a lane that
changes its finalists changes this list), the reference floor, the body-frame engineered models of task 2b
(``final_models``) and one pre-specified cross-lane ensemble. A candidate is rebuilt exactly as its lane documents
it: ``importlib.import_module(module)``, ``getattr(mod, factory)(**params)`` on the input ``getattr(mod, build_x)(d)``.

Inputs are shared: every candidate reads one entry of a dict of per-trial inputs (:data:`INPUTS`), so the nested
cross-lane selection runs every candidate on the same folds through ONE ``run_experiment`` call.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from stage2 import config as C

#: per-trial inputs: key -> (module, function of a Stage2Data). Each is computed before the cross-validation (a
#: function of one trial alone, no label: README section 5.2).
INPUTS: dict[str, tuple[str, str]] = {
    'app': ('stage2.lanes.p1_panel', 'build_x'),                 # the time-normalised approach (N, 17, 64)
    'm1': ('stage2.lanes.m1_models', 'build_x_treat'),           # lane m1 catalogue (+ amplitude-normalised PHYS6)
    'm2': ('stage2.lanes.m2_finalists', 'build_x'),              # lane m2 default features (17 channels)
    'm3body': ('stage2.lanes.m3_body', 'x_approach_body'),       # approach + body-frame channels (N, 21, 64)
    'm3app': ('stage2.lanes.m3_common', 'x_approach'),           # approach as lane m3 reads it
    'm3stft': ('stage2.lanes.m3_tf', 'x_stft'),                  # signed STFT planes
    'm2body12': ('stage2.lanes.final_features', 'build_m2_body12'),   # m2 features on the CNN's 12 channels
    'm2all21': ('stage2.lanes.final_features', 'build_m2_all21'),     # m2 features on 17 + 4 body channels
}

#: lane id -> where its finalists come from (the registry of trial-level finalists; m4 is a stream pipeline whose
#: unit is a recording and x1 is descriptive, so neither enters the trial-level selection)
FINALIST_LANES: tuple[str, ...] = ('m1', 'm2', 'm3', 'p1')
#: build_x of a lane finalist -> the shared input key
_INPUT_OF: dict[tuple[str, str], str] = {
    ('stage2.lanes.m1_models', 'build_x'): 'm1', ('stage2.lanes.m1_models', 'build_x_treat'): 'm1',
    ('stage2.lanes.m2_finalists', 'build_x'): 'm2',
    ('stage2.lanes.m3_finalists', 'x_approach_body'): 'm3body', ('stage2.lanes.m3_finalists', 'x_approach'): 'm3app',
    ('stage2.lanes.m3_finalists', 'x_stft'): 'm3stft',
    ('stage2.lanes.p1_panel', 'build_x'): 'app',
}


@dataclass(frozen=True)
class Candidate:
    key: str                     # short id in this lane (figures, tables, the nested selection)
    source: str | None           # the saved result it reproduces (None: a model new in this lane)
    lane: str
    label: str                   # short label for figures and tables
    family: str
    representation: str
    treatment: str
    frames: tuple[str, ...]      # frames of its inputs: 'gravity', 'body' (heading-following), 'device'
    input: str                   # key of INPUTS
    module: str
    factory: str
    params: dict[str, Any] = field(default_factory=dict)
    n_configs_tried: int = 1     # outer-fold choices made in its own lane before it became a finalist
    nested: bool = True          # hyperparameters tuned inside the fold
    components: tuple[str, ...] = ()   # an ensemble: the candidates it averages (mean logit)
    notes: str = ''

    def make(self) -> Callable[[Any], Any]:
        """fit_predict(fold) of this candidate on its own input (``fold.X_*`` = INPUTS[self.input] rows)."""
        if self.components:
            raise TypeError('an ensemble is built by final_nested.ensemble_fit_predict')
        mod = importlib.import_module(self.module)
        return getattr(mod, self.factory)(**self.params)

    @property
    def deep(self) -> bool:
        return self.module == 'stage2.lanes.m3_finalists' and self.factory in ('deep', 'deep_body')


# ───────────────────────────── descriptions of the lane finalists ─────────────────────────────

#: lane result name -> (label, family, representation, data treatment, frames); a finalist that a lane adds later and
#: that is not described here still enters (with generic text), so the selection follows the finalists files
DESCRIPTIONS: dict[str, tuple[str, str, str, str, tuple[str, ...]]] = {
    'example_yaw_logreg': ('reference: yaw logistic', 'logistic regression (8 features)',
                           'cumulative yaw about gravity at 8 instants', 'mirror augmentation, C = 1', ('gravity',)),
    'm1_rule_fhy_sign_cal': ('m1 rule: first-half yaw sign', 'rule (2 parameters)', 'first-half yaw about gravity',
                             'direction and hit rate learned in-fold', ('gravity',)),
    'm1_lr_phys6_nested': ('m1 PHYS6 logistic', 'antisymmetric logistic (6 features)',
                           'yaw, roll and lateral velocity at 50 / 100 % of the approach',
                           'treatment (scaling, winsorising, asinh) and C chosen in-fold', ('gravity', 'body')),
    'm1_gam_phys6': ('m1 PHYS6 GAM', 'additive spline logistic', 'PHYS6 features',
                     'knots and C chosen in-fold, mirror-symmetrised', ('gravity', 'body')),
    'm2_lr_full': ('m2 logistic (691 features)', 'L2 logistic regression', '691 features of 7 families, 17 channels',
                   'z-score, mirror augmentation, C nested', ('gravity', 'device')),
    'm2_lgbm': ('m2 LightGBM', 'gradient boosting', '691 features, 17 channels',
                'mirror augmentation, Optuna on inner folds', ('gravity', 'device')),
    'm2_ens_stack': ('m2 stack (LR + SVM + LGBM)', 'stacked ensemble', '691 features, 17 channels',
                     'components tuned in-fold, stacker on inner logits', ('gravity', 'device')),
    'm3_cal_deep_platt': ('m3 1D CNN, body frame', '1D CNN ensemble (4 members)',
                          '12 channels: gravity frame + body frame', 'z-score, mirror / warp / scale / jitter '
                          'augmentation, Platt in-fold', ('gravity', 'body')),
    'm3_cal_mr_platt': ('m3 MiniRocket', 'MiniRocket + logistic', '17 channels, 4000 random-kernel features',
                        'per-trial amplitude, C nested, Platt in-fold', ('gravity',)),
    'm3_cal_tf_platt': ('m3 2D CNN on STFT', '2D CNN ensemble', 'signed STFT planes of the gravity frame',
                        'z-score, augmentation, Platt in-fold', ('gravity',)),
    'p1_calib_gbm_auto': ('p1 LightGBM + calibrator', 'gradient boosting (fixed parameters)',
                          '14 summary features x 17 channels', 'mirror augmentation, calibrator chosen in-fold',
                          ('gravity', 'device')),
    'p1_lr_base': ('p1 logistic (chiral bins)', 'logistic regression (16 features)',
                   '8 time bins of yaw rate and cumulative yaw', 'standardised, mirror augmentation, C = 1',
                   ('gravity',)),
    'p1_mr_base': ('p1 MiniRocket + ridge', 'MiniRocket + ridge', 'gravity frame, about 2000 features',
                   'ridge alpha and Platt nested', ('gravity',)),
}

#: the reference floor of the registry (core unit 'reference')
REFERENCE = 'example_yaw_logreg'

#: lane finalists whose configuration is NOT tuned inside the fold (their lanes' reports: a fixed or outer-fold-chosen
#: configuration); reported in the master and decision tables only. The candidate's own ``nested`` flag (from the
#: finalists file) stays as it is, because it is part of the inner-prediction cache keys.
NOT_TUNED_IN_FOLD: frozenset[str] = frozenset({'example_yaw_logreg', 'm1_rule_fhy_sign_cal', 'm3_cal_deep_platt',
                                                'm3_cal_tf_platt', 'p1_lr_base'})


def _short_key(name: str) -> str:
    return {'example_yaw_logreg': 'ref', 'm1_rule_fhy_sign_cal': 'm1_rule', 'm1_lr_phys6_nested': 'm1_phys6',
            'm1_gam_phys6': 'm1_gam', 'm2_lr_full': 'm2_lr', 'm2_lgbm': 'm2_lgbm', 'm2_ens_stack': 'm2_stack',
            'm3_cal_deep_platt': 'm3_deep', 'm3_cal_mr_platt': 'm3_mr', 'm3_cal_tf_platt': 'm3_tf',
            'p1_calib_gbm_auto': 'p1_gbm', 'p1_lr_base': 'p1_lr', 'p1_mr_base': 'p1_mr'}.get(name, name)


def _from_finalist(lane: str, f: dict[str, Any]) -> Candidate:
    name = f['name']
    module, bx = f['module'], f['build_x']
    if (module, bx) not in _INPUT_OF:
        raise KeyError(f'finalist {name}: no shared input for {module}.{bx} (add it to final_candidates.INPUTS)')
    label, fam, rep, treat, frames = DESCRIPTIONS.get(name, (name, f.get('factory', ''), bx, '', ('gravity',)))
    params = dict(f.get('params') or {})
    if module == 'stage2.lanes.m3_finalists':
        params['cache_dir'] = str(FIT_CACHE_M3)       # never write into lane m3's own fit cache
    nested = bool(f.get('nested_tuning', True))
    return Candidate(key=_short_key(name), source=name, lane=lane, label=label, family=fam, representation=rep,
                     treatment=treat, frames=frames, input=_INPUT_OF[(module, bx)], module=module,
                     factory=f['factory'], params=params, n_configs_tried=int(f.get('n_configs_tried', 1)),
                     nested=nested, notes=str(f.get('notes', ''))[:400])


#: this lane's own data folder (inner predictions, fit caches; git-ignored like every analysis/data folder)
DATA_DIR = C.DATA_DIR / 'stage2_final'
FIT_CACHE_M3 = DATA_DIR / 'm3_fit_cache'


def reference_candidate() -> Candidate:
    label, fam, rep, treat, frames = DESCRIPTIONS[REFERENCE]
    return Candidate(key='ref', source=REFERENCE, lane='reference', label=label, family=fam, representation=rep,
                     treatment=treat, frames=frames, input='app', module='stage2.lanes.final_models',
                     factory='make_reference', params={}, n_configs_tried=1, nested=False,
                     notes='the README worked example (pre-specified sanity floor)')


#: task 2b: the engineered models given the body-frame channels (new in this lane; pre-specified here)
BODY_CANDIDATES: tuple[Candidate, ...] = (
    Candidate(key='lr_body_tc', source=None, lane='final', label='logistic, body-frame time courses',
              family='antisymmetric logistic (32 features)',
              representation='8 time bins of yaw rate, cumulative yaw, roll rate, lateral acceleration',
              treatment='RMS scaling (no centring), no intercept, C nested', frames=('gravity', 'body'),
              input='m3body', module='stage2.lanes.final_models', factory='make_lr_body_tc', params={},
              notes='lane m1 small logistic given the body-frame time courses (task 2b)'),
    Candidate(key='m2_body12_lr', source=None, lane='final', label='m2 logistic on the CNN input (12 ch.)',
              family='L2 logistic regression', representation='m2 feature families on the 8 gravity + 4 body channels',
              treatment='z-score, mirror augmentation, C nested', frames=('gravity', 'body'), input='m2body12',
              module='stage2.lanes.final_models', factory='make_m2_lr', params={'channels': 'body12'},
              notes='lane m2 feature pipeline on exactly the 1D CNN input (task 2b, same input)'),
    Candidate(key='m2_all21_lr', source=None, lane='final', label='m2 logistic + body frame (21 ch.)',
              family='L2 logistic regression', representation='m2 feature families on 17 + 4 body channels',
              treatment='z-score, mirror augmentation, C nested', frames=('gravity', 'body', 'device'),
              input='m2all21', module='stage2.lanes.final_models', factory='make_m2_lr', params={'channels': 'all21'},
              notes='lane m2 feature pipeline given the body-frame channels too (task 2b)'),
)

#: the cross-lane ensemble: the mean logit of one pre-specified finalist per lane (diverse families: deep sequence
#: model, rich linear features, physics logistic); fixed before any result of this lane was seen
ENSEMBLE_COMPONENTS: tuple[str, ...] = ('m3_deep', 'm2_lr', 'm1_phys6')


def ensemble_candidate() -> Candidate:
    return Candidate(key='ens_mean3', source=None, lane='final', label='ensemble: mean logit of 3 lanes',
                     family='mean logit of 3 finalists', representation='m3 CNN + m2 logistic + m1 PHYS6',
                     treatment='each component as in its lane', frames=('gravity', 'body', 'device'), input='*',
                     module='stage2.lanes.final_nested', factory='ensemble_fit_predict', params={},
                     components=ENSEMBLE_COMPONENTS, notes='pre-specified cross-lane ensemble (task 2)')


def lane_finalists(lane: str) -> list[dict[str, Any]]:
    p = C.RES_DIR / f'finalists_{lane}.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else []


def candidates(*, include_body: bool = True, include_ensemble: bool = True) -> dict[str, Candidate]:
    """Every candidate, in a fixed order (reference, lanes m1, m2, m3, p1, the 2b models, the ensemble). The order is
    the tie-break of the nested selection (earlier = simpler families first within a lane)."""
    out: dict[str, Candidate] = {'ref': reference_candidate()}
    for lane in FINALIST_LANES:
        for f in lane_finalists(lane):
            c = _from_finalist(lane, f)
            out[c.key] = c
    if include_body:
        for c in BODY_CANDIDATES:
            out[c.key] = c
    if include_ensemble and all(k in out for k in ENSEMBLE_COMPONENTS):
        e = ensemble_candidate()
        out[e.key] = e
    return out


def needed_inputs(cands: dict[str, Candidate]) -> list[str]:
    keys: list[str] = []
    for c in cands.values():
        for k in ([c.input] if not c.components else [cands[x].input for x in c.components]):
            if k not in keys:
                keys.append(k)
    return keys


def build_inputs(d: Any, keys: list[str] | None = None) -> dict[str, Any]:
    """{input key: per-trial input of the Stage2Data d} (rows aligned with d.trials; no label is touched)."""
    out: dict[str, Any] = {}
    for k in (keys or list(INPUTS)):
        mod, fn = INPUTS[k]
        out[k] = getattr(importlib.import_module(mod), fn)(d)
    return out


def describe(c: Candidate) -> dict[str, Any]:
    """A JSON-able description (tables, the finalists file)."""
    return {'key': c.key, 'source': c.source, 'lane': c.lane, 'label': c.label, 'family': c.family,
            'representation': c.representation, 'treatment': c.treatment, 'frames': list(c.frames),
            'input': c.input, 'module': c.module, 'factory': c.factory,
            'params': {k: v for k, v in c.params.items() if k != 'cache_dir'},
            'n_configs_tried': c.n_configs_tried, 'nested': c.nested, 'components': list(c.components)}


def n_params_hint(c: Candidate) -> str:
    """How the size of the model is counted (the decision table measures it on the fitted export when possible)."""
    return c.family


__all__ = ['Candidate', 'INPUTS', 'candidates', 'build_inputs', 'needed_inputs', 'describe', 'DATA_DIR', 'REFERENCE',
           'ENSEMBLE_COMPONENTS', 'FIT_CACHE_M3', 'np']
