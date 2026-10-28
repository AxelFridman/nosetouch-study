"""Lane m3 experiment registry: every experiment is (model family, per-trial input, parameters) and is evaluated
only through ``cv.run_experiment`` (leave-one-person-out). The families are pre-specified; how many were compared on
the outer folds before choosing a finalist is recorded (``n_configs_tried``).

Groups:
* ``arch``      - deep architectures on the default input (approach, gravity frame, z-score, full augmentation);
* ``treat``     - data-treatment comparisons, each paired against its family default (window, channel set - including
                  the grip-invariant body frame of ``m3_body`` -, normalisation, augmentation, antisymmetry, head);
* ``treat_body``- the key treatment comparisons repeated on the body-frame input (window, normalisation, symmetry),
                  paired against ``m3_cnn_body``: do the decisions made on the gravity frame hold for the finalist's
                  input? Sensitivity checks only: never candidates of the deep role (no further outer-fold choice);
* ``grip``      - grip-transfer checks: an input trained on sensor-detected upright grips only, tested on every grip;
* ``mr``        - MiniRocket (C nested), with its own treatment comparisons;
* ``tf``        - time-frequency inputs: L2 logistic on pooled planes (C nested) and 2D CNNs, signed vs magnitude;
* ``calib``     - calibration remedies of each family's best model, fitted on cross-fitted training predictions;
* ``nested``    - the deep architecture (``m3_deep_nested``), the whole deep configuration (``m3_deep_nested_all``) or
                  the grip-invariant deep configuration (``m3_deep_nested_inv``, the shipping policy) chosen INSIDE
                  every outer fold by the inner cross-fitted log loss.

``post_hoc`` marks a configuration designed after outer-fold results had been seen (the body frame was added after
the grip-transfer check of ``m3_cnn_all``): its accuracy and any nested choice that includes it are optimistic by an
amount no nested estimate can remove.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stage2.lanes import m3_common as M

# ───────────────────────────── specification ─────────────────────────────


@dataclass(frozen=True)
class Spec:
    name: str
    family: str                       # 'deep' | 'mr' | 'tflog' | 'nested'
    x: str                            # 'approach' | 'fixed' | 'stft' | 'cwt'
    params: dict[str, Any]
    label: str
    group: str
    baseline: str | None = None       # the experiment it is paired against (data-treatment comparisons)
    treatment: str = ''               # which choice it tests ('window', 'channels', ...)
    option: str = ''                  # the option of that choice
    notes: str = ''
    physics_demo: bool = False        # deliberately crippled input (magnitude-only): never a finalist
    extra: dict[str, Any] = field(default_factory=dict)
    post_hoc: bool = False            # designed after outer-fold results had been seen (see the module docstring)


FIXED_AUG = ('mirror', 'warp', 'shift', 'scale', 'jitter')
D = 'm3_cnn'                          # the deep default (pre-specified)
R = 'm3_mr'                           # the MiniRocket default
T = 'm3_tflog_stft'                   # the time-frequency default


def _deep(name, label, group, x='approach', baseline=None, treatment='', option='', notes='', **params) -> Spec:
    return Spec(name, 'deep', x, params, label, group, baseline, treatment, option, notes)


def _mr(name, label, group, x='approach', baseline=None, treatment='', option='', notes='', **params) -> Spec:
    return Spec(name, 'mr', x, params, label, group, baseline, treatment, option, notes)


def _tf(name, label, x, baseline=None, treatment='', option='', notes='', physics_demo=False, **params) -> Spec:
    return Spec(name, 'tflog', x, params, label, 'tf', baseline, treatment, option, notes, physics_demo)


SPECS: list[Spec] = [
    # deep architectures (approach, gravity channels, z-score, mirror + warp + scale + jitter, 4-member CV ensemble)
    _deep('m3_cnn', '1D CNN', 'arch', notes='deep default (pre-specified)'),
    _deep('m3_cnn_anti', '1D CNN, antisymmetric', 'arch', baseline=D, treatment='symmetry', option='antisymmetric g(x) - g(mirror x)',
          anti=True),
    _deep('m3_resnet', 'ResNet1D', 'arch', arch='resnet'),
    _deep('m3_inception', 'InceptionTime-lite', 'arch', arch='inception'),
    _deep('m3_tcn', 'TCN', 'arch', arch='tcn'),
    _deep('m3_gru', 'GRU', 'arch', arch='gru'),
    _deep('m3_transformer', 'Transformer (tiny)', 'arch', arch='transformer'),
    # data treatment of the deep default
    _deep('m3_cnn_fixed', '1D CNN, fixed 3-s window', 'treat', x='fixed', baseline=D, treatment='window',
          option='fixed 3 s (absolute time, mask channel)', window='fixed', aug=FIXED_AUG),
    _deep('m3_cnn_all', '1D CNN, all 17 channels', 'treat', baseline=D, treatment='channels', option='all 17', channels='all'),
    _deep('m3_cnn_chiral', '1D CNN, chiral channels', 'treat', baseline=D, treatment='channels', option='chiral only (2)',
          channels='chiral'),
    Spec('m3_cnn_body', 'body', 'approach_body', dict(channels='body'), '1D CNN, gravity + body frame (12)', 'treat',
         baseline=D, treatment='channels', option='gravity + body frame (12, grip-invariant)',
         notes='the device information of m3_cnn_all in the heading-following body frame (m3_body); post hoc: '
               'designed after the grip-transfer result of m3_cnn_all', post_hoc=True),
    # the key treatments repeated on the body-frame input (sensitivity checks, not candidates of the deep role)
    Spec('m3_cnn_body_fixed', 'body', 'fixed_body', dict(channels='body', window='fixed', aug=FIXED_AUG),
         '1D CNN, body frame, fixed 3-s window', 'treat_body', baseline='m3_cnn_body', treatment='window (body frame)',
         option='fixed 3 s (absolute time, mask channel)', post_hoc=True,
         notes='does the window decision hold on the body-frame input? (audit re-check, not a role candidate)'),
    Spec('m3_cnn_body_trial', 'body', 'approach_body', dict(channels='body', norm='trial'),
         '1D CNN, body frame, per-trial amplitude', 'treat_body', baseline='m3_cnn_body',
         treatment='normalisation (body frame)', option='per-trial RMS', post_hoc=True,
         notes='does the normalisation decision hold on the body-frame input? (audit re-check, not a role candidate)'),
    Spec('m3_cnn_body_anti', 'body', 'approach_body', dict(channels='body', anti=True),
         '1D CNN, body frame, antisymmetric', 'treat_body', baseline='m3_cnn_body', treatment='symmetry (body frame)',
         option='antisymmetric g(x) - g(mirror x)', post_hoc=True,
         notes='does the symmetry decision hold on the body-frame input? (audit re-check, not a role candidate)'),
    _deep('m3_cnn_norm_none', '1D CNN, no scaling', 'treat', baseline=D, treatment='normalisation', option='none (raw units)',
          norm='none'),
    _deep('m3_cnn_norm_robust', '1D CNN, robust scaling', 'treat', baseline=D, treatment='normalisation',
          option='median/IQR, winsorised at 5', norm='robust'),
    _deep('m3_cnn_norm_trial', '1D CNN, per-trial amplitude', 'treat', baseline=D, treatment='normalisation',
          option='per-trial RMS', norm='trial'),
    _deep('m3_cnn_aug_none', '1D CNN, no augmentation', 'treat', baseline=D, treatment='augmentation', option='none', aug=()),
    _deep('m3_cnn_aug_mirror', '1D CNN, mirror only', 'treat', baseline=D, treatment='augmentation', option='mirror only',
          aug=('mirror',)),
    _deep('m3_cnn_gap', '1D CNN, global-average head', 'treat', baseline=D, treatment='head',
          option='global average (no time position)', head='gap'),
    # MiniRocket (C nested on fold.inner_splits())
    _mr('m3_mr', 'MiniRocket + logistic', 'mr', notes='MiniRocket default (pre-specified)'),
    _mr('m3_mr_anti', 'MiniRocket, antisymmetric', 'mr', baseline=R, treatment='symmetry', option='phi(x) - phi(mirror x)',
        mode='anti'),
    _mr('m3_mr_all', 'MiniRocket, all 17 channels', 'mr', baseline=R, treatment='channels', option='all 17', channels='all'),
    _mr('m3_mr_chiral', 'MiniRocket, chiral channels', 'mr', baseline=R, treatment='channels', option='chiral only (2)',
        channels='chiral'),
    _mr('m3_mr_fixed', 'MiniRocket, fixed 3-s window', 'mr', x='fixed', baseline=R, treatment='window',
        option='fixed 3 s (masked samples zeroed)', window='fixed'),
    _mr('m3_mr_norm_none', 'MiniRocket, no scaling', 'mr', baseline=R, treatment='normalisation', option='none (raw units)',
        norm='none'),
    _mr('m3_mr_norm_trial', 'MiniRocket, per-trial amplitude', 'mr', baseline=R, treatment='normalisation',
        option='per-trial RMS', norm='trial'),
    # time-frequency: logistic on pooled planes (C nested) and 2D CNNs
    _tf('m3_tflog_stft', 'STFT signed + logistic', 'stft', notes='time-frequency default (pre-specified)'),
    _tf('m3_tflog_stft_mag', 'STFT magnitude + logistic', 'stft', baseline=T, treatment='tf sign',
        option='magnitude only, mirror-augmented', physics_demo=True, mode='mag'),
    _tf('m3_tflog_stft_mag_noaug', 'STFT magnitude + logistic, no mirror', 'stft', baseline=T, treatment='tf sign',
        option='magnitude only, no mirror augmentation', physics_demo=True, mode='mag', mirror_aug=False),
    _tf('m3_tflog_cwt', 'CWT signed + logistic', 'cwt', baseline=T, treatment='tf transform', option='CWT (Morlet)', tf='cwt'),
    _tf('m3_tflog_cwt_mag', 'CWT magnitude + logistic', 'cwt', baseline='m3_tflog_cwt', treatment='tf sign',
        option='magnitude only, mirror-augmented', physics_demo=True, tf='cwt', mode='mag'),
    _tf('m3_tflog_cwt_mag_noaug', 'CWT magnitude + logistic, no mirror', 'cwt', baseline='m3_tflog_cwt',
        treatment='tf sign', option='magnitude only, no mirror augmentation', physics_demo=True, tf='cwt', mode='mag',
        mirror_aug=False),
    Spec('m3_cnn_all_upright', 'gripsub', 'approach_grip', dict(channels='all'),
         '1D CNN, all 17 channels, trained on upright grips only', 'grip', baseline='m3_cnn_all',
         treatment='grip transfer', option='trained on sensor-detected upright trials only, tested on every grip',
         notes='robustness of the device-frame channels to grips unseen in training'),
    Spec('m3_cnn_body_upright', 'gripsub_body', 'approach_body_grip', dict(channels='body'),
         '1D CNN, gravity + body frame, trained on upright grips only', 'grip', baseline='m3_cnn_body',
         treatment='grip transfer', option='trained on sensor-detected upright trials only, tested on every grip',
         notes='robustness of the body-frame channels to grips unseen in training', post_hoc=True),
    Spec('m3_cnn2d_stft', 'deep', 'stft', dict(tf='stft', aug=('mirror', 'scale', 'jitter')), '2D CNN on STFT (signed)',
         'tf', baseline=T, treatment='tf model', option='2D CNN on the signed planes'),
    Spec('m3_cnn2d_cwt', 'deep', 'cwt', dict(tf='cwt', aug=('mirror', 'scale', 'jitter')), '2D CNN on CWT (signed)',
         'tf', baseline='m3_tflog_cwt', treatment='tf model', option='2D CNN on the signed planes'),
    Spec('m3_cnn2d_stft_mag', 'deep', 'stft', dict(tf='stft', tf_mode='mag', aug=('scale', 'jitter')),
         '2D CNN on STFT magnitude, no mirror', 'tf', baseline='m3_cnn2d_stft', treatment='tf sign',
         option='magnitude only, no mirror augmentation', physics_demo=True),
]

BY_NAME: dict[str, Spec] = {s.name: s for s in SPECS}
#: Specified but not run by default (CPU budget of the lane: about 3 h on 3 threads). Their questions are answered
#: by cheaper runs: the Transformer is optional in the brief; the magnitude-only 2D CNN repeats the logistic
#: physics demonstration; the CWT 2D CNN (about 2.5 CPU-min per fold) is covered by the STFT 2D CNN and the CWT
#: logistic; the MiniRocket window / channel / no-scaling variants repeat the 1D CNN comparisons.
#: ``M3_EXTENDED=1`` adds them to the base experiments (the summary, tables and figures pick them up).
OPTIONAL: tuple[str, ...] = ('m3_transformer', 'm3_cnn2d_stft_mag', 'm3_cnn2d_cwt', 'm3_mr_fixed', 'm3_mr_all',
                             'm3_mr_norm_none')
EXTENDED = os.environ.get('M3_EXTENDED', '') not in ('', '0')
BASE: tuple[str, ...] = tuple(s.name for s in SPECS if EXTENDED or s.name not in OPTIONAL)

#: run order of the base experiments (the slow ones are split over streams by scripts/run_m3.py)
DEEP_ARCH_CANDIDATES = tuple(n for n in ('m3_cnn', 'm3_cnn_anti', 'm3_resnet', 'm3_inception', 'm3_tcn', 'm3_gru',
                                          'm3_transformer') if n in BASE)

# ───────────────────────────── families and inputs ─────────────────────────────


GRIP_CODES = {'upright': 0, 'top_left': 1, 'top_right': 2}       # anything else: 3


def sensor_grip(d: Any) -> np.ndarray:
    """The grip of every trial from its own sensors only: the mean unit up vector over the rest before the onset
    (device frame) -> motionlab.conventions.rotation_from_gravity (screen toward the person). Per trial; no label and
    no instructed condition is used."""
    from motionlab.conventions import rotation_from_gravity
    U = d.trials[['up_pre_x', 'up_pre_y', 'up_pre_z']].to_numpy(float)
    out = np.full(len(U), 3, dtype=np.int8)
    for i, u in enumerate(U):
        if np.all(np.isfinite(u)):
            out[i] = GRIP_CODES.get(str(rotation_from_gravity(u, 1, 'toward')), 3)
    return out


def x_approach_grip(d: Any) -> dict[str, np.ndarray]:
    """The approach plus the sensor-detected grip code (used only to choose the TRAINING rows of a grip-transfer
    check; never a model input)."""
    return {'x': np.ascontiguousarray(d.approach, dtype=np.float32), 'grip': sensor_grip(d)}


def build_x(kind: str, d: Any) -> dict[str, np.ndarray]:
    from stage2.lanes import m3_body, m3_tf
    if kind == NESTED_ALL_INPUT:
        from stage2.lanes import m3_nested
        return m3_nested.x_multi(d, nested_all_kinds())
    return {'approach': M.x_approach, 'fixed': M.x_fixed, 'stft': m3_tf.x_stft, 'cwt': m3_tf.x_cwt,
            'approach_grip': x_approach_grip, 'approach_body': m3_body.x_approach_body,
            'approach_body_grip': m3_body.x_approach_body_grip, 'fixed_body': m3_body.x_fixed_body}[kind](d)


BUILD_X_FUNCS = {'approach': ('stage2.lanes.m3_common', 'x_approach'), 'fixed': ('stage2.lanes.m3_common', 'x_fixed'),
                 'stft': ('stage2.lanes.m3_tf', 'x_stft'), 'cwt': ('stage2.lanes.m3_tf', 'x_cwt'),
                 'approach_body': ('stage2.lanes.m3_body', 'x_approach_body'),
                 'fixed_body': ('stage2.lanes.m3_body', 'x_fixed_body')}


def grip_dependent(spec: Spec) -> bool:
    """True when the input holds device-frame channels (their values change with the grip)."""
    from stage2.lanes import m3_body
    from stage2.standardize import DEVICE_CHANNELS
    if spec.family in ('nested', 'nested_all'):
        return any(grip_dependent(BY_NAME[c]) for c in spec.params.get('candidates', {}) if c in BY_NAME)
    ch = spec.params.get('channels', 'grav')
    return bool(set(m3_body.channel_names(ch)) & set(DEVICE_CHANNELS))
FACTORIES = {'deep': ('stage2.lanes.m3_deep', 'factory'), 'mr': ('stage2.lanes.m3_minirocket', 'factory'),
             'tflog': ('stage2.lanes.m3_tf', 'factory'), 'nested': ('stage2.lanes.m3_experiments', 'nested_factory'),
             'body': ('stage2.lanes.m3_body', 'factory'), 'nested_all': ('stage2.lanes.m3_nested', 'factory')}


def make_family(spec: Spec) -> Any:
    if spec.family == 'deep':
        from stage2.lanes.m3_deep import DeepFamily
        return DeepFamily(**spec.params)
    if spec.family == 'mr':
        from stage2.lanes.m3_minirocket import MiniRocketFamily
        return MiniRocketFamily(**spec.params)
    if spec.family == 'tflog':
        from stage2.lanes.m3_tf import TFLogisticFamily
        return TFLogisticFamily(**spec.params)
    if spec.family == 'nested':
        return NestedSelect(spec.params['candidates'])
    if spec.family == 'gripsub':
        from stage2.lanes.m3_deep import DeepFamily
        return TrainGripSubset(DeepFamily(**spec.params))
    if spec.family == 'body':
        from stage2.lanes.m3_body import BodyDeepFamily
        return BodyDeepFamily(**spec.params)
    if spec.family == 'gripsub_body':
        from stage2.lanes.m3_body import BodyDeepFamily
        return TrainGripSubset(BodyDeepFamily(**spec.params))
    if spec.family == 'nested_all':
        from stage2.lanes.m3_nested import NestedSelectMulti
        return NestedSelectMulti(spec.params['candidates'])
    raise ValueError(spec.family)


def fit_predict_of(spec: Spec, calib: str = 'none') -> Any:
    return M.make_fit_predict(make_family(spec), calib=calib)


# ───────────────────────────── grip transfer ─────────────────────────────


class TrainGripSubset:
    """Trains the base family only on the training rows whose sensor-detected grip is ``keep`` (upright), with its
    own GroupKFold inner splits over those rows' persons (cv.inner_splits, as fold.inner_splits does), and predicts
    every test row. The other training rows get the ensemble's prediction as their 'oof' logit (they were not trained
    on), so the wrapper fits the make_fit_predict contract; it is a robustness check, never a finalist."""

    def __init__(self, base: Any, keep: int = 0):
        self.base, self.keep = base, int(keep)
        self.n_inner = base.n_inner

    def config(self) -> dict[str, Any]:
        return {'base': self.base.config(), 'keep_grip': self.keep}

    def code_hash(self) -> str:
        return M.module_code_hash(M, sys.modules[__name__]) + self.base.code_hash()

    def cache_name(self) -> str:
        return 'gripsub_' + M.digest(self.config())[:12]

    def fit(self, X, y, groups, splits, seed) -> dict[str, Any]:
        from stage2 import cv
        keep = np.asarray(X['grip']) == self.keep
        Xs = {'x': X['x'][keep]}
        sub = cv.inner_splits(groups[keep], self.n_inner)
        st = M.cached_fit(self.base, Xs, y[keep], groups[keep], sub, seed)
        oof = self.base.logits(st, {'x': X['x']})
        oof[keep] = st['oof_z']
        return {'base': st, 'oof_z': oof, 'info': {'n_train_kept': int(keep.sum()), 'n_train_rows': int(keep.size),
                                                   'epochs': st['info'].get('epochs', [])}}

    def logits(self, state, X) -> np.ndarray:
        return self.base.logits(state['base'], {'x': X['x']})


# ───────────────────────────── nested architecture choice ─────────────────────────────


class NestedSelect:
    """Chooses, inside every outer fold, the candidate deep configuration with the lowest cross-fitted log loss on
    the TRAINING persons (each candidate's member-held-out logits), then predicts with it. The candidates are fitted
    with the same inner splits, so their cached fits are reused."""

    def __init__(self, candidates: dict[str, dict[str, Any]]):
        from stage2.lanes.m3_deep import DeepFamily
        self.candidates = {k: dict(v) for k, v in candidates.items()}
        self.fams = {k: DeepFamily(**v) for k, v in self.candidates.items()}
        ns = {f.n_inner for f in self.fams.values()}
        if len(ns) != 1:
            raise ValueError('nested candidates need the same number of inner splits')
        self.n_inner = ns.pop()

    def config(self) -> dict[str, Any]:
        return {'candidates': self.candidates}

    def code_hash(self) -> str:
        return M.module_code_hash(M, sys.modules[__name__])

    def cache_name(self) -> str:
        return 'nested_' + M.digest(self.config())[:12]

    def fit(self, X, y, groups, splits, seed) -> dict[str, Any]:
        lls, states = {}, {}
        for k, fam in self.fams.items():
            st = M.cached_fit(fam, X, y, groups, splits, seed)
            states[k] = st
            lls[k] = float(np.mean(np.logaddexp(0, -(2 * y - 1) * st['oof_z'])))
        best = min(lls, key=lambda k: (lls[k], k))
        cpu = float(sum(s.get('fit_cpu_s', 0.0) for s in states.values()))
        return {'choice': best, 'state': states[best], 'oof_z': states[best]['oof_z'], 'fit_cpu_s_all': cpu,
                'info': {'choice': best, 'inner_log_loss': {k: round(v, 5) for k, v in lls.items()},
                         'fit_cpu_s_all': round(cpu, 2)}}

    def logits(self, state, X) -> np.ndarray:
        return self.fams[state['choice']].logits(state['state'], X)


def nested_factory(candidates: dict[str, dict[str, Any]], calib: str = 'none', use_cache: bool = True,
                   cache_dir: str | None = None):
    return M.make_fit_predict(NestedSelect(candidates), calib=calib, use_cache=use_cache, cache_dir=cache_dir)


def nested_spec() -> Spec:
    cands = {n: BY_NAME[n].params for n in DEEP_ARCH_CANDIDATES}
    return Spec('m3_deep_nested', 'nested', 'approach', {'candidates': cands},
                'deep, architecture chosen in every fold', 'nested',
                notes='nested: per outer fold, the architecture with the lowest cross-fitted log loss on the training '
                      f'persons among {len(cands)} candidates')


# ───────────────────────────── nested choice among every deep configuration ─────────────────────────────

NESTED_ALL_INPUT = 'deep_all'          # the flat multi-input dict of m3_nested (one entry per candidate input kind)


def deep_role_candidates() -> tuple[str, ...]:
    """Every base configuration of the 'deep' role (architectures and data treatments; no physics demonstration, no
    grip-transfer check): the candidates of the role's outer-fold choice and of its nested counterpart."""
    return tuple(s.name for s in SPECS if s.group in ('arch', 'treat') and not s.physics_demo and s.name in BASE
                 and s.family in ('deep', 'body'))


def nested_all_kinds() -> tuple[str, ...]:
    return tuple(sorted({BY_NAME[n].x for n in deep_role_candidates()}))


def nested_all_spec() -> Spec:
    cands = {n: {'family': BY_NAME[n].family, 'x': BY_NAME[n].x, 'params': BY_NAME[n].params}
             for n in deep_role_candidates()}
    return Spec('m3_deep_nested_all', 'nested_all', NESTED_ALL_INPUT, {'candidates': cands},
                'deep, configuration chosen in every fold', 'nested',
                notes='nested: per outer fold, the deep configuration (architecture and data treatment) with the lowest '
                      f'cross-fitted log loss on the training persons among {len(cands)} candidates, grip-dependent '
                      'device-frame inputs included; every candidate fit is the cached fit of its own experiment',
                post_hoc=any(BY_NAME[n].post_hoc for n in cands))


def nested_inv_spec() -> Spec:
    """The nested estimate of the shipping policy: only grip-invariant candidates (the grip rule excludes inputs whose
    values depend on the grip, because production grips are free)."""
    cands = {n: {'family': BY_NAME[n].family, 'x': BY_NAME[n].x, 'params': BY_NAME[n].params}
             for n in deep_role_candidates() if not grip_dependent(BY_NAME[n])}
    return Spec('m3_deep_nested_inv', 'nested_all', NESTED_ALL_INPUT, {'candidates': cands},
                'deep, grip-invariant configuration chosen in every fold', 'nested',
                notes='nested: per outer fold, the grip-invariant deep configuration with the lowest cross-fitted log '
                      f'loss on the training persons among {len(cands)} candidates (the shipping policy: no '
                      'device-frame input); every candidate fit is the cached fit of its own experiment',
                post_hoc=any(BY_NAME[n].post_hoc for n in cands))


NESTED_SPECS = ('m3_deep_nested', 'm3_deep_nested_all', 'm3_deep_nested_inv')


def nested_spec_of(name: str) -> Spec:
    return {'m3_deep_nested': nested_spec, 'm3_deep_nested_all': nested_all_spec,
            'm3_deep_nested_inv': nested_inv_spec}[name]()
