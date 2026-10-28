"""Lane m1 experiment registry: every experiment, its model factory, parameters, input and design role.

ONE place to read what lane m1 runs. Each :class:`Spec` names a factory of ``m1_models`` and its JSON-able
parameters, so a finalist is rebuilt exactly with ``getattr(m1_models, factory)(**params)`` on
``getattr(m1_models, build_x)(d)``.

Feature sets (named here, nowhere else):

* ``FHY``: first-half yaw, the pilot's feature (docs/09 section 2.1). The pilot is P01's protocol-1.0.0 session,
  which is part of the evaluation set: the zero-parameter rule fixes its direction from it, the finalist rule
  learns the direction on the training persons instead (identical decisions here).
* ``PHYS6``: the three chiral components of the body-frame decomposition (yaw about gravity, roll about the
  facing axis, lateral velocity), each at the half and at the end of the approach. Physics-motivated (a symmetric
  3 x 2 design, nothing tuned) but specified AFTER the pooled description of the components (all persons, labels),
  so it is not pre-registered. Its guards: the nested choice among seven physical channel sets
  (``m1_lr_sets_nested``) and the in-fold forward selection over every chiral approach feature (``m1_lr_fwd``).
* ``EVEN_MOD``: even descriptors (duration, speed, tilt, effort) an unconstrained model may use.

Design roles (``Spec.design``) and ``n_configs_tried``:

* ``pre-specified``: fixed before any leave-one-person-out result (the pilot rule): 1;
* ``post-pooled``: PHYS6, fixed after the pooled description but not chosen by an outer-fold comparison: 1;
* ``nested``: a procedure whose choice (feature set, data treatment) is made inside every fold on
  fold.inner_splits(), so its estimate includes the selection: 1. ``m1_lr_phys6_nested`` was added after the audit
  of the first run, which had seen that asinh compression won on the outer folds; its options are the lane's
  pre-defined treatment set;
* ``family``: one configuration of a family compared on the outer folds (``FAMILIES``): the family size plus the
  anchor it would replace, reported as exploratory;
* ``sensitivity``: a finalist anchor re-run with training rows removed: 1 (paired against its anchor);
* ``analysis``: a window or control that answers a question (absolute time, lead-in, negative control) and is never a
  finalist candidate: 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from stage2.lanes import m1_features as MF
from stage2.lanes import m1_models as _M

FHY = ['yaw_q50']
YAWQ = ['yaw_q25', 'yaw_q50', 'yaw_q75', 'yaw_q100']
YAW2 = ['yaw_q50', 'yaw_q100']
ROLL2 = ['roll_q50', 'roll_q100']
LAT2 = ['latvel_q50', 'latvel_q100']
LATDD2 = [f'latvel_dd_q{MF._pct(q)}' for q in MF.DEDRIFT_QUARTILES]
PHYS6 = YAW2 + ROLL2 + LAT2
FIX6 = ['yaw_fix_0p5', 'yaw_fix_1p0', 'roll_fix_0p5', 'roll_fix_1p0', 'latvel_fix_0p5', 'latvel_fix_1p0']
LEAD3 = ['yaw_lead', 'roll_lead', 'latvel_lead']
PRE3 = ['yaw_pre', 'roll_pre', 'latvel_pre']
EVEN_MOD = ['duration_s', 'peak_gyro', 'peak_acc', 'tilt_end', 'screen_up_start', 'acc_horiz_rms']
PHYS6_AN = [f'{c}_an' for c in PHYS6]
#: candidates of the in-fold best single-feature rule: every odd feature except the negative control and the
#: de-drifted variants (a data-treatment option)
RULE_POOL = [f.name for f in MF.CATALOGUE if f.parity == 'odd' and f.family not in ('control', 'dedrift')]
#: candidates of the nested forward selection: the chiral features of the approach window (the 8-bin yaw rates
#: left out: the 4-bin rates and the quartile integrals already span them; keeps the in-fold search affordable)
FWD_POOL = [c for c in MF.CHIRAL_APPROACH if not c.startswith('yaw_rate_b8_')]
TREE_POOL = list(MF.CHIRAL_APPROACH) + list(MF.EVEN)
#: the declared physical channel sets of the nested channel-set guard (all with the RMS treatment)
CHANNEL_SETS: dict[str, list[str]] = {'yaw': YAW2, 'roll': ROLL2, 'lateral': LAT2, 'yaw_roll': YAW2 + ROLL2,
                                      'yaw_lateral': YAW2 + LAT2, 'roll_lateral': ROLL2 + LAT2, 'phys6': PHYS6}
#: the declared data treatments of PHYS6 among which the nested model chooses in every fold (the baseline first:
#: ties go to it)
TREATMENTS: dict[str, dict[str, Any]] = {'rms': {'cols': PHYS6}, 'none': {'cols': PHYS6, 'scale': 'none'},
                                         'robust': {'cols': PHYS6, 'scale': 'robust'},
                                         'clip': {'cols': PHYS6, 'winsor': 'clip'},
                                         'asinh': {'cols': PHYS6, 'winsor': 'asinh'},
                                         'ampnorm': {'cols': PHYS6_AN}}
GAM_PARAMS: dict[str, Any] = {'cols': PHYS6, 'knots': list(_M.GAM_KNOTS), 'Cs': list(_M.GAM_CS), 'symmetrise': True}
TREE_PARAMS: dict[str, Any] = {'cols': TREE_POOL, 'depths': list(_M.TREE_DEPTHS), 'leafs': list(_M.TREE_LEAFS)}

#: the reference of the registry (core unit 'reference') every m1 model is paired against
REFERENCE = 'example_yaw_logreg'
#: the baseline of the paired data-treatment comparisons
BASELINE = 'm1_lr_phys6'
#: the rule every fitted finalist anchor is paired against ("what do fitted numbers add?")
RULE = 'm1_rule_fhy_sign_cal'
#: the anchor of each finalist slot (m1_select): the sensitivity runs and the rule contrasts use them
ANCHORS = {'rule': RULE, 'small': 'm1_lr_phys6_nested', 'tree_gam': 'm1_gam_phys6'}


@dataclass(frozen=True)
class Spec:
    name: str
    label: str
    group: str                   # rules | small_lr | channels | representation | treatment | temporal | tree | gam | calibration | sensitivity
    factory: str                 # function of m1_models
    params: dict[str, Any] = field(default_factory=dict)
    x: str = 'build_x'           # function of m1_models that builds X from a Stage2Data
    notes: str = ''
    store: bool = False          # save per-fold artefacts (tables/m1/m1_store_<name>.json)
    nested: bool = False         # hyperparameters / features chosen inside fit_predict on fold.inner_splits()
    baseline: str | None = None  # paired comparison against this experiment (besides the reference)
    design: str = 'family'       # pre-specified | post-pooled | nested | family | sensitivity | analysis
    dynamic: str | None = None   # 'swap_blocks': params['blocks'] is filled from the data at compute time
    n_configs: int = 1           # configurations of its family compared on the outer folds (set below)


def _lin(name, label, group, cols, notes='', store=False, baseline=BASELINE, design='family', **kw) -> Spec:
    return Spec(name, label, group, 'make_linear', {'cols': list(cols), **kw}, notes=notes, store=store,
                nested=True, baseline=baseline, design=design)


SPECS: list[Spec] = [
    # ── 2. zero- and one-parameter rules ──
    Spec('m1_rule_fhy_sign', 'first-half yaw > 0 (no training)', 'rules', 'make_sign_rule',
         {'col': 'yaw_q50', 'direction': 1.0, 'calibrated': False}, design='pre-specified',
         notes='zero-parameter rule of the pilot (docs/09 2.1): right iff first-half yaw > 0; P in {0, 1}; the '
               'direction comes from the pilot session of P01, a person of the evaluation set'),
    Spec('m1_rule_fhy_sign_cal', 'first-half yaw sign, hit rate as P', 'rules', 'make_sign_rule',
         {'col': 'yaw_q50', 'direction': None, 'calibrated': True}, design='pre-specified',
         notes="the pilot rule with its direction learned on the training persons and P(right) = the rule's "
               'training hit rate q (two numbers per fold)'),
    Spec('m1_rule_fhy_threshold', 'first-half yaw, fitted threshold', 'rules', 'make_threshold_rule',
         {'col': 'yaw_q50'}, notes='direction and threshold fitted on the training persons (max accuracy)',
         baseline=RULE),
    Spec('m1_rule_nested', 'best single-feature rule (in-fold choice)', 'rules', 'make_nested_rule',
         {'cands': RULE_POOL}, nested=True, baseline=RULE,
         notes='feature, direction and threshold chosen inside every fold by inner-split accuracy'),
    # ── 3. small logistic models ──
    _lin('m1_lr_fhy', 'logistic: first-half yaw', 'small_lr', FHY, notes='antisymmetric, 1 feature'),
    _lin('m1_lr_yawq', 'logistic: yaw at 25/50/75/100 %', 'small_lr', YAWQ, notes='antisymmetric, 4 features'),
    _lin('m1_lr_phys6', 'logistic: yaw + roll + lateral (PHYS6)', 'small_lr', PHYS6, store=True, fit_agg=True,
         baseline=None, design='post-pooled',
         notes='antisymmetric (no intercept, RMS-scaled odd features); the baseline of lane m1; specified after the '
               'pooled description of the components (not pre-registered)'),
    _lin('m1_lr_phys6_aug', 'PHYS6, intercept + mirror augmentation', 'small_lr', PHYS6, antisym=False,
         augment=True, scale='zscore', notes='equivalent to the antisymmetric model at C -> 2C'),
    _lin('m1_lr_phys6_free', 'PHYS6 + even descriptors, unconstrained', 'small_lr', PHYS6 + EVEN_MOD,
         antisym=False, augment=False, scale='zscore',
         notes='intercept, no augmentation: free to use asymmetries (dominance, platform) and even features'),
    _lin('m1_lr_phys6_ix', 'PHYS6 x log(duration, peak speed)', 'small_lr', PHYS6 + ['duration_s', 'peak_gyro'],
         interactions=[(c, e) for c in PHYS6 for e in ('duration_s', 'peak_gyro')],
         notes='antisymmetric with odd x even interactions: speed and duration modulate the chiral slopes'),
    _lin('m1_lr_chiral_all', 'logistic: all approach chiral features', 'small_lr', FWD_POOL,
         notes='every chiral approach feature, L2 only (no selection)'),
    Spec('m1_lr_fwd', 'logistic: forward selection (in-fold, <= 4)', 'small_lr', 'make_forward',
         {'cands': FWD_POOL, 'max_k': 4}, nested=True, baseline=BASELINE,
         notes='antisymmetric; features chosen inside every fold by inner log loss (a guard against PHYS6 optimism)'),
    # ── which component carries the hand (channel sets) ──
    _lin('m1_lr_yaw2', 'yaw only', 'channels', YAW2),
    _lin('m1_lr_roll2', 'roll only', 'channels', ROLL2),
    _lin('m1_lr_lat2', 'lateral only', 'channels', LAT2),
    _lin('m1_lr_yaw_roll', 'yaw + roll', 'channels', YAW2 + ROLL2),
    _lin('m1_lr_yaw_lat', 'yaw + lateral', 'channels', YAW2 + LAT2),
    Spec('m1_lr_sets_nested', 'channel set chosen in-fold', 'channels', 'make_linear_select',
         {'options': {k: {'cols': v} for k, v in CHANNEL_SETS.items()}}, nested=True, baseline=BASELINE,
         design='nested', notes='guard of PHYS6: the set (yaw, roll, lateral, the three pairs or PHYS6) and C chosen '
                                'inside every fold by inner log loss'),
    # ── representation: absolute time, lead-in, negative control ──
    _lin('m1_lr_fix6', 'PHYS6 at 0.5 / 1.0 s (absolute time)', 'representation', FIX6, design='analysis',
         notes='the three components at fixed times after the onset: independent of the end bound'),
    _lin('m1_lr_lead', 'lead-in before the onset', 'representation', LEAD3, design='analysis',
         notes=f'the three components over [onset - {MF.LEAD_S:g} s, onset]'),
    _lin('m1_lr_control', 'negative control: still hold before the onset', 'representation', PRE3, design='analysis',
         notes=f'the three components over [onset - {MF.PRE_S[0]:g} s, onset - {MF.PRE_S[1]:g} s]: expected chance'),
    # ── data treatment on PHYS6: scaling, winsorising, per-trial amplitude, de-drifting ──
    _lin('m1_lr_phys6_scale_none', 'no scaling (raw units)', 'treatment', PHYS6, scale='none'),
    _lin('m1_lr_phys6_scale_robust', 'robust scaling (median |x|)', 'treatment', PHYS6, scale='robust'),
    _lin('m1_lr_phys6_win_clip', 'winsorised at the training 99th percentile', 'treatment', PHYS6, winsor='clip'),
    _lin('m1_lr_phys6_win_asinh', 'asinh-compressed tails', 'treatment', PHYS6, winsor='asinh', store=True,
         fit_agg=True),
    Spec('m1_lr_phys6_ampnorm', 'per-trial amplitude-normalised', 'treatment', 'make_linear', {'cols': PHYS6_AN},
         x='build_x_an', nested=True, baseline=BASELINE,
         notes="each component divided by the trial's own activity (shape, not size)"),
    _lin('m1_lr_phys6_dedrift', 'lateral de-drifted (hold bias removed)', 'treatment', YAW2 + ROLL2 + LATDD2,
         notes="PHYS6 with the lateral velocities integrated after removing the trial's mean a.s over the still hold"),
    Spec('m1_lr_phys6_nested', 'PHYS6, data treatment chosen in-fold', 'treatment', 'make_linear_select',
         {'options': TREATMENTS, 'fit_agg': True}, x='build_x_treat', store=True, nested=True, baseline=BASELINE,
         design='nested', notes='the treatment (RMS scaling, none, robust, clip, asinh, amplitude-normalised) and C '
                                'chosen jointly inside every fold by inner log loss'),
    # ── temporal filters ──
    Spec('m1_tf_yaw', 'temporal filter: yaw rate', 'temporal', 'make_linear',
         {'cols': MF.tc_columns(('yaw_rate',)), 'scale': 'group_rms'}, x='build_tc', store=True, nested=True,
         baseline=BASELINE, notes='antisymmetric logistic on the binned yaw-rate time course; one scale for all bins'),
    Spec('m1_tf_multi', 'temporal filter: yaw + roll + lateral', 'temporal', 'make_linear',
         {'cols': MF.tc_columns(), 'scale': 'group_rms', 'fit_agg': True}, x='build_tc', store=True, nested=True,
         baseline=BASELINE, notes='antisymmetric logistic on the three binned chiral rate time courses; one scale per '
                                  'channel'),
    # ── shallow tree and GAM ──
    Spec('m1_tree', 'shallow tree (depth <= 3)', 'tree', 'make_tree', dict(TREE_PARAMS), store=True, nested=True,
         baseline=BASELINE, notes='mirror-augmented and symmetrised; depth and leaf size chosen in-fold'),
    Spec('m1_gam_phys6', 'GAM on PHYS6 (splines)', 'gam', 'make_gam', {**GAM_PARAMS, 'fit_agg': True}, store=True,
         nested=True, baseline=BASELINE,
         notes='cubic splines + L2 logistic, mirror-augmented and symmetrised; number of knots and C chosen in-fold'),
]
# ── calibration remedies (fitted on inner cross-fitted predictions of the training persons) ──
REMEDIES = (('platt', 'platt', 'Platt'), ('temperature', 'temp', 'temperature'), ('isotonic', 'iso', 'isotonic'),
            ('beta', 'beta', 'beta'))
#: base -> (factory, params without calibration, input, label)
CAL_BASES = {
    'm1_lr_phys6': ('make_linear', {'cols': PHYS6}, 'build_x', 'PHYS6 logistic'),
    'm1_lr_phys6_nested': ('make_linear_select', {'options': TREATMENTS}, 'build_x_treat', 'PHYS6 nested treatment'),
    'm1_tf_multi': ('make_linear', {'cols': MF.tc_columns(), 'scale': 'group_rms'}, 'build_tc', 'temporal filter'),
    'm1_gam_phys6': ('make_gam', dict(GAM_PARAMS), 'build_x', 'GAM'),
}
for _base, (_fac, _params, _x, _lab) in CAL_BASES.items():
    for _cal, _short, _cl in REMEDIES:
        SPECS.append(Spec(f'{_base}_cal_{_short}', f'{_lab} + {_cl}', 'calibration', _fac,
                          {**_params, 'calibration': _cal, 'fit_agg': True}, x=_x, nested=True, baseline=_base,
                          notes=f'{_cl} calibration fitted on the inner cross-fitted predictions of the training persons'))
# ── sensitivity: the finalist anchors trained without suspected wrong-hand rows (evaluation unchanged) ──
for _slot, _anchor in ANCHORS.items():
    _s = next(s for s in SPECS if s.name == _anchor)
    _bp = dict(_s.params)
    SPECS.append(Spec(f'm1_sens_{_slot}_noswap', f'{_s.label}, swap-signature blocks out of training', 'sensitivity',
                      'make_train_filter', {'factory': _s.factory, 'params': _bp, 'drop': 'blocks', 'blocks': []},
                      x='build_x_screen', nested=_s.nested, baseline=_anchor, design='sensitivity',
                      dynamic='swap_blocks',
                      notes="training without the blocks of stage2.screen.block_consistency's swap signature "
                            '(data-driven); evaluation unchanged'))
    SPECS.append(Spec(f'm1_sens_{_slot}_d6', f'{_s.label}, D6 screen in every fold', 'sensitivity', 'make_train_filter',
                      {'factory': _s.factory, 'params': _bp, 'drop': 'd6', 'screen_cols': MF.screen_columns()},
                      x='build_x_screen', nested=_s.nested, baseline=_anchor, design='sensitivity',
                      notes='training without the trials the docs/09 D6 screen flags inside the fold (p(label) < 0.05 '
                            'or runs of >= 3 confidently reversed trials); evaluation unchanged'))

#: The families compared on the outer folds when a finalist is chosen from them (n_configs_tried of a member).
_NOT_FAMILY = ('pre-specified', 'post-pooled', 'nested', 'sensitivity', 'analysis')
FAMILIES = {
    'rule': [s.name for s in SPECS if s.group == 'rules' and s.design not in _NOT_FAMILY],
    'small_lr': [s.name for s in SPECS if s.design not in _NOT_FAMILY
                 and (s.group in ('small_lr', 'channels', 'treatment', 'temporal')
                      or (s.group == 'calibration' and s.factory in ('make_linear', 'make_linear_select')))],
    'tree_gam': [s.name for s in SPECS if s.group in ('tree', 'gam')
                 or (s.group == 'calibration' and s.factory == 'make_gam')],
}
#: the anchor each family member would replace (counted in its n_configs_tried)
FAMILY_ANCHOR = {'rule': ANCHORS['rule'], 'small_lr': ANCHORS['small'], 'tree_gam': None}
FAMILY_SIZE = {f: len(v) + (1 if FAMILY_ANCHOR[f] and FAMILY_ANCHOR[f] not in v else 0) for f, v in FAMILIES.items()}
SPECS = [replace(s, n_configs=next((FAMILY_SIZE[f] for f, v in FAMILIES.items() if s.name in v), 1)
                 if s.design == 'family' else 1) for s in SPECS]
BY_NAME: dict[str, Spec] = {s.name: s for s in SPECS}
#: the experiments fixed before any outer-fold comparison
PRESPECIFIED = tuple(s.name for s in SPECS if s.design == 'pre-specified')
GROUP_ORDER = ('rules', 'small_lr', 'channels', 'representation', 'treatment', 'temporal', 'tree', 'gam', 'calibration',
               'sensitivity')
GROUP_LABELS = {'rules': 'Rules', 'small_lr': 'Small logistic models', 'channels': 'Which component carries the hand',
                'representation': 'Window and controls', 'treatment': 'Data treatment (PHYS6)',
                'temporal': 'Temporal filters', 'tree': 'Shallow tree', 'gam': 'GAM', 'calibration': 'Calibration',
                'sensitivity': 'Sensitivity: suspected wrong-hand training rows'}
DESIGN_LABELS = {'pre-specified': 'pre-specified', 'post-pooled': 'specified after the pooled description',
                 'nested': 'nested (choice inside every fold)', 'family': 'outer-fold family member (exploratory)',
                 'sensitivity': 'sensitivity run', 'analysis': 'analysis of a window (not a finalist candidate)'}


def spec_params_json(s: Spec) -> dict[str, Any]:
    """JSON-able parameters (tuples -> lists)."""
    def conv(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return conv(dict(s.params))


def with_dynamic(s: Spec, swap_blocks: list[tuple[str, int]]) -> Spec:
    """The spec with its data-driven parameters filled in (the swap-signature blocks of the current data)."""
    if s.dynamic == 'swap_blocks':
        return replace(s, params={**s.params, 'blocks': [[str(a), int(b)] for a, b in swap_blocks]})
    return s
