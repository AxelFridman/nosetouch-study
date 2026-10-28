"""Lane p1: printed names, option roles and report families. A LIGHT module (only ``stage2.config``), so the
presentation steps (figures, tables, summary) import it in a fraction of a second; edit a label here and redraw.

Roles of the experiments (how each one is read; set here, never in the experiment specs, so relabelling never
re-runs anything):

* ``option``      a data-treatment alternative to a current default, compared PAIRED with the panel model under the
                  current pipeline; the family of these comparisons is the one the multiplicity correction and the
                  decision rule apply to;
* ``robustness``  a test-time perturbation of a model trained on the current pipeline (segment bounds moved): a cost
                  that a production detector must stay within, never an option to adopt;
* ``remedy_arm``  a remedy that is judged only against the problem it addresses (:data:`REMEDY_PAIRS` in
                  :mod:`p1_derive`), e.g. training with jittered bounds against oracle-trained, both tested on
                  jittered bounds;
* ``scoring``     a run whose only purpose is to score trials an exclusion rule removes, with a model trained on
                  EXACTLY the base training set (D2 cap applied after the training filter); its primary trials are
                  identical to the base run's, so it is not an option;
* ``control``     a size-matched control (random training rows or persons removed, as many as an option removes):
                  it separates an option's effect from the smaller training set (:data:`CONTROL_PAIRS`).
"""
from __future__ import annotations

from stage2 import config as C

MODELS = ('lr', 'gbm', 'mr')
MODEL_LABELS = {'lr': 'Logistic (chiral bins)', 'gbm': 'LightGBM (summary features)', 'mr': 'MiniRocket + ridge'}
SHORT = {'lr': 'LR', 'gbm': 'GBM', 'mr': 'MR'}
#: the panel models of the calibration study (fixed in advance: the tree model and the linear one; MiniRocket
#: already maps its ridge score to P(right) by an inner Platt step)
CALIB_MODELS = ('gbm', 'lr')

#: winsorising quantile of the 'winsor' option (each channel clipped at q and 1 - q of the training rows)
WINSOR_Q = 0.005
#: tightened exclusion thresholds for the E4b / E4c sensitivity: twice the current ones (config, not data)
TIGHT_FACTOR = 2.0
TIGHT_DURATION_MS = TIGHT_FACTOR * C.MIN_DURATION_MS
TIGHT_PEAK_DPS = TIGHT_FACTOR * C.MIN_PEAK_GYRO_DPS
#: cut-off of the 'lowpass' gravity estimate (= p1_variants.LOWPASS_GRAVITY_HZ; a test checks they agree)
LOWPASS_GRAVITY_HZ = 1.0
#: strata with fewer persons than this get no person-bootstrap CI (a CI from 1-4 clusters is meaningless)
MIN_PERSONS_CI = 5
#: predictions of two runs that differ by less than this everywhere are 'identical' (solver precision)
IDENTICAL_TOL = 1e-6
#: per-person accuracy below this counts a person as poorly served (the 'persons below' counts)
PERSON_ACC_THRESHOLD = 0.8

FLAGS = ('grip_mismatch', 'false_start', 'seg_disagree_100ms', 'repeated_samples', 'clip_candidate')
FLAG_LABELS = {'grip_mismatch': 'grip mismatch', 'false_start': 'false start',
               'seg_disagree_100ms': 'segmenters disagree > 100 ms', 'repeated_samples': 'repeated samples dropped',
               'clip_candidate': 'possible sensor clipping'}

#: printed names of the options (tables and figures)
OPTION_LABELS = {
    'base': 'current pipeline', 'lp_none': 'no low-pass', 'lp5': 'low-pass 5 Hz', 'lp10': 'low-pass 10 Hz',
    'lp20': 'low-pass 20 Hz', 'lp_order2': 'filter order 2', 'lp_order8': 'filter order 8',
    'lp_causal': f'causal filter ({C.LOWPASS_HZ:g} Hz)', 'gap50': 'bridge gaps up to 50 ms',
    'gap250': 'bridge gaps up to 250 ms',
    'interp_cubic': 'cubic-spline resampling', 'interp_hold': 'sample-and-hold resampling',
    'e4d_scored': 'E4d trials scored (linear imputation)', 'e4d_trained': 'E4d trials added to training',
    'e4d_scored_cubic': 'E4d trials scored (cubic imputation)', 'e4d_scored_hold': 'E4d trials scored (hold imputation)',
    'fs30': '30 Hz', 'fs20': '20 Hz', 'quant_ios': 'iOS rounded to 0.1', 'tjit2': 'time jitter SD 2 ms',
    'tjit5': 'time jitter SD 5 ms', 'tjit10': 'time jitter SD 10 ms',
    'grav_lowpass': f'gravity: {LOWPASS_GRAVITY_HZ:g}-Hz low-pass',
    'grav_browser': 'gravity: browser fusion', 'frame_gravity': 'gravity-frame channels',
    'frame_device': 'device-frame channels', 'frame_all': 'all 17 channels', 'frame_chiral': 'chiral channels',
    'win_first_half': 'first half only', 'win_hold': 'approach + hold', 'win_pre': 'pre-onset context',
    'win_fixed': 'fixed 3-s window', 'bjit50': 'bounds ±50 ms (test)', 'bjit100': 'bounds ±100 ms (test)',
    'bjit200': 'bounds ±200 ms (test)', 'onset_late100': 'onset 100 ms late (test)',
    'bjit100_trained': 'bounds ±100 ms, jitter-trained', 'norm_zscore': 'z-score per channel',
    'norm_robust': 'median / IQR', 'norm_trial_peak': 'per-trial peak',
    'winsor': f'winsorised at {100 * WINSOR_Q:g} % / {100 - 100 * WINSOR_Q:g} %',
    'drop_grip_mismatch': 'no grip-mismatch trials (train)', 'drop_false_start': 'no false starts (train)',
    'drop_seg_disagree_100ms': 'no segmentation disagreement (train)',
    'drop_repeated_samples': 'no repeated-sample trials (train)', 'drop_clip_candidate': 'no clip candidates (train)',
    'drop_all_flags': 'no flagged trials (train)', 'drop_swap_blocks': 'no swap blocks (train)',
    'drop_all_flags_sizematch': 'as many random trials out (train)',
    'excl_loose': 'E4b/E4c trials added to training', 'excl_scored': 'E4b/E4c trials scored',
    'excl_tight': f'E4b/E4c thresholds ×{TIGHT_FACTOR:g} (train)',
    'platform_cross': 'other platform only', 'platform_same': 'same platform only',
    'platform_cross_sizematch': 'as many random persons as the other platform',
    'platform_same_sizematch': 'as many random persons as the same platform',
    'no_mirror': 'no mirror augmentation', 'time_warp': '+ time warp', 'screen_flip': '+ screen-away flips',
}
FAMILY_LABELS = {
    'filter': 'Filtering', 'gaps': 'Gaps and imputation', 'sampling': 'Sampling and precision',
    'gravity': 'Gravity estimate', 'frame': 'Frame (channel set)', 'window': 'Window and time base',
    'segmentation': 'Segmentation error', 'normalisation': 'Normalisation', 'outliers': 'Outliers and corrections',
    'exclusions': 'Exclusion rules (training side)', 'platform': 'Platform transfer', 'augmentation': 'Augmentation',
}
#: the decision families in report order (the 'segmentation' family holds robustness costs, not options)
DECISION_FAMILIES = ('filter', 'gaps', 'sampling', 'gravity', 'frame', 'window', 'normalisation', 'outliers',
                     'exclusions', 'platform', 'augmentation')

ROBUSTNESS_OPTIONS = ('bjit50', 'bjit100', 'bjit200', 'onset_late100')
REMEDY_ARMS = ('bjit100_trained',)
SCORING_OPTIONS = ('e4d_scored', 'e4d_scored_cubic', 'e4d_scored_hold', 'excl_scored')
CONTROL_OPTIONS = ('drop_all_flags_sizematch', 'platform_same_sizematch', 'platform_cross_sizematch')
#: options whose grid family is re-read as another report family (the grid spec keeps its original family)
REPORT_FAMILY = {'e4d_trained': 'exclusions', 'excl_loose': 'exclusions', 'excl_tight': 'exclusions'}


def role(option: str) -> str:
    """How the comparison of this option with the base run is read (module docstring)."""
    if option in ROBUSTNESS_OPTIONS:
        return 'robustness'
    if option in REMEDY_ARMS:
        return 'remedy_arm'
    if option in SCORING_OPTIONS:
        return 'scoring'
    if option in CONTROL_OPTIONS:
        return 'control'
    return 'option'


def report_family(option: str, family: str) -> str:
    return REPORT_FAMILY.get(option, family)


def option_label(option: str) -> str:
    return OPTION_LABELS.get(option, option)


def subset_label(label: str) -> str:
    """Printed name of a p1_subsets row label ('grip_mismatch: flagged' -> 'flagged: grip mismatch')."""
    for k, v in FLAG_LABELS.items():
        if label == f'{k}: flagged':
            return f'flagged: {v}'
        if label == f'{k}: not flagged':
            return f'not flagged: {v}'
    return label


#: verdict words of a paired comparison (p1_derive.verdicts): the multiplicity-adjusted call and the nominal one
VERDICT_LABELS = {'identical': 'identical (solver precision)', 'better': 'better (BH q < 0.05)',
                  'worse': 'worse (BH q < 0.05)', 'nominally better': 'nominally better (CI > 0, q >= 0.05)',
                  'nominally worse': 'nominally worse (CI < 0, q >= 0.05)', 'no difference': 'no difference'}

CAL_LABELS = {'none': 'uncalibrated', 'auto': 'nested choice', 'platt': 'Platt', 'temperature': 'temperature',
              'isotonic': 'isotonic', 'beta': 'beta'}
RULE_LABELS = {'mean_prob': 'mean of P', 'mean_logit': 'mean of logits', 'sum_logit': 'sum of logits',
               'sum_logit_T': 'sum of logits / T_k'}

__all__ = ['MODELS', 'MODEL_LABELS', 'SHORT', 'OPTION_LABELS', 'FAMILY_LABELS', 'FLAG_LABELS', 'role',
           'report_family', 'option_label', 'subset_label', 'DECISION_FAMILIES', 'CAL_LABELS', 'RULE_LABELS']
