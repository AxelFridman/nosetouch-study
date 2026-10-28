"""Lane p1: every number the report text, the summary and the figure texts cite, built from the derived files only.

``build()`` -> ``{key: report_export.numbers_entry}``. The numbers step writes it to ``results/p1_numbers.json``; the
summary (:mod:`p1_summary`) and the figures (:mod:`p1_figures`) print every number through :class:`Numbers`, which
formats an entry exactly as its LaTeX macro will (``report_export.format_value``) and raises on a key that is not in
the file, so the printed text and the numbers file cannot disagree. A LIGHT module (config, report_export, the
derived JSON files): nothing here imports numpy-heavy lane code or fits anything.

Key scheme (``rk`` = a result name without ``p1_``, e.g. ``gbm_lp5``; ``m`` = lr / gbm / mr):

* runs: ``acc_<rk>`` (+ ``_lo`` / ``_hi``: person-bootstrap 95 % CI), ``n_<rk>`` (trials);
* headline runs (``hl`` = lr, gbm, mr, lr_cal, gbm_cal): ``<hl>_<metric>`` for accuracy, auc, log_loss, brier (each with
  ``_lo`` / ``_hi``), ece with its perfect-calibration null (``_ece_null_q95``, ``_ece_p``), calibration slope and
  calibration-in-the-large with CIs, k = 3 accuracy, natural use (coverage, answered, half credit with CI, forced
  choice: the harness's rule until 2026-10-28), per-person summary;
* comparisons: ``d_<rk>`` (accuracy difference in POINTS, + ``_lo`` / ``_hi``), ``dll_<rk>`` (mean per-person log-loss
  difference, + CI), ``p_<rk>`` (exact sign-flip p), ``q_<rk>`` (BH q in its family), ``holm_<rk>``, ``v_<rk>`` /
  ``vll_<rk>`` (verdict words), ``maxdp_<rk>`` and ``nhc_<rk>`` (largest |change of P(right)|, hands changed);
  remedy pairs ``rem_<rk of A>_*`` and option-against-control pairs ``ctl_<rk of A>_*`` alike;
* subsets: ``sub_<key>`` (+ ``_lo`` / ``_hi``, ``_n``, ``_np``, ``_status``, ``_ntrain`` for the platform rows);
* best-model breakdowns ``best_*``, ``cond_*``, ``grip_*``; calibration ``cal_*``, ``calpd_*`` (paired), ``caltest_*``;
  aggregates ``agg_*``, ``aggd_*``; decisions ``dec_*``; filter delay ``delay_*``; variants ``vdiff_*``; settings.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from stage2 import config as C
from stage2 import report_export as RX
from stage2.lanes import p1_labels as LB
from stage2.pipeline import Lane

L = Lane('p1')
DERIVED = ('p1_runs', 'p1_comparisons', 'p1_subsets', 'p1_best', 'p1_calibration', 'p1_decisions', 'p1_filter_delay',
           'p1_variants')
SRC = {n: f'tables/p1/{n}.json' for n in DERIVED}
HEADLINE = {'lr': 'p1_lr_base', 'gbm': 'p1_gbm_base', 'mr': 'p1_mr_base', 'lr_cal': 'p1_calib_lr_auto',
            'gbm_cal': 'p1_calib_gbm_auto'}
PCT, PTS, F3, F2, PV, D, TXT = '.1%', '+.1f', '.3f', '.2f', '.2g', 'd', ''


def rkey(name: str) -> str:
    """A result name as a key part: 'p1_gbm_lp5' -> 'gbm_lp5'."""
    k = name[len('p1_'):] if name.startswith('p1_') else name
    return re.sub(r'[^a-z0-9_]+', '_', k.lower()).strip('_')


def slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def _num(v: Any) -> Any:
    """None for a missing / non-finite value (the entry then prints '--')."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v if not (isinstance(v, float) and not math.isfinite(v)) else None
    return v


def load() -> dict[str, Any]:
    return {n: L.read_json(n) for n in DERIVED}


class Builder:
    def __init__(self) -> None:
        self.n: dict[str, dict[str, Any]] = {}

    def add(self, key: str, value: Any, fmt: str, src: str, desc: str) -> None:
        if key in self.n:
            raise ValueError(f'duplicate p1 number key {key!r}')
        v = _num(value)
        if fmt == D and v is not None and not isinstance(v, str):
            v = int(round(float(v)))
        self.n[key] = RX.numbers_entry(v, fmt, SRC.get(src, src), desc)

    def pts(self, key: str, value: Any, src: str, desc: str) -> None:
        """An accuracy difference as percentage POINTS (value x 100, signed)."""
        v = _num(value)
        self.add(key, None if v is None else 100.0 * float(v), PTS, src, desc + ' (percentage points)')

    def triple(self, key: str, v: Any, lo: Any, hi: Any, fmt: str, src: str, desc: str, points: bool = False) -> None:
        put = self.pts if points else (lambda k, x, s, d: self.add(k, x, fmt, s, d))
        put(key, v, src, desc)
        put(f'{key}_lo', lo, src, f'{desc}: 95% CI low')
        put(f'{key}_hi', hi, src, f'{desc}: 95% CI high')


# ───────────────────────────── sections ─────────────────────────────


def _settings(B: Builder, S: dict[str, Any]) -> None:
    runs = S['p1_runs']
    pan = runs.get('panel', {})
    mods = pan.get('models', {})
    src = 'p1_runs'
    lr, gbm, mr = mods.get('lr', {}), mods.get('gbm', {}), mods.get('mr', {})
    B.add('n_panel', len(LB.MODELS), D, 'stage2/lanes/p1_labels.py', 'panel models (pre-specified)')
    B.add('lr_c', lr.get('C'), 'g', src, 'LR: inverse L2 penalty C')
    B.add('lr_n_channels', len(lr.get('channels', [])), D, src, 'LR: input channels (chiral gravity frame)')
    B.add('n_bins', pan.get('n_bins'), D, src, 'time bins of the LR and GBM features')
    B.add('n_channels', pan.get('n_channels'), D, src, 'standardised channels')
    gp = gbm.get('gbm_params', {})
    B.add('gbm_n_estimators', gp.get('n_estimators'), D, src, 'GBM: trees')
    B.add('gbm_learning_rate', gp.get('learning_rate'), 'g', src, 'GBM: learning rate')
    B.add('gbm_num_leaves', gp.get('num_leaves'), D, src, 'GBM: leaves per tree')
    B.add('gbm_subsample', gp.get('subsample'), 'g', src, 'GBM: row subsample')
    B.add('gbm_colsample', gp.get('colsample_bytree'), 'g', src, 'GBM: column subsample')
    B.add('gbm_features_per_channel', pan.get('gbm_features_per_channel'), D, src, 'GBM: summary features per channel')
    B.add('gbm_n_channels', len(gbm.get('channels', [])), D, src, 'GBM: input channels')
    B.add('mr_features', pan.get('mr_features_requested'), D, src, 'MiniRocket: features requested')
    B.add('mr_inner_splits', mr.get('mr_inner_splits'), D, src, 'MR: inner GroupKFold folds (ridge alpha, Platt)')
    B.add('mr_n_channels', len(mr.get('channels', [])), D, src, 'MR: input channels (gravity frame)')
    B.add('leak_sample_folds', pan.get('leak_sample_folds'), D, src, 'variants: folds the leak check re-runs')
    B.add('inner_splits', pan.get('inner_splits'), D, src, 'inner GroupKFold folds of the calibration cross-fitting')
    B.add('cap_per_person', pan.get('cap_per_person'), D, src, 'docs/09 D2 cap of training trials per person')
    B.add('warp_max', pan.get('warp_max'), 'g', src, 'time warp: largest amplitude a')
    B.add('n_boot', S['p1_comparisons'].get('n_boot'), D, 'p1_comparisons', 'person-bootstrap resamples')
    B.add('alpha', S['p1_comparisons'].get('alpha'), 'g', 'p1_comparisons', 'level of the BH-adjusted verdicts')
    B.add('identical_tol', S['p1_comparisons'].get('identical_tol'), 'g', 'p1_comparisons',
          'predictions closer than this everywhere are identical (solver precision)')
    B.add('logloss_tol', S['p1_comparisons'].get('logloss_tolerance'), 'g', 'p1_comparisons',
          'smallest mean per-person log-loss difference that can carry a verdict')
    B.add('person_threshold', LB.PERSON_ACC_THRESHOLD, '.0%', 'stage2/lanes/p1_labels.py', 'per-person accuracy threshold of the '
          '"persons below" counts')
    B.add('min_persons_ci', S['p1_subsets'].get('min_persons_ci'), D, 'p1_subsets',
          'fewest persons for a person-bootstrap CI')
    B.add('ece_bins', C.ECE_BINS, D, 'stage2/config.py', 'equal-mass bins of the ECE and the reliability diagrams')
    B.add('lowpass_hz', C.LOWPASS_HZ, 'g', 'stage2/config.py', 'current low-pass cut-off (Hz)')
    B.add('lowpass_order', C.LOWPASS_ORDER, D, 'stage2/config.py', 'current Butterworth order')
    B.add('fs_hz', C.FS_HZ, 'g', 'stage2/config.py', 'stream rate (Hz)')
    B.add('max_gap_ms', C.MAX_GAP_MS, 'g', 'stage2/config.py', 'longest gap bridged by interpolation (ms)')
    B.add('gap_window_pad_s', C.GAP_WINDOW_PAD_S, 'g', 'stage2/config.py', 'margin around the movement in which rule E4d looks for gaps (s)')
    B.add('n_norm', C.N_NORM, D, 'stage2/config.py', 'samples of a time-normalised window')
    B.add('k_max', max(C.K_ATTEMPTS), D, 'stage2/config.py', 'largest number of attempts aggregated')
    B.add('min_duration_ms', C.MIN_DURATION_MS, 'g', 'stage2/config.py', 'E4b: shortest movement kept (ms)')
    B.add('min_peak_dps', C.MIN_PEAK_GYRO_DPS, 'g', 'stage2/config.py', 'E4c: smallest peak rotation kept (deg/s)')
    B.add('tight_factor', LB.TIGHT_FACTOR, 'g', 'stage2/lanes/p1_labels.py', 'factor of the tightened E4b/E4c thresholds')
    B.add('winsor_pct', 100 * LB.WINSOR_Q, 'g', 'stage2/lanes/p1_labels.py', 'winsorising quantile (%)')
    cs = S['p1_calibration'].get('settings', {})
    B.add('n_fixed_calibrators', len(cs.get('fixed_methods', [])), D, 'p1_calibration',
          'fixed calibrators compared on the outer folds (exploratory)')
    B.add('n_auto_candidates', len(cs.get('auto_candidates', [])), D, 'p1_calibration',
          'candidates of the nested calibrator choice (none included)')
    B.add('auto_splits', cs.get('auto_splits'), D, 'p1_calibration', 'GroupKFold folds of the nested calibrator choice')
    B.add('ece_null_sims', cs.get('ece_null_sims'), D, 'p1_calibration', 'simulations of the perfect-calibration ECE')
    B.add('n_boot_strata', cs.get('n_boot_strata'), D, 'p1_calibration', 'bootstrap resamples of the aggregates')


def _runs(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_runs'
    runs = S['p1_runs']['runs']
    for r in runs:
        k = rkey(r['name'])
        B.triple(f'acc_{k}', r.get('accuracy'), r.get('accuracy_lo'), r.get('accuracy_hi'), PCT, src,
                 f"{r['name']}: LOSO accuracy")
        B.add(f'n_{k}', r.get('n'), D, src, f"{r['name']}: trials scored")
    rb = {r['name']: r for r in runs}
    base = [rb[f'p1_{m}_base'] for m in LB.MODELS if f'p1_{m}_base' in rb]
    B.add('n_experiments', sum(1 for r in runs if r['family'] != 'calibration'), D, src, 'grid experiments (LOSO runs)')
    B.add('n_calibration_runs', sum(1 for r in runs if r['family'] == 'calibration'), D, src, 'calibration runs')
    B.add('n_runs', len(runs), D, src, 'p1 results (LOSO runs)')
    if base:
        B.add('n_persons', max(int(r['n_persons']) for r in base), D, src, 'persons (outer folds)')
        B.add('n_trials', int(base[0]['n']), D, src, 'primary trials')
    hl = S['p1_runs'].get('headline', {})
    for h, name in HEADLINE.items():
        r, blk = rb.get(name), hl.get(name)
        if r is None or blk is None:
            continue
        what = f'{name}'
        B.triple(f'{h}_accuracy', blk['accuracy'], blk['accuracy_lo'], blk['accuracy_hi'], PCT, src, f'{what}: accuracy')
        B.triple(f'{h}_auc', blk['roc_auc'], blk['roc_auc_lo'], blk['roc_auc_hi'], F3, src, f'{what}: ROC AUC')
        B.triple(f'{h}_log_loss', blk['log_loss'], blk['log_loss_lo'], blk['log_loss_hi'], F3, src, f'{what}: log loss')
        B.triple(f'{h}_brier', blk['brier'], blk['brier_lo'], blk['brier_hi'], F3, src, f'{what}: Brier score')
        B.add(f'{h}_ece', blk['ece'], F3, src, f'{what}: ECE ({C.ECE_BINS} equal-mass bins)')
        B.add(f'{h}_ece_null_q95', blk['ece_null_q95'], F3, src, f'{what}: 95th percentile of the ECE under perfect '
              'calibration (same predictions and bins)')
        B.add(f'{h}_ece_null_median', blk['ece_null_median'], F3, src, f'{what}: median ECE under perfect calibration')
        B.add(f'{h}_ece_p', blk['ece_p'], PV, src, f'{what}: p of the observed ECE under perfect calibration')
        B.triple(f'{h}_slope', blk['slope'], blk['slope_lo'], blk['slope_hi'], F2, src, f'{what}: calibration slope')
        B.triple(f'{h}_citl', blk['citl'], blk['citl_lo'], blk['citl_hi'], F2, src,
                 f'{what}: calibration-in-the-large (logit offset)')
        B.add(f'{h}_slope_below1', blk['p_slope_below_1'], '.0%', src, f'{what}: bootstrap share of slopes below 1')
        B.add(f'{h}_n_unique', blk.get('n_unique'), D, src, f'{what}: distinct P(right) values')
        B.triple(f'{h}_acc_k3', r.get('acc_k3'), r.get('acc_k3_lo'), r.get('acc_k3_hi'), PCT, src,
                 f'{what}: accuracy of the mean P(right) of 3 attempts (harness k-attempt curve)')
        B.add(f'{h}_k_persons', r.get('k_n_persons'), D, src, f'{what}: persons of the k-attempt curve')
        B.add(f'{h}_nat_n', r.get('natural_use_n'), D, src, f'{what}: natural-use trials')
        B.add(f'{h}_nat_n_right', r.get('natural_use_n_right'), D, src, f'{what}: natural-use trials, right hand')
        B.add(f'{h}_nat_ties', r.get('natural_use_n_ties'), D, src, f'{what}: natural-use P(right) exactly 0.5')
        B.add(f'{h}_nat_coverage', r.get('natural_use_coverage'), '.0%', src, f'{what}: natural use, share answered')
        B.add(f'{h}_nat_answered', r.get('natural_use_answered_accuracy'), '.0%', src,
              f'{what}: natural use, accuracy of the answered trials')
        B.triple(f'{h}_nat_half', r.get('natural_use_half_credit'), r.get('natural_use_half_credit_lo'),
                 r.get('natural_use_half_credit_hi'), '.0%', src, f'{what}: natural use, half credit for a tie')
        B.add(f'{h}_nat_harness', r.get('natural_use_harness_accuracy'), '.0%', src,
              f"{what}: natural use, forced choice (P >= 0.5 counts as right: the harness's rule until 2026-10-28)")
        B.add(f'{h}_person_median', r.get('person_acc_median'), PCT, src, f'{what}: median per-person accuracy')
        B.add(f'{h}_person_min', r.get('person_acc_min'), PCT, src, f'{what}: lowest per-person accuracy')
        B.add(f'{h}_persons_below_80', r.get('n_persons_below_80'), D, src, f'{what}: persons below 80 % accuracy')
        B.add(f'{h}_n_configs_tried', r.get('n_configs_tried'), D, src, f'{what}: n_configs_tried')
        B.add(f'{h}_leak', r.get('leak_check'), TXT, src, f'{what}: leak check')


def _comparison(B: Builder, prefix: str, c: dict[str, Any], what: str) -> None:
    src = 'p1_comparisons'
    B.triple(f'd_{prefix}', c.get('d_acc'), c.get('d_acc_lo'), c.get('d_acc_hi'), PTS, src,
             f'{what}: pooled accuracy difference', points=True)
    B.add(f'dll_{prefix}', c.get('d_logloss'), '+.3f', src, f'{what}: mean per-person log-loss difference')
    B.add(f'dll_{prefix}_lo', c.get('d_logloss_lo'), '+.3f', src, f'{what}: log-loss difference, 95% CI low')
    B.add(f'dll_{prefix}_hi', c.get('d_logloss_hi'), '+.3f', src, f'{what}: log-loss difference, 95% CI high')
    B.add(f'p_{prefix}', c.get('p_sign_flip'), PV, src, f'{what}: exact sign-flip p over persons (accuracy)')
    B.add(f'q_{prefix}', c.get('q_bh'), PV, src, f"{what}: BH q in the family '{c.get('mult_family')}'")
    B.add(f'holm_{prefix}', c.get('p_holm'), PV, src, f"{what}: Holm-adjusted p in the family '{c.get('mult_family')}'")
    B.add(f'v_{prefix}', c.get('verdict'), TXT, src, f'{what}: accuracy verdict')
    B.add(f'vll_{prefix}', c.get('logloss_verdict'), TXT, src, f'{what}: log-loss verdict')
    B.add(f'maxdp_{prefix}', c.get('max_abs_dp'), PV, src, f'{what}: largest |change of P(right)|')
    B.add(f'nhc_{prefix}', c.get('n_hand_changed'), D, src, f'{what}: trials whose predicted hand changes')


def _comparisons(B: Builder, S: dict[str, Any]) -> None:
    cj = S['p1_comparisons']
    for c in cj['comparisons']:
        _comparison(B, rkey(c['name']), c, f"{c['name']} vs {c['baseline']}")
    for kind, rows in (('rem', cj.get('remedies', [])), ('ctl', cj.get('controls', []))):
        for c in rows:
            _comparison(B, f"{kind}_{rkey(c['name'])}", c, f"{c['name']} vs {c['baseline']}")
    src = 'p1_comparisons'
    for fam, f in cj.get('families', {}).items():
        B.add(f'fam_{fam}_size', f.get('size'), D, src, f'comparisons in the multiplicity family {fam}')
    opts = [c for c in cj['comparisons'] if c.get('mult_family') == 'options']
    for v in ('worse', 'better', 'nominally worse', 'nominally better', 'identical', 'no difference'):
        B.add(f'n_options_{slug(v)}', sum(c['verdict'] == v for c in opts), D, src, f"options with verdict '{v}'")
    B.add('n_options_ci_excludes_0', sum(c['verdict'] in ('worse', 'better', 'nominally worse', 'nominally better')
                                         for c in opts), D, src, 'options whose paired accuracy CI excludes 0')
    B.add('n_options_holm', sum((c.get('p_holm') or 1) < (cj.get('alpha') or 0.05) for c in opts), D, src,
          'options with a Holm-adjusted p below alpha')
    B.add('expected_false_ci', (cj.get('alpha') or 0.05) * len(opts), '.0f', src,
          'CIs expected to exclude 0 by chance alone among the options (alpha x comparisons)')


def _subsets(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_subsets'
    for r in S['p1_subsets']['rows']:
        k = f"sub_{r['key']}"
        B.triple(k, r.get('accuracy'), r.get('lo'), r.get('hi'), PCT, src, f"{r['model']}: accuracy on {r['label']}")
        B.add(f'{k}_n', r.get('n'), D, src, f"{r['model']}: trials, {r['label']}")
        B.add(f'{k}_np', r.get('n_persons'), D, src, f"{r['model']}: persons, {r['label']}")
        B.add(f'{k}_status', r.get('status'), TXT, src, f"{r['model']}: {r['label']} against the primary accuracy")
        if r.get('kind') == 'platform':
            B.add(f'{k}_ntrain', r.get('n_train_median'), '.0f', src,
                  f"{r['model']}: median training trials per fold, {r['label']} ({r.get('platform')})")
    gaps = S['p1_subsets'].get('e4d_gaps', [])
    B.add('e4d_n', len(gaps), D, src, 'E4d trials (instructed hand)')
    B.add('e4d_n_inapp', sum(bool(g['gap_in_approach']) for g in gaps), D, src,
          'E4d trials whose unbridged gap falls inside the approach')
    B.add('e4d_n_first_half', sum(bool(g['gap_in_first_half']) for g in gaps), D, src,
          'E4d trials whose unbridged gap falls in the first half of the approach')
    B.add('e4d_n_margin', sum(not g['gap_in_approach'] for g in gaps), D, src,
          'E4d trials whose gap lies only in the 0.5-s margins of rule E4d')
    B.add('e4d_n_persons', len({g['person_id'] for g in gaps}), D, src, 'persons with E4d trials')
    B.add('e4d_n_persons_inapp', len({g['person_id'] for g in gaps if g['gap_in_approach']}), D, src,
          'persons with an E4d trial gapped inside the approach')
    pf = {}
    for g in gaps:
        pf[g['platform']] = pf.get(g['platform'], 0) + 1
    for p, n in sorted(pf.items()):
        B.add(f'e4d_n_{slug(p)}', n, D, src, f'E4d trials on {p}')
    inapp = [g['longest_gap_ms'] for g in gaps if g['gap_in_approach']]
    if inapp:
        inapp = sorted(inapp)
        B.add('e4d_inapp_gap_median_ms', inapp[len(inapp) // 2] if len(inapp) % 2 else
              0.5 * (inapp[len(inapp) // 2 - 1] + inapp[len(inapp) // 2]), '.0f', src,
              'median longest unbridged gap of the in-approach E4d trials (ms)')


def _best(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_best'
    b = S['p1_best']
    B.add('best_model', b.get('best_model'), TXT, src, 'panel model with the highest base accuracy (exploratory)')
    B.add('best_label', b.get('label'), TXT, src, 'its label')
    B.add('best_n_configs_tried', b.get('n_configs_tried'), D, src, 'configurations the best-model designation chose among')
    for p in b.get('platforms', []):
        k = f"best_{slug(p['platform'])}"
        B.add(f'{k}_n', p['n'], D, src, f"best model on {p['platform']}: trials")
        B.add(f'{k}_np', p['n_persons'], D, src, f"best model on {p['platform']}: persons")
        B.triple(f'{k}_accuracy', p['accuracy'], p['accuracy_lo'], p['accuracy_hi'], PCT, src,
                 f"best model on {p['platform']}: accuracy")
        B.triple(f'{k}_auc', p['roc_auc'], p['roc_auc_lo'], p['roc_auc_hi'], F3, src, f"best model on {p['platform']}: AUC")
        B.triple(f'{k}_log_loss', p['log_loss'], p['log_loss_lo'], p['log_loss_hi'], F3, src,
                 f"best model on {p['platform']}: log loss")
        B.triple(f'{k}_brier', p['brier'], p['brier_lo'], p['brier_hi'], F3, src, f"best model on {p['platform']}: Brier")
        B.add(f'{k}_ece', p['ece'], F3, src, f"best model on {p['platform']}: ECE")
        B.add(f'{k}_ece_null_q95', p['ece_null_q95'], F3, src, f"best model on {p['platform']}: ECE null 95th percentile")
        B.add(f'{k}_ece_p', p['ece_p'], PV, src, f"best model on {p['platform']}: ECE null p")
        B.triple(f'{k}_slope', p['slope'], p['slope_lo'], p['slope_hi'], F2, src,
                 f"best model on {p['platform']}: calibration slope")
    for c in b.get('conditions', []):
        k = f"cond_{slug(c['condition'])}_{slug(c['level'])}"
        B.triple(k, c['accuracy'], c['lo'], c['hi'], PCT, src, f"best model, {c['condition']} = {c['level']}: accuracy")
        B.add(f'{k}_n', c['n'], D, src, f"best model, {c['condition']} = {c['level']}: trials")
        B.add(f'{k}_np', c['n_persons'], D, src, f"best model, {c['condition']} = {c['level']}: persons")
    B.add('best_person_median', b.get('person_acc_median'), PCT, src, 'best model: median per-person accuracy')
    B.add('best_person_min', b.get('person_acc_min'), PCT, src, 'best model: lowest per-person accuracy')
    B.add('best_persons_below_80', b.get('n_persons_below_80'), D, src, 'best model: persons below 80 % accuracy')
    for g in b.get('grip_balance', []):
        for h in ('left', 'right'):
            if h in g:
                B.add(f"grip_{slug(g['grip'])}_{h}", g[h], D, src, f"primary trials with grip {g['grip']}, {h} hand")
    B.add('grip_chi2_p', b.get('grip_balance_chi2_p'), F2, src, 'chi-square p of instructed grip by hand')


def _calibration(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_calibration'
    cal = S['p1_calibration']
    for key, blk in cal.get('methods', {}).items():
        m, meth = key.split(':')
        for part, o in blk.items():
            k = f'cal_{m}_{meth}' + ('' if part == 'overall' else f'_{slug(part)}')
            what = f'{m} + {meth}' + ('' if part == 'overall' else f' on {part}')
            B.add(f'{k}_n', o['n'], D, src, f'{what}: trials')
            B.add(f'{k}_np', o['n_persons'], D, src, f'{what}: persons')
            B.triple(f'{k}_accuracy', o['accuracy'], o['accuracy_lo'], o['accuracy_hi'], PCT, src, f'{what}: accuracy')
            B.triple(f'{k}_auc', o['roc_auc'], o['roc_auc_lo'], o['roc_auc_hi'], F3, src, f'{what}: AUC')
            B.triple(f'{k}_log_loss', o['log_loss'], o['log_loss_lo'], o['log_loss_hi'], F3, src, f'{what}: log loss')
            B.triple(f'{k}_brier', o['brier'], o['brier_lo'], o['brier_hi'], F3, src, f'{what}: Brier score')
            B.add(f'{k}_ece', o['ece'], F3, src, f'{what}: ECE')
            B.add(f'{k}_ece_null_q95', o['ece_null_q95'], F3, src, f'{what}: ECE null 95th percentile')
            B.add(f'{k}_ece_p', o['ece_p'], PV, src, f'{what}: ECE null p')
            B.triple(f'{k}_slope', o['slope'], o['slope_lo'], o['slope_hi'], F2, src, f'{what}: calibration slope')
            B.triple(f'{k}_citl', o['citl'], o['citl_lo'], o['citl_hi'], F2, src, f'{what}: calibration-in-the-large')
            B.add(f'{k}_n_unique', o.get('n_unique'), D, src, f'{what}: distinct P(right) values')
    for key, pr in cal.get('paired', {}).items():
        m, meth = key.split(':')
        for met in ('log_loss', 'brier', 'roc_auc', 'slope', 'accuracy'):
            x = pr.get(met)
            if not x:
                continue
            k = f'calpd_{m}_{meth}_{met}'
            fmt = '+.1f' if met == 'accuracy' else ('+.2f' if met == 'slope' else '+.3f')
            sc = 100.0 if met == 'accuracy' else 1.0
            f = (lambda v: None if v is None else sc * v)
            what = f'{m} + {meth} minus uncalibrated: pooled {met}' + (' (points)' if met == 'accuracy' else '')
            B.add(k, f(x.get('diff')), fmt, src, what)
            B.add(f'{k}_lo', f(x.get('lo')), fmt, src, f'{what}: 95% CI low (paired person bootstrap)')
            B.add(f'{k}_hi', f(x.get('hi')), fmt, src, f'{what}: 95% CI high')
            B.add(f'{k}_below0', x.get('share_below_0'), '.0%', src, f'{what}: bootstrap share below 0')
    for key, nat in cal.get('natural_use', {}).items():
        m, meth = key.split(':')
        k = f'calnat_{m}_{meth}'
        B.add(f'{k}_coverage', nat.get('coverage'), '.0%', src, f'{m} + {meth}: natural use, share answered')
        B.add(f'{k}_answered', nat.get('answered_accuracy'), '.0%', src, f'{m} + {meth}: natural use, answered accuracy')
        B.add(f'{k}_half', nat.get('half_credit'), '.0%', src, f'{m} + {meth}: natural use, half credit')
        B.add(f'{k}_ties', nat.get('n_ties'), D, src, f'{m} + {meth}: natural-use P(right) exactly 0.5')
    for m, ch in cal.get('auto_choices', {}).items():
        for meth, cnt in ch.get('counts', {}).items():
            B.add(f'cal_{m}_auto_folds_{slug(meth)}', cnt, D, src, f'outer folds whose nested choice was {meth} ({m})')
        B.add(f'cal_{m}_auto_n_folds', sum(ch.get('counts', {}).values()), D, src, f'outer folds ({m})')
    for m, t in cal.get('temperatures', {}).items():
        B.add(f'cal_{m}_t_n_folds', t.get('n_folds'), D, src, f'{m}: folds with a training-fold temperature')
        B.add(f'cal_{m}_t_above_1', t.get('n_T_above_1'), D, src, f'{m}: folds whose training-fold temperature T > 1')
        for s in ('min', 'max', 'median'):
            B.add(f'cal_{m}_t_{s}', t.get(f'T_{s}'), F2, src, f'{m}: {s} of the training-fold temperatures T')
        for k in [x for x in t if x.startswith('T_sum_k') and x.endswith('_median')]:
            kk = k[len('T_sum_k'):-len('_median')]
            B.add(f'cal_{m}_t_sum_k{kk}', t[k], F2, src, f'{m}: median T_k of the summed logits of {kk} attempts')
    for t in cal.get('tests', []):
        k = f"caltest_{t['model']}_{t['kind']}"
        B.add(f'{k}_p', t.get('p'), PV, src, t['test'])
        B.add(f'{k}_q', t.get('q_bh'), PV, src, t['test'] + ': BH q in the calibration test family')
        B.add(f'{k}_holm', t.get('p_holm'), PV, src, t['test'] + ': Holm-adjusted p')
    B.add('caltest_family_size', len(cal.get('tests', [])), D, src, 'tests in the calibration family')
    for m, a in cal.get('aggregates', {}).items():
        B.add(f'agg_{m}_n_persons', a.get('n_persons'), D, src, f'{m}: persons of the aggregates (same at every k)')
        B.add(f'agg_{m}_n_left_out', len(a.get('persons_left_out', [])), D, src,
              f'{m}: persons left out of the aggregates (fewer than k_max trials of a hand)')
        B.add(f'agg_{m}_n_draws', a.get('n_draws'), D, src, f'{m}: random k-subsets per person and hand')
        for r in a.get('rows', []):
            k = f"agg_{m}_k{r['k']}_{slug(r['rule'])}"
            what = f"{m}, {r['k']} attempts, {r['rule']}"
            B.add(f'{k}_accuracy', r['accuracy'], PCT, src, f'{what}: accuracy (cell-weighted)')
            B.triple(f'{k}_log_loss', r['log_loss'], r['log_loss_lo'], r['log_loss_hi'], F3, src, f'{what}: log loss')
            B.triple(f'{k}_brier', r['brier'], r['brier_lo'], r['brier_hi'], F3, src, f'{what}: Brier score')
            B.add(f'{k}_ece', r['ece'], F3, src, f'{what}: ECE (no CI)')
            B.triple(f'{k}_slope', r['slope'], r['slope_lo'], r['slope_hi'], F2, src, f'{what}: calibration slope')
        for r in a.get('paired', []):
            k = f"aggd_{m}_k{r['k']}_{slug(r['against'])}"
            what = f"{m}, {r['k']} attempts: {r['rule']} minus {r['against']}"
            B.add(f'{k}_ll', r['d_log_loss'], '+.3f', src, f'{what}: log loss')
            B.add(f'{k}_ll_lo', r['d_log_loss_lo'], '+.3f', src, f'{what}: log loss, 95% CI low')
            B.add(f'{k}_ll_hi', r['d_log_loss_hi'], '+.3f', src, f'{what}: log loss, 95% CI high')
            B.add(f'{k}_brier', r['d_brier'], '+.3f', src, f'{what}: Brier')
            B.add(f'{k}_brier_lo', r['d_brier_lo'], '+.3f', src, f'{what}: Brier, 95% CI low')
            B.add(f'{k}_brier_hi', r['d_brier_hi'], '+.3f', src, f'{what}: Brier, 95% CI high')
            B.add(f'{k}_p', r.get('p_boot_log_loss'), PV, src, f'{what}: two-sided bootstrap p of the log-loss difference')


def _decisions(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_decisions'
    rows = S['p1_decisions'].get('rows', [])
    B.add('n_choices', len(rows), D, src, 'data-treatment choices judged')
    B.add('n_choices_kept', sum(r['decision'] == 'keep the default' for r in rows), D, src,
          'choices whose default the pre-specified rule keeps')
    B.add('n_choices_changed', sum(r['decision'] != 'keep the default' for r in rows), D, src,
          'choices the pre-specified rule would change')
    for r in rows:
        k = f"dec_{r['family']}"
        B.add(f'{k}_decision', r['decision'], TXT, src, f"decision on {r['family']}")
        B.add(f'{k}_n_options', r['n_options'], D, src, f"options tested for {r['family']}")
        B.add(f'{k}_n_comparisons', r['n_comparisons'], D, src, f"option x model comparisons for {r['family']}")
        B.add(f'{k}_n_single', len(r.get('single_model_options', [])), D, src,
              f"options of {r['family']} run on one panel model (not decidable by the rule)")
        B.pts(f'{k}_max_gain_hi', r['max_gain']['d_acc_hi'], src,
              f"{r['family']}: largest upper 95% bound of an option's accuracy difference (gain forgone at most)")
        B.pts(f'{k}_max_loss_lo', r['max_loss']['d_acc_lo'], src,
              f"{r['family']}: lowest lower 95% bound of an option's accuracy difference (largest loss compatible)")


def _delay(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_filter_delay'
    for c in S['p1_filter_delay'].get('curves', []):
        B.add(f"delay_{c['cutoff_hz']:g}hz_o{c['order']}_ms".replace('.', '_'), c['delay_ms_at_1hz'], '.0f', src,
              f"group delay at 1 Hz of a causal Butterworth low-pass, {c['cutoff_hz']:g} Hz, order {c['order']} (ms)")
        if c['cutoff_hz'] == C.LOWPASS_HZ and c['order'] == C.LOWPASS_ORDER:
            B.add('causal_delay_ms', c['delay_ms_at_1hz'], '.0f', src,
                  'group delay at 1 Hz of the causal version of the current low-pass (ms)')


def _variants(B: Builder, S: dict[str, Any]) -> None:
    src = 'p1_variants'
    v = S['p1_variants']
    fid = v.get('fidelity', {})
    B.add('variant_fidelity_max_diff', fid.get('max_abs_diff'), 'g', src,
          'max |difference| of the rebuilt default variant against the core cache')
    B.add('variant_fidelity_n_trials', fid.get('n_trials'), D, src, 'trials of the fidelity check')
    B.add('n_variants', len(v.get('variants', [])), D, src, 'variant caches (the default one included)')
    for r in v.get('array_diffs', []):
        if not r.get('comparable'):
            continue
        k = f"vdiff_{slug(r['variant'])}"
        B.add(f'{k}_n_differing', r['n_trials_differing'], D, src, f"{r['variant']}: primary trials whose arrays differ")
        B.add(f'{k}_n_ios', r['n_ios_differing'], D, src, f"{r['variant']}: iOS trials whose arrays differ")
        B.add(f'{k}_max', r['max_abs_diff'], '.2g', src, f"{r['variant']}: max |difference| from the default arrays")


def build(S: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Every p1 number, from the derived files (``S``: the loaded files, read when None)."""
    S = load() if S is None else S
    B = Builder()
    for part in (_settings, _runs, _comparisons, _subsets, _best, _calibration, _decisions, _delay, _variants):
        part(B, S)
    return B.n


MARK_OPEN, MARK_CLOSE = '\x02', '\x03'


class Numbers:
    """Formats numbers entries for printing: ``N('gbm_accuracy')`` -> '85.6%'. A missing key raises KeyError, so
    a text that prints a number the numbers file lacks fails to build. ``N.used`` lists the keys printed.
    ``mark=True`` wraps every printed value in MARK_OPEN / MARK_CLOSE, so a test can strip them and prove that no
    other number is left in a text."""

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None, mark: bool = False) -> None:
        self.e = build() if entries is None else entries
        self.used: set[str] = set()
        self.mark = mark

    def __contains__(self, key: str) -> bool:
        return key in self.e

    def value(self, key: str) -> Any:
        self.used.add(key)
        return self.e[key]['value']

    def __call__(self, key: str) -> str:
        e = self.e[key]
        self.used.add(key)
        s = RX.format_value(e['value'], e['format'])
        return f'{MARK_OPEN}{s}{MARK_CLOSE}' if self.mark else s

    def ci(self, key: str) -> str:
        """'[lo, hi]' of an entry with _lo / _hi companions ('' when either is missing)."""
        lo, hi = self.e.get(f'{key}_lo'), self.e.get(f'{key}_hi')
        if lo is None or hi is None or lo['value'] is None or hi['value'] is None:
            return ''
        return f'[{self(key + "_lo")}, {self(key + "_hi")}]'

    def est(self, key: str) -> str:
        """'value [lo, hi]'."""
        c = self.ci(key)
        return self(key) + (f' {c}' if c else '')

    def ok(self, key: str) -> bool:
        return key in self.e and self.e[key]['value'] is not None


def write() -> None:
    L.write_numbers(build())


__all__ = ['build', 'Numbers', 'rkey', 'slug', 'write', 'HEADLINE']
