"""Lane p1: everything the figures, tables and numbers read, derived from the SAVED results only (no fitting).

``derive()`` writes into ``tables/p1/`` (with ``Lane.write_json``: provenance embedded):

* ``p1_runs.json``        one row per experiment: metrics with person-bootstrap 95 % CIs (none for the ECE: its
  perfect-calibration null band instead), calibration slope, natural use (coverage, answered, half credit), k = 3;
* ``p1_comparisons.json`` every variant paired with its model's base run (:func:`p1_stats.paired_compare`, equal to
  ``cv.paired_compare``), with its ROLE (:mod:`p1_labels`), its multiplicity FAMILY and BH / Holm adjusted p-values,
  the largest prediction change, and the verdict; plus the ``remedies`` (remedy against problem, same trials) and
  the ``controls`` (an option against its size-matched control);
* ``p1_subsets.json``     accuracy on the trials an exclusion rule removes (E4d split by where its gap falls; E4b/E4c
  scored by the base model and by a model trained with them), on flagged and swap-signature trials, per test
  platform under every training regime; no CI for a stratum of fewer than ``MIN_PERSONS_CI`` persons;
* ``p1_best.json``        breakdowns of the panel model with the highest base accuracy: an EXPLORATORY designation
  (the best of three pre-specified models on the outer folds, ``n_configs_tried = 3`` for that choice);
* ``p1_calibration.json`` calibrators overall and per platform (log loss, Brier, AUC with CIs; ECE with its null
  band; calibration slope and intercept with CIs; paired differences against the uncalibrated model), the nested
  choices, the training-fold temperatures, the tests of the calibration family, and the multi-attempt aggregates
  (the same persons at every k; paired log-loss / Brier differences between rules);
* ``p1_decisions.json``   one row per decision family: default, options, evidence (multiplicity-adjusted),
  non-inferiority margins, decidability, decision by the pre-specified rule, rationale;
* ``p1_filter_delay.json`` group delay of the causal low-pass filters;
* ``p1_variants.json``    the variants, the fidelity of the default rebuild, and how much each variant changes the
  arrays of the primary trials (the data behind every 'identical result' explanation).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import p1_experiments as X
from stage2.lanes import p1_labels as LB
from stage2.lanes import p1_stats as ST
from stage2.pipeline import Lane

L = Lane('p1')
N_BOOT = C.N_BOOT
N_BOOT_STRATA = 1000                    # aggregates (8 000 rows per block)
CONDITIONS = ('platform', 'speed', 'orientation', 'variant', 'posture', 'start_position', 'tier')
CALIB_MODELS = LB.CALIB_MODELS
LL_TOL = 1e-3                           # a mean per-person log-loss difference below this is never a verdict
ALPHA = 0.05
#: (A, B, family, what the pair shows): a remedy against the problem it addresses, on the same trials and folds
REMEDY_PAIRS = tuple(
    [(f'p1_{m}_bjit100_trained', f'p1_{m}_bjit100', 'segmentation',
      'training with jittered bound copies vs oracle bounds only, both tested on bounds moved by up to ±100 ms')
     for m in ('lr', 'gbm')]
    + [(f'p1_{m}_e4d_trained', f'p1_{m}_e4d_scored', 'exclusions',
        'E4d trials (imputed) kept in training vs primary-only training, both scored on primary + E4d trials')
       for m in ('lr', 'gbm')])
#: (A, B, family, what the pair shows): an option against its size-matched control (same number of training rows
#: or persons removed at random): separates the option's effect from the smaller training set
CONTROL_PAIRS = tuple(
    [(f'p1_{m}_drop_all_flags', f'p1_{m}_drop_all_flags_sizematch', 'outliers',
      'flagged trials out of training vs as many random trials out (per person and hand)') for m in ('lr', 'gbm')]
    + [(f'p1_{m}_platform_{s}', f'p1_{m}_platform_{s}_sizematch', 'platform',
        f'{"same" if s == "same" else "other"}-platform persons only vs as many random persons (both platforms)')
       for m in ('lr', 'gbm') for s in ('same', 'cross')])
FAMILIES = {
    'options': 'every data-treatment option against the current pipeline (one comparison per option and panel model)',
    'robustness': 'test-time segmentation errors against the oracle bounds (models trained on the current pipeline)',
    'pairs': 'remedy-against-problem pairs, option-against-size-matched-control pairs and the controls against the '
             'current pipeline',
}
AGG_RULES = ('mean_prob', 'mean_logit', 'sum_logit', 'sum_logit_T')
AGG_REF = 'sum_logit_T'
#: the headline rows: the three pre-specified panel models and the nested-calibrated ones
HEADLINE_RUNS = tuple([f'p1_{m}_base' for m in LB.MODELS] + [f'p1_calib_{m}_auto' for m in LB.CALIB_MODELS])


def panel_settings() -> dict[str, Any]:
    """The fixed settings of the panel (from the code that ran), so the presentation steps never import it."""
    from stage2.lanes import p1_panel as PN
    return {'models': {m: PN.describe(m, PN.DEFAULT) for m in LB.MODELS}, 'n_bins': PN.N_BINS,
            'gbm_features_per_channel': int(PN.summary_features(np.zeros((1, 1, C.N_NORM))).shape[1]),
            'mr_features_requested': PN.MR_FEATURES, 'warp_max': PN.WARP_MAX, 'n_channels': len(PN.CHANNELS),
            'leak_sample_folds': C.LEAK_SAMPLE_FOLDS, 'inner_splits': C.INNER_SPLITS,
            'cap_per_person': C.CAP_PER_PERSON}


def clean(v: Any) -> Any:
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v


def _load(name: str) -> cv.ExperimentResult | None:
    p = C.RES_DIR / f'{name}.json'
    return cv.ExperimentResult.load(p) if p.exists() else None


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(ST.correct(y, p)))                 # the harness's rule: a tie earns half credit


def acc_ci(pred: pd.DataFrame, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Accuracy with its person-bootstrap 95 % CI (no CI with fewer than MIN_PERSONS_CI persons)."""
    if not len(pred):
        return {'n': 0, 'n_persons': 0, 'accuracy': None, 'lo': None, 'hi': None, 'ci': False}
    a = _acc(pred['y'], pred['p_right'])
    lo, hi = ST.person_boot_ci(pred[['person_id', 'y', 'p_right']], _acc, n_boot=n_boot)
    return {'n': int(len(pred)), 'n_persons': int(pred['person_id'].nunique()), 'accuracy': a, 'lo': lo, 'hi': hi,
            'ci': lo is not None}


# ───────────────────────────── runs ─────────────────────────────


def fast_metrics(y: Any, p: Any, n_bins: int = C.ECE_BINS) -> dict[str, float]:
    """The metrics of ``cv.metrics`` (same definitions) in plain numpy (test: tests/test_p1.py)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {'accuracy': float(np.mean(ST.correct(y, p))), 'roc_auc': ST._auc(y, p),
            'log_loss': float(np.mean(ST.log_loss_terms(y, p))), 'brier': float(np.mean((p - y) ** 2)),
            'ece': ST.ece(y, p, n_bins)}


BLOCK_METRICS = ('accuracy', 'roc_auc', 'log_loss', 'brier', 'ece')
CI_METRICS = ('accuracy', 'roc_auc', 'log_loss', 'brier')          # never a percentile CI for the ECE


def fast_person_bootstrap(pred: pd.DataFrame, n_boot: int = N_BOOT, seed: int = C.SEED) -> dict[str, tuple[float, float]]:
    """``cv.person_bootstrap`` of :data:`BLOCK_METRICS` with the same resampling sequence (test: equal)."""
    persons = pred['person_id'].astype(str).to_numpy()
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    y = pred['y'].to_numpy().astype(int)
    p = pred['p_right'].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    stats: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([rows[u] for u in pick])
        for k, v in fast_metrics(y[sel], p[sel]).items():
            if math.isfinite(v):
                stats.setdefault(k, []).append(v)
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in stats.items()}


def metric_block(pred: pd.DataFrame, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Metrics of one prediction set: accuracy, AUC, log loss, Brier with person-bootstrap CIs; the ECE with its
    perfect-calibration null band; the calibration slope / intercept with CIs (no CI below MIN_PERSONS_CI)."""
    y = pred['y'].to_numpy().astype(int)
    p = pred['p_right'].to_numpy(dtype=float)
    m = fast_metrics(y, p)
    n_persons = int(pred['person_id'].nunique())
    ci = fast_person_bootstrap(pred, n_boot=n_boot) if n_persons >= LB.MIN_PERSONS_CI else {}
    out: dict[str, Any] = {'n': int(len(pred)), 'n_persons': n_persons}
    for k in CI_METRICS:
        out[k] = m[k]
        out[f'{k}_lo'], out[f'{k}_hi'] = ci.get(k, (None, None))
    en = ST.ece_null(y, p)
    out.update({'ece': m['ece'], 'ece_null_median': en['null_median'], 'ece_null_q95': en['null_q95'],
                'ece_p': en['p_value']})
    out.update(ST.calibration_block(pred[['person_id', 'y', 'p_right']], n_boot=n_boot))
    out['n_ties_half'] = int(np.sum(p == 0.5))
    out['n_unique'] = int(np.unique(p).size)
    return out


def run_row(name: str, r: cv.ExperimentResult, e: X.Exp | None, grid_index: dict[str, int]) -> dict[str, Any]:
    m, ci = r.metrics, r.ci
    pr = r.predictions
    y, p = pr['y'].to_numpy().astype(int), pr['p_right'].to_numpy(dtype=float)
    if e is not None:
        option, family, model = e.option, e.family, e.model
    else:                                   # a calibration run: p1_calib_<model>_<method>
        _, _, model, option = name.split('_', 3)
        family = 'calibration'
    row: dict[str, Any] = {
        'name': name, 'model': model, 'family': family, 'option': option,
        'report_family': LB.report_family(option, family) if e else 'calibration',
        'role': LB.role(option) if e else 'calibration', 'option_label': LB.option_label(option) if e else option,
        'value': e.value if e else None, 'rows': e.rows if e else 'primary', 'grid_index': grid_index.get(name),
        'n': int(m['n']), 'n_persons': int(pr['person_id'].nunique()),
        'n_configs_tried': int(r.info.get('n_configs_tried') or 1), 'leak_check': r.leak_check}
    for k in CI_METRICS:
        row[k] = m.get(k)
        row[f'{k}_lo'], row[f'{k}_hi'] = ci.get(k, (None, None))
    en = ST.ece_null(y, p)
    b, _, a0 = ST.calibration_fit(y, p)
    row.update({'ece': m.get('ece'), 'ece_null_median': en['null_median'], 'ece_null_q95': en['null_q95'],
                'ece_p': en['p_value'], 'cal_slope': b, 'cal_citl': a0})
    tie = p == 0.5
    row['n_ties_half'] = int(tie.sum())
    row['half_credit_accuracy'] = float(np.mean(np.where(tie, 0.5, (p > 0.5).astype(int) == y)))
    pp = r.per_person
    if len(pp):
        acc = pp['accuracy'].to_numpy(float)
        row.update({'person_acc_median': float(np.median(acc)), 'person_acc_min': float(acc.min()),
                    'n_persons_below_80': int(np.sum(acc < LB.PERSON_ACC_THRESHOLD))})
    nat = ST.natural_use(r.extra_predictions)
    row.update({f'natural_use_{k}': v for k, v in nat.items()})
    if r.gated_metrics:
        row['gated_n'] = int(r.gated_metrics.get('n', 0))
        row['gated_accuracy'] = r.gated_metrics.get('accuracy')
    kc = r.k_curve.set_index('k') if len(r.k_curve) else pd.DataFrame()
    for k in (3, 5):
        if k in kc.index:
            row[f'acc_k{k}'], row[f'acc_k{k}_lo'], row[f'acc_k{k}_hi'] = (
                float(kc.loc[k, 'accuracy']), float(kc.loc[k, 'lo']), float(kc.loc[k, 'hi']))
            row['k_n_persons'] = int(kc.loc[k, 'n_persons'])
    row['notes'] = r.info.get('notes', '')
    return row


# ───────────────────────────── comparisons ─────────────────────────────


def compare_row(name: str, base: str, ra: cv.ExperimentResult, rb: cv.ExperimentResult, **meta: Any) -> dict[str, Any]:
    c = ST.paired_compare(ra.predictions, rb.predictions, n_boot=N_BOOT)
    lo, hi = c['pooled_acc_diff_ci95']
    llo, lhi = c['mean_person_logloss_diff_ci95']
    return {'name': name, 'baseline': base, **meta, 'n_shared': c['n_trials'], 'n_persons': c['n_persons'],
            'acc': c['acc_a'], 'acc_base': c['acc_b'], 'd_acc': c['pooled_acc_diff'], 'd_acc_lo': lo, 'd_acc_hi': hi,
            'd_acc_person': c['mean_person_acc_diff'], 'd_acc_person_lo': c['mean_person_acc_diff_ci95'][0],
            'd_acc_person_hi': c['mean_person_acc_diff_ci95'][1], 'p_sign_flip': c['p_sign_flip_acc'],
            'd_logloss': c['mean_person_logloss_diff'], 'd_logloss_lo': llo, 'd_logloss_hi': lhi,
            'p_sign_flip_logloss': c['p_sign_flip_logloss'], 'persons_better': c['persons_a_better'],
            'persons_worse': c['persons_b_better'], 'max_abs_dp': c['max_abs_dp'],
            'n_hand_changed': c['n_hand_changed']}


def verdicts(row: dict[str, Any]) -> dict[str, Any]:
    """Verdict words from the numbers: 'identical' (predictions within solver precision); 'better' / 'worse' when
    the paired 95 % CI of the pooled accuracy difference excludes 0 AND the family's BH q of the sign-flip test is
    below ALPHA; 'nominally better / worse' when only the CI excludes 0; 'no difference' otherwise. The log-loss
    verdict alike (mean per-person difference, lower is better, at least LL_TOL)."""
    q = row.get('q_bh')
    sig = q is not None and q < ALPHA
    if row['max_abs_dp'] is not None and row['max_abs_dp'] < LB.IDENTICAL_TOL:
        v = 'identical'
    elif row['d_acc_lo'] > 0:
        v = 'better' if sig else 'nominally better'
    elif row['d_acc_hi'] < 0:
        v = 'worse' if sig else 'nominally worse'
    else:
        v = 'no difference'
    ql = row.get('q_bh_logloss')
    sigl = ql is not None and ql < ALPHA
    ll = 'no difference'
    if v == 'identical':
        ll = 'identical'
    elif abs(row['d_logloss']) >= LL_TOL:
        if row['d_logloss_hi'] < 0:
            ll = 'better' if sigl else 'nominally better'
        elif row['d_logloss_lo'] > 0:
            ll = 'worse' if sigl else 'nominally worse'
    return {**row, 'verdict': v, 'logloss_verdict': ll, 'nominal': v in ('better', 'worse', 'nominally better',
                                                                          'nominally worse')}


def apply_family(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    """BH q-values and Holm-adjusted p-values within one family (accuracy and log-loss sign-flip tests), then the
    verdicts."""
    q, h = ST.adjust([r['p_sign_flip'] for r in rows])
    ql, hl = ST.adjust([r['p_sign_flip_logloss'] for r in rows])
    out = []
    for r, a, b, c, d in zip(rows, q, h, ql, hl):
        out.append(verdicts({**r, 'mult_family': family, 'family_size': len(rows), 'q_bh': a, 'p_holm': b,
                             'q_bh_logloss': c, 'p_holm_logloss': d}))
    return out


# ───────────────────────────── E4d gaps ─────────────────────────────


def e4d_gaps(d_all: Any) -> pd.DataFrame:
    """Where the long gap of every E4d trial falls (cache stream, ``valid`` False = not bridged): inside the
    approach [onset, end], in its first half, or only in the 0.5-s margins of rule E4d."""
    t = d_all.trials
    sel = np.flatnonzero(((t['excl_rule'] == 'E4d') & (t['hand_source'].astype(str) != 'reported')).to_numpy())
    rows = []
    pad = 1000.0 * C.GAP_WINDOW_PAD_S
    for i in sel:
        s = d_all.stream(int(i))
        tt, ok = s['t'], s['valid'].astype(bool)
        on, en = float(t['onset_ms'].iloc[i]), float(t['end_ms'].iloc[i])
        app = (tt >= on) & (tt <= en)
        first = (tt >= on) & (tt <= 0.5 * (on + en))
        win = (tt >= on - pad) & (tt <= en + pad)
        bad = ~ok & win
        runs, cur = [], 0
        for b in bad:
            cur = cur + 1 if b else 0
            runs.append(cur)
        rows.append({'trial_id': t['trial_id'].iloc[i], 'trial_key': t['trial_key'].iloc[i],
                     'person_id': str(t['person_id'].iloc[i]), 'platform': str(t['platform'].iloc[i]),
                     'gap_in_approach': bool(np.any(~ok & app)), 'gap_in_first_half': bool(np.any(~ok & first)),
                     'invalid_ms_in_window': float(bad.sum() * 1000.0 / C.FS_HZ),
                     'longest_gap_ms': float((max(runs) if runs else 0) * 1000.0 / C.FS_HZ)})
    return pd.DataFrame(rows, columns=['trial_id', 'trial_key', 'person_id', 'platform', 'gap_in_approach',
                                       'gap_in_first_half', 'invalid_ms_in_window', 'longest_gap_ms'])


# ───────────────────────────── subsets ─────────────────────────────


def vs_primary(acc: Any, lo: Any, hi: Any, primary: float) -> str:
    """How a subset's accuracy compares with the primary accuracy of the same model (from its CI)."""
    if acc is None:
        return 'not estimable'
    if lo is None or hi is None:
        return 'too few persons for a CI'
    if hi < primary:
        return 'below the primary accuracy'
    if lo > primary:
        return 'above the primary accuracy'
    if lo <= 0.5:
        return 'inconclusive (CI covers chance and the primary accuracy)'
    return 'compatible with the primary accuracy'


def subsets(res: dict[str, cv.ExperimentResult], trials: pd.DataFrame, gaps: pd.DataFrame) -> list[dict[str, Any]]:
    """Accuracy on the trials a rule or flag concerns, per training regime; CIs need >= MIN_PERSONS_CI persons."""
    from stage2.data import load_cache
    from stage2.scripts.reference import swap_blocks
    out: list[dict[str, Any]] = []
    info = trials.set_index('trial_id')
    swap = set(swap_blocks(load_cache()))
    in_app = set(gaps.loc[gaps['gap_in_approach'], 'trial_id'])

    def add(kind: str, label: str, name: str, pred: pd.DataFrame, key: str, **kw: Any) -> None:
        model = name.split('_')[1]
        a = acc_ci(pred)
        base = res.get(f'p1_{model}_base')
        prim = _acc(base.predictions['y'], base.predictions['p_right']) if base is not None else math.nan
        out.append({'kind': kind, 'label': label, 'name': name, 'model': model, 'key': f'{model}_{key}', **a,
                    'status': vs_primary(a['accuracy'], a['lo'], a['hi'], prim), **kw})

    for m in LB.MODELS:
        base = res.get(f'p1_{m}_base')
        if base is None:
            continue
        pb = base.predictions.join(info[['platform', *LB.FLAGS, 'session_label', 'block_index']], on='trial_id')
        add('base', 'all primary trials', base.name, pb, 'all')
        for f in LB.FLAGS:
            fl = pb[f].astype('boolean').fillna(False).astype(bool)
            add('flag', f'{f}: flagged', base.name, pb[fl], f'flag_{f}', flag=f, flagged=True)
            add('flag', f'{f}: not flagged', base.name, pb[~fl], f'noflag_{f}', flag=f, flagged=False)
        for opt, how, hk in (('e4d_scored', 'linear', 'linear'), ('e4d_scored_cubic', 'cubic', 'cubic'),
                             ('e4d_scored_hold', 'hold', 'hold'), ('e4d_trained', 'linear, E4d in training', 'trained')):
            r = res.get(f'p1_{m}_{opt}')
            if r is None:
                continue
            pr = r.predictions.join(info[['excl_rule']], on='trial_id')
            e4 = pr[pr['excl_rule'] == 'E4d']
            ina = e4['trial_id'].isin(in_app)
            add('excluded', f'E4d trials ({how})', r.name, e4, f'e4d_all_{hk}', rule='E4d', imputation=how, where='all')
            add('excluded', f'E4d, gap inside the approach ({how})', r.name, e4[ina], f'e4d_inapp_{hk}', rule='E4d',
                imputation=how, where='in approach')
            add('excluded', f'E4d, gap only in the margins ({how})', r.name, e4[~ina], f'e4d_margin_{hk}', rule='E4d',
                imputation=how, where='margins only')
        for opt, how, hk in (('excl_scored', 'base model', 'base'), ('excl_loose', 'trained with them', 'trained')):
            r = res.get(f'p1_{m}_{opt}')
            if r is not None:
                pr = r.predictions.join(info[['excl_rule']], on='trial_id')
                add('excluded', f'E4b/E4c trials ({how})', r.name, pr[pr['excl_rule'].isin(['E4b', 'E4c'])],
                    f'e4bc_{hk}', rule='E4b/E4c', training=how)
        dur, peak = info['duration_ms'], info['peak_gyro_dps']
        tight = (dur >= LB.TIGHT_DURATION_MS) & (peak >= LB.TIGHT_PEAK_DPS)
        fail = pb['trial_id'].map(~tight).astype(bool)
        add('threshold', 'trials failing the tightened E4b/E4c thresholds', base.name, pb[fail], 'tight_fail',
            rule='tight')
        add('threshold', 'trials passing the tightened thresholds', base.name, pb[~fail], 'tight_pass', rule='tight')
        r = res.get(f'p1_{m}_excl_tight')
        if r is not None:
            pr = r.predictions
            add('threshold', 'trials failing the tightened thresholds (tight training)', r.name,
                pr[pr['trial_id'].map(~tight).astype(bool)], 'tight_fail_tighttrain', rule='tight')
        if swap:
            key = list(zip(pb['session_label'].astype(str), pd.to_numeric(pb['block_index'], errors='coerce')
                           .fillna(-1).astype(int)))
            ins = np.array([k in swap for k in key])
            add('swap', 'swap-signature blocks', base.name, pb[ins], 'swap')
            add('swap', 'other trials', base.name, pb[~ins], 'swap_other')
            r = res.get(f'p1_{m}_drop_swap_blocks')
            if r is not None:
                rp = r.predictions.set_index('trial_id').loc[pb['trial_id']].reset_index()
                add('swap', 'other trials (swap blocks out of training)', r.name, rp[~ins], 'swap_other_dropped')
        for opt, label, tk in (('base', 'pooled', 'pooled'), ('platform_same', 'same platform only', 'same'),
                               ('platform_same_sizematch', 'random persons, as many as same platform', 'same_ctl'),
                               ('platform_cross', 'other platform only', 'cross'),
                               ('platform_cross_sizematch', 'random persons, as many as other platform', 'cross_ctl')):
            r = res.get(f'p1_{m}_{opt}')
            if r is None:
                continue
            pr = r.predictions.join(info[['platform']], on='trial_id')
            n_train = {int(f): json.loads(s).get('n_train') for f, s in pr.groupby('fold')['params'].first().items()}
            for pf, g in pr.groupby('platform'):
                nt = [n_train[f] for f in sorted(set(g['fold'])) if n_train.get(f) is not None]
                add('platform', label, r.name, g, f'pf_{pf}_{tk}', platform=pf, training=label, training_key=tk,
                    n_train_median=float(np.median(nt)) if nt else None)
    return out


# ───────────────────────────── best model breakdowns ─────────────────────────────


def best_model(runs: pd.DataFrame) -> str:
    """The panel model with the highest base accuracy (log loss breaks ties): an exploratory designation among
    three pre-specified models (n_configs_tried = 3 for the designation)."""
    b = runs[runs['option'] == 'base'].sort_values(['accuracy', 'log_loss'], ascending=[False, True])
    return str(b['model'].iloc[0])


def breakdowns(r: cv.ExperimentResult, trials: pd.DataFrame) -> dict[str, Any]:
    info = trials.set_index('trial_id')
    pr = r.predictions.join(info[list(CONDITIONS)], on='trial_id')
    rows = []
    for c in CONDITIONS:
        for lv, g in pr.groupby(pr[c].astype(str), sort=True):
            rows.append({'condition': c, 'level': lv, **acc_ci(g)})
    plat = [{'platform': pf, **metric_block(g)} for pf, g in pr.groupby('platform')]
    prim = trials[trials['analysis_set'] == 'primary']
    gb = pd.crosstab(prim['instructed_rotation'].astype(str), prim['hand'].astype(str))
    grip = [{'grip': g, **{h: int(gb.loc[g, h]) for h in gb.columns}} for g in gb.index]
    chi = None
    if gb.shape[0] > 1 and gb.shape[1] == 2:
        from scipy.stats import chi2_contingency
        chi = float(chi2_contingency(gb.to_numpy())[1])
    pp = r.per_person
    acc = pp['accuracy'].to_numpy(float)
    return {'name': r.name, 'conditions': rows, 'platforms': plat, 'per_person': pp.to_dict(orient='records'),
            'person_acc_median': float(np.median(acc)), 'person_acc_min': float(acc.min()),
            'n_persons_below_80': int(np.sum(acc < LB.PERSON_ACC_THRESHOLD)), 'n_persons': int(acc.size),
            'grip_balance': grip, 'grip_balance_chi2_p': chi}


# ───────────────────────────── calibration ─────────────────────────────


def reliability_bands(pred: pd.DataFrame, n_bins: int = C.ECE_BINS, n_boot: int = 1000, seed: int = C.SEED
                      ) -> list[dict[str, Any]]:
    """Equal-mass reliability bins of P(right) with person-bootstrap 95 % bands of the observed share in each
    bin (bin edges fixed on the full data; persons resampled with all their trials)."""
    p = pred['p_right'].to_numpy(float)
    y = pred['y'].to_numpy(int)
    persons = pred['person_id'].astype(str).to_numpy()
    k = ST.ece_bins(p, n_bins)
    uniq = np.unique(persons)
    rows_of = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, n_bins), np.nan)
    for b in range(n_boot):
        sel = np.concatenate([rows_of[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        kb, yb = k[sel], y[sel]
        cnt = np.bincount(kb, minlength=n_bins)
        s = np.bincount(kb, yb, minlength=n_bins)
        boots[b] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    out = []
    for i in range(n_bins):
        m = k == i
        if not m.any():
            continue
        v = boots[:, i][np.isfinite(boots[:, i])]
        out.append({'bin': i, 'n': int(m.sum()), 'p_mean': float(p[m].mean()), 'y_mean': float(y[m].mean()),
                    'y_lo': float(np.quantile(v, 0.025)) if v.size else None,
                    'y_hi': float(np.quantile(v, 0.975)) if v.size else None})
    return out


def _boot_p(share_below: float | None) -> float | None:
    """Two-sided bootstrap p-value of a difference from its share of resamples below 0."""
    if share_below is None:
        return None
    return float(min(1.0, 2 * min(share_below, 1 - share_below) + 1.0 / N_BOOT))


def aggregates(r: cv.ExperimentResult, temps_by_fold: dict[int, dict[str, float]], n_draws: int = 200,
               seed: int = C.SEED) -> dict[str, Any]:
    """Calibration of k-attempt aggregates of one person and hand on the SAME persons at every k (those with at
    least max(K_ATTEMPTS) trials of each hand, as the harness's k-attempt curve): mean of P, mean of logits, sum of
    logits and the sum divided by the fold's T_k (fitted on training persons only). Every (person, hand) cell gets
    ``n_draws`` random k-subsets, so cells weigh equally (at k = 1 too, unlike the trial-level metrics). Per rule:
    accuracy, log loss, Brier (person-bootstrap CIs), ECE (no CI), calibration slope (with a person-bootstrap CI at
    k = max(K_ATTEMPTS) only, where the advice is read: each CI costs about a thousand logistic fits on the draws);
    and the paired differences of log loss and Brier against the sum / T_k rule."""
    pr = r.predictions
    z = ST.logit(pr['p_right'].to_numpy(float), clip=1e-9)
    kmax = max(C.K_ATTEMPTS)
    counts = pr.groupby(['person_id', 'y']).size().unstack(fill_value=0)
    eligible = sorted(str(p) for p in counts.index[(counts.get(0, 0) >= kmax) & (counts.get(1, 0) >= kmax)])
    left_out = sorted(set(counts.index.astype(str)) - set(eligible))
    rows, deltas = [], []
    for k in C.K_ATTEMPTS:
        rng = np.random.default_rng([seed, 31, k])
        recs = []
        for (pid, yy), idx in sorted(pr.groupby(['person_id', 'y']).indices.items()):
            if str(pid) not in eligible:
                continue
            fold = int(pr['fold'].iloc[idx[0]])
            T = temps_by_fold.get(fold, {}).get(f'T_sum_k{k}', 1.0) if k > 1 else 1.0
            for _ in range(n_draws):
                zz = z[rng.choice(idx, size=k, replace=False)]
                recs.append((str(pid), int(yy), float(np.mean(ST.sigmoid(zz))), float(ST.sigmoid(zz.mean())),
                             float(ST.sigmoid(zz.sum())), float(ST.sigmoid(zz.sum() / T))))
        df = pd.DataFrame(recs, columns=['person_id', 'y', *AGG_RULES])
        y = df['y'].to_numpy(int)
        uniq, inv = ST.person_index(df['person_id'])
        picks = ST.boot_picks(uniq.size, N_BOOT_STRATA, seed)
        W = np.stack([np.bincount(pk, minlength=uniq.size) for pk in picks]).astype(float)   # (B, persons)
        n_p = np.bincount(inv, minlength=uniq.size).astype(float)
        per: dict[str, Any] = {}
        fits: dict[bytes, tuple[float, float, float, float]] = {}    # rules that coincide (all of them at k = 1)
        for rule in AGG_RULES:
            p = df[rule].to_numpy(float)
            ll_p = np.bincount(inv, ST.log_loss_terms(y, p), minlength=uniq.size)
            br_p = np.bincount(inv, (p - y) ** 2, minlength=uniq.size)
            ac_p = np.bincount(inv, ST.correct(y, p), minlength=uniq.size)
            per[rule] = (ll_p, br_p)
            fk = p.tobytes()
            if fk not in fits:
                b, _, a0 = ST.calibration_fit(y, p)
                if k == kmax:          # the slope CI only where the advice is read (each costs ~10^3 logistic fits)
                    slopes = ST.calibration_fit_many(y, p, W[:, inv])[0]
                    fits[fk] = (b, a0, *ST.q95(slopes.tolist()))
                else:
                    fits[fk] = (b, a0, None, None)
            b, a0, s_lo, s_hi = fits[fk]
            boot_ll = (W @ ll_p) / (W @ n_p)
            boot_br = (W @ br_p) / (W @ n_p)
            rows.append({'k': k, 'rule': rule, 'n': int(len(df)), 'n_persons': int(uniq.size), 'n_cells': int(len(df) // n_draws),
                         'accuracy': float(ac_p.sum() / n_p.sum()), 'log_loss': float(ll_p.sum() / n_p.sum()),
                         'log_loss_lo': ST.q95(boot_ll)[0], 'log_loss_hi': ST.q95(boot_ll)[1],
                         'brier': float(br_p.sum() / n_p.sum()), 'brier_lo': ST.q95(boot_br)[0],
                         'brier_hi': ST.q95(boot_br)[1], 'ece': ST.ece(y, p), 'slope': b, 'slope_lo': s_lo,
                         'slope_hi': s_hi, 'citl': a0})
        ref_ll, ref_br = per[AGG_REF]
        for rule in AGG_RULES:
            if rule == AGG_REF:
                continue
            ll_p, br_p = per[rule]
            dl = (W @ (ref_ll - ll_p)) / (W @ n_p)
            db = (W @ (ref_br - br_p)) / (W @ n_p)
            deltas.append({'k': k, 'rule': AGG_REF, 'against': rule,
                           'd_log_loss': float((ref_ll - ll_p).sum() / n_p.sum()), 'd_log_loss_lo': ST.q95(dl)[0],
                           'd_log_loss_hi': ST.q95(dl)[1], 'd_brier': float((ref_br - br_p).sum() / n_p.sum()),
                           'd_brier_lo': ST.q95(db)[0], 'd_brier_hi': ST.q95(db)[1],
                           'p_boot_log_loss': _boot_p(float(np.mean(dl < 0)))})
    return {'persons': eligible, 'n_persons': len(eligible), 'persons_left_out': left_out, 'n_draws': n_draws,
            'k_max': kmax, 'rows': rows, 'paired': deltas}


def calibration(res: dict[str, cv.ExperimentResult], trials: pd.DataFrame) -> dict[str, Any]:
    """Calibrators overall / per platform, reliability bands, paired differences against the uncalibrated model,
    nested choices, training-fold temperatures, the tests of the calibration family and the aggregates."""
    from stage2.lanes import p1_calibration as K
    from stage2.lanes.p1_calibration import AUTO, METHODS
    info = trials.set_index('trial_id')
    out: dict[str, Any] = {'methods': {}, 'aggregates': {}, 'reliability': {}, 'paired': {}, 'auto_choices': {},
                           'natural_use': {}, 'temperatures': {}, 'tests': [],
                           'settings': {'fixed_methods': list(METHODS), 'auto_candidates': list(K.AUTO_CANDIDATES),
                                        'auto_splits': K.AUTO_SPLITS, 'inner_splits': C.INNER_SPLITS,
                                        'k_agg': list(K.K_AGG), 'n_draws_temperature': K.N_DRAWS,
                                        'n_boot_strata': N_BOOT_STRATA, 'ece_null_sims': ST.N_NULL}}
    for m in CALIB_MODELS:
        base = res.get(f'p1_{m}_base')
        if base is None:
            continue
        auto = res.get(f'p1_calib_{m}_{AUTO}')
        if auto is not None:
            chosen = {int(f): json.loads(ps).get('chosen')
                      for f, ps in auto.predictions.groupby('fold')['params'].first().items()}
            out['auto_choices'][m] = {'per_fold': {str(k): v for k, v in sorted(chosen.items())},
                                      'counts': dict(sorted(Counter(chosen.values()).items())),
                                      'candidates': (auto.params.get('spec') or {}).get('candidates')}
        runs = {'none': base, **{meth: res.get(f'p1_calib_{m}_{meth}') for meth in (*METHODS, AUTO)}}
        for meth, r in runs.items():
            if r is None:
                continue
            key = f'{m}:{meth}'
            pr = r.predictions.join(info[['platform']], on='trial_id')
            block = {'overall': metric_block(pr)}
            for pf, g in pr.groupby('platform'):
                block[pf] = metric_block(g, n_boot=N_BOOT)
            out['methods'][key] = block
            out['natural_use'][key] = ST.natural_use(r.extra_predictions)
            out['reliability'][key] = reliability_bands(pr)
            if meth != 'none':
                d = ST.paired_metric_diff(r.predictions, base.predictions, n_boot=N_BOOT)
                out['paired'][key] = d
        tr = res.get(f'p1_calib_{m}_temperature')
        if tr is not None:
            temps = {}
            for f, ps in tr.predictions.groupby('fold')['params'].first().items():
                dd = json.loads(ps)
                temps[int(f)] = {k: v for k, v in dd.items() if k.startswith('T_sum_k')}
                temps[int(f)]['T'] = dd.get('T')
            Ts = np.array([v['T'] for v in temps.values() if v.get('T') is not None], dtype=float)
            out['temperatures'][m] = {'per_fold': {str(k): v for k, v in temps.items()}, 'n_folds': int(Ts.size),
                                      'n_T_above_1': int(np.sum(Ts > 1)), 'T_min': float(Ts.min()),
                                      'T_max': float(Ts.max()), 'T_median': float(np.median(Ts)),
                                      **{f'{k}_median': float(np.median([v[k] for v in temps.values()
                                                                          if v.get(k) is not None]))
                                         for k in sorted({t for v in temps.values() for t in v if t.startswith('T_sum_k')})}}
            out['aggregates'][m] = aggregates(base, temps)
        o0 = out['methods'][f'{m}:none']['overall']
        out['tests'].append({'test': f'{m}: ECE of the uncalibrated model against its perfect-calibration null',
                             'model': m, 'kind': 'ece_null', 'p': o0['ece_p']})
        pa = out['paired'].get(f'{m}:{AUTO}')
        if pa:
            out['tests'].append({'test': f'{m}: calibration-slope change, nested calibrator vs uncalibrated',
                                 'model': m, 'kind': 'slope_change', 'p': _boot_p(pa['slope']['share_below_0'])})
    q, h = ST.adjust([t['p'] for t in out['tests']])
    for t, a, b in zip(out['tests'], q, h):
        t.update({'q_bh': a, 'p_holm': b, 'family_size': len(out['tests'])})
    out['test_family'] = ('the miscalibration tests of the calibrated panel models (ECE against its null) and the '
                          'slope changes of their nested calibrators')
    return out


# ───────────────────────────── decisions ─────────────────────────────

RATIONALE = {
    'filter': 'the predictor processes a recorded buffer, so zero phase is available in production; a causal '
              'filter would only add a delay (group delay, P1_10)',
    'gaps': 'gap handling only matters for the few trials with long gaps; their accuracy is reported apart',
    'sampling': 'what matters is robustness: a lower rate or platform rounding must not break the model',
    'gravity': 'the gravity estimate defines the chiral channels; the browser fusion is platform-dependent',
    'frame': ('per model: LR uses the chiral gravity-frame channels (the hand is a rotation about gravity, '
              'mirror-odd); MR the {mr_n_channels} grip-invariant gravity-frame channels; GBM all {gbm_n_channels}, '
              'because the device-frame channels add information and cannot act as a grip-to-hand shortcut: the '
              'instructed grips are balanced across hands (chi-square p = {grip_p})'),
    'window': 'production must segment the approach; windows that need more context are harder to deploy',
    'normalisation': 'none keeps physical units (speed and amplitude); scalers fitted on the mirror-augmented '
                     'training rows keep the mirror symmetry; per-trial peak scaling discards amplitude',
    'outliers': 'never drop or relabel evaluation trials; training-side cleaning must earn its place against a '
                'size-matched control',
    'exclusions': 'E4b and E4c remove fragments that are not reaches (a reach from arm\'s length lasts longer and '
                  'rotates the forearm: README section 5.4), E4d removes trials with holes in the movement; the '
                  'evidence here is only their effect on the primary trials when kept in training, and the accuracy '
                  'on the excluded trials themselves (P1_09)',
    'platform': 'platform is a stratum, not a feature; transfer shows whether one platform can train the other, '
                'judged against size-matched controls',
    'augmentation': 'mirror augmentation encodes the left/right symmetry of the task',
}


def decisions(comp: pd.DataFrame, grip_p: float | None) -> list[dict[str, Any]]:
    """One row per decision family (options only). The pre-specified rule: change a default only if one option is
    better (paired 95 % CI of the pooled accuracy difference above 0) for at least two panel models and worse for
    none; an option tested on one model cannot meet it ('not decidable by the rule'). Evidence words follow the
    multiplicity-adjusted verdicts; the non-inferiority margin of the default is the largest upper CI bound of
    an option (the most an alternative could be better), the robustness margin the lowest lower bound."""
    rows = []
    opts = comp[comp['mult_family'] == 'options']
    for fam in LB.DECISION_FAMILIES:
        c = opts[opts['report_family'] == fam]
        if not len(c):
            continue
        per_opt = {o: sorted(g['model']) for o, g in c.groupby('option')}
        single = sorted(o for o, ms in per_opt.items() if len(ms) < 2)
        change = [o for o, g in c.groupby('option')
                  if len(g) >= 2 and (g['d_acc_lo'] > 0).sum() >= 2 and not (g['d_acc_hi'] < 0).any()]
        ev = {}
        for m in LB.MODELS:
            cm = c[c['model'] == m]
            if not len(cm):
                continue
            ev[m] = {'n_options': int(len(cm)), 'min_d': float(cm['d_acc'].min()), 'max_d': float(cm['d_acc'].max()),
                     'n_worse': int((cm['verdict'] == 'worse').sum()), 'n_better': int((cm['verdict'] == 'better').sum()),
                     'n_nominal': int(cm['verdict'].isin(['nominally worse', 'nominally better']).sum()),
                     'n_identical': int((cm['verdict'] == 'identical').sum())}
        i_hi = c['d_acc_hi'].idxmax()
        i_lo = c['d_acc_lo'].idxmin()
        listed = lambda v: [{'option': r['option'], 'model': r['model'], 'd_acc': r['d_acc'],  # noqa: E731
                             'd_acc_lo': r['d_acc_lo'], 'd_acc_hi': r['d_acc_hi'], 'q_bh': r['q_bh'],
                             'p_holm': r['p_holm'], 'p_sign_flip': r['p_sign_flip']}
                            for _, r in c[c['verdict'] == v].sort_values('d_acc').iterrows()]
        ll = lambda v: [{'option': r['option'], 'model': r['model'], 'd_logloss': r['d_logloss'],  # noqa: E731
                         'd_logloss_lo': r['d_logloss_lo'], 'd_logloss_hi': r['d_logloss_hi'],
                         'q_bh_logloss': r['q_bh_logloss']}
                        for _, r in c[c['logloss_verdict'] == v].sort_values('d_logloss').iterrows()]
        template = RATIONALE[fam]
        from stage2.lanes.p1_panel import DEFAULT_CHANNELS
        values = {'grip_p': grip_p, 'mr_n_channels': len(DEFAULT_CHANNELS['mr']),
                  'gbm_n_channels': len(DEFAULT_CHANNELS['gbm'])}
        rationale = template.format(grip_p=f'{grip_p:.2f}' if grip_p is not None else 'n/a',
                                    mr_n_channels=values['mr_n_channels'], gbm_n_channels=values['gbm_n_channels'])
        rows.append({
            'family': fam, 'default': X.DEFAULT_LABELS[fam],
            'options': list(dict.fromkeys(c['option'])), 'option_labels': [LB.option_label(o) for o in dict.fromkeys(c['option'])],
            'n_options': int(c['option'].nunique()), 'n_comparisons': int(len(c)),
            'models_tested': [m for m in LB.MODELS if m in set(c['model'])], 'models_per_option': per_opt,
            'single_model_options': single, 'n_decidable': int(len(per_opt) - len(single)),
            'evidence': ev, 'worse': listed('worse'), 'better': listed('better'),
            'nominally_worse': listed('nominally worse'), 'nominally_better': listed('nominally better'),
            'identical': [{'option': r['option'], 'model': r['model'], 'max_abs_dp': r['max_abs_dp']}
                          for _, r in c[c['verdict'] == 'identical'].iterrows()],
            'logloss_worse': ll('worse'), 'logloss_better': ll('better'),
            'logloss_nominally_worse': ll('nominally worse'), 'logloss_nominally_better': ll('nominally better'),
            'max_gain': {'option': c.loc[i_hi, 'option'], 'model': c.loc[i_hi, 'model'], 'd_acc_hi': float(c.loc[i_hi, 'd_acc_hi']),
                         'd_acc': float(c.loc[i_hi, 'd_acc'])},
            'max_loss': {'option': c.loc[i_lo, 'option'], 'model': c.loc[i_lo, 'model'], 'd_acc_lo': float(c.loc[i_lo, 'd_acc_lo']),
                         'd_acc': float(c.loc[i_lo, 'd_acc'])},
            'change': change, 'decidable': bool(len(per_opt) > len(single)),
            'decision': ('change to ' + ', '.join(LB.option_label(o) for o in change)) if change else 'keep the default',
            'rule_rationale': rationale, 'rationale_template': template,
            'rationale_values': {k: v for k, v in values.items() if '{' + k + '}' in template}})
    return rows


def filter_delay() -> dict[str, Any]:
    """Group delay (ms) of causal Butterworth low-passes at the stream rate (zero phase: 0 by construction)."""
    from scipy import signal
    f = np.linspace(0.1, 10.0, 100)
    out = {'freq_hz': f.tolist(), 'curves': []}
    for cut in (10.0, 15.0, 20.0):
        for order in (2, 4, 8):
            b, a = signal.butter(order, cut, fs=C.FS_HZ)
            _, gd = signal.group_delay((b, a), w=f, fs=C.FS_HZ)
            out['curves'].append({'cutoff_hz': cut, 'order': order, 'delay_ms': (gd / C.FS_HZ * 1000.0).tolist(),
                                  'delay_ms_at_1hz': float(np.interp(1.0, f, gd / C.FS_HZ * 1000.0))})
    return out


def variant_array_diffs(trials: pd.DataFrame) -> list[dict[str, Any]]:
    """How much each variant changes the arrays of the primary trials against the default rebuild (the core
    cache): trials that differ, max |difference| overall and on each panel model's channels."""
    from stage2.lanes import p1_variants as V
    from stage2.lanes.p1_panel import DEFAULT_CHANNELS
    from stage2.standardize import CHANNELS
    ids = trials.loc[trials['analysis_set'] == 'primary', 'trial_id']
    ios = (trials.loc[trials['analysis_set'] == 'primary', 'platform'].astype(str) == 'ios').to_numpy()
    try:
        base = V.load('base', ids)
    except (OSError, KeyError, ValueError):
        return []
    out = []
    for name in V.VARIANTS:
        if name == 'base' or not V.is_current(name):
            continue
        try:
            Xv = V.load(name, ids)
        except (OSError, KeyError, ValueError):
            continue
        if Xv.shape != base.shape:
            out.append({'variant': name, 'comparable': False})
            continue
        D = np.abs(Xv.astype(np.float64) - base.astype(np.float64))
        row = {'variant': name, 'comparable': True, 'n_trials': int(D.shape[0]),
               'n_trials_differing': int(np.sum(D.reshape(D.shape[0], -1).max(axis=1) > 0)),
               'n_ios_differing': int(np.sum((D.reshape(D.shape[0], -1).max(axis=1) > 0) & ios)),
               'max_abs_diff': float(D.max())}
        for m, chs in DEFAULT_CHANNELS.items():
            idx = [CHANNELS.index(c) for c in chs]
            row[f'max_abs_diff_{m}'] = float(D[:, idx].max())
        out.append(row)
    return out


# ───────────────────────────── main ─────────────────────────────


def derive() -> None:
    from stage2.data import load_cache
    d_all = load_cache(sets=None)
    trials = d_all.trials
    grid_index = {e.name: i for i, e in enumerate(X.GRID)}
    res: dict[str, cv.ExperimentResult] = {}
    for e in X.GRID:
        r = _load(e.name)
        if r is not None:
            res[e.name] = r
    for p in sorted(C.RES_DIR.glob('p1_calib_*.json')):
        res[p.stem] = cv.ExperimentResult.load(p)
    runs = [run_row(n, r, X.BY_NAME.get(n), grid_index) for n, r in res.items()]
    runs_df = pd.DataFrame(runs)
    headline = {n: metric_block(res[n].predictions) for n in HEADLINE_RUNS if n in res}
    L.write_json('p1_runs', clean({'runs': runs, 'grid_order': [e.name for e in X.GRID], 'headline': headline,
                                   'panel': panel_settings()}))

    fam_rows: dict[str, list[dict[str, Any]]] = {'options': [], 'robustness': [], 'pairs': []}
    other: list[dict[str, Any]] = []
    for e in X.GRID:
        if not (e.baseline and e.name in res and e.baseline in res):
            continue
        rl = LB.role(e.option)
        row = compare_row(e.name, e.baseline, res[e.name], res[e.baseline], model=e.model, family=e.family,
                          report_family=LB.report_family(e.option, e.family), role=rl, option=e.option,
                          option_label=LB.option_label(e.option), value=e.value, grid_index=grid_index[e.name])
        fam = {'option': 'options', 'robustness': 'robustness', 'control': 'pairs'}.get(rl)
        (fam_rows[fam] if fam else other).append(row)
    pair_rows = []
    for kind, pairs in (('remedy', REMEDY_PAIRS), ('control', CONTROL_PAIRS)):
        for a, b, fam, what in pairs:
            if a in res and b in res:
                ea = X.BY_NAME[a]
                pair_rows.append(compare_row(a, b, res[a], res[b], model=ea.model, family=fam, report_family=fam,
                                             role=kind, option=ea.option, option_label=what, value=ea.value,
                                             grid_index=grid_index[a], pair_kind=kind, what=what))
    fam_rows['pairs'] = fam_rows['pairs'] + pair_rows
    comps: list[dict[str, Any]] = []
    for fam, rows in fam_rows.items():
        comps += apply_family(rows, fam) if rows else []
    comps += [verdicts({**r, 'mult_family': None, 'family_size': None, 'q_bh': None, 'p_holm': None,
                        'q_bh_logloss': None, 'p_holm_logloss': None}) for r in other]
    comp_df = pd.DataFrame(comps)
    main = [c for c in comps if c.get('pair_kind') is None]
    L.write_json('p1_comparisons', clean({
        'n_boot': N_BOOT, 'alpha': ALPHA, 'logloss_tolerance': LL_TOL, 'identical_tol': LB.IDENTICAL_TOL,
        'families': {k: {'description': v, 'size': len(fam_rows[k])} for k, v in FAMILIES.items()},
        'comparisons': main, 'remedies': [c for c in comps if c.get('pair_kind') == 'remedy'],
        'controls': [c for c in comps if c.get('pair_kind') == 'control']}))
    gaps = e4d_gaps(d_all)
    L.write_json('p1_subsets', clean({'rows': subsets(res, trials, gaps), 'min_persons_ci': LB.MIN_PERSONS_CI,
                                      'e4d_gaps': gaps.drop(columns=['trial_id']).to_dict(orient='records')}))
    bm = best_model(runs_df)
    bd = breakdowns(res[f'p1_{bm}_base'], trials)
    L.write_json('p1_best', clean({'best_model': bm, 'label': LB.MODEL_LABELS[bm], 'n_configs_tried': 3,
                                   'exploratory': True,
                                   'rule': 'highest base LOSO accuracy among the three pre-specified panel models '
                                           '(an outer-fold choice among three: exploratory, n_configs_tried = 3)',
                                   **bd}))
    L.write_json('p1_calibration', clean(calibration(res, trials)))
    L.write_json('p1_decisions', clean({
        'rule': 'change a default only if one option is better (paired 95 % CI of the pooled accuracy difference '
                'above 0) for at least two panel models and worse for none',
        'rows': decisions(comp_df, bd.get('grip_balance_chi2_p')) if len(comp_df) else []}))
    L.write_json('p1_filter_delay', clean(filter_delay()))
    from stage2.lanes import p1_variants as V
    L.write_json('p1_variants', clean({'variants': V.describe_variants(), 'fidelity': V.fidelity('base'),
                                       'array_diffs': variant_array_diffs(trials)}))
    stale_cache = L.tab_dir / 'p1_comparisons_cache.json'     # no longer used (the comparisons are fast now)
    if stale_cache.exists():
        stale_cache.unlink()


__all__ = ['derive', 'acc_ci', 'reliability_bands', 'decisions', 'best_model', 'CALIB_MODELS', 'fast_metrics',
           'fast_person_bootstrap', 'metric_block', 'aggregates', 'verdicts', 'apply_family', 'vs_primary']
