"""Lane m4 presentation: report tables (CSV + booktabs), the numbers file and m4_SUMMARY.md, all from saved
files only (results/m4_*.json, tables/m4/*). Nothing here fits a model. Every accuracy is the half-credit
accuracy of ``m4_stats`` (an abstention, P = 0.5, scores half a correct answer); every verdict sentence is
generated from the saved paired comparisons (Holm-adjusted within a named family)."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from stage2 import report_export as RX
from stage2.pipeline import Lane, selected

L = Lane('m4')
E = RX.numbers_entry

#: number-key prefix of every trial experiment
SHORT = {'m4_oracle_yaw8': 'oracle', 'm4_oracle_chiral16': 'oracle_chiral16', 'm4_v0_yaw8': 'vzero',
         'm4_v1phys_yaw8': 'physics', 'm4_v1full_yaw8': 'full', 'm4_v1nest_yaw8': 'nested', 'm4_v1robust_yaw8': 'robust',
         'm4_v1grip_yaw8': 'grip', 'm4_v1nest_yaw8_detwin': 'detwin', 'm4_v1robust_chiral16': 'chiral',
         'm4_cal_temperature': 'cal_temperature', 'm4_cal_platt': 'cal_platt', 'm4_cal_isotonic': 'cal_isotonic',
         'm4_cal_beta': 'cal_beta', 'm4_cal_isotonic_sym': 'cal_isotonic_sym', 'm4_cal_nested': 'prod',
         'm4_ret_app_yaw8': 'ret_app', 'm4_ret_head_yaw8': 'ret_head', 'm4_ret_sum_yaw8': 'ret_sum',
         'm4_prod_return': 'prod_return', 'm4_prod_zp_activity': 'prod_zp', 'm4_prod_causal_lowpass': 'prod_causal',
         'm4_abl_scale_z': 'abl_scale_z', 'm4_abl_scale_none': 'abl_scale_none', 'm4_abl_segc_nested': 'abl_segc_nested',
         'm4_abl_look0p3': 'abl_look_short', 'm4_abl_look0p8': 'abl_look_long'}


def _P() -> Any:
    from stage2.lanes import m4_pipeline as P
    return P


def _res(name: str) -> dict[str, Any] | None:
    p = L.result_path(name)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def _j(name: str) -> dict[str, Any]:
    try:
        return L.read_json(name)
    except FileNotFoundError:
        return {}


def _ok(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _ci(lo: Any, hi: Any, pct: bool = True) -> str:
    if not (_ok(lo) and _ci_ok(hi)):
        return '--'
    return f'[{100 * lo:.1f}, {100 * hi:.1f}]' if pct else f'[{lo:.3f}, {hi:.3f}]'


def _ci_ok(x: Any) -> bool:
    return _ok(x)


def _label(name: str) -> str:
    from stage2.lanes import m4_figures as F
    return F.LABELS.get(name, name)


def _prod() -> str:
    return _P().production_name()


# ───────────────────────────── tables ─────────────────────────────


def model_rows() -> pd.DataFrame:
    """One row per trial experiment: half-credit accuracy with CI, AUC, log loss, ECE, coverage, natural use."""
    P = _P()
    S = _j('m4_scores').get('scores', {})
    rows = []
    for n in P.TRIAL_EXPERIMENTS:
        r = _res(n)
        if r is None or n not in S:
            continue
        sp, sn = S[n].get('primary', {}), S[n].get('natural_use', {})
        m = r['metrics']
        rows.append({'name': n, 'label': _label(n), 'n': int(m['n']), 'accuracy': sp.get('accuracy_half'),
                     'acc_ci': _ci(*sp.get('accuracy_half_ci', [None, None])), 'roc_auc': m['roc_auc'],
                     'log_loss': m['log_loss'], 'ece': m['ece'], 'coverage': sp.get('coverage'),
                     'acc_answered': sp.get('accuracy_answered'), 'acc_forced': sp.get('accuracy_forced'),
                     'nat_answered': (f"{sn['n_answered']}/{sn['n']}" if sn else '--'),
                     'nat_acc_answered': sn.get('accuracy_answered', np.nan) if sn else np.nan,
                     'nat_acc_half': sn.get('accuracy_half', np.nan) if sn else np.nan,
                     'n_configs_tried': r['info'].get('n_configs_tried'), 'leak_check': r['leak_check']})
    return pd.DataFrame(rows)


def _check_table_names() -> None:
    """A report table must never take the name of a file the compute step writes under tables/m4 (the report
    table would overwrite the raw rows a later compute step or figure reads back)."""
    P = _P()
    clash = set(P.RAW_TABLES) & set(P.REPORT_TABLES)
    if clash:
        raise ValueError(f'report tables would overwrite raw compute files: {sorted(clash)}')


def _export(t: pd.DataFrame, name: str, **kw: Any) -> None:
    P = _P()
    if name not in P.REPORT_TABLES or name in P.RAW_TABLES:
        raise ValueError(f'{name}: not a registered m4 report table (m4_pipeline.REPORT_TABLES)')
    L.export_table(t, name, **kw)


HALF = ('an abstention (no approach found, P = 0.5) scores half a correct answer')


def tables(only: list[str] | None = None) -> None:
    P = _P()
    _check_table_names()
    if selected('m4_models', only):
        df = model_rows()
        if len(df):
            t = df[['label', 'n', 'accuracy', 'acc_ci', 'roc_auc', 'log_loss', 'ece', 'coverage', 'nat_answered',
                    'nat_acc_answered', 'nat_acc_half', 'n_configs_tried']].rename(columns={
                        'label': 'Pipeline', 'n': 'Trials', 'accuracy': 'Accuracy', 'acc_ci': '95% CI', 'roc_auc': 'AUC',
                        'log_loss': 'Log loss', 'ece': 'ECE', 'coverage': 'Answered', 'nat_answered': 'Nat. answered',
                        'nat_acc_answered': 'Nat. acc. answered', 'nat_acc_half': 'Nat. acc.',
                        'n_configs_tried': 'Configs'})
            _export(t, 'm4_models', caption=(
                'Lane m4: cue-free pipelines and their references (leave-one-person-out; 95\\,\\% person-cluster '
                f'bootstrap CI). Accuracy: {HALF}. Answered: share of recordings with an approach found. Natural use '
                '(one first free-choice trial per person, all right hands): answered / scored, accuracy on the answered '
                'ones and half-credit accuracy. Configs: n\\_configs\\_tried recorded with the result.'),
                column_formats={'Accuracy': '.1%', 'AUC': '.3f', 'Log loss': '.3f', 'ECE': '.3f', 'Answered': '.0%',
                                'Nat. acc. answered': '.0%', 'Nat. acc.': '.1%'})
    if selected('m4_detection', only):
        det = pd.DataFrame(_j('m4_detection').get('rows', []))
        if len(det):
            det = det[det['set'] == 'primary']
            t = pd.DataFrame({'Detector': det['name'].map(_label).fillna(det['name']),
                              'Approach': det['share_approach'], 'Part': det['share_approach_partial'],
                              'Earlier': det['share_earlier_movement'], 'Return': det['share_return'],
                              'Rejected': det['share_rejected'], 'None': det['share_none'],
                              'Onset err. (ms)': det['onset_error_median_ms'], 'End err. (ms)': det['end_error_median_ms']})
            _export(t.drop_duplicates(), 'm4_detection', caption=(
                'What the detector scores on each full primary recording, against the app\'s stored events (share of '
                'recordings); bound errors are medians over the recordings where the approach was found.'),
                column_formats={c: '.0%' for c in ('Approach', 'Part', 'Earlier', 'Return', 'Rejected', 'None')} |
                {'Onset err. (ms)': '.0f', 'End err. (ms)': '.0f'})
    cal = _j('m4_calibration')
    if selected('m4_calibration', only) and cal:
        rows = []
        for c in cal['candidates']:
            vs = c.get('vs_none', {})
            rows.append({'Calibrator': c['calibration'], 'Antisym.': 'yes' if c.get('antisymmetric') else 'no',
                         'Mirror dev.': c.get('mirror_dev_max', np.nan), 'Accuracy': c['accuracy_half'],
                         'Log loss': c['log_loss'],
                         'd log loss': vs.get('d_ll', np.nan), 'd CI': _ci(*vs.get('d_ll_ci', [None, None]), pct=False),
                         'Holm p': vs.get('p_ll_holm', np.nan), 'Brier': c['brier'], 'ECE': c['ece'],
                         'Null band': _ci(c['ece_null_lo'], c['ece_null_hi'], pct=False), 'Slope': c['slope'],
                         'Slope CI': _ci(*c['slope_ci'], pct=False)})
        t = pd.DataFrame(rows)
        _export(t, 'm4_calibration', caption=(
            'Calibrators of the cue-free production detector, each fitted on cross-fitted predictions of the training '
            'persons and applied to the held-out person; nested = chosen inside each fold among the antisymmetric ones '
            '(the production calibrator). Mirror dev.: largest |P(z) + P(-z) - 1| over the folds (0: a mirror-image '
            'recording gets exactly 1 - P). d log loss: paired difference to none (person bootstrap; Holm over the '
            'calibrators). Null band: 95\\,\\% range of the ECE of a perfectly calibrated P with the same values '
            '(persons resampled, outcomes redrawn). Slope: calibration slope of y on logit(P) (1 = calibrated).'),
            column_formats={'Mirror dev.': '.3f', 'Accuracy': '.1%', 'Log loss': '.3f', 'd log loss': '+.3f',
                            'Holm p': '.3f', 'Brier': '.3f', 'ECE': '.3f', 'Slope': '.2f'})
    S = _j('m4_sessions')
    if S and any(selected(n, only) for n in ('m4_sessions', 'm4_sessions_k', 'm4_stopping', 'm4_stop_calibration')):
        from stage2.lanes.m4_online import AGGREGATION_LABELS
        plain = {h: v.replace('$s_k$', 's_k') for h, v in AGGREGATION_LABELS.items()}
        k = pd.DataFrame(S['k_curve'])
        k = k[k['aggregation'] == 'mean']                  # the sign of the mean: any aggregation
        prod_rule, ref = S['production_rule'], float(S['reference_keep_s'])
        t = k[k['rule'] == prod_rule].pivot_table(index='k', columns='keep_s', values='accuracy').reset_index()
        t.columns = ['Attempts (up to)'] + [f'{c:g} s' for c in t.columns[1:]]
        for rule, lab in (('pairing', f'{ref:g} s, pairing rule'), ('true_only', f'{ref:g} s, true approaches only')):
            g = k[(k['rule'] == rule) & (k['keep_s'] == ref)].set_index('k')['accuracy']
            t[lab] = t['Attempts (up to)'].map(g)
        g = k[(k['rule'] == prod_rule) & (k['keep_s'] == ref)].set_index('k')['share_k_available']
        t['Sessions with k'] = t['Attempts (up to)'].map(g)
        _export(t, 'm4_sessions_k', caption=(
            f"Synthetic multi-attempt sessions ({S['session_recordings']} recordings of one person and hand each): "
            'half-credit accuracy when the first k accepted attempts are aggregated (the sign of their mean '
            'calibrated logit; fewer when the session has fewer: "up to k"), by the stillness left at each junction '
            f'(production attempt rule: {prod_rule.replace("_", " ")}); then the pairing rule and the attainable curve '
            'when only the true approaches are kept, and the share of sessions with at least k accepted attempts.'),
            column_formats={c: '.1%' for c in t.columns if c != 'Attempts (up to)'} | {'Sessions with k': '.0%'})
        st = pd.DataFrame(S['stopping'])
        cal = _j('m4_calibration')
        how = cal.get('production_aggregation', 'mean')
        st = st[(st['rule'] == prod_rule) & (st['aggregation'] == how)]
        t = st[['keep_s', 'p_stop', 'accuracy', 'acc_lo', 'acc_hi', 'mean_attempts', 'mean_true_touches',
                'share_confident_stop', 'acc_confident_stop', 'ece']].rename(
            columns={'keep_s': 'Stillness (s)', 'p_stop': 'p_stop', 'accuracy': 'Accuracy', 'acc_lo': 'CI low',
                     'acc_hi': 'CI high', 'mean_attempts': 'Attempts', 'mean_true_touches': 'Touches',
                     'share_confident_stop': 'Confident stops', 'acc_confident_stop': 'Acc. confident', 'ece': 'ECE'})
        _export(t, 'm4_stopping', caption=(
            'Sequential stopping rule on the synthetic sessions (production attempt rule; attempts aggregated with '
            f"{plain.get(how, how)}): stop when the aggregated P(right) leaves [1 - p_stop, p_stop], else answer after "
            f"{S['k_max']} accepted attempts or at the end of the session. Attempts: accepted attempts used; Touches: "
            'true approaches completed by the decision. Confident stops: share of sessions that stopped on the '
            f'threshold, and their accuracy ({HALF}).'),
            column_formats={'Stillness (s)': '.2g', 'p_stop': '.2f', 'Accuracy': '.1%', 'CI low': '.1%', 'CI high': '.1%',
                            'Attempts': '.2f', 'Touches': '.2f', 'Confident stops': '.0%', 'Acc. confident': '.1%',
                            'ECE': '.3f'})
        k3, stop = cal.get('k3') or {}, cal.get('stop') or {}
        rows = []
        for h in AGGREGATION_LABELS:
            if h not in k3 and h not in stop:
                continue
            a, b = k3.get(h, {}), stop.get(h, {})
            rows.append({'Aggregation': plain[h] + (' (production)' if h == how else ''),
                         'Params': b.get('n_params', np.nan), 'k=3 log loss': a.get('log_loss', np.nan),
                         'k=3 slope': a.get('slope', np.nan), 'Stop accuracy': b.get('accuracy_half', np.nan),
                         'Attempts': b.get('mean_attempts', np.nan), 'Stop log loss': b.get('log_loss', np.nan),
                         'Stop ECE': b.get('ece', np.nan),
                         'Null band': _ci(b.get('ece_null_lo'), b.get('ece_null_hi'), pct=False) if b else '',
                         'Slope': b.get('slope', np.nan),
                         'Slope CI': _ci(*b.get('slope_ci', [None, None]), pct=False) if b else '',
                         'Calibrated': ('yes' if b.get('calibrated') else 'no') if b else ''})
        if rows:
            _export(pd.DataFrame(rows), 'm4_stop_calibration', caption=(
                f"Calibration of several attempts on the synthetic sessions ({cal.get('sessions_keep_s', 1):g} s of "
                'stillness at each junction, production attempt rule): three accepted attempts (k=3) and the answer '
                f"of the stopping rule (p_stop {cal.get('p_stop', np.nan):.2f}) under each aggregation of the calibrated "
                'logits. Calibrated: ECE inside its perfect-calibration null band and slope CI containing 1; the '
                'production aggregation is the calibrated one with the fewest fitted parameters (declared rule).'),
                column_formats={'Params': 'd', 'k=3 log loss': '.3f', 'k=3 slope': '.2f', 'Stop accuracy': '.1%',
                                'Attempts': '.2f', 'Stop log loss': '.3f', 'Stop ECE': '.3f', 'Slope': '.2f'})
    B = _j('m4_breakdown')
    if selected('m4_breakdown', only) and B:
        b = pd.DataFrame(B['rows'])
        b = b[b['name'] == B['production']]
        t = b[['stratum', 'level', 'n', 'n_persons', 'accuracy', 'acc_lo', 'acc_hi', 'coverage', 'ece', 'log_loss']].rename(
            columns={'stratum': 'Stratum', 'level': 'Level', 'n': 'Trials', 'n_persons': 'Persons', 'accuracy': 'Accuracy',
                     'acc_lo': 'CI low', 'acc_hi': 'CI high', 'coverage': 'Answered', 'ece': 'ECE', 'log_loss': 'Log loss'})
        _export(t, 'm4_breakdown', caption=(
            'Cue-free production pipeline by platform and by instructed condition (conditions stratify only; '
            f"person-bootstrap 95\\,\\% CI, none for strata with fewer than {B.get('min_persons_ci', 5)} persons; {HALF})."),
            column_formats={'Accuracy': '.1%', 'CI low': '.1%', 'CI high': '.1%', 'Answered': '.0%', 'ECE': '.3f',
                            'Log loss': '.3f'})
    ST = _j('m4_stress')
    if selected('m4_stress', only) and ST:
        s = pd.DataFrame(ST['summary'])
        t = s[['pipeline', 'keep', 'flip', 'accuracy', 'acc_lo', 'acc_hi', 'coverage', 'share_approach', 'share_return',
               'gate_share', 'acc_gated']].copy()
        t['flip'] = np.where(t['flip'], 'screen away', '')
        t = t.rename(columns={'pipeline': 'Pipeline', 'keep': 'Stillness kept (s)', 'flip': 'Grip', 'accuracy': 'Accuracy',
                              'acc_lo': 'CI low', 'acc_hi': 'CI high', 'coverage': 'Answered',
                              'share_approach': 'Approach found', 'share_return': 'Return',
                              'gate_share': 'Grip check fires', 'acc_gated': 'Acc. with check'})
        _export(t, 'm4_stress', caption=(
            'Stillness stress test of single recordings: the stillness before each approach cut to the given seconds '
            '(recorded = as recorded), and the screen-away twin of every recording (the phone turned 180 deg about its '
            f'long axis); each pipeline fitted without the person ({HALF}). Grip check: the inference page\'s rest-pose '
            'check (the screen-normal up component at rest before the first movement below '
            f"{ST.get('grip_gate_upz', 0):g}: the page asks to turn the screen and does not answer)."),
            column_formats={'Accuracy': '.1%', 'CI low': '.1%', 'CI high': '.1%', 'Answered': '.0%',
                            'Approach found': '.0%', 'Return': '.0%', 'Grip check fires': '.0%', 'Acc. with check': '.1%'})
    PA = _j('m4_paired')
    if selected('m4_paired', only) and PA:
        rows = [{'Comparison': k, 'A': _label(v['a']), 'B': _label(v['b']), 'Acc. A': v['acc_a'], 'Acc. B': v['acc_b'],
                 'Difference': v['d_acc'], 'CI low': v['d_acc_ci'][0], 'CI high': v['d_acc_ci'][1],
                 'p': v['p_acc'], 'Holm p': v['p_acc_holm'], 'Log-loss diff.': v['d_ll'],
                 'Identical': 'yes' if v.get('identical') else ''} for k, v in PA['pairs'].items()]
        _export(pd.DataFrame(rows), 'm4_paired', caption=(
            'Paired comparisons on the same recordings (A - B; half-credit accuracy; person-bootstrap 95\\,\\% CI; exact '
            f"sign-flip test over persons; Holm-adjusted over the {PA.get('n_family', len(rows))} comparisons of this "
            'table, one family).'),
            column_formats={'Acc. A': '.1%', 'Acc. B': '.1%', 'Difference': '+.1%', 'CI low': '+.1%', 'CI high': '+.1%',
                            'p': '.3f', 'Holm p': '.3f', 'Log-loss diff.': '+.3f'})
    SC = _j('m4_scores').get('scores', {})
    if selected('m4_natural', only) and SC:
        rows = []
        for n in ('m4_oracle_yaw8', 'm4_v0_yaw8', 'm4_v1robust_yaw8', 'm4_v1grip_yaw8', _prod()):
            for set_name in ('primary', 'natural_use'):
                e = SC.get(n, {}).get(set_name)
                if not e:
                    continue
                rows.append({'Pipeline': _label(n), 'Set': set_name.replace('_', ' '), 'Trials': e['n'],
                             'Answered': e['coverage'], 'Acc. answered': e['accuracy_answered'],
                             'Forced choice': e['accuracy_forced'], 'Half credit': e['accuracy_half'],
                             'CI': _ci(*e['accuracy_half_ci'])})
        if rows:
            _export(pd.DataFrame(rows), 'm4_natural', caption=(
                'Primary and natural-use recordings (one first free-choice trial per person, all right hands): share '
                'answered (an approach accepted), accuracy on the answered recordings, forced-choice accuracy (the best '
                'candidate always scored) and half-credit accuracy with a person-bootstrap 95\\,\\% CI.'),
                column_formats={'Answered': '.0%', 'Acc. answered': '.1%', 'Forced choice': '.1%', 'Half credit': '.1%'})
    if selected('m4_ablations', only) and PA:
        rows = []
        for k, v in PA['pairs'].items():
            if k.startswith('abl_'):
                rows.append({'Ablation': _label(v['a']), 'Accuracy': v['acc_a'], 'Base': v['acc_b'], 'Difference': v['d_acc'],
                             'CI': _ci(*v['d_acc_ci']), 'Holm p': v['p_acc_holm'], 'Log-loss diff.': v['d_ll'],
                             'Log-loss CI': _ci(*v['d_ll_ci'], pct=False)})
        thr = _j('m4_scores').get('det_threshold', [])
        for r in thr:
            if r.get('vs_production'):
                v = r['vs_production']
                rows.append({'Ablation': f"P(approach) threshold {r['threshold']:.2f}", 'Accuracy': r['accuracy'],
                             'Base': v['acc_b'], 'Difference': v['d_acc'], 'CI': _ci(*v['d_acc_ci']),
                             'Holm p': np.nan, 'Log-loss diff.': v['d_ll'], 'Log-loss CI': _ci(*v['d_ll_ci'], pct=False)})
        if rows:
            _export(pd.DataFrame(rows), 'm4_ablations', caption=(
                'Data-treatment ablations of the production detector (uncalibrated; paired with it on the same '
                f'recordings; {HALF}): feature scaling, the segment classifier\'s C, the look-ahead after a segment, '
                'and the P(approach) threshold (re-thresholded saved predictions of the production pipeline, paired with '
                'it; outside the Holm family).'),
                column_formats={'Accuracy': '.1%', 'Base': '.1%', 'Difference': '+.1%', 'Holm p': '.3f',
                                'Log-loss diff.': '+.3f'})


# ───────────────────────────── numbers ─────────────────────────────


def numbers() -> dict[str, Any]:
    P = _P()
    n: dict[str, Any] = {}
    prod = _prod()
    src = lambda name: f'results/{name}.json'  # noqa: E731
    tsrc = lambda name: f'tables/m4/{name}.json'  # noqa: E731
    SC = _j('m4_scores').get('scores', {})

    def put(key: str, value: Any, fmt: str, source: str, desc: str) -> None:
        n[key] = E(value, fmt, source, desc)

    for name, key in SHORT.items():
        r = _res(name)
        if r is None:
            continue
        m, ci = r['metrics'], r['ci95']
        desc = _label(name)
        put(f'{key}_n', int(m['n']), 'd', src(name), f'{desc}: recordings scored')
        sp = SC.get(name, {}).get('primary')
        if sp:
            put(f'{key}_acc', sp['accuracy_half'], '.1%', tsrc('m4_scores'), f'{desc}: LOSO accuracy (half credit)')
            put(f'{key}_acc_lo', sp['accuracy_half_ci'][0], '.1%', tsrc('m4_scores'), f'{desc}: accuracy 95% CI low')
            put(f'{key}_acc_hi', sp['accuracy_half_ci'][1], '.1%', tsrc('m4_scores'), f'{desc}: accuracy 95% CI high')
            put(f'{key}_coverage', sp['coverage'], '.1%', tsrc('m4_scores'), f'{desc}: share answered')
            put(f'{key}_n_answered', sp['n_answered'], 'd', tsrc('m4_scores'), f'{desc}: recordings answered')
            put(f'{key}_n_answered_tie', sp.get('n_answered_tie'), 'd', tsrc('m4_scores'),
                f'{desc}: answered recordings whose P(right) is exactly 0.5 (no hand shown)')
            put(f'{key}_acc_answered', sp['accuracy_answered'], '.1%', tsrc('m4_scores'), f'{desc}: accuracy on answered')
            put(f'{key}_acc_forced', sp['accuracy_forced'], '.1%', tsrc('m4_scores'), f'{desc}: forced-choice accuracy')
            put(f'{key}_acc_harness', sp['accuracy_ge05'], '.1%', tsrc('m4_scores'),
                f"{desc}: accuracy under the harness's rule until 2026-10-28 (P >= 0.5 predicts right, so an "
                'abstention counts as right)')
        for mk, fmt, what in (('roc_auc', '.3f', 'ROC AUC'), ('log_loss', '.3f', 'log loss'), ('brier', '.3f', 'Brier score')):
            kk = 'auc' if mk == 'roc_auc' else mk
            put(f'{key}_{kk}', m[mk], fmt, src(name), f'{desc}: {what}')
            if mk in ci:
                put(f'{key}_{kk}_lo', ci[mk][0], fmt, src(name), f'{desc}: {what}, 95% CI low')
                put(f'{key}_{kk}_hi', ci[mk][1], fmt, src(name), f'{desc}: {what}, 95% CI high')
        put(f'{key}_ece', m['ece'], '.3f', src(name), f'{desc}: ECE (10 equal-mass bins)')
        sn = SC.get(name, {}).get('natural_use')
        if sn:
            put(f'{key}_nat_n', sn['n'], 'd', tsrc('m4_scores'), f'{desc}: natural-use trials')
            put(f'{key}_nat_answered', sn['n_answered'], 'd', tsrc('m4_scores'), f'{desc}: natural-use trials answered')
            put(f'{key}_nat_answered_tie', sn.get('n_answered_tie'), 'd', tsrc('m4_scores'),
                f'{desc}: natural-use trials answered with P(right) exactly 0.5')
            put(f'{key}_nat_coverage', sn['coverage'], '.0%', tsrc('m4_scores'), f'{desc}: natural use, share answered')
            put(f'{key}_nat_acc_answered', sn['accuracy_answered'], '.0%', tsrc('m4_scores'),
                f'{desc}: natural use, accuracy on the answered trials')
            put(f'{key}_nat_acc_forced', sn['accuracy_forced'], '.0%', tsrc('m4_scores'),
                f'{desc}: natural use, forced-choice accuracy')
            put(f'{key}_nat_acc', sn['accuracy_half'], '.1%', tsrc('m4_scores'), f'{desc}: natural use, half-credit accuracy')
            put(f'{key}_nat_acc_lo', sn['accuracy_half_ci'][0], '.1%', tsrc('m4_scores'), f'{desc}: natural use, CI low')
            put(f'{key}_nat_acc_hi', sn['accuracy_half_ci'][1], '.1%', tsrc('m4_scores'), f'{desc}: natural use, CI high')
            put(f'{key}_nat_acc_harness', sn['accuracy_ge05'], '.0%', tsrc('m4_scores'),
                f"{desc}: natural use, accuracy under the harness's rule until 2026-10-28 (every abstention counted "
                'as a right-hand answer)')
        k = {int(x['k']): x for x in r.get('k_attempts', [])}
        if 3 in k:
            put(f'{key}_acc_k3_indep', k[3]['accuracy'], '.1%', src(name),
                f'{desc}: 3 INDEPENDENT recordings of one person and hand averaged (harness k curve; abstentions '
                'averaged in), not a cue-free session')
            put(f'{key}_acc_k3_indep_lo', k[3]['lo'], '.1%', src(name), f'{desc}: same, 95% CI low')
            put(f'{key}_acc_k3_indep_hi', k[3]['hi'], '.1%', src(name), f'{desc}: same, 95% CI high')
            put(f'{key}_k_n_persons', int(k[3]['n_persons']), 'd', src(name), f'{desc}: persons in the harness k curve')
        put(f'{key}_configs', int(r['info'].get('n_configs_tried') or 1), 'd', src(name), f'{desc}: n_configs_tried')
    r = _res(prod)
    if r is not None:
        put('n_recordings', int(r['metrics']['n']), 'd', src(prod), 'primary recordings scored')
        put('n_persons', int(r['metrics']['n_persons']), 'd', src(prod), 'persons (= LOSO folds)')
        put('prod_name', prod, '', src(prod), 'result name of the production pipeline')
        per_fold = pd.DataFrame(r['predictions']).groupby('fold')['params'].first().map(json.loads)
        seg = per_fold.map(lambda p: p.get('seg_cfg')).value_counts()
        put('prod_seg_cfg_mode', str(seg.index[0]), '', src(prod), 'segmentation configuration chosen most often')
        put('prod_seg_cfg_mode_folds', int(seg.iloc[0]), 'd', src(prod), 'folds choosing it')
        put('prod_folds', int(len(per_fold)), 'd', src(prod), 'folds')
        cal_ch = per_fold.map(lambda p: (p.get('calibration') or {}).get('chosen')).value_counts()
        for meth, cnt in cal_ch.items():
            put(f'prod_calibrator_{meth}_folds', int(cnt), 'd', src(prod), f'folds whose nested choice was {meth}')
    # configuration values the text cites (read from the configuration, never typed in the text)
    from stage2.lanes import m4_config as CFG
    from stage2.lanes import m4_detect as D
    cfg_src = 'stage2/lanes/m4_config.py'
    put('grip_gate_upz', CFG.GRIP_GATE_UPZ, '.2f', cfg_src, 'rest-pose grip check: minimum screen-normal up component')
    put('instructed_stillness_s', D.INSTRUCTED_STILLNESS_S, 'g', 'stage2/lanes/m4_detect.py',
        'stillness the inference page asks for before each touch (s)')
    for k_, fmt in (('hi_dps', '.0f'), ('sustain_ms', '.0f'), ('smooth_hz', 'g'), ('still_cap_s', 'g'),
                    ('min_duration_ms', '.0f')):
        put(f'seg_{k_}', getattr(D.DEFAULT, k_), fmt, 'stage2/lanes/m4_detect.py', f'segmentation parameter {k_}')
    put('det_threshold', P.production_spec().det_threshold, '.2f', 'stage2/lanes/m4_e2e.py',
        'minimum P(approach) of an accepted segment')
    put('n_configs_final', P.N_CONFIGS_FINAL, 'd', 'stage2/lanes/m4_pipeline.py',
        'cue-free pipelines compared on the outer folds (n_configs_tried of the production result)')
    put('n_configs_detector', len(P.DETECTOR_FAMILY), 'd', 'stage2/lanes/m4_pipeline.py', 'detector feature sets compared')
    put('n_configs_sessions', P.N_CONFIGS_SESSIONS, 'd', 'stage2/lanes/m4_pipeline.py',
        'session-level configurations compared on the outer sessions (attempt rules x aggregations)')
    rn = _res('m4_v1nest_yaw8')
    if rn is not None:
        fc = pd.DataFrame(rn['predictions']).groupby('fold')['params'].first().map(lambda s: json.loads(s).get('features_chosen'))
        top = fc.value_counts()
        put('nested_features_chosen_mode', str(top.index[0]), '', src('m4_v1nest_yaw8'), 'feature set chosen most often (nested)')
        put('nested_features_chosen_mode_folds', int(top.iloc[0]), 'd', src('m4_v1nest_yaw8'), 'folds choosing it')
    det = _j('m4_detection').get('rows', [])
    for row in det:
        key = SHORT.get(row['name'])
        if key is None:
            continue
        pre = key if row['set'] == 'primary' else f'{key}_nat'
        for o in ('approach', 'approach_partial', 'earlier_movement', 'return', 'rejected', 'none'):
            put(f'{pre}_share_{o}', row[f'share_{o}'], '.0%', tsrc('m4_detection'),
                f'{pre}: share of recordings where the scored segment is {o}')
        if row['set'] == 'primary':
            put(f'{key}_onset_err_ms', row['onset_error_median_ms'], '.0f', tsrc('m4_detection'),
                f'{key}: median onset error (ms) when the approach is found')
            put(f'{key}_end_err_ms', row['end_error_median_ms'], '.0f', tsrc('m4_detection'),
                f'{key}: median end error (ms) when the approach is found')
    PA = _j('m4_paired')
    for k, v in PA.get('pairs', {}).items():
        what = f"{_label(v['a'])} - {_label(v['b'])}"
        put(f'd_{k}', v['d_acc'], '+.1%', tsrc('m4_paired'), f'paired half-credit accuracy difference {what}')
        put(f'd_{k}_lo', v['d_acc_ci'][0], '+.1%', tsrc('m4_paired'), f'{k}: 95% CI low')
        put(f'd_{k}_hi', v['d_acc_ci'][1], '+.1%', tsrc('m4_paired'), f'{k}: 95% CI high')
        put(f'd_{k}_p', v['p_acc'], '.3f', tsrc('m4_paired'), f'{k}: exact sign-flip p (unadjusted)')
        put(f'd_{k}_p_holm', v['p_acc_holm'], '.3f', tsrc('m4_paired'), f'{k}: Holm-adjusted p (family trial)')
        put(f'd_{k}_ll', v['d_ll'], '+.3f', tsrc('m4_paired'), f'{k}: mean log-loss difference (A - B)')
        put(f'd_{k}_ll_lo', v['d_ll_ci'][0], '+.3f', tsrc('m4_paired'), f'{k}: log loss, CI low')
        put(f'd_{k}_ll_hi', v['d_ll_ci'][1], '+.3f', tsrc('m4_paired'), f'{k}: log loss, CI high')
        put(f'd_{k}_ll_p_holm', v['p_ll_holm'], '.3f', tsrc('m4_paired'), f'{k}: log loss, Holm-adjusted p (family trial)')
        put(f'd_{k}_identical', 'yes' if v.get('identical') else 'no', '', tsrc('m4_paired'), f'{k}: identical predictions')
    if PA:
        put('paired_family_n', int(PA.get('n_family', len(PA.get('pairs', {})))), 'd', tsrc('m4_paired'),
            'comparisons in the Holm family trial')
    cal = _j('m4_calibration')
    if cal:
        for c in cal['candidates']:
            pre = f"cal_{c['calibration']}"
            for mk, fmt in (('accuracy_half', '.1%'), ('log_loss', '.3f'), ('brier', '.3f'), ('ece', '.3f'),
                            ('ece_null_lo', '.3f'), ('ece_null_hi', '.3f'), ('ece_p_null', '.3f'), ('slope', '.2f'),
                            ('intercept', '.2f')):
                put(f'{pre}_{mk}', c[mk], fmt, tsrc('m4_calibration'), f"one touch, calibrator {c['calibration']}: {mk}")
            for mk in ('slope_ci', 'intercept_ci'):
                put(f'{pre}_{mk[:-3]}_lo', c[mk][0], '.2f', tsrc('m4_calibration'), f"{pre}: {mk} low")
                put(f'{pre}_{mk[:-3]}_hi', c[mk][1], '.2f', tsrc('m4_calibration'), f"{pre}: {mk} high")
            put(f'{pre}_calibrated', 'yes' if c['calibrated'] else 'no', '', tsrc('m4_calibration'),
                f'{pre}: ECE inside its null band and slope CI containing 1')
            put(f'{pre}_antisymmetric', 'yes' if c.get('antisymmetric') else 'no', '', tsrc('m4_calibration'),
                f'{pre}: antisymmetric by construction (P(-z) = 1 - P(z))')
            put(f'{pre}_mirror_dev_max', c.get('mirror_dev_max'), '.3f', tsrc('m4_calibration'),
                f'{pre}: largest |P(z) + P(-z) - 1| over the folds (cross-fitted logits it was fitted on)')
            put(f'{pre}_mirror_dev_mean', c.get('mirror_dev_mean'), '.3f', tsrc('m4_calibration'),
                f'{pre}: mean |P(z) + P(-z) - 1|, averaged over the folds')
            vs = c.get('vs_none')
            if vs:
                for mk, fmt in (('d_ll', '+.3f'), ('d_brier', '+.4f'), ('d_acc', '+.1%')):
                    put(f'{pre}_{mk}', vs[mk], fmt, tsrc('m4_calibration'), f'{pre} - none: {mk}')
                    put(f'{pre}_{mk}_lo', vs[f'{mk}_ci'][0], fmt, tsrc('m4_calibration'), f'{pre} - none: {mk}, CI low')
                    put(f'{pre}_{mk}_hi', vs[f'{mk}_ci'][1], fmt, tsrc('m4_calibration'), f'{pre} - none: {mk}, CI high')
                put(f'{pre}_d_ll_p_holm', vs['p_ll_holm'], '.3f', tsrc('m4_calibration'), f'{pre} - none: log loss, Holm p')
            if c['calibration'] in ('none', 'nested'):
                for pl, v in c['per_platform'].items():
                    pp = f'{pre}_{pl}'
                    put(f'{pp}_n', int(v['n']), 'd', tsrc('m4_calibration'), f'{pre}: recordings on {pl}')
                    put(f'{pp}_acc', v['accuracy_half'], '.1%', tsrc('m4_calibration'), f'{pre}: accuracy on {pl}')
                    put(f'{pp}_acc_lo', v['accuracy_half_ci'][0], '.1%', tsrc('m4_calibration'), f'{pre}: {pl}, CI low')
                    put(f'{pp}_acc_hi', v['accuracy_half_ci'][1], '.1%', tsrc('m4_calibration'), f'{pre}: {pl}, CI high')
                    for mk, fmt in (('ece', '.3f'), ('ece_null_lo', '.3f'), ('ece_null_hi', '.3f'), ('log_loss', '.3f'),
                                    ('brier', '.3f'), ('slope', '.2f')):
                        put(f'{pp}_{mk}', v[mk], fmt, tsrc('m4_calibration'), f'{pre}: {mk} on {pl}')
                    put(f'{pp}_slope_lo', v['slope_ci'][0], '.2f', tsrc('m4_calibration'), f'{pre}: slope on {pl}, CI low')
                    put(f'{pp}_slope_hi', v['slope_ci'][1], '.2f', tsrc('m4_calibration'), f'{pre}: slope on {pl}, CI high')
        for part, pre0 in (('k3', 'k3'), ('stop', 'stopcal')):
            for how, v in (cal.get(part) or {}).items():
                pre = f'{pre0}_{how}'
                what = f"{'three accepted attempts' if part == 'k3' else 'stopping-rule answer'}, {how}"
                for mk, fmt in (('accuracy_half', '.1%'), ('log_loss', '.3f'), ('brier', '.3f'), ('ece', '.3f'),
                                ('ece_null_lo', '.3f'), ('ece_null_hi', '.3f'), ('ece_p_null', '.3f'), ('slope', '.2f'),
                                ('intercept', '.2f'), ('roc_auc', '.3f')):
                    put(f'{pre}_{mk}', v[mk], fmt, tsrc('m4_calibration'), f'{what}: {mk}')
                put(f'{pre}_slope_lo', v['slope_ci'][0], '.2f', tsrc('m4_calibration'), f'{what}: slope, CI low')
                put(f'{pre}_slope_hi', v['slope_ci'][1], '.2f', tsrc('m4_calibration'), f'{what}: slope, CI high')
                put(f'{pre}_calibrated', 'yes' if v['calibrated'] else 'no', '', tsrc('m4_calibration'),
                    f'{what}: consistent with calibration')
                if part == 'stop':
                    put(f'{pre}_attempts', v['mean_attempts'], '.2f', tsrc('m4_calibration'), f'{what}: mean accepted attempts')
                    put(f'{pre}_touches', v['mean_true_touches'], '.2f', tsrc('m4_calibration'),
                        f'{what}: mean true touches by the decision')
                    cf = v.get('confident')
                    if cf:
                        put(f'{pre}_conf_acc', cf['accuracy'], '.1%', tsrc('m4_calibration'), f'{what}: accuracy of confident stops')
                        put(f'{pre}_conf_acc_lo', cf['accuracy_ci'][0], '.1%', tsrc('m4_calibration'), f'{what}: confident, CI low')
                        put(f'{pre}_conf_acc_hi', cf['accuracy_ci'][1], '.1%', tsrc('m4_calibration'), f'{what}: confident, CI high')
                        put(f'{pre}_conf_claimed', cf['claimed'], '.1%', tsrc('m4_calibration'),
                            f'{what}: mean confidence claimed by the confident stops')
                        put(f'{pre}_conf_n', cf['n'], 'd', tsrc('m4_calibration'), f'{what}: confident stops')
        put('prod_aggregation', cal['production_aggregation'], '', tsrc('m4_calibration'),
            'production aggregation of attempts (declared rule m4_config.AGGREGATION_RULE)')
        for row in cal.get('stop_per_platform') or []:
            if row['aggregation'] != cal['production_aggregation']:
                continue
            pre = f"stopcal_{row['platform']}"
            what = f"stopping-rule answer, production aggregation, {row['platform']}"
            put(f'{pre}_n', int(row['n']), 'd', tsrc('m4_calibration'), f'{what}: sessions')
            put(f'{pre}_persons', int(row['n_persons']), 'd', tsrc('m4_calibration'), f'{what}: persons')
            put(f'{pre}_acc', row['accuracy_half'], '.1%', tsrc('m4_calibration'), f'{what}: accuracy')
            put(f'{pre}_attempts', row['mean_attempts'], '.2f', tsrc('m4_calibration'), f'{what}: mean attempts')
            for mk in ('ece', 'ece_null_lo', 'ece_null_hi', 'log_loss', 'brier'):
                put(f'{pre}_{mk}', row[mk], '.3f', tsrc('m4_calibration'), f'{what}: {mk}')
            put(f'{pre}_slope', row['slope'], '.2f', tsrc('m4_calibration'), f'{what}: slope')
    ST = _j('m4_stress')
    for row in ST.get('summary', []):
        kk = 'rec' if row['keep'] == 'recorded' else f"{float(row['keep']):g}".replace('.', 'p')
        pk = row['pipeline'].replace('v0', 'vzero')
        pre = f"stress{'flip' if row['flip'] else ''}_{pk}_{kk}"
        what = f"stress test{' (screen-away twin)' if row['flip'] else ''}: {row['pipeline']}, {row['keep']} s kept"
        put(f'{pre}_acc', row['accuracy'], '.1%', tsrc('m4_stress'), f'{what}: accuracy')
        put(f'{pre}_acc_lo', row['acc_lo'], '.1%', tsrc('m4_stress'), f'{what}: CI low')
        put(f'{pre}_acc_hi', row['acc_hi'], '.1%', tsrc('m4_stress'), f'{what}: CI high')
        put(f'{pre}_cov', row['coverage'], '.0%', tsrc('m4_stress'), f'{what}: share answered')
        put(f'{pre}_acc_answered', row['acc_answered'], '.1%', tsrc('m4_stress'), f'{what}: accuracy on the answered')
        put(f'{pre}_ret', row['share_return'], '.0%', tsrc('m4_stress'), f'{what}: share where the return is scored')
        put(f'{pre}_gate', row['gate_share'], '.1%', tsrc('m4_stress'),
            f'{what}: share stopped by the rest-pose grip check')
        put(f'{pre}_acc_gated', row['acc_gated'], '.1%', tsrc('m4_stress'), f'{what}: accuracy with the grip check')
        put(f'{pre}_acc_gated_lo', row['acc_gated_lo'], '.1%', tsrc('m4_stress'), f'{what}: with the check, CI low')
        put(f'{pre}_acc_gated_hi', row['acc_gated_hi'], '.1%', tsrc('m4_stress'), f'{what}: with the check, CI high')
        put(f'{pre}_cov_gated', row['coverage_gated'], '.0%', tsrc('m4_stress'), f'{what}: share answered with the check')
        put(f'{pre}_acc_answered_gated', row['acc_answered_gated'], '.1%', tsrc('m4_stress'),
            f'{what}: accuracy on the answered with the check')
        put(f'{pre}_ret_gated', row['share_return_gated'], '.1%', tsrc('m4_stress'),
            f'{what}: share where the return is scored and answered with the check')
        put(f'{pre}_wrong', row['wrong_share'], '.1%', tsrc('m4_stress'), f'{what}: share answered wrongly')
        put(f'{pre}_wrong_gated', row['wrong_share_gated'], '.1%', tsrc('m4_stress'),
            f'{what}: share answered wrongly with the grip check')
        put(f'{pre}_wrong_gated_seg', row['wrong_share_gated_seg'], '.1%', tsrc('m4_stress'),
            f'{what}: share answered wrongly with the first remedy')
        put(f'{pre}_gate_seg', row['gate_seg_share'], '.0%', tsrc('m4_stress'),
            f"{what}: share stopped by the first remedy (the chosen segment's own start pose)")
        put(f'{pre}_acc_gated_seg', row['acc_gated_seg'], '.1%', tsrc('m4_stress'), f'{what}: accuracy with the first remedy')
        put(f'{pre}_acc_answered_gated_seg', row['acc_answered_gated_seg'], '.1%', tsrc('m4_stress'),
            f'{what}: accuracy on the answered with the first remedy')
    for row in ST.get('pairs', []):
        kk = 'rec' if row['keep'] == 'recorded' else f"{float(row['keep']):g}".replace('.', 'p')
        pre = f"stresspair{'flip' if row['flip'] else ''}_{row['key']}_{kk}"
        what = f"stress pair {row['a']} - {row['b']}, {row['keep']} s{' (screen away)' if row['flip'] else ''}"
        put(f'{pre}_d', row['d_acc'], '+.1%', tsrc('m4_stress'), f'{what}: accuracy difference')
        put(f'{pre}_lo', row['d_acc_ci'][0], '+.1%', tsrc('m4_stress'), f'{what}: CI low')
        put(f'{pre}_hi', row['d_acc_ci'][1], '+.1%', tsrc('m4_stress'), f'{what}: CI high')
        put(f'{pre}_p_holm', row['p_acc_holm'], '.3f', tsrc('m4_stress'), f'{what}: Holm p (family stress)')
    if ST.get('pairs'):
        put('stress_family_n', len(ST['pairs']), 'd', tsrc('m4_stress'), 'comparisons in the Holm family stress')
    for row in ST.get('grip_check_wrong', []):
        kk = 'rec' if row['keep'] == 'recorded' else f"{float(row['keep']):g}".replace('.', 'p')
        pre = f"gripwrong{'flip' if row['flip'] else ''}_{kk}"
        what = (f"production, {row['keep']} s{' (screen-away twin)' if row['flip'] else ''}: share answered wrongly, "
                'with the grip check minus without')
        put(f'{pre}_without', row['b'], '.1%', tsrc('m4_stress'), f'{what}: without the check')
        put(f'{pre}_with', row['a'], '.1%', tsrc('m4_stress'), f'{what}: with the check')
        put(f'{pre}_d', row['d'], '+.1%', tsrc('m4_stress'), f'{what}: difference')
        put(f'{pre}_lo', row['d_ci'][0], '+.1%', tsrc('m4_stress'), f'{what}: CI low')
        put(f'{pre}_hi', row['d_ci'][1], '+.1%', tsrc('m4_stress'), f'{what}: CI high')
        put(f'{pre}_p_holm', row['p_holm'], '.4f', tsrc('m4_stress'), f'{what}: Holm p (family grip)')
    if ST.get('grip_check_wrong'):
        put('grip_family_n', len(ST['grip_check_wrong']), 'd', tsrc('m4_stress'), 'comparisons in the Holm family grip')
    S = _j('m4_sessions')
    if S:
        ref = float(S['reference_keep_s'])
        put('sess_ref_keep_s', ref, 'g', tsrc('m4_sessions'), 'junction stillness of the reported sessions (s)')
        put('sess_recordings', int(S['session_recordings']), 'd', tsrc('m4_sessions'), 'recordings per synthetic session')
        put('k_max', int(S['k_max']), 'd', tsrc('m4_sessions'), 'maximum accepted attempts of the stopping rule')
        put('p_stop', float(S['p_stop']), '.2f', tsrc('m4_sessions'), 'stopping threshold')
        put('prod_rule', S['production_rule'].replace('_', ' '), '', tsrc('m4_sessions'), 'production attempt rule')
        k = pd.DataFrame(S['k_curve'])
        km = k[k['aggregation'] == 'mean']
        prefix = {'pairing': 'sesspair', 'no_pairing': 'sessnp', 'true_only': 'sesstrue'}
        prefix[S['production_rule']] = 'sess'             # the production attempt rule: plain 'sess_' keys
        for _, row in km.iterrows():
            pre = prefix.get(row['rule'])
            if pre is None:
                continue
            kk = f"{row['keep_s']:g}".replace('.', 'p')
            what = f"sessions ({row['rule']}), {row['keep_s']:g} s junctions, up to {int(row['k'])} attempts"
            put(f"{pre}_{kk}_k{int(row['k'])}_acc", row['accuracy'], '.1%', tsrc('m4_sessions'), f'{what}: accuracy')
            if row['keep_s'] == ref:
                put(f"{pre}_{kk}_k{int(row['k'])}_acc_lo", row['acc_lo'], '.1%', tsrc('m4_sessions'), f'{what}: CI low')
                put(f"{pre}_{kk}_k{int(row['k'])}_acc_hi", row['acc_hi'], '.1%', tsrc('m4_sessions'), f'{what}: CI high')
                put(f"{pre}_{kk}_k{int(row['k'])}_avail", row['share_k_available'], '.0%', tsrc('m4_sessions'),
                    f'{what}: share of sessions with at least k accepted attempts')
        put('sess_n', int(km['n_sessions'].max()), 'd', tsrc('m4_sessions'), 'synthetic sessions per stillness setting')
        st = pd.DataFrame(S['stopping'])
        how = cal.get('production_aggregation') if cal else None
        g = st[np.isclose(st['p_stop'], S['p_stop']) & (st['rule'] == S['production_rule']) & (st['aggregation'] == how)]
        for _, row in g.iterrows():
            kk = f"{row['keep_s']:g}".replace('.', 'p')
            what = f"stopping rule (p_stop {S['p_stop']}, {how}), {row['keep_s']:g} s junctions"
            for mk, fmt, d in (('accuracy', '.1%', 'accuracy'), ('acc_lo', '.1%', 'CI low'), ('acc_hi', '.1%', 'CI high'),
                               ('mean_attempts', '.2f', 'mean accepted attempts'), ('mean_true_touches', '.2f',
                                                                                    'mean true touches by the decision'),
                               ('coverage', '.0%', 'share answered'), ('ece', '.3f', 'ECE'), ('roc_auc', '.3f', 'AUC'),
                               ('share_confident_stop', '.0%', 'share of confident stops'),
                               ('acc_confident_stop', '.1%', 'accuracy of confident stops'),
                               ('share_k_max', '.0%', 'share stopped at k_max'),
                               ('share_stream_end', '.0%', 'share ended with the stream'),
                               ('share_no_attempt', '.0%', 'share without any accepted attempt')):
                key = {'accuracy': 'acc', 'mean_attempts': 'attempts', 'mean_true_touches': 'touches',
                       'share_confident_stop': 'conf_share', 'acc_confident_stop': 'conf_acc', 'roc_auc': 'auc'}.get(mk, mk)
                put(f'stop_{kk}_{key}', row[mk], fmt, tsrc('m4_sessions'), f'{what}: {d}')
        cur = st[(st['rule'] == S['production_rule']) & (st['keep_s'] == ref) & (st['aggregation'] == how)]
        for _, row in cur.iterrows():
            kk = f"{row['p_stop']:.2f}".replace('.', 'p')
            what = f'stopping rule ({how}, {ref:g} s) at p_stop {row["p_stop"]:.2f}'
            put(f'pstop_{kk}_acc', row['accuracy'], '.1%', tsrc('m4_sessions'), f'{what}: accuracy')
            put(f'pstop_{kk}_attempts', row['mean_attempts'], '.2f', tsrc('m4_sessions'), f'{what}: mean attempts')
        for row in S.get('answer', []):
            if row['aggregation'] != how:
                continue
            kk = f"{row['p_answer']:.2f}".replace('.', 'p')
            what = f'stopping rule ({how}, {ref:g} s), answer after k_max only when max(P, 1 - P) >= {row["p_answer"]:.2f}'
            put(f'answer_{kk}_coverage', row['coverage'], '.0%', tsrc('m4_sessions'), f'{what}: share answered')
            put(f'answer_{kk}_acc', row['acc_answered'], '.1%', tsrc('m4_sessions'), f'{what}: accuracy on answered')
        for row in S.get('detection', []):
            if row['keep_s'] != ref:
                continue
            pre = 'sessdet' if row['rule'] == S['production_rule'] else 'sessdetalt'
            what = f"{row['rule'].replace('_', ' ')} rule, {ref:g} s"
            put(f'{pre}_recall', row['recall'], '.0%', tsrc('m4_sessions'), f'{what}: share of true approaches accepted')
            put(f'{pre}_precision', row['precision'], '.0%', tsrc('m4_sessions'),
                f'{what}: share of accepted attempts that are (part of) an approach')
            for mk in ('attempts', 'false_return', 'false_other', 'false_other_before', 'false_other_after', 'true', 'found'):
                put(f'{pre}_{mk}', int(row[mk]), 'd', tsrc('m4_sessions'), f'{what}: {mk}')
        for row in S.get('by_kind', []):
            put(f"kind_{row['kind']}_acc", row['accuracy'], '.0%', tsrc('m4_sessions'),
                f"single accepted attempts that are {row['kind']}: accuracy")
            put(f"kind_{row['kind']}_n", row['n'], 'd', tsrc('m4_sessions'), f"accepted attempts that are {row['kind']}")
            put(f"kind_{row['kind']}_share", row['share'], '.0%', tsrc('m4_sessions'), f"share of attempts that are {row['kind']}")
        for row in S.get('paired', []):
            pre = f"sp_{row['key']}"
            what = f"sessions, paired {row['a']} - {row['b']}"
            for mk, fmt in (('d_acc', '+.1%'), ('d_ll', '+.3f'), ('d_brier', '+.4f')):
                put(f'{pre}_{mk}', row[mk], fmt, tsrc('m4_sessions'), f'{what}: {mk}')
                put(f'{pre}_{mk}_lo', row[f'{mk}_ci'][0], fmt, tsrc('m4_sessions'), f'{what}: {mk}, CI low')
                put(f'{pre}_{mk}_hi', row[f'{mk}_ci'][1], fmt, tsrc('m4_sessions'), f'{what}: {mk}, CI high')
            put(f'{pre}_p_holm', row['p_acc_holm'], '.3f', tsrc('m4_sessions'), f'{what}: accuracy, Holm p (family sessions)')
            put(f'{pre}_ll_p_holm', row['p_ll_holm'], '.3f', tsrc('m4_sessions'), f'{what}: log loss, Holm p (family sessions)')
            put(f'{pre}_discordant', row['n_discordant'], 'd', tsrc('m4_sessions'), f'{what}: discordant sessions')
        if S.get('paired'):
            put('sess_family_n', len(S['paired']), 'd', tsrc('m4_sessions'), 'comparisons in the Holm family sessions')
        sc = pd.DataFrame(S.get('scales', []))
        for h in ('scale_recordings', 'scale_sessions'):
            for kk in (1, 3, 5):
                r_ = sc[(sc['aggregation'] == h) & (sc['k'] == kk)] if len(sc) else sc
                if len(r_):
                    put(f'{h}_s{kk}', float(r_['s_median'].iloc[0]), '.2f', tsrc('m4_sessions'),
                        f'aggregate scale s_{kk}, {h} (median over the folds)')
        fm = S.get('fold_medians', {})
        if 'power_exponent_b' in fm:
            put('scale_sessions_b', fm['power_exponent_b'], '.2f', tsrc('m4_sessions'), 'exponent b of s_k = s_1 k^b (median)')
        if 'crossfit_sessions' in fm:
            put('crossfit_sessions_n', int(round(fm['crossfit_sessions'])), 'd', tsrc('m4_sessions'),
                'cross-fitted training sessions with an attempt per fold (median)')
        put('per_k_nonmonotone_folds', int(S.get('per_k_nonmonotone_folds', 0)), 'd', tsrc('m4_sessions'),
            'folds whose per-k session scale decreases somewhere in k')
        put('n_folds_sessions', int(S.get('n_folds', 0)), 'd', tsrc('m4_sessions'), 'folds of the session analysis')
    LAT = _j('m4_latency')
    if LAT:
        for k_, q in (('end', 'decision_minus_oracle_end_ms'), ('touch', 'decision_minus_first_touch_ms')):
            put(f'lat_{k_}_ms', LAT[q]['median'], '.0f', tsrc('m4_latency'), f'median decision time after the {k_} (ms)')
            put(f'lat_{k_}_p90_ms', LAT[q]['p90'], '.0f', tsrc('m4_latency'), f'90th percentile, after the {k_} (ms)')
        put('causal_delay_ms', LAT['causal_delay_ms'], '.0f', tsrc('m4_latency'), 'group delay of the causal activity filter (ms)')
        put('lookahead_s', float(next(iter(LAT['lookahead_s'].values()))), 'g', tsrc('m4_latency'),
            'look-ahead after the end of a segment (s)')
        rw = LAT.get('return_end_minus_approach_end_ms')
        if rw:
            put('return_wait_ms', rw['median'], '.0f', tsrc('m4_latency'), 'median stored end of the return minus end of the approach (ms)')
            put('return_wait_p90_ms', rw['p90'], '.0f', tsrc('m4_latency'), '90th percentile of the same (ms)')
    SF = _j('m4_segfeat')
    for row in SF.get('auc', []):
        if row['feature'] in ('g_disp_z0', 'g_d_up_z', 'disp_z0', 'd_up_z', 'up_z0', 'still_before', 'still_before_inst',
                              'disp_vert', 'v_up_peak', 'g_up_z0'):
            put(f"auc_{row['feature']}_vs_return", row['auc_vs_return'], '.2f', tsrc('m4_segfeat'), f"AUC of {row['feature']}: approach vs return")
            put(f"auc_{row['feature']}_vs_other", row['auc_vs_other'], '.2f', tsrc('m4_segfeat'), f"AUC of {row['feature']}: approach vs other")
    for lab, v in (SF.get('share_grip_sign_negative') or {}).items():
        put(f'grip_sign_neg_{lab}', v, '.1%', tsrc('m4_segfeat'), f'share of {lab} segments whose grip sign s is -1')
    NA = _j('m4_natural')
    if NA:
        t = NA['rejection_test']
        put('nat_rejected', t['natural_rejected'], 'd', tsrc('m4_natural'), 'natural-use recordings rejected (no approach accepted)')
        put('nat_n', t['n_natural'], 'd', tsrc('m4_natural'), 'natural-use recordings')
        put('nat_rejection_rate', t['natural_rejection_rate'], '.0%', tsrc('m4_natural'), 'natural-use rejection rate')
        put('prim_rejection_rate', t['primary_rejection_rate'], '.0%', tsrc('m4_natural'), 'primary rejection rate')
        put('nat_expected_rejections', t['expected_from_persons'], '.1f', tsrc('m4_natural'),
            "rejections expected from each person's own primary rejection rate")
        put('nat_p_poisson', t['p_poisson_binomial'], '.4f', tsrc('m4_natural'), 'Poisson-binomial P(X >= observed)')
        put('nat_p_fisher', t['p_fisher'], '.4f', tsrc('m4_natural'), 'Fisher exact test, natural vs primary rejection')
        put('nat_approach_found', NA['natural_approach_candidate_found'], '.0%', tsrc('m4_natural'),
            'natural use: share with a candidate segment that is the stored approach')
        put('prim_approach_found', NA['primary_approach_candidate_found'], '.0%', tsrc('m4_natural'),
            'primary: share with a candidate segment that is the stored approach')
        put('nat_approach_p_below', NA['natural_approach_p_below_threshold'], '.0%', tsrc('m4_natural'),
            'natural use: share of stored-approach candidates with P(approach) below the threshold')
        put('prim_approach_p_below', NA['primary_approach_p_below_threshold'], '.0%', tsrc('m4_natural'),
            'primary: share of stored-approach candidates with P(approach) below the threshold')
        for set_name, v_ in (NA.get('grip_gate_share') or {}).items():
            put(f"{'nat' if set_name == 'natural_use' else 'prim'}_grip_gate", v_, '.0%', tsrc('m4_natural'),
                f'{set_name}: share of stored approaches that start with the screen facing down (grip gate fires)')
        for row in NA['features']:
            f = row['feature']
            put(f'natfeat_{f}_nat', row['natural_median'], '.2f', tsrc('m4_natural'), f'{f}: natural-use median (stored approach)')
            put(f'natfeat_{f}_prim', row['primary_median'], '.2f', tsrc('m4_natural'), f'{f}: primary median (stored approach)')
            put(f'natfeat_{f}_below', row['natural_share_below_primary_p25'], '.0%', tsrc('m4_natural'),
                f'{f}: share of natural approaches below the primary 25th percentile')
    RP = _j('m4_replay')
    if RP:
        put('replay_n', RP['n'], 'd', tsrc('m4_replay'), 'recordings replayed')
        put('replay_max_dp', RP['max_abs_dp'], '.3f', tsrc('m4_replay'), 'buffer replay: max |P replay - P offline|')
        put('replay_median_dp', RP['median_abs_dp'], '.4f', tsrc('m4_replay'), 'buffer replay: median |dP|')
        put('replay_changed', RP['share_decision_changed'], '.1%', tsrc('m4_replay'), 'buffer replay: share of decisions changed')
        put('replay_status_changed', RP['share_status_changed'], '.1%', tsrc('m4_replay'), 'buffer replay: share of statuses changed')
        put('replay_matched', RP['segments_matched'], '.1%', tsrc('m4_replay'), 'buffer replay: segments found again')
        put('replay_full_check', RP['full_buffer_reproduces_cache_max_abs_gyro'], '.2g', tsrc('m4_replay'),
            'max |gyro| difference between the full-buffer replay and the cached stream (deg/s)')
        pr = RP['paired_replay_minus_offline']
        put('replay_d_acc', pr['d_acc'], '+.1%', tsrc('m4_replay'), 'buffer replay - offline: accuracy')
        put('replay_d_acc_lo', pr['d_acc_ci'][0], '+.1%', tsrc('m4_replay'), 'same, CI low')
        put('replay_d_acc_hi', pr['d_acc_ci'][1], '+.1%', tsrc('m4_replay'), 'same, CI high')
    for r_ in _j('m4_scores').get('det_threshold', []):
        kk = f"{r_['threshold']:.2f}".replace('.', 'p')
        put(f'thr_{kk}_acc', r_['accuracy'], '.1%', tsrc('m4_scores'), f"P(approach) threshold {r_['threshold']:.2f}: accuracy")
        put(f'thr_{kk}_coverage', r_['coverage'], '.0%', tsrc('m4_scores'), f"P(approach) threshold {r_['threshold']:.2f}: answered")
        if r_.get('vs_production'):
            v = r_['vs_production']
            put(f'thr_{kk}_d', v['d_acc'], '+.1%', tsrc('m4_scores'), f"threshold {r_['threshold']:.2f} - production: accuracy")
            put(f'thr_{kk}_d_lo', v['d_acc_ci'][0], '+.1%', tsrc('m4_scores'), 'same, CI low')
            put(f'thr_{kk}_d_hi', v['d_acc_ci'][1], '+.1%', tsrc('m4_scores'), 'same, CI high')
    det_p = L.tab_dir / f'{prod}_details.csv'
    if det_p.exists():
        from stage2.lanes import m4_stats as STS
        from stage2.lanes.m4_config import ABSTAIN_TAU
        dp = pd.read_csv(det_p)
        dp = dp[dp['set'] == 'primary']
        ok = dp['status'] == 'ok'
        conf = np.maximum(dp['p_forced'], 1 - dp['p_forced'])
        keep = ok & (conf >= ABSTAIN_TAU)
        put('abstain_tau', ABSTAIN_TAU, '.2f', 'stage2/lanes/m4_config.py', 'single-touch confidence threshold')
        put('abstain_coverage', float(keep.mean()), '.0%', f'tables/m4/{prod}_details.csv',
            'share answered with the confidence threshold (and an approach found)')
        put('abstain_acc', STS.half_credit(dp.loc[keep, 'y'], dp.loc[keep, 'p_right']) if keep.any() else None, '.1%',
            f'tables/m4/{prod}_details.csv', 'accuracy on those answered')
    DM = _j('m4_demo')
    if DM:
        put('demo_rows', len(DM.get('rows', [])), 'd', tsrc('m4_demo'), 'streams in the predict_stream demo')
        put('demo_trained_persons', int(DM['model']['n_persons_trained']), 'd', tsrc('m4_demo'),
            'persons the demo model was trained on')
        tw = [r for r in DM.get('rows', []) if 'screen-away' in r.get('what', '')]
        if tw:
            put('demo_twin_status', str(tw[0]['status']), '', tsrc('m4_demo'),
                'predict_stream status of the screen-away twin of a held-out recording')
    L.write_numbers(n)
    return n


# ───────────────────────────── summary ─────────────────────────────


def _v(n: dict[str, Any], k: str) -> str:
    e = n.get(k)
    if e is None:
        return '--'
    return RX.format_value(e['value'], e['format'])


def _val(n: dict[str, Any], k: str) -> Any:
    e = n.get(k)
    return None if e is None else e['value']


def pair_verdict(n: dict[str, Any], key: str, a: str, b: str, *, prefix: str = 'd_', p_suffix: str = '_p_holm',
                 lo: str = '_lo', hi: str = '_hi', metric: str = 'accuracy', alpha: float = 0.05) -> str:
    """Generated verdict of a saved paired comparison: identical / A or B better (Holm p < alpha and the CI
    excludes 0) / no difference shown."""
    if _val(n, f'{prefix}{key}_identical') == 'yes':
        return f'identical predictions ({a} = {b})'
    p, l_, h_ = _val(n, f'{prefix}{key}{p_suffix}'), _val(n, f'{prefix}{key}{lo}'), _val(n, f'{prefix}{key}{hi}')
    if not all(_ok(x) for x in (p, l_, h_)):
        return 'not computed'
    if p < alpha and (l_ > 0 or h_ < 0):
        return f'{a if l_ > 0 else b} has the higher {metric}, Holm p = {p:.3f}'
    return f'no {metric} difference shown, Holm p = {p:.2f}'


def write_summary() -> None:
    """reports/stage2/m4_SUMMARY.md from the numbers file and the saved tables (regenerated by `numbers`)."""
    from stage2.lanes import m4_summary
    n = json.loads(L.numbers_path.read_text(encoding='utf-8')) if L.numbers_path.exists() else numbers()
    text = m4_summary.render(n)
    L.summary_path.write_text(text, encoding='utf-8', newline='\n')
