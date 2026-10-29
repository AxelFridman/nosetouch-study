"""Lane x1 numbers, finalists and summary, from saved files only (no fitting, no raw data).

    results/x1_numbers.json      every number the report, the summary and the figure texts cite (macros \\XOne<Key>)
    results/finalists_x1.json    [] - this lane has no model (PD01 is never scored)
    reports/stage2/x1_SUMMARY.md the lane summary, regenerated from the saved results

Every number in the summary and in the figure texts (``x1_texts``) is read from the numbers entries built here
(:func:`numbers`), and every sentence that states a result is generated from them, so both stay true after a rerun
(for example with another reference person).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from stage2 import report_export as RX
from stage2.pipeline import Lane, selected
from stage2.lanes import x1_features as XF
from stage2.lanes import x1_stats as XS

L = Lane('x1')
SRC_MAIN = 'results/x1_characteristics.json'
SRC_MV = 'results/x1_multivariate.json'
SRC_TASK = 'tables/x1/x1_task.json'
SRC_CHECKS = 'tables/x1/x1_checks.json'
SRC_TRIALS = 'tables/x1/x1_trial_characteristics.csv'
SRC_CMP = 'tables/x1/x1_comparisons.csv'
SRC_SPEC = 'tables/x1/x1_spectra.csv'
PLATFORM = {'android': 'Android', 'ios': 'iOS'}
#: Session status codes in plain words.
STATUS_WORDS = {'ended_early': 'ended early', 'in_progress': 'abandoned (left in progress)', 'completed': 'completed',
                'abandoned': 'abandoned'}
#: Spectral bands of the spectrum-ratio numbers (case mean over the reference median).
SPEC_BANDS = {'05_3': (0.5, 3.0), '3_7': (3.0, 7.0), '7_12': (7.0, 12.0), '12_15': (12.0, 15.0)}
SPEC_WINDOWS = {'pre': 'pre', 'reach': 'approach', 'nose': 'nose'}
VARIANT_KEYS = {'first_k': 'firstk', 'same_protocol': 'protocol', 'no_clip': 'noclip', 'same_posture': 'posture',
                'same_platform': 'platform'}
NULL_SETS = {'pre': 'prespecified', 'all': 'all'}
NULL_VARIANTS = {'lfl': 'like_for_like', 'main': 'main'}
#: Characteristics a reader would expect to differ with Parkinson's disease (rest tremor, smoothness, slowness, the
#: hand signature); the summary says which of them do NOT differ (decided by the data, not here).
EXPECTED_TO_DIFFER = ('pre_rms_3_7', 'nose_rms_3_7', 'sparc', 'ldlj', 'hesitations', 'duration_s', 'rt_s', 'fhy_hand')


# ───────────────────────────── inputs ─────────────────────────────


def load() -> dict[str, Any]:
    ch = L.read_json('x1_characteristics', folder='results')
    t = pd.read_csv(L.tab_dir / 'x1_trial_characteristics.csv')
    t['person_id'] = t['person_id'].astype(str)
    cmp = pd.read_csv(L.tab_dir / 'x1_comparisons.csv')
    return {'main': pd.DataFrame(ch['rows']), 'case': ch['case'], 'strata': ch['strata'], 'null': ch['null'],
            'redundancy': pd.DataFrame(ch['redundancy']), 'settings': ch['settings'],
            'mv': L.read_json('x1_multivariate', folder='results'), 'task': L.read_json('x1_task'),
            'checks': L.read_json('x1_checks'), 'trials': t, 'cmp': cmp,
            'spectra': pd.read_csv(L.tab_dir / 'x1_spectra.csv'), 'traces': L.read_json('x1_traces')}


def _slug(text: str) -> str:
    import re
    s = re.sub(r'[^a-z]+', '_', text.lower()).strip('_')
    return {'failed_no_movement_within_the_time_limit_after_go': 'no_movement',
            'failed_still_moving_at_the_time_limit': 'too_long'}.get(s, s[:30])


def pfmt(p: Any) -> str:
    """The format of a p-value entry: three decimals, or scientific below 0.001 (never a misleading 0.000)."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return '.3f'
    return '.1e' if math.isfinite(v) and 0 < v < 0.001 else '.3f'


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _i(v: Any) -> int | None:
    x = _f(v)
    return None if math.isnan(x) else int(round(x))


# ───────────────────────────── numbers ─────────────────────────────


def numbers(D: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every number the summary, the figure texts and the report cite."""
    E = RX.numbers_entry
    m, mv, task, t, chk, cmp = D['main'], D['mv'], D['task'], D['trials'], D['checks'], D['cmp']
    case = D['case']
    out: dict[str, dict[str, Any]] = {}
    for _, r in m.iterrows():
        k, f, lab = r['key'], r['fmt'], r['label']
        c = XS.BY_KEY[k]
        ok_z = bool(r['report_z'])
        out[f'{k}_pd'] = E(_f(r['pd_value']), f, SRC_MAIN, f'{case}: {lab} ({r["stratum"]})')
        for s, col, what in (('med', 'ref_median', 'median'), ('qone', 'ref_q1', '25th percentile'),
                             ('qthree', 'ref_q3', '75th percentile'), ('min', 'ref_min', 'minimum'),
                             ('max', 'ref_max', 'maximum')):
            out[f'{k}_{s}'] = E(_f(r[col]), f, SRC_MAIN, f'reference {what}: {lab}')
        out[f'{k}_pct'] = E(_f(r['percentile']), '.0f', SRC_MAIN, f'mid-rank percentile of the case: {lab}')
        if c.scale == 'log' and _f(r['pd_value']) > 0 and _f(r['ref_median']) > 0:     # a ratio-scale quantity
            out[f'{k}_x_med'] = E(_f(r['pd_value']) / _f(r['ref_median']), '.1f', SRC_MAIN,
                                  f'case value / reference median ("n times the typical value"): {lab}')
        out[f'{k}_z'] = E(_f(r['robust_z']) if ok_z else None, '+.1f', SRC_MAIN,
                          f'robust z on the declared ({r["z_scale"]}) scale: {lab}')
        out[f'{k}_zlin'] = E(_f(r['robust_z_linear']) if ok_z else None, '+.1f', SRC_MAIN, f'robust z, linear: {lab}')
        out[f'{k}_zloo'] = E(_f(r['z_loo_min']) if ok_z else None, '+.1f', SRC_MAIN,
                             f'smallest |robust z| with one reference person left out: {lab}')
        out[f'{k}_zscale'] = E(str(r['z_scale']) if isinstance(r['z_scale'], str) else '', '', SRC_MAIN,
                               f'scale of the robust z: {lab}')
        out[f'{k}_p'] = E(_f(r['p_rank']), '.3f', SRC_MAIN, f'two-sided rank p: {lab}')
        out[f'{k}_q'] = E(_f(r['q_rank']), '.2f', SRC_MAIN, f'BH q of the rank p (pre-specified family): {lab}')
        out[f'{k}_pch'] = E(_f(r['ch_p']), pfmt(r['ch_p']), SRC_MAIN, f'Crawford-Howell p ({r["ch_scale"]}): {lab}')
        out[f'{k}_pchalt'] = E(_f(r['ch_p_alt']), pfmt(r['ch_p_alt']), SRC_MAIN,
                               f'Crawford-Howell p on the other scale: {lab}')
        out[f'{k}_qch'] = E(_f(r['q_ch']), '.3f', SRC_MAIN, f'BH q of the Crawford-Howell p (pre-specified family): {lab}')
        out[f'{k}_nref'] = E(_i(r['n_ref']), 'd', SRC_MAIN, f'reference persons compared: {lab}')
        out[f'{k}_ncase'] = E(_i(r['n_case_trials']), 'd', SRC_MAIN, f'case trials averaged: {lab}')
        out[f'{k}_ntrialsbeyond'] = E(_i(r['n_case_trials_beyond']) if c.level == 'trial' else None, 'd', SRC_MAIN,
                                      f'case trials beyond the reference persons\' range: {lab}')
        out[f'{k}_ntrialsvalid'] = E(_i(r['n_case_trials_valid']) if c.level == 'trial' else None, 'd', SRC_MAIN,
                                     f'case trials with a value: {lab}')
        out[f'{k}_nmore'] = E(_i(r['n_more_extreme']), 'd', SRC_MAIN,
                              f'reference persons at least as extreme as the case: {lab}')
        out[f'{k}_beyond'] = E(int(bool(r['beyond_all'])), 'd', SRC_MAIN, f'1 = beyond every reference person: {lab}')
        out[f'{k}_above'] = E(int(r['side'] == 'above'), 'd', SRC_MAIN, f'1 = above the reference median: {lab}')
        out[f'{k}_robust'] = E(int(bool(r['robust'])), 'd', SRC_MAIN,
                               f'1 = beyond all and robust to every check that can fail: {lab}')
        out[f'{k}_failed'] = E(str(r['checks_failed']) if isinstance(r['checks_failed'], str) else '', '', SRC_MAIN,
                               f'checks a "beyond all" verdict failed: {lab}')
        un = r.get('checks_unavailable')
        out[f'{k}_unavailable'] = E(str(un) if isinstance(un, str) else '', '', SRC_MAIN,
                                    f'checks that could not be made (too few case trials left): {lab}')
        for var, vk in VARIANT_KEYS.items():
            row = cmp[(cmp['key'] == k) & cmp['primary'].astype(bool) & (cmp['variant'] == var)]
            rr = row.iloc[0] if len(row) else None
            ext = math.nan
            if rr is not None:
                ext = _f(rr['ref_max']) if r['side'] == 'above' else _f(rr['ref_min'])
            out[f'{k}_pd{vk}'] = E(_f(rr['pd_value']) if rr is not None else None, f, SRC_CMP,
                                   f'case value ({XS.VARIANT_LABELS[var]}): {lab}')
            out[f'{k}_ext{vk}'] = E(ext, f, SRC_CMP, f'reference extreme on the case\'s side ({XS.VARIANT_LABELS[var]}): {lab}')
            out[f'{k}_pct{vk}'] = E(_f(rr['percentile']) if rr is not None else None, '.0f', SRC_CMP,
                                    f'percentile ({XS.VARIANT_LABELS[var]}): {lab}')
            out[f'{k}_nref{vk}'] = E(_i(rr['n_ref']) if rr is not None else None, 'd', SRC_CMP,
                                     f'reference persons ({XS.VARIANT_LABELS[var]}): {lab}')
    for k in XS.REPEAT_KEYS:                       # the secondary, hand-mixed reference of the across-repeat measures
        row = cmp[(cmp['key'] == k) & (cmp['stratum'] == XS.REPEAT_SECONDARY) & (cmp['variant'] == 'main')]
        rr = row.iloc[0] if len(row) else None
        out[f'{k}_pctboth'] = E(_f(rr['percentile']) if rr is not None else None, '.0f', SRC_CMP,
                                f'percentile against the hand-mixed upright natural reaches: {XS.BY_KEY[k].label}')
        out[f'{k}_nrefboth'] = E(_i(rr['n_ref']) if rr is not None else None, 'd', SRC_CMP,
                                 f'reference persons with enough hand-mixed reaches: {XS.BY_KEY[k].label}')
    pre = m[~m['post_hoc'].astype(bool)]
    post = m[m['post_hoc'].astype(bool)]
    n = int(m['n_ref'].max())
    S = D['settings']
    out |= {
        'n_ref_persons': E(n, 'd', SRC_MAIN, 'modelling persons in the reference'),
        'n_ref_minus_one': E(n - 1, 'd', SRC_MAIN, 'reference persons each reference person is compared with'),
        'n_ref_plus_one': E(n + 1, 'd', SRC_MAIN, 'persons ranked (reference persons and the case)'),
        'p_floor': E(2.0 / (n + 1), '.3f', SRC_MAIN, 'smallest possible two-sided rank p with n reference persons'),
        'p_floor_one_sided': E(1.0 / (n + 1), '.3f', SRC_MAIN, 'smallest possible one-sided rank p'),
        'min_ref': E(S['min_ref'], 'd', SRC_MAIN, 'fewest reference persons for a statistic'),
        'z_clear': E(S['z_clear'], '.0f', SRC_MAIN, '|robust z| of the size check'),
        'redundant_rho': E(S['redundant_rho'], '.1f', SRC_MAIN, '|Spearman rho| above which two findings overlap'),
        'n_characteristics': E(len(m), 'd', SRC_MAIN, 'characteristics compared'),
        'n_prespecified': E(len(pre), 'd', SRC_MAIN, 'pre-specified characteristics (the multiplicity family)'),
        'n_post_hoc': E(len(post), 'd', SRC_MAIN, 'characteristics defined after inspecting the case'),
        'n_phenomena': E(int(m['phenomenon'].nunique()), 'd', SRC_MAIN, 'phenomena the characteristics measure'),
        'n_beyond_prespecified': E(int(pre['beyond_all'].sum()), 'd', SRC_MAIN,
                                   'pre-specified characteristics with the case beyond every reference person'),
        'n_beyond_post_hoc': E(int(post['beyond_all'].sum()), 'd', SRC_MAIN,
                               'post hoc characteristics with the case beyond every reference person'),
        'n_robust_prespecified': E(int(pre['robust'].sum()), 'd', SRC_MAIN,
                                   'pre-specified characteristics beyond all and robust to every check'),
        'n_robust_post_hoc': E(int(post['robust'].sum()), 'd', SRC_MAIN,
                               'post hoc characteristics beyond all and robust to every check'),
        'n_phen_beyond_prespecified': E(int(pre.loc[pre['beyond_all'], 'phenomenon'].nunique()), 'd', SRC_MAIN,
                                        'phenomena with a pre-specified characteristic beyond all'),
        'n_phen_robust_prespecified': E(int(pre.loc[pre['robust'], 'phenomenon'].nunique()), 'd', SRC_MAIN,
                                        'phenomena with a robust pre-specified characteristic'),
        'min_q_rank': E(float(pre['q_rank'].min()), '.2f', SRC_MAIN, 'smallest BH q of the rank tests (pre-specified)'),
        'min_q_ch': E(float(pre['q_ch'].min()), '.2f', SRC_MAIN,
                      'smallest BH q of the Crawford-Howell tests (pre-specified)'),
        'n_q_rank_below_five': E(int((pre['q_rank'] < 0.05).sum()), 'd', SRC_MAIN,
                                 'pre-specified characteristics with a rank q below 0.05'),
        'n_q_ch_below_five': E(int((pre['q_ch'] < 0.05).sum()), 'd', SRC_MAIN,
                               'pre-specified characteristics with a Crawford-Howell q below 0.05'),
    }
    exp = pre[pre['key'].isin(EXPECTED_TO_DIFFER)]
    out['n_expected'] = E(int(len(exp)), 'd', SRC_MAIN, 'declared characteristics a reader would expect to differ '
                                                         '(rest-tremor band, smoothness, slowness, the hand cue)')
    out['n_expected_inside'] = E(int((~exp['beyond_all'].astype(bool)).sum()), 'd', SRC_MAIN,
                                 'of them, characteristics in which the case is NOT beyond every reference person '
                                 '(the lower rows of X1_10)')
    pmin = pre.loc[pre['ch_p'].idxmin()]
    out['min_p_ch_prespecified'] = E(float(pmin['ch_p']), pfmt(pmin['ch_p']), SRC_MAIN,
                                     'smallest Crawford-Howell p of a pre-specified characteristic')
    out['min_p_ch_prespecified_label'] = E(XS.lc(pmin['label']), '', SRC_MAIN, 'its characteristic')
    if len(post):
        qmin = post.loc[post['ch_p'].idxmin()]
        out['posthoc_min_pch'] = E(float(qmin['ch_p']), pfmt(qmin['ch_p']), SRC_MAIN,
                                   'smallest Crawford-Howell p of a post hoc characteristic (not a valid test)')
        out['posthoc_min_pch_alt'] = E(_f(qmin['ch_p_alt']), pfmt(qmin['ch_p_alt']), SRC_MAIN,
                                       'the same on the other scale')
        out['posthoc_min_pch_scale'] = E(str(qmin['ch_scale']), '', SRC_MAIN, 'its scale')
        out['posthoc_min_pch_label'] = E(XS.lc(qmin['label']), '', SRC_MAIN, 'its characteristic')
    # the null reference of the "beyond all" count
    for sk, sname in NULL_SETS.items():
        for vk, vname in NULL_VARIANTS.items():
            nn = D['null'][sname][vname]
            pp = pd.DataFrame(nn['per_person'])
            rp = pp[pp['group'] == 'reference']
            top = rp.sort_values(['n_characteristics', 'n_phenomena', 'person_id'], ascending=[False, False, True])
            pre_ = f'null_{sk}_{vk}'
            desc = f'"beyond all" count, {sname} characteristics, {vname.replace("_", " ")}'
            out |= {
                f'{pre_}_case': E(nn['case_n_characteristics'], 'd', SRC_MAIN, f'{desc}: the case'),
                f'{pre_}_case_phen': E(nn['case_n_phenomena'], 'd', SRC_MAIN, f'{desc}: the case, phenomena'),
                f'{pre_}_ref_median': E(float(rp['n_characteristics'].median()), '.1f', SRC_MAIN,
                                        f'{desc}: reference median'),
                f'{pre_}_ref_max': E(int(rp['n_characteristics'].max()), 'd', SRC_MAIN, f'{desc}: reference maximum'),
                f'{pre_}_ref_min': E(int(rp['n_characteristics'].min()), 'd', SRC_MAIN, f'{desc}: reference minimum'),
                f'{pre_}_ref_max_phen': E(int(rp['n_phenomena'].max()), 'd', SRC_MAIN,
                                          f'{desc}: reference maximum, phenomena'),
                f'{pre_}_n_ge': E(nn['n_ref_ge_case_characteristics'], 'd', SRC_MAIN,
                                  f'{desc}: reference persons reaching the case\'s count'),
                f'{pre_}_n_ge_phen': E(nn['n_ref_ge_case_phenomena'], 'd', SRC_MAIN,
                                       f'{desc}: reference persons reaching the case\'s phenomena'),
                f'{pre_}_p': E(nn['p_rank_characteristics'], '.3f', SRC_MAIN, f'{desc}: rank p of the case'),
                f'{pre_}_p_phen': E(nn['p_rank_phenomena'], '.3f', SRC_MAIN, f'{desc}: rank p, phenomena'),
                f'{pre_}_expected': E(nn['expected_case_n_characteristics'], '.1f', SRC_MAIN,
                                      f'{desc}: expected count of the case under exchangeability'),
                f'{pre_}_top': E(str(top['person_id'].iloc[0]), '', SRC_MAIN, f'{desc}: reference person with the most'),
                f'{pre_}_top_n': E(int(top['n_characteristics'].iloc[0]), 'd', SRC_MAIN, f'{desc}: their count'),
                f'{pre_}_subset_min': E(nn['case_subset_n_characteristics']['min'], 'd', SRC_MAIN,
                                        f'{desc}: the case against each (n - 1)-subset, minimum'),
                f'{pre_}_subset_max': E(nn['case_subset_n_characteristics']['max'], 'd', SRC_MAIN,
                                        f'{desc}: the case against each (n - 1)-subset, maximum'),
            }
            ge = rp[rp['n_characteristics'] >= nn['case_n_characteristics']]['person_id'].tolist()
            out[f'{pre_}_ge_persons'] = E(', '.join(ge), '', SRC_MAIN, f'{desc}: reference persons reaching the case')
    # redundancy of the case's "beyond all" findings
    red = D['redundancy']
    if len(red):
        cross = red[red['redundant'] & ~red['same_phenomenon']]
        out['n_redundant_cross'] = E(int(len(cross)), 'd', SRC_MAIN,
                                     'pairs of "beyond all" characteristics of different phenomena with |rho| >= threshold')
        for i, (_, rr) in enumerate(cross.sort_values('rho', key=abs, ascending=False).iterrows(), start=1):
            out[f'redundant_{i}_rho'] = E(float(rr['rho']), '+.2f', SRC_MAIN,
                                          f'Spearman rho across reference persons: {rr["a"]} vs {rr["b"]}')
            out[f'redundant_{i}_pair'] = E(f'{XS.lc(XS.BY_KEY[rr["a"]].label)} and {XS.lc(XS.BY_KEY[rr["b"]].label)}',
                                           '', SRC_MAIN, 'the pair')
        same = red[red['same_phenomenon']]
        for _, rr in same.iterrows():
            out[f'rho_{rr["a"]}_{rr["b"]}'] = E(_f(rr['rho']), '+.2f', SRC_MAIN,
                                                 f'Spearman rho across reference persons: {rr["a"]} vs {rr["b"]}')
    # multivariate
    ev = np.asarray(mv['explained_variance'])
    sc = np.asarray(mv['scores_ref'], dtype=float)
    loco = pd.DataFrame(mv['loco'])
    lloco = pd.DataFrame(mv['linear']['loco'])
    worst, lworst = loco.loc[loco['n_ref_beyond'].idxmax()], lloco.loc[lloco['n_ref_beyond'].idxmax()]
    out |= {
        'mv_n_keys': E(len(mv['keys']), 'd', SRC_MV, 'key characteristics in the multivariate view'),
        'mv_n_log': E(int(sum(s == 'log' for s in mv['scales'])), 'd', SRC_MV, 'of them on the log scale'),
        'mv_n_missing': E(int(mv['n_missing_imputed']), 'd', SRC_MV, 'missing reference values set to the median'),
        'mv_distance': E(mv['mahalanobis_x'], '.2f', SRC_MV, 'Mahalanobis distance of the case (declared scales)'),
        'mv_p': E(mv['p_mahalanobis'], '.3f', SRC_MV, 'rank p of the case distance (declared scales)'),
        'mv_p_floor': E(mv['p_floor'], '.3f', SRC_MV, 'smallest possible rank p of the distance'),
        'mv_n_beyond': E(mv['n_ref_beyond'], 'd', SRC_MV, 'reference persons at least as far as the case'),
        'mv_pc_one_var': E(float(ev[0]), '.0%', SRC_MV, 'variance explained by PC1'),
        'mv_pc_two_var': E(float(ev[1]), '.0%', SRC_MV, 'variance explained by PC2'),
        'mv_pc_one_case': E(float(mv['score_x'][0]), '.1f', SRC_MV, 'PC1 score of the case'),
        'mv_pc_one_ref_min': E(float(sc[:, 0].min()), '.1f', SRC_MV, 'smallest reference PC1 score'),
        'mv_pc_one_ref_max': E(float(sc[:, 0].max()), '.1f', SRC_MV, 'largest reference PC1 score'),
        'mv_loco_min_beyond': E(int(loco['n_ref_beyond'].min()), 'd', SRC_MV,
                                'fewest reference persons as far, one characteristic left out'),
        'mv_loco_max_beyond': E(int(worst['n_ref_beyond']), 'd', SRC_MV,
                                'most reference persons as far, one characteristic left out'),
        'mv_loco_max_label': E(XS.lc(XS.BY_KEY[worst['left_out']].label), '', SRC_MV, 'the characteristic left out'),
        'mv_lin_distance': E(mv['linear']['mahalanobis_x'], '.2f', SRC_MV, 'Mahalanobis distance, all linear'),
        'mv_lin_p': E(mv['linear']['p_mahalanobis'], '.3f', SRC_MV, 'its rank p'),
        'mv_lin_n_beyond': E(mv['linear']['n_ref_beyond'], 'd', SRC_MV, 'reference persons as far, all linear'),
        'mv_lin_loco_max_beyond': E(int(lworst['n_ref_beyond']), 'd', SRC_MV,
                                    'most reference persons as far, all linear, one characteristic left out'),
        'mv_lin_loco_max_label': E(XS.lc(XS.BY_KEY[lworst['left_out']].label), '', SRC_MV,
                                   'the characteristic left out'),
    }
    # task behaviour
    c, rref = task['case'], task['reference']
    out |= {
        'case_age_band': E(c['age_band'], '', SRC_TASK, 'age band of the case (decade)'),
        'ref_age_min': E(rref['age_min'], '.0f', SRC_TASK, 'youngest reference person (years)'),
        'ref_age_max': E(rref['age_max'], '.0f', SRC_TASK, 'oldest reference person (years)'),
        'case_n_sessions': E(c['n_sessions'], 'd', SRC_TASK, 'sessions of the case'),
        'case_n_attempts': E(c['n_attempts'], 'd', SRC_TASK, 'attempts of the case'),
        'case_n_accepted': E(c['n_accepted'], 'd', SRC_TASK, 'accepted attempts of the case'),
        'case_n_rejected': E(c['n_rejected'], 'd', SRC_TASK, 'rejected attempts of the case'),
        'case_n_primary': E(c['n_primary'], 'd', SRC_TASK, 'instructed (primary-type) trials of the case'),
        'case_n_free': E(c['n_free_choice'], 'd', SRC_TASK, 'free-choice trials of the case'),
        'case_n_free_first': E(c['n_free_first'], 'd', SRC_TASK, 'first free-choice trials of the case'),
        'case_n_free_repeat': E(c['n_free_repeat'], 'd', SRC_TASK, 'repeat free-choice trials of the case'),
        'case_false_starts': E(c['false_starts_total_accepted'], 'd', SRC_TASK, 'false starts in the accepted trials'),
        'case_attempts_per_accepted': E(c['n_attempts'] / c['n_accepted'], '.2f', SRC_TASK, 'attempts per accepted trial'),
        'case_n_clip_accepted': E(c['n_clip_accepted'], 'd', SRC_TASK, 'accepted trials of the case with the clip flag'),
        'case_n_clip_rejected': E(c['n_clip_rejected'], 'd', SRC_TASK, 'rejected attempts of the case with the clip flag'),
        'case_minutes_total': E(c['minutes_total'], '.1f', SRC_TASK, 'total session length of the case (min)'),
        'ref_attempts_per_accepted': E(rref['attempts_per_accepted'], '.2f', SRC_TASK,
                                       'attempts per accepted trial, all reference attempts pooled'),
        'ref_n_attempts': E(rref['n_attempts'], 'd', SRC_TASK, 'attempts in the reference persons\' included sessions'),
        'ref_n_accepted': E(rref['n_accepted'], 'd', SRC_TASK, 'accepted attempts of the reference persons'),
        'ref_n_sessions': E(rref['n_sessions'], 'd', SRC_TASK, 'included sessions of the reference persons'),
        'ref_session_minutes_median': E(rref['session_minutes']['median'], '.1f', SRC_TASK,
                                        'median reference session length (min)'),
        'ref_session_minutes_min': E(rref['session_minutes']['min'], '.1f', SRC_TASK, 'shortest reference session (min)'),
        'ref_session_minutes_max': E(rref['session_minutes']['max'], '.1f', SRC_TASK, 'longest reference session (min)'),
        'ref_n_rr_verified': E(rref['n_rr_verified'], 'd', SRC_TASK,
                               'reference sessions whose rotation-rate mapping was verified'),
        'ref_n_clip_primary': E(rref['n_clip_primary'], 'd', SRC_TASK, 'reference primary trials with the clip flag'),
        'ref_n_primary': E(rref['n_primary'], 'd', SRC_TASK, 'reference primary trials'),
        'ref_n_clip_persons': E(rref['n_clip_persons_primary'], 'd', SRC_TASK, 'reference persons with a clip flag'),
        'ref_n_sessions_other_protocol': E(rref['n_sessions_other_protocol'], 'd', SRC_TASK,
                                           'reference sessions run with another protocol version than the case'),
        'ref_n_trials_other_protocol': E(rref['n_trials_other_protocol'], 'd', SRC_TASK, 'their analysable trials'),
    }
    for kk, v in sorted(rref['platforms'].items()):
        out[f'ref_n_{_slug(kk)}'] = E(v, 'd', SRC_TASK, f'reference persons on {PLATFORM.get(kk, kk)}')
    for kk, v in sorted(rref['postures'].items()):
        out[f'ref_sessions_{_slug(kk)}'] = E(v, 'd', SRC_TASK, f'reference sessions with posture {kk}')
    for kk, v in sorted(rref['session_status'].items()):
        out[f'ref_sessions_status_{_slug(kk)}'] = E(v, 'd', SRC_TASK, f'reference sessions that {STATUS_WORDS.get(kk, kk)}')
    for reason, cnt in sorted(c['rejections'].items()):
        out[f'case_rejected_{_slug(reason)}'] = E(cnt, 'd', SRC_TASK, f'rejected attempts of the case: {reason}')
    for i, s in enumerate(c['sessions'], start=1):
        pre_ = f'case_session{i}'
        out |= {f'{pre_}_minutes': E(s['minutes'], '.1f', SRC_TASK, f'length of session {i} of the case (min)'),
                f'{pre_}_attempts': E(s['n_attempts'], 'd', SRC_TASK, f'attempts in session {i}'),
                f'{pre_}_accepted': E(s['n_accepted'], 'd', SRC_TASK, f'accepted attempts in session {i}'),
                f'{pre_}_rr_closure': E(s['rr_closure_deg'], '.2f', SRC_TASK,
                                        f'gyro-gravity closure of the assumed rotation-rate mapping, session {i} (deg)'),
                f'{pre_}_rr_runner_up': E(s['rr_runner_up_deg'], '.2f', SRC_TASK,
                                          f'closure of the next-best mapping, session {i} (deg)'),
                f'{pre_}_rr_trials': E(s['rr_trials'], 'd', SRC_TASK, f'trials of the closure check, session {i}')}
    # checks
    sm = chk['speed_matched']
    out |= {
        'chk_sm_n_ref_trials': E(sm['n_ref_trials'], 'd', SRC_CHECKS, 'speed-matched reference trials'),
        'chk_sm_n_ref_persons': E(sm['n_ref_persons'], 'd', SRC_CHECKS, 'persons they come from'),
        'chk_sm_lo': E(sm['speed_range'][0], '.0f', SRC_CHECKS, 'lowest case peak speed (deg/s)'),
        'chk_sm_hi': E(sm['speed_range'][1], '.0f', SRC_CHECKS, 'highest case peak speed (deg/s)'),
        'chk_sm_ref_max': E(sm['ref_max'], '.1f', SRC_CHECKS, 'largest 7-12 Hz RMS of a speed-matched reference trial'),
        'chk_sm_case_min': E(min(sm['case_values']), '.1f', SRC_CHECKS, 'smallest 7-12 Hz RMS of a case trial'),
        'chk_sm_case_max': E(max(sm['case_values']), '.1f', SRC_CHECKS, 'largest 7-12 Hz RMS of a case trial'),
        'chk_sm_n_case_above': E(sm['n_case_above_ref_max'], 'd', SRC_CHECKS,
                                 'case trials above every speed-matched reference trial'),
        'chk_sm_n_case': E(sm['n_case'], 'd', SRC_CHECKS, 'case trials in the stratum'),
    }
    for col, short in (('app_bp_7_12', 'whole'), ('app_bp_7_12_pre_touch', 'pretouch'),
                       ('app_bp_7_12_excl_touch', 'excltouch')):
        b = chk['bandpass'][col]
        out |= {f'chk_bp_{short}_pd': E(b['pd_value'], '.1f', SRC_CHECKS, f'case 7-12 Hz band-pass RMS ({short})'),
                f'chk_bp_{short}_max': E(b['ref_max'], '.1f', SRC_CHECKS, f'reference maximum ({short})'),
                f'chk_bp_{short}_med': E(b['ref_median'], '.1f', SRC_CHECKS, f'reference median ({short})'),
                f'chk_bp_{short}_nmore': E(b['n_more_extreme'], 'd', SRC_CHECKS,
                                           f'reference persons at least as high ({short})'),
                f'chk_bp_{short}_nref': E(b['n_ref'], 'd', SRC_CHECKS, f'reference persons ({short})')}
    pk = chk['peak_prominence']
    out |= {'chk_peak_pd': E(_f(pk['pd_value']), '.1f', SRC_CHECKS, 'case: prominence of the reach\'s 3-12 Hz spectral '
                                                                     'peak, mean over trials (dB)'),
            'chk_peak_med': E(_f(pk['ref_median']), '.1f', SRC_CHECKS, 'reference persons: median (dB)'),
            'chk_peak_qthree': E(_f(pk['ref_q3']), '.1f', SRC_CHECKS, 'reference persons: 75th percentile (dB)'),
            'chk_peak_max': E(_f(pk['ref_max']), '.1f', SRC_CHECKS, 'reference persons: maximum (dB)'),
            'chk_peak_pct': E(_f(pk['percentile']), '.0f', SRC_CHECKS, 'percentile of the case'),
            'chk_peak_nref': E(_i(pk['n_ref']), 'd', SRC_CHECKS, 'reference persons')}
    tp = chk['touch_at_acc_peak']
    yt = chk['yaw_peak_to_touch']
    ny = chk['net_yaw_reversed']
    out |= {
        'chk_tap_case_k': E(tp['case']['k'], 'd', SRC_CHECKS, 'case trials whose first touch is at the |a| peak'),
        'chk_tap_case_n': E(tp['case']['n'], 'd', SRC_CHECKS, 'case trials with a first touch'),
        'chk_tap_ref_share': E(tp['reference']['share'], '.0%', SRC_CHECKS,
                               'share of reference touched trials whose first touch is at the |a| peak'),
        'chk_tap_ref_n': E(tp['reference']['n'], 'd', SRC_CHECKS, 'reference trials with a first touch'),
        'chk_tap_window_ms': E(tp['window_ms'], '.0f', SRC_CHECKS, 'coincidence window (ms)'),
        'chk_yt_case_median': E(yt['case_median_s'], '.2f', SRC_CHECKS, 'case: early yaw peak to first touch, median (s)'),
        'chk_yt_case_min': E(yt['case_min_s'], '.2f', SRC_CHECKS, 'case: smallest gap (s)'),
        'chk_yt_case_max': E(yt['case_max_s'], '.2f', SRC_CHECKS, 'case: largest gap (s)'),
        'chk_yt_ref_median': E(yt['ref_median_s'], '.2f', SRC_CHECKS, 'reference touched trials: median gap (s)'),
        'chk_ny_k': E(ny['k_case'], 'd', SRC_CHECKS, 'case trials with the net yaw against the typical direction'),
        'chk_ny_n': E(ny['n_case'], 'd', SRC_CHECKS, 'case trials with a net yaw'),
        'chk_ny_k_minus2': E(ny['k_case_below_minus2'], 'd', SRC_CHECKS, 'of them below -2 deg'),
        'chk_ny_ref_median': E(ny['ref_median_share'], '.0%', SRC_CHECKS,
                               'reference persons\' share of reversed net yaw, median'),
        'chk_ny_ref_max': E(ny['ref_max_share'], '.0%', SRC_CHECKS, 'the same, maximum'),
        'chk_ny_n_ref_ge': E(ny['n_ref_ge_case'], 'd', SRC_CHECKS, 'reference persons with at least the case\'s share'),
    }
    for s, v in chk['case_clip'].items():
        out[f'chk_clip_{_slug(s)}_flagged'] = E(v['flagged'], 'd', SRC_CHECKS, f'clip-flagged case trials ({s})')
        out[f'chk_clip_{_slug(s)}_n'] = E(v['n'], 'd', SRC_CHECKS, f'case trials ({s})')
    for w, v in chk['resolution'].items():
        for kk in ('resolution_hz', 'main_lobe_half_width_hz', 'leak_7_12', 'in_3_7', 'window_s', 'median_duration_s'):
            if kk in v:
                fmt = '.2f' if kk in ('resolution_hz', 'main_lobe_half_width_hz', 'median_duration_s') else '.1f'
                out[f'res_{w}_{kk}'] = E(v[kk], fmt, SRC_CHECKS, f'{w} window: {kk.replace("_", " ")}')
    r0 = chk['resolution']['pre']
    out['res_tone_hz'] = E(r0['tone_hz'], '.1f', SRC_CHECKS, 'frequency of the test tone (Hz)')
    out['res_tone_rms'] = E(r0['tone_rms'], '.2f', SRC_CHECKS, 'RMS of the test tone (deg/s)')
    for w, dec in chk.get('decomposition', {}).items():
        for band, b in dec.items():
            out[f'dec_{w}_{band}_pct'] = E(_f(b['percentile']), '.0f', SRC_CHECKS,
                                           f'percentile of the case, {w} window, {band} Hz')
    # spectra: the case mean over the reference median, per window and band
    sp = D['spectra']
    for wk, w in SPEC_WINDOWS.items():
        ref = sp[(sp['window'] == w) & (sp['group'] == 'reference') & (sp['level'] == 'person')]
        med = ref.groupby('f_hz')['psd'].median()
        cm = sp[(sp['window'] == w) & (sp['group'] == 'exploratory') & (sp['level'] == 'person')].set_index('f_hz')['psd']
        for band, (lo, hi) in SPEC_BANDS.items():
            ff = [f for f in med.index if lo <= f < hi and f in cm.index]
            ratio = (cm.loc[ff] / med.loc[ff]) if ff else pd.Series(dtype=float)
            out[f'spec_{wk}_{band}_min'] = E(float(ratio.min()) if len(ratio) else None, '.1f', SRC_SPEC,
                                             f'case mean / reference median PSD, {wk}, {band} Hz: minimum')
            out[f'spec_{wk}_{band}_max'] = E(float(ratio.max()) if len(ratio) else None, '.1f', SRC_SPEC,
                                             f'case mean / reference median PSD, {wk}, {band} Hz: maximum')
    # trial-table facts
    ref = t[t['group'] == 'reference']
    lun = XS.STRATA['left_upright_natural'].select(ref)
    cl = t[(t['group'] == 'exploratory') & XS.STRATA['left_upright_natural'].select(t)]
    rep = t[(t['group'] == 'exploratory') & XS.STRATA['left_upright_natural'].select(t)]
    out |= {
        'pre_in_hold_share': E(float(ref['pre_in_verified_hold'].mean()), '.1%', SRC_TRIALS,
                               'reference trials whose 1-s pre-movement window lies in the app-verified still window'),
        'nose_window_share': E(float(ref['nose_rms_3_7'].notna().mean()), '.1%', SRC_TRIALS,
                               'reference trials with a 0.6-s nose window before the return'),
        'touch_share_ref': E(float(ref.loc[lun, 'touch_s'].notna().mean()), '.0%', SRC_TRIALS,
                             'reference left-hand upright natural trials with a first touch during the approach or hold'),
        'case_touch_share': E(float(cl['touch_s'].notna().mean()), '.0%', SRC_TRIALS,
                              'case left-hand upright natural trials with a first touch'),
        'case_repeat_sessions': E(int(rep['session_label'].nunique()), 'd', SRC_TRIALS,
                                  'sessions the case\'s left-hand upright natural reaches come from'),
        'n_ref_free': E(int((ref['analysis_set'] == 'natural_use').sum()), 'd', SRC_TRIALS,
                        'first free-choice trials of the reference persons (natural-use set)'),
        'g_mag_dev_max': E(float(t['g_mag_dev'].max()), '.1e', SRC_TRIALS,
                           'largest deviation of the re-derived |a| from the cached fixed window (m/s^2)'),
        'pre_window_s': E(XF.PARAMS['pre_window_s'], '.1f', SRC_TRIALS, 'length of the hold window (s)'),
        'nose_window_s': E(XF.PARAMS['nose_window_s'], '.1f', SRC_TRIALS, 'length of the nose window (s)'),
    }
    lua = t[(t['group'] == 'exploratory') & XS.STRATA['left_upright_all'].select(t)]
    for sp in ('slow', 'natural', 'fast'):
        out[f'case_n_lua_{sp}'] = E(int((lua['speed'] == sp).sum()), 'd', SRC_TRIALS,
                                    f'case left-hand upright trials at {sp} speed')
    for sp, v in sorted(c.get('primary_speeds', {}).items()):
        out[f'case_n_primary_{_slug(sp)}'] = E(v, 'd', SRC_TASK, f'instructed trials of the case at {sp} speed')
    trial_n = m.loc[m['level'] == 'trial', 'n_case_trials']
    inside = (m['pd_value'] >= m['ref_q1']) & (m['pd_value'] <= m['ref_q3'])
    band_low = int(c['age_band'].split('-')[0]) if c['age_band'][:1].isdigit() else None
    tiers = sorted({tr for s in c['sessions'] for tr in s['tiers_reached']})
    from motionlab.qa import CLIP_ACC_MS2, CLIP_GYRO_DPS
    P = XF.PARAMS
    out |= {
        'case_min_trials': E(int(trial_n.min()), 'd', SRC_MAIN, 'fewest case trials behind a person value'),
        'case_max_trials': E(int(trial_n.max()), 'd', SRC_MAIN, 'most case trials behind a person value'),
        'n_inside_iqr': E(int(inside.sum()), 'd', SRC_MAIN, 'characteristics with the case inside the reference IQR'),
        'case_tiers': E(', '.join(str(x) for x in tiers), '', SRC_TASK, 'protocol tiers the case reached'),
        'age_gap_min': E((band_low - rref['age_max']) if band_low is not None else None, '.0f', SRC_TASK,
                         'smallest age difference between the case\'s age band and the reference persons (years)'),
        'case_n_clip_acc': E(c['n_clip_acc_accepted'], 'd', SRC_TASK,
                             'accepted case trials whose accelerometer exceeded the clip threshold'),
        'case_n_clip_gyro': E(c['n_clip_gyro_accepted'], 'd', SRC_TASK,
                              'accepted case trials whose gyroscope exceeded the clip threshold'),
        'clip_acc_g': E(CLIP_ACC_MS2 / 9.80665, '.1f', SRC_TASK, 'clip threshold of the raw specific force (g)'),
        'clip_gyro_dps': E(CLIP_GYRO_DPS, '.0f', SRC_TASK, 'clip threshold of the raw angular velocity (deg/s)'),
        'param_contact_ms': E(P['contact_ms'], '.0f', SRC_TRIALS, 'window around the first touch for its acceleration (ms)'),
        'param_return_dps': E(P['return_dps'], '.0f', SRC_TRIALS, 'angular speed that starts the return (deg/s)'),
        'param_return_sustain_ms': E(P['return_sustain_ms'], '.0f', SRC_TRIALS, 'for at least this long (ms)'),
        'param_movement_lowpass_hz': E(P['movement_lowpass_hz'], '.0f', SRC_TRIALS, 'movement-band low-pass (Hz)'),
        'param_pre_touch_gap_ms': E(P['pre_touch_gap_ms'], '.0f', SRC_TRIALS,
                                    'the "before the touch" window ends this long before it (ms)'),
        'param_touch_excl_before_ms': E(P['touch_exclude_ms'][0], '.0f', SRC_TRIALS,
                                        'excluded before the touch in the "without the touch" check (ms)'),
        'param_touch_excl_after_ms': E(P['touch_exclude_ms'][1], '.0f', SRC_TRIALS,
                                       'excluded after the touch in the "without the touch" check (ms)'),
        'param_hesitation_ms': E(P['hesitation_min_ms'], '.0f', SRC_TRIALS, 'shortest hesitation (ms)'),
        'param_still_dps': E(P['still_dps'], '.0f', SRC_TRIALS, 'the app\'s still threshold (deg/s)'),
        'param_peak_prominence': E(P['peak_prominence'], '.0%', SRC_TRIALS, 'rise of a speed peak, share of the maximum'),
    }
    return out


# ───────────────────────────── phrases (shared with the figure texts) ─────────────────────────────


def ordinal(s: str) -> str:
    """'5' -> '5th', '1' -> '1st', '93' -> '93rd'."""
    try:
        n = int(float(s))
    except ValueError:
        return s
    suf = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suf}'


class Nums:
    """Formatted access to the numbers entries (the only source of numbers in texts)."""

    def __init__(self, entries: dict[str, dict[str, Any]]):
        self.e = entries

    def v(self, key: str) -> Any:
        return self.e[key]['value']

    def has(self, key: str) -> bool:
        return key in self.e and self.e[key]['value'] is not None

    def __call__(self, key: str, fmt: str | None = None) -> str:
        e = self.e[key]
        return RX.format_value(e['value'], e['format'] if fmt is None else fmt)

    def pct(self, k: str) -> str:
        return ordinal(self(f'{k}_pct'))

    def verb(self, key: str, singular: str = 'is', plural: str = 'are') -> str:
        """The verb that agrees with the count ``key`` ('1 is', '4 are')."""
        return singular if self.v(key) == 1 else plural

    def peak(self) -> str:
        """How sharp the reach's 3-12 Hz spectral peak is, against the reference persons (from the check)."""
        if not self.has('chk_peak_pd'):
            return ''
        if self.v('chk_peak_pd') > self.v('chk_peak_max'):
            verdict = 'per trial, a more prominent peak than in any reference person\'s reaches'
        elif self.v('chk_peak_pd') > self.v('chk_peak_qthree'):
            verdict = 'per trial, a more prominent peak than in most reference persons\' reaches'
        else:
            verdict = 'no more prominent than in a typical reference reach, so a broadband rise rather than a peak'
        return (f'its 3-12 Hz spectral peak rises {self("chk_peak_pd")} dB above its surroundings (mean over trials; '
                f'reference persons: median {self("chk_peak_med")} dB, maximum {self("chk_peak_max")} dB): {verdict}')

    def rank(self, k: str, short: bool = False) -> str:
        """'above every reference person' / 'at the 40th percentile' (short: 'above all' / '40th pct')."""
        if not self.has(f'{k}_pct'):
            return 'not available'
        if self.v(f'{k}_beyond'):
            side = 'above' if self.v(f'{k}_above') else 'below'
            return f'{side} all' if short else f'{side} every reference person'
        return f'{self.pct(k)} pct' if short else f'at the {self.pct(k)} percentile'

    def unit(self, k: str) -> str:
        u = XS.BY_KEY[k].unit.replace('m/s$^2$', 'm/s²')
        return f' {u}' if u and u not in ('count', 'share', 'per trial', 'ratio') else ''

    def value(self, k: str, which: str = 'pd') -> str:
        return f'{self(f"{k}_{which}")}{self.unit(k)}'

    def z(self, k: str) -> str:
        if not self.has(f'{k}_z'):
            return ''
        return f'robust z {self(f"{k}_z")} ({self(f"{k}_zscale")} scale)'


@lru_cache(maxsize=1)
def _cached(stamp: tuple[float, ...]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    D = load()
    return D, numbers(D)


def current_numbers() -> tuple[dict[str, Any], Nums]:
    """The saved inputs and their numbers (cached per modification time of the inputs)."""
    files = [L.res_dir / 'x1_characteristics.json', L.res_dir / 'x1_multivariate.json', L.tab_dir / 'x1_task.json',
             L.tab_dir / 'x1_checks.json', L.tab_dir / 'x1_trial_characteristics.csv',
             L.tab_dir / 'x1_comparisons.csv', L.tab_dir / 'x1_spectra.csv']
    D, nums = _cached(tuple(p.stat().st_mtime if p.exists() else -1.0 for p in files))
    return D, Nums(nums)


def check_names(N: Nums, keys: str) -> str:
    """'no_clip;size' -> 'without clip-flagged trials and the size check'."""
    words = {**XS.VARIANT_LABELS, 'size': f'the size check (|z| >= {N("z_clear")})'}
    items = [words.get(k, k) for k in keys.split(';') if k]
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1] if items else ''


def finding(N: Nums, k: str) -> str:
    """One characteristic in running text: value, reference median and range, z, consistency over trials,
    robustness."""
    c = XS.BY_KEY[k]
    parts = [f'reference median {N(f"{k}_med")}, range {N(f"{k}_min")} to {N(f"{k}_max")}']
    if N.has(f'{k}_z'):
        parts.append(N.z(k))
    if c.level == 'trial' and N.has(f'{k}_ntrialsvalid'):
        parts.append(f'{N(f"{k}_ntrialsbeyond")} of {N(f"{k}_ntrialsvalid")} trials beyond the reference range')
    if N.v(f'{k}_beyond'):
        if N.v(f'{k}_robust'):
            parts.append('robust')
        else:
            why = []
            if N.v(f'{k}_failed'):
                why.append(f'fails {check_names(N, N.v(f"{k}_failed"))}')
            if N.v(f'{k}_unavailable'):
                why.append(f'cannot be checked {check_names(N, N.v(f"{k}_unavailable"))} (too few case trials)')
            parts.append('not robust: ' + ', and '.join(why))
    return f'{XS.lc(c.label)} {N.value(k)} ({"; ".join(parts)}){" *" if c.post_hoc else ""}'


# ───────────────────────────── summary ─────────────────────────────


def _md_table(m: pd.DataFrame, N: Nums, case: str) -> list[str]:
    out = [f'| Characteristic | Stratum | n | {case} | Reference median [IQR] | Pct | Robust z | Trials beyond | '
           'p (rank) | q | p (C-H) | Robust |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in m.iterrows():
        k = r['key']
        c = XS.BY_KEY[k]
        u = c.unit.replace('m/s$^2$', 'm/s²')
        tb = f'{N(f"{k}_ntrialsbeyond")}/{N(f"{k}_ntrialsvalid")}' if c.level == 'trial' else '--'
        rob = ('yes' if N.v(f'{k}_robust') else 'no') if N.v(f'{k}_beyond') else ''
        out.append(f"| {c.label}{f' ({u})' if u else ''}{' *' if c.post_hoc else ''} | {XS.STRATA[c.stratum].label} | "
                   f"{N(f'{k}_ncase')} | {N(f'{k}_pd')} | {N(f'{k}_med')} [{N(f'{k}_qone')}, {N(f'{k}_qthree')}] | "
                   f"{N.rank(k, short=True)} | {N(f'{k}_z')} | {tb} | {N(f'{k}_p')} | {N(f'{k}_q')} | "
                   f"{N(f'{k}_pch')} | {rob} |")
    return out


def _ratio_driver(N: Nums, D: dict[str, Any], window: str) -> str:
    """Which band drives a power ratio that puts the case beyond all (from the decomposition check)."""
    dec = D['checks'].get('decomposition', {}).get(window)
    if not dec:
        return ''
    num = f'dec_{window}_3_12_pct'
    lo, hi = f'dec_{window}_05_3_pct', f'dec_{window}_12_15_pct'
    pn, pl, ph = (_f(N.v(x)) for x in (num, lo, hi))
    if any(math.isnan(x) for x in (pn, pl, ph)):
        return ''
    comp = (f'3-12 Hz power at the {ordinal(N(num))} percentile, 0.5-3 Hz at the {ordinal(N(lo))} and 12-15 Hz at the '
            f'{ordinal(N(hi))}')
    if pn > 75 and min(pl, ph) >= 25:
        return f'It comes from more power in 3-12 Hz ({comp}).'
    if pn <= 75 and min(pl, ph) < 25:
        return f'It comes from less power outside 3-12 Hz, not from more inside it ({comp}).'
    return f'Its components: {comp}.'


def _qualifiers(N: Nums, D: dict[str, Any], phen: str, keys: list[str]) -> list[str]:
    """Data-driven context sentences for a phenomenon with a "beyond all" characteristic."""
    out = []
    if 'pre_ratio_db' in keys:
        out.append(_ratio_driver(N, D, 'pre'))
    if 'app_ratio_db' in keys:
        out.append(_ratio_driver(N, D, 'app'))
    if 'nose_ratio_db' in keys:
        out.append(_ratio_driver(N, D, 'nose'))
    if phen == XS.P_ACC:
        for k in keys:
            if N.has(f'{k}_pdnoclip'):
                still = (N.v(f'{k}_pdnoclip') > N.v(f'{k}_extnoclip')) if N.v(f'{k}_above') else \
                    (N.v(f'{k}_pdnoclip') < N.v(f'{k}_extnoclip'))
                out.append(f'Without the clip-flagged trials, {XS.lc(XS.BY_KEY[k].label)} is {N.value(k, "pdnoclip")} '
                           f'against a reference extreme of {N.value(k, "extnoclip")} '
                           f'({"still beyond every reference person" if still else "no longer beyond every person"}).')
        out.append(f'The clip flag (possible accelerometer saturation) is set on {N("case_n_clip_accepted")} of '
                   f'{N("case_n_accepted")} accepted trials of the case and on {N("ref_n_clip_primary")} of '
                   f'{N("ref_n_primary")} reference primary trials ({N("ref_n_clip_persons")} persons), so the case\'s '
                   f'acceleration magnitudes are lower bounds. The first touch falls within {N("chk_tap_window_ms")} ms '
                   f'of the largest acceleration of the approach in {N("chk_tap_case_k")} of {N("chk_tap_case_n")} '
                   f'of the case\'s trials with a touch (reference touched trials: {N("chk_tap_ref_share")}): the peak '
                   f'acceleration and the acceleration at the first touch are one event and count as one finding.')
    if phen == XS.P_TOUCH:
        out.append(f'An early screen touch during an ongoing movement: {N("rot_after_touch_pd")} of the voluntary '
                   f'rotation of the approach comes after it (reference median {N("rot_after_touch_med")}, '
                   f'{N.rank("rot_after_touch")}). What touched the screen (face, finger or grip) is unknown.')
    if phen == XS.P_REACHBP:
        out.append(f'Speed-matched: all reference left-hand upright natural reaches with a voluntary peak speed in the '
                   f'case\'s range ({N("chk_sm_lo")}-{N("chk_sm_hi")} deg/s; {N("chk_sm_n_ref_trials")} trials of '
                   f'{N("chk_sm_n_ref_persons")} persons) stay below {N("chk_sm_ref_max")} deg/s, and '
                   f'{N("chk_sm_n_case_above")} of {N("chk_sm_n_case")} case reaches exceed that '
                   f'({N("chk_sm_case_min")}-{N("chk_sm_case_max")}). With a time-domain 7-12 Hz band-pass the case is '
                   f'at {N("chk_bp_whole_pd")} deg/s (reference maximum {N("chk_bp_whole_max")}), before the first touch '
                   f'at {N("chk_bp_pretouch_pd")} (maximum {N("chk_bp_pretouch_max")}, {N("chk_bp_pretouch_nref")} '
                   f'persons) and without the touch at {N("chk_bp_excltouch_pd")} (maximum '
                   f'{N("chk_bp_excltouch_max")}): not only the contact. Relative to the faster reach (7-12 Hz RMS per '
                   f'peak speed) it is {N.rank("app_7_12_per_speed")} *. The case\'s mean reach spectrum lies '
                   f'{N("spec_reach_7_12_min")}-{N("spec_reach_7_12_max")} times the reference median across 7-12 Hz, '
                   f'and {N.peak()}.')
    if phen == XS.P_NET:
        k1 = [k for k in keys]
        for a in k1:
            for b in k1:
                key = f'rho_{a}_{b}'
                if N.has(key):
                    out.append(f'{XS.BY_KEY[a].label} and {XS.lc(XS.BY_KEY[b].label)} measure one change (Spearman '
                               f'rho {N(key)} across the reference persons).')
    if phen == XS.P_RETURN:
        out.append('A share of trials, not a time: the recording stops a fixed time after the end cue, so a phone kept '
                   'at the nose gives a censored time.')
    if phen == XS.P_REP:
        both = '; '.join(f'{XS.lc(XS.BY_KEY[k].label)} at the {ordinal(N(f"{k}_pctboth"))} percentile'
                         for k in keys if N.has(f'{k}_pctboth'))
        out.append(f'Left-hand reaches only ({N("seq_peak_speed_mv_nref")} reference persons made enough of them); the '
                   f'case\'s reaches come from {N("case_repeat_sessions")} sessions.'
                   + (f' Against the larger, hand-mixed reference ({N("seq_peak_speed_mv_nrefboth")} persons): {both}.'
                      if both else ''))
    if phen == XS.P_EARLY:
        out.append(f'The early yaw peak falls {N("chk_yt_case_min")}-{N("chk_yt_case_max")} s from the first touch '
                   f'(median {N("chk_yt_case_median")} s; reference touched trials: median {N("chk_yt_ref_median")} s).')
    if phen == XS.P_NETYAW:
        out.append(f'The hand-consistent net yaw is reversed (against the typical direction) in {N("chk_ny_k")} of '
                   f'{N("chk_ny_n")} accepted trials ({N("chk_ny_k_minus2")} below -2 deg); reference persons: median '
                   f'{N("chk_ny_ref_median")}, maximum {N("chk_ny_ref_max")}.')
    return [s for s in out if s]


def summary(D: dict[str, Any], N: Nums) -> str:
    from stage2.lanes import x1_texts
    m, mv, task, t, case = D['main'], D['mv'], D['task'], D['trials'], D['case']
    c, rref = task['case'], task['reference']
    by = m.set_index('key')
    pre = m[~m['post_hoc'].astype(bool)]
    post = m[m['post_hoc'].astype(bool)]
    beyond = m[m['beyond_all'].astype(bool)]
    case_t = t[t['group'] == 'exploratory']
    out: list[str] = []
    a = out.append
    plat = PLATFORM.get(c['platform'], c['platform'])
    phen_order = list(dict.fromkeys(cc.phenomenon for cc in XS.CHARACTERISTICS))

    a(f'# Lane x1: exploratory case study of one participant with Parkinson\'s disease ({case})')
    a('')
    a(f'**Exploratory and descriptive only.** {case} is never trained on, validated on, tuned with or scored by any '
      f'model, and no model-performance number (accuracy, AUC, predicted P(right)) is reported for {case}. Health '
      f'information stays in the exploratory outputs (`figures/x1`, `tables/x1`, `results/x1_*`). The wording is '
      f'descriptive: nothing here is a diagnostic claim, and age and device are confounded with the condition.')
    a('')
    a('## In brief')
    a('')
    lfl = 'null_pre_lfl'
    head = ('Unusual, not unique' if N.v(f'{lfl}_n_ge') or N.v(f'{lfl}_n_ge_phen')
            else 'More differences than any reference person')
    a(f'- **{head}.** Over the {N("n_prespecified")} pre-specified characteristics, {case} lies beyond '
      f'every reference person in {N(f"{lfl}_case")} ({N(f"{lfl}_case_phen")} phenomena; '
      f'{N(f"{lfl}_expected")} expected if {case} were exchangeable with the reference persons). Each reference person '
      f'judged the same way, on the mean of as many trials as {case} made, against the other '
      f'{N("n_ref_minus_one")}, lies beyond all others in a median of {N(f"{lfl}_ref_median")} (maximum '
      f'{N(f"{lfl}_ref_max")}); {N(f"{lfl}_n_ge")} of the {N("n_ref_persons")} young reference persons '
      f'{N.verb(f"{lfl}_n_ge", "reaches", "reach")} {case}\'s count (rank p {N(f"{lfl}_p")}) and '
      f'{N(f"{lfl}_n_ge_phen")} {N.verb(f"{lfl}_n_ge_phen", "reaches", "reach")} its number of phenomena (rank p '
      f'{N(f"{lfl}_p_phen")}).')
    a(f'- **Robust differences** (beyond every reference person, and still so when each reference person is judged on '
      f'as many trials as {case}, with the protocol version matched, without the clip-flagged trials, and with |z| >= '
      f'{N("z_clear")} on the declared scale whichever reference person is left out): '
      + ('; '.join(f'{XS.lc(XS.BY_KEY[k].label)}{" *" if XS.BY_KEY[k].post_hoc else ""}'
                   for k in m.loc[m['robust'].astype(bool), 'key']) or 'none')
      + f' ({N("n_robust_prespecified")} pre-specified characteristics in {N("n_phen_robust_prespecified")} phenomena).')
    a(f'- **Multiplicity.** No pre-specified characteristic reaches q < 0.05 by either test (smallest Benjamini-Hochberg '
      f'q {N("min_q_rank")} for the rank test, {N("min_q_ch")} for the Crawford-Howell test, family: the '
      f'{N("n_prespecified")} pre-specified characteristics). The {N("n_post_hoc")} post hoc characteristics (*) carry no '
      f'q-value: they were defined after looking at {case} and are hypotheses, not tests.')
    a(f'- **Multivariate.** On the {N("mv_n_keys")} pre-specified key characteristics, on their declared scales, '
      f'{N("mv_n_beyond")} of the {N("n_ref_persons")} reference persons {N.verb("mv_n_beyond")} at least as far from '
      f'the reference centre as {case} (rank p {N("mv_p")}). On linear scales {N("mv_lin_n_beyond")} '
      f'{N.verb("mv_lin_n_beyond")} (rank p {N("mv_lin_p")}); without {N("mv_lin_loco_max_label")}, '
      f'{N("mv_lin_loco_max_beyond")} {N.verb("mv_lin_loco_max_beyond")}'
      + (f', so the linear result depends on that one right-skewed characteristic.'
         if N.v('mv_lin_loco_max_beyond') > N.v('mv_lin_n_beyond') else '.'))
    a(f'- **Hand signature, without a classifier.** The early turn about gravity has the direction typical of the hand '
      f'used (first-half yaw {N.rank("fhy_hand")}; early yaw peak {N.rank("yaw_peak_hand")} *), but the net turn '
      f'comes back against it: the hand-consistent net yaw is reversed in {N("chk_ny_k")} of {N("chk_ny_n")} accepted '
      f'trials ({N.rank("net_yaw_hand")} on the person mean; reference persons reverse a median {N("chk_ny_ref_median")} '
      f'and at most {N("chk_ny_ref_max")} of their trials).')
    typical = [k for k in EXPECTED_TO_DIFFER if k in by.index and pd.notna(by.loc[k, 'percentile'])
               and by.loc[k, 'ref_q1'] <= by.loc[k, 'pd_value'] <= by.loc[k, 'ref_q3']]
    atyp = [k for k in EXPECTED_TO_DIFFER if k in by.index and pd.notna(by.loc[k, 'percentile']) and k not in typical]
    a('- **Expected to differ, but inside the reference interquartile range:** '
      + ('; '.join(f'{XS.lc(XS.BY_KEY[k].label)} ({N.rank(k)})' for k in typical) or 'none')
      + ('. Outside it, though not necessarily beyond all: '
         + '; '.join(f'{XS.lc(XS.BY_KEY[k].label)} ({N.rank(k)})' for k in atyp) if atyp else '') + '.')
    a('')
    a('## Goal')
    a('')
    a(f'What does {case} do, when performing the nose-touch task, that the {N("n_ref_persons")} modelling persons do '
      f'not? The comparison covers preparation and holding still, movement time and speed (bradykinesia), amplitude '
      f'(hypometria), smoothness, band power in the tremor bands during the hold, the reach and at the nose, '
      f'variability and change across repeats, speed scaling, task behaviour, and the hand signature itself (without a '
      f'classifier).')
    a('')
    a('## Data')
    a('')
    sess = '; '.join(f"{s['session_label']}: {STATUS_WORDS.get(s['status'], s['status'])}, "
                     f"{N(f'case_session{i}_minutes')} min, {N(f'case_session{i}_attempts')} attempts, "
                     f"{N(f'case_session{i}_accepted')} accepted" for i, s in enumerate(c['sessions'], start=1))
    rej = '; '.join(f'{k} ({N(f"case_rejected_{_slug(k)}")})' for k in sorted(c['rejections']))
    per_speed = ', '.join(f'{N(f"case_n_primary_{_slug(sp)}")} {sp}' for sp in sorted(c.get('primary_speeds', {})))
    a(f'- **{case}:** age band {N("case_age_band")} years, {c["handedness"]}-handed, usually holds the phone in the '
      f'{c["phone_hand"]} hand, {plat} phone (device model not reported), {c["posture"]}, protocol '
      f'{c["protocol_version"]}. {N("case_n_sessions")} short sessions ({sess}); {N("case_n_attempts")} attempts, '
      f'{N("case_n_accepted")} accepted, {N("case_n_rejected")} rejected by the app ({rej}). Reference sessions last '
      f'{N("ref_session_minutes_min")}-{N("ref_session_minutes_max")} min (median {N("ref_session_minutes_median")}).')
    hands_f = ' and '.join(c['free_hands']) or 'no'
    hands_p = ' and '.join(c['primary_hands']) or 'no'
    a(f'- **Accepted trials:** {N("case_n_free")} free-choice trials (hand reported: {hands_f}; '
      f'{N("case_n_free_first")} first and {N("case_n_free_repeat")} repeat) and {N("case_n_primary")} instructed '
      f'{hands_p}-hand trials with the {", ".join(o.replace("_", " ") for o in c["primary_orientations"])} grip '
      f'({per_speed}). Their {N("case_false_starts")} false starts and the attempts are in '
      f'`tables/x1/x1_attempts.csv`.')
    a(f'- **Reference:** the {N("n_ref_persons")} modelling persons (age {N("ref_age_min")}-{N("ref_age_max")} years; '
      + ', '.join(f'{N(f"ref_n_{_slug(k)}")} {PLATFORM.get(k, k)}' for k in sorted(rref['platforms']))
      + '), all their accepted analysable trials (primary set, natural-use and natural-use-repeat free trials), '
      f'{N("ref_n_attempts")} attempts and {N("ref_n_accepted")} accepted in their included sessions '
      f'({N("ref_attempts_per_accepted")} attempts per accepted trial pooled).')
    rr = '; '.join(f"{s['session_label']}: closure {N(f'case_session{i}_rr_closure')} deg vs "
                   f"{N(f'case_session{i}_rr_runner_up')} deg for the next mapping over {N(f'case_session{i}_rr_trials')} "
                   f"trials" for i, s in enumerate(c['sessions'], start=1))
    a(f'- **Rotation-rate mapping assumed, not verified.** Both of {case}\'s sessions use the assumed (W3C) mapping of '
      f'the gyroscope axes, not one verified by the app\'s gyro-gravity closure as in {N("ref_n_rr_verified")} of the '
      f'{N("ref_n_sessions")} reference sessions; the closure still favours it ({rr}). The hand signature and the '
      f'gravity-frame characteristics depend on this mapping.')
    a(f'- **Clip flag.** The core flag `clip_candidate` (a raw specific force above {N("clip_acc_g")} g or an angular '
      f'velocity above {N("clip_gyro_dps")} deg/s: a sensor may have saturated) is set on {N("case_n_clip_accepted")} '
      f'of {case}\'s {N("case_n_accepted")} accepted trials (accelerometer in {N("case_n_clip_acc")}, gyroscope in '
      f'{N("case_n_clip_gyro")}) and on {N("case_n_clip_rejected")} rejected attempt(s), against '
      f'{N("ref_n_clip_primary")} of {N("ref_n_primary")} reference primary trials ({N("ref_n_clip_persons")} persons). '
      f'Magnitudes of flagged trials are lower bounds; every comparison is repeated without the flagged trials.')
    a(f'- **Same code path.** {case}\'s raw records live outside the main snapshot (`analysis/data/exploratory/`). '
      f'`x1_data.ensure_cache` writes a main-format curation for them into `analysis/data/exploratory_cache/{case}/` '
      f'(git-ignored) and runs the main cache\'s own `data.build_cache` there: the same loader, standardisation, flags, '
      f'gates, exclusion cascade and cache format. `stage2/tests/test_x1.py` checks identical parameters and builder '
      f'code, and that none of {case}\'s ids reached the main curation or cache.')
    a('')
    a('## Methods')
    a('')
    a('**Windows** (per trial; `x1_features.PARAMS`):')
    a(f'- *Hold before the movement:* [onset - {N("pre_window_s")} s, onset). It lies inside the app\'s verified still '
      f'window in {N("pre_in_hold_share")} of the reference trials, so it is a postural hold. Design limitation: a hold '
      f'that breaks the app\'s still gate restarts (a false start) instead of entering this window, so large movements '
      f'of the hold are censored here and show only as false starts.')
    a('- *Reach:* the oracle approach [onset, end] (the stored online_v1 segment).')
    a(f'- *At the nose:* [end, end + {N("nose_window_s")} s), only when the return of the arm starts later; available '
      f'in {N("nose_window_share")} of the reference trials. The approach ends only after the phone was still, and '
      f'attempts that never became still were rejected ({N("case_rejected_too_long") if N.has("case_rejected_too_long") else "--"} '
      f'of {case}\'s rejections were "still moving at the time limit"), so the nose window describes only trials that '
      f'became still.')
    a(f'- *Spectral resolution:* a window of T s resolves 1/T Hz: {N("res_pre_resolution_hz")} Hz in the hold, '
      f'{N("res_nose_resolution_hz")} Hz at the nose, about {N("res_approach_resolution_hz")} Hz in a median reach '
      f'({N("res_approach_median_duration_s")} s). The Hann main lobe spreads a tone over +-2/T Hz: a '
      f'{N("res_tone_hz")}-Hz tone of RMS {N("res_tone_rms")} deg/s puts {N("res_pre_leak_7_12")} deg/s (hold window) '
      f'and {N("res_nose_leak_7_12")} deg/s (nose window) into the 7-12 Hz band. The 3-7 / 7-12 Hz split of the short '
      f'windows is therefore coarse, and spectral peaks are reported at the window\'s resolution.')
    a('')
    a(f'**Characteristics** ({N("n_characteristics")}: {N("n_prespecified")} pre-specified, {N("n_post_hoc")} post '
      f'hoc; each with a one-line kinematic rationale in `tables/x1/x1_characteristics_values.csv`), grouped into '
      f'{N("n_phenomena")} *phenomena*: characteristics that measure the same thing (for example the time and the '
      f'share of the approach of one touch, or the net rotation and the tilt change) count as one finding. The '
      f'grouping is declared by definition in `x1_stats.CHARACTERISTICS`, not by {case}\'s values.')
    a('- band RMS of the angular-velocity vector (three gyroscope axes, detrended, Hann over the full window, '
      'zero-padded, densities summed) in 3-7 Hz (the band of parkinsonian rest and re-emergent tremor) and 7-12 Hz (the '
      'band of physiological and action tremor), the spectral peak in 3-12 Hz, and the 3-12 Hz power relative to 0.5-3 '
      'and 12-15 Hz (dB), with its components; "band power" is not a tremor unless the spectrum shows a peak;')
    a(f'- the reach both as recorded and in a movement band (gyroscope low-passed at {N("param_movement_lowpass_hz")} '
      f'Hz), which separates the voluntary reach from faster components; smoothness by the log dimensionless jerk and '
      f'SPARC (Balasubramanian et al. 2015), speed peaks and sub-movements (rising {N("param_peak_prominence")} of the '
      f'maximum) and hesitations (at least {N("param_hesitation_ms")} ms below the app\'s {N("param_still_dps")} deg/s '
      f'still threshold mid-reach);')
    a(f'- the first screen touch (time, share of the approach, the voluntary rotation after it, the linear acceleration '
      f'within {N("param_contact_ms")} ms of it on the whole canonical recording: the calibrated |g| the cached stream '
      f'lacks is recovered from the cached fixed window, largest deviation {N("g_mag_dev_max")} m/s²) and time-domain '
      f'7-12 Hz checks before it (up to {N("param_pre_touch_gap_ms")} ms before) and without it '
      f'({N("param_touch_excl_before_ms")} ms before to {N("param_touch_excl_after_ms")} ms after);')
    a('- the hand signature: first-half and net yaw about gravity, mirror-normalised so that positive is the direction '
      'typical of the hand used (the label is used only for this sign, never by a model).')
    a(f'- **Post hoc** (*): the first-touch characteristics, the active movement time, the settling time and the '
      f'active-time ratio, the early yaw peak (defined after {case}\'s traces were inspected), and the components of the '
      f'hold power ratio, the 7-12 Hz power per peak speed and the share of trials with the typical net-yaw sign (added '
      f'after the audit, to test alternative explanations). They are reported apart, without a q-value.')
    a('')
    a('**Matched comparisons (strata).** Each characteristic is judged in one pre-declared stratum: the reach in the '
      'left-hand upright natural trials (standard movement, natural start), the hold, the nose and the hand signature '
      'in all left-hand upright trials, the across-repeat measures in the first left-hand upright natural reaches in '
      'visit order (as many as the case made; the hand-mixed upright natural reaches are a secondary, larger '
      'reference), the speed scaling from the slow and fast trials. Every per-trial characteristic is also reported in '
      'every other stratum (`tables/x1/x1_strata.csv`).')
    a('')
    a(f'**Statistics** (`x1_stats.compare`). A person\'s value is the mean of their trials in the stratum. Against the '
      f'reference persons\' values: median [IQR] and range, mid-rank percentile, robust z = (x - median) / (1.4826 MAD) '
      f'on the characteristic\'s declared scale (log for strictly positive, right-skewed quantities; the linear z is in '
      f'the tables), the two-sided rank p = min(1, 2 min(#(ref >= x) + 1, #(ref <= x) + 1) / (n + 1)), the '
      f'Crawford-Howell (1998) test on the declared scale (the other scale alongside), and how many of the case\'s own '
      f'trials lie beyond the reference persons\' range. **With n = {N("n_ref_persons")} reference persons the smallest '
      f'possible two-sided p is 2/(n + 1) = {N("p_floor")}**; fewer than {N("min_ref")} reference persons: no statistic. '
      f'Benjamini-Hochberg q-values run over the {N("n_prespecified")} pre-specified characteristics (one family).')
    a('')
    a(f'**When is a difference robust?** A "beyond every reference person" verdict is robust when it also holds in every '
      f'sensitivity analysis that recomputes values: (1) *first k*: each reference person judged on their first k matched '
      f'trials, k = the case\'s count (equal averaging noise and practice); (2) *same protocol*: the reference trials of '
      f'protocol {c["protocol_version"]} only ({N("ref_n_sessions_other_protocol")} reference session with '
      f'{N("ref_n_trials_other_protocol")} trials used another protocol); (3) *without clip-flagged trials*, for the case '
      f'and the reference; (4) *same posture*, only when a reference person has trials in both postures; and (5) a size '
      f'check: |robust z| >= {N("z_clear")} on the declared scale, also with each reference person left out. The '
      f'same-posture and same-platform subsets of persons are reported too, but they only drop persons and can never '
      f'overturn a "beyond all" verdict.')
    a('')
    a(f'**Null reference of the count** (`x1_stats.beyond_all_counts`). How many characteristics would put a young '
      f'reference person beyond every other reference person? Each reference person is compared with the other '
      f'{N("n_ref_minus_one")} (like for like: on the mean of their first k trials, as noisy as the case\'s), the case '
      f'with all {N("n_ref_persons")}; the case\'s rank among the {N("n_ref_plus_one")} gives a rank p (conservative: '
      f'the case faces one more person).')
    a('')
    a(f'**Multivariate view** (`x1_stats.multivariate`): {N("mv_n_keys")} pre-specified key characteristics ('
      + '; '.join(XS.lc(XS.BY_KEY[k].label) for k in mv['keys']) + f'), on their declared scales ({N("mv_n_log")} on the '
      'log scale) and robust-standardised; a PCA fitted on the reference persons with the case projected, and a '
      'Mahalanobis distance with a Ledoit-Wolf shrinkage covariance, each reference person left out of the '
      'standardisation and covariance of their own distance; repeated on linear scales and with each characteristic '
      'left out.')
    a('')
    a('## Results')
    a('')
    a('### Key characteristics')
    a('')
    a('Pct: mid-rank percentile, or above / below all when the case lies beyond every reference person. Robust z on the '
      'declared scale. Trials beyond: the case\'s trials beyond the reference persons\' range. q: Benjamini-Hochberg over '
      'the pre-specified characteristics. Robust: see Methods. *: post hoc.')
    a('')
    out.extend(_md_table(m[m['key_set'].astype(bool)], N, case))
    a('')
    a('### What stands out')
    a('')
    a(f'{case} lies beyond every reference person in {N("n_beyond_prespecified")} pre-specified characteristics '
      f'({N("n_phen_beyond_prespecified")} phenomena) and {N("n_beyond_post_hoc")} post hoc ones, each at its rank-p floor '
      f'2/(n + 1) ({N("p_floor")} with all {N("n_ref_persons")} persons). By phenomenon:')
    a('')
    for ph in phen_order:
        ks = [k for k in beyond['key'] if XS.BY_KEY[k].phenomenon == ph]
        if not ks:
            continue
        robust = any(N.v(f'{k}_robust') for k in ks)
        a(f'- **{ph[:1].upper() + ph[1:]}** ({"robust" if robust else "not robust"}): '
          + '; '.join(finding(N, k) for k in ks) + '. ' + ' '.join(_qualifiers(N, D, ph, ks)))
    a('')
    red = [k for k in N.e if k.startswith('redundant_') and k.endswith('_pair')]
    if red:
        a('Across the reference persons, some of these findings of different phenomena also go together: '
          + '; '.join(f'{N(k)} (rho {N(k.replace("_pair", "_rho"))})' for k in sorted(red))
          + '. They may reflect one behaviour (for example a faster reach with a harder arrival).')
        a('')
    outside = m[~m['beyond_all'].astype(bool) & ((m['pd_value'] < m['ref_q1']) | (m['pd_value'] > m['ref_q3']))]
    a('Outside the reference interquartile range but not beyond every person: '
      + '; '.join(f'{XS.lc(r["label"])}{" *" if r["post_hoc"] else ""} ({N.pct(r["key"])} pct)'
                  for _, r in outside.iterrows()) + '.')
    a('')
    a('### What does not stand out')
    a('')
    inside = m[(m['pd_value'] >= m['ref_q1']) & (m['pd_value'] <= m['ref_q3'])]
    a(f'Inside the reference interquartile range ({N("n_inside_iqr")} characteristics): '
      + '; '.join(f'{XS.lc(r["label"])}{" *" if r["post_hoc"] else ""} ({N.pct(r["key"])} pct)'
                  for _, r in inside.iterrows()) + '.')
    a('')
    a(f'The hold before the movement has an ordinary amplitude in both tremor bands (3-7 Hz {N.rank("pre_rms_3_7")}, '
      f'7-12 Hz {N.rank("pre_rms_7_12")}), and so does the phone at the nose (3-7 Hz {N.rank("nose_rms_3_7")}, 7-12 Hz '
      f'{N.rank("nose_rms_7_12")}). Two design limits bound these statements: the still gate before GO keeps large '
      f'hold movements out of the hold window (they restart the hold), and the nose window exists only for trials that '
      f'became still. {case}\'s false starts ({N("false_starts_pd")} per trial, {N.rank("false_starts")}) are recorded '
      f'without their cause.')
    a('')
    a('### Multiplicity and the parametric complement')
    a('')
    a(f'Family: the {N("n_prespecified")} pre-specified characteristics, each in its matched stratum. No pre-specified '
      f'characteristic reaches q < 0.05: the smallest Benjamini-Hochberg q is {N("min_q_rank")} for the rank test (its '
      f'p cannot fall below {N("p_floor")}) and {N("min_q_ch")} for the Crawford-Howell test (smallest p '
      f'{N("min_p_ch_prespecified")}, {N("min_p_ch_prespecified_label")}), which assumes normally distributed reference '
      f'values. '
      + (f'The smallest post hoc Crawford-Howell p ({N("posthoc_min_pch_label")}: {N("posthoc_min_pch")} on the '
         f'{N("posthoc_min_pch_scale")} scale, {N("posthoc_min_pch_alt")} on the other) is not evidence: that '
         f'characteristic was chosen after looking at {case}, and its p depends on the scale.'
         if N.has('posthoc_min_pch') else '')
      + ' The rank p of the count of differences and of the Mahalanobis distance are single tests of the whole '
        'profile, outside the characteristic family.')
    a('')
    a('### Null reference of the count')
    a('')
    for sk, sname in NULL_SETS.items():
        for vk, vname in NULL_VARIANTS.items():
            p_ = f'null_{sk}_{vk}'
            ge = N.v(f'{p_}_ge_persons')
            a(f'- {"Pre-specified" if sname == "prespecified" else "All"} characteristics, '
              f'{vname.replace("_", " ")}: {case} {N(f"{p_}_case")} '
              f'({N(f"{p_}_case_phen")} phenomena; {N(f"{p_}_subset_min")}-{N(f"{p_}_subset_max")} against any '
              f'{N("n_ref_minus_one")} of them); reference persons median {N(f"{p_}_ref_median")}, maximum '
              f'{N(f"{p_}_ref_max")} ({N(f"{p_}_top")}); {N(f"{p_}_n_ge")} '
              f'{N.verb(f"{p_}_n_ge", "reaches", "reach")} {case}\'s count'
              + (f' ({ge})' if ge else '') + f'; rank p {N(f"{p_}_p")} (phenomena {N(f"{p_}_p_phen")}).')
    a('')
    a(f'The post hoc characteristics were chosen after looking at {case}, which inflates {case}\'s count in the "all" '
      f'rows; the like-for-like rows are the fair comparison, since {case}\'s own values are means of few trials.')
    a('')
    a('### Multivariate view')
    a('')
    a(f'{case} lies at PC1 = {N("mv_pc_one_case")} (reference range {N("mv_pc_one_ref_min")} to '
      f'{N("mv_pc_one_ref_max")}; PC1 explains {N("mv_pc_one_var")} of the variance and loads mostly on '
      + '; '.join(XS.lc(XS.BY_KEY[k].label) for k, _ in
                  sorted(zip(mv['keys'], mv['loadings'][0]), key=lambda kv: -abs(kv[1]))[:3])
      + f'). Its Mahalanobis distance is {N("mv_distance")}; {N("mv_n_beyond")} of the {N("n_ref_persons")} reference '
      f'persons (leave-one-out) {N.verb("mv_n_beyond")} at least as far (rank p {N("mv_p")}, floor {N("mv_p_floor")}); with one '
      f'characteristic left out, {N("mv_loco_min_beyond")} to {N("mv_loco_max_beyond")} are. On linear scales the '
      f'distance is {N("mv_lin_distance")} with {N("mv_lin_n_beyond")} farther (rank p {N("mv_lin_p")}), and '
      f'{N("mv_lin_loco_max_beyond")} without {N("mv_lin_loco_max_label")}'
      + (': the linear result depends on that one characteristic.'
         if N.v('mv_lin_loco_max_beyond') > N.v('mv_lin_n_beyond') else '.'))
    a('')
    a('### All characteristics')
    a('')
    out.extend(_md_table(m, N, case))
    a('')
    a('## Confounds and caveats')
    a('')
    a(f'- **One person, few trials.** {N("case_n_accepted")} accepted trials, {N("case_n_primary")} of them '
      f'instructed ({per_speed}). The case\'s values are means of {N("case_min_trials")} to {N("case_max_trials")} '
      f'trials; the speed-scaling ratios rest on {N("case_n_lua_slow")} slow and {N("case_n_lua_fast")} fast trial(s), '
      f'and the across-repeat measures on {N("cv_duration_ncase")} reaches from {N("case_repeat_sessions")} sessions '
      f'against {N("cv_duration_nref")} reference persons with as many left-hand reaches.')
    a(f'- **Age.** The case is in the {N("case_age_band")} age band; every reference person is {N("ref_age_min")}-'
      f'{N("ref_age_max")}. Age alone slows and changes reaching; nothing here separates the disease from age.')
    a(f'- **Device, sampling, body and posture.** One {plat} phone (its sampling, noise floor and accelerometer range '
      f'differ from other phones; same-platform subsets are reported), phone size and mass, grip and arm length, and a '
      f'seated posture whose forearm may have been supported (the reference: '
      + ', '.join(f'{N(f"ref_sessions_{_slug(k)}")} {k}' for k in sorted(rref['postures']))
      + ' sessions; arm support was not recorded). Low slow sway in the hold may reflect support, not the person.')
    a(f'- **Hand.** The instructed trials are {hands_p}-hand and the free-choice trials {hands_f}-hand (reported); the '
      f'case is {c["handedness"]}-handed and usually holds the phone in the {c["phone_hand"]} hand.')
    statuses = '; '.join(STATUS_WORDS.get(s['status'], s['status']) for s in c['sessions'])
    a(f'- **Protocol and practice.** The case reached tier(s) {N("case_tiers")} of the plan (the start '
      f'of a visit); its sessions: {statuses}; the reference sessions: '
      + ', '.join(f'{N(f"ref_sessions_status_{_slug(k)}")} {STATUS_WORDS.get(k, k)}' for k in sorted(rref['session_status']))
      + '. The first-k analysis matches the practice level.')
    a('- **Mapping and clipping.** The rotation-rate mapping was assumed, not verified (see Data); acceleration '
      'magnitudes are lower bounds in clip-flagged trials.')
    a(f'- **The first touch** is the first touchstart after GO. It is recorded in {N("touch_share_ref")} of the reference '
      f'left-hand upright natural trials and in {N("case_touch_share")} of the case\'s, so the touch characteristics '
      f'compare fewer reference persons ({N("touch_s_nref")}). It is not known what touched the screen.')
    a('- **Segmentation.** The segmented approach ends when the phone is still; a residual movement at the face delays '
      'that end, so the approach duration includes a settling phase. The active movement time and the settling time '
      'separate the two.')
    a(f'- **Multiplicity and post hoc choices.** {N("n_characteristics")} characteristics, {N("n_post_hoc")} defined '
      f'after seeing the case or the audit; everything is exploratory and hypothesis-generating.')
    a('')
    a('## What a follow-up study would need')
    a('')
    gap = (f' (the reference persons here are at least {N("age_gap_min")} years younger)'
           if N.has('age_gap_min') else '')
    cand_pre = sorted({XS.BY_KEY[k].phenomenon for k in pre.loc[pre['robust'].astype(bool), 'key']},
                      key=phen_order.index)
    cand_post = sorted({XS.BY_KEY[k].phenomenon for k in post.loc[post['robust'].astype(bool), 'key']}
                       - set(cand_pre), key=phen_order.index)
    a(f'- Several participants with Parkinson\'s disease and age-, sex- and device-matched controls{gap}, with '
      f'clinical scores (for example MDS-UPDRS part III and its tremor items) and the medication state (ON / OFF) '
      f'recorded at the visit.')
    a('- Both hands instructed, several repeats per condition (at least 5 per speed), arm support recorded, and a '
      'longer still hold whose gate is logged rather than enforced, to measure rest and re-emergent tremor.')
    a(f'- A pre-registered, small set of characteristics: the phenomena with a robust pre-specified difference here ('
      + ('; '.join(cand_pre) or 'none') + ')' + (f', and as hypotheses the post hoc ones ({"; ".join(cand_post)})'
                                                  if cand_post else '')
      + ', with an analysis plan using a mixed model over trials instead of single-case ranks.')
    a('- Video or a contact sensor to establish what the first screen touch is, and a verified rotation-rate mapping.')
    a('')
    a('## Analyses and outputs')
    a('')
    a('This lane runs no `run_experiment` experiment (no model; `results/registry_x1.csv` is not written and '
      '`results/finalists_x1.json` is an empty list). Its analyses (`python -m stage2.pipeline compute --lanes x1`):')
    a('')
    a('| Analysis | Saved to | What |')
    a('|---|---|---|')
    a(f'| x1_cache | `analysis/data/exploratory_cache/{case}/` | {case}\'s cache, built by `data.build_cache` |')
    a('| x1_characteristics | `results/x1_characteristics.json`, `tables/x1/x1_trial_characteristics.csv`, '
      '`x1_person_values.csv`, `x1_comparisons.csv` | per-trial characteristics, person values, every comparison and '
      'sensitivity analysis, robustness, the null of the count, the redundancy of the findings |')
    a('| x1_checks | `tables/x1/x1_checks.json` | speed-matched and band-pass checks of the reach power, touch timing, '
      'net-yaw reversal, ratio decomposition, spectral resolution and leakage |')
    a('| x1_spectra | `tables/x1/x1_spectra.csv` | gyroscope spectra of the three windows |')
    a('| x1_traces | `tables/x1/x1_traces.csv`, `x1_traces.json` | example recordings (canonical channels) |')
    a('| x1_yaw_curves | `tables/x1/x1_yaw_curves.csv` | mirror-normalised yaw over the approach |')
    a('| x1_multivariate | `results/x1_multivariate.json` | PCA map, Mahalanobis test, linear and leave-one-out checks |')
    a('| x1_task | `tables/x1/x1_task.json` | attempts, rejections, sessions, mapping, clip flags |')
    a('')
    a('**Headline model numbers (accuracy, AUC, log loss, ECE, natural-use accuracy, k = 3 accuracy, per-platform '
      'and per-condition breakdowns): not applicable.** This lane fits no model and, by design, reports no '
      f'model-performance number for {case}. The per-condition view of the characteristics is `tables/x1/x1_strata`, '
      'the per-platform view the same-platform analysis in `tables/x1/x1_sensitivity`.')
    a('')
    a('## Figures (`figures/x1/`)')
    a('')
    for name in x1_texts.FIGURE_NAMES:
        a(f'- `{name}`: {x1_texts.texts(name)["decision"]}')
    a('')
    a('## Tables (`tables/x1/`)')
    a('')
    a('- `x1_key_characteristics` (main text), `x1_characteristics` (appendix, every characteristic), '
      '`x1_characteristics_values.csv` (full precision, with the rationale of each characteristic), `x1_sensitivity` '
      '(every sensitivity analysis and the robustness verdict), `x1_null_count` (the null reference of the count), '
      '`x1_checks`, `x1_multivariate_loco`, `x1_strata`, `x1_attempts`; numbers in `results/x1_numbers.json` (macros '
      '`\\XOne...`).')
    a('')
    a('## Commands (from `analysis/`)')
    a('')
    a('```')
    a('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes x1           # stale analyses only (a few minutes)')
    a('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes x1 --force   # everything, PD01\'s cache included')
    a('.venv/Scripts/python.exe -m stage2.pipeline figures --lanes x1           # redraw from saved files')
    a('.venv/Scripts/python.exe -m stage2.pipeline tables --lanes x1')
    a('.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes x1           # numbers, finalists [], this summary')
    a('.venv/Scripts/python.exe -m pytest stage2/tests/test_x1.py -q -p no:cacheprovider')
    a('```')
    a('')
    a('Presentation settings are in `x1_figures.FIGURES` (one entry per figure); compute parameters in '
      '`x1_features.PARAMS`; the characteristics, their strata, scales and phenomena in `x1_stats.CHARACTERISTICS`; '
      'the robustness settings in `x1_stats` (`MIN_REF`, `Z_CLEAR`, `VARIANTS`); the case in `x1_compute.CASE`. A new '
      'modelling person (main cache rebuilt) makes every x1 analysis stale, and a plain `compute` recomputes them.')
    a('')
    a('## Open questions')
    a('')
    a(f'- Is the 7-12 Hz band power of the reach an action tremor, or the dynamics of a fast reach that stops against '
      f'the face? It stays high before the first touch ({N("chk_bp_pretouch_pd")} deg/s, reference maximum '
      f'{N("chk_bp_pretouch_max")}) and without it, and {N.peak()}; but one person, without a clinical examination of '
      f'the tremor, cannot separate the two.')
    a('- What touched the screen, and why so early? Video or a contact sensor would settle it.')
    a('- Would the hand signature look the same with the dominant hand instructed and with more repeats, and why does '
      'the net yaw turn back after the early turn?')
    a('')
    return '\n'.join(out) + '\n'


# ───────────────────────────── the step ─────────────────────────────


def write_all(only: list[str] | None = None) -> None:
    D, N = current_numbers()
    if selected('x1_numbers', only):
        L.write_numbers(N.e)
    if selected('finalists_x1', only):
        L.write_finalists([])
    if selected('x1_summary', only):
        L.summary_path.write_text(summary(D, N), encoding='utf-8', newline='\n')
