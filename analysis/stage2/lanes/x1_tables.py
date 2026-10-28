"""Tables of lane x1 (CSV + LaTeX booktabs), from saved files only:

    tables/x1/x1_characteristics.csv|.tex      every characteristic in its matched comparison (appendix)
    tables/x1/x1_characteristics_values.csv    the same, numeric at full precision, with the rationale (no .tex)
    tables/x1/x1_key_characteristics.csv|.tex  the compact key set (main text)
    tables/x1/x1_sensitivity.csv|.tex          every sensitivity analysis and the robustness verdict
    tables/x1/x1_null_count.csv|.tex           the null reference of the "beyond every reference person" count
    tables/x1/x1_checks.csv|.tex               targeted checks of alternative explanations
    tables/x1/x1_multivariate_loco.csv|.tex    the Mahalanobis test with each characteristic left out
    tables/x1/x1_redundancy.csv                Spearman rho between the "beyond all" characteristics (no .tex)
    tables/x1/x1_strata.csv|.tex               percentiles of the per-trial characteristics in every matched stratum
    tables/x1/x1_attempts.csv|.tex             the exploratory person's accepted trials (no ids, no times)

Every number comes from the saved results (``x1_report.current_numbers`` where the summary cites it too).
Run: python -m stage2.pipeline tables --lanes x1
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from stage2 import report_export as RX
from stage2.pipeline import Lane, selected
from stage2.lanes import x1_report as XR
from stage2.lanes import x1_stats as XS

L = Lane('x1')
STRATUM_SHORT = {'left_upright_natural': 'L-up nat', 'left_upright_slow': 'L-up slow', 'left_upright_fast': 'L-up fast',
                 'left_upright_all': 'L-up all', 'upright_natural_both': 'up nat, either', 'free_choice': 'free',
                 'all_trials': 'all'}
NOTE_STRATA = ('Strata: L-up = left hand, upright grip, standard movement, natural start (nat / slow / fast / all '
               'speeds); up nat, either = upright natural reaches of either hand; free = first free-choice trial '
               '(natural-use set); all = every accepted trial.')


def pcell(v: Any) -> str:
    """A p-value cell: three decimals, '<0.001' below that."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return '--'
    if not math.isfinite(x):
        return '--'
    return '<0.001' if x < 0.001 else f'{x:.3f}'


def _v(v: Any, fmt: str) -> str:
    return RX.format_value(v, fmt) if v is not None and not (isinstance(v, float) and math.isnan(v)) else '--'


def _rank(r: pd.Series) -> str:
    if pd.isna(r.get('percentile')):
        return '--'
    if pd.notna(r.get('n_more_extreme')) and int(r['n_more_extreme']) == 0:
        return 'above all' if r['pd_value'] > r['ref_median'] else 'below all'
    return f"{r['percentile']:.0f}"


def _unit(u: str) -> str:
    """The unit as LaTeX (the unit column is written raw: m/s$^2$ stays math, a bare % is escaped)."""
    return RX.latex_escape(u).replace(r'm/s\$\textasciicircum{}2\$', 'm/s$^2$')


def _label(r: pd.Series) -> str:
    """Characteristic (unit) *, as LaTeX."""
    u = _unit(r['unit']) if isinstance(r['unit'], str) and r['unit'] else ''
    return RX.latex_escape(r['label']) + (f' ({u})' if u else '') + (' *' if bool(r['post_hoc']) else '')


def _robust(r: pd.Series) -> str:
    if not bool(r['beyond_all']):
        return ''
    return 'yes' if bool(r['robust']) else 'no'


def _trials_beyond(r: pd.Series) -> str:
    if r['level'] != 'trial' or pd.isna(r.get('n_case_trials_valid')):
        return '--'
    return f"{int(r['n_case_trials_beyond'])}/{int(r['n_case_trials_valid'])}"


def characteristics(only: list[str] | None) -> None:
    D, N = XR.current_numbers()
    m, case = D['main'], D['case']
    rows = []
    for _, r in m.iterrows():
        f = r['fmt']
        rows.append({
            'characteristic': _label(r), 'stratum': STRATUM_SHORT[r['stratum']], 'n_case': int(r['n_case_trials']),
            case: _v(r['pd_value'], f),
            'reference_median_iqr': f"{_v(r['ref_median'], f)} [{_v(r['ref_q1'], f)}, {_v(r['ref_q3'], f)}]",
            'n_ref': int(r['n_ref']), 'percentile': _rank(r),
            'robust_z': r['robust_z'] if bool(r['report_z']) else math.nan,
            'trials_beyond': _trials_beyond(r), 'p_rank': r['p_rank'], 'q_rank': r['q_rank'],
            'p_ch': pcell(r['ch_p']), 'robust': _robust(r),
        })
    df = pd.DataFrame(rows)
    n = N('n_ref_persons')
    head = {'characteristic': 'Characteristic', 'stratum': 'Stratum', 'n_case': 'n',
            'reference_median_iqr': 'Reference median [IQR]', 'n_ref': 'n ref', 'percentile': 'Pct',
            'robust_z': 'z', 'trials_beyond': 'Trials beyond', 'p_rank': 'p (rank)', 'q_rank': 'q',
            'p_ch': 'p (C-H)', 'robust': 'Robust'}
    if selected('x1_characteristics', only):
        L.export_table(df, 'x1_characteristics',
                       caption=(f'{case} against the {n} modelling persons, one row per characteristic in its matched '
                                f'comparison (exploratory). Reference: median [IQR] of the person-level values; Pct: '
                                f'mid-rank percentile of {case} (above / below all: beyond every reference person); z: '
                                f'robust z on the declared scale (log for positive, right-skewed characteristics; none '
                                f'for shares of binary outcomes); Trials beyond: {case}\'s trials beyond the reference '
                                f'persons\' range; p (rank): two-sided, floor 2/(n+1) = {N("p_floor")}; q: '
                                f'Benjamini-Hochberg over the {N("n_prespecified")} pre-specified characteristics; p '
                                f'(C-H): Crawford-Howell test on the declared scale; Robust: a beyond-all verdict that '
                                f'survives every check that can fail (Table x1\\_sensitivity). *: post hoc, no q.'),
                       column_formats={'robust_z': '+.1f', 'p_rank': '.3f', 'q_rank': '.2f'},
                       header=head, note=NOTE_STRATA, size=r'\scriptsize', placement='p',
                       raw=('characteristic',))
        num = m[['key', 'label', 'group_name', 'phenomenon', 'unit', 'stratum', 'n_case_trials', 'pd_value', 'n_ref',
                 'ref_median', 'ref_q1', 'ref_q3', 'ref_min', 'ref_max', 'percentile', 'robust_z', 'z_scale',
                 'robust_z_linear', 'z_loo_min', 'p_rank', 'p_floor', 'n_more_extreme', 'q_rank', 'ch_t', 'ch_p',
                 'ch_scale', 'ch_p_alt', 'q_ch', 'n_case_trials_beyond', 'n_case_trials_valid', 'beyond_all', 'robust',
                 'checks_run', 'checks_failed', 'post_hoc', 'key_set', 'multivariate', 'report_z', 'rationale']]
        L.export_table(num, 'x1_characteristics_values', tex=False)
    if selected('x1_key_characteristics', only):
        k = df[m['key_set'].astype(bool).to_numpy()].drop(columns=['stratum', 'n_ref', 'p_ch', 'p_rank'])
        L.export_table(k, 'x1_key_characteristics',
                       caption=(f'Key characteristics of {case} against the {n} modelling persons (exploratory; the '
                                f'full table is in the appendix). z: robust z on the declared scale; Trials beyond: '
                                f'{case}\'s trials beyond the reference range; q: Benjamini-Hochberg over the '
                                f'pre-specified characteristics; Robust: see the sensitivity table. *: post hoc.'),
                       column_formats={'robust_z': '+.1f', 'q_rank': '.2f'},
                       header={kk: v for kk, v in head.items() if kk in k.columns}, note=NOTE_STRATA,
                       raw=('characteristic',))


def sensitivity(only: list[str] | None) -> None:
    if not selected('x1_sensitivity', only):
        return
    D = XR.current_numbers()[0]
    m, case = D['main'], D['case']
    N = XR.current_numbers()[1]
    rows = []
    for _, r in m.iterrows():
        def cell(var: str) -> str:
            p, ne = r.get(f'percentile_{var}'), r.get(f'n_more_extreme_{var}')
            if p is None or pd.isna(p):
                return '--'
            changed = (pd.notna(r.get(f'n_ref_changed_{var}')) and int(r.get(f'n_ref_changed_{var}')) > 0) \
                or r.get(f'case_changed_{var}') in (True, 1)
            mark = '' if (changed or var == 'first_k') else ' (s)'
            return ('beyond' if int(ne) == 0 else f'{p:.0f}') + mark

        size = ''
        if bool(r['beyond_all']) and bool(r['report_z']):
            size = 'pass' if bool(r['size_clear']) else 'fail'
        rows.append({'characteristic': _label(r), 'main': _rank(r).replace(' all', ''),
                     **{v: cell(v) for v in XS.VARIANTS[1:]},
                     'size': size, 'robust': _robust(r)})
    df = pd.DataFrame(rows)
    L.export_table(df, 'x1_sensitivity',
                   caption=(f'Sensitivity of {case}\'s percentile (beyond: outside every reference person). Main: each '
                            f'person\'s mean over all matched trials; first k: each reference person on their first k '
                            f'matched trials, k = {case}\'s count; same protocol: reference trials of {case}\'s '
                            f'protocol version only; no clip: every person without clip-flagged trials; same posture / '
                            f'same platform: reference persons of {case}\'s posture / platform. (s): a pure subset of '
                            f'persons whose values did not change, which cannot overturn a beyond-all verdict. Size: '
                            f'|z| >= {N("z_clear")} on the declared scale with each reference person left out. Robust: '
                            f'beyond in the main analysis and in every check that recomputed values, and the size '
                            f'check passed. --: not computed, either not applicable (first k for person-level '
                            f'characteristics, no clip for the attempts) or too few case trials left, which withholds '
                            f'"robust". *: post hoc.'),
                   header={'characteristic': 'Characteristic', 'main': 'Main', 'first_k': 'First k',
                           'same_protocol': 'Same protocol', 'no_clip': 'No clip', 'same_posture': 'Same posture',
                           'same_platform': 'Same platform', 'size': 'Size', 'robust': 'Robust'},
                   size=r'\scriptsize', placement='p', raw=('characteristic',))


def null_count(only: list[str] | None) -> None:
    if not selected('x1_null_count', only):
        return
    D = XR.current_numbers()[0]
    case = D['case']
    N = XR.current_numbers()[1]
    frames = []
    for sname in ('prespecified', 'all'):
        for vname in ('like_for_like', 'main'):
            pp = pd.DataFrame(D['null'][sname][vname]['per_person'])
            frames.append(pp[['person_id', 'n_characteristics', 'n_phenomena']].rename(
                columns={'n_characteristics': f'{sname}_{vname}_n', 'n_phenomena': f'{sname}_{vname}_phen'})
                .set_index('person_id'))
    df = pd.concat(frames, axis=1).reset_index()
    df['_o'] = (df['person_id'] == case).astype(int)
    df = df.sort_values(['_o', 'person_id']).drop(columns='_o').reset_index(drop=True)
    L.export_table(df, 'x1_null_count',
                   caption=(f'The null reference of the count of characteristics (n) and phenomena (phen) in which a '
                            f'person lies beyond every reference person: each reference person against the other '
                            f'{N("n_ref_minus_one")}, {case} (last row) against all {N("n_ref_persons")}. Like for '
                            f'like: each reference person judged on the mean of their first k trials, as many as {case} '
                            f'made; main: everyone on all trials; all: the post hoc characteristics included (chosen '
                            f'after looking at {case}).'),
                   header={'person_id': 'Person', 'prespecified_like_for_like_n': 'pre, lfl: n',
                           'prespecified_like_for_like_phen': 'phen', 'prespecified_main_n': 'pre, main: n',
                           'prespecified_main_phen': 'phen', 'all_like_for_like_n': 'all, lfl: n',
                           'all_like_for_like_phen': 'phen', 'all_main_n': 'all, main: n', 'all_main_phen': 'phen'},
                   size=r'\scriptsize')


def checks(only: list[str] | None) -> None:
    if not selected('x1_checks', only):
        return
    D, N = XR.current_numbers()
    case = D['case']
    rows = [
        {'check': 'Reach 7-12 Hz RMS against speed-matched reference reaches (deg/s)',
         'case': f'{N("chk_sm_case_min")}-{N("chk_sm_case_max")}',
         'reference': f'max {N("chk_sm_ref_max")} ({N("chk_sm_n_ref_trials")} trials, {N("chk_sm_n_ref_persons")} persons)',
         'result': f'{N("chk_sm_n_case_above")} of {N("chk_sm_n_case")} case reaches above all'},
    ]
    for short, what in (('whole', 'whole reach'), ('pretouch', 'before the first touch'),
                        ('excltouch', 'without the first touch')):
        rows.append({'check': f'Reach 7-12 Hz band-pass RMS, {what} (deg/s)', 'case': N(f'chk_bp_{short}_pd'),
                     'reference': f'median {N(f"chk_bp_{short}_med")}, max {N(f"chk_bp_{short}_max")} '
                                  f'({N(f"chk_bp_{short}_nref")} persons)',
                     'result': f'{N(f"chk_bp_{short}_nmore")} persons as high'})
    rows += [
        {'check': 'Prominence of the reach\'s 3-12 Hz spectral peak (dB, mean over trials)', 'case': N('chk_peak_pd'),
         'reference': f'median {N("chk_peak_med")}, max {N("chk_peak_max")} ({N("chk_peak_nref")} persons)',
         'result': f'{XR.ordinal(N("chk_peak_pct"))} percentile'},
        {'check': 'Reach 7-12 Hz RMS per peak speed *', 'case': N('app_7_12_per_speed_pd'),
         'reference': f'median {N("app_7_12_per_speed_med")}, max {N("app_7_12_per_speed_max")}',
         'result': N.rank('app_7_12_per_speed')},
        {'check': f'First touch within {N("chk_tap_window_ms")} ms of the largest acceleration',
         'case': f'{N("chk_tap_case_k")} of {N("chk_tap_case_n")} trials',
         'reference': f'{N("chk_tap_ref_share")} of {N("chk_tap_ref_n")} touched trials', 'result': ''},
        {'check': 'Rotation after the first touch (share of the voluntary rotation) *', 'case': N('rot_after_touch_pd'),
         'reference': f'median {N("rot_after_touch_med")}, max {N("rot_after_touch_max")}',
         'result': N.rank('rot_after_touch')},
        {'check': 'Early yaw peak to first touch (s)',
         'case': f'median {N("chk_yt_case_median")} ({N("chk_yt_case_min")}-{N("chk_yt_case_max")})',
         'reference': f'median {N("chk_yt_ref_median")} (touched trials)', 'result': ''},
        {'check': 'Net yaw against the typical direction (share of trials)',
         'case': f'{N("chk_ny_k")} of {N("chk_ny_n")}',
         'reference': f'median {N("chk_ny_ref_median")}, max {N("chk_ny_ref_max")}',
         'result': f'{N("chk_ny_n_ref_ge")} persons as high'},
    ]
    for w, lab in (('pre', 'hold'), ('app', 'reach'), ('nose', 'nose')):
        if N.has(f'dec_{w}_05_3_pct'):
            rows.append({'check': f'Band-power ratio of the {lab}: components (percentiles)',
                         'case': (f'0.5-3 Hz {N(f"dec_{w}_05_3_pct")}, 3-12 Hz {N(f"dec_{w}_3_12_pct")}, '
                                  f'12-15 Hz {N(f"dec_{w}_12_15_pct")}'),
                         'reference': '', 'result': ''})
    for w, lab in (('pre', 'hold'), ('nose', 'nose')):
        rows.append({'check': f'{N("res_tone_hz")}-Hz tone (RMS {N("res_tone_rms")} deg/s) in the {lab} window',
                     'case': f'{N(f"res_{w}_leak_7_12")} deg/s RMS in 7-12 Hz',
                     'reference': f'resolution {N(f"res_{w}_resolution_hz")} Hz', 'result': ''})
    df = pd.DataFrame(rows)
    L.export_table(df, 'x1_checks',
                   caption=(f'Checks of alternative explanations for {case}\'s differences (left-hand upright natural '
                            f'reaches unless stated; *: post hoc). Every value is a numbers entry.'),
                   header={'check': 'Check', 'case': case, 'reference': 'Reference', 'result': 'Result'},
                   size=r'\scriptsize', align='p{6.2cm}lp{4.2cm}l')


def multivariate_loco(only: list[str] | None) -> None:
    if not selected('x1_multivariate_loco', only):
        return
    mv = XR.current_numbers()[0]['mv']
    lo = pd.DataFrame(mv['loco'])
    ll = pd.DataFrame(mv['linear']['loco']).set_index('left_out')
    df = pd.DataFrame({'left_out': [XS.BY_KEY[k].label for k in lo['left_out']],
                       'distance': lo['mahalanobis_x'], 'n_beyond': lo['n_ref_beyond'], 'p': lo['p_mahalanobis'],
                       'distance_linear': ll.loc[lo['left_out'], 'mahalanobis_x'].to_numpy(),
                       'n_beyond_linear': ll.loc[lo['left_out'], 'n_ref_beyond'].to_numpy()})
    N = XR.current_numbers()[1]
    L.export_table(df, 'x1_multivariate_loco',
                   caption=(f'The Mahalanobis test with one key characteristic left out (declared scales; linear '
                            f'scales alongside): {mv["case"]}\'s distance and the number of the {N("n_ref_persons")} '
                            f'reference persons at least as far. All characteristics: {N("mv_n_beyond")} as far on the '
                            f'declared scales, {N("mv_lin_n_beyond")} on linear scales.'),
                   column_formats={'distance': '.2f', 'n_beyond': 'd', 'p': '.3f', 'distance_linear': '.2f',
                                   'n_beyond_linear': 'd'},
                   header={'left_out': 'Left out', 'distance': 'Distance', 'n_beyond': 'As far', 'p': 'p',
                           'distance_linear': 'Distance (lin)', 'n_beyond_linear': 'As far (lin)'})


def redundancy(only: list[str] | None) -> None:
    if not selected('x1_redundancy', only):
        return
    red = XR.current_numbers()[0]['redundancy']
    if len(red):
        L.export_table(red, 'x1_redundancy', tex=False)


def strata(only: list[str] | None) -> None:
    if not selected('x1_strata', only):
        return
    cmp = pd.read_csv(L.tab_dir / 'x1_comparisons.csv')
    cmp = cmp[(cmp['variant'] == 'main') & (cmp['level'] == 'trial')]
    order = {c.key: i for i, c in enumerate(XS.CHARACTERISTICS)}
    rows = []
    for key, g in sorted(cmp.groupby('key'), key=lambda kv: order[kv[0]]):
        row = {'characteristic': XS.BY_KEY[key].label + (' *' if XS.BY_KEY[key].post_hoc else '')}
        for s in XS.REPORT_STRATA:
            r = g[g['stratum'] == s]
            row[STRATUM_SHORT[s]] = _rank(r.iloc[0]) if len(r) else '--'
        rows.append(row)
    df = pd.DataFrame(rows)
    case = XR.current_numbers()[0]['case']
    ncase = {STRATUM_SHORT[s]: int(cmp.loc[cmp['stratum'] == s, 'n_case_stratum'].max()) for s in XS.REPORT_STRATA}
    L.export_table(df, 'x1_strata',
                   caption=(f'{case}\'s percentile for every per-trial characteristic in each matched stratum (above / '
                            f'below all: beyond every reference person). Case trials per stratum: '
                            + ', '.join(f'{k} {v}' for k, v in ncase.items()) + '. *: post hoc.'),
                   header={'characteristic': 'Characteristic'}, note=NOTE_STRATA, size=r'\scriptsize', placement='p')


def attempts(only: list[str] | None) -> None:
    if not selected('x1_attempts', only):
        return
    t = pd.read_csv(L.tab_dir / 'x1_trial_characteristics.csv')
    task = L.read_json('x1_task')
    case = task['case']['person_id']
    c = t[t['group'] == 'exploratory'][['trial_key', 'analysis_set', 'hand', 'hand_source', 'orientation', 'speed',
                                        'false_starts', 'clip_candidate']]
    df = c.rename(columns={'trial_key': 'trial', 'analysis_set': 'set', 'hand_source': 'label source',
                           'orientation': 'grip', 'false_starts': 'false starts', 'clip_candidate': 'clip flag'})
    df['clip flag'] = df['clip flag'].map(lambda v: 'yes' if bool(v) else '')
    rej = pd.DataFrame([{'reason': k, 'attempts': v} for k, v in task['case']['rejections'].items()])
    L.export_table(df, 'x1_attempts',
                   caption=(f'The accepted, analysable trials of {case} ({len(df)} of {task["case"]["n_attempts"]} '
                            f'attempts; set: primary = instructed, natural\\_use / natural\\_use\\_repeat = first / '
                            f'later free-choice trial; clip flag: raw |a| above 3.5 g). Rejected attempts: '
                            + '; '.join(f'{r.reason} ({r.attempts})' for r in rej.itertuples()) + '.'),
                   column_formats={'false starts': 'd'})


def write(only: list[str] | None = None) -> None:
    characteristics(only)
    sensitivity(only)
    null_count(only)
    checks(only)
    multivariate_loco(only)
    redundancy(only)
    strata(only)
    attempts(only)
