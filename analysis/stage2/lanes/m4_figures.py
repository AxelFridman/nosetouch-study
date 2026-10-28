"""Figures of lane m4, drawn from saved files only (results/m4_*.json, tables/m4/*). Every presentation choice
(sizes, limits, ticks, labels, colours, which series) is in FIGURES or LABELS; edit there, then:

    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes m4 [--only M4_03_stress]

Accuracies are half-credit accuracies (an abstention, P = 0.5, scores half a correct answer; m4_stats).
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from stage2.pipeline import Lane, selected
from stage2.plotting import HAND_COLORS, INK2, MUTED, RAMP, SERIES, figure

L = Lane('m4')

ORANGE, DARK, LIGHT, GOLD, GREEN = '#eb6834', '#52514e', '#c3c2b7', '#eda100', '#008300'
OUTCOME_STYLE = {   # outcome -> (label, colour)
    'approach': ('the approach', SERIES),
    'approach_partial': ('part of the approach', RAMP[0]),
    'earlier_movement': ('an earlier movement', MUTED),
    'return': ('the return', ORANGE),
    'later_movement': ('a later movement', LIGHT),
    'rejected': ('rejected (P(approach) < threshold)', DARK),
    'none': ('no movement found', '#e1e0d9'),
}
LABELS = {
    'm4_oracle_yaw8': 'oracle window (no detection)', 'm4_oracle_chiral16': 'oracle window, chiral16',
    'm4_v0_yaw8': 'detector v0 (first movement)', 'm4_v1phys_yaw8': 'learned, physics features',
    'm4_v1full_yaw8': 'learned, physics + context', 'm4_v1nest_yaw8': 'learned, feature set nested',
    'm4_v1robust_yaw8': 'learned, robust (production detector)', 'm4_v1grip_yaw8': 'learned, grip-normalised',
    'm4_v1nest_yaw8_detwin': 'hand model on detector windows', 'm4_v1robust_chiral16': 'production detector, chiral16 (uncal.)',
    'm4_cal_temperature': 'calibrated: temperature', 'm4_cal_platt': 'calibrated: Platt',
    'm4_cal_isotonic': 'calibrated: isotonic', 'm4_cal_beta': 'calibrated: beta',
    'm4_cal_isotonic_sym': 'calibrated: symmetrised isotonic', 'm4_cal_nested': 'production (calibrator nested)',
    'm4_ret_app_yaw8': 'stored windows, approach', 'm4_ret_head_yaw8': 'stored windows, return head',
    'm4_ret_sum_yaw8': 'stored windows, approach + return',
    'm4_prod_return': 'production detector + return head (uncal.)',
    'm4_prod_zp_activity': 'production detector, zero-phase activity (uncal.)',
    'm4_prod_causal_lowpass': 'production detector, causal low-pass (uncal.)',
    'm4_abl_scale_z': 'ablation: z-scored segment features', 'm4_abl_scale_none': 'ablation: unscaled segment features',
    'm4_abl_segc_nested': 'ablation: segment-classifier C nested', 'm4_abl_look0p3': 'ablation: look-ahead 0.3 s',
    'm4_abl_look0p8': 'ablation: look-ahead 0.8 s',
}

FIGURES: dict[str, dict[str, Any]] = {
    'M4_01_detection_outcomes': dict(width='full', height=2.1, xlabel='share of recordings',
                                     detectors=('m4_v0_yaw8', 'm4_v1phys_yaw8', 'm4_v1full_yaw8', 'm4_v1nest_yaw8',
                                                'm4_v1robust_yaw8', 'm4_v1grip_yaw8'), baseline='m4_v0_yaw8',
                                     labels={'m4_v1robust_yaw8': 'learned, robust (production)'}),
    'M4_02_accuracy_ladder': dict(width='full', height=4.3, xlim=(0.4, 0.95), xticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                  xlabel='leave-one-person-out accuracy (abstention = half credit; 95 % person-bootstrap CI)',
                                  names=('m4_oracle_yaw8', 'm4_v0_yaw8', 'm4_v1phys_yaw8', 'm4_v1full_yaw8',
                                         'm4_v1nest_yaw8', 'm4_v1nest_yaw8_detwin', 'm4_v1robust_yaw8', 'm4_v1grip_yaw8',
                                         'm4_cal_nested', 'm4_prod_zp_activity', 'm4_prod_causal_lowpass',
                                         'm4_prod_return', 'm4_oracle_chiral16', 'm4_v1robust_chiral16')),
    'M4_03_stress': dict(width='full', height=2.6, ylim=(0.0, 1.0), acc_ylim=(0.35, 0.9),
                         # '{cap}' is filled with the stillness cap of the uncapped context features (m4_detect.SegParams)
                         pipelines={'production': ('production', SERIES), 'uncapped': ('context capped at {cap:g} s', ORANGE),
                                    'v0': ('detector v0', MUTED)}),
    'M4_04_attempts': dict(width='full', height=3.1, ylim=(0.4, 1.0), stop_ylim=(0.7, 0.95), gain_margin_points=2.0,
                           keep_colors=(DARK, GOLD, SERIES, GREEN, '#6250d6'),
                           aggregations={'scale_recordings': (ORANGE, 's--'), 'scale_sessions': (SERIES, 'o-'),
                                         'mean': (MUTED, '^:')}),
    'M4_05_calibration': dict(width='full', height=2.7, xlabel_diff='log loss minus uncalibrated (95 % CI)'),
    'M4_06_k3_calibration': dict(width='full', height=4.4,
                                 top=('mean', 'sum', 'scale_recordings', 'scale_sessions'),
                                 bottom=('scale_recordings', 'scale_sessions', 'mean'),
                                 colors={'mean': MUTED, 'sum': INK2, 'scale_recordings': ORANGE, 'scale_sessions': SERIES}),
    'M4_07_coverage': dict(width='half', height=2.6, ylim=(0.6, 1.0), min_n=10,
                           taus=tuple(np.round(np.arange(0.5, 0.96, 0.025), 3)),
                           det_thresholds=tuple(np.round(np.arange(0.0, 0.91, 0.05), 2))),
    'M4_08_timeline': dict(width='full', height=3.5, kind_colors={'approach': SERIES, 'approach_partial': RAMP[0],
                                                                  'return': ORANGE, 'other_before': MUTED,
                                                                  'other_after': LIGHT}),
    'M4_09_latency': dict(width='half', height=2.3, xlim=(-500, 2500), bins=40),
    'M4_10_return': dict(width='half', height=2.5, xlim=(0.4, 1.0),
                         rows=(('m4_ret_app_yaw8', 'approach only'), ('m4_ret_head_yaw8', 'return head only'),
                               ('m4_ret_sum_yaw8', 'approach + return'), ('m4_v1robust_yaw8', 'cue-free, approach'),
                               ('m4_prod_return', 'cue-free, + return'))),
    'M4_11_segment_features': dict(width='full', height=2.7,
                                   features=(('disp_z0', 'displacement along the\nscreen normal (m)', (-0.8, 0.8)),
                                             ('d_up_z', 'change of up_z (end - start)', (-0.8, 0.8)),
                                             ('disp_vert', 'vertical displacement (m)', (-0.4, 0.4)),
                                             ('still_before_inst', 'stillness before (s, capped)', (0, 1.05))),
                                   colors={'approach': SERIES, 'return': ORANGE, 'other': MUTED}, bins=31),
    'M4_12_screen_flip': dict(width='full', height=2.7, ylim=(0.0, 1.0),
                              pipelines={'production': 'production', 'grip_uncal': 'grip-normalised\nalternative (uncal.)',
                                         'v0': 'detector v0'},
                              # (label, screen-away twin?, with the rest-pose grip check?)
                              bars=(('screen toward', False, False), ('screen toward + grip check', False, True),
                                    ('screen away', True, False), ('screen away + grip check', True, True)),
                              colors={False: SERIES, True: ORANGE}),
}


def _o(name: str, kw: dict[str, Any]) -> dict[str, Any]:
    unknown = set(kw) - set(FIGURES[name])
    if unknown:
        raise TypeError(f'{name}: unknown settings {sorted(unknown)}')
    return {**FIGURES[name], **kw}


def _result(name: str) -> dict[str, Any] | None:
    p = L.result_path(name)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def _production() -> str:
    """Result name of the production pipeline, as recorded by the compute step (m4_calibration.json); raises
    when the compute step has not run (never a silent fallback)."""
    return L.read_json('m4_calibration')['production']


def _scores() -> dict[str, Any]:
    return L.read_json('m4_scores')['scores']


def _agg_label(how: str) -> str:
    from stage2.lanes.m4_online import AGGREGATION_LABELS
    return AGGREGATION_LABELS.get(how, how)


# ───────────────────────────── figures ─────────────────────────────


def fig_M4_01_detection_outcomes(**kw: Any) -> None:
    o = _o('M4_01_detection_outcomes', kw)
    det = pd.DataFrame(L.read_json('m4_detection')['rows'])
    det = det[(det['set'] == 'primary') & det['name'].isin(o['detectors'])].drop_duplicates('name', keep='first')
    det = det.set_index('name').loc[[n for n in o['detectors'] if n in set(det['name'])]].reset_index()
    det['label'] = det['name'].map({**LABELS, **o['labels']}).fillna(det['name'])
    fig, ax = figure(o['width'], o['height'])
    y = np.arange(len(det))[::-1]
    left = np.zeros(len(det))
    tidy = []
    for oc, (lab, col) in OUTCOME_STYLE.items():
        v = det[f'share_{oc}'].fillna(0).to_numpy()
        if not (v > 0).any():                     # a category no detector produced is left out of the legend
            continue
        ax.barh(y, v, left=left, color=col, height=0.62, label=lab, edgecolor='white', linewidth=0.4)
        for yi, li, vi in zip(y, left, v):
            if vi >= 0.06:
                ax.text(li + vi / 2, yi, f'{100 * vi:.0f}', ha='center', va='center', fontsize=6.5,
                        color='white' if oc in ('approach', 'earlier_movement', 'return', 'rejected') else '#0b0b0b')
        left += v
        tidy += [{'detector': n, 'outcome': oc, 'share': float(s)} for n, s in zip(det['name'], v)]
    ax.set_yticks(y, det['label'])
    ax.set_xlim(0, 1)
    ax.set_xlabel(o['xlabel'])
    ax.grid(axis='y', visible=False)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.42, -0.3), frameon=False, fontsize=6.5, handlelength=1.0)
    n = int(det['n'].iloc[0])
    sh = det.set_index('name')
    base = L.read_json('m4_calibration')['base']           # the production detector (uncalibrated)
    pc = lambda name, oc: 100 * float(sh.loc[name, f'share_{oc}']) if name in sh.index else math.nan  # noqa: E731
    v0 = o['baseline']
    L.save_fig(fig, 'M4_01_detection_outcomes', data=pd.DataFrame(tidy),
               caption=(f'What each cue-free detector hands to the hand classifier on the {n} recordings (one approach '
                        'each, full recording: rest, approach, hold, return), scored against the app\'s stored events '
                        '(numbers are percentages). v0 takes the first movement after a fixed stillness; the learned '
                        'detectors pick the candidate with the highest leave-one-person-out P(approach) and reject it '
                        'when that probability is below the threshold.'),
               decision=(f'Production detector against v0: the scored segment is the approach in {pc(base, "approach"):.0f} % '
                         f'against {pc(v0, "approach"):.0f} %, an earlier movement in {pc(base, "earlier_movement"):.0f} % '
                         f'against {pc(v0, "earlier_movement"):.0f} % and the return in {pc(base, "return"):.0f} % against '
                         f'{pc(v0, "return"):.0f} %; the production detector rejects {pc(base, "rejected"):.0f} % (no '
                         'answer) instead of scoring them.'))


def fig_M4_02_accuracy_ladder(**kw: Any) -> None:
    o = _o('M4_02_accuracy_ladder', kw)
    sc = _scores()
    rows = []
    for n in dict.fromkeys(o['names']):
        r = _result(n)
        if r is None or n not in sc:
            continue
        s = sc[n]['primary']
        nat = sc[n].get('natural_use') or {}
        rows.append({'name': n, 'label': LABELS.get(n, n), 'accuracy': s['accuracy_half'],
                     'lo': s['accuracy_half_ci'][0], 'hi': s['accuracy_half_ci'][1], 'coverage': s['coverage'],
                     'natural_use_half_credit': nat.get('accuracy_half'), 'n': r['metrics']['n']})
    df = pd.DataFrame(rows)
    fig, ax = figure(o['width'], o['height'])
    y = np.arange(len(df))[::-1]
    col = [INK2 if n.startswith('m4_oracle') else SERIES for n in df['name']]
    ax.hlines(y, df['lo'], df['hi'], color=col, lw=1.4)
    ax.scatter(df['accuracy'], y, color=col, s=16, zorder=3)
    for yi, a, hi in zip(y, df['accuracy'], df['hi']):
        ax.text(hi + 0.006, yi, f'{100 * a:.1f}', ha='left', va='center', fontsize=6.5, color=INK2, zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    ref = df.loc[df['name'] == 'm4_oracle_yaw8', 'accuracy']
    if len(ref):
        ax.axvline(float(ref.iloc[0]), color=MUTED, lw=0.8, ls='--')
    ax.axvline(0.5, color=MUTED, lw=0.6)
    ax.set_yticks(y, df['label'])
    ax.set_xlim(*o['xlim'])
    ax.set_xticks(o['xticks'], [f'{100 * t:.0f} %' for t in o['xticks']])
    ax.set_xlabel(o['xlabel'])
    ax.grid(axis='y', visible=False)
    L.save_fig(fig, 'M4_02_accuracy_ladder', data=df,
               caption=('Hand accuracy of the cue-free pipelines on the full recordings (no GO cue, no stored bounds), '
                        'against the same hand model on the stored (oracle) approach window (dashed line; grey: '
                        'oracle-window models). An abstention (no approach accepted, P = 0.5) scores half a correct '
                        'answer. Every pipeline is fitted on the training persons only; leave-one-person-out, '
                        'person-bootstrap 95 % CIs.'),
               decision='The detection loss (oracle minus cue-free) and the choice of detector, training windows, '
                        'activity filter, low-pass and return use (paired comparisons with Holm-adjusted p in the table '
                        'm4_paired).')


def fig_M4_03_stress(**kw: Any) -> None:
    o = _o('M4_03_stress', kw)
    S = L.read_json('m4_stress')
    s = pd.DataFrame(S['summary'])
    s = s[~s['flip'].astype(bool)]
    order = sorted([k for k in s['keep'].unique() if k != 'recorded'], key=float) + ['recorded']
    xpos = {k: i for i, k in enumerate(order)}
    fig, axes = figure(o['width'], o['height'], 1, 2)
    from stage2.lanes.m4_detect import DEFAULT, INSTRUCTED_STILLNESS_S
    style = {k: (v[0].format(cap=DEFAULT.still_cap_s), v[1]) for k, v in o['pipelines'].items() if k in set(s['pipeline'])}
    tidy = []
    for pipe, (lab, colr) in style.items():
        g = s[s['pipeline'] == pipe].set_index('keep').loc[order].reset_index()
        x = [xpos[k] for k in g['keep']]
        axes[0].fill_between(x, g['acc_lo'], g['acc_hi'], color=colr, alpha=0.15, lw=0)
        axes[0].plot(x, g['accuracy'], 'o-', color=colr, ms=3, label=lab)
        axes[1].plot(x, g['coverage'], 'o-', color=colr, ms=3, label=lab)
        axes[1].plot(x, g['share_return'], 's:', color=colr, ms=2.5, lw=0.9)
        tidy.append(g.assign(pipeline=pipe))
    for ax in axes:
        ax.set_xticks(range(len(order)), [('rec.' if k == 'recorded' else f'{float(k):g}') for k in order], fontsize=6.5)
        ax.set_xlabel('stillness kept before the approach (s)')
    axes[0].axhline(0.5, color=MUTED, lw=0.6)
    axes[0].set_ylim(*o['acc_ylim'])
    axes[0].set_ylabel('hand accuracy (abstention = half)')
    axes[0].legend(loc='lower right', fontsize=6.5)
    axes[1].set_ylim(*o['ylim'])
    axes[1].set_ylabel('answered (solid) / return scored (dotted)')
    sv = s.assign(kf=s['keep'].map(lambda k: k if k == 'recorded' else f'{float(k):g}')).set_index(['pipeline', 'kf'])
    q = lambda pipe, k, c: 100 * float(sv.loc[(pipe, k), c]) if (pipe, k) in sv.index else math.nan  # noqa: E731
    inst, none = f'{INSTRUCTED_STILLNESS_S:g}', '0'
    L.save_fig(fig, 'M4_03_stress', data=pd.concat(tidy, ignore_index=True),
               caption=('Stress test: the stillness before each approach, enforced by the collection app, is cut to the '
                        'given number of seconds (rec. = as recorded) and the pipelines run again on the full recording '
                        '(each fitted without the person). Left: hand accuracy, an abstention scoring half a correct '
                        'answer (band: person-bootstrap 95 % CI). Right: share answered (an approach accepted; solid) and '
                        f'share where the scored segment is the return (dotted). "{style.get("uncapped", ("",))[0]}" is '
                        f'the detector whose stillness feature sees up to {DEFAULT.still_cap_s:g} s; the production '
                        f'detector sees at most the instructed {INSTRUCTED_STILLNESS_S:g} s.'),
               decision=(f'Cap the stillness feature at the instructed {inst} s and ask for that stillness on the page: '
                         f'with {inst} s the production pipeline scores {q("production", inst, "accuracy"):.0f} % '
                         f'(context capped at {DEFAULT.still_cap_s:g} s: {q("uncapped", inst, "accuracy"):.0f} %); with '
                         f'no stillness it answers {q("production", none, "coverage"):.0f} % of the recordings and scores '
                         f'the return in {q("production", none, "share_return"):.0f} %, while v0 scores the return in '
                         f'{q("v0", none, "share_return"):.0f} % (paired differences with Holm-adjusted p in the table '
                         'm4_stress).'))


def fig_M4_04_attempts(**kw: Any) -> None:
    o = _o('M4_04_attempts', kw)
    S = L.read_json('m4_sessions')
    cal = L.read_json('m4_calibration')
    ref, prod_agg, p_stop = float(S['reference_keep_s']), cal['production_aggregation'], float(S['p_stop'])
    rule = S['production_rule']
    k_all = pd.DataFrame(S['k_curve'])
    st_all = pd.DataFrame(S['stopping'])
    # the accuracy of k attempts is the sign of their mean logit, the same for every aggregation: show the mean
    k = k_all[k_all['aggregation'] == 'mean']
    st = st_all[(st_all['rule'] == rule) & (st_all['keep_s'] == ref)]
    fig, axes = figure(o['width'], o['height'], 1, 2)
    keeps = sorted(k['keep_s'].unique())
    cols = {kk: o['keep_colors'][i % len(o['keep_colors'])] for i, kk in enumerate(keeps)}
    for kk in keeps:
        g = k[(k['rule'] == rule) & (k['keep_s'] == kk)].sort_values('k')
        axes[0].plot(g['k'], g['accuracy'], 'o-', color=cols[kk], ms=3, lw=1.2 if kk == ref else 0.9,
                     label=f'{kk:g} s still')
        if kk == ref:
            axes[0].fill_between(g['k'], g['acc_lo'], g['acc_hi'], color=cols[kk], alpha=0.15, lw=0)
            for _, r in g.iterrows():
                axes[0].annotate(f"{100 * r['share_k_available']:.0f}%", (r['k'], r['acc_lo']), textcoords='offset points',
                                 xytext=(0, -9), ha='center', fontsize=5.5, color=cols[kk])
    for rname, lab, mk, colr in (('true_only', f'{ref:g} s, true approaches only', 'D--', INK2),
                                 ('pairing' if rule != 'pairing' else 'no_pairing', f'{ref:g} s, other attempt rule', 's:', MUTED)):
        g0 = k[(k['rule'] == rname) & (k['keep_s'] == ref)].sort_values('k')
        if len(g0):
            axes[0].plot(g0['k'], g0['accuracy'], mk, color=colr, ms=2.5, lw=0.9, label=lab)
    axes[0].set_xlabel('accepted attempts aggregated (up to k)')
    axes[0].set_ylabel('session accuracy')
    axes[0].set_xticks(sorted(k['k'].unique()))
    axes[0].set_ylim(*o['ylim'])
    axes[0].set_title(f'first k attempts (% = sessions with k at {ref:g} s)', fontsize=7)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='outside lower center', ncol=3, fontsize=6.3, frameon=False)
    for how, (colr, mk) in o['aggregations'].items():
        g = st[st['aggregation'] == how].sort_values('p_stop')
        if not len(g):
            continue
        axes[1].plot(g['mean_attempts'], g['accuracy'], mk, color=colr, ms=3, lw=1.1,
                     label=_agg_label(how) + (' (production)' if how == prod_agg else ''))
        if how == prod_agg:
            for _, r in g.iterrows():
                axes[1].annotate(f"{r['p_stop']:.2f}", (r['mean_attempts'], r['accuracy']), textcoords='offset points',
                                 xytext=(0, -10), ha='center', fontsize=6, color=colr)
    axes[1].set_xlabel('mean accepted attempts used')
    axes[1].set_ylabel('accuracy of the answer')
    axes[1].set_ylim(*o['stop_ylim'])
    axes[1].set_title(f'stopping rule ({ref:g} s still; labels: p_stop)', fontsize=7)
    axes[1].legend(loc='lower right', fontsize=6)
    n = int(k['n_sessions'].max())
    by_kind = pd.DataFrame(S.get('by_kind', []))
    false_share = float(by_kind.loc[~by_kind['kind'].isin(['approach', 'approach_partial']), 'n'].sum() / by_kind['n'].sum()) \
        if len(by_kind) else math.nan
    kr = k[k['keep_s'] == ref].set_index(['rule', 'k'])['accuracy']
    k_lo, k_hi = int(k['k'].min()), int(k['k'].max())
    acc = lambda r, kk: 100 * float(kr.get((r, kk), math.nan))  # noqa: E731
    gain_true, gain_all = acc('true_only', k_hi) - acc('true_only', k_lo), acc(rule, k_hi) - acc(rule, k_lo)
    lesson = ('rejecting them is what more touches need' if gain_true > gain_all + o['gain_margin_points'] else
              'even the true approaches gain little from more touches')
    L.save_fig(fig, 'M4_04_attempts', data=pd.concat([k_all.assign(panel='k_curve'), st_all.assign(panel='stopping')]),
               caption=(f"Synthetic multi-attempt sessions ({n} per stillness setting; {S['session_recordings']} recordings "
                        'of one person and hand concatenated, the stillness at every junction cut to the given seconds). '
                        'Left: accuracy when the first k accepted attempts are aggregated (the sign of their mean logit; '
                        'sessions with fewer attempts use all they have; band: person-bootstrap 95 % CI at '
                        f'{ref:g} s; % = sessions with at least k attempts); dark dashed: only the attempts that are true '
                        'approaches (the attainable curve); grey dotted: the other attempt rule. Right: the sequential '
                        f'stopping rule at {ref:g} s, one point per threshold p_stop, for three aggregations of the '
                        'attempts.'),
               decision=(f'From {k_lo} to {k_hi} accepted attempts ({ref:g} s junctions) the accuracy goes from '
                         f'{acc(rule, k_lo):.0f} % to {acc(rule, k_hi):.0f} %, against {acc("true_only", k_lo):.0f} % to '
                         f'{acc("true_only", k_hi):.0f} % when only the true approaches are kept: '
                         f'{100 * false_share:.0f} % of the accepted attempts are not approaches (arm settling, returns), '
                         f'and {lesson}. Production: the '
                         f'{rule.replace("_", " ")} attempt rule and the stopping rule with '
                         f'{_agg_label(prod_agg).replace("$s_k$", "s_k")}.'))


def _reliability_panel(ax: Any, rel: list[dict[str, float]], title: str, metrics: dict[str, str] | None,
                       color: str = SERIES) -> None:
    r = pd.DataFrame(rel)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls='--')
    if len(r):
        ax.fill_between(r['p_mean'], r['y_lo'], r['y_hi'], color=color, alpha=0.18, lw=0, step=None)
        ax.plot(r['p_mean'], r['y_mean'], 'o-', color=color, ms=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=7)
    ax.set_xlabel('predicted P(right)')
    if metrics:
        ax.text(0.03, 0.97, '\n'.join(f'{k} {v}' for k, v in metrics.items()), transform=ax.transAxes,
                va='top', ha='left', fontsize=6, color=INK2)


def _cal_text(e: dict[str, Any]) -> dict[str, str]:
    return {'ECE': f"{e['ece']:.3f} (null {e['ece_null_lo']:.3f}-{e['ece_null_hi']:.3f})",
            'slope': f"{e['slope']:.2f} [{e['slope_ci'][0]:.2f}, {e['slope_ci'][1]:.2f}]",
            'log loss': f"{e['log_loss']:.3f}"}


def fig_M4_05_calibration(**kw: Any) -> None:
    o = _o('M4_05_calibration', kw)
    cal = L.read_json('m4_calibration')
    cands = {c['calibration']: c for c in cal['candidates']}
    show = [c for c in ('none', 'nested') if c in cands]
    fig, axes = figure(o['width'], o['height'], 1, 3)
    tidy = []
    for ax, c in zip(axes[:2], show):
        e = cands[c]
        title = 'uncalibrated' if c == 'none' else 'production: calibrator nested'
        _reliability_panel(ax, e['reliability'], title, _cal_text(e))
        tidy += [{**r, 'calibration': c, 'panel': 'reliability'} for r in e['reliability']]
    axes[0].set_ylabel('observed share of right hands')
    ax = axes[2]
    names = [c for c in cands if c != 'none' and 'vs_none' in cands[c]]
    y = np.arange(len(names))[::-1]
    for yi, c in zip(y, names):
        vs = cands[c]['vs_none']
        colr = SERIES if cands[c].get('antisymmetric') else MUTED
        ax.hlines(yi, *vs['d_ll_ci'], color=colr, lw=1.4)
        ax.plot(vs['d_ll'], yi, 'o', color=colr, ms=3.5)
        tidy.append({'calibration': c, 'panel': 'paired', 'd_ll': vs['d_ll'], 'lo': vs['d_ll_ci'][0],
                     'hi': vs['d_ll_ci'][1], 'p_holm': vs['p_ll_holm']})
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(y, [c.replace('_', ' ') for c in names], fontsize=6.5)
    ax.set_xlabel(o['xlabel_diff'], fontsize=6.5)
    ax.set_title('paired with uncalibrated\n(blue: antisymmetric)', fontsize=7)
    ax.grid(axis='y', visible=False)
    nest = cands.get('nested', {})
    ch = nest.get('chosen_per_fold', {})
    ch_txt = ', '.join(f'{k.replace("_", " ")} in {v}' for k, v in sorted(ch.items(), key=lambda kv: -kv[1]))
    L.save_fig(fig, 'M4_05_calibration', data=pd.DataFrame(tidy),
               caption=('Calibration of the cue-free production detector for one touch (10 equal-mass bins; band: '
                        'person-bootstrap 95 % CI of the observed share; text: ECE with the 95 % range a perfectly '
                        'calibrated P of the same values would show, and the calibration slope with its CI). Every '
                        'calibrator is fitted on cross-fitted predictions of the training persons and applied to the '
                        'held-out person; abstentions (P = 0.5) included. Right: paired log-loss difference to the '
                        'uncalibrated pipeline (person bootstrap).'),
               decision=(f"Production calibrator chosen inside each fold among the antisymmetric ones (folds: {ch_txt}); "
                         f"uncalibrated {'consistent' if cands.get('none', {}).get('calibrated') else 'not consistent'} "
                         f"with calibration, production {'consistent' if nest.get('calibrated') else 'not consistent'}."))


def fig_M4_06_k3_calibration(**kw: Any) -> None:
    o = _o('M4_06_k3_calibration', kw)
    cal = L.read_json('m4_calibration')
    k3, stop = cal.get('k3') or {}, cal.get('stop') or {}
    prod, ref, p_stop = cal['production_aggregation'], cal['sessions_keep_s'], cal['p_stop']
    top = [h for h in o['top'] if h in k3]
    bottom = [h for h in o['bottom'] if h in stop]
    ncol = max(len(top), len(bottom) + 1)
    fig, axes = figure(o['width'], o['height'], 2, ncol)
    tidy = []
    for j, how in enumerate(top):
        e = k3[how]
        _reliability_panel(axes[0, j], e['reliability'], _agg_label(how) + (' *' if how == prod else ''),
                           {'slope': f"{e['slope']:.2f}", 'log loss': f"{e['log_loss']:.3f}"},
                           color=o['colors'].get(how, SERIES))
        tidy += [{**r, 'panel': 'k3', 'aggregation': how} for r in e['reliability']]
    for j, how in enumerate(bottom):
        e = stop[how]
        _reliability_panel(axes[1, j], e['reliability'], _agg_label(how) + (' *' if how == prod else ''),
                           {'slope': f"{e['slope']:.2f}", 'log loss': f"{e['log_loss']:.3f}"},
                           color=o['colors'].get(how, SERIES))
        tidy += [{**r, 'panel': 'stop', 'aggregation': how} for r in e['reliability']]
    for j in range(len(top), ncol):
        axes[0, j].set_visible(False)
    axes[0, 0].set_ylabel('k = 3 attempts\nobserved share right')
    axes[1, 0].set_ylabel('answer at the stop\nobserved share right')
    ax = axes[1, len(bottom)]                       # ECE of the answer against its perfect-calibration null band
    names = list(bottom)[::-1]
    xmax = max(max(stop[h]['ece_null_hi'], stop[h]['ece']) for h in names) * 1.25
    for i, how in enumerate(names):
        e = stop[how]
        c = o['colors'].get(how, SERIES)
        ax.hlines(i, e['ece_null_lo'], e['ece_null_hi'], color=LIGHT, lw=4)
        ax.plot(e['ece'], i, 'o', color=c, ms=4)
        ax.text(0.02 * xmax, i + 0.22, _agg_label(how).replace('fitted on ', ''), fontsize=6, color=c, va='bottom',
                ha='left')
        tidy.append({'panel': 'ece', 'aggregation': how, 'ece': e['ece'], 'null_lo': e['ece_null_lo'],
                     'null_hi': e['ece_null_hi']})
    ax.set_yticks([])
    ax.set_ylim(-0.5, len(names) - 0.2)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('ECE (grey: null band)', fontsize=6.5)
    ax.set_title('answer at the stop', fontsize=7)
    ax.grid(axis='y', visible=False)
    ok = [h for h in bottom if stop[h]['calibrated']]
    dec = (f"Production aggregation by the declared rule (calibrated, then fewest fitted parameters): "
           f"{_agg_label(prod).replace('$s_k$', 's_k')}; consistent with calibration at the stop: "
           f"{', '.join(_agg_label(h).replace('$s_k$', 's_k') for h in ok) or 'none'}.")
    L.save_fig(fig, 'M4_06_k3_calibration', data=pd.DataFrame(tidy),
               caption=(f'Calibration of multi-attempt aggregates on the synthetic sessions ({ref:g} s of stillness at '
                        'each junction, production attempt rule; equal-mass bins, person-bootstrap 95 % bands). Top: '
                        'P(right) of the first three accepted attempts under four aggregations of their calibrated '
                        'logits: the mean, the sum (as if independent), the scale s_k fitted on random recordings of '
                        'the training persons, and s_k = s_1 k^b fitted on cross-fitted synthetic sessions of the '
                        f'training persons. Bottom: the answer of the stopping rule (p_stop {p_stop:.2f}, what the user '
                        'sees) under three of them, and its ECE against the 95 % range of a perfectly calibrated answer '
                        '(persons resampled, outcomes redrawn). * production.'),
               decision=dec)


def fig_M4_07_coverage(**kw: Any) -> None:
    from stage2.lanes import m4_stats as ST
    o = _o('M4_07_coverage', kw)
    prod = _production()
    rows = []
    d = pd.read_csv(L.tab_dir / f'{prod}_details.csv')
    d = d[d['set'] == 'primary']
    y = d['y'].to_numpy(int)
    pf = d['p_forced'].to_numpy(float)
    pa = d['p_approach'].fillna(-1).to_numpy(float)
    conf = np.maximum(pf, 1 - pf)
    det_ok = (d['status'] == 'ok').to_numpy()
    for tau in o['taus']:                                   # confidence threshold, detector threshold as fitted
        keep = det_ok & (conf >= tau)
        if keep.sum() >= o['min_n']:
            rows.append({'curve': 'confidence threshold', 'threshold': float(tau), 'coverage': float(keep.mean()),
                         'accuracy': ST.half_credit(y[keep], pf[keep])})
    for t in o['det_thresholds']:                           # P(approach) threshold, always the best candidate
        keep = pa >= t
        if keep.sum() >= o['min_n']:
            rows.append({'curve': 'P(approach) threshold', 'threshold': float(t), 'coverage': float(keep.mean()),
                         'accuracy': ST.half_credit(y[keep], pf[keep])})
    r = _result('m4_oracle_yaw8')
    if r is not None:
        pp = pd.DataFrame(r['predictions'])
        po, yo = pp['p_right'].to_numpy(float), pp['y'].to_numpy(int)
        co = np.maximum(po, 1 - po)
        for tau in o['taus']:
            keep = co >= tau
            if keep.sum() >= o['min_n']:
                rows.append({'curve': 'oracle window, confidence threshold', 'threshold': float(tau),
                             'coverage': float(keep.mean()), 'accuracy': ST.half_credit(yo[keep], po[keep])})
    df = pd.DataFrame(rows)
    fig, ax = figure(o['width'], o['height'])
    styles = {'confidence threshold': (SERIES, 'o-'), 'P(approach) threshold': (ORANGE, 's-'),
              'oracle window, confidence threshold': (MUTED, 'o--')}
    for curve, (colr, mk) in styles.items():
        g = df[df['curve'] == curve]
        ax.plot(g['coverage'], g['accuracy'], mk, ms=2.5, lw=1.0, color=colr, label=curve)
    ax.set_xlabel('coverage (share answered)')
    ax.set_ylabel('accuracy on answered')
    ax.set_xlim(0, 1)
    ax.set_ylim(*o['ylim'])
    ax.legend(loc='lower left', fontsize=6)
    L.save_fig(fig, 'M4_07_coverage', data=df,
               caption=('Abstain rule for one touch (cue-free production pipeline, full recordings): answer only when '
                        'max(P, 1 - P) >= tau (blue; the detector also needs P(approach) >= its threshold) or only '
                        'when the best candidate has P(approach) >= t (orange; always the best candidate); grey: the '
                        'same confidence rule on the stored approach windows. Accuracy on the answered recordings '
                        'against the share answered.'),
               decision='The "please repeat" rule for a single touch: which threshold buys accuracy at what coverage.')


def fig_M4_08_timeline(**kw: Any) -> None:
    o = _o('M4_08_timeline', kw)
    tl = L.read_json('m4_timeline')
    if not tl.get('t_s'):
        return
    cal = L.read_json('m4_calibration')
    how = cal['production_aggregation']
    t = np.asarray(tl['t_s'])
    fig, axes = figure(o['width'], o['height'], 2, 1, sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    ax = axes[0]
    ax.plot(t, tl['activity'], color=INK2, lw=0.7)
    ax.axhline(tl['lo_dps'], color=MUTED, lw=0.5, ls=':')
    ax.axhline(tl['hi_dps'], color=MUTED, lw=0.5, ls='--')
    ymax = max(tl['activity']) * 1.15 if tl['activity'] else 1
    for a in tl['true_approaches']:
        ax.axvspan(a['onset_ms'] / 1000, a['end_ms'] / 1000, color=HAND_COLORS[tl['hand']], alpha=0.12, lw=0)
    cls_col = {'approach': SERIES, 'return': ORANGE, 'other': MUTED}
    for c in tl['candidates']:
        best = max(('approach', 'return', 'other'), key=lambda k: c[f'p_{k}'])
        ax.plot([c['onset_ms'] / 1000, c['end_ms'] / 1000], [ymax * 0.93] * 2, color=cls_col[best], lw=3,
                solid_capstyle='butt')
        ax.text((c['onset_ms'] + c['end_ms']) / 2000, ymax * 0.97, f"{100 * c['p_approach']:.0f}", ha='center',
                va='bottom', fontsize=5.5, color=cls_col[best])
    ax.set_ylim(0, ymax * 1.08)
    ax.set_ylabel('|w| causal (deg/s)')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([0], [0], color=SERIES, lw=3, label='candidate: approach'),
                       Line2D([0], [0], color=ORANGE, lw=3, label='candidate: return'),
                       Line2D([0], [0], color=MUTED, lw=3, label='candidate: other'),
                       Patch(color=HAND_COLORS[tl['hand']], alpha=0.25, label='true approach (app events)')],
              loc='upper left', bbox_to_anchor=(0.0, -0.02), ncol=4, fontsize=6, frameon=False)
    ax2 = axes[1]
    run = pd.DataFrame(tl['running'][how])
    att = pd.DataFrame(tl['attempts'])
    ax2.axhline(0.5, color=MUTED, lw=0.6)
    ax2.axhspan(tl['p_stop'], 1.0, color=HAND_COLORS['right'], alpha=0.08, lw=0)
    ax2.axhspan(0.0, 1 - tl['p_stop'], color=HAND_COLORS['left'], alpha=0.08, lw=0)
    if len(run):
        ax2.step(np.r_[t[0], run['decided_ms'] / 1000, t[-1]], np.r_[0.5, run['p_right'], run['p_right'].iloc[-1]],
                 where='post', color=SERIES, lw=1.2)
        kc = [o['kind_colors'].get(k, MUTED) for k in att['kind']] if len(att) == len(run) else SERIES
        ax2.scatter(run['decided_ms'] / 1000, run['p_right'], color=kc, s=14, zorder=3, edgecolor='white', lw=0.4)
    dec = tl['decisions'][how]
    end_s = tl['stream_end_ms'] / 1000
    if dec.get('stopped') == 'confident' and dec.get('decided_ms') is not None:
        x_dec, txt = dec['decided_ms'] / 1000, f" stop: {100 * dec['p_right']:.0f} % right after {dec['n_used']} attempt(s)"
    elif dec.get('stopped') in ('k_max', 'stream_end'):
        x_dec = dec['decided_ms'] / 1000 if dec['stopped'] == 'k_max' else end_s
        txt = (f" no confident stop: {100 * dec['p_right']:.0f} % right after {dec['n_used']} attempt(s)"
               f"{' (k_max)' if dec['stopped'] == 'k_max' else ' (stream ended)'}")
    else:
        x_dec, txt = end_s, ' no attempt: please repeat'
    ax2.axvline(x_dec, color=INK2, lw=0.8, ls='--')
    ax2.text(x_dec, 0.04, txt, fontsize=6.5, color=INK2, va='bottom', ha='right' if x_dec > t[-1] * 0.6 else 'left')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('P(right) so far')
    ax2.set_xlabel('time in the session (s)')
    correct = (dec.get('p_right', 0.5) >= 0.5) == (tl['hand'] == 'right')
    kinds = att['kind'].value_counts().to_dict() if len(att) else {}
    n_false = sum(v for k, v in kinds.items() if k not in ('approach', 'approach_partial'))
    L.save_fig(fig, 'M4_08_timeline', data=pd.DataFrame(tl['candidates']),
               caption=(f"Online decision on one synthetic session of the {tl['hand']} hand ({tl['keep_s']:g} s of "
                        'stillness at each junction), replayed sample by sample. Top: causal activity with the '
                        'hysteresis thresholds; bars: candidate segments coloured by their most likely class, with '
                        'P(approach) in %; shading: the true approaches. Bottom: the aggregated P(right) after each '
                        f"accepted attempt ({_agg_label(how).replace('$s_k$', 's_k')}; dots coloured by what the attempt "
                        'really was: blue a true approach, orange a return, grey another movement); the pipeline stops '
                        'when it leaves the band between 1 - p_stop and p_stop, else answers with the P reached after '
                        f"{tl['k_max']} attempts or when the stream ends. The session was not selected for success: it is "
                        f"the first one in person order with at least three attempts; {len(att)} attempts were accepted, "
                        f"{n_false} of them not an approach; its final answer is {'right' if correct else 'wrong'}."),
               decision='How the online pipeline trims the stream, which bouts it accepts as attempts and when it stops.')


def fig_M4_09_latency(**kw: Any) -> None:
    o = _o('M4_09_latency', kw)
    lat = L.read_json('m4_latency')
    pt = pd.DataFrame(lat['per_trial'])
    fig, ax = figure(o['width'], o['height'])
    bins = np.linspace(o['xlim'][0], o['xlim'][1], o['bins'])
    ax.hist(pt['end'].dropna().clip(*o['xlim']), bins=bins, color=SERIES, alpha=0.75, label='after the stored end')
    ax.hist(pt['touch'].dropna().clip(*o['xlim']), bins=bins, color=GOLD, alpha=0.6, label='after the first touch')
    ax.set_xlabel('decision time (ms)')
    ax.set_ylabel('recordings')
    ax.set_xlim(*o['xlim'])
    ax.legend(loc='upper right')
    e, tch = lat['decision_minus_oracle_end_ms'], lat['decision_minus_first_touch_ms']
    L.save_fig(fig, 'M4_09_latency', data=pt,
               caption=(f"When the online pipeline decides an approach is over: median {e['median']:.0f} ms after the "
                        f"stored end of the approach (90th percentile {e['p90']:.0f} ms) and {tch['median']:.0f} ms after "
                        f"the first screen touch (90th percentile {tch['p90']:.0f} ms); recordings whose approach was "
                        'found; values beyond the axis are drawn at its edge.'),
               decision=(f"The end-of-approach criterion (stillness at the face plus the look-ahead) decides a median "
                         f"{tch['median']:.0f} ms after the touch."))


def fig_M4_10_return(**kw: Any) -> None:
    o = _o('M4_10_return', kw)
    sc = _scores()
    rows = []
    for n, lab in o['rows']:
        if n not in sc:
            continue
        s = sc[n]['primary']
        rows.append({'name': n, 'label': lab, 'accuracy': s['accuracy_half'], 'lo': s['accuracy_half_ci'][0],
                     'hi': s['accuracy_half_ci'][1], 'n': s['n']})
    df = pd.DataFrame(rows)
    fig, ax = figure(o['width'], o['height'])
    y = np.arange(len(df))[::-1]
    ax.hlines(y, df['lo'], df['hi'], color=SERIES, lw=1.4)
    ax.scatter(df['accuracy'], y, color=SERIES, s=14, zorder=3)
    ax.set_yticks(y, [f"{a} (n={int(n)})" for a, n in zip(df['label'], df['n'])])
    ax.set_xlim(*o['xlim'])
    ax.axvline(0.5, color=MUTED, lw=0.6)
    ax.set_xlabel('accuracy (95 % CI)')
    ax.grid(axis='y', visible=False)
    L.save_fig(fig, 'M4_10_return', data=df,
               caption=('The return as evidence. Top three: stored windows of the trials with a clean return: the '
                        'approach model, a separate return head (same model fitted on returns) and the sum of their '
                        'logits. Bottom two: the cue-free production detector (uncalibrated) without and with the return '
                        'head added when the segment after the chosen approach is classified as a return (all '
                        'recordings).'),
               decision='Whether the production pipeline should also score the return (paired, table m4_paired).')


def fig_M4_11_segment_features(**kw: Any) -> None:
    o = _o('M4_11_segment_features', kw)
    v = pd.read_csv(L.tab_dir / 'm4_segfeat_values.csv')
    sf = L.read_json('m4_segfeat')
    auc = {r['feature']: r for r in sf['auc']}
    feats = [f for f in o['features'] if f[0] in v.columns]
    fig, axes = figure(o['width'], o['height'], 1, len(feats))
    for ax, (f, lab, lim) in zip(np.atleast_1d(axes), feats):
        bins = np.linspace(lim[0], lim[1], o['bins'])
        for c in ('other', 'return', 'approach'):
            x = v.loc[v['label'] == c, f].dropna().clip(*lim)
            ax.hist(x, bins=bins, density=True, histtype='step', lw=1.1, color=o['colors'][c], label=c)
        a = auc.get(f, {})
        ax.set_title(f"AUC vs return {a.get('auc_vs_return', float('nan')):.2f}\nvs other "
                     f"{a.get('auc_vs_other', float('nan')):.2f}", fontsize=6.5)
        ax.set_xlabel(lab, fontsize=6.5)
        ax.set_yticks([])
    h, lb = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h[::-1], [f'segment label: {x}' for x in lb[::-1]], loc='outside lower center', ncol=3, fontsize=6.5,
               frameon=False)
    med = v.groupby('label')[[f for f, _, _ in feats]].median()
    medians = '; '.join(f"{lab.replace(chr(10), ' ')}: approach "
                        f"{med.loc['approach', f]:.2f}, return {med.loc['return', f]:.2f}, other {med.loc['other', f]:.2f}"
                        for f, lab, _ in feats if {'approach', 'return', 'other'} <= set(med.index))
    L.save_fig(fig, 'M4_11_segment_features', data=v,
               caption=(f"Hand-agnostic features of the {sf['n_segments']} candidate segments of the recordings, by "
                        "their label from the app's events (densities; AUC: approach against return / other). Medians "
                        f"({medians}). The screen-normal features are in the device frame, so they change sign when the "
                        'phone is held with the screen away from the face (figure M4_12).'),
               decision='Features that separate the approach from the return without any hand information: the basis '
                        'of the segment classifier.')


def fig_M4_12_screen_flip(**kw: Any) -> None:
    from stage2.lanes.m4_config import GRIP_GATE_UPZ
    o = _o('M4_12_screen_flip', kw)
    S = L.read_json('m4_stress')
    s = pd.DataFrame(S['summary'])
    s = s[s['keep'] == 'recorded']
    s = s.assign(flip=s['flip'].astype(bool))
    pipes = [p for p in o['pipelines'] if p in set(s['pipeline'])]
    fig, ax = figure(o['width'], o['height'])
    nb = len(o['bars'])
    w = 0.8 / nb
    tidy = []
    for j, (lab, flip, check) in enumerate(o['bars']):
        g = s[s['flip'] == flip].set_index('pipeline').reindex(pipes)
        col, lo, hi = ('acc_gated', 'acc_gated_lo', 'acc_gated_hi') if check else ('accuracy', 'acc_lo', 'acc_hi')
        x = np.arange(len(pipes)) + (j - (nb - 1) / 2) * w
        colr = o['colors'][flip]
        ax.bar(x, g[col], width=w, color='white' if check else colr, edgecolor=colr, hatch='////' if check else None,
               lw=0.8, label=lab, alpha=0.9)
        ax.errorbar(x, g[col], yerr=[g[col] - g[lo], g[hi] - g[col]], fmt='none', ecolor=INK2, lw=0.8, capsize=1.5)
        tidy.append(g.reset_index()[['pipeline', col, lo, hi, 'share_return', 'gate_share']].rename(
            columns={col: 'accuracy', lo: 'lo', hi: 'hi'}).assign(bar=lab, screen_away=flip, grip_check=check))
    ax.axhline(0.5, color=MUTED, lw=0.6)
    ax.set_xticks(np.arange(len(pipes)), [o['pipelines'][p] for p in pipes], fontsize=6.5)
    ax.set_ylim(*o['ylim'])
    ax.set_ylabel('hand accuracy (abstention = half)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6, ncol=nb, frameon=False)
    rec = s.set_index(['pipeline', 'flip'])
    f = lambda pipe, flip, c: 100 * float(rec.loc[(pipe, flip), c]) if (pipe, flip) in rec.index else math.nan  # noqa: E731
    pa = (L.read_json('m4_paired').get('pairs') or {}).get('grip_vs_robust', {})
    g_cost = (f" (paired with the production detector on the recordings: {100 * pa['d_acc']:+.1f} points, "
              f"[{100 * pa['d_acc_ci'][0]:+.1f}, {100 * pa['d_acc_ci'][1]:+.1f}])") if pa else ''
    lower = f('grip_uncal', False, 'accuracy') < f('production', False, 'accuracy')
    dec = (f"Screen away, the production pipeline falls from {f('production', False, 'accuracy'):.0f} % to "
           f"{f('production', True, 'accuracy'):.0f} % (it scores the return in {f('production', True, 'share_return'):.0f} %). "
           f"The rest-pose grip check stops {f('production', True, 'gate_share'):.0f} % of the screen-away recordings, "
           f"so the share answered wrongly falls from {f('production', True, 'wrong_share'):.0f} % to "
           f"{f('production', True, 'wrong_share_gated'):.0f} % (they are asked to turn the phone instead), and "
           f"{f('production', False, 'gate_share'):.0f} % of the screen-toward ones (accuracy "
           f"{f('production', False, 'acc_gated'):.0f} % against {f('production', False, 'accuracy'):.0f} %). "
           f"The first remedy, a check of the chosen segment's own start pose, stopped "
           f"{f('production', True, 'gate_seg_share'):.0f} % of the screen-away recordings and left their accuracy at "
           f"{f('production', True, 'acc_gated_seg'):.0f} %. The grip-normalised alternative is invariant "
           f"({f('grip_uncal', False, 'accuracy'):.0f} % and {f('grip_uncal', True, 'accuracy'):.0f} %) but "
           f"{'less' if lower else 'not less'} accurate on the recordings{g_cost}.")
    L.save_fig(fig, 'M4_12_screen_flip', data=pd.concat(tidy, ignore_index=True),
               caption=('Every held-out recording scored as recorded (screen towards the face) and as its screen-away '
                        'twin (the phone turned 180 deg about its long axis: the device x and z axes change sign), by '
                        'the pipeline of its fold; bars: accuracy (abstention = half credit) with person-bootstrap 95 % '
                        "CI. Hatched: with the inference page's rest-pose grip check (model-free: the screen-normal "
                        'component of the up vector while the phone rests before the first movement must be at least '
                        f'{GRIP_GATE_UPZ:g}; otherwise the page asks to turn the screen towards the face and does not '
                        'answer, scored as an abstention).'),
               decision=dec)


FIGURE_FUNCTIONS = {
    'M4_01_detection_outcomes': fig_M4_01_detection_outcomes,
    'M4_02_accuracy_ladder': fig_M4_02_accuracy_ladder,
    'M4_03_stress': fig_M4_03_stress,
    'M4_04_attempts': fig_M4_04_attempts,
    'M4_05_calibration': fig_M4_05_calibration,
    'M4_06_k3_calibration': fig_M4_06_k3_calibration,
    'M4_07_coverage': fig_M4_07_coverage,
    'M4_08_timeline': fig_M4_08_timeline,
    'M4_09_latency': fig_M4_09_latency,
    'M4_10_return': fig_M4_10_return,
    'M4_11_segment_features': fig_M4_11_segment_features,
    'M4_12_screen_flip': fig_M4_12_screen_flip,
}


def draw(only: list[str] | None = None) -> None:
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
