"""Figures of lane final, from saved files only (tables/final/final_*.json). One function per figure; presentation
defaults in FIGURES. Edit, then: python -m stage2.pipeline figures --lanes final [--only FINAL_NN_name]"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from stage2.pipeline import Lane, selected
from stage2.plotting import (HAND_COLORS, INK, INK2, MUTED, PLATFORM_COLORS, PLATFORM_LABELS, RAMP, SERIES,
                             figure)

L = Lane('final')
LANE_COLORS = {'reference': '#52514e', 'm1': '#2a78d6', 'm2': '#eb6834', 'm3': '#1baf7a', 'p1': '#9b59b6',
               'final': '#0b0b0b', 'm4': '#c49a00'}
FIGURES: dict[str, dict[str, Any]] = {
    'FINAL_01_master_forest': dict(width='full', height=3.9),
    'FINAL_02_overview': dict(width='full', height=3.0),
    'FINAL_03_nested_selection': dict(width='full', height=2.6),
    'FINAL_04_paired': dict(width='full', height=3.6),
    'FINAL_05_input_vs_class': dict(width='full', height=2.2),
    'FINAL_06_robustness': dict(width='full', height=4.4),
    'FINAL_07_natural_transfer': dict(width='full', height=3.0),
    'FINAL_08_calibration': dict(width='full', height=2.5),
    'FINAL_09_remedies': dict(width='half', height=2.0),
    'FINAL_10_attempts': dict(width='full', height=2.4),
    'FINAL_11_persons': dict(width='full', height=2.6),
    'FINAL_12_learning_curve': dict(width='full', height=2.4),
    'FINAL_13_segmentation': dict(width='full', height=1.9),
    'FINAL_14_prior': dict(width='half', height=2.2),
}


def _o(name: str, kw: dict[str, Any]) -> dict[str, Any]:
    return {**FIGURES[name], **kw}


def _pct(v: float, nd: int = 1) -> str:
    return f'{100 * v:.{nd}f} %'


def _read(name: str) -> dict[str, Any]:
    from stage2.lanes.final_io import denull
    return denull(L.read_json(name))


def _final() -> tuple[str, dict[str, Any]]:
    dec = _read('final_decision')
    return dec['final_model'], dec


def _label_map() -> dict[str, str]:
    return {r['key']: r['label'] for r in _read('final_master')['rows']}


# ───────────────────────────── 01 master forest ─────────────────────────────


def fig_FINAL_01_master_forest(**kw: Any) -> None:
    name = 'FINAL_01_master_forest'
    o = _o(name, kw)
    m = pd.DataFrame(_read('final_master')['rows']).sort_values('log_loss', ascending=False).reset_index(drop=True)
    if 'prod_log_loss' not in m.columns:
        m['prod_log_loss'] = np.nan
    fin, _ = _final()
    fig, (a1, a2) = figure(o['width'], o['height'], 1, 2, sharey=True)
    y = np.arange(len(m))
    for ax, col, ci, xl in ((a1, 'accuracy', 'accuracy_ci', 'LOSO accuracy'),
                            (a2, 'log_loss', 'log_loss_ci', 'log loss (diamond: after the in-fold remedy)')):
        for i, r in m.iterrows():
            c = LANE_COLORS.get(r['lane'], SERIES)
            lo, hi = r[ci]
            ax.plot([lo, hi], [i, i], color=c, lw=1.0)
            ax.plot(r[col], i, 'o', ms=4.5 if r['key'] == fin else 3.2, color=c,
                    mfc=c if not r['n_configs_tried'] > 1 else 'white', mew=1.0)
            if col == 'log_loss' and np.isfinite(r['prod_log_loss']) and abs(r['prod_log_loss'] - r['log_loss']) > 1e-3:
                ax.plot(r['prod_log_loss'], i, 'D', ms=2.6, color=c, mfc='none', mew=0.8)
        ref = m.loc[m['key'] == 'ref', col]
        if len(ref):
            ax.axvline(float(ref.iloc[0]), color=MUTED, lw=0.6, ls='--')
        ax.set_xlabel(xl)
    if True:
        a1.xaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    labels = [(r['label'] + ('  [final]' if r['key'] == fin else '')) for _, r in m.iterrows()]
    a1.set_yticks(y, labels)
    for t, (_, r) in zip(a1.get_yticklabels(), m.iterrows()):
        t.set_color(LANE_COLORS.get(r['lane'], INK))
        if r['key'] == fin:
            t.set_fontweight('bold')
    from matplotlib.lines import Line2D
    lanes = list(dict.fromkeys(m['lane'][::-1]))
    hs = [Line2D([], [], color=LANE_COLORS.get(ln, SERIES), marker='o', ls='-', lw=1.0, ms=3.2,
                 label='this lane (new)' if ln == 'final' else ln) for ln in lanes]
    hs.append(Line2D([], [], color=INK2, marker='o', ls='', ms=3.2, mfc='white', label='open: exploratory'))
    fig.legend(handles=hs, loc='outside lower center', ncol=len(hs), fontsize=6, handletextpad=0.3,
               columnspacing=0.9)
    n_ex = int((m['n_configs_tried'] > 1).sum())
    data = m[['key', 'label', 'lane', 'accuracy', 'log_loss', 'prod_log_loss', 'n_configs_tried']].assign(
        acc_lo=m['accuracy_ci'].str[0], acc_hi=m['accuracy_ci'].str[1], ll_lo=m['log_loss_ci'].str[0],
        ll_hi=m['log_loss_ci'].str[1])
    best = m.loc[m['log_loss'].idxmin()]
    L.save_fig(fig, name, data=data,
               caption=(f'The {len(m)} candidates of the cross-lane selection as their lanes deliver them (leave-one-'
                        f'person-out, 95 % person-bootstrap CIs), coloured by lane; open markers: chosen on the outer '
                        f'folds in their lane (exploratory, {n_ex} of {len(m)}); diamonds: log loss after the '
                        f'calibration remedy chosen inside each fold (shown where it moves by more than 0.001); dashed: '
                        f'the reference floor. Lowest log loss as delivered: {best["label"]} ({best["log_loss"]:.3f}).'),
               decision='Which candidates are worth a nested selection and a robustness check; the forest is '
                        'exploratory, the nested estimate (FINAL_03) is the honest number.')


# ───────────────────────────── 02 overview of every experiment ─────────────────────────────


def fig_FINAL_02_overview(**kw: Any) -> None:
    name = 'FINAL_02_overview'
    o = _o(name, kw)
    ov = pd.DataFrame(_read('final_overview')['rows']).dropna(subset=['accuracy', 'log_loss'])
    ov = ov[(ov['log_loss'] < 1.2)]
    fig, ax = figure(o['width'], o['height'])
    for lane, g in ov.groupby('lane'):
        c = LANE_COLORS.get(lane, SERIES)
        ex = g['exploratory'].astype(bool)
        ax.scatter(g.loc[~ex, 'log_loss'], g.loc[~ex, 'accuracy'], s=12, color=c, label=f'{lane} ({len(g)})',
                   marker='s' if lane == 'm4' else 'o', lw=0)
        ax.scatter(g.loc[ex, 'log_loss'], g.loc[ex, 'accuracy'], s=12, facecolor='white', edgecolor=c, lw=0.8,
                   marker='s' if lane == 'm4' else 'o')
    ax.set_xlabel('log loss (lower is better)')
    ax.set_ylabel('LOSO accuracy')
    ax.yaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    ax.legend(ncol=4, loc='lower left', fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    n_ex = int(ov['exploratory'].sum())
    L.save_fig(fig, name, data=ov[['lane', 'name', 'accuracy', 'log_loss', 'n_configs_tried', 'exploratory', 'unit']],
               caption=(f'Every experiment of the model lanes\' registries, the reference and this lane\'s own models '
                        f'({len(ov)} with a log loss below 1.2; this lane\'s stress tests are left out), accuracy '
                        f'against log loss; open markers: compared on the outer folds before being kept '
                        f'(n_configs_tried > 1, exploratory; {n_ex}); squares: lane m4, whose unit is a cue-free '
                        f'recording and whose abstentions score one half. Points near 50 % and a log loss of 0.69 are '
                        f'ablations that cannot see the hand by design (for example magnitude-only inputs).'),
               decision='Context for the selection: the finalists sit on the accuracy / log-loss frontier of their lanes.')


# ───────────────────────────── 03 nested selection ─────────────────────────────


def fig_FINAL_03_nested_selection(**kw: Any) -> None:
    name = 'FINAL_03_nested_selection'
    o = _o(name, kw)
    ns = _read('final_nested')
    lab = _label_map()
    freq = pd.Series(ns['frequency'])
    freq = freq[freq > 0].sort_values()
    fig, (a1, a2) = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': [1.1, 1]})
    a1.barh(range(len(freq)), freq.values, color=SERIES)
    a1.set_yticks(range(len(freq)), [lab.get(k, k) for k in freq.index])
    a1.set_xlabel('outer folds in which it was chosen')
    nfold = int(sum(ns['frequency'].values()))
    a1.set_xlim(0, nfold)
    rows = [('nested selection', ns['nested']['accuracy'], ns['nested']['accuracy_ci']),
            ('best on the outer folds', ns['naive']['accuracy'], ns['naive']['accuracy_ci'])]
    for i, (t, v, ci) in enumerate(rows):
        a2.plot(ci, [i, i], color=INK2, lw=1.0)
        a2.plot(v, i, 'o', color=SERIES if i == 0 else MUTED, ms=4)
    a2.set_yticks([0, 1], [r[0] for r in rows])
    a2.set_ylim(-0.6, 1.6)
    a2.set_xlabel('LOSO accuracy')
    a2.xaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    opt = ns['optimism']
    data = pd.DataFrame([{'key': k, 'label': lab.get(k, k), 'folds_chosen': int(v)} for k, v in ns['frequency'].items()])
    L.save_fig(fig, name, data=data,
               caption=(f'Nested cross-lane selection among {ns["n_candidates"]} candidates: (left) how often each was '
                        f'chosen by the lowest inner log loss on the training persons, over {nfold} outer folds; '
                        f'(right) the selection-aware estimate against the best candidate picked on the outer folds '
                        f'({lab.get(ns["naive_key"], ns["naive_key"])}; 95 % person-bootstrap CIs). '
                        f'Optimism of the naive pick: accuracy {100 * opt["accuracy"]["diff"]:+.1f} points '
                        f'[{100 * opt["accuracy"]["ci"][0]:+.1f}, {100 * opt["accuracy"]["ci"][1]:+.1f}], log loss '
                        f'{opt["log_loss"]["diff"]:+.3f}.'),
               decision='The honest headline of the selection procedure is the nested number, not the best of the forest.')


# ───────────────────────────── 04 paired comparisons ─────────────────────────────


def fig_FINAL_04_paired(**kw: Any) -> None:
    name = 'FINAL_04_paired'
    o = _o(name, kw)
    pr = pd.DataFrame(_read('final_paired')['rows'])
    lab = _label_map()
    best = _read('final_paired')['best']
    fig, axes = figure(o['width'], o['height'], 1, 2, sharey=True)
    fam = pr[pr['family'] == 'vs_best'].sort_values('d_ll', ascending=False).reset_index(drop=True)
    order = fam['key'].tolist()
    for ax, col, ci, hc, xl, scale in ((axes[0], 'd_acc', 'd_acc_ci', 'holm_acc', 'accuracy difference (points)', 100),
                                       (axes[1], 'd_ll', 'd_ll_ci', 'holm_ll', 'log-loss difference', 1)):
        for i, k in enumerate(order):
            r = fam[fam['key'] == k].iloc[0]
            sig = r[hc] < 0.05
            c = '#D1495B' if sig else INK2
            ax.plot([scale * r[ci][0], scale * r[ci][1]], [i, i], color=c, lw=1.0)
            ax.plot(scale * r[col], i, 'o', color=c, ms=3.5, mfc=c if sig else 'white')
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel(xl)
    axes[0].set_yticks(range(len(order)), [lab.get(k, k) for k in order])
    n_sig = int(((fam['holm_acc'] < 0.05) | (fam['holm_ll'] < 0.05)).sum())
    L.save_fig(fig, name, data=pr.assign(d_acc_lo=pr['d_acc_ci'].str[0], d_acc_hi=pr['d_acc_ci'].str[1],
                                         d_ll_lo=pr['d_ll_ci'].str[0], d_ll_hi=pr['d_ll_ci'].str[1]).drop(
                                             columns=['d_acc_ci', 'd_ll_ci']),
               caption=(f'Every candidate against the lowest-log-loss candidate ({lab.get(best, best)}) on the same '
                        f'trials, each in its production form (its own pipeline plus the calibration remedy chosen '
                        f'inside every fold): mean per-person differences with paired person-bootstrap 95 % CIs; filled '
                        f'red: Holm-adjusted sign-flip p below 0.05 within the family ({n_sig} of {len(fam)} differ in '
                        f'accuracy or log loss).'),
               decision='Which candidates the data cannot separate from the best: the leaders among which simplicity, '
                        'robustness and portability decide.')


# ───────────────────────────── 05 input vs model class ─────────────────────────────


def fig_FINAL_05_input_vs_class(**kw: Any) -> None:
    name = 'FINAL_05_input_vs_class'
    o = _o(name, kw)
    rows = pd.DataFrame(_read('final_input_class')['rows'])
    lab = _label_map()
    fig, ax = figure(o['width'], o['height'])
    for i, r in rows.iterrows():
        sig = r['holm_acc'] < 0.05
        c = '#D1495B' if sig else INK2
        ax.plot([100 * r['d_acc_ci'][0], 100 * r['d_acc_ci'][1]], [i, i], color=c, lw=1.0)
        ax.plot(100 * r['d_acc'], i, 'o', color=c, ms=3.5, mfc=c if sig else 'white')
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows)), [f'{lab.get(a, a)} - {lab.get(b, b)}' for a, b in zip(rows['a'], rows['b'])])
    ax.set_xlabel('accuracy difference A - B (points)')
    L.save_fig(fig, name, data=rows.drop(columns=['d_acc_ci', 'd_ll_ci']),
               caption=('Input against model class (task 2b): paired accuracy differences (mean per person, 95 % '
                        'person-bootstrap CI; filled red: Holm p below 0.05 over these comparisons) between models '
                        'given the same channels.'),
               decision='Whether the deep model\'s advantage needs the network or only the body-frame input.')


# ───────────────────────────── 06 robustness ─────────────────────────────


def fig_FINAL_06_robustness(**kw: Any) -> None:
    name = 'FINAL_06_robustness'
    o = _o(name, kw)
    rob = _read('final_robustness')['rows']
    master = {r['key']: r for r in _read('final_master')['rows']}
    fin, _ = _final()
    nan = math.nan
    rows = []
    for r in rob:
        k = r['key']
        g, f, pl, nz = r.get('grip') or {}, r.get('flip') or {}, r.get('platform') or {}, r.get('noise') or {}
        rows.append({'key': k, 'label': r['label'], 'recorded': master[k]['accuracy'],
                     'grip_all_grips': g.get('acc_all_grips', nan), 'grip_upright_only': g.get('acc_upright_only', nan),
                     'grip_d': g.get('d_acc', nan), 'grip_holm': g.get('holm_acc', nan),
                     'flip_recorded': f.get('acc_recorded', nan), 'flip_raw': f.get('acc_flipped', nan),
                     'flip_check': f.get('acc_flipped_remedy', nan), 'flip_d': f.get('d_acc', nan),
                     'flip_holm': f.get('holm_acc', nan), 'flip_invariant': bool(f.get('invariant_inputs', False)),
                     'platform_d': pl.get('d_acc', nan), 'platform_holm': pl.get('holm_acc', nan),
                     'noise_d': nz.get('d_acc', nan), 'noise_holm': nz.get('holm_acc', nan)})
    df = pd.DataFrame(rows).sort_values('recorded').reset_index(drop=True)
    fig, axes = figure(o['width'], o['height'], 1, 3, sharey=True, gridspec_kw={'width_ratios': [1, 1.25, 0.9]})
    y = np.arange(len(df))
    red, ok = HAND_COLORS['left'], INK2
    a1, a2, a3 = axes
    for i, r in df.iterrows():
        if np.isfinite(r['grip_upright_only']):
            sig = r['grip_holm'] < 0.05 and r['grip_d'] < 0
            a1.plot([r['grip_all_grips'], r['grip_upright_only']], [i, i], color=MUTED, lw=0.8)
            a1.plot(r['grip_all_grips'], i, 'o', color=INK, ms=3.2)
            a1.plot(r['grip_upright_only'], i, 's', color=red if sig else ok, ms=3.4, mfc=red if sig else 'white')
        if np.isfinite(r['flip_raw']):
            sig = r['flip_holm'] < 0.05 and r['flip_d'] < 0
            a2.plot([min(r['flip_raw'], r['flip_check']), r['flip_recorded']], [i, i], color=MUTED, lw=0.8)
            a2.plot(r['flip_recorded'], i, 'o', color=INK, ms=3.2)
            if not r['flip_invariant']:
                a2.plot(r['flip_raw'], i, 'v', color=red if sig else ok, ms=3.8, mfc=red if sig else 'white')
                a2.plot(r['flip_check'], i, '^', color=PLATFORM_COLORS['android'], ms=3.8)
        for col, mk, c in (('platform_d', 'D', PLATFORM_COLORS['ios']), ('noise_d', 'x', MUTED)):
            if np.isfinite(r[col]):
                a3.plot(100 * r[col], i, mk, color=c, ms=3.4, mew=1.0)
    for a in (a1, a2):
        a.axvline(0.5, color=MUTED, lw=0.6, ls='--')
        a.xaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    a3.axvline(0, color=MUTED, lw=0.6)
    a1.set_xlabel('rotated-grip trials')
    a2.set_xlabel('screen-away twins')
    a3.set_xlabel('change (points)')
    a1.set_title('grip absent from training', fontsize=8)
    a2.set_title('screen held away (synthesised)', fontsize=8)
    a3.set_title('platform / label noise', fontsize=8)
    a1.set_yticks(y, df['label'])
    for t, (_, r) in zip(a1.get_yticklabels(), df.iterrows()):
        if r['key'] == fin:
            t.set_fontweight('bold')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, marker='o', ls='', ms=3.2, label='trained on every grip / as recorded'),
          Line2D([], [], color=ok, marker='s', ls='', ms=3.4, mfc='white', label='trained on upright grips only'),
          Line2D([], [], color=ok, marker='v', ls='', ms=3.8, mfc='white', label='screen away, no check'),
          Line2D([], [], color=PLATFORM_COLORS['android'], marker='^', ls='', ms=3.8, label='screen away, rest-pose check'),
          Line2D([], [], color=PLATFORM_COLORS['ios'], marker='D', ls='', ms=3.4, label='trained on the other platform'),
          Line2D([], [], color=MUTED, marker='x', ls='', ms=3.4, label='trained without suspect blocks'),
          Line2D([], [], color=red, marker='s', ls='', ms=3.4, label='filled red: Holm p < 0.05 (drop)')]
    fig.legend(handles=hs, loc='outside lower center', ncol=4, fontsize=6, handletextpad=0.3, columnspacing=0.9)
    n_grip = int(((df['grip_holm'] < 0.05) & (df['grip_d'] < 0)).sum())
    n_g = int(df['grip_upright_only'].notna().sum())
    flip_tested = df[~df['flip_invariant'] & df['flip_raw'].notna()]
    n_flip = int(((flip_tested['flip_holm'] < 0.05) & (flip_tested['flip_d'] < 0)).sum())
    n_fixed = int((flip_tested['flip_check'] >= flip_tested['recorded'] - 0.05).sum())
    L.save_fig(fig, name, data=df,
               caption=(f'Robustness of every candidate to test conditions the PI may choose (rows sorted by recorded '
                        f'accuracy; bold: the final model). Left: accuracy on the rotated-grip trials when trained on '
                        f'every grip (dot) or on upright grips only (square; {n_grip} of {n_g} candidates drop with a '
                        f'Holm-adjusted p below 0.05). Middle: accuracy as recorded (dot) and on the same trials '
                        f'synthesised as held with the screen away, without (down triangle; {n_flip} of '
                        f'{len(flip_tested)} candidates whose inputs change drop significantly) and with the rest-pose '
                        f'screen check (up triangle; {n_fixed} of {len(flip_tested)} back within 5 points of the '
                        f'recorded accuracy). Gravity-frame-only inputs are unchanged by the flip (dot only). Right: '
                        f'accuracy change when trained on the persons of the other platform only and when trained without '
                        f'the suspect wrong-hand blocks.'),
               decision='A model that is not robust to plausible test conditions is not the final model: device-frame '
                        'models fail an unseen grip, and every model that reads the screen-relative frame needs the '
                        'rest-pose check.')


# ───────────────────────────── 07 natural use ─────────────────────────────


def fig_FINAL_07_natural_transfer(**kw: Any) -> None:
    name = 'FINAL_07_natural_transfer'
    o = _o(name, kw)
    nat = pd.DataFrame(_read('final_natural')['rows']).sort_values('expected').reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'])
    for i, r in nat.iterrows():
        ax.plot(r['range95'], [i, i], color=MUTED, lw=2.5, alpha=0.6, solid_capstyle='butt')
        ax.plot(r['expected'], i, '|', color=INK2, ms=8)
        c = '#D1495B' if r['shortfall'] else HAND_COLORS['right']
        ax.plot(r['observed'], i, 'o', color=c, ms=4)
    ax.set_yticks(range(len(nat)), nat['label'])
    fin, _ = _final()
    for t, k in zip(ax.get_yticklabels(), nat['key']):
        if k == fin:
            t.set_fontweight('bold')
    n = int(nat['n'].max())
    ax.set_xlabel(f'free-choice trials answered correctly (right-hand recall, of {n})')
    L.save_fig(fig, name, data=nat.drop(columns=['range95']).assign(lo=nat['range95'].str[0], hi=nat['range95'].str[1]),
               caption=(f'Person-matched natural-use test: correct free-choice trials (dots; all reported hands are the '
                        f'right hand, so this is right-hand recall) against the count expected from the same persons\' '
                        f'right-hand accuracy on the instructed trials (tick, 95 % Poisson-binomial range in grey); red: '
                        f'a shortfall after Holm over the candidates.'),
               decision='A model whose evidence does not transfer to natural use relies on protocol-specific motion.')


# ───────────────────────────── 08 calibration of the final model ─────────────────────────────


def fig_FINAL_08_calibration(**kw: Any) -> None:
    name = 'FINAL_08_calibration'
    o = _o(name, kw)
    cal = _read('final_calibration')
    fin, _ = _final()
    lab = _label_map()
    chosen = cal['chosen']
    fig, axes = figure(o['width'], o['height'], 1, 3)
    data = []
    for ax, key, title in ((axes[0], 'reliability', 'all trials'), (axes[1], 'reliability_android', 'Android'),
                           (axes[2], 'reliability_ios', 'iOS')):
        for m, c in (('none', MUTED), (chosen, SERIES)):
            if m not in cal['methods'] or key not in cal['methods'][m]:
                continue
            rb = pd.DataFrame(cal['methods'][m][key])
            ax.fill_between(rb['p_mean'], rb['boot_lo'], rb['boot_hi'], color=c, alpha=0.18, lw=0)
            ax.plot(rb['p_mean'], rb['y_mean'], 'o-', color=c, ms=2.5, lw=0.9,
                    label='no extra remedy' if m == 'none' else f'extra remedy: {m}')
            data.append(rb.assign(method=m, panel=title))
            if m == chosen and m == 'none':
                break
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('predicted P(right)')
    axes[0].set_ylabel('observed share of right hands')
    axes[0].legend(loc='upper left', fontsize=6.5)
    ov = cal['methods'][chosen]['overall']
    L.save_fig(fig, name, data=pd.concat(data, ignore_index=True),
               caption=(f'Reliability of the final model ({lab.get(fin, fin)}), equal-mass bins with person-bootstrap '
                        f'95 % bands, overall and per platform; shipped remedy: {chosen}. ECE {ov["ece"]:.3f} (perfect-'
                        f'calibration 95 % band up to {ov["ece_null_hi"]:.3f}), slope {ov["slope"]:.2f} '
                        f'[{ov["slope_ci"][0]:.2f}, {ov["slope_ci"][1]:.2f}].'),
               decision='Whether the percentage shown to the user is honest, overall and on each platform.')


def fig_FINAL_09_remedies(**kw: Any) -> None:
    name = 'FINAL_09_remedies'
    o = _o(name, kw)
    cal = _read('final_calibration')
    rows = [{'method': m, 'diff': e['d_log_loss']['diff'], 'lo': e['d_log_loss']['ci'][0], 'hi': e['d_log_loss']['ci'][1],
             'holm': e['d_log_loss']['holm']} for m, e in cal['methods'].items() if m != 'none']
    df = pd.DataFrame(rows)
    fig, ax = figure(o['width'], o['height'])
    for i, r in df.iterrows():
        c = '#D1495B' if r['holm'] < 0.05 else INK2
        ax.plot([r['lo'], r['hi']], [i, i], color=c, lw=1.0)
        ax.plot(r['diff'], i, 'o', color=c, ms=3.5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(df)), df['method'])
    ax.set_xlabel('log loss: remedy - none')
    L.save_fig(fig, name, data=df,
               caption=(f'Calibration remedies of the final model fitted on the training persons\' inner cross-fitted '
                        f'predictions: paired per-person log-loss difference against no remedy (95 % CI; red: Holm p '
                        f'below 0.05 over the remedies). Shipped: {cal["chosen"]}.'),
               decision='Ship a remedy only if it measurably lowers the log loss; prefer none or one parameter.')


# ───────────────────────────── 10 attempts ─────────────────────────────


def fig_FINAL_10_attempts(**kw: Any) -> None:
    name = 'FINAL_10_attempts'
    o = _o(name, kw)
    ag = _read('final_aggregation')
    labels = {'mean_p': 'mean P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits',
              'sum_tau': r'sum of logits / $\tau_k$ (nested)'}
    cols = {'mean_p': MUTED, 'mean_logit': '#eb6834', 'sum_logit': '#D1495B', 'sum_tau': SERIES}
    fig, (a1, a2) = figure(o['width'], o['height'], 1, 2)
    data = []
    ks = sorted(int(k) for k in ag['rules'])
    for rule in labels:
        ov = [ag['rules'][str(k)][rule]['over_confidence'] for k in ks]
        lo = [ag['rules'][str(k)][rule]['over_ci'][0] for k in ks]
        hi = [ag['rules'][str(k)][rule]['over_ci'][1] for k in ks]
        acc = [ag['rules'][str(k)][rule]['accuracy'] for k in ks]
        ll = [ag['rules'][str(k)][rule]['log_loss'] for k in ks]
        llo = [ag['rules'][str(k)][rule]['log_loss_ci'][0] for k in ks]
        lhi = [ag['rules'][str(k)][rule]['log_loss_ci'][1] for k in ks]
        a1.fill_between(ks, lo, hi, color=cols[rule], alpha=0.15, lw=0)
        a1.plot(ks, ov, 'o-', color=cols[rule], ms=2.5, label=labels[rule])
        a2.fill_between(ks, llo, lhi, color=cols[rule], alpha=0.12, lw=0)
        a2.plot(ks, ll, 'o-', color=cols[rule], ms=2.5)
        data.append(pd.DataFrame({'rule': rule, 'k': ks, 'over_confidence': ov, 'lo': lo, 'hi': hi, 'accuracy': acc,
                                  'log_loss': ll, 'log_loss_lo': llo, 'log_loss_hi': lhi}))
    a1.axhline(0, color=INK2, lw=0.6)
    a1.set_xlabel('attempts k')
    a1.set_ylabel('mean confidence - accuracy')
    a2.set_xlabel('attempts k')
    a2.set_ylabel('log loss (per person and hand)')
    a1.legend(fontsize=6, loc='lower left')
    for a in (a1, a2):
        a.set_xticks(ks)
    t3 = ag['tau_median'].get('3', ag['tau_median'].get(3))
    acc3 = [ag['rules']['3'][r]['accuracy'] for r in labels]
    L.save_fig(fig, name, data=pd.concat(data, ignore_index=True),
               caption=(f'Several attempts of one person and hand ({len(ag["persons"])} persons with at least '
                        f'{max(ks)} trials of each hand; random k-subsets within person and hand, the same persons for '
                        f'every k): over-confidence of four aggregates (left; mean confidence minus accuracy per person '
                        f'and hand, person-bootstrap 95 % bands; above 0: over-confident) and their log loss (right). '
                        f'The aggregates give nearly the same answers (accuracy at k = 3 between {100 * min(acc3):.1f} '
                        f'and {100 * max(acc3):.1f} %) and differ in the percentage shown. tau_k is fitted in every '
                        f'outer fold on the training persons\' inner cross-fitted logits (median tau_3 {t3:.2f}).'),
               decision='Which aggregate the page displays: a calibrated one; the mean P(right) is not shown.')


# ───────────────────────────── 11 persons and conditions ─────────────────────────────


def fig_FINAL_11_persons(**kw: Any) -> None:
    name = 'FINAL_11_persons'
    o = _o(name, kw)
    br = _read('final_breakdown')
    pp = pd.DataFrame(br['persons']).sort_values('accuracy').reset_index(drop=True)
    fig, (a1, a2) = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': [1.6, 1]})
    cols = [PLATFORM_COLORS.get(str(p).lower(), SERIES) for p in pp['platform']]
    a1.bar(range(len(pp)), pp['accuracy'], color=cols)
    for i, r in pp.iterrows():
        if str(r['handedness']).lower() == 'left':
            a1.text(i, r['accuracy'] + 0.01, 'L', ha='center', va='bottom', fontsize=6.5, color=INK)
    a1.set_xticks(range(len(pp)), pp['person_id'], rotation=90, fontsize=6)
    a1.set_ylim(0, 1.08)
    a1.set_ylabel('accuracy')
    a1.yaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=PLATFORM_COLORS[k], label=PLATFORM_LABELS[k]) for k in ('android', 'ios')],
              fontsize=6.5, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2, frameon=False)
    gs = pd.DataFrame(br['grip_speed'])
    gs['cell'] = gs['orientation'].str.replace('_toward', '').str.replace('_', ' ') + ', ' + gs['speed']
    gs = gs.sort_values('accuracy').reset_index(drop=True)
    min_p = int(br['min_persons_ci'])
    has_ci = gs['ci'].apply(lambda v: isinstance(v, list) and all(np.isfinite(v))) if 'ci' in gs else pd.Series(False, index=gs.index)
    for i, r in gs.iterrows():
        a2.plot(r['accuracy'], i, 'o', color=SERIES if has_ci[i] else MUTED, ms=3.5,
                mfc=SERIES if has_ci[i] else 'white')
        if has_ci[i]:
            a2.plot(r['ci'], [i, i], color=SERIES, lw=1.0)
    a2.set_yticks(range(len(gs)), [f'{c} ({n} trials, {q} person{"s" if q != 1 else ""})'
                                   for c, n, q in zip(gs['cell'], gs['n'], gs['persons'])], fontsize=6)
    lows = [v[0] for v, h in zip(gs.get('ci', [None] * len(gs)), has_ci) if h] + gs['accuracy'].tolist()
    a2.set_xlim(max(0.0, np.floor(10 * min(lows) - 0.5) / 10), 1.02)
    a2.xaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    a2.set_xlabel('accuracy')
    n_left = int((pp['handedness'].astype(str).str.lower() == 'left').sum())
    gs_data = gs.assign(ci_lo=[v[0] if h else math.nan for v, h in zip(gs.get('ci', [None] * len(gs)), has_ci)],
                        ci_hi=[v[1] if h else math.nan for v, h in zip(gs.get('ci', [None] * len(gs)), has_ci)])
    data = pd.concat([pp.assign(panel='person'),
                      gs_data.drop(columns=[c for c in ('ci', 'per_person') if c in gs_data]).assign(panel='grip x speed')],
                     ignore_index=True)
    L.save_fig(fig, name, data=data,
               caption=(f'Final model by person (left; sorted; colour: platform; L: the {n_left} left-handed persons) '
                        f'and by grip x speed cell (right; the instructed conditions only stratify the predictions; '
                        f'person-bootstrap 95 % CI for cells with at least {min_p} persons; open markers: fewer persons, '
                        f'no CI, their persons\' values are in the breakdown table).'),
               decision='Which persons and conditions the final model fails on.')


# ───────────────────────────── 12 learning curve ─────────────────────────────


def fig_FINAL_12_learning_curve(**kw: Any) -> None:
    name = 'FINAL_12_learning_curve'
    o = _o(name, kw)
    lc = {k: v for k, v in _read('final_learning').items() if isinstance(v, dict) and 'summary' in v}
    lab = _label_map()
    fig, (a1, a2) = figure(o['width'], o['height'], 1, 2)
    data = []
    for i, (key, v) in enumerate(lc.items()):
        c = SERIES if i == 0 else MUTED
        pts = pd.DataFrame(v['points'])
        s = pd.DataFrame(v['summary'])
        a1.scatter(pts['n'], pts['accuracy'], s=6, color=c, alpha=0.5, lw=0)
        a1.plot(s['n'], s['acc_mean'], 'o-', color=c, ms=3, label=lab.get(key, key))
        a2.scatter(pts['n'], pts['log_loss'], s=6, color=c, alpha=0.5, lw=0)
        a2.plot(s['n'], s['ll_mean'], 'o-', color=c, ms=3)
        if isinstance(v.get('fit'), dict):
            f = v['fit']
            a1.plot(f['grid'], f['fit'], color=c, lw=0.8, ls='--')
            a1.fill_between(f['grid'], f['lo'], f['hi'], color=c, alpha=0.12, lw=0)
        data.append(pts.assign(key=key))
    from matplotlib.ticker import MultipleLocator
    a1.set_xlabel('training persons')
    a1.set_ylabel('LOSO accuracy')
    a1.yaxis.set_major_locator(MultipleLocator(o.get('acc_step', 0.05)))
    a1.yaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    a2.set_xlabel('training persons')
    a2.set_ylabel('log loss')
    sizes = sorted({int(n) for v in lc.values() for n in pd.DataFrame(v['summary'])['n']})
    a2.set_xticks(sizes)
    a1.legend(fontsize=6.5, loc='lower right')
    fin = next(iter(lc))
    f = lc[fin].get('fit') if isinstance(lc[fin].get('fit'), dict) else {}
    at = f.get('at_40')
    ext = (f' The inverse power law fitted to {lab.get(fin, fin)} extrapolates to {100 * at[0]:.1f} % at 40 persons '
           f'[{100 * at[1]:.1f}, {100 * at[2]:.1f}] (bootstrap over repetitions; an extrapolation).') if at else ''
    L.save_fig(fig, name, data=pd.concat(data, ignore_index=True),
               caption=(f'Learning curves: training persons drawn at random inside every outer fold (dots: '
                        f'repetitions; line: their mean; dashed: inverse power law with a 95 % bootstrap band).' + ext),
               decision='Whether recording more participants looks worthwhile.')


def fig_FINAL_13_segmentation(**kw: Any) -> None:
    name = 'FINAL_13_segmentation'
    o = _o(name, kw)
    rows = pd.DataFrame(_read('final_variants')['rows'])
    fig, ax = figure(o['width'], o['height'])
    for i, r in rows.iterrows():
        c = '#D1495B' if r['holm_acc'] < 0.05 else INK2
        ax.plot([100 * r['d_acc_ci'][0], 100 * r['d_acc_ci'][1]], [i, i], color=c, lw=1.0)
        ax.plot(100 * r['d_acc'], i, 'o', color=c, ms=3.5)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows)), rows['description'], fontsize=6)
    ax.set_xlabel('accuracy change (points)')
    L.save_fig(fig, name, data=rows.drop(columns=['d_acc_ci', 'd_ll_ci']),
               caption=('Final model trained on the oracle bounds and tested on windows cut with segmentation errors '
                        'or a causal filter (lane p1\'s variant arrays, the same canonical code path): paired accuracy '
                        'change with 95 % CIs (red: Holm p below 0.05).'),
               decision='What the production detector must achieve: the bound accuracy the model tolerates.')


def fig_FINAL_14_prior(**kw: Any) -> None:
    name = 'FINAL_14_prior'
    o = _o(name, kw)
    pr = pd.DataFrame(_read('final_prior')['rows'])
    fig, ax = figure(o['width'], o['height'])
    ax.plot(pr['prior'], pr['left_recall'], 'o-', color=HAND_COLORS['left'], ms=3, label='left-hand recall')
    ax.plot(pr['prior'], pr['right_recall'], 'o-', color=HAND_COLORS['right'], ms=3, label='right-hand recall')
    ax.plot(pr['prior'], pr['balanced_accuracy'], 'o-', color=INK2, ms=3, label='balanced accuracy')
    ax.set_xlabel('deployment prior of the right hand')
    ax.yaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    ax.legend(fontsize=6.5, loc='lower left')
    L.save_fig(fig, name, data=pr,
               caption=('A deployment prior pi of the right hand adds log(pi / (1 - pi)) to every logit: right-hand '
                        'recall rises, left-hand recall falls and the balanced accuracy drops (primary trials). The '
                        'shipped default is pi = 0.5 (the PI chooses the hand).'),
               decision='Keep the prior at 0.5 for the PI\'s evaluation.')


FIGURE_FUNCTIONS = {k: globals()[f'fig_{k}'] for k in FIGURES}


#: the derived file each figure needs, and the field that must be non-empty (a figure whose experiments have not run is
#: skipped with a message instead of failing the step)
NEEDS: dict[str, tuple[str, str | None]] = {
    'FINAL_03_nested_selection': ('final_nested', 'frequency'), 'FINAL_08_calibration': ('final_calibration', 'methods'),
    'FINAL_09_remedies': ('final_calibration', 'methods'), 'FINAL_10_attempts': ('final_aggregation', 'rules'),
    'FINAL_12_learning_curve': ('final_learning', None), 'FINAL_13_segmentation': ('final_variants', 'rows'),
}


def _ready(name: str) -> bool:
    src, field = NEEDS.get(name, (None, None))
    if src is None:
        return True
    if not (L.tab_dir / f'{src}.json').exists():
        return False
    doc = L.read_json(src)
    if field is None:                                  # the learning curve: at least one model with a summary
        return any(isinstance(v, dict) and 'summary' in v for v in doc.values())
    return bool(doc.get(field))


def draw(only: list[str] | None = None) -> None:
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            if not _ready(name):
                print(f'final: figure {name} skipped (its experiments have not run yet)', flush=True)
                continue
            fn()
