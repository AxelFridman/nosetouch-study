"""Figures of lane demo (the demographics add-on test), from saved files only.
Edit FIGURES, then: python -m stage2.pipeline figures --lanes demo"""
from __future__ import annotations

from typing import Any

import pandas as pd

from stage2.pipeline import Lane, selected
from stage2.plotting import HAND_COLORS, INK, MUTED, figure

L = Lane('demo')
SIG = '#D1495B'
FIGURES: dict[str, dict[str, Any]] = {
    'DEMO_01_paired': dict(width='full', height=1.9, acc_xlabel='accuracy difference (points)',
                           ll_xlabel='log-loss difference', ms=4.0),
    'DEMO_02_base_rate': dict(width='full', height=2.6, xlim=(0, 1.04), ms=5.0,
                              primary_color=INK, natural_color=HAND_COLORS['right']),
}
LABELS = {'demo_sensor_stack': 'sensor only (second stage)', 'demo_cov_main': '+ covariates',
          'demo_cov_inter': '+ covariates and covariate x logit', 'demo_cov_only': 'covariates only',
          'demo_rule_phone': 'rule: usual phone hand', 'demo_rule_right': 'rule: always right',
          'final_cal_m3_deep_none': 'final model as shipped'}


def _fmt_p(p: float) -> str:
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def fig_DEMO_01_paired(**kw: Any) -> None:
    name = 'DEMO_01_paired'
    o = {**FIGURES[name], **kw}
    D = L.read_json('demo_derived')
    fam = D['family']
    ref = D['descriptive']['sensor_vs_shipped']
    rows = []
    for v, f in fam.items():
        rows.append({'comparison': f'{LABELS[v]} - sensor only', 'family': 'add-on (Holm)',
                     'd_acc': 100 * f['mean_person_acc_diff'], 'd_acc_lo': 100 * f['mean_person_acc_diff_ci95'][0],
                     'd_acc_hi': 100 * f['mean_person_acc_diff_ci95'][1], 'p_acc': f['holm_p_acc'],
                     'd_ll': f['mean_person_logloss_diff'], 'd_ll_lo': f['mean_person_logloss_diff_ci95'][0],
                     'd_ll_hi': f['mean_person_logloss_diff_ci95'][1], 'p_ll': f['holm_p_logloss']})
    rows.append({'comparison': 'sensor only - final model as shipped', 'family': 'check (unadjusted)',
                 'd_acc': 100 * ref['mean_person_acc_diff'], 'd_acc_lo': 100 * ref['mean_person_acc_diff_ci95'][0],
                 'd_acc_hi': 100 * ref['mean_person_acc_diff_ci95'][1], 'p_acc': ref['p_sign_flip_acc'],
                 'd_ll': ref['mean_person_logloss_diff'], 'd_ll_lo': ref['mean_person_logloss_diff_ci95'][0],
                 'd_ll_hi': ref['mean_person_logloss_diff_ci95'][1], 'p_ll': ref['p_sign_flip_logloss']})
    df = pd.DataFrame(rows)
    n = len(df)
    fig, axes = figure(o['width'], o['height'], 1, 2, sharey=True)
    for ax, col, pc, xl in ((axes[0], 'd_acc', 'p_acc', o['acc_xlabel']), (axes[1], 'd_ll', 'p_ll', o['ll_xlabel'])):
        for i, r in df.iterrows():
            y = n - 1 - i
            sig = r['family'].startswith('add-on') and r[pc] < 0.05
            c = SIG if sig else (INK if r['family'].startswith('add-on') else MUTED)
            ax.plot([r[f'{col}_lo'], r[f'{col}_hi']], [y, y], color=c, lw=1.1)
            ax.plot(r[col], y, 'o', color=c, ms=o['ms'], mfc=c if sig else 'white')
            ptxt = ('Holm p ' if r['family'].startswith('add-on') else 'p ') + _fmt_p(r[pc])
            ax.annotate(ptxt, (r[f'{col}_hi'], y), xytext=(4, 0), textcoords='offset points', va='center',
                        fontsize=6.5, color=c)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel(xl)
        lo, hi = df[f'{col}_lo'].min(), df[f'{col}_hi'].max()
        span = hi - lo
        ax.set_xlim(min(lo, 0) - 0.08 * span, max(hi, 0) + 0.55 * span)
    axes[0].set_yticks(range(n), df['comparison'].tolist()[::-1])
    axes[0].set_ylim(-0.6, n - 0.4)
    L.save_fig(fig, name, data=df,
               caption=('Adding the person covariates (age, sex, height, self-reported handedness, usual phone hand) '
                        'to a second stage on the final model\'s logit, against the same second stage without them, on '
                        'the same leave-one-person-out folds: mean per-person differences with paired person-bootstrap '
                        '95\\% CIs, Holm-adjusted sign-flip p over the add-on variants within each metric (filled red: '
                        'below 0.05). Grey: the sensor-only second stage against the shipped model (a check that the '
                        'stacking itself is neutral; unadjusted). Negative accuracy and positive log-loss differences '
                        'mean the covariates hurt.'),
               decision='Demographics do not improve the hand prediction even if production knew them; nothing from '
                        'this test enters the production model.')


def fig_DEMO_02_base_rate(**kw: Any) -> None:
    name = 'DEMO_02_base_rate'
    o = {**FIGURES[name], **kw}
    D = L.read_json('demo_derived')
    order = ['final_cal_m3_deep_none', 'demo_sensor_stack', 'demo_cov_main', 'demo_cov_inter', 'demo_cov_only',
             'demo_rule_phone', 'demo_rule_right']
    by = {r['name']: r for r in D['models']}
    df = pd.DataFrame([{'model': LABELS[k], 'primary_balanced_accuracy': by[k]['balanced_accuracy'],
                        'natural_use_right_hand_recall': by[k]['nat_recall'], 'natural_use_n': by[k]['nat_n']}
                       for k in order])
    n = len(df)
    fig, ax = figure(o['width'], o['height'])
    for i, r in df.iterrows():
        y = n - 1 - i
        a, b = r['primary_balanced_accuracy'], r['natural_use_right_hand_recall']
        ax.plot([a, b], [y, y], color=MUTED, lw=0.8, zorder=1)
        ax.plot(a, y, 'o', color=o['primary_color'], ms=o['ms'], zorder=2)
        ax.plot(b, y, 's', color=o['natural_color'], ms=o['ms'], mfc='white', mew=1.3, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls='--')
    ax.set_yticks(range(n), df['model'].tolist()[::-1])
    ax.set_xlim(*o['xlim'])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_ylim(-0.6, n - 0.4)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker='o', ls='', color=o['primary_color'], label='primary set: balanced accuracy'),
                       Line2D([0], [0], marker='s', ls='', color=o['natural_color'], mfc='white', mew=1.3,
                              label='natural use: right-hand recall')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=7, borderaxespad=0.2)
    nn = int(df['natural_use_n'].iloc[0])
    L.save_fig(fig, name, data=df,
               caption=(f'The base-rate effect. Balanced accuracy on the primary set (both hands for every person) and '
                        f'right-hand recall on the {nn} natural-use trials (every reported free-choice hand so far is '
                        f'the right hand, so a constant \'right\' answer is never wrong there). Fixed rules that answer the usual '
                        f'phone hand, or always right, look excellent on natural use and sit at chance on the primary '
                        f'set: a phone-hand prior gains nothing in discrimination. The covariates-only model is at '
                        f'chance on the primary set.'),
               decision='Natural-use recall is never read as a gain in discrimination; the add-on test is judged on the '
                        'primary set.')


FIGURE_FUNCTIONS = {'DEMO_01_paired': fig_DEMO_01_paired, 'DEMO_02_base_rate': fig_DEMO_02_base_rate}


def draw(only=None):
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
