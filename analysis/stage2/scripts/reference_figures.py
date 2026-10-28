"""Figures of the core unit 'reference', drawn ONLY from saved files (tables/reference_summary.json).

One function per figure, named after it. Every presentation detail is a keyword argument whose default
lives in ``FIGURES`` below, so a change is a one-line edit followed by

    python -m stage2.pipeline figures --lanes reference          # seconds, no model is fitted

Each function also writes the plotted data as a tidy CSV (tables/reference/<figure name>.csv).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import pipeline as P
from stage2.plotting import INK, INK2, SERIES, figure, save_fig

#: per-figure presentation defaults (edit here; or pass keyword arguments)
FIGURES: dict[str, dict[str, Any]] = {
    'S2_21_reference_sensitivity': dict(
        width='full', height=2.4, width_ratios=(1, 1.15),
        k_ylim=(0.5, 1.0), k_yticks=None, k_xpad=0.4, k_annotate=True, k_fontsize=6,
        k_xlabel='attempts averaged (same person and hand)', k_ylabel='per-person balanced accuracy',
        diff_xlim=None, diff_label_fontsize=6, diff_ytick_fontsize=6.5,
        diff_xlabel='accuracy minus reference\n(points, paired 95 % CI)',
        color=SERIES, marker_size=3.5,
    ),
}
DATA_DIR = C.TAB_DIR / 'reference'                 # tidy CSV of each figure's plotted data


def _figure_data(df: pd.DataFrame, name: str) -> None:
    from stage2.report_export import export_table
    export_table(df, DATA_DIR / name, tex=False)


def fig_S2_21_reference_sensitivity(**kw: Any) -> dict[str, str]:
    """k-attempt curve of the reference floor (left) and the label-noise variants paired against it (right)."""
    from stage2.scripts.reference import LABELS, REFERENCE, load_summary
    name = 'S2_21_reference_sensitivity'
    o = {**FIGURES[name], **kw}
    s = load_summary()
    ref = s['reference']
    kc = pd.DataFrame(ref['k_attempts'])
    sv = pd.DataFrame([r for r in s['sensitivity'] if r['name'] != REFERENCE]).reset_index(drop=True)
    note_swap = ', '.join(f"{b['session_label']} block {b['block_index']}" for b in s['swap_blocks']) or 'none'

    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    a.fill_between(kc['k'], kc['lo'], kc['hi'], color=o['color'], alpha=0.18, lw=0)
    a.plot(kc['k'], kc['accuracy'], 'o-', color=o['color'], ms=o['marker_size'])
    if o['k_annotate']:
        for _, r in kc.iterrows():
            a.annotate(f"{100 * r['accuracy']:.0f} %", (r['k'], r['accuracy']), textcoords='offset points',
                       xytext=(0, 6), ha='center', fontsize=o['k_fontsize'], color=INK2)
    a.set_xticks(list(kc['k']))
    a.set_xlim(min(kc['k']) - o['k_xpad'], max(kc['k']) + o['k_xpad'])
    a.set_ylim(*o['k_ylim'])
    if o['k_yticks'] is not None:
        a.set_yticks(o['k_yticks'])
    a.set_xlabel(o['k_xlabel'])
    a.set_ylabel(o['k_ylabel'])
    n_k = int(kc['n_persons'].iloc[0]) if len(kc) else 0
    a.set_title(f'Reference: k attempts ({n_k} persons at every k)')

    a = ax[1]
    y = np.arange(len(sv))
    d, lo, hi = sv['acc_diff_vs_reference'] * 100, sv['acc_diff_lo'] * 100, sv['acc_diff_hi'] * 100
    a.errorbar(d, y, xerr=[d - lo, hi - d], fmt='o', color=o['color'], ms=o['marker_size'] + 0.5, capsize=2, lw=1.0)
    for i, r in sv.iterrows():
        a.text(r['acc_diff_hi'] * 100 + 0.3, i, f"{100 * r['accuracy']:.1f} % (n = {r['n_trials']}; "
               f"p = {r['p_sign_flip']:.2f})", va='center', fontsize=o['diff_label_fontsize'], color=INK2)
    a.axvline(0, color=INK, lw=0.7)
    a.set_yticks(y, [LABELS.get(n, n) for n in sv['name']], fontsize=o['diff_ytick_fontsize'])
    a.invert_yaxis()
    xlim = o['diff_xlim'] or (min(-2.5, lo.min() - 1), max(4, hi.max() + 6.5))
    a.set_xlim(*xlim)
    a.set_xlabel(o['diff_xlabel'])
    a.grid(axis='y', visible=False)
    a.set_title(f"Reference {100 * ref['accuracy']:.1f} %: label-noise sensitivity")

    within = bool(((sv['acc_diff_lo'] <= 0) & (sv['acc_diff_hi'] >= 0)).all())
    decision = ('Averaging attempts is worth asking for in the inference app. '
                + ('No label-noise variant differs from the reference beyond its paired 95 % CI, so the labels stay '
                   'as instructed and the D6 screen stays optional.' if within else
                   'A label-noise variant differs from the reference beyond its paired 95 % CI: report both and raise '
                   'the blocks with the researcher before choosing.'))
    caption = ('Left: accuracy of the reference floor (logistic regression on the cumulative yaw at 8 instants, '
               'mirror-augmented, leave-one-person-out) when the P(right) of k attempts of one person and hand are '
               f'averaged; every k uses the same {n_k} persons (those with at least {int(max(kc["k"])) if len(kc) else 0} '
               f'trials per hand). Right: the reference without the swap-signature blocks ({note_swap}) or with the '
               'docs/09 D6 screen, paired against the reference on the same trials.')
    out = save_fig(fig, name, caption=caption, decision=decision)
    data = pd.concat([
        pd.DataFrame({'panel': 'k_attempts', 'series': 'reference', 'x': kc['k'], 'y': kc['accuracy'], 'lo': kc['lo'],
                      'hi': kc['hi'], 'n': kc['n_persons']}),
        pd.DataFrame({'panel': 'sensitivity', 'series': sv['name'], 'x': sv['acc_diff_vs_reference'],
                      'y': sv['accuracy'], 'lo': sv['acc_diff_lo'], 'hi': sv['acc_diff_hi'], 'n': sv['n_trials']}),
    ], ignore_index=True)
    _figure_data(data, name)
    return out


FIGURE_FUNCTIONS = {'S2_21_reference_sensitivity': fig_S2_21_reference_sensitivity}


def draw(only: list[str] | None = None) -> list[str]:
    """Draw every figure of the unit (or those matching ``only``) from the saved files."""
    done = []
    for name, fn in FIGURE_FUNCTIONS.items():
        if P.selected(name, only):
            fn()
            done.append(name)
    return done
