"""Figures of lane p1, drawn from the saved derived files only (tables/p1/*.json); nothing is fitted here.

Every number drawn in a figure or written in a caption is printed through :class:`p1_numbers.Numbers` (the same
entries as results/p1_numbers.json). Imports are light (labels, numbers, plotting), so a redraw needs no model code.

Markers (every paired-difference figure): filled = the paired 95 % CI excludes 0 AND the Benjamini-Hochberg q of the
exact sign-flip test, within the comparison's multiplicity family, is below alpha; light fill = the CI alone excludes
0 (nominal, not significant after the correction); open circle = no difference; open square = identical predictions
(solver precision); hollow star = the current pipeline (the reference, 0 by definition).

Edit FIGURES (sizes, limits, ticks, labels), then: python -m stage2.pipeline figures --lanes p1 [--only P1_02_cutoff]
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.lanes import p1_labels as LB
from stage2.lanes.p1_labels import MODEL_LABELS, MODELS, SHORT
from stage2.lanes.p1_numbers import Numbers, rkey
from stage2.pipeline import Lane, selected
from stage2.plotting import GRID, INK, INK2, MUTED, PLATFORM_COLORS, PLATFORM_LABELS, figure

L = Lane('p1')
MODEL_COLORS = {'lr': '#2a78d6', 'gbm': '#eb6834', 'mr': '#1baf7a'}

FIGURES: dict[str, dict[str, Any]] = {
    'P1_01_forest': dict(width='full', row_in=0.118, head_in=0.62, xlim=None, xstep=10, marker_size=3.2,
                         xlabel='Accuracy difference vs the current pipeline (points)'),
    'P1_02_cutoff': dict(width='full', height=2.5, ylim=None, dodge=0.12, width_ratios=(5, 4),
                         xlabel='Low-pass cut-off (Hz)', xlabel_order=None, ylabel='Accuracy difference (points)'),
    'P1_03_segmentation': dict(width='half', height=2.9, ylim=None, dodge=5.0, side=24.0,
                               xlabel='Bound error at test time, $\\pm J$ (ms)', ylabel='Accuracy difference (points)'),
    'P1_04_sampling': dict(width='full', height=2.4, ylim=None, dodge_rate=0.08, dodge_jitter=0.25,
                           ylabel='Accuracy difference (points)'),
    'P1_05_normalisation': dict(width='full', height=2.0, xlim_acc=None, xlim_ll=None),
    'P1_06_reliability': dict(width='full', height=3.1, lim=(0.0, 1.0), ticks=(0, 0.25, 0.5, 0.75, 1.0), dodge=0.006),
    'P1_07_aggregates': dict(width='full', height=2.3, model='gbm', ks=None),
    'P1_08_platform': dict(width='full', height=2.5, ylim=None, dodge=0.13),
    'P1_09_exclusions': dict(width='full', row_in=0.17, head_in=0.62, xlim=(0.0, 1.03), xticks=(0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    'P1_10_filter_delay': dict(width='half', height=2.2, fmax=10.0, cutoffs=(10.0, 15.0, 20.0), order=None,
                               ylabel='Group delay (ms)', xlabel='Frequency (Hz)'),
}

MARKER_NOTE = ('filled: the paired 95 % CI excludes 0 and the Benjamini-Hochberg q within the comparison family is below '
               '{alpha}; light fill: the CI alone excludes 0 (not significant after the correction); open circle: no '
               'difference; open square: identical predictions (solver precision); hollow star: the current pipeline')


def _N() -> Numbers:
    return Numbers()


def _runs() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('p1_runs')['runs'])


def _comps() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('p1_comparisons')['comparisons'])


def _grid_order() -> list[str]:
    return list(L.read_json('p1_runs').get('grid_order', []))


def _note(N: Numbers) -> str:
    return MARKER_NOTE.format(alpha=N('alpha'))


def tint(color: str, amount: float = 0.6) -> str:
    """The colour mixed with white (amount = share of white): the face of a nominal-only marker."""
    c = color.lstrip('#')
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return '#' + ''.join(f'{round(v + (255 - v) * amount):02x}' for v in rgb)


def marker_kw(verdict: str, color: str, ms: float = 3.2) -> dict[str, Any]:
    """Marker style of a paired difference by its verdict (module docstring)."""
    v = str(verdict)
    if v == 'default':
        return dict(marker='*', ms=ms * 1.9, mfc='white', mec=color, mew=0.9, ls='none')
    if v == 'identical':
        return dict(marker='s', ms=ms * 0.9, mfc='white', mec=color, mew=0.9, ls='none')
    if v in ('better', 'worse'):
        return dict(marker='o', ms=ms, mfc=color, mec=color, mew=0.9, ls='none')
    if v.startswith('nominally'):
        return dict(marker='o', ms=ms, mfc=tint(color), mec=color, mew=0.9, ls='none')
    return dict(marker='o', ms=ms, mfc='white', mec=color, mew=0.9, ls='none')


# ───────────────────────────── P1_01: every option, paired ─────────────────────────────


def fig_P1_01_forest(**kw: Any) -> None:
    o = {**FIGURES['P1_01_forest'], **kw}
    N = _N()
    c = _comps()
    c = c[c['mult_family'] == 'options'].copy()
    order = _grid_order()
    fam_order = [f for f in LB.DECISION_FAMILIES if f in set(c['report_family'])]
    c['fam_i'] = c['report_family'].map({f: i for i, f in enumerate(fam_order)})
    c['opt_i'] = c['name'].map(lambda n: order.index(n) if n in order else len(order))
    rows = (c.groupby(['report_family', 'option'])[['fam_i', 'opt_i']].min().reset_index()
            .sort_values(['fam_i', 'opt_i']).reset_index(drop=True))
    ypos, labels, fam_span, y = {}, [], {}, 0.0
    for _, g in rows.groupby('fam_i', sort=True):
        fname = g['report_family'].iloc[0]
        start = y
        for _, r in g.iterrows():
            ypos[(r['report_family'], r['option'])] = y
            labels.append((y, LB.option_label(r['option'])))
            y += 1.0
        fam_span[fname] = (start, y - 1.0)
        y += 0.8
    height = o['head_in'] + o['row_in'] * y
    fig, axes = figure(o['width'], height, 1, len(MODELS), sharey=True)
    d = 100 * c[['d_acc', 'd_acc_lo', 'd_acc_hi']]
    lim = o['xlim'] or (np.floor(d['d_acc_lo'].min() / o['xstep']) * o['xstep'],
                        np.ceil(max(d['d_acc_hi'].max(), 1) / o['xstep']) * o['xstep'])
    for ax, m in zip(axes, MODELS):
        cm = c[c['model'] == m]
        ax.axvline(0, color=INK2, lw=0.6)
        for (f0, f1) in fam_span.values():
            ax.axhspan(f0 - 0.5, f1 + 0.5, color=GRID, alpha=0.35, lw=0)
        for _, r in cm.iterrows():
            yy = ypos[(r['report_family'], r['option'])]
            ax.plot([100 * r['d_acc_lo'], 100 * r['d_acc_hi']], [yy, yy], color=MODEL_COLORS[m], lw=0.9)
            ax.plot(100 * r['d_acc'], yy, **marker_kw(r['verdict'], MODEL_COLORS[m], o['marker_size']))
        ax.set_xlim(*lim)
        ax.set_xticks(np.arange(lim[0], lim[1] + 1e-9, o['xstep']))
        ax.set_title(f'{SHORT[m]}\n{MODEL_LABELS[m]}', fontsize=7)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=6)
    axes[0].set_ylim(y - 0.3, -0.7)
    for fname, (f0, f1) in fam_span.items():
        axes[-1].text(1.03, (f0 + f1) / 2, LB.FAMILY_LABELS[fname].replace(' (', '\n('),
                      transform=axes[-1].get_yaxis_transform(), fontsize=6, va='center', ha='left', color=INK2)
    fig.supxlabel(o['xlabel'], fontsize=8)
    data = c[['report_family', 'option', 'model', 'n_shared', 'd_acc', 'd_acc_lo', 'd_acc_hi', 'p_sign_flip', 'q_bh',
              'p_holm', 'verdict']].copy()
    L.save_fig(fig, 'P1_01_forest', data=data,
               caption=(f"Every data-treatment option against the current pipeline, for each panel model: paired "
                        f"difference in leave-one-person-out accuracy (points) with its 95 % person-bootstrap CI "
                        f"({N('n_persons')} persons; {N('fam_options_size')} comparisons in one multiplicity family). "
                        f"{_note(N)[0].upper() + _note(N)[1:]}. The exclusion-rule rows add the excluded trials to "
                        'training and score the primary trials only.'),
               decision='Which preprocessing and data-treatment defaults the evidence supports keeping or changing.')


# ───────────────────────────── paired curves (P1_02 - P1_04) ─────────────────────────────


DELTA_COLUMNS = ('model', 'option', 'x', 'd', 'lo', 'hi', 'verdict')


def _delta_points(c: pd.DataFrame, m: str, pts: list[tuple[str, float]]) -> pd.DataFrame:
    """Paired accuracy differences (points) of model m at the options pts = [(option, x)]; 'base' = 0."""
    rows: list[dict[str, Any]] = []
    for opt, x in pts:
        if opt == 'base':
            rows.append({'model': m, 'option': opt, 'x': x, 'd': 0.0, 'lo': 0.0, 'hi': 0.0, 'verdict': 'default'})
            continue
        g = c[(c['model'] == m) & (c['option'] == opt)]
        if len(g):
            g = g.iloc[0]
            rows.append({'model': m, 'option': opt, 'x': x, 'd': 100 * g['d_acc'], 'lo': 100 * g['d_acc_lo'],
                         'hi': 100 * g['d_acc_hi'], 'verdict': g['verdict']})
    return pd.DataFrame(rows, columns=list(DELTA_COLUMNS))


def _concat(frames: list[pd.DataFrame], columns: tuple[str, ...] = DELTA_COLUMNS) -> pd.DataFrame:
    """Concatenated plotted data with a header even when nothing was plotted (never a column-less CSV)."""
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns))


def _base_label(N: Numbers, m: str) -> str:
    return f"{SHORT[m]} ({N(m + '_accuracy')})" if N.ok(f'{m}_accuracy') else SHORT[m]


def _draw_delta(ax: Any, df: pd.DataFrame, m: str, dodge: float, label: str | None, categorical: bool = False) -> None:
    """Points with CIs joined by a line; on a categorical axis only neighbouring categories are joined (a model
    without the option in between is not interpolated across it)."""
    if len(df) < 2:
        return
    df = df.sort_values('x')
    x = df['x'].to_numpy(float) + dodge
    d = df['d'].to_numpy(float)
    if categorical:
        first = True
        for i in range(len(x) - 1):
            if df['x'].iloc[i + 1] - df['x'].iloc[i] <= 1.0 + 1e-9:
                ax.plot(x[i:i + 2], d[i:i + 2], color=MODEL_COLORS[m], lw=1.0, label=label if first else None)
                first = False
        if first:
            ax.plot([], [], color=MODEL_COLORS[m], lw=1.0, label=label)
    else:
        ax.plot(x, d, color=MODEL_COLORS[m], lw=1.0, label=label)
    for xi, (_, r) in zip(x, df.iterrows()):
        if r['verdict'] != 'default':
            ax.plot([xi, xi], [r['lo'], r['hi']], color=MODEL_COLORS[m], lw=0.9)
        ax.plot(xi, r['d'], **marker_kw(r['verdict'], MODEL_COLORS[m]))


def _finish_delta(ax: Any, o: dict[str, Any]) -> None:
    ax.axhline(0, color=INK2, lw=0.6)
    if o.get('ylim'):
        ax.set_ylim(*o['ylim'])


def _draw_cats(ax: Any, c: pd.DataFrame, cats: list[tuple[str, float]], dodge: float, N: Numbers | None
               ) -> list[pd.DataFrame]:
    """Every panel model's paired differences at categorical positions (connected, with CIs)."""
    rows = []
    for j, m in enumerate(MODELS):
        df = _delta_points(c, m, cats)
        _draw_delta(ax, df, m, (j - 1) * dodge, _base_label(N, m) if N is not None else None, categorical=True)
        rows.append(df)
    return rows


def fig_P1_02_cutoff(**kw: Any) -> None:
    o = {**FIGURES['P1_02_cutoff'], **kw}
    N = _N()
    c = _comps()
    cut = [('lp5', 0), ('lp10', 1), ('base', 2), ('lp20', 3), ('lp_none', 4)]
    cut_names = ['5', '10', f"{N('lowpass_hz')}*", '20', 'none']
    order = [('lp_order2', 0), ('base', 1), ('lp_order8', 2), ('lp_causal', 3)]
    order_names = ['order 2', f"order {N('lowpass_order')}*", 'order 8', 'causal']
    xlabel_order = o['xlabel_order'] or f"Filter at {N('lowpass_hz')} Hz"
    fig, axes = figure(o['width'], o['height'], 1, 2, sharey=True, gridspec_kw={'width_ratios': o['width_ratios']})
    rows = [d.assign(panel='cut-off') for d in _draw_cats(axes[0], c, cut, o['dodge'], N)]
    rows += [d.assign(panel='order') for d in _draw_cats(axes[1], c, order, o['dodge'], None)]
    for ax, names, xl in ((axes[0], cut_names, o['xlabel']), (axes[1], order_names, xlabel_order)):
        _finish_delta(ax, o)
        ax.set_xticks(range(len(names)), names)
        ax.set_xlabel(xl)
    axes[0].set_ylabel(o['ylabel'])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3, fontsize=6, frameon=False,
               title='current accuracy', title_fontsize=6)
    L.save_fig(fig, 'P1_02_cutoff', data=_concat(rows, DELTA_COLUMNS + ('panel',)),
               caption=('Low-pass filter of the stream. Left: cut-off of the zero-phase Butterworth filter (order '
                        f"{N('lowpass_order')}); right: its order at {N('lowpass_hz')} Hz, and the causal (one-pass) "
                        'filter of the same design. Paired difference in leave-one-person-out accuracy against the '
                        'current pipeline (*), per panel model (legend: its current accuracy), with 95 % '
                        f'person-bootstrap CIs; {_note(N)}.'),
               decision='Keep or change the low-pass cut-off and order; whether a causal (production-friendly) filter '
                        'costs accuracy.')


def fig_P1_03_segmentation(**kw: Any) -> None:
    o = {**FIGURES['P1_03_segmentation'], **kw}
    N = _N()
    cj = L.read_json('p1_comparisons')
    c = pd.DataFrame(cj['comparisons'])
    rem = pd.DataFrame(cj.get('remedies', []))
    pts = [('base', 0.0), ('bjit50', 50.0), ('bjit100', 100.0), ('bjit200', 200.0)]
    fig, ax = figure(o['width'], o['height'])
    rows = []
    models = [m for m in MODELS if len(c[(c['model'] == m) & (c['option'] == 'bjit100')])]
    for j, m in enumerate(models):
        off = (j - (len(models) - 1) / 2) * o['dodge']
        df = _delta_points(c, m, pts)
        _draw_delta(ax, df, m, off, _base_label(N, m))
        rows.append(df)
        e = _delta_points(c, m, [('onset_late100', 100.0)])
        if len(e):
            r = e.iloc[0]
            xx = 100 + off - o['side']
            ax.plot([xx, xx], [r['lo'], r['hi']], color=MODEL_COLORS[m], lw=0.8)
            st = marker_kw(r['verdict'], MODEL_COLORS[m])
            st['marker'] = '^'
            ax.plot(xx, r['d'], **st)
            rows.append(e)
        e = _delta_points(c, m, [('bjit100_trained', 100.0)])
        rr = rem[rem['name'] == f'p1_{m}_bjit100_trained'] if len(rem) else rem
        if len(e):
            r = e.iloc[0]
            pair_v = rr['verdict'].iloc[0] if len(rr) else 'no difference'
            xx = 100 + off + o['side']
            ax.plot([xx, xx], [r['lo'], r['hi']], color=MODEL_COLORS[m], lw=0.8)
            st = marker_kw(pair_v, MODEL_COLORS[m])
            st['marker'] = 'D'
            st['ms'] = st.get('ms', 3.2) * 0.85
            ax.plot(xx, r['d'], **st)
            rows.append(e.assign(verdict=f'remedy pair: {pair_v}'))
    _finish_delta(ax, o)
    ax.set_xticks([J for _, J in pts])
    ax.set_xlabel(o['xlabel'])
    ax.set_ylabel(o['ylabel'])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, handlelength=1.2, frameon=False,
              title='trained on oracle bounds (current accuracy)', title_fontsize=6)
    ax.text(0.02, 0.03, 'triangle: onset 100 ms late\ndiamond: trained with jittered copies, tested at $\\pm$100 ms',
            transform=ax.transAxes, fontsize=5.5, color=INK2, va='bottom')
    L.save_fig(fig, 'P1_03_segmentation', data=_concat(rows),
               caption=('Robustness to segmentation errors (a cost a production detector must stay within, not an '
                        'option): paired accuracy difference of a model trained on the oracle bounds when the approach '
                        'bounds it is given are moved by independent uniform errors of up to $\\pm J$ at the onset and '
                        'at the end; triangles: the onset 100 ms late. Fills follow the robustness family '
                        f"({N('fam_robustness_size')} comparisons). Diamonds: the model trained with jittered copies "
                        'too, tested at $\\pm 100$ ms; it is judged only against the oracle-trained model at the same '
                        'bounds (the remedy pair), and its fill shows that verdict. 95 % person-bootstrap CIs; '
                        f'{_note(N)}.'),
               decision='How precise a production detector must be, and whether training with bound jitter helps.')


def fig_P1_04_sampling(**kw: Any) -> None:
    o = {**FIGURES['P1_04_sampling'], **kw}
    N = _N()
    c = _comps()
    fig, axes = figure(o['width'], o['height'], 1, 2, sharey=True)
    rows = []
    rates = [('base', C.FS_HZ), ('fs30', 30.0), ('fs20', 20.0)]          # categorical positions 0, 1, 2
    panels = [(axes[0], [(opt, float(i)) for i, (opt, _) in enumerate(rates)],
               [f"{N('fs_hz')}*" if opt == 'base' else f'{v:g}' for opt, v in rates], 'Sampling rate (Hz)',
               o['dodge_rate']),
              (axes[1], [('base', 0.0), ('tjit2', 2.0), ('tjit5', 5.0), ('tjit10', 10.0)], None,
               'Sample-time jitter SD (ms)', o['dodge_jitter'])]
    for ax, pts, ticklabels, xl, dodge in panels:
        for j, m in enumerate(MODELS):
            df = _delta_points(c, m, pts)
            _draw_delta(ax, df, m, (j - 1) * dodge, _base_label(N, m) if ax is axes[0] else None,
                        categorical=ax is axes[0])
            if len(df):
                rows.append(df.assign(panel=xl))
        _finish_delta(ax, o)
        ticks = sorted({v for _, v in pts})
        ax.set_xticks(ticks, ticklabels if ticklabels else [f'{t:g}*' if t == 0 else f'{t:g}' for t in ticks])
        ax.set_xlabel(xl)
    axes[0].set_ylabel(o['ylabel'])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3, fontsize=6, frameon=False,
               title='current accuracy', title_fontsize=6)
    L.save_fig(fig, 'P1_04_sampling', data=_concat(rows, DELTA_COLUMNS + ('panel',)),
               caption=('Robustness to the recording: paired accuracy difference against the current pipeline (*) when '
                        'every second or third raw sample is kept (30 and 20 Hz; the low-pass is capped at 0.4 times the '
                        'rate) and when Gaussian noise is added to the raw sample times before resampling. 95 % '
                        f'person-bootstrap CIs; {_note(N)}. A CI that includes 0 bounds the cost by its lower end; it '
                        'does not show equivalence.'),
               decision='Whether the predictor needs the native rate and exact timestamps, or tolerates slower browsers.')


# ───────────────────────────── P1_05: normalisation ─────────────────────────────


def fig_P1_05_normalisation(**kw: Any) -> None:
    o = {**FIGURES['P1_05_normalisation'], **kw}
    N = _N()
    c = _comps()
    c = c[(c['report_family'] == 'normalisation') & (c['mult_family'] == 'options')]
    opts = [x for x in ('norm_zscore', 'norm_robust', 'norm_trial_peak') if x in set(c['option'])]
    fig, axes = figure(o['width'], o['height'], 1, 2, sharey=True)
    for ax, (col, lo, hi, xl, scale, vcol) in zip(axes, (
            ('d_acc', 'd_acc_lo', 'd_acc_hi', 'Accuracy difference (points)', 100.0, 'verdict'),
            ('d_logloss', 'd_logloss_lo', 'd_logloss_hi', 'Log-loss difference (per person)', 1.0, 'logloss_verdict'))):
        ax.axvline(0, color=INK2, lw=0.6)
        for j, m in enumerate(MODELS):
            cm = c[c['model'] == m]
            for i, opt in enumerate(opts):
                g = cm[cm['option'] == opt]
                if not len(g):
                    continue
                g = g.iloc[0]
                yy = i + (j - 1) * 0.2
                ax.plot([scale * g[lo], scale * g[hi]], [yy, yy], color=MODEL_COLORS[m], lw=0.9)
                ax.plot(scale * g[col], yy, **marker_kw(g[vcol], MODEL_COLORS[m]),
                        label=SHORT[m] if (i == 0 and ax is axes[0]) else None)
        ax.set_xlabel(xl)
        ax.grid(axis='y', visible=False)
    if o['xlim_acc']:
        axes[0].set_xlim(*o['xlim_acc'])
    if o['xlim_ll']:
        axes[1].set_xlim(*o['xlim_ll'])
    axes[0].set_yticks(range(len(opts)), [LB.option_label(x) for x in opts])
    axes[0].set_ylim(len(opts) - 0.5, -0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker='o', ms=3.2, lw=0.9, label=SHORT[m]) for m in MODELS]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, fontsize=6, frameon=False, handlelength=1.2)
    L.save_fig(fig, 'P1_05_normalisation', data=c[['model', 'option', 'd_acc', 'd_acc_lo', 'd_acc_hi', 'q_bh', 'd_logloss',
                                                   'd_logloss_lo', 'd_logloss_hi', 'verdict', 'logloss_verdict']],
               caption=('Normalisation of the input channels against none (physical units), per panel model: paired '
                        'accuracy and mean per-person log-loss differences with 95 % person-bootstrap CIs. Scalers are '
                        'fitted on the mirror-augmented training rows of each fold; the per-trial peak scaling divides '
                        'each channel of a trial by its own peak. The logistic model standardises its own features, so a '
                        f'per-channel affine scaling leaves it unchanged (open squares). {_note(N)[0].upper()}'
                        f'{_note(N)[1:]}.'),
               decision='Which input normalisation the pipeline uses.')


# ───────────────────────────── P1_06: reliability ─────────────────────────────


REL_COLUMNS = ('model', 'calibrator', 'bin', 'n', 'p_mean', 'y_mean', 'y_lo', 'y_hi')


def fig_P1_06_reliability(**kw: Any) -> None:
    o = {**FIGURES['P1_06_reliability'], **kw}
    N = _N()
    cal = L.read_json('p1_calibration')
    models = [m for m in LB.CALIB_MODELS if f'{m}:none' in cal['reliability']]
    fig, axes = figure(o['width'], o['height'], 1, max(len(models), 1))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, m in zip(axes, models):
        show = ['none'] + (['auto'] if f'{m}:auto' in cal['reliability'] else [])
        ax.plot(o['lim'], o['lim'], color=MUTED, lw=0.7, ls='--')
        for j, (meth, col, mk) in enumerate(zip(show, (INK, MODEL_COLORS[m]), ('o', 's'))):
            rel = pd.DataFrame(cal['reliability'][f'{m}:{meth}'])
            k = f'cal_{m}_{meth}'
            off = (j - (len(show) - 1) / 2) * o['dodge']
            ax.errorbar(rel['p_mean'] + off, rel['y_mean'], yerr=[rel['y_mean'] - rel['y_lo'], rel['y_hi'] - rel['y_mean']],
                        color=col, marker=mk, ms=3, lw=0.9, capsize=0,
                        label=(f"{LB.CAL_LABELS.get(meth, meth)}: slope {N.est(k + '_slope')}\n"
                               f"ECE {N(k + '_ece')} (null 95th pct. {N(k + '_ece_null_q95')})"))
            rows += [{'model': m, 'calibrator': meth, **r} for r in rel.to_dict(orient='records')]
        ax.set_xlim(*o['lim'])
        ax.set_ylim(*o['lim'])
        ax.set_xticks(o['ticks'])
        ax.set_yticks(o['ticks'])
        ax.set_aspect('equal')
        ax.set_xlabel('Predicted P(right) (bin mean)')
        ax.set_title(MODEL_LABELS[m], fontsize=7)
        ax.legend(loc='upper left', fontsize=5.5, handlelength=1.2, borderaxespad=0.3)
    axes[0].set_ylabel('Observed share of right hands')
    cands = ', '.join('none' if x == 'none' else LB.CAL_LABELS.get(x, x)
                      for x in cal.get('settings', {}).get('auto_candidates', []))
    L.save_fig(fig, 'P1_06_reliability', data=pd.DataFrame(rows, columns=list(REL_COLUMNS)),
               caption=(f"Reliability of P(right) ({N('ece_bins')} equal-mass bins; bars: 95 % person-bootstrap band "
                        'of the observed share of right hands) before calibration and after the calibrator chosen inside '
                        f'every training fold (nested: {cands}; strictly increasing maps, so the ranking within a fold '
                        "is kept; chosen by grouped cross-validation on the training persons' cross-fitted "
                        'predictions); leave-one-person-out predictions. Legend: calibration slope (logistic regression '
                        'of the hand on logit P; 1 = calibrated, below 1 = over-confident) with its person-bootstrap '
                        '95 % CI, and the ECE with the 95th percentile of the ECE that perfectly calibrated predictions '
                        'would show on the same bins (null 95th pct.).'),
               decision='Whether the panel models need a calibration step, and which one.')


# ───────────────────────────── P1_07: multi-attempt aggregates ─────────────────────────────


AGG_FIG_COLUMNS = ('model', 'k', 'rule', 'n', 'n_persons', 'accuracy', 'log_loss', 'log_loss_lo', 'log_loss_hi', 'ece',
                   'slope', 'slope_lo', 'slope_hi')


def fig_P1_07_aggregates(**kw: Any) -> None:
    o = {**FIGURES['P1_07_aggregates'], **kw}
    N = _N()
    cal = L.read_json('p1_calibration')
    m = o['model'] if o['model'] in cal.get('aggregates', {}) else next(iter(cal.get('aggregates', {})), None)
    fig, axes = figure(o['width'], o['height'], 1, 2)
    rule_style = {'mean_prob': (INK, 'o'), 'mean_logit': (MUTED, 's'), 'sum_logit': ('#eb6834', '^'),
                  'sum_logit_T': ('#2a78d6', 'D')}
    rows: list[dict[str, Any]] = []
    if m is not None:
        a = pd.DataFrame(cal['aggregates'][m]['rows'])
        ks = o['ks'] or sorted(a['k'].unique())
        a = a[a['k'].isin(ks)]
        for ax, col, yl, ci in ((axes[0], 'ece', 'ECE (no CI)', False), (axes[1], 'log_loss', 'Log loss', True)):
            for rule, (cc, mk) in rule_style.items():
                g = a[a['rule'] == rule].sort_values('k')
                lab = LB.RULE_LABELS[rule].replace('T_k', '$T_k$')
                if ci:
                    ax.errorbar(g['k'], g[col], yerr=[g[col] - g[f'{col}_lo'], g[f'{col}_hi'] - g[col]], color=cc,
                                marker=mk, ms=3, lw=0.9, capsize=0, label=lab)
                else:
                    ax.plot(g['k'], g[col], color=cc, marker=mk, ms=3, lw=0.9, label=lab)
            ax.set_xticks(ks)
            ax.set_xlabel('Attempts aggregated, $k$')
            ax.set_ylabel(yl)
        fig.legend(*axes[1].get_legend_handles_labels(), loc='outside upper center', ncol=4, fontsize=6,
                   frameon=False, handlelength=1.2, title=MODEL_LABELS[m], title_fontsize=7)
        rows = a.assign(model=m).to_dict(orient='records')
    np_ = N(f'agg_{m}_n_persons') if m is not None else '--'
    L.save_fig(fig, 'P1_07_aggregates', data=pd.DataFrame(rows, columns=list(AGG_FIG_COLUMNS)),
               caption=('Calibration of the P(right) of k attempts of one person and hand combined (random k-subsets): '
                        'mean of the probabilities, mean of the logits, sum of the logits (the rule for independent '
                        'attempts) and that sum divided by a temperature $T_k$ fitted on the training persons of each '
                        f'fold. The same {np_} persons at every k (those with at least {N("k_max")} trials of each hand); '
                        'every person and hand weighs the same (also at k = 1, so k = 1 differs slightly from the '
                        'trial-level metrics). Log loss with 95 % person-bootstrap CIs; the ECE has no CI (its percentile '
                        'bootstrap is biased). Paired differences between the rules are in the aggregates table.'),
               decision='How the inference app should combine several attempts into one calibrated probability.')


# ───────────────────────────── P1_08: platform transfer ─────────────────────────────


PLATFORM_TRAININGS = (('pooled', 'o', 'full'), ('same', 's', 'full'), ('same_ctl', 's', 'none'),
                      ('cross', '^', 'full'), ('cross_ctl', '^', 'none'))
TRAINING_LABELS = {'pooled': 'all other persons (pooled)', 'same': 'same platform only',
                   'same_ctl': 'as many random persons as same platform', 'cross': 'other platform only',
                   'cross_ctl': 'as many random persons as other platform'}


def fig_P1_08_platform(**kw: Any) -> None:
    o = {**FIGURES['P1_08_platform'], **kw}
    N = _N()
    s = pd.DataFrame(L.read_json('p1_subsets')['rows'])
    s = s[s['kind'] == 'platform'].copy()
    models = [m for m in MODELS if len(s[(s['model'] == m) & (s['training_key'] != 'pooled')])]
    s = s[s['model'].isin(models)]
    fig, axes = figure(o['width'], o['height'], 1, max(len(models), 1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        sm = s[s['model'] == m]
        for i, pf in enumerate(('android', 'ios')):
            for j, (tk, mk, fill) in enumerate(PLATFORM_TRAININGS):
                g = sm[(sm['platform'] == pf) & (sm['training_key'] == tk)]
                if not len(g):
                    continue
                g = g.iloc[0]
                x = i + (j - (len(PLATFORM_TRAININGS) - 1) / 2) * o['dodge']
                col = PLATFORM_COLORS[pf]
                if g['lo'] is not None and np.isfinite(g['lo']):
                    ax.plot([x, x], [g['lo'], g['hi']], color=col, lw=0.9)
                ax.plot(x, g['accuracy'], marker=mk, ms=3.6, mec=col, mfc=col if fill == 'full' else 'white', mew=0.9,
                        ls='none')
        ax.set_xticks([0, 1], [PLATFORM_LABELS['android'], PLATFORM_LABELS['ios']])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(MODEL_LABELS[m], fontsize=7)
        if o['ylim']:
            ax.set_ylim(*o['ylim'])
        ax.yaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    axes[0].set_ylabel('Accuracy on the test platform')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=INK2, marker=mk, ms=3.6, lw=0, mfc=INK2 if fill == 'full' else 'white',
                      label=f'trained on: {TRAINING_LABELS[tk]}') for tk, mk, fill in PLATFORM_TRAININGS]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, fontsize=6, frameon=False, handlelength=1.0)
    L.save_fig(fig, 'P1_08_platform', data=s.drop(columns=[c for c in ('flag', 'flagged') if c in s.columns]),
               caption=('Platform transfer: accuracy on the persons of each platform when the model is trained on all '
                        'other persons (pooled), on the other persons of the same platform only, or on the persons of '
                        'the other platform only, each next to its size-matched control (open markers: as many training '
                        'persons drawn at random from both platforms). Training on one platform also shrinks the '
                        'training set, so only the difference from the control isolates the platform; the paired '
                        'option-against-control comparisons are in the pairs table (leave-one-person-out; 95 % '
                        'person-bootstrap CIs).'),
               decision='Whether one platform can train the other, i.e. whether pooling platforms is safe.')


# ───────────────────────────── P1_09: excluded and flagged trials ─────────────────────────────


EXCL_ROWS: tuple[tuple[str, str], ...] = (
    ('all', 'all primary trials'),
    ('e4d_all_linear', 'E4d, all (linear imputation)'),
    ('e4d_margin_linear', 'E4d, gap only in the margins'),
    ('e4d_inapp_linear', 'E4d, gap inside the approach: linear'),
    ('e4d_inapp_cubic', 'E4d, gap inside the approach: cubic'),
    ('e4d_inapp_hold', 'E4d, gap inside the approach: hold'),
    ('e4d_inapp_trained', 'E4d, gap inside: E4d also in training'),
    ('e4bc_base', 'E4b/E4c trials (current model)'),
    ('e4bc_trained', 'E4b/E4c trials (also in training)'),
    ('tight_fail', 'failing thresholds twice as strict'),
    ('tight_pass', 'passing thresholds twice as strict'),
    ('flag_grip_mismatch', 'flagged: grip mismatch'),
    ('flag_false_start', 'flagged: false start'),
    ('flag_seg_disagree_100ms', 'flagged: segmenters disagree > 100 ms'),
    ('flag_repeated_samples', 'flagged: repeated samples dropped'),
    ('flag_clip_candidate', 'flagged: possible sensor clipping'),
    ('swap', 'swap-signature blocks'),
)


def fig_P1_09_exclusions(**kw: Any) -> None:
    o = {**FIGURES['P1_09_exclusions'], **kw}
    N = _N()
    s = pd.DataFrame(L.read_json('p1_subsets')['rows'])
    s['row'] = s['key'].str.split('_', n=1).str[1]
    keys = [k for k, _ in EXCL_ROWS if k in set(s['row'])]
    lab = dict(EXCL_ROWS)
    height = o['head_in'] + o['row_in'] * len(keys)
    fig, ax = figure(o['width'], height)
    models = [m for m in MODELS if m in set(s['model'])]
    plotted = []
    for j, m in enumerate(models):
        sm = s[s['model'] == m].set_index('row')
        for i, k in enumerate(keys):
            if k not in sm.index:
                continue
            g = sm.loc[k]
            if g['accuracy'] is None or not np.isfinite(g['accuracy']):
                continue
            yy = i + (j - (len(models) - 1) / 2) * 0.22
            has_ci = bool(g['ci'])
            if has_ci:
                ax.plot([g['lo'], g['hi']], [yy, yy], color=MODEL_COLORS[m], lw=0.9)
            ax.plot(g['accuracy'], yy, 'o', ms=3, mec=MODEL_COLORS[m], mfc=MODEL_COLORS[m] if has_ci else 'white',
                    mew=0.9, ls='none')
            plotted.append({'model': m, 'row': k, 'label': lab[k], 'n': g['n'], 'n_persons': g['n_persons'],
                            'accuracy': g['accuracy'], 'lo': g['lo'], 'hi': g['hi'], 'status': g['status']})
    first = {}
    for r in plotted:
        first.setdefault(r['row'], r)
    for i, k in enumerate(keys):
        r = first.get(k)
        if r is not None:
            ax.text(1.01, i, f"n={int(r['n'])}, {int(r['n_persons'])} p.", transform=ax.get_yaxis_transform(),
                    fontsize=5, va='center', color=INK2)
    for m in models:
        if N.ok(f'{m}_accuracy'):
            ax.axvline(N.value(f'{m}_accuracy'), color=MODEL_COLORS[m], lw=0.5, ls=':')
    ax.set_yticks(range(len(keys)), [lab[k] for k in keys], fontsize=6)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xlim(*o['xlim'])
    ax.set_xticks(o['xticks'])
    ax.xaxis.set_major_formatter(lambda v, _: f'{100 * v:.0f} %')
    ax.set_xlabel('Accuracy')
    ax.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], marker='o', ms=3, lw=0.9, label=SHORT[m]) for m in models]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, fontsize=6, frameon=False)
    L.save_fig(fig, 'P1_09_exclusions',
               data=pd.DataFrame(plotted, columns=['model', 'row', 'label', 'n', 'n_persons', 'accuracy', 'lo', 'hi',
                                                   'status']),
               caption=('Accuracy on the trials that an exclusion rule removes, scored by the current models (trained '
                        'on exactly the primary training set) unless stated: E4d (an unbridged sampling gap near the '
                        'movement) split by where the gap falls, with three imputations for the gaps inside the '
                        'approach; E4b/E4c (too short or too little rotation); trials that thresholds twice as strict '
                        'would remove; flagged trials; the swap-signature blocks. Dotted lines: all primary trials. '
                        f"95 % person-bootstrap CIs only for subsets of at least {N('min_persons_ci')} persons (open "
                        'markers: fewer, no CI); n = trials, p. = persons.'),
               decision='Whether the exclusion thresholds and flags remove trials the models cannot score.')


# ───────────────────────────── P1_10: delay of a causal filter ─────────────────────────────


def fig_P1_10_filter_delay(**kw: Any) -> None:
    o = {**FIGURES['P1_10_filter_delay'], **kw}
    N = _N()
    order = o['order'] or C.LOWPASS_ORDER
    fd = L.read_json('p1_filter_delay')
    f = np.asarray(fd['freq_hz'])
    fig, ax = figure(o['width'], o['height'])
    rows = []
    shades = dict(zip(sorted(o['cutoffs']), ('#86b6ef', '#2a78d6', '#0d366b')))
    for c in fd['curves']:
        if c['order'] != order or c['cutoff_hz'] not in o['cutoffs']:
            continue
        sel = f <= o['fmax']
        cur = '*' if c['cutoff_hz'] == C.LOWPASS_HZ else ''
        ax.plot(f[sel], np.asarray(c['delay_ms'])[sel], color=shades.get(c['cutoff_hz'], INK), lw=1.1,
                label=f"causal, {c['cutoff_hz']:g} Hz{cur}")
        rows += [{'cutoff_hz': c['cutoff_hz'], 'order': c['order'], 'freq_hz': float(x), 'delay_ms': float(y)}
                 for x, y in zip(f[sel], np.asarray(c['delay_ms'])[sel])]
    ax.axhline(0, color=INK, lw=1.0, ls='--', label='zero phase (current)')
    ax.set_xlabel(o['xlabel'])
    ax.set_ylabel(o['ylabel'])
    ax.set_xlim(0, o['fmax'])
    ax.legend(fontsize=6, loc='upper left', handlelength=1.4)
    L.save_fig(fig, 'P1_10_filter_delay', data=pd.DataFrame(rows, columns=['cutoff_hz', 'order', 'freq_hz', 'delay_ms']),
               caption=(f'Group delay of causal Butterworth low-pass filters (order {order}) at the stream rate of '
                        f"{N('fs_hz')} Hz; the zero-phase filter of the pipeline has none. The causal version of the "
                        f"current filter ({N('lowpass_hz')} Hz, *) delays a 1-Hz movement by about "
                        f"{N('causal_delay_ms')} ms."),
               decision='What a causal (online) filter would cost in latency.')


FIGURE_FUNCTIONS = {
    'P1_01_forest': fig_P1_01_forest, 'P1_02_cutoff': fig_P1_02_cutoff, 'P1_03_segmentation': fig_P1_03_segmentation,
    'P1_04_sampling': fig_P1_04_sampling, 'P1_05_normalisation': fig_P1_05_normalisation,
    'P1_06_reliability': fig_P1_06_reliability, 'P1_07_aggregates': fig_P1_07_aggregates,
    'P1_08_platform': fig_P1_08_platform, 'P1_09_exclusions': fig_P1_09_exclusions,
    'P1_10_filter_delay': fig_P1_10_filter_delay,
}


def draw(only: list[str] | None = None) -> None:
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()


__all__ = ['draw', 'FIGURES', 'FIGURE_FUNCTIONS', 'marker_kw', 'rkey']
