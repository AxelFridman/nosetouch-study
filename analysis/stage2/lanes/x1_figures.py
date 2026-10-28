"""Figures of lane x1 (the PD01 case study), drawn from saved files only - nothing here fits or recomputes.

Edit ``FIGURES`` (sizes, limits, scales, which characteristics a panel shows) or ``COLORS``, then
    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes x1 [--only X1_03_spectra]

Captions and decision sentences come from ``x1_texts`` (numbers entries only), so the summary lists the same texts
without drawing. Colours: the reference persons are grey context marks; the exploratory person is one dark amber
(#b36b00: contrast >= 3:1 on white, CVD dE >= 15.9 from the dark-grey reference median by the dataviz validator),
always with a dark edge and a direct label, never a hand colour (the hand colours mean left / right everywhere else in
the report).
"""
from __future__ import annotations

import math
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, NullLocator

from stage2.pipeline import Lane, selected
from stage2.plotting import AXIS, INK, INK2, MUTED, figure, plain_log_ticks
from stage2.lanes import x1_report as XR
from stage2.lanes import x1_stats as XS
from stage2.lanes import x1_texts as XT

L = Lane('x1')

COLORS = {'case': '#b36b00', 'case_edge': INK, 'ref': '#c3c2b7', 'ref_edge': MUTED, 'ref_median': '#52514e',
          'band': '#e9e8e2', 'band_text': INK2, 'shade': '#f3f2ee', 'linear': '#8c8a83'}
CASE_MARKER = dict(marker='D', s=34, color=COLORS['case'], edgecolors=COLORS['case_edge'], linewidths=0.6, zorder=6)

#: Presentation defaults per figure (one-line edits; every function also takes them as keyword arguments).
FIGURES: dict[str, dict[str, Any]] = {
    'X1_01_example_traces': dict(width='full', height=4.3, xlim=None, pad_s=(1.0, 1.2), headroom=0.18,
                                 rows=(('gyro_mag', 'Angular speed |ω| (deg/s)'),
                                       ('yaw_rate', 'Yaw rate ω·u (deg/s)'),
                                       ('acc_mag', 'Linear acceleration |a| (m/s²)'))),
    'X1_02_preparation': dict(width='full', height=4.1, ncols=6, title_wrap=16, title_size=6.5,
                              keys=XT.STRIP_KEYS['X1_02_preparation'],
                              log=('hold_wait_s', 'pre_rms_3_7', 'pre_rms_7_12', 'pre_rms_05_3', 'pre_sway_gyro',
                                   'nose_rms_3_7', 'nose_rms_7_12')),
    'X1_03_spectra': dict(width='full', height=2.5, xlim=(0.5, 15.0), ylim=None,
                          windows=(('pre', 'Hold before the movement'), ('approach', 'Reach (natural speed)'),
                                   ('nose', 'At the nose')),
                          bands=((3.0, 7.0, '3–7 Hz'), (7.0, 12.0, '7–12 Hz'))),
    'X1_04_movement': dict(width='full', height=4.1, ncols=6, title_wrap=16, title_size=6.5,
                           keys=XT.STRIP_KEYS['X1_04_movement'],
                           log=('duration_s', 'active_s', 'settle_s', 'peak_speed_mv', 'peak_acc', 'touch_acc',
                                'touch_s', 'rotation_total_mv', 'tilt_change', 'app_rms_7_12')),
    'X1_05_speed_scaling': dict(width='full', height=2.4,
                                keys=(('duration_s', 'Approach duration (s)'), ('active_s', 'Active movement time (s)'),
                                      ('peak_speed_mv', 'Peak angular speed, < 3 Hz (deg/s)')),
                                speeds=(('slow', 'Slow'), ('natural', 'Natural'), ('fast', 'Fast')), log=True),
    'X1_06_repeats': dict(width='full', height=2.4, stratum='left_upright_natural',
                          keys=(('duration_s', 'Approach duration (s)'), ('peak_speed_mv', 'Peak angular speed, < 3 Hz (deg/s)'),
                                ('app_rms_7_12', 'Reach 7–12 Hz band RMS (deg/s)')), log=True),
    'X1_07_hand_signature': dict(width='full', height=2.8, ylim=None, channel='yaw_cum_hand',
                                 ylabel='Cumulative yaw, hand-consistent (deg)'),
    'X1_08_zscore_summary': dict(width='full', row_h=0.118, null_h=1.25, zlim=(-6.0, 6.0), label_size=5.8,
                                 null_set='prespecified', null_variant='like_for_like'),
    'X1_09_multivariate': dict(width='full', height=2.9, label_top=6, width_ratios=(1.15, 0.85, 1.25),
                               loco_label_chars=34),
}

# ───────────────────────────── saved inputs ─────────────────────────────


_CACHE: dict[tuple[str, float], Any] = {}


def _cached(path: Any, loader: Any) -> Any:
    """Read a saved file once per modification time (figures re-read the same tables many times)."""
    key = (str(path), path.stat().st_mtime if path.exists() else -1.0)
    if key not in _CACHE:
        _CACHE[key] = loader()
    return _CACHE[key]


def _characteristics() -> dict[str, Any]:
    return _cached(L.res_dir / 'x1_characteristics.json', lambda: L.read_json('x1_characteristics', folder='results'))


def _main() -> pd.DataFrame:
    return pd.DataFrame(_characteristics()['rows'])


def _case() -> str:
    return str(_characteristics()['case'])


def _nums() -> XR.Nums:
    return XR.current_numbers()[1]


def _csv(name: str) -> pd.DataFrame:
    p = L.tab_dir / f'{name}.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p}: run python -m stage2.pipeline compute --lanes x1 first')

    def load() -> pd.DataFrame:
        df = pd.read_csv(p)
        if 'person_id' in df.columns:
            df['person_id'] = df['person_id'].astype(str)
        return df
    return _cached(p, load).copy()


def _person_values(key: str, stratum: str | None = None, variant: str = 'main') -> tuple[pd.DataFrame, pd.Series]:
    """(reference person values, the case's row) of one characteristic in a stratum (default: its own)."""
    pv = _csv('x1_person_values')
    s = stratum or XS.BY_KEY[key].stratum
    sub = pv[(pv['key'] == key) & (pv['stratum'] == s) & (pv['variant'] == variant)]
    ref = sub[(sub['group'] == 'reference') & sub['value'].notna()]
    case = sub[sub['group'] == 'exploratory']
    return ref, (case.iloc[0] if len(case) else pd.Series({'value': math.nan, 'n_trials': 0}))


def _unit(key: str) -> str:
    return XS.BY_KEY[key].unit.replace('m/s$^2$', 'm/s²')


# ───────────────────────────── shared panels ─────────────────────────────


def _strip_panel(ax: Any, key: str, N: XR.Nums, *, log: bool, case_trials: np.ndarray | None, wrap: int,
                 title_size: float) -> pd.DataFrame:
    """Reference persons as grey dots (deterministic jitter), their median and IQR, the case as a diamond (and its
    trials as small ticks). Returns the plotted data."""
    ref, case = _person_values(key)
    y = ref['value'].to_numpy(dtype=float)
    n = y.size
    rng = np.random.default_rng(len(key))                     # deterministic jitter per panel
    xj = rng.uniform(-0.18, 0.18, n)
    ax.scatter(xj, y, s=12, color=COLORS['ref'], edgecolors=COLORS['ref_edge'], linewidths=0.4, zorder=3)
    if n:
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        ax.add_patch(Rectangle((-0.3, q1), 0.6, q3 - q1, facecolor='none', edgecolor=COLORS['ref_median'], lw=0.6,
                               zorder=2))
        ax.plot([-0.3, 0.3], [med, med], color=COLORS['ref_median'], lw=1.0, zorder=4)
    cv = float(case['value']) if pd.notna(case['value']) else math.nan
    if case_trials is not None and len(case_trials):
        ax.scatter(np.full(len(case_trials), 0.75), case_trials, marker='_', s=40, color=COLORS['case'], lw=1.0,
                   zorder=5)
    if math.isfinite(cv):
        ax.scatter([0.75], [cv], **CASE_MARKER)
    ax.set_xlim(-0.55, 1.15)
    ax.set_xticks([0, 0.75], [f'n = {n}', f'{_case()}\n{N.rank(key, short=True)}'])
    ax.tick_params(axis='x', length=0, labelsize=6.2)
    ax.grid(axis='x', visible=False)
    if log and n and np.all(y > 0) and (not math.isfinite(cv) or cv > 0):
        ax.set_yscale('log')
        allv = np.r_[y, cv, case_trials if case_trials is not None else []]
        allv = allv[np.isfinite(allv) & (allv > 0)]
        lo, hi = float(np.min(allv)), float(np.max(allv))
        plain_log_ticks(ax.yaxis, _log_ticks(lo, hi))
        ax.set_ylim(lo / 1.25, hi * 1.25)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    unit = _unit(key)
    lab = XS.BY_KEY[key].label + (' *' if XS.BY_KEY[key].post_hoc else '') + (f' ({unit})' if unit else '')
    ax.set_title(textwrap.fill(lab, wrap), fontsize=title_size)
    data = pd.DataFrame({'key': key, 'person_id': ref['person_id'], 'group': 'reference', 'value': y})
    extra = pd.DataFrame({'key': [key], 'person_id': [_case()], 'group': ['exploratory'], 'value': [cv]})
    out = pd.concat([data, extra], ignore_index=True)
    if case_trials is not None and len(case_trials):
        out = pd.concat([out, pd.DataFrame({'key': key, 'person_id': _case(), 'group': 'exploratory_trial',
                                            'value': case_trials})], ignore_index=True)
    return out


def _log_ticks(lo: float, hi: float) -> list[float]:
    """Plain log ticks covering [lo, hi]: 1-2-5 steps, thinned to at most 5, at least 2."""
    lo, hi = max(lo, 1e-6), max(hi, 1e-6)
    for subs in ((1, 2, 5), (1, 3), (1,)):
        t = [v * 10.0 ** e for e in range(-4, 5) for v in subs if lo / 1.25 <= v * 10.0 ** e <= hi * 1.25]
        if 2 <= len(t) <= 5:
            return t
    fine = [v * 10.0 ** e for e in range(-4, 5) for v in (1, 1.5, 2, 3, 4, 5, 7) if lo / 1.25 <= v * 10.0 ** e <= hi * 1.25]
    return fine[:: max(1, len(fine) // 4)] if len(fine) >= 2 else [float(f'{lo:.2g}'), float(f'{hi:.2g}')]


def _case_trial_values(key: str) -> np.ndarray:
    t = _csv('x1_trial_characteristics')
    c = XS.BY_KEY[key]
    if c.level != 'trial':
        return np.array([])
    sub = t[(t['group'] == 'exploratory') & XS.STRATA[c.stratum].select(t)]
    return pd.to_numeric(sub[key], errors='coerce').dropna().to_numpy()


def _strip_figure(name: str, o: dict[str, Any]) -> None:
    N = _nums()
    keys = list(o['keys'])
    ncols = int(o['ncols'])
    nrows = int(math.ceil(len(keys) / ncols))
    fig, axes = figure(o['width'], o['height'], nrows=nrows, ncols=ncols)
    axes = np.atleast_1d(axes).ravel()
    data = []
    for ax, k in zip(axes, keys):
        data.append(_strip_panel(ax, k, N, log=k in o['log'], case_trials=_case_trial_values(k),
                                 wrap=int(o['title_wrap']), title_size=float(o['title_size'])))
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    tx = XT.texts(name)
    L.save_fig(fig, name, caption=tx['caption'], decision=tx['decision'], data=pd.concat(data, ignore_index=True))


# ───────────────────────────── figures ─────────────────────────────


def fig_X1_01_example_traces(**kw: Any) -> None:
    o = {**FIGURES['X1_01_example_traces'], **kw}
    tr = _csv('x1_traces')
    ev = L.read_json('x1_traces')
    tc = _csv('x1_trial_characteristics').set_index('trial_key')
    keys = [ev['selected_case'], ev['selected_reference']]
    fig, axes = figure(o['width'], o['height'], nrows=len(o['rows']), ncols=2, sharex=True, sharey='row')
    ends = [ev['events'][k]['end_s'] for k in keys]
    xlim = o['xlim'] or (-o['pad_s'][0], max(ends) + o['pad_s'][1])
    data = []
    for j, k in enumerate(keys):
        g = tr[tr['trial_key'] == k]
        e = ev['events'][k]
        touch = tc.loc[k, 'touch_s'] if k in tc.index else math.nan
        who = e['person_id']
        col = COLORS['case'] if e['group'] == 'exploratory' else COLORS['ref_median']
        for i, (ch, lab) in enumerate(o['rows']):
            ax = axes[i, j]
            ax.axvspan(0.0, e['end_s'], color=COLORS['shade'], zorder=0, lw=0)
            ax.plot(g['t_s'], g[ch], color=col, lw=0.9)
            if ch == 'yaw_rate':
                ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
            if touch is not None and math.isfinite(touch):
                ax.axvline(touch, color=INK2, lw=0.7, ls=(0, (3, 2)))
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xlim(*xlim)
        title = (f'{who}: {e["hand"]} hand, upright, {e["speed"]}' + (' (typical matched trial)' if e['group'] ==
                                                                        'reference' else ' (median-like trial)'))
        axes[0, j].set_title(title, fontsize=7.5)
        if touch is not None and math.isfinite(touch):           # labels where both traces stay low
            axes[-1, j].text(touch, 0.97, ' first touch', transform=axes[-1, j].get_xaxis_transform(), fontsize=6,
                             color=INK2, va='top', ha='left')
        axes[0, j].text(0.03, 0.97, 'approach', transform=axes[0, j].get_xaxis_transform(), fontsize=6,
                        color=INK2, va='top', ha='left')
        axes[-1, j].set_xlabel('Time from movement onset (s)')
        data.append(g.assign(touch_s=touch, end_s=e['end_s']))
    for i in range(len(o['rows'])):                      # headroom: the labels sit above the traces
        lo, hi = axes[i, 0].get_ylim()
        axes[i, 0].set_ylim(lo, hi + o['headroom'] * (hi - lo))
    tx = XT.texts('X1_01_example_traces')
    L.save_fig(fig, 'X1_01_example_traces', data=pd.concat(data, ignore_index=True)[
        ['group', 'person_id', 'trial_key', 't_s', 'gyro_mag', 'yaw_rate', 'acc_mag', 'touch_s', 'end_s']],
        caption=tx['caption'], decision=tx['decision'])


def fig_X1_02_preparation(**kw: Any) -> None:
    _strip_figure('X1_02_preparation', {**FIGURES['X1_02_preparation'], **kw})


def fig_X1_03_spectra(**kw: Any) -> None:
    o = {**FIGURES['X1_03_spectra'], **kw}
    sp = _csv('x1_spectra')
    wins = list(o['windows'])
    fig, axes = figure(o['width'], o['height'], ncols=len(wins), sharey=True)
    data = []
    for ax, (w, title) in zip(np.atleast_1d(axes), wins):
        for (lo, hi, lab), tint in zip(o['bands'], (COLORS['band'], COLORS['shade'])):
            ax.axvspan(lo, hi, color=tint, lw=0, zorder=0)
            for edge in (lo, hi):
                ax.axvline(edge, color=AXIS, lw=0.5, zorder=0)
            ax.text((lo + hi) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6,
                    color=COLORS['band_text'])
        ref = sp[(sp['window'] == w) & (sp['group'] == 'reference') & (sp['level'] == 'person')]
        piv = ref.pivot_table(index='f_hz', columns='person_id', values='psd')
        f = piv.index.to_numpy(dtype=float)
        m = (f >= o['xlim'][0]) & (f <= o['xlim'][1])
        q1, med, q3 = (np.nanpercentile(piv.to_numpy(), q, axis=1) for q in (25, 50, 75))
        ax.fill_between(f[m], q1[m], q3[m], color=COLORS['ref'], alpha=0.55, lw=0, zorder=1,
                        label=f'reference IQR (n={piv.shape[1]})')
        ax.plot(f[m], med[m], color=COLORS['ref_median'], lw=1.1, zorder=3, label='reference median')
        case = sp[(sp['window'] == w) & (sp['group'] == 'exploratory')]
        for k, g in case[case['level'] == 'trial'].groupby('trial_key'):
            g = g[(g['f_hz'] >= o['xlim'][0]) & (g['f_hz'] <= o['xlim'][1])]
            ax.plot(g['f_hz'], g['psd'], color=COLORS['case'], lw=0.5, alpha=0.55, zorder=4)
        cm = case[case['level'] == 'person']
        cm = cm[(cm['f_hz'] >= o['xlim'][0]) & (cm['f_hz'] <= o['xlim'][1])]
        ax.plot(cm['f_hz'], cm['psd'], color=COLORS['case'], lw=1.6, zorder=5, label=f'{_case()} (mean; trials thin)')
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(*o['xlim'])
        if o['ylim']:
            ax.set_ylim(*o['ylim'])
        ax.set_title(title, fontsize=7.5, pad=8)
        ax.set_xlabel('Frequency (Hz)')
        data.append(pd.DataFrame({'window': w, 'f_hz': f[m], 'ref_q1': q1[m], 'ref_median': med[m], 'ref_q3': q3[m]})
                    .merge(cm[['f_hz', 'psd']].rename(columns={'psd': 'case_mean'}), on='f_hz', how='left'))
    axes[0].set_ylabel('Gyroscope PSD ((deg/s)²/Hz)')
    axes[-1].legend(loc='upper right', fontsize=6, bbox_to_anchor=(1.0, 0.93))
    tx = XT.texts('X1_03_spectra')
    L.save_fig(fig, 'X1_03_spectra', data=pd.concat(data, ignore_index=True), caption=tx['caption'],
               decision=tx['decision'])


def fig_X1_04_movement(**kw: Any) -> None:
    _strip_figure('X1_04_movement', {**FIGURES['X1_04_movement'], **kw})


def _lines_panel(ax: Any, piv: pd.DataFrame, case: pd.Series, xlabels: list[str], ylabel: str, log: bool) -> None:
    x = np.arange(len(xlabels))
    for pid, row in piv.iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color=COLORS['ref'], lw=0.7, zorder=2)
    med = np.nanmedian(piv.to_numpy(dtype=float), axis=0)
    ax.plot(x, med, color=COLORS['ref_median'], lw=1.6, zorder=3, marker='o', ms=3, label='reference median')
    ax.plot(x, case.to_numpy(dtype=float), color=COLORS['case'], lw=1.4, zorder=5)
    ax.scatter(x, case.to_numpy(dtype=float), label=_case(), **CASE_MARKER)
    ax.set_xticks(x, xlabels)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.set_ylabel(ylabel)
    if log:
        vals = np.r_[piv.to_numpy(dtype=float).ravel(), case.to_numpy(dtype=float)]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ax.set_yscale('log')
        plain_log_ticks(ax.yaxis, _log_ticks(vals.min(), vals.max()))
        ax.set_ylim(vals.min() / 1.2, vals.max() * 1.2)


def _top_legend(fig: Any, ax: Any) -> None:
    """One legend above the panels (outside the data), from the handles of one panel."""
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc='outside upper center', ncol=len(lab), fontsize=6.5)


def fig_X1_05_speed_scaling(**kw: Any) -> None:
    o = {**FIGURES['X1_05_speed_scaling'], **kw}
    pv = _csv('x1_person_values')
    fig, axes = figure(o['width'], o['height'], ncols=len(o['keys']))
    data = []
    for ax, (key, ylabel) in zip(np.atleast_1d(axes), o['keys']):
        cols = {}
        for sp, _ in o['speeds']:
            s = pv[(pv['key'] == key) & (pv['stratum'] == f'left_upright_{sp}') & (pv['variant'] == 'main')]
            cols[sp] = s.set_index('person_id')['value']
        df = pd.DataFrame(cols)
        ref, case = df.drop(index=_case(), errors='ignore'), df.loc[_case()]
        ref = ref.dropna(how='all')
        _lines_panel(ax, ref, case, [lab for _, lab in o['speeds']], ylabel, o['log'])
        data.append(df.reset_index(names='person_id').melt(id_vars='person_id', var_name='speed', value_name='value')
                    .assign(key=key))
    _top_legend(fig, np.atleast_1d(axes)[0])
    tx = XT.texts('X1_05_speed_scaling')
    L.save_fig(fig, 'X1_05_speed_scaling', data=pd.concat(data, ignore_index=True), caption=tx['caption'],
               decision=tx['decision'])


def fig_X1_06_repeats(**kw: Any) -> None:
    o = {**FIGURES['X1_06_repeats'], **kw}
    t = _csv('x1_trial_characteristics')
    sub = XS.visit_order(t[XS.STRATA[o['stratum']].select(t)])
    k = int((sub['group'] == 'exploratory').sum())
    fig, axes = figure(o['width'], o['height'], ncols=len(o['keys']))
    data = []
    for ax, (key, ylabel) in zip(np.atleast_1d(axes), o['keys']):
        rows = {}
        for pid, g in sub.groupby('person_id', sort=True):
            vals = pd.to_numeric(g[key], errors='coerce').to_numpy()[:k]
            if len(vals) == k:
                rows[pid] = vals
        df = pd.DataFrame(rows, index=[f'{i + 1}' for i in range(k)]).T
        ref, case = df.drop(index=_case(), errors='ignore'), df.loc[_case()]
        _lines_panel(ax, ref, case, list(df.columns), ylabel, o['log'])
        ax.set_xlabel('Repeat (visit order)')
        data.append(df.reset_index(names='person_id').melt(id_vars='person_id', var_name='repeat', value_name='value')
                    .assign(key=key))
    _top_legend(fig, np.atleast_1d(axes)[0])
    tx = XT.texts('X1_06_repeats')
    L.save_fig(fig, 'X1_06_repeats', data=pd.concat(data, ignore_index=True), caption=tx['caption'],
               decision=tx['decision'])


def fig_X1_07_hand_signature(**kw: Any) -> None:
    o = {**FIGURES['X1_07_hand_signature'], **kw}
    N = _nums()
    yc = _csv('x1_yaw_curves')
    fig, axes = figure(o['width'], o['height'], ncols=3, gridspec_kw={'width_ratios': [1.0, 1.0, 1.0]})
    data = []
    col = o['channel']
    for ax, stratum in zip(axes[:2], ('left_upright_all', 'free_choice')):
        ref = yc[(yc['stratum'] == stratum) & (yc['group'] == 'reference')]
        key = 'person_id' if stratum == 'left_upright_all' else 'trial_key'
        piv = ref.pivot_table(index='phase', columns=key, values=col)
        ph = piv.index.to_numpy(dtype=float)
        for c in piv.columns:
            ax.plot(ph, piv[c], color=COLORS['ref'], lw=0.5, zorder=1)
        q1, med, q3 = (np.nanpercentile(piv.to_numpy(), q, axis=1) for q in (25, 50, 75))
        ax.fill_between(ph, q1, q3, color=COLORS['ref'], alpha=0.5, lw=0, zorder=2, label='reference IQR')
        ax.plot(ph, med, color=COLORS['ref_median'], lw=1.5, zorder=3, label='reference median')
        case = yc[(yc['stratum'] == stratum) & (yc['group'] == 'exploratory')]
        for (tk, aset), g in case.groupby(['trial_key', 'analysis_set']):
            ls = '-' if aset != 'natural_use_repeat' else (0, (3, 2))
            ax.plot(g['phase'], g[col], color=COLORS['case'], lw=0.7, ls=ls, alpha=0.8, zorder=4)
        first = case[case['analysis_set'] != 'natural_use_repeat']
        cm = first.groupby('phase')[col].mean()
        ax.plot(cm.index, cm.to_numpy(), color=COLORS['case'], lw=1.8, zorder=5,
                label=f'{_case()} (mean; trials thin)')
        ax.axhline(0, color=AXIS, lw=0.6, zorder=0)
        ax.axvline(0.5, color=AXIS, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.set_xlabel('Share of the approach')
        if stratum == 'left_upright_all':
            title = f'Left hand, upright, mirrored ({piv.shape[1]} persons)'
        else:
            hands = sorted(set(ref['hand'].astype(str)))
            title = f'First free trial ({piv.shape[1]} trials; {" / ".join(hands)} hand)'
        ax.set_title(title, fontsize=7.2)
        if o['ylim']:
            ax.set_ylim(*o['ylim'])
        data.append(pd.DataFrame({'panel': stratum, 'phase': ph, 'ref_q1': q1, 'ref_median': med, 'ref_q3': q3,
                                  'case_mean': cm.reindex(ph).to_numpy()}))
    axes[0].set_ylabel(o['ylabel'])
    _top_legend(fig, axes[0])
    # first-half and net yaw, person level: left-hand upright and free choice
    ax = axes[2]
    pts, ticks = [], []
    groups = (('fhy_hand', 'left_upright_all', 'First half\nleft, up'), ('net_yaw_hand', 'left_upright_all', 'Net\nleft, up'),
              ('fhy_hand', 'free_choice', 'First half\nfree'), ('net_yaw_hand', 'free_choice', 'Net\nfree'))
    cmp = _csv('x1_comparisons')
    for x, (key, stratum, lab) in enumerate(groups):
        row = cmp[(cmp['key'] == key) & (cmp['stratum'] == stratum) & (cmp['variant'] == 'main')]
        rk = ''
        if len(row):
            r = row.iloc[0]
            rk = ((f'{r["side"]} all' if int(r['n_more_extreme']) == 0 else f'{XR.ordinal(f"{r.percentile:.0f}")} pct')
                  if pd.notna(r['n_more_extreme']) else '')
        ref, case = _person_values(key, stratum)
        rng = np.random.default_rng(x + 7)
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(ref)), ref['value'], s=10, color=COLORS['ref'],
                   edgecolors=COLORS['ref_edge'], linewidths=0.4, zorder=3)
        ax.plot([x - 0.25, x + 0.25], [ref['value'].median()] * 2, color=COLORS['ref_median'], lw=1.2, zorder=4)
        if pd.notna(case['value']):
            ax.scatter([x + 0.33], [case['value']], **{**CASE_MARKER, 's': 26})
        ticks.append(f'{lab}\n{rk}')
        pts.append(pd.DataFrame({'panel': f'{key}_{stratum}', 'person_id': list(ref['person_id']) + [_case()],
                                 'value': list(ref['value']) + [case['value']]}))
    ax.axhline(0, color=AXIS, lw=0.6, zorder=0)
    ax.set_xticks(range(len(groups)), ticks)
    ax.tick_params(axis='x', length=0, labelsize=5.8)
    ax.set_xlim(-0.5, len(groups) - 0.4)
    ax.set_ylabel('Yaw, hand-consistent (deg)')
    ax.set_title('Per person', fontsize=7.2)
    data += pts
    tx = XT.texts('X1_07_hand_signature')
    L.save_fig(fig, 'X1_07_hand_signature', data=pd.concat(data, ignore_index=True), caption=tx['caption'],
               decision=tx['decision'])


def _z_rows(main: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(g, main[main['group_name'] == g]) for g in XS.GROUPS if (main['group_name'] == g).any()]


def _z_panel(ax: Any, groups: list[tuple[str, pd.DataFrame]], o: dict[str, Any]) -> list[dict[str, Any]]:
    zlo, zhi = o['zlim']
    y, ylabels, ypos, data, group_y = 0, [], [], [], []
    for gname, g in groups:
        group_y.append((y, gname))
        y += 1
        for _, r in g.iterrows():
            ok_z = bool(r['report_z'])
            z = float(r['robust_z']) if ok_z and pd.notna(r['robust_z']) else math.nan
            inset = 0.025 * (zhi - zlo)                 # a clipped point stays whole inside the axes
            zc = float(np.clip(z, zlo + inset, zhi - inset)) if math.isfinite(z) else math.nan
            beyond, robust = bool(r['beyond_all']), bool(r['robust'])
            if math.isfinite(zc):
                if beyond:
                    ax.scatter([zc], [y], marker='D', s=24, color=COLORS['case'] if robust else 'white',
                               edgecolors=COLORS['case'] if not robust else INK, linewidths=0.9, zorder=5)
                else:
                    ax.scatter([zc], [y], marker='o', s=15, color='white', edgecolors=COLORS['ref_median'],
                               linewidths=0.7, zorder=5)
                if z > zhi or z < zlo:      # clipped: the value, inside the axes
                    ax.annotate(f'{z:.1f}', (zc, y), xytext=(-6 if z > 0 else 6, 0), textcoords='offset points',
                                ha='right' if z > 0 else 'left', va='center', fontsize=5.5, color=INK2)
            else:
                ax.text(0, y, 'share: no z', ha='center', va='center', fontsize=5.2, color=MUTED)
            lab = r['label'] + (' *' if bool(r['post_hoc']) else '')
            ylabels.append(lab)
            ypos.append(y)
            pct = (('above' if r['side'] == 'above' else 'below') if beyond else
                   (f'{r["percentile"]:.0f}' if pd.notna(r['percentile']) else '–'))
            ax.text(1.02, y, pct, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=5.5,
                    color=INK2, clip_on=False)
            data.append({'key': r['key'], 'label': r['label'], 'group': gname, 'robust_z': z, 'plotted_z': zc,
                         'z_scale': r['z_scale'], 'percentile': r['percentile'], 'beyond_all_reference': beyond,
                         'robust': robust, 'p_rank': r['p_rank'], 'q_rank': r['q_rank'],
                         'post_hoc': bool(r['post_hoc'])})
            y += 1
    ax.axvspan(-o['z_clear'], o['z_clear'], color=COLORS['band'], lw=0, zorder=0)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    for gy, gname in group_y:
        ax.text(zlo, gy, gname[:1].upper() + gname[1:], ha='left', va='center', fontsize=6.2, fontweight='bold',
                color=INK)
    ax.set_yticks(ypos, ylabels, fontsize=o['label_size'])
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(zlo, zhi)
    ax.grid(axis='y', visible=False)
    ax.text(1.02, -0.6, 'pct', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=5.5,
            color=INK2, clip_on=False)
    return data


def _null_panel(ax: Any, nn: dict[str, Any], col: str, xlabel: str) -> list[dict[str, Any]]:
    pp = pd.DataFrame(nn['per_person'])
    ref, case = pp[pp['group'] == 'reference'], pp[pp['group'] == 'exploratory'].iloc[0]
    counts = ref[col].to_numpy(dtype=int)
    # stacked dots: one dot per reference person at its count
    seen: dict[int, int] = {}
    for c in counts:
        k = seen.get(int(c), 0)
        ax.scatter([c], [k], s=16, color=COLORS['ref'], edgecolors=COLORS['ref_edge'], linewidths=0.4, zorder=3)
        seen[int(c)] = k + 1
    ax.scatter([case[col]], [max(seen.values(), default=1) + 0.8], **CASE_MARKER)
    ax.annotate(_case(), (case[col], max(seen.values(), default=1) + 0.8), xytext=(5, 0), textcoords='offset points',
                fontsize=6.5, va='center', color=INK, fontweight='bold')
    ax.set_xlim(-0.6, max(int(counts.max()), int(case[col])) + 1.6)
    ax.set_ylim(-0.8, max(seen.values(), default=1) + 1.8)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(xlabel, fontsize=6.5)
    return [{'panel': col, 'person_id': r['person_id'], 'group': r['group'], 'count': int(r[col])}
            for _, r in pp.iterrows()]


def fig_X1_08_zscore_summary(**kw: Any) -> None:
    o = {**FIGURES['X1_08_zscore_summary'], **kw}
    N = _nums()
    o['z_clear'] = float(N.v('z_clear'))
    main = _main()
    groups = _z_rows(main)
    # two columns of groups with about as many rows each
    sizes = [len(g) + 1 for _, g in groups]
    cut = int(np.argmin([abs(sum(sizes[:i]) - sum(sizes[i:])) for i in range(1, len(sizes))])) + 1
    left, right = groups[:cut], groups[cut:]
    rows = max(sum(sizes[:cut]), sum(sizes[cut:]))
    height = o['row_h'] * rows + o['null_h'] + 0.75
    fig = figure(o['width'], height)[0]
    fig.clf()
    gs = fig.add_gridspec(2, 2, height_ratios=[o['row_h'] * rows + 0.35, o['null_h']])
    data = []
    for j, part in enumerate((left, right)):
        ax = fig.add_subplot(gs[0, j])
        data += _z_panel(ax, part, o)
        ax.set_xlabel(f'Robust z, declared scale (clipped at ±{o["zlim"][1]:g})', fontsize=6.5)
        ax.tick_params(axis='x', labelsize=6)
    nn = _characteristics()['null'][o['null_set']][o['null_variant']]
    nd = []
    for j, (col, lab, title) in enumerate((
            ('n_characteristics', 'Pre-specified characteristics beyond all others',
             'Null reference: each reference person against the others'),
            ('n_phenomena', 'Phenomena beyond all others', 'The same, counted in phenomena'))):
        ax = fig.add_subplot(gs[1, j])
        nd += _null_panel(ax, nn, col, lab)
        ax.set_title(title, fontsize=7)
    tx = XT.texts('X1_08_zscore_summary')
    df = pd.concat([pd.DataFrame(data).assign(panel='zscore'), pd.DataFrame(nd)], ignore_index=True)
    L.save_fig(fig, 'X1_08_zscore_summary', data=df, caption=tx['caption'], decision=tx['decision'])


def fig_X1_09_multivariate(**kw: Any) -> None:
    o = {**FIGURES['X1_09_multivariate'], **kw}
    mv = L.read_json('x1_multivariate', folder='results')
    fig, axes = figure(o['width'], o['height'], ncols=3, gridspec_kw={'width_ratios': o['width_ratios']})
    ax = axes[0]
    sc = np.asarray(mv['scores_ref'], dtype=float)
    sx = np.asarray(mv['score_x'], dtype=float)
    ax.scatter(sc[:, 0], sc[:, 1], s=16, color=COLORS['ref'], edgecolors=COLORS['ref_edge'], linewidths=0.4, zorder=3)
    # label only the persons farthest from the reference centre in this plane (the cluster would overlap)
    far = np.argsort(-np.hypot(sc[:, 0], sc[:, 1]))[:int(o['label_top'])]
    for i in far:
        ax.annotate(mv['persons'][i], (sc[i, 0], sc[i, 1]), xytext=(3, 2), textcoords='offset points', fontsize=5.5,
                    color=INK2)
    ax.scatter([sx[0]], [sx[1]], **CASE_MARKER)
    ax.annotate(mv['case'], (sx[0], sx[1]), xytext=(4, 4), textcoords='offset points', fontsize=7, color=INK,
                fontweight='bold')
    ev = mv['explained_variance']
    ax.set_xlabel(f'PC1 ({100 * ev[0]:.0f} % of the variance)')
    ax.set_ylabel(f'PC2 ({100 * ev[1]:.0f} %)')
    ax.axhline(0, color=AXIS, lw=0.5, zorder=0)
    ax.axvline(0, color=AXIS, lw=0.5, zorder=0)
    ax.set_title(f'PCA of {len(mv["keys"])} key characteristics', fontsize=7.5)
    ax = axes[1]
    d = np.asarray(mv['mahalanobis_ref_loo'], dtype=float)
    order = np.argsort(d)
    ax.scatter(d[order], np.arange(len(d)), s=14, color=COLORS['ref'], edgecolors=COLORS['ref_edge'], linewidths=0.4,
               zorder=3)
    for k, i in enumerate(order):
        ax.text(d[i] + 0.1, k, mv['persons'][i], fontsize=5.3, color=INK2, va='center')
    ax.set_xlim(0, 1.15 * max(float(d.max()), float(mv['mahalanobis_x'])))
    ax.axvline(mv['mahalanobis_x'], color=COLORS['case'], lw=1.4, zorder=4)
    ax.text(mv['mahalanobis_x'], 0.0, f' {mv["case"]}', color=INK, fontsize=7, fontweight='bold', va='center',
            ha='left', zorder=5)          # at the bottom, where the short distances leave room
    ax.set_yticks([])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Mahalanobis distance')
    ax.set_title(f'{mv["n_ref_beyond"]} of {len(d)} persons as far\n(rank p = {mv["p_mahalanobis"]:.3f})', fontsize=7.5)
    ax = axes[2]
    lo = pd.DataFrame(mv['loco']).set_index('left_out')
    ll = pd.DataFrame(mv['linear']['loco']).set_index('left_out')
    keys = list(lo.index)
    yy = np.arange(len(keys))
    ax.scatter(lo['n_ref_beyond'], yy, marker='D', s=20, color=COLORS['case'], edgecolors=INK, linewidths=0.5,
               zorder=4, label='declared scales')
    ax.scatter(ll.loc[keys, 'n_ref_beyond'], yy, marker='o', s=16, color='white', edgecolors=COLORS['linear'],
               linewidths=0.9, zorder=3, label='linear scales')
    ax.set_yticks(yy, [textwrap.shorten(XS.BY_KEY[k].label, int(o['loco_label_chars']), placeholder='…')
                       for k in keys], fontsize=5.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, max(int(lo['n_ref_beyond'].max()), int(ll['n_ref_beyond'].max())) + 1)
    ax.set_xlabel('Persons as far, one left out', fontsize=6.5)
    ax.set_title('Leave one characteristic out', fontsize=7.5)
    ax.legend(loc='lower right', fontsize=5.5, handletextpad=0.2)
    rows = [{'panel': 'pca', 'person_id': p, 'pc1': a, 'pc2': b} for (a, b), p in zip(sc, mv['persons'])]
    rows.append({'panel': 'pca', 'person_id': mv['case'], 'pc1': sx[0], 'pc2': sx[1]})
    rows += [{'panel': 'mahalanobis', 'person_id': p, 'distance': v} for p, v in zip(mv['persons'], d)]
    rows.append({'panel': 'mahalanobis', 'person_id': mv['case'], 'distance': mv['mahalanobis_x']})
    rows += [{'panel': 'loco', 'left_out': k, 'n_ref_beyond': int(lo.loc[k, 'n_ref_beyond']),
              'n_ref_beyond_linear': int(ll.loc[k, 'n_ref_beyond'])} for k in keys]
    tx = XT.texts('X1_09_multivariate')
    L.save_fig(fig, 'X1_09_multivariate', data=pd.DataFrame(rows), caption=tx['caption'], decision=tx['decision'])


FIGURE_FUNCTIONS = {
    'X1_01_example_traces': fig_X1_01_example_traces,
    'X1_02_preparation': fig_X1_02_preparation,
    'X1_03_spectra': fig_X1_03_spectra,
    'X1_04_movement': fig_X1_04_movement,
    'X1_05_speed_scaling': fig_X1_05_speed_scaling,
    'X1_06_repeats': fig_X1_06_repeats,
    'X1_07_hand_signature': fig_X1_07_hand_signature,
    'X1_08_zscore_summary': fig_X1_08_zscore_summary,
    'X1_09_multivariate': fig_X1_09_multivariate,
}
assert tuple(FIGURE_FUNCTIONS) == XT.FIGURE_NAMES


def draw(only: list[str] | None = None) -> None:
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
