"""Figures of lane m3, drawn ONLY from saved files (results/m3_*.json and tables/m3/*). Edit FIGURES, then

    python -m stage2.pipeline figures --lanes m3                       # seconds, no model is fitted
    python -m stage2.pipeline figures --lanes m3 --only M3_04_calibration

One function per figure, named after it; every presentation detail is a keyword argument whose default lives in
``FIGURES``. Each figure also writes its plotted data as a tidy CSV (tables/m3/<figure name>.csv). Captions carry no
typed result: every number and every decision sentence is read from the saved files."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from stage2 import config as C
from stage2.lanes import m3_common as M
from stage2.pipeline import Lane, selected
from stage2.plotting import GRID, HAND_COLORS, INK, INK2, MUTED, PLATFORM_COLORS, RAMP, SERIES, figure, plain_log_ticks

L = Lane('m3')
GROUP_LABELS = {'arch': 'Deep architectures', 'treat': 'Deep: data treatment', 'treat_body': 'Deep: body-frame re-checks',
                'grip': 'Deep: grip transfer', 'nested': 'Deep: nested choice', 'mr': 'MiniRocket', 'tf': 'Time-frequency'}
GROUP_ORDER = ('arch', 'treat', 'treat_body', 'grip', 'nested', 'mr', 'tf')
ROLE_LABELS = {'deep': 'Deep finalist', 'mr': 'MiniRocket finalist', 'tf': 'Time-frequency finalist'}
ROLE_COLORS = {'deep': SERIES, 'mr': '#eb6834', 'tf': '#1baf7a'}
CAL_COLORS = {'none': MUTED, 'platt': RAMP[1], 'temperature': RAMP[3], 'isotonic': '#eb6834', 'beta': '#1baf7a',
              'nested': '#9b59b6'}
AGG_COLORS = {'mean_prob': RAMP[1], 'mean_logit': RAMP[3], 'sum_logit': '#eb6834', 'sum_logit_calibrated': '#1baf7a'}
AGG_SHORT = {'mean_prob': 'mean P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits',
             'sum_logit_calibrated': 'sum of logits x c_k'}
VERDICT_COLORS = {'better': SERIES, 'worse': '#D1495B', 'identical': INK2, 'no detectable difference': MUTED}
HAND_CMAP = LinearSegmentedColormap.from_list('hands', [HAND_COLORS['left'], '#ffffff', HAND_COLORS['right']])

FIGURES: dict[str, dict[str, Any]] = {
    'M3_01_model_overview': dict(width='full', row_h=0.16, min_h=4.0, acc_xlim=(0.45, 1.0), acc_xticks=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                                 cost_xticks=(0.1, 1, 10, 100), cost_xlim=(0.05, 300), width_ratios=(3.2, 1.0),
                                 label_fontsize=6.5, marker_size=3.2, show_reference=True),
    'M3_02_tf_sign': dict(width='full', height=4.4, auc_xlim=(0.35, 1.0), auc_xticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                          label_fontsize=6.5, cmap_mag='Greys', cmap_signed='PuOr', marker_size=3.5),
    # clip_beyond: rows whose CI reaches beyond +-clip_beyond points are drawn off scale (arrow at the edge and the
    # value printed), so that one extreme row does not squeeze every other comparison (None: never clip)
    'M3_03_treatments': dict(width='full', row_h=0.2, min_h=3.0, xlim=None, clip_beyond=20.0, label_fontsize=6.3,
                             marker_size=3.2, legend_loc='upper left'),
    'M3_10_vs_engineered': dict(width='full', row_h=0.22, min_h=2.6, xlim=None, clip_beyond=None, label_fontsize=6.3,
                                marker_size=3.2, legend_loc='upper left'),
    'M3_04_calibration': dict(width='full', height=4.8, ece_xmax=None, marker_size=3.0, band_alpha=0.18,
                              legend_loc='lower right'),
    'M3_05_saliency': dict(width='full', height=None, row_h=0.17, min_h=3.0, vmax_quantile=0.99,
                           profile_ylabel='share of the group\'s |IG| per sample', label_fontsize=6.5,
                           legend_loc='lower left'),
    'M3_06_breakdown': dict(width='full', row_h=0.17, min_h=3.0, xlim=None, xticks=None, label_fontsize=6.5,
                            marker_size=3.0, min_n=1),
    'M3_07_attempts': dict(width='full', height=2.7, k_ylim=(0.7, 1.0), oc_ylim=None, marker_size=3.2),
    'M3_08_per_person': dict(width='full', height=2.6, ylim=None, marker_size=3.4),
    'M3_09_grip_transfer': dict(width='full', height=None, xlim=(0.4, 1.0), xticks=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                                diff_xlim=None, marker_size=3.4, label_fontsize=6.3, dodge=0.22),
}
#: markers of the instructed grips (stratification only) in M3_09
GRIP_MARKERS = {'upright': 'o', 'top_left': '<', 'top_right': '>'}
UNSEEN_COLOR = '#eb6834'                        # a grip the model never saw in training (not a hand colour)


def _o(name: str, kw: dict[str, Any]) -> dict[str, Any]:
    return {**FIGURES[name], **kw}


def _summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    s = L.read_json('m3_summary')
    return pd.DataFrame(s['experiments']), s.get('reference') or {}


def _selection() -> dict[str, Any]:
    return L.read_json('m3_selection')['roles']


def _shipped() -> dict[str, str]:
    """role -> the shipped (calibrated) result of the role, as decided in m3_calibration.json."""
    from stage2.lanes.m3_derive import shipped_name
    sel = _selection()
    cal = L.read_json('m3_calibration')['roles']
    return {r: shipped_name(r, sel, cal) for r in ('deep', 'mr', 'tf') if r in sel}


def _ci_bars(ax: Any, x: Any, y: Any, lo: Any, hi: Any, color: Any = INK2, lw: float = 0.8, cap: float = 0.18) -> None:
    """Horizontal CI segments from lo to hi with small caps; rows without a CI (NaN) are skipped."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return
    cols = np.broadcast_to(np.asarray(color, dtype=object), y.shape)[ok] if not isinstance(color, str) else color
    ax.hlines(y[ok], lo[ok], hi[ok], color=cols, lw=lw, zorder=2)
    ax.vlines(lo[ok], y[ok] - cap, y[ok] + cap, color=cols, lw=lw, zorder=2)
    ax.vlines(hi[ok], y[ok] - cap, y[ok] + cap, color=cols, lw=lw, zorder=2)


def _pct(v: float) -> str:
    return f'{100 * v:.1f}%'


def _pfmt(p: float) -> str:
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def _sci(v: float) -> str:
    return '0' if v == 0 else f'{v:.0e}'


# ───────────────────────────── M3_01 ─────────────────────────────


def fig_M3_01_model_overview(**kw: Any) -> None:
    name = 'M3_01_model_overview'
    o = _o(name, kw)
    ex, ref = _summary()
    ex = ex[ex['group'] != 'calib'].copy()
    ex['_g'] = ex['group'].map({g: i for i, g in enumerate(GROUP_ORDER)})
    ex = ex.sort_values(['_g', 'accuracy'], ascending=[True, False]).reset_index(drop=True)
    pos, gstarts, y = [], [], 0.0
    for g in GROUP_ORDER:
        sub = ex[ex['group'] == g]
        if not len(sub):
            continue
        gstarts.append((y, GROUP_LABELS[g]))
        y += 0.9
        for _ in range(len(sub)):
            pos.append(y)
            y += 1
        y += 0.4
    ex['y'] = pos
    h = max(o['min_h'], o['row_h'] * (y + 1))
    fig, ax = figure(o['width'], h, 1, 2, sharey=True, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    if o['show_reference'] and ref:
        a.axvspan(ref['accuracy_lo'], ref['accuracy_hi'], color=GRID, alpha=0.8, lw=0)
        a.axvline(ref['accuracy'], color=INK2, lw=0.8, ls='--')
        a.text(ref['accuracy'], -0.6, f"reference {_pct(ref['accuracy'])}", ha='center', va='bottom', fontsize=6, color=INK2)
    col = np.array([MUTED if r.get('physics_demo') else SERIES for _, r in ex.iterrows()])
    gdep = ex['grip_dependent'].fillna(False).astype(bool).to_numpy()
    post = ex['post_hoc'].fillna(False).astype(bool).to_numpy()
    _ci_bars(a, ex['accuracy'], ex['y'], ex['accuracy_lo'], ex['accuracy_hi'])
    for mask, marker in ((~post, 'o'), (post, 'D')):           # diamond: designed post hoc
        full = mask & ~gdep
        if full.any():
            a.scatter(ex['accuracy'][full], ex['y'][full], s=o['marker_size'] ** 2 * 2, c=list(col[full]), marker=marker,
                      zorder=3)
        opn = mask & gdep                                     # open: grip-dependent input (or candidates)
        if opn.any():
            a.scatter(ex['accuracy'][opn], ex['y'][opn], s=o['marker_size'] ** 2 * 2, facecolors='white',
                      edgecolors=list(col[opn]), linewidths=1.0, marker=marker, zorder=3)
    a.set_yticks(ex['y'], ex['label'], fontsize=o['label_fontsize'])
    for yy, lab in gstarts:
        a.text(o['acc_xlim'][0], yy + 0.15, lab, fontsize=7, fontweight='bold', color=INK, va='center')
    a.set_xlim(*o['acc_xlim'])
    a.set_xticks(o['acc_xticks'])
    a.set_ylim(y, -1.2)
    a.set_xlabel('leave-one-person-out accuracy (95 % person-bootstrap CI)')
    a.grid(axis='y', visible=False)
    b = ax[1]
    b.barh(ex['y'], ex['cpu_min'].clip(lower=o['cost_xlim'][0]), height=0.6, color=SERIES, alpha=0.7)
    b.set_xscale('log')
    b.set_xlim(*o['cost_xlim'])
    plain_log_ticks(b.xaxis, o['cost_xticks'])
    b.set_xlabel('training CPU-min\n(one LOSO run)')
    b.grid(axis='y', visible=False)
    n_p = int(ex['n_persons'].iloc[0])
    L.save_fig(fig, name, data=ex[['name', 'label', 'group', 'accuracy', 'accuracy_lo', 'accuracy_hi', 'roc_auc', 'log_loss',
                                   'cpu_min', 'n_configs_tried', 'selection', 'post_hoc', 'grip_dependent']],
               caption=(f'Every lane-m3 model, leave-one-person-out over {n_p} persons: accuracy with its 95 % '
                        'person-cluster bootstrap CI (left; dashed line and band: the reference yaw logistic regression) '
                        'and the training cost of one LOSO run (right, CPU-minutes, log scale; for a nested choice, of '
                        'all its candidates). Grey points: the magnitude-only physics demonstrations. Open markers: '
                        'grip-dependent device-frame input (for a nested choice: grip-dependent configurations among its '
                        'candidates). Diamonds: configurations designed after outer-fold results had been seen (post hoc).'),
               decision=('Representation learning is compared with the engineered-feature reference at equal footing; '
                         'the cost column shows what each point of accuracy costs.'))


# ───────────────────────────── M3_02 ─────────────────────────────

TF_ROWS = (('m3_tflog_stft', 'STFT signed, logistic'), ('m3_tflog_stft_mag', 'STFT magnitude, logistic (mirror-augm.)'),
           ('m3_tflog_stft_mag_noaug', 'STFT magnitude, logistic (no mirror)'), ('m3_tflog_cwt', 'CWT signed, logistic'),
           ('m3_tflog_cwt_mag', 'CWT magnitude, logistic (mirror-augm.)'),
           ('m3_tflog_cwt_mag_noaug', 'CWT magnitude, logistic (no mirror)'), ('m3_cnn2d_stft', 'STFT signed, 2D CNN'),
           ('m3_cnn2d_cwt', 'CWT signed, 2D CNN'), ('m3_cnn2d_stft_mag', 'STFT magnitude, 2D CNN (no mirror)'))


def fig_M3_02_tf_sign(**kw: Any) -> None:
    name = 'M3_02_tf_sign'
    o = _o(name, kw)
    tx = L.read_json('m3_tf_example')
    ex, _ = _summary()
    exi = ex.set_index('name')
    fig = figure(o['width'], o['height'])[0]
    fig.clf()
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.25))
    x = np.asarray(tx['x'])
    u = np.linspace(0, 1, x.size)
    a = fig.add_subplot(gs[0, 0])
    a.plot(u, x, color=HAND_COLORS['right'])
    a.plot(u, -x, color=HAND_COLORS['left'])
    a.axhline(0, color=INK2, lw=0.5)
    a.set_xlabel('normalised time (onset to end)')
    a.set_ylabel('yaw rate (deg/s)')
    a.set_title('(a) x (teal) and\nits mirror -x (red)', fontsize=7.5)
    mag = np.asarray(tx['stft_mag'])
    re, re_m = np.asarray(tx['stft_real']), np.asarray(tx['stft_real_mirror'])
    ext = (0, 1, 0, mag.shape[0] - 1)
    a = fig.add_subplot(gs[0, 1])
    im = a.imshow(mag, origin='lower', aspect='auto', cmap=o['cmap_mag'], extent=ext)
    a.set_title(f"(b) |STFT x| = |STFT -x|\n(max. difference {_sci(tx['max_abs_mag_diff_stft'])})", fontsize=7.5)
    a.set_xlabel('normalised time')
    a.set_ylabel('frequency bin')
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)
    vm = float(np.max(np.abs(re)))
    a = fig.add_subplot(gs[1, 0])
    a.imshow(re, origin='lower', aspect='auto', cmap=o['cmap_signed'], vmin=-vm, vmax=vm, extent=ext)
    a.set_title('(c) Re STFT of x', fontsize=7.5)
    a.set_xlabel('normalised time')
    a.set_ylabel('frequency bin')
    a.grid(False)
    a = fig.add_subplot(gs[1, 1])
    im = a.imshow(re_m, origin='lower', aspect='auto', cmap=o['cmap_signed'], vmin=-vm, vmax=vm, extent=ext)
    a.set_title('(d) Re STFT of -x: the sign flips', fontsize=7.5)
    a.set_xlabel('normalised time')
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)
    a = fig.add_subplot(gs[:, 2])
    rows = [(n, lab) for n, lab in TF_ROWS if n in exi.index]
    yy = np.arange(len(rows))
    auc = np.array([exi.loc[n, 'roc_auc'] for n, _ in rows])
    lo = np.array([exi.loc[n, 'roc_auc_lo'] for n, _ in rows])
    hi = np.array([exi.loc[n, 'roc_auc_hi'] for n, _ in rows])
    signed = np.array(['mag' not in n for n, _ in rows])
    a.axvline(0.5, color=INK2, lw=0.7, ls=':')
    _ci_bars(a, auc, yy, lo, hi)
    a.scatter(auc, yy, s=o['marker_size'] ** 2 * 2, c=[SERIES if s else MUTED for s in signed], zorder=3)
    a.set_yticks(yy, [lab for _, lab in rows], fontsize=o['label_fontsize'])
    a.yaxis.tick_right()
    a.invert_yaxis()
    a.set_xlim(*o['auc_xlim'])
    a.set_xticks(o['auc_xticks'])
    a.set_xlabel('LOSO ROC AUC (95 % CI)')
    a.grid(axis='y', visible=False)
    a.set_title('(e) signed vs magnitude-only inputs', fontsize=7.5)
    data = pd.DataFrame({'name': [n for n, _ in rows], 'label': [lab for _, lab in rows], 'roc_auc': auc, 'lo': lo,
                         'hi': hi, 'signed': signed})
    L.save_fig(fig, name, data=data,
               caption=(f"Why magnitude-only time-frequency inputs cannot see the hand. (a) the yaw rate of the median "
                        f"right-hand approach by first-half yaw ({tx.get('example_condition', 'any')}) and its mirror image; "
                        f"(b) their STFT magnitudes are identical (largest difference over every chiral channel "
                        f"({', '.join(tx.get('chiral_channels', [tx['channel']]))}) of all {tx['n_trials']} trials: "
                        f"{_sci(tx['max_abs_mag_diff_stft'])}; CWT {_sci(tx['max_abs_mag_diff_cwt'])}); (c, d) the real "
                        'parts change sign. (e) LOSO ROC AUC of signed and magnitude-only models: with mirror augmentation '
                        'a magnitude-only model sees every input twice with opposite labels and sits at chance; without '
                        'it, it can only use what is not mirror-symmetric between the hands.'),
               decision='Time-frequency models of lane m3 keep the sign (real and imaginary parts), never the magnitude alone.')


# ───────────────────────────── M3_03 and M3_10 (paired forest plots) ─────────────────────────────

FAMILY_HEADS = {'treatment:m3_cnn': 'vs the 1D CNN default (gravity frame)', 'treatment:m3_mr': 'vs the MiniRocket default',
                'treatment:m3_tflog_stft': 'vs the STFT logistic default', 'treatment:m3_tflog_cwt': 'vs the CWT logistic',
                'treatment': 'choices against their family default', 'treatment_body': 'body-frame re-checks (vs body CNN)',
                'nested': 'nested estimates (vs the 1D CNN default)', 'grip': 'grip transfer, rotated grips',
                'equal_input': 'deep vs linear on matched input', 'engineered': 'vs engineered features',
                'physics_demo': 'magnitude-only demonstrations'}


def _paired() -> tuple[pd.DataFrame, dict[str, Any]]:
    pj = L.read_json('m3_paired')
    return pd.DataFrame(pj['rows']), pj


def _forest(name: str, pp: pd.DataFrame, labels: list[str], heads: dict[str, str], o: dict[str, Any], caption: str,
            decision: str) -> None:
    """Grouped forest plot of mean per-person accuracy differences (rows of ``pp`` in display order, grouped by
    ``pp['head']``), coloured by the Holm-adjusted verdict."""
    pp = pp.reset_index(drop=True)
    y, hpos, yy, prev = [], [], 0.0, None
    for _, r in pp.iterrows():
        if r['head'] != prev:
            hpos.append((yy, heads.get(r['head'], str(r['head']))))
            yy += 1.0
            prev = r['head']
        y.append(yy)
        yy += 1.0
    y = np.asarray(y)
    h = max(o['min_h'], o['row_h'] * (yy + 2))
    fig, a = figure(o['width'], h)
    d, lo, hi = (pp['diff'] * 100).to_numpy(float), (pp['diff_lo'] * 100).to_numpy(float), (pp['diff_hi'] * 100).to_numpy(float)
    cb = o.get('clip_beyond')
    inside = np.ones(len(pp), bool) if not cb else ((lo >= -cb) & (hi <= cb))
    if not inside.any():
        inside[:] = True
    xl = o['xlim'] or (min(-5.0, float(np.nanmin(lo[inside])) - 1.5), max(5.0, float(np.nanmax(hi[inside])) + 1.5))
    off = (lo < xl[0]) | (hi > xl[1]) | (d < xl[0]) | (d > xl[1])        # drawn off scale
    cols = np.array([VERDICT_COLORS.get(v, MUTED) for v in pp['verdict']], dtype=object)
    a.axvline(0, color=INK, lw=0.7)
    _ci_bars(a, d[~off], y[~off], lo[~off], hi[~off])
    a.scatter(d[~off], y[~off], s=o['marker_size'] ** 2 * 2, c=list(cols[~off]), zorder=3)
    for i in np.flatnonzero(off):                                         # arrow at the edge, the values printed
        left = d[i] < 0
        edge = xl[0] if left else xl[1]
        inset = (0.02 if left else -0.02) * (xl[1] - xl[0])
        a.hlines(y[i], max(lo[i], xl[0]), min(hi[i], xl[1]), color=INK2, lw=0.8, zorder=2)
        a.scatter([edge + inset], [y[i]], marker='<' if left else '>', s=o['marker_size'] ** 2 * 3, c=[cols[i]],
                  zorder=3)
        a.text(edge + 2 * inset, y[i],
               f'off scale: {d[i]:+.1f} [{lo[i]:+.1f}, {hi[i]:+.1f}]', ha='left' if left else 'right', va='center',
               fontsize=5.6, color=INK2)
    a.set_yticks(y, labels, fontsize=o['label_fontsize'])
    a.set_ylim(yy, -0.6)
    for yh, lab in hpos:
        a.text(-0.02, yh, lab, transform=a.get_yaxis_transform(), ha='right', va='center', fontsize=6.8,
               fontweight='bold', color=INK)
    for i, r in pp.iterrows():
        a.text(1.01, y[i], f"{_pct(r['accuracy'])}, Holm {_pfmt(r['p_holm'])}", transform=a.get_yaxis_transform(),
               va='center', fontsize=5.8, color=INK2)
    a.set_xlim(*xl)
    a.set_xlabel('mean per-person accuracy difference (points, 95 % CI)')
    a.grid(axis='y', visible=False)
    hs = [Line2D([], [], marker='o', ls='', color=c, label=v) for v, c in VERDICT_COLORS.items() if v in set(pp['verdict'])]
    a.legend(handles=hs, fontsize=5.8, loc=o.get('legend_loc', 'upper left'), handlelength=0.8,
             title='Holm-adjusted verdict', title_fontsize=5.8)
    L.save_fig(fig, name, data=pp[['name', 'baseline', 'family', 'subset', 'treatment', 'option', 'accuracy', 'diff',
                                   'diff_lo', 'diff_hi', 'p', 'p_holm', 'verdict', 'logloss_diff']],
               caption=caption, decision=decision)


def _verdict_sentence(pp: pd.DataFrame) -> str:
    better = [f"{r['option']}" for _, r in pp.iterrows() if r['verdict'] == 'better']
    worse = [f"{r['option']}" for _, r in pp.iterrows() if r['verdict'] == 'worse']
    return (f"better after adjustment: {'; '.join(better) or 'none'}; worse: {'; '.join(worse) or 'none'}; every other "
            'difference is not detectable')


def fig_M3_03_treatments(**kw: Any) -> None:
    name = 'M3_03_treatments'
    o = _o(name, kw)
    pp, pj = _paired()
    pp = pp[pp['family'].isin(['treatment', 'treatment_body', 'nested']) |
            ((pp['family'] == 'grip') & (pp['subset'] == 'rotated'))].copy()
    pp['head'] = [f'treatment:{b}' if f == 'treatment' else f for f, b in zip(pp['family'], pp['baseline'])]
    heads = list(dict.fromkeys([h for h in FAMILY_HEADS if h in set(pp['head'])] + sorted(set(pp['head']))))
    pp['_f'] = pp['head'].map({h: i for i, h in enumerate(heads)})
    pp = pp.sort_values(['_f', 'treatment', 'diff'], ascending=[True, True, False]).reset_index(drop=True)
    labels = [r['option'] if r['family'] in ('grip', 'nested') else f"{r['treatment']}: {r['option']}"
              for _, r in pp.iterrows()]
    order = {'treatment': 0, 'treatment_body': 1, 'grip': 2, 'nested': 3}
    sizes = ', '.join(f"{f} {int((pp['family'] == f).sum())}" for f in order if (pp['family'] == f).any())
    _forest(name, pp, labels, FAMILY_HEADS, o,
            caption=('Data treatment and model choices of lane m3, each paired against its family default on the same '
                     'trials; the body-frame re-checks against the body-frame CNN; the grip-transfer comparisons on the '
                     'rotated grips (never seen by upright-only training); the nested estimates against the 1D CNN '
                     'default. Mean per-person accuracy difference with its paired person-bootstrap 95 % CI; right: the '
                     'option\'s accuracy on those trials and the sign-flip p Holm-adjusted within its family (family '
                     f'sizes in the m3 summary; rows shown: {sizes}). Colour: the Holm-adjusted verdict.'
                     + (f" Rows whose CI reaches beyond +-{o['clip_beyond']:.0f} points are drawn off scale (arrow at the "
                        'edge, values printed).' if o.get('clip_beyond') else '')),
            decision=(f"Each choice is kept or dropped on this evidence: "
                      f"{_verdict_sentence(pp[pp['family'].isin(['treatment', 'treatment_body'])])}. Grip transfer on the "
                      f"rotated grips: {_verdict_sentence(pp[pp['family'] == 'grip'])}."))


def fig_M3_10_vs_engineered(**kw: Any) -> None:
    name = 'M3_10_vs_engineered'
    o = _o(name, kw)
    pp, _ = _paired()
    pp = pp[pp['family'].isin(['equal_input', 'engineered'])].copy()
    if not len(pp):
        return
    from stage2.lanes.m3_derive import ENGINEERED
    pp['head'] = [f'eng:{b}' if f == 'engineered' else 'equal_input' for f, b in zip(pp['family'], pp['baseline'])]
    order = {'equal_input': 0, **{f'eng:{c}': i + 1 for i, c in enumerate(ENGINEERED)}}
    pp['_f'] = pp['head'].map(order).fillna(len(order))
    pp = pp.sort_values(['_f', 'diff'], ascending=[True, False]).reset_index(drop=True)
    heads = {'equal_input': 'deep vs linear on matched input', **{f'eng:{c}': f'vs {lab}' for c, lab in ENGINEERED.items()}}
    labels = [r['label'] if r['family'] == 'engineered' else f"{r['label']} vs {r['baseline_label']}" for _, r in pp.iterrows()]
    _forest(name, pp, labels, heads, o,
            caption=("Lane-m3 models against linear and engineered-feature models on the same trials. Top: deep against "
                     "linear models on matched information (lane m1's PHYS6 logistic uses the yaw about gravity plus "
                     "body-frame roll and lateral motion). Below: the shipped m3 finalists (calibrated) and the nested deep "
                     "estimates against the reference floor, lane m1's pre-specified physics logistic and lane m2's best "
                     "engineered-feature model. Mean per-person accuracy difference with its paired person-bootstrap 95 % "
                     "CI; right: the m3 model's accuracy and the Holm-adjusted sign-flip p (families equal_input and "
                     "engineered). Colour: the Holm-adjusted verdict."),
            decision=f"Whether representation learning beats engineered features on new persons: {_verdict_sentence(pp)}.")


# ───────────────────────────── M3_04 ─────────────────────────────


def fig_M3_04_calibration(**kw: Any) -> None:
    name = 'M3_04_calibration'
    o = _o(name, kw)
    calj = L.read_json('m3_calibration')
    cal = calj['roles']
    roles = [r for r in ('deep', 'mr', 'tf') if r in cal]
    fig, ax = figure(o['width'], o['height'], 2, len(roles), squeeze=False)
    rows = []
    target = calj['target']['ece_max']
    ece_max = o['ece_xmax'] or 1.12 * max(max(e['ece'], e['ece_null_q95'], target) for r in roles for e in cal[r]['entries'])
    for j, role in enumerate(roles):
        c = cal[role]
        ent = {e['method']: e for e in c['entries']}
        shipped = c['decision']['shipped']
        a = ax[0, j]
        a.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=':')
        for m in dict.fromkeys(('none', shipped)):
            if m not in ent:
                continue
            rel = pd.DataFrame(ent[m]['reliability'])
            colr = CAL_COLORS[m]
            a.fill_between(rel['p_mean'], rel['y_lo'], rel['y_hi'], color=colr, alpha=o['band_alpha'], lw=0)
            lab = 'none' if m == 'none' else M.CALIB_LABELS[m]
            a.plot(rel['p_mean'], rel['y_mean'], 'o-', color=colr, ms=o['marker_size'],
                   label=f"{lab}{' (s)' if m == shipped else ''} {ent[m]['ece']:.3f}")
            off = rel['off'].astype(bool) if 'off' in rel else pd.Series(False, index=rel.index)
            if off.any():                                       # bins whose band excludes the diagonal
                a.scatter(rel['p_mean'][off], rel['y_mean'][off], s=40, facecolors='none', edgecolors=colr, lw=1.0, zorder=4)
            rows += [{'role': role, 'method': m, 'panel': 'reliability', **r_} for r_ in rel.to_dict(orient='records')]
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_aspect('equal', adjustable='box')
        a.set_xlabel('predicted P(right) (bin mean)')
        if j == 0:
            a.set_ylabel('observed share of right hands')
        a.legend(loc=o['legend_loc'], fontsize=5.6, handlelength=1.0, title='ECE', title_fontsize=5.6,
                 borderaxespad=0.2)
        a.set_title(f"{ROLE_LABELS[role]}\n({c['label']})", fontsize=7)
        b = ax[1, j]
        ms = [m for m in M.CALIBRATORS if m in ent]
        yy = np.arange(len(ms))
        e = np.array([ent[m]['ece'] for m in ms])
        nq = np.array([ent[m]['ece_null_q95'] for m in ms])
        b.barh(yy, nq, height=0.55, color=GRID, lw=0, zorder=1)           # perfect-calibration 95th percentile
        b.scatter(e, yy, c=[CAL_COLORS[m] for m in ms], s=o['marker_size'] ** 2 * 2, zorder=3)
        b.axvline(target, color=INK2, lw=0.7, ls='--')
        b.set_yticks(yy, [f"{M.CALIB_LABELS[m]}{' *' if m == c['prespecified'] else ''}{' (s)' if m == shipped else ''}"
                          f"\n(LL {ent[m]['log_loss']:.3f})" for m in ms], fontsize=6.0)
        b.invert_yaxis()
        b.set_xlabel('ECE')
        b.set_xlim(0, ece_max)
        b.grid(axis='y', visible=False)
        for m in ms:
            rows.append({'role': role, 'method': m, 'panel': 'ece', 'ece': ent[m]['ece'], 'ece_null_q95': ent[m]['ece_null_q95'],
                         'cal_slope': ent[m]['cal_slope'], 'log_loss': ent[m]['log_loss'], 'brier': ent[m]['brier'],
                         'target_met': ent[m]['target_met']})
    ship_txt = '; '.join(f"{ROLE_LABELS[r].lower()}: {M.CALIB_LABELS[cal[r]['decision']['shipped']]} "
                         f"({'meets' if cal[r]['decision']['shipped_target_met'] else 'misses'} the target)" for r in roles)
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=('Calibration of each role\'s best lane-m3 model (leave-one-person-out). Top: reliability with '
                        f'{C.ECE_BINS} equal-mass bins and 95 % person-bootstrap bands (bins held fixed), uncalibrated (grey) and with '
                        'the shipped remedy, fitted inside every outer fold on the cross-fitted logits of the training '
                        'persons; ringed points: bins off the diagonal (the person-bootstrap band excludes the bin\'s mean '
                        'prediction and the observed share lies outside the 95 % consistency band of a perfectly '
                        'calibrated model). Bottom: ECE of every remedy '
                        '(points) against the 95th percentile of its distribution under perfect calibration (grey bars); '
                        f'dashed: the docs/09 target ({target}); * pre-specified (docs/09 D19), (s) shipped; LL = log loss.'),
               decision=f'Shipped remedies, decided by paired, Holm-adjusted log-loss comparisons: {ship_txt}.')


# ───────────────────────────── M3_05 ─────────────────────────────


def fig_M3_05_saliency(**kw: Any) -> None:
    name = 'M3_05_saliency'
    o = _o(name, kw)
    s = L.read_json('m3_saliency')
    chans = [c for c in s['channels'] if c != 'mask']
    nC = len(chans)
    R = np.asarray(s['signed_mean']['right'])[:nC]
    Lh = np.asarray(s['signed_mean']['left'])[:nC]
    vm = float(np.quantile(np.abs(np.concatenate([R.ravel(), Lh.ravel()])), o['vmax_quantile'])) or 1.0
    T = R.shape[1]
    window = s['window']
    ext = (0, 1, nC - 0.5, -0.5) if window == 'approach' else (-0.5, 2.5, nC - 0.5, -0.5)
    xlab = 'normalised time' if window == 'approach' else 'time from onset (s)'
    h = o['height'] or max(o['min_h'], o['row_h'] * nC + 1.3)
    fig, ax = figure(o['width'], h, 1, 3, gridspec_kw={'width_ratios': (1, 1, 1.05)})
    for a, Mx, title in ((ax[0], R, '(a) right-hand trials'), (ax[1], Lh, '(b) left-hand trials')):
        im = a.imshow(Mx, aspect='auto', cmap=HAND_CMAP, vmin=-vm, vmax=vm, extent=ext, interpolation='nearest')
        a.set_yticks(range(nC), chans if a is ax[0] else [''] * nC, fontsize=o['label_fontsize'])
        a.set_xlabel(xlab)
        a.set_title(title, fontsize=7.5)
        a.grid(False)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.05, pad=0.02)
    cb.set_label('mean IG (logit units)', fontsize=6.2)
    a = ax[2]
    u = np.linspace(0, 1, T) if window == 'approach' else np.linspace(-0.5, 2.5, T)
    prof = {}
    for grp, ls in (('odd', '-'), ('even', '--')):
        v = (np.asarray(s[f'time_profile_{grp}']['right']) + np.asarray(s[f'time_profile_{grp}']['left'])) / 2
        prof[grp] = v / v.sum()
        a.plot(u, prof[grp], color=INK if grp == 'odd' else MUTED, ls=ls,
               label=f"{grp}: {100 * s[f'share_first_half_{grp}']:.0f} % in 1st half")
    fh_end = 0.5 if window == 'approach' else 0.5 * float(s.get('median_duration_s', 0.0))
    if fh_end > 0:
        a.axvspan(0, fh_end, color=GRID, alpha=0.6, lw=0)
    a.set_xlabel(xlab)
    a.set_ylabel(o['profile_ylabel'])
    a.set_ylim(0, None)
    a.legend(fontsize=5.6, loc=o['legend_loc'], handlelength=1.6)
    a.set_title('(c) timing of |IG|:\nodd vs even channels', fontsize=7.5)
    data = pd.concat([pd.DataFrame({'panel': 'map', 'hand': hd, 'channel': np.repeat(chans, T), 't': np.tile(u, nC),
                                    'value': np.asarray(s['signed_mean'][hd])[:nC].ravel()}) for hd in ('right', 'left')]
                     + [pd.DataFrame({'panel': 'profile', 'hand': 'both', 'channel': grp, 't': u, 'value': v})
                        for grp, v in prof.items()], ignore_index=True)
    odd = ', '.join(s['odd_channels'])
    L.save_fig(fig, name, data=data,
               caption=(f"Integrated gradients of the deep finalist's network ({s['label']}), out of fold: every held-out "
                        'trial is attributed by its own fold\'s ensemble (baseline: the normalised zero, which is '
                        'mirror-invariant). Left, middle: mean attribution per channel and time toward P(right) for '
                        'right- and left-hand trials (teal pushes toward right, red toward left). Right: the time profile '
                        f'of |IG| of the channels that are odd under the mirror and grip-invariant ({odd}; together '
                        f"{100 * s['share_odd']:.0f} % of all |IG|, against {100 * s['share_odd_uniform']:.0f} % if every "
                        'channel weighed the same) and of the other (even) channels; grey: the first half. The odd channels '
                        f"are more front-loaded by {100 * s['first_half_odd_minus_even']:+.1f} points "
                        f"[{100 * s['first_half_odd_minus_even_lo']:+.1f}, {100 * s['first_half_odd_minus_even_hi']:+.1f}] "
                        f"(person bootstrap). Mirror consistency: corr(IG(mirror x), -IG(x)) = {s['mirror_corr']:.2f}."),
               decision=('Whether the network uses the evidence of the engineered features (early, hand-signed motion), '
                         'which channels carry it, and how much of its early timing is generic.'))


# ───────────────────────────── M3_06 ─────────────────────────────


def fig_M3_06_breakdown(**kw: Any) -> None:
    name = 'M3_06_breakdown'
    o = _o(name, kw)
    bdj = L.read_json('m3_breakdown')
    bd = pd.DataFrame(bdj['rows'])
    b = bd[bd['role'] == 'deep'].copy() if (bd['role'] == 'deep').any() else bd.copy()
    b = b[b['n'] >= o['min_n']].reset_index(drop=True)
    y, labels, heads, yy = [], [], [], 0.0
    for cond, g in b.groupby('condition', sort=False):
        heads.append((yy, cond.replace('_', ' ')))
        yy += 0.9
        for _, r in g.iterrows():
            y.append(yy)
            labels.append(f"{str(r['level']).replace('_', ' ')} (n = {r['n']}, {r['n_persons']} p.)")
            yy += 1
        yy += 0.3
    b['y'] = y
    h = max(o['min_h'], o['row_h'] * (yy + 1))
    fig, a = figure(o['width'], h)
    acc = b['accuracy']
    _ci_bars(a, acc, b['y'], b['accuracy_lo'], b['accuracy_hi'])
    nci = b['accuracy_lo'].isna().to_numpy()
    colors = np.array([PLATFORM_COLORS.get(r['level'], SERIES) if r['condition'] == 'platform' else SERIES
                       for _, r in b.iterrows()], dtype=object)
    a.scatter(acc[~nci], b['y'][~nci], c=list(colors[~nci]), s=o['marker_size'] ** 2 * 2, zorder=3)
    if nci.any():                                              # too few persons for a CI: open marker, no bar
        a.scatter(acc[nci], b['y'][nci], facecolors='white', edgecolors=list(colors[nci]), s=o['marker_size'] ** 2 * 2,
                  lw=1.0, zorder=3)
    a.set_yticks(b['y'], labels, fontsize=o['label_fontsize'])
    vals = np.concatenate([acc.to_numpy(float), b['accuracy_lo'].dropna().to_numpy(float)])
    xl = o['xlim'] or (max(0.0, math.floor((np.nanmin(vals) - 0.05) * 10) / 10), 1.0)
    for yh, lab in heads:
        a.text(xl[0], yh + 0.1, lab, fontsize=7, fontweight='bold', va='center')
    a.set_ylim(yy, -1)
    a.set_xlim(*xl)
    if o['xticks']:
        a.set_xticks(o['xticks'])
    pf = b[b['condition'] == 'platform']
    overall = float((pf['accuracy'] * pf['n']).sum() / pf['n'].sum())
    a.axvline(overall, color=INK2, lw=0.7, ls='--')
    a.set_xlabel('accuracy (95 % person-bootstrap CI within the level)')
    a.grid(axis='y', visible=False)
    L.save_fig(fig, name, data=b[['condition', 'level', 'n', 'n_persons', 'accuracy', 'accuracy_lo', 'accuracy_hi']],
               caption=(f"Shipped deep finalist ({bdj['names'].get('deep', '')}) by platform and by instructed condition "
                        f"(dashed: overall accuracy). Levels with fewer than {bdj['min_persons_ci']} persons carry no CI "
                        '(open markers). Conditions only stratify the results; they are never model inputs. The levels '
                        'are compared without a multiplicity adjustment (exploratory).'),
               decision='Which platforms and movement conditions the model is weak on (the PI chooses speed and grip freely).')


# ───────────────────────────── M3_07 ─────────────────────────────


def fig_M3_07_attempts(**kw: Any) -> None:
    name = 'M3_07_attempts'
    o = _o(name, kw)
    ex, ref = _summary()
    exi = ex.set_index('name')
    ship = _shipped()
    agj = L.read_json('m3_aggregate')
    rec = agj['recommended']
    fig, ax = figure(o['width'], o['height'], 1, 2)
    a = ax[0]
    rows = []
    ks = list(C.K_ATTEMPTS)
    for role, n in ship.items():
        r = exi.loc[n]
        acc = [r.get(f'acc_k{k}') for k in ks]
        lo = [r.get(f'acc_k{k}_lo') for k in ks]
        hi = [r.get(f'acc_k{k}_hi') for k in ks]
        a.fill_between(ks, lo, hi, color=ROLE_COLORS[role], alpha=0.12, lw=0)
        a.plot(ks, acc, 'o-', color=ROLE_COLORS[role], ms=o['marker_size'], label=f"{ROLE_LABELS[role]} (mean P)")
        rows += [{'panel': 'k_curve', 'series': n, 'aggregator': 'mean_prob', 'k': k, 'y': v, 'lo': l_, 'hi': h_}
                 for k, v, l_, h_ in zip(ks, acc, lo, hi)]
        if rec['aggregator'] and rec['aggregator'] != 'mean_prob':
            blk = (agj['roles'].get(role) or {}).get('calibrated')
            if blk:
                g = pd.DataFrame(blk['rows'])
                g = g[g['aggregator'] == rec['aggregator']].sort_values('k')
                a.plot(g['k'], g['accuracy'], 's--', color=ROLE_COLORS[role], ms=o['marker_size'] - 0.6, alpha=0.8,
                       label=f"dashed: {AGG_SHORT[rec['aggregator']]} (recommended)" if role == 'deep' else None)
                rows += [{'panel': 'k_curve', 'series': n, 'aggregator': rec['aggregator'], 'k': int(k), 'y': v,
                          'lo': l_, 'hi': h_} for k, v, l_, h_ in zip(g['k'], g['accuracy'], g['accuracy_lo'], g['accuracy_hi'])]
    if ref:
        kc = pd.DataFrame(ref['k_curve'])
        a.plot(kc['k'], kc['accuracy'], 's:', color=INK2, ms=o['marker_size'] - 0.5, label='reference (mean P)')
        rows += [{'panel': 'k_curve', 'series': 'reference', 'aggregator': 'mean_prob', 'k': int(k), 'y': v}
                 for k, v in zip(kc['k'], kc['accuracy'])]
    a.set_xticks(ks)
    a.set_ylim(*o['k_ylim'])
    a.set_xlabel('attempts combined (same person and hand)')
    a.set_ylabel('per-person balanced accuracy')
    a.legend(fontsize=5.4, loc='lower right')
    n_k = int(exi.loc[ship['deep'], 'k_n_persons']) if 'deep' in ship else 0
    a.set_title(f'k attempts ({n_k} persons at every k)', fontsize=7.5)
    b = ax[1]
    role = 'deep' if 'deep' in agj['roles'] else next(iter(agj['roles']))
    block = agj['roles'][role]['calibrated'] or agj['roles'][role]['uncalibrated']
    df = pd.DataFrame(block['rows'])
    for m, g in df.groupby('aggregator', sort=False):
        b.fill_between(g['k'], g['overconfidence_lo'], g['overconfidence_hi'], color=AGG_COLORS[m], alpha=0.12, lw=0)
        b.plot(g['k'], g['overconfidence'], 'o-', color=AGG_COLORS[m], ms=o['marker_size'], label=AGG_SHORT.get(m, m))
        rows += [{'panel': 'overconfidence', 'series': m, 'aggregator': m, 'k': int(k), 'y': v, 'lo': l_, 'hi': h_}
                 for k, v, l_, h_ in zip(g['k'], g['overconfidence'], g['overconfidence_lo'], g['overconfidence_hi'])]
    b.axhline(0, color=INK, lw=0.7)
    b.set_xticks(ks)
    lo_oc = float(np.nanmin(df['overconfidence_lo']))
    hi_oc = float(np.nanmax(df['overconfidence_hi']))
    b.set_ylim(*(o['oc_ylim'] or (lo_oc - 0.35 * (hi_oc - lo_oc), hi_oc + 0.05 * (hi_oc - lo_oc))))
    b.set_xlabel('attempts aggregated')
    b.set_ylabel('mean confidence - accuracy')
    b.legend(fontsize=5.8, loc='lower left')
    b.set_title(f"aggregate calibration, {ROLE_LABELS[role].lower()}", fontsize=7.5)
    kd = rec['k']
    at = df[df['k'] == kd].set_index('aggregator')                 # the decision is read off the plotted data
    ok = [AGG_SHORT.get(m, m) for m in at.index if at.loc[m, 'overconfidence_lo'] <= 0 <= at.loc[m, 'overconfidence_hi']]
    under = [AGG_SHORT.get(m, m) for m in at.index if at.loc[m, 'overconfidence_hi'] < 0]
    over = [AGG_SHORT.get(m, m) for m in at.index if at.loc[m, 'overconfidence_lo'] > 0]
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=('Left: per-person balanced accuracy when k attempts of one person and hand are combined, for the '
                        'shipped lane-m3 finalists (solid: the mean of P(right), the harness definition, with its 95 % '
                        f"person-bootstrap band; dashed: the recommended aggregator, {AGG_SHORT.get(rec['aggregator'], '')}) "
                        'and the reference. Right: over-confidence (mean confidence minus accuracy; > 0 = over-confident) '
                        f"of the {rec['n_compared']} ways to aggregate k attempts with the shipped deep finalist: the mean P(right), the mean "
                        'logit, the sum of logits (attempts treated as independent) and the sum of raw logits scaled by c_k '
                        f"fitted on the training persons of each fold. The aggregator was chosen among {rec['n_compared']} "
                        'on these outer folds (exploratory).'),
               decision=(f"Ask for several attempts and combine them with an aggregator whose confidence stays calibrated "
                         f"(over-confidence CI containing 0 at k = {kd}, deep finalist): {', '.join(ok) or 'none'}"
                         + (f"; under-confident: {', '.join(under)}" if under else '')
                         + (f"; over-confident: {', '.join(over)}" if over else '')
                         + (f"; recommended (rule over every role): {AGG_SHORT.get(rec['aggregator'], rec['aggregator'])}"
                            if rec['aggregator'] else '') + '.'))


# ───────────────────────────── M3_08 ─────────────────────────────


def fig_M3_08_per_person(**kw: Any) -> None:
    name = 'M3_08_per_person'
    o = _o(name, kw)
    ship = _shipped()
    ex, ref = _summary()
    series = {}
    for role, n in ship.items():
        pp = pd.DataFrame(L.load_result(n).per_person)[['person_id', 'accuracy']]
        series[role] = pp.set_index('person_id')['accuracy']
    if ref:
        series['reference'] = pd.DataFrame(ref['per_person']).set_index('person_id')['accuracy']
    df = pd.DataFrame(series)
    df = df.loc[df[next(iter(series))].sort_values().index]
    fig, a = figure(o['width'], o['height'])
    x = np.arange(len(df))
    off = np.linspace(-0.25, 0.25, len(df.columns))
    for col, k in zip(df.columns, off):
        lab = ROLE_LABELS.get(col, 'reference (yaw logreg)')
        a.scatter(x + k, df[col], s=o['marker_size'] ** 2 * 2, color=ROLE_COLORS.get(col, INK2),
                  marker='o' if col in ship else 's', label=lab, zorder=3)
    a.set_xticks(x, df.index, fontsize=6.5, rotation=90)
    lo_y = math.floor((float(np.nanmin(df.to_numpy())) - 0.05) * 10) / 10
    a.set_ylim(*(o['ylim'] or (max(0.0, lo_y), 1.02)))
    a.axhline(0.8, color=GRID, lw=0.8)
    a.set_ylabel('accuracy of the held-out person')
    a.legend(fontsize=5.8, ncol=4, loc='lower right')
    a.grid(axis='x', visible=False)
    tidy = df.reset_index().rename(columns={'index': 'person_id'}).melt(id_vars='person_id', var_name='model',
                                                                       value_name='accuracy')
    L.save_fig(fig, name, data=tidy,
               caption=(f'Accuracy of each of the {len(df)} held-out persons for the shipped lane-m3 finalists '
                        '(calibrated) and the reference (sorted by the deep finalist). Persons are pseudonymous (P01..).'),
               decision='Whether the deep and time-frequency models fail on the same persons as the engineered features.')


# ───────────────────────────── M3_09 ─────────────────────────────


def fig_M3_09_grip_transfer(**kw: Any) -> None:
    name = 'M3_09_grip_transfer'
    o = _o(name, kw)
    if not (L.tab_dir / 'm3_grip_transfer.json').exists():
        return
    gt = pd.DataFrame(L.read_json('m3_grip_transfer')['rows'])
    ex, _ = _summary()
    labels = ex.set_index('name')['label']
    names = list(dict.fromkeys(gt['name']))
    gt['grip'] = gt['level'].astype(str).str.replace('_toward', '', regex=False)
    grips = [g for g in GRIP_MARKERS if g in set(gt['grip'])]
    off = {g: (i - (len(grips) - 1) / 2) * o['dodge'] for i, g in enumerate(grips)}
    pp, _ = _paired()
    gp = pp[(pp['family'] == 'grip') & (pp['subset'] != 'all')].reset_index(drop=True)
    h = o['height'] or max(3.4, 0.34 * (len(names) + len(gp)) + 1.7)
    fig, ax = figure(o['width'], h, 2, 1, gridspec_kw={'height_ratios': (max(len(names), 2), max(len(gp), 2))})
    a = ax[0]
    for i, n in enumerate(names):
        up = n.endswith('_upright')
        for _, r in gt[gt['name'] == n].iterrows():
            yy = i + off[r['grip']]
            seen = (not up) or r['grip'] == 'upright'          # was this grip in the training rows?
            colr = SERIES if seen else UNSEEN_COLOR
            _ci_bars(a, [r['accuracy']], [yy], [r['accuracy_lo']], [r['accuracy_hi']], color=colr, cap=0.06)
            a.scatter([r['accuracy']], [yy], marker=GRIP_MARKERS[r['grip']], s=o['marker_size'] ** 2 * 2.4, color=colr,
                      zorder=3)
    a.axvline(0.5, color=INK2, lw=0.7, ls=':')
    a.set_yticks(range(len(names)), [labels.get(n, n) for n in names], fontsize=o['label_fontsize'])
    a.set_ylim(len(names) - 0.5, -0.5)
    a.set_xlim(*o['xlim'])
    a.set_xticks(o['xticks'])
    a.set_xlabel('accuracy per instructed grip (95 % CI)')
    a.grid(axis='y', visible=False)
    a.set_title('(a) accuracy per grip', fontsize=7.5)
    hs = [Line2D([], [], marker=GRIP_MARKERS[g], ls='', color=INK2, label=g.replace('_', ' ')) for g in grips]
    hs += [Line2D([], [], marker='s', ls='', color=SERIES, label='grip seen in training'),
           Line2D([], [], marker='s', ls='', color=UNSEEN_COLOR, label='grip never seen in training')]
    a.legend(handles=hs, fontsize=5.8, loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, handlelength=1.0,
             borderaxespad=0.0)
    b = ax[1]
    yy = np.arange(len(gp))
    d, lo, hi = gp['diff'] * 100, gp['diff_lo'] * 100, gp['diff_hi'] * 100
    b.axvline(0, color=INK, lw=0.7)
    _ci_bars(b, d, yy, lo, hi)
    b.scatter(d, yy, c=[VERDICT_COLORS.get(v, MUTED) for v in gp['verdict']], s=o['marker_size'] ** 2 * 2, zorder=3)
    b.set_yticks(yy, [f"{r['what']} ({r['subset']} grips)" for _, r in gp.iterrows()], fontsize=o['label_fontsize'])
    b.set_ylim(len(gp) - 0.5, -0.5)
    xl = o['diff_xlim'] or (min(-5.0, float(np.nanmin(lo)) - 2), max(5.0, float(np.nanmax(hi)) + 2))
    b.set_xlim(*xl)
    b.set_xlabel('paired accuracy difference (points, 95 % CI)')
    b.grid(axis='y', visible=False)
    b.set_title('(b) paired differences on rotated / upright grips', fontsize=7.5)
    for i, r in gp.iterrows():
        llv = r.get('verdict_logloss')
        ll_txt = (f"\nlog loss {llv.replace(' log loss', '')}, Holm {_pfmt(r['p_holm_logloss'])}"
                  if llv in ('higher log loss', 'lower log loss') else '')
        b.text(1.01, i, f"Holm {_pfmt(r['p_holm'])}{ll_txt}", transform=b.get_yaxis_transform(), va='center', fontsize=5.6,
               color=INK2)
    rot = gp[(gp['subset'] == 'rotated') & (gp['name'].str.endswith('_upright'))]
    rot_txt = '; '.join(f"{r['what']}: {100 * r['diff']:+.1f} points (accuracy {r['verdict']}; log loss "
                        f"{r['verdict_logloss']})" for _, r in rot.iterrows())
    data = pd.concat([gt[['name', 'grip', 'n', 'n_persons', 'accuracy', 'accuracy_lo', 'accuracy_hi']].assign(panel='per_grip'),
                      gp[['name', 'baseline', 'subset', 'diff', 'diff_lo', 'diff_hi', 'p_holm', 'verdict', 'logloss_diff',
                          'logloss_lo', 'logloss_hi', 'p_holm_logloss', 'verdict_logloss']].assign(panel='paired')],
                     ignore_index=True)
    L.save_fig(fig, name, data=data,
               caption=('(a) Accuracy per instructed grip (leave-one-person-out; the grip only stratifies the result). '
                        'Models "trained on upright grips only" saw no rotated grip in training (training rows chosen by '
                        'the grip detected from the pre-onset up vector); orange: grips they never saw. (b) Paired mean '
                        'per-person accuracy differences on the rotated and on the upright grips, Holm-adjusted within the '
                        'grip family (colour: verdict; right: the Holm-adjusted p, and the per-person log-loss verdict when '
                        'it is significant after the same adjustment). Device-frame channels (all 17) carry evidence tied '
                        'to phone coordinates; the heading-following body frame (gravity + body, 12 channels) is unchanged '
                        'by a rotation of the phone about its screen normal.'),
               decision=f"Grips absent from training, rotated grips: {rot_txt}. Training data must cover the production grips.")


FIGURE_FUNCTIONS = {
    'M3_01_model_overview': fig_M3_01_model_overview,
    'M3_02_tf_sign': fig_M3_02_tf_sign,
    'M3_03_treatments': fig_M3_03_treatments,
    'M3_04_calibration': fig_M3_04_calibration,
    'M3_05_saliency': fig_M3_05_saliency,
    'M3_06_breakdown': fig_M3_06_breakdown,
    'M3_07_attempts': fig_M3_07_attempts,
    'M3_08_per_person': fig_M3_08_per_person,
    'M3_09_grip_transfer': fig_M3_09_grip_transfer,
    'M3_10_vs_engineered': fig_M3_10_vs_engineered,
}


def draw(only: list[str] | None = None) -> list[str]:
    done = []
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
            done.append(name)
    return done
