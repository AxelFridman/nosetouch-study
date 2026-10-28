"""Figures of lane m1, drawn ONLY from saved files (tables/m1/*.json written by the compute step).

One function per figure, named after it; every presentation detail is a keyword argument whose default lives in
``FIGURES`` (a one-line edit), then

    python -m stage2.pipeline figures --lanes m1                      # about a minute, no model is fitted
    python -m stage2.pipeline figures --lanes m1 --only M1_05_model_ladder

Captions and decisions come from ``m1_texts`` (the same builders the numbers step uses, so every number they cite
is in results/m1_numbers.json). Each figure also writes its plotted data as a tidy CSV (tables/m1/<figure>.csv).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stage2.lanes import m1_texts as T
from stage2.pipeline import Lane, selected
from stage2.plotting import (HAND_COLORS, HAND_LABELS, INK, INK2, MUTED, PLATFORM_COLORS, PLATFORM_LABELS, SERIES,
                             figure)

L = Lane('m1')
ODD_COLOR = SERIES          # chiral (mirror-odd) features
EVEN_COLOR = MUTED          # mirror-even features
FOCUS = 'm1_lr_phys6'

FIGURES: dict[str, dict[str, Any]] = {
    'M1_01_feature_catalogue': dict(width='full', row_height=0.112, xlim=(0.25, 1.0), xticks=(0.3, 0.5, 0.7, 0.9),
                                    share_xlim=(0, 1), tick_fontsize=5.6, marker_size=2.6),
    'M1_02_chiral_components': dict(width='full', height=3.8, band_alpha=0.18, show_mirror=True,
                                    legend_loc='outside upper center'),
    'M1_03_key_distributions': dict(width='full', height=3.6, bins=36, clip_quantile=0.01,
                                    legend_loc='outside upper center'),
    'M1_04_channel_sets': dict(width='full', height=3.2, xlim_acc=(0.4, 0.95), xlim_auc=(0.4, 1.0), label_fontsize=6.5),
    'M1_05_model_ladder': dict(width='full', height=4.4, xlim_acc=(0.6, 0.95), xlim_ll=(0.3, 0.75), label_fontsize=6.3,
                               labels={'m1_lr_phys6_win_asinh': 'PHYS6, asinh-compressed tails'}),   # display names
    'M1_06_temporal_filter': dict(width='full', height=2.6, band='q10_q90', ylim=None),
    'M1_07_tree': dict(width='full', height=3.3, box_fontsize=5.4, width_ratios=(3.4, 1)),
    'M1_08_gam_shapes': dict(width='full', height=3.6, band_alpha=0.22),
    'M1_09_per_person': dict(width='full', height=2.8, ylim=(0, 1.05), legend_loc='lower center',
                             legend_anchor=(0.5, 1.0)),
    'M1_10_conditions': dict(width='full', height=4.6, xlim=(0.3, 1.02), min_trials=8, label_fontsize=6.2),
    'M1_11_calibration': dict(width='full', height=2.6, band_alpha=0.2, treated=None,     # None: the small finalist
                              label_fontsize=6.2, text_fontsize=5.4, width_ratios=(1, 1, 1.15),
                              titles={'m1_lr_phys6': 'PHYS6 (RMS scaling)',
                                      'm1_lr_phys6_nested': 'PHYS6, treatment chosen in-fold',
                                      'm1_lr_phys6_win_asinh': 'PHYS6, asinh tails'},
                              treatment_labels={'m1_lr_phys6_win_asinh': 'asinh tails',
                                                'm1_lr_phys6_nested': 'chosen in-fold'}),
    'M1_12_k_attempts': dict(width='full', height=2.8, ylim_acc=(0.6, 1.0), xlabel='k attempts (same person, hand)',
                             legend_loc='outside lower center'),
    'M1_13_data_treatment': dict(width='full', height=3.6, label_fontsize=6.2, xlabel_acc='accuracy minus PHYS6 (points)',
                                 xlabel_ll='log loss minus PHYS6'),
    'M1_14_failures': dict(width='full', height=3.1, legend_loc='outside lower center',
                           titles=('What a confident error looks like (descriptive)',
                                   'Where confident errors concentrate')),
    'M1_15_finalist_calibration': dict(width='full', height=2.6, band_alpha=0.2, text_fontsize=5.4,
                                       titles={'m1_rule_fhy_sign_cal': 'Rule: first-half yaw sign',
                                               'm1_lr_phys6_nested': 'PHYS6 logistic, treatment in-fold',
                                               'm1_lr_phys6_win_asinh': 'PHYS6 logistic, asinh tails',
                                               'm1_gam_phys6': 'GAM on PHYS6'}),
    'M1_16_natural_transfer': dict(width='full', height=3.0, label_fontsize=6.3, legend_loc='outside lower center'),
}


def _rows(name: str, key: str = 'rows') -> pd.DataFrame:
    return pd.DataFrame(L.read_json(name)[key])


def _models() -> pd.DataFrame:
    return _rows('m1_models').set_index('name', drop=False)


def _tint(color: str, frac: float) -> tuple[float, float, float]:
    """``color`` mixed with white (frac = 1: the colour itself), as an opaque RGB triple."""
    from matplotlib.colors import to_rgb
    return tuple(1.0 - frac * (1.0 - c) for c in to_rgb(color))


def _save(fig: Any, name: str, o: dict[str, Any], data: pd.DataFrame | None) -> None:
    cap, dec = T.fig_text(name, T.Book(), o)
    L.save_fig(fig, name, data=data, caption=cap, decision=dec)


# ───────────────────────────── M1_01 catalogue ─────────────────────────────

FAMILY_LABELS = {'yaw': 'yaw about gravity', 'roll': 'roll about the facing axis', 'lateral': 'lateral motion',
                 'dedrift': 'lateral, hold bias removed', 'device': 'device frame (grip-invariant)',
                 'fixed': 'absolute time / lead-in', 'control': 'negative control', 'even': 'even descriptors'}


def fig_M1_01_feature_catalogue(**kw: Any) -> None:
    name = 'M1_01_feature_catalogue'
    o = {**FIGURES[name], **kw}
    cat = L.read_json('m1_catalogue')
    df = pd.DataFrame(cat['rows'])
    order = list(FAMILY_LABELS)
    df['fam_i'] = df['family'].map({f: i for i, f in enumerate(order)})
    df = df.sort_values(['fam_i', 'auc'], ascending=[True, False]).reset_index(drop=True)
    n = len(df)
    gaps = df['fam_i'].diff().fillna(0).ne(0).cumsum()
    ypos = np.arange(n) + 0.8 * gaps.to_numpy()
    fig, ax = figure(o['width'], o['row_height'] * (n + 0.8 * gaps.max()) + 0.7, 1, 2,
                     gridspec_kw={'width_ratios': [3.2, 1]}, sharey=True)
    a = ax[0]
    col = np.where(df['parity'] == 'odd', ODD_COLOR, EVEN_COLOR)
    a.hlines(ypos, df['auc_lo'], df['auc_hi'], color=col, lw=0.8)
    a.scatter(df['auc'], ypos, c=col, s=o['marker_size'] ** 2 * 2, zorder=3)
    a.axvline(0.5, color=INK, lw=0.6)
    a.set_yticks(ypos, df['name'], fontsize=o['tick_fontsize'])
    a.invert_yaxis()
    a.set_xlim(*o['xlim'])
    a.set_xticks(o['xticks'])
    a.set_xlabel('leave-one-person-out AUC (direction fitted on the training persons; 95 % CI)')
    a.grid(axis='y', visible=False)
    for f, g in df.groupby('family', sort=False):
        a.text(o['xlim'][1] - 0.005, ypos[g.index[0]] - 0.75, FAMILY_LABELS.get(f, f), ha='right', va='bottom',
               fontsize=o['tick_fontsize'] + 0.4, color=INK2, style='italic')
    b = ax[1]
    b.barh(ypos, df['share_persons_above_50'], color=col, height=0.7)
    b.axvline(0.5, color=INK, lw=0.6)
    b.set_xlim(*o['share_xlim'])
    b.set_xlabel('share of persons separated\n(within-person AUC > 0.5)')
    b.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], marker='o', ls='', color=ODD_COLOR, label='odd (chiral): flips under the mirror'),
                      Line2D([], [], marker='o', ls='', color=EVEN_COLOR, label='even: identical under the mirror')],
             loc='lower right', fontsize=6)
    _save(fig, name, o, df[['name', 'family', 'parity', 'auc', 'auc_lo', 'auc_hi', 'share_persons_above_50',
                            'within_auc_median']])


# ───────────────────────────── M1_02 chiral components ─────────────────────────────

COMPONENTS = [('yaw_rate', 'yaw rate about gravity', 'deg/s'), ('roll_rate', 'roll rate about the facing axis', 'deg/s'),
              ('acc_lat', 'lateral acceleration', 'm/s$^2$'), ('yaw_cum', 'cumulative yaw', 'deg'),
              ('roll_cum', 'cumulative roll', 'deg'), ('latvel', 'lateral velocity change', 'm/s')]


def fig_M1_02_chiral_components(**kw: Any) -> None:
    name = 'M1_02_chiral_components'
    o = {**FIGURES[name], **kw}
    pc = _rows('m1_physics_curves')
    fig, axs = figure(o['width'], o['height'], 2, 3)
    for a, (ser, title, unit) in zip(axs.ravel(), COMPONENTS):
        g = pc[pc['series'] == ser]
        for hand in ('left', 'right'):
            h = g[g['hand'] == hand]
            a.fill_between(h['tau'] * 100, h['lo'], h['hi'], color=HAND_COLORS[hand], alpha=o['band_alpha'], lw=0)
            a.plot(h['tau'] * 100, h['mean'], color=HAND_COLORS[hand], lw=1.3, label=HAND_LABELS[hand])
        if o['show_mirror']:
            h = g[g['hand'] == 'left_mirrored']
            a.plot(h['tau'] * 100, h['mean'], color=HAND_COLORS['right'], lw=0.9, ls='--', label='mirrored left')
        a.axhline(0, color=INK2, lw=0.5)
        a.set_title(title)
        a.set_ylabel(unit)
        a.set_xticks([0, 25, 50, 75, 100])
    for a in axs[1]:
        a.set_xlabel('time (% of the approach)')
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc=o['legend_loc'], ncol=len(lab), fontsize=6.5, frameon=False)
    _save(fig, name, o, pc)


# ───────────────────────────── M1_03 key distributions ─────────────────────────────


def fig_M1_03_key_distributions(**kw: Any) -> None:
    name = 'M1_03_key_distributions'
    o = {**FIGURES[name], **kw}
    kf = L.read_json('m1_key_features')
    df = pd.DataFrame(kf['rows'])
    cat = pd.DataFrame(L.read_json('m1_catalogue')['rows']).set_index('name')
    feats = kf['features']
    fig, axs = figure(o['width'], o['height'], 2, 3)
    tidy = []
    for a, f in zip(axs.ravel(), feats):
        x = df[f].to_numpy(dtype=float)
        lo, hi = np.quantile(x, [o['clip_quantile'], 1 - o['clip_quantile']])
        bins = np.linspace(lo, hi, o['bins'])
        for hand in ('left', 'right'):
            v = np.clip(df.loc[df['hand'] == hand, f].to_numpy(dtype=float), lo, hi)
            a.hist(v, bins=bins, histtype='stepfilled', alpha=0.25, color=HAND_COLORS[hand])
            a.hist(v, bins=bins, histtype='step', lw=1.0, color=HAND_COLORS[hand], label=HAND_LABELS[hand])
            cnt, _ = np.histogram(v, bins=bins)
            tidy += [{'feature': f, 'hand': hand, 'bin_lo': bins[i], 'bin_hi': bins[i + 1], 'count': int(cnt[i])}
                     for i in range(cnt.size)]
        r = cat.loc[f]
        a.set_title(f'{f} ({r["parity"]}): AUC {r["auc"]:.2f}', fontsize=7.5)
        a.set_xlabel(r['unit'])
        if cat.loc[f, 'parity'] == 'odd':
            a.axvline(0, color=INK2, lw=0.5)
    axs[0, 0].set_ylabel('trials')
    axs[1, 0].set_ylabel('trials')
    h, lab = axs[0, 0].get_legend_handles_labels()
    # an 'outside' figure legend: the constrained layout reserves its space above the panels (no overlap with titles)
    fig.legend(h, lab, loc=o['legend_loc'], ncol=2, fontsize=6.5, frameon=False)
    _save(fig, name, o, pd.DataFrame(tidy))


# ───────────────────────────── forest helpers ─────────────────────────────


def _forest(a, labels, v, lo, hi, color, xlim, xlabel, fontsize, ref=None, marker='o'):
    y = np.arange(len(labels))
    a.hlines(y, lo, hi, color=color, lw=1.0)
    a.scatter(v, y, color=color, s=12, zorder=3, marker=marker)
    if ref is not None:
        a.axvline(ref, color=INK2, lw=0.6, ls='--')
    a.set_yticks(y, labels, fontsize=fontsize)
    if not a.yaxis_inverted():        # shared y axes: invert once (first row on top)
        a.invert_yaxis()
    a.set_xlim(*xlim)
    a.set_xlabel(xlabel)
    a.grid(axis='y', visible=False)


# ───────────────────────────── M1_04 channel sets ─────────────────────────────


def fig_M1_04_channel_sets(**kw: Any) -> None:
    name = 'M1_04_channel_sets'
    o = {**FIGURES[name], **kw}
    mt = _models()
    t = mt.loc[[n for n in T.CHANNEL_ROWS if n in mt.index]].reset_index(drop=True)
    labels = list(t['label'])
    col = [MUTED if g == 'representation' else SERIES for g in t['group']]
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=True)
    _forest(ax[0], labels, t['accuracy'], t['accuracy_lo'], t['accuracy_hi'], col, o['xlim_acc'],
            'accuracy (95 % CI)', o['label_fontsize'], ref=0.5)
    _forest(ax[1], labels, t['roc_auc'], t['roc_auc_lo'], t['roc_auc_hi'], col, o['xlim_auc'], 'AUC (95 % CI)',
            o['label_fontsize'], ref=0.5)
    ax[0].set_title('Accuracy')
    ax[1].set_title('AUC')
    _save(fig, name, o, t[['name', 'label', 'accuracy', 'accuracy_lo', 'accuracy_hi', 'roc_auc', 'roc_auc_lo',
                           'roc_auc_hi', 'log_loss']])


# ───────────────────────────── M1_05 model ladder ─────────────────────────────


def fig_M1_05_model_ladder(**kw: Any) -> None:
    name = 'M1_05_model_ladder'
    o = {**FIGURES[name], **kw}
    mt = _models()
    t = mt.loc[T.ladder_names(mt)].reset_index(drop=True)
    fins = {f['name'] for f in T.finalists()}
    col = [INK if n == 'example_yaw_logreg' else (MUTED if g == 'rules' else SERIES) for n, g in zip(t['name'], t['group'])]
    shown = [o['labels'].get(n, lab) for n, lab in zip(t['name'], t['label'])]
    labels = [f'{lab} *' if n in fins else lab for n, lab in zip(t['name'], shown)]
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=True)
    _forest(ax[0], labels, t['accuracy'], t['accuracy_lo'], t['accuracy_hi'], col, o['xlim_acc'], 'accuracy (95 % CI)',
            o['label_fontsize'])
    hard = t['hard'].astype(bool)
    llv = t['log_loss'].where(~hard)            # a hard 0/1 rule has no finite log loss
    _forest(ax[1], labels, llv, t['log_loss_lo'].where(~hard), t['log_loss_hi'].where(~hard), col,
            o['xlim_ll'], 'log loss (95 % CI; lower is better)', o['label_fontsize'], ref=np.log(2))
    for a, key in ((ax[0], 'accuracy'), (ax[1], 'log_loss')):
        f = t[t['name'].isin(fins)]
        a.scatter(f[key], f.index, marker='D', s=26, facecolor='none', edgecolor=INK, lw=0.8, zorder=4)
    ref = mt.loc['example_yaw_logreg'] if 'example_yaw_logreg' in mt.index else None
    if ref is not None:
        ax[0].axvline(ref['accuracy'], color=INK, lw=0.5, ls=':')
    ax[0].set_title('Interpretable models, leave-one-person-out (* finalist)')
    ax[1].set_title('Probability quality')
    _save(fig, name, o, t[['name', 'label', 'group', 'accuracy', 'accuracy_lo', 'accuracy_hi', 'log_loss',
                           'log_loss_lo', 'log_loss_hi', 'n_configs_tried']])


# ───────────────────────────── M1_06 temporal filter ─────────────────────────────

TC_LABELS = {'yaw_rate': 'yaw rate', 'roll_rate': 'roll rate', 'acc_lat': 'lateral acceleration'}
TC_COLORS = {'yaw_rate': SERIES, 'roll_rate': '#eb6834', 'acc_lat': '#1baf7a'}


def fig_M1_06_temporal_filter(**kw: Any) -> None:
    name = 'M1_06_temporal_filter'
    o = {**FIGURES[name], **kw}
    single = _rows('m1_tf_yaw_profile')
    multi = _rows('m1_tf_multi_profile')
    lo_k, hi_k = ('coef_std_q10', 'coef_std_q90') if o['band'] == 'q10_q90' else ('coef_std_min', 'coef_std_max')
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=False)
    for a, df, title in ((ax[0], single, 'Yaw rate only'), (ax[1], multi, 'Three chiral rates together')):
        for ch, g in df.groupby('channel', sort=False):
            a.fill_between(g['tau'] * 100, g[lo_k], g[hi_k], color=TC_COLORS[ch], alpha=0.2, lw=0)
            a.plot(g['tau'] * 100, g['coef_std_median'], 'o-', ms=2.5, color=TC_COLORS[ch], label=TC_LABELS[ch])
        a.axhline(0, color=INK2, lw=0.5)
        a.set_xlabel('time bin centre (% of the approach)')
        a.set_title(title)
        a.set_xticks([0, 25, 50, 75, 100])
        if o['ylim']:
            a.set_ylim(*o['ylim'])
    ax[0].set_ylabel('weight on the logit\n(per RMS unit of the channel)')
    ax[1].legend(fontsize=6, loc='upper right')
    _save(fig, name, o, pd.concat([single.assign(model='m1_tf_yaw'), multi.assign(model='m1_tf_multi')]))


# ───────────────────────────── M1_07 tree ─────────────────────────────


def fig_M1_07_tree(**kw: Any) -> None:
    name = 'M1_07_tree'
    o = {**FIGURES[name], **kw}
    ts = L.read_json('m1_tree_summary')
    nodes = {n['id']: n for n in ts['nodes_all_rows']}
    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    a.axis('off')
    depth = max(n['depth'] for n in nodes.values())
    order: list[int] = []

    def walk(i: int) -> None:
        n = nodes[i]
        if n['leaf']:
            order.append(i)
            return
        walk(n['left'])
        walk(n['right'])
    walk(0)
    xs: dict[int, float] = {i: k for k, i in enumerate(order)}

    def place(i: int) -> float:
        n = nodes[i]
        if n['leaf']:
            return xs[i]
        xs[i] = 0.5 * (place(n['left']) + place(n['right']))
        return xs[i]
    place(0)
    span = max(len(order) - 1, 1)
    for i, n in nodes.items():
        x, y = xs[i] / span, 1 - n['depth'] / max(depth, 1)
        for c in (n['left'], n['right']):
            if c >= 0:
                a.plot([x, xs[c] / span], [y, 1 - nodes[c]['depth'] / max(depth, 1)], color=MUTED, lw=0.8, zorder=1)
        if n['leaf']:
            txt = f'{n["p_right"]:.2f}\nn={n["n"]}'
            fc = HAND_COLORS['right'] if n['p_right'] >= 0.5 else HAND_COLORS['left']
            strong = abs(n['p_right'] - 0.5) >= 0.25
            # opaque tints (not alpha), so the edges drawn underneath never show through a leaf's text
            a.text(x, y, txt, ha='center', va='center', fontsize=o['box_fontsize'], color='white' if strong else INK,
                   zorder=3, bbox=dict(boxstyle='round,pad=0.3', fc=_tint(fc, 0.9 if strong else 0.3), ec='none'))
        else:
            txt = f'{n["feature"]}\n> {n["threshold"]:.3g}?'
            a.text(x, y, txt, ha='center', va='center', fontsize=o['box_fontsize'], color=INK,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=MUTED))
    a.set_xlim(-0.08, 1.08)
    a.set_ylim(-0.15, 1.12)
    a.set_title(f'Tree on every trial (depth {ts["depth"]}, leaves >= {ts["min_leaf"]}); yes = right branch',
                fontsize=7.5)
    b = ax[1]
    used = pd.Series(ts['features_used_per_fold']).sort_values(ascending=True)
    b.barh(range(len(used)), used.to_numpy(), color=SERIES)
    b.set_yticks(range(len(used)), used.index, fontsize=6)
    b.set_xlabel('splits across the folds')
    b.set_title('Splits in the fold trees', fontsize=7.5)
    b.grid(axis='y', visible=False)
    _save(fig, name, o, pd.DataFrame(ts['nodes_all_rows']))


# ───────────────────────────── M1_08 GAM shapes ─────────────────────────────


def fig_M1_08_gam_shapes(**kw: Any) -> None:
    name = 'M1_08_gam_shapes'
    o = {**FIGURES[name], **kw}
    from stage2.lanes.m1_features import BY_NAME
    gs = L.read_json('m1_gam_shapes')
    df = pd.DataFrame(gs['rows'])
    feats = list(dict.fromkeys(df['feature']))
    fig, axs = figure(o['width'], o['height'], 2, 3)
    for a, f in zip(axs.ravel(), feats):
        g = df[df['feature'] == f]
        a.fill_between(g['x'], g['f_min'], g['f_max'], color=SERIES, alpha=o['band_alpha'], lw=0)
        a.plot(g['x'], g['f_median'], color=SERIES, lw=1.3)
        a.axhline(0, color=INK2, lw=0.5)
        a.axvline(0, color=INK2, lw=0.5)
        a.set_title(f, fontsize=7.5)
        a.set_xlabel(BY_NAME[f].unit if f in BY_NAME else '')
    axs[0, 0].set_ylabel('contribution to the logit')
    axs[1, 0].set_ylabel('contribution to the logit')
    _save(fig, name, o, df)


# ───────────────────────────── M1_09 per person ─────────────────────────────


def fig_M1_09_per_person(**kw: Any) -> None:
    name = 'M1_09_per_person'
    o = {**FIGURES[name], **kw}
    fl = L.read_json('m1_failures')
    pp = pd.DataFrame(fl['per_person']).sort_values('accuracy').reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'])
    x = np.arange(len(pp))
    ax.bar(x, pp['accuracy'], color=[PLATFORM_COLORS.get(p, MUTED) for p in pp['platform']], alpha=0.85)
    ax.scatter(x, pp['acc_left'], marker='_', s=60, color=HAND_COLORS['left'], zorder=3, label='left-hand trials')
    ax.scatter(x, pp['acc_right'], marker='_', s=60, color=HAND_COLORS['right'], zorder=3, label='right-hand trials')
    labels = [f'{p}{"*" if h == "left" else ""}' for p, h in zip(pp['person_id'], pp['handedness'])]
    ax.set_xticks(x, labels, rotation=90, fontsize=6.5)
    ax.set_ylim(*o['ylim'])
    ax.axhline(0.5, color=INK2, lw=0.5)
    ax.set_ylabel('accuracy')
    from matplotlib.patches import Patch
    hs = [Patch(color=PLATFORM_COLORS[k], label=PLATFORM_LABELS[k]) for k in ('android', 'ios')]
    h2, _ = ax.get_legend_handles_labels()
    ax.legend(handles=hs + h2, fontsize=6, loc=o['legend_loc'], bbox_to_anchor=o['legend_anchor'], ncol=4,
              frameon=False)
    ax.grid(axis='x', visible=False)
    _save(fig, name, o, pp[['person_id', 'platform', 'handedness', 'n', 'accuracy', 'acc_left', 'acc_right',
                            'n_confident_errors']])


# ───────────────────────────── M1_10 conditions ─────────────────────────────


def fig_M1_10_conditions(**kw: Any) -> None:
    name = 'M1_10_conditions'
    o = {**FIGURES[name], **kw}
    st = _rows('m1_strata')
    focus = T.focus_name()
    st = st[(st['model'] == focus) & st['stratum'].isin(T.STRATUM_LABELS) & (st['n_trials'] >= o['min_trials'])].copy()
    st['row'] = [f'{T.STRATUM_LABELS[s]}: {lv} (n = {n})' for s, lv, n in zip(st['stratum'], st['level'], st['n_trials'])]
    order = {s: i for i, s in enumerate(T.STRATUM_LABELS)}
    # levels by accuracy, except the ordinal duration quartiles (in their own order)
    st['_k'] = np.where(st['stratum'] == 'duration_bin', pd.to_numeric(st['level_index'], errors='coerce'), -st['accuracy'])
    st['_s'] = st['stratum'].map(order)
    st = st.sort_values(['_s', '_k']).drop(columns=['_s', '_k']).reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'])
    gap = st['stratum'].ne(st['stratum'].shift()).cumsum().to_numpy()
    y = np.arange(len(st)) + 0.6 * gap
    has_ci = st['accuracy_lo'].notna()
    ax.hlines(y[has_ci], st.loc[has_ci, 'accuracy_lo'], st.loc[has_ci, 'accuracy_hi'], color=SERIES, lw=1.0)
    ax.scatter(st['accuracy'], y, color=np.where(has_ci, SERIES, MUTED), s=10, zorder=3)
    overall = _models().loc[focus, 'accuracy']
    ax.axvline(overall, color=INK, lw=0.6, ls='--')
    ax.axvline(0.5, color=INK2, lw=0.5)
    ax.set_yticks(y, st['row'], fontsize=o['label_fontsize'])
    ax.invert_yaxis()
    ax.set_xlim(*o['xlim'])
    ax.set_xlabel('accuracy (95 % person-bootstrap CI; grey: too few persons for a CI); dashed: overall')
    ax.grid(axis='y', visible=False)
    _save(fig, name, o, st.drop(columns=['row']))


# ───────────────────────────── M1_11 calibration ─────────────────────────────


def _reliability(a, r: pd.DataFrame, title: str, alpha: float) -> None:
    a.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls='--')
    a.fill_between(r['p_mean'], r['y_lo'], r['y_hi'], color=SERIES, alpha=alpha, lw=0)
    a.plot(r['p_mean'], r['y_mean'], 'o-', color=SERIES, ms=3)
    out = r[r['outside'].astype(bool)]
    a.scatter(out['p_mean'], out['y_mean'], s=30, facecolor='none', edgecolor=INK, lw=0.8, zorder=4)
    a.set_title(title, fontsize=7.5)
    a.set_xlabel('predicted P(right)')
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.set_box_aspect(1)


def fig_M1_11_calibration(**kw: Any) -> None:
    name = 'M1_11_calibration'
    o = {**FIGURES[name], **kw}
    cal = L.read_json('m1_calibration')
    rel = pd.DataFrame(cal['reliability'])
    met = pd.DataFrame(cal['metrics'])
    pc = pd.DataFrame(cal['paired'])
    mt = _models()
    treated = o['treated'] or T.slot_names()['small']
    fig, ax = figure(o['width'], o['height'], 1, 3, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    for a, m in zip(ax[:2], (FOCUS, treated)):
        r = rel[rel['model'] == m].sort_values('p_mean')
        q = met[(met['model'] == m) & (met['scope'] == 'all')].iloc[0]
        _reliability(a, r, o['titles'].get(m, T.label(mt, m)), o['band_alpha'])
        # bottom right: the empty corner of a reliability diagram (high P, low observed share)
        a.text(0.97, 0.03, f'ECE {q["ece"]:.3f}\nnull 95 %: {q["ece_null_q95"]:.3f}\nslope {q["cox_slope"]:.2f} '
                           f'[{q["cox_slope_lo"]:.2f}, {q["cox_slope_hi"]:.2f}]\n{int(q["bins_outside"])} bins outside band',
               fontsize=o['text_fontsize'], ha='right', va='bottom', transform=a.transAxes)
    ax[0].set_ylabel('observed share of right hands')
    b = ax[2]
    sub = pc[(pc['baseline'] == FOCUS)].copy()
    sub['lab'] = [('treatment: ' + o['treatment_labels'].get(m, T.label(mt, m))) if k == 'treatment'
                  else ('post-hoc: ' + T._remedy(m)) for m, k in zip(sub['model'], sub['kind'])]
    sub = sub.sort_values(['kind', 'll_diff']).reset_index(drop=True)
    colr = [SERIES if k == 'treatment' else MUTED for k in sub['kind']]
    lim = max(0.06, float(np.nanmax(np.abs(sub[['ll_diff_lo', 'll_diff_hi']].to_numpy()))) * 1.15)
    _forest(b, list(sub['lab']), sub['ll_diff'], sub['ll_diff_lo'], sub['ll_diff_hi'], colr, (-lim, lim),
            'log loss minus PHYS6\n(paired 95 % CI)', o['label_fontsize'], ref=0)
    b.set_title('Remedies against PHYS6', fontsize=7.5)
    _save(fig, name, o, pd.concat([rel[rel['model'].isin([FOCUS, treated])].assign(panel='reliability'),
                                   sub.drop(columns=['lab']).assign(panel='remedies')]))


# ───────────────────────────── M1_12 k attempts ─────────────────────────────

AGG_LABELS = {'mean_p': 'mean of P', 'mean_logit': 'mean of logits', 'sum_logit': 'sum of logits',
              'fitted_alpha': 'k^(-alpha) x sum (alpha fitted)'}
AGG_COLORS = {'mean_p': MUTED, 'mean_logit': SERIES, 'sum_logit': '#eb6834', 'fitted_alpha': '#1baf7a'}


def fig_M1_12_k_attempts(**kw: Any) -> None:
    name = 'M1_12_k_attempts'
    o = {**FIGURES[name], **kw}
    ka = _rows('m1_k_aggregation')
    g = ka[ka['model'] == T.agg_model(ka)]
    fig, ax = figure(o['width'], o['height'], 1, 3)
    for meth, gg in g.groupby('method', sort=False):
        for a, key in zip(ax, ('accuracy', 'ece', 'log_loss')):
            if key != 'ece':
                a.fill_between(gg['k'], gg[f'{key}_lo'], gg[f'{key}_hi'], color=AGG_COLORS[meth], alpha=0.12, lw=0)
            a.plot(gg['k'], gg[key], 'o-', ms=2.5, color=AGG_COLORS[meth], label=AGG_LABELS[meth])
    ax[0].set_ylim(*o['ylim_acc'])
    ax[0].set_ylabel('accuracy of the aggregate')
    ax[1].set_ylabel('ECE of the aggregate')
    ax[2].set_ylabel('log loss of the aggregate')
    for a in ax:
        a.set_xlabel(o['xlabel'])
        a.set_xticks(sorted(g['k'].unique()))
    h, lab = ax[2].get_legend_handles_labels()
    fig.legend(h, lab, loc=o['legend_loc'], ncol=len(lab), fontsize=6, frameon=False)
    _save(fig, name, o, g)


# ───────────────────────────── M1_13 data treatment ─────────────────────────────


def fig_M1_13_data_treatment(**kw: Any) -> None:
    name = 'M1_13_data_treatment'
    o = {**FIGURES[name], **kw}
    mt = _models()
    t = mt.loc[[n for n in T.TREATMENT_ROWS if n in mt.index and mt.loc[n, 'baseline'] == FOCUS]].reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=True)
    labels = list(t['label'])
    _forest(ax[0], labels, 100 * t['vs_base_acc_diff'], 100 * t['vs_base_acc_diff_lo'], 100 * t['vs_base_acc_diff_hi'],
            SERIES, (min(-8, 100 * t['vs_base_acc_diff_lo'].min() - 1), max(8, 100 * t['vs_base_acc_diff_hi'].max() + 1)),
            o['xlabel_acc'], o['label_fontsize'], ref=0)
    _forest(ax[1], labels, t['vs_base_ll_diff'], t['vs_base_ll_diff_lo'], t['vs_base_ll_diff_hi'], SERIES,
            (min(-0.08, t['vs_base_ll_diff_lo'].min() - 0.01), max(0.08, t['vs_base_ll_diff_hi'].max() + 0.01)),
            o['xlabel_ll'], o['label_fontsize'], ref=0)
    for i, r in t.iterrows():
        ax[1].text(1.0, i, f' Holm p = {r["vs_base_p_ll_holm"]:.2g}', transform=ax[1].get_yaxis_transform(),
                   fontsize=5.6, va='center', color=INK2, clip_on=False)
    ax[0].set_title('Data treatment and model form against PHYS6')
    _save(fig, name, o, t[['name', 'label', 'vs_base_acc_diff', 'vs_base_acc_diff_lo', 'vs_base_acc_diff_hi',
                           'vs_base_p_acc', 'vs_base_p_acc_holm', 'vs_base_ll_diff', 'vs_base_ll_diff_lo',
                           'vs_base_ll_diff_hi', 'vs_base_p_ll', 'vs_base_p_ll_holm']])


# ───────────────────────────── M1_14 failures ─────────────────────────────


def fig_M1_14_failures(**kw: Any) -> None:
    name = 'M1_14_failures'
    o = {**FIGURES[name], **kw}
    fl = L.read_json('m1_failures')
    cu = pd.DataFrame(fl['curves'])
    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': [1.3, 1]})
    a = ax[0]
    for (hand, grp), g in cu.groupby(['hand', 'group'], sort=True):
        ls = '-' if grp == 'correct' else '--'
        a.plot(g['tau'] * 100, g['yaw_rate'], ls=ls, color=HAND_COLORS[hand], lw=1.2,
               label=f'{HAND_LABELS[hand]}, {"correct" if grp == "correct" else "confident error"} (n = {int(g["n"].iloc[0])})')
    a.axhline(0, color=INK2, lw=0.5)
    a.set_xlabel('time (% of the approach)')
    a.set_ylabel('mean yaw rate (deg/s)')
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc=o['legend_loc'], ncol=2, fontsize=5.8, frameon=False)
    a.set_title(o['titles'][0], fontsize=7.5)
    b = ax[1]
    ov = pd.DataFrame(fl['overrepresentation'])
    ov = ov[ov['stratum'].isin(['instructed_rotation', 'speed', 'platform', 'hand', 'handedness'])]
    ov = ov.sort_values(['stratum', 'level']).reset_index(drop=True)
    yy = np.arange(len(ov))
    b.barh(yy, ov['ratio'], color=SERIES)
    b.axvline(1, color=INK, lw=0.6)
    for i, r in ov.iterrows():
        b.text(r['ratio'], i, f' {int(r["n_confident_errors"])}/{int(fl["n_confident_errors"])}', va='center',
               fontsize=5.4, color=INK2)
    b.set_yticks(yy, [f'{T.STRATUM_LABELS.get(s, s)}: {lv}' for s, lv in zip(ov['stratum'], ov['level'])], fontsize=6)
    b.invert_yaxis()
    b.set_xlim(0, max(1.5, float(ov['ratio'].max()) * 1.35))
    b.set_xlabel('share of confident errors / share of trials\n(text: confident errors in the level / all)')
    b.grid(axis='y', visible=False)
    b.set_title(o['titles'][1], fontsize=7.5)
    _save(fig, name, o, pd.concat([cu.assign(panel='curves'), ov.assign(panel='overrepresentation')]))


# ───────────────────────────── M1_15 finalist calibration ─────────────────────────────


def fig_M1_15_finalist_calibration(**kw: Any) -> None:
    name = 'M1_15_finalist_calibration'
    o = {**FIGURES[name], **kw}
    fins = [f['name'] for f in T.finalists()]
    cal = L.read_json('m1_calibration')
    rel = pd.DataFrame(cal['reliability'])
    met = pd.DataFrame(cal['metrics'])
    mt = _models()
    fins = [f for f in fins if f in set(rel['model'])]
    fig, ax = figure(o['width'], o['height'], 1, len(fins), squeeze=False)
    plotted = []
    for a, m in zip(ax[0], fins):
        r = rel[rel['model'] == m].sort_values('p_mean')
        two_valued = m in mt.index and mt.loc[m, 'group'] == 'rules'
        a.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls='--')
        if two_valued:
            # P is q or 1 - q with q fitted per fold: the equal-mass bins would only sort the folds by q. Show the two
            # decision groups instead (the bins partition the trials, so the n-weighted means are exact).
            g = (r.assign(side=np.where(r['p_mean'] >= 0.5, 'says right', 'says left'), pn=r['p_mean'] * r['n'],
                          yn=r['y_mean'] * r['n'])
                 .groupby('side', sort=True)[['pn', 'yn', 'n']].sum())
            g = g.assign(p_mean=g['pn'] / g['n'], y_mean=g['yn'] / g['n']).reset_index()
            a.plot(g['p_mean'], g['y_mean'], 'o', color=SERIES, ms=4.5)
            for _, gr in g.iterrows():
                right = gr['side'] == 'says right'
                a.annotate(f'{gr["side"]}\nn = {int(gr["n"])}', (gr['p_mean'], gr['y_mean']), textcoords='offset points',
                           xytext=(7, -4) if right else (-7, 6), fontsize=o['text_fontsize'], color=INK2,
                           ha='left' if right else 'right', va='top' if right else 'bottom')
            plotted.append(g.assign(model=m)[['model', 'side', 'p_mean', 'y_mean', 'n']])
        else:
            a.fill_between(r['p_mean'], r['y_lo'], r['y_hi'], color=SERIES, alpha=o['band_alpha'], lw=0)
            a.plot(r['p_mean'], r['y_mean'], 'o-', color=SERIES, ms=3)
            out = r[r['outside'].astype(bool)]
            a.scatter(out['p_mean'], out['y_mean'], s=30, facecolor='none', edgecolor=INK, lw=0.8, zorder=4)
            plotted.append(r.assign(side='bin'))
        q = met[met['model'] == m].set_index('scope')
        al = q.loc['all']
        plat = ', '.join(f'{PLATFORM_LABELS.get(s, s)} {q.loc[s, "ece"]:.3f}' for s in q.index if s != 'all')
        a.text(0.97, 0.03, f'ECE {al["ece"]:.3f} (null 95 %: {al["ece_null_q95"]:.3f})\nslope {al["cox_slope"]:.2f} '
                           f'[{al["cox_slope_lo"]:.2f}, {al["cox_slope_hi"]:.2f}]\nBrier {al["brier"]:.3f}, log loss '
                           f'{al["log_loss"]:.3f}\nECE {plat}',
               fontsize=o['text_fontsize'], ha='right', va='bottom', transform=a.transAxes)
        a.set_title(o['titles'].get(m, T.label(mt, m)), fontsize=7.5)
        a.set_xlabel(f'predicted P(right) ({"group" if two_valued else "bin"} mean)')
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_aspect('equal')
    ax[0, 0].set_ylabel('observed share of right hands')
    _save(fig, name, o, pd.concat(plotted, ignore_index=True))


# ───────────────────────────── M1_16 natural-use transfer ─────────────────────────────


def fig_M1_16_natural_transfer(**kw: Any) -> None:
    name = 'M1_16_natural_transfer'
    o = {**FIGURES[name], **kw}
    nj = L.read_json('m1_natural')
    nt = pd.DataFrame(nj['transfer']).reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': [1.3, 1]})
    a = ax[0]
    y = np.arange(len(nt))
    a.hlines(y, nt['q05'], nt['q95'], color=MUTED, lw=3, alpha=0.6)
    a.scatter(nt['expected'], y, marker='|', s=60, color=INK2, zorder=3)
    sig = nt['p_two_sided_holm'] < T.ALPHA
    a.scatter(nt['observed'], y, color=np.where(sig, HAND_COLORS['left'], SERIES), s=18, zorder=4)
    a.set_yticks(y, nt['feature'], fontsize=o['label_fontsize'])
    a.invert_yaxis()
    a.set_xlim(0, float(nt['n'].max()) + 0.5)
    from matplotlib.ticker import MaxNLocator
    a.xaxis.set_major_locator(MaxNLocator(integer=True))          # counts of trials
    a.set_xlabel('free-choice trials whose sign points to the reported hand')
    a.grid(axis='y', visible=False)
    a.set_title('Sign transfer per feature', fontsize=7.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=3, alpha=0.6, label='expected from the instructed trials (95 % range)'),
                        Line2D([], [], color=INK2, marker='|', ls='', ms=7, label='expected count'),
                        Line2D([], [], color=SERIES, marker='o', ls='', ms=4, label='observed, free choice'),
                        Line2D([], [], color=HAND_COLORS['left'], marker='o', ls='', ms=4,
                               label='observed, lower than expected after Holm')],
               loc=o['legend_loc'], ncol=4, fontsize=5.6, frameon=False)
    b = ax[1]
    cons = [c for c in nj.get('contributions', []) if c['model'] == T.slot_names()['small']] or nj.get('contributions', [])[:1]
    tidy = [nt.assign(panel='transfer')]
    if cons:
        cd = pd.DataFrame(cons[0]['rows'])
        x = np.arange(len(cd))
        b.bar(x - 0.18, cd['primary_mean'], width=0.36, color=MUTED, label='instructed trials (same persons, hand)')
        b.bar(x + 0.18, cd['natural_mean'], width=0.36, color=SERIES, label='free-choice trials')
        b.set_xticks(x, cd['component'], fontsize=6.5)
        b.axhline(0, color=INK2, lw=0.5)
        b.set_ylabel('logit contribution towards\nthe reported hand')
        b.legend(fontsize=5.6, loc='upper right')
        b.set_title(f'{T.label(_models(), cons[0]["model"])}', fontsize=7.5)
        tidy.append(cd.assign(panel='contributions'))
    _save(fig, name, o, pd.concat(tidy, ignore_index=True))


FIGURE_FUNCTIONS = {name: globals()[f'fig_{name}'] for name in FIGURES}


def draw(only: list[str] | None = None) -> list[str]:
    done = []
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
            done.append(name)
    return done
