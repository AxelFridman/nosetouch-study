"""Figures of lane m2, drawn ONLY from the saved derived files in tables/m2/ (written by the compute step).

One function per figure, named after it; every presentation detail is a keyword argument whose default lives in
``FIGURES`` below, so a change is a one-line edit followed by

    python -m stage2.pipeline figures --lanes m2                       # seconds, no model is fitted
    python -m stage2.pipeline figures --lanes m2 --only M2_05_learning_curve

Each function also writes its plotted data as a tidy CSV (tables/m2/<figure name>.csv). Captions and decisions are
generated from the saved results: every number in them is printed through ``m2_report.fmt`` from the entries of the
numbers file (``m2_report.number_values``), and every judgement (better, worse, identical, adopted) is computed.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from stage2.lanes import m2_pipeline as P
from stage2.lanes import m2_report as R
from stage2.pipeline import Lane, selected
from stage2.plotting import INK, INK2, MUTED, PLATFORM_COLORS, PLATFORM_LABELS, SERIES, figure

L = Lane('m2')
GREY = '#9a9890'
ACCENT = '#eb6834'
GREEN = '#1baf7a'
VIOLET = '#8a5cd0'
CAL_COLORS = {'none': INK2, 'platt': GREEN, 'temperature': SERIES, 'isotonic': ACCENT, 'beta': VIOLET,
              'platt_platform': '#b8860b'}
AGG_COLORS = {'mean_p': SERIES, 'mean_logit': GREEN, 'sum_logit': ACCENT, 'sum_logit_tempered': VIOLET}
AGG_LABELS = {'mean_p': 'mean P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits',
              'sum_logit_tempered': 'sum of logits / nested tau_k'}
MODEL_COLORS = (ACCENT, SERIES, GREEN, VIOLET)
#: compact names for tick labels
TICK = {'m2_select': 'selection', 'm2_lr_full': 'L2 LR', 'm2_lgbm': 'LightGBM', 'm2_ens_stack': 'stack',
        'm2_ens_soft': 'soft vote', 'm2_hgb': 'HGB', 'm2_svm_rbf': 'RBF SVM'}
#: short display names of experiments in legends and panel titles (anything else: its label)
DISPLAY = {'m2_select': 'nested selection', 'm2_lr_full': 'L2 logistic regression (default)', 'm2_lgbm': 'LightGBM',
           'm2_ens_stack': 'stack (LR + SVM + LGBM)', 'm2_ens_soft': 'soft vote (LR + SVM + LGBM)',
           'm2_svm_rbf': 'RBF SVM', 'm2_hgb': 'HistGradientBoosting', 'm2_et': 'extra trees', 'm2_rf': 'random forest'}
LEVEL_TEXT = {'android': 'Android', 'ios': 'iOS'}
#: the nested selection's candidates in captions
SELECT_TEXT = {'lr_l2': 'L2 LR', 'svm_rbf': 'RBF SVM', 'lgbm': 'LightGBM', 'ens_soft': 'their soft vote'}

#: per-figure presentation defaults (edit here, or pass keyword arguments)
FIGURES: dict[str, dict[str, Any]] = {
    'M2_01_models': dict(width='full', height=3.4, acc_xlim=None, ll_xlim=None, marker_size=3.5,
                         acc_xlabel='accuracy (95 % CI)', ll_xlabel='log loss (95 % CI)', label_fontsize=6.5),
    'M2_02_data_treatment': dict(width='full', height=3.6, acc_xlim=None, ll_xlim=None, marker_size=3.2,
                                 acc_xlabel='accuracy minus default (points)', ll_xlabel='log loss minus default',
                                 label_fontsize=6.3),
    'M2_03_families': dict(width='full', height=4.4, marker_size=3.0, drop_xlim=None, only_xlim=None,
                           drop_xlabel='log loss minus all families (paired 95 % CI)',
                           only_xlabel='accuracy with one family only (95 % CI)', label_fontsize=6.3),
    'M2_04_mirror': dict(width='full', height=2.5, marker_size=3.5, acc_ylim=None, ll_ylim=None, hand_ylim=None),
    'M2_05_learning_curve': dict(width='full', height=2.5, acc_ylim=None, ll_ylim=None, marker_size=3.0,
                                 xlabel='training persons per outer fold', band_alpha=0.15, legend_loc='upper left'),
    'M2_06_importance': dict(width='full', height=2.9, n_channels=10, xlabel='share of held-out |contribution|'),
    'M2_07_calibration': dict(width='full', height=2.0, lim=(0.0, 1.0), marker_size=2.6, band_alpha=0.18,
                              text_fontsize=5.6),
    'M2_08_calibration_platform': dict(width='full', height=2.6, marker_size=3.2, slope_ylim=None,
                                       ece_ylabel='ECE (grey: null 95 % band)',
                                       pp_ylabel='log loss: per-platform - global Platt'),
    'M2_09_breakdown': dict(width='full', height=3.8, xlim=None, marker_size=3.2, label_fontsize=6.3,
                            factors=('platform', 'orientation', 'speed', 'variant', 'handedness', 'gate_pass')),
    'M2_10_persons': dict(width='full', height=2.4, ylim=(0.0, 1.08), bar_width=0.38, line=0.8),
    'M2_11_k_attempts': dict(width='full', height=2.5, acc_ylim=None, ll_ylim=None, gap_ylim=None, marker_size=3.0),
    'M2_12_calibration_remedies': dict(width='full', height=2.4, marker_size=3.2),
}

_NUM: dict[str, Any] = {}


# ───────────────────────────── loading ─────────────────────────────


def nums() -> dict[str, Any]:
    """The numbers entries (built once per draw from the saved files)."""
    if not _NUM:
        _NUM.update(R.number_values())
    return _NUM


def v(key: str) -> str:
    """Number ``key`` as its macro prints it."""
    return R.fmt(nums(), key)


def pe(key: str) -> str:
    """A p-value number as a relation: '= 0.042' or '< 0.001'."""
    return R.p_is(nums(), key)


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def ci(prefix: str, lo: str = '_lo', hi: str = '_hi') -> str:
    return f'[{v(prefix + lo)}, {v(prefix + hi)}]'


def has(*keys: str) -> bool:
    return all(k in nums() for k in keys)


def _exp() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('m2_experiments')['rows'])


def _summary() -> dict[str, Any]:
    return L.read_json('m2_summary')


def _csv(name: str) -> pd.DataFrame:
    p = L.tab_dir / f'{name}.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p}: run python -m stage2.pipeline compute --lanes m2')
    return pd.read_csv(p)


def disp(name: str, label: str | None = None) -> str:
    if name in DISPLAY:
        return DISPLAY[name]
    return (label or name).replace(' (pre-specified)', '')


class MissingData(RuntimeError):
    """The saved inputs of a figure do not exist yet (its experiments have not been computed)."""


def _dot_ci(ax: Any, x: Any, y: Any, lo: Any, hi: Any, color: str, ms: float, **kw: Any) -> None:
    """Horizontal point + interval (a segment: a bootstrap CI need not be symmetric)."""
    x, y, lo, hi = (np.atleast_1d(np.asarray(z, float)) for z in (x, y, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.hlines(y[ok], lo[ok], hi[ok], color=color, lw=0.9)
    ax.plot(x, y, 'o', color=color, ms=ms, **kw)


def _vci(ax: Any, x: Any, y: Any, lo: Any, hi: Any, color: str, ms: float, marker: str = 'o', **kw: Any) -> None:
    """Vertical point + interval (see _dot_ci)."""
    x, y, lo, hi = (np.atleast_1d(np.asarray(z, float)) for z in (x, y, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.vlines(x[ok], lo[ok], hi[ok], color=color, lw=0.9)
    ax.plot(x, y, marker, color=color, ms=ms, **kw)


def _pad(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
    span = max(hi - lo, 1e-9)
    return lo - frac * span, hi + frac * span


def _sig_text(s: str, metric: str = 'll') -> str:
    """'better / worse / not different / identical' of an experiment against its baseline, with the Holm p."""
    verdict = v(f'{s}_{metric}_verdict')
    if verdict == 'identical':
        return f'identical (max |dP| {v(f"{s}_max_dp")})'
    p = pe(f'{s}_p_{metric}_holm') if has(f'{s}_p_{metric}_holm') else '= n/a'
    return f'{verdict}, Holm p {p}'


# ───────────────────────────── figures ─────────────────────────────


def fig_M2_01_models(**kw: Any) -> None:
    name = 'M2_01_models'
    o = {**FIGURES[name], **kw}
    t = _exp()
    s = _summary()
    t = t[t['group'].isin(['default', 'model', 'select'])].sort_values(['log_loss', 'name'], ascending=[False, True])
    t = t.reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=True)
    y = np.arange(len(t))
    colors = [ACCENT if g == 'select' else (SERIES if g == 'default' else INK2) for g in t['group']]
    for i, r in t.iterrows():
        _dot_ci(ax[0], [r['accuracy']], [i], [r['accuracy_lo']], [r['accuracy_hi']], colors[i], o['marker_size'])
        _dot_ci(ax[1], [r['log_loss']], [i], [r['log_loss_lo']], [r['log_loss_hi']], colors[i], o['marker_size'])
    if has('ref_accuracy', 'ref_log_loss'):
        ref = s.get('reference_metrics') or {}
        ax[0].axvline(ref['accuracy'], color=MUTED, lw=0.8, ls='--')
        ax[0].text(ref['accuracy'], -0.55, f" reference {v('ref_accuracy')}", fontsize=6, color=MUTED, ha='left',
                   va='center')
        ax[1].axvline(ref['log_loss'], color=MUTED, lw=0.8, ls='--')
    labels = [disp(r['name'], r['label']) + ('*' if r['model'] in ('rf', 'et') else '') for _, r in t.iterrows()]
    ax[0].set_yticks(y, labels, fontsize=o['label_fontsize'])
    ax[0].set_xlabel(o['acc_xlabel'])
    ax[1].set_xlabel(o['ll_xlabel'])
    if o['acc_xlim']:
        ax[0].set_xlim(*o['acc_xlim'])
    if o['ll_xlim']:
        ax[1].set_xlim(*o['ll_xlim'])
    for a in ax:
        a.grid(axis='y', visible=False)
    ax[0].set_ylim(-0.8, len(t) - 0.3)
    best = s.get('best_by_log_loss')
    bs = R.short(best) if best else None
    sel = 'select'
    lo_name = t.sort_values('accuracy').iloc[0]['name']
    hi_name = t.sort_values('accuracy').iloc[-1]['name']
    dec = [f"Lowest outer log loss: {v('best_model')} ({v(f'{bs}_accuracy')}, log loss {v(f'{bs}_log_loss')})"]
    if bs and has(f'{bs}_d_ll'):
        dec.append(f"; against the default its log loss differs by {v(f'{bs}_d_ll')} {ci(f'{bs}_d_ll')}, "
                   f"{_sig_text(bs)} over the {v('fam_model_vs_default_size')} comparisons with the default")
    dec.append(f". Accuracies span {v(R.short(lo_name) + '_accuracy')} to {v(R.short(hi_name) + '_accuracy')} over "
               f"{v('n_models_compared')} pipelines plus the selection. The honest headline is the nested selection: "
               f"{v(f'{sel}_accuracy')} {ci(f'{sel}_accuracy')}, log loss {v(f'{sel}_log_loss')} (one procedure, no "
               'outer-fold choice); picking the best of the pipelines on the outer folds is exploratory.')
    L.save_fig(fig, name, data=t[['name', 'label', 'group', 'accuracy', 'accuracy_lo', 'accuracy_hi', 'log_loss',
                                  'log_loss_lo', 'log_loss_hi', 'roc_auc', 'ece', 'natural_use_accuracy',
                                  'n_configs_tried']],
               caption=(f"Classical models on the same {v('n_features')} per-trial features (approach window), "
                        f"leave-one-person-out over {v('n_persons')} persons; hyperparameters tuned inside every outer "
                        f"fold on {v('inner_folds')} inner person folds (* fixed hyperparameters, not tuned), "
                        'mirror-augmented. Blue: the pre-specified default; orange: the nested selection among '
                        + ' / '.join(SELECT_TEXT.get(c, c) for c in P.SELECT_CANDIDATES)
                        + '; dashed: the reference floor (yaw logistic regression). Sorted by log loss (best at the '
                          'top).'),
               decision=''.join(dec))


def _forest(ax: Any, t: pd.DataFrame, col: str, o: dict[str, Any], xlim: Any, xlabel: str, scale: float = 100.0,
            labels: bool = True) -> None:
    y = np.arange(len(t))
    d, lo, hi = t[f'vs_base_{col}'] * scale, t[f'vs_base_{col}_lo'] * scale, t[f'vs_base_{col}_hi'] * scale
    metric = 'acc' if col == 'diff_acc' else 'll'
    sig = t.get(f'vs_base_sig_{metric}', pd.Series(False, index=t.index)).fillna(False).astype(bool)
    ident = t.get('vs_base_identical', pd.Series(False, index=t.index)).fillna(False).astype(bool)
    for i in range(len(t)):
        c = ACCENT if sig.iloc[i] else INK2
        _dot_ci(ax, [d.iloc[i]], [i], [lo.iloc[i]], [hi.iloc[i]], c, o['marker_size'],
                mfc='white' if ident.iloc[i] else c)
    ax.axvline(0, color=INK, lw=0.7)
    if labels:
        ax.set_yticks(y, t['row_label'], fontsize=o['label_fontsize'])
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)


def fig_M2_02_data_treatment(**kw: Any) -> None:
    name = 'M2_02_data_treatment'
    o = {**FIGURES[name], **kw}
    t = _exp()
    order = ['window', 'frame', 'bins', 'scaling', 'mirror']
    t = t[t['group'].isin(order) & (t['model'] == 'lr_l2') & t['vs_base_diff_acc'].notna()].copy()
    if t.empty:
        raise MissingData('data-treatment results missing')
    t['g'] = t['group'].map({g: i for i, g in enumerate(order)})
    t['row_label'] = [str(r['label']).replace('Logistic regression (L2), ', '') for _, r in t.iterrows()]
    t = t.sort_values(['g', 'name'], ascending=[False, False]).reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=True)
    _forest(ax[0], t, 'diff_acc', o, o['acc_xlim'], o['acc_xlabel'])
    _forest(ax[1], t, 'diff_ll', o, o['ll_xlim'], o['ll_xlabel'], scale=1.0, labels=False)
    for i in range(1, len(t)):
        if t['g'].iloc[i] != t['g'].iloc[i - 1]:
            for a in ax:
                a.axhline(i - 0.5, color='#e1e0d9', lw=0.6)
    worse = [r for _, r in t.iterrows() if v(f"{R.short(r['name'])}_ll_verdict") == 'worse']
    better = [r for _, r in t.iterrows() if v(f"{R.short(r['name'])}_ll_verdict") == 'better']
    ident = [r for _, r in t.iterrows() if v(f"{R.short(r['name'])}_ll_verdict") == 'identical']
    acc_diff = [r for _, r in t.iterrows() if v(f"{R.short(r['name'])}_acc_verdict") in ('better', 'worse')]

    def item(r: pd.Series) -> str:
        s = R.short(r['name'])
        return f"{r['row_label']} ({v(s + '_d_ll')} {ci(s + '_d_ll')}, Holm p {pe(s + '_p_ll_holm')})"
    dec = (f"Within the family of {v('fam_data_treatment_size')} data-treatment comparisons (Holm), "
           + (f"worse log loss than the default: {'; '.join(item(r) for r in worse)}. " if worse else
              'no variant has a worse log loss than the default. ')
           + (f"Better: {'; '.join(item(r) for r in better)}. " if better else 'None is better. ')
           + (f"Accuracy differs for: {', '.join(r['row_label'] for r in acc_diff)}. " if acc_diff else
              'No accuracy difference survives the adjustment. ')
           + (f"Identical to the default within solver precision: {', '.join(r['row_label'] for r in ident)}. "
              if ident else '')
           + 'The default treatment is kept unless a variant is better after the adjustment.')
    L.save_fig(fig, name, data=t[['name', 'group', 'row_label', 'accuracy', 'vs_base_diff_acc', 'vs_base_diff_acc_lo',
                                  'vs_base_diff_acc_hi', 'vs_base_p_acc', 'vs_base_p_acc_holm', 'log_loss',
                                  'vs_base_diff_ll', 'vs_base_diff_ll_lo', 'vs_base_diff_ll_hi', 'vs_base_p_ll',
                                  'vs_base_p_ll_holm', 'vs_base_identical']],
               caption=('Data-treatment choices for the L2 logistic regression, each paired against the default '
                        f"(approach window, {v('n_channels')} channels, k = {v('n_bins')} bins, z-score, mirror "
                        f"augmentation; {v('lr_full_accuracy')}) on the same trials: "
                        + '; '.join(f"{g}: " + ', '.join(t[t['group'] == g]['row_label'].str.split(': ').str[-1])
                                    for g in order if (t['group'] == g).any())
                        + '. Differences with paired person-bootstrap 95 % CIs; orange: Holm-adjusted sign-flip p < '
                        f"{v('alpha')} within the family; open marker: identical to the default within solver "
                        'precision (not a test).'),
               decision=dec)


def _family_rows(t: pd.DataFrame, group: str, abstain: pd.Series) -> pd.DataFrame:
    d = t[t['group'] == group].copy()
    d['family_name'] = d['label'].str.replace('without ', '', regex=False).str.replace('only ', '', regex=False)
    d['abstains'] = d['name'].map(abstain).fillna(False).astype(bool)
    return d


def fig_M2_03_families(**kw: Any) -> None:
    name = 'M2_03_families'
    o = {**FIGURES[name], **kw}
    t = _exp()
    mask = ((t['p_min'] - 0.5).abs() < 1e-9) & ((t['p_max'] - 0.5).abs() < 1e-9)
    abstain = pd.Series(mask.to_numpy(), index=t['name'].to_numpy())      # relabel (a Series argument would realign)
    rows_def = (('family_drop', 'family_only', 'with mirror augmentation', 'lr_full'),
                ('family_drop_nomirror', 'family_only_nomirror', 'without mirroring', 'lr_mirror_none'))
    rows_def = [r for r in rows_def if (t['group'] == r[0]).any() and (t['group'] == r[1]).any()]
    if not rows_def:
        raise MissingData('family ablation results missing')
    fig, axes = figure(o['width'], o['height'], len(rows_def), 2, squeeze=False)
    data = []
    reg_odd = {f: nums().get(f'reg_{f}_odd', {}).get('value') for f in
               set(t['families'].str.split('+').explode().dropna())}
    reg_n = {f: nums().get(f'reg_{f}_n', {}).get('value') for f in reg_odd}
    dec = []
    for j, (gd, go, title, base) in enumerate(rows_def):
        dr = _family_rows(t, gd, abstain).sort_values('vs_base_diff_ll').reset_index(drop=True)
        on = _family_rows(t, go, abstain).sort_values('accuracy').reset_index(drop=True)
        dr['row_label'] = [f"{f} ({reg_odd.get(f)}/{reg_n.get(f)})" if j == 0 else f"{f} ({reg_n.get(f)})"
                           for f in dr['family_name']]
        _forest(axes[j, 0], dr, 'diff_ll', o, o['drop_xlim'], o['drop_xlabel'] if j == len(rows_def) - 1 else '',
                scale=1.0)
        axes[j, 0].set_title(f'drop one family, {title}', fontsize=7.2)
        y = np.arange(len(on))
        for i, r in on.iterrows():
            if r['abstains']:
                axes[j, 1].plot([r['accuracy']], [i], 'o', color=GREY, mfc='white', ms=o['marker_size'])
                axes[j, 1].text(r['accuracy'], i, '  P = 0.5 for every trial', fontsize=5.6, color=MUTED,
                                va='center', ha='left')
            else:
                _dot_ci(axes[j, 1], [r['accuracy']], [i], [r['accuracy_lo']], [r['accuracy_hi']], INK2,
                        o['marker_size'])
        full_acc = float(t.set_index('name').loc[f'm2_{base}', 'accuracy'])
        axes[j, 1].axvline(full_acc, color=SERIES, lw=0.8, ls='--')
        axes[j, 1].text(full_acc, len(on) - 0.5, f"all families {v(base + '_accuracy')}", color=SERIES, fontsize=5.8,
                        ha='right', va='bottom')
        axes[j, 1].set_yticks(y, [f"{f} ({reg_odd.get(f)}/{reg_n.get(f)})" if j == 0 else f"{f} ({reg_n.get(f)})"
                                  for f in on['family_name']], fontsize=o['label_fontsize'])
        axes[j, 1].set_ylim(-0.6, len(on) + 0.2)
        if o['only_xlim']:
            axes[j, 1].set_xlim(*o['only_xlim'])
        axes[j, 1].set_xlabel(o['only_xlabel'] if j == len(rows_def) - 1 else '')
        axes[j, 1].grid(axis='y', visible=False)
        axes[j, 1].set_title(f'one family only, {title}', fontsize=7.2)
        data.append(pd.concat([dr.assign(panel=f'drop_one_{title}'), on.assign(panel=f'only_one_{title}')]))
        worst = dr.iloc[-1]
        ws = R.short(worst['name'])
        best_on = on[~on['abstains']].iloc[-1]
        bo = R.short(best_on['name'])
        worse_fams = [f for f, n in zip(dr['family_name'], dr['name']) if v(f"{R.short(n)}_ll_verdict") == 'worse']
        dec.append(f"{title.capitalize()}: dropping {worst['family_name']} costs most ({v(ws + '_d_ll')} "
                   f"{ci(ws + '_d_ll')} log loss, {_sig_text(ws)}); removals that worsen the log loss after Holm: "
                   f"{', '.join(worse_fams) or 'none'}; best single family {best_on['family_name']} "
                   f"({v(bo + '_accuracy')})")
        ab = on[on['abstains']]
        if len(ab):
            dec[-1] += ('; alone, ' + ', '.join(ab['family_name']) + ' predicts P = 0.5 for every trial (an '
                        'abstention: no mirror-odd feature)')
    fig.align_ylabels()
    keep = ['panel', 'name', 'family_name', 'n_features', 'abstains', 'accuracy', 'accuracy_lo', 'accuracy_hi',
            'log_loss', 'vs_base_diff_ll', 'vs_base_diff_ll_lo', 'vs_base_diff_ll_hi', 'vs_base_p_ll_holm',
            'vs_base_identical']
    L.save_fig(fig, name, data=pd.concat(data)[keep],
               caption=('Feature-family ablation of the L2 logistic regression (leave-one-person-out), with mirror '
                        'augmentation (top; numbers: mirror-odd or swapped features / all features of the family) and '
                        'without mirroring (bottom; all features of the family). Left: log-loss change when one family '
                        f"is removed (paired 95 % CI; orange: Holm-adjusted p < {v('alpha')} within the row's family "
                        'of comparisons; open: identical within solver precision). Right: accuracy of each family '
                        'alone; dashed: all families. Mirror augmentation forces the weights of mirror-even features '
                        'to zero in a linear model, so the top row tests only the odd part of each family; the '
                        'bottom row lets every family carry weight.'),
               decision='. '.join(dec) + '. All families stay in the pre-specified default.')


def fig_M2_04_mirror(**kw: Any) -> None:
    name = 'M2_04_mirror'
    o = {**FIGURES[name], **kw}
    t = _exp().set_index('name')
    mh = L.read_json('m2_summary').get('mirror_handedness') or {}
    hand = pd.DataFrame(mh.get('groups') or [])
    hp = pd.DataFrame(mh.get('persons') or [])
    variants = ('none', 'aug', 'anti', 'both')
    vlab = {'none': 'none', 'aug': 'augment', 'anti': 'antisym.', 'both': 'both'}
    models = (('lr_l2', 'm2_lr_full', 'm2_lr_mirror_', 'L2 logistic regression'),
              ('lgbm', 'm2_lgbm', 'm2_lgbm_mirror_', 'LightGBM'))
    fig, ax = figure(o['width'], o['height'], 1, 3)
    rows, run = [], {}
    for j, (m, base, pre, lab) in enumerate(models):
        names = [(vv, base if vv == 'aug' else f'{pre}{vv}') for vv in variants]
        names = [(vv, n) for vv, n in names if n in t.index]
        run[lab] = [vv for vv, _ in names]
        col = SERIES if j == 0 else ACCENT
        for vv, n in names:
            x = variants.index(vv) + (j - 0.5) * 0.22
            r = t.loc[n]
            _vci(ax[0], [x], [r['accuracy']], [r['accuracy_lo']], [r['accuracy_hi']], col, o['marker_size'],
                 label=lab if vv == names[0][0] else None)
            _vci(ax[1], [x], [r['log_loss']], [r['log_loss_lo']], [r['log_loss_hi']], col, o['marker_size'])
            rows.append({'panel': 'variants', 'model': m, 'variant': vv, 'name': n, 'accuracy': r['accuracy'],
                         'acc_lo': r['accuracy_lo'], 'acc_hi': r['accuracy_hi'], 'log_loss': r['log_loss'],
                         'll_lo': r['log_loss_lo'], 'll_hi': r['log_loss_hi']})
            for hnd, mk, dx in (('left', 's', -0.06), ('right', 'o', 0.06)):
                h = hand[(hand['name'] == n) & (hand['handedness'] == hnd)] if len(hand) else hand
                if not len(h):
                    continue
                hr = h.iloc[0]
                if hr['ci']:
                    _vci(ax[2], [x + dx], [hr['accuracy']], [hr['acc_lo']], [hr['acc_hi']], col, o['marker_size'],
                         marker=mk)
                else:
                    pv = hp[(hp['name'] == n) & (hp['handedness'] == hnd)]['accuracy'].to_numpy() if len(hp) else []
                    ax[2].plot(np.full(len(pv), x + dx), pv, mk, color=col, mfc='white', ms=o['marker_size'] - 0.4)
    for a in ax:
        a.set_xticks(range(len(variants)), [vlab[x] for x in variants])
        a.grid(axis='x', visible=False)
    if o['acc_ylim']:
        ax[0].set_ylim(*o['acc_ylim'])
    ax[0].set_ylabel('LOSO accuracy (95 % CI)')
    ax[0].legend(loc='lower right', fontsize=6)
    if o['ll_ylim']:
        ax[1].set_ylim(*o['ll_ylim'])
    ax[1].set_ylabel('log loss (95 % CI)')
    if o['hand_ylim']:
        ax[2].set_ylim(*o['hand_ylim'])
    ax[2].set_ylabel('accuracy by handedness')
    from matplotlib.lines import Line2D
    n_by = {h: int(g['n_persons'].max()) for h, g in hand.groupby('handedness')} if len(hand) else {}
    ci_by = {h: bool(g['ci'].all()) for h, g in hand.groupby('handedness')} if len(hand) else {}
    ax[2].legend(handles=[Line2D([], [], marker='s' if h == 'left' else 'o', ls='', color=INK2,
                                 mfc='white' if not ci_by.get(h) else INK2,
                                 label=f"{h}-handed ({n_by[h]}{'' if ci_by.get(h) else ', each person'})")
                          for h in ('left', 'right') if h in n_by], loc='lower center', bbox_to_anchor=(0.5, 1.0),
                 ncol=2, fontsize=5.6, frameon=False, handletextpad=0.2, columnspacing=0.8)
    lr_none, lr_both, lg_none = 'lr_mirror_none', 'lr_mirror_both', 'lgbm_mirror_none'
    parts = []
    for s, lab in ((lr_none, 'L2 logistic regression'), (lg_none, 'LightGBM')):
        if has(s + '_d_ll'):
            parts.append(f"{lab} without mirroring: log loss {v(s + '_d_ll')} {ci(s + '_d_ll')} against "
                         f"augmentation ({_sig_text(s)}), accuracy {v(s + '_d_acc')} {ci(s + '_d_acc')} points")
    dec = ('Mirror augmentation is kept (pre-specified). ' + '; '.join(parts) + '. '
           + ((f"For the logistic regression augmentation plus the wrapper is {_sig_text(lr_both)} against "
               'augmentation alone' + (', as expected: an augmented linear model is already antisymmetric. '
                                       if v(lr_both + '_ll_verdict') == 'identical' else '. '))
              if has(lr_both + '_ll_verdict') else '')
           + (f"With {v('mh_lr_full_left_persons')} left-handed persons the handedness panel is descriptive only "
              '(each person shown, no CI).' if 'left' in n_by and not ci_by.get('left')
              and has('mh_lr_full_left_persons') else ''))
    vrun = '; '.join(f"{lab}: {', '.join(vlab[x] for x in run[lab])}" for lab in run)
    data = pd.DataFrame(rows)
    if len(hand):
        data = pd.concat([data, hand.assign(panel='handedness'), hp.assign(panel='handedness_person')])
    L.save_fig(fig, name, data=data,
               caption=('Mirror handling: none, mirror augmentation of the training rows (flipped labels), the '
                        'antisymmetric wrapper p(x) = (f(x) + 1 - f(m(x)))/2, and both; variants computed: ' + vrun
                        + '. Right: accuracy of left- and right-handed persons (handedness stratifies only; a '
                        f"person-bootstrap CI only for groups of at least {v('min_persons_ci')} persons, otherwise "
                        'each person is shown).'),
               decision=dec)


def fig_M2_05_learning_curve(**kw: Any) -> None:
    name = 'M2_05_learning_curve'
    o = {**FIGURES[name], **kw}
    lc = _csv('m2_lc_runs')
    sub = lc[lc['n_train_persons_requested'].astype(str) != 'all']
    full = lc[lc['n_train_persons_requested'].astype(str) == 'all']
    if sub.empty:
        raise MissingData('learning-curve results missing')
    fig, ax = figure(o['width'], o['height'], 1, 2)
    g = sub.groupby('n_train_persons').agg(acc=('accuracy', 'mean'), acc_min=('accuracy', 'min'),
                                           acc_max=('accuracy', 'max'), ll=('log_loss', 'mean'),
                                           ll_min=('log_loss', 'min'), ll_max=('log_loss', 'max'),
                                           n=('accuracy', 'size')).reset_index()
    if len(full):
        f0 = full.iloc[0]
        g = pd.concat([g, pd.DataFrame([{'n_train_persons': f0['n_train_persons'], 'acc': f0['accuracy'],
                                         'acc_min': f0['accuracy'], 'acc_max': f0['accuracy'], 'll': f0['log_loss'],
                                         'll_min': f0['log_loss'], 'll_max': f0['log_loss'], 'n': 1}])],
                      ignore_index=True)
    for a, col, gc, lab in ((ax[0], 'accuracy', 'acc', 'LOSO accuracy'), (ax[1], 'log_loss', 'll', 'log loss')):
        a.fill_between(g['n_train_persons'], g[f'{gc}_min'], g[f'{gc}_max'], color=SERIES, alpha=o['band_alpha'], lw=0,
                       label='range over repetitions')
        a.plot(sub['n_train_persons'], sub[col], 'o', color=GREY, ms=o['marker_size'] - 0.8, alpha=0.9,
               label='one random subset of persons')
        a.plot(g['n_train_persons'], g[gc], '-', color=SERIES, lw=1.4, label='mean over repetitions')
        a.set_xlabel(o['xlabel'])
        a.set_ylabel(lab)
        a.set_xticks(sorted(g['n_train_persons'].unique()))
    if len(full):
        r = full.iloc[0]
        ax[0].errorbar([r['n_train_persons']], [r['accuracy']],
                       yerr=[[r['accuracy'] - r['acc_lo']], [r['acc_hi'] - r['accuracy']]], fmt='o', color=SERIES,
                       ms=o['marker_size'] + 0.5, capsize=2, lw=1.0, label='every training person (95 % CI)')
    if o['acc_ylim']:
        ax[0].set_ylim(*o['acc_ylim'])
    if o['ll_ylim']:
        ax[1].set_ylim(*o['ll_ylim'])
    ax[0].legend(loc=o['legend_loc'], fontsize=5.6)
    sizes = sorted(int(x) for x in sub['n_train_persons_requested'].astype(float).unique())
    s0, s1 = f'lc_n{sizes[0]:02d}', f'lc_n{sizes[-1]:02d}'
    dec = (f"Mean accuracy {v(s0 + '_acc_mean')} (range {v(s0 + '_acc_min')} to {v(s0 + '_acc_max')}) with "
           f"{v(s0 + '_persons')} training persons, {v(s1 + '_acc_mean')} (range {v(s1 + '_acc_min')} to "
           f"{v(s1 + '_acc_max')}) with {v(s1 + '_persons')} and {v('lc_all_acc')} with all {v('lc_all_persons')}; "
           f"mean log loss {v(s0 + '_ll_mean')}, {v(s1 + '_ll_mean')} and {v('lc_all_ll')}. From "
           f"{v(s1 + '_persons')} to all persons the accuracy changes by {v('lc_last_step_acc')} points ("
           + ('within' if v('lc_last_step_acc_within') == 'yes' else 'outside')
           + f" the range of the {v(s1 + '_reps')} subsets) and the log loss by {v('lc_last_step_ll')} ("
           + ('within' if v('lc_last_step_ll_within') == 'yes' else 'outside') + ' their range). ')
    acc_flat = v('lc_last_step_acc_within') == 'yes'
    ll_flat = v('lc_last_step_ll_within') == 'yes'
    dec += ('Both have flattened at this size: more participants may add little (tentative, one run at the full size).'
            if acc_flat and ll_flat else
            'Accuracy has flattened while the probabilities still sharpen: more participants would mainly improve the '
            'log loss (tentative, one run at the full size).' if acc_flat else
            'Accuracy is still rising at this size: more participants are likely to help (tentative, one run at the '
            'full size).')
    L.save_fig(fig, name, data=lc,
               caption=('Learning curve of the default L2 logistic regression: inside every outer fold only a random '
                        f"subset of the training persons is used (seeded; {v(s0 + '_reps')} repetitions per size; "
                        'band: range over repetitions); the held-out person is always scored. The right-most point '
                        'uses every training person (one run, 95 % person-bootstrap CI).'),
               decision=dec)


def fig_M2_06_importance(**kw: Any) -> None:
    name = 'M2_06_importance'
    o = {**FIGURES[name], **kw}
    imp = L.read_json('m2_importance')
    models = [m for m in ('m2_lr_full', 'm2_lgbm') if m in imp]
    if not models:
        raise MissingData('importance missing')
    fams = sorted({f for m in models for f in imp[m]['family_share']}, key=lambda f: -imp[models[0]]['family_share'].get(f, 0))
    fig, ax = figure(o['width'], o['height'], 1, 2)
    w = 0.38
    cols = {'m2_lr_full': SERIES, 'm2_lgbm': ACCENT}
    rows = []
    for j, m in enumerate(models):
        vals = [imp[m]['family_share'].get(f, 0.0) for f in fams]
        ax[0].barh(np.arange(len(fams)) + (j - 0.5) * w, vals, height=w, color=cols[m], label=disp(m))
        rows += [{'panel': 'family', 'model': m, 'item': f, 'share': s} for f, s in zip(fams, vals)]
    ax[0].set_yticks(range(len(fams)), fams)
    ax[0].invert_yaxis()
    ax[0].set_xlabel(o['xlabel'])
    ax[0].legend(loc='lower right', fontsize=6)
    ax[0].set_title('by feature family', fontsize=7.5)
    ch = pd.Series(imp[models[0]]['channel_share']).sort_values(ascending=False).head(o['n_channels']).index.tolist()
    for j, m in enumerate(models):
        vals = [imp[m]['channel_share'].get(c, 0.0) for c in ch]
        ax[1].barh(np.arange(len(ch)) + (j - 0.5) * w, vals, height=w, color=cols[m])
        rows += [{'panel': 'channel', 'model': m, 'item': c, 'share': s} for c, s in zip(ch, vals)]
    ax[1].set_yticks(range(len(ch)), ch)
    ax[1].invert_yaxis()
    ax[1].set_xlabel(o['xlabel'])
    ax[1].set_title(f'by channel (top {len(ch)})', fontsize=7.5)
    for a in ax:
        a.grid(axis='y', visible=False)
    parts = []
    for m in models:
        s = f'imp_{R.short(m)}'
        top_f = max(imp[m]['family_share'], key=imp[m]['family_share'].get)
        top_c = v(f'{s}_top_channel')
        parts.append(f"{disp(m)}: family {top_f} {v(f'{s}_fam_{top_f}')}, channel {top_c} "
                     f"{v(f'{s}_ch_{R.slug(top_c)}')}, gravity-frame channels {v(f'{s}_frame_gravity')}")
    # the frame ablation (M2_02) says whether each frame is needed: keeping only the gravity frame drops the device
    # channels and vice versa ('worse' = Holm-adjusted log-loss loss against all channels)
    needed = {fr: has(f'lr_frame_{only}_ll_verdict') and v(f'lr_frame_{only}_ll_verdict') == 'worse'
              for fr, only in (('device', 'gravity'), ('gravity', 'device'))}
    frames = ('Both frames contribute: keeping only one frame worsens the held-out log loss either way (M2_02).'
              if all(needed.values()) else
              f"Only the {[f for f, n in needed.items() if n][0]} frame is needed by the frame ablation (M2_02)."
              if any(needed.values()) else
              'The frame ablation (M2_02) shows no frame whose removal worsens the held-out log loss.')
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=('What drives the predictions on held-out persons: share of the mean absolute contribution to '
                        'the logit (L2 logistic regression: coefficient x scaled value; LightGBM: TreeSHAP '
                        'pred_contrib), aggregated over the leave-one-person-out folds, by feature family (left) and by '
                        'channel (right; a correlation pair is split equally between its two channels).'),
               decision='Largest shares: ' + '; '.join(parts) + '. ' + frames)


def fig_M2_07_calibration(**kw: Any) -> None:
    name = 'M2_07_calibration'
    o = {**FIGURES[name], **kw}
    cal = L.read_json('m2_calibration')
    order = [m for m in cal.get('order', []) if 'none' in cal['models'][m]['variants']]
    if not order:
        raise MissingData('calibration results missing')
    fig, ax = figure(o['width'], o['height'], 1, len(order), sharey=True)
    ax = np.atleast_1d(ax)
    rows = []
    for j, m in enumerate(order):
        a = ax[j]
        rel = pd.DataFrame(cal['models'][m]['variants']['none']['reliability'])
        col = MODEL_COLORS[j % len(MODEL_COLORS)]
        a.fill_between(rel['p_mean'], rel['y_lo'], rel['y_hi'], color=col, alpha=o['band_alpha'], lw=0)
        a.plot(rel['p_mean'], rel['y_mean'], 'o-', color=col, ms=o['marker_size'], lw=1.0)
        a.plot(o['lim'], o['lim'], color=MUTED, lw=0.7, ls='--')
        a.set_xlim(*o['lim'])
        a.set_ylim(*o['lim'])
        a.set_aspect('equal')
        a.set_xlabel('predicted P(right)')
        if j == 0:
            a.set_ylabel('observed share right')
        a.set_title(disp(m), fontsize=6.6)
        k = f'cal_{R.short(m)}_none'
        a.text(0.97, 0.03, f"ECE {v(k + '_ece')}\nnull 97.5% {v(k + '_ece_null_hi')}\nslope {v(k + '_slope')}\n"
                           f"{ci(k + '_slope')}", transform=a.transAxes, fontsize=o['text_fontsize'], ha='right',
               va='bottom', color=INK2)
        rows += [{'model': m, **r} for r in rel.to_dict(orient='records')]
    det = [m for m in order if v(f'cal_{R.short(m)}_detected') == 'yes']
    parts = '; '.join(f"{disp(m)} slope {v(f'cal_{R.short(m)}_none_slope')} {ci(f'cal_{R.short(m)}_none_slope')}, "
                      f"ECE {v(f'cal_{R.short(m)}_none_ece')} (null mean {v(f'cal_{R.short(m)}_none_ece_null_mean')})"
                      for m in order)
    dec = (('No detectable miscalibration: every slope CI includes 1, every calibration-in-the-large CI includes 0 and '
            'every ECE lies within the band that perfectly calibrated outcomes give. ' if not det else
            f"Miscalibration detected for {', '.join(disp(m) for m in det)}. ")
           + 'Per model: ' + parts + '.')
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=(f"Reliability on held-out persons of the uncalibrated models ({v('ece_bins')} equal-mass bins; "
                        'band: 95 % person-bootstrap interval of the observed share; dashed: perfect calibration). '
                        'Text: ECE and the 97.5th percentile of the ECE that perfectly calibrated outcomes (y drawn '
                        'from Bernoulli(p)) would give with the same predictions; calibration slope of the logistic '
                        'recalibration with its person-bootstrap 95 % CI (1 = calibrated, above 1 = under-confident).'),
               decision=dec)


def fig_M2_08_calibration_platform(**kw: Any) -> None:
    name = 'M2_08_calibration_platform'
    o = {**FIGURES[name], **kw}
    cal = L.read_json('m2_calibration')
    order = [m for m in cal.get('order', []) if 'none' in cal['models'][m]['variants']]
    if not order:
        raise MissingData('calibration results missing')
    plats = sorted({p for m in order for p in cal['models'][m]['variants']['none']['platform']})
    fig, ax = figure(o['width'], o['height'], 1, 3)
    rows = []
    for i, m in enumerate(order):
        e = cal['models'][m]['variants']['none']
        for j, pl in enumerate(plats):
            st = e['platform'].get(pl)
            if not st:
                continue
            x = i + (j - (len(plats) - 1) / 2) * 0.28
            c = PLATFORM_COLORS.get(pl, INK2)
            lab = f"{PLATFORM_LABELS.get(pl, pl)} ({st['n_persons']} persons)" if i == 0 else None
            _vci(ax[0], [x], [st['cal_slope']], [st.get('cal_slope_lo')], [st.get('cal_slope_hi')], c,
                 o['marker_size'], label=lab)
            ax[1].vlines(x, st['ece_null_lo'], st['ece_null_hi'], color=GREY, lw=3.0, alpha=0.5)
            ax[1].plot([x], [st['ece']], 'o', color=c, ms=o['marker_size'])
            rows.append({'model': m, 'platform': pl, **{k: st.get(k) for k in (
                'n', 'n_persons', 'cal_slope', 'cal_slope_lo', 'cal_slope_hi', 'cal_in_the_large', 'ece',
                'ece_null_lo', 'ece_null_hi', 'log_loss', 'brier')}})
    ax[0].axhline(1.0, color=INK, lw=0.7, ls='--')
    if o['slope_ylim']:
        ax[0].set_ylim(*o['slope_ylim'])
    ax[0].set_ylabel('calibration slope (95 % CI)')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='outside upper center', ncol=2, fontsize=6, frameon=False)
    ax[1].set_ylabel(o['ece_ylabel'])
    names = [TICK.get(m, disp(m)) for m in order]
    for a in ax[:2]:
        a.set_xticks(range(len(order)), names, rotation=20, ha='right', fontsize=6)
        a.grid(axis='x', visible=False)
    pp = [m for m in order if cal['models'][m]['variants'].get('platt_platform', {}).get('vs_platt')]
    for i, m in enumerate(pp):
        vv = cal['models'][m]['variants']['platt_platform']['vs_platt']
        _vci(ax[2], [i], [vv['diff_ll']], [vv['diff_ll_lo']], [vv['diff_ll_hi']], INK2, o['marker_size'])
        rows.append({'model': m, 'platform': 'per-platform minus global Platt', 'diff_ll': vv['diff_ll'],
                     'diff_ll_lo': vv['diff_ll_lo'], 'diff_ll_hi': vv['diff_ll_hi'], 'p_ll': vv['p_ll']})
    ax[2].axhline(0.0, color=INK, lw=0.7)
    ax[2].set_xticks(range(len(pp)), [TICK.get(m, disp(m)) for m in pp], rotation=20, ha='right', fontsize=6)
    ax[2].set_xlim(-0.6, len(pp) - 0.4)
    ax[2].set_ylabel(o['pp_ylabel'])
    ax[2].grid(axis='x', visible=False)
    det = [(m, pl) for m in order for pl, d in cal['models'][m]['decision'].get('platform_miscalibration_detected',
                                                                                     {}).items() if d]
    def pk(m: str) -> str:
        return f'cal_{R.short(m)}_platt_platform_vs_platt'

    def verdict(m: str) -> str:
        """better / worse / not different from the Holm-adjusted p over the analysed models (m2_derive)."""
        return v(pk(m) + '_verdict') if has(pk(m) + '_verdict') else 'not different'
    pp_txt = '; '.join(f"{disp(m)} {v(pk(m) + '_d_ll')} {ci(pk(m) + '_d_ll')} (Holm p {pe(pk(m) + '_p_ll_holm')}, "
                       f"{verdict(m)})" for m in pp if has(pk(m) + '_p_ll_holm'))
    head = order[0]
    hk = f'cal_{R.short(head)}_none'
    slopes = ', '.join(f"{PLATFORM_LABELS.get(pl, pl)} {v(f'{hk}_{R.slug(pl)}_slope')} {ci(f'{hk}_{R.slug(pl)}_slope')}"
                       for pl in plats if has(f'{hk}_{R.slug(pl)}_slope'))
    better = [m for m in pp if verdict(m) == 'better']
    worse = [m for m in pp if verdict(m) == 'worse']
    small = min(plats, key=lambda pl: cal['models'][head]['variants']['none']['platform'][pl]['n_persons'])
    dec = (f"{cap(disp(head))} per platform: slope {slopes}. "
           + ('No platform shows detectable miscalibration. ' if not det else
              'Miscalibration detected for ' + ', '.join(f'{disp(m)} on {PLATFORM_LABELS.get(pl, pl)}'
                                                         for m, pl in det) + '. ')
           + (f"A platform-specific Platt map against one global map changes the log loss by {pp_txt}; Holm over "
              f"the {v(pk(pp[0]) + '_family_size')} models. " if pp_txt else '')
           + ('No evidence that the platforms need different calibration maps: a per-platform map is never better '
              'than one global map' + (f" and is worse for the {', '.join(disp(m) for m in worse)}" if worse else '')
              + f"; underpowered ({PLATFORM_LABELS.get(small, small)}: {v(f'{hk}_{R.slug(small)}_n_persons')} persons)."
              if (not better and not det) else
              'The platforms may need different maps: ' + (', '.join(disp(m) for m in better) or 'see above') + '.'))
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=('Calibration per platform of the uncalibrated models on held-out persons: calibration slope '
                        'with person-bootstrap 95 % CI (left; 1 = calibrated), ECE against the 95 % band of the ECE '
                        'that perfectly calibrated outcomes would give with the same predictions (middle), and the '
                        'paired log-loss difference of a platform-specific Platt map against one global Platt map, '
                        'both fitted inside the outer folds (right; below 0 = the per-platform map helps).'),
               decision=dec)


def fig_M2_09_breakdown(**kw: Any) -> None:
    name = 'M2_09_breakdown'
    o = {**FIGURES[name], **kw}
    bd = _csv('m2_strata')
    bp = _csv('m2_strata_persons')
    s = _summary()
    model = s['headline'][0]
    d = bd[(bd['model'] == model) & bd['factor'].isin(o['factors'])].copy()
    d['f_order'] = d['factor'].map({f: i for i, f in enumerate(o['factors'])})
    d = d.sort_values(['f_order', 'accuracy'], ascending=[True, True]).reset_index(drop=True)
    fig, ax = figure(o['width'], o['height'])
    y = np.arange(len(d))[::-1]
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    # x range: the strata with a CI and the pooled accuracy of the small strata; a per-person value below it is drawn
    # as an arrow at the edge, labelled with its value (nothing is clipped silently)
    has_ci = d['ci'].astype(bool)
    lo = min(([float(d.loc[has_ci, 'acc_lo'].min())] if has_ci.any() else []) + [float(d['accuracy'].min())])
    xmin, xmax = o['xlim'] or (max(0.0, lo - 0.04), 1.02)
    for i, r in d.iterrows():
        if bool(r['ci']):
            _dot_ci(ax, [r['accuracy']], [y[i]], [r['acc_lo']], [r['acc_hi']], INK2, o['marker_size'])
        else:
            pv = bp[(bp['model'] == model) & (bp['factor'] == r['factor']) & (bp['level'].astype(str) == str(r['level']))]
            inside = pv[pv['accuracy'] >= xmin]
            ax.plot(inside['accuracy'], np.full(len(inside), y[i]), 'o', color=GREY, mfc='white',
                    ms=o['marker_size'] - 0.6)
            for _, pr in pv[pv['accuracy'] < xmin].iterrows():
                key = f"bdp_{R.short(model)}_{R.slug(r['factor'])}_{R.slug(r['level'])}_{R.slug(pr['person_id'])}"
                ax.plot([xmin + 0.006], [y[i]], '<', color=GREY, ms=o['marker_size'])
                ax.text(xmin + 0.016, y[i], v(key), fontsize=5.4, color=MUTED, ha='left', va='center')
            ax.plot([r['accuracy']], [y[i]], 'D', color=INK2, ms=o['marker_size'] - 0.4)
        ax.text(1.01, y[i], f"{int(r['n_trials'])} / {int(r['n_persons'])}", fontsize=5.8, color=MUTED,
                va='center', ha='left', transform=tr, clip_on=False)
    ax.text(1.01, y[0] + 1.0, 'trials / persons', fontsize=5.8, color=MUTED, va='center', ha='left', transform=tr,
            clip_on=False)
    ax.set_yticks(y, [f"{r['factor_label']}: {LEVEL_TEXT.get(str(r['level']), str(r['level']).replace('_', ' '))}"
                      for _, r in d.iterrows()], fontsize=o['label_fontsize'])
    for i in range(1, len(d)):
        if d['factor'].iloc[i] != d['factor'].iloc[i - 1]:
            ax.axhline(y[i] + 0.5, color='#e1e0d9', lw=0.6)
    overall = float(_exp().set_index('name').loc[model, 'accuracy'])
    ax.axvline(overall, color=SERIES, lw=0.8, ls='--')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('accuracy (95 % person-bootstrap CI)')
    ax.set_ylim(y.min() - 0.6, y.max() + 1.4)
    ax.grid(axis='y', visible=False)
    big = d[d['ci'].astype(bool)].sort_values('accuracy')
    worst = big.iloc[0]
    wk = f"bd_{R.short(model)}_{R.slug(worst['factor'])}_{R.slug(worst['level'])}"
    small = d[~d['ci'].astype(bool)]
    L.save_fig(fig, name, data=pd.concat([d.drop(columns=['f_order']).assign(panel='stratum'),
                                          bp[bp['model'] == model].assign(panel='person')]),
               caption=(f'Accuracy of the headline model ({disp(model)}) by stratum: platform, instructed grip, speed '
                        'and movement variant, self-reported handedness and the predictor gates (strata are used only '
                        'after prediction, never as inputs). A person-bootstrap 95 % CI only for strata with at least '
                        f"{v('min_persons_ci')} persons; smaller strata show each person (open circles; an arrow at the "
                        'left edge, labelled with its value, for a person below the axis) and the pooled accuracy '
                        f"(diamond). Dashed: overall ({v(R.short(model) + '_accuracy')})."),
               decision=(f"Lowest stratum with at least {v('min_persons_ci')} persons: {worst['factor_label']} = "
                         f"{LEVEL_TEXT.get(str(worst['level']), str(worst['level']).replace('_', ' '))} "
                         f"({v(wk + '_acc')} {ci(wk + '_acc')}, {v(wk + '_n')} trials, {v(wk + '_persons')} persons)"
                         + ('; the strata with fewer persons are descriptive only' if len(small) else '')
                         + '. Descriptive: the strata are not a test of the conditions.'))


def fig_M2_10_persons(**kw: Any) -> None:
    name = 'M2_10_persons'
    o = {**FIGURES[name], **kw}
    pp = _csv('m2_per_person')
    s = _summary()
    models = [m for m in s['headline'] if m in set(pp['model'])][:2]
    a_m = pp[pp['model'] == models[0]].sort_values(['accuracy', 'person_id']).reset_index(drop=True)
    order = a_m['person_id'].tolist()
    fig, ax = figure(o['width'], o['height'])
    x = np.arange(len(order))
    for j, m in enumerate(models):
        sub = pp[pp['model'] == m].set_index('person_id').loc[order]
        cols = [PLATFORM_COLORS.get(p, INK2) for p in sub['platform']]
        ax.bar(x + (j - 0.5) * o['bar_width'] * (len(models) > 1), sub['accuracy'], width=o['bar_width'], color=cols,
               alpha=1.0 if j == 0 else 0.45)
    ax.set_xticks(x, order, rotation=90, fontsize=6)
    ax.set_ylim(*o['ylim'])
    ax.axhline(o['line'], color=MUTED, lw=0.7, ls='--')
    ax.set_ylabel('accuracy of the held-out person')
    from matplotlib.patches import Patch
    handles = [Patch(color=PLATFORM_COLORS[p], label=PLATFORM_LABELS[p]) for p in ('android', 'ios')]
    handles += [Patch(color=INK2, alpha=a, label=disp(m)) for m, a in zip(models, (1.0, 0.45))]
    ax.legend(handles=handles, loc='upper left', fontsize=6, ncol=4)
    ax.grid(axis='x', visible=False)
    k = f'pp_{R.short(models[0])}'
    L.save_fig(fig, name, data=pp[pp['model'].isin(models)],
               caption=('Accuracy of every held-out person (leave-one-person-out), sorted by the first model; colour: '
                        f"the person's platform; solid: {disp(models[0])}"
                        + (f'; light: {disp(models[1])}' if len(models) > 1 else '') + '. Dashed: 80 %.'),
               decision=(f"{v(k + '_below80')} of {v(k + '_n')} persons are below 80 % with the {disp(models[0])} "
                         f"(lowest: {v(k + '_min_person')}, {v(k + '_min')}; median {v(k + '_median')}): per-person "
                         'accuracy, not the pooled mean, is what a new user experiences.'))


def fig_M2_11_k_attempts(**kw: Any) -> None:
    name = 'M2_11_k_attempts'
    o = {**FIGURES[name], **kw}
    ka = L.read_json('m2_k_attempts')
    s = _summary()
    model = s['headline'][0]
    if model not in ka:
        raise MissingData('k-attempt analysis missing')
    fig, ax = figure(o['width'], o['height'], 1, 3)
    rows = []
    for j, m in enumerate([n for n in s['headline'] if n in ka]):
        kc = pd.DataFrame(ka[m]['k_curve'])
        col = MODEL_COLORS[j % len(MODEL_COLORS)]
        if j == 0:                                       # one band (the headline's) keeps the panel readable
            ax[0].fill_between(kc['k'], kc['lo'], kc['hi'], color=col, alpha=0.15, lw=0)
        ax[0].plot(kc['k'], kc['accuracy'], 'o-', color=col, ms=o['marker_size'], label=disp(m))
        rows += [{'panel': 'k_curve', 'model': m, **r} for r in kc[['k', 'accuracy', 'lo', 'hi', 'n_persons']].to_dict(orient='records')]
    ax[0].set_xlabel('attempts averaged (mean P)')
    ax[0].set_ylabel('per-person balanced accuracy')
    if o['acc_ylim']:
        ax[0].set_ylim(*o['acc_ylim'])
    ax[0].legend(loc='lower right', fontsize=5.6)
    ag = pd.DataFrame(ka[model]['rows'])
    aggs = [a for a in AGG_COLORS if a in set(ag['aggregator'])]
    for i, agg in enumerate(aggs):
        g = ag[ag['aggregator'] == agg]
        xk = g['k'] + (i - (len(aggs) - 1) / 2) * 0.08
        _vci(ax[1], xk, g['log_loss'], g['log_loss_lo'], g['log_loss_hi'], AGG_COLORS[agg], o['marker_size'],
             label=AGG_LABELS[agg])
        ax[1].plot(xk, g['log_loss'], '-', color=AGG_COLORS[agg], lw=0.9)
        _vci(ax[2], xk, g['conf_gap'], g['conf_gap_lo'], g['conf_gap_hi'], AGG_COLORS[agg], o['marker_size'])
        ax[2].plot(xk, g['conf_gap'], '-', color=AGG_COLORS[agg], lw=0.9)
    rows += [{'panel': 'aggregates', 'model': model, **r} for r in ag.to_dict(orient='records')]
    ax[1].set_xlabel('attempts aggregated')
    ax[1].set_ylabel('log loss of the aggregate (95 % CI)')
    if o['ll_ylim']:
        ax[1].set_ylim(*o['ll_ylim'])
    ax[1].legend(loc='lower left', fontsize=5.4)
    ax[2].axhline(0.0, color=INK, lw=0.7, ls='--')
    ax[2].set_xlabel('attempts aggregated')
    ax[2].set_ylabel('confidence minus share correct (95 % CI)', fontsize=6.6)
    if o['gap_ylim']:
        ax[2].set_ylim(*o['gap_ylim'])
    for a in ax:
        a.set_xticks(sorted(ag['k'].unique()))
    sk = f'ka_{R.short(model)}'
    km = v(f'{sk}_k_max')
    rec = ka[model].get('recommended')
    ra = R.AGG_SHORT.get(rec, '')

    def conf_word(a: str) -> str:
        lo, hi = nums()[f'{sk}_k{km}_{a}_gap_lo']['value'], nums()[f'{sk}_k{km}_{a}_gap_hi']['value']
        word = 'over-confident' if lo > 0 else ('under-confident' if hi < 0 else 'calibrated within its CI')
        return f"{word} (confidence minus share correct {v(f'{sk}_k{km}_{a}_gap')} {ci(f'{sk}_k{km}_{a}_gap')})"
    dec = (f"Displayed probability: the {v(sk + '_recommended')} (lowest log loss at k = {km}: "
           f"{v(f'{sk}_k{km}_{ra}_ll')} {ci(f'{sk}_k{km}_{ra}_ll')}; {conf_word(ra)}). ")
    if rec == 'sum_logit_tempered' and has(f'{sk}_k{km}_tempered_vs_sum_d_ll'):
        pk = f'{sk}_k{km}_tempered_vs_sum'
        dec += (f"Against the plain sum of logits its log loss differs by {v(pk + '_d_ll')} {ci(pk + '_d_ll')} "
                f"(Holm p {pe(pk + '_p_holm')}). ")
    dec += (f"Decision rule: the mean P(right) (the accuracy curve, {v(f'{sk}_k{km}_acc')} at k = {km}), whose "
            f"probability is {conf_word('meanp')}; the plain sum of logits is {conf_word('sum')}")
    if has(f'{sk}_tau_k{km}'):
        dec += (f"; the nested tau_{km} (median {v(f'{sk}_tau_k{km}')}) implies about {v(f'{sk}_eff_k{km}')} "
                f'independent attempts out of {km}')
    dec += '.'
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=(f"Several attempts of one person and hand. Left: per-person balanced accuracy when the P(right) "
                        f"of k attempts are averaged (the harness's k-attempt curve; {v(sk + '_n_persons')} persons with "
                        f"at least {km} trials of each hand). Middle and right: log loss and confidence gap (mean "
                        'max(P, 1 - P) minus the share correct: above 0 over-confident, below 0 under-confident) of four '
                        f"aggregates of k attempts for the {disp(model)}: mean P, mean logit, sum of logits (independent "
                        'evidence) and the sum divided by tau_k, fitted inside each outer fold on the training persons\' '
                        'cross-fitted predictions (nested); person-bootstrap 95 % CIs.'),
               decision=dec)


def fig_M2_12_calibration_remedies(**kw: Any) -> None:
    name = 'M2_12_calibration_remedies'
    o = {**FIGURES[name], **kw}
    cal = L.read_json('m2_calibration')
    models = [m for m in cal.get('order', []) if len(cal['models'][m]['variants']) > 1]
    if not models:
        raise MissingData('calibrator variants missing')
    cals = [c for c in ('platt', 'temperature', 'isotonic', 'beta', 'platt_platform')
            if any(c in cal['models'][m]['variants'] for m in models)]
    fig, ax = figure(o['width'], o['height'], 1, 2)
    rows = []
    for j, m in enumerate(models):
        col = MODEL_COLORS[j % len(MODEL_COLORS)]
        for i, c in enumerate(cals):
            e = cal['models'][m]['variants'].get(c)
            if not e:
                continue
            vv = e['vs_uncalibrated']
            x = i + (j - (len(models) - 1) / 2) * 0.22
            sig_ll = vv['p_ll_holm'] < cal['alpha'] and not vv['identical']
            for a, met in ((ax[0], 'll'), (ax[1], 'brier')):
                sig = vv[f'p_{met}_holm'] < cal['alpha'] and not vv['identical']
                _vci(a, [x], [vv[f'diff_{met}']], [vv[f'diff_{met}_lo']], [vv[f'diff_{met}_hi']], col,
                     o['marker_size'], mfc=col if sig else 'white', label=disp(m) if (i == 0 and met == 'll') else None)
            rows.append({'model': m, 'calibrator': c, **{k: vv[k] for k in (
                'diff_ll', 'diff_ll_lo', 'diff_ll_hi', 'p_ll', 'p_ll_holm', 'diff_brier', 'diff_brier_lo',
                'diff_brier_hi', 'p_brier', 'p_brier_holm')}, 'sig_ll': sig_ll})
    for a, lab in ((ax[0], 'log loss minus uncalibrated'), (ax[1], 'Brier score minus uncalibrated')):
        a.axhline(0, color=INK, lw=0.7)
        a.set_xticks(range(len(cals)), [R.CAL_TEXT[c].replace('Platt per platform', 'Platt/platform') for c in cals],
                     rotation=15, ha='right', fontsize=6)
        a.set_ylabel(lab + ' (paired 95 % CI)', fontsize=6.6)
        a.grid(axis='x', visible=False)
    ax[0].legend(loc='upper left', fontsize=5.6)
    parts = []
    for m in models:
        sk = f'cal_{R.short(m)}'
        b = cal['models'][m]['decision'].get('best_calibrator')
        parts.append(f"{disp(m)}: best {v(sk + '_best')} {v(f'{sk}_{b}_d_ll')} {ci(f'{sk}_{b}_d_ll')} (Holm p "
                     f"{pe(f'{sk}_{b}_p_ll_holm')}), adopted: {v(sk + '_adopted')}")
    adopted = [m for m in models if cal['models'][m]['decision'].get('adopted') != 'none']
    dec = (('No post-hoc calibrator improves the held-out log loss after the Holm adjustment over the calibrators of '
            'a model, so every model stays uncalibrated. ' if not adopted else
            f"A calibrator is adopted for {', '.join(disp(m) for m in adopted)}. ") + '; '.join(parts) + '.')
    L.save_fig(fig, name, data=pd.DataFrame(rows),
               caption=('Post-hoc calibration remedies, each fitted inside every outer fold on cross-fitted predictions '
                        'of the training persons (inner person folds) and applied to the held-out person: paired '
                        'per-person differences against the uncalibrated model in log loss (left) and Brier score '
                        '(right), person-bootstrap 95 % CIs (below 0 = better). Filled: Holm-adjusted sign-flip p < '
                        f"{v('alpha')} over the calibrators of the model."),
               decision=dec)


FIGURE_FUNCTIONS = {
    'M2_01_models': fig_M2_01_models, 'M2_02_data_treatment': fig_M2_02_data_treatment,
    'M2_03_families': fig_M2_03_families, 'M2_04_mirror': fig_M2_04_mirror,
    'M2_05_learning_curve': fig_M2_05_learning_curve, 'M2_06_importance': fig_M2_06_importance,
    'M2_07_calibration': fig_M2_07_calibration, 'M2_08_calibration_platform': fig_M2_08_calibration_platform,
    'M2_09_breakdown': fig_M2_09_breakdown, 'M2_10_persons': fig_M2_10_persons,
    'M2_11_k_attempts': fig_M2_11_k_attempts, 'M2_12_calibration_remedies': fig_M2_12_calibration_remedies,
}


def draw(only: list[str] | None = None) -> list[str]:
    """Draw every figure (or those matching ``only``) from the saved files; a figure whose inputs are missing is
    reported and skipped (the compute step writes them)."""
    _NUM.clear()
    done = []
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            try:
                fn()
            except (MissingData, FileNotFoundError) as e:
                print(f'm2 figures: {name} skipped ({e}); run python -m stage2.pipeline compute --lanes m2', flush=True)
                continue
            except KeyError as e:                       # a number its text cites is missing: an experiment not run
                print(f'm2 figures: {name} skipped (number {e} missing: an experiment it needs has no saved result); '
                      'run python -m stage2.pipeline compute --lanes m2', flush=True)
                continue
            done.append(name)
    return done


def caption_of(name: str) -> dict[str, str]:
    return json.loads((L.fig_dir / 'captions.json').read_text(encoding='utf-8')).get(name, {})
