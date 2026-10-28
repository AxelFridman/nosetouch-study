"""Figures of lane sec, drawn from saved files only (results/sec_*.json, tables/sec/*): no model is fitted here.

Edit FIGURES (presentation defaults) and redraw:  python -m stage2.pipeline figures --lanes sec [--only SEC_01_*]
Every number drawn in a title or an annotation, and every number of a caption or decision, is a formatted entry of
``sec_report.build_numbers()`` (the same function the numbers step writes to results/sec_numbers.json), so the figure
texts cannot drift from the numbers. Each decision is a one-line message whose wording follows the saved verdicts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stage2.lanes import sec_compute as SC
from stage2.lanes import sec_report as SR
from stage2.pipeline import Lane, selected
from stage2.plotting import HAND_COLORS, INK, INK2, MUTED, RAMP, SPEED_COLORS, figure, plain_log_ticks

L = Lane('sec')
GRIP_COLORS = {'upright': RAMP[2], 'top_left': '#eda100', 'top_right': '#6250d6'}
#: handedness markers: the report's hand palette (the colour of the person's dominant hand) plus a shape
#: (diamonds = left-handed), so the identity never rests on colour alone
LEFT_HANDED, RIGHT_HANDED = HAND_COLORS['left'], HAND_COLORS['right']
FIGURES: dict[str, dict[str, Any]] = {
    'SEC_01_speed_manipulation': dict(width='full', height=3.9, jitter=0.08, offsets={'fast': -0.25, 'natural': 0.0,
                                                                                       'slow': 0.25},
                                      dot_size=4, median_size=22, dur_ticks=(0.5, 1, 2, 5, 10),
                                      peak_ticks=(30, 50, 100, 200, 400), note_size=6.5),
    'SEC_02_speed_confusion': dict(width='full', height=2.75, cmap='Blues'),
    'SEC_03_speed_features': dict(width='full', height=3.6, colors={'fast': SPEED_COLORS['fast'],
                                                                    'slow': SPEED_COLORS['slow']}),
    'SEC_04_grip_confusion': dict(width='full', height=2.5, cmap='Blues'),
    'SEC_05_grip_physics': dict(width='full', height=3.1, dot_size=7, err_size=40, group_pad=13, xlabel_pad=16),
    'SEC_06_dominance_auc': dict(width='full', height=2.6, bins=24),
    'SEC_07_dominance_paired': dict(width='full', height=3.8, jitter=0.12),
    'SEC_08_speed_followers': dict(width='half', height=2.6, xticks=(0.5, 0.7, 1, 1.5, 2, 3)),
    'SEC_09_segmentation': dict(width='full', height=2.9, dot_size=7, ticks=(0.2, 0.5, 1, 2, 5, 10), pad=1.3,
                                colors={'oracle': INK, 'common': '#eda100'}, offset=0.12),
}
SPEED_ORDER = ('fast', 'natural', 'slow')
GRIP_ORDER = ('upright', 'top_left', 'top_right')


_NUMBERS: dict[str, dict[str, Any]] = {}


def _numbers() -> dict[str, dict[str, Any]]:
    """The lane's numbers (sec_report.build_numbers, from the saved files), built once per draw()."""
    if not _NUMBERS:
        _NUMBERS.update(SR.build_numbers())
    return _NUMBERS


def _v(key: str) -> str:
    return SR.fmt(_numbers(), key)


def _val(key: str) -> Any:
    return (_numbers().get(key) or {}).get('value')


def _confusion_panel(ax: Any, r: dict[str, Any], title: str, cmap: str) -> pd.DataFrame:
    cls = r['classes']
    cm = np.asarray(r['confusion'], dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    labels = [SR.GRIP_LABELS.get(c, c) for c in cls]
    ax.set_xticks(range(len(cls)), labels)
    ax.set_yticks(range(len(cls)), labels)
    ax.grid(False)
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax.text(j, i, f'{int(cm[i, j])}\n{share[i, j]:.1%}', ha='center', va='center', fontsize=6.5,
                    color='white' if share[i, j] > 0.55 else INK)
    ax.set_xlabel('Predicted')
    ax.set_title(title, fontsize=7.5)
    return pd.DataFrame([{'experiment': r['name'], 'instructed': cls[i], 'predicted': cls[j], 'count': int(cm[i, j]),
                          'row_share': float(share[i, j])} for i in range(len(cls)) for j in range(len(cls))])


def fig_SEC_01_speed_manipulation(**kw: Any) -> None:
    o = {**FIGURES['SEC_01_speed_manipulation'], **kw}
    T = pd.read_csv(L.tab_dir / 'sec_trial_features.csv')
    S = T[T['variant'] == 'standard'].copy()
    S['duration_s'] = np.exp(S['duration_log'])
    med = S.groupby(['person_id', 'speed'])[['duration_s', 'ang_speed_peak']].median().reset_index()
    order = med[med['speed'] == 'natural'].sort_values('duration_s')['person_id'].tolist()
    order += sorted(set(S['person_id']) - set(order))
    pos = {p: i for i, p in enumerate(order)}
    rng = np.random.default_rng(0)
    fig, axes = figure(o['width'], o['height'], nrows=2, ncols=1, sharex=True)
    rows = []
    for ax, meas, lab, ticks, key in ((axes[0], 'duration_s', 'Duration (s)', o['dur_ticks'], 'dur'),
                                      (axes[1], 'ang_speed_peak', 'Peak angular speed (deg/s)', o['peak_ticks'], 'peak')):
        for sp in SPEED_ORDER:
            sub = S[S['speed'] == sp]
            x = sub['person_id'].map(pos).to_numpy() + o['offsets'][sp] + rng.uniform(-o['jitter'], o['jitter'], len(sub))
            ax.scatter(x, sub[meas], s=o['dot_size'], color=SPEED_COLORS[sp], alpha=0.45, lw=0)
            m = med[med['speed'] == sp]
            ax.scatter(m['person_id'].map(pos) + o['offsets'][sp], m[meas], s=o['median_size'], color=SPEED_COLORS[sp],
                       edgecolor=INK, lw=0.4, zorder=3, label=f'{sp} (person median)')
            rows += [{'panel': meas, 'kind': 'person median', 'person_id': r.person_id, 'speed': sp,
                      'value': getattr(r, meas)} for r in m.itertuples()]
            rows += [{'panel': meas, 'kind': 'trial', 'person_id': r.person_id, 'speed': sp, 'trial_key': r.trial_key,
                      'value': getattr(r, meas)} for r in sub.itertuples()]
        ax.set_yscale('log')
        plain_log_ticks(ax.yaxis, ticks)
        ax.set_ylabel(lab)
        if key == 'dur':
            k1, k2 = 'manip_standard_dur_fast_natural', 'manip_standard_dur_natural_slow'
            t1, t2 = 'fast shorter than natural', 'slow longer than natural'
        else:
            k1, k2 = 'manip_standard_peak_natural_fast', 'manip_standard_peak_slow_natural'
            t1, t2 = 'fast faster than natural', 'natural faster than slow'
        extra = ''
        if key == 'dur':                 # the slow contrast rests on the label-dependent end criterion (SEC_09)
            kc = 'manip_common_dur_natural_slow'
            extra = f" ({_v(kc + '_n_expected')}/{_v(kc + '_n')} on label-blind windows)"
        ax.set_title(f"{'a' if key == 'dur' else 'b'}  {t1}: {_v(k1 + '_n_expected')}/{_v(k1 + '_n')} persons "
                     f"(Holm p {_v(k1 + '_p_holm')});  {t2}: {_v(k2 + '_n_expected')}/{_v(k2 + '_n')}{extra} "
                     f"(Holm p {_v(k2 + '_p_holm')})", loc='left', fontsize=o['note_size'], color=INK2)
    axes[1].set_xticks(range(len(order)), order, rotation=90)
    axes[1].set_xlabel('Person (ordered by the median natural duration)')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='outside upper right', ncol=3, fontsize=6.5, handletextpad=0.2)
    fk, pk = 'manip_standard_dur_fast_natural', 'manip_standard_peak_natural_fast'
    clear = (SR.ci_excludes(_numbers(), fk + '_ratio', 1.0)
             and SR.verdict(_v(fk + '_p_holm')) == 'significant')
    L.save_fig(fig, 'SEC_01_speed_manipulation', data=pd.DataFrame(rows), caption=(
        f"Manipulation check of the speed instruction: every standard-variant trial (small dots) and the per-person "
        f"median (large dots) of the approach duration (a) and the peak angular speed (b) per instructed speed, "
        f"for each of the {_v('manip_standard_n_persons')} persons; the counts are persons whose medians follow "
        "the instruction (Wilcoxon signed-rank over persons, Holm over the contrasts). Oracle windows: the app ended "
        "slow trials with a stricter criterion, so the slow durations are partly the criterion (in brackets: the "
        "count on label-blind windows; SEC_09)."),
        decision=(f"Fast reaches were faster at the peak for {_v(pk + '_n_expected')}/{_v(pk + '_n')} persons; their "
                  f"shortening (natural/fast duration ratio {_v(fk + '_ratio')} [{_v(fk + '_ratio_lo')}, "
                  f"{_v(fk + '_ratio_hi')}]) is " + ('clear.' if clear else 'uncertain (order-confounded, see the '
                                                     'position-adjusted contrasts).')))


def fig_SEC_02_speed_confusion(**kw: Any) -> None:
    o = {**FIGURES['SEC_02_speed_confusion'], **kw}
    # the label-blind models (the sensor-only estimates): duration only, the headline LR, LightGBM
    names = [m for m, s in SC.MULTICLASS.items()
             if s['task'] == 'speed' and SC.label_blind(m) and s['role'] in ('baseline', 'interpretable', 'rich')
             and s['subset'] == 'reseg_ok']
    fig, axes = figure(o['width'], o['height'], nrows=1, ncols=len(names))
    data = []
    titles = {'baseline': 'LR, duration only', 'interpretable': 'LR, kinematics (headline)', 'rich': 'LightGBM, rich'}
    for ax, name in zip(np.atleast_1d(axes), names):
        p = SR.short(name)
        t = (f"{titles[SC.MULTICLASS[name]['role']]}\nbal. acc. {_v(p + '_bacc')} "
             f"[{_v(p + '_bacc_lo')}, {_v(p + '_bacc_hi')}]\nmacro-F1 {_v(p + '_f1')}")
        data.append(_confusion_panel(ax, SC.load_multiclass(name), t, o['cmap']))
    np.atleast_1d(axes)[0].set_ylabel('Instructed')
    hs = SR.short(SC.HEADLINE['speed'])
    ho = SR.short(SC.ORACLE_TWIN[SC.HEADLINE['speed']])
    best = max(SC.SPEED_CLASSES, key=lambda c: _val(f'{hs}_f1_{c}'))
    L.save_fig(fig, 'SEC_02_speed_confusion', data=pd.concat(data, ignore_index=True), caption=(
        'Confusion matrices of the instructed speed on label-blind windows (the app\'s detector re-run with one '
        'parameter set for every trial; leave-one-person-out; counts and row shares; rows = instructed speed) for the '
        'duration-only baseline, the headline kinematic logistic regression and LightGBM on the rich features; '
        'balanced accuracy with its person-bootstrap 95% CI and the macro-F1. Per-class precision and F1: table '
        'sec_speed_classes.'),
        decision=(f"From the sensors alone the headline reaches balanced accuracy {_v(hs + '_bacc')} (macro-F1 "
                  f"{_v(hs + '_f1')}; label-informed oracle windows {_v(ho + '_bacc')}); its best F1 is {best}'s "
                  f"({_v(f'{hs}_f1_{best}')}), and the class-balanced LR over-predicts slow (precision "
                  f"{_v(hs + '_prec_slow')})."))


def fig_SEC_03_speed_features(**kw: Any) -> None:
    o = {**FIGURES['SEC_03_speed_features'], **kw}
    uv = pd.DataFrame(SR.L.read_json('sec_speed_univariate')['windows'][o.get('windows', 'common')])
    co = pd.DataFrame(SR.L.read_json('sec_speed_coef')['rows'])
    feats = list(SC.F.EVEN_FEATURES)
    order = (uv[uv['contrast'] == 'slow - natural'].set_index('feature')['median_std_diff'].abs()
             .reindex(feats).sort_values().index.tolist())
    ypos = {f: i for i, f in enumerate(order)}
    fig, axes = figure(o['width'], o['height'], nrows=1, ncols=2, sharey=True)
    rows = []
    for c, dy in (('fast', 0.17), ('slow', -0.17)):
        u = uv[uv['contrast'] == f'{c} - natural']
        y = u['feature'].map(ypos) + dy
        axes[0].errorbar(u['median_std_diff'], y, xerr=[u['median_std_diff'] - u['q25'], u['q75'] - u['median_std_diff']],
                         fmt='o', ms=3, color=o['colors'][c], elinewidth=0.8, capsize=0, label=f'{c} - natural')
        k = co[co['contrast'] == f'{c} vs natural']
        y2 = k['feature'].map(ypos) + dy
        axes[1].errorbar(k['mean'], y2, xerr=[k['mean'] - k['min'], k['max'] - k['mean']], fmt='o', ms=3,
                         color=o['colors'][c], elinewidth=0.8, capsize=0, label=f'{c} vs natural')
        rows += [{'panel': 'within-person', 'feature': r.feature, 'contrast': r.contrast, 'value': r.median_std_diff,
                  'lo': r.q25, 'hi': r.q75} for r in u.itertuples()]
        rows += [{'panel': 'lr', 'feature': r.feature, 'contrast': r.contrast, 'value': r.mean, 'lo': r.min,
                  'hi': r.max} for r in k.itertuples()]
    for ax in axes:
        ax.axvline(0, color=MUTED, lw=0.6)
    axes[0].set_yticks(range(len(order)), [SR.FEATURE_TEXT[f] for f in order])
    axes[0].set_xlabel('Within-person difference of medians (SD)')
    axes[1].set_xlabel('LR log-odds per SD (mean, range over folds)')
    axes[0].set_title('a  Model-free (standard-variant trials)', loc='left', fontsize=7.5)
    axes[1].set_title('b  Headline logistic regression', loc='left', fontsize=7.5)
    axes[0].legend(loc='lower right', fontsize=6.5)
    L.save_fig(fig, 'SEC_03_speed_features', data=pd.DataFrame(rows), caption=(
        'Which kinematic features carry the instructed speed, on label-blind windows. a: per person the difference of '
        'the medians between two instructed speeds in units of the feature\'s SD (median and IQR over persons; '
        'standard-variant trials). b: the log-odds contrasts of the headline logistic regression per SD of each '
        'feature (mean and range over the outer folds; conditional on the other, correlated features). Features '
        'ordered by the slow effect.'),
        decision=(f"Fast differs from natural most in {_v('uni_common_fast_top1_feature')} and "
                  f"{_v('uni_common_fast_top2_feature')}, slow in {_v('uni_common_slow_top1_feature')} and "
                  f"{_v('uni_common_slow_top2_feature')} (within-person, model-free, label-blind windows)."))


def fig_SEC_04_grip_confusion(**kw: Any) -> None:
    o = {**FIGURES['SEC_04_grip_confusion'], **kw}
    names = list(SC.GRIP_MODELS)
    fig, axes = figure(o['width'], o['height'], nrows=1, ncols=len(names))
    titles = {'baseline': 'LR, up vector before onset', 'interpretable': 'LR, orientation (headline)',
              'rich': 'LightGBM, rich'}
    data = []
    for ax, name in zip(np.atleast_1d(axes), names):
        p = SR.short(name)
        t = (f"{titles[SC.MULTICLASS[name]['role']]}\nbal. acc. {_v(p + '_bacc')} "
             f"[{_v(p + '_bacc_lo')}, {_v(p + '_bacc_hi')}]")
        data.append(_confusion_panel(ax, SC.load_multiclass(name), t, o['cmap']))
    np.atleast_1d(axes)[0].set_ylabel('Instructed')
    gp = SR.short(SC.HEADLINE['grip'])
    clean = _val(gp + '_ref_n_errors_confirmed') == 0
    L.save_fig(fig, 'SEC_04_grip_confusion', data=pd.concat(data, ignore_index=True), caption=(
        'Confusion matrices of the instructed grip (leave-one-person-out; counts and row shares; rows = instructed '
        'grip): the up vector before the onset alone, the headline orientation logistic regression and LightGBM on the '
        'rich features; balanced accuracy with its person-bootstrap 95% CI.'),
        decision=(f"The grip is recognised with balanced accuracy {_v(gp + '_bacc')} against the instruction; "
                  + (f"its {_v(gp + '_n_errors')} errors ({_v(gp + '_n_sideways_swaps')} sideways swaps) look like "
                     'trials held in another grip than instructed (physics and consistency checks; no independent '
                     'record).' if clean else
                     f"{_v(gp + '_ref_n_errors_confirmed')} errors fall on trials whose grip the app confirmed.")))


def fig_SEC_05_grip_physics(**kw: Any) -> None:
    o = {**FIGURES['SEC_05_grip_physics'], **kw}
    T = pd.read_csv(L.tab_dir / 'sec_trial_features.csv')
    ge = SR.L.read_json('sec_grip_errors')
    head = SC.HEADLINE['grip']
    gp = SR.short(head)
    E = pd.DataFrame(ge['models'][head]['errors'])
    err = set(E['trial_key']) if len(E) else set()
    fig, axes = figure(o['width'], o['height'], nrows=1, ncols=2)
    rows = []
    for g in GRIP_ORDER:
        sub = T[T['grip'] == g]
        axes[0].scatter(sub['up_before_x'], sub['up_before_y'], s=o['dot_size'], color=GRIP_COLORS[g], alpha=0.6, lw=0,
                        label=SR.GRIP_LABELS[g])
        rows += [{'trial_key': r.trial_key, 'grip': g, 'hand': r.hand, 'up_before_x': r.up_before_x,
                  'up_before_y': r.up_before_y, 'misclassified': r.trial_key in err} for r in sub.itertuples()]
    if len(E):
        axes[0].scatter(E['up_before_x'], E['up_before_y'], s=o['err_size'], facecolor='none', edgecolor=INK, lw=0.7,
                        label='misclassified')
    axes[0].set_xlabel('Up vector before the onset, device x')
    axes[0].set_ylabel('Up vector, device y')
    axes[0].set_xlim(-1.15, 1.15)
    axes[0].set_ylim(-0.3, 1.15)
    axes[0].legend(loc='lower center', ncol=2, fontsize=6.5)
    axes[0].set_title('a  Where gravity points in the device frame', loc='left', fontsize=7.5)
    # b: the device-x component per instructed grip and hand, the misclassified trials ringed
    hands = ('left', 'right')
    xs = {(g, h): 3 * gi + hi for gi, g in enumerate(GRIP_ORDER) for hi, h in enumerate(hands)}
    rng = np.random.default_rng(1)
    for (g, h), x in xs.items():
        sub = T[(T['grip'] == g) & (T['hand'] == h)]
        xx = x + rng.uniform(-0.25, 0.25, len(sub))
        axes[1].scatter(xx, sub['up_before_x'], s=o['dot_size'], color=GRIP_COLORS[g], alpha=0.6, lw=0)
        bad = sub['trial_key'].isin(err).to_numpy()
        if bad.any():
            axes[1].scatter(xx[bad], sub['up_before_x'].to_numpy()[bad], s=o['err_size'], facecolor='none',
                            edgecolor=INK, lw=0.7)
    axes[1].set_xticks(list(xs.values()), [h for _, h in xs])
    axes[1].tick_params(axis='x', labelsize=6)
    axes[1].set_xlabel('Instructed grip and hand', labelpad=o['xlabel_pad'])      # below the second tick row
    grp = axes[1].secondary_xaxis('bottom')                     # second tick row: the instructed grip of each pair
    grp.set_xticks([3 * gi + 0.5 for gi in range(len(GRIP_ORDER))], [SR.GRIP_LABELS[g] for g in GRIP_ORDER])
    grp.tick_params(axis='x', length=0, pad=o['group_pad'], labelsize=7)
    grp.spines['bottom'].set_visible(False)
    axes[1].set_ylabel('Up vector before the onset, device x')
    axes[1].axhline(0, color=MUTED, lw=0.6)
    axes[1].set_title(f"b  {_v(gp + '_n_sideways_swaps')} of {_v(gp + '_n_errors')} errors are sideways swaps",
                      loc='left', fontsize=7.5)
    all_match = _val(gp + '_n_up_matches_predicted') == _val(gp + '_n_errors')
    L.save_fig(fig, 'SEC_05_grip_physics', data=pd.DataFrame(rows), caption=(
        'Why the grip is recognised and where it fails: the unit up vector in the device frame during the still hold '
        'before the onset, per trial, coloured by the instructed grip (a: x against y; b: x per instructed grip and '
        'hand). Upright puts the top edge (+y) up, top left the right edge (+x), top right the left edge (-x). Ringed: '
        f"the trials the headline grip model misclassifies; in {_v(gp + '_n_up_matches_predicted')} of the "
        f"{_v(gp + '_n_errors')} the up vector lies closer to the median of the predicted grip than to that of the "
        'instructed one (the phone was held in the predicted grip).'),
        decision=('The up vector of the still hold separates the three grips; '
                  + (f"every misclassified trial ({_v(gp + '_n_errors')}) sits in the cluster of the predicted grip."
                     if all_match else
                     f"{_v(gp + '_n_up_matches_predicted')} of {_v(gp + '_n_errors')} misclassified trials sit in the "
                     'cluster of the predicted grip.')))


def fig_SEC_06_dominance_auc(**kw: Any) -> None:
    o = {**FIGURES['SEC_06_dominance_auc'], **kw}
    nl = SR.L.read_json('sec_dom_null')
    nt = SR.L.read_json('sec_dom_null_test')
    dp = pd.DataFrame(SR.L.read_json('sec_dom_persons')['per_person'])
    fig, axes = figure(o['width'], o['height'], nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 1.4]})
    null = np.asarray(nl['pooled_auc_null'])
    axes[0].hist(null, bins=o['bins'], color=MUTED, alpha=0.7, label='permuted labels')
    axes[0].axvline(nt['pooled_observed'], color=INK, lw=1.2, label='observed')
    axes[0].axvline(0.5, color=MUTED, lw=0.6, ls='--', label='AUC 0.5')
    axes[0].set_xlabel('Pooled AUC (dominant hand)')
    axes[0].set_ylabel('Permutations')
    axes[0].set_title(f"a  AUC {_v('dom_auc')}, permutation p {_v('dom_p_pooled')}", loc='left', fontsize=7.5)
    axes[0].legend(loc='upper left', fontsize=6.5)
    d = dp.sort_values('auc').reset_index(drop=True)
    lh = d['handedness'] == 'left'
    axes[1].scatter(np.flatnonzero(~lh), d.loc[~lh, 'auc'], s=14, color=RIGHT_HANDED, label='right-handed', zorder=3)
    axes[1].scatter(np.flatnonzero(lh), d.loc[lh, 'auc'], s=22, marker='D', color=LEFT_HANDED, label='left-handed',
                    zorder=3)
    axes[1].axhline(0.5, color=MUTED, lw=0.6, ls='--', label='chance')
    axes[1].set_xticks(range(len(d)), d['person_id'], rotation=90)
    axes[1].set_ylabel('Within-person AUC')
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"b  mean {_v('dom_within_auc')}, permutation p {_v('dom_p_within')}", loc='left', fontsize=7.5)
    axes[1].legend(loc='lower right', fontsize=6.5, ncol=3)
    data = pd.concat([pd.DataFrame({'panel': 'permutation', 'value': null}),
                      pd.DataFrame({'panel': 'person', 'person_id': d['person_id'], 'value': d['auc'],
                                    'handedness': d['handedness']})], ignore_index=True)
    sig = SR.verdict(_v('dom_p_pooled')) == 'significant'
    L.save_fig(fig, 'SEC_06_dominance_auc', data=data, caption=(
        'Can mirror-invariant kinematics tell a dominant-hand trial (leave-one-person-out, nested logistic regression)? '
        f"a: the pooled AUC against its label-permutation null ({_v('dom_n_perm')} permutations within person, "
        'the whole nested pipeline re-run; dashed: AUC 0.5, which is not the reference). b: the AUC within each '
        'person (dashed: 0.5); diamonds: the left-handed persons, shown individually (for the right-handed the '
        'dominant hand is the right hand).'),
        decision=(('A dominant-hand signal is detected' if sig else
                   'No dominant-hand signal detected with the pre-specified model')
                  + f": pooled AUC {_v('dom_auc')} [{_v('dom_auc_lo')}, {_v('dom_auc_hi')}] against a permutation "
                  f"null mean of {_v('dom_null_pooled_mean')} (p {_v('dom_p_pooled')}); an exploratory LightGBM "
                  f"reached {_v('domgbm_auc')} without a permutation baseline."))


def fig_SEC_07_dominance_paired(**kw: Any) -> None:
    o = {**FIGURES['SEC_07_dominance_paired'], **kw}
    pdd = SR.L.read_json('sec_dom_paired')
    per = pd.DataFrame(pdd['variants']['unmatched']['per_person'])
    rows = {r['feature']: r for r in pdd['variants']['unmatched']['rows']}
    feats = pdd['features']
    ncols = 3
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = figure(o['width'], o['height'], nrows=nrows, ncols=ncols)
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(2)
    out = []
    for ax, f in zip(axes, feats):
        sub = per[per['feature'] == f].copy()
        ratio = rows[f]['ratio']
        val = (np.exp(sub['diff']) - 1.0) * 100 if ratio else sub['diff']
        x = rng.uniform(-o['jitter'] * 1.5, o['jitter'] * 1.5, len(sub))
        lh = (sub['handedness'] == 'left').to_numpy()
        ax.scatter(x[~lh], val[~lh], s=10, color=RIGHT_HANDED, lw=0, zorder=3)
        ax.scatter(x[lh], val[lh], s=18, marker='D', color=LEFT_HANDED, zorder=3)
        for xi, vi, pid in zip(x[lh], np.asarray(val)[lh], sub['person_id'][lh]):
            ax.annotate(pid, (xi, vi), xytext=(4, 0), textcoords='offset points', fontsize=6, color=LEFT_HANDED,
                        va='center')
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.hlines(np.median(val), -0.3, 0.3, color=INK, lw=1.2)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('dominant - non-dominant (%)' if ratio else 'dominant - non-dominant', fontsize=6.5)
        ax.set_title(f"{SR.FEATURE_TEXT[f]}\nHolm p {_v('pd_' + f + '_p_holm')}", fontsize=7)
        out += [{'feature': f, 'person_id': pid, 'handedness': h, 'value': float(vv), 'scale': 'percent' if ratio else
                 'difference'} for pid, h, vv in zip(sub['person_id'], sub['handedness'], val)]
    for ax in axes[len(feats):]:
        ax.set_visible(False)
    none_sig = _val('pd_n_sig') == 0
    L.save_fig(fig, 'SEC_07_dominance_paired', data=pd.DataFrame(out), caption=(
        'Dominant vs non-dominant movements within each person: per person the difference of the medians (dominant - '
        'non-dominant; a percent change for the duration, the peak angular speed and the path lengths), the bar is '
        f"the median over persons; Wilcoxon signed-rank over persons, Holm over the {_v('pd_n_features')} "
        'features. Diamonds: the left-handed persons (their dominant hand is the left).'),
        decision=(f"No detectable difference between the hands within persons (Holm); the CIs still allow e.g. "
                  f"duration {_v('pd_duration_log_lo')} to {_v('pd_duration_log_hi')} and angular path "
                  f"{_v('pd_ang_path_lo')} to {_v('pd_ang_path_hi')}." if none_sig else
                  f"{_v('pd_n_sig')} of {_v('pd_n_features')} features differ between the dominant and the "
                  'non-dominant hand within persons (Holm).'))


def fig_SEC_08_speed_followers(**kw: Any) -> None:
    o = {**FIGURES['SEC_08_speed_followers'], **kw}
    fo = SR.L.read_json('sec_speed_followers')
    D = pd.DataFrame(fo['per_person'])
    fig, ax = figure(o['width'], o['height'])
    fol = D['log_ratio'] > 0
    ax.scatter(D.loc[fol, 'ratio'], D.loc[fol, 'bacc_fast_natural'], s=16, color=SPEED_COLORS['fast'], zorder=3,
               label='fast shorter than natural')
    ax.scatter(D.loc[~fol, 'ratio'], D.loc[~fol, 'bacc_fast_natural'], s=16, facecolor='white',
               edgecolor=SPEED_COLORS['fast'], lw=0.8, zorder=3, label='fast not shorter')
    for r in D[~fol].itertuples():                              # the persons who did not follow, by name
        ax.annotate(r.person_id, (r.ratio, r.bacc_fast_natural), xytext=(3, 2), textcoords='offset points',
                    fontsize=5.5, color=INK2)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls='--')
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=':')
    ax.set_xscale('log')
    plain_log_ticks(ax.xaxis, o['xticks'])
    ax.set_xlabel('Median natural / fast duration (per person)')
    ax.set_ylabel('Fast vs natural, two-class bal. acc.')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Spearman rho {_v('fol_rho')} (p {_v('fol_p')})", loc='left', fontsize=7.5)
    ax.legend(loc='lower right', fontsize=6.5)
    L.save_fig(fig, 'SEC_08_speed_followers', data=D, caption=(
        'Description, not evidence: per person, how much shorter the fast reaches were than the natural ones (median '
        'duration ratio, standard-variant trials, label-blind windows; right of the dashed line = shorter) against how '
        'well the headline speed model tells that person\'s fast trials from the natural ones on the same trials '
        '(the class decided between fast and natural only, mean of the two recalls; dotted line = chance 0.5). '
        'Partly circular: the duration is an input of the model. The persons whose fast reaches were not shorter '
        'are named.'),
        decision=(f"Descriptive: two-class fast-vs-natural balanced accuracy {_v('fol_bacc_followed')} for the "
                  f"{_v('fol_n_followed')} persons whose fast reaches were shorter, {_v('fol_bacc_not')} for the other "
                  f"{_v('fol_n_not')} (partly circular)."))


def fig_SEC_09_segmentation(**kw: Any) -> None:
    o = {**FIGURES['SEC_09_segmentation'], **kw}
    R = pd.read_csv(L.tab_dir / f'{SC.RESEG_TABLE}.csv')
    R = R[(R['common_ok'].astype(str).str.lower() == 'true') & (R['set'] == 'primary')]
    angc, ang = SC.SEGMENTATION_PAIR
    pa, pac, ksg = SR.short(ang), SR.short(angc), SR.pair_key(angc, ang)
    fig, axes = figure(o['width'], o['height'], nrows=1, ncols=2, gridspec_kw={'width_ratios': [1.15, 1]})
    rows = []
    for sp in SPEED_ORDER:                                  # a: per trial, oracle vs label-independent duration
        sub = R[R['speed'] == sp]
        axes[0].scatter(sub['duration_s'], sub['common_duration_s'], s=o['dot_size'], color=SPEED_COLORS[sp],
                        alpha=0.7, lw=0, label=sp, zorder=3)
        rows += [{'panel': 'duration', 'trial_key': r.trial_key, 'speed': sp, 'oracle_s': r.duration_s,
                  'common_s': r.common_duration_s} for r in sub.itertuples()]
    both = pd.concat([R['duration_s'], R['common_duration_s']]).astype(float)
    lim = (float(both.min()) / o['pad'], float(both.max()) * o['pad'])     # from the data: no point is dropped
    ticks = [t for t in o['ticks'] if lim[0] <= t <= lim[1]]
    axes[0].plot(lim, lim, color=MUTED, lw=0.6, ls='--', zorder=1)
    for ax_set in (axes[0].set_xscale, axes[0].set_yscale):
        ax_set('log')
    axes[0].set_xlim(*lim)
    axes[0].set_ylim(*lim)
    plain_log_ticks(axes[0].xaxis, ticks)
    plain_log_ticks(axes[0].yaxis, ticks)
    axes[0].set_xlabel('Duration, oracle window (s)')
    axes[0].set_ylabel('Duration, label-blind (s)')
    axes[0].legend(loc='upper left', fontsize=6.5, title='instructed', title_fontsize=6.5)
    axes[0].set_title(f"a  slow: median {_v('reseg_slow_dur')} s, label-blind {_v('reseg_slow_dur_common')} s",
                      loc='left', fontsize=7.5)
    x = np.arange(len(SPEED_ORDER))                         # b: recall per class, the same model on both windows
    for key, p, dx, lab in (('oracle', pa, -o['offset'], 'oracle windows'),
                            ('common', pac, o['offset'], 'label-blind windows')):
        rec = [_val(f'{p}_rec_{c}') for c in SPEED_ORDER]
        lo = [_val(f'{p}_rec_{c}_lo') for c in SPEED_ORDER]
        hi = [_val(f'{p}_rec_{c}_hi') for c in SPEED_ORDER]
        axes[1].errorbar(x + dx, rec, yerr=[np.subtract(rec, lo), np.subtract(hi, rec)], fmt='o', ms=4, capsize=0,
                         elinewidth=0.9, color=o['colors'][key], label=lab)
        rows += [{'panel': 'recall', 'windows': key, 'speed': c, 'recall': r_, 'lo': l_, 'hi': h_}
                 for c, r_, l_, h_ in zip(SPEED_ORDER, rec, lo, hi)]
    axes[1].axhline(1 / len(SPEED_ORDER), color=MUTED, lw=0.6, ls=':')
    axes[1].set_xticks(x, list(SPEED_ORDER))
    axes[1].set_xlim(-0.5, len(SPEED_ORDER) - 0.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Instructed speed')
    axes[1].set_ylabel('Recall (95% CI)')
    axes[1].legend(loc='lower left', fontsize=6.5)
    axes[1].set_title(f"b  balanced accuracy {_v(pa + '_bacc')} vs {_v(pac + '_bacc')}", loc='left', fontsize=7.5)
    drop = SR.verdict(_v(ksg + '_d_bacc_p_holm')) == 'significant' and (_val(ksg + '_d_bacc') or 0) < 0
    L.save_fig(fig, 'SEC_09_segmentation', data=pd.DataFrame(rows), caption=(
        "The app chose its segmentation parameters per instructed speed. a: per trial, the approach duration of the "
        "app's stored segment (the oracle window; a slow trial ends only after a stricter, longer stillness) against "
        "that of the app's own detector re-run with one parameter set for every trial (label-blind; dashed: equal). "
        "b: recall per instructed speed of the headline kinematic logistic regression fitted on each kind of window "
        "(leave-one-person-out, person-bootstrap 95% CIs; dotted: chance)."),
        decision=(f"With one parameter set a slow reach ends as early as a natural one ({_v('reseg_slow_dur_common')} "
                  f"s against {_v('reseg_natural_dur_common')} s), and the headline model falls from "
                  f"{_v(pa + '_bacc')} to {_v(pac + '_bacc')} balanced accuracy: the oracle-window speed results are "
                  'optimistic (the segmentation used the label).'
                  if drop else
                  f"The headline model reaches {_v(pac + '_bacc')} on label-blind windows against "
                  f"{_v(pa + '_bacc')} on the oracle windows: the speed result does not rest on the segmentation."))


FIGURE_FUNCTIONS = {
    'SEC_01_speed_manipulation': fig_SEC_01_speed_manipulation,
    'SEC_02_speed_confusion': fig_SEC_02_speed_confusion,
    'SEC_03_speed_features': fig_SEC_03_speed_features,
    'SEC_04_grip_confusion': fig_SEC_04_grip_confusion,
    'SEC_05_grip_physics': fig_SEC_05_grip_physics,
    'SEC_06_dominance_auc': fig_SEC_06_dominance_auc,
    'SEC_07_dominance_paired': fig_SEC_07_dominance_paired,
    'SEC_08_speed_followers': fig_SEC_08_speed_followers,
    'SEC_09_segmentation': fig_SEC_09_segmentation,
}


def draw(only: list[str] | None = None) -> None:
    _NUMBERS.clear()
    for name, fn in FIGURE_FUNCTIONS.items():
        if selected(name, only):
            fn()
