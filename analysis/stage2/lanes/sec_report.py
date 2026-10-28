"""Lane sec: numbers, tables and the summary, from the saved files only (nothing here fits a model).

``build_numbers()`` is the single source of every number the lane cites: the numbers step writes it to
results/sec_numbers.json, the summary and the figure annotations format their values from it, and
``untraced_numbers`` checks that every numeric token of the summary is one of its formatted values. Verdict
sentences are generated from saved p-values (Holm-adjusted within a named family) and saved flags, never typed.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import report_export as RX
from stage2.lanes import sec_compute as SC
from stage2.lanes import sec_features as F
from stage2.lanes import sec_models as M
from stage2.lanes import sec_stats as SS
from stage2.pipeline import Lane

L = Lane('sec')
ALPHA = 0.05
#: below this many persons a group gets no model and no bootstrap CI: its persons are shown individually
MIN_PERSONS_PER_CLASS = 5
MEASURE_SHORT = {'duration_log': 'dur', 'ang_speed_peak': 'peak'}
GRIP_LABELS = {'upright': 'upright', 'top_left': 'top left', 'top_right': 'top right'}
FEATURE_TEXT = {
    'duration_log': 'duration', 'ang_speed_peak': 'peak angular speed', 'ang_speed_mean': 'mean angular speed',
    'ang_peak_time': 'time of the angular-speed peak', 'ang_path': 'angular path length', 'ang_sparc': 'SPARC',
    'ang_ldlj': 'log dimensionless jerk', 'ang_n_peaks': 'angular-speed peaks', 'tilt_net': 'net tilt',
    'yaw_net_abs': 'absolute net yaw', 'tilt_path': 'tilt path', 'yaw_path': 'yaw path',
    'acc_peak': 'peak linear acceleration', 'acc_mean': 'mean linear acceleration', 'lin_path': 'linear path length',
    'lin_speed_peak': 'peak linear speed'}
PAIR_METRIC_TEXT = {'d_bacc': 'balanced accuracy', 'd_f1': 'macro-F1', 'd_acc': 'accuracy', 'd_ll': 'log loss'}
MANIP_SUBSET_TEXT = {'standard': 'standard variant', 'all': 'all primary',
                     'common': 'standard variant, label-independent windows'}


def src_res(name: str) -> str:
    return f'results/{name}.json'


def src_tab(name: str, ext: str = 'json') -> str:
    return f'tables/sec/{name}.{ext}'


def fmt_p(p: Any) -> str:
    """A p-value as printed everywhere in this lane: '< 0.001' below 0.001, else three decimals."""
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return '--'
    p = float(p)
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def short(name: str) -> str:
    return name[len('sec_'):]


def pair_key(a: str, b: str) -> str:
    return f'pair_{short(a)}_vs_{short(b)}'


def ratio_pct(v: float) -> float:
    """A difference of logs as a percent change (exp(v) - 1)."""
    return float(math.exp(v) - 1.0)


# ───────────────────────────── numbers ─────────────────────────────


def _numbers_data(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> pd.DataFrame:
    T = pd.read_csv(L.tab_dir / 'sec_trial_features.csv')
    st = src_tab('sec_trial_features', 'csv')
    n['n_trials'] = E(len(T), 'd', st, 'primary trials')
    n['n_persons'] = E(int(T['person_id'].nunique()), 'd', st, 'persons')
    pids = sorted(T['person_id'].astype(str).unique())
    n['person_first'] = E(pids[0], '', st, 'first person code (study codes only)')
    n['person_last'] = E(pids[-1], '', st, 'last person code (study codes only)')
    n['chance_speed'] = E(1.0 / len(SC.SPEED_CLASSES), '.1%', 'stage2/lanes/sec_compute.py',
                          'chance balanced accuracy of the speed task (1 / number of classes)')
    n['chance_grip'] = E(1.0 / len(SC.GRIP_CLASSES), '.1%', 'stage2/lanes/sec_compute.py',
                         'chance balanced accuracy of the grip task (1 / number of classes)')
    n['chance_side'] = E(1.0 / len(SC.SIDE_CLASSES), '.1%', 'stage2/lanes/sec_compute.py',
                         'chance balanced accuracy of the sideways-vs-upright task (1 / number of classes)')
    for c in SC.SPEED_CLASSES:
        n[f'n_speed_{c}'] = E(int((T['speed'] == c).sum()), 'd', st, f'primary trials instructed {c}')
    for g in SC.GRIP_CLASSES:
        n[f'n_grip_{g}'] = E(int((T['grip'] == g).sum()), 'd', st, f'primary trials instructed grip {g}')
    std = T[T['variant'] == 'standard']
    n['n_standard'] = E(len(std), 'd', st, 'primary trials of the standard variant')
    n['n_fastslow_nonstandard'] = E(int((T['speed'].isin(['fast', 'slow']) & (T['variant'] != 'standard')).sum()), 'd',
                                    st, 'fast or slow trials of a variant other than standard')
    slow_pp = T[T['speed'] == 'slow'].groupby('person_id').size().reindex(T['person_id'].unique(), fill_value=0)
    n['slow_per_person_median'] = E(float(slow_pp.median()), 'g', st, 'slow trials per person, median')
    n['slow_per_person_max'] = E(int(slow_pp.max()), 'd', st, 'slow trials per person, maximum')
    n['n_persons_no_slow'] = E(int((slow_pp == 0).sum()), 'd', st, 'persons without a slow trial')
    n['n_even_features'] = E(len(F.EVEN_FEATURES), 'd', 'stage2/lanes/sec_features.py', 'mirror-invariant features')
    n['n_grip_features'] = E(len(F.GRIP_FEATURES), 'd', 'stage2/lanes/sec_features.py', 'orientation features')
    n['n_boot'] = E(int(C.N_BOOT), 'd', 'stage2/config.py', 'person-bootstrap resamples of every CI')
    n['min_persons_ci'] = E(MIN_PERSONS_PER_CLASS, 'd', 'stage2/lanes/sec_report.py',
                            'fewer persons than this: no bootstrap CI, persons shown individually')
    n['ll_precision'] = E(SS.LL_PRECISION, 'g', 'stage2/lanes/sec_stats.py',
                          "'identical': no class change and |log-loss difference| below this")
    return T


def _numbers_models(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    for name in SC.MULTICLASS:
        r = SC.load_multiclass(name)
        p, s = short(name), src_res(name)
        m, ci = r['metrics'], r['ci95']
        n[f'{p}_n'] = E(int(m['n']), 'd', s, f'{name}: trials')
        n[f'{p}_n_features'] = E(len(r['features']), 'd', s, f'{name}: input features')
        for key, fmt, what in (('balanced_accuracy', '.1%', 'bacc'), ('macro_f1', '.3f', 'f1'),
                               ('accuracy', '.1%', 'acc'), ('log_loss', '.3f', 'll')):
            n[f'{p}_{what}'] = E(m[key], fmt, s, f'{name}: {key} (leave-one-person-out)')
            if key in ci:
                n[f'{p}_{what}_lo'] = E(ci[key][0], fmt, s, f'{name}: {key}, person-bootstrap 95% CI low')
                n[f'{p}_{what}_hi'] = E(ci[key][1], fmt, s, f'{name}: {key}, person-bootstrap 95% CI high')
        cls = r['classes']
        cm = np.asarray(r['confusion'])
        for i, c in enumerate(cls):
            n[f'{p}_rec_{c}'] = E(m[f'recall_{c}'], '.1%', s, f'{name}: recall of {c}')
            n[f'{p}_rec_{c}_lo'] = E(ci[f'recall_{c}'][0], '.1%', s, f'{name}: recall of {c}, CI low')
            n[f'{p}_rec_{c}_hi'] = E(ci[f'recall_{c}'][1], '.1%', s, f'{name}: recall of {c}, CI high')
            n[f'{p}_n_{c}'] = E(int(cm[i].sum()), 'd', s, f'{name}: trials of class {c}')
            n[f'{p}_prec_{c}'] = E(m[f'precision_{c}'], '.1%', s, f'{name}: precision of {c}')
            n[f'{p}_f1_{c}'] = E(m[f'f1_{c}'], '.3f', s, f'{name}: F1 of {c}')
            n[f'{p}_npred_{c}'] = E(int(cm[:, i].sum()), 'd', s, f'{name}: trials predicted {c}')
            for j, q in enumerate(cls):
                n[f'{p}_cm_{c}_{q}'] = E(int(cm[i, j]), 'd', s, f'{name}: {c} trials predicted {q}')
                n[f'{p}_cmr_{c}_{q}'] = E(float(cm[i, j] / max(cm[i].sum(), 1)), '.1%', s,
                                          f'{name}: share of {c} trials predicted {q}')
        cfg = pd.Series([', '.join(f'{a} {b:g}' for a, b in c.items()) if isinstance(c, dict) else f'C = {c:g}'
                         for c in (v['config'] for v in r['fold_info'].values())]).value_counts()
        n[f'{p}_config_mode'] = E(cfg.index[0], '', s, f'{name}: the configuration chosen in most outer folds')
        n[f'{p}_config_mode_folds'] = E(int(cfg.iloc[0]), 'd', s, f'{name}: outer folds choosing it')
        n[f'{p}_n_folds'] = E(int(r['n_folds']), 'd', s, f'{name}: outer folds')
        if 'extra' in r:
            rows = r['extra']['rows']
            n[f'{p}_nat_n'] = E(int(r['extra']['n']), 'd', s, f'{name}: free-choice trials scored')
            n[f'{p}_nat_instructed_natural'] = E(int(sum(x['instructed'] == 'natural' for x in rows)), 'd', s,
                                                 f'{name}: free-choice trials whose speed field is natural')
            for c in cls:
                n[f'{p}_nat_{c}'] = E(int(r['extra']['predicted_counts'][c]), 'd', s,
                                      f'{name}: free-choice trials predicted {c}')
            if rows:
                n[f'{p}_nat_share_natural'] = E(r['extra']['predicted_counts'].get('natural', 0) / len(rows), '.1%',
                                                s, f'{name}: share of the free-choice trials predicted natural '
                                                '(natural-speed recall on free-choice trials only)')


def _numbers_pairs(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    pr = L.read_json('sec_pairs')
    s = src_tab('sec_pairs')
    for row in pr['pairs']:
        k = pair_key(row['a'], row['b'])
        n[f'{k}_n'] = E(int(row['n_trials']), 'd', s, f"{row['a']} vs {row['b']}: trials both scored ({row['scope']})")
        n[f'{k}_bacc_a'] = E(row['bacc_a'], '.1%', s, f"{row['a']}: balanced accuracy on the shared trials")
        n[f'{k}_bacc_b'] = E(row['bacc_b'], '.1%', s, f"{row['b']}: balanced accuracy on the shared trials")
        n[f'{k}_f1_a'] = E(row['macro_f1_a'], '.3f', s, f"{row['a']}: macro-F1 on the shared trials")
        n[f'{k}_f1_b'] = E(row['macro_f1_b'], '.3f', s, f"{row['b']}: macro-F1 on the shared trials")
        for key, fm, what in (('d_bacc', '+.1%', 'pooled balanced-accuracy difference A - B'),
                              ('d_f1', '+.3f', 'pooled macro-F1 difference A - B (person-swap test)'),
                              ('d_acc', '+.1%', 'mean per-person accuracy difference A - B'),
                              ('d_ll', '+.3f', 'mean per-person log-loss difference A - B')):
            n[f'{k}_{key}'] = E(row[key], fm, s, f"{row['a']} - {row['b']}: {what}")
            n[f'{k}_{key}_lo'] = E(row[f'{key}_ci'][0], fm, s, 'CI low (person bootstrap)')
            n[f'{k}_{key}_hi'] = E(row[f'{key}_ci'][1], fm, s, 'CI high (person bootstrap)')
            n[f'{k}_{key}_p_holm'] = E(fmt_p(row[f'{key}_p_holm']), '', s,
                                       f'p over persons (sign-flip; person swap for macro-F1), Holm over the '
                                       f'{PAIR_METRIC_TEXT[key]} family')
        n[f'{k}_n_changed'] = E(int(row['n_changed']), 'd', s, 'trials whose predicted class differs')
        n[f'{k}_identical'] = E('identical' if row['identical'] else 'not identical', '', s,
                                'no trial changes class and |log-loss difference| below the solver precision')
    n['n_model_pairs'] = E(len(pr['pairs']), 'd', s, 'paired model comparisons per Holm family')
    dm = pr['dominance']
    for key, src_key, fm in (('d_acc', 'mean_person_acc_diff', '+.1%'), ('d_ll', 'mean_person_logloss_diff', '+.3f')):
        n[f'pair_dom_{key}'] = E(dm[src_key], fm, s, f"{dm['a']} - {dm['b']}: {src_key.replace('_', ' ')} "
                                 '(cv.paired_compare, exploratory)')
        n[f'pair_dom_{key}_lo'] = E(dm[f'{src_key}_ci95'][0], fm, s, 'CI low (person bootstrap)')
        n[f'pair_dom_{key}_hi'] = E(dm[f'{src_key}_ci95'][1], fm, s, 'CI high (person bootstrap)')
    n['pair_dom_d_acc_p'] = E(fmt_p(dm['p_sign_flip_acc']), '', s, 'exact sign-flip p (one exploratory comparison)')
    n['pair_dom_d_ll_p'] = E(fmt_p(dm['p_sign_flip_logloss']), '', s, 'exact sign-flip p (one exploratory comparison)')


def _numbers_speed(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    mp = L.read_json('sec_speed_manip')
    s = src_tab('sec_speed_manip')
    for sub, d in mp['subsets'].items():
        n[f'manip_{sub}_n_trials'] = E(d['n_trials'], 'd', s, f'manipulation check ({sub}): trials')
        n[f'manip_{sub}_n_persons'] = E(d['n_persons'], 'd', s, f'manipulation check ({sub}): persons')
        for c in d['contrasts']:
            k = f"manip_{sub}_{MEASURE_SHORT[c['measure']]}_{c['low']}_{c['high']}"
            n[f'{k}_n'] = E(c['n_persons'], 'd', s, f'persons with both speeds ({c["measure"]})')
            n[f'{k}_n_expected'] = E(c['n_expected'], 'd', s, 'persons in the instructed order')
            n[f'{k}_ratio'] = E(c['median_ratio'], '.2f', s, 'median per-person ratio (expected direction > 1)')
            n[f'{k}_ratio_lo'] = E(c['median_ratio_lo'], '.2f', s, 'that median ratio, person-bootstrap CI low')
            n[f'{k}_ratio_hi'] = E(c['median_ratio_hi'], '.2f', s, 'that median ratio, person-bootstrap CI high')
            n[f'{k}_p'] = E(fmt_p(c['p']), '', s, 'Wilcoxon signed-rank p over persons')
            n[f'{k}_p_holm'] = E(fmt_p(c['p_holm']), '', s, 'Holm over the contrasts of this subset')
        for meas, fo in d['full_order'].items():
            k = f'manip_{sub}_{MEASURE_SHORT[meas]}_full'
            n[f'{k}_n'] = E(fo['n_persons'], 'd', s, f'persons with all three speeds ({meas})')
            n[f'{k}_order'] = E(fo['n_full_order'], 'd', s, f'persons with all three medians in order ({meas})')
        n[f'manip_{sub}_both_full_n'] = E(d['full_order_both']['n_persons'], 'd', s, 'persons with all speeds')
        n[f'manip_{sub}_both_full_order'] = E(d['full_order_both']['n_full_order'], 'd', s,
                                              'persons in order on both measures')
        for c in d.get('adjusted', []):
            k = f"adj_{sub}_{c['contrast']}"
            n[f'{k}_n'] = E(c['n_persons'], 'd', s, f"fast vs natural, {c['adjustment']}: persons")
            n[f'{k}_n_expected'] = E(c['n_expected'], 'd', s, 'persons in the instructed direction')
            n[f'{k}_ratio'] = E(c['median_ratio'], '.2f', s, 'median per-person ratio (instructed direction > 1)')
            n[f'{k}_ratio_lo'] = E(c['median_ratio_lo'], '.2f', s, 'person-bootstrap CI low')
            n[f'{k}_ratio_hi'] = E(c['median_ratio_hi'], '.2f', s, 'person-bootstrap CI high')
            n[f'{k}_p'] = E(fmt_p(c['p']), '', s, 'Wilcoxon signed-rank p over persons (nominal)')
            n[f'{k}_p_holm'] = E(fmt_p(c['p_holm']), '', s, 'Holm over the position-adjusted contrasts of this subset')
    n['manip_n_contrasts'] = E(len(mp['subsets']['standard']['contrasts']), 'd', s, 'contrasts per Holm family')
    n['manip_n_adjusted'] = E(len(mp['subsets']['standard'].get('adjusted', [])), 'd', s,
                              'position-adjusted contrasts per Holm family')

    fo = L.read_json('sec_speed_followers')
    s = src_tab('sec_speed_followers')
    n['fol_rho'] = E(fo['spearman_rho'], '+.2f', s, 'Spearman rho: natural/fast duration ratio vs the per-person '
                     'two-class fast-vs-natural accuracy of the headline model (descriptive, partly circular)')
    n['fol_p'] = E(fmt_p(fo['spearman_p']), '', s, 'Spearman p (one exploratory test)')
    n['fol_n'] = E(fo['n_persons'], 'd', s, 'persons with fast and natural standard trials')
    n['fol_n_followed'] = E(fo['n_followed'], 'd', s, 'persons whose fast median duration is below the natural')
    n['fol_n_not'] = E(fo['n_not_followed'], 'd', s, 'persons whose fast median duration is not below the natural')
    n['fol_bacc_followed'] = E(fo['bacc_followed'], '.1%', s, 'mean two-class fast-vs-natural balanced accuracy, '
                               'followed')
    n['fol_bacc_not'] = E(fo['bacc_not_followed'], '.1%', s, 'mean two-class fast-vs-natural balanced accuracy, other')
    D = pd.DataFrame(fo['per_person'])
    for grp, mask in (('followed', D['log_ratio'] > 0), ('not', D['log_ratio'] <= 0)):
        if 0 < mask.sum() < MIN_PERSONS_PER_CLASS:          # a small group: its persons individually
            for i, r in enumerate(D[mask].sort_values('person_id').itertuples(), start=1):
                n[f'fol_{grp}{i}_person'] = E(r.person_id, '', s, f'{grp} group: person {i}')
                n[f'fol_{grp}{i}_bacc'] = E(r.bacc_fast_natural, '.1%', s, f'{grp} group: person {i}, balanced acc.')

    uv = L.read_json('sec_speed_univariate')
    s = src_tab('sec_speed_univariate')
    for win, rows in uv['windows'].items():
        for row in rows:
            c = row['contrast'].split(' ')[0]
            k = f"uni_{win}_{row['feature']}_{c}"
            n[f'{k}_d'] = E(row['median_std_diff'], '+.2f', s, f"{row['feature']} ({row['contrast']}, {win} windows): "
                            'median per-person difference in SDs')
            n[f'{k}_npos'] = E(row['n_positive'], 'd', s, 'persons with a positive difference')
            n[f'{k}_n'] = E(row['n_persons'], 'd', s, 'persons with both speeds')
            n[f'{k}_p_holm'] = E(fmt_p(row['p_holm']), '', s, 'Wilcoxon p, Holm over every feature x contrast')
        n[f'uni_{win}_n_tests'] = E(len(rows), 'd', s, f'tests in the univariate Holm family ({win} windows)')
        U = pd.DataFrame(rows)
        for c in ('fast', 'slow'):
            sub = U[U['contrast'] == f'{c} - natural']
            n[f'uni_{win}_{c}_n_sig'] = E(int(sum(q < ALPHA for q in sub['p_holm'])), 'd', s,
                                          f'{c} - natural ({win} windows): features with Holm p < {ALPHA}')
            top = sub.assign(a=lambda x: x['median_std_diff'].abs()).sort_values('a', ascending=False,
                                                                                  kind='mergesort').head(3)
            for i, row in enumerate(top.itertuples(), start=1):
                kk = f'uni_{win}_{c}_top{i}'
                n[f'{kk}_feature'] = E(FEATURE_TEXT[row.feature], '', s, f'{c} - natural ({win}): largest effect {i}')
                n[f'{kk}_d'] = E(row.median_std_diff, '+.2f', s, f'{c} - natural ({win}): effect {i}, median SDs')
                n[f'{kk}_npos'] = E(row.n_positive, 'd', s, f'{c} - natural ({win}): effect {i}, persons positive')
                n[f'{kk}_n'] = E(row.n_persons, 'd', s, f'{c} - natural ({win}): effect {i}, persons')
                n[f'{kk}_p_holm'] = E(fmt_p(row.p_holm), '', s, f'{c} - natural ({win}): effect {i}, Holm p')
    co = L.read_json('sec_speed_coef')
    s = src_tab('sec_speed_coef')
    cdf = pd.DataFrame(co['rows'])
    for c in ('fast', 'slow'):
        sub = cdf[cdf['contrast'] == f'{c} vs natural'].assign(a=lambda x: x['mean'].abs()).sort_values(
            'a', ascending=False, kind='mergesort')
        for i, row in enumerate(sub.head(3).itertuples(), start=1):
            n[f'coef_{c}_top{i}_feature'] = E(FEATURE_TEXT[row.feature], '', s, f'{c} vs natural: feature rank {i}')
            n[f'coef_{c}_top{i}'] = E(row.mean, '+.2f', s, f'{c} vs natural: mean log-odds per SD over the folds')


def _numbers_design(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    dz = L.read_json('sec_design')
    s = src_tab('sec_design')
    for sp, v in dz['share_sideways_standard'].items():
        n[f'des_side_{sp}'] = E(v, '.1%', s, f'standard-variant {sp} trials held sideways (top left or top right)')
    for sp, row in dz['speed_by_grip_all'].items():
        n[f'des_{sp}_n'] = E(int(sum(row.values())), 'd', s, f'{sp} trials (all variants)')
        n[f'des_{sp}_upright'] = E(int(row.get('upright', 0)), 'd', s, f'{sp} trials held upright (all variants)')
    pf = dz['persons_fast_more_sideways']
    n['des_fast_more_side'] = E(pf['n_more'], 'd', s, 'persons whose fast trials are more often sideways than natural')
    n['des_fast_more_side_n'] = E(pf['n'], 'd', s, 'persons with fast and natural standard trials')
    for sp, v in dz['position_median'].items():
        n[f'des_pos_{sp}'] = E(v, '.2f', s, f'median relative position of the {sp} trials in the session (0..1)')
    for a in ('slow', 'fast'):
        d = dz[f'persons_{a}_before_natural']
        n[f'des_{a}_before'] = E(d['n_before'], 'd', s, f'persons whose {a} trials come before the natural (medians)')
        n[f'des_{a}_before_n'] = E(d['n'], 'd', s, f'persons with {a} and natural trials')
    pr = dz['practice_natural']
    n['des_practice_rho'] = E(pr['median_rho'], '+.2f', s, 'median within-person Spearman rho of the natural log '
                              'duration with the position (standard variant)')
    n['des_practice_neg'] = E(pr['n_negative'], 'd', s, 'persons whose natural reaches shorten over the session')
    n['des_practice_n'] = E(pr['n_persons'], 'd', s, 'persons with enough natural standard trials')
    n['des_practice_p'] = E(fmt_p(pr['wilcoxon_p']), '', s, 'Wilcoxon p of the per-person rho (one test)')


def _numbers_strata(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    st = L.read_json('sec_speed_strata')
    s = src_tab('sec_speed_strata')
    for name, by in st['models'].items():
        for stratum, d in by.items():
            k = f'str_{short(name)}_{stratum}'
            n[f'{k}_n'] = E(d['n'], 'd', s, f'{name}, {stratum} trials of the included speeds')
            n[f'{k}_bacc'] = E(d['balanced_accuracy'], '.1%', s, f"{name}, {stratum}: balanced accuracy over "
                               f"{', '.join(d['included'])}")
            n[f'{k}_f1'] = E(d['macro_f1'], '.3f', s, f'{name}, {stratum}: macro-F1 over the included speeds')
            n[f'{k}_acc'] = E(d['accuracy'], '.1%', s, f'{name}, {stratum}: plain accuracy over the included speeds')
            n[f'{k}_included'] = E(', '.join(d['included']), '', s, f'{name}, {stratum}: speeds included')
            n[f'{k}_excluded'] = E(', '.join(d['excluded']) or 'none', '', s, f'{name}, {stratum}: speeds left out '
                                   f"(fewer than {st['min_persons']} persons)")
            for c, v in d['pred_share'].items():
                n[f'{k}_pred_{c}'] = E(v, '.1%', s, f'{name}, {stratum}: share of trials predicted {c}')
                n[f'{k}_true_{c}'] = E(d['true_share'][c], '.1%', s, f'{name}, {stratum}: share of trials instructed {c}')
            for c in d['included']:
                n[f'{k}_rec_{c}'] = E(d[f'recall_{c}'], '.1%', s, f'{name}, {stratum}: recall of {c}')
    for r in st['pairs']:
        k = f"str_{pair_key(r['a'], r['b'])}_{r['stratum']}"
        n[f'{k}_n'] = E(r['n_trials'], 'd', s, f"{r['a']} vs {r['b']} within {r['stratum']} trials: trials")
        for key, fm in (('d_bacc', '+.1%'), ('d_f1', '+.3f')):
            n[f'{k}_{key}'] = E(r[key], fm, s, f"{r['a']} - {r['b']} within {r['stratum']}: {key}")
            n[f'{k}_{key}_lo'] = E(r[f'{key}_ci'][0], fm, s, 'CI low (person bootstrap)')
            n[f'{k}_{key}_hi'] = E(r[f'{key}_ci'][1], fm, s, 'CI high (person bootstrap)')
            n[f'{k}_{key}_p_holm'] = E(fmt_p(r[f'{key}_p_holm']), '', s, 'Holm over the stratum comparisons')
        n[f'{k}_d_acc'] = E(r['d_acc'], '+.1%', s, f"{r['a']} - {r['b']} within {r['stratum']}: plain accuracy")
    n['str_min_persons'] = E(st['min_persons'], 'd', s, 'a speed enters a grip stratum only with this many persons')


def _numbers_grip(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    ge = L.read_json('sec_grip_errors')
    s = src_tab('sec_grip_errors')
    for name, d in ge['models'].items():
        p = short(name)
        for key in ('n_errors', 'n_flagged_mismatch', 'n_sideways_swaps', 'n_app_agrees_with_prediction',
                    'n_persons_with_errors', 'n_up_matches_predicted'):
            n[f'{p}_{key}'] = E(d[key], 'd', s, f'{name}: {key.replace("_", " ")}')
        eu = d['error_persons_upright']
        n[f'{p}_err_upright_n'] = E(eu['n'], 'd', s, f'{name}: upright trials of the persons with errors')
        n[f'{p}_err_upright_correct'] = E(eu['n_correct'], 'd', s, f'{name}: of those, recognised as upright')
        ref = d['reference']
        for key, fm, what in (('n_known', 'd', 'trials whose start grip the app detected'),
                              ('n_unknown', 'd', "trials the app's start grip is unknown for"),
                              ('n_agree', 'd', "trials whose predicted grip equals the app's start grip"),
                              ('share_agree', '.1%', "share of those trials"),
                              ('bacc_vs_reference', '.1%', "balanced accuracy against the app's start grip"),
                              ('n_confirmed', 'd', 'trials flagged neither by the app nor by the gravity rule'),
                              ('n_errors_confirmed', 'd', 'errors against the instruction on those trials'),
                              ('bacc_confirmed', '.1%', 'balanced accuracy against the instruction on those trials'),
                              ('bacc_all', '.1%', 'balanced accuracy against the instruction on every trial')):
            n[f'{p}_ref_{key}'] = E(ref[key], fm, s, f'{name}: {what}')
    head = pd.DataFrame(ge['models'][SC.HEADLINE['grip']]['per_person'])
    i = 0
    for i, (pid, g) in enumerate(head.groupby('person_id', sort=True), start=1):
        n[f'grip_err{i}_person'] = E(str(pid), '', s, f'headline grip model: person {i} with errors')
        n[f'grip_err{i}_n'] = E(int(g['n'].sum()), 'd', s, f'headline grip model: errors of person {i}')
        n[f'grip_err{i}_pattern'] = E(', '.join(
            f"{GRIP_LABELS[r.instructed]} held as {GRIP_LABELS[r.predicted]} in {r.n} of {r.n_instructed}"
            for r in g.sort_values(['instructed', 'predicted']).itertuples()), '', s,
            f'headline grip model: person {i}, instructed -> predicted (of the trials instructed so)')
    n['grip_err_n_persons'] = E(i, 'd', s, 'persons with an error of the headline grip model')
    for g, v in ge['up_before_median'].items():
        for a in 'xyz':
            n[f'upmed_{g}_{a}'] = E(v[a], '+.2f', s, f'median up vector before the onset, grip {g}, device {a}')
    n['grip_flagged_trials'] = E(ge['flagged_trials'], 'd', s, 'primary trials flagged grip_mismatch')
    n['grip_flagged_persons'] = E(ge['flagged_persons'], 'd', s, 'persons with a flagged trial')
    n['grip_app_unknown'] = E(ge['app_unknown_trials'], 'd', s, "trials whose start grip the app reports as unknown")
    for src_, cnt in ge['gravity_sign_source'].items():
        n[f'grip_sign_{src_}'] = E(int(cnt), 'd', s, f'trials whose gravity sign came from {src_}')
    n['grip_sign_inferred'] = E(int(ge['gravity_sign_source'].get('inferred', 0)), 'd', s,
                                'trials whose gravity sign was inferred from the instructed orientation')


def _numbers_dominance(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    dp = L.read_json('sec_dom_persons')
    s = src_tab('sec_dom_persons')
    n['dom_auc'] = E(dp['pooled_auc'], '.3f', s, 'dominant-hand model: pooled AUC (leave-one-person-out)')
    n['dom_auc_lo'] = E(dp['pooled_auc_ci'][0], '.3f', s, 'pooled AUC, person-bootstrap CI low')
    n['dom_auc_hi'] = E(dp['pooled_auc_ci'][1], '.3f', s, 'pooled AUC, person-bootstrap CI high')
    n['dom_acc'] = E(dp['accuracy'], '.1%', s, 'dominant-hand model: accuracy')
    n['dom_acc_lo'] = E(dp['accuracy_ci'][0], '.1%', s, 'accuracy CI low')
    n['dom_acc_hi'] = E(dp['accuracy_ci'][1], '.1%', s, 'accuracy CI high')
    n['dom_ll'] = E(dp['log_loss'], '.3f', s, 'dominant-hand model: log loss')
    n['dom_ll_const'] = E(math.log(2.0), '.3f', s, 'log loss of a constant P = 0.5 (ln 2), the uninformative reference')
    n['dom_within_auc'] = E(dp['within_auc_mean'], '.3f', s, 'mean within-person AUC')
    n['dom_within_auc_lo'] = E(dp['within_auc_ci'][0], '.3f', s, 'mean within-person AUC, CI low')
    n['dom_within_auc_hi'] = E(dp['within_auc_ci'][1], '.3f', s, 'mean within-person AUC, CI high')
    n['dom_n_auc_above_half'] = E(dp['n_persons_auc_above_half'], 'd', s, 'persons with a within-person AUC > 0.5')
    n['dom_n_persons'] = E(dp['n_persons'], 'd', s, 'persons in the dominance analysis')
    cmode = max(dp['chosen_C'].items(), key=lambda kv: kv[1])
    n['dom_c_mode'] = E(float(cmode[0]), 'g', s, 'C chosen for most trials (nested)')
    n['dom_c_mode_trials'] = E(int(cmode[1]), 'd', s, 'trials whose fold chose that C')
    n['dom_n_trials'] = E(int(dp['n_trials']), 'd', s, 'trials of the dominance model')
    n['dom_n_folds'] = E(int(dp['n_folds']), 'd', s, 'outer folds of the dominance model')
    n['dom_n_folds_grid_min'] = E(int(dp['n_folds_at_grid_min']), 'd', s, 'outer folds choosing the smallest C')
    n['dom_p_min'] = E(dp['p_min'], '.2f', s, 'smallest predicted P(dominant)')
    n['dom_p_max'] = E(dp['p_max'], '.2f', s, 'largest predicted P(dominant)')
    n['dom_c_min'] = E(float(min(M.LR_C_GRID)), 'g', 'stage2/lanes/sec_models.py', 'the smallest C of the grid')
    pp = pd.DataFrame(dp['per_person'])
    lh = pp[pp['handedness'] == 'left'].sort_values('person_id')
    for i, row in enumerate(lh.itertuples(), start=1):
        n[f'dom_lh{i}_person'] = E(row.person_id, '', s, f'left-handed person {i}')
        n[f'dom_lh{i}_auc'] = E(row.auc, '.3f', s, f'left-handed person {i}: within-person AUC')
    n['dom_n_left'] = E(len(lh), 'd', s, 'left-handed persons')
    n['dom_n_right'] = E(int((pp['handedness'] == 'right').sum()), 'd', s, 'right-handed persons')
    rh = pp[pp['handedness'] == 'right']
    n['dom_rh_auc_median'] = E(float(rh['auc'].median()), '.3f', s, 'median within-person AUC of the right-handed')
    n['dom_rh_auc_min'] = E(float(rh['auc'].min()), '.3f', s, 'lowest within-person AUC of the right-handed')
    n['dom_rh_auc_max'] = E(float(rh['auc'].max()), '.3f', s, 'highest within-person AUC of the right-handed')
    rg = L.load_result(SC.DOMINANCE_GBM)
    s2 = src_res(SC.DOMINANCE_GBM)
    n['domgbm_auc'] = E(rg.metrics['roc_auc'], '.3f', s2, 'exploratory LightGBM dominant-hand model: pooled AUC')
    n['domgbm_auc_lo'] = E(rg.ci['roc_auc'][0], '.3f', s2, 'pooled AUC, person-bootstrap CI low')
    n['domgbm_auc_hi'] = E(rg.ci['roc_auc'][1], '.3f', s2, 'pooled AUC, person-bootstrap CI high')
    n['domgbm_n_configs'] = E(int(rg.info['n_configs_tried']), 'd', s2, 'model classes tried for the dominant hand')
    nl = L.read_json('sec_dom_null_test')
    s = src_tab('sec_dom_null_test')
    n['dom_n_perm'] = E(nl['n_perm'], 'd', s, 'label permutations (within person)')
    for key in ('pooled', 'within'):
        n[f'dom_null_{key}_mean'] = E(nl[f'{key}_null_mean'], '.3f', s, f'{key} AUC under permuted labels: mean')
        n[f'dom_null_{key}_q95'] = E(nl[f'{key}_null_q95'], '.3f', s, f'{key} AUC under permuted labels: 95th pct')
        n[f'dom_p_{key}'] = E(fmt_p(nl[f'{key}_p']), '', s, f'{key} AUC: one-sided permutation p')
    pdd = L.read_json('sec_dom_paired')
    s = src_tab('sec_dom_paired')
    for variant, pre in (('unmatched', 'pd'), ('matched', 'pdm')):
        rows = pdd['variants'][variant]['rows']
        for row in rows:
            f = row['feature']
            tr: Callable[[float], float] = ratio_pct if row['ratio'] else float
            fm = '+.1%' if row['ratio'] else '+.3f'
            k = f'{pre}_{f}'
            n[f'{k}_med'] = E(tr(row['median_diff']), fm, s, f'{variant}: median per-person dominant - non-dominant '
                              + ('(percent change)' if row['ratio'] else '(difference)'))
            n[f'{k}_lo'] = E(tr(row['median_diff_lo']), fm, s, 'person-bootstrap CI low')
            n[f'{k}_hi'] = E(tr(row['median_diff_hi']), fm, s, 'person-bootstrap CI high')
            n[f'{k}_n'] = E(row['n_persons'], 'd', s, 'persons')
            n[f'{k}_npos'] = E(row['n_dominant_higher'], 'd', s, 'persons with a higher value for the dominant hand')
            n[f'{k}_p'] = E(fmt_p(row['p']), '', s, 'Wilcoxon signed-rank p over persons')
            n[f'{k}_p_holm'] = E(fmt_p(row['p_holm']), '', s, f'Holm over the {len(rows)} features')
        n[f'{pre}_n_sig'] = E(int(sum(r['p_holm'] < ALPHA for r in rows)), 'd', s,
                              f'{variant}: features with Holm p < {ALPHA}')
        if variant == 'unmatched':
            per = pd.DataFrame(pdd['variants'][variant]['per_person'])
            lhp = sorted(per.loc[per['handedness'] == 'left', 'person_id'].unique())
            for i, pid in enumerate(lhp, start=1):
                for row in rows:
                    f = row['feature']
                    v = float(per[(per['person_id'] == pid) & (per['feature'] == f)]['diff'].iloc[0])
                    n[f'pd_lh{i}_{f}'] = E(ratio_pct(v) if row['ratio'] else v, '+.1%' if row['ratio'] else '+.3f', s,
                                           f'left-handed person {i}: dominant - non-dominant {f}')
    n['pd_n_features'] = E(len(pdd['features']), 'd', s, 'features of the paired comparison (Holm family)')


def _numbers_handedness(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    cc = L.read_json('sec_concordance')
    s = src_tab('sec_concordance')
    n['conc_n_persons'] = E(cc['n_persons'], 'd', s, 'persons')
    n['conc_n_nat'] = E(cc['n_natural_use'], 'd', s, 'persons with a free-choice trial')
    n['conc_nat_eq_handed'] = E(cc['natural_use_eq_handedness'], 'd', s, 'free-choice hand = self-reported handedness')
    n['conc_nat_eq_phone'] = E(cc['natural_use_eq_phone_hand'], 'd', s, 'free-choice hand = usual phone hand')
    n['conc_n_phone_clear'] = E(cc['n_phone_hand_clear'], 'd', s, 'persons naming one usual phone hand')
    n['conc_n_left'] = E(cc['n_left_handed'], 'd', s, 'left-handed persons')
    n['conc_left_nat_right'] = E(sum(v == 'right' for v in cc['left_handed_natural_use'].values()), 'd', s,
                                 'left-handed persons whose free-choice trial used the right hand')
    n['conc_left_phone_right'] = E(sum(v == 'right' for v in cc['left_handed_phone_hand'].values()), 'd', s,
                                   'left-handed persons who usually hold the phone in the right hand')
    n['conc_nat_right'] = E(sum(r['n_persons'] for r in cc['table'] if r['natural_use_hand'] == 'right'), 'd', s,
                            'persons whose first free-choice trial used the right hand')
    n['conc_n_phone_right'] = E(cc['n_phone_hand_right'], 'd', s, 'persons naming the right hand as usual phone hand')
    n['conc_n_nat_right'] = E(cc['n_natural_use_right'], 'd', s, 'persons whose first free-choice trial was right')
    n['conc_rep_n'] = E(len(cc['repeats']), 'd', s, 'repeat-visit free-choice trials')
    n['conc_rep_left'] = E(sum(r['free_choice_hand'] == 'left' for r in cc['repeats']), 'd', s,
                           'repeat-visit free-choice trials with the left hand')
    bl = L.read_json('sec_balance')
    s = src_tab('sec_balance')
    dpp = pd.DataFrame(bl['dominant_per_person'])
    n['bal_min_dominant'] = E(int(dpp['n_dominant'].min()), 'd', s, 'fewest dominant-hand trials of a person')
    n['bal_min_non_dominant'] = E(int((dpp['n'] - dpp['n_dominant']).min()), 'd', s,
                                  'fewest non-dominant-hand trials of a person')
    n['bal_first_left'] = E(int(bl['first_block_hand'].get('left', 0)), 'd', s, 'sessions whose first block was left')
    n['bal_first_right'] = E(int(bl['first_block_hand'].get('right', 0)), 'd', s, 'sessions whose first block was right')
    n['bal_first_p'] = E(fmt_p(bl['first_block_binom_p']), '', s, 'exact two-sided binomial p of the first-block hand '
                         '(left vs right, H0: one half)')


#: the online_v1 parameters the app changes per instructed speed (the slow override), as cited in the text
SEG_PARAMS_CITED: tuple[str, ...] = ('end_gyro_dps', 'end_acc_ms2', 'end_sustain_ms', 'max_movement_ms')


def _numbers_reseg(n: dict[str, Any], E: Callable[..., dict[str, Any]]) -> None:
    from motionlab import conventions as CV
    rs = L.read_json('sec_reseg')
    s = src_tab('sec_reseg')
    n['reseg_n_trials'] = E(rs['n_trials'], 'd', s, 'primary trials re-segmented')
    n['reseg_n_reproduced'] = E(rs['n_reproduced'], 'd', s, "trials whose stored bounds the app's detector reproduces "
                                'exactly with the stored parameters')
    n['reseg_n_common_ok'] = E(rs['n_common_ok'], 'd', s, 'trials the common parameter set ends')
    for sp, v in rs['by_speed'].items():
        k = f'reseg_{sp}'
        n[f'{k}_n'] = E(v['n'], 'd', s, f'{sp} trials')
        n[f'{k}_n_params_differ'] = E(v['n_params_differ'], 'd', s, f'{sp} trials whose stored segmentation parameters '
                                      "differ from the app's defaults")
        n[f'{k}_n_same'] = E(v['n_same_bounds'], 'd', s, f'{sp} trials with the same bounds under the common set')
        n[f'{k}_share_same'] = E(v['n_same_bounds'] / v['n'], '.1%', s, f'share of the {sp} trials with the same bounds')
        n[f'{k}_dur'] = E(v['median_duration_s'], '.2f', s, f'{sp}: median approach duration, oracle window (s)')
        n[f'{k}_dur_common'] = E(v['median_common_duration_s'], '.2f', s,
                                 f'{sp}: median approach duration, label-independent window (s)')
        n[f'{k}_ratio'] = E(v['median_ratio'], '.2f', s, f'{sp}: median per-trial ratio label-independent / oracle')
        n[f'{k}_n_std'] = E(v['n_standard'], 'd', s, f'{sp}: standard-variant trials')
        n[f'{k}_n_std_same'] = E(v['n_standard_same_bounds'], 'd', s, f'{sp}: standard-variant trials, same bounds')
        n[f'{k}_share_std_same'] = E(v['n_standard_same_bounds'] / max(v['n_standard'], 1), '.1%', s,
                                     f'{sp}: share of the standard-variant trials with the same bounds')
    n['reseg_n_common_fail'] = E(rs['n_trials'] - rs['n_common_ok'], 'd', s, 'primary trials the common set never ends')
    chk = rs['restandardize_check']['primary']
    n['restd_n_checked'] = E(chk['n_checked'], 'd', s, 'trials re-standardised on the oracle bounds (validation)')
    n['restd_max_diff'] = E(max(chk['approach_max_abs_diff'], chk['pre_max_abs_diff']), '.1e', s,
                            'largest |re-standardised - cached| of the approach and pre-onset arrays (oracle bounds)')
    n['restd_mask_mismatch'] = E(chk['pre_mask_mismatches'], 'd', s, 'pre-onset mask samples that differ')
    nu = rs['natural_use']
    n['reseg_nat_n'] = E(nu['n'], 'd', s, 'free-choice trials re-segmented')
    n['reseg_nat_ok'] = E(nu['n_common_ok'], 'd', s, 'free-choice trials the common set ends')
    src = 'motionlab/conventions.py'
    for p in SEG_PARAMS_CITED:
        n[f'seg_default_{p}'] = E(float(CV.DEFAULT_SEGMENTATION_PARAMS[p]), 'g', src, f'app default online_v1 {p}')
        n[f'seg_slow_{p}'] = E(float(CV._SPEED_OVERRIDES['slow'].get(p, CV.DEFAULT_SEGMENTATION_PARAMS[p])), 'g', src,
                               f'online_v1 {p} of a slow trial (per-speed override)')
    n['seg_common_max_movement_ms'] = E(float(rs['common_params']['max_movement_ms']), 'g', s,
                                        'movement cap of the common parameter set (the longest of any condition)')
    # trials whose stored parameters differ in a key no per-condition override touches: another base parameter set,
    # re-segmented with the same common set as every other trial
    R = pd.read_csv(L.tab_dir / f'{SC.RESEG_TABLE}.csv')
    R = R[R['set'] == 'primary']
    ov_keys = {k for o in (*CV._SPEED_OVERRIDES.values(), *CV._VARIANT_OVERRIDES.values()) for k in o}
    other = R['params_differ'].fillna('').astype(str).map(lambda x: bool({k for k in x.split(',') if k} - ov_keys))
    s2 = src_tab(SC.RESEG_TABLE, 'csv')
    n['reseg_n_other_base'] = E(int(other.sum()), 'd', s2, 'trials recorded with another base parameter set (a key '
                                'that no per-condition override changes differs from the defaults)')
    n['reseg_other_base_persons'] = E(int(R.loc[other, 'person_id'].nunique()), 'd', s2, 'persons of those trials')
    sh = L.read_json('sec_reseg_shares')['by_speed']
    s = src_tab('sec_reseg_shares')
    for sp, v in sh.items():
        n[f'reseg_{sp}_path_share'] = E(v['path_share_median'], '.0%', s, f'{sp}: median share of the oracle angular '
                                        'path inside the label-independent window')
        n[f'reseg_{sp}_path_share_q25'] = E(v['path_share_q25'], '.0%', s, f'{sp}: that share, 25th percentile')
        n[f'reseg_{sp}_path_share_q75'] = E(v['path_share_q75'], '.0%', s, f'{sp}: that share, 75th percentile')
        n[f'reseg_{sp}_dur_share'] = E(v['duration_share_median'], '.0%', s, f'{sp}: median share of the oracle '
                                       'duration inside the label-independent window')
        n[f'reseg_{sp}_strict_ok'] = E(v['strict_ok_share'], '.1%', s, f"{sp}: share of trials the slow trials' "
                                       'stricter end criterion ends before the return (applied to every trial)')
        n[f'reseg_{sp}_strict_ok_std'] = E(v['strict_ok_share_standard'], '.1%', s,
                                           f'{sp}: that share on the standard-variant trials')
    mc = L.read_json('sec_mirror_check')
    s = src_tab('sec_mirror_check')
    n['mirror_n_trials'] = E(mc['n_trials'], 'd', s, 'trials whose even features were compared with their mirror image')
    n['mirror_n_identical'] = E(mc['n_identical_trials'], 'd', s, 'trials with every even feature bit-identical')
    n['mirror_max_abs_diff'] = E(mc['max_abs_diff_all'], 'g', s, 'largest |feature - feature of the mirror image|')


def build_numbers() -> dict[str, dict[str, Any]]:
    E = RX.numbers_entry
    n: dict[str, dict[str, Any]] = {}
    _numbers_data(n, E)
    _numbers_models(n, E)
    _numbers_pairs(n, E)
    _numbers_speed(n, E)
    _numbers_design(n, E)
    _numbers_strata(n, E)
    _numbers_reseg(n, E)
    _numbers_grip(n, E)
    _numbers_dominance(n, E)
    _numbers_handedness(n, E)
    return n


def numbers() -> None:
    L.write_numbers(build_numbers())


def read_numbers() -> dict[str, dict[str, Any]]:
    return json.loads(L.numbers_path.read_text(encoding='utf-8'))


def fmt(n: dict[str, dict[str, Any]], key: str) -> str:
    e = n.get(key)
    return '--' if e is None else RX.format_value(e['value'], e['format'])


# ───────────────────────────── tables ─────────────────────────────


def _ci(v: float, lo: float, hi: float, f: str) -> str:
    return f'{RX.format_value(v, f)} [{RX.format_value(lo, f)}, {RX.format_value(hi, f)}]'


MODEL_SHORT = {'lr': 'LR', 'gbm': 'LightGBM'}


def model_table(task: str) -> pd.DataFrame:
    """One row per model of ``task``: windows, inputs, trials, balanced accuracy and macro-F1 with CIs, log loss (the
    per-class numbers are in :func:`class_table`, so that the table fits the text width)."""
    rows = []
    for name, spec in SC.MULTICLASS.items():
        if spec['task'] != task:
            continue
        r = SC.load_multiclass(name)
        m, ci = r['metrics'], r['ci95']
        rows.append({'experiment': name, 'windows': SC.windows_label(name).split(' (')[0],
                     'model': f"{MODEL_SHORT[spec['model']]}, {SC.FEATURE_LABELS[spec['features']]} "
                              f"({len(r['features'])})", 'n': int(m['n']),
                     'bal. acc.': _ci(m['balanced_accuracy'], *ci['balanced_accuracy'], '.1%'),
                     'macro-F1': _ci(m['macro_f1'], *ci['macro_f1'], '.3f'), 'log loss': m['log_loss']})
    return pd.DataFrame(rows)


def class_table(task: str) -> pd.DataFrame:
    """Per model and class: recall, precision and F1 (and the trials of the class / predicted as it)."""
    rows = []
    for name, spec in SC.MULTICLASS.items():
        if spec['task'] != task:
            continue
        r = SC.load_multiclass(name)
        m = r['metrics']
        cm = np.asarray(r['confusion'])
        for i, c in enumerate(r['classes']):
            rows.append({'experiment': name, 'class': GRIP_LABELS.get(c, c), 'n': int(cm[i].sum()),
                         'predicted': int(cm[:, i].sum()), 'recall': m[f'recall_{c}'],
                         'precision': m[f'precision_{c}'], 'F1': m[f'f1_{c}']})
    return pd.DataFrame(rows)


def confusion_table(names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        r = SC.load_multiclass(name)
        cm = np.asarray(r['confusion'])
        for i, c in enumerate(r['classes']):
            row = {'experiment': name, 'instructed': GRIP_LABELS.get(c, c), 'n': int(cm[i].sum())}
            for j, q in enumerate(r['classes']):
                row[f'predicted {GRIP_LABELS.get(q, q)}'] = f'{int(cm[i, j])} ({cm[i, j] / max(cm[i].sum(), 1):.1%})'
            rows.append(row)
    return pd.DataFrame(rows)


def tables(only: list[str] | None = None) -> None:
    from stage2.pipeline import selected
    T: dict[str, Callable[[], None]] = {}

    def table(name: str):
        def deco(fn: Callable[[], None]) -> Callable[[], None]:
            T[name] = fn
            return fn
        return deco

    @table('sec_speed_models')
    def _speed() -> None:
        L.export_table(model_table('speed'), 'sec_speed_models', caption=(
            'Instructed speed from sensor data (fast, natural, slow): leave-one-person-out, hyperparameters tuned on '
            'inner folds of the training persons, person-bootstrap 95\\% CIs. Label-blind windows: the app\'s '
            'detector re-run with one parameter set for every trial (the sensor-only estimates); oracle windows: the '
            'app\'s stored segment, whose end criterion depended on the instructed speed (label-informed, '
            f"optimistic). Chance balanced accuracy is {RX.format_value(1 / len(SC.SPEED_CLASSES), '.1%')}."),
            column_formats={'log loss': '.3f'})
        L.export_table(class_table('speed'), 'sec_speed_classes', caption=(
            'Per-class recall, precision and F1 of the speed models (leave-one-person-out); n: trials of the class, '
            'predicted: trials predicted as it.'), column_formats={'recall': '.1%', 'precision': '.1%', 'F1': '.3f'})

    @table('sec_grip_models')
    def _grip() -> None:
        L.export_table(model_table('grip'), 'sec_grip_models', caption=(
            'Instructed grip from sensor data (upright, top left, top right): leave-one-person-out, nested tuning, '
            'person-bootstrap 95\\% CIs, scored against the instruction. Chance balanced accuracy is '
            f"{RX.format_value(1 / len(SC.GRIP_CLASSES), '.1%')}."), column_formats={'log loss': '.3f'})
        L.export_table(class_table('grip'), 'sec_grip_classes', caption=(
            'Per-class recall, precision and F1 of the grip models (against the instruction).'),
            column_formats={'recall': '.1%', 'precision': '.1%', 'F1': '.3f'})

    @table('sec_speed_confusion')
    def _scm() -> None:
        names = [n for n, s in SC.MULTICLASS.items()
                 if s['task'] == 'speed' and s['role'] in ('baseline', 'interpretable', 'rich')]
        L.export_table(confusion_table(names), 'sec_speed_confusion', caption=(
            'Confusion of the instructed speed: counts and row shares (rows = instructed speed); label-blind '
            '(*\\_common) and oracle windows.'))

    @table('sec_design')
    def _design() -> None:
        dz = L.read_json('sec_design')
        rows = []
        for sp in SC.SPEED_CLASSES:
            g = dz['speed_by_grip_all'].get(sp, {})
            rows.append({'instructed speed': sp, 'trials': int(sum(g.values())),
                         **{GRIP_LABELS[k]: int(g.get(k, 0)) for k in SC.GRIP_CLASSES},
                         'sideways share (standard)': dz['share_sideways_standard'].get(sp, math.nan),
                         'median position': dz['position_median'].get(sp, math.nan)})
        L.export_table(pd.DataFrame(rows), 'sec_design', caption=(
            'The protocol confounds the instructed speed with the grip and with the position in the session: trials '
            'per speed and instructed grip (all variants), the share of standard-variant trials held sideways (top '
            'left or top right), and the median relative position of the trials in the person\'s sequence (0 = '
            'first, 1 = last). Descriptive.'),
            column_formats={'sideways share (standard)': '.1%', 'median position': '.2f'})

    @table('sec_speed_strata')
    def _strata() -> None:
        st = L.read_json('sec_speed_strata')
        rows = []
        for name, by in st['models'].items():
            for stratum, d in by.items():
                rows.append({'experiment': name, 'grip stratum': stratum, 'speeds': ', '.join(d['included']),
                             'n': d['n'], 'bal. acc.': d['balanced_accuracy'], 'macro-F1': d['macro_f1'],
                             'accuracy': d['accuracy'],
                             'predicted fast / true fast': f"{d['pred_share'].get('fast', math.nan):.1%} / "
                                                           f"{d['true_share'].get('fast', math.nan):.1%}"})
        L.export_table(pd.DataFrame(rows), 'sec_speed_strata', caption=(
            'The speed models within grip strata, after prediction (instructed grip: upright; sideways = top left or '
            f"top right). A speed enters a stratum only with trials of at least {st['min_persons']} persons (slow "
            'trials are almost all upright). Predicted / true fast: the share of the stratum\'s trials predicted '
            'fast and instructed fast.'),
            column_formats={'bal. acc.': '.1%', 'macro-F1': '.3f', 'accuracy': '.1%'})
        prs = [{'A': r['a'], 'B': r['b'], 'stratum': r['stratum'], 'trials': r['n_trials'],
                'bal. acc. A - B': _ci(r['d_bacc'], *r['d_bacc_ci'], '+.1%'), 'Holm p': fmt_p(r['d_bacc_p_holm']),
                'macro-F1 A - B': _ci(r['d_f1'], *r['d_f1_ci'], '+.3f'), 'Holm p (F1)': fmt_p(r['d_f1_p_holm'])}
               for r in st['pairs']]
        L.export_table(pd.DataFrame(prs), 'sec_speed_strata_pairs', caption=(
            'LightGBM vs the logistic regression within grip strata (paired on the same trials; person-bootstrap '
            '95\\% CIs; Holm within each metric over the stratum comparisons).'))

    @table('sec_grip_confusion')
    def _gcm() -> None:
        L.export_table(confusion_table(list(SC.GRIP_MODELS)), 'sec_grip_confusion', caption=(
            'Confusion of the instructed grip: counts and row shares (rows = instructed grip).'))

    @table('sec_model_pairs')
    def _pairs() -> None:
        pr = L.read_json('sec_pairs')
        rows, rows2 = [], []
        for r in pr['pairs']:
            rows.append({'A': r['a'], 'B': r['b'], 'trials': r['n_trials'],
                         'bal. acc. A - B': _ci(r['d_bacc'], *r['d_bacc_ci'], '+.1%'),
                         'Holm p': fmt_p(r['d_bacc_p_holm']),
                         'macro-F1 A - B': _ci(r['d_f1'], *r['d_f1_ci'], '+.3f'),
                         'Holm p (F1)': fmt_p(r['d_f1_p_holm']),
                         'verdict': 'identical' if r['identical'] else ''})
            rows2.append({'A': r['a'], 'B': r['b'], 'log loss A - B': _ci(r['d_ll'], *r['d_ll_ci'], '+.3f'),
                          'Holm p (log loss)': fmt_p(r['d_ll_p_holm']),
                          'accuracy A - B': _ci(r['d_acc'], *r['d_acc_ci'], '+.1%'),
                          'Holm p (accuracy)': fmt_p(r['d_acc_p_holm']), 'changed': r['n_changed']})
        L.export_table(pd.DataFrame(rows), 'sec_model_pairs', caption=(
            'Paired comparisons on the trials both models scored (a model of a subset of trials on that subset): '
            'the pooled balanced-accuracy difference (the primary metric; exact sign-flip test on the per-person '
            'contributions) and the pooled macro-F1 difference (person-swap test), A - B, each with a '
            'person-bootstrap 95\\% CI, Holm-adjusted within each metric over the comparisons listed. Identical: no '
            'trial changes class and the log-loss difference is below the solver precision.'))
        L.export_table(pd.DataFrame(rows2), 'sec_model_pairs_ll', caption=(
            'The same comparisons on the mean per-person log-loss and accuracy differences (A - B; person-bootstrap '
            '95\\% CIs; exact sign-flip p over persons, Holm within each metric); changed: trials whose predicted '
            'class differs.'))

    @table('sec_speed_manipulation')
    def _manip() -> None:
        mp = L.read_json('sec_speed_manip')
        rows = []
        for sub, d in mp['subsets'].items():
            for c in d['contrasts']:
                rows.append({'trials': MANIP_SUBSET_TEXT[sub],
                             'measure': FEATURE_TEXT[c['measure']], 'contrast': f"{c['high']} vs {c['low']}",
                             'persons': c['n_persons'], 'as instructed': c['n_expected'],
                             'median ratio': _ci(c['median_ratio'], c['median_ratio_lo'], c['median_ratio_hi'], '.2f'),
                             'Wilcoxon p': fmt_p(c['p']), 'Holm p': fmt_p(c['p_holm'])})
            for c in d.get('adjusted', []):
                rows.append({'trials': MANIP_SUBSET_TEXT[sub] + ', ' + c['adjustment'],
                             'measure': FEATURE_TEXT[c['measure']], 'contrast': f"{c['high']} vs {c['low']}",
                             'persons': c['n_persons'], 'as instructed': c['n_expected'],
                             'median ratio': _ci(c['median_ratio'], c['median_ratio_lo'], c['median_ratio_hi'], '.2f'),
                             'Wilcoxon p': fmt_p(c['p']), 'Holm p': fmt_p(c['p_holm'])})
        L.export_table(pd.DataFrame(rows), 'sec_speed_manipulation', caption=(
            'Manipulation check of the speed instruction: per person the median of each measure per instructed '
            'speed; the median per-person ratio of the higher-expected to the lower-expected speed (1 = no '
            'difference), the persons whose medians follow the instruction, a Wilcoxon signed-rank test over '
            'persons, Holm-adjusted within each set of trials (the position-adjusted fast vs natural contrasts form '
            'their own family). Median ratio with its person-bootstrap 95\\% CI. Oracle windows: the app\'s stored '
            'segment, whose parameters depend on the instructed speed; label-blind windows: the app\'s detector '
            're-run with one parameter set for every trial. Position-adjusted: per person an OLS of the log measure '
            'on speed and the relative position in the session, or only the natural trials made before the '
            'person\'s last fast trial.'))

    @table('sec_segmentation')
    def _seg() -> None:
        rs = L.read_json('sec_reseg')
        sh = L.read_json('sec_reseg_shares')['by_speed']
        rows = []
        for sp in SC.SPEED_CLASSES:
            v = rs['by_speed'].get(sp)
            if v is None:
                continue
            rows.append({'speed': sp, 'trials': v['n'],
                         'same bounds': f"{v['n_same_bounds']}/{v['n']}",
                         'duration, oracle (s)': v['median_duration_s'],
                         'duration, label-blind (s)': v['median_common_duration_s'],
                         'median ratio': v['median_ratio'],
                         'strict end before return': sh.get(sp, {}).get('strict_ok_share', math.nan)})
        L.export_table(pd.DataFrame(rows), 'sec_segmentation', caption=(
            'The approach window depends on the instructed speed. The app resolved its online segmentation '
            'parameters per condition (a slow trial ends only after a stricter and longer stillness, with a longer '
            'movement cap; a few trials also differ in their base parameters). Re-running the app\'s own detector on every stored stream with ONE parameter set (the '
            'defaults, with the longest movement cap of any condition) leaves fast and natural trials almost '
            'unchanged and ends a slow movement at its first slow phase. Durations: medians over trials; ratio: '
            'median per-trial ratio of the label-blind to the oracle duration. Strict end before return: the share '
            'of trials the slow trials\' stricter end criterion, applied to every trial, ends before the arm starts '
            'back (why it is not the common set).'),
            column_formats={'duration, oracle (s)': '.2f', 'duration, label-blind (s)': '.2f',
                            'median ratio': '.2f', 'strict end before return': '.1%'})

    @table('sec_speed_univariate')
    def _uni() -> None:
        uv = L.read_json('sec_speed_univariate')
        rows = [{'windows': 'label-blind' if win == 'common' else 'oracle', 'feature': FEATURE_TEXT[r['feature']],
                 'contrast': r['contrast'],
                 'median difference (SD)': r['median_std_diff'], 'IQR': f"[{r['q25']:+.2f}, {r['q75']:+.2f}]",
                 'persons positive': f"{r['n_positive']}/{r['n_persons']}", 'Holm p': fmt_p(r['p_holm'])}
                for win, rs_ in uv['windows'].items() for r in rs_]
        L.export_table(pd.DataFrame(rows), 'sec_speed_univariate', caption=(
            'Which kinematic features carry the instructed speed (standard-variant trials, no model): per person the '
            'difference of the medians between two instructed speeds in units of the feature\'s SD over all trials; '
            'across persons the median, IQR, the persons with a positive difference and a Wilcoxon test, '
            'Holm-adjusted over every feature and contrast.'), column_formats={'median difference (SD)': '+.2f'})

    @table('sec_grip_errors')
    def _gerr() -> None:
        ge = L.read_json('sec_grip_errors')
        cols = ['trial_key', 'person_id', 'hand', 'instructed', 'predicted', 'app_detected', 'grip_mismatch',
                'up_before_x', 'up_before_y']
        E = pd.DataFrame(ge['models'][SC.HEADLINE['grip']]['errors'], columns=[*cols, 'up_before_z',
                                                                                'up_matches_predicted'])
        df = E[cols].rename(columns={'app_detected': 'app grip at start', 'grip_mismatch': 'flagged'})
        for c in ('instructed', 'predicted', 'app grip at start'):
            df[c] = df[c].map(lambda g: GRIP_LABELS.get(g, g))
        L.export_table(df, 'sec_grip_errors', caption=(
            'The trials the headline grip model misclassifies: the instructed and predicted grip, the grip the app '
            'itself detected at the start and the grip-mismatch flag (both used after prediction only), and the up '
            'vector before the onset in the device frame.'),
            column_formats={'up_before_x': '+.2f', 'up_before_y': '+.2f'})

    @table('sec_grip_reference')
    def _gref() -> None:
        ge = L.read_json('sec_grip_errors')
        rows = []
        for name, d in ge['models'].items():
            ref = d['reference']
            rows.append({'experiment': name,
                         'vs instruction, all trials': f"{RX.format_value(ref['bacc_all'], '.1%')} "
                                                       f"({d['n_errors']} errors / {ref['n_trials']})",
                         'vs instruction, confirmed trials': f"{RX.format_value(ref['bacc_confirmed'], '.1%')} "
                                                             f"({ref['n_errors_confirmed']} errors / "
                                                             f"{ref['n_confirmed']})",
                         "vs the app's start grip": f"{RX.format_value(ref['bacc_vs_reference'], '.1%')} "
                                                    f"({ref['n_agree']} agree / {ref['n_known']})"})
        L.export_table(pd.DataFrame(rows), 'sec_grip_reference', caption=(
            'The grip models scored against three references (balanced accuracy, counts): the instruction on every '
            'trial; the instruction on the trials neither the app nor the gravity rule flagged as held in another '
            'grip (confirmed); and the grip the app itself detected at the start, on the trials where it could tell. '
            'The app reads the same physical quantity, so this separates wrong labels from sensor errors; it is not '
            'an independent sensor.'))

    @table('sec_dominance_model')
    def _dom() -> None:
        dp, nl = L.read_json('sec_dom_persons'), L.read_json('sec_dom_null_test')
        rg = L.load_result(SC.DOMINANCE_GBM)
        df = pd.DataFrame([
            {'model': 'logistic regression (pre-specified)', 'persons': dp['n_persons'],
             'pooled AUC': _ci(dp['pooled_auc'], *dp['pooled_auc_ci'], '.3f'),
             'permutation p': fmt_p(nl['pooled_p']),
             'null AUC mean [q95]': f"{nl['pooled_null_mean']:.3f} [{nl['pooled_null_q95']:.3f}]",
             'within-person AUC': _ci(dp['within_auc_mean'], *dp['within_auc_ci'], '.3f'),
             'permutation p (within)': fmt_p(nl['within_p']),
             'accuracy': _ci(dp['accuracy'], *dp['accuracy_ci'], '.1%')},
            {'model': 'LightGBM (exploratory)', 'persons': int(rg.metrics.get('n_persons', dp['n_persons'])),
             'pooled AUC': _ci(rg.metrics['roc_auc'], *rg.ci['roc_auc'], '.3f'), 'permutation p': 'not run',
             'null AUC mean [q95]': '', 'within-person AUC': '', 'permutation p (within)': '',
             'accuracy': _ci(rg.metrics['accuracy'], *rg.ci['accuracy'], '.1%')}])
        L.export_table(df, 'sec_dominance_model', caption=(
            'Can a leave-one-person-out model tell a dominant-hand trial from mirror-invariant kinematics alone? '
            f"Nested models; person-bootstrap 95\\% CIs; label-permutation baseline of the pre-specified model "
            f"({nl['n_perm']} permutations: labels shuffled within each person, the whole nested pipeline re-run; "
            'one-sided p).'))

    @table('sec_dominance_persons')
    def _domp() -> None:
        dp = pd.DataFrame(L.read_json('sec_dom_persons')['per_person'])
        df = dp[['person_id', 'handedness', 'n', 'n_pos', 'auc', 'accuracy']].rename(
            columns={'n_pos': 'dominant trials', 'auc': 'within-person AUC'})
        L.export_table(df, 'sec_dominance_persons', caption=(
            'Dominant-hand model per person (leave-one-person-out): the AUC of its predictions within the person and '
            'its accuracy; the left-handed persons are shown individually (too few for a confidence interval).'),
            column_formats={'within-person AUC': '.3f', 'accuracy': '.1%'})

    @table('sec_dominance_paired')
    def _domd() -> None:
        pdd = L.read_json('sec_dom_paired')
        um = {r['feature']: r for r in pdd['variants']['unmatched']['rows']}
        mt = {r['feature']: r for r in pdd['variants']['matched']['rows']}
        rows = []
        for f in pdd['features']:
            u, m = um[f], mt[f]
            tr = ratio_pct if u['ratio'] else float
            fm = '+.1%' if u['ratio'] else '+.3f'
            rows.append({'feature': FEATURE_TEXT[f], 'scale': 'percent change' if u['ratio'] else 'difference',
                         'dominant - non-dominant': _ci(tr(u['median_diff']), tr(u['median_diff_lo']),
                                                        tr(u['median_diff_hi']), fm),
                         'dominant higher': f"{u['n_dominant_higher']}/{u['n_persons']}", 'Wilcoxon p': fmt_p(u['p']),
                         'Holm p': fmt_p(u['p_holm']),
                         'matched: difference': RX.format_value(tr(m['median_diff']), fm),
                         'matched: Holm p': fmt_p(m['p_holm'])})
        L.export_table(pd.DataFrame(rows), 'sec_dominance_paired', caption=(
            'Dominant vs non-dominant movements within each person: per person the difference of the medians '
            '(a percent change for the duration, the peak angular speed and the path lengths); across persons the '
            'median with a person-bootstrap 95\\% CI, a Wilcoxon signed-rank test, Holm over the features. Matched: '
            'the same difference inside cells of equal instructed speed, grip and variant.'))

    @table('sec_concordance')
    def _conc() -> None:
        cc = L.read_json('sec_concordance')
        df = pd.DataFrame(cc['table']).rename(columns={'handedness_self': 'handedness', 'phone_hand': 'usual phone hand',
                                                       'natural_use_hand': 'free-choice hand', 'n_persons': 'persons'})
        L.export_table(df, 'sec_concordance', caption=(
            'Self-reported handedness, usual phone hand (profile) and the hand of the first free-choice trial: '
            'counts of persons (descriptive only).'))

    @table('sec_features')
    def _feat() -> None:
        L.export_table(F.catalogue()[['set', 'feature', 'unit', 'source', 'definition']], 'sec_features', caption=(
            'The interpretable features of lane sec. Every feature of the even set is identical for a trial and its '
            'left/right mirror image (unit-tested); the grip set is device-frame and changes under the mirror.'))

    for name, fn in T.items():
        if selected(name, only):
            fn()


# ───────────────────────────── summary ─────────────────────────────


def verdict(p_text: str) -> str:
    """'significant' / 'not significant' from a Holm-adjusted p printed by :func:`fmt_p`."""
    if p_text.startswith('<'):
        return 'significant'
    try:
        return 'significant' if float(p_text) < ALPHA else 'not significant'
    except ValueError:
        return 'not tested'


def pair_verdict(n: dict[str, dict[str, Any]], a: str, b: str, metric: str = 'd_bacc') -> str:
    """'identical' (saved flag), else the Holm verdict of the pair on ``metric``."""
    k = pair_key(a, b)
    if fmt(n, f'{k}_identical') == 'identical':
        return 'identical'
    return verdict(fmt(n, f'{k}_{metric}_p_holm'))


def compare_text(n: dict[str, dict[str, Any]], a: str, b: str, metric: str = 'd_bacc') -> str:
    """'beats' / 'is worse than' / 'does not differ significantly from' / 'makes the same predictions as' (A vs B;
    the last one: no trial changes class and the log-loss difference is below the solver precision)."""
    vd = pair_verdict(n, a, b, metric)
    if vd == 'identical':
        return 'makes the same predictions as'
    if vd != 'significant':
        return 'does not differ significantly from'
    diff = (n.get(f'{pair_key(a, b)}_{metric}') or {}).get('value') or 0.0
    better = diff > 0 if metric != 'd_ll' else diff < 0
    return 'beats' if better else 'is worse than'


def ci_excludes(n: dict[str, dict[str, Any]], key: str, null: float) -> bool:
    """True when the saved CI of ``key`` (entries key_lo, key_hi) lies entirely on one side of ``null``."""
    lo, hi = (n.get(key + '_lo') or {}).get('value'), (n.get(key + '_hi') or {}).get('value')
    return lo is not None and hi is not None and (lo > null or hi < null)


def render_summary(n: dict[str, dict[str, Any]]) -> str:
    v = lambda k: fmt(n, k)                                    # noqa: E731
    val = lambda k: (n.get(k) or {}).get('value')              # noqa: E731
    ci = lambda k: f'{v(k)} [{v(k + "_lo")}, {v(k + "_hi")}]'  # noqa: E731
    H = SC.HEADLINE
    pk = pair_key
    hs = H['speed']                                            # the headline: label-blind windows
    ho = SC.ORACLE_TWIN[hs]                                    # the same model on the oracle windows
    gbc, gbo = 'sec_speed_gbm_common', 'sec_speed_gbm'
    duc, gripm, stdm = 'sec_speed_dur_common', 'sec_speed_grip', 'sec_speed_lr_std'
    angc, ango = 'sec_speed_ang_common', 'sec_speed_ang'
    sp, so, gc, go = short(hs), short(ho), short(gbc), short(gbo)
    gp = short(H['grip'])
    up_g, gbm_g = 'sec_grip_up', 'sec_grip_gbm'
    k_seg = pk(*SC.SEGMENTATION_PAIR)                          # label-blind - oracle, the headline model
    k_segg = pk(gbc, gbo)
    k_sega = pk(angc, ango)
    k_gl = pk(gbc, hs)                                         # model class, label-blind windows
    k_glo = pk(gbo, ho)                                        # model class, oracle windows
    k_dur = pk(hs, duc)
    k_grip = pk(hs, gripm)
    k_std = pk(stdm, hs)
    seg_drop = verdict(v(k_seg + '_d_bacc_p_holm')) == 'significant' and (val(k_seg + '_d_bacc') or 0) < 0
    fk, pkk = 'manip_standard_dur_fast_natural', 'manip_standard_peak_natural_fast'
    fk_c = 'manip_common_dur_fast_natural'
    adj = 'adj_standard_duration_log'
    # 'clear' only when the bootstrap CI of the median ratio excludes 1 AND the Holm-adjusted test agrees
    fast_dur_clear = ci_excludes(n, fk + '_ratio', 1.0) and verdict(v(fk + '_p_holm')) == 'significant'
    peak_ok = verdict(v(pkk + '_p_holm')) == 'significant'
    chance = v('chance_speed')
    lines: list[str] = []
    w = lines.append
    w('# Lane sec: secondary tasks (instructed speed, instructed grip, handedness)')
    w('')
    w('Generated by `python -m stage2.pipeline numbers --lanes sec` from the saved results; every number below is an '
      f"entry of `results/sec_numbers.json` (LaTeX macros `\\Sec...`). Persons are referred to by their study codes "
      f"only ({v('person_first')} to {v('person_last')}).")
    w('')
    w('## Goal')
    w('')
    w('The researcher asked for three analyses after the hand model: predict the instructed **speed** from the sensors '
      'and check that the persons followed the speed instruction; predict the instructed **grip** and explain the '
      'confusions physically; and **handedness**: what can be said about the dominant hand. The instructed conditions '
      'are TARGETS here; they never enter a model as inputs, and no model of this lane is a hand model '
      '(`results/finalists_sec.json` is an empty list).')
    w('')
    w('## Headline numbers (95% person-bootstrap CIs, leave-one-person-out, nested tuning)')
    w('')
    w(f"- **Speed from the sensors** (label-blind windows: the app's detector re-run with one parameter set for every "
      f"trial, every trial re-standardised on those bounds; pre-specified multinomial L2 logistic regression on "
      f"{v(sp + '_n_features')} mirror-invariant kinematic features, `{hs}`): balanced accuracy {ci(sp + '_bacc')}, "
      f"macro-F1 {ci(sp + '_f1')} (chance balanced accuracy {chance}). Label-informed reference, NOT a sensor-only "
      f"estimate (the app's stored windows, whose end criterion depended on the instructed speed; `{ho}`): "
      f"{ci(so + '_bacc')}; label-blind minus label-informed {ci(k_seg + '_d_bacc')} (Holm p "
      f"{v(k_seg + '_d_bacc_p_holm')}, {pair_verdict(n, *SC.SEGMENTATION_PAIR)}).")
    w(f"- Speed, LightGBM on the rich features, label-blind windows (`{gbc}`): balanced accuracy {ci(gc + '_bacc')}, "
      f"macro-F1 {ci(gc + '_f1')}; against the headline: balanced accuracy {ci(k_gl + '_d_bacc')} (Holm p "
      f"{v(k_gl + '_d_bacc_p_holm')}, {pair_verdict(n, gbc, hs)}), macro-F1 {ci(k_gl + '_d_f1')} (Holm p "
      f"{v(k_gl + '_d_f1_p_holm')}, {pair_verdict(n, gbc, hs, 'd_f1')}).")
    w(f"- Speed and the protocol: the grip alone (up vector before the onset, `{gripm}`) predicts the instructed speed "
      f"with balanced accuracy {ci(short(gripm) + '_bacc')}, because slow trials were almost all upright and fast "
      f"trials more often sideways (problem 2).")
    w(f"- Speed manipulation (standard-variant trials, oracle windows): peak angular speed higher for fast than natural "
      f"in {v(pkk + '_n_expected')} of {v(pkk + '_n')} persons (Holm p {v(pkk + '_p_holm')}); fast reaches shorter than "
      f"natural in {v(fk + '_n_expected')} of {v(fk + '_n')} (median natural/fast duration ratio "
      f"{ci(fk + '_ratio')}, Holm p {v(fk + '_p_holm')}).")
    w(f"- Grip: balanced accuracy {ci(gp + '_bacc')} against the instruction (chance {v('chance_grip')}); the "
      f"{v(gp + '_n_errors')} errors come from {v('grip_err_n_persons')} persons, and by their up vector "
      f"{v(gp + '_n_up_matches_predicted')} of them sit in the cluster of the predicted grip.")
    sig_dom = verdict(v('dom_p_pooled')) == 'significant'
    w(f"- Dominant hand from mirror-invariant kinematics (pre-specified LR): pooled AUC {ci('dom_auc')}, "
      f"within-person label-permutation null mean {v('dom_null_pooled_mean')} (p {v('dom_p_pooled')}): "
      + ('a dominant-hand signal is detected.' if sig_dom else
         f"no dominant-hand signal detected; the CI allows a pooled AUC up to {v('dom_auc_hi')}."))
    w('')
    w('## Data and evaluation')
    w('')
    w(f"- Primary set: {v('n_trials')} trials of {v('n_persons')} persons. Instructed speed: fast {v('n_speed_fast')}, "
      f"natural {v('n_speed_natural')}, slow {v('n_speed_slow')}. Instructed grip: upright {v('n_grip_upright')}, top "
      f"left {v('n_grip_top_left')}, top right {v('n_grip_top_right')}. Standard-variant trials: {v('n_standard')}; "
      f"fast or slow trials of another variant: {v('n_fastslow_nonstandard')}"
      + (' (the movement variants such as pause or jerky are natural-speed trials only).'
         if val('n_fastslow_nonstandard') == 0 else '.'))
    w('- Multi-class targets (speed, grip): the SAME leave-one-person-out splits (`cv.loso_splits`), the D2 cap '
      '(`cv.cap_rows`), nested tuning on `cv.inner_splits` of the training persons (class-balanced log loss) and '
      f"person-cluster bootstrap CIs (`cv.person_bootstrap`, {v('n_boot')} resamples) in the lane's own loop "
      '(`sec_models.loso_multiclass`); every fold is re-predicted on a random half of its test rows (no change '
      'allowed). Class-balanced weights. The binary dominant-hand target goes through `cv.run_experiment`.')
    w(f"- Windows. The app resolved its segmentation parameters per instructed condition, so the stored (oracle) "
      f"window of a slow, pause or jerky trial is chosen by its label (problem 1). `sec_reseg.py` re-runs the app's "
      f"own detector on every stored stream with one parameter set and re-standardises every trial on those "
      f"label-blind bounds through the cache's own code path (`standardize.stream_from_record` -> `canonical` -> "
      f"`time_normalised`); on the oracle bounds the same path reproduces the cached arrays of all "
      f"{v('restd_n_checked')} trials (largest difference {v('restd_max_diff')}), so every feature of the lane can be "
      f"computed on label-blind windows. The common set ends all {v('reseg_n_common_ok')} of {v('reseg_n_trials')} "
      f"trials and {v('reseg_nat_ok')} of {v('reseg_nat_n')} free-choice trials.")
    w(f"- Features (`sec_features.py`, table `sec_features`): {v('n_even_features')} mirror-invariant kinematic "
      f"features (duration, angular speed, smoothness, path lengths), {v('n_grip_features')} device-frame orientation "
      'features (up vector before the onset, at the onset and at the end; mean angular velocity), and a rich set of '
      'summary statistics of every channel for the gradient-boosting models. Every feature is a function of one '
      'trial; scalers and models are fitted inside the folds.')
    w('- Model comparisons are paired on the trials both models scored: the pooled balanced-accuracy difference '
      '(exact sign-flip test on the per-person contributions, `sec_stats.paired_balanced`), the pooled macro-F1 '
      'difference (person-swap test, `sec_stats.paired_swap`), and the mean per-person log-loss and accuracy '
      f"differences (`sec_models.paired_multiclass`); Holm within each metric over the {v('n_model_pairs')} comparisons "
      f"(tables `sec_model_pairs`, `sec_model_pairs_ll`). A pair in which no trial changes class and the log-loss "
      f"difference is below {v('ll_precision')} is reported as identical.")
    w('')
    w('## Experiments')
    w('')
    for name, s in SC.MULTICLASS.items():
        p = short(name)
        w(f"- `{name}` ({s['task']}, {SC.MODEL_LABELS[s['model']]}, {SC.FEATURE_LABELS[s['features']]}, "
          f"{SC.windows_label(name)} windows"
          f"{'' if s['subset'] == 'all' else ', ' + SC.subset_label(s['subset']) + ' only'}"
          f"{'; the headline model of its task' if name in H.values() else ''}): balanced accuracy "
          f"{ci(p + '_bacc')}, macro-F1 {ci(p + '_f1')}; `results/{name}.json`.")
    w(f"- `{SC.DOMINANCE}` (dominant hand, nested L2 logistic regression on the mirror-invariant kinematics, "
      f"pre-specified): pooled AUC {ci('dom_auc')}; `results/{SC.DOMINANCE}.json`, registry `results/registry_sec.csv`.")
    w(f"- `{SC.DOMINANCE_GBM}` (the same with nested LightGBM; exploratory, {v('domgbm_n_configs')} model classes "
      f"tried): pooled AUC {ci('domgbm_auc')}; `results/{SC.DOMINANCE_GBM}.json`.")
    w(f"- `sec_dom_null`: label-permutation baseline, {v('dom_n_perm')} permutations within person through the whole "
      'nested pipeline of the pre-specified model (`tables/sec/sec_dom_null.json`); its test against the current '
      'result is recomputed at every compute step (`sec_dom_null_test.json`).')
    w('- `sec_reseg`: the re-segmentation and the label-blind arrays (`tables/sec/sec_reseg_bounds.csv`, '
      '`sec_reseg_arrays.npz`, summary `sec_reseg.json`). It reads the whole snapshot, so it is recomputed only when '
      'its code (this lane\'s module, the standardisation, the app\'s detector port and its parameter tables), its '
      'parameters or the data change; every experiment on label-blind windows records the hashes of those files, so '
      'a changed re-segmentation makes it stale.')
    w('- Derived from the saved results, no model fitted (`tables/sec/`): the design of the protocol (`sec_design`), '
      'the speed manipulation check with the position-adjusted contrasts (`sec_speed_manip`), univariate speed effects '
      'on both windows (`sec_speed_univariate`), the headline LR coefficients (`sec_speed_coef`), the speed models '
      'within grip strata (`sec_speed_strata`), the per-person fast/natural description (`sec_speed_followers`), '
      'paired model comparisons (`sec_pairs`), grip errors (`sec_grip_errors`), dominant vs non-dominant movements '
      '(`sec_dom_paired`), per-person dominance AUCs (`sec_dom_persons`), handedness concordance (`sec_concordance`), '
      'plan balance (`sec_balance`), the window shares (`sec_reseg_shares`) and the mirror check (`sec_mirror_check`).')
    w('')

    # ── speed
    w('## Instructed speed')
    w('')
    w(f"**Design of the protocol** (table `sec_design`). Standard-variant trials held sideways: fast "
      f"{v('des_side_fast')}, natural {v('des_side_natural')}, slow {v('des_side_slow')}; slow trials upright "
      f"{v('des_slow_upright')} of {v('des_slow_n')}; fast trials more often sideways than natural ones for "
      f"{v('des_fast_more_side')} of {v('des_fast_more_side_n')} persons. Median relative position in the session: slow "
      f"{v('des_pos_slow')}, fast {v('des_pos_fast')}, natural {v('des_pos_natural')}; slow before natural for "
      f"{v('des_slow_before')} of {v('des_slow_before_n')} persons, fast before natural for {v('des_fast_before')} of "
      f"{v('des_fast_before_n')}. Natural reaches shorten over the session (within-person Spearman rho of the log "
      f"duration with the position: median {v('des_practice_rho')}, negative for {v('des_practice_neg')} of "
      f"{v('des_practice_n')} persons, Wilcoxon p {v('des_practice_p')}).")
    w('')
    w(f"**Manipulation check** (per-person medians; Wilcoxon signed-rank over persons, Holm over the "
      f"{v('manip_n_contrasts')} contrasts of each set of trials; median per-person ratio with its person-bootstrap CI; "
      'table `sec_speed_manipulation`, figure `SEC_01_speed_manipulation`).')
    for sub, label in (('standard', 'oracle windows'), ('common', 'label-blind windows')):
        k1, k2 = f'manip_{sub}_dur_natural_slow', f'manip_{sub}_dur_fast_natural'
        k3, k4 = f'manip_{sub}_peak_slow_natural', f'manip_{sub}_peak_natural_fast'
        w(f"- Standard variant, {label}: slow longer than natural for {v(k1 + '_n_expected')} of {v(k1 + '_n')} persons "
          f"(ratio {ci(k1 + '_ratio')}, Holm p {v(k1 + '_p_holm')}); fast shorter than natural for "
          f"{v(k2 + '_n_expected')} of {v(k2 + '_n')} (ratio {ci(k2 + '_ratio')}, Holm p {v(k2 + '_p_holm')}); "
          f"natural faster at the peak than slow for {v(k3 + '_n_expected')} of {v(k3 + '_n')} (Holm p "
          f"{v(k3 + '_p_holm')}); fast faster than natural for {v(k4 + '_n_expected')} of {v(k4 + '_n')} (ratio "
          f"{ci(k4 + '_ratio')}, Holm p {v(k4 + '_p_holm')}).")
    w(f"- Every trial (variants included, oracle windows): fast shorter than natural for "
      f"{v('manip_all_dur_fast_natural_n_expected')} of {v('manip_all_dur_fast_natural_n')} persons (Holm p "
      f"{v('manip_all_dur_fast_natural_p_holm')}).")
    w(f"- Position-adjusted fast vs natural (sensitivity; own Holm family of {v('manip_n_adjusted')}; nominal p in "
      f"brackets): duration, OLS on speed and position: natural longer than fast for {v(adj + '_n_expected')} of "
      f"{v(adj + '_n')} persons, ratio {ci(adj + '_ratio')} (Holm p {v(adj + '_p_holm')} [{v(adj + '_p')}]); only the "
      f"natural trials made before the person's last fast trial: {v('adj_standard_duration_overlap_n_expected')} of "
      f"{v('adj_standard_duration_overlap_n')}, ratio {ci('adj_standard_duration_overlap_ratio')} (Holm p "
      f"{v('adj_standard_duration_overlap_p_holm')}); peak angular speed, OLS: fast faster for "
      f"{v('adj_standard_ang_speed_peak_n_expected')} of {v('adj_standard_ang_speed_peak_n')} (Holm p "
      f"{v('adj_standard_ang_speed_peak_p_holm')}).")
    w(f"- Reading: "
      + ('the persons followed the fast instruction at the peak' if peak_ok else
         'a faster peak for the fast trials is not established')
      + f" (higher peak angular speed for {v(pkk + '_n_expected')} of {v(pkk + '_n')} persons, "
      f"{verdict(v(pkk + '_p_holm'))}); whether they also "
      + ('finished sooner is clear: the duration ratio CI excludes 1 and the Holm-adjusted test agrees.'
         if fast_dur_clear else
         f"finished sooner is uncertain: the median duration ratio is {ci(fk + '_ratio')}, not significant unadjusted "
         f"(Holm p {v(fk + '_p_holm')}); the protocol ran the fast trials earlier in the session while natural "
         f"reaches shortened with practice, so the contrast mixes instruction and practice (position-adjusted: ratio "
         f"{ci(adj + '_ratio')}, {v(adj + '_n_expected')} of {v(adj + '_n')} persons as instructed, Holm p "
         f"{v(adj + '_p_holm')}).")
      + f" Slow reaches were longer than natural on the oracle windows for "
      f"{v('manip_standard_dur_natural_slow_n_expected')} of {v('manip_standard_dur_natural_slow_n')} persons but on "
      f"label-blind windows for {v('manip_common_dur_natural_slow_n_expected')} of "
      f"{v('manip_common_dur_natural_slow_n')}: the size of that contrast is largely the stricter end criterion the app "
      f"applied to slow trials (problem 1). At the peak, natural reaches were faster than slow ones for "
      f"{v('manip_common_peak_slow_natural_n_expected')} of {v('manip_common_peak_slow_natural_n')} persons on "
      f"label-blind windows (Holm p {v('manip_common_peak_slow_natural_p_holm')}, "
      f"{verdict(v('manip_common_peak_slow_natural_p_holm'))}).")
    w('')
    cls = list(SC.SPEED_CLASSES)
    cpairs = [(x, y) for i, x in enumerate(cls) for y in cls[i + 1:]]
    a, b = max(cpairs, key=lambda xy: val(f'{sp}_cmr_{xy[0]}_{xy[1]}') + val(f'{sp}_cmr_{xy[1]}_{xy[0]}'))
    bestf = max(cls, key=lambda c: val(f'{sp}_f1_{c}'))
    w('**Models** (tables `sec_speed_models`, `sec_speed_classes`, `sec_speed_confusion`, `sec_model_pairs`; figure '
      '`SEC_02_speed_confusion`).')
    w(f"- Headline (pre-specified, interpretable, label-blind windows): `{hs}`, balanced accuracy {ci(sp + '_bacc')}, "
      f"macro-F1 {ci(sp + '_f1')}. Per class (recall / precision / F1): "
      + '; '.join(f"{c} {v(f'{sp}_rec_{c}')} / {v(f'{sp}_prec_{c}')} / {v(f'{sp}_f1_{c}')}" for c in cls)
      + f". The best F1 is {bestf}'s; the largest mutual confusion is {a} <-> {b}: {v(f'{sp}_cmr_{a}_{b}')} of the "
      f"{a} trials predicted {b}, {v(f'{sp}_cmr_{b}_{a}')} of the {b} trials predicted {a}. Class-balanced weights "
      f"make the model call slow {v(sp + '_npred_slow')} times for {v(sp + '_n_slow')} slow trials.")
    w(f"- Duration only (label-blind): {ci(short(duc) + '_bacc')}; the kinematic model {compare_text(n, hs, duc)} it "
      f"(balanced accuracy {ci(k_dur + '_d_bacc')}, Holm p {v(k_dur + '_d_bacc_p_holm')}).")
    w(f"- LightGBM (label-blind, `{gbc}`): balanced accuracy {ci(gc + '_bacc')}, macro-F1 {ci(gc + '_f1')}; per class "
      + '; '.join(f"{c} {v(f'{gc}_rec_{c}')} / {v(f'{gc}_prec_{c}')} / {v(f'{gc}_f1_{c}')}" for c in cls)
      + f". Against the headline: balanced accuracy {ci(k_gl + '_d_bacc')} (Holm p {v(k_gl + '_d_bacc_p_holm')}, "
      f"{pair_verdict(n, gbc, hs)}), macro-F1 {ci(k_gl + '_d_f1')} (Holm p {v(k_gl + '_d_f1_p_holm')}, "
      f"{pair_verdict(n, gbc, hs, 'd_f1')}), log loss {ci(k_gl + '_d_ll')} (Holm p {v(k_gl + '_d_ll_p_holm')}, "
      f"{pair_verdict(n, gbc, hs, 'd_ll')}), accuracy {ci(k_gl + '_d_acc')} (Holm p {v(k_gl + '_d_acc_p_holm')}).")
    w(f"- Label-informed references (oracle windows; optimistic, not sensor-only): LR {ci(so + '_bacc')}, macro-F1 "
      f"{ci(so + '_f1')}; LightGBM {ci(go + '_bacc')}, macro-F1 {ci(go + '_f1')}. Label-blind minus oracle: LR "
      f"{ci(k_seg + '_d_bacc')} (Holm p {v(k_seg + '_d_bacc_p_holm')}), LightGBM {ci(k_segg + '_d_bacc')} (Holm p "
      f"{v(k_segg + '_d_bacc_p_holm')}), the {v(short(angc) + '_n_features')} stream features "
      f"{ci(k_sega + '_d_bacc')} (Holm p {v(k_sega + '_d_bacc_p_holm')}). On the oracle windows LightGBM vs LR: "
      f"balanced accuracy {ci(k_glo + '_d_bacc')} (Holm p {v(k_glo + '_d_bacc_p_holm')}), macro-F1 "
      f"{ci(k_glo + '_d_f1')} (Holm p {v(k_glo + '_d_f1_p_holm')}).")
    w(f"- The grip as a nuisance reference (`{gripm}`, the up vector before the onset only): balanced accuracy "
      f"{ci(short(gripm) + '_bacc')}, slow recall {v(short(gripm) + '_rec_slow')}; the headline "
      f"{compare_text(n, hs, gripm)} it ({ci(k_grip + '_d_bacc')}, Holm p {v(k_grip + '_d_bacc_p_holm')}). The "
      f"mirror-invariant features themselves carry the grip: they tell sideways from upright with balanced accuracy "
      f"{ci('side_even_bacc')} (`sec_side_even`, chance {v('chance_side')}).")
    us, ss = 'str_' + sp + '_upright', 'str_' + sp + '_sideways'
    gus, gss = 'str_' + gc + '_upright', 'str_' + gc + '_sideways'
    kgu = f'str_{k_gl}_upright'
    kgs = f'str_{k_gl}_sideways'
    w(f"- Within grip strata (after prediction; table `sec_speed_strata`; a speed enters a stratum only with trials of "
      f"at least {v('str_min_persons')} persons): upright trials ({v(us + '_included')}): headline balanced accuracy "
      f"{v(us + '_bacc')}, LightGBM {v(gus + '_bacc')}, difference {ci(kgu + '_d_bacc')} (Holm p "
      f"{v(kgu + '_d_bacc_p_holm')}); sideways trials ({v(ss + '_included')}; left out: {v(ss + '_excluded')}): "
      f"headline {v(ss + '_bacc')}, LightGBM {v(gss + '_bacc')}, difference {ci(kgs + '_d_bacc')} (Holm p "
      f"{v(kgs + '_d_bacc_p_holm')}). LightGBM predicts fast for {v(gus + '_pred_fast')} of the upright trials "
      f"(instructed fast {v(gus + '_true_fast')}) and {v(gss + '_pred_fast')} of the sideways ones "
      f"({v(gss + '_true_fast')}); the headline {v(us + '_pred_fast')} and {v(ss + '_pred_fast')}.")
    w(f"- Training on the standard-variant trials only (sensitivity, `{stdm}`): {ci(short(stdm) + '_bacc')} on those "
      f"trials; against the all-trials model on the same {v(k_std + '_n')} trials {ci(k_std + '_d_bacc')} (Holm p "
      f"{v(k_std + '_d_bacc_p_holm')}, {pair_verdict(n, stdm, hs)}).")
    fol_extra = ''
    for grp, label in (('followed', 'shorter-fast'), ('not', 'other')):
        ids = sorted(int(m.group(1)) for kk in n for m in [re.fullmatch(rf'fol_{grp}(\d+)_person', kk)] if m)
        if ids:
            fol_extra += (f" The {label} group has fewer than {v('min_persons_ci')} persons, so individually: "
                          + ', '.join(f"{v(f'fol_{grp}{i}_person')} {v(f'fol_{grp}{i}_bacc')}" for i in ids) + '.')
    w(f"- Description, not evidence (figure `SEC_08_speed_followers`; partly circular, since the duration it "
      f"stratifies by is an input of the model it scores): per person, the headline model's two-class fast-vs-natural "
      f"balanced accuracy (standard trials, the class decided between fast and natural only) against the natural/fast "
      f"duration ratio on label-blind windows: Spearman rho {v('fol_rho')} (p {v('fol_p')}, {v('fol_n')} persons); "
      f"mean {v('fol_bacc_followed')} for the {v('fol_n_followed')} persons whose fast reaches were shorter, "
      f"{v('fol_bacc_not')} for the other {v('fol_n_not')}.{fol_extra}")
    w(f"- Free-choice trials ({v(sp + '_nat_n')}, each scored by its person's fold model on its label-blind window; the "
      f"speed field is natural for {v(sp + '_nat_instructed_natural')}): predicted natural {v(sp + '_nat_natural')}, "
      f"fast {v(sp + '_nat_fast')}, slow {v(sp + '_nat_slow')}. This is natural-speed recall on free-choice trials "
      'only (a constant natural answer would score 100%), not a balanced measure.')
    w(f"- What carries speed on the label-blind windows (table `sec_speed_univariate`, figure `SEC_03_speed_features`; "
      f"model-free within-person effects: median difference of the per-person medians in SDs, persons with a positive "
      f"difference, Holm p over {v('uni_common_n_tests')} tests): "
      + '; '.join(f"{c} - natural: " + ', '.join(
          f"{v(f'uni_common_{c}_top{i}_feature')} {v(f'uni_common_{c}_top{i}_d')} "
          f"({v(f'uni_common_{c}_top{i}_npos')}/{v(f'uni_common_{c}_top{i}_n')}, p "
          f"{v(f'uni_common_{c}_top{i}_p_holm')})" for i in (1, 2, 3)) for c in ('fast', 'slow'))
      + f". Features with Holm p < {ALPHA}: {v('uni_common_fast_n_sig')} for fast - natural, "
      f"{v('uni_common_slow_n_sig')} for slow - natural (oracle windows: {v('uni_oracle_fast_n_sig')} and "
      f"{v('uni_oracle_slow_n_sig')}). In the headline LR, fast vs natural is driven most by "
      f"{v('coef_fast_top1_feature')} ({v('coef_fast_top1')} log-odds per SD) and slow vs natural by "
      f"{v('coef_slow_top1_feature')} ({v('coef_slow_top1')}); these coefficients are conditional on the correlated "
      'other features, so the model-free effects are the better guide.')
    w('')

    # ── grip
    w('## Instructed grip')
    w('')
    kg1, kg2 = pk(H['grip'], up_g), pk(gbm_g, H['grip'])
    w(f"- Headline (pre-specified, interpretable): `{H['grip']}`, balanced accuracy {ci(gp + '_bacc')}, macro-F1 "
      f"{ci(gp + '_f1')}; recall upright {v(gp + '_rec_upright')}, top left {v(gp + '_rec_top_left')}, top right "
      f"{v(gp + '_rec_top_right')} (tables `sec_grip_models`, `sec_grip_classes`, `sec_grip_confusion`; figure "
      '`SEC_04_grip_confusion`).')
    w(f"- The {v(gp + '_n_features')}-feature orientation model {compare_text(n, H['grip'], up_g)} the up vector "
      f"before the onset alone ({ci('grip_up_bacc')}; {v(kg1 + '_n_changed')} trials change class, log-loss "
      f"difference {v(kg1 + '_d_ll')}). LightGBM on the rich features: {ci('grip_gbm_bacc')}; vs the headline "
      f"{ci(kg2 + '_d_bacc')} (Holm p {v(kg2 + '_d_bacc_p_holm')}, {pair_verdict(n, gbm_g, H['grip'])}).")
    w(f"- Physics (figure `SEC_05_grip_physics`): the grip is the orientation of the phone in the hand, i.e. where "
      f"gravity points in the device frame during the still hold. Median up vector before the onset (device x, y, z): "
      f"upright ({v('upmed_upright_x')}, {v('upmed_upright_y')}, {v('upmed_upright_z')}): the top edge up; top left "
      f"({v('upmed_top_left_x')}, {v('upmed_top_left_y')}, {v('upmed_top_left_z')}): the right edge up; top right "
      f"({v('upmed_top_right_x')}, {v('upmed_top_right_y')}, {v('upmed_top_right_z')}): the left edge up. The "
      'device-frame up vector carries the grip directly; the gravity-frame channels do not encode the orientation, '
      'but the grip changes how the arm moves, so mirror-invariant kinematics carry some grip information too '
      f"(sideways vs upright {v('side_even_bacc')}, above).")
    eids = sorted(int(m.group(1)) for kk in n for m in [re.fullmatch(r'grip_err(\d+)_person', kk)] if m)
    w(f"- Errors of the headline model (tables `sec_grip_errors`, `sec_grip_reference`): {v(gp + '_n_errors')} of "
      f"{v(gp + '_n')} trials, all from {v('grip_err_n_persons')} persons: "
      + '; '.join(f"{v(f'grip_err{i}_person')}: {v(f'grip_err{i}_pattern')}" for i in eids)
      + f". {v(gp + '_n_sideways_swaps')} are top left <-> top right swaps. In {v(gp + '_n_up_matches_predicted')} of "
      f"the {v(gp + '_n_errors')} errors the up vector before the onset lies closer to the median of the PREDICTED "
      f"grip than to that of the instructed one; the gravity sign was inferred from the instructed orientation for "
      f"{v('grip_sign_inferred')} trials (it comes from calibration or the engine vote), and "
      f"{v(gp + '_err_upright_correct')} of the {v(gp + '_err_upright_n')} upright trials of these persons are "
      'recognised, which argues against a sign error. So these trials were most likely held in the predicted grip. '
      'Consistency checks only (no video, no instruction log): '
      f"{v(gp + '_ref_bacc_confirmed')} on the {v(gp + '_ref_n_confirmed')} trials neither the app nor the gravity rule "
      f"flagged, and agreement with the app's start grip on {v(gp + '_ref_n_agree')} of {v(gp + '_ref_n_known')} "
      'trials; both references read the same up vector as the model, so they agree by construction and are not an '
      'independent confirmation.')
    w('')

    # ── handedness
    w('## Handedness and the dominant hand')
    w('')
    lh_ids = sorted(int(m.group(1)) for k in n for m in [re.fullmatch(r'dom_lh(\d+)_person', k)] if m)
    if (val('conc_n_left') or 0) < MIN_PERSONS_PER_CLASS:
        w(f"- **Person-level handedness is not estimable**: {v('conc_n_left')} of {v('conc_n_persons')} persons are "
          f"left-handed (fewer than {v('min_persons_ci')}). No model of handedness is trained or evaluated; the "
          'left-handed persons\' values are shown individually next to the right-handed distribution, never with a '
          'confidence interval (figures `SEC_06_dominance_auc`, `SEC_07_dominance_paired`).')
    else:
        w(f"- Person-level handedness: {v('conc_n_left')} of {v('conc_n_persons')} persons are left-handed; a "
          'person-level model is not part of this lane (open question).')
    sig = verdict(v('dom_p_pooled')) == 'significant'
    at_min = val('dom_n_folds_grid_min') == val('dom_n_folds')
    w(f"- **Trial level** (table `sec_dominance_model`, figure `SEC_06_dominance_auc`): can a leave-one-person-out "
      f"model tell a dominant-hand trial from mirror-INVARIANT kinematics alone? Pre-specified LR: pooled AUC "
      f"{ci('dom_auc')}. Pre-specified reference: the label-permutation null ({v('dom_n_perm')} permutations within "
      f"person through the whole nested pipeline; it keeps every person's class balance): null mean "
      f"{v('dom_null_pooled_mean')}, 95th percentile {v('dom_null_pooled_q95')}, one-sided p {v('dom_p_pooled')}. Mean "
      f"within-person AUC {ci('dom_within_auc')} (null mean {v('dom_null_within_mean')}, p {v('dom_p_within')}); above "
      f"0.5 in {v('dom_n_auc_above_half')} of {v('dom_n_persons')} persons. Accuracy {ci('dom_acc')}; log loss "
      f"{v('dom_ll')} (a constant 0.5 scores {v('dom_ll_const')}). "
      + (f"The nested C was the smallest of the grid ({v('dom_c_min')}) in every outer fold ({v('dom_n_folds_grid_min')} "
         f"of {v('dom_n_folds')}): the tuning hit the grid boundary, the inner folds preferred the strongest "
         f"regularisation, and the predicted P spans only {v('dom_p_min')} to {v('dom_p_max')}. "
         if at_min else
         f"The nested C was {v('dom_c_mode')} for {v('dom_c_mode_trials')} of {v('dom_n_trials')} trials. ")
      + ('A dominant-hand signal is detectable across persons.' if sig else
         f"No dominant-hand signal is detected; the CI is compatible with a pooled AUC up to {v('dom_auc_hi')}."))
    w(f"- Exploratory second model class (LightGBM, {v('domgbm_n_configs')} model classes tried, no permutation "
      f"baseline): pooled AUC {ci('domgbm_auc')}, above the LR's null mean but close to its 95th percentile; against "
      f"the LR on the same trials (`cv.paired_compare`) per-person accuracy {ci('pair_dom_d_acc')} (sign-flip p "
      f"{v('pair_dom_d_acc_p')}), log loss {ci('pair_dom_d_ll')} (p {v('pair_dom_d_ll_p')}). A weak signal that only a "
      'non-linear model finds cannot be excluded.')
    w(f"- Interpretation: the inputs are identical for a trial and its mirror image (unit-tested, and checked on every "
      f"trial: {v('mirror_n_identical')} of {v('mirror_n_trials')} bit-identical, largest difference "
      f"{v('mirror_max_abs_diff')}), so they cannot encode WHICH hand moved; whatever they carry is HOW the dominant "
      f"hand moves. Dominance is confounded with the right hand for the right-handed majority ({v('dom_n_right')} of "
      f"{v('dom_n_persons')} persons). Left-handed persons' within-person AUCs: "
      + ', '.join(f"{v(f'dom_lh{i}_person')} {v(f'dom_lh{i}_auc')}" for i in lh_ids)
      + f" (right-handed median {v('dom_rh_auc_median')}, range {v('dom_rh_auc_min')} to {v('dom_rh_auc_max')}).")
    w('- **Paired within-person comparison** (dominant - non-dominant, per-person medians; Wilcoxon over persons, Holm '
      f"over {v('pd_n_features')} features; table `sec_dominance_paired`, figure `SEC_07_dominance_paired`): "
      + (f"no detectable difference (no feature with Holm p < {ALPHA})" if val('pd_n_sig') == 0 else
         f"{v('pd_n_sig')} of {v('pd_n_features')} features with Holm p < {ALPHA}")
      + f" ({v('pdm_n_sig')} when matched on speed, grip and variant). The CIs bound the differences still compatible "
      'with the data:')
    pdd = L.read_json('sec_dom_paired')
    for f in pdd['features']:
        k = f'pd_{f}'
        w(f"  - {FEATURE_TEXT[f]}: {v(k + '_med')} [{v(k + '_lo')}, {v(k + '_hi')}], dominant higher in "
          f"{v(k + '_npos')} of {v(k + '_n')} persons, Holm p {v(k + '_p_holm')}; condition-matched "
          f"{v('pdm_' + f + '_med')} (Holm p {v('pdm_' + f + '_p_holm')})"
          + ('; left-handed ' + ', '.join(f"{v(f'dom_lh{i}_person')} {v(f'pd_lh{i}_{f}')}" for i in lh_ids)
             if lh_ids else '') + '.')
    w(f"- **Concordance** (table `sec_concordance`, descriptive): neither variable varies outside the left-handed "
      f"persons. All {v('conc_n_phone_right')} of the {v('conc_n_phone_clear')} persons naming a usual phone hand name "
      f"the right hand, and {v('conc_n_nat_right')} of the {v('conc_n_nat')} first free-choice trials used the right "
      f"hand, which a constant would also give. The only contrast is the {v('conc_n_left')} left-handed persons: "
      f"{v('conc_left_nat_right')} of them chose the right hand for the free trial and {v('conc_left_phone_right')} "
      'usually hold the phone in the right hand. Whether the usual phone hand or handedness goes with the free choice '
      'cannot be distinguished with so few left-handed persons and no left-hand first free choice. Repeat-visit free '
      f"trials: {v('conc_rep_n')}, of which {v('conc_rep_left')} with the left hand (so metrics over the repeats are "
      'not right-hand recall only).')
    both_hands = min(val('bal_min_dominant') or 0, val('bal_min_non_dominant') or 0) > 0
    w(f"- Plan balance (`sec_balance`): "
      + (f"every person has dominant and non-dominant trials (at least {v('bal_min_dominant')} and "
         f"{v('bal_min_non_dominant')})" if both_hands else
         f"some persons lack the trials of one hand (fewest dominant {v('bal_min_dominant')}, fewest non-dominant "
         f"{v('bal_min_non_dominant')})")
      + f"; the first block was left-handed in {v('bal_first_left')} sessions and right-handed in "
      f"{v('bal_first_right')} (exact binomial p {v('bal_first_p')}).")
    w('')

    # ── problems
    w('## Problems found: detection, why it matters, remedy, evidence')
    w('')
    probs: list[str] = []
    p_grip = 0                                          # the grip problem's number (0: not listed)
    probs.append(
        f"**The approach window of a slow trial is chosen by its label.** Detected: the app resolved the parameters of "
        f"its online segmentation per instructed condition (`resolveParams` in app/src/core/protocol.ts, mirrored in "
        f"`motionlab.conventions`): a slow approach ends only after {v('seg_slow_end_sustain_ms')} ms below "
        f"{v('seg_slow_end_gyro_dps')} deg/s and {v('seg_slow_end_acc_ms2')} m/s² (default "
        f"{v('seg_default_end_sustain_ms')} ms below {v('seg_default_end_gyro_dps')} deg/s and "
        f"{v('seg_default_end_acc_ms2')} m/s²), with a {v('seg_slow_max_movement_ms')} ms movement cap instead of "
        f"{v('seg_default_max_movement_ms')} ms; pause and jerky trials (natural speed only) also wait longer for their "
        f"end. Re-running the app's own detector on every stored stream (with the stored parameters it reproduces "
        f"{v('reseg_n_reproduced')} of {v('reseg_n_trials')} stored bounds exactly) with ONE parameter set (the "
        f"defaults with the {v('seg_common_max_movement_ms')} ms cap) keeps the bounds of "
        f"{v('reseg_natural_share_std_same')} of the standard natural and {v('reseg_fast_share_same')} of the fast "
        f"trials but of {v('reseg_slow_n_same')} of the {v('reseg_slow_n')} slow trials: the median slow approach "
        f"shrinks from {v('reseg_slow_dur')} s to {v('reseg_slow_dur_common')} s (natural {v('reseg_natural_dur_common')} "
        f"s), ending after a median {v('reseg_slow_path_share')} of its rotation (IQR {v('reseg_slow_path_share_q25')} "
        f"to {v('reseg_slow_path_share_q75')}). The slow trials' stricter criterion is no usable common choice: applied "
        f"to every trial it ends before the arm starts back in only {v('reseg_fast_strict_ok_std')} of the fast and "
        f"{v('reseg_natural_strict_ok_std')} of the natural standard trials (table `sec_segmentation`, figure "
        "`SEC_09_segmentation`). Why it matters: for slow trials the duration and every feature computed over the "
        'window depend on a criterion chosen by the label, which a sensor-only system does not have. Remedy: every '
        'trial re-standardised on the label-blind bounds through the cache\'s code path (exact on the oracle bounds), '
        'every speed model refitted there, and the label-blind LR made the speed headline; the oracle-window numbers '
        'are kept as a label-informed, optimistic reference. Evidence: the headline features on label-blind windows '
        f"{ci(sp + '_bacc')} against {ci(so + '_bacc')} on the oracle windows (paired {ci(k_seg + '_d_bacc')}, Holm p "
        f"{v(k_seg + '_d_bacc_p_holm')}, {pair_verdict(n, *SC.SEGMENTATION_PAIR)}); slow recall {v(sp + '_rec_slow')} "
        f"against {v(so + '_rec_slow')}; LightGBM {v(gc + '_bacc')} against {v(go + '_bacc')} (Holm p "
        f"{v(k_segg + '_d_bacc_p_holm')}); slow longer than natural for "
        f"{v('manip_standard_dur_natural_slow_n_expected')} of {v('manip_standard_dur_natural_slow_n')} persons on the "
        f"oracle windows, {v('manip_common_dur_natural_slow_n_expected')} of {v('manip_common_dur_natural_slow_n')} on "
        "label-blind ones. These label-blind numbers belong to the app's default end rule, which truncates slow "
        'reaches at their first slow phase; a detector built for slow movements could recover more, so they are not a '
        'bound either way. No segmentation parameter depends on the hand, so the hand models are unaffected.')
    probs.append(
        f"**The protocol confounds speed with the grip and with the position in the session.** Detected: the design "
        f"table (`sec_design`): slow trials were upright in {v('des_slow_upright')} of {v('des_slow_n')}, standard fast "
        f"trials sideways in {v('des_side_fast')} against {v('des_side_natural')} of the natural ones; slow trials came "
        f"first (median position {v('des_pos_slow')}), then fast ({v('des_pos_fast')}), then natural "
        f"({v('des_pos_natural')}). Why it matters: a model can predict the speed from the grip or from practice "
        f"effects instead of the movement's speed: the grip alone gives {ci(short(gripm) + '_bacc')}, and LightGBM, "
        f"which sees the device-frame channels, predicts fast for {v(gus + '_pred_fast')} of the upright and "
        f"{v(gss + '_pred_fast')} of the sideways trials (instructed fast: {v(gus + '_true_fast')} and "
        f"{v(gss + '_true_fast')}). Remedy: the grip-only model as a nuisance reference, and every speed comparison "
        'repeated within grip strata. Evidence: the headline '
        f"{compare_text(n, hs, gripm)} the grip-only model ({ci(k_grip + '_d_bacc')}, Holm p "
        f"{v(k_grip + '_d_bacc_p_holm')}); within the upright trials it reaches {v(us + '_bacc')} and LightGBM "
        f"{v(gus + '_bacc')} ({ci(kgu + '_d_bacc')}, Holm p {v(kgu + '_d_bacc_p_holm')}). The speed accuracies may still "
        'partly reflect the protocol\'s order and grip structure.')
    probs.append(
        f"**The fast vs natural duration contrast mixes instruction and practice.** Detected: natural reaches shorten over the "
        f"session (median within-person rho {v('des_practice_rho')}, negative for {v('des_practice_neg')} of "
        f"{v('des_practice_n')} persons, p {v('des_practice_p')}) and fast trials came before the natural ones for "
        f"{v('des_fast_before')} of {v('des_fast_before_n')} persons. Why it matters: the unadjusted contrast compares "
        'early fast reaches with partly later natural ones, so instruction and practice are mixed. Remedy: '
        f"position-adjusted contrasts (sensitivity). Evidence: natural/fast duration ratio {ci(fk + '_ratio')} "
        f"unadjusted ({v(fk + '_n_expected')} of {v(fk + '_n')} persons as instructed, Holm p {v(fk + '_p_holm')}), "
        f"{ci(adj + '_ratio')} adjusted ({v(adj + '_n_expected')} of {v(adj + '_n')}, Holm p {v(adj + '_p_holm')}, "
        f"nominal p {v(adj + '_p')}), {ci('adj_standard_duration_overlap_ratio')} with only the natural trials made "
        f"before the last fast one ({v('adj_standard_duration_overlap_n_expected')} of "
        f"{v('adj_standard_duration_overlap_n')}, Holm p {v('adj_standard_duration_overlap_p_holm')}): the direction "
        'is the instructed one in each, the size stays uncertain. The peak angular speed rose for '
        f"{v(pkk + '_n_expected')} of {v(pkk + '_n')} persons (Holm p {v(pkk + '_p_holm')}).")
    if val(gp + '_n_errors'):
        p_grip = len(probs) + 1
        probs.append(
            f"**Grip labels that are probably not the grip performed.** Detected: the confusion matrix (`SEC_04`): the "
            f"{v(gp + '_n_errors')} errors of the headline grip model come from {v('grip_err_n_persons')} persons, "
            f"{v(gp + '_n_flagged_mismatch')} of them flagged as held in another grip. Why it matters: the accuracy "
            'against the instruction understates the sensors, and any result stratified by instructed grip (in every '
            'lane) mis-assigns these trials. Remedy: none applied to the labels (kept as instructed); the errors are '
            'explained by the physics (the up vector of each error lies in the cluster of the predicted grip, '
            f"{v(gp + '_n_up_matches_predicted')} of {v(gp + '_n_errors')}) and by the sign check (gravity signs "
            f"inferred from the instruction: {v('grip_sign_inferred')}; upright trials of these persons recognised: "
            f"{v(gp + '_err_upright_correct')} of {v(gp + '_err_upright_n')}). Evidence: consistency "
            f"checks only: {v(gp + '_ref_bacc_confirmed')} on the {v(gp + '_ref_n_confirmed')} unflagged trials and "
            f"agreement with the app's start grip on {v(gp + '_ref_n_agree')} of {v(gp + '_ref_n_known')}; both read the "
            'same signal as the model, and no independent record of the grip held exists.')
    same = [(r_a, r_b) for r_a, r_b in SC.MODEL_PAIRS if pair_verdict(n, r_a, r_b) == 'identical']
    if same:
        probs.append(
            '**Models that make the same predictions.** Detected: '
            + '; '.join(f"`{r_a}` and `{r_b}` change no trial's class (log-loss difference "
                        f"{v(pk(r_a, r_b) + '_d_ll')})" for r_a, r_b in same)
            + '. Why it matters: a test of such a pair has no meaning. Remedy and evidence: such a pair is reported as '
            'identical, never as significant or not significant (`sec_model_pairs`, column verdict).')
    for i, txt in enumerate(probs, start=1):
        w(f'{i}. {txt}')
    w('')

    # ── decisions
    w('## Decisions and their evidence')
    w('')
    w(f"- **Speed headline** = the pre-specified interpretable LR on the label-blind windows (`HEADLINE` in "
      f"`sec_compute.py`: `{hs}`): the only windows a sensor-only system has (problem 1). It stays the headline "
      f"because it was pre-specified and is interpretable, not because it is the best: LightGBM "
      f"{compare_text(n, gbc, hs)} it on balanced accuracy ({ci(k_gl + '_d_bacc')}, Holm p "
      f"{v(k_gl + '_d_bacc_p_holm')}), {compare_text(n, gbc, hs, 'd_f1')} it on macro-F1 ({ci(k_gl + '_d_f1')}, Holm p "
      f"{v(k_gl + '_d_f1_p_holm')}) and {compare_text(n, gbc, hs, 'd_ll')} it on log loss ({ci(k_gl + '_d_ll')}, Holm "
      f"p {v(k_gl + '_d_ll_p_holm')}).")
    w(f"- **Oracle-window speed numbers** ({v(so + '_bacc')} LR, {v(go + '_bacc')} LightGBM) are reported only as a "
      'label-informed, optimistic reference, never as the sensor-only result.')
    w(f"- **Speed features**: the {v('n_even_features')} kinematic features, not the duration alone "
      f"({ci(k_dur + '_d_bacc')} balanced accuracy, Holm p {v(k_dur + '_d_bacc_p_holm')}).")
    w(f"- **Training trials**: all primary trials; training on the standard variant only changes balanced accuracy on "
      f"the standard trials by {ci(k_std + '_d_bacc')} (Holm p {v(k_std + '_d_bacc_p_holm')}).")
    w(f"- **Common segmentation parameters**: the app's defaults with the longest movement cap of any condition; the "
      f"slow trials' criterion ends too few fast ({v('reseg_fast_strict_ok_std')}) and natural "
      f"({v('reseg_natural_strict_ok_std')}) reaches before the return (problem 1).")
    w(f"- **Metrics**: balanced accuracy (primary) and macro-F1, with per-class recall, precision and F1; tuning on the "
      f"class-balanced log loss with class-balanced weights. Justification (a metric choice, not a tested remedy): "
      f"natural trials are {v('n_speed_natural')} of {v('n_trials')}, so plain accuracy rewards predicting natural; "
      'the price of the class-balanced weights is a low precision of the rare slow class (above).')
    w('- **Speed vs the protocol**: the grip-only model is reported as a nuisance reference and the model comparisons '
      'are repeated within grip strata (problem 2).')
    w('- **Manipulation check** on the standard-variant trials (the variants occur at natural speed only), on both '
      'windows, with the position-adjusted contrasts as a sensitivity check (problem 3).')
    w(f"- **Grip features**: the up vector before the onset carries the grip (the full orientation set "
      f"{compare_text(n, H['grip'], up_g)} it); the agreements with the app's start grip and on the unflagged trials "
      'are consistency checks' + (f' (problem {p_grip}).' if p_grip else '.'))
    w(f"- **Dominance**: mirror-invariant features only (bit-identical for {v('mirror_n_identical')} of "
      f"{v('mirror_n_trials')} trials and their mirror images); the pre-specified reference is the within-person "
      'label-permutation null (it keeps each person\'s class balance), not 0.5.')
    w('- **Person-level handedness**: no model (too few left-handed persons); they are shown individually, never with a '
      'bootstrap CI.')
    w('- **Calibration is not assessed**: none of these models feeds the displayed percentage (the product is the '
      'P(right) of the hand model); their log losses are reported with the other metrics.')
    w('')
    w('## Figures (`figures/sec/`)')
    w('')
    cp = L.fig_dir / 'captions.json'
    caps = json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else {}
    for k in sorted(caps):
        w(f"- `{k}`: {caps[k].get('decision', '')}")
    w('')
    w('## Tables (`tables/sec/`, CSV + booktabs .tex)')
    w('')
    for t, what in TABLES.items():
        w(f'- `{t}`: {what}')
    w('')
    w('## Commands')
    w('')
    w('```')
    w('cd analysis')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes sec            # missing / stale experiments')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes sec --force    # everything (the permutation '
      'baseline is the slowest part)')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes sec --only "sec_speed_*" --force')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes sec --only sec_reseg --force   # re-segmentation')
    w('.venv/Scripts/python.exe -m stage2.pipeline figures --lanes sec          # from the saved files only')
    w('.venv/Scripts/python.exe -m stage2.pipeline figures --lanes sec --only SEC_09_segmentation')
    w('.venv/Scripts/python.exe -m stage2.pipeline tables --lanes sec')
    w('.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes sec          # numbers, this summary, finalists')
    w('.venv/Scripts/python.exe -m pytest stage2/tests/test_sec.py -q -p no:cacheprovider')
    w('```')
    w('')
    w('The pipeline takes one target per call. A saved result is current while the data, its declared specification '
      '(features, classes, grid, cap, seed, bootstrap size; on label-blind windows also the parameter set and the '
      'hashes of the saved re-segmentation) and, for the permutation baseline and the re-segmentation, their code are '
      'unchanged; after editing a feature or model definition, recompute with `--force`. Derived files, figures, '
      'tables, numbers and this summary are always rewritten from the saved results.')
    w('')
    w('## Conclusions')
    w('')
    w(f"- Speed from the sensors alone (label-blind windows): balanced accuracy {ci(sp + '_bacc')} with the "
      f"interpretable LR and {ci(gc + '_bacc')} with LightGBM (chance {chance}). With the app's default end detector, "
      f"which truncates slow reaches at their first slow phase, slow recall is {v(sp + '_rec_slow')} against "
      f"{v(so + '_rec_slow')} on the label-informed oracle windows"
      + (f"; the oracle-window accuracy ({v(so + '_bacc')}) is optimistic because the segmentation used the label."
         if seg_drop else '.'))
    w(f"- Part of what separates the speeds is the protocol: the grip alone reaches {v(short(gripm) + '_bacc')}, and "
      f"the slow trials were nearly all upright and early in the session. Within the upright trials the headline "
      f"reaches {v(us + '_bacc')} over {v(us + '_included')}.")
    w(('- The persons followed the fast instruction at the peak of the movement' if peak_ok else
       '- A faster peak for the fast trials is not established')
      + f" ({v(pkk + '_n_expected')} of {v(pkk + '_n')} faster); the shortening of the whole reach is "
      + ('clear' if fast_dur_clear else 'uncertain') + f" (ratio {ci(fk + '_ratio')}), and the protocol's order mixes "
      'it with practice.')
    w(f"- Grip: recognised from the up vector of the still hold ({v(gp + '_bacc')} against the instruction); the "
      f"{v(gp + '_n_errors')} errors look like grips performed differently by {v('grip_err_n_persons')} persons, a "
      'reading supported by physics and consistency checks but not independently confirmed.')
    w('- Dominant hand: '
      + ('a dominant-hand signal is detected' if sig else 'no dominant-hand signal detected')
      + f" in mirror-invariant kinematics (pooled AUC {ci('dom_auc')}, permutation p {v('dom_p_pooled')}); within "
      + ('persons no detectable difference between the hands, with CIs that still allow moderate differences (above).'
         if val('pd_n_sig') == 0 else
         f"persons {v('pd_n_sig')} of {v('pd_n_features')} features differ (Holm).")
      + ' Whether the hand model\'s evidence lies only in the chiral channels is a hypothesis this does not test.')
    w('')
    w('## Caveats and open questions')
    w('')
    w('- None of these models runs in production; a production speed or grip estimate would take its bounds from the '
      'cue-free detector (lane m4), whose end detection meets the same difficulty with slow reaches, and its '
      'sensitivity to end-detection errors would have to be measured.')
    w('- The label-blind estimate belongs to one detector setting (the app\'s defaults); other label-blind windows '
      '(a fixed window from the onset, a detector tuned for slow movements) would give other numbers.')
    w('- The linear path length integrates the acceleration with zero velocity at both bounds; the drift that the '
      'linear correction cannot remove grows with the duration.')
    w(f"- Slow trials are few: median {v('slow_per_person_median')} per person (at most {v('slow_per_person_max')}), so "
      'per-person slow statistics rest on very few movements.')
    w('- Self-reported handedness is the only dominance measure (no questionnaire such as the Edinburgh inventory).')
    w('- Added after seeing the results (exploratory, p-values not corrected for that): the per-person fast/natural '
      'description (`SEC_08`, partly circular), the LightGBM dominance model, the grip consistency checks, the '
      'segmentation check, the grip-only reference, the grip strata and the position-adjusted contrasts.')
    w('- Open: a person-relative speed measure (normalised by the person\'s own natural movements); a dominance-related '
      'movement style that transfers to the left-handed (needs more left-handed persons).')
    return '\n'.join(lines) + '\n'


#: the report tables of the lane and what each holds (the summary lists them)
TABLES: dict[str, str] = {
    'sec_speed_models': 'speed models: windows, balanced accuracy, macro-F1 (CIs), log loss',
    'sec_speed_classes': 'speed models: recall, precision and F1 per class',
    'sec_speed_confusion': 'speed confusion matrices (counts, row shares), both windows',
    'sec_design': 'the protocol: speed by grip and by position in the session',
    'sec_speed_strata': 'the speed models within grip strata',
    'sec_speed_strata_pairs': 'LightGBM vs LR within grip strata',
    'sec_speed_manipulation': 'manipulation check of the speed instruction (both windows, position-adjusted)',
    'sec_speed_univariate': 'model-free within-person speed effects of every kinematic feature (both windows)',
    'sec_segmentation': "the approach window per instructed speed: the app's stored segment vs one parameter set",
    'sec_grip_models': 'grip models',
    'sec_grip_classes': 'grip models: recall, precision and F1 per class',
    'sec_grip_confusion': 'grip confusion matrices',
    'sec_grip_errors': 'the misclassified trials of the headline grip model',
    'sec_grip_reference': 'grip models against the instruction, the unflagged trials and the app\'s start grip '
                          '(consistency checks)',
    'sec_model_pairs': 'paired model comparisons: balanced accuracy and macro-F1 (Holm)',
    'sec_model_pairs_ll': 'paired model comparisons: log loss and accuracy (Holm)',
    'sec_dominance_model': 'dominant-hand models with the permutation baseline',
    'sec_dominance_persons': 'dominant-hand model per person',
    'sec_dominance_paired': 'dominant vs non-dominant movements within person',
    'sec_concordance': 'handedness, usual phone hand and free-choice hand',
    'sec_features': 'feature catalogue (definitions, units, mirror parity)',
}


#: numeric tokens that are conventions, not results: the 95 % of a CI, AUC 0.5 (chance), the 0.05 level, the 100 %
#: of a constant answer; '1/3' (chance balanced accuracy) is removed before tokenising
CONVENTIONS = {'95', '0.5', '0.05', '100%'}
_NUMBER = re.compile(r'[-+]?\d+(?:\.\d+)?%?')


def numeric_tokens(text: str) -> list[str]:
    """Numbers of a text outside code blocks / spans and identifiers (P05, L2, stage2) and not glued to a word."""
    body = re.sub(r'```.*?```', ' ', text, flags=re.S)
    body = re.sub(r'`[^`]*`', ' ', body)
    body = body.replace('1/3', ' ')
    out = []
    for m in _NUMBER.finditer(body):
        a, b = m.span()
        prev = body[a - 1] if a else ' '
        nxt = body[b] if b < len(body) else ' '
        if prev.isalnum() or prev in '_.\\' or nxt.isalpha() or nxt == '_':
            continue
        out.append(m.group())
    return out


def untraced_numbers(text: str, n: dict[str, dict[str, Any]]) -> list[str]:
    """Numeric tokens of ``text`` that are no formatted value of the numbers file and no convention."""
    vals: set[str] = set()
    for k in n:
        s = fmt(n, k)
        for t in numeric_tokens(s):
            vals |= {t, t.lstrip('+'), t.rstrip('%'), t.lstrip('+').rstrip('%')}
    out = []
    for tok in numeric_tokens(text):
        bare = tok.lstrip('+').rstrip('%')
        if tok in vals or bare in vals or tok in CONVENTIONS or bare in CONVENTIONS:
            continue
        out.append(tok)
    return out


def write_summary() -> None:
    n = read_numbers()
    text = render_summary(n)
    bad = untraced_numbers(text, n)
    if bad:
        raise ValueError(f'sec summary: numbers not traced to results/sec_numbers.json: {sorted(set(bad))}')
    L.summary_path.write_text(text, encoding='utf-8', newline='\n')
