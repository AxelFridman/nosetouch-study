"""Tables (CSV + LaTeX booktabs) of lane m1, from saved files only.

    python -m stage2.pipeline tables --lanes m1      # tables/m1/m1_*.csv|.tex

Conventions: accuracy differences are in percentage points ('pts'), p values are sign-flip p values with their
Holm-adjusted value inside the named comparison family, the ECE is reported with the 95th percentile of its
perfect-calibration null (no percentile bootstrap CI), and strata with fewer than five persons get no CI.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from stage2.lanes import m1_experiments as E
from stage2.pipeline import Lane, selected

L = Lane('m1')
PCT = '.1%'


def _rows(name: str, key: str = 'rows') -> pd.DataFrame:
    return pd.DataFrame(L.read_json(name)[key])


def _ci(lo: Any, hi: Any, pct: bool = True, nd: int = 1, scale: float = 100.0) -> str:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or pd.isna(lo):
        return '--'
    return f'[{scale * lo:.{nd}f}, {scale * hi:.{nd}f}]' if pct else f'[{lo:.3f}, {hi:.3f}]'


def _models() -> pd.DataFrame:
    return _rows('m1_models').set_index('name', drop=False)


def wilson(acc: Any, n: Any, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95 % interval of a proportion ``acc`` of ``n`` trials (the natural-use set has one trial per person, so a
    person bootstrap is a trial bootstrap; the Wilson interval is the standard for a single small proportion; a
    half-credit tie gives a non-integer count, which the formula accepts)."""
    if acc is None or n is None or not n or (isinstance(acc, float) and math.isnan(acc)):
        return math.nan, math.nan
    n = int(n)
    p = float(acc)
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _focus() -> str:
    return str(L.read_json('m1_failures').get('model') or E.BASELINE)


# ───────────────────────────── tables ─────────────────────────────


def t_catalogue() -> None:
    cat = pd.DataFrame(L.read_json('m1_catalogue')['rows'])
    L.export_table(cat, 'm1_feature_catalogue_full', tex=False)
    t = pd.DataFrame({
        'Feature': cat['name'], 'Family': cat['family'], 'Parity': cat['parity'], 'Unit': cat['unit'],
        'Screen flip': np.where(cat['flips'].astype(bool), 'sign', 'none'),
        'AUC': cat['auc'], '95% CI': [_ci(a, b, False) for a, b in zip(cat['auc_lo'], cat['auc_hi'])],
        'Persons >0.5': cat['share_persons_above_50'], 'Within median': cat['within_auc_median'],
        'Sign rule': cat.get('sign_rule_accuracy', pd.Series([np.nan] * len(cat))),
        'Median L': cat['left_median'], 'Median R': cat['right_median']})
    L.export_table(t, 'm1_feature_catalogue', label='tab:m1_feature_catalogue',
                   caption=('Feature catalogue of lane m1: univariate leave-one-person-out AUC (direction-only rule '
                            'sigmoid(s x / RMS(x)): direction s and scale fitted on the training persons, no intercept; '
                            '95\\,\\% person-bootstrap CI), share of persons whose own trials the feature orders correctly '
                            'more often than not, median within-person AUC, accuracy of the sign rule (odd features; '
                            'direction from the training persons) and per-hand medians. Screen flip: whether the feature '
                            'changes sign when the phone is turned screen-away (every feature is unchanged by the three '
                            'screen-toward grips). Formulas, frames and meanings: m1\\_feature\\_catalogue\\_full.csv.'),
                   column_formats={'AUC': '.3f', 'Persons >0.5': '.0%', 'Within median': '.2f', 'Sign rule': '.1%',
                                   'Median L': '.3g', 'Median R': '.3g'}, size=r'\scriptsize')


def t_models() -> None:
    mt = _models()
    L.export_table(mt.drop(columns=['notes']), 'm1_models_full', tex=False)
    t = pd.DataFrame({
        'Model': mt['label'], 'Design': mt['design'], 'Accuracy': mt['accuracy'],
        '95% CI': [_ci(a, b) for a, b in zip(mt['accuracy_lo'], mt['accuracy_hi'])], 'AUC': mt['roc_auc'],
        'Log loss': mt['log_loss'].where(~mt['hard'].astype(bool)), 'ECE': mt['ece'].where(~mt['hard'].astype(bool)),
        'ECE null 95%': mt.get('ece_null_q95'), 'Slope': mt.get('cox_slope'),
        'Natural use': mt['nat_acc_half'], 'k = 3': mt.get('acc_k3'),
        'vs ref. (pts)': 100 * mt.get('vs_ref_acc_diff'), 'Holm p': mt.get('vs_ref_p_acc_holm'),
        'Configs': mt['n_configs_tried']})
    L.export_table(t.reset_index(drop=True), 'm1_models', label='tab:m1_models',
                   caption=('Every lane m1 model, leave-one-person-out over the primary set (95\\,\\% person-bootstrap '
                            'CI). ECE with the 95th percentile of its perfect-calibration null; Slope: calibration slope '
                            '(1 = calibrated). Natural use: accuracy on the free-choice trials (never trained on; a P of '
                            'exactly 0.5 counts one half); k = 3: per-person balanced accuracy when three attempts are '
                            'averaged; vs ref.: paired accuracy difference to the reference floor in percentage points '
                            'with the Holm-adjusted sign-flip p over every m1 model; Configs: configurations of its family '
                            'compared on the outer folds (1 = not chosen on the outer folds). Design: pre-specified, '
                            'specified after the pooled description, nested (choice inside every fold), outer-fold family '
                            'member (exploratory) or sensitivity run.'),
                   column_formats={'Accuracy': PCT, 'AUC': '.3f', 'Log loss': '.3f', 'ECE': '.3f', 'ECE null 95%': '.3f',
                                   'Slope': '.2f', 'Natural use': '.0%', 'k = 3': PCT, 'vs ref. (pts)': '+.1f',
                                   'Holm p': '.2g', 'Configs': 'd'}, size=r'\scriptsize')


def t_rules() -> None:
    mt = _models()
    st = _rows('m1_strata')
    rules = [n for n in mt.index if mt.loc[n, 'group'] == 'rules']
    rows = []
    for n in rules:
        r = mt.loc[n]
        row = {'Rule': r['label'], 'Accuracy': r['accuracy'], '95% CI': _ci(r['accuracy_lo'], r['accuracy_hi'])}
        for pl, lab in (('android', 'Android'), ('ios', 'iOS')):
            q = st[(st['model'] == n) & (st['stratum'] == 'platform') & (st['level'] == pl)]
            row[lab] = q['accuracy'].iloc[0] if len(q) else np.nan
        row.update({'Median person': r['person_acc_median'], 'Natural use': r['nat_acc_half'],
                    'k = 3': r.get('acc_k3'), 'Parameters': {'m1_rule_fhy_sign': '0', 'm1_rule_fhy_sign_cal': '1 + 1',
                                                             'm1_rule_fhy_threshold': '2 + 2'}.get(n, 'in-fold choice')})
        rows.append(row)
    L.export_table(pd.DataFrame(rows), 'm1_rules', label='tab:m1_rules',
                   caption=('Zero- and one-parameter rules, leave-one-person-out (95\\,\\% person-bootstrap CI). Parameters: '
                            'fitted numbers per fold (the calibrated sign rule learns a direction and its hit rate; the '
                            'threshold rule a direction, a threshold and the two side probabilities; the nested rule also '
                            'chooses its feature inside every fold).'),
                   column_formats={'Accuracy': PCT, 'Android': PCT, 'iOS': PCT, 'Median person': PCT,
                                   'Natural use': '.0%', 'k = 3': PCT})


def t_rules_per_person() -> None:
    """Per-person accuracy of every rule and of the small-model finalist (from the saved results' per-person tables)."""
    from stage2 import cv
    mt = _models()
    focus = _focus()
    names = [n for n in mt.index if mt.loc[n, 'group'] == 'rules'] + [focus]
    cols = {}
    for n in names:
        try:
            pp = cv.load_result(n).per_person.set_index('person_id')
        except FileNotFoundError:
            continue
        cols[mt.loc[n, 'label']] = pp['accuracy']
        if n == focus:
            cols['Trials'] = pp['n']
    t = pd.DataFrame(cols).reset_index().rename(columns={'index': 'Person', 'person_id': 'Person'})
    L.export_table(t, 'm1_rules_per_person', label='tab:m1_rules_per_person',
                   caption='Per-person leave-one-person-out accuracy of the rules and of the small-model finalist.',
                   column_formats={c: '.0%' for c in t.columns if c not in ('Person', 'Trials')} | {'Trials': 'd'},
                   size=r'\scriptsize')


def t_conditions() -> None:
    sj = L.read_json('m1_strata')
    st = pd.DataFrame(sj['rows'])
    min_p = int(sj.get('min_persons_ci', 5))
    focus = _focus()
    s = st[st['model'] == focus]
    L.export_table(st, 'm1_strata_all', tex=False)
    t = pd.DataFrame({'Stratum': s['stratum'], 'Level': s['level'], 'Trials': s['n_trials'], 'Persons': s['n_persons'],
                      'Accuracy': s['accuracy'], '95% CI': [_ci(a, b) for a, b in zip(s['accuracy_lo'], s['accuracy_hi'])],
                      'Log loss': s['log_loss']})
    L.export_table(t.reset_index(drop=True), 'm1_conditions', label='tab:m1_conditions',
                   caption=('Accuracy of the small-model finalist by condition, platform, handedness, duration quartile '
                            'and quality flag (strata after prediction; person-bootstrap CIs only for levels with at least '
                            f'{min_p} persons).'),
                   column_formats={'Accuracy': PCT, 'Log loss': '.3f'}, size=r'\scriptsize')


def t_calibration() -> None:
    cal = L.read_json('m1_calibration')
    met = pd.DataFrame(cal['metrics'])
    pc = pd.DataFrame(cal['paired'])
    L.export_table(met, 'm1_calibration_full', tex=False)
    L.export_table(pc, 'm1_calibration_paired', tex=False)
    from stage2.lanes.m1_texts import bins_outside_p
    mt = _models()
    labels = dict(zip(mt['name'], mt['label']))
    m = met[met['scope'] == 'all'].copy()
    t = pd.DataFrame({'Model': m['model'].map(labels).fillna(m['model']), 'ECE': m['ece'],
                      'Null 95%': m['ece_null_q95'], 'p': m['ece_p'], 'Slope': m['cox_slope'],
                      'Slope CI': [_ci(a, b, False) for a, b in zip(m['cox_slope_lo'], m['cox_slope_hi'])],
                      'Intercept': m['cox_intercept'], 'Bins out': m['bins_outside'],
                      'Bins p': [bins_outside_p(r) for _, r in m.iterrows()], 'Brier': m['brier'],
                      'Log loss': m['log_loss']})
    for pl, lab in (('android', 'ECE Android'), ('ios', 'ECE iOS')):
        q = met[met['scope'] == pl].set_index('model')
        t[lab] = m['model'].map(q['ece']).to_numpy()
    L.export_table(t.reset_index(drop=True), 'm1_calibration', label='tab:m1_calibration',
                   caption=('Calibration of the lane m1 models and remedies (leave-one-person-out probabilities). ECE: '
                            f'{cal.get("ece_bins", 10)} equal-mass bins, with the 95th percentile and the p value of its '
                            'perfect-calibration null (labels redrawn from the model\'s own P); Slope / Intercept: the '
                            'logistic recalibration of the outcome on logit(P) (slope > 1: under-confident), slope with a '
                            '95\\,\\% person-bootstrap CI; Bins out: reliability bins whose person-bootstrap band excludes '
                            'the diagonal; Bins p: binomial tail of that count against the bands\' nominal 5\\,\\% miss '
                            'rate. Every calibrator is fitted on the inner cross-fitted predictions of the training '
                            'persons.'),
                   column_formats={'ECE': '.3f', 'Null 95%': '.3f', 'p': '.2g', 'Slope': '.2f', 'Intercept': '+.2f',
                                   'Bins out': 'd', 'Bins p': '.2g', 'Brier': '.3f', 'Log loss': '.3f', 'ECE Android': '.3f',
                                   'ECE iOS': '.3f'}, size=r'\scriptsize')
    if len(pc):
        r = pd.DataFrame({'Model': pc['model'].map(labels).fillna(pc['model']), 'Against': pc['baseline'].map(labels),
                          'Kind': pc['kind'], 'dLog loss': pc['ll_diff'],
                          'dLL CI': [_ci(a, b, False) for a, b in zip(pc['ll_diff_lo'], pc['ll_diff_hi'])],
                          'p': pc['p_ll'], 'Holm p': pc['p_ll_holm'], 'dBrier': pc['brier_diff'],
                          'dBrier CI': [_ci(a, b, False) for a, b in zip(pc['brier_diff_lo'], pc['brier_diff_hi'])],
                          'dECE': pc['ece_diff']})
        L.export_table(r, 'm1_calibration_remedies', label='tab:m1_calibration_remedies',
                       caption=('Calibration remedies and treatments, paired against the uncalibrated model on the same '
                                'trials: per-person log-loss and Brier differences with paired person-bootstrap 95\\,\\% '
                                'CIs and sign-flip p (Holm within the remedies of one model), and the pooled ECE '
                                'difference (point only).'),
                       column_formats={'dLog loss': '+.3f', 'p': '.2g', 'Holm p': '.2g', 'dBrier': '+.4f', 'dECE': '+.3f'},
                       size=r'\scriptsize')


def t_k_aggregation() -> None:
    js = L.read_json('m1_k_aggregation')
    ka = pd.DataFrame(js['rows'])
    L.export_table(ka, 'm1_k_aggregation_full', tex=False)
    pr = pd.DataFrame(js.get('paired', []))
    if len(pr):
        L.export_table(pr, 'm1_k_aggregation_paired', tex=False)
    from stage2.lanes.m1_texts import agg_model
    model = agg_model(ka)
    g = ka[ka['model'] == model]
    t = pd.DataFrame({'k': g['k'].astype(int), 'Aggregation': g['method'], 'Accuracy': g['accuracy'],
                      'ECE': g['ece'], 'Lean': g['lean'], 'Log loss': g['log_loss'],
                      'LL 95% CI': [_ci(a, b, False) for a, b in zip(g['log_loss_lo'], g['log_loss_hi'])],
                      'Persons': g['n_persons'].astype(int)})
    L.export_table(t.reset_index(drop=True), 'm1_k_aggregation', label='tab:m1_k_aggregation',
                   caption=('Calibration of k-attempt aggregates of the small-model finalist (random draws of k trials of '
                            'one person and hand): mean of P, mean of logits, sum of logits, and $k^{-\\alpha}$ times the '
                            'sum, with $\\alpha$ fitted on the training persons of each fold. Lean: mean of (observed share - P) towards '
                            'the predicted side (> 0 under-confident, < 0 over-confident).'),
                   column_formats={'Accuracy': PCT, 'ECE': '.3f', 'Lean': '+.3f', 'Log loss': '.3f'}, size=r'\scriptsize')


def t_treatment() -> None:
    mt = _models()
    fam = f'{E.BASELINE}:options'
    t = mt[mt.get('vs_base_family') == fam]
    out = pd.DataFrame({'Option': t['label'], 'Design': t['design'], 'Accuracy': t['accuracy'],
                        'dAcc (pts)': 100 * t['vs_base_acc_diff'],
                        'dAcc CI': [_ci(a, b) for a, b in zip(t['vs_base_acc_diff_lo'], t['vs_base_acc_diff_hi'])],
                        'Holm p (acc)': t['vs_base_p_acc_holm'], 'Log loss': t['log_loss'], 'dLL': t['vs_base_ll_diff'],
                        'dLL CI': [_ci(a, b, False) for a, b in zip(t['vs_base_ll_diff_lo'], t['vs_base_ll_diff_hi'])],
                        'p (LL)': t['vs_base_p_ll'], 'Holm p (LL)': t['vs_base_p_ll_holm']})
    L.export_table(out.reset_index(drop=True), 'm1_data_treatment', label='tab:m1_data_treatment',
                   caption=('Data treatment, representation and model form, each paired against the PHYS6 antisymmetric '
                            'logistic model on the same trials (pooled accuracy difference in percentage points and mean '
                            'per-person log-loss difference, 95\\,\\% paired person-bootstrap CI, exact sign-flip p; Holm '
                            f'over the {len(out)} options of this family).'),
                   column_formats={'Accuracy': PCT, 'dAcc (pts)': '+.1f', 'Holm p (acc)': '.2g', 'Log loss': '.3f',
                                   'dLL': '+.3f', 'p (LL)': '.2g', 'Holm p (LL)': '.2g'}, size=r'\scriptsize')


def t_failures() -> None:
    fl = L.read_json('m1_failures')
    ce = pd.DataFrame(fl['confident_errors'])
    if len(ce):
        L.export_table(ce, 'm1_confident_errors', label='tab:m1_confident_errors',
                       caption=(f'Confident errors of the small-model finalist (P(true hand) < '
                                f'{fl["confident_threshold"]:.1f}), with their conditions and flags.'),
                       column_formats={'duration_s': '.2f', 'p_label': '.3f'}, size=r'\scriptsize')
    pp = pd.DataFrame(fl['per_person'])
    L.export_table(pp, 'm1_per_person', label='tab:m1_per_person',
                   caption='Per-person leave-one-person-out accuracy of the small-model finalist.',
                   column_formats={'accuracy': PCT, 'acc_left': PCT, 'acc_right': PCT, 'log_loss': '.3f'},
                   size=r'\scriptsize')
    bl = pd.DataFrame(fl['blocks_of_swap_persons'])
    if len(bl):
        L.export_table(bl, 'm1_swap_person_blocks', label='tab:m1_swap_person_blocks',
                       caption=('Blocks of the persons with swap-signature blocks (screen.block\\_consistency, '
                                'data-driven): accuracy and mean P(instructed hand) of the small-model finalist per block. '
                                'Labels are never changed.'),
                       column_formats={'accuracy': PCT, 'mean_p_label': '.2f'})
    cb = pd.DataFrame(fl.get('candidate_blocks', []))
    if len(cb):
        L.export_table(cb, 'm1_candidate_blocks', label='tab:m1_candidate_blocks',
                       caption=('Candidate wrong-hand blocks: blocks of at least two trials whose mean P(instructed hand) '
                                f'under the small-model finalist is below {fl["confident_threshold"]:.1f}, with the mean '
                                'P(instructed hand) of the neighbouring blocks of the same session. Labels are never '
                                'changed; the D6-screen sensitivity run removes such trials inside each fold.'),
                       column_formats={'accuracy': PCT, 'mean_p_label': '.2f', 'neighbour_mean_p_label': '.2f'})


def t_coefficients() -> None:
    for nm, label in (('m1_phys6_coefficients', 'the PHYS6 antisymmetric logistic model'),
                      ('m1_small_coefficients', 'the small-model finalist (folds of its modal treatment)')):
        try:
            cs = L.read_json(nm)
        except FileNotFoundError:
            continue
        df = pd.DataFrame(cs['rows'])
        if not len(df):                       # never write an empty table
            continue
        L.export_table(df, nm, label=f'tab:{nm}',
                       caption=(f'Coefficients of {label} across the leave-one-person-out folds (per scaled unit of the '
                                'feature: median, min, max; raw: per unit of the feature) and the share of folds that '
                                'agree with the median sign.'),
                       column_formats={'coef_std_median': '+.2f', 'coef_std_min': '+.2f', 'coef_std_max': '+.2f',
                                       'coef_raw_median': '+.3g', 'sign_agreement': '.0%'})


def t_natural() -> None:
    nj = L.read_json('m1_natural')
    nt = pd.DataFrame(nj['transfer'])
    t = pd.DataFrame({'Feature': nt['feature'], 'Component': nt['component'], 'Trials': nt['n'],
                      'Observed': nt['observed'], 'Expected': nt['expected'],
                      '95% range': [f'[{a}, {b}]' for a, b in zip(nt['q05'], nt['q95'])],
                      'p': nt['p_two_sided'], 'Holm p': nt['p_two_sided_holm']})
    L.export_table(t, 'm1_natural_transfer', label='tab:m1_natural_transfer',
                   caption=('Natural-use transfer of each chiral feature: free-choice trials whose sign points to the '
                            'reported hand (direction learned on the other persons), against the count expected from the '
                            'same persons\' instructed trials of that hand (Poisson-binomial; two-sided p, Holm over the '
                            'features).'),
                   column_formats={'Expected': '.1f', 'p': '.2g', 'Holm p': '.2g'})
    rows = [r for c in nj.get('contributions', []) for r in c['rows']]
    if rows:
        cd = pd.DataFrame(rows)
        L.export_table(cd, 'm1_natural_contributions', label='tab:m1_natural_contributions',
                       caption=('Mean logit contribution of each component towards the reported hand on the free-choice '
                                'trials and on the same persons\' instructed trials of that hand; per-person difference '
                                'with a person-bootstrap 95\\,\\% CI, sign-flip p and Holm over the components.'),
                       column_formats={'natural_mean': '+.2f', 'primary_mean': '+.2f', 'diff': '+.2f', 'diff_lo': '+.2f',
                                       'diff_hi': '+.2f', 'p': '.2g', 'p_holm': '.2g', 'natural_share_positive': '.0%',
                                       'primary_share_positive': '.0%'})


def t_sensitivity() -> None:
    sj = L.read_json('m1_sensitivity')
    s = pd.DataFrame(sj['rows'])
    if not len(s):
        return
    mt = _models()
    t = pd.DataFrame({'Run': s['name'].map(dict(zip(mt['name'], mt['label']))), 'Anchor': s['anchor'],
                      'Dropped (median)': s['dropped_median'], 'dAcc (pts)': 100 * s['vs_base_acc_diff'],
                      'dAcc CI': [_ci(a, b) for a, b in zip(s['vs_base_acc_diff_lo'], s['vs_base_acc_diff_hi'])],
                      'dLL': s['vs_base_ll_diff'],
                      'dLL CI': [_ci(a, b, False) for a, b in zip(s['vs_base_ll_diff_lo'], s['vs_base_ll_diff_hi'])]})
    L.export_table(t, 'm1_sensitivity', label='tab:m1_sensitivity',
                   caption=('Sensitivity of the finalist anchors to suspected wrong-hand training rows: trained without the '
                            'swap-signature blocks, or with the docs/09 D6 screen inside every fold; evaluation unchanged '
                            '(paired against the anchor; Dropped: training rows removed per fold).'),
                   column_formats={'Dropped (median)': '.0f', 'dAcc (pts)': '+.1f', 'dLL': '+.3f'})


def t_grid() -> None:
    L.export_table(_rows('m1_grid'), 'm1_grid_edges', tex=False)


TABLES = {'m1_feature_catalogue': t_catalogue, 'm1_models': t_models, 'm1_rules': t_rules,
          'm1_rules_per_person': t_rules_per_person, 'm1_conditions': t_conditions, 'm1_calibration': t_calibration,
          'm1_k_aggregation': t_k_aggregation, 'm1_data_treatment': t_treatment, 'm1_failures': t_failures,
          'm1_coefficients': t_coefficients, 'm1_natural_transfer': t_natural, 'm1_sensitivity': t_sensitivity,
          'm1_grid_edges': t_grid}


def write(only: list[str] | None = None) -> None:
    for name, fn in TABLES.items():
        if selected(name, only):
            fn()
