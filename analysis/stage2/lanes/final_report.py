"""Lane final: tables (CSV + booktabs .tex), the numbers file, the summary and the finalists file, from saved files
only (tables/final/final_*.json written by the compute step). Every sentence that states a decision is generated
from those files."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from stage2 import report_export as RX
from stage2.pipeline import Lane, selected

L = Lane('final')
E = RX.numbers_entry
P = '.1%'


def _read(name: str) -> dict[str, Any]:
    from stage2.lanes.final_io import denull
    return denull(L.read_json(name))


def _ci(v: list[float] | tuple[float, float], fmt: str = 'pct') -> str:
    if v is None or any(x is None or (isinstance(x, float) and not math.isfinite(x)) for x in v):
        return '--'
    return f'[{100 * v[0]:.1f}, {100 * v[1]:.1f}]' if fmt == 'pct' else f'[{v[0]:.3f}, {v[1]:.3f}]'


def _p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return '--'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def _pts(v: float) -> str:
    return f'{100 * v:+.1f}'


def _final() -> tuple[str, dict[str, Any]]:
    dec = _read('final_decision')
    return dec['final_model'], dec


def _labels() -> dict[str, str]:
    return {r['key']: r['label'] for r in _read('final_master')['rows']}


# ───────────────────────────── tables ─────────────────────────────


def _f3(v: Any) -> str:
    return f'{v:.3f}' if v is not None and isinstance(v, (int, float)) and np.isfinite(v) else '--'


def table_master() -> pd.DataFrame:
    m = pd.DataFrame(_read('final_master')['rows']).sort_values('log_loss')
    for c in ('prod_log_loss', 'prod_cal_ece'):
        if c not in m.columns:
            m[c] = np.nan
    return pd.DataFrame({
        'model': m['label'], 'lane': m['lane'], 'family': m['family'], 'representation': m['representation'],
        'data treatment': m['treatment'],
        'accuracy': [f'{100 * a:.1f} {_ci(c)}' for a, c in zip(m['accuracy'], m['accuracy_ci'])],
        'AUC': [f'{a:.3f}' for a in m['auc']],
        'log loss': [f'{a:.3f} {_ci(c, "f")}' for a, c in zip(m['log_loss'], m['log_loss_ci'])],
        'ECE (null 95th pct.)': [f'{e:.3f} ({q:.3f})' + (' *' if cw else '')
                                 for e, q, cw in zip(m['cal_ece'], m['cal_ece_null_q95'], m['constant_within_fold'])],
        'after in-fold remedy: log loss / ECE': [f'{_f3(a)} / {_f3(b)}' for a, b in zip(m['prod_log_loss'],
                                                                                        m['prod_cal_ece'])],
        'natural use: right recall': [f'{100 * r:.0f} %' if np.isfinite(r) else '--' for r in m['nat_recall']],
        'k = 3': [f'{100 * k:.1f}' for k in m['k3']], 'configs': m['n_configs_tried'].astype(int),
        'nested': ['yes' if n else 'no' for n in m['nested']],
        'CPU s / LOSO': [f'{s:.0f}' if s is not None and np.isfinite(s) else '--' for s in m['loso_seconds']]})


def table_nested() -> pd.DataFrame:
    ns = _read('final_nested')
    lab = _labels()
    n, nv, o = ns['nested'], ns['naive'], ns['optimism']
    return pd.DataFrame([
        {'estimate': 'nested cross-lane selection (selection-aware)', 'accuracy': f'{100 * n["accuracy"]:.1f} {_ci(n["accuracy_ci"])}',
         'AUC': f'{n["auc"]:.3f} {_ci(n["auc_ci"], "f")}', 'log loss': f'{n["log_loss"]:.3f} {_ci(n["log_loss_ci"], "f")}'},
        {'estimate': f'best on the outer folds: {lab.get(ns["naive_key"], ns["naive_key"])} (exploratory)',
         'accuracy': f'{100 * nv["accuracy"]:.1f} {_ci(nv["accuracy_ci"])}', 'AUC': f'{nv["auc"]:.3f}',
         'log loss': f'{nv["log_loss"]:.3f} {_ci(nv["log_loss_ci"], "f")}'},
        {'estimate': 'optimism (naive - nested)', 'accuracy': f'{_pts(o["accuracy"]["diff"])} {_ci(o["accuracy"]["ci"])}',
         'AUC': f'{o["auc"]["diff"]:+.3f} {_ci(o["auc"]["ci"], "f")}',
         'log loss': f'{o["log_loss"]["diff"]:+.3f} {_ci(o["log_loss"]["ci"], "f")}'}])


def table_selection() -> pd.DataFrame:
    ns = _read('final_nested')
    lab = _labels()
    rows = [{'candidate': lab.get(k, k), 'folds chosen': int(v),
             'mean inner log loss': f'{ns["inner_logloss_mean"].get(k, float("nan")):.3f}'}
            for k, v in sorted(ns['frequency'].items(), key=lambda kv: ns['inner_logloss_mean'].get(kv[0], 9))]
    return pd.DataFrame(rows)


def table_paired() -> pd.DataFrame:
    pr = pd.DataFrame(_read('final_paired')['rows'])
    lab = _labels()
    rows = []
    for k, g in pr.groupby('key', sort=False):
        row = {'model': lab.get(k, k)}
        for fam, tag in (('vs_best', 'vs best'), ('vs_reference', 'vs reference')):
            r = g[g['family'] == fam]
            if len(r):
                r = r.iloc[0]
                row[f'{tag}: accuracy'] = f'{_pts(r["d_acc"])} {_ci(r["d_acc_ci"])} (Holm {_p(r["holm_acc"])})'
                row[f'{tag}: log loss'] = f'{r["d_ll"]:+.3f} {_ci(r["d_ll_ci"], "f")} (Holm {_p(r["holm_ll"])})'
            else:
                row[f'{tag}: accuracy'] = row[f'{tag}: log loss'] = '(anchor)'
        rows.append(row)
    return pd.DataFrame(rows)


def table_input_class() -> pd.DataFrame:
    rows = _read('final_input_class')['rows']
    lab = _labels()
    return pd.DataFrame([{'comparison (A - B)': f'{lab.get(r["a"], r["a"])} - {lab.get(r["b"], r["b"])}',
                          'isolates': r['what'],
                          'accuracy': f'{_pts(r["d_acc"])} {_ci(r["d_acc_ci"])} (Holm {_p(r["holm_acc"])})',
                          'log loss': f'{r["d_ll"]:+.3f} {_ci(r["d_ll_ci"], "f")} (Holm {_p(r["holm_ll"])})'}
                         for r in rows])


def table_robustness() -> pd.DataFrame:
    rob = _read('final_robustness')['rows']
    out = []
    for r in rob:
        g, f, pl, n = r.get('grip') or {}, r.get('flip') or {}, r.get('platform') or {}, r.get('noise') or {}
        out.append({'model': r['label'],
                    'rotated grips, upright-only training': (f'{100 * g["acc_upright_only"]:.1f} ({_pts(g["d_acc"])}, Holm '
                                                             f'{_p(g.get("holm_acc"))})') if g else '--',
                    'screen away': (f'{100 * f["acc_flipped"]:.1f}' + (' (invariant)' if f.get('invariant_inputs') else ''))
                    if f else '--',
                    'screen away + check': f'{100 * f["acc_flipped_remedy"]:.1f}' if f else '--',
                    'check cost (recorded)': _pts(f['remedy_cost']) if f else '--',
                    'other platform only': (f'{_pts(pl["d_acc"])} (Holm {_p(pl.get("holm_acc"))})'
                                            + (' [p1]' if pl.get('from_lane_p1') else '')) if pl else '--',
                    'without suspect blocks': f'{_pts(n["d_acc"])} (Holm {_p(n.get("holm_acc"))})' if n else '--'})
    return pd.DataFrame(out)


def table_natural() -> pd.DataFrame:
    rows = _read('final_natural')['rows']
    return pd.DataFrame([{'model': r['label'], 'correct (right recall)': f'{r["observed"]:g} of {r["n"]}',
                          'expected (person-matched)': f'{r["expected"]:.1f} [{r["range95"][0]}, {r["range95"][1]}]',
                          'p (lower tail)': _p(r['p_lower']), 'Holm': _p(r['holm_lower']),
                          'shortfall': 'yes' if r['shortfall'] else 'no'} for r in rows])


def table_decision() -> pd.DataFrame:
    dec = _read('final_decision')
    rows = []
    for r in sorted(dec['rows'], key=lambda r: r['prod_log_loss']):
        rows.append({'model': r['label'], 'accuracy': f'{100 * r["prod_accuracy"]:.1f}',
                     'log loss': f'{r["prod_log_loss"]:.3f}',
                     'worse than best (Holm)': 'yes' if r['worse_than_best'] else 'no',
                     'ECE before / after remedy': (f'{r["ece"]:.3f} / {r["ece_after"]:.3f}'
                                                   + (' *' if r['constant_within_fold'] else '')),
                     'calibration after remedy': 'miscalibrated' if r['miscalibrated_after'] else 'consistent',
                     'worst platform': (f'{r["worst_platform"].get("level", "")} {100 * r["worst_platform"]["accuracy"]:.1f}'
                                        if r['worst_platform'] else '--'),
                     'worst variant (>= 5 persons)': (f'{r["worst_variant"].get("level", "")} '
                                                      f'{100 * r["worst_variant"]["accuracy"]:.0f}' if r['worst_variant'] else '--'),
                     'grip transfer fails': 'yes' if r['grip_fails'] else 'no',
                     'screen away (raw / with check)': (f'{100 * r["flip_acc"]:.0f} / {100 * r["flip_acc_remedy"]:.0f}'
                                                        if np.isfinite(r['flip_acc']) else '--'),
                     'natural-use shortfall': 'yes' if r['nat_shortfall'] else 'no',
                     'bounds +-200 ms (points)': (f'{r["segmentation"]["d"]:+.1f}' + (' [p1]' if str(
                         r['segmentation'].get('source', '')).startswith('p1') else ''))
                     if isinstance(r.get('segmentation'), dict) and r['segmentation'] else '--',
                     'robust': 'yes' if r['robust'] else 'no', 'port (1 easy - 6 hard)': r['port_rank'],
                     'nested picks': r['nested_selected'],
                     'decision': ('FINAL' if r['key'] == dec['final_model'] else
                                  ('recommended' if r['key'] == dec['recommended'] else ''))})
    return pd.DataFrame(rows)


def table_calibration() -> pd.DataFrame:
    cal = _read('final_calibration')
    rows = []
    for m, e in cal['methods'].items():
        o = e['overall']
        d = e.get('d_log_loss')
        rows.append({'remedy': m + (' (shipped)' if m == cal['chosen'] else ''),
                     'ECE (null 95 % band)': f'{o["ece"]:.3f} (up to {o["ece_null_hi"]:.3f})',
                     'slope': f'{o["slope"]:.2f} [{o["slope_ci"][0]:.2f}, {o["slope_ci"][1]:.2f}]',
                     'intercept': f'{o["intercept"]:+.2f} [{o["intercept_ci"][0]:+.2f}, {o["intercept_ci"][1]:+.2f}]',
                     'log loss': f'{o["log_loss"]:.3f}', 'Brier': f'{o["brier"]:.3f}',
                     'log loss vs none': (f'{d["diff"]:+.4f} {_ci(d["ci"], "f")} (Holm {_p(d.get("holm"))})' if d else '--'),
                     'ECE vs none (basic CI)': (f'{e["d_ece"]["diff"]:+.3f} {_ci(e["d_ece"]["ci_basic"], "f")}'
                                                if e.get('d_ece') else '--'),
                     'Android ECE / slope': f'{e["platform"]["android"]["ece"]:.3f} / {e["platform"]["android"]["slope"]:.2f}',
                     'iOS ECE / slope': f'{e["platform"]["ios"]["ece"]:.3f} / {e["platform"]["ios"]["slope"]:.2f}',
                     'verdict': 'miscalibrated' if o['miscalibrated'] else 'consistent with calibration'})
    return pd.DataFrame(rows)


def table_aggregation() -> pd.DataFrame:
    ag = _read('final_aggregation')
    names = {'mean_p': 'mean P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits',
             'sum_tau': 'sum / tau_k (nested)'}
    rows = []
    for k, per in ag['rules'].items():
        for rule, v in per.items():
            rows.append({'k': int(k), 'aggregate': names[rule], 'accuracy': f'{100 * v["accuracy"]:.1f}',
                         'log loss': f'{v["log_loss"]:.3f} {_ci(v["log_loss_ci"], "f")}',
                         'confidence - accuracy': f'{v["over_confidence"]:+.3f} {_ci(v["over_ci"], "f")}',
                         'ECE': f'{v["ece"]:.3f}', 'calibrated': 'yes' if v['calibrated'] else 'no'})
    return pd.DataFrame(rows)


def table_breakdown() -> pd.DataFrame:
    br = _read('final_breakdown')
    rows = []
    for fac, key in (('platform', 'platform'), ('grip x speed', 'grip_speed'), ('variant', 'variant'),
                     ('handedness', 'handedness')):
        for r in br[key]:
            lev = ', '.join(str(r[c]) for c in r if c in ('platform', 'orientation', 'speed', 'variant', 'handedness_self'))
            ci = _ci(r['ci']) if 'ci' in r else ('persons: ' + ', '.join(f'{p} {100 * v:.0f}' for p, v in r['per_person'].items()))
            rows.append({'factor': fac, 'level': lev, 'trials': r['n'], 'persons': r['persons'],
                         'accuracy': f'{100 * r["accuracy"]:.1f}', '95 % CI (or per person)': ci})
    return pd.DataFrame(rows)


def table_learning() -> pd.DataFrame:
    lc = {k: v for k, v in _read('final_learning').items() if isinstance(v, dict) and 'summary' in v}
    lab = _labels()
    rows = []
    for key, v in lc.items():
        for s in v['summary']:
            rows.append({'model': lab.get(key, key), 'training persons': int(s['n']), 'repetitions': int(s['reps']),
                         'accuracy (mean, min-max)': f'{100 * s["acc_mean"]:.1f} ({100 * s["acc_min"]:.1f}-{100 * s["acc_max"]:.1f})',
                         'log loss (mean)': f'{s["ll_mean"]:.3f}'})
    return pd.DataFrame(rows)


def table_variants() -> pd.DataFrame:
    rows = _read('final_variants')['rows']
    return pd.DataFrame([{'test-time variant': r['description'], 'accuracy': f'{100 * r["accuracy"]:.1f}',
                          'change': f'{_pts(r["d_acc"])} {_ci(r["d_acc_ci"])} (Holm {_p(r["holm_acc"])})',
                          'log loss change': f'{r["d_ll"]:+.3f} {_ci(r["d_ll_ci"], "f")}'} for r in rows])


def table_screen() -> pd.DataFrame:
    """The screen-away remedies of the final model on the recorded holds and on their screen-away twins."""
    fin, _ = _final()
    fl = next((r.get('flip') or {} for r in _read('final_robustness')['rows'] if r['key'] == fin), {})
    if not fl or fl.get('invariant_inputs'):
        return pd.DataFrame(columns=['remedy'])
    rem = _read('final_remedies')
    fa, fx = rem.get('flipaug_recorded'), rem.get('flipaug_twins')
    rows = [('model as trained, no check', fl['acc_recorded'], fl['ll_recorded'], fl['acc_flipped']),
            ('rest-pose check, then un-flip the window', fl['acc_recorded_remedy'], fl['ll_recorded_remedy'],
             fl['acc_flipped_remedy']),
            ('rest-pose check, then prompt (abstain: half credit)', fl['acc_recorded_abstain'],
             fl['ll_recorded_abstain'], fl['acc_flipped_abstain'])]
    if isinstance(fa, dict) and isinstance(fx, dict):
        rows.append(('screen-flip augmentation of the training rows', fa['acc_a'], fa['ll_a'], fx['acc_a']))
    return pd.DataFrame([{'remedy': r, 'recorded holds: accuracy': f'{100 * a:.1f}', 'recorded holds: log loss': f'{b:.3f}',
                          'screen-away twins: accuracy': f'{100 * c:.1f}'} for r, a, b, c in rows])


def table_model_card() -> pd.DataFrame:
    card = _read('final_model_card')
    return pd.DataFrame([{'item': k, 'value': v} for k, v in card['items']])


TABLES: dict[str, tuple[Any, str]] = {
    'final_master': (table_master, 'Candidates of the cross-lane selection as their lanes deliver them (leave-one-person-'
                     'out, 95 % person-bootstrap CIs). ECE with the 95th percentile of its perfect-calibration null '
                     '(*: P constant within each fold, so the pooled ECE mostly sorts the folds and is not comparable); '
                     'after in-fold remedy: the production form (remedy chosen inside every fold); natural use: '
                     'right-hand recall on the free-choice trials; configs: outer-fold choices in the candidate\'s lane.'),
    'final_nested': (table_nested, 'Selection-aware estimate of the cross-lane selection, the naive best-of-candidates '
                     'number and its optimism (paired person bootstrap).'),
    'final_selection': (table_selection, 'How often each candidate was chosen by the inner log loss across the outer folds.'),
    'final_paired': (table_paired, 'Paired comparisons on the same trials: mean per-person differences with 95 % CIs; '
                     'Holm within each family (against the best, against the reference).'),
    'final_input_class': (table_input_class, 'Input against model class (task 2b): models given the same channels.'),
    'final_robustness': (table_robustness, 'Robustness to unknown test conditions (accuracy in %, paired changes in '
                         'points, Holm over the candidates within each test).'),
    'final_natural': (table_natural, 'Person-matched natural-use test: observed correct free-choice trials (right-hand '
                      'recall) against the Poisson-binomial expectation from the same persons\' instructed trials.'),
    'final_decision': (table_decision, 'Decision table: every candidate in its production form (accuracy, log loss, ECE '
                       'before and after the in-fold remedy), robustness, portability and nested selection frequency '
                       '(*: P constant within each fold, ECE not comparable).'),
    'final_calibration': (table_calibration, 'Calibration remedies of the final model, fitted on the training persons\' '
                          'inner cross-fitted predictions; Holm over the remedies.'),
    'final_aggregation': (table_aggregation, 'Multi-attempt aggregates of the final model (tau_k fitted on inner '
                          'cross-fitted predictions of the training persons in every outer fold).'),
    'final_breakdown': (table_breakdown, 'Final model by stratum (stratification only; a CI needs at least 5 persons).'),
    'final_learning': (table_learning, 'Learning curve: training persons drawn at random inside every outer fold.'),
    'final_variants': (table_variants, 'Final model on test windows with segmentation errors or a causal filter.'),
    'final_screen': (table_screen, 'Screen-away remedies of the final model: accuracy and log loss on the recorded '
                     'holds and accuracy on their screen-away twins (synthesised with standardize.screen_flip_channels); '
                     'the rest-pose check is the model-free rule of lane m4.'),
    'final_model_card': (table_model_card, 'Model card of the exported final model.'),
}


def tables(only: list[str] | None = None) -> None:
    for name, (fn, cap) in TABLES.items():
        if not selected(name, only):
            continue
        try:
            df = fn()
        except FileNotFoundError as e:
            print(f'final: table {name} skipped ({e})', flush=True)
            continue
        if df.shape[1] == 0:
            print(f'final: table {name} skipped (no rows: its experiments have not run)', flush=True)
            continue
        L.export_table(df, name, caption=cap)
    for name, src, key in (('final_overview_rows', 'final_overview', 'rows'),
                           ('final_reproduction_rows', 'final_reproduction', 'rows')):
        try:
            L.export_table(pd.DataFrame(_read(src)[key]), name, tex=False)
        except FileNotFoundError:
            pass
    try:
        sel = pd.DataFrame(_read('final_nested')['selection'])
        L.export_table(sel, 'final_selection_rows', tex=False)
    except FileNotFoundError:
        pass
