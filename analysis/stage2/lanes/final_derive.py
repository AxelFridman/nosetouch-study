"""Lane final: every derived analysis, computed from SAVED results only (no model is evaluated here; the few refits
below are decompositions of already evaluated models, never new scores). Writes ``tables/final/final_*.json`` that
the figures, tables, numbers and the summary read.

Contents: master comparison of the candidates, overview of every registry experiment, the nested selection
(selection-aware estimate, naive best, optimism, selection stability), paired comparisons (Holm within families),
input vs model class (task 2b), robustness (grip, screen flip with remedies, platform, label noise, conditions),
the person-matched natural-use test, the decision table, and the analyses of the final model (calibration remedies,
multi-attempt aggregation, breakdowns, segmentation and filter sensitivity, learning curve, the prior).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import final_calibration as FCal
from stage2.lanes import final_candidates as FC
from stage2.lanes import final_experiments as FE
from stage2.lanes import final_robust as FR

ALPHA = 0.05
#: strata need at least this many persons for a CI (fewer: the persons' values are shown individually)
MIN_PERSONS_CI = 5
#: the calibration remedies ranked by their number of fitted parameters (the tie-break of the shipping rule)
REMEDY_PARAMS = {'none': 0, 'temperature': 1, 'platt': 2, 'beta': 3, 'isotonic': None}


# ───────────────────────────── helpers ─────────────────────────────


@lru_cache(maxsize=None)
def res(name: str) -> cv.ExperimentResult:
    return cv.load_result(name)


def exists(name: str) -> bool:
    return (C.RES_DIR / f'{name}.json').exists()


def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p values (step-down, monotone), NaN kept as NaN."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.size, np.nan)
    if not ok.any():
        return out.tolist()
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind='stable')]
    m = order.size
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out.tolist()


def fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def pc(a: pd.DataFrame | cv.ExperimentResult, b: pd.DataFrame | cv.ExperimentResult) -> dict[str, Any]:
    out = cv.paired_compare(a, b)
    out.pop('per_person', None)
    return out


def trial_table(d: Any) -> pd.DataFrame:
    """Conditions of the primary trials (stratification only, after prediction) and the person's handedness."""
    from stage2.data import load_tables
    t = d.trials[['trial_id', 'person_id', 'platform', 'orientation', 'speed', 'variant', 'session_label',
                  'block_index', 'y_right']].copy()
    persons = load_tables()['persons']
    t = t.merge(persons[['person_id', 'handedness_self']], on='person_id', how='left')
    return t


def preds(name: str, extra: bool = False) -> pd.DataFrame:
    r = res(name)
    return (r.extra_predictions if extra else r.predictions).copy()


def nat_counts(r: cv.ExperimentResult) -> dict[str, Any]:
    e = r.extra_predictions
    if e is None or not len(e):
        return {'n': 0}
    c = cv.credit(e['y'].to_numpy(), e['p_right'].to_numpy())
    return {'n': int(len(e)), 'credit_sum': float(c.sum()), 'recall': float(c.mean()),
            'n_right': int(e['y'].sum()), 'answered': int(np.sum(~cv.is_tie(e['p_right'].to_numpy())))}


# ───────────────────────────── master table ─────────────────────────────


def calibration_summary(r: cv.ExperimentResult, n_boot: int = 1000) -> dict[str, Any]:
    p = r.predictions
    blk = FCal.calibration_block(p['y'].to_numpy(), p['p_right'].to_numpy(), p['person_id'].to_numpy(),
                                 n_boot=n_boot)
    return {k: blk[k] for k in ('ece', 'ece_null_q95', 'ece_null_hi', 'ece_p_value', 'ece_inside_band', 'slope',
                                'slope_ci', 'intercept', 'intercept_ci', 'miscalibrated', 'brier')}


def prod_name(key: str) -> str:
    """The production form of a candidate: its own pipeline plus the calibration remedy chosen inside every outer fold
    (``final_cal_<key>_auto``); every decision compares candidates in this form (the product is the calibrated P)."""
    return f'final_cal_{key}_auto'


def compare_name(c: FC.Candidate) -> str:
    """The result a decision uses for a candidate: its production form when it has been run, else as delivered."""
    return prod_name(c.key) if exists(prod_name(c.key)) else FE.base_result_name(c)


def constant_within_fold(r: cv.ExperimentResult) -> bool:
    """True for a predictor whose P(right) takes at most two values within every fold (a rule scoring its training hit
    rate): its pooled ECE and AUC mostly sort the folds and are not comparable with a fitted model's."""
    p = r.predictions
    return bool(p.groupby('fold')['p_right'].nunique().max() <= 2)


def remedy_counts(r: cv.ExperimentResult) -> dict[str, int]:
    """How often each remedy was chosen across the outer folds of an 'auto' result."""
    out: dict[str, int] = {}
    for s in r.predictions.groupby('fold')['params'].first():
        m = json.loads(s).get('chosen', 'none')
        out[m] = out.get(m, 0) + 1
    return out


def master(cands: dict[str, FC.Candidate], repro: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for k, c in cands.items():
        name = FE.base_result_name(c)
        if not exists(name):
            continue
        r = res(name)
        kc = r.k_curve.set_index('k') if len(r.k_curve) else pd.DataFrame()
        nat = nat_counts(r)
        cost = (repro.get(k) or {}).get('loso_seconds') or r.info.get('wall_seconds')
        row = {'key': k, 'result': name, **FC.describe(c),
               'accuracy': r.metrics['accuracy'], 'accuracy_ci': list(r.ci.get('accuracy', (math.nan,) * 2)),
               'auc': r.metrics['roc_auc'], 'auc_ci': list(r.ci.get('roc_auc', (math.nan,) * 2)),
               'log_loss': r.metrics['log_loss'], 'log_loss_ci': list(r.ci.get('log_loss', (math.nan,) * 2)),
               'brier': r.metrics['brier'], 'balanced_accuracy': r.metrics['balanced_accuracy'],
               'nat_n': nat.get('n', 0), 'nat_recall': nat.get('recall', math.nan),
               'nat_credit': nat.get('credit_sum', math.nan),
               'k3': float(kc.loc[3, 'accuracy']) if 3 in kc.index else math.nan,
               'k3_ci': [float(kc.loc[3, 'lo']), float(kc.loc[3, 'hi'])] if 3 in kc.index else [math.nan] * 2,
               'k_persons': int(kc.loc[3, 'n_persons']) if 3 in kc.index else 0,
               'leak_check': r.leak_check, 'loso_seconds': cost,
               'reproduced': (repro.get(k) or {}).get('reproduced'),
               'max_abs_dp': (repro.get(k) or {}).get('max_abs_dp_primary'),
               'person_acc_median': r.metrics.get('person_acc_median'),
               'n_persons_below_80': r.metrics.get('n_persons_below_80'),
               'constant_within_fold': constant_within_fold(r)}
        if c.source in FC.NOT_TUNED_IN_FOLD:
            row['nested'] = False
        row['result_n_configs_tried'] = r.info.get('n_configs_tried')   # as the saved result records it
        row.update({f'cal_{a}': b for a, b in calibration_summary(r).items()})
        pn = prod_name(k)
        if exists(pn):
            q = res(pn)
            row.update({'prod_result': pn, 'prod_accuracy': q.metrics['accuracy'],
                        'prod_accuracy_ci': list(q.ci['accuracy']), 'prod_auc': q.metrics['roc_auc'],
                        'prod_log_loss': q.metrics['log_loss'], 'prod_log_loss_ci': list(q.ci['log_loss']),
                        'prod_brier': q.metrics['brier'], 'prod_nat_recall': nat_counts(q).get('recall', math.nan),
                        'prod_remedies': remedy_counts(q), 'prod_leak_check': q.leak_check})
            row.update({f'prod_cal_{a}': b for a, b in calibration_summary(q).items()})
        rows.append(row)
    return rows


#: the registries of the overview: the reference, the model lanes (their unit is a hand prediction) and this lane's
#: own models; other synthesis lanes (secondary targets, demographics) and this lane's stress tests are left out
OVERVIEW_LANES: tuple[str, ...] = ('reference',) + FC.FINALIST_LANES + ('m4', 'final')


def registry_overview() -> list[dict[str, Any]]:
    """Every experiment of the model lanes' registries (and the core registry): accuracy, log loss, configurations;
    of this lane only its own models (the new candidates and the nested selection), not its stress tests."""
    rows = []
    files = [('reference', C.RES_DIR / 'registry.csv')] + [(p.stem.replace('registry_', ''), p)
                                                            for p in sorted(C.RES_DIR.glob('registry_*.csv'))
                                                            if p.stem != 'registry_all']
    own = {s.name for s in FE.plan(FC.candidates(), None) if s.kind in ('candidate', 'nested')}
    for lane, p in files:
        if not p.exists() or lane not in OVERVIEW_LANES:
            continue
        t = pd.read_csv(p)
        if lane == 'final':
            t = t[t['name'].isin(own)]

        def num(v: Any) -> float | None:            # JSON has no NaN: a missing value is null
            return None if v is None or pd.isna(v) else float(v)
        for _, r in t.iterrows():
            k = num(r.get('n_configs_tried'))
            rows.append({'lane': lane, 'name': r['name'], 'accuracy': num(r.get('accuracy')),
                         'log_loss': num(r.get('log_loss')), 'n_configs_tried': None if k is None else int(k),
                         'exploratory': bool((k or 1) > 1),
                         'unit': 'recording (cue-free)' if lane == 'm4' else 'trial',
                         'coverage': num(r.get('coverage')) if 'coverage' in t.columns else None})
    return rows


# ───────────────────────────── nested selection ─────────────────────────────


def nested(cands: dict[str, FC.Candidate], master_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from stage2.lanes.final_nested import selection_table
    if not exists('final_nested_select'):
        return {}
    r = res('final_nested_select')
    sel = selection_table(r)
    freq = sel['chosen'].value_counts().reindex(list(cands), fill_value=0)
    m = pd.DataFrame(master_rows).set_index('key')
    naive_key = str(m['log_loss'].idxmin())               # best on the outer folds (exploratory)
    naive = res(m.loc[naive_key, 'result'])
    cmp_ = pc(naive, r)                                   # naive - nested
    # inner log loss: mean over the outer folds of each candidate's inner (training-person) log loss
    inner = {k: float(sel[f'inner_ll:{k}'].mean()) for k in cands if f'inner_ll:{k}' in sel.columns}
    # bootstrap of the optimism in log loss and AUC on the same person resamples
    pn = naive.predictions.set_index('trial_id')
    pr = r.predictions.set_index('trial_id').reindex(pn.index)
    y = pn['y'].to_numpy()
    persons = pn['person_id'].to_numpy()
    boots = {'accuracy': [], 'log_loss': [], 'auc': []}
    for idx in FCal.person_resamples(persons, C.N_BOOT):
        a, b = pn['p_right'].to_numpy()[idx], pr['p_right'].to_numpy()[idx]
        boots['accuracy'].append(float(np.mean(cv.credit(y[idx], a)) - np.mean(cv.credit(y[idx], b))))
        boots['log_loss'].append(FCal.log_loss(y[idx], a) - FCal.log_loss(y[idx], b))
        boots['auc'].append(cv.roc_auc(y[idx], a) - cv.roc_auc(y[idx], b))
    opt = {k: {'diff': float(v0), 'ci': [float(np.nanquantile(boots[k], 0.025)), float(np.nanquantile(boots[k], 0.975))]}
           for k, v0 in (('accuracy', naive.metrics['accuracy'] - r.metrics['accuracy']),
                         ('log_loss', naive.metrics['log_loss'] - r.metrics['log_loss']),
                         ('auc', naive.metrics['roc_auc'] - r.metrics['roc_auc']))}
    return {'nested': {'accuracy': r.metrics['accuracy'], 'accuracy_ci': list(r.ci['accuracy']),
                       'auc': r.metrics['roc_auc'], 'auc_ci': list(r.ci['roc_auc']),
                       'log_loss': r.metrics['log_loss'], 'log_loss_ci': list(r.ci['log_loss']),
                       'ece': r.metrics['ece'], 'leak_check': r.leak_check,
                       'nat': nat_counts(r), 'k3': float(r.k_curve.set_index('k').loc[3, 'accuracy'])},
            'naive_key': naive_key, 'naive_result': str(m.loc[naive_key, 'result']),
            'naive': {'accuracy': naive.metrics['accuracy'], 'accuracy_ci': list(naive.ci['accuracy']),
                      'auc': naive.metrics['roc_auc'], 'log_loss': naive.metrics['log_loss'],
                      'log_loss_ci': list(naive.ci['log_loss'])},
            'n_candidates': int(len(cands)), 'optimism': opt, 'paired_naive_minus_nested': cmp_,
            'selection': sel.to_dict(orient='records'), 'frequency': {k: int(v) for k, v in freq.items()},
            'inner_logloss_mean': inner}


# ───────────────────────────── paired comparisons ─────────────────────────────


def identical(a: cv.ExperimentResult, b: cv.ExperimentResult, d_ll: float, tol: float = 1e-3) -> bool:
    """Two prediction sets that differ by less than solver precision: |log-loss difference| below ``tol`` and no
    trial changing its answer. Such a pair is 'identical', never 'significant'."""
    pa = a.predictions.set_index('trial_id')['p_right']
    pb = b.predictions.set_index('trial_id')['p_right'].reindex(pa.index)
    same_answers = bool(np.all(np.sign(pa.to_numpy() - 0.5) == np.sign(pb.to_numpy() - 0.5)))
    return bool(abs(d_ll) < tol and same_answers)


def paired_families(cands: dict[str, FC.Candidate], master_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every candidate against the best and against the reference, both in their production forms (own pipeline plus
    the in-fold remedy; ``compare_name``), on the same trials; Holm within each family."""
    m = pd.DataFrame(master_rows).set_index('key')
    comp = {k: compare_name(cands[k]) for k in m.index}
    ll = {k: res(comp[k]).metrics['log_loss'] for k in m.index}
    best = min(ll, key=lambda k: (ll[k], list(m.index).index(k)))
    rows = []
    for anchor, fam in ((best, 'vs_best'), ('ref', 'vs_reference')):
        for k in m.index:
            if k == anchor:
                continue
            ra, rb = res(comp[k]), res(comp[anchor])
            out = pc(ra, rb)
            rows.append({'family': fam, 'key': k, 'anchor': anchor, 'result': comp[k], 'anchor_result': comp[anchor],
                         'd_acc': out['mean_person_acc_diff'],
                         'd_acc_ci': list(out['mean_person_acc_diff_ci95']), 'p_acc': out['p_sign_flip_acc'],
                         'pooled_d_acc': out['pooled_acc_diff'],
                         'd_ll': out['mean_person_logloss_diff'], 'd_ll_ci': list(out['mean_person_logloss_diff_ci95']),
                         'p_ll': out['p_sign_flip_logloss'], 'n_persons': out['n_persons'],
                         'identical': identical(ra, rb, out['mean_person_logloss_diff'])})
    df = pd.DataFrame(rows)
    for fam in df['family'].unique():
        sel = df['family'] == fam
        df.loc[sel, 'holm_acc'] = holm(df.loc[sel, 'p_acc'].tolist())
        df.loc[sel, 'holm_ll'] = holm(df.loc[sel, 'p_ll'].tolist())
    df['sig_acc'] = (df['holm_acc'] < ALPHA) & ~df['identical']
    df['sig_ll'] = (df['holm_ll'] < ALPHA) & ~df['identical']
    return {'best': best, 'rows': df.to_dict(orient='records'), 'form': 'production',
            'families': {'vs_best': f'every candidate against the lowest log loss in production form ({best}); Holm '
                                    'over the family',
                         'vs_reference': 'every candidate against the reference floor (production forms); Holm over the '
                                         'family'}}


#: task 2b: (comparison, A, B, what it isolates); A - B on the same trials
INPUT_CLASS: tuple[tuple[str, str, str, str], ...] = (
    ('cnn_vs_m2_same12', 'm3_deep', 'm2_body12_lr', 'model class on the same 12 channels (CNN vs m2 features + LR)'),
    ('cnn_vs_lrtc_same', 'm3_deep', 'lr_body_tc', 'CNN vs a 32-feature logistic on the odd channels of its input'),
    ('m2_body12_vs_lrtc', 'm2_body12_lr', 'lr_body_tc', 'rich features vs time-binned odd channels (same frames)'),
    ('m2_all21_vs_m2', 'm2_all21_lr', 'm2_lr', 'adding the body-frame channels to lane m2\'s logistic'),
    ('m2_all21_vs_cnn', 'm2_all21_lr', 'm3_deep', 'm2 logistic with 21 channels vs the CNN'),
    ('lrtc_vs_phys6', 'lr_body_tc', 'm1_phys6', 'time courses vs 6 physics features (same components)'),
)


def input_vs_class(cands: dict[str, FC.Candidate]) -> list[dict[str, Any]]:
    rows = []
    for name, a, b, what in INPUT_CLASS:
        if a not in cands or b not in cands:
            continue
        na, nb = FE.base_result_name(cands[a]), FE.base_result_name(cands[b])
        if not (exists(na) and exists(nb)):
            continue
        out = pc(res(na), res(nb))
        rows.append({'comparison': name, 'a': a, 'b': b, 'what': what, 'd_acc': out['mean_person_acc_diff'],
                     'd_acc_ci': list(out['mean_person_acc_diff_ci95']), 'p_acc': out['p_sign_flip_acc'],
                     'd_ll': out['mean_person_logloss_diff'], 'd_ll_ci': list(out['mean_person_logloss_diff_ci95']),
                     'p_ll': out['p_sign_flip_logloss'], 'acc_a': res(na).metrics['accuracy'],
                     'acc_b': res(nb).metrics['accuracy']})
    if rows:
        ha, hl = holm([r['p_acc'] for r in rows]), holm([r['p_ll'] for r in rows])
        for r, x, z in zip(rows, ha, hl):
            r['holm_acc'], r['holm_ll'] = x, z
    return rows


# ───────────────────────────── robustness ─────────────────────────────


def _subset(df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return df[df['trial_id'].astype(str).isin(ids)].reset_index(drop=True)


def _acc(df: pd.DataFrame) -> float:
    return float(np.mean(cv.credit(df['y'].to_numpy(), df['p_right'].to_numpy()))) if len(df) else math.nan


def worst_cells(p: pd.DataFrame, tt: pd.DataFrame) -> dict[str, Any]:
    j = p.merge(tt.drop(columns=['person_id', 'y_right']), on='trial_id', how='left')
    j['credit'] = cv.credit(j['y'].to_numpy(), j['p_right'].to_numpy())
    out = {}
    for f in ('platform', 'orientation', 'speed', 'variant'):
        g = j.groupby(f).agg(acc=('credit', 'mean'), n=('credit', 'size'), persons=('person_id', 'nunique'))
        g = g[g['persons'] >= MIN_PERSONS_CI]
        if len(g):
            w = g['acc'].idxmin()
            out[f] = {'level': str(w), 'accuracy': float(g.loc[w, 'acc']), 'n': int(g.loc[w, 'n']),
                      'persons': int(g.loc[w, 'persons'])}
    return out


def flip_inputs_unchanged(c: FC.Candidate, d: Any) -> float:
    """max |input(screen-flipped data) - input(data)|: 0 for an input built from gravity-frame channels only."""
    keys = FC.needed_inputs({c.key: c})
    a = FC.build_inputs(d, keys)[keys[0]]
    b = FC.build_inputs(FR.flipped(d), keys)[keys[0]]

    def arr(x: Any) -> np.ndarray:
        if isinstance(x, dict):
            return np.concatenate([np.asarray(v, dtype=float).reshape(len(v), -1) for v in x.values()], axis=1)
        return np.asarray(x, dtype=float).reshape(len(x), -1)
    return float(np.max(np.abs(arr(a) - arr(b))))


def robustness(cands: dict[str, FC.Candidate], d: Any, tt: pd.DataFrame) -> list[dict[str, Any]]:
    rotated = set(tt.loc[tt['orientation'].astype(str) != 'upright_toward', 'trial_id'].astype(str))
    away_rec = pd.Series(FR.screen_away(d.fixed), index=d.trials['trial_id'].astype(str))
    away_flip = pd.Series(FR.screen_away(FR.flipped(d).fixed), index=d.trials['trial_id'].astype(str))
    rows = []
    for k, c in cands.items():
        base_name = FE.base_result_name(c)
        if not exists(base_name):
            continue
        base = preds(base_name)
        row: dict[str, Any] = {'key': k, 'label': c.label}
        row['worst'] = worst_cells(base, tt)
        # grip transfer
        gname = f'final_grip_{k}'
        if exists(gname):
            g = preds(gname)
            bs, gs = _subset(base, rotated), _subset(g, rotated)
            out = pc(gs, bs)
            row['grip'] = {'acc_all_grips': _acc(bs), 'acc_upright_only': _acc(gs), 'd_acc': out['mean_person_acc_diff'],
                           'd_acc_ci': list(out['mean_person_acc_diff_ci95']), 'p_acc': out['p_sign_flip_acc'],
                           'd_ll': out['mean_person_logloss_diff'], 'd_ll_ci': list(out['mean_person_logloss_diff_ci95']),
                           'p_ll': out['p_sign_flip_logloss'], 'n_trials': int(len(bs)), 'n_persons': out['n_persons'],
                           'acc_upright_trials_upright_only': _acc(_subset(g, set(tt['trial_id']) - rotated))}
        # screen flip
        fname = f'final_flip_{k}'
        if exists(fname) or FE.flip_invariant(c):
            f = preds(fname) if exists(fname) else base
            fb = f.set_index('trial_id').reindex(base['trial_id'])
            p_rec, p_flip = base['p_right'].to_numpy(), fb['p_right'].to_numpy()
            ids = base['trial_id'].astype(str)
            rem_rec = FR.detect_and_correct(p_rec, p_flip, away_rec.reindex(ids).to_numpy())
            rem_flip = FR.detect_and_correct(p_flip, p_rec, away_flip.reindex(ids).to_numpy())
            abst_flip = FR.detect_and_correct(p_flip, p_rec, away_flip.reindex(ids).to_numpy(), mode='abstain')
            abst_rec = FR.detect_and_correct(p_rec, p_flip, away_rec.reindex(ids).to_numpy(), mode='abstain')
            y = base['y'].to_numpy()
            frame = base[['trial_id', 'person_id', 'y']].copy()
            df_flip = frame.assign(p_right=p_flip)
            df_rem_flip = frame.assign(p_right=rem_flip)
            df_rem_rec = frame.assign(p_right=rem_rec)
            c1 = pc(df_flip, base)
            c2 = pc(df_rem_flip, df_flip)
            c3 = pc(df_rem_rec, base)
            row['flip'] = {'invariant_inputs': bool(FE.flip_invariant(c)), 'acc_recorded': _acc(base),
                           'acc_flipped': _acc(df_flip), 'd_acc': c1['mean_person_acc_diff'],
                           'd_acc_ci': list(c1['mean_person_acc_diff_ci95']), 'p_acc': c1['p_sign_flip_acc'],
                           'acc_flipped_remedy': _acc(df_rem_flip),
                           'acc_flipped_abstain': float(np.mean(cv.credit(y, abst_flip))),
                           'remedy_gain': c2['mean_person_acc_diff'], 'remedy_gain_ci': list(c2['mean_person_acc_diff_ci95']),
                           'p_remedy': c2['p_sign_flip_acc'], 'acc_recorded_remedy': _acc(df_rem_rec),
                           'remedy_cost': c3['mean_person_acc_diff'], 'remedy_cost_ci': list(c3['mean_person_acc_diff_ci95']),
                           'p_remedy_cost': c3['p_sign_flip_acc'],
                           'check_fires_recorded': float(away_rec.mean()), 'check_fires_flipped': float(away_flip.mean()),
                           'acc_recorded_abstain': float(np.mean(cv.credit(y, abst_rec))),
                           'll_recorded_abstain': FCal.log_loss(y, abst_rec),
                           'll_flipped': FCal.log_loss(y, p_flip), 'll_recorded': FCal.log_loss(y, p_rec),
                           'll_flipped_remedy': FCal.log_loss(y, rem_flip), 'll_recorded_remedy': FCal.log_loss(y, rem_rec),
                           'remedy_cost_ll': c3['mean_person_logloss_diff'],
                           'remedy_cost_ll_ci': list(c3['mean_person_logloss_diff_ci95']),
                           'p_remedy_cost_ll': c3['p_sign_flip_logloss']}
        # platform transfer
        pname = FE.P1_PLATFORM.get(k, f'final_plat_{k}')
        if pname and exists(pname):
            q = preds(pname)
            out = pc(q, base)
            per = {}
            for plat in ('android', 'ios'):
                ids_p = set(tt.loc[tt['platform'].astype(str) == plat, 'trial_id'].astype(str))
                per[plat] = {'pooled': _acc(_subset(base, ids_p)), 'other_only': _acc(_subset(q, ids_p))}
            row['platform'] = {'result': pname, 'd_acc': out['mean_person_acc_diff'],
                               'd_acc_ci': list(out['mean_person_acc_diff_ci95']), 'p_acc': out['p_sign_flip_acc'],
                               'd_ll': out['mean_person_logloss_diff'], 'p_ll': out['p_sign_flip_logloss'], 'per': per,
                               'from_lane_p1': k in FE.P1_PLATFORM}
        # label noise
        nname = f'final_noisy_{k}'
        if exists(nname):
            q = preds(nname)
            out = pc(q, base)
            row['noise'] = {'acc_clean_training': _acc(q), 'd_acc': out['mean_person_acc_diff'],
                            'd_acc_ci': list(out['mean_person_acc_diff_ci95']), 'p_acc': out['p_sign_flip_acc'],
                            'd_ll': out['mean_person_logloss_diff'], 'd_ll_ci': list(out['mean_person_logloss_diff_ci95']),
                            'p_ll': out['p_sign_flip_logloss']}
        rows.append(row)
    # Holm over the candidates within each robustness family
    for fam, key in (('grip', 'p_acc'), ('flip', 'p_acc'), ('platform', 'p_acc'), ('noise', 'p_acc')):
        idx = [i for i, r in enumerate(rows) if fam in r]
        adj = holm([rows[i][fam][key] for i in idx])
        for i, a in zip(idx, adj):
            rows[i][fam]['holm_acc'] = a
    return rows


# ───────────────────────────── natural use: the person-matched test ─────────────────────────────


def poisson_binomial(probs: np.ndarray) -> np.ndarray:
    """P(S = s), s = 0..n, of a sum of independent Bernoulli(probs)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for q in probs:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= (1 - q)
    return dist


def natural_test(cands: dict[str, FC.Candidate]) -> list[dict[str, Any]]:
    """Each free-choice trial's expected correctness = the same person's accuracy on the primary trials of the
    reported hand (every reported hand so far is the right hand); the observed count of correct free-choice trials
    (a tie counts one half) against the Poisson-binomial distribution of that expectation (lower tail: a shortfall)."""
    rows = []
    for k, c in cands.items():
        name = FE.base_result_name(c)
        if not exists(name):
            continue
        r = res(name)
        e = r.extra_predictions
        if e is None or not len(e):
            continue
        p = r.predictions
        p = p.assign(credit=cv.credit(p['y'].to_numpy(), p['p_right'].to_numpy()))
        exp = []
        for _, t in e.iterrows():
            same = p[(p['person_id'] == t['person_id']) & (p['y'] == t['y'])]
            exp.append(float(same['credit'].mean()) if len(same) else math.nan)
        exp = np.asarray(exp)
        obs_c = cv.credit(e['y'].to_numpy(), e['p_right'].to_numpy())
        ok = np.isfinite(exp)
        dist = poisson_binomial(exp[ok])
        obs = float(obs_c[ok].sum())
        lo = float(dist[: int(math.floor(obs + 1e-9)) + 1].sum())
        hi = float(dist[int(math.ceil(obs - 1e-9)):].sum())
        cdf = np.cumsum(dist)
        rng95 = [int(np.searchsorted(cdf, 0.025)), int(np.searchsorted(cdf, 0.975))]
        rows.append({'key': k, 'label': c.label, 'n': int(ok.sum()), 'observed': obs, 'expected': float(exp[ok].sum()),
                     'range95': rng95, 'p_lower': lo, 'p_two_sided': float(min(1.0, 2 * min(lo, hi))),
                     'recall': float(obs_c[ok].mean())})
    adj = holm([r['p_lower'] for r in rows])
    for r, a in zip(rows, adj):
        r['holm_lower'] = a
        r['shortfall'] = bool(a < ALPHA)
    return rows


#: the odd (hand-carrying) evidence of a model on the body-frame input, by component: channel groups of
#: ``m3_body.EXT_CHANNELS`` that the occlusion analysis sets to 0 (the mirror-neutral value of an odd channel)
COMPONENTS: dict[str, tuple[str, ...]] = {'yaw': ('yaw_rate', 'yaw_cum'), 'roll': ('roll_rate',),
                                          'lateral': ('acc_lat',)}


def component_transfer(key: str, cands: dict[str, FC.Candidate], d: Any, nat: Any) -> dict[str, Any]:
    """Which components of a model's evidence transfer to natural use (task 4b): for every trial, the logit
    contribution of each odd component towards the reported hand, by occlusion (the model's logit minus its logit
    with that component's channels set to 0 in the test window; exact for a linear model, indicative for a network).
    Free-choice trials against the same persons' instructed trials of the reported hand (every reported hand so far
    is the right one): per-person means, paired person bootstrap and sign-flip p, Holm over the components.

    The model is refitted on the harness's own outer folds (``final_nested.harness_folds``: the same training rows as
    its evaluated result, so a cached fit is reused) and only predicts modified copies of the held-out person's
    windows: a decomposition of an evaluated model, not a new score. Models on another input return {}."""
    import dataclasses
    from stage2.lanes import final_nested as FN
    from stage2.lanes.m3_body import EXT_CHANNELS, x_approach_body
    c = cands.get(key)
    if c is None or c.components or c.input != 'm3body':
        return {}
    X, Xn = x_approach_body(d), x_approach_body(nat)
    fp = c.make()
    folds = FN.harness_folds(X, d.trials)
    pid_d = d.trials['person_id'].astype(str).to_numpy()
    pid_n = nat.trials['person_id'].astype(str).to_numpy()
    y_d = d.trials['y_right'].astype(int).to_numpy()
    y_n = nat.trials['y_right'].astype(int).to_numpy()
    groups = list(COMPONENTS.items())
    rows = []
    for f in folds:
        ip, inn = np.flatnonzero(pid_d == f.test_person), np.flatnonzero(pid_n == f.test_person)
        base = np.concatenate([X['x'][ip], Xn['x'][inn]], axis=0)
        versions = [base]
        for _, chans in groups:
            v = base.copy()
            v[:, [EXT_CHANNELS.index(ch) for ch in chans], :] = 0.0
            versions.append(v)
        Xt = {'x': np.ascontiguousarray(np.concatenate(versions, axis=0), dtype=np.float32)}
        out = fp(dataclasses.replace(f, X_test=Xt))
        p = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
        z = FCal.logit(p).reshape(len(versions), base.shape[0])
        yy = np.concatenate([y_d[ip], y_n[inn]])
        sets = ['primary'] * ip.size + ['natural'] * inn.size
        sign = np.where(yy == 1, 1.0, -1.0)
        for j, (gname, _) in enumerate(groups):
            contrib = (z[0] - z[j + 1]) * sign
            rows += [{'person_id': f.test_person, 'set': s, 'component': gname, 'y': int(t),
                      'toward_reported': float(v)} for s, t, v in zip(sets, yy, contrib)]
        rows += [{'person_id': f.test_person, 'set': s, 'component': 'total', 'y': int(t), 'toward_reported': float(v)}
                 for s, t, v in zip(sets, yy, z[0] * sign)]
    df = pd.DataFrame(rows)
    out = []
    for gname in [g for g, _ in groups] + ['total']:
        g = df[df['component'] == gname]
        natp = g[g['set'] == 'natural'].groupby('person_id')['toward_reported'].mean()
        reported = set(g.loc[g['set'] == 'natural', 'y'].unique().tolist())
        prim = g[(g['set'] == 'primary') & (g['y'].isin(reported))].groupby('person_id')['toward_reported'].mean()
        common = natp.index.intersection(prim.index)
        diff = (natp[common] - prim[common]).to_numpy()
        rng = np.random.default_rng(C.SEED)
        boots = [float(np.mean(diff[rng.integers(0, diff.size, diff.size)])) for _ in range(C.N_BOOT)]
        nat_pos = g[g['set'] == 'natural']['toward_reported'].to_numpy()
        out.append({'component': gname, 'natural': float(natp[common].mean()),
                    'primary_same_persons': float(prim[common].mean()), 'diff': float(diff.mean()),
                    'diff_ci': [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                    'p': float(cv._sign_flip_p(diff)), 'n_persons': int(common.size),
                    'natural_pointing_right_way': int(np.sum(nat_pos > 0)), 'natural_n': int(nat_pos.size)})
    comp = [r for r in out if r['component'] != 'total']
    for r, a in zip(comp, holm([r['p'] for r in comp])):
        r['holm'] = a
    return {'key': key, 'method': 'occlusion (component channels set to 0 in the test window)', 'rows': out}


def lr_body_tc_components(d: Any, nat: Any) -> dict[str, Any]:
    """Logit contribution of each component (yaw, roll, lateral) of the body-frame logistic towards the reported hand
    on the free-choice trials against the same persons' primary trials of that hand (refits of the evaluated model on
    the harness's own folds: a decomposition, not a new score)."""
    from stage2.lanes import final_models as FM
    from stage2.lanes import final_nested as FN
    from stage2.lanes.m3_body import x_approach_body
    X, Xn = x_approach_body(d), x_approach_body(nat)
    store: dict[int, Any] = {}
    fp = FM.make_lr_body_tc(store=store)
    folds = FN.harness_folds(X, d.trials)
    groups = {'yaw': ('yaw_rate', 'yaw_cum'), 'roll': ('roll_rate',), 'lateral': ('acc_lat',)}
    rows = []
    persons_nat = nat.trials['person_id'].astype(str).to_numpy()
    for f in folds:
        fp(f)
        s, w = np.asarray(store[f.fold]['scale']), np.asarray(store[f.fold]['weights'])
        for which, data, mask in (('primary', X['x'], d.trials['person_id'].astype(str).to_numpy() == f.test_person),
                                  ('natural', Xn['x'], persons_nat == f.test_person)):
            if not mask.any():
                continue
            F = FM.tc_features(data[mask]) / s
            y = (d.trials['y_right'].to_numpy()[mask] if which == 'primary' else nat.trials['y_right'].to_numpy()[mask])
            sign = np.where(y == 1, 1.0, -1.0)
            for gname, chans in groups.items():
                cols = [i for i, cc in enumerate(FM.TC_CHANNELS) for b in range(FM.TC_BINS) if cc in chans]
                idx = [FM.TC_CHANNELS.index(cc) * FM.TC_BINS + b for cc in chans for b in range(FM.TC_BINS)]
                contrib = (F[:, idx] @ w[idx]) * sign
                for v, yy in zip(contrib, y):
                    rows.append({'person_id': f.test_person, 'set': which, 'component': gname, 'y': int(yy),
                                 'toward_reported': float(v)})
                del cols
    df = pd.DataFrame(rows)
    out = []
    for gname in groups:
        g = df[df['component'] == gname]
        nat_g = g[g['set'] == 'natural']
        prim = g[(g['set'] == 'primary') & (g['y'] == 1)].groupby('person_id')['toward_reported'].mean()
        natp = nat_g.groupby('person_id')['toward_reported'].mean()
        common = natp.index.intersection(prim.index)
        diff = (natp[common] - prim[common]).to_numpy()
        rng = np.random.default_rng(C.SEED)
        boots = [float(np.mean(diff[rng.integers(0, diff.size, diff.size)])) for _ in range(C.N_BOOT)]
        out.append({'component': gname, 'natural': float(natp[common].mean()), 'primary_same_persons': float(prim[common].mean()),
                    'diff': float(diff.mean()), 'diff_ci': [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                    'p': float(cv._sign_flip_p(diff)), 'n_persons': int(common.size)})
    adj = holm([r['p'] for r in out])
    for r, a in zip(out, adj):
        r['holm'] = a
    return {'rows': out}


__all__ = ['res', 'exists', 'holm', 'fmt_p', 'master', 'registry_overview', 'nested', 'paired_families',
           'input_vs_class', 'robustness', 'natural_test', 'component_transfer', 'COMPONENTS', 'lr_body_tc_components', 'trial_table', 'poisson_binomial',
           'worst_cells', 'flip_inputs_unchanged', 'json']
