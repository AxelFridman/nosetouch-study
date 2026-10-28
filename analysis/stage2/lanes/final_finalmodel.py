"""Lane final: the decision table and the analyses of the final model (tasks 4-7), from saved results only.

* :func:`decision_table` - every candidate scored on accuracy, log loss, calibration, robustness, natural-use
  transfer, simplicity, portability and nested-selection frequency, and the pre-specified rule that turns them
  into a recommendation (``recommend``); the analyst's choice lives in ``final_pipeline.FINAL_MODEL``.
* :func:`calibration` - the remedies of the final model (``final_cal_<method>`` results): the reporting standard
  per remedy, overall and per platform, paired log-loss / Brier / ECE differences against none (Holm over the
  remedies), and the shipping rule (fewest parameters among the remedies that lower the log loss after Holm).
* :func:`aggregation` - k = 1..5 attempts of one person and hand: mean P, mean logit, sum of logits and the sum
  divided by tau_k fitted in every outer fold on the training persons' INNER cross-fitted predictions.
* :func:`breakdowns`, :func:`variants`, :func:`learning_curve`, :func:`prior_shift`.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import final_calibration as FCal
from stage2.lanes import final_candidates as FC
from stage2.lanes import final_derive as FD
from stage2.lanes import final_experiments as FE

#: how hard each family is to port to JavaScript in a browser (1 = a dot product ... 5 = a deep ensemble) and what
#: has to be ported; a pre-specified scale of this lane (the decision prefers the lower rank among equals)
PORTABILITY: dict[str, tuple[int, str]] = {
    'ref': (1, 'dot product of 8 samples'), 'm1_rule': (1, 'one sign test'),
    'm1_phys6': (1, 'body-frame integrals + dot product of 6 features'),
    'p1_lr': (1, 'time bins + dot product of 16 features'),
    'lr_body_tc': (1, 'body-frame time bins + dot product of 32 features'),
    'm1_gam': (2, 'splines of 6 features'),
    'm2_lr': (3, '691 features (FFT, wavelets, correlations) + dot product'),
    'm2_body12_lr': (3, 'm2 feature families on 12 channels + dot product'),
    'm2_all21_lr': (3, 'm2 feature families on 21 channels + dot product'),
    'p1_gbm': (4, '238 summary features + 100 boosted trees + calibrator'),
    'm2_lgbm': (4, '691 features + boosted trees'), 'm2_stack': (4, '691 features + LR, RBF-SVM and trees + stacker'),
    'm3_mr': (4, 'MiniRocket transform (4000 features) + logistic'), 'p1_mr': (4, 'MiniRocket transform + ridge'),
    'm3_tf': (5, 'STFT + 2D CNN ensemble'), 'm3_deep': (5, '1D CNN ensemble (4 members)'),
    'ens_mean3': (6, 'three pipelines incl. the 1D CNN ensemble (harder than the CNN alone)'),
}


def _rob(rob: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return next((r for r in rob if r['key'] == key), {})


#: lane p1's panel model of a candidate (its segmentation-sensitivity rows, tables/p1/P1_03_segmentation.csv)
P1_PANEL_MODEL: dict[str, str] = {'p1_lr': 'lr', 'p1_gbm': 'gbm', 'p1_mr': 'mr'}
#: the test-time segmentation error the decision table reports (the largest bound jitter of both lanes)
SEG_VARIANT = 'bjit200'


def segmentation_sensitivity(key: str, cands: dict[str, FC.Candidate]) -> dict[str, Any]:
    """Accuracy change (points) when the test windows' bounds are jittered by U(-200, +200) ms (lane p1's variant
    arrays, the same canonical code path): this lane's run for a candidate that has one (the final model), else lane
    p1's panel run of the same model family; {} where neither exists."""
    name = f'final_var_{SEG_VARIANT}_{key}'
    if FD.exists(name):
        o = FD.pc(FD.res(name), FD.res(FE.base_result_name(cands[key])))
        return {'source': 'final', 'd': 100 * o['mean_person_acc_diff'],
                'ci': [100 * v for v in o['mean_person_acc_diff_ci95']], 'p': o['p_sign_flip_acc']}
    pm = P1_PANEL_MODEL.get(key)
    p = C.TAB_DIR / 'p1' / 'P1_03_segmentation.csv'
    if pm and p.exists():
        t = pd.read_csv(p)
        r = t[(t['model'] == pm) & (t['option'] == SEG_VARIANT)]
        if len(r):
            r = r.iloc[0]
            return {'source': 'p1 (panel model, uncalibrated)', 'd': float(r['d']), 'ci': [float(r['lo']), float(r['hi'])]}
    return {}


def decision_table(cands: dict[str, FC.Candidate], master_rows: list[dict[str, Any]], paired: dict[str, Any],
                   rob: list[dict[str, Any]], nat: list[dict[str, Any]], nested: dict[str, Any]) -> dict[str, Any]:
    m = {r['key']: r for r in master_rows}
    pr = pd.DataFrame(paired['rows'])
    best = paired['best']
    natd = {r['key']: r for r in nat}
    freq = (nested or {}).get('frequency', {})
    rows = []
    for k in m:
        r, rb = m[k], _rob(rob, k)
        vb = pr[(pr['family'] == 'vs_best') & (pr['key'] == k)]
        has_prod = 'prod_log_loss' in r and r.get('prod_log_loss') is not None and np.isfinite(r['prod_log_loss'])
        row = {'key': k, 'label': r['label'], 'lane': r['lane'], 'accuracy': r['accuracy'],
               'accuracy_ci': r['accuracy_ci'], 'log_loss': r['log_loss'], 'log_loss_ci': r['log_loss_ci'],
               'auc': r['auc'], 'ece': r['cal_ece'], 'ece_null_q95': r['cal_ece_null_q95'],
               'slope': r['cal_slope'], 'slope_ci': r['cal_slope_ci'], 'miscalibrated': r['cal_miscalibrated'],
               'prod_accuracy': r['prod_accuracy'] if has_prod else r['accuracy'],
               'prod_log_loss': r['prod_log_loss'] if has_prod else r['log_loss'],
               'prod_log_loss_ci': r['prod_log_loss_ci'] if has_prod else r['log_loss_ci'],
               'ece_after': r['prod_cal_ece'] if has_prod else r['cal_ece'],
               'ece_null_q95_after': r['prod_cal_ece_null_q95'] if has_prod else r['cal_ece_null_q95'],
               'slope_after': r['prod_cal_slope'] if has_prod else r['cal_slope'],
               'slope_ci_after': r['prod_cal_slope_ci'] if has_prod else r['cal_slope_ci'],
               'miscalibrated_after': r['prod_cal_miscalibrated'] if has_prod else r['cal_miscalibrated'],
               'remedies': r.get('prod_remedies') or {}, 'constant_within_fold': bool(r.get('constant_within_fold')),
               'k3': r['k3'], 'n_configs_tried': r['n_configs_tried'], 'nested': r['nested'],
               'vs_best_holm_acc': float(vb['holm_acc'].iloc[0]) if len(vb) else math.nan,
               'vs_best_holm_ll': float(vb['holm_ll'].iloc[0]) if len(vb) else math.nan,
               'vs_best_d_acc': float(vb['d_acc'].iloc[0]) if len(vb) else 0.0,
               'vs_best_d_ll': float(vb['d_ll'].iloc[0]) if len(vb) else 0.0,
               'worse_than_best': bool(len(vb) and ((vb['sig_acc'].iloc[0] and vb['d_acc'].iloc[0] < 0)
                                                    or (vb['sig_ll'].iloc[0] and vb['d_ll'].iloc[0] > 0))),
               'worst_platform': (rb.get('worst') or {}).get('platform', {}),
               'worst_grip': (rb.get('worst') or {}).get('orientation', {}),
               'worst_speed': (rb.get('worst') or {}).get('speed', {}),
               'worst_variant': (rb.get('worst') or {}).get('variant', {}),
               'nat_recall': r['nat_recall'], 'nat_p': (natd.get(k) or {}).get('holm_lower', math.nan),
               'nat_shortfall': bool((natd.get(k) or {}).get('shortfall', False)),
               'nat_observed': (natd.get(k) or {}).get('observed'), 'nat_expected': (natd.get(k) or {}).get('expected'),
               'grip_rotated_upright_only': (rb.get('grip') or {}).get('acc_upright_only', math.nan),
               'grip_d': (rb.get('grip') or {}).get('d_acc', math.nan),
               'grip_holm': (rb.get('grip') or {}).get('holm_acc', math.nan),
               'flip_acc': (rb.get('flip') or {}).get('acc_flipped', math.nan),
               'flip_acc_remedy': (rb.get('flip') or {}).get('acc_flipped_remedy', math.nan),
               'flip_invariant': (rb.get('flip') or {}).get('invariant_inputs'),
               'platform_d': (rb.get('platform') or {}).get('d_acc', math.nan),
               'platform_holm': (rb.get('platform') or {}).get('holm_acc', math.nan),
               'noise_d': (rb.get('noise') or {}).get('d_acc', math.nan),
               'noise_holm': (rb.get('noise') or {}).get('holm_acc', math.nan),
               'segmentation': segmentation_sensitivity(k, cands),
               'port_rank': PORTABILITY.get(k, (5, ''))[0], 'port': PORTABILITY.get(k, (5, ''))[1],
               'loso_seconds': r.get('loso_seconds'), 'nested_selected': int(freq.get(k, 0))}
        fl = rb.get('flip') or {}
        row['flip_d'] = fl.get('d_acc', math.nan)
        row['flip_holm'] = fl.get('holm_acc', math.nan)
        # a screen-away failure: a Holm-significant accuracy drop on the screen-away twins (paired, same trials)
        flip_fail = bool(np.isfinite(row['flip_holm']) and row['flip_holm'] < FD.ALPHA and row['flip_d'] < 0)
        row['flip_fails_raw'] = bool(flip_fail)
        row['flip_fixed_by_remedy'] = bool(np.isfinite(row['flip_acc_remedy'])
                                           and row['flip_acc_remedy'] >= row['accuracy'] - 0.05)
        row['grip_fails'] = bool(np.isfinite(row['grip_holm']) and row['grip_holm'] < FD.ALPHA and row['grip_d'] < 0)
        row['platform_fails'] = bool(np.isfinite(row['platform_holm']) and row['platform_holm'] < FD.ALPHA
                                     and row['platform_d'] < 0)
        # platform transfer and label noise are diagnostics (training covers both platforms; labels are never
        # changed), so they inform but do not filter: the plausible unknown test conditions are a grip absent from
        # training, the screen-away grip and natural (uninstructed) use
        row['robust'] = bool(not row['grip_fails'] and not row['nat_shortfall']
                             and (not row['flip_fails_raw'] or row['flip_fixed_by_remedy']))
        rows.append(row)
    df = pd.DataFrame(rows)
    rec = recommend(df)
    return {'best_by_log_loss': best, 'rows': df.to_dict(orient='records'), 'recommended': rec['key'],
            'rule': rec['rule'], 'leaders': rec['leaders'], 'robust_leaders': rec['robust_leaders']}


def recommend(df: pd.DataFrame) -> dict[str, Any]:
    """The pre-specified decision rule: (1) leaders = candidates not detectably worse than the lowest-log-loss
    candidate (paired accuracy and log loss, Holm within the family); (2) keep the robust ones (no Holm-significant
    drop on grips absent from training, no natural-use shortfall against the person-matched expectation, and a
    screen-away failure (a Holm-significant drop on the screen-away twins) only if the rest-pose check fixes it, i.e.
    brings the twins back within 5 points of the recorded accuracy); (3) among them the most portable, then the fewest
    outer-fold choices upstream (n_configs_tried), then the lowest log loss."""
    leaders = df[~df['worse_than_best']]
    robust = leaders[leaders['robust']]
    pool = robust if len(robust) else leaders
    pick = pool.sort_values(['port_rank', 'n_configs_tried', 'prod_log_loss']).iloc[0]
    return {'key': str(pick['key']), 'leaders': leaders['key'].tolist(), 'robust_leaders': robust['key'].tolist(),
            'rule': 'every candidate in its production form (own pipeline + the calibration remedy chosen inside each '
                    'fold); leaders = not detectably worse than the lowest-log-loss candidate in accuracy or log loss '
                    '(paired, Holm over the family) -> robust = no Holm-significant drop on grips absent from training, '
                    'no natural-use shortfall against the person-matched expectation, and a screen-away failure (Holm-'
                    'significant drop on the screen-away twins) only if the rest-pose check brings the twins back within '
                    '5 points of the recorded accuracy -> most portable -> fewest upstream outer-fold choices -> lowest '
                    'log loss'}


# ───────────────────────────── calibration of the final model ─────────────────────────────


def calibration(final_key: str, tt: pd.DataFrame, n_boot: int = 1000) -> dict[str, Any]:
    out: dict[str, Any] = {'methods': {}}
    names = {m: f'final_cal_{final_key}_{m}' for m in FE.CAL_METHODS if FD.exists(f'final_cal_{final_key}_{m}')}
    if 'none' not in names:
        return {}
    base = FD.preds(names['none'])
    plat = tt.set_index('trial_id')['platform'].astype(str)
    y = base['y'].to_numpy()
    persons = base['person_id'].to_numpy()
    for m, n in names.items():
        p = FD.preds(n).set_index('trial_id').reindex(base['trial_id'])['p_right'].to_numpy()
        blk = FCal.calibration_block(y, p, persons, n_boot=n_boot)
        per = {}
        for pl in ('android', 'ios'):
            mask = plat.reindex(base['trial_id']).to_numpy() == pl
            per[pl] = FCal.calibration_block(y[mask], p[mask], persons[mask], n_boot=n_boot)
        r = FD.res(n)
        params = [json.loads(s) for s in r.predictions.groupby('fold')['params'].first()]
        entry = {'overall': blk, 'platform': per, 'natural_use': FD.nat_counts(r), 'auc': r.metrics['roc_auc'],
                 'fold_params': [q.get('calib_params', {}) for q in params]}
        if m != 'none':
            pn = FD.preds(names['none']).set_index('trial_id').reindex(base['trial_id'])['p_right'].to_numpy()
            entry['d_log_loss'] = FCal.paired_metric(FCal.ll_terms(y, p), FCal.ll_terms(y, pn), persons)
            entry['d_brier'] = FCal.paired_metric((p - y) ** 2, (pn - y) ** 2, persons)
            entry['d_ece'] = FCal.paired_ece(y, p, pn, persons)
            entry['d_auc'] = r.metrics['roc_auc'] - FD.res(names['none']).metrics['roc_auc']
        out['methods'][m] = entry
    rem = [m for m in out['methods'] if m != 'none']
    adj = FD.holm([out['methods'][m]['d_log_loss']['p'] for m in rem])
    for m, a in zip(rem, adj):
        out['methods'][m]['d_log_loss']['holm'] = a
    # shipping rule: fewest parameters among the remedies that lower the log loss after Holm; else none
    helps = [m for m in rem if out['methods'][m]['d_log_loss']['holm'] < FD.ALPHA
             and out['methods'][m]['d_log_loss']['diff'] < 0]
    order = sorted(helps, key=lambda m: (FD.REMEDY_PARAMS.get(m) if FD.REMEDY_PARAMS.get(m) is not None else 99))
    out['chosen'] = order[0] if order else 'none'
    out['rule'] = ('a remedy is shipped only when its paired log-loss difference against none is negative with a '
                   'Holm-adjusted p below 0.05 over the remedies; among several, the fewest fitted parameters')
    for m in dict.fromkeys(['none', out['chosen']]):
        p = FD.preds(names[m]).set_index('trial_id').reindex(base['trial_id'])['p_right'].to_numpy()
        pl_arr = plat.reindex(base['trial_id']).to_numpy()
        out['methods'][m]['platform_difference'] = FCal.platform_difference(y, p, persons, pl_arr, n_boot=n_boot)
        nat_pred = FD.preds(names[m], extra=True)
        prim = base[['trial_id', 'person_id', 'y']].assign(p_right=p)
        out['methods'][m]['natural_calibration'] = FCal.natural_calibration(nat_pred, prim)
        out['methods'][m]['reliability'] = FCal.reliability_bands(y, p, persons, n_boot=n_boot).to_dict(orient='records')
        for pl in ('android', 'ios'):
            mask = plat.reindex(base['trial_id']).to_numpy() == pl
            out['methods'][m][f'reliability_{pl}'] = FCal.reliability_bands(
                y[mask], p[mask], persons[mask], n_boot=n_boot, n_bins=5).to_dict(orient='records')
    return out


# ───────────────────────────── multi-attempt aggregation ─────────────────────────────


def aggregation(final_key: str, cands: dict[str, FC.Candidate], d: Any, calibrator: str = 'none',
                n_draws: int = 200, n_boot: int = 1000) -> dict[str, Any]:
    """k attempts of one person and hand (the persons with at least max(k) trials of each hand, the same for every
    k). tau_k is fitted in every outer fold on random k-subsets of the TRAINING persons' inner cross-fitted logits
    (``final_nested`` cache; the held-out person never informs its own tau_k)."""
    from stage2.lanes import final_nested as FN
    c = cands[final_key]
    name = (f'final_cal_{final_key}_{calibrator}' if FD.exists(f'final_cal_{final_key}_{calibrator}')
            else FE.base_result_name(c))
    outer = FD.preds(name)
    folds = FN.harness_folds(np.zeros((len(d.trials), 1)), d.trials)
    fps = {k: FN._cand_fingerprint(x) for k, x in cands.items() if not x.components}
    kmax = max(FCal.AGG_KS)
    cnt = outer.groupby(['person_id', 'y']).size().unstack(fill_value=0)
    eligible = sorted(cnt.index[(cnt.get(0, 0) >= kmax) & (cnt.get(1, 0) >= kmax)].astype(str))
    rows, taus = [], []
    for f in folds:
        if f.test_person not in eligible:
            continue
        p_in = FN.inner_predictions(c, f, cands, compute=False, fingerprints=fps)
        if p_in is None:
            raise RuntimeError(f'inner predictions of {final_key} missing for {f.test_person}')
        z_in = FCal.logit(p_in)
        if calibrator != 'none':
            apply, _ = FCal.fit_calibrator(calibrator, z_in, f.y_train)
            z_in = FCal.logit(apply(z_in))
        o = outer[outer['person_id'].astype(str) == f.test_person]
        for k in FCal.AGG_KS:
            tr = FCal.k_subsets(z_in, f.groups_train, f.y_train, k, 40, f.seed)
            tau = FCal.fit_tau(tr['sum_z'].to_numpy(), tr['y'].to_numpy()) if k > 1 else 1.0
            taus.append({'person_id': f.test_person, 'k': k, 'tau': tau})
            te = FCal.k_subsets(FCal.logit(o['p_right'].to_numpy()), o['person_id'].to_numpy(), o['y'].to_numpy(), k,
                                n_draws, f.seed + 7)
            te['p_mean_p'] = te['mean_p']
            te['p_mean_logit'] = FCal.sigmoid(te['sum_z'] / k)
            te['p_sum_logit'] = FCal.sigmoid(te['sum_z'])
            te['p_sum_tau'] = FCal.sigmoid(te['sum_z'] / tau)
            rows.append(te)
    A = pd.concat(rows, ignore_index=True)
    out: dict[str, Any] = {'persons': eligible, 'n_draws': n_draws, 'calibrator': calibrator, 'rules': {}}
    for k in FCal.AGG_KS:
        a = A[A['k'] == k]
        y = a['y'].to_numpy()
        per = {}
        for rule in FCal.AGG_RULES:
            p = a[f'p_{rule}'].to_numpy()
            cred = cv.credit(y, p)
            conf = np.maximum(p, 1 - p)
            dfp = pd.DataFrame({'person': a['person'].to_numpy(), 'y': y, 'cred': cred, 'conf': conf,
                                'll': FCal.ll_terms(y, p)})
            pp = dfp.groupby(['person', 'y']).mean().groupby(level=0).mean()
            acc = float(pp['cred'].mean())
            over = pp['conf'] - pp['cred']
            rng = np.random.default_rng([C.SEED, k])
            vals = over.to_numpy()
            llv = pp['ll'].to_numpy()
            bi = [rng.integers(0, vals.size, vals.size) for _ in range(n_boot)]
            ob = [float(vals[i].mean()) for i in bi]
            lb = [float(llv[i].mean()) for i in bi]
            per[rule] = {'accuracy': acc, 'log_loss': float(llv.mean()), 'log_loss_ci': [float(np.quantile(lb, 0.025)),
                                                                                      float(np.quantile(lb, 0.975))],
                         'over_confidence': float(vals.mean()), 'over_ci': [float(np.quantile(ob, 0.025)),
                                                                            float(np.quantile(ob, 0.975))],
                         'ece': float(cv.ece(y, p)), 'calibrated': bool(np.quantile(ob, 0.025) <= 0 <= np.quantile(ob, 0.975))}
        out['rules'][k] = per
    T = pd.DataFrame(taus)
    out['tau_median'] = {int(k): float(g['tau'].median()) for k, g in T.groupby('k')}
    out['tau_range'] = {int(k): [float(g['tau'].min()), float(g['tau'].max())] for k, g in T.groupby('k')}
    # paired at k = 3: sum / tau_k against the mean P (per-person log loss)
    a = A[A['k'] == 3]
    y = a['y'].to_numpy()
    lt, lm = FCal.ll_terms(y, a['p_sum_tau'].to_numpy()), FCal.ll_terms(y, a['p_mean_p'].to_numpy())
    ls = FCal.ll_terms(y, a['p_sum_logit'].to_numpy())
    pers = a['person'].to_numpy()
    out['paired_k3'] = {'sum_tau_minus_mean_p': FCal.paired_metric(lt, lm, pers),
                        'sum_tau_minus_sum_logit': FCal.paired_metric(lt, ls, pers)}
    adj = FD.holm([out['paired_k3'][x]['p'] for x in out['paired_k3']])
    for x, h in zip(out['paired_k3'], adj):
        out['paired_k3'][x]['holm'] = h
    return out


def shipped_summary(name: str, key: str, remedy: str) -> dict[str, Any]:
    """The headline numbers of the shipped form (the final model with its shipped remedy): metrics with their person-
    bootstrap CIs, natural-use right-hand recall and the harness k-curve, read from its saved result."""
    r = FD.res(name)
    kc = r.k_curve.set_index('k')
    cands = FC.candidates()
    return {'result': name, 'candidate': key, 'remedy': remedy,
            # the candidate's own lane compared this many configurations on the outer folds, and this lane chose it
            # among n_candidates by the decision rule: its numbers are exploratory, the nested estimate is the honest one
            'n_configs_tried': int(cands[key].n_configs_tried), 'n_candidates': int(len(cands)),
            'leak_check': r.leak_check,
            **{m: r.metrics[m] for m in ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier', 'ece')},
            **{f'{m}_ci': list(r.ci[m]) for m in ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier')},
            'n_trials': int(r.metrics['n']), 'n_persons': int(r.metrics['n_persons']),
            'natural': FD.nat_counts(r),
            'k_curve': {int(k): {'accuracy': float(v['accuracy']), 'ci': [float(v['lo']), float(v['hi'])],
                                 'n_persons': int(v['n_persons'])} for k, v in kc.iterrows()}}


# ───────────────────────────── breakdowns ─────────────────────────────


def breakdowns(name: str, tt: pd.DataFrame) -> dict[str, Any]:
    r = FD.res(name)
    p = r.predictions.merge(tt.drop(columns=['person_id', 'y_right']), on='trial_id', how='left')
    p['credit'] = cv.credit(p['y'].to_numpy(), p['p_right'].to_numpy())
    p['p_true'] = np.where(p['y'] == 1, p['p_right'], 1 - p['p_right'])
    persons = (p.groupby('person_id').agg(accuracy=('credit', 'mean'), n=('credit', 'size'),
                                          handedness=('handedness_self', 'first'), platform=('platform', 'first'))
               .reset_index().sort_values('accuracy'))
    out: dict[str, Any] = {'persons': persons.to_dict(orient='records'), 'min_persons_ci': FD.MIN_PERSONS_CI}

    def strata(cols: list[str]) -> list[dict[str, Any]]:
        rows = []
        for lev, g in p.groupby(cols):
            lev = lev if isinstance(lev, tuple) else (lev,)
            n_p = int(g['person_id'].nunique())
            row = {**{c: str(v) for c, v in zip(cols, lev)}, 'n': int(len(g)), 'persons': n_p,
                   'accuracy': float(g['credit'].mean())}
            if n_p >= FD.MIN_PERSONS_CI:
                ci = cv.person_bootstrap(g[['person_id', 'y', 'p_right']], lambda yy, pp: float(np.mean(cv.credit(yy, pp))),
                                         n_boot=1000)
                row['ci'] = list(ci.get('stat', (math.nan, math.nan)))
            else:
                row['per_person'] = {str(k): float(v) for k, v in g.groupby('person_id')['credit'].mean().items()}
            rows.append(row)
        return rows
    out['platform'] = strata(['platform'])
    out['grip_speed'] = strata(['orientation', 'speed'])
    out['variant'] = strata(['variant'])
    out['handedness'] = strata(['handedness_self'])
    conf = p[p['p_true'] < 0.2]
    out['confident_errors'] = {'n': int(len(conf)), 'n_errors': int((p['credit'] < 1).sum()),
                               'by_person': conf['person_id'].value_counts().to_dict(),
                               'by_variant': conf['variant'].value_counts().to_dict(),
                               'by_speed': conf['speed'].value_counts().to_dict(),
                               'by_orientation': conf['orientation'].value_counts().to_dict(),
                               'rows': conf[['trial_key', 'person_id', 'platform', 'orientation', 'speed', 'variant',
                                             'y', 'p_right']].to_dict(orient='records')}   # report-safe trial keys
    out['k_curve'] = r.k_curve.to_dict(orient='records')
    out['natural'] = FD.nat_counts(r)
    return out


def variants(final_key: str, base_name: str) -> list[dict[str, Any]]:
    """Sensitivity of the final model to test-time segmentation errors and to a causal filter (lane p1's variant
    arrays, the same canonical code path), paired against the oracle-bound predictions."""
    from stage2.lanes.final_experiments import VARIANTS_TEST
    rows = []
    base = FD.preds(base_name)
    for v, desc in VARIANTS_TEST.items():
        n = f'final_var_{v}_{final_key}'
        if not FD.exists(n):
            continue
        q = FD.preds(n)
        out = FD.pc(q, base)
        rows.append({'variant': v, 'description': desc, 'accuracy': FD._acc(q), 'base_accuracy': FD._acc(base),
                     'd_acc': out['mean_person_acc_diff'], 'd_acc_ci': list(out['mean_person_acc_diff_ci95']),
                     'p_acc': out['p_sign_flip_acc'], 'd_ll': out['mean_person_logloss_diff'],
                     'd_ll_ci': list(out['mean_person_logloss_diff_ci95']), 'p_ll': out['p_sign_flip_logloss']})
    adj = FD.holm([r['p_acc'] for r in rows])
    for r, a in zip(rows, adj):
        r['holm_acc'] = a
    return rows


def remedies_final(final_key: str, cands: dict[str, FC.Candidate]) -> dict[str, Any]:
    """The final model's extra robustness experiments, paired on the same trials: screen-flip augmentation of the
    training rows (recorded test trials against the model as trained; screen-away twins against the model as trained
    on the same twins) and the size-matched control of the platform transfer (other platform only against as many
    randomly drawn persons of either platform: what the platform costs beyond the smaller training set)."""
    c = cands[final_key]
    base = FE.base_result_name(c)
    out: dict[str, Any] = {'key': final_key}

    def pair(a: str, b: str) -> dict[str, Any] | None:
        if not (FD.exists(a) and FD.exists(b)):
            return None
        o = FD.pc(FD.res(a), FD.res(b))
        return {'a': a, 'b': b, 'acc_a': FD.res(a).metrics['accuracy'], 'acc_b': FD.res(b).metrics['accuracy'],
                'll_a': FD.res(a).metrics['log_loss'], 'll_b': FD.res(b).metrics['log_loss'],
                'd_acc': o['mean_person_acc_diff'], 'd_acc_ci': list(o['mean_person_acc_diff_ci95']),
                'p_acc': o['p_sign_flip_acc'], 'd_ll': o['mean_person_logloss_diff'],
                'd_ll_ci': list(o['mean_person_logloss_diff_ci95']), 'p_ll': o['p_sign_flip_logloss']}
    plat = FE.P1_PLATFORM.get(final_key, f'final_plat_{final_key}')
    for k, (a, b) in {'flipaug_recorded': (f'final_flipaug_{final_key}', base),
                      'flipaug_twins': (f'final_flipaugx_{final_key}', f'final_flip_{final_key}'),
                      'platform_vs_control': (plat, f'final_platctl_{final_key}'),
                      'control_vs_all': (f'final_platctl_{final_key}', base)}.items():
        q = pair(a, b)
        if q is not None:                      # a comparison whose results are missing is left out (never null)
            out[k] = q
    tests = [k for k in ('flipaug_recorded', 'flipaug_twins', 'platform_vs_control', 'control_vs_all') if out.get(k)]
    for k, h in zip(tests, FD.holm([out[k]['p_acc'] for k in tests])):
        out[k]['holm_acc'] = h
    out['family'] = 'Holm over these paired accuracy comparisons of the final model'
    return out


def learning_curve(keys: list[str], cands: dict[str, FC.Candidate], n_boot: int = 1000) -> dict[str, Any]:
    from scipy.optimize import curve_fit
    out: dict[str, Any] = {}
    for key in keys:
        rows = []
        for n in FE.LC_SIZES:
            for r in range(FE.LC_REPS):
                nm = f'final_lc_{key}_n{n:02d}_r{r}'
                if FD.exists(nm):
                    x = FD.res(nm)
                    rows.append({'n': n, 'rep': r, 'accuracy': x.metrics['accuracy'], 'log_loss': x.metrics['log_loss']})
        full = FD.res(FE.base_result_name(cands[key]))
        n_all = int(full.predictions['person_id'].nunique()) - 1
        rows.append({'n': n_all, 'rep': 0, 'accuracy': full.metrics['accuracy'], 'log_loss': full.metrics['log_loss']})
        df = pd.DataFrame(rows)
        if df['n'].nunique() < 4:
            continue
        g = df.groupby('n').agg(acc_mean=('accuracy', 'mean'), acc_sd=('accuracy', 'std'), acc_min=('accuracy', 'min'),
                                acc_max=('accuracy', 'max'), ll_mean=('log_loss', 'mean'), ll_sd=('log_loss', 'std'),
                                reps=('accuracy', 'size')).reset_index()

        def law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
            return a - b * np.power(n, -c)
        fit, band = None, None
        try:
            par, _ = curve_fit(law, df['n'].to_numpy(float), df['accuracy'].to_numpy(float), p0=(0.9, 0.5, 0.5),
                               bounds=([0.5, 0.0, 0.05], [1.0, 5.0, 3.0]), maxfev=20000)
            grid = np.arange(3, 61)
            rng = np.random.default_rng(C.SEED)
            curves = []
            for _ in range(n_boot):          # resample the repetitions within each size
                bs = pd.concat([grp.sample(len(grp), replace=True, random_state=int(rng.integers(1 << 31)))
                                for _, grp in df.groupby('n')])
                try:
                    pb, _ = curve_fit(law, bs['n'].to_numpy(float), bs['accuracy'].to_numpy(float), p0=par,
                                      bounds=([0.5, 0.0, 0.05], [1.0, 5.0, 3.0]), maxfev=5000)
                    curves.append(law(grid, *pb))
                except RuntimeError:
                    continue
            curves = np.asarray(curves)
            fit = {'a': float(par[0]), 'b': float(par[1]), 'c': float(par[2]), 'grid': grid.tolist(),
                   'fit': law(grid, *par).tolist(),
                   'lo': np.quantile(curves, 0.025, axis=0).tolist(), 'hi': np.quantile(curves, 0.975, axis=0).tolist(),
                   'n_boot_ok': int(len(curves))}
            for nn in (30, 40, 60):
                i = int(np.flatnonzero(grid == nn)[0])
                fit[f'at_{nn}'] = [float(fit['fit'][i]), float(fit['lo'][i]), float(fit['hi'][i])]
            # the extrapolated gain of doubling the training persons (from all of them to 40), on the same bootstrap
            # curves: is recording more participants worthwhile?
            ia = int(np.flatnonzero(grid == n_all)[0]) if n_all in grid else None
            if ia is not None:
                i40 = int(np.flatnonzero(grid == 40)[0])
                gain = curves[:, i40] - curves[:, ia]
                fit['gain_to_40'] = [float(fit['fit'][i40] - fit['fit'][ia]), float(np.quantile(gain, 0.025)),
                                     float(np.quantile(gain, 0.975))]
            band = True
        except (RuntimeError, ValueError):
            fit = None
        out[key] = {'points': df.to_dict(orient='records'), 'summary': g.to_dict(orient='records'), 'fit': fit,
                    'n_all': n_all, 'fit_ok': bool(band)}
    return out


def prior_shift(name: str, priors: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict[str, Any]]:
    """P' = sigmoid(logit(P) + log(pi / (1 - pi))): the deployment prior pi of the right hand moves every logit by the
    same amount. Primary trials (balanced hands): left and right recall, balanced accuracy, log loss; natural use:
    right-hand recall."""
    r = FD.res(name)
    p, e = r.predictions, r.extra_predictions
    rows = []
    for pi in priors:
        s = math.log(pi / (1 - pi))
        q = FCal.sigmoid(FCal.logit(p['p_right'].to_numpy()) + s)
        y = p['y'].to_numpy()
        c = cv.credit(y, q)
        qe = FCal.sigmoid(FCal.logit(e['p_right'].to_numpy()) + s) if e is not None and len(e) else np.array([])
        rows.append({'prior': pi, 'shift': s, 'left_recall': float(c[y == 0].mean()),
                     'right_recall': float(c[y == 1].mean()), 'balanced_accuracy': float((c[y == 0].mean() + c[y == 1].mean()) / 2),
                     'log_loss': FCal.log_loss(y, q),
                     'natural_recall': float(np.mean(cv.credit(e['y'].to_numpy(), qe))) if qe.size else math.nan})
    return rows


import json  # noqa: E402  (used by calibration)

__all__ = ['decision_table', 'recommend', 'calibration', 'aggregation', 'breakdowns', 'variants', 'remedies_final',
           'learning_curve',
           'prior_shift', 'PORTABILITY']
