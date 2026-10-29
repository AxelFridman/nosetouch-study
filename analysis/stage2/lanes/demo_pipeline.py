"""Lane demo: THE DEMOGRAPHICS ADD-ON TEST (fast-track first pass). Steps: compute (experiments + derived comparisons),
figures, tables, numbers (+ the generated demo_SUMMARY.md).

Would demographics help if production knew them? The final model's logit is stacked with the person covariates
(design (a): second-stage logistic fitted inside each outer fold on lane final's INNER cross-fitted logits of the
training persons, applied to the held-out person's saved LOSO logit), against the same second stage without them.
Nothing from this lane enters the production model; results/finalists_demo.json is an empty list.
"""
from stage2.pipeline import Lane, limit_threads

limit_threads(3)                                   # fast-track thread budget (3)

import math                                         # noqa: E402

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from stage2 import report_export as RX              # noqa: E402

L = Lane('demo')
ORDER = 930
DEPENDS = ('final',)

#: the add-on variants (one Holm family per metric: each vs the sensor-only second stage on the same folds)
ADDON_VARIANTS: tuple[str, ...] = ('demo_cov_main', 'demo_cov_inter')
SENSOR_ONLY = 'demo_sensor_stack'
#: name -> (kind, argument, n_configs_tried, label)
EXPERIMENTS: dict[str, tuple[str, str, int, str]] = {
    'demo_sensor_stack': ('stack', 'sensor', 16, 'final logit, second stage (sensor only)'),
    'demo_cov_main': ('stack', 'main', 16, '+ covariates (main effects)'),
    'demo_cov_inter': ('stack', 'inter', 16, '+ covariates and covariate x logit'),
    'demo_cov_only': ('stack', 'cov_only', 1, 'covariates only (floor)'),
    'demo_rule_phone': ('rule', 'phone', 1, 'rule: usual phone hand'),
    'demo_rule_right': ('rule', 'right', 1, 'rule: always right'),
}
NOTE = ('SANCTIONED DEMOGRAPHICS ADD-ON TEST (lane demo): person covariates (age, sex, height, self-reported '
        'handedness, usual phone hand) enter as the separate block person_covariates, the test person\'s covariates '
        'at prediction time; never part of the production model. ')
DESIGN = ('Second-stage L2 logistic on the final model (lane final m3_deep) logit, fitted per outer fold on lane '
          'final\'s inner cross-fitted logits of the training persons, applied to the held-out person\'s saved LOSO '
          'logit; C nested over fold.inner_splits(). ')
SHIPPED = 'final_cal_m3_deep_none'


def _spec_notes(name: str) -> str:
    kind, arg, k, label = EXPERIMENTS[name]
    if kind == 'rule':
        return NOTE + f'{label}: a fixed rule (no fitting), for the natural-use base-rate note only.'
    base = ('exploratory base: the final model was chosen among 16 configurations on the outer folds; '
            if k > 1 else '')
    return NOTE + DESIGN + f'Feature set {arg!r} ({label}); ' + base + 'instructed conditions never used.'


def compute(force=False, only=None):
    todo = [n for n in EXPERIMENTS if L.should_run(n, force, only)]
    derived_missing = not (L.tab_dir / 'demo_derived.json').exists()
    if todo:
        from stage2.data import load_cache
        from stage2.lanes import demo_models as DM
        d, nat = load_cache(), load_cache(('natural_use',))
        persons = sorted(d.trials['person_id'].astype(str).unique())
        X, Xn, info = DM.build_inputs(d, nat, persons)
        L.write_json('demo_inner_cache_check', info)
        for name in todo:
            kind, arg, k, label = EXPERIMENTS[name]
            fp = DM.stack_fit_predict(arg, persons) if kind == 'stack' else DM.rule_fit_predict(arg)
            print(f'demo: {name} ...', flush=True)
            L.run_experiment(name, fp, X, d.trials, extra=(Xn, nat.trials), notes=_spec_notes(name),
                             n_configs_tried=k, model=label, params={'feature_set': arg, 'kind': kind})
    if todo or force or derived_missing or (only and 'demo_derived' in only):
        derive()
    else:
        print('demo: results current; skipped (--force recomputes)')


# ───────────────────────────── derived comparisons (saved; figures and tables only read them) ─────────────────────────────


def _holm(ps):
    from statsmodels.stats.multitest import multipletests
    ps = [1.0 if (p is None or not np.isfinite(p)) else float(p) for p in ps]
    return [float(v) for v in multipletests(ps, method='holm')[1]] if ps else []


def _pair(a, b) -> dict:
    from stage2 import cv
    with np.errstate(all='ignore'):
        pc = cv.paired_compare(a, b)
    j = a.predictions[['trial_id', 'p_right']].merge(b.predictions[['trial_id', 'p_right']], on='trial_id',
                                                     suffixes=('_a', '_b'))
    flips = int(np.sum((j['p_right_a'] > 0.5) != (j['p_right_b'] > 0.5)))
    out = {k: v for k, v in pc.items() if k != 'per_person'}
    out['n_class_changes'] = flips
    out['identical'] = bool(abs(pc['mean_person_logloss_diff']) < 1e-3 and flips == 0)
    out['per_person'] = pc['per_person'][['person_id', 'd_acc', 'd_ll']].to_dict(orient='records')
    return out


def _nat(r) -> dict:
    e = r.extra_predictions
    if e is None or not len(e):
        return {'n': 0, 'recall': math.nan, 'n_right_answers': 0, 'n_ties': 0}
    p = e['p_right'].to_numpy(dtype=float)
    ties = np.abs(p - 0.5) <= 1e-12
    return {'n': int(len(e)), 'recall': float(r.extra_metrics['accuracy']), 'n_right_answers': int(np.sum(p > 0.5 + 1e-12)),
            'n_ties': int(ties.sum()), 'all_right': bool((e['y'] == 1).all())}


def derive():
    from stage2 import cv
    from stage2.data import load_cache
    from stage2.lanes import demo_models as DM
    R = {n: L.load_result(n) for n in EXPERIMENTS}
    shipped = cv.load_result(SHIPPED)
    rows = []
    for n, r in list(R.items()) + [(SHIPPED, shipped)]:
        m, ci = r.metrics, r.ci
        rows.append({'name': n, 'label': EXPERIMENTS[n][3] if n in EXPERIMENTS else 'final model as shipped',
                     'n_configs_tried': int(r.info.get('n_configs_tried') or 1),
                     **{k: m.get(k) for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier', 'n')},
                     **{f'{k}_lo': ci.get(k, (math.nan, math.nan))[0] for k in ('accuracy', 'log_loss', 'roc_auc')},
                     **{f'{k}_hi': ci.get(k, (math.nan, math.nan))[1] for k in ('accuracy', 'log_loss', 'roc_auc')},
                     **{f'nat_{k}': v for k, v in _nat(r).items()}})
    # paired comparisons: the Holm family (add-on variants vs the sensor-only second stage), per metric
    fam = {v: _pair(R[v], R[SENSOR_ONLY]) for v in ADDON_VARIANTS}
    h_acc = _holm([fam[v]['p_sign_flip_acc'] for v in ADDON_VARIANTS])
    h_ll = _holm([fam[v]['p_sign_flip_logloss'] for v in ADDON_VARIANTS])
    for v, pa, pl in zip(ADDON_VARIANTS, h_acc, h_ll):
        fam[v]['holm_p_acc'], fam[v]['holm_p_logloss'] = pa, pl
    # descriptive (outside the family): the second stage itself vs the shipped model; the floor vs the sensor stack
    desc = {'sensor_vs_shipped': _pair(R[SENSOR_ONLY], shipped), 'cov_only_vs_sensor': _pair(R['demo_cov_only'],
                                                                                           R[SENSOR_ONLY])}
    # natural-use class changes between each add-on and the sensor-only stack
    for v in ADDON_VARIANTS:
        a, b = R[v].extra_predictions, R[SENSOR_ONLY].extra_predictions
        j = a[['trial_id', 'p_right']].merge(b[['trial_id', 'p_right']], on='trial_id', suffixes=('_a', '_b'))
        fam[v]['nat_class_changes'] = int(np.sum((j['p_right_a'] > 0.5) != (j['p_right_b'] > 0.5)))
    # covariate profile (counts only) and the base rate of the natural-use set
    d = load_cache()
    P = d.persons
    prof = {'n_persons': int(len(P)), 'n_female': int((P['sex'] == 'female').sum()),
            'n_left_handed': int((P['handedness_self'] == 'left').sum()),
            'n_right_handed': int((P['handedness_self'] == 'right').sum()),
            'n_phone_right': int((P['phone_hand'] == 'right').sum()),
            'n_phone_both': int((P['phone_hand'] == 'both').sum()),
            'n_phone_left': int((P['phone_hand'] == 'left').sum()),
            'n_height_missing': int(pd.to_numeric(P['height_cm'], errors='coerce').isna().sum()),
            'age_min': float(P['age_years'].min()), 'age_max': float(P['age_years'].max())}
    # second-stage coefficients per fold (what the covariates change)
    coefs = pd.concat([DM.fold_coefficients(R[v]).assign(model=v) for v in ('demo_sensor_stack',) + ADDON_VARIANTS +
                       ('demo_cov_only',)], ignore_index=True)
    coef_sum = (coefs.groupby(['model', 'term'], sort=False)['coef']
                .agg(median='median', q10=lambda s: float(np.quantile(s, 0.1)), q90=lambda s: float(np.quantile(s, 0.9)),
                     n_folds='size').reset_index())
    chosen_c = {v: pd.Series([c for c in coefs[coefs['model'] == v].groupby('fold')['C'].first()]).value_counts()
                .sort_index().to_dict() for v in coefs['model'].unique()}
    L.write_json('demo_derived', {
        'models': rows, 'family': {v: {k: x for k, x in fam[v].items()} for v in ADDON_VARIANTS},
        'family_name': 'add-on variants vs the sensor-only second stage; Holm within each metric '
                       '(accuracy: per-person accuracy difference, log loss: per-person log-loss difference; '
                       'exact sign-flip p-values over persons)',
        'descriptive': desc, 'profile': prof, 'coef_summary': coef_sum.to_dict(orient='records'),
        'chosen_C': {k: {str(c): int(n) for c, n in v.items()} for k, v in chosen_c.items()}},
        inputs=[*EXPERIMENTS, SHIPPED])


# ───────────────────────────── presentation ─────────────────────────────


def figures(force=False, only=None):
    from stage2.lanes import demo_figures
    demo_figures.draw(only)


def _fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def tables(force=False, only=None):
    D = L.read_json('demo_derived')
    fam = D['family']
    out = []
    for r in D['models']:
        n = r['name']
        f = fam.get(n)
        nat = (f"{r['nat_recall']:.1%} ({r['nat_n_right_answers']} of {r['nat_n']}"
               + (f", {r['nat_n_ties']} ties" if r['nat_n_ties'] else '') + ')') if r['nat_n'] else ''
        out.append({
            'model': r['label'],
            'accuracy': f"{r['accuracy']:.1%} [{r['accuracy_lo']:.1%}, {r['accuracy_hi']:.1%}]",
            'balanced accuracy': f"{r['balanced_accuracy']:.1%}",
            'log loss': f"{r['log_loss']:.3f} [{r['log_loss_lo']:.3f}, {r['log_loss_hi']:.3f}]",
            'AUC': '' if r['roc_auc'] is None or not np.isfinite(r['roc_auc']) else f"{r['roc_auc']:.3f}",
            'natural use: right-hand recall': nat,
            'vs sensor only: d accuracy (points)': '' if f is None else
            (f"{100 * f['mean_person_acc_diff']:+.1f} [{100 * f['mean_person_acc_diff_ci95'][0]:+.1f}, "
             f"{100 * f['mean_person_acc_diff_ci95'][1]:+.1f}], Holm p {_fmt_p(f['holm_p_acc'])}"),
            'vs sensor only: d log loss': '' if f is None else
            (f"{f['mean_person_logloss_diff']:+.3f} [{f['mean_person_logloss_diff_ci95'][0]:+.3f}, "
             f"{f['mean_person_logloss_diff_ci95'][1]:+.3f}], Holm p {_fmt_p(f['holm_p_logloss'])}"),
        })
    df = pd.DataFrame(out)
    L.export_table(df, 'demo_models',
                   header={'balanced accuracy': 'bal. acc.', 'natural use: right-hand recall': 'natural use',
                           'vs sensor only: d accuracy (points)': 'd accuracy (points)',
                           'vs sensor only: d log loss': 'd log loss'},
                   caption=('The demographics add-on test (leave-one-person-out; person-bootstrap 95\\% CIs). '
                            'Second stage on the final model\'s logit, fitted on inner cross-fitted logits of the '
                            'training persons; d = add-on minus the sensor-only second stage on the same folds '
                            '(mean per-person accuracy and log-loss differences, paired person bootstrap), Holm across the '
                            'add-on variants within each metric. Natural use: right-hand recall on free-choice '
                            'trials (every one is a right-hand trial), not a balanced accuracy. Rules are fixed '
                            '(no fitting) and shown for the base-rate effect only.'))


def numbers(force=False, only=None):
    E = RX.numbers_entry
    D = L.read_json('demo_derived')
    src = 'tables/demo/demo_derived.json'
    n = {}
    by = {r['name']: r for r in D['models']}
    short = {'demo_sensor_stack': 'sensor', 'demo_cov_main': 'main', 'demo_cov_inter': 'inter',
             'demo_cov_only': 'covonly', 'demo_rule_phone': 'rulephone', 'demo_rule_right': 'ruleright',
             SHIPPED: 'shipped'}
    for name, s in short.items():
        r = by[name]
        rs = f'results/{name}.json'
        n[f'{s}_acc'] = E(r['accuracy'], '.1%', rs, f'{r["label"]}: LOSO accuracy')
        n[f'{s}_acc_lo'] = E(r['accuracy_lo'], '.1%', rs, f'{r["label"]}: accuracy 95% CI low')
        n[f'{s}_acc_hi'] = E(r['accuracy_hi'], '.1%', rs, f'{r["label"]}: accuracy 95% CI high')
        n[f'{s}_bacc'] = E(r['balanced_accuracy'], '.1%', rs, f'{r["label"]}: balanced accuracy')
        n[f'{s}_ll'] = E(r['log_loss'], '.3f', rs, f'{r["label"]}: log loss')
        n[f'{s}_ll_lo'] = E(r['log_loss_lo'], '.3f', rs, f'{r["label"]}: log loss 95% CI low')
        n[f'{s}_ll_hi'] = E(r['log_loss_hi'], '.3f', rs, f'{r["label"]}: log loss 95% CI high')
        if r['roc_auc'] is not None and np.isfinite(r['roc_auc']):
            n[f'{s}_auc'] = E(r['roc_auc'], '.3f', rs, f'{r["label"]}: ROC AUC')
        n[f'{s}_nat_recall'] = E(r['nat_recall'], '.1%', rs, f'{r["label"]}: natural-use right-hand recall '
                                 '(every free-choice trial is right-hand; not a balanced accuracy)')
        n[f'{s}_nat_right'] = E(int(r['nat_n_right_answers']), 'd', rs, f'{r["label"]}: natural-use trials answered right')
        n[f'{s}_nat_ties'] = E(int(r['nat_n_ties']), 'd', rs, f'{r["label"]}: natural-use ties (half credit)')
    rc = L.load_result('demo_cov_only')
    pall = np.concatenate([rc.predictions['p_right'].to_numpy(float), rc.extra_predictions['p_right'].to_numpy(float)])
    n['covonly_max_dev'] = E(float(np.max(np.abs(pall - 0.5))), '.3f', 'results/demo_cov_only.json',
                             'covariates-only model: largest |P(right) - 0.5| over primary and natural-use trials')
    n['primary_right_share'] = E(float(rc.predictions['y'].mean()), '.1%', 'results/demo_cov_only.json',
                                 'share of right-hand trials in the primary set')
    n['chance'] = E(0.5, '.1f', src, 'chance level of P(right)')
    n['ci_level'] = E(0.95, '.0%', src, 'confidence level of every interval')
    n['nat_n'] = E(int(by[SENSOR_ONLY]['nat_n']), 'd', src, 'natural-use headline trials (one per person)')
    for v in ADDON_VARIANTS:
        f, s = D['family'][v], short[v]
        n[f'{s}_dacc'] = E(100 * f['mean_person_acc_diff'], '+.1f', src, f'{v} minus sensor-only: mean per-person accuracy difference (points)')
        n[f'{s}_dacc_lo'] = E(100 * f['mean_person_acc_diff_ci95'][0], '+.1f', src, f'{v}: accuracy difference CI low')
        n[f'{s}_dacc_hi'] = E(100 * f['mean_person_acc_diff_ci95'][1], '+.1f', src, f'{v}: accuracy difference CI high')
        n[f'{s}_dll'] = E(f['mean_person_logloss_diff'], '+.3f', src, f'{v} minus sensor-only: mean per-person log loss')
        n[f'{s}_dll_lo'] = E(f['mean_person_logloss_diff_ci95'][0], '+.3f', src, f'{v}: log-loss difference CI low')
        n[f'{s}_dll_hi'] = E(f['mean_person_logloss_diff_ci95'][1], '+.3f', src, f'{v}: log-loss difference CI high')
        n[f'{s}_holm_acc'] = E(f['holm_p_acc'], '.3f', src, f'{v}: Holm-adjusted sign-flip p, accuracy')
        n[f'{s}_holm_ll'] = E(f['holm_p_logloss'], '.3f', src, f'{v}: Holm-adjusted sign-flip p, log loss')
        n[f'{s}_flips'] = E(int(f['n_class_changes']), 'd', src, f'{v}: primary trials whose answer changes')
        n[f'{s}_nat_flips'] = E(int(f['nat_class_changes']), 'd', src, f'{v}: natural-use trials whose answer changes')
        n[f'{s}_better'] = E(int(f['persons_a_better']), 'd', src, f'{v}: persons more accurate with the covariates')
        n[f'{s}_worse'] = E(int(f['persons_b_better']), 'd', src, f'{v}: persons less accurate with the covariates')
        n[f'{s}_same'] = E(int(sum(abs(p['d_acc']) < 1e-12 for p in f['per_person'])), 'd', src,
                           f'{v}: persons whose accuracy does not change with the covariates')
    s = D['descriptive']['sensor_vs_shipped']
    n['stack_vs_shipped_dacc'] = E(100 * s['mean_person_acc_diff'], '+.1f', src, 'sensor-only second stage minus shipped')
    n['stack_vs_shipped_dll'] = E(s['mean_person_logloss_diff'], '+.3f', src, 'sensor-only stage minus shipped: log loss')
    n['stack_vs_shipped_p_acc'] = E(s['p_sign_flip_acc'], '.3f', src, 'sign-flip p, accuracy (unadjusted, descriptive)')
    n['stack_vs_shipped_p_ll'] = E(s['p_sign_flip_logloss'], '.3f', src, 'sign-flip p (unadjusted, descriptive)')
    n['base_configs'] = E(int(by[SENSOR_ONLY]['n_configs_tried']), 'd', 'results/demo_sensor_stack.json',
                          'configurations the final model was chosen among on the outer folds (exploratory base)')
    n['alpha'] = E(0.05, '.2f', src, 'significance level of the Holm-adjusted tests')
    for k, v in D['profile'].items():
        n[f'profile_{k}'] = E(v, 'd' if isinstance(v, int) else '.0f', src, f'covariate profile: {k}')
    L.write_numbers(n)
    L.write_finalists([])
    write_summary(n)


def write_summary(n: dict) -> None:
    from stage2.lanes import demo_summary
    demo_summary.write(L, n)


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
