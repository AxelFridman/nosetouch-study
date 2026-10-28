"""Lane p1: report tables (CSV + LaTeX booktabs), the numbers file and the finalists, from saved files only.

Tables (tables/p1/): ``p1_decisions`` (main text: one row per data-treatment choice), ``p1_decisions_by_model``,
``p1_options_a`` / ``p1_options_b`` (appendix: every option x panel model, paired with the default, BH q within the
options family), ``p1_robustness`` (test-time segmentation errors, and the jitter-training remedy pair),
``p1_pairs`` (remedies against their problem, options against size-matched controls), ``p1_panel`` (the panel and the
nested-calibrated models), ``p1_calibration`` / ``p1_calibration_paired`` / ``p1_calibration_platform`` /
``p1_calibration_tests``, ``p1_aggregates`` (k-attempt rules) and ``p1_aggregates_paired``, ``p1_subsets`` (excluded,
flagged, swap-signature trials) and ``p1_conditions`` (best panel model per condition, exploratory designation).
Numbers: results/p1_numbers.json (:mod:`p1_numbers`). Finalists: results/finalists_p1.json. Summary:
reports/stage2/p1_SUMMARY.md (:mod:`p1_summary`, written by the numbers step). Light imports only.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from stage2.lanes import p1_labels as LB
from stage2.lanes.p1_labels import MODEL_LABELS, MODELS, SHORT
from stage2.pipeline import Lane, selected

L = Lane('p1')
RAG = r'>{\raggedright\arraybackslash}'          # ragged-right paragraph cells (array package)
METHOD_LABELS = {**LB.CAL_LABELS, 'none': 'none', 'auto': 'nested'}


# ───────────────────────────── cell formatting ─────────────────────────────


def _ok(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def fmt(v: Any, spec: str, scale: float = 1.0) -> str:
    return format(scale * float(v), spec) if _ok(v) else '--'


def est(v: Any, lo: Any, hi: Any, spec: str = '.1f', scale: float = 100.0) -> str:
    """'85.6 [81.0, 89.8]' (estimate and 95 % CI in one cell; '--' when missing; no CI part when it is missing)."""
    if not _ok(v):
        return '--'
    ci = f' [{fmt(lo, spec, scale)}, {fmt(hi, spec, scale)}]' if _ok(lo) and _ok(hi) else ''
    return fmt(v, spec, scale) + ci


def pval(v: Any) -> str:
    if not _ok(v):
        return '--'
    v = float(v)
    return '<0.001' if v < 0.001 else (f'{v:.3f}' if v < 0.1 else f'{v:.2f}')


def verdict_short(v: Any) -> str:
    return {'better': 'better*', 'worse': 'worse*', 'nominally better': 'better (nom.)',
            'nominally worse': 'worse (nom.)', 'identical': 'identical', 'no difference': '--'}.get(str(v), '--')


def comparisons() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('p1_comparisons')['comparisons'])


def runs() -> pd.DataFrame:
    return pd.DataFrame(L.read_json('p1_runs')['runs'])


def _order(c: pd.DataFrame) -> pd.DataFrame:
    order = list(L.read_json('p1_runs').get('grid_order', []))
    fam = {f: i for i, f in enumerate(LB.DECISION_FAMILIES)}
    return (c.assign(_f=c['report_family'].map(fam).fillna(len(fam)),
                     _o=c['name'].map(lambda n: order.index(n) if n in order else len(order)))
            .sort_values(['_f', '_o']))


# ───────────────────────────── decisions and options ─────────────────────────────


FAMILY_SPLIT = ('filter', 'gaps', 'sampling', 'gravity', 'frame')    # first half of the options table
OPTION_COLUMNS = ('Choice', 'Option', 'Model', 'Acc. %', 'Diff. [95% CI]', 'q (BH)', 'Diff. LL [95% CI]', 'Verdict')


def options_table(part: str = 'all') -> pd.DataFrame:
    """Every option x model against the current pipeline; part 'a' = FAMILY_SPLIT, 'b' = the other families."""
    c = comparisons()
    c = c[c['mult_family'] == 'options']
    if part == 'a':
        c = c[c['report_family'].isin(FAMILY_SPLIT)]
    elif part == 'b':
        c = c[~c['report_family'].isin(FAMILY_SPLIT)]
    c = _order(c)
    return pd.DataFrame({
        'Choice': c['report_family'].map(LB.FAMILY_LABELS), 'Option': c['option'].map(LB.option_label),
        'Model': c['model'].map(SHORT), 'Acc. %': [fmt(v, '.1f', 100) for v in c['acc']],
        'Diff. [95% CI]': [est(d, a, b, '+.1f') for d, a, b in zip(c['d_acc'], c['d_acc_lo'], c['d_acc_hi'])],
        'q (BH)': [pval(v) for v in c['q_bh']],
        'Diff. LL [95% CI]': [est(d, a, b, '+.3f', 1.0) for d, a, b in
                              zip(c['d_logloss'], c['d_logloss_lo'], c['d_logloss_hi'])],
        'Verdict': [verdict_short(v) for v in c['verdict']]}, columns=list(OPTION_COLUMNS))


DECISION_COLUMNS = ('Choice', 'Current default', 'Options (models)', 'Evidence (accuracy, points [95% CI], q)',
                    'Gain forgone at most', 'Decision')


def _ev_list(items: list[dict[str, Any]]) -> str:
    return '; '.join(f"{LB.option_label(x['option'])} ({SHORT[x['model']]} {fmt(x['d_acc'], '+.1f', 100)} "
                     f"[{fmt(x['d_acc_lo'], '+.1f', 100)}, {fmt(x['d_acc_hi'], '+.1f', 100)}], q {pval(x['q_bh'])})"
                     for x in items)


def evidence_text(r: dict[str, Any]) -> str:
    """The evidence cell of a decision row: options worse / better after the BH correction, then the nominal ones."""
    parts = []
    if r['worse']:
        parts.append('worse: ' + _ev_list(r['worse']))
    if r['better']:
        parts.append('better: ' + _ev_list(r['better']))
    nom = r['nominally_worse'] + r['nominally_better']
    if nom:
        parts.append(f'nominal only ({len(nom)}): ' + '; '.join(f"{LB.option_label(x['option'])} ({SHORT[x['model']]})"
                                                               for x in nom))
    if r['identical']:
        parts.append('identical: ' + ', '.join(f"{LB.option_label(x['option'])} ({SHORT[x['model']]})"
                                               for x in r['identical']))
    return '. '.join(parts) if parts else 'no difference for any option'


def decision_text(r: dict[str, Any]) -> str:
    d = r['decision']
    if d == 'keep the default' and not r.get('decidable', True):
        return 'keep (no option decidable by the rule)'
    return 'keep' if d == 'keep the default' else d


def decisions_table() -> pd.DataFrame:
    rows = []
    for r in L.read_json('p1_decisions')['rows']:
        models = ', '.join(SHORT[m] for m in r.get('models_tested', []))
        g = r['max_gain']
        rows.append(dict(zip(DECISION_COLUMNS, (
            LB.FAMILY_LABELS[r['family']], r['default'], f"{r['n_options']} ({models})", evidence_text(r),
            f"{fmt(g['d_acc_hi'], '+.1f', 100)} ({LB.option_label(g['option'])}, {SHORT[g['model']]})",
            decision_text(r)))))
    return pd.DataFrame(rows, columns=list(DECISION_COLUMNS))


def decisions_by_model_table() -> pd.DataFrame:
    """Per choice and panel model: the range of the paired differences, how many options, how many worse / better
    after the BH correction, how many only nominally, how many identical."""
    rows = []
    for r in L.read_json('p1_decisions')['rows']:
        cells = {}
        for m in MODELS:
            ev = r['evidence'].get(m)
            if not ev:
                cells[SHORT[m]] = '--'
                continue
            extra = [f"{ev[k]} {w}" for k, w in (('n_worse', 'worse*'), ('n_better', 'better*'), ('n_nominal', 'nominal'),
                                                  ('n_identical', 'identical')) if ev.get(k)]
            cells[SHORT[m]] = (f"{fmt(ev['min_d'], '+.1f', 100)} to {fmt(ev['max_d'], '+.1f', 100)} "
                               f"({ev['n_options']} opt." + ('; ' + '; '.join(extra) if extra else '') + ')')
        rows.append({'Choice': LB.FAMILY_LABELS[r['family']], **cells, 'Decision': decision_text(r)})
    return pd.DataFrame(rows, columns=['Choice', *[SHORT[m] for m in MODELS], 'Decision'])


ROBUST_COLUMNS = ('Model', 'Test-time perturbation', 'Acc. %', 'Diff. [95% CI]', 'q (BH)', 'Diff. LL [95% CI]',
                  'Verdict')


def robustness_table() -> pd.DataFrame:
    """Segmentation errors at test time (the robustness family), then the jitter-training remedy pairs."""
    cj = L.read_json('p1_comparisons')
    c = pd.DataFrame(cj['comparisons'])
    c = _order(c[c['mult_family'] == 'robustness']).sort_values(['model', '_o'])
    rows = [dict(zip(ROBUST_COLUMNS, (SHORT[r['model']], LB.option_label(r['option']), fmt(r['acc'], '.1f', 100),
                                      est(r['d_acc'], r['d_acc_lo'], r['d_acc_hi'], '+.1f'), pval(r['q_bh']),
                                      est(r['d_logloss'], r['d_logloss_lo'], r['d_logloss_hi'], '+.3f', 1.0),
                                      verdict_short(r['verdict'])))) for _, r in c.iterrows()]
    for r in cj.get('remedies', []):
        if r.get('family') != 'segmentation':
            continue
        rows.append(dict(zip(ROBUST_COLUMNS, (SHORT[r['model']], 'jitter-trained vs oracle-trained, both at ±100 ms',
                                              fmt(r['acc'], '.1f', 100), est(r['d_acc'], r['d_acc_lo'], r['d_acc_hi'], '+.1f'),
                                              pval(r['q_bh']),
                                              est(r['d_logloss'], r['d_logloss_lo'], r['d_logloss_hi'], '+.3f', 1.0),
                                              verdict_short(r['verdict'])))))
    return pd.DataFrame(rows, columns=list(ROBUST_COLUMNS))


PAIR_COLUMNS = ('Model', 'A against B', 'Acc. A %', 'Acc. B %', 'Diff. [95% CI]', 'q (BH)', 'Diff. LL [95% CI]', 'Verdict')


def pairs_table() -> pd.DataFrame:
    """Remedies against the problem they address, options against their size-matched controls, and the controls
    against the current pipeline (one multiplicity family)."""
    cj = L.read_json('p1_comparisons')
    rows = []
    for r in cj.get('remedies', []) + cj.get('controls', []):
        rows.append((r['family'], SHORT[r['model']], r['what'], r['acc'], r['acc_base'], r))
    c = pd.DataFrame(cj['comparisons'])
    for _, r in c[c['role'] == 'control'].iterrows():
        rows.append((r['report_family'], SHORT[r['model']], f"control: {LB.option_label(r['option'])} vs the current "
                     'pipeline', r['acc'], r['acc_base'], r))
    out = [dict(zip(PAIR_COLUMNS, (m, what, fmt(a, '.1f', 100), fmt(b, '.1f', 100),
                                   est(r['d_acc'], r['d_acc_lo'], r['d_acc_hi'], '+.1f'), pval(r['q_bh']),
                                   est(r['d_logloss'], r['d_logloss_lo'], r['d_logloss_hi'], '+.3f', 1.0),
                                   verdict_short(r['verdict'])))) for _, m, what, a, b, r in rows]
    return pd.DataFrame(out, columns=list(PAIR_COLUMNS))


# ───────────────────────────── panel and calibration ─────────────────────────────


PANEL_COLUMNS = ('Model', 'Accuracy % [95% CI]', 'AUC [95% CI]', 'Log loss [95% CI]', 'ECE (null 95th pct.)',
                 'Slope [95% CI]', 'k = 3 %', 'Natural use')


def _nat(r: Any) -> str:
    """Natural use: answered accuracy, coverage, half credit (a tie never counts as a right-hand answer)."""
    if not _ok(r.get('natural_use_n')) or not r['natural_use_n']:
        return '--'
    return (f"{fmt(r['natural_use_answered_accuracy'], '.0f', 100)}% correct ({fmt(r['natural_use_coverage'], '.0f', 100)}% "
            f"answered); half credit {fmt(r['natural_use_half_credit'], '.0f', 100)}%")


def panel_table() -> pd.DataFrame:
    rj = L.read_json('p1_runs')
    rb = {r['name']: r for r in rj['runs']}
    hl = rj.get('headline', {})
    rows = []
    names = [(f'p1_{m}_base', SHORT[m]) for m in MODELS] + [(f'p1_calib_{m}_auto', f'{SHORT[m]} + nested calibrator')
                                                           for m in LB.CALIB_MODELS]
    for name, label in names:
        if name not in rb or name not in hl:
            continue
        r, b = rb[name], hl[name]
        rows.append(dict(zip(PANEL_COLUMNS, (
            label, est(b['accuracy'], b['accuracy_lo'], b['accuracy_hi']),
            est(b['roc_auc'], b['roc_auc_lo'], b['roc_auc_hi'], '.3f', 1.0),
            est(b['log_loss'], b['log_loss_lo'], b['log_loss_hi'], '.3f', 1.0),
            f"{fmt(b['ece'], '.3f')} ({fmt(b['ece_null_q95'], '.3f')})",
            est(b['slope'], b['slope_lo'], b['slope_hi'], '.2f', 1.0),
            est(r.get('acc_k3'), r.get('acc_k3_lo'), r.get('acc_k3_hi'), '.1f'), _nat(r)))))
    return pd.DataFrame(rows, columns=list(PANEL_COLUMNS))


CALIB_COLUMNS = ('Model', 'Calibrator', 'Log loss [95% CI]', 'Brier [95% CI]', 'AUC', 'ECE (null 95th pct.; p)',
                 'Slope [95% CI]')


def calibration_table() -> pd.DataFrame:
    cal = L.read_json('p1_calibration')
    rows = []
    for key, blk in cal['methods'].items():
        m, meth = key.split(':')
        o = blk['overall']
        rows.append(dict(zip(CALIB_COLUMNS, (
            SHORT[m], METHOD_LABELS.get(meth, meth), est(o['log_loss'], o['log_loss_lo'], o['log_loss_hi'], '.3f', 1.0),
            est(o['brier'], o['brier_lo'], o['brier_hi'], '.3f', 1.0), fmt(o['roc_auc'], '.3f'),
            f"{fmt(o['ece'], '.3f')} ({fmt(o['ece_null_q95'], '.3f')}; {pval(o['ece_p'])})",
            est(o['slope'], o['slope_lo'], o['slope_hi'], '.2f', 1.0)))))
    return pd.DataFrame(rows, columns=list(CALIB_COLUMNS))


CALIB_PAIRED_COLUMNS = ('Model', 'Calibrator', 'Diff. log loss', 'Diff. Brier', 'Diff. AUC', 'Diff. slope',
                        'Diff. acc. (points)')


def calibration_paired_table() -> pd.DataFrame:
    """Every calibrator minus the uncalibrated model on the same trials: pooled metrics, paired person bootstrap."""
    cal = L.read_json('p1_calibration')
    rows = []
    for key, pr in cal['paired'].items():
        m, meth = key.split(':')
        cell = lambda met, spec, sc=1.0: est(pr[met]['diff'], pr[met]['lo'], pr[met]['hi'], spec, sc)  # noqa: E731
        rows.append(dict(zip(CALIB_PAIRED_COLUMNS, (SHORT[m], METHOD_LABELS.get(meth, meth), cell('log_loss', '+.3f'),
                                                    cell('brier', '+.3f'), cell('roc_auc', '+.3f'),
                                                    cell('slope', '+.2f'), cell('accuracy', '+.1f', 100.0)))))
    return pd.DataFrame(rows, columns=list(CALIB_PAIRED_COLUMNS))


CALIB_PF_COLUMNS = ('Model', 'Calibrator', 'Platform', 'Trials', 'Pers.', 'Log loss [95% CI]', 'Brier [95% CI]',
                    'ECE (null 95th pct.)', 'Slope [95% CI]')


def calibration_platform_table() -> pd.DataFrame:
    """Per test platform: log loss, Brier, ECE with its null and the slope of every calibrator (CIs within the
    platform; none below MIN_PERSONS_CI persons)."""
    cal = L.read_json('p1_calibration')
    rows = []
    for key, blk in cal['methods'].items():
        m, meth = key.split(':')
        for pf in sorted(k for k in blk if k != 'overall'):
            o = blk[pf]
            rows.append(dict(zip(CALIB_PF_COLUMNS, (
                SHORT[m], METHOD_LABELS.get(meth, meth), pf, int(o['n']), int(o['n_persons']),
                est(o['log_loss'], o['log_loss_lo'], o['log_loss_hi'], '.3f', 1.0),
                est(o['brier'], o['brier_lo'], o['brier_hi'], '.3f', 1.0),
                f"{fmt(o['ece'], '.3f')} ({fmt(o['ece_null_q95'], '.3f')})",
                est(o['slope'], o['slope_lo'], o['slope_hi'], '.2f', 1.0)))))
    return pd.DataFrame(rows, columns=list(CALIB_PF_COLUMNS))


def calibration_tests_table() -> pd.DataFrame:
    cal = L.read_json('p1_calibration')
    return pd.DataFrame([{'Test': t['test'], 'p': pval(t['p']), 'q (BH)': pval(t['q_bh']), 'p (Holm)': pval(t['p_holm'])}
                         for t in cal.get('tests', [])], columns=['Test', 'p', 'q (BH)', 'p (Holm)'])


AGG_TABLE_COLUMNS = ('Model', 'k', 'Rule', 'Acc. %', 'Log loss [95% CI]', 'Brier [95% CI]', 'ECE', 'Slope [95% CI]')


def aggregates_table() -> pd.DataFrame:
    cal = L.read_json('p1_calibration')
    rows = []
    for m, a in cal['aggregates'].items():
        for r in a['rows']:
            rows.append(dict(zip(AGG_TABLE_COLUMNS, (
                SHORT[m], int(r['k']), LB.RULE_LABELS[r['rule']], fmt(r['accuracy'], '.1f', 100),
                est(r['log_loss'], r['log_loss_lo'], r['log_loss_hi'], '.3f', 1.0),
                est(r['brier'], r['brier_lo'], r['brier_hi'], '.3f', 1.0), fmt(r['ece'], '.3f'),
                est(r['slope'], r['slope_lo'], r['slope_hi'], '.2f', 1.0)))))
    return pd.DataFrame(rows, columns=list(AGG_TABLE_COLUMNS))


AGG_PAIRED_COLUMNS = ('Model', 'k', 'Sum of logits / T_k minus', 'Diff. log loss [95% CI]', 'Diff. Brier [95% CI]', 'p')


def aggregates_paired_table() -> pd.DataFrame:
    cal = L.read_json('p1_calibration')
    rows = []
    for m, a in cal['aggregates'].items():
        for r in a.get('paired', []):
            if r['k'] == 1:
                continue
            rows.append(dict(zip(AGG_PAIRED_COLUMNS, (
                SHORT[m], int(r['k']), LB.RULE_LABELS[r['against']],
                est(r['d_log_loss'], r['d_log_loss_lo'], r['d_log_loss_hi'], '+.3f', 1.0),
                est(r['d_brier'], r['d_brier_lo'], r['d_brier_hi'], '+.3f', 1.0), pval(r.get('p_boot_log_loss'))))))
    return pd.DataFrame(rows, columns=list(AGG_PAIRED_COLUMNS))


# ───────────────────────────── subsets and conditions ─────────────────────────────


SUBSET_KINDS = ('base', 'excluded', 'threshold', 'flag', 'swap')


def subsets_table() -> pd.DataFrame:
    s = pd.DataFrame(L.read_json('p1_subsets')['rows'])
    s = s[s['kind'].isin(SUBSET_KINDS)]
    return pd.DataFrame({'Model': s['model'].map(SHORT), 'Trials': s['label'].map(LB.subset_label),
                         'n': s['n'].astype(int), 'Pers.': s['n_persons'].astype(int),
                         'Accuracy % [95% CI]': [est(a, b, c) for a, b, c in zip(s['accuracy'], s['lo'], s['hi'])],
                         'Against the primary accuracy': s['status']},
                        columns=['Model', 'Trials', 'n', 'Pers.', 'Accuracy % [95% CI]', 'Against the primary accuracy'])


def conditions_table() -> pd.DataFrame:
    b = L.read_json('p1_best')
    c = pd.DataFrame(b['conditions'])
    return pd.DataFrame({'Condition': c['condition'], 'Level': c['level'], 'Trials': c['n'].astype(int),
                         'Pers.': c['n_persons'].astype(int),
                         'Accuracy % [95% CI]': [est(a, d, e) for a, d, e in zip(c['accuracy'], c['lo'], c['hi'])]},
                        columns=['Condition', 'Level', 'Trials', 'Pers.', 'Accuracy % [95% CI]'])


# ───────────────────────────── captions (computed, no typed counts) ─────────────────────────────


def _N() -> Any:
    from stage2.lanes.p1_numbers import Numbers
    return Numbers()


def _verdict_note(N: Any) -> str:
    return (f"Verdict: * = the paired 95% CI excludes 0 and the Benjamini-Hochberg q of the exact sign-flip test "
            f"(within the family) is below {N('alpha')}; nom. = the CI alone excludes 0; identical = predictions "
            f"within {N('identical_tol')} of each other.")


def cap_decisions() -> str:
    N = _N()
    return ('Data-treatment decisions of the pipeline, judged by a fixed panel of three pre-specified models (LR: '
            'logistic on chiral bins; GBM: LightGBM on summary features; MR: MiniRocket + ridge). Options: how many '
            'alternatives were tested, on which models. Evidence: paired leave-one-person-out accuracy differences '
            f"against the current default ({N('n_persons')} persons; 95% person-bootstrap CIs; q: Benjamini-Hochberg "
            f"over the {N('fam_options_size')} option comparisons); nominal: the CI excludes 0 but q does not pass. "
            'Gain forgone at most: the largest upper CI bound of any option, i.e. the most the default could be '
            'beaten by. Rule (pre-specified): change a default only if one option is better for at least two panel '
            'models and worse for none; an option run on one model cannot meet it.')


def cap_options(part: str) -> str:
    N = _N()
    return (f'Data-treatment options against the current pipeline, per panel model ({part}). Paired on the same '
            'trials and folds: pooled accuracy difference in points with its 95% person-cluster bootstrap CI; q: '
            f"Benjamini-Hochberg over the {N('fam_options_size')} option comparisons (exact sign-flip test over persons); "
            'Diff. LL: mean per-person log-loss difference (negative = better). ' + _verdict_note(N))


def cap_robustness() -> str:
    N = _N()
    return ('Robustness to segmentation errors: models trained on the oracle bounds, tested on bounds moved by '
            'independent uniform errors (or an onset 100 ms late), against the same models on the oracle bounds; q: '
            f"Benjamini-Hochberg over the {N('fam_robustness_size')} robustness comparisons. Last rows: training with "
            'jittered copies judged against oracle-only training, both tested on the same jittered bounds (q in the '
            'family of pairs). ' + _verdict_note(N))


def cap_pairs() -> str:
    N = _N()
    return ('Remedies against the problem they address, options against their size-matched controls (as many '
            'training trials per person and hand, or as many training persons, removed at random), and the controls '
            f"against the current pipeline; one family of {N('fam_pairs_size')} comparisons (q: Benjamini-Hochberg). "
            + _verdict_note(N))


def cap_panel() -> str:
    N = _N()
    nat_all_right = N.ok('gbm_nat_n_right') and N.value('gbm_nat_n_right') == N.value('gbm_nat_n')
    return ('The fixed panel under the current pipeline, and with the calibrator chosen inside each training fold '
            f"(leave-one-person-out, {N('n_persons')} persons; 95% person-cluster bootstrap CIs). ECE: "
            f"{N('ece_bins')} equal-mass bins, with the 95th percentile of the ECE of perfectly calibrated predictions "
            'on the same bins (no CI: the percentile bootstrap of the ECE is biased); slope: calibration slope (1 = '
            'calibrated, below 1 = over-confident); k = 3: mean P(right) of three attempts of one person and hand. '
            'Natural use: one free-choice trial per person, scored apart; a P(right) of exactly 0.5 is not an answer'
            + ('; every natural-use trial is right-handed, so the answered accuracy is right-hand recall.' if nat_all_right
               else '.'))


def cap_calibration() -> str:
    N = _N()
    return ('Calibrators fitted inside every training fold on inner cross-fitted predictions (GroupKFold over the '
            'training persons) and applied to the held-out person. Nested: the calibrator (or none) chosen inside each '
            'fold by grouped cross-validation among the rank-preserving candidates; the fixed calibrators are compared '
            f"on the outer folds and are exploratory ({N('n_fixed_calibrators')} configurations). ECE with the 95th "
            'percentile and the p-value of the ECE under perfect calibration (trials treated as independent). '
            '95% person-cluster bootstrap CIs.')


def cap_calibration_paired() -> str:
    return ('Each calibrator minus the uncalibrated model on the same trials: pooled log loss, Brier score, AUC, '
            'calibration slope and accuracy, with paired person-bootstrap 95% CIs (both arms on the same resampled '
            'persons). A negative log-loss or Brier difference is an improvement; a slope difference above 0 '
            'corrects over-confidence; a strictly increasing calibrator cannot change the ranking within a fold.')


def cap_calibration_platform() -> str:
    N = _N()
    return ('Calibration per test platform (95% person-cluster bootstrap CIs within the platform; none below '
            f"{N('min_persons_ci')} persons).")


def cap_calibration_tests() -> str:
    return ('The calibration test family: the ECE of each uncalibrated panel model against perfect calibration, and '
            'the change of the calibration slope by its nested calibrator (two-sided bootstrap p); '
            'Benjamini-Hochberg q and Holm-adjusted p within the family.')


def cap_aggregates() -> str:
    N = _N()
    m = next((x for x in LB.CALIB_MODELS if N.ok(f'agg_{x}_n_persons')), LB.CALIB_MODELS[0])
    return ('Calibration of $k$-attempt aggregates ($k$ random attempts of one person and hand; $T_k$ fitted on the '
            f"training persons of each fold), on the same {N(f'agg_{m}_n_persons')} persons at every $k$; every person "
            'and hand weighs the same. ECE without a CI (biased percentile bootstrap); 95% person-bootstrap CIs.')


def cap_aggregates_paired() -> str:
    return ('Paired differences between aggregation rules (sum of the logits divided by the training-fold $T_k$ minus '
            'each other rule), with person-bootstrap 95% CIs and the two-sided bootstrap p of the log-loss '
            'difference; negative = the $T_k$ rule is better.')


def cap_subsets() -> str:
    N = _N()
    return ('Accuracy of the panel models on the trials that an exclusion rule removes (scored by models trained on '
            'exactly the primary training set unless stated), that thresholds twice as strict would remove, or that '
            f"a flag marks; 95% person-cluster bootstrap CI only for subsets of at least {N('min_persons_ci')} persons. "
            'Status: CI below / above the model\'s primary accuracy, inconclusive when it covers both chance and the '
            'primary accuracy.')


def cap_conditions() -> str:
    N = _N()
    return (f"{N('best_label')} (the best panel model by outer-fold accuracy: an exploratory designation among "
            f"{N('best_n_configs_tried')}) per instructed condition, stratified after prediction (conditions are "
            'never inputs; 95% person-cluster bootstrap CIs).')


SMALL = r'\scriptsize'
#: wide tables: script size and narrower column gaps (scoped to the float), so they fit the report's text width
NARROW = r'\scriptsize\setlength{\tabcolsep}{4pt}'
#: name -> (builder, caption (callable), extra booktabs options)
TABLES: dict[str, tuple[Any, Any, dict[str, Any]]] = {
    'p1_decisions': (decisions_table, cap_decisions,
                     dict(size=NARROW, align=RAG + 'p{1.9cm}' + RAG + 'p{2.4cm}' + RAG + 'p{1.2cm}' + RAG + 'p{5.2cm}'
                          + RAG + 'p{1.7cm}' + RAG + 'p{1.4cm}')),
    'p1_decisions_by_model': (decisions_by_model_table,
                              lambda: ('Data-treatment choices per panel model: range of the paired accuracy '
                                       'differences against the current default over the options tested (points), the '
                                       'number of options, and how many are worse or better after the Benjamini-Hochberg '
                                       'correction (*), only nominally, or identical.'),
                              dict(size=r'\footnotesize', align=RAG + 'p{2.4cm}' + RAG + 'p{3.0cm}' + RAG + 'p{3.0cm}'
                                   + RAG + 'p{2.4cm}' + RAG + 'p{2.4cm}')),
    'p1_options_a': (lambda: options_table('a'), lambda: cap_options('recording and representation'),
                     dict(size=NARROW, align=RAG + 'p{1.9cm}' + RAG + 'p{2.3cm}lr' + RAG + 'p{2.1cm}r' + RAG + 'p{2.4cm}l')),
    'p1_options_b': (lambda: options_table('b'), lambda: cap_options('windows, normalisation, data handling, '
                                                                     'exclusions, platform and augmentation'),
                     dict(size=NARROW, align=RAG + 'p{1.9cm}' + RAG + 'p{2.3cm}lr' + RAG + 'p{2.1cm}r' + RAG + 'p{2.4cm}l')),
    'p1_robustness': (robustness_table, cap_robustness,
                      dict(size=SMALL, align='l' + RAG + 'p{3.6cm}r' + RAG + 'p{2.2cm}r' + RAG + 'p{2.6cm}l')),
    'p1_pairs': (pairs_table, cap_pairs,
                 dict(size=NARROW, align='l' + RAG + 'p{3.2cm}rr' + RAG + 'p{2.1cm}r' + RAG + 'p{2.2cm}l')),
    'p1_panel': (panel_table, cap_panel,
                 dict(size=NARROW, align=RAG + 'p{1.5cm}' + RAG + 'p{1.6cm}' + RAG + 'p{1.6cm}' + RAG + 'p{1.6cm}'
                      + RAG + 'p{1.2cm}' + RAG + 'p{1.5cm}' + RAG + 'p{1.6cm}' + RAG + 'p{2.4cm}')),
    'p1_calibration': (calibration_table, cap_calibration,
                       dict(size=SMALL, align='ll' + RAG + 'p{2.4cm}' + RAG + 'p{2.4cm}r' + RAG + 'p{2.3cm}'
                            + RAG + 'p{2.1cm}')),
    'p1_calibration_paired': (calibration_paired_table, cap_calibration_paired,
                              dict(size=NARROW, align='ll' + RAG + 'p{2.2cm}' + RAG + 'p{2.2cm}' + RAG + 'p{2.2cm}'
                                   + RAG + 'p{2.0cm}' + RAG + 'p{2.0cm}')),
    'p1_calibration_platform': (calibration_platform_table, cap_calibration_platform,
                                dict(size=NARROW, align='lllrr' + RAG + 'p{2.2cm}' + RAG + 'p{2.2cm}' + RAG + 'p{1.4cm}'
                                     + RAG + 'p{1.9cm}')),
    'p1_calibration_tests': (calibration_tests_table, cap_calibration_tests,
                             dict(size=r'\footnotesize', align=RAG + 'p{9.5cm}rrr')),
    'p1_aggregates': (aggregates_table, cap_aggregates,
                      dict(size=SMALL, align='lr' + RAG + 'p{2.4cm}r' + RAG + 'p{2.4cm}' + RAG + 'p{2.4cm}r'
                           + RAG + 'p{2.0cm}')),
    'p1_aggregates_paired': (aggregates_paired_table, cap_aggregates_paired,
                             dict(size=r'\footnotesize', align='lr' + RAG + 'p{3.0cm}' + RAG + 'p{3.0cm}' + RAG
                                  + 'p{3.0cm}r')),
    'p1_subsets': (subsets_table, cap_subsets,
                   dict(size=SMALL, align='l' + RAG + 'p{5.0cm}rr' + RAG + 'p{2.4cm}' + RAG + 'p{3.4cm}')),
    'p1_conditions': (conditions_table, cap_conditions, dict(size=r'\footnotesize')),
}


def write(only: list[str] | None = None) -> None:
    """Every report table from the derived files (the registry results/registry_p1.csv is rebuilt by compute). A full
    run also removes report tables of earlier versions that are no longer built (``p1_*.csv`` / ``.tex`` only)."""
    if only is None:
        for f in list(L.tab_dir.glob('*.csv')) + list(L.tab_dir.glob('*.tex')):
            if f.name.startswith('p1_') and f.stem not in TABLES:     # case-sensitive: figure data are P1_*.csv
                f.unlink()
    for name, (fn, caption, kw) in TABLES.items():
        if not selected(name, only):
            continue
        df = fn()
        L.export_table(df, name, caption=caption(), column_formats={}, **kw)


# ───────────────────────────── numbers, finalists, summary ─────────────────────────────


def numbers() -> None:
    from stage2.lanes import p1_numbers, p1_summary
    entries = p1_numbers.build()
    L.write_numbers(entries)
    write_finalists(entries)
    p1_summary.write(entries)


def write_finalists(entries: dict[str, dict[str, Any]]) -> None:
    """Three pipelines, each runnable as-is: ``import_module(module).factory(**params)`` -> fit_predict,
    ``build_x(load_cache())`` -> (N, 17, 64). The GBM is offered with its nested calibrator (its plain version is
    ``factory(model='gbm')``, results/p1_gbm_base.json); LR and MiniRocket as pre-specified references."""
    from stage2.lanes.p1_numbers import Numbers
    N = Numbers(entries)
    fin = []
    spec = (('p1_calib_gbm_auto', 'gbm_cal', {'model': 'gbm', 'calibration': 'auto'},
             f"{MODEL_LABELS['gbm']} with the calibrator chosen inside every training fold (none, temperature, Platt "
             "or beta, by grouped CV on the training persons' cross-fitted predictions); the plain model is "
             "factory(model='gbm') (results/p1_gbm_base.json)."),
            ('p1_lr_base', 'lr', {'model': 'lr'}, f"{MODEL_LABELS['lr']}: the physics reference of the panel."),
            ('p1_mr_base', 'mr', {'model': 'mr'}, f"{MODEL_LABELS['mr']} (numpy MiniRocket; ridge alpha and Platt "
                                                  'scaling chosen by inner GroupKFold).'))
    for name, h, params, what in spec:
        if not N.ok(f'{h}_accuracy'):
            continue
        note = (f"{what} Lane p1 panel, current data treatment; pre-specified, nothing chosen on the outer folds. "
                f"LOSO accuracy {N(h + '_accuracy')} {N.ci(h + '_accuracy')}, AUC {N(h + '_auc')}, log loss "
                f"{N(h + '_log_loss')}, ECE {N(h + '_ece')}, calibration slope {N.est(h + '_slope')}.")
        fin.append({'name': name, 'results_json': f'results/{name}.json', 'module': 'stage2.lanes.p1_panel',
                    'build_x': 'build_x', 'factory': 'factory', 'params': params,
                    'n_configs_tried': int(N.value(f'{h}_n_configs_tried') or 1), 'notes': note})
    L.write_finalists(fin)


__all__ = ['write', 'numbers', 'TABLES', 'options_table', 'decisions_table', 'write_finalists', 'AGG_TABLE_COLUMNS']
