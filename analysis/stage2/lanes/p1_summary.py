"""Lane p1: reports/stage2/p1_SUMMARY.md, generated from the saved derived files (called by the numbers step).

Every number in the summary is printed through :class:`p1_numbers.Numbers`, i.e. it is an entry of
results/p1_numbers.json formatted exactly as its LaTeX macro (a missing key fails the build); the verdict words and
the conclusions are chosen by rules on those values, so a rebuild with more participants rewrites them consistently.
Light imports only (labels, numbers, the derived JSON files).
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from stage2.lanes import p1_labels as LB
from stage2.lanes.p1_labels import MODEL_LABELS, MODELS, SHORT
from stage2.lanes.p1_numbers import Numbers, load, rkey
from stage2.pipeline import Lane

L = Lane('p1')
TABLE_FILES = {
    'p1_decisions': 'main text: one row per data-treatment choice (default, options, evidence, decision)',
    'p1_decisions_by_model': 'appendix: the same choices per panel model',
    'p1_options_a': 'appendix: every option x model, recording and representation (filter, gaps, sampling, gravity, frame)',
    'p1_options_b': 'appendix: every option x model, windows, normalisation, outliers, exclusions, platform, augmentation',
    'p1_robustness': 'appendix: test-time segmentation errors, and the jitter-training remedy pair',
    'p1_pairs': 'appendix: remedies against their problem, options against size-matched controls',
    'p1_panel': 'the panel models and the nested-calibrated ones (headline metrics)',
    'p1_calibration': 'calibrators: log loss, Brier, AUC, ECE with its null, calibration slope',
    'p1_calibration_paired': 'calibrators minus the uncalibrated model, paired person bootstrap',
    'p1_calibration_platform': 'calibrators per test platform',
    'p1_calibration_tests': 'the calibration test family (BH / Holm)',
    'p1_aggregates': 'k-attempt aggregation rules: accuracy, log loss, Brier, ECE, slope',
    'p1_aggregates_paired': 'paired differences between aggregation rules',
    'p1_subsets': 'accuracy on excluded, flagged and swap-signature trials',
    'p1_conditions': 'best panel model per instructed condition (exploratory designation)',
}


# ───────────────────────────── phrasing helpers ─────────────────────────────


def _opt(o: str) -> str:
    return LB.option_label(o)


def dci(N: Numbers, k: str) -> str:
    """'-1.5 [-3.2, +0.0] points' of a comparison key part."""
    return f"{N('d_' + k)} [{N('d_' + k + '_lo')}, {N('d_' + k + '_hi')}] points"


def dq(N: Numbers, k: str) -> str:
    """Accuracy difference with its CI and BH q."""
    return f"{dci(N, k)}, q = {N('q_' + k)}"


def dll(N: Numbers, k: str) -> str:
    return f"{N('dll_' + k)} [{N('dll_' + k + '_lo')}, {N('dll_' + k + '_hi')}]"


def ci3(N: Numbers, k: str) -> str:
    """'value [lo, hi]' of an entry with CI companions."""
    return N.est(k)


def word(v: str) -> str:
    return LB.VERDICT_LABELS.get(v, v)


def _ok(N: Numbers, *keys: str) -> bool:
    return all(N.ok(k) for k in keys)


# ───────────────────────────── loading ─────────────────────────────


class Saved:
    """Every derived file the summary reads, and the numbers formatter."""

    def __init__(self, entries: dict[str, Any] | None = None, mark: bool = False) -> None:
        self.S = load()
        self.N = Numbers(entries, mark=mark)
        self.runs = pd.DataFrame(self.S['p1_runs']['runs'])
        cj = self.S['p1_comparisons']
        self.comp = pd.DataFrame(cj['comparisons'])
        self.rem = pd.DataFrame(cj.get('remedies', []))
        self.ctl = pd.DataFrame(cj.get('controls', []))
        self.subs = pd.DataFrame(self.S['p1_subsets']['rows'])
        self.best = self.S['p1_best']
        self.cal = self.S['p1_calibration']
        self.dec = self.S['p1_decisions']
        p = L.fig_dir / 'captions.json'
        self.caps = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        self.cal_verdict: dict[str, str] = {}       # per calibrated model: 'restored' | 'harmful' | 'neutral'
        self.short: list[str] = []                   # one-line verdicts of the data-handling problems (conclusions)

    def c(self, model: str, option: str) -> Any:
        g = self.comp[(self.comp['model'] == model) & (self.comp['option'] == option)]
        return g.iloc[0] if len(g) else None

    def has(self, name: str) -> bool:
        return bool(len(self.comp)) and name in set(self.comp['name'])


# ───────────────────────────── sections ─────────────────────────────


def method(S: Saved, w: Any) -> None:
    N = S.N
    w('## Method\n')
    w('- **Panel** (`stage2/lanes/p1_panel.py`; three models of different kinds, fixed before any variant was compared, '
      '`n_configs_tried = 1` each):')
    w(f"  - **LR** ({MODEL_LABELS['lr']}): L2 logistic regression (C = {N('lr_c')}) on the standardised means of "
      f"{N('n_bins')} time bins of the {N('lr_n_channels')} chiral gravity-frame channels (`yaw_rate`, `yaw_cum`), "
      'mirror-augmented; the scaler is fitted on the augmented training rows. With mirror augmentation the fitted LR is '
      'antisymmetric (p(mirror x) = 1 - p(x)), so mirror-even inputs get zero weight up to solver precision.')
    w(f"  - **GBM** ({MODEL_LABELS['gbm']}): LightGBM with fixed parameters ({N('gbm_n_estimators')} trees, learning "
      f"rate {N('gbm_learning_rate')}, {N('gbm_num_leaves')} leaves, row subsample {N('gbm_subsample')}, column "
      f"subsample {N('gbm_colsample')}) on {N('gbm_features_per_channel')} summary features per channel (bin means, "
      f"mean, SD, min, max, times of the min and max) of all {N('gbm_n_channels')} channels, mirror-augmented.")
    w(f"  - **MR** ({MODEL_LABELS['mr']}): MiniRocket in numpy (`p1_minirocket.py`; {N('mr_features')} features "
      f"requested, biases fitted on the training rows of each fold) on the {N('mr_n_channels')} gravity-frame channels; "
      f"ridge with alpha chosen by inner GroupKFold ({N('mr_inner_splits')} folds over the training persons) and Platt "
      'scaling fitted on the inner cross-fitted decisions.')
    w('- **Variants** (`p1_variants.py`): the raw records of the snapshot re-standardised with ONE parameter changed '
      '(stream: cut-off, order, causal filter, gap bridging, interpolation, sampling rate, iOS rounding, time jitter; '
      'window: gravity estimate, window, bound errors), cached under `analysis/data/stage2_variants/<variant>/` '
      f"({N('n_variants')} variants) and matched to the core cache by `trial_id`. With every option at its default "
      f"the rebuilt arrays equal the core cache (max |difference| {N('variant_fidelity_max_diff')} over "
      f"{N('variant_fidelity_n_trials')} trials).")
    w('- **Treatments inside `fit_predict`** (`p1_panel.Treatment`): channel sets, normalisation and winsorising (fitted '
      'on the mirror-augmented training rows), augmentation (mirror, screen-away flips, time warp, jittered bounds) and '
      'training-row filters (flags, swap blocks, exclusion thresholds, platform, size-matched random removals). Runs '
      "that put excluded trials back use `sets=('primary', 'excluded')` on purpose; runs that only SCORE excluded "
      'trials train on exactly the base training set (the D2 cap applied after the primary-only filter).')
    w(f"- **Evaluation**: leave-one-person-out over {N('n_persons')} persons and {N('n_trials')} primary trials "
      '(`stage2.cv.run_experiment`, docs/09 D2 cap); natural-use trials scored apart (`extra=`). The leak check runs '
      f"on every fold for the headline runs and on {N('leak_sample_folds')} seeded folds for the other variants. Each "
      'option is compared PAIRED with the same model under the current pipeline (`p1_stats.paired_compare`, equal to '
      f"`stage2.cv.paired_compare`): pooled accuracy difference with a 95 % person-bootstrap CI (B = {N('n_boot')}), "
      'mean per-person log-loss difference with its CI, exact sign-flip p over persons.')
    w(f"- **Multiplicity**: three comparison families, each Benjamini-Hochberg (q) and Holm adjusted: the options "
      f"({N('fam_options_size')} option x model comparisons), the robustness costs ({N('fam_robustness_size')}: "
      f"segmentation errors at test time) and the pairs ({N('fam_pairs_size')}: remedies against their problem, options "
      'against size-matched controls, controls against the pipeline). Verdicts: *better / worse* = the paired CI '
      f"excludes 0 AND q < {N('alpha')}; *nominally better / worse* = only the CI excludes 0; *identical* = every "
      f"prediction within {N('identical_tol')} (solver precision); a log-loss verdict also needs |difference| >= "
      f"{N('logloss_tol')}. A CI that includes 0 is absence of evidence; its lower end is the largest loss the data "
      'allow (the non-inferiority margin quoted below).')
    w('- **Roles** (`p1_labels.py`): an *option* is a data-treatment alternative (the decision rule applies); a '
      '*robustness* row is a test-time perturbation of the current model (a requirement on the production detector, '
      'never an option); a *remedy* is judged only against the problem it addresses; a *scoring* run scores the '
      'trials an exclusion rule removes with the base model; a *control* removes as many training trials or persons '
      'at random as an option does.')
    w(f"- **Decision rule** (pre-specified): {S.dec['rule']}; an option run on one model cannot meet it.")
    w(f"- **Calibration** (`p1_calibration.py`): {N('n_fixed_calibrators')} fixed calibrators (Platt, temperature, "
      'isotonic, beta) fitted inside every outer training fold on cross-fitted predictions of the training persons '
      '(`fold.inner_splits()`) and applied to the held-out person (compared on the outer folds: exploratory); and the '
      f"NESTED choice among {N('n_auto_candidates')} candidates (none, temperature, Platt, beta: strictly increasing, so "
      f"the ranking within a fold is kept) by {N('auto_splits')}-fold grouped CV on those predictions (the headline). "
      'Evidence per model: calibration slope and calibration-in-the-large with person-bootstrap CIs, the ECE against '
      f"its distribution under perfect calibration ({N('ece_null_sims')} simulations on the same predictions and bins; "
      'trials treated as independent), paired person-bootstrap differences of log loss, Brier, AUC and slope.')
    w('- **Multi-attempt aggregates**: random k-subsets of one person and hand, on the SAME persons at every k (those '
      f"with at least {N('k_max')} trials of each hand), every person and hand weighted equally; rules: mean of P, mean "
      'of logits, sum of logits, and the sum divided by a temperature T_k fitted on the training persons of each fold.\n')


def _nat(N: Numbers, h: str) -> str:
    if not N.ok(f'{h}_nat_n') or not N.value(f'{h}_nat_n'):
        return '--'
    right = N.ok(f'{h}_nat_n_right') and N.value(f'{h}_nat_n_right') == N.value(f'{h}_nat_n')
    return (f"{N(h + '_nat_answered')} correct ({N(h + '_nat_coverage')} answered; half credit {N.est(h + '_nat_half')}; "
            f"n = {N(h + '_nat_n')}" + (', all right-handed: right-hand recall' if right else '') + ')')


HEADLINE_ROWS = (('lr', 'LR'), ('gbm', 'GBM'), ('mr', 'MR'), ('gbm_cal', 'GBM + nested calibrator'),
                 ('lr_cal', 'LR + nested calibrator'))


def headline(S: Saved, w: Any) -> None:
    N = S.N
    w('## Headline numbers\n')
    w('| Model | Accuracy [95 % CI] | AUC [CI] | Log loss [CI] | Brier [CI] | ECE (null 95th pct.; p) | Slope [CI] | '
      'k = 3 [CI] | Natural use |')
    w('|---|---|---|---|---|---|---|---|---|')
    for h, lab in HEADLINE_ROWS:
        if not N.ok(f'{h}_accuracy'):
            continue
        w(f"| {lab} | {N.est(h + '_accuracy')} | {N.est(h + '_auc')} | {N.est(h + '_log_loss')} | "
          f"{N.est(h + '_brier')} | {N(h + '_ece')} ({N(h + '_ece_null_q95')}; p = {N(h + '_ece_p')}) | "
          f"{N.est(h + '_slope')} | {N.est(h + '_acc_k3')} | {_nat(N, h)} |")
    bm = N.value('best_model')
    w(f"\nEvery row is pre-specified (`n_configs_tried = 1`; the nested rows choose their calibrator inside each "
      f"training fold). Calling {N('best_label')} the *best* panel model is an outer-fold choice among "
      f"{N('best_n_configs_tried')} models, i.e. exploratory (`n_configs_tried = {N('best_n_configs_tried')}` for that "
      'designation); the synthesis lane re-selects among all lanes inside its own nested loop. ECE: '
      f"{N('ece_bins')} equal-mass bins, no CI (its percentile bootstrap is biased and can exclude the estimate); in "
      'brackets the 95th percentile of the ECE of perfectly calibrated predictions on the same bins and its p. Slope: '
      'calibration slope (1 = calibrated, below 1 = over-confident). k = 3: the harness k-attempt curve (mean P(right) '
      f"of three attempts of one person and hand; {N(bm + '_k_persons')} persons with at least {N('k_max')} trials of "
      'each hand). Natural use: one free-choice trial per person, never trained on; a P(right) of exactly 0.5 would '
      'not count as an answer. Leak check of the headline runs: '
      + '; '.join(f"{lab} {N(h + '_leak')}" for h, lab in HEADLINE_ROWS if N.ok(f'{h}_leak')) + '.\n')


def _family_evidence(S: Saved, fam: str) -> str:
    N = S.N
    c = S.comp[(S.comp['mult_family'] == 'options') & (S.comp['report_family'] == fam)]
    parts = []
    for v in ('worse', 'better', 'nominally worse', 'nominally better'):
        g = c[c['verdict'] == v].sort_values('d_acc')
        if len(g):
            parts.append(f'{v}: ' + '; '.join(f"{_opt(r['option'])} ({SHORT[r['model']]} {dq(N, rkey(r['name']))})"
                                                for _, r in g.iterrows()))
    g = c[c['verdict'] == 'identical']
    if len(g):
        parts.append('identical: ' + ', '.join(f"{_opt(r['option'])} ({SHORT[r['model']]})" for _, r in g.iterrows()))
    if not parts:
        parts.append('no difference for any option')
    return '. '.join(parts)


def decisions(S: Saved, w: Any) -> None:
    N = S.N
    w('## Decisions (`tables/p1/p1_decisions.tex`)\n')
    w(f"Pre-specified rule: {S.dec['rule']}. Gain forgone at most: the largest upper CI bound of any option of the "
      'choice (the most an alternative could beat the default by); largest loss compatible: the lowest lower bound.\n')
    w('| Choice | Current default | Options (comparisons) | Evidence: paired accuracy difference vs the default, q (BH) '
      '| Gain forgone at most | Largest loss compatible | Decision |')
    w('|---|---|---|---|---|---|---|')
    for r in S.dec['rows']:
        f = r['family']
        g, lo = r['max_gain'], r['max_loss']
        dec = N(f'dec_{f}_decision') + ('' if r.get('decidable', True) else ' (no option decidable: one model each)')
        single = N.value(f'dec_{f}_n_single')
        w(f"| {LB.FAMILY_LABELS[f]} | {r['default']} | {N(f'dec_{f}_n_options')} ({N(f'dec_{f}_n_comparisons')}"
          + (f"; {N(f'dec_{f}_n_single')} on one model" if single else '') + f") | {_family_evidence(S, f)} | "
          f"{N(f'dec_{f}_max_gain_hi')} ({_opt(g['option'])}, {SHORT[g['model']]}) | {N(f'dec_{f}_max_loss_lo')} "
          f"({_opt(lo['option'])}, {SHORT[lo['model']]}) | **{dec}** |")
    w('\nRationale per choice (the principle; the numbers above decide):\n')
    keys = {'grip_p': 'grip_chi2_p', 'mr_n_channels': 'mr_n_channels', 'gbm_n_channels': 'gbm_n_channels'}
    for r in S.dec['rows']:
        tpl = r.get('rationale_template', r['rule_rationale'])
        text = tpl.format(**{k: N(keys[k]) for k in r.get('rationale_values', {})})
        w(f"- {LB.FAMILY_LABELS[r['family']]}: {text}.")
    w('')


# ───────────────────────────── problems ─────────────────────────────


class Items:
    def __init__(self, w: Any) -> None:
        self.w, self.k = w, 0

    def __call__(self, title: str, det: str, why: str, rem: str, ev: str) -> None:
        self.k += 1
        self.w(f'{self.k}. **{title}**')
        self.w(f'   - Detected: {det}')
        self.w(f'   - Why it matters: {why}')
        self.w(f'   - Remedy: {rem}')
        self.w(f'   - Evidence: {ev}')


def _below1(N: Numbers, k: str) -> bool:
    return N.ok(f'{k}_hi') and N.value(f'{k}_hi') < 1.0


def _covers1(N: Numbers, k: str) -> bool:
    return _ok(N, f'{k}_lo', f'{k}_hi') and N.value(f'{k}_lo') <= 1.0 <= N.value(f'{k}_hi')


def _ci_side(N: Numbers, k: str) -> int:
    """-1 / +1 when the CI of entry k lies below / above 0, else 0."""
    if not _ok(N, f'{k}_lo', f'{k}_hi'):
        return 0
    return -1 if N.value(f'{k}_hi') < 0 else (1 if N.value(f'{k}_lo') > 0 else 0)


def calibration_problem(S: Saved, it: Items, m: str) -> None:
    N = S.N
    k0, ka = f'cal_{m}_none', f'cal_{m}_auto'
    if not N.ok(f'{k0}_slope'):
        return
    counts = S.cal.get('auto_choices', {}).get(m, {}).get('counts', {})
    over = _below1(N, f'{k0}_slope')
    ece_sig = N.ok(f'{k0}_ece_p') and N.value(f'{k0}_ece_p') < N.value('alpha')
    det = (f"calibration slope {N.est(k0 + '_slope')} ({N(m + '_slope_below1')} "
           f"of the person-bootstrap slopes below 1; calibration-in-the-large {N.est(k0 + '_citl')}); the temperature "
           "fitted on the training persons' cross-fitted predictions exceeds 1 in "
           f"{N(f'cal_{m}_t_above_1')} of {N(f'cal_{m}_t_n_folds')} folds (median {N(f'cal_{m}_t_median')}, range "
           f"{N(f'cal_{m}_t_min')} to {N(f'cal_{m}_t_max')}); ECE {N(k0 + '_ece')} against "
           f"{N(k0 + '_ece_null_q95')} for perfectly calibrated predictions on the same bins (p = {N(k0 + '_ece_p')}, "
           f"BH q = {N(f'caltest_{m}_ece_null_q')} in the calibration test family; trials treated as independent, so "
           'supporting evidence only); per platform ECE Android '
           f"{N(k0 + '_android_ece')} (null {N(k0 + '_android_ece_null_q95')}), iOS {N(k0 + '_ios_ece')} "
           f"(null {N(k0 + '_ios_ece_null_q95')}); reliability diagram `P1_06_reliability`.")
    why = ('the product is the probability shown to the user (the percentage chance of the right hand); an '
           'over-confident P(right) overstates certainty, an under-confident one hides it.')
    if not N.ok(f'{ka}_slope'):
        it(f'Calibration of the single-trial P(right), {SHORT[m]}', det, why, 'none run', '--')
        return
    rem = ("a calibrator chosen INSIDE every training fold among none, temperature, Platt and beta (strictly increasing "
           "maps: the ranking within a fold is kept) by grouped CV on the training persons' cross-fitted predictions, "
           'refitted on all of them; chosen per fold: ' + ', '.join(
               f"{'none' if a == 'none' else LB.CAL_LABELS.get(a, a)} {N(f'cal_{m}_auto_folds_' + a)}"
               for a in counts) + '.')
    ll_side, sl_side = _ci_side(N, f'calpd_{m}_auto_log_loss'), _ci_side(N, f'calpd_{m}_auto_slope')
    auc_side = _ci_side(N, f'calpd_{m}_auto_roc_auc')
    restored = _covers1(N, f'{ka}_slope')
    ev = (f"after the nested calibrator: slope {N.est(ka + '_slope')} (paired change {N.est(f'calpd_{m}_auto_slope')}, "
          f"BH q = {N(f'caltest_{m}_slope_change_q')}); ECE {N(ka + '_ece')} (null {N(ka + '_ece_null_q95')}, p = "
          f"{N(ka + '_ece_p')}); paired differences against the uncalibrated model: log loss "
          f"{N.est(f'calpd_{m}_auto_log_loss')}, Brier {N.est(f'calpd_{m}_auto_brier')}, AUC "
          f"{N.est(f'calpd_{m}_auto_roc_auc')}, accuracy {N.est(f'calpd_{m}_auto_accuracy')} points; per platform ECE "
          f"Android {N(ka + '_android_ece')}, iOS {N(ka + '_ios_ece')}. Fixed temperature (exploratory): slope "
          f"{N.est(f'cal_{m}_temperature_slope')}, log loss {N.est(f'calpd_{m}_temperature_log_loss')}. Fixed isotonic "
          f"(exploratory): ECE {N(f'cal_{m}_isotonic_ece')}, AUC {N.est(f'calpd_{m}_isotonic_roc_auc')}, "
          f"{N(f'cal_{m}_isotonic_n_unique')} distinct values of P(right) out of {N(f'cal_{m}_none_n')} (ties), which is "
          'why isotonic is not a nested candidate. ')
    all_t = N.value(f'cal_{m}_t_above_1') == N.value(f'cal_{m}_t_n_folds')
    t_txt = ('in every training fold' if all_t else
             f"in {N(f'cal_{m}_t_above_1')} of {N(f'cal_{m}_t_n_folds')} training folds")
    iso_trade = (N.ok(f'cal_{m}_isotonic_ece') and N.value(f'cal_{m}_isotonic_ece') < N.value(f'{k0}_ece')
                 and _ci_side(N, f'calpd_{m}_isotonic_roc_auc') < 0)
    if over and sl_side > 0 and restored and ll_side <= 0 and auc_side >= 0:
        S.cal_verdict[m] = 'restored'
        verdict = (f'**over-confident** (slope CI below 1, training-fold temperature above 1 {t_txt}); the in-fold '
                   'calibrator restores a slope of about 1 without costing ranking or accuracy. Use the calibrated '
                   f"{SHORT[m]} as the product P(right). Its log-loss gain "
                   + ('is significant.' if ll_side < 0 else 'is within noise at this number of persons.'))
    elif ll_side > 0:
        S.cal_verdict[m] = 'harmful'
        verdict = (('**no clear miscalibration in the slope** (CI covers 1)' if not over else '**over-confident**')
                   + (', although the ECE exceeds its perfect-calibration null' if ece_sig else '')
                   + '; the nested calibrator does NOT help (paired log-loss CI above 0): leave the '
                   f'{SHORT[m]} uncalibrated.'
                   + (' Its deviations are not a monotone distortion that a rank-preserving map can remove: isotonic '
                      'lowers the ECE but ties predictions and costs ranking (AUC CI below 0).' if iso_trade else ''))
    else:
        S.cal_verdict[m] = 'neutral'
        verdict = (('**over-confident**' if over else '**no clear miscalibration in the slope** (CI covers 1)')
                   + '; the nested calibrator changes log loss within noise; '
                   + ('keep it as a safeguard.' if restored else 'it does not restore a slope of 1.'))
    it(f'Calibration of the single-trial P(right), {SHORT[m]}', det, why, rem, ev + 'Verdict: ' + verdict)


def aggregates_problem(S: Saved, it: Items, m: str) -> None:
    N = S.N
    a = S.cal.get('aggregates', {}).get(m)
    if not a:
        return
    k = int(N.value('k_max'))
    b = f'agg_{m}_k{k}'
    if not N.ok(f'{b}_sum_logit_log_loss'):
        return
    det = (f"{MODEL_LABELS[m]}, {N('k_max')} attempts of one person and hand ({N(f'agg_{m}_n_persons')} persons, the "
           f"same at every k): calibration slope of the summed logits (the rule for independent attempts) "
           f"{N.est(b + '_sum_logit_slope')}, of the mean of P {N.est(b + '_mean_prob_slope')}, of the mean of logits "
           f"{N.est(b + '_mean_logit_slope')}; log loss {N.est(b + '_sum_logit_log_loss')}, "
           f"{N.est(b + '_mean_prob_log_loss')} and {N.est(b + '_mean_logit_log_loss')}; the training-fold temperature of "
           f"the summed logits is {N(f'cal_{m}_t_sum_k{k}')} (median over folds; {N('k_max')} would be the mean of the "
           'logits) (`P1_07_aggregates`).')
    why = ('the inference app may combine several attempts; attempts of one person are correlated, so summing their '
           'evidence overstates the certainty, while averaging probabilities understates it.')
    rem = 'divide the summed logits by the temperature T_k fitted on the training persons of each fold.'
    sides = [_ci_side(N, f'aggd_{m}_k{k}_{x}_ll') for x in ('sum_logit', 'mean_logit', 'mean_prob')]
    ev = (f"sum of logits / T_k: slope {N.est(b + '_sum_logit_t_slope')}, log loss {N.est(b + '_sum_logit_t_log_loss')}, "
          f"ECE {N(b + '_sum_logit_t_ece')}; paired log-loss differences of this rule minus the sum of logits "
          f"{N(f'aggd_{m}_k{k}_sum_logit_ll')} [{N(f'aggd_{m}_k{k}_sum_logit_ll_lo')}, "
          f"{N(f'aggd_{m}_k{k}_sum_logit_ll_hi')}], minus the mean of logits {N(f'aggd_{m}_k{k}_mean_logit_ll')} "
          f"[{N(f'aggd_{m}_k{k}_mean_logit_ll_lo')}, {N(f'aggd_{m}_k{k}_mean_logit_ll_hi')}], minus the mean of P "
          f"{N(f'aggd_{m}_k{k}_mean_prob_ll')} [{N(f'aggd_{m}_k{k}_mean_prob_ll_lo')}, "
          f"{N(f'aggd_{m}_k{k}_mean_prob_ll_hi')}]. ")
    rules = ('sum_logit_t', 'sum_logit', 'mean_logit', 'mean_prob')
    dist = {r: abs(N.value(f'{b}_{r}_slope') - 1.0) for r in rules if N.ok(f'{b}_{r}_slope')}
    closest = min(dist, key=dist.get) if dist else None
    why_t = ('its slope is the closest to 1' if closest == 'sum_logit_t' else
             'it is the Bayes rule for correlated attempts, with the correlation measured on the training persons')
    if all(s == 0 for s in sides):
        ev += (f"Verdict: the T_k rule is the principled choice ({why_t}), but with "
               f"{N(f'agg_{m}_n_persons')} persons it is not distinguishable from the mean of the logits or the other "
               'rules by log loss; the app should report sum / T_k, with T_k refitted when the training data change.')
    else:
        better = [x for x, s in zip(('the sum of logits', 'the mean of logits', 'the mean of P'), sides) if s < 0]
        ev += ('Verdict: the T_k rule has a lower log loss than ' + ', '.join(better) + ' (paired CI below 0); the '
               'app should report sum / T_k.' if better else 'Verdict: the T_k rule does not beat the other rules.')
    it(f'Calibration of multi-attempt aggregates, {SHORT[m]}', det, why, rem, ev)


def robustness_problem(S: Saved, it: Items) -> None:
    N = S.N
    rob = S.comp[S.comp['mult_family'] == 'robustness']
    if not len(rob):
        return
    parts = []
    for m in MODELS:
        g = rob[rob['model'] == m]
        if len(g):
            parts.append(f'{SHORT[m]}: ' + '; '.join(f"{_opt(r['option'])} {dq(N, rkey(r['name']))} ({r['verdict']})"
                                                     for _, r in g.sort_values('value').iterrows()))
    rem_rows = S.rem[S.rem['family'] == 'segmentation'] if len(S.rem) else S.rem
    ev = '; '.join(f"{SHORT[r['model']]} jitter-trained minus oracle-trained, both at the jittered bounds: "
                   f"{dq(N, 'rem_' + rkey(r['name']))} ({r['verdict']})" for _, r in rem_rows.iterrows())
    helps = len(rem_rows) and (rem_rows['verdict'].isin(['better', 'nominally better'])).any()
    worst = rob.loc[rob['d_acc_lo'].idxmin()]
    S.short.append('segment bounds: ' + ('train with jittered bounds' if helps else
                                         'jitter training does not help; a production detector must keep its bound '
                                         f"errors small (up to {N('d_' + rkey(worst['name']) + '_lo')} points lost at "
                                         f"{_opt(worst['option'])}, {SHORT[worst['model']]})"))
    it('Sensitivity to segmentation errors (a robustness cost, not an option)',
       'models trained on the oracle bounds, tested on bounds moved by independent uniform errors or an onset 100 ms '
       'late (`P1_03_segmentation`, `p1_robustness`): ' + ' | '.join(parts) + '.',
       'production has no GO cue: a cue-free detector finds the approach with errors of tens to hundreds of ms.',
       'training with jittered copies of the bounds (same trials, bounds moved by up to 100 ms), judged only against '
       'oracle-only training at the same jittered test bounds (the remedy pair).',
       (ev or '--') + '. Verdict: ' + ('jitter training helps; adopt it for a cue-free detector.' if helps else
                                        'jitter training does not help, so the default (oracle bounds in training) '
                                        'stays; the cost of bound errors is a requirement for the production detector: '
                                        f"the largest loss compatible with the data is {N('d_' + rkey(worst['name']) + '_lo')} "
                                        f"points ({_opt(worst['option'])}, {SHORT[worst['model']]})."))


def platform_problem(S: Saved, it: Items) -> None:
    N = S.N
    if not (S.has('p1_lr_platform_cross') or S.has('p1_gbm_platform_cross')):
        return
    models = [m for m in MODELS if S.has(f'p1_{m}_platform_cross')]
    det = '; '.join(f"{SHORT[m]} other platform only {dq(N, f'{m}_platform_cross')} ({N(f'v_{m}_platform_cross')}), "
                    f"same platform only {dq(N, f'{m}_platform_same')} ({N(f'v_{m}_platform_same')})" for m in models)
    per = []
    for m in models:
        for pf in ('android', 'ios'):
            ks = [f'sub_{m}_pf_{pf}_{t}' for t in ('pooled', 'same', 'same_ctl', 'cross', 'cross_ctl')]
            if all(N.ok(k) for k in ks):
                per.append(f"{SHORT[m]} on {pf}: pooled {N(ks[0])} (median {N(ks[0] + '_ntrain')} training trials), "
                           f"same platform {N(ks[1])} ({N(ks[1] + '_ntrain')}) vs its control {N(ks[2])} "
                           f"({N(ks[2] + '_ntrain')}), other platform {N(ks[3])} ({N(ks[3] + '_ntrain')}) vs its control "
                           f"{N(ks[4])} ({N(ks[4] + '_ntrain')})")
    ctl = '; '.join(f"{SHORT[m]} {side} platform minus its control {dq(N, f'ctl_{m}_platform_{side}')} "
                    f"({N(f'v_ctl_{m}_platform_{side}')})" for m in models for side in ('same', 'cross')
                    if N.ok(f'd_ctl_{m}_platform_{side}'))
    vs = [(m, side, N.value(f'v_ctl_{m}_platform_{side}')) for m in models for side in ('same', 'cross')
          if N.ok(f'v_ctl_{m}_platform_{side}')]
    flagged = [f'{SHORT[m]} {side}-platform training' for m, side, v in vs if v in ('worse', 'nominally worse')]
    sig = [m for m, side, v in vs if v == 'worse']
    verdict = ('no platform effect beyond the smaller training set (every option-minus-control CI includes 0): '
               'single-platform training only loses data. ' if not flagged else
               f"a platform effect beyond the training-set size for {', '.join(flagged)} "
               + ('(significant after the correction). ' if sig else '(nominal only: not significant after the '
                                                                      'correction). '))
    it('Platform transfer (confounded with training-set size, separated by size-matched controls)',
       det + ' (`P1_08_platform`).',
       'Android and iOS differ in sensors, rates, gaps and sensor fusion; a model that only transfers within a platform '
       'would need per-platform training data. Training on one platform also shrinks the training set, so the raw '
       'differences mix the two effects.',
       'keep pooling both platforms (platform is a stratum, never an input of the panel); compare each single-platform '
       'option with a control that removes as many training persons at random.',
       'option minus control: ' + (ctl or '--') + '. Per test platform (accuracy; median training trials per fold): '
       + ('; '.join(per) or '--') + '. Verdict: ' + verdict + 'Pooling stays.')
    S.short.append('platforms: pool them' + ('' if not flagged else f" ({', '.join(flagged)} loses more than its "
                                                                     'size-matched control'
                                                                     + (')' if sig else ', nominally)')))


def exclusions_problem(S: Saved, it: Items) -> None:
    N = S.N
    models = [m for m in ('lr', 'gbm') if N.ok(f'sub_{m}_e4d_all_linear')]
    if not models:
        return
    det = (f"rule E4d removes {N('e4d_n')} instructed trials of {N('e4d_n_persons')} persons (an unbridged gap of more "
           f"than {N('max_gap_ms')} ms near the movement; Android {N('e4d_n_android') if N.ok('e4d_n_android') else '--'}, "
           f"iOS {N('e4d_n_ios') if N.ok('e4d_n_ios') else '--'}); the gap falls inside the approach in "
           f"{N('e4d_n_inapp')} of them ({N('e4d_n_persons_inapp')} persons; median longest gap "
           f"{N('e4d_inapp_gap_median_ms')} ms) and only in the {N('gap_window_pad_s')}-s margins in {N('e4d_n_margin')}. "
           "Scored by the base "
           'models after imputation (`P1_09_exclusions`, `p1_subsets`): '
           + '; '.join(f"{SHORT[m]}: margins only {N.est(f'sub_{m}_e4d_margin_linear')} ({N(f'sub_{m}_e4d_margin_linear_status')}), "
                       f"gap inside the approach, linear {N.est(f'sub_{m}_e4d_inapp_linear')} "
                       f"({N(f'sub_{m}_e4d_inapp_linear_status')}), cubic {N.est(f'sub_{m}_e4d_inapp_cubic')}, hold "
                       f"{N.est(f'sub_{m}_e4d_inapp_hold')}" for m in models)
           + f". Rules E4b/E4c (shorter than {N('min_duration_ms')} ms or peak rotation below {N('min_peak_dps')} deg/s): "
           + '; '.join(f"{SHORT[m]} {N.est(f'sub_{m}_e4bc_base')} on {N(f'sub_{m}_e4bc_base_n')} trials of "
                       f"{N(f'sub_{m}_e4bc_base_np')} persons ({N(f'sub_{m}_e4bc_base_status')})" for m in models
                       if N.ok(f'sub_{m}_e4bc_base')) + '; primary accuracy '
           + ', '.join(f"{SHORT[m]} {N(m + '_accuracy')}" for m in models) + '.')
    why = ('an exclusion that removes scorable trials makes the headline optimistic and the production gate needlessly '
           'strict; one that keeps unscorable trials adds label noise.')
    rem = ('keep the rules for the analysis set; E4b/E4c are justified a priori (fragments: a reach from arm\'s length '
           'lasts longer and rotates the forearm; README section 5.4), not by their small scored sample; E4d is judged '
           'on the trials whose gap falls inside the approach, where imputation actually fills the movement.')
    inapp_low = [m for m in models if str(N.value(f'sub_{m}_e4d_inapp_linear_status')).startswith('below')]
    ev_train = '; '.join(f"{SHORT[m]} E4d added to training, scored on the primary trials {dq(N, f'{m}_e4d_trained')}; "
                         f"E4b/E4c added {dq(N, f'{m}_excl_loose')}; thresholds x{N('tight_factor')} in training "
                         f"{dq(N, f'{m}_excl_tight')}" for m in models)
    ev_pair = '; '.join(f"{SHORT[m]} E4d in training minus primary-only training, on primary + E4d trials "
                        f"{dci(N, f'rem_{m}_e4d_trained')}" for m in models if N.ok(f'd_rem_{m}_e4d_trained'))
    verdict = ('E4d stays: the trials whose gap falls inside the approach score below the primary accuracy for '
               + ', '.join(SHORT[m] for m in inapp_low) + ', whatever the imputation; the margin-only trials score '
               'like primary trials, so a production gate should refuse (or flag) an attempt only when the gap is '
               'inside the movement.' if inapp_low else
               'E4d stays for comparability; no evidence that the imputed in-approach trials score worse, but the '
               f"sample is small ({N('e4d_n_inapp')} trials of {N('e4d_n_persons_inapp')} persons), so production should "
               'flag such attempts rather than trust them.')
    it('Trials removed by the exclusion rules E4b/E4c (fragments) and E4d (long sampling gaps)', det, why, rem,
       'training-side effects (on the primary trials): ' + ev_train + '. Remedy pair: ' + (ev_pair or '--')
       + '. Verdict: ' + verdict)
    S.short.append('exclusions: keep E4b/E4c (a priori) and E4d; '
                   + ('refuse or flag an attempt only when its gap falls inside the movement' if inapp_low else
                      'flag attempts with a gap inside the movement (too few to show a loss)'))


def flags_problem(S: Saved, it: Items) -> None:
    N = S.N
    fl = S.comp[(S.comp['mult_family'] == 'options') & S.comp['option'].str.startswith('drop_')]
    if not len(fl):
        return
    per = '; '.join(f"{SHORT[r['model']]} {_opt(r['option'])} {dq(N, rkey(r['name']))} ({r['verdict']})"
                    for _, r in fl.sort_values(['model', 'grid_index']).iterrows())
    acc = '; '.join(f"{LB.FLAG_LABELS[f]}: GBM {N.est(f'sub_gbm_flag_{f}')} (n = {N(f'sub_gbm_flag_{f}_n')})"
                    for f in LB.FLAGS if N.ok(f'sub_gbm_flag_{f}_n'))
    ctl = '; '.join(f"{SHORT[m]} {dq(N, f'ctl_{m}_drop_all_flags')} ({N(f'v_ctl_{m}_drop_all_flags')})"
                    for m in ('lr', 'gbm') if N.ok(f'd_ctl_{m}_drop_all_flags'))
    ctl_base = '; '.join(f"{SHORT[m]} {dq(N, f'{m}_drop_all_flags_sizematch')}" for m in ('lr', 'gbm')
                         if N.ok(f'd_{m}_drop_all_flags_sizematch'))
    helped = fl['verdict'].isin(['better', 'nominally better']).any()
    ctl_sides = [_ci_side(N, f'd_ctl_{m}_drop_all_flags') for m in ('lr', 'gbm') if N.ok(f'd_ctl_{m}_drop_all_flags')]
    if helped:
        tail = 'cleaning helped for some option; see the decision table.'
    elif ctl_sides and all(s == 0 for s in ctl_sides):
        tail = ('cleaning never helped; removing all flagged trials costs no more than removing as many random trials '
                '(option-minus-control CIs include 0), so its cost is the smaller training set, not the value of the '
                'flagged trials: they stay in training.')
    elif any(s < 0 for s in ctl_sides):
        tail = ('cleaning never helped, and removing the flagged trials costs more than removing as many random ones: '
                'the flagged trials carry information; they stay in training.')
    else:
        tail = 'cleaning never helped: the flagged trials stay in training.'
    it('Flagged trials (grip mismatch, false start, segmentation disagreement, repeated samples, clip candidates) and '
       'the swap-signature blocks',
       'flags mark trials of doubtful quality, the swap-signature blocks look like swapped hands; accuracy of the base '
       'GBM on the flagged trials: ' + (acc or '--') + ' (`p1_subsets`).',
       'training on doubtful trials could teach the model noise; dropping them shrinks the training set.',
       'remove them from TRAINING only (evaluation keeps every primary trial; labels are never changed); the removal of '
       'all flagged trials is compared with a size-matched control (as many random trials out per person and hand).',
       per + '. All flags out minus its control: ' + (ctl or '--') + '; the control itself against the pipeline: '
       + (ctl_base or '--') + '. Verdict: ' + tail)
    S.short.append('flagged trials: ' + ('see the decision table' if helped else 'keep them in training'))


def problems(S: Saved, w: Any) -> None:
    w('## Problems found and how they were handled\n')
    w('Each item: **detected** (a number or figure), **why it matters**, **remedy**, **evidence** (before / after).\n')
    it = Items(w)
    for m in LB.CALIB_MODELS:
        calibration_problem(S, it, m)
    for m in LB.CALIB_MODELS:
        aggregates_problem(S, it, m)
    robustness_problem(S, it)
    platform_problem(S, it)
    exclusions_problem(S, it)
    flags_problem(S, it)
    if it.k == 0:
        w('No problem was detected.')
    w('')


# ───────────────────────────── breakdowns, detail, lists ─────────────────────────────


def breakdowns(S: Saved, w: Any) -> None:
    N = S.N
    w(f"## Breakdowns of the best panel model ({N('best_label')}, current pipeline; exploratory designation)\n")
    w('| Platform | Trials | Persons | Accuracy [CI] | AUC [CI] | Log loss [CI] | Brier [CI] | ECE (null 95th pct.) | '
      'Slope [CI] |')
    w('|---|---|---|---|---|---|---|---|---|')
    for p in S.best.get('platforms', []):
        k = f"best_{p['platform']}"
        w(f"| {p['platform']} | {N(k + '_n')} | {N(k + '_np')} | {N.est(k + '_accuracy')} | {N.est(k + '_auc')} | "
          f"{N.est(k + '_log_loss')} | {N.est(k + '_brier')} | {N(k + '_ece')} ({N(k + '_ece_null_q95')}) | "
          f"{N.est(k + '_slope')} |")
    w('\n| Condition | Level | Trials | Persons | Accuracy [CI] |')
    w('|---|---|---|---|---|')
    from stage2.lanes.p1_numbers import slug
    for c in S.best.get('conditions', []):
        if c['condition'] == 'platform':
            continue
        k = f"cond_{slug(c['condition'])}_{slug(c['level'])}"
        w(f"| {c['condition']} | {c['level']} | {N(k + '_n')} | {N(k + '_np')} | {N.est(k)} |")
    w(f"\nConditions stratify the predictions after the fact; they are never model inputs. CIs need at least "
      f"{N('min_persons_ci')} persons.")
    gb = S.best.get('grip_balance') or []
    if gb:
        cells = []
        for g in gb:
            gk = slug(g['grip'])
            cells.append(f"{g['grip']}: " + ', '.join(f"{h} {N('grip_' + gk + '_' + h)}" for h in ('left', 'right')
                                                       if h in g))
        w('Instructed grip by hand in the primary set (balanced grips rule out a grip-to-hand shortcut through the '
          'device-frame channels): ' + '; '.join(cells) + f" (chi-square p = {N('grip_chi2_p')}).")
    w(f"Per person: median accuracy {N('best_person_median')}, lowest {N('best_person_min')}, "
      f"{N('best_persons_below_80')} of {N('n_persons')} persons below {N('person_threshold')}.\n")


def calibration_detail(S: Saved, w: Any) -> None:
    N = S.N
    w('## Calibration detail (`p1_calibration`, `p1_calibration_paired`, `p1_calibration_platform`, '
      '`p1_calibration_tests`)\n')
    w('| Model | Calibrator | Log loss [CI] | Brier [CI] | AUC | ECE (null 95th pct.; p) | Slope [CI] | Paired log loss '
      '[CI] | Paired AUC [CI] | Paired slope [CI] |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for key in S.cal['methods']:
        m, meth = key.split(':')
        k, pk = f'cal_{m}_{meth}', f'calpd_{m}_{meth}'
        paired = (f"{N.est(pk + '_log_loss')} | {N.est(pk + '_roc_auc')} | {N.est(pk + '_slope')}"
                  if N.ok(pk + '_log_loss') else '(reference) | -- | --')
        w(f"| {SHORT[m]} | {LB.CAL_LABELS.get(meth, meth)} | {N.est(k + '_log_loss')} | {N.est(k + '_brier')} | "
          f"{N(k + '_auc')} | {N(k + '_ece')} ({N(k + '_ece_null_q95')}; p = {N(k + '_ece_p')}) | {N.est(k + '_slope')} | "
          f'{paired} |')
    w('\nThe fixed calibrators are compared on the outer folds: an exploratory comparison of '
      f"{N('n_fixed_calibrators')} configurations; only the nested choice is a headline. Calibration test family "
      f"({N('caltest_family_size')} tests; BH q / Holm p): "
      + '; '.join(f"{t['test']}: p = {N('caltest_' + t['model'] + '_' + t['kind'] + '_p')}, q = "
                  f"{N('caltest_' + t['model'] + '_' + t['kind'] + '_q')}, Holm {N('caltest_' + t['model'] + '_' + t['kind'] + '_holm')}"
                  for t in S.cal.get('tests', [])) + '.')
    for m, a in S.cal.get('aggregates', {}).items():
        ks = sorted({r['k'] for r in a['rows']})
        w(f"\nMulti-attempt aggregates, {MODEL_LABELS[m]} ({N(f'agg_{m}_n_persons')} persons at every k; log loss "
          '[CI] / slope [CI] by k):\n')
        w('| Rule | ' + ' | '.join(f'k = {k}' for k in ks) + ' |')
        w('|---|' + '---|' * len(ks))
        from stage2.lanes.p1_numbers import slug
        for rule in ('mean_prob', 'mean_logit', 'sum_logit', 'sum_logit_T'):
            w(f'| {LB.RULE_LABELS[rule]} | ' + ' | '.join(
                f"{N.est(f'agg_{m}_k{k}_{slug(rule)}_log_loss')} / {N.est(f'agg_{m}_k{k}_{slug(rule)}_slope')}"
                for k in ks) + ' |')
        tk = [f"T_{k} = {N(f'cal_{m}_t_sum_k{k}')}" for k in ks if N.ok(f'cal_{m}_t_sum_k{k}')]
        if tk:
            w('\nMedian over folds of the training-fold temperatures of the summed logits: ' + ', '.join(tk)
              + ' (T_k = k would be the mean of the logits).')
    w('')


def subsets(S: Saved, w: Any) -> None:
    N = S.N
    w('## Excluded, flagged, swap-signature and transferred trials (`p1_subsets`)\n')
    w('| Model | Trials | n | Persons | Accuracy [CI] | Against the primary accuracy |')
    w('|---|---|---|---|---|---|')
    for _, s in S.subs.iterrows():
        k = f"sub_{s['key']}"
        what = (f"{s['platform']} test persons, trained on: {s['training']}" if s['kind'] == 'platform'
                else LB.subset_label(s['label']))
        w(f"| {SHORT.get(s['model'], s['model'])} | {what} | {N(k + '_n')} | "
          f"{N(k + '_np')} | {N.est(k)} | {N(k + '_status')} |")
    w(f"\nNo CI for a subset of fewer than {N('min_persons_ci')} persons. The swap-signature blocks are a sensitivity "
      'analysis only: labels are never changed.\n')


def experiments(S: Saved, w: Any) -> None:
    N = S.N
    w('## Every experiment\n')
    w('One line per LOSO run (`results/<name>.json`): family and option, trials, accuracy [95 % CI]; for a variant the '
      'paired difference against its reference [95 % CI], the BH q within its family and the verdict.\n')
    order = list(S.S['p1_runs'].get('grid_order', []))
    cm = S.comp.set_index('name') if len(S.comp) else S.comp
    rr = S.runs[S.runs['name'].isin(order)].copy()
    rr['gi'] = rr['name'].map(order.index)
    for _, r in rr.sort_values('gi').iterrows():
        k = rkey(r['name'])
        line = (f"- `{r['name']}` ({r['family']}: {_opt(r['option'])}; {N('n_' + k)} trials): {N.est('acc_' + k)}")
        if len(cm) and r['name'] in cm.index:
            c = cm.loc[r['name']]
            fam = c['mult_family'] if isinstance(c['mult_family'], str) else f"{c['role']}, no family"
            line += f"; vs `{c['baseline']}` {dci(N, k)}, q = {N('q_' + k)} ({fam}), {N('v_' + k)}"
        w(line)
    for _, r in S.runs[S.runs['family'] == 'calibration'].sort_values('name').iterrows():
        k = rkey(r['name'])
        _, _, m, meth = r['name'].split('_', 3)
        w(f"- `{r['name']}` (calibration: {SHORT[m]} + {LB.CAL_LABELS.get(meth, meth)}): {N.est('acc_' + k)}, log loss "
          f"{N(f'cal_{m}_{meth}_log_loss')}, ECE {N(f'cal_{m}_{meth}_ece')}, slope {N.est(f'cal_{m}_{meth}_slope')}")
    for r in S.S['p1_comparisons'].get('remedies', []) + S.S['p1_comparisons'].get('controls', []):
        k = ('rem_' if r.get('pair_kind') == 'remedy' else 'ctl_') + rkey(r['name'])
        w(f"- pair `{r['name']}` minus `{r['baseline']}` ({r['what']}): {dci(N, k)}, q = {N('q_' + k)} (pairs), "
          f"{N('v_' + k)}")
    w('')


def figures(S: Saved, w: Any) -> None:
    w('## Figures (`reports/stage2/figures/p1/`, PDF + PNG; plotted data in `tables/p1/<figure>.csv`)\n')
    for name in sorted(S.caps):
        c = S.caps[name]
        w(f"- `{name}`: {c.get('decision', '')} {c.get('caption', '')}")
    w('\n## Tables (`reports/stage2/tables/p1/`, CSV + booktabs .tex)\n')
    for name, what in TABLE_FILES.items():
        if (L.tab_dir / f'{name}.csv').exists():
            w(f'- `{name}`: {what}')
    w('- Derived JSON (the inputs of every figure, table and number): `p1_runs`, `p1_comparisons` (with `remedies` and '
      '`controls`), `p1_subsets` (with the E4d gap locations), `p1_best`, `p1_calibration`, `p1_decisions`, '
      '`p1_filter_delay`, `p1_variants`.\n')


def commands(w: Any) -> None:
    w('## Commands (from `analysis/`)\n')
    w('```')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes p1      # missing / stale variants, experiments, calibration; derive')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes p1 --only p1_derive            # re-derive the tables only')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes p1 --only "p1_gbm_*" --force   # re-run some experiments')
    w('.venv/Scripts/python.exe -m stage2.pipeline figures --lanes p1      # every P1 figure from the derived files')
    w('.venv/Scripts/python.exe -m stage2.pipeline figures --lanes p1 --only P1_02_cutoff   # one figure')
    w('.venv/Scripts/python.exe -m stage2.pipeline tables --lanes p1       # CSV + booktabs .tex, registry_p1.csv')
    w('.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes p1      # p1_numbers.json, finalists_p1.json, this file')
    w('.venv/Scripts/python.exe -m stage2.pipeline all --lanes p1 --force   # everything from the snapshot')
    w('.venv/Scripts/python.exe -m pytest stage2/tests/test_p1.py stage2/tests/test_p1_variants.py -q -p no:cacheprovider')
    w('```')
    w('Timing on the shared lab machine: redrawing every P1 figure takes under a minute, mostly importing '
      'matplotlib and writing PDFs (the presentation modules import no model code); the derive step takes several '
      'minutes (bootstraps); a forced rebuild of every experiment takes hours on the lane\'s two threads. Presentation '
      'settings (sizes, limits, ticks, labels) are in `FIGURES` at the top of `stage2/lanes/p1_figures.py`; printed '
      'labels in `stage2/lanes/p1_labels.py`. A new participant makes the cache, the variant caches and every result '
      'stale; a plain `compute` rebuilds them. Editing the panel code (`p1_panel.py`) needs `compute --lanes p1 '
      '--force`.\n')


def conclusions(S: Saved, w: Any) -> None:
    N = S.N
    w('## Conclusions\n')
    dec = S.dec['rows']
    change = [r for r in dec if r['decision'] != 'keep the default']
    w(f"- {N('n_choices_kept')} of {N('n_choices')} data-treatment choices keep the current default by the "
      'pre-specified rule' + (f"; the rule indicates a change for "
                              f"{', '.join(LB.FAMILY_LABELS[r['family']] + ' (' + r['decision'] + ')' for r in change)}."
                              if change else '; no option beat the current pipeline for two panel models, so no default '
                                             'changes. The rule cannot change a default whose options ran on one model.'))
    opts = S.comp[S.comp['mult_family'] == 'options']
    worse = opts[opts['verdict'] == 'worse'].sort_values('d_acc')
    if len(worse):
        w('- Options that cost accuracy after the multiplicity correction (evidence FOR the current choice): '
          + '; '.join(f"{_opt(r['option'])} ({SHORT[r['model']]} {dq(N, rkey(r['name']))}, Holm p = "
                      f"{N('holm_' + rkey(r['name']))})" for _, r in worse.iterrows()) + '.')
    nom = opts[opts['verdict'] == 'nominally worse'].sort_values('d_acc')
    if len(nom):
        w('- Nominally worse only (CI below 0, not significant after the correction; suggestive): '
          + '; '.join(f"{_opt(r['option'])} ({SHORT[r['model']]} {dq(N, rkey(r['name']))})" for _, r in nom.iterrows())
          + '.')
    better = opts[opts['verdict'].isin(['better', 'nominally better'])].sort_values('d_acc', ascending=False)
    if len(better):
        w('- Options with a CI above 0 (single-model gains, below the change rule): '
          + '; '.join(f"{_opt(r['option'])} ({SHORT[r['model']]} {dq(N, rkey(r['name']))}, {r['verdict']})"
                      for _, r in better.iterrows()) + '.')
    w(f"- Multiplicity: {N('fam_options_size')} option comparisons at 95 % would put about {N('expected_false_ci')} CIs "
      f"beyond 0 by chance; {N('n_options_ci_excludes_0')} do; Benjamini-Hochberg keeps {N('n_options_worse')} (all "
      f"worse) and Holm keeps {N('n_options_holm')}.")
    llw = opts[opts['logloss_verdict'] == 'worse'].sort_values('d_logloss', ascending=False)
    if len(llw):
        w('- Log-loss evidence for the current choices (paired per-person log-loss CI above 0 and its BH q below '
          f"{N('alpha')} in the options family): "
          + '; '.join(f"{_opt(r['option'])} ({SHORT[r['model']]} {dll(N, rkey(r['name']))})" for _, r in llw.iterrows())
          + '.')
    llb = opts[opts['logloss_verdict'].isin(['better', 'nominally better'])].sort_values('d_logloss')
    if len(llb):
        w('- Leads for the synthesis (lower log loss on the oracle bounds; accuracy decides the rule, so these are not '
          'changes): ' + '; '.join(f"{_opt(r['option'])} ({SHORT[r['model']]} log loss {dll(N, rkey(r['name']))}, "
                                    f"{r['logloss_verdict']}; accuracy {dci(N, rkey(r['name']))})"
                                    for _, r in llb.iterrows()) + '.')
    rec = []
    for fam in ('sampling', 'gravity', 'gaps', 'filter'):
        if N.ok(f'dec_{fam}_max_loss_lo'):
            rec.append(f"{LB.FAMILY_LABELS[fam].lower()} {N(f'dec_{fam}_max_loss_lo')}")
    if rec:
        w('- Robustness to the recording is bounded, not proven: the largest accuracy loss compatible with the data '
          'over the options of a choice (lowest lower CI bound, points): ' + ', '.join(rec) + '.')
    if N.ok('d_gbm_lp_causal') or N.ok('d_lr_lp_causal'):
        w(f"- A causal low-pass at the current cut-off delays a 1-Hz movement by about {N('causal_delay_ms')} ms "
          '(`P1_10_filter_delay`); its paired accuracy differences are '
          + ', '.join(f"{SHORT[m]} {dci(N, f'{m}_lp_causal')}" for m in MODELS if N.ok(f'd_{m}_lp_causal'))
          + ', so the largest loss compatible with the data is '
          + ', '.join(f"{N(f'd_{m}_lp_causal_lo')} points ({SHORT[m]})" for m in MODELS if N.ok(f'd_{m}_lp_causal'))
          + '. Zero phase stays the default because the predictor filters a recorded buffer.')
    ident = opts[opts['verdict'] == 'identical']
    if len(ident):
        vd = {r['variant']: r for r in S.S['p1_variants'].get('array_diffs', []) if r.get('comparable')}
        lines = []
        for _, r in ident.iterrows():
            k = rkey(r['name'])
            v = vd.get(r['option'])
            why = (f"its arrays differ from the default on {N('vdiff_' + r['option'] + '_n_differing')} primary trials"
                   if v is not None else
                   ("a per-channel affine scaling, which the logistic model's own feature standardisation cancels"
                    if r['option'] in ('norm_zscore', 'norm_robust') and r['model'] == 'lr' else
                    'a training-side treatment'))
            lines.append(f"{SHORT[r['model']]} {_opt(r['option'])} (largest change of P(right) {N('maxdp_' + k)}; {why})")
        w('- Identical within solver precision: ' + '; '.join(lines) + '.')
    if S.short:
        w('- Data handling (problems above): ' + '; '.join(S.short) + '.')
    restored = [m for m, v in S.cal_verdict.items() if v == 'restored']
    harmful = [m for m, v in S.cal_verdict.items() if v == 'harmful']
    w('- Calibration: ' + ('; '.join(f'{SHORT[m]} is over-confident and its nested calibrator restores a slope of '
                                     'about 1, so the product P(right) should come from the calibrated '
                                     f'{SHORT[m]}' for m in restored) or 'no panel model needs a calibration step')
      + ('; ' + ', '.join(SHORT[m] for m in harmful) + ': the nested calibrator does not help, leave it uncalibrated'
         if harmful else '') + '. Several attempts of one person: report the summed logits divided by the '
      'training-fold T_k (problems on the aggregates).')
    nat_all = N.ok('gbm_nat_n_right') and N.value('gbm_nat_n_right') == N.value('gbm_nat_n')
    w('- Caveats: the panel is fixed and small (three pre-specified models), and not every option runs on every model '
      f"(MiniRocket only on the most informative ones, for compute); variants run the leak check on "
      f"{N('leak_sample_folds')} seeded folds; with {N('n_persons')} persons a CI of a few points is the resolution, "
      'so "no difference" is absence of evidence; '
      + ('every natural-use trial is right-handed, so natural-use accuracy is right-hand recall; ' if nat_all else '')
      + 'segmentation errors are synthetic (uniform), not those of a real cue-free detector; the ECE null treats '
        'trials as independent; the fixed calibrators are compared on the outer folds (exploratory); the best-model '
        'designation is exploratory.')
    w('- Open questions: whether the cue-free detector (not the oracle bounds) changes the ranking of the window '
      'options; whether the GBM calibration transfers to a person recorded later; whether attempts with gaps inside '
      'the movement can be scored once more such data exist.')
    w('')


def build(entries: dict[str, Any] | None = None, mark: bool = False) -> str:
    """The summary text. ``mark=True`` wraps every number printed from the numbers file in sentinel characters
    (tests: what is left outside them must hold no number)."""
    S = Saved(entries, mark=mark)
    out: list[str] = []
    w = out.append
    w('# Lane p1: preprocessing and data-treatment decisions\n')
    w('*Generated by `python -m stage2.pipeline numbers --lanes p1` from the saved results; every number is an entry of '
      '`results/p1_numbers.json`; do not edit by hand.*\n')
    w('## Goal\n')
    w('Justify every signal-processing and data-treatment choice of the Stage 2 pipeline with evidence: for each choice '
      '(filtering, gaps and imputation, sampling rate and precision, gravity estimate, frame, window, segmentation error, '
      'normalisation, outliers and exclusions, platform, augmentation) the options run through the same '
      'leave-one-person-out harness with a FIXED panel of three pre-specified models of different kinds, and each option '
      "is compared PAIRED (same trials, same folds) with the current pipeline. The calibration of the panel's P(right), "
      'and of multi-attempt aggregates, is studied with remedies fitted strictly inside the training folds.\n')
    method(S, w)
    headline(S, w)
    decisions(S, w)
    problems(S, w)
    breakdowns(S, w)
    calibration_detail(S, w)
    subsets(S, w)
    experiments(S, w)
    figures(S, w)
    commands(w)
    conclusions(S, w)
    return '\n'.join(out) + '\n'


def write(entries: dict[str, Any] | None = None) -> None:
    L.summary_path.write_text(build(entries), encoding='utf-8', newline='\n')


__all__ = ['build', 'write']
