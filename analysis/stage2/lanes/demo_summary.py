"""demo_SUMMARY.md of lane demo, generated from the numbers file (every number in it is a numbers entry)."""
from __future__ import annotations

from typing import Any

from stage2 import report_export as RX


def write(L: Any, n: dict[str, dict[str, Any]]) -> None:
    def F(k: str) -> str:
        return RX.format_value(n[k]['value'], n[k]['format'])

    def P(k: str) -> str:
        v = n[k]['value']
        return '< 0.001' if v < 0.001 else F(k)

    def ci(k: str) -> str:
        return f'{F(k)} [{F(k + "_lo")}, {F(k + "_hi")}]'

    def verdict(s: str) -> str:
        acc_sig, ll_sig = n[f'{s}_holm_acc']['value'] < 0.05, n[f'{s}_holm_ll']['value'] < 0.05
        worse_acc = n[f'{s}_dacc_hi']['value'] < 0
        better_acc = n[f'{s}_dacc_lo']['value'] > 0
        worse_ll = n[f'{s}_dll_lo']['value'] > 0
        better_ll = n[f'{s}_dll_hi']['value'] < 0
        parts = []
        if better_acc or better_ll:
            parts.append('improves on the sensor-only stage' + (' (Holm-significant)' if (acc_sig or ll_sig) else
                                                                 ' (not significant after Holm)'))
        if worse_acc:
            parts.append('lowers accuracy' + (' (Holm-significant)' if acc_sig else
                                              ' (paired CI below zero, not significant after Holm)'))
        if worse_ll:
            parts.append('raises the log loss' + (' (Holm-significant)' if ll_sig else
                                                  ' (paired CI above zero, not significant after Holm)'))
        return '; '.join(parts) if parts else 'cannot be separated from the sensor-only stage'

    any_gain = any(n[f'{s}_dacc_lo']['value'] > 0 or n[f'{s}_dll_hi']['value'] < 0 for s in ('main', 'inter'))
    decision = ('Demographics are NOT added, even as a hypothetical: '
                + ('no add-on variant improves accuracy or log loss on the same folds; '
                   if not any_gain else 'an add-on variant improves a metric on the same folds (see the table); ')
                + 'nothing from this lane enters the production model (finalists_demo.json is an empty list).')
    inter_clause = (', and the interaction variant does not help here' if not (n['inter_dacc_lo']['value'] > 0 or
                    n['inter_dll_hi']['value'] < 0) else ', and the interaction variant helps (see above)')
    nm = max(n['main_nat_recall']['value'], n['inter_nat_recall']['value'])
    nat_cmp = 'not above' if nm <= n['sensor_nat_recall']['value'] else 'partly above'
    conclusion = ('even in the hypothetical where production knew them, the demographics do not improve the hand '
                  'prediction; they add person-level noise that the few training persons cannot regularise away.'
                  if not any_gain else 'an add-on variant improves a metric; see the paired table before any use.')
    t = f"""# Lane demo: the demographics add-on test (fast-track first pass, not audited)

## Goal
Demographics (age, sex, height, self-reported handedness, usual phone hand) are not available in production. Would
adding them help if they were? This lane is the only place where demographics enter a model, as an explicitly
labelled add-on experiment. Nothing from it enters the production model.

## Design (design (a) of the brief)
- **Base:** the final model (lane final `m3_deep`, a four-member 1D CNN ensemble, shipped as `{'final_cal_m3_deep_none'}`).
  Refitting it with covariates for several variants was too slow for the fast track, so the covariates enter a
  **second-stage logistic regression on the final model's logit**.
- **Training side of each outer fold:** the final model's INNER cross-fitted logits of the training persons, read
  from lane final's inner-prediction cache (`analysis/data/stage2_final/inner/m3_deep/`). The cache was checked to
  hold the candidate for every outer fold, with the current code fingerprint and exactly the harness's capped
  training trials (`tables/demo/demo_inner_cache_check.json`).
- **Held-out person:** the logit from the final model's saved LOSO predictions (the fold model that never saw that
  person; natural-use trials from the same result). The second stage never uses other persons' outer-fold
  predictions, because those fold models were trained on the held-out person.
- **Covariates:** age, female, height with a missing indicator ({F('profile_n_height_missing')} heights missing,
  median-imputed in the fold), handedness sign and usual-phone-hand sign. They are passed as the separate block
  `person_covariates`; the test person's covariates are used at prediction time. Instructed conditions are never
  used.
- **Fitting:** standardised in the fold; the L2 penalty is chosen from a grid of five by grouped inner
  cross-validation (`fold.inner_splits()`) on the training persons.
- **Evaluation:** every variant runs through `cv.run_experiment` (leave-one-person-out, leak check on every fold,
  person-bootstrap CIs). Paired comparisons use `cv.paired_compare` on the same folds, with Holm across the add-on
  variants within each metric. The sign-flip p-values and the CIs are both per person.
- **Cohort profile** (why interactions are weakly identified): {F('profile_n_persons')} persons,
  {F('profile_n_left_handed')} left-handed, {F('profile_n_phone_both')} with phone hand 'both',
  {F('profile_n_phone_left')} with phone hand 'left', {F('profile_n_female')} female, aged {F('profile_age_min')} to
  {F('profile_age_max')}.

## Experiments (results/<name>.json; registry results/registry_demo.csv)
- `demo_sensor_stack`: second stage on the final logit only (the sensor-only comparator).
- `demo_cov_main`: + covariates (main effects).
- `demo_cov_inter`: + covariates and covariate x logit interactions (the logit is the model's chiral evidence).
- `demo_cov_only`: covariates only (the floor).
- `demo_rule_phone`: a fixed rule answering the usual phone hand ('both' is a tie). For the base-rate note only.
- `demo_rule_right`: a fixed rule that always answers right. For the base-rate note only.

## Headline (LOSO, person-bootstrap {F('ci_level')} CIs)
| model | accuracy | balanced acc. | log loss | natural use: right-hand recall |
|---|---|---|---|---|
| final model as shipped | {ci('shipped_acc')} | {F('shipped_bacc')} | {ci('shipped_ll')} | {F('shipped_nat_recall')} |
| sensor-only second stage | {ci('sensor_acc')} | {F('sensor_bacc')} | {ci('sensor_ll')} | {F('sensor_nat_recall')} |
| + covariates | {ci('main_acc')} | {F('main_bacc')} | {ci('main_ll')} | {F('main_nat_recall')} |
| + covariates and covariate x logit | {ci('inter_acc')} | {F('inter_bacc')} | {ci('inter_ll')} | {F('inter_nat_recall')} |
| covariates only (floor) | {ci('covonly_acc')} | {F('covonly_bacc')} | {ci('covonly_ll')} | {F('covonly_nat_recall')} |
| rule: usual phone hand | {ci('rulephone_acc')} | {F('rulephone_bacc')} | {ci('rulephone_ll')} | {F('rulephone_nat_recall')} |
| rule: always right | {ci('ruleright_acc')} | {F('ruleright_bacc')} | {ci('ruleright_ll')} | {F('ruleright_nat_recall')} |

**Paired against the sensor-only second stage.** Mean per-person differences, paired person-bootstrap {F('ci_level')} CIs,
Holm over the add-on variants within each metric:
- **+ covariates:** accuracy {ci('main_dacc')} points (Holm p {P('main_holm_acc')}); log loss {ci('main_dll')} (Holm p
  {P('main_holm_ll')}); primary trials that change answer: {F('main_flips')}; persons more / less accurate:
  {F('main_better')} / {F('main_worse')}. Verdict: it {verdict('main')}.
- **+ covariates and covariate x logit:** accuracy {ci('inter_dacc')} points (Holm p {P('inter_holm_acc')}); log loss
  {ci('inter_dll')} (Holm p {P('inter_holm_ll')}); primary trials that change answer: {F('inter_flips')}; persons
  more / less accurate: {F('inter_better')} / {F('inter_worse')}. Verdict: it {verdict('inter')}.
- **Check that the stacking itself is neutral:** sensor-only second stage minus the shipped model, accuracy
  {F('stack_vs_shipped_dacc')} points, log loss {F('stack_vs_shipped_dll')} (unadjusted sign-flip p
  {P('stack_vs_shipped_p_ll')}).

## Decisions and their evidence
- {decision} Evidence: the paired differences above (`DEMO_01_paired`, `tables/demo/demo_models`).
- **The covariates-only floor is at chance**, as it must be when every person did both hands. Primary accuracy is
  {ci('covonly_acc')} and balanced accuracy {F('covonly_bacc')}; its P(right) never moves more than {F('covonly_max_dev')} from {F('chance')}. Its natural-use
  recall ({F('covonly_nat_recall')}) is not a signal: an almost constant P(right) sits a hair below {F('chance')}
  because the primary set holds slightly fewer right-hand trials ({F('primary_right_share')}). A
  person-level covariate can only help through interactions{inter_clause}.
- **Base-rate effect (natural use).** Every natural-use trial so far is a right-hand trial
  ({F('nat_n')} trials, one per person). The phone-hand rule reaches {F('rulephone_nat_recall')} right-hand recall
  there ({F('rulephone_nat_ties')} ties at half credit) and the always-right rule {F('ruleright_nat_recall')}. On the
  primary set both are at chance (balanced accuracy {F('rulephone_bacc')} and {F('ruleright_bacc')}). Their
  natural-use score reflects the base rate, not discrimination (`DEMO_02_base_rate`). The covariate add-ons change
  {F('main_nat_flips')} (main effects) and {F('inter_nat_flips')} (interactions) natural-use answers. Their
  natural-use recall ({F('main_nat_recall')}, {F('inter_nat_recall')}) is {nat_cmp} the sensor-only stage's
  ({F('sensor_nat_recall')}).

## Figures (figures/demo/)
- `DEMO_01_paired`: the paired accuracy and log-loss differences of the add-on variants against the sensor-only
  stage, with Holm p (filled red: below {F('alpha')}). Message: + covariates {verdict('main')}; + covariates and
  covariate x logit {verdict('inter')}.
- `DEMO_02_base_rate`: primary balanced accuracy against natural-use right-hand recall per model. A phone-hand prior
  looks excellent on natural use for a trivial reason (the base rate).

## Tables (tables/demo/)
- `demo_models.csv|.tex`: every model with accuracy, balanced accuracy, log loss, AUC, natural-use recall, and the
  paired differences with Holm p.
- Derived data: `demo_derived.json` (comparisons, profile counts, per-fold coefficients), `demo_inner_cache_check.json`.

## Regenerate
```
cd analysis
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes demo            # missing experiments + derived comparisons
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes demo --force    # all of them again
.venv/Scripts/python.exe -m stage2.pipeline figures --lanes demo
.venv/Scripts/python.exe -m stage2.pipeline tables --lanes demo
.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes demo            # numbers, finalists ([]), this summary
.venv/Scripts/python.exe -m pytest stage2/tests/test_demo.py -q -p no:cacheprovider
```
It needs lane final's saved result `final_cal_m3_deep_none` and its inner cache for `m3_deep`. If a fold is missing,
compute stops and names the `python -m stage2.lanes.final_nested --jobs m3_deep:<fold>` command that fills it.

## Conclusions, caveats, open questions
- **Conclusion:** {conclusion}
- **Caveat, power:** the cohort has {F('profile_n_left_handed')} left-handers and {F('profile_n_phone_left')} persons whose usual
  phone hand is left, so an interaction such as 'left-handers move differently' cannot be estimated. This is a null result on a
  small, mostly right-handed sample, not evidence of no effect.
- **Caveat, design:** the second stage stacks the final model's inner cross-fitted logits (models trained on about
  four fifths of the training persons) and applies the result to the outer logit (a model trained on all of them),
  the same convention lane final uses for its calibrators. The final model itself is exploratory (chosen among
  {F('base_configs')} configurations on the outer folds), so the absolute levels inherit its optimism. The paired
  differences share that base, so the optimism largely cancels in them. The penalty is chosen on the same inner
  logits that the stage is fitted on, a second-order effect that is not removed.
- **Not done (fast track):** no refit of the CNN with covariate inputs; no option-(b) cheap-base replication; no
  calibration analysis of the add-ons (they lose on log loss already); no audit.
"""
    L.summary_path.write_text(t, encoding='utf-8')
