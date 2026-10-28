"""Lane m4: m4_SUMMARY.md rendered from the numbers file (results/m4_numbers.json) and saved tables only. Every
number is read from the numbers file; every verdict sentence is generated from saved paired comparisons
(Holm-adjusted within a named family) or saved calibration flags, never typed."""
from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from stage2 import report_export as RX
from stage2.pipeline import Lane

L = Lane('m4')


def render(n: dict[str, Any]) -> str:
    from stage2.lanes import m4_pipeline as P
    from stage2.lanes import m4_report as R
    from stage2.lanes.m4_config import N_BOOT
    from stage2.lanes.m4_online import AGGREGATION_LABELS

    def v(k: str) -> str:
        e = n.get(k)
        return '--' if e is None else RX.format_value(e['value'], e['format'])

    def val(k: str) -> Any:
        e = n.get(k)
        return None if e is None else e['value']

    def ci(k: str) -> str:
        return f'[{v(k + "_lo")}, {v(k + "_hi")}]'

    def verdict(key: str, a: str, b: str, metric: str = 'accuracy', prefix: str = 'd_') -> str:
        return R.pair_verdict(n, key, a, b, prefix=prefix, metric=metric)

    def ll_verdict(key: str, a: str, b: str) -> str:
        return R.pair_verdict(n, key, a, b, prefix='d_', p_suffix='_ll_p_holm', lo='_ll_lo', hi='_ll_hi',
                              metric='log loss (lower is better)').replace('has the higher log loss (lower is better)',
                                                                           'has the HIGHER (worse) log loss')

    def sp_verdict(key: str, a: str, b: str, metric: str = 'accuracy') -> str:
        if metric == 'accuracy':
            return R.pair_verdict(n, key, a, b, prefix='sp_', p_suffix='_p_holm', lo='_d_acc_lo', hi='_d_acc_hi')
        return R.pair_verdict(n, key, a, b, prefix='sp_', p_suffix='_ll_p_holm', lo='_d_ll_lo', hi='_d_ll_hi',
                              metric='log loss').replace('has the higher log loss', 'has the HIGHER (worse) log loss')

    lab = {h: x.replace('$s_k$', 's_k') for h, x in AGGREGATION_LABELS.items()}
    prod = val('prod_name') or P.PRODUCTION
    pa = val('prod_aggregation') or 'mean'
    ref = val('sess_ref_keep_s') or 1.0
    kr = f'{ref:g}'.replace('.', 'p')
    rule = val('prod_rule') or 'no pairing'
    other_rule = 'pairing' if rule == 'no pairing' else 'no pairing'
    rule_key = rule.replace(' ', '_')
    other_key = other_rule.replace(' ', '_')
    chosen = ', '.join(f"{m.replace('_', ' ')} in {v(f'prod_calibrator_{m}_folds')}" for m in
                       ('none', 'temperature', 'isotonic_sym') if f'prod_calibrator_{m}_folds' in n)
    models = R.model_rows()
    exp_lines = []
    for _, r in models.iterrows():
        nat = '--' if r['nat_answered'] == '--' else f"{r['nat_answered']}, {100 * r['nat_acc_half']:.1f} %"
        acc = '--' if not _num(r['accuracy']) else f"{100 * r['accuracy']:.1f} % {r['acc_ci']}"
        exp_lines.append(f"| `{r['name']}` | {r['label']} | {acc} | {r['roc_auc']:.3f} | {r['log_loss']:.3f} | "
                         f"{r['ece']:.3f} | {100 * r['coverage']:.0f} % | {nat} | {r['n_configs_tried']} |")
    B = R._j('m4_breakdown')
    br = []
    if B:
        b = pd.DataFrame(B['rows'])
        b = b[b['name'] == B['production']]
        for _, r in b.iterrows():
            c = '' if not (_num(r['acc_lo']) and _num(r['acc_hi'])) else f" [{100 * r['acc_lo']:.1f}, {100 * r['acc_hi']:.1f}]"
            br.append(f"| {r['stratum']} | {r['level']} | {int(r['n'])} | {int(r['n_persons'])} | {100 * r['accuracy']:.1f} %{c} "
                      f"| {100 * r['coverage']:.0f} % | {r['ece']:.3f} |")
    caps = {}
    cp = L.fig_dir / 'captions.json'
    if cp.exists():
        caps = json.loads(cp.read_text(encoding='utf-8'))
    fig_lines = [f"- `figures/m4/{k}.pdf` - {c['decision']}" for k, c in sorted(caps.items())]
    demo = R._j('m4_demo')
    cal_ok = {c: val(f'cal_{c}_calibrated') for c in P.CALIBRATION_CANDIDATES}
    stop_ok = {h: val(f'stopcal_{h}_calibrated') for h in P.STOP_AGGREGATIONS}

    def pts(x: Any) -> str:
        return '--' if not _num(x) else f'{100 * x:+.1f} points'

    def diff(a: str, b: str) -> Any:
        x, y = val(a), val(b)
        return x - y if _num(x) and _num(y) else None

    # ---- generated statements (conditional on the saved numbers, never typed) ----
    g_true, g_all = diff(f'sesstrue_{kr}_k3_acc', f'sesstrue_{kr}_k1_acc'), diff(f'sess_{kr}_k3_acc', f'sess_{kr}_k1_acc')
    if _num(g_true) and _num(g_all):
        multi_touch = (f"From 1 to 3 accepted attempts, the true approaches alone change the accuracy by {pts(g_true)} and "
                       f"every accepted attempt by {pts(g_all)}: "
                       + ('the false attempts cost the multi-touch gain, so rejecting them is the first next step.'
                          if g_true > g_all + 0.02 else
                          'even the true approaches gain little, because the attempts of one person share their errors.'))
    else:
        multi_touch = 'The attainable curve was not computed.'
    # stillness settings of the stress test, read from the numbers keys (m4_config.KEEP_S_STRESS; never typed here)
    inst = v('instructed_stillness_s')
    kinst = f"{val('instructed_stillness_s') or 0:g}".replace('.', 'p')

    def keeps_of(prefix: str) -> list[str]:
        ks = [k[len(prefix):-len('_acc')] for k in n if k.startswith(prefix) and k.endswith('_acc')]
        ks = [k for k in ks if '_' not in k]
        num = sorted((k for k in ks if k != 'rec'), key=lambda s: -float(s.replace('p', '.')))
        return (['rec'] if 'rec' in ks else []) + num

    def disp(kk: str) -> str:
        return 'as recorded' if kk == 'rec' else ('none' if float(kk.replace('p', '.')) == 0 else f"{kk.replace('p', '.')} s")

    stress_keeps = keeps_of('stress_production_')
    stress_list = ', '.join(f"{v(f'stress_production_{kk}_acc')} {disp(kk) if kk == 'rec' else 'with ' + disp(kk)}"
                            for kk in stress_keeps)
    pair_list = '; '.join(f"{disp(kk)} {v(f'stresspair_robust_vs_full_{kk}_d')} [{v(f'stresspair_robust_vs_full_{kk}_lo')}, "
                          f"{v(f'stresspair_robust_vs_full_{kk}_hi')}] (Holm p {v(f'stresspair_robust_vs_full_{kk}_p_holm')})"
                          for kk in stress_keeps if f'stresspair_robust_vs_full_{kk}_d' in n)
    sig = []
    for kk in stress_keeps:
        p_, lo_, hi_ = (val(f'stresspair_robust_vs_full_{kk}{s_}') for s_ in ('_p_holm', '_lo', '_hi'))
        if all(_num(x) for x in (p_, lo_, hi_)) and p_ < 0.05 and (lo_ > 0 or hi_ < 0):
            sig.append(disp(kk) + (' (robust better)' if lo_ > 0 else ' (full better)'))
    stress_sig = ', '.join(sig) if sig else 'no stillness setting'
    least = [kk for kk in stress_keeps if kk != 'rec'][-1:] or ['rec']
    drop = diff(f'stress_production_{kinst}_acc', f'stress_production_{least[0]}_acc')
    stillness_req = ((f'The production accuracy falls as the stillness drops below the instructed {inst} s, so the page '
                      f'must enforce at least {inst} s of stillness, measured exactly as the feature is (the causal '
                      'activity below the stillness threshold).') if _num(drop) and drop > 0.02 else
                     'The production accuracy does not fall below the instructed stillness in this test.')
    abstains = diff(f'stress_production_{kinst}_cov', f'stress_production_{least[0]}_cov')
    stillness_fail = (f"with less than the instructed stillness the detector answers less often (answered: {disp(kinst)} "
                      f"{v(f'stress_production_{kinst}_cov')}, {disp(least[0])} {v(f'stress_production_{least[0]}_cov')})"
                      if _num(abstains) and abstains > 0.05 else
                      'the share answered does not fall with less stillness')
    nat_more = _num(val('nat_p_poisson')) and val('nat_p_poisson') < 0.05
    nat_test = ("more rejections than the persons' own primary rates predict" if nat_more else
                "not more rejections than the persons' own primary rates predict")
    gneg = diff('nat_grip_gate', 'prim_grip_gate')
    nat_grip = (f"The rest-pose grip check fires on {v('nat_grip_gate')} of the natural-use recordings and "
                f"{v('prim_grip_gate')} of the primary ones, "
                + ('so the grip does not explain the rejections.' if _num(gneg) and gneg <= 0.1 else
                   'so part of the rejections may come from the grip.'))
    non_anti =[c for c in P.CALIBRATION_CANDIDATES if val(f'cal_{c}_antisymmetric') == 'no']
    cal_names = {'platt': 'Platt', 'isotonic': 'plain isotonic', 'beta': 'beta'}
    non_anti_text = ', '.join(f"{cal_names.get(c, c.replace('_', ' '))} (largest |P(z) + P(-z) - 1| "
                              f"{v(f'cal_{c}_mirror_dev_max')})" for c in non_anti) or 'none'
    gate_fail = (_num(val('stressflip_production_rec_acc_gated_seg')) and _num(val('stressflip_production_rec_acc'))
                 and val('stressflip_production_rec_acc_gated_seg') - val('stressflip_production_rec_acc') < 0.02)
    first_remedy = (f"The first remedy, a check of the chosen segment's own start pose, stopped "
                    f"{v('stressflip_production_rec_gate_seg')} of the twins and left their accuracy at "
                    f"{v('stressflip_production_rec_acc_gated_seg')} (the answers it let through were right "
                    f"{v('stressflip_production_rec_acc_answered_gated_seg')}; answered wrongly "
                    f"{v('stressflip_production_rec_wrong_gated_seg')})"
                    + (': when the detector takes the return for the approach, the chosen segment starts at the nose, '
                       'so its start pose says nothing about the grip; it was replaced.' if gate_fail else '.'))
    check_works = (_num(val('stressflip_production_rec_acc_gated')) and _num(val('stressflip_production_rec_acc'))
                   and val('stressflip_production_rec_acc_gated') > val('stressflip_production_rec_acc'))
    grip_lower = _num(val('d_grip_vs_robust')) and val('d_grip_vs_robust') < 0
    keep_robust = ((', so production keeps the robust features behind the grip check: a user asked to turn the phone '
                    'then gets the screen-toward accuracy.') if check_works and grip_lower else
                   '; the grip check alone does not settle the choice of feature set.')
    # P(approach) threshold (re-thresholded saved predictions) and look-ahead ablations, from the numbers keys
    thr_keys = sorted(k[len('thr_'):-len('_acc')] for k in n if k.startswith('thr_') and k.endswith('_acc')
                      and '_' not in k[len('thr_'):-len('_acc')])
    thr_list = ', '.join(f"{kk.replace('p', '.')}: {v(f'thr_{kk}_acc')} (answers {v(f'thr_{kk}_coverage')})"
                         for kk in thr_keys)
    gains = [kk for kk in thr_keys if _num(val(f'thr_{kk}_d_lo')) and val(f'thr_{kk}_d_lo') > 0]
    thr_decision = (f"{v('det_threshold')} kept (pre-specified). Exploratory (re-thresholded on the outer folds, "
                    "outside the Holm family): "
                    + ', '.join(f"{kk.replace('p', '.')} {v(f'thr_{kk}_d')} [{v(f'thr_{kk}_d_lo')}, {v(f'thr_{kk}_d_hi')}]"
                                for kk in gains)
                    + ' answer more and gain half-credit accuracy: a candidate for a threshold chosen inside the folds'
                    if gains else f"{v('det_threshold')} kept (pre-specified; no other threshold gains accuracy)")
    looks = sorted(P.LOOKAHEAD_ABLATION.items(), key=lambda kv: kv[1])
    look_opts = ' / '.join(f'{s:g}' for s in sorted([s for _, s in looks] + [val('lookahead_s') or 0])) + ' s'
    look_ev = '; '.join(f"{s:g} s - production {v('d_' + nm.replace('m4_abl_', 'abl_') + '_vs_base')} "
                        f"{ci('d_' + nm.replace('m4_abl_', 'abl_') + '_vs_base')}, log loss "
                        f"{v('d_' + nm.replace('m4_abl_', 'abl_') + '_vs_base_ll')}" for nm, s in looks)
    look_dec = '; '.join(verdict(nm.replace('m4_abl_', 'abl_') + '_vs_base', f'{s:g} s', 'the production look-ahead')
                         for nm, s in looks)
    nat_fail = ('a first, unpractised movement is rejected as "not an approach" more often than a practised one'
                if nat_more else 'no excess rejection of first, unpractised movements was shown')
    detector_reason = (f"learned (with no stillness v0 scores the return in {v('stress_vzero_0_ret')} of the recordings, "
                       f"the production pipeline in {v('stress_production_0_ret')})")
    orient = []
    if B:
        bb = pd.DataFrame(B['rows'])
        orient = sorted(bb.loc[(bb['stratum'] == 'orientation') & (bb['name'] == B['production']), 'level']
                        .astype(str).unique())
    grips_text = (f"Recorded grips: {', '.join(o.replace('_', ' ') for o in orient)}; any other grip is untested."
                  if orient else '')

    def stop_line(h: str) -> str:
        return (f"{lab[h]}: accuracy {v(f'stopcal_{h}_accuracy_half')} with {v(f'stopcal_{h}_attempts')} accepted attempts, "
                f"log loss {v(f'stopcal_{h}_log_loss')}, ECE {v(f'stopcal_{h}_ece')} (null band {v(f'stopcal_{h}_ece_null_lo')}"
                f"-{v(f'stopcal_{h}_ece_null_hi')}), slope {v(f'stopcal_{h}_slope')} {ci(f'stopcal_{h}_slope')}: "
                f"{'consistent with calibration' if stop_ok.get(h) == 'yes' else 'NOT consistent with calibration'}")

    text = f"""# Lane m4: the cue-free production pipeline

All numbers below are regenerated by `python -m stage2.pipeline numbers --lanes m4` from the saved results
(`results/m4_numbers.json`, LaTeX macros `\\MFour...` in `results/numbers.tex`); this file is rewritten by that step.
**Accuracy convention**: an abstention (no approach found, P(right) = 0.5) scores HALF a correct answer everywhere
in this lane, as in the harness since 2026-10-28 (the former `P >= 0.5` rule counted it as a right-hand answer,
which inflates the all-right natural-use set); coverage, accuracy on the answered recordings and forced-choice
accuracy are given beside it.
Significance: exact sign-flip tests over persons, Holm-adjusted within a named family (`trial`: the
{v('paired_family_n')} trial-level comparisons of `m4_paired`; `stress`: the {v('stress_family_n')} comparisons of the
stress test; `sessions`: the {v('sess_family_n')} session comparisons; `calibrators`: the calibrators against none).

## Goal

In production (the Stage 3 inference link) nobody marks when a movement starts: the person holds the phone at arm's
length and brings it to the nose once or a few times, and the pipeline must find the approach movement(s) in the
continuous stream, ignore the return to arm's length (a time-reversed approach that predicts the OTHER hand) and any
other movement, decide when to stop, and output a calibrated P(right) from sensor data only.

## What was built

1. **Candidate segments** (`lanes/m4_detect.py`): a causal activity signal (|w|, 2nd-order {v('seg_smooth_hz')} Hz
   one-pass low-pass, bounds moved back by its group delay of {v('causal_delay_ms')} ms) and hysteresis bouts (still
   below lo for at least end_sustain ends a bout; above {v('seg_hi_dps')} deg/s for {v('seg_sustain_ms')} ms makes it a
   movement), with NO stillness requirement. Three (lo, end_sustain) settings are compared INSIDE each fold
   (`SEG_GRID`): `{v('prod_seg_cfg_mode')}` in {v('prod_seg_cfg_mode_folds')} of {v('prod_folds')} folds.
2. **A hand-agnostic segment classifier** (`m4_models.SegmentClassifier`, multinomial logistic regression on robust-
   scaled, winsorised features, C = 1): approach / return / other, trained LOSO on labels from the app's own events.
   Feature sets compared: `physics`, `full` (+ the stillness before capped at {v('seg_still_cap_s')} s and recording
   context), `nested` (physics or full chosen inside each fold: `{v('nested_features_chosen_mode')}` in
   {v('nested_features_chosen_mode_folds')} folds), `robust` (physics + the stillness before capped at the
   INSTRUCTED {inst} s + the stillness after; **the production set**) and `grip`: robust with the four screen-normal
   quantities signed by the side of the phone that faces up over the segment, which makes every feature exactly
   invariant to the screen-away grip (tests) but discards information, because that side is a poor grip estimate
   on real recordings (the sign is -1 for {v('grip_sign_neg_approach')} of the approach and
   {v('grip_sign_neg_return')} of the return segments of screen-toward recordings). The strongest single features:
   the displacement along the screen normal (AUC approach vs return {v('auc_disp_z0_vs_return')}; grip-signed
   {v('auc_g_disp_z0_vs_return')}), the change of the up vector's screen-normal component
   ({v('auc_d_up_z_vs_return')}, i.e. the reverse ordering), the vertical displacement
   ({v('auc_disp_vert_vs_return')}) and the capped stillness before ({v('auc_still_before_inst_vs_other')} against
   other movements). Every feature is even under the left/right mirror (tests).
3. **The hand classifier is pluggable** (`m4_models.HAND_MODELS`, `register_hand_model`): the reference `yaw8`
   (mirror-augmented logistic on the cumulative yaw at 8 instants) is the default; `chiral16` is the second. Windows are
   cut with the SAME `standardize.canonical` code as the training windows, from samples the online detector has seen
   when it decides.
4. **Calibration of one touch**: the calibrator is chosen INSIDE each fold among the antisymmetric ones (none,
   temperature, symmetrised isotonic) by a person-grouped cross-validation of the cross-fitted logits (folds: {chosen}),
   so P(mirror image) = 1 - P holds exactly.
5. **Several touches** (`m4_online.py`, `m4_aggregate.py`): every bout classified as an approach is an attempt
   (production attempt rule: {rule}); the attempts' calibrated logits are aggregated as P_k = sigmoid(s_k x mean
   logit) with the production aggregation **{lab.get(pa, pa)}** (declared rule, below); stop when P_k leaves
   [1 - p_stop, p_stop] (p_stop {v('p_stop')}, a product choice fixed before the sessions were evaluated) or after
   {v('k_max')} attempts.
6. **The grip check of the inference page** (`m4_detect.rest_up_z`, model-free): the screen-normal component of the
   up vector while the phone rests at arm's length before the first movement must be at least {v('grip_gate_upz')};
   otherwise the page asks to turn the screen towards the face and does not answer (Robustness).
7. **Production callable** (`m4_production.fit_production` / `predict_stream`) and the synthesis finalist
   (`m4_finalist.build_x` / `factory`, window-only hand-model plug-in).

Evaluation: trial-level experiments through `cv.run_experiment` (unit = one full recording: rest, approach, hold,
return; X = indices into label-free streams without the event-derived phase codes; every fold leak-checked).
Stream-level analyses (stress grid, synthetic sessions, stopping rule, replay) reuse each fold's fitted pipeline on
the same LOSO splits and bootstrap persons (`cv.person_bootstrap`, {N_BOOT} resamples).

## Experiments

Accuracy: half credit for an abstention. Natural use: answered / scored, then the half-credit accuracy. Configs:
`n_configs_tried` recorded in the result and the registry (the production pipeline carries all
{v('n_configs_final')} cue-free pipelines compared on the outer folds; its detector the {v('n_configs_detector')}
feature sets; everything else is one pre-specified configuration).

| Result JSON | Pipeline | Accuracy [95 % CI] | AUC | Log loss | ECE | Answered | Natural use | Configs |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(exp_lines)}

Derived analyses (same LOSO splits; `tables/m4/`): `m4_scores.json` (half-credit, answered and forced-choice
accuracy of every result, primary and natural use; the P(approach) threshold ablation), `m4_detection.json`,
`m4_stress.json` + `m4_stress_details.csv` (stillness grid and screen-away twins), `m4_sessions.json` +
`m4_sessions_krows.csv`, `_stop.csv`, `_answer.csv`, `_detection.csv`, `_attempts.csv`, `_scales.csv` (synthetic
sessions), `m4_calibration.json` (one touch and several touches: ECE null bands, calibration slopes, reliability,
paired differences, the production aggregation), `m4_natural.json` (natural-use failure analysis), `m4_replay.json`
(online front-end replay), `m4_latency.json`, `m4_breakdown.json`, `m4_segfeat.json`, `m4_paired.json`,
`m4_timeline.json`, `m4_demo.json`, and per-recording `<result>_details.csv`.

## Headline (production pipeline `{prod}`)

- **One touch, cue-free: {v('prod_acc')} {ci('prod_acc')}** on {v('n_recordings')} full recordings of
  {v('n_persons')} persons (LOSO, person-bootstrap 95 % CIs); AUC {v('prod_auc')} {ci('prod_auc')}; log loss
  {v('prod_log_loss')} {ci('prod_log_loss')}; Brier {v('prod_brier')}; ECE {v('cal_nested_ece')} (perfect-calibration
  null band {v('cal_nested_ece_null_lo')}-{v('cal_nested_ece_null_hi')}), calibration slope {v('cal_nested_slope')}
  {ci('cal_nested_slope')}. It answers {v('prod_coverage')} of the recordings ({v('prod_n_answered_tie')} of the
  {v('prod_n_answered')} answers are exactly 50 %, the flat centre of the symmetrised isotonic calibrator: no hand
  shown), {v('prod_acc_answered')} right on the answered (a 50 % answer scores half); forced choice (always the best
  candidate) {v('prod_acc_forced')}.
- **Natural use** (one first free-choice trial per person, all right hands): answered {v('prod_nat_answered')} of
  {v('prod_nat_n')} ({v('prod_nat_coverage')}; {v('prod_nat_answered_tie')} of them at exactly 50 %), right on
  {v('prod_nat_acc_answered')} of those; forced choice
  {v('prod_nat_acc_forced')}; half credit **{v('prod_nat_acc')} {ci('prod_nat_acc')}** (the oracle window:
  {v('oracle_nat_acc')}). The harness's former rule (until 2026-10-28) gives {v('prod_nat_acc_harness')}: it counts
  every abstention as a correct right-hand answer and must not be cited.
- **Several touches, cue-free** (synthetic sessions: {v('sess_n')} per setting, {v('sess_recordings')} recordings of
  one person and hand each, {v('sess_ref_keep_s')} s of stillness at each junction): up to 3 accepted attempts
  **{v(f'sess_{kr}_k3_acc')} {ci(f'sess_{kr}_k3_acc')}** (1: {v(f'sess_{kr}_k1_acc')}, 5: {v(f'sess_{kr}_k5_acc')}).
  The stopping rule (p_stop {v('p_stop')}, at most {v('k_max')} attempts, {lab.get(pa, pa)}): **{v(f'stop_{kr}_acc')}
  {ci(f'stop_{kr}_acc')}** with {v(f'stop_{kr}_attempts')} accepted attempts and {v(f'stop_{kr}_touches')} true
  touches by the decision on average; AUC {v(f'stop_{kr}_auc')}; ECE {v(f'stopcal_{pa}_ece')}; log loss
  {v(f'stopcal_{pa}_log_loss')}. The harness k = 3 of the same pipeline, {v('prod_acc_k3_indep')}
  {ci('prod_acc_k3_indep')}, averages three INDEPENDENT recordings of one person and hand, each with its best
  candidate; it is not a cue-free session and must not be compared with the oracle-window k = 3 of other lanes as if
  it were.
- **Detection loss** (oracle window {v('oracle_acc')} {ci('oracle_acc')}): {v('d_prod_vs_oracle')}
  {ci('d_prod_vs_oracle')} ({verdict('prod_vs_oracle', 'the production pipeline', 'the oracle window')}). Against v0
  ({v('vzero_acc')}): {v('d_prod_vs_v0')} {ci('d_prod_vs_v0')} ({verdict('prod_vs_v0', 'the production pipeline', 'v0')}).
- **Latency**: the decision comes {v('lat_touch_ms')} ms after the first screen touch (median; 90th percentile
  {v('lat_touch_p90_ms')} ms).

## Calibration of one touch (problem, detection, remedy, evidence)

- Detected by: the reliability diagram, the ECE against its perfect-calibration null band and the calibration slope
  (`M4_05_calibration`, table `m4_calibration`). Uncalibrated: ECE {v('cal_none_ece')} (null band
  {v('cal_none_ece_null_lo')}-{v('cal_none_ece_null_hi')}, p = {v('cal_none_ece_p_null')}), slope {v('cal_none_slope')}
  {ci('cal_none_slope')}, intercept {v('cal_none_intercept')} {ci('cal_none_intercept')}:
  {'consistent with calibration' if cal_ok.get('none') == 'yes' else 'NOT consistent with calibration'}.
- Why it matters: the product is the percentage shown to the user.
- Remedies (fitted on cross-fitted predictions of the training persons only; paired log loss minus uncalibrated,
  Holm over the calibrators): temperature {v('cal_temperature_d_ll')} {ci('cal_temperature_d_ll')}, Platt
  {v('cal_platt_d_ll')} {ci('cal_platt_d_ll')}, isotonic {v('cal_isotonic_d_ll')} {ci('cal_isotonic_d_ll')} (Holm p
  {v('cal_isotonic_d_ll_p_holm')}), beta {v('cal_beta_d_ll')} {ci('cal_beta_d_ll')}, symmetrised isotonic
  {v('cal_isotonic_sym_d_ll')} {ci('cal_isotonic_sym_d_ll')}. Not antisymmetric, so a mirror-image recording no
  longer gets exactly 1 - P although the mirror-augmented hand model guarantees it: {non_anti_text}; they are compared
  but never chosen (the antisymmetric ones have a mirror deviation of {v('cal_nested_mirror_dev_max')}).
- Decision: the calibrator is chosen inside each fold among none, temperature and symmetrised isotonic (nested; folds:
  {chosen}). Evidence: production minus uncalibrated, log loss {v('cal_nested_d_ll')} {ci('cal_nested_d_ll')} (Holm p
  {v('cal_nested_d_ll_p_holm')}), Brier {v('cal_nested_d_brier')} {ci('cal_nested_d_brier')}, accuracy
  {v('cal_nested_d_acc')} {ci('cal_nested_d_acc')}; ECE {v('cal_nested_ece')} (null {v('cal_nested_ece_null_lo')}-
  {v('cal_nested_ece_null_hi')}), slope {v('cal_nested_slope')} {ci('cal_nested_slope')}:
  {'consistent with calibration' if cal_ok.get('nested') == 'yes' else 'NOT consistent with calibration'}.
- Per platform (production): Android ({v('cal_nested_android_n')} recordings) accuracy {v('cal_nested_android_acc')}
  {ci('cal_nested_android_acc')}, ECE {v('cal_nested_android_ece')} (null {v('cal_nested_android_ece_null_lo')}-
  {v('cal_nested_android_ece_null_hi')}), slope {v('cal_nested_android_slope')}, log loss
  {v('cal_nested_android_log_loss')}, Brier {v('cal_nested_android_brier')}; iOS ({v('cal_nested_ios_n')}) accuracy
  {v('cal_nested_ios_acc')} {ci('cal_nested_ios_acc')}, ECE {v('cal_nested_ios_ece')} (null
  {v('cal_nested_ios_ece_null_lo')}-{v('cal_nested_ios_ece_null_hi')}), slope {v('cal_nested_ios_slope')}, log loss
  {v('cal_nested_ios_log_loss')}, Brier {v('cal_nested_ios_brier')}.

## Calibration of several touches (problem, detection, remedy, evidence)

- Detected by: the reliability of the multi-touch answers on the synthetic sessions (`M4_06_k3_calibration`, table
  `m4_stop_calibration`): the answer of the stopping rule (p_stop {v('p_stop')}) under each aggregation:
  - {stop_line('scale_recordings')};
  - {stop_line('scale_sessions')};
  - {stop_line('mean')}.
- Why it matters: the answer is the percentage the user sees, and an over-confident aggregate also stops too early.
- Cause of the difference: accepted attempts are not independent pieces of evidence: {v('sessdet_precision')} of them
  are (part of) an approach ({v('sessdet_false_return')} returns and {v('sessdet_false_other')} other movements among
  {v('sessdet_attempts')}), and the attempts of one session share the person's bias. The scale fitted on random
  recordings (median s_3 = {v('scale_recordings_s3')}) assumes neither; the session-fitted scale s_k = s_1 k^b (median
  b = {v('scale_sessions_b')}, s_3 = {v('scale_sessions_s3')}) is learned on cross-fitted sessions of the training
  persons; the mean logit fits nothing.
- Decision (declared rule `m4_config.AGGREGATION_RULE`, applied to the saved results: the aggregations consistent with
  calibration, then the fewest fitted parameters): **{lab.get(pa, pa)}**. This is an exploratory choice on the outer
  sessions (attempt rule x aggregation: {v('n_configs_sessions')} configurations); the session-fitted scale was added
  after the held-out sessions had shown the problem.
- Paired evidence (same sessions, family `sessions`, Holm): recording-fitted minus session-fitted scale: accuracy
  {v('sp_agg_scale_recordings_vs_scale_sessions_d_acc')}
  {ci('sp_agg_scale_recordings_vs_scale_sessions_d_acc')}, log loss
  {v('sp_agg_scale_recordings_vs_scale_sessions_d_ll')} {ci('sp_agg_scale_recordings_vs_scale_sessions_d_ll')}
  ({sp_verdict('agg_scale_recordings_vs_scale_sessions', 'the recording-fitted scale', 'the session-fitted scale', 'log loss')});
  session-fitted minus mean logit: log loss {v('sp_agg_scale_sessions_vs_mean_d_ll')}
  {ci('sp_agg_scale_sessions_vs_mean_d_ll')} ({sp_verdict('agg_scale_sessions_vs_mean', 'the session-fitted scale', 'the mean logit', 'log loss')}),
  accuracy {v('sp_agg_scale_sessions_vs_mean_d_acc')} {ci('sp_agg_scale_sessions_vs_mean_d_acc')}.
- Confident stops (P beyond p_stop for one hand), production aggregation: right {v(f'stopcal_{pa}_conf_acc')}
  {ci(f'stopcal_{pa}_conf_acc')} of the time while claiming {v(f'stopcal_{pa}_conf_claimed')} on average
  ({v(f'stopcal_{pa}_conf_n')} sessions); recording-fitted scale: {v('stopcal_scale_recordings_conf_acc')}
  {ci('stopcal_scale_recordings_conf_acc')} while claiming {v('stopcal_scale_recordings_conf_claimed')}.
- p_stop sets touches against certainty (exploratory reading of the same sessions, table `m4_stopping`):
  {', '.join(f"p_stop {k[6:10].replace('p', '.')}: {v(k)} with {v(k.replace('_acc', '_attempts'))} attempts" for k in sorted(n) if k.startswith('pstop_') and k.endswith('_acc'))}.
- Abstain after {v('k_max')} attempts unless max(P, 1 - P) >= p_answer:
  {', '.join(f"p_answer {k[7:11].replace('p', '.')}: answers {v(k)}, right on {v(k.replace('_coverage', '_acc'))}" for k in sorted(n) if k.startswith('answer_') and k.endswith('_coverage'))}.
- Per platform (production aggregation, answer at the stop): Android {v('stopcal_android_acc')} (ECE
  {v('stopcal_android_ece')}, null {v('stopcal_android_ece_null_lo')}-{v('stopcal_android_ece_null_hi')}; log loss
  {v('stopcal_android_log_loss')}; {v('stopcal_android_n')} sessions of {v('stopcal_android_persons')} persons), iOS
  {v('stopcal_ios_acc')} (ECE {v('stopcal_ios_ece')}, null {v('stopcal_ios_ece_null_lo')}-{v('stopcal_ios_ece_null_hi')};
  log loss {v('stopcal_ios_log_loss')}; {v('stopcal_ios_n')} sessions of {v('stopcal_ios_persons')} persons).

## Several touches: what the accepted attempts are

- Up to k accepted attempts, {v('sess_ref_keep_s')} s junctions: k = 1 {v(f'sess_{kr}_k1_acc')}, 2
  {v(f'sess_{kr}_k2_acc')}, 3 {v(f'sess_{kr}_k3_acc')}, 4 {v(f'sess_{kr}_k4_acc')}, 5 {v(f'sess_{kr}_k5_acc')};
  sessions with at least 5 accepted attempts: {v(f'sess_{kr}_k5_avail')}.
- Keeping only the attempts that are true approaches (the stored events; the attainable curve): k = 1
  {v(f'sesstrue_{kr}_k1_acc')}, 3 {v(f'sesstrue_{kr}_k3_acc')} {ci(f'sesstrue_{kr}_k3_acc')}, 5 {v(f'sesstrue_{kr}_k5_acc')}.
- One accepted attempt is right {v('kind_approach_acc')} of the time when it is the approach ({v('kind_approach_n')}
  attempts), {v('kind_approach_partial_acc')} when part of it, {v('kind_other_before_acc')} when it is the arm settling
  before the approach ({v('kind_other_before_n')}), {v('kind_return_acc')} when it is a return ({v('kind_return_n')})
  and {v('kind_other_after_acc')} for other later movements ({v('kind_other_after_n')}). {multi_touch}
- Attempt rule (paired, family `sessions`): {rule} minus {other_rule}, stopping rule with {lab.get(pa, pa)}: accuracy
  {v(f'sp_rule_{rule_key}_vs_{other_key}_{pa}_d_acc')} {ci(f'sp_rule_{rule_key}_vs_{other_key}_{pa}_d_acc')}
  ({sp_verdict(f'rule_{rule_key}_vs_{other_key}_{pa}', rule, other_rule)}, {v(f'sp_rule_{rule_key}_vs_{other_key}_{pa}_discordant')}
  discordant sessions); recall of true approaches {v('sessdet_recall')} against {v('sessdetalt_recall')}, precision
  {v('sessdet_precision')} against {v('sessdetalt_precision')}. The pairing rule (the bout after an accepted approach
  is its return) can discard a true approach when an arm-settling movement before it was accepted.
- With no stillness at the junctions: up to 3 attempts {v('sess_0_k3_acc')}; with 0.5 s {v('sess_0p5_k3_acc')}; 2 s
  {v('sess_2_k3_acc')}.

## Data-treatment decisions (options, evidence, decision)

Accuracy differences are paired on the same recordings (half credit; person bootstrap; Holm over the family
`trial`). Verdicts are generated from those numbers.

| Choice | Options | Evidence | Verdict / decision |
|---|---|---|---|
| Detector | v0 / learned classifier | v0 {v('vzero_acc')}, production {v('prod_acc')}; paired {v('d_prod_vs_v0')} {ci('d_prod_vs_v0')}; log loss {v('d_prod_vs_v0_ll')} {ci('d_prod_vs_v0_ll')} | accuracy: {verdict('prod_vs_v0', 'learned', 'v0')}; log loss: {ll_verdict('prod_vs_v0', 'learned', 'v0')}; {detector_reason} |
| Segment features | physics / full / nested / robust / grip | physics {v('physics_acc')}, full {v('full_acc')}, nested {v('nested_acc')}, robust {v('robust_acc')}, grip {v('grip_acc')}; robust - nested {v('d_robust_vs_nested')} {ci('d_robust_vs_nested')}; grip - robust {v('d_grip_vs_robust')} {ci('d_grip_vs_robust')}; {inst} s of stillness: full {v(f'stress_uncapped_{kinst}_acc')} (answers {v(f'stress_uncapped_{kinst}_cov')}), robust {v(f'stress_robust_uncal_{kinst}_acc')} ({v(f'stress_robust_uncal_{kinst}_cov')}), grip {v(f'stress_grip_uncal_{kinst}_acc')} ({v(f'stress_grip_uncal_{kinst}_cov')}); screen away: robust {v('stressflip_robust_uncal_rec_acc')}, grip {v('stressflip_grip_uncal_rec_acc')} | robust (production; an exploratory choice among {v('n_configs_detector')} sets): grip - robust {verdict('grip_vs_robust', 'grip', 'robust')}; the screen-away grip is handled by the instructions and the page's grip check (Robustness) |
| Hand training windows | oracle / detector | detector - oracle {v('d_detwin_vs_oraclewin')} {ci('d_detwin_vs_oraclewin')} | {verdict('detwin_vs_oraclewin', 'detector windows', 'oracle windows')}; oracle windows (the cached arrays, any hand model plugs in) |
| Activity filter | causal / zero-phase (both uncalibrated) | zero-phase - causal {v('d_zp_activity_vs_prod')} {ci('d_zp_activity_vs_prod')} | {verdict('zp_activity_vs_prod', 'zero-phase', 'causal')}; causal (runs online) |
| Stream low-pass | zero-phase over the buffer / causal (both uncalibrated) | causal - zero-phase {v('d_causal_lowpass_vs_prod')} {ci('d_causal_lowpass_vs_prod')}; zero-phase over the buffer received so far (replay) - over the whole recording {v('replay_d_acc')} {ci('replay_d_acc')} | {verdict('causal_lowpass_vs_prod', 'causal', 'zero-phase')}; zero-phase over the buffer received so far (the training code path; replay below) |
| Feature scaling | robust (median / IQR, clip 5) / z-score / none | z - robust {v('d_abl_scale_z_vs_base')} {ci('d_abl_scale_z_vs_base')}; none - robust {v('d_abl_scale_none_vs_base')} {ci('d_abl_scale_none_vs_base')} | {verdict('abl_scale_z_vs_base', 'z-score', 'robust')}; {verdict('abl_scale_none_vs_base', 'none', 'robust')}; robust kept (pre-specified) |
| Segment-classifier C | 1 / nested (0.1, 1, 10) | nested - 1 {v('d_abl_segc_nested_vs_base')} {ci('d_abl_segc_nested_vs_base')} | {verdict('abl_segc_nested_vs_base', 'nested C', 'C = 1')}; C = 1 kept |
| Look-ahead after a segment | {look_opts} | {look_ev} | {look_dec}; {v('lookahead_s')} s kept (pre-specified; the look-ahead adds directly to the latency) |
| P(approach) threshold | {', '.join(kk.replace('p', '.') for kk in thr_keys)} | accuracy {thr_list} | {thr_decision} (`M4_07`) |
| Return evidence | approach only / + return head | stored windows: sum - approach {v('d_ret_sum_vs_app')} {ci('d_ret_sum_vs_app')}; cue-free {v('d_return_vs_prod')} {ci('d_return_vs_prod')}; the return ends {v('return_wait_ms')} ms after the approach (median) | {verdict('ret_sum_vs_app', 'approach + return', 'approach only')} (stored); {verdict('return_vs_prod', 'with the return head', 'without')} (cue-free); not in production yet (needs its own calibration; Next steps) |
| Hand model | yaw8 / chiral16 | oracle {v('d_oracle_chiral16_vs_yaw8')} {ci('d_oracle_chiral16_vs_yaw8')}; cue-free {v('d_chiral16_vs_yaw8')} {ci('d_chiral16_vs_yaw8')} | {verdict('chiral16_vs_yaw8', 'chiral16', 'yaw8')}; yaw8 kept as the pre-specified default; pluggable |
| Calibrator (one touch) | none / temperature / Platt / isotonic / beta / symmetrised isotonic / nested | table `m4_calibration` | nested among the antisymmetric ones |
| Attempt rule | pairing / plain | sessions, paired above | {rule} |
| Aggregation of touches | s_k on recordings / s_k on sessions / mean logit | table `m4_stop_calibration` | {lab.get(pa, pa)} (declared rule) |

## Robustness

- **Screen-away grip** (`M4_12_screen_flip`). Detected by: every held-out recording also scored as its screen-away
  twin (the phone turned 180 deg about its long axis). The production pipeline: {v('stress_production_rec_acc')}
  as recorded, {v('stressflip_production_rec_acc')} {ci('stressflip_production_rec_acc')} screen away; it scores the
  return in {v('stressflip_production_rec_ret')} of the twins and its answers are right {v('stressflip_production_rec_acc_answered')}
  of the time. Why it matters: a confident wrong percentage. Cause: the strongest segment features are device-frame
  quantities along the screen normal (Segment features above), which change sign with the grip.
  **Remedy, the page's rest-pose grip check** (`m4_detect.rest_up_z`; threshold `m4_config.GRIP_GATE_UPZ` =
  {v('grip_gate_upz')}): while the phone rests at arm's length before the first movement, the screen-normal up
  component must be at least the threshold, otherwise the page asks to turn the screen towards the face and does not
  answer. Evidence (paired, production with the check minus without; Holm within the family `grip`): on the
  screen-away twins it stops {v('stressflip_production_rec_gate')} of the recordings and the share answered WRONGLY
  falls from {v('gripwrongflip_rec_without')} to {v('gripwrongflip_rec_with')} ({v('gripwrongflip_rec_d')}
  [{v('gripwrongflip_rec_lo')}, {v('gripwrongflip_rec_hi')}], Holm p {v('gripwrongflip_rec_p_holm')}); their half-credit
  accuracy moves from {v('stressflip_production_rec_acc')} to {v('stressflip_production_rec_acc_gated')}
  {ci('stressflip_production_rec_acc_gated')}, because a stopped recording scores as an abstention. On the
  screen-toward recordings it stops {v('stress_production_rec_gate')} (answered {v('stress_production_rec_cov')} to
  {v('stress_production_rec_cov_gated')}; wrong {v('gripwrong_rec_without')} to {v('gripwrong_rec_with')},
  {v('gripwrong_rec_d')} [{v('gripwrong_rec_lo')}, {v('gripwrong_rec_hi')}]; accuracy
  {v('stress_production_rec_acc_gated')} against {v('stress_production_rec_acc')}, paired
  {v('stresspair_grip_check_rec_d')} [{v('stresspair_grip_check_rec_lo')}, {v('stresspair_grip_check_rec_hi')}], Holm p
  {v('stresspair_grip_check_rec_p_holm')} in the family `stress`). {first_remedy} The grip-normalised alternative is invariant by
  construction ({v('stress_grip_uncal_rec_acc')} and {v('stressflip_grip_uncal_rec_acc')}) but changes the accuracy on
  the recordings by {v('d_grip_vs_robust')} {ci('d_grip_vs_robust')} ({verdict('grip_vs_robust', 'grip', 'robust')}){keep_robust}
  The check reads gravity only, so a phone held near vertical or tilted forward is asked to adjust. {grips_text}
- **Stillness before the approach** (`M4_03_stress`, table `m4_stress`): production {stress_list} (answered with none:
  {v('stress_production_0_cov')}); v0 with none {v('stress_vzero_0_acc')} (return scored in {v('stress_vzero_0_ret')}).
  Robust minus full (uncapped context), both uncalibrated, paired, Holm over the family `stress`: {pair_list}.
  Significant (Holm p < 0.05, CI excluding 0) at: {stress_sig}. {stillness_req}
- **Natural use** (table `m4_natural`): the detector rejects {v('nat_rejected')} of {v('nat_n')} first free-choice
  approaches ({v('nat_rejection_rate')} against {v('prim_rejection_rate')} on the primary set; {v('nat_expected_rejections')}
  expected from each person's own primary rejection rate, Poisson-binomial p = {v('nat_p_poisson')}; Fisher p =
  {v('nat_p_fisher')}). A candidate segment is the stored approach in {v('nat_approach_found')} of the natural
  recordings ({v('prim_approach_found')} primary), and that candidate's P(approach) is below the threshold in
  {v('nat_approach_p_below')} of them ({v('prim_approach_p_below')} primary): {nat_test}. Stored-approach
  candidates, natural against primary (medians): displacement along the screen normal
  {v('natfeat_disp_z0_nat')} m against
  {v('natfeat_disp_z0_prim')} m ({v('natfeat_disp_z0_below')} below the primary 25th percentile), vertical
  displacement {v('natfeat_disp_vert_nat')} against {v('natfeat_disp_vert_prim')} m, duration {v('natfeat_dur_s_nat')}
  against {v('natfeat_dur_s_prim')} s, capped stillness before {v('natfeat_still_before_inst_nat')} against
  {v('natfeat_still_before_inst_prim')} s, P(approach) {v('natfeat_p_approach_nat')} against
  {v('natfeat_p_approach_prim')}. {nat_grip}
- **Online front end** (replay, `m4_replay.json`): every candidate re-computed from the raw record re-standardised up
  to its decision time (the zero-phase low-pass over the buffer received so far, as the page would): {v('replay_matched')}
  of the segments are found again; decisions changed in {v('replay_changed')} of the recordings (statuses in
  {v('replay_status_changed')}); max |dP| {v('replay_max_dp')}, median {v('replay_median_dp')}; accuracy
  {v('replay_d_acc')} {ci('replay_d_acc')}. The full-buffer replay reproduces the cached streams (max |gyro
  difference| {v('replay_full_check')} deg/s). A fully causal low-pass costs {v('d_causal_lowpass_vs_prod')}
  {ci('d_causal_lowpass_vs_prod')} ({verdict('causal_lowpass_vs_prod', 'causal', 'zero-phase')}).

## The return as evidence (ablation A10)

- Stored windows of the trials with a clean return (n = {v('ret_app_n')}): approach model {v('ret_app_acc')}
  {ci('ret_app_acc')}, return head alone {v('ret_head_acc')} {ci('ret_head_acc')}, the sum of the two logits
  {v('ret_sum_acc')} {ci('ret_sum_acc')}; paired sum - approach {v('d_ret_sum_vs_app')} {ci('d_ret_sum_vs_app')}
  ({verdict('ret_sum_vs_app', 'approach + return', 'approach only')}).
- Cue-free (production detector, uncalibrated): adding the return head's logit when the segment after the chosen
  approach is classified as a return gives {v('prod_return_acc')} against {v('robust_acc')}; paired
  {v('d_return_vs_prod')} {ci('d_return_vs_prod')} ({verdict('return_vs_prod', 'with the return head', 'without')}).
- Price: the return's evidence arrives a median {v('return_wait_ms')} ms (90th percentile {v('return_wait_p90_ms')}
  ms) after the end of the approach. Between two touches the return happens anyway, so the wait would delay only the
  final answer.

## Per platform and condition (production pipeline, one touch)

| Stratum | Level | Trials | Persons | Accuracy [95 % CI] | Answered | ECE |
|---|---|---|---|---|---|---|
{chr(10).join(br)}

Strata with fewer than {B.get('min_persons_ci', 5) if B else 5} persons have no CI.

## Online operation (what runs on the phone)

1. Every incoming sample (60 Hz, standardised) updates the causal activity |w| (one-pass 2nd-order
   {v('seg_smooth_hz')} Hz Butterworth; its group delay, {v('causal_delay_ms')} ms, is subtracted from every bound).
2. **Grip check** (before the first touch): while the phone rests at arm's length, the screen-normal up component
   (`m4_detect.rest_up_z`) must be at least {v('grip_gate_upz')}; otherwise the page asks to turn the screen towards
   the face and answers nothing.
3. A bout starts at the first sample above the stillness threshold after a stillness of at least end_sustain (the onset
   is back-dated to it); it is a movement once |w| > {v('seg_hi_dps')} deg/s for {v('seg_sustain_ms')} ms.
4. **End of approach**: the first sample of a stillness of end_sustain at the face (deceleration to rest; the contact
   transient is not needed). The segment is confirmed {v('lookahead_s')} s later (look-ahead), when its window is
   complete; `standardize.canonical` cuts the window from that buffer.
5. The segment classifier labels the bout. Returns and other movements are dropped; every approach adds its
   calibrated logit (attempt rule: {rule}).
6. **Stopping rule**: P_k = sigmoid(s_k x mean logit) ({lab.get(pa, pa)}) after each attempt; stop at max(P, 1 - P) >=
   {v('p_stop')} or after {v('k_max')} attempts. Latency from the touch to the decision: {v('lat_touch_ms')} ms
   (median; 90th percentile {v('lat_touch_p90_ms')} ms).
7. **One code path**: the same `m4_detect.segment_table` runs online and offline (a test replays a stream cut at each
   decision time; the replay above re-standardises the raw record up to each decision time).
8. **Single-touch abstain rule** (`M4_07_coverage`): with max(P, 1 - P) >= {v('abstain_tau')} the pipeline answers
   {v('abstain_coverage')} of the recordings with accuracy {v('abstain_acc')}.

## Recommendation for the inference page

- **Algorithm** (`m4_production.fit_production` / `predict_stream`, parameters in `model.describe()`): causal
  {v('seg_smooth_hz')} Hz activity, hysteresis segments (the fold-chosen lo / end_sustain; {v('seg_hi_dps')} deg/s for
  {v('seg_sustain_ms')} ms), look-ahead {v('lookahead_s')} s, approach / return / other classifier on the robust
  features (stillness seen up to {inst} s), P(approach) >= {v('det_threshold')}, every approach an attempt, yaw8 on
  the canonical window, the nested antisymmetric calibrator, aggregate {lab.get(pa, pa)}, stop at max(P, 1 - P) >=
  {v('p_stop')} or after {v('k_max')} attempts; the rest-pose grip check in front of it.
- **User instructions**: "Hold the phone upright with the screen facing you and tilted slightly upwards, still, at
  arm's length for at least {inst} s; bring it to your nose and touch it; hold a moment; bring it back to arm's
  length. Repeat until the page tells you it is sure (at most {v('k_max')} times)." The page must check the {inst} s
  of stillness (the causal activity below the stillness threshold) before it accepts a touch, and say "hold still"
  otherwise; and it must apply the grip check (screen facing down or away at rest: "turn the screen towards you").
- **Expected accuracy**: one touch {v('prod_acc')} {ci('prod_acc')}; the stopping rule {v(f'stop_{kr}_acc')}
  {ci(f'stop_{kr}_acc')} with {v(f'stop_{kr}_attempts')} accepted attempts (synthetic sessions spliced from recordings
  made with a GO cue). Natural use: answered {v('prod_nat_coverage')} of the first free-choice movements
  ({nat_test}).
- **Failure modes**: {nat_fail}; movements just before an
  approach (the arm settling at arm's length) accepted as attempts ({v('sessdet_false_other_before')} of the
  {v('sessdet_false_other')} other movements accepted lie before the approach of their recording) and returns accepted
  ({v('sessdet_false_return')}); {stillness_fail}; a screen held away from the face (stopped by the grip check in
  {v('stressflip_production_rec_gate')} of the screen-away twins; without it their answers are right
  {v('stressflip_production_rec_acc_answered')} of the time) or near
  vertical (the check asks {v('stress_production_rec_gate')} of the screen-toward recordings to adjust); persons whose
  approaches the hand model misreads stay wrong however many touches they make.
- **Next steps**: reject pre-approach settling movements and returns (a sequence model over the bouts: rest,
  approach, hold, return), whose ceiling is the true-approach curve above; the return head in the stopping rule with
  its own cross-fitted calibration; stillness and grip augmentation of the training streams; a rehearsal session with
  real cue-free streams; the synthesis's final hand model plugged in (`register_hand_model`, `factory(hand_model=...)`,
  window-only contract).

## Figures

{chr(10).join(fig_lines)}

## Regenerate

```
cd analysis
.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m4 --force   # experiments + derived data (about 1-2 h on 2 threads)
.venv/Scripts/python.exe -m stage2.pipeline figures --lanes m4           # figures from saved files (seconds)
.venv/Scripts/python.exe -m stage2.pipeline tables --lanes m4            # CSV + LaTeX tables
.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes m4           # numbers + this summary
.venv/Scripts/python.exe -m pytest stage2/tests/test_m4.py -q -p no:cacheprovider
```

`compute --lanes m4 --only m4_sessions m4_calibration m4_demo --force` reruns only the session analyses;
`figures --lanes m4 --only M4_06_k3_calibration` redraws one figure.

## Changes in this revision (after the audit of the first version)

- Abstentions score half credit everywhere, and natural use is reported as coverage, accuracy on the answered,
  forced choice and half credit: the first version's natural-use figure counted every abstention as a correct
  right-hand answer on an all-right set.
- The memo of fitted pipelines is keyed on the CONTENT of the prepared data (it had let the causal low-pass variant
  reuse the main streams' fit in the same process); that variant is recomputed and a test guards the memo.
- Grip: the first version read a screen-away approach as a return. A check of the chosen segment's own start pose
  did not fix it (kept in the stress test for the record); the remedy is the page's model-free rest-pose grip check.
  A grip-normalised feature set was built and evaluated: invariant, but less accurate. Calibrator: chosen inside each
  fold among the antisymmetric ones (plain isotonic, chosen on the outer folds before, broke P(mirror image) = 1 - P;
  its mirror deviation is now reported). Attempt rule: plain (the pairing rule discarded true approaches).
  Sessions: {v('sess_recordings')} recordings, so {v('k_max')} accepted attempts can be reached. Aggregation of
  touches: chosen by a declared rule from calibration evidence, and called exploratory.
- The multi-touch headline is the cue-free sessions' k = 3; the harness k = 3 is labelled as independent recordings.
- Significance is Holm-adjusted within named families; ECE is read against its perfect-calibration null band with
  the calibration slope; new ablations (feature scaling, C, look-ahead, P(approach) threshold), a replay of the online
  front end on re-standardised raw records, stress points near the stillness cap, the screen-away twin, and an
  analysis of the natural-use rejections.

## Caveats and open questions

- The recordings come from a protocol with a GO cue and an enforced stillness before every approach; the synthetic
  sessions and the stress test approximate production but are spliced from those recordings, and the session-fitted
  aggregate scale is learned from such spliced sessions too. Real cue-free streams still need the rehearsal session.
- Exploratory choices on the outer folds: the detector feature set (robust; {v('n_configs_detector')} sets), the hand
  model, the attempt rule and the aggregation; the production result records every cue-free pipeline compared
  (n_configs_tried = {v('n_configs_final')}). The segmentation configuration and the calibrator are nested.
- Untested assumptions: iOS sample stalls during an approach are not simulated; the hold at the nose (the stillness
  after a segment is a feature) is the app's end-cue hold and is not shortened by any stress test; the look-ahead,
  scaling and C ablations are single paired comparisons (table `m4_ablations`); the grip check is evaluated on
  synthetic screen-away twins, not on real screen-away recordings.
- Mild in-fold optimism (not leakage of the held-out person): the recording-fitted aggregate scale and the
  calibrator are fitted on the same cross-fitted logits, and the session-fitted scale uses the outer calibrator on the
  same training persons.
- p_stop {v('p_stop')} is a product choice fixed before the sessions were evaluated; the other thresholds of the
  table `m4_stopping` are an exploratory reading of the same sessions.
- The natural-use set has only right hands and one trial per person: its accuracy is right-hand recall, with wide
  intervals.
- Session-level intervals are wide ({v('sess_n')} sessions of {v('n_folds_sessions')} persons). ECE has no percentile
  bootstrap CI here (it can exclude its own estimate); it is read against its perfect-calibration null band, with the
  calibration slope and the paired log-loss and Brier differences.
- Demo (`tables/m4/m4_demo.json`): held-out {demo.get('person', '--')}, {v('demo_rows')} streams through
  `predict_stream` by the pipeline of the fold that held that person out (trained on {v('demo_trained_persons')} persons).
"""
    return text


def _num(x: Any) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
