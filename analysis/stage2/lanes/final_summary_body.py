"""Lane final: the body of final_SUMMARY.md (sections 1-9, experiments, figures, tables, commands, conclusions).
Every number goes through ``Num`` (an entry of the numbers file); every verdict is computed from the saved files."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.pipeline import Lane

L = Lane('final')


def _src(name: str) -> str:
    return f'tables/final/{name}.json'


def _read(name: str) -> dict[str, Any]:
    from stage2.lanes.final_io import denull
    return denull(L.read_json(name))


def _has(name: str) -> bool:
    return (L.tab_dir / f'{name}.json').exists()


def _sig(p: float | None) -> bool:
    return p is not None and np.isfinite(p) and p < 0.05


def body(n: Any, w: Any, lab: dict[str, str], fin: str, dec: dict[str, Any], master: dict[str, Any],
         cal: dict[str, Any], ns: dict[str, Any]) -> None:
    src_m = _src('final_master')
    # ── 1. master comparison
    w('## 1. Master comparison (`tables/final/final_master.csv`, figures FINAL_01, FINAL_02)\n')
    rows = sorted(master.values(), key=lambda r: r['log_loss'])
    for r in rows:
        k = r['key']
        lbl = r['label']
        w(f'- {r["label"]} (`{r["result"]}`, lane {r["lane"]}): accuracy {n.pct(f"{k}_accuracy", r["accuracy"], src_m, "")} '
          f'{n.ci(f"{k}_accuracy", r["accuracy_ci"], src_m, f"{lbl}: accuracy")}, '
          f'log loss {n(f"{k}_log_loss", r["log_loss"], ".3f", src_m, f"{lbl}: log loss")}, '
          f'AUC {n(f"{k}_auc", r["auc"], ".3f", src_m, f"{lbl}: AUC")}, '
          f'ECE {n(f"{k}_ece", r["cal_ece"], ".3f", src_m, "ECE")} (null 95th percentile '
          f'{n(f"{k}_ece_null_q", r["cal_ece_null_q95"], ".3f", src_m, "ECE null")}), natural-use right recall '
          f'{n.pct(f"{k}_nat_recall", r["nat_recall"], src_m, "natural-use right-hand recall")}, k = 3 '
          f'{n.pct(f"{k}_kthree", r["k3"], src_m, "k = 3")}; configurations compared in its lane '
          f'{n(f"{k}_configs", int(r["n_configs_tried"]), "d", src_m, "configs")}; '
          f'{"nested tuning" if r["nested"] else "no nested tuning"}.')
    w('')
    if _has('final_reproduction'):
        rep = _read('final_reproduction')['rows']
        mx = max(r['max_abs_dp_primary'] for r in rep)
        ok = sum(bool(r['reproduced']) for r in rep)
        w(f'**Reproduction check.** Every lane finalist was rebuilt from its finalists file (module, build_x, factory, '
          f'params) and re-run through `run_experiment` (`save=False`, leak check off: the saved result passed it): '
          f'{n("repro_ok", ok, "d", _src("final_reproduction"), "finalists that reproduce their saved predictions")} of '
          f'{n("repro_n", len(rep), "d", _src("final_reproduction"), "finalists re-run")} reproduce their saved '
          f'predictions on every primary and natural-use trial (largest |dP| '
          f'{n("repro_max_dp", mx, ".1e", _src("final_reproduction"), "largest difference to the saved predictions")}). '
          f'Lane m3\'s deep models were refitted in this lane\'s own fit cache (`analysis/data/stage2_final/m3_fit_cache`; '
          f'lane m3\'s runtime pins one torch thread per fit and keys the cache on the torch version and thread count, so '
          f'refits are bit-identical), never in lane m3\'s.')
        mism = [(k, r) for k, r in master.items() if isinstance(r.get('source'), str)
                and np.isfinite(r.get('result_n_configs_tried', math.nan))
                and int(r['result_n_configs_tried']) != int(r['n_configs_tried'])]
        for k, r in mism:
            w(f'**Contradiction found:** for {lab.get(k, k)} the finalists file of lane {r["lane"]} records '
              f'{n(f"nconf_file_{k}", int(r["n_configs_tried"]), "d", "results/finalists_" + str(r["lane"]) + ".json", "configurations in the finalists file")} '
              f'configurations compared on the outer folds, its saved result `{r["result"]}` '
              f'{n(f"nconf_result_{k}", int(r["result_n_configs_tried"]), "d", "results/" + str(r["result"]) + ".json", "configurations in the saved result")}. '
              f'This lane uses the finalists file\'s count (the role in which it became a candidate)'
              + (', so its numbers count as exploratory.' if int(r['n_configs_tried']) > 1 else '.'))
        done = {r['key'] for r in rep}
        skipped = [k for k, r in master.items() if isinstance(r.get('source'), str) and k not in done]
        if skipped:
            w(f'Not re-run (the expensive ones; the task asks to re-run the cheap ones): '
              f'{", ".join(lab.get(k, k) for k in skipped)}. Their LOSO results are used as their lanes saved them; the '
              f'robustness runs of this lane rebuild them through the same finalists entries.')
        w('')
    # ── 2. nested selection
    if ns:
        src = _src('final_nested')
        w('## 2. Selection-aware estimate: the nested cross-lane selection (`final_nested_select`, FINAL_03)\n')
        w('Inside every outer fold every candidate is cross-validated on the training persons only (5 inner person '
          'folds, each wrapped as a `cv.Fold` with test-side columns only), the candidate with the lowest pooled inner '
          'log loss (pre-specified criterion) is refitted on all training persons and predicts the held-out person; '
          'inner predictions are cached per (candidate, outer training set) under `analysis/data/stage2_final/inner/`.\n')
        freq = {k: v for k, v in ns['frequency'].items() if v > 0}
        parts = [f'{lab.get(k, k)} {n(f"sel_{k}", v, "d", src, f"outer folds choosing {k}")}'
                 for k, v in sorted(freq.items(), key=lambda kv: -kv[1])]
        w(f'- Chosen per outer fold: {"; ".join(parts)} (of '
          f'{n("sel_folds", int(sum(ns["frequency"].values())), "d", src, "outer folds")}).')
        best_inner = min(ns['inner_logloss_mean'], key=ns['inner_logloss_mean'].get)
        w(f'- Lowest mean inner log loss: {lab.get(best_inner, best_inner)} '
          f'({n("inner_ll_best", ns["inner_logloss_mean"][best_inner], ".3f", src, "lowest mean inner log loss")}).')
        pn = ns['paired_naive_minus_nested']
        w(f'- Naive best (exploratory, chosen on the outer folds) minus nested, as the mean per-person difference '
          f'(the headline gives the pooled difference): accuracy '
          f'{n.pts("naive_vs_nested_acc", pn["mean_person_acc_diff"], src, "naive minus nested: mean per-person accuracy")} '
          f'{n.pts_ci("naive_vs_nested_acc", pn["mean_person_acc_diff_ci95"], src, "naive minus nested")} points, '
          f'sign-flip p {n.p("naive_vs_nested_p", pn["p_sign_flip_acc"], src, "naive minus nested: sign-flip p")}.\n')
    # ── 2b. input vs model class
    if _has('final_input_class'):
        src = _src('final_input_class')
        w('## 2b. Input or model class? (task 2b, `tables/final/final_input_class.csv`, FINAL_05)\n')
        w('Lane m2\'s feature pipeline and a lane-m1-style antisymmetric logistic were given the body-frame channels '
          '(roll rate, pitch rate, lateral and forward acceleration of `m3_body`) and evaluated exactly like the other '
          'candidates (per-trial features, nested tuning). Holm over these comparisons:\n')
        for r in _read('final_input_class')['rows']:
            c = r['comparison']
            verdict = ('A better' if _sig(r['holm_acc']) and r['d_acc'] > 0 else
                       'A worse' if _sig(r['holm_acc']) and r['d_acc'] < 0 else 'no detectable accuracy difference')
            w(f'- {lab.get(r["a"], r["a"])} minus {lab.get(r["b"], r["b"])} ({r["what"]}): accuracy '
              f'{n.pts(f"ic_{c}_acc", r["d_acc"], src, f"{c}: accuracy difference")} '
              f'{n.pts_ci(f"ic_{c}_acc", r["d_acc_ci"], src, f"{c}")} points (Holm p '
              f'{n.p(f"ic_{c}_holm_acc", r["holm_acc"], src, f"{c}: Holm p, accuracy")}), log loss '
              f'{n(f"ic_{c}_ll", r["d_ll"], "+.3f", src, f"{c}: log-loss difference")} '
              f'{n.ci(f"ic_{c}_ll", r["d_ll_ci"], src, f"{c}: log loss", "+.3f")} (Holm p '
              f'{n.p(f"ic_{c}_holm_ll", r["holm_ll"], src, f"{c}: Holm p, log loss")}): {verdict}.')
        w('')
    # ── 3. paired comparisons
    pr = pd.DataFrame(_read('final_paired')['rows'])
    best = _read('final_paired')['best']
    src = _src('final_paired')
    w(f'## 3. Paired comparisons (`tables/final/final_paired.csv`, FINAL_04)\n')
    w('Every candidate is compared in its **production form**: its own pipeline plus the calibration remedy (none, '
      'temperature, Platt or beta) chosen inside every outer fold by grouped cross-validation on the training persons\' '
      'inner cross-fitted predictions (`final_cal_<key>_auto`), because the product is the calibrated percentage. A '
      'pair whose log losses differ by less than 0.001 with no trial changing its answer is called identical, never '
      'significant.\n')
    vb = pr[pr['family'] == 'vs_best']
    sep = vb[vb['sig_acc'] | vb['sig_ll']]
    w(f'- Against the lowest-log-loss candidate ({lab.get(best, best)}; Holm over '
      f'{n("paired_best_m", len(vb), "d", src, "comparisons against the best")} comparisons): '
      f'{n("paired_best_sep", len(sep), "d", src, "candidates separated from the best")} candidates are separated in '
      f'accuracy or log loss ({", ".join(lab.get(k, k) for k in sep["key"]) or "none"}); the others cannot be told apart '
      f'from it at this number of persons.')
    vr = pr[pr['family'] == 'vs_reference']
    better = vr[vr['sig_acc'] & (vr['d_acc'] > 0)]
    w(f'- Against the reference floor: {n("paired_ref_better", len(better), "d", src, "candidates more accurate than the reference after Holm")} '
      f'of {n("paired_ref_m", len(vr), "d", src, "comparisons against the reference")} candidates are more accurate after '
      f'Holm.')
    fr = vb[vb['key'] == fin]
    if len(fr):
        r = fr.iloc[0]
        w(f'- Final model against the best: accuracy {n.pts("final_vs_best_acc", r["d_acc"], src, "final minus best: accuracy")} '
          f'{n.pts_ci("final_vs_best_acc", r["d_acc_ci"], src, "final minus best")} points (Holm p '
          f'{n.p("final_vs_best_holm_acc", r["holm_acc"], src, "final minus best: Holm p accuracy")}), log loss '
          f'{n("final_vs_best_ll", r["d_ll"], "+.3f", src, "final minus best: log loss")} '
          f'{n.ci("final_vs_best_ll", r["d_ll_ci"], src, "final minus best: log loss", "+.3f")} (Holm p '
          f'{n.p("final_vs_best_holm_ll", r["holm_ll"], src, "final minus best: Holm p log loss")}).')
    rr = vr[vr['key'] == fin]
    if len(rr):
        r = rr.iloc[0]
        w(f'- Final model against the reference: accuracy {n.pts("final_vs_ref_acc", r["d_acc"], src, "final minus reference: accuracy")} '
          f'{n.pts_ci("final_vs_ref_acc", r["d_acc_ci"], src, "final minus reference")} points (Holm p '
          f'{n.p("final_vs_ref_holm_acc", r["holm_acc"], src, "final minus reference: Holm p accuracy")}), log loss '
          f'{n("final_vs_ref_ll", r["d_ll"], "+.3f", src, "final minus reference: log loss")} (Holm p '
          f'{n.p("final_vs_ref_holm_ll", r["holm_ll"], src, "final minus reference: Holm p log loss")}).')
    w('')
    # ── 4. decision
    src = _src('final_decision')
    rows = {r['key']: r for r in dec['rows']}
    w('## 4. Decision (`tables/final/final_decision.csv`)\n')
    w(f'Pre-specified rule (`final_finalmodel.recommend`): {dec["rule"]}.\n')
    w(f'- Leaders (not detectably worse than {lab.get(dec["best_by_log_loss"], dec["best_by_log_loss"])}): '
      f'{", ".join(lab.get(k, k) for k in dec["leaders"])}.')
    w(f'- Robust leaders: {", ".join(lab.get(k, k) for k in dec["robust_leaders"]) or "none"}.')
    for k in dec['leaders']:
        r = rows[k]
        why = []
        if r['grip_fails']:
            why.append('accuracy drops on grips absent from training (Holm)')
        if r['platform_fails']:
            why.append('drops when trained on the other platform (Holm)')
        if r['nat_shortfall']:
            why.append('natural-use shortfall against the person-matched expectation (Holm)')
        if r['flip_fails_raw'] and not r['flip_fixed_by_remedy']:
            why.append('screen-away failure not fixed by the rest-pose check')
        if why:
            w(f'  - {lab.get(k, k)} is not robust: {"; ".join(why)}.')
    w(f'- Recommended by the rule: **{lab.get(dec["recommended"], dec["recommended"])}** (portability rank '
      f'{n("final_port_rank", int(rows[dec["recommended"]]["port_rank"]), "d", src, "portability rank of the recommendation (1 easiest)")}: '
      f'{rows[dec["recommended"]]["port"]}).')
    w(f'- `FINAL_MODEL = \'{fin}\'` {"agrees with" if dec["agrees_with_rule"] else "DIFFERS from"} the rule.')
    # the trade-off against the simplest robust candidate (generated: the most portable robust candidate, then the
    # lowest log loss), which the rule leaves out because the paired comparisons separate it from the best
    fr_ = rows[fin]
    simple = sorted([r for r in rows.values() if r['robust'] and r['key'] != fin and not r['constant_within_fold']],
                    key=lambda r: (r['port_rank'], r['prod_log_loss']))
    if simple and simple[0]['port_rank'] < fr_['port_rank']:
        s_ = simple[0]
        k = s_['key']
        rob_k = (next((x for x in _read('final_robustness')['rows'] if x['key'] == k), {}) or {}).get('grip') or {}
        rob_f = (next((x for x in _read('final_robustness')['rows'] if x['key'] == fin), {}) or {}).get('grip') or {}
        w(f'- **Trade-off against the simplest robust candidate**, {lab.get(k, k)} (portability rank '
          f'{n("simple_port_rank", int(s_["port_rank"]), "d", src, "portability rank of the simplest robust candidate")}): '
          f'in its production form it is {"detectably" if s_["worse_than_best"] else "not detectably"} worse than the '
          f'best (accuracy {n.pts("simple_vs_best_acc", s_["vs_best_d_acc"], src, "simplest robust candidate minus the best: accuracy")} '
          f'points, Holm p {n.p("simple_vs_best_holm_acc", s_["vs_best_holm_acc"], src, "simplest robust minus best: Holm p accuracy")}; '
          f'log loss {n("simple_vs_best_ll", s_["vs_best_d_ll"], "+.3f", src, "simplest robust minus best: log loss")}, '
          f'Holm p {n.p("simple_vs_best_holm_ll", s_["vs_best_holm_ll"], src, "simplest robust minus best: Holm p log loss")}). '
          + (f'On a grip absent from training the difference vanishes: on the rotated-grip trials with upright-only '
             f'training it reaches {n.pct("simple_grip_acc", rob_k["acc_upright_only"], src, "simplest robust candidate: rotated grips, upright-only training")} '
             f'against {n.pct("final_grip_acc", rob_f["acc_upright_only"], src, "final model: rotated grips, upright-only training")} '
             f'for the final model, whose accuracy there changes by {n.pts("final_grip_d", rob_f["d_acc"], src, "final model: change on the rotated grips")} '
             f'points {n.pts_ci("final_grip_d", rob_f["d_acc_ci"], src, "final model: change on the rotated grips")} (unadjusted '
             f'p {n.p("final_grip_p", rob_f["p_acc"], src, "final model: unadjusted sign-flip p of the grip change")}; '
             f'log loss {n("final_grip_dll", rob_f["d_ll"], "+.3f", src, "final model: log-loss change on the rotated grips")}, '
             f'p {n.p("final_grip_p_ll", rob_f["p_ll"], src, "final model: unadjusted p of the log-loss change")}) while '
             f'the simple model\'s changes by {n.pts("simple_grip_d", rob_k["d_acc"], src, "simplest robust candidate: change on the rotated grips")}. '
             if rob_k and rob_f else '')
          + ('So the final model is the better choice when the PI uses the recorded grips (which the training data '
             'cover), and the simple one is the fallback if a grip unlike any recorded one matters more than the '
             'accuracy on the recorded ones.'
             if rob_k and rob_f and rob_k['acc_upright_only'] >= rob_f['acc_upright_only'] else
             'The final model is also at least as accurate on a grip absent from training.'))
    p1d = C.TAB_DIR / 'p1' / 'p1_decisions.csv'
    if p1d.exists():
        t = pd.read_csv(p1d)
        dcol = [c for c in t.columns if c.lower().startswith('decision')]
        keep = int(t[dcol[0]].astype(str).str.contains('keep').sum()) if dcol else 0
        w(f'- Data treatment (lane p1): {n("p1_keep", keep, "d", "tables/p1/p1_decisions.csv", "p1 choices that keep the default")} '
          f'of {n("p1_choices", len(t), "d", "tables/p1/p1_decisions.csv", "p1 data-treatment choices")} choices keep the '
          f'current default under p1\'s pre-specified rule (better for at least two panel models and worse for none), so '
          f'there is no preprocessing change to adopt; the final model uses the cache\'s treatment.')
    w('')
    # ── 4b. robustness
    rob = {r['key']: r for r in _read('final_robustness')['rows']}
    src = _src('final_robustness')
    w('## 4b. Robustness to the unknown test conditions (`tables/final/final_robustness.csv`, FINAL_06, FINAL_07)\n')
    for k, r in rob.items():
        bits = []
        g = r.get('grip')
        if g:
            bits.append(f'rotated grips with upright-only training {n.pct(f"rob_{k}_grip", g["acc_upright_only"], src, "accuracy on the rotated grips, upright-only training")} '
                        f'({n.pts(f"rob_{k}_grip_d", g["d_acc"], src, "change on the rotated grips")} points, Holm p '
                        f'{n.p(f"rob_{k}_grip_holm", g.get("holm_acc"), src, "Holm p of the grip change")})')
        f = r.get('flip')
        if f:
            bits.append(f'screen away {n.pct(f"rob_{k}_flip", f["acc_flipped"], src, "accuracy on screen-away twins")}'
                        + (' (inputs invariant)' if f['invariant_inputs'] else
                           f', with the rest-pose check {n.pct(f"rob_{k}_flip_fix", f["acc_flipped_remedy"], src, "screen-away accuracy with the check")}'))
        pl = r.get('platform')
        if pl:
            bits.append(f'other platform only {n.pts(f"rob_{k}_plat_d", pl["d_acc"], src, "change when trained on the other platform")} '
                        f'points (Holm p {n.p(f"rob_{k}_plat_holm", pl.get("holm_acc"), src, "Holm p platform")})'
                        + (' [lane p1 run]' if pl.get('from_lane_p1') else ''))
        nz = r.get('noise')
        if nz:
            bits.append(f'without the suspect blocks {n.pts(f"rob_{k}_noise_d", nz["d_acc"], src, "change without the suspect blocks")} '
                        f'points (Holm p {n.p(f"rob_{k}_noise_holm", nz.get("holm_acc"), src, "Holm p label noise")})')
        if bits:
            w(f'- {r["label"]}: ' + '; '.join(bits) + '.')
    fr = rob.get(fin, {}).get('flip')
    if fr and not fr['invariant_inputs']:
        w(f'\n**Screen-away grip, final model** (detection, why, remedy, evidence). Detected: on the screen-away twins '
          f'(`standardize.screen_flip_channels` on the test trials, the model\'s own feature code) its accuracy falls to '
          f'{n.pct("final_flip_raw", fr["acc_flipped"], src, "final model: accuracy on screen-away twins")} (paired '
          f'{n.pts("final_flip_raw_d", fr["d_acc"], src, "final model: change screen away")} points '
          f'{n.pts_ci("final_flip_raw_d", fr["d_acc_ci"], src, "final model: change screen away")}). Why it matters: '
          f'the body-frame roll and lateral channels change sign when the screen faces away, so the model answers the '
          f'other hand with confidence. Remedy: lane m4\'s model-free rest-pose check (the screen-normal up component '
          f'at rest below its threshold means "screen away"), then un-flip the window (or ask the user to turn the '
          f'phone). Evidence: the check fires on {n.pct("final_check_fires_flip", fr["check_fires_flipped"], src, "share of screen-away twins the check detects")} '
          f'of the twins and {n.pct("final_check_fires_rec", fr["check_fires_recorded"], src, "share of recorded trials the check flags")} '
          f'of the recorded trials; screen-away accuracy with the check '
          f'{n.pct("final_flip_fixed", fr["acc_flipped_remedy"], src, "final model: screen-away accuracy with the check")} '
          f'(gain {n.pts("final_flip_gain", fr["remedy_gain"], src, "gain of the check on the twins")} points '
          f'{n.pts_ci("final_flip_gain", fr["remedy_gain_ci"], src, "gain of the check")}); its cost on the recorded '
          f'trials {n.pts("final_flip_cost", fr["remedy_cost"], src, "cost of the check on the recorded trials")} points '
          f'{n.pts_ci("final_flip_cost", fr["remedy_cost_ci"], src, "cost of the check")}; abstaining instead of '
          f'un-flipping gives {n.pct("final_flip_abstain", fr["acc_flipped_abstain"], src, "screen-away accuracy when the check abstains")} '
          f'(half credit).')
    rem = _read('final_remedies') if _has('final_remedies') else {}
    srem = _src('final_remedies')
    fa, fx = rem.get('flipaug_recorded'), rem.get('flipaug_twins')
    if fa and fx:
        w(f'\n**Second remedy, screen-flip augmentation** of the training rows (every training trial also enters as its '
          f'screen-away twin with the same label; the check is not used): on the screen-away twins '
          f'{n.pct("flipaug_twins_acc", fx["acc_a"], srem, "augmented model: accuracy on screen-away twins")} against '
          f'{n.pct("flipaug_twins_base", fx["acc_b"], srem, "model as trained: accuracy on screen-away twins")} '
          f'({n.pts("flipaug_twins_d", fx["d_acc"], srem, "augmentation: change on the twins")} points '
          f'{n.pts_ci("flipaug_twins_d", fx["d_acc_ci"], srem, "augmentation: change on the twins")}, Holm p '
          f'{n.p("flipaug_twins_holm", fx.get("holm_acc"), srem, "augmentation on the twins: Holm p")}); on the recorded '
          f'trials {n.pct("flipaug_rec_acc", fa["acc_a"], srem, "augmented model: accuracy on the recorded trials")} against '
          f'{n.pct("flipaug_rec_base", fa["acc_b"], srem, "model as trained: accuracy on the recorded trials")} '
          f'({n.pts("flipaug_rec_d", fa["d_acc"], srem, "augmentation: change on the recorded trials")} points '
          f'{n.pts_ci("flipaug_rec_d", fa["d_acc_ci"], srem, "augmentation: change on the recorded trials")}, Holm p '
          f'{n.p("flipaug_rec_holm", fa.get("holm_acc"), srem, "augmentation on the recorded trials: Holm p")}; log loss '
          f'{n("flipaug_rec_dll", fa["d_ll"], "+.3f", srem, "augmentation: log-loss change on the recorded trials")}).')
        if fr and 'll_recorded_remedy' in fr:
            same_twins = abs(fr['acc_flipped_remedy'] - fx['acc_a']) <= 0.02
            same_rec = abs(fr['ll_recorded_remedy'] - fa['ll_a']) <= 0.01
            w(f'\n**Choice of the screen-away remedy** (recorded holds: accuracy, log loss / screen-away twins: accuracy): '
              f'as trained {n.pct("scr_rec_acc", fr["acc_recorded"], src, "as trained: recorded accuracy")}, '
              f'{n("scr_rec_ll", fr["ll_recorded"], ".3f", src, "as trained: recorded log loss")} / '
              f'{n.pct("scr_twin_acc", fr["acc_flipped"], src, "as trained: twins accuracy")}; rest-pose check then '
              f'un-flip {n.pct("scr_chk_rec_acc", fr["acc_recorded_remedy"], src, "check + un-flip: recorded accuracy")}, '
              f'{n("scr_chk_rec_ll", fr["ll_recorded_remedy"], ".3f", src, "check + un-flip: recorded log loss")} / '
              f'{n.pct("scr_chk_twin_acc", fr["acc_flipped_remedy"], src, "check + un-flip: twins accuracy")}; check then '
              f'prompt (abstain, half credit) {n.pct("scr_abs_rec_acc", fr["acc_recorded_abstain"], src, "check + prompt: recorded accuracy")}, '
              f'{n("scr_abs_rec_ll", fr["ll_recorded_abstain"], ".3f", src, "check + prompt: recorded log loss")} / '
              f'{n.pct("scr_abs_twin_acc", fr["acc_flipped_abstain"], src, "check + prompt: twins accuracy")}; '
              f'augmentation {n.pct("scr_aug_rec_acc", fa["acc_a"], srem, "augmentation: recorded accuracy")}, '
              f'{n("scr_aug_rec_ll", fa["ll_a"], ".3f", srem, "augmentation: recorded log loss")} / '
              f'{n.pct("scr_aug_twin_acc", fx["acc_a"], srem, "augmentation: twins accuracy")}. '
              + ('The two remedies are indistinguishable here (twins within 2 points, recorded log loss within 0.01), so '
                 'production keeps the model as trained and the model-free check, exported with it: the check leaves the '
                 'model unchanged, its cost on the recorded holds comes only from the holds it flags (the screen tilted '
                 'down at rest), and on a flag the page can ask the user to turn the phone instead of un-flipping.'
                 if same_twins and same_rec else
                 ('The augmented model is better on these data; production still exports the model as trained with the '
                  'check, and adopting the augmentation is an open question for Stage 3.'
                  if fa['ll_a'] < fr['ll_recorded_remedy'] and fx['acc_a'] >= fr['acc_flipped_remedy'] else
                  'The check is the better remedy on these data, so production keeps the model as trained with the '
                  'check, exported with it.')))
    pc_, ca_ = rem.get('platform_vs_control'), rem.get('control_vs_all')
    if pc_ and ca_:
        w(f'\n**Platform transfer of the final model against its size-matched control:** trained on the other '
          f'platform\'s persons only {n.pct("plat_only_acc", pc_["acc_a"], srem, "final model: other platform only")}, '
          f'trained on as many randomly drawn persons of either platform '
          f'{n.pct("plat_ctl_acc", pc_["acc_b"], srem, "final model: size-matched control")}: training on the other '
          f'platform changes the accuracy by '
          f'{n.pts("plat_vs_ctl_d", pc_["d_acc"], srem, "other platform minus control: accuracy")} points '
          f'{n.pts_ci("plat_vs_ctl_d", pc_["d_acc_ci"], srem, "other platform minus control")} (Holm p '
          f'{n.p("plat_vs_ctl_holm", pc_.get("holm_acc"), srem, "other platform minus control: Holm p")}), the smaller '
          f'training set by {n.pts("ctl_vs_all_d", ca_["d_acc"], srem, "control minus all persons: accuracy")} points '
          f'{n.pts_ci("ctl_vs_all_d", ca_["d_acc_ci"], srem, "control minus all persons")} (Holm p '
          f'{n.p("ctl_vs_all_holm", ca_.get("holm_acc"), srem, "control minus all persons: Holm p")}).')
    nat = _read('final_natural')
    src = _src('final_natural')
    w('\n**Transfer to natural use** (person-matched test; each free-choice trial\'s expected correctness is the same '
      'person\'s right-hand accuracy on the instructed trials; Poisson-binomial lower tail, Holm over the candidates):')
    for r in nat['rows']:
        k = r['key']
        w(f'- {r["label"]}: {n(f"nat_{k}_obs", r["observed"], "g", src, "correct free-choice trials")} of '
          f'{n(f"nat_{k}_n", r["n"], "d", src, "free-choice trials")} against '
          f'{n(f"nat_{k}_exp", r["expected"], ".1f", src, "expected correct free-choice trials")} expected (Holm p '
          f'{n.p(f"nat_{k}_holm", r["holm_lower"], src, "Holm p of the shortfall")}){": a shortfall" if r["shortfall"] else ""}.')
    for ck, comp in (nat.get('components') or {}).items():
        if not comp.get('rows'):
            continue
        w(f'- Components of {lab.get(ck, ck)} ({comp["method"]}; logit contribution towards the reported hand on the '
          f'free-choice trials minus the same persons\' instructed trials of that hand; mean over persons, paired person '
          f'bootstrap, Holm over the components):')
        for r in comp['rows']:
            c = r['component']
            what = 'whole logit' if c == 'total' else c
            hp = (f', Holm p {n.p(f"natc_{ck}_{c}_holm", r["holm"], src, f"{ck} {c}: Holm p")}' if 'holm' in r else
                  f', p {n.p(f"natc_{ck}_{c}_p", r["p"], src, f"{ck} {c}: sign-flip p")}')
            w(f'  - {what}: free choice '
              f'{n(f"natc_{ck}_{c}_nat", r["natural"], "+.2f", src, f"{ck} {c}: mean contribution, free choice")} '
              f'against {n(f"natc_{ck}_{c}_prim", r["primary_same_persons"], "+.2f", src, f"{ck} {c}: mean contribution, instructed")} '
              f'instructed: difference {n(f"natc_{ck}_{c}_diff", r["diff"], "+.2f", src, f"{ck} {c}: contribution difference")} '
              f'{n.ci(f"natc_{ck}_{c}_diff", r["diff_ci"], src, f"{ck} {c}: contribution difference", "+.2f")}{hp}; '
              f'points to the reported hand in '
              f'{n(f"natc_{ck}_{c}_right", r["natural_pointing_right_way"], "d", src, f"{ck} {c}: free-choice trials where it points to the reported hand")} '
              f'of {n(f"natc_{ck}_{c}_n", r["natural_n"], "d", src, "free-choice trials")} free-choice trials.')
    w('')
    # ── 5. calibration
    if cal:
        src = _src('final_calibration')
        ch = cal['chosen']
        none = cal['methods']['none']['overall']
        w('## 5. Calibration of the final model (`tables/final/final_calibration.csv`, FINAL_08, FINAL_09)\n')
        verdict = 'miscalibrated' if none['miscalibrated'] else 'consistent with calibration'
        w(f'- **Detection.** As fitted (no extra remedy): ECE {n("cal_none_ece", none["ece"], ".3f", src, "uncalibrated: ECE")} '
          f'against a perfect-calibration band up to {n("cal_none_null_hi", none["ece_null_hi"], ".3f", src, "uncalibrated: ECE null band high")} '
          f'(p {n.p("cal_none_ece_p", none["ece_p_value"], src, "uncalibrated: ECE null p")}); slope '
          f'{n("cal_none_slope", none["slope"], ".2f", src, "uncalibrated: slope")} '
          f'{n.ci("cal_none_slope", none["slope_ci"], src, "uncalibrated: slope", ".2f")}, intercept '
          f'{n("cal_none_int", none["intercept"], "+.2f", src, "uncalibrated: intercept")} '
          f'{n.ci("cal_none_int", none["intercept_ci"], src, "uncalibrated: intercept", "+.2f")}: {verdict}.')
        w('- **Why it matters.** The product is the percentage shown to the user; an over- or under-confident '
          'percentage misleads even when the decision is right.')
        parts = []
        for m, e in cal['methods'].items():
            if m == 'none':
                continue
            d = e['d_log_loss']
            parts.append(f'{m} {n(f"cal_{m}_dll", d["diff"], "+.4f", src, f"{m}: log loss minus none")} '
                         f'{n.ci(f"cal_{m}_dll", d["ci"], src, f"{m}: log-loss difference", "+.4f")} (Holm p '
                         f'{n.p(f"cal_{m}_holm", d.get("holm"), src, f"{m}: Holm p")})')
        w(f'- **Remedies** (fitted in every outer fold on the training persons\' inner cross-fitted predictions, applied '
          f'to the held-out person; paired log loss minus none): ' + '; '.join(parts) + '.')
        w(f'- **Decision** ({cal["rule"]}): ship **{ch}**.')
        if ch != 'none':
            e = cal['methods'][ch]
            w(f'- **Evidence after the remedy:** ECE {n("cal_ch_ece", e["overall"]["ece"], ".3f", src, "shipped: ECE")}, '
              f'ECE change {n("cal_ch_dece", e["d_ece"]["diff"], "+.3f", src, "shipped: ECE minus none")} '
              f'{n.ci("cal_ch_dece", e["d_ece"]["ci_basic"], src, "shipped: ECE change (basic bootstrap)", "+.3f")} '
              f'(basic bootstrap on the same person resamples), slope {n("cal_ch_slope", e["overall"]["slope"], ".2f", src, "shipped: slope")}.')
        pdiff = cal['methods'][ch].get('platform_difference')
        if pdiff:
            w(f'- Platform difference ({pdiff["a"]} minus {pdiff["b"]}, persons resampled within each platform; overlapping '
              f'per-platform CIs are not a test): calibration slope '
              f'{n("cal_platdiff_slope", pdiff["slope"]["diff"], "+.2f", src, "platform difference: slope")} '
              f'{n.ci("cal_platdiff_slope", pdiff["slope"]["ci"], src, "platform difference: slope", "+.2f")} (p '
              f'{n.p("cal_platdiff_slope_p", pdiff["slope"]["p"], src, "platform difference: slope p")}), calibration-in-the-'
              f'large {n("cal_platdiff_citl", pdiff["citl"]["diff"], "+.2f", src, "platform difference: CITL")} '
              f'{n.ci("cal_platdiff_citl", pdiff["citl"]["ci"], src, "platform difference: CITL", "+.2f")} (p '
              f'{n.p("cal_platdiff_citl_p", pdiff["citl"]["p"], src, "platform difference: CITL p")}), log loss '
              f'{n("cal_platdiff_ll", pdiff["log_loss"]["diff"], "+.3f", src, "platform difference: log loss")} '
              f'{n.ci("cal_platdiff_ll", pdiff["log_loss"]["ci"], src, "platform difference: log loss", "+.3f")} (p '
              f'{n.p("cal_platdiff_ll_p", pdiff["log_loss"]["p"], src, "platform difference: log loss p")}; Holm over '
              f'these three: {n.p("cal_platdiff_holm_min", min(1.0, 3 * min(pdiff[x]["p"] for x in ("slope", "citl", "log_loss"))), src, "platform difference: smallest Holm p")} '
              f'at the smallest).')
        ncal = cal['methods'][ch].get('natural_calibration')
        if ncal:
            w(f'- Natural use, calibration-in-the-large (every reported hand is the right one, so only the mean P of the '
              f'reported hand can be checked): mean P(right) on the free-choice trials '
              f'{n.pct("cal_nat_meanp", ncal["mean_p_reported"], src, "natural use: mean P(reported hand)")} against '
              f'{n.pct("cal_nat_primary", ncal["mean_p_same_persons_primary"], src, "same persons: mean P on instructed right-hand trials")} '
              f'on the same persons\' instructed right-hand trials (per-person difference '
              f'{n.pts("cal_nat_diff", ncal["diff"], src, "natural minus instructed: mean P")} points '
              f'{n.pts_ci("cal_nat_diff", ncal["diff_ci"], src, "natural minus instructed: mean P")}, sign-flip p '
              f'{n.p("cal_nat_p", ncal["p"], src, "natural minus instructed: p")}): the model is '
              + ('less' if ncal['diff'] < 0 else 'not less') + ' sure of the right hand on uninstructed movements'
              + (' (a detectable shift).' if ncal['p'] < 0.05 else ' (no detectable shift).'))
        for pl in ('android', 'ios'):
            e = cal['methods'][ch]['platform'][pl]
            w(f'- {pl}: ECE {n(f"cal_{pl}_ece", e["ece"], ".3f", src, f"{pl}: ECE")} (band up to '
              f'{n(f"cal_{pl}_null_hi", e["ece_null_hi"], ".3f", src, f"{pl}: ECE null high")}), slope '
              f'{n(f"cal_{pl}_slope", e["slope"], ".2f", src, f"{pl}: slope")} '
              f'{n.ci(f"cal_{pl}_slope", e["slope_ci"], src, f"{pl}: slope", ".2f")} '
              f'({"miscalibrated" if e["miscalibrated"] else "consistent with calibration"}; '
              f'{n(f"cal_{pl}_persons", e["n_persons"], "d", src, f"{pl}: persons")} persons).')
        w('')
    if _has('final_aggregation'):
        ag = _read('final_aggregation')
        src = _src('final_aggregation')
        names = {'mean_p': 'mean P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits',
                 'sum_tau': 'sum of logits / tau_k'}
        w('**Several attempts** (FINAL_10, `tables/final/final_aggregation.csv`; k-subsets of one person and hand, '
          f'{n("agg_persons", len(ag["persons"]), "d", src, "persons with enough trials of each hand")} persons, the same '
          'for every k; tau_k fitted in every outer fold on the training persons\' inner cross-fitted logits):')
        for k in ('3', '5'):
            per = ag['rules'][k]
            w(f'- k = {k}: ' + '; '.join(
                f'{names[r]} accuracy {n.pct(f"agg_k{k}_{r}_acc", v["accuracy"], src, f"k={k} {r}: accuracy")}, '
                f'confidence minus accuracy {n(f"agg_k{k}_{r}_over", v["over_confidence"], "+.3f", src, f"k={k} {r}: over-confidence")} '
                f'{n.ci(f"agg_k{k}_{r}_over", v["over_ci"], src, f"k={k} {r}: over-confidence", "+.3f")}'
                for r, v in per.items()) + '.')
        t3 = ag['tau_median'].get('3')
        pk = ag['paired_k3']['sum_tau_minus_mean_p']
        calibrated_rules = [names[r] for r, v in ag['rules']['3'].items() if v['calibrated']]
        w(f'- Median tau_3 {n("agg_tau3", t3, ".2f", src, "median tau_3 over the outer folds")} (3 would be the mean '
          f'logit, 1 the sum). Sum / tau_k minus mean P at k = 3: log loss '
          f'{n("agg_dll_k3", pk["diff"], "+.3f", src, "k = 3: sum/tau minus mean P, log loss")} '
          f'{n.ci("agg_dll_k3", pk["ci"], src, "k = 3: sum/tau minus mean P", "+.3f")} (Holm p '
          f'{n.p("agg_dll_k3_holm", pk["holm"], src, "k = 3: Holm p")}). Calibrated at k = 3 (over-confidence CI '
          f'containing 0): {", ".join(calibrated_rules) or "none"}. **Decision:** the page shows sigmoid(sum of logits '
          f'/ tau_k) with tau_k exported (`final_model.json`), never the mean P(right).\n')
    if _has('final_prior'):
        pr_ = _read('final_prior')['rows']
        src = _src('final_prior')
        a, b = pr_[0], pr_[-1]
        w(f'**The prior.** The model assumes equal prior odds (balanced training data). A deployment prior pi of the '
          f'right hand adds log(pi / (1 - pi)) to every logit (`final_infer.apply_prior`): at pi = '
          f'{n("prior_hi", b["prior"], ".1f", src, "largest prior shown")} the left-hand recall on the primary trials '
          f'falls from {n.pct("prior_left_lo", a["left_recall"], src, "left recall at prior 0.5")} to '
          f'{n.pct("prior_left_hi", b["left_recall"], src, "left recall at the largest prior")} while the right-hand recall '
          f'rises to {n.pct("prior_right_hi", b["right_recall"], src, "right recall at the largest prior")} (FINAL_14). '
          f'The PI chooses the hand freely, so the default stays at 0.5: a right-leaning prior would buy natural-use '
          f'recall with left-hand errors.\n')
    # ── 6. breakdowns
    if _has('final_breakdown'):
        br = _read('final_breakdown')
        src = _src('final_breakdown')
        w('## 6. Breakdowns of the final model (`tables/final/final_breakdown.csv`, FINAL_11)\n')
        for r in br['platform']:
            plat = r['platform']
            ci = n.ci(f'brk_{r["platform"]}', r['ci'], src, f'{r["platform"]}: accuracy') if 'ci' in r else ''
            w(f'- {r["platform"]}: {n.pct(f"brk_{plat}_acc", r["accuracy"], src, "accuracy")} '
              f'{ci} ({n(f"brk_{plat}_n", r["n"], "d", src, "trials")} trials).')
        pp = pd.DataFrame(br['persons']).sort_values('accuracy')
        low = pp.iloc[0]
        below = int((pp['accuracy'] < 0.8).sum())
        w(f'- Persons: lowest {low["person_id"]} {n.pct("brk_low_person", low["accuracy"], src, "lowest person accuracy")}; '
          f'{n("brk_below80", below, "d", src, "persons below 80 %")} persons below 80 %.')
        for _, r in pp[pp['handedness'].astype(str).str.lower() == 'left'].iterrows():
            pid = r['person_id']
            w(f'- Left-handed {r["person_id"]}: {n.pct(f"brk_left_{pid.lower()}", r["accuracy"], src, "left-handed person accuracy")} '
              f'(no CI: fewer than 5 left-handed persons).')
        gs = [r for r in br['grip_speed'] if r['persons'] >= 5]
        if gs:
            wst = min(gs, key=lambda r: r['accuracy'])
            cell = str(wst['orientation']).replace('_toward', '').replace('_', ' ')
            w(f'- Weakest grip x speed cell with at least {br["min_persons_ci"]} persons: {cell}, {wst["speed"]} '
              f'{n.pct("brk_worst_cell", wst["accuracy"], src, "weakest grip x speed cell")} '
              f'{n.ci("brk_worst_cell", wst["ci"], src, "weakest grip x speed cell")} '
              f'({n("brk_worst_cell_n", wst["n"], "d", src, "trials in the weakest cell")} trials).')
        vs = [r for r in br['variant'] if r['persons'] >= br['min_persons_ci']]
        if vs:
            wv = min(vs, key=lambda r: r['accuracy'])
            w(f'- Weakest movement variant with at least {br["min_persons_ci"]} persons: {wv["variant"]} '
              f'{n.pct("brk_worst_variant", wv["accuracy"], src, "weakest movement variant")} '
              f'{n.ci("brk_worst_variant", wv["ci"], src, "weakest movement variant")} '
              f'({n("brk_worst_variant_n", wv["n"], "d", src, "trials of the weakest variant")} trials, '
              f'{n("brk_worst_variant_p", wv["persons"], "d", src, "persons of the weakest variant")} persons).')
        ce = br['confident_errors']
        top = sorted(ce['by_person'].items(), key=lambda kv: -kv[1])[:3]
        w(f'- Confident errors (P(true hand) below 0.2): {n("brk_conf_err", ce["n"], "d", src, "confident errors")} of '
          f'{n("brk_err", ce["n_errors"], "d", src, "errors")} errors; most of them in '
          + ', '.join(f'{pid} ({n(f"brk_conf_{pid.lower()}", c, "d", src, f"confident errors of {pid}")})' for pid, c in top)
          + '; by grip: ' + ', '.join(f'{str(k).replace("_toward", "").replace("_", " ")} '
                                      f'{n("brk_conf_grip_" + str(k).lower(), int(v), "d", src, "confident errors by grip")}'
                                      for k, v in sorted(ce['by_orientation'].items()))
          + '; by speed: ' + ', '.join(f'{k} {n("brk_conf_speed_" + str(k).lower(), int(v), "d", src, "confident errors by speed")}'
                                       for k, v in sorted(ce['by_speed'].items())) + '.\n')
    # ── 7. learning curve
    if _has('final_learning'):
        lc = {k: v for k, v in _read('final_learning').items() if isinstance(v, dict) and 'summary' in v}
        src = _src('final_learning')
        w('## 7. Learning curve (`tables/final/final_learning.csv`, FINAL_12)\n')
        for key, v in lc.items():
            s = pd.DataFrame(v['summary'])
            first, last = s.iloc[0], s.iloc[-1]
            w(f'- {lab.get(key, key)}: {n(f"lc_{key}_n0", int(first["n"]), "d", src, "smallest training size")} persons '
              f'{n.pct(f"lc_{key}_acc0", first["acc_mean"], src, "mean accuracy at the smallest size")}, all '
              f'{n(f"lc_{key}_nall", int(last["n"]), "d", src, "all training persons")} persons '
              f'{n.pct(f"lc_{key}_accall", last["acc_mean"], src, "accuracy with all training persons")}.'
              + (f' Inverse power law: {n.pct(f"lc_{key}_at40", v["fit"]["at_40"][0], src, "extrapolated accuracy at 40 persons")} '
                 f'at 40 persons {n.ci(f"lc_{key}_at40", v["fit"]["at_40"][1:], src, "extrapolation at 40 persons")}, '
                 f'asymptote {n.pct(f"lc_{key}_asym", v["fit"]["a"], src, "fitted asymptote")} (an extrapolation of '
                 f'{n(f"lc_{key}_reps", int(s["reps"].iloc[0]), "d", src, "repetitions per size")} repetitions per size).'
                 if isinstance(v.get('fit'), dict) else '')
              + (f' Extrapolated gain from {int(last["n"])} to 40 training persons: '
                 f'{n.pts(f"lc_{key}_gain40", v["fit"]["gain_to_40"][0], src, "extrapolated gain to 40 training persons")} '
                 f'points {n.pts_ci(f"lc_{key}_gain40", v["fit"]["gain_to_40"][1:], src, "extrapolated gain to 40 persons")}.'
                 if isinstance(v.get('fit'), dict) and 'gain_to_40' in v['fit'] else ''))
        fk = lc.get(fin, {}).get('fit')
        if isinstance(fk, dict) and 'gain_to_40' in fk:
            g0, lo, hi = fk['gain_to_40']
            w('\nDoes recording more participants look worthwhile? '
              + ('For the average accuracy, little: the curve of the final model has nearly flattened, and even the '
                 'upper end of the extrapolated gain to 40 persons stays below 1 point. ' if hi < 0.01 else
                 'For the average accuracy, marginally: the extrapolated gain to 40 persons is positive but below 1 '
                 'point at its estimate. ' if g0 < 0.01 and lo > 0 else
                 'Yes, by the extrapolated gain to 40 persons. ' if lo > 0 else
                 'Not clearly: the extrapolated gain to 40 persons has a band that includes 0. ')
              + 'The band resamples the repetitions at fixed persons, so it understates the uncertainty, and the curve '
                'is fitted to few sizes (an extrapolation, not a forecast). More participants would still narrow every '
                'interval of this report and widen the coverage of movement styles, grips and natural use, where the '
                'robustness checks above found the open risks.')
        w('')
    # ── 8. segmentation and filters
    if _has('final_variants') and _read('final_variants')['rows']:
        src = _src('final_variants')
        w('## 8. Segmentation errors and a causal filter (FINAL_13; lane p1\'s variant arrays)\n')
        for r in _read('final_variants')['rows']:
            v = r['variant']
            w(f'- {r["description"]}: {n.pts(f"var_{v}_d", r["d_acc"], src, f"{v}: accuracy change")} points '
              f'{n.pts_ci(f"var_{v}_d", r["d_acc_ci"], src, f"{v}")} (Holm p {n.p(f"var_{v}_holm", r["holm_acc"], src, f"{v}: Holm p")}).')
        w('')
    # ── 9. export
    if _has('final_export'):
        ex = _read('final_export')
        src = _src('final_export')
        w('## 9. Production export (`reports/stage2/export/final_model.json`, `stage2/lanes/final_infer.py`)\n')
        w(f'- Fitted on every person with the procedure of the outer folds (D2 cap; the inner person splits of the model '
          f'for its in-fold calibrator; the shipped remedy and tau_k fitted on cross-fitted predictions of the persons). '
          f'JSON '
          f'{n("export_kb", ex["json_kb"], ".0f", src, "size of final_model.json (kB)")} kB, '
          f'{n("export_params", ex["n_parameters"], "d", src, "fitted numbers of the model")} fitted numbers, '
          f'{n("export_ms", ex["ms_per_prediction_numpy"], ".2f", src, "ms per prediction (numpy reference)")} ms per '
          f'prediction in numpy.')
        w(f'- The numpy-only reference implementation reproduces the fitted pipeline on all '
          f'{n("export_rows", ex["n_check_rows"], "d", src, "trials checked")} primary and natural-use trials to '
          f'{n("export_maxdp", ex["max_abs_diff_reference"], ".1e", src, "largest |dP| of the reference implementation")} '
          f'(`tests/test_final_export.py` requires 1e-9).')
        w('- Contents: channel table and preprocessing parameters, body-frame definition, weights, calibrator, tau_k, the '
          'rest-pose screen check (threshold and flip signs), prior 0.5, the decision threshold, the cache data identity.')
        if _has('final_finalist_check'):
            fc_ = _read('final_finalist_check')
            w(f'- `results/finalists_final.json` rebuilds the shipped result `{fc_["name"]}` by the finalists convention '
              f'(module, build_x, factory, params): largest |dP| '
              f'{n("finalist_rebuild_dp", max(fc_["max_abs_dp_primary"], fc_["max_abs_dp_natural_use"]), ".1e", _src("final_finalist_check"), "finalist rebuild: largest difference to the saved predictions")} '
              f'on the primary and natural-use trials '
              f'({"reproduced" if fc_["reproduced"] else "NOT reproduced"}).')
        w('')
    # ── experiments, figures, tables, commands
    reg = C.RES_DIR / 'registry_final.csv'
    w('## Every experiment of the lane (`analysis/reports/stage2/results/<name>.json`; registry `results/registry_final.csv`)\n')
    if reg.exists():
        t = pd.read_csv(reg).sort_values('name')
        from stage2.lanes import final_candidates as FCd
        from stage2.lanes import final_experiments as FEx
        cands_ = FCd.candidates()
        specs = {sp.name: sp for sp in FEx.plan(cands_, fin)}
        what = {'candidate': 'the candidate itself (new in this lane)', 'nested': 'nested cross-lane selection',
                'flip': 'screen-away twins of the test trials', 'grip': 'trained on upright grips only',
                'plat': 'trained on the other platform only', 'noisy': 'trained without the suspect blocks',
                'flipaug': 'screen-flip augmentation of the training rows', 'variant': 'test windows of a p1 variant',
                'cal': 'calibration remedy', 'lc': 'learning curve'}
        for _, r in t.iterrows():
            nm = r['name']
            sp = specs.get(nm)
            if sp is None:
                w(f'- `{nm}.json`: not in the current plan (kept from an earlier plan)')
                continue
            who = lab.get(sp.cand, sp.cand) if sp.cand else 'all candidates'
            arg = ''
            if sp.kind == 'cal':
                arg = ' (chosen inside each fold)' if sp.arg == 'auto' else f' ({sp.arg})'
            elif sp.kind == 'plat' and sp.arg == 'platform_control':
                arg = ' (size-matched control: random persons)'
            elif sp.kind == 'flipaug':
                arg = ', screen-away twins tested' if sp.arg == 'xflip' else ', recorded trials tested'
            elif sp.kind == 'variant':
                arg = f' ({FEx.VARIANTS_TEST.get(sp.arg, sp.arg)})'
            elif sp.kind == 'lc':
                arg = f' ({sp.arg[0]} training persons, repetition {sp.arg[1]})'
            w(f'- `{nm}.json`: {who}: {what.get(sp.kind, sp.kind)}{arg}.')
        missing = [sp for sp in FEx.plan(cands_, fin) if not (C.RES_DIR / f'{sp.name}.json').exists()]
        if missing:
            excluded = all(sp.cand != fin and sp.cand not in dec['robust_leaders'] for sp in missing)
            w(f'\nNot run ({n("n_not_run", len(missing), "d", "results/registry_final.csv", "planned experiments without a saved result")} '
              f'planned experiments, left out for the compute budget of the lane'
              + ('; all are diagnostics of candidates that the decision already excludes (neither the final model nor a '
                 'robust leader)' if excluded else '')
              + '; `python -m stage2.pipeline compute --lanes final` runs them): '
              + ', '.join(f'`{sp.name}`' for sp in missing) + '.')
    w('\nReproduction checks (`final_rep_<key>`) are not saved results: they compare with the lanes\' saved results '
      '(`tables/final/final_reproduction.json`).\n')
    caps = json.loads((L.fig_dir / 'captions.json').read_text(encoding='utf-8')) if (L.fig_dir / 'captions.json').exists() else {}
    w('## Figures (`analysis/reports/stage2/figures/final/`, plotted data in `tables/final/<name>.csv`)\n')
    for k in sorted(caps):
        w(f'- `{k}`: {caps[k]["decision"]}')
    from stage2.lanes.final_report import TABLES
    w('\n## Tables (`analysis/reports/stage2/tables/final/`, CSV + booktabs .tex)\n')
    for k, (_, cap) in TABLES.items():
        w(f'- `{k}`: {cap}')
    w('\nDerived data (CSV / JSON only): `final_overview_rows`, `final_reproduction_rows`, `final_selection_rows`, '
      '`final_*.json`.\n')
    w('## Regenerate\n')
    w('```\ncd analysis\n'
      '.venv/Scripts/python.exe -m stage2.pipeline compute --lanes final              # missing experiments, derive, export\n'
      '.venv/Scripts/python.exe -m stage2.pipeline compute --lanes final --only final_derive final_export --force\n'
      '.venv/Scripts/python.exe -m stage2.lanes.final_nested --jobs @jobs.txt          # optional: fill the nested cache in workers\n'
      '.venv/Scripts/python.exe -m stage2.lanes.final_experiments --names "final_grip_*"   # optional: experiments in a worker\n'
      '.venv/Scripts/python.exe -m stage2.pipeline figures --lanes final              # from saved files only\n'
      '.venv/Scripts/python.exe -m stage2.pipeline tables --lanes final\n'
      '.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes final              # numbers, this summary, finalists\n'
      '.venv/Scripts/python.exe -m pytest stage2/tests/test_final.py stage2/tests/test_final_export.py -q -p no:cacheprovider\n```\n')
    # ── conclusions and caveats
    w('## Conclusions\n')
    rows_d = {r['key']: r for r in dec['rows']}
    fails_grip = [k for k in dec['leaders'] if rows_d[k]['grip_fails']]
    w(f'1. **The final model is {lab.get(fin, fin)}** (`FINAL_MODEL`, which '
      f'{"agrees with" if dec["agrees_with_rule"] else "differs from"} the pre-specified rule): of the '
      f'{n("concl_n_leaders", len(dec["leaders"]), "d", _src("final_decision"), "leaders (not detectably worse than the lowest log loss)")} '
      f'leaders that the paired comparisons cannot separate from the lowest log loss, '
      + (f'{", ".join(lab.get(k, k) for k in fails_grip)} lose accuracy on a grip absent from training after Holm, and '
         if fails_grip else '')
      + f'the robust leaders are {", ".join(lab.get(k, k) for k in dec["robust_leaders"]) or "none"}; the rule takes the '
        f'most portable of them.')
    if ns:
        w(f'2. The honest headline of the whole selection is the nested estimate, '
          f'{n.pct("nested_accuracy", ns["nested"]["accuracy"], _src("final_nested"), "nested selection: accuracy")} '
          f'{n.ci("nested_accuracy", ns["nested"]["accuracy_ci"], _src("final_nested"), "nested selection: accuracy")}; '
          f'the best candidate picked on the outer folds looks '
          f'{n.pts("optimism_accuracy", ns["optimism"]["accuracy"]["diff"], _src("final_nested"), "optimism of the naive pick: accuracy")} '
          f'points better (optimism), and the lanes\' own design choices add an optimism that no nested estimate here '
          f'removes.')
    ic = {r['comparison']: r for r in (_read('final_input_class')['rows'] if _has('final_input_class') else [])}
    same = [ic[k] for k in ('cnn_vs_m2_same12', 'cnn_vs_lrtc_same') if k in ic]
    if same:
        sig_same = [r for r in same if _sig(r['holm_acc']) and r['d_acc'] > 0]
        if len(sig_same) == len(same):
            w('3. Input or model class: on the same body-frame input the 1D CNN is more accurate than the engineered '
              'models given those channels (' + '; '.join(
                  f'than {lab.get(r["b"], r["b"])} by '
                  f'{n.pts("concl_" + r["comparison"], r["d_acc"], _src("final_input_class"), "CNN minus engineered model on the same input: accuracy")} '
                  f'points' for r in same) + ', Holm-adjusted): the gain needs more than the input.')
        else:
            w('3. Input or model class: on the same body-frame input the 1D CNN is not detectably more accurate than at '
              'least one engineered model given those channels, so the input carries most of the gain.')
    if cal:
        ch_ = cal['chosen']
        ov_ = cal['methods'][ch_]['overall']
        state = 'consistent with calibration' if not ov_['miscalibrated'] else 'miscalibrated'
        if ch_ == 'none':
            w(f'4. The percentage is {state} without an extra remedy (no in-fold remedy lowers the log loss after Holm); '
              f'with several attempts the page shows sigmoid(sum of logits / tau_k), not the mean P(right).')
        else:
            w(f'4. The percentage is {state} after the in-fold remedy {ch_} (it lowers the log loss after Holm); with '
              f'several attempts the page shows sigmoid(sum of logits / tau_k), not the mean P(right).')
    rob_rows = _read('final_robustness')['rows']
    n_flip_fail = sum(1 for r in dec['rows'] if r['flip_fails_raw'])
    n_flip_tested = sum(1 for r in rob_rows if (r.get('flip') or {}) and not r['flip'].get('invariant_inputs'))
    w(f'5. Screen held away: '
      f'{n("concl_flip_fail", n_flip_fail, "d", _src("final_decision"), "candidates with a Holm-significant screen-away drop")} '
      f'of the {n("concl_flip_tested", n_flip_tested, "d", _src("final_robustness"), "candidates whose inputs change under the screen flip")} '
      f'candidates whose input changes under the flip lose accuracy on the screen-away twins after Holm, so the '
      f'rest-pose screen check (or a prompt to turn the phone) must run on the inference page whatever the model.')
    comp = ((_read('final_natural').get('components') or {}).get(fin) or {}).get('rows') or []
    lat = next((r for r in comp if r['component'] == 'lateral'), None)
    yaw = next((r for r in comp if r['component'] == 'yaw'), None)
    if lat and yaw:
        w(f'6. Natural use: the final model\'s yaw evidence points to the reported hand in '
          f'{n("concl_yaw_right", yaw["natural_pointing_right_way"], "d", _src("final_natural"), "final model: free-choice trials where the yaw component points to the reported hand")} '
          f'of {n("concl_nat_n", yaw["natural_n"], "d", _src("final_natural"), "free-choice trials")} free-choice trials and its '
          f'lateral evidence in '
          f'{n("concl_lat_right", lat["natural_pointing_right_way"], "d", _src("final_natural"), "final model: free-choice trials where the lateral component points to the reported hand")}'
          + (': the lateral component transfers less to uninstructed movements (as lane m1\'s audit found for its '
             'lateral feature), so leaning on yaw is the safer direction for Stage 3.'
             if lat['natural_pointing_right_way'] < yaw['natural_pointing_right_way'] else '.'))
    p1d = C.TAB_DIR / 'p1' / 'p1_decisions.csv'
    if p1d.exists():
        t = pd.read_csv(p1d)
        dcol = [c for c in t.columns if c.lower().startswith('decision')]
        keep = int(t[dcol[0]].astype(str).str.contains('keep').sum()) if dcol else 0
        w(f'7. Data treatment: lane p1 kept the default in '
          f'{n("p1_keep", keep, "d", "tables/p1/p1_decisions.csv", "p1 choices that keep the default")} of '
          f'{n("p1_choices", len(t), "d", "tables/p1/p1_decisions.csv", "p1 data-treatment choices")} choices, '
          + ('so the final model keeps the cache\'s treatment.\n' if keep == len(t) else
             'and the final model keeps the cache\'s treatment (the other choices are discussed in section 4).\n'))
    w('## Caveats and open questions\n')
    w('- Every candidate reached this lane through its own lane\'s choices on the same outer folds; the nested selection '
      'removes the optimism of choosing among the candidates here, not of the lanes\' own designs (n_configs_tried in '
      'the master table; lane m3\'s body-frame input was designed post hoc).')
    w('- The natural-use set holds one free-choice trial per person, all with the right hand: right-hand recall only, '
      'with wide intervals; the person-matched test is the sharper tool and still rests on few trials.')
    w('- The screen-away grip is synthesised (`screen_flip_channels`), never recorded; the rest-pose check reads gravity '
      'only and assumes a still phone before the movement; unrecorded grips (landscape, phone tilted forward) are untested.')
    w('- The windows are the oracle (app) segments; production windows come from a cue-free detector (lane m4): '
      'section 8 measures the cost of bound errors, and the production end to end belongs to lane prod.')
    w('- The ECE null band treats trials as independent (person clustering ignored): an ECE inside it is good evidence '
      'of calibration, one above it weaker evidence than it looks; slopes and reliability bands resample persons.')
    w('- The learning-curve extrapolation is a fit to few sizes and few persons: an indication, not a forecast.')
