"""reports/stage2/m1_SUMMARY.md and results/m1_numbers.json, rendered together from lane m1's saved files (the
numbers step).

Every number of the summary and of the figure texts goes through one ``m1_texts.Book``, which registers it as a
numbers entry while the text is rendered; a standard block adds every model's main metrics (for LaTeX macros).
No number is typed here: a forced rebuild (a new participant) regenerates the text and the numbers.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.lanes import m1_experiments as E
from stage2.lanes import m1_features as MF
from stage2.lanes import m1_texts as T
from stage2.lanes.m1_texts import (F2, F3, INT, PCT, PTS, PV, SRC_AGG, SRC_CAL, SRC_CAT, SRC_DRIFT, SRC_FAIL, SRC_FIN,
                                   SRC_GRID, SRC_M, SRC_NAT, SRC_SENS, SRC_STRATA, Book, _isnum, _key, join, mkey, render)
from stage2.pipeline import Lane

L = Lane('m1')
PLATFORM_NAMES = {'android': 'Android', 'ios': 'iOS'}
HEADLINE_EXTRA = (E.BASELINE, 'm1_lr_phys6_win_asinh', 'm1_tf_multi', E.REFERENCE)


def _cuefree_earlier(b: Book) -> str:
    """The share of recordings in which the v0 cue-free detector first finds an earlier movement (describe unit)."""
    p = C.RES_DIR / 'describe_numbers.json'
    try:
        v = json.loads(p.read_text(encoding='utf-8'))['cuefree_share_earlier_movement']['value']
    except (OSError, KeyError, json.JSONDecodeError):
        return 'a share'
    return b.t('cuefree_share_earlier_movement', v, '.0%', 'results/describe_numbers.json',
               'recordings in which the v0 cue-free detector first finds an earlier movement (describe unit)')


def _antisymmetric(name: str) -> bool:
    """Is the experiment an antisymmetric logistic model (no intercept, odd features only) by its declared spec?"""
    s = E.BY_NAME.get(name)
    if s is None:
        return False
    p = s.params.get('params', s.params) if s.factory == 'make_train_filter' else s.params
    fac = s.params.get('factory', s.factory) if s.factory == 'make_train_filter' else s.factory
    if fac == 'make_linear_select':
        return all(o.get('antisym', True) and not o.get('augment', False) for o in p['options'].values())
    if fac in ('make_linear', 'make_forward'):
        return bool(p.get('antisym', True))
    return False


def _strata_line(b: Book, st: pd.DataFrame, model: str, strata: tuple[str, ...]) -> str:
    out = []
    for s in strata:
        sub = st[(st['model'] == model) & (st['stratum'] == s)]
        for _, r in sub.iterrows():
            lev = str(r['level'])
            idx = r.get('level_index')
            lk = f'q{int(idx)}' if s == 'duration_bin' and _isnum(idx) else _key(lev)
            k = f'strata_{mkey(model)}_{s}_{lk}'
            lab = f'{model}, {s} {lev}'
            txt = (f'{T.level_text(b, s, lev, idx)} {b.t(f"{k}_acc", r["accuracy"], PCT, SRC_STRATA, f"{lab}: accuracy")} '
                   f'{b.ci(f"{k}_acc", r["accuracy_lo"], r["accuracy_hi"], PCT, SRC_STRATA, f"{lab}: accuracy")}').strip()
            txt += f' (n = {b.t(f"{k}_n", r["n_trials"], INT, SRC_STRATA, f"{lab}: trials")}'
            if not _isnum(r['accuracy_lo']):
                txt += f', {b.t(f"{k}_persons", r["n_persons"], INT, SRC_STRATA, f"{lab}: persons")} persons, no CI'
            out.append(txt + ')')
    return '; '.join(out)


def _rule_groups(b: Book, rel: pd.DataFrame, name: str) -> str:
    rr = rel[rel['model'] == name]
    if not len(rr):
        return ''
    g = (rr.assign(right=rr['p_mean'] >= 0.5, pn=rr['p_mean'] * rr['n'], yn=rr['y_mean'] * rr['n'])
         .groupby('right')[['pn', 'yn', 'n']].sum())
    out = []
    for k, v in g.sort_index(ascending=False).iterrows():
        side = 'right' if k else 'left'
        kk = f'{mkey(name)}_says_{side}'
        out.append(f'when it says {side}, P {b.t(f"{kk}_p", v["pn"] / v["n"], F2, SRC_CAL, f"{name}: mean P(right) when it says {side}")} '
                   f'against an observed share of right hands of {b.t(f"{kk}_obs", v["yn"] / v["n"], F2, SRC_CAL, f"{name}: observed share of right hands when it says {side}")} '
                   f'(n = {b.t(f"{kk}_n", v["n"], INT, SRC_CAL, f"{name}: trials when it says {side}")})')
    return '; '.join(out)


# ───────────────────────────── the standard block (every model's main numbers) ─────────────────────────────


def standard(b: Book, mt: pd.DataFrame) -> None:
    b.reg('ci_level', 0.95, '.0%', SRC_M, 'level of every confidence interval')
    b.reg('k_headline', 3, INT, SRC_M, 'attempts averaged in the headline multi-attempt accuracy')
    b.reg('ece_bins', C.ECE_BINS, INT, SRC_CAL, 'equal-mass bins of the ECE')
    b.reg('n_boot', C.N_BOOT, INT, SRC_M, 'person-bootstrap resamples')
    b.reg('alpha_level', T.ALPHA, F2, SRC_M, 'significance level of every test (after the Holm correction)')
    b.reg('calibration_slope_target', 1.0, '.0f', SRC_CAL, 'calibration slope of a calibrated model')
    for _, r in mt.iterrows():
        T.acc(b, r)
        T.auc(b, r)
        T.ll(b, r)
        T.brier(b, r)
        T.ece(b, r)
        T.cox(b, r)
        T.nat(b, r)
        T.k3(b, r)
        T.configs(b, r)
        for pre in ('vs_ref', 'vs_base', 'vs_rule', 'vs_anchor'):
            T.paired(b, r, pre)
    for f in T.finalists():                    # the finalists by slot: \MOneFinSmallAcc, ... (whatever the name)
        if f['name'] not in mt.index:
            continue
        r = mt.loc[f['name']]
        fb = Book()
        for fn in (T.acc, T.auc, T.ll, T.brier, T.ece, T.cox, T.nat, T.k3, T.configs):
            fn(fb, r)
        for pre in ('vs_ref', 'vs_base', 'vs_rule', 'vs_anchor'):
            T.paired(fb, r, pre)
        stem = mkey(f['name']) + '_'
        for k, e in fb.entries.items():
            if k.startswith(stem):
                b.entries[f'fin_{f["slot"]}_{k[len(stem):]}'] = dict(e, description=f'finalist ({f["slot"]}): {e["description"]}')
        b.reg(f'fin_{f["slot"]}_name', f['name'], '', SRC_FIN, f'finalist of the {f["slot"]} slot')


# ───────────────────────────── the summary ─────────────────────────────


def render_summary(b: Book) -> str:
    mt = T.models()
    mj = L.read_json('m1_models')
    cat = pd.DataFrame(L.read_json('m1_catalogue')['rows']).set_index('name', drop=False)
    catj = L.read_json('m1_catalogue')
    st = T.rows('m1_strata')
    cal = L.read_json('m1_calibration')
    cm = pd.DataFrame(cal['metrics'])
    cp = pd.DataFrame(cal['paired'])
    rel = pd.DataFrame(cal['reliability'])
    ka = T.rows('m1_k_aggregation')
    fl = L.read_json('m1_failures')
    nj = L.read_json('m1_natural')
    dr = L.read_json('m1_drift')
    grid = T.rows('m1_grid')
    sens = pd.DataFrame(L.read_json('m1_sensitivity')['rows'])
    coef = pd.DataFrame(L.read_json('m1_phys6_coefficients')['rows'])
    fins = T.finalists()
    slots = T.slot_names()
    focus = fl['model']
    base = mt.loc[E.BASELINE]
    ref = mt.loc[E.REFERENCE] if E.REFERENCE in mt.index else None
    out: list[str] = []
    w = out.append
    n_p = b.t('n_persons', int(base['n_persons']), INT, SRC_M, 'persons (one leave-one-person-out fold each)')
    n_tr = b.t('n_trials', int(base['n_trials']), INT, SRC_M, 'primary trials')
    own = mt[mt['name'].str.startswith('m1_')]
    n_exp = int(len(own))
    leak_ok = int(own['leak_check'].astype(str).str.startswith('passed (all').sum())

    w('# Lane m1: interpretable models, and why the hand is detectable\n')
    w('*Generated by `python -m stage2.pipeline numbers --lanes m1` from the saved results; do not edit by hand. Every '
      'number below is an entry of `results/m1_numbers.json` (LaTeX macros `\\MOne...` in `results/numbers.tex`), '
      'registered while this text is rendered.*\n')
    w('## Goal\n')
    w('Show how far simple, explainable models go in telling which hand brought the phone to the nose, and explain '
      'physically what carries the hand.\n')
    w('## Evaluation contract\n')
    w(f'- Leave-one-person-out with `stage2.cv.run_experiment`: {n_p} folds, {n_tr} primary trials. '
      f'{b.t("n_experiments", n_exp, INT, SRC_M, "lane m1 experiments")} experiments; '
      f'{b.t("n_leak_all_folds", leak_ok, INT, SRC_M, "experiments whose leak check passed on every fold")} of them passed the '
      'harness leak check on every fold.')
    w('- Hyperparameters, in-fold feature or treatment choices, thresholds, scalers, winsorising limits, spline knots, '
      'calibrators and the aggregation exponent are fitted on the training persons only (inner GroupKFold over persons). '
      'The natural-use trials are scored, never trained on; a P(right) of exactly 0.5 counts one half there.')
    w('- `n_configs_tried` = configurations of a family compared on the outer folds (its family size for an outer-fold '
      'family member, else 1). Designs: ' + '; '.join(E.DESIGN_LABELS.values()) + '. The pilot rule is pre-specified; '
      'PHYS6 was specified after the pooled description of the components.')
    w('- Every p value is an exact sign-flip test over persons; families of comparisons are Holm-adjusted (BH q values '
      'in `tables/m1/m1_models_full.csv`). The ECE comes with the 95th percentile and p value of its perfect-calibration '
      'null (labels redrawn from the model\'s own P), never with a percentile bootstrap CI.\n')

    # ── headline ──
    w('## Headline\n')
    w('| Model | Design (configs) | Accuracy [95 % CI] | AUC [CI] | Log loss [CI] | ECE (null 95 %, p) | Slope [CI] | '
      'Natural use | k = 3 [CI] |')
    w('|---|---|---|---|---|---|---|---|---|')
    names = [f['name'] for f in fins] + [n for n in HEADLINE_EXTRA if n not in {f['name'] for f in fins}]
    for n in names:
        if n not in mt.index:
            continue
        r = mt.loc[n]
        role = next((f'finalist ({f["slot"]})' for f in fins if f['name'] == n), '')
        two = r['group'] == 'rules' and not bool(r.get('hard'))
        w(f'| {T.label(mt, n)}{" (a)" if two else ""} (`{n}`{", " + role if role else ""}) | '
          f'{E.DESIGN_LABELS.get(r["design"], r["design"])} ({T.configs(b, r)}) | {T.acc(b, r)} | {T.auc(b, r)} | '
          f'{T.ll(b, r)} | {T.ece(b, r)} | {T.cox(b, r)} | {T.nat(b, r)} | {T.k3(b, r)} |')
    w('')
    if any(mt.loc[n, 'group'] == 'rules' and not bool(mt.loc[n, 'hard']) for n in names if n in mt.index):
        w('(a) A rule\'s P(right) takes two values per fold (its training hit rate q or 1 - q), so its AUC and its ECE '
          'over equal-mass bins mostly sort the folds by q: neither describes the rule well. Its calibration by its two '
          'decisions is in the calibration section.\n')
    kref = int(base['k_n_persons'])
    w(f'k = 3: per-person balanced accuracy when the P(right) of three attempts of one person and hand are averaged '
      f'({b.t("k_n_persons", kref, INT, SRC_M, "persons of the k-attempt curve")} persons with at least '
      f'{b.t("k_max", max(C.K_ATTEMPTS), INT, SRC_M, "trials per hand needed for the k-attempt curve")} trials per hand). '
      'Slope: calibration slope of the outcome on logit(P) (1 = calibrated; above 1 under-confident).\n')
    nu_all = int(base['nat_n_right']) == int(base['nat_n'])
    w(f'Natural use: {b.t("nat_n_trials", int(base["nat_n"]), INT, SRC_M, "natural-use trials")} free-choice trials, one per '
      f'person, {b.t("nat_n_right", int(base["nat_n_right"]), INT, SRC_M, "natural-use trials done with the right hand")} '
      + ('of them right-hand choices: natural-use accuracy is the recall of the right hand and says nothing about the '
         'left hand.' if nu_all else 'of them right-hand choices.') + '\n')
    w('**Finalists** (`results/finalists_m1.json`, fixed criteria in `stage2/lanes/m1_select.py`, which deviate from '
      'docs/09 D15 on purpose: the product is a probability, so a slot is judged by the log loss, and a replacement '
      'must survive the Holm correction over its family):\n')
    for f in fins:
        r = mt.loc[f['name']]
        w(f'- **{f["slot"]}**: `{f["name"]}`: `{f["module"]}.{f["factory"]}(**params)` on `{f["module"]}.{f["build_x"]}(d)`; '
          f'n_configs_tried = {T.configs(b, r)}. {f["notes"]}')
        w('  - Breakdown (`tables/m1/m1_strata_all.csv`): '
          + _strata_line(b, st, f['name'], ('platform', 'instructed_rotation', 'speed')) + '.')
        fname = str(f['name'])
        cmf = cm[(cm['model'] == fname)]
        if len(cmf):
            parts = []
            for _, q in cmf.iterrows():
                kk = f'cal_{mkey(f["name"])}_{q["scope"]}'
                parts.append(f'{PLATFORM_NAMES.get(q["scope"], q["scope"])}: ECE {T.ece(b, q, SRC_CAL, kk)}, Brier '
                             f'{b.t(f"{kk}_brier", q["brier"], F3, SRC_CAL, fname + f" ({q.scope}): Brier score")}, '
                             f'log loss {b.t(f"{kk}_ll", q["log_loss"], F3, SRC_CAL, fname + f" ({q.scope}): log loss")}')
            w('  - Calibration: ' + '; '.join(parts) + '.')
    w('')
    vr = mt[mt['vs_rule_acc_diff'].notna()] if 'vs_rule_acc_diff' in mt else mt.iloc[:0]
    if len(vr):
        w(f'**What fitted numbers add to the rule** (each paired against `{E.RULE}`; Holm over '
          f'{T.family_size(b, mt, "vs_rule", "vs_rule")} contrasts):\n')
        for _, r in vr.iterrows():
            w(f'- {T.label(mt, r["name"])}: {T.paired(b, r, "vs_rule")}.')
        w('')
    tfm = mt.loc['m1_tf_multi'] if 'm1_tf_multi' in mt.index else None
    if tfm is not None and tfm['name'] not in {f['name'] for f in fins}:
        tq = cm[(cm['model'] == 'm1_tf_multi') & (cm['scope'] == 'all')]
        rem = cp[(cp['baseline'] == 'm1_tf_multi') & (cp['kind'] == 'remedy')]
        best = rem.loc[rem['ll_diff'].idxmin()] if len(rem) else None
        pool = mt[~mt['group'].isin(['reference', 'sensitivity', 'representation'])]
        top = pool.loc[pool['accuracy'].idxmax()]
        txt = (f'**Most accurate model of the lane**: {top["label"]} (`{top["name"]}`, {T.acc(b, top)}; '
               f'{E.DESIGN_LABELS.get(top["design"], top["design"])}, n_configs_tried = {T.configs(b, top)}). '
               f'**The temporal filter on three rates** (`m1_tf_multi`, {T.acc(b, tfm)}, log loss '
               f'{T.ll(b, tfm, ci=False)}, k = 3 {T.k3(b, tfm)}; n_configs_tried = {T.configs(b, tfm)}) against PHYS6: '
               f'{T.paired(b, tfm, "vs_base")}. '
               + (f'Its probabilities show {T.SHAPE_TEXT[T.miscal_shape(tq.iloc[0])].split(":")[0]}: ECE '
                  f'{T.ece(b, tq.iloc[0], SRC_CAL, "cal_tf_multi")}, calibration slope '
                  f'{T.cox(b, tq.iloc[0], SRC_CAL, "cal_tf_multi")}, {T.bins_text(b, tq.iloc[0], "cal_tf_multi", "temporal filter")}.'
                  if len(tq) else ''))
        if best is not None:
            bm = mt.loc[best['model']] if best['model'] in mt.index else None
            txt += (f' The best remedy fitted on inner cross-fitted predictions ({T._remedy(best["model"])}) changes its '
                    f'log loss by {b.t("tf_multi_best_remedy_dll", best["ll_diff"], "+.3f", SRC_CAL, "temporal filter: log-loss change of its best remedy")} '
                    f'{b.ci("tf_multi_best_remedy_dll", best["ll_diff_lo"], best["ll_diff_hi"], "+.3f", SRC_CAL, "temporal filter: log-loss change of its best remedy")} '
                    f'(Holm p {b.t("tf_multi_best_remedy_p_holm", best["p_ll_holm"], PV, SRC_CAL, "temporal filter: Holm p of its best remedy")})'
                    + (f', to a log loss of {T.ll(b, bm, ci=False)} and an ECE of {T.ece(b, bm)}; against the small-model '
                       f'finalist {T.paired(b, bm, "vs_anchor")} (n_configs_tried = {T.configs(b, bm)})'
                       if bm is not None else '') + '.')
        txt += (' It is reported as an exploratory alternative for the synthesis, which re-selects across lanes in a '
                'nested loop.\n')
        w(txt)

    # ── physics ──
    w('## Why the hand is detectable\n')
    w('**A heading-free body frame.** From the gyro-propagated up vector u and the screen normal, lane m1 builds h (the '
      'horizontal direction out of the screen, towards the face) and s = u x h (towards the person\'s right) and projects '
      'the angular velocity w and the linear acceleration a on (u, h, s). Under the left/right mirror u and h are polar '
      'and s is axial, so **yaw (w.u), roll (w.h) and lateral acceleration (a.s) are odd** (they flip with the hand) while '
      'pitch, forward and vertical acceleration are even. The three grips of the protocol differ by a rotation about the '
      'screen normal, which leaves every feature unchanged; under the screen-away flip roll and lateral change sign and '
      'yaw does not (unit tests `test_body_frame_is_invariant_to_the_three_grips`, `test_features_have_exact_mirror_parity`, '
      '`test_grip_labels_match_the_screen_flip`).\n')
    odd, even = cat[cat['parity'] == 'odd'], cat[cat['parity'] == 'even']
    cap1, dec1 = T.t_M1_01(b, {})
    w(f'**Catalogue** (`tables/m1/m1_feature_catalogue.csv|.tex`, figure M1_01): '
      f'{b.t("cat_n_features", len(cat), INT, SRC_CAT, "catalogue features")} features, '
      f'{b.t("cat_n_odd", len(odd), INT, SRC_CAT, "odd catalogue features")} odd and '
      f'{b.t("cat_n_even", len(even), INT, SRC_CAT, "even catalogue features")} even. Univariate leave-one-person-out AUC of '
      'the direction-only rule (direction and scale fitted on the training persons, no intercept; person-bootstrap CI '
      f'with {b.t("cat_n_boot", catj.get("n_boot", C.N_BOOT), INT, SRC_CAT, "bootstrap resamples of the catalogue CIs")} '
      'resamples; leak checks on a seeded sample of folds). A univariate logistic model with an intercept would rank '
      'the held-out persons by their fold\'s intercept, which follows the training class balance (anticorrelated with '
      'the held-out person\'s own balance) and drags the AUC of an uninformative feature below 0.5 '
      '(`test_catalogue_rule_avoids_the_loso_intercept_artefact`).\n')
    top = odd.sort_values('auc', ascending=False).head(6)
    for nm in list(top.index) + [n for n in ('yaw_q50', 'yaw_q100', 'yaw_second_half', 'roll_q25', 'latvel_q75',
                                             'latvel_dd_q100') if n in cat.index and n not in top.index]:
        r = cat.loc[nm]
        k = f'cat_{_key(nm)}'
        w(f'- `{nm}` ({r["meaning"]}): AUC {b.t(f"{k}_auc", r["auc"], F3, SRC_CAT, f"{nm}: univariate LOSO AUC")} '
          f'{b.ci(f"{k}_auc", r["auc_lo"], r["auc_hi"], F3, SRC_CAT, f"{nm}: AUC")}; '
          f'{b.t(f"{k}_persons", r["share_persons_above_50"], ".0%", SRC_CAT, f"{nm}: share of persons with within-person AUC > 0.5")} '
          f'of persons above 0.5.')
    w(f'- Every even feature: AUC between {b.t("cat_even_auc_min", even["auc"].min(), F2, SRC_CAT, "smallest AUC of an even feature")} '
      f'and {b.t("cat_even_auc_max", even["auc"].max(), F2, SRC_CAT, "largest AUC of an even feature")}. A leave-one-person-out '
      'AUC below 0.5 is not evidence for the opposite direction: for a feature without generalisable hand information '
      'the direction learned on the other persons flips with the held-out person; '
      f'{b.t("cat_n_direction_flips", int((~cat["direction_consistent"].astype(bool)).sum()), INT, SRC_CAT, "features whose learned direction changes across folds")} '
      'features change direction across folds.')
    ctl = cat[cat['family'] == 'control']
    lead = cat[cat['name'].str.endswith('_lead')]
    mc = mt.loc['m1_lr_control'] if 'm1_lr_control' in mt.index else None
    ctl_txt = ', '.join(f'{r["name"]} {b.t(f"cat_{_key(r.name)}_auc", r["auc"], F3, SRC_CAT, f"{r.name}: univariate LOSO AUC")} '
                        f'{b.ci(f"cat_{_key(r.name)}_auc", r["auc_lo"], r["auc_hi"], F3, SRC_CAT, f"{r.name}: AUC")}'
                        for _, r in ctl.iterrows())
    not_chance = mc is not None and mc['roc_auc_lo'] > 0.5
    w(f'- Planned negative control (the still hold {b.t("pre_s_start", MF.PRE_S[0], ".1f", SRC_CAT, "start of the hold window before the onset, s")} '
      f'to {b.t("pre_s_end", MF.PRE_S[1], ".1f", SRC_CAT, "end of the hold window before the onset, s")} s before the onset): '
      f'AUC {ctl_txt}.' + (f' Together (`m1_lr_control`): {T.acc(b, mc)}, AUC {T.auc(b, mc)}.' if mc is not None else '')
      + (' **Not at chance**: the phone held still at arm\'s length already carries a weak chiral trace of the hand. Two '
         'explanations fit and cannot be separated here: a hand-specific drift of the held arm, and the settling of the '
         'arm after the previous trial, which in this protocol used the same hand (hands are blocked; the v0 cue-free '
         f'detector finds such an earlier movement first in {_cuefree_earlier(b)} of recordings). In production the hold '
         'is done with the hand being judged, so it is legitimate evidence, but its size here may be inflated by the '
         'blocked protocol.' if not_chance else ' At chance, as expected of a still phone.'))
    lead_txt = ', '.join(f'{r["name"]} {b.t(f"cat_{_key(r.name)}_auc", r["auc"], F3, SRC_CAT, f"{r.name}: univariate LOSO AUC")} '
                         f'{b.ci(f"cat_{_key(r.name)}_auc", r["auc_lo"], r["auc_hi"], F3, SRC_CAT, f"{r.name}: AUC")}'
                         for _, r in lead.iterrows())
    w(f'- Lead-in (the last {b.t("lead_s", MF.LEAD_S, ".1f", SRC_CAT, "lead-in window before the onset, s")} s before the '
      f'oracle onset): AUC {lead_txt}. The movement starts slightly before the detected onset (online_v1 fires on |w| '
      'crossing its threshold): a cue-free detector should start its window earlier, not later.\n')

    # components
    w('**Which component carries the hand** (figure M1_04; antisymmetric logistic models, nested L2; paired against PHYS6 '
      f'with the Holm p over the {T.family_size(b, mt, "vs_base", E.BASELINE + ":options")} options of that family):\n')
    w('| Model | Design | Accuracy [CI] | AUC | Log loss | vs PHYS6 |')
    w('|---|---|---|---|---|---|')
    for n in T.CHANNEL_ROWS:
        if n not in mt.index:
            continue
        r = mt.loc[n]
        vs = '' if n == E.BASELINE else T.paired(b, r, 'vs_base')
        w(f'| {r["label"]} | {E.DESIGN_LABELS.get(r["design"], r["design"])} | {T.acc(b, r)} | {T.auc(b, r, ci=False)} | '
          f'{T.ll(b, r, ci=False)} | {vs} |')
    w('')
    cap4, dec4 = T.t_M1_04(b, {})
    w(f'- {dec4}')
    roll_only = mt.loc['m1_lr_roll2'] if 'm1_lr_roll2' in mt.index else None
    yl = mt.loc['m1_lr_yaw_lat'] if 'm1_lr_yaw_lat' in mt.index else None
    if roll_only is not None and yl is not None:
        roll_adds = T.ci_excludes(yl, 'vs_base', 'acc') < 0 or T.ci_excludes(yl, 'vs_base', 'll') > 0
        w(f'- Roll alone reaches {T.acc(b, roll_only)} (AUC {T.auc(b, roll_only, ci=False)}); '
          + ('it adds measurably to yaw and lateral.' if roll_adds else
             'beyond yaw and lateral it adds nothing measurable: yaw about gravity and lateral motion carry the hand, '
             'roll is a weak, largely redundant early signal.'))
    cap6, dec6 = T.t_M1_06(b, {'band': 'q10_q90'})
    w(f'- **Where in the movement** (figure M1_06, `m1_tf_yaw`, {T.acc(b, mt.loc["m1_tf_yaw"])}): {dec6}')
    w('- **PHYS6 coefficients across folds** (`tables/m1/m1_phys6_coefficients.csv`): '
      + '; '.join(f'`{r["feature"]}` {b.t(f"phys6_coef_{_key(r.feature)}", r["coef_std_median"], "+.2f", "tables/m1/m1_phys6_coefficients.json", f"PHYS6: median coefficient of {r.feature} per RMS unit")} per RMS unit '
                  f'(sign agreement {b.t(f"phys6_coef_{_key(r.feature)}_agree", r["sign_agreement"], ".0%", "tables/m1/m1_phys6_coefficients.json", f"PHYS6: share of folds agreeing with the median sign of {r.feature}")})'
                  for _, r in coef.iterrows()) + '.')
    if {'latvel_q100', 'yaw_q100', 'yaw_q50'} <= set(cat.index):
        w('- **Why the lateral feature behaves like yaw** (a derivation, not a fitted result). The frame (h, s) turns with '
          'the heading of the screen normal at a rate W (the yaw rate w.u for a turn about the vertical; part of the roll '
          'adds to it when the screen is tilted), so ds/dt = -W h and d(v.s)/dt = a.s - W (v.h). For a movement that starts '
          'and ends at rest, the integral of a.s over the approach is the integral of W (v.h): the turn rate weighted by '
          'the speed towards the face, plus sensor drift. The univariate AUC of `latvel_q100` '
          f'({b.t("cat_latvel_q100_auc", cat.loc["latvel_q100", "auc"], F3, SRC_CAT, "latvel_q100: univariate LOSO AUC")}) lies '
          + ('above' if cat.loc['latvel_q100', 'auc'] > cat.loc['yaw_q100', 'auc'] else 'below')
          + f' that of the net yaw ({b.t("cat_yaw_q100_auc", cat.loc["yaw_q100", "auc"], F3, SRC_CAT, "yaw_q100: univariate LOSO AUC")}); '
          f'the first-half yaw reaches {b.t("cat_yaw_q50_auc", cat.loc["yaw_q50", "auc"], F3, SRC_CAT, "yaw_q50: univariate LOSO AUC")}.')
    w('')

    # natural-use transfer
    nt = pd.DataFrame(nj['transfer'])
    w('**Does each component transfer to natural use?** (figure M1_16, `tables/m1/m1_natural_transfer.csv`). For every '
      'free-choice trial, the feature\'s sign (direction learned on the other persons) is compared with the reported '
      'hand; the expected count is the Poisson-binomial count from the same persons\' instructed trials of that hand, '
      f'so differences between persons cannot explain a shortfall. Holm over the '
      f'{b.t("nat_n_features", len(nt), INT, SRC_NAT, "features of the natural-use transfer test")} features.\n')
    w('| Feature | Observed | Expected [95 % range] | p | Holm p |')
    w('|---|---|---|---|---|')
    for _, r in nt.iterrows():
        k = f'nat_{_key(r["feature"])}'
        w(f'| `{r["feature"]}` | {b.t(f"{k}_observed", r["observed"], INT, SRC_NAT, f"{r.feature}: free-choice trials pointing to the reported hand")} of '
          f'{b.t("nat_transfer_n", r["n"], INT, SRC_NAT, "free-choice trials in the transfer test")} | '
          f'{b.t(f"{k}_expected", r["expected"], ".1f", SRC_NAT, f"{r.feature}: expected count from the persons instructed trials")} '
          f'[{b.t(f"{k}_q05", r["q05"], INT, SRC_NAT, f"{r.feature}: expected count, 2.5th percentile", True)}, '
          f'{b.t(f"{k}_q95", r["q95"], INT, SRC_NAT, f"{r.feature}: expected count, 97.5th percentile", True)}] | '
          f'{b.t(f"{k}_p", r["p_two_sided"], PV, SRC_NAT, f"{r.feature}: two-sided Poisson-binomial p")} | '
          f'{b.t(f"{k}_p_holm", r["p_two_sided_holm"], PV, SRC_NAT, f"{r.feature}: Holm-adjusted Poisson-binomial p")} |')
    w('')
    w('- ' + T.natural_decision(b, nt, nj))
    for c in nj.get('contributions', []):
        cd = pd.DataFrame(c['rows'])
        lab = T.label(mt, c['model'])
        parts = []
        for _, r in cd.iterrows():
            k = f'natc_{mkey(c["model"])}_{r["component"]}'
            parts.append(f'{r["component"]} {b.t(f"{k}_nat", r["natural_mean"], "+.2f", SRC_NAT, f"{lab}: mean {r.component} contribution, free-choice trials")} '
                         f'against {b.t(f"{k}_prim", r["primary_mean"], "+.2f", SRC_NAT, f"{lab}: mean {r.component} contribution, instructed trials")} '
                         f'(difference {b.t(f"{k}_diff", r["diff"], "+.2f", SRC_NAT, f"{lab}: per-person difference, {r.component}")} '
                         f'{b.ci(f"{k}_diff", r["diff_lo"], r["diff_hi"], "+.2f", SRC_NAT, f"{lab}: difference, {r.component}")}, '
                         f'Holm p {b.t(f"{k}_p_holm", r["p_holm"], PV, SRC_NAT, f"{lab}: Holm p, {r.component}")})')
        w(f'- Logit contributions towards the reported hand, {lab}: ' + '; '.join(parts) + '.')
    rn = nj.get('rule_nested')
    rnm = mt.loc['m1_rule_nested'] if 'm1_rule_nested' in mt.index else None
    if rn and rnm is not None:
        fr = nt.set_index('feature')
        tr_txt = ''
        if rn['mode'] in fr.index:
            kf = f'nat_{_key(rn["mode"])}'
            tr_txt = (f'; that feature points to the reported hand in '
                      f'{b.t(f"{kf}_observed", fr.loc[rn["mode"], "observed"], INT, SRC_NAT, "free-choice trials pointing to the reported hand")} of '
                      f'{b.t("nat_transfer_n", fr.loc[rn["mode"], "n"], INT, SRC_NAT, "free-choice trials in the transfer test")} free-choice trials')
        w(f'- The in-fold best single-feature rule (`m1_rule_nested`) picks `{rn["mode"]}` in '
          f'{b.t("rule_nested_mode_folds", rn["mode_folds"], INT, SRC_NAT, "folds in which the nested rule picks its modal feature")} of '
          f'{b.t("rule_nested_n_folds", rn["n_folds"], INT, SRC_NAT, "folds of the nested rule")} folds{tr_txt}: hence its '
          f'natural-use result of {T.nat(b, rnm)}.')
    pairs = pd.DataFrame(nj['models'].get('pairs', []))
    if len(pairs):
        from stage2.lanes.m1_analysis import holm
        sel = pairs[(pairs['a'] == slots['rule']) | (pairs['b'] == slots['rule'])].copy()
        sel['p_holm'] = holm(sel['p_mcnemar'].to_numpy(dtype=float))
        txt, sig = [], []
        for _, r in sel.iterrows():
            other = r['b'] if r['a'] == slots['rule'] else r['a']
            ro = r['a_only'] if r['a'] == slots['rule'] else r['b_only']
            oo = r['b_only'] if r['a'] == slots['rule'] else r['a_only']
            k = f'nat_mcnemar_{mkey(other)}'
            txt.append(f'{T.label(mt, other)} {b.t(f"{k}_rule_only", ro, INT, SRC_NAT, f"natural use: trials only the rule gets right against {other}")} '
                       f'against {b.t(f"{k}_other_only", oo, INT, SRC_NAT, f"natural use: trials only {other} gets right")} '
                       f'(p {b.t(f"{k}_p", r["p_mcnemar"], PV, SRC_NAT, f"natural use: exact McNemar p, rule against {other}")}, '
                       f'Holm {b.t(f"{k}_p_holm", r["p_holm"], PV, SRC_NAT, f"natural use: Holm-adjusted McNemar p, rule against {other}")})')
            if r['p_holm'] < T.ALPHA:
                sig.append(T.label(mt, other))
        w('- Discordant free-choice trials, the rule against each other model (only the rule right against only the other '
          f'right; exact McNemar, Holm over the {b.t("nat_mcnemar_n", len(sel), INT, SRC_NAT, "McNemar comparisons of the rule")} '
          'comparisons): ' + '; '.join(txt) + '. '
          + (f'After the correction the rule is better than {join(sig)} on these trials.' if sig else
             'None of these differences survives the correction: with one trial per person, the person-matched sign test '
             'above is the sharper evidence.'))
    w('')

    # drift
    dd = mt.loc['m1_lr_phys6_dedrift'] if 'm1_lr_phys6_dedrift' in mt.index else None
    w('**Integration drift of the lateral feature** (detection, why it matters, remedy, evidence):\n')
    w(f'- **Detection.** Over the still hold the lateral acceleration has a median absolute bias of '
      f'{b.t("drift_bias_abs_median", dr["bias_abs_median"], ".3f", SRC_DRIFT, "median |mean a.s| over the still hold, m/s^2")} m/s^2 '
      f'(90th percentile {b.t("drift_bias_abs_q90", dr["bias_abs_q90"], ".3f", SRC_DRIFT, "90th percentile of |mean a.s| over the hold, m/s^2")}); '
      f'carried through the approach it amounts to {b.t("drift_ratio_median", dr["ratio_median"], ".0%", SRC_DRIFT, "median implied drift over |latvel_q100|")} '
      f'of |`latvel_q100`| in the median trial (more than half of it in '
      f'{b.t("drift_share_ratio_above_half", dr["share_ratio_above_half"], ".0%", SRC_DRIFT, "trials whose implied drift exceeds half of |latvel_q100|")} '
      f'of the trials). '
      + ('The bias itself differs between the hands' if (dr['bias_auc_lo'] > 0.5 or dr['bias_auc_hi'] < 0.5) else
         'The bias itself does not differ measurably between the hands')
      + f' (AUC {b.t("drift_bias_auc", dr["bias_auc"], F3, SRC_DRIFT, "AUC of the hold bias")} '
        f'{b.ci("drift_bias_auc", dr["bias_auc_lo"], dr["bias_auc_hi"], F3, SRC_DRIFT, "AUC of the hold bias")}).')
    w('- **Why it matters.** A drift that is a large share of the feature could carry sensor or protocol artefacts '
      'instead of the movement.')
    if dd is not None:
        verdict = T.better(dd, 'vs_base', 'll')
        w(f'- **Remedy and evidence.** The de-drifted features (`latvel_dd_q*`: the hold bias removed before integrating) '
          f'in PHYS6 (`m1_lr_phys6_dedrift`): {T.acc(b, dd)}; against PHYS6 {T.paired(b, dd, "vs_base")}. '
          + ('**Decision**: de-drifting improves the model; adopt it.' if verdict is True else
             '**Decision**: de-drifting does not improve the model (or makes it worse); the raw lateral integral is kept, '
             'and its drift is a stated limitation.'))
    w('')

    # ── data treatment ──
    w('## Data treatment and model form: options, evidence, decision\n')
    fam = f'{E.BASELINE}:options'
    w(f'Each option paired against PHYS6 on the same trials (figure M1_13, `tables/m1/m1_data_treatment.csv|.tex`); '
      f'Holm over the {T.family_size(b, mt, "vs_base", fam)} options of this family:\n')
    w('| Option | Design | dAccuracy (points) [CI], p, Holm | dLog loss [CI], p, Holm |')
    w('|---|---|---|---|')
    tr = mt[mt.get('vs_base_family') == fam]
    for _, r in tr.iterrows():
        w(f'| {r["label"]} | {T.DESIGN_SHORT.get(r["design"], r["design"])} | {T.paired(b, r, "vs_base", "acc")} | '
          f'{T.paired(b, r, "vs_base", "ll")} |')
    w('')
    w('Design: exploratory = an outer-fold family member (its family size as `n_configs_tried`); nested = a choice made '
      'inside every fold; analysis = a window or control, never a finalist candidate.\n')
    w('- **Decision.** ' + T.treatment_decision(b, mt))
    if 'aug_vs_antisym_max_abs_dp' in mj:
        w(f'- **Augmentation vs antisymmetry.** On odd features, mirror augmentation with an intercept is the '
          'antisymmetric model: the intercept vanishes and the z-score of the augmented rows is the RMS scaling; only the '
          'L2 penalty halves (the rows are doubled), so the in-fold choice of C can differ. The two agree on '
          f'{b.t("aug_same_decisions", mj["aug_vs_antisym_same_decisions"], PCT, SRC_M, "decisions shared by the augmented and the antisymmetric PHYS6")} '
          f'of the decisions (max |dP| {b.t("aug_max_dp", mj["aug_vs_antisym_max_abs_dp"], F3, SRC_M, "largest |P(aug) - P(antisym)|")}); '
          'the unit test pins the exact equivalence at C -> 2C.')
    fr = mt.loc['m1_lr_phys6_free'] if 'm1_lr_phys6_free' in mt.index else None
    if fr is not None:
        w(f'- **Unconstrained model** (intercept, even descriptors, no augmentation): {T.acc(b, fr)}; against PHYS6 '
          f'{T.paired(b, fr, "vs_base", "ll")}. It may learn asymmetries of this mostly right-handed sample, which would '
          'not transfer to a left-handed user, so the shipped form stays antisymmetric.')
    sn = mt.loc['m1_lr_sets_nested'] if 'm1_lr_sets_nested' in mt.index else None
    fw = mt.loc['m1_lr_fwd'] if 'm1_lr_fwd' in mt.index else None
    if sn is not None or fw is not None:
        # the verdict follows the saved comparisons: the wider in-fold search is worse in log loss (unadjusted CI
        # above 0), and the nested set choice is not better than PHYS6
        fw_worse = fw is not None and T.ci_excludes(fw, 'vs_base', 'll') > 0
        sn_same = sn is None or T.ci_excludes(sn, 'vs_base', 'll') == 0
        w('- **Guards against the optimism of PHYS6** (specified after a pooled look at all persons): '
          + join([f'the nested channel-set choice (`m1_lr_sets_nested`) {T.acc(b, sn)}, against PHYS6 {T.paired(b, sn, "vs_base")}' if sn is not None else '',
                  f'the in-fold forward selection over every chiral approach feature (`m1_lr_fwd`) {T.acc(b, fw)}, against PHYS6 {T.paired(b, fw, "vs_base")}' if fw is not None else ''])
          + ('. Choosing the set inside the folds costs nothing measurable and the wider in-fold search is worse in log '
             'loss (unadjusted paired CI above 0): the small physical set is not an artefact of the pooled look.'
             if fw_worse and sn_same else '.'))
    edges = grid[(grid['hyperparameter'] != 'option') & ((grid['folds_at_low_edge'] > 0) | (grid['folds_at_high_edge'] > 0))]
    n_edge = (edges['folds_at_low_edge'] + edges['folds_at_high_edge']).astype(int)
    most = edges[n_edge > edges['n_folds'] / 2]
    w(f'- **Hyperparameter grids** (`tables/m1/m1_grid_edges.csv`): the L2 grid is now in half-decade steps '
      f'({b.t("grid_c_min", min(float(x) for x in grid.loc[grid["hyperparameter"] == "C", "grid"].iloc[0].split(";")), ".4g", SRC_GRID, "smallest C of the logistic grid")} '
      f'to {b.t("grid_c_max", max(float(x) for x in grid.loc[grid["hyperparameter"] == "C", "grid"].iloc[0].split(";")), ".4g", SRC_GRID, "largest C of the logistic grid")}); '
      f'{b.t("grid_n_edge_rows", len(edges), INT, SRC_GRID, "experiment-hyperparameter pairs with a fold at a grid edge")} '
      f'experiment-hyperparameter pairs chose an edge in at least one fold'
      + (': ' + join([f'{r["model"]} {r["hyperparameter"]} (in {b.t(f"grid_{mkey(r.model)}_{r.hyperparameter}_edge", k, INT, SRC_GRID, f"{r.model}: folds at an edge of the {r.hyperparameter} grid")} '
                      f'{"fold" if k == 1 else "folds"})' for k, (_, r) in zip(n_edge, edges.iterrows())]) if len(edges) else '')
      + '.' + (f' At an edge in most folds: {join([f"{r.model} {r.hyperparameter}" for _, r in most.iterrows()])}'
               + (f' (the tree\'s depth grid ends at its design cap of '
                  f'{b.t("tree_depth_cap", max(E.TREE_PARAMS["depths"]), INT, SRC_GRID, "largest depth of the shallow-tree grid (design cap)")}, '
                  'kept shallow to stay readable'
                  + ('; the tree is not a finalist' if 'm1_tree' not in {f['name'] for f in fins} else '') + ')'
                  if 'm1_tree' in set(most['model']) else '') + '.' if len(most) else ''))
    w('')

    # ── calibration ──
    w('## Calibration (detection, why it matters, remedy, evidence)\n')
    a_ = cm[(cm['model'] == E.BASELINE) & (cm['scope'] == 'all')].iloc[0]
    temps = cal.get('temperatures', {}).get(f'{E.BASELINE}_cal_temp')
    w(f'- **Detection.** PHYS6: ECE {T.ece(b, a_, SRC_CAL, "cal_phys6")}; {T.bins_text(b, a_, "cal_phys6", "PHYS6")} '
      f'(person-bootstrap bands); calibration slope {T.cox(b, a_, SRC_CAL, "cal_phys6")}'
      + (f'; the temperature fitted on inner predictions is below 1 (sharpen) in '
         f'{b.t("cal_phys6_T_below_1", temps["n_below_1"], INT, SRC_CAL, "folds whose in-fold temperature is below 1")} of '
         f'{b.t("cal_phys6_T_n_folds", temps["n_folds"], INT, SRC_CAL, "folds with an in-fold temperature")} folds' if temps else '')
      + f'. Verdict: {T.SHAPE_TEXT[T.miscal_shape(a_)]} (figure M1_11).')
    w('- **Why it matters.** The product shows P(right) as a percentage: an over- or under-confident model misleads the '
      'user even when its decision is right.')
    rem = cp[cp['kind'] == 'remedy']
    for base_name in E.CAL_BASES:
        rb = rem[rem['baseline'] == base_name]
        if not len(rb):
            continue
        parts = []
        for _, r in rb.iterrows():
            k = f'cal_{mkey(r["model"])}'
            parts.append(f'{T._remedy(r["model"])} log loss {b.t(f"{k}_dll", r["ll_diff"], "+.3f", SRC_CAL, f"{r.model}: log-loss change against its base")} '
                         f'{b.ci(f"{k}_dll", r["ll_diff_lo"], r["ll_diff_hi"], "+.3f", SRC_CAL, f"{r.model}: log-loss change against its base")} '
                         f'(Holm p {b.t(f"{k}_p_holm", r["p_ll_holm"], PV, SRC_CAL, f"{r.model}: Holm p of the log-loss change")}), Brier '
                         f'{b.t(f"{k}_dbrier", r["brier_diff"], "+.4f", SRC_CAL, f"{r.model}: Brier change against its base")} '
                         f'{b.ci(f"{k}_dbrier", r["brier_diff_lo"], r["brier_diff_hi"], "+.4f", SRC_CAL, f"{r.model}: Brier change")}')
        w(f'- **Post-hoc remedies of {T.label(mt, base_name)}** (fitted on the inner cross-fitted predictions of the '
          f'training persons): ' + '; '.join(parts) + '.')
    trt = cp[cp['kind'] == 'treatment']
    for _, t in trt.iterrows():
        q = cm[(cm['model'] == t['model']) & (cm['scope'] == 'all')]
        if not len(q):
            continue
        q = q.iloc[0]
        k = f'cal_{mkey(t["model"])}'
        w(f'- **Remedy by data treatment: {T.label(mt, t["model"])}** against PHYS6: log loss '
          f'{b.t(f"{k}_dll", t["ll_diff"], "+.3f", SRC_CAL, f"{t.model}: log-loss change against PHYS6")} '
          f'{b.ci(f"{k}_dll", t["ll_diff_lo"], t["ll_diff_hi"], "+.3f", SRC_CAL, f"{t.model}: log-loss change against PHYS6")} '
          f'(p {b.t(f"{k}_p_ll", t["p_ll"], PV, SRC_CAL, f"{t.model}: sign-flip p of the log-loss change")}), Brier '
          f'{b.t(f"{k}_dbrier", t["brier_diff"], "+.4f", SRC_CAL, f"{t.model}: Brier change against PHYS6")} '
          f'{b.ci(f"{k}_dbrier", t["brier_diff_lo"], t["brier_diff_hi"], "+.4f", SRC_CAL, f"{t.model}: Brier change")}; '
          f'ECE {T.ece(b, q, SRC_CAL, k)}, slope {T.cox(b, q, SRC_CAL, k)}, {T.bins_text(b, q, k, str(t["model"]))}.')
    small = slots['small']
    sq = cm[(cm['model'] == small) & (cm['scope'] == 'all')]
    post_ok = rem[(rem['baseline'] == E.BASELINE) & (rem['ll_diff_hi'] < 0)]
    if len(sq):
        sq = sq.iloc[0]
        ok = not T.miscalibrated(sq)
        own_ok = rem[(rem['baseline'] == small) & (rem['ll_diff_hi'] < 0)]        # the finalist's own post-hoc remedies
        tr_small = trt[trt['model'] == small]
        tr_better = bool(len(tr_small)) and float(tr_small['ll_diff_hi'].iloc[0]) < 0
        ch = T._options_chosen(small)
        opt_txt = (f'; its data treatment, chosen inside every fold, is {ch.idxmax()} in '
                   f'{b.t(f"{mkey(small)}_option_top_folds", ch.max(), INT, SRC_GRID, f"{small}: folds choosing its modal option")} of '
                   f'{b.t("n_folds", ch.sum(), INT, SRC_GRID, "leave-one-person-out folds")} folds' if len(ch) else '')
        sent = [('no post-hoc map lowers the log loss of PHYS6 beyond its paired CI' if not len(post_ok) else
                 join([T._remedy(m) for m in post_ok['model']]) + ' lower the log loss of PHYS6 beyond its paired CI'),
                f'the small-model finalist (`{small}`{opt_txt}) '
                + ('lowers it beyond the paired CI' if tr_better else 'does not lower it beyond the paired CI')
                + ' and shows ' + T.SHAPE_TEXT[T.miscal_shape(sq)]
                + f' (ECE p {b.t("cal_small_ece_p", sq["ece_p"], PV, SRC_CAL, "small finalist: p of the ECE under perfect calibration")}, '
                  f'{b.t("cal_small_bins_outside", sq["bins_outside"], INT, SRC_CAL, "small finalist: reliability bins outside their band")} '
                  f'bins outside their band)',
                ('none of its own post-hoc remedies lowers its log loss beyond the paired CI, so no calibration step is '
                 'shipped on top of it' if not len(own_ok) else
                 join([T._remedy(m) for m in own_ok['model']]) + ' lower its log loss beyond the paired CI (see its remedies '
                 'above; the finalist rule decides after the Holm correction)')
                + ('' if ok else '; the residual miscalibration is a stated limitation')]
        w('- **Decision.** ' + '; '.join(s[:1].upper() + s[1:] if i == 0 else s for i, s in enumerate(sent)) + '.')
    w('- **Finalists** (figure M1_15):')
    for f in fins:
        q = cm[(cm['model'] == f['name']) & (cm['scope'] == 'all')]
        if not len(q):
            continue
        q = q.iloc[0]
        fname = str(f['name'])
        k = f'fin_{f["slot"]}_cal'
        if f['slot'] == 'rule':
            w(f'  - {T.label(mt, f["name"])}: P takes two values per fold (q or 1 - q), so equal-mass bins only sort the '
              f'folds by q (its ECE of {b.t(f"{k}_ece", q["ece"], F3, SRC_CAL, "rule: ECE (inflated by the per-fold constant P)")} '
              f'is inflated by that); by its two decisions: {_rule_groups(b, rel, f["name"])}.')
        else:
            w(f'  - {T.label(mt, f["name"])}: ECE {T.ece(b, q, SRC_CAL, k)}, slope {T.cox(b, q, SRC_CAL, k)}, '
              f'{T.bins_text(b, q, k, fname)}: '
              + ('miscalibrated.' if T.miscalibrated(q) else 'consistent with calibration.'))
    w('')
    am = T.agg_model(ka)
    g = ka[ka['model'] == am]
    if len(g):
        w(f'**Multi-attempt aggregates** of {T.label(mt, am)} (figure M1_12, `tables/m1/m1_k_aggregation.csv|.tex`):\n')
        w('| k | method | accuracy | ECE | lean | log loss [CI] |')
        w('|---|---|---|---|---|---|')
        for _, r in g[g['k'].isin([1, 3, max(C.K_ATTEMPTS)])].iterrows():
            k = f'agg_{mkey(am)}_k{int(r["k"])}_{r["method"]}'
            w(f'| {b.t(f"agg_k_{int(r.k)}", int(r["k"]), INT, SRC_AGG, "attempts averaged")} | {r["method"]} | {b.t(f"{k}_acc", r["accuracy"], PCT, SRC_AGG, f"{am}: accuracy of {int(r.k)}-attempt aggregates, {r.method}")} | '
              f'{b.t(f"{k}_ece", r["ece"], F3, SRC_AGG, f"{am}: ECE of {int(r.k)}-attempt aggregates, {r.method}")} | '
              f'{b.t(f"{k}_lean", r["lean"], "+.3f", SRC_AGG, f"{am}: signed miscalibration, {int(r.k)} attempts, {r.method}")} | '
              f'{b.t(f"{k}_ll", r["log_loss"], F3, SRC_AGG, f"{am}: log loss of {int(r.k)}-attempt aggregates, {r.method}")} '
              f'{b.ci(f"{k}_ll", r["log_loss_lo"], r["log_loss_hi"], F3, SRC_AGG, f"{am}: log loss, {int(r.k)} attempts, {r.method}")} |')
        w('')
        w('- Lean: mean of (observed share - P) towards the predicted side; > 0 under-confident.')
        w('- **Why it matters.** The inference app averages several attempts; the probability shown must stay honest for '
          'the aggregate, not only for one attempt.')
        w('- **Decision.** ' + T.agg_decision(b, ka, am))
        w('')

    # ── failures ──
    w(f'## Failure analysis ({T.label(mt, focus)})\n')
    for s_, lab in (('platform', 'Platform'), ('instructed_rotation', 'Grip'), ('speed', 'Speed'),
                    ('duration_bin', 'Duration quartile'), ('handedness', 'Handedness'), ('hand', 'Hand')):
        line = _strata_line(b, st, focus, (s_,))
        if line:
            w(f'- **{lab}**: {line}.')
    var = st[(st['model'] == focus) & (st['stratum'] == 'variant')].sort_values('accuracy')
    if len(var):
        w('- **Variant** (worst first): ' + _strata_line(b, var.assign(model=focus), focus, ('variant',)) + '.')
    pp = pd.DataFrame(fl['per_person']).sort_values('accuracy')
    low = []
    for _, r in pp.head(4).iterrows():
        pid = str(r['person_id'])
        low.append(f'{pid} {b.t(f"person_{_key(pid)}_acc", r["accuracy"], PCT, SRC_FAIL, f"{pid}: accuracy of the failure-analysis model")} '
                   f'({r["platform"]}, {r["handedness"]}-handed)')
    w(f'- **Persons** (figure M1_09): lowest {join(low)}; '
      f'{b.t("persons_below_70", int((pp["accuracy"] < 0.7).sum()), INT, SRC_FAIL, "persons below 70 % accuracy")} of '
      f'{b.t("fail_n_persons", len(pp), INT, SRC_FAIL, "persons of the failure analysis")} persons below '
      f'{b.t("threshold_70", 0.7, ".0%", SRC_FAIL, "per-person accuracy threshold of the failure count")}.')
    lhp = pp[pp['handedness'] == 'left']
    if len(lhp):
        items = []
        for _, r in lhp.iterrows():
            pid = str(r['person_id'])
            items.append(f'{pid} {b.t(f"person_{_key(pid)}_acc", r["accuracy"], PCT, SRC_FAIL, f"{pid}: accuracy of the failure-analysis model")} '
                         f'(rank {b.t(f"person_{_key(pid)}_rank", r["rank_lowest"], INT, SRC_FAIL, f"{pid}: rank from the lowest accuracy")} '
                         f'from the lowest)')
        min_p = int(L.read_json('m1_strata').get('min_persons_ci', 5))
        n_right_h = int((pp['handedness'] == 'right').sum())
        prior_txt = ('The model is antisymmetric (no intercept, no even feature), so it cannot learn a right-hand prior '
                     + ('from this mostly right-handed sample; ' if n_right_h > len(pp) / 2 else 'from the sample; ')
                     if _antisymmetric(focus) else '')
        w(f'- **Left-handed persons**: {join(items)}. {prior_txt}'
          + ('w' if prior_txt else 'W') + f'ith {b.t("n_left_handed", len(lhp), INT, SRC_FAIL, "left-handed persons")} '
          'left-handed persons a gap cannot be told apart from person-to-person variation (no CI for fewer than '
          f'{b.t("strata_min_persons_ci", min_p, INT, SRC_STRATA, "minimum persons of a level with a CI")} persons).')
    swap_names = ', '.join(f'{x["session_label"]} block '
                           + b.t(f'swap_{_key(x["session_label"])}_block_{int(x["block_index"])}', int(x['block_index']), INT,
                                 SRC_FAIL, 'index of a swap-signature block') for x in fl['swap_blocks']) or 'none'
    sv = sens[sens['drop'] == 'blocks'] if len(sens) else sens
    sv_txt = []
    for _, r in sv.iterrows():
        rr = mt.loc[r['name']] if r['name'] in mt.index else None
        if rr is not None:
            sv_txt.append(f'{T.label(mt, r["anchor"])}: {T.paired(b, rr, "vs_base")}')
    w(f'- **Swap-signature blocks** (`stage2.screen.block_consistency`, data-driven; labels never changed): '
      f'{b.t("swap_n_trials", fl["swap_block_n"], INT, SRC_FAIL, "trials in swap-signature blocks")} trials ({swap_names}); '
      f'accuracy on them {b.t("swap_acc", fl["swap_block_accuracy"], PCT, SRC_FAIL, "accuracy on the swap-signature blocks")}, '
      f'{b.t("swap_acc_without", fl["accuracy_without_swap_blocks"], PCT, SRC_FAIL, "accuracy without the swap-signature blocks")} '
      'on every other trial. Trained without these blocks (evaluation unchanged), against each finalist anchor: '
      + ('; '.join(sv_txt) if sv_txt else 'not run') + '.')
    bc = fl['block_clustering']
    w(f'- **Confident errors** (P(true hand) < {b.t("confident_threshold", fl["confident_threshold"], ".1f", SRC_FAIL, "P(true hand) below which an error is confident")}): '
      f'{b.t("fail_n_confident", fl["n_confident_errors"], INT, SRC_FAIL, "confident errors of the small-model finalist")} of '
      f'{b.t("fail_n_trials", fl["n_trials"], INT, SRC_FAIL, "trials of the failure analysis")} trials '
      f'({b.t("fail_n_errors", fl["n_errors"], INT, SRC_FAIL, "errors of the small-model finalist")} errors in all; '
      '`tables/m1/m1_confident_errors.csv|.tex`, figure M1_14). Their yaw time course has the other hand\'s sign by '
      'construction (they are selected on that evidence), so it says nothing about why they happen.')
    ov = pd.DataFrame(fl['overrepresentation'])
    ov = ov[ov['stratum'].isin(['instructed_rotation', 'speed', 'platform', 'hand', 'handedness'])]
    top = ov.loc[ov['ratio'].idxmax()]
    w(f'- Most over-represented among them: {T.STRATUM_LABELS.get(top["stratum"], top["stratum"])} {top["level"]}, '
      f'{b.t("fail_top_n", top["n_confident_errors"], INT, SRC_FAIL, "confident errors in the most over-represented level")} of '
      f'{b.t("fail_n_confident", fl["n_confident_errors"], INT, SRC_FAIL, "confident errors of the small-model finalist")} confident errors (Wilson '
      f'{b.ci("fail_top_share", top["share_lo"], top["share_hi"], ".0%", SRC_FAIL, "share of confident errors in that level")} %) '
      f'against {b.t("fail_top_share_trials", top["share_trials"], ".0%", SRC_FAIL, "share of trials of that level")} of all '
      f'trials ({b.t("fail_top_ratio", top["ratio"], ".1f", SRC_FAIL, "over-representation ratio of that level")} times); a '
      'handful of trials, so the ratio is uncertain.')
    w(f'- **Block clustering** (non-circular evidence): the confident errors fall in '
      f'{b.t("fail_blocks_with_confident", bc["blocks_with_confident"], INT, SRC_FAIL, "blocks holding a confident error")} of '
      f'{b.t("fail_n_blocks", bc["n_blocks"], INT, SRC_FAIL, "planned blocks with trials")} blocks, against '
      f'{b.t("fail_blocks_null_mean", bc["null_mean"], ".1f", SRC_FAIL, "blocks expected under a within-person permutation")} '
      f'expected when each person\'s confident errors are spread at random over that person\'s trials '
      f'(one-sided p {b.t("fail_blocks_p", bc["p_clustered"], PV, SRC_FAIL, "one-sided p of block clustering")}, '
      f'{b.t("fail_blocks_n_perm", bc["n_perm"], INT, SRC_FAIL, "permutations of the block-clustering test")} permutations).')
    cand = pd.DataFrame(fl.get('candidate_blocks', []))
    if len(cand):
        items = []
        for _, r in cand.iterrows():
            kk = f'cand_{_key(r["session_label"])}_b{int(r["block_index"])}'
            items.append(f'{r["session_label"]} block {b.t(f"{kk}_index", int(r["block_index"]), INT, SRC_FAIL, "index of the candidate block")} ({r["hand"]}, '
                         f'{b.t(f"{kk}_n", r["n"], INT, SRC_FAIL, "trials of the candidate block")} trials, mean P(instructed hand) '
                         f'{b.t(f"{kk}_p", r["mean_p_label"], F2, SRC_FAIL, "mean P(instructed hand) of the candidate block")}, neighbours '
                         f'{b.t(f"{kk}_nb", r["neighbour_mean_p_label"], F2, SRC_FAIL, "mean P(instructed hand) of the neighbouring blocks")})')
        w(f'- **Candidate wrong-hand blocks** (at least two trials, mean P(instructed hand) below the confident threshold; '
          f'labels unchanged; `tables/m1/m1_candidate_blocks.csv`): {join(items)}. Without them the accuracy would be '
          f'{b.t("acc_without_candidates", fl["accuracy_without_candidate_blocks"], PCT, SRC_FAIL, "accuracy outside the candidate blocks (descriptive)")} '
          '(descriptive: an upper bound if they are label errors).')
    s6 = sens[sens['drop'] == 'd6'] if len(sens) else sens
    s6_txt = []
    for _, r in s6.iterrows():
        rr = mt.loc[r['name']] if r['name'] in mt.index else None
        if rr is not None:
            rn = str(r['name'])
            s6_txt.append(f'{T.label(mt, r["anchor"])} (median '
                          f'{b.t("sens_" + mkey(rn) + "_dropped", r["dropped_median"], ".0f", SRC_SENS, rn + ": median training rows dropped per fold")} '
                          f'training trials dropped per fold): {T.paired(b, rr, "vs_base")}')
    if s6_txt:
        w('- **Sensitivity to suspected wrong-hand trials** (the docs/09 D6 screen inside every fold: trials it gives '
          'P(label) < 0.05 and runs of at least three confidently reversed trials are removed from training only): '
          + '; '.join(s6_txt) + '.')
    w('')

    # ── rules ──
    w('## Rules\n')
    for n in [x for x in mt.index if mt.loc[x, 'group'] == 'rules']:
        r = mt.loc[n]
        w(f'- `{n}` ({r["label"]}): {T.acc(b, r)}; ' + _strata_line(b, st, n, ('platform',))
          + f'; median person {b.t(f"{mkey(n)}_person_median", r["person_acc_median"], PCT, SRC_M, f"{n}: median per-person accuracy")}'
          + f'; natural use {T.nat(b, r, short=True)}.')
    w('- The AUC of a rule is not a useful number: its P(right) takes two values per fold, and in leave-one-person-out '
      'the training hit rate is lower when the held-out person is easy, which reorders trials across folds.\n')

    # ── experiments ──
    w('## Experiments\n')
    w('One line per experiment: results JSON (`analysis/reports/stage2/results/<name>.json`), what it is, design, '
      'accuracy [95 % CI], log loss, configurations of its family compared on the outer folds.\n')
    for gname in E.GROUP_ORDER:
        sub = mt[mt['group'] == gname]
        if not len(sub):
            continue
        w(f'**{E.GROUP_LABELS[gname]}**\n')
        for _, r in sub.iterrows():
            notes = str(r['notes'] or '')
            notes = (notes[:1].upper() + notes[1:]).rstrip('.') + '.' if notes else ''
            w(f'- `{r["name"]}.json`: {r["label"]}; {E.DESIGN_LABELS.get(r["design"], r["design"])}; {T.acc(b, r)}; '
              f'log loss {T.ll(b, r, ci=False)}; configs {T.configs(b, r)}. {notes}')
        w('')
    w(f'Catalogue: `tables/m1/m1_catalogue.json` ({b.t("cat_n_features", len(cat), INT, SRC_CAT, "catalogue features")} univariate models through `run_experiment`, not '
      'saved as separate results).\n')

    # ── figures and tables ──
    w('## Figures (`analysis/reports/stage2/figures/m1/`)\n')
    for k in sorted(T.FIG_TEXTS):
        _, dec = T.fig_text(k, b)
        w(f'- `{k}.pdf|png`: {dec}')
    w('')
    w('## Tables (`analysis/reports/stage2/tables/m1/`)\n')
    w('Report-facing (CSV + booktabs .tex): `m1_feature_catalogue`, `m1_models`, `m1_rules`, `m1_rules_per_person`, '
      '`m1_conditions`, `m1_calibration`, `m1_calibration_remedies`, `m1_k_aggregation`, `m1_data_treatment`, '
      '`m1_natural_transfer`, `m1_natural_contributions`, `m1_sensitivity`, `m1_confident_errors`, `m1_candidate_blocks`, '
      '`m1_per_person`, `m1_swap_person_blocks`, `m1_phys6_coefficients`, `m1_small_coefficients`. Full CSVs: '
      '`m1_feature_catalogue_full`, `m1_models_full` (with BH q values), `m1_strata_all`, `m1_calibration_full`, '
      '`m1_calibration_paired`, `m1_k_aggregation_full`, `m1_k_aggregation_paired`, `m1_grid_edges`. Figure data: '
      '`M1_NN_*.csv`. Derived JSON (with cache provenance): `m1_*.json`, per-fold artefacts `m1_store_*.json`.\n')
    w('## Regenerate\n')
    w('```')
    w('cd analysis')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1            # missing / stale experiments, catalogue, derived tables, finalists')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1 --force    # everything of lane m1 again')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1 --only "m1_lr_phys6*" --force   # some experiments (+ derived)')
    w('.venv/Scripts/python.exe -m stage2.pipeline compute --lanes m1 --only m1_catalogue m1_derived --force')
    w('.venv/Scripts/python.exe -m stage2.pipeline figures --lanes m1            # no model is fitted; edit m1_figures.FIGURES first')
    w('.venv/Scripts/python.exe -m stage2.pipeline tables --lanes m1')
    w('.venv/Scripts/python.exe -m stage2.pipeline numbers --lanes m1            # m1_numbers.json, numbers.tex and this summary')
    w('.venv/Scripts/python.exe -m pytest stage2/tests/test_m1.py -q -p no:cacheprovider')
    w('```')
    w('Set `M1_THREADS=1` to run two m1 processes side by side within the lane\'s 2-thread budget '
      '(`M1_SKIP_DERIVE=1` on partial runs, then `compute --only m1_derived`).\n')

    # ── conclusions ──
    w('## Conclusions\n')
    w(f'1. The hand is carried by mirror-odd motion only: odd features reach univariate AUCs up to '
      f'{b.t("cat_odd_auc_max", odd["auc"].max(), F2, SRC_CAT, "largest AUC of an odd feature")}, every even one stays within '
      f'{b.t("cat_even_auc_min", even["auc"].min(), F2, SRC_CAT, "smallest AUC of an even feature")}-'
      f'{b.t("cat_even_auc_max", even["auc"].max(), F2, SRC_CAT, "largest AUC of an even feature")}.')
    if roll_only is not None and yl is not None:
        w(f'2. Yaw about gravity and lateral motion carry the hand ({T.label(mt, "m1_lr_yaw2")} {T.acc(b, mt.loc["m1_lr_yaw2"])}, '
          f'{T.label(mt, "m1_lr_lat2")} {T.acc(b, mt.loc["m1_lr_lat2"])}, together {T.acc(b, yl)}); roll alone gives '
          f'{T.acc(b, roll_only)} and '
          + ('adds measurably to them.' if roll_adds else 'adds nothing measurable to them.'))
    if {'yaw_q50', 'yaw_q100', 'yaw_second_half'} <= set(cat.index):
        w('3. The evidence is time-resolved: a turn early in the reach, reversed later; whole-movement integrals lose part '
          f'of it (first-half yaw AUC {b.t("cat_yaw_q50_auc", cat.loc["yaw_q50", "auc"], F3, SRC_CAT, "yaw_q50: univariate LOSO AUC")}, '
          f'net yaw {b.t("cat_yaw_q100_auc", cat.loc["yaw_q100", "auc"], F3, SRC_CAT, "yaw_q100: univariate LOSO AUC")}, '
          f'second-half yaw {b.t("cat_yaw_second_half_auc", cat.loc["yaw_second_half", "auc"], F3, SRC_CAT, "yaw_second_half: univariate LOSO AUC")}).')
    rule = mt.loc[E.RULE] if E.RULE in mt.index else None
    if rule is not None and len(vr):
        by_ll = [T.label(mt, r['name']) for _, r in vr.iterrows() if T.better(r, 'vs_rule', 'll') is True]
        by_acc = [T.label(mt, r['name']) for _, r in vr.iterrows() if T.better(r, 'vs_rule', 'acc') is True]
        none_ = [T.label(mt, r['name']) for _, r in vr.iterrows()
                 if T.better(r, 'vs_rule', 'll') is not True and T.better(r, 'vs_rule', 'acc') is not True]
        w(f'4. The pilot rule (sign of first-half yaw, its hit rate as P) already reaches {T.acc(b, rule)}. '
          + (('After the Holm correction over the contrasts, fitted models improve on it'
              + (f' in log loss ({"; ".join(by_ll)})' if by_ll else '')
              + (' and' if by_ll and by_acc else '') + (f' in accuracy ({"; ".join(by_acc)})' if by_acc else '')
              + (f'; not so {"; ".join(none_)}' if none_ else '') + '.')
             if by_ll or by_acc else
             'No fitted model improves on it after the Holm correction over the contrasts; the fitted models give a '
             'graded probability instead of two values.'))
    flex = mt[mt.get('vs_base_family') == fam]
    flex_better = [r['label'] for _, r in flex.iterrows() if T.better(r, 'vs_base', 'll') is True]
    flex_worse = [r['label'] for _, r in flex.iterrows() if T.better(r, 'vs_base', 'll') is False]
    w(f'5. Against PHYS6, after the Holm correction over its {T.family_size(b, mt, "vs_base", fam)} options: '
      + (f'lower log loss: {join(flex_better)}' if flex_better else 'no option lowers the log loss')
      + (f'; higher log loss: {join(flex_worse)}' if flex_worse else '') + ' (figures M1_05, M1_13).')
    if len(sq):
        shape0, shape1 = T.miscal_shape(a_), T.miscal_shape(sq)
        w('6. Calibration: PHYS6 shows ' + T.SHAPE_TEXT[shape0].split(':')[0]
          + ('' if shape0 == 'none' else (', which no post-hoc map fixes' if not len(post_ok) else
                                           ', which a post-hoc map reduces'))
          + '; the small-model finalist (data treatment chosen inside the folds) shows '
          + T.SHAPE_TEXT[shape1].split(':')[0] + '.')
    drop = nt[(nt['p_two_sided_holm'] < T.ALPHA) & (nt['observed'] < nt['expected'])]
    if len(drop):
        comps = sorted(set(drop['component']))
        kept = sorted(set(nt.loc[nt['p_two_sided_holm'] >= T.ALPHA, 'component']) - set(comps))
        rule_col = E.BY_NAME[slots['rule']].params.get('col') if slots['rule'] in E.BY_NAME else None
        rule_ok = rule_col is not None and rule_col in set(nt['feature']) and rule_col not in set(drop['feature'])
        w(f'7. On the free-choice trials the {join(comps)} features point to the reported hand less often than on the same '
          'persons\' instructed trials (Holm over the features)'
          + (f', while no {join(kept, "or")} feature shows such a drop' if kept else '')
          + f': the {join(comps)} evidence depends on the instructed protocol.'
          + (f' The rule finalist (`{rule_col}` only) shows no drop and is the robust candidate for natural use; with '
             'one free-choice trial per person this rests on few trials.' if rule_ok else ''))
    w('')
    w('## Caveats\n')
    w(f'- {n_p} persons: per-person CIs are wide, and the CI of every model spans several points.')
    w('- Outer-fold family members (n_configs_tried > 1) are exploratory; the headline numbers come from pre-specified, '
      'post-pooled or nested designs. PHYS6 was specified after a pooled look at all persons (not pre-registered); its '
      'guards are reported above. The nested treatment selection was added after the first run had seen the outer-fold '
      'result of asinh compression.')
    w('- The ECE null treats trials as independent given P; person effects make the real sampling variation larger, so '
      'its p value is optimistic. The reliability bands and the calibration slope resample persons.')
    w('- Roll and lateral acceleration assume the screen faces the person (all recorded grips); under a screen-away grip '
      'they change sign. Yaw does not. A production model must check the grip or use yaw-only features for screen-away '
      'use.')
    w('- The natural-use set is one free-choice trial per person, all right-hand choices: it measures right-hand recall '
      'only, with a wide Wilson interval.')
    w('- The pilot rule\'s direction comes from P01\'s pilot session, which is part of the evaluation set; the finalist '
      'rule learns its direction in-fold instead.')
    w('- The oracle onset comes from online_v1; cue-free production windows will differ (lane on detection).')
    w('')
    w('## Open questions\n')
    w('- Should the deployed model add the lead-in window? It carries roll evidence but depends on the detector\'s onset '
      'placement.')
    w('- Is the screen-away grip possible in the PI session? If so, the roll and lateral features need a grip check.')
    w('- The candidate wrong-hand blocks: keep the instructed labels (current rule) or ask the participants?')
    w('- The lateral component transfers less well to natural use: a yaw-dominant model may be the safer product even '
      'at a lower instructed accuracy.')
    return '\n'.join(out) + '\n'


def build() -> tuple[str, Book]:
    """(summary text, the book of every number it and the figure texts cite, plus the standard block)."""
    b = Book()
    for name in T.FIG_TEXTS:
        try:
            T.fig_text(name, b)
        except FileNotFoundError:
            continue
    text = render_summary(b)
    standard(b, T.models())
    return text, b


def write() -> None:
    text, b = build()
    L.write_numbers(b.entries)
    L.summary_path.parent.mkdir(parents=True, exist_ok=True)
    L.summary_path.write_text(text, encoding='utf-8', newline='\n')
