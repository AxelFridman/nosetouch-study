"""Lane m1 finalists, chosen from the saved model table by criteria fixed before the second run
(results/finalists_m1.json).

The criteria deviate from docs/09 D15 ("ship the simplest rung whose ACCURACY lies inside the 95 % CI of the
best rung") on purpose: the product is a probability, so a slot is judged by the log loss, and a replacement
must survive the Holm correction over its family (the first run's rule used uncorrected paired CIs, which the
audit showed to be one test out of 23).

1. **Rule**: the pilot's rule (sign of first-half yaw) with its direction learned on the training persons and its
   training hit rate as the probability (``m1_rule_fhy_sign_cal``), pre-specified (docs/09 section 2.1):
   n_configs_tried = 1. The fitted threshold and the in-fold best single-feature rule are reported against it,
   never chosen over it on the outer folds.
2. **Small logistic model**: the antisymmetric PHYS6 logistic model with its data treatment chosen inside every
   fold (``m1_lr_phys6_nested``): nested, n_configs_tried = 1. A member of the small-model family
   (``m1_experiments.FAMILIES['small_lr']``: other feature sets, treatments, temporal filters, calibration
   remedies) replaces it only when its log loss is lower with a paired person-bootstrap 95 % CI below zero AND a
   Holm-adjusted sign-flip p < 0.05 over the family; it then carries the family size as n_configs_tried.
3. **Tree or GAM**: the member of the tree / GAM family (``FAMILIES['tree_gam']``) with the lowest
   leave-one-person-out log loss; n_configs_tried = the family size (exploratory).

Each entry rebuilds exactly: ``getattr(import_module(module), factory)(**params)`` is the fit_predict and
``getattr(import_module(module), build_x)(d)`` the input (per trial, no label).
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from stage2.lanes import m1_experiments as E
from stage2.pipeline import Lane

L = Lane('m1')
MODULE = 'stage2.lanes.m1_models'
ALPHA = 0.05
SLOTS = ('rule', 'small', 'tree_gam')


def _entry(name: str, slot: str, n_configs: int, why: str, mt: pd.DataFrame) -> dict[str, Any]:
    s = E.BY_NAME[name]
    r = mt.loc[name] if name in mt.index else None
    params = E.spec_params_json(s)
    nat = ''
    if r is not None and pd.notna(r.get('nat_acc_half')):
        n, n_right = int(r['nat_n']), int(r.get('nat_n_right', -1))
        nat = (f' Natural use: {round(float(r["nat_acc_half"]) * n, 1):g} of {n} free-choice trials correct (a P of exactly 0.5 '
               'counts one half)' + (' - all right-hand choices, so this is right-hand recall.' if n_right == n else '.'))
    return {'name': name, 'results_json': f'results/{name}.json', 'module': MODULE, 'build_x': s.x,
            'factory': s.factory, 'params': params, 'n_configs_tried': int(n_configs), 'slot': slot,
            'design': s.design, 'nested_tuning': bool(s.nested),
            'notes': ' '.join(x[:1].upper() + x[1:].rstrip('.') + '.' for x in (s.label, s.notes, f'Chosen because: {why}')
                              if x) + nat}


def choose_finalists() -> list[dict[str, Any]]:
    try:
        mt = pd.DataFrame(L.read_json('m1_models')['rows']).set_index('name', drop=False)
    except FileNotFoundError:
        return []
    out = []
    rule = E.ANCHORS['rule']
    if rule in mt.index:
        out.append(_entry(rule, 'rule', 1, 'the pre-specified pilot rule (docs/09 2.1), direction learned in-fold, its '
                                           'training hit rate as P', mt))
    anchor = E.ANCHORS['small']
    if anchor in mt.index:
        fam = [n for n in E.FAMILIES['small_lr'] if n in mt.index]
        cand = mt.loc[fam] if fam else mt.iloc[:0]
        if len(cand) and 'vs_anchor_ll_diff_hi' in cand:
            cand = cand[(cand['vs_anchor_ll_diff_hi'] < 0) & (cand['vs_anchor_p_ll_holm'] < ALPHA)]
        else:
            cand = cand.iloc[:0]
        if len(cand):
            pick = cand['log_loss'].idxmin()
            out.append(_entry(pick, 'small', E.FAMILY_SIZE['small_lr'],
                              f'lower log loss than {anchor} beyond the paired 95 % CI and after the Holm correction '
                              f'over the {E.FAMILY_SIZE["small_lr"]} configurations of the small-model family '
                              '(exploratory)', mt))
        else:
            out.append(_entry(anchor, 'small', 1,
                              'nested: the data treatment is chosen inside every fold; no member of the small-model '
                              'family beats it in log loss after the Holm correction over the family', mt))
    fam = [n for n in E.FAMILIES['tree_gam'] if n in mt.index]
    if fam:
        pick = mt.loc[fam, 'log_loss'].idxmin()
        out.append(_entry(pick, 'tree_gam', E.FAMILY_SIZE['tree_gam'],
                          f'lowest leave-one-person-out log loss among the {E.FAMILY_SIZE["tree_gam"]} tree / GAM '
                          'configurations (an outer-fold choice: exploratory)', mt))
    return out


def read_finalists() -> list[dict[str, Any]]:
    """The saved finalists (empty before the first compute)."""
    p = L.finalists_path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else []


def slot_names() -> dict[str, str]:
    """slot -> experiment name of the saved finalists, the slot anchors where none is saved yet."""
    out = {'rule': E.ANCHORS['rule'], 'small': E.ANCHORS['small'], 'tree_gam': E.ANCHORS['tree_gam']}
    for f in read_finalists():
        if f.get('slot') in out:
            out[f['slot']] = f['name']
    return out
