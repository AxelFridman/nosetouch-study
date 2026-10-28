"""Lane m3, second phase and derived files (everything the figures, tables and numbers read).

Phase 2 (data-driven, re-run after every base change):
* **roles**: 'deep' (1D sequence networks: the architecture and data-treatment groups), 'mr' (MiniRocket) and
  'tf' (time-frequency models with signed inputs). The best of a role is the highest LOSO accuracy (ties: lower
  log loss) among its non-demo configurations, with the grip rule for the deep role. It is chosen on the outer folds,
  so its ``n_configs_tried`` is the size of the role, recorded on the base result itself (re-saved through
  ``cv.run_experiment``; the fits are cached) and on its calibration variants;
* **calibration remedies** of each role's best model (``m3_cal_<role>_<method>``: platt, temperature, isotonic, beta
  and 'nested', the calibrator chosen inside every outer fold), each fitted inside every outer fold on the
  cross-fitted logits of the training persons; the pre-specified remedy (docs/09 D19) is temperature for an
  antisymmetric model and Platt otherwise;
* **nested deep estimates**: the architecture (``m3_deep_nested``), every deep configuration (``m3_deep_nested_all``)
  and every grip-invariant deep configuration (``m3_deep_nested_inv``, the shipping policy) chosen inside every fold.

Statistics (``m3_stats``): every paired comparison uses ONE estimand, the mean per-person difference, with its paired
person-bootstrap CI and the exact sign-flip p over persons, Holm-adjusted within its family (``FAMILIES``); a verdict
needs the adjusted p below 0.05 and the CI away from 0. ECE is judged against its perfect-calibration null band (no
percentile bootstrap CI), next to the calibration slope and intercept with person-bootstrap CIs.

Derived files (tables/m3/): ``m3_summary.json`` (every experiment), ``m3_selection.json``, ``m3_paired.json`` (every
comparison, by family), ``m3_calibration.json``, ``m3_aggregate.json``, ``m3_breakdown.json`` (the shipped deep
finalist by platform and condition), ``m3_grip_transfer.json``, ``m3_saliency.npz`` + ``.json``, ``m3_tf_example.json``,
``m3_label_noise.json``."""
from __future__ import annotations

import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2 import report_export as RX
from stage2.lanes import m3_common as M
from stage2.lanes import m3_experiments as E
from stage2.lanes import m3_stats as S
from stage2.lanes.m3_pipeline import L, data, inputs, run_spec
from stage2.pipeline import file_current, selected

ROLES = ('deep', 'mr', 'tf')
ROLE_LABELS = {'deep': 'best deep sequence model', 'mr': 'best MiniRocket', 'tf': 'best time-frequency model'}
CAL_METHODS = ('platt', 'temperature', 'isotonic', 'beta', 'nested')
COND_COLUMNS = ('platform', 'speed', 'orientation', 'variant', 'posture', 'start_position')
AGG_METHODS = ('mean_prob', 'mean_logit', 'sum_logit', 'sum_logit_calibrated')
AGG_LABELS = {'mean_prob': 'mean of P(right)', 'mean_logit': 'mean logit', 'sum_logit': 'sum of logits (independence)',
              'sum_logit_calibrated': 'sum of logits x c_k (fitted on training persons)'}
AGG_DRAWS = 200
#: the instructed orientation level of the upright grips (stratification only); every other level is a rotated grip
UPRIGHT_PREFIX = 'upright'
#: grip-transfer check: the gravity-frame default, the device-frame (all channels) and body-frame inputs, each also
#: trained on upright grips only
GRIP_TRANSFER = ('m3_cnn', 'm3_cnn_all', 'm3_cnn_all_upright', 'm3_cnn_body', 'm3_cnn_body_upright')
#: paired grip comparisons on the rotated / upright subsets: (a, b, what, subsets)
GRIP_PAIRS = (('m3_cnn_all_upright', 'm3_cnn_all', 'device frame (17): upright-only vs all-grip training',
               ('rotated', 'upright')),
              ('m3_cnn_body_upright', 'm3_cnn_body', 'body frame (12): upright-only vs all-grip training',
               ('rotated', 'upright')),
              ('m3_cnn_body', 'm3_cnn', 'body (12) vs gravity frame (8), all-grip training', ('rotated', 'upright')),
              ('m3_cnn_body_upright', 'm3_cnn', 'body frame upright-only vs gravity frame all-grip', ('rotated',)))
#: Deep-finalist rule (declared before the body-frame results existed): when the most accurate deep configuration
#: depends on the grip (device-frame channels), the most accurate grip-invariant configuration replaces it unless the
#: paired accuracy difference between them (mean per-person difference, person bootstrap) excludes 0.
GRIP_RULE = ('most accurate configuration; if it depends on the grip (device-frame channels), the most accurate '
             'grip-invariant configuration replaces it unless their paired accuracy difference excludes 0')
#: Engineered-feature comparators of the finalists (paired on the same trials when their results exist). Read-only;
#: another lane's result can change while it runs (the inputs signature then recomputes the derived files).
ENGINEERED = {'example_yaw_logreg': 'reference (yaw logreg)', 'm1_lr_phys6': 'm1 physics logistic (pre-specified)',
              'm2_ens_stack': 'm2 engineered-feature stack (best of lane m2)'}
#: deep vs linear on matched inputs: (m3 model, comparator, comparator label, what is matched)
EQUAL_INPUT = (('m3_cnn', 'm1_lr_yaw2', 'm1 logistic on 2 yaw features',
                'gravity frame only: CNN on the 8 gravity channels vs logistic on the yaw about gravity'),
               ('m3_cnn', 'm1_lr_phys6', 'm1 physics logistic (PHYS6)',
                'CNN on the gravity frame vs logistic on yaw plus body-frame roll and lateral (the CNN has less input)'),
               ('m3_cnn_body', 'm1_lr_phys6', 'm1 physics logistic (PHYS6)',
                'gravity + body frame: CNN on 12 channels vs logistic on yaw, roll and lateral features'))
FAMILIES = {
    'treatment': 'data-treatment and model choices, each against its family default (gravity-frame 1D CNN, '
                 'MiniRocket, STFT logistic)',
    'treatment_body': 'the key treatments repeated on the body-frame input, against the body-frame CNN',
    'grip': 'grip transfer: upright-only against all-grip training and body against gravity frame, on all, rotated '
            'and upright grips',
    'nested': 'nested deep estimates against the pre-specified deep default',
    'equal_input': 'deep against linear models on matched inputs',
    'engineered': 'the shipped m3 finalists and the nested deep estimates against engineered-feature models',
    'physics_demo': 'magnitude-only time-frequency inputs against their signed twins',
    'calib_vs_none': 'every calibration remedy against the uncalibrated model (per-person log loss)',
    'calib_vs_pre': 'every other remedy against the pre-specified one (per-person log loss)',
}


def role_of(spec: E.Spec) -> str | None:
    if spec.physics_demo:
        return None
    if spec.group in ('arch', 'treat'):
        return 'deep'
    if spec.group == 'mr':
        return 'mr'
    if spec.group == 'tf':
        return 'tf'
    return None


def tab(name: str) -> Any:
    return L.tab_dir / name


_RES: dict[str, Any] = {}


def res(name: str) -> Any:
    """A saved result (m3 or a comparator), read once per process."""
    if name not in _RES:
        _RES[name] = L.load_result(name) if name.startswith('m3_') else cv.load_result(name)
    return _RES[name]


def exists(name: str) -> bool:
    return (C.RES_DIR / f'{name}.json').exists()


# ───────────────────────────── phase 2 ─────────────────────────────


def grip_dependent(spec: E.Spec) -> bool:
    return E.grip_dependent(spec)


def selection() -> dict[str, Any]:
    """Each role's best base configuration (accuracy, then log loss), from the saved results, with the grip rule
    (``GRIP_RULE``) applied to the deep role."""
    out: dict[str, Any] = {}
    for role in ROLES:
        cands = [s for s in E.SPECS if role_of(s) == role and s.name in E.BASE and exists(s.name)]
        rows = []
        for s in cands:
            r = L.load_result(s.name)
            rows.append({'name': s.name, 'label': s.label, 'accuracy': r.metrics['accuracy'],
                         'log_loss': r.metrics['log_loss'], 'grip_dependent': grip_dependent(s),
                         'post_hoc': bool(s.post_hoc)})
        if not rows:
            continue
        ranked = sorted(rows, key=lambda r: (-r['accuracy'], r['log_loss'], r['name']))
        best = top = ranked[0]
        rule: dict[str, Any] = {'rule': GRIP_RULE, 'most_accurate': top['name']}
        if top['grip_dependent']:
            inv = [r for r in ranked if not r['grip_dependent']]
            if inv:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cmp = cv.paired_compare(L.load_result(top['name']), L.load_result(inv[0]['name']))
                lo, hi = cmp['mean_person_acc_diff_ci95']
                rule.update({'grip_invariant_alternative': inv[0]['name'], 'acc_diff': cmp['mean_person_acc_diff'],
                             'acc_diff_lo': lo, 'acc_diff_hi': hi, 'p_sign_flip': cmp['p_sign_flip_acc'],
                             'replaced': bool(not lo > 0)})
                if not lo > 0:
                    best = inv[0]
        spec = E.BY_NAME[best['name']]
        anti = bool(spec.params.get('anti')) or spec.params.get('mode') == 'anti'
        out[role] = {'best': best['name'], 'label': best['label'], 'n_configs': len(rows), 'candidates': rows,
                     'antisymmetric': anti, 'prespecified_calibration': 'temperature' if anti else 'platt',
                     'grip_dependent': best['grip_dependent'], 'post_hoc': best['post_hoc'], 'grip_rule': rule}
    return out


def cal_name(role: str, method: str) -> str:
    return f'm3_cal_{role}_{method}'


def selection_note(spec: E.Spec, role: str, k: int) -> str:
    note = spec.notes or ''
    if k > 1:
        note = (f'{note}; ' if note else '') + (f'best of the {k} configurations of the {role} role, chosen on the outer '
                                                 'folds (exploratory)')
    return note


def created_at(name: str) -> str:
    """When a saved result was made (ISO time of ``cv.run_experiment``; '' when missing or unreadable)."""
    try:
        return str(json.loads(L.result_path(name).read_text(encoding='utf-8'))['info'].get('created_at') or '')
    except (OSError, ValueError, KeyError):
        return ''


def saved_before_a_source(name: str, sources: list[str]) -> bool:
    """True when one of the results ``name`` is computed from (its base, or a nested choice's candidates) was saved
    after it: a re-run base experiment makes its calibration variants and nested choices stale."""
    t = created_at(name)
    return bool(t) and any(created_at(s) > t for s in sources if exists(s))


def compute_phase2(force: bool = False, only: list[str] | None = None) -> list[str]:
    """Phase 2. A phase-2 result runs when ``should_run`` says so (missing, stale cache or ``force``), or when it is
    stale against its sources (another base, another role size, or a base / candidate saved after it); with ``only``,
    a stale result also runs when one of its sources is selected (re-running a base refreshes its dependents)."""
    missing = [n for n in E.BASE if not exists(n)]
    if missing:
        print(f'm3: phase 2 skipped: base results missing for {missing}', flush=True)
        return []
    sel = selection()
    L.write_json('m3_selection', {'roles': sel})
    ran = []

    def wanted(name: str, sources: list[str]) -> bool:
        return selected(name, only) or any(selected(s, only) for s in sources)

    # (1) the outer-fold choice is recorded on the base results themselves (n_configs_tried = role size for the kept one)
    for role, s in sel.items():
        for c in s['candidates']:
            k = int(s['n_configs']) if c['name'] == s['best'] else 1
            if selected(c['name'], only) and int(L.load_result(c['name']).info.get('n_configs_tried', 1)) != k:
                spec = E.BY_NAME[c['name']]
                run_spec(spec, n_configs_tried=k, notes=selection_note(spec, role, k))
                ran.append(c['name'])
    # (2) calibration variants of each role's best
    for role, s in sel.items():
        spec = E.BY_NAME[s['best']]
        for method in CAL_METHODS:
            name = cal_name(role, method)
            stale = exists(name) and (L.load_result(name).params.get('base_experiment') != s['best']
                                      or int(L.load_result(name).info.get('n_configs_tried', 1)) != int(s['n_configs'])
                                      or saved_before_a_source(name, [s['best']]))
            if L.should_run(name, force, only) or (wanted(name, [s['best']]) and stale):
                pre = ' (pre-specified, docs/09 D19)' if method == s['prespecified_calibration'] else ''
                how = ('the calibrator chosen inside every outer fold by cross-fitting the candidates over the training '
                       "persons' logits" if method == 'nested' else
                       f'{method} calibration fitted on the cross-fitted training logits')
                run_spec(spec, calib=method, name=name, n_configs_tried=s['n_configs'],
                         notes=f"{s['best']} + {how}{pre}; the base was chosen among {s['n_configs']} configurations on "
                               'the outer folds (exploratory)')
                ran.append(name)
    # (3) nested choices: their candidates' fold fits are cached by now
    for nn in E.NESTED_SPECS:
        ns = E.nested_spec_of(nn)
        cands = [c for c in ns.params.get('candidates', {})]
        stale = exists(ns.name) and saved_before_a_source(ns.name, cands)
        if L.should_run(ns.name, force, only) or (wanted(ns.name, cands) and stale):
            run_spec(ns, n_configs_tried=1)
            ran.append(ns.name)
    return ran


def ALL_NAMES() -> list[str]:  # noqa: N802
    return [*E.BASE, *[cal_name(r, m) for r in ROLES for m in CAL_METHODS], *E.NESTED_SPECS]


def _selection_file() -> dict[str, Any]:
    return json.loads(tab('m3_selection.json').read_text(encoding='utf-8'))['roles']


def spec_of(name: str) -> E.Spec:
    if name in E.BY_NAME:
        return E.BY_NAME[name]
    if name in E.NESTED_SPECS:
        return E.nested_spec_of(name)
    role = name.split('_')[2]
    return E.BY_NAME[_selection_file()[role]['best']]


def shipped_name(role: str, sel: dict[str, Any], cal: dict[str, Any] | None = None) -> str:
    """The result of a role's shipped pipeline: the calibration decision of m3_calibration.json when it exists,
    otherwise the pre-specified remedy."""
    m = ((cal or {}).get(role) or {}).get('decision', {}).get('shipped') or sel[role]['prespecified_calibration']
    return sel[role]['best'] if m == 'none' else cal_name(role, m)


# ───────────────────────────── per-experiment rows ─────────────────────────────


def _fold_infos(r: Any) -> list[dict[str, Any]]:
    seen, out = set(), []
    for f, p in zip(r.predictions['fold'], r.predictions['params']):
        if f in seen:
            continue
        seen.add(f)
        out.append(json.loads(p) if isinstance(p, str) else dict(p))
    return out


def cost_cpu_min(r: Any) -> float:
    infos = _fold_infos(r)
    tot = sum(float(i.get('fit_cpu_s_all', i.get('fit_cpu_s', 0.0)) or 0.0) for i in infos)
    return tot / 60.0


def selection_kind(name: str, sel: dict[str, Any]) -> str:
    """How the configuration came to be: 'pre-specified', 'best of K', 'nested', 'check' or 'demonstration'."""
    if name in E.NESTED_SPECS:
        return 'nested'
    if name.startswith('m3_cal_'):
        return f"best of {sel[name.split('_')[2]]['n_configs']} + remedy"
    s = E.BY_NAME[name]
    if s.physics_demo:
        return 'demonstration'
    role = role_of(s)
    if role and sel.get(role, {}).get('best') == name:
        return f"best of {sel[role]['n_configs']}"
    if s.group in ('grip', 'treat_body'):
        return 'check'
    return 'pre-specified' if not s.baseline else 'comparison'


def experiment_row(name: str, r: Any, sel: dict[str, Any]) -> dict[str, Any]:
    spec = spec_of(name)
    kc = r.k_curve.set_index('k') if len(r.k_curve) else pd.DataFrame()
    infos = _fold_infos(r)
    is_cal = name.startswith('m3_cal_')
    row = {'name': name, 'label': r.model, 'family': spec.family,
           'group': ('calib' if is_cal else 'nested' if name in E.NESTED_SPECS else spec.group),
           'input': spec.x, 'n_trials': int(r.metrics['n']), 'n_persons': int(r.predictions['person_id'].nunique()),
           'n_folds': r.info.get('n_folds'), 'n_configs_tried': int(r.info.get('n_configs_tried', 1)),
           'selection': selection_kind(name, sel), 'post_hoc': bool(spec.post_hoc),
           'nested_tuning': spec.family in ('mr', 'tflog', 'nested', 'nested_all'), 'leak_check': r.leak_check,
           'grip_dependent': grip_dependent(spec), 'cpu_min': cost_cpu_min(r),
           'physics_demo': bool(name in E.BY_NAME and spec.physics_demo),
           'calib': (json.loads(r.predictions['params'].iloc[0]).get('calib') or {}).get('method', 'none') if not is_cal
           else name.split('_', 3)[3]}
    for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier', 'ece'):
        row[k] = r.metrics.get(k)
        lo, hi = r.ci.get(k, (math.nan, math.nan))
        if k != 'ece':                                   # ECE: no percentile bootstrap CI (biased upward)
            row[f'{k}_lo'], row[f'{k}_hi'] = lo, hi
    pr = r.predictions
    en = S.ece_null(pr['y'].to_numpy(), pr['p_right'].to_numpy(float))
    row['ece_null_q95'] = en['ece_null_q95']
    row['ece_null_p'] = en['ece_null_p']
    row['person_acc_median'] = r.metrics.get('person_acc_median')
    row['n_persons_below_80'] = r.metrics.get('n_persons_below_80')
    ex = r.extra_predictions
    nu = S.natural_use(ex['y'].to_numpy(), ex['p_right'].to_numpy(float)) if ex is not None and len(ex) else {'n': 0}
    row['natural_use_n'] = int(nu.get('n', 0))
    for k in ('coverage', 'answered_accuracy', 'answered_lo', 'answered_hi', 'forced_accuracy', 'forced_lo',
              'forced_hi', 'half_credit_accuracy', 'n_answered', 'n_correct'):
        row[f'natural_use_{k}'] = nu.get(k)
    row['natural_use_accuracy'] = nu.get('forced_accuracy')
    for k in C.K_ATTEMPTS:
        if len(kc) and k in kc.index:
            row[f'acc_k{k}'], row[f'acc_k{k}_lo'], row[f'acc_k{k}_hi'] = (kc.loc[k, 'accuracy'], kc.loc[k, 'lo'],
                                                                          kc.loc[k, 'hi'])
    row['k_n_persons'] = int(kc['n_persons'].iloc[0]) if len(kc) else 0
    row['gated_accuracy'] = (r.gated_metrics or {}).get('accuracy')
    if spec.family in ('gripsub', 'gripsub_body'):
        row['train_rows_kept_median'] = float(np.median([i.get('n_train_kept', np.nan) for i in infos]))
    if name in E.NESTED_SPECS:
        ch = pd.Series([i.get('choice') for i in infos]).value_counts()
        row['nested_choices'] = '; '.join(f'{k} x{v}' for k, v in sorted(ch.items(), key=lambda kv: (-kv[1], kv[0])))
        row['nested_n_candidates'] = len(spec.params.get('candidates', {}))
        row['nested_choice_counts'] = {str(k): int(v) for k, v in ch.items()}
        row['grip_dependent_chosen_folds'] = int(sum(v for k, v in ch.items()
                                                     if k in E.BY_NAME and grip_dependent(E.BY_NAME[k])))
    cal_ch = pd.Series([(i.get('calib') or {}).get('nested_choice') for i in infos]).dropna()
    if len(cal_ch):
        row['calib_choices'] = '; '.join(f'{k} x{v}' for k, v in cal_ch.value_counts().sort_index().items())
    row['C_chosen'] = '; '.join(f'{k} x{v}' for k, v in pd.Series([i.get('C') for i in infos if i.get('C') is not None]
                                                                   ).value_counts().items())
    row['epochs_median'] = float(np.median([e for i in infos for e in i.get('epochs', [])])) if any(
        i.get('epochs') for i in infos) else None
    rt = next((i.get('runtime') for i in infos if i.get('runtime')), None)
    row['runtime'] = json.dumps(rt, sort_keys=True) if rt else ''
    return row


def summary(sel: dict[str, Any]) -> dict[str, Any]:
    rows = [experiment_row(n, res(n), sel) for n in ALL_NAMES() if exists(n)]
    out: dict[str, Any] = {'experiments': rows}
    if exists('example_yaw_logreg'):
        ref = res('example_yaw_logreg')
        out['reference'] = {'name': 'example_yaw_logreg', 'accuracy': ref.metrics['accuracy'],
                            'accuracy_lo': ref.ci['accuracy'][0], 'accuracy_hi': ref.ci['accuracy'][1],
                            'roc_auc': ref.metrics['roc_auc'], 'log_loss': ref.metrics['log_loss'],
                            'ece': ref.metrics['ece'],
                            'k_curve': ref.k_curve[['k', 'accuracy', 'lo', 'hi', 'n_persons']].to_dict(orient='records'),
                            'per_person': ref.per_person[['person_id', 'accuracy']].to_dict(orient='records')}
    return out


def label_of(name: str) -> str:
    if name in ENGINEERED:
        return ENGINEERED[name]
    for a, b, lab, _ in EQUAL_INPUT:
        if name == b:
            return lab
    try:
        return spec_of(name).label
    except (KeyError, FileNotFoundError, IndexError):
        return name


# ───────────────────────────── paired comparisons ─────────────────────────────


def _orientation() -> pd.Series:
    d, _ = data()
    return d.trials.set_index('trial_id')['orientation'].astype(str)


def subset_trials(subset: str | None) -> set[str] | None:
    if subset in (None, 'all'):
        return None
    o = _orientation()
    up = o.str.startswith(UPRIGHT_PREFIX)
    return set(o.index[up if subset == 'upright' else ~up].astype(str))


def paired_row(a: str, b: str, family: str, *, subset: str | None = None, **kw: Any) -> dict[str, Any]:
    """A vs B on the same trials (optionally a subset): the mean per-person accuracy difference with its paired
    person-bootstrap CI and exact sign-flip p (cv.paired_compare), the same for log loss and Brier, the pooled trial
    difference (secondary), and whether the two give identical predictions."""
    pa, pb = res(a).predictions, res(b).predictions
    keep = subset_trials(subset)
    if keep is not None:
        pa, pb = pa[pa['trial_id'].astype(str).isin(keep)], pb[pb['trial_id'].astype(str).isin(keep)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cmp = cv.paired_compare(pa, pb)
    j = pa[['trial_id', 'person_id', 'y', 'p_right']].merge(pb[['trial_id', 'p_right']], on='trial_id',
                                                             suffixes=('_a', '_b'))
    ba = S.per_person_brier(j['person_id'], j['y'], j['p_right_a'])
    bb = S.per_person_brier(j['person_id'], j['y'], j['p_right_b'])
    br = S.person_mean_diff(np.array([ba[u] - bb[u] for u in sorted(ba)]))
    same = S.identical(j['p_right_a'], j['p_right_b'])
    lo, hi = cmp['mean_person_acc_diff_ci95']
    plo, phi = cmp['pooled_acc_diff_ci95']
    llo, lhi = cmp['mean_person_logloss_diff_ci95']
    return {'name': a, 'baseline': b, 'family': family, 'subset': subset or 'all', **kw,
            'n_trials': cmp['n_trials'], 'n_persons': cmp['n_persons'],
            'accuracy': cmp['acc_a'], 'baseline_accuracy': cmp['acc_b'],
            'diff': cmp['mean_person_acc_diff'], 'diff_lo': lo, 'diff_hi': hi, 'p': cmp['p_sign_flip_acc'],
            'pooled_diff': cmp['pooled_acc_diff'], 'pooled_lo': plo, 'pooled_hi': phi,
            'logloss_diff': cmp['mean_person_logloss_diff'], 'logloss_lo': llo, 'logloss_hi': lhi,
            'p_logloss': cmp['p_sign_flip_logloss'], 'brier_diff': br['diff'], 'brier_lo': br['lo'],
            'brier_hi': br['hi'], 'p_brier': br['p'], 'persons_better': cmp['persons_a_better'],
            'persons_worse': cmp['persons_b_better'], 'identical': same,
            'max_abs_dp': float(np.max(np.abs(j['p_right_a'] - j['p_right_b']))) if len(j) else math.nan}


def adjust(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Holm within every family, then the verdicts (accuracy, and log loss as a secondary reading)."""
    df = pd.DataFrame(rows)
    if not len(df):
        return rows
    df['family_size'] = df.groupby('family')['p'].transform(lambda s: int(np.isfinite(s).sum()))
    df['p_holm'] = np.nan
    df['p_holm_logloss'] = np.nan
    for _, idx in df.groupby('family').groups.items():
        df.loc[idx, 'p_holm'] = S.holm(df.loc[idx, 'p'].to_numpy(float))
        df.loc[idx, 'p_holm_logloss'] = S.holm(df.loc[idx, 'p_logloss'].to_numpy(float))
    df['verdict'] = [S.verdict(r.diff, r.diff_lo, r.diff_hi, r.p_holm, same=bool(r.identical)) for r in df.itertuples()]
    df['verdict_logloss'] = [S.verdict(-r.logloss_diff, -r.logloss_hi, -r.logloss_lo, r.p_holm_logloss,
                                       same=bool(r.identical), better='lower log loss', worse='higher log loss')
                             for r in df.itertuples()]
    return df.to_dict(orient='records')


def paired(sel: dict[str, Any], cal: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(a: str, b: str, family: str, **kw: Any) -> None:
        if exists(a) and exists(b):
            rows.append(paired_row(a, b, family, **kw))

    for s in E.SPECS:
        if not (s.baseline and s.name in E.BASE):
            continue
        fam = ('physics_demo' if s.physics_demo else 'grip' if s.group == 'grip' else
               'treatment_body' if s.group == 'treat_body' else 'treatment')
        add(s.name, s.baseline, fam, treatment=s.treatment, option=s.option, label=s.label,
            baseline_label=label_of(s.baseline), physics_demo=s.physics_demo, what=s.option)
    for a, b, what, subsets in GRIP_PAIRS:
        for sub in subsets:
            add(a, b, 'grip', subset=sub, treatment='grip transfer', option=f'{what} ({sub} grips)', label=label_of(a),
                baseline_label=label_of(b), physics_demo=False, what=what)
    for nn in E.NESTED_SPECS:
        add(nn, E.D, 'nested', treatment='nested choice', option=label_of(nn), label=label_of(nn),
            baseline_label=label_of(E.D), physics_demo=False, what=label_of(nn))
    for a, b, blab, what in EQUAL_INPUT:
        add(a, b, 'equal_input', treatment='equal input', option=what, label=label_of(a), baseline_label=blab,
            physics_demo=False, what=what)
    firsts = [(ROLE_LABELS[r], shipped_name(r, sel, cal)) for r in ROLES if r in sel]
    firsts += [(label_of(nn), nn) for nn in E.NESTED_SPECS]
    for what, name in firsts:
        for comp, clab in ENGINEERED.items():
            add(name, comp, 'engineered', treatment='vs engineered features', option=f'{what} vs {clab}',
                label=label_of(name), baseline_label=clab, physics_demo=False, what=what)
    return adjust(rows)


# ───────── fast person-cluster bootstrap ─────────


def fast_boot(persons: np.ndarray, y: np.ndarray, p: np.ndarray, *, n_boot: int = C.N_BOOT, seed: int = C.SEED
              ) -> dict[str, tuple[float, float]]:
    """95 % person-cluster bootstrap CIs of accuracy, log loss, Brier and over-confidence (mean confidence minus
    accuracy), vectorised: resampling persons with replacement = multinomial person weights on per-person sums."""
    persons = np.asarray(persons).astype(str)
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    uniq, inv = np.unique(persons, return_inverse=True)
    correct = ((p >= 0.5) == (y == 1)).astype(float)
    conf = np.maximum(p, 1 - p)
    vals = {'accuracy': correct, 'log_loss': -(y * np.log(p) + (1 - y) * np.log(1 - p)), 'brier': (p - y) ** 2,
            'overconfidence': conf - correct}
    cnt = np.bincount(inv, minlength=uniq.size).astype(float)
    rng = np.random.default_rng([seed, 5])
    W = rng.multinomial(uniq.size, np.full(uniq.size, 1.0 / uniq.size), size=n_boot).astype(float)
    den = W @ cnt
    return {k: tuple(float(x) for x in np.quantile((W @ np.bincount(inv, weights=v, minlength=uniq.size)) / den,
                                                    [0.025, 0.975])) for k, v in vals.items()}


# ───────── calibration ─────────


#: simulations of a perfectly calibrated model for the chance rate of the 'bin off' criterion (``bins_off_null_rate``)
BINS_OFF_NULL_SIMS = 400


def _bin_bands(y: np.ndarray, kbin: np.ndarray, inv: np.ndarray, n_persons: int, n_bins: int, W: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """95 % person-bootstrap band of the observed share in every bin (bins fixed): (lo, hi), each (n_bins,)."""
    SY = np.zeros((n_persons, n_bins))
    SN = np.zeros((n_persons, n_bins))
    np.add.at(SY, (inv, kbin), y)
    np.add.at(SN, (inv, kbin), 1.0)
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)              # empty bins (ties share a bin): all-NaN columns
        YB = (W @ SY) / (W @ SN)
        return np.nanquantile(YB, 0.025, axis=0), np.nanquantile(YB, 0.975, axis=0)


def reliability_bands(y: np.ndarray, p: np.ndarray, persons: np.ndarray, n_bins: int = C.ECE_BINS,
                      n_boot: int = C.N_BOOT, seed: int = C.SEED) -> pd.DataFrame:
    """Equal-mass bins (cv.reliability: ties share a bin); per bin the mean P(right), the observed share of right
    hands with its 95 % person-bootstrap band (persons resampled with replacement, bins fixed; vectorised) and the 95 %
    consistency band of the observed share under perfect calibration (y* ~ Bernoulli(p) for the bin's own predictions;
    Broecker and Smith 2007).

    A bin is 'off' only when BOTH readings agree: its person-bootstrap band excludes the bin's mean prediction
    (``off_boot``) AND its observed share lies outside the consistency band (``off_consistency``). The bootstrap band
    alone collapses to a point when every trial of a bin has the same label (an observed share of exactly 0 or 1), so it
    'excludes' any prediction short of 0 or 1 (for example a bin predicted at 0.985 whose 60 trials are all right-handed,
    which a perfectly calibrated model produces with probability 0.985^60 = 0.40); the consistency band alone ignores
    the clustering of trials within persons. The frame also carries ``bins_off_null_rate`` (attrs): how often a
    perfectly calibrated model with these predictions and persons has at least one bin off."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rel = cv.reliability(y, p, n_bins).copy()
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    kbin = np.searchsorted(edges[1:-1], p, side='right')
    used = sorted(set(kbin.tolist()))
    uniq, inv = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    rng = np.random.default_rng([seed, 9])
    W = rng.multinomial(uniq.size, np.full(uniq.size, 1.0 / uniq.size), size=n_boot).astype(float)
    lo, hi = _bin_bands(y, kbin, inv, uniq.size, n_bins, W)
    rel['y_lo'] = [float(lo[b]) for b in used]
    rel['y_hi'] = [float(hi[b]) for b in used]
    # consistency band: the observed share of each bin if the predictions were perfectly calibrated
    rs = np.random.default_rng([seed, 31])
    Ystar = (rs.random((n_boot, p.size)) < p).astype(float)
    M_ = np.zeros((p.size, n_bins))
    M_[np.arange(p.size), kbin] = 1.0
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        share = (Ystar @ M_) / M_.sum(axis=0)                            # (n_boot, n_bins)
        clo, chi = np.nanquantile(share, 0.025, axis=0), np.nanquantile(share, 0.975, axis=0)
    rel['cons_lo'] = [float(clo[b]) for b in used]
    rel['cons_hi'] = [float(chi[b]) for b in used]
    rel['off_boot'] = (rel['y_lo'] > rel['p_mean']) | (rel['y_hi'] < rel['p_mean'])
    rel['off_consistency'] = (rel['y_mean'] < rel['cons_lo']) | (rel['y_mean'] > rel['cons_hi'])
    rel['degenerate_band'] = rel['y_lo'] == rel['y_hi']
    rel['off'] = rel['off_boot'] & rel['off_consistency']
    # chance rate of 'at least one bin off' for a perfectly calibrated model (same predictions, persons and bins)
    hits = 0
    for s in range(BINS_OFF_NULL_SIMS):
        ys = Ystar[s]
        slo, shi = _bin_bands(ys, kbin, inv, uniq.size, n_bins, W[: max(200, n_boot // 10)])
        obs = (ys @ M_) / np.maximum(M_.sum(axis=0), 1)
        pm = (p @ M_) / np.maximum(M_.sum(axis=0), 1)
        ob = ((slo > pm) | (shi < pm)) & ((obs < clo) | (obs > chi))
        hits += bool(ob[used].any())
    rel.attrs['bins_off_null_rate'] = hits / BINS_OFF_NULL_SIMS
    return rel


def calibration_stats(y: np.ndarray, p: np.ndarray, persons: np.ndarray, *, ci: bool = True) -> dict[str, Any]:
    """ECE with its perfect-calibration null band, calibration slope / intercept (CIs when ``ci``), Brier, log loss,
    over-confidence."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    m = cv.metrics(y, p)
    en = S.ece_null(y, p)
    rc = S.recalibration(y, p)
    conf = np.maximum(p, 1 - p)
    out = {'n': int(m['n']), 'n_persons': int(len(set(np.asarray(persons).astype(str)))), 'accuracy': m['accuracy'],
           'roc_auc': m['roc_auc'], 'ece': m['ece'], 'ece_null_mean': en['ece_null_mean'],
           'ece_null_q95': en['ece_null_q95'], 'ece_null_p': en['ece_null_p'], 'brier': m['brier'],
           'log_loss': m['log_loss'], 'mean_confidence': float(conf.mean()),
           'overconfidence': float(conf.mean() - m['accuracy']), **rc}
    if ci and out['n_persons'] >= S.MIN_PERSONS_CI:
        fb = fast_boot(persons, y, p)
        rci = S.recalibration_ci(y, p, persons)
        for k in ('accuracy', 'brier', 'log_loss', 'overconfidence'):
            out[f'{k}_lo'], out[f'{k}_hi'] = fb[k]
        for k in ('cal_slope', 'cal_intercept'):
            out[f'{k}_lo'], out[f'{k}_hi'] = rci[k]
    return out


def miscalibration_reasons(e: dict[str, Any]) -> list[str]:
    """What flags a model as miscalibrated (empty: no evidence)."""
    why = []
    if e['ece'] > e['ece_null_q95']:
        why.append('ECE above the 95th percentile of its perfect-calibration null')
    if e.get('cal_slope_lo') is not None and not e['cal_slope_lo'] <= 1 <= e['cal_slope_hi']:
        why.append('calibration slope CI excludes 1')
    if e.get('cal_intercept_lo') is not None and not e['cal_intercept_lo'] <= 0 <= e['cal_intercept_hi']:
        why.append('calibration intercept CI excludes 0')
    if e.get('bins_off', 0):
        why.append(f"{e['bins_off']} reliability bin(s) off the diagonal (person-bootstrap band and perfect-calibration "
                   'consistency band agree)')
    return why


def calibration(sel: dict[str, Any]) -> dict[str, Any]:
    d, _ = data()
    plat = d.trials.set_index('trial_id')['platform']
    out: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    for role, s in sel.items():
        entries = []
        for method, name in [('none', s['best'])] + [(m, cal_name(role, m)) for m in CAL_METHODS]:
            if not exists(name):
                continue
            pr = res(name).predictions
            y, p, per = pr['y'].to_numpy().astype(int), pr['p_right'].to_numpy(float), pr['person_id'].astype(str).to_numpy()
            rel = reliability_bands(y, p, per)
            e = {'method': method, 'name': name, 'prespecified': method == s['prespecified_calibration'],
                 **calibration_stats(y, p, per), 'bins_off': int(rel['off'].sum()),
                 'bins_off_bootstrap_only': int(rel['off_boot'].sum()),
                 'bins_off_degenerate': int((rel['off_boot'] & rel['degenerate_band']).sum()),
                 'bins_off_null_rate': float(rel.attrs['bins_off_null_rate']),
                 'reliability': rel.to_dict(orient='records')}
            e['target_met'] = bool(e['ece'] <= S.ECE_TARGET and e['bins_off'] == 0)
            e['miscalibration'] = miscalibration_reasons(e)
            nat = res(name).extra_predictions
            e['natural_use_accuracy'] = (S.natural_use(nat['y'], nat['p_right'])['forced_accuracy']
                                         if nat is not None and len(nat) else None)
            plats = []
            for pf, g in pr.assign(platform=pr['trial_id'].map(plat).to_numpy()).groupby('platform'):
                gp = g['person_id'].astype(str).to_numpy()
                plats.append({'platform': pf, **calibration_stats(g['y'].to_numpy(), g['p_right'].to_numpy(float), gp)})
            e['per_platform'] = plats
            if method == 'nested':
                ch = pd.Series([(i.get('calib') or {}).get('nested_choice') for i in _fold_infos(res(name))]).dropna()
                e['nested_choices'] = {str(k): int(v) for k, v in ch.value_counts().sort_index().items()}
            entries.append(e)
        pre = s['prespecified_calibration']
        for e in entries:
            if e['method'] != 'none':
                pair_rows.append(paired_row(e['name'], s['best'], 'calib_vs_none', role=role, method=e['method']))
            if e['method'] not in ('none', pre) and exists(cal_name(role, pre)):
                pair_rows.append(paired_row(e['name'], cal_name(role, pre), 'calib_vs_pre', role=role, method=e['method']))
        out[role] = {'best': s['best'], 'label': s['label'], 'prespecified': pre, 'entries': entries}
    # Holm over each family (all roles together); the log loss is the primary reading of a remedy
    adj = pd.DataFrame(pair_rows)
    if len(adj):
        for _, idx in adj.groupby('family').groups.items():
            adj.loc[idx, 'p_holm_logloss'] = S.holm(adj.loc[idx, 'p_logloss'].to_numpy(float))
            adj.loc[idx, 'family_size'] = int(len(idx))
        adj['verdict_logloss'] = [S.verdict(-r.logloss_diff, -r.logloss_hi, -r.logloss_lo, r.p_holm_logloss,
                                            same=bool(r.identical), better='lower log loss', worse='higher log loss')
                                  for r in adj.itertuples()]
    for role, c in out.items():
        for e in c['entries']:
            for fam, key in (('calib_vs_none', 'vs_none'), ('calib_vs_pre', 'vs_pre')):
                hit = adj[(adj['family'] == fam) & (adj['role'] == role) & (adj['method'] == e['method'])] if len(adj) \
                    else adj
                if len(hit):
                    h = hit.iloc[0]
                    e[key] = {k: (h[k].item() if hasattr(h[k], 'item') else h[k]) for k in
                              ('logloss_diff', 'logloss_lo', 'logloss_hi', 'p_logloss', 'p_holm_logloss',
                               'verdict_logloss', 'brier_diff', 'brier_lo', 'brier_hi', 'p_brier', 'identical',
                               'family_size')}
        c['decision'] = calibration_decision(c)
    return {'roles': out, 'target': {'ece_max': S.ECE_TARGET, 'bins_off_max': 0,
                                     'source': 'docs/09 M4 acceptance (Rec.)'},
            'families': {k: FAMILIES[k] for k in ('calib_vs_none', 'calib_vs_pre')}}


def calibration_decision(c: dict[str, Any]) -> dict[str, Any]:
    """The shipped remedy, from the saved comparisons: the pre-specified one, unless it raises the log loss beyond its
    adjusted CI (then none) or the in-fold choice ('nested', itself free of outer-fold selection) lowers it below the
    pre-specified one's. Other remedies that beat it are reported as candidates, not adopted (choosing among them on
    these folds would be an outer-fold selection). Then the docs/09 target is checked on the shipped one."""
    ent = {e['method']: e for e in c['entries']}
    pre = c['prespecified']
    shipped, reason = pre, f'the pre-specified {pre} remedy'
    if pre in ent and (ent[pre].get('vs_none') or {}).get('verdict_logloss') == 'higher log loss':
        shipped, reason = 'none', f'the pre-specified {pre} raises the log loss beyond its adjusted CI'
    nested = ent.get('nested')
    if nested and (nested.get('vs_pre') or {}).get('verdict_logloss') == 'lower log loss':
        shipped, reason = 'nested', 'the in-fold choice lowers the log loss below the pre-specified remedy'
    cands = [m for m, e in ent.items() if m not in ('none', pre, 'nested')
             and (e.get('vs_pre') or {}).get('verdict_logloss') == 'lower log loss']
    base = ent.get('none') or {}
    sh = ent.get(shipped) or base
    return {'shipped': shipped, 'reason': reason, 'candidates': cands,
            'uncalibrated_miscalibration': base.get('miscalibration', []),
            'shipped_miscalibration': sh.get('miscalibration', []), 'shipped_target_met': bool(sh.get('target_met')),
            'shipped_ece': sh.get('ece'), 'shipped_bins_off': sh.get('bins_off'),
            'lowest_ece_method': min(ent, key=lambda m: (ent[m]['ece'], m)) if ent else None}


# ───────── multi-attempt aggregates ─────────


def aggregate(name: str, base: str, *, n_draws: int = AGG_DRAWS, seed: int = C.SEED, n_boot: int = C.N_BOOT
              ) -> dict[str, Any]:
    """Calibration of the aggregate of k attempts of one person and hand (out-of-fold predictions; the persons with
    at least max(k) trials of each hand, as in cv.k_attempt_curve). The c_k-scaled sum uses the RAW ensemble logits
    (from the uncalibrated run ``base``, the same fits) and c_k from the person's own outer fold (fitted on its
    training persons' cross-fitted logits)."""
    pr = res(name).predictions.copy()
    pr['z'] = M.logit(pr['p_right'].to_numpy(float))
    raw = res(base).predictions.set_index('trial_id')['p_right']
    pr['z_raw'] = M.logit(pr['trial_id'].map(raw).to_numpy(float))
    ck = {}
    for f, p in zip(pr['fold'], pr['params']):
        if f not in ck:
            ck[f] = {int(k): v for k, v in (json.loads(p).get('sum_scale') or {}).items()}
    kmax = max(C.K_ATTEMPTS)
    cnt = pr.groupby(['person_id', 'y']).size().unstack(fill_value=0)
    elig = sorted(cnt.index[(cnt.get(0, 0) >= kmax) & (cnt.get(1, 0) >= kmax)].astype(str))
    rng = np.random.default_rng([seed, 11])
    rows = []
    for k in C.K_ATTEMPTS:
        recs: dict[str, list[np.ndarray]] = {a: [] for a in AGG_METHODS}
        pers, ys = [], []
        for pid in elig:
            for yy in (0, 1):
                g = pr[(pr['person_id'].astype(str) == pid) & (pr['y'] == yy)]
                z, zr, pp = g['z'].to_numpy(), g['z_raw'].to_numpy(float), g['p_right'].to_numpy(float)
                sub = M.k_subsets(z.size, k, n_draws, rng)
                c = ck[int(g['fold'].iloc[0])].get(k, 1.0 / k)
                recs['mean_prob'].append(pp[sub].mean(axis=1))
                recs['mean_logit'].append(M.sigmoid(z[sub].mean(axis=1)))
                recs['sum_logit'].append(M.sigmoid(z[sub].sum(axis=1)))
                recs['sum_logit_calibrated'].append(M.sigmoid(c * zr[sub].sum(axis=1)))
                pers.append(np.full(n_draws, pid))
                ys.append(np.full(n_draws, yy))
        per, yv = np.concatenate(pers), np.concatenate(ys)
        for a in AGG_METHODS:
            p = np.clip(np.concatenate(recs[a]), 1e-12, 1 - 1e-12)
            m = cv.metrics(yv, p)
            conf = np.maximum(p, 1 - p)
            ci = fast_boot(per, yv, p, n_boot=n_boot, seed=seed + k)
            rows.append({'k': k, 'aggregator': a, 'accuracy': m['accuracy'], 'ece': m['ece'], 'log_loss': m['log_loss'],
                         'brier': m['brier'], 'mean_confidence': float(conf.mean()),
                         'overconfidence': float(conf.mean() - m['accuracy']), 'overconfidence_lo': ci['overconfidence'][0],
                         'overconfidence_hi': ci['overconfidence'][1], 'accuracy_lo': ci['accuracy'][0],
                         'accuracy_hi': ci['accuracy'][1], 'log_loss_lo': ci['log_loss'][0], 'log_loss_hi': ci['log_loss'][1],
                         'n_persons': len(elig), 'n_draws_per_cell': n_draws})
    cks = pd.DataFrame([{'fold': f, 'k': k, 'c_k': v} for f, d_ in ck.items() for k, v in d_.items()])
    ck_summary = (cks.groupby('k')['c_k'].agg(['median', 'min', 'max']).reset_index().to_dict(orient='records')
                  if len(cks) else [])
    return {'name': name, 'base': base, 'n_persons': len(elig), 'rows': rows, 'c_k': ck_summary}


#: the aggregator designed for correlated attempts (c_k fitted inside every fold on the training persons' cross-fitted
#: logits); it is recommended whenever the outer folds show it calibrated (``recommend_aggregator``)
PREFERRED_AGG = 'sum_logit_calibrated'


def recommend_aggregator(ag: dict[str, Any], k: int = 3) -> dict[str, Any]:
    """The aggregator of several attempts. By design the c_k-scaled sum (``PREFERRED_AGG``: summing logits treats
    correlated attempts as independent evidence; c_k corrects that, fitted on training persons only), kept when its
    over-confidence CI at ``k`` contains 0 for every role; otherwise the aggregator calibrated for every role with the
    lowest mean log loss at ``k`` (or the lowest log loss overall when none is). The check runs on these outer folds,
    so the decision is exploratory. Also recorded: at which numbers of attempts each aggregator is calibrated for every
    role (supporting evidence across k)."""
    ok, ll, ok_k = {}, {}, {}
    for role, a in ag.items():
        blk = a.get('calibrated') or a.get('uncalibrated')
        df = pd.DataFrame(blk['rows'])
        for _, r in df.iterrows():
            inside = bool(r['overconfidence_lo'] <= 0 <= r['overconfidence_hi'])
            ok_k.setdefault(r['aggregator'], {}).setdefault(int(r['k']), []).append(inside)
            if int(r['k']) == k:
                ok.setdefault(r['aggregator'], []).append(inside)
                ll.setdefault(r['aggregator'], []).append(float(r['log_loss']))
    good = [m for m in AGG_METHODS if m in ok and all(ok[m])]
    if PREFERRED_AGG in good:
        best, how = PREFERRED_AGG, 'the aggregator designed for correlated attempts, calibrated for every role'
    else:
        pool = good or [m for m in AGG_METHODS if m in ll]
        best = min(pool, key=lambda m: (float(np.mean(ll[m])), AGG_METHODS.index(m))) if pool else None
        how = ('the designed aggregator is not calibrated for every role; the lowest mean log loss among the '
               + ('calibrated aggregators' if good else 'aggregators (none is calibrated for every role)'))
    return {'aggregator': best, 'k': k, 'calibrated_for_every_role': good, 'n_compared': len(AGG_METHODS),
            'preferred': PREFERRED_AGG, 'how': how,
            'rule': (f'the c_k-scaled sum of logits when its over-confidence CI at k = {k} contains 0 for every role; '
                     'otherwise the calibrated aggregator with the lowest mean log loss'),
            'calibrated_at_k': {m: [kk for kk, v in sorted(d.items()) if all(v)] for m, d in ok_k.items()},
            'mean_log_loss': {m: float(np.mean(v)) for m, v in ll.items()}}


# ───────── breakdowns ─────────


def breakdown(name: str) -> pd.DataFrame:
    d, _ = data()
    t = d.trials.set_index('trial_id')
    pr = res(name).predictions.copy()
    rows = []
    for col in COND_COLUMNS:
        pr[col] = pr['trial_id'].map(t[col]).astype(str).to_numpy()
        for lev, g in pr.groupby(col, sort=True):
            m = cv.metrics(g['y'], g['p_right'])
            n_p = int(g['person_id'].nunique())
            ci = (fast_boot(g['person_id'].to_numpy(), g['y'].to_numpy(), g['p_right'].to_numpy())['accuracy']
                  if n_p >= S.MIN_PERSONS_CI else (math.nan, math.nan))
            rows.append({'name': name, 'condition': col, 'level': lev, 'n': int(m['n']), 'n_persons': n_p,
                         'accuracy': m['accuracy'], 'accuracy_lo': ci[0], 'accuracy_hi': ci[1],
                         'roc_auc': m['roc_auc'], 'ece': m['ece'], 'log_loss': m['log_loss']})
    return pd.DataFrame(rows)


def grip_transfer() -> dict[str, Any]:
    from stage2.lanes.m3_experiments import GRIP_CODES, sensor_grip
    d, _ = data()
    names = [n for n in GRIP_TRANSFER if exists(n)]
    gt = pd.concat([breakdown(n) for n in names], ignore_index=True) if names else pd.DataFrame()
    if len(gt):
        gt = gt[gt['condition'] == 'orientation'].reset_index(drop=True)
    code_name = {v: k for k, v in GRIP_CODES.items()}
    sg = pd.Series(sensor_grip(d)).map(lambda c: code_name.get(int(c), 'other'))
    conf = pd.crosstab(d.trials['orientation'].astype(str).to_numpy(), sg.to_numpy())
    return {'rows': gt.to_dict(orient='records'), 'names': names,
            'sensor_vs_instructed': {str(i): {str(c): int(conf.loc[i, c]) for c in conf.columns} for i in conf.index}}


# ───────── saliency (integrated gradients, out of fold) ─────────


def input_mirror_sign(x: np.ndarray) -> np.ndarray:
    """Mirror sign of every channel of a per-trial input: the cache channels (17), or the cache channels followed by
    the body channels (``m3_body.EXT_CHANNELS``, 21)."""
    from stage2.lanes import m3_body
    from stage2.standardize import MIRROR_SIGN
    if x.shape[1] == len(m3_body.EXT_CHANNELS):
        return np.array([m3_body.EXT_SIGN[c] for c in m3_body.EXT_CHANNELS], dtype=np.float32)
    if x.shape[1] == len(MIRROR_SIGN):
        return MIRROR_SIGN
    raise ValueError(f'no mirror signs for an input with {x.shape[1]} channels')


def _share_first_half(absA: np.ndarray, fh: np.ndarray, idx: list[int], persons: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-person sums of |IG| in the first half and in total over channels ``idx``: (first, total), each (P,)."""
    first = (absA[:, idx] * fh[:, None, :]).sum(axis=(1, 2))
    tot = absA[:, idx].sum(axis=(1, 2))
    uniq, inv = np.unique(persons, return_inverse=True)
    return np.bincount(inv, weights=first, minlength=uniq.size), np.bincount(inv, weights=tot, minlength=uniq.size)


def saliency(name: str) -> dict[str, Any]:
    """IG of the deep finalist's network: for every outer fold, the fold's cached ensemble (trained on the training
    persons only; the folds are rebuilt exactly as cv.run_experiment builds them) attributes the logit of each held-out
    trial and of its mirror image. The calibrators are monotone maps of this logit."""
    from stage2.lanes import m3_body
    from stage2.standardize import DEVICE_CHANNELS
    spec = spec_of(name)
    fam = E.make_family(spec)
    d, _ = data()
    X, _ = inputs(spec.x)
    g = d.groups.astype(str)
    y = d.trials['y_right'].astype(int).to_numpy()
    IG = IGm = Z = Z0 = None
    for s in cv.loso_splits(g):
        tr = s.train[cv.cap_rows(g[s.train], y[s.train], C.CAP_PER_PERSON, C.SEED)]
        Xtr = {k: v[tr].copy() for k, v in X.items()}
        st = M.cached_fit(fam, Xtr, y[tr].copy(), g[tr].copy(), cv.inner_splits(g[tr], fam.n_inner), C.SEED + s.fold,
                          cache_only=True)
        Xte = {k: v[s.test] for k, v in X.items()}
        Xm = dict(Xte)
        Xm['x'] = Xte['x'] * input_mirror_sign(Xte['x']).reshape((1, -1) + (1,) * (Xte['x'].ndim - 2))
        ig, igm = fam.integrated_gradients(st, Xte), fam.integrated_gradients(st, Xm)
        if IG is None:
            IG = np.zeros((len(y),) + ig.shape[1:])
            IGm, Z, Z0 = np.zeros_like(IG), np.zeros(len(y)), np.zeros(len(y))
        IG[s.test], IGm[s.test] = ig, igm
        Z[s.test] = fam.logits(st, Xte)
        Z0[s.test] = fam.baseline_logit(st, Xte)
    chans = list(fam.channel_names() if hasattr(fam, 'channel_names')
                 else M.channel_names(spec.params.get('channels', 'grav')))
    if IG.shape[1] > len(chans):
        chans += ['mask']
    flat, flatm = IG.reshape(len(y), -1), IGm.reshape(len(y), -1)
    corr = float(np.corrcoef(flat.ravel(), -flatm.ravel())[0, 1])
    rel_err = float(np.linalg.norm(flat + flatm) / max(np.linalg.norm(flat), 1e-12))
    dz = Z - Z0                                                           # IG completeness: sum IG = z(x) - z(baseline)
    complete = float(np.median(np.abs(IG.sum(axis=(1, 2)) - dz) / np.maximum(np.abs(dz), 1e-6)))
    toward = IG * (2 * y - 1)[:, None, None]                              # attribution toward the TRUE hand
    T = IG.shape[-1]
    window = spec.params.get('window', 'approach')
    dur_s = d.trials['duration_ms'].to_numpy(float) / 1000.0
    if window == 'fixed':                       # the first half of each trial's own approach: onset .. onset + dur / 2
        on = int(round(C.FIXED_PRE_S * C.FS_HZ))
        ti = np.arange(T)[None, :]
        fh = (ti >= on) & (ti < on + np.round(dur_s[:, None] * C.FS_HZ / 2))
    else:                                       # time-normalised: the first half of the samples
        fh = np.broadcast_to(np.arange(T)[None, :] < T // 2, (len(y), T))
    fh = fh.astype(float)
    absA = np.abs(IG)
    sig = [i for i, c in enumerate(chans) if c != 'mask']
    odd = [i for i, c in enumerate(chans) if c in m3_body.INVARIANT_ODD]           # grip-invariant, odd under the mirror
    even = [i for i in sig if i not in odd]
    yaw = [i for i, c in enumerate(chans) if c.startswith('yaw')]
    device = [i for i, c in enumerate(chans) if c in DEVICE_CHANNELS]
    tot = absA[:, sig].sum()
    persons = g
    # first-half concentration of the odd channels against the even ones (the generic time profile), person bootstrap
    f_o, t_o = _share_first_half(absA, fh, odd, persons)
    f_e, t_e = _share_first_half(absA, fh, even, persons)
    rng = np.random.default_rng([C.SEED, 29])
    n_p = f_o.size
    W = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=C.N_BOOT).astype(float)
    boot = (W @ f_o) / (W @ t_o) - (W @ f_e) / (W @ t_e)
    fh_odd, fh_even = float(f_o.sum() / t_o.sum()), float(f_e.sum() / t_e.sum())
    np.savez_compressed(tab('m3_saliency.npz'), ig=IG.astype(np.float32), ig_mirror=IGm.astype(np.float32),
                        z=Z.astype(np.float32), y=y.astype(np.int8), channels=np.array(chans))
    prof = absA.mean(axis=0)                                              # (C, T) mean |IG| per channel and time
    by_channel = {c: {'share_of_all': float(absA[:, i].sum() / tot),
                      'first_half_share': float((absA[:, i] * fh).sum() / max(absA[:, i].sum(), 1e-12)),
                      'peak_time_frac': float(np.argmax(prof[i]) / max(T - 1, 1)),
                      'toward_true_hand': float(toward[:, i].sum() / max(absA[:, i].sum(), 1e-12))}
                  for i, c in enumerate(chans) if c != 'mask'}
    agree = float(np.mean(np.sign(toward.sum(axis=(1, 2))) == np.sign(Z * (2 * y - 1))))
    return {'name': name, 'label': spec.label, 'window': window, 'channels': chans,
            'n_trials': int(len(y)), 'n_persons': int(n_p), 'T': int(T), 'mirror_corr': corr, 'mirror_rel_err': rel_err,
            'median_duration_s': float(np.median(dur_s)),
            'first_half_definition': ('onset to onset + half of each trial\'s duration' if window == 'fixed'
                                      else 'the first half of the time-normalised window'),
            'completeness_median_rel_err': complete, 'ig_steps': 32,
            'odd_channels': [chans[i] for i in odd], 'even_channels': [chans[i] for i in even],
            'share_odd': float(absA[:, odd].sum() / tot), 'share_odd_uniform': len(odd) / len(sig),
            'share_yaw': float(absA[:, yaw].sum() / tot) if yaw else 0.0, 'yaw_channels': [chans[i] for i in yaw],
            'share_device': float(absA[:, device].sum() / tot) if device else 0.0,
            'share_first_half_all': float((absA[:, sig] * fh[:, None, :]).sum() / tot),
            'share_first_half_odd': fh_odd, 'share_first_half_even': fh_even,
            'first_half_odd_minus_even': fh_odd - fh_even,
            'first_half_odd_minus_even_lo': float(np.quantile(boot, 0.025)),
            'first_half_odd_minus_even_hi': float(np.quantile(boot, 0.975)),
            'by_channel': by_channel, 'ig_sign_agrees_with_logit': agree,
            'toward_true_hand': {h: toward[y == v].mean(axis=0).tolist() for h, v in (('left', 0), ('right', 1))},
            'signed_mean': {h: IG[y == v].mean(axis=0).tolist() for h, v in (('left', 0), ('right', 1))},
            'time_profile_odd': {h: absA[y == v][:, odd].sum(axis=1).mean(axis=0).tolist()
                                 for h, v in (('left', 0), ('right', 1))},
            'time_profile_even': {h: absA[y == v][:, even].sum(axis=1).mean(axis=0).tolist()
                                  for h, v in (('left', 0), ('right', 1))}}


# ───────── the time-frequency sign physics ─────────


EXAMPLE_VARIANT, EXAMPLE_SPEED = 'standard', 'natural'      # the condition the illustrated trial is drawn from


def tf_example() -> dict[str, Any]:
    from stage2.lanes import m3_tf
    from stage2.standardize import CHIRAL_CHANNELS
    d, _ = data()
    k = d.channels.index('yaw_rate')
    X = d.approach[:, [k]]
    planes, planes_m = m3_tf.stft_planes(X), m3_tf.stft_planes(-X)
    mag = np.hypot(planes[:, :, 0], planes[:, :, 1])
    mag_m = np.hypot(planes_m[:, :, 0], planes_m[:, :, 1])
    odd = [d.channels.index(c) for c in CHIRAL_CHANNELS]                     # every chiral channel, every trial
    Xo = d.approach[:, odd]
    so, som = m3_tf.stft_planes(Xo), m3_tf.stft_planes(-Xo)
    co, com = m3_tf.cwt_planes(Xo), m3_tf.cwt_planes(-Xo)
    y = d.y
    fhy = d.trials['first_half_yaw_deg'].to_numpy(float)
    right = np.flatnonzero(y == 1)
    typical = right[(d.trials['variant'].iloc[right] == EXAMPLE_VARIANT).to_numpy()
                    & (d.trials['speed'].iloc[right] == EXAMPLE_SPEED).to_numpy()]
    pool = typical if typical.size else right                                 # illustration only: a typical movement
    i = int(pool[np.argmin(np.abs(fhy[pool] - np.median(fhy[pool])))])        # the median right-hand trial
    return {'channel': 'yaw_rate', 'chiral_channels': list(CHIRAL_CHANNELS),
            'trial_key': str(d.trials['trial_key'].iloc[i]), 'first_half_yaw_deg': float(fhy[i]),
            'example_condition': f'{EXAMPLE_VARIANT} variant, {EXAMPLE_SPEED} speed' if typical.size else 'any',
            'duration_ms': float(d.trials['duration_ms'].iloc[i]),
            'x': X[i, 0].astype(float).tolist(),
            'stft_mag': mag[i, 0].tolist(), 'stft_mag_mirror': mag_m[i, 0].tolist(),
            'stft_real': planes[i, 0, 0].tolist(), 'stft_real_mirror': planes_m[i, 0, 0].tolist(),
            'max_abs_mag_diff_stft': float(np.max(np.abs(np.hypot(so[:, :, 0], so[:, :, 1])
                                                         - np.hypot(som[:, :, 0], som[:, :, 1])))),
            'max_abs_mag_diff_cwt': float(np.max(np.abs(np.hypot(co[:, :, 0], co[:, :, 1])
                                                        - np.hypot(com[:, :, 0], com[:, :, 1])))),
            'max_abs_signed_sum_stft': float(np.max(np.abs(so + som))),
            'max_abs_stft_mag': float(np.max(mag)), 'n_trials': int(len(y)),
            'stft_nperseg': m3_tf.STFT_NPERSEG, 'stft_hop': m3_tf.STFT_HOP}


# ───────── label-noise sensitivity ─────────


def label_noise(names: dict[str, str]) -> dict[str, Any]:
    """Accuracy of the shipped finalists without the trials of the blocks that show the hand-swap signature (found
    from the data by ``reference.swap_blocks``; labels are never changed), and on those trials alone."""
    from stage2.scripts.reference import swap_blocks
    d, _ = data()
    blocks = swap_blocks(d)
    bset = {(str(s), int(b)) for s, b in blocks}
    t = d.trials
    bi = pd.to_numeric(t['block_index'], errors='coerce').fillna(-1).astype(int)
    on_block = np.array([(s, int(b)) in bset for s, b in zip(t['session_label'].astype(str), bi)], dtype=bool)
    swap_ids = set(t['trial_id'][on_block].astype(str))
    out = {'blocks': [{'session_label': s, 'block_index': int(b)} for s, b in blocks], 'n_trials': len(swap_ids),
           'n_persons': int(t[t['trial_id'].astype(str).isin(swap_ids)]['person_id'].nunique()), 'roles': {}}
    for role, name in names.items():
        pr = res(name).predictions
        on = pr['trial_id'].astype(str).isin(swap_ids).to_numpy()
        keep = pr[~on]
        ci = fast_boot(keep['person_id'].to_numpy(), keep['y'].to_numpy(), keep['p_right'].to_numpy())['accuracy']
        ok = (pr['p_right'].to_numpy(float) >= 0.5) == (pr['y'].to_numpy() == 1)
        out['roles'][role] = {'name': name, 'accuracy': float(ok.mean()), 'accuracy_without': float(ok[~on].mean()),
                              'accuracy_without_lo': ci[0], 'accuracy_without_hi': ci[1],
                              'accuracy_on_swap': float(ok[on].mean()) if on.any() else None,
                              'n_without': int((~on).sum())}
    return out


# ───────── derive (all of the above) ─────────

DERIVED = ('m3_summary', 'm3_paired', 'm3_calibration', 'm3_aggregate', 'm3_breakdown', 'm3_grip_transfer',
           'm3_saliency', 'm3_tf_example', 'm3_label_noise')


def comparators() -> list[str]:
    return sorted(set(ENGINEERED) | {b for _, b, _, _ in EQUAL_INPUT})


def inputs_signature(sel: dict[str, Any]) -> str:
    """Digest of everything the derived files are computed from besides the cache: the content of every saved m3 result,
    of the comparators and the role selection. A derived file whose signature differs is recomputed, so an interrupted
    derive or a re-run experiment never leaves a figure input behind (the cache identity alone would not see it)."""
    import hashlib
    h = hashlib.sha256()
    paths = [L.result_path(n) for n in ALL_NAMES()] + [C.RES_DIR / f'{c}.json' for c in comparators()]
    for p in sorted(paths, key=lambda q: q.name):
        if p.exists():
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    h.update(json.dumps({r: s['best'] for r, s in sel.items()}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _clean(v: Any) -> Any:
    """JSON-safe copy: NaN / inf become None (the derived files are strict JSON), numpy scalars become Python ones."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, 'item') and not isinstance(v, (str, bytes)) and getattr(v, 'ndim', 0) == 0:
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _saved_sig(path: Any) -> str | None:
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('inputs_sig')
    except (OSError, ValueError):
        return None


def derive(force: bool = False, only: list[str] | None = None) -> None:
    _RES.clear()                                     # results may have been re-saved earlier in this process
    if not all(exists(n) for n in E.BASE):
        print('m3: derived files skipped (base results missing)', flush=True)
        return
    sel = _selection_file()
    sig = inputs_signature(sel)

    def need(n: str) -> bool:
        p = tab(f'{n}.json')
        return selected(n, only) and (force or not file_current(p) or _saved_sig(p) != sig)

    def put(n: str, payload: dict[str, Any]) -> None:
        L.write_json(n, _clean({**payload, 'inputs_sig': sig}))

    if need('m3_summary'):
        put('m3_summary', summary(sel))
    if need('m3_calibration'):
        put('m3_calibration', calibration(sel))
    cal = L.read_json('m3_calibration')['roles'] if tab('m3_calibration.json').exists() else None
    names = {r: shipped_name(r, sel, cal) for r in sel}
    if need('m3_paired'):
        put('m3_paired', {'rows': paired(sel, cal), 'families': FAMILIES, 'alpha': S.ALPHA})
    if need('m3_aggregate'):
        out = {role: {'uncalibrated': aggregate(s['best'], s['best']),
                      'calibrated': aggregate(names[role], s['best']) if exists(names[role]) else None}
               for role, s in sel.items()}
        put('m3_aggregate', {'roles': out, 'recommended': recommend_aggregator(out)})
    if need('m3_breakdown'):
        bd = pd.concat([breakdown(names[r]).assign(role=r) for r in sel], ignore_index=True)
        put('m3_breakdown', {'rows': bd.to_dict(orient='records'), 'names': names,
                             'min_persons_ci': S.MIN_PERSONS_CI})
    if need('m3_grip_transfer'):
        put('m3_grip_transfer', grip_transfer())
    if 'deep' in sel and need('m3_saliency'):
        put('m3_saliency', saliency(sel['deep']['best']))
    if need('m3_tf_example'):
        put('m3_tf_example', tf_example())
    if need('m3_label_noise'):
        put('m3_label_noise', label_noise(names))
    write_finalists(sel, cal)


def write_finalists(sel: dict[str, Any], cal: dict[str, Any] | None) -> None:
    from stage2.lanes import m3_finalists as FN
    rows = []
    for role in ROLES:
        if role not in sel:
            continue
        s = sel[role]
        spec = E.BY_NAME[s['best']]
        dec = ((cal or {}).get(role) or {}).get('decision') or {}
        method = dec.get('shipped') or s['prespecified_calibration']
        name = shipped_name(role, sel, cal)
        if not exists(name):
            continue
        r = res(name)
        gr = s.get('grip_rule') or {}
        notes = [f"{ROLE_LABELS[role]}: {s['label']} ({s['best']}), chosen on the outer folds among {s['n_configs']} "
                 f"configurations of its role (exploratory; rule: {GRIP_RULE if role == 'deep' else 'highest LOSO accuracy, then log loss'})."]
        if s.get('post_hoc'):
            notes.append('Post hoc: its input was designed after outer-fold results had been seen, so its accuracy is '
                         'optimistic by an amount no nested estimate removes.')
        notes.append(f"Calibration: {M.CALIB_LABELS.get(method, method)} ({dec.get('reason', 'pre-specified, docs/09 D19')}), "
                     'fitted inside every fold on cross-fitted training logits.')
        if dec and not dec.get('shipped_target_met', True):
            notes.append('Does NOT meet the docs/09 calibration target (' + '; '.join(dec.get('shipped_miscalibration') or
                         ['ECE above the target']) + ').')
        if gr.get('replaced'):
            notes.append(f"Grip rule: the most accurate configuration, {gr['most_accurate']}, uses grip-dependent "
                         'device-frame channels and is not detectably more accurate, so this grip-invariant one replaces it.')
        elif s.get('grip_dependent'):
            notes.append('Caution: grip-dependent device-frame channels; valid only for grips seen in training.')
        notes.append(f'LOSO accuracy {r.metrics["accuracy"]:.3f}; training cost {cost_cpu_min(r):.1f} CPU-min per LOSO '
                     'run (one thread per fit; fits cached in analysis/data/m3_fit_cache, keyed by data, code and runtime).')
        params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in E.make_family(spec).config().items()}
        rows.append({'name': name, 'results_json': f'results/{name}.json', 'module': 'stage2.lanes.m3_finalists',
                     'build_x': FN.BUILD_X_OF_INPUT[spec.x], 'factory': FN.FACTORY_OF_FAMILY[spec.family],
                     'params': {**params, 'calib': method}, 'n_configs_tried': int(s['n_configs']),
                     'notes': ' '.join(notes), 'grip_dependent': bool(s.get('grip_dependent', False)),
                     'post_hoc': bool(s.get('post_hoc', False)), 'role': role, 'base_experiment': s['best'],
                     'input': spec.x, 'calibration': method,
                     'calibration_target_met': bool(dec.get('shipped_target_met', True)),
                     'accuracy': r.metrics['accuracy'], 'accuracy_ci95': list(r.ci['accuracy'])})
    if rows:
        L.write_finalists(rows)


# ───────────────────────────── tables ─────────────────────────────


def _ci(lo: Any, hi: Any, scale: float = 100, fmt: str = '.1f') -> str:
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or (isinstance(hi, float) and math.isnan(hi)):
        return '--'
    return f'[{scale * lo:{fmt}}, {scale * hi:{fmt}}]'


def rebuild_registry() -> None:
    names = [n for n in ALL_NAMES() if exists(n)]
    if names:
        pd.DataFrame([L.load_result(n).summary_row() for n in sorted(names)]).to_csv(L.registry, index=False)


def _sel_cell(r: Any) -> str:
    s = str(r['selection'])
    return s + (', post hoc' if r.get('post_hoc') else '')


def tables(only: list[str] | None = None) -> None:
    rebuild_registry()
    if selected('m3_experiments', only):
        experiments_markdown()
    Sm = L.read_json('m3_summary')
    ex = pd.DataFrame(Sm['experiments'])
    grp_order = {'arch': 0, 'treat': 1, 'treat_body': 2, 'grip': 3, 'nested': 4, 'mr': 5, 'tf': 6, 'calib': 7}
    if selected('m3_models', only):
        t = ex[ex['group'] != 'calib'].copy()
        t['_o'] = t['group'].map(grp_order)
        t = t.sort_values(['_o', 'name']).reset_index(drop=True)
        out = pd.DataFrame({'Model': t['label'], 'Accuracy': t['accuracy'],
                            '95% CI': [_ci(a, b) for a, b in zip(t['accuracy_lo'], t['accuracy_hi'])],
                            'AUC': t['roc_auc'], 'Log loss': t['log_loss'], 'ECE': t['ece'],
                            'Natural use': t['natural_use_answered_accuracy'], 'k = 3': t.get('acc_k3'),
                            'CPU min': t['cpu_min'], 'Configs': t['n_configs_tried'].astype(int),
                            'Selection': [_sel_cell(r) for _, r in t.iterrows()]})
        L.export_table(out, 'm3_models', caption=(
            'Lane m3 models, leave-one-person-out (95\\,\\% person-cluster bootstrap CI). Natural use: accuracy on the '
            'answered first free-choice trials (one per person, all right-handed: the recall of the right hand; a '
            'P(right) of exactly 0.5 is an abstention). $k = 3$: per-person balanced accuracy when the P(right) of 3 '
            'attempts are averaged. CPU min: training cost of one LOSO run (a nested choice: all its candidates). '
            'Configs: configurations compared on the outer folds before keeping this one (1 = pre-specified, a '
            'comparison or a check). Selection: how the configuration came to be; post hoc = designed after outer-fold '
            'results had been seen.'),
            column_formats={'Accuracy': '.1%', 'AUC': '.3f', 'Log loss': '.3f', 'ECE': '.3f', 'Natural use': '.0%',
                            'k = 3': '.1%', 'CPU min': '.1f', 'Configs': 'd'}, align='lrlrrrrrrrl')
    pp = pd.DataFrame(L.read_json('m3_paired')['rows'])

    def pair_table(df: pd.DataFrame, name: str, caption: str) -> None:
        out = pd.DataFrame({'Family': df['family'], 'Comparison': df['option'],
                            'Against': df['baseline_label'], 'Accuracy': df['accuracy'],
                            'Diff.': df['diff'], '95% CI': [_ci(a, b) for a, b in zip(df['diff_lo'], df['diff_hi'])],
                            'p': df['p'], 'p (Holm)': df['p_holm'], 'Log-loss diff.': df['logloss_diff'],
                            'Verdict': df['verdict']})
        L.export_table(out, name, caption=caption,
                       column_formats={'Accuracy': '.1%', 'Diff.': '+.1%', 'p': '.3f', 'p (Holm)': '.3f',
                                       'Log-loss diff.': '+.3f'}, align='lllrrlrrrl')

    if selected('m3_treatments', only) and len(pp):
        df = pp[pp['family'].isin(['treatment', 'treatment_body', 'nested', 'physics_demo'])]
        pair_table(df, 'm3_treatments', (
            'Data treatment and model choices of lane m3, each paired against its default on the same trials: mean '
            'per-person accuracy difference with its paired person-bootstrap 95\\,\\% CI, exact sign-flip p over '
            'persons and the p Holm-adjusted within the family. Verdict: better / worse only when the adjusted p is '
            'below 0.05 and the CI excludes 0; identical: the same P(right) on every trial.'))
    if selected('m3_vs_engineered', only) and len(pp):
        df = pp[pp['family'].isin(['equal_input', 'engineered'])]
        pair_table(df, 'm3_vs_engineered', (
            'Lane-m3 models against linear and engineered-feature models on the same trials (mean per-person accuracy '
            'difference, paired person-bootstrap 95\\,\\% CI, sign-flip p, Holm-adjusted within the family). Equal '
            'input: deep and linear models on matched information; engineered: the shipped finalists and the nested '
            'deep estimates against the reference, lane m1 and lane m2.'))
    if selected('m3_grip', only) and len(pp):
        df = pp[pp['family'] == 'grip']
        pair_table(df, 'm3_grip', (
            'Grip transfer: upright-only against all-grip training and the body frame against the gravity frame, on all '
            'trials and on the rotated (never seen by upright-only training) and upright grips (instructed orientation, '
            'used only to stratify). Mean per-person difference, paired person-bootstrap 95\\,\\% CI, Holm-adjusted p.'))
    if selected('m3_calibration', only):
        calj = L.read_json('m3_calibration')['roles']
        rows, prow = [], []
        for role, c in calj.items():
            for e in c['entries']:
                vs = e.get('vs_none') or {}
                rows.append({'Model': ROLE_LABELS[role], 'Calibration': M.CALIB_LABELS[e['method']] + (
                    ' (pre-specified)' if e['prespecified'] else '') + (' (shipped)' if e['method'] == c['decision'][
                        'shipped'] else ''),
                    'Accuracy': e['accuracy'], 'ECE': e['ece'], 'ECE null 95%': e['ece_null_q95'],
                    'Slope': e['cal_slope'], 'Slope 95% CI': _ci(e.get('cal_slope_lo'), e.get('cal_slope_hi'), 1, '.2f'),
                    'Brier': e['brier'], 'Log loss': e['log_loss'],
                    'LL diff. vs none': vs.get('logloss_diff'), 'p (Holm)': vs.get('p_holm_logloss'),
                    'Target': 'met' if e['target_met'] else 'missed'})
                if e['method'] in ('none', c['decision']['shipped']):
                    for pl in e['per_platform']:
                        prow.append({'Model': ROLE_LABELS[role], 'Calibration': M.CALIB_LABELS[e['method']],
                                     'Platform': pl['platform'], 'Trials': pl['n'], 'Persons': pl['n_persons'],
                                     'Accuracy': pl['accuracy'], 'ECE': pl['ece'], 'ECE null 95%': pl['ece_null_q95'],
                                     'Slope': pl['cal_slope'],
                                     'Slope 95% CI': _ci(pl.get('cal_slope_lo'), pl.get('cal_slope_hi'), 1, '.2f'),
                                     'Brier': pl['brier'], 'Log loss': pl['log_loss']})
        L.export_table(pd.DataFrame(rows), 'm3_calibration', caption=(
            'Calibration of each role\'s best m3 model before and after the remedies, each fitted inside every outer '
            'fold on the cross-fitted logits of the training persons ("chosen in fold": the calibrator picked inside '
            f'the fold). ECE ({C.ECE_BINS} equal-mass bins) with the 95th percentile of its distribution under perfect calibration; '
            'calibration slope (1 = calibrated, < 1 over-confident) with its person-bootstrap CI; log-loss difference '
            'against the uncalibrated model per person, Holm-adjusted. Target: docs/09 M4 (ECE at most 0.05 and no '
            'reliability bin off by more than its CI).'),
            column_formats={'Accuracy': '.1%', 'ECE': '.3f', 'ECE null 95%': '.3f', 'Slope': '.2f', 'Brier': '.3f',
                            'Log loss': '.3f', 'LL diff. vs none': '+.4f', 'p (Holm)': '.3f'},
            align='llrrrrlrrrrl')
        L.export_table(pd.DataFrame(prow), 'm3_calibration_platform',
                       caption=('Calibration per platform, uncalibrated and shipped (leave-one-person-out; ECE with its '
                                'perfect-calibration 95th percentile; slope with its person-bootstrap CI).'),
                       column_formats={'Trials': 'd', 'Persons': 'd', 'Accuracy': '.1%', 'ECE': '.3f',
                                       'ECE null 95%': '.3f', 'Slope': '.2f', 'Brier': '.3f', 'Log loss': '.3f'},
                       align='lllrrrrrrlrr')
    if selected('m3_breakdown', only):
        bdj = L.read_json('m3_breakdown')
        bd = pd.DataFrame(bdj['rows'])
        b = bd[bd['role'] == 'deep'] if (bd['role'] == 'deep').any() else bd
        out = pd.DataFrame({'Condition': b['condition'], 'Level': b['level'], 'Trials': b['n'], 'Persons': b['n_persons'],
                            'Accuracy': b['accuracy'], '95% CI': [_ci(a, c) for a, c in zip(b['accuracy_lo'],
                                                                                              b['accuracy_hi'])],
                            'AUC': b['roc_auc']})
        L.export_table(out, 'm3_breakdown', caption=(
            f"Shipped m3 deep finalist ({bdj['names'].get('deep', '')}) by platform and instructed condition (conditions "
            'stratify the results only; they are never inputs; 95\\,\\% person-cluster bootstrap CI within each level, '
            f"none for levels with fewer than {bdj['min_persons_ci']} persons)."),
            column_formats={'Trials': 'd', 'Persons': 'd', 'Accuracy': '.1%', 'AUC': '.3f'}, align='llrrrlr')
    if selected('m3_aggregate', only):
        agj = L.read_json('m3_aggregate')
        rows = []
        for role, a in agj['roles'].items():
            for which in ('uncalibrated', 'calibrated'):
                if not a.get(which):
                    continue
                for r in a[which]['rows']:
                    rows.append({'Model': ROLE_LABELS[role], 'Single-trial calibration': which, 'k': r['k'],
                                 'Aggregator': AGG_LABELS[r['aggregator']], 'Accuracy': r['accuracy'],
                                 'Log loss': r['log_loss'], 'Over-conf.': r['overconfidence'],
                                 '95% CI': _ci(r['overconfidence_lo'], r['overconfidence_hi'], 1, '+.3f')})
        df = pd.DataFrame(rows)
        L.export_table(df, 'm3_aggregate_all', tex=False)
        k3 = df[df['k'] == agj['recommended']['k']].drop(columns=['k'])
        L.export_table(k3, 'm3_aggregate', caption=(
            f"Aggregating $k = {agj['recommended']['k']}$ attempts of one person and hand (out of fold; over-confidence = "
            'mean confidence minus accuracy, person-bootstrap 95\\,\\% CI). Sum of logits treats the attempts as '
            'independent; $c_k$ is fitted on the training persons of each fold. The recommended aggregator was chosen '
            f"among {agj['recommended']['n_compared']} on these outer folds (exploratory)."),
            column_formats={'Accuracy': '.1%', 'Log loss': '.3f', 'Over-conf.': '+.3f'}, align='lllrrrl')


# ───────────────────────────── numbers ─────────────────────────────


def _k(*parts: Any) -> str:
    """A numbers key from parts (lower case; anything that is not a letter or a digit becomes '_')."""
    import re
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', '_'.join(str(p) for p in parts if p != '').lower())).strip('_')


def comparison_key(r: dict[str, Any]) -> str:
    sub = '' if r.get('subset', 'all') == 'all' else r['subset']
    return _k(r['family'], str(r['name']).replace('m3_', ''), 'vs', str(r['baseline']).replace('m3_', ''), sub)


def numbers() -> None:
    _RES.clear()
    E_ = RX.numbers_entry
    Sm = L.read_json('m3_summary')
    ex = {r['name']: r for r in Sm['experiments']}
    sel = L.read_json('m3_selection')['roles']
    calj = L.read_json('m3_calibration')['roles']
    src_s, src_sel, src_p = 'tables/m3/m3_summary.json', 'tables/m3/m3_selection.json', 'tables/m3/m3_paired.json'
    src_c, src_a, src_b = 'tables/m3/m3_calibration.json', 'tables/m3/m3_aggregate.json', 'tables/m3/m3_breakdown.json'
    n: dict[str, Any] = {}

    def put(key: str, v: Any, fmt: str, src: str, desc: str) -> None:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return
        if isinstance(v, (np.bool_, bool)) and fmt != '':
            v = int(v)
        n[_k(key)] = E_(v, fmt, src, desc)

    any_r = next(iter(ex.values()))
    put('n_trials', any_r['n_trials'], 'd', src_s, 'primary trials scored by every m3 experiment')
    put('n_persons', any_r['n_persons'], 'd', src_s, 'persons = leave-one-person-out folds')
    base_cpu = sum(float(r['cpu_min'] or 0) for r in ex.values() if r['name'] in E.BASE)
    put('base_cpu_min', base_cpu, '.0f', src_s, 'training CPU-minutes of every base experiment (one LOSO run each)')
    from stage2.lanes.m3_pipeline import THREADS          # the lane's budget (not the caller's setting): idempotent
    put('base_wall_min_workers', base_cpu / THREADS, '.0f', src_s,
        f'expected wall-clock minutes of the base experiments on {THREADS} one-thread workers (empty fit cache)')
    put('n_workers', THREADS, 'd', src_s, 'default worker processes of the compute step (the lane thread budget)')
    put('n_experiments', len([r for r in ex.values() if r['group'] != 'calib']), 'd', src_s,
        'm3 experiments (base and nested, without the calibration variants)')
    put('n_natural_use', any_r['natural_use_n'], 'd', src_s, 'natural-use trials (one first free-choice trial per person)')
    put('k_n_persons', any_r['k_n_persons'], 'd', src_s, 'persons in the k-attempt curve (enough trials of each hand)')
    put('alpha', S.ALPHA, '.2f', src_p, 'significance level of the Holm-adjusted tests')
    put('ece_target', S.ECE_TARGET, '.2f', src_c, 'docs/09 M4 calibration target (ECE at most)')
    put('ece_bins', C.ECE_BINS, 'd', src_c, 'equal-mass bins of the ECE and the reliability diagrams')
    put('min_persons_ci', S.MIN_PERSONS_CI, 'd', src_b, 'strata with fewer persons get no bootstrap CI')
    put('bins_off_by_chance', 1 - 0.95 ** C.ECE_BINS, '.0%', src_c,
        'chance that at least one of the bins of a perfectly calibrated model falls outside its 95% band with one '
        'band per bin (independent bins)')

    def exp_numbers(prefix: str, r: dict[str, Any], what: str, full: bool = False) -> None:
        for k, fmt in (('accuracy', '.1%'), ('roc_auc', '.3f'), ('log_loss', '.3f'), ('brier', '.3f')):
            put(f'{prefix}_{k}', r.get(k), fmt, src_s, f'{k}, {what}')
            put(f'{prefix}_{k}_lo', r.get(f'{k}_lo'), fmt, src_s, f'{k} 95% CI low, {what}')
            put(f'{prefix}_{k}_hi', r.get(f'{k}_hi'), fmt, src_s, f'{k} 95% CI high, {what}')
        put(f'{prefix}_label', r.get('label'), '', src_s, f'model label, {what}')
        put(f'{prefix}_ece', r.get('ece'), '.3f', src_s, f'ECE ({C.ECE_BINS} equal-mass bins), {what}')
        put(f'{prefix}_ece_null_q95', r.get('ece_null_q95'), '.3f', src_s, f'ECE under perfect calibration, 95th pct, {what}')
        put(f'{prefix}_nat_acc', r.get('natural_use_answered_accuracy'), '.0%', src_s,
            f'natural-use accuracy over answered trials (right-hand recall), {what}')
        put(f'{prefix}_nat_lo', r.get('natural_use_answered_lo'), '.0%', src_s, f'natural use, exact 95% CI low, {what}')
        put(f'{prefix}_nat_hi', r.get('natural_use_answered_hi'), '.0%', src_s, f'natural use, exact 95% CI high, {what}')
        put(f'{prefix}_nat_coverage', r.get('natural_use_coverage'), '.0%', src_s, f'natural use answered (P != 0.5), {what}')
        put(f'{prefix}_nat_correct', r.get('natural_use_n_correct'), 'd', src_s, f'natural-use trials right, {what}')
        put(f'{prefix}_nat_half_credit', r.get('natural_use_half_credit_accuracy'), '.0%', src_s,
            f'natural use, P = 0.5 counted as half right, {what}')
        for k in (C.K_ATTEMPTS if full else (3,)):
            put(f'{prefix}_acc_k{k}', r.get(f'acc_k{k}'), '.1%', src_s, f'k={k} attempts (mean P), {what}')
            put(f'{prefix}_acc_k{k}_lo', r.get(f'acc_k{k}_lo'), '.1%', src_s, f'k={k} 95% CI low, {what}')
            put(f'{prefix}_acc_k{k}_hi', r.get(f'acc_k{k}_hi'), '.1%', src_s, f'k={k} 95% CI high, {what}')
        put(f'{prefix}_cpu_min', r.get('cpu_min'), '.1f', src_s, f'training CPU-minutes per LOSO run, {what}')
        put(f'{prefix}_n_configs', r.get('n_configs_tried'), 'd', src_s, f'configurations compared on the outer folds, {what}')

    for name, r in ex.items():
        exp_numbers(name.replace('m3_', 'x_'), r, f"{r['label']} ({name})", full=name in E.NESTED_SPECS)
        if r.get('nested_choice_counts'):
            for c, v in r['nested_choice_counts'].items():
                put(f"{name.replace('m3_', 'x_')}_chose_{c.replace('m3_', '')}", v, 'd', src_s,
                    f'folds in which {name} chose {c}')
            put(f"{name.replace('m3_', 'x_')}_n_candidates", r.get('nested_n_candidates'), 'd', src_s,
                f'candidates of {name}')
            put(f"{name.replace('m3_', 'x_')}_grip_dependent_folds", r.get('grip_dependent_chosen_folds'), 'd', src_s,
                f'folds in which {name} chose a grip-dependent configuration')
    for role in ROLES:
        if role not in sel:
            continue
        s = sel[role]
        ship = shipped_name(role, sel, calj)
        exp_numbers(f'{role}', ex[s['best']], f"{ROLE_LABELS[role]}, base ({s['best']})", full=True)
        if ship in ex:
            exp_numbers(f'{role}_ship', ex[ship], f"{ROLE_LABELS[role]}, shipped ({ship})", full=True)
        put(f'{role}_n_configs', int(s['n_configs']), 'd', src_sel, f'configurations compared for the {ROLE_LABELS[role]}')
        put(f'{role}_post_hoc', 'yes' if s.get('post_hoc') else 'no', '', src_sel, f'{ROLE_LABELS[role]} designed post hoc')
        gr = s.get('grip_rule') or {}
        if 'acc_diff' in gr:
            put(f'{role}_grip_rule_diff', 100 * gr['acc_diff'], '+.1f', src_sel,
                'grip rule: mean per-person accuracy difference (points)')
            put(f'{role}_grip_rule_lo', 100 * gr['acc_diff_lo'], '+.1f', src_sel, 'grip rule: 95% CI low (points)')
            put(f'{role}_grip_rule_hi', 100 * gr['acc_diff_hi'], '+.1f', src_sel, 'grip rule: 95% CI high (points)')
    ref = Sm.get('reference')
    if ref:
        put('reference_accuracy', ref['accuracy'], '.1%', src_s, 'reference floor (yaw logreg) accuracy, for comparison')
        put('reference_accuracy_lo', ref['accuracy_lo'], '.1%', src_s, 'reference floor accuracy, 95% CI low')
        put('reference_accuracy_hi', ref['accuracy_hi'], '.1%', src_s, 'reference floor accuracy, 95% CI high')
    pj = L.read_json('m3_paired')
    for r in pj['rows']:
        key = comparison_key(r)
        what = f"{r['option']} (against {r['baseline']}; family {r['family']})"
        put(f'{key}_diff', 100 * r['diff'], '+.1f', src_p, f'mean per-person accuracy difference (points), {what}')
        put(f'{key}_lo', 100 * r['diff_lo'], '+.1f', src_p, f'paired 95% CI low (points), {what}')
        put(f'{key}_hi', 100 * r['diff_hi'], '+.1f', src_p, f'paired 95% CI high (points), {what}')
        put(f'{key}_p', r['p'], '.3f', src_p, f'sign-flip p, {what}')
        put(f'{key}_p_holm', r['p_holm'], '.3f', src_p, f"Holm-adjusted p (family size {r['family_size']}), {what}")
        put(f'{key}_ll', r['logloss_diff'], '+.3f', src_p, f'per-person log-loss difference, {what}')
        put(f'{key}_ll_lo', r['logloss_lo'], '+.3f', src_p, f'log-loss difference 95% CI low, {what}')
        put(f'{key}_ll_hi', r['logloss_hi'], '+.3f', src_p, f'log-loss difference 95% CI high, {what}')
        put(f'{key}_ll_p_holm', r.get('p_holm_logloss'), '.3f', src_p,
            f"Holm-adjusted sign-flip p of the log-loss difference (family size {r['family_size']}), {what}")
        put(f'{key}_ll_verdict', r.get('verdict_logloss'), '', src_p, f'log-loss verdict (Holm-adjusted), {what}')
        put(f'{key}_verdict', r['verdict'], '', src_p, f'verdict (Holm-adjusted), {what}')
        put(f'{key}_acc', r['accuracy'], '.1%', src_p, f"accuracy of {r['name']} on these trials, {what}")
        put(f'{key}_base_acc', r['baseline_accuracy'], '.1%', src_p, f"accuracy of {r['baseline']} on these trials, {what}")
        put(f'{key}_n_trials', r['n_trials'], 'd', src_p, f'trials compared, {what}')
        put(f'{key}_n_persons', r['n_persons'], 'd', src_p, f'persons compared, {what}')
    for fam in sorted({r['family'] for r in pj['rows']}):
        put(f'family_{fam}_size', sum(r['family'] == fam for r in pj['rows']), 'd', src_p, f'comparisons in family {fam}')
    for role, c in calj.items():
        dec = c['decision']
        put(f'{role}_cal_shipped', M.CALIB_LABELS.get(dec['shipped'], dec['shipped']), '', src_c, f'shipped calibration, {role}')
        put(f'{role}_cal_target_met', 'yes' if dec['shipped_target_met'] else 'no', '', src_c, f'docs/09 target met, {role}')
        for e in c['entries']:
            p = f"{role}_cal_{e['method']}"
            for k, fmt in (('ece', '.3f'), ('ece_null_q95', '.3f'), ('ece_null_p', '.3f'), ('brier', '.3f'),
                           ('log_loss', '.3f'), ('cal_slope', '.2f'), ('cal_slope_lo', '.2f'), ('cal_slope_hi', '.2f'),
                           ('cal_intercept', '+.2f'), ('cal_intercept_lo', '+.2f'), ('cal_intercept_hi', '+.2f'),
                           ('overconfidence', '+.3f'), ('accuracy', '.1%'), ('bins_off', 'd'),
                           ('bins_off_bootstrap_only', 'd'), ('bins_off_degenerate', 'd'),
                           ('bins_off_null_rate', '.0%')):
                put(f'{p}_{k}', e.get(k), fmt, src_c, f'{k}, {ROLE_LABELS[role]}, {e["method"]}')
            for vs in ('vs_none', 'vs_pre'):
                v = e.get(vs) or {}
                for k, fmt in (('logloss_diff', '+.4f'), ('logloss_lo', '+.4f'), ('logloss_hi', '+.4f'),
                               ('p_holm_logloss', '.3f'), ('brier_diff', '+.4f'), ('brier_lo', '+.4f'),
                               ('brier_hi', '+.4f')):
                    put(f'{p}_{vs}_{k}', v.get(k), fmt, src_c, f'{k} {vs}, {ROLE_LABELS[role]}, {e["method"]}')
            for pl in e['per_platform']:
                for k, fmt in (('ece', '.3f'), ('ece_null_q95', '.3f'), ('brier', '.3f'), ('log_loss', '.3f'),
                               ('cal_slope', '.2f'), ('accuracy', '.1%'), ('n', 'd'), ('n_persons', 'd')):
                    put(f"{p}_{pl['platform']}_{k}", pl.get(k), fmt, src_c,
                        f"{k} on {pl['platform']}, {ROLE_LABELS[role]}, {e['method']}")
            for ch, v in (e.get('nested_choices') or {}).items():
                put(f'{p}_chose_{ch}', v, 'd', src_c, f'folds whose in-fold calibrator choice was {ch}, {role}')
    agj = L.read_json('m3_aggregate')
    rec = agj['recommended']
    put('agg_recommended', AGG_LABELS.get(rec['aggregator'], str(rec['aggregator'])), '', src_a,
        f"recommended aggregator ({rec['rule']}; chosen among {rec['n_compared']} on the outer folds)")
    put('agg_n_compared', rec['n_compared'], 'd', src_a, 'aggregators compared')
    put('agg_preferred', AGG_LABELS.get(rec.get('preferred'), str(rec.get('preferred'))), '', src_a,
        'the aggregator designed for correlated attempts (kept when calibrated for every role)')
    for m, ks in (rec.get('calibrated_at_k') or {}).items():
        put(f'agg_{m}_calibrated_ks', ', '.join(str(k) for k in ks) or 'none', '', src_a,
            f'numbers of attempts at which {AGG_LABELS.get(m, m)} is calibrated for every role (over-confidence CI '
            'contains 0)')
    for role, a in agj['roles'].items():
        for which, tag in (('uncalibrated', 'u'), ('calibrated', 'c')):
            blk = a.get(which)
            if not blk:
                continue
            for r in blk['rows']:
                if r['k'] != rec['k'] and r['aggregator'] != rec['aggregator']:
                    continue                       # every aggregator at the decision k; the recommended one at every k
                p = f"{role}_agg{tag}_{r['aggregator']}_k{r['k']}"
                put(f'{p}_acc', r['accuracy'], '.1%', src_a, f"accuracy, {AGG_LABELS[r['aggregator']]}, {which}")
                put(f'{p}_acc_lo', r['accuracy_lo'], '.1%', src_a, f"95% CI low, {AGG_LABELS[r['aggregator']]}, {which}")
                put(f'{p}_acc_hi', r['accuracy_hi'], '.1%', src_a, f"95% CI high, {AGG_LABELS[r['aggregator']]}, {which}")
                put(f'{p}_oc', r['overconfidence'], '+.3f', src_a, f"over-confidence, {AGG_LABELS[r['aggregator']]}, {which}")
                put(f'{p}_oc_lo', r['overconfidence_lo'], '+.3f', src_a, f'over-confidence 95% CI low, {which}')
                put(f'{p}_oc_hi', r['overconfidence_hi'], '+.3f', src_a, f'over-confidence 95% CI high, {which}')
                put(f'{p}_ll', r['log_loss'], '.3f', src_a, f"log loss, {AGG_LABELS[r['aggregator']]}, {which}")
            for c in blk['c_k']:
                put(f"{role}_agg{tag}_c{int(c['k'])}", c['median'], '.2f', src_a, f"median c_{int(c['k'])} over folds, {which}")
    bdj = L.read_json('m3_breakdown')
    for r in bdj['rows']:
        p = f"bd_{r['role']}_{r['condition']}_{r['level']}"
        put(f'{p}_acc', r['accuracy'], '.1%', src_b, f"accuracy, {r['condition']} {r['level']}, {r['name']}")
        put(f'{p}_lo', r['accuracy_lo'], '.1%', src_b, f'95% CI low ({r["n_persons"]} persons)')
        put(f'{p}_hi', r['accuracy_hi'], '.1%', src_b, '95% CI high')
        put(f'{p}_n', r['n'], 'd', src_b, f"trials, {r['condition']} {r['level']}")
        put(f'{p}_persons', r['n_persons'], 'd', src_b, f"persons, {r['condition']} {r['level']}")
    try:
        sal = L.read_json('m3_saliency')
        src = 'tables/m3/m3_saliency.json'
        for k, fmt in (('share_odd', '.0%'), ('share_odd_uniform', '.0%'), ('share_yaw', '.0%'), ('share_device', '.0%'),
                       ('share_first_half_all', '.0%'), ('share_first_half_odd', '.0%'), ('share_first_half_even', '.0%'),
                       ('mirror_corr', '.2f'),
                       ('completeness_median_rel_err', '.3f'), ('ig_sign_agrees_with_logit', '.0%')):
            put(f'ig_{k}', sal.get(k), fmt, src, f'integrated gradients of the deep finalist: {k}')
        for k in ('first_half_odd_minus_even', 'first_half_odd_minus_even_lo', 'first_half_odd_minus_even_hi'):
            put(f'ig_{k}', 100 * sal[k], '+.1f', src, f'first-half share of |IG|, odd minus even channels (points): {k}')
        body_odd = [c for c in sal['odd_channels'] if c not in sal.get('yaw_channels', [])]
        if body_odd:
            put('ig_share_body_odd', sum(sal['by_channel'][c]['share_of_all'] for c in body_odd), '.0%', src,
                f"share of |IG| on the odd body-frame channels ({', '.join(body_odd)})")
        for c, v in sal['by_channel'].items():
            put(f'ig_{c}_share', v['share_of_all'], '.0%', src, f'share of |IG| on {c}')
            put(f'ig_{c}_first_half', v['first_half_share'], '.0%', src, f'share of {c} |IG| in the first half')
            put(f'ig_{c}_peak', v['peak_time_frac'], '.2f', src, f'time of the largest mean |IG| of {c} (window fraction)')
            put(f'ig_{c}_toward', v['toward_true_hand'], '+.2f', src,
                f'attribution of {c} toward the true hand (sum of IG x hand sign / sum of |IG|; 1 = always toward it)')
    except FileNotFoundError:
        pass
    tfx = L.read_json('m3_tf_example')
    put('tf_max_mag_diff_stft', tfx['max_abs_mag_diff_stft'], '.1e', 'tables/m3/m3_tf_example.json',
        'largest |STFT magnitude(x) - STFT magnitude(mirror x)| over every chiral channel and trial')
    put('tf_max_mag_diff_cwt', tfx['max_abs_mag_diff_cwt'], '.1e', 'tables/m3/m3_tf_example.json',
        'the same for the CWT')
    gtj = L.read_json('m3_grip_transfer')
    sv = gtj.get('sensor_vs_instructed') or {}
    put('grip_rotated_detected_upright', sum(int(d.get('upright', 0)) for i, d in sv.items()
                                             if not str(i).startswith(UPRIGHT_PREFIX)), 'd',
        'tables/m3/m3_grip_transfer.json', 'rotated-grip trials whose sensor-detected grip is upright')
    put('grip_upright_detected_other', sum(v for i, d in sv.items() if str(i).startswith(UPRIGHT_PREFIX)
                                           for c, v in d.items() if c != 'upright'), 'd',
        'tables/m3/m3_grip_transfer.json', 'upright-grip trials whose sensor-detected grip is not upright')
    for r in gtj['rows']:
        put(f"grip_{str(r['name']).replace('m3_', '')}_{str(r['level']).replace('_toward', '')}", r['accuracy'], '.1%',
            'tables/m3/m3_grip_transfer.json', f"accuracy on {r['level']} trials, {r['name']}")
    try:
        ln = L.read_json('m3_label_noise')
        src = 'tables/m3/m3_label_noise.json'
        put('swap_n_trials', ln['n_trials'], 'd', src, 'trials in the blocks with the hand-swap signature')
        put('swap_n_blocks', len(ln['blocks']), 'd', src, 'blocks with the hand-swap signature')
        put('swap_n_persons', ln['n_persons'], 'd', src, 'persons with such blocks')
        for role, v in ln['roles'].items():
            put(f'{role}_swapfree_acc', v['accuracy_without'], '.1%', src, f'accuracy without those trials, {role}')
            put(f'{role}_swapfree_acc_lo', v['accuracy_without_lo'], '.1%', src, f'95% CI low, {role}')
            put(f'{role}_swapfree_acc_hi', v['accuracy_without_hi'], '.1%', src, f'95% CI high, {role}')
            put(f'{role}_swap_acc', v['accuracy_on_swap'], '.1%', src, f'accuracy on those trials, {role}')
    except FileNotFoundError:
        pass
    L.write_numbers(n)


# ───────────────────────────── markdown digest (for the lane summary) ─────────────────────────────


def experiments_markdown() -> str:
    """tables/m3/m3_experiments.md: one line per experiment and the figure list with captions, from the saved files."""
    Sm = L.read_json('m3_summary')
    ex = pd.DataFrame(Sm['experiments'])
    lines = ['| Experiment (results/<name>.json) | Model | Accuracy [95 % CI] | AUC | Log loss | ECE | Natural use | k = 3 | '
             'CPU-min | Configs | Selection |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in ex.iterrows():
        nu = '--' if pd.isna(r.get('natural_use_answered_accuracy')) else f"{100 * r['natural_use_answered_accuracy']:.0f}%"
        k3 = '--' if pd.isna(r.get('acc_k3')) else f"{100 * r['acc_k3']:.1f}%"
        lines.append(f"| `{r['name']}` | {r['label']} | {100 * r['accuracy']:.1f}% [{100 * r['accuracy_lo']:.1f}, "
                     f"{100 * r['accuracy_hi']:.1f}] | {r['roc_auc']:.3f} | {r['log_loss']:.3f} | {r['ece']:.3f} | {nu} | "
                     f"{k3} | {r['cpu_min']:.1f} | {int(r['n_configs_tried'])} | {_sel_cell(r)} |")
    caps_p = L.fig_dir / 'captions.json'
    if caps_p.exists():
        caps = json.loads(caps_p.read_text(encoding='utf-8'))
        lines += ['', '| Figure | Message (decision) |', '|---|---|']
        for k, v in sorted(caps.items()):
            lines.append(f"| `figures/m3/{k}.pdf` | {v.get('decision', '')} |")
    text = '\n'.join(lines) + '\n'
    p = L.tab_dir / 'm3_experiments.md'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8', newline='\n')
    return text
