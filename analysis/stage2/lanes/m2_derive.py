"""Lane m2: everything the figures, tables and numbers need, derived from the SAVED results (no model is fitted).

``derive()`` runs at the end of the compute step and writes, in ``tables/m2/``:

* ``m2_experiments.json`` / ``.csv``  one row per experiment: metrics with person-bootstrap 95 % CIs, natural use
  (coverage, answered, forced-choice and half-credit accuracy), k attempts, gated metrics, solver convergence, the
  paired comparison against its baseline (with the equivalence check and Holm-adjusted p within its comparison
  family) and against the reference floor;
* ``m2_feature_registry.csv``         the default feature set: family, channel, statistic, parity, mirror map;
* ``m2_importance.json``              held-out |contribution| per feature, family and channel (default LR, LightGBM);
* ``m2_calibration.json``             per model and calibrator: ECE against its perfect-calibration null band,
  calibration slope / calibration-in-the-large with person-bootstrap CIs, log loss and Brier (overall and per
  platform), reliability bands, and paired log-loss / Brier differences against the uncalibrated model;
* ``m2_strata.csv`` (+ ``m2_strata_persons.csv``)  accuracy by condition, handedness, flag and gate (no CI below
  ``m2_stats.MIN_PERSONS_CI`` persons; their per-person values instead);
* ``m2_per_person.csv``               per-person accuracy of the headline models;
* ``m2_k_attempts.json``              k-attempt curves and the calibration of multi-attempt aggregates, with the
  tempered sum using the NESTED tau_k of each held-out person's fold (``info['attempt_tau']``);
* ``m2_lc_runs.csv``                  accuracy / log loss by number of training persons;
* ``m2_select_choices.csv``           the model the nested selection chose in every outer fold, with the margins;
* ``m2_summary.json``                 headline, finalists, comparison families, scales, convergence, fixed-window record;

and rebuilds ``results/registry_m2.csv`` from the saved results (robust to parallel workers), and
``results/finalists_m2.json``. Strata (platform, conditions, handedness, flags, gates) come from the cache's trial
and person tables and are used ONLY to stratify predictions after the fact.
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import m2_pipeline as P
from stage2.lanes import m2_stats as S
from stage2.lanes.m2_calibration import logit, sigmoid

L = P.L
REFERENCE = 'example_yaw_logreg'
#: the condition strata of the breakdowns (trial-table column -> label); conditions only stratify, never enter a model
STRATA = {'platform': 'Platform', 'orientation': 'Grip', 'speed': 'Speed', 'variant': 'Variant', 'posture': 'Posture',
          'start_position': 'Start position', 'handedness': 'Handedness (self-reported)'}
#: flag and gate strata (README section 2.1): trial-level columns that never remove a trial
FLAG_STRATA = {'gate_pass': 'Predictor gates G1 + G2', 'grip_mismatch': 'Flag: grip mismatch',
               'seg_disagree_100ms': 'Flag: segmentations disagree > 100 ms', 'clip_candidate': 'Flag: possible clipping',
               'repeated_samples': 'Flag: repeated samples dropped', 'false_start': 'Flag: false start',
               'screen_rotated': 'Flag: screen rotated'}
FLAG_LEVELS = {'gate_pass': {'True': 'pass', 'False': 'fail'}}
HEADLINE_ORDER = ('m2_select', 'm2_lr_full')         # nested selection first, then the pre-specified default
N_BOOT_DERIVED = 1000                                 # person-bootstrap resamples of the derived descriptive CIs
N_BOOT_ATTEMPTS = 300                                 # ... of the multi-attempt aggregates (k-subsets: many rows)
AGGREGATORS = ('mean_p', 'mean_logit', 'sum_logit', 'sum_logit_tempered')
#: paired comparisons of the multi-attempt aggregates (a, b): log loss of a minus log loss of b
AGG_PAIRS = (('sum_logit_tempered', 'sum_logit'), ('sum_logit_tempered', 'mean_p'), ('sum_logit', 'mean_p'))

#: comparison families of the multiplicity adjustment (Holm, per metric): every paired comparison of an experiment
#: against its baseline belongs to the family of its question
FAMILY_TEXT = {
    'model_vs_default': 'the model pipelines and the nested selection, each against the pre-specified default',
    'data_treatment': 'the data-treatment variants of the logistic regression (window, frame, bins, scaling, mirror), '
                      'each against the default',
    'families_aug': 'the feature-family ablations with mirror augmentation, each against all families',
    'families_nomirror': 'the feature-family ablations without mirroring, each against all families without mirroring',
    'vs_reference': 'the model pipelines, the default and the nested selection, each against the reference floor',
}


def family_of(e: P.Exp) -> str | None:
    """The comparison family of experiment ``e`` (None: no test against a baseline, e.g. the learning curve)."""
    g = e.group
    if g in ('model', 'select'):
        return 'model_vs_default'
    if g == 'calibration':
        return f'calibration_{e.baseline}'
    if g == 'mirror':
        return 'data_treatment' if e.model == 'lr_l2' else f'mirror_{e.model}'
    if g in ('window', 'frame', 'bins', 'scaling'):
        return 'data_treatment'
    if g.startswith('family_'):
        return 'families_nomirror' if g.endswith('_nomirror') else 'families_aug'
    return None


def family_text(fam: str) -> str:
    if fam in FAMILY_TEXT:
        return FAMILY_TEXT[fam]
    if fam.startswith('calibration_'):
        return f'the post-hoc calibrators of {fam[len("calibration_"):]}, each against the uncalibrated model'
    if fam.startswith('mirror_'):
        return f'the mirror variants of {fam[len("mirror_"):]} against mirror augmentation'
    return fam


# ───────────────────────────── helpers ─────────────────────────────


def clean(v: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python, recursively."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def write(name: str, obj: dict[str, Any]) -> None:
    L.write_json(name, clean(obj))


def available() -> dict[str, cv.ExperimentResult]:
    """The saved results of every m2 experiment that exists."""
    out = {}
    for name in P.experiments():
        p = C.RES_DIR / f'{name}.json'
        if p.exists():
            out[name] = cv.load_result(name)
    return out


def _quant(v: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def boot_metrics(y: np.ndarray, p: np.ndarray, persons: np.ndarray, n_boot: int = N_BOOT_DERIVED,
                 seed: int = C.SEED) -> dict[str, tuple[float, float]]:
    """Person-cluster bootstrap 95 % CIs of accuracy, log loss and Brier (vectorised; the experiments' own CIs come
    from the harness, B = 2000). No ECE: resampling inflates it, so its percentile CI can exclude the estimate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    persons = np.asarray(persons).astype(str)
    uniq, inv = np.unique(persons, return_inverse=True)
    cnt = np.bincount(inv, minlength=uniq.size).astype(float)
    terms = {'accuracy': cv.credit(y, p), 'log_loss': S.log_loss_terms(y, p),
             'brier': (p - y) ** 2}
    idx = np.random.default_rng(seed).integers(0, uniq.size, (n_boot, uniq.size))
    den = cnt[idx].sum(1)
    return {k: _quant(np.bincount(inv, v, uniq.size)[idx].sum(1) / den) for k, v in terms.items()}


def _ci(r: cv.ExperimentResult, k: str) -> tuple[float, float]:
    v = r.ci.get(k, (math.nan, math.nan))
    return float(v[0]), float(v[1])


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def fold_infos(r: cv.ExperimentResult) -> list[tuple[str, dict[str, Any]]]:
    """(held-out person, fit_predict info) of every outer fold."""
    pr = r.predictions.drop_duplicates('fold')[['fold', 'person_id', 'params']].sort_values('fold')
    return [(str(pid), json.loads(s) if isinstance(s, str) else {}) for pid, s in zip(pr['person_id'], pr['params'])]


def _params_summary(r: cv.ExperimentResult) -> str:
    """The chosen hyperparameters across folds: median for numbers, the mode for categories."""
    vals: dict[str, list[Any]] = {}
    for _, info in fold_infos(r):
        if 'chosen' in info:
            vals.setdefault('chosen', []).append(info['chosen'])
        for k, v in (info.get('params') or {}).items():
            vals.setdefault(k, []).append(v)
    parts = []
    for k, v in vals.items():
        num = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(num) == len(v) and num:
            parts.append(f'{k}={np.median(num):.3g}')
        else:
            s = pd.Series([str(x) for x in v]).value_counts()
            parts.append(f'{k}={s.index[0]}({s.iloc[0]}/{len(v)})')
    return '; '.join(parts)


sign_flip_p = S.sign_flip_p


def fast_paired_compare(a: cv.ExperimentResult, b: cv.ExperimentResult, *, n_boot: int = C.N_BOOT,
                        seed: int = C.SEED) -> dict[str, Any]:
    """``cv.paired_compare`` vectorised: the same statistics from the same random stream (the person draws of
    ``Generator.choice`` with replacement are ``Generator.integers``), so the CIs agree with the core function to
    floating-point rounding (``tests/test_m2_models.py``) at a fraction of its cost (no pandas in the loop); plus the
    paired Brier difference and the equivalence check of the two prediction vectors."""
    pa, pb = a.predictions, b.predictions
    j = pa[['trial_id', 'person_id', 'y', 'p_right']].merge(pb[['trial_id', 'p_right']], on='trial_id',
                                                             suffixes=('_a', '_b'))
    y = j['y'].to_numpy().astype(int)
    qa, qb = j['p_right_a'].to_numpy(dtype=float), j['p_right_b'].to_numpy(dtype=float)
    ca, cb = cv.credit(y, qa), cv.credit(y, qb)          # the harness's rule: a tie earns half credit
    la, lb = S.log_loss_terms(y, qa), S.log_loss_terms(y, qb)
    ba, bb = (qa - y) ** 2, (qb - y) ** 2
    persons = j['person_id'].astype(str).to_numpy()
    uniq, inv = np.unique(persons, return_inverse=True)
    n_p = np.bincount(inv, minlength=uniq.size).astype(float)
    sa, sb = np.bincount(inv, ca, uniq.size), np.bincount(inv, cb, uniq.size)
    d_acc = (sa - sb) / n_p
    d_ll = (np.bincount(inv, la, uniq.size) - np.bincount(inv, lb, uniq.size)) / n_p
    d_br = (np.bincount(inv, ba, uniq.size) - np.bincount(inv, bb, uniq.size)) / n_p
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.integers(0, uniq.size, uniq.size) for _ in range(n_boot)])
    boot_pool = (sa[idx].sum(1) - sb[idx].sum(1)) / n_p[idx].sum(1)
    eq = S.equivalent(qa, qb)
    return {
        'n_trials': int(len(j)), 'n_persons': int(uniq.size), 'acc_a': float(ca.mean()), 'acc_b': float(cb.mean()),
        'pooled_acc_diff': float(ca.mean() - cb.mean()), 'pooled_acc_diff_ci95': _quant(boot_pool),
        'mean_person_acc_diff': float(d_acc.mean()), 'mean_person_acc_diff_ci95': _quant(d_acc[idx].mean(1)),
        'p_sign_flip_acc': sign_flip_p(d_acc, seed=seed),
        'mean_person_logloss_diff': float(d_ll.mean()), 'mean_person_logloss_diff_ci95': _quant(d_ll[idx].mean(1)),
        'p_sign_flip_logloss': sign_flip_p(d_ll, seed=seed),
        'mean_person_brier_diff': float(d_br.mean()), 'mean_person_brier_diff_ci95': _quant(d_br[idx].mean(1)),
        'p_sign_flip_brier': sign_flip_p(d_br, seed=seed),
        'persons_a_better': int((d_acc > 0).sum()), 'persons_b_better': int((d_acc < 0).sum()),
        'max_abs_dp': eq['max_abs_dp'], 'n_class_changes': eq['n_class_changes'], 'identical': eq['identical'],
    }


def _paired(a: cv.ExperimentResult, b: cv.ExperimentResult) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='paired_compare')
        c = fast_paired_compare(a, b)
    return {'paired_n': c['n_trials'], 'diff_acc': c['pooled_acc_diff'], 'diff_acc_lo': c['pooled_acc_diff_ci95'][0],
            'diff_acc_hi': c['pooled_acc_diff_ci95'][1], 'diff_acc_person': c['mean_person_acc_diff'],
            'diff_acc_person_lo': c['mean_person_acc_diff_ci95'][0],
            'diff_acc_person_hi': c['mean_person_acc_diff_ci95'][1], 'p_acc': c['p_sign_flip_acc'],
            'diff_ll': c['mean_person_logloss_diff'], 'diff_ll_lo': c['mean_person_logloss_diff_ci95'][0],
            'diff_ll_hi': c['mean_person_logloss_diff_ci95'][1], 'p_ll': c['p_sign_flip_logloss'],
            'diff_brier': c['mean_person_brier_diff'], 'diff_brier_lo': c['mean_person_brier_diff_ci95'][0],
            'diff_brier_hi': c['mean_person_brier_diff_ci95'][1], 'p_brier': c['p_sign_flip_brier'],
            'persons_better': c['persons_a_better'], 'persons_worse': c['persons_b_better'],
            'max_abs_dp': c['max_abs_dp'], 'n_class_changes': c['n_class_changes'], 'identical': c['identical']}


def _holm_rows(rows: list[dict[str, Any]], prefix: str, metrics: tuple[str, ...], fam_key: str) -> None:
    """Holm-adjust ``<prefix>p_<metric>`` within every family (rows sharing ``fam_key``), in place. A comparison of
    practically identical pipelines is not a test (it would only measure solver noise): it gets no adjusted p and
    does not count towards the family size."""
    fams: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        f = r.get(fam_key)
        if f and r.get(f'{prefix}p_acc') is not None and math.isfinite(_f(r.get(f'{prefix}p_acc'))):
            fams.setdefault(f, []).append(i)
    for f, idx in fams.items():
        test = [i for i in idx if not rows[i].get(f'{prefix}identical')]
        for m in metrics:
            adj = S.holm([_f(rows[i].get(f'{prefix}p_{m}')) for i in test])
            for i, a in zip(test, adj):
                rows[i][f'{prefix}p_{m}_holm'] = float(a)
                d = _f(rows[i].get(f'{prefix}diff_{m}'))
                rows[i][f'{prefix}sig_{m}'] = bool(a < S.ALPHA and math.isfinite(d) and d != 0)
            for i in idx:
                if rows[i].get(f'{prefix}identical'):
                    rows[i][f'{prefix}p_{m}_holm'] = math.nan
                    rows[i][f'{prefix}sig_{m}'] = False
        for i in idx:
            rows[i][f'{prefix}family_size'] = len(test)


# ───────────────────────────── experiment table ─────────────────────────────


def _convergence(r: cv.ExperimentResult) -> dict[str, Any]:
    fits = nonconv = finals_bad = 0
    seen = False
    for _, info in fold_infos(r):
        if 'n_inner_fits' in info:
            seen = True
            fits += int(info.get('n_inner_fits') or 0)
            nonconv += int(info.get('n_inner_nonconverged') or 0)
            finals_bad += int(info.get('final_converged') is False)
    if not seen:
        return {}
    return {'n_inner_fits': fits, 'n_inner_nonconverged': nonconv, 'n_final_nonconverged': finals_bad}


def _tau_medians(r: cv.ExperimentResult) -> dict[str, float]:
    taus: dict[str, list[float]] = {}
    for _, info in fold_infos(r):
        for k, v in (info.get('attempt_tau') or {}).items():
            if v is not None and math.isfinite(float(v)):
                taus.setdefault(str(k), []).append(float(v))
    return {f'tau_k{k}_median': float(np.median(v)) for k, v in sorted(taus.items())}


def experiment_table(res: dict[str, cv.ExperimentResult], ref: cv.ExperimentResult | None) -> pd.DataFrame:
    exps = P.experiments()
    rows = []
    for name, r in res.items():
        e = exps[name]
        kc = r.k_curve.set_index('k') if len(r.k_curve) else pd.DataFrame()
        row = {'name': name, 'group': e.group, 'label': e.label, 'model': e.model, 'mirror': e.mirror,
               'calibrator': e.calibrator, 'window': e.feat.window, 'channels': e.feat.channels,
               'families': '+'.join(e.feat.families), 'n_bins': e.feat.n_bins, 'amp_norm': e.feat.amp_norm,
               'scaler': e.scaler, 'winsor': e.winsor, 'n_train_persons': e.n_train_persons, 'rep': e.rep,
               'n_features': (r.params or {}).get('n_features'), 'n_trials': int(r.metrics['n']),
               'n_persons': int(r.predictions['person_id'].nunique()), 'n_folds': r.info.get('n_folds'),
               'n_configs_tried': r.info.get('n_configs_tried'), 'leak_check': r.leak_check,
               'baseline': e.baseline, 'family': family_of(e) if e.baseline and e.baseline != name else None,
               'chosen_params': _params_summary(r)}
        for k in cv.METRIC_NAMES:
            row[k] = _f(r.metrics.get(k))
            row[f'{k}_lo'], row[f'{k}_hi'] = _ci(r, k)
        row['ece_lo'] = row['ece_hi'] = math.nan            # never reported: see boot_metrics
        row['person_acc_median'] = _f(r.metrics.get('person_acc_median'))
        row['n_persons_below_80'] = _f(r.metrics.get('n_persons_below_80'))
        pr = r.predictions
        pp = pr['p_right'].to_numpy(dtype=float)
        row['p_min'], row['p_max'] = float(pp.min()), float(pp.max())
        right = pr['y'].to_numpy() == 1
        prim = S.natural_use(pp[right], np.ones(int(right.sum()), int), pr['person_id'].to_numpy()[right])
        row['primary_right_recall'] = prim['half_credit_accuracy']
        row['primary_coverage'] = float(np.mean(np.abs(pp - 0.5) > S.ABSTAIN_TOL))
        ex = r.extra_predictions
        if ex is not None and len(ex):
            nu = S.natural_use(ex['p_right'].to_numpy(dtype=float), ex['y'].to_numpy(), ex['person_id'].to_numpy())
            row.update({'natural_use_n': nu['n'], 'natural_use_n_right': nu['n_right'], 'natural_use_n_left': nu['n_left'],
                        'natural_use_n_persons': nu['n_persons'], 'natural_use_n_abstain': nu['n_abstain'],
                        'natural_use_coverage': nu['coverage'],
                        'natural_use_forced_accuracy': nu['forced_choice_accuracy'],
                        'natural_use_answered_accuracy': nu['answered_accuracy'],
                        'natural_use_accuracy': nu['half_credit_accuracy'],
                        'natural_use_lo': nu.get('half_credit_lo', math.nan),
                        'natural_use_hi': nu.get('half_credit_hi', math.nan),
                        'natural_use_correct': int(round(nu['forced_choice_accuracy'] * nu['n'])),
                        'natural_use_correct_answered': nu['n_correct_answered']})
        else:
            row.update({'natural_use_n': 0, 'natural_use_accuracy': math.nan, 'natural_use_correct': 0})
        for k in C.K_ATTEMPTS:
            if k in kc.index:
                row[f'acc_k{k}'] = _f(kc.loc[k, 'accuracy'])
                row[f'acc_k{k}_lo'], row[f'acc_k{k}_hi'] = _f(kc.loc[k, 'lo']), _f(kc.loc[k, 'hi'])
        row['k_n_persons'] = int(kc['n_persons'].iloc[0]) if len(kc) else 0
        gm = r.gated_metrics or {}
        row['gated_n'], row['gated_accuracy'] = int(gm.get('n', 0) or 0), _f(gm.get('accuracy'))
        row['gated_log_loss'] = _f(gm.get('log_loss'))
        row['n_failing_gates'] = int(gm.get('n_failing_gates', 0) or 0)
        row.update(_convergence(r))
        row.update(_tau_medians(r))
        if e.baseline and e.baseline in res and e.baseline != name:
            row.update({f'vs_base_{k}': v for k, v in _paired(r, res[e.baseline]).items()})
        if ref is not None and e.group in ('default', 'model', 'select'):
            row.update({f'vs_ref_{k}': v for k, v in _paired(r, ref).items()})
            row['ref_family'] = 'vs_reference'
        rows.append(row)
    _holm_rows(rows, 'vs_base_', ('acc', 'll'), 'family')
    _holm_rows(rows, 'vs_ref_', ('acc', 'll'), 'ref_family')
    return pd.DataFrame(rows)


# ───────────────────────────── calibration ─────────────────────────────


def reliability_bands(pred: pd.DataFrame, n_bins: int = C.ECE_BINS, n_boot: int = N_BOOT_DERIVED,
                      seed: int = C.SEED) -> pd.DataFrame:
    """Equal-mass reliability bins (edges from all trials, as cv.reliability) with person-bootstrap 95 % bands of
    the observed share of right hands in every bin."""
    p = pred['p_right'].to_numpy(dtype=float)
    y = pred['y'].to_numpy(dtype=float)
    persons = pred['person_id'].astype(str).to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    k = np.searchsorted(edges[1:-1], p, side='right')
    bins = [b for b in range(n_bins) if np.any(k == b)]
    uniq, inv = np.unique(persons, return_inverse=True)
    idx = np.random.default_rng(seed).integers(0, uniq.size, (n_boot, uniq.size))
    out = []
    for j, b in enumerate(bins):
        m = k == b
        sy = np.bincount(inv[m], y[m], uniq.size)
        cn = np.bincount(inv[m], minlength=uniq.size).astype(float)
        num, den = sy[idx].sum(1), cn[idx].sum(1)
        share = num[den > 0] / den[den > 0]
        out.append({'bin': j, 'p_lo': float(p[m].min()), 'p_hi': float(p[m].max()), 'p_mean': float(p[m].mean()),
                    'y_mean': float(y[m].mean()), 'n': int(m.sum()), 'n_persons': int(np.unique(inv[m]).size),
                    'y_lo': float(np.quantile(share, 0.025)), 'y_hi': float(np.quantile(share, 0.975))})
    return pd.DataFrame(out)


def calibration_stats(pred: pd.DataFrame, n_boot: int = N_BOOT_DERIVED, seed: int = C.SEED) -> dict[str, Any]:
    """Accuracy, AUC, log loss and Brier (person-bootstrap CIs), the ECE against its perfect-calibration null band
    (no bootstrap CI), and the calibration slope and calibration-in-the-large with person-bootstrap CIs. Below
    ``MIN_PERSONS_CI`` persons: point estimates only."""
    y = pred['y'].to_numpy(dtype=float)
    p = pred['p_right'].to_numpy(dtype=float)
    persons = pred['person_id'].astype(str).to_numpy()
    m = cv.metrics(y, p)
    out: dict[str, Any] = {'n': int(m['n']), 'n_persons': int(np.unique(persons).size)}
    for k in ('accuracy', 'log_loss', 'brier', 'roc_auc'):
        out[k] = _f(m.get(k))
    out.update(S.ece_null(y, p, seed=seed))
    out.update(S.recalibration(y, p))
    out['mean_p'], out['share_right'] = float(p.mean()), float(y.mean())
    if out['n_persons'] >= S.MIN_PERSONS_CI and 0 < y.sum() < y.size:
        ci = boot_metrics(y, p, persons, n_boot, seed)
        for k, v in ci.items():
            out[f'{k}_lo'], out[f'{k}_hi'] = v
        rc = S.recalibration_ci(y, p, persons, n_boot=n_boot, seed=seed)
        out['cal_slope_lo'], out['cal_slope_hi'] = rc['cal_slope']
        out['cal_in_the_large_lo'], out['cal_in_the_large_hi'] = rc['cal_in_the_large']
    return out


def miscalibration_detected(st: dict[str, Any]) -> bool:
    """Evidence of miscalibration: ECE above its null band, or a slope CI excluding 1, or a calibration-in-the-large
    CI excluding 0."""
    ece_high = _f(st.get('ece')) > _f(st.get('ece_null_hi'))
    slope = (_f(st.get('cal_slope_lo')) > 1) or (_f(st.get('cal_slope_hi')) < 1)
    citl = (_f(st.get('cal_in_the_large_lo')) > 0) or (_f(st.get('cal_in_the_large_hi')) < 0)
    return bool(ece_high or slope or citl)


def best_by_log_loss(tab: pd.DataFrame) -> str | None:
    """The model pipeline (default or comparison) with the lowest outer log loss (an exploratory choice)."""
    cand = tab[tab['group'].isin(['default', 'model'])]
    return None if cand.empty else str(cand.sort_values(['log_loss', 'name']).iloc[0]['name'])


def calibration_models(res: dict[str, cv.ExperimentResult], tab: pd.DataFrame) -> list[str]:
    """The models whose calibration is analysed: the headline (nested selection), the pre-specified default, the
    models with in-fold calibrator variants and the pipeline with the lowest outer log loss."""
    best = best_by_log_loss(tab)
    names = list(HEADLINE_ORDER) + [src for src, _ in P.CALIBRATED] + ([best] if best else [])
    return [n for n in dict.fromkeys(names) if n in res]


def calibration_summary(res: dict[str, cv.ExperimentResult], trials: pd.DataFrame, tab: pd.DataFrame) -> dict[str, Any]:
    """Per analysed model and calibrator: overall and per-platform calibration statistics, reliability bands, the
    fitted calibrator parameters (median over folds), and paired log-loss / Brier differences against the
    uncalibrated model (Holm-adjusted within the model's calibrators); the platform-specific map also against the
    global Platt map. ``decision``: whether miscalibration is detected and whether a calibrator is adopted."""
    plat = trials.set_index('trial_id')['platform'].astype(str)
    out: dict[str, Any] = {'models': {}, 'order': calibration_models(res, tab), 'alpha': S.ALPHA}
    for src in out['order']:
        entry: dict[str, Any] = {}
        for cal in ('none', *P.CALIBRATORS):
            name = src if cal == 'none' else f'{src}_cal_{cal}'
            if name not in res:
                continue
            pred = res[name].predictions
            row: dict[str, Any] = {'name': name, 'calibrator': cal, 'overall': calibration_stats(pred), 'platform': {}}
            pl = pred['trial_id'].map(plat).to_numpy()
            for v in sorted(pd.Series(pl).dropna().unique()):
                row['platform'][str(v)] = calibration_stats(pred[pl == v].reset_index(drop=True))
            row['reliability'] = reliability_bands(pred).to_dict(orient='records')
            cp = [info.get('calib_params') or {} for _, info in fold_infos(res[name])]
            keys = sorted({k for c in cp for k in c})
            row['calib_params_median'] = {k: float(np.median([c[k] for c in cp if k in c])) for k in keys}
            if cal != 'none':
                row['vs_uncalibrated'] = _paired(res[name], res[src])
            if cal == 'platt_platform' and f'{src}_cal_platt' in res:
                row['vs_platt'] = _paired(res[name], res[f'{src}_cal_platt'])
            entry[cal] = row
        cals = [c for c in entry if c != 'none']
        for m in ('ll', 'brier'):
            adj = S.holm([_f(entry[c]['vs_uncalibrated'][f'p_{m}']) for c in cals])
            for c, a in zip(cals, adj):
                entry[c]['vs_uncalibrated'][f'p_{m}_holm'] = float(a)
        adopted = [c for c in cals if entry[c]['vs_uncalibrated']['diff_ll'] < 0
                   and entry[c]['vs_uncalibrated']['p_ll_holm'] < S.ALPHA and not entry[c]['vs_uncalibrated']['identical']]
        best = min(cals, key=lambda c: entry[c]['overall']['log_loss']) if cals else None
        none = entry.get('none', {}).get('overall', {})
        out['models'][src] = {'variants': entry, 'decision': {
            'miscalibration_detected': miscalibration_detected(none) if none else None,
            'platform_miscalibration_detected': {v: miscalibration_detected(s) for v, s in
                                                 entry.get('none', {}).get('platform', {}).items()},
            'best_calibrator': best, 'adopted': adopted[0] if adopted else 'none', 'n_calibrators': len(cals)}}
    # the platform-specific map against one global map is its own comparison family (one per analysed model):
    # Holm-adjusted across the models, and its verdict (better / worse / not different) uses the adjusted p
    vp = [(src, out['models'][src]['variants']['platt_platform']['vs_platt']) for src in out['models']
          if out['models'][src]['variants'].get('platt_platform', {}).get('vs_platt')]
    for m in ('ll', 'brier'):
        for (_, v), a in zip(vp, S.holm([_f(v[f'p_{m}']) for _, v in vp])):
            v[f'p_{m}_holm'] = float(a)
    for src, v in vp:
        sig = v['p_ll_holm'] < S.ALPHA and not v['identical']
        v['verdict_ll'] = ('identical' if v['identical'] else
                           ('better' if sig and v['diff_ll'] < 0 else 'worse' if sig else 'not different'))
        v['family_size'] = len(vp)
        out['models'][src]['decision']['platform_map_verdict'] = v['verdict_ll']
    return out


# ───────────────────────────── breakdowns ─────────────────────────────


def _acc_ci(g: pd.DataFrame, n_boot: int = N_BOOT_DERIVED, seed: int = C.SEED) -> tuple[float, float, float]:
    ok = cv.credit(g['y'], g['p_right'])
    persons = g['person_id'].astype(str).to_numpy()
    uniq, inv = np.unique(persons, return_inverse=True)
    if uniq.size < S.MIN_PERSONS_CI:
        return float(ok.mean()), math.nan, math.nan
    cnt = np.bincount(inv, minlength=uniq.size).astype(float)
    idx = np.random.default_rng(seed).integers(0, uniq.size, (n_boot, uniq.size))
    bs = np.bincount(inv, ok, uniq.size)[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = _quant(bs)
    return float(ok.mean()), lo, hi


def _strata_frame(pred: pd.DataFrame, trials: pd.DataFrame, persons: pd.DataFrame) -> pd.DataFrame:
    t = trials.set_index('trial_id')
    hand_of = persons.set_index('person_id')['handedness_self'].astype(str)
    pred = pred.copy()
    for col in list(STRATA) + list(FLAG_STRATA):
        if col == 'handedness':
            pred[col] = pred['person_id'].map(hand_of).to_numpy()
        elif col in t.columns:
            v = pred['trial_id'].map(t[col])
            if col in FLAG_STRATA:
                v = v.map(lambda x: str(bool(x)) if pd.notna(x) else 'nan')
            pred[col] = v.astype(str).to_numpy()
    return pred


def breakdown(res: dict[str, cv.ExperimentResult], trials: pd.DataFrame, persons: pd.DataFrame,
              names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(strata, per-person values of the strata with fewer than MIN_PERSONS_CI persons)."""
    rows, prow = [], []
    for name in names:
        if name not in res:
            continue
        pred = _strata_frame(res[name].predictions, trials, persons)
        for col, lab in {**STRATA, **FLAG_STRATA}.items():
            if col not in pred.columns:
                continue
            kind = 'flag' if col in FLAG_STRATA else 'condition'
            for lev, g in pred.groupby(col, sort=True):
                acc, lo, hi = _acc_ci(g)
                n_p = int(g['person_id'].nunique())
                yy = g['y'].to_numpy()
                ll = float(S.log_loss_terms(yy, g['p_right'].to_numpy(dtype=float)).mean())
                lev_s = str(lev)
                level = FLAG_LEVELS.get(col, {'True': 'flagged', 'False': 'not flagged'}).get(lev_s, lev_s) \
                    if kind == 'flag' else lev_s
                rows.append({'model': name, 'kind': kind, 'factor': col, 'factor_label': lab, 'level': level,
                             'n_trials': len(g), 'n_persons': n_p, 'accuracy': acc, 'acc_lo': lo, 'acc_hi': hi,
                             'ci': bool(n_p >= S.MIN_PERSONS_CI), 'log_loss': ll})
                if n_p < S.MIN_PERSONS_CI:
                    for pid, gp in g.groupby('person_id', sort=True):
                        okp = np.mean(cv.credit(gp['y'], gp['p_right']))
                        prow.append({'model': name, 'factor': col, 'level': level, 'person_id': str(pid),
                                     'n_trials': len(gp), 'accuracy': float(okp)})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def per_person(res: dict[str, cv.ExperimentResult], persons: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    pinfo = persons.set_index('person_id')[['platform', 'handedness_self']]
    rows = []
    for name in names:
        if name not in res:
            continue
        pp = res[name].per_person.copy()
        pp.insert(0, 'model', name)
        pp['platform'] = pp['person_id'].map(pinfo['platform']).to_numpy()
        pp['handedness'] = pp['person_id'].map(pinfo['handedness_self']).to_numpy()
        rows.append(pp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ───────────────────────────── k attempts and their calibration ─────────────────────────────


def _draws(pred: pd.DataFrame, k: int, n_draws: int, seed: int, eligible: set[str]) -> pd.DataFrame:
    """Random k-subsets of every eligible (person, hand) cell (the scheme of cv.k_attempt_curve)."""
    import itertools
    rng = np.random.default_rng(seed)
    out = []
    for (pid, yy), g in pred.groupby(['person_id', 'y'], sort=True):
        if str(pid) not in eligible:
            continue
        p = g['p_right'].to_numpy(dtype=float)
        if p.size < k:
            continue
        if math.comb(p.size, k) <= n_draws:
            subsets = np.array(list(itertools.combinations(range(p.size), k)))
        else:
            subsets = np.array([rng.choice(p.size, size=k, replace=False) for _ in range(n_draws)])
        z = logit(p[subsets])
        out.append(pd.DataFrame({'person_id': str(pid), 'y': int(yy), 'k': k,
                                 'mean_p': p[subsets].mean(1), 'mean_logit': sigmoid(z.mean(1)),
                                 'sum_logit_z': z.sum(1)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def nested_tau(r: cv.ExperimentResult) -> dict[str, dict[str, float]]:
    """{held-out person: {k: tau_k}}: the multi-attempt temperatures fitted inside that person's outer fold on the
    training persons' inner cross-fitted predictions (empty for results without them)."""
    out = {}
    for pid, info in fold_infos(r):
        tau = {str(k): float(v) for k, v in (info.get('attempt_tau') or {}).items()
               if v is not None and math.isfinite(float(v)) and float(v) > 0}
        if tau:
            out[pid] = tau
    return out


def _boot_mean_ci(v: np.ndarray, persons: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    uniq, inv = np.unique(np.asarray(persons).astype(str), return_inverse=True)
    s = np.bincount(inv, v, uniq.size)
    c = np.bincount(inv, minlength=uniq.size).astype(float)
    idx = np.random.default_rng(seed).integers(0, uniq.size, (n_boot, uniq.size))
    return _quant(s[idx].sum(1) / c[idx].sum(1))


def k_attempt_analysis(res: dict[str, cv.ExperimentResult], names: list[str], n_draws: int = 200) -> dict[str, Any]:
    """k-attempt accuracy (per-person balanced, as the harness) and the calibration of four aggregates of k attempts
    of one person and hand: mean P, mean logit, sum of logits (independent evidence) and the sum tempered by the
    NESTED tau_k of the person's outer fold (fitted on the training persons' inner cross-fitted predictions, so the
    held-out person never informs its own temperature). Per aggregate: log loss (person-bootstrap CI), Brier, ECE,
    and the confidence gap (mean max(P, 1-P) minus the share correct, CI: above 0 = over-confident, below 0 =
    under-confident); paired per-person log-loss differences between aggregates (sign-flip p, Holm over the pairs at
    the largest k); the recommended displayed aggregate is the one with the lowest log loss at the largest k."""
    out: dict[str, Any] = {}
    kmax = max(C.K_ATTEMPTS)
    for name in names:
        if name not in res:
            continue
        pred = res[name].predictions
        counts = pred.groupby(['person_id', 'y']).size().unstack(fill_value=0)
        eligible = set(counts.index[(counts.get(0, 0) >= kmax) & (counts.get(1, 0) >= kmax)].astype(str))
        taus = nested_tau(res[name])
        rows, pairs = [], []
        for k in C.K_ATTEMPTS:
            dr = _draws(pred, k, n_draws, C.SEED + k, eligible)
            if not len(dr):
                continue
            z, yv = dr['sum_logit_z'].to_numpy(), dr['y'].to_numpy()
            pid = dr['person_id'].to_numpy()
            dr['sum_logit'] = sigmoid(z)
            aggs = [a for a in AGGREGATORS if a != 'sum_logit_tempered']
            if taus and all(str(u) in taus and str(k) in taus[str(u)] for u in np.unique(pid)):
                tk = np.array([taus[str(u)][str(k)] for u in pid])
                dr['sum_logit_tempered'] = sigmoid(z / tk)
                aggs.append('sum_logit_tempered')
            ll_terms = {}
            for agg in aggs:
                q = dr[agg].to_numpy()
                corr = pd.Series(cv.credit(yv, q))
                bal = corr.groupby([dr['person_id'], dr['y']]).mean().groupby(level=0).mean()
                mm = cv.metrics(yv, q)
                ll_terms[agg] = S.log_loss_terms(yv, q)
                lo, hi = _boot_mean_ci(ll_terms[agg], pid, N_BOOT_ATTEMPTS, C.SEED + k)
                conf = float(np.mean(np.maximum(q, 1 - q)))
                gap = np.maximum(q, 1 - q) - cv.credit(yv, q)
                glo, ghi = _boot_mean_ci(gap, pid, N_BOOT_ATTEMPTS, C.SEED + k)
                # the calibration slope is not used here: summed logits of several attempts are nearly separable,
                # so its maximum-likelihood fit diverges; confidence minus accuracy is the robust measure
                rows.append({'k': k, 'aggregator': agg, 'n_draws': len(dr), 'n_persons': len(np.unique(pid)),
                             'balanced_accuracy': float(bal.mean()), 'accuracy': mm['accuracy'],
                             'log_loss': mm['log_loss'], 'log_loss_lo': lo, 'log_loss_hi': hi, 'ece': mm['ece'],
                             'brier': mm['brier'], 'mean_confidence': conf, 'conf_gap': float(gap.mean()),
                             'conf_gap_lo': glo, 'conf_gap_hi': ghi})
            for a, b in AGG_PAIRS:
                if a in ll_terms and b in ll_terms:
                    pairs.append({'k': k, 'a': a, 'b': b, **S.paired_losses(ll_terms[a], ll_terms[b], pid)})
        last = [i for i, r in enumerate(pairs) if r['k'] == kmax]
        for i, a in zip(last, S.holm([pairs[i]['p'] for i in last])):
            pairs[i]['p_holm'] = float(a)
        rk = [r for r in rows if r['k'] == kmax]
        rec = min(rk, key=lambda r: r['log_loss'])['aggregator'] if rk else None
        tau_med = {}
        for k in C.K_ATTEMPTS:
            v = [t[str(k)] for t in taus.values() if str(k) in t]
            if v:
                tau_med[str(k)] = float(np.median(v))
        out[name] = {'rows': rows, 'pairs': pairs, 'recommended': rec, 'decision_rule': 'mean_p',
                     'tau_nested_median': tau_med,
                     'effective_independent_attempts': {k: (int(k) / v if v > 0 else math.nan)
                                                        for k, v in tau_med.items()},
                     'k_curve': res[name].k_curve.to_dict(orient='records'), 'k_max': kmax,
                     'persons_left_out': sorted(set(counts.index.astype(str)) - eligible)}
    return out


# ───────────────────────────── learning curve, selection, importance ─────────────────────────────


def learning_curve(res: dict[str, cv.ExperimentResult]) -> pd.DataFrame:
    exps = P.experiments()
    rows = []
    for name, r in res.items():
        e = exps[name]
        if e.group != 'learning_curve' and name != 'm2_lr_full':
            continue
        n_tr = [info.get('n_train_persons') for _, info in fold_infos(r)]
        rows.append({'name': name, 'n_train_persons_requested': e.n_train_persons or 'all',
                     'n_train_persons': float(np.median([v for v in n_tr if v is not None])), 'rep': e.rep,
                     'accuracy': r.metrics['accuracy'], 'acc_lo': r.ci['accuracy'][0], 'acc_hi': r.ci['accuracy'][1],
                     'log_loss': r.metrics['log_loss'], 'roc_auc': r.metrics['roc_auc'], 'ece': r.metrics['ece']})
    return pd.DataFrame(rows).sort_values(['n_train_persons', 'rep']).reset_index(drop=True) if rows else pd.DataFrame()


def select_choices(res: dict[str, cv.ExperimentResult]) -> pd.DataFrame:
    if 'm2_select' not in res:
        return pd.DataFrame()
    rows = []
    for pid, info in fold_infos(res['m2_select']):
        ll = info.get('inner_logloss') or {}
        srt = sorted(ll.values())
        rows.append({'person_id': pid, 'chosen': info.get('chosen'),
                     'margin': (srt[1] - srt[0]) if len(srt) > 1 else math.nan,
                     **{f'inner_ll_{k}': v for k, v in ll.items()}})
    return pd.DataFrame(rows)


def _channels_of(c: Any) -> list[str]:
    return [] if str(c) == '-' else str(c).split('|')


def importance_summary() -> dict[str, Any]:
    """Held-out |contribution| shares by family, parity, frame and channel; a correlation pair's contribution is
    split equally between its two channels (crediting it to the first channel would favour the channels listed
    first)."""
    from stage2.standardize import DEVICE_CHANNELS
    out = {}
    for name in ('m2_lr_full', 'm2_lgbm'):
        p = L.tab_dir / f'{P.importance_name(name)}.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        f = pd.DataFrame(d['features'])
        tot = f['mean_abs_contrib'].sum()
        fam = (f.groupby('family')['mean_abs_contrib'].sum() / tot).sort_values(ascending=False)
        par = (f.groupby('parity')['mean_abs_contrib'].sum() / tot)
        split = [(ch, v / len(chs)) for chs, v in zip(f['channel'].map(_channels_of), f['mean_abs_contrib'])
                 for ch in chs]
        ch = (pd.DataFrame(split, columns=['ch', 'v']).groupby('ch')['v'].sum() / tot).sort_values(ascending=False)
        frame = pd.Series({'device': float(ch[[c for c in ch.index if c in DEVICE_CHANNELS]].sum()),
                           'gravity': float(ch[[c for c in ch.index if c not in DEVICE_CHANNELS]].sum()),
                           'trial': float(f.loc[f['channel'] == '-', 'mean_abs_contrib'].sum() / tot)})
        top = f.sort_values('mean_abs_contrib', ascending=False).head(20)
        out[name] = {'n_rows': d['n_rows'], 'n_folds': d['n_folds'], 'what': d['what'],
                     'family_share': fam.to_dict(), 'parity_share': par.to_dict(), 'frame_share': frame.to_dict(),
                     'channel_share': ch.to_dict(),
                     'top_features': top[['name', 'family', 'channel', 'stat', 'parity', 'mean_abs_contrib']]
                     .assign(share=lambda x: x['mean_abs_contrib'] / tot).to_dict(orient='records')}
    return out


# ───────────────────────────── headline and finalists ─────────────────────────────


def choose_finalists(tab: pd.DataFrame) -> list[dict[str, Any]]:
    """The best linear, the best tree and the best ensemble pipeline (lowest outer log loss within each kind). These
    are exploratory choices made on the outer folds, so ``n_configs_tried`` = every model pipeline compared on them
    (the same number in the results, the registry, this file and the text). The nested selection (m2_select) is the
    lane's honest headline; the cross-lane synthesis re-selects among finalists nested."""
    exps = P.experiments()
    cand = tab[tab['group'].isin(['default', 'model'])]
    k = int(len(cand))
    fin = []
    for kind, models in (('linear', P.LINEAR), ('tree', P.TREES)):
        sub = cand[cand['model'].isin(models)]
        if not len(sub):
            continue
        best = sub.sort_values(['log_loss', 'name']).iloc[0]
        e = exps[best['name']]
        fin.append({'name': best['name'], 'results_json': f"results/{best['name']}.json",
                    'module': 'stage2.lanes.m2_finalists', 'build_x': 'build_x', 'factory': 'factory',
                    'params': {'model': e.model, 'mirror': e.mirror, 'scaler': e.scaler, 'winsor': e.winsor,
                               'calibrator': e.calibrator, 'n_trials': e.n_trials, 'n_splits': e.n_splits},
                    'n_configs_tried': k,
                    'notes': f'best {kind} pipeline by outer log loss among the {len(sub)} {kind} pipelines of the '
                             f'{k} model pipelines compared on the outer folds (exploratory); features '
                             'm2_features.DEFAULT_SPEC; hyperparameters tuned inside fit_predict on inner person folds'})
    ens = cand[cand['model'].isin(['ens_soft', 'ens_stack'])]
    if len(ens):
        best = ens.sort_values(['log_loss', 'name']).iloc[0]
        e = exps[best['name']]
        fin.append({'name': best['name'], 'results_json': f"results/{best['name']}.json",
                    'module': 'stage2.lanes.m2_finalists', 'build_x': 'build_x', 'factory': 'factory_ensemble',
                    'params': {'components': list(e.ensemble), 'method': e.method, 'mirror': e.mirror,
                               'n_trials': {c: P.TRIALS[c] for c in e.ensemble if c in P.TRIALS},
                               'n_splits': e.n_splits},
                    'n_configs_tried': k,
                    'notes': f'best ensemble by outer log loss among the {len(ens)} ensembles of the {k} model '
                             'pipelines compared on the outer folds (exploratory); components tuned in the fold as in '
                             'their own experiments'})
    return fin


def mirror_handedness(res: dict[str, cv.ExperimentResult], persons: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Accuracy of left- and right-handed persons (self-reported; stratification only) for every mirror variant: a
    CI only with >= MIN_PERSONS_CI persons, otherwise the per-person values."""
    hand_of = persons.set_index('person_id')['handedness_self'].astype(str)
    names = [n for n, e in P.experiments().items() if e.group == 'mirror'] + ['m2_lr_full', 'm2_lgbm']
    rows, prow = [], []
    for n in names:
        if n not in res:
            continue
        pred = res[n].predictions.assign(handedness=lambda x: x['person_id'].map(hand_of))
        for h, g in pred.groupby('handedness', sort=True):
            acc, lo, hi = _acc_ci(g)
            n_p = int(g['person_id'].nunique())
            rows.append({'name': n, 'handedness': h, 'n_trials': len(g), 'n_persons': n_p, 'accuracy': acc,
                         'acc_lo': lo, 'acc_hi': hi, 'ci': bool(n_p >= S.MIN_PERSONS_CI)})
            if n_p < S.MIN_PERSONS_CI:
                for pid, gp in g.groupby('person_id', sort=True):
                    prow.append({'name': n, 'handedness': h, 'person_id': str(pid), 'n_trials': len(gp),
                                 'accuracy': float(np.mean(cv.credit(gp['y'], gp['p_right'])))})
    return {'groups': rows, 'persons': prow}


def feature_scales(d: Any) -> dict[str, Any]:
    """Spread of the default features' standard deviations over the primary trials (no label)."""
    from stage2.lanes import m2_features as F
    X, reg = F.build_features(d, F.DEFAULT_SPEC)
    sd = X.std(0)
    pos = sd[sd > 0]
    fam = reg.assign(sd=sd).groupby('family')['sd'].median()
    return {'n_features': int(X.shape[1]), 'sd_min': float(pos.min()), 'sd_max': float(pos.max()),
            'sd_median': float(np.median(pos)), 'log10_sd_ratio': float(np.log10(pos.max() / pos.min())),
            'n_constant': int((sd == 0).sum()), 'sd_median_by_family': fam.to_dict()}


def fixed_window_record(d: Any) -> dict[str, Any]:
    """How much of the fixed 3-s window the cache froze after the app's end cue (carry-forward, ``fixed_mask``
    False): the window is not onset-only, its tail depends on the end cue."""
    mask = np.asarray(d.fixed_mask, dtype=bool)
    onset = int(round(C.FIXED_PRE_S * C.FS_HZ))
    post = ~mask[:, onset:]
    frozen = post.any(1)
    share = post.mean(1)
    dur = d.trials['duration_ms'].to_numpy(dtype=float)
    first = np.where(frozen, post.argmax(1), post.shape[1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(first[frozen], dur[frozen]).statistic) if frozen.sum() > 2 else math.nan
    return {'n_trials': int(mask.shape[0]), 'n_frozen_tail': int(frozen.sum()),
            'median_frozen_share_of_frozen': float(np.median(share[frozen])) if frozen.any() else math.nan,
            'spearman_freeze_vs_duration': rho, 'window_s': float(mask.shape[1] / C.FS_HZ),
            'pre_s': float(C.FIXED_PRE_S)}


def unscaled_record(res: dict[str, cv.ExperimentResult], tab: pd.DataFrame) -> dict[str, Any]:
    """Convergence of the unscaled logistic regression (iteration cap ``P.MAX_ITER_UNSCALED``), from its result."""
    e = next((x for x in P.experiments().values() if x.group == 'scaling' and x.scaler == 'none'), None)
    if e is None or e.name not in res:
        return {}
    conv = _convergence(res[e.name])
    return {'name': e.name, 'max_iter': e.max_iter, **conv}


def rebuild_registry(res: dict[str, cv.ExperimentResult]) -> None:
    """results/registry_m2.csv rebuilt from every saved m2 result (one row each, sorted), under the registry's
    cross-process lock (``stage2.locking``, as ``cv.update_registry``)."""
    from stage2.locking import atomic_path, file_lock
    rows = [r.summary_row() for r in res.values()]
    if rows:
        with file_lock(L.registry), atomic_path(L.registry) as tmp:
            pd.DataFrame(rows).sort_values('name').to_csv(tmp, index=False)


# ───────────────────────────── derive ─────────────────────────────


def derive() -> dict[str, Any]:
    """Write every derived file (see the module docstring) from the saved results."""
    from stage2.data import load_cache, load_tables
    from stage2.lanes import m2_features as F
    res = available()
    if not res:
        print('m2 derive: no results yet', flush=True)
        return {}
    tabs = load_tables()
    trials = tabs['trials']
    persons = tabs['persons']
    ref = cv.load_result(REFERENCE) if (C.RES_DIR / f'{REFERENCE}.json').exists() else None
    rebuild_registry(res)
    tab = experiment_table(res, ref)
    L.export_table(tab, 'm2_experiments', tex=False)
    write('m2_experiments', {'rows': tab.to_dict(orient='records')})
    reg = F.registry(F.DEFAULT_SPEC)
    L.export_table(reg, 'm2_feature_registry', tex=False)
    headline = [n for n in HEADLINE_ORDER if n in res]
    best_tree = tab[tab['model'].isin(P.TREES) & tab['group'].isin(['model'])].sort_values(['log_loss', 'name'])
    names = list(dict.fromkeys(headline + ([best_tree['name'].iloc[0]] if len(best_tree) else [])))
    write('m2_calibration', calibration_summary(res, trials, tab))
    bd, bdp = breakdown(res, trials, persons, names)
    if len(bd):
        L.export_table(bd, 'm2_strata', tex=False)
    L.export_table(bdp if len(bdp) else pd.DataFrame(columns=['model', 'factor', 'level', 'person_id', 'n_trials',
                                                               'accuracy']), 'm2_strata_persons', tex=False)
    pp = per_person(res, persons, names)
    if len(pp):
        L.export_table(pp, 'm2_per_person', tex=False)
    write('m2_k_attempts', k_attempt_analysis(res, names))
    lc = learning_curve(res)
    if len(lc):
        L.export_table(lc, 'm2_lc_runs', tex=False)
    sc = select_choices(res)
    if len(sc):
        L.export_table(sc, 'm2_select_choices', tex=False)
    write('m2_importance', importance_summary())
    fin = choose_finalists(tab)
    if fin:
        L.write_finalists(fin)
    ref_m = None
    if ref is not None:
        ref_m = {k: _f(ref.metrics.get(k)) for k in cv.METRIC_NAMES}
        ref_m.update({'accuracy_lo': ref.ci['accuracy'][0], 'accuracy_hi': ref.ci['accuracy'][1],
                      'natural_use_accuracy': _f((ref.extra_metrics or {}).get('accuracy'))})
    d = load_cache()
    fams = {}
    for col, prefix in (('family', 'vs_base_'), ('ref_family', 'vs_ref_')):
        if col not in tab:
            continue
        for f, g in tab[tab[col].notna()].groupby(col):
            size = g[f'{prefix}family_size'].max() if f'{prefix}family_size' in g else len(g)
            fams[str(f)] = {'text': family_text(str(f)), 'members': g['name'].tolist(),
                            'size': int(size) if pd.notna(size) else 0}
    summary = {'headline': names, 'finalists': fin, 'reference': REFERENCE if ref is not None else None,
               'reference_metrics': ref_m, 'mirror_handedness': mirror_handedness(res, persons),
               'feature_scales': feature_scales(d), 'unscaled': unscaled_record(res, tab),
               'fixed_window': fixed_window_record(d), 'families': fams,
               'best_by_log_loss': best_by_log_loss(tab),
               'n_experiments': len(res), 'experiments_missing': [n for n in P.experiments() if n not in res]}
    write('m2_summary', summary)
    print(f'm2 derive: {len(res)} experiment(s); headline {names}; missing {len(summary["experiments_missing"])}',
          flush=True)
    return summary
