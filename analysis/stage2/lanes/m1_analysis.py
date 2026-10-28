"""Lane m1 analyses: the feature catalogue (univariate leave-one-person-out) and every derived table the figures,
tables and numbers read. Called by the compute step only; the presentation steps never import the models.

* :func:`evaluate_catalogue` - one univariate model per catalogue feature through ``cv.run_experiment`` (one
  catalogue file holds them), the person-bootstrap CI of its AUC, the within-person AUC of every person,
  per-hand distributions and, for odd features, the accuracy of the sign rule with the fold's direction.
* :func:`derive` - from the saved results: the model table (metrics, ties, natural-use coverage, ECE against
  its perfect-calibration null, Cox calibration slope and intercept, paired comparisons with Holm / BH
  adjustment inside named families), the finalists, the per-condition strata, calibration (per platform,
  reliability with person-bootstrap bands, remedies), k-attempt aggregates, the failure analysis (confident
  errors, block clustering, candidate wrong-hand blocks), the natural-use transfer of each chiral component,
  the integration drift of the lateral feature, the in-fold hyperparameter choices against their grid edges,
  the sensitivity runs and the model artefacts (tree, temporal-filter profiles, GAM shapes, coefficients).

Everything is written with ``Lane.write_json`` (provenance embedded) under tables/m1/.
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import m1_experiments as E
from stage2.lanes import m1_features as MF
from stage2.lanes import m1_models as M
from stage2.pipeline import Lane

L = Lane('m1')
#: the baseline of the data-treatment comparisons (PHYS6) and the default focus before finalists exist
FOCUS = E.BASELINE
#: models whose strata are derived (besides the finalists)
STRATA_MODELS = (FOCUS, 'm1_lr_phys6_win_asinh', 'm1_rule_fhy_sign', 'm1_rule_fhy_sign_cal', 'm1_rule_fhy_threshold',
                 'm1_rule_nested', 'm1_gam_phys6', 'm1_tree', 'm1_tf_multi')
#: models whose calibration is derived (besides the finalists): every remedy base, its remedies, the treatments
CAL_MODELS = tuple(dict.fromkeys(
    [b for b in E.CAL_BASES] + [s.name for s in E.SPECS if s.group == 'calibration']
    + ['m1_lr_phys6_win_asinh', 'm1_rule_fhy_sign_cal', 'm1_tree', 'm1_lr_fwd', E.REFERENCE]))
#: models whose k-attempt aggregates are derived (they carry the in-fold aggregation exponent)
AGG_MODELS = (FOCUS, 'm1_lr_phys6_nested', 'm1_lr_phys6_win_asinh', 'm1_gam_phys6', 'm1_tf_multi')
#: the treatment comparisons of the calibration chain: (treated model, untreated model)
TREATMENT_REMEDIES = (('m1_lr_phys6_win_asinh', FOCUS), ('m1_lr_phys6_nested', FOCUS))
#: "what do fitted numbers add to the rule?": each fitted anchor paired against the rule
VS_RULE = (FOCUS, 'm1_lr_phys6_win_asinh', 'm1_lr_phys6_nested', 'm1_gam_phys6', 'm1_tf_multi')
CONFIDENT = 0.2            # a confident error: P(true hand) < 0.2, i.e. >= 80 % for the wrong hand
KEY_FEATURES = ('yaw_q50', 'yaw_q100', 'latvel_q75', 'roll_q25', 'duration_s', 'peak_gyro')
#: natural-use transfer: the chiral features whose sign is checked on the free-choice trials
NAT_FEATURES = ('yaw_q50', 'yaw_q100', 'roll_q50', 'roll_q100', 'latvel_q50', 'latvel_q75', 'latvel_q100',
                'latvel_dd_q50', 'latvel_dd_q100')
COMPONENTS = {'yaw': 'yaw_', 'roll': 'roll_', 'lateral': 'latvel_'}
STRATA = ('platform', 'instructed_rotation', 'speed', 'variant', 'start_position', 'posture', 'tier', 'hand',
          'handedness', 'duration_bin', 'grip_mismatch', 'false_start', 'clip_candidate', 'seg_disagree_100ms',
          'repeated_samples', 'gate_pass')
N_DURATION_BINS = 4
MIN_PERSONS_CI = 5         # strata with fewer persons get no bootstrap CI (cross-lane rule)
N_SIM_NULL = 2000          # simulations of the perfect-calibration null of the ECE
IDENTICAL_DP = 1e-6        # two models whose P differ by less than this everywhere are 'identical' (solver precision)
N_PERM_BLOCKS = 2000       # within-person permutations of the block-clustering test
N_DRAWS_AGG = 200          # random k-subsets per (person, hand) cell of the aggregation analysis
#: the catalogue runs one univariate rule per feature (dozens): seeded leak checks on a sample of folds (a
#: direction and a scale fitted on the training rows only); its CIs use statistic-only person bootstraps
CATALOGUE_LEAK_CHECK = 'sample'


def _clean(v: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None (Lane.write_json refuses NaN), numpy scalars -> Python."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def _write(name: str, obj: dict[str, Any]) -> None:
    L.write_json(name, _clean(obj))


def swap_blocks(d: Any) -> list[tuple[str, int]]:
    """The swap-signature blocks of ``stage2.screen.block_consistency`` (two adjacent blocks of opposite instructed
    hands in which every trial's first-half yaw sign contradicts the label): data-driven, never a typed list."""
    from stage2 import screen
    _, blocks = screen.block_consistency(d.trials, d.approach)
    return [(str(r['session_label']), int(r['block_index'])) for _, r in blocks[blocks['fhy_swap_signature']].iterrows()]


_CACHE: dict[str, cv.ExperimentResult | None] = {}


def _load(name: str) -> cv.ExperimentResult | None:
    if name not in _CACHE:
        try:
            _CACHE[name] = cv.load_result(name)
        except FileNotFoundError:
            _CACHE[name] = None
    return _CACHE[name]


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) >= 0.5).astype(int) == np.asarray(y).astype(int))) if len(y) else math.nan


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return cv.roc_auc(y, p)


def _boot_sets(persons: np.ndarray, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> list[np.ndarray]:
    """Row indices of ``n_boot`` person-bootstrap resamples (all rows of a drawn person come along)."""
    persons = np.asarray(persons).astype(str)
    uniq = np.unique(persons)
    rows = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows[u] for u in rng.choice(uniq, uniq.size, replace=True)]) for _ in range(n_boot)]


def _q(v: Sequence[float]) -> tuple[float, float]:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v.size else (math.nan, math.nan)


# ───────────────────────────── multiplicity ─────────────────────────────


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p values (NaN stays NaN; the family is the finite entries)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    run = 0.0
    for rank, i in enumerate(ok[np.argsort(p[ok], kind='stable')]):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def bh(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if not m:
        return out
    order = ok[np.argsort(p[ok], kind='stable')]
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


# ───────────────────────────── calibration statistics ─────────────────────────────


def _bins(p: np.ndarray, n_bins: int = C.ECE_BINS) -> np.ndarray:
    """Equal-mass bin of every prediction, exactly as ``cv.reliability`` (tied predictions share a bin)."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    return np.searchsorted(edges[1:-1], p, side='right')


def ece_null(p: np.ndarray, n_sim: int = N_SIM_NULL, seed: int = C.SEED) -> np.ndarray:
    """ECEs of ``n_sim`` label sets drawn as y* ~ Bernoulli(p) given the model's own P (consistency resampling,
    Broecker and Smith 2007): the ECE a perfectly calibrated model would show on these predictions. Trials are
    drawn independently given P (no person clustering)."""
    p = np.asarray(p, dtype=float)
    k = _bins(p)
    rng = np.random.default_rng(seed)
    Y = rng.random((n_sim, p.size)) < p
    out = np.zeros(n_sim)
    for b in np.unique(k):
        m = k == b
        out += m.sum() / p.size * np.abs(Y[:, m].mean(axis=1) - p[m].mean())
    return out


def ece_test(y: np.ndarray, p: np.ndarray, n_sim: int = N_SIM_NULL, seed: int = C.SEED) -> dict[str, float]:
    """The ECE with its perfect-calibration null: mean, 95th percentile and the one-sided p of the observed ECE."""
    e = cv.ece(y, p)
    null = ece_null(p, n_sim, seed)
    return {'ece': e, 'ece_null_mean': float(null.mean()), 'ece_null_q95': float(np.quantile(null, 0.95)),
            'ece_p': float((1 + np.sum(null >= e - 1e-12)) / (1 + null.size))}


def cox(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(intercept, slope) of the logistic recalibration y ~ a + b logit(P) (Cox 1958): b < 1 over-confident,
    b > 1 under-confident; a != 0 a shift towards one hand. Newton iterations, deterministic."""
    y = np.asarray(y, dtype=float)
    z = M.logit(np.asarray(p, dtype=float))
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(100):
        mu = M.sigmoid(X @ beta)
        w = np.maximum(mu * (1 - mu), 1e-12)
        H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def cox_ci(pred: pd.DataFrame, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> dict[str, float]:
    """Cox intercept and slope with person-bootstrap 95 % CIs."""
    y = pred['y'].to_numpy(dtype=float)
    p = pred['p_right'].to_numpy(dtype=float)
    a, b = cox(y, p)
    A, B = [], []
    for sel in _boot_sets(pred['person_id'].to_numpy(), n_boot, seed):
        try:
            aa, bb = cox(y[sel], p[sel])
        except np.linalg.LinAlgError:
            continue
        A.append(aa)
        B.append(bb)
    (alo, ahi), (blo, bhi) = _q(A), _q(B)
    return {'cox_intercept': a, 'cox_intercept_lo': alo, 'cox_intercept_hi': ahi, 'cox_slope': b,
            'cox_slope_lo': blo, 'cox_slope_hi': bhi}


def lean(rel: pd.DataFrame) -> float:
    """Signed miscalibration over reliability bins: mean of (observed share - P) towards the side the prediction
    leans to, weighted by n; > 0 under-confident (outcomes more extreme than P), < 0 over-confident."""
    s = np.sign(rel['p_mean'] - 0.5)
    return float((rel['n'] * (rel['y_mean'] - rel['p_mean']) * s).sum() / rel['n'].sum())


# ───────────────────────────── 1. the catalogue ─────────────────────────────


def evaluate_catalogue(d: Any, nat: Any, F: pd.DataFrame, Fn: pd.DataFrame, *, verbose: bool = True) -> dict[str, Any]:
    """Univariate leave-one-person-out evaluation of every catalogue feature: the direction-only rule
    (``m1_models.make_direction``: direction and scale from the training persons, no intercept) through
    ``cv.run_experiment``; AUC and sign-rule accuracy with person-bootstrap CIs (``C.N_BOOT`` resamples of persons,
    statistic-only bootstraps for speed), within-person AUC of every person, per-hand distributions."""
    y = d.y
    persons = d.groups
    rows, within = [], {}
    for f in MF.CATALOGUE:
        fp = M.make_direction(f.name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = cv.run_experiment(f'm1_uni_{f.name}', fp, F[[f.name]], d.trials, extra=(Fn[[f.name]], nat.trials),
                                  save=False, registry=None, notes='m1 catalogue (direction-only rule)',
                                  leak_check=CATALOGUE_LEAK_CHECK, n_boot=0)
        pred = r.predictions.set_index('trial_id').loc[d.trials['trial_id']].reset_index()
        dirs = pred['params'].map(lambda v: json.loads(v)['direction'])
        pp = {}
        for pid in sorted(set(persons.tolist())):
            m = persons == pid
            pp[pid] = _auc(y[m], pred['p_right'].to_numpy()[m])
        within[f.name] = pp
        w = np.array([v for v in pp.values() if np.isfinite(v)])
        x = F[f.name].to_numpy(dtype=float)
        pr = pred[['person_id', 'y', 'p_right']]
        auc_lo, auc_hi = cv.person_bootstrap(pr, fn=_auc)['stat']
        row = {**{k: getattr(f, k) for k in ('name', 'family', 'parity', 'unit', 'window', 'frame', 'flips', 'grip',
                                              'formula', 'meaning')},
               'auc': r.metrics['roc_auc'], 'auc_lo': auc_lo, 'auc_hi': auc_hi,
               'natural_use_share_right': (r.extra_metrics or {}).get('accuracy') if f.parity == 'odd' else None,
               'direction_mean': float(dirs.mean()), 'direction_consistent': bool(dirs.nunique() == 1),
               'within_auc_median': float(np.median(w)), 'within_auc_min': float(np.min(w)),
               'share_persons_above_50': float(np.mean(w > 0.5)), 'share_persons_ge_75': float(np.mean(w >= 0.75)),
               'n_persons': int(w.size), 'leak_check': r.leak_check}
        for hv, hn in ((0, 'left'), (1, 'right')):
            xv = x[y == hv]
            row.update({f'{hn}_median': float(np.median(xv)), f'{hn}_q25': float(np.quantile(xv, 0.25)),
                        f'{hn}_q75': float(np.quantile(xv, 0.75)), f'{hn}_mean': float(xv.mean())})
        if f.parity == 'odd':               # p >= 0.5 iff s * x >= 0: the decision is the sign rule
            row['sign_rule_accuracy'] = r.metrics['accuracy']
            row['sign_rule_accuracy_lo'], row['sign_rule_accuracy_hi'] = cv.person_bootstrap(pr, fn=_acc)['stat']
        rows.append(row)
        if verbose:
            print(f'  catalogue {f.name:22s} AUC {row["auc"]:.3f} [{auc_lo:.3f}, {auc_hi:.3f}] '
                  f'persons>0.5 {row["share_persons_above_50"]:.2f}', flush=True)
    out = {'rows': rows, 'within_person_auc': within, 'n_trials': int(len(y)), 'n_persons': int(len(set(persons))),
           'n_features': len(rows), 'n_odd': int(sum(r['parity'] == 'odd' for r in rows)),
           'model': 'direction-only rule (m1_models.make_direction)', 'n_boot': C.N_BOOT,
           'leak_check': CATALOGUE_LEAK_CHECK}
    _write('m1_catalogue', out)
    return out


# ───────────────────────────── 2. the model table ─────────────────────────────


def _paired(a: cv.ExperimentResult, b: cv.ExperimentResult) -> dict[str, Any]:
    """A vs B on the same trials: cv.paired_compare (pooled accuracy, per-person log loss) plus the per-person
    Brier difference (paired person bootstrap, sign-flip p), the pooled ECE difference (point only) and whether the
    two models are identical within solver precision."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c = cv.paired_compare(a, b)
    j = a.predictions[['trial_id', 'person_id', 'y', 'p_right']].merge(b.predictions[['trial_id', 'p_right']],
                                                                         on='trial_id', suffixes=('_a', '_b'))
    y = j['y'].to_numpy(dtype=float)
    pa, pb = j['p_right_a'].to_numpy(dtype=float), j['p_right_b'].to_numpy(dtype=float)
    db = (pa - y) ** 2 - (pb - y) ** 2
    per = pd.Series(db).groupby(j['person_id'].astype(str).to_numpy()).mean()
    lo, hi = _q(per.to_numpy()[np.random.default_rng(C.SEED).integers(0, per.size, (C.N_BOOT, per.size))].mean(axis=1))
    maxdp = float(np.max(np.abs(pa - pb))) if len(j) else math.nan
    return {'acc_diff': c['pooled_acc_diff'], 'acc_diff_lo': c['pooled_acc_diff_ci95'][0],
            'acc_diff_hi': c['pooled_acc_diff_ci95'][1], 'p_acc': c['p_sign_flip_acc'],
            'll_diff': c['mean_person_logloss_diff'], 'll_diff_lo': c['mean_person_logloss_diff_ci95'][0],
            'll_diff_hi': c['mean_person_logloss_diff_ci95'][1], 'p_ll': c['p_sign_flip_logloss'],
            'brier_diff': float(per.mean()), 'brier_diff_lo': lo, 'brier_diff_hi': hi,
            'p_brier': cv._sign_flip_p(per.to_numpy()),
            'ece_diff': cv.ece(y, pa) - cv.ece(y, pb), 'max_abs_dp': maxdp, 'identical': bool(maxdp < IDENTICAL_DP),
            'persons_better': c['persons_a_better'], 'persons_worse': c['persons_b_better'],
            'n_persons': c['n_persons']}


def _ties(p: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(p, dtype=float) - 0.5) < 1e-12


def _nat_metrics(r: cv.ExperimentResult) -> dict[str, Any]:
    """Natural use with the tie rule: a P(right) of exactly 0.5 is an abstention, never a right-hand answer.
    coverage = share answered; answered accuracy; half credit (a tie counts 1/2: the headline, and the harness's rule
    since 2026-10-28); forced choice (p >= 0.5 is right, the harness's rule before: optimistic on an all-right set)."""
    ex = r.extra_predictions
    if ex is None or not len(ex):
        return {'nat_n': 0}
    y = ex['y'].to_numpy(dtype=int)
    p = ex['p_right'].to_numpy(dtype=float)
    t = _ties(p)
    ok = ((p >= 0.5).astype(int) == y) & ~t
    return {'nat_n': int(y.size), 'nat_n_right': int(y.sum()), 'nat_ties': int(t.sum()),
            'nat_coverage': float(np.mean(~t)), 'nat_correct': int(ok.sum()),
            'nat_acc_answered': float(ok.sum() / max((~t).sum(), 1)),
            'nat_acc_half': float((ok.sum() + 0.5 * t.sum()) / y.size),
            'nat_acc_forced': float(np.mean((p >= 0.5).astype(int) == y))}


def _grid_of(s: E.Spec) -> dict[str, list[Any]]:
    """The in-fold hyperparameter grid of an experiment (for the grid-edge summary)."""
    p = s.params.get('params', s.params) if s.factory == 'make_train_filter' else s.params
    fac = s.params.get('factory', s.factory) if s.factory == 'make_train_filter' else s.factory
    if fac in ('make_linear', 'make_forward', 'make_linear_select'):
        return {'C': list(p.get('Cs', M.CS_DEFAULT))}
    if fac == 'make_gam':
        return {'C': list(p.get('Cs', M.GAM_CS)), 'n_knots': list(p.get('knots', M.GAM_KNOTS))}
    if fac == 'make_tree':
        return {'depth': list(p.get('depths', M.TREE_DEPTHS)), 'min_leaf': list(p.get('leafs', M.TREE_LEAFS))}
    return {}


def model_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per saved m1 experiment (and the reference): metrics and CIs (no percentile CI for the ECE: its
    perfect-calibration null instead), ties and natural use, the k-attempt curve, Cox calibration, the paired
    comparisons against the reference, the experiment's baseline, the rule and the family anchor, each with Holm-
    and BH-adjusted p values inside a named family. Returns (table, family descriptions)."""
    ref = _load(E.REFERENCE)
    rule = _load(E.RULE)
    rows = []
    for s in list(E.SPECS) + [None]:
        name = s.name if s else E.REFERENCE
        r = _load(name)
        if r is None:
            continue
        kc = r.k_curve.set_index('k')
        p = r.predictions['p_right'].to_numpy(dtype=float)
        y = r.predictions['y'].to_numpy(dtype=int)
        t = _ties(p)
        row = {'name': name, 'label': s.label if s else 'reference: yaw logreg (8 instants)',
               'group': s.group if s else 'reference', 'design': s.design if s else 'pre-specified',
               'nested': bool(s.nested) if s else False, 'factory': s.factory if s else 'yaw_logreg',
               'n_configs_tried': int(r.info.get('n_configs_tried', 1)), 'n_trials': int(r.metrics['n']),
               'n_persons': int(r.metrics.get('n_persons', r.predictions['person_id'].nunique())),
               'leak_check': r.leak_check, 'notes': r.info.get('notes', ''), 'baseline': s.baseline if s else None,
               'n_ties': int(t.sum()),
               'accuracy_half': float((((p >= 0.5).astype(int) == y) & ~t).sum() / y.size + 0.5 * t.mean())}
        for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'log_loss', 'brier', 'ece'):
            row[k] = r.metrics[k]
            if k != 'ece':
                lo, hi = r.ci.get(k, (math.nan, math.nan))
                row[f'{k}_lo'], row[f'{k}_hi'] = lo, hi
        hard = bool(np.all((p == 0) | (p == 1)))            # a hard 0/1 rule: no finite log loss, no calibration
        row['hard'] = hard
        if not hard:
            pr = r.predictions[['person_id', 'y', 'p_right']]
            row.update(ece_test(y, p))
            row.update(cox_ci(pr))
            row['lean'] = lean(cv.reliability(y, p))
        row.update({'person_acc_median': r.metrics.get('person_acc_median'),
                    'n_persons_below_80': r.metrics.get('n_persons_below_80'),
                    'gated_accuracy': (r.gated_metrics or {}).get('accuracy'), **_nat_metrics(r)})
        for k in sorted(kc.index):
            row[f'acc_k{k}'], row[f'acc_k{k}_lo'], row[f'acc_k{k}_hi'] = kc.loc[k, 'accuracy'], kc.loc[k, 'lo'], kc.loc[k, 'hi']
        row['k_n_persons'] = int(kc['n_persons'].iloc[0]) if len(kc) else 0
        if ref is not None and name != E.REFERENCE:
            row.update({f'vs_ref_{k}': v for k, v in _paired(r, ref).items()})
        base = s.baseline if s else None
        if base and _load(base) is not None:
            row.update({f'vs_base_{k}': v for k, v in _paired(r, _load(base)).items()})
            row['vs_base_family'] = (f'{base}:remedies' if s.group == 'calibration' else
                                     f'{base}:sensitivity' if s.group == 'sensitivity' else
                                     f'{base}:rules' if s.group == 'rules' else f'{base}:options')
        if rule is not None and name in VS_RULE:
            row.update({f'vs_rule_{k}': v for k, v in _paired(r, rule).items()})
            row['vs_rule_family'] = 'vs_rule'
        fam = next((f for f, v in E.FAMILIES.items() if name in v), None)
        anchor = E.FAMILY_ANCHOR.get(fam) if fam else None
        if anchor and anchor != name and _load(anchor) is not None:
            row.update({f'vs_anchor_{k}': v for k, v in _paired(r, _load(anchor)).items()})
            row['vs_anchor_family'] = f'{fam}:vs_{anchor}'
            row['anchor'] = anchor
        rows.append(row)
    mt = pd.DataFrame(rows)
    fams: dict[str, Any] = {}
    if 'vs_ref_p_acc' in mt:
        mt['vs_ref_family'] = np.where(mt['name'] != E.REFERENCE, 'vs_reference', None)
    for pre in ('vs_ref', 'vs_base', 'vs_rule', 'vs_anchor'):
        if f'{pre}_family' not in mt:
            continue
        for fam, g in mt.groupby(f'{pre}_family', dropna=True):
            for met in ('acc', 'll', 'brier'):
                col = f'{pre}_p_{met}'
                if col not in g:
                    continue
                mt.loc[g.index, f'{col}_holm'] = holm(g[col].to_numpy(dtype=float))
                mt.loc[g.index, f'{col}_bh'] = bh(g[col].to_numpy(dtype=float))
            fams[str(fam)] = {'prefix': pre, 'size': int(len(g)), 'members': g['name'].tolist()}
    return mt, fams


# ───────────────────────────── 3. strata ─────────────────────────────


def _trial_frame(d: Any, F: pd.DataFrame) -> pd.DataFrame:
    t = d.trials.reset_index(drop=True)
    persons = d.persons.set_index('person_id')
    hd = t['person_id'].map(persons['handedness_self']).fillna('unknown').astype(str)
    q = np.quantile(F['duration_s'], np.linspace(0, 1, N_DURATION_BINS + 1))
    bins = np.clip(np.searchsorted(q[1:-1], F['duration_s'].to_numpy(), side='right'), 0, N_DURATION_BINS - 1)
    lab = [f'{q[i]:.2f}-{q[i + 1]:.2f} s' for i in range(N_DURATION_BINS)]
    out = t[['trial_id', 'trial_key', 'person_id', 'session_label', 'block_index', 'platform', 'instructed_rotation',
             'speed', 'variant', 'start_position', 'posture', 'tier', 'hand', 'grip_mismatch', 'false_start',
             'clip_candidate', 'seg_disagree_100ms', 'repeated_samples', 'gate_pass', 'device_model']].copy()
    out['tier'] = out['tier'].astype('Int64').astype(str)
    out['handedness'] = hd.to_numpy()
    out['duration_bin'] = [lab[i] for i in bins]
    out['duration_bin_index'] = bins + 1
    out['duration_s'] = F['duration_s'].to_numpy()
    for c in ('grip_mismatch', 'false_start', 'clip_candidate', 'seg_disagree_100ms', 'repeated_samples', 'gate_pass'):
        out[c] = out[c].astype(bool).map({True: 'yes', False: 'no'})
    return out


def strata_table(r: cv.ExperimentResult, tf: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per level of every stratum (after prediction); a person-bootstrap 95 % CI only for levels with at
    least MIN_PERSONS_CI persons."""
    pred = r.predictions[['trial_id', 'person_id', 'y', 'p_right']].merge(tf.drop(columns=['person_id']), on='trial_id')
    rows = []
    for s in STRATA:
        for lev, g in pred.groupby(s, sort=True):
            np_ = g['person_id'].nunique()
            acc = _acc(g['y'].to_numpy(), g['p_right'].to_numpy())
            lo, hi = (cv.person_bootstrap(g, fn=_acc)['stat'] if np_ >= MIN_PERSONS_CI else (math.nan, math.nan))
            ll = float(-np.mean(np.log(np.clip(np.where(g['y'] == 1, g['p_right'], 1 - g['p_right']), 1e-15, 1))))
            rows.append({'model': r.name, 'stratum': s, 'level': str(lev), 'n_trials': int(len(g)), 'n_persons': int(np_),
                         'n_correct': int(round(acc * len(g))), 'accuracy': acc, 'accuracy_lo': lo, 'accuracy_hi': hi,
                         'log_loss': ll,
                         'level_index': int(g['duration_bin_index'].iloc[0]) if s == 'duration_bin' else None})
    return pd.DataFrame(rows)


# ───────────────────────────── 4. calibration ─────────────────────────────


def reliability_bands(pred: pd.DataFrame, n_bins: int = C.ECE_BINS, n_boot: int = C.N_BOOT, seed: int = C.SEED) -> pd.DataFrame:
    """Equal-mass reliability bins (edges from all trials) with person-bootstrap 95 % bands of the observed share,
    whether the band excludes the bin's mean prediction, and the direction of the miss."""
    y = pred['y'].to_numpy(dtype=float)
    p = pred['p_right'].to_numpy(dtype=float)
    k = _bins(p, n_bins)
    sets = _boot_sets(pred['person_id'].to_numpy(), n_boot, seed)
    boot = np.full((len(sets), n_bins), np.nan)
    for b, sel in enumerate(sets):
        kk, yy = k[sel], y[sel]
        for i in range(n_bins):
            m = kk == i
            if m.any():
                boot[b, i] = yy[m].mean()
    rows = []
    for i in range(n_bins):
        m = k == i
        if not m.any():
            continue
        pm, ym = float(p[m].mean()), float(y[m].mean())
        lo, hi = float(np.nanquantile(boot[:, i], 0.025)), float(np.nanquantile(boot[:, i], 0.975))
        out = bool(lo > pm or hi < pm)
        miss = 'none' if not out else ('under' if (ym - pm) * np.sign(pm - 0.5) > 0 else 'over')
        rows.append({'bin': i, 'p_mean': pm, 'p_lo': float(p[m].min()), 'p_hi': float(p[m].max()), 'y_mean': ym,
                     'y_lo': lo, 'y_hi': hi, 'n': int(m.sum()), 'outside': out, 'miss': miss})
    return pd.DataFrame(rows)


def _metric_boot(pred: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return cv.person_bootstrap(pred, names=('accuracy', 'roc_auc', 'log_loss', 'brier'))


def calibration_tables(results: dict[str, cv.ExperimentResult], tf: pd.DataFrame) -> dict[str, Any]:
    """Per model and scope (all, per platform): accuracy, AUC, log loss, Brier with person-bootstrap CIs; the ECE
    with its perfect-calibration null; Cox intercept and slope with CIs; reliability bins with bands (all trials);
    the paired comparisons of every remedy and of the treatments against the untreated model; the in-fold
    temperatures."""
    rows, rel = [], []
    plat = tf.set_index('trial_id')['platform']
    for name, r in results.items():
        pred = r.predictions[['trial_id', 'person_id', 'y', 'p_right']].copy()
        pred['platform'] = pred['trial_id'].map(plat).to_numpy()
        if np.all((pred['p_right'] == 0) | (pred['p_right'] == 1)):
            continue
        for scope, g in [('all', pred)] + [(pl, gg) for pl, gg in pred.groupby('platform', sort=True)]:
            m = cv.metrics(g['y'], g['p_right'])
            ci = _metric_boot(g)
            row = {'model': name, 'scope': scope, 'n_trials': int(len(g)), 'n_persons': int(g['person_id'].nunique())}
            for k in ('accuracy', 'roc_auc', 'log_loss', 'brier'):
                row[k], row[f'{k}_lo'], row[f'{k}_hi'] = m[k], *ci.get(k, (math.nan, math.nan))
            row.update(ece_test(g['y'].to_numpy(), g['p_right'].to_numpy()))
            row.update(cox_ci(g))
            rb = reliability_bands(g)
            row.update({'bins_outside': int(rb['outside'].sum()), 'bins_under': int((rb['miss'] == 'under').sum()),
                        'bins_over': int((rb['miss'] == 'over').sum()), 'n_bins': int(len(rb)), 'lean': lean(rb)})
            rows.append(row)
            if scope == 'all':
                rel.append(rb.assign(model=name))
    paired = []
    for name, r in results.items():
        s = E.BY_NAME.get(name)
        if s is not None and s.group == 'calibration' and s.baseline in results:
            paired.append({'model': name, 'baseline': s.baseline, 'kind': 'remedy', 'family': f'{s.baseline}:remedies',
                           **_paired(r, results[s.baseline])})
    for a, b in TREATMENT_REMEDIES:
        if a in results and b in results:
            paired.append({'model': a, 'baseline': b, 'kind': 'treatment', 'family': f'{b}:treatments',
                           **_paired(results[a], results[b])})
    pc = pd.DataFrame(paired)
    if len(pc):
        for fam, g in pc.groupby('family'):
            for met in ('ll', 'brier', 'acc'):
                pc.loc[g.index, f'p_{met}_holm'] = holm(g[f'p_{met}'].to_numpy(dtype=float))
    temps = {}
    for name, r in results.items():
        s = E.BY_NAME.get(name)
        if s is not None and s.group == 'calibration' and s.params.get('calibration') == 'temperature':
            T = r.predictions.groupby('fold')['params'].first().map(lambda v: json.loads(v).get('calibration', {}).get('T'))
            T = T.dropna().astype(float)
            temps[name] = {'base': s.baseline, 'T_per_fold': T.tolist(), 'n_folds': int(T.size),
                           'n_below_1': int((T < 1).sum()), 'T_min': float(T.min()), 'T_max': float(T.max()),
                           'T_median': float(T.median())}
    return {'metrics': pd.DataFrame(rows).to_dict(orient='records'),
            'reliability': pd.concat(rel, ignore_index=True).to_dict(orient='records') if rel else [],
            'paired': pc.to_dict(orient='records'), 'temperatures': temps,
            'n_sim_null': N_SIM_NULL, 'n_boot': C.N_BOOT, 'ece_bins': C.ECE_BINS}


# ───────────────────────────── 5. k-attempt aggregates ─────────────────────────────

AGG_METHODS = ('mean_p', 'mean_logit', 'sum_logit', 'fitted_alpha')


def _light(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """accuracy, log loss and Brier score (as cv.metrics computes them), fast enough for bootstraps."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, cv._EPS, 1 - cv._EPS)
    return {'accuracy': float(np.mean((p >= 0.5) == (y == 1))),
            'log_loss': float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            'brier': float(np.mean((p - y) ** 2))}


def k_aggregation(r: cv.ExperimentResult, ks: Sequence[int] = C.K_ATTEMPTS, n_draws: int = N_DRAWS_AGG,
                  seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate k attempts of one person and hand (the persons with >= max(ks) trials per hand, as the harness's
    k-attempt curve): mean probability, mean logit (alpha = 1), summed logits (alpha = 0) or k**(-alpha) * sum with
    the alpha fitted on the training persons of that fold (``info.agg_alpha``). Returns (metrics per k and method:
    accuracy, log loss and Brier with person-bootstrap CIs, ECE as a point; reliability bins of k = 3; paired
    fitted-alpha minus sum-of-logits differences of log loss and ECE per k on the same draws)."""
    pred = r.predictions.copy()
    alpha = pred['params'].map(lambda s: json.loads(s).get('agg_alpha', np.nan))
    pred['alpha'] = alpha.to_numpy(dtype=float)
    pred['z'] = M.logit(pred['p_right'].to_numpy())
    kmax = max(ks)
    cnt = pred.groupby(['person_id', 'y']).size().unstack(fill_value=0)
    ok = set(cnt.index[(cnt.get(0, 0) >= kmax) & (cnt.get(1, 0) >= kmax)].astype(str))
    rng = np.random.default_rng(seed)
    draws = []
    for (pid, yy), g in pred.groupby(['person_id', 'y'], sort=True):
        if str(pid) not in ok:
            continue
        z = g['z'].to_numpy()
        p = g['p_right'].to_numpy()
        a = float(g['alpha'].iloc[0])
        for k in ks:
            for _ in range(n_draws):
                s = rng.choice(z.size, size=k, replace=False)
                draws.append({'person_id': pid, 'y': int(yy), 'k': k, 'mean_p': p[s].mean(),
                              'mean_logit': M.sigmoid(z[s].mean()), 'sum_logit': M.sigmoid(z[s].sum()),
                              'fitted_alpha': M.sigmoid(z[s].sum() / k ** a) if np.isfinite(a) else np.nan,
                              'alpha': a})
    D = pd.DataFrame(draws)
    rows, rel, pair = [], [], []
    for k, g in D.groupby('k', sort=True):
        sets = _boot_sets(g['person_id'].to_numpy(), C.N_BOOT, seed + int(k))
        yv = g['y'].to_numpy()
        for meth in AGG_METHODS:
            if g[meth].isna().all():
                continue
            pv = g[meth].to_numpy(dtype=float)
            m = cv.metrics(yv, pv)
            row = {'model': r.name, 'k': int(k), 'method': meth, 'n_draws': int(len(g)),
                   'n_persons': int(g['person_id'].nunique()), 'ece': m['ece']}
            boot = [_light(yv[sel], pv[sel]) for sel in sets]
            for key in ('accuracy', 'log_loss', 'brier'):
                row[key], (row[f'{key}_lo'], row[f'{key}_hi']) = m[key], _q([b[key] for b in boot])
            rb = cv.reliability(yv, pv)
            row['lean'] = lean(rb)
            row['alpha_median'] = float(np.nanmedian(g['alpha'])) if meth == 'fitted_alpha' else math.nan
            rows.append(row)
            if k == 3:
                rel.append(rb.assign(model=r.name, method=meth))
        if not g['fitted_alpha'].isna().all():
            pa, ps = g['fitted_alpha'].to_numpy(dtype=float), g['sum_logit'].to_numpy(dtype=float)

            def ll(y_, p_):
                return _light(y_, p_)['log_loss']
            dll = [ll(yv[sel], pa[sel]) - ll(yv[sel], ps[sel]) for sel in sets]
            dece = [cv.ece(yv[sel], pa[sel]) - cv.ece(yv[sel], ps[sel]) for sel in sets]
            (a1, b1), (a2, b2) = _q(dll), _q(dece)
            pair.append({'model': r.name, 'k': int(k), 'll_diff': ll(yv, pa) - ll(yv, ps), 'll_diff_lo': a1,
                         'll_diff_hi': b1, 'ece_diff': cv.ece(yv, pa) - cv.ece(yv, ps), 'ece_diff_lo': a2,
                         'ece_diff_hi': b2})
    return pd.DataFrame(rows), (pd.concat(rel, ignore_index=True) if rel else pd.DataFrame()), pd.DataFrame(pair)


# ───────────────────────────── 6. failures ─────────────────────────────


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def block_clustering(pred: pd.DataFrame, n_perm: int = N_PERM_BLOCKS, seed: int = C.SEED) -> dict[str, Any]:
    """Do confident errors cluster in blocks beyond what their number per person implies? Statistic: the number of
    distinct blocks holding at least one confident error (fewer = more clustered); null: the confident-error flags
    permuted among the trials of the same person (keeps each person's count). One-sided p of the observed value
    being as low or lower."""
    key = pred['session_label'].astype(str) + '#' + pred['block_index'].astype(str)
    flags = pred['confident_error'].to_numpy(dtype=bool)
    persons = pred['person_id'].astype(str).to_numpy()
    blk = key.to_numpy()
    obs = len(set(blk[flags]))
    rng = np.random.default_rng(seed)
    idx = {p: np.flatnonzero(persons == p) for p in np.unique(persons)}
    null = np.zeros(n_perm, dtype=int)
    for b in range(n_perm):
        f = np.zeros_like(flags)
        for p, ix in idx.items():
            f[ix] = rng.permutation(flags[ix])
        null[b] = len(set(blk[f]))
    return {'n_confident': int(flags.sum()), 'n_blocks': int(len(set(blk))), 'blocks_with_confident': int(obs),
            'null_mean': float(null.mean()), 'null_q05': float(np.quantile(null, 0.05)),
            'p_clustered': float((1 + np.sum(null <= obs)) / (1 + n_perm)), 'n_perm': n_perm}


def failures(r: cv.ExperimentResult, d: Any, tf: pd.DataFrame, F: pd.DataFrame) -> dict[str, Any]:
    """Per person (with platform, handedness and per-hand accuracy), confident errors and where they concentrate
    (counts, Wilson CI of the share), block clustering of confident errors, candidate wrong-hand blocks (at least 2
    trials and a mean P(instructed hand) below CONFIDENT), the swap-signature blocks, and the class-mean yaw rate of
    confident errors against correct trials (descriptive: the errors are selected on that evidence)."""
    pred = r.predictions[['trial_id', 'person_id', 'y', 'p_right']].merge(tf.drop(columns=['person_id']), on='trial_id')
    pred['p_label'] = np.where(pred['y'] == 1, pred['p_right'], 1 - pred['p_right'])
    pred['correct'] = (pred['p_right'] >= 0.5).astype(int) == pred['y']          # forced choice (no ties here)
    pred['confident_error'] = pred['p_label'] < CONFIDENT
    pp = r.per_person.copy()
    meta = tf.drop_duplicates('person_id').set_index('person_id')[['platform', 'handedness']]
    pp = pp.join(meta, on='person_id')
    pp['n_confident_errors'] = pp['person_id'].map(pred.groupby('person_id')['confident_error'].sum()).astype(int)
    pp['rank_lowest'] = pp['accuracy'].rank(method='min').astype(int)
    swap = swap_blocks(d)
    pred['block_index'] = pd.to_numeric(pred['block_index'], errors='coerce').fillna(-1).astype(int)
    key = list(zip(pred['session_label'].astype(str), pred['block_index']))
    pred['swap_block'] = [k in set(swap) for k in key]
    blk = (pred.groupby(['person_id', 'session_label', 'block_index'], sort=True)
           .agg(hand=('hand', 'first'), grip=('instructed_rotation', 'first'), n=('y', 'size'),
                n_correct=('correct', 'sum'), accuracy=('correct', 'mean'), mean_p_label=('p_label', 'mean'),
                n_confident=('confident_error', 'sum'), swap_block=('swap_block', 'first'))
           .reset_index())
    blk['candidate'] = (blk['n'] >= 2) & (blk['mean_p_label'] < CONFIDENT)
    # neighbours (same session, previous and next block) of every candidate block
    nb = []
    for _, b in blk[blk['candidate']].iterrows():
        ses = blk[blk['session_label'] == b['session_label']].set_index('block_index')
        nbrs = [ses.loc[i] for i in (b['block_index'] - 1, b['block_index'] + 1) if i in ses.index]
        nb.append({'session_label': b['session_label'], 'block_index': int(b['block_index']),
                   'neighbour_mean_p_label': float(np.mean([x['mean_p_label'] for x in nbrs])) if nbrs else math.nan,
                   'neighbour_hands': ';'.join(sorted({str(x['hand']) for x in nbrs}))})
    cand = blk[blk['candidate']].merge(pd.DataFrame(nb), on=['session_label', 'block_index'], how='left') if nb else blk.iloc[:0]
    in_cand = pd.Series([k in set(zip(cand['session_label'].astype(str), cand['block_index'].astype(int))) for k in key],
                        index=pred.index)
    swap_persons = sorted(set(pred.loc[pred['swap_block'], 'person_id']))
    blk_swap = blk[blk['person_id'].isin(swap_persons)]
    ce = pred[pred['confident_error']].sort_values('p_label')
    ce_cols = ['trial_key', 'person_id', 'platform', 'hand', 'instructed_rotation', 'speed', 'variant', 'start_position',
               'duration_s', 'p_label', 'grip_mismatch', 'false_start', 'clip_candidate', 'swap_block']
    n_ce = int(pred['confident_error'].sum())
    over = []
    for s in ('platform', 'instructed_rotation', 'speed', 'variant', 'hand', 'handedness', 'duration_bin'):
        tot = pred[s].value_counts()
        for lev in sorted(tot.index, key=str):
            k = int((pred['confident_error'] & (pred[s] == lev)).sum())
            lo, hi = _wilson(k, n_ce)
            over.append({'stratum': s, 'level': str(lev), 'n_trials': int(tot[lev]),
                         'share_trials': float(tot[lev] / len(pred)), 'n_confident_errors': k,
                         'share_confident_errors': k / n_ce if n_ce else math.nan,
                         'share_lo': lo, 'share_hi': hi,
                         'ratio': (k / n_ce) / (tot[lev] / len(pred)) if n_ce else math.nan})
    # class means of the yaw rate: correct vs confident errors, by instructed hand (descriptive)
    ix = d.channels.index('yaw_rate')
    pos = {t: i for i, t in enumerate(d.trials['trial_id'].to_numpy())}
    curves = []
    for hv, hn in ((0, 'left'), (1, 'right')):
        for what, m in (('correct', pred['correct']), ('confident_error', pred['confident_error'])):
            rows = [pos[t] for t in pred.loc[m & (pred['y'] == hv), 'trial_id']]
            if rows:
                c = d.approach[rows, ix].mean(axis=0)
                curves += [{'hand': hn, 'group': what, 'n': len(rows), 'tau': float(k / (c.size - 1)), 'yaw_rate': float(v)}
                           for k, v in enumerate(c)]
    lh = pp[pp['handedness'] == 'left']
    return {'model': r.name, 'per_person': pp.to_dict(orient='records'),
            'confident_errors': ce[ce_cols].to_dict(orient='records'),
            'n_confident_errors': n_ce, 'n_errors': int((~pred['correct']).sum()),
            'n_trials': int(len(pred)), 'n_persons': int(pred['person_id'].nunique()), 'confident_threshold': CONFIDENT,
            'confident_error_share_swap_blocks': float(pred.loc[pred['confident_error'], 'swap_block'].mean())
            if n_ce else math.nan,
            'swap_blocks': [{'session_label': s, 'block_index': b} for s, b in swap], 'swap_persons': swap_persons,
            'blocks_of_swap_persons': blk_swap.to_dict(orient='records'),
            'swap_block_accuracy': float(pred.loc[pred['swap_block'], 'correct'].mean()) if pred['swap_block'].any() else math.nan,
            'swap_block_n': int(pred['swap_block'].sum()),
            'accuracy_without_swap_blocks': float(pred.loc[~pred['swap_block'], 'correct'].mean()),
            'block_clustering': block_clustering(pred),
            'candidate_blocks': cand.to_dict(orient='records'), 'n_candidate_blocks': int(len(cand)),
            'n_candidate_trials': int(in_cand.sum()), 'candidate_persons': sorted(set(cand['person_id'])) if len(cand) else [],
            'accuracy_without_candidate_blocks': float(pred.loc[~in_cand, 'correct'].mean()),
            'overrepresentation': over, 'curves': curves,
            'left_handed_persons': lh['person_id'].tolist(),
            'left_handed_accuracy_mean': float(lh['accuracy'].mean()) if len(lh) else math.nan,
            'right_handed_accuracy_mean': float(pp.loc[pp['handedness'] != 'left', 'accuracy'].mean())}


# ───────────────────────────── 7. natural use: does each component transfer? ─────────────────────────────


def poisson_binomial_pmf(p: Sequence[float]) -> np.ndarray:
    """Exact pmf of a sum of independent Bernoulli(p_i) (dynamic programming)."""
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for q in p:
        pmf[1:] = pmf[1:] * (1 - q) + pmf[:-1] * q
        pmf[0] *= (1 - q)
    return pmf


def natural_transfer(d: Any, nat: Any, F: pd.DataFrame, Fn: pd.DataFrame) -> pd.DataFrame:
    """Person-matched test of each chiral feature's sign on the natural-use trials. For person i the direction is
    learned on the OTHER persons' primary trials (as a fold would); the expected chance that i's free-choice trial
    points to the reported hand is the share of i's own primary trials of that hand that do; the number observed
    on the free-choice trials is compared with the Poisson-binomial distribution of that count (two-sided p, and
    the lower tail). Holm over the features of the table."""
    y, g = d.y, d.groups.astype(str)
    yn, gn = nat.y, nat.groups.astype(str)
    rows = []
    for c in NAT_FEATURES:
        x, xn = F[c].to_numpy(dtype=float), Fn[c].to_numpy(dtype=float)
        hits, rates = [], []
        for i, (pid, yy) in enumerate(zip(gn, yn)):
            other = g != pid
            s = 1.0 if x[other & (y == 1)].mean() - x[other & (y == 0)].mean() >= 0 else -1.0
            own = (g == pid) & (y == yy)
            if not own.any():
                continue
            pts = (s * x[own] > 0) if yy == 1 else (s * x[own] < 0)
            rates.append(float(np.mean(pts)))
            hits.append(bool((s * xn[i] > 0) if yy == 1 else (s * xn[i] < 0)))
        pmf = poisson_binomial_pmf(rates)
        k = int(np.sum(hits))
        p2 = float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-9)].sum()))
        # share of the persons' own primary trials of the reported hand that point to it (pooled over trials)
        rows.append({'feature': c, 'component': next((k_ for k_, pre in COMPONENTS.items() if c.startswith(pre)), 'other'),
                     'dedrift': '_dd_' in c, 'n': len(hits), 'observed': k, 'expected': float(np.sum(rates)),
                     'expected_share': float(np.mean(rates)), 'observed_share': k / len(hits) if hits else math.nan,
                     'p_two_sided': p2, 'p_lower': float(pmf[:k + 1].sum()),
                     'q05': int(np.searchsorted(np.cumsum(pmf), 0.025)), 'q95': int(np.searchsorted(np.cumsum(pmf), 0.975))})
    out = pd.DataFrame(rows)
    out['p_two_sided_holm'] = holm(out['p_two_sided'].to_numpy())
    return out


def _store_contributions(store: dict[str, Any], X: pd.DataFrame, folds: np.ndarray, sign: np.ndarray) -> pd.DataFrame:
    """Per-component logit contributions (towards the given sign: +1 right, -1 left) of a stored linear model, row by
    row with each row's own fold."""
    out = np.full((len(X), len(COMPONENTS)), np.nan)
    for f in np.unique(folds):
        e = store['folds'].get(str(int(f)))
        if e is None:
            continue
        m = folds == f
        Z = M.stored_transform(e, X.loc[m, e['names']].to_numpy(dtype=float)) * np.asarray(e['coef_std'])
        for j, (comp, pre) in enumerate(COMPONENTS.items()):
            cols = [k for k, nm in enumerate(e['names']) if nm.startswith(pre)]
            out[m, j] = Z[:, cols].sum(axis=1) * sign[m] if cols else 0.0
    return pd.DataFrame(out, columns=list(COMPONENTS))


def natural_contributions(name: str, d: Any, nat: Any, F: pd.DataFrame, Fn: pd.DataFrame) -> dict[str, Any] | None:
    """For a stored linear model: the mean logit contribution of each component towards the REPORTED hand on the
    natural-use trials against the same person's primary trials of that hand (per-person difference, person
    bootstrap CI, sign-flip p, Holm over the components)."""
    st = _read_store(name)
    r = _load(name)
    if st is None or r is None or r.extra_predictions is None:
        return None
    Ft = pd.concat([F, MF.amp_normalised(F, E.PHYS6)], axis=1)
    Fnt = pd.concat([Fn, MF.amp_normalised(Fn, E.PHYS6)], axis=1)
    fold_of = r.predictions.set_index('trial_id')['fold']
    fp = fold_of.reindex(d.trials['trial_id']).to_numpy()
    pf = r.predictions.groupby('person_id')['fold'].first()
    fn_ = pf.reindex(nat.trials['person_id'].astype(str)).to_numpy()
    cp = _store_contributions(st, Ft, fp, np.where(d.y == 1, 1.0, -1.0))
    cn = _store_contributions(st, Fnt, fn_, np.where(nat.y == 1, 1.0, -1.0))
    g, gn = d.groups.astype(str), nat.groups.astype(str)
    rows = []
    for comp in COMPONENTS:
        diffs, nat_v, prim_v, prim_share = [], [], [], []
        for i, (pid, yy) in enumerate(zip(gn, nat.y)):
            own = (g == pid) & (d.y == yy)
            if not own.any() or not np.isfinite(cn.loc[i, comp]):
                continue
            a, b = float(cn.loc[i, comp]), float(cp.loc[own, comp].mean())
            nat_v.append(a)
            prim_v.append(b)
            prim_share.append(float((cp.loc[own, comp] > 0).mean()))
            diffs.append(a - b)
        dv = np.asarray(diffs)
        lo, hi = _q(dv[np.random.default_rng(C.SEED).integers(0, dv.size, (C.N_BOOT, dv.size))].mean(axis=1))
        rows.append({'model': name, 'component': comp, 'n': int(dv.size), 'natural_mean': float(np.mean(nat_v)),
                     'primary_mean': float(np.mean(prim_v)), 'diff': float(dv.mean()), 'diff_lo': lo, 'diff_hi': hi,
                     'p': cv._sign_flip_p(dv), 'natural_share_positive': float(np.mean(np.asarray(nat_v) > 0)),
                     'primary_share_positive': float(np.mean(prim_share))})
    out = pd.DataFrame(rows)
    out['p_holm'] = holm(out['p'].to_numpy())
    return {'model': name, 'rows': out.to_dict(orient='records')}


def natural_models(names: Sequence[str]) -> dict[str, Any]:
    """Natural-use correctness of the models given (tie rule applied) and the exact McNemar comparison of every
    pair: discordant trials and the two-sided binomial p."""
    from scipy.stats import binomtest
    cor = {}
    for n in names:
        r = _load(n)
        if r is None or r.extra_predictions is None:
            continue
        ex = r.extra_predictions.set_index('trial_id')
        p = ex['p_right'].to_numpy(dtype=float)
        cor[n] = pd.Series(np.where(_ties(p), 0.5, ((p >= 0.5).astype(int) == ex['y'].to_numpy()).astype(float)),
                           index=ex.index)
    pairs = []
    ks = list(cor)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            j = pd.concat([cor[a], cor[b]], axis=1, join='inner')
            na, nb_ = int(((j.iloc[:, 0] == 1) & (j.iloc[:, 1] == 0)).sum()), int(((j.iloc[:, 0] == 0) & (j.iloc[:, 1] == 1)).sum())
            p = float(binomtest(na, na + nb_, 0.5).pvalue) if na + nb_ else 1.0
            pairs.append({'a': a, 'b': b, 'a_only': na, 'b_only': nb_, 'p_mcnemar': p})
    return {'pairs': pairs, 'correct': {k: v.tolist() for k, v in cor.items()}}


def rule_nested_choice() -> dict[str, Any] | None:
    """The features the in-fold best single-feature rule chose (per fold) and its thresholds."""
    r = _load('m1_rule_nested')
    if r is None:
        return None
    par = r.predictions.groupby('fold')['params'].first().map(json.loads)
    feats = par.map(lambda v: v.get('feature')).value_counts()
    th = par[par.map(lambda v: v.get('feature')) == feats.index[0]].map(lambda v: v.get('threshold')).astype(float)
    return {'features': feats.to_dict(), 'mode': str(feats.index[0]), 'mode_folds': int(feats.iloc[0]),
            'n_folds': int(par.size), 'threshold_median': float(th.median()), 'unit': MF.BY_NAME[feats.index[0]].unit}


# ───────────────────────────── 8. integration drift of the lateral feature ─────────────────────────────


def drift_summary(d: Any, F: pd.DataFrame) -> dict[str, Any]:
    """Detection of the lateral integration drift: the mean lateral acceleration over the still hold (the bias the
    de-drifted features remove), the drift it implies over the approach (bias x duration) against the size of
    the lateral feature, and whether the bias itself differs between the hands."""
    dur = F['duration_s'].to_numpy(dtype=float)
    q = max(MF.DEDRIFT_QUARTILES)
    b = (F[f'latvel_q{MF._pct(q)}'] - F[f'latvel_dd_q{MF._pct(q)}']).to_numpy(dtype=float) / (q * dur)
    drift = np.abs(b) * dur
    lat = np.abs(F[f'latvel_q{MF._pct(q)}'].to_numpy(dtype=float))
    ratio = drift / np.maximum(lat, 1e-9)
    y = d.y
    pr = pd.DataFrame({'person_id': d.groups, 'y': y, 'p_right': b})
    auc = _auc(y, b)
    lo, hi = cv.person_bootstrap(pr, fn=_auc)['stat']
    return {'bias_abs_median': float(np.median(np.abs(b))), 'bias_abs_q90': float(np.quantile(np.abs(b), 0.9)),
            'bias_mean_left': float(b[y == 0].mean()), 'bias_mean_right': float(b[y == 1].mean()),
            'bias_auc': auc, 'bias_auc_lo': lo, 'bias_auc_hi': hi,
            'drift_abs_median': float(np.median(drift)), 'lat_abs_median': float(np.median(lat)),
            'ratio_median': float(np.median(ratio)), 'ratio_q90': float(np.quantile(ratio, 0.9)),
            'share_ratio_above_half': float(np.mean(ratio > 0.5)), 'quartile': q,
            'hold_window': list(MF.PRE_S), 'n_trials': int(len(b))}


# ───────────────────────────── 9. grid edges, sensitivity ─────────────────────────────


def grid_edges() -> pd.DataFrame:
    """Per experiment and hyperparameter: the values chosen across the folds and how many folds chose an edge of
    its grid (a grid that is too narrow shows here)."""
    rows = []
    for s in E.SPECS:
        r = _load(s.name)
        grid = _grid_of(s)
        if r is None or not grid:
            continue
        par = r.predictions.groupby('fold')['params'].first().map(json.loads)
        for hp, values in grid.items():
            ch = par.map(lambda v: v.get(hp)).dropna()
            if not len(ch):
                continue
            vals = sorted(float(v) for v in values)
            chf = ch.astype(float)
            lo_edge = int(np.isclose(chf, vals[0]).sum()) if len(vals) > 1 else 0
            hi_edge = int(np.isclose(chf, vals[-1]).sum()) if len(vals) > 1 else 0
            rows.append({'model': s.name, 'hyperparameter': hp, 'grid': ';'.join(f'{v:g}' for v in vals),
                         'n_folds': int(chf.size), 'chosen': ';'.join(f'{k:g}:{v}' for k, v in chf.value_counts().sort_index().items()),
                         'folds_at_low_edge': lo_edge, 'folds_at_high_edge': hi_edge})
        if s.factory == 'make_linear_select':
            ch = par.map(lambda v: v.get('option')).value_counts()
            rows.append({'model': s.name, 'hyperparameter': 'option', 'grid': ';'.join(s.params['options']),
                         'n_folds': int(ch.sum()), 'chosen': ';'.join(f'{k}:{v}' for k, v in ch.items()),
                         'folds_at_low_edge': 0, 'folds_at_high_edge': 0})
    return pd.DataFrame(rows)


def sensitivity_summary(mt: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for s in E.SPECS:
        if s.group != 'sensitivity':
            continue
        r = _load(s.name)
        if r is None:
            continue
        drop = r.predictions.groupby('fold')['params'].first().map(lambda v: json.loads(v).get('n_train_dropped', 0))
        row = mt.set_index('name').loc[s.name] if s.name in set(mt['name']) else None
        out.append({'name': s.name, 'anchor': s.baseline, 'drop': s.params.get('drop'),
                    'blocks': s.params.get('blocks', []), 'dropped_median': float(drop.median()),
                    'dropped_min': int(drop.min()), 'dropped_max': int(drop.max()),
                    **({k: row[k] for k in row.index if k.startswith('vs_base_')} if row is not None else {})})
    return out


# ───────────────────────────── 10. model artefacts (tree, profiles, shapes, coefficients) ─────────────────────────────


def tree_summary(d: Any, F: pd.DataFrame) -> dict[str, Any] | None:
    st = _read_store('m1_tree')
    if st is None:
        return None
    folds = st['folds']
    cfg = pd.Series([(v['depth'], v['min_leaf']) for v in folds.values()]).value_counts()
    depth, leaf = cfg.index[0]
    nodes = M.fit_tree_full(F, d.y, E.BY_NAME['m1_tree'].params['cols'], int(depth), int(leaf), seed=C.SEED)
    roots = [{'fold': int(k), 'feature': v['nodes'][0]['feature'], 'threshold': v['nodes'][0]['threshold'],
              'depth': v['depth'], 'min_leaf': v['min_leaf']} for k, v in sorted(folds.items(), key=lambda kv: int(kv[0]))]
    used = pd.Series([n['feature'] for v in folds.values() for n in v['nodes'] if not n['leaf']]).value_counts()
    return {'depth': int(depth), 'min_leaf': int(leaf), 'n_folds_with_this_config': int(cfg.iloc[0]),
            'n_folds': len(folds), 'nodes_all_rows': nodes, 'roots_per_fold': roots,
            'features_used_per_fold': used.to_dict()}


def profile_summary(name: str) -> dict[str, Any] | None:
    st = _read_store(name)
    if st is None:
        return None
    folds = st['folds']
    names = folds[next(iter(folds))]['names']
    W = np.array([v['coef_std'] for v in folds.values()])
    R = np.array([v['coef_raw'] for v in folds.values()])
    rows = []
    for j, c in enumerate(names):
        ch, tk = c.rsplit('_t', 1)
        k = int(tk) - 1
        rows.append({'channel': ch, 'bin': k + 1, 'tau': float(MF.tc_bin_centres(MF.TC_BINS)[k]),
                     'coef_std_median': float(np.median(W[:, j])), 'coef_std_min': float(W[:, j].min()),
                     'coef_std_max': float(W[:, j].max()), 'coef_std_q10': float(np.quantile(W[:, j], 0.1)),
                     'coef_std_q90': float(np.quantile(W[:, j], 0.9)), 'coef_raw_median': float(np.median(R[:, j])),
                     'coef_raw_q10': float(np.quantile(R[:, j], 0.1)), 'coef_raw_q90': float(np.quantile(R[:, j], 0.9)),
                     'sign_agreement': float(np.mean(np.sign(W[:, j]) == np.sign(np.median(W[:, j]))))})
    return {'name': name, 'n_folds': len(folds), 'n_bins': MF.TC_BINS, 'C_per_fold': [v['C'] for v in folds.values()],
            'rows': rows}


def gam_summary(name: str = 'm1_gam_phys6', n_grid: int = 61) -> dict[str, Any] | None:
    """Median shape per feature across folds, the range across folds, and descriptors of the median shape:
    monotone (every step in one direction) and saturating (outer-fifth slope below the central slope)."""
    st = _read_store(name)
    if st is None:
        return None
    folds = st['folds']
    cols = list(next(iter(folds.values()))['shapes'])
    out, desc = [], []
    for c in cols:
        lo = min(min(v['shapes'][c]['x']) for v in folds.values())
        hi = max(max(v['shapes'][c]['x']) for v in folds.values())
        g = np.linspace(lo, hi, n_grid)
        Fm = np.full((len(folds), n_grid), np.nan)
        for i, v in enumerate(folds.values()):
            x, f = np.asarray(v['shapes'][c]['x']), np.asarray(v['shapes'][c]['f'])
            inside = (g >= x.min()) & (g <= x.max())
            Fm[i, inside] = np.interp(g[inside], x, f)
        med = []
        for k in range(n_grid):
            col = Fm[:, k]
            ok = np.isfinite(col)
            if ok.sum() < max(3, len(folds) // 2):
                continue
            med.append((g[k], float(np.median(col[ok]))))
            out.append({'feature': c, 'x': float(g[k]), 'f_median': float(np.median(col[ok])),
                        'f_min': float(col[ok].min()), 'f_max': float(col[ok].max()), 'n_folds': int(ok.sum())})
        if len(med) >= 5:
            xs, fs = np.array(med).T
            dfs = np.diff(fs)
            sgn = np.sign(np.sum(dfs))
            monotone = bool(np.all(dfs * sgn >= -1e-9))
            n5 = max(1, len(xs) // 5)
            sl = lambda a, b: (fs[b] - fs[a]) / (xs[b] - xs[a])  # noqa: E731
            centre = sl(2 * n5, len(xs) - 2 * n5 - 1) if len(xs) - 4 * n5 > 1 else sl(0, len(xs) - 1)
            outer = 0.5 * (sl(0, n5) + sl(len(xs) - 1 - n5, len(xs) - 1))
            desc.append({'feature': c, 'monotone': monotone, 'direction': 'increasing' if sgn > 0 else 'decreasing',
                         'saturating': bool(abs(outer) < abs(centre)), 'slope_centre': float(centre),
                         'slope_outer': float(outer), 'unit': MF.BY_NAME[c].unit})
    return {'name': name, 'n_folds': len(folds), 'knots_per_fold': [v['n_knots'] for v in folds.values()],
            'C_per_fold': [v['C'] for v in folds.values()], 'rows': out, 'shape_descriptors': desc}


def coef_summary(name: str = FOCUS) -> dict[str, Any] | None:
    st = _read_store(name)
    if st is None:
        return None
    folds = st['folds']
    names = list(pd.Series([tuple(v['names']) for v in folds.values()]).value_counts().index[0])
    same = [v for v in folds.values() if v['names'] == names]
    W = np.array([v['coef_std'] for v in same])
    R = np.array([v['coef_raw'] for v in same])
    return {'name': name, 'n_folds': len(folds), 'n_folds_shown': len(same),
            'options': pd.Series([v.get('option', '') for v in folds.values()]).value_counts().to_dict(),
            'rows': [{'feature': c, 'coef_std_median': float(np.median(W[:, j])),
                      'coef_std_min': float(W[:, j].min()), 'coef_std_max': float(W[:, j].max()),
                      'coef_raw_median': float(np.median(R[:, j])), 'unit': MF.BY_NAME[c].unit,
                      'sign_agreement': float(np.mean(np.sign(W[:, j]) == np.sign(np.median(W[:, j]))))}
                     for j, c in enumerate(names)],
            'C_per_fold': [v['C'] for v in folds.values()]}


def store_path(name: str):
    return L.tab_dir / f'm1_store_{name[3:]}.json'


def _read_store(name: str) -> dict[str, Any] | None:
    p = store_path(name)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


# ───────────────────────────── 11. physics: class-mean time courses ─────────────────────────────


def physics_curves(d: Any, n_boot: int = 1000, seed: int = C.SEED) -> pd.DataFrame:
    """Per hand: the mean over persons of each person's mean time course (yaw rate, roll rate, lateral acceleration
    and their integrals over the approach), with person-bootstrap 95 % bands, and the mirrored left-hand mean."""
    A = d.approach.astype(float)
    ix = {c: i for i, c in enumerate(d.channels)}
    bc = MF.body_channels(A, d.channels)
    dur = d.trials['duration_ms'].to_numpy(dtype=float) / 1000.0
    dt = dur / (A.shape[2] - 1)
    series = {'yaw_rate': A[:, ix['yaw_rate']], 'roll_rate': bc['roll_rate'], 'acc_lat': bc['acc_lat'],
              'yaw_cum': A[:, ix['yaw_cum']], 'roll_cum': MF._cumtrapz(bc['roll_rate'], dt),
              'latvel': MF._cumtrapz(bc['acc_lat'], dt)}
    y, g = d.y, d.groups
    rng = np.random.default_rng(seed)
    tau = np.linspace(0, 1, A.shape[2])
    rows = []
    for name, X in series.items():
        for hv, hn in ((0, 'left'), (1, 'right')):
            persons = sorted(set(g[y == hv].tolist()))
            PM = np.stack([X[(g == p) & (y == hv)].mean(axis=0) for p in persons])
            mean = PM.mean(axis=0)
            bs = np.stack([PM[rng.integers(0, len(persons), len(persons))].mean(axis=0) for _ in range(n_boot)])
            lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
            for k in range(tau.size):
                rows.append({'series': name, 'hand': hn, 'tau': float(tau[k]), 'mean': float(mean[k]),
                             'lo': float(lo[k]), 'hi': float(hi[k]), 'n_persons': len(persons)})
            if hv == 0:
                for k in range(tau.size):
                    rows.append({'series': name, 'hand': 'left_mirrored', 'tau': float(tau[k]), 'mean': float(-mean[k]),
                                 'lo': float(-hi[k]), 'hi': float(-lo[k]), 'n_persons': len(persons)})
    return pd.DataFrame(rows)


# ───────────────────────────── derive: everything the presentation reads ─────────────────────────────


def derive(d: Any, nat: Any, F: pd.DataFrame, Fn: pd.DataFrame, *, verbose: bool = True) -> None:
    """Every derived file of lane m1 from the saved results (no model is evaluated here; the tree drawn in the
    report is refitted on every row with the modal in-fold hyperparameters). Writes the finalists right after the
    model table, so the finalist-specific analyses below use them."""
    from stage2.lanes import m1_select

    def say(msg: str) -> None:
        if verbose:
            print(f'  derive: {msg}', flush=True)

    _CACHE.clear()
    mt, fams = model_table()
    extra = {}
    a, b = _load('m1_lr_phys6_aug'), _load(FOCUS)
    if a is not None and b is not None:   # mirror augmentation + intercept vs the antisymmetric model: same trials
        j = a.predictions[['trial_id', 'p_right']].merge(b.predictions[['trial_id', 'p_right']], on='trial_id')
        extra['aug_vs_antisym_max_abs_dp'] = float(np.max(np.abs(j['p_right_x'] - j['p_right_y'])))
        extra['aug_vs_antisym_same_decisions'] = float(np.mean((j['p_right_x'] >= 0.5) == (j['p_right_y'] >= 0.5)))
    _write('m1_models', {'rows': mt.to_dict(orient='records'), 'families': fams, **extra})
    say(f'model table ({len(mt)} rows, {len(fams)} comparison families)')
    fins = m1_select.choose_finalists()
    if fins:
        L.write_finalists(fins)
    slots = m1_select.slot_names()
    fin = [f['name'] for f in fins]
    say(f'finalists {fin}')
    tf = _trial_frame(d, F)
    st = [strata_table(r, tf) for n in dict.fromkeys([*STRATA_MODELS, *fin]) if (r := _load(n)) is not None]
    if st:
        _write('m1_strata', {'rows': pd.concat(st, ignore_index=True).to_dict(orient='records'),
                             'min_persons_ci': MIN_PERSONS_CI})
        say('strata')
    res = {n: r for n in dict.fromkeys([*CAL_MODELS, *fin]) if (r := _load(n)) is not None}
    if res:
        _write('m1_calibration', calibration_tables(res, tf))
        say('calibration')
    ag, agr, agp = [], [], []
    for n in dict.fromkeys([*AGG_MODELS, *[f for f in fin if f in mt['name'].tolist()]]):
        r = _load(n)
        if r is not None and r.predictions['params'].str.contains('agg_alpha').any():
            a_, b_, c_ = k_aggregation(r)
            ag.append(a_)
            agr.append(b_)
            agp.append(c_)
    if ag:
        _write('m1_k_aggregation', {'rows': pd.concat(ag, ignore_index=True).to_dict(orient='records'),
                                    'reliability_k3': pd.concat(agr, ignore_index=True).to_dict(orient='records'),
                                    'paired': pd.concat(agp, ignore_index=True).to_dict(orient='records'),
                                    'n_draws': N_DRAWS_AGG})
        say('k aggregation')
    focus = slots['small'] if _load(slots['small']) is not None else FOCUS
    r = _load(focus)
    if r is not None:
        _write('m1_failures', failures(r, d, tf, F))
        say(f'failures ({focus})')
    nt = natural_transfer(d, nat, F, Fn)
    contrib = [c for n in dict.fromkeys([FOCUS, 'm1_lr_phys6_win_asinh', slots['small']])
               if (c := natural_contributions(n, d, nat, F, Fn)) is not None]
    _write('m1_natural', {'transfer': nt.to_dict(orient='records'), 'contributions': contrib,
                          'models': natural_models([slots['rule'], slots['small'], slots['tree_gam'], FOCUS,
                                                    'm1_rule_nested', E.REFERENCE]),
                          'rule_nested': rule_nested_choice()})
    say('natural use')
    _write('m1_drift', drift_summary(d, F))
    _write('m1_grid', {'rows': grid_edges().to_dict(orient='records')})
    _write('m1_sensitivity', {'rows': sensitivity_summary(mt)})
    say('drift, grid edges, sensitivity')
    for fn, nm in ((lambda: tree_summary(d, F), 'm1_tree_summary'), (lambda: profile_summary('m1_tf_yaw'), 'm1_tf_yaw_profile'),
                   (lambda: profile_summary('m1_tf_multi'), 'm1_tf_multi_profile'), (gam_summary, 'm1_gam_shapes'),
                   (coef_summary, 'm1_phys6_coefficients'),
                   (lambda: coef_summary(slots['small']) or {'name': slots['small'], 'rows': [],
                                                             'note': 'no per-fold coefficients stored'},
                    'm1_small_coefficients')):
        out = fn()
        if out is not None:
            _write(nm, out)
            say(nm)
    pc = physics_curves(d)
    _write('m1_physics_curves', {'rows': pc.to_dict(orient='records')})
    kf = pd.concat([tf[['trial_id', 'person_id', 'hand', 'platform']].reset_index(drop=True),
                    F[list(KEY_FEATURES)].reset_index(drop=True)], axis=1)
    _write('m1_key_features', {'features': list(KEY_FEATURES), 'rows': _report_rows(kf)})
    say('physics curves, key features')


def _report_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    from stage2.report_export import report_safe
    return report_safe(df).to_dict(orient='records')
