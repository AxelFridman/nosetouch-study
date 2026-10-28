"""Lane demo: THE DEMOGRAPHICS ADD-ON TEST (the only place where demographics enter a model; nothing here goes into
the production model).

Design (a) of the fast-track brief: a second-stage logistic regression on the final model's logit plus the person
covariates, fitted inside each outer fold on INNER cross-fitted logits of the training persons.

* Training side of outer fold j: the final model's inner cross-fitted P(right) of every training row, read from lane
  final's inner-prediction cache (``analysis/data/stage2_final/inner/<candidate>/``; computed on exactly the harness's
  capped training rows of fold j, by models that never saw person j). They enter X as one column per outer fold of the
  matrix ``inner_logit_by_fold`` (NaN where a row was not a training row of that fold); fit_predict reads the column
  of its own fold only.
* Test side: the held-out person's logit from the final model's saved LOSO predictions (``outer_logit``: the fold
  model that never saw that person; natural-use trials from the same result's extra predictions).
* Person covariates (``person_covariates``): age, female, height (with a missing indicator), self-reported
  handedness sign and usual phone hand sign, one row per trial from the person table. The test person's covariates
  are used at prediction time (the hypothetical where production knows them). Instructed conditions are never used.

Never stack on outer-fold predictions of other persons: those fold models were trained on the held-out person.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stage2 import cv

#: the candidate of lane final whose inner predictions are stacked, and its shipped LOSO result
FINAL_CANDIDATE = 'm3_deep'
FINAL_RESULT = 'final_cal_m3_deep_none'

#: the covariate columns of ``person_covariates`` (encoded, before in-fold standardisation)
COVARIATES: tuple[str, ...] = ('age', 'female', 'height', 'height_missing', 'handed_sign', 'phone_sign')
#: readable names (tables, figures)
COVARIATE_LABELS = {'age': 'age', 'female': 'female', 'height': 'height', 'height_missing': 'height missing',
                    'handed_sign': 'handedness (+1 right, -1 left)', 'phone_sign': 'usual phone hand (+1 right, 0 both)'}
SIDE_SIGN = {'right': 1.0, 'left': -1.0, 'both': 0.0, 'ambidextrous': 0.0, 'ambi': 0.0}

#: penalty grid of the second stage (inverse L2 strength), chosen inside each outer fold by fold.inner_splits()
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)

#: the second-stage feature sets: name -> (uses the final model's logit, covariate main effects, covariate x logit)
FEATURE_SETS: dict[str, tuple[bool, bool, bool]] = {
    'sensor': (True, False, False),
    'main': (True, True, False),
    'inter': (True, True, True),
    'cov_only': (False, True, False),
}
EPS = 1e-6


def logit(p: Any, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def encode_persons(persons: pd.DataFrame) -> pd.DataFrame:
    """Person table -> encoded covariates (index person_id, columns COVARIATES). Height missing -> NaN (imputed in-fold
    with the training median) plus an indicator."""
    P = persons.set_index('person_id')
    out = pd.DataFrame(index=P.index)
    out['age'] = pd.to_numeric(P['age_years'], errors='coerce').astype(float)
    out['female'] = (P['sex'].astype(str).str.lower() == 'female').astype(float)
    h = pd.to_numeric(P['height_cm'], errors='coerce').astype(float)
    out['height'] = h
    out['height_missing'] = h.isna().astype(float)
    out['handed_sign'] = P['handedness_self'].astype(str).str.lower().map(SIDE_SIGN).fillna(0.0).astype(float)
    out['phone_sign'] = P['phone_hand'].astype(str).str.lower().map(SIDE_SIGN).fillna(0.0).astype(float)
    return out[list(COVARIATES)]


def covariate_rows(person_ids: Sequence[str], enc: pd.DataFrame) -> np.ndarray:
    """One covariate row per trial (float array, NaN for a missing height)."""
    missing = sorted(set(map(str, person_ids)) - set(enc.index.astype(str)))
    if missing:
        raise KeyError(f'persons without a profile: {missing}')
    return enc.loc[list(map(str, person_ids))].to_numpy(dtype=float)


def inner_logit_matrix(d: Any, persons_order: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """(N primary rows, n persons): column j = the final candidate's inner cross-fitted logit of every training row of
    outer fold j (lane final's cache, checked against the current code fingerprint and the exact training trial ids
    of the harness fold); NaN elsewhere. Raises when a fold is missing from the cache."""
    from stage2.lanes import final_candidates as FC
    from stage2.lanes import final_nested as FN
    cands = FC.candidates()
    c = cands[FINAL_CANDIDATE]
    fps = {c.key: FN._cand_fingerprint(c)}
    trials = d.trials.reset_index(drop=True)
    folds = FN.harness_folds(np.zeros((len(trials), 1)), trials)
    pos = pd.Index(trials['trial_id'].astype(str))
    Z = np.full((len(trials), len(persons_order)), np.nan)
    check = []
    for f in folds:
        p = FN.inner_predictions(c, f, cands, compute=False, fingerprints=fps)
        if p is None:
            raise RuntimeError(f'lane final inner cache lacks {FINAL_CANDIDATE} for fold {f.test_person}: run '
                               f'python -m stage2.lanes.final_nested --jobs {FINAL_CANDIDATE}:{f.fold}')
        idx = pos.get_indexer(f.meta_train['trial_id'].astype(str))
        if np.any(idx < 0):
            raise KeyError('inner cache rows not in the trial table')
        j = list(persons_order).index(f.test_person)
        Z[idx, j] = logit(p)
        check.append({'fold': int(f.fold), 'person_id': f.test_person, 'n_train_rows': int(idx.size),
                      'inner_log_loss': float(-np.mean(f.y_train * np.log(np.clip(p, 1e-15, 1)) +
                                                       (1 - f.y_train) * np.log(np.clip(1 - p, 1e-15, 1))))})
    return Z, {'candidate': FINAL_CANDIDATE, 'folds': check}


def outer_logits(trial_ids: Sequence[str], pred: pd.DataFrame) -> np.ndarray:
    """The saved LOSO P(right) of the final model (held-out person's fold model) as a logit, aligned to trial_ids."""
    m = pred.set_index(pred['trial_id'].astype(str))['p_right']
    ids = list(map(str, trial_ids))
    missing = [t for t in ids if t not in m.index]
    if missing:
        raise KeyError(f'{len(missing)} trials lack a saved LOSO prediction of {FINAL_RESULT}')
    return logit(m.loc[ids].to_numpy(dtype=float))


def build_inputs(d: Any, nat: Any, persons_order: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                                                                          dict[str, Any]]:
    """X of the primary rows and of the natural-use rows (same keys), and a provenance record."""
    r = cv.load_result(FINAL_RESULT)
    enc = encode_persons(d.persons)
    Z, info = inner_logit_matrix(d, persons_order)
    X = {'inner_logit_by_fold': Z,
         'outer_logit': outer_logits(d.trials['trial_id'], r.predictions),
         'person_covariates': covariate_rows(d.trials['person_id'], enc)}
    Xn = {'inner_logit_by_fold': np.full((len(nat.trials), len(persons_order)), np.nan),
          'outer_logit': outer_logits(nat.trials['trial_id'], r.extra_predictions),
          'person_covariates': covariate_rows(nat.trials['person_id'], enc)}
    return X, Xn, info


# ───────────────────────────── the second stage ─────────────────────────────


def _design(z: np.ndarray, cov: np.ndarray, which: str, stats: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    use_z, use_c, use_i = FEATURE_SETS[which]
    cs = (np.where(np.isnan(cov), stats['median'], cov) - stats['mean']) / stats['sd']
    cols, names = [], []
    if use_z:
        cols.append(z[:, None])
        names.append('logit')
    if use_c:
        cols.append(cs)
        names += list(COVARIATES)
    if use_i:
        cols.append(cs * z[:, None])
        names += [f'{c} x logit' for c in COVARIATES]
    return np.hstack(cols), names


def _cov_stats(cov: np.ndarray) -> dict[str, np.ndarray]:
    med = np.nanmedian(cov, axis=0)
    filled = np.where(np.isnan(cov), med, cov)
    sd = filled.std(axis=0)
    return {'median': med, 'mean': filled.mean(axis=0), 'sd': np.where(sd > 1e-12, sd, 1.0)}


def _fit(Xd: np.ndarray, y: np.ndarray, C: float) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return m.fit(Xd, y)


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def stack_fit_predict(which: str, persons_order: Sequence[str]) -> Callable[[cv.Fold], Any]:
    """fit_predict of the second stage ``which`` (FEATURE_SETS). C is chosen by grouped inner cross-validation
    (fold.inner_splits()) on the training persons' inner logits; the held-out person enters only at prediction."""
    order = list(persons_order)

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        j = order.index(fold.test_person)
        z_tr = np.asarray(fold.X_train['inner_logit_by_fold'][:, j], dtype=float)
        if not np.all(np.isfinite(z_tr)):
            raise ValueError(f'fold {fold.test_person}: {int(np.sum(~np.isfinite(z_tr)))} training rows lack an inner '
                             'cross-fitted logit (the inner cache must cover the harness training rows)')
        c_tr = np.asarray(fold.X_train['person_covariates'], dtype=float)
        y = fold.y_train.astype(int)
        scores = {}
        for C in C_GRID:
            p_in = np.full(y.size, np.nan)
            for itr, iva in fold.inner_splits():
                st = _cov_stats(c_tr[itr])
                A, _ = _design(z_tr[itr], c_tr[itr], which, st)
                B, _ = _design(z_tr[iva], c_tr[iva], which, st)
                p_in[iva] = _fit(A, y[itr], C).predict_proba(B)[:, 1]
            scores[C] = _ll(y, p_in)
        best = min(C_GRID, key=lambda c: (round(scores[c], 10), C_GRID.index(c)))
        st = _cov_stats(c_tr)
        A, names = _design(z_tr, c_tr, which, st)
        m = _fit(A, y, best)
        B, _ = _design(np.asarray(fold.X_test['outer_logit'], dtype=float),
                       np.asarray(fold.X_test['person_covariates'], dtype=float), which, st)
        p = m.predict_proba(B)[:, 1]
        coef = m[-1].coef_.ravel()
        info = {'C': best, 'inner_logloss': {str(k): round(v, 6) for k, v in scores.items()},
                'coef_std': {n: round(float(v), 6) for n, v in zip(names, coef)},
                'intercept': round(float(m[-1].intercept_[0]), 6)}
        return p, info

    return fit_predict


def rule_fit_predict(kind: str) -> Callable[[cv.Fold], Any]:
    """Fixed rules for the base-rate note (no fitting): 'phone' answers the usual phone hand (P = 0.5 + 0.25 x sign:
    0.75 right, 0.5 both, 0.25 left); 'right' always answers right (P = 0.75)."""
    k = COVARIATES.index('phone_sign')

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        n = cv._n_rows(fold.X_test)
        if kind == 'phone':
            s = np.asarray(fold.X_test['person_covariates'], dtype=float)[:, k]
            return 0.5 + 0.25 * s, {'rule': 'usual phone hand'}
        return np.full(n, 0.75), {'rule': 'constant right'}

    return fit_predict


def fold_coefficients(result: cv.ExperimentResult) -> pd.DataFrame:
    """Per outer fold: chosen C and standardised second-stage coefficients (from the saved per-trial info)."""
    rows = []
    for (fold, person), g in result.predictions.groupby(['fold', 'person_id'], sort=True):
        info = json.loads(g['params'].iloc[0])
        for name, v in (info.get('coef_std') or {}).items():
            rows.append({'fold': int(fold), 'person_id': str(person), 'C': info.get('C'), 'term': name, 'coef': v})
    return pd.DataFrame(rows, columns=['fold', 'person_id', 'C', 'term', 'coef'])


__all__ = ['FINAL_CANDIDATE', 'FINAL_RESULT', 'COVARIATES', 'COVARIATE_LABELS', 'FEATURE_SETS', 'C_GRID',
           'encode_persons', 'covariate_rows', 'inner_logit_matrix', 'outer_logits', 'build_inputs',
           'stack_fit_predict', 'rule_fit_predict', 'fold_coefficients', 'logit', 'sigmoid']
