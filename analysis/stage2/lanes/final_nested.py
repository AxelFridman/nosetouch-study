"""Lane final: the nested cross-lane selection (task 2) and the cross-lane ensemble.

The selection procedure is evaluated as ONE model through ``cv.run_experiment``: inside every outer fold,

1. every candidate (``final_candidates``) is cross-validated on the TRAINING persons only: ``fold.inner_splits()``
   (GroupKFold over the training persons) gives inner folds, each wrapped as a ``cv.Fold`` exactly like the harness
   wraps an outer fold (training rows with their meta; validation rows with the test-side columns only, in a seeded
   random order, no label), and the candidate's own ``fit_predict`` (with its own nested tuning) predicts the inner
   validation persons;
2. the candidate with the lowest pooled inner log loss is chosen (pre-specified criterion; ties: the earlier
   candidate of the registry order);
3. the chosen candidate is refitted on every training person and predicts the held-out person.

The held-out person never enters steps 1-2. Inner predictions are cached on disk per (candidate, outer training set)
under ``analysis/data/stage2_final/inner/`` (resumable; the cache key holds the candidate's description, its code
fingerprint and the training trial ids), and :func:`prewarm` fills that cache in parallel worker processes on
exactly the folds the harness builds (:func:`harness_folds` replicates its fold construction, D2 cap included).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.lanes import final_candidates as FC

INNER_DIR = FC.DATA_DIR / 'inner'
#: version of the inner-fold construction (part of every cache key)
INNER_VERSION = 'inner-v1'
EPS = 1e-15


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ───────────────────────────── folds ─────────────────────────────


def harness_folds(X: Any, meta: pd.DataFrame, *, extra: tuple[Any, pd.DataFrame] | None = None, seed: int = C.SEED,
                  cap_per_person: int | None = C.CAP_PER_PERSON,
                  test_meta_columns: tuple[str, ...] = cv.TEST_META_COLUMNS) -> list[cv.Fold]:
    """The outer folds exactly as ``cv.run_experiment`` builds them (same capped training rows, same seeds, same
    test-row order; checks of the harness are not repeated). Used to fill caches outside the harness."""
    meta = meta.reset_index(drop=True)
    persons = meta['person_id'].astype(str).to_numpy()
    yv = meta['y_right'].astype(int).to_numpy()
    ge, Xe, me = None, None, None
    if extra is not None:
        Xe, me = extra
        me = me.reset_index(drop=True)
        ge = me['person_id'].astype(str).to_numpy()
    folds = []
    for s in cv.loso_splits(persons):
        tr = s.train[cv.cap_rows(persons[s.train], yv[s.train], cap_per_person, seed)]
        ie = np.flatnonzero(ge == s.person) if ge is not None else np.zeros(0, dtype=int)
        X_test = cv._take(X, s.test)
        m_test = cv._test_meta(meta, s.test, test_meta_columns)
        if ie.size:
            X_test = cv._concat(X_test, cv._take(Xe, ie))
            m_test = pd.concat([m_test, cv._test_meta(me, ie, test_meta_columns)], ignore_index=True)
        n_t = s.test.size + ie.size
        perm = np.random.default_rng([seed, 2, s.fold]).permutation(n_t)
        X_test = cv._take(X_test, perm)
        m_test = m_test.iloc[perm].reset_index(drop=True)
        if 'row' in test_meta_columns:
            m_test.insert(0, 'row', np.arange(n_t))
        folds.append(cv.Fold(s.fold, s.person, cv._take(X, tr), yv[tr].copy(), persons[tr].copy(),
                             meta.iloc[tr].reset_index(drop=True).copy(), X_test, m_test, seed + s.fold))
    return folds


def subfold(fold: cv.Fold, key: str) -> cv.Fold:
    """The fold with X restricted to one input of the shared input dict."""
    return dataclasses.replace(fold, X_train=fold.X_train[key], X_test=fold.X_test[key])


def inner_fold(fold: cv.Fold, itr: np.ndarray, iva: np.ndarray, k: int,
               test_meta_columns: tuple[str, ...] = cv.TEST_META_COLUMNS) -> tuple[cv.Fold, np.ndarray]:
    """Inner fold k of an outer fold: training rows ``itr`` (with labels and meta), validation rows ``iva`` as a test
    side (test-side columns only, seeded random order, an opaque row index). Returns (fold, order): the validation
    row of position i of the inner test side is ``order[i]`` (an index into the outer training rows)."""
    order = iva[np.random.default_rng([fold.seed, 3, k]).permutation(iva.size)]
    m_test = cv._test_meta(fold.meta_train, order, test_meta_columns)
    m_test.insert(0, 'row', np.arange(order.size))
    f = cv.Fold(fold=1000 * (fold.fold + 1) + k, test_person='+'.join(sorted(set(fold.groups_train[iva].tolist()))),
                X_train=cv._take(fold.X_train, itr), y_train=fold.y_train[itr].copy(),
                groups_train=fold.groups_train[itr].copy(), meta_train=fold.meta_train.iloc[itr].reset_index(drop=True),
                X_test=cv._take(fold.X_train, order), meta_test=m_test, seed=fold.seed * 10 + k + 1)
    return f, order


# ───────────────────────────── inner predictions (cached) ─────────────────────────────


def _cand_fingerprint(c: FC.Candidate) -> str:
    from stage2.pipeline import source_fingerprint
    mods = [c.module, 'stage2.lanes.final_nested']
    if c.module == 'stage2.lanes.m3_finalists':
        mods += ['stage2.lanes.m3_common', 'stage2.lanes.m3_deep', 'stage2.lanes.m3_nets', 'stage2.lanes.m3_body',
                 'stage2.lanes.m3_minirocket', 'stage2.lanes.m3_tf']
    if c.module == 'stage2.lanes.m2_finalists' or c.factory == 'make_m2_lr':
        mods += ['stage2.lanes.m2_models', 'stage2.lanes.m2_calibration', 'stage2.lanes.m2_features']
    if c.factory == 'make_m2_lr':
        mods += ['stage2.lanes.final_features']
    if c.module == 'stage2.lanes.m1_models':
        mods += ['stage2.lanes.m1_features']
    if c.module == 'stage2.lanes.p1_panel':
        mods += ['stage2.lanes.p1_calibration', 'stage2.lanes.p1_minirocket']
    if c.factory == 'make_reference':
        mods += ['stage2.scripts.reference']
    return source_fingerprint(*dict.fromkeys(mods))


def cache_key(c: FC.Candidate, fold: cv.Fold, fingerprint: str | None = None) -> str:
    ids = '|'.join(fold.meta_train['trial_id'].astype(str).tolist())
    desc = json.dumps(FC.describe(c), sort_keys=True, default=str)
    raw = '\n'.join([INNER_VERSION, c.key, desc, fingerprint or _cand_fingerprint(c), str(C.INNER_SPLITS), ids])
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_path(c: FC.Candidate, fold: cv.Fold, key: str) -> Path:
    return INNER_DIR / c.key / f'{fold.test_person}_{key[:20]}.npz'


def _compute_inner(c: FC.Candidate, fold_c: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
    fp = c.make()
    p = np.full(fold_c.y_train.size, np.nan)
    t0 = time.time()
    for k, (itr, iva) in enumerate(fold_c.inner_splits()):
        f, order = inner_fold(fold_c, itr, iva, k)
        out = fp(f)
        pk = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float).reshape(-1)
        p[order] = pk
    if not np.all(np.isfinite(p)):
        raise ValueError(f'{c.key}: inner predictions missing for {int(np.sum(~np.isfinite(p)))} rows')
    return p, {'seconds': round(time.time() - t0, 2)}


def inner_predictions(c: FC.Candidate, fold: cv.Fold, cands: dict[str, FC.Candidate], *, compute: bool = True,
                      fingerprints: dict[str, str] | None = None) -> np.ndarray | None:
    """Cross-validated P(right) of candidate ``c`` for every TRAINING row of the outer ``fold`` (the fold holds the
    whole input dict). Read from the cache, else computed (and saved) when ``compute``; None when missing."""
    if c.components:
        parts = [inner_predictions(cands[k], fold, cands, compute=compute, fingerprints=fingerprints)
                 for k in c.components]
        if any(p is None for p in parts):
            return None
        return sigmoid(np.mean([logit(p) for p in parts], axis=0))
    fpr = (fingerprints or {}).get(c.key)
    key = cache_key(c, fold, fpr)
    path = cache_path(c, fold, key)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z['key']) == key:
            return np.asarray(z['p'], dtype=float)
    if not compute:
        return None
    p, info = _compute_inner(c, subfold(fold, c.input))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f'.tmp{os.getpid()}.npz')
    np.savez(tmp, p=p, key=np.array(key), seconds=np.array(info['seconds']),
             trial_ids=np.array(fold.meta_train['trial_id'].astype(str).tolist()))
    os.replace(tmp, path)
    return p


def inner_seconds(c: FC.Candidate, fold: cv.Fold, fingerprints: dict[str, str] | None = None) -> float | None:
    key = cache_key(c, fold, (fingerprints or {}).get(c.key))
    path = cache_path(c, fold, key)
    if not path.exists():
        return None
    return float(np.load(path, allow_pickle=False)['seconds'])


# ───────────────────────────── the cross-lane ensemble ─────────────────────────────


def ensemble_fit_predict(components: tuple[str, ...] = FC.ENSEMBLE_COMPONENTS,
                         cands: dict[str, FC.Candidate] | None = None) -> Callable[[cv.Fold], Any]:
    """fit_predict of the mean-logit ensemble; X = the shared input dict (the components read their own inputs)."""
    cands = cands or FC.candidates()
    fps = {k: cands[k].make() for k in components}

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        zs = []
        for k in components:
            out = fps[k](subfold(fold, cands[k].input))
            zs.append(logit(np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)))
        return sigmoid(np.mean(zs, axis=0)), {'components': list(components)}

    return fit_predict


# ───────────────────────────── the nested selection ─────────────────────────────


def make_select_fit_predict(cands: dict[str, FC.Candidate], *, compute_missing: bool = True) -> Callable:
    """fit_predict of the nested cross-lane selection: inner log loss of every candidate on the training persons,
    the lowest one refitted on every training person. X = the shared input dict."""
    order = list(cands)
    fingerprints = {k: _cand_fingerprint(c) for k, c in cands.items() if not c.components}
    made: dict[str, Callable] = {}

    def refit(k: str) -> Callable:
        if k not in made:
            c = cands[k]
            made[k] = ensemble_fit_predict(c.components, cands) if c.components else c.make()
        return made[k]

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        scores = {}
        for k in order:
            p = inner_predictions(cands[k], fold, cands, compute=compute_missing, fingerprints=fingerprints)
            if p is None:
                raise RuntimeError(f'inner predictions of {k} missing for fold {fold.test_person} (run prewarm)')
            scores[k] = log_loss(fold.y_train, p)
        best = min(order, key=lambda k: (round(scores[k], 12), order.index(k)))
        c = cands[best]
        sub = fold if c.components else subfold(fold, c.input)
        out = refit(best)(sub)
        p = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
        return p, {'chosen': best, 'inner_logloss': {k: round(v, 6) for k, v in scores.items()}}

    return fit_predict


def selection_table(result: cv.ExperimentResult) -> pd.DataFrame:
    """Per outer fold: the chosen candidate and every candidate's inner log loss (from the saved per-trial info)."""
    rows = []
    for (fold, person), g in result.predictions.groupby(['fold', 'person_id'], sort=True):
        info = json.loads(g['params'].iloc[0])
        row = {'fold': int(fold), 'person_id': person, 'chosen': info['chosen']}
        row.update({f'inner_ll:{k}': v for k, v in info['inner_logloss'].items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ───────────────────────────── parallel cache filling ─────────────────────────────


def prewarm(jobs: list[tuple[str, int]], *, log: Path | None = None) -> None:
    """Fill the inner-prediction cache for (candidate key, outer fold index) jobs, in this process. Loads the cache
    and the inputs itself (this runs outside any fit_predict)."""
    from stage2.data import load_cache
    cands = FC.candidates()
    d = load_cache()
    need = sorted({cands[k].input for k, _ in jobs if not cands[k].components})
    X = FC.build_inputs(d, need)
    folds = harness_folds(X, d.trials)
    fps = {k: _cand_fingerprint(cands[k]) for k in {k for k, _ in jobs}}
    for k, fi in jobs:
        t0 = time.time()
        p = inner_predictions(cands[k], folds[fi], cands, compute=True, fingerprints=fps)
        msg = f'{time.strftime("%H:%M:%S")} {k} fold {fi} ({folds[fi].test_person}) {time.time() - t0:.1f} s ' \
              f'inner ll {log_loss(folds[fi].y_train, p):.4f}'
        print(msg, flush=True)
        if log is not None:
            with open(log, 'a', encoding='utf-8') as fh:
                fh.write(msg + '\n')


def main(argv: list[str] | None = None) -> int:
    """python -m stage2.lanes.final_nested --jobs m3_deep:0,m3_deep:1 [--threads 1] [--log file]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True, help='comma-separated key:fold pairs, or @file with one per line')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--log', default=None)
    a = ap.parse_args(argv)
    from stage2.pipeline import limit_threads
    limit_threads(a.threads)
    spec = Path(a.jobs[1:]).read_text().split() if a.jobs.startswith('@') else a.jobs.split(',')
    jobs = [(s.split(':')[0], int(s.split(':')[1])) for s in spec if s.strip()]
    prewarm(jobs, log=Path(a.log) if a.log else None)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


__all__ = ['harness_folds', 'subfold', 'inner_fold', 'inner_predictions', 'make_select_fit_predict',
           'ensemble_fit_predict', 'selection_table', 'prewarm', 'log_loss', 'logit', 'sigmoid', 'math']
