"""Every production model fitted on ALL persons with the Stage 2 code path that was evaluated for it.

* ``m4``    - lane m4's production pipeline: ``m4_production.fit_production()`` (detector configuration, segment
              classifier, yaw8 hand model, nested calibrator, session-fitted aggregate scale, stopping parameters);
* ``phys6`` - lane m1's finalist ``m1_lr_phys6_nested``: ``m1_models.make_linear_select(TREATMENTS, fit_agg=True)``
              on ``build_x_treat`` (data treatment and C chosen jointly on the inner splits, no calibration, the
              k-attempt exponent alpha fitted on the inner cross-fitted logits);
* ``yaw``   - the registry reference ``example_yaw_logreg``: ``stage2.scripts.reference.yaw_logreg`` (logistic
              regression on yaw_cum at 8 instants, mirror-augmented; no calibration).

PHYS6 and the yaw reference are fitted through ``final_export._all_fold`` (every primary trial a training row, the
docs/09 D2 cap, seed ``C.SEED``), the fold lane final used for its own all-persons export. Their fitted objects are
taken from the factories themselves (``_capture``), never refitted by a copy of their code.

The fits are cached in ``analysis/data/stage3_cache/fitted.pkl`` (git-ignored) with the data identity they were made
from; a cache made from another data identity is refitted.
"""
from __future__ import annotations

import contextlib
import pickle
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from stage2 import config as C

CACHE_DIR = C.DATA_DIR / 'stage3_cache'
CACHE_FILE = CACHE_DIR / 'fitted.pkl'
PHYS6 = ('yaw_q50', 'yaw_q100', 'roll_q50', 'roll_q100', 'latvel_q50', 'latvel_q100')


def data_id() -> str | None:
    from stage2.pipeline import cache_data_id
    return cache_data_id()


@contextlib.contextmanager
def _capture(module: Any, name: str, sink: list) -> Iterator[None]:
    """Record every return value of ``module.name`` while the block runs (the factories keep their fitted objects in
    closures; this takes them from the evaluated code path instead of refitting them with a copy of it)."""
    orig = getattr(module, name)

    def spy(*a: Any, **k: Any) -> Any:
        out = orig(*a, **k)
        sink.append(out)
        return out
    setattr(module, name, spy)
    try:
        yield
    finally:
        setattr(module, name, orig)


def fit_m4() -> Any:
    """Lane m4's production model on every person (``fit_production``: minutes)."""
    from stage2.lanes import m4_production as PR
    return PR.fit_production()


def fit_phys6() -> dict[str, Any]:
    """Lane m1's ``m1_lr_phys6_nested`` on every person. Returns the fitted LogisticRegression and Prep, the chosen
    option and C, the k-attempt exponent and the stored artefacts, plus the in-memory logit function itself."""
    from stage2.data import load_cache
    from stage2.lanes import final_export as FE
    from stage2.lanes import m1_experiments as ME
    from stage2.lanes import m1_models as M1
    spec = ME.BY_NAME['m1_lr_phys6_nested']
    d = load_cache()
    X = getattr(M1, spec.x)(d)                                  # build_x_treat: catalogue + amplitude-normalised
    fold = FE._all_fold(X, d)
    store: dict[int, Any] = {}
    fp = getattr(M1, spec.factory)(**{**spec.params, 'store': store})
    sink: list = []
    with _capture(M1, '_finish', sink):
        p, info = fp(fold)
    assert len(sink) == 1, len(sink)
    _p, _info, cfg, zfun = sink[0]
    option, Cval = cfg
    opt = ME.TREATMENTS[option]
    cols = list(opt['cols'])
    allcols = list(dict.fromkeys(c for v in ME.TREATMENTS.values() for c in v['cols']))
    return {'name': spec.name, 'option': option, 'C': float(Cval), 'cols': cols, 'allcols': allcols,
            'model': zfun.model, 'prep': zfun.prep, 'zfun': zfun, 'info': info, 'store': store[-1],
            'agg_alpha': float(info['agg_alpha']), 'calibration': 'none',
            'n_trials': int(fold.y_train.size), 'n_persons': int(len(set(fold.groups_train.tolist()))),
            'persons': sorted(set(fold.groups_train.tolist())),
            'train_logits': np.asarray(zfun(X.iloc[np.arange(len(X))][allcols].to_numpy(float)), float)}


def fit_yaw_ref() -> dict[str, Any]:
    """The registry reference ``example_yaw_logreg`` on every person (its own fit_predict, the fitted model taken
    from it)."""
    from sklearn.linear_model import LogisticRegression
    from stage2.data import load_cache
    from stage2.lanes import final_export as FE
    from stage2.scripts import reference as R
    d = load_cache()
    fold = FE._all_fold(d.approach, d)
    fitted: list = []

    class _Recorded(LogisticRegression):                    # the reference's own LogisticRegression, recorded
        def fit(self, X, y, sample_weight=None):            # noqa: N803 (sklearn's argument names)
            out = super().fit(X, y, sample_weight=sample_weight)
            fitted.append(self)
            return out
    _Recorded.__name__ = _Recorded.__qualname__ = 'LogisticRegression'
    orig = R.LogisticRegression
    R.LogisticRegression = _Recorded
    try:
        p, info = R.yaw_logreg(d.channels)(fold)
    finally:
        R.LogisticRegression = orig
    assert len(fitted) == 1
    m = fitted[0]
    plain = LogisticRegression(**m.get_params())            # a plain sklearn object with the same fitted state
    for k, v in vars(m).items():
        setattr(plain, k, v)
    k = d.channels.index('yaw_cum')
    return {'name': R.REFERENCE, 'model': plain, 'channel': 'yaw_cum', 'index': k, 'step': 8,
            'p_train_fold': np.asarray(p, float), 'n_trials': int(fold.y_train.size),
            'n_persons': int(len(set(fold.groups_train.tolist()))),
            'persons': sorted(set(fold.groups_train.tolist())), 'dtype': str(m.coef_.dtype)}


def fitted_models(refit: bool = False, verbose: bool = True) -> dict[str, Any]:
    """{m4, phys6, yaw, data_id, seconds}: every model fitted on all persons, from the cache when it was made from the
    current data identity (``refit`` forces a new fit)."""
    did = data_id()
    if not refit and CACHE_FILE.exists():
        with open(CACHE_FILE, 'rb') as fh:
            obj = pickle.load(fh)
        if obj.get('data_id') == did:
            return obj
    out: dict[str, Any] = {'data_id': did, 'seconds': {}}
    for name, fn in (('phys6', fit_phys6), ('yaw', fit_yaw_ref), ('m4', fit_m4)):
        t0 = time.time()
        res = fn()
        if name == 'phys6':                                 # a closure cannot be pickled: keep its parts
            res = {k: v for k, v in res.items() if k != 'zfun'}
        out[name] = res
        out['seconds'][name] = round(time.time() - t0, 1)
        if verbose:
            print(f'stage3: fitted {name} on all persons ({out["seconds"][name]:.0f} s)', flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix('.tmp')
    with open(tmp, 'wb') as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_FILE)
    return out
