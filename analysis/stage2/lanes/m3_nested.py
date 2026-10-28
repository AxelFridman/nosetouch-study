"""Lane m3: the deep configuration chosen INSIDE every outer fold among all deep configurations (nested).

``m3_deep_nested`` chooses among the architectures only (gravity-frame input, ``m3_experiments.NestedSelect``).
``m3_deep_nested_all`` chooses among every configuration of the 'deep' role (architectures, windows, channel sets,
normalisations, augmentations, heads, symmetry), each with its own per-trial input, by the cross-fitted log loss
of the TRAINING persons (each candidate's member-held-out logits). It is the nested counterpart of the role's
outer-fold choice, whose accuracy is optimistic by selection, and it costs no extra training: a candidate's fit in
an outer fold is the cached fit of its own experiment (same training rows, inner splits and seed), so only the
choice and the prediction run.

The harness needs a flat dict of arrays, so the inputs of all candidates travel as ``'<input kind>__<key>'``
(for example ``'fixed__mask'``); every candidate receives exactly the dict its own experiment received."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from stage2.lanes import m3_common as M

SEP = '__'


def make_candidate(family: str, params: dict[str, Any]) -> Any:
    if family == 'deep':
        from stage2.lanes.m3_deep import DeepFamily
        return DeepFamily(**params)
    if family == 'body':
        from stage2.lanes.m3_body import BodyDeepFamily
        return BodyDeepFamily(**params)
    raise ValueError(f'nested candidates are deep families, not {family!r}')


def x_multi(d: Any, kinds: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The per-trial inputs of several input kinds in one flat dict (``'<kind>__<key>'``)."""
    from stage2.lanes import m3_experiments as E
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        for k, v in E.build_x(kind, d).items():
            out[f'{kind}{SEP}{k}'] = v
    return out


def sub_input(X: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    """The dict one candidate's own experiment received (keys without the ``'<kind>__'`` prefix)."""
    pre = f'{kind}{SEP}'
    out = {k[len(pre):]: v for k, v in X.items() if k.startswith(pre)}
    if not out:
        raise KeyError(f'no input of kind {kind!r} in X ({sorted(X)})')
    return out


class NestedSelectMulti:
    """Per outer fold, the candidate with the lowest cross-fitted log loss on the training persons, then its
    prediction. ``candidates``: name -> {'family': 'deep' | 'body', 'x': input kind, 'params': {...}}."""

    def __init__(self, candidates: dict[str, dict[str, Any]]):
        self.candidates = {k: {'family': v['family'], 'x': v['x'], 'params': dict(v['params'])}
                           for k, v in candidates.items()}
        self.fams = {k: make_candidate(v['family'], v['params']) for k, v in self.candidates.items()}
        ns = {f.n_inner for f in self.fams.values()}
        if len(ns) != 1:
            raise ValueError('nested candidates need the same number of inner splits')
        self.n_inner = ns.pop()

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted({v['x'] for v in self.candidates.values()}))

    def config(self) -> dict[str, Any]:
        return {'candidates': self.candidates}

    def code_hash(self) -> str:
        return M.module_code_hash(M, sys.modules[__name__])

    def cache_name(self) -> str:
        return 'nestedall_' + M.digest(self.config())[:12]

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
            splits: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict[str, Any]:
        lls, states = {}, {}
        for k, fam in self.fams.items():
            st = M.cached_fit(fam, sub_input(X, self.candidates[k]['x']), y, groups, splits, seed)
            states[k] = st
            lls[k] = float(np.mean(np.logaddexp(0, -(2 * y - 1) * st['oof_z'])))
        best = min(lls, key=lambda k: (lls[k], k))
        cpu = float(sum(s.get('fit_cpu_s', 0.0) for s in states.values()))
        return {'choice': best, 'state': states[best], 'oof_z': states[best]['oof_z'], 'fit_cpu_s_all': cpu,
                'info': {'choice': best, 'inner_log_loss': {k: round(v, 5) for k, v in lls.items()},
                         'fit_cpu_s_all': round(cpu, 2)}}

    def logits(self, state: dict[str, Any], X: dict[str, np.ndarray]) -> np.ndarray:
        k = state['choice']
        return self.fams[k].logits(state['state'], sub_input(X, self.candidates[k]['x']))


def factory(candidates: dict[str, dict[str, Any]], calib: str = 'none', use_cache: bool = True,
            cache_dir: str | None = None):
    """fit_predict of a nested choice; ``use_cache`` and ``cache_dir`` also hold for every candidate's fit."""
    return M.make_fit_predict(NestedSelectMulti(candidates), calib=calib, use_cache=use_cache, cache_dir=cache_dir)
