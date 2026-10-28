"""Lane m4 finalist: the cue-free production pipeline as ``build_x`` + ``factory`` for the synthesis step.

    from stage2.lanes import m4_finalist as F
    X = F.build_x(d)                                   # per trial, label-free: its stream and cached windows
    fp = F.factory(spec={...})                         # params of results/finalists_m4.json, or hand_model='...'
    cv.run_experiment('syn_m4', fp, X, d.trials, extra=(F.build_x(nat), nat.trials))

X is a list with one label-free item per trial: the standardised full recording (t, f, acc, gyro, valid; no
phase codes), the cached oracle approach window and return window (per-trial transforms) and the platform.
``fit_predict`` builds the candidate segments of the training and test recordings (memoised by content
hash, so every fold after the first is fast), fits the whole pipeline on the training rows only
(``m4_e2e.fit_pipeline``: nested segmentation configuration, the calibrator chosen inside the fold) and returns
P(right) of each test recording (0.5 when it abstains). Deterministic (seeded by ``fold.seed``).

The oracle and return windows are cut at the app's stored events, so they are used for the TRAINING items only:
``fit_predict`` replaces those of the test items by NaN before anything else sees them (a plugged-in hand model
cannot read the held-out person's events).

Plugging in another hand model (the synthesis): ``factory(hand_model=name)`` after
``m4_models.register_hand_model(name, factory)``. The contract is WINDOW-ONLY: the object returned by the factory
has ``fit(X, y, groups, seed)`` and ``logit(X)`` on (N, 17, 64) time-normalised windows of
``standardize.CHANNELS`` (the cached ``d.approach`` layout); it is trained on the oracle windows of the training
rows and applied to the windows of the detected segments. A model that needs fixed-time windows or whole streams
needs its own adapter. Only gravity-frame channels keep the pipeline invariant to the screen-away grip.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from . import m4_detect as D
from . import m4_e2e as E

_SEG_MEMO: dict[tuple[str, str], tuple[Any, np.ndarray]] = {}


def build_x(d: Any) -> list[dict[str, Any]]:
    """One label-free item per trial of ``d`` (a Stage2Data)."""
    out = []
    for i in range(d.n):
        st = d.stream(i)
        item = {k: st[k] for k in ('t', 'f', 'acc', 'gyro', 'valid')}
        h = hashlib.sha256(np.ascontiguousarray(item['gyro'], dtype=np.float32).tobytes()
                           + np.ascontiguousarray(item['t']).tobytes()).hexdigest()[:20]
        out.append({'key': h, 'stream': item, 'oracle': d.approach[i], 'ret': d.ret[i], 'ret_ok': bool(d.ret_ok[i]),
                    'platform': str(d.trials['platform'].iloc[i])})
    return out


def _tables(items: list[dict[str, Any]]) -> E.Prepared:
    streams = [it['stream'] for it in items]
    prep = E.Prepared(streams, np.stack([it['oracle'] for it in items]), np.stack([it['ret'] for it in items]),
                      np.array([it['ret_ok'] for it in items], bool),
                      platform=np.array([it['platform'] for it in items]))
    for cfg, p in E.SEG_GRID.items():
        tabs, wins = [], []
        for k, it in enumerate(items):
            key = (cfg, it['key'])
            if key not in _SEG_MEMO:
                _SEG_MEMO[key] = D.segment_table(it['stream'], p)
            tab, W = _SEG_MEMO[key]
            tabs.append(tab.assign(stream=k))
            wins.append(W)
        import pandas as pd
        prep.seg[cfg] = (pd.concat(tabs, ignore_index=True), np.concatenate(wins))
    return prep


def factory(spec: dict[str, Any] | None = None, hand_model: str | None = None, **kw: Any):
    """fit_predict of the pipeline ``spec`` (a PipelineSpec as a dict; default: the lane's production spec).
    ``hand_model`` plugs in another registered hand classifier (``m4_models.register_hand_model``)."""
    if spec is None:
        from . import m4_pipeline as LP
        sp = LP.production_spec()
    else:
        sp = E.PipelineSpec(**{k: v for k, v in (spec if isinstance(spec, dict) else json.loads(spec)).items()})
    if hand_model is not None:
        from dataclasses import replace
        sp = replace(sp, hand_model=hand_model)
    if sp.detector != 'learned':
        raise ValueError('the m4 finalist is the learned cue-free pipeline')

    def fit_predict(fold):
        tr = list(fold.X_train)
        # the test items' event-cut windows never reach the fit or a plugged-in model (module docstring)
        te = [{**it, 'oracle': np.full_like(it['oracle'], np.nan), 'ret': np.full_like(it['ret'], np.nan),
               'ret_ok': False} for it in fold.X_test]
        prep = _tables(tr + te)
        idx = np.arange(len(tr))
        fp = E.fit_pipeline(prep, sp, idx, fold.y_train, fold.groups_train, fold.meta_train, fold.seed)
        out = fp.choose(prep, np.arange(len(tr), len(tr) + len(te)))
        info = {k: v for k, v in fp.info.items() if k in ('seg_cfg', 'features_chosen', 'seg_C')}
        if isinstance(fp.info.get('calibration'), dict):
            info['calibrator'] = fp.info['calibration'].get('chosen', fp.info['calibration'].get('method'))
        return out['p_right'].to_numpy(float), info

    return fit_predict
