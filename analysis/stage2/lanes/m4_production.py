"""Lane m4, part 6: the production algorithm as one callable, and a demo on held-out recordings.

    from stage2.lanes.m4_production import fit_production, predict_stream
    model = fit_production()                       # every person of the primary set (or persons=[...])
    out = predict_stream(t, accg, acc, gyro, model)
    out['p_right'], out['percent_right'], out['answered'], out['attempts']

Input: one STANDARDISED stream (``stage2.standardize.stream_from_record`` or ``Stage2Data.stream(i)``): t (ms,
uniform 60 Hz grid), accg = specific force f (W3C sign, up at rest), acc = the browser's linear
acceleration (unused by the model, kept for parity with the record), gyro (deg/s, W3C device frame).

What it does (the exact parameters are in ``model.describe()``):
1. causal activity |w| (2nd-order 5 Hz one-pass low-pass), hysteresis bouts (still < lo_dps for >= end_sustain
   ends a bout, > 30 deg/s for >= 100 ms makes it a movement), each confirmed ``lookahead_s`` after its end;
2. each bout: ``standardize.canonical`` on its bounds, hand-agnostic features, P(approach / return / other);
3. every bout whose most likely class is 'approach' with P(approach) >= det_threshold is an attempt (the
   production attempt rule of ``m4_config``: no pairing); its (17, 64) window goes to the hand classifier, whose
   logit is calibrated (the calibrator chosen inside the fit among the antisymmetric ones);
4. the grip check (model-free, ``m4_detect.rest_up_z``): when the phone rests at arm's length before the first
   movement with the screen facing down or away (the screen-normal up component below ``m4_config.GRIP_GATE_UPZ``),
   the stream is not answered and the page asks to turn the screen towards the face;
5. sequential stopping rule over the attempts in time order (``m4_online.stopping_rule``): the calibrated logits
   are aggregated as P_k = sigmoid(s_k * mean z) (the production aggregation, chosen by the declared rule
   ``m4_config.AGGREGATION_RULE`` from the saved session results: s_k = 1 for the plain mean logit, or a scale
   fitted on the training persons); answer as soon as P_k >= p_stop or <= 1 - p_stop, else after k_max attempts
   answer if it reaches p_answer, else abstain ("please repeat").
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C

from . import m4_config as CFG
from . import m4_detect as D
from . import m4_e2e as E
from . import m4_models as M
from . import m4_online as O


@dataclass
class ProductionModel:
    fitted: E.FittedPipeline
    seg_params: D.SegParams
    p_stop: float = CFG.P_STOP
    k_max: int = CFG.K_MAX
    p_answer: float = CFG.P_ANSWER
    trained_on: tuple[str, ...] = ()
    agg: M.AggregateScale | None = None      # the stopping rule's aggregate scale (None: the plain mean logit)
    aggregation: str = 'mean'                # its name (m4_online.AGGREGATIONS)
    grip_gate: float | None = CFG.GRIP_GATE_UPZ   # rest-pose grip check threshold (None: off)

    def describe(self) -> dict[str, Any]:
        f = self.fitted
        return {'segmentation': asdict(self.seg_params), 'seg_cfg': f.seg_cfg, 'spec': asdict(f.spec),
                'segment_features': list(f.segclf.features) if f.segclf is not None else [],
                'calibration': f.info.get('calibration'), 'aggregation': self.aggregation,
                'pair_returns': bool(f.spec.pair_returns),
                'aggregate_scale': (dict(self.agg.s) if self.agg is not None else None),
                'aggregate_scale_recordings': f.info.get('agg_scale'),
                'causal_delay_ms': D.causal_delay_ms(self.seg_params), 'p_stop': self.p_stop, 'k_max': self.k_max,
                'p_answer': self.p_answer, 'grip_gate_upz': self.grip_gate, 'grip_rest_min_s': CFG.GRIP_REST_MIN_S,
                'grip_rest_params': asdict(D.DEFAULT), 'n_persons_trained': len(self.trained_on)}


def production_model(fitted: E.FittedPipeline, agg: M.AggregateScale | None = None, aggregation: str | None = None,
                     **kw: Any) -> ProductionModel:
    """The runtime model of a fitted pipeline. ``agg``: the stopping rule's aggregate scale (default: the
    pipeline's recording-fitted scale; pass the session-fitted one of ``m4_aggregate`` for production)."""
    seg = E.SEG_GRID[fitted.seg_cfg] if fitted.seg_cfg else D.DEFAULT
    if agg is None and aggregation is None:
        agg, aggregation = fitted.agg, 'scale_recordings'
    return ProductionModel(fitted, seg, agg=agg, aggregation=aggregation or 'mean', **kw)


def predict_stream(t: Any, accg: Any, acc: Any, gyro: Any, model: ProductionModel, *, valid: Any = None
                   ) -> dict[str, Any]:
    """P(right) of a cue-free stream with one or several nose touches (module docstring). Returns
    {p_right, percent_right, answered, status, n_attempts_used, decided_ms, rest_up_z, grip_warning,
    attempts: [...], segments: [...]}. With the grip check on (``model.grip_gate``), a stream whose rest pose at arm's
    length (``m4_detect.rest_up_z``, before the first movement) is below the threshold is not answered: status
    'grip', P = 0.5, ``grip_warning`` True (the page asks to turn the screen towards the face); its attempts are still
    listed."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(accg, dtype=float)
    a = np.asarray(acc, dtype=float) if acc is not None else np.full_like(f, np.nan)
    w = np.asarray(gyro, dtype=float)
    v = (np.isfinite(f).all(axis=1) & np.isfinite(w).all(axis=1)) if valid is None else np.asarray(valid, bool)
    st = {'t': t, 'f': f, 'acc': a, 'gyro': w, 'valid': v}
    tab, W = D.segment_table(st, model.seg_params)
    fp = model.fitted
    segs = []
    if len(tab):
        P = fp.segment_proba(tab)
        for i, r in tab.iterrows():
            segs.append({'onset_ms': float(r['onset_ms']), 'end_ms': float(r['end_ms']),
                         'decided_ms': float(r['decided_ms']), **{f'p_{c}': float(P[c][i]) for c in D.CLASSES}})
    att = fp.attempts(tab, W)
    rest = D.rest_up_z(st, D.DEFAULT, CFG.GRIP_REST_MIN_S)
    grip = model.grip_gate is not None and math.isfinite(rest) and rest < model.grip_gate
    if grip:
        dec = {'p_right': 0.5, 'n_used': 0, 'stopped': 'grip', 'answered': False, 'decided_ms': math.nan}
    else:
        dec = O.stopping_rule(att, model.agg, p_stop=model.p_stop, k_max=model.k_max, p_answer=model.p_answer)
    attempts = []
    for _, r in att.iterrows():
        attempts.append({'onset_ms': float(r['onset_ms']), 'end_ms': float(r['end_ms']),
                         'decided_ms': float(r['decided_ms']), 'p_approach': float(r['p_approach']),
                         'p_right': float(1.0 / (1.0 + math.exp(-float(r['z']))))})
    attempts.sort(key=lambda a_: a_['decided_ms'])
    p = float(dec['p_right'])
    return {'p_right': p, 'percent_right': int(round(100 * p)), 'answered': bool(dec['answered']),
            'status': dec['stopped'], 'n_attempts_used': int(dec['n_used']), 'decided_ms': dec['decided_ms'],
            'rest_up_z': rest, 'grip_warning': bool(grip), 'attempts': attempts, 'segments': segs}


def fit_production(spec: E.PipelineSpec | None = None, persons: Any = None, **kw: Any) -> ProductionModel:
    """Fit the production pipeline on the primary set (all persons, or ``persons``), docs/09 D2 cap applied,
    with the lane's production aggregation of attempts (``m4_pipeline.production_aggregation()``, read from the
    saved session results) and the stopping parameters of ``m4_config`` unless given in ``kw``."""
    from stage2 import cv
    from stage2.data import load_cache
    from . import m4_pipeline as LP
    spec = spec or LP.production_spec()
    d = load_cache()
    prep = LP.prepared(d)
    meta = d.trials
    g = meta['person_id'].astype(str).to_numpy()
    sel = np.ones(len(meta), bool) if persons is None else np.isin(g, list(persons))
    idx = np.flatnonzero(sel)
    y = meta['y_right'].astype(int).to_numpy()
    idx = idx[cv.cap_rows(g[idx], y[idx], C.CAP_PER_PERSON, C.SEED)]
    m = meta.iloc[idx].reset_index(drop=True)
    fitted = E.fit_pipeline(prep, spec, idx, y[idx], g[idx], m, C.SEED)
    how = LP.production_aggregation()
    agg = LP.production_aggregate(prep, fitted, idx, y[idx], g[idx], m, C.SEED, how=how)
    return production_model(fitted, agg=agg, aggregation=how, trained_on=tuple(sorted(set(g[idx].tolist()))), **kw)
