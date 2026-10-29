"""``reference_predict``: every intermediate of the production path for one stream (docs/stage3/PORT_SPEC.md 4.6).

Everything is computed by the Stage 2 functions themselves - ``standardize.stream_from_record``, ``m4_detect``
(activity, find_segments, segment_table, window_for, rest_up_z), ``m4_e2e.FittedPipeline`` (segment_proba, attempts),
``m4_online`` (aggregate, stopping_rule), ``final_infer`` and ``m1_features`` - and the m4 part is checked against
``m4_production.predict_stream`` on the same stream (``check=True`` raises on any difference). The only new code is
glue and the per-model stopping rule of the page (PORT_SPEC 3.5: lane m4's rule applied to each model's own P_k over
the shared attempt list, skipping the attempts a model cannot score), which for m4 is checked to equal
``m4_online.stopping_rule``.

Modes. ``offline``: the decision on the complete recording (``predict_stream``'s view; a recording that ends without a
final decision is 'stream_end' or 'no_attempt'). ``live``: what the page's engine shows after the same samples - only
attempts that are ended, have their whole look-ahead in the buffer and were decided at least ``settle_ms`` before
the last grid sample count (the engine freezes a bout once the stream extends ``settle_ms`` past its decision time,
APP_SPEC 4.2), and without a final decision the status is 'watching' until ``max_stream_ms``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from stage2 import config as C

FORMAT = 'nosetouch-ref-1'
ENGINE_SIGN = {'webkit': -1, 'blink': 1, 'gecko': 1, 'unknown': 1}
MOTION_COLUMNS = ('t', 'acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma')


def production_normalization(engine: str | None) -> Any:
    """The page's normalisation: the engine prior (webkit -1, every other engine +1), W3C deg/s, nothing else."""
    from motionlab.conventions import GravitySign, Normalization, RateMapping
    e = engine if engine in ('webkit', 'blink', 'gecko') else None
    sign = ENGINE_SIGN.get(engine or 'unknown', 1)
    return Normalization(gravity=GravitySign(sign, 'engine', False, engine=e), rates=RateMapping(verified=True), engine=e)


def as_record(source: Any) -> dict[str, Any]:
    """A minimal TrialRecord from a record, a ``streams.motion`` column dict or a raw buffer
    ({t, acc: [[x, y, z] | None], accg, rr: [[alpha, beta, gamma] | None]})."""
    if isinstance(source, dict) and 'streams' in source:
        m = source['streams']['motion']
    elif isinstance(source, dict) and 'acc_x' in source:
        m = source
    elif isinstance(source, dict) and 'accg' in source and 'rr' in source:
        cols: dict[str, list] = {c: [] for c in MOTION_COLUMNS}
        cols['t'] = list(source['t'])
        for key, names in (('acc', ('acc_x', 'acc_y', 'acc_z')), ('accg', ('accg_x', 'accg_y', 'accg_z')),
                           ('rr', ('rr_alpha', 'rr_beta', 'rr_gamma'))):
            for v in source[key]:
                for j, nm in enumerate(names):
                    cols[nm].append(None if v is None else v[j])
        m = cols
    else:
        raise TypeError('source: a TrialRecord, a streams.motion column dict or a raw buffer')
    n = len(m['t'])
    motion = {c: list(m[c]) for c in MOTION_COLUMNS}
    motion['ts'] = [None] * n
    return {'trial_id': None, 'streams': {'motion': motion}}


def _num(v: Any) -> Any:
    if v is None:
        return None
    x = float(v)
    return x if math.isfinite(x) else None


def _rows(X: np.ndarray) -> list:
    return [[_num(v) for v in row] for row in np.asarray(X, dtype=float)]


def _vec(x: np.ndarray) -> list:
    return [_num(v) for v in np.asarray(x, dtype=float).reshape(-1)]


def percent(p: float) -> int:
    return int(round(100 * p))                              # Python's round: half to even (PORT_SPEC finding 9)


def stopping(pk: list[float], decided: list[float], *, p_stop: float, k_max: int, p_answer: float,
             stream_ended: bool = True) -> dict[str, Any]:
    """lane m4's stopping rule (m4_online.stopping_rule) on precomputed P_1..P_K of one model; with
    ``stream_ended`` False an undecided stream is 'watching' (the page's live view)."""
    p, k = 0.5, 0
    for k in range(1, min(k_max, len(pk)) + 1):
        p = pk[k - 1]
        if max(p, 1 - p) >= p_stop:
            return {'p_right': p, 'n_used': k, 'stopped': 'confident', 'answered': True, 'final': True,
                    'decided_ms': float(decided[k - 1])}
    if k == 0:
        return {'p_right': 0.5, 'n_used': 0, 'stopped': 'no_attempt' if stream_ended else 'watching',
                'answered': False, 'final': bool(stream_ended), 'decided_ms': math.nan}
    if k < k_max and not stream_ended:
        return {'p_right': p, 'n_used': k, 'stopped': 'watching', 'answered': False, 'final': False,
                'decided_ms': float(decided[k - 1])}
    answered = max(p, 1 - p) >= p_answer
    return {'p_right': p, 'n_used': k, 'stopped': 'k_max' if k >= k_max else 'stream_end', 'answered': bool(answered),
            'final': True, 'decided_ms': float(decided[k - 1])}


def _rest_details(st: dict[str, np.ndarray], min_s: float) -> dict[str, Any]:
    """The inputs of ``m4_detect.rest_up_z`` with the DEFAULT parameters (its value is taken from the function)."""
    from stage2.lanes import m4_detect as D
    t, f = st['t'], st['f']
    if t.size == 0:
        return {'n_sel': 0, 'fallback': False, 'first_onset_ms': None}
    s = D.activity(t, st['gyro'], D.DEFAULT)
    segs = D.find_segments(t, st['gyro'], D.DEFAULT, s=s)
    first_on = segs[0].onset_ms if segs else math.inf
    nrm = np.linalg.norm(f, axis=1)
    ok = np.isfinite(nrm) & (nrm > 0)
    sel = ok & (t < first_on) & (s < D.DEFAULT.lo_dps)
    fallback = bool(sel.sum() < max(1, int(round(min_s * C.FS_HZ))))
    if fallback:
        sel = ok & (t < t[0] + min_s * 1000.0)
    return {'n_sel': int(sel.sum()), 'fallback': fallback, 'first_onset_ms': _num(first_on),
            'n_default_bouts': len(segs)}


def reference_predict(source: Any, models: Any, *, engine: str | None = None, mode: str = 'offline',
                      settle_ms: float = 1000.0, max_stream_ms: float | None = None, check: bool = True,
                      include_stream: bool = True) -> dict[str, Any]:
    """Every intermediate of the production path for one stream (module docstring). ``source``: a TrialRecord, a
    ``streams.motion`` column dict, a raw buffer, or a standardised stream dict {t, f, acc, gyro} (offline only);
    ``models``: a ``runtime.ModelSet``; ``engine``: the browser engine whose accelerometer sign the raw values carry."""
    from stage2 import standardize as S
    from stage2.lanes import m4_config as CFG
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_models as M
    from stage2.lanes import m4_online as O
    from stage2.lanes import m4_production as PR
    if mode not in ('offline', 'live'):
        raise ValueError(mode)
    model = models.m4
    fp = model.fitted
    seg = model.seg_params
    out: dict[str, Any] = {'format': FORMAT, 'mode': mode, 'engine': engine, 'models_source': models.source}
    if isinstance(source, dict) and 'f' in source and 'gyro' in source:
        t = np.asarray(source['t'], float)
        f = np.asarray(source['f'], float)
        acc = np.asarray(source['acc'], float) if source.get('acc') is not None else np.full_like(f, np.nan)
        gyro = np.asarray(source['gyro'], float)
        out.update({'sign': None, 'n_raw': None, 'repeats_dropped': None})
    else:
        norm = production_normalization(engine)
        stream = S.stream_from_record(as_record(source), norm)
        t, f, acc, gyro = stream.t, stream.f, stream.acc, stream.gyro
        out.update({'sign': int(norm.sign), 'n_raw': int(stream.t_raw.size), 'repeats_dropped': int(stream.repeats_dropped),
                    'normalization': {'gravitySign': int(norm.sign), 'rrVerdict': norm.rates.verdict,
                                      'rrUnits': norm.rates.units, 'engine': norm.engine, 'quantizeIos': False,
                                      'source': 'engine_prior'}})
    valid = np.isfinite(f).all(axis=1) & np.isfinite(gyro).all(axis=1)          # as predict_stream builds it
    st = {'t': t, 'f': f, 'acc': acc, 'gyro': gyro, 'valid': valid}
    n = int(t.size)
    out['n_grid'] = n
    out['t_first_ms'], out['t_last_ms'] = (_num(t[0]), _num(t[-1])) if n else (None, None)
    if include_stream:
        out['stream'] = {'t': _vec(t), 'f': _rows(f), 'acc': _rows(acc), 'gyro': _rows(gyro),
                         'valid': [bool(v) for v in valid]}
    # ---- detector ----
    act = D.activity(t, gyro, seg)
    bouts = D.find_segments(t, gyro, seg, s=act)
    if include_stream:
        out['activity'] = _vec(act)
    dt = float(np.median(np.diff(t))) if n >= 2 else math.nan
    n_end = max(1, int(round(seg.end_sustain_ms / dt))) if n >= 3 else None
    n_look = max(n_end, int(round(seg.lookahead_s * 1000.0 / dt))) if n >= 3 else None
    out['detector'] = {'seg_cfg': fp.seg_cfg, 'seg_params': {k: getattr(seg, k) for k in seg.__dataclass_fields__},
                       'causal_delay_ms': D.causal_delay_ms(seg), 'dt_ms': _num(dt), 'n_end': n_end, 'n_look': n_look,
                       'n_move': max(1, int(round(seg.sustain_ms / dt))) if n >= 3 else None}
    tab, W = D.segment_table(st, seg)
    if len(tab) != len(bouts):
        raise AssertionError('segment_table and find_segments disagree')
    P = fp.segment_proba(tab) if len(tab) else {c: np.zeros(0) for c in D.CLASSES}
    cls = (np.array(D.CLASSES)[np.argmax(np.column_stack([P[c] for c in D.CLASSES]), axis=1)] if len(tab)
           else np.zeros(0, dtype=object))
    att = fp.attempts(tab, W)
    att_by_seg = {int(r['seg']): r for _, r in att.iterrows()}
    feats = list(fp.segclf.features)
    segments = []
    for i, b in enumerate(bouts):
        row = tab.iloc[i]
        assert int(row['seg']) == b.index and float(row['onset_ms']) == b.onset_ms
        ok = bool(row['canonical_ok'])
        rec: dict[str, Any] = {
            'seg': b.index, 'onset_ms': b.onset_ms, 'end_ms': b.end_ms, 'decided_ms': b.decided_ms, 'i_on': b.i_on,
            'i_end': b.i_end, 'still_before_s': b.still_before_s, 'still_after_s': b.still_after_s,
            'started_active': bool(b.started_active), 'ended': bool(b.ended), 'peak_dps': b.peak_dps,
            'canonical_ok': ok, 'crop': {'start_ms': b.onset_ms - seg.crop_s * 1000.0,
                                         'stop_ms': b.end_ms + seg.lookahead_s * 1000.0},
            'features': {k: _num(row[k]) for k in feats},
            'proba': {c: float(P[c][i]) for c in D.CLASSES}, 'cls': str(cls[i]),
            'is_attempt': b.index in att_by_seg,
            'live_frozen': bool(b.ended and n_look is not None and b.i_end + n_look - 1 <= n - 1
                                and b.decided_ms + settle_ms <= float(t[-1])),
            'freeze_after_ms': b.decided_ms + settle_ms,
        }
        if ok:
            win, can = D.window_for(st, b.onset_ms, b.end_ms, seg.crop_s, seg.lookahead_s)
            w32 = np.asarray(win, dtype=np.float32)
            if not np.array_equal(w32, W[i], equal_nan=True):
                raise AssertionError('window_for and segment_table give different windows')
            rec['crop'].update({'n': int(can.t.size), 't0_ms': float(can.t[0]), 'i_on': int(can.i_on),
                                'i_end': int(can.i_end)})
            rec['gravity'] = {'method': can.gravity_method, 'pre_n': int(can.rest_pre_n), 'post_n': int(can.rest_post_n),
                              'closure_deg': _num(can.closure_deg),
                              'magnitude': _num(np.linalg.norm(can.g[can.i_on]))}   # |g| of every row (gravity_v1)
            rec['window'] = _rows(W[i])
            outs: dict[str, Any] = {}
            if b.index in att_by_seg:                       # the attempt rows exactly as predict_stream computes them
                r = att_by_seg[b.index]
                zr, z = float(r['z_raw']), float(r['z'])
                outs['m4'] = {'available': True, 'z_raw': zr, 'p': float(fp.calib(np.array([zr]))[0]), 'z': z,
                              'p_attempt': 1.0 / (1.0 + math.exp(-z))}
            else:
                outs['m4'] = models.m4_predict(W[i])
            cn = models.cnn_predict(W[i])
            rec['body'] = _rows(np.asarray(cn.pop('body'), dtype=np.float32))
            outs['cnn'] = cn
            outs['phys6'] = models.phys6.predict(W[i], b.onset_ms, b.end_ms)
            outs['yaw'] = models.yaw.predict(W[i], b.onset_ms, b.end_ms)
            rec['models'] = {k: {kk: (_num(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in o.items()}
                             for k, o in outs.items()}
        segments.append(rec)
    out['segments'] = segments
    # ---- attempts in decision order (stable: decided_ms, then seg) ----
    order = sorted((s for s in segments if s['is_attempt']), key=lambda s: (s['decided_ms'], s['seg']))
    if mode == 'live':
        order = [s for s in order if s['live_frozen']]
    out['attempts'] = [s['seg'] for s in order]
    # ---- grip check (model-free) ----
    rest = D.rest_up_z(st, D.DEFAULT, CFG.GRIP_REST_MIN_S)
    grip = model.grip_gate is not None and math.isfinite(rest) and rest < model.grip_gate
    # live: the page applies the gate when it freezes a bout (detect/engine.ts), so before any bout is frozen the
    # stream is still 'watching' even when the rest pose already fails the check
    grip_live = grip and (mode == 'offline' or any(s['live_frozen'] for s in segments))
    out['grip'] = {'rest_up_z': _num(rest), 'gate_upz': model.grip_gate, 'grip_warning': bool(grip),
                   **_rest_details(st, CFG.GRIP_REST_MIN_S)}
    # ---- every model's P_k and decision ----
    stream_ended = mode == 'offline' or (max_stream_ms is not None and n > 0 and float(t[-1]) >= max_stream_ms)
    decisions = {}
    for mid in ('cnn', 'm4', 'phys6', 'yaw'):
        avail = [s for s in order if s['models'][mid]['available']]
        outs = [s['models'][mid] for s in avail]
        pk = [models.aggregate(mid, outs[:k]) for k in range(1, len(outs) + 1)]
        dec = stopping(pk, [s['decided_ms'] for s in avail], p_stop=model.p_stop, k_max=model.k_max,
                       p_answer=model.p_answer, stream_ended=stream_ended)
        if grip_live:
            dec = {'p_right': 0.5, 'n_used': 0, 'stopped': 'grip', 'answered': False, 'final': True,
                   'decided_ms': math.nan}
        decisions[mid] = {**{k: (_num(v) if isinstance(v, float) else v) for k, v in dec.items()},
                          'percent_right': percent(float(dec['p_right'])),
                          'k_curve': [{'k': k + 1, 'p': pk[k], 'seg': avail[k]['seg'],
                                       'decided_ms': avail[k]['decided_ms']} for k in range(len(pk))],
                          'n_unavailable': len(order) - len(avail)}
    out['decisions'] = decisions
    # ---- the m4 view must be predict_stream's, exactly ----
    if mode == 'offline':
        ps = PR.predict_stream(t, f, acc, gyro, model)
        out['m4_predict_stream'] = {k: (_num(v) if isinstance(v, float) else v) for k, v in ps.items()
                                    if k not in ('attempts', 'segments')}
        out['m4_predict_stream']['attempts'] = ps['attempts']
        out['m4_predict_stream']['segments'] = ps['segments']
        if check:
            _check_m4(out, ps, att, model, grip)
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if a is None and isinstance(b, float) and math.isnan(b):
        return True
    return a == b


def _check_m4(out: dict[str, Any], ps: dict[str, Any], att: Any, model: Any, grip: bool) -> None:
    """The reference's m4 decision, attempts and segment probabilities equal predict_stream's (bitwise)."""
    from stage2.lanes import m4_online as O
    dec = out['decisions']['m4']
    errs = []
    if not grip:
        ref = O.stopping_rule(att, model.agg, p_stop=model.p_stop, k_max=model.k_max, p_answer=model.p_answer)
        for k_ref, k_out in (('p_right', 'p_right'), ('n_used', 'n_used'), ('stopped', 'stopped'),
                             ('answered', 'answered'), ('decided_ms', 'decided_ms')):
            if not _same(dec[k_out], ref[k_ref]):
                errs.append(f'stopping_rule {k_ref}: {dec[k_out]!r} != {ref[k_ref]!r}')
    for k_ps, k_out in (('p_right', 'p_right'), ('percent_right', 'percent_right'), ('answered', 'answered'),
                        ('status', 'stopped'), ('n_attempts_used', 'n_used'), ('decided_ms', 'decided_ms')):
        if not _same(dec[k_out], ps[k_ps]):
            errs.append(f'predict_stream {k_ps}: {dec[k_out]!r} != {ps[k_ps]!r}')
    if not _same(out['grip']['rest_up_z'], ps['rest_up_z']) or out['grip']['grip_warning'] != ps['grip_warning']:
        errs.append('grip')
    segs = {s['seg']: s for s in out['segments']}
    want = [(a['onset_ms'], a['end_ms'], a['decided_ms'], a['p_approach'], a['p_right']) for a in ps['attempts']]
    have = [(segs[k]['onset_ms'], segs[k]['end_ms'], segs[k]['decided_ms'], segs[k]['proba']['approach'],
             segs[k]['models']['m4']['p_attempt']) for k in out['attempts']]
    if want != have:
        errs.append(f'attempts {have} != {want}')
    sw = [(s['onset_ms'], s['end_ms'], s['decided_ms'], s['p_approach'], s['p_return'], s['p_other'])
          for s in ps['segments']]
    sh = [(s['onset_ms'], s['end_ms'], s['decided_ms'], s['proba']['approach'], s['proba']['return'], s['proba']['other'])
          for s in out['segments']]
    if sw != sh:
        errs.append('segments')
    if errs:
        raise AssertionError('reference_predict differs from predict_stream: ' + '; '.join(errs))
