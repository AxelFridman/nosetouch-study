"""Movement segmentation: an exact port of the app's causal detector (online_v1) and a better
non-causal detector for analysis (offline_v1).

online_v1
---------
Port of the algorithm specified in the doc comment of ``app/src/core/segmentation.ts`` (THE spec; read
it before touching this code). It returns the same events and indices as the TypeScript
``OnlineSegmenter`` on the same input; the literal readings a-l of the spec are implemented as written:

  * dt_i = max(0, t_i - t_{i-1}) between consecutive PUSHED samples; samples with a non-finite t are
    ignored. EMA step S + k (x - S) with k = 1 - exp(-dt/tau), S_0 = x_0; tau <= 0 (or NaN) -> S_i = x_i
    exactly. |v| = sqrt(x*x + y*y + z*z).
  * 'Present' = 3 finite numbers (acc, accg and rr alike). 'highpass': a sample with missing accg skips
    the G update (G keeps its value; the next valid sample uses its own sample-to-sample dt) and reuses
    the previous raw a.
  * Run starts are tracked per condition; a run restarts at the next sample where the condition holds.
  * A sample is processed by the state the segmenter is in when it arrives (at most one event per
    sample). ARMED: the onset test runs before the max_wait test. MOVING: end test, then too_long test.
    Peaks cover onset_index through the last sample processed in MOVING.
  * mark_go outside 'still' or with a non-finite t is ignored (it never raises). ``run_online_v1`` gives
    GO at the first sample with t >= go_t, passing THAT SAMPLE's t to mark_go; if the segmenter is not
    'still' at that sample, GO is not given at all (go_given = False). With go_t = None, GO is given at
    the first sample with t >= still_t + foreperiod_ms while in 'still' (default params.foreperiod_min_ms).
  * Baseline: means / population SDs of the SMOOTHED values of the buffer, as plain left-to-right
    sequential sums (not ``sum()``, which is compensated since Python 3.12); gravity = mean raw accg of the
    buffer samples where it is present (None if none).

Agreement with the app: indices and events are identical on every TS reference run checked (see
README); floating-point values may differ in the last bit because ``Math.exp`` differs between
JavaScript engines (V8, JavaScriptCore) and the C library CPython uses.

offline_v1
----------
Non-causal: uniform 100 Hz resampling, zero-phase low-pass of the vector components, magnitudes of the
angular velocity and of the gravity-free acceleration, per-channel thresholds = max(percentage of the
movement peak, baseline-noise floor), onset = start of the supra-threshold run that leads to the first
significant movement, end = first sustained quiet period after the last significant movement, searched
before the app's end cue so the arm's return movement cannot be merged in.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .conventions import DEFAULT_SEGMENTATION_PARAMS, OPTIONAL_SEGMENTATION_PARAM_KEYS, SEGMENTATION_PARAM_KEYS
from .preprocess import lowpass, resample_uniform

ONLINE_METHOD = 'online_v1'
OFFLINE_METHOD = 'offline_v1'
BACKDATE_MS = 300.0  # onset back-dating window before GO (spec: k_min = first index with t >= go_t - 300)

Vec = tuple[float, float, float]


def _vec(x: Any) -> Vec | None:
    """3 finite numbers -> tuple, else None (the spec's 'present')."""
    if x is None:
        return None
    try:
        a, b, c = float(x[0]), float(x[1]), float(x[2])
    except (TypeError, ValueError, IndexError):
        return None
    if math.isfinite(a) and math.isfinite(b) and math.isfinite(c):
        return (a, b, c)
    return None


def _ema_step(prev: float, x: float, dt: float, tau: float) -> float:
    """One EMA step exactly like segmentation.ts emaStep (tau <= 0 or NaN -> x itself)."""
    if not tau > 0:
        return x
    k = 1.0 - math.exp(-dt / tau)
    return prev + k * (x - prev)


def _index_or(index: Any, pos: int) -> int:
    """``index`` when it is a non-negative integer value (JS Number.isInteger semantics), else ``pos``."""
    if index is None or isinstance(index, bool):
        return pos
    try:
        f = float(index)
    except (TypeError, ValueError):
        return pos
    return int(f) if (math.isfinite(f) and f.is_integer() and f >= 0) else pos


def _seq_sum(xs: Iterable[float]) -> float:
    """Left-to-right float sum (the TS loop); builtin sum() is compensated from Python 3.12 on."""
    s = 0.0
    for x in xs:
        s += x
    return s


class OnlineSegmenterV1:
    """Causal state machine, sample by sample. States: waiting_still, still, armed, moving, ended, failed.

    ``push`` takes one sample (t relative ms, acc/accg as 3-sequences or None, rr as (alpha, beta, gamma)
    or None) and returns the events it triggered as dicts with the same fields as the TS SegmenterEvent.
    """

    def __init__(self, params: Mapping[str, float] | None = None):
        p = dict(DEFAULT_SEGMENTATION_PARAMS)
        if params:
            missing = [k for k in SEGMENTATION_PARAM_KEYS if k not in params]
            if missing:
                raise KeyError(f'segmentation params missing {missing}')
            p = {k: float(params[k]) for k in SEGMENTATION_PARAM_KEYS}
            # optional schema 1.2 keys (return_cap_ms) are not read here but echoed in the result's params,
            # like the app does: the result then equals the stored segmentation block
            p.update({k: float(params[k]) for k in OPTIONAL_SEGMENTATION_PARAM_KEYS if params.get(k) is not None})
        self.params: dict[str, float] = p
        self._state = 'waiting_still'
        # per pushed sample
        self._t: list[float] = []
        self._idx: list[int] = []
        self._gyro_raw: list[float] = []
        self._acc_raw: list[float] = []
        self._gyro_s: list[float] = []
        self._acc_s: list[float] = []
        self._still: list[bool] = []
        self._accg: list[Vec | None] = []
        # signal state
        self._acc_source: str | None = None
        self._G: list[float] | None = None
        self._prev_gyro_raw = 0.0
        self._prev_acc_raw = 0.0
        # runs / phases
        self._still_run: int | None = None
        self._still_progress = 0.0
        self._buffer: list[int] = []
        self._go_t: float | None = None
        self._go_pos: int | None = None
        self._baseline: dict[str, Any] | None = None
        self._thr_gyro = math.inf
        self._thr_acc = math.inf
        self._on_run: int | None = None
        self._onset_pos: int | None = None
        self._end_run: int | None = None
        self._end_pos: int | None = None
        self._end_detected_t: float | None = None
        self._peak_gyro: float | None = None
        self._peak_acc: float | None = None
        self._failure: str | None = None

    # ── public state ──
    @property
    def state(self) -> str:
        return self._state

    @property
    def live(self) -> dict[str, float]:
        g = self._gyro_s[-1] if self._gyro_s else 0.0
        a = self._acc_s[-1] if self._acc_s else 0.0
        return {'gyro': g, 'acc': a, 'stillProgress': self._still_progress}

    @property
    def baseline(self) -> dict[str, Any] | None:
        return None if self._baseline is None else dict(self._baseline)

    @property
    def acc_source(self) -> str:
        return self._acc_source or 'none'

    @property
    def thresholds(self) -> tuple[float, float]:
        """(thr_gyro, thr_acc) frozen at GO (inf before)."""
        return self._thr_gyro, self._thr_acc

    @property
    def go_t(self) -> float | None:
        return self._go_t

    # ── processing ──
    def push(self, t: float, acc: Any = None, accg: Any = None, rr: Any = None, index: int | None = None) -> list[dict[str, Any]]:
        """Feed one sample; returns the events it triggered (usually none). Non-finite t is ignored."""
        if self._state in ('ended', 'failed'):
            return []
        try:
            t = float(t)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(t):
            return []
        p = self.params
        pos = len(self._t)
        dt = 0.0 if pos == 0 else max(0.0, t - self._t[-1])  # reading j: decreasing t -> dt = 0
        acc_v, accg_v, rr_v = _vec(acc), _vec(accg), _vec(rr)

        # 1. raw signals
        if rr_v is not None:
            g_raw = math.sqrt(rr_v[0] * rr_v[0] + rr_v[1] * rr_v[1] + rr_v[2] * rr_v[2])
        else:
            g_raw = self._prev_gyro_raw
        self._prev_gyro_raw = g_raw
        if pos == 0:
            self._acc_source = 'linear' if acc_v is not None else ('highpass' if accg_v is not None else 'none')
        src = self._acc_source
        if src == 'linear':
            a_raw = math.sqrt(acc_v[0] * acc_v[0] + acc_v[1] * acc_v[1] + acc_v[2] * acc_v[2]) if acc_v is not None else self._prev_acc_raw
        elif src == 'highpass':
            if accg_v is not None:
                if self._G is None:
                    self._G = [accg_v[0], accg_v[1], accg_v[2]]
                else:
                    tau = p['gravity_tau_ms']
                    G = self._G
                    G[0] = _ema_step(G[0], accg_v[0], dt, tau)
                    G[1] = _ema_step(G[1], accg_v[1], dt, tau)
                    G[2] = _ema_step(G[2], accg_v[2], dt, tau)
                G = self._G
                dx, dy, dz = accg_v[0] - G[0], accg_v[1] - G[1], accg_v[2] - G[2]
                a_raw = math.sqrt(dx * dx + dy * dy + dz * dz)
            else:
                a_raw = self._prev_acc_raw
        else:
            a_raw = 0.0
        self._prev_acc_raw = a_raw
        if pos == 0:
            gs, as_ = g_raw, a_raw
        else:
            tau = p['smoothing_ms']
            gs = _ema_step(self._gyro_s[-1], g_raw, dt, tau)
            as_ = _ema_step(self._acc_s[-1], a_raw, dt, tau)
        c_still = gs < p['still_gyro_dps'] and as_ < p['still_acc_ms2']

        self._t.append(t)
        # reading a: the sample's own index when it is a non-negative integer, else its position
        self._idx.append(_index_or(index, pos))
        self._gyro_raw.append(g_raw)
        self._acc_raw.append(a_raw)
        self._gyro_s.append(gs)
        self._acc_s.append(as_)
        self._still.append(c_still)
        self._accg.append(accg_v)

        events: list[dict[str, Any]] = []
        st = self._state
        if st == 'waiting_still':
            if c_still:
                if self._still_run is None:
                    self._still_run = pos
                el = t - self._t[self._still_run]
                hold = p['still_hold_ms']
                self._still_progress = min(1.0, el / hold) if hold > 0 else 1.0
                if el >= hold:
                    self._buffer = list(range(self._still_run, pos + 1))
                    self._still_run = None
                    self._still_progress = 1.0
                    self._state = 'still'
                    events.append({'type': 'still', 't': t})
            else:
                self._still_run = None
                self._still_progress = 0.0
        elif st == 'still':
            if c_still:
                self._buffer.append(pos)
            else:
                if gs > p['onset_gyro_min_dps'] or as_ > p['onset_acc_min_ms2']:
                    events.append({'type': 'false_start', 't': t})
                else:
                    events.append({'type': 'still_lost', 't': t})
                self._state = 'waiting_still'
                self._buffer = []
                self._still_run = None
                self._still_progress = 0.0
        elif st == 'armed':
            c_on = gs > self._thr_gyro or as_ > self._thr_acc
            if c_on:
                if self._on_run is None:
                    self._on_run = pos
                if t - self._t[self._on_run] >= p['onset_sustain_ms']:
                    j0 = self._on_run
                    look_from = self._go_t - BACKDATE_MS
                    k_min = j0  # first k <= j0 with t_k >= go_t - 300 (confirmOnset)
                    for k in range(0, j0 + 1):
                        if self._t[k] >= look_from:
                            k_min = k
                            break
                    onset = k_min
                    for k in range(j0 - 1, k_min - 1, -1):
                        if self._still[k]:
                            onset = k + 1
                            break
                    if onset > j0:
                        onset = j0
                    self._onset_pos = onset
                    pg, pa = self._gyro_s[onset], self._acc_s[onset]
                    for k in range(onset + 1, pos + 1):
                        if self._gyro_s[k] > pg:
                            pg = self._gyro_s[k]
                        if self._acc_s[k] > pa:
                            pa = self._acc_s[k]
                    self._peak_gyro, self._peak_acc = pg, pa
                    self._on_run = None
                    self._state = 'moving'
                    events.append({'type': 'onset', 't': self._t[onset], 'index': self._idx[onset]})
                    return events
            else:
                self._on_run = None
            if t - self._go_t > p['max_wait_ms']:
                events.append({'type': 'no_movement', 't': t})
                self._state = 'failed'
                self._failure = 'no_movement'
        elif st == 'moving':
            if gs > self._peak_gyro:
                self._peak_gyro = gs
            if as_ > self._peak_acc:
                self._peak_acc = as_
            t_on = self._t[self._onset_pos]
            c_end = gs < p['end_gyro_dps'] and as_ < p['end_acc_ms2'] and (t - t_on) >= p['min_movement_ms']
            if c_end:
                if self._end_run is None:
                    self._end_run = pos
                c0 = self._end_run
                if t - self._t[c0] >= p['end_sustain_ms']:
                    self._end_pos = c0
                    self._end_detected_t = t
                    self._state = 'ended'
                    events.append({'type': 'end', 't': self._t[c0], 'index': self._idx[c0], 'detected_t': t})
                    return events
            else:
                self._end_run = None
            if t - t_on > p['max_movement_ms']:
                events.append({'type': 'too_long', 't': t})
                self._state = 'failed'
                self._failure = 'too_long'
        return events

    def mark_go(self, t: float) -> bool:
        """GO was emitted at time t (relative ms), right after push() of the GO sample.

        Only valid in state 'still' with a finite t; otherwise ignored (reading i). Returns whether GO
        was accepted.
        """
        try:
            t = float(t)
        except (TypeError, ValueError):
            return False
        if self._state != 'still' or not math.isfinite(t) or not self._buffer:
            return False
        p = self.params
        buf = self._buffer
        n = len(buf)
        gv = [self._gyro_s[k] for k in buf]
        av = [self._acc_s[k] for k in buf]
        gm = _seq_sum(gv) / n
        am = _seq_sum(av) / n
        gsd = math.sqrt(_seq_sum((x - gm) * (x - gm) for x in gv) / n)
        asd = math.sqrt(_seq_sum((x - am) * (x - am) for x in av) / n)
        gr = [self._accg[k] for k in buf if self._accg[k] is not None]
        gravity = None
        if gr:
            m = len(gr)
            gravity = [_seq_sum(v[0] for v in gr) / m, _seq_sum(v[1] for v in gr) / m, _seq_sum(v[2] for v in gr) / m]
        self._baseline = {'n': n, 'gyro_mean': gm, 'gyro_sd': gsd, 'acc_mean': am, 'acc_sd': asd, 'gravity': gravity}
        self._thr_gyro = max(p['onset_gyro_min_dps'], gm + p['onset_k_sd'] * gsd)
        self._thr_acc = max(p['onset_acc_min_ms2'], am + p['onset_k_sd'] * asd)
        self._go_t = t
        self._go_pos = len(self._t) - 1
        self._on_run = None
        self._state = 'armed'
        return True

    def result(self) -> dict[str, Any]:
        """SegmentationResult as stored in TrialRecord.segmentation (schema shape)."""
        onset_i = self._idx[self._onset_pos] if self._onset_pos is not None else None
        end_i = self._idx[self._end_pos] if self._end_pos is not None else None
        dur = None
        if self._onset_pos is not None and self._end_pos is not None:
            dur = self._t[self._end_pos] - self._t[self._onset_pos]
        return {
            'method': ONLINE_METHOD,
            'params': dict(self.params),
            'acc_source': self.acc_source,
            'onset_index': onset_i,
            'end_index': end_i,
            'duration_ms': dur,
            'peak_gyro_dps': self._peak_gyro,
            'peak_acc_ms2': self._peak_acc,
            'baseline': self.baseline,
        }

    def trace(self) -> dict[str, np.ndarray]:
        """Per-sample internals (for plots / debugging): t, index, raw and smoothed magnitudes, C_still."""
        return {
            't': np.asarray(self._t), 'index': np.asarray(self._idx, dtype=int),
            'gyro_raw': np.asarray(self._gyro_raw), 'acc_raw': np.asarray(self._acc_raw),
            'gyro_s': np.asarray(self._gyro_s), 'acc_s': np.asarray(self._acc_s),
            'still': np.asarray(self._still, dtype=bool),
        }


@dataclass
class OnlineRun:
    events: list[dict[str, Any]]
    result: dict[str, Any]
    go_t: float | None
    go_given: bool
    still_t: float | None
    status: str  # 'complete' (end found) | 'no_movement' | 'too_long' | 'no_go' | 'incomplete'
    thresholds: tuple[float, float]
    trace: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    @property
    def onset_index(self) -> int | None:
        return self.result['onset_index']

    @property
    def end_index(self) -> int | None:
        return self.result['end_index']

    def event(self, kind: str) -> dict[str, Any] | None:
        return next((e for e in self.events if e['type'] == kind), None)


def _col(motion: Mapping[str, Any], key: str, n: int) -> list[float | None]:
    v = motion.get(key)
    if v is None:
        return [None] * n
    out = []
    for x in v:
        if x is None:
            out.append(None)
        else:
            f = float(x)
            out.append(f if math.isfinite(f) else None)
    if len(out) != n:
        raise ValueError(f'motion column {key} has {len(out)} values, t has {n}')
    return out


def iter_motion_samples(motion: Mapping[str, Any]) -> Iterator[tuple[int, float, Vec | None, Vec | None, Vec | None]]:
    """(index, t, acc, accg, rr) per sample of a MotionStream-like mapping (lists or arrays; None/NaN = missing).

    acc/accg/rr are None unless all three components are finite (MotionSample semantics); rr is
    (alpha, beta, gamma)."""
    t = [float(x) for x in motion['t']]
    n = len(t)
    cols = {k: _col(motion, k, n) for k in (
        'acc_x', 'acc_y', 'acc_z', 'accg_x', 'accg_y', 'accg_z', 'rr_alpha', 'rr_beta', 'rr_gamma')}
    for i in range(n):
        acc = _vec((cols['acc_x'][i], cols['acc_y'][i], cols['acc_z'][i])) if cols['acc_x'][i] is not None else None
        accg = _vec((cols['accg_x'][i], cols['accg_y'][i], cols['accg_z'][i])) if cols['accg_x'][i] is not None else None
        rr = _vec((cols['rr_alpha'][i], cols['rr_beta'][i], cols['rr_gamma'][i])) if cols['rr_alpha'][i] is not None else None
        yield i, t[i], acc, accg, rr


def run_online_v1(
    motion: Mapping[str, Any],
    params: Mapping[str, float] | None,
    go_t: float | None,
    foreperiod_ms: float | None = None,
    *,
    keep_trace: bool = True,
) -> OnlineRun:
    """Port of ``runOnlineSegmentation``: feeds every sample of ``motion`` (MotionStream columns) in order.

    go_t: recorded GO time (``timing.go_ms``); GO is marked at the first sample with t >= go_t (with that
    sample's t) if the segmenter is in 'still' there. None -> automatic GO ``foreperiod_ms`` (default
    params.foreperiod_min_ms) after the latest 'still' event, like the trial controller.
    """
    seg = OnlineSegmenterV1(params)
    fp = seg.params['foreperiod_min_ms'] if foreperiod_ms is None else float(foreperiod_ms)
    events: list[dict[str, Any]] = []
    go_done = False
    go_used: float | None = None
    still_t: float | None = None
    last_still: float | None = None
    for i, t, acc, accg, rr in iter_motion_samples(motion):
        evs = seg.push(t, acc, accg, rr, index=i)
        for e in evs:
            if e['type'] == 'still':
                last_still = e['t']
                if still_t is None:
                    still_t = e['t']
            elif e['type'] in ('still_lost', 'false_start'):
                last_still = None
        events.extend(evs)
        if go_done:
            continue
        if go_t is not None:
            if t >= go_t:
                go_done = True
                if seg.state == 'still':
                    seg.mark_go(t)
                    go_used = t
        elif last_still is not None and seg.state == 'still' and t >= last_still + fp:
            seg.mark_go(t)
            go_done = True
            go_used = t
    res = seg.result()
    if seg.state == 'ended':
        status = 'complete'
    elif seg.state == 'failed':
        status = seg._failure or 'failed'
    elif go_used is None:
        status = 'no_go'
    else:
        status = 'incomplete'
    if go_used is not None:
        still_t = next((e['t'] for e in reversed(events) if e['type'] == 'still' and e['t'] <= go_used), still_t)
    return OnlineRun(
        events=events, result=res, go_t=go_used, go_given=go_used is not None, still_t=still_t, status=status,
        thresholds=seg.thresholds, trace=seg.trace() if keep_trace else {},
    )


def rerun_online_v1(record: Mapping[str, Any], params: Mapping[str, float] | None = None, **kw: Any) -> OnlineRun:
    """Re-run online_v1 on a stored TrialRecord exactly as the app ran it.

    Uses the record's own resolved ``segmentation.params`` (or ``params``) and gives GO at
    ``timing.go_ms`` - the t of the motion sample at which the app emitted GO (sync rule: GO is marked
    right after pushing the first sample with t >= go_ms, with that sample's t). The app pushed every
    stored sample, already rounded as stored, so position == stream index and the result's
    onset_index / end_index must equal the stored ones (:func:`compare_with_stored`).
    """
    seg = record.get('segmentation') or {}
    p = params or seg.get('params') or DEFAULT_SEGMENTATION_PARAMS
    go = (record.get('timing') or {}).get('go_ms')
    return run_online_v1(record['streams']['motion'], p, go, **kw)


#: Backwards-compatible name of :func:`rerun_online_v1`.
online_v1_from_record = rerun_online_v1


def compare_with_stored(record: Mapping[str, Any], run: OnlineRun | None = None, atol: float = 1e-6) -> dict[str, Any]:
    """Stored (app) online_v1 result vs this port, for one trial. Indices must match exactly."""
    run = run or online_v1_from_record(record, keep_trace=False)
    stored = record.get('segmentation') or {}
    out: dict[str, Any] = {'trial_id': record.get('trial_id')}
    for k in ('onset_index', 'end_index'):
        out[f'stored_{k}'] = stored.get(k)
        out[f'port_{k}'] = run.result[k]
        out[f'{k}_match'] = stored.get(k) == run.result[k]
    out['stored_acc_source'] = stored.get('acc_source')
    out['port_acc_source'] = run.result['acc_source']
    for k in ('duration_ms', 'peak_gyro_dps', 'peak_acc_ms2'):
        a, b = stored.get(k), run.result[k]
        out[f'stored_{k}'] = a
        out[f'port_{k}'] = b
        if a is None or b is None:
            out[f'{k}_match'] = a is None and b is None
        else:
            out[f'{k}_match'] = abs(float(a) - float(b)) <= max(atol, 1e-3 * abs(float(a)))
    out['go_given'] = run.go_given
    out['port_status'] = run.status
    out['all_match'] = bool(
        out['onset_index_match'] and out['end_index_match'] and out['stored_acc_source'] == out['port_acc_source']
    )
    return out


def port_agreement(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Re-runs online_v1 on every stored trial and counts how often the port reproduces the app.

    Returns {'n', 'n_match', 'agreement', 'onset_agreement', 'end_agreement', 'mismatches': [rows]}; a
    row is the :func:`compare_with_stored` dict of a trial whose indices or acc_source differ.
    """
    n = n_ok = n_on = n_en = 0
    bad: list[dict[str, Any]] = []
    for rec in records:
        row = compare_with_stored(rec)
        n += 1
        n_ok += int(row['all_match'])
        n_on += int(row['onset_index_match'])
        n_en += int(row['end_index_match'])
        if not row['all_match']:
            bad.append(row)
    frac = (lambda k: k / n if n else float('nan'))
    return {'n': n, 'n_match': n_ok, 'agreement': frac(n_ok), 'onset_agreement': frac(n_on),
            'end_agreement': frac(n_en), 'mismatches': bad}


# ───────────────────────────── offline_v1 ─────────────────────────────


@dataclass(frozen=True)
class OfflineParams:
    fs_hz: float = 100.0
    max_gap_ms: float = 100.0
    smooth_hz: float = 8.0  # zero-phase low-pass of the vector components before magnitudes
    gravity_hz: float = 0.5  # high-pass of accg when the platform gives no gravity-free acceleration
    baseline_ms: float = 500.0  # still window before GO for the noise floor
    k_sd: float = 5.0  # noise floor = k_sd * robust SD of the baseline
    onset_pct: float = 0.05  # of (peak - baseline)
    end_pct: float = 0.08
    sig_pct: float = 0.30  # a 'significant' movement sample reaches this fraction of the peak
    pre_go_ms: float = 300.0  # anticipation allowed before GO
    react_ms: float = 150.0  # the search ends this long after the end cue (no reaction is faster)
    end_sustain_ms: float = 200.0
    min_final_quiet_ms: float = 80.0  # a quiet tail reaching the window end is accepted from this length
    max_wait_ms: float = 6000.0
    max_window_ms: float = 20000.0


@dataclass
class OfflineSegmentation:
    ok: bool
    reason: str
    onset_ms: float | None = None
    end_ms: float | None = None
    onset_index: int | None = None  # first ORIGINAL motion sample with t >= onset_ms
    end_index: int | None = None
    duration_ms: float | None = None
    peak_gyro_dps: float | None = None
    peak_acc_ms2: float | None = None
    peak_ms: float | None = None
    channels: tuple[str, ...] = ()
    method: str = OFFLINE_METHOD

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['channels'] = ','.join(self.channels)
        return d


def _robust_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float('nan')
    mad = float(np.median(np.abs(x - np.median(x))))
    sd = 1.4826 * mad
    return sd if sd > 0 else float(np.std(x))


def activity_signals(
    t_ms: Any, acc: Any, accg: Any, rr: Any, params: OfflineParams = OfflineParams(),
) -> dict[str, np.ndarray]:
    """Uniform-grid magnitudes used by offline_v1: {'t', 'gyro', 'acc'} (NaN where unavailable).

    acc is gravity-free acceleration when available in >= half of the samples, else accg minus its
    low-pass. Magnitudes are sign- and rotation-invariant, so no gravity-sign normalisation is needed.
    """
    t = np.asarray(t_ms, dtype=float)
    cols = np.column_stack([np.asarray(acc, dtype=float), np.asarray(accg, dtype=float), np.asarray(rr, dtype=float)])
    tu, U = resample_uniform(t, cols, params.fs_hz, params.max_gap_ms)
    A, AG, W = U[:, 0:3], U[:, 3:6], U[:, 6:9]
    acc_ok = np.isfinite(A).all(axis=1)
    if acc_ok.size and acc_ok.mean() >= 0.5:
        lin = A
    else:
        lin = AG - lowpass(AG, params.fs_hz, params.gravity_hz)
    lin = lowpass(lin, params.fs_hz, params.smooth_hz)
    W = lowpass(W, params.fs_hz, params.smooth_hz)
    return {'t': tu, 'gyro': np.linalg.norm(W, axis=1), 'acc': np.linalg.norm(lin, axis=1)}


def offline_v1(
    t_ms: Any,
    acc: Any,
    accg: Any,
    rr: Any,
    go_ms: float | None,
    *,
    end_cue_ms: float | None = None,
    stop_ms: float | None = None,
    params: OfflineParams = OfflineParams(),
) -> OfflineSegmentation:
    """Non-causal onset/end detection on one trial (raw stream arrays; see the module docstring)."""
    t = np.asarray(t_ms, dtype=float)
    if t.size < 5:
        return OfflineSegmentation(False, 'too_few_samples')
    sig = activity_signals(t, acc, accg, rr, params)
    tu = sig['t']
    if tu.size < 5:
        return OfflineSegmentation(False, 'too_few_samples')
    t_first, t_last = float(tu[0]), float(tu[-1])
    stop = t_last if stop_ms is None else min(float(stop_ms), t_last)
    if go_ms is not None and math.isfinite(go_ms):
        b0, b1 = float(go_ms) - params.baseline_ms, float(go_ms)
        w0 = float(go_ms) - params.pre_go_ms
    else:
        b0, b1 = t_first, t_first + params.baseline_ms
        w0 = t_first
    if end_cue_ms is not None and math.isfinite(end_cue_ms):
        w1 = min(float(end_cue_ms) + params.react_ms, stop)
    else:
        w1 = stop
    w1 = min(w1, w0 + params.max_window_ms)
    win = (tu >= w0) & (tu <= w1)
    if win.sum() < 5:
        return OfflineSegmentation(False, 'empty_search_window')
    base = (tu >= b0) & (tu <= b1)
    if base.sum() < 5:
        base = tu <= t_first + params.baseline_ms
    chans: list[tuple[str, np.ndarray, float, float, float, float]] = []
    for name in ('gyro', 'acc'):
        x = sig[name]
        if not np.isfinite(x[win]).any():
            continue
        b = float(np.nanmedian(x[base])) if np.isfinite(x[base]).any() else float(np.nanmin(x[win]))
        sd = _robust_sd(x[base])
        if not math.isfinite(sd):
            sd = 0.0
        peak = float(np.nanmax(x[win]))
        rng_ = peak - b
        if not rng_ > 1e-9:
            continue
        floor = params.k_sd * sd / rng_
        chans.append((name, x, b, rng_, max(params.onset_pct, floor), max(params.end_pct, floor)))
    if not chans:
        return OfflineSegmentation(False, 'no_signal')
    with np.errstate(invalid='ignore'):
        Z = np.stack([(x - b) / r for _, x, b, r, _, _ in chans], axis=1)
        on_thr = np.array([c[4] for c in chans])
        end_thr = np.array([c[5] for c in chans])
        Zf = np.where(np.isfinite(Z), Z, -np.inf)
        active = (Zf > on_thr).any(axis=1)
        quiet = (np.where(np.isfinite(Z), Z, np.inf) < end_thr).all(axis=1) | ~np.isfinite(Z).any(axis=1)
        significant = (Zf >= np.maximum(params.sig_pct, on_thr)).any(axis=1)
        A = Zf.max(axis=1)
    idx = np.flatnonzero(win)
    i_start, i_stop = int(idx[0]), int(idx[-1])
    sig_idx = idx[significant[idx]]
    if sig_idx.size == 0:
        return OfflineSegmentation(False, 'no_significant_movement', channels=tuple(c[0] for c in chans))
    i_sig = int(sig_idx[0])
    if go_ms is not None and math.isfinite(go_ms) and tu[i_sig] - float(go_ms) > params.max_wait_ms:
        return OfflineSegmentation(False, 'no_movement_within_max_wait', channels=tuple(c[0] for c in chans))
    i_on = i_sig
    while i_on - 1 >= i_start and active[i_on - 1]:
        i_on -= 1
    i_last = int(sig_idx[-1])
    sustain = max(1, int(round(params.end_sustain_ms * params.fs_hz / 1000.0)))
    min_tail = max(1, int(round(params.min_final_quiet_ms * params.fs_hz / 1000.0)))
    i_end = None
    k = i_last + 1
    while k <= i_stop:
        if quiet[k]:
            j = k
            while j <= i_stop and quiet[j]:
                j += 1
            run = j - k
            if run >= sustain or (j > i_stop and run >= min_tail):
                i_end = k
                break
            k = j
        else:
            k += 1
    if i_end is None:
        return OfflineSegmentation(False, 'no_end', onset_ms=float(tu[i_on]), channels=tuple(c[0] for c in chans))
    onset_ms, end_ms = float(tu[i_on]), float(tu[i_end])
    seg_sl = slice(i_on, i_end + 1)
    peaks = {c[0]: float(np.nanmax(c[1][seg_sl])) for c in chans}
    i_pk = i_on + int(np.nanargmax(A[seg_sl]))
    return OfflineSegmentation(
        ok=True, reason='ok', onset_ms=onset_ms, end_ms=end_ms,
        onset_index=int(np.searchsorted(t, onset_ms, side='left')),
        end_index=int(min(np.searchsorted(t, end_ms, side='left'), t.size - 1)),
        duration_ms=end_ms - onset_ms, peak_gyro_dps=peaks.get('gyro'), peak_acc_ms2=peaks.get('acc'),
        peak_ms=float(tu[i_pk]), channels=tuple(c[0] for c in chans),
    )


def _motion_arrays(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = record['streams']['motion']

    def col(k: str) -> np.ndarray:
        return np.asarray([np.nan if v is None else v for v in m[k]], dtype=float)

    t = np.asarray(m['t'], dtype=float)
    acc = np.column_stack([col('acc_x'), col('acc_y'), col('acc_z')])
    accg = np.column_stack([col('accg_x'), col('accg_y'), col('accg_z')])
    rr = np.column_stack([col('rr_alpha'), col('rr_beta'), col('rr_gamma')])
    return t, acc, accg, rr


def offline_v1_from_record(record: Mapping[str, Any], params: OfflineParams = OfflineParams()) -> OfflineSegmentation:
    """offline_v1 on a TrialRecord, using its GO time, end cue time and stop time."""
    t, acc, accg, rr = _motion_arrays(record)
    tim = record.get('timing') or {}
    return offline_v1(t, acc, accg, rr, tim.get('go_ms'), end_cue_ms=tim.get('end_detected_ms'),
                      stop_ms=tim.get('stop_ms'), params=params)


def compare_segmentations(records: Iterable[Mapping[str, Any]], params: OfflineParams = OfflineParams()) -> list[dict[str, Any]]:
    """Per trial: stored online_v1 (app), this port re-run, and offline_v1, with timing differences (ms).

    Returns a list of row dicts (wrap in a DataFrame). ``port_*_match`` columns verify the TS <-> Python
    port on real data; ``offline_minus_online_*_ms`` show how the two methods differ.
    """
    rows = []
    for rec in records:
        t = rec['streams']['motion']['t']
        run = online_v1_from_record(rec, keep_trace=False)
        row = compare_with_stored(rec, run)
        off = offline_v1_from_record(rec, params)
        st = rec.get('segmentation') or {}
        on_i, en_i = st.get('onset_index'), st.get('end_index')
        on_ms = t[on_i] if on_i is not None and on_i < len(t) else None
        en_ms = t[en_i] if en_i is not None and en_i < len(t) else None
        row.update({
            'stored_onset_ms': on_ms, 'stored_end_ms': en_ms,
            'offline_ok': off.ok, 'offline_reason': off.reason,
            'offline_onset_ms': off.onset_ms, 'offline_end_ms': off.end_ms,
            'offline_duration_ms': off.duration_ms,
            'offline_minus_online_onset_ms': (off.onset_ms - on_ms) if (off.onset_ms is not None and on_ms is not None) else None,
            'offline_minus_online_end_ms': (off.end_ms - en_ms) if (off.end_ms is not None and en_ms is not None) else None,
        })
        rows.append(row)
    return rows


def segment_bounds_ms(record: Mapping[str, Any], method: str = 'auto') -> tuple[float | None, float | None, str]:
    """(onset_ms, end_ms, source) of the movement segment of a trial.

    method: 'stored' (app's online_v1 indices), 'online' (this port re-run), 'offline' (offline_v1),
    'auto' (offline_v1, falling back to stored, then to the port).
    """
    t = record['streams']['motion']['t']

    def from_idx(i: Any, j: Any) -> tuple[float | None, float | None]:
        if i is None or j is None or not (0 <= i < len(t)) or not (0 <= j < len(t)) or j <= i:
            return None, None
        return float(t[i]), float(t[j])

    if method in ('offline', 'auto'):
        off = offline_v1_from_record(record)
        if off.ok:
            return off.onset_ms, off.end_ms, 'offline'
        if method == 'offline':
            return None, None, 'offline'
    if method in ('stored', 'auto'):
        st = record.get('segmentation') or {}
        a, b = from_idx(st.get('onset_index'), st.get('end_index'))
        if a is not None:
            return a, b, 'stored'
        if method == 'stored':
            return None, None, 'stored'
    run = online_v1_from_record(record, keep_trace=False)
    a, b = from_idx(run.result['onset_index'], run.result['end_index'])
    return a, b, 'online'


def summarize_runs(runs: Sequence[OnlineRun]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in runs:
        out[r.status] = out.get(r.status, 0) + 1
    return out
