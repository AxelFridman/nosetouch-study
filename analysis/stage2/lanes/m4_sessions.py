"""Lane m4, part 4: production-like streams built from the recordings (never mixing persons).

* :func:`stress_stream` - one recording with the stillness before its approach cut to ``keep_s`` seconds
  (the samples right after the earlier activity are removed, as ``cuefree.shorten_stillness`` does, but on
  every channel), so whatever preceded the stillness comes ``keep_s`` before the approach.
* :func:`build_sessions` - synthetic multi-attempt sessions: several recordings of ONE person and ONE hand,
  each with its pre-approach stillness cut to ``keep_s``, its leading stillness (before its first movement)
  removed and its trailing stillness cut to ``keep_s``, concatenated on one 60 Hz clock. Every junction
  therefore holds at most ``keep_s`` of stillness, and the stillness right before each approach is
  ``keep_s`` (plus at most ``keep_s`` from the previous recording when nothing moved in between).
  The recordings of a (person, hand) cell are shuffled with a seed and dealt into sessions of
  ``n_attempts`` recordings, a new session starting every ``stride`` recordings of the shuffled list (the list
  wraps around, so sessions overlap when stride < n_attempts; a cell with at most ``n_attempts`` recordings gives
  one session of all of them).

The streams are label-free (t, f, acc, gyro, valid); the events of each recording are carried separately in
session time (``events``) and are read by the evaluation only.
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.cuefree import CueFreeParams, activity, still_run_before

from . import m4_detect as D

CHANNEL_KEYS = ('f', 'acc', 'gyro', 'valid')


def _still_runs(t: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    p = CueFreeParams()
    s = activity(t, gyro, C.FS_HZ, p.smooth_hz)
    return s, D._runs(s < p.still_dps)


def _remove(st: Mapping[str, np.ndarray], a: int, b: int) -> tuple[dict[str, np.ndarray], float]:
    """Remove samples [a, b) and close the gap on the uniform grid (later samples move back)."""
    if b <= a:
        return {k: np.asarray(v) for k, v in st.items()}, 0.0
    t = np.asarray(st['t'], dtype=float)
    dt = float(np.median(np.diff(t)))
    k = b - a
    out = {'t': np.concatenate([t[:a], t[b:] - k * dt])}
    for c in CHANNEL_KEYS:
        if c in st:
            v = np.asarray(st[c])
            out[c] = np.concatenate([v[:a], v[b:]], axis=0)
    return out, float(k * dt)


def stress_stream(st: Mapping[str, np.ndarray], onset_ms: float, keep_s: float) -> tuple[dict[str, np.ndarray], float]:
    """(stream, removed_ms): the still run that ends within 300 ms of ``onset_ms`` cut to its last ``keep_s``
    seconds (cuefree.shorten_stillness on every channel). Events after the cut move back by removed_ms."""
    t = np.asarray(st['t'], dtype=float)
    s, _ = _still_runs(t, st['gyro'])
    r = still_run_before(t, s, onset_ms, CueFreeParams().still_dps, 300.0)
    base = {k: np.asarray(st[k]) for k in ('t', *CHANNEL_KEYS) if k in st}
    if r is None:
        return base, 0.0
    a, j = r
    dt = float(np.median(np.diff(t)))
    k = (j - a + 1) - max(0, int(round(keep_s * 1000.0 / dt)))
    if k <= 0:
        return base, 0.0
    return _remove(base, a, a + k)


@dataclass
class Session:
    person_id: str
    y: int
    stream: dict[str, np.ndarray]
    events: pd.DataFrame            # per recording: trial row, onset/end/cue/return in session time
    keep_s: float
    sid: str = ''
    meta: dict[str, Any] = field(default_factory=dict)


def _shift(v: Any, d: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x - d if math.isfinite(x) else math.nan


def trimmed_recording(st: Mapping[str, np.ndarray], ev: Mapping[str, Any], keep_s: float
                      ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """One recording prepared for a session (module docstring): pre-approach stillness -> keep_s, leading
    stillness removed, trailing stillness -> keep_s. Returns the stream and its events in its own new time."""
    on = float(ev['onset_ms'])
    s1, rem1 = stress_stream(st, on, keep_s)
    evs = {c: _shift(ev.get(c), rem1) for c in D.EVENT_COLUMNS}
    t = s1['t']
    _, runs = _still_runs(t, s1['gyro'])
    # leading stillness (unless it is the pre-approach run, already cut to keep_s)
    if runs and runs[0][0] == 0:
        a, b = runs[0]
        if t[b - 1] < evs['onset_ms'] - 300.0:
            s1, rem2 = _remove(s1, 0, b)
            evs = {c: _shift(v, rem2) for c, v in evs.items()}
            t = s1['t']
            _, runs = _still_runs(t, s1['gyro'])
    # trailing stillness after the last movement -> keep_s
    n = t.size
    if runs and runs[-1][1] == n and t[runs[-1][0]] > evs['end_ms']:
        a, b = runs[-1]
        dt = float(np.median(np.diff(t)))
        keep_n = max(1, int(round(keep_s * 1000.0 / dt)))
        if b - a > keep_n:
            s1 = {k: np.asarray(v)[: a + keep_n] for k, v in s1.items()}
    t0 = float(s1['t'][0])
    s1['t'] = s1['t'] - t0
    evs = {c: _shift(v, t0) for c, v in evs.items()}
    return s1, evs


def concat_streams(parts: Sequence[Mapping[str, np.ndarray]]) -> tuple[dict[str, np.ndarray], list[float]]:
    """Concatenate streams on one uniform clock; returns the stream and each part's time offset."""
    dt = 1000.0 / C.FS_HZ
    out: dict[str, list[np.ndarray]] = {k: [] for k in ('t', *CHANNEL_KEYS)}
    offs, t_next = [], 0.0
    for p in parts:
        t = np.asarray(p['t'], dtype=float)
        t = t - t[0]
        offs.append(t_next)
        out['t'].append(t + t_next)
        for c in CHANNEL_KEYS:
            out[c].append(np.asarray(p[c]))
        t_next = t_next + t[-1] + dt
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}, offs


def build_sessions(trials: pd.DataFrame, get_stream: Any, *, n_attempts: int = 5, keep_s: float = 1.0,
                   seed: int = C.SEED, persons: Sequence[str] | None = None, stride: int | None = None) -> list[Session]:
    """Synthetic multi-attempt sessions (module docstring). ``get_stream(i)`` returns row i's stream dict;
    ``stride`` defaults to ``n_attempts`` (no overlap)."""
    sessions: list[Session] = []
    T = trials.reset_index(drop=True)
    pids = sorted(T['person_id'].astype(str).unique()) if persons is None else list(persons)
    stride = n_attempts if stride is None else int(stride)
    for pid in pids:
        for y in (0, 1):
            idx = np.flatnonzero((T['person_id'].astype(str) == pid).to_numpy() & (T['y_right'].astype(int) == y).to_numpy())
            if not idx.size:
                continue
            rng = np.random.default_rng([seed, zlib.crc32(f'{pid}-{y}'.encode())])
            perm = idx[rng.permutation(idx.size)]
            k = min(n_attempts, perm.size)
            n_sess = 1 if perm.size <= n_attempts else math.ceil(perm.size / stride)
            for s in range(n_sess):
                pick = [perm[(s * stride + j) % perm.size] for j in range(k)]
                parts, evs = [], []
                for i in pick:
                    st, ev = trimmed_recording(get_stream(int(i)), T.iloc[int(i)].to_dict(), keep_s)
                    parts.append(st)
                    evs.append({'row': int(i), **ev})
                stream, offs = concat_streams(parts)
                ev = pd.DataFrame(evs)
                for c in D.EVENT_COLUMNS:
                    ev[c] = ev[c].to_numpy(float) + np.asarray(offs)
                ev['part_start_ms'] = offs
                sessions.append(Session(pid, y, stream, ev, keep_s, sid=f'{pid}-{"RL"[1 - y]}-{s}-k{keep_s:g}'))
    return sessions


def attempt_kinds(att: pd.DataFrame, ev: pd.DataFrame) -> list[str]:
    """The kind of every accepted attempt (rows of ``att``, in their order): 'approach' (IoU >= 0.5 with a true
    approach), 'approach_partial', 'return', 'other_before' or 'other_after' (session_detection)."""
    return session_detection(att, ev)['kinds'].split(';') if len(att) else []


def session_detection(att: pd.DataFrame, ev: pd.DataFrame) -> dict[str, Any]:
    """How the attempts accepted in a session relate to its true approaches: how many true approaches were
    accepted (IoU >= 0.5 or partial overlap), and how many accepted attempts were a return or another movement
    ('other_before': before the approach of the recording it lies in, i.e. the arm settling at arm's length;
    'other_after': after that approach and outside its return)."""
    n_true = len(ev)
    hit = np.zeros(n_true, bool)
    kinds = []
    for _, a in att.iterrows():
        oc, iou = D.classify_arrays(np.array([a['onset_ms']]), np.array([a['end_ms']]), ev['onset_ms'].to_numpy(float),
                                    ev['end_ms'].to_numpy(float), ev['end_detected_ms'].to_numpy(float),
                                    ev['return_onset_ms'].to_numpy(float), ev['return_end_ms'].to_numpy(float))
        j = int(np.argmax(iou))
        if iou[j] >= 0.5:
            kinds.append('approach')
            hit[j] = True
        elif iou[j] > 0:
            kinds.append('approach_partial')
            hit[j] = True
        else:
            # not overlapping any approach: a return of any recording, or another movement
            is_ret = (oc == 'return').any() and _overlaps_return(a, ev)
            kinds.append('return' if is_ret else _other_kind(a, ev))
    n_other = sum(k.startswith('other') for k in kinds)
    return {'n_true': n_true, 'n_found': int(hit.sum()), 'n_attempts': len(att),
            'n_false_return': kinds.count('return'), 'n_false_other': n_other,
            'n_false_other_before': kinds.count('other_before'), 'n_false_other_after': kinds.count('other_after'),
            'kinds': ';'.join(kinds)}


def _other_kind(a: Mapping[str, Any], ev: pd.DataFrame) -> str:
    """'other_before' when the movement ends before the approach of the recording (part) it starts in."""
    starts = ev['part_start_ms'].to_numpy(float) if 'part_start_ms' in ev else np.zeros(len(ev))
    j = int(np.searchsorted(starts, float(a['onset_ms']), side='right') - 1)
    j = min(max(j, 0), len(ev) - 1)
    return 'other_before' if float(a['end_ms']) <= float(ev['onset_ms'].iloc[j]) else 'other_after'


def _overlaps_return(a: Mapping[str, Any], ev: pd.DataFrame) -> bool:
    on, end = float(a['onset_ms']), float(a['end_ms'])
    for _, r in ev.iterrows():
        lo = r['end_detected_ms'] if math.isfinite(r['end_detected_ms']) else r['end_ms']
        hi = r['return_end_ms'] if math.isfinite(r['return_end_ms']) else lo + 4000.0
        if end > lo and on < hi:
            return True
    return False
