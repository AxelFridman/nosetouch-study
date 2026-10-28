"""Cue-free movement detection, baseline v0 (no GO time, no end cue: the production situation).

In the inference app the user brings the phone to the nose once or a few times and nothing tells the
app when. The phone is first held still with the arm out, so v0 looks for the FIRST SUSTAINED MOVEMENT
THAT FOLLOWS AT LEAST ~1 s OF STILLNESS and ends it at the next stillness:

1. activity s(t) = |w| (deg/s) of the standardised stream (60 Hz, 15 Hz low-pass), gaps filled, then a
   zero-phase 5 Hz low-pass (production processes a recorded buffer, so zero phase is allowed);
2. still = s < ``still_dps`` (20 deg/s, the app's still_gyro_dps); a still run qualifies when it lasts at
   least ``min_still_s`` (1.0 s). Gyroscope only: requiring the browser's linear acceleration to be
   still too (|a| < 0.5 m/s^2) leaves fewer approaches with 1 s of stillness before them than the
   gyroscope alone (both shares: ``stillness_before_onset`` in tables/describe_summary.json, cited in
   stage2/README.md section 4);
3. the bout that starts right after a qualifying still run counts as a movement when s stays above
   ``move_dps`` (30 deg/s, the app's onset floor) for ``move_sustain_ms`` (100 ms) before the bout
   settles; onset = the first non-still sample after the still run (back-dated, like online_v1);
4. end = the first sample of the next stillness of ``end_sustain_ms`` (250 ms, the app's end_sustain)
   with s < ``end_dps`` (20 deg/s).

:func:`detect` returns every such movement in time order (a stream with several attempts gives several);
v0 uses the first. :func:`evaluate_trial` compares it with the oracle approach of a recorded trial and
says what was found: the approach (IoU >= 0.5), part of it, an earlier movement (a fidget or the arm
settling), the return of the arm, another later movement, or nothing. A dedicated lane improves the
detector; this module is the baseline and the evaluation contract.

**The recorded stillness is protocol-enforced.** Before every GO the collection app waited for a 400 ms
hold-still gate, then a 400-900 ms foreperiod, then the reaction time (its median:
``durations.reaction_median_ms`` in tables/describe_summary.json), so almost every
recorded approach follows more than 1 s of stillness. Production has no GO and no gate, so the shares of
:func:`evaluate` are UPPER BOUNDS for production. :func:`stress_evaluate` measures how v0 degrades when only
``keep_s`` seconds of that stillness remain (:func:`shorten_stillness` splices out the rest).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from motionlab.preprocess import fill_gaps_linear, lowpass

from . import config as C


@dataclass(frozen=True)
class CueFreeParams:
    still_dps: float = 20.0
    min_still_s: float = 1.0
    move_dps: float = 30.0
    move_sustain_ms: float = 100.0
    end_dps: float = 20.0
    end_sustain_ms: float = 250.0
    smooth_hz: float = 5.0
    max_duration_s: float = 12.0


@dataclass
class Detection:
    onset_ms: float
    end_ms: float
    i_on: int
    i_end: int
    still_before_s: float
    peak_dps: float
    ended: bool  # False when the stream stopped before the stillness after the movement


def activity(t: np.ndarray, gyro: np.ndarray, fs_hz: float = C.FS_HZ, smooth_hz: float = 5.0) -> np.ndarray:
    """|w| (deg/s) with gaps filled and a zero-phase low-pass at ``smooth_hz``."""
    w = fill_gaps_linear(np.asarray(t, dtype=float), np.asarray(gyro, dtype=float))
    s = np.linalg.norm(w, axis=1)
    if smooth_hz and s.size >= 12:
        s = lowpass(s, fs_hz, smooth_hz, order=2)
    return s


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def detect(t: np.ndarray, gyro: np.ndarray, params: CueFreeParams = CueFreeParams(),
           fs_hz: float = C.FS_HZ) -> list[Detection]:
    """Every movement that follows a qualifying stillness, in time order (see the module docstring)."""
    t = np.asarray(t, dtype=float)
    if t.size < 3:
        return []
    dt = float(np.median(np.diff(t)))
    s = activity(t, gyro, fs_hz, params.smooth_hz)
    still = s < params.still_dps
    n_move = max(1, int(round(params.move_sustain_ms / dt)))
    n_end = max(1, int(round(params.end_sustain_ms / dt)))
    n_still = max(1, int(round(params.min_still_s * 1000.0 / dt)))
    end_quiet = s < params.end_dps
    out: list[Detection] = []
    k = 0
    runs = _runs(still)
    for a, b in runs:
        if b <= k or b - a < n_still or b >= t.size:
            continue
        i_on = b
        # the bout lasts until the first stillness of n_end samples (or the end of the stream)
        j = i_on
        i_end = None
        while j < t.size:
            if end_quiet[j]:
                q = j
                while q < t.size and end_quiet[q]:
                    q += 1
                if q - j >= n_end:
                    i_end = j
                    break
                j = q
            else:
                j += 1
        stop = t.size if i_end is None else i_end
        above = s[i_on:stop] > params.move_dps
        sustained = any(r1 - r0 >= n_move for r0, r1 in _runs(above))
        if sustained:
            i_e = (t.size - 1) if i_end is None else i_end
            if (t[i_e] - t[i_on]) <= params.max_duration_s * 1000.0:
                out.append(Detection(float(t[i_on]), float(t[i_e]), int(i_on), int(i_e),
                                     float((t[b - 1] - t[a]) / 1000.0), float(np.max(s[i_on:i_e + 1])),
                                     i_end is not None))
        k = stop
    return out


def first_movement(t: np.ndarray, gyro: np.ndarray, params: CueFreeParams = CueFreeParams()) -> Detection | None:
    d = detect(t, gyro, params)
    return d[0] if d else None


# ───────────────────────────── evaluation against the oracle ─────────────────────────────

OUTCOMES = ('approach', 'approach_partial', 'earlier_movement', 'return', 'later_movement', 'none')


def _ok(v: Any) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))


def classify(det: Detection | None, oracle_on: float, oracle_end: float, *, cue_ms: float | None = None,
             return_onset_ms: float | None = None, return_end_ms: float | None = None) -> dict[str, Any]:
    """What a detection found, relative to the oracle approach [oracle_on, oracle_end] of one trial."""
    if det is None:
        return {'outcome': 'none', 'onset_error_ms': math.nan, 'end_error_ms': math.nan, 'iou': 0.0}
    inter = max(0.0, min(det.end_ms, oracle_end) - max(det.onset_ms, oracle_on))
    union = max(det.end_ms, oracle_end) - min(det.onset_ms, oracle_on)
    iou = inter / union if union > 0 else 0.0
    if iou >= 0.5:
        outcome = 'approach'
    elif inter > 0:
        outcome = 'approach_partial'
    elif det.end_ms <= oracle_on:
        outcome = 'earlier_movement'
    else:
        r_on = return_onset_ms if _ok(return_onset_ms) else None
        r_end = return_end_ms if _ok(return_end_ms) else None
        hits_return = r_on is not None and det.end_ms > r_on and (r_end is None or det.onset_ms < r_end)
        after_cue = _ok(cue_ms) and det.onset_ms >= float(cue_ms)
        outcome = 'return' if (hits_return or (r_on is None and after_cue)) else 'later_movement'
    return {'outcome': outcome, 'onset_error_ms': det.onset_ms - oracle_on, 'end_error_ms': det.end_ms - oracle_end,
            'iou': iou}


def evaluate_trial(stream: Mapping[str, np.ndarray], row: Mapping[str, Any],
                   params: CueFreeParams = CueFreeParams()) -> dict[str, Any]:
    """Run v0 on one standardised full stream (``Stage2Data.stream(i)``) and score it against the oracle."""
    dets = detect(stream['t'], stream['gyro'], params)
    det = dets[0] if dets else None
    res = classify(det, float(row['onset_ms']), float(row['end_ms']), cue_ms=row.get('end_detected_ms'),
                   return_onset_ms=row.get('return_onset_ms'), return_end_ms=row.get('return_end_ms'))
    res.update({'trial_id': row['trial_id'], 'person_id': row['person_id'], 'n_detections': len(dets),
                'det_onset_ms': det.onset_ms if det else math.nan, 'det_end_ms': det.end_ms if det else math.nan,
                'det_still_before_s': det.still_before_s if det else math.nan,
                'det_ended': det.ended if det else None,
                # is the oracle approach among ALL detections (a multi-attempt detector could still find it)?
                'approach_in_any': any(classify(d, float(row['onset_ms']), float(row['end_ms']))['outcome'] == 'approach'
                                       for d in dets)})
    return res


def evaluate(data: Any, params: CueFreeParams = CueFreeParams()) -> pd.DataFrame:
    """v0 on every row of a :class:`stage2.data.Stage2Data` (one row per trial)."""
    rows = []
    for i in range(data.n):
        r = data.trials.iloc[i]
        rows.append(evaluate_trial(data.stream(i), r.to_dict(), params))
    out = pd.DataFrame(rows)
    out.attrs['params'] = asdict(params)
    return out


def still_run_before(t: np.ndarray, s: np.ndarray, onset_ms: float, still_dps: float = 20.0,
                     tol_ms: float = 300.0) -> tuple[int, int] | None:
    """Indices (first, last) of the longest still run (s < still_dps) of the activity signal ``s`` that ends
    within ``tol_ms`` of the onset (its last still sample in [onset - tol, onset + tol]; a brief dip below the
    threshold just after the onset does not hide the stillness before it); None when there is none.
    Nothing is added for the smoothing: the run is measured in the detector's own signal."""
    t = np.asarray(t, dtype=float)
    best: tuple[int, int] | None = None
    for a, b in _runs(np.asarray(s) < still_dps):
        j = b - 1
        if onset_ms - tol_ms <= t[j] <= onset_ms + tol_ms and (best is None or j - a > best[1] - best[0]):
            best = (a, j)
    return best


def still_before_s(t: np.ndarray, gyro: np.ndarray, onset_ms: float, params: CueFreeParams = CueFreeParams(),
                   tol_ms: float = 300.0, fs_hz: float = C.FS_HZ) -> float:
    """Seconds of stillness (v0's activity signal) that end within ``tol_ms`` of the onset (0 when none)."""
    t = np.asarray(t, dtype=float)
    s = activity(t, gyro, fs_hz, params.smooth_hz)
    r = still_run_before(t, s, onset_ms, params.still_dps, tol_ms)
    if r is None:
        return 0.0
    dt = float(np.median(np.diff(t)))
    return float((t[r[1]] - t[r[0]] + dt) / 1000.0)


def shorten_stillness(t: np.ndarray, gyro: np.ndarray, onset_ms: float, keep_s: float,
                      params: CueFreeParams = CueFreeParams(), tol_ms: float = 300.0,
                      fs_hz: float = C.FS_HZ) -> tuple[np.ndarray, np.ndarray, float]:
    """(t, gyro, removed_ms): the stream with the still run before the onset cut to its last ``keep_s``
    seconds. The samples right after the earlier activity are removed and every later sample moves back by
    ``removed_ms`` (the grid stays uniform), so whatever preceded the stillness (the arm settling, a fidget)
    now comes ``keep_s`` before the approach, as when a user moves without holding still first."""
    t = np.asarray(t, dtype=float)
    g = np.asarray(gyro, dtype=float)
    s = activity(t, g, fs_hz, params.smooth_hz)
    r = still_run_before(t, s, onset_ms, params.still_dps, tol_ms)
    if r is None:
        return t, g, 0.0
    a, j = r
    dt = float(np.median(np.diff(t)))
    k = (j - a + 1) - max(0, int(round(keep_s * 1000.0 / dt)))
    if k <= 0:
        return t, g, 0.0
    t2 = np.concatenate([t[:a], t[a + k:] - k * dt])
    return t2, np.concatenate([g[:a], g[a + k:]], axis=0), float(k * dt)


def stress_evaluate(data: Any, keep_s: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), *,
                    random_max_s: float = 1.5, params: CueFreeParams = CueFreeParams(),
                    seed: int = C.SEED) -> pd.DataFrame:
    """v0 when the pre-onset stillness is cut to ``keep_s`` seconds (one row per trial and setting; the
    setting 'random' draws keep ~ U(0, random_max_s) per trial with a fixed seed; 'recorded' is unchanged)."""
    rng = np.random.default_rng(seed)
    rand = rng.uniform(0.0, random_max_s, data.n)
    rows = []
    for i in range(data.n):
        r = data.trials.iloc[i]
        st = data.stream(i)
        on, en = float(r['onset_ms']), float(r['end_ms'])
        settings = [('recorded', math.inf)] + [(f'{k:g}', float(k)) for k in keep_s] + [('random', float(rand[i]))]
        for name, k in settings:
            if math.isinf(k):
                t2, g2, rem = st['t'], st['gyro'], 0.0
            else:
                t2, g2, rem = shorten_stillness(st['t'], st['gyro'], on, k, params)
            dets = detect(t2, g2, params)
            det = dets[0] if dets else None
            sh = lambda v: (float(v) - rem) if _ok(v) else None  # noqa: E731
            res = classify(det, on - rem, en - rem, cue_ms=sh(r.get('end_detected_ms')),
                           return_onset_ms=sh(r.get('return_onset_ms')), return_end_ms=sh(r.get('return_end_ms')))
            rows.append({'trial_id': r['trial_id'], 'person_id': r['person_id'], 'platform': r['platform'],
                         'setting': name, 'keep_s': (math.nan if math.isinf(k) else k), 'removed_ms': rem,
                         'outcome': res['outcome'], 'onset_error_ms': res['onset_error_ms'], 'iou': res['iou'],
                         'approach_in_any': any(classify(d, on - rem, en - rem)['outcome'] == 'approach' for d in dets)})
    return pd.DataFrame(rows)


def summarize(ev: pd.DataFrame) -> dict[str, Any]:
    """Outcome shares and onset-error statistics of :func:`evaluate`."""
    n = len(ev)
    shares = {f'share_{o}': float((ev['outcome'] == o).mean()) if n else math.nan for o in OUTCOMES}
    ok = ev[ev['outcome'] == 'approach']
    err = ok['onset_error_ms'].abs()
    return {
        'n_trials': n, **shares,
        'share_approach_in_any_detection': float(ev['approach_in_any'].mean()) if n else math.nan,
        'onset_error_median_ms': float(ok['onset_error_ms'].median()) if len(ok) else math.nan,
        'onset_abs_error_p90_ms': float(err.quantile(0.9)) if len(ok) else math.nan,
        'share_onset_within_100ms': float((ev['onset_error_ms'].abs() <= 100).mean()) if n else math.nan,
        'share_onset_within_250ms': float((ev['onset_error_ms'].abs() <= 250).mean()) if n else math.nan,
        'iou_median_when_approach': float(ok['iou'].median()) if len(ok) else math.nan,
    }
