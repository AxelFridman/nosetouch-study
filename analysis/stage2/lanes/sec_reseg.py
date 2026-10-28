"""Lane sec: label-blind re-segmentation and re-standardisation of the approach (is a speed result a product of the
segmentation?).

The oracle approach window of every lane is the app's stored online_v1 segment, and the app resolves the parameters
of that detector PER INSTRUCTED CONDITION (``resolveParams`` in app/src/core/protocol.ts, mirrored in
``motionlab.conventions``): a slow trial ends only when the phone stays below a stricter stillness threshold for
longer, with a longer movement cap; a fast trial confirms its onset sooner; pause and jerky trials wait longer for the
end. For the hand model this is harmless (no parameter depends on the hand), but for the SPEED target the window, and
so the duration and every feature computed over it, is chosen by the label.

This module re-runs the app's own detector (``motionlab.segment.rerun_online_v1``, the exact port) on every stored
motion stream three times:

* with the trial's stored parameters, which must reproduce the stored bounds (the validation of the port);
* with ONE parameter set for every trial (:func:`common_params`: the app's defaults, with the longest movement cap any
  condition uses, so that no long movement fails on the cap): the label-blind windows;
* with the slow trials' stricter end criterion for every trial (:func:`strict_params`): the evidence for the choice
  of the common set (a common criterion must end every trial before the arm returns; the strict one rarely does).

On the label-blind bounds every trial is then re-standardised through the cache's own code path
(``standardize.stream_from_record`` -> ``standardize.canonical`` -> ``time_normalised`` / ``fixed_window``, the rest
gravity taken around the label-blind bounds), so that EVERY feature of the lane, not only the stream-based ones, can
be computed on label-blind windows. The same path on the oracle bounds must reproduce the cached arrays (checked on
every trial and saved).

Nothing here fits a model. Outputs: one row per trial (keys, conditions, bounds, detector status, the parameters that
differ from the defaults; no app identifier) and the label-blind arrays (:func:`write_npz`, byte-deterministic).
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.lanes import sec_features as F

#: the features of the 60 Hz stream (duration + angular kinematics): computable on any bounds without
#: re-canonicalising the stream (the approach-array features need the canonical gravity frame)
ANG_FEATURES: tuple[str, ...] = ('duration_log', 'ang_speed_peak', 'ang_speed_mean', 'ang_peak_time', 'ang_path',
                                 'ang_sparc', 'ang_ldlj', 'ang_n_peaks')
#: samples of the fixed window kept for the label-blind arrays: the still hold before the onset (the only part of the
#: fixed window any feature of the lane reads)
N_PRE = F.FIXED_ONSET_INDEX


def default_params() -> dict[str, float]:
    from motionlab import conventions as CV
    return {k: float(v) for k, v in CV.DEFAULT_SEGMENTATION_PARAMS.items()}


def common_params() -> dict[str, float]:
    """One parameter set for every trial: the app's default online_v1 parameters, with the longest movement cap of
    any per-condition override (derived from ``motionlab.conventions``, never typed)."""
    from motionlab import conventions as CV
    base = default_params()
    overrides = [*CV._SPEED_OVERRIDES.values(), *CV._VARIANT_OVERRIDES.values()]
    caps = [float(o['max_movement_ms']) for o in overrides if 'max_movement_ms' in o]
    return {**base, 'max_movement_ms': max([base['max_movement_ms'], *caps])}


def strict_params() -> dict[str, float]:
    """The slow trials' resolved parameters (the app's own per-speed override), applied to every trial: the
    alternative common criterion whose failure justifies :func:`common_params`."""
    from motionlab import conventions as CV
    p = CV.resolve_params(CV.DEFAULT_SEGMENTATION_PARAMS, {'speed': 'slow', 'variant': 'standard'})
    return {k: float(v) for k, v in p.items()}


def _bounds(run: Any, t: np.ndarray) -> tuple[float, float]:
    oi, ei = run.result.get('onset_index'), run.result.get('end_index')
    if oi is None or ei is None:
        return (float('nan'), float('nan'))
    return (float(t[int(oi)]), float(t[int(ei)]))


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return x if np.isfinite(x) else float('nan')


def resegment(d: Any, ds: Any | None = None) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Per trial of ``d`` (a Stage2Data): the stored bounds; the bounds the app's detector finds with the stored
    parameters (validation), with :func:`common_params` (label-blind) and with :func:`strict_params`; the parameters
    that differ from the defaults; and the canonical arrays re-standardised on the label-blind bounds.

    Returns (bounds table, arrays {'approach': (N, C, 64), 'pre': (N, C, N_PRE), 'pre_mask': (N, N_PRE), 'trial_key'},
    validation {max |difference| of the arrays re-standardised on the ORACLE bounds from the cached ones}).
    Reads the snapshot (``motionlab.io.load_dataset``) unless ``ds`` is given."""
    from motionlab import conventions as CV
    from motionlab.io import load_dataset
    from motionlab.segment import rerun_online_v1
    from stage2 import standardize as SD
    ds = ds if ds is not None else load_dataset(C.SNAPSHOT_DIR, simulated='include')
    base, common, strict = default_params(), common_params(), strict_params()
    keys = [k for k in CV.SEGMENTATION_PARAM_KEYS if k in base]
    n, nc = d.n, len(SD.CHANNELS)
    A = np.full((n, nc, C.N_NORM), np.nan, dtype=np.float32)
    P = np.full((n, nc, N_PRE), np.nan, dtype=np.float32)
    PM = np.zeros((n, N_PRE), dtype=bool)
    val = {'approach_max_abs_diff': 0.0, 'pre_max_abs_diff': 0.0, 'pre_mask_mismatches': 0, 'n_checked': 0}
    rows = []
    for i, r in enumerate(d.trials.itertuples()):
        rec = ds.record(r.trial_id, compact=True)
        t = np.asarray(rec['streams']['motion']['t'], dtype=float)
        stored = (rec.get('segmentation') or {}).get('params') or {}
        diff = [k for k in keys if k in stored and float(stored[k]) != base[k]]
        s_on, s_end = _bounds(rerun_online_v1(rec, keep_trace=False), t)
        run_c = rerun_online_v1(rec, common, keep_trace=False)
        c_on, c_end = _bounds(run_c, t)
        run_k = rerun_online_v1(rec, strict, keep_trace=False)
        _, k_end = _bounds(run_k, t)
        # the cache's standardisation, first on the oracle bounds (must reproduce the cached arrays), then on the
        # label-blind bounds (the fixed window frozen from the label-blind end, not from the app's end cue)
        st = SD.stream_from_record(rec, ds.trial_normalization(r.trial_id))
        on, en = float(r.onset_ms), float(r.end_ms)
        can_o = SD.canonical(st, on, en)
        cue = SD._num((rec.get('timing') or {}).get('end_detected_ms'))
        fx_o, fm_o, _ = SD.fixed_window(can_o, on, cut_ms=cue)
        val['approach_max_abs_diff'] = max(val['approach_max_abs_diff'], float(np.nanmax(np.abs(
            SD.time_normalised(can_o, on, en) - d.approach[i]))))
        val['pre_max_abs_diff'] = max(val['pre_max_abs_diff'], float(np.nanmax(np.abs(
            fx_o[:, :N_PRE] - d.fixed[i][:, :N_PRE]))))
        val['pre_mask_mismatches'] += int((fm_o[:N_PRE] != d.fixed_mask[i][:N_PRE]).sum())
        val['n_checked'] += 1
        ok_c = str(run_c.status) == 'complete' and np.isfinite(c_end)
        if ok_c:
            can_c = SD.canonical(st, c_on, c_end)
            A[i] = SD.time_normalised(can_c, c_on, c_end)
            fx_c, fm_c, _ = SD.fixed_window(can_c, c_on, cut_ms=c_end)
            P[i], PM[i] = fx_c[:, :N_PRE], fm_c[:N_PRE]
        ret_on = _num(getattr(r, 'return_onset_ms', np.nan))
        rows.append({'trial_key': str(r.trial_key), 'person_id': str(r.person_id), 'speed': str(r.speed),
                     'variant': str(r.variant), 'onset_ms': on, 'end_ms': en,
                     'rerun_onset_ms': s_on, 'rerun_end_ms': s_end, 'common_onset_ms': c_on, 'common_end_ms': c_end,
                     'common_status': str(run_c.status), 'strict_end_ms': k_end, 'strict_status': str(run_k.status),
                     'return_onset_ms': ret_on, 'params_differ': ','.join(diff)})
    out = pd.DataFrame(rows)
    out['reproduced'] = np.isclose(out['rerun_onset_ms'], out['onset_ms']) & np.isclose(out['rerun_end_ms'],
                                                                                         out['end_ms'])
    out['common_ok'] = (out['common_status'] == 'complete') & np.isfinite(out['common_end_ms'])
    out['same_bounds'] = np.isclose(out['common_onset_ms'], out['onset_ms']) & np.isclose(out['common_end_ms'],
                                                                                           out['end_ms'])
    # the strict criterion ends a trial only when it completes before the arm starts back (the return onset; a trial
    # whose return was not observed counts when the run completes)
    out['strict_ok'] = (out['strict_status'] == 'complete') & np.isfinite(out['strict_end_ms']) & ~(
        out['strict_end_ms'] > out['return_onset_ms'])
    out['duration_s'] = (out['end_ms'] - out['onset_ms']) / 1000.0
    out['common_duration_s'] = (out['common_end_ms'] - out['common_onset_ms']) / 1000.0
    arrays = {'approach': A, 'pre': P, 'pre_mask': PM, 'trial_key': out['trial_key'].to_numpy().astype(str)}
    return out, arrays, val


def angular_table(d: Any, onset_ms: np.ndarray, end_ms: np.ndarray) -> pd.DataFrame:
    """The :data:`ANG_FEATURES` of every trial of ``d`` between the given bounds (60 Hz stream; the functions of
    ``sec_features``, so on the stored bounds they equal the EVEN columns of the headline model)."""
    rows = []
    for i in range(d.n):
        on, en = float(onset_ms[i]), float(end_ms[i])
        g = F.approach_gyro(d.stream(i), on, en)
        rows.append({'duration_log': float(np.log((en - on) / 1000.0)),
                     **F.angular_features(F.angular_speed(g))})
    return pd.DataFrame(rows)[list(ANG_FEATURES)].astype(float)


def summary(R: pd.DataFrame) -> dict[str, Any]:
    """Counts and medians per instructed speed (all trials, and the standard variant)."""
    out: dict[str, Any] = {'n_trials': int(len(R)), 'n_reproduced': int(R['reproduced'].sum()),
                           'n_common_ok': int(R['common_ok'].sum()), 'common_params': common_params(),
                           'default_params': default_params(), 'strict_params': strict_params(), 'by_speed': {}}
    for sp, g in R.groupby('speed', sort=True):
        ok = g[g['common_ok']]
        std = g[g['variant'] == 'standard']
        out['by_speed'][str(sp)] = {
            'n': int(len(g)), 'n_params_differ': int((g['params_differ'].fillna('') != '').sum()),
            'params_differ': sorted({k for s in g['params_differ'].fillna('') for k in str(s).split(',') if k}),
            'n_same_bounds': int(g['same_bounds'].sum()),
            'n_standard': int(len(std)), 'n_standard_same_bounds': int(std['same_bounds'].sum()),
            'n_strict_ok': int(g['strict_ok'].sum()), 'n_standard_strict_ok': int(std['strict_ok'].sum()),
            'median_duration_s': float(g['duration_s'].median()),
            'median_common_duration_s': float(ok['common_duration_s'].median()) if len(ok) else float('nan'),
            'median_ratio': float((ok['common_duration_s'] / ok['duration_s']).median()) if len(ok) else float('nan'),
        }
    return out


def write_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """A byte-deterministic .npz (fixed member order and zip timestamps; ``numpy.savez`` stamps the current time),
    written atomically, so the same arrays always give the same file and the same input hash."""
    path = Path(path)
    tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(f'{k}.npy', date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16
            zf.writestr(zi, buf.getvalue())
    os.replace(tmp, path)
    return path


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
