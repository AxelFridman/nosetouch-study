"""Descriptive analysis of the Stage 2 data: core unit 'describe' of ``stage2.pipeline``.

    cd analysis
    python -m stage2.pipeline describe                                   # compute (skipped when current), figures, tables, numbers
    python -m stage2.pipeline figures --lanes describe                   # redraw every S2_01..S2_20 figure (seconds)
    python -m stage2.pipeline figures --lanes describe --only S2_05_sampling_gaps
    python -m stage2.pipeline compute --lanes describe --only raw_quality --force   # one section (or a figure's name)
    python -m stage2.scripts.describe [--force] [--rebuild-cache]       # the same as 'pipeline describe'

Compute is separate from presentation:

* ``compute`` runs the analysis ``SECTIONS`` on the cache and on the raw snapshot (sampling, noise, spectra
  and the example traces need the raw records) and saves everything the presentation reads:
  the descriptive tables ``tables/*.csv``, every summary number in ``tables/describe_summary.json``, and the
  figure inputs ``tables/describe/describe_<name>.json|csv`` (curves, per-person means, per-trial values).
  About 45 s with one thread. It is skipped when those files were made from the current cache; ``--force``
  reruns it, and ``--only`` with section names (or the names of the figures a section feeds) reruns just
  those sections when the saved outputs are current (otherwise every section runs).
* ``figures`` (``stage2/scripts/describe_figures.py``) draws the figures from the saved files only, one
  function per figure with its presentation defaults in ``FIGURES``, and writes each figure's plotted data to
  ``tables/describe/<figure name>.csv``.
* ``tables`` writes the report tables ``tables/describe/describe_<name>.csv`` + ``.tex`` (booktabs).
* ``numbers`` writes ``results/describe_numbers.json`` (LaTeX macros ``\\Describe<Key>``), from saved files.

Persons appear as P01, P02, ... only; report tables carry pseudonymous session labels and trial keys, never a
backend key, app code, timestamp or device detail beyond the model (``write_table`` refuses them). Nothing
data-specific (counts, person ids, results) is typed here: it is computed from the data.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # run as a file
    import _bootstrap  # type: ignore # noqa: F401

import numpy as np
import pandas as pd

from motionlab.io import load_dataset
from motionlab.preprocess import quantize_like_chrome, resample_uniform
from motionlab.report import write_csv

from stage2 import config as C
from stage2 import cuefree, screen
from stage2 import pipeline as PL
from stage2 import report_export as RX
from stage2.cv import roc_auc
from stage2.data import (
    FLAG_COLUMNS, GATE_COLUMNS, QUASI_IDENTIFIER_COLUMNS, build_cache, load_cache, load_tables, report_sessions,
)
from stage2.qc import cumulative_power, fft_length, share_below
from stage2.standardize import canonical, stream_from_record

ORDER = 10
SUMMARY_JSON = C.TAB_DIR / 'describe_summary.json'      # every summary number (README, numbers, captions)
DATA_DIR = C.TAB_DIR / 'describe'                       # figure inputs, figure data and report tables of this unit
NUMBERS_JSON = C.RES_DIR / 'describe_numbers.json'

# analysis settings (one place; changing one changes the outputs)
PLATFORMS = ('android', 'ios')
GAP_LIST_MS = 50.0                  # gaps.csv lists every sample interval longer than this
NEAR_PAD_MS = 500.0                 # a gap is 'near the movement' within [onset - 0.5 s, end + 0.5 s]
CUE_LAG_MS = (-20.0, 400.0)         # a gap 'follows a cue' when it starts within this lag after an audio cue
SAMPLING_INTERVAL_TRIALS = 120      # trials per platform whose raw sample intervals S2_05 shows (raw reads are slow)
SPECTRUM_PAD_MS = 250.0             # the spectrum window is [onset - 0.25 s, end + 0.25 s]
CLOSURE_REPORT_DEG = 3.0            # gyro-gravity closures above this are counted
STRESS_KEEP_S = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)   # stillness kept before the approach (stress test)
STRESS_RANDOM_MAX_S = 1.5           # the 'random' setting keeps U(0, this) seconds
TRACE_CHANNELS = ('gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z', 'yaw_rate', 'gyro_horiz')
CLASS_MEAN_CHANNELS = ('yaw_rate', 'yaw_cum', 'gyro_horiz', 'acc_vert', 'acc_horiz', 'tilt_from_start')
CHIRAL = ('yaw_rate', 'yaw_cum')    # mirror-odd channels among those (standardize.MIRROR_SIGN)
#: plain labels of the flag and gate columns (figure S2_18 and the flags table)
FLAG_LABELS = {'grip_mismatch': 'T8 grip not as instructed', 'seg_disagree_100ms': 'T9 onsets differ > 100 ms',
               'clip_candidate': 'T10 possible clipping', 'repeated_samples': 'T11 repeated samples',
               'false_start': 'T12 false start before GO', 'screen_rotated': 'screen rotated in the movement',
               'gate_g1': 'fails gate G1 (gap > 50 ms near it)', 'gate_g2': 'fails gate G2 (longer than 8 s)',
               'gate_pass': 'fails G1 or G2'}

#: backend keys -> pseudonymous labels for report tables (filled from the cached tables)
TRIAL_KEY: dict[str, str] = {}
SESSION_LABEL: dict[str, str] = {}


def write_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Formula-safe CSV (text from records posted to a public endpoint: motionlab.report.write_csv).

    Report tables carry no backend key: trial_id becomes the pseudonymous trial_key (P14-s3-t07a1) and
    session_id the session_label (P14-s3); any other quasi-identifier column is refused."""
    df = df.copy()
    if 'trial_id' in df.columns:
        df.insert(int(df.columns.get_loc('trial_id')), 'trial_key', df['trial_id'].map(TRIAL_KEY))
        df = df.drop(columns='trial_id')
    if 'session_id' in df.columns:
        df.insert(int(df.columns.get_loc('session_id')), 'session_label', df['session_id'].map(SESSION_LABEL))
        df = df.drop(columns='session_id')
    bad = [c for c in df.columns if c in QUASI_IDENTIFIER_COLUMNS]
    if bad:
        raise ValueError(f'table {name}: quasi-identifier columns {bad} do not belong in a report table')
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(df, C.TAB_DIR / f'{name}.csv', index=index)


def q(x: Any, p: float) -> float:
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    return float(np.quantile(x, p)) if len(x) else math.nan


# ───────────────────────────── saved figure inputs ─────────────────────────────


def _jsonable(v: Any) -> Any:
    """numpy arrays and scalars -> lists and Python numbers; NaN / inf -> None (strict JSON)."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if v is None or isinstance(v, (int, str)):
        return v
    if pd.isna(v):
        return None
    return str(v)


def input_path(name: str, ext: str = 'json') -> Path:
    return DATA_DIR / f'describe_{name}.{ext}'


def save_input(name: str, obj: dict[str, Any]) -> Path:
    """tables/describe/describe_<name>.json: a figure input, with the cache provenance (NaN -> null)."""
    p = input_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'provenance': PL.provenance(), **_jsonable(obj)}, indent=1, allow_nan=False) + '\n',
                 encoding='utf-8', newline='\n')
    return p


def load_input(name: str) -> dict[str, Any]:
    p = input_path(name)
    if not p.exists():
        raise FileNotFoundError(f'{p}: run python -m stage2.pipeline compute --lanes describe')
    return json.loads(p.read_text(encoding='utf-8'))


def save_input_csv(df: pd.DataFrame, name: str) -> Path:
    """tables/describe/describe_<name>.csv: a per-trial figure input (report-safe: trial keys, no backend id)."""
    return RX.export_table(df, DATA_DIR / f'describe_{name}', tex=False)['csv']


def load_input_csv(name: str) -> pd.DataFrame:
    p = input_path(name, 'csv')
    if not p.exists():
        raise FileNotFoundError(f'{p}: run python -m stage2.pipeline compute --lanes describe')
    return pd.read_csv(p)


def load_table(name: str) -> pd.DataFrame:
    """A descriptive table written by compute (tables/<name>.csv)."""
    p = C.TAB_DIR / f'{name}.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p}: run python -m stage2.pipeline compute --lanes describe')
    return pd.read_csv(p)


def load_summary() -> dict[str, Any]:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f'{SUMMARY_JSON}: run python -m stage2.pipeline compute --lanes describe')
    return json.loads(SUMMARY_JSON.read_text(encoding='utf-8'))


#: every figure input compute writes (the currency check of the compute step)
INPUT_FILES = ('sets.json', 'primary_trials.csv', 'sampling_intervals.json', 'spectrum.json', 'complete_attempts.csv',
               'example_traces.json', 'class_means.json', 'sign_patterns.json', 'platform_means.json',
               'return_means.json', 'stillness_before_onset.csv', 'stress_settings.json')


# ───────────────────────────── the inputs of one compute run ─────────────────────────────


class Ctx:
    """What the sections read, loaded once (the raw snapshot lazily: only some sections need it)."""

    def __init__(self, summary: dict[str, Any] | None = None) -> None:
        tabs = load_tables()
        self.T, self.P, self.S = tabs['trials'], tabs['persons'], tabs['sessions']
        self.flow, self.e1 = tabs['exclusion_flow'], tabs['exclusion_e1']
        TRIAL_KEY.update(zip(self.T['trial_id'], self.T['trial_key']))
        SESSION_LABEL.update(zip(self.S['session_id'], self.S['session_label']))
        self.d = load_cache(('primary',))
        self.summary: dict[str, Any] = {} if summary is None else summary
        self.rawq: dict[str, pd.DataFrame] | None = None
        self._ds: Any = None

    @property
    def ds(self) -> Any:
        if self._ds is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._ds = load_dataset(C.SNAPSHOT_DIR, simulated='include')
        return self._ds

    def per_trial_quality(self) -> pd.DataFrame:
        """The raw per-trial quality of this run (section raw_quality) or, when it did not run, its saved table."""
        return self.rawq['per_trial'] if self.rawq is not None else load_table('sampling_per_trial')


# ───────────────────────────── 1. participants, demographics ─────────────────────────────


def participants(ctx: Ctx) -> None:
    T, P, S = ctx.T, ctx.P, ctx.S
    write_table(report_sessions(S), 'sessions')
    cols = ['person_id', 'platform', 'device_models', 'age_years', 'sex', 'handedness_self', 'phone_hand', 'height_cm',
            'n_sessions', 'session_statuses', 'protocols', 'n_attempts', 'n_accepted', 'n_primary', 'n_primary_left',
            'n_primary_right', 'n_natural_use', 'max_tier_accepted', 'bonus_attempted', 'profile_consistent']
    write_table(P[cols], 'participants')
    age = P['age_years'].astype(float)
    hgt = P['height_cm'].astype(float)
    prim = T[T['analysis_set'] == 'primary']
    demo = {
        'n_persons': len(P), 'age_mean': age.mean(), 'age_sd': age.std(ddof=1), 'age_median': age.median(),
        'age_min': age.min(), 'age_max': age.max(),
        'n_female': int((P['sex'] == 'female').sum()), 'n_male': int((P['sex'] == 'male').sum()),
        'n_right_handed': int((P['handedness_self'] == 'right').sum()),
        'n_left_handed': int((P['handedness_self'] == 'left').sum()),
        'n_phone_hand_right': int((P['phone_hand'] == 'right').sum()),
        'n_phone_hand_both': int((P['phone_hand'] == 'both').sum()),
        'n_phone_hand_left': int((P['phone_hand'] == 'left').sum()),
        'height_n': int(hgt.notna().sum()), 'height_mean': hgt.mean(), 'height_sd': hgt.std(ddof=1),
        'height_min': hgt.min(), 'height_max': hgt.max(),
        'persons_android': int((P['platform'] == 'android').sum()), 'persons_ios': int((P['platform'] == 'ios').sum()),
        'sessions_included': int(S['include'].sum()), 'sessions_android': int((S['include'] & (S['platform'] == 'android')).sum()),
        'sessions_ios': int((S['include'] & (S['platform'] == 'ios')).sum()),
        'primary_trials': len(prim), 'primary_android': int((prim['platform'] == 'android').sum()),
        'primary_ios': int((prim['platform'] == 'ios').sum()),
        'primary_left': int((prim['hand'] == 'left').sum()), 'primary_right': int((prim['hand'] == 'right').sum()),
        'n_device_models': int(S.loc[S['include'], 'device_model'].nunique()),
        'n_persons_multi_session': int((P['n_sessions'] > 1).sum()),
    }
    write_table(pd.DataFrame([demo]).T.reset_index().rename(columns={'index': 'quantity', 0: 'value'}), 'demographics_summary')
    ctx.summary['demographics'] = demo
    inc = S[S['include']]
    norm = inc[['person_id', 'session_id', 'platform', 'engine', 'calibration_method', 'gravity_sign', 'gravity_sign_source',
                'rr_mapping', 'rr_units', 'rr_source', 'rr_closure_deg', 'rr_closure_runner_up', 'rr_closure_runner_up_deg',
                'rr_closure_trials']]
    write_table(norm, 'normalisation_per_session')
    ctx.summary['normalisation'] = {
        'gravity_sign_source': inc['gravity_sign_source'].value_counts().to_dict(),
        'gravity_sign_by_platform': inc.groupby('platform')['gravity_sign'].agg(lambda s: sorted(set(s.astype(int)))).to_dict(),
        'rr_mapping': inc['rr_mapping'].value_counts().to_dict(), 'rr_source': inc['rr_source'].value_counts().to_dict(),
        'rr_closure_median_deg': float(inc['rr_closure_deg'].median()),
        'rr_runner_up_median_deg': float(inc['rr_closure_runner_up_deg'].median()),
    }


# ───────────────────────────── 2. exclusion flow ─────────────────────────────


def exclusion_flow(ctx: Ctx) -> None:
    flow, e1, T = ctx.flow, ctx.e1, ctx.T
    write_table(flow, 'exclusion_flow')
    write_table(e1, 'exclusion_E1_breakdown')
    by = T[T['excl_rule'].notna()].groupby(['excl_rule', 'platform']).size().unstack(fill_value=0)
    write_table(by.reset_index(), 'exclusion_by_platform')
    ctx.summary['exclusion_flow'] = flow[['step', 'removed', 'remaining', 'persons', 'sessions', 'natural_use']].to_dict(orient='records')
    ctx.summary['exclusion_E1'] = e1.to_dict(orient='records')
    sets = {}
    for name in ('primary', 'natural_use', 'natural_use_repeat'):
        g = T[T['analysis_set'] == name]
        sets[name] = {'n': len(g), 'persons': int(g['person_id'].nunique()),
                      'left': int((g['hand'] == 'left').sum()), 'right': int((g['hand'] == 'right').sum())}
    save_input('sets', {'sets': sets})


# ───────────────────────────── 3. completion and visit durations ─────────────────────────────


def completion(ctx: Ctx) -> None:
    T, S = ctx.T, ctx.S
    inc = S[S['include']].copy()
    tr = T[T['session_included']].copy()
    acc = tr[tr['accepted'].fillna(False).astype(bool)]
    rows = []
    for _, s in inc.iterrows():
        a = acc[acc['session_id'] == s['session_id']]
        t = tr[tr['session_id'] == s['session_id']]
        r = {'person_id': s['person_id'], 'session_id': s['session_id'], 'session_order': int(s['session_order']),
             'status': s['status'],
             'protocol': s['protocol_version'], 'platform': s['platform'], 'attempts': len(t), 'accepted': len(a),
             'max_tier': int(a['tier'].max()) if a['tier'].notna().any() else None,
             'bonus_attempted': bool((t['tier'] == 4).any())}
        for k in range(5):
            r[f'accepted_tier{k}'] = int((a['tier'] == k).sum())
        rows.append(r)
    comp = pd.DataFrame(rows)
    write_table(comp, 'completion')
    v2 = comp[comp['protocol'] == '2.0.0']
    ctx.summary['completion'] = {
        'sessions_v2': len(v2), 'completed': int((v2['status'] == 'completed').sum()),
        'ended_early': int((v2['status'] == 'ended_early').sum()),
        'bonus_uptake_sessions': int(v2['bonus_attempted'].sum()),
        'bonus_uptake_completed': f"{int(v2.loc[v2['status'] == 'completed', 'bonus_attempted'].sum())}/"
                                  f"{int((v2['status'] == 'completed').sum())}",
        'max_tier_counts': {str(int(k)): int(v) for k, v in v2['max_tier'].value_counts().sort_index().items()},
        'attempts_per_accepted': float(v2['attempts'].sum() / v2['accepted'].sum()),
    }
    # visit durations from the phone clock (t0_epoch + stop)
    t = tr.copy()
    t['start_ms'] = t['t0_epoch_ms'].astype(float)
    t['end_ms_epoch'] = t['start_ms'] + t['stop_ms'].astype(float)
    drow = []
    for sid, g in t.groupby('session_id'):
        s = inc[inc['session_id'] == sid].iloc[0]
        started = pd.Timestamp(s['started_at']).value / 1e6
        r = {'person_id': s['person_id'], 'session_id': sid, 'status': s['status'], 'protocol': s['protocol_version'],
             'setup_s': (g['start_ms'].min() - started) / 1000.0,
             'trials_s': (g['end_ms_epoch'].max() - g['start_ms'].min()) / 1000.0,
             'total_s': (g['end_ms_epoch'].max() - started) / 1000.0, 'attempts': len(g)}
        for k in range(5):
            gk = g[g['tier'] == k]
            r[f'tier{k}_s'] = (gk['end_ms_epoch'].max() - gk['start_ms'].min()) / 1000.0 if len(gk) else math.nan
        drow.append(r)
    dur = pd.DataFrame(drow)
    write_table(dur, 'visit_durations')
    done = dur[(dur['protocol'] == '2.0.0') & (dur['status'] == 'completed')]
    ctx.summary['visit_durations_completed_v2_min'] = {
        'n_sessions': len(done), 'total_median': float(done['total_s'].median() / 60),
        'total_iqr': [float(done['total_s'].quantile(.25) / 60), float(done['total_s'].quantile(.75) / 60)],
        'setup_median': float(done['setup_s'].median() / 60),
        **{f'tier{k}_median': float(done[f'tier{k}_s'].median() / 60) for k in range(5)},
    }


# ───────────────────────────── 4. trials per condition ─────────────────────────────


def conditions(ctx: Ctx) -> None:
    T = ctx.T
    prim = T[T['analysis_set'] == 'primary'].copy()
    prim['grip'] = prim['orientation'].str.replace('_toward', '', regex=False)
    long = prim.groupby(['grip', 'speed', 'variant', 'start_position', 'hand']).size().rename('trials').reset_index()
    write_table(long, 'trials_per_condition_long')
    wide = long.pivot_table(index=['grip', 'speed', 'variant', 'start_position'], columns='hand', values='trials',
                            fill_value=0).reset_index()
    wide['total'] = wide[['left', 'right']].sum(axis=1)
    wide = wide.sort_values('total', ascending=False)
    write_table(wide, 'trials_per_condition')
    ctx.summary['conditions'] = {'n_conditions': len(wide), 'top': wide.head(6).to_dict(orient='records')}


# ───────────────────────────── 5. per-trial values of the primary set (figure input) ─────────────────────────────


def primary_trials(ctx: Ctx) -> None:
    """The per-trial descriptive values the figures plot (first-half and net yaw, closure, duration)."""
    t = ctx.d.trials
    df = t[['trial_id', 'person_id', 'platform', 'hand', 'first_half_yaw_deg', 'net_yaw_deg', 'closure_deg',
            'duration_ms', 'peak_gyro_dps']].copy()
    df['ret_ok'] = np.asarray(ctx.d.ret_ok, dtype=bool)
    save_input_csv(df, 'primary_trials')


# ───────────────────────────── 6. sampling, gaps, noise, spectrum (raw) ─────────────────────────────


def raw_quality(ctx: Ctx) -> None:
    """Sampling intervals, gaps by phase, rest noise and quantisation, approach spectra (raw records)."""
    ds, T = ctx.ds, ctx.T
    cand = T[T['session_included'] & (T['status'] == 'complete') & T['oracle_ok'].fillna(False).astype(bool)]
    gaps, per, wins = [], [], []
    for _, r in cand.iterrows():
        s = ds.streams(r['trial_id'])
        t = s.t
        dt = np.diff(t)
        on, en = float(r['onset_ms']), float(r['end_ms'])
        cue_t = s.cues['t'].astype(float).to_numpy() if len(s.cues) else np.zeros(0)
        for i in np.flatnonzero(dt > GAP_LIST_MS):
            mid = t[i]
            phase = ('pre-onset (hold, settling)' if t[i + 1] <= on - NEAR_PAD_MS else
                     'near the movement' if mid < en + NEAR_PAD_MS else 'after the movement')
            lag = mid - cue_t
            lag = lag[(lag >= CUE_LAG_MS[0]) & (lag <= CUE_LAG_MS[1])]
            gaps.append({'trial_id': r['trial_id'], 'person_id': r['person_id'], 'platform': r['platform'],
                         'gap_ms': float(dt[i]), 'gap_start_ms': float(mid), 'gap_start_rel_onset_ms': float(mid - on),
                         'phase': phase, 'within_400ms_after_a_cue': bool(lag.size), 'analysis_set': r['analysis_set']})
        # rest noise in the still window before GO (raw values, platform sign irrelevant for SD)
        go = r['go_ms']
        st = r['still_ms'] if pd.notna(r['still_ms']) else go - 400
        w = (t >= float(st) - 400) & (t <= float(go))
        accg_sd = float(np.nanstd(s.accg[w], axis=0).mean()) if w.sum() >= 10 else math.nan
        gyro_sd = float(np.nanstd(s.rr[w], axis=0).mean()) if w.sum() >= 10 else math.nan
        on_grid = float(np.mean(np.abs(np.round(s.accg[:, 1] * 10) - s.accg[:, 1] * 10) < 1e-6))
        per.append({'trial_id': r['trial_id'], 'person_id': r['person_id'], 'platform': r['platform'],
                    'analysis_set': r['analysis_set'], 'n': int(t.size), 'hz': float((t.size - 1) / (t[-1] - t[0]) * 1000),
                    'dt_median': float(np.median(dt)), 'dt_p95': float(np.percentile(dt, 95)),
                    'dt_p99': float(np.percentile(dt, 99)), 'dt_max': float(dt.max()),
                    'n_gaps_50': int(np.sum(dt > 50)), 'n_gaps_100': int(np.sum(dt > 100)),
                    'rest_accg_sd': accg_sd, 'rest_gyro_sd': gyro_sd, 'share_on_0p1_grid': on_grid,
                    'rest_gyro_sd_if_quantized': float(np.nanstd(quantize_like_chrome(s.rr[w], 'rr'), axis=0).mean()) if w.sum() >= 10 else math.nan,
                    'rest_accg_sd_if_quantized': float(np.nanstd(quantize_like_chrome(s.accg[w], 'acc'), axis=0).mean()) if w.sum() >= 10 else math.nan})
        # spectrum of the movement: raw gyro and accg resampled to 60 Hz WITHOUT low-pass
        if r['analysis_set'] == 'primary':
            tu, U = resample_uniform(t, np.column_stack([s.rr, s.accg]), C.FS_HZ, C.MAX_GAP_MS,
                                     t_start=on - SPECTRUM_PAD_MS, t_end=en + SPECTRUM_PAD_MS)
            if tu.size >= 16 and np.isfinite(U).all():
                wins.append((r['platform'], U))
    # one FFT length for every window, long enough for the longest (no window is cropped); each window is
    # Hann-tapered over its full length and zero-padded, so all share one frequency grid
    n_fft = fft_length(max(U.shape[0] for _, U in wins))
    spec = []
    freqs = None
    for pl, U in wins:
        freqs, cg = cumulative_power(U[:, 0:3], C.FS_HZ, n_fft)
        _, ca = cumulative_power(U[:, 3:6], C.FS_HZ, n_fft)
        spec.append({'platform': pl, 'gyro': cg, 'acc': ca, 'n': U.shape[0]})
    G = pd.DataFrame(gaps)
    Q = pd.DataFrame(per)
    write_table(G, 'gaps')
    rows = []
    for pl, g in Q.groupby('platform'):
        gg = G[G['platform'] == pl]
        rows.append({'platform': pl, 'trials': len(g), 'hz_median': g['hz'].median(), 'hz_min': g['hz'].min(),
                     'dt_median_ms': g['dt_median'].median(), 'dt_p95_ms': g['dt_p95'].median(), 'dt_p99_ms': g['dt_p99'].median(),
                     'trials_with_gap_50': int((g['n_gaps_50'] > 0).sum()), 'trials_with_gap_100': int((g['n_gaps_100'] > 0).sum()),
                     'gaps_50': len(gg), 'gap_ms_median': gg['gap_ms'].median() if len(gg) else math.nan,
                     'gaps_near_movement': int((gg['phase'] == 'near the movement').sum()),
                     'gaps_near_movement_over_100': int(((gg['phase'] == 'near the movement') & (gg['gap_ms'] > 100)).sum()),
                     'share_values_on_0p1_grid': g['share_on_0p1_grid'].median(),
                     'rest_accg_sd_median': g['rest_accg_sd'].median(), 'rest_gyro_sd_median': g['rest_gyro_sd'].median()})
    samp = pd.DataFrame(rows)
    write_table(samp, 'sampling_quality')
    write_table(Q, 'sampling_per_trial')
    ctx.rawq = {'gaps': G, 'per_trial': Q}
    S = ctx.summary
    S['sampling'] = samp.to_dict(orient='records')
    S['quantisation_noise_sd'] = 0.1 / math.sqrt(12)
    big = G[(G['platform'] == 'ios') & (G['gap_ms'] > 200)]
    S['ios_long_gaps'] = {
        'n': len(big), 'trials': int(big['trial_id'].nunique()),
        'start_after_recording_start_median_s': float(big['gap_start_ms'].median() / 1000) if len(big) else math.nan,
        'start_iqr_s': [q(big['gap_start_ms'], .25) / 1000, q(big['gap_start_ms'], .75) / 1000],
        'share_within_400ms_after_a_cue': float(big['within_400ms_after_a_cue'].mean()) if len(big) else math.nan,
        'duration_median_ms': float(big['gap_ms'].median()) if len(big) else math.nan,
    }
    # S2_05: the raw sample intervals of the first trials of each platform, stored as exact value counts
    iv = {}
    for pl in PLATFORMS:
        tids = Q.loc[Q['platform'] == pl, 'trial_id'].head(SAMPLING_INTERVAL_TRIALS)
        if not len(tids):
            continue
        dts = np.concatenate([np.diff(ds.streams(tid).t) for tid in tids])
        vals, counts = np.unique(dts, return_counts=True)
        iv[pl] = {'n_trials': int(len(tids)), 'n_intervals': int(dts.size), 'median_ms': float(np.median(dts)),
                  'value_ms': vals, 'count': counts}
    save_input('sampling_intervals', {'gap_list_ms': GAP_LIST_MS, 'near_pad_ms': NEAR_PAD_MS,
                                      'cue_lag_ms': list(CUE_LAG_MS), 'platforms': iv})
    # S2_07: median cumulative power spectra (the summary shares come from the medians over every trial)
    allg = np.median(np.stack([x['gyro'] for x in spec]), axis=0)
    alla = np.median(np.stack([x['acc'] for x in spec]), axis=0)
    g15, a15 = share_below(freqs, allg, C.LOWPASS_HZ), share_below(freqs, alla, C.LOWPASS_HZ)
    S['spectrum'] = {'gyro_power_below_15hz': g15, 'acc_power_below_15hz': a15,
                     'gyro_power_below_5hz': share_below(freqs, allg, 5.0),
                     'acc_power_below_5hz': share_below(freqs, alla, 5.0),
                     'n_trials': len(spec), 'n_fft': int(n_fft),
                     'longest_window_samples': int(max(x['n'] for x in spec)),
                     'windows_over_256_samples': int(sum(x['n'] > 256 for x in spec)),
                     'method': 'per trial: detrend, Hann over the full window, zero-padded FFT of n_fft '
                               'points (no cropping); median over trials of the cumulative power share'}
    curves = {}
    for pl in PLATFORMS:
        sp = [x for x in spec if x['platform'] == pl]
        if sp:
            curves[pl] = {'n_trials': len(sp), 'gyro': np.median(np.stack([x['gyro'] for x in sp]), axis=0),
                          'acc': np.median(np.stack([x['acc'] for x in sp]), axis=0)}
    save_input('spectrum', {'freqs_hz': freqs, 'n_fft': int(n_fft), 'fs_hz': C.FS_HZ, 'lowpass_hz': C.LOWPASS_HZ,
                            'pad_ms': SPECTRUM_PAD_MS, 'platforms': curves,
                            'all': {'n_trials': len(spec), 'gyro': allg, 'acc': alla}})


# ───────────────────────────── 7. durations, segmentation, contact ─────────────────────────────


def durations_segmentation(ctx: Ctx) -> None:
    T = ctx.T
    cand = T[T['session_included'] & (T['status'] == 'complete') & T['oracle_ok'].fillna(False).astype(bool)].copy()
    prim = T[T['analysis_set'] == 'primary'].copy()
    rows = []
    for (sp, pl), g in prim.groupby(['speed', 'platform']):
        rows.append({'speed': sp, 'platform': pl, 'n': len(g), 'duration_median_s': g['duration_ms'].median() / 1000,
                     'q25_s': g['duration_ms'].quantile(.25) / 1000, 'q75_s': g['duration_ms'].quantile(.75) / 1000,
                     'peak_gyro_median_dps': g['peak_gyro_dps'].median(), 'peak_acc_median_ms2': g['peak_acc_ms2'].median()})
    dur = pd.DataFrame(rows)
    write_table(dur, 'durations_by_speed_platform')
    write_table(prim.groupby('variant')['duration_ms'].describe().reset_index(), 'durations_by_variant')
    ctx.summary['durations'] = {
        'primary_median_s': float(prim['duration_ms'].median() / 1000),
        'by_speed_median_s': (prim.groupby('speed')['duration_ms'].median() / 1000).to_dict(),
        'share_le_2p5s': float((prim['duration_ms'] <= 2500).mean()),
        'peak_gyro_median_dps': float(prim['peak_gyro_dps'].median()),
        'reaction_median_ms': float((prim['onset_ms'] - prim['go_ms']).median()),
    }
    # segmentation agreement and contact
    c = cand.copy()
    c['d_onset'] = c['off_onset_ms'] - c['onset_ms']
    c['d_end'] = c['off_end_ms'] - c['end_ms']
    c['touch_frac'] = (c['first_touch_ms'] - c['onset_ms']) / (c['end_ms'] - c['onset_ms'])
    toward = c['orientation'].astype(str).str.endswith('toward')
    seg = {
        'complete_attempts': len(c), 'offline_end_found': int(c['off_ok'].fillna(False).sum()),
        'offline_no_end_share': float(1 - c['off_ok'].fillna(False).mean()),
        'offline_no_end_share_v2': float(1 - c.loc[c['protocol'] == 'v2_street', 'off_ok'].fillna(False).mean()),
        'offline_no_end_share_v1': float(1 - c.loc[c['protocol'] == 'v1_quick', 'off_ok'].fillna(False).mean()),
        'onset_diff_median_ms': float(c['d_onset'].median()), 'onset_diff_iqr_ms': [q(c['d_onset'], .25), q(c['d_onset'], .75)],
        'end_diff_median_ms': float(c['d_end'].median()), 'end_diff_iqr_ms': [q(c['d_end'], .25), q(c['d_end'], .75)],
        'share_onset_within_100ms': float((c['d_onset'].abs() <= 100).mean()),
        'share_end_within_250ms': float((c['d_end'].abs() <= 250).sum() / c['d_end'].notna().sum()),
        'toward_trials_with_touch': float(c.loc[toward, 'first_touch_ms'].notna().mean()),
        'touch_position_median': float(c.loc[toward, 'touch_frac'].median()),
        'touch_position_iqr': [q(c.loc[toward, 'touch_frac'], .25), q(c.loc[toward, 'touch_frac'], .75)],
        'touch_inside_window_share': float(c.loc[toward & c['first_touch_ms'].notna(), 'touch_frac'].between(0, 1).mean()),
        'touch_before_end_ms_median': float((c.loc[toward, 'end_ms'] - c.loc[toward, 'first_touch_ms']).median()),
    }
    ctx.summary['segmentation'] = seg
    write_table(pd.DataFrame([seg]).T.reset_index().rename(columns={'index': 'quantity', 0: 'value'}), 'segmentation_agreement')
    att = c[['trial_id', 'analysis_set', 'platform', 'protocol', 'speed', 'orientation', 'duration_ms', 'peak_gyro_dps',
             'd_onset', 'd_end', 'touch_frac']].rename(columns={'d_onset': 'd_onset_ms', 'd_end': 'd_end_ms'})
    att['toward'] = toward.to_numpy()
    save_input_csv(att, 'complete_attempts')          # S2_08, S2_09: every complete attempt with an oracle window


# ───────────────────────────── 8. example traces ─────────────────────────────


def example_traces(ctx: Ctx) -> None:
    ds, T = ctx.ds, ctx.T
    prim = T[(T['analysis_set'] == 'primary') & (T['orientation'] == 'upright_toward') & (T['speed'] == 'natural')
             & (T['variant'] == 'standard') & T['first_touch_ms'].notna() & T['return_onset_ms'].notna()]
    # a TYPICAL pair (not the best one): per hand, the trial whose first-half yaw is closest to that hand's
    # median over all primary trials, durations near the median; the person with the most typical pair wins
    allp = T[T['analysis_set'] == 'primary']
    med = {h: float(allp.loc[allp['hand'] == h, 'first_half_yaw_deg'].median()) for h in ('left', 'right')}
    med_d = float(allp['duration_ms'].median())
    best = None
    for pid, g in prim.groupby('person_id'):
        if set(g['hand']) != {'left', 'right'}:
            continue
        pick = {}
        for h in ('left', 'right'):
            gh = g[g['hand'] == h]
            cost = (gh['first_half_yaw_deg'] - med[h]).abs() + (gh['duration_ms'] - med_d).abs() / 500.0
            pick[h] = gh.loc[cost.idxmin()]
            pick[h + '_cost'] = float(cost.min())
        score = pick['left_cost'] + pick['right_cost']
        if best is None or score < best[0]:
            best = (score, pick['left'], pick['right'])
    if best is None:
        save_input('example_traces', {'trials': []})
        return
    _, L, R = best
    trials = []
    for r in (L, R):
        rec = ds.record(r['trial_id'], compact=True)
        st = stream_from_record(rec, ds.trial_normalization(r['trial_id']))
        can = canonical(st, float(r['onset_ms']), float(r['end_ms']))
        go = float(r['go_ms'])
        ev = {'GO': r['go_ms'], 'onset': r['onset_ms'], 'contact': r['first_touch_ms'], 'end': r['end_ms'],
              'end cue': r['end_detected_ms'], 'return': r['return_onset_ms'], 'still again': r['return_end_ms']}
        trials.append({'hand': r['hand'], 'person_id': r['person_id'], 'trial_key': r['trial_key'],
                       'orientation': r['orientation'], 'speed': r['speed'], 'variant': r['variant'],
                       'first_half_yaw_deg': float(r['first_half_yaw_deg']),
                       'events_s': {k: ((float(v) - go) / 1000 if pd.notna(v) else None) for k, v in ev.items()},
                       't_s': (st.t - go) / 1000, 'channels': {ch: can.channel(ch) for ch in TRACE_CHANNELS}})
    save_input('example_traces', {'person_id': L['person_id'], 'trials': trials})
    ctx.summary['example_traces'] = {'person': L['person_id'], 'left_trial': L['trial_key'], 'right_trial': R['trial_key']}


# ───────────────────────────── 9. class means, mirror symmetry, first-half yaw ─────────────────────────────


def person_means(X: np.ndarray, persons: np.ndarray, hands: np.ndarray, hand: str) -> tuple[np.ndarray, list[str]]:
    out, ids = [], []
    for p in sorted(set(persons.tolist())):
        m = (persons == p) & (hands == hand)
        if m.any():
            out.append(np.nanmean(X[m], axis=0))
            ids.append(p)
    return np.stack(out), ids


def class_means(ctx: Ctx) -> None:
    d = ctx.d
    hands = d.trials['hand'].to_numpy()
    persons = d.groups
    x = np.linspace(0, 100, d.approach.shape[2])
    rows = []
    saved: dict[str, Any] = {}
    for ch in CLASS_MEAN_CHANNELS:
        k = d.channels.index(ch)
        X = d.approach[:, k, :]
        stats = {h: person_means(X, persons, hands, h) for h in ('left', 'right')}
        sign = -1.0 if ch in CHIRAL else 1.0
        common = sorted(set(stats['left'][1]) & set(stats['right'][1]))
        Lm = stats['left'][0][[stats['left'][1].index(p) for p in common]]
        Rm = stats['right'][0][[stats['right'][1].index(p) for p in common]]
        rms_lr = float(np.sqrt(np.mean((Lm.mean(0) - Rm.mean(0)) ** 2)))
        rms_mir = float(np.sqrt(np.mean((sign * Lm.mean(0) - Rm.mean(0)) ** 2)))
        scale = float(np.sqrt(np.mean(Rm.mean(0) ** 2)))
        # per-person: is the right-hand mean closer to the mirrored left mean than to the left mean?
        closer = int(np.sum(np.sqrt(np.mean((sign * Lm - Rm) ** 2, axis=1)) < np.sqrt(np.mean((Lm - Rm) ** 2, axis=1))))
        rows.append({'channel': ch, 'mirror_sign': int(sign), 'rms_class_mean': scale, 'rms_left_vs_right': rms_lr,
                     'rms_mirrored_left_vs_right': rms_mir, 'persons': len(common),
                     'persons_closer_to_mirror': closer if sign < 0 else None})
        saved[ch] = {'mirror_sign': int(sign), 'hands': {h: {'persons': ids, 'means': M} for h, (M, ids) in stats.items()}}
    mt = pd.DataFrame(rows)
    write_table(mt, 'mirror_symmetry')
    ctx.summary['mirror_symmetry'] = mt.to_dict(orient='records')
    save_input('class_means', {'x_pct': x, 'channels': saved})


SIGN_CHANNELS = ('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')
SIGMA_STAR = (-1, 1, 1, 1, -1, -1)           # the mirror predicted by the conventions (standardize.MIRROR_SIGN)


def sign_patterns(ctx: Ctx) -> None:
    """docs/05 section 3.5 item 4: which of the 64 sign patterns maps right-hand trials onto left-hand ones."""
    d = ctx.d
    idx = [d.channels.index(c) for c in SIGN_CHANNELS]
    X = d.approach[:, idx, :].astype(float)
    tr = d.trials
    scale = np.sqrt(np.nanmean(X ** 2, axis=(0, 2)))
    Xs = X / scale[None, :, None]
    mirror_o = {'upright_toward': 'upright_toward', 'top_left_toward': 'top_right_toward',
                'top_right_toward': 'top_left_toward'}
    star = np.array(SIGMA_STAR, dtype=float)
    pats = np.array([[1 - 2 * ((m >> j) & 1) for j in range(6)] for m in range(64)], dtype=float)
    rows = []
    for pl in (*PLATFORMS, 'all'):
        num = np.zeros(64)
        den = 0.0
        sel = np.ones(len(tr), bool) if pl == 'all' else (tr['platform'] == pl).to_numpy()
        for pid in sorted(set(tr.loc[sel, 'person_id'])):
            for o, mo in mirror_o.items():
                mL = sel & (tr['person_id'] == pid).to_numpy() & (tr['hand'] == 'left').to_numpy() & (tr['orientation'] == o).to_numpy()
                mR = sel & (tr['person_id'] == pid).to_numpy() & (tr['hand'] == 'right').to_numpy() & (tr['orientation'] == mo).to_numpy()
                if not (mL.any() and mR.any()):
                    continue
                xl, xr = Xs[mL].mean(axis=0), Xs[mR].mean(axis=0)
                num += np.array([np.sum((xl - s[:, None] * xr) ** 2) for s in pats])
                den += float(np.sum(xl ** 2))
        D = num / den
        order = np.argsort(D)
        k_star = int(np.flatnonzero((pats == star).all(axis=1))[0])
        rank = int(np.flatnonzero(order == k_star)[0]) + 1
        margins = {}
        for j, c in enumerate(SIGN_CHANNELS):
            s2 = star.copy()
            s2[j] *= -1
            k2 = int(np.flatnonzero((pats == s2).all(axis=1))[0])
            margins[c] = float(D[k2] - D[k_star])
        best = pats[order[0]]
        rows.append({'platform': pl, 'rank_sigma_star': rank, 'D_sigma_star': float(D[k_star]),
                     'D_best': float(D[order[0]]), 'best_pattern': ''.join('+' if v > 0 else '-' for v in best),
                     'D_identity': float(D[0]), **{f'margin_{c}': v for c, v in margins.items()}})
        if pl != 'all':
            ctx.summary.setdefault('sign_patterns_D', {})[pl] = D.tolist()
    sp = pd.DataFrame(rows)
    write_table(sp, 'sign_patterns')
    ctx.summary['sign_patterns'] = sp.to_dict(orient='records')
    saved = {}
    for pl, D in ctx.summary['sign_patterns_D'].items():
        D = np.asarray(D)
        order = np.argsort(D)
        k_star = int(np.flatnonzero((pats == star).all(axis=1))[0])
        saved[pl] = {'D_sorted': D[order], 'rank_sigma_star': int(np.flatnonzero(order == k_star)[0]) + 1,
                     'D_sigma_star': float(D[k_star])}
    save_input('sign_patterns', {'channels': list(SIGN_CHANNELS),
                                 'sigma_star': '(' + ','.join('+' if v > 0 else '-' for v in SIGMA_STAR[:3]) + ';'
                                               + ','.join('+' if v > 0 else '-' for v in SIGMA_STAR[3:]) + ')',
                                 'platforms': saved})


def first_half_yaw(ctx: Ctx) -> None:
    d = ctx.d
    tr = d.trials.copy()
    feats = {'first_half_yaw_deg': 'first-half yaw', 'net_yaw_deg': 'net yaw'}
    tr['second_half_yaw_deg'] = tr['net_yaw_deg'] - tr['first_half_yaw_deg']
    feats['second_half_yaw_deg'] = 'second-half yaw'
    rows = []
    for pid, g in tr.groupby('person_id'):
        y = (g['hand'] == 'right').astype(int).to_numpy()
        r = {'person_id': pid, 'platform': g['platform'].iloc[0], 'n_left': int((y == 0).sum()), 'n_right': int(y.sum())}
        for f in feats:
            r[f'auc_{f}'] = roc_auc(y, g[f].to_numpy())
            r[f'median_left_{f}'] = float(g.loc[g['hand'] == 'left', f].median())
            r[f'median_right_{f}'] = float(g.loc[g['hand'] == 'right', f].median())
        sgn = np.sign(g['first_half_yaw_deg'].to_numpy())
        r['sign_rule_accuracy'] = float(np.mean((sgn > 0) == (y == 1)))
        rows.append(r)
    pp = pd.DataFrame(rows)
    y = (tr['hand'] == 'right').astype(int).to_numpy()
    pooled = {f: roc_auc(y, tr[f].to_numpy()) for f in feats}
    by_pl = {pl: roc_auc((g['hand'] == 'right').astype(int).to_numpy(), g['first_half_yaw_deg'].to_numpy())
             for pl, g in tr.groupby('platform')}
    sign_acc = float(np.mean((tr['first_half_yaw_deg'] > 0).to_numpy() == (y == 1)))
    write_table(pp.copy(), 'first_half_yaw_per_person')
    ctx.summary['first_half_yaw'] = {
        'pooled_auc': pooled, 'pooled_auc_by_platform': by_pl, 'sign_rule_accuracy_pooled': sign_acc,
        'persons_auc_gt_0p5': int((pp['auc_first_half_yaw_deg'] > 0.5).sum()),
        'persons_auc_ge_0p8': int((pp['auc_first_half_yaw_deg'] >= 0.8).sum()),
        'per_person_auc_median': float(pp['auc_first_half_yaw_deg'].median()),
        'per_person_auc_min': float(pp['auc_first_half_yaw_deg'].min()),
        'median_right_deg': float(tr.loc[tr['hand'] == 'right', 'first_half_yaw_deg'].median()),
        'median_left_deg': float(tr.loc[tr['hand'] == 'left', 'first_half_yaw_deg'].median()),
        'net_yaw_per_person_auc_median': float(pp['auc_net_yaw_deg'].median()),
    }
    write_table(pd.DataFrame([{'feature': f, 'pooled_auc': v} for f, v in pooled.items()]), 'first_half_yaw_pooled')


def platform_differences(ctx: Ctx) -> None:
    d = ctx.d
    tr = d.trials
    Q = ctx.per_trial_quality()
    rows = []
    for pl, g in tr.groupby('platform'):
        y = (g['hand'] == 'right').astype(int).to_numpy()
        qq = Q[(Q['platform'] == pl) & (Q['analysis_set'] == 'primary')]
        rows.append({'platform': pl, 'persons': g['person_id'].nunique(), 'trials': len(g),
                     'duration_median_s': g['duration_ms'].median() / 1000, 'peak_gyro_median_dps': g['peak_gyro_dps'].median(),
                     'peak_acc_median_ms2': g['peak_acc_ms2'].median(), 'closure_median_deg': g['closure_deg'].median(),
                     'rest_accg_sd_median': qq['rest_accg_sd'].median(), 'rest_gyro_sd_median': qq['rest_gyro_sd'].median(),
                     'first_half_yaw_median_right': g.loc[g['hand'] == 'right', 'first_half_yaw_deg'].median(),
                     'first_half_yaw_median_left': g.loc[g['hand'] == 'left', 'first_half_yaw_deg'].median(),
                     'auc_first_half_yaw': roc_auc(y, g['first_half_yaw_deg'].to_numpy()),
                     'return_ok_share': float(d.ret_ok[(tr['platform'] == pl).to_numpy()].mean())})
    pdf = pd.DataFrame(rows)
    write_table(pdf, 'platform_differences')
    ctx.summary['platform_differences'] = pdf.to_dict(orient='records')
    k = d.channels.index('yaw_rate')
    saved = {}
    for pl in PLATFORMS:
        m = (tr['platform'] == pl).to_numpy()
        if not m.any():
            continue
        hands = {}
        for h in ('left', 'right'):
            M, ids = person_means(d.approach[m, k, :], d.groups[m], tr['hand'].to_numpy()[m], h)
            hands[h] = {'persons': ids, 'means': M}
        saved[pl] = {'n_trials': int(m.sum()), 'n_persons': len(set(d.groups[m])), 'hands': hands}
    save_input('platform_means', {'channel': 'yaw_rate', 'x_pct': np.linspace(0, 100, d.approach.shape[2]),
                                  'platforms': saved})


def return_risk(ctx: Ctx) -> None:
    """What an approach-trained rule says about the RETURN movement (the cue-free risk)."""
    d = ctx.d
    tr = d.trials
    ok = d.ret_ok
    k = d.channels.index('yaw_cum')
    mid = d.ret.shape[2] // 2
    ret_fh = np.where(ok, 0.5 * (d.ret[:, k, mid - 1] + d.ret[:, k, mid]), np.nan)
    app_fh = tr['first_half_yaw_deg'].to_numpy()
    y = (tr['hand'] == 'right').astype(int).to_numpy()
    sel = ok & np.isfinite(ret_fh)
    acc_app = float(np.mean((app_fh[sel] > 0) == (y[sel] == 1)))
    acc_ret = float(np.mean((ret_fh[sel] > 0) == (y[sel] == 1)))
    net_ret = np.where(ok, d.ret[:, k, -1], np.nan)
    acc_ret_net = float(np.mean((net_ret[sel] > 0) == (y[sel] == 1)))
    out = {'trials_with_return': int(sel.sum()), 'persons_with_return': int(len(set(d.groups[sel]))),
           'sign_rule_accuracy_on_approach': acc_app, 'sign_rule_accuracy_on_return_first_half': acc_ret,
           'sign_rule_accuracy_on_return_net_yaw': acc_ret_net,
           'auc_return_first_half_yaw': roc_auc(y[sel], ret_fh[sel]),
           'auc_return_net_yaw': roc_auc(y[sel], net_ret[sel]),
           'return_duration_median_s': float(np.nanmedian(tr.loc[sel, 'return_duration_ms']) / 1000)}
    ctx.summary['return_risk'] = out
    write_table(pd.DataFrame([out]).T.reset_index().rename(columns={'index': 'quantity', 0: 'value'}), 'return_risk')
    kr = d.channels.index('yaw_rate')
    parts = {}
    for part, X in (('approach', d.approach[sel, kr, :]), ('return', d.ret[sel, kr, :])):
        parts[part] = {}
        for h in ('left', 'right'):
            M, ids = person_means(X, d.groups[sel], tr['hand'].to_numpy()[sel], h)
            parts[part][h] = {'persons': ids, 'means': M}
    save_input('return_means', {'channel': 'yaw_rate', 'x_pct': np.linspace(0, 100, d.approach.shape[2]),
                                'n_trials': int(sel.sum()), 'parts': parts})


# ───────────────────────────── 10. cue-free baseline, stillness, closure ─────────────────────────────


def cuefree_baseline(ctx: Ctx) -> None:
    d = ctx.d
    ev = cuefree.evaluate(d)
    ev['platform'] = d.trials['platform'].to_numpy()
    summ = cuefree.summarize(ev)
    by_pl = {pl: cuefree.summarize(g) for pl, g in ev.groupby('platform')}
    write_table(ev, 'cuefree_v0_per_trial')
    write_table(pd.DataFrame([{'platform': 'all', **summ}] + [{'platform': k, **v} for k, v in by_pl.items()]), 'cuefree_v0_summary')
    ctx.summary['cuefree_v0'] = {'all': summ, **by_pl, 'params': ev.attrs.get('params')}
    # stillness before the oracle onset, measured in v0's own activity signal: the still run whose last still
    # sample lies within 300 ms of the onset (nothing added for the smoothing), for the gyro-only criterion of
    # v0 and for gyro AND browser linear acceleration (|a| < 0.5 m/s^2, the app's still_acc_ms2)
    def run_s(t: np.ndarray, still: np.ndarray, on: float) -> float:
        r = cuefree.still_run_before(t, np.where(still, 0.0, 1.0), on, still_dps=0.5, tol_ms=300.0)
        return 0.0 if r is None else float((t[r[1]] - t[r[0]] + 1000.0 / C.FS_HZ) / 1000.0)

    pre, pre_both = [], []
    for i in range(d.n):
        s = d.stream(i)
        act = cuefree.activity(s['t'], s['gyro'])
        acc = cuefree.activity(s['t'], np.nan_to_num(s['acc']))
        on = float(d.trials['onset_ms'].iloc[i])
        pre.append(run_s(s['t'], act < 20.0, on))
        pre_both.append(run_s(s['t'], (act < 20.0) & (acc < 0.5), on))
    pre, pre_both = np.asarray(pre), np.asarray(pre_both)
    ctx.summary['stillness_before_onset'] = {
        'definition': "still run of v0's activity signal (|w| smoothed, < 20 deg/s) whose last still sample is within "
                      '300 ms of the oracle onset; nothing added for the smoothing',
        'protocol_enforced': 'the collection app held GO back until 400 ms of stillness, then waited a 400-900 ms '
                             'foreperiod, then the reaction time: this stillness is not what production will see',
        'median_s': float(np.median(pre)), 'share_ge_1s': float(np.mean(pre >= 1.0)),
        'share_ge_0p5s': float(np.mean(pre >= 0.5)), 'gyro_and_acc_median_s': float(np.median(pre_both)),
        'gyro_and_acc_share_ge_1s': float(np.mean(pre_both >= 1.0))}
    save_input_csv(pd.DataFrame({'trial_id': d.trials['trial_id'].to_numpy(), 'platform': d.trials['platform'].to_numpy(),
                                 'still_s_gyro': pre, 'still_s_gyro_and_acc': pre_both}), 'stillness_before_onset')


def cuefree_stress(ctx: Ctx) -> None:
    """v0 when the protocol-enforced stillness before the approach is cut to keep_s seconds."""
    st = cuefree.stress_evaluate(ctx.d, keep_s=STRESS_KEEP_S, random_max_s=STRESS_RANDOM_MAX_S)
    rows = []
    for (setting, pl), g in [((s, 'all'), g) for s, g in st.groupby('setting', sort=False)] + \
            [((s, p), g) for (s, p), g in st.groupby(['setting', 'platform'], sort=False)]:
        r = {'setting': setting, 'platform': pl, 'n': len(g), 'keep_s_mean': float(g['keep_s'].mean()),
             'removed_s_median': float(g['removed_ms'].median() / 1000)}
        for o in cuefree.OUTCOMES:
            r[f'share_{o}'] = float((g['outcome'] == o).mean())
        r['share_approach_in_any_detection'] = float(g['approach_in_any'].mean())
        rows.append(r)
    summ = pd.DataFrame(rows)
    write_table(summ, 'cuefree_v0_stress')
    allp = summ[summ['platform'] == 'all'].set_index('setting')
    ctx.summary['cuefree_v0_stress'] = {
        'what': 'the still run before the onset is cut to keep_s seconds (the samples right after the earlier '
                f'activity are removed); random = keep ~ U(0, {STRESS_RANDOM_MAX_S:g} s) per trial, seeded',
        **{s: {'share_approach': float(r['share_approach']), 'share_earlier_movement': float(r['share_earlier_movement']),
               'share_none': float(r['share_none']), 'share_return': float(r['share_return']),
               'share_approach_in_any_detection': float(r['share_approach_in_any_detection'])}
           for s, r in allp.iterrows()}}
    save_input('stress_settings', {'keep_s': list(STRESS_KEEP_S), 'random_max_s': STRESS_RANDOM_MAX_S,
                                  'min_still_s': cuefree.CueFreeParams().min_still_s})


def gravity_closure(ctx: Ctx) -> None:
    tr = ctx.d.trials
    v = tr['closure_deg'].astype(float)
    ctx.summary['gravity_closure'] = {'median_deg': float(v.median()), 'p90_deg': float(v.quantile(.9)),
                                      'share_over_3deg': float((v > CLOSURE_REPORT_DEG).mean()),
                                      'by_platform_median': tr.groupby('platform')['closure_deg'].median().to_dict(),
                                      'rest_pre_fallback_share': float((tr['rest_pre_n'] == 0).mean()),
                                      'rest_post_missing_share': float((tr['rest_post_n'] == 0).mean())}


# ───────────────────────────── 11. flags, gates, label consistency ─────────────────────────────


def flags_gates(ctx: Ctx) -> None:
    """Counts of the kept flags (docs/02 T8-T12) and of the predictor gates (docs/09 D5) in the primary set."""
    T = ctx.T
    prim = T[T['analysis_set'] == 'primary']
    rows = []
    for c, desc in {**FLAG_COLUMNS, **GATE_COLUMNS, 'gate_pass': 'G1 and G2 both pass'}.items():
        kind = 'flag (kept)' if c in FLAG_COLUMNS else 'gate (scored apart)'
        v = prim[c].astype(bool) if c in FLAG_COLUMNS else ~prim[c].astype(bool)
        rows.append({'column': c, 'kind': kind, 'counted': 'flagged' if c in FLAG_COLUMNS else 'failing',
                     'meaning': desc, 'primary_trials': int(v.sum()),
                     'android': int(v[prim['platform'] == 'android'].sum()), 'ios': int(v[prim['platform'] == 'ios'].sum()),
                     'left': int(v[prim['hand'] == 'left'].sum()), 'right': int(v[prim['hand'] == 'right'].sum()),
                     'persons': int(prim.loc[v, 'person_id'].nunique()),
                     'by_person': ';'.join(f'{p}:{k}' for p, k in prim.loc[v, 'person_id'].value_counts().sort_index().items())})
    fg = pd.DataFrame(rows)
    write_table(fg, 'flags_and_gates')
    gm = prim[prim['grip_mismatch']]
    detail = (gm.groupby(['person_id', 'session_id', 'block_index', 'hand', 'instructed_rotation', 'rotation_start',
                          'grip_detected'], dropna=False).size().rename('trials').reset_index())
    write_table(detail, 'grip_mismatch_detail')
    g1 = prim[~prim['gate_g1']]
    S = ctx.summary
    S['flags'] = {r['column']: {'primary_trials': r['primary_trials'], 'android': r['android'], 'ios': r['ios'],
                                'persons': r['persons']} for r in rows if r['column'] in FLAG_COLUMNS}
    S['flags']['grip_mismatch_app'] = int(prim['grip_mismatch_app'].sum())
    S['flags']['grip_mismatch_gravity'] = int(prim['grip_mismatch_gravity'].sum())
    S['flags']['grip_mismatch_definite'] = int((prim['grip_mismatch'] & prim['grip_detected'].isin(
        ['upright', 'top_left', 'top_right', 'upside_down', 'flat'])).sum())
    S['gates'] = {r['column']: {'primary_trials_failing': r['primary_trials'], 'android': r['android'],
                                'ios': r['ios']} for r in rows if r['column'] not in FLAG_COLUMNS}
    S['gates']['g1_failures_max_gap_ms'] = [q(g1['g1_gap_max_ms'], 0), q(g1['g1_gap_max_ms'], 1)] if len(g1) else []
    S['gates']['g2_failures_over_8s_by_speed'] = prim.loc[prim['duration_ms'] > C.G2_MAX_DURATION_MS, 'speed'].value_counts().to_dict()


def label_consistency(ctx: Ctx) -> None:
    """Per-block label consistency (the docs/09 D6 screen, descriptive) and the swap signature."""
    d = ctx.d
    t, b = screen.block_consistency(d.trials, d.approach)
    write_table(b, 'label_consistency_blocks')
    y = t['y_right'].astype(int).to_numpy()
    p = np.where(y == 1, t['p_label_screen'], 1 - t['p_label_screen'])
    sw = {ev: b.loc[b[f'{ev}_swap_signature'], ['session_label', 'block_index', 'hand', 'n', f'{ev}_agree']]
          .to_dict(orient='records') for ev in ('fhy', 'screen')}
    # the blocks with the swap signature (data-driven, never a typed person or block), trial by trial
    detail = []
    for sl, g in b[b['fhy_swap_signature']].groupby('session_label', sort=True):
        blocks = sorted(int(v) for v in g['block_index'])
        tt = t[(t['session_label'] == sl) & t['block_index'].isin(blocks)]
        detail.append({'session_label': sl, 'person_id': str(g['person_id'].iloc[0]), 'blocks': blocks,
                       'trials': int(len(tt)), 'fhy_agree': int(tt['fhy_agrees'].sum()),
                       'screen_agree': int(tt['screen_agrees'].sum()),
                       'screen_p_label': [round(float(v), 2) for v in tt['p_label_screen']],
                       'first_half_yaw_deg': [round(float(v), 1) for v in tt['first_half_yaw_deg']],
                       'd6_flags': int(tt['screen_flag'].sum())})
    n_persons = int(t['person_id'].nunique())
    ctx.summary['label_consistency'] = {
        'screen_model': 'antisymmetric logistic regression (no intercept) on yaw_cum and yaw_rate at 1/8..8/8 of '
                        f'the approach, each person scored by a model fitted on the other {n_persons - 1} (docs/09 D6)',
        'screen_loso_accuracy': float(np.mean((p >= 0.5) == (y == 1))),
        'trials_p_label_below_0p05': int((t['p_label_screen'] < screen.P_FLAG).sum()),
        'trials_flagged_by_d6_rules': int(t['screen_flag'].sum()),
        'blocks': int(len(b)), 'blocks_all_contradict_fhy': int(b['fhy_all_contradict'].sum()),
        'blocks_all_contradict_screen': int(b['screen_all_contradict'].sum()),
        'swap_signature_fhy': sw['fhy'], 'swap_signature_screen': sw['screen'],
        'swap_signature_blocks_detail': detail,
    }


# ───────────────────────────── compute ─────────────────────────────

#: the analysis sections in run order (the order of the summary keys); each writes its tables, its summary
#: keys and the inputs of its figures
SECTIONS: dict[str, Callable[[Ctx], None]] = {
    'participants': participants, 'exclusion_flow': exclusion_flow, 'completion': completion,
    'conditions': conditions, 'primary_trials': primary_trials, 'raw_quality': raw_quality,
    'durations_segmentation': durations_segmentation, 'example_traces': example_traces, 'class_means': class_means,
    'sign_patterns': sign_patterns, 'first_half_yaw': first_half_yaw, 'platform_differences': platform_differences,
    'return_risk': return_risk, 'cuefree': cuefree_baseline, 'cuefree_stress': cuefree_stress,
    'gravity_closure': gravity_closure, 'flags_gates': flags_gates, 'label_consistency': label_consistency,
}
#: the figures whose inputs each section writes (``--only S2_05_sampling_gaps`` selects raw_quality)
SECTION_FIGURES: dict[str, tuple[str, ...]] = {
    'participants': ('S2_01_demographics',), 'exclusion_flow': ('S2_02_exclusion_flow',),
    'completion': ('S2_03_completion_durations',), 'conditions': ('S2_04_conditions',),
    'primary_trials': ('S2_13_first_half_yaw', 'S2_14_platforms', 'S2_17_gravity_closure'),
    'raw_quality': ('S2_05_sampling_gaps', 'S2_06_noise_vs_quantisation', 'S2_07_spectrum'),
    'durations_segmentation': ('S2_08_durations_peaks', 'S2_09_segmentation_contact'),
    'example_traces': ('S2_10_example_traces',), 'class_means': ('S2_11_class_means',),
    'sign_patterns': ('S2_12_sign_patterns',), 'first_half_yaw': ('S2_13_first_half_yaw',),
    'platform_differences': ('S2_14_platforms',), 'return_risk': ('S2_15_return_vs_approach',),
    'cuefree': ('S2_16_cuefree_v0',), 'cuefree_stress': ('S2_19_cuefree_stress',),
    'gravity_closure': ('S2_17_gravity_closure',), 'flags_gates': ('S2_18_flags_gates',),
    'label_consistency': ('S2_20_label_consistency',),
}


def outputs_current() -> bool:
    """The summary and every figure input exist and were made from the current cache."""
    if not PL.file_current(SUMMARY_JSON):
        return False
    for f in INPUT_FILES:
        p = DATA_DIR / f'describe_{f}'
        if not p.exists() or (p.suffix == '.json' and not PL.file_current(p)):
            return False
    return True


def selected_sections(only: list[str] | None) -> list[str]:
    """Sections selected by name or by the name of a figure they feed (shell patterns allowed)."""
    return [s for s in SECTIONS if PL.selected(s, only) or any(PL.selected(f, only) for f in SECTION_FIGURES[s])]


def compute(force: bool = False, only: list[str] | None = None) -> None:
    """Run the analysis sections and save their tables, the summary and the figure inputs.

    Skipped when the saved outputs were made from the current cache (``force`` reruns). With ``only`` (section
    or figure names) and current outputs, just those sections rerun and their summary keys are updated in place;
    when the outputs are missing or stale, every section runs."""
    current = outputs_current()
    if current and not force:
        print('describe: outputs current (same cache); skipped (--force recomputes)', flush=True)
        return
    names = list(SECTIONS)
    summary: dict[str, Any] = {}
    if only:
        if current:
            names = selected_sections(only)
            if not names:
                print(f'describe: no section matches {only} (sections: {", ".join(SECTIONS)})', flush=True)
                return
            summary = load_summary()
        else:
            print('describe: saved outputs are missing or from another cache: every section runs', flush=True)
    t0 = time.time()
    C.ensure_out_dirs()
    ctx = Ctx(summary)
    print(f'cache: {ctx.d.n} primary trials; describing ({", ".join(names)}) ...', flush=True)
    for name in names:
        t1 = time.time()
        SECTIONS[name](ctx)
        print(f'  {name}: {time.time() - t1:.1f} s', flush=True)
    ctx.summary['provenance'] = PL.provenance()          # the cache this summary describes (stage2.pipeline staleness)
    SUMMARY_JSON.write_text(json.dumps(ctx.summary, indent=1, default=float), encoding='utf-8')
    print(f'describe compute done in {time.time() - t0:.0f} s -> {C.OUT_DIR}', flush=True)


# ───────────────────────────── presentation: figures, report tables, numbers (saved files only) ─────────────────────────────


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.scripts import describe_figures
    describe_figures.draw(only)


def _tex_text(s: Any) -> str:
    """Plain text -> LaTeX (escaped; |w| -> $|\\omega|$, comparison signs in math mode, typographic quotes)."""
    t = RX.latex_escape(str(s))
    t = t.replace('|w|', r'$|\omega|$').replace('<=', r'$\le$').replace('>=', r'$\ge$')
    t = re.sub(r'(?<!\$)<(?!\$)', r'$<$', t)
    t = re.sub(r'(?<!\$)>(?!\$)', r'$>$', t)
    return re.sub(r'"([^"]*)"', r"``\1''", t)


def _export(df: pd.DataFrame, name: str, caption: str, **kw: Any) -> None:
    RX.export_table(df, DATA_DIR / name, caption=caption, label=f'tab:{name}', **kw)


PLATFORM_NAMES = {'all': 'All', 'android': 'Android', 'ios': 'iOS'}


def _by_platform(df: pd.DataFrame, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """A narrow table: one row per quantity (LaTeX label, column, format), one column per platform."""
    out: dict[str, list[str]] = {'Quantity': [lab for lab, _, _ in rows]}
    for _, r in df.iterrows():
        out[PLATFORM_NAMES.get(r['platform'], r['platform'])] = [RX.format_value(r[c], f) for _, c, f in rows]
    return pd.DataFrame(out)


def _export_by_platform(df: pd.DataFrame, rows: list[tuple[str, str, str]], name: str, caption: str) -> None:
    t = _by_platform(df, rows)
    _export(t, name, caption=caption, raw=('Quantity',), align='l' + 'r' * (t.shape[1] - 1))


def table_exclusion_flow() -> None:
    f = load_table('exclusion_flow')
    df = pd.DataFrame({'Step': f['step'], 'Rule': [_tex_text(r) for r in f['rule']], 'Removed': f['removed'],
                       'Remaining': f['remaining'], 'Persons': f['persons'], 'Sessions': f['sessions'],
                       'Natural use': f['natural_use']})
    _export(df, 'describe_exclusion_flow', raw=('Rule',), align='lp{4.6cm}rrrrr',
            caption=('Exclusion cascade: rules apply in order and an attempt is charged to the first rule it fails. '
                     'E3 moves the free-choice trials (hand self-reported) to the natural-use set, which then passes '
                     'the same quality rules; N1 keeps one natural-use trial per person (the earliest).'))


def table_e1_reasons() -> None:
    e = load_table('exclusion_E1_breakdown')
    df = pd.DataFrame({'Reason (first that applies)': [_tex_text(r) for r in e['reason']], 'Attempts': e['attempts']})
    _export(df, 'describe_e1_reasons', raw=('Reason (first that applies)',),
            caption='Why attempts were not accepted (rule E1).')


def table_participants() -> None:
    dm = load_summary()['demographics']
    rows = [('Persons', f"{dm['n_persons']}"),
            ('Women / men', f"{dm['n_female']} / {dm['n_male']}"),
            ('Age (years), mean $\\pm$ SD (range)',
             f"{dm['age_mean']:.1f} $\\pm$ {dm['age_sd']:.1f} ({dm['age_min']:.0f}--{dm['age_max']:.0f})"),
            ('Height (cm), mean $\\pm$ SD (range; n)',
             f"{dm['height_mean']:.1f} $\\pm$ {dm['height_sd']:.1f} ({dm['height_min']:.0f}--{dm['height_max']:.0f}; "
             f"{dm['height_n']})"),
            ('Dominant hand right / left', f"{dm['n_right_handed']} / {dm['n_left_handed']}"),
            ('Usual phone hand right / both / left',
             f"{dm['n_phone_hand_right']} / {dm['n_phone_hand_both']} / {dm['n_phone_hand_left']}"),
            ('Persons on Android / iOS', f"{dm['persons_android']} / {dm['persons_ios']}"),
            ('Included sessions (Android / iOS)',
             f"{dm['sessions_included']} ({dm['sessions_android']} / {dm['sessions_ios']})"),
            ('Device models', f"{dm['n_device_models']}"),
            ('Persons with several sessions', f"{dm['n_persons_multi_session']}"),
            ('Primary trials (left / right)', f"{dm['primary_trials']} ({dm['primary_left']} / {dm['primary_right']})"),
            ('Primary trials (Android / iOS)', f"{dm['primary_trials']} ({dm['primary_android']} / {dm['primary_ios']})")]
    df = pd.DataFrame(rows, columns=['Quantity', 'Value'])
    _export(df, 'describe_participants', raw=('Quantity', 'Value'), caption='Participants of the analysis.')


def table_sampling() -> None:
    s = load_table('sampling_quality')
    rows = [('Complete attempts', 'trials', 'd'), ('Sampling rate, median (Hz)', 'hz_median', '.1f'),
            ('Sample interval, median (ms)', 'dt_median_ms', '.1f'),
            ('Sample interval, 99th percentile (ms)', 'dt_p99_ms', '.1f'),
            (f'Gaps $>$ {GAP_LIST_MS:g} ms', 'gaps_50', 'd'), ('Gap duration, median (ms)', 'gap_ms_median', '.0f'),
            ('Attempts with a gap $>$ 100 ms', 'trials_with_gap_100', 'd'),
            ('Rest SD of the specific force, median (m/s$^2$)', 'rest_accg_sd_median', '.3f'),
            ('Rest SD of the angular velocity, median (deg/s)', 'rest_gyro_sd_median', '.2f')]
    _export_by_platform(s, rows, 'describe_sampling',
                        caption=('Sampling of the complete attempts per platform (medians over attempts): the rate, '
                                 'the sample intervals, the gaps and the hand-held noise in the still window before GO.'))


def table_flags_gates() -> None:
    f = load_table('flags_and_gates')
    df = pd.DataFrame({'Check': [_tex_text(FLAG_LABELS.get(c, c)) for c in f['column']],
                       'Trials': f['primary_trials'], 'Android': f['android'], 'iOS': f['ios'],
                       'Left': f['left'], 'Right': f['right'], 'Persons': f['persons']})
    _export(df, 'describe_flags_gates', raw=('Check',), align='p{5.0cm}rrrrrr',
            caption=('Flags of docs/02 T8--T12 (kept in every analysis) and the predictor gates G1, G2 of docs/09 D5 '
                     '(failing trials stay in training and in the headline metrics, and every result also reports the '
                     'metrics on the trials that pass both gates): primary trials flagged or failing, by platform and '
                     'by hand, and the persons concerned.'))


OUTCOME_ROWS = {'approach': 'the approach (IoU $\\ge$ 0.5)', 'approach_partial': 'part of the approach',
                'earlier_movement': 'an earlier movement', 'return': 'the return of the arm',
                'later_movement': 'a later movement', 'none': 'nothing'}


def table_cuefree_v0() -> None:
    s = load_table('cuefree_v0_summary')
    rows = ([('Trials', 'n_trials', 'd')] + [(f'First detection: {OUTCOME_ROWS[o]}', f'share_{o}', '.0%')
                                             for o in cuefree.OUTCOMES]
            + [('The approach among all detections', 'share_approach_in_any_detection', '.0%'),
               ('Onset lag when it finds the approach, median (ms)', 'onset_error_median_ms', '.0f'),
               ('Absolute onset error, 90th percentile (ms)', 'onset_abs_error_p90_ms', '.0f')])
    _export_by_platform(s, rows, 'describe_cuefree_v0',
                        caption=('Cue-free detector v0 on the recorded primary trials: what it finds first, whether '
                                 'the approach is among all its detections, and how late its onset is. The stillness '
                                 'before each approach was enforced by the collection protocol, so these shares are '
                                 'upper bounds for production (see the stress test).'))


def table_cuefree_stress() -> None:
    s = load_table('cuefree_v0_stress')
    a = s[s['platform'] == 'all']
    df = pd.DataFrame({'Stillness kept (s)': a['setting'].astype(str), 'Approach': a['share_approach'],
                       'Part': a['share_approach_partial'], 'Earlier': a['share_earlier_movement'],
                       'Return': a['share_return'], 'Nothing': a['share_none'],
                       'Anywhere': a['share_approach_in_any_detection']})
    _export(df, 'describe_cuefree_stress',
            column_formats={c: '.0%' for c in df.columns if c != 'Stillness kept (s)'},
            caption=('Stress test of v0: the stillness before each approach is cut to the given length (recorded: '
                     'unchanged; random: uniform per trial, seeded) and v0 is run again. Shares of what it finds first '
                     '(the approach, part of it, an earlier movement, the return, nothing) and of trials whose approach '
                     'is among all its detections (anywhere).'))


def table_first_half_yaw() -> None:
    s = load_summary()['first_half_yaw']
    pp = load_table('first_half_yaw_per_person')
    names = {'first_half_yaw_deg': 'First-half yaw', 'net_yaw_deg': 'Net yaw', 'second_half_yaw_deg': 'Second-half yaw'}
    df = pd.DataFrame([{'Feature': lab, 'Pooled': s['pooled_auc'][f],
                        'Person median': float(pp[f'auc_{f}'].median()), 'Person min': float(pp[f'auc_{f}'].min()),
                        'Persons > 0.5': int((pp[f'auc_{f}'] > 0.5).sum()), 'Persons': len(pp)}
                       for f, lab in names.items()])
    _export(df, 'describe_first_half_yaw',
            column_formats={'Pooled': '.2f', 'Person median': '.2f', 'Person min': '.2f'},
            caption=('How well the yaw about gravity accumulated over parts of the approach separates the hands: AUC of '
                     'right vs left pooled over trials, and within each person (median, lowest, and the persons above '
                     '0.5).'))


def table_platforms() -> None:
    p = load_table('platform_differences')
    rows = [('Persons', 'persons', 'd'), ('Trials', 'trials', 'd'), ('Approach duration (s)', 'duration_median_s', '.2f'),
            ('Peak $|\\omega|$ (deg/s)', 'peak_gyro_median_dps', '.0f'),
            ('Gyro-gravity closure (deg)', 'closure_median_deg', '.1f'),
            ('First-half yaw, right hand (deg)', 'first_half_yaw_median_right', '.1f'),
            ('First-half yaw, left hand (deg)', 'first_half_yaw_median_left', '.1f'),
            ('AUC of the first-half yaw', 'auc_first_half_yaw', '.2f'),
            ('Trials with a clean return window', 'return_ok_share', '.0%')]
    _export_by_platform(p, rows, 'describe_platforms',
                        caption='Android and iOS after standardisation (medians over the primary trials unless stated).')


def table_mirror_symmetry() -> None:
    m = load_table('mirror_symmetry')
    df = pd.DataFrame({'Channel': m['channel'], 'Sign': m['mirror_sign'], 'RMS': m['rms_class_mean'],
                       'L vs R': m['rms_left_vs_right'], 'Mirrored L vs R': m['rms_mirrored_left_vs_right'],
                       'Closer': m['persons_closer_to_mirror'], 'Persons': m['persons']})
    _export(df, 'describe_mirror_symmetry',
            column_formats={'RMS': '.2f', 'L vs R': '.2f', 'Mirrored L vs R': '.2f', 'Closer': 'd'},
            caption=('Mirror symmetry of the gravity-frame channels (sign: how the channel changes under the mirror; '
                     'chiral channels change sign): RMS of the right-hand mean, RMS difference between the right-hand '
                     '(R) and the left-hand (L) mean, unmirrored and mirrored, and the persons whose right-hand mean is '
                     'closer to their mirrored left-hand mean (closer).'))


def table_durations() -> None:
    d = load_table('durations_by_speed_platform')
    df = pd.DataFrame({'Speed': d['speed'], 'Platform': d['platform'].map({'android': 'Android', 'ios': 'iOS'}).fillna(d['platform']),
                       'Trials': d['n'], 'Median (s)': d['duration_median_s'], 'Q1 (s)': d['q25_s'], 'Q3 (s)': d['q75_s'],
                       'Peak |w| (deg/s)': d['peak_gyro_median_dps']})
    _export(df, 'describe_durations',
            column_formats={'Median (s)': '.2f', 'Q1 (s)': '.2f', 'Q3 (s)': '.2f', 'Peak |w| (deg/s)': '.0f'},
            caption='Oracle approach duration of the primary trials by instructed speed and platform.')


def table_return_risk() -> None:
    r = load_summary()['return_risk']
    rows = [('Primary trials with a clean return window', f"{r['trials_with_return']}"),
            ('Persons with a return window', f"{r['persons_with_return']}"),
            ('Sign rule (right iff first-half yaw $>$ 0) on the approach', f"{100 * r['sign_rule_accuracy_on_approach']:.0f}\\,\\%"),
            ('The same rule on the return', f"{100 * r['sign_rule_accuracy_on_return_first_half']:.0f}\\,\\%"),
            ('The rule on the net yaw of the return', f"{100 * r['sign_rule_accuracy_on_return_net_yaw']:.0f}\\,\\%"),
            ('AUC of the first-half yaw on the return', f"{r['auc_return_first_half_yaw']:.2f}"),
            ('AUC of the net yaw on the return', f"{r['auc_return_net_yaw']:.2f}"),
            ('Return duration, median (s)', f"{r['return_duration_median_s']:.2f}")]
    _export(pd.DataFrame(rows, columns=['Quantity', 'Value']), 'describe_return_risk', raw=('Quantity', 'Value'),
            caption='What the approach rule says about the return of the arm (the cue-free risk).')


TABLES: dict[str, Callable[[], None]] = {
    'describe_participants': table_participants, 'describe_exclusion_flow': table_exclusion_flow,
    'describe_e1_reasons': table_e1_reasons, 'describe_sampling': table_sampling,
    'describe_durations': table_durations, 'describe_first_half_yaw': table_first_half_yaw,
    'describe_mirror_symmetry': table_mirror_symmetry, 'describe_platforms': table_platforms,
    'describe_return_risk': table_return_risk, 'describe_cuefree_v0': table_cuefree_v0,
    'describe_cuefree_stress': table_cuefree_stress, 'describe_flags_gates': table_flags_gates,
}


def tables(force: bool = False, only: list[str] | None = None) -> None:
    """Report tables tables/describe/describe_<name>.csv + .tex (booktabs), from saved files only."""
    for name, fn in TABLES.items():
        if PL.selected(name, only):
            fn()


def _key(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')


def describe_numbers() -> dict[str, dict[str, Any]]:
    """The descriptive numbers the report text cites, read from saved outputs only."""
    E = RX.numbers_entry
    s = load_summary()
    src = 'tables/describe_summary.json'
    dm, fy, rr = s['demographics'], s['first_half_yaw'], s['return_risk']
    cf = s['cuefree_v0']['all']
    start = next(r for r in s['exclusion_flow'] if r['step'] == 'start')
    part = load_table('participants')
    pc = 'tables/participants.csv'
    sets = load_input('sets')['sets']
    sc = 'tables/describe/describe_sets.json'
    nat, rep = sets['natural_use'], sets['natural_use_repeat']
    n = {
        'n_persons': E(dm['n_persons'], 'd', src, 'persons in the primary set'),
        'n_female': E(dm['n_female'], 'd', src, 'women'),
        'n_male': E(dm['n_male'], 'd', src, 'men'),
        'age_mean': E(dm['age_mean'], '.1f', src, 'mean age (years)'),
        'age_sd': E(dm['age_sd'], '.1f', src, 'SD of age (years)'),
        'age_min': E(dm['age_min'], 'd', src, 'youngest (years)'),
        'age_max': E(dm['age_max'], 'd', src, 'oldest (years)'),
        'n_right_handed': E(dm['n_right_handed'], 'd', src, 'right-handed persons (self-report)'),
        'n_left_handed': E(dm['n_left_handed'], 'd', src, 'left-handed persons (self-report)'),
        'persons_android': E(dm['persons_android'], 'd', src, 'persons on Android'),
        'persons_ios': E(dm['persons_ios'], 'd', src, 'persons on iOS'),
        'n_device_models': E(dm['n_device_models'], 'd', src, 'device models of the included sessions'),
        'sessions_included': E(dm['sessions_included'], 'd', src, 'included sessions'),
        'sessions_total': E(start['sessions'], 'd', src, 'sessions in the snapshot'),
        'attempts_total': E(start['remaining'], 'd', src, 'trial attempts in the snapshot'),
        'primary_trials': E(dm['primary_trials'], 'd', src, 'primary trials'),
        'primary_left': E(dm['primary_left'], 'd', src, 'primary trials, left hand'),
        'primary_right': E(dm['primary_right'], 'd', src, 'primary trials, right hand'),
        'primary_android': E(dm['primary_android'], 'd', src, 'primary trials on Android'),
        'primary_ios': E(dm['primary_ios'], 'd', src, 'primary trials on iOS'),
        'primary_per_person_min': E(int(part['n_primary'].min()), 'd', pc, 'fewest primary trials of one person'),
        'primary_per_person_max': E(int(part['n_primary'].max()), 'd', pc, 'most primary trials of one person'),
        'primary_per_person_median': E(float(part['n_primary'].median()), '.0f', pc, 'median primary trials per person'),
        'natural_use_n': E(nat['n'], 'd', sc, 'natural-use trials (one free-choice trial per person)'),
        'natural_use_right': E(nat['right'], 'd', sc, 'natural-use trials with the right hand'),
        'natural_use_left': E(nat['left'], 'd', sc, 'natural-use trials with the left hand'),
        'natural_use_repeat_n': E(rep['n'], 'd', sc, 'free-choice trials of repeat visits (reported apart)'),
        'fhy_auc_pooled': E(fy['pooled_auc']['first_half_yaw_deg'], '.2f', src, 'pooled AUC of the first-half yaw'),
        'fhy_auc_android': E(fy['pooled_auc_by_platform']['android'], '.2f', src, 'pooled first-half yaw AUC, Android'),
        'fhy_auc_ios': E(fy['pooled_auc_by_platform']['ios'], '.2f', src, 'pooled first-half yaw AUC, iOS'),
        'net_yaw_auc_pooled': E(fy['pooled_auc']['net_yaw_deg'], '.2f', src, 'pooled AUC of the net yaw'),
        'second_half_yaw_auc_pooled': E(fy['pooled_auc']['second_half_yaw_deg'], '.2f', src,
                                        'pooled AUC of the second-half yaw'),
        'fhy_person_auc_median': E(fy['per_person_auc_median'], '.2f', src, 'median within-person first-half yaw AUC'),
        'fhy_person_auc_min': E(fy['per_person_auc_min'], '.2f', src, 'lowest within-person first-half yaw AUC'),
        'fhy_persons_auc_above_half': E(fy['persons_auc_gt_0p5'], 'd', src, 'persons with a within-person AUC above 0.5'),
        'fhy_persons_auc_ge80': E(fy['persons_auc_ge_0p8'], 'd', src, 'persons with a within-person AUC of at least 0.8'),
        'fhy_sign_rule_accuracy': E(fy['sign_rule_accuracy_pooled'], '.0%', src, 'accuracy of the first-half yaw sign'),
        'fhy_median_right_deg': E(fy['median_right_deg'], '.1f', src, 'median first-half yaw, right hand (deg)'),
        'fhy_median_left_deg': E(fy['median_left_deg'], '.1f', src, 'median first-half yaw, left hand (deg)'),
        'return_trials': E(rr['trials_with_return'], 'd', src, 'primary trials with a clean return window'),
        'return_rule_acc_approach': E(rr['sign_rule_accuracy_on_approach'], '.0%', src,
                                      'first-half yaw sign rule on the approach'),
        'return_rule_acc_return': E(rr['sign_rule_accuracy_on_return_first_half'], '.0%', src,
                                    'the same rule on the return'),
        'return_auc_first_half': E(rr['auc_return_first_half_yaw'], '.2f', src, 'first-half yaw AUC on the return'),
    }
    outcomes = (('approach', 'the approach'), ('approach_partial', 'part of the approach (IoU < 0.5)'),
                ('earlier_movement', 'an earlier movement'), ('return', 'the return'), ('none', 'nothing'))
    for o, txt in outcomes:
        n[f'cuefree_share_{o}'] = E(cf[f'share_{o}'], '.0%', src, f'cue-free v0: its first detection is {txt}')
    n['cuefree_share_any_detection'] = E(cf['share_approach_in_any_detection'], '.0%', src,
                                         'cue-free v0: the approach is among all its detections')
    n['cuefree_onset_error_median_ms'] = E(cf['onset_error_median_ms'], '.0f', src, 'cue-free v0: median onset lag (ms)')
    n.update(_more_numbers(s, src))
    return n


def _more_numbers(s: dict[str, Any], src: str) -> dict[str, dict[str, Any]]:
    """The rest of the descriptive numbers the report and the README quote (all from describe_summary.json)."""
    E = RX.numbers_entry
    n: dict[str, dict[str, Any]] = {}
    dm, comp, vd = s['demographics'], s['completion'], s['visit_durations_completed_v2_min']
    n['height_mean'] = E(dm['height_mean'], '.1f', src, 'mean height (cm)')
    n['height_sd'] = E(dm['height_sd'], '.1f', src, 'SD of height (cm)')
    n['height_n'] = E(dm['height_n'], 'd', src, 'persons who gave a height')
    n['n_phone_hand_right'] = E(dm['n_phone_hand_right'], 'd', src, 'usual phone hand: right')
    n['n_phone_hand_both'] = E(dm['n_phone_hand_both'], 'd', src, 'usual phone hand: both')
    n['n_phone_hand_left'] = E(dm['n_phone_hand_left'], 'd', src, 'usual phone hand: left')
    n['sessions_android'] = E(dm['sessions_android'], 'd', src, 'included sessions on Android')
    n['sessions_ios'] = E(dm['sessions_ios'], 'd', src, 'included sessions on iOS')
    n['n_persons_multi_session'] = E(dm['n_persons_multi_session'], 'd', src, 'persons with several included sessions')
    # exclusion cascade: attempts each rule removed
    for r in s['exclusion_flow']:
        if r['step'] not in ('start', 'N1'):
            n[f"excl_{_key(r['step'])}_removed"] = E(r['removed'], 'd', src, f"attempts removed by rule {r['step']}")
    e3 = next((r for r in s['exclusion_flow'] if r['step'] == 'E3'), None)
    if e3 is not None:
        n['free_choice_trials'] = E(e3['natural_use'], 'd', src, 'free-choice trials moved to the natural-use set (E3)')
    n['street_visits'] = E(comp['sessions_v2'], 'd', src, 'included street visits (protocol 2.0.0)')
    n['visits_completed'] = E(comp['completed'], 'd', src, 'street visits completed')
    n['visits_ended_early'] = E(comp['ended_early'], 'd', src, 'street visits ended early')
    n['bonus_taken_completed'] = E(int(comp['bonus_uptake_completed'].split('/')[0]), 'd', src,
                                   'completed visits that took the bonus tier')
    n['attempts_per_accepted'] = E(comp['attempts_per_accepted'], '.2f', src, 'attempts per accepted trial')
    n['visit_minutes_median'] = E(vd['total_median'], '.1f', src, 'duration of a completed visit, median (min)')
    n['visit_minutes_q1'] = E(vd['total_iqr'][0], '.1f', src, 'duration of a completed visit, first quartile (min)')
    n['visit_minutes_q3'] = E(vd['total_iqr'][1], '.1f', src, 'duration of a completed visit, third quartile (min)')
    for k in range(5):
        if f'tier{k}_median' in vd:
            n[f'tier{k}_minutes_median'] = E(vd[f'tier{k}_median'], '.1f', src, f'duration of tier {k}, median (min)')
    for r in s['sampling']:
        pl = _key(r['platform'])
        n[f'{pl}_interval_median_ms'] = E(r['dt_median_ms'], '.1f', src, f"median sample interval, {r['platform']} (ms)")
        n[f'{pl}_interval_p99_ms'] = E(r['dt_p99_ms'], '.1f', src, f"99th-percentile sample interval, {r['platform']} (ms)")
        n[f'{pl}_gap_median_ms'] = E(r['gap_ms_median'], '.0f', src, f"median gap > 50 ms, {r['platform']} (ms)")
        n[f'{pl}_rest_acc_sd'] = E(r['rest_accg_sd_median'], '.3f', src, f"hand-held specific-force SD at rest, {r['platform']} (m/s^2)")
        n[f'{pl}_rest_gyro_sd'] = E(r['rest_gyro_sd_median'], '.1f', src, f"hand-held angular-velocity SD at rest, {r['platform']} (deg/s)")
    n['quantisation_noise_sd'] = E(s['quantisation_noise_sd'], '.3f', src, "SD of Chrome's 0.1-unit rounding")
    ig = s['ios_long_gaps']
    n['ios_long_gaps'] = E(ig['n'], 'd', src, 'iOS sampling gaps longer than 200 ms')
    n['ios_long_gap_attempts'] = E(ig['trials'], 'd', src, 'iOS attempts with such a gap')
    n['ios_long_gap_median_ms'] = E(ig['duration_median_ms'], '.0f', src, 'their median duration (ms)')
    n['ios_long_gap_start_median_s'] = E(ig['start_after_recording_start_median_s'], '.1f', src,
                                         'their median start after the recording start (s)')
    n['ios_long_gap_share_after_cue'] = E(ig['share_within_400ms_after_a_cue'], '.0%', src,
                                          'share of them that start within 0.4 s after an audio cue')
    sp = s['spectrum']
    n['gyro_power_below_lowpass'] = E(sp['gyro_power_below_15hz'], '.1%', src, 'gyroscope power of the approach below 15 Hz')
    n['acc_power_below_lowpass'] = E(sp['acc_power_below_15hz'], '.1%', src, 'accelerometer power of the approach below 15 Hz')
    du = s['durations']
    n['duration_median_s'] = E(du['primary_median_s'], '.2f', src, 'oracle approach duration, median (s)')
    n['duration_share_le_2p5s'] = E(du['share_le_2p5s'], '.0%', src, 'approaches of at most 2.5 s')
    n['reaction_median_ms'] = E(du['reaction_median_ms'], '.0f', src, 'reaction time from GO to onset, median (ms)')
    for sp_name, v in du['by_speed_median_s'].items():
        n[f'duration_{_key(sp_name)}_s'] = E(v, '.2f', src, f'approach duration at {sp_name} speed, median (s)')
    sg = s['segmentation']
    n['offline_no_end_share_street'] = E(sg['offline_no_end_share_v2'], '.0%', src, 'street attempts where offline_v1 finds no end')
    n['touch_share_toward'] = E(sg['toward_trials_with_touch'], '.0%', src, 'screen-toward attempts that record a touch')
    n['touch_position_median'] = E(sg['touch_position_median'], '.0%', src, 'first touch, median position in the approach')
    n['touch_before_end_ms'] = E(sg['touch_before_end_ms_median'], '.0f', src, 'first touch before the end, median (ms)')
    ms = {r['channel']: r for r in s['mirror_symmetry']}
    if 'yaw_rate' in ms:
        y = ms['yaw_rate']
        n['mirror_rms_yaw_rate'] = E(y['rms_mirrored_left_vs_right'], '.1f', src, 'RMS mirrored-left vs right mean yaw rate (deg/s)')
        n['unmirrored_rms_yaw_rate'] = E(y['rms_left_vs_right'], '.1f', src, 'RMS left vs right mean yaw rate (deg/s)')
        n['mirror_persons_closer'] = E(y['persons_closer_to_mirror'], 'd', src, 'persons whose right mean is closer to the mirrored left')
        n['mirror_persons'] = E(y['persons'], 'd', src, 'persons with both hands (mirror check)')
    for r in s['sign_patterns']:
        n[f"sigma_star_rank_{_key(r['platform'])}"] = E(r['rank_sigma_star'], 'd', src,
                                                        f"rank of the conventions' mirror among 64 sign patterns, {r['platform']}")
    for r in s['platform_differences']:
        pl = _key(r['platform'])
        n[f'{pl}_duration_median_s'] = E(r['duration_median_s'], '.2f', src, f"approach duration, median, {r['platform']} (s)")
        n[f'{pl}_peak_gyro_median'] = E(r['peak_gyro_median_dps'], '.0f', src, f"peak |w|, median, {r['platform']} (deg/s)")
        n[f'{pl}_closure_median_deg'] = E(r['closure_median_deg'], '.1f', src, f"gyro-gravity closure, median, {r['platform']} (deg)")
    n['return_rule_acc_return_net'] = E(s['return_risk']['sign_rule_accuracy_on_return_net_yaw'], '.0%', src,
                                        'the sign rule on the net yaw of the return')
    n['return_auc_net'] = E(s['return_risk']['auc_return_net_yaw'], '.2f', src, 'net yaw AUC on the return')
    cfa = s['cuefree_v0']['all']
    n['cuefree_onset_abs_error_p90_ms'] = E(cfa['onset_abs_error_p90_ms'], '.0f', src,
                                            'cue-free v0: 90th percentile of the absolute onset error (ms)')
    st = s['stillness_before_onset']
    n['still_before_onset_median_s'] = E(st['median_s'], '.1f', src, 'stillness before the approach, median (s)')
    n['still_before_onset_share_1s'] = E(st['share_ge_1s'], '.0%', src, 'approaches after at least 1 s of stillness (gyro)')
    n['still_both_median_s'] = E(st['gyro_and_acc_median_s'], '.1f', src, 'the same with the accelerometer still too, median (s)')
    n['still_both_share_1s'] = E(st['gyro_and_acc_share_ge_1s'], '.0%', src, 'the same with the accelerometer still too, share >= 1 s')
    for setting, r in s['cuefree_v0_stress'].items():
        if not isinstance(r, dict):
            continue
        tag = (setting if setting in ('recorded', 'random')
               else f'keep{int(round(float(setting) * 1000))}ms')          # 'keep500ms' -> \DescribeStressKeepFiveHundredMs...
        n[f'stress_{tag}_share_approach'] = E(r['share_approach'], '.0%', src,
                                              f'cue-free v0 finds the approach first, stillness {setting}')
    gc = s['gravity_closure']
    n['closure_median_deg'] = E(gc['median_deg'], '.1f', src, 'gyro-gravity closure, median (deg)')
    n['closure_share_over_3deg'] = E(gc['share_over_3deg'], '.0%', src, 'closures above 3 deg')
    for c, r in s['flags'].items():
        if isinstance(r, dict) and 'primary_trials' in r:
            n[f'flag_{_key(c)}'] = E(r['primary_trials'], 'd', src, f'primary trials flagged {c}')
    for c, r in s['gates'].items():
        if isinstance(r, dict) and 'primary_trials_failing' in r:
            n[f'{_key(c)}_failing'] = E(r['primary_trials_failing'], 'd', src, f'primary trials failing {c}')
    lc = s['label_consistency']
    n['screen_loso_accuracy'] = E(lc['screen_loso_accuracy'], '.1%', src, 'D6 screen model, leave-one-person-out accuracy')
    n['screen_flagged_trials'] = E(lc['trials_flagged_by_d6_rules'], 'd', src, 'trials the D6 rules flag')
    n['label_blocks'] = E(lc['blocks'], 'd', src, 'planned blocks of the primary trials')
    n['swap_signature_blocks'] = E(len(lc['swap_signature_fhy']), 'd', src, 'blocks with the swap signature')
    nm = s['normalisation']
    n['rr_closure_median_deg'] = E(nm['rr_closure_median_deg'], '.1f', src, 'rotationRate mapping closure, median (deg)')
    n['rr_runner_up_median_deg'] = E(nm['rr_runner_up_median_deg'], '.1f', src, 'runner-up mapping closure, median (deg)')
    n['n_conditions'] = E(s['conditions']['n_conditions'], 'd', src, 'instructed conditions with primary trials')
    return n


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    RX.write_numbers(NUMBERS_JSON, describe_numbers())


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}


def main(argv: list[str] | None = None) -> None:
    """``python -m stage2.scripts.describe``: the 'describe' unit (compute when stale, figures, tables, numbers)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true', help='recompute even when the saved outputs are current')
    ap.add_argument('--rebuild-cache', action='store_true', help='rebuild the cache first (never while lanes run)')
    a = ap.parse_args(argv)
    if a.rebuild_cache or not (C.CACHE_DIR / 'arrays.npz').exists():
        build_cache()
    PL.run('describe', force=a.force or a.rebuild_cache)


if __name__ == '__main__':
    main()
