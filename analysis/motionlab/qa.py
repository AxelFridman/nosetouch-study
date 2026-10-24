"""Dataset quality assurance: who is in the data, what was recorded, and what looks wrong.

``qa_summary(dataset)`` returns a :class:`QAReport` with DataFrames (``tables``), headline numbers
(``summary``) and human-readable ``warnings``. It covers (docs/02 section 12, docs/05 section 1):
  * participants, sessions, attempts, accept rate, redo attempts, false starts, participant flags;
  * duplicates: dropped copies of the same record id, two trial ids with the same attempt key, several
    accepted attempts of one planned trial;
  * sampling: rate per platform, dt, gaps > 50 ms, hidden pages, missing gyroscope / acc, repeated
    samples (duplicated snapshots) and clipping candidates (|accg| > 3.5 g, |w| > 950 deg/s);
  * normalisation (norm_v2): calibration method, gravity-sign source per platform / engine (engine prior
    confirmed by the votes = 'engine_vote'), rotationRate mapping and its source (axis_check, the closure
    check of the trials, or assumed) per session with the closure numbers, and a gyro-gravity closure
    check on a sample of trials (a wrong axis mapping or sign shows up as large closure errors);
  * protocol 2.0.0 / schema 1.2: session status (ended_early) and review mode, choice trials (hand
    reported), the return of the arm, the optional streams and the touch contact details;
  * segmentation success: stored online_v1 found, this package's port agreeing with the stored indices
    (``check_online``), offline_v1 success and its timing differences (``offline``);
  * orientation mismatch: the app's ``detected.orientation_match`` and a recomputation from the baseline.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .conventions import GRAVITY, rotation_from_gravity
from .io import Dataset
from .preprocess import lowpass, repeated_sample_mask, resample_uniform
from .report import df_to_markdown, dict_to_markdown
from .segment import compare_with_stored, offline_v1_from_record, rerun_online_v1

CLIP_ACC_MS2 = 3.5 * GRAVITY
CLIP_GYRO_DPS = 950.0  # Android CDD only guarantees +-1000 deg/s


@dataclass
class QAReport:
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self, title: str = 'Dataset QA', figures: dict[str, str] | None = None) -> str:
        out = [f'# {title}\n']
        if self.warnings:
            out.append('## Attention\n')
            out += [f'- {w}' for w in self.warnings]
            out.append('')
        out.append('## Summary\n')
        out.append(dict_to_markdown(self.summary))
        for name, df in self.tables.items():
            out.append(f'## {name.replace("_", " ").capitalize()}\n')
            out.append(df_to_markdown(df, index=not isinstance(df.index, pd.RangeIndex)))
        if figures:
            out.append('## Figures\n')
            for cap, path in figures.items():
                out.append(f'### {cap}\n\n![{cap}]({path})\n')
        return '\n'.join(out)


def acc_accg_consistency(record: dict[str, Any], fs: float = 100.0) -> float:
    """Correlation between ``acc`` and the high-passed ``accg`` (all axes pooled). Should be ~+1; a strongly
    negative value means the platform reports the two fields with opposite sign conventions."""
    m = record['streams']['motion']
    t = np.asarray(m['t'], dtype=float)

    def col(k: str) -> np.ndarray:
        return np.asarray([np.nan if v is None else v for v in m[k]], dtype=float)

    A = np.column_stack([col(f'acc_{a}') for a in 'xyz'])
    G = np.column_stack([col(f'accg_{a}') for a in 'xyz'])
    if t.size < 10 or not np.isfinite(A).any():
        return math.nan
    tu, U = resample_uniform(t, np.column_stack([A, G]), fs, 100.0)
    A, G = U[:, :3], U[:, 3:]
    H = G - lowpass(G, fs, 0.5)
    ok = np.isfinite(A).all(axis=1) & np.isfinite(H).all(axis=1)
    a, h = A[ok].ravel(), H[ok].ravel()
    if a.size < 10 or a.std() == 0 or h.std() == 0:
        return math.nan
    return float(np.corrcoef(a, h)[0, 1])


def stream_checks(record: dict[str, Any], engine: str | None = None) -> dict[str, Any]:
    """Per-trial stream checks: repeated samples, clipping candidates, missing channels."""
    m = record['streams']['motion']

    def col(k: str) -> np.ndarray:
        v = m.get(k)
        if isinstance(v, np.ndarray):
            return v.astype(float, copy=False)
        return np.asarray([np.nan if x is None else x for x in (v or [])], dtype=float)

    A = np.column_stack([col(f'acc_{a}') for a in 'xyz'])
    G = np.column_stack([col(f'accg_{a}') for a in 'xyz'])
    W = np.column_stack([col(f'rr_{a}') for a in ('alpha', 'beta', 'gamma')])
    n = int(len(col('t')))
    if n == 0:
        return {'n_repeats': 0, 'n_clip_acc': 0, 'n_clip_gyro': 0, 'acc_missing_frac': math.nan, 'rr_missing_frac': math.nan}
    with np.errstate(invalid='ignore'):
        g_mag = np.linalg.norm(G, axis=1)
        w_mag = np.linalg.norm(W, axis=1)
    return {
        'n_repeats': int(repeated_sample_mask(A, G, W, engine=engine).sum()),
        'n_clip_acc': int(np.sum(g_mag > CLIP_ACC_MS2)),
        'n_clip_gyro': int(np.sum(w_mag > CLIP_GYRO_DPS)),
        'acc_missing_frac': float((~np.isfinite(A).all(axis=1)).mean()),
        'rr_missing_frac': float((~np.isfinite(W).all(axis=1)).mean()),
    }


def _crosstab(df: pd.DataFrame, r: str, c: str) -> pd.DataFrame:
    if not len(df) or r not in df.columns or c not in df.columns:
        return pd.DataFrame()
    t = pd.crosstab(df[r].astype(object).fillna('unknown'), df[c].astype(object).fillna('unknown'), margins=True, margins_name='total')
    t.index.name = r
    return t


def _bool(s: pd.Series) -> pd.Series:
    return s.astype('boolean').fillna(False).astype(bool)


def qa_summary(ds: Dataset, *, check_online: bool = True, offline: bool = True, consistency_sample: int = 200,
               gravity_sample: int = 200) -> QAReport:
    tr = ds.trials
    ss = ds.sessions
    pp = ds.participants
    warnings: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        'participants': int(len(pp)), 'sessions': int(len(ss)), 'trials (attempts)': int(len(tr)),
    }
    if not len(tr):
        warnings.append('no trials loaded')
        if len(ds.issues):
            tables['load_issues'] = ds.issues.groupby(['level', 'kind', 'message']).size().rename('count').reset_index()
        return QAReport(summary, tables, warnings)
    acc = _bool(tr['accepted'])
    summary.update({
        'accepted trials': int(acc.sum()), 'accept rate': float(acc.mean()),
        'simulated trials': int(_bool(tr['simulated']).sum()),
        'redo attempts': int((tr['attempt'] > 1).sum()),
        'false starts (total)': int(tr['false_starts'].fillna(0).sum()),
    })
    if summary['simulated trials']:
        warnings.append(f"{summary['simulated trials']} trials are SIMULATED (simulated: true) - never mix them into real-data results")

    # participants
    if len(pp):
        tables['participants_by_sex_and_handedness'] = _crosstab(pp, 'sex', 'handedness_self')
        cols = [c for c in ('participant_id', 'sex', 'age_years', 'handedness_self', 'ehi_class', 'ehi_lq', 'n_sessions', 'n_trials') if c in pp.columns]
        per = pp[cols].copy()
        plat = tr.groupby('participant_id')['platform'].agg(lambda s: ','.join(sorted({str(x) for x in s.dropna()})))
        nacc = tr[acc].groupby('participant_id').size()
        per['platform'] = per['participant_id'].map(plat)
        per['n_accepted'] = per['participant_id'].map(nacc).fillna(0).astype(int)
        tables['participants'] = per
        if pp['handedness_self'].isin(['left']).sum() < 3:
            warnings.append('fewer than 3 left-handed participants: left-handed error rates will be unreliable')

    # trials
    tables['trials_by_status'] = tr.groupby(['status', 'accepted'], dropna=False).size().rename('n').reset_index()
    a = tr[acc]
    tables['accepted_trials_orientation_x_hand'] = _crosstab(a, 'orientation', 'hand')
    tables['accepted_trials_speed_x_hand'] = _crosstab(a, 'speed', 'hand')
    tables['accepted_trials_start_position_x_hand'] = _crosstab(a, 'start_position', 'hand')
    tables['accepted_trials_variant_x_hand'] = _crosstab(a, 'variant', 'hand')
    tables['accepted_trials_platform_x_hand'] = _crosstab(a, 'platform', 'hand')
    flags = tr['participant_flag'].dropna()
    if len(flags):
        tables['participant_flags'] = flags.value_counts().rename('n').to_frame()
        summary['participant-flagged trials'] = int(len(flags))
    att = tr.groupby(['session_id', 'trial_index'])['attempt'].max()
    summary['planned trials with a redo'] = int((att > 1).sum())

    # duplicates (docs/02 section 10)
    dups = ds.duplicates
    summary['duplicate record copies dropped'] = int(len(dups))
    dup_rows = []
    if len(dups):
        for kind, g in dups.groupby('kind'):
            dup_rows.append({'what': f'{kind} copies dropped (same id)', 'n': int(len(g)),
                             'distinct ids': int(g['record_id'].nunique())})
    n_key = int(_bool(tr['dup_attempt_key']).sum()) if 'dup_attempt_key' in tr.columns else 0
    n_multi = int(_bool(tr['multi_accepted']).sum()) if 'multi_accepted' in tr.columns else 0
    dup_rows.append({'what': 'trials sharing (session, trial_index, attempt) with another trial id', 'n': n_key,
                     'distinct ids': n_key})
    dup_rows.append({'what': 'attempts of planned trials with several accepted attempts', 'n': n_multi,
                     'distinct ids': n_multi})
    tables['duplicates'] = pd.DataFrame(dup_rows)
    if n_key:
        warnings.append(f'{n_key} trials share an attempt key with another trial id (same session resumed on two app copies?)')
    if n_multi:
        warnings.append(f'{n_multi} attempts belong to planned trials with more than one accepted attempt')

    # sampling and streams
    samp = tr.groupby(tr['platform'].astype(object).fillna('unknown')).agg(
        trials=('trial_id', 'size'),
        motion_hz_median=('motion_hz', 'median'),
        motion_hz_p05=('motion_hz', lambda s: float(np.nanpercentile(s.astype(float), 5)) if s.notna().any() else math.nan),
        motion_hz_p95=('motion_hz', lambda s: float(np.nanpercentile(s.astype(float), 95)) if s.notna().any() else math.nan),
        dt_median_ms=('motion_dt_ms_median', 'median'), dt_max_ms_median=('motion_dt_ms_max', 'median'),
        trials_with_gaps=('gaps_over_50ms', lambda s: int((s.fillna(0) > 0).sum())),
        gaps_over_50ms=('gaps_over_50ms', 'sum'),
        acc_available=('acc_available', lambda s: float(_bool(s).mean())),
        rotation_available=('rotation_available', lambda s: float(_bool(s).mean())),
        quaternion_stream=('has_quaternion_stream', 'mean'),
        hidden_during=('hidden_during', lambda s: int(_bool(s).sum())),
    )
    samp.index.name = 'platform'
    summary['motion Hz (median)'] = float(tr['motion_hz'].median()) if tr['motion_hz'].notna().any() else math.nan
    low = tr['motion_hz'] < 50
    if low.any():
        warnings.append(f'{int(low.sum())} trials sampled below 50 Hz (exclusion rule T5: a degraded sensor path)')
    if (tr['gaps_over_50ms'].fillna(0) > 0).mean() > 0.1:
        warnings.append('more than 10% of trials have sampling gaps > 50 ms (exclusion rule T6)')
    if (~_bool(tr['rotation_available'])).any():
        warnings.append(f"{int((~_bool(tr['rotation_available'])).sum())} trials without rotation rate: chiral features are NaN for them")
    if _bool(tr['hidden_during']).any():
        warnings.append(f"{int(_bool(tr['hidden_during']).sum())} trials with the page hidden during recording (rule T4)")
    eng = dict(zip(ss['session_id'], ss['engine_effective'])) if len(ss) and 'engine_effective' in ss.columns else {}
    srows = []
    for tid, sid in zip(tr['trial_id'], tr['session_id']):
        c = stream_checks(ds.record(tid, compact=True), eng.get(sid))
        srows.append({'trial_id': tid, **c})
    sc = pd.DataFrame(srows).set_index('trial_id')
    plat_of = tr.set_index('trial_id')['platform'].astype(object).fillna('unknown')
    st_tab = sc.groupby(plat_of.reindex(sc.index)).agg(
        repeated_samples=('n_repeats', 'sum'), trials_with_repeats=('n_repeats', lambda s: int((s > 0).sum())),
        clip_acc_samples=('n_clip_acc', 'sum'), clip_gyro_samples=('n_clip_gyro', 'sum'),
        acc_missing_frac=('acc_missing_frac', 'mean'), rr_missing_frac=('rr_missing_frac', 'mean'))
    st_tab.index.name = 'platform'
    tables['sampling'] = samp.join(st_tab)
    n_clip = int(((sc['n_clip_acc'] > 0) | (sc['n_clip_gyro'] > 0)).sum())
    summary['trials with clipping candidates'] = n_clip
    if n_clip:
        warnings.append(f'{n_clip} trials reach |accg| > 3.5 g or |w| > 950 deg/s (possible sensor clipping, rule T10)')

    # normalisation (norm_v2)
    if len(ss):
        keys = [c for c in ('platform', 'engine_effective', 'calibration_method', 'gravity_sign_source', 'gravity_sign')
                if c in ss.columns]
        tables['gravity_sign'] = ss.assign(**{c: ss[c].astype(object).fillna('unknown') for c in keys}).groupby(keys).size().rename('sessions').reset_index()
        _normalisation_summary(ss, summary, tables, warnings)
        nw = int(_bool(ss['gravity_sign_warning'].astype('boolean').fillna(True)).sum()) if 'gravity_sign_warning' in ss.columns else 0
        summary['sessions with uncertain gravity sign'] = nw
        if nw:
            warnings.append(f'{nw} session(s) without an undisputed calibrated gravity sign (chiral features depend on it!)')
        nd = int((ss['gravity_sign_source'] == 'default').sum()) if 'gravity_sign_source' in ss.columns else 0
        if nd:
            warnings.append(f'{nd} session(s) with NO support for the gravity sign (default +1): excluded from chiral modelling')
        if {'calibration_gravity_sign', 'gravity_sign_source'} <= set(ss.columns):
            # a calibrated sign that ended up 'inferred' was overridden by the trial votes
            over = ss['calibration_gravity_sign'].isin([1, -1]) & (ss['gravity_sign_source'] == 'inferred')
            summary['calibrated gravity signs overridden by the trial votes'] = int(over.sum())
            if over.any():
                ids = ', '.join(str(x) for x in ss.loc[over, 'session_id'].head(5))
                warnings.append(f'{int(over.sum())} session(s) whose calibrated gravity sign the trial votes clearly contradict '
                                f'(flat step done face-down?): the vote sign is used, check them by hand ({ids})')
        if 'rr_mapping' in ss.columns:
            ax = ss.assign(axis_check_verdict=ss['axis_check_verdict'].astype(object).fillna('absent'),
                           rr_units=ss['rr_units'].astype(object).fillna('unknown'),
                           rr_source=(ss['rr_source'] if 'rr_source' in ss.columns else pd.Series('assumed', index=ss.index)
                                      ).astype(object).fillna('unknown'))
            tables['rotation_rate_mapping'] = ax.groupby(['platform', 'axis_check_verdict', 'rr_source', 'rr_mapping',
                                                          'rr_units'], dropna=False).size().rename('sessions').reset_index()
            nu = int((~_bool(ss['rr_verified'])).sum())
            summary['sessions with unverified rotationRate mapping'] = nu
            nr = int((ss['rr_mapping'] != 'w3c').sum())
            if nr:
                warnings.append(f'{nr} session(s) remapped from a non-W3C rotationRate verdict (check the axis_check details)')
    sample = tr.loc[acc, 'trial_id']
    sample = sample.iloc[:: max(1, len(sample) // max(1, consistency_sample))]
    cons = [acc_accg_consistency(ds.record(t, compact=True)) for t in sample]
    cons = [c for c in cons if math.isfinite(c)]
    if cons:
        summary['acc vs high-passed accg correlation (median)'] = float(np.median(cons))
        neg = sum(1 for c in cons if c < -0.3)
        if neg:
            warnings.append(f'{neg} sampled trial(s) where acc and accg disagree in sign (platform convention problem)')
    if gravity_sample:
        tables_g, warn_g = _gravity_check(ds, tr.loc[acc], gravity_sample)
        if tables_g is not None:
            tables['gravity_closure'] = tables_g
        warnings += warn_g

    # segmentation
    seg_ok = tr['onset_index'].notna() & tr['end_index'].notna()
    summary['stored segmentation found (accepted trials)'] = float(seg_ok[acc].mean()) if acc.any() else math.nan
    seg_tab = tr.assign(segmented=seg_ok).groupby('status').agg(n=('trial_id', 'size'), segmented=('segmented', 'mean'))
    rows = []
    if check_online or offline:
        for tid in tr['trial_id']:
            rec = ds.record(tid, compact=True)
            row: dict[str, Any] = {'trial_id': tid}
            if check_online:
                c = compare_with_stored(rec, rerun_online_v1(rec, keep_trace=False))
                row.update(port_onset_match=c['onset_index_match'], port_end_match=c['end_index_match'],
                           port_all_match=c['all_match'])
            if offline:
                o = offline_v1_from_record(rec)
                row.update(offline_ok=o.ok, offline_reason=o.reason, offline_duration_ms=o.duration_ms,
                           offline_onset_ms=o.onset_ms, offline_end_ms=o.end_ms)
            rows.append(row)
        chk = pd.DataFrame(rows).set_index('trial_id')
        tj = tr.set_index('trial_id').join(chk)
        if check_online:
            agree = float(tj['port_all_match'].mean())
            summary['online_v1 port == stored (all trials)'] = agree
            seg_tab['port_agreement'] = tj.groupby('status')['port_all_match'].mean()
            if agree < 1.0:
                bad = tj.index[~tj['port_all_match'].astype(bool)]
                warnings.append(f'online_v1 port disagrees with the app on {len(bad)} trial(s) (e.g. {", ".join(map(str, bad[:3]))}); '
                                'see segment.compare_segmentations')
        if offline:
            acc_np = acc.to_numpy()
            summary['offline_v1 success (accepted trials)'] = float(tj.loc[acc_np, 'offline_ok'].mean()) if acc.any() else math.nan
            seg_tab['offline_ok'] = tj.groupby('status')['offline_ok'].mean()
            d_on = (tj['offline_onset_ms'] - tj['onset_ms']).astype(float)
            d_en = (tj['offline_end_ms'] - tj['end_ms']).astype(float)
            by_speed = tj[acc_np].assign(d_on=d_on, d_en=d_en).groupby('speed').agg(
                n=('status', 'size'), online_movement_ms=('movement_ms', 'median'),
                offline_duration_ms=('offline_duration_ms', 'median'),
                offline_minus_online_onset_ms=('d_on', 'median'), offline_minus_online_end_ms=('d_en', 'median'))
            tables['movement_duration_by_speed'] = by_speed
            big = (d_on.abs() > 100) | (d_en.abs() > 100)
            summary['accepted trials where offline and online differ by > 100 ms'] = int(big[acc_np].sum())
            fails = tj.loc[acc_np & ~tj['offline_ok'].fillna(False).astype(bool), 'offline_reason'].value_counts()
            if len(fails):
                tables['offline_failures_accepted_trials'] = fails.rename('n').to_frame()
    tables['segmentation_by_status'] = seg_tab

    # orientation checks
    om_all = tr.loc[acc, 'orientation_match']
    summary['orientation mismatches (app, accepted trials)'] = int((om_all.astype('boolean') == False).sum())  # noqa: E712
    o_rows = []
    for o, g in tr[acc].groupby('orientation'):
        rec_match = []
        for _, r in g.iterrows():
            bg = (r['baseline_gravity_x'], r['baseline_gravity_y'], r['baseline_gravity_z'])
            inst = r['instructed_rotation']
            if inst is None or (isinstance(inst, float) and math.isnan(inst)) or any(pd.isna(v) for v in bg):
                continue
            det = rotation_from_gravity(np.asarray(bg, dtype=float), int(r['gravity_sign']), r['instructed_screen'])
            rec_match.append(det == inst)
        om = g['orientation_match'].dropna()
        o_rows.append({'orientation': o, 'n': len(g),
                       'app_mismatches': int((om.astype(bool) == False).sum()),  # noqa: E712
                       'app_match_rate': float(om.astype(bool).mean()) if len(om) else math.nan,
                       'recomputed_match_rate': float(np.mean(rec_match)) if rec_match else math.nan})
    if o_rows:
        ot = pd.DataFrame(o_rows).set_index('orientation')
        tables['orientation_check'] = ot
        badr = ot[(ot['recomputed_match_rate'] < 0.8)]
        for o in badr.index:
            warnings.append(f'orientation {o}: only {badr.loc[o, "recomputed_match_rate"]:.0%} of baselines match the instructed rotation')

    _protocol_v2_summary(tr, ss, summary, tables)

    # warnings & issues
    wc = Counter(w for ws in tr['warnings'].fillna('') for w in str(ws).split(';') if w)
    if wc:
        tables['quality_warnings'] = pd.DataFrame(sorted(wc.items(), key=lambda kv: -kv[1]), columns=['warning', 'trials'])
    if len(ds.issues):
        tables['load_issues'] = ds.issues.groupby(['level', 'kind', 'message']).size().rename('count').reset_index()
        ne = int((ds.issues['level'] == 'error').sum())
        summary['load errors'] = ne
        if ne:
            warnings.append(f'{ne} record(s)/file(s) failed validation or parsing and were skipped (see load issues)')
    return QAReport(summary, tables, warnings)


#: Per-session normalisation columns of the 'normalisation' table (the headline numbers of norm_v2).
_NORM_COLUMNS = ('session_id', 'platform', 'engine_effective', 'calibration_method', 'gravity_sign', 'gravity_sign_source',
                 'gravity_votes_for', 'gravity_votes_against', 'rr_mapping', 'rr_units', 'rr_source', 'rr_verified',
                 'rr_closure_best', 'rr_closure_deg', 'rr_closure_runner_up', 'rr_closure_runner_up_deg',
                 'rr_closure_trials', 'rr_closure_verified', 'gyro_bias_subtracted', 'norm_flags')


def _counts_text(s: pd.Series) -> str:
    vc = s.astype(object).fillna('unknown').value_counts()
    return ', '.join(f'{k} {int(v)}' for k, v in vc.items())


def _normalisation_summary(ss: pd.DataFrame, summary: dict[str, Any], tables: dict[str, pd.DataFrame],
                           warnings: list[str]) -> None:
    """Per-session norm_v2 table and headline numbers: calibration method, gravity-sign source, mapping
    source and the closure check (``conventions.infer_rate_mapping``)."""
    cols = [c for c in _NORM_COLUMNS if c in ss.columns]
    tables['normalisation'] = ss[cols].reset_index(drop=True)
    if 'calibration_method' in ss.columns:
        summary['calibration methods'] = _counts_text(ss['calibration_method'])
    if 'gravity_sign_source' in ss.columns:
        summary['gravity sign sources'] = _counts_text(ss['gravity_sign_source'])
    if 'rr_source' in ss.columns:
        summary['rotationRate mapping sources'] = _counts_text(ss['rr_source'].astype(object) + ' ' +
                                                               ss['rr_mapping'].astype(object) + '/' +
                                                               ss['rr_units'].astype(object))
    if 'rr_closure_deg' in ss.columns:
        ran = ss['rr_closure_deg'].notna()
        summary['sessions with a rotationRate closure check'] = int(ran.sum())
        if ran.any():
            v = ss.loc[ran]
            summary['closure check: best / runner-up (median over sessions, deg)'] = (
                f"{float(v['rr_closure_deg'].median()):.2f} / {float(v['rr_closure_runner_up_deg'].median()):.2f}")
            summary['closure check: decisive sessions'] = int(v['rr_closure_verified'].astype('boolean').fillna(False).sum())
    if 'rr_closure_disagrees' in ss.columns:
        bad = ss['rr_closure_disagrees'].astype('boolean').fillna(False).astype(bool)
        if bad.any():
            ids = ', '.join(str(x) for x in ss.loc[bad, 'session_id'].head(5))
            warnings.append(f'{int(bad.sum())} session(s) where the closure check of the trials names another '
                            f'rotationRate mapping than the calibration axis_check ({ids}); the axis_check is used')
    if 'gyro_bias_subtracted' in ss.columns:
        summary['sessions with a trusted calibration gyro bias'] = int(ss['gyro_bias_subtracted'].astype('boolean').fillna(False).sum())


def _protocol_v2_summary(tr: pd.DataFrame, ss: pd.DataFrame, summary: dict[str, Any],
                         tables: dict[str, pd.DataFrame]) -> None:
    """Protocol 2.0.0 / schema 1.2 fields: session status and review mode, choice trials, the return of the
    arm, the optional streams and the touch details (absent in older records: reported as such)."""
    if len(ss) and 'status' in ss.columns:
        keys = [c for c in ('preset', 'status', 'review_mode') if c in ss.columns]
        tables['sessions_by_status'] = ss.assign(**{c: ss[c].astype(object).fillna('unknown') for c in keys}).groupby(
            keys).size().rename('sessions').reset_index()
        summary['sessions ended early'] = int((ss['status'] == 'ended_early').sum())
    if 'hand_source' in tr.columns:
        rep = tr['hand_source'] == 'reported'
        summary['choice trials (hand reported)'] = int(rep.sum())
        if rep.any():
            summary['choice trials accepted'] = int(_bool(tr.loc[rep, 'accepted']).sum())
    v2_cols = ('return_observed', 'has_quaternion_abs_stream', 'has_orientation_abs_stream', 'has_magnetometer_stream')
    has_return = tr['return_observed'].notna() if 'return_observed' in tr.columns else pd.Series(False, index=tr.index)
    v2_present = bool(has_return.any()) or any(_bool(tr[c]).any() for c in v2_cols[1:] if c in tr.columns) or (
        'touch_radius_max' in tr.columns and tr['touch_radius_max'].notna().any())
    summary['schema 1.2 trial fields present'] = v2_present
    if not v2_present:
        return
    plat = tr['platform'].astype(object).fillna('unknown')
    rows = []
    for p, g in tr.groupby(plat):
        hr = g['return_observed'].notna()
        ro = g.loc[hr, 'return_observed'].astype('boolean').fillna(False).astype(bool)
        starts = pd.to_numeric(g['n_touch_starts'], errors='coerce').fillna(0) > 0
        rows.append({
            'platform': p, 'trials': int(len(g)),
            'return_recorded_trials': int(hr.sum()),
            'return_observed_rate': float(ro.mean()) if len(ro) else math.nan,
            'return_delay_ms_median': float(pd.to_numeric(g['return_delay_ms'], errors='coerce').median()),
            'return_duration_ms_median': float(pd.to_numeric(g['return_duration_ms'], errors='coerce').median()),
            'post_end_ms_median': float(pd.to_numeric(g['post_end_ms'], errors='coerce').median()),
            'quaternion_abs_rate': float(_bool(g['has_quaternion_abs_stream']).mean()),
            'orientation_abs_rate': float(_bool(g['has_orientation_abs_stream']).mean()),
            'magnetometer_rate': float(_bool(g['has_magnetometer_stream']).mean()),
            'trials_with_screen_events': int((pd.to_numeric(g['n_screen_events'], errors='coerce').fillna(0) > 0).sum()),
            'touch_radius_rate': float(g.loc[starts, 'touch_radius_max'].notna().mean()) if starts.any() else math.nan,
            'touch_force_positive_rate': float((pd.to_numeric(g.loc[starts, 'touch_force_max'], errors='coerce') > 0).mean())
            if starts.any() else math.nan,
        })
    tables['schema_1_2_fields'] = pd.DataFrame(rows).set_index('platform')


def _gravity_check(ds: Dataset, trials: pd.DataFrame, n: int) -> tuple[pd.DataFrame | None, list[str]]:
    """gravity_v1 closure on a sample of accepted trials, per platform (needs features.prepare_trial)."""
    from .features import prepare_trial

    if not len(trials):
        return None, []
    ids = trials['trial_id'].iloc[:: max(1, len(trials) // max(1, n))]
    rows = []
    for tid in ids:
        pt = prepare_trial(ds.record(tid, compact=True), norm=ds.trial_normalization(tid))
        if pt.ok and pt.gravity_method == 'gyro_fwd_bwd':
            rows.append({'trial_id': tid, 'closure_deg': pt.closure_deg, 'vs_browser_deg': pt.browser_angle_mean_deg})
    if not rows:
        return None, []
    df = pd.DataFrame(rows).set_index('trial_id')
    plat = trials.set_index('trial_id')['platform'].astype(object).fillna('unknown')
    tab = df.groupby(plat.reindex(df.index)).agg(
        trials=('closure_deg', 'size'), closure_median_deg=('closure_deg', 'median'),
        closure_p90_deg=('closure_deg', lambda s: float(np.nanpercentile(s, 90))),
        over_3deg=('closure_deg', lambda s: float((s > 3.0).mean())),
        vs_browser_median_deg=('vs_browser_deg', 'median'))
    tab.index.name = 'platform'
    warn = []
    for p, r in tab.iterrows():
        if r['closure_median_deg'] > 3.0:
            warn.append(f'platform {p}: median gyro-gravity closure {r["closure_median_deg"]:.1f} deg (> 3): suspect the '
                        'rotationRate mapping / units, a gyro bias, or rest windows that are not still')
    return tab, warn
