"""Lane x1 compute: the exploratory case study of one participant with Parkinson's disease (PD01) - descriptive only.

PD01 is never trained on, validated on, tuned with or scored by any model: this module fits nothing and computes no
prediction. It (1) builds the exploratory person's own cache with the main cache's code path (``x1_data``), (2)
computes every per-trial characteristic (``x1_features``) for the modelling persons and for PD01, (3) aggregates them
per person, (4) compares PD01 with the modelling persons in matched strata (``x1_stats``) and (5) saves everything the
figures, tables, numbers and summary need:

    tables/x1/x1_trial_characteristics.csv   one row per trial (trial keys, conditions, characteristics)
    tables/x1/x1_person_values.csv           person-level values per characteristic, stratum and variant
    tables/x1/x1_comparisons.csv             every comparison (statistics, sensitivity analyses, q-values)
    results/x1_characteristics.json          the main table (one row per characteristic, robustness), the null
                                             reference of the "beyond all" count, the redundancy of the findings
    tables/x1/x1_checks.json                 targeted checks of alternative explanations and the spectral leakage
    tables/x1/x1_spectra.csv                 gyroscope spectra (person means, the case's trials)
    tables/x1/x1_traces.csv, x1_traces.json  example recordings (canonical channels) and their events
    tables/x1/x1_yaw_curves.csv              mirror-normalised yaw courses of the approach
    tables/x1/x1_task.json                   attempts, rejections, sessions (length, protocol, mapping), clip flags
    results/x1_multivariate.json             PCA map, Mahalanobis test and its leave-one-characteristic-out check
    tables/x1/x1_manifest.json               what the outputs were made from (staleness)

Every derived file carries the provenance of the main cache (``pipeline.provenance``) plus the exploratory cache's
data identity and a hash of the x1 compute code; ``compute`` skips the analyses whose outputs are current.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2.pipeline import Lane, provenance, selected
from stage2.lanes import x1_data as XD
from stage2.lanes import x1_features as XF
from stage2.lanes import x1_stats as XS

L = Lane('x1')

#: The exploratory person this lane presents (one clearly named place; must be listed in curation/exploratory.json).
CASE = 'PD01'
#: The analyses of the compute step, in order (``--only`` selects among them).
ANALYSES = ('x1_cache', 'x1_characteristics', 'x1_checks', 'x1_spectra', 'x1_traces', 'x1_yaw_curves',
            'x1_multivariate', 'x1_task')
#: Spectra are interpolated onto one grid so trials of different lengths can be averaged.
SPECTRUM_GRID_HZ = np.round(np.arange(0.0, 15.0 + 1e-9, 0.25), 2)
#: Each window's spectra come from the stratum its characteristics are judged in.
SPECTRUM_STRATA = {'pre': 'left_upright_all', 'approach': 'left_upright_natural', 'nose': 'left_upright_all'}
#: The typical trials of the example-trace figure: closest to the median on these characteristics.
TYPICAL_KEYS = ('duration_s', 'peak_speed_mv', 'app_rms_7_12')
TRACE_PAD_S = (1.0, 1.5)              # traces from onset - 1.0 s to end + 1.5 s
#: The tone of the spectral-leakage check: just below the 7-Hz band edge, in each short window.
LEAKAGE_TONE_HZ = 6.5
#: Two characteristics are "redundant" (one finding) when their person-level Spearman |rho| reaches this.
REDUNDANT_RHO = 0.7

CODE_FILES = ('x1_data.py', 'x1_features.py', 'x1_stats.py', 'x1_compute.py')

# ───────────────────────────── provenance and staleness ─────────────────────────────


def code_sha256() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update(hashlib.sha256((C.LANES_DIR / name).read_bytes()).digest())
    return h.hexdigest()


def x1_provenance() -> dict[str, Any]:
    """The main cache's provenance, the exploratory cache's data identity and the x1 code hash."""
    ident = XD.identity(CASE)
    return {'main': provenance(), 'exploratory_case': CASE, 'exploratory_data_id': ident['data_id'],
            'x1_code_sha256': code_sha256(), 'params': json.loads(json.dumps(XF.PARAMS, default=str))}


def manifest_path() -> Path:
    return L.tab_dir / 'x1_manifest.json'


def current(name: str, prov: dict[str, Any]) -> bool:
    """True when analysis ``name`` was made from the same caches, code and parameters."""
    try:
        man = json.loads(manifest_path().read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return man.get(name) == prov


def mark_done(name: str, prov: dict[str, Any]) -> None:
    p = manifest_path()
    try:
        man = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        man = {}
    man[name] = prov
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(sorted(man.items())), indent=1) + '\n', encoding='utf-8', newline='\n')


def json_clean(obj: Any) -> Any:
    """NaN / inf -> None and numpy scalars -> Python (strict JSON, as ``Lane.write_json`` requires)."""
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def write_json(name: str, obj: dict[str, Any], prov: dict[str, Any], folder: str = 'tables') -> Path:
    """``Lane.write_json`` (main-cache provenance) plus the exploratory provenance."""
    return L.write_json(name, json_clean({'x1_provenance': prov, **obj}), folder=folder)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    """A derived CSV in tables/x1/ (privacy-checked by report_safe; no figure or LaTeX table)."""
    return L.export_table(df, name, tex=False)['csv']


def read_trials() -> pd.DataFrame:
    t = pd.read_csv(L.tab_dir / 'x1_trial_characteristics.csv')
    t['person_id'] = t['person_id'].astype(str)
    return t


# ───────────────────────────── data ─────────────────────────────

TRIAL_COLS = ('person_id', 'trial_key', 'session_label', 'analysis_set', 'hand', 'hand_source', 'orientation', 'speed',
              'variant', 'start_position', 'posture', 'platform', 'protocol_version', 'clip_candidate',
              'session_order', 'trial_index', 'attempt', 'tier')


def load_inputs() -> dict[str, Any]:
    """Reference (main cache) and case (exploratory cache) data: every accepted analysable trial and every attempt."""
    from stage2.data import load_cache, load_tables
    XD.ensure_cache(CASE)
    ref = load_cache(XS.SETS_ALL)
    case = XD.load(CASE, XS.SETS_ALL)
    if set(ref.trials['person_id'].astype(str)) & set(case.trials['person_id'].astype(str)):
        raise RuntimeError('the exploratory person appears among the modelling persons')
    rt, ct = load_tables(), XD.load_tables(CASE)
    return {'ref': ref, 'case': case, 'ref_attempts': rt['trials'], 'case_attempts': ct['trials'],
            'ref_persons': rt['persons'], 'case_persons': ct['persons'], 'ref_sessions': rt['sessions'],
            'case_sessions': ct['sessions']}


def characteristics_table(D: Any, group: str) -> pd.DataFrame:
    """One row per trial of a Stage2Data: conditions and every per-trial characteristic."""
    rows = []
    for i in range(D.n):
        r = D.trials.iloc[i]
        c = XF.trial_characteristics(D.stream(i), r.to_dict(), D.approach[i], D.channels, D.fixed[i], D.fixed_mask[i])
        rows.append({'group': group, **{k: r.get(k) for k in TRIAL_COLS}, **c})
    df = pd.DataFrame(rows)
    df['person_id'] = df['person_id'].astype(str)
    df['clip_candidate'] = df['clip_candidate'].astype('boolean').fillna(False).astype(bool)
    return df


def all_characteristics(inp: dict[str, Any]) -> pd.DataFrame:
    t = pd.concat([characteristics_table(inp['ref'], 'reference'), characteristics_table(inp['case'], 'exploratory')],
                  ignore_index=True)
    for c in ('session_order', 'trial_index', 'attempt'):
        t[c] = pd.to_numeric(t[c], errors='coerce')
    bad = t['g_mag_dev'].dropna()
    if len(bad) and float(bad.max()) > 1e-3:
        raise RuntimeError(f'the calibrated |g| could not be recovered for {int((bad > 1e-3).sum())} trial(s)')
    return t


def person_attribute(t: pd.DataFrame, col: str) -> pd.Series:
    """The most frequent value of a column per person (posture and platform are per session)."""
    return t.groupby('person_id')[col].agg(lambda s: s.dropna().astype(str).mode().iloc[0] if s.notna().any() else '')


# ───────────────────────────── comparisons ─────────────────────────────


def _mode(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    return s.mode().iloc[0] if len(s) else ''


def variant_tables(t: pd.DataFrame, attempts: pd.DataFrame, var: str, case: dict[str, str]
                   ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(trials, attempts) of a sensitivity analysis: the reference restricted to the case's protocol version or
    posture, or every person's clip-flagged trials dropped (attempts: not applicable, None)."""
    ref = t['group'] == 'reference'
    aref = attempts['person_id'].astype(str) != CASE
    if var == 'same_protocol':
        return (t[~ref | (t['protocol_version'].astype(str) == case['protocol_version'])],
                attempts[~aref | (attempts['protocol_version'].astype(str) == case['protocol_version'])])
    if var == 'same_posture':
        return (t[~ref | (t['posture'].astype(str) == case['posture'])],
                attempts[~aref | (attempts['posture'].astype(str) == case['posture'])])
    if var == 'no_clip':
        return t[~t['clip_candidate'].astype(bool)], None
    return t, attempts


def _values(t: pd.DataFrame, key: str, stratum: str, *, first_k: int | None, k_repeats: int,
            attempts: pd.DataFrame | None) -> pd.DataFrame:
    c = XS.BY_KEY[key]
    empty = pd.DataFrame(columns=['person_id', 'value', 'n_trials'])
    if c.level == 'trial':
        return XS.person_trial_values(t, key, stratum, first_k=first_k)
    if first_k is not None or (key == 'attempts_per_accepted' and attempts is None):
        return empty
    return XS.person_level_values(t, key, k_repeats=k_repeats, attempts=attempts, stratum=stratum)


def comparisons(t: pd.DataFrame, attempts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(long comparison table, person values). Rows: characteristic x stratum x variant (``x1_stats.VARIANTS``; the
    variants only in the characteristic's own stratum). ``n_ref_changed`` / ``case_changed``: whether a variant
    recomputed values (it can then overturn a verdict) or only dropped persons."""
    case_t = t[t['group'] == 'exploratory']
    case_attr = {'protocol_version': _mode(case_t['protocol_version']), 'posture': _mode(case_t['posture'])}
    platform = person_attribute(t, 'platform')
    posture = person_attribute(t, 'posture')
    case_platform = platform.get(CASE, '')
    tables = {v: variant_tables(t, attempts, v, case_attr) for v in XS.VARIANTS}
    rows, pv = [], []
    for c in XS.CHARACTERISTICS:
        strata = [c.stratum]
        if c.level == 'trial':
            strata += [s for s in XS.REPORT_STRATA if s != c.stratum]
        if c.key in XS.REPEAT_KEYS:
            strata.append(XS.REPEAT_SECONDARY)
        for s in strata:
            n_case = int(XS.STRATA[s].select(case_t).sum())
            main_vals = None
            for var in (XS.VARIANTS if s == c.stratum else ('main',)):
                tt, aa = tables[var]
                k = n_case if var == 'first_k' else None
                if var == 'first_k' and c.level != 'trial':
                    continue
                vals = _values(tt, c.key, s, first_k=k, k_repeats=n_case, attempts=aa)
                if vals.empty:
                    continue
                vals = vals.assign(group=np.where(vals['person_id'] == CASE, 'exploratory', 'reference'))
                ref = vals[(vals['group'] == 'reference') & vals['value'].notna()]
                if var == 'same_platform':
                    ref = ref[ref['person_id'].map(platform) == case_platform]
                cv = vals.loc[vals['person_id'] == CASE, 'value']
                x = float(cv.iloc[0]) if len(cv) and pd.notna(cv.iloc[0]) else math.nan
                trials = None
                if c.level == 'trial':
                    sub = tt[(tt['group'] == 'exploratory') & XS.STRATA[s].select(tt)]
                    trials = pd.to_numeric(sub[c.key], errors='coerce').dropna().tolist()
                st = XS.compare(ref['value'].tolist(), x, c.scale, trials)
                if var == 'main':
                    main_vals = vals
                    n_changed, case_changed = 0, False
                else:
                    a = main_vals.set_index('person_id')
                    b = vals.set_index('person_id')
                    common = [p for p in ref['person_id'] if p in a.index]
                    dv = (a.loc[common, 'value'] - b.loc[common, 'value']).abs()
                    dn = a.loc[common, 'n_trials'] != b.loc[common, 'n_trials']
                    n_changed = int(((dv > 1e-9 * (1 + a.loc[common, 'value'].abs())) | dn).sum())
                    xm = main_vals.loc[main_vals['person_id'] == CASE, 'value']
                    x0 = float(xm.iloc[0]) if len(xm) and pd.notna(xm.iloc[0]) else math.nan
                    case_changed = not (math.isnan(x) and math.isnan(x0)) and not (abs(x - x0) <= 1e-9 * (1 + abs(x0)))
                n_case_trials = vals.loc[vals['person_id'] == CASE, 'n_trials']
                rows.append({'key': c.key, 'label': c.label, 'unit': c.unit, 'group_name': c.group,
                             'phenomenon': c.phenomenon, 'stratum': s, 'primary': s == c.stratum, 'variant': var,
                             'level': c.level, 'post_hoc': c.post_hoc,
                             'n_case_trials': int(n_case_trials.iloc[0]) if len(n_case_trials) else 0,
                             'n_case_stratum': n_case, 'n_ref_changed': n_changed, 'case_changed': case_changed, **st})
                if var == 'main' or s == c.stratum:
                    pv.append(vals.assign(key=c.key, stratum=s, variant=var,
                                          posture=vals['person_id'].map(posture),
                                          platform=vals['person_id'].map(platform)))
    df = pd.DataFrame(rows)
    # Benjamini-Hochberg over the PRE-SPECIFIED characteristics only (post hoc ones carry no q-value): the primary
    # family (one comparison per characteristic, its matched stratum, main variant), and one family per secondary
    # stratum
    df['q_rank'] = np.nan
    df['q_ch'] = np.nan
    main = (df['variant'] == 'main') & ~df['post_hoc'].astype(bool)
    fam = main & df['primary']
    df.loc[fam, 'q_rank'] = XS.bh(df.loc[fam, 'p_rank'])
    df.loc[fam, 'q_ch'] = XS.bh(df.loc[fam, 'ch_p'])
    for s in (*XS.REPORT_STRATA, XS.REPEAT_SECONDARY):
        m = main & ~df['primary'] & (df['stratum'] == s)
        if m.any():
            df.loc[m, 'q_rank'] = XS.bh(df.loc[m, 'p_rank'])
            df.loc[m, 'q_ch'] = XS.bh(df.loc[m, 'ch_p'])
    pv = pd.concat(pv, ignore_index=True)[['key', 'stratum', 'variant', 'person_id', 'group', 'value', 'n_trials',
                                           'posture', 'platform']]
    return df, pv


def _beyond(r: pd.Series | dict[str, Any]) -> bool:
    v = r.get('n_more_extreme')
    return v is not None and not pd.isna(v) and int(v) == 0


def main_table(cmp: pd.DataFrame) -> pd.DataFrame:
    """One row per characteristic: its primary comparison, the sensitivity analyses as extra columns and the
    robustness verdict. A "beyond every reference person" verdict is *robust* when it also holds in every
    sensitivity analysis that recomputed values (first k, same protocol, without clip-flagged trials, same posture
    when it changed a person's value) and, for characteristics with a robust z, when |z| >= ``x1_stats.Z_CLEAR`` on
    the declared scale with each reference person left out in turn (a size check). A check whose case value cannot
    be computed (too few case trials left) is listed in ``checks_unavailable`` and also withholds "robust"."""
    base = cmp[cmp['primary'] & (cmp['variant'] == 'main')].set_index('key')
    out = base.copy()
    for var in XS.VARIANTS[1:]:
        s = cmp[cmp['primary'] & (cmp['variant'] == var)].set_index('key')
        for col in ('pd_value', 'percentile', 'robust_z', 'p_rank', 'n_ref', 'n_more_extreme', 'side', 'n_ref_changed',
                    'case_changed', 'n_case_trials'):
            out[f'{col}_{var}'] = s[col].reindex(out.index) if col in s else np.nan
    order = {c.key: i for i, c in enumerate(XS.CHARACTERISTICS)}
    out = out.reset_index()
    by = {c.key: c for c in XS.CHARACTERISTICS}
    out['order'] = out['key'].map(order)
    for f in ('key_set', 'multivariate', 'rationale', 'fmt', 'scale', 'report_z'):
        out[f] = out['key'].map({k: getattr(c, f) for k, c in by.items()})
    beyond, robust, ran, failed, size, unavail = [], [], [], [], [], []
    for _, r in out.iterrows():
        b = _beyond(r)
        checks, fails, na = [], [], []
        if b:
            for var in XS.VALUE_CHECKS:
                ne = r.get(f'n_more_extreme_{var}')
                changes = (not pd.isna(r.get(f'n_ref_changed_{var}')) and int(r.get(f'n_ref_changed_{var}') or 0) > 0) \
                    or bool(r.get(f'case_changed_{var}') is True or r.get(f'case_changed_{var}') == 1)
                if pd.isna(r.get(f'n_ref_{var}')) or not changes:
                    continue
                checks.append(var)
                if pd.isna(r.get(f'pd_value_{var}')):        # too few case trials left: the check cannot be made
                    na.append(var)
                elif pd.isna(ne) or int(ne) != 0 or r.get(f'side_{var}') != r['side']:
                    fails.append(var)
            if bool(r['report_z']):
                checks.append('size')
                zmin = r.get('z_loo_min')
                if pd.isna(zmin) or abs(float(zmin)) < XS.Z_CLEAR or abs(float(r['robust_z'])) < XS.Z_CLEAR:
                    fails.append('size')
        beyond.append(b)
        robust.append(b and not fails and not na)
        ran.append(';'.join(checks))
        failed.append(';'.join(fails))
        unavail.append(';'.join(na))
        size.append('size' in checks and 'size' not in fails)
    out['beyond_all'] = beyond
    out['robust'] = robust
    out['checks_run'] = ran
    out['checks_failed'] = failed
    out['checks_unavailable'] = unavail
    out['size_clear'] = size
    return out.sort_values('order').reset_index(drop=True)


def primary_matrix(pv: pd.DataFrame, keys: list[str], variant: str = 'main') -> pd.DataFrame:
    """Persons x characteristics: each characteristic's person value in its own stratum (one variant)."""
    cols = {}
    for k in keys:
        s = pv[(pv['key'] == k) & (pv['stratum'] == XS.BY_KEY[k].stratum) & (pv['variant'] == variant)]
        cols[k] = s.set_index('person_id')['value']
    return pd.DataFrame(cols)


def null_counts(pv: pd.DataFrame) -> dict[str, Any]:
    """The leave-one-person-out null reference of the "beyond every reference person" count (x1_stats), over the
    pre-specified characteristics and over all of them (the post hoc ones were chosen after looking at the case,
    which favours the case in that count). ``like_for_like`` (the primary null): each reference person is judged on
    the mean of their first k trials of the stratum (k = the case's count; person-level characteristics as they
    are), as noisy as the case's own means, against the others' full means; ``main``: everyone on all trials."""
    phen = {c.key: c.phenomenon for c in XS.CHARACTERISTICS}
    out: dict[str, Any] = {}
    for name, keys in (('prespecified', [c.key for c in XS.CHARACTERISTICS if not c.post_hoc]),
                       ('all', [c.key for c in XS.CHARACTERISTICS])):
        M = primary_matrix(pv, keys)
        if CASE not in M.index:
            raise RuntimeError(f'{CASE} has no person values')
        own = primary_matrix(pv, keys, 'first_k').reindex(index=M.index)
        own.loc[CASE] = M.loc[CASE]
        out[name] = {'like_for_like': XS.beyond_all_counts(M, CASE, phen, own),
                     'main': XS.beyond_all_counts(M, CASE, phen)}
    return out


def redundancy(pv: pd.DataFrame, tab: pd.DataFrame) -> list[dict[str, Any]]:
    """Spearman correlation, across the reference persons, of every pair of characteristics that put the case
    beyond every reference person (does one difference show up several times?)."""
    keys = list(tab.loc[tab['beyond_all'], 'key'])
    M = primary_matrix(pv, keys).drop(index=CASE, errors='ignore')
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ok = M[[a, b]].dropna()
            rho = float(ok[a].corr(ok[b], method='spearman')) if len(ok) >= XS.MIN_REF else math.nan
            rows.append({'a': a, 'b': b, 'n': int(len(ok)), 'rho': rho,
                         'same_phenomenon': XS.BY_KEY[a].phenomenon == XS.BY_KEY[b].phenomenon,
                         'redundant': bool(math.isfinite(rho) and abs(rho) >= REDUNDANT_RHO)})
    return rows


# ───────────────────────────── targeted checks ─────────────────────────────


def _trial_share(df: pd.DataFrame, col: str) -> dict[str, Any]:
    v = pd.to_numeric(df[col], errors='coerce').dropna()
    return {'n': int(len(v)), 'k': int((v > 0.5).sum()), 'share': float(v.mean()) if len(v) else None}


def checks(t: pd.DataFrame) -> dict[str, Any]:
    """Checks of alternative explanations raised by the audit, from the trial table (no stream needed):

    * the 7-12 Hz band power of the reach against speed-matched reference trials (trial level) and, per person,
      before the first touch, without the touch and relative to the peak speed;
    * whether the first touch coincides with the acceleration peak, and how far the early yaw peak lies from it;
    * how often the hand-consistent net yaw is reversed;
    * the spectral resolution of the windows and the leakage of a tone just below 7 Hz into the 7-12 Hz band."""
    ref, case = t[t['group'] == 'reference'], t[t['group'] == 'exploratory']
    lun = XS.STRATA['left_upright_natural']
    out: dict[str, Any] = {}
    # speed-matched 7-12 Hz (left-hand upright natural reaches)
    rl, cl = ref[lun.select(ref)], case[lun.select(case)]
    lo, hi = float(cl['peak_speed_mv'].min()), float(cl['peak_speed_mv'].max())
    sm = rl[(rl['peak_speed_mv'] >= lo) & (rl['peak_speed_mv'] <= hi)]
    out['speed_matched'] = {
        'stratum': lun.key, 'speed_range': [lo, hi], 'n_ref_trials': int(len(sm)),
        'n_ref_persons': int(sm['person_id'].nunique()), 'ref_max': float(sm['app_rms_7_12'].max()) if len(sm) else None,
        'ref_median': float(sm['app_rms_7_12'].median()) if len(sm) else None,
        'case_values': [float(v) for v in cl['app_rms_7_12']],
        'n_case_above_ref_max': int((cl['app_rms_7_12'] > sm['app_rms_7_12'].max()).sum()) if len(sm) else None,
        'n_case': int(len(cl))}
    # person-level band-pass checks (left-hand upright natural)
    pers = {}
    for col in ('app_bp_7_12', 'app_bp_7_12_pre_touch', 'app_bp_7_12_excl_touch'):
        vals = XS.person_trial_values(t, col, lun.key)
        r = vals[vals['person_id'] != CASE]['value'].dropna().tolist()
        x = vals.loc[vals['person_id'] == CASE, 'value']
        x = float(x.iloc[0]) if len(x) and pd.notna(x.iloc[0]) else math.nan
        pers[col] = XS.compare(r, x, 'log', pd.to_numeric(cl[col], errors='coerce').dropna().tolist())
    out['bandpass'] = pers
    # how sharp the reach's 3-12 Hz spectral peak is (a tremor gives a prominent peak, a broadband rise does not):
    # the mean of each person's per-trial prominences, so persons with more trials are not smoothed more
    vals = XS.person_trial_values(t, 'app_peak_prom_db', lun.key)
    r = vals[vals['person_id'] != CASE]['value'].dropna().tolist()
    x = vals.loc[vals['person_id'] == CASE, 'value']
    x = float(x.iloc[0]) if len(x) and pd.notna(x.iloc[0]) else math.nan
    out['peak_prominence'] = XS.compare(r, x, 'linear', pd.to_numeric(cl['app_peak_prom_db'], errors='coerce')
                                        .dropna().tolist())
    # the first touch and the acceleration peak, the early yaw peak (every accepted trial with a touch)
    rt, ctt = ref[ref['touch_s'].notna()], case[case['touch_s'].notna()]
    out['touch_at_acc_peak'] = {'case': _trial_share(ctt, 'touch_at_acc_peak'),
                                'reference': _trial_share(rt, 'touch_at_acc_peak'),
                                'reference_lun': _trial_share(rt[lun.select(rt)], 'touch_at_acc_peak'),
                                'case_lun': _trial_share(ctt[lun.select(ctt)], 'touch_at_acc_peak'),
                                'window_ms': XF.PARAMS['touch_peak_ms']}
    dyc = (ctt['yaw_peak_s'] - ctt['touch_s']).abs()
    dyr = (rt['yaw_peak_s'] - rt['touch_s']).abs()
    out['yaw_peak_to_touch'] = {'case_median_s': float(dyc.median()) if len(dyc) else None,
                                'case_min_s': float(dyc.min()) if len(dyc) else None,
                                'case_max_s': float(dyc.max()) if len(dyc) else None, 'n_case': int(len(dyc)),
                                'ref_median_s': float(dyr.median()) if len(dyr) else None, 'n_ref': int(len(dyr))}
    # the net yaw against the typical direction (every accepted trial)
    share = ref.groupby('person_id')['net_yaw_hand'].apply(lambda s: float((s.dropna() < 0).mean()))
    cs = float((case['net_yaw_hand'].dropna() < 0).mean())
    out['net_yaw_reversed'] = {
        'n_case': int(case['net_yaw_hand'].notna().sum()), 'k_case': int((case['net_yaw_hand'] < 0).sum()),
        'k_case_below_minus2': int((case['net_yaw_hand'] < -2.0).sum()), 'case_share': cs,
        'ref_median_share': float(share.median()), 'ref_max_share': float(share.max()),
        'n_ref_ge_case': int((share >= cs).sum()), 'n_ref': int(len(share))}
    # the components of each band-power ratio (window -> band -> comparison in the window's stratum)
    dec = {}
    for w, s in (('pre', 'left_upright_all'), ('app', 'left_upright_natural'), ('nose', 'left_upright_all')):
        dec[w] = {}
        for band in XF.PARAMS['decomp_bands_hz']:
            col = f'{w}_rms_{band}'
            vals = XS.person_trial_values(t, col, s)
            r = vals[vals['person_id'] != CASE]['value'].dropna().tolist()
            x = vals.loc[vals['person_id'] == CASE, 'value']
            x = float(x.iloc[0]) if len(x) and pd.notna(x.iloc[0]) else math.nan
            dec[w][band] = XS.compare(r, x, 'log')
    out['decomposition'] = dec
    # clip-flagged trials of the case: in which strata
    out['case_clip'] = {s: {'n': int(XS.STRATA[s].select(case).sum()),
                            'flagged': int((XS.STRATA[s].select(case) & case['clip_candidate']).sum())}
                        for s in ('left_upright_natural', 'left_upright_all')}
    # spectral resolution and leakage
    res = {'pre': {'window_s': XF.PARAMS['pre_window_s']}, 'nose': {'window_s': XF.PARAMS['nose_window_s']}}
    for w in res.values():
        w['resolution_hz'] = 1.0 / w['window_s']
        w['main_lobe_half_width_hz'] = 2.0 / w['window_s']
        leak = XF.sine_leakage(LEAKAGE_TONE_HZ, w['window_s'])
        w['tone_hz'] = LEAKAGE_TONE_HZ
        w['tone_rms'] = 10.0 / math.sqrt(2.0)
        w['in_3_7'] = leak['rms_3_7']
        w['leak_7_12'] = leak['rms_7_12']
    dur = float(rl['duration_s'].median())
    res['approach'] = {'median_duration_s': dur, 'resolution_hz': 1.0 / dur}
    out['resolution'] = res
    return out


# ───────────────────────────── spectra, traces, yaw curves ─────────────────────────────


def spectra(inp: dict[str, Any]) -> pd.DataFrame:
    """Gyroscope PSDs on SPECTRUM_GRID_HZ, each window from its own stratum (``SPECTRUM_STRATA``): per reference
    person the mean over their trials, per case trial, and the case mean."""
    rows = []
    for group, D in (('reference', inp['ref']), ('exploratory', inp['case'])):
        masks = {w: XS.STRATA[s].select(D.trials).to_numpy() for w, s in SPECTRUM_STRATA.items()}
        anym = np.logical_or.reduce(list(masks.values()))
        acc: dict[tuple[str, str], list[np.ndarray]] = {}
        for i in np.flatnonzero(anym):
            r = D.trials.iloc[i]
            sp = XF.trial_spectra(D.stream(i), r.to_dict())
            for w in SPECTRUM_STRATA:
                if sp[w] is None or not masks[w][i]:
                    continue
                f, P = sp[w]
                Pi = np.interp(SPECTRUM_GRID_HZ, f, P)
                acc.setdefault((str(r['person_id']), w), []).append(Pi)
                if group == 'exploratory':
                    rows += [{'window': w, 'stratum': SPECTRUM_STRATA[w], 'group': group,
                              'person_id': str(r['person_id']), 'trial_key': r['trial_key'], 'level': 'trial',
                              'f_hz': fq, 'psd': float(p)} for fq, p in zip(SPECTRUM_GRID_HZ, Pi)]
        for (pid, w), lst in sorted(acc.items()):
            P = np.mean(lst, axis=0)
            rows += [{'window': w, 'stratum': SPECTRUM_STRATA[w], 'group': group, 'person_id': pid, 'trial_key': '',
                      'level': 'person', 'f_hz': fq, 'psd': float(p), 'n_trials': len(lst)}
                     for fq, p in zip(SPECTRUM_GRID_HZ, P)]
    return pd.DataFrame(rows)


def typical_trial(t: pd.DataFrame, group: str, stratum: str = 'left_upright_natural') -> str:
    """The trial (key) of ``group`` in ``stratum`` closest to the median person on TYPICAL_KEYS (robust z units;
    the reference's person-level medians and scales; for the case, the case's own trial medians)."""
    sub = t[(t['group'] == group) & XS.STRATA[stratum].select(t)].copy()
    ref = t[(t['group'] == 'reference') & XS.STRATA[stratum].select(t)]
    d = np.zeros(len(sub))
    for k in TYPICAL_KEYS:
        pv = ref.groupby('person_id')[k].mean().dropna().to_numpy()
        med, sc = XS.robust_scale(pv)
        centre = med if group == 'reference' else float(sub[k].median())
        d += ((sub[k].to_numpy(dtype=float) - centre) / sc) ** 2
    return str(sub['trial_key'].iloc[int(np.nanargmin(d))])


def traces(inp: dict[str, Any], t: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Canonical channels (gyro magnitude, yaw rate, linear-acceleration magnitude) over each case trial and the
    reference's typical matched trial, re-derived with ``standardize.canonical`` from the cached stream with the
    calibrated |g| recovered from the cached fixed window (``x1_features.cached_canonical``: the cache's values)."""
    sel_case = typical_trial(t, 'exploratory')
    sel_ref = typical_trial(t, 'reference')
    rows, events = [], {}
    for group, D in (('exploratory', inp['case']), ('reference', inp['ref'])):
        keys = D.trials['trial_key'].astype(str)
        idx = np.flatnonzero(keys.to_numpy() == sel_ref) if group == 'reference' else np.arange(D.n)
        for i in idx:
            r = D.trials.iloc[i]
            s = D.stream(int(i))
            on, en = float(r['onset_ms']), float(r['end_ms'])
            can, _, _ = XF.cached_canonical(s, r.to_dict(), D.fixed[int(i)], D.fixed_mask[int(i)], D.channels)
            m = (can.t >= on - TRACE_PAD_S[0] * 1000.0) & (can.t <= en + TRACE_PAD_S[1] * 1000.0)
            key = str(r['trial_key'])
            df = pd.DataFrame({'group': group, 'person_id': str(r['person_id']), 'trial_key': key,
                               'selected': key in (sel_case, sel_ref), 't_s': (can.t[m] - on) / 1000.0,
                               'gyro_mag': can.channel('gyro_mag')[m], 'yaw_rate': can.channel('yaw_rate')[m],
                               'acc_mag': can.channel('acc_mag')[m], 'valid': can.valid[m]})
            rows.append(df)
            cue = r.get('end_detected_ms')
            events[key] = {'group': group, 'person_id': str(r['person_id']), 'hand': r['hand'], 'speed': r['speed'],
                           'analysis_set': r['analysis_set'], 'end_s': (en - on) / 1000.0,
                           'cue_s': (float(cue) - on) / 1000.0 if pd.notna(cue) else None,
                           'go_s': (float(r['go_ms']) - on) / 1000.0, 'selected': key in (sel_case, sel_ref)}
    return pd.concat(rows, ignore_index=True), {'selected_case': sel_case, 'selected_reference': sel_ref,
                                                'typical_keys': list(TYPICAL_KEYS), 'events': events}


def yaw_curves(inp: dict[str, Any]) -> pd.DataFrame:
    """Mirror-normalised yaw rate and cumulative yaw over the time-normalised approach (cached canonical arrays):
    reference person means and case trials of the left-hand upright trials, and the free-choice trials (the reference
    persons' first free trial, i.e. the natural-use set; the case's free trials, first and repeat, each marked by
    its analysis set). The cumulative yaw is what the first-half yaw reads at half of the approach."""
    rows = []
    phase = np.linspace(0.0, 1.0, C.N_NORM)
    for group, D in (('reference', inp['ref']), ('exploratory', inp['case'])):
        sgn = D.trials['hand'].map(XF.hand_sign).to_numpy(dtype=float)[:, None]
        Y = D.approach[:, D.channels.index('yaw_rate'), :].astype(float) * sgn
        Yc = D.approach[:, D.channels.index('yaw_cum'), :].astype(float) * sgn
        free_sets = ('natural_use',) if group == 'reference' else ('natural_use', 'natural_use_repeat')
        for stratum, m in (('left_upright_all', XS.STRATA['left_upright_all'].select(D.trials)),
                           ('free_choice', D.trials['analysis_set'].isin(free_sets))):
            m = m.to_numpy()
            tr = D.trials[m]
            Ym, Ycm = Y[m], Yc[m]

            def emit(level: str, pid: str, key: str, aset: str, hand: str, yr: np.ndarray, yc: np.ndarray) -> None:
                rows.extend({'stratum': stratum, 'group': group, 'person_id': pid, 'trial_key': key,
                             'analysis_set': aset, 'hand': hand, 'level': level, 'phase': float(ph),
                             'yaw_rate_hand': float(a), 'yaw_cum_hand': float(b)} for ph, a, b in zip(phase, yr, yc))

            if group == 'reference' and stratum == 'left_upright_all':
                for pid, ix in tr.groupby('person_id').indices.items():
                    emit('person', str(pid), '', 'primary', 'left', Ym[ix].mean(axis=0), Ycm[ix].mean(axis=0))
            else:
                for j in range(len(tr)):
                    r = tr.iloc[j]
                    emit('trial', str(r['person_id']), r['trial_key'], r['analysis_set'], str(r['hand']), Ym[j],
                         Ycm[j])
    return pd.DataFrame(rows)


# ───────────────────────────── multivariate and task behaviour ─────────────────────────────


def multivariate_inputs(pv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    M = primary_matrix(pv, [c.key for c in XS.CHARACTERISTICS if c.multivariate])
    return M.drop(index=CASE).sort_index(), M.loc[CASE]


def _session_minutes(sessions: pd.DataFrame, attempts: pd.DataFrame) -> pd.Series:
    """Session length in minutes: from the session's start to the end of its last recorded attempt (defined also
    for a session left in progress). Only the length is kept, never a time."""
    st = pd.to_datetime(sessions['started_at'], utc=True, errors='coerce')
    st_ms = (st - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds() * 1000.0
    end = (pd.to_numeric(attempts['t0_epoch_ms'], errors='coerce')
           + pd.to_numeric(attempts['stop_ms'], errors='coerce').fillna(0.0)).groupby(attempts['session_id']).max()
    return ((sessions['session_id'].map(end) - st_ms.to_numpy()) / 60000.0).rename('minutes')


def task_summary(inp: dict[str, Any]) -> dict[str, Any]:
    """Attempts, rejections and how sessions ended, their length, protocol and rotation-rate mapping, and the
    clip flags, for the case and the reference persons (no ids, no times)."""
    from stage2.data import e1_reason
    ca = inp['case_attempts']
    ca = ca[ca['session_included'].astype(bool)]
    acc = ca['accepted'].astype('boolean').fillna(False).astype(bool)
    rej = ca[~acc]
    reasons = e1_reason(rej).fillna('other').value_counts().sort_index()
    cs = inp['case_sessions']
    cs = cs[cs['include'].astype(bool)].sort_values('session_order')
    cp = inp['case_persons']
    ra = inp['ref_attempts']
    ra = ra[ra['session_included'].astype(bool)]
    racc = ra['accepted'].astype('boolean').fillna(False).astype(bool)
    rs = inp['ref_sessions']
    rs = rs[rs['include'].astype(bool)]
    rp = inp['ref_persons']
    age = pd.to_numeric(cp['age_years'], errors='coerce').iloc[0] if len(cp) else math.nan
    band = f'{int(age // 10 * 10)}-{int(age // 10 * 10 + 9)}' if math.isfinite(age) else 'unknown'
    ref_age = pd.to_numeric(rp['age_years'], errors='coerce')
    cmin = _session_minutes(cs, ca)
    rmin = _session_minutes(rs, ra)
    per_session = []
    for (_, s), mins in zip(cs.iterrows(), cmin):
        a = ca[ca['session_label'] == s['session_label']]
        aa = a['accepted'].astype('boolean').fillna(False).astype(bool)
        per_session.append({
            'session_label': s['session_label'], 'status': s['status'], 'n_attempts': int(len(a)),
            'n_accepted': int(aa.sum()), 'minutes': float(mins),
            'n_planned_trials': int(s['n_planned_trials']) if pd.notna(s.get('n_planned_trials')) else None,
            'tiers_reached': sorted({int(v) for v in a['tier'].dropna()}),
            'protocol_version': str(s['protocol_version']), 'preset': str(s['preset']), 'posture': str(s['posture']),
            'rr_source': str(s['rr_source']), 'rr_verified': bool(s['rr_verified']),
            'rr_closure_deg': float(s['rr_closure_deg']) if pd.notna(s['rr_closure_deg']) else None,
            'rr_runner_up_deg': (float(s['rr_closure_runner_up_deg']) if pd.notna(s['rr_closure_runner_up_deg'])
                                 else None),
            'rr_trials': int(s['rr_closure_trials']) if pd.notna(s['rr_closure_trials']) else None})
    clip_acc = ca.loc[acc, 'clip_candidate'].astype('boolean').fillna(False).astype(bool)
    clip_rej = ca.loc[~acc, 'clip_candidate'].astype('boolean').fillna(False).astype(bool)
    raw_acc = pd.to_numeric(ca.loc[acc, 'raw_n_clip_acc'], errors='coerce').fillna(0) > 0
    raw_gyro = pd.to_numeric(ca.loc[acc, 'raw_n_clip_gyro'], errors='coerce').fillna(0) > 0
    ref_an = ra[ra['analysis_set'].isin(XS.SETS_ALL)]
    ref_pr = ra[ra['analysis_set'] == 'primary']
    rclip = ref_an['clip_candidate'].astype('boolean').fillna(False).astype(bool)
    pclip = ref_pr['clip_candidate'].astype('boolean').fillna(False).astype(bool)
    case_protocol = _mode(ca['protocol_version'])
    other_proto = rs[rs['protocol_version'].astype(str) != case_protocol]
    op_trials = ref_an[ref_an['session_label'].isin(set(other_proto['session_label']))]
    return {
        'case': {'person_id': CASE, 'age_band': band, 'handedness': str(cp['handedness_self'].iloc[0]) if len(cp) else '',
                 'phone_hand': str(cp['phone_hand'].iloc[0]) if len(cp) else '',
                 'platform': ';'.join(sorted(set(cs['platform'].astype(str)))),
                 'posture': ';'.join(sorted(set(ca['posture'].dropna().astype(str)))),
                 'protocol_version': case_protocol,
                 'n_sessions': int(len(cs)), 'n_attempts': int(len(ca)), 'n_accepted': int(acc.sum()),
                 'n_rejected': int((~acc).sum()), 'rejections': {k: int(v) for k, v in reasons.items()},
                 'n_primary': int((ca['analysis_set'] == 'primary').sum()),
                 'n_free_choice': int(ca['analysis_set'].isin(('natural_use', 'natural_use_repeat')).sum()),
                 'n_free_first': int((ca['analysis_set'] == 'natural_use').sum()),
                 'n_free_repeat': int((ca['analysis_set'] == 'natural_use_repeat').sum()),
                 'free_hands': sorted(set(ca.loc[ca['analysis_set'].isin(('natural_use', 'natural_use_repeat')),
                                                 'hand'].dropna().astype(str))),
                 'primary_hands': sorted(set(ca.loc[ca['analysis_set'] == 'primary', 'hand'].dropna().astype(str))),
                 'primary_orientations': sorted(set(ca.loc[ca['analysis_set'] == 'primary', 'orientation']
                                                    .dropna().astype(str))),
                 'false_starts_total_accepted': int(pd.to_numeric(ca.loc[acc, 'false_starts'], errors='coerce').sum()),
                 'n_clip_accepted': int(clip_acc.sum()), 'n_clip_rejected': int(clip_rej.sum()),
                 'n_clip_acc_accepted': int(raw_acc.sum()), 'n_clip_gyro_accepted': int(raw_gyro.sum()),
                 'primary_speeds': {k: int(v) for k, v in ca.loc[ca['analysis_set'] == 'primary', 'speed']
                                    .value_counts().sort_index().items()},
                 'minutes_total': float(cmin.sum()), 'sessions': per_session},
        'reference': {'n_persons': int(rp['person_id'].nunique()), 'age_min': float(ref_age.min()),
                      'age_max': float(ref_age.max()),
                      'n_sessions': int(len(rs)),
                      'session_status': {k: int(v) for k, v in rs['status'].value_counts().sort_index().items()},
                      'session_minutes': {'median': float(rmin.median()), 'min': float(rmin.min()),
                                          'max': float(rmin.max())},
                      'n_attempts': int(len(ra)), 'n_accepted': int(racc.sum()),
                      'attempts_per_accepted': float(len(ra) / racc.sum()),
                      'rejections': {k: int(v) for k, v in e1_reason(ra[~racc]).fillna('other').value_counts().items()},
                      'platforms': {k: int(v) for k, v in rp['platform'].value_counts().sort_index().items()},
                      'postures': {k: int(v) for k, v in rs['posture'].value_counts().sort_index().items()},
                      'n_rr_verified': int(rs['rr_verified'].astype('boolean').fillna(False).sum()),
                      'n_clip_primary': int(pclip.sum()), 'n_primary': int(len(ref_pr)),
                      'n_clip_persons_primary': int(ref_pr.loc[pclip, 'person_id'].nunique()),
                      'n_clip_analysable': int(rclip.sum()), 'n_analysable': int(len(ref_an)),
                      'n_sessions_other_protocol': int(len(other_proto)),
                      'other_protocol_persons': sorted(set(other_proto['person_id'].astype(str))),
                      'n_trials_other_protocol': int(len(op_trials))},
    }


# ───────────────────────────── the step ─────────────────────────────


def run(force: bool = False, only: list[str] | None = None) -> None:
    """Run the selected analyses whose outputs are missing or stale (``force``: all selected)."""
    todo = [a for a in ANALYSES if selected(a, only)]
    if 'x1_cache' in todo:
        XD.ensure_cache(CASE, force=force)
    prov = x1_provenance()
    todo = [a for a in todo if a != 'x1_cache' and (force or not current(a, prov))]
    if not todo:
        print('x1: derived files current; skipped (--force recomputes)', flush=True)
        return
    inp = load_inputs()
    need_t = {'x1_characteristics', 'x1_checks', 'x1_traces', 'x1_multivariate'} & set(todo)
    if 'x1_characteristics' in todo or (need_t and not current('x1_characteristics', prov)):
        print('x1: per-trial characteristics ...', flush=True)
        t = all_characteristics(inp)
        write_csv(t, 'x1_trial_characteristics')
        attempts = pd.concat([inp['ref_attempts'], inp['case_attempts']], ignore_index=True)
        cmp, pv = comparisons(t, attempts)
        write_csv(cmp, 'x1_comparisons')
        write_csv(pv, 'x1_person_values')
        tab = main_table(cmp)
        write_json('x1_characteristics', {
            'case': CASE, 'rows': tab.to_dict(orient='records'),
            'strata': {k: {'label': s.label, 'note': s.note} for k, s in XS.STRATA.items()},
            'null': null_counts(pv), 'redundancy': redundancy(pv, tab),
            'settings': {'min_ref': XS.MIN_REF, 'z_clear': XS.Z_CLEAR, 'redundant_rho': REDUNDANT_RHO,
                         'value_checks': list(XS.VALUE_CHECKS), 'variants': list(XS.VARIANTS)}},
            prov, folder='results')
        mark_done('x1_characteristics', prov)
    if 'x1_checks' in todo:
        print('x1: checks ...', flush=True)
        write_json('x1_checks', checks(read_trials()), prov)
        mark_done('x1_checks', prov)
    if 'x1_spectra' in todo:
        print('x1: spectra ...', flush=True)
        write_csv(spectra(inp), 'x1_spectra')
        mark_done('x1_spectra', prov)
    if 'x1_traces' in todo:
        print('x1: example traces ...', flush=True)
        tr, ev = traces(inp, read_trials())
        write_csv(tr, 'x1_traces')
        write_json('x1_traces', ev, prov)
        mark_done('x1_traces', prov)
    if 'x1_yaw_curves' in todo:
        write_csv(yaw_curves(inp), 'x1_yaw_curves')
        mark_done('x1_yaw_curves', prov)
    if 'x1_multivariate' in todo:
        print('x1: multivariate map ...', flush=True)
        pv = pd.read_csv(L.tab_dir / 'x1_person_values.csv')
        pv['person_id'] = pv['person_id'].astype(str)
        ref, x = multivariate_inputs(pv)
        mv = XS.multivariate(ref, x)
        lin = XS.multivariate(ref, x, declared_scales=False)
        write_json('x1_multivariate', {
            'case': CASE, **mv, 'loco': XS.multivariate_loco(ref, x),
            'linear': {k: lin[k] for k in ('mahalanobis_x', 'mahalanobis_ref_loo', 'n_ref_beyond', 'p_mahalanobis')}
            | {'loco': XS.multivariate_loco(ref, x, declared_scales=False)},
            'labels': {k: XS.BY_KEY[k].label for k in mv['keys']}}, prov, folder='results')
        mark_done('x1_multivariate', prov)
    if 'x1_task' in todo:
        write_json('x1_task', task_summary(inp), prov)
        mark_done('x1_task', prov)
