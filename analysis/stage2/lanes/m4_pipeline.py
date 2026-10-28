"""Lane m4: the cue-free production pipeline (detector + hand classifier + calibration + stopping rule).

    cd analysis
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m4            # missing / stale experiments
    .venv/Scripts/python.exe -m stage2.pipeline compute --lanes m4 --force    # everything (about 1-2 h, 2 threads)
    .venv/Scripts/python.exe -m stage2.pipeline figures --lanes m4            # redraw from saved files (seconds)
    .venv/Scripts/python.exe -m stage2.pipeline tables --lanes m4
    .venv/Scripts/python.exe -m stage2.pipeline numbers --lanes m4            # m4_numbers.json + m4_SUMMARY.md

compute runs the trial-level experiments through ``run_experiment`` (unit = one recording, LOSO by person,
leak-checked; X = indices into label-free streams), then the stream-level analyses (stress grid with the
screen-away twin, synthetic multi-attempt sessions, online stopping rule, calibration, latency, timeline, demo,
natural-use failure analysis, replay of the online front end), which reuse each fold's fitted pipeline on the SAME
LOSO splits (``cv.loso_splits``, the harness's training rows and seeds) and bootstrap persons. Everything a figure,
table or number needs is saved under results/ and tables/m4/. Every accuracy of the lane scores an abstention
(P = 0.5) as half a correct answer (``m4_stats``).
"""
from __future__ import annotations

import os  # noqa: E402

from stage2.pipeline import Lane, file_current, limit_threads, selected

#: the lane's thread budget (2), or a smaller cap already set (e.g. --threads 1 for two parallel compute processes)
limit_threads(max(1, min(2, int(os.environ.get('OMP_NUM_THREADS') or 2))))

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stage2 import config as C  # noqa: E402
from stage2.lanes.m4_config import (ABSTAIN_TAU, AGGREGATION_N_PARAMS, AGGREGATION_RULE, ALPHA, K_MAX,  # noqa: E402,F401
                                    GRIP_GATE_UPZ, GRIP_REST_MIN_S, KEEP_S_SESSIONS, KEEP_S_STRESS, MIN_PERSONS_CI,
                                    N_BOOT, P_ANSWER, P_ANSWER_GRID,
                                    P_STOP, P_STOP_GRID, PRODUCTION_PAIRING, SESSION_RECORDINGS, SESSION_RULES,
                                    SESSION_STRIDE, STOP_AGGREGATIONS, STRESS_FLIP_KEEPS)

L = Lane('m4')
ORDER = 140
DEPENDS: tuple[str, ...] = ()

# ───────────────────────────── configuration (with m4_config: the one place for every choice) ─────────────────

#: files the compute step writes under tables/m4 (raw rows read back by later compute steps and figures) and the
#: report tables of the tables step: the two lists must never share a name (one would overwrite the other)
RAW_TABLES = ('m4_sessions_krows', 'm4_sessions_stop', 'm4_sessions_answer', 'm4_sessions_detection',
              'm4_sessions_attempts', 'm4_sessions_scales', 'm4_stress_details', 'm4_segfeat_values',
              'm4_replay_details')
REPORT_TABLES = ('m4_models', 'm4_detection', 'm4_calibration', 'm4_sessions_k', 'm4_stopping', 'm4_stop_calibration',
                 'm4_breakdown', 'm4_stress', 'm4_paired', 'm4_natural', 'm4_ablations')
#: look-ahead (s) of the ablation experiments (production: SegParams.lookahead_s)
LOOKAHEAD_ABLATION = {'m4_abl_look0p3': 0.3, 'm4_abl_look0p8': 0.8}
#: result files of earlier versions of the lane that no experiment writes any more (removed by compute)
OBSOLETE = ('m4_v1grip_chiral16',)


def reference_keep_s() -> float:
    """The junction stillness at which the sessions are reported in full: the instructed stillness of the
    inference page (``m4_detect.INSTRUCTED_STILLNESS_S``, one place); it must be one of KEEP_S_SESSIONS."""
    from stage2.lanes.m4_detect import INSTRUCTED_STILLNESS_S
    assert INSTRUCTED_STILLNESS_S in KEEP_S_SESSIONS
    return float(INSTRUCTED_STILLNESS_S)


#: calibrators compared on the outer folds (none = the uncalibrated base); 'nested' = chosen inside each fold
CALIBRATION_CANDIDATES = ('none', 'temperature', 'platt', 'isotonic', 'beta', 'isotonic_sym', 'nested')
BASE_FOR_CALIBRATION = 'm4_v1robust_yaw8'         # the production detector, uncalibrated
PRODUCTION = 'm4_cal_nested'                      # detector + yaw8 + the calibrator chosen inside each fold
ORACLE_EXPERIMENTS = ('m4_oracle_yaw8', 'm4_oracle_chiral16')
DETECTOR_FAMILY = ('m4_v1phys_yaw8', 'm4_v1full_yaw8', 'm4_v1nest_yaw8', 'm4_v1robust_yaw8', 'm4_v1grip_yaw8')
DETECTOR_ABLATION = ('m4_v0_yaw8', *DETECTOR_FAMILY)
CUE_FREE_BASE = (*DETECTOR_ABLATION, 'm4_v1nest_yaw8_detwin', 'm4_v1robust_chiral16')
CAL_EXPERIMENTS = tuple(f'm4_cal_{c}' for c in CALIBRATION_CANDIDATES if c != 'none')
RETURN_EXPERIMENTS = ('m4_ret_app_yaw8', 'm4_ret_head_yaw8', 'm4_ret_sum_yaw8')
PROD_VARIANTS = ('m4_prod_return', 'm4_prod_zp_activity', 'm4_prod_causal_lowpass')
ABLATIONS = ('m4_abl_scale_z', 'm4_abl_scale_none', 'm4_abl_segc_nested', *LOOKAHEAD_ABLATION)
TRIAL_EXPERIMENTS = (*ORACLE_EXPERIMENTS, *CUE_FREE_BASE, *CAL_EXPERIMENTS, *RETURN_EXPERIMENTS, *PROD_VARIANTS,
                     *ABLATIONS)
#: every cue-free single-recording pipeline evaluated on the outer folds: each could have become the production
#: one, so the production result carries this count (conservative; cross-lane rule)
CUE_FREE_COMPARED = (*CUE_FREE_BASE, *CAL_EXPERIMENTS, *PROD_VARIANTS, *ABLATIONS)
N_CONFIGS_FINAL = len(CUE_FREE_COMPARED)
#: the session-level choices made on the outer sessions: attempt rule x aggregation
N_CONFIGS_SESSIONS = len(SESSION_RULES) * len(STOP_AGGREGATIONS)


def n_configs(name: str) -> int:
    """n_configs_tried recorded with each result: the production pipeline carries every cue-free configuration
    compared on the outer folds, the production detector the detector feature sets compared; any other result is
    one pre-specified configuration."""
    if name == PRODUCTION:
        return N_CONFIGS_FINAL
    if name == BASE_FOR_CALIBRATION:
        return len(DETECTOR_FAMILY)
    return 1


def _specs() -> dict[str, Any]:
    from stage2.lanes.m4_e2e import PipelineSpec as S
    base = S(detector='learned', seg='nested', features='nested', hand_model='yaw8', hand_train='oracle')
    robust = replace(base, features='robust')     # stillness before capped at the instructed 1 s (production)
    grip = replace(base, features='grip')         # robust + screen-odd features signed by the grip (invariant)
    uncal = replace(robust, calibration='none', inner_calibration=False)
    return {
        'm4_v0_yaw8': S(detector='v0'),
        'm4_v1phys_yaw8': replace(base, features='physics'),
        'm4_v1full_yaw8': replace(base, features='full'),
        'm4_v1nest_yaw8': base,
        'm4_v1robust_yaw8': robust,
        'm4_v1grip_yaw8': grip,
        'm4_v1nest_yaw8_detwin': replace(base, hand_train='detector'),
        'm4_v1robust_chiral16': replace(uncal, hand_model='chiral16'),
        **{f'm4_cal_{c}': replace(robust, calibration=c) for c in CALIBRATION_CANDIDATES if c != 'none'},
        'm4_abl_scale_z': replace(uncal, seg_scaling='z'),
        'm4_abl_scale_none': replace(uncal, seg_scaling='none'),
        'm4_abl_segc_nested': replace(uncal, seg_C='nested'),
        **{k: uncal for k in LOOKAHEAD_ABLATION},
    }


def production_spec() -> Any:
    return _specs()[PRODUCTION]


def variant_spec(spec: Any = None) -> Any:
    """The spec of the production VARIANTS (return head, zero-phase activity, causal low-pass, the chiral16
    plug-in, the ablations): the production detector WITHOUT calibration and without the inner cross-fitting that
    only the calibrator needs, compared pairwise with the uncalibrated production detector
    (``BASE_FOR_CALIBRATION``), so each comparison isolates the variant's own change."""
    base = _specs()[BASE_FOR_CALIBRATION] if spec is None else spec
    return replace(base, calibration='none', inner_calibration=False)


def production_name() -> str:
    return PRODUCTION


def production_aggregation(cal: dict[str, Any] | None = None) -> str:
    """The production aggregation of attempts by the declared rule (m4_config.AGGREGATION_RULE) applied to the
    saved stopping-rule calibration (tables/m4/m4_calibration.json, key 'stop'). Raises when it is missing."""
    cal = cal if cal is not None else L.read_json('m4_calibration')
    stop = cal['stop']
    ok = [h for h in STOP_AGGREGATIONS if h in stop and stop[h]['calibrated']]
    if ok:
        return min(ok, key=lambda h: (AGGREGATION_N_PARAMS[h], STOP_AGGREGATIONS.index(h)))
    return min((h for h in STOP_AGGREGATIONS if h in stop), key=lambda h: stop[h]['log_loss'])


def production_aggregate(prep: Any, fp: Any, idx: np.ndarray, y: np.ndarray, groups: np.ndarray, meta: pd.DataFrame,
                         seed: int, how: str) -> Any:
    """The stopping rule's aggregate scale of a fitted pipeline under aggregation ``how`` (None = mean logit)."""
    from stage2.lanes import m4_aggregate as A
    if how == 'scale_sessions':
        return A.session_aggregate(prep, fp, idx, y, groups, meta, seed)[0]
    if how == 'scale_recordings':
        return fp.agg
    if how == 'mean':
        return None
    raise ValueError(how)


_PREP: dict[str, Any] = {}


def _clean(v: Any) -> Any:
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python (Lane.write_json refuses NaN)."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(name: str, obj: dict[str, Any]) -> None:
    L.write_json(name, _clean(obj))  # the only direct call


STREAM_KEYS = ('t', 'f', 'acc', 'gyro', 'valid')


def label_free_stream(st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A cached stream without its 'phase' codes (derived from the app's events: never part of the model input)."""
    return {k: st[k] for k in STREAM_KEYS}


def _chosen_seg_cfgs() -> list[str]:
    """Segmentation configurations the production detector chose in any fold (its saved result)."""
    r = L.load_result(BASE_FOR_CALIBRATION)
    return sorted({json.loads(p).get('seg_cfg') for p in r.predictions['params']} - {None})


def prepared(d: Any = None, nat: Any = None, kind: str = 'main') -> Any:
    """The label-free prepared data (streams, oracle windows, candidate segments of every SEG_GRID config),
    built once per process. kind: 'main' (cache streams, causal activity), 'zp' (zero-phase activity),
    'causal_lowpass' (raw streams with a causal 15 Hz low-pass instead of the zero-phase one), 'look<s>' (another
    look-ahead; only the segmentation configurations the production detector chose)."""
    if kind in _PREP:
        return _PREP[kind]
    from stage2.data import load_cache
    from stage2.lanes import m4_e2e as E
    d = d if d is not None else load_cache()
    nat = nat if nat is not None else load_cache(('natural_use',))
    plat = list(d.trials['platform']) + list(nat.trials['platform'])
    ret = np.concatenate([d.ret, nat.ret])
    rok = np.concatenate([d.ret_ok, nat.ret_ok])
    if kind == 'main':
        streams = [label_free_stream(d.stream(i)) for i in range(d.n)] + [label_free_stream(nat.stream(i))
                                                                           for i in range(nat.n)]
        prep = E.prepare(streams, np.concatenate([d.approach, nat.approach]), ret, rok, platform=plat,
                         progress='m4 prepare')
    elif kind == 'zp':
        base = prepared(d, nat, 'main')
        grid = {k: replace(p, causal=False) for k, p in E.SEG_GRID.items()}
        prep = E.prepare(base.streams, base.oracle, ret, rok, platform=plat, grid=grid, with_v0=False,
                         progress='m4 prepare zero-phase activity')
    elif kind == 'causal_lowpass':
        streams, oracle = causal_lowpass_streams(d, nat)
        prep = E.prepare(streams, oracle, ret, rok, platform=plat, with_v0=False, progress='m4 prepare causal')
    elif kind.startswith('look'):
        base = prepared(d, nat, 'main')
        look = float(kind[4:])
        grid = {k: replace(E.SEG_GRID[k], lookahead_s=look) for k in _chosen_seg_cfgs()}
        prep = E.prepare(base.streams, base.oracle, ret, rok, platform=plat, grid=grid, with_v0=False,
                         progress=f'm4 prepare look-ahead {look:g} s')
    else:
        raise ValueError(kind)
    _PREP[kind] = prep
    return prep


def causal_lowpass_streams(d: Any, nat: Any) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Streams rebuilt from the raw records with a CAUSAL 15 Hz Butterworth (order 4) instead of the zero-phase
    one, and the oracle windows recomputed on them (same bounds): what a fully causal front end would feed the
    same pipeline. Reads the snapshot (about 20 s)."""
    from scipy import signal
    from stage2.lanes import m4_detect as D
    sos = signal.butter(C.LOWPASS_ORDER, C.LOWPASS_HZ, btype='low', fs=C.FS_HZ, output='sos')
    streams, wins = [], []
    for data, raw in ((d, _raw_streams(d)), (nat, _raw_streams(nat))):
        for i in range(data.n):
            r = data.trials.iloc[i]
            st = raw[i]
            out = {'t': st['t'], 'acc': st['acc'], 'valid': st['valid']}
            for k in ('f', 'gyro'):
                v = np.array(st[k], dtype=float)
                for j in range(3):
                    col = v[:, j]
                    ok = np.isfinite(col)
                    if ok.sum() > 12:
                        first = col[ok][0]
                        y, _ = signal.sosfilt(sos, np.where(ok, col, np.interp(st['t'], st['t'][ok], col[ok])),
                                              zi=signal.sosfilt_zi(sos) * first)
                        v[:, j] = np.where(ok, y, np.nan)
                out[k] = v
            streams.append(out)
            W, _ = D.window_for(out, float(r['onset_ms']), float(r['end_ms']), 1.0)
            wins.append(W)
    return streams, np.stack(wins).astype(np.float32)


_DATASET: dict[str, Any] = {}


def _raw_streams(data: Any) -> list[dict[str, np.ndarray]]:
    """The recordings of ``data`` standardised from the raw records WITHOUT the zero-phase low-pass (t, f, acc,
    gyro, valid on the uniform grid): the input of the causal front end and of the buffer replay."""
    import warnings
    from motionlab.io import load_dataset
    from stage2.standardize import stream_from_record
    if 'ds' not in _DATASET:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _DATASET['ds'] = load_dataset(C.SNAPSHOT_DIR, simulated='include')
    ds = _DATASET['ds']
    out = []
    for i in range(data.n):
        tid = data.trials['trial_id'].iloc[i]
        st = stream_from_record(ds.record(tid, compact=True), ds.trial_normalization(tid), lowpass_hz=None)
        out.append({'t': st.t, 'f': st.f, 'acc': st.acc, 'gyro': st.gyro, 'valid': st.valid})
    return out


# ───────────────────────────── experiments (trial unit, run_experiment) ─────────────────────────────


def _oracle_fp(model: str):
    from stage2.lanes import m4_models as M

    def fit_predict(fold):
        h = M.make_hand_model(model).fit(fold.X_train, fold.y_train, fold.groups_train, fold.seed)
        info = {'hand_C': getattr(h, 'C_chosen', None)} if hasattr(h, 'C_chosen') else {}
        return M.sigmoid(h.logit(fold.X_test)), info
    return fit_predict


def _sum_fp(model: str):
    """Approach model on the approach window + return head on the return window: the sum of the two logits."""
    from stage2.lanes import m4_models as M

    def fit_predict(fold):
        Xa, Xr = fold.X_train['approach'], fold.X_train['ret']
        ha = M.make_hand_model(model).fit(Xa, fold.y_train, fold.groups_train, fold.seed)
        hr = M.make_hand_model(model).fit(Xr, fold.y_train, fold.groups_train, fold.seed)
        return M.sigmoid(ha.logit(fold.X_test['approach']) + hr.logit(fold.X_test['ret']))
    return fit_predict


def _details_path(name: str):
    return L.tab_dir / f'{name}_details.csv'


#: notes recorded with the cue-free experiments that are not plain feature-set variants
NOTES = {
    'm4_v1robust_chiral16': 'chiral16 plugged into the production detector (uncalibrated; paired with its yaw8 base)',
    'm4_prod_return': 'production detector (uncalibrated) + return head when the next segment is a return',
    'm4_prod_zp_activity': 'production detector (uncalibrated) with a ZERO-PHASE activity filter (offline only)',
    'm4_prod_causal_lowpass': 'production detector (uncalibrated) on streams low-passed CAUSALLY (15 Hz, order 4)',
    'm4_abl_scale_z': 'ablation: segment features z-scored (mean / SD, clipped at 5 SD) instead of median / IQR',
    'm4_abl_scale_none': 'ablation: segment features unscaled (NaN -> training median)',
    'm4_abl_segc_nested': 'ablation: segment-classifier C chosen inside each fold (inner person-grouped CV)',
    **{k: f'ablation: look-ahead {v:g} s after the end of a segment (production 0.5 s)'
       for k, v in LOOKAHEAD_ABLATION.items()},
}


def experiment_plan(name: str) -> tuple[str, Any, str]:
    """(prepared-data kind, spec, notes) of a cue-free experiment."""
    specs = _specs()
    if name in PROD_VARIANTS:
        spec = variant_spec()
        kind = {'m4_prod_return': 'main', 'm4_prod_zp_activity': 'zp', 'm4_prod_causal_lowpass': 'causal_lowpass'}[name]
        if name == 'm4_prod_return':
            spec = replace(spec, use_return=True)
        return kind, spec, NOTES[name]
    spec = specs[name]
    kind = f'look{LOOKAHEAD_ABLATION[name]:g}' if name in LOOKAHEAD_ABLATION else 'main'
    if name == PRODUCTION:
        notes = ('PRODUCTION: cue-free end to end; calibrator chosen inside each fold among the antisymmetric ones '
                 f'(m4_models.NESTED_CANDIDATES); n_configs_tried = every cue-free pipeline compared: {spec.label()}')
    else:
        notes = NOTES.get(name, f'cue-free end to end: {spec.label()}')
    return kind, spec, notes


def run_trial_experiment(name: str, d: Any, nat: Any) -> None:
    from stage2.lanes import m4_e2e as E
    t0 = time.time()
    X = np.arange(d.n)
    Xe = np.arange(d.n, d.n + nat.n)
    extra = (Xe, nat.trials)
    if name in ORACLE_EXPERIMENTS:
        model = name.split('_')[-1]
        r = L.run_experiment(name, _oracle_fp(model), d.approach, d.trials, extra=(nat.approach, nat.trials),
                             notes=f'hand model {model} on the ORACLE approach windows (no detection)',
                             n_configs_tried=n_configs(name))
    elif name in RETURN_EXPERIMENTS:
        dr = d.subset(d.ret_ok)
        if name == 'm4_ret_app_yaw8':
            r = L.run_experiment(name, _oracle_fp('yaw8'), dr.approach, dr.trials, notes='approach model (yaw8) on the '
                                 'trials with a clean return window (paired base of the return head)', n_configs_tried=1)
        elif name == 'm4_ret_head_yaw8':
            r = L.run_experiment(name, _oracle_fp('yaw8'), dr.ret, dr.trials, notes='return head: yaw8 fitted on the '
                                 'RETURN windows (time-reversed evidence), trials with a clean return', n_configs_tried=1)
        else:
            r = L.run_experiment(name, _sum_fp('yaw8'), {'approach': dr.approach, 'ret': dr.ret}, dr.trials,
                                 notes='approach model + return head: sum of the two logits (pre-specified)',
                                 n_configs_tried=1)
    else:
        details: dict[int, dict[str, Any]] = {}
        kind, spec, notes = experiment_plan(name)
        prep = prepared(d, nat, kind)
        r = L.run_experiment(name, E.make_fit_predict(prep, spec, details), X, d.trials, extra=extra, notes=notes,
                             n_configs_tried=n_configs(name), params={'spec': json.loads(spec.label())})
        save_details(name, details, d, nat, prep)
    print(f'  m4 {name}: accuracy {100 * r.metrics["accuracy"]:.1f} % ({r.leak_check}; {time.time() - t0:.0f} s)',
          flush=True)


def save_details(name: str, details: dict[int, dict[str, Any]], d: Any, nat: Any, prep: Any) -> None:
    """Per recording: the chosen segment, its status and P(approach), forced and answered P(right), and the
    detection outcome against the stored events (evaluation only, after run_experiment)."""
    from stage2.lanes import m4_e2e as E
    ch = pd.DataFrame([details[k] for k in sorted(details)])
    T = pd.concat([d.trials, nat.trials], ignore_index=True)
    oc = E.detection_outcomes(ch, T)
    ch = ch.merge(oc, on='stream', how='left')
    ch['trial_id'] = T['trial_id'].to_numpy()[ch['stream'].to_numpy(int)]
    ch['person_id'] = T['person_id'].to_numpy()[ch['stream'].to_numpy(int)]
    ch['y'] = T['y_right'].astype(int).to_numpy()[ch['stream'].to_numpy(int)]
    ch['platform'] = T['platform'].to_numpy()[ch['stream'].to_numpy(int)]
    ch['set'] = np.where(ch['stream'] < d.n, 'primary', 'natural_use')
    ch['oracle_end_ms'] = T['end_ms'].to_numpy(float)[ch['stream'].to_numpy(int)]
    ch['first_touch_ms'] = pd.to_numeric(T['first_touch_ms'], errors='coerce').to_numpy(float)[ch['stream'].to_numpy(int)]
    keep = ['trial_id', 'person_id', 'set', 'platform', 'y', 'status', 'n_seg', 'p_approach', 'onset_ms', 'end_ms',
            'decided_ms', 'oracle_end_ms', 'first_touch_ms', 'outcome', 'iou', 'onset_error_ms', 'end_error_ms', 'z',
            'p_forced', 'p_right', 'up_z0']
    for c in keep:
        if c not in ch.columns:
            ch[c] = np.nan
    L.export_table(ch[keep], f'{name}_details', tex=False)


# ───────────────────────────── stream-level analyses (same LOSO splits) ─────────────────────────────


def fold_persons(d: Any) -> list[str]:
    return sorted(d.trials['person_id'].astype(str).unique())


def _choose_on(fitted: dict[str, Any], prep: Any, streams: list[dict[str, np.ndarray]]) -> dict[str, pd.DataFrame]:
    """Each fitted pipeline's choice on new streams (segment tables built once per segmentation config)."""
    from stage2.cuefree import CueFreeParams, detect
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    outs, tab_cache = {}, {}
    for k, fp in fitted.items():
        if fp.spec.detector == 'v0':
            v0rows, v0w = [], []
            for j, s in enumerate(streams):
                dets = detect(s['t'], s['gyro'], CueFreeParams())
                if dets:
                    try:
                        w, _ = D.window_for(s, dets[0].onset_ms, dets[0].end_ms, 1.0)
                        v0rows.append({'stream': j, 'found': True, 'onset_ms': dets[0].onset_ms, 'end_ms': dets[0].end_ms})
                        v0w.append(w)
                        continue
                    except ValueError:
                        pass
                v0rows.append({'stream': j, 'found': False, 'onset_ms': np.nan, 'end_ms': np.nan})
                v0w.append(np.full((17, C.N_NORM), np.nan, np.float32))
            outs[k] = fp.choose(prep, list(range(len(streams))), seg_override=(pd.DataFrame(v0rows), np.stack(v0w)))
            continue
        cfg = fp.seg_cfg
        if cfg not in tab_cache:
            tabs = [D.segment_table(s, E.SEG_GRID[cfg]) for s in streams]
            tab_cache[cfg] = (pd.concat([t.assign(stream=j) for j, (t, _) in enumerate(tabs)], ignore_index=True),
                              np.concatenate([w for _, w in tabs]) if tabs else np.zeros((0, 17, C.N_NORM), np.float32))
        outs[k] = fp.choose(prep, list(range(len(streams))), seg_override=tab_cache[cfg])
    return outs


#: the pipelines of the stress test and the trial experiment each one is (its 'recorded' rows must reproduce that
#: result): production, its detector uncalibrated (robust features), the grip-normalised alternative, the detector
#: with the uncapped context features, and v0
STRESS_PIPELINES = {'production': PRODUCTION, 'robust_uncal': BASE_FOR_CALIBRATION, 'grip_uncal': 'm4_v1grip_yaw8',
                    'uncapped': 'm4_v1full_yaw8', 'v0': 'm4_v0_yaw8'}


def stress_pipelines() -> dict[str, Any]:
    """The specs of STRESS_PIPELINES (the trial experiments' own specs)."""
    s = _specs()
    return {k: s[v] for k, v in STRESS_PIPELINES.items()}


#: paired comparisons of the stress test at every condition (family 'stress', Holm-adjusted): (key, A, B); robust and
#: full both uncalibrated (the audit's request), production against v0, the grip alternative against robust
STRESS_PAIRS = (('robust_vs_full', 'robust_uncal', 'uncapped'), ('production_vs_v0', 'production', 'v0'),
                ('grip_vs_robust', 'grip_uncal', 'robust_uncal'))


def compute_stress(d: Any, nat: Any) -> None:
    """Single recordings with the pre-approach stillness cut to keep_s, and their screen-away twins: every
    pipeline's choice on every stream and the stream's rest pose (m4_stress_details.csv), then the summary
    (:func:`summarize_stress`, from that file only)."""
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_sessions as S
    prep = prepared(d, nat, 'main')
    pipes = stress_pipelines()
    conditions = [(str(k), False) for k in ('recorded', *KEEP_S_STRESS)] + [(str(k), True) for k in STRESS_FLIP_KEEPS]
    rows = []
    T = d.trials
    for person in fold_persons(d):
        fitted = {k: E.pipeline_for_person(prep, sp, T, person) for k, sp in pipes.items()}
        idx = np.flatnonzero(T['person_id'].astype(str).to_numpy() == person)
        for keep, flip in conditions:
            streams, evs = [], []
            for i in idx:
                st = label_free_stream(d.stream(int(i)))
                ev = T.iloc[int(i)]
                if keep == 'recorded':
                    s2, rem = st, 0.0
                else:
                    s2, rem = S.stress_stream(st, float(ev['onset_ms']), float(keep))
                if flip:
                    s2 = D.screen_flip_stream(s2)
                streams.append(s2)
                evs.append({c: (float(ev[c]) - rem if pd.notna(ev[c]) else np.nan) for c in D.EVENT_COLUMNS})
            rest = [D.rest_up_z(s, D.DEFAULT, GRIP_REST_MIN_S) for s in streams]     # model-free grip check input
            for name, o in _choose_on(fitted, prep, streams).items():
                for _, r in o.iterrows():
                    ev = evs[int(r['stream'])]
                    if np.isfinite(r.get('onset_ms', np.nan)) and r['status'] != 'none':
                        oc, _iou = D.classify_arrays([r['onset_ms']], [r['end_ms']], [ev['onset_ms']], [ev['end_ms']],
                                                     [ev['end_detected_ms']], [ev['return_onset_ms']], [ev['return_end_ms']])
                        outcome = 'rejected' if r['status'] == 'no_approach' else str(oc[0])
                    else:
                        outcome = 'none'
                    i = int(idx[int(r['stream'])])
                    rows.append({'pipeline': name, 'keep': keep, 'flip': bool(flip), 'trial_key': T['trial_key'].iloc[i],
                                 'person_id': person, 'platform': T['platform'].iloc[i], 'y': int(T['y_right'].iloc[i]),
                                 'status': r['status'], 'outcome': outcome, 'p_right': float(r['p_right']),
                                 'p_forced': float(r['p_forced']), 'up_z0': float(r.get('up_z0', np.nan)),
                                 'up_z_rest': float(rest[int(r['stream'])])})
    L.export_table(pd.DataFrame(rows), 'm4_stress_details', tex=False)
    summarize_stress()


def grip_check(g: pd.DataFrame) -> np.ndarray:
    """The inference page's rest-pose grip check (m4_config.GRIP_GATE_UPZ) on stress rows: True where it stops the
    stream ("turn the screen towards you"; scored as an abstention)."""
    return (g['up_z_rest'].to_numpy(float) < GRIP_GATE_UPZ)


def summarize_stress() -> None:
    """tables/m4/m4_stress.json from m4_stress_details.csv only: per pipeline, stillness and grip the half-credit
    accuracy with its CI, coverage, outcome shares, the rest-pose grip check's share and the accuracy behind it, the
    first remedy's per-segment check for the record ('gate_seg': the chosen segment's own start pose), the paired
    comparisons (family 'stress', Holm) and the consistency check against the trial experiments."""
    from stage2.lanes import m4_stats as ST
    df = pd.read_csv(L.tab_dir / 'm4_stress_details.csv', dtype={'keep': str})
    df['flip'] = df['flip'].astype(bool)
    summ = []
    for (pipe, keep, flip), g in df.groupby(['pipeline', 'keep', 'flip'], sort=False):
        ok = (g['status'] == 'ok').to_numpy()
        sc = ST.scores(g[['person_id', 'y', 'p_right']], p_forced=g['p_forced'], answered=ok)
        gate = grip_check(g)
        gated = g[['person_id', 'y']].assign(p_right=np.where(gate, 0.5, g['p_right']))
        gci = ST.person_ci(gated, ST.half_credit)
        ans_g = ok & ~gate
        gate_seg = ok & (g['up_z0'].to_numpy(float) < GRIP_GATE_UPZ)
        ans_s = ok & ~gate_seg
        wrong = ok & (ST.correct(g['y'], g['p_right']) == 0)          # answered, and answered wrongly
        summ.append({'pipeline': pipe, 'keep': keep, 'flip': bool(flip), 'n': len(g), 'accuracy': sc['accuracy_half'],
                     'wrong_share': float(wrong.mean()), 'wrong_share_gated': float((wrong & ~gate).mean()),
                     'wrong_share_gated_seg': float((wrong & ~gate_seg).mean()),
                     'acc_lo': sc['accuracy_half_ci'][0], 'acc_hi': sc['accuracy_half_ci'][1], 'coverage': sc['coverage'],
                     'acc_answered': sc['accuracy_answered'], 'acc_forced': sc['accuracy_forced'],
                     'gate_share': float(gate.mean()), 'acc_gated': ST.half_credit(gated['y'], gated['p_right']),
                     'acc_gated_lo': gci[0], 'acc_gated_hi': gci[1], 'coverage_gated': float(ans_g.mean()),
                     'acc_answered_gated': ST.half_credit(g.loc[ans_g, 'y'], g.loc[ans_g, 'p_right']) if ans_g.any() else None,
                     'share_return_gated': float((ans_g & (g['outcome'] == 'return').to_numpy()).mean()),
                     'gate_seg_share': float(gate_seg.mean()),
                     'acc_gated_seg': ST.half_credit(g['y'], np.where(gate_seg, 0.5, g['p_right'])),
                     'acc_answered_gated_seg': ST.half_credit(g.loc[ans_s, 'y'], g.loc[ans_s, 'p_right']) if ans_s.any() else None,
                     **{f'share_{o}': float((g['outcome'] == o).mean()) for o in
                        ('approach', 'approach_partial', 'earlier_movement', 'return', 'later_movement', 'rejected', 'none')}})
    pairs = []
    for (keep, flip), g in df.groupby(['keep', 'flip'], sort=False):
        for key, a, b in STRESS_PAIRS:
            ga, gb = g[g['pipeline'] == a], g[g['pipeline'] == b]
            if len(ga) and len(gb):
                pairs.append({'key': key, 'a': a, 'b': b, 'keep': keep, 'flip': bool(flip), **ST.paired(ga, gb)})
        gp = g[g['pipeline'] == 'production']
        if len(gp):                                      # the production pipeline with the grip check against without
            gg = gp.assign(p_right=np.where(grip_check(gp), 0.5, gp['p_right']))
            pairs.append({'key': 'grip_check', 'a': 'production + grip check', 'b': 'production', 'keep': keep,
                          'flip': bool(flip), **ST.paired(gg, gp)})
    adj = ST.holm([p['p_acc'] for p in pairs])
    for p, h in zip(pairs, adj):
        p['p_acc_holm'] = h
    # the grip check's purpose: fewer recordings answered WRONGLY (production with the check minus without, at every
    # condition; family 'grip', Holm)
    wrong_pairs = []
    for (keep, flip), g in df[df['pipeline'] == 'production'].groupby(['keep', 'flip'], sort=False):
        ok = (g['status'] == 'ok').to_numpy()
        wrong = ok & (ST.correct(g['y'], g['p_right']) == 0)
        wrong_pairs.append({'keep': keep, 'flip': bool(flip),
                            **ST.paired_shares(g['person_id'], wrong & ~grip_check(g), wrong)})
    for p, h in zip(wrong_pairs, ST.holm([p['p'] for p in wrong_pairs])):
        p['p_holm'] = h
    check = {}
    for pipe, name in STRESS_PIPELINES.items():
        if L.result_path(name).exists():
            saved = L.load_result(name).predictions[['trial_key', 'p_right']]
            g = df[(df['pipeline'] == pipe) & (df['keep'] == 'recorded') & ~df['flip']].merge(saved, on='trial_key',
                                                                                           suffixes=('', '_saved'))
            check[pipe] = float(np.max(np.abs(g['p_right'] - g['p_right_saved']))) if len(g) else None
    write_json('m4_stress', {'summary': summ, 'pairs': pairs, 'keep_s': list(KEEP_S_STRESS),
                             'flip_keeps': [str(k) for k in STRESS_FLIP_KEEPS], 'production': PRODUCTION,
                             'pipelines': dict(STRESS_PIPELINES), 'grip_gate_upz': GRIP_GATE_UPZ,
                             'grip_check_wrong': wrong_pairs,
                             'family': 'stress: every STRESS_PAIRS comparison and the grip check at every condition (Holm)',
                             'family_grip': 'grip: share answered wrongly, production with the grip check minus without, '
                                           'at every condition (Holm)',
                             'recorded_matches_trial_results_max_abs_dp': check})


def compute_sessions(d: Any, nat: Any) -> None:
    """Synthetic multi-attempt sessions: attempts found, k-attempt aggregation, stopping rule, calibration.

    Per outer fold (held-out person): the fold's fitted production pipeline, its recording-fitted aggregate scale
    and its session-fitted scale (cross-fitted on the TRAINING persons' sessions under the production attempt
    rule, ``m4_aggregate``); the held-out person's sessions at every junction stillness, under both attempt rules,
    every aggregation and threshold. Each accepted attempt is labelled from the stored events (evaluation only),
    which gives the attainable curve when only the true approaches are kept."""
    from stage2.lanes import m4_aggregate as A
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_online as O
    from stage2.lanes import m4_sessions as S
    prep = prepared(d, nat, 'main')
    spec = production_spec()
    T = d.trials
    ref_keep = reference_keep_s()
    prod_rule = next(r for r, pair in SESSION_RULES.items() if pair == PRODUCTION_PAIRING)
    rows_k, rows_stop, rows_ans, rows_det, att_rows, scales = [], [], [], [], [], []
    timeline = None
    for person in fold_persons(d):
        fp, agg_s, sess_att = A.session_aggregate_for_person(prep, spec, T, person)
        aggs = {'scale_recordings': fp.agg, 'scale_sessions': agg_s, 'mean': None}
        per_k = A.fit_session_scale(sess_att, form='per_k')          # for the record (non-monotone, not used)
        for kind, a in (('scale_recordings', fp.agg), ('scale_sessions', agg_s), ('scale_sessions_per_k', per_k)):
            scales += [{'person_id': person, 'aggregation': kind, 'k': int(k), 's': float(s)} for k, s in a.s.items()]
        scales += [{'person_id': person, 'aggregation': 'crossfit_sessions', 'k': 0,
                    's': float(sess_att.groupby(['person_id', 'session']).ngroups)},
                   {'person_id': person, 'aggregation': 'crossfit_attempts', 'k': 0, 's': float(len(sess_att))},
                   {'person_id': person, 'aggregation': 'power_exponent_b', 'k': 0, 's': float(getattr(agg_s, 'b', np.nan))}]
        seg_p = E.SEG_GRID[fp.seg_cfg]
        for keep in KEEP_S_SESSIONS:
            sess = A.sessions_with_tables(T, np.arange(d.n), prep.streams.__getitem__, person, seg_p, keep,
                                          SESSION_RECORDINGS, SESSION_STRIDE, source=prep.fingerprint())
            for si, (ss, tab, W) in enumerate(sess):
                sid = f'{person}|{ss.y}|{si}'
                base = {'person_id': person, 'y': ss.y, 'keep_s': keep, 'session': si, 'sid': sid,
                        'n_true': int(len(ss.events))}
                for rule, pair in SESSION_RULES.items():
                    a_r = fp.attempts(tab, W, pair_returns=pair)
                    a_r = a_r.sort_values('decided_ms').reset_index(drop=True) if len(a_r) else a_r
                    kinds = S.attempt_kinds(a_r, ss.events)
                    rows_det.append({**base, 'rule': rule, **S.session_detection(a_r, ss.events)})
                    variants = [(rule, a_r)]
                    if rule == prod_rule:                    # the ceiling: only the true approaches kept
                        keep_true = np.array([k in ('approach', 'approach_partial') for k in kinds], bool)
                        variants.append(('true_only', a_r[keep_true].reset_index(drop=True) if len(a_r) else a_r))
                    for rname, att in variants:
                        for r in O.k_curve_rows(att, {'scale_recordings': fp.agg, 'scale_sessions': agg_s},
                                                range(1, K_MAX + 1)):
                            rows_k.append({**base, 'rule': rname, **r})
                    for how in STOP_AGGREGATIONS:
                        for ps in P_STOP_GRID:
                            dec = O.stopping_rule(a_r, aggs[how], p_stop=ps, k_max=K_MAX, p_answer=P_ANSWER)
                            rows_stop.append({**base, 'rule': rule, 'aggregation': how, 'p_stop': ps, **dec,
                                              'true_touches': _true_touches(ss.events, dec['decided_ms'])})
                        if rule == prod_rule:
                            for pa in P_ANSWER_GRID:
                                dec = O.stopping_rule(a_r, aggs[how], p_stop=P_STOP, k_max=K_MAX, p_answer=pa)
                                rows_ans.append({**base, 'rule': rule, 'aggregation': how, 'p_answer': pa, **dec})
                    if rule == prod_rule:
                        for j, a in a_r.iterrows():
                            att_rows.append({**base, 'attempt': int(j), 'p_approach': a['p_approach'], 'z': a['z'],
                                             'decided_ms': a['decided_ms'], 'kind': kinds[j]})
                        if timeline is None and keep == ref_keep and len(a_r) >= 3:
                            timeline = session_timeline(ss, tab, fp, a_r, kinds, seg_p, aggs)
    L.export_table(pd.DataFrame(rows_k), 'm4_sessions_krows', tex=False)
    L.export_table(pd.DataFrame(rows_stop), 'm4_sessions_stop', tex=False)
    L.export_table(pd.DataFrame(rows_ans), 'm4_sessions_answer', tex=False)
    L.export_table(pd.DataFrame(rows_det).drop(columns=['kinds']), 'm4_sessions_detection', tex=False)
    L.export_table(pd.DataFrame(att_rows), 'm4_sessions_attempts', tex=False)
    L.export_table(pd.DataFrame(scales), 'm4_sessions_scales', tex=False)
    write_json('m4_timeline', timeline or {})
    summarize_sessions()


def _true_touches(ev: pd.DataFrame, decided_ms: float) -> int:
    """True approaches completed by the decision (all of the session's when it never decided)."""
    if decided_ms is None or not math.isfinite(decided_ms):
        return int(len(ev))
    return int((ev['end_ms'].to_numpy(float) <= decided_ms + 1e-6).sum())


def session_timeline(ss: Any, tab: pd.DataFrame, fp: Any, att: pd.DataFrame, kinds: list[str], seg_p: Any,
                     aggs: dict[str, Any]) -> dict[str, Any]:
    """The data of the timeline figure for one session: activity, candidates with class probabilities, the
    running aggregate after each accepted attempt and the stopping decision under every aggregation, the true
    approaches and returns, and the kind of every accepted attempt (from the stored events)."""
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_online as O
    st = ss.stream
    s = D.activity(st['t'], st['gyro'], seg_p)
    P = fp.segment_proba(tab) if len(tab) else {c: np.zeros(0) for c in D.CLASSES}
    cands = [{'onset_ms': float(r['onset_ms']), 'end_ms': float(r['end_ms']), 'decided_ms': float(r['decided_ms']),
              **{f'p_{c}': float(P[c][i]) for c in D.CLASSES}} for i, r in tab.reset_index(drop=True).iterrows()]
    z = att['z'].to_numpy(float)
    running = {how: [{'k': k + 1, 'decided_ms': float(att['decided_ms'].iloc[k]), 'p_right': O.aggregate(z[:k + 1], agg)}
                     for k in range(len(att))] for how, agg in aggs.items()}
    decisions = {how: {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                       for k, v in O.stopping_rule(att, agg, p_stop=P_STOP, k_max=K_MAX, p_answer=P_ANSWER).items()}
                 for how, agg in aggs.items()}
    step = 3                                              # every 3rd sample is enough for a plot
    return {'hand': 'right' if ss.y == 1 else 'left', 'keep_s': ss.keep_s, 'person_id': ss.person_id,
            't_s': (st['t'][::step] / 1000.0).round(3).tolist(), 'activity': np.round(s[::step], 2).tolist(),
            'stream_end_ms': float(st['t'][-1]), 'lo_dps': seg_p.lo_dps, 'hi_dps': seg_p.hi_dps, 'candidates': cands,
            'attempts': [{'decided_ms': float(a['decided_ms']), 'onset_ms': float(a['onset_ms']),
                          'end_ms': float(a['end_ms']), 'kind': k} for (_, a), k in zip(att.iterrows(), kinds)],
            'running': running, 'decisions': decisions, 'p_stop': P_STOP, 'k_max': K_MAX,
            'true_approaches': ss.events[['onset_ms', 'end_ms']].to_dict(orient='records'),
            'true_returns': ss.events[['return_onset_ms', 'return_end_ms']].replace({np.nan: None}).to_dict(orient='records')}


def summarize_sessions() -> None:
    """Session-level metrics with person-bootstrap CIs (from the saved session tables only)."""
    from stage2 import cv
    from stage2.lanes import m4_online as O
    from stage2.lanes import m4_stats as ST
    k = pd.read_csv(L.tab_dir / 'm4_sessions_krows.csv')
    st = pd.read_csv(L.tab_dir / 'm4_sessions_stop.csv')
    ans = pd.read_csv(L.tab_dir / 'm4_sessions_answer.csv')
    det = pd.read_csv(L.tab_dir / 'm4_sessions_detection.csv')
    sc = pd.read_csv(L.tab_dir / 'm4_sessions_scales.csv')
    ref_keep = reference_keep_s()
    prod_rule = next(r for r, pair in SESSION_RULES.items() if pair == PRODUCTION_PAIRING)
    rows = []
    for (rule, keep, kk), g in k.groupby(['rule', 'keep_s', 'k']):
        acc_ci = ST.person_ci(g[['person_id', 'y']].assign(p_right=g['p_mean']), ST.half_credit)
        for how in O.AGGREGATIONS:                # the accuracy is the sign of the mean logit for every aggregation
            pr = g[['person_id', 'y']].assign(p_right=g[f'p_{how}'])
            m = cv.metrics(pr['y'], pr['p_right'])
            rows.append({'rule': rule, 'keep_s': keep, 'k': int(kk), 'aggregation': how, 'n_sessions': len(g),
                         'n_persons': int(g['person_id'].nunique()), 'accuracy': ST.half_credit(pr['y'], pr['p_right']),
                         'acc_lo': acc_ci[0], 'acc_hi': acc_ci[1], 'log_loss': m['log_loss'], 'ece': m['ece'],
                         'brier': m['brier'], 'share_k_available': float((g['n_available'] >= kk).mean()),
                         'n_k_available': int((g['n_available'] >= kk).sum())})
    stop = []
    for (rule, keep, how, ps), g in st.groupby(['rule', 'keep_s', 'aggregation', 'p_stop']):
        a = g['answered'].astype(bool)
        pr = g[['person_id', 'y', 'p_right']]
        conf = g['stopped'] == 'confident'
        acc_ci = ST.person_ci(pr, ST.half_credit)
        conf_ci = ST.person_ci(pr[conf], ST.half_credit) if conf.any() else (math.nan, math.nan)
        m = cv.metrics(pr['y'], pr['p_right'])
        stop.append({'rule': rule, 'keep_s': keep, 'aggregation': how, 'p_stop': ps, 'n_sessions': len(g),
                     'n_persons': int(g['person_id'].nunique()),
                     'accuracy': ST.half_credit(pr['y'], pr['p_right']), 'acc_lo': acc_ci[0], 'acc_hi': acc_ci[1],
                     'mean_attempts': float(g['n_used'].mean()), 'mean_true_touches': float(g['true_touches'].mean()),
                     'coverage': float(a.mean()),
                     'acc_answered': ST.half_credit(g.loc[a, 'y'], g.loc[a, 'p_right']) if a.any() else math.nan,
                     'share_confident_stop': float(conf.mean()),
                     'acc_confident_stop': ST.half_credit(g.loc[conf, 'y'], g.loc[conf, 'p_right']) if conf.any() else math.nan,
                     'acc_conf_lo': conf_ci[0], 'acc_conf_hi': conf_ci[1],
                     'mean_conf_claimed': float(np.maximum(g.loc[conf, 'p_right'], 1 - g.loc[conf, 'p_right']).mean())
                     if conf.any() else math.nan,
                     'share_no_attempt': float((g['stopped'] == 'no_attempt').mean()),
                     'share_k_max': float((g['stopped'] == 'k_max').mean()),
                     'share_stream_end': float((g['stopped'] == 'stream_end').mean()),
                     'ece': m['ece'], 'log_loss': m['log_loss'], 'brier': m['brier'], 'roc_auc': m['roc_auc']})
    answer = []
    for (how, pa), g in ans[ans['keep_s'] == ref_keep].groupby(['aggregation', 'p_answer']):
        a = g['answered'].astype(bool)
        answer.append({'aggregation': how, 'p_answer': pa, 'n_sessions': len(g), 'coverage': float(a.mean()),
                       'acc_answered': ST.half_credit(g.loc[a, 'y'], g.loc[a, 'p_right']) if a.any() else math.nan,
                       'accuracy': ST.half_credit(g['y'], np.where(a, g['p_right'], 0.5))})
    dd = det.groupby(['rule', 'keep_s']).agg(n_sessions=('n_true', 'size'), true=('n_true', 'sum'), found=('n_found', 'sum'),
                                   attempts=('n_attempts', 'sum'), false_return=('n_false_return', 'sum'),
                                   false_other=('n_false_other', 'sum'), false_other_before=('n_false_other_before', 'sum'),
                                   false_other_after=('n_false_other_after', 'sum')).reset_index()
    dd['recall'] = dd['found'] / dd['true']
    dd['false_per_session'] = (dd['false_return'] + dd['false_other']) / dd['n_sessions']
    dd['precision'] = 1 - (dd['false_return'] + dd['false_other']) / dd['attempts'].clip(lower=1)
    s_med = (sc[sc['k'] > 0].groupby(['aggregation', 'k'])['s'].agg(['median', 'min', 'max']).reset_index()
             .rename(columns={'median': 's_median', 'min': 's_min', 'max': 's_max'}))
    per_k = sc[sc['aggregation'] == 'scale_sessions_per_k'].pivot(index='person_id', columns='k', values='s')
    nonmono = int((per_k.diff(axis=1).iloc[:, 1:] < -1e-9).any(axis=1).sum()) if len(per_k) else 0
    fold_stat = {kind: float(sc.loc[sc['aggregation'] == kind, 's'].median())
                 for kind in ('crossfit_sessions', 'crossfit_attempts', 'power_exponent_b')
                 if (sc['aggregation'] == kind).any()}
    # single-attempt accuracy by the kind of the accepted attempt (production rule, reference stillness)
    att = pd.read_csv(L.tab_dir / 'm4_sessions_attempts.csv')
    att = att[att['keep_s'] == ref_keep]
    by_kind = []
    for kind, g in att.groupby('kind'):
        by_kind.append({'kind': kind, 'n': int(len(g)), 'share': float(len(g) / max(len(att), 1)),
                        'accuracy': ST.half_credit(g['y'], 1.0 / (1.0 + np.exp(-g['z'].to_numpy(float))))})
    write_json('m4_sessions', {'k_curve': rows, 'stopping': stop, 'answer': answer,
                               'detection': dd.to_dict(orient='records'), 'by_kind': by_kind,
                               'scales': s_med.to_dict(orient='records'), 'per_k_nonmonotone_folds': nonmono,
                               'n_folds': int(sc['person_id'].nunique()), 'fold_medians': fold_stat,
                               'p_stop': P_STOP, 'k_max': K_MAX, 'session_recordings': SESSION_RECORDINGS,
                               'session_stride': SESSION_STRIDE, 'production': PRODUCTION, 'production_rule': prod_rule,
                               'stop_aggregations': list(STOP_AGGREGATIONS), 'reference_keep_s': ref_keep,
                               'p_answer': P_ANSWER, 'n_configs_sessions': N_CONFIGS_SESSIONS,
                               'paired': session_pairs(st, ref_keep, prod_rule)})


def session_pairs(st: pd.DataFrame, ref_keep: float, prod_rule: str) -> list[dict[str, Any]]:
    """Paired comparisons of the stopping rule's answer on the same sessions (reference stillness, P_STOP; family
    'sessions', Holm-adjusted): the two attempt rules under every aggregation, and the aggregations pairwise under
    the production attempt rule."""
    from stage2.lanes import m4_stats as ST
    g = st[(st['keep_s'] == ref_keep) & np.isclose(st['p_stop'], P_STOP)]
    out = []
    other_rule = next(r for r in SESSION_RULES if r != prod_rule)
    for how in STOP_AGGREGATIONS:
        a = g[(g['rule'] == prod_rule) & (g['aggregation'] == how)]
        b = g[(g['rule'] == other_rule) & (g['aggregation'] == how)]
        out.append({'key': f'rule_{prod_rule}_vs_{other_rule}_{how}', 'a': f'{prod_rule} ({how})',
                    'b': f'{other_rule} ({how})', **ST.paired(a, b, key='sid')})
    hs = list(STOP_AGGREGATIONS)
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            a = g[(g['rule'] == prod_rule) & (g['aggregation'] == hs[i])]
            b = g[(g['rule'] == prod_rule) & (g['aggregation'] == hs[j])]
            out.append({'key': f'agg_{hs[i]}_vs_{hs[j]}', 'a': hs[i], 'b': hs[j], **ST.paired(a, b, key='sid')})
    for p, h_acc, h_ll in zip(out, ST.holm([p['p_acc'] for p in out]), ST.holm([p['p_ll'] for p in out])):
        p['p_acc_holm'], p['p_ll_holm'] = h_acc, h_ll
    return out


def compute_calibration_summary() -> None:
    """Calibration of one touch (every calibrator on the production detector) and of several touches (k = 3 and
    the stopping rule's answer under every aggregation): metrics, ECE with its perfect-calibration null band,
    calibration slope / intercept with person-bootstrap CIs, reliability with person-bootstrap bands, per platform,
    and paired log-loss / Brier differences (from the saved results and session tables only). Then the declared
    rule picks the production aggregation (m4_config.AGGREGATION_RULE)."""
    from stage2.lanes import m4_models as M
    from stage2.lanes import m4_online as O
    from stage2.lanes import m4_stats as ST
    out: dict[str, Any] = {'candidates': [], 'production': PRODUCTION, 'base': BASE_FOR_CALIBRATION}
    names = [(c, BASE_FOR_CALIBRATION if c == 'none' else f'm4_cal_{c}') for c in CALIBRATION_CANDIDATES]
    det = pd.read_csv(_details_path(BASE_FOR_CALIBRATION))
    plat = dict(zip(det['trial_key'], det['platform']))
    base_pred = L.load_result(BASE_FOR_CALIBRATION).predictions
    for c, name in names:
        if not L.result_path(name).exists():
            continue
        r = L.load_result(name)
        pred = r.predictions.assign(platform=lambda x: x['trial_key'].map(plat))
        entry = {'calibration': c, 'name': name, **ST.calibration_summary(pred),
                 'reliability': reliability_bands(pred), 'per_platform': {},
                 'antisymmetric': bool(M.CALIBRATORS[c].antisymmetric)}
        for pl, g in pred.groupby('platform'):
            entry['per_platform'][pl] = {**ST.calibration_summary(g),
                                         'accuracy_half_ci': list(ST.person_ci(g, ST.half_credit))}
        fold_cal = [json.loads(p).get('calibration') or {} for p in r.predictions.groupby('fold')['params'].first()]
        devs = [fc['mirror_dev_max'] for fc in fold_cal if 'mirror_dev_max' in fc]
        entry['mirror_dev_max'] = float(max(devs)) if devs else (0.0 if c == 'none' else None)
        entry['mirror_dev_mean'] = (float(np.mean([fc['mirror_dev_mean'] for fc in fold_cal if 'mirror_dev_mean' in fc]))
                                    if devs else (0.0 if c == 'none' else None))
        if c != 'none':
            entry['vs_none'] = ST.paired(pred, base_pred)
        if c == 'nested':
            chosen = pd.Series([fc.get('chosen') for fc in fold_cal])
            entry['chosen_per_fold'] = chosen.value_counts().to_dict()
        dd = pd.read_csv(_details_path(name)) if _details_path(name).exists() else None
        if dd is not None:
            dp = dd[dd['set'] == 'primary']
            ok = dp['status'] == 'ok'
            entry['answered'] = {'coverage': float(ok.mean()), 'accuracy': ST.half_credit(dp.loc[ok, 'y'], dp.loc[ok, 'p_right'])}
        out['candidates'].append(entry)
    fam = [e for e in out['candidates'] if 'vs_none' in e]
    for e, h_ll, h_acc in zip(fam, ST.holm([e['vs_none']['p_ll'] for e in fam]), ST.holm([e['vs_none']['p_acc'] for e in fam])):
        e['vs_none']['p_ll_holm'], e['vs_none']['p_acc_holm'] = h_ll, h_acc
    out['family_calibrators'] = 'every calibrator minus none on the same recordings (Holm over the calibrators)'
    ref_keep = reference_keep_s()
    prod_rule = next(r for r, pair in SESSION_RULES.items() if pair == PRODUCTION_PAIRING)
    out['sessions_keep_s'] = ref_keep
    out['sessions_rule'] = prod_rule
    k = pd.read_csv(L.tab_dir / 'm4_sessions_krows.csv')
    k3 = k[(k['k'] == 3) & (k['keep_s'] == ref_keep) & (k['rule'] == prod_rule)]
    out['k3'] = {}
    for how in O.AGGREGATIONS:
        pr = k3[['person_id', 'y']].assign(p_right=k3[f'p_{how}'])
        out['k3'][how] = {**ST.calibration_summary(pr), 'reliability': reliability_bands(pr)}
    st = pd.read_csv(L.tab_dir / 'm4_sessions_stop.csv')
    st = st[np.isclose(st['p_stop'], P_STOP) & (st['keep_s'] == ref_keep) & (st['rule'] == prod_rule)]
    plat_of = det[det['set'] == 'primary'].groupby('person_id')['platform'].agg(lambda s: s.mode().iloc[0])
    st = st.assign(platform=st['person_id'].map(plat_of))
    out['stop'], out['stop_per_platform'] = {}, []
    for how, g in st.groupby('aggregation', sort=False):
        pr = g[['person_id', 'y', 'p_right']]
        conf = g['stopped'] == 'confident'
        out['stop'][how] = {**ST.calibration_summary(pr), 'reliability': reliability_bands(pr),
                            'mean_attempts': float(g['n_used'].mean()), 'mean_true_touches': float(g['true_touches'].mean()),
                            'n_params': AGGREGATION_N_PARAMS[how]}
        for pl, gp in g.groupby('platform', sort=True):
            m = ST.calibration_summary(gp[['person_id', 'y', 'p_right']])
            out['stop_per_platform'].append({'aggregation': how, 'platform': pl, 'mean_attempts': float(gp['n_used'].mean()),
                                             **{kk: m[kk] for kk in ('n', 'n_persons', 'accuracy_half', 'ece', 'ece_null_lo',
                                                                     'ece_null_hi', 'log_loss', 'brier', 'slope')}})
        if conf.any():
            out['stop'][how]['confident'] = {'n': int(conf.sum()), 'accuracy': ST.half_credit(g.loc[conf, 'y'], g.loc[conf, 'p_right']),
                                             'accuracy_ci': list(ST.person_ci(pr[conf], ST.half_credit)),
                                             'claimed': float(np.maximum(pr.loc[conf, 'p_right'], 1 - pr.loc[conf, 'p_right']).mean())}
    out['production_aggregation'] = production_aggregation(out)
    out['aggregation_rule'] = AGGREGATION_RULE
    out['p_stop'] = P_STOP
    write_json('m4_calibration', out)


def reliability_bands(pred: pd.DataFrame, n_bins: int = C.ECE_BINS, n_boot: int = N_BOOT) -> list[dict[str, float]]:
    """Equal-mass reliability bins of P(right) with a person-bootstrap 95 % band of each bin's observed share
    (bin edges fixed on the full predictions; persons resampled with replacement)."""
    p = pred['p_right'].to_numpy(float)
    y = pred['y'].to_numpy(float)
    if p.size < n_bins:
        return []
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.searchsorted(edges[1:-1], p, side='right')
    persons = pred['person_id'].astype(str).to_numpy()
    uniq = np.unique(persons)
    rows_of = {u: np.flatnonzero(persons == u) for u in uniq}
    rng = np.random.default_rng(C.SEED)
    boots = np.full((n_boot, n_bins), np.nan)
    for i in range(n_boot):
        sel = np.concatenate([rows_of[u] for u in rng.choice(uniq, uniq.size, replace=True)])
        cnt = np.bincount(b[sel], minlength=n_bins)
        tot = np.bincount(b[sel], weights=y[sel], minlength=n_bins)
        boots[i] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    out = []
    for j in range(n_bins):
        m = b == j
        if not m.any():
            continue
        out.append({'bin': j, 'n': int(m.sum()), 'p_mean': float(p[m].mean()), 'y_mean': float(y[m].mean()),
                    'y_lo': float(np.nanquantile(boots[:, j], 0.025)), 'y_hi': float(np.nanquantile(boots[:, j], 0.975))})
    return out


def compute_latency(d: Any) -> None:
    """Online decision latency of the production pipeline on the recordings (from its saved details), and how
    much later the RETURN's evidence would arrive (the stored end of the return minus the stored end of the
    approach, recordings with a clean return: the price of adding the return head to the decision)."""
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    det = pd.read_csv(_details_path(PRODUCTION))
    dp = det[(det['set'] == 'primary') & (det['status'] == 'ok') & (det['outcome'] == 'approach')]
    lat_end = (dp['decided_ms'] - dp['oracle_end_ms']).dropna()
    lat_touch = (dp['decided_ms'] - dp['first_touch_ms']).dropna()
    q = lambda s: {'n': int(len(s)), 'median': float(s.median()), 'p25': float(s.quantile(.25)),  # noqa: E731
                   'p75': float(s.quantile(.75)), 'p90': float(s.quantile(.9))}
    T = d.trials
    ret_wait = (pd.to_numeric(T['return_end_ms'], errors='coerce')
                - pd.to_numeric(T['end_ms'], errors='coerce'))[np.asarray(d.ret_ok, bool)].dropna()
    write_json('m4_latency', {'decision_minus_oracle_end_ms': q(lat_end), 'decision_minus_first_touch_ms': q(lat_touch),
                              'end_error_ms': q(dp['end_error_ms'].dropna()),
                              'onset_error_ms': q(dp['onset_error_ms'].dropna()),
                              'return_end_minus_approach_end_ms': q(ret_wait),
                              'lookahead_s': {k: p.lookahead_s for k, p in E.SEG_GRID.items()},
                              'causal_delay_ms': D.causal_delay_ms(D.DEFAULT),
                              'per_trial': pd.DataFrame({'end': dp['decided_ms'] - dp['oracle_end_ms'],
                                                         'touch': dp['decided_ms'] - dp['first_touch_ms']}
                                                        ).to_dict(orient='list')})


def compute_demo(d: Any, nat: Any) -> None:
    """predict_stream on a few recordings and one synthetic session of a held-out person (the first person
    in id order with both hands), with the pipeline of the fold that held that person out."""
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_production as PR
    from stage2.lanes import m4_sessions as S
    prep = prepared(d, nat, 'main')
    T = d.trials
    person = fold_persons(d)[0]
    tr, y, g, m, seed = E.fold_inputs(T, person)
    fp = E.get_pipeline(prep, production_spec(), tr, y, g, m, seed)
    how = production_aggregation()
    model = PR.production_model(fp, agg=production_aggregate(prep, fp, tr, y, g, m, seed, how), aggregation=how,
                                trained_on=tuple(sorted(set(g.tolist()))))
    idx = np.flatnonzero(T['person_id'].astype(str).to_numpy() == person)
    picks = [int(idx[np.flatnonzero(T['y_right'].to_numpy()[idx] == h)[0]]) for h in (0, 1)]
    from stage2.lanes import m4_detect as D
    rows = []

    def run(what: str, st: dict[str, np.ndarray], key: str, hand: str) -> None:
        out = PR.predict_stream(st['t'], st['f'], st['acc'], st['gyro'], model)
        rows.append({'what': what, 'trial_key': key, 'true_hand': hand, 'percent_right': out['percent_right'],
                     'answered': out['answered'], 'status': out['status'], 'n_attempts': len(out['attempts']),
                     'n_used': out['n_attempts_used'], 'n_segments': len(out['segments']),
                     'rest_up_z': out['rest_up_z'], 'grip_warning': out['grip_warning']})
    for i in picks:
        run('one recording', label_free_stream(d.stream(i)), T['trial_key'].iloc[i], T['hand'].iloc[i])
    i = picks[0]                                   # the same recording held with the screen away (grip check)
    run('one recording, screen-away twin', D.screen_flip_stream(label_free_stream(d.stream(i))), T['trial_key'].iloc[i],
        T['hand'].iloc[i])
    ss_all = S.build_sessions(T, d.stream, n_attempts=3, keep_s=reference_keep_s(), persons=[person])
    for h in (0, 1):
        ss = [s for s in ss_all if s.y == h][0]
        run('synthetic session of 3 touches', ss.stream, '', 'right' if h else 'left')
    write_json('m4_demo', {'person': person, 'model': model.describe(), 'rows': rows})


# ───────────────────────────── steps ─────────────────────────────

DERIVED = ('m4_detection', 'm4_latency', 'm4_sessions', 'm4_stress', 'm4_calibration', 'm4_demo', 'm4_breakdown',
           'm4_segfeat', 'm4_natural', 'm4_paired', 'm4_scores', 'm4_replay')


def compute(force: bool = False, only: list[str] | None = None) -> None:
    remove_obsolete()
    todo = [n for n in TRIAL_EXPERIMENTS if L.should_run(n, force, only)]
    derived = [n for n in DERIVED if selected(n, only) and (force or todo or not file_current(L.tab_dir / f'{n}.json'))]
    if not todo and not derived:
        print('m4: results current; skipped (--force recomputes)', flush=True)
        return
    from stage2.data import load_cache
    d, nat = load_cache(), load_cache(('natural_use',))
    late = (*PROD_VARIANTS, *LOOKAHEAD_ABLATION)          # other prepared data: after the main experiments
    for name in [n for n in TRIAL_EXPERIMENTS if n in todo and n not in late]:
        run_trial_experiment(name, d, nat)
    for name in [n for n in late if n in todo]:
        run_trial_experiment(name, d, nat)
    steps = {'m4_detection': lambda: compute_detection_summary(),
             'm4_calibration': lambda: compute_calibration_summary(),
             'm4_latency': lambda: compute_latency(d),
             'm4_stress': lambda: compute_stress(d, nat),
             'm4_sessions': lambda: compute_sessions(d, nat),
             'm4_demo': lambda: compute_demo(d, nat),
             'm4_breakdown': lambda: compute_breakdown(d, nat),
             'm4_segfeat': lambda: compute_segment_features(d, nat),
             'm4_natural': lambda: compute_natural(d, nat),
             'm4_paired': lambda: compute_paired(),
             'm4_scores': lambda: compute_scores(d, nat),
             'm4_replay': lambda: compute_replay(d, nat)}
    for name in DERIVED:
        if name in derived:
            t0 = time.time()
            steps[name]()
            print(f'  m4 {name}: {time.time() - t0:.0f} s', flush=True)
    rebuild_registry()
    write_finalists()


def remove_obsolete() -> None:
    """Result and detail files of experiments an earlier version of the lane ran (``OBSOLETE``)."""
    for name in OBSOLETE:
        for p in (L.result_path(name), _details_path(name)):
            if p.exists():
                p.unlink()


def compute_detection_summary() -> None:
    """Detection outcome shares and bound errors of the chosen segment for every detector (saved details)."""
    rows = []
    for name in (*DETECTOR_ABLATION, PRODUCTION):
        p = _details_path(name)
        if not p.exists():
            continue
        det = pd.read_csv(p)
        for set_name, dp in det.groupby('set'):
            ok = dp['outcome'] == 'approach'
            rows.append({'name': name, 'set': set_name, 'n': int(len(dp)),
                         **{f'share_{o}': float((dp['outcome'] == o).mean()) for o in
                            ('approach', 'approach_partial', 'earlier_movement', 'return', 'later_movement', 'rejected', 'none')},
                         'onset_error_median_ms': float(dp.loc[ok, 'onset_error_ms'].median()),
                         'onset_abs_error_p90_ms': float(dp.loc[ok, 'onset_error_ms'].abs().quantile(0.9)),
                         'end_error_median_ms': float(dp.loc[ok, 'end_error_ms'].median()),
                         'end_abs_error_p90_ms': float(dp.loc[ok, 'end_error_ms'].abs().quantile(0.9)),
                         'coverage': float((dp['status'] == 'ok').mean())})
    write_json('m4_detection', {'rows': rows})


CONDITIONS = ('platform', 'speed', 'orientation', 'posture', 'start_position', 'variant')


def compute_breakdown(d: Any, nat: Any) -> None:
    """Half-credit accuracy, coverage and calibration of the production pipeline and of the oracle reference per
    platform and per instructed condition (conditions only STRATIFY, after prediction; a stratum with fewer than
    MIN_PERSONS_CI persons gets no CI)."""
    from stage2 import cv
    from stage2.lanes import m4_stats as ST
    T = pd.concat([d.trials, nat.trials], ignore_index=True)[['trial_key', *CONDITIONS]]
    out = []
    for name in (PRODUCTION, 'm4_oracle_yaw8', 'm4_v0_yaw8'):
        r = L.load_result(name)
        pred = r.predictions.merge(T, on='trial_key', how='left')
        det = pd.read_csv(_details_path(name)) if _details_path(name).exists() else None
        status = dict(zip(det['trial_key'], det['status'])) if det is not None else {}
        pred['answered'] = pred['trial_key'].map(status).fillna('ok') == 'ok'
        for col in CONDITIONS:
            for lev, g in pred.groupby(col):
                m = cv.metrics(g['y'], g['p_right'])
                ci = ST.person_ci(g, ST.half_credit)
                a = g['answered']
                out.append({'name': name, 'stratum': col, 'level': str(lev), 'n': int(len(g)),
                            'n_persons': int(g['person_id'].nunique()), 'accuracy': ST.half_credit(g['y'], g['p_right']),
                            'acc_lo': ci[0], 'acc_hi': ci[1], 'roc_auc': m['roc_auc'], 'log_loss': m['log_loss'],
                            'ece': m['ece'], 'brier': m['brier'], 'coverage': float(a.mean()),
                            'acc_answered': ST.half_credit(g.loc[a, 'y'], g.loc[a, 'p_right']) if a.any() else np.nan})
    write_json('m4_breakdown', {'rows': out, 'production': PRODUCTION, 'min_persons_ci': MIN_PERSONS_CI})


SEGFEAT_COLUMNS = ('g_disp_z0', 'g_d_up_z', 'disp_z0', 'd_up_z', 'disp_vert', 'still_before', 'still_before_inst',
                   'dur_s', 'peak_w')


def compute_segment_features(d: Any, nat: Any) -> None:
    """Descriptive: hand-agnostic features of every candidate segment of the recordings, labelled from the
    app's events (configuration chosen most often by the folds), and their approach-vs-return and
    approach-vs-other AUCs (evaluation only; the classifier is trained inside the folds)."""
    from collections import Counter
    from stage2 import cv
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    prep = prepared(d, nat, 'main')
    r = L.load_result(PRODUCTION)
    cfgs = Counter(json.loads(p).get('seg_cfg') for p in r.predictions['params'])
    cfg = cfgs.most_common(1)[0][0]
    tab, _ = prep.seg[cfg]
    lab = E.training_labels(tab, np.arange(d.n), d.trials)
    lab['person_id'] = d.trials['person_id'].to_numpy()[lab['stream'].to_numpy(int)]
    auc = []
    for f in D.ALL_FEATURES:
        row = {'feature': f, 'family': ('physics' if f in D.PHYSICS_FEATURES else
                                        'grip' if f in D.GRIP_PHYSICS else 'context')}
        for other in ('return', 'other'):
            s = lab[lab['label'].isin(['approach', other])]
            v = s[f].to_numpy(float)
            ok = np.isfinite(v)
            row[f'auc_vs_{other}'] = cv.roc_auc((s['label'].to_numpy()[ok] == 'approach').astype(int), v[ok])
        auc.append(row)
    keep = lab[['person_id', 'label', *SEGFEAT_COLUMNS]]
    L.export_table(keep, 'm4_segfeat_values', tex=False)
    counts = lab['label'].value_counts().to_dict()
    sign = (lab['g_disp_z0'] * lab['disp_z0'] < 0)        # segments whose grip sign s is -1
    write_json('m4_segfeat', {'seg_cfg': cfg, 'seg_cfg_votes': dict(cfgs), 'auc': auc, 'label_counts': counts,
                              'n_streams': int(d.n), 'n_segments': int(len(lab)),
                              'share_grip_sign_negative': {k: float(sign[lab['label'] == k].mean())
                                                           for k in ('approach', 'return', 'other', 'partial')
                                                           if (lab['label'] == k).any()}})


NATURAL_FEATURES = ('p_approach', 'disp_z0', 'disp_vert', 'dur_s', 'peak_w', 'still_before_inst', 'still_after',
                    'd_up_z', 'v_up_peak', 'up_z0', 'up_z1')


def compute_natural(d: Any, nat: Any) -> None:
    """Why the production detector rejects natural-use approaches (evaluation only): (1) the rejection rate on the
    natural-use recordings against what each person's own primary rejection rate predicts (Poisson-binomial
    tail) and against the primary set (Fisher exact); (2) for every recording, the candidate segment that is the
    stored approach (largest IoU >= 0.5): its P(approach) under the fold's pipeline and its hand-agnostic
    features, natural use against primary."""
    from scipy.stats import fisher_exact
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_stats as ST
    prep = prepared(d, nat, 'main')
    spec = production_spec()
    det = pd.read_csv(_details_path(PRODUCTION))
    prim, natd = det[det['set'] == 'primary'], det[det['set'] == 'natural_use']
    rej_rate = prim.groupby('person_id')['status'].apply(lambda s: float((s != 'ok').mean()))
    probs = natd['person_id'].map(rej_rate).fillna(float((prim['status'] != 'ok').mean())).to_numpy(float)
    n_rej = int((natd['status'] != 'ok').sum())
    table = [[n_rej, int((natd['status'] == 'ok').sum())],
             [int((prim['status'] != 'ok').sum()), int((prim['status'] == 'ok').sum())]]
    test = {'n_natural': int(len(natd)), 'natural_rejected': n_rej, 'natural_rejection_rate': n_rej / max(len(natd), 1),
            'primary_rejection_rate': float((prim['status'] != 'ok').mean()),
            'expected_from_persons': float(probs.sum()), 'p_poisson_binomial': ST.poisson_binomial_sf(probs, n_rej),
            'p_fisher': float(fisher_exact(table, alternative='two-sided')[1])}
    feats = [f for f in NATURAL_FEATURES if f != 'p_approach']
    T = pd.concat([d.trials, nat.trials], ignore_index=True)
    rows = []
    cfg_tab = {}
    for person in fold_persons(d):
        fp = E.pipeline_for_person(prep, spec, d.trials, person)
        if fp.seg_cfg not in cfg_tab:
            cfg_tab[fp.seg_cfg] = prep.seg[fp.seg_cfg][0]
        tab = cfg_tab[fp.seg_cfg]
        ks = list(np.flatnonzero(d.trials['person_id'].astype(str).to_numpy() == person))
        ks += list(d.n + np.flatnonzero(nat.trials['person_id'].astype(str).to_numpy() == person))
        sub = tab[tab['stream'].isin(ks)].reset_index(drop=True)
        if not len(sub):
            continue
        P = fp.segment_proba(sub)
        lab = E.training_labels(sub, np.asarray(ks), T.iloc[ks].reset_index(drop=True))
        sub = sub.assign(p_approach=P['approach'], iou=lab['iou'].to_numpy(float), label=lab['label'].to_numpy())
        for k in ks:
            g = sub[(sub['stream'] == k) & (sub['iou'] >= 0.5)]
            row = {'person_id': person, 'set': 'primary' if k < d.n else 'natural_use', 'stream': int(k),
                   'approach_found': bool(len(g))}
            if len(g):
                j = g['iou'].idxmax()
                row.update({f: float(g.loc[j, f]) for f in NATURAL_FEATURES})
                row['best_other_p_approach'] = float(sub.loc[(sub['stream'] == k) & (sub.index != j), 'p_approach'].max()) \
                    if ((sub['stream'] == k) & (sub.index != j)).any() else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    comp = []
    for f in NATURAL_FEATURES:
        a = df.loc[df['set'] == 'natural_use', f].dropna()
        b = df.loc[df['set'] == 'primary', f].dropna()
        if len(a) and len(b):
            comp.append({'feature': f, 'natural_median': float(a.median()), 'primary_median': float(b.median()),
                         'primary_p25': float(b.quantile(0.25)), 'primary_p75': float(b.quantile(0.75)),
                         'natural_share_below_primary_p25': float((a < b.quantile(0.25)).mean()),
                         'n_natural': int(len(a)), 'n_primary': int(len(b))})
    nat_rows = df[df['set'] == 'natural_use']
    found = nat_rows[nat_rows['approach_found']]
    from stage2.lanes import m4_detect as D
    rest = np.array([D.rest_up_z(s, D.DEFAULT, GRIP_REST_MIN_S) for s in prep.streams])   # the page's grip check
    gate = {'primary': float((rest[:d.n] < GRIP_GATE_UPZ).mean()),
            'natural_use': float((rest[d.n:d.n + nat.n] < GRIP_GATE_UPZ).mean())}
    write_json('m4_natural', {'rejection_test': test, 'features': comp, 'grip_gate_share': gate,
                              'natural_approach_candidate_found': float(nat_rows['approach_found'].mean()) if len(nat_rows) else None,
                              'primary_approach_candidate_found': float(df.loc[df['set'] == 'primary', 'approach_found'].mean()),
                              'natural_approach_p_below_threshold': float((found['p_approach'] < spec.det_threshold).mean())
                              if len(found) else None,
                              'primary_approach_p_below_threshold': float(
                                  (df.loc[(df['set'] == 'primary') & df['approach_found'], 'p_approach'] < spec.det_threshold).mean()),
                              'natural_rows': nat_rows.drop(columns=['stream']).to_dict(orient='records'),
                              'det_threshold': spec.det_threshold})


def compute_scores(d: Any, nat: Any) -> None:
    """Half-credit accuracy, coverage, answered and forced-choice accuracy (person-bootstrap CIs) of every trial
    experiment on the primary set and on natural use (abstentions never count as right-hand answers)."""
    from stage2.lanes import m4_stats as ST
    key_of = dict(zip(pd.concat([d.trials['trial_id'], nat.trials['trial_id']]),
                      pd.concat([d.trials['trial_key'], nat.trials['trial_key']])))
    out = {}
    for name in TRIAL_EXPERIMENTS:
        if not L.result_path(name).exists():
            continue
        r = L.load_result(name)
        det = pd.read_csv(_details_path(name)) if _details_path(name).exists() else None
        entry = {}
        sets = {'primary': r.predictions}
        if r.extra_predictions is not None and len(r.extra_predictions):
            sets['natural_use'] = r.extra_predictions
        for set_name, pred in sets.items():
            if 'trial_key' not in pred.columns:                 # natural-use predictions carry trial_id only
                pred = pred.assign(trial_key=pred['trial_id'].map(key_of))
            pred = pred[['trial_key', 'person_id', 'y', 'p_right']]
            if det is not None:
                dd = det[det['set'] == set_name].set_index('trial_key')
                pf = pred['trial_key'].map(dd['p_forced']).fillna(0.5)
                ans = pred['trial_key'].map(dd['status']).fillna('none') == 'ok'
                entry[set_name] = ST.scores(pred, p_forced=pf, answered=ans)
            else:
                entry[set_name] = ST.scores(pred, answered=np.ones(len(pred), bool))
        out[name] = entry
    # the detector threshold: the production pipeline's saved predictions re-thresholded (no refit: the chosen
    # segment and its P(right) do not depend on the threshold, only whether the pipeline answers)
    thr = []
    if _details_path(PRODUCTION).exists():
        det = pd.read_csv(_details_path(PRODUCTION))
        dp = det[det['set'] == 'primary'].reset_index(drop=True)
        pa = dp['p_approach'].fillna(-1.0).to_numpy(float)
        base = dp[['trial_key', 'person_id', 'y', 'p_right']]
        for t in DET_THRESHOLDS:
            pred = dp[['trial_key', 'person_id', 'y']].assign(p_right=np.where(pa >= t, dp['p_forced'], 0.5))
            sc = ST.scores(pred, answered=pa >= t)
            same = math.isclose(t, production_spec().det_threshold)
            thr.append({'threshold': t, 'accuracy': sc['accuracy_half'], 'acc_ci': sc['accuracy_half_ci'],
                        'coverage': sc['coverage'], 'accuracy_answered': sc['accuracy_answered'],
                        'reproduces_production': bool(np.max(np.abs(pred['p_right'] - base['p_right'])) < 1e-12) if same else None,
                        'vs_production': None if same else ST.paired(pred, base)})
    write_json('m4_scores', {'scores': out, 'det_threshold': thr,
                             'rule': 'P = 0.5 scores half a correct answer (m4_stats.half_credit)'})


#: P(approach) thresholds of the re-thresholding ablation (production: PipelineSpec.det_threshold)
DET_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def paired_pairs() -> list[tuple[str, str, str]]:
    """(key, A, B) of every trial-level paired comparison the report cites (A - B on the same recordings): the
    family 'trial' of the Holm adjustment."""
    base = BASE_FOR_CALIBRATION
    return [('prod_vs_oracle', PRODUCTION, 'm4_oracle_yaw8'), ('prod_vs_v0', PRODUCTION, 'm4_v0_yaw8'),
            ('full_vs_physics', 'm4_v1full_yaw8', 'm4_v1phys_yaw8'), ('nested_vs_physics', 'm4_v1nest_yaw8', 'm4_v1phys_yaw8'),
            ('robust_vs_nested', 'm4_v1robust_yaw8', 'm4_v1nest_yaw8'), ('robust_vs_physics', 'm4_v1robust_yaw8', 'm4_v1phys_yaw8'),
            ('grip_vs_robust', 'm4_v1grip_yaw8', base),
            ('detwin_vs_oraclewin', 'm4_v1nest_yaw8_detwin', 'm4_v1nest_yaw8'),
            ('chiral16_vs_yaw8', 'm4_v1robust_chiral16', base),
            ('oracle_chiral16_vs_yaw8', 'm4_oracle_chiral16', 'm4_oracle_yaw8'),
            ('return_vs_prod', 'm4_prod_return', base),
            ('zp_activity_vs_prod', 'm4_prod_zp_activity', base),
            ('causal_lowpass_vs_prod', 'm4_prod_causal_lowpass', base),
            ('ret_sum_vs_app', 'm4_ret_sum_yaw8', 'm4_ret_app_yaw8'), ('ret_head_vs_app', 'm4_ret_head_yaw8', 'm4_ret_app_yaw8'),
            *[(f'{n.replace("m4_abl_", "abl_")}_vs_base', n, base) for n in ABLATIONS],
            *[(f'cal_{c}_vs_none', f'm4_cal_{c}', base) for c in CALIBRATION_CANDIDATES if c != 'none']]


def compute_paired() -> None:
    """Paired person-bootstrap comparisons of the saved results (half credit; m4_stats.paired), Holm-adjusted over
    the family 'trial' (every comparison listed by paired_pairs)."""
    from stage2.lanes import m4_stats as ST
    out = {}
    for key, a, b in paired_pairs():
        if not (L.result_path(a).exists() and L.result_path(b).exists()):
            continue
        out[key] = {'a': a, 'b': b, **ST.paired(L.load_result(a).predictions, L.load_result(b).predictions)}
    keys = list(out)
    for k, h_acc, h_ll in zip(keys, ST.holm([out[k]['p_acc'] for k in keys]), ST.holm([out[k]['p_ll'] for k in keys])):
        out[k]['p_acc_holm'], out[k]['p_ll_holm'] = h_acc, h_ll
    write_json('m4_paired', {'pairs': out, 'family': 'trial: every comparison in this file (Holm over all of them)',
                             'n_family': len(keys)})


def compute_replay(d: Any, nat: Any) -> None:
    """Replay of the online front end: every candidate segment of the production detector re-computed from the
    raw record re-standardised up to the segment's decision time (zero-phase low-pass over the buffer received
    so far, as the inference page would do), then each recording's choice and P(right) by its fold's pipeline,
    against the offline values (zero-phase over the whole recording)."""
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_stats as ST
    from motionlab.preprocess import lowpass
    prep = prepared(d, nat, 'main')
    spec = production_spec()
    raw = _raw_streams(d)
    T = d.trials
    rows, check = [], []
    for person in fold_persons(d):
        fp = E.pipeline_for_person(prep, spec, T, person)
        p = E.SEG_GRID[fp.seg_cfg]
        tab_all, W_all = prep.seg[fp.seg_cfg]
        for i in np.flatnonzero(T['person_id'].astype(str).to_numpy() == person):
            st = raw[int(i)]
            full = _lowpassed(st, lowpass)
            check.append(float(np.nanmax(np.abs(full['gyro'] - prep.streams[int(i)]['gyro']))))
            sel = np.flatnonzero(tab_all['stream'].to_numpy() == i)
            if not sel.size:
                rows.append({'trial_key': T['trial_key'].iloc[i], 'person_id': person, 'y': int(T['y_right'].iloc[i]),
                             'p_offline': 0.5, 'p_replay': 0.5, 'n_seg': 0, 'n_matched': 0, 'max_feat_diff': 0.0})
                continue
            tab_i = tab_all.iloc[sel].reset_index(drop=True)
            new_rows, new_w, diffs = [], [], []
            for j, r in tab_i.iterrows():
                cut = {k: v[st['t'] <= r['decided_ms'] + 1e-6] for k, v in st.items()}
                part, Wp = D.segment_table(_lowpassed(cut, lowpass), p)
                m = np.flatnonzero(np.abs(part['onset_ms'].to_numpy(float) - r['onset_ms']) < 0.5 * 1000.0 / C.FS_HZ) \
                    if len(part) else np.zeros(0, int)
                if m.size:
                    q = part.iloc[int(m[0])]
                    new_rows.append({**q.to_dict(), 'stream': int(i)})
                    new_w.append(Wp[int(m[0])])
                    feats = list(fp.segclf.features)
                    a = np.asarray([q[f] for f in feats], float)
                    b = np.asarray([r[f] for f in feats], float)
                    diffs.append(float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0)
                else:                              # the segment does not exist yet / any more in the buffer
                    new_rows.append({**r.to_dict(), 'canonical_ok': False, 'stream': int(i)})
                    new_w.append(np.full_like(W_all[sel[j]], np.nan))
            rep_tab = pd.DataFrame(new_rows)
            for c in tab_i.columns:
                if c not in rep_tab.columns:
                    rep_tab[c] = np.nan
            off = fp.choose(prep, [int(i)])
            on = fp.choose(prep, [int(i)], seg_override=(rep_tab[tab_i.columns.tolist()], np.stack(new_w)))
            rows.append({'trial_key': T['trial_key'].iloc[i], 'person_id': person, 'y': int(T['y_right'].iloc[i]),
                         'p_offline': float(off['p_right'].iloc[0]), 'p_replay': float(on['p_right'].iloc[0]),
                         'status_offline': off['status'].iloc[0], 'status_replay': on['status'].iloc[0],
                         'n_seg': int(len(tab_i)), 'n_matched': int(len(diffs)),
                         'max_feat_diff': float(max(diffs)) if diffs else 0.0})
    df = pd.DataFrame(rows)
    L.export_table(df, 'm4_replay_details', tex=False)
    a = df[['trial_key', 'person_id', 'y']].assign(p_right=df['p_replay'])
    b = df[['trial_key', 'person_id', 'y']].assign(p_right=df['p_offline'])
    write_json('m4_replay', {'n': int(len(df)), 'max_abs_dp': float(np.max(np.abs(df['p_replay'] - df['p_offline']))),
                             'median_abs_dp': float(np.median(np.abs(df['p_replay'] - df['p_offline']))),
                             'share_decision_changed': float(np.mean((df['p_replay'] >= 0.5) != (df['p_offline'] >= 0.5))),
                             'share_status_changed': float(np.mean(df['status_replay'] != df['status_offline'])),
                             'segments_matched': float(df['n_matched'].sum() / max(df['n_seg'].sum(), 1)),
                             'max_feature_diff_median': float(df['max_feat_diff'].median()),
                             'full_buffer_reproduces_cache_max_abs_gyro': float(np.nanmax(check)) if check else None,
                             'paired_replay_minus_offline': ST.paired(a, b)})


def _lowpassed(st: dict[str, np.ndarray], lowpass: Any) -> dict[str, np.ndarray]:
    """The zero-phase 15 Hz low-pass of the standardisation applied to a buffer (f, acc, gyro together, as
    ``standardize.stream_from_record`` does)."""
    U = lowpass(np.column_stack([st['f'], st['acc'], st['gyro']]), C.FS_HZ, C.LOWPASS_HZ, order=C.LOWPASS_ORDER)
    return {'t': st['t'], 'f': U[:, 0:3], 'acc': U[:, 3:6], 'gyro': U[:, 6:9], 'valid': st['valid']}


def rebuild_registry() -> None:
    """results/registry_m4.csv rebuilt from the saved m4 result files (one row each, sorted by name), so
    compute runs in parallel processes cannot leave it incomplete."""
    from stage2 import cv
    rows = [cv.ExperimentResult.load(L.result_path(n)).summary_row() for n in TRIAL_EXPERIMENTS
            if L.result_path(n).exists()]
    if rows:
        pd.DataFrame(rows).sort_values('name').to_csv(L.registry, index=False)


def write_finalists() -> None:
    if not L.result_path(PRODUCTION).exists():
        return
    spec = production_spec()
    L.write_finalists([{
        'name': PRODUCTION, 'results_json': f'results/{PRODUCTION}.json', 'module': 'stage2.lanes.m4_finalist',
        'build_x': 'build_x', 'factory': 'factory', 'params': {'spec': json.loads(spec.label())},
        'n_configs_tried': N_CONFIGS_FINAL,
        'notes': ('cue-free production pipeline: causal hysteresis segments (segmentation configuration chosen inside '
                  'each fold) + approach/return/other classifier on the robust hand-agnostic features (stillness '
                  'before capped at the instructed stillness; screen-normal features assume the screen faces the user: '
                  'the inference page applies the rest-pose grip check m4_detect.rest_up_z / m4_config.GRIP_GATE_UPZ, '
                  'not part of this trial-level callable) + hand model + calibrator chosen '
                  'inside each fold among the antisymmetric ones (none, temperature, symmetrised isotonic). The '
                  'detector feature set, the hand model and the attempt rule were chosen '
                  f'after comparisons on the outer folds: n_configs_tried = {N_CONFIGS_FINAL} counts every cue-free '
                  'pipeline evaluated (exploratory, conservative). X = one label-free item per trial (build_x: the '
                  'standardised recording and its cached windows; the test items\' event-cut windows are masked '
                  'inside fit_predict). One recording per row; an abstention is P = 0.5 (score it as half credit). '
                  'Hand-model plug-in: factory(hand_model=name) after m4_models.register_hand_model; WINDOW-ONLY '
                  'contract: fit(X, y, groups, seed) / logit(X) on (N, 17, 64) time-normalised windows of '
                  'standardize.CHANNELS (see the m4_finalist docstring). Multi-touch runtime (stopping rule): '
                  'stage2.lanes.m4_production.fit_production / predict_stream.')}])


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m4_figures
    m4_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m4_report
    m4_report.tables(only)


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.lanes import m4_report
    m4_report.numbers()
    m4_report.write_summary()


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}
