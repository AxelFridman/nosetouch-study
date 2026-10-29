"""The production models as Python objects behind one interface, from the in-memory fits or rebuilt from the JSON.

``ModelSet`` holds lane m4's ``ProductionModel`` (detector, segment classifier, yaw8 hand model, calibrator, aggregate
scale, stopping rule), the final CNN (the ``final_infer`` dict), PHYS6 and the yaw reference. ``fitted_set()`` wraps
the objects ``stage3.fit`` produced; ``json_set(models_dir)`` rebuilds the SAME Stage 2 / scikit-learn classes from
the exported JSON (``to_json_*`` below write them), so every downstream computation - ``m4_production.predict_stream``
included - runs the evaluated code with either parameter source. Comparing the two is the export's round-trip check.

Per attempt every model returns ``{available, z_raw, p, z}``: z_raw the raw logit, p the calibrated P(right) of the
attempt (the model's own calibrator and clipping), z the logit its k-attempt aggregation reads. The aggregations:

* cnn   - ``final_infer.aggregate``: P_k = sigmoid(sum of logit(p_i, eps 1e-12) / tau_k);
* m4    - ``m4_online.aggregate``: P_k = sigmoid(s_k * mean z_i), z_i = logit(calibrated p_i, eps 1e-6), sigmoid argument
          clipped to +-40 (the session-fitted scale s_k = s_1 k^b);
* phys6 - lane m1's ``fitted_alpha`` aggregate (``m1_analysis.k_aggregation``): P_k = sigmoid(sum z_i / k^alpha),
          z_i = logit(p_i, eps 1e-6);
* yaw   - the harness's k-attempt curve (``cv.k_attempt_curve``): P_k = mean p_i (never fitted: not calibrated for k >= 2).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C

MODEL_IDS = ('cnn', 'm4', 'phys6', 'yaw')
PHYS6 = ('yaw_q50', 'yaw_q100', 'roll_q50', 'roll_q100', 'latvel_q50', 'latvel_q100')
FILES = {'cnn': 'final_cnn.json', 'm4': 'm4_production.json', 'phys6': 'phys6.json', 'yaw': 'yaw_ref.json'}
NOTE_THRESHOLDS = ("the detector (segmentation, segment classifier, attempt rule) and the stopping thresholds (p_stop, "
                   "k_max, p_answer) were fitted for lane m4's own hand classifier; lane prod, which would refit them for "
                   "the final model, has not run")
#: The page's grip gate (rest_up_z below it: 'turn the screen towards your face'). Lane m4 uses 0.0
#: (m4_config.GRIP_GATE_UPZ); a near-vertical hold has rest_up_z close to 0, so at 0.0 the gate refused 27 of 620
#: screen-toward recordings (26 of 396 on Android), 23 of them in [-0.11, 0). Screen-toward refused / screen-away twins
#: refused, all local recordings (Stage 3 parity review): 0.0 -> 27/620, 63/64; -0.05 -> 8/620, 62/64; -0.1 -> 4/620,
#: 58/64. A screen-away hold that passes is mostly answered with the wrong hand (the final CNN is right on 28.6 % of
#: screen-flipped trials), while a refusal only asks for a retry, so the page takes -0.05. Stage 2 is unchanged.
PRODUCTION_GRIP_GATE_UPZ = -0.05


def production_m4(m4: Any) -> Any:
    """Lane m4's production model with the page's grip gate (PRODUCTION_GRIP_GATE_UPZ); everything else unchanged."""
    from dataclasses import replace
    return replace(m4, grip_gate=PRODUCTION_GRIP_GATE_UPZ)


def _f(v: Any) -> Any:
    """JSON-safe float (repr round-trips float64); NaN / inf -> None."""
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    return _f(v)


# ───────────────────────────── PHYS6 (lane m1) ─────────────────────────────


def phys6_frame(window: np.ndarray, duration_s: float) -> pd.DataFrame | None:
    """The m1 feature row of one detected window, as ``m1_models.build_x_treat`` builds it for a cached trial
    (``m1_features.compute_features`` on the window and its own duration, then ``amp_normalised``). The fixed-window
    input is a zero placeholder: no PHYS6 column reads it. None when compute_features refuses the trial (a non-finite
    feature: the body frame is undefined with the screen exactly horizontal)."""
    from stage2.lanes import m1_features as MF
    from stage2.standardize import CHANNELS
    A = np.asarray(window, dtype=np.float32)[None]
    fixed = np.zeros((1, len(CHANNELS), int(round((C.FIXED_PRE_S + C.FIXED_POST_S) * C.FS_HZ))))
    mask = np.ones((1, fixed.shape[2]), dtype=bool)
    try:
        F = MF.compute_features(A, fixed, mask, np.array([float(duration_s)]))
    except ValueError:
        return None
    return pd.concat([F, MF.amp_normalised(F, list(PHYS6))], axis=1)


@dataclass
class Phys6:
    option: str
    C: float
    cols: list[str]
    model: Any                      # sklearn LogisticRegression (no intercept: antisymmetric)
    prep: Any                       # m1_models.Prep (winsorising + scaling fitted on the training rows)
    alpha: float
    eps: float = 1e-6

    def z_raw(self, F: pd.DataFrame) -> float:
        """The logit of the fitted model, computed as its own closure does: decision_function(prep.transform(X))."""
        X = F[self.cols].to_numpy(dtype=float)
        return float(self.model.decision_function(self.prep.transform(X))[0])

    def predict(self, window: np.ndarray, onset_ms: float, end_ms: float) -> dict[str, Any]:
        from stage2.lanes import m1_models as M1
        dur = (float(end_ms) - float(onset_ms)) / 1000.0
        F = phys6_frame(window, dur)
        feats = None if F is None else [float(F[c].iloc[0]) for c in PHYS6]
        if F is None or not np.all(np.isfinite(feats)):
            return {'available': False, 'z_raw': None, 'p': None, 'z': None, 'features6': feats, 'duration_s': dur}
        zr = self.z_raw(F)
        p = float(np.clip(M1.sigmoid(np.array([zr])), 0.0, 1.0)[0])        # _finish: clip(apply(z), 0, 1), apply = sigmoid
        z = float(M1.logit(np.array([p]))[0])
        return {'available': True, 'z_raw': zr, 'p': p, 'z': z, 'features6': feats, 'duration_s': dur}

    def aggregate(self, z: list[float]) -> float:
        """lane m1's fitted_alpha aggregate: sigmoid(sum z / k^alpha) (m1_analysis.k_aggregation)."""
        from stage2.lanes import m1_models as M1
        k = len(z)
        if k == 0:
            return 0.5
        zz = np.asarray(z, dtype=float)
        return float(M1.sigmoid(zz.sum() / k ** self.alpha))


def phys6_to_json(ph: dict[str, Any]) -> dict[str, Any]:
    prep, m = ph['prep'], ph['model']
    st = ph['store']
    return {
        'features': list(ph['cols']), 'option': ph['option'], 'C': ph['C'],
        'winsor': prep.winsor, 'winsor_med': prep.med.tolist(), 'winsor_s': prep.s_w.tolist(),
        'winsor_lo': [None if not np.isfinite(v) else float(v) for v in prep.lo],
        'winsor_hi': [None if not np.isfinite(v) else float(v) for v in prep.hi],
        'scale_kind': prep.scale, 'sign': prep.sign.tolist(), 'centre': prep.c.tolist(), 'scale': prep.s.tolist(),
        'coef': m.coef_[0].tolist(), 'intercept': float(m.intercept_[0]) if np.ndim(m.intercept_) else float(m.intercept_),
        'fit_intercept': bool(m.fit_intercept), 'coef_raw_store': st.get('coef_raw'),
        'calibrator': {'method': 'none', 'p_clip_out': [0.0, 1.0]}, 'logitEps': 1e-6,
        'aggregation': {'kind': 'sum_over_k_pow', 'alpha': ph['agg_alpha'],
                        'rule': 'P_k = sigmoid(sum of the k attempts\' logits / k^alpha), logit with p clipped to '
                                '[1e-6, 1 - 1e-6] (lane m1 fitted_alpha)'},
        'input': {'window': 'the (17, 64) float32 approach window (standardize.time_normalised), read as float64',
                  'duration': 'duration_s = (end_ms - onset_ms) / 1000 of the detected bounds; dt = duration_s / 63',
                  'body_frame': {'h_min': 1e-6, 'nan_below_h_min': True,
                                 'definition': 'h = e_z - (e_z.u) u, h /= |h| (NaN when |h| <= 1e-6), s = u x h; '
                                               'roll_rate = w.h, acc_lat = a.s (float64, NOT rounded to float32)'},
                  'features': {'yaw_q50': '0.5 * yaw_cum[31] + 0.5 * yaw_cum[32]', 'yaw_q100': 'yaw_cum[63]',
                               'roll_q50 / roll_q100': 'the same of roll = cumtrapz(roll_rate, dt)',
                               'latvel_q50 / latvel_q100': 'the same of lat = cumtrapz(acc_lat, dt)',
                               'cumtrapz': 'c[0] = 0, c[k + 1] = c[k] + (0.5 * (x[k + 1] + x[k])) * dt'}},
        'formula': 'z_raw = sum_j coef_j * (W_j - centre_j) / scale_j with W_j = winsor_med_j + winsor_s_j * asinh((x_j - '
                   'winsor_med_j) / winsor_s_j) (winsor asinh; clip: clip(x, lo, hi); none: x); p = sigmoid(z_raw); '
                   'unavailable when any feature is not finite',
    }


def phys6_from_json(doc: dict[str, Any]) -> Phys6:
    from sklearn.linear_model import LogisticRegression
    from stage2.lanes import m1_models as M1
    prep = M1.Prep(np.asarray(doc['sign'], float), doc['winsor'], doc['scale_kind'])
    n = len(doc['features'])
    prep.lo = np.asarray([-np.inf if v is None else v for v in doc['winsor_lo']], float)
    prep.hi = np.asarray([np.inf if v is None else v for v in doc['winsor_hi']], float)
    prep.med = np.asarray(doc['winsor_med'], float)
    prep.s_w = np.asarray(doc['winsor_s'], float)
    prep.c = np.asarray(doc['centre'], float)
    prep.s = np.asarray(doc['scale'], float)
    m = LogisticRegression(C=float(doc['C']), fit_intercept=bool(doc['fit_intercept']), max_iter=10000, tol=1e-6)
    m.classes_ = np.array([0, 1])
    m.coef_ = np.asarray(doc['coef'], float).reshape(1, n)
    m.intercept_ = np.asarray([doc['intercept']], float)
    m.n_features_in_ = n
    return Phys6(doc['option'], float(doc['C']), list(doc['features']), m, prep, float(doc['aggregation']['alpha']),
                 float(doc['logitEps']))


def phys6_fitted(ph: dict[str, Any]) -> Phys6:
    return Phys6(ph['option'], ph['C'], list(ph['cols']), ph['model'], ph['prep'], ph['agg_alpha'])


# ───────────────────────────── the yaw reference (registry) ─────────────────────────────


@dataclass
class YawRef:
    model: Any                       # sklearn LogisticRegression as fitted by example_yaw_logreg
    index: int = 15
    step: int = 8

    def x8(self, window: np.ndarray) -> np.ndarray:
        return np.asarray(window)[self.index, ::self.step]

    def predict(self, window: np.ndarray, onset_ms: float = math.nan, end_ms: float = math.nan) -> dict[str, Any]:
        x = self.x8(window)
        if not np.all(np.isfinite(x)):
            return {'available': False, 'z_raw': None, 'p': None, 'z': None, 'x8': [float(v) for v in x]}
        x64 = x.astype(np.float64)[None]
        zr = float(self.model.decision_function(x64)[0])
        p = float(self.model.predict_proba(x64)[0, 1])
        x32 = x.astype(np.float32)[None]                     # the evaluated path: float32 input (d.approach is float32)
        out = {'available': True, 'z_raw': zr, 'p': p, 'z': zr, 'x8': [float(v) for v in x]}
        if np.asarray(self.model.coef_).dtype == np.float32:
            out['z_raw_f32'] = float(self.model.decision_function(x32)[0])
            out['p_f32'] = float(self.model.predict_proba(x32)[0, 1])
        return out

    @staticmethod
    def aggregate(p: list[float]) -> float:
        """the harness's k-attempt curve: the mean probability (cv.k_attempt_curve)."""
        return float(np.mean(np.asarray(p, dtype=float))) if len(p) else 0.5


def yaw_to_json(y: dict[str, Any]) -> dict[str, Any]:
    m = y['model']
    return {'channel': y['channel'], 'index': int(y['index']), 'step': int(y['step']), 'n': 8,
            'coef': np.asarray(m.coef_[0], dtype=np.float64).tolist(), 'intercept': float(m.intercept_[0]),
            'coef_dtype': str(np.asarray(m.coef_).dtype), 'C': float(m.C), 'nan_to_num': False,
            'calibrator': {'method': 'none'},
            'aggregation': {'kind': 'mean_p', 'calibrated_beyond_k1': False,
                            'rule': 'P_k = mean of the k attempts\' P (the harness k-curve; never fitted, so P_k for k >= 2 '
                                    'is not calibrated)'},
            'formula': 'x = window[yaw_cum][0::8] (indices 0, 8, ..., 56); z_raw = coef . x + intercept (float64 with the '
                       'float32-valued coefficients); p = 1 / (1 + exp(-z_raw)); unavailable when any x is not finite',
            'precision_note': 'fitted in float32 (scikit-learn keeps the float32 dtype of the cached windows): the '
                              'evaluated predictions were float32 arithmetic; the page computes in float64 with the same '
                              'coefficients (|dP| below 1e-6)'}


def yaw_from_json(doc: dict[str, Any]) -> YawRef:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=float(doc['C']), max_iter=1000)
    m.classes_ = np.array([0, 1])
    dt = np.dtype(doc.get('coef_dtype', 'float64'))        # float32: as example_yaw_logreg fitted it (exact values)
    m.coef_ = np.asarray(doc['coef'], dtype=dt).reshape(1, -1)
    m.intercept_ = np.asarray([doc['intercept']], dtype=dt)
    m.n_features_in_ = int(doc['n'])
    return YawRef(m, int(doc['index']), int(doc['step']))


# ───────────────────────────── lane m4's production model ─────────────────────────────


def _iso_to_json(iso: Any) -> dict[str, Any]:
    return {'x_thresholds': iso.X_thresholds_.tolist(), 'y_thresholds': iso.y_thresholds_.tolist(),
            'x_min': float(iso.X_min_), 'x_max': float(iso.X_max_), 'increasing': bool(iso.increasing_),
            'y_min': iso.y_min, 'y_max': iso.y_max, 'out_of_bounds': iso.out_of_bounds}


def _iso_from_json(doc: dict[str, Any]) -> Any:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds=doc['out_of_bounds'], y_min=doc['y_min'], y_max=doc['y_max'])
    iso.X_thresholds_ = np.asarray(doc['x_thresholds'], float)
    iso.y_thresholds_ = np.asarray(doc['y_thresholds'], float)
    iso.X_min_, iso.X_max_ = float(doc['x_min']), float(doc['x_max'])
    iso.increasing_ = bool(doc['increasing'])
    iso._build_f(iso.X_thresholds_, iso.y_thresholds_)
    return iso


def calibrator_to_json(cal: Any) -> dict[str, Any]:
    from stage2.lanes import m4_models as M
    if isinstance(cal, M.Nested):
        return {'method': 'nested', 'chosen': cal.chosen, 'cv_log_loss': dict(cal.cv_log_loss),
                'inner': calibrator_to_json(cal.inner)}
    if isinstance(cal, M.IsotonicSym):
        return {'method': 'isotonic_sym', **_iso_to_json(cal.m), 'p_lo': 0.01, 'p_hi': 0.99,
                'interp': 'interp1d_left', 'eval': 'abs_reflect',
                'formula': 'T = clip(|z|, x_min, x_max); j = searchsorted(x_thresholds, T, side=left) clipped to '
                           '[1, n - 1]; iso = (y[j] - y[j-1]) / (x[j] - x[j-1]) * (T - x[j-1]) + y[j-1] (scipy '
                           'interp1d); p = clip(iso, 0.01, 0.99); P = p if z >= 0 else 1 - p'}
    if isinstance(cal, M.Temperature):
        return {'method': 'temperature', 'T': cal.T, 'logit_clip': 40.0,
                'formula': 'P = 1 / (1 + exp(-clip(z / T, -40, 40)))'}
    if type(cal) is M.Calibrator:
        return {'method': 'none', 'logit_clip': 40.0, 'formula': 'P = 1 / (1 + exp(-clip(z, -40, 40)))'}
    raise ValueError(f'calibrator {type(cal).__name__} has no export (the production choice is among none, '
                     'temperature, isotonic_sym)')


def calibrator_from_json(doc: dict[str, Any]) -> Any:
    from stage2.lanes import m4_models as M
    k = doc['method']
    if k == 'nested':
        c = M.Nested()
        c.chosen = doc['chosen']
        c.cv_log_loss = dict(doc.get('cv_log_loss') or {})
        c.inner = calibrator_from_json(doc['inner'])
        return c
    c = M.CALIBRATORS[k]()
    if k == 'temperature':
        c.T = float(doc['T'])
    elif k == 'isotonic_sym':
        c.m = _iso_from_json(doc)
    elif k != 'none':
        raise ValueError(k)
    return c


def filter_constants(seg: Any) -> dict[str, Any]:
    """Filter coefficients the port needs, recomputed with scipy (never typed)."""
    from scipy import signal
    from motionlab import preprocess as PP  # noqa: F401  (documents where the low-pass lives)
    sos_a = signal.butter(2, seg.smooth_hz, btype='low', fs=C.FS_HZ, output='sos')
    sos_l = signal.butter(C.LOWPASS_ORDER, C.LOWPASS_HZ, btype='low', fs=C.FS_HZ, output='sos')
    pad = 3 * (2 * len(sos_l) + 1 - min(int((sos_l[:, 2] == 0).sum()), int((sos_l[:, 5] == 0).sum())))
    return {'activitySos': sos_a.tolist(), 'activityZi': signal.sosfilt_zi(sos_a).tolist(),
            'lowpassSos': sos_l.tolist(), 'lowpassZi': signal.sosfilt_zi(sos_l).tolist(), 'lowpassPadlen': int(pad),
            'lowpassRule': 'scipy sosfiltfilt per finite run, padlen = min(lowpassPadlen, run - 1), runs shorter than '
                           '3 samples copied unfiltered (motionlab.preprocess.lowpass)'}


def m4_to_json(model: Any) -> dict[str, Any]:
    from dataclasses import asdict
    from stage2.lanes import m4_config as CFG
    from stage2.lanes import m4_detect as D
    fp = model.fitted
    sc = fp.segclf
    hand = fp.hand
    desc = model.describe()
    info = {k: v for k, v in desc.items() if k not in ('segment_features',)}
    agg = model.agg
    detector = {
        'segParams': asdict(model.seg_params), 'segCfg': fp.seg_cfg, 'causalDelayMs': D.causal_delay_ms(model.seg_params),
        **filter_constants(model.seg_params),
        'segmentClassifier': {
            'features': list(sc.features), 'classes': [str(c) for c in sc.m.classes_], 'scaling': sc.scaling,
            'center': np.asarray(sc.med, float).tolist(), 'scale': np.asarray(sc.iqr, float).tolist(),
            'fill': None, 'clip': 5.0, 'C': float(sc.C_chosen), 'coef': sc.m.coef_.tolist(),
            'intercept': sc.m.intercept_.tolist(),
            'formula': 'Z = (F - center) / scale; Z = 0 where not finite; Z = clip(Z, -5, 5); L = coef @ Z + intercept; '
                       'P = softmax(L) (exp(L - max L) / sum); classes in the order of "classes"'},
        'attemptRule': {'detThreshold': float(fp.spec.det_threshold), 'pairReturns': bool(fp.spec.pair_returns),
                        'argmaxOrder': list(D.CLASSES),
                        'rule': 'cls = argmax over (approach, return, other) of P (ties: the first); attempt = cls == '
                                'approach AND P(approach) >= detThreshold AND canonical_ok'},
        'stopping': {'pStop': float(model.p_stop), 'kMax': int(model.k_max), 'pAnswer': float(model.p_answer)},
        'grip': {'gateUpz': None if model.grip_gate is None else float(model.grip_gate),
                 'restMinS': float(CFG.GRIP_REST_MIN_S), 'restParams': asdict(D.DEFAULT),
                 'restCausalDelayMs': D.causal_delay_ms(D.DEFAULT)},
        'constants': {'FS_HZ': C.FS_HZ, 'LOWPASS_HZ': C.LOWPASS_HZ, 'LOWPASS_ORDER': C.LOWPASS_ORDER,
                      'MAX_GAP_MS': C.MAX_GAP_MS, 'STILL_GYRO_DPS': C.STILL_GYRO_DPS, 'REST_MS': C.REST_MS,
                      'MIN_REST_N': 5, 'N_NORM': C.N_NORM, 'GRAVITY': 9.80665, 'MAG_MIN': 8.6, 'MAG_MAX': 11.0,
                      'INSTRUCTED_STILLNESS_S': D.INSTRUCTED_STILLNESS_S, 'MIN_ROWS': 5, 'REPEAT_MIN_DPS': 20.0,
                      'MIN_ACTIVITY_N': 12},
        'note': NOTE_THRESHOLDS,
    }
    s = {str(k): float(v) for k, v in sorted(agg.s.items())} if agg is not None else None
    return {
        'detector': detector,
        'hand': {'kind': 'yaw8', 'channel': 'yaw_cum', 'index': 15, 'step': 8, 'n': 8, 'C': float(hand.C),
                 'coef': hand.m.coef_[0].tolist(), 'intercept': float(hand.m.intercept_[0]), 'nanToNum': True,
                 'formula': 'x = nan_to_num(window[yaw_cum][0::8]); z_raw = coef . x + intercept'},
        'calibrator': calibrator_to_json(fp.calib),
        'logitEps': 1e-6, 'sigmoidClip': 40.0,
        'aggregation': {'kind': 'scaled_mean', 'name': model.aggregation, 'form': 'power',
                        's1': getattr(agg, 's1', None), 'b': getattr(agg, 'b', None), 's': s,
                        'rule': 'P_k = 1 / (1 + exp(-clip(s_k * mean(z_1..z_k), -40, 40))), z_i = logit(P_i) with P_i '
                                'clipped to [1e-6, 1 - 1e-6]; k above the largest key uses the largest'},
        'attemptP': 'the P(right) listed per attempt is 1 / (1 + exp(-z)) of its calibrated logit z (no clip)',
        'percent': 'percent_right = round(100 * P) with round-half-to-even (Python round)',
        'spec': asdict(fp.spec), 'info': jsonable(info),
    }


def m4_from_json(doc: dict[str, Any]) -> Any:
    """A ``ProductionModel`` with real Stage 2 / scikit-learn objects carrying the exported parameters."""
    from sklearn.linear_model import LogisticRegression
    from stage2.lanes import m4_detect as D
    from stage2.lanes import m4_e2e as E
    from stage2.lanes import m4_models as M
    from stage2.lanes import m4_production as PR
    det = doc['detector']
    sc_doc = det['segmentClassifier']
    sc = M.SegmentClassifier(sc_doc['features'], C=sc_doc['C'], scaling=sc_doc['scaling'])
    sc.C_chosen = float(sc_doc['C'])
    sc.med = np.asarray(sc_doc['center'], float)
    sc.iqr = np.asarray(sc_doc['scale'], float)
    lr = LogisticRegression(C=float(sc_doc['C']), max_iter=3000)
    lr.classes_ = np.array(sc_doc['classes'])
    # scikit-learn's multinomial lbfgs leaves coef_ as a Fortran-ordered view of its solution vector; the same layout
    # makes BLAS sum in the same order, so the rebuilt classifier reproduces the fitted one bit for bit
    lr.coef_ = np.asfortranarray(np.asarray(sc_doc['coef'], float))
    lr.intercept_ = np.asarray(sc_doc['intercept'], float)
    lr.n_features_in_ = len(sc_doc['features'])
    sc.m = lr
    hd = doc['hand']
    hand = M.Yaw8(C=float(hd['C']))
    hm = LogisticRegression(C=float(hd['C']), max_iter=2000)
    hm.classes_ = np.array([0, 1])
    hm.coef_ = np.asarray(hd['coef'], float).reshape(1, -1)
    hm.intercept_ = np.asarray([hd['intercept']], float)
    hm.n_features_in_ = int(hd['n'])
    hand.m = hm
    calib = calibrator_from_json(doc['calibrator'])
    ag = doc['aggregation']
    agg = None
    if ag.get('s') is not None:
        agg = M.AggregateScale()
        agg.s = {int(k): float(v) for k, v in ag['s'].items()}
        if ag.get('s1') is not None:
            agg.s1, agg.b = float(ag['s1']), float(ag['b'])
    spec = E.PipelineSpec(**doc['spec'])
    info = {'calibration': (doc.get('info') or {}).get('calibration'), 'seg_cfg': det['segCfg']}
    fitted = E.FittedPipeline(spec, det['segCfg'], sc, hand, calib, None, None, info)
    seg = D.SegParams(**det['segParams'])
    assert seg == E.SEG_GRID[det['segCfg']], 'segParams differ from the SEG_GRID entry they name'
    st = det['stopping']
    return PR.ProductionModel(fitted, seg, p_stop=float(st['pStop']), k_max=int(st['kMax']), p_answer=float(st['pAnswer']),
                              trained_on=tuple(doc.get('training', {}).get('persons') or ()), agg=agg,
                              aggregation=ag['name'], grip_gate=det['grip']['gateUpz'])


# ───────────────────────────── the model set ─────────────────────────────


@dataclass
class ModelSet:
    m4: Any                          # m4_production.ProductionModel
    cnn: dict[str, Any]              # final_infer model dict
    phys6: Phys6
    yaw: YawRef
    source: str = 'fitted'           # 'fitted' (in-memory) | 'json' (rebuilt from the exported files)
    docs: dict[str, Any] = field(default_factory=dict)

    # ---- per-attempt outputs of the models that do not belong to the detector ----
    def cnn_predict(self, window: np.ndarray) -> dict[str, Any]:
        from stage2.lanes import final_infer as FI
        A = np.asarray(window, dtype=np.float32)[None]
        X, names = FI.extended(self.cnn, A)
        m = self.cnn['model']
        members = [float(FI.logits_cnn({**m, 'members': [mem]}, X, names)[0]) for mem in m['members']]
        zr = float(FI.model_logits(self.cnn, A)[0])
        body = X[0, len(self.cnn['input']['channels']):]
        if not math.isfinite(zr):
            return {'available': False, 'z_raw': None, 'p': None, 'z': None, 'member_logits': members,
                    'body': body}
        p = float(FI.calibrate(self.cnn, np.array([zr]))[0])
        z = float(FI.logit(np.array([p]))[0])
        return {'available': True, 'z_raw': zr, 'p': p, 'z': z, 'member_logits': members, 'body': body}

    def m4_predict(self, window: np.ndarray) -> dict[str, Any]:
        """m4's hand model + calibrator on one window (the batched attempt rows come from FittedPipeline.attempts)."""
        from stage2.lanes import m4_models as M
        fp = self.m4.fitted
        zr = float(fp.hand.logit(np.asarray(window, dtype=np.float32)[None])[0])
        p = float(fp.calib(np.array([zr]))[0])
        z = float(M.logit(np.array([p]))[0])
        return {'available': True, 'z_raw': zr, 'p': p, 'z': z, 'p_attempt': 1.0 / (1.0 + math.exp(-z))}

    def aggregate(self, model_id: str, outs: list[dict[str, Any]]) -> float:
        """P_k of one model over the outputs of its first k AVAILABLE attempts (decision order)."""
        from stage2.lanes import final_infer as FI
        from stage2.lanes import m4_online as O
        outs = [o for o in outs if o['available']]
        if not outs:
            return 0.5
        if model_id == 'cnn':
            return FI.aggregate(self.cnn, np.array([o['p'] for o in outs]))
        if model_id == 'm4':
            return O.aggregate([o['z'] for o in outs], self.m4.agg)
        if model_id == 'phys6':
            return self.phys6.aggregate([o['z'] for o in outs])
        if model_id == 'yaw':
            return self.yaw.aggregate([o['p'] for o in outs])
        raise KeyError(model_id)


def fitted_set(fitted: dict[str, Any] | None = None) -> ModelSet:
    from stage2.lanes import final_infer as FI
    from . import fit as F
    fitted = fitted if fitted is not None else F.fitted_models(verbose=False)
    cnn = FI.load(C.OUT_DIR / 'export' / 'final_model.json')
    return ModelSet(production_m4(fitted['m4']), cnn, phys6_fitted(fitted['phys6']), YawRef(fitted['yaw']['model'],
                    int(fitted['yaw']['index']), int(fitted['yaw']['step'])), 'fitted')


def json_set(models_dir: str | Path) -> ModelSet:
    d = Path(models_dir)
    docs = {k: json.loads((d / f).read_text(encoding='utf-8')) for k, f in FILES.items()}
    return ModelSet(m4_from_json(docs['m4']), docs['cnn'], phys6_from_json(docs['phys6']), yaw_from_json(docs['yaw']),
                    'json', docs)
