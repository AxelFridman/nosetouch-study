"""Lane m4, part 3: the end-to-end cue-free pipeline (detector + hand classifier + calibration), fitted on
the training persons of one fold and applied to the held-out person's streams.

Leakage design. Everything that ``fit_predict`` closes over is LABEL-FREE and computed per stream:
:class:`Prepared` holds the standardised streams, the oracle approach windows (the cached ``d.approach``,
a per-trial transform), the return windows, and for every segmentation configuration the candidate segments
of every stream with their features and windows. ``run_experiment`` passes stream INDICES as X. Labels
(the hand) and the app's events (onset, end, end cue, return) come ONLY from ``fold.y_train`` /
``fold.meta_train``, so the held-out person's events and hands are never within reach.

Fitting one fold (:func:`fit_pipeline`):
1. segmentation configuration: the one of :data:`SEG_GRID` whose candidate segments contain the stored
   approach (IoU >= 0.5) in the most TRAINING streams (nested; the held-out person plays no part);
2. segment classifier (``m4_models.SegmentClassifier``) on the training streams' segments, labelled from the
   training rows' events (approach / return / other; partial approaches are left out of its training);
3. hand classifier on the training persons' oracle windows or detector windows (``spec.hand_train``);
4. calibration on CROSS-FITTED logits: ``inner_splits`` over the training persons, steps 2-3 refitted on the
   inner training part and run end to end (detector + hand model) on the inner validation part; the
   calibrator and the k-attempt aggregate scale are fitted on those logits.

Predicting a single recording: the candidate with the highest P(approach) is THE approach; the pipeline
abstains (P = 0.5, recorded) when there is no candidate or that probability is below ``spec.det_threshold``.
A stream with several attempts: every candidate whose most likely class is 'approach' and whose P(approach)
>= det_threshold is an attempt (:meth:`FittedPipeline.attempts`; ``spec.pair_returns`` adds the pairing rule).

Memoisation. Fits are memoised on everything that determines them: the CONTENT of the prepared data
(:meth:`Prepared.fingerprint`, a hash of its segment tables, windows and oracle / return windows), the spec, the
training rows with their labels, persons and events, and the seed. (An earlier key without the prepared data
let a variant on other streams reuse a fit made on the main streams in the same process.)
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.cuefree import CueFreeParams, Detection, classify, detect

from . import m4_config as CFG
from . import m4_detect as D
from . import m4_models as M

#: segmentation configurations compared inside each fold (lo_dps, end_sustain_ms); the first is v0's
SEG_GRID: dict[str, D.SegParams] = {
    'lo20_es250': D.SegParams(lo_dps=20.0, end_sustain_ms=250.0),
    'lo15_es250': D.SegParams(lo_dps=15.0, end_sustain_ms=250.0),
    'lo20_es400': D.SegParams(lo_dps=20.0, end_sustain_ms=400.0),
}


@dataclass(frozen=True)
class PipelineSpec:
    detector: str = 'learned'          # 'learned' (segments + classifier), 'v0' (cuefree v0, first detection)
    seg: str = 'nested'                # a SEG_GRID key or 'nested'
    features: str = 'physics'          # 'physics', 'full' (adds the stillness / sequence context) or 'nested'
    hand_model: str = 'yaw8'
    hand_train: str = 'oracle'         # 'oracle' windows or 'detector' windows of the training persons
    calibration: str = 'none'          # m4_models.CALIBRATORS key
    det_threshold: float = 0.5         # minimum P(approach) of a selected segment
    use_return: bool = False           # add the return head's logit when the next segment is a return
    inner_calibration: bool = True     # cross-fit on inner splits (needed by calibration / aggregate scale)
    seg_scaling: str = 'robust'        # segment-classifier feature scaling ('robust', 'z', 'none'; ablation)
    seg_C: Any = 1.0                   # segment-classifier C (a float, or 'nested'; ablation)
    pair_returns: bool = CFG.PRODUCTION_PAIRING   # attempt rule of a multi-touch stream (m4_config)

    def label(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# ───────────────────────────── label-free prepared data ─────────────────────────────


@dataclass
class Prepared:
    """Label-free per-stream data (module docstring). Index k = position in ``streams``."""
    streams: list[dict[str, np.ndarray]]
    oracle: np.ndarray                     # (N, 17, 64) cached approach windows
    ret: np.ndarray                        # (N, 17, 64) cached return windows (NaN where not observed)
    ret_ok: np.ndarray                     # (N,) bool
    seg: dict[str, tuple[pd.DataFrame, np.ndarray]] = field(default_factory=dict)
    v0: tuple[pd.DataFrame, np.ndarray] | None = None
    platform: np.ndarray | None = None     # per stream (production knows it)
    _fp: str | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.streams)

    def fingerprint(self) -> str:
        """Content hash of everything a fit reads from this object (segment tables and windows of every
        configuration, v0 detections, oracle and return windows): the memo key of the fits (module docstring).
        Computed once; the object must not be modified afterwards."""
        if self._fp is None:
            h = hashlib.sha256()
            for name in sorted(self.seg):
                tab, W = self.seg[name]
                h.update(name.encode())
                num = tab.select_dtypes(include=[np.number, bool]).to_numpy(dtype=float)
                h.update(np.ascontiguousarray(np.nan_to_num(num, nan=-9.87e9)).tobytes())
                h.update(','.join(map(str, tab.columns)).encode())
                h.update(np.ascontiguousarray(np.nan_to_num(W, nan=-9.87e9)).tobytes())
            if self.v0 is not None:
                num = self.v0[0].select_dtypes(include=[np.number, bool]).to_numpy(dtype=float)
                h.update(np.ascontiguousarray(np.nan_to_num(num, nan=-9.87e9)).tobytes())
                h.update(np.ascontiguousarray(np.nan_to_num(self.v0[1], nan=-9.87e9)).tobytes())
            for a in (self.oracle, self.ret, self.ret_ok.astype(np.int8)):
                h.update(np.ascontiguousarray(np.nan_to_num(np.asarray(a, dtype=np.float32), nan=-9.87e9)).tobytes())
            self._fp = h.hexdigest()[:24]
        return self._fp


def v0_table(streams: Sequence[Mapping[str, np.ndarray]], crop_s: float = 1.0) -> tuple[pd.DataFrame, np.ndarray]:
    """cuefree v0's FIRST detection of every stream and its canonical window (label-free)."""
    from stage2.standardize import CHANNELS
    rows, wins = [], []
    for k, st in enumerate(streams):
        dets = detect(st['t'], st['gyro'], CueFreeParams())
        if not dets:
            rows.append({'stream': k, 'found': False, 'onset_ms': math.nan, 'end_ms': math.nan, 'n_det': 0})
            wins.append(np.full((len(CHANNELS), C.N_NORM), np.nan, np.float32))
            continue
        dt = dets[0]
        try:
            W, _ = D.window_for(st, dt.onset_ms, dt.end_ms, crop_s)
            ok = True
        except ValueError:
            W, ok = np.full((len(CHANNELS), C.N_NORM), np.nan, np.float32), False
        rows.append({'stream': k, 'found': ok, 'onset_ms': dt.onset_ms, 'end_ms': dt.end_ms, 'n_det': len(dets)})
        wins.append(W)
    return pd.DataFrame(rows), np.stack(wins).astype(np.float32)


def prepare(streams: list[dict[str, np.ndarray]], oracle: np.ndarray, ret: np.ndarray, ret_ok: np.ndarray, *,
            platform: Sequence[str] | None = None, grid: Mapping[str, D.SegParams] | None = None,
            with_v0: bool = True, progress: str = '') -> Prepared:
    grid = SEG_GRID if grid is None else grid
    prep = Prepared(streams, oracle, ret, np.asarray(ret_ok, bool),
                    platform=None if platform is None else np.asarray(platform).astype(str))
    for name, p in grid.items():
        prep.seg[name] = D.segment_all(streams, p, progress=f'{progress} {name}' if progress else '')
    if with_v0:
        prep.v0 = v0_table(streams)
    return prep


# ───────────────────────────── labels of training segments (events of training rows only) ─────────────────


def training_labels(tab: pd.DataFrame, idx: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Segments of streams ``idx`` (training rows, in the order of ``meta``) with labels from ``meta``'s events
    (vectorised; ``train_pos`` = position of the segment's stream in ``idx``)."""
    idx = np.asarray(idx).astype(int)
    pos_of = pd.Series(np.arange(idx.size), index=idx)
    pos_of = pos_of[~pos_of.index.duplicated()]
    sel = tab['stream'].isin(pos_of.index).to_numpy()
    seg = tab[sel].copy()
    if not len(seg):
        return seg.assign(label=[], outcome=[], iou=[], train_pos=[])
    pos = pos_of.loc[seg['stream'].to_numpy()].to_numpy()
    ev = {c: pd.to_numeric(meta[c], errors='coerce').to_numpy(float)[pos] for c in D.EVENT_COLUMNS}
    oc, iou = D.classify_arrays(seg['onset_ms'].to_numpy(float), seg['end_ms'].to_numpy(float), ev['onset_ms'],
                                ev['end_ms'], ev['end_detected_ms'], ev['return_onset_ms'], ev['return_end_ms'])
    seg['outcome'] = oc
    seg['label'] = [D.LABEL_OF_OUTCOME[o] for o in oc]
    seg['iou'] = iou
    seg['train_pos'] = pos
    return seg


# ───────────────────────────── the fitted pipeline ─────────────────────────────


@dataclass
class FittedPipeline:
    spec: PipelineSpec
    seg_cfg: str | None
    segclf: M.SegmentClassifier | None
    hand: Any
    calib: M.Calibrator
    agg: M.AggregateScale | None
    ret_hand: Any = None
    info: dict[str, Any] = field(default_factory=dict)

    # ---- one recording: choose the approach ----
    def choose(self, prep: Prepared, idx: Sequence[int], *, seg_override: tuple[pd.DataFrame, np.ndarray] | None = None
               ) -> pd.DataFrame:
        """One row per stream: chosen segment, P(approach), raw hand logit, calibrated P(right), status."""
        rows = []
        if self.spec.detector == 'v0':
            tab, W = seg_override if seg_override is not None else prep.v0
            byk = {int(r['stream']): (i, r) for i, r in tab.reset_index(drop=True).iterrows()}
            for k in idx:
                i, r = byk.get(int(k), (None, None))
                if r is None or not bool(r['found']):
                    rows.append({'stream': int(k), 'status': 'none'})
                    continue
                z = float(self.hand.logit(W[i:i + 1])[0])
                rows.append({'stream': int(k), 'status': 'ok', 'onset_ms': r['onset_ms'], 'end_ms': r['end_ms'],
                             'p_approach': math.nan, 'z_raw': z})
        else:
            tab, W = seg_override if seg_override is not None else prep.seg[self.seg_cfg]
            tab = tab.reset_index(drop=True)
            P = self.segment_proba(tab)
            pos = {int(k): g.index.to_numpy() for k, g in tab.groupby('stream', sort=False)}
            for k in idx:
                ii = pos.get(int(k))
                if ii is None or not len(ii):
                    rows.append({'stream': int(k), 'status': 'none', 'n_seg': 0})
                    continue
                ok_c = tab.loc[ii, 'canonical_ok'].to_numpy(bool)
                pa = np.where(ok_c, P['approach'][ii], -1.0)
                j = int(ii[int(np.argmax(pa))])
                r = tab.loc[j]
                if not ok_c.any():
                    rows.append({'stream': int(k), 'status': 'none', 'n_seg': len(ii)})
                    continue
                z = float(self.hand.logit(W[j:j + 1])[0])
                status = 'ok' if pa.max() >= self.spec.det_threshold else 'no_approach'
                row = {'stream': int(k), 'status': status, 'n_seg': len(ii), 'p_approach': float(pa.max()),
                       'onset_ms': r['onset_ms'], 'end_ms': r['end_ms'], 'decided_ms': r['decided_ms'], 'z_raw': z,
                       'seg': int(r['seg']), 'up_z0': float(r['up_z0']) if 'up_z0' in r.index else math.nan}
                if self.spec.use_return and self.ret_hand is not None:
                    nxt = [q for q in ii if tab.loc[q, 'seg'] == r['seg'] + 1]
                    if nxt:
                        q = int(nxt[0])
                        if P['return'][q] >= max(P['approach'][q], P['other'][q]) and bool(tab.loc[q, 'canonical_ok']):
                            row['z_ret'] = float(self.ret_hand.logit(W[q:q + 1])[0])
                rows.append(row)
        out = pd.DataFrame(rows)
        for c in ('z_raw', 'p_approach', 'onset_ms', 'end_ms', 'z_ret'):
            if c not in out.columns:
                out[c] = np.nan
        z = out['z_raw'].to_numpy(float)
        if self.spec.use_return:
            z = z + np.nan_to_num(out['z_ret'].to_numpy(float))
        out['z'] = z
        pf = np.full(len(out), 0.5)
        fin = np.isfinite(z)
        if fin.any():
            pf[fin] = self.calib(z[fin])
        out['p_forced'] = pf                       # the chosen segment's P(right) even below det_threshold
        out['p_right'] = np.where((out['status'] == 'ok').to_numpy() & fin, pf, 0.5)
        return out

    def segment_proba(self, tab: pd.DataFrame) -> dict[str, np.ndarray]:
        assert self.segclf is not None
        if not len(tab):
            return {c: np.zeros(0) for c in D.CLASSES}
        return self.segclf.proba(tab[self.segclf.features].to_numpy(float))

    # ---- a stream with several attempts ----
    def attempts(self, tab: pd.DataFrame, W: np.ndarray, *, pair_returns: bool | None = None) -> pd.DataFrame:
        """Every accepted attempt of one stream (rows of ``tab``, its segments in time order) with the
        calibrated logit of its hand evidence.

        Pairing rule (``pair_returns``; default: ``spec.pair_returns``, i.e. the production rule of m4_config):
        the bout right after an accepted approach is that approach's return (the phone must come back before
        the next touch), so it is never an attempt, whatever the segment classifier says. A causal rule: it only
        looks at the order of the bouts."""
        pair_returns = self.spec.pair_returns if pair_returns is None else pair_returns
        tab = tab.reset_index(drop=True)
        if not len(tab):
            return pd.DataFrame(columns=['seg', 'onset_ms', 'end_ms', 'decided_ms', 'p_approach', 'z_raw', 'z'])
        P = self.segment_proba(tab)
        cls = np.array(D.CLASSES)[np.argmax(np.column_stack([P[c] for c in D.CLASSES]), axis=1)]
        keep = (cls == 'approach') & (P['approach'] >= self.spec.det_threshold) & tab['canonical_ok'].to_numpy(bool)
        if pair_returns:
            order = np.argsort(tab['onset_ms'].to_numpy(float), kind='stable')
            prev_accepted = False
            for i in order:
                if prev_accepted:
                    keep[i] = False                 # the return of the attempt just accepted
                    prev_accepted = False
                    continue
                prev_accepted = bool(keep[i])
        idx = np.flatnonzero(keep)
        if not idx.size:
            return pd.DataFrame(columns=['seg', 'onset_ms', 'end_ms', 'decided_ms', 'p_approach', 'z_raw', 'z'])
        zr = self.hand.logit(W[idx])
        z = M.logit(self.calib(zr))
        out = tab.loc[idx, ['seg', 'onset_ms', 'end_ms', 'decided_ms']].copy()
        out['p_approach'] = P['approach'][idx]
        out['p_return'] = P['return'][idx]
        out['z_raw'] = zr
        out['z'] = z
        out['up_z0'] = tab['up_z0'].to_numpy(float)[idx] if 'up_z0' in tab.columns else np.nan
        return out.reset_index(drop=True)


_CORE_MEMO: dict[tuple, tuple] = {}
_BASE_MEMO: dict[tuple, tuple] = {}


def _rows_hash(idx: np.ndarray, y: np.ndarray, groups: np.ndarray, meta: pd.DataFrame) -> str:
    """Identity of a training set: rows, labels, persons and the events its segment labels come from."""
    h = hashlib.sha256()
    ev = meta[list(D.EVENT_COLUMNS)].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    for a in (np.asarray(idx, np.int64), np.asarray(y, np.int64), np.asarray(groups).astype('U'), ev):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _fit_core(prep: Prepared, spec: PipelineSpec, cfg: str | None, idx: np.ndarray, y: np.ndarray,
              groups: np.ndarray, meta: pd.DataFrame, seed: int) -> tuple[M.SegmentClassifier | None, Any, Any, dict]:
    """Steps 2-3 (and the return head) on training streams ``idx``, memoised on everything that determines
    them (prepared data, detector / features / hand model / training windows, configuration, training rows with
    their labels and events, seed): variants that share a fit (calibrators, the return head) reuse it."""
    rh = _rows_hash(idx, y, groups, meta)
    key = (prep.fingerprint(), spec.detector, spec.features, spec.hand_model, spec.hand_train, spec.seg_scaling,
           str(spec.seg_C), cfg, rh, seed)
    if key not in _CORE_MEMO:
        _CORE_MEMO[key] = _fit_core_uncached(prep, replace(spec, use_return=False), cfg, idx, y, groups, meta, seed)
    segclf, hand, _, info = _CORE_MEMO[key]
    ret_hand = None
    if spec.use_return:                       # the return head: fitted on the training rows' return windows
        rkey = ('ret', prep.fingerprint(), spec.hand_model, rh, seed)
        if rkey not in _CORE_MEMO:
            ok = prep.ret_ok[idx]
            _CORE_MEMO[rkey] = M.make_hand_model(spec.hand_model).fit(prep.ret[idx[ok]], y[ok], groups[ok], seed)
        ret_hand = _CORE_MEMO[rkey]
    return segclf, hand, ret_hand, info


def _fit_core_uncached(prep: Prepared, spec: PipelineSpec, cfg: str | None, idx: np.ndarray, y: np.ndarray,
                       groups: np.ndarray, meta: pd.DataFrame, seed: int) -> tuple[M.SegmentClassifier | None, Any, Any, dict]:
    info: dict[str, Any] = {}
    segclf = None
    lab = None
    if spec.detector == 'learned':
        tab, W = prep.seg[cfg]
        lab = training_labels(tab, idx, meta)
        tr = lab[lab['label'].isin(D.CLASSES) & lab['canonical_ok'].astype(bool)]
        feats = list(D.FEATURE_SETS[spec.features])
        segclf = M.SegmentClassifier(feats, C=spec.seg_C, scaling=spec.seg_scaling).fit(
            tr[feats].to_numpy(float), tr['label'].to_numpy(), seed, groups=groups[tr['train_pos'].to_numpy(int)])
        info['n_train_segments'] = int(len(tr))
        info['seg_C'] = segclf.C_chosen
    hand = M.make_hand_model(spec.hand_model)
    if spec.hand_train == 'oracle' or spec.detector != 'learned':
        hand.fit(prep.oracle[idx], y, groups, seed)
    else:
        tab, W = prep.seg[cfg]
        a = lab[lab['label'] == 'approach']
        a = a[a['canonical_ok'].astype(bool)].drop_duplicates('train_pos')
        hand.fit(W[a.index.to_numpy()], y[a['train_pos'].to_numpy(int)], groups[a['train_pos'].to_numpy(int)], seed)
        info['n_hand_train_detector_windows'] = int(len(a))
    return segclf, hand, None, info


def choose_seg_cfg(prep: Prepared, idx: np.ndarray, meta: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """SEG_GRID entry whose candidates contain the stored approach (IoU >= 0.5) in most TRAINING streams."""
    scores = {}
    for name in prep.seg:
        lab = training_labels(prep.seg[name][0], idx, meta)
        found = lab[lab['label'] == 'approach']['train_pos'].nunique()
        scores[name] = found / max(len(idx), 1)
    best = max(scores, key=lambda k: (round(scores[k], 9), -list(prep.seg).index(k)))
    return best, scores


def fit_pipeline(prep: Prepared, spec: PipelineSpec, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 meta: pd.DataFrame, seed: int) -> FittedPipeline:
    """Fit the whole pipeline on training streams ``idx`` (module docstring, steps 1-4). Everything but the
    calibrator is memoised (:func:`_fit_base`), so calibration variants share one fit."""
    idx = np.asarray(idx).astype(int)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    meta = meta.reset_index(drop=True)
    cfg, spec_fit, segclf, hand, ret_hand, cross, info0 = _fit_base(prep, spec, idx, y, groups, meta, seed)
    info = dict(info0)
    spec_fit = replace(spec_fit, calibration=spec.calibration)
    calib: M.Calibrator = M.make_calibrator('none')
    agg = None
    if spec.inner_calibration:
        z_cf, y_cf, p_cf, _ = cross[spec_fit.features]
        calib = M.make_calibrator(spec.calibration).fit(z_cf, y_cf, groups=p_cf)
        agg = M.AggregateScale().fit(M.logit(calib(z_cf)), y_cf, p_cf, seed)
        # mirror deviation |P(z) + P(-z) - 1| over the logits the calibrator was fitted on: 0 for an antisymmetric
        # calibrator (a mirror-image recording then gets exactly 1 - P)
        dev = np.abs(calib(z_cf) + calib(-z_cf) - 1.0) if z_cf.size else np.zeros(1)
        info['calibration'] = {'method': spec.calibration, **calib.params(), 'n_crossfit': int(z_cf.size),
                               'mirror_dev_max': float(dev.max()), 'mirror_dev_mean': float(dev.mean())}
        info['agg_scale'] = agg.s
    if hasattr(hand, 'C_chosen'):
        info['hand_C'] = hand.C_chosen
    return FittedPipeline(spec_fit, cfg, segclf, hand, calib, agg, ret_hand, info)


def _fit_base(prep: Prepared, spec: PipelineSpec, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
              meta: pd.DataFrame, seed: int) -> tuple:
    """Steps 1-3 and the cross-fitted inner predictions (memoised; the calibrator is not part of it)."""
    key = (prep.fingerprint(), replace(spec, calibration='none').label(), _rows_hash(idx, y, groups, meta), seed)
    if key in _BASE_MEMO:
        return _BASE_MEMO[key]
    info: dict[str, Any] = {}
    cfg = None
    if spec.detector == 'learned':
        if spec.seg == 'nested':
            cfg, scores = choose_seg_cfg(prep, idx, meta)
            info['seg_scores'] = scores
        else:
            cfg = spec.seg
        info['seg_cfg'] = cfg
    candidates = [spec]
    if spec.features == 'nested':
        candidates = [replace(spec, features=f) for f in ('physics', 'full')]
    cross: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    if spec.inner_calibration or len(candidates) > 1:
        for cand in candidates:
            zs, ys, ps, accs = [], [], [], []
            for tr, va in cv.inner_splits(groups):
                s2, h2, r2, _ = _fit_core(prep, cand, cfg, idx[tr], y[tr], groups[tr],
                                          meta.iloc[tr].reset_index(drop=True), seed)
                fp = FittedPipeline(replace(cand, calibration='none'), cfg, s2, h2, M.make_calibrator('none'), None, r2)
                out = fp.choose(prep, idx[va])
                ok = (out['status'] == 'ok').to_numpy()
                zs.append(out['z'].to_numpy(float)[ok])
                ys.append(y[va][ok])
                ps.append(groups[va][ok])
                pr = out['p_right'].to_numpy(float)
                accs.append(np.mean(np.where(ok, (pr >= 0.5) == y[va], 0.5)))
            cross[cand.features] = (np.concatenate(zs), np.concatenate(ys), np.concatenate(ps), float(np.mean(accs)))
    if len(candidates) > 1:
        best = max(candidates, key=lambda c: (round(cross[c.features][3], 9), c.features == 'physics'))
        info['features_inner_acc'] = {k: v[3] for k, v in cross.items()}
        info['features_chosen'] = best.features
        spec_fit = best
    else:
        spec_fit = spec
    segclf, hand, ret_hand, inf = _fit_core(prep, spec_fit, cfg, idx, y, groups, meta, seed)
    info.update(inf)
    out = (cfg, spec_fit, segclf, hand, ret_hand, cross, info)
    _BASE_MEMO[key] = out
    return out


# ───────────────────────────── run_experiment adaptor ─────────────────────────────

_MEMO: dict[str, FittedPipeline] = {}
#: at most this many fitted pipelines are kept (oldest dropped first; fitting is deterministic, so a refit
#: after an eviction changes no number)
MEMO_MAX = 512


def _key(prep: Prepared, spec: PipelineSpec, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
         meta: pd.DataFrame, seed: int) -> str:
    """Memo key of a fitted pipeline: the prepared data's content, the spec, the training rows with their labels,
    persons and events, and the seed (module docstring)."""
    return (f'{prep.fingerprint()}|{spec.label()}|'
            f'{_rows_hash(np.asarray(idx), np.asarray(y), np.asarray(groups), meta.reset_index(drop=True))}|{seed}')


def clear_memos() -> None:
    """Forget every memoised fit (tests; a long process that changes data)."""
    _MEMO.clear()
    _BASE_MEMO.clear()
    _CORE_MEMO.clear()


def get_pipeline(prep: Prepared, spec: PipelineSpec, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 meta: pd.DataFrame, seed: int) -> FittedPipeline:
    """fit_pipeline, memoised (:func:`_key`): the leak check and the stream-level evaluations reuse the fold's
    fitted pipeline (fitting is deterministic, so this changes no number)."""
    k = _key(prep, spec, idx, y, groups, meta, seed)
    if k not in _MEMO:
        while len(_MEMO) >= MEMO_MAX:
            _MEMO.pop(next(iter(_MEMO)))
        _MEMO[k] = fit_pipeline(prep, spec, idx, y, groups, meta, seed)
    return _MEMO[k]


def make_fit_predict(prep: Prepared, spec: PipelineSpec, details: dict[int, dict[str, Any]] | None = None):
    """fit_predict for ``run_experiment`` with X = stream indices into ``prep`` (label-free). ``details``
    (optional) receives each test stream's chosen segment and status, keyed by stream index."""

    def fit_predict(fold: cv.Fold):
        idx = np.asarray(fold.X_train).astype(int).reshape(-1)
        fp = get_pipeline(prep, spec, idx, fold.y_train, fold.groups_train, fold.meta_train, fold.seed)
        test = np.asarray(fold.X_test).astype(int).reshape(-1)
        out = fp.choose(prep, test)
        if details is not None:
            for _, r in out.iterrows():
                details[int(r['stream'])] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                             for k, v in r.items()}
        info = {k: v for k, v in fp.info.items() if k in ('seg_cfg', 'calibration', 'agg_scale', 'hand_C',
                                                          'features_chosen', 'seg_C')}
        return out['p_right'].to_numpy(float), info

    return fit_predict


def fold_training_rows(meta: pd.DataFrame, person: str, seed: int = C.SEED) -> np.ndarray:
    """The training rows the harness gives the fold of ``person`` (everyone else, docs/09 D2 cap applied)."""
    g = meta['person_id'].astype(str).to_numpy()
    y = meta['y_right'].astype(int).to_numpy()
    tr = np.flatnonzero(g != person)
    return tr[cv.cap_rows(g[tr], y[tr], C.CAP_PER_PERSON, seed)]


def fold_inputs(meta: pd.DataFrame, person: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    """(training rows, their labels, their persons, their trial table, fold seed) of the outer fold that holds
    ``person`` out: exactly what run_experiment hands that fold's fit_predict."""
    splits = cv.loso_splits(meta['person_id'].astype(str).to_numpy())
    fold = next(s for s in splits if s.person == person)
    tr = fold_training_rows(meta, person)
    m = meta.iloc[tr].reset_index(drop=True)
    return tr, m['y_right'].astype(int).to_numpy(), m['person_id'].astype(str).to_numpy(), m, C.SEED + fold.fold


def pipeline_for_person(prep: Prepared, spec: PipelineSpec, meta: pd.DataFrame, person: str) -> FittedPipeline:
    """The pipeline of the outer fold that holds ``person`` out (same rows and seed as run_experiment's fold)."""
    tr, y, g, m, seed = fold_inputs(meta, person)
    return get_pipeline(prep, spec, tr, y, g, m, seed)


def inner_pipelines(prep: Prepared, fp: FittedPipeline, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    meta: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray, FittedPipeline]]:
    """The cross-fitting pipelines of a fitted pipeline: for each inner split (``cv.inner_splits`` over the
    training persons, as in :func:`_fit_base`) the segment classifier and hand model fitted on the inner
    training part (memoised: the fit that produced the calibrator's logits), with the OUTER calibrator.
    Returns [(inner train positions, inner validation positions, pipeline)] (positions into ``idx``)."""
    idx = np.asarray(idx).astype(int)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups).astype(str)
    meta = meta.reset_index(drop=True)
    out = []
    for tr, va in cv.inner_splits(groups):
        s2, h2, r2, _ = _fit_core(prep, fp.spec, fp.seg_cfg, idx[tr], y[tr], groups[tr],
                                  meta.iloc[tr].reset_index(drop=True), seed)
        out.append((tr, va, FittedPipeline(fp.spec, fp.seg_cfg, s2, h2, fp.calib, None, r2)))
    return out


# ───────────────────────────── detection outcome of the chosen segment ─────────────────────────────


def detection_outcomes(chosen: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """cuefree.classify of each stream's chosen segment against its trial's events (evaluation only)."""
    rows = []
    for _, r in chosen.iterrows():
        k = int(r['stream'])
        ev = trials.iloc[k]
        if r.get('status') in ('none',) or not np.isfinite(r.get('onset_ms', np.nan)):
            c = {'outcome': 'none', 'onset_error_ms': math.nan, 'end_error_ms': math.nan, 'iou': 0.0}
        else:
            det = Detection(float(r['onset_ms']), float(r['end_ms']), 0, 0, 0.0, 0.0, True)
            c = classify(det, float(ev['onset_ms']), float(ev['end_ms']), cue_ms=D._num(ev.get('end_detected_ms')),
                         return_onset_ms=D._num(ev.get('return_onset_ms')), return_end_ms=D._num(ev.get('return_end_ms')))
            if r.get('status') == 'no_approach':
                c = {**c, 'outcome': 'rejected'}
        rows.append({'stream': k, **c})
    return pd.DataFrame(rows)
