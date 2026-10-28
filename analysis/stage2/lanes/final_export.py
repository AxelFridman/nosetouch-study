"""Lane final: fit the final pipeline on ALL persons and export it as plain JSON (task 8).

``export(key)`` writes ``reports/stage2/export/final_model.json``: the input (channel table and the preprocessing
parameters of ``stage2.config`` / ``standardize``), the body-frame definition, the model (weights as plain arrays),
the calibrator, the multi-attempt temperatures tau_k, the rest-pose screen check, the prior, the version and the
cache data identity; and ``final_model_check.json``: the fitted pipeline's P(right) of every primary and natural-use
trial (by trial key), which ``final_infer`` must reproduce to within 1e-9 (``tests/test_final_export.py``).

Fitting on all persons uses the same procedure as inside every outer fold: the D2 cap on the training rows, the
candidate's own nested tuning, the calibrator chosen by ``final_finalmodel.calibration`` fitted on cross-fitted
predictions of the persons (GroupKFold over all persons), and tau_k fitted on the same cross-fitted predictions.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from stage2 import config as C
from stage2 import cv
from stage2.pipeline import Lane

L = Lane('final')
EXPORT_DIR = C.OUT_DIR / 'export'
FORMAT_VERSION = 'nosetouch-final-1'


def _all_fold(X: Any, d: Any, seed: int = C.SEED) -> cv.Fold:
    """A fold whose training side is every person (D2 cap applied) and whose test side is every primary trial."""
    persons = d.trials['person_id'].astype(str).to_numpy()
    y = d.y.astype(int)
    tr = cv.cap_rows(persons, y, C.CAP_PER_PERSON, seed)
    m_test = cv._test_meta(d.trials.reset_index(drop=True), np.arange(len(y)), cv.TEST_META_COLUMNS)
    m_test.insert(0, 'row', np.arange(len(y)))
    return cv.Fold(-1, 'none', cv._take(X, tr), y[tr].copy(), persons[tr].copy(),
                   d.trials.iloc[tr].reset_index(drop=True), cv._take(X, np.arange(len(y))), m_test, seed)


def _cross_fitted(fp: Any, fold: cv.Fold) -> np.ndarray:
    """Cross-fitted P(right) of every training row of ``fold`` (GroupKFold over its persons)."""
    from stage2.lanes.final_nested import inner_fold
    p = np.full(fold.y_train.size, np.nan)
    for k, (itr, iva) in enumerate(fold.inner_splits()):
        f, order = inner_fold(fold, itr, iva, k)
        out = fp(f)
        p[order] = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=float)
    return p


def _taus(p_cf: np.ndarray, fold: cv.Fold) -> dict[str, float]:
    from stage2.lanes import final_calibration as FCal
    out = {}
    z = FCal.logit(p_cf)
    for k in FCal.AGG_KS:
        if k == 1:
            out['1'] = 1.0
            continue
        t = FCal.k_subsets(z, fold.groups_train, fold.y_train, k, 40, fold.seed)
        out[str(k)] = FCal.fit_tau(t['sum_z'].to_numpy(), t['y'].to_numpy())
    return out


def _identity() -> dict[str, Any]:
    from stage2.pipeline import cache_meta, provenance
    m = cache_meta() or {}
    return {**provenance(), 'cache_parameters': m.get('parameters'), 'code': (m.get('code') or {})}


def _input_block() -> dict[str, Any]:
    from stage2 import standardize as S
    from stage2.lanes.m3_body import BODY_CHANNELS, H_MIN
    params = {k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, bool))
              and any(t in k for t in ('LOWPASS', 'FS', 'GAP', 'APPROACH', 'FIXED', 'ONSET', 'GRAVITY', 'MAX_', 'MIN_'))}
    return {'window': 'the approach from the detected onset to the detected end, time-normalised to 64 samples '
                      '(standardize.canonical; README section 3)',
            'n_samples': 64, 'channels': list(S.CHANNELS),
            'channel_info': {c: list(v[:4]) for c, v in S.CHANNEL_INFO.items()},
            'preprocessing': params,
            'body_frame': {'channels': list(BODY_CHANNELS), 'h_min': H_MIN,
                           'definition': 'h = (e_z - (e_z.u) u) / |e_z - (e_z.u) u|, s = u x h; roll_rate = w.h, '
                                         'pitch_rate = w.s, acc_lat = a.s, acc_fwd = a.h (device frame, per sample)'}}


def _screen_block() -> dict[str, Any]:
    from stage2.lanes.m4_config import GRIP_GATE_UPZ
    from stage2.standardize import SCREEN_FLIP_SIGN
    return {'rule': 'median of up_z over the rest before the movement (0.5 s) below the threshold: the screen faces '
                    'away; multiply the device-frame channels by the screen-flip signs (un-flip) before the model, or '
                    'ask the user to turn the phone', 'threshold': float(GRIP_GATE_UPZ),
            'flip_signs': [float(v) for v in SCREEN_FLIP_SIGN]}


# ───────────────────────────── per-family export ─────────────────────────────


def _export_lr_body_tc(cand: Any, d: Any, nat: Any) -> tuple[dict[str, Any], Any]:
    from stage2.lanes import final_models as FM
    from stage2.lanes.m3_body import x_approach_body
    X = x_approach_body(d)
    fold = _all_fold(X, d)
    store: dict[int, Any] = {}
    fp = FM.make_lr_body_tc(store=store)
    p_all, info = fp(fold)
    st = store[-1]
    model = {'kind': 'linear_time_bins', 'channels': list(FM.TC_CHANNELS), 'n_bins': FM.TC_BINS,
             'scale': st['scale'], 'weights': st['weights'], 'C': st['C'], 'intercept': 0.0,
             'n_parameters': 2 * len(st['weights']),
             'formula': 'z = sum_j w_j * f_j / s_j, f = channel means over 8 equal bins of the time-normalised '
                        'approach (channel-major), no intercept: P(mirror image) = 1 - P exactly'}

    from stage2.lanes.m3_body import body_channels

    def predict(A: np.ndarray) -> np.ndarray:                 # the fitted pipeline's logits (training code path)
        F = FM.tc_features(np.concatenate([A, body_channels(A)], axis=1))
        return (F / np.asarray(st['scale'])) @ np.asarray(st['weights'])

    def own_calibrate(z: np.ndarray) -> np.ndarray:           # the fitted pipeline's own link (no calibrator)
        return FM.sigmoid(z)
    return model, (fp, fold, predict, X, own_calibrate)


def _export_cnn(cand: Any, d: Any, nat: Any) -> tuple[dict[str, Any], Any]:
    import torch
    from stage2.lanes import m3_body, m3_common as M, m3_deep
    params = {k: v for k, v in cand.params.items() if k not in ('calib', 'use_cache', 'cache_dir')}
    fam = m3_body.BodyDeepFamily(**params)
    X = m3_body.x_approach_body(d)
    fold = _all_fold(X, d)
    splits = fold.inner_splits(fam.n_inner)
    from stage2.lanes.final_candidates import FIT_CACHE_M3
    with M.cache_settings(True, str(FIT_CACHE_M3)):
        state = M.cached_fit(fam, M._rows_dict(fold.X_train), fold.y_train.astype(int),
                             fold.groups_train.astype(str), splits, fold.seed)
    par = M.fit_calibrator(cand.params.get('calib', 'none'), state['oof_z'], fold.y_train.astype(int),
                           fold.groups_train.astype(str))
    members = []
    for mem in state['members']:
        sd = mem['state']
        layers = []
        for i in range(3):
            layers.append({'weight': sd[f'body.{i}.0.weight'].double().numpy().tolist(),
                           'bn_weight': sd[f'body.{i}.1.weight'].double().numpy().tolist(),
                           'bn_bias': sd[f'body.{i}.1.bias'].double().numpy().tolist(),
                           'bn_mean': sd[f'body.{i}.1.running_mean'].double().numpy().tolist(),
                           'bn_var': sd[f'body.{i}.1.running_var'].double().numpy().tolist(), 'bn_eps': 1e-5})
        members.append({'center': mem['norm'].center.astype(np.float64).tolist(),
                        'scale': mem['norm'].scale.astype(np.float64).tolist(), 'layers': layers,
                        'fc_weight': sd['head.fc.weight'].double().numpy().reshape(-1).tolist(),
                        'fc_bias': float(sd['head.fc.bias'].double().numpy()[0]), 'epochs': mem['epochs']})
    n_par = int(sum(np.asarray(l['weight']).size + 4 * len(l['bn_weight']) for l in members[0]['layers'])
                + len(members[0]['fc_weight']) + 1)
    model = {'kind': 'cnn1d_ensemble', 'channels': list(fam.channel_names()), 'n_bins': int(fam.p['n_bins']),
             'kernels': [7, 5, 3], 'width': int(fam.p['width']), 'members': members,
             'n_parameters': n_par * len(members),
             'formula': 'per member: z-score per channel, 3 x (conv1d same padding, batch norm, ReLU), mean over 8 '
                        'equal time bins, linear; ensemble logit = mean of the members; then the calibrator'}
    calib = {'method': par['method'], **{k: v for k, v in par.items() if k != 'method'}, 'logit_clip': 40.0,
             'p_clip': M.P_CLIP if par['method'] != 'none' else None}

    def own_calibrate(z: np.ndarray) -> np.ndarray:           # the fitted pipeline's own calibrator (lane m3 code)
        return M.apply_calibrator(par, np.asarray(z, dtype=np.float64))

    def predict(A: np.ndarray) -> np.ndarray:                 # the fitted pipeline's own logits (float32 torch)
        return fam.logits(state, {'x': np.concatenate([A, m3_body.body_channels(A)], axis=1).astype(np.float32)})

    def predict64(A: np.ndarray) -> np.ndarray:               # the same networks evaluated in float64 (torch)
        Xe = np.concatenate([A, m3_body.body_channels(A)], axis=1).astype(np.float32)
        Xs = fam._select({'x': Xe}).astype(np.float64)
        zs = []
        with m3_deep.torch_runtime():
            for mem in state['members']:
                net = fam._member_net(mem).double()
                c = mem['norm'].center.astype(np.float64).reshape(1, -1, 1)
                s = mem['norm'].scale.astype(np.float64).reshape(1, -1, 1)
                with torch.no_grad():
                    zs.append(net(torch.from_numpy((Xs - c) / s)).numpy())
        return np.mean(zs, axis=0)
    return model, (None, fold, predict, X, calib, predict64, own_calibrate)


def export(key: str) -> dict[str, Any]:
    from stage2.data import load_cache
    from stage2.lanes import final_calibration as FCal
    from stage2.lanes import final_candidates as FC
    from stage2.lanes import final_infer as FI
    from stage2.lanes import final_models as FM
    t0 = time.time()
    cands = FC.candidates()
    cand = cands[key]
    d, nat = load_cache(), load_cache(('natural_use',))
    cal = L.read_json('final_calibration') if (L.tab_dir / 'final_calibration.json').exists() else {}
    chosen = cal.get('chosen', 'none')
    if key == 'lr_body_tc':
        model, (fp, fold, predict, X, own_calibrate) = _export_lr_body_tc(cand, d, nat)
        p_cf = _cross_fitted(fp, fold)
        base_calib = {'method': 'none'}
        predict64 = predict
    elif cand.deep and cand.module == 'stage2.lanes.m3_finalists' and cand.factory == 'deep_body':
        model, (_, fold, predict, X, base_calib, predict64, own_calibrate) = _export_cnn(cand, d, nat)
        p_cf = _cross_fitted(cand.make(), fold)
    else:
        raise NotImplementedError(f'no exporter for candidate {key!r} (add one to final_export)')
    remedy = None
    if chosen != 'none':                                      # the shipped remedy, fitted on cross-fitted predictions
        remedy, cpar = FCal.fit_calibrator(chosen, FCal.logit(p_cf), fold.y_train)
        calib = {'method': 'stacked', 'inner': base_calib, 'outer': {'method': chosen, **cpar}}
        p_cf = np.clip(remedy(FCal.logit(p_cf)), 0.0, 1.0)
    else:
        calib = base_calib

    def fitted_proba(z: np.ndarray) -> np.ndarray:            # the fitted pipeline: its own calibrator, then the remedy
        p = own_calibrate(z)
        return p if remedy is None else np.clip(remedy(FCal.logit(p)), 0.0, 1.0)
    taus = _taus(p_cf, fold)
    doc = {'format': FORMAT_VERSION, 'candidate': key, 'label': cand.label, 'identity': _identity(),
           'input': _input_block(), 'model': model, 'calibrator': calib,
           'aggregation': {'rule': 'P_k = sigmoid(sum of the k attempts\' logits / tau_k)', 'tau': taus},
           'screen_check': _screen_block(), 'prior_right': 0.5,
           'decision': {'threshold': 0.5, 'note': 'P(right) > 0.5 answers right, < 0.5 left; exactly 0.5 abstains'},
           'training': {'n_trials': int(fold.y_train.size), 'n_persons': int(len(set(fold.groups_train.tolist()))),
                        'cap_per_person': C.CAP_PER_PERSON, 'seed': C.SEED}}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    from stage2.report_export import write_text_atomic
    write_text_atomic(EXPORT_DIR / 'final_model.json', json.dumps(doc, indent=1) + '\n')
    # the fitted pipeline's P(right) of every trial, and the reference implementation's
    rows = []
    for which, data in (('primary', d), ('natural_use', nat)):
        A = np.asarray(data.approach, dtype=np.float32)
        p_fit = fitted_proba(predict64(A))                    # the fitted networks / weights in float64, own code
        p_ref = FI.predict_proba(doc, A)                      # the numpy reference from the JSON
        p32 = fitted_proba(predict(A))                        # the fitted pipeline as evaluated (float32 torch)
        for tk, a, b, c32 in zip(data.trials['trial_key'].astype(str), p_fit, p_ref, p32):
            rows.append({'trial_key': tk, 'set': which, 'p_fitted': float(a), 'p_reference': float(b),
                         'p_fitted_float32': float(c32)})
    chk = pd.DataFrame(rows)
    diff = float(np.max(np.abs(chk['p_fitted'] - chk['p_reference'])))
    diff32 = float(np.max(np.abs(chk['p_fitted'] - chk['p_fitted_float32'])))
    write_text_atomic(EXPORT_DIR / 'final_model_check.json',
                      json.dumps({'candidate': key, 'max_abs_diff_reference': diff, 'max_abs_diff_float32': diff32,
                                  'rows': rows}, indent=0) + '\n')
    # timing of the reference implementation (one window at a time)
    A1 = np.asarray(d.approach[:1], dtype=np.float32)
    t1 = time.perf_counter()
    for _ in range(50):
        FI.predict_proba(doc, A1)
    ms = (time.perf_counter() - t1) / 50 * 1000
    size_kb = (EXPORT_DIR / 'final_model.json').stat().st_size / 1024
    summary = {'candidate': key, 'calibrator': chosen, 'max_abs_diff_reference': diff,
               'max_abs_diff_float32': diff32, 'ms_per_prediction_numpy': ms, 'json_kb': size_kb,
               'n_parameters': model.get('n_parameters'), 'tau': taus, 'seconds': round(time.time() - t0, 1),
               'n_check_rows': int(len(chk))}
    from stage2.lanes.final_io import clean
    L.write_json('final_export', clean(summary))
    model_card(key, doc, summary)
    print(f'final: exported {key} ({size_kb:.0f} kB; reference max |dP| {diff:.1e}; {ms:.2f} ms per prediction)',
          flush=True)
    return summary


def _rd(name: str) -> Any:
    from stage2.lanes.final_io import denull
    return denull(L.read_json(name)) if (L.tab_dir / f'{name}.json').exists() else {}


def three_attempts(master: dict[str, Any]) -> str:
    """The displayed aggregate at k = 3 (sum of logits / tau_3) when the aggregation has been derived."""
    ag = _rd('final_aggregation')
    v = ((ag.get('rules') or {}).get('3') or {}).get('sum_tau')
    if v:
        return (f'{100 * v["accuracy"]:.1f} % per person and hand with sigmoid(sum of logits / tau_3), confidence minus '
                f'accuracy {v["over_confidence"]:+.3f} [{v["over_ci"][0]:+.3f}, {v["over_ci"][1]:+.3f}]')
    return f'{100 * master["k3"]:.1f} % (mean P, harness k-curve)'


def failure_modes(key: str, rob: dict[str, Any]) -> str:
    """Known failure modes, each with its number from the saved derived files."""
    out = []
    br = _rd('final_breakdown')
    if br.get('persons'):
        low = min(br['persons'], key=lambda r: r['accuracy'])
        out.append(f'some persons are misread consistently (lowest {low["person_id"]} {100 * low["accuracy"]:.0f} %), and '
                   f'more attempts do not fix them')
    vs = [r for r in br.get('variant', []) if r['persons'] >= br.get('min_persons_ci', 5)]
    if vs:
        wv = min(vs, key=lambda r: r['accuracy'])
        out.append(f'unusual movements (weakest variant: {wv["variant"]} {100 * wv["accuracy"]:.0f} %)')
    g = rob.get('grip') or {}
    if g:
        out.append(f'a grip absent from the training data ({100 * g["d_acc"]:+.1f} points on rotated grips when trained '
                   f'on upright ones)')
    fl = rob.get('flip') or {}
    if fl and not fl.get('invariant_inputs'):
        out.append(f'a screen held away without the rest-pose check ({100 * fl["acc_flipped"]:.0f} %)')
    comp = (((_rd('final_natural').get('components') or {}).get(key)) or {}).get('rows') or []
    lat = next((r for r in comp if r['component'] == 'lateral'), None)
    if lat:
        out.append(f'uninstructed movements carry less lateral evidence (it points to the reported hand in '
                   f'{lat["natural_pointing_right_way"]} of {lat["natural_n"]} free-choice trials)')
    out.append('windows from a detector that misses the onset or end (final_variants)')
    return '; '.join(out)


def model_card(key: str, doc: dict[str, Any], summary: dict[str, Any]) -> None:
    """The model card (tables/final/final_model_card.json): every value read from the saved derived files."""
    rd = _rd
    dec = rd('final_decision')
    master = {r['key']: r for r in rd('final_master')['rows']}[key]
    cal = rd('final_calibration') if (L.tab_dir / 'final_calibration.json').exists() else {}
    rob = {r['key']: r for r in rd('final_robustness')['rows']}.get(key, {})
    nat = {r['key']: r for r in rd('final_natural')['rows']}.get(key, {})
    ch = cal.get('chosen', 'none')
    ov = ((cal.get('methods') or {}).get(ch) or {}).get('overall', {})
    fl = rob.get('flip') or {}
    items = [
        ('model', f'{master["label"]} ({master["family"]})'),
        ('intended use', 'P(right) that the right hand brought the phone from arm\'s length to the nose, from the '
                         'phone\'s motion sensors only (no demographic, no instructed condition)'),
        ('inputs', f'canonical approach window: {len(doc["input"]["channels"])} channels x 64 samples; '
                   f'body-frame channels derived from it ({", ".join(doc["input"]["body_frame"]["channels"])})'),
        ('preprocessing', 'the cache pipeline (60 Hz, 15 Hz zero-phase low-pass, gyro-propagated gravity, onset to end '
                          'time-normalised to 64 samples); README section 3'),
        ('parameters', f'{doc["model"].get("n_parameters")} fitted numbers; JSON {summary["json_kb"]:.0f} kB; '
                       f'{summary["ms_per_prediction_numpy"]:.2f} ms per prediction in numpy'),
        ('calibrator', ch),
        ('accuracy (LOSO)', f'{100 * master["accuracy"]:.1f} % [{100 * master["accuracy_ci"][0]:.1f}, '
                            f'{100 * master["accuracy_ci"][1]:.1f}]'),
        ('log loss', f'{master["log_loss"]:.3f} [{master["log_loss_ci"][0]:.3f}, {master["log_loss_ci"][1]:.3f}]'),
        ('AUC', f'{master["auc"]:.3f}'),
        ('calibration', (f'ECE {ov.get("ece", float("nan")):.3f} (perfect-calibration band up to '
                         f'{ov.get("ece_null_hi", float("nan")):.3f}), slope {ov.get("slope", float("nan")):.2f}')
         if ov else 'see final_calibration'),
        ('three attempts', three_attempts(master)),
        ('natural use', f'right-hand recall {nat.get("observed", float("nan")):g} of {nat.get("n", 0)} free-choice '
                        f'trials (expected {nat.get("expected", float("nan")):.1f} from the same persons\' instructed '
                        f'right-hand trials)'),
        ('screen away', (f'{100 * fl.get("acc_flipped", float("nan")):.0f} % without the rest-pose check, '
                         f'{100 * fl.get("acc_flipped_remedy", float("nan")):.0f} % with it (check exported)')
         if fl and not fl.get('invariant_inputs') else 'inputs unchanged by the flip'),
        ('known failure modes', failure_modes(key, rob)),
        ('prior', 'P(right) assumes equal prior odds of the two hands (balanced training data); '
                  'final_infer.apply_prior shifts it'),
        ('rule check', f'the decision rule recommends {dec["recommended"]}; FINAL_MODEL = {dec["final_model"]}'),
    ]
    from stage2.lanes.final_io import clean
    L.write_json('final_model_card', clean({'items': items}))


__all__ = ['export', 'model_card', 'EXPORT_DIR']
