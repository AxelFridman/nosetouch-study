"""Reference implementation of the exported final model: numpy only (no scikit-learn, scipy or torch), so that the
Stage 3 JavaScript port has an exact, readable specification to match.

    import json
    from stage2.lanes.final_infer import load, predict_proba
    model = load('analysis/reports/stage2/export/final_model.json')
    p = predict_proba(model, approach)          # approach: (N, 17, 64) canonical windows -> P(right), shape (N,)

Input: the canonical approach window of ``standardize.canonical`` (17 channels in ``model['input']['channels']``
order, 64 samples from the detected onset to the detected end), exactly as the cache stores it.
Steps: the body-frame channels (roll rate, pitch rate, lateral and forward acceleration) from the up vector,
angular velocity and linear acceleration; the model (a dot product on time-binned features, or the 1D CNN ensemble);
the calibrator; optionally the rest-pose screen check (un-flip a phone held with the screen away) and the
multi-attempt aggregate sum(logit) / tau_k.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sigmoid(z: np.ndarray, clip: float | None = None) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


# ───────────────────────────── inputs ─────────────────────────────


def body_channels(X: np.ndarray, channels: list[str], h_min: float) -> np.ndarray:
    """(N, 17, T) -> (N, 4, T): roll rate w.h, pitch rate w.s, lateral acceleration a.s, forward acceleration a.h, with
    h = (e_z - (e_z.u) u) / |.| (horizontal, out of the screen) and s = u x h."""
    ix = {c: i for i, c in enumerate(channels)}
    X = np.asarray(X, dtype=np.float64)
    u = X[:, [ix['up_x'], ix['up_y'], ix['up_z']], :]
    w = X[:, [ix['gyro_x'], ix['gyro_y'], ix['gyro_z']], :]
    a = X[:, [ix['acc_x'], ix['acc_y'], ix['acc_z']], :]
    h = -u[:, 2:3, :] * u
    h[:, 2, :] += 1.0
    nrm = np.sqrt(np.sum(h * h, axis=1, keepdims=True))
    ok = nrm > h_min
    h = np.where(ok, h / np.where(ok, nrm, 1.0), 0.0)
    s = np.stack([u[:, 1] * h[:, 2] - u[:, 2] * h[:, 1], u[:, 2] * h[:, 0] - u[:, 0] * h[:, 2],
                  u[:, 0] * h[:, 1] - u[:, 1] * h[:, 0]], axis=1)
    out = np.stack([np.sum(w * h, axis=1), np.sum(w * s, axis=1), np.sum(a * s, axis=1), np.sum(a * h, axis=1)], axis=1)
    return out.astype(np.float32) if np.asarray(X).dtype == np.float32 else out


def extended(model: dict[str, Any], approach: np.ndarray) -> tuple[np.ndarray, list[str]]:
    inp = model['input']
    A = np.asarray(approach, dtype=np.float32)
    B = body_channels(A, inp['channels'], inp['body_frame']['h_min']).astype(np.float32)
    return np.concatenate([A, B], axis=1), list(inp['channels']) + list(inp['body_frame']['channels'])


def screen_away(model: dict[str, Any], rest_up_z: np.ndarray) -> np.ndarray:
    """The rest-pose check: the median screen-normal up component over the rest before the movement below the
    threshold means the screen faces away."""
    return np.asarray(rest_up_z, dtype=np.float64) < float(model['screen_check']['threshold'])


def unflip(model: dict[str, Any], approach: np.ndarray) -> np.ndarray:
    """The recorded window of a phone held with the screen away, mapped back to the screen-toward grip (the screen-flip
    signs of the channels; the flip is its own inverse)."""
    s = np.asarray(model['screen_check']['flip_signs'], dtype=np.float32).reshape(1, -1, 1)
    return np.asarray(approach, dtype=np.float32) * s


# ───────────────────────────── models ─────────────────────────────


def _bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    T = x.shape[-1]
    e = np.round(np.linspace(0, T, n_bins + 1)).astype(int)
    return np.stack([x[..., e[b]:e[b + 1]].mean(-1) for b in range(n_bins)], axis=-1)


def logits_linear_tc(m: dict[str, Any], X: np.ndarray, names: list[str]) -> np.ndarray:
    idx = [names.index(c) for c in m['channels']]
    F = np.concatenate([_bins(np.asarray(X[:, i, :], dtype=np.float64), m['n_bins']) for i in idx], axis=1)
    return (F / np.asarray(m['scale'], dtype=np.float64)) @ np.asarray(m['weights'], dtype=np.float64)


def _conv1d_same(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x (N, Cin, T), w (Cout, Cin, K), zero padding (K - 1) // 2 on both sides, no bias (cross-correlation, as in
    torch.nn.Conv1d)."""
    N, Cin, T = x.shape
    Cout, _, K = w.shape
    pad = (K - 1) // 2
    xp = np.zeros((N, Cin, T + 2 * pad), dtype=x.dtype)
    xp[:, :, pad:pad + T] = x
    cols = np.stack([xp[:, :, k:k + T] for k in range(K)], axis=-1)          # (N, Cin, T, K)
    return np.einsum('nctk,ock->not', cols, w)


def logits_cnn(m: dict[str, Any], X: np.ndarray, names: list[str]) -> np.ndarray:
    """Mean logit of the CNN members (float64): per member normalise, 3 x (conv, batch norm, ReLU), average-pool to
    n_bins time bins, linear layer."""
    idx = [names.index(c) for c in m['channels']]
    x0 = np.asarray(X[:, idx, :], dtype=np.float64)
    zs = []
    for mem in m['members']:
        c = np.asarray(mem['center'], dtype=np.float64).reshape(1, -1, 1)
        s = np.asarray(mem['scale'], dtype=np.float64).reshape(1, -1, 1)
        h = (x0 - c) / s
        for layer in mem['layers']:
            h = _conv1d_same(h, np.asarray(layer['weight'], dtype=np.float64))
            g = np.asarray(layer['bn_weight']).reshape(1, -1, 1)
            b = np.asarray(layer['bn_bias']).reshape(1, -1, 1)
            mu = np.asarray(layer['bn_mean']).reshape(1, -1, 1)
            var = np.asarray(layer['bn_var']).reshape(1, -1, 1)
            h = (h - mu) / np.sqrt(var + layer['bn_eps']) * g + b
            h = np.maximum(h, 0.0)
        pooled = _bins(h, m['n_bins'])                                         # (N, width, n_bins)
        flat = pooled.reshape(pooled.shape[0], -1)
        zs.append(flat @ np.asarray(mem['fc_weight'], dtype=np.float64).reshape(-1) + float(mem['fc_bias']))
    return np.mean(zs, axis=0)


def model_logits(model: dict[str, Any], approach: np.ndarray) -> np.ndarray:
    X, names = extended(model, approach)
    m = model['model']
    if m['kind'] == 'linear_time_bins':
        return logits_linear_tc(m, X, names)
    if m['kind'] == 'cnn1d_ensemble':
        return logits_cnn(m, X, names)
    raise ValueError(f'unknown model kind {m["kind"]!r}')


def calibrate(model: dict[str, Any], z: np.ndarray) -> np.ndarray:
    c = model['calibrator']
    if c['method'] == 'stacked':                    # the model's own calibrator, then the shipped remedy on its logit
        p = calibrate({'calibrator': c['inner']}, z)
        return calibrate({'calibrator': c['outer']}, logit(p))
    k = c['method']
    if k == 'none':
        p = sigmoid(z, c.get('logit_clip'))
    elif k == 'temperature':
        p = sigmoid(np.asarray(z) / c['T'], c.get('logit_clip'))
    elif k == 'platt':
        p = sigmoid(c['a'] * np.asarray(z) + c['b'], c.get('logit_clip'))
    elif k == 'beta':
        p0 = np.clip(sigmoid(z), c.get('p0_clip', 0.0), 1 - c.get('p0_clip', 0.0))
        p = sigmoid(c['a'] * np.log(p0) - c['b'] * np.log1p(-p0) + c['c'])
    elif k == 'isotonic':
        p = np.interp(np.asarray(z, dtype=np.float64), np.asarray(c['x']), np.asarray(c['y']))
    else:
        raise ValueError(k)
    lo = c.get('p_clip')
    return np.clip(p, lo, 1 - lo) if lo else p


def predict_proba(model: dict[str, Any], approach: np.ndarray, rest_up_z: np.ndarray | None = None) -> np.ndarray:
    """P(right) of every window. With ``rest_up_z`` (the rest pose before each movement) the screen check un-flips
    the windows of a phone held with the screen away before the model sees them."""
    A = np.asarray(approach, dtype=np.float32)
    if rest_up_z is not None:
        away = screen_away(model, rest_up_z)
        A = np.where(away[:, None, None], unflip(model, A), A)
    return calibrate(model, model_logits(model, A))


def aggregate(model: dict[str, Any], p_attempts: np.ndarray) -> float:
    """P(right) shown after k attempts of one person: sigmoid(sum of logits / tau_k) (tau_k fitted on cross-fitted
    predictions of the training persons; k above the largest fitted k uses the largest)."""
    z = logit(np.asarray(p_attempts, dtype=np.float64))
    k = int(z.size)
    taus = {int(a): float(b) for a, b in model['aggregation']['tau'].items()}
    tau = taus.get(k, taus[max(taus)]) if k > 1 else 1.0
    return float(sigmoid(z.sum() / tau))


def apply_prior(p: np.ndarray, prior_right: float) -> np.ndarray:
    """Shift P(right) to a deployment prior of the right hand: logit + log(pi / (1 - pi)) (the model assumes 0.5)."""
    return sigmoid(logit(p) + np.log(prior_right / (1 - prior_right)))


__all__ = ['load', 'predict_proba', 'model_logits', 'calibrate', 'aggregate', 'apply_prior', 'screen_away', 'unflip',
           'body_channels']
