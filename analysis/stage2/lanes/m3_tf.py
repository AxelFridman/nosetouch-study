"""Lane m3 time-frequency inputs and the non-deep time-frequency baseline.

Per-trial transforms (functions of one trial alone, so they run before the cross-validation):

* STFT of every channel of the time-normalised approach (Hann window of ``STFT_NPERSEG`` samples, hop
  ``STFT_HOP``): (N, C, 2, F, T') with the real and imaginary parts;
* continuous wavelet transform with the complex Morlet ``CWT_WAVELET`` (PyWavelets) at ``CWT_SCALES``:
  (N, C, 2, S, T) with the real (the signed, real-Morlet-like) and imaginary parts.

Physics: under the left/right mirror a chiral channel only changes sign, x -> -x. Then STFT(-x) = -STFT(x) and
CWT(-x) = -CWT(x), so the MAGNITUDE |.| of a trial and of its mirror image are identical, while the real and
imaginary parts change sign. A magnitude-only input therefore cannot tell a trial from its mirror image;
with mirror augmentation it sees every input twice with opposite labels and is at chance by construction
(``mode='mag'``), and without augmentation it can only use what is NOT mirror-symmetric between the hands
(e.g. the left hand's smaller early yaw of mostly right-handed persons). ``mode='signed'`` keeps the sign.

The baseline (:class:`TFLogisticFamily`): the planes are average-pooled into frequency bands x time blocks
(per trial), standardised on the training rows and their mirrors, and fed to an L2 logistic regression whose C
is chosen on ``fold.inner_splits()`` (log loss)."""
from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np

from stage2.lanes import m3_common as M

STFT_NPERSEG = 16
STFT_HOP = 4
STFT_BANDS = ((0, 1), (1, 2), (2, 4), (4, None))         # frequency bins (cycles per window of 16 samples)
CWT_WAVELET = 'cmor1.5-1.0'
CWT_SCALES = tuple(float(s) for s in 2.0 ** np.linspace(1, 5, 10))   # 2 .. 32 samples
CWT_SCALE_GROUPS = 5
TIME_BLOCKS = {'stft': 4, 'cwt': 8}


def stft_planes(X: np.ndarray) -> np.ndarray:
    """(N, C, T) -> (N, C, 2, F, T') float32: real and imaginary STFT planes of every channel."""
    from scipy.signal import stft
    _, _, Z = stft(np.asarray(X, dtype=np.float64), nperseg=STFT_NPERSEG, noverlap=STFT_NPERSEG - STFT_HOP,
                   window='hann', axis=-1)
    return np.stack([Z.real, Z.imag], axis=2).astype(np.float32)


def cwt_planes(X: np.ndarray) -> np.ndarray:
    """(N, C, T) -> (N, C, 2, S, T) float32: real and imaginary complex-Morlet coefficients."""
    import pywt
    W, _ = pywt.cwt(np.asarray(X, dtype=np.float64), np.asarray(CWT_SCALES), CWT_WAVELET, axis=-1)   # (S, N, C, T)
    W = np.moveaxis(W, 0, 2)                                                                     # (N, C, S, T)
    return np.stack([W.real, W.imag], axis=2).astype(np.float32)


def x_stft(d: Any) -> dict[str, np.ndarray]:
    """STFT planes of the approach (all 17 channels)."""
    return {'x': stft_planes(d.approach)}


def x_cwt(d: Any) -> dict[str, np.ndarray]:
    """CWT planes of the approach (all 17 channels)."""
    return {'x': cwt_planes(d.approach)}


def _blocks(n: int, k: int) -> list[tuple[int, int]]:
    e = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(e[:-1], e[1:]) if b > a]


def pooled(P: np.ndarray, tf: str, mode: str) -> np.ndarray:
    """Per-trial pooled features of planes P (N, C, 2, F, T): means over frequency bands x time blocks of the real
    and imaginary parts ('signed') or of the magnitude ('mag'). Columns are channel-major."""
    N, Cc = P.shape[:2]
    F, T = P.shape[3], P.shape[4]
    if tf == 'stft':
        bands = [(a, F if b is None else min(b, F)) for a, b in STFT_BANDS if a < F]
    else:
        bands = _blocks(F, CWT_SCALE_GROUPS)
    times = _blocks(T, TIME_BLOCKS[tf])
    Q = np.sqrt(P[:, :, 0] ** 2 + P[:, :, 1] ** 2)[:, :, None] if mode == 'mag' else P     # (N, C, 1|2, F, T)
    cols = []
    for (a, b) in bands:
        for (s, e) in times:
            cols.append(Q[:, :, :, a:b, s:e].mean(axis=(3, 4)))                       # (N, C, 1|2)
    out = np.stack(cols, axis=-1)                                                        # (N, C, 1|2, B)
    return out.reshape(N, -1).astype(np.float64)


def pooled_signs(channels: Any, n_per_channel: int, mode: str) -> np.ndarray:
    s = M.mirror_signs(channels) if mode == 'signed' else np.ones(len(M.channel_names(channels)), np.float32)
    return np.repeat(s, n_per_channel).astype(np.float64)


DEFAULTS: dict[str, Any] = dict(tf='stft', channels='grav', mode='signed', mirror_aug=True,
                                Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0))


class TFLogisticFamily:
    def __init__(self, **params: Any):
        bad = set(params) - set(DEFAULTS)
        if bad:
            raise ValueError(f'unknown TF parameters {sorted(bad)}')
        self.p = {**DEFAULTS, **params}
        self.p['Cs'] = tuple(float(c) for c in self.p['Cs'])
        self.n_inner = None

    def config(self) -> dict[str, Any]:
        return dict(self.p)

    def code_hash(self) -> str:
        return M.module_code_hash(M, sys.modules[__name__])

    def cache_name(self) -> str:
        return 'tflog_' + M.digest(self.config())[:12]

    def features(self, X: dict[str, np.ndarray]) -> np.ndarray:
        P = X['x'][:, M.channel_idx(self.p['channels'])]
        return pooled(P, self.p['tf'], self.p['mode'])

    def _signs(self, F: np.ndarray) -> np.ndarray:
        n = len(M.channel_names(self.p['channels']))
        return pooled_signs(self.p['channels'], F.shape[1] // n, self.p['mode'])

    def _prep(self, F: np.ndarray, y: np.ndarray, s: np.ndarray):
        if self.p['mirror_aug']:
            Fa, ya = np.concatenate([F, F * s]), np.concatenate([y, 1 - y])
        else:
            Fa, ya = F, y
        mu = np.concatenate([F, F * s]).mean(axis=0)
        mu[s < 0] = 0.0                                          # mirror-symmetric centring
        sd = np.concatenate([F, F * s]).std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        return (Fa - mu) / sd, ya, mu, sd

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
            splits: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        F = self.features(X)
        s = self._signs(F)
        Cs = self.p['Cs']
        oof = {c: np.full(y.size, np.nan) for c in Cs}
        for tr, va in splits:
            Fa, ya, mu, sd = self._prep(F[tr], y[tr], s)
            for c in Cs:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = LogisticRegression(C=c, max_iter=2000).fit(Fa, ya)
                oof[c][va] = m.decision_function((F[va] - mu) / sd)
        ll = {c: float(np.mean(np.logaddexp(0, -(2 * y - 1) * oof[c]))) for c in Cs}
        best = min(Cs, key=lambda c: (ll[c], c))
        Fa, ya, mu, sd = self._prep(F, y, s)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = LogisticRegression(C=best, max_iter=2000).fit(Fa, ya)
        return {'mu': mu, 'sd': sd, 'coef': m.coef_.ravel().astype(float), 'intercept': float(m.intercept_[0]),
                'oof_z': oof[best], 'info': {'C': best, 'inner_log_loss': {str(c): round(v, 5) for c, v in ll.items()},
                                              'n_features': int(F.shape[1])}}

    def logits(self, state: dict[str, Any], X: dict[str, np.ndarray]) -> np.ndarray:
        F = self.features(X)
        return ((F - state['mu']) / state['sd']) @ state['coef'] + state['intercept']


def factory(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """fit_predict of a time-frequency logistic configuration (finalists: ``factory(**params)``)."""
    return M.make_fit_predict(TFLogisticFamily(**params), calib=calib, use_cache=use_cache, cache_dir=cache_dir)
