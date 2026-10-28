"""Lane m2: hand-crafted per-trial features with an exact left/right mirror map (feature registry).

Every feature is a function of ONE trial (its cached arrays and its own approach duration), so features can be
computed before ``run_experiment`` (README rule 3). Families (``FAMILIES``):

==========  ==========================================================================  =====================
family      what (per channel unless noted)                                              parity under mirror
==========  ==========================================================================  =====================
bins        mean of the channel in k equal time bins (k = 4 / 8 / 16)                   s_c (linear)
stats       mean, sd, min, max, argmin, argmax (relative position), skewness, kurtosis,  mean/skew s_c; min<->max
            energy (mean square), zero crossings of the mean-removed signal             and argmin<->argmax swap
                                                                                         for odd channels; rest even
fft         log band powers (5 bands) and the real / imaginary parts of the 3 lowest    powers even; re/im s_c
            non-zero Fourier coefficients (the phase-carrying, sign-covariant form)
wavelet     log energy of every db2 sub-band (PyWavelets ``wavedec``)                    even
xcorr       Pearson correlation of every channel pair within one frame group            s_a * s_b
cumint      cumulative integral (trapezoid, seconds) at 25/50/75/100 % of the window    s_c (linear)
            for the rate / acceleration channels (angles and the up vector excluded)
global      log approach duration (s)                                                    even
==========  ==========================================================================  =====================

s_c is the channel's sign under the left/right mirror (``standardize.MIRROR_SIGN``). The registry gives, for
every feature j, ``mirror_index[j]`` and ``mirror_sign[j]`` so that the features of the mirrored trial are
``F[:, mirror_index] * mirror_sign`` EXACTLY (IEEE negation commutes with sums, FFTs, wavelets and arg-extrema);
``tests/test_m2_features.py`` checks it against features computed on ``standardize.mirror_channels`` arrays.

Windows: ``approach`` = the time-normalised oracle approach (64 samples onset..end); ``fixed`` = the fixed 3-s window
around the onset (60 Hz, -0.5..+2.5 s); ``both`` = the two feature sets side by side. The fixed window is NOT an
onset-only window: the cache freezes every channel from the app's end cue on (carry-forward, ``fixed_mask`` False
there, so the return never enters), and its gravity reference (gravity_v1) uses the rest after the oracle end as well
as before the onset. These features do not use ``fixed_mask``, so the frozen tail enters the dispersion, energy,
spectral, wavelet and cumulative-integral features (the derive step reports how many trials have a frozen tail).

Channel sets: ``all`` (every channel of the cache), ``gravity`` (the grip-invariant gravity-frame channels) or
``device`` (the device-frame channels). ``amp_norm`` divides each channel of a trial by its own RMS before the
features (per-trial amplitude normalisation; the up vector is left alone).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from stage2.standardize import CHANNEL_INFO, CHANNELS, DEVICE_CHANNELS, GRAVITY_CHANNELS

FAMILIES: tuple[str, ...] = ('bins', 'stats', 'fft', 'wavelet', 'xcorr', 'cumint', 'global')
STATS: tuple[str, ...] = ('mean', 'sd', 'min', 'max', 'argmin', 'argmax', 'skew', 'kurt', 'energy', 'zc')
#: channels whose cumulative integral is a meaningful quantity (rates and accelerations; not angles, not the up vector)
CUMINT_EXCLUDE: frozenset[str] = frozenset({'yaw_cum', 'tilt_from_start', 'up_x', 'up_y', 'up_z'})
#: per-trial amplitude normalisation leaves the unit up vector alone (already dimensionless)
AMPNORM_EXCLUDE: frozenset[str] = frozenset({'up_x', 'up_y', 'up_z'})
CUM_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
N_COEF = 3                       # lowest non-zero Fourier coefficients kept as (re, im)
WAVELET = 'db2'
MAX_WAVELET_LEVEL = 5
EPS = 1e-12
FS_HZ = 60.0
#: fft bands: approach window in cycles per movement, fixed window in Hz (upper edge inclusive of the last bin)
FFT_BANDS = {'approach': ((1, 2), (2, 4), (4, 8), (8, 16), (16, 33)),
             'fixed': ((1 / 3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 15.01))}


@dataclass(frozen=True)
class FeatureSpec:
    """Which features to build (all per-trial). ``window``: approach | fixed | both; ``channels``: all | gravity |
    device; ``families``: subset of FAMILIES; ``n_bins``: k of the bins family; ``amp_norm``: per-trial RMS
    normalisation of every channel before the features."""
    window: str = 'approach'
    channels: str = 'all'
    families: tuple[str, ...] = FAMILIES
    n_bins: int = 8
    amp_norm: bool = False

    def __post_init__(self) -> None:
        if self.window not in ('approach', 'fixed', 'both'):
            raise ValueError(f'window {self.window!r}')
        if self.channels not in ('all', 'gravity', 'device'):
            raise ValueError(f'channels {self.channels!r}')
        bad = set(self.families) - set(FAMILIES)
        if bad or not self.families:
            raise ValueError(f'families {self.families!r}')

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['families'] = list(self.families)
        return d

    def with_(self, **kw: Any) -> 'FeatureSpec':
        if 'families' in kw:
            kw['families'] = tuple(f for f in FAMILIES if f in kw['families'])   # canonical order
        return replace(self, **kw)


DEFAULT_SPEC = FeatureSpec()


def channel_set(which: str) -> tuple[str, ...]:
    return {'all': CHANNELS, 'gravity': GRAVITY_CHANNELS, 'device': DEVICE_CHANNELS}[which]


def _sign(c: str) -> int:
    return int(CHANNEL_INFO[c][2])


# ───────────────────────────── registry ─────────────────────────────


@dataclass
class _Reg:
    rows: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[np.ndarray] = field(default_factory=list)

    def add(self, block: np.ndarray, names: Sequence[str], family: str, window: str, channel: Sequence[str],
            stat: Sequence[str], parity: Sequence[str], partner: Sequence[int | None], sign: Sequence[int]) -> None:
        """Append a (N, k) block; partner[i] = index WITHIN the block of the mirror partner (None = itself)."""
        base = sum(b.shape[1] for b in self.blocks)
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] != len(names):
            raise ValueError(f'{family}: {block.shape[1]} columns for {len(names)} names')
        for i, nm in enumerate(names):
            self.rows.append({'name': nm, 'family': family, 'window': window, 'channel': channel[i], 'stat': stat[i],
                              'parity': parity[i], 'mirror_index': base + (i if partner[i] is None else partner[i]),
                              'mirror_sign': int(sign[i])})
        self.blocks.append(block)


def _win_tag(window: str) -> str:
    return {'approach': 'app', 'fixed': 'fix'}[window]


def _bins(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str, k: int, start: int = 0) -> None:
    T = X.shape[-1] - start
    edges = np.linspace(start, X.shape[-1], k + 1).round().astype(int)
    vals, names, ch, st, par, sg = [], [], [], [], [], []
    for ci, c in enumerate(chans):
        for b in range(k):
            vals.append(X[:, ci, edges[b]:edges[b + 1]].mean(-1))
            names.append(f'{_win_tag(window)}:bins:{c}:b{b + 1}of{k}')
            ch.append(c)
            st.append(f'b{b + 1}of{k}')
            par.append('odd' if _sign(c) < 0 else 'even')
            sg.append(_sign(c))
    if T <= 0:
        raise ValueError('empty window')
    R.add(np.column_stack(vals), names, 'bins', window, ch, st, par, [None] * len(names), sg)


def _stats(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str) -> None:
    T = X.shape[-1]
    mean = X.mean(-1)
    Xc = X - mean[..., None]
    m2 = (Xc ** 2).mean(-1)
    m3 = (Xc ** 3).mean(-1)
    m4 = (Xc ** 4).mean(-1)
    out = {
        'mean': mean, 'sd': np.sqrt(m2), 'min': X.min(-1), 'max': X.max(-1),
        'argmin': X.argmin(-1) / (T - 1), 'argmax': X.argmax(-1) / (T - 1),
        'skew': m3 / (m2 ** 1.5 + EPS), 'kurt': m4 / (m2 ** 2 + EPS), 'energy': (X ** 2).mean(-1),
        'zc': (np.diff(np.signbit(Xc), axis=-1) != 0).sum(-1) / (T - 1),
    }
    pos = {s: i for i, s in enumerate(STATS)}
    for ci, c in enumerate(chans):
        s = _sign(c)
        cols, names, ch, st, par, partner, sg = [], [], [], [], [], [], []
        for nm in STATS:
            cols.append(out[nm][:, ci])
            names.append(f'{_win_tag(window)}:stats:{c}:{nm}')
            ch.append(c)
            st.append(nm)
            if s > 0:
                par.append('even'), partner.append(None), sg.append(1)
            elif nm in ('mean', 'skew'):
                par.append('odd'), partner.append(None), sg.append(-1)
            elif nm in ('min', 'max'):
                par.append('swap'), partner.append(pos['max' if nm == 'min' else 'min']), sg.append(-1)
            elif nm in ('argmin', 'argmax'):
                par.append('swap'), partner.append(pos['argmax' if nm == 'argmin' else 'argmin']), sg.append(1)
            else:
                par.append('even'), partner.append(None), sg.append(1)
        R.add(np.column_stack(cols), names, 'stats', window, ch, st, par, partner, sg)


def _fft(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str) -> None:
    T = X.shape[-1]
    F = np.fft.rfft(X, axis=-1) / T                          # (N, C, T//2+1); linear in X
    P = (F.real ** 2 + F.imag ** 2)
    if window == 'approach':
        k = np.arange(F.shape[-1])                           # cycles per movement
    else:
        k = np.fft.rfftfreq(T, d=1.0 / FS_HZ)                # Hz
    bands = FFT_BANDS[window]
    for ci, c in enumerate(chans):
        s = _sign(c)
        cols, names, st, par, sg = [], [], [], [], []
        for bi, (lo, hi) in enumerate(bands):
            sel = (k >= lo) & (k < hi)
            cols.append(np.log10(P[:, ci, sel].sum(-1) + EPS))
            names.append(f'{_win_tag(window)}:fft:{c}:logpow_band{bi + 1}')
            st.append(f'logpow_band{bi + 1}')
            par.append('even'), sg.append(1)
        for j in range(1, N_COEF + 1):
            for part, v in (('re', F[:, ci, j].real), ('im', F[:, ci, j].imag)):
                cols.append(v)
                names.append(f'{_win_tag(window)}:fft:{c}:{part}{j}')
                st.append(f'{part}{j}')
                par.append('odd' if s < 0 else 'even'), sg.append(s)
        R.add(np.column_stack(cols), names, 'fft', window, [c] * len(names), st, par, [None] * len(names), sg)


def _wavelet(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str) -> None:
    import pywt
    T = X.shape[-1]
    level = min(pywt.dwt_max_level(T, pywt.Wavelet(WAVELET).dec_len), MAX_WAVELET_LEVEL)
    coefs = pywt.wavedec(X, WAVELET, level=level, axis=-1)   # [cA_L, cD_L, ..., cD_1]; linear in X
    labels = [f'A{level}'] + [f'D{level - i}' for i in range(level)]
    E = np.stack([(cf ** 2).sum(-1) / T for cf in coefs], axis=-1)   # (N, C, level+1)
    for ci, c in enumerate(chans):
        names = [f'{_win_tag(window)}:wavelet:{c}:logE_{lb}' for lb in labels]
        R.add(np.log10(E[:, ci, :] + EPS), names, 'wavelet', window, [c] * len(names), [f'logE_{lb}' for lb in labels],
              ['even'] * len(names), [None] * len(names), [1] * len(names))


def _xcorr(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str) -> None:
    Xc = X - X.mean(-1, keepdims=True)
    sd = np.sqrt((Xc ** 2).mean(-1))
    groups = [g for g in (DEVICE_CHANNELS, GRAVITY_CHANNELS) if set(g) <= set(chans)]
    idx = {c: i for i, c in enumerate(chans)}
    cols, names, ch, st, par, sg = [], [], [], [], [], []
    for g in groups:
        for a_i, a in enumerate(g):
            for b in g[a_i + 1:]:
                ia, ib = idx[a], idx[b]
                cols.append((Xc[:, ia] * Xc[:, ib]).mean(-1) / (sd[:, ia] * sd[:, ib] + EPS))
                names.append(f'{_win_tag(window)}:xcorr:{a}|{b}')
                ch.append(f'{a}|{b}')
                st.append('corr')
                s = _sign(a) * _sign(b)
                par.append('odd' if s < 0 else 'even'), sg.append(s)
    if cols:
        R.add(np.column_stack(cols), names, 'xcorr', window, ch, st, par, [None] * len(names), sg)


def _cumint(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str, dt: np.ndarray, start: int = 0) -> None:
    keep = [c for c in chans if c not in CUMINT_EXCLUDE]
    if not keep:
        return
    ci = [chans.index(c) for c in keep]
    Y = X[:, ci, start:]
    inc = 0.5 * (Y[..., 1:] + Y[..., :-1]) * dt[:, None, None]
    cum = np.concatenate([np.zeros(Y.shape[:-1] + (1,)), np.cumsum(inc, axis=-1)], axis=-1)
    T = Y.shape[-1]
    at = [int(round(f * (T - 1))) for f in CUM_FRACTIONS]
    cols, names, ch, st, par, sg = [], [], [], [], [], []
    for j, c in enumerate(keep):
        for f, t in zip(CUM_FRACTIONS, at):
            cols.append(cum[:, j, t])
            names.append(f'{_win_tag(window)}:cumint:{c}:at{int(round(100 * f))}')
            ch.append(c)
            st.append(f'at{int(round(100 * f))}')
            par.append('odd' if _sign(c) < 0 else 'even'), sg.append(_sign(c))
    R.add(np.column_stack(cols), names, 'cumint', window, ch, st, par, [None] * len(names), sg)


def _amp_norm(X: np.ndarray, chans: Sequence[str]) -> np.ndarray:
    """Each channel of each trial divided by its own RMS (an even quantity, so the mirror map is unchanged)."""
    X = X.copy()
    for ci, c in enumerate(chans):
        if c in AMPNORM_EXCLUDE:
            continue
        rms = np.sqrt((X[:, ci] ** 2).mean(-1, keepdims=True))
        X[:, ci] = X[:, ci] / (rms + EPS)
    return X


def _window_features(R: _Reg, X: np.ndarray, chans: Sequence[str], window: str, spec: FeatureSpec,
                     duration_s: np.ndarray) -> None:
    fam = spec.families
    if spec.amp_norm:
        X = _amp_norm(X, chans)
    # the fixed window starts 0.5 s before the onset: cumulative integrals start at the onset sample
    onset_i = 0 if window == 'approach' else int(round(0.5 * FS_HZ))
    dt = (duration_s / (X.shape[-1] - 1)) if window == 'approach' else np.full(X.shape[0], 1.0 / FS_HZ)
    if 'bins' in fam:
        _bins(R, X, chans, window, spec.n_bins)
    if 'stats' in fam:
        _stats(R, X, chans, window)
    if 'fft' in fam:
        _fft(R, X, chans, window)
    if 'wavelet' in fam:
        _wavelet(R, X, chans, window)
    if 'xcorr' in fam:
        _xcorr(R, X, chans, window)
    if 'cumint' in fam:
        _cumint(R, X, chans, window, dt, start=onset_i)


def features_from_arrays(approach: np.ndarray | None, fixed: np.ndarray | None, duration_ms: np.ndarray,
                         spec: FeatureSpec = DEFAULT_SPEC, channels: Sequence[str] = CHANNELS
                         ) -> tuple[np.ndarray, pd.DataFrame]:
    """(F (N, P) float64, registry) from (N, C, T) arrays in ``channels`` order. Per-trial only: row i of F
    depends on row i of the inputs alone (no statistic across trials, no label)."""
    channels = tuple(channels)
    chans = channel_set(spec.channels)
    ci = [channels.index(c) for c in chans]
    dur_s = np.asarray(duration_ms, dtype=np.float64) / 1000.0
    R = _Reg()
    windows = ('approach', 'fixed') if spec.window == 'both' else (spec.window,)
    for w in windows:
        A = approach if w == 'approach' else fixed
        if A is None:
            raise ValueError(f'the {w} window is needed by {spec}')
        _window_features(R, np.asarray(A, dtype=np.float64)[:, ci, :], chans, w, spec, dur_s)
    if 'global' in spec.families:
        R.add(np.log(dur_s)[:, None], ['glob:global:log_duration_s'], 'global', 'trial', ['-'], ['log_duration_s'],
              ['even'], [None], [1])
    F = np.concatenate(R.blocks, axis=1)
    reg = pd.DataFrame(R.rows)
    reg.insert(0, 'j', np.arange(len(reg)))
    _check_registry(reg)
    _check_finite(F, reg)
    return F, reg


def _check_finite(F: np.ndarray, reg: pd.DataFrame) -> None:
    """Every feature of every trial is finite (a scaler or solver downstream would otherwise fail silently or
    late, e.g. after a new participant's recording with an unhandled gap)."""
    bad = ~np.isfinite(F)
    if bad.any():
        cols = np.flatnonzero(bad.any(axis=0))
        raise ValueError(f'{int(bad.any(axis=1).sum())} trial(s) have non-finite features in {cols.size} column(s), '
                         f'e.g. {", ".join(reg["name"].iloc[cols[:5]].tolist())}: check the input arrays')


def _check_registry(reg: pd.DataFrame) -> None:
    mi = reg['mirror_index'].to_numpy()
    if not np.array_equal(mi[mi], np.arange(len(reg))):
        raise AssertionError('mirror_index is not an involution')
    ms = reg['mirror_sign'].to_numpy()
    if not np.array_equal(ms[mi] * ms, np.ones(len(reg), dtype=ms.dtype)):
        raise AssertionError('mirror signs are inconsistent between partners')
    if reg['name'].duplicated().any():
        raise AssertionError('duplicated feature names')


def mirror_map(reg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(index, sign): the features of the mirrored trial are F[:, index] * sign."""
    return reg['mirror_index'].to_numpy().astype(int), reg['mirror_sign'].to_numpy().astype(np.float64)


def mirror_features(F: np.ndarray, index: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return F[:, index] * sign


def registry(spec: FeatureSpec = DEFAULT_SPEC, channels: Sequence[str] = CHANNELS, n_fixed: int = 180,
             n_approach: int = 64) -> pd.DataFrame:
    """The registry of ``spec`` without data (computed on a tiny synthetic input: names do not depend on data)."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, len(channels), n_approach))
    Fx = rng.normal(size=(3, len(channels), n_fixed))
    return features_from_arrays(A, Fx, np.array([1000.0, 1500.0, 2000.0]), spec, channels)[1]


def build_features(d: Any, spec: FeatureSpec = DEFAULT_SPEC) -> tuple[np.ndarray, pd.DataFrame]:
    """Features of a ``Stage2Data`` (primary or natural-use set): per trial, no label."""
    return features_from_arrays(d.approach, d.fixed, d.trials['duration_ms'].to_numpy(dtype=float), spec, d.channels)


def build_x(d: Any) -> np.ndarray:
    """The finalists' input: the default feature set (approach window, every family and channel, k = 8)."""
    return build_features(d, DEFAULT_SPEC)[0]


def build_x_spec(**spec_kw: Any):
    """A build_x for another spec (``build_x_spec(window='both')``)."""
    spec = DEFAULT_SPEC.with_(**spec_kw)

    def _build(d: Any) -> np.ndarray:
        return build_features(d, spec)[0]
    _build.__name__ = 'build_x_' + '_'.join(f'{k}_{v}' for k, v in sorted(spec_kw.items()))
    return _build
