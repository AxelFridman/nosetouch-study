"""Lane final: lane m2's feature families on channel tables that include the heading-following body frame (task 2b).

Lane m2's feature code (``m2_features``) knows the 17 cache channels only: it reads each channel's mirror sign from
``standardize.CHANNEL_INFO`` and pairs cross-correlations within the device and gravity frames. This module computes
the SAME families (bins, stats, fft, wavelet, xcorr, cumint, global; approach window, k = 8) on

* ``body12``: the 1D CNN's input of lane m3 (8 gravity-frame + 4 body-frame channels: ``m3_body.BODY_SET``), and
* ``all21``: the 17 cache channels plus the 4 body-frame channels,

by calling lane m2's family functions with the body channels' signs added to the table it reads (a temporary
attribute patch of ``m2_features.CHANNEL_INFO`` inside :func:`_ext_channel_info`; lane m2's files are never edited)
and its own cross-correlation groups (one per frame). With the 17 cache channels and m2's two groups the result
equals ``m2_finalists.build_x`` exactly (``test_final.py``). Every feature is a function of one trial (no label).
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stage2.lanes import m2_features as M2F
from stage2.lanes.m3_body import BODY_CHANNELS, BODY_SET, EXT_CHANNELS, body_channels
from stage2.standardize import CHANNEL_INFO, CHANNELS, DEVICE_CHANNELS, GRAVITY_CHANNELS

#: CHANNEL_INFO rows of the body-frame channels: (frame, unit, mirror sign, screen-flip sign, meaning). Under the
#: screen flip (the phone turned 180 deg about its y axis) the facing axis h and the lateral axis s both change sign,
#: so every body-frame projection changes sign (``test_final.py`` checks it on the arrays).
BODY_INFO: dict[str, tuple[str, str, int, int, str]] = {
    'roll_rate': ('body', 'deg/s', -1, -1, 'w.h: roll rate about the facing axis'),
    'pitch_rate': ('body', 'deg/s', +1, -1, 'w.s: pitch rate about the lateral axis'),
    'acc_lat': ('body', 'm/s^2', -1, -1, 'a.s: lateral linear acceleration (+ = to the right)'),
    'acc_fwd': ('body', 'm/s^2', +1, -1, 'a.h: linear acceleration towards the face'),
}
EXT_INFO: dict[str, tuple[str, str, int, int, str]] = {**CHANNEL_INFO, **BODY_INFO}
#: channel tables and their cross-correlation groups (pairs only within a group, like lane m2)
CHANNEL_TABLES: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = {
    'm2': (tuple(CHANNELS), (tuple(DEVICE_CHANNELS), tuple(GRAVITY_CHANNELS))),
    'body12': (tuple(BODY_SET), (tuple(GRAVITY_CHANNELS), tuple(BODY_CHANNELS))),
    'all21': (tuple(CHANNELS) + tuple(BODY_CHANNELS), (tuple(DEVICE_CHANNELS), tuple(GRAVITY_CHANNELS),
                                                         tuple(BODY_CHANNELS))),
}
EXT_SCREEN_FLIP = np.array([EXT_INFO[c][3] for c in EXT_CHANNELS], dtype=np.float32)


@contextlib.contextmanager
def _ext_channel_info() -> Iterator[None]:
    """Lane m2's family functions read the mirror sign of a channel from ``m2_features.CHANNEL_INFO``: add the body
    channels for the duration of one call (restored afterwards, also on an error)."""
    old = M2F.CHANNEL_INFO
    M2F.CHANNEL_INFO = EXT_INFO
    try:
        yield
    finally:
        M2F.CHANNEL_INFO = old


def _xcorr(R: Any, X: np.ndarray, chans: Sequence[str], groups: Sequence[Sequence[str]]) -> None:
    """``m2_features._xcorr`` with explicit groups (the same formula, names and mirror signs)."""
    Xc = X - X.mean(-1, keepdims=True)
    sd = np.sqrt((Xc ** 2).mean(-1))
    idx = {c: i for i, c in enumerate(chans)}
    cols, names, ch, st, par, sg = [], [], [], [], [], []
    for g in groups:
        if not set(g) <= set(chans):
            continue
        for a_i, a in enumerate(g):
            for b in g[a_i + 1:]:
                ia, ib = idx[a], idx[b]
                cols.append((Xc[:, ia] * Xc[:, ib]).mean(-1) / (sd[:, ia] * sd[:, ib] + M2F.EPS))
                names.append(f'app:xcorr:{a}|{b}')
                ch.append(f'{a}|{b}')
                st.append('corr')
                s = EXT_INFO[a][2] * EXT_INFO[b][2]
                par.append('odd' if s < 0 else 'even'), sg.append(s)
    if cols:
        R.add(np.column_stack(cols), names, 'xcorr', 'approach', ch, st, par, [None] * len(names), sg)


def features_ext(A_ext: np.ndarray, duration_ms: np.ndarray, table: str) -> tuple[np.ndarray, pd.DataFrame]:
    """(F, registry) of lane m2's default families on the approach window of ``A_ext`` (N, 21, 64) in
    ``m3_body.EXT_CHANNELS`` order, for the channel table ``table`` (see :data:`CHANNEL_TABLES`)."""
    chans, groups = CHANNEL_TABLES[table]
    ci = [EXT_CHANNELS.index(c) for c in chans]
    X = np.asarray(A_ext, dtype=np.float64)[:, ci, :]
    dur_s = np.asarray(duration_ms, dtype=np.float64) / 1000.0
    dt = dur_s / (X.shape[-1] - 1)
    spec = M2F.DEFAULT_SPEC
    R = M2F._Reg()
    with _ext_channel_info():
        fam = spec.families
        if 'bins' in fam:
            M2F._bins(R, X, chans, 'approach', spec.n_bins)
        if 'stats' in fam:
            M2F._stats(R, X, chans, 'approach')
        if 'fft' in fam:
            M2F._fft(R, X, chans, 'approach')
        if 'wavelet' in fam:
            M2F._wavelet(R, X, chans, 'approach')
        if 'xcorr' in fam:
            _xcorr(R, X, chans, groups)
        if 'cumint' in fam:
            M2F._cumint(R, X, chans, 'approach', dt, start=0)
    if 'global' in spec.families:
        R.add(np.log(dur_s)[:, None], ['glob:global:log_duration_s'], 'global', 'trial', ['-'], ['log_duration_s'],
              ['even'], [None], [1])
    F = np.concatenate(R.blocks, axis=1)
    reg = pd.DataFrame(R.rows)
    reg.insert(0, 'j', np.arange(len(reg)))
    M2F._check_registry(reg)
    M2F._check_finite(F, reg)
    return F, reg


def ext_arrays(approach: np.ndarray) -> np.ndarray:
    """(N, 17, 64) cache approach -> (N, 21, 64) with the body-frame channels appended (``m3_body`` definition)."""
    A = np.asarray(approach, dtype=np.float32)
    return np.concatenate([A, body_channels(A)], axis=1)


def registry(table: str) -> pd.DataFrame:
    """The feature registry of ``table`` without data (names and mirror map do not depend on the data)."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, len(EXT_CHANNELS), 64))
    return features_ext(A, np.array([1000.0, 1500.0, 2000.0]), table)[1]


def mirror_map(table: str) -> tuple[np.ndarray, np.ndarray]:
    return M2F.mirror_map(registry(table))


def build_m2_body12(d: Any) -> np.ndarray:
    """Lane m2's features on the 12 channels of the 1D CNN (gravity + body frame)."""
    return features_ext(ext_arrays(d.approach), d.trials['duration_ms'].to_numpy(float), 'body12')[0]


def build_m2_all21(d: Any) -> np.ndarray:
    """Lane m2's features on the 17 cache channels plus the 4 body-frame channels."""
    return features_ext(ext_arrays(d.approach), d.trials['duration_ms'].to_numpy(float), 'all21')[0]


def build_m2_m2(d: Any) -> np.ndarray:
    """Lane m2's own default features recomputed here (equal to ``m2_finalists.build_x``; a test pins it)."""
    return features_ext(ext_arrays(d.approach), d.trials['duration_ms'].to_numpy(float), 'm2')[0]
