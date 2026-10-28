"""Lane m3: the heading-following body frame as a deep-model input (grip-invariant device information).

Why. Lane m3's most accurate deep configuration (``m3_cnn_all``) adds the device-frame channels (acceleration,
angular velocity and up vector in phone coordinates) to the gravity frame. Those channels depend on the grip:
trained on upright grips only, the same network falls to chance on the rotated grips (``m3_cnn_all_upright``).
What they add is body-relative: which way the phone moves sideways and how it rolls. This module expresses that
information in the heading-following body frame, with the same definition as lane m1
(``m1_features.body_channels``; re-implemented here so that one lane never depends on another lane's code):

    h = (e_z - (e_z.u) u) / |...|    the horizontal direction out of the screen (towards the face)
    s = u x h                        the horizontal direction to the person's right
    roll_rate  = w.h   (odd)         rotation about the facing axis
    pitch_rate = w.s   (even)        rotation about the lateral axis
    acc_lat    = a.s   (odd)         lateral linear acceleration
    acc_fwd    = a.h   (even)        linear acceleration towards the face

u is the cache's gyro-propagated unit up vector, w the angular velocity and a the linear acceleration, all in the
device frame. A rotation of the phone about its screen normal (the three grips of the protocol and every grip in
between) leaves every projection unchanged; under the left/right mirror (a -> S a, w -> -S w, S = diag(-1, 1, 1))
roll_rate and acc_lat change sign and pitch_rate and acc_fwd do not (``test_m3.py`` checks both). h is undefined
only when the screen is exactly horizontal, which no approach sample comes near (the projections are then set to 0).

Everything here is a per-trial transform (no statistic over trials), so ``x_approach_body`` (time-normalised
approach) and ``x_fixed_body`` (the fixed 3-s window in absolute time, with its mask) may run before the
cross-validation. :class:`BodyDeepFamily` is the m3 deep family on the extended channel table; it is a separate
class (with its own fit-cache name and code hash) so the cached fits of every other m3 experiment stay valid.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from stage2.lanes import m3_common as M
from stage2.lanes import m3_deep, m3_nets
from stage2.standardize import CHANNEL_INFO, CHANNELS, CHIRAL_CHANNELS, GRAVITY_CHANNELS

#: body-frame channels and their sign under the left/right mirror
BODY_CHANNELS: tuple[str, ...] = ('roll_rate', 'pitch_rate', 'acc_lat', 'acc_fwd')
BODY_SIGN: dict[str, int] = {'roll_rate': -1, 'pitch_rate': +1, 'acc_lat': -1, 'acc_fwd': +1}
BODY_INFO: dict[str, str] = {'roll_rate': 'w.h: roll rate about the facing axis (deg/s, odd)',
                             'pitch_rate': 'w.s: pitch rate about the lateral axis (deg/s, even)',
                             'acc_lat': 'a.s: lateral linear acceleration, + = to the right (m/s^2, odd)',
                             'acc_fwd': 'a.h: linear acceleration towards the face (m/s^2, even)'}
#: the cache channels followed by the body channels: the layout of ``x_approach_body(d)['x']``
EXT_CHANNELS: tuple[str, ...] = CHANNELS + BODY_CHANNELS
EXT_SIGN: dict[str, int] = {**{c: int(CHANNEL_INFO[c][2]) for c in CHANNELS}, **BODY_SIGN}
#: the grip-invariant input of this module: the gravity frame (8) plus the body frame (4)
BODY_SET: tuple[str, ...] = GRAVITY_CHANNELS + BODY_CHANNELS
#: odd (hand-carrying) channels that do not depend on the grip: the chiral evidence of a body-frame model
INVARIANT_ODD: tuple[str, ...] = tuple(CHIRAL_CHANNELS) + ('roll_rate', 'acc_lat')
H_MIN = 1e-6


def body_channels(X: np.ndarray, channels: tuple[str, ...] = CHANNELS) -> np.ndarray:
    """(N, C, T) canonical arrays -> (N, 4, T) float32 in ``BODY_CHANNELS`` order (per sample, per trial)."""
    ix = {c: i for i, c in enumerate(channels)}
    X = np.asarray(X, dtype=np.float64)
    u = X[:, [ix['up_x'], ix['up_y'], ix['up_z']], :]
    w = X[:, [ix['gyro_x'], ix['gyro_y'], ix['gyro_z']], :]
    a = X[:, [ix['acc_x'], ix['acc_y'], ix['acc_z']], :]
    h = -u[:, 2:3, :] * u
    h[:, 2, :] += 1.0                                         # e_z - (e_z.u) u
    nrm = np.linalg.norm(h, axis=1, keepdims=True)
    ok = nrm > H_MIN
    h = np.where(ok, h / np.where(ok, nrm, 1.0), 0.0)         # undefined only with the screen exactly horizontal
    s = np.cross(u, h, axis=1)
    out = np.stack([np.sum(w * h, axis=1), np.sum(w * s, axis=1), np.sum(a * s, axis=1), np.sum(a * h, axis=1)],
                   axis=1)
    return out.astype(np.float32)


def x_approach_body(d: Any) -> dict[str, np.ndarray]:
    """The time-normalised approach with the body channels appended: (N, 21, 64) in ``EXT_CHANNELS`` order."""
    A = np.asarray(d.approach, dtype=np.float32)
    return {'x': np.ascontiguousarray(np.concatenate([A, body_channels(A)], axis=1), dtype=np.float32)}


def x_fixed_body(d: Any) -> dict[str, np.ndarray]:
    """The fixed 3-s window (absolute time) with the body channels appended, (N, 21, 180) in ``EXT_CHANNELS``
    order, and its mask (False from the end cue on and outside the recording; the cache freezes the values there)."""
    F = np.asarray(d.fixed, dtype=np.float32)
    return {'x': np.ascontiguousarray(np.concatenate([F, body_channels(F)], axis=1), dtype=np.float32),
            'mask': np.ascontiguousarray(d.fixed_mask, dtype=np.float32)}


def x_approach_body_grip(d: Any) -> dict[str, np.ndarray]:
    """``x_approach_body`` plus the sensor-detected grip code (chooses the TRAINING rows of the grip-transfer check
    only; never a model input)."""
    from stage2.lanes.m3_experiments import sensor_grip
    return {**x_approach_body(d), 'grip': sensor_grip(d)}


def channel_names(channels: Any) -> tuple[str, ...]:
    if isinstance(channels, str):
        return BODY_SET if channels == 'body' else M.CHANNEL_SETS[channels]
    return tuple(channels)


class BodyDeepFamily(m3_deep.DeepFamily):
    """``m3_deep.DeepFamily`` on the extended channel table (cache channels + body channels). ``channels='body'``
    is ``BODY_SET``; a tuple of names from ``EXT_CHANNELS`` is also accepted. Time-domain inputs only."""

    def __init__(self, **params: Any):
        super().__init__(**params)
        if self.p['tf'] is not None:
            raise ValueError('BodyDeepFamily takes time-domain inputs only')
        bad = [c for c in channel_names(self.p['channels']) if c not in EXT_SIGN]
        if bad:
            raise ValueError(f'unknown channels {bad}')

    def channel_names(self) -> tuple[str, ...]:
        return channel_names(self.p['channels'])

    def code_hash(self) -> str:
        return M.module_code_hash(M, m3_nets, m3_deep, sys.modules[__name__])

    def cache_name(self) -> str:
        return 'body_' + M.digest(self.config())[:12]

    def _signs(self) -> np.ndarray:
        return np.array([EXT_SIGN[c] for c in self.channel_names()], dtype=np.float32)

    def _select(self, X: dict[str, np.ndarray]) -> np.ndarray:
        x = np.asarray(X['x'])
        if x.shape[1] != len(EXT_CHANNELS):
            raise ValueError(f'BodyDeepFamily needs x_approach_body inputs ({len(EXT_CHANNELS)} channels), got {x.shape[1]}')
        idx = [EXT_CHANNELS.index(c) for c in self.channel_names()]
        return np.ascontiguousarray(x[:, idx], dtype=np.float32)


def factory(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """fit_predict of a body-frame deep configuration (finalists: ``factory(**params)``)."""
    return M.make_fit_predict(BodyDeepFamily(**params), calib=calib, use_cache=use_cache, cache_dir=cache_dir)
