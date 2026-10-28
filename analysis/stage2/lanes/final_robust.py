"""Lane final: robustness of every candidate to the test conditions the PI may choose (task 4b).

Wrappers around a candidate's ``fit_predict`` (all evaluated through ``cv.run_experiment``, so every result is a
leave-one-person-out result with the harness's checks):

* **grip transfer** (:func:`train_filter` 'upright'): training restricted to the trials whose grip, detected from
  the trial's own up vector before the onset (``m3_experiments.sensor_grip`` rule; never the instructed grip), is
  upright; every trial is predicted, and the derive step scores the rotated grips (the instructed orientation only
  stratifies the predictions);
* **platform transfer** ('other_platform'): only the persons of the other platform train (the held-out person's
  platform is a test-side column known in production); 'platform_control' trains on as many randomly chosen persons
  of either platform (seeded), to separate the platform from the smaller training set (lane p1's design);
* **label noise** ('no_blocks'): training without the candidate wrong-hand blocks (lane m1's data-driven list) and
  the swap-signature blocks (``reference.swap_blocks``); labels are never changed, evaluation keeps every trial;
* **screen-away grip** (:func:`flip_predict`): the model is trained as usual and predicts the SAME test trials as
  if the phone had been held with the screen away (``standardize.screen_flip_channels`` on the arrays, then the
  candidate's own feature code: :func:`flipped`);
* remedies of the screen-away failure: the rest-pose screen check (:func:`screen_away`: the up vector's screen-normal
  component over the rest before the onset, lane m4's model-free rule and threshold) followed by un-flipping
  (:func:`detect_and_correct`, a deterministic map of the saved predictions) or abstaining; and screen-flip
  augmentation of the training rows (:func:`flip_augment`).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
import pandas as pd

from stage2 import cv
from stage2.standardize import CHANNELS, screen_flip_channels

#: samples of the cached fixed window before the oracle onset (the window starts 0.5 s before it, at 60 Hz)
PRE_ONSET_SAMPLES = 30


# ───────────────────────────── per-trial sensor quantities (no label) ─────────────────────────────


def sensor_grip_meta(meta: pd.DataFrame) -> np.ndarray:
    """Lane m3's sensor-detected grip code (0 upright, 1 top left, 2 top right, 3 other) from the trial table's
    pre-onset up vector (``up_pre_x/y/z``), computed exactly like ``m3_experiments.sensor_grip``."""
    from motionlab.conventions import rotation_from_gravity
    from stage2.lanes.m3_experiments import GRIP_CODES
    U = meta[['up_pre_x', 'up_pre_y', 'up_pre_z']].to_numpy(float)
    out = np.full(len(U), 3, dtype=np.int8)
    for i, u in enumerate(U):
        if np.all(np.isfinite(u)):
            out[i] = GRIP_CODES.get(str(rotation_from_gravity(u, 1, 'toward')), 3)
    return out


def rest_up_z(fixed: np.ndarray) -> np.ndarray:
    """The rest pose's screen-normal up component of every trial: the median of ``up_z`` over the cached fixed
    window's samples before the onset (the phone held at arm's length). Odd under the screen flip, even under the
    left/right mirror."""
    k = CHANNELS.index('up_z')
    return np.median(np.asarray(fixed, dtype=np.float64)[:, k, :PRE_ONSET_SAMPLES], axis=1)


def screen_away(fixed: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """True where the rest-pose check says the screen faces away (lane m4's rule: screen-normal up component below
    ``m4_config.GRIP_GATE_UPZ``; a fixed rule, nothing fitted)."""
    if threshold is None:
        from stage2.lanes.m4_config import GRIP_GATE_UPZ
        threshold = float(GRIP_GATE_UPZ)
    return rest_up_z(fixed) < threshold


class Flipped:
    """A Stage2Data seen as if every phone had been held with the screen away: the arrays multiplied by the
    screen-flip signs (the phone turned 180 deg about its y axis; labels unchanged). Exposes what the candidates'
    ``build_x`` functions read (approach, fixed, fixed_mask, ret, ret_ok, trials, channels, y, groups)."""

    def __init__(self, d: Any):
        self._d = d
        self.approach = screen_flip_channels(np.asarray(d.approach))
        self.fixed = screen_flip_channels(np.asarray(d.fixed))
        self.fixed_mask = d.fixed_mask
        self.ret = screen_flip_channels(np.asarray(d.ret))
        self.ret_ok = d.ret_ok
        self.trials = d.trials
        self.channels = d.channels

    @property
    def y(self) -> np.ndarray:
        return self._d.y

    @property
    def groups(self) -> np.ndarray:
        return self._d.groups


def flipped(d: Any) -> Flipped:
    return Flipped(d)


# ───────────────────────────── fold surgery ─────────────────────────────


def restrict_training(fold: cv.Fold, keep: np.ndarray) -> cv.Fold:
    """The fold with only the training rows ``keep`` (boolean mask or indices)."""
    idx = np.flatnonzero(keep) if np.asarray(keep).dtype == bool else np.asarray(keep, dtype=int)
    return dataclasses.replace(fold, X_train=cv._take(fold.X_train, idx), y_train=fold.y_train[idx].copy(),
                               groups_train=fold.groups_train[idx].copy(),
                               meta_train=fold.meta_train.iloc[idx].reset_index(drop=True))


def _blocks_mask(meta: pd.DataFrame, blocks: list[tuple[str, int]]) -> np.ndarray:
    bi = pd.to_numeric(meta['block_index'], errors='coerce').fillna(-1).astype(int)
    return pd.MultiIndex.from_arrays([meta['session_label'].astype(str), bi]).isin(blocks)


def train_filter(fp: Callable[[cv.Fold], Any], how: str, *, blocks: list[tuple[str, int]] | None = None
                 ) -> Callable[[cv.Fold], Any]:
    """fit_predict that trains ``fp`` on a subset of the training rows (``how``: 'upright', 'other_platform',
    'platform_control', 'no_blocks'); the test rows are predicted unchanged."""
    from stage2.lanes.m3_experiments import GRIP_CODES

    def fit_predict(fold: cv.Fold) -> Any:
        m = fold.meta_train
        if how == 'upright':
            keep = sensor_grip_meta(m) == GRIP_CODES['upright']
        elif how in ('other_platform', 'platform_control'):
            test_platform = str(fold.meta_test['platform'].iloc[0])
            other = m['platform'].astype(str).to_numpy() != test_platform
            if how == 'other_platform':
                keep = other
            else:                                  # as many persons as the other platform has, drawn at random
                persons = np.unique(fold.groups_train.astype(str))
                n = len(set(fold.groups_train[other].astype(str).tolist()))
                pick = np.random.default_rng([fold.seed, 77]).choice(persons, size=n, replace=False)
                keep = np.isin(fold.groups_train.astype(str), pick)
        elif how == 'no_blocks':
            keep = ~_blocks_mask(m, blocks or [])
        else:
            raise ValueError(how)
        if len(set(fold.groups_train[keep].astype(str).tolist())) < 2:
            raise ValueError(f'{how}: fewer than 2 training persons left in fold {fold.fold}')
        out = fp(restrict_training(fold, keep))
        p, info = (out if isinstance(out, tuple) else (out, {}))
        return p, {**info, 'filter': how, 'n_train_kept': int(np.sum(keep)),
                   'n_train_persons_kept': int(len(set(fold.groups_train[keep].astype(str).tolist())))}

    return fit_predict


def flip_predict(fp: Callable[[cv.Fold], Any]) -> Callable[[cv.Fold], Any]:
    """fit_predict on X = {'x': input, 'xflip': the same input computed from screen-flipped arrays}: trains on the
    recorded training rows ('x') and predicts the test rows as if held with the screen away ('xflip')."""
    def fit_predict(fold: cv.Fold) -> Any:
        f = dataclasses.replace(fold, X_train=fold.X_train['x'], X_test=fold.X_test['xflip'])
        return fp(f)
    return fit_predict


def flip_augment(fp: Callable[[cv.Fold], Any], predict: str = 'x') -> Callable[[cv.Fold], Any]:
    """Remedy: train on every training row AND its screen-flipped copy (same label), predict the test rows as
    recorded (``predict='x'``) or screen-flipped ('xflip')."""
    def fit_predict(fold: cv.Fold) -> Any:
        n = fold.y_train.size
        Xtr = cv._concat(fold.X_train['x'], fold.X_train['xflip'])
        f = dataclasses.replace(fold, X_train=Xtr, y_train=np.concatenate([fold.y_train, fold.y_train]),
                                groups_train=np.concatenate([fold.groups_train, fold.groups_train]),
                                meta_train=pd.concat([fold.meta_train, fold.meta_train], ignore_index=True),
                                X_test=fold.X_test[predict])
        out = fp(f)
        p, info = (out if isinstance(out, tuple) else (out, {}))
        return p, {**info, 'augmented_rows': int(n)}
    return fit_predict


def detect_and_correct(p_as_held: np.ndarray, p_other: np.ndarray, away: np.ndarray, mode: str = 'unflip'
                       ) -> np.ndarray:
    """The answer of the rest-pose screen check for trials held one way: where the check says 'screen away' the
    model gets the un-flipped input, whose prediction is ``p_other`` (the prediction of the other grip of the same
    movement), or abstains (``mode='abstain'``: P = 0.5); elsewhere it keeps ``p_as_held``."""
    away = np.asarray(away, dtype=bool)
    alt = np.full_like(np.asarray(p_as_held, dtype=float), 0.5) if mode == 'abstain' else np.asarray(p_other, float)
    return np.where(away, alt, np.asarray(p_as_held, dtype=float))


__all__ = ['sensor_grip_meta', 'rest_up_z', 'screen_away', 'flipped', 'Flipped', 'restrict_training', 'train_filter',
           'flip_predict', 'flip_augment', 'detect_and_correct', 'PRE_ONSET_SAMPLES']


class VariantView(Flipped):
    """A Stage2Data whose time-normalised approach comes from one of lane p1's variant caches (the raw records
    re-standardised with one parameter changed: segment bounds jittered, the onset late, a causal low-pass); every
    other array is the cache's. Built through p1's own loader (``p1_variants.load``)."""

    def __init__(self, d: Any, variant: str):
        from stage2.lanes.p1_variants import load
        self._d = d
        self.approach = load(variant, d.trials['trial_id'].astype(str).tolist()).astype(np.float32)
        self.fixed = np.asarray(d.fixed)
        self.fixed_mask = d.fixed_mask
        self.ret = np.asarray(d.ret)
        self.ret_ok = d.ret_ok
        self.trials = d.trials
        self.channels = d.channels
