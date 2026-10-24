"""motionlab - analysis starter kit for the NoseTouch study (left vs right hand from phone motion).

Modules (import what you need):
  schema       validation against <repo>/schema/records.schema.json
  io           load_dataset(): backend folder trees, export bundles, any mix; dedupe; tidy tables; norm_v2
  conventions  frames, units, rotation-rate mapping (axis_check or closure), gravity-sign resolution (norm_v2),
               screen flip, orientation tables
  preprocess   resampling, zero-phase filtering, gyro-propagated gravity (gravity_v1), Chrome rounding,
               W3C Euler / matrix / quaternion conversions, gyro integration
  segment      online_v1 (exact port of the app's detector, rerun_online_v1) and offline_v1 (non-causal)
  features     per-trial features (basic / device / invariant / chiral) and sequence tensors
  augment      left/right mirroring, grip rotation, time warp, magnitude scaling, quantisation, noise
  synth        independent synthetic record generator (tests, demos)
  models       baselines (YawSignRule, logistic regression, gradient boosting) and the model registry
  evaluate     subject-wise CV schemes, metrics, subgroup errors with participant-bootstrap CIs, mirror consistency
  qa           dataset quality summary
  plots, report  figures (matplotlib Agg), Markdown and formula-safe CSV helpers for the scripts
"""
from __future__ import annotations

__version__ = '0.3.0'

from .conventions import NORM_VERSION, Normalization  # noqa: E402,F401
from .io import Dataset, load_dataset  # noqa: E402,F401

__all__ = ['Dataset', 'load_dataset', 'Normalization', 'NORM_VERSION', '__version__']
