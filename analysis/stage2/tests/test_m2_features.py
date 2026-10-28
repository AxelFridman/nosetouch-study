"""Lane m2 features: exact mirror map, per-trial computation, registry consistency, shapes."""
from __future__ import annotations

import numpy as np
import pytest

from stage2.lanes import m2_features as F
from stage2.standardize import CHANNELS, mirror_channels

SPECS = [F.DEFAULT_SPEC, F.DEFAULT_SPEC.with_(window='both'), F.DEFAULT_SPEC.with_(window='fixed', amp_norm=True),
         F.DEFAULT_SPEC.with_(channels='gravity', n_bins=16), F.DEFAULT_SPEC.with_(channels='device', n_bins=4),
         F.DEFAULT_SPEC.with_(families=['stats'])]


def _synthetic(n: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, len(CHANNELS), 64)).cumsum(-1).astype(np.float32)
    X = rng.normal(size=(n, len(CHANNELS), 180)).cumsum(-1).astype(np.float32)
    X[:, :, 150:] = X[:, :, 149:150]            # a frozen tail, as the cache stores after the end cue
    dur = rng.uniform(400, 5000, n)
    return A, X, dur


@pytest.mark.parametrize('spec', SPECS, ids=lambda s: f'{s.window}-{s.channels}-{len(s.families)}-{s.n_bins}-{s.amp_norm}')
def test_mirror_map_is_exact_on_synthetic_arrays(spec):
    A, X, dur = _synthetic()
    F0, reg = F.features_from_arrays(A, X, dur, spec)
    F1, reg1 = F.features_from_arrays(mirror_channels(A), mirror_channels(X), dur, spec)
    assert list(reg['name']) == list(reg1['name'])
    idx, sg = F.mirror_map(reg)
    np.testing.assert_array_equal(F.mirror_features(F0, idx, sg), F1)   # exact, not approximately
    assert np.isfinite(F0).all()


def test_mirror_map_is_exact_on_the_cache(primary):
    sub = primary.subset(np.arange(min(80, primary.n)))
    for spec in (F.DEFAULT_SPEC, F.DEFAULT_SPEC.with_(window='both')):
        F0, reg = F.build_features(sub, spec)
        F1, _ = F.features_from_arrays(mirror_channels(sub.approach), mirror_channels(sub.fixed),
                                       sub.trials['duration_ms'].to_numpy(float), spec, sub.channels)
        idx, sg = F.mirror_map(reg)
        np.testing.assert_array_equal(F.mirror_features(F0, idx, sg), F1)


def test_features_are_per_trial():
    """Row i of the feature matrix depends on trial i only (no statistic across trials)."""
    A, X, dur = _synthetic(20, seed=1)
    full, _ = F.features_from_arrays(A, X, dur, F.DEFAULT_SPEC.with_(window='both'))
    pick = np.array([7, 3, 15])
    part, _ = F.features_from_arrays(A[pick], X[pick], dur[pick], F.DEFAULT_SPEC.with_(window='both'))
    np.testing.assert_array_equal(full[pick], part)


def test_registry_shapes_and_parity():
    reg = F.registry(F.DEFAULT_SPEC)
    assert reg['name'].is_unique
    assert set(reg['family']) == set(F.FAMILIES)
    idx, sg = F.mirror_map(reg)
    assert np.array_equal(idx[idx], np.arange(len(reg)))           # an involution
    odd = reg[reg['parity'] == 'odd']
    assert (odd['mirror_sign'] == -1).all() and (odd['mirror_index'] == odd['j']).all()
    even = reg[reg['parity'] == 'even']
    assert (even['mirror_sign'] == 1).all()
    swap = reg[reg['parity'] == 'swap']
    assert (swap['mirror_index'] != swap['j']).all()
    # chiral gravity channels are odd in the linear families
    assert set(reg[(reg['family'] == 'bins') & (reg['parity'] == 'odd')]['channel']) >= {'yaw_rate', 'yaw_cum'}
    # the default set has a known structure: 17 channels x 8 bins
    assert int((reg['family'] == 'bins').sum()) == len(CHANNELS) * F.DEFAULT_SPEC.n_bins


def test_build_x_matches_build_features(primary):
    sub = primary.subset(np.arange(10))
    from stage2.lanes import m2_finalists
    np.testing.assert_array_equal(m2_finalists.build_x(sub), F.build_features(sub)[0])
