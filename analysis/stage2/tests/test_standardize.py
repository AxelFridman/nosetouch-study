"""The canonical representation: shapes, references, and exact covariance under the mirror and the screen flip."""
from __future__ import annotations

import numpy as np
import pytest

from stage2 import config as C
from stage2.standardize import (
    CHANNEL_INFO, CHANNELS, CHIRAL_CHANNELS, GRAVITY_CHANNELS, PHASES, T_FIXED, canonical, fixed_window,
    mirror_channels, phase_labels, screen_flip_channels, standardize_trial, stream_from_record, time_normalised,
)


def _bounds(rec):
    tim = rec['timing']
    return float(tim['onset_ms']), float(tim['end_ms'])


def test_channel_tables_are_consistent():
    assert len(CHANNELS) == 17 and set(CHANNELS) == set(CHANNEL_INFO)
    assert set(CHIRAL_CHANNELS) == {'yaw_rate', 'yaw_cum'}
    for c in GRAVITY_CHANNELS:  # heading-free channels do not change under the screen flip (a proper rotation)
        assert CHANNEL_INFO[c][3] == 1
    X = np.random.default_rng(0).normal(size=(3, len(CHANNELS), 5))
    assert np.allclose(mirror_channels(mirror_channels(X)), X)
    assert np.allclose(screen_flip_channels(screen_flip_channels(X)), X)


def test_trial_arrays_shapes_and_references(street_records):
    rec, nm = street_records[0]
    on, en = _bounds(rec)
    ta = standardize_trial(rec, nm, on, en)
    assert ta.approach.shape == (len(CHANNELS), C.N_NORM) and ta.approach.dtype == np.float32
    # QC for the flags and gates: the up vector before the onset (grip check) and the G1 gap window
    assert {'up_pre_x', 'up_pre_y', 'up_pre_z', 'g1_n_gaps', 'g1_gap_max_ms'} <= set(ta.qc)
    assert abs(np.linalg.norm([ta.qc['up_pre_x'], ta.qc['up_pre_y'], ta.qc['up_pre_z']]) - 1.0) < 0.05
    assert ta.fixed.shape == (len(CHANNELS), T_FIXED) and ta.fixed_mask.shape == (T_FIXED,)
    assert np.isfinite(ta.approach).all()
    k = CHANNELS.index('yaw_cum')
    assert ta.approach[k, 0] == pytest.approx(0.0, abs=1e-6)
    assert ta.approach[CHANNELS.index('tilt_from_start'), 0] == pytest.approx(0.0, abs=0.5)
    up = ta.approach[[CHANNELS.index(c) for c in ('up_x', 'up_y', 'up_z')]]
    assert np.allclose(np.linalg.norm(up, axis=0), 1.0, atol=1e-3)
    # the stream is the 60 Hz grid
    assert np.allclose(np.diff(ta.stream.t), 1000 / 60, atol=1e-6)
    # the fixed window is masked from the end cue on and phases are ordered
    cue = rec['timing']['end_detected_ms']
    grid = on - 500 + np.arange(T_FIXED) * 1000 / 60
    assert not ta.fixed_mask[grid >= cue].any()
    assert np.all(np.diff(ta.fixed_phase.astype(int)) >= 0)
    assert set(np.unique(ta.stream_phase)) <= set(PHASES)


def test_mirror_is_exact_on_the_cached_channels(street_records):
    """standardise(mirror(record)) == mirror_channels(standardise(record)): every channel is covariant."""
    for rec, nm in street_records[:6]:
        on, en = _bounds(rec)
        a = standardize_trial(rec, nm, on, en)
        m = standardize_trial(rec, nm, on, en, mirror=True)
        scale = np.maximum(1.0, np.abs(a.approach).max(axis=1, keepdims=True))
        assert np.allclose(mirror_channels(a.approach), m.approach, atol=1e-4 * scale.max())
        assert np.allclose(mirror_channels(a.fixed), m.fixed, atol=1e-3)


def test_screen_flip_keeps_gravity_frame_channels(street_records):
    for rec, nm in street_records[:4]:
        on, en = _bounds(rec)
        a = standardize_trial(rec, nm, on, en)
        f = standardize_trial(rec, nm, on, en, screen_flip=True)
        assert np.allclose(screen_flip_channels(a.approach), f.approach, atol=1e-3)
        g = [CHANNELS.index(c) for c in GRAVITY_CHANNELS]
        assert np.allclose(a.approach[g], f.approach[g], atol=1e-3)


def test_the_same_code_runs_on_any_bounds(street_records):
    """Detected bounds (shifted by 100 ms) go through exactly the same canonicalisation."""
    rec, nm = street_records[1]
    on, en = _bounds(rec)
    st = stream_from_record(rec, nm)
    c1 = canonical(st, on, en)
    c2 = canonical(st, on + 100, en + 100)
    x1 = time_normalised(c1, on, en)
    x2 = time_normalised(c2, on + 100, en + 100)
    assert x1.shape == x2.shape
    assert not np.allclose(x1, x2)
    X, mask, grid = fixed_window(c1, on, cut_ms=None)
    assert mask.all() or (~mask).sum() < 5


def test_phase_labels():
    t = np.arange(0, 10000, 100.0)
    ph = phase_labels(t, 2000, 3000, 3250, 4000, 5000)
    assert ph[t < 2000].max() == 0 and (ph[(t >= 2000) & (t <= 3000)] == 1).all()
    assert (ph[(t > 3000) & (t < 3250)] == 2).all() and (ph[(t >= 3250) & (t < 4000)] == 3).all()
    assert (ph[(t >= 4000) & (t < 5000)] == 4).all() and (ph[t >= 5000] == 5).all()


def test_cache_shapes_and_alignment(primary):
    n = primary.n
    assert primary.approach.shape == (n, len(CHANNELS), C.N_NORM)
    assert primary.fixed.shape == (n, len(CHANNELS), T_FIXED)
    assert primary.ret.shape == (n, len(CHANNELS), C.N_NORM) and primary.ret_ok.shape == (n,)
    assert tuple(primary.channels) == CHANNELS
    assert np.isfinite(primary.approach).all()
    assert np.isfinite(primary.ret[primary.ret_ok]).all()
    assert np.isnan(primary.ret[~primary.ret_ok]).all()
    assert len(primary.y) == n and set(np.unique(primary.y)) == {0, 1}
    # the first-half yaw of the table equals the cached array at the window midpoint (same pipeline)
    k = CHANNELS.index('yaw_cum')
    mid = 0.5 * (primary.approach[:, k, 31] + primary.approach[:, k, 32])
    assert np.corrcoef(mid, primary.trials['first_half_yaw_deg'])[0, 1] > 0.99
    # full streams line up with their rows
    s = primary.stream(0)
    assert s['t'].shape[0] == s['gyro'].shape[0] == s['phase'].shape[0]
    on = float(primary.trials['onset_ms'].iloc[0])
    assert s['phase'][s['t'] < on].max() == 0


def test_return_windows_never_overlap_the_approach(primary):
    tr = primary.trials
    ok = primary.ret_ok
    assert (tr.loc[ok, 'return_onset_ms'] > tr.loc[ok, 'end_ms']).all()
    # and the fixed window carries no sample after the end cue
    pre, post = C.FIXED_PRE_S * 1000, C.FIXED_POST_S * 1000
    for i in np.flatnonzero(ok)[:50]:
        grid = tr['onset_ms'].iloc[i] - pre + np.arange(T_FIXED) * 1000 / C.FS_HZ
        assert not primary.fixed_mask[i][grid >= tr['end_detected_ms'].iloc[i]].any()
        assert not (primary.fixed_phase[i][primary.fixed_mask[i]] >= 3).any()
