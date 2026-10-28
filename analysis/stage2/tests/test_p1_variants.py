"""Fast tests of lane p1's variant builder (p1_variants): synthetic signals only, no raw data, a few seconds.

The exact reproduction of the core cache by the default variant is checked on the real data when the variants are
built (``p1_variants.fidelity``, recorded in tables/p1/p1_variants.json and asserted in test_p1.py)."""
from __future__ import annotations

import numpy as np
import pytest

from stage2 import config as C
from stage2.lanes import p1_variants as V


def test_default_variant_is_the_core_configuration():
    s, w = V.VARIANTS['base'].stream, V.VARIANTS['base'].window
    assert (s.lowpass_hz, s.order, s.causal, s.max_gap_ms, s.interp, s.decimate) == (
        C.LOWPASS_HZ, C.LOWPASS_ORDER, False, C.MAX_GAP_MS, 'linear', 1)
    assert s.quantize_ios == C.QUANTIZE_IOS and s.t_jitter_ms == 0.0
    assert (w.gravity, w.window, w.bound_jitter_ms, w.onset_shift_ms) == ('gyro', 'approach', 0.0, 0.0)
    for name, v in V.VARIANTS.items():        # every other variant changes exactly what its name says
        assert v.name == name
        if name != 'base':
            assert v.spec() != V.VARIANTS['base'].spec(), name


def test_lowpass_capped_for_decimated_streams():
    assert V.lowpass_of(V.StreamSpec()) == C.LOWPASS_HZ
    assert V.lowpass_of(V.StreamSpec(lowpass_hz=None)) is None
    assert V.lowpass_of(V.StreamSpec(decimate=3)) == pytest.approx(0.4 * C.FS_HZ / 3)
    assert V.fs_of(V.StreamSpec(decimate=2)) == C.FS_HZ / 2


def test_bound_jitter_is_seeded_bounded_and_keeps_a_minimum_duration():
    tr = {'trial_id': 'abc', 'onset_ms': 1000.0, 'end_ms': 2500.0}
    w = V.WindowSpec(bound_jitter_ms=100.0)
    a, b = V.bounds(tr, w), V.bounds(tr, w)
    assert a == b                                                   # deterministic per trial
    assert abs(a[0] - 1000.0) <= 100.0 and abs(a[1] - 2500.0) <= 100.0
    assert V.bounds({**tr, 'trial_id': 'other'}, w) != a            # independent between trials
    on, en = V.bounds(tr, V.WindowSpec(onset_shift_ms=100.0))
    assert (on, en) == (1100.0, 2500.0)
    short = {'trial_id': 'x', 'onset_ms': 0.0, 'end_ms': 100.0}
    on, en = V.bounds(short, V.WindowSpec(onset_shift_ms=50.0))
    assert en - on == pytest.approx(V.MIN_JITTERED_MS)


@pytest.mark.parametrize('interp', ['linear', 'cubic', 'hold'])
def test_resample_keeps_the_grid_and_the_gap_mask(interp):
    t = np.concatenate([np.arange(0, 500, 16.7), np.arange(800, 1300, 16.7)])     # one 300-ms gap
    x = np.column_stack([np.sin(t / 100.0), np.full(t.size, 2.0)])
    tu, U = V._resample(t, x, C.FS_HZ, C.MAX_GAP_MS, interp)
    assert np.allclose(np.diff(tu), 1000.0 / C.FS_HZ)
    gap = (tu > 500) & (tu < 800)
    assert np.isnan(U[gap]).all() and np.isfinite(U[~gap]).all()
    assert np.allclose(U[~gap, 1], 2.0)                             # a constant stays constant
    if interp != 'hold':
        assert np.max(np.abs(U[~gap, 0] - np.sin(tu[~gap] / 100.0))) < 0.01


def test_hold_and_cubic_filling_of_long_gaps():
    t = np.arange(0.0, 1000.0, 1000.0 / 60)
    x = np.column_stack([t / 1000.0])
    x[20:30, 0] = np.nan
    hold = V._fill(t, x, 'hold')
    assert np.allclose(hold[20:30, 0], x[19, 0])                   # the last value before the gap is held
    cub = V._fill(t, x, 'cubic')
    assert np.allclose(cub[:, 0], t / 1000.0, atol=1e-9)           # a line is filled exactly


def test_causal_filter_delays_and_zero_phase_does_not():
    fs, n = C.FS_HZ, 600
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 1.0 * t)[:, None]
    zp = V._lowpass(x.copy(), fs, 15.0, 4, causal=False)
    ca = V._lowpass(x.copy(), fs, 15.0, 4, causal=True)
    mid = slice(120, 480)
    lag = lambda y: int(np.argmax(np.correlate(y[mid, 0], x[mid, 0], 'full')) - (mid.stop - mid.start - 1))  # noqa: E731
    assert lag(zp) == 0 and lag(ca) > 0                              # samples of delay at 1 Hz
    assert np.array_equal(V._lowpass(x.copy(), fs, None, 4, causal=True), x)   # no cut-off: unchanged
