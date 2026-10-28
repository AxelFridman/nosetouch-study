"""Finding 6 (the spectrum never crops a window) and finding 10 (every cache and result names its code)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from stage2 import config as C
from stage2.qc import cumulative_power, fft_length, share_below


def test_fft_length_covers_the_window():
    assert fft_length(10) == 256 and fft_length(256) == 256 and fft_length(257) == 512 and fft_length(721) == 1024


def test_long_window_is_not_cropped():
    """A 2 Hz movement followed by a 20 Hz tail of equal amplitude: numpy's rfft(x, n=256) would see only the
    first 256 samples (all 2 Hz) and report ~100 % below 15 Hz; the full-window spectrum sees the tail."""
    fs = 60.0
    t = np.arange(600) / fs
    x = np.where(np.arange(600) < 256, np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 20 * t))[:, None]
    f, cum = cumulative_power(x, fs, fft_length(x.shape[0]))
    below = share_below(f, cum, 15.0)
    assert 0.3 < below < 0.6
    with pytest.raises(ValueError):
        cumulative_power(x, fs, 256)
    cropped = np.abs(np.fft.rfft(x[:, 0] * np.hanning(600), n=256)) ** 2   # the old computation, for contrast
    assert np.cumsum(cropped)[64] / cropped.sum() > 0.95


def test_zero_padding_only_interpolates():
    fs = 60.0
    x = np.sin(2 * np.pi * 3 * np.arange(200) / fs)[:, None]
    f1, c1 = cumulative_power(x, fs, 256)
    f2, c2 = cumulative_power(x, fs, 1024)
    assert share_below(f1, c1, 15.0) == pytest.approx(share_below(f2, c2, 15.0), abs=0.01)


def test_spectrum_summary_uses_the_full_windows():
    p = C.TAB_DIR / 'describe_summary.json'
    if not p.exists():
        pytest.skip('no describe output (python -m stage2.scripts.describe)')
    s = json.loads(p.read_text(encoding='utf-8'))['spectrum']
    assert s['n_fft'] >= s['longest_window_samples'] and s['windows_over_256_samples'] > 0
    assert s['gyro_power_below_15hz'] > 0.98 and s['acc_power_below_15hz'] > 0.97


def test_cache_and_results_carry_the_code_fingerprint(primary):
    meta = primary.meta
    fp = C.code_fingerprint()
    assert set(meta['code']) == set(fp) and all(len(v) == 64 for v in meta['code'].values())
    # the code that BUILDS the cache (data.py, standardize.py, motionlab) and its inputs form its data identity;
    # editing presentation or pipeline modules leaves the cache valid (stage2.pipeline rebuilds it otherwise)
    from stage2.data import builder_sha256, data_identity
    ident = meta['data_identity']
    assert ident['builder_sha256'] == builder_sha256(), \
        'the cache was built by other builder code: rebuild it (python -m stage2.pipeline cache)'
    assert ident == data_identity(), 'the cache does not match its inputs: python -m stage2.pipeline cache'
    assert meta['code']['motionlab_sha256'] == fp['motionlab_sha256']
    assert meta['cache_version'] == C.CACHE_VERSION
    res = C.RES_DIR / 'example_yaw_logreg.json'
    if res.exists():
        info = json.loads(res.read_text(encoding='utf-8'))['info']
        assert set(info['code']) == set(fp) and info['cache_code']['code'] == meta['code']
