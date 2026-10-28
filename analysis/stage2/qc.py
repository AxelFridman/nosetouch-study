"""Numerical helpers of the descriptive analysis (kept out of the script so they can be unit-tested)."""
from __future__ import annotations

import math

import numpy as np
from scipy import signal


def fft_length(n_samples: int, minimum: int = 256) -> int:
    """The smallest power of two >= max(n_samples, minimum): an FFT of this length never crops the window."""
    return int(max(minimum, 2 ** math.ceil(math.log2(max(1, int(n_samples))))))


def cumulative_power(U: np.ndarray, fs_hz: float, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, cumulative share of power) of the columns of U (n, k) summed over columns.

    The window is detrended and Hann-tapered over its FULL length, then zero-padded to ``n_fft`` (which must
    be >= n: numpy's rfft would otherwise silently crop the window, losing its end and leaving it untapered).
    Zero padding only interpolates the spectrum, so windows of different lengths share one frequency grid."""
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U[:, None]
    n = U.shape[0]
    if n > n_fft:
        raise ValueError(f'window of {n} samples is longer than the FFT ({n_fft}): use fft_length(n)')
    X = signal.detrend(U, axis=0) * signal.windows.hann(n)[:, None]
    P = (np.abs(np.fft.rfft(X, n=n_fft, axis=0)) ** 2).sum(axis=1)
    tot = P.sum()
    return np.fft.rfftfreq(n_fft, d=1.0 / fs_hz), (np.cumsum(P) / tot if tot > 0 else np.full(P.size, np.nan))


def share_below(freqs: np.ndarray, cum: np.ndarray, f_hz: float) -> float:
    """Share of the power at frequencies <= f_hz (from :func:`cumulative_power`)."""
    k = int(np.searchsorted(freqs, f_hz, side='right')) - 1
    return float(cum[max(k, 0)])
