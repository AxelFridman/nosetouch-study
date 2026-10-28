"""MiniRocket (multivariate) in plain numpy, for lane p1's panel (aeon / sktime do not install with numpy 2.4).

Follows Dempster, Schmidt and Webb (2021), "MINIROCKET: a very fast (almost) deterministic transform for time
series classification", and the multivariate variant of sktime (``MiniRocketMultivariate``):

* 84 fixed kernels of length 9: weight -1 everywhere and +2 at 3 of the 9 positions (every 3-subset);
* dilations 2^0 .. (T - 1) / 8, exponentially spaced, features shared out over them;
* each (dilation, kernel) pair sums the convolutions of a random subset of 1..min(C, 9) channels;
* biases are quantiles (low-discrepancy sequence) of the convolution output of one random TRAINING series
  per (dilation, kernel) pair, so :meth:`MiniRocket.fit` must only ever see training rows;
* a feature is the proportion of positive values (PPV) of the convolution output minus one bias; half of
  the (dilation, kernel) pairs use zero padding, the other half only the valid part of the output.

Deterministic for a given ``seed``. ``fit`` uses no label.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

KERNEL_LENGTH = 9
INDICES = np.array(list(combinations(range(KERNEL_LENGTH), 3)), dtype=np.int64)   # (84, 3): the +2 positions
N_KERNELS = len(INDICES)


def _quantiles(n: int) -> np.ndarray:
    """Low-discrepancy quantiles of the original implementation: frac(k * golden ratio), k = 1..n."""
    return np.array([(k * ((np.sqrt(5) + 1) / 2)) % 1 for k in range(1, n + 1)], dtype=np.float64)


def _dilations(length: int, num_features: int, max_dilations_per_kernel: int = 32) -> tuple[np.ndarray, np.ndarray]:
    per_kernel = num_features // N_KERNELS
    true_max = min(per_kernel, max_dilations_per_kernel)
    multiplier = per_kernel / true_max
    max_exponent = np.log2((length - 1) / (KERNEL_LENGTH - 1))
    dil, counts = np.unique(np.logspace(0, max_exponent, true_max, base=2).astype(np.int32), return_counts=True)
    feats = (counts * multiplier).astype(np.int32)
    remainder = per_kernel - int(feats.sum())
    i = 0
    while remainder > 0:
        feats[i] += 1
        remainder -= 1
        i = (i + 1) % len(feats)
    return dil.astype(np.int64), feats.astype(np.int64)


def _taps(X: np.ndarray, dilation: int) -> np.ndarray:
    """(C, 9, N, T) float32: B[c, j] = 3 x (channel c shifted by (j - 4) * dilation, zeros outside) - A[c] / 3,
    with A[c] the sum of the 9 shifted copies. A kernel with +2 taps (i0, i1, i2) on channels ch then outputs
    sum over c in ch of B[c, i0] + B[c, i1] + B[c, i2] (= -1 x all 9 taps + 3 x the three +2 taps)."""
    N, C, T = X.shape
    Xc = np.ascontiguousarray(np.transpose(X, (1, 0, 2)))          # (C, N, T)
    S = np.zeros((C, KERNEL_LENGTH, N, T), dtype=np.float32)
    for j in range(KERNEL_LENGTH):
        s = (j - KERNEL_LENGTH // 2) * dilation
        if s == 0:
            S[:, j] = Xc
        elif s > 0:
            if s < T:
                S[:, j, :, :T - s] = Xc[:, :, s:]
        elif -s < T:
            S[:, j, :, -s:] = Xc[:, :, :T + s]
    A = S.sum(axis=1)                                                # (C, N, T)
    S *= 3.0
    S -= (A / 3.0)[:, None]
    return S


def _ppv(Cout: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """(N, nb): proportion of the T outputs above each bias (row-wise: one row never affects another)."""
    T = Cout.shape[1]
    return np.stack([np.count_nonzero(Cout > b, axis=1) for b in biases], axis=1).astype(np.float32) / T


class MiniRocket:
    """Multivariate MiniRocket transform. ``fit(X)`` (training rows only) sets the channel subsets and biases;
    ``transform(X)`` returns (N, n_features) PPV features (float32). X: (N, C, T) float."""

    def __init__(self, num_features: int = 10_000, seed: int = 0, max_dilations_per_kernel: int = 32):
        self.num_features = int(num_features)
        self.seed = int(seed)
        self.max_dilations_per_kernel = int(max_dilations_per_kernel)

    def fit(self, X: np.ndarray) -> 'MiniRocket':
        X = np.asarray(X, dtype=np.float32)
        N, C, T = X.shape
        if T < KERNEL_LENGTH:
            raise ValueError(f'series of {T} samples: MiniRocket needs at least {KERNEL_LENGTH}')
        rng = np.random.default_rng(self.seed)
        self.n_channels_, self.length_ = C, T
        self.dilations_, self.feats_per_dilation_ = _dilations(T, self.num_features, self.max_dilations_per_kernel)
        n_comb = len(self.dilations_) * N_KERNELS
        max_ch = min(C, 9)
        n_ch = (2 ** rng.uniform(0, np.log2(max_ch + 1), n_comb)).astype(np.int64)
        n_ch = np.clip(n_ch, 1, max_ch)
        self.channels_ = [np.sort(rng.choice(C, int(k), replace=False)) for k in n_ch]
        q = _quantiles(int(self.feats_per_dilation_.sum()) * N_KERNELS)
        self.biases_: list[np.ndarray] = []
        examples = rng.integers(0, N, n_comb)
        qi = 0
        comb = 0
        for di, dil in enumerate(self.dilations_):
            nf = int(self.feats_per_dilation_[di])
            for k in range(N_KERNELS):
                ex = int(examples[comb])
                B = _taps(X[ex:ex + 1], int(dil))
                Cout = self._conv(B, k, self.channels_[comb])[0]
                self.biases_.append(np.quantile(Cout, q[qi:qi + nf]).astype(np.float32))
                qi += nf
                comb += 1
        self.n_features_ = int(sum(b.size for b in self.biases_))
        return self

    @staticmethod
    def _conv(B: np.ndarray, k: int, ch: np.ndarray) -> np.ndarray:
        """(N, T) output of kernel k summed over the channels ch (B from :func:`_taps`)."""
        i0, i1, i2 = INDICES[k]
        out = None
        for c in ch:
            t = B[c, i0] + B[c, i1]
            t += B[c, i2]
            out = t if out is None else out + t
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        N, C, T = X.shape
        if C != self.n_channels_:
            raise ValueError(f'fitted on {self.n_channels_} channels, got {C}')
        out = np.empty((N, self.n_features_), dtype=np.float32)
        f0 = 0
        comb = 0
        for di, dil in enumerate(self.dilations_):
            B = _taps(X, int(dil))
            pad = ((KERNEL_LENGTH - 1) * int(dil)) // 2
            for k in range(N_KERNELS):
                Cout = self._conv(B, k, self.channels_[comb])        # (N, T)
                if (di + k) % 2 == 1 and T - 2 * pad > 0:
                    Cout = Cout[:, pad:T - pad]
                b = self.biases_[comb]
                out[:, f0:f0 + b.size] = _ppv(Cout, b)
                f0 += b.size
                comb += 1
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
