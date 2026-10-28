"""MiniRocket (Dempster, Schmidt and Webb 2021), multivariate, written from scratch in numpy (aeon and sktime do not
install with numpy 2.4), plus the lane-m3 classifier on top.

Transform (as in the reference implementation):
* 84 fixed kernels of length 9: weight -1 everywhere and +2 at 3 of the 9 positions (C(9, 3) = 84);
* dilations 2^0 .. (T - 1) / 8 on a logarithmic grid, about ``num_features`` / 84 features per kernel spread over
  them; zero padding on every other (kernel, dilation) combination;
* every combination sums the convolution over a random subset of channels;
* biases: quantiles (a low-discrepancy sequence) of the convolution output of one random TRAINING example per
  combination - fitted inside fit_predict, on the training rows only;
* features: PPV, the proportion of positive values of (convolution - bias).

Classifier: features standardised on the training rows, then an L2 logistic regression whose C is chosen on
``fold.inner_splits()`` (log loss, GroupKFold by person). The label-free transform (channel normaliser, kernels,
biases) is fitted once per outer fold on its training rows; the inner splits refit the scaling and the regression. ``mode='aug'`` trains on the training rows plus their
mirror images (label flipped); ``mode='anti'`` uses the antisymmetric features phi(x) - phi(mirror(x)) without an
intercept, so p(mirror x) = 1 - p(x) exactly. The cross-fitted logits at the chosen C are kept (``oof_z``) for the
calibrators."""
from __future__ import annotations

import itertools
import sys
import warnings
from typing import Any

import numpy as np

from stage2.lanes import m3_common as M

KERNELS = np.array(list(itertools.combinations(range(9), 3)), dtype=int)      # positions of the +2 weights
PHI = (np.sqrt(5) + 1) / 2


def _quantiles(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) * PHI) % 1


def fit_dilations(t_len: int, num_features: int = 10_000, max_dilations_per_kernel: int = 32
                  ) -> tuple[np.ndarray, np.ndarray]:
    n_k = len(KERNELS)
    per_kernel = num_features // n_k
    true_max = min(per_kernel, max_dilations_per_kernel)
    multiplier = per_kernel / true_max
    max_exp = np.log2((t_len - 1) / (9 - 1))
    dil, counts = np.unique(np.logspace(0, max_exp, true_max, base=2).astype(np.int32), return_counts=True)
    feats = (counts * multiplier).astype(np.int32)
    rem = per_kernel - feats.sum()
    i = 0
    while rem > 0:
        feats[i] += 1
        rem -= 1
        i = (i + 1) % len(feats)
    return dil.astype(int), feats.astype(int)


def _shifted(X: np.ndarray, d: int) -> np.ndarray:
    """(9, N, C, T): X shifted by (j - 4) * d samples with zero padding ('same' dilated convolution taps)."""
    N, Cc, T = X.shape
    out = np.zeros((9, N, Cc, T), dtype=np.float32)
    for j in range(9):
        s = (j - 4) * d
        if s == 0:
            out[j] = X
        elif s > 0:
            if s < T:
                out[j, :, :, :T - s] = X[:, :, s:]
        else:
            if -s < T:
                out[j, :, :, -s:] = X[:, :, :T + s]
    return out


class MiniRocket:
    """Multivariate MiniRocket, vectorised over the 84 kernels of a dilation: a combination's channel subset is a 0/1
    mask, so the channel sums of all kernels and taps are one matrix product (tests compare it with a plain loop)."""

    def __init__(self, num_features: int = 10_000, seed: int = 0):
        self.num_features, self.seed = int(num_features), int(seed)

    def fit(self, X: np.ndarray) -> 'MiniRocket':
        X = np.asarray(X, dtype=np.float32)
        N, Cc, T = X.shape
        rng = np.random.default_rng(self.seed)
        self.dilations, self.feats_per_dil = fit_dilations(T, self.num_features)
        nd, nk = len(self.dilations), len(KERNELS)
        max_ch = min(Cc, 9)
        n_ch = (2 ** rng.uniform(0, np.log2(max_ch + 1), nd * nk)).astype(int)
        self.masks = np.zeros((nd, nk, Cc), dtype=np.float32)
        for c in range(nd * nk):
            ch = rng.choice(Cc, size=max(1, min(int(n_ch[c]), Cc)), replace=False)
            self.masks[c // nk, c % nk, ch] = 1.0
        self.biases = []
        kk = np.arange(nk)
        for di in range(nd):
            ex = rng.integers(N, size=nk)                                    # one random training example per kernel
            sh = _shifted(X[ex], int(self.dilations[di]))                    # (9, nk, C, T)
            S = np.einsum('jkct,kc->kjt', sh, self.masks[di], optimize=True)  # (nk, 9, T): kernel k on its own example
            conv = -S.sum(axis=1) + 3 * (S[kk, KERNELS[:, 0]] + S[kk, KERNELS[:, 1]] + S[kk, KERNELS[:, 2]])
            q = _quantiles(int(self.feats_per_dil[di]))
            self.biases.append(np.quantile(conv.astype(np.float64), q, axis=1).T.astype(np.float32))   # (nk, nf)
        return self

    @property
    def n_features(self) -> int:
        return int(sum(b.size for b in self.biases))

    def transform(self, X: np.ndarray, chunk: int = 128) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = [self._transform(X[i0:i0 + chunk]) for i0 in range(0, X.shape[0], chunk)]
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.n_features), np.float32)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Dilated taps by zero-padded shifting, the channel sums of all 84 kernels as one product with the masks,
        PPV by comparison with the biases (over the central part for the unpadded combinations)."""
        X = np.asarray(X, dtype=np.float32)
        N, Cc, T = X.shape
        kk = np.arange(len(KERNELS))
        feats = []
        for di, d in enumerate(self.dilations):
            sh = _shifted(X, int(d))                                         # (9, N, C, T)
            S = np.einsum('jnct,kc->kjnt', sh, self.masks[di], optimize=True)   # (nk, 9, N, T)
            conv = -S.sum(axis=1) + 3 * (S[kk, KERNELS[:, 0]] + S[kk, KERNELS[:, 1]] + S[kk, KERNELS[:, 2]])
            B = self.biases[di]
            ppv = (conv[:, :, None, :] > B[:, None, :, None]).mean(axis=-1)     # (nk, N, nf)
            pad = ((9 - 1) * int(d)) // 2
            if pad > 0 and T - 2 * pad > 0:
                unp = (di + kk) % 2 == 1
                ppv[unp] = (conv[unp][:, :, None, pad:T - pad] > B[unp][:, None, :, None]).mean(axis=-1)
            feats.append(np.transpose(ppv, (1, 0, 2)).reshape(N, -1))
        return np.concatenate(feats, axis=1).astype(np.float32)


# ───────────────────────────── the m3 family ─────────────────────────────

DEFAULTS: dict[str, Any] = dict(channels='grav', window='approach', norm='zscore', mode='aug',
                                num_features=4_000, Cs=(1e-4, 1e-3, 1e-2, 1e-1), max_iter=300)


def _lr(C: float, intercept: bool, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, fit_intercept=intercept, max_iter=max_iter, tol=1e-4)


class MiniRocketFamily:
    """MiniRocket features + L2 logistic regression (C nested on fold.inner_splits())."""

    def __init__(self, **params: Any):
        bad = set(params) - set(DEFAULTS)
        if bad:
            raise ValueError(f'unknown MiniRocket parameters {sorted(bad)}')
        self.p = {**DEFAULTS, **params}
        self.p['Cs'] = tuple(float(c) for c in self.p['Cs'])
        self.n_inner = None                                     # fold.inner_splits() default (5)

    def config(self) -> dict[str, Any]:
        return dict(self.p)

    def code_hash(self) -> str:
        return M.module_code_hash(M, sys.modules[__name__])

    def cache_name(self) -> str:
        return 'minirocket_' + M.digest(self.config())[:12]

    def _select(self, X: dict[str, np.ndarray]) -> np.ndarray:
        x = X['x'][:, M.channel_idx(self.p['channels'])].astype(np.float32)
        if 'mask' in X:                      # the fixed window: zero the frozen samples after the end cue
            x = x * X['mask'][:, None, :]
        return x

    def _features(self, mr: MiniRocket, norm: M.Normaliser, X: np.ndarray, signs: np.ndarray) -> np.ndarray:
        Z = norm.transform(X)
        F = mr.transform(Z)
        if self.p['mode'] == 'anti':
            F = F - mr.transform(Z * signs[None, :, None])
        return F

    def _design(self, F: np.ndarray, Fm: np.ndarray, y: np.ndarray):
        """Training design matrix, labels and scaling: 'aug' = rows + mirror rows (labels flipped), centred and
        scaled; 'anti' = phi(x) - phi(mirror x), scaled only (centring would break the antisymmetry)."""
        if self.p['mode'] == 'anti':
            Ftr, ytr, mu = F - Fm, y, None
            sd = Ftr.std(axis=0)
        else:
            Ftr, ytr = np.concatenate([F, Fm]), np.concatenate([y, 1 - y])
            mu, sd = Ftr.mean(axis=0), Ftr.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)                     # constant training features stay unscaled
        return Ftr, ytr, mu, sd

    @staticmethod
    def _scale(F: np.ndarray, mu: np.ndarray | None, sd: np.ndarray) -> np.ndarray:
        return (F - mu) / sd if mu is not None else F / sd

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
            splits: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict[str, Any]:
        """The label-free parts (channel normaliser, MiniRocket kernels and biases) are fitted once on the outer
        TRAINING rows and their mirrors; every inner split refits the feature scaling and the logistic regression on its
        own training persons, and predicts its held-out persons (cross-fitted logits, C chosen by their log loss)."""
        Xs = self._select(X)
        signs = M.mirror_signs(self.p['channels'])
        anti = self.p['mode'] == 'anti'
        norm = M.Normaliser(self.p['norm']).fit(Xs, signs)
        Z = norm.transform(Xs)
        Zm = Z * signs[None, :, None]
        mr = MiniRocket(self.p['num_features'], int(seed)).fit(np.concatenate([Z, Zm]))
        F, Fm = mr.transform(Z), mr.transform(Zm)
        Fte_all = F - Fm if anti else F
        Cs = self.p['Cs']
        oof = {c: np.full(y.size, np.nan) for c in Cs}
        for tr, va in splits:
            Ftr, ytr, mu, sd = self._design(F[tr], Fm[tr], y[tr])
            Ftr = self._scale(Ftr, mu, sd)
            Fva = self._scale(Fte_all[va], mu, sd)
            for c in Cs:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = _lr(c, not anti, self.p['max_iter']).fit(Ftr, ytr)
                oof[c][va] = m.decision_function(Fva)
        ll = {c: float(np.mean(np.logaddexp(0, -(2 * y - 1) * oof[c]))) for c in Cs}
        best = min(Cs, key=lambda c: (ll[c], c))
        Ftr, ytr, mu, sd = self._design(F, Fm, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = _lr(best, not anti, self.p['max_iter']).fit(self._scale(Ftr, mu, sd), ytr)
        return {'norm': norm, 'mr': mr, 'mu': mu, 'sd': sd, 'coef': m.coef_.ravel().astype(np.float64),
                'intercept': float(m.intercept_[0]) if not anti else 0.0, 'oof_z': oof[best],
                'info': {'C': best, 'inner_log_loss': {str(c): round(v, 5) for c, v in ll.items()},
                         'n_features': int(Ftr.shape[1])}}

    def logits(self, state: dict[str, Any], X: dict[str, np.ndarray]) -> np.ndarray:
        signs = M.mirror_signs(self.p['channels'])
        F = self._scale(self._features(state['mr'], state['norm'], self._select(X), signs), state['mu'], state['sd'])
        return F @ state['coef'] + state['intercept']


def factory(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """fit_predict of a MiniRocket configuration (finalists: ``factory(**params)``)."""
    return M.make_fit_predict(MiniRocketFamily(**params), calib=calib, use_cache=use_cache, cache_dir=cache_dir)
