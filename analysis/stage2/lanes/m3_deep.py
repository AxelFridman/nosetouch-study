"""Lane m3 deep sequence models: a cross-validated ensemble trained on CPU with PyTorch.

For one outer fold (the training persons only):

* the training persons are split by ``fold.inner_splits(n_inner)`` (GroupKFold by person); member k trains on
  the other parts and is EARLY-STOPPED on part k (validation log loss on real trials, never augmented, never
  the test person), with its own seed;
* every member fits its own channel normaliser on its training rows and their mirror images
  (``m3_common.Normaliser``);
* augmentation on the training rows of a member: the mirror image with the flipped label (the training set is
  doubled), time warp (approach) or shift and speed change about the onset (fixed window), magnitude scaling
  and jitter;
* the ensemble logit is the mean of the member logits; member k's logits on part k give the cross-fitted
  training logits (``oof_z``) that calibrators and the attempt-aggregation scale are fitted on.

``anti=True`` builds the mirror-antisymmetric model z(x) = g(x) - g(mirror(x)) (p(mirror x) = 1 - p(x) exactly;
mirror augmentation is then redundant and not used).

Integrated gradients (:func:`integrated_gradients`) attribute the ensemble logit to every (channel, time)
input, with the normalised zero (the training mean, mirror-invariant) as the baseline.

Reproducibility: torch's CPU kernels sum in an order that depends on the number of intra-op threads, so the same
seed gives different networks at 1 and at 2 threads. Every fit, prediction and attribution therefore runs inside
:func:`torch_runtime`: ``TRAIN_THREADS`` (1) intra-op thread and deterministic algorithms, whatever the process's
own thread setting; parallelism comes from processes (``stage2.scripts.run_m3``, the compute step's worker
streams). The torch version and ``TRAIN_THREADS`` are part of the fit-cache key (``DeepFamily.runtime``)."""
from __future__ import annotations

import copy
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn

from stage2 import config as C
from stage2.lanes import m3_common as M
from stage2.lanes import m3_nets

#: intra-op threads of every deep fit, prediction and attribution (fixed: see the module docstring)
TRAIN_THREADS = 1


@contextmanager
def torch_runtime() -> Iterator[None]:
    """``TRAIN_THREADS`` intra-op threads and deterministic algorithms inside the block; the process's settings are
    restored afterwards."""
    n0 = torch.get_num_threads()
    det0 = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(TRAIN_THREADS)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det0)
        torch.set_num_threads(n0)


DEFAULTS: dict[str, Any] = dict(
    arch='cnn', channels='grav', window='approach', norm='zscore', aug=('mirror', 'warp', 'scale', 'jitter'),
    anti=False, head='bins', n_bins=8, width=32, dropout=0.2, n_inner=4, max_epochs=40, patience=6,
    min_epochs=6, lr=2e-3, weight_decay=1e-3, batch=64, warp=0.3, scale=0.1, jitter=0.03, shift=8,
    speed=0.15, tf=None, tf_mode='signed')


class DeepFamily:
    """A deep model family (see the module docstring). ``tf`` (None, 'stft', 'cwt') with ``tf_mode`` ('signed':
    real and imaginary planes; 'mag': magnitude only) switches the input to time-frequency planes and the
    network to the 2D CNN."""

    def __init__(self, **params: Any):
        bad = set(params) - set(DEFAULTS)
        if bad:
            raise ValueError(f'unknown deep parameters {sorted(bad)}')
        self.p = {**DEFAULTS, **params}
        self.p['aug'] = tuple(self.p['aug'])
        if self.p['tf'] is not None:
            self.p['arch'] = 'cnn2d'
        self.n_inner = int(self.p['n_inner'])

    # identity of the family for the fit cache
    def config(self) -> dict[str, Any]:
        return dict(self.p)

    def code_hash(self) -> str:
        return M.module_code_hash(M, m3_nets, sys.modules[__name__])

    def runtime(self) -> dict[str, Any]:
        """What shapes a fit besides code, configuration and inputs (part of the cache key, ``M.runtime_signature``)."""
        return {'torch': torch.__version__, 'train_threads': TRAIN_THREADS, 'deterministic_algorithms': True}

    def cache_name(self) -> str:
        return 'deep_' + M.digest(self.config())[:12]

    # ───────── inputs ─────────
    def _signs(self) -> np.ndarray:
        s = M.mirror_signs(self.p['channels'])
        if self.p['tf'] is not None:
            s = np.repeat(s, 2) if self.p['tf_mode'] == 'signed' else np.ones_like(s)
        return s

    def _select(self, X: dict[str, np.ndarray]) -> np.ndarray:
        """(N, C', T) or (N, C', F, T) float32: the chosen channels (TF: planes) of the per-trial input."""
        idx = M.channel_idx(self.p['channels'])
        x = X['x'][:, idx]
        if self.p['tf'] is not None:                       # x: (N, C, 2, F, T) real and imaginary parts
            if self.p['tf_mode'] == 'signed':
                x = x.reshape(x.shape[0], -1, *x.shape[3:])
            else:
                x = np.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
        return np.ascontiguousarray(x, dtype=np.float32)

    def _with_mask(self, Z: np.ndarray, X: dict[str, np.ndarray]) -> np.ndarray:
        if 'mask' in X and self.p['tf'] is None:
            return np.concatenate([Z, X['mask'][:, None, :].astype(np.float32)], axis=1)
        return Z

    def _full_signs(self, X: dict[str, np.ndarray]) -> np.ndarray:
        s = self._signs()
        if 'mask' in X and self.p['tf'] is None:
            s = np.concatenate([s, [1.0]]).astype(np.float32)
        return s

    # ───────── augmentation (numpy, on standardised batches) ─────────
    def _augment(self, xb: np.ndarray, rng: np.random.Generator, n_sig: int) -> np.ndarray:
        p, aug = self.p, self.p['aug']
        B, T = xb.shape[0], xb.shape[-1]
        x = xb.copy()
        if self.p['tf'] is None and ('warp' in aug or 'shift' in aug):
            u = np.arange(T, dtype=np.float64)
            if self.p['window'] == 'fixed':                 # shift and speed change about the onset sample
                on = int(round(M_FIXED_ONSET_FRAC * T))
                r = 1.0 + rng.uniform(-p['speed'], p['speed'], size=(B, 1)) if 'warp' in aug else np.ones((B, 1))
                s = rng.integers(-p['shift'], p['shift'] + 1, size=(B, 1)) if 'shift' in aug else np.zeros((B, 1))
                idx = on + (u[None, :] - on) * r + s
            else:                                           # smooth monotone warp of normalised time
                a = rng.uniform(-p['warp'], p['warp'], size=(B, 1))
                v = u[None, :] / (T - 1)
                idx = (v + a * v * (1 - v)) * (T - 1)
            idx = np.clip(idx, 0, T - 1)
            lo = np.floor(idx).astype(int)
            hi = np.minimum(lo + 1, T - 1)
            w = (idx - lo)[:, None, :].astype(np.float32)
            lo3 = np.broadcast_to(lo[:, None, :], x.shape)
            hi3 = np.broadcast_to(hi[:, None, :], x.shape)
            x = np.take_along_axis(x, lo3, axis=2) * (1 - w) + np.take_along_axis(x, hi3, axis=2) * w
        if 'scale' in aug:
            f = 1.0 + rng.uniform(-p['scale'], p['scale'], size=(B,) + (1,) * (x.ndim - 1))
            x[:, :n_sig] = x[:, :n_sig] * f
        if 'jitter' in aug:
            x[:, :n_sig] = x[:, :n_sig] + rng.normal(0, p['jitter'], size=x[:, :n_sig].shape)
        return x.astype(np.float32)

    # ───────── model ─────────
    def _net(self, cin: int, t_len: int, seed: int) -> nn.Module:
        torch.manual_seed(int(seed) % (2 ** 31))
        return m3_nets.build_net(self.p['arch'], cin, t_len, width=self.p['width'], n_bins=self.p['n_bins'],
                                 dropout=self.p['dropout'], head=self.p['head'])

    def _forward(self, net: nn.Module, x: torch.Tensor, sign_t: torch.Tensor) -> torch.Tensor:
        if not self.p['anti']:
            return net(x)
        g = net(torch.cat([x, x * sign_t], dim=0))
        return g[: x.shape[0]] - g[x.shape[0]:]

    def _logits_np(self, net: nn.Module, Z: np.ndarray, sign_t: torch.Tensor, bs: int = 512) -> np.ndarray:
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, Z.shape[0], bs):
                out.append(self._forward(net, torch.from_numpy(Z[i:i + bs]), sign_t).numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)

    def _train_member(self, Xa: np.ndarray, ya: np.ndarray, Xv: np.ndarray, yv: np.ndarray, signs: np.ndarray,
                      seed: int, mask_a: dict[str, np.ndarray], mask_v: dict[str, np.ndarray]) -> dict[str, Any]:
        p = self.p
        rng = np.random.default_rng(seed)
        norm = M.Normaliser(p['norm']).fit(Xa, signs)
        Za = self._with_mask(norm.transform(Xa), mask_a)
        Zv = self._with_mask(norm.transform(Xv), mask_v)
        fs = self._full_signs(mask_a)
        n_sig = Xa.shape[1]
        ya = ya.astype(np.float32)
        if 'mirror' in p['aug'] and not p['anti']:
            Za = np.concatenate([Za, Za * fs.reshape((1, -1) + (1,) * (Za.ndim - 2))])
            ya = np.concatenate([ya, 1 - ya])
        sign_t = torch.from_numpy(fs.reshape((1, -1) + (1,) * (Za.ndim - 2)).astype(np.float32))
        net = self._net(Za.shape[1], Za.shape[-1], seed)
        opt = torch.optim.AdamW(net.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, best_ep, best_zv, bad = np.inf, None, 0, None, 0
        yv_f = yv.astype(float)
        for ep in range(int(p['max_epochs'])):
            net.train()
            perm = rng.permutation(Za.shape[0])
            for i in range(0, perm.size, p['batch']):
                b = perm[i:i + p['batch']]
                if b.size < 2:
                    continue
                xb = torch.from_numpy(self._augment(Za[b], rng, n_sig))
                opt.zero_grad()
                loss = lossf(self._forward(net, xb, sign_t), torch.from_numpy(ya[b]))
                loss.backward()
                opt.step()
            zv = self._logits_np(net, Zv, sign_t)
            pv = np.clip(M.sigmoid(zv), 1e-7, 1 - 1e-7)
            nll = float(-np.mean(yv_f * np.log(pv) + (1 - yv_f) * np.log(1 - pv)))
            if nll < best - 1e-5:
                best, best_state, best_ep, best_zv, bad = nll, copy.deepcopy(net.state_dict()), ep + 1, zv, 0
            else:
                bad += 1
                if ep + 1 >= p['min_epochs'] and bad >= p['patience']:
                    break
        return {'state': best_state, 'norm': norm, 'epochs': best_ep, 'val_nll': best, 'zv': best_zv,
                'cin': int(Za.shape[1]), 't_len': int(Za.shape[-1])}

    # ───────── the family API ─────────
    def fit(self, X: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
            splits: list[tuple[np.ndarray, np.ndarray]], seed: int) -> dict[str, Any]:
        with torch_runtime():
            Xs = self._select(X)
            signs = self._signs()
            oof = np.full(y.size, np.nan)
            members = []
            for k, (tr, va) in enumerate(splits):
                ms = int(np.random.SeedSequence([int(seed), 31, k]).generate_state(1)[0])
                sub_a = {kk: v[tr] for kk, v in X.items() if kk == 'mask'}
                sub_v = {kk: v[va] for kk, v in X.items() if kk == 'mask'}
                m = self._train_member(Xs[tr], y[tr], Xs[va], y[va], signs, ms, sub_a, sub_v)
                oof[va] = m.pop('zv')
                members.append(m)
            if np.isnan(oof).any():
                raise RuntimeError('the inner splits did not cover every training row')
            n_par = m3_nets.n_params(self._net(members[0]['cin'], members[0]['t_len'], 0))
        return {'members': members, 'oof_z': oof,
                'info': {'epochs': [m['epochs'] for m in members], 'n_members': len(members), 'n_params': n_par,
                         'threads': TRAIN_THREADS}}

    def _member_net(self, m: dict[str, Any]) -> nn.Module:
        net = self._net(m['cin'], m['t_len'], 0)
        net.load_state_dict(m['state'])
        net.eval()
        return net

    def member_inputs(self, m: dict[str, Any], X: dict[str, np.ndarray]) -> np.ndarray:
        return self._with_mask(m['norm'].transform(self._select(X)), X)

    def logits(self, state: dict[str, Any], X: dict[str, np.ndarray]) -> np.ndarray:
        with torch_runtime():
            fs = self._full_signs(X)
            zs = []
            for m in state['members']:
                Z = self.member_inputs(m, X)
                sign_t = torch.from_numpy(fs.reshape((1, -1) + (1,) * (Z.ndim - 2)).astype(np.float32))
                zs.append(self._logits_np(self._member_net(m), Z, sign_t))
            return np.mean(zs, axis=0)

    def baseline_logit(self, state: dict[str, Any], X: dict[str, np.ndarray]) -> float:
        """Ensemble logit at the integrated-gradients baseline (each member's normalised zero)."""
        with torch_runtime():
            fs = self._full_signs(X)
            zs = []
            for m in state['members']:
                Z0 = np.zeros((1, m['cin']) + tuple(self.member_inputs(m, {k: v[:1] for k, v in X.items()}).shape[2:]),
                              dtype=np.float32)
                sign_t = torch.from_numpy(fs.reshape((1, -1) + (1,) * (Z0.ndim - 2)).astype(np.float32))
                zs.append(float(self._logits_np(self._member_net(m), Z0, sign_t)[0]))
            return float(np.mean(zs))

    def integrated_gradients(self, state: dict[str, Any], X: dict[str, np.ndarray], steps: int = 32) -> np.ndarray:
        """IG of the ensemble logit, (N, C_in, T): sum_m IG_m / n_members (IG is linear in the logit), baseline =
        the normalised zero of each member. Completeness: the sum over (channel, time) is z(x) - z(baseline)."""
        with torch_runtime():
            fs = self._full_signs(X)
            out = None
            alphas = (np.arange(steps) + 0.5) / steps
            for m in state['members']:
                net = self._member_net(m)
                Z = torch.from_numpy(self.member_inputs(m, X))
                sign_t = torch.from_numpy(fs.reshape((1, -1) + (1,) * (Z.ndim - 2)).astype(np.float32))
                grads = torch.zeros_like(Z)
                for a in alphas:
                    xa = (Z * float(a)).requires_grad_(True)
                    z = self._forward(net, xa, sign_t)
                    g, = torch.autograd.grad(z.sum(), xa)
                    grads += g
                ig = (Z * grads / steps).detach().numpy().astype(np.float64)
                out = ig if out is None else out + ig
            return out / len(state['members'])


M_FIXED_ONSET_FRAC = C.FIXED_PRE_S / (C.FIXED_PRE_S + C.FIXED_POST_S)   # where the onset sits in the fixed window


def family(**params: Any) -> DeepFamily:
    return DeepFamily(**params)


def factory(calib: str = 'none', use_cache: bool = True, cache_dir: str | None = None, **params: Any):
    """fit_predict of a deep configuration (finalists: ``factory(**params)``)."""
    return M.make_fit_predict(DeepFamily(**params), calib=calib, use_cache=use_cache, cache_dir=cache_dir)
