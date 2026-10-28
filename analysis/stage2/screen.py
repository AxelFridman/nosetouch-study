"""docs/09 D6: screening for the wrong hand, and the per-block label-consistency table.

With hands-free review, ``accepted`` only means "completed and not redone": a participant who used the wrong
hand for a whole block leaves no trace in the record. The labels are never changed automatically. This
module gives two things:

* :func:`wrong_hand_flags` - the D6 screen for use INSIDE a fold: an antisymmetric logistic model (no
  intercept, chiral features only, so p(mirror(x)) = 1 - p(x)) scored by inner leave-one-person-out over
  the TRAINING persons. It flags training trials with p(label) < ``p_flag`` (0.05) and runs of at least
  ``run_len`` (3) consecutive trials of a session with p(label) < ``run_p`` (0.25). Flags remove trials from
  training only, never from evaluation, so the cleaning cannot inflate accuracy::

      F = screen.screen_features(fold.X_train)          # X = the approach arrays (N, C, 64)
      keep = ~screen.wrong_hand_flags(F, fold.y_train, fold.groups_train, fold.meta_train)['flag'].to_numpy()

* :func:`block_consistency` - the descriptive table for the researcher: for every planned block, how many
  of its trials agree with the instructed hand (first-half yaw sign rule, and the screen model scored
  leave-one-person-out), and the **swap signature**: two adjacent blocks of one session with opposite
  instructed hands, each of at least 3 trials, in which EVERY trial contradicts its label. A real hand swap
  at a block change produces exactly that; a dominance effect (one hand moving differently) reverses one
  hand only.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .standardize import CHANNELS

#: Chiral (mirror-odd), grip-invariant channels and the instants of the time-normalised approach used by the
#: screen model: the cumulative yaw and the yaw rate at 1/8, 2/8, ..., 8/8 of the approach.
SCREEN_CHANNELS: tuple[str, ...] = ('yaw_cum', 'yaw_rate')
SCREEN_STEPS: tuple[int, ...] = (7, 15, 23, 31, 39, 47, 55, 63)
P_FLAG = 0.05
RUN_LEN = 3
RUN_P = 0.25


def screen_features(approach: np.ndarray, channels: Sequence[str] = SCREEN_CHANNELS,
                    steps: Sequence[int] = SCREEN_STEPS) -> np.ndarray:
    """(N, len(channels) * len(steps)) chiral features from the approach arrays (N, C, 64) of the cache."""
    idx = [CHANNELS.index(c) for c in channels]
    A = np.asarray(approach, dtype=float)
    return np.concatenate([A[:, k, list(steps)] for k in idx], axis=1)


def antisymmetric_logreg(F: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple[LogisticRegression, np.ndarray]:
    """Logistic regression without intercept on RMS-scaled (not centred) odd features: p(-x) = 1 - p(x)."""
    s = np.sqrt(np.mean(F ** 2, axis=0))
    s[~(s > 0)] = 1.0
    m = LogisticRegression(fit_intercept=False, C=C, max_iter=5000).fit(F / s, y)
    return m, s


def loso_scores(F: np.ndarray, y: np.ndarray, groups: Sequence[Any], C: float = 1.0) -> np.ndarray:
    """P(right) of every row from the antisymmetric model fitted on the OTHER groups (leave-one-group-out)."""
    g = np.asarray(groups).astype(str)
    y = np.asarray(y).astype(int)
    p = np.full(y.size, np.nan)
    for pid in sorted(set(g.tolist())):
        te = g == pid
        if (~te).sum() < 2 or len(set(y[~te].tolist())) < 2:
            continue
        m, s = antisymmetric_logreg(F[~te], y[~te], C)
        p[te] = m.predict_proba(F[te] / s)[:, 1]
    return p


def _runs(flag: np.ndarray) -> np.ndarray:
    """Length of the run of consecutive True values each element belongs to (0 where False)."""
    out = np.zeros(flag.size, dtype=int)
    i = 0
    while i < flag.size:
        if flag[i]:
            j = i
            while j < flag.size and flag[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def _session_order(meta: pd.DataFrame | None, n: int) -> list[np.ndarray]:
    """Row indices of each session in trial order (one block of all rows when meta has no session info)."""
    if meta is None or 'session_id' not in meta.columns:
        return [np.arange(n)]
    m = meta.reset_index(drop=True)
    keys = [c for c in ('trial_index', 'attempt') if c in m.columns]
    out = []
    for _, g in m.groupby('session_id', sort=False):
        out.append((g.sort_values(keys) if keys else g).index.to_numpy())
    return out


def wrong_hand_flags(F: np.ndarray, y: Sequence[int], groups: Sequence[Any], meta: pd.DataFrame | None = None, *,
                     p_flag: float = P_FLAG, run_len: int = RUN_LEN, run_p: float = RUN_P, C: float = 1.0) -> pd.DataFrame:
    """The D6 screen on the rows given (call it on TRAINING rows only): p_label, flag_low_p, run_length,
    flag_run and flag (either) per row, in the input order."""
    y = np.asarray(y).astype(int)
    p = loso_scores(np.asarray(F, dtype=float), y, groups, C)
    pl = np.where(y == 1, p, 1 - p)
    low = np.nan_to_num(pl, nan=1.0) < p_flag
    conf = np.nan_to_num(pl, nan=1.0) < run_p
    run = np.zeros(y.size, dtype=int)
    for idx in _session_order(meta, y.size):
        run[idx] = _runs(conf[idx])
    flag_run = run >= run_len
    return pd.DataFrame({'p_right': p, 'p_label': pl, 'flag_low_p': low, 'run_length': run, 'flag_run': flag_run,
                         'flag': low | flag_run})


def block_consistency(trials: pd.DataFrame, approach: np.ndarray, *, min_block: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-trial scores, per-block table) of the primary trials (descriptive QA, every person scored by a
    model that never saw them). Per block: n, instructed hand, grip(s), grip mismatches, the share of trials
    whose first-half yaw sign agrees with the label, the screen model's mean p(label) and number of reversed
    trials, and the swap signature under each evidence."""
    t = trials.reset_index(drop=True).copy()
    y = t['y_right'].astype(int).to_numpy()
    F = screen_features(approach)
    fl = wrong_hand_flags(F, y, t['person_id'].astype(str).to_numpy(), t)
    t['p_label_screen'] = fl['p_label'].to_numpy()
    t['screen_flag'] = fl['flag'].to_numpy()
    t['fhy_agrees'] = (t['first_half_yaw_deg'] > 0).to_numpy() == (y == 1)
    t['screen_agrees'] = t['p_label_screen'] >= 0.5
    keys = ['person_id', 'session_label', 'block_index']
    rows = []
    for (pid, sl, bi), g in t.groupby(keys, sort=True):
        tier = g['tier'].iloc[0]
        rows.append({'person_id': pid, 'session_label': sl, 'block_index': int(bi),
                     'tier': int(tier) if pd.notna(tier) else None,
                     'hand': g['hand'].iloc[0], 'grip': ';'.join(sorted(set(g['orientation'].astype(str)))),
                     'n': len(g), 'grip_mismatch': int(g['grip_mismatch'].sum()) if 'grip_mismatch' in g else 0,
                     'fhy_agree': int(g['fhy_agrees'].sum()), 'fhy_share_agree': float(g['fhy_agrees'].mean()),
                     'screen_agree': int(g['screen_agrees'].sum()), 'screen_mean_p_label': float(g['p_label_screen'].mean()),
                     'screen_flags': int(g['screen_flag'].sum())})
    b = pd.DataFrame(rows)
    for ev in ('fhy', 'screen'):
        b[f'{ev}_all_contradict'] = (b[f'{ev}_agree'] == 0) & (b['n'] >= min_block)
        sig = np.zeros(len(b), dtype=bool)
        for _, g in b.groupby(['person_id', 'session_label'], sort=False):
            g = g.sort_values('block_index')
            ix = g.index.to_numpy()
            for a, c in zip(ix[:-1], ix[1:]):
                if (b.at[a, 'hand'] != b.at[c, 'hand'] and b.at[a, f'{ev}_all_contradict']
                        and b.at[c, f'{ev}_all_contradict']):
                    sig[[a, c]] = True
        b[f'{ev}_swap_signature'] = sig
    return t, b
