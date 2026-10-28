"""Lane m4, part 5: online operation - when to stop and answer.

The detector of ``m4_detect`` is causal: the activity filter is one-pass, a segment is confirmed when the
stillness after it has lasted the look-ahead (``SegParams.lookahead_s``), and its window is cut from samples
the detector has already seen (``window_for(..., post_s=lookahead_s)``). Running it on a recorded stream
therefore gives exactly what it gives online, at the times ``decided_ms``; this module replays streams in
that order.

Sequential stopping rule (:func:`stopping_rule`): after each accepted attempt k (a segment classified as an
approach) the attempts' calibrated logits are aggregated as P_k = sigmoid(s_k * mean(z_1..z_k))
(``m4_models.AggregateScale``; ``agg=None`` is the plain mean logit, s_k = 1). The pipeline stops and answers
as soon as max(P_k, 1 - P_k) >= ``p_stop``; after ``k_max`` attempts it answers when max(P, 1 - P) >=
``p_answer`` and otherwise abstains ("please repeat"). Returns and other movements are ignored (they never
count as attempts).

Aggregations compared (:data:`AGGREGATIONS`): the mean logit (s_k = 1, as if the attempts were one), the sum
(s_k = k, as if they were independent), s_k fitted on random k-subsets of the training persons' RECORDINGS
(``AggregateScale.fit``) and s_k fitted on the attempts of cross-fitted SESSIONS of the training persons
(``AggregateScale.fit_sessions``, ``m4_aggregate``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import m4_models as M

#: every aggregation of k attempts that the sessions evaluate (the two scales are fitted per fold)
AGGREGATIONS = ('mean', 'sum', 'scale_recordings', 'scale_sessions')
AGGREGATION_LABELS = {'mean': 'mean logit', 'sum': 'sum of logits', 'scale_recordings': '$s_k$ fitted on recordings',
                      'scale_sessions': '$s_k$ fitted on sessions'}


def aggregate(z: Sequence[float], agg: M.AggregateScale | None, how: str = 'scaled') -> float:
    """P(right) of k attempts' calibrated logits: 'scaled' (s_k * mean; agg None = mean), 'mean' or 'sum'."""
    z = np.asarray(z, dtype=float)
    if not z.size:
        return 0.5
    if how == 'mean':
        return float(M.sigmoid(z.mean()))
    if how == 'sum':
        return float(M.sigmoid(z.sum()))
    s = agg.scale(int(z.size)) if agg is not None else 1.0
    return float(M.sigmoid(s * z.mean()))


def stopping_rule(att: pd.DataFrame, agg: M.AggregateScale | None, *, p_stop: float = 0.9, k_max: int = 5,
                  p_answer: float = 0.5) -> dict[str, Any]:
    """Replay the attempts (in decision order) through the stopping rule (module docstring)."""
    att = att.sort_values('decided_ms').reset_index(drop=True) if len(att) else att
    z = att['z'].to_numpy(float) if len(att) else np.zeros(0)
    p, k = 0.5, 0
    for k in range(1, min(k_max, z.size) + 1):
        p = aggregate(z[:k], agg)
        if max(p, 1 - p) >= p_stop:
            return {'p_right': p, 'n_used': k, 'stopped': 'confident', 'answered': True,
                    'decided_ms': float(att['decided_ms'].iloc[k - 1])}
    if k == 0:
        return {'p_right': 0.5, 'n_used': 0, 'stopped': 'no_attempt', 'answered': False, 'decided_ms': math.nan}
    answered = max(p, 1 - p) >= p_answer
    return {'p_right': p, 'n_used': k, 'stopped': 'k_max' if k >= k_max else 'stream_end', 'answered': bool(answered),
            'decided_ms': float(att['decided_ms'].iloc[k - 1])}


def k_curve_rows(att: pd.DataFrame, scales: Mapping[str, M.AggregateScale | None], ks: Sequence[int] = (1, 2, 3, 4, 5)
                 ) -> list[dict[str, Any]]:
    """P(right) from the first k attempts (decision order) for every k: the mean logit, the sum, and one
    column ``p_<name>`` per fitted scale in ``scales`` (e.g. scale_recordings, scale_sessions)."""
    att = att.sort_values('decided_ms') if len(att) else att
    z = att['z'].to_numpy(float) if len(att) else np.zeros(0)
    rows = []
    for k in ks:
        n = min(k, z.size)
        row = {'k': k, 'n_available': int(n), 'p_mean': aggregate(z[:n], None, 'mean'),
               'p_sum': aggregate(z[:n], None, 'sum')}
        for name, agg in scales.items():
            row[f'p_{name}'] = aggregate(z[:n], agg, 'scaled')
        rows.append(row)
    return rows
