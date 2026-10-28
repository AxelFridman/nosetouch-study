"""Lane m4, part 7: calibration of multi-attempt aggregates on cross-fitted SESSION attempts.

Problem (figure M4_06): the k-attempt scale s_k of ``m4_models.AggregateScale.fit`` is learned from k random
RECORDINGS of one person and hand, i.e. from each recording's chosen segment. A cue-free session is different:
its accepted attempts include false ones (returns and other movements, about one in five on the synthetic
sessions) and the attempts of one session share the person's bias, so P = sigmoid(s_k * mean z) learned on
recordings is over-confident there (the stopping rule's answer is what the user sees).

Remedy (added after M4_06 showed the problem on the held-out sessions, so an exploratory choice, compared with
the recording-fitted scale and the plain mean logit under a declared rule, m4_config.AGGREGATION_RULE): fit s_k
on the attempts the production pipeline actually accepts in synthetic sessions of the TRAINING persons,
cross-fitted like the calibrator. For each inner split (``cv.inner_splits`` over the training persons,
the same splits that produced the calibrator's logits) the inner pipeline (segment classifier and hand model
fitted without the inner validation persons; the outer calibrator) runs on the validation persons' sessions
(recordings of one person and hand, ``m4_sessions.build_sessions``, the instructed stillness at each junction);
s_k = s_1 * k^b is then fitted by maximum likelihood on the mean calibrated logit of each session's first k
attempts, every k at once (``AggregateScale.fit_sessions``; one free scale per k was non-monotone in k because
few sessions reach 4-5 attempts, so the per-k fit is only kept for the record). The held-out person never
enters. Session segment tables are label-free functions of the stream, so they are cached across folds
(:data:`_TABLES`).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import m4_config as CFG
from . import m4_detect as D
from . import m4_e2e as E
from . import m4_models as M
from . import m4_sessions as S

#: stillness at each junction of the training sessions: the instructed stillness of the inference page
SESSION_KEEP_S = D.INSTRUCTED_STILLNESS_S
#: recordings per training session and the stride between sessions: the evaluated sessions' (m4_config)
SESSION_RECORDINGS = CFG.SESSION_RECORDINGS
SESSION_STRIDE = CFG.SESSION_STRIDE

_TABLES: dict[tuple, tuple[pd.DataFrame, np.ndarray]] = {}


def sessions_with_tables(trials: pd.DataFrame, rows_global: np.ndarray, get_stream: Callable[[int], Any], person: str,
                         seg_p: D.SegParams, keep_s: float, n_attempts: int = SESSION_RECORDINGS,
                         stride: int = SESSION_STRIDE, source: str = '') -> list[tuple[S.Session, pd.DataFrame, np.ndarray]]:
    """The synthetic sessions of one person built from ``trials`` (row i is global stream ``rows_global[i]``),
    each with its candidate-segment table and windows (cached by segmentation, stillness, recordings and
    ``source``, the identity of the streams ``get_stream`` returns: e.g. a Prepared fingerprint)."""
    trials = trials.reset_index(drop=True)
    rows_global = np.asarray(rows_global).astype(int)
    sess = S.build_sessions(trials, lambda i: get_stream(int(rows_global[i])), n_attempts=n_attempts, keep_s=keep_s,
                            persons=[person], stride=stride)
    out = []
    for ss in sess:
        key = (source, seg_p, float(keep_s), tuple(int(rows_global[r]) for r in ss.events['row'].to_numpy(int)))
        if key not in _TABLES:
            _TABLES[key] = D.segment_table(ss.stream, seg_p)
        tab, W = _TABLES[key]
        out.append((ss, tab, W))
    return out


def crossfit_session_attempts(prep: E.Prepared, fp: E.FittedPipeline, idx: np.ndarray, y: np.ndarray,
                              groups: np.ndarray, meta: pd.DataFrame, seed: int, *, keep_s: float = SESSION_KEEP_S,
                              n_attempts: int = SESSION_RECORDINGS, stride: int = SESSION_STRIDE) -> pd.DataFrame:
    """Accepted attempts (the production attempt rule, ``fp.spec.pair_returns``) of the training persons'
    synthetic sessions, each person scored by the inner pipeline fitted without it (module docstring). One row per
    attempt: person_id, y, session, order, z (calibrated logit), p_approach; sessions without an accepted attempt
    add no row."""
    idx = np.asarray(idx).astype(int)
    groups = np.asarray(groups).astype(str)
    meta = meta.reset_index(drop=True)
    seg_p = E.SEG_GRID[fp.seg_cfg]
    rows = []
    for _tr, va, inner in E.inner_pipelines(prep, fp, idx, y, groups, meta, seed):
        for person in sorted(set(groups[va].tolist())):
            pos = np.flatnonzero(groups == person)
            for si, (ss, tab, W) in enumerate(sessions_with_tables(meta.iloc[pos], idx[pos], prep.streams.__getitem__,
                                                                   person, seg_p, keep_s, n_attempts, stride,
                                                                   source=prep.fingerprint())):
                att = inner.attempts(tab, W)
                if not len(att):
                    continue
                att = att.sort_values('decided_ms').reset_index(drop=True)
                for j, r in att.iterrows():
                    rows.append({'person_id': person, 'y': int(ss.y), 'session': si, 'order': int(j),
                                 'z': float(r['z']), 'p_approach': float(r['p_approach'])})
    return pd.DataFrame(rows, columns=['person_id', 'y', 'session', 'order', 'z', 'p_approach'])


def fit_session_scale(att: pd.DataFrame, ks: tuple[int, ...] = tuple(range(1, CFG.K_MAX + 1)), form: str = 'power'
                      ) -> M.AggregateScale:
    """AggregateScale.fit_sessions on cross-fitted session attempts (one sequence per person and session);
    form 'power' (s_k = s_1 k^b, the production form) or 'per_k' (one scale per k, kept for the record)."""
    seqs, ys = [], []
    for (_p, _s), g in att.groupby(['person_id', 'session'], sort=True):
        g = g.sort_values('order')
        seqs.append(g['z'].to_numpy(float))
        ys.append(int(g['y'].iloc[0]))
    return M.AggregateScale(ks).fit_sessions(seqs, ys, form=form)


def session_aggregate(prep: E.Prepared, fp: E.FittedPipeline, idx: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      meta: pd.DataFrame, seed: int, *, keep_s: float = SESSION_KEEP_S,
                      n_attempts: int = SESSION_RECORDINGS, stride: int = SESSION_STRIDE
                      ) -> tuple[M.AggregateScale, pd.DataFrame]:
    """(session-fitted AggregateScale, the cross-fitted attempts it was fitted on) for one fitted pipeline."""
    att = crossfit_session_attempts(prep, fp, idx, y, groups, meta, seed, keep_s=keep_s, n_attempts=n_attempts,
                                    stride=stride)
    return fit_session_scale(att), att


def session_aggregate_for_person(prep: E.Prepared, spec: E.PipelineSpec, meta: pd.DataFrame, person: str, **kw: Any
                                 ) -> tuple[E.FittedPipeline, M.AggregateScale, pd.DataFrame]:
    """The outer fold that holds ``person`` out: its fitted pipeline and its session-fitted aggregate scale."""
    tr, y, g, m, seed = E.fold_inputs(meta, person)
    fp = E.get_pipeline(prep, spec, tr, y, g, m, seed)
    agg, att = session_aggregate(prep, fp, tr, y, g, m, seed, **kw)
    return fp, agg, att
