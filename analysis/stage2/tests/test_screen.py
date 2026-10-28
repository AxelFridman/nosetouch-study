"""Finding 4: the docs/09 D6 wrong-hand screen and the per-block label-consistency table (labels never changed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stage2 import screen
from stage2.standardize import CHANNELS


def synthetic(n_persons: int = 6, blocks: int = 6, per_block: int = 4, swap: tuple[str, int, int] | None = None,
              seed: int = 0):
    """Approach arrays whose chiral channels carry the hand (right: + yaw early), labels from an alternating plan;
    ``swap`` = (person, block a, block b): the participant used the other hand in those two blocks."""
    rng = np.random.default_rng(seed)
    rows, A = [], []
    t = np.linspace(0, 1, 64)
    for p in range(n_persons):
        pid = f'P{p + 1:02d}'
        for b in range(1, blocks + 1):
            hand = 'left' if b % 2 else 'right'
            for k in range(per_block):
                used = hand
                if swap and pid == swap[0] and b in swap[1:]:
                    used = 'right' if hand == 'left' else 'left'
                s = 1.0 if used == 'right' else -1.0
                X = rng.normal(0, 0.3, (len(CHANNELS), 64))
                X[CHANNELS.index('yaw_rate')] += s * 30 * np.sin(np.pi * t) * (t < 0.6) + rng.normal(0, 5)
                X[CHANNELS.index('yaw_cum')] = np.cumsum(X[CHANNELS.index('yaw_rate')]) / 60
                A.append(X)
                rows.append({'trial_id': f'{pid}-{b}-{k}', 'person_id': pid, 'session_id': f's{pid}',
                             'session_label': f'{pid}-s1', 'block_index': b, 'trial_index': (b - 1) * per_block + k,
                             'attempt': 1, 'tier': 1, 'hand': hand, 'y_right': int(hand == 'right'),
                             'orientation': 'upright_toward', 'grip_mismatch': False,
                             'first_half_yaw_deg': float(X[CHANNELS.index('yaw_cum'), 31])})
    return pd.DataFrame(rows), np.stack(A).astype(np.float32)


def test_swap_signature_marks_only_the_swapped_pair():
    trials, A = synthetic(swap=('P03', 1, 2))
    _, b = screen.block_consistency(trials, A)
    for ev in ('fhy', 'screen'):
        hit = b[b[f'{ev}_swap_signature']]
        assert set(zip(hit['person_id'], hit['block_index'])) == {('P03', 1), ('P03', 2)}, ev
    _, clean = screen.block_consistency(*synthetic())
    assert not clean['fhy_swap_signature'].any() and not clean['screen_swap_signature'].any()


def test_a_one_hand_reversal_is_no_swap_signature():
    """A dominance-type effect reverses one hand only (blocks of the same hand are never adjacent)."""
    trials, A = synthetic(swap=('P02', 3, 5))
    _, b = screen.block_consistency(trials, A)
    assert not b['fhy_swap_signature'].any()
    assert b.loc[(b['person_id'] == 'P02') & b['block_index'].isin([3, 5]), 'fhy_all_contradict'].all()


def test_d6_screen_flags_confident_runs_on_training_rows():
    trials, A = synthetic(swap=('P03', 1, 2))
    F = screen.screen_features(A)
    fl = screen.wrong_hand_flags(F, trials['y_right'].to_numpy(), trials['person_id'].to_numpy(), trials)
    flagged = trials.loc[fl['flag'].to_numpy(), ['person_id', 'block_index']]
    assert len(flagged) and set(flagged['person_id']) == {'P03'} and set(flagged['block_index']) <= {1, 2}
    assert (fl['p_label'] >= 0).all() and (fl['p_label'] <= 1).all()
    # antisymmetric: the mirrored features (chiral channels flip) give exactly 1 - p
    m, s = screen.antisymmetric_logreg(F, trials['y_right'].to_numpy())
    assert np.allclose(m.predict_proba(-F / s)[:, 1], 1 - m.predict_proba(F / s)[:, 1])


def test_swap_signature_on_the_real_data(primary, expected):
    """The verifier's P08 blocks 1-2: every trial contradicts its label by the first-half yaw sign, but the
    richer screen model agrees with 2 of the 8 labels and the D6 rules flag none of them."""
    t, b = screen.block_consistency(primary.trials, primary.approach)
    hit = b[b['fhy_swap_signature']]
    assert sorted([s, int(b)] for s, b in zip(hit['session_label'], hit['block_index'])) == \
        expected['swap_signature_blocks']
    assert {('P08-s1', 1), ('P08-s1', 2)} <= set(zip(hit['session_label'], hit['block_index']))
    p08 = t[(t['person_id'] == 'P08') & t['block_index'].isin([1, 2])]
    assert len(p08) == 8 and not p08['fhy_agrees'].any() and not p08['screen_flag'].any()
    assert not b['screen_swap_signature'].any()
    # labels are untouched: the plan's hand is still the label
    assert (p08['y_right'] == (p08['hand'] == 'right').astype(int)).all()
