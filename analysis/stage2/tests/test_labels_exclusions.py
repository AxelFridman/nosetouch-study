"""Label integrity (instructed vs reported hands) and the exclusion cascade on the frozen snapshot."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stage2 import config as C
from stage2.data import RULES, apply_exclusions


def test_primary_labels_are_instructed(tables, expected):
    t = tables['trials']
    prim = t[t['analysis_set'] == 'primary']
    assert len(prim) == expected['primary']['trials'] and prim['person_id'].nunique() == expected['primary']['persons']
    assert (prim['hand'] == 'left').sum() == expected['primary']['left']
    assert (prim['hand_source'] == 'instructed').all()
    assert prim['hand'].isin(['left', 'right']).all()
    assert (prim['y_right'] == (prim['hand'] == 'right').astype(int)).all()
    assert (prim['status'] == 'complete').all() and prim['accepted'].astype(bool).all()
    assert prim['participant_flag'].isna().all()
    assert not prim['orientation'].astype(str).str.endswith('_away').any()
    assert prim['session_included'].all()


def test_natural_use_set_is_the_reported_choice_trials(tables, expected):
    t = tables['trials']
    both = t[t['analysis_set'].isin(['natural_use', 'natural_use_repeat'])]
    e = expected['natural_use']
    assert len(both) == e['with_repeats'] and both['person_id'].nunique() == e['persons']
    assert (both['hand_source'] == 'reported').all()
    assert (both['tier'] == 0).all() and (both['orientation'] == 'free').all()
    assert both['accepted'].astype(bool).all()
    assert not set(both['trial_id']) & set(t.loc[t['analysis_set'] == 'primary', 'trial_id'])


def test_natural_use_headline_is_one_first_trial_per_person(tables, expected):
    """Finding 12: one natural-use trial per person (the earliest that passed every rule); the free trials of
    repeat visits (P09's second, P14's third session) are reported apart."""
    t = tables['trials']
    nat = t[t['analysis_set'] == 'natural_use']
    rep = t[t['analysis_set'] == 'natural_use_repeat']
    assert len(nat) == expected['natural_use']['headline'] and nat['person_id'].is_unique
    assert sorted(rep['trial_key'].str.rsplit('-', n=1).str[0]) == expected['natural_use']['repeat_session_labels']
    assert (rep['excl_rule'] == 'N1').all()
    for _, r in rep.iterrows():  # the headline trial of that person came first
        first = nat[nat['person_id'] == r['person_id']].iloc[0]
        assert first['session_started_at'] < r['session_started_at']
    # P14's first free trial failed E4c (peak 29.6 deg/s), so the headline uses the second visit's
    p14 = t[(t['person_id'] == 'P14') & (t['hand_source'] == 'reported') & t['session_included']
            & (t['status'] == 'complete') & t['accepted'].astype(bool)]
    assert set(p14['analysis_set']) == {'excluded', 'natural_use', 'natural_use_repeat'}


def test_instructed_hand_matches_the_session_plan(tables):
    """Every primary label equals the hand of its planned block (read from the raw session records)."""
    t = tables['trials']
    prim = t[t['analysis_set'] == 'primary']
    plan = {}
    for f in C.SNAPSHOT_DIR.rglob('session.json'):
        s = json.loads(f.read_text(encoding='utf-8'))
        for b in s['protocol']['plan']:
            plan[(s['session_id'], b['block_index'])] = (b['hand'], b['kind'])
    if not plan:
        pytest.skip('snapshot not available')
    planned = [plan[(sid, int(bi))] for sid, bi in zip(prim['session_id'], prim['block_index'])]
    assert all(h == ph for h, (ph, _) in zip(prim['hand'], planned))
    assert all(kind != 'choice' for _, kind in planned)


def test_both_hands_per_person(primary):
    tr = primary.trials
    per = tr.groupby('person_id')['hand'].nunique()
    assert (per == 2).all()
    share = tr.groupby('person_id')['y_right'].mean()
    assert share.between(0.3, 0.7).all()


def test_flow_counts(tables, expected):
    f = tables['exclusion_flow'].set_index('step')
    got = {step: {c: int(f.loc[step, c]) for c in cols} for step, cols in expected['flow'].items()}
    assert got == expected['flow']
    assert list(f.index) == list(expected['flow'])                  # the same rules, in the same order
    # E4b/E4c are the bounds of docs/09 gate G2 (0.25 s, 30 deg/s); E3 diverts, never removes
    assert int(f.loc['E3', 'removed']) == 0 and int(f.loc['E3', 'natural_use_added']) > 0
    assert int(f['remaining'].iloc[-1]) == expected['primary']['trials']
    assert int(f['natural_use'].iloc[-1]) == expected['natural_use']['headline']
    rem = f['remaining'].to_numpy()
    assert np.all(np.diff(rem) <= 0)
    assert int(tables['exclusion_e1']['attempts'].sum()) == expected['e1_attempts'] == int(f.loc['E1', 'removed'])


def test_every_excluded_attempt_names_its_first_failing_rule(tables):
    t = tables['trials']
    ex = t[t['analysis_set'] == 'excluded']
    assert ex['excl_rule'].notna().all()
    order = [r.code for r in RULES]
    for code, g in ex.groupby('excl_rule'):
        earlier = [r for r in RULES[: order.index(code)] if r.divert is None]
        for r in earlier:  # an attempt charged to a rule passed every earlier one
            assert not r.fails(g).fillna(False).astype(bool).any(), (code, r.code)


def test_cascade_is_order_sensitive_and_first_rule_wins():
    d = pd.DataFrame({
        'session_included': [True, False, True, True], 'simulated': [False] * 4,
        'status': ['complete', 'complete', 'no_movement', 'complete'], 'accepted': [True, True, False, True],
        'participant_flag': [None] * 4, 'note': [None] * 4, 'orientation': ['upright_toward'] * 4,
        'hand_source': ['instructed', 'instructed', 'instructed', 'reported'], 'oracle_ok': [True] * 4,
        'duration_ms': [1500.0, 1500.0, np.nan, 200.0], 'peak_gyro_dps': [100.0] * 4, 'win_n_gaps_100': [0] * 4,
        'hidden_during': [False] * 4, 'motion_hz': [60.0] * 4, 'rotation_available': [True] * 4,
        'gravity_sign_source': ['engine_vote'] * 4, 'person_id': ['P01', 'P02', 'P01', 'P01'],
        'person_key': ['P01', 'P02', 'P01', 'P01'], 'session_id': ['a', 'b', 'a', 'a'],
    })
    out, flow, e1 = apply_exclusions(d)
    assert out['analysis_set'].tolist() == ['primary', 'excluded', 'excluded', 'excluded']
    assert out['excl_rule'].tolist()[1:] == ['E0', 'E1', 'E4b']  # the reported trial is diverted, then fails E4b
    assert int(flow.set_index('step').loc['E3', 'natural_use_added']) == 1
    assert int(flow['natural_use'].iloc[-1]) == 0


def test_flags_are_never_exclusions_and_gates_are_scored_apart(tables, expected):
    """Finding 3: the T8-T12 flags exist as columns (kept, never excluded) with their counts; finding 5: the
    docs/09 gates G1 / G2 are columns of the trial table, failing trials stay primary."""
    from stage2.data import FLAG_COLUMNS, GATE_COLUMNS
    t = tables['trials']
    prim = t[t['analysis_set'] == 'primary']
    counts = {c: int(prim[c].sum()) for c in FLAG_COLUMNS}
    assert counts == expected['flags_primary']
    assert sorted(set(prim.loc[prim['grip_mismatch'], 'person_id'])) == expected['grip_mismatch_persons']
    assert prim.loc[prim['grip_mismatch'], 'grip_mismatch_app'].all()
    # P16 blocks 5-6 were instructed top_right but held top_left (gravity +x before the onset)
    p16 = prim[(prim['person_id'] == 'P16') & prim['block_index'].isin([5, 6])]
    assert (p16['grip_detected'] == 'top_left').all() and (p16['instructed_rotation'] == 'top_right').all()
    assert (prim['seg_disagree_100ms'] == (prim['seg_onset_diff_ms'].abs() > 100)).all()
    fail = {c: int((~prim[c]).sum()) for c in (*GATE_COLUMNS, 'gate_pass')}
    assert fail == expected['gates_failed_primary']
    assert (prim.loc[~prim['gate_g1'], 'platform'] == 'android').all()
    assert (prim.loc[~prim['gate_g2'], 'duration_ms'] > 8000).all()
    assert prim.loc[~prim['gate_g1'], 'g1_gap_max_ms'].between(50, 100).all()  # interpolated, so not E4d


def test_grip_from_up():
    from stage2.data import grip_from_up
    assert grip_from_up([0.05, 0.99, 0.1]) == 'upright'
    assert grip_from_up([0.95, 0.1, 0.2], 'top_left_toward') == 'top_left'
    assert grip_from_up([-0.95, 0.1, 0.2], 'top_right_toward') == 'top_right'
    assert grip_from_up([-0.95, 0.1, 0.2], 'top_left_away') == 'top_left'   # screen away mirrors x
    assert grip_from_up([0.6, 0.6, 0.5]) == 'unknown' and grip_from_up([np.nan, 1, 0]) == 'unknown'
