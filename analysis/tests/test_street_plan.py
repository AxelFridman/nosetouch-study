"""The 'street' plan of protocol 2.0.0: synth.build_street_plan is an exact port of app/src/core/protocol.ts.

Ground truth: fixtures/street_plans_ts.json.gz, written by the app's own code
(cd app && npx tsx ../analysis/tests/fixtures/make_street_plans.mts; rerun it whenever protocol.ts or util.ts
change). Same seed -> same blocks, tiers, cells and trial order, so a stored protocol.seed rebuilds the plan."""
from __future__ import annotations

import gzip
import json
from collections import Counter

import pytest
from _mlhelpers import FIXTURES

from motionlab import conventions as C
from motionlab import synth

TIERS = [0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4]


@pytest.fixture(scope='module')
def ts_plans() -> dict:
    return json.loads(gzip.decompress((FIXTURES / 'street_plans_ts.json.gz').read_bytes()))


def test_rng_ports_are_bit_exact():
    """util.ts mixSeed / makeRng (mulberry32) / shuffle, values printed by the app's code."""
    pins = {(0, synth.PLAN_SALT): (3358841099, [0.30136527260765433, 0.7210628378670663, 0.4674570234492421]),
            (12345, synth.PLAN_SALT): (1885379954, [0.5887745528016239, 0.8361853940878063, 0.7967650464270264]),
            (4294967295, synth.PLAN_SALT): (1443240566, [0.9097785912454128, 0.2433920030016452, 0.6415150314569473]),
            (2147483653, 7): (1053594887, [0.1620208031963557, 0.6223921517375857, 0.5729880221188068])}
    for (seed, salt), (mix, draws) in pins.items():
        assert synth.mix_seed(seed, salt) == mix
        rng = synth.mulberry32(mix)
        assert [rng(), rng(), rng()] == draws
    assert synth.js_shuffle(range(9), synth.mulberry32(99)) == [1, 8, 5, 7, 0, 4, 3, 6, 2]


def test_plans_match_the_app(ts_plans):
    assert ts_plans['protocol_version'] == C.APP_PROTOCOL_VERSION == synth.PROTOCOL_VERSION == '2.0.0'
    assert len(ts_plans['plans']) == 8
    for case in ts_plans['plans']:
        py = synth.build_street_plan(case['seed'], case['posture'], case['first_hand'])
        assert py == case['plan'], (case['seed'], case['first_hand'])
        assert synth.estimate_minutes(py) == case['minutes'] == 5


@pytest.mark.parametrize('seed', range(0, 200, 7))
def test_street_plan_structure(seed):
    plan = synth.build_street_plan(seed)
    trials = [t for b in plan for t in b['trials']]
    assert [b['tier'] for b in plan] == TIERS and [b['optional'] for b in plan] == [t == 4 for t in TIERS]
    assert [b['block_index'] for b in plan] == list(range(13))
    assert [t['trial_index'] for t in trials] == list(range(39))
    assert sum(len(b['trials']) for b in plan if not b['optional']) == 29
    assert [len(b['trials']) for b in plan] == [1, 4, 4, 3, 3, 3, 3, 4, 4, 3, 3, 2, 2]
    a = 'left' if seed & 1 else 'right'
    b = 'right' if a == 'left' else 'left'
    s1, s2 = ('top_right_toward', 'top_left_toward') if (seed >> 1) & 1 else ('top_left_toward', 'top_right_toward')
    assert [(x['hand'], x['orientation']) for x in plan] == [
        (a, 'free'), (a, 'upright_toward'), (b, 'upright_toward'), (b, s1), (a, s1), (a, s2), (b, s2),
        (b, 'upright_toward'), (a, 'upright_toward'), (a, 'upright_toward'), (b, 'upright_toward'),
        (b, 'upright_toward'), (a, 'upright_toward')]
    assert [x['kind'] for x in plan] == ['choice', 'main', 'main', 'main', 'main', 'main', 'main', 'odd', 'odd',
                                         'main', 'main', 'odd', 'odd']
    # hands balanced at every tier boundary
    for tier in (1, 2, 3, 4):
        c = Counter(t['condition']['hand'] for x in plan if x['tier'] == tier for t in x['trials'])
        assert c['left'] == c['right'], (seed, tier)
    # conditions: no screen-away grip, no chin / forehead; the choice trial is the only reported one
    assert {t['condition']['orientation'] for t in trials} <= set(C.COLLECTED_ORIENTATIONS)
    assert not {t['condition']['variant'] for t in trials} & {'chin', 'forehead'}
    assert [t['condition']['hand_source'] for t in trials] == ['reported'] + ['instructed'] * 38
    speeds = Counter(t['condition']['speed'] for x in plan if x['tier'] in (1, 2) for t in x['trials'])
    assert speeds == {'natural': 12, 'fast': 6, 'slow': 2}
    cells = lambda x: sorted((t['condition']['speed'], t['condition']['start_position'], t['condition']['variant'])  # noqa: E731
                             for t in x['trials'])
    tier3 = cells(plan[7])
    assert cells(plan[8]) == tier3 and synth.STREET_ALWAYS_CELL in tier3 and len(set(tier3)) == 4
    assert set(tier3) - {synth.STREET_ALWAYS_CELL} <= set(synth.STREET_VARIETY_POOL)
    bonus = cells(plan[11])
    assert cells(plan[12]) == bonus and not set(bonus) & set(tier3) and set(bonus) <= set(synth.STREET_VARIETY_POOL)
    # rep counts repeats of an identical cell inside its block, in presentation order
    for x in plan:
        seen: Counter = Counter()
        for t in x['trials']:
            c = t['condition']
            key = (c['speed'], c['start_position'], c['variant'])
            assert c['rep'] == seen[key]
            seen[key] += 1
    assert synth.estimate_minutes(plan) == 5.0


def test_first_hand_override_only_swaps_the_hands():
    for seed in (0, 1, 2, 3, 12345):
        base = synth.build_street_plan(seed)
        other = 'left' if base[0]['hand'] == 'right' else 'right'
        swapped = synth.build_street_plan(seed, first_hand=other)
        swap = {'left': 'right', 'right': 'left'}
        for b0, b1 in zip(base, swapped):
            assert b1['hand'] == swap[b0['hand']] and b1['orientation'] == b0['orientation']
            assert [t['condition']['speed'] for t in b1['trials']] == [t['condition']['speed'] for t in b0['trials']]
        sitting = synth.build_street_plan(seed, 'sitting')
        assert {t['condition']['posture'] for b in sitting for t in b['trials']} == {'sitting'}


def test_a_stored_street_session_rebuilds_its_plan(street_dataset):
    for s in street_dataset.sessions:
        p = s['protocol']
        assert p['preset'] == 'street' and p['custom'] is None
        assert synth.build_street_plan(p['seed'], p['posture']) == p['plan']
        assert p['first_hand'] == p['plan'][0]['hand'] and p['n_planned_trials'] == 39 and p['estimated_minutes'] == 5.0
