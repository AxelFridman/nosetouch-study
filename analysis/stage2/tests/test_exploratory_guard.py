"""Exploratory persons (analysis/data/exploratory, listed in curation/exploratory.json) never reach the main
analysis: none of their person ids, session ids or app codes appears in the main snapshot, the main curation or
the main cache. Only the ids are read from exploratory.json; no exploratory data is processed here.

The three tests that read exploratory.json are marked ``exploratory``: only lane x1 and the orchestrator may read
that file, so they run only when selected (``python -m pytest stage2/tests -m exploratory``, or
``STAGE2_EXPLORATORY_TESTS=1``; see conftest.py). Their assertions are unchanged. The checks that need no
exploratory file run in every suite."""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from stage2 import config as C

#: the curated person id of the main analysis (P01, P02, ...): an exploratory id (PD01, ...) never has this form
MAIN_PERSON_ID = re.compile(r'P\d+')


@pytest.fixture(scope='module')
def exploratory_ids() -> dict[str, set[str]]:
    if not C.EXPLORATORY_CURATION.exists():
        pytest.skip('no exploratory curation file')
    doc = json.loads(C.EXPLORATORY_CURATION.read_text(encoding='utf-8'))
    persons = doc.get('persons') or []
    sessions = [s for p in persons for s in (p.get('sessions') or [])]
    ids = {'person_id': {p['person_id'] for p in persons if p.get('person_id')},
           'session_id': {s['session_id'] for s in sessions if s.get('session_id')},
           'participant_code': {s['participant_code'] for s in sessions if s.get('participant_code')}}
    assert ids['person_id'] and ids['session_id'], 'exploratory.json lists no person or session'
    return ids


def test_exploratory_paths_are_apart():
    for p in (C.EXPLORATORY_DIR, C.EXPLORATORY_CACHE_DIR):
        assert C.SNAPSHOT_DIR.resolve() not in [p.resolve(), *p.resolve().parents]
        assert p.resolve() not in [C.SNAPSHOT_DIR.resolve(), *C.SNAPSHOT_DIR.resolve().parents]
    assert C.EXPLORATORY_CACHE_DIR.resolve() != C.CACHE_DIR.resolve()
    assert C.EXPLORATORY_CURATION.resolve() != C.CURATION_FILE.resolve()


def test_main_person_ids_have_the_main_form():
    """Without reading any exploratory file: every person of the main curation and cache is a main id (P01, ...)."""
    doc = json.loads(C.assert_not_private(C.CURATION_FILE).read_text(encoding='utf-8'))
    ids = {str(s['person_id']) for s in doc['sessions'] if s.get('person_id')}
    assert ids and all(MAIN_PERSON_ID.fullmatch(i) for i in ids), sorted(i for i in ids if not MAIN_PERSON_ID.fullmatch(i))
    if (C.CACHE_DIR / 'trials.pkl').exists():
        for name in ('trials', 'persons'):
            got = set(pd.read_pickle(C.CACHE_DIR / f'{name}.pkl')['person_id'].dropna().astype(str))
            assert all(MAIN_PERSON_ID.fullmatch(i) for i in got), name


@pytest.mark.exploratory
def test_exploratory_persons_are_not_in_the_snapshot(exploratory_ids):
    root = C.SNAPSHOT_DIR
    if not root.exists():
        pytest.skip('no snapshot')
    dirs = [p for p in root.rglob('*') if p.is_dir()]
    names = {p.name for p in dirs}
    assert not names & exploratory_ids['participant_code'], 'an exploratory app code is a snapshot folder'
    assert not names & exploratory_ids['session_id'], 'an exploratory session is a snapshot folder'
    for f in root.rglob('session.json'):
        s = json.loads(f.read_text(encoding='utf-8'))
        assert s.get('session_id') not in exploratory_ids['session_id'], f.parent.name
        assert (s.get('participant') or {}).get('participant_id') not in exploratory_ids['participant_code']


@pytest.mark.exploratory
def test_exploratory_persons_are_not_in_the_main_curation(exploratory_ids):
    doc = json.loads(C.CURATION_FILE.read_text(encoding='utf-8'))
    sessions = doc['sessions']
    assert not {s['session_id'] for s in sessions} & exploratory_ids['session_id']
    assert not {s.get('person_id') for s in sessions} & exploratory_ids['person_id']


@pytest.mark.exploratory
def test_exploratory_persons_are_not_in_the_main_cache(exploratory_ids):
    if not (C.CACHE_DIR / 'trials.pkl').exists():
        pytest.skip('no Stage 2 cache')
    for name in ('trials', 'sessions', 'persons'):
        df = pd.read_pickle(C.CACHE_DIR / f'{name}.pkl')
        for col, ids in exploratory_ids.items():
            for c in (col, f'{col}s'):  # persons.pkl lists codes and sessions as text (participant_codes, ...)
                if c in df.columns:
                    vals = set(df[c].dropna().astype(str))
                    joined = ' '.join(vals)
                    assert not vals & ids and not any(i in joined for i in ids), (name, c)
