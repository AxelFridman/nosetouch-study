"""The researcher's curation is applied exactly (counts pinned in expected_counts.json), grouping by person."""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from stage2 import config as C
from stage2.data import EXCLUDE_CODES, load_curation


@pytest.fixture(scope='module')
def cur() -> pd.DataFrame:
    if not C.CURATION_FILE.exists():
        pytest.skip('no curation file')
    return load_curation()


def test_counts(cur, expected):
    e = expected['curation']
    assert len(cur) == e['sessions']
    assert int(cur['include'].sum()) == e['included']
    assert cur.loc[cur['include'], 'person_id'].nunique() == e['persons']
    assert sorted(cur.loc[cur['include'], 'person_id'].unique()) == [f'P{i:02d}' for i in range(1, e['persons'] + 1)]
    assert int((~cur['include']).sum()) == e['excluded']


def test_multi_session_persons(cur, expected):
    n = cur[cur['include']].groupby('person_id').size()
    assert n.to_dict() == {**{p: 1 for p in n.index}, **expected['curation']['multi_session_persons']}


def test_excluded_reasons_are_codes_without_names(cur, expected):
    ex = cur[~cur['include']]
    assert set(ex['exclude_code']) <= set(EXCLUDE_CODES)
    for r in ex['exclude_reason']:
        assert r in EXCLUDE_CODES.values()
    # the two unassigned excluded sessions belong to one person (the same reason in the curation file)
    un = ex[ex['person_id'].isna()]
    assert len(un) == expected['curation']['unassigned_excluded_sessions'] and un['person_key'].nunique() == 1
    # a superseded first run stays attached to its person and is excluded; the person's later run is included
    sup = ex[ex['exclude_code'] == 'superseded']
    assert sorted(sup['person_id']) == expected['curation']['superseded_persons']
    for p in sup['person_id']:
        assert int(cur[cur['include'] & (cur['person_id'] == p)].shape[0]) >= 1


def test_curation_columns_never_carry_free_text(cur):
    assert set(cur.columns) == {'session_id', 'person_id', 'include', 'exclude_code', 'exclude_reason', 'person_key'}
    raw = json.loads(C.CURATION_FILE.read_text(encoding='utf-8'))
    free = [s.get('exclude_reason') for s in raw['sessions'] if s.get('exclude_reason')]
    for text in free:  # the curation file's own wording (it starts with a first name) never reaches the table
        assert text not in set(cur['exclude_reason'].dropna())


def test_private_file_is_refused():
    with pytest.raises(PermissionError):
        C.assert_not_private(C.CURATION_FILE.parent / C.PRIVATE_FILE_NAME)


def test_one_code_two_persons_and_one_person_two_codes(tables):
    s = tables['sessions']
    inc = s[s['include']]
    by_code = inc.groupby('participant_code')['person_id'].nunique()
    assert (by_code > 1).sum() == 1          # the researcher's phone: P01 and P13 under one code
    by_person = inc.groupby('person_id')['participant_code'].nunique()
    assert (by_person > 1).sum() == 1        # P14 used two codes
    assert by_person.idxmax() == 'P14'


def test_person_demographics_come_from_each_session(tables):
    """P01 and P13 share a P-code; their profiles must not be merged (motionlab keys profiles by code)."""
    p = tables['persons'].set_index('person_id')
    assert p.loc['P01', 'participant_codes'] == p.loc['P13', 'participant_codes']
    assert (p.loc['P01', 'age_years'], p.loc['P01', 'phone_hand']) != (p.loc['P13', 'age_years'], p.loc['P13', 'phone_hand'])
    assert p['profile_consistent'].all()


def test_no_identifying_columns(tables):
    bad = re.compile(r'name|nickname|user_agent', re.I)
    for k, df in tables.items():
        assert not [c for c in df.columns if bad.search(str(c))], k
    for c in ('age_years', 'sex', 'handedness_self', 'phone_hand', 'height_cm', 'dominant_hand'):
        assert c not in tables['trials'].columns  # demographics live in the person table only
