"""The pinned snapshot counts (expected_counts.json) are exactly what the cache holds."""
from __future__ import annotations

from stage2.scripts.expected_counts import diff, load_fixture, snapshot_counts


def test_fixture_matches_the_cache(primary):
    d = diff(snapshot_counts(), load_fixture())
    assert not d, 'the cache differs from tests/expected_counts.json (see its _note to update it):\n' + '\n'.join(d)


def test_fixture_is_internally_consistent(expected):
    f = expected['flow']
    assert list(f)[0] == 'start' and f[list(f)[-1]]['remaining'] == expected['primary']['trials']
    assert expected['primary']['left'] + expected['primary']['right'] == expected['primary']['trials']
    assert expected['curation']['included'] + expected['curation']['excluded'] == expected['curation']['sessions']
    assert expected['natural_use']['headline'] + len(expected['natural_use']['repeat_session_labels']) == \
        expected['natural_use']['with_repeats']
