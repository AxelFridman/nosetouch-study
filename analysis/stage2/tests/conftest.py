"""Fixtures of the stage2 tests (run: cd analysis && python -m pytest stage2/tests -q).

Tests that need the real data use the cache (analysis/data/stage2_cache, built by
``python -m stage2.scripts.build_cache``) and are skipped when it is missing; the rest run on synthetic
records from motionlab.synth.

Snapshot counts (sessions, persons, trials, the exclusion flow, flags, gates) are pinned in ONE file,
``expected_counts.json`` (fixture ``expected``); its ``_note`` says how to update it when a participant is added
(``python -m stage2.scripts.expected_counts --without <new person>``, then ``--write``).

Tests marked ``exploratory`` read the exploratory persons' files (analysis/data/exploratory*, the exploratory cache,
curation/exploratory.json), which only lane x1 and the orchestrator may read. They are skipped unless explicitly
selected, so any lane can run the whole suite; the orchestrator (or lane x1) runs them with::

    .venv/Scripts/python.exe -m pytest stage2/tests -m exploratory -q -p no:cacheprovider
    # or, with the rest of the suite:  STAGE2_EXPLORATORY_TESTS=1 python -m pytest tests stage2/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[2]
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from stage2 import config as C  # noqa: E402


EXPECTED_COUNTS = Path(__file__).resolve().parent / 'expected_counts.json'
EXPLORATORY_ENV = 'STAGE2_EXPLORATORY_TESTS'


def pytest_configure(config):
    config.addinivalue_line('markers', 'exploratory: reads the exploratory persons\' files; runs only when selected '
                                       f'(-m exploratory, or {EXPLORATORY_ENV}=1)')


def exploratory_selected(markexpr: str | None, env: str | None) -> bool:
    """Run the exploratory tests? Only when asked: the -m expression names the marker (and does not negate it), or
    the environment variable is set to something other than '' / '0'."""
    if (env or '').strip() not in ('', '0'):
        return True
    expr = (markexpr or '').replace('(', ' ').replace(')', ' ').split()
    return 'exploratory' in expr and not any(a == 'not' and b == 'exploratory' for a, b in zip(expr, expr[1:]))


def pytest_collection_modifyitems(config, items):
    if exploratory_selected(config.getoption('markexpr', ''), os.environ.get(EXPLORATORY_ENV)):
        return
    skip = pytest.mark.skip(reason='reads the exploratory persons\' files (only lane x1 and the orchestrator may): '
                                   f'select it with -m exploratory or {EXPLORATORY_ENV}=1')
    for item in items:
        if item.get_closest_marker('exploratory') is not None:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def expected() -> dict:
    """The pinned snapshot counts (expected_counts.json)."""
    return json.loads(EXPECTED_COUNTS.read_text(encoding='utf-8'))


def _have_cache() -> bool:
    return (C.CACHE_DIR / 'arrays.npz').exists() and (C.CACHE_DIR / 'trials.pkl').exists()


@pytest.fixture(scope='session')
def tables():
    if not _have_cache():
        pytest.skip('no Stage 2 cache (python -m stage2.scripts.build_cache)')
    from stage2.data import load_tables
    return load_tables()


@pytest.fixture(scope='session')
def primary():
    if not _have_cache():
        pytest.skip('no Stage 2 cache (python -m stage2.scripts.build_cache)')
    from stage2.data import load_cache
    return load_cache(('primary',))


@pytest.fixture(scope='session')
def natural():
    if not _have_cache():
        pytest.skip('no Stage 2 cache (python -m stage2.scripts.build_cache)')
    from stage2.data import load_cache
    return load_cache(('natural_use',))


@pytest.fixture(scope='session')
def street_records(tmp_path_factory):
    """Complete trials of 2 synthetic street visits (one Android with Chrome's rounding, one iOS) with the
    session normalisation, loaded through motionlab.io like the real data."""
    from motionlab import load_dataset, synth
    ds_s = synth.make_dataset(2, seed=3, preset='street', failure_rate=0.0,
                              platform_overrides={'android': {'quantize': True}})
    root = tmp_path_factory.mktemp('street2')
    synth.write_folder_tree(ds_s, root)
    ds = load_dataset(root)
    done = ds.trials[(ds.trials['status'] == 'complete') & ds.trials['onset_ms'].notna()]
    out = []
    for tid in done['trial_id'].head(12):
        out.append((ds.record(tid, compact=True), ds.trial_normalization(tid)))
    return out
