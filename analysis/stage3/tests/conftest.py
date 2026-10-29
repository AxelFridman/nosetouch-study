"""Fixtures of the stage3 tests (run: cd analysis && python -m pytest stage3/tests -q -p no:cacheprovider).

The tests on the committed synthetic fixtures and the exported JSON need no participant data. The tests that compare
with the in-memory fits need the Stage 2 cache (they refit or load ``analysis/data/stage3_cache``) and the ones on real
recordings need ``analysis/data/stage3_fixtures``: both are skipped when the files are missing (CI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[2]
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from stage2 import config as C  # noqa: E402


def have_cache() -> bool:
    return (C.CACHE_DIR / 'arrays.npz').exists() and (C.CACHE_DIR / 'trials.pkl').exists()


@pytest.fixture(scope='session')
def json_models():
    from stage3 import export as EX
    from stage3 import runtime as RT
    if not all((EX.APP_MODELS_DIR / f).exists() for f in RT.FILES.values()):
        pytest.skip('no exported models (python -m stage3 export)')
    return RT.json_set(EX.APP_MODELS_DIR)


@pytest.fixture(scope='session')
def fitted_models():
    if not have_cache():
        pytest.skip('no Stage 2 cache: the in-memory fits need the participant data')
    from stage3 import runtime as RT
    return RT.fitted_set()
