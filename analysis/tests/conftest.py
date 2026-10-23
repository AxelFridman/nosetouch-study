"""Shared fixtures of the motionlab tests (run: python -m pytest analysis/tests -q, from the repo root or analysis/)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from _mlhelpers import FIXTURES  # also puts analysis/ on sys.path

from motionlab import synth  # noqa: E402


@pytest.fixture(scope='session')
def ts_reference() -> dict:
    """Ground truth produced by the app's own TypeScript code (see fixtures/make_ts_reference.mts)."""
    return json.loads(gzip.decompress((FIXTURES / 'ts_online_v1_reference.json.gz').read_bytes()))


@pytest.fixture(scope='session')
def symmetric_participant() -> synth.SynthParticipant:
    """A participant without dominance asymmetry: left trials are exact mirrors of right trials."""
    return synth.random_participant(np.random.default_rng(11), nondominant_slowdown=1.0, nondominant_jitter=1.0)


@pytest.fixture(scope='session')
def tiny_dataset() -> synth.SynthDataset:
    """6 participants (android / ios alternating), 2 orientations x 2 speeds x both hands, a few redos."""
    return synth.make_dataset(6, seed=7, preset='tiny', failure_rate=0.08)


@pytest.fixture(scope='session')
def tiny_tree(tiny_dataset: synth.SynthDataset, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('tree')
    synth.write_folder_tree(tiny_dataset, root)
    return root
