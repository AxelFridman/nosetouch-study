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


@pytest.fixture(scope='session')
def street_dataset() -> synth.SynthDataset:
    """4 protocol 2.0.0 'street' visits (schema 1.2, implicit_v1 calibration, short profile, auto-accept):
    android (Chrome's 0.1 grid) / ios alternating; a few failures, redos, one early finish and "not sure"
    answers to the hand question (seed 5: statuses completed x3 + ended_early; 2 choice trials unsure)."""
    return synth.make_dataset(4, seed=5, preset='street', failure_rate=0.05, ended_early_fraction=0.25,
                              redo_rate=0.05, choice_unsure_fraction=0.25,
                              platform_overrides={'android': {'quantize': True}})


@pytest.fixture(scope='session')
def street_tree(street_dataset: synth.SynthDataset, tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('street')
    synth.write_folder_tree(street_dataset, root)
    return root


@pytest.fixture(scope='session')
def street_ds(street_tree: Path):
    """``street_tree`` loaded with the default options (norm_v2: closure check of every session). Read-only."""
    from motionlab import load_dataset
    return load_dataset(street_tree)
