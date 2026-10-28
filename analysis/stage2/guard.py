"""Leakage guard shared by the data layer and the LOSO harness.

While ``cv.run_experiment`` runs a lane's ``fit_predict``, :func:`fit_predict_scope` is active and the cache
loaders (``data.load_cache``, ``data.load_tables``, ``data.canonical_trial_map``) refuse to run: they would
hand the labels of the held-out person to the code that must predict them. Load data before calling
``run_experiment`` and pass it in ``X`` (or close over feature arrays that hold no label).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_DEPTH = 0


class LeakageError(AssertionError):
    """A person in both parts of a fold, a label within reach of fit_predict, or predictions that depend on
    the composition of the test set."""


@contextmanager
def fit_predict_scope() -> Iterator[None]:
    """Mark the code that runs inside a lane's fit_predict (nesting allowed)."""
    global _DEPTH
    _DEPTH += 1
    try:
        yield
    finally:
        _DEPTH -= 1


def inside_fit_predict() -> bool:
    return _DEPTH > 0


def assert_outside_fit_predict(what: str) -> None:
    if _DEPTH > 0:
        raise LeakageError(f'{what} was called inside fit_predict: the cache holds the labels of the held-out '
                           'person. Load the data before run_experiment and pass features through X.')
