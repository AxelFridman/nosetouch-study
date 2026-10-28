"""Lane final: JSON helpers of the derived files. JSON has no NaN, so the compute step writes a missing number as null
(:func:`clean`) and the presentation steps read null back as NaN (:func:`denull`); strings, booleans and structure are
untouched. Light on purpose (numpy only): the figures, tables and numbers steps import it."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def clean(obj: Any) -> Any:
    """A JSON-safe copy: NaN / inf (Python or numpy floats) -> None, numpy scalars -> Python scalars, tuples -> lists."""
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def denull(obj: Any) -> Any:
    """The inverse of :func:`clean` for numbers: None inside a dict or list -> NaN (so ``np.isfinite`` and the
    formatting helpers treat a missing number as missing)."""
    if isinstance(obj, dict):
        return {k: (math.nan if v is None else denull(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [math.nan if v is None else denull(v) for v in obj]
    return obj


__all__ = ['clean', 'denull']
