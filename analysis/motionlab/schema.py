"""Validation against the shared data contract ``schema/records.schema.json`` (JSON Schema 2020-12).

The schema file is the single source of truth shared with the app (ajv) and the Apps Script backend; it
is read from the repository (``<repo>/schema/records.schema.json``), never copied.

Two validation modes:
  * ``'full'``  — jsonschema on the whole record. Exact, but pure-Python and slow on long streams
                  (each stream column holds thousands of numbers).
  * ``'fast'``  — jsonschema on a copy whose stream columns are truncated to their first elements
                  (structure, enums, required fields, nullability of the kept elements), plus vectorised
                  checks of every stream element (number / null where nullable). Equivalent to 'full' for
                  every constraint the schema expresses on streams; ~100x faster.
"""
from __future__ import annotations

import copy
import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH: Path = Path(__file__).resolve().parents[2] / 'schema' / 'records.schema.json'

#: Definitions that can be validated directly.
DEFS = (
    'SessionRecord', 'TrialRecord', 'ParticipantProfile', 'UploadEnvelope', 'ExportBundle',
    'CalibrationResult', 'PlannedBlock', 'SegmentationParams', 'MotionStream', 'OrientationStream',
    'QuaternionStream',
)

_STREAM_DEFS = {'motion': 'MotionStream', 'orientation': 'OrientationStream', 'quaternion': 'QuaternionStream'}
_KEEP = 2  # stream elements kept for the structural (jsonschema) pass in fast mode


class SchemaValidationError(ValueError):
    """Raised by :func:`check` with every error message of the record."""

    def __init__(self, def_name: str, errors: list[str]):
        self.def_name = def_name
        self.errors = errors
        head = '\n  '.join(errors[:10])
        more = f'\n  ... and {len(errors) - 10} more' if len(errors) > 10 else ''
        super().__init__(f'{def_name}: {len(errors)} schema error(s):\n  {head}{more}')


@functools.lru_cache(maxsize=4)
def load_schema(path: str | None = None) -> dict[str, Any]:
    """The parsed schema (cached). ``path`` overrides the repository location."""
    p = Path(path) if path else SCHEMA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f'records schema not found at {p}. motionlab expects to live in <repo>/analysis/motionlab next to '
            f'<repo>/schema/records.schema.json.'
        )
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def schema_version_pattern() -> str:
    return load_schema()['$defs']['SessionRecord']['properties']['schema_version']['pattern']


def enum_values(def_name: str) -> list[Any]:
    """Values of an enum definition, e.g. ``enum_values('OrientationId')``."""
    d = load_schema()['$defs'][def_name]
    if 'enum' not in d:
        raise KeyError(f'{def_name} is not an enum definition')
    return list(d['enum'])


@functools.lru_cache(maxsize=32)
def validator_for(def_name: str) -> Draft202012Validator:
    """A cached validator for one ``$defs`` entry (refs resolve inside the same document)."""
    schema = load_schema()
    if def_name not in schema['$defs']:
        raise KeyError(f'unknown schema definition {def_name!r}')
    root = dict(schema)
    root['$ref'] = f'#/$defs/{def_name}'
    Draft202012Validator.check_schema(root)
    return Draft202012Validator(root)


def detect_def(obj: Any) -> str | None:
    """Which definition a parsed JSON document claims to be (by its discriminator fields)."""
    if not isinstance(obj, Mapping):
        return None
    if obj.get('format') == 'nose-touch-export':
        return 'ExportBundle'
    rt = obj.get('record_type')
    if rt == 'trial':
        return 'TrialRecord'
    if rt == 'session':
        return 'SessionRecord'
    if obj.get('kind') in ('session', 'trial') and isinstance(obj.get('record'), Mapping):
        return 'UploadEnvelope'
    if 'participant_id' in obj and 'handedness_self' in obj:
        return 'ParticipantProfile'
    return None


def _fmt_path(path: Any) -> str:
    out = ''
    for p in path:
        out += f'[{p}]' if isinstance(p, int) else (f'.{p}' if out else str(p))
    return out or '(root)'


def _jsonschema_errors(def_name: str, obj: Any, max_errors: int) -> list[str]:
    errs = []
    for e in validator_for(def_name).iter_errors(obj):
        errs.append(f'{_fmt_path(e.absolute_path)}: {e.message[:300]}')
        if len(errs) >= max_errors:
            break
    return errs


@functools.lru_cache(maxsize=4)
def _stream_column_kinds(stream_def: str) -> dict[str, bool]:
    """column -> nullable, for the array-of-number columns of a stream definition."""
    props = load_schema()['$defs'][stream_def]['properties']
    out: dict[str, bool] = {}
    for k, v in props.items():
        ref = v.get('$ref', '')
        if ref.endswith('/NumArray'):
            out[k] = False
        elif ref.endswith('/NullableNumArray'):
            out[k] = True
    return out


_SCALAR_TYPES = {int, float, bool, str, type(None)}


def _check_column(values: Any, nullable: bool) -> str | None:
    if not isinstance(values, list):
        return None  # structural pass reports the type error
    allowed = {int, float, type(None)} if nullable else {int, float}
    types = set(map(type, values))
    if types <= allowed:
        return None
    for i, v in enumerate(values):
        if type(v) not in allowed:
            what = 'number or null' if nullable else 'number'
            return f'[{i}]: {v!r} is not a {what}'
    return None


def _truncate_trial_streams(trial: Any, prefix: str, problems: list[str]) -> Any:
    """Copy of a trial with long stream columns truncated; full columns are checked into ``problems``."""
    if not isinstance(trial, Mapping):
        return trial
    streams = trial.get('streams')
    if not isinstance(streams, Mapping):
        return trial
    out = dict(trial)
    s_out = dict(streams)
    for name, def_name in _STREAM_DEFS.items():
        st = streams.get(name)
        if not isinstance(st, Mapping):
            continue
        st_out = dict(st)
        for col, nullable in _stream_column_kinds(def_name).items():
            vals = st.get(col)
            if isinstance(vals, list) and len(vals) > _KEEP:
                msg = _check_column(vals, nullable)
                if msg:
                    problems.append(f'{prefix}streams.{name}.{col}{msg}')
                st_out[col] = vals[:_KEEP]
        s_out[name] = st_out
    out['streams'] = s_out
    return out


def _fast_copy(def_name: str, obj: Any, problems: list[str]) -> Any:
    if def_name == 'TrialRecord':
        return _truncate_trial_streams(obj, '', problems)
    if def_name == 'UploadEnvelope' and isinstance(obj, Mapping) and isinstance(obj.get('record'), Mapping):
        out = dict(obj)
        if obj['record'].get('record_type') == 'trial':
            out['record'] = _truncate_trial_streams(obj['record'], 'record.', problems)
        return out
    if def_name == 'ExportBundle' and isinstance(obj, Mapping) and isinstance(obj.get('trials'), list):
        out = dict(obj)
        out['trials'] = [_truncate_trial_streams(t, f'trials[{i}].', problems) for i, t in enumerate(obj['trials'])]
        return out
    return obj


def validate(obj: Any, def_name: str | None = None, *, mode: str = 'fast', max_errors: int = 50) -> list[str]:
    """Schema errors of ``obj`` (empty list = valid).

    def_name: a ``$defs`` name; detected from the discriminator fields when None.
    mode: 'fast' (default) or 'full' — see the module docstring.
    """
    name = def_name or detect_def(obj)
    if name is None:
        return ['(root): not a recognised nose-touch document (no record_type / format / kind)']
    if mode == 'full':
        return _jsonschema_errors(name, obj, max_errors)
    if mode != 'fast':
        raise ValueError(f"mode must be 'fast' or 'full', got {mode!r}")
    problems: list[str] = []
    light = _fast_copy(name, obj, problems)
    errs = _jsonschema_errors(name, light, max_errors)
    return (errs + problems)[:max_errors]


def is_valid(obj: Any, def_name: str | None = None, *, mode: str = 'fast') -> bool:
    return not validate(obj, def_name, mode=mode, max_errors=1)


def check(obj: Any, def_name: str | None = None, *, mode: str = 'fast') -> None:
    """Raise :class:`SchemaValidationError` when ``obj`` is invalid."""
    errs = validate(obj, def_name, mode=mode)
    if errs:
        raise SchemaValidationError(def_name or detect_def(obj) or '?', errs)


def deep_copy(record: Any) -> Any:
    """Deep copy that is fast for the large list-of-number stream columns (records are JSON trees)."""
    if isinstance(record, dict):
        return {k: deep_copy(v) for k, v in record.items()}
    if isinstance(record, list):
        if set(map(type, record)) <= _SCALAR_TYPES:
            return list(record)  # numbers/None/str/bool are immutable: a shallow copy is a deep copy
        return [deep_copy(v) for v in record]
    return copy.copy(record)
