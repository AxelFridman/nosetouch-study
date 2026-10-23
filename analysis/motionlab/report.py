"""Markdown and CSV helpers for the report scripts (no dependency on ``tabulate``).

Every CSV the kit writes for researchers goes through :func:`write_csv`: record strings reach the tables
unchanged (``quality.warnings``, notes, phone models, ids, validation messages), the upload endpoint is
public, and a spreadsheet opening the CSV would run a cell such as ``=HYPERLINK(...)`` as a formula
(CSV / formula injection). :func:`csv_safe` makes such cells text.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[1] / 'reports'

#: First characters that make Excel, Google Sheets or LibreOffice treat a cell as a formula (OWASP "CSV
#: injection": = + - @, tab, carriage return; line feed for good measure). Leading spaces are skipped.
FORMULA_TRIGGERS: tuple[str, ...] = ('=', '+', '-', '@', '\t', '\r', '\n')


def escape_formula(v: Any) -> Any:
    """``v`` with a leading single quote when it is a string a spreadsheet would evaluate; anything else
    (numbers - also negative ones -, NaN, None, booleans) unchanged. The quote makes the cell text."""
    if isinstance(v, str) and v.lstrip(' ')[:1] in FORMULA_TRIGGERS:
        return "'" + v
    return v


def _texty(dtype: Any) -> bool:
    # object, string dtypes (the pandas >= 3 default 'str', Arrow strings) and categoricals can hold strings
    return dtype == object or isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dtype)


def _safe_labels(idx: pd.Index) -> pd.Index:
    if isinstance(idx, pd.MultiIndex):
        if not len(idx):
            return idx
        return pd.MultiIndex.from_tuples([tuple(escape_formula(x) for x in t) for t in idx], names=idx.names)
    if not _texty(idx.dtype):
        return idx
    return pd.Index([escape_formula(x) for x in idx], name=idx.name, dtype=object)


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` whose string cells, index labels and column names cannot run as spreadsheet formulas
    (:func:`escape_formula`). Numeric columns are left exactly as they are."""
    out = df.copy()
    for i in range(out.shape[1]):
        col = out.iloc[:, i]
        if _texty(col.dtype):
            out.isetitem(i, col.astype(object).map(escape_formula))
    out.index = _safe_labels(out.index)
    out.columns = _safe_labels(out.columns)
    return out


def write_csv(df: pd.DataFrame, path: str | Path, **to_csv_kwargs: Any) -> Path:
    """``df.to_csv(path, **to_csv_kwargs)`` after :func:`csv_safe`; use it for every researcher-facing CSV."""
    p = Path(path)
    csv_safe(df).to_csv(p, **to_csv_kwargs)
    return p


def _fmt(v: Any, digits: int) -> str:
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        if math.isnan(v):
            return ''
        if v == int(v) and abs(v) < 1e12 and digits >= 0:
            return str(int(v))
        return f'{v:.{digits}g}' if abs(v) >= 1e4 or abs(v) < 1e-3 else f'{v:.{digits}f}'.rstrip('0').rstrip('.')
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    return str(v).replace('|', '\\|').replace('\n', ' ')


def df_to_markdown(df: pd.DataFrame, *, index: bool = False, digits: int = 3, max_rows: int = 200) -> str:
    """GitHub-flavoured Markdown table."""
    if df is None or len(df.columns) == 0:
        return '_(empty)_\n'
    d = df.reset_index() if index else df
    if len(d) > max_rows:
        d = d.head(max_rows)
        note = f'\n_(first {max_rows} of {len(df)} rows)_\n'
    else:
        note = ''
    cols = [str(c) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join('---' for _ in cols) + ' |']
    for row in d.itertuples(index=False):
        lines.append('| ' + ' | '.join(_fmt(v, digits) for v in row) + ' |')
    if not len(d):
        lines.append('| ' + ' | '.join('' for _ in cols) + ' |')
    return '\n'.join(lines) + '\n' + note


def dict_to_markdown(d: dict[str, Any], digits: int = 3) -> str:
    return df_to_markdown(pd.DataFrame({'key': list(d.keys()), 'value': [_fmt(v, digits) for v in d.values()]}))


def report_dir(name: str, base: str | Path | None = None) -> Path:
    """analysis/reports/<name>/ (created). ``base`` overrides analysis/reports."""
    safe = ''.join(ch if (ch.isalnum() or ch in '-_.') else '_' for ch in name).strip('.') or 'report'
    p = Path(base) if base else REPORTS_DIR
    out = p / safe
    out.mkdir(parents=True, exist_ok=True)
    return out
