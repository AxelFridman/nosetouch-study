"""Shared helpers that turn saved results into report material: LaTeX tables, number macros, figure files.

Nothing here fits a model or reads raw data; every function is deterministic (same inputs, same bytes).

* :func:`df_to_booktabs` - a DataFrame -> a standalone LaTeX table fragment (booktabs, escaped, formatted).
* :func:`export_table` - CSV (formula-safe, privacy-checked) + the .tex fragment of one table.
* :func:`write_numbers` / :func:`numbers_entry` - ``results/<unit>_numbers.json``: a flat map
  ``key -> {value, format, source, description}`` of every number the report text cites.
* :func:`numbers_to_tex` - every ``*_numbers.json`` -> one .tex file of ``\\newcommand`` macros, named
  :func:`macro_name` (unit prefix + CamelCase key, digits spelled out: ``m2`` / ``acc_k3`` ->
  ``\\MTwoAccKThree``; LaTeX macro names cannot contain digits).
* :func:`collect_figures`, :func:`collect_tables`, :func:`collect_report_inputs` - copy the figure PDFs
  listed in every ``captions.json`` registry, the .tex tables and the macros into one folder for the report.

Formats are Python format specs: ``'.1f'``, ``'.3f'``, ``'d'``, ``',d'``, ``'.1%'`` (x 100, ``%`` escaped in
LaTeX), ``''`` for text. A minus sign is typeset as ``\\ensuremath{-}``, a missing value as ``--``.

Every file is written atomically (a temporary file next to it, then ``os.replace``; ``stage2.locking``), so a
reader in another process never sees a half-written table or numbers file. In a tables folder a report table is a
CSV with its ``.tex`` next to it, derived and figure data are CSVs without one; :func:`check_table_kind` refuses to
let one kind silently overwrite the other (``TableNameCollision``).
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import config as C
from .locking import atomic_path

Formatter = str | Callable[[Any], str] | None

# ───────────────────────────── formatting ─────────────────────────────

_LATEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
                  '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
_LATEX_RE = re.compile('|'.join(re.escape(k) for k in _LATEX_SPECIAL))
MISSING = '--'


def latex_escape(text: Any) -> str:
    """Escape the LaTeX special characters of a plain string (``50 % of P_01`` -> ``50 \\% of P\\_01``)."""
    return _LATEX_RE.sub(lambda m: _LATEX_SPECIAL[m.group(0)], str(text))


def _is_missing(v: Any) -> bool:
    if v is None or type(v).__name__ in ('NAType', 'NaTType'):
        return True
    if isinstance(v, (str, bool)):
        return False
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def format_value(v: Any, spec: Formatter = None) -> str:
    """Plain-text value (not escaped): ``spec`` is a Python format spec, a callable, or None (str / default
    float format ``.2f``). Integral floats accept ``'d'``. Missing values give ``--``."""
    if _is_missing(v):
        return MISSING
    if callable(spec):
        return str(spec(v))
    if spec is None:
        if isinstance(v, bool) or isinstance(v, str):
            return str(v)
        if isinstance(v, int) or (hasattr(v, 'dtype') and getattr(v.dtype, 'kind', '') in 'iu'):
            return str(int(v))
        if isinstance(v, float) or hasattr(v, '__float__'):
            return format(float(v), '.2f')
        return str(v)
    if spec in ('', 's'):
        return str(v)
    if spec.endswith('d') and not isinstance(v, str):
        fv = float(v)
        if not fv.is_integer():
            raise ValueError(f'format {spec!r} needs an integer, got {v!r}')
        return format(int(fv), spec)
    return format(float(v) if not isinstance(v, str) else v, spec)


def latex_value(v: Any, spec: Formatter = None) -> str:
    """:func:`format_value`, LaTeX-escaped, with a typographic minus."""
    s = format_value(v, spec)
    if s == MISSING:
        return s
    s = latex_escape(s)
    return re.sub(r'(?<![\w.])-(?=\d|\.\d)', r'\\ensuremath{-}', s)


def _bare_percent(text: str) -> str:
    """Captions and notes are LaTeX; a bare % would comment out the rest of the line, so it is escaped."""
    return re.sub(r'(?<!\\)%', r'\\%', text or '')


#: what keeps an underscore legal in a caption: math ($...$, $$...$$, \(...\), \[...\]) and the argument of a
#: reference-like command (\ref{tab:m2_x}, \label, \cite, \url, \input, \includegraphics, ...)
_UNDERSCORE_SAFE = re.compile(
    r'(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]'
    r'|\\(?:ref|eqref|cref|Cref|autoref|nameref|pageref|label|cite[pt]?|url|href|hyperref|input|include'
    r'|includegraphics)\*?(?:\[[^\]]*\])?\{[^}]*\}', re.S)
_BARE_UNDERSCORE = re.compile(r'(?<!\\)_')


def _outside_safe(text: str, fn: Callable[[str], str]) -> str:
    out, pos = [], 0
    for m in _UNDERSCORE_SAFE.finditer(text):
        out += [fn(text[pos:m.start()]), m.group(0)]
        pos = m.end()
    return ''.join(out + [fn(text[pos:])])


def bare_underscores(text: str) -> int:
    """How many underscores of a LaTeX caption or note would stop pdflatex ('Missing $ inserted'): not escaped,
    not in math and not in the argument of a reference-like command."""
    n = 0

    def count(s: str) -> str:
        nonlocal n
        n += len(_BARE_UNDERSCORE.findall(s))
        return s
    _outside_safe(text or '', count)
    return n


def escape_bare_underscores(text: str) -> str:
    """Escape the underscores :func:`bare_underscores` counts (``m3_cal`` -> ``m3\\_cal``); math, references and
    already escaped underscores are kept. Captions and notes stay LaTeX otherwise."""
    return _outside_safe(text or '', lambda s: _BARE_UNDERSCORE.sub(r'\\_', s))


#: the other text-mode characters that stop pdflatex outside math: a bare & ('Misplaced alignment tab character'),
#: # ('You can't use macro parameter character') and ^ ('Missing $ inserted')
_BARE_SPECIAL = re.compile(r'(?<!\\)[&#^]')
_SPECIAL_TEXT = {'&': r'\&', '#': r'\#', '^': r'\textasciicircum{}'}


def escape_bare_specials(text: str) -> str:
    """Escape a bare ``&``, ``#`` or ``^`` of a caption or note (``k^(-a)`` -> ``k\\textasciicircum{}(-a)``) outside
    math and outside the argument of a reference-like command, like :func:`escape_bare_underscores`; escaped ones
    (``\\&``) and accents (``\\^{o}``) are kept."""
    return _outside_safe(text or '', lambda s: _BARE_SPECIAL.sub(lambda m: _SPECIAL_TEXT[m.group(0)], s))


def _latex_text(text: str) -> str:
    """A caption or note made safe for pdflatex: a bare %, _, &, # or ^ is escaped (outside math and references)."""
    return escape_bare_specials(escape_bare_underscores(_bare_percent(text)))


# ───────────────────────────── atomic writes ─────────────────────────────


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Write UTF-8 text with LF line ends through a temporary file and ``os.replace`` (``locking.atomic_path``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(p) as tmp:
        tmp.write_text(text, encoding='utf-8', newline='\n')
    return p


def write_csv_atomic(df: Any, path: str | Path, **to_csv_kw: Any) -> Path:
    """``motionlab.report.write_csv`` (formula-safe CSV) through a temporary file and ``os.replace``."""
    from motionlab.report import write_csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(p) as tmp:
        write_csv(df, tmp, **to_csv_kw)
    return p


# ───────────────────────────── table kinds in a tables folder ─────────────────────────────


class TableNameCollision(FileExistsError):
    """A report table and derived (or figure) data would share one CSV file in a tables folder."""


def check_table_kind(csv_path: str | Path, kind: str) -> None:
    """Refuse a silent overwrite between the two kinds of CSV that share ``tables/<id>/``.

    ``csv_path``: the CSV about to be written. ``kind='report'``: a report table (:func:`export_table` with
    ``tex=True``): refused when the CSV exists WITHOUT its ``.tex`` (derived or figure data has the name).
    ``kind='data'``: derived or figure data (``export_table(..., tex=False)``, ``Lane.figure_data``): refused when
    the ``.tex`` exists (the CSV is a report table's). A report table writes its ``.tex`` first, so an interrupted
    export never leaves a CSV that looks like data. Rename one of the two (derived data e.g. ``<name>_rows``), or
    delete the obsolete file. (Derived JSON files cannot collide: no report table is a JSON file.)"""
    csv = Path(csv_path)
    tex = csv.with_name(csv.stem + '.tex')
    if kind == 'report':
        if csv.exists() and not tex.exists():
            raise TableNameCollision(
                f'{csv}: derived or figure data (a CSV without a .tex) already has this name, and the report table '
                f'{csv.stem} would overwrite it. Rename one of them (e.g. the derived data to {csv.stem}_rows), or '
                'delete the CSV if it is obsolete')
    elif kind == 'data':
        if tex.exists():
            raise TableNameCollision(
                f'{csv}: this CSV belongs to the report table {tex.name}; derived or figure data of the same name would '
                f'overwrite it. Rename the derived data (e.g. {csv.stem}_rows), or delete both files of the report '
                'table if it is obsolete')
    else:
        raise ValueError("kind must be 'report' or 'data'")


def _refuse_columnless(df: Any, name: str) -> None:
    if len(getattr(df, 'columns', ())) == 0:
        raise ValueError(f'{name}: a table without columns would be written as an unreadable CSV; build an empty '
                         'table with its columns (pd.DataFrame(rows, columns=[...])), which is written header-only')


# ───────────────────────────── privacy ─────────────────────────────

QUASI_PATTERN = re.compile(r'\bP-[A-Z0-9]{6}\b|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
                           r'|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}')


def _cache_maps() -> tuple[dict[str, str], dict[str, str]]:
    import pandas as pd
    t = C.CACHE_DIR / 'trials.pkl'
    s = C.CACHE_DIR / 'sessions.pkl'
    tk = pd.read_pickle(t).set_index('trial_id')['trial_key'].astype(str).to_dict() if t.exists() else {}
    sl = pd.read_pickle(s).set_index('session_id')['session_label'].astype(str).to_dict() if s.exists() else {}
    return tk, sl


def report_safe(df: Any) -> Any:
    """A copy fit for a report-facing table: ``trial_id`` -> ``trial_key`` (P14-s3-t07a1) and ``session_id`` ->
    ``session_label`` (P14-s3) from the cache; any other quasi-identifier column (app code, timestamps, OS /
    browser version, screen geometry: ``data.QUASI_IDENTIFIER_COLUMNS``) or a cell that looks like an app
    code, a backend id or a timestamp raises ``ValueError``."""
    import pandas as pd

    from .data import QUASI_IDENTIFIER_COLUMNS
    df = df.copy()
    if 'trial_id' in df.columns or 'session_id' in df.columns:
        tk, sl = _cache_maps()
        for col, new, m in (('trial_id', 'trial_key', tk), ('session_id', 'session_label', sl)):
            if col not in df.columns:
                continue
            mapped = df[col].astype(str).map(m)
            if mapped.isna().any():
                raise ValueError(f'{int(mapped.isna().sum())} {col} value(s) are not in the cache: map them to '
                                 f'pseudonymous labels before exporting')
            if new in df.columns:
                df = df.drop(columns=[col])
            else:
                df.insert(int(df.columns.get_loc(col)), new, mapped)
                df = df.drop(columns=[col])
    bad = [c for c in df.columns if str(c) in QUASI_IDENTIFIER_COLUMNS]
    if bad:
        raise ValueError(f'quasi-identifier column(s) {bad} cannot enter a report table')
    for c in df.columns:
        if QUASI_PATTERN.search(str(c)):
            raise ValueError(f'column name {c!r} looks like an app code, backend id or timestamp')
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):   # pandas 3: 'str' dtype
            hit = df[c].astype(str).str.contains(QUASI_PATTERN, na=False)
            if hit.any():
                raise ValueError(f'column {c!r} holds an app code, backend id or timestamp')
    return df


# ───────────────────────────── tables ─────────────────────────────


def df_to_booktabs(df: Any, path: str | Path | None, caption: str = '', label: str = '',
                   column_formats: Mapping[str, Formatter] | None = None, *, align: str | None = None,
                   header: Mapping[str, str] | None = None, raw: Sequence[str] = (), note: str = '',
                   environment: str | None = 'table', placement: str = 'htbp', size: str | None = r'\small',
                   index: bool = False, source: str = '') -> str:
    """Write ``df`` as a standalone LaTeX table fragment (booktabs) and return its text.

    column_formats: column -> Python format spec ('.1f', '.1%', 'd', ...) or callable(value) -> str; other
    columns: integers as is, floats '.2f', text as is. Every cell and header is LaTeX-escaped except the
    columns in ``raw`` (already LaTeX). ``caption`` and ``note`` are LaTeX (a bare ``%`` is escaped, and so is a
    bare ``_``, ``&``, ``#`` or ``^`` outside math and outside the argument of \\ref, \\label, \\cite, \\url, ...). header:
    column -> printed name. align: the tabular column spec
    (default 'l' for text, 'r' for numbers). environment='table' wraps it in a float with caption and label
    (None: the bare tabular, the caption kept as a comment). note: a footnote line under the table.
    ``path=None`` only returns the text. Needs booktabs in the preamble."""
    import pandas as pd
    if index:
        df = df.reset_index()
    fmts = dict(column_formats or {})
    unknown = set(fmts) - set(df.columns)
    if unknown:
        raise KeyError(f'column_formats names unknown columns: {sorted(unknown)}')
    cols = list(df.columns)
    if align is None:
        align = ''.join('r' if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) else 'l'
                        for c in cols)
    if len(re.sub(r'[^lrcpXS]', '', re.sub(r'\{[^}]*\}', '', align))) != len(cols):
        raise ValueError(f'align {align!r} does not have {len(cols)} columns')
    caption, note = _latex_text(caption), _latex_text(note)
    head = [latex_escape((header or {}).get(c, c)) for c in cols]
    lines = []
    for i in range(len(df)):                     # by column, so every column keeps its own dtype
        cells = []
        for c in cols:
            v = df[c].iloc[i]
            if c in raw:
                cells.append(MISSING if _is_missing(v) else str(v))
            else:
                cells.append(latex_value(v, fmts.get(c)))
        lines.append(' & '.join(cells) + r' \\')
    body = [r'\begin{tabular}{' + align + '}', r'\toprule', ' & '.join(head) + r' \\', r'\midrule', *lines,
            r'\bottomrule', r'\end{tabular}']
    src = f' from {source}' if source else ''
    out = [f'% generated by stage2.report_export{src}; do not edit by hand (rerun: python -m stage2.pipeline tables)']
    if environment:
        out += [rf'\begin{{{environment}}}[{placement}]', r'\centering']
        if size:
            out.append(size)
        if caption:
            out.append(r'\caption{' + caption + '}')
        if label:
            out.append(r'\label{' + label + '}')
        out += body
        if note:
            out.append(r'\par\smallskip{\footnotesize ' + note + '}')
        out.append(rf'\end{{{environment}}}')
    else:
        if caption:
            out.append('% caption: ' + caption.replace('\n', ' '))
        if size:
            out.append('{' + size)
        out += body
        if note:
            out.append(r'\par\smallskip{\footnotesize ' + note + '}')
        if size:
            out.append('}')
    text = '\n'.join(out) + '\n'
    if path is not None:
        write_text_atomic(path, text)
    return text


def export_table(df: Any, path_stem: str | Path, *, caption: str = '', label: str = '',
                 column_formats: Mapping[str, Formatter] | None = None, tex: bool = True, safe: bool = True,
                 **booktabs_kw: Any) -> dict[str, Path]:
    """Write ``<path_stem>.csv`` (formula-safe, full precision) and ``<path_stem>.tex`` (:func:`df_to_booktabs`).
    With ``safe`` the table first goes through :func:`report_safe` (no backend key, app code or timestamp).

    ``tex=True`` writes a report table, ``tex=False`` derived data (a CSV only); :func:`check_table_kind` refuses to
    overwrite a CSV of the other kind. A table needs columns (an empty one is written header-only). Both files are
    replaced atomically, the ``.tex`` first."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    _refuse_columnless(df, stem.name)
    check_table_kind(stem.with_suffix('.csv'), 'report' if tex else 'data')
    if safe:
        df = report_safe(df)
    out = {'csv': stem.with_suffix('.csv')}
    if tex:
        out['tex'] = stem.with_suffix('.tex')
        df_to_booktabs(df, out['tex'], caption, label, column_formats, **booktabs_kw)
    write_csv_atomic(df, out['csv'], index=False)
    return out


# ───────────────────────────── numbers ─────────────────────────────

_ONES = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve',
         'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen')
_TENS = ('', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety')
KEY_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def spell_number(digits: str) -> str:
    """'3' -> 'Three', '95' -> 'NinetyFive', '100' -> 'OneHundred', '2026' -> 'TwoZeroTwoSix' (digit by digit
    above 999 or with a leading zero)."""
    if not digits.isdigit():
        raise ValueError(digits)
    if (len(digits) > 1 and digits[0] == '0') or len(digits) > 3:
        return ''.join(_ONES[int(c)] for c in digits)
    n = int(digits)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else '')
    rest = n % 100
    return _ONES[n // 100] + 'Hundred' + (spell_number(str(rest)) if rest else '')


def _camel(text: str) -> str:
    parts = re.findall(r'[A-Za-z]+|\d+', text)
    return ''.join(spell_number(p) if p.isdigit() else p[0].upper() + p[1:] for p in parts)


def macro_name(unit: str, key: str) -> str:
    """LaTeX macro name (without the backslash) of number ``key`` of unit ``unit``: CamelCase, digits spelled
    out. ``macro_name('m2', 'acc_k3')`` -> 'MTwoAccKThree'; ``macro_name('reference', 'auc')`` -> 'ReferenceAuc'."""
    name = _camel(unit) + _camel(key)
    if not re.fullmatch(r'[A-Za-z]+', name):
        raise ValueError(f'cannot make a macro name of {unit!r} / {key!r}')
    return name


def numbers_entry(value: Any, fmt: str, source: str, description: str) -> dict[str, Any]:
    """One entry of a numbers file. value: a number (or text); fmt: Python format spec ('.1%', '.2f', 'd', '');
    source: the saved file it was read from (relative to the output folder); description: what it is."""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    if value is not None and not isinstance(value, (int, float, str, bool)):
        raise TypeError(f'numbers hold scalars or text, not {type(value).__name__}')
    format_value(value, fmt)                       # fail early on a format that does not fit the value
    return {'value': value, 'format': fmt, 'source': str(source).replace('\\', '/'), 'description': description}


def validate_numbers(entries: Mapping[str, Any]) -> None:
    for k, e in entries.items():
        if not KEY_RE.match(k):
            raise ValueError(f'number key {k!r}: use lower_snake_case starting with a letter')
        if not isinstance(e, Mapping) or set(e) != {'value', 'format', 'source', 'description'}:
            raise ValueError(f'number {k!r} must be {{value, format, source, description}}')


def write_numbers(path: str | Path, entries: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write a numbers file (sorted keys, deterministic). Its name must be ``<unit>_numbers.json``."""
    p = Path(path)
    if not p.name.endswith('_numbers.json'):
        raise ValueError('a numbers file is named <unit>_numbers.json')
    validate_numbers(entries)
    names = {}
    unit = p.name[: -len('_numbers.json')]
    for k in entries:
        m = macro_name(unit, k)
        if m in names:
            raise ValueError(f'keys {names[m]!r} and {k!r} give the same macro \\{m}')
        names[m] = k
    return write_text_atomic(p, json.dumps(dict(sorted(entries.items())), indent=1, ensure_ascii=False,
                                           allow_nan=False) + '\n')


def numbers_files(res_dir: str | Path = C.RES_DIR) -> list[Path]:
    return sorted(Path(res_dir).glob('*_numbers.json'))


def numbers_to_tex(paths: Iterable[str | Path] | str | Path | None = None, out: str | Path | None = None) -> str:
    """Every numbers file -> ``\\newcommand`` macros (one per key, sorted; the unit is the file name before
    ``_numbers.json``). ``paths`` may be files or folders (default: results/). Returns the text and writes it
    to ``out`` (default results/numbers.tex). Duplicate macro names raise."""
    if paths is None:
        paths = [C.RES_DIR]
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files += numbers_files(p) if p.is_dir() else [p]
    lines = ['% generated by stage2.report_export.numbers_to_tex from results/*_numbers.json; do not edit by hand',
             '% (rerun: python -m stage2.pipeline numbers). One macro per number: \\<Unit><Key>, digits spelled out.']
    seen: dict[str, str] = {}
    for f in sorted(set(files)):
        unit = f.name[: -len('_numbers.json')] if f.name.endswith('_numbers.json') else f.stem
        entries = json.loads(f.read_text(encoding='utf-8'))
        validate_numbers(entries)
        lines.append(f'% --- {unit} ({f.name}) ---')
        for k in sorted(entries):
            e = entries[k]
            m = macro_name(unit, k)
            if m in seen:
                raise ValueError(f'macro \\{m} is defined by {seen[m]} and {unit}/{k}')
            seen[m] = f'{unit}/{k}'
            desc = ' '.join(str(e['description']).split())
            lines.append(f'\\newcommand{{\\{m}}}{{{latex_value(e["value"], e["format"])}}}% {desc} [{e["source"]}]')
    text = '\n'.join(lines) + '\n'
    write_text_atomic(Path(out) if out is not None else C.RES_DIR / 'numbers.tex', text)
    return text


# ───────────────────────────── collecting report inputs ─────────────────────────────


def caption_registries(fig_root: str | Path = C.FIG_DIR) -> list[Path]:
    """Every ``captions.json`` under the figure folder (the core one and one per lane folder)."""
    root = Path(fig_root)
    return sorted(p for p in root.rglob('captions.json') if p.is_file())


def collect_figures(target_dir: str | Path, fig_root: str | Path = C.FIG_DIR, *, formats: Sequence[str] = ('pdf',)
                    ) -> list[Path]:
    """Copy every figure listed in a captions registry under ``fig_root`` to ``target_dir`` (keeping the lane
    sub-folders) and write ``target_dir/captions.json`` (relative path -> caption, decision). A registry entry
    whose file is missing raises: the registry and the figures must agree."""
    root, target = Path(fig_root), Path(target_dir)
    copied, merged = [], {}
    for reg in caption_registries(root):
        caps = json.loads(reg.read_text(encoding='utf-8'))
        sub = reg.parent.relative_to(root)
        for name, c in sorted(caps.items()):
            for ext in formats:
                src = reg.parent / f'{name}.{ext}'
                if not src.exists():
                    raise FileNotFoundError(f'{src} is in {reg} but does not exist: redraw the figures')
                dst = target / sub / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied.append(dst)
            merged[(sub / name).as_posix()] = {'caption': c.get('caption', ''), 'decision': c.get('decision', '')}
    target.mkdir(parents=True, exist_ok=True)
    (target / 'captions.json').write_text(json.dumps(merged, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return copied


def collect_tables(target_dir: str | Path, tab_root: str | Path = C.TAB_DIR) -> list[Path]:
    """Copy every LaTeX table fragment (``*.tex``) under ``tab_root`` to ``target_dir`` (keeping sub-folders)."""
    root, target = Path(tab_root), Path(target_dir)
    out = []
    for f in sorted(root.rglob('*.tex')):
        dst = target / f.relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
        out.append(dst)
    return out


def collect_report_inputs(target: str | Path = C.REPORT_INPUTS_DIR) -> dict[str, Any]:
    """figures/ (PDFs + captions.json), tables/ (.tex) and numbers.tex in one folder for the LaTeX report."""
    target = Path(target)
    figs = collect_figures(target / 'figures')
    tabs = collect_tables(target / 'tables')
    numbers_to_tex(None, target / 'numbers.tex')
    return {'target': str(target), 'figures': len(figs), 'tables': len(tabs)}
