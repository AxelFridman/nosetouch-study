"""report_export: booktabs tables, number macros, privacy checks and the collection of report inputs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage2 import report_export as RX


def test_latex_escape_and_values():
    assert RX.latex_escape('50 % of P_01 & #2') == r'50 \% of P\_01 \& \#2'
    assert RX.latex_value(0.7792, '.1%') == r'77.9\%'
    assert RX.latex_value(-3.81, '.1f') == r'\ensuremath{-}3.8'
    assert RX.latex_value(5.0, 'd') == '5' and RX.latex_value(np.int64(7), None) == '7'
    assert RX.latex_value(math.nan, '.2f') == '--' and RX.latex_value(None, 'd') == '--'
    assert RX.latex_value('P08-s1', '') == 'P08-s1'          # a hyphen between words is no minus sign
    with pytest.raises(ValueError):
        RX.format_value(2.5, 'd')


def test_df_to_booktabs(tmp_path):
    df = pd.DataFrame({'model': ['yaw_logreg', 'cnn'], 'n': [598, 598], 'accuracy': [0.779, 0.8123],
                       'p': [0.5, np.nan]})
    p = tmp_path / 't.tex'
    text = RX.df_to_booktabs(df, p, 'Accuracy of 2 models', 'tab:acc', {'accuracy': '.1%', 'p': '.2f'},
                             header={'accuracy': 'Accuracy (%)'}, note='LOSO, 95 % CI')
    assert p.read_text(encoding='utf-8') == text
    for s in (r'\begin{table}[htbp]', r'\toprule', r'\midrule', r'\bottomrule', r'\caption{Accuracy of 2 models}',
              r'\label{tab:acc}', r'yaw\_logreg & 598 & 77.9\% & 0.50 \\', r'cnn & 598 & 81.2\% & -- \\',
              r'model & n & Accuracy (\%) & p \\', r'\begin{tabular}{lrrr}', r'LOSO, 95 \% CI'):
        assert s in text, s
    bare = RX.df_to_booktabs(df, None, 'cap', 'lab', environment=None)
    assert r'\begin{table}' not in bare and '% caption: cap' in bare
    assert RX.df_to_booktabs(df, None, 'c', 'l', {'accuracy': '.1%'}) == RX.df_to_booktabs(df, None, 'c', 'l', {'accuracy': '.1%'})
    with pytest.raises(KeyError):
        RX.df_to_booktabs(df, None, column_formats={'nope': '.1f'})
    with pytest.raises(ValueError):
        RX.df_to_booktabs(df, None, align='lr')


def test_report_safe_refuses_quasi_identifiers():
    with pytest.raises(ValueError):
        RX.report_safe(pd.DataFrame({'participant_code': ['x']}))
    with pytest.raises(ValueError):
        RX.report_safe(pd.DataFrame({'note': ['from P-ABC123']}))
    with pytest.raises(ValueError):
        RX.report_safe(pd.DataFrame({'when': ['2026-10-27T14:34:00Z']}))
    ok = RX.report_safe(pd.DataFrame({'person_id': ['P01'], 'acc': [0.8]}))
    assert list(ok.columns) == ['person_id', 'acc']


def test_report_safe_maps_trial_ids_to_keys(tables):
    t = tables['trials'].head(3)
    out = RX.report_safe(pd.DataFrame({'trial_id': t['trial_id'], 'p': [0.1, 0.2, 0.3]}))
    assert list(out.columns) == ['trial_key', 'p'] and list(out['trial_key']) == list(t['trial_key'])
    with pytest.raises(ValueError):
        RX.report_safe(pd.DataFrame({'trial_id': ['not-a-cached-trial']}))


def test_macro_names():
    assert RX.macro_name('m2', 'acc_k3') == 'MTwoAccKThree'
    assert RX.macro_name('reference', 'accuracy_lo') == 'ReferenceAccuracyLo'
    assert RX.macro_name('x1', 'ci95_hi') == 'XOneCiNinetyFiveHi'
    assert RX.macro_name('describe', 'fhy_persons_auc_ge80') == 'DescribeFhyPersonsAucGeEighty'
    assert RX.spell_number('100') == 'OneHundred' and RX.spell_number('05') == 'ZeroFive'
    assert RX.spell_number('2026') == 'TwoZeroTwoSix' and RX.spell_number('17') == 'Seventeen'


def test_numbers_round_trip(tmp_path):
    E = RX.numbers_entry
    RX.write_numbers(tmp_path / 'm2_numbers.json', {
        'accuracy': E(0.8123, '.1%', 'results/m2_cnn.json', 'LOSO accuracy of the CNN'),
        'n_persons': E(np.int64(20), 'd', 'results/m2_cnn.json', 'persons'),
        'best_model': E('MiniRocket', '', 'results/m2_cnn.json', 'name of the best model'),
        'delta': E(-0.012, '+.1%', 'results/m2_cnn.json', 'difference'),
    })
    RX.write_numbers(tmp_path / 'x1_numbers.json', {'n_trials': E(12, 'd', 'tables/x1/a.csv', 'trials')})
    d = json.loads((tmp_path / 'm2_numbers.json').read_text(encoding='utf-8'))
    assert list(d) == sorted(d) and d['n_persons'] == {'value': 20, 'format': 'd', 'source': 'results/m2_cnn.json',
                                                       'description': 'persons'}
    out = tmp_path / 'numbers.tex'
    text = RX.numbers_to_tex(tmp_path, out)
    assert out.read_text(encoding='utf-8') == text
    for s in (r'\newcommand{\MTwoAccuracy}{81.2\%}', r'\newcommand{\MTwoNPersons}{20}',
              r'\newcommand{\MTwoBestModel}{MiniRocket}', r'\newcommand{\MTwoDelta}{\ensuremath{-}1.2\%}',
              r'\newcommand{\XOneNTrials}{12}'):
        assert s in text, s
    assert text == RX.numbers_to_tex(tmp_path, out)                     # deterministic
    with pytest.raises(ValueError):
        RX.write_numbers(tmp_path / 'bad_numbers.json', {'Bad-Key': E(1, 'd', 's', 'd')})
    with pytest.raises(ValueError):
        RX.write_numbers(tmp_path / 'm3_numbers.json', {'k_1': E(1, 'd', 's', 'd'), 'k1': E(1, 'd', 's', 'd')})
    with pytest.raises(ValueError):
        RX.numbers_entry(2.5, 'd', 's', 'not an integer')


def test_collect_figures_and_tables(tmp_path):
    from stage2.plotting import figure, save_fig
    figs, tabs = tmp_path / 'figures', tmp_path / 'tables'
    for sub, name in (('', 'S2_99_core'), ('m9', 'M9_01_demo')):
        fig, ax = figure('half', 1.5)
        ax.plot([0, 1], [0, 1])
        save_fig(fig, name, caption=f'caption of {name}', decision='a decision', fig_dir=figs / sub if sub else figs)
    RX.df_to_booktabs(pd.DataFrame({'a': [1]}), tabs / 'm9' / 'm9_table.tex', 'c', 'tab:m9')
    target = tmp_path / 'report_inputs'
    copied = RX.collect_figures(target / 'figures', figs)
    assert sorted(p.relative_to(target / 'figures').as_posix() for p in copied) == ['S2_99_core.pdf', 'm9/M9_01_demo.pdf']
    caps = json.loads((target / 'figures' / 'captions.json').read_text(encoding='utf-8'))
    assert caps['m9/M9_01_demo']['caption'] == 'caption of M9_01_demo'
    assert [p.name for p in RX.collect_tables(target / 'tables', tabs)] == ['m9_table.tex']
    (figs / 'm9' / 'M9_01_demo.pdf').unlink()
    with pytest.raises(FileNotFoundError):
        RX.collect_figures(target / 'figures', figs)


# ───────────────────────────── core maintenance 2026-10-28 ─────────────────────────────


def test_a_report_table_and_derived_data_never_overwrite_each_other(tmp_path, monkeypatch):
    df = pd.DataFrame({'model': ['a'], 'accuracy': [0.8]})
    RX.export_table(df, tmp_path / 'm9_rows', tex=False)                      # derived data first ...
    with pytest.raises(RX.TableNameCollision, match='derived or figure data'):
        RX.export_table(df, tmp_path / 'm9_rows', caption='c')               # ... a report table may not replace it
    RX.export_table(df, tmp_path / 'm9_table', caption='c')                  # a report table first ...
    with pytest.raises(RX.TableNameCollision, match='belongs to the report table'):
        RX.export_table(df, tmp_path / 'm9_table', tex=False)                 # ... derived data may not replace it
    RX.export_table(df.assign(accuracy=0.9), tmp_path / 'm9_table', caption='c')   # same kind: overwritten
    RX.export_table(df.assign(accuracy=0.9), tmp_path / 'm9_rows', tex=False)
    assert pd.read_csv(tmp_path / 'm9_table.csv')['accuracy'].tolist() == [0.9]
    # the Lane helpers: figure data cannot replace a report table's CSV either (same file ignoring case on Windows)
    from stage2 import config as C
    from stage2.pipeline import Lane
    monkeypatch.setattr(C, 'TAB_DIR', tmp_path)
    L = Lane('m9')
    RX.export_table(df, tmp_path / 'm9' / 'M9_02_clash', caption='c')         # a report table with a figure's name
    with pytest.raises(RX.TableNameCollision):
        L.figure_data(df, 'M9_02_clash')
    assert L.figure_data(df, 'M9_01_fine').exists()
    L.export_table(df, 'm9_ok', caption='c')
    assert (tmp_path / 'm9' / 'm9_ok.tex').exists()
    with pytest.raises(ValueError):
        RX.check_table_kind(tmp_path / 'x.csv', 'other')


def test_an_empty_table_is_written_header_only_and_a_columnless_one_is_refused(tmp_path):
    out = RX.export_table(pd.DataFrame(columns=['model', 'accuracy']), tmp_path / 'm9_empty', caption='c')
    back = pd.read_csv(out['csv'])
    assert list(back.columns) == ['model', 'accuracy'] and back.empty
    for kw in ({}, {'tex': False}):
        with pytest.raises(ValueError, match='without columns'):
            RX.export_table(pd.DataFrame(), tmp_path / 'm9_nothing', **kw)
    assert not list(tmp_path.glob('m9_nothing*'))
    from stage2.pipeline import Lane
    with pytest.raises(ValueError, match='without columns'):
        Lane('m9').figure_data(pd.DataFrame(), 'M9_09_nothing')


def test_tables_and_numbers_are_replaced_atomically(tmp_path, monkeypatch):
    """Every write goes through a temporary file and os.replace (stage2.locking): no reader sees half a file,
    no temporary file stays behind, and the bytes are those of a direct write."""
    from stage2 import locking
    seen = []
    real = locking.replace_file

    def spy(src, dst, **kw):
        seen.append((Path(src).name, Path(dst).name))
        return real(src, dst, **kw)
    monkeypatch.setattr(locking, 'replace_file', spy)
    df = pd.DataFrame({'model': ['a'], 'accuracy': [0.8]})
    out = RX.export_table(df, tmp_path / 'm9_t', caption='cap')
    RX.write_numbers(tmp_path / 'm9_numbers.json', {'n': RX.numbers_entry(3, 'd', 'results/x.json', 'n')})
    RX.numbers_to_tex(tmp_path, tmp_path / 'numbers.tex')
    dst = [d for _, d in seen]
    assert dst[:2] == ['m9_t.tex', 'm9_t.csv']                               # the .tex first (marks the kind)
    assert {'m9_numbers.json', 'numbers.tex'} <= set(dst)
    assert all(s.startswith('.') and s.endswith('.tmp') for s, _ in seen)
    assert not list(tmp_path.glob('.*.tmp'))
    from motionlab.report import write_csv
    write_csv(df, tmp_path / 'direct.csv', index=False)
    assert (tmp_path / 'direct.csv').read_bytes() == out['csv'].read_bytes()
    text = RX.df_to_booktabs(df, None, 'cap', '')
    assert out['tex'].read_bytes() == text.encode('utf-8')


def test_bare_underscores_in_captions_are_escaped_outside_math_and_references():
    cap = r'Model m3_cal_deep (see \ref{tab:m3_x}, $s_k$ and \(a_b\)); already m2\_select; 50 % done'
    assert RX.bare_underscores(cap) == 2                                     # the two in m3_cal_deep
    out = RX.escape_bare_underscores(cap)
    assert r'm3\_cal\_deep' in out and r'\ref{tab:m3_x}' in out and '$s_k$' in out and r'\(a_b\)' in out
    assert r'm2\_select' in out and r'm2\\_select' not in out and RX.bare_underscores(out) == 0
    text = RX.df_to_booktabs(pd.DataFrame({'a': [1]}), None, caption=cap, label='tab:m9_a', note='with s_k')
    assert r'\caption{Model m3\_cal\_deep (see \ref{tab:m3_x}, $s_k$' in text and r'50 \% done' in text
    assert r'\label{tab:m9_a}' in text and r'with s\_k' in text
    plain = 'no underscore here'
    assert RX.escape_bare_underscores(plain) == plain


def test_save_fig_moves_the_images_in_with_their_registry_entries(tmp_path):
    """save_fig renders to temporary files, then moves the images into place and updates both registries under
    the folder's lock: no temporary file stays, the registries name every PDF, and redrawing gives the same bytes."""
    import json as _json

    from stage2.plotting import figure, save_fig

    def draw():
        fig, ax = figure('half', 1.5)
        ax.plot([0, 1], [0, 1], label='a line')
        ax.legend()
        return save_fig(fig, 'M9_01_demo', caption='a demo', decision='none', fig_dir=tmp_path)
    draw()
    pdf1, png1 = (tmp_path / 'M9_01_demo.pdf').read_bytes(), (tmp_path / 'M9_01_demo.png').read_bytes()
    draw()
    assert (tmp_path / 'M9_01_demo.pdf').read_bytes() == pdf1 and (tmp_path / 'M9_01_demo.png').read_bytes() == png1
    assert pdf1.startswith(b'%PDF') and png1.startswith(b'\x89PNG')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['M9_01_demo.pdf', 'M9_01_demo.png', 'captions.json',
                                                          'figure_texts.json']
    texts = _json.loads((tmp_path / 'figure_texts.json').read_text(encoding='utf-8'))
    assert 'a line' in texts['M9_01_demo']
    assert _json.loads((tmp_path / 'captions.json').read_text(encoding='utf-8'))['M9_01_demo']['caption'] == 'a demo'


def test_bare_specials_in_captions_are_escaped_outside_math():
    """A bare ^, & or # stops pdflatex in a caption or note (lane m1's 'k^(-alpha)' did): escaped outside math and
    reference arguments; math, escaped characters and accents are kept; a caption without them is unchanged."""
    cap = r'k^(-alpha) x sum, A & B, #3; math $k^{-\alpha}$ and \(x^2\); kept: \& \# \^{o} \ref{tab:m9_a}'
    out = RX.escape_bare_specials(cap)
    assert out.startswith(r'k\textasciicircum{}(-alpha) x sum, A \& B, \#3;')
    assert r'$k^{-\alpha}$' in out and r'\(x^2\)' in out and r'\ref{tab:m9_a}' in out
    assert r'kept: \& \# \^{o}' in out
    assert RX.escape_bare_specials(out) == out                              # idempotent
    plain = 'Accuracy (95 % CI) of m9_model > 0.5 for every k'
    assert RX.escape_bare_specials(plain) == plain
    text = RX.df_to_booktabs(pd.DataFrame({'a': [1]}), None, caption='k^(-alpha) & #2 of m9_a, 5 %', label='tab:m9_b',
                             note='a^b')
    assert r'\caption{k\textasciicircum{}(-alpha) \& \#2 of m9\_a, 5 \%}' in text
    assert r'a\textasciicircum{}b' in text
