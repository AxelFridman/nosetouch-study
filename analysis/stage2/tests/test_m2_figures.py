"""Lane m2 presentation: every figure, table and number is redrawn from the saved derived files alone (into a
temporary output folder, so the real figures are untouched); nothing is fitted; and every number the summary and
the figure texts print is an entry of the numbers file. Skipped before the compute step. The figures are drawn once
per module (drawing them is the slow part)."""
from __future__ import annotations

import json
import shutil

import pytest

from stage2 import config as C

DERIVED = ('m2_experiments.json', 'm2_summary.json', 'm2_calibration.json', 'm2_k_attempts.json', 'm2_importance.json',
           'm2_lc_runs.csv', 'm2_strata.csv', 'm2_strata_persons.csv', 'm2_per_person.csv', 'm2_feature_registry.csv',
           'm2_select_choices.csv')


def _no_fit(mp: pytest.MonkeyPatch) -> None:
    import stage2.lanes.m2_models as M
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError('a presentation step fitted a model'))  # noqa: E731
    mp.setattr(M, 'tune', boom)
    mp.setattr(M, 'fit_base', boom)


@pytest.fixture(scope='module')
def drawn(tmp_path_factory):
    """Copies of the derived files in a temporary tables folder, outputs redirected there, model fitting disabled,
    and every figure drawn once: (output root, figures drawn)."""
    src = C.TAB_DIR / 'm2'
    if not all((src / f).exists() for f in DERIVED):
        pytest.skip('lane m2 derived files missing (python -m stage2.pipeline compute --lanes m2)')
    root = tmp_path_factory.mktemp('m2_presentation')
    tab = root / 'tables'
    (tab / 'm2').mkdir(parents=True)
    for f in DERIVED:
        shutil.copyfile(src / f, tab / 'm2' / f)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, 'FIG_DIR', root / 'figures')
        mp.setattr(C, 'TAB_DIR', tab)
        mp.setattr(C, 'RES_DIR', root / 'results')
        _no_fit(mp)
        from stage2.lanes import m2_figures
        yield root, m2_figures.draw()


def test_figures_redraw_from_saved_files(drawn):
    root, done = drawn
    from stage2.lanes import m2_figures
    assert done == list(m2_figures.FIGURE_FUNCTIONS)
    caps = json.loads((root / 'figures' / 'm2' / 'captions.json').read_text(encoding='utf-8'))
    for name in done:
        assert (root / 'figures' / 'm2' / f'{name}.png').exists()
        assert (root / 'tables' / 'm2' / f'{name}.csv').exists()           # the plotted data, tidy
        assert caps[name]['caption'] and caps[name]['decision']


def test_tables_numbers_and_summary_from_saved_files(drawn):
    root, _ = drawn
    from stage2.lanes import m2_report, m2_summary
    m2_report.tables()
    for t in ('m2_models', 'm2_data_treatment', 'm2_calibration', 'm2_calibration_platform', 'm2_feature_families',
              'm2_breakdown', 'm2_k_attempts', 'm2_learning_curve', 'm2_finalists'):
        assert (root / 'tables' / 'm2' / f'{t}.tex').exists(), t
    m2_report.numbers()
    nums = json.loads((root / 'results' / 'm2_numbers.json').read_text(encoding='utf-8'))
    assert {'lr_full_accuracy', 'lr_full_accuracy_lo', 'lr_full_accuracy_hi', 'n_features', 'select_accuracy',
            'n_models_compared'} <= set(nums)
    assert all(set(v) == {'value', 'format', 'source', 'description'} for v in nums.values())
    text = m2_summary.build()                                               # build only: never overwrite the real file
    assert 'Shown as n/a' not in text                                       # every cited number exists
    assert m2_summary.untraced_numbers(text, nums) == []
    caps = json.loads((root / 'figures' / 'm2' / 'captions.json').read_text(encoding='utf-8'))
    for name, c in caps.items():
        for part in ('caption', 'decision'):
            assert m2_summary.untraced_numbers(c[part], nums) == [], (name, part)
