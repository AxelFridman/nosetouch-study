"""The core unit 'describe': compute is separate from presentation, figures read saved files only, nothing
data-specific is typed into captions, and the numbers file covers what the README and the report cite."""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from stage2 import config as C
from stage2 import pipeline as P


@pytest.fixture(scope='module')
def described(primary):
    """The describe outputs of the current cache (run python -m stage2.pipeline describe first)."""
    from stage2.scripts import describe as D
    if not D.SUMMARY_JSON.exists():
        pytest.skip('no describe output (python -m stage2.pipeline describe)')
    if not D.outputs_current():
        pytest.skip('describe outputs are from another cache or incomplete (python -m stage2.pipeline describe)')
    return D, D.load_summary()


def test_unit_contract_and_sections():
    from stage2.scripts import describe as D
    from stage2.scripts import describe_figures as F
    assert D.ORDER == 10 and set(D.STEPS) == set(P.STEP_NAMES)
    assert set(D.SECTION_FIGURES) == set(D.SECTIONS)
    fed = {f for figs in D.SECTION_FIGURES.values() for f in figs}
    assert fed == set(F.FIGURE_FUNCTIONS), 'every figure is fed by a section and every fed name is a figure'
    assert set(F.FIGURES) == set(F.FIGURE_FUNCTIONS)
    assert all(fn.__name__ == f'fig_{name}' for name, fn in F.FIGURE_FUNCTIONS.items())
    assert D.selected_sections(['S2_05_sampling_gaps']) == ['raw_quality']
    assert D.selected_sections(['cuefree*']) == ['cuefree', 'cuefree_stress']
    assert D.selected_sections(['S2_13_first_half_yaw']) == ['primary_trials', 'first_half_yaw']
    assert D.selected_sections(['nothing_like_this']) == []


def test_presentation_options_are_checked():
    from stage2.scripts import describe_figures as F
    with pytest.raises(TypeError, match='unknown presentation option'):
        F.fig_S2_17_gravity_closure(no_such_option=1)


def test_figures_draw_from_saved_files_only(described, monkeypatch, tmp_path):
    """Every figure function runs with the cache, the raw snapshot and the analysis sections locked away, and
    hands save_fig a caption and a tidy DataFrame of its plotted data."""
    D, s = described
    from stage2 import data
    from stage2.plotting import save_fig
    from stage2.scripts import describe_figures as F

    def refuse(*a, **k):
        raise AssertionError('a figure function read the cache or the raw data')
    for mod, name in ((data, 'load_cache'), (data, 'load_tables'), (D, 'load_cache'), (D, 'load_tables'),
                      (D, 'load_dataset'), (D, 'Ctx')):
        monkeypatch.setattr(mod, name, refuse)
    saved = {}

    def fake_save(fig, name, *, caption, decision, data):
        assert isinstance(data, pd.DataFrame) and len(data), name
        saved[name] = caption
        return save_fig(fig, name, caption=caption, decision=decision, fig_dir=tmp_path)
    monkeypatch.setattr(F, '_save', fake_save)
    done = F.draw()
    assert done == list(F.FIGURE_FUNCTIONS)
    assert sorted(p.stem for p in tmp_path.glob('*.png')) == sorted(saved)
    n = s['demographics']['n_persons']
    assert saved['S2_01_demographics'].startswith(f'The {n} persons')
    assert f'all {n} persons' in saved['S2_13_first_half_yaw']
    assert f'across the {n} persons' in saved['S2_11_class_means']
    caps = json.loads((tmp_path / 'captions.json').read_text(encoding='utf-8'))
    assert sorted(caps) == sorted(saved)


def test_saved_outputs_follow_the_contract(described):
    D, s = described
    from stage2.scripts import describe_figures as F
    caps = json.loads((C.FIG_DIR / 'captions.json').read_text(encoding='utf-8'))
    for name in F.FIGURE_FUNCTIONS:
        assert (C.FIG_DIR / f'{name}.pdf').exists() and (C.FIG_DIR / f'{name}.png').exists(), name
        assert caps[name]['caption'] and caps[name]['decision'], name
        assert (D.DATA_DIR / f'{name}.csv').exists(), f'{name}: no plotted-data CSV'
    for name in D.TABLES:
        assert (D.DATA_DIR / f'{name}.csv').exists() and (D.DATA_DIR / f'{name}.tex').exists(), name
        tex = (D.DATA_DIR / f'{name}.tex').read_text(encoding='utf-8')
        assert r'\begin{tabular}' in tex and r'\label{tab:' + name + '}' in tex
    for f in D.INPUT_FILES:
        p = D.DATA_DIR / f'describe_{f}'
        assert p.exists(), p
        if p.suffix == '.json':
            assert P.file_current(p), f'{p.name} was made from another cache'
    assert P.file_current(D.SUMMARY_JSON)


def test_no_typed_counts_in_captions(described):
    """Captions take every count from the data: no person count other than the current one appears."""
    D, s = described
    n = s['demographics']['n_persons']
    caps = json.loads((C.FIG_DIR / 'captions.json').read_text(encoding='utf-8'))
    for name, c in caps.items():
        if not name.startswith('S2_'):
            continue
        for m in re.finditer(r'(\d+) persons', c['caption'] + ' ' + c['decision']):
            assert int(m.group(1)) <= n, (name, m.group(0))
    lc = s['label_consistency']
    assert f'the other {n - 1} ' in lc['screen_model']
    assert 'swap_signature_blocks_detail' in lc and not any(k.startswith('P0') for k in lc)


def test_numbers_cover_the_cited_values(described):
    D, s = described
    nums = json.loads(D.NUMBERS_JSON.read_text(encoding='utf-8'))
    dm = s['demographics']
    assert nums['n_persons']['value'] == dm['n_persons'] and nums['primary_trials']['value'] == dm['primary_trials']
    sets = D.load_input('sets')['sets']
    assert nums['natural_use_n']['value'] == sets['natural_use']['n'] == sets['natural_use']['persons']
    assert nums['cuefree_share_approach']['value'] == s['cuefree_v0']['all']['share_approach']
    assert nums['stress_random_share_approach']['value'] == s['cuefree_v0_stress']['random']['share_approach']
    assert nums['gyro_power_below_lowpass']['value'] == s['spectrum']['gyro_power_below_15hz']
    flow = {r['step']: r for r in s['exclusion_flow']}
    assert all(nums[f'excl_{k.lower()}_removed']['value'] == flow[k]['removed'] for k in ('E0', 'E1', 'E4b', 'E4c', 'E4d'))
    for k, e in nums.items():
        assert e['source'] and not e['source'].startswith('cache/'), k      # numbers read saved outputs only
    tex = (C.RES_DIR / 'numbers.tex').read_text(encoding='utf-8')
    assert r'\newcommand{\DescribeNPersons}' in tex and r'\newcommand{\DescribeStressRandomShareApproach}' in tex
