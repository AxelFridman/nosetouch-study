"""End-to-end mini pipeline on synthetic data: the three scripts on a folder tree plus an export bundle."""
from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest
from _mlhelpers import ANALYSIS_DIR

from motionlab import synth


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f'scripts_{name}', ANALYSIS_DIR / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def data(tiny_dataset, tiny_tree, tmp_path_factory):
    """The folder tree, and one session again as a phone export (duplicates must be dropped)."""
    d = tmp_path_factory.mktemp('exports')
    bundle = synth.export_bundle(tiny_dataset, session_ids=[tiny_dataset.sessions[0]['session_id']])
    path = d / 'nosetouch_all_20261120-1530.json'
    path.write_text(json.dumps(bundle), encoding='utf-8')
    return [str(tiny_tree), str(path)]


def test_qa_report(data, tmp_path):
    out = tmp_path / 'qa'
    assert _script('qa_report').main([*data, '--out', str(out), '--quiet', '--sample', '20']) == 0
    md = (out / 'qa_report.md').read_text(encoding='utf-8')
    for section in ('## Summary', '## Sampling', '## Duplicates', '## Gravity sign', '## Segmentation by status',
                    '## Orientation check', '## Gravity closure'):
        assert section in md, section
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['online_v1 port == stored (all trials)'] == 1.0
    assert summary['duplicate record copies dropped'] > 0
    assert (out / 'sampling.png').exists() and (out / 'tables' / 'sampling.csv').exists()


def test_train_baseline(data, tmp_path):
    out = tmp_path / 'base'
    rc = _script('train_baseline').main([*data, '--out', str(out), '--quiet', '--n-boot', '50', '--n-splits', '3',
                                         '--schemes', 'stratified_group_kfold,loso,orientation,platform'])
    assert rc == 0
    m = pd.read_csv(out / 'metrics.csv')
    assert set(m['scheme']) == {'stratified_group_kfold', 'loso', 'orientation', 'platform'}
    assert set(m['model']) == {'yaw_rule', 'logreg', 'hgb'}
    dev = m[m['scheme'] == 'stratified_group_kfold'].set_index('model')
    assert dev.loc['yaw_rule', 'accuracy'] > 0.9 and dev.loc['logreg', 'accuracy'] > 0.9
    assert {'accuracy_lo', 'accuracy_hi', 'roc_auc_lo'} <= set(m.columns)
    sub = pd.read_csv(out / 'subgroups.csv')
    assert {'hand', 'platform', 'orientation', 'sex'} <= set(sub['variable'])
    mir = pd.read_csv(out / 'mirror_consistency.csv')
    assert (mir.loc[mir['model'] == 'yaw_rule', 'label_flip_rate'] == 1.0).all()
    md = (out / 'report.md').read_text(encoding='utf-8')
    for section in ('## Analysis set', '## Results', '## Mirror consistency', '## Leakage control', '## Subgroup errors'):
        assert section in md, section
    for f in ('roc.png', 'subgroups.png', 'features.csv', 'oof_loso.csv', 'participant_accuracy.csv', 'exclusions.csv'):
        assert (out / f).exists(), f


def test_train_baseline_ablation_options(data, tmp_path):
    out = tmp_path / 'abl'
    rc = _script('train_baseline').main([*data, '--out', str(out), '--quiet', '--n-boot', '0', '--n-splits', '3',
                                         '--schemes', 'group_kfold', '--models', 'yaw_rule,logreg_chiral',
                                         '--quantize', 'ios', '--augment-mirror', '--no-plots'])
    assert rc == 0
    m = pd.read_csv(out / 'metrics.csv').set_index('model')
    assert m.loc['yaw_rule', 'accuracy'] > 0.9


def test_qa_report_on_protocol_2_data(street_tree, tmp_path):
    """Street visits (implicit calibration): the closure check names the mapping, the engine + vote consensus
    the sign; the protocol 2.0.0 tables are written."""
    out = tmp_path / 'qa_street'
    assert _script('qa_report').main([str(street_tree), '--out', str(out), '--quiet', '--sample', '10', '--no-plots']) == 0
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    assert summary['calibration methods'] == 'implicit_v1 4' and summary['gravity sign sources'] == 'engine_vote 4'
    assert summary['rotationRate mapping sources'] == 'closure w3c/deg_per_s 4'
    assert summary['sessions with unverified rotationRate mapping'] == 0
    assert summary['online_v1 port == stored (all trials)'] == 1.0
    for t in ('normalisation', 'schema_1_2_fields', 'sessions_by_status'):
        assert (out / 'tables' / f'{t}.csv').exists(), t
    md = (out / 'qa_report.md').read_text(encoding='utf-8')
    assert '## Normalisation' in md and '## Schema 1 2 fields' in md and 'norm_v2' in md


def test_train_baseline_screen_away_augmentation(data, street_tree, tmp_path):
    out = tmp_path / 'away'
    rc = _script('train_baseline').main([*data, str(street_tree), '--out', str(out), '--quiet', '--n-boot', '0',
                                         '--n-splits', '3', '--schemes', 'group_kfold', '--models', 'yaw_rule,hgb',
                                         '--augment-screen-away', '--require-sign-support', '--no-plots',
                                         '--no-permutation-check'])
    assert rc == 0
    m = pd.read_csv(out / 'metrics.csv').set_index('model')
    assert m.loc['yaw_rule', 'accuracy'] > 0.9 and m.loc['hgb', 'n'] == m.loc['yaw_rule', 'n']
    md = (out / 'report.md').read_text(encoding='utf-8')
    assert 'Screen-away augmentation' in md and 'never report screen-away accuracy' in md
    assert 'handedness x platform' in md or 'self-reported handedness x platform' in md
    exc = pd.read_csv(out / 'exclusions.csv')
    assert 'S3 gravity sign supported' in set(exc['rule'])
    oof = pd.read_csv(out / 'oof_group_kfold.csv')
    assert len(oof) and not oof['trial_id'].astype(str).str.endswith('-away').any()  # test trials are always real
    feats = pd.read_csv(out / 'features.csv')
    assert 'qc_screen_flipped' in feats.columns and (feats['qc_screen_flipped'] == 0).all()


def test_plot_trial(data, tiny_dataset, tmp_path, capsys):
    tid = tiny_dataset.trials[0]['trial_id']
    mod = _script('plot_trial')
    assert mod.main([*data, '--trial', tid[:8], '--out', str(tmp_path)]) == 0
    assert (tmp_path / f'trial_{tid}.png').exists()
    assert mod.main([*data, '--trial', 'no-such-trial', '--out', str(tmp_path)]) == 1
    capsys.readouterr()
    assert mod.main([*data, '--list', '3']) == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0].split()[:2] == ['trial_id', 'participant_id'] and len(listing) == 4
