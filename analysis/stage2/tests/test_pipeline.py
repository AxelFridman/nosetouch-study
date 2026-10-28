"""stage2.pipeline: unit discovery and order, the step contract, failures, selection and staleness helpers."""
from __future__ import annotations

import json
import textwrap

import pytest

from stage2 import config as C
from stage2 import pipeline as P

LANE = '''
"""Lane {id} (test)."""
from pathlib import Path
LOG = Path(r"{log}")
ORDER = {order}
DEPENDS = {depends}

def _step(name):
    def fn(force, only):
        if {fail!r} == name:
            raise RuntimeError('boom in {id}.' + name)
        with open(LOG, 'a') as f:
            f.write(f"{id}.{{name}} force={{force}} only={{only}}\\n")
    return fn

STEPS = {{s: _step(s) for s in {steps!r}}}
'''


@pytest.fixture
def lanes(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'PIPELINE_LOG', tmp_path / 'pipeline_log.jsonl')
    log = tmp_path / 'calls.txt'
    d = tmp_path / 'lanes'
    d.mkdir()

    def make(lid, order=100, depends=(), steps=('compute', 'figures', 'tables', 'numbers'), fail=''):
        (d / f'{lid}_pipeline.py').write_text(textwrap.dedent(LANE.format(
            id=lid, log=log, order=order, depends=tuple(depends), steps=tuple(steps), fail=fail)), encoding='utf-8')

    def calls():
        return log.read_text().splitlines() if log.exists() else []
    return d, make, calls


def test_discovery_order_and_depends(lanes):
    d, make, _ = lanes
    make('m2', order=100)
    make('m1', order=100)
    make('s1', order=900, depends=('m1', 'm2'))
    make('m3', order=50, depends=('s1',))          # DEPENDS wins over ORDER
    assert P.lane_ids(d) == ['m1', 'm2', 'm3', 's1']
    units = P.discover(['m1', 'm2', 'm3', 's1'], lanes_dir=d)
    assert [u.id for u in units] == ['m1', 'm2', 's1', 'm3']
    assert all(u.kind == 'lane' and set(u.steps) == set(P.STEP_NAMES) for u in units)
    with pytest.raises(P.PipelineError):
        P.discover(['nope'], lanes_dir=d)


def test_circular_and_unknown_depends(lanes):
    d, make, _ = lanes
    make('a1', depends=('b1',))
    make('b1', depends=('a1',))
    with pytest.raises(P.PipelineError, match='circular'):
        P.discover(['a1', 'b1'], lanes_dir=d)
    make('c1', depends=('zz',))
    with pytest.raises(P.PipelineError, match='unknown'):
        P.discover(['c1'], lanes_dir=d)


def test_bad_contract_is_refused(lanes):
    d, _, _ = lanes
    (d / 'b2_pipeline.py').write_text('STEPS = {"train": print}\n', encoding='utf-8')
    with pytest.raises(P.PipelineError, match='unknown steps'):
        P.discover(['b2'], lanes_dir=d)
    (d / 'b3_pipeline.py').write_text('X = 1\n', encoding='utf-8')
    with pytest.raises(P.PipelineError, match='STEPS'):
        P.discover(['b3'], lanes_dir=d)


def test_run_passes_force_and_only_and_logs(lanes):
    d, make, calls = lanes
    make('m1')
    make('m2', steps=('compute', 'figures'))
    recs = P.run('compute', lanes=['m2', 'm1'], only=['m1_a*'], force=True, lanes_dir=d)
    assert calls() == ["m1.compute force=True only=['m1_a*']", "m2.compute force=True only=['m1_a*']"]
    assert [(r.unit, r.step, r.status) for r in recs] == [('m1', 'compute', 'ok'), ('m2', 'compute', 'ok')]
    log = [json.loads(x) for x in C.PIPELINE_LOG.read_text(encoding='utf-8').splitlines()]
    assert [x['unit'] for x in log] == ['m1', 'm2'] and all(x['status'] == 'ok' and 'seconds' in x for x in log)
    P.run('figures', lanes=['m1'], lanes_dir=d)
    assert calls()[-1] == 'm1.figures force=False only=None'


def test_a_failing_step_stops_the_run(lanes):
    d, make, calls = lanes
    make('m1', fail='compute')
    make('m2', order=200)
    with pytest.raises(P.PipelineError, match=r'm1\.compute FAILED .*boom in m1\.compute'):
        P.run('compute', lanes=['m1', 'm2'], lanes_dir=d)
    assert calls() == []                            # m2 never ran
    assert json.loads(C.PIPELINE_LOG.read_text(encoding='utf-8').splitlines()[-1])['status'] == 'failed'


def test_core_targets_refuse_lanes():
    with pytest.raises(P.PipelineError):
        P.run('reference', lanes=['m1'])
    with pytest.raises(P.PipelineError):
        P.run('nonsense')


def test_core_units_follow_the_contract():
    units = {u.id: u for u in P.discover(['cache', 'describe', 'reference'])}
    assert [u.id for u in P.discover(['reference', 'describe', 'cache'])] == ['cache', 'describe', 'reference']
    assert set(units['cache'].steps) == {'compute'}
    assert set(units['describe'].steps) == set(P.STEP_NAMES) == set(units['reference'].steps)
    assert all(u.kind == 'core' for u in units.values())


def test_selection_and_lane_conventions():
    assert P.selected('m2_cnn', None) and P.selected('m2_cnn', ['m2_*']) and not P.selected('m2_cnn', ['m3_*'])
    L = P.Lane('m2')
    assert L.check_name('m2_cnn') == 'm2_cnn' and L.check_figure('M2_01_curve') == 'M2_01_curve'
    for bad in ('m3_cnn', 'cnn'):
        with pytest.raises(ValueError):
            L.check_name(bad)
    with pytest.raises(ValueError):
        L.check_figure('m2_01_curve')
    assert L.fig_dir == C.FIG_DIR / 'm2' and L.tab_dir == C.TAB_DIR / 'm2'
    assert L.registry.name == 'registry_m2.csv' and L.numbers_path.name == 'm2_numbers.json'
    assert L.finalists_path.name == 'finalists_m2.json' and L.summary_path.name == 'm2_SUMMARY.md'
    with pytest.raises(ValueError):
        P.Lane('M2')


def test_staleness_helpers(tmp_path, primary):
    status, why = P.cache_status()
    assert status == 'current', f'the cache is {status} ({why}): python -m stage2.pipeline cache'
    did = P.cache_data_id()
    assert did and len(did) == 64
    (tmp_path / 'm9_x.json').write_text(json.dumps({'info': {'cache_code': {'data_id': did}}}), encoding='utf-8')
    (tmp_path / 'm9_old.json').write_text(json.dumps({'info': {'cache_code': {'data_id': 'other'}}}), encoding='utf-8')
    assert P.result_current('m9_x', tmp_path) and not P.result_current('m9_old', tmp_path)
    assert not P.result_current('m9_missing', tmp_path)
    assert not P.should_run('m9_x', False, None, tmp_path) and P.should_run('m9_x', True, None, tmp_path)
    assert P.should_run('m9_old', False, None, tmp_path) and not P.should_run('m9_old', False, ['m8_*'], tmp_path)
    (tmp_path / 'derived.json').write_text(json.dumps({'provenance': P.provenance(), 'x': 1}), encoding='utf-8')
    assert P.file_current(tmp_path / 'derived.json') and not P.file_current(tmp_path / 'nope.json')


def test_reference_outputs_follow_the_contract(primary):
    """The worked example: its numbers, tables and figure data exist after 'pipeline reference' and its
    results were made from the current cache."""
    from stage2.scripts import reference
    if not reference.NUMBERS_JSON.exists():
        pytest.skip('run python -m stage2.pipeline reference first')
    for name in reference.EXPERIMENTS:
        assert P.result_current(name), f'{name} was made from another cache'
    nums = json.loads(reference.NUMBERS_JSON.read_text(encoding='utf-8'))
    assert {'accuracy', 'accuracy_lo', 'accuracy_hi', 'n_persons', 'acc_k1'} <= set(nums)
    assert nums['n_persons']['value'] == primary.trials['person_id'].nunique()
    for stem in ('reference_models', 'reference_k_attempts'):
        assert (reference.TAB_DIR / f'{stem}.tex').exists() and (reference.TAB_DIR / f'{stem}.csv').exists()
    assert (reference.TAB_DIR / 'S2_21_reference_sensitivity.csv').exists()
    tex = (C.RES_DIR / 'numbers.tex').read_text(encoding='utf-8')
    assert r'\newcommand{\ReferenceAccuracy}' in tex and r'\newcommand{\DescribeNPersons}' in tex


# ───────────────────────────── core maintenance 2026-10-28 ─────────────────────────────

THREAD_LANE = '''
"""Lane t9 (test): records the thread budget its own limit_threads(2) leaves in force."""
import os
from pathlib import Path
from stage2.pipeline import limit_threads
limit_threads(2)
Path(r"{out}").write_text(os.environ["OMP_NUM_THREADS"] + " " + os.environ["OPENBLAS_NUM_THREADS"])
STEPS = {{'figures': lambda force, only: None}}
'''


def _run_cli(args, tmp_path, env_extra=None):
    import os
    import subprocess
    import sys
    env = {**os.environ, 'STAGE2_LANES': str(tmp_path / 'lanes'), 'PYTHONIOENCODING': 'utf-8',
           'OMP_NUM_THREADS': '7', **(env_extra or {})}
    env.pop(P.THREADS_OVERRIDE_VAR, None)
    return subprocess.run([sys.executable, '-m', 'stage2.pipeline', *args], cwd=str(C.ANALYSIS_DIR), env=env,
                          capture_output=True, text=True, timeout=300)


def test_threads_on_the_command_line_win_over_the_lane_budget(tmp_path):
    """python -m stage2.pipeline --threads N: the lane module's own limit_threads(2) runs with N in force (the
    pipeline runs as __main__ while the lane imports stage2.pipeline: two module objects, one override)."""
    (tmp_path / 'lanes').mkdir()
    out = tmp_path / 'threads.txt'
    (tmp_path / 'lanes' / 't9_pipeline.py').write_text(THREAD_LANE.format(out=out.as_posix()), encoding='utf-8')
    r = _run_cli(['--list', '--lanes', 't9', '--threads', '1'], tmp_path)
    assert r.returncode == 0, r.stderr
    assert out.read_text() == '1 1'
    r = _run_cli(['--list', '--lanes', 't9'], tmp_path)          # without --threads the lane's budget holds
    assert r.returncode == 0, r.stderr
    assert out.read_text() == '2 2'


def test_thread_override_is_process_local(monkeypatch):
    import os
    monkeypatch.delenv(P.THREADS_OVERRIDE_VAR, raising=False)
    for v in P._THREAD_VARS:
        monkeypatch.setenv(v, os.environ.get(v, '1'))
    P.set_thread_override(3)
    try:
        P.limit_threads(1)
        assert os.environ['OMP_NUM_THREADS'] == '3' and P.thread_override() == 3
        monkeypatch.setenv(P.THREADS_OVERRIDE_VAR, '5@999999999')   # another process's override: ignored
        assert P.thread_override() is None
        P.limit_threads(2)
        assert os.environ['OMP_NUM_THREADS'] == '2'
    finally:
        P.set_thread_override(None)
    with pytest.raises(ValueError):
        P.set_thread_override(0)


def test_opt_in_spec_and_input_staleness(tmp_path, primary, monkeypatch):
    """A result is current for its own declared spec and stale for an edited one; a derived file written with
    inputs= is stale once an input changes; files without that record keep the data-identity rule."""
    did = P.cache_data_id()
    d = {'name': 'm9_x', 'model': 'lr', 'params': {'C': 1.0},
         'info': {'cache_code': {'data_id': did}, 'notes': 'n', 'n_configs_tried': 2, 'sets': ['primary'],
                  'cap_per_person': 40, 'seed': 1, 'n_boot': 0, 'leak_check_mode': 'all', 'test_meta_columns': ['row']}}
    (tmp_path / 'm9_x.json').write_text(json.dumps(d), encoding='utf-8')
    assert P.result_current('m9_x', tmp_path)
    assert P.result_current('m9_x', tmp_path, spec={'n_configs_tried': 2, 'notes': 'n', 'params': {'C': 1.0}})
    assert not P.result_current('m9_x', tmp_path, spec={'n_configs_tried': 3})
    assert not P.result_current('m9_x', tmp_path, spec={'code': 'v2'})         # an extra the result never declared
    assert P.should_run('m9_x', False, None, tmp_path, spec={'notes': 'edited'})
    assert not P.should_run('m9_x', False, None, tmp_path, spec={'notes': 'n'})
    # derived files: inputs recorded by Lane.write_json
    monkeypatch.setattr(C, 'OUT_DIR', tmp_path)
    monkeypatch.setattr(C, 'TAB_DIR', tmp_path / 'tables')
    monkeypatch.setattr(C, 'RES_DIR', tmp_path)
    src = tmp_path / 'm9_x.json'
    L = P.Lane('m9')
    f = L.write_json('m9_derived', {'value': 1}, inputs=['m9_x'])
    doc = json.loads(f.read_text(encoding='utf-8'))
    assert doc['provenance'] == P.provenance() and list(doc['provenance_inputs'].values())[0]
    assert P.file_current(f)
    src.write_text(json.dumps({**d, 'model': 'lr2'}), encoding='utf-8')     # the input changed
    assert not P.file_current(f)
    plain = L.write_json('m9_plain', {'value': 1})
    assert 'provenance_inputs' not in json.loads(plain.read_text(encoding='utf-8')) and P.file_current(plain)
    with pytest.raises(ValueError):
        L.write_json('m9_bad', {'provenance_inputs': 1}, inputs=[])


def test_every_saved_result_is_current_so_compute_is_a_no_op(primary):
    """After the 2026-10-28 maintenance (re-scored results, spec fingerprints read from the saved fields) every
    saved result is still current, under the data identity and under its own declared spec: a plain
    'python -m stage2.pipeline compute' reruns no experiment."""
    from stage2 import cv
    n = 0
    for f in sorted(C.RES_DIR.glob('*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        if not (isinstance(d, dict) and 'predictions' in d and 'info' in d):
            continue
        assert P.result_current(f.stem), f'{f.stem} is stale'
        assert P.result_current(f.stem, spec=cv.result_spec(d))
        assert not P.should_run(f.stem, False, None)
        n += 1
    if not n:
        pytest.skip('no saved results')


def test_the_pipeline_log_is_appended_under_its_lock(tmp_path, monkeypatch):
    """Parallel pipelines append whole lines only (stage2.locking); a busy lock delays, never drops, a line."""
    import os
    import subprocess
    import sys
    log = tmp_path / 'pipeline_log.jsonl'
    code = ('import sys; from stage2 import config as C, pipeline as P; from pathlib import Path; '
            'C.PIPELINE_LOG = Path(sys.argv[1]); '
            '[P._log({"unit": sys.argv[2], "i": i, "pad": "x" * 3000}) for i in range(60)]')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'PYTHONIOENCODING': 'utf-8'}
    procs = [subprocess.Popen([sys.executable, '-c', code, str(log), f'u{k}'], cwd=str(C.ANALYSIS_DIR), env=env)
             for k in range(4)]
    assert all(p.wait(timeout=300) == 0 for p in procs)
    lines = log.read_text(encoding='utf-8').splitlines()
    recs = [json.loads(x) for x in lines]                    # every line is whole JSON
    assert len(recs) == 240 and len({(r['unit'], r['i']) for r in recs}) == 240
    from stage2 import locking
    monkeypatch.setattr(C, 'PIPELINE_LOG', tmp_path / 'held.jsonl')
    with locking.file_lock(C.PIPELINE_LOG):
        import threading
        t = threading.Thread(target=P._log, args=({'unit': 'late'},))
        t.start()
        t.join(0.3)
        assert t.is_alive() and not (tmp_path / 'held.jsonl').exists()   # waits for the lock
    t.join(30)
    assert json.loads((tmp_path / 'held.jsonl').read_text(encoding='utf-8'))['unit'] == 'late'


def test_exploratory_tests_run_only_when_selected():
    """Any lane can run the whole suite: the tests that read the exploratory persons' files are skipped unless
    selected (-m exploratory or STAGE2_EXPLORATORY_TESTS=1, conftest.py). Checked in a child pytest that never
    selects them (so nothing exploratory is read here), plus a collection with the marker (nothing runs)."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != 'STAGE2_EXPLORATORY_TESTS'}
    env['PYTHONIOENCODING'] = 'utf-8'
    base = [sys.executable, '-m', 'pytest', 'stage2/tests/test_exploratory_guard.py', '-q', '-p', 'no:cacheprovider']
    r = subprocess.run([*base, '-rs'], cwd=str(C.ANALYSIS_DIR), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:]
    assert '3 skipped' in r.stdout and 'select it with -m exploratory' in r.stdout, r.stdout[-2000:]
    assert '2 passed' in r.stdout                     # the checks that read no exploratory file still run
    c = subprocess.run([*base, '--collect-only', '-m', 'exploratory'], cwd=str(C.ANALYSIS_DIR), env=env,
                       capture_output=True, text=True, timeout=600)
    assert c.returncode == 0 and '3/5 tests collected' in c.stdout, c.stdout[-2000:]


def test_source_fingerprint_is_a_code_version(tmp_path):
    """pipeline.source_fingerprint: the same code gives the same version (whatever the order or the way a module is
    named), an edit of it another one; a missing module is an error."""
    from stage2 import cv
    v = P.source_fingerprint(cv)
    assert len(v) == 16 and v == P.source_fingerprint('stage2.cv') == P.source_fingerprint('stage2/cv.py')
    assert P.source_fingerprint(cv, 'stage2.locking') == P.source_fingerprint('stage2.locking', cv) != v
    a, b = tmp_path / 'a.py', tmp_path / 'b.py'
    a.write_text('X = 1\n', encoding='utf-8')
    b.write_text('Y = 2\n', encoding='utf-8')
    before = P.source_fingerprint(a, b)
    assert before == P.source_fingerprint(b, a)
    a.write_text('X = 1  # edited\n', encoding='utf-8')
    assert P.source_fingerprint(a, b) != before
    with pytest.raises(ModuleNotFoundError):
        P.source_fingerprint('stage2.no_such_module')


def test_input_hashes_find_source_files_relative_to_analysis():
    """A relative input that is not in the output folder but in analysis/ (a source file) is hashed there, so an
    edit of it makes the derived file stale; a module object stands for its file."""
    from stage2 import cv
    h = P.input_hashes(['stage2/cv.py'])
    assert list(h) == ['stage2/cv.py'] and h['stage2/cv.py'] is not None
    assert P.input_hashes([cv]) == h
    assert list(P.input_hashes(['tables/m9/never_written.json']).values()) == [None]   # missing: recorded as None
