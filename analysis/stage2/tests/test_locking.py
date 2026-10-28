"""stage2.locking: the cross-process lock and atomic replacement behind cv.update_registry and plotting.save_fig.

The two parallel tests start several Python processes that hammer one registry (and one figure folder) at
the same moment; without the lock, read-modify-write loses rows (checked once against the unlocked code)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from stage2 import locking

ANALYSIS = Path(__file__).resolve().parents[2]
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'PYTHONIOENCODING': 'utf-8'}


def test_lock_is_exclusive_and_released(tmp_path):
    target = tmp_path / 'registry.csv'
    with locking.file_lock(target) as lock:
        assert lock == tmp_path / 'registry.csv.lock' and lock.exists()
        assert 'pid=' in lock.read_text(encoding='utf-8')
        t0 = time.monotonic()
        with pytest.raises(locking.LockTimeout, match='stayed busy'):
            with locking.file_lock(target, timeout=0.3):
                pass
        assert time.monotonic() - t0 >= 0.3
    assert not lock.exists()
    with locking.file_lock(target, timeout=0.3):    # free again
        pass


def test_lock_is_released_when_the_block_raises(tmp_path):
    target = tmp_path / 'x.json'
    with pytest.raises(RuntimeError):
        with locking.file_lock(target):
            raise RuntimeError('boom')
    assert not locking.lock_path(target).exists()


def test_a_stale_lock_is_broken_and_a_fresh_one_is_not(tmp_path):
    target = tmp_path / 'captions.json'
    lock = locking.lock_path(target)
    lock.write_text('pid=1 host=gone time=0 token=dead\n', encoding='utf-8')
    old = time.time() - 3600
    os.utime(lock, (old, old))                      # its owner died an hour ago
    with locking.file_lock(target, stale_after=60, timeout=5):
        assert 'token=dead' not in lock.read_text(encoding='utf-8')
    assert not lock.exists() and not list(tmp_path.glob('*.stale'))
    lock.write_text('pid=1 host=busy time=0 token=alive\n', encoding='utf-8')   # fresh: never broken
    with pytest.raises(locking.LockTimeout):
        with locking.file_lock(target, stale_after=60, timeout=0.2):
            pass
    assert 'token=alive' in lock.read_text(encoding='utf-8')


def test_release_never_removes_another_owners_lock(tmp_path):
    target = tmp_path / 'registry.csv'
    with locking.file_lock(target) as lock:
        lock.write_text('pid=2 host=other time=0 token=theirs\n', encoding='utf-8')   # broken and re-taken
    assert lock.read_text(encoding='utf-8').endswith('token=theirs\n')


def test_atomic_path_replaces_on_success_and_leaves_the_target_on_failure(tmp_path):
    target = tmp_path / 'r.csv'
    target.write_text('old\n', encoding='utf-8')
    with locking.atomic_path(target) as tmp:
        assert tmp.parent == tmp_path and tmp != target
        tmp.write_text('new\n', encoding='utf-8')
    assert target.read_text(encoding='utf-8') == 'new\n'
    with pytest.raises(ValueError):
        with locking.atomic_path(target) as tmp:
            tmp.write_text('half', encoding='utf-8')
            raise ValueError('failed mid-write')
    assert target.read_text(encoding='utf-8') == 'new\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['r.csv']        # no temporary file left


def test_update_registry_writes_what_it_wrote_before(tmp_path):
    """Same signature, same bytes as the unlocked read-modify-write it replaces, and replace-by-name."""
    from stage2.cv import update_registry
    rows = [{'name': 'a', 'accuracy': 0.75, 'notes': 'x'}, {'name': 'b', 'accuracy': 0.5, 'notes': ''},
            {'name': 'a', 'accuracy': 0.8, 'notes': 'again'}]
    new_p, old_p = tmp_path / 'new.csv', tmp_path / 'old.csv'
    for r in rows:
        out = update_registry(r, new_p)
        reg = pd.read_csv(old_p) if old_p.exists() else pd.DataFrame()     # the code before the lock
        if len(reg) and 'name' in reg.columns:
            reg = reg[reg['name'] != r['name']]
        reg = pd.DataFrame([r]) if not len(reg) else pd.concat([reg, pd.DataFrame([r])], ignore_index=True)
        reg.to_csv(old_p, index=False)
    assert new_p.read_bytes() == old_p.read_bytes()
    assert list(out['name']) == ['b', 'a'] and out['accuracy'].tolist() == [0.5, 0.8]
    assert not list(tmp_path.glob('*.lock')) and not list(tmp_path.glob('.*.tmp'))


REGISTRY_WORKER = r'''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from stage2.cv import update_registry
reg, prefix, n, go = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
while not go.exists():
    time.sleep(0.002)
for i in range(n):
    update_registry({'name': f'{prefix}_{i:03d}', 'accuracy': i / n, 'notes': prefix}, reg)
'''

FIGURE_WORKER = r'''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from stage2.plotting import figure, save_fig
d, prefix, n, go = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
fig, ax = figure('half', 1.0)
while not go.exists():
    time.sleep(0.002)
for i in range(n):
    fig, ax = figure('half', 1.0)
    ax.plot([0, 1], [0, i])
    ax.set_title(f'{prefix} {i}')
    save_fig(fig, f'{prefix}_{i:02d}', caption=f'caption {prefix} {i}', decision='test', fig_dir=d)
'''


def _run_parallel(code: str, target: Path, n_proc: int, n_each: int, tmp_path: Path) -> None:
    go = tmp_path / 'go'
    procs = [subprocess.Popen([sys.executable, '-c', code, str(ANALYSIS), str(target), f'w{k}', str(n_each), str(go)],
                              cwd=str(ANALYSIS), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             for k in range(n_proc)]
    time.sleep(1.0)                                  # let every worker import, then release them together
    go.write_text('go', encoding='utf-8')
    errors = []
    for p in procs:
        out, err = p.communicate(timeout=300)
        if p.returncode:
            errors.append(err.decode('utf-8', errors='replace')[-2000:])
    assert not errors, errors


def test_parallel_registry_updates_lose_no_row(tmp_path):
    reg = tmp_path / 'results' / 'registry_m9.csv'
    reg.parent.mkdir()
    n_proc, n_each = 4, 25
    _run_parallel(REGISTRY_WORKER, reg, n_proc, n_each, tmp_path)
    df = pd.read_csv(reg)
    assert len(df) == n_proc * n_each and df['name'].is_unique
    assert sorted(df['name']) == sorted(f'w{k}_{i:03d}' for k in range(n_proc) for i in range(n_each))
    assert not list(reg.parent.glob('*.lock')) and not list(reg.parent.glob('.*.tmp'))


def test_parallel_save_fig_keeps_every_caption(tmp_path):
    d = tmp_path / 'figures'
    n_proc, n_each = 3, 4
    _run_parallel(FIGURE_WORKER, d, n_proc, n_each, tmp_path)
    names = sorted(f'w{k}_{i:02d}' for k in range(n_proc) for i in range(n_each))
    caps = json.loads((d / 'captions.json').read_text(encoding='utf-8'))
    texts = json.loads((d / 'figure_texts.json').read_text(encoding='utf-8'))
    assert sorted(caps) == names and sorted(texts) == names
    assert all(caps[n]['caption'] == f"caption {n.split('_')[0]} {int(n.split('_')[1])}" for n in names)
    assert sorted(p.stem for p in d.glob('*.pdf')) == names
    assert not list(d.glob('*.lock')) and not list(d.glob('.*.tmp'))
