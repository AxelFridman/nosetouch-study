"""Thin CLI around lane m3's compute step, to run the slow base experiments in parallel processes.

    cd analysis
    python -m stage2.scripts.run_m3 --threads 1 --names m3_cnn m3_resnet      # these base experiments only
    python -m stage2.scripts.run_m3 --threads 1 --phase2 "m3_cal_deep_*" m3_deep_nested   # part of the second phase
    python -m stage2.scripts.run_m3 --threads 3 --derive                         # the derived files only
    python -m stage2.scripts.run_m3 --threads 3                                  # = python -m stage2.pipeline compute --lanes m3

With --names only those base experiments run, in this process (missing or stale ones; --force: all of them); without
it the whole compute step runs: the base experiments in --threads worker processes of one thread each (every deep fit
uses one intra-op thread anyway, so the results do not depend on the thread count), then the second phase and the
derived files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--names', nargs='+', default=None)
    ap.add_argument('--phase2', nargs='+', default=None, metavar='PATTERN',
                    help='run only these second-phase experiments (role winners re-saved with their n_configs_tried, '
                         'calibration variants, nested choices)')
    ap.add_argument('--derive', action='store_true', help='only write the missing or stale derived files (--force: all)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    os.environ['M3_THREADS'] = str(a.threads)
    if a.names:
        os.environ['M3_WORKERS'] = '1'                 # a named stream runs here, never spawns workers
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from stage2.lanes import m3_pipeline as P      # sets the thread limits before numpy loads
    if a.names:
        P.compute_base(a.names, a.force)
    elif a.phase2:
        from stage2.lanes import m3_derive
        m3_derive.compute_phase2(a.force, a.phase2)
    elif a.derive:
        from stage2.lanes import m3_derive
        m3_derive.derive(force=a.force, only=None)
    else:
        P.compute(a.force, None)


if __name__ == '__main__':
    main()
