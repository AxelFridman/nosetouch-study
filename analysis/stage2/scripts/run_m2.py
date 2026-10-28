"""Run lane m2's experiments only (no derived data), e.g. as one of two single-thread workers:

    cd analysis
    .venv/Scripts/python.exe -m stage2.scripts.run_m2 --threads 1 --only "m2_lr_*" "m2_lc_*"
    .venv/Scripts/python.exe -m stage2.scripts.run_m2 --list

The full lane step (experiments + derived data) is ``python -m stage2.pipeline compute --lanes m2``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', default=None, metavar='NAME', help='experiment names / shell patterns')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--threads', type=int, default=2, help='thread budget of this process (<= 2)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    from stage2.pipeline import limit_threads
    limit_threads(max(1, min(2, a.threads)))
    from stage2.lanes import m2_pipeline as P
    if a.list:
        for e in P.experiments().values():
            print(f'{e.name:<28} {e.group:<15} {e.label}')
        return 0
    P.run_experiments(a.force, a.only)
    return 0


if __name__ == '__main__':
    sys.exit(main())
