"""python -m stage3 export | fixtures | all [--refit] [--real-only | --synthetic-only]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='python -m stage3', description=__doc__)
    ap.add_argument('what', choices=('export', 'fixtures', 'all'))
    ap.add_argument('--refit', action='store_true', help='refit the models instead of using the cached fits')
    ap.add_argument('--synthetic-only', action='store_true', help='fixtures: only the committed synthetic ones')
    ap.add_argument('--real-only', action='store_true', help='fixtures: only the local real-recording ones')
    a = ap.parse_args(argv)
    if a.what in ('export', 'all'):
        from .export import export_models
        export_models(refit=a.refit)
    if a.what in ('fixtures', 'all'):
        from .fixtures import write_fixtures
        write_fixtures(synthetic=not a.real_only, real=not a.synthetic_only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
