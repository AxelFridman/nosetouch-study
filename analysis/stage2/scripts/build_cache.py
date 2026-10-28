"""Build the Stage 2 cache from the frozen snapshot and the curation file (about 1 minute).

    cd analysis && python -m stage2.scripts.build_cache [--quantize-ios] [--cache DIR]

Core unit 'cache' of ``stage2.pipeline``: ``python -m stage2.pipeline cache`` builds it only when it is missing
or stale (snapshot, curation, parameters or builder code changed: ``data.data_identity``); ``--force`` always.
"""
from __future__ import annotations

import argparse

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # run as a file
    import _bootstrap  # type: ignore # noqa: F401

from stage2 import config as C
from stage2.data import build_cache


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', default=str(C.SNAPSHOT_DIR))
    ap.add_argument('--curation', default=str(C.CURATION_FILE))
    ap.add_argument('--cache', default=str(C.CACHE_DIR))
    ap.add_argument('--quantize-ios', action='store_true', help="Chrome's 0.1-unit rounding on iOS values (ablation A3)")
    a = ap.parse_args(argv)
    build_cache(a.snapshot, a.curation, a.cache, quantize_ios=a.quantize_ios)


def step_compute(force: bool = False, only: list[str] | None = None) -> None:
    """Pipeline step: (re)build the cache when it is missing or stale, or when forced."""
    from stage2.pipeline import cache_status
    status, why = cache_status()
    if status == 'current' and not force:
        print(f'cache current ({why}); skipped (--force rebuilds)', flush=True)
        return
    print(f'cache {status} ({why}); building ...', flush=True)
    build_cache(C.SNAPSHOT_DIR, C.CURATION_FILE, C.CACHE_DIR)


ORDER = 0
STEPS = {'compute': step_compute}


if __name__ == '__main__':
    main()
