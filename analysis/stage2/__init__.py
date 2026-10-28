"""stage2 - the Stage 2 foundation of NoseTouch (left vs right hand from phone motion, evaluated on unseen persons).

Modules:
  config       paths (snapshot, curation, cache, outputs), seeds, constants, code fingerprint
  data         curated trial table, flags and gates, exclusion cascade, array cache: load_cache()
  standardize  the canonical per-trial representation (60 Hz, W3C device frame + gravity frame, windows)
  cuefree      baseline cue-free movement detector v0, its evaluation against the oracle window, stress test
  cv           leak-proof leave-one-person-out harness, metrics, bootstrap CIs, paired tests, k-attempt curves
  guard        the LeakageError and the lock that keeps the cache out of a fit_predict
  screen       docs/09 D6 wrong-hand screen and the per-block label-consistency table
  qc           numerical helpers of the descriptive analysis (spectra)
  plotting     one figure style; save_fig() writes vector PDF + PNG preview and the caption registry
  locking      cross-process file lock and atomic replacement for the shared registries (parallel lanes)
  pipeline     the orchestrator (python -m stage2.pipeline TARGET), unit discovery, staleness, the Lane helper
  report_export  booktabs tables, numbers files -> LaTeX macros, report-safe tables, report inputs
  lanes/       lane modules <id>_pipeline.py / <id>_figures.py / <id>_*.py, discovered by the pipeline
Import with analysis/ on sys.path (``cd analysis`` or ``sys.path.insert(0, 'analysis')``).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ANALYSIS = str(_Path(__file__).resolve().parents[1])
if _ANALYSIS not in _sys.path:  # motionlab lives next to stage2
    _sys.path.insert(0, _ANALYSIS)

__version__ = '0.3.0'
