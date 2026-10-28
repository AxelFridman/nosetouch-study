"""Paths, seeds and constants of the Stage 2 pipeline (one place; every module imports from here).

Paths can be redirected with environment variables (tests, other snapshots):
  STAGE2_SNAPSHOT   the frozen snapshot folder (backend layout nose-touch-v1/<code>/<session>/...)
  STAGE2_CURATION   the researcher's curation file (sessions.json)
  STAGE2_CACHE      the cache folder (git-ignored)
  STAGE2_OUT        the output folder for figures, tables and results (git-ignored)
  STAGE2_REPORT_INPUTS  where ``python -m stage2.pipeline report`` collects figures, tables and numbers
  STAGE2_LANES      the folder the pipeline discovers lane modules in (tests; default stage2/lanes)

Privacy: the curation folder also holds a private file with real names. Nothing in this package opens
it (``PRIVATE_FILE_NAME`` is checked by :func:`assert_not_private`), and no output carries a name: persons
are P01, P02, ... only (the curation assigns the ids).

Exploratory persons (``EXPLORATORY_*``): participants staged OUTSIDE the snapshot (PD01, ...) for one
exploratory lane only. They are never part of the snapshot, the curation,
the cache or any model; ``stage2/tests/test_exploratory_guard.py`` checks that none of their session ids or
app codes appears in the main snapshot, curation or cache.
"""
from __future__ import annotations

import os
from pathlib import Path

# ───────────────────────────── paths ─────────────────────────────

ANALYSIS_DIR = Path(__file__).resolve().parents[1]          # analysis/
REPO_DIR = ANALYSIS_DIR.parent
DATA_DIR = ANALYSIS_DIR / 'data'                             # git-ignored (analysis/data/*)


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


SNAPSHOT_NAME = '2026-10-27'
SNAPSHOT_DIR = _env_path('STAGE2_SNAPSHOT', DATA_DIR / 'real' / SNAPSHOT_NAME)
EXPORT_BACKUP = DATA_DIR / 'backups' / 'export_20261027-165030.json.gz'  # the full DB after P20 (14:34Z) was recorded
CURATION_FILE = _env_path('STAGE2_CURATION', DATA_DIR / 'curation' / 'sessions.json')
PRIVATE_FILE_NAME = 'person_names_PRIVATE.json'              # never opened by this package
CACHE_DIR = _env_path('STAGE2_CACHE', DATA_DIR / 'stage2_cache')
OUT_DIR = _env_path('STAGE2_OUT', ANALYSIS_DIR / 'reports' / 'stage2')
FIG_DIR = OUT_DIR / 'figures'
TAB_DIR = OUT_DIR / 'tables'
RES_DIR = OUT_DIR / 'results'
REGISTRY_CSV = RES_DIR / 'registry.csv'
CAPTIONS_JSON = FIG_DIR / 'captions.json'
LANES_DIR = _env_path('STAGE2_LANES', ANALYSIS_DIR / 'stage2' / 'lanes')   # lane modules: lanes/<id>_pipeline.py
REPORT_INPUTS_DIR = _env_path('STAGE2_REPORT_INPUTS', OUT_DIR / 'report_inputs')   # filled by 'pipeline report'
PIPELINE_LOG = OUT_DIR / 'pipeline_log.jsonl'                # one line per pipeline step run (duration, status)

# exploratory persons: outside the snapshot, read only by the exploratory lane (x1); never modelled
EXPLORATORY_DIR = DATA_DIR / 'exploratory'                   # raw data (backend layout per person)
EXPLORATORY_CURATION = DATA_DIR / 'curation' / 'exploratory.json'
EXPLORATORY_CACHE_DIR = _env_path('STAGE2_EXPLORATORY_CACHE', DATA_DIR / 'exploratory_cache')


def assert_not_private(path: str | Path) -> Path:
    """Refuse to touch the private name file (defence in depth: no code path should ever name it)."""
    p = Path(path)
    if p.name == PRIVATE_FILE_NAME:
        raise PermissionError('the private name file is never read by the Stage 2 pipeline')
    return p


def ensure_out_dirs() -> None:
    for d in (FIG_DIR, TAB_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)


REQUIREMENTS_FILE = ANALYSIS_DIR / 'requirements-stage2.txt'


def _tree_sha256(root: Path, pattern: str, skip: tuple[str, ...] = ()) -> str:
    """sha256 over (relative path, sha256 of the bytes) of every matching file, sorted by path."""
    import hashlib
    h = hashlib.sha256()
    files = sorted(p for p in root.glob(pattern)
                   if p.is_file() and not any(s in p.relative_to(root).parts for s in skip))
    for f in files:
        h.update(str(f.relative_to(root)).replace('\\', '/').encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def code_fingerprint() -> dict[str, str]:
    """Hashes of the code that produced an output, so a cache or result can be traced to it even while the
    code is not committed: the stage2 package modules (``stage2/*.py``: what builds the cache and runs the
    harness), the stage2 scripts (tables and figures), the motionlab sources it builds on, and the frozen
    environment (``requirements-stage2.txt``). Tests are left out: they change no output."""
    import hashlib
    req = REQUIREMENTS_FILE
    return {
        'stage2_sha256': _tree_sha256(ANALYSIS_DIR / 'stage2', '*.py'),
        'stage2_scripts_sha256': _tree_sha256(ANALYSIS_DIR / 'stage2' / 'scripts', '*.py'),
        'motionlab_sha256': _tree_sha256(ANALYSIS_DIR / 'motionlab', '**/*.py', skip=('__pycache__',)),
        'requirements_sha256': hashlib.sha256(req.read_bytes()).hexdigest() if req.exists() else '',
    }


# ───────────────────────────── seeds ─────────────────────────────

SEED = 20261027                 # the global seed; derive others as SEED + k
N_BOOT = 2000                   # person-cluster bootstrap resamples (docs/09 D17: B >= 2000)

# ───────────────────────────── standardisation ─────────────────────────────

CACHE_VERSION = 'stage2_cache_v2'
FS_HZ = 60.0                    # the native browser rate (Chromium 60 Hz pump, WebKit 1/60 s timer)
LOWPASS_HZ = 15.0               # zero-phase Butterworth (order 4) on the 60 Hz grid
LOWPASS_ORDER = 4
MAX_GAP_MS = 100.0              # gaps up to this are interpolated linearly; longer gaps stay NaN
STILL_GYRO_DPS = 20.0           # rest windows of gravity_v1 (= protocol 2.0.0 still_gyro_dps)
REST_MS = 300.0                 # rest window length for gravity_v1 (docs/05 section 2.3)
N_NORM = 64                     # samples of a time-normalised window
FIXED_PRE_S = 0.5               # fixed window around the onset: [onset - 0.5 s, onset + 2.5 s)
FIXED_POST_S = 2.5
QUANTIZE_IOS = False            # Chrome's 0.1-unit rounding applied to iOS values (ablation A3 only)
GAP_WINDOW_PAD_S = 0.5          # rule E4d looks for gaps in [onset - 0.5 s, end + 0.5 s] (docs/02 T6 variant)
PRE_GRAVITY_MS = 300.0          # the up vector before the onset is averaged over [onset - 300 ms, onset] (grip check)

# ───────────────────────────── exclusion thresholds (rule E4) ─────────────────────────────

MIN_DURATION_MS = 250.0         # E4b = the lower duration bound of docs/09 gate G2 (a reach cannot be shorter)
MIN_PEAK_GYRO_DPS = 30.0        # E4c = the peak rule of docs/09 gate G2 (online_v1's onset floor)
MIN_MOTION_HZ = 50.0            # docs/02 T5

# ───────────────────────────── predictor gates of docs/09 D5 (evaluated, not excluded) ─────────────────

G1_MAX_GAP_MS = 50.0            # G1: no sampling gap > 50 ms ...
G1_PAD_S = 0.3                  # ... within [onset - 0.3 s, end + 0.3 s]
G2_MIN_DURATION_MS = 250.0      # G2: 0.25 s <= duration <= 8 s and peak |w| >= 30 deg/s
G2_MAX_DURATION_MS = 8000.0

# ───────────────────────────── flags (docs/02 T8-T12: kept, never excluded) ─────────────────────────────

# T10 clipping: motionlab.qa.CLIP_ACC_MS2 (|accg| > 3.5 g) and CLIP_GYRO_DPS (|w| > 950 deg/s), whole recording
SEG_DISAGREE_MS = 100.0         # T9: offline_v1 and the stored online_v1 onsets differ by more than this

# ───────────────────────────── evaluation ─────────────────────────────

ECE_BINS = 10
K_ATTEMPTS = (1, 2, 3, 4, 5)
INNER_SPLITS = 5                # GroupKFold over the training persons for nested tuning
CAP_PER_PERSON = 40             # docs/09 D2: at most 40 trials of one person enter a training fold (seeded)
LEAK_CHECK = 'all'              # re-run every fold on a random half of its test rows ('sample': LEAK_SAMPLE_FOLDS)
LEAK_SAMPLE_FOLDS = 3
TRAIN_SETS = ('primary',)       # analysis sets run_experiment accepts as rows unless the caller opts in

# ───────────────────────────── label coding ─────────────────────────────

POSITIVE_HAND = 'right'         # y = 1 for the right hand; p = P(right)
HANDS = ('left', 'right')
