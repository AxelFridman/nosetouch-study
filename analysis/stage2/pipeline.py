"""The Stage 2 pipeline: one command rebuilds, recomputes or redraws any part of the analysis.

    cd analysis
    .venv/Scripts/python.exe -m stage2.pipeline TARGET [--lanes m1,m2] [--only NAME ...] [--force] [--list]
                                                       [--threads N]

TARGET
  all        cache, then compute, figures, tables, numbers for every unit, then report
  cache      build the cache when it is missing or stale (another snapshot, curation, parameter or builder);
             --force rebuilds it anyway
  describe   the descriptive analysis (core unit 'describe': every S2 table and figure, describe_summary.json)
  reference  the reference floor (core unit 'reference': its experiments, figure, tables and numbers)
  compute    every unit's compute step: experiments whose results are missing or were made from another cache
             (--force: all of them; --only: these names)
  figures    every unit's figures step: redraw from saved results only (seconds; never fits a model)
  tables     every unit's tables step (CSV + LaTeX booktabs .tex), then results/registry_all.csv
  numbers    every unit's numbers step (results/<unit>_numbers.json), then results/numbers.tex (LaTeX macros)
  report     the report: runs report/scripts/make_assets.py (copies every registered figure, .tex table and
             numbers file into report/), then latexmk in report/ (--no-latex: the asset step only;
             REPORT_MODE=final: the final-mode guards)

Units are the core units (``cache``, ``describe``, ``reference``: modules under stage2/scripts) and the lanes
discovered from ``stage2/lanes/<id>_pipeline.py``. A unit module exposes

    STEPS = {'compute': fn, 'figures': fn, 'tables': fn, 'numbers': fn}   # any subset
    ORDER = 100                    # run order (core 0-20, model lanes 100-199, synthesis lanes 900)
    DEPENDS = ('m1',)              # optional: units that must run before it (ordering only)

and every step is ``fn(force: bool, only: list[str] | None) -> None``. ``--lanes`` selects units by id (core
ids too: ``--lanes reference,m2``); without it every unit runs. A lane runs alone with
``python -m stage2.pipeline compute --lanes m2``. Each step logs its duration (also to
``reports/stage2/pipeline_log.jsonl``); the first failure stops the run with the unit, the step and the
traceback, and the exit code is 1.

Helpers for lane modules: :class:`Lane` (paths, naming checks, ``run_experiment`` with the lane's registry,
``should_run``, figures, tables, numbers, finalists), :func:`selected`, :func:`result_current`,
:func:`cache_status`, :func:`provenance`, :func:`limit_threads`. This module imports no numpy at import time,
so a lane can limit its threads before numpy loads.

``--threads N`` wins over the lanes' own budgets: ``main`` records it for this process before it imports the
lane modules, and every later ``limit_threads(n)`` of the process then uses N (worker processes a lane starts keep
their own budgets). Staleness is the cache's data identity; a lane may also pin a result to its declared
specification (``should_run(..., spec=...)``; :func:`source_fingerprint` gives a code version for it) and a derived
file to the files it was computed from (``write_json(..., inputs=...)``), see :func:`result_current` and
:func:`file_current`.
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ''):  # run as a file: python analysis/stage2/pipeline.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'stage2'  # noqa: A001

from . import config as C

STEP_NAMES = ('compute', 'figures', 'tables', 'numbers')
TARGETS = ('all', 'cache', 'describe', 'reference', 'compute', 'figures', 'tables', 'numbers', 'report')
StepFn = Callable[[bool, 'list[str] | None'], Any]

#: core units: id -> module (each exposes STEPS and ORDER like a lane)
CORE_UNITS = {'cache': 'stage2.scripts.build_cache', 'describe': 'stage2.scripts.describe',
              'reference': 'stage2.scripts.reference'}
FINALIST_KEYS = ('name', 'results_json', 'module', 'build_x', 'factory', 'params', 'n_configs_tried', 'notes')
_PACKAGE_LANES = Path(__file__).resolve().parent / 'lanes'      # the package's own lane modules (stage2.lanes.*)


class PipelineError(RuntimeError):
    """A unit or step that cannot run (unknown lane, bad module contract, failed step)."""


# ───────────────────────────── threads ─────────────────────────────

_THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS')
#: ``<N>@<pid>``: the --threads N of the pipeline command in process <pid>. An environment variable, because
#: ``python -m stage2.pipeline`` runs this file as ``__main__`` while the lanes import ``stage2.pipeline`` (two module
#: objects); the pid keeps it from leaking into the worker processes a lane starts (they inherit the environment).
THREADS_OVERRIDE_VAR = 'STAGE2_THREADS_OVERRIDE'


def thread_override() -> int | None:
    """The ``--threads N`` of the pipeline command running in THIS process, or None."""
    n, _, pid = os.environ.get(THREADS_OVERRIDE_VAR, '').partition('@')
    return int(n) if n.isdigit() and int(n) >= 1 and pid == str(os.getpid()) else None


def set_thread_override(n: int | None) -> None:
    """Make every later :func:`limit_threads` call of this process use ``n`` (None removes the override)."""
    if n is None:
        os.environ.pop(THREADS_OVERRIDE_VAR, None)
    else:
        if int(n) < 1:
            raise ValueError('--threads needs at least 1 thread')
        os.environ[THREADS_OVERRIDE_VAR] = f'{int(n)}@{os.getpid()}'


def limit_threads(n: int) -> None:
    """Cap the BLAS / OpenMP / torch threads of this process at ``n``. Call it at the top of a lane module,
    before numpy is imported (the environment variables are read once, when the libraries load).

    ``python -m stage2.pipeline --threads N`` wins: once set (:func:`set_thread_override`, done by :func:`main`
    before it imports any lane), every call in this process uses N instead of ``n``."""
    o = thread_override()
    n = int(n) if o is None else o
    for v in _THREAD_VARS:
        os.environ[v] = str(n)
    if 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(n)


# ───────────────────────────── staleness ─────────────────────────────


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def cache_meta() -> dict[str, Any] | None:
    return _read_json(C.CACHE_DIR / 'meta.json')


def cache_data_id() -> str | None:
    """The data identity of the current cache (``meta.json`` ``data_identity.data_id``), None without a cache."""
    m = cache_meta()
    return ((m or {}).get('data_identity') or {}).get('data_id')


def cache_status() -> tuple[str, str]:
    """('missing' | 'stale' | 'current', detail): is the cache what the snapshot, the curation, the cache
    parameters and the builder code would give now (``data.data_identity``)?"""
    if not ((C.CACHE_DIR / 'arrays.npz').exists() and (C.CACHE_DIR / 'trials.pkl').exists()):
        return 'missing', f'no cache in {C.CACHE_DIR}'
    meta = cache_meta() or {}
    have = meta.get('data_identity') or {}
    from .data import data_identity
    now = data_identity()
    if have.get('data_id') == now['data_id']:
        return 'current', f"data_id {now['data_id'][:12]}"
    if not have:
        return 'stale', 'the cache was made before data identities were recorded'
    diff = [k for k in now if k != 'data_id' and have.get(k) != now[k]]
    return 'stale', 'changed: ' + ', '.join(diff)


def provenance() -> dict[str, Any]:
    """Deterministic provenance to embed in a derived JSON file (no timestamp): the cache's data identity."""
    m = cache_meta() or {}
    return {'data_id': cache_data_id(), 'cache_version': m.get('cache_version'),
            'snapshot_sha256': (m.get('snapshot_fingerprint') or {}).get('sha256')}


def result_current(name: str, out_dir: str | Path = C.RES_DIR, *, spec: dict[str, Any] | None = None) -> bool:
    """A saved ``run_experiment`` result exists and was made from the current cache (same data_id).

    ``spec`` (opt-in): the lane's CURRENT declaration of the experiment, any of the keys of
    :func:`stage2.cv.result_spec` (model, params, notes, n_configs_tried, sets, cap_per_person, seed, n_boot,
    leak_check, test_meta_columns) and the lane's own ``spec=`` extras of ``run_experiment`` (for example a code
    version). The result is then current only when every given key equals the saved one, so an edited
    specification is recomputed. A saved result's specification is read from the fields every result stores, so
    this needs no rewrite of existing results; without ``spec`` only the data identity counts (as before)."""
    d = _read_json(Path(out_dir) / f'{name}.json')
    if d is None:
        return False
    rid = ((d.get('info') or {}).get('cache_code') or {}).get('data_id')
    if rid is None or rid != cache_data_id():
        return False
    if spec is not None:
        from .cv import result_spec, spec_matches
        return spec_matches(result_spec(d), spec)
    return True


def _input_key(path: Path) -> str:
    """How a derived file records one of its inputs: relative to analysis/ when inside it, else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(C.ANALYSIS_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _input_path(item: Any) -> Path:
    """A derived file's input: a result name, a module object, or a path. A relative path is taken relative to the
    output folder (``tables/m9/m9_y.json``), or to analysis/ when it exists only there (``stage2/lanes/m9_x.py``)."""
    if hasattr(item, '__file__') and not isinstance(item, (str, Path)):
        return Path(item.__file__)                      # a module object: its source file
    s = str(item)
    if isinstance(item, str) and not any(c in s for c in '/\\') and not Path(s).suffix:
        return C.RES_DIR / f'{s}.json'                  # a result name
    p = Path(item)
    if p.is_absolute():
        return p
    if not (C.OUT_DIR / p).exists() and (C.ANALYSIS_DIR / p).exists():
        return C.ANALYSIS_DIR / p
    return C.OUT_DIR / p


def _sha256_or_none(path: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def input_hashes(inputs: Sequence[Any]) -> dict[str, str | None]:
    """{input: sha256 of its bytes (None: missing)} of the files a derived file is computed from: result names
    (``m9_x`` -> results/m9_x.json), output-relative paths (``tables/m9/m9_y.json``), absolute paths, or source files
    (module objects, or paths relative to analysis/ such as ``stage2/lanes/m9_derive.py``: a code change then makes
    the derived file stale)."""
    out: dict[str, str | None] = {}
    for item in inputs:
        p = _input_path(item)
        out[_input_key(p)] = _sha256_or_none(p)
    return dict(sorted(out.items()))


def source_fingerprint(*items: Any) -> str:
    """A code version for an experiment's declared specification: 16 hex digits of the sha256 over the source files of
    ``items`` (module objects, dotted module names such as ``'stage2.lanes.m9_models'``, or paths, relative ones to
    analysis/), order-independent. Opt-in staleness for code: pass it as ``run_experiment(..., spec={'code': v})`` and
    ``L.should_run(name, force, only, spec={'code': v})``, and an edit of that code recomputes the result (any edit,
    a comment included: fingerprint only the modules that define the model)."""
    import hashlib
    import importlib.util as iu
    files: dict[str, bytes] = {}
    for it in items:
        if isinstance(it, str) and '/' not in it and '\\' not in it and not it.endswith('.py'):
            spec = iu.find_spec(it)                     # a dotted module name, located without importing it
            if spec is None or not spec.origin:
                raise ModuleNotFoundError(f'source_fingerprint: no module {it!r}')
            p = Path(spec.origin)
        else:
            p = _input_path(it) if not isinstance(it, (str, Path)) else Path(it)
            if not p.is_absolute():
                p = C.ANALYSIS_DIR / p
        files[_input_key(p)] = Path(p).read_bytes()
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(k.encode() + b'\0' + hashlib.sha256(files[k]).digest())
    return h.hexdigest()[:16]


def file_current(path: str | Path) -> bool:
    """A derived JSON file exists and its ``provenance.data_id`` is the current cache's (see :func:`provenance`).

    A file written with ``Lane.write_json(..., inputs=[...])`` also records the sha256 of every input
    (``provenance_inputs``); it is current only while each input still has those bytes (a re-run result, a changed
    selection file or an edited source makes it stale). Files without that record are judged by the data identity
    alone, as before."""
    d = _read_json(Path(path))
    now = cache_data_id()
    if not (bool(d) and now is not None and (d.get('provenance') or {}).get('data_id') == now):
        return False
    rec = d.get('provenance_inputs')
    if rec:
        root = C.ANALYSIS_DIR
        return all(_sha256_or_none(Path(k) if Path(k).is_absolute() else root / k) == v for k, v in rec.items())
    return True


def selected(name: str, only: Sequence[str] | None) -> bool:
    """True when ``only`` is None or ``name`` matches one of its names / shell patterns (``m2_*``)."""
    return only is None or any(fnmatch.fnmatchcase(name, p) for p in only)


def should_run(name: str, force: bool, only: Sequence[str] | None, out_dir: str | Path = C.RES_DIR, *,
               spec: dict[str, Any] | None = None) -> bool:
    """Compute an experiment? Selected by ``only`` and (``force`` or no current result; with ``spec``: a result made
    from another specification is not current, see :func:`result_current`)."""
    return selected(name, only) and (force or not result_current(name, out_dir, spec=spec))


# ───────────────────────────── lanes ─────────────────────────────


@dataclass(frozen=True)
class Lane:
    """Paths and conventions of lane ``id`` (``m1``, ``x1``, ...)::

        figures/<id>/<ID>_NN_short_name.pdf|png (+ captions.json)    tables/<id>/<name>.csv|.tex
        results/<id>_*.json (experiments)   results/registry_<id>.csv   results/<id>_numbers.json
        results/finalists_<id>.json         reports/stage2/<id>_SUMMARY.md
    """
    id: str

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace('_', '').isalnum() or self.id != self.id.lower():
            raise ValueError(f'lane id {self.id!r}: lower-case letters and digits')

    # paths
    @property
    def fig_dir(self) -> Path:
        return C.FIG_DIR / self.id

    @property
    def tab_dir(self) -> Path:
        return C.TAB_DIR / self.id

    @property
    def res_dir(self) -> Path:
        return C.RES_DIR

    @property
    def registry(self) -> Path:
        return C.RES_DIR / f'registry_{self.id}.csv'

    @property
    def numbers_path(self) -> Path:
        return C.RES_DIR / f'{self.id}_numbers.json'

    @property
    def finalists_path(self) -> Path:
        return C.RES_DIR / f'finalists_{self.id}.json'

    @property
    def summary_path(self) -> Path:
        return C.OUT_DIR / f'{self.id}_SUMMARY.md'

    def result_path(self, name: str) -> Path:
        return C.RES_DIR / f'{self.check_name(name)}.json'

    # naming
    def check_name(self, name: str) -> str:
        if not name.startswith(f'{self.id}_'):
            raise ValueError(f'lane {self.id}: result and table names start with {self.id}_ (got {name!r})')
        return name

    def check_figure(self, name: str) -> str:
        if not name.startswith(f'{self.id.upper()}_'):
            raise ValueError(f'lane {self.id}: figure names are {self.id.upper()}_NN_short_name (got {name!r})')
        return name

    # compute
    def should_run(self, name: str, force: bool, only: Sequence[str] | None, *,
                   spec: dict[str, Any] | None = None) -> bool:
        """:func:`should_run` for a lane result; ``spec`` (opt-in) also recomputes a result whose saved
        specification differs from the lane's current one (:func:`result_current`)."""
        return should_run(self.check_name(name), force, only, spec=spec)

    def run_experiment(self, name: str, fit_predict: Any, X: Any, meta: Any, **kw: Any) -> Any:
        """``cv.run_experiment`` with the lane's registry (results/registry_<id>.csv) and name check."""
        from . import cv
        kw.setdefault('registry', self.registry)
        return cv.run_experiment(self.check_name(name), fit_predict, X, meta, **kw)

    def load_result(self, name: str) -> Any:
        from . import cv
        return cv.load_result(self.check_name(name))

    def write_json(self, name: str, obj: Any, *, folder: str = 'tables',
                   inputs: Sequence[str | Path] | None = None) -> Path:
        """A derived JSON file (curves, importances, figure data) with the cache provenance embedded, in
        tables/<id>/ (or results/ with folder='results'); check it with :func:`file_current`. Written atomically.

        ``inputs`` (opt-in): the files it is computed from (result names, output-relative paths, source files);
        their sha256 is recorded (``provenance_inputs``) and :func:`file_current` then also requires them unchanged."""
        p = (self.tab_dir if folder == 'tables' else C.RES_DIR) / f'{self.check_name(name)}.json'
        doc = {'provenance': provenance(), **obj}
        if inputs is not None:
            if 'provenance_inputs' in obj:
                raise ValueError("'provenance_inputs' is reserved for the recorded inputs")
            doc['provenance_inputs'] = input_hashes(list(inputs))
        from .report_export import write_text_atomic
        return write_text_atomic(p, json.dumps(doc, indent=1, default=_json_default, allow_nan=False) + '\n')

    def read_json(self, name: str, *, folder: str = 'tables') -> dict[str, Any]:
        p = (self.tab_dir if folder == 'tables' else C.RES_DIR) / f'{self.check_name(name)}.json'
        d = _read_json(p)
        if d is None:
            raise FileNotFoundError(f'{p}: run the compute step first (python -m stage2.pipeline compute --lanes {self.id})')
        return d

    # presentation
    def save_fig(self, fig: Any, name: str, *, caption: str, decision: str = '', data: Any = None) -> dict[str, str]:
        """plotting.save_fig into figures/<id>/; ``data`` (a tidy DataFrame of what is plotted) is written to
        tables/<id>/<name>.csv."""
        from .plotting import save_fig
        out = save_fig(fig, self.check_figure(name), caption=caption, decision=decision, fig_dir=self.fig_dir)
        if data is not None:
            self.figure_data(data, name)
        return out

    def figure_data(self, df: Any, name: str) -> Path:
        """The plotted data of figure ``name`` as a tidy CSV (tables/<id>/<name>.csv, privacy-checked, written
        atomically; refused when it would overwrite a report table's CSV: ``report_export.check_table_kind``)."""
        from .report_export import _refuse_columnless, check_table_kind, report_safe, write_csv_atomic
        p = self.tab_dir / f'{self.check_figure(name)}.csv'
        _refuse_columnless(df, name)
        check_table_kind(p, 'data')
        return write_csv_atomic(report_safe(df), p, index=False)

    def export_table(self, df: Any, name: str, *, caption: str = '', label: str = '', column_formats: Any = None,
                     **kw: Any) -> dict[str, Path]:
        """tables/<id>/<name>.csv + .tex (report_export.export_table); label defaults to tab:<name>. With
        ``tex=False`` it writes derived data (a CSV only); either kind refuses to overwrite the other's CSV."""
        from .report_export import export_table
        return export_table(df, self.tab_dir / self.check_name(name), caption=caption,
                            label=label or f'tab:{name}', column_formats=column_formats, **kw)

    def write_numbers(self, entries: dict[str, dict[str, Any]]) -> Path:
        """results/<id>_numbers.json (entries from ``report_export.numbers_entry``)."""
        from .report_export import write_numbers
        return write_numbers(self.numbers_path, entries)

    def write_finalists(self, finalists: Sequence[dict[str, Any]]) -> Path:
        """results/finalists_<id>.json: the lane's chosen models, each {name, results_json, module, build_x,
        factory, params, n_configs_tried, notes} (module + factory rebuild the fit_predict, build_x the input)."""
        rows = []
        for f in finalists:
            missing = [k for k in FINALIST_KEYS if k not in f]
            if missing:
                raise ValueError(f'finalist {f.get("name")!r} lacks {missing}')
            self.check_name(f['name'])
            if not (C.RES_DIR / Path(f['results_json']).name).exists():
                raise FileNotFoundError(f"finalist {f['name']}: {f['results_json']} does not exist")
            rows.append({k: f[k] for k in FINALIST_KEYS} | {k: v for k, v in f.items() if k not in FINALIST_KEYS})
        from .report_export import write_text_atomic
        return write_text_atomic(self.finalists_path, json.dumps(rows, indent=1, default=_json_default) + '\n')


def _json_default(v: Any) -> Any:
    if hasattr(v, 'tolist'):
        return v.tolist()
    if hasattr(v, 'item'):
        return v.item()
    if isinstance(v, Path):
        return v.as_posix()
    return str(v)


# ───────────────────────────── units ─────────────────────────────


@dataclass
class Unit:
    id: str
    kind: str                       # 'core' or 'lane'
    module: str
    order: int = 100
    steps: dict[str, StepFn] = field(default_factory=dict)
    depends: tuple[str, ...] = ()
    doc: str = ''


def lane_ids(lanes_dir: str | Path = C.LANES_DIR) -> list[str]:
    """Lane ids from the files lanes/<id>_pipeline.py (not imported)."""
    d = Path(lanes_dir)
    return sorted(p.name[: -len('_pipeline.py')] for p in d.glob('*_pipeline.py') if p.is_file()) if d.exists() else []


def load_unit(uid: str, lanes_dir: str | Path = C.LANES_DIR) -> Unit:
    """Import a unit's module and check its contract."""
    if uid in CORE_UNITS:
        modname, kind = CORE_UNITS[uid], 'core'
    else:
        path = Path(lanes_dir) / f'{uid}_pipeline.py'
        if not path.exists():
            raise PipelineError(f'unknown unit {uid!r}: core units {sorted(CORE_UNITS)}, lanes {lane_ids(lanes_dir)}')
        modname, kind = f'stage2.lanes.{uid}_pipeline', 'lane'
        if Path(lanes_dir).resolve() != _PACKAGE_LANES.resolve():   # lanes from another folder (tests, STAGE2_LANES)
            spec = importlib.util.spec_from_file_location(f'_stage2_lane_{uid}_pipeline', path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return _unit_from_module(uid, kind, mod.__name__, mod)
    try:
        mod = importlib.import_module(modname)
    except Exception as e:  # noqa: BLE001 - report which unit failed to import
        raise PipelineError(f'unit {uid}: importing {modname} failed: {type(e).__name__}: {e}') from e
    return _unit_from_module(uid, kind, modname, mod)


def _unit_from_module(uid: str, kind: str, modname: str, mod: Any) -> Unit:
    steps = getattr(mod, 'STEPS', None)
    if not isinstance(steps, dict) or not steps:
        raise PipelineError(f'unit {uid}: {modname} must define STEPS = {{step: fn}} with steps from {STEP_NAMES}')
    bad = set(steps) - set(STEP_NAMES)
    if bad:
        raise PipelineError(f'unit {uid}: unknown steps {sorted(bad)} (allowed: {STEP_NAMES})')
    for s, fn in steps.items():
        if not callable(fn):
            raise PipelineError(f'unit {uid}: STEPS[{s!r}] is not callable')
    order = getattr(mod, 'ORDER', 100)
    depends = tuple(getattr(mod, 'DEPENDS', ()) or ())
    doc = ((mod.__doc__ or '').strip().splitlines() or [''])[0]
    return Unit(uid, kind, modname, int(order), dict(steps), depends, doc)


def discover(lanes: Sequence[str] | None = None, *, include_core: bool = True,
             lanes_dir: str | Path = C.LANES_DIR) -> list[Unit]:
    """The selected units in run order: ORDER, then id, with every DEPENDS before its dependants."""
    ids = list(lanes) if lanes is not None else ([*CORE_UNITS] if include_core else []) + lane_ids(lanes_dir)
    units = {u: load_unit(u, lanes_dir) for u in dict.fromkeys(ids)}
    known = set(CORE_UNITS) | set(lane_ids(lanes_dir))
    for u in units.values():
        unknown = [d for d in u.depends if d not in known]
        if unknown:
            raise PipelineError(f'unit {u.id} depends on unknown unit(s) {unknown}')
    out: list[Unit] = []
    pending = sorted(units.values(), key=lambda u: (u.order, u.id))
    while pending:
        for i, u in enumerate(pending):
            if all(d not in units or d in {x.id for x in out} for d in u.depends):
                out.append(pending.pop(i))
                break
        else:
            raise PipelineError(f'circular DEPENDS among {[u.id for u in pending]}')
    return out


# ───────────────────────────── running ─────────────────────────────


@dataclass
class StepRecord:
    unit: str
    step: str
    status: str
    seconds: float


def _log(rec: dict[str, Any]) -> None:
    """Append one line to reports/stage2/pipeline_log.jsonl under its cross-process lock (``stage2.locking``), so
    pipelines running in parallel never interleave or lose a line. Logging never stops a run."""
    from .locking import file_lock
    try:
        C.PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(C.PIPELINE_LOG, timeout=60):
            with open(C.PIPELINE_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
    except OSError:                                   # includes locking.LockTimeout (a TimeoutError)
        pass


def run_step(unit: Unit, step: str, *, force: bool, only: list[str] | None, target: str = '') -> StepRecord | None:
    fn = unit.steps.get(step)
    if fn is None:
        return None
    label = f'{unit.id}.{step}'
    print(f'[pipeline] {label} ...' + (' (forced)' if force else '') + (f' only={only}' if only else ''), flush=True)
    t0 = time.time()
    started = time.strftime('%Y-%m-%dT%H:%M:%S')
    try:
        fn(force, only)
    except BaseException as e:
        dt = time.time() - t0
        _log({'started': started, 'target': target, 'unit': unit.id, 'step': step, 'status': 'failed',
              'seconds': round(dt, 1), 'error': f'{type(e).__name__}: {e}'})
        if isinstance(e, KeyboardInterrupt):
            raise
        tb = traceback.format_exc()
        raise PipelineError(f'{label} FAILED after {dt:.1f} s: {type(e).__name__}: {e}\n{tb}') from e
    dt = time.time() - t0
    print(f'[pipeline] {label} ok ({dt:.1f} s)', flush=True)
    _log({'started': started, 'target': target, 'unit': unit.id, 'step': step, 'status': 'ok',
          'seconds': round(dt, 1), 'force': force, 'only': only})
    return StepRecord(unit.id, step, 'ok', dt)


def merge_registries(res_dir: str | Path = C.RES_DIR) -> Path | None:
    """results/registry_all.csv: the core registry and every lane registry, one row per result name."""
    import pandas as pd
    files = [p for p in [Path(res_dir) / 'registry.csv', *sorted(Path(res_dir).glob('registry_*.csv'))]
             if p.exists() and p.name != 'registry_all.csv']
    if not files:
        return None
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.insert(0, 'registry', f.stem)
        frames.append(df)
    reg = pd.concat(frames, ignore_index=True).drop_duplicates('name', keep='last').sort_values('name')
    out = Path(res_dir) / 'registry_all.csv'
    from .locking import atomic_path
    with atomic_path(out) as tmp:                   # a reader never sees a half-written merged registry
        reg.to_csv(tmp, index=False)
    return out


def _stage(units: Sequence[Unit], step: str, *, force: bool, only: list[str] | None, target: str,
           records: list[StepRecord]) -> None:
    for u in units:
        if u.id == 'cache' and step == 'compute':
            # 'compute --force' recomputes experiments; only 'cache --force' and 'all --force' rebuild the cache
            rec = run_step(u, step, force=force and target in ('cache', 'all'), only=None, target=target)
        else:
            rec = run_step(u, step, force=force, only=only, target=target)
        if rec:
            records.append(rec)


def run(target: str, *, lanes: Sequence[str] | None = None, only: Sequence[str] | None = None, force: bool = False,
        lanes_dir: str | Path = C.LANES_DIR) -> list[StepRecord]:
    """Run ``target`` (see the module docstring) and return the steps that ran."""
    if target not in TARGETS:
        raise PipelineError(f'unknown target {target!r}: {TARGETS}')
    only = list(only) if only else None
    records: list[StepRecord] = []
    t0 = time.time()
    if target in ('cache', 'describe', 'reference'):
        if lanes:
            raise PipelineError(f"'{target}' is a core unit; --lanes applies to compute / figures / tables / numbers / all")
        units = discover([target], lanes_dir=lanes_dir)
        steps = ('compute',) if target == 'cache' else STEP_NAMES
        for s in steps:
            _stage(units, s, force=force, only=only, target=target, records=records)
        if target != 'cache':
            _after_stage(target, steps, records)
    elif target == 'report':
        _report(records)
    else:
        units = discover(lanes, lanes_dir=lanes_dir)
        if target == 'all':
            for s in STEP_NAMES:
                _stage(units, s, force=force, only=only, target=target, records=records)
                _after_stage(target, (s,), records)
            if lanes is None:
                _report(records)
        else:
            _stage(units, target, force=force, only=only, target=target, records=records)
            _after_stage(target, (target,), records)
    print(f'[pipeline] {target}: {len(records)} step(s) in {time.time() - t0:.1f} s', flush=True)
    return records


def _after_stage(target: str, steps: Sequence[str], records: list[StepRecord]) -> None:
    from . import report_export as RX
    if 'tables' in steps:
        p = merge_registries()
        if p:
            print(f'[pipeline] merged registries -> {p.relative_to(C.OUT_DIR).as_posix()}', flush=True)
    if 'numbers' in steps and RX.numbers_files():
        RX.numbers_to_tex(None, C.RES_DIR / 'numbers.tex')
        print(f'[pipeline] {len(RX.numbers_files())} numbers file(s) -> results/numbers.tex', flush=True)


#: the LaTeX report (report/ at the repository root; STAGE2_REPORT_DIR overrides it)
REPORT_DIR = Path(os.environ.get('STAGE2_REPORT_DIR', Path(__file__).resolve().parents[2] / 'report'))
_REPORT_OPTS = {'latex': True}


def _latex_env() -> tuple[str | None, dict[str, str]]:
    """latexmk and an environment that finds pdflatex, biber and perl (MiKTeX's per-user install and Git's perl
    are appended to PATH only when they are not found already)."""
    import shutil
    env = dict(os.environ)
    exe = shutil.which('latexmk')
    if exe is None:
        miktex = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'MiKTeX' / 'miktex' / 'bin' / 'x64'
        if (miktex / 'latexmk.exe').exists():
            env['PATH'] = str(miktex) + os.pathsep + env.get('PATH', '')
            exe = str(miktex / 'latexmk.exe')
    if exe and shutil.which('perl', path=env.get('PATH')) is None:
        git_perl = Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'Git' / 'usr' / 'bin'
        if (git_perl / 'perl.exe').exists():
            env['PATH'] = env.get('PATH', '') + os.pathsep + str(git_perl)   # last: keeps Windows' find / sort first
    return exe, env


def _report(records: list[StepRecord]) -> None:
    """The report: the asset step (report/scripts/make_assets.py), then latexmk unless --no-latex."""
    import subprocess
    mode = os.environ.get('REPORT_MODE', 'draft')
    script = REPORT_DIR / 'scripts' / 'make_assets.py'
    if not script.exists():
        raise PipelineError(f'report: {script} does not exist')
    t0 = time.time()
    r = subprocess.run([sys.executable, str(script), '--config', 'report.toml', '--mode', mode], cwd=REPORT_DIR)
    dt = time.time() - t0
    records.append(StepRecord('report', 'assets', 'ok' if r.returncode == 0 else 'failed', dt))
    if r.returncode != 0:
        raise PipelineError(f'report: the asset step failed (exit {r.returncode}, mode {mode})')
    print(f'[pipeline] report: assets copied into {REPORT_DIR} ({dt:.1f} s)', flush=True)
    if not _REPORT_OPTS['latex']:
        print('[pipeline] report: --no-latex, LaTeX skipped', flush=True)
        return
    exe, env = _latex_env()
    if exe is None:
        print('[pipeline] report: latexmk not found (install MiKTeX or TeX Live); LaTeX skipped', flush=True)
        records.append(StepRecord('report', 'latex', 'skipped', 0.0))
        return
    env['REPORT_SKIP_ASSETS'] = '1'                       # the asset step just ran
    env['REPORT_MODE'] = mode
    t0 = time.time()
    r = subprocess.run([exe], cwd=REPORT_DIR, env=env)
    dt = time.time() - t0
    records.append(StepRecord('report', 'latex', 'ok' if r.returncode == 0 else 'failed', dt))
    if r.returncode != 0:
        raise PipelineError(f'report: latexmk failed (exit {r.returncode}); see {REPORT_DIR / "build" / "main.log"}')
    print(f'[pipeline] report: {REPORT_DIR / "main.pdf"} ({dt:.1f} s)', flush=True)


def describe_units(units: Sequence[Unit]) -> str:
    lines = [f"{'unit':<11}{'kind':<6}{'order':>6}  {'steps':<34}{'depends':<14}module"]
    for u in units:
        lines.append(f"{u.id:<11}{u.kind:<6}{u.order:>6}  {','.join(s for s in STEP_NAMES if s in u.steps):<34}"
                     f"{','.join(u.depends) or '-':<14}{u.module}")
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='python -m stage2.pipeline', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target', nargs='?', default=None, choices=TARGETS)
    ap.add_argument('--lanes', default=None, help='comma-separated unit ids (lanes or core units); default: all')
    ap.add_argument('--only', nargs='+', default=None, metavar='NAME',
                    help='only these experiment / figure / table names (shell patterns allowed)')
    ap.add_argument('--force', action='store_true', help='recompute even when the saved outputs are current')
    ap.add_argument('--list', action='store_true', help='list the units and their steps, then exit')
    ap.add_argument('--threads', type=int, default=None,
                    help="BLAS/OpenMP/torch threads of this process (set before numpy loads; wins over the lanes' "
                         'own limit_threads budgets)')
    ap.add_argument('--no-latex', action='store_true',
                    help="report (and the end of all): run the asset step only, not latexmk")
    a = ap.parse_args(argv)
    _REPORT_OPTS['latex'] = not a.no_latex
    if a.threads:
        set_thread_override(a.threads)             # before discover() imports the lanes: their limit_threads(n) obey it
        limit_threads(a.threads)
    lanes = [x.strip() for x in a.lanes.split(',') if x.strip()] if a.lanes else None
    try:
        if a.list:
            print(describe_units(discover(lanes)))
            st, why = cache_status()
            print(f'\ncache: {st} ({why})')
            return 0
        if a.target is None:
            ap.error('a target is required (or --list)')
        run(a.target, lanes=lanes, only=a.only, force=a.force)
    except PipelineError as e:
        print(f'[pipeline] {e}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
