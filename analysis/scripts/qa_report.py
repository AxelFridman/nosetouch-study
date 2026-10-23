"""Data-quality report of a NoseTouch dataset.

Usage (from analysis/, with the venv active):
    python scripts/qa_report.py data/synthetic
    python scripts/qa_report.py data/real/2026-11-20 nosetouch_all_20261120-1530.json --out reports/qa_pilot

Inputs: any mix of folder trees (<root>/<study>/<participant>/<session>/{session.json, trial_*.json}),
export bundles (.json / .json.txt), upload envelopes, zip archives. Writes to --out (default
analysis/reports/qa_<data name>/): qa_report.md (summary, warnings, tables, figures), tables/*.csv,
summary.json and PNG figures. Exit code 1 when no trial could be loaded. Text cells that a spreadsheet
would run as a formula (= + - @ ...) are written with a leading single quote (``report.write_csv``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from motionlab import __version__, load_dataset  # noqa: E402
from motionlab.qa import qa_summary  # noqa: E402
from motionlab.report import report_dir, write_csv  # noqa: E402


def _name_of(paths: list[str]) -> str:
    p = Path(paths[0]).resolve()
    stem = p.name[:-len(''.join(p.suffixes))] if p.is_file() and p.suffixes else p.name
    return stem or 'data'


def _jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, 'item'):
        return _jsonable(v.item())
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Data-quality report of a NoseTouch dataset (folder trees and/or export bundles).')
    ap.add_argument('data', nargs='+', help='data folder(s), export bundle(s) or zip archive(s)')
    ap.add_argument('--out', help='output directory (default: analysis/reports/qa_<data name>)')
    ap.add_argument('--simulated', choices=['include', 'exclude', 'only'], default='include',
                    help='simulated trials: include (default), exclude (real data only) or only')
    ap.add_argument('--validate', choices=['fast', 'full', 'none'], default='fast', help='schema validation mode')
    ap.add_argument('--no-online-check', action='store_true', help='skip re-running online_v1 against the stored indices')
    ap.add_argument('--no-offline', action='store_true', help='skip offline_v1')
    ap.add_argument('--sample', type=int, default=200, help='trials sampled for the slower consistency / gravity checks')
    ap.add_argument('--no-plots', action='store_true', help='do not draw figures')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    validate = False if args.validate == 'none' else args.validate
    ds = load_dataset(args.data, validate=validate, simulated=args.simulated)
    out = Path(args.out) if args.out else report_dir(f'qa_{_name_of(args.data)}')
    out.mkdir(parents=True, exist_ok=True)
    rep = qa_summary(ds, check_online=not args.no_online_check, offline=not args.no_offline,
                     consistency_sample=args.sample, gravity_sample=args.sample)
    figures: dict[str, str] = {}
    if not args.no_plots and len(ds.trials):
        from motionlab import plots
        plots.plot_sampling(ds.trials, out / 'sampling.png')
        plots.plot_counts(ds.trials, out / 'counts.png')
        plots.plot_durations(ds.trials, out / 'durations.png')
        figures = {'Sampling rate and largest gap per trial': 'sampling.png',
                   'Accepted trials per orientation and hand': 'counts.png',
                   'online_v1 movement duration by speed': 'durations.png'}
    title = f'Dataset QA: {", ".join(args.data)}'
    md = rep.to_markdown(title=title, figures=figures)
    md += f'\n---\nmotionlab {__version__}; simulated={args.simulated}; validation={args.validate}.\n'
    (out / 'qa_report.md').write_text(md, encoding='utf-8')
    tdir = out / 'tables'
    tdir.mkdir(exist_ok=True)
    for name, df in rep.tables.items():
        write_csv(df, tdir / f'{name}.csv', index=True)  # formula-escaped: record strings are untrusted
    (out / 'summary.json').write_text(json.dumps({k: _jsonable(v) for k, v in rep.summary.items()}, indent=2),
                                      encoding='utf-8')
    if not args.quiet:
        print(ds)
        for k, v in rep.summary.items():
            print(f'  {k}: {v:.3f}' if isinstance(v, float) else f'  {k}: {v}')
        for w in rep.warnings:
            print(f'  ! {w}')
        print(f'report: {out / "qa_report.md"}')
    return 0 if len(ds.trials) else 1


if __name__ == '__main__':
    sys.exit(main())
