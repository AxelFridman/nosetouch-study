"""Plot one or more trials: raw signals with the online / offline segmentation markers.

Usage (from analysis/, with the venv active):
    python scripts/plot_trial.py data/synthetic --trial 3f2c9a1e
    python scripts/plot_trial.py data/synthetic --trial <id> --trial <id2> --out reports/trials
    python scripts/plot_trial.py data/synthetic --list 20          # print trial ids to choose from

--trial takes a full trial_id or a unique prefix. Figures go to --out (default analysis/reports/trials/)
as trial_<trial_id>.png. Panels: |w| and |a| as online_v1 sees them (with its thresholds), the
normalised acceleration axes, the angular velocity axes and the gravity-referenced yaw rate; vertical
lines mark GO, the stored online_v1 onset / end, this package's re-run of online_v1, offline_v1 and the
end cue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from motionlab import load_dataset  # noqa: E402
from motionlab.report import report_dir  # noqa: E402


def resolve_trial(ids: list[str], wanted: str) -> str:
    """Full trial id from an id or a unique prefix (ValueError otherwise)."""
    if wanted in ids:
        return wanted
    hits = [t for t in ids if t.startswith(wanted)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f'no trial id starts with {wanted!r}')
    raise ValueError(f'{wanted!r} matches {len(hits)} trials: {", ".join(hits[:5])} ...')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Plot trials with their online_v1 / offline_v1 segmentation markers.')
    ap.add_argument('data', nargs='+', help='data folder(s), export bundle(s) or zip archive(s)')
    ap.add_argument('--trial', action='append', default=[], help='trial id or unique prefix (repeatable)')
    ap.add_argument('--list', type=int, metavar='N', help='print the first N trial ids with their condition and exit')
    ap.add_argument('--out', help='output directory (default: analysis/reports/trials)')
    args = ap.parse_args(argv)
    if not args.trial and args.list is None:
        ap.error('give --trial <id> (or --list N to see the ids)')
    ds = load_dataset(args.data, validate=False)
    ids = ds.trial_ids
    if args.list is not None:
        cols = ['trial_id', 'participant_id', 'hand', 'orientation', 'speed', 'variant', 'status', 'platform']
        print(ds.trials[cols].head(args.list).to_string(index=False))
        return 0
    from motionlab.plots import plot_trial

    out = Path(args.out) if args.out else report_dir('trials')
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for want in args.trial:
        try:
            tid = resolve_trial(ids, want)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            rc = 1
            continue
        path = out / f'trial_{tid}.png'
        plot_trial(ds.record(tid, compact=True), norm=ds.trial_normalization(tid), path=path)
        print(path)
    return rc


if __name__ == '__main__':
    sys.exit(main())
