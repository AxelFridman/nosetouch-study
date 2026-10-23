"""Subject-wise cross-validation of the baseline left/right classifiers.

Usage (from analysis/, with the venv active):
    python scripts/train_baseline.py data/synthetic
    python scripts/train_baseline.py data/real/2026-11-20 --simulated exclude --augment-mirror --out reports/base_pilot
    python scripts/train_baseline.py data/synthetic --models yaw_rule,logreg,my_pkg.my_mod:make_model

Pipeline: load + validate + deduplicate -> norm_v1 -> analysis set (exclusion rules T1-T7) -> features of
every trial and of its left/right mirror image -> for each CV scheme (development StratifiedGroupKFold,
LOSO, leave-one-orientation-out, leave-one-platform-out) and each model: out-of-fold p(right), metrics
with participant-bootstrap 95 % CIs, mirror consistency; subgroup error table of the development scheme;
a label-permutation leakage control. Writes to --out (default analysis/reports/baseline_<data name>/):
report.md, metrics.csv, subgroups.csv, mirror_consistency.csv, oof_<scheme>.csv, participant_accuracy.csv,
exclusions.csv, features.csv and PNG figures. CSV text cells that a spreadsheet would run as a formula are
written with a leading single quote (``report.write_csv``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import sys
import time
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import pandas as pd  # noqa: E402

from motionlab import NORM_VERSION, __version__, load_dataset  # noqa: E402
from motionlab.augment import quantize_trial  # noqa: E402
from motionlab.evaluate import (  # noqa: E402
    SCHEMES, CVResult, analysis_set, evaluate_models, modeling_table, participant_accuracy,
    shuffle_labels_within, subgroup_table, summary_table,
)
from motionlab.features import extract_features, feature_groups  # noqa: E402
from motionlab.models import DEFAULT_MODELS, get_model_factory  # noqa: E402
from motionlab.report import df_to_markdown, dict_to_markdown, report_dir, write_csv  # noqa: E402

DEFAULT_SCHEMES = ('stratified_group_kfold', 'loso', 'orientation', 'platform')


def _name_of(paths: list[str]) -> str:
    p = Path(paths[0]).resolve()
    stem = p.name[:-len(''.join(p.suffixes))] if p.is_file() and p.suffixes else p.name
    return stem or 'data'


def _fmt_ci(v: Any, lo: Any, hi: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ''
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return f'{v:.3f}'
    return f'{v:.3f} [{lo:.3f}, {hi:.3f}]'


def _scheme_ok(scheme: str, meta: pd.DataFrame) -> str | None:
    """None when the scheme can run on this table, else the reason it is skipped."""
    n_p = meta['participant_id'].nunique()
    if n_p < 2:
        return 'fewer than 2 participants'
    if scheme == 'orientation' and meta['orientation'].nunique() < 2:
        return 'only one orientation'
    if scheme == 'platform' and meta['platform'].astype(object).fillna('unknown').nunique() < 2:
        return 'only one platform'
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Subject-wise CV of the baseline left/right models with a Markdown report.')
    ap.add_argument('data', nargs='+', help='data folder(s), export bundle(s) or zip archive(s)')
    ap.add_argument('--out', help='output directory (default: analysis/reports/baseline_<data name>)')
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS),
                    help=f'comma-separated registry names or module:function factories (default {",".join(DEFAULT_MODELS)})')
    ap.add_argument('--schemes', default=','.join(DEFAULT_SCHEMES), help=f'comma-separated subset of {",".join(SCHEMES)}')
    ap.add_argument('--n-splits', type=int, default=5, help='subject folds (default 5)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=1000, help='participant-bootstrap resamples for the CIs (0 = none)')
    ap.add_argument('--augment-mirror', action='store_true', help='add the mirror image of every TRAINING trial (labels flipped)')
    ap.add_argument('--quantize', choices=['none', 'ios', 'all'], default='none',
                    help="Chrome-style 0.1 quantisation of iOS (or all) trials before the features (ablation A3)")
    ap.add_argument('--segment', choices=['auto', 'offline', 'stored', 'online'], default='auto',
                    help='segment source: auto = offline_v1 with the stored online_v1 as fallback')
    ap.add_argument('--simulated', choices=['include', 'exclude', 'only'], default='include',
                    help='simulated trials (real-data analyses: exclude)')
    ap.add_argument('--require-calibration', action='store_true', help='drop sessions without a calibration (rule S3)')
    ap.add_argument('--no-permutation-check', action='store_true', help='skip the label-permutation leakage control')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    t_start = time.time()

    ds = load_dataset(args.data, simulated=args.simulated)
    say(ds)
    out = Path(args.out) if args.out else report_dir(f'baseline_{_name_of(args.data)}')
    out.mkdir(parents=True, exist_ok=True)
    if not len(ds.trials):
        print('no trials loaded', file=sys.stderr)
        return 1
    keep, excl = analysis_set(ds.trials, simulated=args.simulated, require_calibration=args.require_calibration,
                              sessions=ds.sessions)
    write_csv(excl, out / 'exclusions.csv', index=False)
    ids = list(ds.trials.loc[keep, 'trial_id'])
    say(f'analysis set: {len(ids)} of {len(ds.trials)} attempts')
    if not ids:
        print('no trial passes the exclusion rules (see exclusions.csv)', file=sys.stderr)
        return 1

    transform = None
    if args.quantize != 'none':
        plat = dict(zip(ds.trials['trial_id'], ds.trials['platform']))

        def transform(rec: dict[str, Any]) -> dict[str, Any]:  # noqa: F811
            if args.quantize == 'all' or plat.get(rec['trial_id']) == 'ios':
                return quantize_trial(rec)
            return rec

    say('features (trials and their mirror images) ...')
    F, Fm = extract_features(ds, ids, mirror='both', transform=transform, segment=args.segment)
    write_csv(F, out / 'features.csv')
    X, meta = modeling_table(F, ds.trials)
    Xm = Fm.reindex(X.index)
    say(f'modelling table: {len(X)} trials, {meta["participant_id"].nunique()} participants')
    if meta['participant_id'].nunique() < 2 or meta['y_right'].nunique() < 2:
        print('need >= 2 participants and both hands for subject-wise CV', file=sys.stderr)
        return 1

    factories = {spec.split(':')[-1] if ':' in spec else spec: get_model_factory(spec)
                 for spec in [m.strip() for m in args.models.split(',') if m.strip()]}
    schemes = [s.strip() for s in args.schemes.split(',') if s.strip()]
    for s in schemes:
        if s not in SCHEMES:
            raise SystemExit(f'unknown scheme {s!r}; choose from {SCHEMES}')
    results: dict[str, dict[str, CVResult]] = {}
    skipped: dict[str, str] = {}
    for scheme in schemes:
        why = _scheme_ok(scheme, meta)
        if why:
            skipped[scheme] = why
            continue
        say(f'CV {scheme} ...')
        results[scheme] = evaluate_models(factories, X, meta, scheme=scheme, n_splits=args.n_splits, X_mirror=Xm,
                                          augment_mirror=args.augment_mirror, n_boot=args.n_boot, seed=args.seed)
    if not results:
        print(f'no CV scheme could run: {skipped}', file=sys.stderr)
        return 1

    # tables
    mrows = []
    for scheme, res in results.items():
        st = summary_table(res)
        mrows.append(st)
        oof = pd.concat({name: r.oof for name, r in res.items()}, names=['model', 'trial_id']).reset_index()
        write_csv(oof, out / f'oof_{scheme}.csv', index=False)
    metrics = pd.concat(mrows, ignore_index=True)
    write_csv(metrics, out / 'metrics.csv', index=False)
    primary = schemes[0] if schemes[0] in results else next(iter(results))
    sub_rows, mir_rows, pacc = [], [], []
    for name, r in results[primary].items():
        t = subgroup_table(r.oof, meta, n_boot=args.n_boot, seed=args.seed)
        t.insert(0, 'model', name)
        sub_rows.append(t)
        pa = participant_accuracy(r.oof).reset_index()
        pa.insert(0, 'model', name)
        pacc.append(pa)
    for scheme, res in results.items():
        for name, r in res.items():
            if r.mirror:
                mir_rows.append({'scheme': scheme, 'model': name, **r.mirror})
    subgroups = pd.concat(sub_rows, ignore_index=True)
    write_csv(subgroups, out / 'subgroups.csv', index=False)
    mirror_tab = pd.DataFrame(mir_rows)
    write_csv(mirror_tab, out / 'mirror_consistency.csv', index=False)
    write_csv(pd.concat(pacc, ignore_index=True), out / 'participant_accuracy.csv', index=False)

    perm = None
    if not args.no_permutation_check and 'logreg' in factories:
        y_perm = shuffle_labels_within(meta['y_right'].astype(int).to_numpy(), meta['participant_id'].to_numpy(), args.seed)
        meta_p = meta.assign(y_right=y_perm)
        rp = evaluate_models({'logreg': factories['logreg']}, X, meta_p, scheme=primary, n_splits=args.n_splits,
                             seed=args.seed)['logreg']
        perm = rp.metrics

    # figures
    figures: dict[str, str] = {}
    if not args.no_plots:
        from motionlab import plots
        plots.plot_roc(results[primary], out / 'roc.png')
        figures[f'ROC, {primary}'] = 'roc.png'
        best = max(results[primary], key=lambda k: results[primary][k].metrics.get('accuracy', 0.0))
        plots.plot_subgroups(subgroups[subgroups['model'] == best], out / 'subgroups.png')
        figures[f'Subgroup error rates, {best} ({primary})'] = 'subgroups.png'
        mt = meta.set_index('trial_id')
        feats = [f for f in ('chi_yaw_net_deg', 'chi_yaw_net_norm', 'chi_omega_dot_ahat_dps', 'chi_triple_acc_norm')
                 if f in X.columns]
        plots.plot_feature_by_hand(X, mt.reset_index(), feats, out / 'chiral_by_hand.png')
        figures['Chiral features by hand'] = 'chiral_by_hand.png'
        if 'platform' in mt.columns and mt['platform'].nunique() > 1:
            for p in sorted(mt['platform'].dropna().unique()):
                ids_p = mt.index[mt['platform'] == p]
                plots.plot_feature_by_hand(X.loc[ids_p], mt.loc[ids_p].reset_index(), feats[:2], out / f'chiral_by_hand_{p}.png')
                figures[f'Chiral features by hand, {p}'] = f'chiral_by_hand_{p}.png'

    # report
    groups = feature_groups()
    n_sim = int(meta['simulated'].fillna(False).astype(bool).sum()) if 'simulated' in meta.columns else 0
    lines = [f'# Baseline models: {", ".join(args.data)}\n']
    if n_sim:
        lines.append(f'> **{n_sim} of {len(meta)} modelled trials are SIMULATED.** These numbers test the pipeline; '
                     'they say nothing about real participants.\n')
    lines.append('## Setup\n')
    lines.append(dict_to_markdown({
        'date': _dt.datetime.now().strftime('%Y-%m-%d %H:%M'), 'motionlab': __version__, 'normalisation': NORM_VERSION,
        'models': ', '.join(factories), 'schemes': ', '.join(results), 'subject folds': args.n_splits,
        'seed': args.seed, 'bootstrap resamples (participants)': args.n_boot,
        'mirror augmentation of training folds': args.augment_mirror, 'quantisation': args.quantize,
        'segment source': args.segment, 'simulated trials': args.simulated,
        'participants': int(meta['participant_id'].nunique()), 'trials': int(len(meta)),
        'right-hand trials': int(meta['y_right'].sum()),
        'feature counts': ', '.join(f'{k} {len(v)}' for k, v in groups.items()),
    }))
    if skipped:
        lines.append('Skipped schemes: ' + '; '.join(f'{k} ({v})' for k, v in skipped.items()) + '\n')
    lines.append('## Analysis set\n')
    lines.append('Attempts failing each exclusion rule, applied in order (docs/02 section 12). The modelling table '
                 'then keeps trials with usable features (`qc_ok`) and a supported gravity sign.\n')
    lines.append(df_to_markdown(excl))
    qc = F.loc[F.index.intersection(ids)]
    qc_info = {
        'trials with features (qc_ok)': f"{int((qc['qc_ok'] == 1).sum())} / {len(qc)}",
        'gravity estimate': ', '.join(f'{k} {v}' for k, v in qc['qc_gravity_method'].value_counts().items()),
        'gravity closure, median deg': qc['qc_closure_deg'].median(),
        'gyro gravity vs browser gravity, median deg': qc['qc_gravity_vs_browser_deg'].median(),
        'segment source': ', '.join(f'{k} {v}' for k, v in qc['qc_segment_source'].value_counts().items()),
        'gravity sign source': ', '.join(f'{k} {v}' for k, v in qc['qc_gravity_sign_source'].value_counts().items()),
    }
    reasons = qc.loc[qc['qc_ok'] != 1, 'qc_reason'].value_counts()
    if len(reasons):
        qc_info['unusable trials'] = ', '.join(f'{k} {v}' for k, v in reasons.items())
    lines.append(dict_to_markdown(qc_info))
    lines.append('## Results\n')
    lines.append('Out-of-fold predictions of every test trial; every split is subject-disjoint (checked). '
                 'Brackets: 95 % participant-bootstrap CI. `stratified_group_kfold` (stratified on '
                 'EHI class x platform) is the development estimate; `orientation` holds out one grip (mirror '
                 'pairs together) for unseen people; `platform` trains on one platform and tests on the other.\n')
    show = pd.DataFrame({
        'scheme': metrics['scheme'], 'model': metrics['model'], 'n': metrics['n'].astype(int),
        'participants': metrics['n_participants'].astype(int),
        'accuracy': [_fmt_ci(v, lo, hi) for v, lo, hi in zip(metrics['accuracy'], metrics.get('accuracy_lo', [None] * len(metrics)),
                                                            metrics.get('accuracy_hi', [None] * len(metrics)))],
        'balanced acc.': metrics['balanced_accuracy'],
        'ROC AUC': [_fmt_ci(v, lo, hi) for v, lo, hi in zip(metrics['roc_auc'], metrics.get('roc_auc_lo', [None] * len(metrics)),
                                                           metrics.get('roc_auc_hi', [None] * len(metrics)))],
        'log loss': metrics['log_loss'], 'Brier': metrics['brier'],
        'participant acc. median / p10': [f'{a:.3f} / {b:.3f}' if math.isfinite(a) else '' for a, b in
                                          zip(metrics['participant_acc_median'], metrics['participant_acc_p10'])],
        'participants < 80 %': metrics['n_participants_below_80'],
    })
    lines.append(df_to_markdown(show))
    lines.append('## Mirror consistency\n')
    lines.append('p(mirror(x)) should equal 1 - p(x): a model that learned chirality flips its decision when the '
                 'trial is mirrored (label_flip_rate near 1, mean_abs_dev near 0). Low values mean it relies on '
                 'mirror-even cues such as dominance, speed or platform artefacts.\n')
    lines.append(df_to_markdown(mirror_tab) if len(mirror_tab) else '_(not computed)_\n')
    if perm is not None:
        lines.append('## Leakage control\n')
        lines.append(f'logreg on hand labels shuffled within participants ({primary}): accuracy '
                     f'{perm["accuracy"]:.3f}, ROC AUC {perm["roc_auc"]:.3f}. It must stay near 0.5.\n')
    lines.append(f'## Subgroup errors ({primary})\n')
    lines.append('Error rate per level, with a 95 % CI from resampling the participants of that level.\n')
    for name in results[primary]:
        t = subgroups[subgroups['model'] == name].drop(columns=['model'])
        lines.append(f'### {name}\n')
        lines.append(df_to_markdown(t))
    if figures:
        lines.append('## Figures\n')
        for cap, path in figures.items():
            lines.append(f'### {cap}\n\n![{cap}]({path})\n')
    lines.append(f'\n---\nRun time {time.time() - t_start:.0f} s. CSV files next to this report hold every number above.\n')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    if not args.quiet:
        cols = ['scheme', 'model', 'n', 'accuracy', 'roc_auc', 'log_loss']
        print(metrics[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        if len(mirror_tab):
            print(mirror_tab[['scheme', 'model', 'label_flip_rate', 'mean_abs_dev']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        print(f'report: {out / "report.md"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
