"""The reference floor of the registry and the label-noise sensitivity analyses: core unit 'reference'.

    cd analysis
    python -m stage2.pipeline reference                  # compute (skips current results), figures, tables, numbers
    python -m stage2.pipeline figures --lanes reference  # redraw its figure from the saved results (seconds)
    python -m stage2.scripts.reference                   # the same as 'pipeline reference' (kept for old commands)

This module is the WORKED EXAMPLE of the lane contract (README 'Pipeline and lane conventions'): ``compute``
runs the experiments through ``cv.run_experiment`` and saves everything the presentation needs; ``figures``
(stage2/scripts/reference_figures.py), ``tables`` and ``numbers`` only read those saved files.

Experiments (leave-one-person-out, every fold leak-checked, docs/09 D2 cap, natural-use trials scored apart):

* ``example_yaw_logreg`` - the README example and the sanity floor of the registry: logistic regression on
  the cumulative yaw about gravity at 8 instants of the approach, mirror-augmented (pre-specified);
* ``sens_swap_blocks_train_only`` - the same, without the blocks that show the swap signature
  (``screen.block_consistency``; data-driven) in TRAINING only, so the evaluation set is unchanged;
* ``sens_swap_blocks_removed`` - the same blocks removed from training AND evaluation;
* ``sens_d6_screen`` - the docs/09 D6 screen inside every fold removes flagged training trials.

Saved by compute: results/<name>.json and registry rows, tables/sensitivity_label_screen.csv (paired
comparisons against the reference) and tables/reference_summary.json (everything the figure, the tables and
the numbers read). tables: tables/reference/reference_models.csv|.tex and reference_k_attempts.csv|.tex.
numbers: results/reference_numbers.json. About 1 minute for compute; seconds for the rest.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # run as a file
    import _bootstrap  # type: ignore # noqa: F401

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stage2 import config as C
from stage2 import cv, screen
from stage2 import pipeline as P
from stage2 import report_export as RX
from stage2.standardize import mirror_channels

ORDER = 20
REFERENCE = 'example_yaw_logreg'
SENSITIVITY = ('sens_swap_blocks_train_only', 'sens_swap_blocks_removed', 'sens_d6_screen')
EXPERIMENTS = (REFERENCE, *SENSITIVITY)
SUMMARY_JSON = C.TAB_DIR / 'reference_summary.json'          # the saved data every presentation step reads
SENS_TABLE = C.TAB_DIR / 'sensitivity_label_screen.csv'
TAB_DIR = C.TAB_DIR / 'reference'                            # report tables (CSV + .tex) of this unit
NUMBERS_JSON = C.RES_DIR / 'reference_numbers.json'
LABELS = {REFERENCE: 'reference (yaw logreg)', 'sens_swap_blocks_train_only': 'swap blocks out of training',
          'sens_swap_blocks_removed': 'swap blocks out of training and test', 'sens_d6_screen': 'D6 screen in every fold'}


# ───────────────────────────── the model ─────────────────────────────


def yaw_logreg(channels: tuple[str, ...], drop_blocks: list[tuple[str, int]] | None = None, d6: bool = False):
    """fit_predict of the reference model, optionally without some training blocks or with the D6 screen."""
    k = channels.index('yaw_cum')

    def feat(X: np.ndarray) -> np.ndarray:
        return X[:, k, ::8]                              # cumulative yaw about gravity at 8 instants

    def fit_predict(fold: cv.Fold) -> tuple[np.ndarray, dict[str, Any]]:
        keep = np.ones(len(fold.y_train), dtype=bool)
        if drop_blocks:
            bi = pd.to_numeric(fold.meta_train['block_index'], errors='coerce').fillna(-1).astype(int)
            idx = pd.MultiIndex.from_arrays([fold.meta_train['session_label'], bi])
            keep &= ~idx.isin(drop_blocks)
        if d6:
            fl = screen.wrong_hand_flags(screen.screen_features(fold.X_train), fold.y_train, fold.groups_train,
                                         fold.meta_train)
            keep &= ~fl['flag'].to_numpy()
        X, y = fold.X_train[keep], fold.y_train[keep]
        Xtr = np.concatenate([X, mirror_channels(X)])   # mirror augmentation: the mirrored trial, other hand
        ytr = np.concatenate([y, 1 - y])
        m = LogisticRegression(max_iter=1000).fit(feat(Xtr), ytr)
        return m.predict_proba(feat(fold.X_test))[:, 1], {'n_train_dropped': int((~keep).sum())}

    return fit_predict


def swap_blocks(d: Any) -> list[tuple[str, int]]:
    """Blocks whose trials all contradict their label by the first-half yaw sign while the neighbouring block
    of the other hand does too (``screen.block_consistency``): data-driven, never a hard-coded list."""
    _, blocks = screen.block_consistency(d.trials, d.approach)
    return [(r['session_label'], int(r['block_index'])) for _, r in blocks[blocks['fhy_swap_signature']].iterrows()]


# ───────────────────────────── compute ─────────────────────────────


def compute(force: bool = False, only: list[str] | None = None) -> None:
    """Run the experiments whose results are missing or stale (all with ``force``; ``only`` selects names),
    then save the derived data of the figure, tables and numbers."""
    todo = [n for n in EXPERIMENTS if P.should_run(n, force, only)]
    if not todo and P.file_current(SUMMARY_JSON):
        print('reference: results current; skipped (--force recomputes)', flush=True)
        return
    from stage2.data import load_cache
    d = load_cache()                                     # primary set
    nat = load_cache(('natural_use',))                   # one free-choice trial per person
    swap = swap_blocks(d)
    note_swap = ', '.join(f'{s} block {b}' for s, b in swap) or 'none'
    common = dict(extra=(nat.approach, nat.trials))
    bi = pd.to_numeric(d.trials['block_index'], errors='coerce').fillna(-1).astype(int)
    keep = ~pd.MultiIndex.from_arrays([d.trials['session_label'], bi]).isin(swap)
    dk = d.subset(keep)
    specs = {
        REFERENCE: (yaw_logreg(d.channels), d, 'sanity floor: logreg on yaw_cum at 8 instants, mirror-augmented; '
                    'pre-specified (README section 9)'),
        'sens_swap_blocks_train_only': (yaw_logreg(d.channels, drop_blocks=swap), d,
                                        f'reference without the swap-signature blocks in training ({note_swap}); '
                                        'evaluation unchanged'),
        'sens_swap_blocks_removed': (yaw_logreg(d.channels), dk, 'reference with the swap-signature blocks removed '
                                     f'from training and evaluation ({note_swap})'),
        'sens_d6_screen': (yaw_logreg(d.channels, d6=True), d,
                           'reference with the docs/09 D6 screen removing flagged training trials in every fold'),
    }
    for name in todo:
        fp, data, notes = specs[name]
        t0 = time.time()
        r = cv.run_experiment(name, fp, data.approach, data.trials, notes=notes, **common)
        print(f'  {name}: accuracy {100 * r.metrics["accuracy"]:.1f} % ({r.leak_check}; {time.time() - t0:.0f} s)',
              flush=True)
    derive(swap)


def derive(swap: list[tuple[str, int]]) -> dict[str, Any]:
    """Paired comparisons and the summary, from the saved results only (no fitting)."""
    res = {n: cv.load_result(n) for n in EXPERIMENTS}
    ref = res[REFERENCE]
    rows = []
    for name, r in res.items():
        row = {'name': name, 'label': LABELS[name], 'notes': r.info.get('notes'), 'n_trials': int(r.metrics['n']),
               'n_persons': int(r.predictions['person_id'].nunique()),
               'accuracy': r.metrics['accuracy'], 'accuracy_lo': r.ci['accuracy'][0], 'accuracy_hi': r.ci['accuracy'][1],
               'roc_auc': r.metrics['roc_auc'], 'roc_auc_lo': r.ci['roc_auc'][0], 'roc_auc_hi': r.ci['roc_auc'][1],
               'log_loss': r.metrics['log_loss'], 'ece': r.metrics['ece'],
               'gated_n': int(r.gated_metrics['n']), 'gated_accuracy': r.gated_metrics['accuracy'],
               'natural_use_n': int(r.extra_metrics['n']), 'natural_use_accuracy': r.extra_metrics['accuracy'],
               'train_rows_dropped_per_fold_median': float(np.median([json.loads(p).get('n_train_dropped', 0)
                                                                      for p in r.predictions['params']])),
               'leak_check': r.leak_check}
        if name != REFERENCE:
            with warnings.catch_warnings():  # the 'removed' variant shares only part of the trials, by design
                warnings.filterwarnings('ignore', message='paired_compare')
                cmp = cv.paired_compare(r, ref)
            row.update({'paired_trials': cmp['n_trials'], 'acc_diff_vs_reference': cmp['pooled_acc_diff'],
                        'acc_diff_lo': cmp['pooled_acc_diff_ci95'][0], 'acc_diff_hi': cmp['pooled_acc_diff_ci95'][1],
                        'p_sign_flip': cmp['p_sign_flip_acc'], 'logloss_diff_vs_reference': cmp['mean_person_logloss_diff'],
                        'p_sign_flip_logloss': cmp['p_sign_flip_logloss']})
        rows.append(row)
    sens = pd.DataFrame(rows)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    RX.export_table(sens, SENS_TABLE.with_suffix(''), tex=False)
    kc = ref.k_curve
    summary = {
        'provenance': P.provenance(),
        'reference': {'name': REFERENCE, 'n_trials': int(ref.metrics['n']),
                      'n_persons': int(ref.predictions['person_id'].nunique()), 'n_folds': ref.info.get('n_folds'),
                      'accuracy': ref.metrics['accuracy'], 'ci95': list(ref.ci['accuracy']),
                      'roc_auc': ref.metrics['roc_auc'], 'roc_auc_ci95': list(ref.ci['roc_auc']),
                      'log_loss': ref.metrics['log_loss'], 'ece': ref.metrics['ece'],
                      'person_acc_median': ref.metrics['person_acc_median'],
                      'n_persons_below_80': ref.metrics.get('n_persons_below_80'),
                      'gated': {'n': ref.gated_metrics['n'], 'accuracy': ref.gated_metrics['accuracy']},
                      'natural_use': {'n': ref.extra_metrics['n'], 'accuracy': ref.extra_metrics['accuracy']},
                      'k_attempts': kc[['k', 'accuracy', 'lo', 'hi', 'n_persons']].to_dict(orient='records'),
                      'k_persons_left_out': kc['persons_left_out'].iloc[0] if len(kc) else '',
                      'leak_check': ref.leak_check, 'cap_per_person': ref.info.get('cap_per_person'),
                      'capped_training_rows': ref.info.get('capped_training_rows_per_fold')},
        'swap_blocks': [{'session_label': s, 'block_index': b} for s, b in swap],
        'sensitivity': sens.drop(columns=['notes']).to_dict(orient='records'),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=1, default=float), encoding='utf-8')
    return summary


def load_summary() -> dict[str, Any]:
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f'{SUMMARY_JSON}: run python -m stage2.pipeline compute --lanes reference')
    return json.loads(SUMMARY_JSON.read_text(encoding='utf-8'))


# ───────────────────────────── presentation (saved files only) ─────────────────────────────


def figures(force: bool = False, only: list[str] | None = None) -> None:
    from stage2.scripts import reference_figures
    reference_figures.draw(only)


def tables(force: bool = False, only: list[str] | None = None) -> None:
    """tables/reference/reference_models.csv|.tex (one row per experiment) and reference_k_attempts.csv|.tex."""
    s = load_summary()
    sens = pd.DataFrame(s['sensitivity'])
    if P.selected('reference_models', only):
        t = pd.DataFrame({
            'Model': sens['label'], 'Trials': sens['n_trials'], 'Persons': sens['n_persons'],
            'Accuracy': sens['accuracy'], '95% CI': [f'[{100 * a:.1f}, {100 * b:.1f}]' for a, b in
                                                     zip(sens['accuracy_lo'], sens['accuracy_hi'])],
            'AUC': sens['roc_auc'], 'Log loss': sens['log_loss'],
            'Diff. vs ref.': sens.get('acc_diff_vs_reference', pd.Series([np.nan] * len(sens))),
            'p (sign flip)': sens.get('p_sign_flip', pd.Series([np.nan] * len(sens))),
            'Natural use': sens['natural_use_accuracy']})
        RX.export_table(t, TAB_DIR / 'reference_models', label='tab:reference_models',
                        caption=('Reference floor and label-noise sensitivity (leave-one-person-out; 95\\,\\% '
                                 'person-cluster bootstrap CI; differences paired against the reference).'),
                        column_formats={'Accuracy': '.1%', 'AUC': '.3f', 'Log loss': '.3f', 'Diff. vs ref.': '+.1%',
                                        'p (sign flip)': '.2f', 'Natural use': '.0%'},
                        align='lrrrrrrrrr')
    if P.selected('reference_k_attempts', only):
        k = pd.DataFrame(s['reference']['k_attempts'])
        t = pd.DataFrame({'Attempts averaged': k['k'].astype(int), 'Accuracy': k['accuracy'], 'CI low': k['lo'],
                          'CI high': k['hi'], 'Persons': k['n_persons'].astype(int)})
        RX.export_table(t, TAB_DIR / 'reference_k_attempts', label='tab:reference_k_attempts',
                        caption=('Reference floor: per-person balanced accuracy when the P(right) of $k$ attempts '
                                 'of one person and hand are averaged (95\\,\\% person-cluster bootstrap CI).'),
                        column_formats={'Accuracy': '.1%', 'CI low': '.1%', 'CI high': '.1%'})


def numbers(force: bool = False, only: list[str] | None = None) -> None:
    """results/reference_numbers.json: every number of this unit the report text cites."""
    E = RX.numbers_entry
    s = load_summary()
    r = s['reference']
    src = 'tables/reference_summary.json'
    n = {
        'n_trials': E(r['n_trials'], 'd', src, 'primary trials scored by the reference'),
        'n_persons': E(r['n_persons'], 'd', src, 'persons (= leave-one-person-out folds)'),
        'n_folds': E(r['n_folds'], 'd', src, 'outer folds'),
        'accuracy': E(r['accuracy'], '.1%', src, 'reference LOSO accuracy'),
        'accuracy_lo': E(r['ci95'][0], '.1%', src, 'reference accuracy, 95% CI low (person-cluster bootstrap)'),
        'accuracy_hi': E(r['ci95'][1], '.1%', src, 'reference accuracy, 95% CI high'),
        'roc_auc': E(r['roc_auc'], '.3f', src, 'reference ROC AUC'),
        'log_loss': E(r['log_loss'], '.3f', src, 'reference log loss'),
        'ece': E(r['ece'], '.3f', src, 'reference expected calibration error (10 equal-mass bins)'),
        'person_acc_median': E(r['person_acc_median'], '.1%', src, 'median per-person accuracy'),
        'natural_use_n': E(r['natural_use']['n'], 'd', src, 'natural-use trials scored'),
        'natural_use_accuracy': E(r['natural_use']['accuracy'], '.0%', src, 'reference accuracy on natural use'),
        'gated_accuracy': E(r['gated']['accuracy'], '.1%', src, 'reference accuracy on trials passing gates G1, G2'),
        'n_swap_blocks': E(len(s['swap_blocks']), 'd', src, 'blocks with the swap signature'),
    }
    for row in r['k_attempts']:
        k = int(row['k'])
        n[f'acc_k{k}'] = E(row['accuracy'], '.1%', src, f'per-person balanced accuracy, {k} attempt(s) averaged')
    if r['k_attempts']:
        n['k_n_persons'] = E(int(r['k_attempts'][0]['n_persons']), 'd', src, 'persons in the k-attempt curve')
    for row in s['sensitivity']:
        if row['name'] == REFERENCE:
            continue
        key = row['name'].replace('sens_', '')
        n[f'{key}_accuracy'] = E(row['accuracy'], '.1%', src, f"accuracy, {row['label']}")
        n[f'{key}_diff'] = E(row['acc_diff_vs_reference'], '+.1%', src, f"accuracy minus reference, {row['label']}")
        n[f'{key}_p'] = E(row['p_sign_flip'], '.2f', src, f"sign-flip p-value, {row['label']}")
    RX.write_numbers(NUMBERS_JSON, n)


STEPS = {'compute': compute, 'figures': figures, 'tables': tables, 'numbers': numbers}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true', help='recompute the experiments even when current')
    a = ap.parse_args(argv)
    P.run('reference', force=a.force)


if __name__ == '__main__':
    main()
