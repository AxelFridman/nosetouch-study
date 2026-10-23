"""Figures for the report scripts. Uses the non-interactive Agg backend (safe on servers and in tests)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .conventions import Normalization, normalize_motion  # noqa: E402
from .features import prepare_trial  # noqa: E402
from .io import streams_from_record  # noqa: E402
from .segment import offline_v1_from_record, rerun_online_v1  # noqa: E402

HAND_COLORS = {'left': '#d1495b', 'right': '#00798c'}
AXIS_COLORS = ('#d1495b', '#3c8d2f', '#30638e')


def _save(fig: Any, path: str | Path | None) -> Any:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_trial(record: dict[str, Any], *, norm: Normalization | None = None, gravity_sign: int = 1,
               path: str | Path | None = None, title: str | None = None) -> Any:
    """Raw signals with the segmentation markers of one trial.

    Panels: (1) |w| raw and smoothed as online_v1 sees it, with its thresholds; (2) |a| likewise; (3) the
    normalised acceleration axes (W3C sign); (4) the angular velocity axes and the gravity-referenced
    yaw rate w·g_hat (its integral is the main chiral feature). Vertical lines: GO, the stored online_v1
    onset / end (blue), this package's re-run of online_v1 (dotted black, must coincide), offline_v1
    (red) and the end cue. ``norm`` = the session normalisation (``Dataset.trial_normalization``).
    """
    nm = norm or Normalization.from_sign(gravity_sign)
    s = streams_from_record(record)
    sig = normalize_motion(s.t, s.acc, s.accg, s.rr, nm)
    run = rerun_online_v1(record)
    off = offline_v1_from_record(record)
    tim = record.get('timing') or {}
    seg = record.get('segmentation') or {}
    tr = run.trace
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax = axes[0]
    ax.plot(sig.t, np.linalg.norm(sig.gyro, axis=1), color='0.75', lw=0.8, label='|w| raw')
    ax.plot(tr['t'], tr['gyro_s'], color='k', lw=1.3, label='|w| EMA (online_v1, until it stops)')
    p = seg.get('params') or {}
    for k, ls in (('still_gyro_dps', ':'), ('end_gyro_dps', '--')):
        if k in p:
            ax.axhline(p[k], color='0.4', ls=ls, lw=0.8, label=k)
    if math.isfinite(run.thresholds[0]):
        ax.axhline(run.thresholds[0], color='tab:orange', ls='--', lw=0.8, label='onset threshold')
    ax.set_ylabel('deg/s')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax2 = axes[1]
    ax2.plot(tr['t'], tr['acc_raw'], color='0.75', lw=0.8, label=f"|a| as online_v1 sees it ({run.result['acc_source']})")
    ax2.plot(tr['t'], tr['acc_s'], color='k', lw=1.3, label='|a| EMA')
    ax2.set_ylabel('m/s²')
    ax2.legend(loc='upper right', fontsize=7)
    acc_plot = sig.acc if np.isfinite(sig.acc).any() else sig.accg
    for j, lab in enumerate('xyz'):
        axes[2].plot(s.t, acc_plot[:, j], color=AXIS_COLORS[j], lw=0.9, label=f'acc {lab}')
    axes[2].set_ylabel('acc (W3C sign) m/s²')
    axes[2].legend(loc='upper right', fontsize=7, ncol=3)
    pt = prepare_trial(record, norm=nm)
    if pt.t.size:
        yaw_rate = np.sum(pt.gyro * pt.up, axis=1)
        axes[3].plot(pt.t, yaw_rate, color='tab:purple', lw=1.4, label=f'yaw rate w·g_hat ({pt.gravity_method})')
        for j, lab in enumerate('xyz'):
            axes[3].plot(pt.t, pt.gyro[:, j], color=AXIS_COLORS[j], lw=0.6, alpha=0.6, label=f'gyro {lab}')
    axes[3].set_ylabel('deg/s')
    axes[3].set_xlabel('ms since t0')
    axes[3].legend(loc='upper right', fontsize=7, ncol=4)
    t = s.t
    marks = []
    if tim.get('go_ms') is not None:
        marks.append((tim['go_ms'], 'GO', 'tab:green', '-'))
    if seg.get('onset_index') is not None and seg['onset_index'] < t.size:
        marks.append((t[seg['onset_index']], 'stored onset', 'tab:blue', '-'))
    if seg.get('end_index') is not None and seg['end_index'] < t.size:
        marks.append((t[seg['end_index']], 'stored end', 'tab:blue', '--'))
    r_on, r_en = run.result['onset_index'], run.result['end_index']
    if r_on is not None and r_on < t.size:
        marks.append((t[r_on], 'online_v1 re-run onset', 'k', ':'))
    if r_en is not None and r_en < t.size:
        marks.append((t[r_en], 'online_v1 re-run end', 'k', ':'))
    if off.ok:
        marks.append((off.onset_ms, 'offline onset', 'tab:red', '-'))
        marks.append((off.end_ms, 'offline end', 'tab:red', '--'))
    if tim.get('end_detected_ms') is not None:
        marks.append((tim['end_detected_ms'], 'end cue', '0.3', '-.'))
    for a in axes:
        for x, lab, col, ls in marks:
            a.axvline(x, color=col, ls=ls, lw=1.0)
    for x, lab, col, ls in marks:
        axes[0].text(x, axes[0].get_ylim()[1], lab, rotation=90, va='top', ha='right', fontsize=7, color=col)
    c = record.get('condition') or {}
    fig.suptitle(title or f"{record.get('trial_id')}  {c.get('hand')} hand, {c.get('orientation')}, {c.get('speed')}, "
                 f"{c.get('variant')}  status={record.get('outcome', {}).get('status')}  port={run.status}  "
                 f"offline={off.reason}  gravity sign {nm.sign:+d} ({nm.gravity.source})", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_sampling(trials: pd.DataFrame, path: str | Path | None = None) -> Any:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for plat, g in trials.groupby(trials['platform'].astype(object).fillna('unknown')):
        hz = g['motion_hz'].dropna().astype(float)
        if len(hz):
            ax[0].hist(hz, bins=30, alpha=0.6, label=str(plat))
        dt = g['motion_dt_ms_max'].dropna().astype(float)
        if len(dt):
            ax[1].hist(dt, bins=30, alpha=0.6, label=str(plat))
    ax[0].set_xlabel('motion sampling rate per trial (Hz)')
    ax[1].set_xlabel('largest gap between motion samples (ms)')
    for a in ax:
        a.legend(fontsize=8)
        a.set_ylabel('trials')
    fig.tight_layout()
    return _save(fig, path)


def plot_counts(trials: pd.DataFrame, path: str | Path | None = None) -> Any:
    acc = trials[trials['accepted'].fillna(False).astype(bool)]
    tab = pd.crosstab(acc['orientation'], acc['hand']) if len(acc) else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    if len(tab):
        x = np.arange(len(tab))
        w = 0.4
        for k, h in enumerate(('left', 'right')):
            if h in tab.columns:
                ax.bar(x + (k - 0.5) * w, tab[h].to_numpy(), w, label=h, color=HAND_COLORS[h])
        ax.set_xticks(x)
        ax.set_xticklabels(tab.index, rotation=30, ha='right', fontsize=8)
        ax.legend()
    ax.set_ylabel('accepted trials')
    fig.tight_layout()
    return _save(fig, path)


def plot_durations(trials: pd.DataFrame, path: str | Path | None = None) -> Any:
    acc = trials[trials['accepted'].fillna(False).astype(bool)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    speeds = [s for s in ('fast', 'natural', 'slow') if s in set(acc['speed'])]
    data = [acc.loc[acc['speed'] == s, 'movement_ms'].dropna().astype(float).to_numpy() for s in speeds]
    if speeds and any(len(d) for d in data):
        ax.boxplot(data)  # 'labels' was renamed 'tick_labels' in matplotlib 3.9: set the ticks directly
        ax.set_xticks(range(1, len(speeds) + 1))
        ax.set_xticklabels(speeds)
    ax.set_ylabel('online_v1 movement duration (ms)')
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_by_hand(X: pd.DataFrame, meta: pd.DataFrame, feats: list[str], path: str | Path | None = None) -> Any:
    n = len(feats)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    hands = meta.set_index('trial_id')['hand'] if 'trial_id' in meta.columns else meta['hand']
    for k, f in enumerate(feats):
        ax = axes[k // cols][k % cols]
        for h in ('left', 'right'):
            v = X.loc[hands.index[hands == h].intersection(X.index), f].dropna().astype(float)
            if len(v):
                ax.hist(v, bins=30, alpha=0.6, color=HAND_COLORS[h], label=h)
        ax.set_title(f, fontsize=9)
        ax.legend(fontsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.tight_layout()
    return _save(fig, path)


def plot_roc(results: dict[str, Any], path: str | Path | None = None) -> Any:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for name, r in results.items():
        y = r.oof['y_true'].to_numpy()
        if 0 < y.sum() < y.size:
            fpr, tpr, _ = roc_curve(y, r.oof['p_right'])
            ax.plot(fpr, tpr, label=f"{name} (AUC {r.metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color='0.6', ls=':')
    ax.set_xlabel('false positive rate (left called right)')
    ax.set_ylabel('true positive rate (right called right)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_subgroups(table: pd.DataFrame, path: str | Path | None = None) -> Any:
    t = table.dropna(subset=['error']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(t) + 1)))
    if len(t):
        y = np.arange(len(t))[::-1]
        lo = (t['error'] - t['error_lo']).clip(lower=0).fillna(0)
        hi = (t['error_hi'] - t['error']).clip(lower=0).fillna(0)
        ax.errorbar(t['error'], y, xerr=[lo, hi], fmt='o', ms=4, color='k', ecolor='0.5', capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{v}={lv} (n={n})" for v, lv, n in zip(t['variable'], t['level'], t['n_trials'])], fontsize=7)
    ax.set_xlabel('error rate (95% participant-bootstrap CI)')
    fig.tight_layout()
    return _save(fig, path)
