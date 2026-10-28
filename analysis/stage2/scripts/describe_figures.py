"""Figures of the core unit 'describe' (S2_01..S2_20), drawn ONLY from the files its compute step saved.

One function per figure, named after it. Every presentation detail (sizes, axis limits and scales, ticks,
bins, labels, colours) is a keyword argument whose default lives in ``FIGURES`` below, so a change is a one-line
edit followed by

    python -m stage2.pipeline figures --lanes describe                          # every figure, seconds
    python -m stage2.pipeline figures --lanes describe --only S2_05_sampling_gaps

or, from Python, ``fig_S2_05_sampling_gaps(interval_clip_ms=100)``. No function fits a model or reads the cache or
the raw snapshot: the inputs are tables/*.csv, tables/describe_summary.json and tables/describe/describe_*.json|csv.
Each function also writes its plotted data as a tidy CSV, tables/describe/<figure name>.csv. Captions and titles
take every count from the saved data.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from stage2 import pipeline as PL
from stage2 import report_export as RX
from stage2.plotting import (
    AXIS, EVENT_COLORS, GRID, HAND_COLORS, HAND_LABELS, INK, INK2, MUTED, PLATFORM_COLORS, PLATFORM_LABELS, RAMP,
    SERIES, figure, hand_handles, plain_log_ticks, save_fig,
)
from stage2.scripts import describe as D

PLATFORMS = D.PLATFORMS
AXIS3 = ('#2a78d6', '#eb6834', '#1baf7a')   # x, y, z (validated categorical slots 1-3)
DASH = (0, (3, 2))

#: per-figure presentation defaults (edit here, or pass keyword arguments to the figure function)
FIGURES: dict[str, dict[str, Any]] = {
    'S2_01_demographics': dict(
        width='full', height=2.0, color=SERIES, age_bin_width=1.0, age_first_edge=None, marker_size=4,
        height_spread=0.12, bar_height=0.55, count_fontsize=7, persons_xpad=2,
        categories=('right', 'both', 'left')),
    'S2_02_exclusion_flow': dict(
        width='full', height=3.0, width_ratios=(3.2, 1), color=SERIES, natural_color=MUTED, bar_height=0.6,
        label_pad=8, natural_label_pad=0.5, label_fontsize=7, ytick_fontsize=7, xmax_factor=1.35,
        natural_xlim=None, xlabel='attempts remaining (primary path)', natural_xlabel='natural-use trials'),
    'S2_03_completion_durations': dict(
        width='full', height=2.4, width_ratios=(1.25, 1), bar_height=0.7, ytick_fontsize=6, legend_fontsize=6,
        jitter=0.15, marker_size=3, protocol='2.0.0', bonus_tier=4,
        xlabel_left='accepted attempts (street sessions)', xlabel_right='minutes (completed visits)'),
    'S2_04_conditions': dict(
        width='full', height=3.4, marker_size=4.5, ytick_fontsize=6, xscale='log', xticks=(1, 2, 5, 10, 20, 50, 100),
        xlabel='primary trials (log scale)'),
    'S2_05_sampling_gaps': dict(
        width='full', height=2.3, interval_clip_ms=60, interval_bin_ms=1, interval_yscale='log', gap_jitter=0.25,
        marker_size=3, platform_gap=3.6, ytick_fontsize=6, note_fontsize=6),
    'S2_06_noise_vs_quantisation': dict(
        width='full', height=2.0, jitter=0.2, marker_size=2.2, alpha=0.5, xscale='log',
        acc_xticks=(0.03, 0.1, 0.3), gyro_xticks=(0.03, 0.1, 0.3, 1, 3, 10), ylim=(1.7, -0.6), note_fontsize=6),
    'S2_07_spectrum': dict(
        width='half', height=2.2, xlim=(0, 30), ylim=(0, 1.02), legend_fontsize=6, gyro_lw=1.3, acc_lw=0.9,
        acc_alpha=0.6),
    'S2_08_durations_peaks': dict(
        width='full', height=2.4, speeds=('fast', 'natural', 'slow'), jitter=0.1, offset=0.36, marker_size=2.2,
        alpha=0.55, yscale='log', yticks=(0.2, 0.5, 1, 2, 5, 10), peak_bins=(0, 400, 10), note_fontsize=6),
    'S2_09_segmentation_contact': dict(
        width='full', height=2.2, diff_clip_ms=800, diff_bin_ms=25, touch_range=(-0.25, 1.5), touch_bin=0.05,
        note_fontsize=6),
    'S2_10_example_traces': dict(
        width='full', height=4.6, xlim_left=-1.0, xpad_right=0.6, xlim_right_default=5.0, event_fontsize=5.5,
        legend_fontsize=6, trace_lw=0.9, yaw_lw=1.3),
    'S2_11_class_means': dict(
        width='full', height=3.6, ncols=3, band_alpha=0.18, lw=1.4, mirror_lw=0.9, legend_fontsize=6,
        titles={'yaw_rate': 'Yaw rate about gravity (chiral)', 'yaw_cum': 'Cumulative yaw (chiral)',
                'gyro_horiz': 'Tilting rate', 'acc_vert': 'Vertical acceleration',
                'acc_horiz': 'Horizontal acceleration', 'tilt_from_start': 'Tilt from the start pose'},
        units={'yaw_rate': 'deg/s', 'yaw_cum': 'deg', 'gyro_horiz': 'deg/s', 'acc_vert': 'm/s$^2$',
               'acc_horiz': 'm/s$^2$', 'tilt_from_start': 'deg'}),
    'S2_12_sign_patterns': dict(width='half', height=2.2, marker_size=2.4, ring_size=7, note_fontsize=6),
    'S2_13_first_half_yaw': dict(
        width='full', height=2.5, width_ratios=(1.2, 1), ytick_fontsize=5.5, xlim=(-0.02, 1.02), clip_deg=60,
        bin_deg=3, fill_alpha=0.25),
    'S2_14_platforms': dict(
        width='full', height=2.2, band_alpha=0.18, lw=1.3, legend_fontsize=6, clip_deg=60, jitter=0.1,
        offset=0.36, marker_size=2, alpha=0.45),
    'S2_15_return_vs_approach': dict(width='full', height=2.1, band_alpha=0.18, lw=1.3, legend_fontsize=6),
    'S2_16_cuefree_v0': dict(
        width='full', height=2.3, width_ratios=(1.25, 1, 1), ytick_fontsize=6.5, error_clip_ms=600,
        error_bin_ms=25, still_clip_s=4.0, still_bin_s=0.1),
    'S2_17_gravity_closure': dict(width='half', height=2.0, clip_deg=15, bin_deg=0.5),
    'S2_18_flags_gates': dict(width='full', height=2.4, ytick_fontsize=6.5, note_fontsize=6, xmax_factor=1.45,
                              label_pad=1.5),
    'S2_19_cuefree_stress': dict(
        width='full', height=2.4, xlim=(-0.05, 2.6), ylim=(-0.02, 1.02), label_x=2.35, label_gap=0.045,
        marker_size=3, legend_fontsize=6.5, note_fontsize=6),
    'S2_20_label_consistency': dict(
        width='full', height=4.5, ytick_fontsize=5.5, text_fontsize=4.8, size_base=14, size_per_trial=9,
        ring_extra=70, legend_fontsize=6.5),
}


# ───────────────────────────── helpers ─────────────────────────────


def _opts(name: str, kw: dict[str, Any]) -> dict[str, Any]:
    unknown = set(kw) - set(FIGURES[name])
    if unknown:
        raise TypeError(f'{name}: unknown presentation option(s) {sorted(unknown)}; see FIGURES[{name!r}]')
    return {**FIGURES[name], **kw}


def _save(fig: Any, name: str, *, caption: str, decision: str, data: pd.DataFrame) -> dict[str, str]:
    out = save_fig(fig, name, caption=caption, decision=decision)
    RX.export_table(data, D.DATA_DIR / name, tex=False)          # tidy plotted data (report-safe)
    return out


def _arr(v: Any) -> np.ndarray:
    """A saved list (None = NaN) as a float array."""
    return np.array([np.nan if x is None else x for x in v], dtype=float) if isinstance(v, list) else np.asarray(v, float)


def _means(entry: dict[str, Any]) -> np.ndarray:
    return np.array([_arr(m) for m in entry['means']], dtype=float)


def _mean_se(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return M.mean(axis=0), M.std(axis=0, ddof=1) / math.sqrt(M.shape[0])


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f'{n} {word if n == 1 else (plural or word + "s")}'


def _pct(v: float) -> str:
    return f'{100 * v:.0f} %'


# ───────────────────────────── 1-4: participants, exclusions, completion, conditions ─────────────────────────────


def fig_S2_01_demographics(**kw: Any) -> dict[str, str]:
    name = 'S2_01_demographics'
    o = _opts(name, kw)
    P = D.load_table('participants')
    demo = D.load_summary()['demographics']
    age = P['age_years'].astype(float)
    hgt = P['height_cm'].astype(float)
    fig, ax = figure(o['width'], o['height'], 1, 4)
    a = ax[0]
    first = o['age_first_edge'] if o['age_first_edge'] is not None else math.floor(age.min()) - 0.5
    a.hist(age, bins=np.arange(first, age.max() + 1.5, o['age_bin_width']), color=o['color'], edgecolor='white',
           linewidth=0.8)
    a.set_xlabel('age (years)')
    a.set_ylabel('persons')
    a.set_title(f'Age: {age.mean():.1f} $\\pm$ {age.std(ddof=1):.1f} y')
    a = ax[1]
    sexes = sorted(P['sex'].dropna().unique())
    rows = [{'panel': 'age', 'person_id': p, 'category': '', 'value': v} for p, v in zip(P['person_id'], age)]
    for k, sx in enumerate(sexes):
        g = P[P['sex'] == sx]
        h = g['height_cm'].astype(float).dropna()
        a.plot(np.full(len(h), k) + np.linspace(-o['height_spread'], o['height_spread'], len(h)), h, 'o',
               ms=o['marker_size'], color=o['color'], mec='white', mew=0.6)
        rows += [{'panel': 'height', 'person_id': p, 'category': sx, 'value': v}
                 for p, v in zip(g.loc[h.index, 'person_id'], h)]
    a.set_xticks(range(len(sexes)), sexes)
    a.set_xlim(-0.6, len(sexes) - 0.4)
    a.set_ylabel('height (cm)')
    a.set_title(f'Height (n = {int(hgt.notna().sum())})')
    cats = list(o['categories'])
    for a, col, title in ((ax[2], 'handedness_self', 'Dominant hand'), (ax[3], 'phone_hand', 'Usual phone hand')):
        vc = P[col].value_counts().reindex(cats).fillna(0)
        a.barh(range(len(cats)), vc.to_numpy(), color=o['color'], height=o['bar_height'])
        a.set_yticks(range(len(cats)), vc.index)
        a.invert_yaxis()
        for i, v in enumerate(vc.to_numpy()):
            a.text(v + 0.3, i, f'{int(v)}', va='center', fontsize=o['count_fontsize'], color=INK2)
        a.set_xlim(0, len(P) + o['persons_xpad'])
        a.set_xlabel('persons')
        a.set_title(title)
        a.grid(axis='y', visible=False)
        rows += [{'panel': col, 'person_id': '', 'category': c, 'value': float(v)} for c, v in vc.items()]
    nl = int(demo['n_left_handed'])
    lh = 'left-hander' if nl == 1 else 'left-handers'
    return _save(fig, name, data=pd.DataFrame(rows), caption=(
        f"The {demo['n_persons']} persons of the analysis: age {age.mean():.1f} ± {age.std(ddof=1):.1f} years (range "
        f"{age.min():.0f}-{age.max():.0f}), {demo['n_female']} women and {demo['n_male']} men, {nl} {lh}, "
        f"{demo['persons_ios']} on iPhone and {demo['persons_android']} on Android."),
        decision=f'Who the model is evaluated on; with {nl} {lh}, handedness effects can only be exploratory.')


def fig_S2_02_exclusion_flow(**kw: Any) -> dict[str, str]:
    name = 'S2_02_exclusion_flow'
    o = _opts(name, kw)
    rows = D.load_table('exclusion_flow')
    labels = [(f"{r['step']}  {r['rule']}" if r['step'] != 'start' else 'all attempts in the snapshot').replace('|w|', r'$|\omega|$')
              for _, r in rows.iterrows()]
    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    y = np.arange(len(rows))
    a.barh(y, rows['remaining'], color=o['color'], height=o['bar_height'])
    for i, r in rows.iterrows():
        txt = f"{int(r['remaining'])}"
        if r['removed'] > 0:
            txt += f"   (−{int(r['removed'])})"
        if r['natural_use_added'] > 0:
            txt += f"   ({int(r['natural_use_added'])} to natural use)"
        a.text(r['remaining'] + o['label_pad'], i, txt, va='center', fontsize=o['label_fontsize'], color=INK2)
    a.set_yticks(y, labels, fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    a.set_xlim(0, rows['remaining'].max() * o['xmax_factor'])
    a.set_xlabel(o['xlabel'])
    a.grid(axis='y', visible=False)
    a.set_title(f"Primary set: {int(rows['remaining'].iloc[-1])} trials of {int(rows['persons'].iloc[-1])} persons")
    a = ax[1]
    first = np.flatnonzero(rows['natural_use_added'].to_numpy() > 0)
    nat = rows.iloc[int(first[0]):] if first.size else rows.iloc[0:0]   # from the rule that creates the set on
    a.barh(np.arange(len(nat)), nat['natural_use'], color=o['natural_color'], height=o['bar_height'])
    last = rows['step'].iloc[-1]
    for i, (_, r) in enumerate(nat.iterrows()):
        k = int(r['natural_use_removed'])
        note = '' if not k else (f' ({k} repeats)' if r['step'] == last else f' (−{k})')
        a.text(r['natural_use'] + o['natural_label_pad'], i, f"{int(r['natural_use'])}{note}", va='center',
               fontsize=o['label_fontsize'], color=INK2)
    a.set_yticks(np.arange(len(nat)), nat['step'], fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    xmax = float(nat['natural_use'].max()) if len(nat) else 1.0
    a.set_xlim(*(o['natural_xlim'] or (0, math.ceil(xmax * 1.4))))
    a.set_xlabel(o['natural_xlabel'])
    a.set_title('Natural-use set')
    a.grid(axis='y', visible=False)
    data = rows[['step', 'rule', 'removed', 'remaining', 'persons', 'natural_use_added', 'natural_use_removed',
                 'natural_use']]
    return _save(fig, name, data=data, caption=(
        'Exclusion cascade, applied in order (an attempt is charged to the first rule it fails). E3 moves the '
        'tier-0 free-choice trials, whose hand is self-reported, to the separate natural-use set, which then '
        'passes the same quality rules; N1 keeps one natural-use trial per person (the earliest) and moves the '
        'free trials of repeat visits to a separate repeat set.'),
        decision='Primary training/evaluation set and natural-use set; every rule and its count.')


def fig_S2_03_completion_durations(**kw: Any) -> dict[str, str]:
    name = 'S2_03_completion_durations'
    o = _opts(name, kw)
    comp = D.load_table('completion')
    dur = D.load_table('visit_durations')
    s = D.load_summary()
    cs, vd = s['completion'], s['visit_durations_completed_v2_min']
    tiers = sorted(int(c[len('accepted_tier'):]) for c in comp.columns if c.startswith('accepted_tier'))
    done = dur[(dur['protocol'].astype(str) == o['protocol']) & (dur['status'] == 'completed')]
    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    order = comp.sort_values(['person_id', 'session_order'])
    order = order[order['protocol'].astype(str) == o['protocol']]
    lab, data = [], []
    for i, (_, r) in enumerate(order.iterrows()):
        left = 0
        for k in tiers:
            n = r[f'accepted_tier{k}']
            if n:
                a.barh(i, n, left=left, color=RAMP[k % len(RAMP)], height=o['bar_height'], edgecolor='white', linewidth=0.6)
                left += n
            data.append({'panel': 'accepted_per_tier', 'session_label': r['session_label'], 'item': f'tier {k}',
                         'value': float(n)})
        lab.append(f"{r['session_label']}" + (' (ended early)' if r['status'] == 'ended_early' else ''))
    a.set_yticks(range(len(order)), lab, fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    a.set_xlabel(o['xlabel_left'])
    a.grid(axis='y', visible=False)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=RAMP[k % len(RAMP)], label=f'tier {k}') for k in tiers], ncol=len(tiers),
             loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=o['legend_fontsize'], handlelength=1.0,
             columnspacing=0.8)
    a = ax[1]
    cols = ['setup_s'] + [f'tier{k}_s' for k in tiers] + ['total_s']
    names = (['setup'] + [f'tier {k}' + (' (bonus)' if k == o['bonus_tier'] else '') for k in tiers]
             + ['whole visit'])
    for i, c in enumerate(cols):
        v = done[c].dropna() / 60
        a.plot(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-o['jitter'], o['jitter'], len(v)), 'o',
               ms=o['marker_size'], color=SERIES, alpha=0.75, mec='none')
        if len(v):
            a.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4)
        data += [{'panel': 'minutes', 'session_label': sl, 'item': names[i], 'value': float(x)}
                 for sl, x in zip(done.loc[v.index, 'session_label'], v)]
    a.set_yticks(range(len(cols)), names)
    a.invert_yaxis()
    a.set_xlabel(o['xlabel_right'])
    a.grid(axis='y', visible=False)
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f"Left: accepted attempts per tier and street session ({cs['completed']} completed, "
        f"{cs['ended_early']} ended early; the bonus tier {o['bonus_tier']} was taken in "
        f"{cs['bonus_uptake_completed']} completed visits). Right: wall-clock durations of the "
        f"completed visits (median {vd['total_median']:.1f} min)."),
        decision='Partial sessions still give balanced tier-1 data (both hands per tier), so they are kept.')


def fig_S2_04_conditions(**kw: Any) -> dict[str, str]:
    name = 'S2_04_conditions'
    o = _opts(name, kw)
    wide = D.load_table('trials_per_condition')
    lab = [f"{r['grip']} · {r['speed']} · {r['variant']}" + ('' if r['start_position'] == 'natural' else f" · {r['start_position']} start")
           for _, r in wide.iterrows()]
    fig, a = figure(o['width'], o['height'])
    y = np.arange(len(wide))
    a.hlines(y, wide['left'].clip(upper=wide['right']), wide[['left', 'right']].max(axis=1), color=GRID, lw=1.2)
    for h in ('left', 'right'):
        a.plot(wide[h], y, 'o', ms=o['marker_size'], color=HAND_COLORS[h], mec='white', mew=0.6, label=HAND_LABELS[h])
    a.set_yticks(y, lab, fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    a.set_xscale(o['xscale'])
    if o['xscale'] == 'log':
        plain_log_ticks(a.xaxis, list(o['xticks']))
    a.set_xlabel(o['xlabel'])
    a.legend(loc='lower right')
    a.grid(axis='y', visible=False)
    return _save(fig, name, data=wide[['grip', 'speed', 'variant', 'start_position', 'left', 'right']], caption=(
        f'Primary trials per instructed condition (grip · speed · movement variant; {len(wide)} conditions), left vs '
        'right hand. Upright natural-speed standard reaches dominate; every condition has both hands.'
        if bool(((wide['left'] > 0) & (wide['right'] > 0)).all()) else
        f'Primary trials per instructed condition (grip · speed · movement variant; {len(wide)} conditions), left vs '
        'right hand.'),
        decision='Conditions are balanced between hands, so a condition cannot stand in for the label; '
                 'rare variants are too few for per-condition accuracy.')


# ───────────────────────────── 5-7: sampling, noise, spectrum ─────────────────────────────


def fig_S2_05_sampling_gaps(**kw: Any) -> dict[str, str]:
    name = 'S2_05_sampling_gaps'
    o = _opts(name, kw)
    iv = D.load_input('sampling_intervals')
    G = D.load_table('gaps')
    big = D.load_summary()['ios_long_gaps']
    fig, ax = figure(o['width'], o['height'], 1, 2)
    a = ax[0]
    clip = float(o['interval_clip_ms'])
    bins = np.arange(0, clip + o['interval_bin_ms'] / 2, o['interval_bin_ms'])
    data = []
    n_trials = {}
    for pl in PLATFORMS:
        p = iv['platforms'].get(pl)
        if not p:
            continue
        v, c = _arr(p['value_ms']), _arr(p['count'])
        dens, edges, _ = a.hist(np.clip(v, 0, clip), bins=bins, weights=c, histtype='step', color=PLATFORM_COLORS[pl],
                                lw=1.2, density=True, label=f"{PLATFORM_LABELS[pl]} (median {p['median_ms']:.1f} ms)")
        n_trials[pl] = int(p['n_trials'])
        data += [{'panel': 'interval_density', 'platform': pl, 'phase': '', 'x_lo': lo, 'x_hi': hi, 'value': float(dv)}
                 for lo, hi, dv in zip(edges[:-1], edges[1:], dens)]
    a.set_yscale(o['interval_yscale'])
    a.set_xlabel(f'interval between samples (ms; >{clip:g} shown at {clip:g})')
    a.set_ylabel('density (log)' if o['interval_yscale'] == 'log' else 'density')
    a.legend(loc='upper right')
    a.set_title('Sampling intervals: both platforms run at 60 Hz')
    a = ax[1]
    ph_order = ['pre-onset (hold, settling)', 'near the movement', 'after the movement']
    ph_short = ('before', 'near movement', 'after')
    yt, yl = [], []
    for k, pl in enumerate(PLATFORMS):
        gg = G[G['platform'] == pl]
        for j, ph in enumerate(ph_order):
            v = gg.loc[gg['phase'] == ph, 'gap_ms'].to_numpy()
            yy = k * o['platform_gap'] + j + np.random.default_rng(10 * k + j).uniform(-o['gap_jitter'], o['gap_jitter'], v.size)
            a.plot(v, yy, 'o', ms=o['marker_size'], color=PLATFORM_COLORS[pl], alpha=0.8, mec='none')
            yt.append(k * o['platform_gap'] + j)
            yl.append(f'{PLATFORM_LABELS[pl]}: {ph_short[j]}')
            data += [{'panel': 'gap_ms', 'platform': pl, 'phase': ph, 'x_lo': x, 'x_hi': x, 'value': np.nan} for x in v]
    a.axvline(D.C.MAX_GAP_MS, color=INK, lw=0.9)
    a.text(D.C.MAX_GAP_MS + 6, 2.8, f'{D.C.MAX_GAP_MS:.0f} ms: a longer gap near the movement\nexcludes the trial (rule E4d)',
           fontsize=o['note_fontsize'], color=INK2, va='center')
    a.set_yticks(yt, yl, fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    a.set_xlabel(f"gap duration (ms), every interval > {iv['gap_list_ms']:g} ms")
    ga, gi = G.loc[G['platform'] == 'android', 'gap_ms'], G.loc[G['platform'] == 'ios', 'gap_ms']
    parts = []
    if len(ga):
        parts.append(f'Android {ga.min():.0f}-{ga.max():.0f} ms')
    if len(gi):
        parts.append(f'iOS median {gi.median():.0f} ms')
    a.set_title('Gaps: ' + ', '.join(parts))
    a.grid(axis='y', visible=False)
    ntxt = ' and '.join(f'{n} {PLATFORM_LABELS[pl]}' for pl, n in n_trials.items())
    pad = iv['near_pad_ms'] / 1000
    ios_txt = (f" The long iOS gaps (> 0.2 s; median {big['duration_median_ms']:.0f} ms) start a median "
               f"{big['start_after_recording_start_median_s']:.1f} s into the recording and only "
               f"{_pct(big['share_within_400ms_after_a_cue'])} of them follow an audio cue within 0.4 s."
               if big.get('n') else '')
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f'Left: distribution of sample intervals ({ntxt} trials). Right: every interval longer than '
        f"{iv['gap_list_ms']:g} ms in the complete attempts, by platform and by where it falls relative to the movement "
        f'([onset - {pad:g} s, end + {pad:g} s] is "near").' + ios_txt),
        decision='60 Hz grid; gaps up to 100 ms are interpolated, a longer gap near the movement excludes the '
                 'trial (E4d), gaps elsewhere stay NaN in the full stream.')


def fig_S2_06_noise_vs_quantisation(**kw: Any) -> dict[str, str]:
    name = 'S2_06_noise_vs_quantisation'
    o = _opts(name, kw)
    Q = D.load_table('sampling_per_trial')
    qsd = D.load_summary()['quantisation_noise_sd']
    fig, ax = figure(o['width'], o['height'], 1, 2)
    data = []
    for a, col, unit, title, ticks in ((ax[0], 'rest_accg_sd', 'm/s$^2$', 'Specific force at rest', o['acc_xticks']),
                                       (ax[1], 'rest_gyro_sd', 'deg/s', 'Angular velocity at rest', o['gyro_xticks'])):
        for k, pl in enumerate(PLATFORMS):
            g = Q.loc[Q['platform'] == pl, ['trial_key', col]].dropna()
            v = g[col].to_numpy()
            a.plot(v, k + np.random.default_rng(k).uniform(-o['jitter'], o['jitter'], v.size), 'o', ms=o['marker_size'],
                   alpha=o['alpha'], color=PLATFORM_COLORS[pl], mec='none')
            a.plot([np.median(v)] * 2, [k - 0.32, k + 0.32], color=INK, lw=1.3)
            data += [{'quantity': col, 'platform': pl, 'trial_key': t, 'sd': x} for t, x in zip(g['trial_key'], v)]
        a.axvline(qsd, color=INK2, lw=0.9)
        a.text(qsd * 1.08, 0.5, f"rounding noise of\nChrome's 0.1 step\n(SD {qsd:.3f})", fontsize=o['note_fontsize'],
               color=INK2, va='center')
        a.set_xscale(o['xscale'])
        if o['xscale'] == 'log':
            plain_log_ticks(a.xaxis, list(ticks))
        a.set_yticks(range(len(PLATFORMS)), [PLATFORM_LABELS[p] for p in PLATFORMS])
        a.set_ylim(*o['ylim'])
        a.set_xlabel(f'SD in the still window before GO ({unit}, log)')
        a.set_title(title)
        a.grid(axis='y', visible=False)
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        'Hand-held noise in the still window before GO, per trial and platform (bar = median), against the noise '
        f"that Chrome's 0.1-unit rounding adds (SD 0.1/sqrt(12) = {qsd:.3f}). Medians: specific force "
        f"{Q['rest_accg_sd'].median():.3f} m/s^2, angular velocity {Q['rest_gyro_sd'].median():.2f} deg/s."),
        decision='Native precision is kept on both platforms: the rounding step is far below the physiological '
                 'noise floor (iOS quantisation stays an ablation, A3).')


def fig_S2_07_spectrum(**kw: Any) -> dict[str, str]:
    name = 'S2_07_spectrum'
    o = _opts(name, kw)
    sp = D.load_input('spectrum')
    s = D.load_summary()['spectrum']
    f = _arr(sp['freqs_hz'])
    fig, a = figure(o['width'], o['height'])
    data = []
    for pl in PLATFORMS:
        c = sp['platforms'].get(pl)
        if not c:
            continue
        cg, ca = _arr(c['gyro']), _arr(c['acc'])
        a.plot(f, cg, color=PLATFORM_COLORS[pl], lw=o['gyro_lw'], label=f'{PLATFORM_LABELS[pl]} gyroscope')
        a.plot(f, ca, color=PLATFORM_COLORS[pl], lw=o['acc_lw'], alpha=o['acc_alpha'],
               label=f'{PLATFORM_LABELS[pl]} accelerometer')
        data += [{'platform': pl, 'sensor': sen, 'freq_hz': x, 'cum_share': y}
                 for sen, cv in (('gyroscope', cg), ('accelerometer', ca)) for x, y in zip(f, cv)]
    a.axvline(sp['lowpass_hz'], color=INK, lw=0.9)
    a.set_xlabel('frequency (Hz)')
    a.set_ylabel('cumulative share of power')
    a.set_ylim(*o['ylim'])
    a.set_xlim(*o['xlim'])
    a.legend(loc='lower right', fontsize=o['legend_fontsize'])
    a.set_title(f"Movement spectrum (raw, {sp['fs_hz']:g} Hz)")
    pad = sp['pad_ms'] / 1000
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f'Cumulative power spectrum of the approach (onset - {pad:g} s to end + {pad:g} s, raw values resampled to '
        f"{sp['fs_hz']:g} Hz, Hann-tapered over the whole window and zero-padded to {sp['n_fft']} points so that no "
        f"window is cropped; median over {s['n_trials']} primary trials): {100 * s['gyro_power_below_15hz']:.1f} % of "
        f"the gyroscope power and {100 * s['acc_power_below_15hz']:.1f} % of the accelerometer power lie below "
        f"{sp['lowpass_hz']:g} Hz."),
        decision=f"Zero-phase {sp['lowpass_hz']:g} Hz low-pass after resampling: removes sensor noise, keeps the movement.")


# ───────────────────────────── 8-10: durations, segmentation, example traces ─────────────────────────────


def fig_S2_08_durations_peaks(**kw: Any) -> dict[str, str]:
    name = 'S2_08_durations_peaks'
    o = _opts(name, kw)
    cand = D.load_input_csv('complete_attempts')
    du = D.load_summary()['durations']
    C = D.C
    fig, ax = figure(o['width'], o['height'], 1, 2)
    a = ax[0]
    speeds = list(o['speeds'])
    off = o['offset']
    data = []
    for i, sp in enumerate(speeds):
        for k, pl in enumerate(PLATFORMS):
            g = cand[(cand['speed'] == sp) & (cand['platform'] == pl)]
            v = g['duration_ms'].to_numpy() / 1000
            xx = i + (k - 0.5) * off + np.random.default_rng(i * 7 + k).uniform(-o['jitter'], o['jitter'], v.size)
            a.plot(xx, v, 'o', ms=o['marker_size'], alpha=o['alpha'], color=PLATFORM_COLORS[pl], mec='none',
                   label=PLATFORM_LABELS[pl] if i == 0 else None)
            if v.size:
                a.plot([i + (k - 0.5) * off - 0.14, i + (k - 0.5) * off + 0.14], [np.median(v)] * 2, color=INK, lw=1.3)
            data += [{'panel': 'duration_s', 'speed': sp, 'platform': pl, 'trial_key': t, 'value': x}
                     for t, x in zip(g['trial_key'], v)]
    a.axhline(C.MIN_DURATION_MS / 1000, color=INK2, lw=0.9)
    a.text(-0.45, C.MIN_DURATION_MS / 1000 * 1.08, f'E4b (= gate G2): {C.MIN_DURATION_MS / 1000:.2f} s',
           fontsize=o['note_fontsize'], color=INK2, ha='left')
    a.axhline(C.G2_MAX_DURATION_MS / 1000, color=INK2, lw=0.9, ls=DASH)
    a.text(-0.45, C.G2_MAX_DURATION_MS / 1000 * 1.08, f'gate G2: {C.G2_MAX_DURATION_MS / 1000:.0f} s (scored apart)',
           fontsize=o['note_fontsize'], color=INK2, ha='left')
    a.set_yscale(o['yscale'])
    if o['yscale'] == 'log':
        plain_log_ticks(a.yaxis, list(o['yticks']))
    a.set_xticks(range(len(speeds)), speeds)
    a.set_ylabel('approach duration (s, log)' if o['yscale'] == 'log' else 'approach duration (s)')
    a.set_title('Oracle approach duration by instructed speed')
    a.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1), markerscale=2, handletextpad=0.2)
    a = ax[1]
    v = cand['peak_gyro_dps'].to_numpy()
    a.hist(v, bins=np.arange(*o['peak_bins']), color=SERIES, edgecolor='white', linewidth=0.5)
    a.axvline(C.MIN_PEAK_GYRO_DPS, color=INK, lw=0.9, label=f'E4c: {C.MIN_PEAK_GYRO_DPS:.0f} deg/s')
    a.legend(loc='upper right', fontsize=o['note_fontsize'] + 1)
    a.set_xlabel('peak $|\\omega|$ in the approach (deg/s)')
    a.set_ylabel('complete attempts')
    a.set_title('How fast the phone turns')
    data += [{'panel': 'peak_gyro_dps', 'speed': s_, 'platform': p_, 'trial_key': t, 'value': x}
             for s_, p_, t, x in zip(cand['speed'], cand['platform'], cand['trial_key'], v)]
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        'Left: oracle approach duration (stored online_v1 segment) of the complete attempts by instructed speed and '
        f'platform (bar = median; primary median {du["primary_median_s"]:.2f} s). Right: peak angular '
        'speed within the approach. The solid lines mark the E4b and E4c thresholds (the bounds of gate G2 of '
        f'docs/09); the dashed line is the upper duration bound of G2 ({C.G2_MAX_DURATION_MS / 1000:.0f} s): '
        'longer approaches stay in the primary set and are scored apart (gated metrics).'),
        decision=f'Absurd segments (< {C.MIN_DURATION_MS / 1000:.2f} s, or a phone that does not turn) are excluded; '
                 'instructed speed separates durations only loosely, so models should not rely on it.')


def fig_S2_09_segmentation_contact(**kw: Any) -> dict[str, str]:
    name = 'S2_09_segmentation_contact'
    o = _opts(name, kw)
    c = D.load_input_csv('complete_attempts')
    seg = D.load_summary()['segmentation']
    toward = c['toward'].astype(bool)
    fig, ax = figure(o['width'], o['height'], 1, 2)
    a = ax[0]
    lim = o['diff_clip_ms']
    bins = np.arange(-lim, lim + 1, o['diff_bin_ms'])
    a.hist(c['d_onset_ms'].clip(-lim, lim), bins=bins, histtype='step', color=SERIES, lw=1.2, label='onset')
    a.hist(c['d_end_ms'].dropna().clip(-lim, lim), bins=bins, histtype='step', color=MUTED, lw=1.2, label='end')
    a.axvline(0, color=INK, lw=0.6)
    a.set_xlabel(f'offline_v1 minus stored online_v1 (ms; clipped at ±{lim:g})')
    a.set_ylabel('complete attempts')
    a.legend(loc='upper left')
    a.set_title(f"offline_v1 finds no end in {100 * seg['offline_no_end_share_v2']:.0f} % of street attempts")
    a = ax[1]
    lo, hi = o['touch_range']
    v = c.loc[toward, 'touch_frac'].dropna().clip(lo, hi)
    a.hist(v, bins=np.arange(lo, hi + 0.01, o['touch_bin']), color=SERIES, edgecolor='white', linewidth=0.5)
    a.axvline(0, color=INK, lw=0.8)
    a.axvline(1, color=INK, lw=0.8)
    a.text(1.02, a.get_ylim()[1] * 0.92, 'end', fontsize=o['note_fontsize'], color=INK2)
    a.text(0.02, a.get_ylim()[1] * 0.92, 'onset', fontsize=o['note_fontsize'], color=INK2)
    a.set_xlabel('first touch of the screen, as a fraction of the approach')
    a.set_ylabel('screen-toward attempts')
    a.set_title(f"Contact at {100 * seg['touch_position_median']:.0f} % of the approach (median)")
    data = c[['trial_key', 'protocol', 'd_onset_ms', 'd_end_ms', 'touch_frac', 'toward']]
    return _save(fig, name, data=data, caption=(
        'Left: where the analysis segmentation offline_v1 and the app\'s stored online_v1 disagree (the oracle is '
        'online_v1, which exists for every complete attempt). Right: the first screen touch after GO (nose contact) '
        f'falls at a median {100 * seg["touch_position_median"]:.0f} % of the oracle window: the last part of the '
        'window is the phone settling against the nose.'),
        decision='Oracle approach = stored online_v1 bounds; offline_v1 bounds are kept for ablations.')


def fig_S2_10_example_traces(**kw: Any) -> dict[str, str] | None:
    name = 'S2_10_example_traces'
    o = _opts(name, kw)
    ex = D.load_input('example_traces')
    if len(ex['trials']) != 2:
        print(f'{name}: no person with a typical left and right trial; not drawn', flush=True)
        return None
    fig, ax = figure(o['width'], o['height'], 3, 2, sharex='col', sharey='row')
    data = []
    for col, r in enumerate(ex['trials']):
        tt = _arr(r['t_s'])
        ch = {k: _arr(v) for k, v in r['channels'].items()}
        for k, ax_ in enumerate('xyz'):
            ax[0, col].plot(tt, ch[f'gyro_{ax_}'], color=AXIS3[k], lw=o['trace_lw'], label=f'$\\omega_{ax_}$')
            ax[1, col].plot(tt, ch[f'acc_{ax_}'], color=AXIS3[k], lw=o['trace_lw'], label=f'$a_{ax_}$')
        ax[2, col].plot(tt, ch['yaw_rate'], color=HAND_COLORS[r['hand']], lw=o['yaw_lw'],
                        label='yaw rate $\\omega\\cdot\\hat u$')
        ax[2, col].plot(tt, ch['gyro_horiz'], color=MUTED, lw=o['trace_lw'], label='tilt rate $|\\omega_\\perp|$')
        ev = r['events_s']
        for row in range(3):
            a = ax[row, col]
            a.axvspan(ev['onset'], ev['end'], color='#f3f2ee', zorder=0)
            if ev.get('return') is not None and ev.get('still again') is not None:
                a.axvspan(ev['return'], ev['still again'], color='#faf6ea', zorder=0)
            for nm, v in ev.items():
                if v is not None:
                    a.axvline(v, color=EVENT_COLORS.get(nm.split()[0], MUTED), lw=0.6,
                              ls='-' if nm in ('onset', 'end') else (0, (2, 2)))
        for nm, v in ev.items():
            if v is not None:
                ax[0, col].text(v, 1.0, nm, transform=ax[0, col].get_xaxis_transform(), rotation=90,
                                fontsize=o['event_fontsize'], color=INK2, ha='right', va='top')
        grip = str(r['orientation']).replace('_toward', '').replace('_', ' ')
        ax[0, col].set_title(f"{r['person_id']}, {HAND_LABELS[r['hand']].lower()} ({grip}, {r['speed']}; "
                             f"first-half yaw {r['first_half_yaw_deg']:+.1f}°)")
        ax[2, col].set_xlabel('time from GO (s)')
        right = ev['still again'] + o['xpad_right'] if ev.get('still again') is not None else o['xlim_right_default']
        ax[0, col].set_xlim(o['xlim_left'], right)
        data += [{'hand': r['hand'], 'trial_key': r['trial_key'], 'channel': k, 't_s': t, 'value': v}
                 for k, arr in ch.items() for t, v in zip(tt, arr)]
    ax[0, 0].set_ylabel('device frame\n$\\omega$ (deg/s)')
    ax[1, 0].set_ylabel('device frame\n$a$ (m/s$^2$)')
    ax[2, 0].set_ylabel('gravity frame\n(deg/s)')
    for row in range(3):  # legends sit over the still period before GO
        ax[row, 0].legend(loc='lower left', ncol=3 if row < 2 else 1, fontsize=o['legend_fontsize'], handlelength=1.2,
                          columnspacing=0.8)
    L = ex['trials'][0]
    grip = str(L['orientation']).replace('_toward', '').replace('_', ' ')
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f"One left and one right trial of {ex['person_id']} (same phone, grip ({grip}) and speed ({L['speed']})), "
        'standardised: device-frame angular velocity and linear acceleration (top two rows) and heading-free '
        'gravity-frame channels (bottom). Grey band: oracle approach; beige band: the return of the arm. The yaw rate '
        'about gravity has opposite signs early in the reach for the two hands.'),
        decision='What the canonical channels look like; why the return must be kept apart from the approach.')


# ───────────────────────────── 11-15: class means, mirror, first-half yaw, platforms, return ─────────────────────────────


def fig_S2_11_class_means(**kw: Any) -> dict[str, str]:
    name = 'S2_11_class_means'
    o = _opts(name, kw)
    cm = D.load_input('class_means')
    x = _arr(cm['x_pct'])
    chans = list(cm['channels'])
    nrows = math.ceil(len(chans) / o['ncols'])
    fig, axes = figure(o['width'], o['height'], nrows, o['ncols'])
    axes = np.atleast_2d(axes)
    data, persons = [], set()
    for a, ch in zip(axes.ravel(), chans):
        e = cm['channels'][ch]
        M = {h: _means(e['hands'][h]) for h in ('left', 'right')}
        for h in ('left', 'right'):
            mu, se = _mean_se(M[h])
            persons |= set(e['hands'][h]['persons'])
            a.fill_between(x, mu - se, mu + se, color=HAND_COLORS[h], alpha=o['band_alpha'], lw=0)
            a.plot(x, mu, color=HAND_COLORS[h], lw=o['lw'])
            data += [{'channel': ch, 'series': h, 'x_pct': xi, 'mean': m, 'se': s} for xi, m, s in zip(x, mu, se)]
        if e['mirror_sign'] < 0:
            mir = -M['left'].mean(axis=0)
            a.plot(x, mir, color=HAND_COLORS['right'], lw=o['mirror_lw'], ls=DASH)
            data += [{'channel': ch, 'series': 'mirrored left', 'x_pct': xi, 'mean': m, 'se': np.nan} for xi, m in zip(x, mir)]
        a.axhline(0, color=AXIS, lw=0.6)
        a.set_title(o['titles'].get(ch, ch))
        a.set_ylabel(o['units'].get(ch, ''))
    for a in axes[-1]:
        a.set_xlabel('time (% of the approach)')
    from matplotlib.lines import Line2D
    fig.legend(handles=hand_handles() + [Line2D([0], [0], color=HAND_COLORS['right'], lw=o['mirror_lw'], ls=DASH,
                                                label='mirrored left (chiral channels)')],
               loc='outside upper center', ncol=3, fontsize=o['legend_fontsize'])
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        'Time-normalised oracle approach, gravity frame: mean over persons of each person\'s mean trace (band = ± SE '
        f'across the {len(persons)} persons), left vs right hand. Dashed: the left-hand mean mirrored (chiral channels '
        'change sign); it should lie on the right-hand mean if the movement is mirror-symmetric.'),
        decision='The chiral channels carry the hand early in the reach; the mirror augmentation is plausible for '
                 'them, while the even channels show the two hands are not exact mirror images.')


def fig_S2_12_sign_patterns(**kw: Any) -> dict[str, str]:
    name = 'S2_12_sign_patterns'
    o = _opts(name, kw)
    sp = D.load_input('sign_patterns')
    fig, a = figure(o['width'], o['height'])
    data = []
    n_pat = 0
    for pl in PLATFORMS:
        p = sp['platforms'].get(pl)
        if not p:
            continue
        Ds = _arr(p['D_sorted'])
        n_pat = Ds.size
        a.plot(np.arange(1, Ds.size + 1), Ds, 'o', ms=o['marker_size'], color=PLATFORM_COLORS[pl], mec='none',
               label=PLATFORM_LABELS[pl])
        a.plot(p['rank_sigma_star'], p['D_sigma_star'], 'o', ms=o['ring_size'], mfc='none', mec=INK, mew=0.9)
        data += [{'platform': pl, 'rank': i + 1, 'distance': v, 'sigma_star': i + 1 == p['rank_sigma_star']}
                 for i, v in enumerate(Ds)]
    a.text(2.5, a.get_ylim()[1] - 0.04 * np.ptp(a.get_ylim()), f"circled: $\\sigma^*={sp['sigma_star']}$",
           fontsize=o['note_fontsize'], color=INK2, va='top')
    a.set_xlabel('sign pattern, sorted by distance')
    a.set_ylabel('normalised distance D($\\sigma$)')
    a.legend(loc='lower right')
    a.set_title(f'Left vs mirrored right: {n_pat} sign patterns')
    ranks = {pl: p['rank_sigma_star'] for pl, p in sp['platforms'].items()}
    first = all(r == 1 for r in ranks.values())
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f'For each of the {n_pat} sign patterns of (a_x, a_y, a_z, w_x, w_y, w_z), the distance between every person\'s '
        'mean left-hand waveform and the sign-flipped mean right-hand waveform of the mirrored grip (device frame, '
        f"channels scaled to unit RMS). The mirror predicted by the conventions, sigma* = {sp['sigma_star']}, is circled "
        '(rank ' + ', '.join(f'{r} on {PLATFORM_LABELS.get(pl, pl)}' for pl, r in ranks.items()) + ').'),
        decision=('Checks axes, signs and the iOS sign normalisation on real data: the theoretical mirror ranks first.'
                  if first else 'The theoretical mirror does not rank first on every platform: check the axes and '
                                'the sign normalisation before relying on mirror augmentation.'))


def fig_S2_13_first_half_yaw(**kw: Any) -> dict[str, str]:
    name = 'S2_13_first_half_yaw'
    o = _opts(name, kw)
    pp = D.load_table('first_half_yaw_per_person')
    tr = D.load_input_csv('primary_trials')
    fy = D.load_summary()['first_half_yaw']
    pooled = fy['pooled_auc']
    fig, ax = figure(o['width'], o['height'], 1, 2, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    od = pp.sort_values('auc_first_half_yaw_deg')
    yy = np.arange(len(od))
    a.hlines(yy, od[['auc_first_half_yaw_deg', 'auc_net_yaw_deg']].min(axis=1),
             od[['auc_first_half_yaw_deg', 'auc_net_yaw_deg']].max(axis=1), color=GRID, lw=1.0)
    a.plot(od['auc_net_yaw_deg'], yy, 'o', ms=3.5, mfc='white', mec=MUTED, mew=0.9, label='net yaw')
    a.plot(od['auc_first_half_yaw_deg'], yy, 'o', ms=4, color=SERIES, mec='white', mew=0.5, label='first-half yaw')
    a.axvline(0.5, color=AXIS, lw=0.8)
    a.set_yticks(yy, [f"{p} ({PLATFORM_LABELS.get(pl, pl)})" for p, pl in zip(od['person_id'], od['platform'])],
                 fontsize=o['ytick_fontsize'])
    a.set_xlim(*o['xlim'])
    a.set_xlabel('within-person AUC (right vs left)')
    a.legend(loc='upper left')
    a.grid(axis='y', visible=False)
    a.set_title(f"Per person: median AUC {fy['per_person_auc_median']:.2f}")
    a = ax[1]
    lim = o['clip_deg']
    bins = np.arange(-lim, lim + 1, o['bin_deg'])
    for h in ('left', 'right'):
        v = tr.loc[tr['hand'] == h, 'first_half_yaw_deg'].clip(-lim, lim)
        a.hist(v, bins=bins, histtype='stepfilled', alpha=o['fill_alpha'], color=HAND_COLORS[h])
        a.hist(v, bins=bins, histtype='step', lw=1.1, color=HAND_COLORS[h], label=HAND_LABELS[h])
    a.axvline(0, color=INK, lw=0.6)
    a.set_xlabel(f'first-half yaw about gravity (deg; clipped at ±{lim:g})')
    a.set_ylabel('primary trials')
    a.legend(loc='upper left')
    a.set_title(f"Pooled AUC {pooled['first_half_yaw_deg']:.2f} (net yaw {pooled['net_yaw_deg']:.2f})")
    data = pd.concat([
        pd.DataFrame({'panel': 'per_person', 'person_id': od['person_id'], 'platform': od['platform'], 'hand': '',
                      'trial_key': '', 'first_half_yaw_auc': od['auc_first_half_yaw_deg'], 'net_yaw_auc': od['auc_net_yaw_deg'],
                      'first_half_yaw_deg': np.nan}),
        pd.DataFrame({'panel': 'trials', 'person_id': tr['person_id'], 'platform': tr['platform'], 'hand': tr['hand'],
                      'trial_key': tr['trial_key'], 'first_half_yaw_auc': np.nan, 'net_yaw_auc': np.nan,
                      'first_half_yaw_deg': tr['first_half_yaw_deg']})], ignore_index=True)
    fhy, net = pooled['first_half_yaw_deg'], pooled['net_yaw_deg']
    sec = pooled.get('second_half_yaw_deg')
    compare = ('separates the hands better than the net yaw over the whole approach' if fhy > net else
               'does not separate the hands better than the net yaw over the whole approach')
    return _save(fig, name, data=data, caption=(
        f'The pilot finding re-checked on all {len(pp)} persons: the yaw about gravity accumulated over the first half '
        f'of the approach {compare} (pooled AUC {fhy:.2f} against {net:.2f}'
        + (f'; second half {sec:.2f}' if sec is not None else '') + f'; within-person median '
        f"{fy['per_person_auc_median']:.2f}, above 0.5 for {fy['persons_auc_gt_0p5']} of {len(pp)} persons). Left: "
        'within-person AUC per person; right: pooled distribution.'),
        decision='Hand evidence is in the time course of the chiral channels, so models get time-resolved '
                 'gravity-frame inputs, not only whole-movement integrals.')


def fig_S2_14_platforms(**kw: Any) -> dict[str, str]:
    name = 'S2_14_platforms'
    o = _opts(name, kw)
    pm = D.load_input('platform_means')
    tr = D.load_input_csv('primary_trials')
    x = _arr(pm['x_pct'])
    fig, ax = figure(o['width'], o['height'], 1, 3)
    ax[1].sharey(ax[0])
    data = []
    for j, pl in enumerate(PLATFORMS):
        e = pm['platforms'].get(pl)
        if not e:
            continue
        a = ax[j]
        for h in ('left', 'right'):
            mu, se = _mean_se(_means(e['hands'][h]))
            a.fill_between(x, mu - se, mu + se, color=HAND_COLORS[h], alpha=o['band_alpha'], lw=0)
            a.plot(x, mu, color=HAND_COLORS[h], lw=o['lw'])
            data += [{'panel': 'yaw_rate', 'platform': pl, 'hand': h, 'x': xi, 'value': m, 'se': s}
                     for xi, m, s in zip(x, mu, se)]
        a.axhline(0, color=AXIS, lw=0.6)
        a.set_title(f"{PLATFORM_LABELS[pl]} ({e['n_trials']} trials, {e['n_persons']} persons)")
        a.set_xlabel('time (% of the approach)')
    ax[0].set_ylabel('yaw rate about gravity (deg/s)')
    ax[0].legend(handles=hand_handles(), loc='lower left', fontsize=o['legend_fontsize'])
    a = ax[2]
    lim = o['clip_deg']
    med = {}
    for i, pl in enumerate(PLATFORMS):
        for j, h in enumerate(('left', 'right')):
            v = tr.loc[(tr['platform'] == pl) & (tr['hand'] == h), 'first_half_yaw_deg'].clip(-lim, lim).to_numpy()
            if not v.size:
                continue
            xx = i + (j - 0.5) * o['offset'] + np.random.default_rng(10 * i + j).uniform(-o['jitter'], o['jitter'], v.size)
            a.plot(xx, v, 'o', ms=o['marker_size'], alpha=o['alpha'], color=HAND_COLORS[h], mec='none')
            a.plot([i + (j - 0.5) * o['offset'] - 0.13, i + (j - 0.5) * o['offset'] + 0.13], [np.median(v)] * 2,
                   color=INK, lw=1.3)
            med[(pl, h)] = float(np.median(v))
            data += [{'panel': 'first_half_yaw', 'platform': pl, 'hand': h, 'x': i, 'value': y, 'se': np.nan} for y in v]
    a.axhline(0, color=AXIS, lw=0.6)
    a.set_xticks(range(len(PLATFORMS)), [PLATFORM_LABELS[p] for p in PLATFORMS])
    a.set_ylabel('first-half yaw (deg)')
    a.set_title('First-half yaw by platform')
    same = all((med.get((pl, 'right'), 0) > 0) and (med.get((pl, 'left'), 0) < 0) for pl in PLATFORMS) or \
        all((med.get((pl, 'right'), 0) < 0) and (med.get((pl, 'left'), 0) > 0) for pl in PLATFORMS)
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        'iOS vs Android after standardisation: class-mean yaw rate about gravity (± SE across persons) and the '
        'first-half yaw per trial (bar = median). '
        + ('The hands separate with the same sign on both platforms, which confirms the platform sign normalisation.'
           if same else 'The median first-half yaw does not have the same sign per hand on both platforms: check the '
                        'platform sign normalisation.')),
        decision='One model for both platforms; platform stays a reporting stratum (and a leave-one-platform-out check).')


def fig_S2_15_return_vs_approach(**kw: Any) -> dict[str, str]:
    name = 'S2_15_return_vs_approach'
    o = _opts(name, kw)
    rm = D.load_input('return_means')
    rr = D.load_summary()['return_risk']
    x = _arr(rm['x_pct'])
    fig, ax = figure(o['width'], o['height'], 1, 2, sharey=True)
    data = []
    for a, part, title in ((ax[0], 'approach', 'Approach (oracle window)'),
                           (ax[1], 'return', 'Return of the arm (after the end cue)')):
        for h in ('left', 'right'):
            mu, se = _mean_se(_means(rm['parts'][part][h]))
            a.fill_between(x, mu - se, mu + se, color=HAND_COLORS[h], alpha=o['band_alpha'], lw=0)
            a.plot(x, mu, color=HAND_COLORS[h], lw=o['lw'])
            data += [{'movement': part, 'hand': h, 'x_pct': xi, 'mean': m, 'se': s} for xi, m, s in zip(x, mu, se)]
        a.axhline(0, color=AXIS, lw=0.6)
        a.set_title(title)
        a.set_xlabel('time (% of the movement)')
    ax[0].set_ylabel('yaw rate about gravity (deg/s)')
    ax[0].legend(handles=hand_handles(), loc='lower left', fontsize=o['legend_fontsize'])
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f"Class-mean yaw rate in the approach and in the return of the same {rm['n_trials']} trials. The sign rule "
        f"fitted on approaches (right iff first-half yaw > 0) is {_pct(rr['sign_rule_accuracy_on_approach'])} correct on "
        f"the approaches and {_pct(rr['sign_rule_accuracy_on_return_first_half'])} on the returns (net yaw of the return: "
        f"{_pct(rr['sign_rule_accuracy_on_return_net_yaw'])})."),
        decision='The return is a time-reversed approach; a cue-free detector that picks it up would feed the '
                 'model a movement whose chiral evidence is scrambled or reversed, so returns are never pooled '
                 'with approaches and the detector must reject them.')


# ───────────────────────────── 16-20: cue-free, closure, flags, stress, label consistency ─────────────────────────────

OUTCOME_TEXT = {'approach': 'approach (IoU ≥ 0.5)', 'approach_partial': 'part of the approach',
                'earlier_movement': 'an earlier movement', 'return': 'the return', 'later_movement': 'a later movement',
                'none': 'nothing'}


def fig_S2_16_cuefree_v0(**kw: Any) -> dict[str, str]:
    name = 'S2_16_cuefree_v0'
    o = _opts(name, kw)
    ev = D.load_table('cuefree_v0_per_trial')
    still = D.load_input_csv('stillness_before_onset')
    summ = D.load_summary()['cuefree_v0']['all']
    pre = still['still_s_gyro'].to_numpy()
    min_still = float(D.load_summary()['cuefree_v0']['params']['min_still_s'])
    outs = list(OUTCOME_TEXT)
    fig, ax = figure(o['width'], o['height'], 1, 3, gridspec_kw={'width_ratios': list(o['width_ratios'])})
    a = ax[0]
    data = []
    for j, pl in enumerate(PLATFORMS):
        g = ev[ev['platform'] == pl]
        share = [float((g['outcome'] == oc).mean()) if len(g) else 0.0 for oc in outs]
        a.barh(np.arange(len(outs)) + (j - 0.5) * 0.36, share, height=0.34, color=PLATFORM_COLORS[pl],
               label=PLATFORM_LABELS[pl])
        data += [{'panel': 'first_detection', 'platform': pl, 'item': oc, 'value': s_} for oc, s_ in zip(outs, share)]
    a.set_yticks(range(len(outs)), [OUTCOME_TEXT[oc] for oc in outs], fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    a.set_xlabel('share of primary trials')
    a.legend(loc='lower right')
    a.grid(axis='y', visible=False)
    a.set_title('What v0 finds first')
    a = ax[1]
    lim = o['error_clip_ms']
    e = ev.loc[ev['outcome'].isin(['approach', 'approach_partial']), 'onset_error_ms'].clip(-lim, lim)
    a.hist(e, bins=np.arange(-lim, lim + 1, o['error_bin_ms']), color=SERIES, edgecolor='white', linewidth=0.4)
    a.axvline(0, color=INK, lw=0.6)
    a.set_xlabel('detected − oracle onset (ms)')
    a.set_ylabel('trials')
    a.set_title(f"Onset error, median {summ['onset_error_median_ms']:.0f} ms")
    data += [{'panel': 'onset_error_ms', 'platform': p_, 'item': '', 'value': v} for p_, v in
             zip(ev.loc[e.index, 'platform'], ev.loc[e.index, 'onset_error_ms'])]
    a = ax[2]
    a.hist(np.clip(pre, 0, o['still_clip_s']), bins=np.arange(0, o['still_clip_s'] + 0.01, o['still_bin_s']),
           color=SERIES, edgecolor='white', linewidth=0.4)
    a.axvline(min_still, color=INK, lw=0.9)
    a.set_xlabel('stillness before onset (s)')
    a.set_ylabel('trials')
    a.set_title(f"{100 * np.mean(pre >= min_still):.0f} % have ≥ {min_still:g} s")
    data += [{'panel': 'stillness_s', 'platform': p_, 'item': '', 'value': v} for p_, v in zip(still['platform'], pre)]
    fails = {oc: summ[f'share_{oc}'] for oc in outs if oc != 'approach'}
    main_fail = max(fails, key=fails.get)
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        f"Baseline cue-free detector v0 (first sustained movement after ≥ {min_still:g} s of stillness, no GO time) on "
        f"the {len(ev)} primary recordings: it finds the approach in {_pct(summ['share_approach'])} (IoU ≥ 0.5), an "
        f"earlier movement in {_pct(summ['share_earlier_movement'])} and the return in {_pct(summ['share_return'])}; "
        "right: how long the phone was still before the approach (measured in v0's own signal). That stillness was "
        'enforced by the collection app (hold-still gate, foreperiod, reaction time), so these shares are upper bounds '
        'for production (see S2_19).'),
        decision=(f'Baseline and evaluation contract for the cue-free lane: the main failure is '
                  f'{OUTCOME_TEXT[main_fail]} ({_pct(fails[main_fail])})'
                  + (', not the return.' if main_fail != 'return' else '.')))


def fig_S2_17_gravity_closure(**kw: Any) -> dict[str, str]:
    name = 'S2_17_gravity_closure'
    o = _opts(name, kw)
    tr = D.load_input_csv('primary_trials')
    gc = D.load_summary()['gravity_closure']
    thr = D.CLOSURE_REPORT_DEG
    lim = o['clip_deg']
    fig, a = figure(o['width'], o['height'])
    for pl in PLATFORMS:
        xx = tr.loc[tr['platform'] == pl, 'closure_deg'].clip(0, lim)
        a.hist(xx, bins=np.arange(0, lim + 0.1, o['bin_deg']), histtype='step', lw=1.2, color=PLATFORM_COLORS[pl],
               label=PLATFORM_LABELS[pl])
    a.axvline(thr, color=INK, lw=0.8)
    a.set_xlabel(f'gyro-gravity closure (deg; clipped at {lim:g})')
    a.set_ylabel('primary trials')
    a.legend()
    a.set_title(f"Median {gc['median_deg']:.1f}°, {100 * gc['share_over_3deg']:.0f} % above {thr:g}°")
    return _save(fig, name, data=tr[['trial_key', 'platform', 'closure_deg']], caption=(
        'Angle between the rest gravity before the approach, carried through the movement by the gyroscope, and the '
        'rest gravity at the nose (no gyro-bias correction). Small closures validate the axis mapping and the '
        'gyro-propagated gravity of the gravity-frame channels.'),
        decision='gravity_v1 without bias subtraction is good enough for the heading-free channels.')


def fig_S2_18_flags_gates(**kw: Any) -> dict[str, str]:
    name = 'S2_18_flags_gates'
    o = _opts(name, kw)
    fg = D.load_table('flags_and_gates')
    n_prim = D.load_summary()['demographics']['primary_trials']
    fig, a = figure(o['width'], o['height'])
    y = np.arange(len(fg))
    for j, pl in enumerate(PLATFORMS):
        a.barh(y + (j - 0.5) * 0.36, fg[pl], height=0.34, color=PLATFORM_COLORS[pl], label=PLATFORM_LABELS[pl])
    for i, r in fg.iterrows():
        a.text(max(r['android'], r['ios']) + o['label_pad'], i, f"{r['primary_trials']} trials, "
               f"{_plural(int(r['persons']), 'person')}", fontsize=o['note_fontsize'], color=INK2, va='center')
    a.set_yticks(y, [D.FLAG_LABELS.get(c, c) for c in fg['column']], fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    a.set_xlim(0, fg[list(PLATFORMS)].to_numpy().max() * o['xmax_factor'])
    a.set_xlabel('primary trials (kept: flags are reported, gates are scored apart)')
    a.legend(loc='lower right')
    a.grid(axis='y', visible=False)
    a.set_title(f"Flags and predictor gates in the {n_prim} primary trials")
    return _save(fig, name, data=fg[['column', 'kind', 'primary_trials', 'android', 'ios', 'persons']], caption=(
        'Flags of docs/02 T8-T12 (kept in every analysis) and the predictor gates G1 and G2 of docs/09 D5 (trials '
        'that fail a gate stay in training and in the headline metrics, and every result also reports the '
        'metrics on the trials that pass both gates, as the predictor will see them).'),
        decision='No new exclusions: grip mismatches keep their hand label (T8); gate failures are scored apart.')


def fig_S2_19_cuefree_stress(**kw: Any) -> dict[str, str]:
    name = 'S2_19_cuefree_stress'
    o = _opts(name, kw)
    summ = D.load_table('cuefree_v0_stress')
    meta = D.load_input('stress_settings')
    allp = summ[summ['platform'] == 'all'].set_index('setting')
    allp.index = allp.index.astype(str)
    grid = allp.drop(index=['recorded', 'random'], errors='ignore').copy()
    grid['k'] = grid['keep_s_mean']
    grid = grid.sort_values('k')
    fig, a = figure(o['width'], o['height'])
    show = [('share_approach', 'the approach (IoU ≥ 0.5)', SERIES, '-'),
            ('share_approach_partial', 'part of the approach', SERIES, DASH),
            ('share_earlier_movement', 'an earlier movement', '#eb6834', '-'),
            ('share_return', 'the return', MUTED, '-'), ('share_none', 'nothing', INK2, (0, (1, 1.5)))]
    ys = {col: float(allp.at['recorded', col]) for col, *_ in show}
    pos, last = {}, -1.0
    for col in sorted(ys, key=ys.get):  # right-margin labels of the recorded shares, at least label_gap apart
        pos[col] = max(ys[col], last + o['label_gap'])
        last = pos[col]
    data = []
    for col, lab, colr, ls in show:
        a.plot(grid['k'], grid[col], marker='o', ms=o['marker_size'], color=colr, ls=ls, lw=1.2, label=lab)
        a.annotate(f"{100 * ys[col]:.0f} %", (o['label_x'], pos[col]), fontsize=o['note_fontsize'], color=colr,
                   va='center', annotation_clip=False)
        data += [{'setting': s_, 'keep_s': k, 'outcome': col.replace('share_', ''), 'share': v}
                 for s_, k, v in zip(grid.index, grid['k'], grid[col])]
        data += [{'setting': s_, 'keep_s': np.nan, 'outcome': col.replace('share_', ''), 'share': float(allp.at[s_, col])}
                 for s_ in ('recorded', 'random') if s_ in allp.index]
    ms = float(meta['min_still_s'])
    a.axvline(ms, color=INK, lw=0.8)
    a.text(ms + 0.03, 0.97, f"v0's {ms:g} s stillness rule", fontsize=o['note_fontsize'], color=INK2, va='top',
           transform=a.get_xaxis_transform())
    a.set_xlim(*o['xlim'])
    a.set_ylim(*o['ylim'])
    a.set_xlabel('stillness kept before the approach (s); right margin: as recorded')
    a.set_ylabel('share of primary trials')
    a.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), fontsize=o['legend_fontsize'], title='v0 finds first',
             title_fontsize=7)
    rnd = allp.loc['random']
    a.set_title(f"Shortened stillness: with a random 0-{meta['random_max_s']:g} s, v0 finds the approach first in "
                f"{100 * rnd['share_approach']:.0f} % (as recorded: {100 * allp.at['recorded', 'share_approach']:.0f} %)")
    return _save(fig, name, data=pd.DataFrame(data), caption=(
        "Stress test of the cue-free baseline v0: the stillness before each approach, which the collection app "
        "enforced, is cut to the given number of seconds (the samples right after the preceding activity are "
        f"removed, the rest of the recording moves back), and v0 is run again. Below v0's own {ms:g} s stillness rule "
        "it can no longer separate the approach from what came before."),
        decision='The v0 shares on the recorded data are upper bounds; a production detector must not rely on a '
                 'long stillness (or the inference app must ask for one and check it).')


def fig_S2_20_label_consistency(**kw: Any) -> dict[str, str]:
    name = 'S2_20_label_consistency'
    o = _opts(name, kw)
    b = D.load_table('label_consistency_blocks')
    sess = sorted(b['session_label'].unique(), key=lambda s: (s.split('-')[0], s))
    yi = {s: i for i, s in enumerate(sess)}
    fig, a = figure(o['width'], o['height'])
    for _, r in b.iterrows():
        col = HAND_COLORS[r['hand']]
        share = r['fhy_share_agree']
        size = o['size_base'] + o['size_per_trial'] * r['n']
        a.scatter(r['block_index'], yi[r['session_label']], s=size, facecolor=col, alpha=0.15 + 0.85 * share,
                  edgecolor=col, linewidth=0.6)
        if r['fhy_swap_signature']:
            a.scatter(r['block_index'], yi[r['session_label']], s=size + o['ring_extra'], facecolor='none', edgecolor=INK,
                      linewidth=1.1)
        a.text(r['block_index'] + 0.32, yi[r['session_label']], f"{int(r['fhy_agree'])}/{int(r['n'])}",
               fontsize=o['text_fontsize'], color=INK2, va='center')
    a.set_yticks(range(len(sess)), sess, fontsize=o['ytick_fontsize'])
    a.invert_yaxis()
    b0, b1 = int(b['block_index'].min()), int(b['block_index'].max())
    a.set_xticks(range(b0, b1 + 1))
    a.set_xlim(b0 - 0.6, b1 + 0.9)
    a.set_xlabel('block of the session plan (hands alternate; street visits: tier 1 = blocks 1-2; the pilot starts at 0)')
    a.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([0], [0], marker='o', ls='', color=HAND_COLORS[h], label=f'instructed {h} hand')
                      for h in ('left', 'right')]
             + [Line2D([0], [0], marker='o', ls='', mfc='none', mec=INK, ms=8, label='swap signature')],
             loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=o['legend_fontsize'], ncol=3)
    a.set_title('Label consistency per block: trials whose first-half yaw sign agrees with the instructed hand')
    return _save(fig, name, data=b[['session_label', 'block_index', 'hand', 'n', 'fhy_agree', 'fhy_share_agree',
                                    'fhy_swap_signature']], caption=(
        'Every planned block of every session (dot size = trials, fill = share of trials whose first-half yaw sign '
        'agrees with the instructed hand; text = agreeing/total). Ringed: the swap signature, two adjacent blocks of '
        'opposite hands in which every trial contradicts its label, as a hand swap at the start of a session would '
        'produce. The labels are never changed automatically; the D6 screen removes flagged trials from training '
        'only, and a sensitivity analysis drops the ringed blocks.'),
        decision='Raise the ringed blocks with the researcher; keep the labels; report results with and without them.')


FIGURE_FUNCTIONS = {
    'S2_01_demographics': fig_S2_01_demographics, 'S2_02_exclusion_flow': fig_S2_02_exclusion_flow,
    'S2_03_completion_durations': fig_S2_03_completion_durations, 'S2_04_conditions': fig_S2_04_conditions,
    'S2_05_sampling_gaps': fig_S2_05_sampling_gaps, 'S2_06_noise_vs_quantisation': fig_S2_06_noise_vs_quantisation,
    'S2_07_spectrum': fig_S2_07_spectrum, 'S2_08_durations_peaks': fig_S2_08_durations_peaks,
    'S2_09_segmentation_contact': fig_S2_09_segmentation_contact, 'S2_10_example_traces': fig_S2_10_example_traces,
    'S2_11_class_means': fig_S2_11_class_means, 'S2_12_sign_patterns': fig_S2_12_sign_patterns,
    'S2_13_first_half_yaw': fig_S2_13_first_half_yaw, 'S2_14_platforms': fig_S2_14_platforms,
    'S2_15_return_vs_approach': fig_S2_15_return_vs_approach, 'S2_16_cuefree_v0': fig_S2_16_cuefree_v0,
    'S2_17_gravity_closure': fig_S2_17_gravity_closure, 'S2_18_flags_gates': fig_S2_18_flags_gates,
    'S2_19_cuefree_stress': fig_S2_19_cuefree_stress, 'S2_20_label_consistency': fig_S2_20_label_consistency,
}


def draw(only: list[str] | None = None) -> list[str]:
    """Draw every figure of the unit (or those matching ``only``) from the saved files."""
    done = []
    for name, fn in FIGURE_FUNCTIONS.items():
        if PL.selected(name, only):
            fn()
            done.append(name)
    return done
