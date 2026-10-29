"""Captions and decision sentences of the x1 figures, from saved files and the numbers entries only (no drawing, no
matplotlib): ``x1_figures`` puts them on the figures and ``x1_report`` lists them in the summary, so the summary
does not depend on the figures having been drawn, and every number in a figure text is a numbers entry."""
from __future__ import annotations

from typing import Any

from stage2.lanes import x1_report as XR
from stage2.lanes import x1_stats as XS

#: The figures of the lane, in order (``x1_figures.FIGURE_FUNCTIONS`` draws each).
FIGURE_NAMES = ('X1_01_example_traces', 'X1_02_preparation', 'X1_03_spectra', 'X1_04_movement',
                'X1_05_speed_scaling', 'X1_06_repeats', 'X1_07_hand_signature', 'X1_08_zscore_summary',
                'X1_09_multivariate', 'X1_10_pd01_vs_reference')
#: The characteristics of the strip-plot figures (also their panels, in this order).
STRIP_KEYS = {
    'X1_02_preparation': ('hold_wait_s', 'false_starts', 'rt_s', 'pre_rms_3_7', 'pre_rms_7_12', 'pre_ratio_db',
                          'pre_rms_05_3', 'pre_sway_gyro', 'nose_rms_3_7', 'nose_rms_7_12', 'attempts_per_accepted',
                          'nose_no_return'),
    'X1_04_movement': ('duration_s', 'active_s', 'settle_s', 'peak_speed_mv', 'peak_acc', 'touch_acc', 'touch_s',
                       'rot_after_touch', 'rotation_total_mv', 'tilt_change', 'ldlj', 'app_rms_7_12'),
}


def _join(items: list[str]) -> str:
    """'a, b and c'; with '; ' between items when an item itself contains a comma."""
    if len(items) <= 1:
        return ''.join(items)
    sep = '; ' if any(',' in i for i in items) else ', '
    return sep.join(items[:-1]) + (';' if sep == '; ' else '') + ' and ' + items[-1]


def pattern(N: XR.Nums, keys: tuple[str, ...] | list[str]) -> str:
    """Which of these characteristics put the case beyond every reference person, outside the reference
    interquartile range, or inside it (from the numbers entries)."""
    beyond, outside, inside = [], [], []
    for k in keys:
        if not N.has(f'{k}_pct'):
            continue
        lab = XS.lc(XS.BY_KEY[k].label) + (' *' if XS.BY_KEY[k].post_hoc else '')
        if N.v(f'{k}_beyond'):
            beyond.append(f'{lab} ({"above" if N.v(f"{k}_above") else "below"}'
                          f'{"" if N.v(f"{k}_robust") else ", not robust"})')
        elif N.v(f'{k}_pd') < N.v(f'{k}_qone') or N.v(f'{k}_pd') > N.v(f'{k}_qthree'):
            outside.append(lab)
        else:
            inside.append(lab)
    parts = []
    case = 'The case'
    if beyond:
        parts.append(f'{case} is beyond every reference person in {_join(beyond)}.')
    if outside:
        parts.append(f'Outside the reference interquartile range: {_join(outside)}.')
    if inside:
        parts.append(f'Inside it: {_join(inside)}.')
    return ' '.join(parts)


def _x1_01(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    return {'caption': (f'One left-hand upright natural reach of {case} (left; the most median-like of the case\'s '
                        f'matched trials) next to the reference trial closest to the reference persons\' medians '
                        f'(right): angular speed, yaw rate about gravity and linear acceleration over time from the '
                        f'onset (shaded: the segmented approach; dashed: the first screen touch). All '
                        f'{N("case_n_accepted")} accepted trials of the case are in the figure data.'),
            'decision': ('The raw signals behind the compared characteristics: the angular speed, the hand-relevant '
                         'yaw rate and the linear acceleration of the reach, its first screen touch and the wait '
                         'before its end. ' + pattern(N, ('app_rms_7_12', 'touch_acc', 'touch_s', 'settle_s')))}


def _x1_02(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    return {'caption': (f'Preparation, hold, the phone at the nose and task behaviour: each grey dot is one of the '
                        f'{N("n_ref_persons")} modelling persons (the mean of their left-hand upright trials; attempts '
                        f'per accepted trial over all their attempts), with their median and interquartile box; the '
                        f'diamond is {case} (its trials as ticks). Under the case: its percentile, or above / below all '
                        f'when it lies beyond every reference person. "No return within the recording" is a share of '
                        f'trials, because the time at the nose is censored when the recording stops first. '
                        f'*: post hoc.'),
            'decision': ('Preparation, hold and task behaviour. ' + pattern(N, STRIP_KEYS['X1_02_preparation']) + ' '
                         + XR._ratio_driver(N, D, 'pre').replace('It comes', 'The hold power ratio comes'))}


def _x1_03(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    return {'caption': (f'Power spectra of the angular velocity (three axes summed) in the hold before the movement '
                        f'({N("pre_window_s")} s, {N("res_pre_resolution_hz")}-Hz resolution; left-hand upright '
                        f'trials), during the reach (left-hand upright natural trials) and at the nose '
                        f'({N("nose_window_s")} s, {N("res_nose_resolution_hz")}-Hz resolution; left-hand upright '
                        f'trials): median and interquartile band of the modelling persons\' person-mean spectra and '
                        f'{case}\'s mean and trials. Shaded: the 3-7 and 7-12 Hz bands; in the short windows a tone '
                        f'leaks across their border (a {N("res_tone_hz")}-Hz tone puts {N("res_pre_leak_7_12")} deg/s '
                        f'RMS into 7-12 Hz in the hold window).'),
            'decision': (f'Where in the trial the case\'s spectrum departs from the reference. 7-12 Hz band RMS: hold '
                         f'{N.rank("pre_rms_7_12")}, reach {N.rank("app_rms_7_12")}, nose {N.rank("nose_rms_7_12")}; '
                         f'3-7 Hz band RMS: hold {N.rank("pre_rms_3_7")}, reach {N.rank("app_rms_3_7")}, nose '
                         f'{N.rank("nose_rms_3_7")}. The case\'s mean reach spectrum lies {N("spec_reach_7_12_min")}-'
                         f'{N("spec_reach_7_12_max")} times the reference median across 7-12 Hz, and {N.peak()}. The '
                         f'hold spectrum lies at {N("spec_pre_05_3_min")}-{N("spec_pre_05_3_max")} times the reference '
                         f'median in 0.5-3 Hz. '
                         + XR._ratio_driver(N, D, 'pre').replace('It comes', 'The hold power ratio comes'))}


def _x1_04(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    return {'caption': (f'The reach (left hand, upright, natural speed): each grey dot is one of the modelling persons '
                        f'(up to {N("n_ref_persons")}; the first-touch panels only the {N("touch_s_nref")} who touched '
                        f'the screen in these trials), with their median and interquartile box; the diamond is {case} '
                        f'and the ticks its trials. Under the case: its percentile, or above / below all when it lies '
                        f'beyond every reference person. Accelerations of clip-flagged trials are lower bounds. '
                        f'*: post hoc.'),
            'decision': ('Speed, first screen touch, amplitude, smoothness and band power of the reach. '
                         + pattern(N, STRIP_KEYS['X1_04_movement']))}


def _x1_05(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    counts = _join([f'{N(f"case_n_lua_{sp}")} {sp}' for sp in ('slow', 'natural', 'fast')])
    return {'caption': (f'Instructed speed (left hand, upright): each grey line is one of the modelling persons (their '
                        f'mean per speed), the dark line their median, the diamonds {case} ({counts} trial(s)).'),
            'decision': (f'How much the reach changes with the instructed speed: {case}\'s slow-to-fast duration '
                         f'ratio is {N.rank("speed_ratio")} ({N("speed_ratio_pd")} against a median of '
                         f'{N("speed_ratio_med")}), from {N("case_n_lua_slow")} slow and {N("case_n_lua_fast")} fast '
                         f'trial(s), so a single trial can move it.')}


def _x1_06(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    return {'caption': (f'The first {N("cv_duration_ncase")} left-hand upright natural reaches in visit order: each '
                        f'grey line is one of the {N("cv_duration_nref")} modelling persons with at least as many, the '
                        f'dark line their median, the diamonds {case} (whose reaches come from '
                        f'{N("case_repeat_sessions")} sessions).'),
            'decision': (f'Change across repeats (sequence effect): {case}\'s change of the voluntary peak speed per '
                         f'repeat is {N.rank("seq_peak_speed_mv")} ({N("seq_peak_speed_mv_pd")} % per repeat) and of '
                         f'the duration {N.rank("seq_duration")}; with {N("cv_duration_ncase")} repeats and '
                         f'{N("cv_duration_nref")} reference persons a trend can only be suggested.')}


def _x1_07(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    rep = (f'; {case}\'s repeat-visit free trial dashed, left out of its mean' if N.v('case_n_free_repeat') else '')
    return {'caption': (f'The hand signature without a classifier: cumulative yaw about gravity over the time-normalised '
                        f'approach, mirror-normalised so that positive is the direction typical of the hand used (its '
                        f'value at half of the approach, dashed line, is the first-half yaw; at the end, the net yaw). '
                        f'Left: left-hand upright trials (grey: each modelling person\'s mean; band: interquartile '
                        f'range); middle: the first free-choice trial of each modelling person ({N("n_ref_free")} '
                        f'trials){rep}; right: the first-half and the net yaw per person, with {case}\'s percentile '
                        f'under each group.'),
            'decision': (f'Whether the chiral hand evidence is typical for the hand used: {case}\'s hand-consistent '
                         f'first-half yaw is {N.rank("fhy_hand")} and its early yaw peak {N.rank("yaw_peak_hand")} *, '
                         f'but the net yaw turns back: {N.rank("net_yaw_hand")}, reversed in {N("chk_ny_k")} of '
                         f'{N("chk_ny_n")} accepted trials (reference persons: median {N("chk_ny_ref_median")}, '
                         f'maximum {N("chk_ny_ref_max")}).')}


def _x1_08(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    case = D['case']
    p = 'null_pre_lfl'
    return {'caption': (f'Top: every characteristic of {case} in its matched comparison, robust z against the '
                        f'{N("n_ref_persons")} modelling persons\' person-level values on the characteristic\'s declared '
                        f'scale (grey band: |z| < {N("z_clear")}), percentile at the right. Diamonds: beyond every '
                        f'reference person (filled: robust to every check; open: not robust); circles: inside the '
                        f'reference range; shares of binary outcomes have no z (percentile only). *: post hoc. Bottom: '
                        f'the null reference of the count: for each reference person, the number of pre-specified '
                        f'characteristics (left) and phenomena (right) in which they lie beyond all other reference '
                        f'persons, judged on as many trials as {case} made; the diamond is {case} against all of them.'),
            'decision': (f'The overview, and how unusual the count of differences is: {case} lies beyond every '
                         f'reference person in {N(f"{p}_case")} pre-specified characteristics ({N(f"{p}_case_phen")} '
                         f'phenomena); {N(f"{p}_n_ge")} of the {N("n_ref_persons")} reference persons '
                         f'{N.verb(f"{p}_n_ge", "reaches", "reach")} that count (rank p {N(f"{p}_p")}) and '
                         f'{N(f"{p}_n_ge_phen")} {N.verb(f"{p}_n_ge_phen", "reaches", "reach")} the number of phenomena '
                         f'(rank p {N(f"{p}_p_phen")})'
                         + (': not unique.' if N.v(f'{p}_n_ge') or N.v(f'{p}_n_ge_phen') else '.'))}


def _x1_09(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    mv = D['mv']
    case = D['case']
    labels = '; '.join(XS.lc(mv['labels'][k]) for k in mv['keys'])
    return {'caption': (f'Multivariate view of the pre-specified key characteristics ({labels}) on their declared '
                        f'scales, robust-standardised against the {N("n_ref_persons")} modelling persons. Left: their '
                        f'first two principal components (fitted on the reference persons; {case} projected). Middle: '
                        f'Mahalanobis distances with a Ledoit-Wolf covariance, each reference person left out of its '
                        f'own standardisation and covariance; the line is {case}. Right: the number of reference '
                        f'persons at least as far as {case} with each characteristic left out, on declared and on '
                        f'linear scales.'),
            'decision': (f'Taken together: {N("mv_n_beyond")} of the {N("n_ref_persons")} reference persons '
                         f'{N.verb("mv_n_beyond")} at least as far from the reference centre as {case} (rank p '
                         f'{N("mv_p")}, floor {N("mv_p_floor")}); with one characteristic left out, '
                         f'{N("mv_loco_min_beyond")} to {N("mv_loco_max_beyond")}. On linear scales '
                         f'{N("mv_lin_n_beyond")} {N.verb("mv_lin_n_beyond")} (rank p {N("mv_lin_p")}), and without '
                         f'{N("mv_lin_loco_max_label")} {N("mv_lin_loco_max_beyond")} {N.verb("mv_lin_loco_max_beyond")}'
                         + (f': the linear result depends on {N("mv_lin_loco_max_label")}.'
                            if N.v('mv_lin_loco_max_beyond') > N.v('mv_lin_n_beyond') else '.'))}


def overview_rows(main: Any, inside_keys: tuple[str, ...] = XR.EXPECTED_TO_DIFFER) -> tuple[list[str], list[str]]:
    """The rows of X1_10, decided by the saved results: (the pre-specified characteristics in which the case lies
    beyond every reference person, the declared "expected to differ" characteristics in which it does not), each in
    the order of ``x1_stats.CHARACTERISTICS``."""
    pre = main[~main['post_hoc'].astype(bool)]
    order = {c.key: i for i, c in enumerate(XS.CHARACTERISTICS)}
    outside = sorted(pre.loc[pre['beyond_all'].astype(bool), 'key'], key=order.__getitem__)
    inside = sorted((k for k in inside_keys if k in set(pre['key']) and k not in outside), key=order.__getitem__)
    return outside, inside


def _x1_10(D: dict[str, Any], N: XR.Nums) -> dict[str, str]:
    """The plain overview of the short report: plain words, no test names (the reader knows no statistics)."""
    case = D['case']
    p = 'null_pre_lfl'
    outside, inside = overview_rows(D['main'])
    m = D['main'].set_index('key')
    fewer = any(m.loc[k, 'n_ref'] < m['n_ref'].max() for k in outside + inside)
    return {'caption': (f'(a) {case} and the {N("n_ref_persons")} reference participants on the pre-specified '
                        f'characteristics in which {case} lies outside the range of all of them (top) and on a declared '
                        f'selection in which a difference might be expected (bottom). Each row is rescaled so that the '
                        f'lowest reference participant sits at 0 and the highest at 1 (log scale for skewed '
                        f'quantities)' + ('; a row that compares fewer reference participants (those who made enough '
                                          'trials) shows their number n' if fewer else '')
                        + f'. (b) For each person, the number of the {N("n_prespecified")} pre-specified '
                        f'characteristics in which that person lies outside the range of all the others, each '
                        f'reference participant judged on as many trials as {case} made.'),
            'decision': (f'Unusual, not unique: {case} lies outside the range of every reference participant compared '
                         f'in {N(f"{p}_case")} of {N("n_prespecified")} pre-specified characteristics, and '
                         f'{N(f"{p}_n_ge")} of the {N("n_ref_persons")} reference participants '
                         f'({N(f"{p}_ge_persons")}) {N.verb(f"{p}_n_ge", "reaches", "reach")} that count too.')}


_TEXTS = {'X1_01_example_traces': _x1_01, 'X1_02_preparation': _x1_02, 'X1_03_spectra': _x1_03,
          'X1_04_movement': _x1_04, 'X1_05_speed_scaling': _x1_05, 'X1_06_repeats': _x1_06,
          'X1_07_hand_signature': _x1_07, 'X1_08_zscore_summary': _x1_08, 'X1_09_multivariate': _x1_09,
          'X1_10_pd01_vs_reference': _x1_10}


def texts(name: str) -> dict[str, str]:
    """{'caption', 'decision'} of figure ``name``."""
    D, N = XR.current_numbers()
    return _TEXTS[name](D, N)
