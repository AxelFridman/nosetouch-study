"""Lane x1: per-trial movement characteristics for the exploratory case study (PD01 against the modelling persons).

Every function here works on ONE trial: its standardised recording (``Stage2Data.stream``: the 60 Hz grid, the
15-Hz zero-phase low-pass and the W3C device frame of ``standardize.stream_from_record``), its cached canonical
approach and fixed windows (gravity frame, ``standardize.canonical``) and its timing columns. So the same numbers are
computed in the same way for the modelling persons (main cache) and the exploratory persons (their own cache, built
by the same ``data.build_cache``). Nothing is fitted across trials and no model is involved. The hand label is used
only by :func:`hand_sign`, to mirror-normalise the chiral yaw so that a positive value is the typical direction for
the hand that was used (a description, not a classifier).

Windows (``PARAMS``; the evidence for each choice is in ``x1_SUMMARY.md``):

* ``pre``: [onset - 1.0 s, onset): the still hold before the movement (arm out, phone held). The app verified
  stillness over [still_ms - 400 ms, go_ms] and the onset is back-dated to the end of the stillness, so this window
  is a postural hold. A hold that breaks the app's still gate (smoothed |w| < 20 deg/s and |a| < 0.5 m/s^2) restarts
  (a false start) instead of entering this window: the gate censors large movements of the hold.
* ``approach``: [onset, end], the oracle approach (the stored online_v1 segment).
* ``nose``: [end, end + 0.6 s), the phone resting at the nose, used only when the return of the arm starts later
  (first 100 ms of |w| > 30 deg/s after the end); otherwise the window would hold the return. The approach itself
  ends only after the phone was still for 250 ms (450 ms for slow trials), and attempts that never became still were
  rejected by the app: the nose window describes only trials that became still.

Spectra (:func:`spectrum`): the three gyroscope axes are linearly detrended, Hann-tapered over the whole window and
zero-padded (``qc.fft_length``, never cropping), and their one-sided power spectral densities are summed, i.e. the
PSD of the angular-velocity VECTOR (invariant to the grip). Band power is the integral of that PSD over the band;
its square root is the band's RMS angular velocity in deg/s. A window of T seconds resolves 1/T Hz (1 Hz for the
hold, 1.67 Hz at the nose) and the Hann main lobe spreads a pure tone over +-2/T Hz, so the 3-7 / 7-12 Hz split of
the short windows is coarse (``x1_compute.resolution_check`` quantifies the leakage).

Movement band: the gyroscope low-passed at 3 Hz (zero-phase Butterworth, order 4, the same ``motionlab`` filter as
the cache) separates the voluntary reach from superimposed faster components; speeds and sub-movements are reported
both as recorded and in the movement band.

The canonical linear acceleration around the first screen touch is read from the WHOLE canonical recording, not
from the cached fixed window (which ends 2.5 s after the onset and would drop late touches): :func:`cached_canonical`
re-runs ``standardize.canonical`` on the cached stream with the session's calibrated gravity magnitude, which the
stream does not carry and which is recovered from the cached fixed window (|f - m u| = cached |a| at every valid
sample; the residual is stored per trial and is at float precision).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar

from motionlab.preprocess import fill_gaps_linear, integrate_gyro, lowpass
from stage2 import config as C
from stage2 import qc as QC

#: Every compute parameter of the lane (part of the staleness hash of the derived files).
PARAMS: dict[str, Any] = {
    'fs_hz': C.FS_HZ,
    'pre_window_s': 1.0,                 # postural hold: [onset - 1.0 s, onset)
    'nose_window_s': 0.6,                # phone at the nose: [end, end + 0.6 s), only when the return starts later
    'verified_hold_ms': 400.0,           # protocol still_hold_ms: [still_ms - 400 ms, go_ms] was verified still
    'bands_hz': {'3_7': (3.0, 7.0), '7_12': (7.0, 12.0)},
    # the decomposition of the band-power ratio: slow sway, the 3-12 Hz band and the top of the analysable band
    'decomp_bands_hz': {'05_3': (0.5, 3.0), '3_12': (3.0, 12.0), '12_15': (12.0, 15.0)},
    'tremor_range_hz': (3.0, 12.0),      # where a spectral peak is looked for
    'nontremor_hz': ((0.5, 3.0), (12.0, 15.0)),   # the rest of the analysable band (the cache low-passes at 15 Hz)
    'nfft_min': 256,
    'movement_lowpass_hz': 3.0,          # voluntary-movement band of the angular speed
    'peak_prominence': 0.10,             # a speed peak / sub-movement must rise 10 % of the trial's peak speed
    'active_dps': C.MIN_PEAK_GYRO_DPS,   # 30 deg/s: the app's onset floor (moving)
    'still_dps': C.STILL_GYRO_DPS,       # 20 deg/s: the app's hold-still threshold (still)
    'hesitation_min_ms': 100.0,          # a hesitation: movement-band |w| < 20 deg/s for >= 100 ms mid-reach
    'return_dps': C.MIN_PEAK_GYRO_DPS,   # the return of the arm starts at the first 100 ms of |w| > 30 deg/s ...
    'return_sustain_ms': 100.0,          # ... after the end of the approach
    'contact_ms': 100.0,                 # the acceleration at the first touch: max |a| within +-100 ms of it
    'touch_peak_ms': 100.0,              # the touch "coincides" with the approach's |a| peak within +-100 ms
    'bp_band_hz': (7.0, 12.0),           # time-domain band-pass (Butterworth, zero-phase) for the touch checks
    'bp_order': 4,
    'pre_touch_gap_ms': 100.0,           # 7-12 Hz before the touch: [onset, touch - 100 ms] ...
    'touch_exclude_ms': (100.0, 200.0),  # ... and without [touch - 100 ms, touch + 200 ms]
    'min_bp_window_ms': 200.0,           # shortest window for a band-pass RMS
    'gmag_bounds': (8.0, 11.5),          # where the calibrated |g| is searched (standardize accepts this range)
    'sparc_fc_hz': 10.0, 'sparc_amp_th': 0.05, 'sparc_padlevel': 4,   # Balasubramanian et al. (2015) defaults
}

NAN = float('nan')

#: Per-trial columns of the spectral windows (every key of spectral_characteristics without its prefix).
SPECTRAL_SUFFIXES = ('rms_3_7', 'rms_7_12', 'rms_05_3', 'rms_3_12', 'rms_12_15', 'ratio_db', 'peak_hz', 'peak_prom_db')


# ───────────────────────────── small helpers ─────────────────────────────


def hand_sign(hand: Any) -> float:
    """+1 for the right hand, -1 for the left: multiplies the chiral yaw so that positive = typical for the hand."""
    return {'right': 1.0, 'left': -1.0}.get(str(hand), NAN)


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def window(stream: Mapping[str, np.ndarray], a_ms: float, b_ms: float) -> dict[str, np.ndarray] | None:
    """Samples with a_ms <= t < b_ms, or None when any of them was not measured (a gap) or the window is too short
    for a spectrum (< 0.5 s)."""
    t = stream['t']
    m = (t >= a_ms) & (t < b_ms)
    n = int(m.sum())
    if n < int(0.5 * PARAMS['fs_hz']):
        return None
    w, f = stream['gyro'][m].astype(float), stream['f'][m].astype(float)
    if not (stream['valid'][m].all() and np.isfinite(w).all() and np.isfinite(f).all()):
        return None
    return {'t': t[m], 'gyro': w, 'f': f}


def spectrum(X: np.ndarray, fs: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, PSD summed over the columns of X) - one-sided density, detrended, Hann over the full window."""
    fs = fs or PARAMS['fs_hz']
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    nfft = QC.fft_length(X.shape[0], PARAMS['nfft_min'])
    f, P = signal.periodogram(X, fs=fs, window='hann', nfft=nfft, detrend='linear', scaling='density', axis=0)
    return f, P.sum(axis=1)


def band_power(f: np.ndarray, P: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(P[m].sum() * (f[1] - f[0])) if m.any() else NAN


def tremor_peak(f: np.ndarray, P: np.ndarray) -> tuple[float, float]:
    """(frequency, prominence in dB) of the highest LOCAL maximum of the PSD inside 3-12 Hz. The prominence is how
    far the peak rises above the higher of the two lowest points that separate it from higher ground (scipy
    ``peak_prominences`` on 10 log10 PSD over 2-13 Hz): a tremor gives a sharp, prominent peak, a broadband rise a
    low one. Without a local maximum (the spectrum only falls there): (NaN, 0)."""
    lo, hi = PARAMS['tremor_range_hz']
    m = (f >= lo - 1.0) & (f <= hi + 1.0)
    ff, pp = f[m], P[m]
    idx, _ = signal.find_peaks(pp)
    idx = idx[(ff[idx] >= lo) & (ff[idx] <= hi)]
    if not idx.size or not np.all(pp > 0):
        return NAN, 0.0
    j = int(idx[np.argmax(pp[idx])])
    prom = signal.peak_prominences(10.0 * np.log10(pp), [j])[0][0]
    return float(ff[j]), float(prom)


def tremor_peak_hz(f: np.ndarray, P: np.ndarray) -> float:
    """Frequency of the highest local maximum of the PSD inside 3-12 Hz (NaN without one), see :func:`tremor_peak`."""
    return tremor_peak(f, P)[0]


def spectral_characteristics(W: np.ndarray, prefix: str) -> dict[str, float]:
    """Band RMS (deg/s) in 3-7 and 7-12 Hz and in the decomposition bands 0.5-3, 3-12 and 12-15 Hz, the frequency of
    the highest spectral peak in 3-12 Hz and the 3-12 Hz power relative to 0.5-3 + 12-15 Hz (dB) of the
    angular-velocity vector W (n, 3)."""
    f, P = spectrum(W)
    out: dict[str, float] = {}
    for name, (lo, hi) in {**PARAMS['bands_hz'], **PARAMS['decomp_bands_hz']}.items():
        out[f'{prefix}_rms_{name}'] = math.sqrt(band_power(f, P, lo, hi))
    lo, hi = PARAMS['tremor_range_hz']
    trem = band_power(f, P, lo, hi)
    non = sum(band_power(f, P, a, b) for a, b in PARAMS['nontremor_hz'])
    out[f'{prefix}_ratio_db'] = 10.0 * math.log10(trem / non) if trem > 0 and non > 0 else NAN
    out[f'{prefix}_peak_hz'], out[f'{prefix}_peak_prom_db'] = tremor_peak(f, P)
    return out


def _nan_spectral(prefix: str) -> dict[str, float]:
    return {f'{prefix}_{s}': NAN for s in SPECTRAL_SUFFIXES}


def sway(X: np.ndarray) -> float:
    """RMS of the deviation of a vector signal (n, 3) from its window mean: total sway."""
    X = np.asarray(X, dtype=float)
    return float(np.sqrt(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))))


def bandpass_rms(t: np.ndarray, W: np.ndarray, keep: np.ndarray, fs: float | None = None) -> float:
    """RMS of the angular-velocity vector band-passed at ``bp_band_hz`` (zero-phase Butterworth over the whole
    segment ``W``, so the window edges carry no filter transient) over the samples ``keep``; NaN when fewer than
    ``min_bp_window_ms`` of samples are kept."""
    fs = fs or PARAMS['fs_hz']
    if keep.sum() < PARAMS['min_bp_window_ms'] / 1000.0 * fs or W.shape[0] < 30:
        return NAN
    sos = signal.butter(PARAMS['bp_order'], PARAMS['bp_band_hz'], btype='bandpass', fs=fs, output='sos')
    Y = signal.sosfiltfilt(sos, W, axis=0)
    return float(np.sqrt(np.mean(np.sum(Y[keep] ** 2, axis=1))))


# ───────────────────────────── smoothness (speed profiles) ─────────────────────────────


def ldlj(v: np.ndarray, fs: float | None = None) -> float:
    """Log dimensionless jerk of a speed profile (Hogan & Sternad 2009; Balasubramanian et al. 2015, LDLJ-V):
    -ln( T^3 / v_peak^2 * integral (d2v/dt2)^2 dt ). Higher (less negative) = smoother."""
    fs = fs or PARAMS['fs_hz']
    v = np.asarray(v, dtype=float)
    if v.size < 5 or not np.isfinite(v).all():
        return NAN
    dt = 1.0 / fs
    vpk = float(np.max(np.abs(v)))
    if vpk <= 0:
        return NAN
    jerk = np.diff(v, 2) / dt ** 2
    dlj = (v.size * dt) ** 3 / vpk ** 2 * float(np.sum(jerk ** 2)) * dt
    return -math.log(dlj) if dlj > 0 else NAN


def sparc(v: np.ndarray, fs: float | None = None) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015, reference implementation): the arc length
    of the normalised magnitude spectrum up to an adaptive cut-off (<= 10 Hz, amplitude threshold 0.05). Closer to
    zero = smoother; a movement made of several sub-movements or carrying faster components has a longer arc (more
    negative)."""
    fs = fs or PARAMS['fs_hz']
    v = np.asarray(v, dtype=float)
    if v.size < 5 or not np.isfinite(v).all():
        return NAN
    nfft = int(2 ** (math.ceil(math.log2(v.size)) + PARAMS['sparc_padlevel']))
    f = np.arange(nfft) * (fs / nfft)
    M = np.abs(np.fft.fft(v, nfft))
    if M.max() <= 0:
        return NAN
    M = M / M.max()
    sel = f <= PARAMS['sparc_fc_hz']
    fs_, Ms = f[sel], M[sel]
    inx = np.flatnonzero(Ms >= PARAMS['sparc_amp_th'])
    fs_, Ms = fs_[inx[0]:inx[-1] + 1], Ms[inx[0]:inx[-1] + 1]
    if fs_.size < 2:
        return NAN
    return float(-np.sum(np.sqrt((np.diff(fs_) / (fs_[-1] - fs_[0])) ** 2 + np.diff(Ms) ** 2)))


def n_peaks(v: np.ndarray) -> float:
    """Local maxima of a speed profile that rise at least ``peak_prominence`` of its peak (speed peaks)."""
    v = np.asarray(v, dtype=float)
    if v.size < 3 or not np.isfinite(v).all() or v.max() <= 0:
        return NAN
    idx, _ = signal.find_peaks(v, prominence=PARAMS['peak_prominence'] * float(v.max()))
    return float(max(idx.size, 1))


def hesitations(v: np.ndarray, fs: float | None = None) -> float:
    """Mid-reach hesitations: runs of >= 100 ms with |w| below the app's still threshold (20 deg/s) between the first
    and the last sample above its onset floor (30 deg/s). The online segmenter ends the approach after 250 ms of
    stillness (450 ms for slow trials), so longer pauses cannot lie inside an approach by construction."""
    fs = fs or PARAMS['fs_hz']
    v = np.asarray(v, dtype=float)
    act = np.flatnonzero(v >= PARAMS['active_dps'])
    if act.size < 2:
        return 0.0
    core = v[act[0]:act[-1] + 1] < PARAMS['still_dps']
    k = int(round(PARAMS['hesitation_min_ms'] / 1000.0 * fs))
    n, run = 0, 0
    for s in core:
        run = run + 1 if s else 0
        if run == k:
            n += 1
    return float(n)


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix (deg)."""
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


# ───────────────────────────── the canonical recording of a cached trial ─────────────────────────────


def cached_canonical(stream: Mapping[str, np.ndarray], row: Mapping[str, Any], fixed: np.ndarray | None,
                     fixed_mask: np.ndarray | None, channels: tuple[str, ...]) -> tuple[Any, float, float]:
    """(Canonical over the whole recording, gravity magnitude used, max |deviation| from the cached fixed window).

    ``standardize.canonical`` is re-run on the cached stream with the cache's oracle bounds. The stream does not
    carry the session's calibrated |g|; it is recovered as the magnitude m whose |f - m u| (u: the canonical gravity
    direction, which does not depend on m) reproduces the cached fixed-window |a| at every valid sample. Without a
    valid fixed window the rest |g| of ``gyro_gravity`` is used and the deviation is NaN."""
    from stage2.standardize import Stream, canonical

    def make(mag: float | None) -> Any:
        st = Stream(trial_id=None, t=stream['t'].astype(float), f=stream['f'].astype(float),
                    acc=stream['acc'].astype(float), gyro=stream['gyro'].astype(float),
                    valid=stream['valid'].astype(bool), fs_hz=C.FS_HZ, gravity_magnitude=mag)
        return canonical(st, _num(row.get('onset_ms')), _num(row.get('end_ms')))

    can = make(None)
    mag0 = float(np.linalg.norm(can.g[can.i_on]))
    if fixed is None or fixed_mask is None or not np.asarray(fixed_mask, dtype=bool).any():
        return can, mag0, NAN
    on = _num(row.get('onset_ms'))
    grid = on - C.FIXED_PRE_S * 1000.0 + np.arange(fixed.shape[-1]) * (1000.0 / PARAMS['fs_hz'])
    msk = np.asarray(fixed_mask, dtype=bool)
    target = np.asarray(fixed[channels.index('acc_mag')], dtype=float)[msk]
    u = can.g / np.linalg.norm(can.g, axis=1, keepdims=True)
    f = fill_gaps_linear(can.t, stream['f'].astype(float))

    def loss(m: float) -> float:
        a = np.linalg.norm(f - m * u, axis=1)
        return float(np.sum((np.interp(grid[msk], can.t, a) - target) ** 2))

    lo, hi = PARAMS['gmag_bounds']
    mag = float(minimize_scalar(loss, bounds=(lo, hi), method='bounded', options={'xatol': 1e-7}).x)
    can = make(mag)
    dev = float(np.max(np.abs(np.interp(grid[msk], can.t, can.channel('acc_mag')) - target)))
    return can, mag, dev


# ───────────────────────────── one trial ─────────────────────────────


def return_start_ms(stream: Mapping[str, np.ndarray], end_ms: float) -> tuple[float, bool]:
    """(time from the end of the approach to the start of the arm's return in s, censored). The return starts at
    the first ``return_sustain_ms`` of |w| > ``return_dps`` after the end; when none is recorded the value is the
    time to the end of the recording and ``censored`` is True (the phone stayed at the nose until the recording
    stopped)."""
    t = stream['t']
    m = t > end_ms
    if not m.any():
        return 0.0, True
    tt = t[m]
    v = np.linalg.norm(np.nan_to_num(stream['gyro'][m].astype(float)), axis=1)
    k = max(1, int(round(PARAMS['return_sustain_ms'] / 1000.0 * PARAMS['fs_hz'])))
    above = (v > PARAMS['return_dps']).astype(int)
    if above.size >= k:
        run = np.convolve(above, np.ones(k, dtype=int), 'valid') >= k
        if run.any():
            return float(tt[int(np.argmax(run))] - end_ms) / 1000.0, False
    return float(tt[-1] - end_ms) / 1000.0, True


def touch_characteristics(row: Mapping[str, Any], can: Any, ts: np.ndarray, W: np.ndarray, vmv: np.ndarray,
                          app: np.ndarray) -> dict[str, float]:
    """The first screen touch (added after PD01's traces were seen, so post hoc) and the movement around it.

    * ``touch_s``: onset to the first screen touch (``first_touch_ms``, the first touchstart after GO) when it falls
      inside the approach or the hold before the end cue (the app cannot tell the nose from a finger or the grip);
      NaN without such a touch; ``touch_share``: that time as a share of the approach window;
    * ``touch_acc``: the largest canonical linear acceleration |a| within +-``contact_ms`` of the touch, never after
      the end cue (the arm may return from there);
    * ``acc_peak_s``: onset to the largest |a| of the approach; ``touch_at_acc_peak``: 1 when the touch lies within
      +-``touch_peak_ms`` of it;
    * ``rot_after_touch``: share of the approach's movement-band rotation (integral of the 3-Hz angular speed) that
      comes after the touch (a touch at the arrival leaves little rotation after it);
    * ``app_bp_7_12_pre_touch``: 7-12 Hz band-passed angular-velocity RMS from the onset to 100 ms before the touch;
      ``app_bp_7_12_excl_touch``: over the approach without [touch - 100 ms, touch + 200 ms] (both NaN without a
      touch or with less than 200 ms of samples); ``app_bp_7_12``: over the whole approach (every trial).
    """
    on, en = _num(row.get('onset_ms')), _num(row.get('end_ms'))
    cue, ft = _num(row.get('end_detected_ms')), _num(row.get('first_touch_ms'))
    lim = cue if math.isfinite(cue) else en
    out = {k: NAN for k in ('touch_s', 'touch_share', 'touch_acc', 'touch_at_acc_peak', 'rot_after_touch',
                            'app_bp_7_12_pre_touch', 'app_bp_7_12_excl_touch')}
    amag = can.channel('acc_mag')
    seg = (can.t >= on) & (can.t <= en)
    out['acc_peak_s'] = float((can.t[seg][int(np.argmax(amag[seg]))] - on) / 1000.0) if seg.any() else NAN
    out['app_bp_7_12'] = bandpass_rms(ts, W, app)
    if not (math.isfinite(ft) and on < ft <= lim):
        return out
    out['touch_s'] = (ft - on) / 1000.0
    out['touch_share'] = (ft - on) / (en - on)
    near = (np.abs(can.t - ft) <= PARAMS['contact_ms']) & (can.t <= lim) & can.valid
    out['touch_acc'] = float(np.max(amag[near])) if near.any() else NAN
    if math.isfinite(out['acc_peak_s']):
        out['touch_at_acc_peak'] = float(abs(out['acc_peak_s'] - out['touch_s']) * 1000.0 <= PARAMS['touch_peak_ms'])
    ta = ts[app]
    if ta.size > 1:
        total = float(np.trapezoid(vmv[app], ta / 1000.0))
        after = ta >= ft
        part = float(np.trapezoid(vmv[app][after], ta[after] / 1000.0)) if after.sum() > 1 else 0.0
        out['rot_after_touch'] = part / total if total > 0 else NAN
    gap = PARAMS['pre_touch_gap_ms']
    out['app_bp_7_12_pre_touch'] = bandpass_rms(ts, W, app & (ts <= ft - gap))
    a, b = PARAMS['touch_exclude_ms']
    out['app_bp_7_12_excl_touch'] = bandpass_rms(ts, W, app & ~((ts >= ft - a) & (ts <= ft + b)))
    return out


def trial_characteristics(stream: Mapping[str, np.ndarray], row: Mapping[str, Any], approach: np.ndarray,
                          channels: tuple[str, ...], fixed: np.ndarray | None = None,
                          fixed_mask: np.ndarray | None = None) -> dict[str, float]:
    """Every per-trial characteristic of one trial (see ``x1_stats.CHARACTERISTICS`` for labels and rationale).

    stream: ``Stage2Data.stream(i)``; row: its trial-table row (timing, QC, hand); approach: its cached (C, 64)
    canonical approach window; channels: the channel names of the arrays; fixed, fixed_mask: its cached fixed window
    (C, 180) and mask (to recover the calibrated |g|, see :func:`cached_canonical`)."""
    fs = PARAMS['fs_hz']
    on, en = _num(row.get('onset_ms')), _num(row.get('end_ms'))
    go, still = _num(row.get('go_ms')), _num(row.get('still_ms'))
    sgn = hand_sign(row.get('hand'))
    out: dict[str, float] = {}

    # ── preparation: the postural hold before the movement, reaction time, false starts ──
    out['rt_s'] = (on - go) / 1000.0
    out['false_starts'] = _num(row.get('false_starts'))
    out['hold_wait_s'] = still / 1000.0
    pre_ms = PARAMS['pre_window_s'] * 1000.0
    out['pre_in_verified_hold'] = float(on - pre_ms >= still - PARAMS['verified_hold_ms'])
    w = window(stream, on - pre_ms, on)
    if w is not None:
        out.update(spectral_characteristics(w['gyro'], 'pre'))
        out['pre_sway_gyro'] = sway(w['gyro'])
        out['pre_sway_acc'] = sway(w['f'])
    else:
        out.update(_nan_spectral('pre') | {'pre_sway_gyro': NAN, 'pre_sway_acc': NAN})

    # ── the approach: speed, amplitude, smoothness, band power ──
    t = stream['t']
    seg = (t >= on - 500.0) & (t <= en + 500.0)          # filter with margins, then cut the approach
    ts = t[seg]
    W = fill_gaps_linear(ts, stream['gyro'][seg].astype(float))
    Wmv = lowpass(W, fs, PARAMS['movement_lowpass_hz'], order=C.LOWPASS_ORDER)
    app = (ts >= on) & (ts <= en)
    ta, Wa, Wmva = ts[app], W[app], Wmv[app]
    v = np.linalg.norm(Wa, axis=1)                        # angular speed as recorded (15-Hz band)
    vmv_all = np.linalg.norm(Wmv, axis=1)                 # movement band (< 3 Hz), with the margins
    vmv = vmv_all[app]
    out['duration_s'] = (en - on) / 1000.0
    out['peak_speed'] = _num(row.get('peak_gyro_dps'))    # the cache's own QC value (canonical |w|, 60 Hz)
    out['peak_speed_mv'] = float(vmv.max()) if vmv.size else NAN
    out['mean_speed_mv'] = float(np.trapezoid(vmv, ta / 1000.0) / ((ta[-1] - ta[0]) / 1000.0)) if ta.size > 1 else NAN
    out['peak_acc'] = _num(row.get('peak_acc_ms2'))       # the cache's own QC value (canonical |a|, 60 Hz)
    out['time_to_peak'] = float((ta[int(np.argmax(vmv))] - on) / (en - on)) if vmv.size else NAN
    act = np.flatnonzero(v >= PARAMS['active_dps'])
    last = float(ta[act[-1]]) if act.size else NAN
    out['active_s'] = (last - on) / 1000.0                # onset to the last sample faster than 30 deg/s
    out['settle_s'] = (en - last) / 1000.0                # the end detector's wait for stillness
    # amplitude: the angle turned (as recorded and in the movement band) and the net orientation change; a path
    # ratio (total / net) was tried and dropped: the net change is often near 0 (median 12 deg), so it explodes
    out['rotation_total'] = float(np.trapezoid(v, ta / 1000.0)) if ta.size > 1 else NAN
    out['rotation_total_mv'] = float(np.trapezoid(vmv, ta / 1000.0)) if ta.size > 1 else NAN
    R, _ = integrate_gyro(ta, Wa)
    out['rotation_net'] = rotation_angle_deg(R[-1]) if ta.size > 1 else NAN
    k_tilt = channels.index('tilt_from_start')
    out['tilt_change'] = float(approach[k_tilt, -1])
    out['net_yaw_abs'] = abs(_num(row.get('net_yaw_deg')))
    out['ldlj'] = ldlj(v, fs)
    out['sparc'] = sparc(v, fs)
    out['speed_peaks'] = n_peaks(v)
    out['submovements'] = n_peaks(vmv)
    out['hesitations'] = hesitations(vmv, fs)
    out.update(spectral_characteristics(Wa, 'app') if Wa.shape[0] >= int(0.5 * fs) else _nan_spectral('app'))
    pk = out['peak_speed_mv']
    out['app_7_12_per_speed'] = out['app_rms_7_12'] / pk if math.isfinite(pk) and pk > 0 else NAN

    # ── the first screen touch, on the whole canonical recording (calibrated |g| recovered) ──
    can, gmag, gdev = cached_canonical(stream, row, fixed, fixed_mask, channels)
    out['g_mag'] = gmag
    out['g_mag_dev'] = gdev
    out.update(touch_characteristics(row, can, ts, W, vmv_all, app))

    # ── the phone at the nose ──
    dwell, cens = return_start_ms(stream, en)
    out['nose_dwell_s'] = dwell
    out['nose_no_return'] = float(cens)
    nose_ms = PARAMS['nose_window_s'] * 1000.0
    w = window(stream, en, en + nose_ms) if dwell * 1000.0 >= nose_ms else None
    if w is not None:
        out.update(spectral_characteristics(w['gyro'], 'nose'))
        out['nose_sway_gyro'] = sway(w['gyro'])
    else:
        out.update(_nan_spectral('nose') | {'nose_sway_gyro': NAN})

    # ── the hand signature, mirror-normalised (positive = the typical direction for the hand used) ──
    out['fhy_hand'] = sgn * _num(row.get('first_half_yaw_deg'))
    out['net_yaw_hand'] = sgn * _num(row.get('net_yaw_deg'))
    # post hoc: the largest hand-consistent turn reached in the first half (the first-half yaw reads only its value
    # at half of the approach, after which a reach can turn back), and when it is reached
    ycum = sgn * approach[channels.index('yaw_cum')].astype(float)
    half = ycum.size // 2 + 1
    if np.isfinite(ycum).all():
        j = int(np.argmax(ycum[:half]))
        out['yaw_peak_hand'] = float(ycum[j])
        out['yaw_peak_s'] = float(j / (ycum.size - 1) * (en - on) / 1000.0)
    else:
        out['yaw_peak_hand'] = out['yaw_peak_s'] = NAN
    out['sign_typical'] = float(out['fhy_hand'] > 0) if math.isfinite(out['fhy_hand']) else NAN
    out['net_sign_typical'] = float(out['net_yaw_hand'] > 0) if math.isfinite(out['net_yaw_hand']) else NAN
    return out


def trial_spectra(stream: Mapping[str, np.ndarray], row: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """The gyroscope PSD of each window of one trial (for the spectra figure): {'pre', 'approach', 'nose'} ->
    (f, P) or None."""
    on, en = _num(row.get('onset_ms')), _num(row.get('end_ms'))
    pre_ms = PARAMS['pre_window_s'] * 1000.0
    nose_ms = PARAMS['nose_window_s'] * 1000.0
    out: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    w = window(stream, on - pre_ms, on)
    out['pre'] = spectrum(w['gyro']) if w is not None else None
    t = stream['t']
    m = (t >= on) & (t <= en)
    Wa = stream['gyro'][m].astype(float)
    out['approach'] = spectrum(Wa) if (m.sum() >= int(0.5 * PARAMS['fs_hz']) and np.isfinite(Wa).all()) else None
    dwell, _ = return_start_ms(stream, en)
    w = window(stream, en, en + nose_ms) if dwell * 1000.0 >= nose_ms else None
    out['nose'] = spectrum(w['gyro']) if w is not None else None
    return out


def sine_leakage(freq_hz: float, window_s: float, rms: float = 10.0 / math.sqrt(2.0), n_phase: int = 8
                 ) -> dict[str, float]:
    """The band RMS that :func:`spectral_characteristics` reports for a pure tone of ``freq_hz`` (RMS ``rms``) in a
    window of ``window_s``, averaged over ``n_phase`` phases: how much of a tone leaks into the neighbouring band."""
    n = int(round(window_s * PARAMS['fs_hz']))
    tt = np.arange(n) / PARAMS['fs_hz']
    acc: dict[str, list[float]] = {}
    for ph in np.linspace(0.0, 2.0 * math.pi, n_phase, endpoint=False):
        W = np.zeros((n, 3))
        W[:, 0] = rms * math.sqrt(2.0) * np.sin(2.0 * math.pi * freq_hz * tt + ph)
        for k, v in spectral_characteristics(W, 'x').items():
            acc.setdefault(k[2:], []).append(v)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}
