# Stage 3: porting spec of the Python production path

This is a read-only analysis of commit `8eec3d4` (branch `v2-street`, 2026-10-28). No code was changed.

- **Ground truth.** The Python code is the ground truth. Every rule cites `file:line`. Paths are relative to `analysis/` unless they start with `app/` or `docs/`.
- **Companions.** [`APP_SPEC.md`](APP_SPEC.md) covers the page: the live loop, the UI and the tests. [`app/src/predict/types.ts`](../../app/src/predict/types.ts) holds the contracts. The names used here follow both. §8 lists the changes to `types.ts` that this analysis calls for.
- **Notation.** `t[i]` is the 60 Hz grid. `n` is the number of grid samples. "deg/s" is the unit of angular velocity. `step` = `1000/60` = `16.666666666666668` ms.

---

## 0. Ten findings that shape the port

1. **The stream filter is not causal, so parity has two levels.**
   - `predict_stream` is causal for a fixed stream (`stage2/lanes/m4_online.py:3-8`). The stream itself is not: it is low-passed with zero phase over the whole buffer (`stage2/standardize.py:16-17,201-202`; `motionlab/preprocess.py:122-145`).
   - So a sample's filtered value keeps changing while later samples arrive. Filtering a prefix instead of the full recording changes values, relative to the signal amplitude, by:

     | Distance from the prefix end | Change |
     |---|---|
     | 0 samples | 3e-3 |
     | 10 samples | 2e-5 |
     | 30 samples | 5e-9 |
     | 60 samples | 3e-14 |
     | 90 samples or more | 0 |

     This was measured on a synthetic signal (§5.2). The filter's largest pole has |p| = 0.668.
   - Lane m4 replayed every candidate on the buffer received up to its decision time (`stage2/lanes/m4_pipeline.py:1300-1367`). On 598 recordings the largest |dP| was 2.0e-5, no answer changed and 0.3 % of statuses changed (`reports/stage2/tables/m4/m4_replay.json`).
   - APP_SPEC §4.2 freezes a bout only 1 s (`settleMs`) after its `decided_ms`. At that point its inputs equal the full recording's to about 1e-13. **This spec adopts that rule** (§2.15). Then one Python reference serves both parity tests:
     - **offline**: TS `predictStream(full stream)` against `predict_stream`;
     - **live replay**: the TS engine fed raw events against `reference_predict(mode='live')`.
2. **The windows are rounded to float32 in two places.**
   - `time_normalised` returns float32 (`stage2/standardize.py:301`).
   - The final model's body-frame channels are computed in float64 from the float32 window, then cast to float32 (`stage2/lanes/final_infer.py:59-60,66`).
   - TS must apply `Math.fround` at exactly these two points. Every other computation is float64.
3. **`valid` and `acc` are carried but never read.** `predict_stream` builds `valid` (`m4_production.py:93`) and passes `acc`. No function on the production path reads either one: `fixed_window` and `window_valid` are not called.
   - The port may drop `acc`.
   - It must still produce NaN inside long gaps, because the NaN pattern drives the gap filling.
4. **There are two hysteresis passes over one activity signal.**
   - The detector uses the fitted configuration (`SEG_GRID`: lo 15 or 20 deg/s, `end_sustain` 250 or 400 ms; `m4_e2e.py:52-56`). It is chosen inside the fit. The fold that held P01 out chose `lo15_es250` (`tables/m4/m4_demo.json`).
   - The grip check always uses `m4_detect.DEFAULT` (lo 20, `end_sustain` 250; `m4_production.py:104`, `m4_detect.py:424-444`).
5. **The m4 production model has never been exported.** `fit_production()` (`m4_production.py:122-142`) must be run on all persons and written to JSON (§4.3). Its calibrator is the nested choice. The P01-out fold picked `isotonic_sym`, which needs an `interp1d`-exact port, not `np.interp`. Its aggregation is `scale_sessions`, with s_k = s₁·kᵇ.
6. **The CNN, PHYS6 and the yaw reference were evaluated on oracle windows only.** They read the same (17, 64) window that `window_for` produces, with no adapter. But none of them was evaluated end to end on detector windows. Only lane m4's own classifier was (§3.4).
7. **The yaw reference is m4's hand model without its calibrator.** `example_yaw_logreg` (`stage2/scripts/reference.py:64-85`) and m4's `Yaw8` (`m4_models.py:57-77`) are the same mirror-augmented logistic regression on `yaw_cum[0::8]`, C = 1, fitted on the same capped rows. Only the iteration cap differs (1000 and 2000). The reference is shown uncalibrated and has no fitted k-aggregation (§3.3).
8. **PHYS6 needs the segment duration and fails on a horizontal screen.** Its integrals use `dt = duration/63` (`m1_features.py:281`). Its body frame is NaN when |e_z − (e_z·u)u| ≤ 1e-6, and `compute_features` then raises (`m1_features.py:240,359-361`). The page must treat that attempt as "unavailable" for PHYS6.
9. **`percent_right` uses Python's rounding.** `int(round(100*p))` rounds half to even (`m4_production.py:117`). JS `Math.round` rounds half up, so TS needs a half-even rounder for parity. The value 0.5 maps to 50, and 0.125 maps to 12, not 13.
10. **Two stopping-rule details matter.**
    - `p_answer` = 0.5 means the page always answers after `k_max` attempts, even at P = 0.5 exactly (`m4_online.py:64`).
    - `stream_end` exists only when a recording ends. Live, the page keeps watching. APP_SPEC caps a stream at 180 s and then reports `stream_end`.

---

## 1. Raw events to the standardised stream (`stage2.standardize.stream_from_record`)

Python path: `stream_from_record(record, norm)` (`standardize.py:155-207`). `predict_stream` then reads the stream as a dict and rebuilds a `Stream` with `m4_detect.as_stream` (`m4_detect.py:233-239`). That step drops `gravity_magnitude` and `baseline_gravity`: production has neither.

### 1.1 Input fields and units

One DeviceMotion event gives one row. The collector's handler is `app/src/core/sensors.ts:448-461`, and the recorder rounding is `app/src/core/recorder.ts:288-313`.

| Field | Source | Unit | Rounding (`ROUNDING`, `app/src/core/constants.ts:128-134`) | Missing |
|---|---|---|---|---|
| `t` | `performance.now()` at the top of the handler, minus t0 | ms | 2 decimals, half away from zero (`recorder.ts:88-96`) | event dropped |
| `acc_x/y/z` | `event.acceleration` | m/s², RAW sign | 4 decimals | whole vector NaN when any component is missing (`sensors.ts:28-30`) |
| `accg_x/y/z` | `event.accelerationIncludingGravity` | m/s², RAW sign | 4 decimals | as above |
| `rr_alpha/beta/gamma` | `event.rotationRate` | deg/s, schema order | 3 decimals | as above |

- `motionlab.io.streams_from_record` reads `t`, the nine value columns and `ts` (`motionlab/io.py:195-210,260`). A null value becomes NaN (`io.py:88-97`).
- `ts` must have the same length as `t` (`io.py:202-206`), but it is never used. The Python reference fills it with nulls.
- **Live buffer.** Round exactly like the recorder, so that a live session replays bit for bit as a record (APP_SPEC §1.3).

### 1.2 Sign conventions and axes

- **Gravity sign.** `acc` and `accg` are multiplied by the session sign (`conventions.py:740-744`, called from `device_signals` at `:893-906`).
  - Engine prior (`ENGINE_GRAVITY_SIGN`, `conventions.py:476`): `webkit` −1; `blink` and `gecko` +1.
  - Engine detection: `webkit` if and only if `'webkitCompassHeading' in DeviceOrientationEvent.prototype` (`sensors.ts:990-995`).
  - In Stage 2 the sign of every included session was the engine prior, confirmed by the trial votes (source `engine_vote`, `conventions.py:710-715`; `stage2/README.md` §3).
  - Production has no votes and no calibration, so it uses the engine prior alone. `unknown` maps to +1, the default at `conventions.py:736-737`; flag it in the UI.
  - **A wrong sign is a parity transformation.** Polar vectors flip and ω does not (`conventions.py:40-45`). A correct screen-towards-face hold then gives `rest_up_z < 0`, so the grip gate refuses instead of answering the wrong hand (§2.12).
- **Rotation rate.** `[alpha, beta, gamma]` is taken as device `[x, y, z]` in deg/s (`RR_MAPPINGS['w3c']`, `conventions.py:311-332`). Every session was W3C deg/s (README §3). Production assumes it.
- **Not on the production path:**
  - gyro bias (norm_v2 subtracts none: `standardize.py:20-21`);
  - `quantize_ios` (`config.py:119` = False; `standardize.py:172-176`);
  - mirror and screen flip (`standardize.py:183-190`; both False).
- **After this step:**
  - `f` = signed `accg`: specific force, W3C sign, pointing up at rest;
  - `acc` = signed browser linear acceleration;
  - `gyro` = ω in deg/s;

  all in the W3C device frame: x to the right edge, y to the top edge, z out of the screen (`conventions.py:14-15`).

### 1.3 Isolated repeated samples (`motionlab/preprocess.py:575-595`)

This runs on the RAW arrays (`standardize.py:191-196`). The sign does not matter here because it tests equality.

```
A[i]     = the 9 raw values of row i (acc, accg, rr)
same[0]  = false
same[i]  = (every column: A[i] == A[i-1], with NaN equal to NaN) AND (any value of A[i] is finite)
isolated[i] = same[i] AND NOT same[i+1] AND NOT same[i-1]    (same[n] = same[-1] = false)
drop[i]  = isolated[i]                                   if engine == 'webkit'
         = isolated[i] AND |rr_raw[i]| > 20 deg/s        otherwise (norm of the raw [alpha, beta, gamma])
```

- If fewer than 5 rows remain, Python raises `ValueError` (`standardize.py:197-198`). TS must report "not enough data".
- **Live note.** Whether row i is isolated depends on row i+1. So the newest one or two rows can change status, which the tail recomputation of §1.9 covers.

### 1.4 Time cleaning and the uniform grid (`preprocess.py:40-64`)

`_prepare_times`:

1. Keep the rows with finite `t`.
2. Sort them stably by `t`.
3. Drop a row whose `t` equals the previous one, keeping the first.

The browser clock is monotonic. The live buffer therefore rejects a non-increasing `t` (APP_SPEC §1.3), which is the same thing.

`uniform_grid(t0 = first kept t, t1 = last kept t, 60 Hz)`:

```
step   = 1000.0 / 60.0                                   (= 16.666666666666668)
count  = floor((t1 - t0) / step + 1e-9) + 1
t[i]   = t0 + i * step          (numpy: t_start + arange(n) * step; compute exactly this expression)
```

### 1.5 Resampling and gaps (`preprocess.py:67-108`; `MAX_GAP_MS` = 100, `config.py:113`)

Each of the 9 columns (`accg`, `acc`, `gyro`, after signing) is resampled on its own, from its own finite samples `(tv, xv)`. For each grid time `tn`:

```
r      = searchsorted(tv, tn, side='left')          # first tv >= tn
exact  = r < len(tv) AND tv[r] == tn
inside = 1 <= r < len(tv)
gap    = tv[r] - tv[r-1]  (inside only)
value  = np.interp(tn, tv, xv)  if exact OR (inside AND gap <= 100)  else NaN
         (a column with a single finite sample: value only where exact)
```

`np.interp` here is numpy's algorithm (§5.1). A grid point after the column's last finite sample is NaN.

### 1.6 `valid` (`standardize.py:200`)

`valid[i]` = all of `f[i]` finite AND all of `gyro[i]` finite, evaluated before the low-pass. It is carried along but never read (finding 3).

### 1.7 The low-pass (`preprocess.py:122-145`; `config.py:110-112`)

It is a zero-phase Butterworth of order 4 at 15 Hz, `fs` = 60. It runs column by column, separately over each run of finite samples (`finite_runs`, `preprocess.py:111-119`), so NaN stays NaN and never smears.

```
sos (scipy.signal.butter(4, 15, 'low', fs=60, output='sos')), rows [b0, b1, b2, a0, a1, a2]:
  [0.09398085143379445, 0.1879617028675889, 0.09398085143379445, 1.0, -1.387778780781446e-16, 0.03956612989658006]
  [1.0, 2.0, 1.0, 1.0, -1.1102230246251565e-16, 0.4464626921716894]
zi (scipy.signal.sosfilt_zi(sos)), rows [z0, z1]:
  [0.26763482160912794, 0.07967311874153896]
  [0.6383843269570777, -0.0848470191287671]
default padlen = 3 * (2*n_sections + 1 - min(#(b2 == 0), #(a2 == 0))) = 15      (preprocess.py:136)
```

The two values of `a1` near 1e-16 are not zero: export them as they are. The exporter recomputes every one of these numbers with scipy and never types them. A TS test checks the bundled values.

For one finite run `x[0..n-1]`:

```
if n < 3: copy unfiltered                                             (preprocess.py:141-142)
p    = min(15, n - 1)                                                  (preprocess.py:143)
ext  = [2x0 - x[p], 2x0 - x[p-1], ..., 2x0 - x[1],  x[0..n-1],
        2x[n-1] - x[n-2], ..., 2x[n-1] - x[n-1-p]]                     (scipy odd_ext; length n + 2p)
y    = sosfilt(sos, ext, zi = zi * ext[0])                             (every section's state scaled by ext[0])
r    = reverse(y)
y2   = sosfilt(sos, r,   zi = zi * r[0])
out  = reverse(y2)[p : p + n]
```

`sosfilt` is the direct-form-II-transposed recursion that scipy uses (`_sosfilt.pyx`). Keep this exact evaluation order:

```
for each sample x:  for each section s (b0,b1,b2,1,a1,a2; state z0,z1):
    y  = b0*x + z0
    z0 = (b1*x - a1*y) + z1
    z1 = b2*x - a2*y
    x  = y
```

The cutoff is `fs`/4, a half-band design. Its poles have |p| = 0.668 and 0.199, so the backward pass forgets the end of the buffer in about 60 to 90 samples (§5.2).

### 1.8 Output

`Stream{t, f = U[:,0:3], acc = U[:,3:6], gyro = U[:,6:9], valid}` (`standardize.py:203-207`). As `predict_stream` sees it: t in ms, f in m/s², gyro in deg/s, float64 throughout.

- The Stage 2 cache stores `f`, `acc` and `gyro` as float32 (`stage2/data.py:328-329`). Lane m4 trained its detector on those float32 streams.
- Parity fixtures must start from records, through `stream_from_record` in float64, never from the cache.

### 1.9 Computing it live

Live, the page computes `stream_from_record(prefix)` on the raw buffer received so far. Everything except the backward filter pass is incremental:

- **Repeats and resampling.** Only the last raw rows can change: the isolated status of the newest row, and a column whose newest values are NaN until its next finite sample arrives.
- **Forward pass.** Incremental per run, with the state carried over.
- **Backward pass.** It depends on the end of the buffer. Recompute it over the last L ≥ 90 samples of each run, starting from the current end with `odd_ext` and `zi * r[0]`. Older values are frozen.
  - The recomputed tail equals the whole-prefix result exactly, because the backward recursion only looks at later samples.
  - The frozen values differ from it by at most about 0.668^L of the amplitude: below 1e-16 at L = 90.
- **Runs.** A run shorter than 16 samples changes its pad length with every sample. Refilter it whole.

APP_SPEC §4.2 refilters the whole buffer every 100 ms, which is O(n). That is also correct, and simpler.

### 1.10 Constants of this stage

| Constant | Value | Source |
|---|---|---|
| `FS_HZ` | 60.0 | `config.py:110` |
| `LOWPASS_HZ`, `LOWPASS_ORDER` | 15.0, 4 | `config.py:111-112` |
| `MAX_GAP_MS` | 100.0 | `config.py:113` |
| minimum kept rows | 5 | `standardize.py:197` |
| repeated-sample speed on non-WebKit engines | 20 deg/s | `preprocess.py:576` |
| engine sign | webkit −1, blink +1, gecko +1, unknown +1 | `conventions.py:476,736` |
| rounding | t 2, acc 4, rr 3 decimals | `app/src/core/constants.ts:128-134` |

---

## 2. Detector, windows, m4 classifier, grip check and stopping rule

`predict_stream(t, accg, acc, gyro, model)` (`m4_production.py:81-119`) runs these steps:

1. `st = {t, f = accg, acc, gyro, valid}`;
2. `tab, W = segment_table(st, model.seg_params)`;
3. segment probabilities;
4. `attempts`;
5. `rest_up_z`;
6. the grip gate;
7. `stopping_rule`.

### 2.1 Activity (`m4_detect.py:105-117`)

```
w  = fill_gaps_linear(t, gyro)                 # per column: interp over NaN rows, edges hold the nearest valid value,
                                               # an all-NaN column stays NaN (preprocess.py:148-159)
s  = |w| row norm  (sqrt((w0*w0 + w1*w1) + w2*w2))
s  = where(isfinite(s), s, 0)
if len(s) < 12: return s                       # UNFILTERED for streams shorter than 12 samples
sos_a = butter(2, 5 Hz, fs=60, 'sos') = [[0.04948995626867706, 0.09897991253735412, 0.04948995626867706, 1.0, -1.2796324249978088, 0.47759225007251704]]
zi_a  = sosfilt_zi(sos_a) = [[0.9505100437313232, -0.4281022938038401]]
act   = sosfilt(sos_a, s, zi = zi_a * s[0])    # ONE pass, causal (poles |p| = 0.691)
```

- `smooth_hz` = 5 and `causal` = True (`m4_detect.py:76-77`). The zero-phase variant (`:117`) is not used in production.
- `fill_gaps_linear` uses `np.interp(t[nan], t[ok], X[ok])` (§5.1). Live, a trailing NaN run holds the last valid value until the next valid sample arrives.

### 2.2 Hysteresis bouts (`find_segments`, `m4_detect.py:147-203`)

The parameters come from `SegParams` (`m4_detect.py:68-82`) and the fitted `SEG_GRID` entry.

```
if n < 3: no segments
dt     = median(diff(t))                                   (the prefix's own grid, about 16.667)
lag    = causal_delay_ms(p) = 44.429095814613106 ms        (group delay of (b, a) at 0.5 Hz, m4_detect.py:91-99; export it)
n_end  = max(1, round(end_sustain_ms / dt))                # 15 (es250) or 24 (es400)
n_move = max(1, round(sustain_ms / dt))                    # 6
n_look = max(n_end, round(lookahead_s * 1000 / dt))        # 30
still  = act < lo_dps
runs   = maximal True runs [a, b) of still
seps   = runs with (b - a) >= n_end  OR  a == 0           # a still run touching the start always separates
bouts  = stretches between separators:
         prev_end = 0, prev_sep = none
         for each sep k = (a, b):  if a > prev_end: bout (prev_end, a, prev_sep);  prev_end = b; prev_sep = k
         if prev_end < n: bout (prev_end, n, prev_sep)     # the last bout, possibly not ended
for each bout (a, b, ks):
    keep only if some run of (act[a:b] > hi_dps) has length >= n_move
    ended  = b < n
    i_on   = a;  i_end = b if ended else n - 1             # end = FIRST still sample of the closing separator
    onset_ms = t[i_on] - lag;  end_ms = t[i_end] - lag
    drop if end_ms - onset_ms < min_duration_ms (150)      # max_duration_s (12) is NOT applied
    still_before_s = 0 and started_active = True           if ks is none
                   = (t[sb-1] - t[sa] + dt) / 1000         with (sa, sb) = seps[ks], otherwise
    if ended:  j = i_end; while j < n and still[j] and j - i_end < n_look: j++
               still_after_s = (j - i_end) * dt / 1000;  dec_i = min(n - 1, i_end + n_look - 1)
    else:      still_after_s = 0;  dec_i = n - 1
    Segment(onset_ms, end_ms, i_on, i_end, decided_ms = t[dec_i],
            min(still_before_s, still_cap_s = 3.0), min(still_after_s, lookahead_s = 0.5),
            started_active, ended, peak_dps = max(act[i_on..i_end]), index = number of segments kept so far)
```

`round()` is Python's half-even rounder. The ratios are about 15.0, 6.0 and 30.0, never a tie.

### 2.3 Segment table (`m4_detect.py:327-350`)

For each segment, in time order (`prev` = the previous kept segment):

- `win, can = window_for(st, onset_ms, end_ms, crop_s = 1.0, post_s = lookahead_s = 0.5)` (§2.4);
- `feats = segment_features(can, seg, prev)` (§2.8);
- `canonical_ok` = True.

If `canonical` raises `ValueError` (§2.4), the window is all NaN, the features are absent (NaN) and `canonical_ok` = False. The windows are stacked as float32 (`:349`).

Row fields: `seg`, `onset_ms`, `end_ms`, `decided_ms`, `ended`, `peak_dps`, `canonical_ok` and every name of `ALL_FEATURES` (`:228-229`; NaN when it is not computed).

### 2.4 `window_for` and `canonical` (`m4_detect.py:242-256`; `standardize.py:239-284`)

```
crop    = rows with  onset_ms - 1000 <= t <= end_ms + 500        (crop_stream, inclusive both ends)
stream' = as_stream(crop)                                         (gravity_magnitude = None, baseline_gravity = None)
canonical(stream', onset_ms, end_ms):
    raise if not finite or end_ms <= onset_ms
    i_on  = searchsorted(t, onset_ms, 'left');  i_end = searchsorted(t, end_ms, 'right') - 1
    i_on  = max(i_on, 0);  i_end = min(i_end, n - 1);  raise if i_end - i_on < 2      (fewer than 3 samples)
    f = fill_gaps_linear(t, stream'.f);  w = fill_gaps_linear(t, stream'.gyro)        # on the CROP
    g = gyro_gravity(t, f, w, i_on, i_end, still_dps = 20, rest_ms = 300, magnitude = None, fallback = None)   (§2.5)
    u = g / |g| per row (a zero row gives NaN; unit_rows, preprocess.py:195-200)
    then the channels of §2.6
```

With `lag` = 44.43 ms (2.67 samples), `i_on` is the stream index of the bout start minus 2, and `i_end` is the bout end minus 3. The window therefore spans `[t[i_on] − lag, t[i_end] − lag]`. The crop ends about 456 ms after the bout end. That is before `decided_ms`, which is 483 ms after it: the window uses nothing the detector has not seen.

### 2.5 gravity_v1 (`preprocess.py:376-544`)

**`integrate_gyro(t_ms, w_dps)`** (`:376-418`): this gives the device orientation relative to the first sample of the crop.

```
ts = t_ms / 1000;  W = w_dps * (pi/180)                                   (DEG2RAD = 0.017453292519943295)
q  = (x, y, z, w) = (0, 0, 0, 1)
for k in 0..n-2:
   dt = ts[k+1] - ts[k];  a = W[k];  b = W[k+1]
   if any non-finite in a, b, dt: q unchanged (missing += 1); continue
   o  = ((a + b) * 0.5) * dt  (per component)
   th = sqrt((ox*ox + oy*oy) + oz*oz)
   if th < 1e-15: q unchanged; continue
   h = 0.5*th;  s = sin(h)/th;  dq = (ox*s, oy*s, oz*s, cos(h))
   q = q ⊗ dq:  (w*dx + x*dw + y*dz - z*dy,  w*dy - x*dz + y*dw + z*dx,
                 w*dz + x*dy - y*dx + z*dw,  w*dw - x*dx - y*dy - z*dz)
   q = q / sqrt(((x*x + y*y) + z*z) + w*w)
R[k] = quaternion_to_matrix(q_k)         (normalises again, then the standard [x,y,z,w] matrix, :304-319)
```

**`gyro_gravity`** (`:465-544`; arguments from `standardize.py:261-262`):

```
Fw[k]   = R[k] @ f[k]                                    # specific force in the frame of sample 0
wmag[k] = |w[k]|  (NaN gives +inf)
t_on = t[i_on];  t_end = t[i_end]
rest(t_from, t_to) = mean of Fw[k] over k with t_from <= t[k] <= t_to AND wmag[k] < 20 AND Fw[k] all finite
pre  = rest(t_on - 300, t_on - 1e-9);  if fewer than 5: rest(t_on - 600, t_on - 1e-9)
       if still fewer than 5: gw_pre = Fw[i_on], pre_n = 0           (no fallback: as_stream drops it)
post = rest(t_end, t_end + 300);       if fewer than 5: rest(t_end, t_end + 600);  if still fewer than 5: none
g_fwd[k] = R[k]^T gw_pre
with post:    g_bwd[k] = R[k]^T gw_post;  lam = clip((t - t_on)/(t_end - t_on), 0, 1)   (span > 0 is guaranteed)
              g = (1 - lam) g_fwd + lam g_bwd;  closure_deg = angle_deg(gw_pre, gw_post)
without post: g = g_fwd;  closure_deg = NaN;  method 'fwd_only'
magnitude = |gw_pre| if (pre_n > 0 and 8.6 <= |gw_pre| <= 11.0) else 9.80665
g = g / |g| * magnitude   (row norm; zero gives NaN)
```

The boundaries `t_on − 300` and `t_end + 300` fall exactly on grid points in real arithmetic, because 18 × step = 300 and 36 × step = 600. Whether such a sample is included is decided by float64 rounding. TS must evaluate exactly these expressions on the exact grid values (§5.3).

### 2.6 Channels (`standardize.py:263-282`; order in `CHANNELS`, `:61-64`)

```
a  = f - g;  wv = w·u;  av = a·u;  ah = |a - av u|;  wh = |w - wv u|          (each dot product (x0y0 + x1y1) + x2y2)
yaw = cumulative_trapezoid(wv, t/1000, initial 0):  yaw[0] = 0,
      yaw[k+1] = yaw[k] + ((ts[k+1]-ts[k]) * (wv[k+1] + wv[k])) / 2.0;   then yaw -= yaw[i_on]
tilt = degrees(atan2(|u × u[i_on]|, u·u[i_on]))                               (angle_deg, preprocess.py:424-430)
X rows: acc_x,y,z = a | gyro_x,y,z = w | up_x,y,z = u | gyro_mag = |w| | acc_mag = |a| | acc_vert = av |
        acc_horiz = ah | yaw_rate = wv | gyro_horiz = wh | yaw_cum = yaw | tilt_from_start = tilt     (float64)
```

`np.degrees(x)` is `x * (180/π)`, and the cross product is `[u1v2 − u2v1, u2v0 − u0v2, u0v1 − u1v0]`.

### 2.7 `time_normalised` (`standardize.py:290-301`; `N_NORM` = 64, `config.py:116`)

```
grid[i] = i * ((end - start)/63) + start  for i < 63;  grid[63] = end      (np.linspace exactly; checked equal)
Xn[c]   = np.interp(grid, can.t, X[c]) for every channel                     (out of range: edge values)
Xn[yaw_cum] -= Xn[yaw_cum][0]                                                (so it is exactly 0.0 at index 0)
window  = float32(Xn)                                                        (Math.fround every value)
```

`tilt_from_start` is not re-zeroed. It is 0 at the grid sample `i_on`, which lies up to one sample after `onset_ms`.

### 2.8 Segment features: the 'robust' set (`m4_detect.py:208-229,259-324`)

The feature order is `PHYSICS_FEATURES + ('still_before_inst', 'still_after')`: 35 names. Read the order from the export, never from the code.

Let `X`, `i0 = can.i_on` and `i1 = can.i_end` refer to the crop. `ch(c)` is `X[c][i0..i1]` (inclusive), and `t` is `can.t[i0..i1]`.

| Feature | Formula (`m4_detect.py:289-316`) |
|---|---|
| `dur_s` | max((end_ms − onset_ms)/1000, 1e-3) |
| `log_dur` | ln(dur_s) |
| `peak_w`, `mean_w` | max and mean of wm = ch(gyro_mag) |
| `rot_total` | sum(wm) / 60.0 |
| `peak_a`, `mean_a` | max(am); mean(am), or 1e-9 if it is 0 (am = ch(acc_mag)) |
| `abs_net_yaw` | \|yaw_cum[i1] − yaw_cum[i0]\| |
| `tilt_end` | tilt_from_start[i1] |
| `peak_tilt_rate` | max ch(gyro_horiz) |
| `peak_abs_yaw_rate` | max \|ch(yaw_rate)\| |
| `t_peak_w` / `t_peak_a` | tt[argmax wm] / tt[argmax am] (first maximum), where tt = (t − t0)/max(t_last − t0, 1e-9) |
| `w_first_frac` | sum(wm[tt ≤ 0.5]) / tot_w, where tot_w = sum(wm), or 1e-9 if it is 0 |
| `a_last_frac`, `a_first_frac` | mean(am[last q]) / mean_a and mean(am[first q]) / mean_a, where q = max(1, round(0.2·len)) |
| `up_y0`, `up_z0`, `abs_up_x0` | u at i0 (y, z, \|x\|) |
| `up_y1`, `up_z1`, `abs_up_x1` | u at i1 |
| `d_up_y`, `d_up_z`, `d_abs_up_x` | up_y1 − up_y0; up_z1 − up_z0; \|up_x1\| − \|up_x0\| |
| `closure` | can.closure_deg, or 0 if it is NaN |
| `still_before_inst` | min(still_before_s, 1.0) (`INSTRUCTED_STILLNESS_S`, `:218`) |
| `still_after` | still_after_s |
| `disp_vert`, `disp_y0`, `disp_z0`, `abs_disp_x0`, `disp_norm`, `v_up_peak`, `v_down_peak` | `_kinematics` below |

`_kinematics` (`m4_detect.py:259-281`) uses `t` in s, `a` and `w` (rows i0..i1) and `u0` = u at i0. If there are fewer than 3 samples, or any a or w value is not finite, all seven are NaN. Otherwise:

```
R, _ = integrate_gyro(t*1000, w)              # NOTE: (t_ms/1000)*1000, as written; device(k) -> device(onset)
aw[k] = R[k] @ a[k]
v  = cumulative_trapezoid(aw, t, axis 0, initial 0)
T  = t_last - t0;  if T > 0: v -= outer((t - t0)/T, v_last)          # ZUPT: zero velocity at both ends
dp = trapezoid(v, t, axis 0)  = sum_k (t[k+1]-t[k]) * (v[k+1] + v[k]) / 2
vv = v @ u0
disp_vert = dp·u0;  disp_y0 = dp[1];  disp_z0 = dp[2];  abs_disp_x0 = |dp[0]|;  disp_norm = |dp|
v_up_peak = max(0, max vv);  v_down_peak = max(0, -min vv)
```

The `g_*` copies (`:318-323`) and the context features `seg_index`, `prev_dur` and `started_active` are computed, but the 'robust' set does not read them.

### 2.9 The segment classifier (`m4_models.py:149-237`)

The production settings are `scaling` 'robust' and C = 1 (`m4_e2e.py:70-71`; `m4_pipeline.py:105-106`).

```
med = nanmedian of the training segments' features (NaN gives 0);  iqr = q75 - q25 (np.nanpercentile, linear; 1 when <= 1e-9)
Z   = (F - med) / iqr;  Z = where(isfinite(Z), Z, 0);  Z = clip(Z, -5, 5)          (m4_models.py:162-172)
L   = Z @ coef.T + intercept                       # multinomial (lbfgs), coef (3, 35)
P   = softmax(L): exp(L - max L) / sum             (sklearn utils.extmath.softmax)
classes_ = ['approach', 'other', 'return']         (sklearn sorts the labels; map by name)
```

Missing features (a failed canonical) become 0 after scaling, which is the training median. Their probabilities are still reported (`m4_production.py:98-102`).

### 2.10 The attempt rule (`m4_e2e.py:260-295`)

```
cls     = argmax over ('approach', 'return', 'other') of P      # this order; a tie goes to the first
attempt = cls == 'approach' AND P(approach) >= det_threshold (0.5) AND canonical_ok
```

There is no pairing (`PRODUCTION_PAIRING` = False, `m4_config.py:45`). Every bout classified as an approach is an attempt. Returns and other movements are ignored.

### 2.11 The m4 hand classifier and its calibrator

**Yaw8** (`m4_models.py:57-77`):

```
x = nan_to_num(window[yaw_cum = 15][0::8])     # 8 values at indices 0, 8, ..., 56 (x[0] is always 0)
z_raw = coef · x + intercept                   # LogisticRegression(C = 1), mirror-augmented, fitted on oracle windows
```

`nan_to_num` maps NaN to 0 and ±inf to ±1.7976931348623157e308.

**Calibrator.** `nested` picks one of `none`, `temperature` or `isotonic_sym` inside the fit (`m4_models.py:352-388`):

| Method | P(right) of one attempt |
|---|---|
| `none` | sigmoid(clip(z, −40, 40)) (`m4_models.py:42-43,257-258`) |
| `temperature` | sigmoid(clip(z / T, −40, 40)) (`:278-279`) |
| `isotonic_sym` | p = clip(iso(\|z\|), 0.01, 0.99); P = p if z ≥ 0 else 1 − p (`:324-328`) |

`iso` is sklearn's `IsotonicRegression.predict`, fitted on the cross-fitted logits and their mirror images (`:320-322`):

```
T  = clip(|z|, X_min_, X_max_)
iso(T) = interp1d linear on (X_thresholds_, y_thresholds_):
         j = searchsorted(x, T, 'left') clipped to [1, len-1];  lo = j-1;  hi = j
         (y[hi] - y[lo]) / (x[hi] - x[lo]) * (T - x[lo]) + y[lo]
         (with one threshold: the constant y[0])
```

This is not `np.interp`: at a knot the result can differ by one ulp. `CalibratorJson` in `types.ts` must get this variant (§8).

**Calibrated logit:** z = logit(P) = ln(p/(1 − p)) with p clipped to [1e-6, 1 − 1e-6] (`m4_models.py:39,46-48`; `m4_e2e.py:288`). The attempt's own displayed P is `1/(1 + exp(−z))` with no clip (`m4_production.py:114`).

### 2.12 The grip check (`m4_detect.py:424-444`; `m4_config.py:25,28`)

```
rest_up_z(st, DEFAULT, min_s = 0.25):
  if n == 0: NaN
  act = activity(t, gyro, DEFAULT);  segs = find_segments(t, gyro, DEFAULT, act)
  first_on = segs[0].onset_ms if any else +inf     # the first candidate bout, ended or not, lag-corrected
  nrm = |f| per row;  ok = isfinite(nrm) AND nrm > 0;  upz = f_z / nrm (NaN where not ok)
  sel = ok AND t < first_on AND act < 20
  if count(sel) < max(1, round(0.25 * 60)) = 15:  sel = ok AND t < t[0] + 250
  return median(upz[sel]) if any else NaN          # an even count averages the two middle values: (a + b)/2
grip = isfinite(rest) AND rest < GRIP_GATE_UPZ (0.0; the page uses -0.05: analysis/stage3/runtime.py PRODUCTION_GRIP_GATE_UPZ)
```

- With the gate closed: P = 0.5, status `grip`, not answered. The attempts are still listed (`m4_production.py:104-109`).
- The rest pose comes from the samples before the stream's first movement. A stream that failed the gate can therefore never pass it. The page must start a new stream after the user turns the phone (APP_SPEC §4.2.5).
- The check reads the raw specific-force direction `f/|f|`, not the canonical `up_z`.

### 2.13 Aggregation and the stopping rule (`m4_online.py:38-66`; `m4_config.py:38,49,51`)

```
attempts sorted by decided_ms (pandas sort_values: quicksort; use (decided_ms, seg) for a stable equivalent)
p = 0.5;  k = 0
for k in 1..min(k_max = 5, number of attempts):
    p = sigmoid(s_k * mean(z_1..z_k))           # M.sigmoid clips the argument to ±40; mean = sum/k, summed left to right
    if max(p, 1 - p) >= p_stop (0.9): return confident, n_used = k, decided_ms of attempt k
if k == 0: return no_attempt, P = 0.5, not answered, decided_ms = NaN
answered = max(p, 1 - p) >= p_answer (0.5)      # always true
status = 'k_max' if k >= k_max else 'stream_end';  decided_ms of attempt k
```

The production aggregation is `scale_sessions`:

- It is the declared rule `calibrated_then_simplest` (`m4_config.py:55-62`; `m4_pipeline.py:143-151`).
- Only `scale_sessions` has `calibrated` = true in `tables/m4/m4_calibration.json` `stop`.
- Its scale is `AggregateScale.fit_sessions(form='power')` (`m4_models.py:459-476`; `m4_aggregate.py:93-112`): s_k = s₁·kᵇ.
- `scale(k)` returns `s[k]`, or `s[max k]` beyond 5 (`m4_models.py:478-483`). That cannot happen with `k_max` = 5.
- For scale only (not the all-persons values), the P01-out fold fitted s = 0.944, 1.582, 2.139, 2.650, 3.128.

### 2.14 What `predict_stream` returns (`m4_production.py:110-119`)

```
{ p_right, percent_right = int(round(100 p)) (half to even), answered, status, n_attempts_used, decided_ms,
  rest_up_z, grip_warning,
  attempts: [{onset_ms, end_ms, decided_ms, p_approach, p_right = sigmoid(z)}] sorted by decided_ms,
  segments: [{onset_ms, end_ms, decided_ms, p_approach, p_return, p_other}] for every row of the table }
```

This is `StreamPrediction` in `types.ts`.

### 2.15 Live operation

This rule is normative and agrees with APP_SPEC §4.2.

**Freezing.** A bout of the current prefix is frozen once three conditions hold:

- `ended`;
- `i_end + n_look − 1 ≤ n − 1`;
- `t_last ≥ decided_ms + settle`, with `settle` = 1000 ms.

Its row, window and every model's output are then computed on the prefix of that tick and never again. A later bout whose onset lies within half a grid step (8.33 ms) of a frozen one is that same bout.

**Why this matches the reference.** Every input of a frozen bout lies at or before `decided_ms`:

- the crop ends at `end_ms` + 500 ms;
- the still-after count ends at `i_end` + 29;
- the separator needs `n_end` samples.

After 60 settled samples the prefix values equal the full recording's to within about 1e-13 (§5.2). The frozen result therefore equals `predict_stream`'s row for that bout on any longer recording. The one exception is a sensor gap of more than 1 s that crosses `decided_ms` (`fill_gaps_linear` would hold the value). APP_SPEC resets the stream when a gap exceeds 1 s.

**The rest of the loop:**

- The stopping rule runs over the frozen attempts in `decided_ms` order. Live, the statuses are `watching`, `confident`, `k_max`, and `stream_end` at the cap.
- The grip gate is evaluated on the prefix when the first attempt is frozen. Once the first DEFAULT bout exists, `rest_up_z` no longer changes.

**Python counterpart:** `reference_predict(mode='live')` (§4.6).

Lane m4 evaluated a different rule: decide at `decided_ms` on the prefix itself, with no settle margin. Its cost relative to the full recording is in finding 1. Either rule is defensible. The settle rule is chosen because it makes the live result equal the full-recording result.

---

## 3. How the other models read one attempt

Every model reads the same frozen float32 `ApproachWindow`: (17, 64) from §2.7, with `onset_ms` and `end_ms`. The attempt list is m4's (§2.10) for every model.

### 3.1 The final CNN (`final_model.json`; `stage2/lanes/final_infer.py`)

1. **Inputs.** `A` = the float32 window. The body-frame channels are computed in float64 from `A` (`:44-60`):

   ```
   u = A[up], w = A[gyro], a = A[acc]       (per sample)
   h = e_z - u_z u   (h = -u_z*u; h_z += 1)
   nrm = sqrt((h0² + h1²) + h2²)
   h = h/nrm if nrm > h_min (1e-6) else 0
   s = u × h
   roll_rate = w·h;  pitch_rate = w·s;  acc_lat = a·s;  acc_fwd = a·h
   ```

   Then the result is cast to float32 (`:66`), and the 21-channel input is `[A; body]` (`:63-67`).
2. **Screen flip.** Not used on the page (APP_SPEC §4.4). If `rest_up_z_before < threshold` (0.0), the window is multiplied by `flip_signs`, giving `[−1,1,−1, −1,1,−1, −1,1,−1, 1×8]` (`:70-80,167-174`). Here `rest_up_z_before` is the median of `up_z` over the 30 samples 0.5 s before the onset (`final_robust.py:54-59`).
3. **Channel selection.** `model.channels` takes 12 channels:

   ```
   [gyro_mag, acc_mag, acc_vert, acc_horiz, yaw_rate, gyro_horiz, yaw_cum, tilt_from_start,
    roll_rate, pitch_rate, acc_lat, acc_fwd]
   ```

   They are converted to float64 (`:113-114`).
4. **Each of the 4 members** (`:116-130`):

   ```
   h = (x - center)/scale
   3 times:
       h = conv1d_same(h, W (32, Cin, K))      K = 7, 5, 3; zero padding (K-1)/2; no bias;
                                                cross-correlation: out[o,t] = Σ_c Σ_k W[o,c,k]·h_pad[c, t+k]
       h = (h - bn_mean)/sqrt(bn_var + 1e-5)*bn_weight + bn_bias
       h = max(h, 0)
   pooled[c, b] = mean of h[c, 8b..8b+7]       (_bins: edges round(linspace(0, 64, 9)) = 0, 8, ..., 64)
   z_m = flatten(pooled) · fc_weight + fc_bias  (flatten is channel-major: index c*8 + b; 256 weights)
   ```

5. **Ensemble.** `z_raw` = mean of the 4 member logits (`:131`).
6. **Calibrator.** `platt`, with a = 1.0183420401819008 and b = −0.011375652259257248 (`:144-164`):

   ```
   p = 1/(1 + exp(-clip(a·z_raw + b, ±40)))
   p = clip(p, 1e-4, 1 - 1e-4)
   ```

   This is lane m3's in-fold calibrator. Lane final's remedy is 'none' (`final_export.py:161-162,207-213`).
7. **k-aggregation** (`:177-184`):

   ```
   z_i = ln(p_i) - log1p(-p_i), with p clipped to [1e-12, 1 - 1e-12]
   P_k = 1/(1 + exp(-(Σ z_i)/tau_k))       tau = {1: 1.0, 2: 1.2422…, 3: 1.5528…, 4: 1.6447…, 5: 1.9725…}
   ```

   tau_1 = 1, and k above 5 uses tau_5. No clip is applied in this sigmoid.

`final_infer` reproduces the fitted pipeline to 2.2e-16 (`final_model_check.json`). The fitted float32 torch pipeline differs from it by 1.0e-7.

### 3.2 PHYS6 logistic (`m1_lr_phys6_nested`)

**Features** (`m1_features.py:229-308`; `m1_experiments.py:43-47`). Use the float32 window `A`, read as float64, and `dur` = (end_ms − onset_ms)/1000 of the detected bounds. The cache used `duration_ms` = end − onset, oracle (`standardize.py:451`).

```
dt   = dur / 63
body: as in §3.1 but h = h/nrm if nrm > 1e-6 else NaN, and NOT rounded to float32          (m1_features.py:229-243)
roll = cumtrapz(roll_rate, dt);  lat = cumtrapz(acc_lat, dt)
       cumtrapz: c[0] = 0, c[k+1] = c[k] + (0.5*(x[k+1] + x[k]))*dt                           (:246-250)
_at(x, 0.5) = 0.5*x[31] + 0.5*x[32];  _at(x, 1.0) = x[63]                                   (:253-260)
features (order): yaw_q50 = _at(yaw_cum, .5), yaw_q100 = yaw_cum[63], roll_q50, roll_q100, latvel_q50, latvel_q100
```

If any of the six is not finite, Python raises (`:359-361`). The page shows PHYS6 as unavailable for that attempt.

**Fitted pipeline** (`m1_models.py:428-465`, `_finish` at `:290-313`, `Prep` at `:129-193`):

- The treatment and C are chosen jointly inside the fit, from the treatments `rms`, `none`, `robust`, `clip`, `asinh` and `ampnorm` (`m1_experiments.py:65-69`) and C in `CS_DEFAULT` (`m1_models.py:46`).
- In all 20 LOSO folds the choice was `asinh` with C = 0.1 (`results/m1_lr_phys6_nested.json`). The all-persons fit is expected to choose the same, but the export must record what it did choose.
- The model is antisymmetric: every feature is odd, no feature is centred and there is no intercept (`m1_models.py:345,361`).

```
asinh: W = s_w * asinh(x / s_w)         s_w = median|x_train| (1 when 0)          (m1_models.py:157-160,188-189)
rms:   Z = W / s                         s = sqrt(mean(W_train²))  (1 when 0)       (:166-168,182,193)
clip:  W = clip(x, -q99|x|, +q99|x|);  robust: s = 1.4826·median|W|;  none: s = 1;  ampnorm: other columns (§8)
z_raw = coef · Z   (decision_function; intercept_ = 0)
p     = clip(1/(1 + exp(-z_raw)), 0, 1)            (calibration 'none': lane m1, and lane final's in-fold remedy in 598/598 rows)
z     = ln(p'/(1 - p')), with p' = clip(p, 1e-6, 1 - 1e-6)                              (m1_models.py:43,76-78)
P_k   = 1/(1 + exp(-(Σ z_i) / k^alpha))                                                (m1_analysis.py:596-615)
```

`alpha` comes from `fit_agg_alpha` on the grid 0, 0.05, …, 1 (`m1_models.py:243-265`). The LOSO folds chose 0.15 to 0.30.

### 3.3 The yaw reference (`example_yaw_logreg`; `stage2/scripts/reference.py:64-85`)

```
x     = window[yaw_cum][0::8]             (no nan_to_num: a NaN gives a NaN P, so unavailable)
z_raw = coef · x + intercept             (LogisticRegression(max_iter=1000), C = 1, trained on the window and its mirror)
P     = expit(z_raw) = 1/(1 + exp(-z_raw))  (predict_proba[:, 1]; uncalibrated, as the reference unit evaluates it)
z     = z_raw (for aggregation)
```

**No k-aggregation was fitted for this model.** The harness's k-curve averages P (`cv.py:542-580`): `P_k` = mean(P_i). Its calibration for k ≥ 2 is unknown, and it is under-confident by construction. This is a decision for the orchestrator (§7).

In lane final, the in-fold remedy for this model chose none (486 rows), temperature (84) or Platt (28) (`results/final_cal_ref_auto.json`). The page uses none.

### 3.4 Can each model run on the detector's window exactly as evaluated?

| Model | Input contract on a detector window | Evaluated end to end on detector windows? | What differs from its evaluation | What the page does |
|---|---|---|---|---|
| m4 (yaw8 + nested calibrator + s_k) | identical | **yes**: the calibrator was fitted on cross-fitted logits of detector-chosen segments (`m4_e2e.py:415-430`), s_k on synthetic-session attempts with 1 s stillness at each junction (`m4_aggregate.py:36-37`) | the hand model itself was trained on oracle windows (`hand_train` 'oracle'); its training streams were float32 | run as is |
| final CNN | identical: `standardize.canonical` + `time_normalised` + float32, the same code path as the cache | **no**: trained and tested on oracle windows (`final_SUMMARY.md` caveat, line 416) | (a) the bounds come from the causal hysteresis, lag-corrected, and not from the app's online_v1 (bound jitter of ±200 ms cost −0.8 points, n.s.; a 100 ms late onset −0.9, n.s.; FINAL_13); (b) canonical runs on the crop, so the widened post-rest window is cut about 456 ms after the end; (c) no calibration magnitude or baseline-gravity fallback (as_stream); (d) tau_k was fitted on k random RECORDINGS of one person and hand: lane m4 found the recording-fitted scale over-confident on cue-free sessions (`m4_aggregate.py:3-8`); (e) its screen check is replaced by m4's grip gate | run as is; label it "end-to-end on detected windows not yet validated (lane prod)"; stopping thresholds are m4's |
| PHYS6 | window + detected duration | **no** (oracle windows; duration = oracle) | as for the CNN; integrals scale with the detected duration; NaN on a horizontal screen | run as is; unavailable when NaN |
| yaw reference | identical | **no** (oracle windows) | = m4's raw logit without the calibrator; no k-aggregation fitted | run as is; P_k = mean P (§7) |

### 3.5 Every model's stopping

- **Shared thresholds.** One stopping rule (§2.13), with `p_stop` = 0.9, `k_max` = 5 and `p_answer` = 0.5, is applied to each model's own P_k over the shared attempt list.
- **Code comment.** State in the code that these thresholds and the detector were chosen for lane m4's classifier, and that lane prod, which would refit them for the final model, has not run.
- **Unavailable attempts.** An attempt where a model is unavailable is skipped for that model only. Its k then counts fewer attempts than m4's.

---

## 4. What to export, and the Python reference

### 4.1 Files

These follow APP_SPEC §6 and go in `app/src/predict/models/json/`. Every file carries `format`, `identity` (`data_id`, cache version, snapshot sha256, code fingerprint, git commit), `training` (n_trials, n_persons, `cap_per_person` 40, seed 20261027) and `persons` P01..P20.

| File | Content | Writer |
|---|---|---|
| `final_model.json` | as written by `final_export.py` (1.6 MB, kept byte for byte) | existing |
| `m4_production.json` | `detector` (`DetectorBundle`), `hand` (yaw8), `calibrator`, `aggregation`, `stopping` | NEW: `fit_production()` |
| `m1_phys6.json` | PHYS6 pipeline | NEW: `make_linear_select(..., store=)` on an all-persons fold |
| `ref_yaw8.json` | yaw reference | NEW: `yaw_logreg` on an all-persons fold |

- **Fitting on all persons.** Fit with the same code paths as the evaluation. Use the rows from `cv.cap_rows(person_id, y, 40, SEED)` (`cv.py:187-211`), which is what `final_export._all_fold` (`final_export.py:32-40`) and `fit_production` (`m4_production.py:137`) both use.
- **Number format.** Write numbers with Python `repr`, which round-trips float64.
- **Sizes.** The weights in `final_model.json` are float32 values stored as float64. Shrinking them with float32 strings plus `Math.fround` is exact, but APP_SPEC keeps the file byte for byte. Keep it.

### 4.2 `m4_production.json` › `detector`

This matches `DetectorBundle` in `types.ts` plus §8.

| Key | Shape and type | Value or source |
|---|---|---|
| `segParams` | object of 10 numbers and a bool (`causal`) | `asdict(E.SEG_GRID[fitted.seg_cfg])` (`m4_production.py:75`) |
| `segCfg` | string | `fitted.seg_cfg` |
| `causalDelayMs` | float64 | `causal_delay_ms(segParams)` = 44.429095814613106 |
| `activitySos`, `activityZi` | (1, 6), (1, 2) float64 | §2.1 |
| `lowpassSos`, `lowpassZi`, `lowpassPadlen` | (2, 6), (2, 2) float64, int 15 | §1.7 (proposed, §8) |
| `segmentClassifier.features` | 35 strings | `fitted.segclf.features` |
| `segmentClassifier.classes` | 3 strings | `segclf.m.classes_` (['approach', 'other', 'return']) |
| `segmentClassifier.scaling`, `center`, `scale`, `clip`, `fill` | 'robust', (35), (35), 5.0, null | `segclf.med`, `segclf.iqr` (`m4_models.py:162-172`) |
| `segmentClassifier.coef`, `intercept` | (3, 35), (3) float64 | `segclf.m.coef_`, `segclf.m.intercept_` |
| `attemptRule` | {detThreshold 0.5, pairReturns false} | `spec` |
| `stopping` | {pStop 0.9, kMax 5, pAnswer 0.5} | `ProductionModel` (`m4_config.py:38,49,51`) |
| `grip` | {gateUpz -0.05 (lane m4: 0.0), restMinS 0.25, restParams DEFAULT} | `m4_config.py:25,28`; `m4_detect.py:88`; `stage3/runtime.py` |
| `constants` | numbers | FS_HZ 60, LOWPASS_HZ 15, LOWPASS_ORDER 4, MAX_GAP_MS 100, STILL_GYRO_DPS 20, REST_MS 300, MIN_REST_N 5, N_NORM 64, GRAVITY 9.80665, MAG_MIN 8.6, MAG_MAX 11.0, INSTRUCTED_STILLNESS_S 1.0, MIN_ROWS 5, REPEAT_MIN_DPS 20, MIN_ACTIVITY_N 12 |

### 4.3 `m4_production.json` › model

| Key | Shape | Source |
|---|---|---|
| `hand` | {kind 'yaw8', channel 'yaw_cum', index 15, step 8, coef (8), intercept, nanToNum true} | `fitted.hand.m.coef_[0]`, `intercept_[0]` |
| `calibrator` | `{method: 'nested', chosen, inner}`; inner = `{method: 'none', logit_clip: 40}` \| `{method: 'temperature', T, logit_clip: 40}` \| `{method: 'isotonic_sym', x_thresholds, y_thresholds, x_min, x_max, p_lo: 0.01, p_hi: 0.99, interp: 'interp1d_left', eval: 'abs_reflect'}` | `fitted.calib.chosen`, `.inner` (`X_thresholds_`, `y_thresholds_`, `X_min_`, `X_max_`) |
| `logitEps` | 1e-6 | `m4_models.py:39` |
| `aggregation` | {kind 'scaled_mean', name 'scale_sessions', form 'power', s1, b, s: {1..5}} | `ProductionModel.agg` (`.s1`, `.b`, `.s`) |
| `info` | `describe()` without arrays | `m4_production.py:58-68` |

### 4.4 `m1_phys6.json`

```
{ features: [yaw_q50, yaw_q100, roll_q50, roll_q100, latvel_q50, latvel_q100], option, C,
  winsor: 'asinh'|'clip'|'none', winsor_s (6) | winsor_lo/hi (6), centre (6, zeros), scale (6), coef (6), intercept 0,
  hMin 1e-6, bodyNaN true, calibrator {method 'none', p_clip_out [0, 1]}, logitEps 1e-6,
  aggregation {kind 'sum_over_k_pow', alpha} }
```

The fields come from `_linear_store` (`m1_models.py:372-382`) and `info['agg_alpha']`. If `option` comes out as `ampnorm`, the exporter must also add the `_an` divisors (`m1_features.py:447-465`). That needs four more even features (`yaw_abs_total`, `roll_abs_total`, `acc_horiz_rms`, `duration_s`), so it is a §7 issue.

### 4.5 `ref_yaw8.json`

```
{ channel 'yaw_cum', index 15, step 8, coef (8), intercept, calibrator {method 'none'}, aggregation {kind 'mean_p'} }
```

### 4.6 `reference_predict` (Python, for golden fixtures)

Proposed location: `analysis/stage3/reference.py`, with the exporter in `analysis/stage3/export.py`. It is driven by the exported JSON; adapters build the m4 objects from it. A round-trip test checks that the JSON-built models equal the fitted objects to 1e-12.

```python
def reference_predict(source, models, *, engine=None, norm=None, mode='offline', settle_ms=1000.0,
                      max_stream_ms=None) -> dict:
```

**Arguments:**

- `source` is one of:
  - `{'t': [...], 'acc': [[x, y, z]|None], 'accg': [...], 'rr': [[alpha, beta, gamma]|None]}`, a raw buffer rounded as in §1.1;
  - a `TrialRecord` (uses `record['streams']['motion']`);
  - a standardised stream dict `{t, f, acc, gyro, valid}` (offline only).
- `norm`: a raw buffer uses `Normalization(gravity=GravitySign(engine_sign(engine), 'engine', False), rates=RateMapping(verified=True), engine=engine)`. A record uses the session's `Dataset.trial_normalization`.
- `models` is the parsed JSON files: `cnn`, `m4`, `phys6` and `yaw`.

**Steps:**

1. `stream = stream_from_record({'trial_id': None, 'streams': {'motion': {... 'ts': [None]*n}}}, norm)`.
2. `act = activity(stream.t, stream.gyro, segParams)`, and `segs = find_segments(...)` with the model's `segParams`.
3. `tab, W = segment_table(st, segParams)`. For each row, add:
   - the crop bounds and `i_on`/`i_end` of the crop;
   - the gravity metadata (`pre_n`, `post_n`, method, closure, magnitude);
   - the 35 features and `proba` from the exported classifier;
   - `cls` and `is_attempt`;
   - `window` (17 × 64 float32 values) and `body` (4 × 64 float32);
   - for every model: `{z_raw, p, z}`, plus its inputs (`phys6.features6`, `yaw.x8`, `cnn.member_logits[4]`).
4. `rest = rest_up_z(st, DEFAULT, 0.25)` with `{n_sel, fallback, first_onset_ms}`.
5. Per model, compute `k_curve = [{k, P_k, decided_ms}]` and the stopping decision.
6. `m4_check = predict_stream(stream.t, stream.f, stream.acc, stream.gyro, model_from_json(models['m4']))`. Assert that it equals steps 2-5 for the m4 model. This pins the reference to the real function.
7. In `mode='live'`, keep only the attempts with `ended` and `i_end + n_look − 1 ≤ n − 1` and `decided_ms + settle_ms ≤ t_last`. Rerun step 5 with `streamEnded = (max_stream_ms is not None and t_last ≥ max_stream_ms)`, so the status is `watching` until then. Record `frozen_at_ms = max(decided_ms + settle_ms, the first tick time)` per attempt for the replay test.

**Returned dict** (JSON-safe, with NaN as null):

```
{ format: 'nosetouch-ref-1', mode, engine, sign, n_raw, repeats_dropped,
  stream: {t, f, gyro, valid},            # acc optional
  activity: [...],
  segments: [...],                        # step 3
  attempts: [segment indices, decision order],
  grip: {...},
  decisions: {model_id: {...}},
  m4_predict_stream: {...} }
```

**Fixtures:**

- **Committed and synthetic only** (`app/tests/fixtures/predict/*.json.gz`). Make them from `motionlab.synth.simulate_trial` records (`motionlab/synth.py:660`), concatenated on the raw level with time offsets for multi-touch streams. Include edge cases:
  - both engines' signs;
  - a gap of more than 100 ms (an iOS-like 314 ms gap);
  - an all-NaN acc;
  - a stream that starts moving;
  - an unended last bout;
  - the screen away (`flip_device_vectors` on the raw vectors) for the grip;
  - a horizontal screen for the PHYS6 NaN case;
  - mirrored copies.
- **Real recordings** go ONLY to `analysis/data/stage3_fixtures/` (git-ignored). Use single raw records of P01..P20 from the snapshot, and raw concatenations of several records of one person and hand.
  - Never PD01 or anything under `analysis/data/exploratory*`.
  - No names: key them by `trial_key`.

---

## 5. Numerical pitfalls for a float64 TypeScript port, and the tolerances

### 5.1 Exact algorithms to copy

- **`np.interp(x, xp, fp)`**:
  - NaN x gives NaN;
  - x < xp[0] gives fp[0]; x > xp[last] gives fp[last]; x == xp[last] gives fp[last];
  - otherwise find j with xp[j] ≤ x < xp[j+1]; x == xp[j] gives fp[j];
  - else `slope*(x − xp[j]) + fp[j]`, with `slope = (fp[j+1] − fp[j])/(xp[j+1] − xp[j])`. If that is NaN, retry `slope*(x − xp[j+1]) + fp[j+1]`; if that is still NaN and fp[j] == fp[j+1], return fp[j];
  - a single xp gives fp[0].

  It is used in resampling, gap filling, `time_normalised` and `fill_gaps_linear`.
- **`interp1d` (isotonic)**: §2.11. The two differ at knots.
- **`np.linspace`**: §2.7. The last point is set to `stop` exactly.
- **Sums.**
  - A reduction over 3 components runs left to right: `(a0 + a1) + a2` (checked).
  - `np.sum`, `np.mean`, `np.median` over longer arrays (features, rest means) use numpy's pairwise summation, in blocks of 8 for n ≥ 8. A left-to-right sum differs by about 1e-16 relative, which the tolerances allow. Copy the pairwise scheme only if bitwise equality is wanted.
- **Library functions.** `exp`, `log`, `log1p`, `sin`, `cos`, `atan2` and `asinh` may differ by 1 ulp between the C runtime and V8. Never compare them bitwise.

### 5.2 Live filter settling

These figures come from a synthetic test (scratch script, not in the repo): 50 deg/s sine plus a random walk, a prefix against the full recording, relative to the amplitude.

| Prefix | Distance 0 | 10 | 30 | 60 | 90 |
|---|---|---|---|---|---|
| 1000 samples | 2.5e-3 | 1.6e-5 | 5.1e-9 | 2.9e-14 | 0 |

The 1 s settle margin (60 samples) brings the frozen inputs to about 1e-13 or better. APP_SPEC measured 3e-10 at 60 samples on a gyro-like signal; either value is far below the 1e-6 target.

### 5.3 Exact ties

These are decided by float64 rounding. Use the same expressions and never rewrite them as index arithmetic.

- `t[i] = t0 + i*step`.
- The rest windows `t_on − 300`, `t_on − 600`, `t_end + 300` and `t_end + 600` (§2.5), and `t < first_on`.
- The crop bounds `onset_ms − 1000` and `end_ms + 500`.
- `searchsorted` left and right for `i_on` and `i_end`.
- `tt ≤ 0.5` in `w_first_frac`.
- Threshold comparisons (`act < lo`, `> hi`, `P(approach) ≥ 0.5`, `max(p, 1 − p) ≥ 0.9`, `rest < 0`). A value within 1e-9 of a threshold may legitimately differ. The fixture comparer reports such cases instead of failing.

### 5.4 NaN handling

| Situation | Behaviour |
|---|---|
| Gaps over 100 ms | NaN in the stream; the low-pass runs per finite run; `fill_gaps_linear` everywhere downstream |
| All-NaN gyro | activity 0 |
| All-NaN column in a crop | NaN channels, NaN features, probabilities still computed (NaN goes to the median) |
| Window values | can be NaN only in that last case |
| yaw8 | `nan_to_num` |
| CNN | NaN propagates: treat the attempt as unavailable |
| PHYS6 | NaN means unavailable |
| Yaw reference | NaN means unavailable |

### 5.5 Rounding and ordering

- `percent_right` rounds half to even (finding 9).
- Attempts are sorted by `(decided_ms, seg)`.
- argmax ties go to the first index (`np.argmax`).
- The order of classes is by name, never by position.

### 5.6 Tolerances

| Stage | Expected | Assert |
|---|---|---|
| grid t | bitwise | bitwise |
| repeats dropped, NaN pattern | identical | identical |
| stream f, gyro (offline) | about 1e-13 | abs ≤ 1e-9 |
| activity | about 1e-13 | abs ≤ 1e-9 |
| bouts: i_on, i_end, dec_i, ended | identical | identical, except the near-threshold cases of §5.3 |
| onset_ms, end_ms, decided_ms | bitwise | abs ≤ 1e-9 |
| canonical channels | about 1e-12 | abs ≤ 1e-8 (deg, deg/s, m/s²) |
| window (float32) | identical | ≤ 1 float32 ulp per value (a float64 noise of 1e-15 can flip a rounding about once in 1e8 values) |
| segment features | about 1e-12 | rel ≤ 1e-8 |
| P(class) | about 1e-12 | abs ≤ 1e-9 |
| z_raw: yaw8, PHYS6, yaw | about 1e-13 | abs ≤ 1e-9 |
| z_raw: CNN | about 1e-11 | abs ≤ 1e-6 |
| calibrated P, P_k | about 1e-12 | abs ≤ 1e-6 |
| attempts, statuses, n_used, answered, percent_right | identical | identical |
| live replay against `reference_predict(mode='live')` | about 1e-12 | the same asserts |

---

## 6. Constants and their sources

| Name | Value | Source |
|---|---|---|
| lo / hi / sustain / end_sustain | 20 (or 15 fitted) / 30 deg/s / 100 ms / 250 (or 400) ms | `m4_detect.py:72-75`; `m4_e2e.py:52-56` |
| smooth_hz, causal | 5 Hz, True | `m4_detect.py:76-77` |
| min_duration, max_duration | 150 ms, 12 s (not applied) | `m4_detect.py:78-79` |
| still_cap, lookahead, crop | 3 s, 0.5 s, 1 s | `m4_detect.py:80-82` |
| causal delay | 44.429095814613106 ms | `m4_detect.py:91-99` |
| instructed stillness | 1.0 s | `m4_detect.py:218` |
| still_dps, rest_ms, min_rest_n | 20 deg/s, 300 ms, 5 | `config.py:114-115`; `preprocess.py:475` |
| gravity magnitude | \|g_pre\| in [8.6, 11.0], else 9.80665 | `preprocess.py:538-540`; `conventions.py:66` |
| det_threshold, pair_returns | 0.5, False | `m4_e2e.py:67,72` |
| p_stop, k_max, p_answer | 0.9, 5, 0.5 | `m4_config.py:38,49,51` |
| grip gate, rest_min_s | 0.0, 0.25 s | `m4_config.py:25,28` |
| logit eps: m4, m1 / final | 1e-6 / 1e-12 | `m4_models.py:39`; `m1_models.py:43`; `final_infer.py:36` |
| sigmoid clip: m4, final calibrator | ±40 / ±40 (`logit_clip`) | `m4_models.py:43`; `final_model.json` |
| isotonic_sym output clip | [0.01, 0.99] | `m4_models.py:311,327` |
| final p_clip | 1e-4 | `final_model.json` |

---

## 7. Open issues for the orchestrator

1. **The m4, PHYS6 and yaw exports do not exist.** Running `fit_production()` on all persons fits the inner cross-fitting and the session aggregate, which takes minutes. Its chosen `seg_cfg`, calibrator and s₁ and b must be read from the result.
2. **The yaw reference needs a k-aggregation.** Mean P (the harness's k-curve) is honest about what was evaluated. Fitting tau_k like `final_export._taus` would be a new procedure. The proposal is mean P, labelled "not calibrated for k ≥ 2".
3. **PHYS6 option `ampnorm`.** If the all-persons fit chooses it, which is unlikely (LOSO folds 20/20 `asinh`), the export must carry four more even features.
4. **The final CNN on detector windows is unvalidated.** So are tau_k on session attempts and the stopping thresholds for any model but m4: lane prod has not run. The page must say so in the code and in the debug panel.
5. **Training and production differ slightly.** Lane m4's detector was trained on float32 streams. The oracle windows used the session's calibration magnitude and the baseline-gravity fallback, which production lacks (README §13, "deferred"). Both are negligible, but they are real differences.
6. **Sign check.** Production uses the engine prior with no device check (docs/08 D23 had one). A misdetected engine shows up as a grip refusal on a correct hold (§1.2). The UI could say "turn the screen towards your face; if this persists, the browser may not be supported".
7. **Python offline `stream_end` against live.** A recorded fixture that ends right after its last attempt has fewer frozen attempts in live mode than offline. The parity test must use the mode that matches.

---

## 8. Changes this analysis calls for in `app/src/predict/types.ts`

These are proposals only. `types.ts` is not edited by this task.

1. `CalibratorJson`:
   - add `{method: 'isotonic_sym', x_thresholds, y_thresholds, x_min, x_max, p_lo, p_hi}`, evaluated as interp1d on |z| and reflected (§2.11), instead of `np.interp(z, x, y)` on a symmetrised table;
   - add `{method: 'nested', chosen, inner}` for m4.
2. `AggregationSpec`:
   - add `{kind: 'sum_over_k_pow', alpha}` for PHYS6, P_k = σ(Σz/k^α);
   - add `{kind: 'mean_p'}` for the yaw reference;
   - make the logit eps per model (`logitEps` 1e-6 or 1e-12) part of `ModelSpec`, or of the aggregation.
3. `DetectorBundle`:
   - add `lowpassSos`, `lowpassZi` and `lowpassPadlen` (§1.7), `segCfg`, and the constants of §4.2;
   - `segmentClassifier.center`/`scale` are `med`/`iqr` and `clip` is 5; for 'robust', `fill` is null.
4. `ModelAttemptOutput`: add `available: boolean` (PHYS6 NaN, CNN NaN), or exclude such attempts per model.
5. `StreamPrediction` stays as it is. The fixtures carry the per-model `{z_raw, p, z}` of §4.6 in a separate `ReferenceDump` test type.
