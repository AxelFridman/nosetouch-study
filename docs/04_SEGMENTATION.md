# 04 · Movement segmentation: detecting when the movement starts and ends

Every trial records a long raw window: settling, a hold-still at the extended arm, a random foreperiod, the GO beep, the movement, a hold at the nose, a post-roll and, since protocol 2.0.0, the return of the arm to arm's length. Somewhere inside it is the movement itself. This document explains how the app finds it **online** (while recording, to drive the beeps and quality checks) and how the analysis should find it **offline** (non-causally, as the ground truth for features).

Two rules govern everything below:

1. **The app never crops.** The online boundaries are stored as indices next to the full raw stream ([02 Data schema](02_DATA_SCHEMA.md)), so any analysis can re-segment with a better method later.
2. **The online detector is for cues and QA, not for science.** Timing features (reaction time, duration, time to peak) must come from offline re-segmentation ([§7](#7-offline-segmentation-for-analysis)).

Related: [01 Protocol](01_PROTOCOL.md) (the trial timeline the participant experiences), [05 Modeling plan](05_MODELING_PLAN.md) (what the segments are used for).

---

## 1. Signals

| Signal | Definition | Why |
|---|---|---|
| Angular speed \|ω\| (primary) | Euclidean norm of `rotationRate` (deg/s) | Independent of how the phone is gripped and of which hand holds it. The forearm rotates about 50–60° during a hand-to-nose movement, so \|ω\| is large even for slow trials (≥ 40 deg/s peak) ([segmentation research](https://pmc.ncbi.nlm.nih.gov/articles/PMC6427602/)). Wrist-IMU studies use it as the standard segmentation signal ([Carpinella 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3998062/), [Dotti 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435557/)). |
| Linear acceleration \|a\| (secondary) | Norm of `acceleration` when the browser provides it (`acc_source = 'linear'`), else a high-pass of `accelerationIncludingGravity` (`'highpass'`) | Catches movements with little rotation (a phone translated without turning). Weak in slow trials: 0.03–0.07 g, close to Chrome's 0.1 m/s² rounding step ([Chromium](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)). |
| Euler angles | **Not used.** | `deviceorientation` angles hit gimbal lock exactly at the most common start pose (phone upright, screen toward the participant, beta ≈ 90°), fire only on ≥ 0.1° changes on Chrome, and have an arbitrary heading ([03 Sensors](03_SENSORS_AND_PLATFORMS.md)). |

Both magnitudes are smoothed with a causal exponential moving average (time constant `smoothing_ms` = 40 ms) that uses the actual time between samples, because browser sampling is jittery.

At rest with the arm extended, the signals are dominated by physiological tremor (about 1–2 deg/s, 0.05–0.3 m/s²) and arm sway, not by sensor noise ([Elble 2005](https://smithengineering.queensu.ca/mme/faculty/deluzio/jam/files/Elble.pdf)). The thresholds below sit 10× or more above that.

## 2. The trial timeline

```
start ─ [awaitReturn: the arm goes back out] ─ hold still (still_hold_ms) ─ [grip gate] ─ foreperiod ─ GO ─ RT ─ movement ─ end_sustain ─ double beep ─ post_roll (≥ 500 ms) ─ return of the arm, still again ─ stop
                                               ↑ still criterion first met               ↑ markGo          ↑ onset (back-dated)   ↑ end (back-dated) ↑ end confirmed                 ↑ return_onset_ms ↑ return_end_ms
```

A trial starts with the block-intro tap (the first trial of a block) or, in the hands-free flow of protocol 2.0.0, by itself right after the previous trial, without the ready beep. The participant extends the arm and holds still. Once the hold-still criterion has held for `still_hold_ms` (and the gates of §4.1 pass), the controller waits a random foreperiod (so GO cannot be anticipated), beeps and arms the onset detector. After the movement the participant holds the phone at the nose until the double beep, which means "arm out again". Recording continues for at least `post_roll_ms` and then through the return of the arm until the arm is still again, or until `return_cap_ms` after the end was confirmed (§4.1).

```mermaid
stateDiagram-v2
    [*] --> waiting_still
    waiting_still --> waiting_still: a gate holds GO back (awaitReturn, wrong grip)
    waiting_still --> foreperiod: hold-still met for still_hold_ms, gates pass
    foreperiod --> waiting_still: still_lost (silent) / false_start (error beep)
    foreperiod --> armed: markGo (after the random foreperiod)
    armed --> moving: onset (sustained above threshold, back-dated)
    armed --> failed: no_movement (max_wait_ms)
    moving --> post_roll: end (quiet for end_sustain_ms, back-dated; double beep)
    moving --> failed: too_long (max_movement_ms)
    post_roll --> returning: post_roll_ms elapsed
    returning --> complete: return movement, then still for still_hold_ms (or return_cap_ms)
    complete --> [*]
    failed --> [*]
```

## 3. Algorithm `online_v1`

The exact, normative specification is the doc comment at the top of [`app/src/core/segmentation.ts`](../app/src/core/segmentation.ts). The Python port in [`analysis/motionlab/segment.py`](../analysis/motionlab/segment.py) follows it to the index. In prose:

1. **Smoothing.** EMA(x; τ): S₀ = x₀, Sᵢ = Sᵢ₋₁ + kᵢ(xᵢ − Sᵢ₋₁) with kᵢ = 1 − exp(−Δtᵢ/τ). A missing sample value reuses the previous raw value.
2. **Hold-still.** `C_still` = smoothed \|ω\| < `still_gyro_dps` **and** smoothed \|a\| < `still_acc_ms2`. When it has held continuously for `still_hold_ms`, the detector emits `still` and starts a **baseline buffer** of the smoothed values (and the raw gravity vector).
3. **Before GO.** Any sample that breaks `C_still` ends the hold and the detector goes back to waiting, so a premature movement never contaminates the baseline. The segmenter labels the break `false_start` when the motion is already above the (per-condition) onset floors, else `still_lost`. Since protocol 2.0.0 the trial controller decides what counts (§4.1): a loss of stillness while GO is pending is a counted false start (error beep, `timing.false_starts` + 1) only if the smoothed motion exceeds 30 deg/s or 0.6 m/s² at the loss or within the following 300 ms; smaller motion (posture sway) restarts the hold silently.
4. **GO.** The controller gives GO at the first sample after `still` + foreperiod. The thresholds are frozen from the baseline: `thr = max(floor, mean + onset_k_sd · sd)`, separately for \|ω\| (floor `onset_gyro_min_dps`) and \|a\| (floor `onset_acc_min_ms2`). The floors matter on Chrome, where rounding can make the baseline SD zero.
5. **Onset.** Motion counts once either signal stays above its threshold for `onset_sustain_ms`. The onset is then **back-dated** to the first sample after the last still sample, searching back to at most 300 ms before GO. This removes most of the delay that any threshold introduces.
6. **End.** The end candidate starts when both signals are below the end thresholds (`end_gyro_dps`, `end_acc_ms2`) at least `min_movement_ms` after onset. It is confirmed when that quiet holds for `end_sustain_ms`, and the end is **back-dated** to the start of the quiet run. The end beep plays at confirmation.
7. **Failures.** No onset within `max_wait_ms` of GO gives `no_movement`; a movement longer than `max_movement_ms` gives `too_long`. Both are saved with `outcome.accepted = false`, and the participant repeats the trial.

## 4. Parameters

Defaults live in `DEFAULT_SEGMENTATION_PARAMS` ([`constants.ts`](../app/src/core/constants.ts)); `analysis/motionlab/conventions.py` mirrors them (drift test `analysis/tests/test_conventions.py`). Every trial stores the exact parameters it used in `segmentation.params`, after the per-condition overrides. Protocol 2.0.0 (app 0.2.0) loosened the hold-still and the end so that a hands-free trial takes about 5–6 s (spec v2 §4). The protocol 1.0.0 values are kept for comparison; the pilot's records carry them.

| Parameter | Default (2.0.0) | 1.0.0 | Meaning |
|---|---|---|---|
| `smoothing_ms` | 40 | 40 | EMA time constant for both magnitudes |
| `gravity_tau_ms` | 600 | 600 | Gravity estimate for the `highpass` fallback |
| `still_gyro_dps`, `still_acc_ms2` | 20, 0.5 | 15, 0.35 | Hold-still thresholds |
| `still_hold_ms` | 400 | 600 | Continuous stillness required before GO is scheduled |
| `foreperiod_min_ms`, `foreperiod_max_ms` | 400, 900 | 500, 1200 | Random delay between stillness and GO (seeded per attempt) |
| `onset_k_sd` | 6 | 6 | Baseline-adaptive threshold multiplier |
| `onset_gyro_min_dps`, `onset_acc_min_ms2` | 30, 0.6 | 30, 0.6 | Floors of the onset thresholds |
| `onset_sustain_ms` | 60 | 60 | Suprathreshold duration required for onset |
| `end_gyro_dps`, `end_acc_ms2` | 30, 0.6 | 20, 0.45 | End thresholds |
| `end_sustain_ms` | 250 | 300 | Quiet duration required to confirm the end |
| `min_movement_ms` | 150 | 150 | No end can be declared earlier than this after onset |
| `max_wait_ms` | 3000 | 6000 | GO without movement → `no_movement` |
| `max_movement_ms` | 6000 | 8000 | Movement still going → `too_long` |
| `post_roll_ms` | 500 | 1000 | Recording after the end is confirmed (2.0.0: the minimum; the return of the arm follows) |
| `return_cap_ms` | 3000 | – | The recording stops at the latest this long after the end was confirmed. Optional: 1.0.0 records lack it and end after `post_roll_ms` |

**Per-condition overrides** (`resolveParams` in [`protocol.ts`](../app/src/core/protocol.ts)). Each override moves a parameter in one direction only (floors and end thresholds can only go down, sustain times and limits only up), so combinations compose safely:

| Condition | Overrides | Reason |
|---|---|---|
| speed `slow` | onset floors 25 deg/s and 0.55 m/s² (1.0.0: 15 and 0.3), end thresholds 10 deg/s and 0.25 m/s², `end_sustain_ms` 450, `max_movement_ms` 12000 | Slow movements have low peaks and long tails. The onset floors stay above the hold-still thresholds: at or below them, every loss of stillness counted as a false start (7 of the pilot's 8) |
| speed `fast` | `onset_sustain_ms` 40 | Fast movements cross the threshold quickly |
| variant `pause` | `end_sustain_ms` 1800, `max_movement_ms` 12000 | A one-second pause mid-way must not end the trial |
| variant `jerky` | `end_sustain_ms` 600 | Stop-and-go movements have short quiet gaps |

**Why these timing margins.** The hold plus the foreperiod guarantee 0.8–1.3 s of stillness before GO (1.0.0: 1.1–1.8 s), still a clean baseline for the adaptive thresholds and an anchor for gravity propagation. The rest at the nose (`post_roll_ms`, the participant waits for the double beep) and the stillness after the return keep rest windows at both ends, which offline segmentation and zero-velocity corrections need ([research: hold ≥ 1 s at both ends](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823748/)).

### 4.1 Around the segmenter: gates, false starts and the return of the arm (protocol 2.0.0)

The segmenter (`segmentation.ts`, ported to `segment.py`) is unchanged by protocol 2.0.0 apart from its parameters. Everything below is decided by the trial controller ([`trialController.ts`](../app/src/core/trialController.ts)) on the same smoothed signals (`segmentation.MotionSignals`, which keep running after the end), so re-segmenting a record with `online_v1` still reproduces its onset and end.

- **Return movement.** Smoothed \|ω\| > 45 deg/s or \|a\| > 1.5 m/s², continuously for at least 100 ms (`RETURN_GYRO_DPS`, `RETURN_ACC_MS2`, `RETURN_SUSTAIN_MS`).
- **Recording the return.** After the end is confirmed the controller records at least `post_roll_ms`, then (phase `returning`) until a return movement that started after the end has been followed by stillness (`still_gyro_dps`, `still_acc_ms2`) held for `still_hold_ms`, or until `return_cap_ms` after the end. `timing.return_onset_ms` is the first sample of that movement and `timing.return_end_ms` the first sample of the stillness after it; both are null when not observed. In simulation the return is recorded on every complete trial, and the recorded return onset lags the true start of the return by 94 ms (median; 151 ms at the 95th percentile; [`segmentation_eval.md`](../app/scripts/out/segmentation_eval.md)). An abort while the trial is `returning` (pause, page hidden) keeps it `complete` with the warning `return_cut_short`: the movement itself was recorded.
- **`awaitReturn`.** When a trial ended without its return being seen (`trialController.returnMissed`), the next chained trial starts with `awaitReturn`: stillness counts only after a return movement in that trial (message `return_arm`), so GO is never given with the phone still at the nose.
- **Grip gate** (`gripGate`: hands-free chained trials, never `free`). At `still`, the mean acceleration including gravity of the last 300 ms is classified like `detected.rotation_start` (35° cone, gravity sign from `calibrationForDetection`). A confident match to another grip than the instructed one withholds GO (message `wrong_grip`; the screen shows the grip picture); `unknown` never blocks. A gravity sign that no measured upright hold has confirmed (the engine prior, or none) could be wrong, and a wrong sign turns upright into upside-down and top-left into top-right. With an unconfirmed sign a mismatch therefore blocks only when it survives flipping the sign, and `detected.orientation_match` is then null rather than false. The hands-free session auto-rejects only `orientation_match === false`.
- **False starts.** See §3, step 3. A loss of stillness while GO is pending counts (error beep) when the motion exceeds 30 deg/s or 0.6 m/s² at the loss or within `FALSE_START_WATCH_MS` = 300 ms after it, so an anticipation that starts gently gets its error beep up to 300 ms late. A loss while a gate holds GO back is always silent.
- **Ready beep.** Chained trials skip it (`readyCue: false`); the first trial after a block-intro tap keeps it.

## 5. How well it works (simulation)

[`app/scripts/evalSegmentation.ts`](../app/scripts/evalSegmentation.ts) (`npm run eval:seg -- --n 50`) replays simulated trials with known ground truth through the real `TrialController`. It compares `timing.onset_ms` with the moment the simulated phone starts moving, and `timing.end_ms` with the moment it reaches the nose. The setup:

- 50 trials per speed × variant × platform cell (`--n 50`; a plain `npm run eval:seg` uses 30), 3,300 normal trials per parameter set (11 variants since protocol 2.0.0 added `tilted_touch`);
- Android-like data quantised like Chromium;
- the same trials replayed for every parameter set, so differences are paired.

The full tables are in [`app/scripts/out/segmentation_eval.md`](../app/scripts/out/segmentation_eval.md).

| Parameters | Trials | Onset error, median [IQR] | Onset \|error\|, 95th pct | End error, median | \|End error\| > 300 ms | Detected |
|---|---|---|---|---|---|---|
| 2.0.0 (default) | all 3,300 | +38 ms [19, 107] | 338 ms | +72 ms | 0.1% | 99.97% (1 `no_movement`) |
| 2.0.0 | natural | +39 ms | 79 ms | +69 ms | 0% | 100% |
| 2.0.0 | fast | +17 ms | 30 ms | +123 ms | 0.1% | 100% |
| 2.0.0 | slow | +171 ms | 483 ms | −18 ms | 0.3% | 99.9% |
| 1.0.0 | all 3,300 | +31 ms [16, 73] | 184 ms | +84 ms | 0.2% | 100% |
| 1.0.0 | slow | +107 ms | 247 ms | −17 ms | 0.5% | 100% |

Positive errors mean the detector is late. "2.0.0" is the `default` set (the current `DEFAULT_SEGMENTATION_PARAMS`), "1.0.0" the `v1` set (the protocol 1.0.0 values of §4, slow onset floors 15 / 0.3), replayed on the same simulated trials. Further results:

- **2.0.0 against 1.0.0, paired.** GO comes 0.4 s earlier (median 1.07 s instead of 1.46 s after the start of the trial) and the end cue 48 ms earlier. The recording keeps 0.75 s more after the phone reaches the nose (the return of the arm), so a recorded trial is 0.35 s longer (median 4.8 s), while the hands-free flow removes the taps and screens between trials.
- **Onset accuracy is worse, and this is reported, not tuned.** The 95th percentile of the onset error rose by 155 ms (184 → 338 ms), far more than the 40 ms the v2 spec allows without a report. Almost all of it comes from slow trials (95th percentile 247 → 483 ms), because their onset floors were raised from 15 / 0.3 to 25 / 0.55 on purpose (§4: the old floors made every sway a false start). The raw streams allow re-segmentation, and timing features come from offline segmentation anyway (§7). Watch the share of `no_movement` among real slow trials: 1 in 3,300 in simulation.
- **Failures.** Every simulated false start (400 of 400) was counted before GO and the trial still completed. Every no-move trial ended as `no_movement`. With twice the simulated tremor, the hold-still was still met at its earliest possible time (about 0.41 s).
- **Weak spot.** Slow trials: late onsets and early ends come from the absolute thresholds meeting the long, low tails of a slow minimum-jerk profile. This is the expected bias of any threshold method (§6) and the main reason timing features must be computed offline.

**Caveat.** These are sanity checks against a model, not real-world accuracy. Real movements have sub-movements, hesitations and grip adjustments the simulator does not produce ([§9](#9-validation-and-tuning-plan)).

## 6. Expected kinematics and threshold bias

Minimum-jerk reaching ([Flash & Hogan 1985](https://www.socsci.ru.nl/wilberth/physics/smooth.html)) gives a bell-shaped speed profile, v/v_peak = 16τ²(1−τ)² with τ = t/D, so peak speed is 1.875 × the mean speed. For a 0.45 m path with 60° of rotation:

| Duration D | Peak hand speed | Peak linear acceleration | Peak angular speed |
|---|---|---|---|
| 0.35 s (fast) | 2.4 m/s | 21 m/s² | 321 deg/s |
| 0.7 s (natural) | 1.2 m/s | 5.3 m/s² | 161 deg/s |
| 2 s (slow) | 0.42 m/s | 0.65 m/s² | 56 deg/s |
| 3 s (very slow) | 0.28 m/s | 0.29 m/s² | 38 deg/s |

Healthy finger-to-nose movements take about 0.6 s one way ([Chen 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8072355/)). Fast or odd trials can exceed 4 g and 500 deg/s, near the ranges of some phone sensors, so the quality checks flag clipping (\|ω\| ≈ 950 deg/s, \|accg\| ≈ 3.5 g).

Threshold bias:

- **Relative threshold.** A threshold at p% of the peak cuts a fixed fraction of the duration off each end: 5% → 5.9% of D, i.e. 18 ms at D = 0.3 s but 119 ms at D = 2 s.
- **Absolute threshold.** These bias slow movements far more ([Botzer & Karniel 2009](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.20.002.2009/full); [Blinch 2024](https://pubmed.ncbi.nlm.nih.gov/37626277)).
- **Quantisation.** At 60 Hz the onset quantisation SD is only 4.8 ms (T/√12). The method's bias dominates the timing error, not the sampling rate.

## 7. Offline segmentation for analysis

`motionlab.segment` provides:

- `rerun_online_v1(record)`: re-runs `online_v1` on a stored trial with its stored parameters and GO time. On the synthetic dataset it reproduces the app's onset and end indices on **683/683 trials**, so the port can be trusted. A CI fixture keeps the TypeScript and Python versions in sync.
- `offline_v1`: zero-phase smoothing, percentage-of-peak thresholds with a baseline-noise floor, and a sustained end. It uses the whole window, so it is not biased by causality.

Recommended practice for the stage-2 analysis:

1. Compute several boundary estimates per trial:
   - relative thresholds at 5%, 11% and 25% of peak \|ω\| ([Dotti 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435557/), [Carpinella 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3998062/));
   - baseline μ + kσ with backtracking;
   - the MACC fit ([Botzer & Karniel 2009](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/neuro.20.002.2009/full));
   - a change-point estimate (PELT, [Truong et al.](https://arxiv.org/abs/1801.00718)).
2. Combine them MSI-style ([Schot 2010](https://personal.fbw.vu.nl/~jsmeets/2010/Robust_Segmentation-JNM10.pdf)) or pick one method and keep it fixed. **Use identical segmentation for both hands**, otherwise segmentation differences become spurious hand differences.
3. Flag trials where the methods disagree by more than 100 ms, or where \|ω\| shows several movement units (peaks ≥ 10–20 deg/s of prominence, ≥ 150 ms apart). These are "odd" movements whatever the instruction said.
4. Bin trials by **measured** duration, not by the instructed speed, when analysing error against speed.

## 8. Failure modes

| Situation | What happens | Data consequence |
|---|---|---|
| Participant moves before GO | Error beep, "Wait for the beep", the hold restarts | `timing.false_starts` counts it; the stored baseline is from the final clean hold |
| Participant never moves | `no_movement` after 6 s | Saved as not accepted; the trial is repeated |
| Pause or hesitation mid-way | Normal trials may end early at the pause; the `pause` variant has an 1800 ms end sustain | Offline segmentation decides; flag multi-unit movements |
| Slow movement with little rotation | Lower onset floors (slow override); onset late, end early by a few hundred ms | Use offline boundaries for timing |
| Chrome rounding | Baseline SD can be 0; the floors keep thresholds sensible | None |
| Page hidden, sensors stall | The UI aborts the trial (`interrupted`) | Saved as not accepted with a `ui_abort:*` warning |
| Nose touches the screen | A touch event is logged (`first_touch_ms`) | A free contact timestamp to validate the end detector (§9) |

## 9. Validation and tuning plan

1. **Simulation (done).** `npm run eval:seg -- --n 50` (§5). The TypeScript-to-Python parity check runs in CI.
2. **Contact timestamps.** In `*_toward` trials the nose usually touches the screen, which produces a touch event. Compare `first_touch_ms` with `end_ms` across the pilot; it is a real ground truth for the end that costs nothing.
3. **Video subset.** Film 2–3 pilot participants at 60 fps with a visible GO beep (an LED or the phone screen in view). Annotate onset and contact by hand for 50–100 trials and compute the online and offline errors.
4. **Pilot tuning, in this order:**
   - Check that participants can satisfy the hold-still. Inspect `timing.still_ms` and the false-start count, then decide whether to adopt the stricter 10 deg/s / 0.25 m/s² thresholds (§5).
   - Check slow trials for premature end beeps. If they happen, raise the slow `end_sustain_ms`.
   - Re-run `npm run eval:seg -- --n 50` after any change, and bump nothing silently: parameters are stored per trial.
5. **Report** onset and end errors (median, IQR, 95th percentile) by speed and platform in the stage-2 write-up.

## 10. Reproduce

```bash
cd app
npm run eval:seg -- --n 50            # ~1 min; writes scripts/out/segmentation_eval.md (the run quoted in §5)
npx tsx scripts/evalSegmentation.ts --n 50 --sets all --out scripts/out/segmentation_eval_all.md   # adds the exploration sets (v2+v1slow, v2+wait6000, v2+cap5000); --out keeps the §5 report
npm run synth -- --participants 12 --preset street --seed 42 --out ../analysis/data/synthetic --validate
cd ../analysis
.venv/Scripts/python.exe scripts/qa_report.py data/synthetic    # includes the online_v1 port agreement
```

## 11. Open questions for the PI

1. Is the online detector's slow-trial bias acceptable for the stage-3 demo, or should predict mode re-segment offline before classifying? (Recommended: re-segment; it costs a few milliseconds.)
2. Should a separate "hold at the nose" be required (a longer `end_sustain_ms`), at the cost of about 0.2 s per trial, to give cleaner end windows?
3. Is a video-annotated validation subset worth the effort, or are the screen-touch timestamps enough?
