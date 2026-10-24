# Modeling plan (Stage 2)

This document describes how the Stage-1 recordings become a validated left/right classifier, an error analysis by subgroup, and the model that Stage 3 hands to the PI.

Related: [Protocol](01_PROTOCOL.md) · [Segmentation](04_SEGMENTATION.md) · [Stage-3 predictor](08_STAGE3_PREDICTOR.md) · data contract [records.schema.json](../schema/records.schema.json) · mirror code [mirror.ts](../app/src/core/mirror.ts) · Python kit `analysis/motionlab`.

**Legend.** **Rec.** marks a recommendation, which is a choice and can be changed. **Hyp.** marks a hypothesis to test on data. Unmarked statements are sourced facts or derivations. The derivations in §3–4 were also checked numerically while this plan was written.

**Conventions.** The device frame follows W3C: x points to the right edge, y to the top edge, z out of the screen. `rr_alpha`, `rr_beta` and `rr_gamma` are the rotation rates about device x, y and z ([W3C CRD, 12 Feb 2025](https://www.w3.org/TR/orientation-event/), changed by [PR #43](https://github.com/w3c/deviceorientation/pull/43)). MDN's "alpha = Z" page is outdated. ĝ is the unit vector pointing **up**, i.e. the direction of the normalised `accg` vector at rest. ω is the angular velocity, f the specific force (`accg`), and a the linear acceleration.

---

## 1. Task framing

| Item | Definition |
|---|---|
| Unit of prediction | One trial: one movement from the extended arm to the nose. |
| Label | `condition.hand` ∈ {`left`, `right`}, the instructed hand (`condition.hand_source` `instructed` or absent). The free-choice trial of a street visit carries the hand the participant reported (`hand_source = reported`) and is kept apart (below). The label is never inferred from handedness: the side people use the phone on matches their dominant hand only about 70% of the time ([Seidman 2013](https://sciencedaily.com/releases/2013/05/130516161655.htm), survey, low confidence). |
| Model input | Sensor streams, calibration and segmentation only. The model never sees `condition` (hand, orientation, speed, variant). |
| Use case | An unseen person performs a single trial on an unknown phone (iOS or Android, any model), with an unknown grip, any speed, and possibly an odd movement. |
| Primary metric | Trial-level accuracy on unseen participants, with a participant-level bootstrap 95% CI. |
| Secondary metrics | Balanced accuracy, ROC AUC, log-loss, calibration, coverage vs accuracy under abstention, and the per-participant accuracy distribution. |

Which data each analysis uses (**Rec.**):

| Set | Filter |
|---|---|
| Primary analysis | `simulated` false; `outcome.status` `complete`; `outcome.accepted` true; `outcome.participant_flag` null; `quality.motion_hz` ≥ 50; `quality.gaps_over_50ms` = 0; page not hidden during the movement (rule T4: `quality.hidden_during` false, or a complete trial with the warning `return_cut_short`); `quality.rotation_available` true; gravity sign resolved from the calibration (`guided_v1` or `implicit_v1`), the trial votes or the engine, never `default` (rule S3 of [02 §12](02_DATA_SCHEMA.md#12-data-quality-and-exclusion-rules)); `condition.hand_source` is `instructed` or absent. |
| Natural use | Choice trials (`hand_source = reported`), accepted copies only: an unanswered or "Not sure" copy (`outcome.note` `hand_not_reported` / `hand_unsure`) keeps the plan's placeholder hand. Never trained on; scored out of fold; reported separately ([09 D4](09_STAGE2_3_PLAN.md#42-labels-and-exclusions)). |
| Sensitivity | The primary set plus complete attempts that were not accepted, plus flagged trials. |
| Pipeline tests only | `npm run synth` output (`simulated: true`). It is never mixed into real-data results. |

With `review_mode = auto` (every street session), `accepted` means "completed and not redone", not "confirmed by the participant"; label noise is screened per [09 D6](09_STAGE2_3_PLAN.md#42-labels-and-exclusions). `motionlab.io` already loads `hand_source` (absent → `instructed`); dropping reported trials from the modelling table by default is part of milestone M1 ([09 §8](09_STAGE2_3_PLAN.md#8-milestones-and-acceptance-criteria)).

Browsers deliver about 60 Hz on both platforms. Chromium uses a 60 Hz pump ([source](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_sensor_event_pump.h)) and WebKit a 1/60 s timer ([source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.h)). A rate below 50 Hz or a gap over 50 ms therefore means a broken recording.

## 2. Preprocessing pipeline

Raw records are never modified. Each step is versioned (e.g. `norm_v2`), and the version string is stored with every derived table and model. Stage 3 can then refuse a model built with a different pipeline.

| # | Step | Output | Why |
|---|---|---|---|
| 0 | Load records, validate them against the schema, apply the §1 filters | trial table | Contract drift fails loudly. |
| 1 | Normalise conventions (§2.1) | f in m/s², ω in deg/s, W3C signs | Platforms differ, so conventions are measured per device. |
| 2 | Resample to a uniform grid: 60 Hz for PWA data, 100 Hz for the phyphox cohort (§2.2) | uniform arrays | Browser samples are jittered snapshots. |
| 3 | Estimate gravity by gyro propagation from the rest windows (§2.3) | ĝ(t), a(t) = f − g | Accelerometer-corrected filters fail during the movement. |
| 4 | Segmentation: online_v1 indices plus an offline consensus (§2.4) | onset, end, flags | The offline consensus is the analysis truth. |
| 5 | Features (§4) and/or sequences | X | |

### 2.1 Normalisation (`norm_v2`)

`norm_v2` (protocol 2.0.0) replaced `norm_v1` because street sessions carry only an implicit calibration: no flat pose, no tilt step ([03 §4](03_SENSORS_AND_PLATFORMS.md#4-calibration-procedure)). Its three changes are marked below; the full rules are in [03 §5](03_SENSORS_AND_PLATFORMS.md#5-normalisation-layer-for-analysis) and [analysis/README.md](../analysis/README.md#normalisation-rules-norm_v2).

| Quantity | Raw fields | Normalised | Rule |
|---|---|---|---|
| Specific force | `accg_x/y/z` | f = s · accg | s = the `gravity_sign` of a guided calibration (+1 = W3C, −1 = inverted), unless the trial votes clearly contradict it (≥ 5 votes, ≥ 80 % against): then the vote sign is used and the session flagged. **norm_v2:** otherwise the engine expectation (`device.engine` `webkit` → −1, otherwise +1) when a confident vote of the trial baselines confirms it (source `engine_vote`, not flagged: the normal protocol 2.0.0 case); then a confident vote alone, the engine alone, the implicit calibration's sign, and +1, each flagged ([03 §5.2](03_SENSORS_AND_PLATFORMS.md#52-gravity-sign-resolution)). |
| Browser linear acceleration | `acc_x/y/z` | s · acc | QA only. When a phone has no linear-acceleration sensor, Chrome falls back to a crude high-pass filter ([source](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/linear_acceleration_fusion_algorithm_using_accelerometer.cc)). |
| Angular velocity | `rr_alpha/beta/gamma` (deg/s) | ω = (x, y, z) in deg/s | Depends on `calibration.axis_check.verdict`: `w3c` → (alpha, beta, gamma); `legacy` → (beta, gamma, alpha); `*_inverted` → negate. If `units` is `rad_per_s`, multiply by 180/π. **norm_v2:** if the verdict is `unclear` or absent (every implicit calibration), the mapping whose gravity closure over the session's trials is decisive (< 3°, < half the runner-up, ≥ 8 trials); otherwise assume `w3c` and flag the session (`axis_unverified`). |
| Gyro bias | `calibration.gyro_bias_dps` | subtract, with the same mapping | Taken from the flat-on-table pose, which is true rest, but only when that window was still (`calibration.flat.still`) and the bias norm is below 3 deg/s. **norm_v2:** otherwise nothing is subtracted; a hand-held pre-GO mean is tremor, not bias (on the pilot, subtracting it worsened the median gravity closure from 1.6° to 2.7°). `norm_v1` used the mean rate of the 400 ms before GO when its norm was below 3 deg/s. **Deploy pipeline (`feat_d1`):** street sessions (`implicit_v1`) have no flat window, so the 400 ms before GO is the only source; every deployed model subtracts it when its norm is below 3 deg/s, in training (`prepare_trial(remove_gyro_bias='pre_go')`) and in the predictor alike ([09 D9](09_STAGE2_3_PLAN.md#43-preprocessing-and-segmentation)). Ablation A5 shows what that costs against the research default. |
| rotationRate axes in the deploy pipeline | as above | W3C assumed | Every current engine follows W3C, and a single predict-mode attempt cannot establish a mapping. The predictor assumes W3C and its gravity-closure gate (G3) catches a mismatch; on training data a per-session closure check flags sessions whose closure favours another mapping ([09 D8](09_STAGE2_3_PLAN.md#43-preprocessing-and-segmentation)). The implemented `norm_v2` rule above also remaps a session when its closure is decisive; on the pilot and the synthetic street data it confirmed W3C. |
| Time | `t` (handler time) | seconds | `t` is on the same clock as the cues. `ts` is a QA cross-check. Never use `interval_raw`: iOS reports it in seconds, Chrome reports a constant 16 and Firefox a constant 100 ([WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/DeviceMotionClientIOS.mm), [Chromium source](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc)). |
| Repeated samples | | drop | On iOS, drop exact 9-channel repeats: they are duplicated timer snapshots, and iOS values are unrounded. On Chrome, drop a repeat only while ‖ω‖ > 20 deg/s. **Rec.** |
| iOS precision | WebKit values are unrounded | round raw `acc`/`accg` to 0.1 m/s² and `rr` to 0.1 deg/s | **Rec.** Evaluated as ablation A3 (`train_baseline.py --quantize ios`); off by default until A3 decides. Chrome rounds to this grid ([platform_sensor_util.h](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)). Without the rounding, a model can learn the platform from value granularity. |

Why the sign step exists: WebKit forwards Core Motion's sign, so a face-up iPhone reads `accg_z` ≈ −9.81, while Android reads +9.81. rotationRate signs agree across the two platforms. Sources: [WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm), [W3C list, 2014](https://lists.w3.org/Archives/Public/public-geolocation/2014Jul/0024.html), [re-measured on iPadOS in 2026](https://github.com/soroushdty/gaitscope/issues/96). Protocol 1.0.0 measured the sign from the flat pose. Protocol 2.0.0 has no flat pose: it uses the engine expectation (derived from the user agent) only when the measured trial baselines confirm it, and flags every other case. §3.3 explains why the sign cannot be recovered from the movements themselves.

### 2.2 Resampling (**Rec.**)

- **PWA data.** Linearly interpolate onto a 60 Hz grid after removing repeats. At 60 Hz, movements of 0.25 s or longer keep at least 99% of their acceleration energy, because 99% of it lies below about 2.53/D Hz ([minimum-jerk profile](https://www.socsci.ru.nl/wilberth/physics/smooth.html); spectrum from the digest's calculation).
- **Deployed models.** Every model that ships is trained on `feat_d1` at 60 Hz ([09 D11](09_STAGE2_3_PLAN.md#43-preprocessing-and-segmentation)), the pipeline the browser runs.
- **phyphox cohort.** Each sensor has its own time buffer, and the gyroscope reports rad/s ([format](https://phyphox.org/docs/file-format/input/)). Apply a zero-phase low-pass at about 25 Hz, then resample to 100 Hz, the iPhone ceiling ([WWDC23](https://developer.apple.com/videos/play/wwdc2023/10179)). For comparisons, also build a "PWA emulation": 60 Hz sample-and-hold of the latest sample, plus 0.1-unit rounding. Record a flat pose in every phyphox session so the same sign normalisation applies.

### 2.3 Gravity and linear acceleration (`gravity_v1`)

During the movement, tangential and centripetal accelerations of about 0.3 to more than 2 g make accelerometer-corrected filters (Madgwick, Mahony, complementary) mis-estimate tilt unless their correction is gated off. Gyro propagation from the rest pose keeps the tilt error near 1° for movements of 2 s or less ([Madgwick docs](https://ahrs.readthedocs.io/en/latest/filters/madgwick.html), [Brändle 2025](https://arxiv.org/abs/2502.03681); error estimate from the digest).

1. **Rest windows.** "pre" is the last 300 ms or more before onset with ‖ω‖ < `still_gyro_dps`. "post" is the first 300 ms or more of the hold at the nose. g_pre and g_post are the mean of f over each window.
2. **Propagate.** Forward from onset, g_{k+1} = exp(−[ω_k Δt]×) g_k, i.e. rotate by ‖ω_k‖Δt about −ω̂_k (dg/dt = −ω × g). Propagate backward from the end in the same way with +ω.
3. **Blend.** g(t) = (1 − λ) g_fwd + λ g_bwd, with λ = (t − t_on)/(t_end − t_on). Rescale to ‖g‖ = `gravity_magnitude`, or 9.80665 if that is missing. ĝ = g/‖g‖.
4. **Linear acceleration.** a(t) = f(t) − g(t). Every degree of tilt error leaks about 0.17 m/s² into a.
5. **QA.** Closure error = the angle between g_fwd(t_end) and g_post. **Rec.**: flag trials above 3°, which point to a gyro bias or scale problem or to a rest window that was not still.
6. **Attitude.** R₀(t), the attitude relative to onset, comes from integrating ω. It puts vectors from different instants into one frame, D₀.

Velocity and displacement (ZUPT at both ends plus linear detrending) are computed only for movements of 1.5 s or less. The expected errors are about 1–4 cm up to 1 s and 6–43 cm at 2–3 s (digest Monte-Carlo; [Bai 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823748/)).

### 2.4 Segmentation

- online_v1 stored `segmentation.onset_index` and `end_index`, and drove the cues and QA. The analysis truth is an offline re-segmentation that runs several methods: ‖ω‖ relative thresholds, baseline μ + kσ with backtracking, MACC, and change points. A consensus picks the boundaries, and the pipeline flags disagreements over 100 ms and multi-peak movements ([04_SEGMENTATION.md](04_SEGMENTATION.md)).
- Both hands use the same method. Otherwise segmentation differences turn into spurious hand differences ([Melendez-Calderon 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7841375/)).
- The deployed model uses `offline_v1`, identically in Python and TypeScript ([09 D10](09_STAGE2_3_PLAN.md#43-preprocessing-and-segmentation); [08 §2.4](08_STAGE3_PREDICTOR.md#24-training-and-deployment-parity)). online_v1 with the old predict-mode parameters re-segments 0 of 18 pilot trials, and the protocol-2.0.0 natural parameters cut slow movements short ([09 §2.2](09_STAGE2_3_PLAN.md#22-the-predict-mode-parameters-of-08-24-cannot-re-segment-training-data)).
- Windows (09 D11): **W0** the pre-GO rest (gyro bias, the starting ĝ); **W1** the approach, onset → end; **W2** the hold at the nose (g_post); **W3** the return of the arm, recorded by the collector only. Integrals run over W1; sequences run over [onset − 0.5 s, end + 0.5 s].
- **Time reversal** (ω → −ω, a unchanged) flips every pseudoscalar, so a return that retraces the approach looks like the other hand. Never pool W1 and W3; ablation A10 tests a time-reversed or separately modelled return.

## 3. Mirror symmetry (left ↔ right)

### 3.1 Derivation

Frames: in the body frame B, x is the participant's right, y is up and z is backward (as in [sim/model.ts](../app/src/sim/model.ts)). D is the device frame. R(t) maps device coordinates to body coordinates.

```
f     = Rᵀ (p̈ + g_up)      accelerationIncludingGravity (W3C sign), g_up = (0, 9.81, 0) in B
a     = Rᵀ p̈               acceleration
[ω]×  = Rᵀ Ṙ               rotationRate as (x, y, z) = (rr_alpha, rr_beta, rr_gamma)

Mirror across the sagittal plane:  p' = M p,  R' = M R S,  M = S = diag(−1, 1, 1),  det R' = +1
f'    = R'ᵀ (M p̈ + g_up) = S Rᵀ M (M p̈ + g_up) = S Rᵀ (p̈ + M g_up) = S f        (M g_up = g_up)
[ω']× = R'ᵀ Ṙ' = S (Rᵀ Ṙ) S = S [ω]× S = [det(S) · S ω]×    ⇒    ω' = −S ω
```

The mirror image of a phone has the same top edge and screen normal, but a left-handed frame. S relabels its x axis so that a real, right-handed phone describes it. Because S acts on the device side and M on the body side, the result does not depend on R. **It holds for every grip and every movement.** Acceleration is a polar vector and reflects like a position. Angular velocity is a pseudovector and picks up an extra factor of det = −1.

### 3.2 Resulting rules

| Quantity | Fields | Mirrored |
|---|---|---|
| Acceleration, x | `acc_x`, `accg_x` | negated |
| Acceleration, y and z | `acc_y`, `acc_z`, `accg_y`, `accg_z` | unchanged |
| ω_x | `rr_alpha` | unchanged |
| ω_y, ω_z | `rr_beta`, `rr_gamma` | negated |
| Hand | `condition.hand` | `left` ↔ `right` |
| Orientation | `condition.orientation` | `top_left_toward` ↔ `top_right_toward`, `top_left_away` ↔ `top_right_away`; all others unchanged (`ORIENTATIONS[o].mirror`) |
| Euler angles, Z-X′-Y″ | `alpha`, `beta`, `gamma` | alpha′ = −alpha relative to the mirror plane; in absolute terms 2ψ − alpha, where ψ is the unknown facing direction. beta′ = beta, gamma′ = −gamma. |
| Quaternion (x, y, z, w) | `streams.quaternion` | (x, −y, −z, w) |
| Compass | `compass` | not physically valid, because Earth's field is not mirrored; do not use it |

- **Orientation labels.** M · R_grip(o) · S = R_grip(mirror(o)) holds for all nine grip matrices in `sim/model.ts`.
- **Euler angles.** The rule follows from M Rz(α) Rx(β) Ry(γ) M = Rz(−α) Rx(β) Ry(−γ). Near gimbal lock (beta = ±90°, which is the upright start pose) and at gamma = −90°, go through the rotation matrix, as `mirror.ts` does. Euler angles are metadata only.
- **Where to mirror.** `mirrorTrial` in `mirror.ts` works on raw field names and assumes the `w3c` mapping. In Python, mirror the normalised physical axes after §2.1, then recompute every derived quantity from the mirrored raw streams ([digest](https://www.w3.org/TR/orientation-event/)).

Published transforms disagree. OREBA flips acc x and gyro y, z ([paper](https://ar5iv.labs.arxiv.org/html/2007.15831)), as does Kyritsis et al. ([paper](https://arxiv.org/pdf/2010.07051)). OpenWatch (2026) flips acc X, gyro X and gyro Z, which violates the pseudovector rule ([paper](https://arxiv.org/html/2605.04791)). [Wang et al. 2026](https://arxiv.org/abs/2607.01376) (§IV-A3, eqs. 4–5) derives the same matrices as ours independently: diag(−1, 1, 1) for the accelerometer and diag(1, −1, −1) for the gyroscope. **Do not copy a transform from a paper; verify it (§3.5, item 4).**

### 3.3 What the iOS sign inversion means: parity

Raw iOS data are (−a, ω) relative to Android: the accelerometer is negated and the gyro is not. That is exactly the parity transformation P = −I applied to polar vectors, with pseudovectors unchanged, and −I = S · Rx(180°) has det = −1. Two consequences:

1. Every chiral feature (§4.3) flips sign on un-normalised iOS data, so **an iOS right-hand trial looks like an Android left-hand trial**. No grip change can undo this, because grips are proper rotations.
2. The flip cannot be detected from gyro–accelerometer consistency, because dĝ/dt = −ω × ĝ also holds for −ĝ. The axis check in [calibration.ts](../app/src/core/calibration.ts) and the gravity-closure check of `norm_v2` are independent of the accelerometer sign for exactly this reason. Only a known pose reveals the sign: in protocol 1.0.0 the flat calibration step, in protocol 2.0.0 the still hold before GO of the instructed `upright_toward` trials (the implicit calibration and the trial votes), checked against the browser engine (§2.1). Gravity-sign normalisation is therefore a prerequisite for every model.

### 3.4 Limits of the mirror model

- **Bodies and movements are not symmetric.** Dominant and non-dominant arms move differently ([Sainburg 2002](https://link.springer.com/doi/10.1007/s00221-001-0913-8)). Mirrored right-hand trials are therefore not real left-hand trials, and augmentation must always be evaluated on real data.
- **The gyroscope mirror is exact; the accelerometer mirror is approximate.** ω is the same at every point of a rigid phone, but the IMU sits off the phone's centre line, at x offset r_x. The real reading minus the augmented one is α′ × d + ω′ × (ω′ × d), with d = (2r_x, 0, 0). r_x is unknown per model and is less than half the phone's width. For r_x = 2 cm, my minimum-jerk estimate of this error is about 0.5 m/s² at natural speed (peak a ≈ 5 m/s²) and about 3 m/s² at fast speed (peak a ≈ 29 m/s²), roughly 10% in both cases.
- **Phones are not symmetric.** The camera bump and buttons can change how people grip the phone.

### 3.5 Uses

1. **Augmentation.** For every training trial, add its mirror: streams mirrored per §3.2, hand swapped, orientation mapped. The copies stay in the source participant's fold. Evaluate on real trials only, and report the ablation (§8.2, A1). Augmentation runs inside the estimator's `fit` (a `MirrorAugmented` wrapper, [09 D13](09_STAGE2_3_PLAN.md#44-models)), so every inner and outer validation fold holds real trials only. For antisymmetric models it is redundant: the loss on (x, y) equals the loss on (m(x), 1 − y).
2. **Antisymmetric models.** Define the logit as f(x) = g(x) − g(m(x)), where m is the mirror map and g is any model. This gives three properties:
   - p(R | m(x)) = 1 − p(R | x) exactly.
   - A mirror-symmetric input gets p = 0.5, which is a natural abstention.
   - Mirror-even cues such as speed, smoothness and tremor cannot decide on their own. The dominance shortcut (§8.1) is therefore removed by construction, at the price of ignoring real but confounded asymmetries.

   Implementations:
   - Logistic regression on odd (chiral) features with `fit_intercept=False` and `StandardScaler(with_mean=False)` (`models.make_logreg_chiral`).
   - Any scikit-learn model: a `MirrorSymmetrized(base)` wrapper evaluates g on x and on m(x) from a closed-form `features.feature_mirror(F)` (`chi_*` change sign; for `dev_acc_x_*`, `dev_gyro_y_*` and `dev_gyro_z_*` the mean changes sign and (min, max) becomes (−max, −min); `dev_up_*_x`, `dev_rot_y_deg` and `dev_rot_z_deg` change sign; basic and invariant features stay). A unit test checks that `feature_mirror(F)` equals `extract_features(..., mirror=True)`. Both are planned for milestone M2 ([09 D12](09_STAGE2_3_PLAN.md#44-models)).
   - Neural nets: a siamese head that evaluates g on both x and m(x).
   - Calibrate with temperature scaling, fitted by inner cross-fitting ([09 D19](09_STAGE2_3_PLAN.md#46-probability-output)): it has no bias term, so the antisymmetry is kept. Platt scaling only for unconstrained models.
3. **Sanity checks.**
   - Mirroring twice gives the identity. Unit-test this in TypeScript and Python.
   - On synthetic data, mirrored right-hand trials match left-hand trials.
   - After normalisation, the median first-half yaw (§4.3) of right-hand trials has the same sign on iOS and Android.
   - For unconstrained models, |p(x) + p(m(x)) − 1| stays small. Large values mean the model relies on cues that are not mirror-symmetric, such as dominance or platform artefacts.
   - Trials where an antisymmetric model gives the recorded label less than 5% probability are reviewed for wrong-hand errors. About 16% of participants wore a sensor on the wrong wrist on at least one day of a supervised wrist study ([PLOE 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538999/)).
4. **64-sign-pattern pilot check.** Run it once on the pilot, separately for iOS and Android ([rationale](https://arxiv.org/html/2605.04791)).
   - **Procedure.** For each of the 2⁶ sign patterns σ over (a_x, a_y, a_z, ω_x, ω_y, ω_z), apply σ to every right-hand trial and map its orientation through `ORIENTATIONS[o].mirror`. Then compute D(σ) = Σ‖x̄_L − σ ⊙ x̄_R‖² / Σ‖x̄_L‖² over participants, orientations and channels. Here x̄ are per-participant mean waveforms, time-normalised to 101 points between onset and end.
   - **Pass criteria.** σ* = (−, +, +; +, −, −) ranks first on both platforms. Also report the six single-channel flip margins D(σ* with one channel flipped) − D(σ*). A margin near zero means that channel carries little signal.
   - **Reading the result.** In field names, σ* is `acc_x` −, `rr_alpha` +, `rr_beta` −, `rr_gamma` −. If the winner is `rr_alpha` −, `rr_beta` +, `rr_gamma` −, the data follow the legacy alpha = z mapping.
   - **What it cannot do.** The check cannot see a global accelerometer sign, because both hands are recorded on the same phone. It does not replace the gravity-sign resolution of §2.1.

### 3.6 Screen-away synthesis (protocol 2.0.0)

Protocol 2.0.0 never asks for a screen-away grip (`*_away`): participants could not do them on the street. Because the PI's test conditions are unknown, the analysis synthesises them instead, by turning the phone 180° about its own y axis: F = diag(−1, 1, −1), and `GRIP_MATRICES[o_away] = GRIP_MATRICES[o_toward] · F`.

- **Rule.** F is a proper rotation, so f, a and ω all map (x, y, z) ↦ (−x, y, −z) (W3C mapping: `acc_x`, `acc_z`, `accg_x`, `accg_z`, `rr_alpha`, `rr_gamma` negated); the orientation R ↦ R · F, i.e. quaternion (x, y, z, w) ↦ (−z, w, x, −y); the label `*_toward` ↦ `*_away`; touches cleared (the nose would meet the back of the phone); detected rotations and gravity votes unchanged; trial id + `-away`. Flipping twice is the identity, and the flip commutes with the mirror (§3.2).
- **Code.** `conventions.flip_device_vectors` (the rule), `augment.screen_away_trial` (raw records), `features.prepare_trial(..., screen_flip=True)` (normalised signals), `evaluate.cross_validate(augment_screen_flip=True)` and `train_baseline.py --augment-screen-away` (training folds only; test trials are always real).
- **What it changes.** The basic, invariant and chiral features of a flipped trial equal the original's, so `yaw_rule` and logistic regression on chiral features gain nothing; only models on device-frame signals (gradient boosting, sequence models) see new data.
- **Limits.** It is not a real screen-away movement: wrist rotation, fingers, approach and contact differ, and the accelerometer's lever arm is not flipped (about 0.5 m/s² at natural speed, as for the mirror, §3.4). Use it to augment training data only, and never report screen-away accuracy from synthetic trials.

## 4. Features

### 4.1 Parity rule

Under a reflection P (det P = −1), a polar vector u (f, a, v or ĝ) maps to P·u and a pseudovector (ω) maps to −P·ω. Every cross product or determinant adds another factor det P. **A scalar flips sign under mirroring exactly when the number of pseudovectors plus the number of cross products or determinants is odd**; such a scalar is a pseudoscalar. Proper rotations (any grip, any heading) leave all these scalars unchanged, provided the vectors are expressed in one frame when they are combined: either at the same instant, or transported into D₀ with R₀(t).

### 4.2 Families

| Family | Examples | Grip-invariant | Heading-invariant | Under the mirror | Role |
|---|---|---|---|---|---|
| Basic (timing, magnitude, smoothness) | duration T; time to peak ‖ω‖ divided by T; peak ‖ω‖ and ‖a‖; total rotation ‖log R₀(t_end)‖; movement units (‖ω‖ peaks with prominence ≥ 10–20 deg/s, at least 150 ms apart, [Thrane 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7296942/)); SPARC on ‖ω‖; LDLJ-A ([Melendez-Calderon 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7841375/)) | yes | yes | unchanged | context, speed bins, odd-movement flags, handedness add-on |
| Device-frame | mean, extremes and signed integral of each normalised channel; ĝ at onset and at end, in device coordinates | no | yes | a_x, ω_y, ω_z flip; a_y, a_z, ω_x keep; orientation relabelled | strong within known grips but leaks the grip; test with leave-one-orientation-out |
| Invariant, even | a·ĝ (vertical), ‖a × ĝ‖ (horizontal), ‖ω × ĝ‖, \|ω·ĝ\| | yes | yes | unchanged | context for the odd features |
| Chiral pseudoscalars | §4.3 | yes | yes | flip sign | core left/right evidence |

### 4.3 Chiral pseudoscalars

| Feature | Formula, onset → end | Pseudovectors + ×/det | Notes |
|---|---|---|---|
| Net yaw about gravity | Ψ = ∫ ω·ĝ dt; normalised Ψ_n = Ψ / ∫ ‖ω‖ dt ∈ [−1, 1] | 1 + 0 | Gyro-based once ĝ is known. It stays usable in slow trials, where the gyro still peaks at 40 deg/s or more. FNT studies report rotation "counterclockwise for the right hand, clockwise for the left" ([Mohammadi-Ghazi 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9609238/)), but about a wrist-sensor axis not stated to be vertical, so this is suggestive, not proof. A call-handedness patent uses a swivel sign in the same way ([US10375228B2](https://patents.google.com/patent/US10375228B2/en)). |
| Helicity | H = ∫ ω·a dt / ∫ ‖ω‖‖a‖ dt | 1 + 0 | Only samples with ‖a‖ ≥ 0.5 m/s² contribute. Weak in slow trials, where a is about 0.3–0.7 m/s², close to Chrome's 0.1 m/s² step. |
| Acceleration turning | K = Σ det[a(t), a(t+τ), ĝ(t)] / Σ ‖a(t)‖‖a(t+τ)‖, with vectors in D₀ and τ ≈ 50–100 ms | 0 + 1 | The sense in which the acceleration vector turns, seen from above. Needs no integration. |
| Path curvature | C = ∫ (v × a)·ĝ dt / ∫ ‖v‖‖a‖ dt, in D₀ | 0 + 1 | Needs v from ZUPT, so only for T ≤ 1.5 s. |

Time-resolved versions are also pseudoscalars, e.g. ω·ĝ averaged in 10 time-normalised bins.

**Why these features are invariant.** A constant change of grip multiplies every device-frame vector by the same rotation Q, with det Q = +1, which preserves dot products and determinants. Where the person faces never enters, because ĝ is the only external reference. Re-gripping the phone during the movement, as the `twist` variant asks, breaks the constant-Q assumption, so `twist` doubles as a robustness test.

**Hyp.** The hand shows in the time course of ω·ĝ: in the first half of the reach the yaw is counter-clockwise seen from above for the right hand, then the wrist turns the phone back, so the net Ψ largely cancels (pilot, 16 trials, one person: first-half yaw AUC 0.95, net yaw 0.58; [09 §2.1](09_STAGE2_3_PLAN.md#21-the-hand-is-in-the-yaw-time-course-not-in-the-net-yaw)). Rules learn their sign from data. The features are `chi_yaw_first_half_deg`, `chi_yaw_second_half_deg`, `chi_yaw_rate_b1` … `b5` (5 time bins), `chi_yaw_peak_dps` and `chi_yaw_net_deg` (`motionlab.features`).

### 4.4 Do not use

- **Raw `deviceorientation` alpha, beta, gamma, or their derivatives.** They hit gimbal lock at beta = ±90°, which is exactly the upright start pose ([W3C](https://w3c.github.io/deviceorientation/)).
- **Absolute heading** (`compass`, absolute alpha). It cannot be mirrored and it leaks room and identity.
- **`interval_raw`, or the browser's `acc` as the primary linear acceleration** (see §2.1).
- **Smoothness metrics computed from IMU-integrated velocity.** They correlated with motion capture at ρ ≈ −0.1 ([Melendez-Calderon 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7841375/)).

## 5. Baselines ladder

Move up a rung only when the rung below is working and evaluated.

| # | Model | Input | Mirror handling | Purpose | Stage-3 export |
|---|---|---|---|---|---|
| 0 | Chance; label-permutation control | none | none | detect leakage: must give ≈ 50% | none |
| 1 | `YawSignRule`: predict right iff s·x > θ; abstain if \|x\| < δ; the sign s is learnt from training data | x = `chi_yaw_first_half_deg` (`YawSignRule(feature=...)`; net Ψ and the signed peak as ablation A14) | antisymmetric when θ = 0 | physics floor, interpretable | 3 numbers in JSON |
| 2 | Logistic regression (L2), antisymmetric: **the default candidate** | chiral features, optionally odd × even interactions (e.g. Ψ_n · log T) | by construction | main interpretable model | JSON weights |
| 2b | Logistic regression, unconstrained | 2, plus even and device-frame features | augmentation | how much non-mirror information adds | JSON weights |
| 3 | `HistGradientBoostingClassifier`: unconstrained and augmented, vs `MirrorSymmetrized(HGB)` | basic + invariant + chiral, optionally device-frame | augmentation, or g(x) − g(m(x)) by the wrapper | non-linear tabular model | JSON trees |
| 3b | MiniRocket on the 8 invariant channels + ridge, symmetrised | sequences | symmetrised | research only ([09 D12](09_STAGE2_3_PLAN.md#44-models)) | none |
| 4 | 1D-CNN or GRU (≲ 50k parameters) | 60 Hz sequences. Variant "dev": 9 device-frame channels (a, ω, ĝ). Variant "inv": ω·ĝ, ‖ω × ĝ‖, a·ĝ, ‖a × ĝ‖, ω·â | augmentation + siamese antisymmetric head | only worth keeping if it beats rung 3 by more than the CI | ONNX or a hand-written TS forward pass |

**Augmentation for rung 4.** Um et al. compared augmentations on wrist-accelerometer data: a CNN baseline reached 77.54%, rotation 82.62%, time-warping 82.00%, permutation 81.16%, scaling 79.46%, jittering 77.52% and cropping 73.58% ([Um 2017](https://ar5iv.arxiv.org/html/1706.00527)). Their task tolerates operations that ours does not:

| Augmentation | Use? | How / why |
|---|---|---|
| Mirror + label flip | yes | §3.5. |
| Small rotations | yes | Apply one random rotation of at most 15–20° to all device-frame vectors of a trial (a, f, ω, ĝ). It mimics grip variation. |
| Time-warp | yes | Smooth monotone warp with local speed × 0.8–1.25. It mimics speed variation. |
| Magnitude scaling | yes | Scale a and ω by one factor per vector (0.9–1.1). Never scale per axis, which bends directions, and never scale f, which changes ‖g‖. |
| Screen-away flip, Q = diag(−1, 1, −1) | yes, for device-frame and sequence models | Acc and gyro (−x, y, −z), label `*_toward` → `*_away` (§3.6). Q is a proper rotation, so the chiral features are invariant (a unit test checks it) and antisymmetric chiral models gain nothing. Training folds only (`augment_screen_flip`, `--augment-screen-away`). |
| Jitter | optional | No gain in Um et al. |
| Permutation, cropping | no | They destroy the order start → reach → nose → hold. |
| Unrestricted 3-D rotation | no | Proper rotations keep chirality, but unrestricted ones create grips nobody uses and erase the relation between pose and gravity. The collected grips, the screen-away flip and small rotations cover the use case. |

**Training rules (Rec.).**
- Select models by log-loss.
- Use early stopping on held-out inner participants.
- Classes are balanced automatically, because every person does both hands.
- Calibrate inside each outer training fold ([09 D19](09_STAGE2_3_PLAN.md#46-probability-output)): compute inner cross-fitted logits (GroupKFold by person), fit a temperature T (antisymmetric models) or Platt's (a, b) (unconstrained models) on them, then refit the base model on the whole fold. The outer test fold is never seen; the shipped model goes through the same steps on all development people. Isotonic regression only as ablation A11.
- Log every random seed.

## 6. Evaluation protocol

The PI will test new people, so every split is subject-wise. Record-wise CV understates error. A review of 62 sensor and smartphone papers found a median reported error of 5.6% with record-wise CV and 13.0% with subject-wise CV, across the 47 papers whose accuracy could be extracted ([Saeb 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/)). The same arm movement is used as a personal biometric ([Conti 2011](https://iris.unitn.it/handle/11572/89854)), so record-wise splits leak identity.

### 6.1 Splits

| Split | scikit-learn tool | Question answered |
|---|---|---|
| Development CV | `StratifiedGroupKFold(n_splits=5, shuffle=True)`, grouped by person: `person_key`, the earliest P-code of a person after merging aliases ([09 D2](09_STAGE2_3_PLAN.md#41-data); `participant_id` until milestone M1 adds it). Stratified on dominant hand × platform (`handedness_self × platform`, `evaluate.DEFAULT_STRATA`; the street profile has no EHI, so `ehi_class × platform` only for protocol-1.0.0 data). Repeated with 5 seeds. | Expected accuracy on new people. |
| LOSO | `LeaveOneGroupOut()`, splitting with `groups=participant_id` | Per-participant accuracy distribution; handedness add-on. |
| Nested tuning | `GridSearchCV(..., cv=GroupKFold(n_splits=4))` inside each outer training fold, `fit(X, y, groups=...)` | Unbiased hyperparameters. |
| Leave-one-orientation-out | Outer subject-wise folds × one held-out orientation. Hold out mirror pairs (`top_left_*` with `top_right_*`) together, because mirroring one creates the other. | A new grip. |
| Leave-one-platform-out | `LeaveOneGroupOut()`, splitting with `groups=device.platform` | iOS ↔ Android transfer. |
| Leave-one-device-model-out | `LeaveOneGroupOut()`, splitting with `groups=device.model_hint` on Android. On iOS `model_hint` is only the device class (`iPhone` / `iPad`) or a hardware id in in-app browsers, and protocol 2.0.0 no longer asks the phone model, so the proxy is the screen, `<family> <short>x<long>@<dpr>` (e.g. `iPhone 390x844@3`, `motionlab.io.device_model`), which groups iPhones of one size class | A new phone model. |
| Leave-one-speed-out | Outer person-wise folds × one held-out speed (planned: the `speed` scheme of [09 D16](09_STAGE2_3_PLAN.md#45-evaluation)) | The PI's speed is unknown. |
| Leave-one-variant-out | The same for the movement variant (the `variant` scheme) | The PI may move in a way nobody trained. |
| Learning curve | Train on 5, 10, 15, … people and test on the rest (`evaluate.learning_curve`, planned) | A plateau means more people only narrow the CI; a rising curve means: collect more. |
| Locked test | About 25% of people (§6.4) | The PI's scenario, run once. |

The implemented schemes today are `stratified_group_kfold`, `group_kfold`, `loso`, `orientation` and `platform` (`motionlab.evaluate`). Nested cross-validation, as it must run ([09 §4.5](09_STAGE2_3_PLAN.md#45-evaluation)):

```python
# sketch; names marked + are new
for outer_train, outer_test in make_splits(scheme, person_key, ...):          # people never shared
    fold = cap_per_person(outer_train, 40)                                        # D2
    fold = drop_wrong_hand_flags(fold, inner='loso')                              # D6, training rows only
    est = Calibrated(MirrorAugmented(base) if not base.antisymmetric else base,  # D13, D19 (+)
                     method='temperature' if base.antisymmetric else 'platt',
                     inner_cv=GroupKFold(4))         # inner cross-fitted logits -> T, then refit on the fold
    search = GridSearchCV(est, grid, cv=GroupKFold(4), scoring='neg_log_loss')
    search.fit(X[fold], y[fold], groups=person_key[fold])   # inner validation folds hold real trials only
    p_right[outer_test] = search.predict_proba(X[outer_test])[:, 1]               # real trials only
# tau, OOD ranges and the reliability table come from these OOF predictions (D20); the locked test is untouched
```

### 6.2 Leakage rules

1. The group is the participant in every split, including the orientation, platform and device splits.
2. Augmentation happens inside the estimator's `fit`, so inner and outer validation folds hold real trials only.
3. Every fitted step (scalers, feature selection, calibration, abstention threshold) is fitted on training participants only.
4. Nothing is tuned on the locked test set.
5. Only real trials are evaluated.

### 6.3 Metrics and uncertainty

- **CIs.** Resample people, not trials, with B ≥ 2,000 (10⁴ when cheap). The implemented bootstrap (`motionlab.evaluate`) resamples people with replacement, one level. The hierarchical version also resamples trials within each person; on nested data it kept type-I error near 5%, where naive tests gave 46–96% false positives ([Saravanan 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290)).
- **Differences** between models or ablations use a paired person bootstrap: both arms on the same resampled people ([09 D17](09_STAGE2_3_PLAN.md#45-evaluation)).
- **Never report the fold SE alone.** It strongly underestimates uncertainty; with about 100 test samples the error bars are around ±10% ([Varoquaux 2018](https://arxiv.org/abs/1706.07581)).
- **Per person.** Report the median, the 10th percentile and the number of people below 80%.
- **Also report.** Calibration: the reliability diagram and the ECE with 10 equal-mass bins; and a coverage–accuracy curve for abstention.

### 6.4 Locked test set (**Rec.**)

- **A rule fixed in advance, about 25 %, by person** ([09 D3](09_STAGE2_3_PLAN.md#41-data)): a person is in the locked test iff `int(sha256(("nosetouch-locked-v1:" + person_key).encode()).hexdigest(), 16) % 4 == 0`. There is no salt: the app draws the P-code at random, so nobody chooses the assignment, and the commit date of 09 shows that the rule predates the data. A salt held by the PI would add nothing, since the student must know the assignment to keep those people out of development.
- **Always excluded from the test set,** whatever the hash says: the student (every session) and the pilot participant.
- **Access.** Development code loads only development people (`load_dataset(..., partition='dev')`, milestone M1). The locked-test run is a separate script that writes `locked_test_opened_at` into its manifest and refuses a second run for the same model sha256.
- **No redraw.** If the test set comes out small or unbalanced, the report states its composition and CI width as they are.
- **Run the test set once**, with the final pipeline.

### 6.5 Checks before trusting any number

1. Shuffling hand labels within participants gives ≈ 50%.
2. Report record-wise and subject-wise accuracy once each. A large gap means identity leakage ([Chaibub Neto 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789029/)), though under-fitting can also cause one ([Little 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441396/)).
3. **Platform probe.** Train a classifier to predict `device.platform` from the features. It should drop to near chance once iOS data are quantised.
4. If accuracy on canonical trials is well below about 90%, check axes, signs, units and labels before blaming the task (digest inference from [Chen et al. 2023](https://scholars.cityu.edu.hk/en/publications/deep-learning-based-unobtrusive-handedness-prediction-for-one-han/)).
5. **Sign-source agreement** ([09 D7](09_STAGE2_3_PLAN.md#43-preprocessing-and-segmentation)): per session, does the implicit calibration's sign agree with the trial votes and with the engine prior? Predict mode has only a check and the engine, so this table estimates how often that pair could be wrong.
6. **Retry and abstention rates by hand** ([09 D5](09_STAGE2_3_PLAN.md#42-labels-and-exclusions)) should be similar; a gate that fires more for one hand biases the score.

## 7. Error analysis by subgroup

### 7.1 Model

The GLMM is fitted on out-of-fold predictions from the development CV ([lme4](https://www.jstatsoft.org/article/view/v067i01)):

```r
library(lme4)
m <- glmer(correct ~ hand * handedness + sex + age_c + speed + orientation + platform + (1 | participant),
           data = oof, family = binomial)
```

| Term | Coding | Field |
|---|---|---|
| `correct` | 1 if the out-of-fold prediction (threshold 0.5) equals the label | development CV |
| `hand` | left / right | `condition.hand` |
| `handedness` | right / both / left: the self-reported dominant hand, because street sessions have no Edinburgh short form. For protocol-1.0.0 data the EHI-SF classes (61..100, −60..60, −100..−61, [Veale 2014](https://www.millisecond.com/library/v7/edinburghhandednessinventory/edinburghhandednessinventory_sf.manual)) and the continuous LQ are available as a sensitivity analysis. | `handedness_self`; `ehi_class`, `ehi_lq` (1.0.0 only) |
| `hand:handedness` | **Key term**: dominant vs non-dominant hand | |
| `sex` | female / male; `other` and `prefer_not_to_say` pooled or dropped | `sex` |
| `age_c` | age in decades, centred | `age_years` |
| `speed` | Instructed speed. A second model uses tertiles of measured duration, because instructions and achieved speed differ. | `condition.speed`, T |
| `orientation` | The collected grips: 3 in street data (`upright_toward`, `top_left_toward`, `top_right_toward`), up to 10 levels with researcher presets and protocol-1.0.0 data. When errors are sparse, collapse to rotation plus screen. | `condition.orientation` |
| `platform` | ios / android | `device.platform` |
| Optional | `variant`, `start_position`, `(1 \| device_model)` | |

In Python, `statsmodels` `BinomialBayesMixedGLM` is an alternative, but lme4 is the reference implementation.

### 7.2 Power

- **Fixed-effect budget.** The number of errors limits the model, at about 10 errors per fixed-effect degree of freedom ([Peduzzi 1996](https://doi.org/10.1016/S0895-4356(96)00236-3), a heuristic [criticised as too rigid](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-016-0267-3)). 2,400 trials at 5% error give 120 errors, about 12 degrees of freedom. The full formula above uses about 20, so collapse factor levels or use a penalised or Bayesian GLMM.
- **People, not trials.** Subgroup conclusions depend on the number of people per subgroup. With the 28 instructed trials of tiers 1–3 of a street visit, the design effect 1 + 27·ICC is 3.7 at ICC 0.1 and 6.4 at ICC 0.2. Detecting 5% vs 15% error (about 141 effective trials per group) then needs about 19 or 32 people per group (our computation; with protocol 1.0.0's 40 trials per person it was about 18 or 31).
- **Simulate power.** Once the first 10 people give the ICC, simulate power with [simr](https://doi.org/10.1111/2041-210X.12504).

### 7.3 What to report

Every number, table and data figure in the report and the deck is generated by `report/scripts/make_assets.py` from `analysis/reports/<run>/`; nothing is copied by hand ([09 D31](09_STAGE2_3_PLAN.md#6-report-and-deck)). The registry of figures and tables, with their inputs, is in [report/README.md](../report/README.md); the IDs below map onto it.

| ID | Report ID | Table or figure |
|---|---|---|
| T1 | T05 | Participants by handedness × sex × platform; trials per condition. |
| T2 | T09 | Per model rung: accuracy, balanced accuracy, AUC and log-loss with 95% CIs; median and worst-decile participant accuracy. |
| T3 | T13 | GLMM odds ratios with 95% CIs. |
| F1 | F13 | Per-participant accuracy, sorted, coloured by handedness, marker by platform. |
| F2 | F18 | Heatmap of accuracy by orientation × speed, with n per cell. |
| F3 | F19 | Accuracy vs measured duration, split by hand × handedness. |
| F4 | F20 | Forest plot of the GLMM effects. |
| F5 | F16 | Reliability diagrams, overall and per platform. |
| F6 | F09 | Distributions of the first-half yaw (and Ψ) by hand × handedness × platform. This shows the mirror symmetry and the sign check. |
| F7 | F14 | Accuracy per held-out grip, platform, speed, odd variant and start position. |
| F8 | F17 | Coverage vs accuracy under abstention, τ marked. |

## 8. Confounds and ablations

### 8.1 Confounds

| Confound | Mechanism | Design mitigation (Stage 1) | Analysis check |
|---|---|---|---|
| **Dominance** | For about 90% of people, the right hand is the dominant hand ([Papadatou-Pastou 2020](https://research-repository.st-andrews.ac.uk/handle/10023/19889): 10.6% left-handed). Dominant reaches are straighter; the non-dominant arm shows more curvature and larger initial-direction errors, and left-handers show smaller asymmetry ([Przybyla 2012](https://pure.psu.edu/en/publications/dynamic-dominance-varies-with-handedness-reduced-interlimb-asymme/)). A model can learn "dominant style" instead of "side". | Both hands for everyone, balanced at every tier boundary; left-handed friends recruited on purpose (target 10 or more, [09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection)). | `hand:handedness` in the GLMM. Train on right-handers, test on left-handers. **Dominance probe**: a model on mirror-even features only. Its signature is above-chance accuracy in right-handers and **below-chance** accuracy in left-handers. Antisymmetric vs unconstrained model (A1). |
| **Platform** | Chrome rounds to 0.1 units and iOS does not ([source](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)); iOS signs are inverted; rates and jitter differ. Both hands use one phone, so the platform cannot predict the label directly, but the model can learn platform-specific rules. | The engine's sign confirmed by the trial votes (§2.1); engine detection. | Quantise iOS (A3); leave-one-platform-out; platform probe (§6.5). |
| **Device model** | IMU noise, bias, range (±4 g on some phones, so fast trials can clip), rate, size and weight. Devices measurably differ ([Stisen 2015](https://archive.ics.uci.edu/dataset/344/hhar)). | `model_hint` (and `phone_model` in protocol-1.0.0 data), screen size and achieved rates are recorded. | Leave-one-device-model-out; `(1 \| device_model)`; clipping flags. |
| **Session order and fatigue** | Time on task correlates with block. Street tiers run in a fixed order: upright speeds, then sideways, then varied movements, then the bonus. | Hands balanced at every tier boundary; the first hand and the first sideways side from the seed; ABBA in the researcher presets. | `trial_index` as a covariate; compare hands within a tier. |
| **Start pose** | The start position sets the lateral component. A `center` start removes most lateral translation. | `start_position` factor (`natural`/`shoulder`/`center`), plus `bent_start`. | Accuracy per start position; gyro-only model. |
| **Label noise** | People use the wrong hand; hands-free acceptance means nobody confirms each trial. | The hand named at every block intro, on its Start button and on the live panel; the grip gate; "Last one was the wrong hand" on the pause sheet. | `participant_flag`; wrong-hand screening by antisymmetric-model outliers inside training folds (§3.5, [09 D6](09_STAGE2_3_PLAN.md#42-labels-and-exclusions)); sensitivity analysis with and without it. |
| **Segmentation bias** | Thresholds bias onset by a fixed fraction of the duration: 5% of peak shifts onset by 18 ms at 0.3 s but by 119 ms at 2 s (digest). | Raw windows are kept; same algorithm for both hands. | online_v1 vs offline consensus; disagreement rate by hand. |
| **Posture, phone case, `arm_issue`** | Grip and movement differ. | Posture recorded per session ("I’m sitting"); a case and `arm_issue` only in protocol-1.0.0 data (2.0.0 writes null and false: the consent checkbox excludes arm or hand injuries). | Stratify by posture; exclude `arm_issue` from the primary analysis. |

### 8.2 Ablations

| ID | Comparison | Decision it informs |
|---|---|---|
| A1 | With vs without mirror augmentation, and antisymmetric vs unconstrained. Evaluate on real trials, overall and for left-handers. | Whether to ship an antisymmetric model. |
| A2 | 60 Hz vs high rate. **Out of scope unless a phyphox cohort is run.** In such a cohort (10–20 people, same protocol, both OSes), compare native rate, PWA emulation (60 Hz sample-and-hold plus 0.1 rounding) and real PWA data. phyphox is free and open source and reaches about 100 Hz on iPhone and up to 500 Hz on some Android phones ([phyphox forum](https://phyphox.org/forums/printthread.php?tid=2356)). Record raw samples with `rate="0"` or `rateStrategy="request"` ([format](https://phyphox.org/docs/file-format/input/)). | Build a native app only if the gap in accuracy, or in onset/offset error, exceeds the CI width. |
| A3 | iOS data quantised vs raw. | The normalisation default. |
| A4 | Feature sets: chiral only; plus invariant; plus device-frame; sequences; plus touch. Evaluated leave-one-orientation-out. | Robustness to new grips. |
| A5 | Segmentation and features: research (offline_v1, 100 Hz, filtered, ZUPT) vs deploy (`feat_d1`) vs online_v1 with the stored parameters. | The cost of deployability. |
| A6 | Gyro only, accelerometer only, both. | Robustness to sign and quantisation problems. |
| A7 | Approach only vs approach plus the hold at the nose. | Value of the end pose. |
| A8 | Train on right-handers only, test on left-handers. | Size of the dominance shortcut. |
| A9 | Screen-away flip augmentation (device-frame and sequence models), checked on the rehearsal stress set. | Robustness to screen-away grips. |
| A10 | Approach only vs approach plus the time-reversed return (or a separate return head). | Whether the return of the arm adds evidence. |
| A11 | Calibration: none, temperature, Platt, (symmetric) isotonic. | The calibration method. |
| A12 | OOD rule: per-feature ranges vs a Ledoit–Wolf Mahalanobis distance on mirror-even features (risk–coverage curve). | The abstention rule. |
| A13 | A heading-canonical body frame vs the gravity frame. | Kept as a check only; the design draft found no gain on the pilot. |
| A14 | Rung-1 feature: first-half yaw vs net yaw vs the signed peak yaw rate. | Confirms the pilot's yaw finding (§4.3) on more people. |

## 9. Handedness add-on (exploratory)

- **Target and unit of analysis.** The target is one label per person: the self-reported dominant hand `handedness_self` (left vs right; "both" analysed separately), since street sessions have no Edinburgh short form. For protocol-1.0.0 data `ehi_class` and the continuous `ehi_lq` are alternatives. The unit of analysis is the participant.
- **Features.** Within-person asymmetry between the two hands, AI = (R − L)/(|R| + |L|), on per-hand medians of: path or rotation-profile straightness, initial-direction proxies, SPARC on ‖ω‖, movement units, rep-to-rep variability, and steadiness during the hold at the nose. Dynamic-dominance theory predicts curvature and initial-direction differences, not end-point accuracy ([Bagesteiro & Sainburg 2002](https://pure.psu.edu/en/publications/handedness-dominant-arm-advantages-in-control-of-limb-dynamics/); [Shabbott & Sainburg 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2525729)). Speed is a poor cue: in one small study the non-dominant hand was even faster ([Akbas 2026](https://pure.psu.edu/en/publications/the-influence-of-hand-dominance-object-distance-and-size-on-reach/)).
- **Behaviour priors.** Use `phone_hand` from the profile, and the hand each person chose in the free-choice trial of protocol 2.0.0 (tier 0, `hand_source = reported`). Protocol 1.0.0's `free` blocks freed the grip, not the hand.
- **Model and evaluation.** Use logistic regression with at most 5 features, because n is small. Evaluate LOSO, computing features from both hands of the held-out person. Report AUC with a CI. Compare three variants: motion only, priors only, and combined.
- **Sample size.** An AUC of 0.75 has a 95% CI of about ±0.16 with 15 left-handers vs 45 right-handers, and about ±0.12 with 30 vs 30 ([Hanley & McNeil](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747), digest computation). Present results as exploratory.

## 10. Prior work and expected accuracy

No phone-to-nose benchmark exists. The closest published results, all from small samples or evaluations that are not subject-wise:

| Study | Signal | N | Result | Caveat |
|---|---|---|---|---|
| [GripSense 2012](https://ubicomplab.cs.washington.edu/publications/gripsense/) | gyroscope + touch + vibration | 10 | 84.26% hand posture | controlled study |
| [Löchtefeld 2015](https://vbn.aau.dk/en/publications/detecting-users-handedness-for-ergonomic-adaptation-of-mobile-use-2) | sensors during the unlock gesture, k-NN | 12 | 98.51% | all right-handed |
| [Guo 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017479/) | touch traces; acc + gyro for hand switches | 14 | about 94% precision; 93.9% for hand switches | random 10-fold CV |
| [Nelavelli & Ploetz 2018](https://ar5iv.labs.arxiv.org/html/1805.08367) | sign of the swipe-arc curvature | 12 (2 left-handed) | 99.53% consistent | no held-out users; one mirror-antisymmetric feature |
| [Fernandez 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8093950/) | touch features | 174 (35 left-handed) | up to 99.92% TPR after 7 interactions | oversampling before random splits |
| [Chen, Zhu & Yang 2023](https://scholars.cityu.edu.hk/en/publications/deep-learning-based-unobtrusive-handedness-prediction-for-one-han/) | motion sensors, LSTM | not stated | 97.4% sitting, 94.6% standing, 89.2% in real life | unclear whether test users were unseen |
| [Singh & Singh 2025](https://conf.researchr.org/details/isec-2025/isec-2025-papers/11/Dominant-Hand-detection-using-Base-Smartphone-sensors) | acc + gyro + orientation, XGBoost, 3 classes | not reported | 85.04% | protocol not reported |
| [Mohammadi-Ghazi 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9609238/) | wrist IMU, finger-to-nose test | 17 | opposite rotation sense for the two hands | clinical study, no classifier |

Expectations. These are hypotheses, not results:

- **Hyp.** Canonical trials (`standard` variant, natural or fast speed): subject-wise accuracy of 90% or more. Reasoning: everyday phone use already gives about 85–95%, and a deliberate, large reach should be easier.
- **Hyp.** Lower for `slow` trials (accelerometer features near the 0.1 m/s² floor), for the `center` start, and for `twist`, `curved` and `jerky`.
- With 30–60 participants, CIs will be several points wide.
- The handedness add-on will be weak from kinematics alone ([Xiao 2019](https://aimspress.com/aimspress-data/mbe/2019/3/PDF/mbe-16-03-077.pdf)).

## 11. Stage-2 timeline

The milestones are those of [09 §8](09_STAGE2_3_PLAN.md#8-milestones-and-acceptance-criteria); each starts when its gate is met. There is no ethics-approval gate: the researcher decided on 2026-10-24 that the volunteers, adult friends who consent in the app, need no ethics review ([06 §1](06_CONSIDERATIONS.md#1-ethics-and-law)).

| Milestone | Work | Accept when |
|---|---|---|
| M0 Plan frozen (now) | 09 committed before the first frozen pull; it fixes the locked-test rule and the pre-registered choices (primary metric, default model family, gates, τ target). | Run manifests record that commit. |
| M1 Data plumbing (before 10 people are analysed) | Freeze step and `snapshot.json`; `person_key` and the alias table; the `partition='dev'` loader and the locked-test script; `io` exposes `hand_source`, block `kind`, `tier`, `review_mode`, `calibration.method`; the modelling table drops reported trials; strata `dominant_hand × platform`; QA shows the sign-source agreement and the axis check. | The pilot gives the same features as before; tests prove that a choice trial never enters training and that a test person never reaches development code. |
| M2 Baselines on the first 10 development people (S2-M1, S2-M2) | The 64-sign-pattern check per platform; rungs 0–2 on the deploy pipeline (`feat_d1`, `offline_v1`, 60 Hz); `feature_mirror`, `MirrorSymmetrized` and the augmentation wrappers with tests; A14 and the first ICC. | Shuffled labels give 45–55 %; σ* = (−, +, +; +, −, −) ranks first; antisymmetric models deviate from the mirror rule by at most 1e-9; the record- vs subject-wise gap is reported. |
| M3 Predictor v0 (in parallel with collection) | [08](08_STAGE3_PREDICTOR.md). | Parity tests pass; zero requests after load. |
| M4 Model selection and freeze (30 or more development people) (S2-M3) | Calibration (D19) and abstention (D20); rungs 3–5 where allowed; ablations A1–A14 and the learning curve; selection; export of the model JSON. | Development-OOF ECE at most 0.05 (**Rec.**); no reliability bin off by more than its CI; the model sha256 recorded. |
| M5 Locked test (40 or more people) (S2-M4) | One run, in its own folder; the expectation for the PI committed. | The manifest carries `locked_test_opened_at`; the numbers are reported whether or not they fall inside the development CI. |
| M6, M7 | Report and deck; the PI session. | [09 §8](09_STAGE2_3_PLAN.md#8-milestones-and-acceptance-criteria). |

Recruitment targets ([09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection)): at least 30 people with tiers 0–2 complete (target 40 or more), iPhone users at least 25 % (target 30 % or more), at least 8 left-handers (target 10 or more), 2 or more Android brands (target 3), one session per person, and 3–5 rehearsal people for predict mode who never enter training. Development can start before then on `npm run synth -- --preset street` data and the pilot.

## 12. Open questions for the PI

1. Is the use case "unseen person, single trial, unknown phone and grip" right? Is a per-user calibration allowed, i.e. a few labelled trials before testing?
2. What counts as success: a minimum accuracy, the primary metric, and whether the predictor may abstain?
3. Should odd variants, including the tilted touch, be part of the primary evaluation, or a separate robustness set? Should `chin` and `forehead` count as nose touches?
4. Are 30–40 or more friends, including 8–10 or more left-handers, enough ([09 §7](09_STAGE2_3_PLAN.md#7-what-we-need-from-data-collection))?
5. Should we pre-register on OSF, or is the dated commit of 09 enough? (The locked-test rule needs no salt, §6.4.)
6. Is R (lme4) acceptable for the GLMM?
7. Is a lab-run phyphox validation cohort wanted (ablation A2)? Without one, A2 is out of scope.
8. ~~For the handedness add-on, should a free-choice-hand trial be added?~~ Answered: protocol 2.0.0 has one choice trial in tier 0 (`hand_source = reported`).
9. May the dataset and the model be shared or published later? This affects consent wording.
