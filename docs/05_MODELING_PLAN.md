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
| Label | `condition.hand` ∈ {`left`, `right`}, the instructed hand. It is never inferred from handedness. The side people use the phone on matches their dominant hand only about 70% of the time ([Seidman 2013](https://sciencedaily.com/releases/2013/05/130516161655.htm), survey, low confidence). |
| Model input | Sensor streams, calibration and segmentation only. The model never sees `condition` (hand, orientation, speed, variant). |
| Use case | An unseen person performs a single trial on an unknown phone (iOS or Android, any model), with an unknown grip, any speed, and possibly an odd movement. |
| Primary metric | Trial-level accuracy on unseen participants, with a participant-level bootstrap 95% CI. |
| Secondary metrics | Balanced accuracy, ROC AUC, log-loss, calibration, coverage vs accuracy under abstention, and the per-participant accuracy distribution. |

Which data each analysis uses (**Rec.**):

| Set | Filter |
|---|---|
| Primary analysis | `simulated` false; `outcome.status` `complete`; `outcome.accepted` true; `outcome.participant_flag` null; `quality.motion_hz` ≥ 50; `quality.gaps_over_50ms` = 0; `quality.hidden_during` false; `quality.rotation_available` true; calibration present. |
| Sensitivity | The primary set plus complete attempts that were not accepted, plus flagged trials. |
| Pipeline tests only | `npm run synth` output (`simulated: true`). It is never mixed into real-data results. |

Browsers deliver about 60 Hz on both platforms. Chromium uses a 60 Hz pump ([source](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_sensor_event_pump.h)) and WebKit a 1/60 s timer ([source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.h)). A rate below 50 Hz or a gap over 50 ms therefore means a broken recording.

## 2. Preprocessing pipeline

Raw records are never modified. Each step is versioned (e.g. `norm_v1`), and the version string is stored with every derived table and model. Stage 3 can then refuse a model built with a different pipeline.

| # | Step | Output | Why |
|---|---|---|---|
| 0 | Load records, validate them against the schema, apply the §1 filters | trial table | Contract drift fails loudly. |
| 1 | Normalise conventions (§2.1) | f in m/s², ω in deg/s, W3C signs | Platforms differ, so conventions are measured per device. |
| 2 | Resample to a uniform grid: 60 Hz for PWA data, 100 Hz for the phyphox cohort (§2.2) | uniform arrays | Browser samples are jittered snapshots. |
| 3 | Estimate gravity by gyro propagation from the rest windows (§2.3) | ĝ(t), a(t) = f − g | Accelerometer-corrected filters fail during the movement. |
| 4 | Segmentation: online_v1 indices plus an offline consensus (§2.4) | onset, end, flags | The offline consensus is the analysis truth. |
| 5 | Features (§4) and/or sequences | X | |

### 2.1 Normalisation (`norm_v1`)

| Quantity | Raw fields | Normalised | Rule |
|---|---|---|---|
| Specific force | `accg_x/y/z` | f = s · accg | s = `calibration.gravity_sign` (+1 = W3C, −1 = inverted), unless the trial votes clearly contradict it (≥ 5 votes, ≥ 80 % against): then the vote sign is used and the session flagged. If the calibrated sign is null, use a confident vote of the trial baselines, then the engine (`device.engine` `webkit` → −1, otherwise +1), and flag the session ([03 §5.2](03_SENSORS_AND_PLATFORMS.md#52-gravity-sign-resolution)). |
| Browser linear acceleration | `acc_x/y/z` | s · acc | QA only. When a phone has no linear-acceleration sensor, Chrome falls back to a crude high-pass filter ([source](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/linear_acceleration_fusion_algorithm_using_accelerometer.cc)). |
| Angular velocity | `rr_alpha/beta/gamma` (deg/s) | ω = (x, y, z) in deg/s | Depends on `calibration.axis_check.verdict`: `w3c` → (alpha, beta, gamma); `legacy` → (beta, gamma, alpha); `*_inverted` → negate. If `units` is `rad_per_s`, multiply by 180/π. If the verdict is `unclear` or absent, assume `w3c` and flag the session (`axis_unverified`). |
| Gyro bias | `calibration.gyro_bias_dps` | subtract, with the same mapping | Taken from the flat-on-table pose, which is true rest, but only when that window was still (`calibration.flat.still`) and the bias norm is below 3 deg/s. Otherwise the mean rate of the 400 ms before GO is used if its norm is below 3 deg/s. |
| Time | `t` (handler time) | seconds | `t` is on the same clock as the cues. `ts` is a QA cross-check. Never use `interval_raw`: iOS reports it in seconds, Chrome reports a constant 16 and Firefox a constant 100 ([WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/DeviceMotionClientIOS.mm), [Chromium source](https://github.com/chromium/chromium/blob/main/third_party/blink/renderer/modules/device_orientation/device_motion_event_pump.cc)). |
| Repeated samples | | drop | On iOS, drop exact 9-channel repeats: they are duplicated timer snapshots, and iOS values are unrounded. On Chrome, drop a repeat only while ‖ω‖ > 20 deg/s. **Rec.** |
| iOS precision | WebKit values are unrounded | round raw `acc`/`accg` to 0.1 m/s² and `rr` to 0.1 deg/s | **Rec.** Evaluated as ablation A3 (`train_baseline.py --quantize ios`); off by default in norm_v1 until A3 decides. Chrome rounds to this grid ([platform_sensor_util.h](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)). Without the rounding, a model can learn the platform from value granularity. |

Why the sign step exists: WebKit forwards Core Motion's sign, so a face-up iPhone reads `accg_z` ≈ −9.81, while Android reads +9.81. rotationRate signs agree across the two platforms. Sources: [WebKit source](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/platform/ios/WebCoreMotionManager.mm), [W3C list, 2014](https://lists.w3.org/Archives/Public/public-geolocation/2014Jul/0024.html), [re-measured on iPadOS in 2026](https://github.com/soroushdty/gaitscope/issues/96). The sign is measured from the flat pose and never inferred from the user agent. §3.3 explains why it cannot be recovered any other way.

### 2.2 Resampling (**Rec.**)

- **PWA data.** Linearly interpolate onto a 60 Hz grid after removing repeats. At 60 Hz, movements of 0.25 s or longer keep at least 99% of their acceleration energy, because 99% of it lies below about 2.53/D Hz ([minimum-jerk profile](https://www.socsci.ru.nl/wilberth/physics/smooth.html); spectrum from the digest's calculation).
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
- The deployed model must be trained on segments the browser can reproduce exactly ([08 §2.4](08_STAGE3_PREDICTOR.md#24-training-and-deployment-parity)).
- Windows: integrals run over [onset, end]; sequences run over [onset − 0.5 s, end + 0.5 s].

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

Published transforms disagree. OREBA flips acc x and gyro y, z ([paper](https://ar5iv.labs.arxiv.org/html/2007.15831)), as does Kyritsis et al. ([paper](https://arxiv.org/pdf/2010.07051)). OpenWatch (2026) flips acc X, gyro X and gyro Z, which violates the pseudovector rule ([paper](https://arxiv.org/html/2605.04791)). **Do not copy a transform from a paper; verify it (§3.5, item 4).**

### 3.3 What the iOS sign inversion means: parity

Raw iOS data are (−a, ω) relative to Android: the accelerometer is negated and the gyro is not. That is exactly the parity transformation P = −I applied to polar vectors, with pseudovectors unchanged, and −I = S · Rx(180°) has det = −1. Two consequences:

1. Every chiral feature (§4.3) flips sign on un-normalised iOS data, so **an iOS right-hand trial looks like an Android left-hand trial**. No grip change can undo this, because grips are proper rotations.
2. The flip cannot be detected from gyro–accelerometer consistency, because dĝ/dt = −ω × ĝ also holds for −ĝ. The axis check in [calibration.ts](../app/src/core/calibration.ts) is independent of the accelerometer sign for exactly this reason. Only a known pose reveals the sign, here calibration step (a), the phone flat with the screen up. Gravity-sign normalisation is therefore a prerequisite for every model.

### 3.4 Limits of the mirror model

- **Bodies and movements are not symmetric.** Dominant and non-dominant arms move differently ([Sainburg 2002](https://link.springer.com/doi/10.1007/s00221-001-0913-8)). Mirrored right-hand trials are therefore not real left-hand trials, and augmentation must always be evaluated on real data.
- **The gyroscope mirror is exact; the accelerometer mirror is approximate.** ω is the same at every point of a rigid phone, but the IMU sits off the phone's centre line, at x offset r_x. The real reading minus the augmented one is α′ × d + ω′ × (ω′ × d), with d = (2r_x, 0, 0). r_x is unknown per model and is less than half the phone's width. For r_x = 2 cm, my minimum-jerk estimate of this error is about 0.5 m/s² at natural speed (peak a ≈ 5 m/s²) and about 3 m/s² at fast speed (peak a ≈ 29 m/s²), roughly 10% in both cases.
- **Phones are not symmetric.** The camera bump and buttons can change how people grip the phone.

### 3.5 Uses

1. **Augmentation.** For every training trial, add its mirror: streams mirrored per §3.2, hand swapped, orientation mapped. The copies stay in the source participant's fold. Evaluate on real trials only, and report the ablation (§8.2, A1).
2. **Antisymmetric models.** Define the logit as f(x) = g(x) − g(m(x)), where m is the mirror map and g is any model. This gives three properties:
   - p(R | m(x)) = 1 − p(R | x) exactly.
   - A mirror-symmetric input gets p = 0.5, which is a natural abstention.
   - Mirror-even cues such as speed, smoothness and tremor cannot decide on their own. The dominance shortcut (§8.1) is therefore removed by construction, at the price of ignoring real but confounded asymmetries.

   Implementations:
   - Logistic regression on odd (chiral) features with `fit_intercept=False` and `StandardScaler(with_mean=False)`.
   - GBM or neural net: evaluate g on both x and m(x) (siamese).
   - Calibrate with temperature scaling, which has no bias term, so the antisymmetry is kept.
3. **Sanity checks.**
   - Mirroring twice gives the identity. Unit-test this in TypeScript and Python.
   - On synthetic data, mirrored right-hand trials match left-hand trials.
   - After normalisation, the median net yaw Ψ (§4.3) of right-hand trials has the same sign on iOS and Android.
   - For unconstrained models, |p(x) + p(m(x)) − 1| stays small. Large values mean the model relies on cues that are not mirror-symmetric, such as dominance or platform artefacts.
   - Trials where an antisymmetric model gives the recorded label less than 5% probability are reviewed for wrong-hand errors. About 16% of participants wore a sensor on the wrong wrist on at least one day of a supervised wrist study ([PLOE 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6538999/)).
4. **64-sign-pattern pilot check.** Run it once on the pilot, separately for iOS and Android ([rationale](https://arxiv.org/html/2605.04791)).
   - **Procedure.** For each of the 2⁶ sign patterns σ over (a_x, a_y, a_z, ω_x, ω_y, ω_z), apply σ to every right-hand trial and map its orientation through `ORIENTATIONS[o].mirror`. Then compute D(σ) = Σ‖x̄_L − σ ⊙ x̄_R‖² / Σ‖x̄_L‖² over participants, orientations and channels. Here x̄ are per-participant mean waveforms, time-normalised to 101 points between onset and end.
   - **Pass criteria.** σ* = (−, +, +; +, −, −) ranks first on both platforms. Also report the six single-channel flip margins D(σ* with one channel flipped) − D(σ*). A margin near zero means that channel carries little signal.
   - **Reading the result.** In field names, σ* is `acc_x` −, `rr_alpha` +, `rr_beta` −, `rr_gamma` −. If the winner is `rr_alpha` −, `rr_beta` +, `rr_gamma` −, the data follow the legacy alpha = z mapping.
   - **What it cannot do.** The check cannot see a global accelerometer sign, because both hands are recorded on the same phone. It does not replace the flat-pose calibration.

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

**Hyp.** Ψ > 0 for the right hand, i.e. counter-clockwise seen from above with ĝ pointing up. The simulator encodes the same convention (`R_motion = Ry(h·yaw)`). Axis and viewpoint conventions differ between papers, so the YawSignRule learns the sign from training data instead of hard-coding it.

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
| 1 | `YawSignRule`: predict right iff s·Ψ > θ; abstain if \|Ψ\| < δ | Ψ | antisymmetric when θ = 0 | physics floor, interpretable | 3 numbers in JSON |
| 2 | Logistic regression (L2), antisymmetric | chiral features, optionally odd × even interactions (e.g. Ψ_n · log T) | by construction | main interpretable model | JSON weights |
| 2b | Logistic regression, unconstrained | 2, plus even and device-frame features | augmentation | how much non-mirror information adds | JSON weights |
| 3 | `HistGradientBoostingClassifier` | basic + invariant + chiral, optionally device-frame | augmentation; optional g(x) − g(m(x)) | non-linear tabular model | JSON trees or ONNX |
| 4 | 1D-CNN or GRU (≲ 50k parameters) | 60 Hz sequences. Variant "dev": 9 device-frame channels (a, ω, ĝ). Variant "inv": ω·ĝ, ‖ω × ĝ‖, a·ĝ, ‖a × ĝ‖, ω·â | augmentation + siamese antisymmetric head | only worth keeping if it beats rung 3 by more than the CI | ONNX or a hand-written TS forward pass |

**Augmentation for rung 4.** Um et al. compared augmentations on wrist-accelerometer data: a CNN baseline reached 77.54%, rotation 82.62%, time-warping 82.00%, permutation 81.16%, scaling 79.46%, jittering 77.52% and cropping 73.58% ([Um 2017](https://ar5iv.arxiv.org/html/1706.00527)). Their task tolerates operations that ours does not:

| Augmentation | Use? | How / why |
|---|---|---|
| Mirror + label flip | yes | §3.5. |
| Small rotations | yes | Apply one random rotation of at most 15–20° to all device-frame vectors of a trial (a, f, ω, ĝ). It mimics grip variation. |
| Time-warp | yes | Smooth monotone warp with local speed × 0.8–1.25. It mimics speed variation. |
| Magnitude scaling | yes | Scale a and ω by one factor per vector (0.9–1.1). Never scale per axis, which bends directions, and never scale f, which changes ‖g‖. |
| Jitter | optional | No gain in Um et al. |
| Permutation, cropping | no | They destroy the order start → reach → nose → hold. |
| Unrestricted 3-D rotation | no | Proper rotations keep chirality, but unrestricted ones create grips nobody uses and erase the relation between pose and gravity. The 10 real orientations plus small rotations cover the use case. |

**Training rules (Rec.).**
- Select models by log-loss.
- Use early stopping on held-out inner participants.
- Classes are balanced automatically, because every person does both hands.
- Calibrate inside the training fold: temperature or Platt scaling via `CalibratedClassifierCV(cv=<group splits>)`.
- Log every random seed.

## 6. Evaluation protocol

The PI will test new people, so every split is subject-wise. Record-wise CV understates error. A review of 62 sensor and smartphone papers found a median reported error of 5.6% with record-wise CV and 13.0% with subject-wise CV, across the 47 papers whose accuracy could be extracted ([Saeb 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/)). The same arm movement is used as a personal biometric ([Conti 2011](https://iris.unitn.it/handle/11572/89854)), so record-wise splits leak identity.

### 6.1 Splits

| Split | scikit-learn tool | Question answered |
|---|---|---|
| Development CV | `StratifiedGroupKFold(n_splits=5, shuffle=True)`, `groups=participant_id`, stratified on the participant stratum `ehi_class × platform`, repeated with 5 seeds | Expected accuracy on new people. |
| LOSO | `LeaveOneGroupOut()`, splitting with `groups=participant_id` | Per-participant accuracy distribution; handedness add-on. |
| Nested tuning | `GridSearchCV(..., cv=GroupKFold(n_splits=4))` inside each outer training fold, `fit(X, y, groups=...)` | Unbiased hyperparameters. |
| Leave-one-orientation-out | Outer subject-wise folds × one held-out orientation. Hold out mirror pairs (`top_left_*` with `top_right_*`) together, because mirroring one creates the other. | A new grip. |
| Leave-one-platform-out | `LeaveOneGroupOut()`, splitting with `groups=device.platform` | iOS ↔ Android transfer. |
| Leave-one-device-model-out | `LeaveOneGroupOut()`, splitting with `groups=device.model_hint` on Android and the participant's `phone_model` on iOS, where `model_hint` is only the device class (`iPhone` / `iPad`) or a hardware id in in-app browsers | A new phone model. |
| Locked test | About 20% of participants (§6.4) | The PI's scenario, run once. |

```python
# sketch: helper names (add_mirrored, stratum, pid) are illustrative
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
for tr, te in outer.split(X, y=stratum, groups=pid):        # y = stratum: used only to stratify
    Xa, ya, ga = add_mirrored(X[tr], hand[tr], pid[tr])      # copies stay in the training fold
    search = GridSearchCV(model, grid, cv=GroupKFold(n_splits=4), scoring="neg_log_loss")
    search.fit(Xa, ya, groups=ga)
    p_right[te] = search.predict_proba(X[te])[:, 1]          # real trials only
```

### 6.2 Leakage rules

1. The group is the participant in every split, including the orientation, platform and device splits.
2. Augmentation happens inside the training fold. Mirrored copies never appear in a test fold.
3. Every fitted step (scalers, feature selection, calibration, abstention threshold) is fitted on training participants only.
4. Nothing is tuned on the locked test set.
5. Only real trials are evaluated.

### 6.3 Metrics and uncertainty

- **CIs.** Use a hierarchical bootstrap: resample participants with replacement, then trials within each participant, with B ≥ 2,000 (10⁴ when cheap). On nested data it keeps type-I error near 5%, where naive tests gave 46–96% false positives ([Saravanan 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290)).
- **Never report the fold SE alone.** It strongly underestimates uncertainty; with about 100 test samples the error bars are around ±10% ([Varoquaux 2018](https://arxiv.org/abs/1706.07581)).
- **Per participant.** Report the median, the 10th percentile and the number of participants below 80%.
- **Also report.** Calibration (reliability diagram), and a coverage–accuracy curve for abstention.

### 6.4 Locked test set (**Rec.**)

- **Assign at enrolment, with a rule fixed in advance.** A participant goes to the test set iff `int(sha256(salt + participant_id), 16) % 5 == 0`. The PI holds the salt.
- **Exclude pilot participants**, since their data shape development.
- **Check the strata.** At the end, check that the test set has at least 5 left-handers and both platforms.
- **Pre-register on OSF** before opening the test set: hypotheses, primary metric, subgroups and analysis plan.
- **Run the test set once**, with the final pipeline.

### 6.5 Checks before trusting any number

1. Shuffling hand labels within participants gives ≈ 50%.
2. Report record-wise and subject-wise accuracy once each. A large gap means identity leakage ([Chaibub Neto 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6789029/)), though under-fitting can also cause one ([Little 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5441396/)).
3. **Platform probe.** Train a classifier to predict `device.platform` from the features. It should drop to near chance once iOS data are quantised.
4. If accuracy on canonical trials is well below about 90%, check axes, signs, units and labels before blaming the task (digest inference from [Chen et al. 2023](https://scholars.cityu.edu.hk/en/publications/deep-learning-based-unobtrusive-handedness-prediction-for-one-han/)).

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
| `handedness` | right / mixed / left: EHI-SF classes 61..100, −60..60, −100..−61 ([Veale 2014](https://www.millisecond.com/library/v7/edinburghhandednessinventory/edinburghhandednessinventory_sf.manual)). Sensitivity analysis with the continuous LQ. | `ehi_class`, `ehi_lq` |
| `hand:handedness` | **Key term**: dominant vs non-dominant hand | |
| `sex` | female / male; `other` and `prefer_not_to_say` pooled or dropped | `sex` |
| `age_c` | age in decades, centred | `age_years` |
| `speed` | Instructed speed. A second model uses tertiles of measured duration, because instructions and achieved speed differ. | `condition.speed`, T |
| `orientation` | 10 levels. When errors are sparse, collapse to rotation (4 levels) plus screen (2 levels). | `condition.orientation` |
| `platform` | ios / android | `device.platform` |
| Optional | `variant`, `start_position`, `(1 \| device_model)` | |

In Python, `statsmodels` `BinomialBayesMixedGLM` is an alternative, but lme4 is the reference implementation.

### 7.2 Power

- **Fixed-effect budget.** The number of errors limits the model, at about 10 errors per fixed-effect degree of freedom ([Peduzzi 1996](https://doi.org/10.1016/S0895-4356(96)00236-3), a heuristic [criticised as too rigid](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-016-0267-3)). 2,400 trials at 5% error give 120 errors, about 12 degrees of freedom. The full formula above uses about 20, so collapse factor levels or use a penalised or Bayesian GLMM.
- **People, not trials.** Subgroup conclusions depend on the number of people per subgroup. With 40 trials per person, the design effect is 4.9 at ICC 0.1 and 8.8 at ICC 0.2. Detecting 5% vs 15% error then needs about 18 or 31 people per group (digest computation).
- **Simulate power.** Once the pilot gives the ICC, simulate power with [simr](https://doi.org/10.1111/2041-210X.12504).

### 7.3 What to report

| ID | Table or figure |
|---|---|
| T1 | Participants by handedness × sex × platform; trials per condition. |
| T2 | Per model rung: accuracy, balanced accuracy, AUC and log-loss with 95% CIs; median and worst-decile participant accuracy. |
| T3 | GLMM odds ratios with 95% CIs. |
| F1 | Per-participant accuracy, sorted, coloured by handedness, marker by platform. |
| F2 | Heatmap of accuracy by orientation × speed, with n per cell. |
| F3 | Accuracy vs measured duration, split by hand × handedness. |
| F4 | Forest plot of the GLMM effects. |
| F5 | Reliability diagrams, overall and per platform. |
| F6 | Distributions of Ψ by hand × handedness × platform. This shows the mirror symmetry and the sign check. |
| F7 | Accuracy per odd variant and per start position. |
| F8 | Coverage vs accuracy under abstention. |

## 8. Confounds and ablations

### 8.1 Confounds

| Confound | Mechanism | Design mitigation (Stage 1) | Analysis check |
|---|---|---|---|
| **Dominance** | For about 90% of people, the right hand is the dominant hand ([Papadatou-Pastou 2020](https://research-repository.st-andrews.ac.uk/handle/10023/19889): 10.6% left-handed). Dominant reaches are straighter; the non-dominant arm shows more curvature and larger initial-direction errors, and left-handers show smaller asymmetry ([Przybyla 2012](https://pure.psu.edu/en/publications/dynamic-dominance-varies-with-handedness-reduced-interlimb-asymme/)). A model can learn "dominant style" instead of "side". | Both hands for everyone; deliberate recruitment of 20–30 or more left-handers; ABBA hand order. | `hand:handedness` in the GLMM. Train on right-handers, test on left-handers. **Dominance probe**: a model on mirror-even features only. Its signature is above-chance accuracy in right-handers and **below-chance** accuracy in left-handers. Antisymmetric vs unconstrained model (A1). |
| **Platform** | Chrome rounds to 0.1 units and iOS does not ([source](https://github.com/chromium/chromium/blob/main/services/device/generic_sensor/platform_sensor_util.h)); iOS signs are inverted; rates and jitter differ. Both hands use one phone, so the platform cannot predict the label directly, but the model can learn platform-specific rules. | Calibration measures the sign; engine detection. | Quantise iOS (A3); leave-one-platform-out; platform probe (§6.5). |
| **Device model** | IMU noise, bias, range (±4 g on some phones, so fast trials can clip), rate, size and weight. Devices measurably differ ([Stisen 2015](https://archive.ics.uci.edu/dataset/344/hhar)). | `model_hint`, `phone_model` and achieved rates are recorded. | Leave-one-device-model-out; `(1 \| device_model)`; clipping flags. |
| **Session order and fatigue** | Time on task correlates with block; the first block is always `upright_toward`. | ABBA order with seed-parity counterbalancing. | `trial_index` as a covariate; first vs second half of the session. |
| **Start pose** | The start position sets the lateral component. A `center` start removes most lateral translation. | `start_position` factor (`natural`/`shoulder`/`center`), plus `bent_start`. | Accuracy per start position; gyro-only model. |
| **Label noise** | People use the wrong hand. | Colour-coded hand cue per block; the flag button. | `participant_flag`; antisymmetric-model outliers (§3.5); sensitivity analysis without flagged trials. |
| **Segmentation bias** | Thresholds bias onset by a fixed fraction of the duration: 5% of peak shifts onset by 18 ms at 0.3 s but by 119 ms at 2 s (digest). | Raw windows are kept; same algorithm for both hands. | online_v1 vs offline consensus; disagreement rate by hand. |
| **Posture, phone case, `arm_issue`** | Grip and movement differ. | Recorded in the profile and protocol. | Stratify; exclude `arm_issue` from the primary analysis. |

### 8.2 Ablations

| ID | Comparison | Decision it informs |
|---|---|---|
| A1 | With vs without mirror augmentation, and antisymmetric vs unconstrained. Evaluate on real trials, overall and for left-handers. | Whether to ship an antisymmetric model. |
| A2 | 60 Hz vs high rate. In the phyphox cohort (10–20 people, same protocol, both OSes), compare native rate, PWA emulation (60 Hz sample-and-hold plus 0.1 rounding) and real PWA data. phyphox is free and open source and reaches about 100 Hz on iPhone and up to 500 Hz on some Android phones ([phyphox forum](https://phyphox.org/forums/printthread.php?tid=2356)). Record raw samples with `rate="0"` or `rateStrategy="request"` ([format](https://phyphox.org/docs/file-format/input/)). | Build a native app only if the gap in accuracy, or in onset/offset error, exceeds the CI width. |
| A3 | iOS data quantised vs raw. | The default for `norm_v1`. |
| A4 | Feature sets: chiral only; plus invariant; plus device-frame; sequences. Evaluated leave-one-orientation-out. | Robustness to new grips. |
| A5 | Segmentation source: online_v1, offline consensus, or none (a fixed window after GO). | How much Stage 3 depends on segmentation. |
| A6 | Gyro only, accelerometer only, both. | Robustness to sign and quantisation problems. |
| A7 | Movement only vs movement plus holds. | Value of the end pose. |
| A8 | Train on right-handers only, test on left-handers. | Size of the dominance shortcut. |

## 9. Handedness add-on (exploratory)

- **Target and unit of analysis.** The target is one label per person: `ehi_class` (or left vs right, excluding mixed), with `ehi_lq` as a continuous alternative. The unit of analysis is the participant.
- **Features.** Within-person asymmetry between the two hands, AI = (R − L)/(|R| + |L|), on per-hand medians of: path or rotation-profile straightness, initial-direction proxies, SPARC on ‖ω‖, movement units, rep-to-rep variability, and steadiness during the hold at the nose. Dynamic-dominance theory predicts curvature and initial-direction differences, not end-point accuracy ([Bagesteiro & Sainburg 2002](https://pure.psu.edu/en/publications/handedness-dominant-arm-advantages-in-control-of-limb-dynamics/); [Shabbott & Sainburg 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2525729)). Speed is a poor cue: in one small study the non-dominant hand was even faster ([Akbas 2026](https://pure.psu.edu/en/publications/the-influence-of-hand-dominance-object-distance-and-size-on-reach/)).
- **Behaviour priors.** Use `phone_hand` from the profile. A free-choice-hand trial would add another prior, but it is **not** in PROTOCOL_VERSION 1.0.0: the `free` block frees the grip, not the hand. See the open questions.
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

The timeline is given in weeks, relative to approval of the ethics request. Each phase starts only when its gate is met.

| Weeks | Work | Output | Gate |
|---|---|---|---|
| Before approval | `motionlab` end-to-end on `npm run synth` data: io → normalisation → gravity → features → `YawSignRule` and logistic regression. Unit tests: mirror involution, f′ = S f and ω′ = −S ω on random rigid-body trajectories, online_v1 parity with the TS spec, gravity closure. Fix the locked-test rule and draft the OSF pre-registration. | A pipeline that passes known-answer tests. | none |
| 1–2 | Pilot with 8–10 people on both platforms. | `qa_report.py` output, the 64-pattern check, the sign check, ICC, first baselines, a power simulation. | Ethics approval. Protocol go/no-go. |
| 3–6 | Main collection: 40–60 or more participants, at least 20–30 of them left-handers, each platform at least 30%. In parallel, a phyphox cohort of 10–20. | Weekly QA. Baselines on development participants only. | Recruitment targets met. |
| 7 | Freeze the data; final pre-registration. | Frozen dataset hash. | |
| 7–9 | GBM, sequence models, ablations, GLMM, handedness add-on. | Report draft. | |
| 10 | One run on the locked test set. Export the chosen model. | Final report; model JSON for [Stage 3](08_STAGE3_PREDICTOR.md). | |

## 12. Open questions for the PI

1. Is the use case "unseen person, single trial, unknown phone and grip" right? Is a per-user calibration allowed, i.e. a few labelled trials before testing?
2. What counts as success: a minimum accuracy, the primary metric, and whether the predictor may abstain?
3. Should odd variants be part of the primary evaluation, or a separate robustness set? Should `chin` and `forehead` count as nose touches?
4. Are 40–60 participants, including 20–30 left-handers, feasible through the lab's participant pool?
5. Should we pre-register on OSF? Will you hold the salt for the locked test set?
6. Is R (lme4) acceptable for the GLMM?
7. Is a lab-run phyphox validation cohort wanted (ablation A2)?
8. For the handedness add-on, should a free-choice-hand trial be added? It is a protocol and schema change.
9. May the dataset and the model be shared or published later? This affects consent wording.
